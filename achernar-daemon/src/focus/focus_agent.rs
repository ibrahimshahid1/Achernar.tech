//! Focus agent — Section 7.3.4.
//! FocusRunState machine and run_autofocus() orchestration.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::future::Future;

use super::classical_focus_solver::ClassicalFocusSolver;

/// The order a sweep visits its positions in.
///
/// This is a *data collection* control, not a focusing one: both patterns visit
/// the same set of positions and fit the same curve, so they converge equally
/// well. What differs is what the run teaches the Focus Offset Predictor.
///
/// Nine of the FOP's 38 features — the History and Direction groups, listed in
/// `ml/contracts/fop_features.py` — describe what the focuser has recently done:
/// `prev_delta_1/2/3`, `dir_changed_recently`, `steps_since_flip_norm`,
/// `dir_encoded`. A sweep that marches from one end to the other never reverses,
/// so every one of those features holds the same value on every row of every
/// run. The direction head is not merely trained poorly by such a dataset; it
/// has nothing to learn from at all, because its target never varies.
///
/// The observing plan therefore calls for roughly one run in three to be a
/// reversal. That is a per-night decision, so the pattern is overridable on the
/// request rather than fixed in the config file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SweepPattern {
    /// Visit positions in ascending order, low to high.
    ///
    /// The default, and the right choice for ordinary focusing: approaching
    /// every position from the same direction means backlash enters each
    /// measurement identically instead of appearing as a step in the curve.
    #[default]
    Monotonic,

    /// Start in the middle, run outward to the high end, then reverse and cover
    /// the low end descending.
    ///
    /// Produces exactly one direction change, at a known point, and both signs
    /// of travel. Deliberately takes backlash on the chin — the reverse leg's
    /// positions carry the focuser's mechanical slop — because a model that has
    /// never seen a reversal cannot compensate for one. Expect a slightly worse
    /// curve fit on these runs; that is the trade being made.
    Reversal,
}

impl SweepPattern {
    /// Parse the value accepted by `POST /focus/run`.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "monotonic" | "sorted" => Some(Self::Monotonic),
            "reversal" | "reverse" => Some(Self::Reversal),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Monotonic => "monotonic",
            Self::Reversal => "reversal",
        }
    }
}

/// Which HFR the autofocus curve fit uses: `[focus] hfr_source`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HfrSource {
    /// The classical curve-of-growth HFR, the authority (default).
    #[default]
    Classical,
    /// PROTOTYPE (owner authorization F-006; public-archive weights, not
    /// qualified for R50/Z114). Each sweep sample's fit HFR is the FQE frame
    /// HFR when the prototype produced a finite prediction from at least
    /// [`FQE_PROTOTYPE_MIN_STARS`] stars, otherwise that sample's classical
    /// HFR. Requires `[vision.fqe] prototype = true` (config validation).
    /// `FocusSample::hfr`, verification and the drift forecaster stay
    /// classical.
    FqePrototype,
}

impl HfrSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Classical => "classical",
            Self::FqePrototype => "fqe_prototype",
        }
    }
}

/// Fewest FQE crops behind a frame prediction before the prototype may stand
/// in for that sample's classical HFR.
pub const FQE_PROTOTYPE_MIN_STARS: u32 = 3;

/// Which measurement a sample's curve-fit HFR is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitHfrSource {
    Classical,
    FqePrototype,
}

/// The curve-fit HFR of one sweep sample on a run whose `hfr_source` is not
/// classical, and where it came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitHfr {
    /// The value the curve fit used (px).
    pub hfr: f64,
    pub source: FitHfrSource,
    /// Why the classical HFR stood in, when the run asked for the prototype.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
    /// The frame's FQE reading that was consulted (`None` when absent).
    #[serde(default)]
    pub fqe_hfr_px: Option<f64>,
    #[serde(default)]
    pub fqe_star_count: Option<u32>,
    #[serde(default)]
    pub fqe_model_version: Option<String>,
}

/// A frame's FQE reading, as the vision loop published it.
#[derive(Debug, Clone, Copy, Default)]
pub struct FqeReading<'a> {
    pub hfr_px: Option<f64>,
    pub star_count: Option<u32>,
    /// The reading came from the verified prototype path (`fqe_prototype`).
    pub prototype: bool,
    pub model_version: Option<&'a str>,
}

/// The fit HFR for one sweep sample. `None` for [`HfrSource::Classical`]
/// (the fit uses `FocusSample::hfr`, as it always has).
pub fn choose_fit_hfr(
    source: HfrSource,
    classical_hfr: f64,
    fqe: &FqeReading<'_>,
) -> Option<FitHfr> {
    if source == HfrSource::Classical {
        return None;
    }
    let fallback = match (fqe.hfr_px, fqe.star_count) {
        (None, _) => Some("no_fqe_prediction"),
        (Some(_), _) if !fqe.prototype => Some("fqe_not_prototype"),
        (Some(hfr), _) if !(hfr.is_finite() && hfr > 0.0) => Some("fqe_not_finite"),
        (Some(_), stars) if stars.unwrap_or(0) < FQE_PROTOTYPE_MIN_STARS => {
            Some("fqe_too_few_stars")
        }
        _ => None,
    };
    Some(FitHfr {
        hfr: match fallback {
            None => fqe.hfr_px.unwrap_or(classical_hfr),
            Some(_) => classical_hfr,
        },
        source: match fallback {
            None => FitHfrSource::FqePrototype,
            Some(_) => FitHfrSource::Classical,
        },
        fallback_reason: fallback.map(str::to_string),
        fqe_hfr_px: fqe.hfr_px,
        fqe_star_count: fqe.star_count,
        fqe_model_version: fqe.model_version.map(str::to_string),
    })
}

/// How many of a non-classical run's samples the curve fit took from each
/// source.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct HfrSourceCounts {
    pub requested: HfrSource,
    pub fqe_prototype: u32,
    pub classical: u32,
    /// Classical stand-ins by reason (`no_fqe_prediction`,
    /// `fqe_too_few_stars`, ...).
    pub fallback_reasons: BTreeMap<String, u32>,
}

/// Counts over `samples`, or `None` on a classical run.
pub fn hfr_source_counts(source: HfrSource, samples: &[FocusSample]) -> Option<HfrSourceCounts> {
    if source == HfrSource::Classical {
        return None;
    }
    let mut counts = HfrSourceCounts {
        requested: source,
        ..Default::default()
    };
    for fit in samples.iter().filter_map(|sample| sample.fit_hfr.as_ref()) {
        match fit.source {
            FitHfrSource::FqePrototype => counts.fqe_prototype += 1,
            FitHfrSource::Classical => {
                counts.classical += 1;
                if let Some(reason) = &fit.fallback_reason {
                    *counts.fallback_reasons.entry(reason.clone()).or_default() += 1;
                }
            }
        }
    }
    Some(counts)
}

/// State machine for an autofocus run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FocusRunState {
    Idle,
    Initializing,
    Sweeping,
    Measuring,
    Evaluating,
    Moving,
    Settling,
    Converged,
    Failed,
    Aborted,
}

/// Metadata for a focus run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusRunContext {
    pub run_id: uuid::Uuid,
    pub rig_id: String,
    pub filter_name: Option<String>,
    pub temperature: Option<f64>,
    pub start_position: i32,
    pub state: FocusRunState,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub samples: Vec<FocusSample>,
    pub best_position: Option<i32>,
    /// The fitted curve's HFR at `best_position`, in the units of the fit's
    /// input: classical HFR, or on an `fqe_prototype` run the per-sample mix
    /// counted in `hfr_source_counts`.
    pub best_hfr: Option<f64>,
    /// Curve-fit HFR sources, on a run whose `[focus] hfr_source` is not
    /// classical. `None` (omitted) on classical runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hfr_source_counts: Option<HfrSourceCounts>,
}

/// A single focus measurement sample.
///
/// Each sample corresponds to exactly one captured frame. `frame_id` links back
/// to the preserved pixels in the `frames` table so the FQE dataset builder can
/// recover the (crop, HFR) training pairs. Repeated frames at a position are
/// kept as distinct samples — never averaged — so per-position scatter survives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusSample {
    pub position: i32,
    pub hfr: f64,
    /// Half-flux diameter (`2 * HFR`), in pixels.
    #[serde(default)]
    pub hfd: Option<f64>,
    /// Fitted PSF full width at half maximum, in pixels. This remains `None`
    /// unless a Gaussian/Moffat fit was actually performed.
    #[serde(default)]
    pub fwhm: Option<f64>,
    pub star_count: u32,
    pub quality_score: Option<f64>,
    pub temperature: Option<f64>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Id of the preserved frame (`frames.id`) this sample was measured from.
    #[serde(default)]
    pub frame_id: Option<String>,
    /// What the curve fit used for this sample on a run whose `hfr_source`
    /// is not classical. `None` (omitted): the fit used `hfr`. `hfr` is the
    /// classical measurement either way.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit_hfr: Option<FitHfr>,
}

impl FocusSample {
    /// The HFR the curve fit consumes: `fit_hfr` when the run chose one,
    /// otherwise the classical `hfr`.
    pub fn curve_fit_hfr(&self) -> f64 {
        self.fit_hfr.as_ref().map_or(self.hfr, |fit| fit.hfr)
    }
}

/// The focus agent orchestrates an autofocus run.
pub struct FocusAgent {
    state: FocusRunState,
    current_run: Option<FocusRunContext>,
    history: Vec<FocusRunContext>,
}

impl FocusAgent {
    pub fn new() -> Self {
        Self {
            state: FocusRunState::Idle,
            current_run: None,
            history: Vec::new(),
        }
    }

    pub fn state(&self) -> FocusRunState {
        self.state
    }

    pub fn current_run(&self) -> Option<&FocusRunContext> {
        self.current_run.as_ref()
    }

    pub fn history(&self) -> &[FocusRunContext] {
        &self.history
    }

    /// Start a new autofocus run with a fresh run id.
    #[cfg(test)]
    pub async fn run_autofocus<F, Fut>(
        &mut self,
        config: &crate::config::FocusConfig,
        rig_id: String,
        filter_name: Option<String>,
        start_position: i32,
        measure_at: F,
    ) -> crate::error::Result<FocusRunContext>
    where
        F: FnMut(i32) -> Fut,
        Fut: Future<Output = crate::error::Result<Vec<FocusSample>>>,
    {
        self.run_autofocus_with_id(
            uuid::Uuid::new_v4(),
            config,
            rig_id,
            filter_name,
            start_position,
            measure_at,
        )
        .await
    }

    /// Start a new autofocus run under `run_id`, the id the caller already
    /// published for it (the autofocus operation's `autofocus_started`, its
    /// gate op and every later record share the `focus_runs` id).
    pub async fn run_autofocus_with_id<F, Fut>(
        &mut self,
        run_id: uuid::Uuid,
        config: &crate::config::FocusConfig,
        rig_id: String,
        filter_name: Option<String>,
        start_position: i32,
        mut measure_at: F,
    ) -> crate::error::Result<FocusRunContext>
    where
        F: FnMut(i32) -> Fut,
        Fut: Future<Output = crate::error::Result<Vec<FocusSample>>>,
    {
        if matches!(
            self.state,
            FocusRunState::Initializing
                | FocusRunState::Sweeping
                | FocusRunState::Measuring
                | FocusRunState::Moving
                | FocusRunState::Settling
                | FocusRunState::Evaluating
        ) {
            return Err(crate::error::FocusError::RunFailed(
                "Autofocus is already running".to_string(),
            )
            .into());
        }

        let mut run = FocusRunContext {
            run_id,
            rig_id,
            filter_name,
            temperature: None,
            start_position,
            state: FocusRunState::Initializing,
            started_at: chrono::Utc::now(),
            samples: Vec::new(),
            best_position: None,
            best_hfr: None,
            hfr_source_counts: hfr_source_counts(config.hfr_source, &[]),
        };
        self.transition(FocusRunState::Initializing);
        self.current_run = Some(run.clone());

        self.transition(FocusRunState::Sweeping);
        run.state = FocusRunState::Sweeping;
        self.current_run = Some(run.clone());

        let sweep = sweep_positions(config, start_position);
        if sweep.len() < 5 {
            return self.fail_run(
                run,
                format!(
                    "Autofocus travel budget and position limits permit only {} distinct sweep positions; at least five are required",
                    sweep.len()
                ),
            );
        }

        for position in sweep {
            self.transition(FocusRunState::Moving);
            run.state = FocusRunState::Moving;
            self.current_run = Some(run.clone());

            let samples = match measure_at(position).await {
                Ok(samples) => samples,
                Err(err) => {
                    self.transition(FocusRunState::Failed);
                    run.state = FocusRunState::Failed;
                    self.current_run = Some(run.clone());
                    self.history.push(run);
                    return Err(err);
                }
            };

            self.transition(FocusRunState::Measuring);
            run.state = FocusRunState::Measuring;
            // Preserve every frame at this position; keep the most recent
            // temperature reading. The curve fit consumes repeated positions
            // directly, so per-position scatter is retained rather than averaged.
            if let Some(temperature) = samples.iter().rev().find_map(|sample| sample.temperature) {
                run.temperature = Some(temperature);
            }
            run.samples.extend(samples);
            run.hfr_source_counts = hfr_source_counts(config.hfr_source, &run.samples);
            self.current_run = Some(run.clone());
        }

        self.transition(FocusRunState::Evaluating);
        run.state = FocusRunState::Evaluating;
        self.current_run = Some(run.clone());

        // Repeated exposures quantify the distribution at each position. Reduce
        // them with a median before fitting so a position with one bad seeing
        // realization cannot pull the optimum, and so duplicate x values do not
        // masquerade as extra geometric support for the curve.
        let (positions, hfr_values) = aggregate_samples_by_position(&run.samples);
        if positions.len() < 5 {
            return self.fail_run(
                run,
                "Autofocus needs at least five distinct valid positions".to_string(),
            );
        }

        // Every candidate passes the same bracketing and confidence gates; the
        // best R² wins. The hyperbola is the physical model (linear blur growth
        // over a seeing floor), the V and parabola its limiting shapes.
        let fit = [
            ClassicalFocusSolver::fit_hyperbola(&positions, &hfr_values),
            ClassicalFocusSolver::fit_v_curve(&positions, &hfr_values),
            ClassicalFocusSolver::fit_parabola(&positions, &hfr_values),
        ]
        .into_iter()
        .flatten()
        .filter(|fit| {
            fit.r_squared >= config.safety.min_confidence_to_act
                && fit.predicted_hfr.is_finite()
                && fit.predicted_hfr > 0.0
        })
        .max_by(|a, b| a.r_squared.total_cmp(&b.r_squared));

        if let Some(fit) = fit {
            let best_position = fit.optimal_position.round() as i32;
            run.best_position =
                Some(best_position.clamp(config.safety.min_position, config.safety.max_position));
            run.best_hfr = Some(fit.predicted_hfr);
        } else {
            return self.fail_run(
                run,
                format!(
                    "No bracketed focus curve met the minimum fit confidence {:.2}",
                    config.safety.min_confidence_to_act
                ),
            );
        }

        self.transition(FocusRunState::Converged);
        run.state = FocusRunState::Converged;
        self.current_run = Some(run.clone());
        self.history.push(run.clone());
        Ok(run)
    }

    /// Abort the current focus run.
    pub fn abort(&mut self) {
        self.state = FocusRunState::Aborted;
        if let Some(run) = self.current_run.as_mut() {
            run.state = FocusRunState::Aborted;
        }
        tracing::warn!("Focus run aborted");
    }

    /// Downgrade a run after a post-fit hardware action fails. Curve fitting and
    /// the final focuser move live on opposite sides of the API/agent boundary;
    /// without this hook a failed move left status/history claiming Converged.
    pub fn mark_current_run_failed(&mut self, run_id: uuid::Uuid) -> Option<FocusRunContext> {
        let mut failed = self
            .current_run
            .as_ref()
            .filter(|run| run.run_id == run_id)?
            .clone();
        self.transition(FocusRunState::Failed);
        failed.state = FocusRunState::Failed;
        self.current_run = Some(failed.clone());
        if let Some(history) = self
            .history
            .iter_mut()
            .rev()
            .find(|run| run.run_id == run_id)
        {
            *history = failed.clone();
        } else {
            self.history.push(failed.clone());
        }
        Some(failed)
    }

    fn transition(&mut self, new_state: FocusRunState) {
        tracing::info!("Focus state: {:?} → {:?}", self.state, new_state);
        self.state = new_state;
    }

    fn fail_run(
        &mut self,
        mut run: FocusRunContext,
        message: String,
    ) -> crate::error::Result<FocusRunContext> {
        self.transition(FocusRunState::Failed);
        run.state = FocusRunState::Failed;
        self.current_run = Some(run.clone());
        self.history.push(run);
        Err(crate::error::FocusError::RunFailed(message).into())
    }
}

fn aggregate_samples_by_position(samples: &[FocusSample]) -> (Vec<i32>, Vec<f64>) {
    let mut grouped: BTreeMap<i32, Vec<f64>> = BTreeMap::new();
    for sample in samples {
        let hfr = sample.curve_fit_hfr();
        if hfr.is_finite() && hfr > 0.0 {
            grouped.entry(sample.position).or_default().push(hfr);
        }
    }
    let mut positions = Vec::with_capacity(grouped.len());
    let mut hfr_values = Vec::with_capacity(grouped.len());
    for (position, mut values) in grouped {
        values.sort_by(|a, b| a.total_cmp(b));
        let middle = values.len() / 2;
        let median = if values.len() % 2 == 1 {
            values[middle]
        } else {
            0.5 * (values[middle - 1] + values[middle])
        };
        positions.push(position);
        hfr_values.push(median);
    }
    (positions, hfr_values)
}

fn sweep_positions(config: &crate::config::FocusConfig, start_position: i32) -> Vec<i32> {
    let ascending = sweep_positions_ascending(config, start_position);
    order_sweep(ascending, start_position, config.sweep_pattern)
}

/// The set of positions to sample, low to high. Independent of visit order.
fn sweep_positions_ascending(
    config: &crate::config::FocusConfig,
    start_position: i32,
) -> Vec<i32> {
    let step = config
        .initial_step_size
        .saturating_abs()
        .max(1)
        .min(config.safety.max_step_single.saturating_abs().max(1));
    // The worst complete path is start -> one end -> other end -> fitted
    // optimum: 5R, for both patterns. Monotonic walks start->low->high->optimum
    // (R + 2R + 2R); reversal walks start->high->low->optimum (R + 2R + 2R).
    // Reserve the final return move up front so a valid sweep cannot exhaust the
    // run budget and leave the focuser at the last sampled position.
    let radius_points = i32::try_from(config.sweep_radius_points.max(1)).unwrap_or(i32::MAX);
    let max_radius = (config.safety.max_total_travel_per_run.max(0) / 5)
        .min(step.saturating_mul(radius_points));
    let mut positions = (-radius_points..=radius_points)
        .map(|offset: i32| start_position.saturating_add(offset.saturating_mul(step)))
        .filter(|position| {
            (i64::from(*position) - i64::from(start_position)).abs() <= i64::from(max_radius)
        })
        .map(|position| position.clamp(config.safety.min_position, config.safety.max_position))
        .collect::<Vec<_>>();
    positions.sort_unstable();
    positions.dedup();
    positions
}

/// Impose the visit order for `pattern` on an ascending position list.
///
/// Split from position selection so the two can be reasoned about separately:
/// which positions get sampled is a safety-and-optics question, the order they
/// are visited in is a data question, and conflating them is what made the
/// original `(-3..=3).sort()` look like a complete answer to both.
fn order_sweep(ascending: Vec<i32>, start_position: i32, pattern: SweepPattern) -> Vec<i32> {
    match pattern {
        SweepPattern::Monotonic => ascending,
        SweepPattern::Reversal => {
            // Split at the start position: everything at or above it is walked
            // outward first, then the remainder is walked back down. The flip
            // lands at the high end, so the run carries one clean direction
            // change with samples either side of it.
            let split = ascending.partition_point(|position| *position < start_position);

            // A start position outside the sampled range leaves nothing to
            // reverse through. Fall back rather than emit a sweep that claims a
            // flip it does not contain — the FOP's direction features would be
            // recorded as varying when they did not.
            if split == 0 || split == ascending.len() {
                return ascending;
            }

            let mut ordered = Vec::with_capacity(ascending.len());
            ordered.extend_from_slice(&ascending[split..]);
            ordered.extend(ascending[..split].iter().rev().copied());
            ordered
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        choose_fit_hfr, sweep_positions, FitHfrSource, FocusAgent, FocusRunState, FocusSample,
        FqeReading, HfrSource, SweepPattern, FQE_PROTOTYPE_MIN_STARS,
    };
    use crate::config::FocusConfig;

    fn prototype_reading(hfr: f64, stars: u32) -> FqeReading<'static> {
        FqeReading {
            hfr_px: Some(hfr),
            star_count: Some(stars),
            prototype: true,
            model_version: Some("1.1.0"),
        }
    }

    #[test]
    fn classical_hfr_source_leaves_the_fit_on_the_classical_hfr() {
        assert_eq!(HfrSource::default(), HfrSource::Classical);
        assert!(choose_fit_hfr(HfrSource::Classical, 2.0, &prototype_reading(3.0, 9)).is_none());
    }

    #[test]
    fn fqe_prototype_uses_a_finite_prediction_from_enough_stars_else_classical() {
        let fqe = HfrSource::FqePrototype;
        let used =
            choose_fit_hfr(fqe, 2.0, &prototype_reading(3.0, FQE_PROTOTYPE_MIN_STARS)).unwrap();
        assert_eq!((used.hfr, used.source), (3.0, FitHfrSource::FqePrototype));
        assert_eq!(used.fallback_reason, None);
        assert_eq!(used.fqe_model_version.as_deref(), Some("1.1.0"));

        let cases = [
            (FqeReading::default(), "no_fqe_prediction"),
            (
                FqeReading {
                    prototype: false,
                    ..prototype_reading(3.0, 9)
                },
                "fqe_not_prototype",
            ),
            (prototype_reading(f64::NAN, 9), "fqe_not_finite"),
            (prototype_reading(0.0, 9), "fqe_not_finite"),
            (
                prototype_reading(3.0, FQE_PROTOTYPE_MIN_STARS - 1),
                "fqe_too_few_stars",
            ),
            (
                FqeReading {
                    star_count: None,
                    ..prototype_reading(3.0, 9)
                },
                "fqe_too_few_stars",
            ),
        ];
        for (reading, reason) in cases {
            let fallback = choose_fit_hfr(fqe, 2.0, &reading).unwrap();
            assert_eq!(fallback.hfr, 2.0, "{reason}");
            assert_eq!(fallback.source, FitHfrSource::Classical, "{reason}");
            assert_eq!(fallback.fallback_reason.as_deref(), Some(reason));
        }
    }

    /// The classical V-curve bottoms at 10 000, the FQE's at 10 100. With
    /// `fqe_prototype` the fit follows the FQE where it has >= 3 stars; one
    /// position (10 050, where both curves read 1.45 px) has only two FQE
    /// stars and falls back to its classical HFR.
    #[tokio::test]
    async fn fqe_prototype_run_fits_the_fqe_hfr_and_records_its_sources() {
        let mut config = FocusConfig::default();
        config.initial_step_size = 50;
        config.safety.min_position = 0;
        config.safety.max_position = 20_000;
        config.safety.max_step_single = 500;
        config.safety.max_total_travel_per_run = 1_000;

        let measure = |source: HfrSource| {
            move |position: i32| async move {
                let classical = 1.2 + ((position - 10_000) as f64 / 100.0).powi(2);
                let fqe_hfr = 1.2 + ((position - 10_100) as f64 / 100.0).powi(2);
                let stars = if position == 10_050 { 2 } else { 5 };
                Ok(vec![FocusSample {
                    position,
                    hfr: classical,
                    hfd: Some(classical * 2.0),
                    fwhm: None,
                    star_count: 6,
                    quality_score: None,
                    temperature: None,
                    timestamp: chrono::Utc::now(),
                    frame_id: None,
                    fit_hfr: choose_fit_hfr(source, classical, &prototype_reading(fqe_hfr, stars)),
                }])
            }
        };

        let mut agent = FocusAgent::new();
        let classical = agent
            .run_autofocus(
                &config,
                "rig".into(),
                None,
                10_000,
                measure(HfrSource::Classical),
            )
            .await
            .unwrap();
        assert_eq!(classical.best_position, Some(10_000));
        assert!(classical.hfr_source_counts.is_none());
        assert!(classical
            .samples
            .iter()
            .all(|sample| sample.fit_hfr.is_none()));
        let json = serde_json::to_value(&classical).unwrap();
        assert!(json.get("hfr_source_counts").is_none());
        assert!(json["samples"][0].get("fit_hfr").is_none());

        config.hfr_source = HfrSource::FqePrototype;
        let prototype = agent
            .run_autofocus(
                &config,
                "rig".into(),
                None,
                10_000,
                measure(HfrSource::FqePrototype),
            )
            .await
            .unwrap();
        assert_eq!(prototype.state, FocusRunState::Converged);
        let best = prototype.best_position.unwrap();
        assert!((10_050..=10_150).contains(&best), "best {best}");
        let counts = prototype.hfr_source_counts.clone().unwrap();
        assert_eq!(counts.requested, HfrSource::FqePrototype);
        assert_eq!((counts.fqe_prototype, counts.classical), (6, 1));
        assert_eq!(counts.fallback_reasons.get("fqe_too_few_stars"), Some(&1));
        // `hfr` stays classical; the fit value and its source ride beside it.
        let fallback = prototype
            .samples
            .iter()
            .find(|sample| sample.position == 10_050)
            .unwrap();
        assert_eq!(fallback.curve_fit_hfr(), fallback.hfr);
        assert_eq!(fallback.fit_hfr.as_ref().unwrap().fqe_star_count, Some(2));
        let fqe_sample = prototype
            .samples
            .iter()
            .find(|sample| sample.position == 10_100)
            .unwrap();
        assert!((fqe_sample.curve_fit_hfr() - 1.2).abs() < 1e-12);
        assert!((fqe_sample.hfr - 2.2).abs() < 1e-12);
    }

    #[tokio::test]
    async fn autofocus_converges_on_synthetic_u_curve() {
        let mut config = FocusConfig::default();
        config.initial_step_size = 50;
        config.safety.min_position = 0;
        config.safety.max_position = 20_000;
        config.safety.max_step_single = 500;
        config.safety.max_total_travel_per_run = 1_000;

        let mut agent = FocusAgent::new();
        let run = agent
            .run_autofocus(
                &config,
                "test-rig".to_string(),
                Some("L".to_string()),
                10_000,
                |position| async move {
                    let hfr = 1.2 + ((position - 10_000) as f64 / 100.0).powi(2);
                    Ok(vec![FocusSample {
                        position,
                        hfr,
                        hfd: Some(hfr * 2.0),
                        fwhm: None,
                        star_count: 1,
                        quality_score: None,
                        temperature: Some(5.0),
                        timestamp: chrono::Utc::now(),
                        frame_id: None,
                        fit_hfr: None,
                    }])
                },
            )
            .await
            .unwrap();

        assert_eq!(run.state, FocusRunState::Converged);
        assert_eq!(run.filter_name.as_deref(), Some("L"));
        assert_eq!(run.best_position, Some(10_000));
        assert_eq!(run.samples.len(), 7);
        assert_eq!(agent.state(), FocusRunState::Converged);
        assert_eq!(agent.history().len(), 1);
    }

    #[tokio::test]
    async fn every_frame_is_preserved_not_averaged() {
        let mut config = FocusConfig::default();
        config.initial_step_size = 50;
        config.safety.min_position = 0;
        config.safety.max_position = 20_000;
        config.safety.max_step_single = 500;
        config.safety.max_total_travel_per_run = 1_000;

        // Three frames per position, each a distinct sample with its own frame id.
        let samples_per_position = 3usize;
        let mut agent = FocusAgent::new();
        let run = agent
            .run_autofocus(
                &config,
                "test-rig".to_string(),
                None,
                10_000,
                |position| async move {
                    let hfr = 1.2 + ((position - 10_000) as f64 / 100.0).powi(2);
                    Ok((0..3)
                        .map(|i| FocusSample {
                            position,
                            hfr,
                            hfd: Some(hfr * 2.0),
                            fwhm: None,
                            star_count: 1,
                            quality_score: None,
                            temperature: Some(5.0),
                            timestamp: chrono::Utc::now(),
                            frame_id: Some(format!("frame-{position}-{i}")),
                            fit_hfr: None,
                        })
                        .collect())
                },
            )
            .await
            .unwrap();

        // 7 sweep positions × 3 frames each, none collapsed.
        assert_eq!(run.samples.len(), 7 * samples_per_position);
        assert!(run.samples.iter().all(|sample| sample.frame_id.is_some()));
        let unique_frames: std::collections::HashSet<_> = run
            .samples
            .iter()
            .filter_map(|s| s.frame_id.clone())
            .collect();
        assert_eq!(unique_frames.len(), run.samples.len());
    }

    #[tokio::test]
    async fn post_fit_move_failure_downgrades_current_run_and_history() {
        let mut config = FocusConfig::default();
        config.initial_step_size = 50;
        config.safety.min_position = 0;
        config.safety.max_position = 20_000;
        config.safety.max_step_single = 500;
        config.safety.max_total_travel_per_run = 1_000;

        let mut agent = FocusAgent::new();
        let run = agent
            .run_autofocus(
                &config,
                "test-rig".to_string(),
                None,
                10_000,
                |position| async move {
                    let hfr = 1.2 + ((position - 10_000) as f64 / 100.0).powi(2);
                    Ok(vec![FocusSample {
                        position,
                        hfr,
                        hfd: Some(hfr * 2.0),
                        fwhm: None,
                        star_count: 1,
                        quality_score: None,
                        temperature: None,
                        timestamp: chrono::Utc::now(),
                        frame_id: None,
                        fit_hfr: None,
                    }])
                },
            )
            .await
            .unwrap();

        let failed = agent.mark_current_run_failed(run.run_id).unwrap();
        assert_eq!(failed.state, FocusRunState::Failed);
        assert_eq!(agent.state(), FocusRunState::Failed);
        assert_eq!(agent.current_run().unwrap().state, FocusRunState::Failed);
        assert_eq!(agent.history().last().unwrap().state, FocusRunState::Failed);
    }

    #[tokio::test]
    async fn insufficient_travel_budget_fails_before_the_first_measurement() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let mut config = FocusConfig::default();
        config.initial_step_size = 50;
        config.safety.min_position = 0;
        config.safety.max_position = 20_000;
        config.safety.max_total_travel_per_run = 300;

        let calls = Arc::new(AtomicUsize::new(0));
        let measured_calls = calls.clone();
        let mut agent = FocusAgent::new();
        let result = agent
            .run_autofocus(
                &config,
                "test-rig".to_string(),
                None,
                10_000,
                move |_position| {
                    let measured_calls = measured_calls.clone();
                    async move {
                        measured_calls.fetch_add(1, Ordering::SeqCst);
                        Ok(Vec::new())
                    }
                },
            )
            .await;

        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(agent.state(), FocusRunState::Failed);
    }

    #[test]
    fn sweep_positions_are_sorted_deduped_and_safety_clamped() {
        let mut config = FocusConfig::default();
        config.initial_step_size = 50;
        config.safety.min_position = 9_900;
        config.safety.max_position = 10_100;
        config.safety.max_step_single = 500;
        config.safety.max_total_travel_per_run = 500;

        let positions = sweep_positions(&config, 10_000);

        assert_eq!(positions, vec![9_900, 9_950, 10_000, 10_050, 10_100]);
    }

    fn wide_config() -> FocusConfig {
        let mut config = FocusConfig::default();
        config.initial_step_size = 50;
        config.safety.min_position = 0;
        config.safety.max_position = 20_000;
        config.safety.max_step_single = 500;
        config.safety.max_total_travel_per_run = 10_000;
        config
    }

    /// Sign changes in the position sequence. This is what the FOP's Direction
    /// feature group is computed from, so it is the thing worth asserting.
    fn direction_flips(positions: &[i32]) -> usize {
        positions
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).signum())
            .filter(|sign| *sign != 0)
            .collect::<Vec<_>>()
            .windows(2)
            .filter(|signs| signs[0] != signs[1])
            .count()
    }

    #[test]
    fn monotonic_sweep_never_reverses() {
        let mut config = wide_config();
        config.sweep_pattern = SweepPattern::Monotonic;

        let positions = sweep_positions(&config, 10_000);

        assert_eq!(direction_flips(&positions), 0);
        let mut sorted = positions.clone();
        sorted.sort_unstable();
        assert_eq!(positions, sorted);
    }

    /// The point of the whole pattern: one direction change, so
    /// `dir_changed_recently` and `steps_since_flip_norm` have something to be.
    #[test]
    fn reversal_sweep_changes_direction_exactly_once() {
        let mut config = wide_config();
        config.sweep_pattern = SweepPattern::Reversal;

        let positions = sweep_positions(&config, 10_000);

        assert_eq!(
            direction_flips(&positions),
            1,
            "reversal sweep should flip exactly once, got {positions:?}"
        );
    }

    #[test]
    fn reversal_sweep_visits_the_same_positions_as_monotonic() {
        // Both patterns must fit the same curve from the same samples; only the
        // order differs. A reversal that also changed the sampled set would
        // confound "did reversing hurt the fit" with "did we sample elsewhere".
        let mut monotonic = wide_config();
        monotonic.sweep_pattern = SweepPattern::Monotonic;
        let mut reversal = wide_config();
        reversal.sweep_pattern = SweepPattern::Reversal;

        let mut a = sweep_positions(&monotonic, 10_000);
        let mut b = sweep_positions(&reversal, 10_000);
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }

    #[test]
    fn reversal_sweep_starts_by_moving_outward_from_the_start() {
        let mut config = wide_config();
        config.sweep_pattern = SweepPattern::Reversal;

        let positions = sweep_positions(&config, 10_000);

        assert_eq!(positions[0], 10_000, "should begin where the focuser is");
        assert!(positions[1] > positions[0], "first leg runs outward");
        assert_eq!(
            *positions.last().unwrap(),
            10_000 - 3 * 50,
            "should finish at the low end after reversing"
        );
    }

    /// Clamping can push the start to the edge of the sampled range, leaving no
    /// samples on one side to reverse through.
    #[test]
    fn reversal_falls_back_to_monotonic_when_no_flip_is_possible() {
        let mut config = wide_config();
        config.sweep_pattern = SweepPattern::Reversal;
        config.safety.min_position = 10_000;
        config.safety.max_position = 10_150;

        let positions = sweep_positions(&config, 10_000);

        assert_eq!(direction_flips(&positions), 0);
        let mut sorted = positions.clone();
        sorted.sort_unstable();
        assert_eq!(
            positions, sorted,
            "with nothing below the start, the sweep is plain ascending"
        );
    }

    /// The campaign wants 25-position sweeps; the agent used to hard-code 7.
    #[test]
    fn sweep_radius_controls_the_position_count() {
        let mut config = wide_config();
        config.sweep_radius_points = 12;

        let positions = sweep_positions(&config, 10_000);

        assert_eq!(positions.len(), 25);
        assert_eq!(*positions.first().unwrap(), 10_000 - 12 * 50);
        assert_eq!(*positions.last().unwrap(), 10_000 + 12 * 50);
    }

    /// A wide sweep must still be truncated by the travel budget rather than
    /// overrunning it — the budget guards the focuser, not the dataset.
    #[test]
    fn travel_budget_still_truncates_a_wide_sweep() {
        let mut config = wide_config();
        config.sweep_radius_points = 12;
        config.safety.max_total_travel_per_run = 500; // radius <= 100 steps

        let positions = sweep_positions(&config, 10_000);

        assert!(
            positions.iter().all(|p| (p - 10_000).abs() <= 100),
            "positions {positions:?} exceed the travel budget"
        );
    }

    #[test]
    fn sweep_pattern_parses_the_request_spellings() {
        assert_eq!(SweepPattern::parse("reversal"), Some(SweepPattern::Reversal));
        assert_eq!(SweepPattern::parse("Reverse"), Some(SweepPattern::Reversal));
        assert_eq!(
            SweepPattern::parse(" MONOTONIC "),
            Some(SweepPattern::Monotonic)
        );
        assert_eq!(SweepPattern::parse("zigzag"), None);
        assert_eq!(SweepPattern::default(), SweepPattern::Monotonic);
    }
}

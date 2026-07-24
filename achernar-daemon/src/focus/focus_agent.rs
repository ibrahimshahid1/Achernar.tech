//! Focus agent — Section 7.3.4.
//! FocusRunState machine and run_autofocus() orchestration.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::future::Future;

use super::classical_focus_solver::ClassicalFocusSolver;

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
    pub best_hfr: Option<f64>,
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

    /// Start a new autofocus run.
    pub async fn run_autofocus<F, Fut>(
        &mut self,
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
            run_id: uuid::Uuid::new_v4(),
            rig_id,
            filter_name,
            temperature: None,
            start_position,
            state: FocusRunState::Initializing,
            started_at: chrono::Utc::now(),
            samples: Vec::new(),
            best_position: None,
            best_hfr: None,
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

        let fit = [
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
    for sample in samples
        .iter()
        .filter(|sample| sample.hfr.is_finite() && sample.hfr > 0.0)
    {
        grouped.entry(sample.position).or_default().push(sample.hfr);
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
    let step = config
        .initial_step_size
        .saturating_abs()
        .max(1)
        .min(config.safety.max_step_single.saturating_abs().max(1));
    // The worst complete path is start -> low -> high -> fitted optimum: 5R.
    // Reserve the final return move up front so a valid sweep cannot exhaust the
    // run budget and leave the focuser at the last sampled position.
    let max_radius =
        (config.safety.max_total_travel_per_run.max(0) / 5).min(step.saturating_mul(3));
    let mut positions = (-3..=3)
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

#[cfg(test)]
mod tests {
    use super::{sweep_positions, FocusAgent, FocusRunState, FocusSample};
    use crate::config::FocusConfig;

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
}

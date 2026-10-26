//! Guiding bandit — Section 8.3 (M-1, v6 §8.3.1; equipment spec §7.2.15).
//!
//! A LinUCB aggressiveness tuner for the RA and Dec guide algorithms.
//!
//! **Context.** [`GuidingContext::from_window`] builds the feature vector from
//! a hub [`DeviceWindow`] (guide RMS of the window, mount hour angle,
//! altitude, pier side and guide rate, wind) and the raw guide steps of the
//! same window (SNR, star mass). Required inputs that are missing stay
//! missing: the context is then incomplete and the decision freezes. Wind is
//! optional (few rigs have an anemometer) and carries an explicit
//! known/unknown indicator rather than a synthesised value.
//!
//! **Freeze rules.** [`GuidingFreeze`] lists every reason the outcome of an
//! arm cannot be attributed (v6 §8.3.1): a resume block, safety not allowing
//! automatic actions, PHD2 not guiding, a dither or settle, star loss, an
//! operator parameter edit (`guide_param_changed`), mount motion inside the
//! window, cloud, too few guide steps, or no discoverable aggressiveness
//! parameter. A frozen decision keeps the executed parameters, and the
//! bandit refuses to learn from a frozen window.
//!
//! **Decision record.** Every decision is an H23 `policy_decisions` row:
//! context, measurement ages, safe arms, the chosen arm per axis and the
//! `set_algo_param` proposals (L2). The choice is a deterministic argmax with
//! ties to the executed value, so no propensities are recorded. Actuation is
//! Tier 2 (WP2-7) and behind `[guider] allow_policy_actuation`, which the
//! gate enforces.

// The M-1 actuator and its scheduler are Tier 2 (WP2-7, WP2-8); until then
// only tests call the decision path.
#![allow(dead_code)]

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

use crate::equipment::commands::{Axis, GatedCommand, GatedOp, GuiderCmd};
use crate::equipment::events::{EventKind, PolicyDecisionRecord};
use crate::equipment::snapshot::{
    DeviceWindow, EquipmentSnapshot, GuideStepSample, GuiderAppState, GuiderSnapshot,
    ResumeBlockKind, SafetyGate,
};
use crate::equipment::types::{AutonomyMode, CommandSource, PierSide, Priority};

/// `policy_decisions.policy`.
pub const POLICY: &str = "m1_guiding_aggressiveness";
/// `policy_decisions.policy_version`.
pub const POLICY_VERSION: &str = "linucb-0.2";
/// Context feature names, in vector order.
pub const FEATURE_NAMES: [&str; 11] = [
    "bias",
    "rms_ra_arcsec",
    "rms_dec_arcsec",
    "snr",
    "star_mass_log10",
    "ha_h",
    "alt_deg_over_90",
    "pier_side_west",
    "guide_rate_ra_x_sidereal",
    "wind_known",
    "wind_speed_ms",
];
/// Fewer guide steps than this cannot attribute an outcome.
pub const MIN_GUIDE_STEPS: u32 = 10;
/// Cloud probability at or above this freezes learning.
pub const CLOUD_FREEZE_PROBABILITY: f32 = 0.5;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidingBanditState {
    pub ra_aggressiveness: f64,
    pub dec_aggressiveness: f64,
}

pub struct GuidingBandit {
    ra_alpha: f64,
    dec_alpha: f64,
    arms: Vec<f64>,
    ra_state: Vec<AxisArmState>,
    dec_state: Vec<AxisArmState>,
}

impl GuidingBandit {
    pub fn new(alpha: f64) -> Self {
        let arms = vec![0.10, 0.30, 0.50, 0.70, 1.00, 1.30];
        Self {
            ra_alpha: alpha,
            dec_alpha: alpha,
            ra_state: arms.iter().map(|_| AxisArmState::new(0)).collect(),
            dec_state: arms.iter().map(|_| AxisArmState::new(0)).collect(),
            arms,
        }
    }

    /// The arms, as fractions of the parameter's full scale.
    pub fn arms(&self) -> &[f64] {
        &self.arms
    }

    pub fn select_aggressiveness(&self, context: &[f64]) -> GuidingBanditState {
        let ra_aggressiveness =
            select_axis_arm(&self.arms, &self.ra_state, context, self.ra_alpha).unwrap_or(0.70);
        let dec_aggressiveness =
            select_axis_arm(&self.arms, &self.dec_state, context, self.dec_alpha).unwrap_or(0.70);

        GuidingBanditState {
            ra_aggressiveness,
            dec_aggressiveness,
        }
    }

    /// Credit `reward` to the arm that was actually executed on `axis`.
    ///
    /// LinUCB must update the executed arm, never a recomputed greedy arm:
    /// exploration deliberately runs non-greedy arms. Returns `false` and
    /// leaves all state unchanged when the axis or executed aggressiveness is
    /// not one of this bandit's arms, or the reward or context is not finite.
    pub fn update(
        &mut self,
        axis: &str,
        executed_aggressiveness: f64,
        reward: f64,
        context: &[f64],
    ) -> bool {
        if context.is_empty()
            || !reward.is_finite()
            || context.iter().any(|value| !value.is_finite())
        {
            return false;
        }
        let is_ra = axis.eq_ignore_ascii_case("ra");
        if !is_ra && !axis.eq_ignore_ascii_case("dec") {
            return false;
        }
        let Some(arm_idx) = self
            .arms
            .iter()
            .position(|arm| (arm - executed_aggressiveness).abs() <= 1e-9)
        else {
            return false;
        };

        self.ensure_dimension(context.len());
        let target = if is_ra {
            &mut self.ra_state
        } else {
            &mut self.dec_state
        };
        target[arm_idx].update(context, reward.clamp(-5.0, 2.0));
        true
    }

    /// Learn from one attributed window: refused while frozen or with an
    /// incomplete context. `executed` and `rewards` are per axis (RA, Dec);
    /// the reward is minus the RMS about zero, arcsec ([`window_rewards`]).
    pub fn learn(
        &mut self,
        ctx: &GuidingContext,
        freeze: &GuidingFreeze,
        executed: (Option<f64>, Option<f64>),
        rewards: (f64, f64),
    ) -> bool {
        if !freeze.reasons().is_empty() || !ctx.missing.is_empty() {
            return false;
        }
        let mut learned = false;
        if let Some(arm) = executed.0 {
            learned |= self.update("ra", arm, rewards.0, &ctx.features);
        }
        if let Some(arm) = executed.1 {
            learned |= self.update("dec", arm, rewards.1, &ctx.features);
        }
        learned
    }

    fn ensure_dimension(&mut self, dimension: usize) {
        for state in self.ra_state.iter_mut().chain(self.dec_state.iter_mut()) {
            if state.dimension() != dimension {
                *state = AxisArmState::new(dimension);
            }
        }
    }

    /// Decide the next aggressiveness per axis (see the module docs).
    pub fn decide(
        &self,
        ctx: &GuidingContext,
        freeze: &GuidingFreeze,
        guider: &GuiderSnapshot,
        limits: ActuationLimits,
        mode: AutonomyMode,
        now: DateTime<Utc>,
    ) -> GuidingDecision {
        let decision_id = Uuid::new_v4();
        let mut frozen = freeze.reasons();
        frozen.extend(ctx.missing.iter().map(|name| format!("missing:{name}")));
        let executed = ExecutedParams::from_snapshot(guider);
        let safe: Vec<f64> = self
            .arms
            .iter()
            .copied()
            .filter(|arm| *arm >= limits.min && *arm <= limits.max)
            .collect();
        let choose = |states: &[AxisArmState], alpha: f64, current: Option<f64>| {
            if !frozen.is_empty() {
                return current;
            }
            choose_with_ties_to_current(&self.arms, states, &safe, &ctx.features, alpha, current)
        };
        let ra = choose(&self.ra_state, self.ra_alpha, executed.ra_fraction);
        let dec = choose(&self.dec_state, self.dec_alpha, executed.dec_fraction);

        let mut proposals = Vec::new();
        for (axis, name, chosen, current, scale) in [
            (
                Axis::Ra,
                executed.ra_name.as_deref(),
                ra,
                executed.ra_fraction,
                executed.ra_scale,
            ),
            (
                Axis::Dec,
                executed.dec_name.as_deref(),
                dec,
                executed.dec_fraction,
                executed.dec_scale,
            ),
        ] {
            let (Some(name), Some(chosen)) = (name, chosen) else {
                continue;
            };
            if current.is_some_and(|c| (c - chosen).abs() <= 1e-9) {
                continue;
            }
            let value = chosen.clamp(limits.min, limits.max) * scale;
            proposals.push(
                GatedCommand::new(
                    GatedOp::Guider(GuiderCmd::SetAlgoParam {
                        axis,
                        name: name.to_string(),
                        value,
                    }),
                    CommandSource::Policy {
                        model: POLICY.to_string(),
                        version: POLICY_VERSION.to_string(),
                    },
                    mode,
                    Priority::Recommendation,
                )
                .with_params(json!({
                    "op": "set_algo_param",
                    "axis": axis,
                    "name": name,
                    "value": value,
                    "fraction": chosen,
                    "policy": POLICY,
                    "policy_version": POLICY_VERSION,
                    "decision_id": decision_id,
                })),
            );
        }

        let features: BTreeMap<String, f64> = ctx
            .feature_names
            .iter()
            .cloned()
            .zip(ctx.features.iter().copied())
            .collect();
        let arm_label = |value: Option<f64>| value.map(|v| format!("{v:.2}"));
        let record = PolicyDecisionRecord {
            decision_id,
            policy: POLICY.to_string(),
            policy_version: POLICY_VERSION.to_string(),
            mode,
            context: json!({
                "features": features,
                "feature_vector": ctx.features,
                "missing": ctx.missing,
                "window_start": ctx.window_start,
                "window_end": ctx.window_end,
                "freeze": freeze,
                "frozen": frozen,
                "executed": executed,
                "limits": limits,
            }),
            measurement_ages_ms: ctx.measurement_ages_ms.clone(),
            safe_arms: safe.iter().map(|arm| format!("{arm:.2}")).collect(),
            chosen_arm: Some(json!({"ra": arm_label(ra), "dec": arm_label(dec)}).to_string()),
            propensities: None,
            requested_action: Some(json!({"ra": ra, "dec": dec})),
            validated_action: None,
            executed_action: None,
            command_id: proposals.first().map(|cmd| cmd.command_id),
            reward_components: None,
            attribution_window_start: Some(ctx.window_start),
            attribution_window_end: Some(ctx.window_end),
            interrupted: false,
            override_applied: false,
            parent_frame_ids: ctx.parent_frame_ids.clone(),
            created_at: now,
        };
        GuidingDecision {
            decision_id,
            frozen,
            ra,
            dec,
            proposals,
            record,
        }
    }
}

/// `[guider] algo_param_min` and `algo_param_max`, as fractions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ActuationLimits {
    pub min: f64,
    pub max: f64,
}

impl Default for ActuationLimits {
    fn default() -> Self {
        Self { min: 0.1, max: 1.0 }
    }
}

/// The aggressiveness PHD2 is running, per axis, as a fraction of its scale.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ExecutedParams {
    pub ra_name: Option<String>,
    pub dec_name: Option<String>,
    pub ra_fraction: Option<f64>,
    pub dec_fraction: Option<f64>,
    /// 100 when PHD2 reports the parameter in percent, else 1 (G-j: the
    /// scale is learned from the current value).
    pub ra_scale: f64,
    pub dec_scale: f64,
}

impl ExecutedParams {
    pub fn from_snapshot(guider: &GuiderSnapshot) -> Self {
        let find = |axis: Axis, name: Option<&String>| {
            let name = name?;
            guider
                .algo_params
                .iter()
                .find(|p| p.axis == axis && p.name.eq_ignore_ascii_case(name))
                .map(|p| p.value)
                .filter(|v| v.is_finite())
        };
        let split = |value: Option<f64>| match value {
            Some(v) if v > 1.5 => (Some(v / 100.0), 100.0),
            Some(v) => (Some(v), 1.0),
            None => (None, 1.0),
        };
        let (ra_fraction, ra_scale) =
            split(find(Axis::Ra, guider.aggressiveness_param_ra.as_ref()));
        let (dec_fraction, dec_scale) =
            split(find(Axis::Dec, guider.aggressiveness_param_dec.as_ref()));
        Self {
            ra_name: guider.aggressiveness_param_ra.clone(),
            dec_name: guider.aggressiveness_param_dec.clone(),
            ra_fraction,
            dec_fraction,
            ra_scale,
            dec_scale,
        }
    }
}

/// The M-1 context of one window.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct GuidingContext {
    pub features: Vec<f64>,
    pub feature_names: Vec<String>,
    /// Required inputs that were unknown.
    pub missing: Vec<String>,
    pub measurement_ages_ms: BTreeMap<String, u64>,
    pub window_start: DateTime<Utc>,
    pub window_end: DateTime<Utc>,
    pub parent_frame_ids: Vec<String>,
}

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let (sum, n) = values
        .filter(|v| v.is_finite())
        .fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
    (n > 0).then(|| sum / n as f64)
}

impl GuidingContext {
    /// Build the context of `window` (a hub window), with the guide steps of
    /// the same interval (SNR and star mass come from them).
    pub fn from_window(
        window: &DeviceWindow,
        steps: &[GuideStepSample],
        parent_frame_ids: Vec<String>,
    ) -> Self {
        let mut missing = Vec::new();
        let mut ages = BTreeMap::new();
        let mut need = |name: &str, value: Option<f64>| -> f64 {
            match value.filter(|v| v.is_finite()) {
                Some(v) => v,
                None => {
                    missing.push(name.to_string());
                    0.0
                }
            }
        };
        let rms = window.guide.rms.as_ref();
        let rms_ra = need("rms_ra_arcsec", rms.and_then(|r| r.ra_arcsec));
        let rms_dec = need("rms_dec_arcsec", rms.and_then(|r| r.dec_arcsec));
        let guiding: Vec<&GuideStepSample> = steps
            .iter()
            .filter(|s| s.phase == crate::equipment::snapshot::GuidePhase::Guiding)
            .collect();
        let snr = need("snr", mean(guiding.iter().filter_map(|s| s.snr)));
        let star_mass = need(
            "star_mass_log10",
            mean(guiding.iter().filter_map(|s| s.star_mass))
                .filter(|m| *m > 0.0)
                .map(f64::log10),
        );
        let mount = window.mount.mid.as_ref().or(window.mount.start.as_ref());
        let ha = need("ha_h", mount.and_then(|m| m.ha_h));
        let alt = need(
            "alt_deg_over_90",
            mount
                .and_then(|m| m.state.alt_deg.known().copied())
                .map(|alt| alt / 90.0),
        );
        let pier = need(
            "pier_side_west",
            mount
                .and_then(|m| m.state.side_of_pier.known().copied())
                .and_then(|side| match side {
                    PierSide::West => Some(1.0),
                    PierSide::East => Some(0.0),
                    PierSide::Unknown => None,
                }),
        );
        let rate = need(
            "guide_rate_ra_x_sidereal",
            mount
                .and_then(|m| m.state.guide_rate_ra_arcsec_s.known().copied())
                .map(|r| r / crate::equipment::mount::SIDEREAL_RATE_ARCSEC_S),
        );
        if let Some(mount) = mount {
            if let Some(age) = (window.end - mount.published_at)
                .num_milliseconds()
                .checked_abs()
            {
                ages.insert("mount".to_string(), u64::try_from(age).unwrap_or(0));
            }
        }
        if let Some(last) = guiding.last() {
            let age = (window.end - last.received_at_utc)
                .num_milliseconds()
                .max(0);
            ages.insert("guide_steps".to_string(), u64::try_from(age).unwrap_or(0));
        }
        let wind = window
            .weather
            .as_ref()
            .and_then(|w| w.wind_speed_ms.known().copied())
            .filter(|w| w.is_finite());
        let features = vec![
            1.0,
            rms_ra,
            rms_dec,
            snr,
            star_mass,
            ha,
            alt,
            pier,
            rate,
            if wind.is_some() { 1.0 } else { 0.0 },
            wind.unwrap_or(0.0),
        ];
        Self {
            features,
            feature_names: FEATURE_NAMES.iter().map(|s| s.to_string()).collect(),
            missing,
            measurement_ages_ms: ages,
            window_start: window.start,
            window_end: window.end,
            parent_frame_ids,
        }
    }
}

/// Per-axis rewards of a window: minus the RMS about zero, arcsec.
pub fn window_rewards(
    window: &DeviceWindow,
    pixel_scale_arcsec: Option<f64>,
) -> Option<(f64, f64)> {
    let rms = window.guide.rms.as_ref()?;
    let scale = pixel_scale_arcsec.filter(|s| s.is_finite() && *s > 0.0)?;
    Some((-rms.rms0_ra_px * scale, -rms.rms0_dec_px * scale))
}

/// What can freeze M-1 (v6 §8.3.1). The default freezes nothing known.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct GuidingFreeze {
    pub resume_blocks: Vec<ResumeBlockKind>,
    pub safety: SafetyGate,
    pub guider_state: Option<GuiderAppState>,
    pub dither_in_window: bool,
    pub settle_in_window: bool,
    pub star_lost_in_window: bool,
    pub operator_param_edit: bool,
    pub mount_motion_in_window: bool,
    pub cloud: bool,
    pub guide_steps: u32,
    pub actuation_disabled_reason: Option<String>,
}

impl GuidingFreeze {
    /// The freeze inputs of `window`, from the hub snapshot and the window's
    /// events; `cloud_probability` comes from the vision bus.
    pub fn from_window(
        window: &DeviceWindow,
        snapshot: &EquipmentSnapshot,
        cloud_probability: Option<f32>,
    ) -> Self {
        let any = |pred: fn(&EventKind) -> bool| window.events.iter().any(|e| pred(&e.event));
        Self {
            resume_blocks: snapshot
                .resume_blocks
                .iter()
                .map(|b| b.kind.clone())
                .collect(),
            safety: snapshot.safety.state,
            guider_state: window.guide.state,
            dither_in_window: window.guide.dither_in_window,
            settle_in_window: window.guide.settle_in_window,
            star_lost_in_window: any(|e| {
                matches!(e, EventKind::StarLost { .. } | EventKind::LockPositionLost)
            }),
            operator_param_edit: any(|e| matches!(e, EventKind::GuideParamChanged { .. })),
            mount_motion_in_window: any(|e| {
                matches!(
                    e,
                    EventKind::SlewStarted
                        | EventKind::SlewSettled { .. }
                        | EventKind::MoveAxis { .. }
                        | EventKind::MeridianFlip { .. }
                        | EventKind::PierSideChanged { .. }
                )
            }) || window.motion.iter().any(|m| m.slewing == Some(true)),
            cloud: cloud_probability.is_some_and(|p| p >= CLOUD_FREEZE_PROBABILITY),
            guide_steps: window.guide.rms.as_ref().map_or(0, |r| r.steps_n),
            actuation_disabled_reason: snapshot
                .guider
                .as_ref()
                .and_then(|g| g.actuation_disabled_reason.clone()),
        }
    }

    /// Every reason the outcome cannot be attributed; empty when it can.
    pub fn reasons(&self) -> Vec<String> {
        let mut reasons = Vec::new();
        if !self.resume_blocks.is_empty() {
            reasons.push("resume_block_active".to_string());
        }
        if !self.safety.allows_automatic() {
            let state = serde_json::to_value(self.safety)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default();
            reasons.push(format!("safety_{state}"));
        }
        if self.guider_state != Some(GuiderAppState::Guiding) {
            reasons.push("guider_not_guiding".to_string());
        }
        if self.dither_in_window {
            reasons.push("dither_in_window".to_string());
        }
        if self.settle_in_window {
            reasons.push("settle_in_window".to_string());
        }
        if self.star_lost_in_window {
            reasons.push("star_lost_in_window".to_string());
        }
        if self.operator_param_edit {
            reasons.push("operator_param_edit".to_string());
        }
        if self.mount_motion_in_window {
            reasons.push("mount_motion_in_window".to_string());
        }
        if self.cloud {
            reasons.push("cloud".to_string());
        }
        if self.guide_steps < MIN_GUIDE_STEPS {
            reasons.push("too_few_guide_steps".to_string());
        }
        if let Some(reason) = &self.actuation_disabled_reason {
            reasons.push(reason.clone());
        }
        reasons
    }
}

/// One M-1 decision.
#[derive(Debug, Clone, PartialEq)]
pub struct GuidingDecision {
    pub decision_id: Uuid,
    pub frozen: Vec<String>,
    /// The chosen aggressiveness per axis (fractions); the executed value
    /// when frozen.
    pub ra: Option<f64>,
    pub dec: Option<f64>,
    /// `set_algo_param` proposals for the axes whose value changes.
    pub proposals: Vec<GatedCommand>,
    pub record: PolicyDecisionRecord,
}

/// The safe arm with the highest UCB; ties (including two untrained arms)
/// go to the executed value, so an untrained bandit never moves.
fn choose_with_ties_to_current(
    arms: &[f64],
    states: &[AxisArmState],
    safe: &[f64],
    context: &[f64],
    alpha: f64,
    current: Option<f64>,
) -> Option<f64> {
    let score = |arm: f64| {
        let idx = arms.iter().position(|a| (a - arm).abs() <= 1e-9)?;
        let state = &states[idx];
        Some(if state.dimension() == context.len() {
            axis_score(state, context, alpha).2
        } else {
            f64::INFINITY
        })
    };
    let current_score = current.and_then(score);
    let mut best: Option<(f64, f64)> = current.zip(current_score);
    for &arm in safe {
        let Some(s) = score(arm) else {
            continue;
        };
        match best {
            Some((_, b)) if b >= s => {}
            _ => best = Some((arm, s)),
        }
    }
    best.map(|(arm, _)| arm).or(current)
}

#[derive(Debug, Clone)]
struct AxisArmState {
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
}

impl AxisArmState {
    fn new(dimension: usize) -> Self {
        let mut a = vec![vec![0.0; dimension]; dimension];
        for (idx, row) in a.iter_mut().enumerate() {
            row[idx] = 1.0;
        }
        Self {
            a,
            b: vec![0.0; dimension],
        }
    }

    fn dimension(&self) -> usize {
        self.b.len()
    }

    fn update(&mut self, context: &[f64], reward: f64) {
        for row in 0..context.len() {
            self.b[row] += reward * context[row];
            for col in 0..context.len() {
                self.a[row][col] += context[row] * context[col];
            }
        }
    }
}

fn select_axis_arm(
    arms: &[f64],
    states: &[AxisArmState],
    context: &[f64],
    alpha: f64,
) -> Option<f64> {
    if context.is_empty() {
        return None;
    }

    states
        .iter()
        .enumerate()
        .filter(|(_, state)| state.dimension() == context.len())
        .map(|(idx, state)| (idx, axis_score(state, context, alpha).2))
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .and_then(|(idx, _)| arms.get(idx).copied())
}

fn axis_score(state: &AxisArmState, context: &[f64], alpha: f64) -> (f64, f64, f64) {
    if state.dimension() != context.len() {
        return (0.0, 1.0, f64::MAX);
    }
    let theta = solve_linear(&state.a, &state.b).unwrap_or_else(|| vec![0.0; context.len()]);
    let inv_context =
        solve_linear(&state.a, context).unwrap_or_else(|| context.iter().map(|_| 0.0).collect());
    let expected = dot(&theta, context);
    let uncertainty = dot(context, &inv_context).max(0.0).sqrt();
    (expected, uncertainty, expected + alpha * uncertainty)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn solve_linear(matrix: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = rhs.len();
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    let mut a = matrix.to_vec();
    let mut b = rhs.to_vec();

    for col in 0..n {
        let pivot = (col..n)
            .max_by(|a_idx, b_idx| a[*a_idx][col].abs().total_cmp(&a[*b_idx][col].abs()))?;
        if a[pivot][col].abs() <= f64::EPSILON {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let divisor = a[col][col];
        for value in &mut a[col][col..] {
            *value /= divisor;
        }
        b[col] /= divisor;
        let pivot_row = a[col][col..].to_vec();
        let pivot_rhs = b[col];
        for (row, (values, rhs)) in a.iter_mut().zip(b.iter_mut()).enumerate() {
            if row == col {
                continue;
            }
            let factor = values[col];
            for (value, pivot) in values[col..].iter_mut().zip(&pivot_row) {
                *value -= factor * pivot;
            }
            *rhs -= factor * pivot_rhs;
        }
    }

    Some(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::events::EquipmentEvent;
    use crate::equipment::snapshot::{
        AlgoParam, GuideRms, GuideWindowStats, MountSnapshot, MountState, WeatherReading,
        WindowSamples,
    };
    use crate::equipment::types::{Observed, Source};

    const CONTEXT: [f64; 6] = [1.0, 0.5, 0.2, 0.1, 0.8, 0.6];

    fn updated_arms(states: &[AxisArmState]) -> Vec<usize> {
        states
            .iter()
            .enumerate()
            .filter(|(_, state)| state.b.iter().any(|value| *value != 0.0))
            .map(|(idx, _)| idx)
            .collect()
    }

    #[test]
    fn update_credits_the_executed_arm_not_the_greedy_arm() {
        let mut bandit = GuidingBandit::new(1.0);
        // All arms tie before learning, so a greedy recomputation would pick
        // the last arm (1.30). The executed arm here is the first (0.10).
        assert!(bandit.update("ra", 0.10, 1.0, &CONTEXT));
        assert_eq!(updated_arms(&bandit.ra_state), vec![0]);
        assert!(updated_arms(&bandit.dec_state).is_empty());
    }

    #[test]
    fn exploratory_selection_is_credited_to_that_arm() {
        let mut bandit = GuidingBandit::new(1_000.0);
        // Teach the 0.70 arm a strong positive reward so it is greedy-best.
        for _ in 0..20 {
            assert!(bandit.update("dec", 0.70, 2.0, &CONTEXT));
        }
        let selected = bandit.select_aggressiveness(&CONTEXT).dec_aggressiveness;
        assert_ne!(selected, 0.70, "large alpha should explore an untried arm");

        let before = bandit.dec_state[3].b.clone();
        assert!(bandit.update("dec", selected, -1.0, &CONTEXT));
        let explored = bandit.arms.iter().position(|arm| *arm == selected).unwrap();
        assert!(bandit.dec_state[explored]
            .b
            .iter()
            .any(|value| *value != 0.0));
        assert_eq!(
            bandit.dec_state[3].b, before,
            "greedy arm must not absorb the reward"
        );
    }

    #[test]
    fn unknown_axis_or_arm_or_bad_reward_leaves_state_unchanged() {
        let mut bandit = GuidingBandit::new(1.0);
        assert!(!bandit.update("alt", 0.70, 1.0, &CONTEXT));
        assert!(!bandit.update("ra", 0.42, 1.0, &CONTEXT));
        assert!(!bandit.update("ra", 0.70, f64::NAN, &CONTEXT));
        let mut poisoned = CONTEXT;
        poisoned[4] = f64::NAN;
        assert!(!bandit.update("ra", 0.70, 1.0, &poisoned));
        assert!(!bandit.update("ra", 0.70, 1.0, &[]));
        // No dimension reset happened either: states remain uninitialised.
        assert!(bandit.ra_state.iter().all(|state| state.dimension() == 0));
        assert!(bandit.dec_state.iter().all(|state| state.dimension() == 0));
    }

    fn t(s: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_792_029_600 + s, 0).unwrap()
    }

    fn mount() -> MountSnapshot {
        let fresh = |v| Observed::fresh(v, Source::DeviceState, t(30), 0);
        MountSnapshot {
            state: MountState {
                alt_deg: fresh(45.0),
                guide_rate_ra_arcsec_s: fresh(7.52),
                side_of_pier: Observed::fresh(PierSide::West, Source::DeviceState, t(30), 0),
                ..MountState::default()
            },
            ha_h: Some(-1.25),
            published_at: t(30),
            ..MountSnapshot::default()
        }
    }

    fn step(s: i64, snr: f64) -> GuideStepSample {
        GuideStepSample {
            received_at_utc: t(s),
            snr: Some(snr),
            star_mass: Some(1000.0),
            ..GuideStepSample::default()
        }
    }

    fn window() -> DeviceWindow {
        DeviceWindow {
            start: t(0),
            end: t(60),
            mount: WindowSamples {
                mid: Some(mount()),
                samples_n: 1,
                ..WindowSamples::default()
            },
            guide: GuideWindowStats {
                state: Some(GuiderAppState::Guiding),
                rms: Some(GuideRms {
                    ra_arcsec: Some(0.8),
                    dec_arcsec: Some(0.6),
                    rms0_ra_px: 0.4,
                    rms0_dec_px: 0.3,
                    steps_n: 30,
                    ..GuideRms::default()
                }),
                ..GuideWindowStats::default()
            },
            ..DeviceWindow::default()
        }
    }

    fn guider() -> GuiderSnapshot {
        GuiderSnapshot {
            aggressiveness_param_ra: Some("aggression".into()),
            aggressiveness_param_dec: Some("Aggressiveness".into()),
            algo_params: vec![
                AlgoParam {
                    axis: Axis::Ra,
                    name: "aggression".into(),
                    value: 0.7,
                },
                AlgoParam {
                    axis: Axis::Dec,
                    name: "aggressiveness".into(),
                    value: 50.0,
                },
            ],
            ..GuiderSnapshot::default()
        }
    }

    #[test]
    fn the_context_comes_from_the_hub_window_and_its_guide_steps() {
        let steps = vec![step(10, 20.0), step(20, 30.0)];
        let ctx = GuidingContext::from_window(&window(), &steps, vec!["f-1".into()]);
        assert!(ctx.missing.is_empty(), "{:?}", ctx.missing);
        assert_eq!(ctx.features.len(), FEATURE_NAMES.len());
        let feature =
            |name: &str| ctx.features[FEATURE_NAMES.iter().position(|n| *n == name).unwrap()];
        assert_eq!(feature("rms_ra_arcsec"), 0.8);
        assert_eq!(feature("snr"), 25.0);
        assert_eq!(feature("star_mass_log10"), 3.0);
        assert_eq!(feature("ha_h"), -1.25);
        assert_eq!(feature("alt_deg_over_90"), 0.5);
        assert_eq!(feature("pier_side_west"), 1.0);
        assert!((feature("guide_rate_ra_x_sidereal") - 0.5).abs() < 1e-3);
        // No anemometer: an explicit unknown, not a synthesised calm.
        assert_eq!(feature("wind_known"), 0.0);
        assert_eq!(ctx.measurement_ages_ms["guide_steps"], 40_000);

        let mut windy = window();
        windy.weather = Some(WeatherReading {
            wind_speed_ms: Observed::fresh(4.0, Source::Property, t(30), 0),
            ..WeatherReading::default()
        });
        let ctx = GuidingContext::from_window(&windy, &steps, Vec::new());
        assert_eq!(ctx.features[9..], [1.0, 4.0]);
    }

    #[test]
    fn missing_required_inputs_stay_missing_and_freeze_the_decision() {
        let mut no_mount = window();
        no_mount.mount = WindowSamples::default();
        let ctx = GuidingContext::from_window(&no_mount, &[], Vec::new());
        for name in ["snr", "ha_h", "alt_deg_over_90", "pier_side_west"] {
            assert!(ctx.missing.contains(&name.to_string()), "{name}");
        }
        let freeze = GuidingFreeze {
            guider_state: Some(GuiderAppState::Guiding),
            guide_steps: 30,
            ..GuidingFreeze::default()
        };
        let decision = GuidingBandit::new(1.0).decide(
            &ctx,
            &freeze,
            &guider(),
            ActuationLimits::default(),
            AutonomyMode::Proposal,
            t(61),
        );
        assert!(decision.frozen.iter().any(|r| r == "missing:snr"));
        assert!(decision.proposals.is_empty());
        assert_eq!((decision.ra, decision.dec), (Some(0.7), Some(0.5)));
    }

    #[test]
    fn freeze_rules_read_the_window_events_and_the_snapshot() {
        let mut busy = window();
        let event = |kind| {
            EquipmentEvent::new(
                kind,
                &crate::equipment::testing::manual_clock::ManualClock::new(),
            )
        };
        busy.events = vec![
            event(EventKind::GuideParamChanged {
                name: "aggression".into(),
                value: json!(0.5),
            }),
            event(EventKind::LockPositionLost),
            event(EventKind::SlewStarted),
        ];
        busy.guide.dither_in_window = true;
        let mut snapshot = EquipmentSnapshot::default();
        snapshot.safety.state = SafetyGate::Unsafe;
        let freeze = GuidingFreeze::from_window(&busy, &snapshot, Some(0.9));
        let reasons = freeze.reasons();
        for reason in [
            "safety_unsafe",
            "dither_in_window",
            "star_lost_in_window",
            "operator_param_edit",
            "mount_motion_in_window",
            "cloud",
        ] {
            assert!(
                reasons.contains(&reason.to_string()),
                "{reason}: {reasons:?}"
            );
        }
        let calm = GuidingFreeze::from_window(&window(), &EquipmentSnapshot::default(), Some(0.1));
        assert!(calm.reasons().is_empty(), "{:?}", calm.reasons());
        let mut bandit = GuidingBandit::new(1.0);
        let ctx = GuidingContext::from_window(&window(), &[step(10, 20.0)], Vec::new());
        assert!(!bandit.learn(&ctx, &freeze, (Some(0.7), Some(0.5)), (-0.5, -0.4)));
        assert!(bandit.learn(&ctx, &calm, (Some(0.7), Some(0.5)), (-0.5, -0.4)));
        assert_eq!(window_rewards(&window(), Some(2.0)), Some((-0.8, -0.6)));
        assert_eq!(window_rewards(&window(), None), None);
    }

    #[test]
    fn an_untrained_bandit_keeps_the_executed_values_and_records_why() {
        let ctx = GuidingContext::from_window(&window(), &[step(10, 20.0)], vec!["f-1".into()]);
        let calm = GuidingFreeze::from_window(&window(), &EquipmentSnapshot::default(), None);
        let decision = GuidingBandit::new(1.0).decide(
            &ctx,
            &calm,
            &guider(),
            ActuationLimits::default(),
            AutonomyMode::Proposal,
            t(61),
        );
        assert!(decision.frozen.is_empty());
        // PHD2 reports Dec in percent (50 -> 0.5); both are arms.
        assert_eq!((decision.ra, decision.dec), (Some(0.7), Some(0.5)));
        assert!(decision.proposals.is_empty());
        let record = &decision.record;
        assert_eq!(record.policy, POLICY);
        assert_eq!(record.propensities, None);
        // 1.30 is outside [algo_param_min, algo_param_max].
        assert!(!record.safe_arms.contains(&"1.30".to_string()));
        assert_eq!(record.attribution_window_end, Some(t(60)));
        assert_eq!(record.parent_frame_ids, vec!["f-1".to_string()]);
    }

    #[test]
    fn a_learned_better_arm_is_proposed_through_the_gate_in_the_parameters_own_scale() {
        let ctx = GuidingContext::from_window(&window(), &[step(10, 20.0)], Vec::new());
        let calm = GuidingFreeze::from_window(&window(), &EquipmentSnapshot::default(), None);
        let mut bandit = GuidingBandit::new(0.0);
        for _ in 0..10 {
            assert!(bandit.update("ra", 0.70, -2.0, &ctx.features));
            assert!(bandit.update("ra", 0.30, 1.0, &ctx.features));
            assert!(bandit.update("dec", 0.50, 1.0, &ctx.features));
        }
        let decision = bandit.decide(
            &ctx,
            &calm,
            &guider(),
            ActuationLimits::default(),
            AutonomyMode::Proposal,
            t(61),
        );
        assert_eq!(decision.ra, Some(0.30));
        assert_eq!(decision.proposals.len(), 1);
        let proposal = &decision.proposals[0];
        assert_eq!(proposal.priority, Priority::Recommendation);
        match &proposal.op {
            GatedOp::Guider(GuiderCmd::SetAlgoParam { axis, name, value }) => {
                assert_eq!((*axis, name.as_str()), (Axis::Ra, "aggression"));
                assert!((value - 0.30).abs() < 1e-9);
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(decision.record.command_id, Some(proposal.command_id));
    }
}

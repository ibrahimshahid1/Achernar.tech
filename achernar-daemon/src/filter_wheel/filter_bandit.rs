//! Filter bandit — Section 8.2.3 (FW-2).
//!
//! A LinUCB contextual bandit over the wheel's filters plus an explicit
//! **continue** arm: stay on the current filter, with no mechanical change
//! (v6 §8.2.3). Ties go to continue.
//!
//! Exploration freezes when observations are invalid, the conditions are
//! unsafe or unknown, a resume block is active, weather, recovery or a locked
//! plan owns the sequence, or the current filter is not verified. A frozen
//! decision is always `continue`, and the bandit refuses to learn from it.
//!
//! Every decision is an H23 `policy_decisions` record: the context and the
//! age of each input, the safe arms, the chosen arm and the requested action.
//! The choice is a deterministic argmax, so no propensities are recorded (DC
//! RB-F: never invent propensities). A switch is proposed to the gate as a
//! filter handshake at Recommendation priority; the handshake then records
//! the arm and policy version in `filter_switch_events`.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

use crate::equipment::commands::{FilterTarget, GatedCommand, GatedOp};
use crate::equipment::events::{GroundTruthRecord, PolicyDecisionRecord};
use crate::equipment::filter::change::PolicyAttribution;
use crate::equipment::snapshot::{EquipmentSnapshot, ResumeBlockKind, SafetyGate};
use crate::equipment::traits::RecorderSink;
use crate::equipment::types::{AutonomyMode, CommandSource, Priority};

/// `policy_decisions.policy`.
pub const POLICY: &str = "fw2_filter_sequencer";
/// `policy_decisions.policy_version`.
pub const POLICY_VERSION: &str = "linucb-0.2";
/// The label of the continue arm.
pub const CONTINUE_ARM: &str = "continue";

/// One arm of the sequencer.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "arm", content = "filter", rename_all = "snake_case")]
pub enum FilterArm {
    /// Keep the current filter.
    Continue,
    /// Change to the named filter through the handshake.
    Switch(String),
}

impl FilterArm {
    pub fn label(&self) -> String {
        match self {
            FilterArm::Continue => CONTINUE_ARM.to_string(),
            FilterArm::Switch(filter) => format!("switch:{filter}"),
        }
    }
}

/// An arm and its current LinUCB estimate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterBanditArm {
    pub arm: FilterArm,
    pub estimated_reward: f64,
    pub uncertainty: f64,
    pub ucb_score: f64,
    pub times_pulled: u32,
}

/// What can freeze exploration (v6 §8.2.3). The default freezes: nothing is
/// known to be valid.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct FreezeInputs {
    pub observations_valid: bool,
    pub invalid_reason: Option<String>,
    pub safety: SafetyGate,
    pub resume_blocks: Vec<ResumeBlockKind>,
    pub weather_owns_sequence: bool,
    pub recovery_owns_sequence: bool,
    pub plan_locked: bool,
    pub current_filter_verified: bool,
}

impl FreezeInputs {
    /// Safety, resume blocks and wheel verification from the hub snapshot;
    /// the caller supplies observation validity and the plan lock.
    pub fn from_snapshot(
        snap: &EquipmentSnapshot,
        observations_valid: bool,
        plan_locked: bool,
    ) -> Self {
        Self {
            observations_valid,
            invalid_reason: None,
            safety: snap.safety.state,
            resume_blocks: snap.resume_blocks.iter().map(|b| b.kind.clone()).collect(),
            weather_owns_sequence: false,
            recovery_owns_sequence: false,
            plan_locked,
            current_filter_verified: snap.filter_wheel.as_ref().is_some_and(|w| w.verified),
        }
    }

    /// Every reason exploration is frozen; empty when it is not.
    pub fn reasons(&self) -> Vec<String> {
        let mut reasons = Vec::new();
        if !self.observations_valid {
            reasons.push(match &self.invalid_reason {
                Some(why) => format!("observations_invalid: {why}"),
                None => "observations_invalid".to_string(),
            });
        }
        if !self.safety.allows_automatic() {
            let state = serde_json::to_value(self.safety)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default();
            reasons.push(format!("safety_{state}"));
        }
        if !self.resume_blocks.is_empty() {
            reasons.push("resume_block_active".to_string());
        }
        if self.weather_owns_sequence {
            reasons.push("weather_owns_sequence".to_string());
        }
        if self.recovery_owns_sequence {
            reasons.push("recovery_owns_sequence".to_string());
        }
        if self.plan_locked {
            reasons.push("plan_locked".to_string());
        }
        if !self.current_filter_verified {
            reasons.push("current_filter_unverified".to_string());
        }
        reasons
    }
}

/// The decision context, built by the caller from hub windows and vision.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct BanditContext {
    /// The feature vector, in `feature_names` order.
    pub features: Vec<f64>,
    pub feature_names: Vec<String>,
    /// Age of each input measurement, ms.
    pub measurement_ages_ms: BTreeMap<String, u64>,
    /// The verified current filter.
    pub current_filter: Option<String>,
    /// Filters a switch may target now.
    pub switchable: Vec<String>,
    /// The exposure boundary, interval and frequency limits allow a switch.
    pub switch_allowed: bool,
    pub parent_frame_ids: Vec<String>,
}

/// One FW-2 decision.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterDecision {
    pub decision_id: Uuid,
    pub arm: FilterArm,
    /// Why exploration was frozen; empty when it was not.
    pub frozen: Vec<String>,
    pub safe_arms: Vec<FilterArm>,
    pub scores: Vec<FilterBanditArm>,
    /// The handshake to propose when the arm is a switch.
    pub proposal: Option<GatedCommand>,
    pub record: PolicyDecisionRecord,
}

impl FilterDecision {
    /// What the handshake records in `filter_switch_events`.
    pub fn attribution(&self) -> PolicyAttribution {
        PolicyAttribution {
            policy_version: Some(POLICY_VERSION.to_string()),
            arm: Some(self.arm.label()),
            propensity: None,
        }
    }
}

pub struct FilterBandit {
    alpha: f64,
    dimension: usize,
    arms: Vec<FilterBanditArm>,
    states: Vec<LinUcbState>,
}

#[derive(Debug, Clone)]
struct LinUcbState {
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
}

impl FilterBandit {
    /// The continue arm plus one arm per distinct filter name, for contexts
    /// of `dimension` features.
    pub fn new(filter_names: &[String], dimension: usize, alpha: f64) -> Self {
        let mut arms = vec![FilterArm::Continue];
        for name in filter_names {
            let name = name.trim();
            if !name.is_empty() && !arms.contains(&FilterArm::Switch(name.to_string())) {
                arms.push(FilterArm::Switch(name.to_string()));
            }
        }
        let arms: Vec<FilterBanditArm> = arms
            .into_iter()
            .map(|arm| FilterBanditArm {
                arm,
                estimated_reward: 0.0,
                uncertainty: 1.0,
                ucb_score: f64::MAX,
                times_pulled: 0,
            })
            .collect();
        let states = arms.iter().map(|_| LinUcbState::new(dimension)).collect();
        Self {
            alpha: if alpha.is_finite() {
                alpha.max(0.0)
            } else {
                0.0
            },
            dimension,
            arms,
            states,
        }
    }

    pub fn arms(&self) -> &[FilterBanditArm] {
        &self.arms
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    fn index_of(&self, arm: &FilterArm) -> Option<usize> {
        self.arms.iter().position(|a| &a.arm == arm)
    }

    fn context_problem(&self, features: &[f64]) -> Option<String> {
        if features.len() != self.dimension {
            Some(format!(
                "context_dimension_mismatch: expected {}, got {}",
                self.dimension,
                features.len()
            ))
        } else if features.iter().any(|x| !x.is_finite()) {
            Some("context_not_finite".to_string())
        } else {
            None
        }
    }

    /// Choose an arm. Never panics; a frozen or degenerate context continues.
    pub fn decide(
        &self,
        ctx: &BanditContext,
        freeze: &FreezeInputs,
        mode: AutonomyMode,
        now: DateTime<Utc>,
    ) -> FilterDecision {
        let decision_id = Uuid::new_v4();
        let mut frozen = freeze.reasons();
        let context_ok = match self.context_problem(&ctx.features) {
            Some(problem) => {
                frozen.push(problem);
                false
            }
            None => true,
        };

        let scores: Vec<FilterBanditArm> = self
            .arms
            .iter()
            .zip(&self.states)
            .map(|(arm, state)| {
                let mut arm = arm.clone();
                if context_ok {
                    let (reward, uncertainty, ucb) = linucb_score(state, &ctx.features, self.alpha);
                    arm.estimated_reward = reward;
                    arm.uncertainty = uncertainty;
                    arm.ucb_score = ucb;
                }
                arm
            })
            .collect();

        let current = ctx.current_filter.as_deref().map(str::trim);
        let switchable = |name: &str| {
            ctx.switchable
                .iter()
                .any(|s| s.trim().eq_ignore_ascii_case(name))
                && current.is_none_or(|c| !c.eq_ignore_ascii_case(name))
        };
        let safe_arms: Vec<FilterArm> = scores
            .iter()
            .filter(|s| match &s.arm {
                FilterArm::Continue => true,
                FilterArm::Switch(name) => {
                    frozen.is_empty() && ctx.switch_allowed && switchable(name)
                }
            })
            .map(|s| s.arm.clone())
            .collect();

        let arm = if frozen.is_empty() {
            // First maximum wins, and continue is first: ties never switch.
            scores
                .iter()
                .filter(|s| safe_arms.contains(&s.arm))
                .fold(None::<&FilterBanditArm>, |best, s| match best {
                    Some(b) if b.ucb_score >= s.ucb_score => Some(b),
                    _ => Some(s),
                })
                .map_or(FilterArm::Continue, |s| s.arm.clone())
        } else {
            FilterArm::Continue
        };

        let proposal = match &arm {
            FilterArm::Switch(filter) => {
                let params = json!({
                    "op": "filter_handshake",
                    "target": {"name": filter},
                    "policy": POLICY,
                    "policy_version": POLICY_VERSION,
                    "decision_id": decision_id,
                    "arm": arm.label(),
                });
                Some(
                    GatedCommand::new(
                        GatedOp::FilterHandshake {
                            target: FilterTarget::Name(filter.clone()),
                        },
                        CommandSource::Policy {
                            model: POLICY.to_string(),
                            version: POLICY_VERSION.to_string(),
                        },
                        mode,
                        Priority::Recommendation,
                    )
                    .with_params(params),
                )
            }
            FilterArm::Continue => None,
        };

        let features: BTreeMap<String, f64> = ctx
            .feature_names
            .iter()
            .cloned()
            .zip(ctx.features.iter().copied())
            .collect();
        let record = PolicyDecisionRecord {
            decision_id,
            policy: POLICY.to_string(),
            policy_version: POLICY_VERSION.to_string(),
            mode,
            context: json!({
                "features": features,
                "feature_vector": ctx.features,
                "current_filter": ctx.current_filter,
                "switchable": ctx.switchable,
                "switch_allowed": ctx.switch_allowed,
                "freeze": freeze,
                "frozen": frozen,
                "scores": scores,
            }),
            measurement_ages_ms: ctx.measurement_ages_ms.clone(),
            safe_arms: safe_arms.iter().map(FilterArm::label).collect(),
            chosen_arm: Some(arm.label()),
            propensities: None,
            requested_action: Some(match &arm {
                FilterArm::Continue => json!({"arm": CONTINUE_ARM}),
                FilterArm::Switch(filter) => json!({"arm": arm.label(), "target": filter}),
            }),
            validated_action: None,
            executed_action: None,
            command_id: proposal.as_ref().map(|cmd| cmd.command_id),
            reward_components: None,
            attribution_window_start: None,
            attribution_window_end: None,
            interrupted: false,
            override_applied: false,
            parent_frame_ids: ctx.parent_frame_ids.clone(),
            created_at: now,
        };

        FilterDecision {
            decision_id,
            arm,
            frozen,
            safe_arms,
            scores,
            proposal,
            record,
        }
    }

    /// [`decide`](Self::decide), then write the H23 record.
    pub fn decide_and_record(
        &self,
        ctx: &BanditContext,
        freeze: &FreezeInputs,
        mode: AutonomyMode,
        now: DateTime<Utc>,
        sink: &dyn RecorderSink,
    ) -> FilterDecision {
        let decision = self.decide(ctx, freeze, mode, now);
        sink.record(GroundTruthRecord::Decision(Box::new(
            decision.record.clone(),
        )));
        decision
    }

    /// Learn from an observed reward. Refused while exploration is frozen,
    /// for an unknown arm, or for an invalid context or reward.
    pub fn update(
        &mut self,
        arm: &FilterArm,
        reward: f64,
        context: &[f64],
        freeze: &FreezeInputs,
    ) -> Result<(), String> {
        let reasons = freeze.reasons();
        if !reasons.is_empty() {
            return Err(format!("learning is frozen: {}", reasons.join(", ")));
        }
        if let Some(problem) = self.context_problem(context) {
            return Err(problem);
        }
        if !reward.is_finite() {
            return Err("the reward is not finite".into());
        }
        let idx = self
            .index_of(arm)
            .ok_or_else(|| format!("unknown arm {}", arm.label()))?;
        self.states[idx].update(context, reward.clamp(-1.0, 1.0));
        let (estimated_reward, uncertainty, ucb_score) =
            linucb_score(&self.states[idx], context, self.alpha);
        let entry = &mut self.arms[idx];
        entry.estimated_reward = estimated_reward;
        entry.uncertainty = uncertainty;
        entry.ucb_score = ucb_score;
        entry.times_pulled = entry.times_pulled.saturating_add(1);
        Ok(())
    }
}

impl LinUcbState {
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

fn linucb_score(state: &LinUcbState, context: &[f64], alpha: f64) -> (f64, f64, f64) {
    if state.dimension() != context.len() {
        return (0.0, 1.0, f64::MAX);
    }

    let theta = solve_linear(&state.a, &state.b).unwrap_or_else(|| vec![0.0; context.len()]);
    let inv_context =
        solve_linear(&state.a, context).unwrap_or_else(|| context.iter().map(|_| 0.0).collect());
    let estimated_reward = dot(&theta, context);
    let uncertainty = dot(context, &inv_context).max(0.0).sqrt();
    let ucb_score = estimated_reward + alpha * uncertainty;
    (estimated_reward, uncertainty, ucb_score)
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
        for value in a[col].iter_mut().skip(col) {
            *value /= divisor;
        }
        b[col] /= divisor;

        let pivot_row = a[col].clone();
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            for (value, pivot) in a[row].iter_mut().zip(&pivot_row).skip(col) {
                *value -= factor * pivot;
            }
            b[row] -= factor * b[col];
        }
    }

    Some(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::snapshot::{FilterWheelSnapshot, ResumeBlock};
    use crate::equipment::testing::fake_handles::RecordingSink;
    use std::sync::Arc;

    fn names() -> Vec<String> {
        ["L", "R", "Ha"].iter().map(|n| n.to_string()).collect()
    }

    fn open() -> FreezeInputs {
        FreezeInputs {
            observations_valid: true,
            safety: SafetyGate::Safe,
            current_filter_verified: true,
            ..FreezeInputs::default()
        }
    }

    fn context(features: Vec<f64>) -> BanditContext {
        BanditContext {
            feature_names: (0..features.len()).map(|i| format!("x{i}")).collect(),
            features,
            measurement_ages_ms: [("background".to_string(), 1200)].into_iter().collect(),
            current_filter: Some("L".into()),
            switchable: names(),
            switch_allowed: true,
            parent_frame_ids: vec!["frame-1".into()],
        }
    }

    #[test]
    fn the_continue_arm_is_always_present_and_wins_ties() {
        let bandit = FilterBandit::new(&names(), 2, 1.0);
        assert_eq!(bandit.arms()[0].arm, FilterArm::Continue);
        assert_eq!(bandit.arms().len(), 4);
        let decision = bandit.decide(
            &context(vec![1.0, 0.5]),
            &open(),
            AutonomyMode::Proposal,
            Utc::now(),
        );
        assert!(decision.frozen.is_empty());
        assert_eq!(
            decision.arm,
            FilterArm::Continue,
            "an untrained bandit does not move the wheel"
        );
        assert!(decision.proposal.is_none());
        // A wheel with no filters still decides (it used to panic).
        let empty = FilterBandit::new(&[], 2, 1.0);
        let decision = empty.decide(
            &context(vec![1.0, 0.5]),
            &open(),
            AutonomyMode::Proposal,
            Utc::now(),
        );
        assert_eq!(decision.arm, FilterArm::Continue);
        assert_eq!(decision.safe_arms, vec![FilterArm::Continue]);
    }

    #[test]
    fn a_learned_switch_is_proposed_at_recommendation_priority() {
        let mut bandit = FilterBandit::new(&names(), 2, 0.1);
        let x = [1.0, 0.0];
        for _ in 0..5 {
            bandit
                .update(&FilterArm::Switch("Ha".into()), 0.9, &x, &open())
                .unwrap();
            bandit
                .update(&FilterArm::Continue, -0.2, &x, &open())
                .unwrap();
        }
        let sink = RecordingSink::new();
        let decision = bandit.decide_and_record(
            &context(x.to_vec()),
            &open(),
            AutonomyMode::Proposal,
            Utc::now(),
            sink.as_ref(),
        );
        assert_eq!(decision.arm, FilterArm::Switch("Ha".into()));
        let proposal = decision.proposal.clone().unwrap();
        assert_eq!(proposal.priority, Priority::Recommendation);
        assert!(
            matches!(proposal.source, CommandSource::Policy { ref model, .. } if model == POLICY)
        );
        assert_eq!(
            proposal.op,
            GatedOp::FilterHandshake {
                target: FilterTarget::Name("Ha".into())
            }
        );
        let record = &decision.record;
        assert_eq!(record.command_id, Some(proposal.command_id));
        assert_eq!(record.chosen_arm.as_deref(), Some("switch:Ha"));
        assert_eq!(
            record.propensities, None,
            "a deterministic choice has no propensity"
        );
        assert_eq!(record.measurement_ages_ms["background"], 1200);
        assert_eq!(record.parent_frame_ids, vec!["frame-1".to_string()]);
        assert!(record.safe_arms.contains(&CONTINUE_ARM.to_string()));
        assert!(
            !record.safe_arms.contains(&"switch:L".to_string()),
            "the current filter is the continue arm, not a switch"
        );
        assert!(matches!(
            &sink.records()[0],
            GroundTruthRecord::Decision(rec) if rec.decision_id == decision.decision_id
        ));
        let attribution = decision.attribution();
        assert_eq!(attribution.arm.as_deref(), Some("switch:Ha"));
        assert_eq!(attribution.propensity, None);

        // A switch that the limits do not allow is not a safe arm.
        let mut blocked = context(x.to_vec());
        blocked.switch_allowed = false;
        let decision = bandit.decide(&blocked, &open(), AutonomyMode::Proposal, Utc::now());
        assert_eq!(decision.arm, FilterArm::Continue);
        assert_eq!(decision.safe_arms, vec![FilterArm::Continue]);
    }

    #[test]
    fn exploration_freezes_and_learning_is_refused_while_frozen() {
        let mut bandit = FilterBandit::new(&names(), 2, 0.1);
        let x = [1.0, 0.0];
        for _ in 0..5 {
            bandit
                .update(&FilterArm::Switch("Ha".into()), 0.9, &x, &open())
                .unwrap();
        }
        let cases: Vec<(FreezeInputs, &str)> = vec![
            (
                FreezeInputs {
                    observations_valid: false,
                    invalid_reason: Some("cloud".into()),
                    ..open()
                },
                "observations_invalid: cloud",
            ),
            (
                FreezeInputs {
                    safety: SafetyGate::Unsafe,
                    ..open()
                },
                "safety_unsafe",
            ),
            (
                FreezeInputs {
                    safety: SafetyGate::Unknown,
                    ..open()
                },
                "safety_unknown",
            ),
            (
                FreezeInputs {
                    resume_blocks: vec![ResumeBlockKind::FilterHandshakeIncomplete],
                    ..open()
                },
                "resume_block_active",
            ),
            (
                FreezeInputs {
                    weather_owns_sequence: true,
                    ..open()
                },
                "weather_owns_sequence",
            ),
            (
                FreezeInputs {
                    recovery_owns_sequence: true,
                    ..open()
                },
                "recovery_owns_sequence",
            ),
            (
                FreezeInputs {
                    plan_locked: true,
                    ..open()
                },
                "plan_locked",
            ),
            (
                FreezeInputs {
                    current_filter_verified: false,
                    ..open()
                },
                "current_filter_unverified",
            ),
        ];
        for (freeze, reason) in cases {
            let decision = bandit.decide(
                &context(x.to_vec()),
                &freeze,
                AutonomyMode::Proposal,
                Utc::now(),
            );
            assert_eq!(decision.arm, FilterArm::Continue, "{reason}");
            assert_eq!(decision.frozen, vec![reason.to_string()]);
            assert_eq!(decision.safe_arms, vec![FilterArm::Continue]);
            assert_eq!(decision.record.context["frozen"][0], reason);
            assert!(bandit
                .update(&FilterArm::Continue, 1.0, &x, &freeze)
                .unwrap_err()
                .contains(reason));
        }
        assert!(
            !FreezeInputs::default().reasons().is_empty(),
            "the default freezes"
        );

        // A context of the wrong shape freezes too.
        let decision = bandit.decide(
            &context(vec![1.0]),
            &open(),
            AutonomyMode::Proposal,
            Utc::now(),
        );
        assert_eq!(decision.arm, FilterArm::Continue);
        assert!(decision.frozen[0].starts_with("context_dimension_mismatch"));
        assert!(bandit
            .update(&FilterArm::Switch("SII".into()), 0.5, &x, &open())
            .is_err());
        assert!(bandit
            .update(&FilterArm::Continue, f64::NAN, &x, &open())
            .is_err());
    }

    #[test]
    fn freeze_inputs_come_from_the_hub_snapshot() {
        let mut snap = EquipmentSnapshot::empty(Utc::now(), 1);
        let freeze = FreezeInputs::from_snapshot(&snap, true, false);
        assert!(
            !freeze.current_filter_verified,
            "no wheel, nothing verified"
        );
        assert_eq!(freeze.safety, SafetyGate::None);
        snap.filter_wheel = Some(Arc::new(FilterWheelSnapshot {
            verified: true,
            ..FilterWheelSnapshot::default()
        }));
        snap.safety.state = SafetyGate::Safe;
        snap.resume_blocks.push(ResumeBlock {
            id: Uuid::new_v4(),
            kind: ResumeBlockKind::GuiderLost,
            raised_at: Utc::now(),
            raised_mono_ns: 1,
            device_kind: None,
            connection_id: None,
            command_id: None,
            reason: "guider lost".into(),
        });
        let freeze = FreezeInputs::from_snapshot(&snap, true, true);
        assert!(freeze.current_filter_verified);
        assert_eq!(
            freeze.reasons(),
            vec!["resume_block_active".to_string(), "plan_locked".to_string()]
        );
    }
}

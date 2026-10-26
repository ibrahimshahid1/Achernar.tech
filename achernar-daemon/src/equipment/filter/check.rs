//! The filter-wheel `SafetyCheck` plug-in (spec §2.9 step 3, v6 §8.2.5).
//! Owner: WP1A-6.
//!
//! Rules, shared by the gate check and the handshake's own preconditions
//! (W-c step 1) through [`evaluate`]:
//!
//! 1. No switch while the camera integrates an exposure. When the exposing
//!    state is unknown, automatic sources are refused and an operator change
//!    is allowed with the audited note `exposure_state_unknown`.
//! 2. A locked observing plan cannot be overridden by a recommendation
//!    source (FW-2 or any policy).
//! 3. `min_switch_interval_s` applies to every source.
//! 4. A rolling one-hour frequency cap applies to automatic sources.
//!
//! Minimum settling, verified position and focus coordination are enforced
//! by the handshake itself (`change.rs`).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::equipment::clock::duration_ns;
use crate::equipment::commands::{GatedCommand, GatedOp, SafetyRejection};
use crate::equipment::snapshot::EquipmentSnapshot;
use crate::equipment::traits::SafetyCheck;
use crate::equipment::types::{CommandSource, Observed, Priority};

use super::FilterSettings;

/// `safety_rejected{check}` name of this plug-in.
pub const CHECK_NAME: &str = "filter_wheel";

const HOUR: Duration = Duration::from_secs(3600);

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Monotonic times of commanded wheel moves, kept for one hour.
#[derive(Debug, Default)]
pub struct SwitchHistory {
    times: Mutex<VecDeque<u64>>,
}

impl SwitchHistory {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record a commanded switch at `at_mono_ns`.
    pub fn record(&self, at_mono_ns: u64) {
        let mut times = lock(&self.times);
        times.push_back(at_mono_ns);
        let horizon = at_mono_ns.saturating_sub(duration_ns(HOUR));
        while times.front().is_some_and(|t| *t < horizon) {
            times.pop_front();
        }
    }

    pub fn last(&self) -> Option<u64> {
        lock(&self.times).back().copied()
    }

    /// Switches at or after `since_mono_ns`.
    pub fn count_since(&self, since_mono_ns: u64) -> usize {
        lock(&self.times)
            .iter()
            .filter(|t| **t >= since_mono_ns)
            .count()
    }
}

/// A locked observing plan (FW-3 or an operator plan).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LockedPlan {
    pub plan_id: String,
    pub locked_at: DateTime<Utc>,
    pub reason: Option<String>,
}

/// Whether a plan currently owns the filter sequence (v6 §8.2.5).
#[derive(Debug, Default)]
pub struct PlanLock {
    plan: Mutex<Option<LockedPlan>>,
}

impl PlanLock {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn lock(&self, plan: LockedPlan) {
        *lock(&self.plan) = Some(plan);
    }

    pub fn unlock(&self) -> Option<LockedPlan> {
        lock(&self.plan).take()
    }

    pub fn current(&self) -> Option<LockedPlan> {
        lock(&self.plan).clone()
    }

    pub fn is_locked(&self) -> bool {
        lock(&self.plan).is_some()
    }
}

/// Interval and frequency limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwitchLimits {
    pub min_interval: Duration,
    pub max_automatic_per_hour: Option<u32>,
}

impl SwitchLimits {
    pub fn from_settings(settings: &FilterSettings) -> Self {
        Self {
            min_interval: settings.min_switch_interval(),
            max_automatic_per_hour: settings.max_automatic_switches_per_hour,
        }
    }
}

impl Default for SwitchLimits {
    fn default() -> Self {
        Self::from_settings(&FilterSettings::default())
    }
}

/// What a switch request is judged on.
#[derive(Debug, Clone, Copy)]
pub struct SwitchContext<'a> {
    pub source: &'a CommandSource,
    pub priority: Priority,
    pub exposing: &'a Observed<bool>,
    pub now_mono_ns: u64,
}

fn reject(reason: impl Into<String>) -> SafetyRejection {
    SafetyRejection::new(CHECK_NAME, reason)
}

/// Recommendation sources (FW-2 and any policy) never override a locked
/// plan.
fn is_recommendation(source: &CommandSource, priority: Priority) -> bool {
    priority == Priority::Recommendation || matches!(source, CommandSource::Policy { .. })
}

/// The shared rule set. `Ok` carries audit annotations for an accepted
/// request.
pub fn evaluate(
    ctx: &SwitchContext<'_>,
    limits: &SwitchLimits,
    history: &SwitchHistory,
    plan: &PlanLock,
) -> Result<Vec<String>, SafetyRejection> {
    let mut notes = Vec::new();

    // 1. Exposure boundary. A known `true`, even a stale one, refuses.
    if ctx.exposing.known() == Some(&true) {
        return Err(reject(
            "exposing: filter changes run only between exposures",
        ));
    }
    if ctx.exposing.fresh_value().is_none() {
        if ctx.source.is_automatic() {
            return Err(reject(
                "exposure_state_unknown: automatic filter changes need a known exposure state",
            ));
        }
        notes.push("exposure_state_unknown".to_string());
    }

    // 2. Locked plan.
    if let Some(locked) = plan.current() {
        if is_recommendation(ctx.source, ctx.priority) {
            return Err(reject(format!(
                "plan_locked: plan '{}' owns the filter sequence; a recommendation cannot override it",
                locked.plan_id
            )));
        }
    }

    // 3. Minimum interval, every source.
    if !limits.min_interval.is_zero() {
        if let Some(last) = history.last() {
            let since = ctx.now_mono_ns.saturating_sub(last);
            if since < duration_ns(limits.min_interval) {
                return Err(reject(format!(
                    "min_switch_interval: {} ms since the last switch, {} ms required",
                    since / 1_000_000,
                    limits.min_interval.as_millis()
                )));
            }
        }
    }

    // 4. Frequency cap, automatic sources.
    if let (true, Some(max)) = (ctx.source.is_automatic(), limits.max_automatic_per_hour) {
        let since = ctx.now_mono_ns.saturating_sub(duration_ns(HOUR));
        let count = history.count_since(since);
        if count >= usize::try_from(max).unwrap_or(usize::MAX) {
            return Err(reject(format!(
                "switch_frequency: {count} switches in the last hour; automatic sources are limited to {max}"
            )));
        }
    }

    Ok(notes)
}

/// The gate plug-in. It shares the history and the plan lock with the
/// handshake that records switches.
#[derive(Debug, Clone)]
pub struct FilterSafetyCheck {
    limits: SwitchLimits,
    history: Arc<SwitchHistory>,
    plan: Arc<PlanLock>,
}

impl FilterSafetyCheck {
    pub fn new(limits: SwitchLimits, history: Arc<SwitchHistory>, plan: Arc<PlanLock>) -> Self {
        Self {
            limits,
            history,
            plan,
        }
    }

    fn context<'a>(cmd: &'a GatedCommand, snap: &'a EquipmentSnapshot) -> SwitchContext<'a> {
        SwitchContext {
            source: &cmd.source,
            priority: cmd.priority,
            exposing: &snap.camera.exposing,
            now_mono_ns: snap.taken_mono_ns,
        }
    }
}

impl SafetyCheck for FilterSafetyCheck {
    fn name(&self) -> &'static str {
        CHECK_NAME
    }

    fn applies_to(&self, cmd: &GatedCommand) -> bool {
        matches!(
            cmd.op,
            GatedOp::FilterHandshake { .. } | GatedOp::FilterWheel(_)
        )
    }

    fn check(&self, cmd: &GatedCommand, snap: &EquipmentSnapshot) -> Result<(), SafetyRejection> {
        evaluate(
            &Self::context(cmd, snap),
            &self.limits,
            &self.history,
            &self.plan,
        )
        .map(|_| ())
    }

    fn annotations(&self, cmd: &GatedCommand, snap: &EquipmentSnapshot) -> Vec<String> {
        evaluate(
            &Self::context(cmd, snap),
            &self.limits,
            &self.history,
            &self.plan,
        )
        .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::commands::{FilterTarget, FilterWheelCmd, FocuserCmd};
    use crate::equipment::types::{AutonomyMode, Source};

    const SEC: u64 = 1_000_000_000;

    fn exposing(value: Option<bool>) -> Observed<bool> {
        match value {
            Some(v) => Observed::fresh(v, Source::Derived, Utc::now(), 1),
            None => Observed::not_set(Source::Derived),
        }
    }

    fn command(source: CommandSource, priority: Priority) -> GatedCommand {
        GatedCommand::new(
            GatedOp::FilterHandshake {
                target: FilterTarget::Name("Ha".into()),
            },
            source,
            AutonomyMode::Manual,
            priority,
        )
    }

    fn snapshot(exposing_now: Option<bool>, now_mono_ns: u64) -> EquipmentSnapshot {
        let mut snap = EquipmentSnapshot::empty(Utc::now(), now_mono_ns);
        snap.camera.exposing = exposing(exposing_now);
        snap
    }

    fn policy() -> CommandSource {
        CommandSource::Policy {
            model: "fw2".into(),
            version: "1".into(),
        }
    }

    #[test]
    fn switches_are_refused_while_exposing() {
        let check = FilterSafetyCheck::new(
            SwitchLimits::default(),
            SwitchHistory::new(),
            PlanLock::new(),
        );
        let operator = command(
            CommandSource::Operator { overriding: true },
            Priority::Exposure,
        );
        let err = check
            .check(&operator, &snapshot(Some(true), 10 * SEC))
            .unwrap_err();
        assert_eq!(err.check, CHECK_NAME);
        assert!(err.reason.starts_with("exposing"));
        assert!(check
            .check(&operator, &snapshot(Some(false), 10 * SEC))
            .is_ok());

        // A stale `true` still refuses: fail safe.
        let mut snap = snapshot(Some(true), 10 * SEC);
        snap.camera.exposing = snap.camera.exposing.clone().into_stale();
        assert!(check.check(&operator, &snap).is_err());
    }

    #[test]
    fn an_unknown_exposure_state_refuses_automatic_sources_only() {
        let check = FilterSafetyCheck::new(
            SwitchLimits::default(),
            SwitchHistory::new(),
            PlanLock::new(),
        );
        let snap = snapshot(None, 10 * SEC);
        let auto = command(policy(), Priority::Recommendation);
        assert!(check
            .check(&auto, &snap)
            .unwrap_err()
            .reason
            .starts_with("exposure_state_unknown"));
        let operator = command(
            CommandSource::Operator { overriding: false },
            Priority::Exposure,
        );
        assert!(check.check(&operator, &snap).is_ok());
        assert_eq!(
            check.annotations(&operator, &snap),
            vec!["exposure_state_unknown".to_string()]
        );
    }

    #[test]
    fn a_locked_plan_refuses_recommendations_but_not_operators() {
        let plan = PlanLock::new();
        let check = FilterSafetyCheck::new(
            SwitchLimits::default(),
            SwitchHistory::new(),
            Arc::clone(&plan),
        );
        let snap = snapshot(Some(false), 10 * SEC);
        let recommendation = command(policy(), Priority::Recommendation);
        assert!(check.check(&recommendation, &snap).is_ok());
        plan.lock(LockedPlan {
            plan_id: "fw3-night-1".into(),
            locked_at: Utc::now(),
            reason: None,
        });
        assert!(check
            .check(&recommendation, &snap)
            .unwrap_err()
            .reason
            .contains("fw3-night-1"));
        let trigger = command(
            CommandSource::Trigger {
                rule: "af_filter".into(),
            },
            Priority::Autofocus,
        );
        assert!(check.check(&trigger, &snap).is_ok());
        let operator = command(
            CommandSource::Operator { overriding: false },
            Priority::Exposure,
        );
        assert!(check.check(&operator, &snap).is_ok());
        assert!(plan.unlock().is_some());
        assert!(check.check(&recommendation, &snap).is_ok());
    }

    #[test]
    fn interval_applies_to_everyone_and_frequency_to_automatic_sources() {
        let history = SwitchHistory::new();
        let limits = SwitchLimits {
            min_interval: Duration::from_secs(30),
            max_automatic_per_hour: Some(3),
        };
        let check = FilterSafetyCheck::new(limits, Arc::clone(&history), PlanLock::new());
        let operator = command(
            CommandSource::Operator { overriding: false },
            Priority::Exposure,
        );
        let auto = command(
            CommandSource::Trigger {
                rule: "sequence".into(),
            },
            Priority::Exposure,
        );
        history.record(100 * SEC);
        let err = check
            .check(&operator, &snapshot(Some(false), 110 * SEC))
            .unwrap_err();
        assert!(err.reason.starts_with("min_switch_interval"));
        assert!(check
            .check(&operator, &snapshot(Some(false), 130 * SEC))
            .is_ok());

        history.record(200 * SEC);
        history.record(300 * SEC);
        let err = check
            .check(&auto, &snapshot(Some(false), 400 * SEC))
            .unwrap_err();
        assert!(err.reason.starts_with("switch_frequency"), "{err}");
        assert!(
            check
                .check(&operator, &snapshot(Some(false), 400 * SEC))
                .is_ok(),
            "operators are not frequency-limited"
        );
        // An hour after the first switch it drops out of the window.
        assert!(check
            .check(&auto, &snapshot(Some(false), 100 * SEC + 3601 * SEC))
            .is_ok());
        assert_eq!(history.count_since(0), 3);
        history.record(5000 * SEC);
        assert_eq!(
            history.count_since(0),
            1,
            "entries older than an hour are pruned"
        );
    }

    #[test]
    fn the_check_applies_to_wheel_commands_only() {
        let check = FilterSafetyCheck::new(
            SwitchLimits::default(),
            SwitchHistory::new(),
            PlanLock::new(),
        );
        let source = CommandSource::Operator { overriding: false };
        let raw = GatedCommand::new(
            GatedOp::FilterWheel(FilterWheelCmd::SetPosition(1)),
            source.clone(),
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        assert!(check.applies_to(&raw));
        assert!(check.applies_to(&command(source.clone(), Priority::Exposure)));
        let focuser = GatedCommand::new(
            GatedOp::Focuser(FocuserCmd::MoveBy(10)),
            source,
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        assert!(!check.applies_to(&focuser));
        assert_eq!(check.name(), CHECK_NAME);
    }
}

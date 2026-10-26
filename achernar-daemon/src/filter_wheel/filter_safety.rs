//! Filter wheel safety — Section 8.2.5.
//!
//! One rule set, shared with the gate plug-in
//! (`equipment::filter::check::FilterSafetyCheck`): no switching during an
//! exposure (fail-safe when the exposure state is unknown), the minimum
//! interval, the switch-frequency limit, and a locked plan that a
//! recommendation cannot override. Minimum settling, the verified wheel
//! position and focus coordination are enforced by the handshake
//! (`equipment::filter::change`).
//!
//! The former `block_during_exposure` switch is gone: the exposure interlock
//! cannot be configured off (v6 §8.2.5).

use std::sync::Arc;

use crate::equipment::commands::SafetyRejection;
use crate::equipment::filter::change::FilterChanger;
use crate::equipment::filter::check::{
    evaluate, PlanLock, SwitchContext, SwitchHistory, SwitchLimits,
};
use crate::equipment::types::{CommandSource, Observed, Priority};

/// The filter rules for callers outside the gate, for example a sequencer
/// checking a step before proposing it.
#[derive(Debug, Clone)]
pub struct FilterWheelSafetyGuard {
    limits: SwitchLimits,
    history: Arc<SwitchHistory>,
    plan: Arc<PlanLock>,
}

impl FilterWheelSafetyGuard {
    pub fn new(limits: SwitchLimits, history: Arc<SwitchHistory>, plan: Arc<PlanLock>) -> Self {
        Self {
            limits,
            history,
            plan,
        }
    }

    /// A guard that shares the wheel's switch history and plan lock.
    pub fn for_changer(changer: &FilterChanger) -> Self {
        Self::new(
            SwitchLimits::from_settings(changer.settings()),
            changer.history(),
            changer.plan(),
        )
    }

    /// `Ok` carries audit notes (for example `exposure_state_unknown` for an
    /// operator change).
    pub fn validate_switch(
        &self,
        source: &CommandSource,
        priority: Priority,
        exposing: &Observed<bool>,
        now_mono_ns: u64,
    ) -> Result<Vec<String>, SafetyRejection> {
        evaluate(
            &SwitchContext {
                source,
                priority,
                exposing,
                now_mono_ns,
            },
            &self.limits,
            &self.history,
            &self.plan,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::types::Source;
    use chrono::Utc;

    #[test]
    fn the_guard_applies_the_gate_rules() {
        let guard = FilterWheelSafetyGuard::new(
            SwitchLimits::default(),
            SwitchHistory::new(),
            PlanLock::new(),
        );
        let operator = CommandSource::Operator { overriding: false };
        let exposing = Observed::fresh(true, Source::Derived, Utc::now(), 1);
        assert!(guard
            .validate_switch(&operator, Priority::Exposure, &exposing, 5)
            .is_err());
        let idle = Observed::fresh(false, Source::Derived, Utc::now(), 1);
        assert_eq!(
            guard.validate_switch(&operator, Priority::Exposure, &idle, 5),
            Ok(Vec::new())
        );
        let unknown: Observed<bool> = Observed::not_set(Source::Derived);
        let trigger = CommandSource::Trigger {
            rule: "sequence".into(),
        };
        assert!(guard
            .validate_switch(&trigger, Priority::Exposure, &unknown, 5)
            .is_err());
    }
}

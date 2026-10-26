//! Temperature compensation around moves (spec §3.3 F-d).
//!
//! A focuser with temperature compensation (TempComp) moves itself as the
//! temperature changes. During a commanded move that would fight the move,
//! so with `[focuser] tempcomp_during_moves = "off"` (the default) TempComp
//! is switched off before the first leg and switched back on after the move
//! has settled and been verified, on every path: success, failure, timeout,
//! halt and preemption. `"leave"` keeps it as it is. Autofocus always runs
//! with TempComp off, whatever the setting ([`TempCompGuard::for_autofocus`]).
//!
//! The state before and after is recorded with every move
//! (`focuser_moves.tempcomp_before` / `tempcomp_after`), and each change Achernar
//! makes emits `tempcomp_changed`.

use crate::config::TempCompDuringMoves;
use crate::equipment::traits::FocuserDriver;
use crate::equipment::types::DeviceResult;

/// A TempComp write Achernar made, for the `tempcomp_changed` event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TempCompChange {
    pub from: Option<bool>,
    pub to: bool,
    pub reason: &'static str,
}

/// Turns TempComp off for one move and restores it exactly once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TempCompGuard {
    policy: TempCompDuringMoves,
    force_off: bool,
    /// TempComp as read before the move; `None` when unknown or unsupported.
    before: Option<bool>,
    /// Achernar switched it off and owes the restore.
    owed_restore: bool,
    restored: bool,
    restore_failed: bool,
}

impl TempCompGuard {
    pub fn new(policy: TempCompDuringMoves) -> Self {
        Self {
            policy,
            force_off: false,
            before: None,
            owed_restore: false,
            restored: false,
            restore_failed: false,
        }
    }

    /// Autofocus always runs with TempComp off (F-d), even under `"leave"`.
    pub fn for_autofocus() -> Self {
        Self {
            force_off: true,
            ..Self::new(TempCompDuringMoves::Off)
        }
    }

    /// Before the first leg. `available` is `TempCompAvailable`; `current`
    /// is the latest TempComp reading. Returns the change made, if any.
    pub fn before_move(
        &mut self,
        drv: &mut dyn FocuserDriver,
        available: bool,
        current: Option<bool>,
    ) -> DeviceResult<Option<TempCompChange>> {
        self.before = if available { current } else { None };
        let wants_off = self.force_off || self.policy == TempCompDuringMoves::Off;
        if !(available && wants_off && current == Some(true)) {
            return Ok(None);
        }
        drv.set_temp_comp(false)?;
        self.owed_restore = true;
        Ok(Some(TempCompChange {
            from: Some(true),
            to: false,
            reason: if self.force_off {
                "autofocus"
            } else {
                "focuser_move"
            },
        }))
    }

    /// Restore TempComp after the move. Idempotent: the second call does
    /// nothing. A failed restore is reported and not retried.
    pub fn restore(&mut self, drv: &mut dyn FocuserDriver) -> DeviceResult<Option<TempCompChange>> {
        if !self.owed_restore || self.restored {
            return Ok(None);
        }
        self.restored = true;
        if let Err(err) = drv.set_temp_comp(true) {
            self.restore_failed = true;
            return Err(err);
        }
        Ok(Some(TempCompChange {
            from: Some(false),
            to: true,
            reason: "restore_after_move",
        }))
    }

    /// TempComp before the move (`focuser_moves.tempcomp_before`).
    pub fn before(&self) -> Option<bool> {
        self.before
    }

    /// True while Achernar holds TempComp off.
    pub fn holding_off(&self) -> bool {
        self.owed_restore && !self.restored
    }

    /// TempComp after the move (`focuser_moves.tempcomp_after`). When
    /// Achernar switched it, what it last wrote (the latest poll predates the
    /// restore); otherwise the latest reading.
    pub fn after(&self, observed: Option<bool>) -> Option<bool> {
        if self.owed_restore {
            return Some(self.restored && !self.restore_failed);
        }
        observed.or(self.before)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::sim::focuser::{SimCall, SimFocuser, SimFocuserConfig};
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::traits::{Device, PollPlan};
    use std::sync::Arc;
    use std::time::Duration;

    fn sim(temp_comp: bool, available: bool) -> (SimFocuser, Arc<ManualClock>) {
        let clock = Arc::new(ManualClock::new());
        let mut cfg = SimFocuserConfig::absolute();
        cfg.temp_comp_available = available;
        cfg.temp_comp = temp_comp;
        let mut drv = SimFocuser::new(cfg, clock.clone());
        drv.connect(Duration::from_secs(1)).unwrap();
        drv.poll(&PollPlan::default()).unwrap();
        (drv, clock)
    }

    #[test]
    fn tempcomp_is_turned_off_for_the_move_and_restored_once() {
        let (mut drv, _clock) = sim(true, true);
        let control = drv.control();
        let mut guard = TempCompGuard::new(TempCompDuringMoves::Off);
        let change = guard.before_move(&mut drv, true, Some(true)).unwrap();
        assert_eq!(
            change,
            Some(TempCompChange {
                from: Some(true),
                to: false,
                reason: "focuser_move"
            })
        );
        assert!(guard.holding_off());
        assert!(!control.temp_comp());
        let restored = guard.restore(&mut drv).unwrap().unwrap();
        assert!(restored.to);
        assert!(control.temp_comp());
        assert_eq!(guard.restore(&mut drv).unwrap(), None, "idempotent");
        assert_eq!(guard.before(), Some(true));
        assert_eq!(guard.after(None), Some(true));
        let writes: Vec<_> = control
            .calls()
            .into_iter()
            .filter(|c| matches!(c, SimCall::SetTempComp(_)))
            .collect();
        assert_eq!(
            writes,
            vec![SimCall::SetTempComp(false), SimCall::SetTempComp(true)]
        );
    }

    #[test]
    fn leave_policy_and_off_tempcomp_write_nothing() {
        let (mut drv, _clock) = sim(true, true);
        let control = drv.control();
        let mut leave = TempCompGuard::new(TempCompDuringMoves::Leave);
        assert_eq!(leave.before_move(&mut drv, true, Some(true)).unwrap(), None);
        assert_eq!(leave.restore(&mut drv).unwrap(), None);
        assert_eq!(leave.after(None), Some(true));
        let mut already_off = TempCompGuard::new(TempCompDuringMoves::Off);
        assert_eq!(
            already_off
                .before_move(&mut drv, true, Some(false))
                .unwrap(),
            None
        );
        let mut unavailable = TempCompGuard::new(TempCompDuringMoves::Off);
        assert_eq!(
            unavailable
                .before_move(&mut drv, false, Some(true))
                .unwrap(),
            None
        );
        assert_eq!(unavailable.before(), None);
        assert!(control
            .calls()
            .iter()
            .all(|c| !matches!(c, SimCall::SetTempComp(_))));
    }

    #[test]
    fn autofocus_forces_tempcomp_off_even_under_leave() {
        let (mut drv, _clock) = sim(true, true);
        let mut guard = TempCompGuard::for_autofocus();
        let change = guard
            .before_move(&mut drv, true, Some(true))
            .unwrap()
            .unwrap();
        assert_eq!(change.reason, "autofocus");
        assert!(!drv.control().temp_comp());
        guard.restore(&mut drv).unwrap();
        assert!(drv.control().temp_comp());
    }
}

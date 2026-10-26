//! Temperature compensation around moves (spec §3.3 F-d).
//!
//! A focuser with temperature compensation (TempComp) moves itself as the
//! temperature changes. During a commanded move that would fight the move,
//! so with `[focuser] tempcomp_during_moves = "off"` (the default) TempComp
//! is switched off before the first leg and switched back on after the move
//! has settled and been verified, on every path: success, failure, timeout,
//! halt and preemption. `"leave"` keeps it as it is.
//!
//! Autofocus always runs with TempComp off, whatever the setting, and for the
//! **whole run**, not move by move: the exposures between the moves are the
//! ones that must not see the focuser move on its own. A [`TempCompRunHold`]
//! owns the restore for the run. Its first switch-off (at
//! `FocuserSession::begin_autofocus_run`, or the run's first move) is handed
//! to it by a [`TempCompGuard::for_autofocus_run`] guard, later moves of the
//! run restore nothing, and the run end restores TempComp exactly once.
//!
//! A switch-off write that fails in transit (timeout, transport, link
//! down) may still have reached the device (C-6), so it is treated as
//! possibly applied: the restore is still owed, and its `tempcomp_changed`
//! event reports `from: None`.
//!
//! The state before and after is recorded with every move
//! (`focuser_moves.tempcomp_before` / `tempcomp_after`), and each change Achernar
//! makes emits `tempcomp_changed`.

use crate::config::TempCompDuringMoves;
use crate::equipment::traits::FocuserDriver;
use crate::equipment::types::{DeviceError, DeviceResult, ErrorClass};

/// A TempComp write Achernar made, for the `tempcomp_changed` event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TempCompChange {
    /// `None` when the state before is unknown (an unconfirmed switch-off).
    pub from: Option<bool>,
    pub to: bool,
    pub reason: &'static str,
}

/// A failed write that may still have reached the device (C-6).
fn may_have_applied(err: &DeviceError) -> bool {
    matches!(
        err.class,
        ErrorClass::Timeout
            | ErrorClass::Transport
            | ErrorClass::Disconnected
            | ErrorClass::OutcomeUnknown
    )
}

/// Who restores TempComp after a switch-off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RestoreBy {
    /// The move, once it has finished (ordinary moves).
    Move,
    /// The autofocus run, at its end.
    Run,
}

/// Turns TempComp off for one move and restores it exactly once, or hands
/// the restore to the autofocus run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TempCompGuard {
    policy: TempCompDuringMoves,
    force_off: bool,
    restore_by: RestoreBy,
    /// TempComp as read before the move; `None` when unknown or unsupported.
    before: Option<bool>,
    /// Achernar switched it off (or may have) and owes the restore.
    owed_restore: bool,
    /// The switch-off failed in transit; whether it applied is unknown.
    unconfirmed: bool,
    /// The restore went to the autofocus run.
    handed_to_run: bool,
    restored: bool,
    restore_failed: bool,
}

impl TempCompGuard {
    pub fn new(policy: TempCompDuringMoves) -> Self {
        Self {
            policy,
            force_off: false,
            restore_by: RestoreBy::Move,
            before: None,
            owed_restore: false,
            unconfirmed: false,
            handed_to_run: false,
            restored: false,
            restore_failed: false,
        }
    }

    /// A move (or the start) of an autofocus run: TempComp is forced off
    /// even under `"leave"` (F-d), and the restore belongs to the run
    /// ([`TempCompGuard::take_run_restore`]), so [`TempCompGuard::restore`]
    /// does nothing.
    pub fn for_autofocus_run() -> Self {
        Self {
            force_off: true,
            restore_by: RestoreBy::Run,
            ..Self::new(TempCompDuringMoves::Off)
        }
    }

    /// An ordinary move that follows an autofocus run which was never ended
    /// takes over the run's restore: TempComp is switched back on once this
    /// move finishes.
    pub fn inherit_restore(&mut self, unconfirmed: bool) {
        if self.restore_by == RestoreBy::Move {
            self.owed_restore = true;
            self.unconfirmed |= unconfirmed;
        }
    }

    /// Before the first leg. `available` is `TempCompAvailable`; `current`
    /// is the latest TempComp reading. Returns the change made, if any.
    ///
    /// A write that fails in transit still leaves the restore owed (it may
    /// have applied); the error is returned so the move fails.
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
        if let Err(err) = drv.set_temp_comp(false) {
            if may_have_applied(&err) {
                self.owed_restore = true;
                self.unconfirmed = true;
                tracing::warn!(
                    error = %err,
                    "the TempComp=False write failed in transit and may have applied; \
                     TempComp will be restored"
                );
            }
            return Err(err);
        }
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

    /// An autofocus guard's switch-off, handed to the run: `Some(unconfirmed)`
    /// once when this guard switched TempComp off (or may have).
    pub fn take_run_restore(&mut self) -> Option<bool> {
        if self.restore_by != RestoreBy::Run || !self.owed_restore || self.handed_to_run {
            return None;
        }
        self.owed_restore = false;
        self.handed_to_run = true;
        Some(self.unconfirmed)
    }

    /// Restore TempComp after the move. Idempotent: the second call does
    /// nothing. A failed restore is reported and not retried. An autofocus
    /// guard never restores: the run does.
    pub fn restore(&mut self, drv: &mut dyn FocuserDriver) -> DeviceResult<Option<TempCompChange>> {
        if self.restore_by == RestoreBy::Run || !self.owed_restore || self.restored {
            return Ok(None);
        }
        self.restored = true;
        if let Err(err) = drv.set_temp_comp(true) {
            self.restore_failed = true;
            return Err(err);
        }
        Ok(Some(TempCompChange {
            from: if self.unconfirmed { None } else { Some(false) },
            to: true,
            reason: "restore_after_move",
        }))
    }

    /// TempComp before the move (`focuser_moves.tempcomp_before`).
    pub fn before(&self) -> Option<bool> {
        self.before
    }

    /// True while Achernar holds TempComp off for this move.
    pub fn holding_off(&self) -> bool {
        self.owed_restore && !self.restored
    }

    /// TempComp after the move (`focuser_moves.tempcomp_after`). When
    /// Achernar switched it, what it last wrote (the latest poll predates the
    /// write); otherwise the latest reading. `None` when a write's outcome is
    /// unknown.
    pub fn after(&self, observed: Option<bool>) -> Option<bool> {
        if self.handed_to_run {
            // Still off: the autofocus run holds it until the run ends.
            return (!self.unconfirmed).then_some(false);
        }
        if self.owed_restore {
            if self.restored && !self.restore_failed {
                return Some(true);
            }
            return (!self.unconfirmed).then_some(false);
        }
        observed.or(self.before)
    }
}

/// TempComp held off for one autofocus run (F-d). Lives in the focuser
/// session from the run's start to its end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TempCompRunHold {
    run_id: String,
    /// Achernar switched TempComp off for the run (or may have) and owes one
    /// restore at the run end.
    owed_restore: bool,
    unconfirmed: bool,
}

impl TempCompRunHold {
    pub fn new(run_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            owed_restore: false,
            unconfirmed: false,
        }
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// A new run took over a run that was never ended: the restore it owes
    /// carries over.
    pub fn rekey(&mut self, run_id: impl Into<String>) {
        self.run_id = run_id.into();
    }

    pub fn owes_restore(&self) -> bool {
        self.owed_restore
    }

    /// The switch-off that the restore will undo failed in transit.
    pub fn unconfirmed(&self) -> bool {
        self.unconfirmed
    }

    /// Take over an autofocus guard's switch-off. The latest write decides
    /// whether the state is known.
    pub fn absorb(&mut self, guard: &mut TempCompGuard) {
        if let Some(unconfirmed) = guard.take_run_restore() {
            self.owed_restore = true;
            self.unconfirmed = unconfirmed;
        }
    }

    /// The operator set TempComp explicitly during the run: the run no
    /// longer restores it (a later forced switch-off owes one again).
    pub fn forget_restore(&mut self) {
        self.owed_restore = false;
        self.unconfirmed = false;
    }

    /// The run end: restore TempComp once if the run switched it off. A
    /// failed restore is reported and not retried.
    pub fn restore(self, drv: &mut dyn FocuserDriver) -> DeviceResult<Option<TempCompChange>> {
        if !self.owed_restore {
            return Ok(None);
        }
        drv.set_temp_comp(true)?;
        Ok(Some(TempCompChange {
            from: if self.unconfirmed { None } else { Some(false) },
            to: true,
            reason: "restore_after_autofocus",
        }))
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

    fn tempcomp_writes(drv: &SimFocuser) -> Vec<SimCall> {
        drv.control()
            .calls()
            .into_iter()
            .filter(|c| matches!(c, SimCall::SetTempComp(_)))
            .collect()
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
        assert_eq!(
            tempcomp_writes(&drv),
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
    fn autofocus_guards_hand_the_restore_to_the_run() {
        let (mut drv, _clock) = sim(true, true);
        let mut hold = TempCompRunHold::new("run-1");
        // The run's first move switches it off (even under "leave") ...
        let mut first = TempCompGuard::for_autofocus_run();
        let change = first
            .before_move(&mut drv, true, Some(true))
            .unwrap()
            .unwrap();
        assert_eq!(change.reason, "autofocus");
        hold.absorb(&mut first);
        assert!(hold.owes_restore());
        // ... and finishing the move does not switch it back on.
        assert_eq!(first.restore(&mut drv).unwrap(), None);
        assert!(!drv.control().temp_comp());
        assert_eq!(
            (first.before(), first.after(Some(true))),
            (Some(true), Some(false))
        );
        // A later move of the run finds it off and owes nothing itself.
        let mut second = TempCompGuard::for_autofocus_run();
        assert_eq!(
            second.before_move(&mut drv, true, Some(false)).unwrap(),
            None
        );
        hold.absorb(&mut second);
        assert_eq!(second.restore(&mut drv).unwrap(), None);
        assert!(!drv.control().temp_comp());
        // The run end restores it exactly once.
        let restored = hold.clone().restore(&mut drv).unwrap().unwrap();
        assert_eq!(
            restored,
            TempCompChange {
                from: Some(false),
                to: true,
                reason: "restore_after_autofocus"
            }
        );
        assert!(drv.control().temp_comp());
        assert_eq!(
            tempcomp_writes(&drv),
            vec![SimCall::SetTempComp(false), SimCall::SetTempComp(true)]
        );
        // An operator write during a run cancels the run's restore.
        hold.forget_restore();
        assert_eq!(hold.restore(&mut drv).unwrap(), None);
    }

    #[test]
    fn an_ordinary_move_after_an_unended_run_restores_tempcomp() {
        let (mut drv, _clock) = sim(false, true);
        let mut guard = TempCompGuard::new(TempCompDuringMoves::Leave);
        guard.inherit_restore(false);
        assert_eq!(
            guard.before_move(&mut drv, true, Some(false)).unwrap(),
            None
        );
        let change = guard.restore(&mut drv).unwrap().unwrap();
        assert_eq!(change.reason, "restore_after_move");
        assert!(drv.control().temp_comp());
        assert_eq!(guard.after(Some(false)), Some(true));
        // An autofocus guard never inherits: the run keeps the restore.
        let mut af = TempCompGuard::for_autofocus_run();
        af.inherit_restore(false);
        assert!(!af.holding_off());
    }

    #[test]
    fn a_switch_off_lost_in_transit_still_owes_the_restore() {
        let (mut drv, _clock) = sim(true, true);
        drv.control().set_link_down(true);
        let mut guard = TempCompGuard::new(TempCompDuringMoves::Off);
        let err = guard.before_move(&mut drv, true, Some(true)).unwrap_err();
        assert_eq!(err.class, ErrorClass::Disconnected);
        assert!(guard.holding_off(), "possibly applied");
        drv.control().set_link_down(false);
        let change = guard.restore(&mut drv).unwrap().unwrap();
        assert_eq!(change.from, None, "the state before the restore is unknown");
        assert!(drv.control().temp_comp());

        // A refusal from the device is a plain failure: nothing is owed.
        let mut refused = TempCompGuard::new(TempCompDuringMoves::Off);
        let (mut unavailable, _clock) = sim(true, false);
        assert!(refused
            .before_move(&mut unavailable, true, Some(true))
            .is_err());
        assert!(!refused.holding_off());
        assert_eq!(refused.restore(&mut unavailable).unwrap(), None);

        // The same holds for an autofocus run's switch-off.
        let (mut drv, _clock) = sim(true, true);
        drv.control().set_link_down(true);
        let mut af = TempCompGuard::for_autofocus_run();
        assert!(af.before_move(&mut drv, true, Some(true)).is_err());
        let mut hold = TempCompRunHold::new("run-2");
        hold.absorb(&mut af);
        assert!(hold.owes_restore() && hold.unconfirmed());
        drv.control().set_link_down(false);
        assert_eq!(hold.restore(&mut drv).unwrap().unwrap().from, None);
    }
}

//! Park, unpark, set-park and home (§3.2 M-f). Owner: WP1A-4.
//!
//! - **Park** requires CanPark. Completion is AtPark true, and the park is
//!   **verified**: if motion stops and AtPark still reads false after a
//!   grace period, the park fails with `park_failed`. NINA treats a normal
//!   return as success; Achernar does not. A mount whose AtPark cannot be
//!   read gives `Unknown`.
//! - **Unpark** requires CanUnpark; completion is AtPark false. Tracking is
//!   left as the driver reports it (NINA forces it off).
//! - **FindHome** requires CanFindHome and completes on AtHome.
//! - **SetPark** is allowed only when not parked.
//!
//! Stopping guiding first (D20) is `coord.rs`'s job. The Tier 2 fallback
//! park for mounts without CanPark is WP2-6's `flip.rs`.

use std::sync::Arc;
use std::time::Duration;

use crate::equipment::events::{EventKind, ParkMethod};
use crate::equipment::ops::{OpKind, OpProgress, Operation};
use crate::equipment::snapshot::MountState;
use crate::equipment::traits::Clock;
use crate::equipment::types::{DeviceError, DeviceResult, ErrorClass, ObsState, OpOutcome};

use super::slew::abort_slew;
use super::{
    duration_ns, is_parked, parked_error, CommandMeta, DynMount, MountShared, ACTIVE_POLL,
    SETTER_DEADLINE,
};

/// How long AtPark (or AtHome) may lag behind the end of motion before the
/// operation is judged not verified.
pub const VERIFY_GRACE: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParkDirection {
    Park,
    Unpark,
}

/// Park or unpark, verified through AtPark.
pub struct ParkOp {
    shared: Arc<MountShared>,
    meta: CommandMeta,
    direction: ParkDirection,
    stopped_since_mono_ns: Option<u64>,
    finished: Option<OpOutcome>,
}

impl ParkOp {
    pub fn park(shared: Arc<MountShared>, meta: CommandMeta) -> Self {
        Self::new(shared, meta, ParkDirection::Park)
    }

    pub fn unpark(shared: Arc<MountShared>, meta: CommandMeta) -> Self {
        Self::new(shared, meta, ParkDirection::Unpark)
    }

    fn new(shared: Arc<MountShared>, meta: CommandMeta, direction: ParkDirection) -> Self {
        Self {
            shared,
            meta,
            direction,
            stopped_since_mono_ns: None,
            finished: None,
        }
    }

    fn emit(&self, kind: EventKind) {
        self.shared.emit(kind, Some(self.meta.command_id));
    }

    fn fail(&mut self, err: DeviceError) -> DeviceError {
        if self.direction == ParkDirection::Park {
            self.emit(EventKind::ParkFailed {
                class: err.class,
                reason: err.message.clone(),
            });
        }
        err
    }

    fn succeed(&mut self) -> OpProgress {
        match self.direction {
            ParkDirection::Park => {
                self.emit(EventKind::Parked {
                    method: ParkMethod::Driver,
                });
                self.shared.set_announced_park(true);
            }
            ParkDirection::Unpark => {
                self.emit(EventKind::Unparked);
                self.shared.set_announced_park(false);
            }
        }
        self.finished = Some(OpOutcome::Succeeded);
        OpProgress::Done(OpOutcome::Succeeded)
    }

    fn evaluate(&mut self, state: &MountState, now_mono_ns: u64) -> OpProgress {
        let want = self.direction == ParkDirection::Park;
        let at_park = &state.at_park;
        let slewing = state.slewing.fresh_value().copied();
        if slewing == Some(true) {
            self.stopped_since_mono_ns = None;
            return OpProgress::Continue {
                next_poll: ACTIVE_POLL,
            };
        }
        if at_park.fresh_value() == Some(&want) {
            return self.succeed();
        }
        // Motion has stopped (or Slewing is unreadable): allow AtPark a
        // grace period, then judge.
        let stopped = *self.stopped_since_mono_ns.get_or_insert(now_mono_ns);
        if now_mono_ns.saturating_sub(stopped) < duration_ns(VERIFY_GRACE) {
            return OpProgress::Continue {
                next_poll: ACTIVE_POLL,
            };
        }
        let outcome = if at_park.state == ObsState::Unsupported {
            OpOutcome::Unknown(self.fail(DeviceError::outcome_unknown(
                "AtPark is not readable; the park state cannot be verified",
            )))
        } else if at_park.is_fresh() {
            let message = if want {
                "park_not_verified: AtPark is still false after the mount stopped"
            } else {
                "unpark_not_verified: AtPark is still true after unpark"
            };
            OpOutcome::Failed(self.fail(DeviceError::driver(message)))
        } else {
            return OpProgress::Continue {
                next_poll: ACTIVE_POLL,
            };
        };
        self.finished = Some(outcome.clone());
        OpProgress::Done(outcome)
    }
}

impl Operation<DynMount> for ParkOp {
    fn kind(&self) -> OpKind {
        match self.direction {
            ParkDirection::Park => OpKind::Park,
            ParkDirection::Unpark => OpKind::Unpark,
        }
    }

    fn deadline(&self) -> Duration {
        self.shared.settings().slew_timeout
    }

    fn start(&mut self, drv: &mut DynMount, _clock: &dyn Clock) -> DeviceResult<()> {
        let caps = self.shared.capabilities();
        let state = drv.state()?;
        match self.direction {
            ParkDirection::Park => {
                if !caps.can_park {
                    return Err(self.fail(DeviceError::unsupported(
                        "the mount does not report CanPark (a fallback park is Tier 2)",
                    )));
                }
                if is_parked(&state) {
                    return Ok(());
                }
                drv.park().map_err(|err| self.fail(err))
            }
            ParkDirection::Unpark => {
                if !caps.can_unpark {
                    return Err(DeviceError::unsupported(
                        "the mount does not report CanUnpark",
                    ));
                }
                if state.at_park.known() == Some(&false) {
                    return Ok(());
                }
                drv.unpark()
            }
        }
    }

    fn step(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> OpProgress {
        if let Some(outcome) = &self.finished {
            return OpProgress::Done(outcome.clone());
        }
        match drv.state() {
            Ok(state) => self.evaluate(&state, clock.now_mono_ns()),
            Err(_) => OpProgress::Continue {
                next_poll: ACTIVE_POLL,
            },
        }
    }

    fn cancel(&mut self, drv: &mut DynMount) -> DeviceResult<()> {
        if self.direction == ParkDirection::Park && self.finished.is_none() {
            self.emit(EventKind::ParkFailed {
                class: ErrorClass::Cancelled,
                reason: "the park was cancelled or timed out before AtPark read true".into(),
            });
        }
        abort_slew(drv)
    }
}

/// FindHome, completed on AtHome.
pub struct HomeOp {
    shared: Arc<MountShared>,
    meta: CommandMeta,
    stopped_since_mono_ns: Option<u64>,
}

impl HomeOp {
    pub fn new(shared: Arc<MountShared>, meta: CommandMeta) -> Self {
        Self {
            shared,
            meta,
            stopped_since_mono_ns: None,
        }
    }
}

impl Operation<DynMount> for HomeOp {
    fn kind(&self) -> OpKind {
        OpKind::Home
    }

    fn deadline(&self) -> Duration {
        self.shared.settings().slew_timeout
    }

    fn start(&mut self, drv: &mut DynMount, _clock: &dyn Clock) -> DeviceResult<()> {
        if !self.shared.capabilities().can_find_home {
            return Err(DeviceError::unsupported(
                "the mount does not report CanFindHome",
            ));
        }
        if is_parked(&drv.state()?) {
            return Err(parked_error("homing"));
        }
        drv.find_home()
    }

    fn step(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> OpProgress {
        let Ok(state) = drv.state() else {
            return OpProgress::Continue {
                next_poll: ACTIVE_POLL,
            };
        };
        if state.slewing.fresh_value() == Some(&true) {
            self.stopped_since_mono_ns = None;
            return OpProgress::Continue {
                next_poll: ACTIVE_POLL,
            };
        }
        if state.at_home.fresh_value() == Some(&true) {
            self.shared
                .emit(EventKind::Homed, Some(self.meta.command_id));
            return OpProgress::Done(OpOutcome::Succeeded);
        }
        let now = clock.now_mono_ns();
        let stopped = *self.stopped_since_mono_ns.get_or_insert(now);
        if now.saturating_sub(stopped) < duration_ns(VERIFY_GRACE) {
            return OpProgress::Continue {
                next_poll: ACTIVE_POLL,
            };
        }
        if state.at_home.state == ObsState::Unsupported {
            OpProgress::Done(OpOutcome::Unknown(DeviceError::outcome_unknown(
                "AtHome is not readable; the home position cannot be verified",
            )))
        } else {
            OpProgress::Done(OpOutcome::Failed(DeviceError::driver(
                "home_not_verified: AtHome is still false after the mount stopped",
            )))
        }
    }

    fn cancel(&mut self, drv: &mut DynMount) -> DeviceResult<()> {
        abort_slew(drv)
    }
}

/// SetPark: the current position becomes the park position.
pub struct SetParkOp {
    shared: Arc<MountShared>,
    _meta: CommandMeta,
}

impl SetParkOp {
    pub fn new(shared: Arc<MountShared>, meta: CommandMeta) -> Self {
        Self {
            shared,
            _meta: meta,
        }
    }
}

impl Operation<DynMount> for SetParkOp {
    fn kind(&self) -> OpKind {
        OpKind::SetPark
    }

    fn deadline(&self) -> Duration {
        SETTER_DEADLINE
    }

    fn start(&mut self, drv: &mut DynMount, _clock: &dyn Clock) -> DeviceResult<()> {
        if !self.shared.capabilities().can_set_park {
            return Err(DeviceError::unsupported(
                "the mount does not report CanSetPark",
            ));
        }
        if is_parked(&drv.state()?) {
            return Err(parked_error("setting a new park position"));
        }
        drv.set_park()
    }

    fn step(&mut self, _drv: &mut DynMount, _clock: &dyn Clock) -> OpProgress {
        OpProgress::Done(OpOutcome::Succeeded)
    }

    fn cancel(&mut self, _drv: &mut DynMount) -> DeviceResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;
    use crate::equipment::commands::MountCmd;
    use crate::equipment::sim::mount::{SimFaults, SimMountProfile};

    #[test]
    fn park_is_verified_through_at_park() {
        let mut rig = rig(SimMountProfile::esp32_like());
        rig.runtime.take_events();
        assert_eq!(
            rig.run(MountCmd::Park, operator()).unwrap(),
            OpOutcome::Succeeded
        );
        let snapshot = rig.tick(Duration::from_secs(1), None);
        assert_eq!(snapshot.state.at_park.value, Some(true));
        let names = rig.names();
        assert_eq!(
            names.iter().filter(|n| **n == "parked").count(),
            1,
            "{names:?}"
        );
        assert!(!names.contains(&"park_failed"));
        // Parking an already parked mount succeeds without moving it.
        let calls_before = rig.sim.control().calls().len();
        assert_eq!(
            rig.run(MountCmd::Park, operator()).unwrap(),
            OpOutcome::Succeeded
        );
        assert_eq!(rig.sim.control().calls().len(), calls_before);
        // Unpark: AtPark false; tracking left as the driver reports it.
        assert_eq!(
            rig.run(MountCmd::Unpark, operator()).unwrap(),
            OpOutcome::Succeeded
        );
        let snapshot = rig.tick(Duration::from_secs(1), None);
        assert_eq!(snapshot.state.at_park.value, Some(false));
        assert_eq!(
            snapshot.state.tracking.value,
            Some(false),
            "never forced on or off"
        );
        assert!(rig.names().contains(&"unparked"));
    }

    #[test]
    fn a_park_that_never_sets_at_park_fails_verification() {
        let mut rig = rig(SimMountProfile::esp32_like());
        rig.sim.control().set_faults(SimFaults {
            park_never_sets_atpark: true,
            ..SimFaults::default()
        });
        rig.runtime.take_events();
        let outcome = rig.run(MountCmd::Park, operator()).unwrap();
        assert!(
            matches!(&outcome, OpOutcome::Failed(e) if e.message.starts_with("park_not_verified")),
            "{outcome:?}"
        );
        let events = rig.runtime.take_events();
        assert!(events.iter().any(|e| matches!(
            &e.event,
            EventKind::ParkFailed { class: ErrorClass::Driver, reason } if reason.starts_with("park_not_verified")
        )));
        assert!(!events.iter().any(|e| e.name() == "parked"));
    }

    #[test]
    fn park_needs_can_park_and_records_the_refusal() {
        let mut profile = SimMountProfile::esp32_like();
        profile.can_park = false;
        let mut rig = rig(profile);
        rig.runtime.take_events();
        let outcome = rig.run(MountCmd::Park, operator()).unwrap();
        assert!(matches!(outcome, OpOutcome::Failed(e) if e.class == ErrorClass::Unsupported));
        assert!(rig.names().contains(&"park_failed"));
    }

    #[test]
    fn an_unpark_that_does_not_clear_at_park_fails() {
        let mut profile = SimMountProfile::esp32_like();
        profile.start_parked = true;
        let mut rig = rig(profile);
        rig.sim.control().set_faults(SimFaults {
            unpark_never_clears_atpark: true,
            ..SimFaults::default()
        });
        let outcome = rig.run(MountCmd::Unpark, operator()).unwrap();
        assert!(
            matches!(&outcome, OpOutcome::Failed(e) if e.message.starts_with("unpark_not_verified"))
        );
    }

    #[test]
    fn a_cancelled_park_is_recorded_and_halted() {
        let mut rig = rig(SimMountProfile::esp32_like());
        let meta = operator();
        let mut op =
            super::super::build_operation(&rig.runtime.shared(), MountCmd::Park, meta.clone())
                .unwrap();
        op.start(&mut rig.sim, rig.clock.as_ref()).unwrap();
        assert_eq!(advance(&mut rig, &mut op, &meta, 2), None);
        rig.runtime.take_events();
        op.cancel(&mut rig.sim).unwrap();
        assert!(!rig.sim.control().slewing());
        let events = rig.runtime.take_events();
        assert!(events.iter().any(|e| matches!(
            e.event,
            EventKind::ParkFailed {
                class: ErrorClass::Cancelled,
                ..
            }
        )));
    }

    #[test]
    fn home_completes_on_at_home_and_needs_can_find_home() {
        let mut rig = rig(SimMountProfile::german_equatorial());
        rig.runtime.take_events();
        assert_eq!(
            rig.run(MountCmd::FindHome, operator()).unwrap(),
            OpOutcome::Succeeded
        );
        assert!(rig.names().contains(&"homed"));
        // The ESP32 reports CanFindHome = false.
        let mut rig = super::super::test_support::rig(SimMountProfile::esp32_like());
        let outcome = rig.run(MountCmd::FindHome, operator()).unwrap();
        assert!(matches!(outcome, OpOutcome::Failed(e) if e.class == ErrorClass::Unsupported));
        // AtHome unreadable: the outcome is unknown, never assumed.
        let mut profile = SimMountProfile::german_equatorial();
        profile.reports_at_home = false;
        let mut rig = super::super::test_support::rig(profile);
        let outcome = rig.run(MountCmd::FindHome, operator()).unwrap();
        assert!(matches!(outcome, OpOutcome::Unknown(_)), "{outcome:?}");
    }

    #[test]
    fn set_park_is_refused_while_parked() {
        let mut rig = rig(SimMountProfile::german_equatorial());
        assert_eq!(
            rig.run(MountCmd::SetPark, operator()).unwrap(),
            OpOutcome::Succeeded
        );
        assert_eq!(
            rig.run(MountCmd::Park, operator()).unwrap(),
            OpOutcome::Succeeded
        );
        let outcome = rig.run(MountCmd::SetPark, operator()).unwrap();
        assert!(
            matches!(outcome, OpOutcome::Failed(e) if e.class == ErrorClass::InvalidWhileParked)
        );
    }
}

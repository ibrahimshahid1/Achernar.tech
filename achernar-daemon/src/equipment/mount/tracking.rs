//! Tracking on/off and TrackingRate (§3.2 M-e). Owner: WP1A-4.
//!
//! The rate is chosen from the advertised `trackingrates` with the correct
//! ASCOM mapping (0 Sidereal, 1 Lunar, 2 Solar, 3 King; the OLS wrapper set
//! Lunar for Solar) and is written only on request. Every setter verifies
//! its read-back (C-2) and emits `tracking_changed{on, rate, source}`.

use std::sync::Arc;
use std::time::Duration;

use crate::equipment::events::EventKind;
use crate::equipment::ops::{OpKind, OpProgress, Operation};
use crate::equipment::traits::Clock;
use crate::equipment::types::{DeviceError, DeviceResult, DriveRate, ObsState, OpOutcome};

use super::{
    duration_ns, is_parked, parked_error, CommandMeta, DynMount, MountShared, ACTIVE_POLL,
    SETTER_DEADLINE, VERIFY_TIMEOUT,
};

/// The requested rate, if the mount advertises it.
pub fn select_rate(advertised: &[DriveRate], requested: DriveRate) -> DeviceResult<DriveRate> {
    if advertised.is_empty() {
        return Err(DeviceError::unsupported(
            "the mount advertises no TrackingRates",
        ));
    }
    if advertised.contains(&requested) {
        Ok(requested)
    } else {
        Err(DeviceError::invalid_value(format!(
            "{requested:?} is not in the mount's TrackingRates {advertised:?}"
        )))
    }
}

/// Tracking on or off.
pub struct SetTrackingOp {
    shared: Arc<MountShared>,
    meta: CommandMeta,
    on: bool,
    sent_mono_ns: u64,
}

impl SetTrackingOp {
    pub fn new(shared: Arc<MountShared>, meta: CommandMeta, on: bool) -> Self {
        Self {
            shared,
            meta,
            on,
            sent_mono_ns: 0,
        }
    }

    fn announce(&self, rate: Option<DriveRate>, reason: Option<&str>) {
        self.shared.emit(
            EventKind::TrackingChanged {
                on: Some(self.on),
                rate,
                source: Some(self.meta.source.clone()),
                reason: reason.map(str::to_string),
            },
            Some(self.meta.command_id),
        );
        self.shared.set_announced_tracking(self.on);
    }
}

impl Operation<DynMount> for SetTrackingOp {
    fn kind(&self) -> OpKind {
        OpKind::SetTracking
    }

    fn deadline(&self) -> Duration {
        SETTER_DEADLINE
    }

    fn start(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> DeviceResult<()> {
        if !self.shared.capabilities().can_set_tracking {
            return Err(DeviceError::unsupported(
                "the mount does not report CanSetTracking",
            ));
        }
        if self.on && is_parked(&drv.state()?) {
            return Err(parked_error("turning tracking on"));
        }
        drv.set_tracking(self.on)?;
        self.sent_mono_ns = clock.now_mono_ns();
        Ok(())
    }

    fn step(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> OpProgress {
        let Ok(state) = drv.state() else {
            return OpProgress::Continue {
                next_poll: ACTIVE_POLL,
            };
        };
        let rate = state.tracking_rate.value;
        if state.tracking.state == ObsState::Unsupported {
            self.announce(rate, Some("unverified"));
            return OpProgress::Done(OpOutcome::Succeeded);
        }
        if state.tracking.fresh_value() == Some(&self.on) {
            self.announce(rate, None);
            return OpProgress::Done(OpOutcome::Succeeded);
        }
        if clock.now_mono_ns().saturating_sub(self.sent_mono_ns) >= duration_ns(VERIFY_TIMEOUT) {
            return OpProgress::Done(OpOutcome::Failed(DeviceError::driver(format!(
                "Tracking still reads {:?} after it was set to {}",
                state.tracking.value, self.on
            ))));
        }
        OpProgress::Continue {
            next_poll: ACTIVE_POLL,
        }
    }

    fn cancel(&mut self, _drv: &mut DynMount) -> DeviceResult<()> {
        Ok(())
    }
}

/// TrackingRate, from the advertised list.
pub struct SetTrackingRateOp {
    shared: Arc<MountShared>,
    meta: CommandMeta,
    rate: DriveRate,
    sent_mono_ns: u64,
}

impl SetTrackingRateOp {
    pub fn new(shared: Arc<MountShared>, meta: CommandMeta, rate: DriveRate) -> Self {
        Self {
            shared,
            meta,
            rate,
            sent_mono_ns: 0,
        }
    }

    fn announce(&self, on: Option<bool>, reason: &str) {
        self.shared.emit(
            EventKind::TrackingChanged {
                on,
                rate: Some(self.rate),
                source: Some(self.meta.source.clone()),
                reason: Some(reason.into()),
            },
            Some(self.meta.command_id),
        );
    }
}

impl Operation<DynMount> for SetTrackingRateOp {
    fn kind(&self) -> OpKind {
        OpKind::SetTrackingRate
    }

    fn deadline(&self) -> Duration {
        SETTER_DEADLINE
    }

    fn start(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> DeviceResult<()> {
        let rate = select_rate(&self.shared.capabilities().tracking_rates, self.rate)?;
        drv.set_tracking_rate(rate)?;
        self.sent_mono_ns = clock.now_mono_ns();
        Ok(())
    }

    fn step(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> OpProgress {
        let Ok(state) = drv.state() else {
            return OpProgress::Continue {
                next_poll: ACTIVE_POLL,
            };
        };
        let on = state.tracking.value;
        if state.tracking_rate.state == ObsState::Unsupported {
            self.announce(on, "rate_unverified");
            return OpProgress::Done(OpOutcome::Succeeded);
        }
        let fresh_since_write = state
            .tracking_rate
            .mono_ns
            .is_some_and(|at| at >= self.sent_mono_ns);
        if fresh_since_write && state.tracking_rate.fresh_value() == Some(&self.rate) {
            self.announce(on, "rate_changed");
            return OpProgress::Done(OpOutcome::Succeeded);
        }
        if clock.now_mono_ns().saturating_sub(self.sent_mono_ns) >= duration_ns(VERIFY_TIMEOUT) {
            return OpProgress::Done(OpOutcome::Failed(DeviceError::driver(format!(
                "TrackingRate reads {:?} after it was set to {:?}",
                state.tracking_rate.value, self.rate
            ))));
        }
        OpProgress::Continue {
            next_poll: ACTIVE_POLL,
        }
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
    use crate::equipment::types::ErrorClass;

    #[test]
    fn rates_come_only_from_the_advertised_list() {
        let all = [
            DriveRate::Sidereal,
            DriveRate::Lunar,
            DriveRate::Solar,
            DriveRate::King,
        ];
        assert_eq!(
            select_rate(&all, DriveRate::Solar).unwrap(),
            DriveRate::Solar
        );
        assert_eq!(
            select_rate(&[DriveRate::Sidereal], DriveRate::Lunar)
                .unwrap_err()
                .class,
            ErrorClass::InvalidValue
        );
        assert_eq!(
            select_rate(&[], DriveRate::Sidereal).unwrap_err().class,
            ErrorClass::Unsupported
        );
    }

    #[test]
    fn tracking_rate_writes_and_verifies_solar_as_solar() {
        let mut rig = rig(SimMountProfile::esp32_like());
        rig.runtime.take_events();
        assert_eq!(
            rig.run(MountCmd::SetTrackingRate(DriveRate::Solar), operator())
                .unwrap(),
            OpOutcome::Succeeded
        );
        let snapshot = rig.tick(Duration::from_secs(1), None);
        assert_eq!(snapshot.state.tracking_rate.value, Some(DriveRate::Solar));
        let events = rig.runtime.take_events();
        assert!(events.iter().any(|e| matches!(
            &e.event,
            EventKind::TrackingChanged { rate: Some(DriveRate::Solar), reason: Some(r), .. } if r == "rate_changed"
        )));
        let mut profile = SimMountProfile::esp32_like();
        profile.tracking_rates = vec![DriveRate::Sidereal];
        let mut rig = super::super::test_support::rig(profile);
        let outcome = rig
            .run(MountCmd::SetTrackingRate(DriveRate::King), operator())
            .unwrap();
        assert!(matches!(outcome, OpOutcome::Failed(e) if e.class == ErrorClass::InvalidValue));
    }

    #[test]
    fn tracking_writes_are_verified() {
        let mut rig = rig(SimMountProfile::esp32_like());
        assert_eq!(
            rig.run(MountCmd::SetTracking(false), operator()).unwrap(),
            OpOutcome::Succeeded
        );
        assert!(!rig.sim.control().tracking());
        rig.sim.control().set_faults(SimFaults {
            tracking_writes_ignored: true,
            ..SimFaults::default()
        });
        let outcome = rig.run(MountCmd::SetTracking(true), operator()).unwrap();
        assert!(matches!(outcome, OpOutcome::Failed(e) if e.class == ErrorClass::Driver));
    }

    #[test]
    fn tracking_cannot_be_turned_on_while_parked() {
        let mut profile = SimMountProfile::esp32_like();
        profile.start_parked = true;
        let mut rig = rig(profile);
        let outcome = rig.run(MountCmd::SetTracking(true), operator()).unwrap();
        assert!(
            matches!(outcome, OpOutcome::Failed(e) if e.class == ErrorClass::InvalidWhileParked)
        );
        assert_eq!(
            rig.run(MountCmd::SetTracking(false), operator()).unwrap(),
            OpOutcome::Succeeded
        );
    }
}

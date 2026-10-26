//! Slew and abort operations (§3.2 M-c, M-d, M-j). Owner: WP1A-4.
//!
//! A slew is gated on **CanSlewAsync** and uses only the asynchronous
//! methods (O10): compliant Alpaca mounts report CanSlew = false. It:
//!
//! 1. refuses while parked (InvalidWhileParked);
//! 2. converts the target to the native frame (M-a) and applies the
//!    `sync_mode = "offset"` pointing offset (M-g);
//! 3. on a GermanPolar mount that implements it, queries
//!    DestinationSideOfPier and refuses an unknown answer (M-j, §2.9);
//! 4. turns tracking on first when it is off, recorded as
//!    `tracking_changed{reason: "slew_precondition"}`; an Alt/Az slew
//!    turns it off instead, as ASCOM requires;
//! 5. sends the slew; completion is Slewing false (C-5), never a position
//!    comparison;
//! 6. settles for `settle_s` on top of the driver's own settle time;
//! 7. verifies the native position against the commanded native target and
//!    records `arrived_off_target` beyond `arrival_tolerance_arcmin` (not a
//!    failure).
//!
//! Events: `slew_commanded`, `slew_started`, `slew_arrived`, `slew_settled`
//! (which replaces the probe inference in `D/mount/plate_solve_trigger.rs`)
//! and `slew_ended`. Stopping guiding first (D20) is `coord.rs`'s job.

use std::sync::Arc;
use std::time::Duration;

use crate::equipment::events::EventKind;
use crate::equipment::ops::{OpKind, OpProgress, Operation};
use crate::equipment::snapshot::{DestinationSideOfPier, MountState};
use crate::equipment::traits::Clock;
use crate::equipment::types::{
    AlignmentMode, AltAz, DeviceError, DeviceResult, EqCoord, EqFrame, ErrorClass, ObsState,
    OpOutcome, PierSide,
};

use super::limits::{angular_separation_deg, ha_dec_from_alt_az};
use super::{
    duration_ns, elapsed_ms, fresh_position, is_parked, parked_error, position, CommandMeta,
    DynMount, MountShared, ACTIVE_POLL, SETTER_DEADLINE, VERIFY_TIMEOUT,
};

/// Polls allowed for a fresh position before the settle is reported without
/// one.
const VERIFY_POLLS: u32 = 3;

/// What a slew aims at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SlewTarget {
    Equatorial(EqCoord),
    AltAz(AltAz),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Phase {
    Idle,
    Moving,
    Settling { until_mono_ns: u64 },
    Verifying { polls: u32 },
    Cancelling,
    Finished,
}

/// GoTo to RA/Dec or Alt/Az.
pub struct SlewOp {
    shared: Arc<MountShared>,
    meta: CommandMeta,
    target: SlewTarget,
    /// The native coordinate actually sent (after any offset).
    commanded: Option<EqCoord>,
    phase: Phase,
    started_mono_ns: u64,
}

impl SlewOp {
    pub fn equatorial(shared: Arc<MountShared>, meta: CommandMeta, target: EqCoord) -> Self {
        Self::new(shared, meta, SlewTarget::Equatorial(target))
    }

    pub fn altaz(shared: Arc<MountShared>, meta: CommandMeta, target: AltAz) -> Self {
        Self::new(shared, meta, SlewTarget::AltAz(target))
    }

    fn new(shared: Arc<MountShared>, meta: CommandMeta, target: SlewTarget) -> Self {
        Self {
            shared,
            meta,
            target,
            commanded: None,
            phase: Phase::Idle,
            started_mono_ns: 0,
        }
    }

    fn emit(&self, kind: EventKind) {
        self.shared.emit(kind, Some(self.meta.command_id));
    }

    fn duration_ms(&self, clock: &dyn Clock) -> u64 {
        elapsed_ms(self.started_mono_ns, clock.now_mono_ns())
    }

    /// M-j / §2.9: on a GermanPolar mount that implements it, the
    /// destination side must be known before the slew is sent.
    fn check_destination(&self, drv: &mut DynMount, commanded: EqCoord) -> DeviceResult<()> {
        let caps = self.shared.capabilities();
        if caps.alignment_mode != Some(AlignmentMode::GermanPolar)
            || self.shared.destination_unsupported()
        {
            return Ok(());
        }
        match drv.destination_side_of_pier(commanded) {
            Ok(PierSide::Unknown) => Err(DeviceError::invalid_operation(
                "destination_side_of_pier_unknown: the mount cannot say which side of the pier \
                 it would use for this target",
            )),
            Ok(side) => {
                self.shared.set_destination(DestinationSideOfPier {
                    target: commanded,
                    side,
                    at: self.shared.clock().now_utc(),
                });
                Ok(())
            }
            Err(err) if err.is_unknown_member() || err.class == ErrorClass::Unsupported => {
                self.shared.mark_destination_unsupported();
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    fn start_equatorial(
        &mut self,
        drv: &mut DynMount,
        target: EqCoord,
        state: &MountState,
    ) -> DeviceResult<()> {
        let caps = self.shared.capabilities();
        if !caps.can_slew_async {
            return Err(DeviceError::unsupported(
                "the mount does not report CanSlewAsync; GoTo uses only the asynchronous slew (O10)",
            ));
        }
        let native = self.shared.to_native(target)?;
        let commanded = match self.shared.settings().sync_mode {
            crate::equipment::types::MountSyncMode::Offset => {
                self.shared.with_offsets(|offsets| offsets.apply(native))
            }
            crate::equipment::types::MountSyncMode::Mount => native,
        };
        self.check_destination(drv, commanded)?;
        self.emit(EventKind::SlewCommanded {
            target,
            native_target: Some(commanded),
            altaz: None,
        });
        if state.tracking.known() == Some(&false) && caps.can_set_tracking {
            drv.set_tracking(true)?;
            self.emit(EventKind::TrackingChanged {
                on: Some(true),
                rate: state.tracking_rate.value,
                source: Some(self.meta.source.clone()),
                reason: Some("slew_precondition".into()),
            });
            self.shared.set_announced_tracking(true);
        }
        drv.slew_async(commanded)?;
        self.commanded = Some(commanded);
        Ok(())
    }

    fn start_altaz(
        &mut self,
        drv: &mut DynMount,
        target: AltAz,
        state: &MountState,
    ) -> DeviceResult<()> {
        let caps = self.shared.capabilities();
        if !target.alt_deg.is_finite()
            || !target.az_deg.is_finite()
            || !(-90.0..=90.0).contains(&target.alt_deg)
        {
            return Err(DeviceError::invalid_value(
                "altitude must be within [-90, 90]",
            ));
        }
        if !caps.can_slew_altaz_async {
            return Err(DeviceError::unsupported(
                "the mount does not report CanSlewAltAzAsync",
            ));
        }
        let target = AltAz {
            az_deg: target.az_deg.rem_euclid(360.0),
            ..target
        };
        // The event's RA/Dec is informative only; `altaz` is authoritative.
        let equivalent = self
            .shared
            .geometry_site()
            .zip(state.sidereal_time_h.known().copied())
            .map(|(site, lst)| {
                let (ha, dec) = ha_dec_from_alt_az(target, site.lat_deg);
                EqCoord::new(EqCoord::wrap_ra_h(lst - ha), dec, EqFrame::Unknown)
            })
            .unwrap_or_default();
        self.emit(EventKind::SlewCommanded {
            target: equivalent,
            native_target: None,
            altaz: Some(target),
        });
        if state.tracking.known() == Some(&true) && caps.can_set_tracking {
            drv.set_tracking(false)?;
            self.emit(EventKind::TrackingChanged {
                on: Some(false),
                rate: state.tracking_rate.value,
                source: Some(self.meta.source.clone()),
                reason: Some("slew_precondition".into()),
            });
            self.shared.set_announced_tracking(false);
        }
        drv.slew_altaz_async(target)?;
        self.target = SlewTarget::AltAz(target);
        Ok(())
    }

    fn verify(&mut self, state: &MountState, polls: u32, clock: &dyn Clock) -> OpProgress {
        let native = self.shared.capabilities().native_frame;
        let fresh = fresh_position(state, native);
        if fresh.is_none() && polls < VERIFY_POLLS {
            self.phase = Phase::Verifying { polls: polls + 1 };
            return OpProgress::Continue {
                next_poll: ACTIVE_POLL,
            };
        }
        let error_arcmin = match self.target {
            SlewTarget::Equatorial(_) => fresh.zip(self.commanded).and_then(|(at, aim)| {
                at.separation_arcmin(&EqCoord {
                    frame: at.frame,
                    ..aim
                })
            }),
            SlewTarget::AltAz(aim) => state
                .alt_deg
                .fresh_value()
                .zip(state.az_deg.fresh_value())
                .map(|(alt, az)| angular_separation_deg(*az, *alt, aim.az_deg, aim.alt_deg) * 60.0),
        };
        if let Some(error) = error_arcmin {
            if error > self.shared.settings().arrival_tolerance_arcmin {
                self.emit(EventKind::ArrivedOffTarget {
                    error_arcmin: error,
                });
            }
        }
        self.emit(EventKind::SlewSettled {
            position: fresh,
            error_arcmin,
        });
        self.emit(EventKind::SlewEnded {
            outcome: OpOutcome::Succeeded,
            duration_ms: self.duration_ms(clock),
        });
        self.phase = Phase::Finished;
        OpProgress::Done(OpOutcome::Succeeded)
    }
}

impl Operation<DynMount> for SlewOp {
    fn kind(&self) -> OpKind {
        match self.target {
            SlewTarget::Equatorial(_) => OpKind::Slew,
            SlewTarget::AltAz(_) => OpKind::SlewAltAz,
        }
    }

    fn deadline(&self) -> Duration {
        self.shared.settings().slew_timeout
    }

    fn start(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> DeviceResult<()> {
        let state = drv.state()?;
        if is_parked(&state) {
            return Err(parked_error("slewing"));
        }
        self.started_mono_ns = clock.now_mono_ns();
        match self.target {
            SlewTarget::Equatorial(target) => self.start_equatorial(drv, target, &state)?,
            SlewTarget::AltAz(target) => self.start_altaz(drv, target, &state)?,
        }
        self.phase = Phase::Moving;
        self.emit(EventKind::SlewStarted);
        Ok(())
    }

    fn step(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> OpProgress {
        let Ok(state) = drv.state() else {
            return OpProgress::Continue {
                next_poll: ACTIVE_POLL,
            };
        };
        match self.phase {
            Phase::Idle | Phase::Moving => match state.slewing.fresh_value() {
                Some(false) => {
                    let native = self.shared.capabilities().native_frame;
                    self.emit(EventKind::SlewArrived {
                        position: fresh_position(&state, native),
                    });
                    let settle = self.shared.settings().settle;
                    if settle.is_zero() {
                        return self.verify(&state, 0, clock);
                    }
                    let until_mono_ns = clock.now_mono_ns() + duration_ns(settle);
                    self.phase = Phase::Settling { until_mono_ns };
                    OpProgress::Settling { until_mono_ns }
                }
                // No completion property at all (the OLS legacy adapter):
                // arrival cannot be observed, so the outcome is unknown
                // (C-5, C-6) rather than a guess or a 10-minute timeout.
                _ if state.slewing.state == ObsState::Unsupported => {
                    let outcome = OpOutcome::Unknown(DeviceError::outcome_unknown(
                        "the mount does not report Slewing; arrival cannot be observed",
                    ));
                    self.emit(EventKind::SlewEnded {
                        outcome: outcome.clone(),
                        duration_ms: self.duration_ms(clock),
                    });
                    self.phase = Phase::Finished;
                    OpProgress::Done(outcome)
                }
                _ => OpProgress::Continue {
                    next_poll: ACTIVE_POLL,
                },
            },
            Phase::Settling { until_mono_ns } => {
                if clock.now_mono_ns() < until_mono_ns {
                    OpProgress::Settling { until_mono_ns }
                } else {
                    self.verify(&state, 0, clock)
                }
            }
            Phase::Verifying { polls } => self.verify(&state, polls, clock),
            // After `cancel`, a poll that reads Slewing false confirms the stop.
            Phase::Cancelling => match state.slewing.fresh_value() {
                Some(false) => {
                    self.phase = Phase::Finished;
                    OpProgress::Done(OpOutcome::Aborted { by: None })
                }
                _ => OpProgress::Continue {
                    next_poll: ACTIVE_POLL,
                },
            },
            Phase::Finished => OpProgress::Done(OpOutcome::Succeeded),
        }
    }

    /// AbortSlew (C-4). Tracking is not changed (M-d). The `slew_ended`
    /// outcome is `timed_out` past the deadline, else `aborted`.
    fn cancel(&mut self, drv: &mut DynMount) -> DeviceResult<()> {
        let clock = self.shared.clock();
        let timed_out = clock.now_mono_ns().saturating_sub(self.started_mono_ns)
            >= duration_ns(self.deadline());
        let result = abort_slew(drv);
        if !matches!(self.phase, Phase::Finished | Phase::Cancelling) {
            self.emit(EventKind::SlewEnded {
                outcome: if timed_out {
                    OpOutcome::TimedOut
                } else {
                    OpOutcome::Aborted { by: None }
                },
                duration_ms: self.duration_ms(clock),
            });
        }
        self.phase = Phase::Cancelling;
        result
    }
}

/// AbortSlew that treats "parked, nothing to abort" as success.
pub fn abort_slew(drv: &mut DynMount) -> DeviceResult<()> {
    match drv.abort_slew() {
        Err(err) if err.class == ErrorClass::InvalidWhileParked => Ok(()),
        other => other,
    }
}

/// `MountCmd::AbortSlew` sent while no operation runs (a slew by another
/// client, or a jog): AbortSlew, plus MoveAxis 0 on jogging axes. Completes
/// when Slewing reads false. Never gated on a Can* flag (O10, D19).
pub struct AbortSlewOp {
    shared: Arc<MountShared>,
    meta: CommandMeta,
    sent_mono_ns: u64,
}

impl AbortSlewOp {
    pub fn new(shared: Arc<MountShared>, meta: CommandMeta) -> Self {
        Self {
            shared,
            meta,
            sent_mono_ns: 0,
        }
    }
}

impl Operation<DynMount> for AbortSlewOp {
    fn kind(&self) -> OpKind {
        OpKind::AbortSlew
    }

    fn deadline(&self) -> Duration {
        Duration::from_secs(30)
    }

    fn start(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> DeviceResult<()> {
        self.sent_mono_ns = clock.now_mono_ns();
        let result = abort_slew(drv);
        let axes = self.shared.with_jog(|jog| jog.stop_all());
        for axis in axes {
            drv.move_axis(axis, 0.0)?;
            self.shared.emit(
                EventKind::MoveAxis {
                    axis,
                    rate_deg_s: 0.0,
                },
                Some(self.meta.command_id),
            );
        }
        result
    }

    fn step(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> OpProgress {
        let slewing = drv
            .state()
            .ok()
            .and_then(|s| s.slewing.fresh_value().copied());
        match slewing {
            Some(false) => OpProgress::Done(OpOutcome::Succeeded),
            None if clock.now_mono_ns().saturating_sub(self.sent_mono_ns)
                >= duration_ns(VERIFY_TIMEOUT) =>
            {
                OpProgress::Done(OpOutcome::Unknown(DeviceError::outcome_unknown(
                    "Slewing could not be read after AbortSlew",
                )))
            }
            _ => OpProgress::Continue {
                next_poll: ACTIVE_POLL,
            },
        }
    }

    fn cancel(&mut self, drv: &mut DynMount) -> DeviceResult<()> {
        abort_slew(drv)
    }
}

/// SideOfPier write (a pier flip slew; used by the Tier 2 flip workflow).
/// Requires CanSetPierSide; completes when Slewing is false and SideOfPier
/// reads the requested side.
pub struct SetSideOfPierOp {
    shared: Arc<MountShared>,
    _meta: CommandMeta,
    side: PierSide,
    saw_motion_stop_mono_ns: Option<u64>,
}

impl SetSideOfPierOp {
    pub fn new(shared: Arc<MountShared>, meta: CommandMeta, side: PierSide) -> DeviceResult<Self> {
        if side == PierSide::Unknown {
            return Err(DeviceError::invalid_value(
                "the pier side must be east or west",
            ));
        }
        Ok(Self {
            shared,
            _meta: meta,
            side,
            saw_motion_stop_mono_ns: None,
        })
    }
}

impl Operation<DynMount> for SetSideOfPierOp {
    fn kind(&self) -> OpKind {
        OpKind::SetSideOfPier
    }

    fn deadline(&self) -> Duration {
        self.shared.settings().slew_timeout
    }

    fn start(&mut self, drv: &mut DynMount, _clock: &dyn Clock) -> DeviceResult<()> {
        if !self.shared.capabilities().can_set_pier_side {
            return Err(DeviceError::unsupported(
                "the mount does not report CanSetPierSide",
            ));
        }
        if is_parked(&drv.state()?) {
            return Err(parked_error("changing the pier side"));
        }
        drv.set_side_of_pier(self.side)
    }

    fn step(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> OpProgress {
        let Ok(state) = drv.state() else {
            return OpProgress::Continue {
                next_poll: ACTIVE_POLL,
            };
        };
        if state.slewing.fresh_value() != Some(&false) {
            self.saw_motion_stop_mono_ns = None;
            return OpProgress::Continue {
                next_poll: ACTIVE_POLL,
            };
        }
        if state.side_of_pier.fresh_value() == Some(&self.side) {
            return OpProgress::Done(OpOutcome::Succeeded);
        }
        let now = clock.now_mono_ns();
        let stopped = *self.saw_motion_stop_mono_ns.get_or_insert(now);
        if now.saturating_sub(stopped) >= duration_ns(VERIFY_TIMEOUT) {
            return OpProgress::Done(OpOutcome::Failed(DeviceError::driver(format!(
                "SideOfPier still reads {:?} after the pier flip stopped",
                state.side_of_pier.value
            ))));
        }
        OpProgress::Continue {
            next_poll: ACTIVE_POLL,
        }
    }

    fn cancel(&mut self, drv: &mut DynMount) -> DeviceResult<()> {
        abort_slew(drv)
    }
}

/// `destinationsideofpier` for a planned target (M-j). The answer is
/// published as `MountSnapshot::destination_side_of_pier`.
pub struct DestinationQueryOp {
    shared: Arc<MountShared>,
    _meta: CommandMeta,
    target: EqCoord,
    result: Option<OpOutcome>,
}

impl DestinationQueryOp {
    pub fn new(shared: Arc<MountShared>, meta: CommandMeta, target: EqCoord) -> Self {
        Self {
            shared,
            _meta: meta,
            target,
            result: None,
        }
    }
}

impl Operation<DynMount> for DestinationQueryOp {
    fn kind(&self) -> OpKind {
        OpKind::DestinationSideOfPier
    }

    fn deadline(&self) -> Duration {
        SETTER_DEADLINE
    }

    fn start(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> DeviceResult<()> {
        let native = self.shared.to_native(self.target)?;
        let side = match drv.destination_side_of_pier(native) {
            Ok(side) => side,
            Err(err) if err.is_unknown_member() || err.class == ErrorClass::Unsupported => {
                self.shared.mark_destination_unsupported();
                return Err(err);
            }
            Err(err) => return Err(err),
        };
        self.shared.set_destination(DestinationSideOfPier {
            target: native,
            side,
            at: clock.now_utc(),
        });
        self.result = Some(OpOutcome::Succeeded);
        Ok(())
    }

    fn step(&mut self, _drv: &mut DynMount, _clock: &dyn Clock) -> OpProgress {
        OpProgress::Done(self.result.clone().unwrap_or(OpOutcome::Succeeded))
    }

    fn cancel(&mut self, _drv: &mut DynMount) -> DeviceResult<()> {
        Ok(())
    }
}

/// The current native position, for callers that need it outside an op.
pub fn native_position(shared: &MountShared, state: &MountState) -> Option<EqCoord> {
    position(state, shared.capabilities().native_frame)
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::super::{build_operation, MountSettings};
    use super::*;
    use crate::equipment::commands::MountCmd;
    use crate::equipment::sim::mount::{SimFaults, SimMountProfile};
    use crate::equipment::traits::{Device, MountDriver};
    use crate::equipment::types::{EqFrame, MountSyncMode};

    fn ahead(rig: &mut Rig, ha_h: f64, dec: f64) -> EqCoord {
        let lst = rig
            .tick(Duration::from_millis(10), None)
            .state
            .sidereal_time_h
            .value
            .unwrap();
        EqCoord::new(EqCoord::wrap_ra_h(lst - ha_h), dec, EqFrame::Unknown)
    }

    #[test]
    fn a_slew_completes_from_slewing_then_settles_and_verifies() {
        let mut rig = rig(SimMountProfile::esp32_like());
        rig.runtime.take_events();
        let target = ahead(&mut rig, 1.0, 45.0);
        let meta = operator();
        let started = rig.clock.now_mono_ns();
        let outcome = rig.run(MountCmd::Slew { target }, meta.clone()).unwrap();
        assert_eq!(outcome, OpOutcome::Succeeded);
        let names = rig.names();
        let expected = [
            "slew_commanded",
            "slew_started",
            "slew_arrived",
            "slew_settled",
            "slew_ended",
        ];
        let positions: Vec<_> = expected
            .iter()
            .map(|n| {
                names
                    .iter()
                    .position(|x| x == n)
                    .unwrap_or_else(|| panic!("{n} missing: {names:?}"))
            })
            .collect();
        assert!(positions.windows(2).all(|w| w[0] < w[1]), "{names:?}");
        assert!(!names.contains(&"arrived_off_target"));
        let events = rig.runtime.take_events();
        let settled = events.iter().find(|e| e.name() == "slew_settled").unwrap();
        assert_eq!(settled.command_id, Some(meta.command_id));
        assert!(
            matches!(settled.event, EventKind::SlewSettled { error_arcmin: Some(e), .. } if e < 1.0)
        );
        // Settle (5 s) comes after arrival.
        let arrived = events.iter().find(|e| e.name() == "slew_arrived").unwrap();
        assert!(settled.mono_ns - arrived.mono_ns >= 5_000_000_000);
        assert!(arrived.mono_ns > started);
        // The O10 profile never touched a synchronous slew.
        assert!(rig.sim.control().calls().iter().any(|c| c == "slew_async"));
    }

    #[test]
    fn a_slew_turns_tracking_on_first_and_records_why() {
        let mut rig = rig(SimMountProfile::esp32_like());
        rig.sim.set_tracking(false).unwrap();
        rig.tick(Duration::from_secs(1), None);
        rig.runtime.take_events();
        let target = ahead(&mut rig, 1.0, 20.0);
        assert_eq!(
            rig.run(MountCmd::Slew { target }, operator()).unwrap(),
            OpOutcome::Succeeded
        );
        let events = rig.runtime.take_events();
        let changed: Vec<_> = events
            .iter()
            .filter(|e| e.name() == "tracking_changed")
            .collect();
        assert_eq!(changed.len(), 1, "announced once, by the slew");
        assert!(matches!(
            &changed[0].event,
            EventKind::TrackingChanged { on: Some(true), reason: Some(r), .. } if r == "slew_precondition"
        ));
    }

    #[test]
    fn slews_are_refused_while_parked_and_without_can_slew_async() {
        let mut profile = SimMountProfile::esp32_like();
        profile.start_parked = true;
        let mut rig = rig(profile);
        let target = ahead(&mut rig, 1.0, 20.0);
        assert_eq!(
            rig.run(MountCmd::Slew { target }, operator()).unwrap(),
            OpOutcome::Failed(parked_error("slewing"))
        );
        let mut profile = SimMountProfile::esp32_like();
        profile.can_slew_async = false;
        profile.can_slew = true;
        let mut rig = super::super::test_support::rig(profile);
        let target = ahead(&mut rig, 1.0, 20.0);
        let outcome = rig.run(MountCmd::Slew { target }, operator()).unwrap();
        assert!(matches!(outcome, OpOutcome::Failed(e) if e.class == ErrorClass::Unsupported));
    }

    #[test]
    fn frame_sensitive_slews_need_a_known_native_frame() {
        let mut profile = SimMountProfile::esp32_like();
        profile.equatorial_system = Some(0);
        let mut rig = rig(profile);
        let mut target = ahead(&mut rig, 1.0, 20.0);
        target.frame = EqFrame::J2000;
        let outcome = rig.run(MountCmd::Slew { target }, operator()).unwrap();
        assert!(matches!(outcome, OpOutcome::Failed(e) if e.class == ErrorClass::InvalidOperation));
        // The legacy "mount frame" still works.
        target.frame = EqFrame::Unknown;
        assert_eq!(
            rig.run(MountCmd::Slew { target }, operator()).unwrap(),
            OpOutcome::Succeeded
        );
    }

    #[test]
    fn a_j2000_target_is_converted_to_the_native_frame() {
        let mut rig = rig(SimMountProfile::esp32_like());
        let mut target = ahead(&mut rig, 1.0, 20.0);
        target.frame = EqFrame::J2000;
        rig.runtime.take_events();
        assert_eq!(
            rig.run(MountCmd::Slew { target }, operator()).unwrap(),
            OpOutcome::Succeeded
        );
        let events = rig.runtime.take_events();
        let commanded = events
            .iter()
            .find(|e| e.name() == "slew_commanded")
            .unwrap();
        assert!(matches!(
            commanded.event,
            EventKind::SlewCommanded { target: t, native_target: Some(n), .. }
                if t.frame == EqFrame::J2000 && n.frame == EqFrame::JNowApparent
        ));
    }

    #[test]
    fn an_abort_mid_slew_stops_the_mount_and_ends_the_slew_aborted() {
        let mut rig = rig(SimMountProfile::esp32_like());
        let target = ahead(&mut rig, -3.0, -10.0);
        let meta = operator();
        let mut op = build_operation(
            &rig.runtime.shared(),
            MountCmd::Slew { target },
            meta.clone(),
        )
        .unwrap();
        op.start(&mut rig.sim, rig.clock.as_ref()).unwrap();
        assert_eq!(advance(&mut rig, &mut op, &meta, 4), None);
        assert!(rig.sim.control().slewing());
        // The handle's abort runs the out-of-band path, then the worker cancels.
        rig.sim.abort_path().unwrap().abort().unwrap();
        op.cancel(&mut rig.sim).unwrap();
        let outcome = advance(&mut rig, &mut op, &meta, 4).unwrap();
        assert_eq!(outcome, OpOutcome::Aborted { by: None });
        assert!(!rig.sim.control().slewing());
        assert!(
            rig.sim.control().tracking(),
            "abort leaves tracking alone (M-d)"
        );
        let events = rig.runtime.take_events();
        assert!(events.iter().any(|e| matches!(
            e.event,
            EventKind::SlewEnded {
                outcome: OpOutcome::Aborted { .. },
                ..
            }
        )));
        assert!(!events.iter().any(|e| e.name() == "slew_settled"));
    }

    #[test]
    fn a_hung_slew_times_out_and_is_halted() {
        let settings = MountSettings {
            slew_timeout: Duration::from_secs(20),
            ..MountSettings::default()
        };
        let mut rig = rig_with(SimMountProfile::esp32_like(), settings, Some(site()));
        rig.sim.control().set_faults(SimFaults {
            slew_hangs: true,
            ..SimFaults::default()
        });
        let target = ahead(&mut rig, 2.0, 10.0);
        assert_eq!(
            rig.run(MountCmd::Slew { target }, operator()).unwrap(),
            OpOutcome::TimedOut
        );
        assert!(
            !rig.sim.control().slewing(),
            "cancel halts the hardware (C-4)"
        );
        let events = rig.runtime.take_events();
        assert!(events.iter().any(|e| matches!(
            e.event,
            EventKind::SlewEnded {
                outcome: OpOutcome::TimedOut,
                ..
            }
        )));
    }

    #[test]
    fn an_off_target_arrival_is_recorded_not_failed() {
        let settings = MountSettings {
            arrival_tolerance_arcmin: 1.0,
            ..MountSettings::default()
        };
        let mut rig = rig_with(SimMountProfile::esp32_like(), settings, Some(site()));
        let target = ahead(&mut rig, 1.0, 20.0);
        let meta = operator();
        let mut op = build_operation(
            &rig.runtime.shared(),
            MountCmd::Slew { target },
            meta.clone(),
        )
        .unwrap();
        op.start(&mut rig.sim, rig.clock.as_ref()).unwrap();
        // Arrive, then nudge the mount 10' during the settle.
        loop {
            rig.tick(
                ACTIVE_POLL,
                Some(active_op(OpKind::Slew, &meta, &rig.clock)),
            );
            if let OpProgress::Settling { .. } = op.step(&mut rig.sim, rig.clock.as_ref()) {
                break;
            }
        }
        let (ra, dec) = rig.sim.control().position();
        rig.sim
            .control()
            .place(EqCoord::new(ra, dec + 10.0 / 60.0, EqFrame::JNowApparent));
        let outcome = advance(&mut rig, &mut op, &meta, 40).unwrap();
        assert_eq!(outcome, OpOutcome::Succeeded);
        let events = rig.runtime.take_events();
        assert!(events.iter().any(|e| matches!(
            e.event,
            EventKind::ArrivedOffTarget { error_arcmin } if (error_arcmin - 10.0).abs() < 0.5
        )));
    }

    #[test]
    fn german_mounts_need_a_known_destination_side() {
        let mut rig = rig(SimMountProfile::german_equatorial());
        let target = ahead(&mut rig, 2.0, 20.0);
        assert_eq!(
            rig.run(MountCmd::Slew { target }, operator()).unwrap(),
            OpOutcome::Succeeded
        );
        let snapshot = rig.tick(Duration::from_secs(1), None);
        assert_eq!(
            snapshot.destination_side_of_pier.map(|d| d.side),
            Some(PierSide::East)
        );
        rig.sim
            .control()
            .update_faults(|f| f.destination_unknown = true);
        let target = ahead(&mut rig, -2.0, 20.0);
        let outcome = rig.run(MountCmd::Slew { target }, operator()).unwrap();
        assert!(
            matches!(&outcome, OpOutcome::Failed(e) if e.message.contains("destination_side_of_pier_unknown"))
        );
        // Query on its own.
        rig.sim
            .control()
            .update_faults(|f| f.destination_unknown = false);
        let outcome = rig
            .run(MountCmd::QueryDestinationSideOfPier(target), operator())
            .unwrap();
        assert_eq!(outcome, OpOutcome::Succeeded);
        let snapshot = rig.tick(Duration::from_secs(1), None);
        assert_eq!(
            snapshot.destination_side_of_pier.map(|d| d.side),
            Some(PierSide::West)
        );
    }

    #[test]
    fn alt_az_slews_turn_tracking_off_and_verify_alt_az() {
        let mut rig = rig(SimMountProfile::german_equatorial());
        rig.runtime.take_events();
        let target = AltAz {
            alt_deg: 45.0,
            az_deg: 200.0,
        };
        assert_eq!(
            rig.run(MountCmd::SlewAltAz(target), operator()).unwrap(),
            OpOutcome::Succeeded
        );
        let snapshot = rig.tick(Duration::from_secs(1), None);
        assert_eq!(snapshot.state.tracking.value, Some(false));
        assert!((snapshot.state.alt_deg.value.unwrap() - 45.0).abs() < 0.01);
        let events = rig.runtime.take_events();
        assert!(events.iter().any(|e| matches!(
            &e.event,
            EventKind::SlewCommanded { altaz: Some(a), native_target: None, .. } if a.az_deg == 200.0
        )));
        assert!(events.iter().any(
            |e| matches!(e.event, EventKind::SlewSettled { error_arcmin: Some(x), .. } if x < 1.0)
        ));
        // The ESP32 profile has no Alt/Az slew.
        let mut rig = super::super::test_support::rig(SimMountProfile::esp32_like());
        let outcome = rig.run(MountCmd::SlewAltAz(target), operator()).unwrap();
        assert!(matches!(outcome, OpOutcome::Failed(e) if e.class == ErrorClass::Unsupported));
    }

    #[test]
    fn offset_mode_slews_apply_the_recorded_offset() {
        let settings = MountSettings {
            sync_mode: MountSyncMode::Offset,
            ..MountSettings::default()
        };
        let mut rig = rig_with(SimMountProfile::esp32_like(), settings, Some(site()));
        let target = ahead(&mut rig, 1.0, 30.0);
        let native = EqCoord {
            frame: EqFrame::JNowApparent,
            ..target
        };
        // The mount reads 0.5 deg higher in Dec than the sky truth here.
        rig.runtime.shared().with_offsets(|o| {
            o.record(
                EqCoord::new(native.ra_h, native.dec_deg + 0.5, native.frame),
                native,
                chrono::Utc::now(),
            )
        });
        rig.runtime.take_events();
        assert_eq!(
            rig.run(MountCmd::Slew { target }, operator()).unwrap(),
            OpOutcome::Succeeded
        );
        let events = rig.runtime.take_events();
        assert!(events.iter().any(|e| matches!(
            e.event,
            EventKind::SlewCommanded { native_target: Some(n), .. } if (n.dec_deg - 30.5).abs() < 1e-9
        )));
    }

    #[test]
    fn abort_slew_alone_stops_an_external_slew_and_a_jog() {
        let mut rig = rig(SimMountProfile::esp32_like());
        let target = ahead(&mut rig, -4.0, 0.0);
        rig.sim.control().external_goto(EqCoord {
            frame: EqFrame::JNowApparent,
            ..target
        });
        rig.tick(Duration::from_secs(1), None);
        assert!(rig.sim.control().slewing());
        assert_eq!(
            rig.run(MountCmd::AbortSlew, operator()).unwrap(),
            OpOutcome::Succeeded
        );
        assert!(!rig.sim.control().slewing());
        // Parked: AbortSlew is still accepted (InvalidWhileParked is not a failure).
        let mut profile = SimMountProfile::esp32_like();
        profile.start_parked = true;
        let mut rig = super::super::test_support::rig(profile);
        assert_eq!(
            rig.run(MountCmd::AbortSlew, operator()).unwrap(),
            OpOutcome::Succeeded
        );
    }

    #[test]
    fn set_side_of_pier_flips_and_verifies() {
        let mut rig = rig(SimMountProfile::german_equatorial());
        let side = rig
            .tick(Duration::from_secs(1), None)
            .state
            .side_of_pier
            .value
            .unwrap();
        let other = if side == PierSide::East {
            PierSide::West
        } else {
            PierSide::East
        };
        assert_eq!(
            rig.run(MountCmd::SetSideOfPier(other), operator()).unwrap(),
            OpOutcome::Succeeded
        );
        assert_eq!(
            rig.tick(Duration::from_secs(1), None)
                .state
                .side_of_pier
                .value,
            Some(other)
        );
        assert!(rig.names().contains(&"pier_side_changed"));
        let mut rig = super::super::test_support::rig(SimMountProfile::esp32_like());
        let outcome = rig
            .run(MountCmd::SetSideOfPier(PierSide::West), operator())
            .unwrap();
        assert!(matches!(outcome, OpOutcome::Failed(e) if e.class == ErrorClass::Unsupported));
    }
}

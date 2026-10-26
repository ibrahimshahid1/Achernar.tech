//! MoveAxis jog with a dead-man (§3.2 M-h). Owner: WP1A-4.
//!
//! - Requires **CanMoveAxis(axis)**, never CanSlew (O10): compliant Alpaca
//!   mounts report CanSlew = false. Refused while parked.
//! - The rate is clamped into the advertised AxisRates intervals, keeping
//!   its sign.
//! - The MoveAxis itself counts as the first renewal. A `RenewJog` (or a
//!   further MoveAxis) is required every `jog_deadman_ms`; when it lapses the
//!   jog sends rate 0 on every moving axis and ends `Failed(timeout)` with
//!   `jog_deadman_lapsed`.
//! - Further MoveAxis commands while a jog runs (a second axis, a new rate,
//!   rate 0 to stop one axis) are absorbed by the active jog
//!   ([`super::MountPlan::Absorbed`]). The jog ends when every axis is at 0.
//! - Jog is direct-transport only (D12; the gate check refuses the relay).
//!
//! [`JogFlags`] lets a driver's out-of-band abort path know which axes it
//! must stop, so `/mount/abort` adds MoveAxis 0 only while a jog is active.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::equipment::events::EventKind;
use crate::equipment::ops::{OpKind, OpProgress, Operation};
use crate::equipment::snapshot::{MountCapabilities, RateRange};
use crate::equipment::traits::Clock;
use crate::equipment::types::{DeviceError, DeviceResult, OpOutcome};

use super::{
    duration_ns, is_parked, parked_error, CommandMeta, DynMount, MountShared, ACTIVE_POLL,
};

/// Axes 0 (primary), 1 (secondary) and 2 (tertiary).
pub const AXES: u8 = 3;

/// Which axes a driver has commanded to a non-zero rate. Shared between a
/// driver and its abort path.
#[derive(Debug, Default)]
pub struct JogFlags([AtomicBool; 3]);

impl JogFlags {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn set(&self, axis: u8, rate_deg_s: f64) {
        if let Some(flag) = self.0.get(usize::from(axis)) {
            flag.store(rate_deg_s != 0.0, Ordering::SeqCst);
        }
    }

    pub fn moving_axes(&self) -> Vec<u8> {
        (0..AXES)
            .filter(|axis| self.0[usize::from(*axis)].load(Ordering::SeqCst))
            .collect()
    }

    pub fn clear(&self) {
        for flag in &self.0 {
            flag.store(false, Ordering::SeqCst);
        }
    }
}

/// Clamp `rate_deg_s` into the advertised intervals, keeping its sign. Zero
/// stays zero (stop).
pub fn clamp_axis_rate(rate_deg_s: f64, ranges: &[RateRange]) -> DeviceResult<f64> {
    if !rate_deg_s.is_finite() {
        return Err(DeviceError::invalid_value(
            "the axis rate must be a finite number",
        ));
    }
    if rate_deg_s == 0.0 {
        return Ok(0.0);
    }
    let valid: Vec<_> = ranges
        .iter()
        .filter(|r| r.min_deg_s.is_finite() && r.max_deg_s.is_finite() && r.max_deg_s > 0.0)
        .map(|r| {
            (
                r.min_deg_s.abs().min(r.max_deg_s.abs()),
                r.max_deg_s.abs().max(r.min_deg_s.abs()),
            )
        })
        .collect();
    if valid.is_empty() {
        return Err(DeviceError::invalid_value(
            "the mount advertises no AxisRates for this axis",
        ));
    }
    let magnitude = rate_deg_s.abs();
    let clamped = valid
        .iter()
        .map(|(lo, hi)| magnitude.clamp(*lo, *hi))
        .min_by(|a, b| (a - magnitude).abs().total_cmp(&(b - magnitude).abs()))
        .unwrap_or(magnitude);
    Ok(clamped.copysign(rate_deg_s))
}

/// Validate an axis against the capabilities and clamp its rate.
pub fn validated_rate(caps: &MountCapabilities, axis: u8, rate_deg_s: f64) -> DeviceResult<f64> {
    if axis >= AXES {
        return Err(DeviceError::invalid_value(format!(
            "axis {axis} is not 0, 1 or 2"
        )));
    }
    if !caps.can_move_axis[usize::from(axis)] {
        return Err(DeviceError::unsupported(format!(
            "the mount does not report CanMoveAxis({axis})"
        )));
    }
    if rate_deg_s == 0.0 {
        return Ok(0.0);
    }
    let ranges = caps
        .axis_rates
        .iter()
        .find(|r| r.axis == axis)
        .map(|r| r.ranges.as_slice())
        .unwrap_or(&[]);
    clamp_axis_rate(rate_deg_s, ranges)
}

/// What the operator wants the axes to do, and when they last said so.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct JogDemand {
    rates: [f64; 3],
    last_renewal_mono_ns: u64,
    active: bool,
}

impl JogDemand {
    pub fn active(&self) -> bool {
        self.active
    }

    pub fn rates(&self) -> [f64; 3] {
        self.rates
    }

    pub fn last_renewal_mono_ns(&self) -> u64 {
        self.last_renewal_mono_ns
    }

    pub fn renew(&mut self, now_mono_ns: u64) {
        self.last_renewal_mono_ns = now_mono_ns;
    }

    /// A MoveAxis (also a renewal).
    pub fn demand(&mut self, axis: u8, rate_deg_s: f64, now_mono_ns: u64) {
        if let Some(rate) = self.rates.get_mut(usize::from(axis)) {
            *rate = rate_deg_s;
        }
        self.active = true;
        self.renew(now_mono_ns);
    }

    /// Stop everything; returns the axes that were moving.
    pub fn stop_all(&mut self) -> Vec<u8> {
        let moving = (0..AXES)
            .filter(|axis| self.rates[usize::from(*axis)] != 0.0)
            .collect();
        self.rates = [0.0; 3];
        self.active = false;
        moving
    }
}

/// The jog operation. One per device; further axes are absorbed.
pub struct JogOp {
    shared: Arc<MountShared>,
    meta: CommandMeta,
    axis: u8,
    rate: f64,
    applied: [f64; 3],
}

impl JogOp {
    pub fn new(
        shared: Arc<MountShared>,
        meta: CommandMeta,
        axis: u8,
        rate_deg_s: f64,
    ) -> DeviceResult<Self> {
        let rate = validated_rate(&shared.capabilities(), axis, rate_deg_s)?;
        Ok(Self {
            shared,
            meta,
            axis,
            rate,
            applied: [0.0; 3],
        })
    }

    /// Bring the hardware to the demanded rates.
    fn apply(&mut self, drv: &mut DynMount, demand: [f64; 3]) -> DeviceResult<()> {
        for axis in 0..AXES {
            let i = usize::from(axis);
            if demand[i] != self.applied[i] {
                drv.move_axis(axis, demand[i])?;
                self.applied[i] = demand[i];
                self.shared.emit(
                    EventKind::MoveAxis {
                        axis,
                        rate_deg_s: demand[i],
                    },
                    Some(self.meta.command_id),
                );
            }
        }
        Ok(())
    }

    fn stop(&mut self, drv: &mut DynMount) -> DeviceResult<()> {
        self.shared.with_jog(|jog| jog.stop_all());
        let mut first_error = None;
        for axis in 0..AXES {
            let i = usize::from(axis);
            if self.applied[i] != 0.0 {
                match drv.move_axis(axis, 0.0) {
                    Ok(()) => {
                        self.applied[i] = 0.0;
                        self.shared.emit(
                            EventKind::MoveAxis {
                                axis,
                                rate_deg_s: 0.0,
                            },
                            Some(self.meta.command_id),
                        );
                    }
                    Err(err) => {
                        first_error.get_or_insert(err);
                    }
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

impl Operation<DynMount> for JogOp {
    fn kind(&self) -> OpKind {
        OpKind::MoveAxis
    }

    /// Bounded by the dead-man in practice; this is the outer limit.
    fn deadline(&self) -> Duration {
        self.shared.settings().slew_timeout
    }

    fn start(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> DeviceResult<()> {
        if self.rate != 0.0 && is_parked(&drv.state()?) {
            return Err(parked_error("moving an axis"));
        }
        let now = clock.now_mono_ns();
        let (axis, rate) = (self.axis, self.rate);
        let demand = self.shared.with_jog(|jog| {
            jog.stop_all();
            jog.demand(axis, rate, now);
            jog.rates()
        });
        if let Err(err) = self.apply(drv, demand) {
            self.shared.with_jog(|jog| jog.stop_all());
            return Err(err);
        }
        Ok(())
    }

    fn step(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> OpProgress {
        let now = clock.now_mono_ns();
        let demand = self.shared.with_jog(|jog| *jog);
        let deadman = duration_ns(self.shared.settings().jog_deadman);
        let lapse_at = demand.last_renewal_mono_ns().saturating_add(deadman);
        if now > lapse_at {
            let stopped = self.stop(drv);
            return OpProgress::Done(OpOutcome::Failed(match stopped {
                Ok(()) => DeviceError::timeout(
                    "jog_deadman_lapsed: no renewal within jog_deadman_ms; the axes were stopped",
                ),
                Err(err) => DeviceError::timeout(format!(
                    "jog_deadman_lapsed: no renewal within jog_deadman_ms; stopping failed: {err}"
                )),
            }));
        }
        if let Err(err) = self.apply(drv, demand.rates()) {
            let _ = self.stop(drv);
            return OpProgress::Done(OpOutcome::Failed(err));
        }
        if self.applied.iter().all(|rate| *rate == 0.0) {
            self.shared.with_jog(|jog| jog.stop_all());
            return OpProgress::Done(OpOutcome::Succeeded);
        }
        OpProgress::Continue {
            next_poll: ACTIVE_POLL.min(Duration::from_nanos(lapse_at.saturating_sub(now).max(1))),
        }
    }

    fn cancel(&mut self, drv: &mut DynMount) -> DeviceResult<()> {
        self.stop(drv)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::super::{build_operation, MountPlan};
    use super::*;
    use crate::equipment::commands::MountCmd;
    use crate::equipment::sim::mount::SimMountProfile;
    use crate::equipment::traits::Device;
    use crate::equipment::types::ErrorClass;

    #[test]
    fn rates_clamp_into_axis_rates_with_their_sign() {
        let ranges = [
            RateRange {
                min_deg_s: 0.1,
                max_deg_s: 1.0,
            },
            RateRange {
                min_deg_s: 2.0,
                max_deg_s: 4.0,
            },
        ];
        assert_eq!(clamp_axis_rate(0.5, &ranges).unwrap(), 0.5);
        assert_eq!(clamp_axis_rate(-9.0, &ranges).unwrap(), -4.0);
        assert_eq!(clamp_axis_rate(0.01, &ranges).unwrap(), 0.1);
        assert_eq!(clamp_axis_rate(1.4, &ranges).unwrap(), 1.0);
        assert_eq!(clamp_axis_rate(1.6, &ranges).unwrap(), 2.0);
        assert_eq!(clamp_axis_rate(0.0, &ranges).unwrap(), 0.0);
        assert!(clamp_axis_rate(1.0, &[]).is_err());
        assert!(clamp_axis_rate(f64::NAN, &ranges).is_err());
    }

    #[test]
    fn a_jog_needs_can_move_axis_not_can_slew() {
        // The ESP32 profile reports CanSlew = false (O10); jog still works.
        let rig = rig(SimMountProfile::esp32_like());
        let caps = rig.runtime.shared().capabilities();
        assert!(!caps.can_slew);
        assert_eq!(validated_rate(&caps, 0, 9.0).unwrap(), 4.0);
        assert_eq!(
            validated_rate(&caps, 2, 1.0).unwrap_err().class,
            ErrorClass::Unsupported
        );
        assert_eq!(
            validated_rate(&caps, 5, 1.0).unwrap_err().class,
            ErrorClass::InvalidValue
        );
    }

    #[test]
    fn the_dead_man_stops_the_axes_when_renewals_lapse() {
        let mut rig = rig(SimMountProfile::esp32_like());
        let meta = operator();
        rig.runtime.take_events();
        let mut op = build_operation(
            &rig.runtime.shared(),
            MountCmd::MoveAxis {
                axis: 0,
                rate_deg_s: 2.0,
            },
            meta.clone(),
        )
        .unwrap();
        op.start(&mut rig.sim, rig.clock.as_ref()).unwrap();
        assert_eq!(rig.sim.control().axis_rates()[0], 2.0);
        assert!(rig.runtime.shared().jog_active());
        // Renewals every 500 ms keep it alive for 3 s.
        for _ in 0..6 {
            assert_eq!(advance(&mut rig, &mut op, &meta, 2), None);
            assert!(matches!(
                rig.runtime
                    .plan(MountCmd::RenewJog, meta.clone(), Some(OpKind::MoveAxis)),
                MountPlan::Absorbed(OpOutcome::Succeeded)
            ));
        }
        assert_eq!(rig.sim.control().axis_rates()[0], 2.0);
        // No renewal: it lapses within the dead-man plus one active poll.
        let lapsed_at = rig.clock.now_mono_ns();
        let outcome = advance(&mut rig, &mut op, &meta, 8).unwrap();
        assert!(
            matches!(&outcome, OpOutcome::Failed(e) if e.class == ErrorClass::Timeout && e.message.starts_with("jog_deadman_lapsed"))
        );
        assert!(rig.clock.now_mono_ns() - lapsed_at <= 1_250_000_000);
        assert_eq!(rig.sim.control().axis_rates(), [0.0; 3]);
        assert!(!rig.runtime.shared().jog_active());
        let events = rig.runtime.take_events();
        let moves: Vec<_> = events
            .iter()
            .filter_map(|e| match e.event {
                EventKind::MoveAxis { axis, rate_deg_s } => Some((axis, rate_deg_s)),
                _ => None,
            })
            .collect();
        assert_eq!(moves, vec![(0, 2.0), (0, 0.0)]);
    }

    #[test]
    fn a_second_axis_and_a_stop_are_absorbed_by_the_active_jog() {
        let mut rig = rig(SimMountProfile::esp32_like());
        let meta = operator();
        let mut op = build_operation(
            &rig.runtime.shared(),
            MountCmd::MoveAxis {
                axis: 0,
                rate_deg_s: 1.0,
            },
            meta.clone(),
        )
        .unwrap();
        op.start(&mut rig.sim, rig.clock.as_ref()).unwrap();
        assert!(matches!(
            rig.runtime.plan(
                MountCmd::MoveAxis {
                    axis: 1,
                    rate_deg_s: -0.5
                },
                meta.clone(),
                Some(OpKind::MoveAxis)
            ),
            MountPlan::Absorbed(_)
        ));
        assert_eq!(advance(&mut rig, &mut op, &meta, 1), None);
        assert_eq!(rig.sim.control().axis_rates(), [1.0, -0.5, 0.0]);
        for axis in [0, 1] {
            rig.runtime.plan(
                MountCmd::MoveAxis {
                    axis,
                    rate_deg_s: 0.0,
                },
                meta.clone(),
                Some(OpKind::MoveAxis),
            );
        }
        assert_eq!(
            advance(&mut rig, &mut op, &meta, 1),
            Some(OpOutcome::Succeeded)
        );
        assert_eq!(rig.sim.control().axis_rates(), [0.0; 3]);
    }

    #[test]
    fn cancel_and_the_abort_path_stop_a_jog() {
        let mut rig = rig(SimMountProfile::esp32_like());
        let meta = operator();
        let mut op = build_operation(
            &rig.runtime.shared(),
            MountCmd::MoveAxis {
                axis: 1,
                rate_deg_s: 3.0,
            },
            meta.clone(),
        )
        .unwrap();
        op.start(&mut rig.sim, rig.clock.as_ref()).unwrap();
        rig.sim.abort_path().unwrap().abort().unwrap();
        op.cancel(&mut rig.sim).unwrap();
        assert_eq!(rig.sim.control().axis_rates(), [0.0; 3]);
        assert!(!rig.runtime.shared().jog_active());
    }

    #[test]
    fn jogs_are_refused_while_parked() {
        let mut profile = SimMountProfile::esp32_like();
        profile.start_parked = true;
        let mut rig = rig(profile);
        let outcome = rig
            .run(
                MountCmd::MoveAxis {
                    axis: 0,
                    rate_deg_s: 1.0,
                },
                operator(),
            )
            .unwrap();
        assert!(
            matches!(outcome, OpOutcome::Failed(e) if e.class == ErrorClass::InvalidWhileParked)
        );
    }

    #[test]
    fn jog_flags_track_moving_axes() {
        let flags = JogFlags::new();
        flags.set(0, 1.0);
        flags.set(2, -1.0);
        assert_eq!(flags.moving_axes(), vec![0, 2]);
        flags.set(0, 0.0);
        assert_eq!(flags.moving_axes(), vec![2]);
        flags.clear();
        assert!(flags.moving_axes().is_empty());
        flags.set(7, 1.0);
        assert!(flags.moving_axes().is_empty());
    }
}

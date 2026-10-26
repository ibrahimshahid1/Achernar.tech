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
//! - A zero-rate MoveAxis always reaches the hardware, whether it starts a
//!   one-shot stop (no jog running) or is absorbed by a running jog, so it
//!   stops an axis another client started or a failed stop left moving. A
//!   one-shot stop ends at its first step, after sending any further
//!   MoveAxis 0 absorbed meanwhile (e-stop stops each axis this way). A stop
//!   that fails leaves that axis's rate in the shared [`JogDemand`], so
//!   `AbortSlew`, a forced disconnect and shutdown still send it MoveAxis 0.
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
use crate::equipment::traits::{Clock, MountDriver};
use crate::equipment::types::{CommandId, DeviceError, DeviceResult, ErrorClass, OpOutcome};

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

/// A rate the hardware may be moving at but nobody knows: a MoveAxis whose
/// result is unknown. It compares unequal to every rate, so the next apply
/// or stop always sends.
pub const UNKNOWN_RATE: f64 = f64::NAN;

/// What the operator wants the axes to do, and when they last said so.
///
/// An axis keeps its non-zero rate until a MoveAxis 0 to it succeeds, even
/// after the jog operation ends: a stop that failed is then retried by the
/// next zero-rate MoveAxis, `AbortSlew`, a forced disconnect or shutdown.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct JogDemand {
    rates: [f64; 3],
    last_renewal_mono_ns: u64,
    active: bool,
    /// A zero-rate MoveAxis was absorbed for this axis: the running jog
    /// sends it even if it never moved that axis (another client may have).
    resend: [bool; 3],
}

impl JogDemand {
    /// A jog operation is running.
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

    /// A MoveAxis (also a renewal). A zero rate is marked for resending,
    /// so the running jog sends it even to an axis it never moved.
    pub fn demand(&mut self, axis: u8, rate_deg_s: f64, now_mono_ns: u64) {
        let i = usize::from(axis);
        if let Some(rate) = self.rates.get_mut(i) {
            *rate = rate_deg_s;
            self.resend[i] = rate_deg_s == 0.0;
        }
        self.active = true;
        self.renew(now_mono_ns);
    }

    /// A new jog: `axis` at `rate_deg_s`, every other axis at 0. Returns the
    /// previous rates, which may include axes a failed stop left moving;
    /// the new jog must bring those to 0 too. The new jog sends the
    /// commanded axis itself, so nothing is left marked for resending.
    pub fn begin(&mut self, axis: u8, rate_deg_s: f64, now_mono_ns: u64) -> [f64; 3] {
        let previous = self.rates;
        self.rates = [0.0; 3];
        self.demand(axis, rate_deg_s, now_mono_ns);
        self.resend = [false; 3];
        previous
    }

    /// Axes with a zero-rate MoveAxis still to send; clears the marks.
    pub fn take_resend(&mut self) -> [bool; 3] {
        std::mem::take(&mut self.resend)
    }

    /// Axes that may be moving: a non-zero (or unknown) rate.
    pub fn moving_axes(&self) -> Vec<u8> {
        (0..AXES)
            .filter(|axis| self.rates[usize::from(*axis)] != 0.0)
            .collect()
    }

    /// A MoveAxis 0 to `axis` succeeded.
    pub fn stopped(&mut self, axis: u8) {
        let i = usize::from(axis);
        if let Some(rate) = self.rates.get_mut(i) {
            *rate = 0.0;
            self.resend[i] = false;
        }
    }

    /// A stop of `axis` failed: remember that it may still move at
    /// `rate_deg_s` so a later stop retries it.
    pub fn left_moving(&mut self, axis: u8, rate_deg_s: f64) {
        if let Some(rate) = self.rates.get_mut(usize::from(axis)) {
            *rate = if rate_deg_s == 0.0 {
                UNKNOWN_RATE
            } else {
                rate_deg_s
            };
        }
    }

    /// The jog operation ended. Axes whose stop failed keep their rate.
    pub fn end(&mut self) {
        self.active = false;
        self.resend = [false; 3];
    }
}

/// Send MoveAxis 0 to each of `axes`, recording every success in the shared
/// demand. Every axis is attempted; the first error is returned.
pub(crate) fn stop_axes(
    shared: &MountShared,
    drv: &mut dyn MountDriver,
    axes: &[u8],
    command_id: Option<CommandId>,
) -> DeviceResult<()> {
    let mut first_error = None;
    for &axis in axes {
        match drv.move_axis(axis, 0.0) {
            Ok(()) => {
                shared.with_jog(|jog| jog.stopped(axis));
                shared.emit(
                    EventKind::MoveAxis {
                        axis,
                        rate_deg_s: 0.0,
                    },
                    command_id,
                );
            }
            Err(err) => {
                tracing::warn!("could not stop mount axis {axis}: {err}");
                first_error.get_or_insert(err);
            }
        }
    }
    first_error.map_or(Ok(()), Err)
}

/// E-stop's axis step (§2.9 step 1): MoveAxis 0 on every CanMoveAxis axis,
/// whoever moved it. A parked mount refusing MoveAxis is not a failure (a
/// parked mount is not moving an axis); any other error is returned after
/// every axis was attempted.
pub(crate) fn stop_movable_axes(
    shared: &MountShared,
    drv: &mut dyn MountDriver,
    command_id: Option<CommandId>,
) -> DeviceResult<()> {
    let caps = shared.capabilities();
    let axes: Vec<u8> = (0..AXES)
        .filter(|axis| caps.can_move_axis[usize::from(*axis)])
        .collect();
    let mut first_error = None;
    for axis in axes {
        match stop_axes(shared, drv, &[axis], command_id) {
            Err(err) if err.class != ErrorClass::InvalidWhileParked => {
                first_error.get_or_insert(err);
            }
            _ => {}
        }
    }
    first_error.map_or(Ok(()), Err)
}

/// The jog operation. One per device; further axes are absorbed.
///
/// A zero-rate MoveAxis with no jog running is a one-shot stop: it always
/// sends MoveAxis(axis, 0), since the axis may have been started by another
/// client or left moving by a stop that failed. It ends `Succeeded` at its
/// first step, after sending any further zero-rate MoveAxis absorbed
/// meanwhile; a stop needs no dead-man renewal.
pub struct JogOp {
    shared: Arc<MountShared>,
    meta: CommandMeta,
    axis: u8,
    rate: f64,
    /// The rate each axis was last commanded to; [`UNKNOWN_RATE`] when that
    /// is not known (the next apply or stop always sends).
    applied: [f64; 3],
    finished: bool,
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
            finished: false,
        })
    }

    /// Zero-rate MoveAxis commands absorbed since the last call are sent
    /// whatever this jog last applied to their axes (another client may
    /// have moved them).
    fn take_resend(&mut self) {
        let resend = self.shared.with_jog(|jog| jog.take_resend());
        for (applied, again) in self.applied.iter_mut().zip(resend) {
            if again {
                *applied = UNKNOWN_RATE;
            }
        }
    }

    /// Bring the hardware to the demanded rates.
    fn apply(&mut self, drv: &mut DynMount, demand: [f64; 3]) -> DeviceResult<()> {
        for axis in 0..AXES {
            let i = usize::from(axis);
            // `!=` is also true when either side is UNKNOWN_RATE.
            if demand[i] != self.applied[i] {
                if let Err(err) = drv.move_axis(axis, demand[i]) {
                    // The axis may or may not have changed rate.
                    self.applied[i] = UNKNOWN_RATE;
                    return Err(err);
                }
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

    /// MoveAxis 0 on every axis this jog may have moving, then end the jog.
    /// The shared demand is cleared only for axes that acknowledged the
    /// stop; the others keep their rate so a later stop retries them.
    fn stop(&mut self, drv: &mut DynMount) -> DeviceResult<()> {
        self.take_resend();
        let mut first_error = None;
        for axis in 0..AXES {
            let i = usize::from(axis);
            if self.applied[i] == 0.0 {
                // Demanded but never applied: the hardware was not moved.
                self.shared.with_jog(|jog| jog.stopped(axis));
                continue;
            }
            match drv.move_axis(axis, 0.0) {
                Ok(()) => {
                    self.applied[i] = 0.0;
                    self.shared.with_jog(|jog| jog.stopped(axis));
                    self.shared.emit(
                        EventKind::MoveAxis {
                            axis,
                            rate_deg_s: 0.0,
                        },
                        Some(self.meta.command_id),
                    );
                }
                Err(err) => {
                    let rate = self.applied[i];
                    self.shared.with_jog(|jog| jog.left_moving(axis, rate));
                    tracing::warn!("could not stop mount axis {axis}: {err}");
                    first_error.get_or_insert(err);
                }
            }
        }
        self.shared.with_jog(|jog| jog.end());
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
        let (previous, demand) = self.shared.with_jog(|jog| {
            let previous = jog.begin(axis, rate, now);
            (previous, jog.rates())
        });
        // Axes a failed stop left moving are brought to 0 by the apply below.
        // The commanded axis is always sent, even at rate 0 and even when
        // the demand already says so: another client, or a stop whose
        // result was lost, may have left it moving.
        self.applied = previous;
        self.applied[usize::from(axis)] = UNKNOWN_RATE;
        if let Err(err) = self.apply(drv, demand) {
            let _ = self.stop(drv);
            return Err(err);
        }
        // A stop with no jog running (every axis now at 0) ends at the first
        // step, which first sends any zero-rate MoveAxis absorbed meanwhile.
        Ok(())
    }

    fn step(&mut self, drv: &mut DynMount, clock: &dyn Clock) -> OpProgress {
        if self.finished {
            return OpProgress::Done(OpOutcome::Succeeded);
        }
        let now = clock.now_mono_ns();
        self.take_resend();
        let demand = self.shared.with_jog(|jog| *jog);
        let deadman = duration_ns(self.shared.settings().jog_deadman);
        let lapse_at = demand.last_renewal_mono_ns().saturating_add(deadman);
        // Every axis demanded at 0 is a stop, which needs no renewal.
        let stopping = demand.rates().iter().all(|rate| *rate == 0.0);
        if now > lapse_at && !stopping {
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
            self.shared.with_jog(|jog| jog.end());
            self.finished = true;
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
    fn a_zero_rate_move_axis_always_reaches_the_hardware() {
        let mut rig = rig(SimMountProfile::esp32_like());
        // Another client (or a lost stop) has the axis moving.
        rig.sim.move_axis(0, 2.0).unwrap();
        let before = rig.sim.control().calls().len();
        let outcome = rig
            .run(
                MountCmd::MoveAxis {
                    axis: 0,
                    rate_deg_s: 0.0,
                },
                operator(),
            )
            .unwrap();
        assert_eq!(outcome, OpOutcome::Succeeded);
        assert_eq!(rig.sim.control().axis_rates(), [0.0; 3]);
        assert_eq!(
            rig.sim.control().calls()[before..]
                .iter()
                .filter(|c| c.starts_with("move_axis"))
                .collect::<Vec<_>>(),
            vec!["move_axis:0:0"]
        );
        assert!(!rig.runtime.shared().jog_active());
    }

    #[test]
    fn an_absorbed_zero_rate_move_axis_reaches_an_axis_the_jog_never_moved() {
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
        // Another client moves axis 1; the operator stops it mid-jog.
        rig.sim.move_axis(1, 2.0).unwrap();
        assert!(matches!(
            rig.runtime.plan(
                MountCmd::MoveAxis {
                    axis: 1,
                    rate_deg_s: 0.0
                },
                operator(),
                Some(OpKind::MoveAxis)
            ),
            MountPlan::Absorbed(OpOutcome::Succeeded)
        ));
        assert_eq!(advance(&mut rig, &mut op, &meta, 1), None, "axis 0 jogs on");
        assert_eq!(rig.sim.control().axis_rates(), [1.0, 0.0, 0.0]);
        op.cancel(&mut rig.sim).unwrap();
        assert_eq!(rig.sim.control().axis_rates(), [0.0; 3]);
    }

    fn failed_stop(rig: &mut Rig, meta: &CommandMeta) {
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
        rig.sim
            .control()
            .update_faults(|f| f.move_axis_fails = true);
        // The dead-man lapses and the stop fails: the axis keeps moving.
        let outcome = advance(rig, &mut op, meta, 8).unwrap();
        assert!(matches!(&outcome, OpOutcome::Failed(e) if e.message.contains("stopping failed")));
        assert_eq!(rig.sim.control().axis_rates()[0], 2.0);
        let shared = rig.runtime.shared();
        assert!(!shared.jog_active(), "the jog operation ended");
        assert_eq!(shared.with_jog(|jog| jog.moving_axes()), vec![0]);
        rig.sim
            .control()
            .update_faults(|f| f.move_axis_fails = false);
    }

    #[test]
    fn a_failed_stop_is_retried_by_a_later_zero_rate_move_axis() {
        let mut rig = rig(SimMountProfile::esp32_like());
        let meta = operator();
        failed_stop(&mut rig, &meta);
        let outcome = rig
            .run(
                MountCmd::MoveAxis {
                    axis: 0,
                    rate_deg_s: 0.0,
                },
                operator(),
            )
            .unwrap();
        assert_eq!(outcome, OpOutcome::Succeeded);
        assert_eq!(rig.sim.control().axis_rates(), [0.0; 3]);
        assert!(rig
            .runtime
            .shared()
            .with_jog(|jog| jog.moving_axes())
            .is_empty());
    }

    #[test]
    fn a_failed_stop_is_retried_by_a_new_jog_on_another_axis() {
        let mut rig = rig(SimMountProfile::esp32_like());
        let meta = operator();
        failed_stop(&mut rig, &meta);
        let mut op = build_operation(
            &rig.runtime.shared(),
            MountCmd::MoveAxis {
                axis: 1,
                rate_deg_s: 1.0,
            },
            meta.clone(),
        )
        .unwrap();
        op.start(&mut rig.sim, rig.clock.as_ref()).unwrap();
        assert_eq!(rig.sim.control().axis_rates(), [0.0, 1.0, 0.0]);
        op.cancel(&mut rig.sim).unwrap();
        assert_eq!(rig.sim.control().axis_rates(), [0.0; 3]);
    }

    #[test]
    fn shutdown_and_abort_stop_an_axis_a_failed_stop_left_moving() {
        for path in ["shutdown", "disconnect", "abort"] {
            let mut rig = rig(SimMountProfile::esp32_like());
            let meta = operator();
            failed_stop(&mut rig, &meta);
            match path {
                "shutdown" => rig.runtime.shutdown(&mut rig.sim, None).unwrap(),
                "disconnect" => rig.runtime.prepare_disconnect(&mut rig.sim).unwrap(),
                _ => assert_eq!(
                    rig.run(MountCmd::AbortSlew, operator()).unwrap(),
                    OpOutcome::Succeeded
                ),
            }
            assert_eq!(rig.sim.control().axis_rates(), [0.0; 3], "{path}");
            assert!(
                rig.runtime
                    .shared()
                    .with_jog(|jog| jog.moving_axes())
                    .is_empty(),
                "{path}"
            );
        }
    }

    #[test]
    fn shutdown_during_a_jog_stops_every_movable_axis() {
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
        // Axis 1 was started by another client meanwhile.
        rig.sim.move_axis(1, 2.0).unwrap();
        rig.runtime
            .shutdown(&mut rig.sim, Some(OpKind::MoveAxis))
            .unwrap();
        assert_eq!(rig.sim.control().axis_rates(), [0.0; 3]);
        let calls = rig.sim.control().calls();
        assert!(calls.contains(&"move_axis:1:0".to_string()), "{calls:?}");
        assert!(
            !calls.iter().any(|c| c.starts_with("move_axis:2")),
            "axis 2 cannot move"
        );
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

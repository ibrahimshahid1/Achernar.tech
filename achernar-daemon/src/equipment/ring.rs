//! Timestamped per-device ring buffers (spec §2.5, §4.1, D7). Owner: WP1A-2.
//!
//! Every published device snapshot is pushed into its kind's [`Ring`]
//! (default `ring_buffer_minutes = 30`), and `window()` and `quiescent()`
//! are answered from the rings and the event journal alone: they never call a
//! device (§4.1).
//!
//! [`DeviceSnapshot`] is the small common view of the ten typed snapshots the
//! worker, the hub and the window assembly need: identity, publication time,
//! the active operation, and a motion predicate. Ring queries scan linearly
//! by UTC, so a host clock step never breaks them; at 10 Hz for 30 minutes a
//! scan touches at most 18 000 entries, once per exposure.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use super::events::EquipmentEvent;
use super::ops::{ActiveOp, OpKind, OpPhase};
use super::snapshot::{
    CoverCalibratorSnapshot, CoverState, DomeSnapshot, FilterWheelSnapshot, FocuserSnapshot,
    GuiderSnapshot, MountSnapshot, RotatorSnapshot, SafetySnapshot, ShutterState, SwitchSnapshot,
    WeatherSnapshot, WindowSamples,
};
use super::types::{DeviceKind, EqCoord};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// Timestamps and the common snapshot view
// ---------------------------------------------------------------------------

/// Anything stored in a ring: a UTC time for window queries and a monotonic
/// time for retention.
pub trait Timestamped {
    fn at_utc(&self) -> DateTime<Utc>;
    fn at_mono_ns(&self) -> u64;
}

impl Timestamped for EquipmentEvent {
    fn at_utc(&self) -> DateTime<Utc> {
        self.at_utc
    }

    fn at_mono_ns(&self) -> u64 {
        self.mono_ns
    }
}

/// The common view of a published device snapshot.
pub trait DeviceSnapshot: Timestamped + Clone + Send + Sync + 'static {
    const KIND: DeviceKind;

    fn connection_id(&self) -> Uuid;
    fn set_connection_id(&mut self, id: Uuid);
    fn set_published(&mut self, at: DateTime<Utc>, mono_ns: u64);
    /// `None` for kinds whose snapshot carries no operation (weather, safety).
    fn active_op(&self) -> Option<&ActiveOp>;
    /// A no-op for kinds whose snapshot carries no operation.
    fn set_active_op(&mut self, op: Option<ActiveOp>);
    /// What the device itself reports about motion (Slewing, IsMoving, a
    /// moving wheel or cover). `None` when the device has not said. Never
    /// derived from the active operation, so a poll after a cancel can
    /// confirm that motion stopped.
    fn device_moving(&self) -> Option<bool>;

    /// True while the device moves, settles, or runs a hardware-moving
    /// operation. Quiescence and the busy set use this.
    fn in_motion(&self) -> bool {
        self.device_moving() == Some(true)
            || self
                .active_op()
                .is_some_and(|op| op.kind.moves_hardware() || op.phase == OpPhase::Settling)
    }
}

macro_rules! snapshot_basics {
    ($ty:ty) => {
        fn connection_id(&self) -> Uuid {
            self.connection_id
        }

        fn set_connection_id(&mut self, id: Uuid) {
            self.connection_id = id;
        }

        fn set_published(&mut self, at: DateTime<Utc>, mono_ns: u64) {
            self.published_at = at;
            self.published_mono_ns = mono_ns;
        }
    };
}

macro_rules! timestamped_snapshot {
    ($($ty:ty),* $(,)?) => {
        $(
            impl Timestamped for $ty {
                fn at_utc(&self) -> DateTime<Utc> {
                    self.published_at
                }

                fn at_mono_ns(&self) -> u64 {
                    self.published_mono_ns
                }
            }
        )*
    };
}

timestamped_snapshot!(
    MountSnapshot,
    FocuserSnapshot,
    FilterWheelSnapshot,
    GuiderSnapshot,
    RotatorSnapshot,
    CoverCalibratorSnapshot,
    SwitchSnapshot,
    WeatherSnapshot,
    SafetySnapshot,
    DomeSnapshot,
);

macro_rules! active_op_field {
    () => {
        fn active_op(&self) -> Option<&ActiveOp> {
            self.active_op.as_ref()
        }

        fn set_active_op(&mut self, op: Option<ActiveOp>) {
            self.active_op = op;
        }
    };
}

impl DeviceSnapshot for MountSnapshot {
    const KIND: DeviceKind = DeviceKind::Mount;
    snapshot_basics!(MountSnapshot);
    active_op_field!();

    /// PHD2's pulses are not motion here: guiding during an exposure is
    /// expected (D20). Achernar's own pulses count through the active op.
    fn device_moving(&self) -> Option<bool> {
        if self.jog_active {
            return Some(true);
        }
        self.state.slewing.known().copied()
    }
}

impl DeviceSnapshot for FocuserSnapshot {
    const KIND: DeviceKind = DeviceKind::Focuser;
    snapshot_basics!(FocuserSnapshot);
    active_op_field!();

    fn device_moving(&self) -> Option<bool> {
        self.state.is_moving.known().copied()
    }

    fn in_motion(&self) -> bool {
        self.settling
            || self.device_moving() == Some(true)
            || self
                .active_op
                .as_ref()
                .is_some_and(|op| op.kind.moves_hardware() || op.phase == OpPhase::Settling)
    }
}

impl DeviceSnapshot for FilterWheelSnapshot {
    const KIND: DeviceKind = DeviceKind::FilterWheel;
    snapshot_basics!(FilterWheelSnapshot);
    active_op_field!();

    fn device_moving(&self) -> Option<bool> {
        if self.moving {
            Some(true)
        } else if self.position.known().is_some() {
            Some(false)
        } else {
            None
        }
    }
}

impl DeviceSnapshot for GuiderSnapshot {
    const KIND: DeviceKind = DeviceKind::Guider;
    snapshot_basics!(GuiderSnapshot);
    active_op_field!();

    /// A guider "moves" the mount while it dithers or settles.
    fn device_moving(&self) -> Option<bool> {
        if self.settling {
            Some(true)
        } else {
            self.app_state.known().map(|_| false)
        }
    }

    fn in_motion(&self) -> bool {
        self.settling
            || self
                .active_op
                .as_ref()
                .is_some_and(|op| op.kind == OpKind::Dither)
    }
}

impl DeviceSnapshot for RotatorSnapshot {
    const KIND: DeviceKind = DeviceKind::Rotator;
    snapshot_basics!(RotatorSnapshot);
    active_op_field!();

    fn device_moving(&self) -> Option<bool> {
        self.state.is_moving.known().copied()
    }
}

impl DeviceSnapshot for CoverCalibratorSnapshot {
    const KIND: DeviceKind = DeviceKind::CoverCalibrator;
    snapshot_basics!(CoverCalibratorSnapshot);
    active_op_field!();

    fn device_moving(&self) -> Option<bool> {
        match (
            self.state.cover_moving.known(),
            self.state.cover_state.known(),
        ) {
            (Some(true), _) | (_, Some(CoverState::Moving)) => Some(true),
            (Some(false), _) => Some(false),
            (None, Some(_)) => Some(false),
            (None, None) => None,
        }
    }
}

impl DeviceSnapshot for SwitchSnapshot {
    const KIND: DeviceKind = DeviceKind::Switch;
    snapshot_basics!(SwitchSnapshot);
    active_op_field!();

    fn device_moving(&self) -> Option<bool> {
        Some(false)
    }
}

impl DeviceSnapshot for DomeSnapshot {
    const KIND: DeviceKind = DeviceKind::Dome;
    snapshot_basics!(DomeSnapshot);
    active_op_field!();

    fn device_moving(&self) -> Option<bool> {
        let shutter_moving = self
            .state
            .shutter
            .known()
            .map(|s| matches!(s, ShutterState::Opening | ShutterState::Closing));
        match (self.state.slewing.known().copied(), shutter_moving) {
            (Some(true), _) | (_, Some(true)) => Some(true),
            (Some(false), _) | (None, Some(false)) => Some(false),
            (None, None) => None,
        }
    }
}

impl DeviceSnapshot for WeatherSnapshot {
    const KIND: DeviceKind = DeviceKind::Weather;
    snapshot_basics!(WeatherSnapshot);

    fn active_op(&self) -> Option<&ActiveOp> {
        None
    }

    fn set_active_op(&mut self, _op: Option<ActiveOp>) {}

    fn device_moving(&self) -> Option<bool> {
        Some(false)
    }
}

impl DeviceSnapshot for SafetySnapshot {
    const KIND: DeviceKind = DeviceKind::SafetyMonitor;
    snapshot_basics!(SafetySnapshot);

    fn active_op(&self) -> Option<&ActiveOp> {
        None
    }

    fn set_active_op(&mut self, _op: Option<ActiveOp>) {}

    fn device_moving(&self) -> Option<bool> {
        Some(false)
    }
}

// ---------------------------------------------------------------------------
// The ring
// ---------------------------------------------------------------------------

struct RingInner<T> {
    items: VecDeque<Arc<T>>,
    retention: Duration,
    max_len: usize,
    last_mono: Option<u64>,
    evicted: u64,
}

/// A bounded, time-retained buffer of shared samples.
pub struct Ring<T> {
    inner: Mutex<RingInner<T>>,
}

impl<T: Timestamped> Ring<T> {
    /// Keep samples for `retention` (monotonic) and never more than
    /// `max_len` of them.
    pub fn new(retention: Duration, max_len: usize) -> Self {
        Self {
            inner: Mutex::new(RingInner {
                items: VecDeque::new(),
                retention,
                max_len: max_len.max(1),
                last_mono: None,
                evicted: 0,
            }),
        }
    }

    /// Sized for `retention` at the fastest expected cadence, plus slack.
    pub fn for_cadence(retention: Duration, fastest: Duration) -> Self {
        let fastest = fastest.max(Duration::from_millis(10));
        let n = (retention.as_secs_f64() / fastest.as_secs_f64()).ceil() as usize;
        Self::new(retention, n.saturating_add(16).min(200_000))
    }

    fn evict(inner: &mut RingInner<T>, newest_mono: u64) {
        let retention = super::clock::duration_ns(inner.retention);
        while let Some(front) = inner.items.front() {
            let too_old = front.at_mono_ns().saturating_add(retention) < newest_mono;
            if too_old || inner.items.len() > inner.max_len {
                inner.items.pop_front();
                inner.evicted += 1;
            } else {
                break;
            }
        }
    }

    /// Append a snapshot. Samples whose monotonic time is not newer than the
    /// last accepted one are ignored, so a worker and a watch follower can
    /// both push without duplicating. Returns true when stored.
    pub fn push(&self, item: Arc<T>) -> bool {
        let mut inner = lock(&self.inner);
        let mono = item.at_mono_ns();
        if inner.last_mono.is_some_and(|last| mono <= last) {
            return false;
        }
        inner.last_mono = Some(mono);
        inner.items.push_back(item);
        Self::evict(&mut inner, mono);
        true
    }

    /// Append unconditionally (events arrive in sequence order, which may
    /// differ slightly from their creation order across threads).
    pub fn append(&self, item: Arc<T>) {
        let mut inner = lock(&self.inner);
        let mono = item.at_mono_ns();
        inner.last_mono = Some(inner.last_mono.map_or(mono, |last| last.max(mono)));
        inner.items.push_back(item);
        let newest = inner.last_mono.unwrap_or(mono);
        Self::evict(&mut inner, newest);
    }

    pub fn len(&self) -> usize {
        lock(&self.inner).items.len()
    }

    pub fn is_empty(&self) -> bool {
        lock(&self.inner).items.is_empty()
    }

    /// Samples dropped by retention or size so far.
    pub fn evicted(&self) -> u64 {
        lock(&self.inner).evicted
    }

    pub fn latest(&self) -> Option<Arc<T>> {
        lock(&self.inner).items.back().cloned()
    }

    pub fn oldest(&self) -> Option<Arc<T>> {
        lock(&self.inner).items.front().cloned()
    }

    /// Every stored sample, oldest first.
    pub fn items(&self) -> Vec<Arc<T>> {
        lock(&self.inner).items.iter().cloned().collect()
    }

    /// Samples with `from <= at_utc <= to`, in storage order.
    pub fn range(&self, from: DateTime<Utc>, to: DateTime<Utc>) -> Vec<Arc<T>> {
        lock(&self.inner)
            .items
            .iter()
            .filter(|item| {
                let at = item.at_utc();
                at >= from && at <= to
            })
            .cloned()
            .collect()
    }

    /// The latest sample at or before `t`.
    pub fn at_or_before(&self, t: DateTime<Utc>) -> Option<Arc<T>> {
        lock(&self.inner)
            .items
            .iter()
            .filter(|item| item.at_utc() <= t)
            .max_by_key(|item| item.at_utc())
            .cloned()
    }

    /// The earliest sample at or after `t`.
    pub fn at_or_after(&self, t: DateTime<Utc>) -> Option<Arc<T>> {
        lock(&self.inner)
            .items
            .iter()
            .filter(|item| item.at_utc() >= t)
            .min_by_key(|item| item.at_utc())
            .cloned()
    }

    /// The sample nearest to `t`.
    pub fn nearest(&self, t: DateTime<Utc>) -> Option<Arc<T>> {
        lock(&self.inner)
            .items
            .iter()
            .min_by_key(|item| (item.at_utc() - t).num_microseconds().map(i64::abs))
            .cloned()
    }
}

impl<T: Timestamped + Clone> Ring<T> {
    /// Start, middle and end samples for an exposure window (D7).
    ///
    /// - start: the state at the start (latest sample at or before it), else
    ///   the first sample inside the window;
    /// - mid: the sample nearest the middle;
    /// - end: the first sample at or after the end. When none exists yet the
    ///   latest sample at or before the end is used and `has_post_end` is
    ///   false; the caller marks it stale once the grace period passes.
    pub fn window_samples(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> WindowPick<T> {
        let inside = self.range(start, end);
        let first_sample = self.at_or_before(start).or_else(|| inside.first().cloned());
        let mid_time = midpoint(start, end);
        let mid = self.nearest(mid_time);
        let post_end = self.at_or_after(end);
        let has_post_end = post_end.is_some();
        let end_sample = post_end.or_else(|| self.at_or_before(end));
        WindowPick {
            samples: WindowSamples {
                start: first_sample.map(|s| (*s).clone()),
                mid: mid.map(|s| (*s).clone()),
                end: end_sample.map(|s| (*s).clone()),
                end_stale: false,
                samples_n: u32::try_from(inside.len()).unwrap_or(u32::MAX),
            },
            has_post_end,
            has_data: !self.is_empty(),
        }
    }
}

/// [`Ring::window_samples`] plus what the caller needs to finalise (D7).
#[derive(Debug, Clone)]
pub struct WindowPick<T> {
    pub samples: WindowSamples<T>,
    pub has_post_end: bool,
    pub has_data: bool,
}

/// The midpoint of two instants.
pub fn midpoint(start: DateTime<Utc>, end: DateTime<Utc>) -> DateTime<Utc> {
    start + (end - start) / 2
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

/// Linear interpolation of a mount position between two samples, with RA
/// wrap (§4.1). Both coordinates must be in the same frame and `t` must lie
/// within `[t_a, t_b]`; otherwise there is nothing honest to return.
pub fn interpolate_eq(
    t_a: DateTime<Utc>,
    a: EqCoord,
    t_b: DateTime<Utc>,
    b: EqCoord,
    t: DateTime<Utc>,
) -> Option<EqCoord> {
    if a.frame != b.frame {
        return None;
    }
    let (t_a, a, t_b, b) = if t_b < t_a {
        (t_b, b, t_a, a)
    } else {
        (t_a, a, t_b, b)
    };
    if t < t_a || t > t_b {
        return None;
    }
    let span = (t_b - t_a).num_microseconds()? as f64;
    if span <= 0.0 {
        return Some(a.normalized());
    }
    let f = (t - t_a).num_microseconds()? as f64 / span;
    // Shortest way round the RA circle, in (-12, 12] hours.
    let mut dra = (b.ra_h - a.ra_h).rem_euclid(24.0);
    if dra > 12.0 {
        dra -= 24.0;
    }
    Some(EqCoord {
        ra_h: EqCoord::wrap_ra_h(a.ra_h + f * dra),
        dec_deg: a.dec_deg + f * (b.dec_deg - a.dec_deg),
        frame: a.frame,
    })
}

/// The mount position a snapshot reports in its native frame, only when both
/// axes were read fresh (a stale value is not interpolated).
pub fn mount_position(snap: &MountSnapshot) -> Option<EqCoord> {
    let ra = *snap.state.ra_h.fresh_value()?;
    let dec = *snap.state.dec_deg.fresh_value()?;
    Some(EqCoord::new(ra, dec, snap.capabilities.native_frame))
}

/// Mount RA/Dec at `t`, interpolated between the bracketing ring samples.
pub fn mount_position_at(ring: &Ring<MountSnapshot>, t: DateTime<Utc>) -> Option<EqCoord> {
    let before = ring.at_or_before(t)?;
    let after = ring.at_or_after(t)?;
    interpolate_eq(
        before.at_utc(),
        mount_position(&before)?,
        after.at_utc(),
        mount_position(&after)?,
        t,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::types::{EqFrame, Observed, Source};
    use chrono::TimeZone;

    fn t0() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap()
    }

    fn mount_at(sec: i64, ra: f64, dec: f64) -> Arc<MountSnapshot> {
        let at = t0() + chrono::Duration::seconds(sec);
        let mono = u64::try_from(sec).unwrap() * 1_000_000_000 + 1;
        let mut snap = MountSnapshot::default();
        snap.state.ra_h = Observed::fresh(ra, Source::DeviceState, at, mono);
        snap.state.dec_deg = Observed::fresh(dec, Source::DeviceState, at, mono);
        snap.state.slewing = Observed::fresh(false, Source::DeviceState, at, mono);
        snap.capabilities.native_frame = EqFrame::JNowApparent;
        snap.set_published(at, mono);
        Arc::new(snap)
    }

    #[test]
    fn interpolation_wraps_ra_across_zero_hours() {
        let a = EqCoord::new(23.9, 10.0, EqFrame::J2000);
        let b = EqCoord::new(0.1, 12.0, EqFrame::J2000);
        let ta = t0();
        let tb = t0() + chrono::Duration::seconds(10);
        let mid = interpolate_eq(ta, a, tb, b, t0() + chrono::Duration::seconds(5)).unwrap();
        assert!(
            mid.ra_h.abs() < 1e-9 || (mid.ra_h - 24.0).abs() < 1e-9,
            "{mid:?}"
        );
        assert!((mid.dec_deg - 11.0).abs() < 1e-9);
        let quarter = interpolate_eq(ta, a, tb, b, ta + chrono::Duration::seconds(2)).unwrap();
        assert!((quarter.ra_h - 23.94).abs() < 1e-9, "{quarter:?}");
        let three = interpolate_eq(ta, a, tb, b, ta + chrono::Duration::seconds(8)).unwrap();
        assert!((three.ra_h - 0.06).abs() < 1e-9, "{three:?}");
        // The other way round the circle, and argument order does not matter.
        let back = interpolate_eq(tb, b, ta, a, ta + chrono::Duration::seconds(5)).unwrap();
        assert!(back.ra_h < 1e-9 || back.ra_h > 23.999_999);
        // Different frames or a time outside the bracket give nothing.
        let other = EqCoord::new(0.1, 12.0, EqFrame::JNowApparent);
        assert_eq!(interpolate_eq(ta, a, tb, other, ta), None);
        assert_eq!(
            interpolate_eq(ta, a, tb, b, tb + chrono::Duration::seconds(1)),
            None
        );
    }

    #[test]
    fn ring_retains_by_time_and_size_and_ignores_duplicates() {
        let ring: Ring<MountSnapshot> = Ring::new(Duration::from_secs(5), 100);
        for sec in 0..10 {
            assert!(ring.push(mount_at(sec, 1.0, 2.0)));
        }
        // A watch follower re-pushing the same sample is ignored.
        assert!(!ring.push(mount_at(9, 1.0, 2.0)));
        assert_eq!(ring.len(), 6, "samples 4..=9 are within 5 s of the newest");
        assert_eq!(
            ring.oldest().unwrap().at_utc(),
            t0() + chrono::Duration::seconds(4)
        );
        let small: Ring<MountSnapshot> = Ring::new(Duration::from_secs(3600), 3);
        for sec in 0..10 {
            small.push(mount_at(sec, 1.0, 2.0));
        }
        assert_eq!(small.len(), 3);
        assert_eq!(small.evicted(), 7);
    }

    #[test]
    fn window_samples_pick_start_mid_and_post_end_states() {
        let ring: Ring<MountSnapshot> =
            Ring::for_cadence(Duration::from_secs(600), Duration::from_secs(1));
        for sec in 0..=10 {
            ring.push(mount_at(sec, 23.5 + 0.1 * sec as f64, 0.0));
        }
        let start = t0() + chrono::Duration::milliseconds(2500);
        let end = t0() + chrono::Duration::milliseconds(6500);
        let pick = ring.window_samples(start, end);
        assert!(pick.has_post_end && pick.has_data);
        let s = &pick.samples;
        assert_eq!(
            s.start.as_ref().unwrap().at_utc(),
            t0() + chrono::Duration::seconds(2)
        );
        assert_eq!(
            s.end.as_ref().unwrap().at_utc(),
            t0() + chrono::Duration::seconds(7)
        );
        assert_eq!(s.samples_n, 4);
        // Mid-exposure position, interpolated across 0h.
        let mid = mount_position_at(&ring, midpoint(start, end)).unwrap();
        assert!((mid.ra_h - 23.95).abs() < 1e-9, "{mid:?}");
        let later = mount_position_at(&ring, t0() + chrono::Duration::milliseconds(7500)).unwrap();
        assert!((later.ra_h - 0.25).abs() < 1e-9, "{later:?}");
        // No sample after the end yet: the end state is the latest known one.
        let open = ring.window_samples(start, t0() + chrono::Duration::seconds(20));
        assert!(!open.has_post_end);
        assert_eq!(
            open.samples.end.unwrap().at_utc(),
            t0() + chrono::Duration::seconds(10)
        );
    }

    #[test]
    fn stale_positions_are_never_interpolated() {
        let ring: Ring<MountSnapshot> = Ring::new(Duration::from_secs(60), 100);
        ring.push(mount_at(0, 1.0, 1.0));
        let mut stale = (*mount_at(2, 1.2, 1.0)).clone();
        stale.state.ra_h = stale.state.ra_h.clone().into_stale();
        ring.push(Arc::new(stale));
        assert_eq!(
            mount_position_at(&ring, t0() + chrono::Duration::seconds(1)),
            None
        );
    }

    #[test]
    fn motion_predicates_follow_device_state_and_operations() {
        let mut mount = (*mount_at(0, 1.0, 1.0)).clone();
        assert_eq!(mount.device_moving(), Some(false));
        assert!(!mount.in_motion());
        mount.set_active_op(Some(ActiveOp {
            command_id: crate::equipment::types::CommandId::new(),
            kind: OpKind::Slew,
            phase: OpPhase::Cancelling,
            started_at: t0(),
            started_mono_ns: 0,
            deadline_mono_ns: 1,
        }));
        // The operation keeps it in motion, but the device-reported state
        // (used to confirm a cancel) still says stopped.
        assert!(mount.in_motion());
        assert_eq!(mount.device_moving(), Some(false));
        let mut focuser = FocuserSnapshot::default();
        assert_eq!(focuser.device_moving(), None);
        focuser.settling = true;
        assert!(focuser.in_motion());
        let mut wheel = FilterWheelSnapshot::default();
        assert_eq!(wheel.device_moving(), None);
        wheel.moving = true;
        assert!(wheel.in_motion());
        assert!(!WeatherSnapshot::default().in_motion());
        let mut weather = WeatherSnapshot::default();
        weather.set_active_op(None);
        assert!(weather.active_op().is_none());
    }
}

//! The equipment hub (spec §2.10, §4.1, D6, D7, D11, H6, H14, H20, H22).
//! Owner: WP1A-2.
//!
//! [`EquipmentHub`] implements [`EquipmentReader`], the only way models read
//! equipment (D6). It holds:
//!
//! - one handle slot per device kind ([`EquipmentHub::attach`],
//!   [`EquipmentHub::handle`]); the spine (WP1B-1) keeps `AppState`'s mount,
//!   focuser and filter-wheel slots and attaches the same handles here;
//! - the [`EventBus`]: sequence numbers (continuing from
//!   `MAX(equipment_events.seq)` once a store is attached), broadcast to
//!   subscribers, the journal tail and guide-step rings, the recorder, and
//!   the connection-id filter that drops late events from an old connection
//!   (§2.8);
//! - one snapshot [`Ring`] per kind, fed by the workers and by
//!   [`EquipmentHub::refresh`], from which `window()` and `quiescent()` are
//!   answered without touching a device;
//! - the [`ResumeBlocks`] (H22), raised automatically for
//!   `outcome_unknown`, a lost mount or guider, and `limit_reached`;
//! - the [`BusyPublisher`] (H20) and the [`CommandGate`].
//!
//! [`EquipmentHub::empty`] is inert (D11): it spawns no task and opens no
//! database writer, so existing `AppState::new` callers are unaffected.
//! [`EquipmentHub::start`] spawns the 250 ms publisher, which refreshes the
//! rings, the busy set, the watchdog and the registry projection.

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
use std::time::Duration;

use chrono::{DateTime, Utc};
use tokio::sync::{broadcast, watch};
use uuid::Uuid;

use super::busy::{derive_busy, BusyPublisher};
use super::clock::SystemClock;
use super::commands::{CoverCalibratorCmd, DomeCmd, FilterWheelCmd, FocuserCmd, GuiderCmd};
use super::commands::{MountCmd, RotatorCmd, SafetyCmd, SwitchCmd, WeatherCmd};
use super::events::{EquipmentEvent, EventKind, GroundTruthRecord};
use super::gate::{CommandGate, GateConfig, GateContext};
use super::handle::{
    AbortTicket, CoverCalibratorHandle, DeviceHandle, DomeHandle, FilterWheelHandle, FocuserHandle,
    GuiderHandle, MountHandle, RotatorHandle, SafetyMonitorHandle, SwitchHandle, WeatherHandle,
};
use super::health_bridge::{HealthPolicy, RegistryBridge, Watchdog};
use super::resume::{BlockRequest, ClearError, Raised, ResumeBlocks};
use super::ring::{midpoint, mount_position_at, DeviceSnapshot, Ring, Timestamped};
use super::snapshot::{
    BusySet, CameraSummary, CocContext, CocCorroboration, CorroborationLevel,
    CoverCalibratorSnapshot, DeviceWindow, DomeSnapshot, EquipmentSnapshot, FilterWheelSnapshot,
    FocuserSnapshot, GuidePhase, GuideRms, GuideStepSample, GuideWindowStats, GuiderSnapshot,
    MotionInterval, MountSnapshot, MovedFlags, Quiescence, ResumeBlock, ResumeBlockKind,
    RotatorSnapshot, SafetyGate, SafetyLevel, SafetySnapshot, SafetySource, SafetySummary,
    SwitchSnapshot, TemperatureReading, TemperatureSourceKind, Twilight, WeatherSnapshot,
};
use super::traits::{CameraActivity, Clock, EquipmentReader, NullRecorder, RecorderSink};
use super::types::{
    kind_name, AltAz, ClockQuality, CommandSource, DeviceIdentity, DeviceKind, HealthState,
    LinkState, ObsState, Observed, Site, Source, TemperatureRole,
};
use super::worker::{default_cadence, WorkerHandle};
use crate::config::EquipmentConfigs;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

fn read<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|e| e.into_inner())
}

fn write<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|e| e.into_inner())
}

fn chrono_of(duration: Duration) -> chrono::Duration {
    chrono::Duration::from_std(duration).unwrap_or(chrono::Duration::MAX)
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Hub behaviour, from `[equipment]`, `[safety]` and `[guider]`.
#[derive(Debug, Clone, PartialEq)]
pub struct HubConfig {
    pub policy: HealthPolicy,
    /// `[equipment] ring_buffer_minutes`.
    pub ring_retention: Duration,
    /// `[equipment] estop_stops_tracking`.
    pub estop_stops_tracking: bool,
    /// `[safety] warning_is_unsafe`.
    pub warning_is_unsafe: bool,
    pub gate: GateConfig,
    /// Broadcast channel capacity (slow subscribers see `Lagged`).
    pub event_capacity: usize,
    /// Journal tail kept in memory for windows and `equipment/events`.
    pub journal_len: usize,
    /// Publisher period (§2.5: woken on change or every 250 ms).
    pub publish_interval: Duration,
    /// E-stop reports per-device results within this budget (§2.9).
    pub estop_budget: Duration,
}

impl Default for HubConfig {
    fn default() -> Self {
        Self {
            policy: HealthPolicy::default(),
            ring_retention: Duration::from_secs(30 * 60),
            estop_stops_tracking: true,
            warning_is_unsafe: true,
            gate: GateConfig::default(),
            event_capacity: 4096,
            journal_len: 50_000,
            publish_interval: Duration::from_millis(250),
            estop_budget: Duration::from_secs(2),
        }
    }
}

impl HubConfig {
    pub fn from_configs(configs: &EquipmentConfigs) -> Self {
        Self {
            policy: HealthPolicy::from_config(&configs.equipment),
            ring_retention: Duration::from_secs(configs.equipment.ring_buffer_minutes.max(1) * 60),
            estop_stops_tracking: configs.equipment.estop_stops_tracking,
            warning_is_unsafe: configs.safety.warning_is_unsafe,
            gate: GateConfig {
                allow_policy_actuation: configs.guider.allow_policy_actuation,
                ..GateConfig::default()
            },
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Ephemeris hook (H14)
// ---------------------------------------------------------------------------

/// Sun and Moon geometry at a site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkyGeometry {
    pub sun_alt_deg: f64,
    pub moon_alt_deg: f64,
    /// Illuminated fraction, 0..1.
    pub moon_illumination: f64,
}

/// Low-precision Sun and Moon (implemented by `astro/ephem.rs`, WP1A-3,
/// and installed by the spine). Without one the COC context reports the sky
/// as unknown; it is never assumed.
pub trait SkyEphemeris: Send + Sync {
    fn sky(&self, at: DateTime<Utc>, site: &Site) -> Option<SkyGeometry>;
}

/// Twilight band of a solar altitude (civil -6, nautical -12, astronomical
/// -18 degrees; day above the -0.833 degree sunrise altitude).
pub fn twilight_for_sun_alt(sun_alt_deg: f64) -> Twilight {
    if sun_alt_deg > -0.833 {
        Twilight::Day
    } else if sun_alt_deg > -6.0 {
        Twilight::Civil
    } else if sun_alt_deg > -12.0 {
        Twilight::Nautical
    } else if sun_alt_deg > -18.0 {
        Twilight::Astronomical
    } else {
        Twilight::Night
    }
}

// ---------------------------------------------------------------------------
// Event bus
// ---------------------------------------------------------------------------

/// Anything that publishes equipment events (the hub, or a test bus).
pub trait EventSink: Send + Sync {
    /// Publish an event; returns it sequenced, or `None` when it was dropped
    /// (a late event from an old connection).
    fn emit(&self, event: EquipmentEvent) -> Option<Arc<EquipmentEvent>>;
}

/// One guide step kept for window statistics (H9).
#[derive(Debug, Clone, PartialEq)]
pub struct GuideStepEntry {
    pub at_utc: DateTime<Utc>,
    pub mono_ns: u64,
    pub connection_id: Option<Uuid>,
    pub step: GuideStepSample,
}

impl Timestamped for GuideStepEntry {
    fn at_utc(&self) -> DateTime<Utc> {
        self.at_utc
    }

    fn at_mono_ns(&self) -> u64 {
        self.mono_ns
    }
}

/// Retired connection ids remembered for the late-event filter.
const RETIRED_IDS: usize = 256;

#[derive(Debug, Default)]
struct BusInner {
    next_seq: u64,
    live: BTreeMap<DeviceKind, Uuid>,
    retired: VecDeque<Uuid>,
}

/// Sequencing, fan-out and the in-memory journal (§4.2).
pub struct EventBus {
    tx: broadcast::Sender<Arc<EquipmentEvent>>,
    inner: Mutex<BusInner>,
    journal: Ring<EquipmentEvent>,
    guide_steps: Ring<GuideStepEntry>,
    recorder: Arc<dyn RecorderSink>,
    dropped_stale: AtomicU64,
}

impl std::fmt::Debug for EventBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventBus")
            .field("next_seq", &lock(&self.inner).next_seq)
            .field("journal", &self.journal.len())
            .finish()
    }
}

impl EventBus {
    pub fn new(
        recorder: Arc<dyn RecorderSink>,
        retention: Duration,
        capacity: usize,
        journal_len: usize,
    ) -> Self {
        let (tx, _) = broadcast::channel(capacity.max(16));
        Self {
            tx,
            inner: Mutex::new(BusInner {
                next_seq: 1,
                ..BusInner::default()
            }),
            journal: Ring::new(retention, journal_len.max(16)),
            guide_steps: Ring::new(retention, journal_len.max(16)),
            recorder,
            dropped_stale: AtomicU64::new(0),
        }
    }

    /// Continue numbering after `MAX(equipment_events.seq)` (WP1A-9 calls
    /// this when a store is attached). Never moves backwards.
    pub fn continue_after(&self, max_seq: u64) {
        let mut inner = lock(&self.inner);
        inner.next_seq = inner.next_seq.max(max_seq.saturating_add(1));
    }

    pub fn next_seq(&self) -> u64 {
        lock(&self.inner).next_seq
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Arc<EquipmentEvent>> {
        self.tx.subscribe()
    }

    /// The live connection of `kind`; its events pass, others' are dropped.
    pub fn set_live(&self, kind: DeviceKind, connection_id: Uuid) {
        let mut inner = lock(&self.inner);
        if let Some(previous) = inner.live.insert(kind, connection_id) {
            if previous != connection_id {
                inner.retired.push_back(previous);
            }
        }
        while inner.retired.len() > RETIRED_IDS {
            inner.retired.pop_front();
        }
    }

    /// `connection_id` ended; its late events are dropped from now on.
    pub fn retire(&self, kind: DeviceKind, connection_id: Uuid) {
        let mut inner = lock(&self.inner);
        if inner.live.get(&kind) == Some(&connection_id) {
            inner.live.remove(&kind);
        }
        if !inner.retired.contains(&connection_id) {
            inner.retired.push_back(connection_id);
        }
        while inner.retired.len() > RETIRED_IDS {
            inner.retired.pop_front();
        }
    }

    /// Events dropped by the connection-id filter.
    pub fn dropped_stale(&self) -> u64 {
        self.dropped_stale.load(Ordering::Relaxed)
    }

    fn is_stale(inner: &BusInner, event: &EquipmentEvent) -> bool {
        let (Some(kind), Some(id)) = (event.device_kind, event.connection_id) else {
            return false;
        };
        inner.retired.contains(&id) || inner.live.get(&kind).is_some_and(|live| *live != id)
    }

    /// Sequence and fan out one event.
    pub fn publish(&self, mut event: EquipmentEvent) -> Option<Arc<EquipmentEvent>> {
        let mut inner = lock(&self.inner);
        if Self::is_stale(&inner, &event) {
            drop(inner);
            self.dropped_stale.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(
                kind = event.name(),
                "dropped an event from a stale connection id"
            );
            return None;
        }
        event.seq = inner.next_seq;
        inner.next_seq += 1;
        let event = Arc::new(event);
        if let EventKind::GuideStep { step } = &event.event {
            self.guide_steps.append(Arc::new(GuideStepEntry {
                at_utc: event.at_utc,
                mono_ns: event.mono_ns,
                connection_id: event.connection_id,
                step: (**step).clone(),
            }));
        } else {
            self.journal.append(Arc::clone(&event));
        }
        // Sent under the lock so subscribers see events in sequence order.
        let _ = self.tx.send(Arc::clone(&event));
        drop(inner);
        if event.journaled() {
            self.recorder
                .record(GroundTruthRecord::Event(Arc::clone(&event)));
        }
        Some(event)
    }

    /// Journal events with `seq > since`, optionally filtered by catalogue
    /// name, at most `limit` (the `equipment/events` route caps it at 500).
    pub fn since(&self, since: u64, kinds: &[String], limit: usize) -> Vec<Arc<EquipmentEvent>> {
        self.journal
            .items()
            .into_iter()
            .filter(|event| event.seq > since)
            .filter(|event| kinds.is_empty() || kinds.iter().any(|k| k == event.name()))
            .take(limit)
            .collect()
    }

    pub fn events_between(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<Arc<EquipmentEvent>> {
        self.journal.range(start, end)
    }

    pub fn guide_steps_between(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<Arc<GuideStepEntry>> {
        self.guide_steps.range(start, end)
    }
}

impl EventSink for EventBus {
    fn emit(&self, event: EquipmentEvent) -> Option<Arc<EquipmentEvent>> {
        self.publish(event)
    }
}

// ---------------------------------------------------------------------------
// Handle slots
// ---------------------------------------------------------------------------

/// A handle of any kind.
#[derive(Debug, Clone)]
pub enum AnyHandle {
    Mount(MountHandle),
    Focuser(FocuserHandle),
    FilterWheel(FilterWheelHandle),
    Guider(GuiderHandle),
    Rotator(RotatorHandle),
    CoverCalibrator(CoverCalibratorHandle),
    Switch(SwitchHandle),
    Weather(WeatherHandle),
    SafetyMonitor(SafetyMonitorHandle),
    Dome(DomeHandle),
}

macro_rules! dispatch {
    ($any:expr, $h:ident => $body:expr) => {
        match $any {
            AnyHandle::Mount($h) => $body,
            AnyHandle::Focuser($h) => $body,
            AnyHandle::FilterWheel($h) => $body,
            AnyHandle::Guider($h) => $body,
            AnyHandle::Rotator($h) => $body,
            AnyHandle::CoverCalibrator($h) => $body,
            AnyHandle::Switch($h) => $body,
            AnyHandle::Weather($h) => $body,
            AnyHandle::SafetyMonitor($h) => $body,
            AnyHandle::Dome($h) => $body,
        }
    };
}

impl AnyHandle {
    pub fn kind(&self) -> DeviceKind {
        dispatch!(self, h => h.kind())
    }

    pub fn connection_id(&self) -> Uuid {
        dispatch!(self, h => h.connection_id())
    }

    pub fn identity(&self) -> Arc<DeviceIdentity> {
        dispatch!(self, h => h.identity())
    }

    pub fn health(&self) -> HealthState {
        dispatch!(self, h => h.health())
    }

    pub fn is_retired(&self) -> bool {
        dispatch!(self, h => h.is_retired())
    }

    /// The out-of-band abort (D19).
    pub fn abort(&self) -> AbortTicket {
        dispatch!(self, h => h.abort())
    }

    /// The latest snapshot as JSON (for the registry projection).
    pub fn snapshot_json(&self) -> serde_json::Value {
        dispatch!(self, h => serde_json::to_value(&*h.snapshot()).unwrap_or(serde_json::Value::Null))
    }

    /// Age of the latest snapshot.
    pub fn snapshot_age_ms(&self, now_mono_ns: u64) -> Option<u64> {
        let published = dispatch!(self, h => h.snapshot().at_mono_ns());
        (published > 0).then(|| now_mono_ns.saturating_sub(published) / 1_000_000)
    }

    fn push_into(&self, rings: &Rings) {
        match self {
            AnyHandle::Mount(h) => push_latest(&rings.mount, h),
            AnyHandle::Focuser(h) => push_latest(&rings.focuser, h),
            AnyHandle::FilterWheel(h) => push_latest(&rings.filter_wheel, h),
            AnyHandle::Guider(h) => push_latest(&rings.guider, h),
            AnyHandle::Rotator(h) => push_latest(&rings.rotator, h),
            AnyHandle::CoverCalibrator(h) => push_latest(&rings.cover_calibrator, h),
            AnyHandle::Switch(h) => push_latest(&rings.switch, h),
            AnyHandle::Weather(h) => push_latest(&rings.weather, h),
            AnyHandle::SafetyMonitor(h) => push_latest(&rings.safety_monitor, h),
            AnyHandle::Dome(h) => push_latest(&rings.dome, h),
        }
    }
}

/// Push a handle's latest snapshot, skipping the default (never-published)
/// one so a ring never holds a synthesised sample.
fn push_latest<C, S: DeviceSnapshot>(ring: &Ring<S>, handle: &DeviceHandle<C, S>) {
    let snapshot = handle.snapshot();
    if snapshot.at_mono_ns() > 0 {
        ring.push(snapshot);
    }
}

/// One snapshot ring per kind.
pub struct Rings {
    pub mount: Arc<Ring<MountSnapshot>>,
    pub focuser: Arc<Ring<FocuserSnapshot>>,
    pub filter_wheel: Arc<Ring<FilterWheelSnapshot>>,
    pub guider: Arc<Ring<GuiderSnapshot>>,
    pub rotator: Arc<Ring<RotatorSnapshot>>,
    pub cover_calibrator: Arc<Ring<CoverCalibratorSnapshot>>,
    pub switch: Arc<Ring<SwitchSnapshot>>,
    pub weather: Arc<Ring<WeatherSnapshot>>,
    pub safety_monitor: Arc<Ring<SafetySnapshot>>,
    pub dome: Arc<Ring<DomeSnapshot>>,
}

/// The fastest publication rate expected per kind (sizes the rings).
fn fastest_cadence(kind: DeviceKind) -> Duration {
    match kind {
        // 10 Hz during an Achernar-issued pulse (§2.7).
        DeviceKind::Mount => Duration::from_millis(100),
        DeviceKind::Guider => Duration::from_millis(250),
        other => default_cadence(other).active,
    }
}

impl Rings {
    fn new(retention: Duration) -> Self {
        fn ring<S: Timestamped>(kind: DeviceKind, retention: Duration) -> Arc<Ring<S>> {
            Arc::new(Ring::for_cadence(retention, fastest_cadence(kind)))
        }
        Self {
            mount: ring(DeviceKind::Mount, retention),
            focuser: ring(DeviceKind::Focuser, retention),
            filter_wheel: ring(DeviceKind::FilterWheel, retention),
            guider: ring(DeviceKind::Guider, retention),
            rotator: ring(DeviceKind::Rotator, retention),
            cover_calibrator: ring(DeviceKind::CoverCalibrator, retention),
            switch: ring(DeviceKind::Switch, retention),
            weather: ring(DeviceKind::Weather, retention),
            safety_monitor: ring(DeviceKind::SafetyMonitor, retention),
            dome: ring(DeviceKind::Dome, retention),
        }
    }
}

/// A handle type the hub can hold (one per kind).
pub trait HubDevice: Clone + Send + Sync + 'static {
    type Snap: DeviceSnapshot;
    fn wrap(self) -> AnyHandle;
    fn unwrap_any(any: &AnyHandle) -> Option<Self>;
    fn ring(rings: &Rings) -> &Arc<Ring<Self::Snap>>;
}

macro_rules! hub_device {
    ($cmd:ty, $snap:ty, $variant:ident, $ring:ident) => {
        impl HubDevice for DeviceHandle<$cmd, $snap> {
            type Snap = $snap;

            fn wrap(self) -> AnyHandle {
                AnyHandle::$variant(self)
            }

            fn unwrap_any(any: &AnyHandle) -> Option<Self> {
                match any {
                    AnyHandle::$variant(h) => Some(h.clone()),
                    _ => None,
                }
            }

            fn ring(rings: &Rings) -> &Arc<Ring<$snap>> {
                &rings.$ring
            }
        }
    };
}

hub_device!(MountCmd, MountSnapshot, Mount, mount);
hub_device!(FocuserCmd, FocuserSnapshot, Focuser, focuser);
hub_device!(
    FilterWheelCmd,
    FilterWheelSnapshot,
    FilterWheel,
    filter_wheel
);
hub_device!(GuiderCmd, GuiderSnapshot, Guider, guider);
hub_device!(RotatorCmd, RotatorSnapshot, Rotator, rotator);
hub_device!(
    CoverCalibratorCmd,
    CoverCalibratorSnapshot,
    CoverCalibrator,
    cover_calibrator
);
hub_device!(SwitchCmd, SwitchSnapshot, Switch, switch);
hub_device!(WeatherCmd, WeatherSnapshot, Weather, weather);
hub_device!(SafetyCmd, SafetySnapshot, SafetyMonitor, safety_monitor);
hub_device!(DomeCmd, DomeSnapshot, Dome, dome);

// ---------------------------------------------------------------------------
// Safety summary
// ---------------------------------------------------------------------------

/// The combined safety state for one monitor (§2.8 fail-safe rule, §3.10).
///
/// A confirmed Unsafe reading (or Warning when `warning_is_unsafe`) is
/// `Unsafe`. Otherwise anything but a Ready monitor with a fresh Safe
/// reading is `Unknown`. No monitor is `None`. The safety hub (WP1A-8) may
/// publish its own AND-combined summary with
/// [`EquipmentHub::set_safety_summary`].
pub fn derive_safety_summary(
    monitor: Option<(&SafetySnapshot, LinkState, &DeviceIdentity)>,
    warning_is_unsafe: bool,
) -> SafetySummary {
    let Some((snap, link, identity)) = monitor else {
        return SafetySummary::default();
    };
    let unsafe_level = |level: SafetyLevel| match level {
        SafetyLevel::Unsafe => true,
        SafetyLevel::Warning => warning_is_unsafe,
        SafetyLevel::Safe => false,
    };
    let state = match snap.level.known() {
        Some(level) if unsafe_level(*level) => SafetyGate::Unsafe,
        _ if link != LinkState::Ready => SafetyGate::Unknown,
        _ => match snap.level.fresh_value() {
            Some(_) => SafetyGate::Safe,
            None => SafetyGate::Unknown,
        },
    };
    SafetySummary {
        state,
        sources: vec![SafetySource {
            connection_id: identity.connection_id,
            name: identity.name.clone(),
            link,
            level: snap.level.known().copied(),
        }],
        changed_at: snap.level.observed_at,
    }
}

// ---------------------------------------------------------------------------
// Window helpers
// ---------------------------------------------------------------------------

/// Device kinds whose motion an event signals (quiescence and `moved`).
pub fn motion_kinds(event: &EquipmentEvent) -> &'static [DeviceKind] {
    use DeviceKind::*;
    match &event.event {
        EventKind::SlewStarted
        | EventKind::SlewArrived { .. }
        | EventKind::SlewSettled { .. }
        | EventKind::SlewEnded { .. }
        | EventKind::ArrivedOffTarget { .. }
        | EventKind::MoveAxis { .. }
        | EventKind::PulseGuide { .. }
        | EventKind::MeridianFlip { .. }
        | EventKind::Parked { .. }
        | EventKind::Unparked
        | EventKind::Homed
        | EventKind::Centered { .. } => &[Mount],
        EventKind::FocuserMoveStarted { .. }
        | EventKind::FocuserMoveStopped { .. }
        | EventKind::FocuserSettled { .. }
        | EventKind::FocuserHalted { .. }
        | EventKind::FilterOffsetApplied { .. } => &[Focuser],
        EventKind::FilterMoving { .. }
        | EventKind::FilterArrived { .. }
        | EventKind::FilterSettled { .. } => &[FilterWheel],
        // A dither physically moves the mount.
        EventKind::Dithered { .. } | EventKind::SettleBegin => &[Guider, Mount],
        EventKind::RotatorMoved { .. } => &[Rotator],
        EventKind::CoverChanged { .. } => &[CoverCalibrator],
        EventKind::DomeShutterChanged { .. } => &[Dome],
        _ => &[],
    }
}

fn signals_motion_of(event: &EquipmentEvent, kind: DeviceKind) -> bool {
    motion_kinds(event).contains(&kind)
}

/// Guide statistics over `(ra, dec, pixel scale)` samples: standard
/// deviation (population, as NINA's Welford RMS), RMS about zero and peak per
/// axis; arcsec only when every sample has a pixel scale.
pub fn guide_rms(samples: &[(f64, f64, Option<f64>)]) -> Option<GuideRms> {
    let n = samples.len();
    if n == 0 {
        return None;
    }
    let stats = |values: &mut dyn Iterator<Item = (f64, f64)>| {
        let (mut sum_ra, mut sum_dec, mut sq_ra, mut sq_dec, mut count) = (0.0, 0.0, 0.0, 0.0, 0);
        let (mut peak_ra, mut peak_dec) = (0.0_f64, 0.0_f64);
        for (ra, dec) in values {
            sum_ra += ra;
            sum_dec += dec;
            sq_ra += ra * ra;
            sq_dec += dec * dec;
            peak_ra = peak_ra.max(ra.abs());
            peak_dec = peak_dec.max(dec.abs());
            count += 1;
        }
        let n = f64::from(count);
        let (mean_ra, mean_dec) = (sum_ra / n, sum_dec / n);
        let sd_ra = (sq_ra / n - mean_ra * mean_ra).max(0.0).sqrt();
        let sd_dec = (sq_dec / n - mean_dec * mean_dec).max(0.0).sqrt();
        let rms0_ra = (sq_ra / n).sqrt();
        let rms0_dec = (sq_dec / n).sqrt();
        (sd_ra, sd_dec, rms0_ra, rms0_dec, peak_ra, peak_dec)
    };
    let (sd_ra, sd_dec, rms0_ra, rms0_dec, peak_ra, peak_dec) =
        stats(&mut samples.iter().map(|(ra, dec, _)| (*ra, *dec)));
    let mut rms = GuideRms {
        ra_px: sd_ra,
        dec_px: sd_dec,
        total_px: (sd_ra * sd_ra + sd_dec * sd_dec).sqrt(),
        rms0_ra_px: rms0_ra,
        rms0_dec_px: rms0_dec,
        rms0_total_px: (rms0_ra * rms0_ra + rms0_dec * rms0_dec).sqrt(),
        peak_ra_px: peak_ra,
        peak_dec_px: peak_dec,
        steps_n: u32::try_from(n).unwrap_or(u32::MAX),
        ..GuideRms::default()
    };
    if samples.iter().all(|(_, _, scale)| scale.is_some()) {
        let (sd_ra, sd_dec, rms0_ra, rms0_dec, _, _) = stats(
            &mut samples
                .iter()
                .map(|(ra, dec, scale)| (ra * scale.unwrap_or(0.0), dec * scale.unwrap_or(0.0))),
        );
        rms.ra_arcsec = Some(sd_ra);
        rms.dec_arcsec = Some(sd_dec);
        rms.total_arcsec = Some((sd_ra * sd_ra + sd_dec * sd_dec).sqrt());
        rms.rms0_total_arcsec = Some((rms0_ra * rms0_ra + rms0_dec * rms0_dec).sqrt());
    }
    Some(rms)
}

enum KindQuiet {
    Quiet(DateTime<Utc>),
    Moved(u64),
    Unknown(String),
}

/// State shared by the per-kind passes of one `window()` call.
struct WindowPass<'a> {
    attached: &'a BTreeSet<DeviceKind>,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    finalised: bool,
    stale_kinds: Vec<DeviceKind>,
}

// ---------------------------------------------------------------------------
// The hub core
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct CameraSensor {
    celsius: Observed<f64>,
    name: Option<String>,
}

/// Shared hub state. The gate holds it as its [`GateContext`].
pub struct HubCore {
    config: HubConfig,
    clock: Arc<dyn Clock>,
    recorder: Arc<dyn RecorderSink>,
    bus: EventBus,
    resume: ResumeBlocks,
    busy: Arc<BusyPublisher>,
    devices: RwLock<BTreeMap<DeviceKind, AnyHandle>>,
    rings: Rings,
    site: RwLock<Option<Site>>,
    camera: RwLock<Option<Arc<dyn CameraActivity>>>,
    camera_sensor: RwLock<Option<CameraSensor>>,
    ephemeris: RwLock<Option<Arc<dyn SkyEphemeris>>>,
    view_direction: RwLock<Option<AltAz>>,
    safety_override: RwLock<Option<SafetySummary>>,
    registry: RwLock<Option<Arc<RegistryBridge>>>,
    watchdog: Arc<Watchdog>,
    lost_seen: Mutex<HashSet<Uuid>>,
}

impl std::fmt::Debug for HubCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HubCore")
            .field("devices", &read(&self.devices).keys().collect::<Vec<_>>())
            .field("bus", &self.bus)
            .finish()
    }
}

impl HubCore {
    fn new(config: HubConfig, clock: Arc<dyn Clock>, recorder: Arc<dyn RecorderSink>) -> Self {
        Self {
            bus: EventBus::new(
                Arc::clone(&recorder),
                config.ring_retention,
                config.event_capacity,
                config.journal_len,
            ),
            resume: ResumeBlocks::new(Arc::clone(&clock)),
            busy: Arc::new(BusyPublisher::new()),
            devices: RwLock::new(BTreeMap::new()),
            rings: Rings::new(config.ring_retention),
            site: RwLock::new(None),
            camera: RwLock::new(None),
            camera_sensor: RwLock::new(None),
            ephemeris: RwLock::new(None),
            view_direction: RwLock::new(None),
            safety_override: RwLock::new(None),
            registry: RwLock::new(None),
            watchdog: Arc::new(Watchdog::new()),
            lost_seen: Mutex::new(HashSet::new()),
            config,
            clock,
            recorder,
        }
    }

    fn handles(&self) -> Vec<AnyHandle> {
        read(&self.devices).values().cloned().collect()
    }

    fn device(&self, kind: DeviceKind) -> Option<AnyHandle> {
        read(&self.devices).get(&kind).cloned()
    }

    fn link(&self, kind: DeviceKind) -> Option<LinkState> {
        self.device(kind).map(|h| h.health().state)
    }

    // -- events and blocks -------------------------------------------------------

    fn ingest(&self, event: EquipmentEvent) -> Option<Arc<EquipmentEvent>> {
        let event = self.bus.publish(event)?;
        let request = match (&event.event, event.device_kind) {
            (EventKind::OutcomeUnknown { op, reason }, Some(kind)) => Some(BlockRequest::new(
                ResumeBlockKind::OutcomeUnknown {
                    device: kind,
                    op: *op,
                },
                reason.clone(),
            )),
            (EventKind::DeviceLost { reason, .. }, Some(DeviceKind::Mount)) => {
                Some(BlockRequest::new(
                    ResumeBlockKind::MountLost,
                    reason.clone().unwrap_or_else(|| "mount lost".into()),
                ))
            }
            (EventKind::DeviceLost { reason, .. }, Some(DeviceKind::Guider)) => {
                Some(BlockRequest::new(
                    ResumeBlockKind::GuiderLost,
                    reason.clone().unwrap_or_else(|| "guider lost".into()),
                ))
            }
            (EventKind::LimitReached { kind, value, limit }, _) => Some(BlockRequest::new(
                ResumeBlockKind::LimitReached,
                format!("{kind:?} limit {limit} reached (value {value:?})"),
            )),
            _ => None,
        };
        if let Some(request) = request {
            if let (Some(kind), Some(id)) = (event.device_kind, event.connection_id) {
                if matches!(event.event, EventKind::DeviceLost { .. }) {
                    lock(&self.lost_seen).insert(id);
                }
                self.raise(
                    request
                        .for_device(kind, Some(id))
                        .with_command(event.command_id),
                );
            } else {
                self.raise(request.with_command(event.command_id));
            }
        }
        Some(event)
    }

    fn raise(&self, request: BlockRequest) -> ResumeBlock {
        match self.resume.raise(request) {
            Raised::New(block) => {
                self.ingest(EquipmentEvent::new(
                    EventKind::ResumeBlocked {
                        block: block.clone(),
                    },
                    self.clock.as_ref(),
                ));
                block
            }
            Raised::Existing(block) => block,
        }
    }

    fn cleared(&self, block: &ResumeBlock, by: &CommandSource) {
        self.ingest(EquipmentEvent::new(
            EventKind::ResumeCleared {
                block: block.clone(),
                by: by.clone(),
            },
            self.clock.as_ref(),
        ));
    }

    // -- the aggregate snapshot --------------------------------------------------

    fn camera_summary(&self) -> CameraSummary {
        match read(&self.camera).clone() {
            Some(camera) => CameraSummary {
                exposing: camera.exposing(),
                exposure: camera.current_exposure(),
                name: camera.camera_name(),
            },
            None => CameraSummary::default(),
        }
    }

    fn build_snapshot(&self) -> EquipmentSnapshot {
        let mut snap = EquipmentSnapshot::empty(self.clock.now_utc(), self.clock.now_mono_ns());
        let mut monitor = None;
        for handle in self.handles() {
            if handle.is_retired() {
                continue;
            }
            let link = handle.health().state;
            snap.links.insert(handle.kind(), link);
            match &handle {
                AnyHandle::Mount(h) => snap.mount = Some(h.snapshot()),
                AnyHandle::Focuser(h) => snap.focuser = Some(h.snapshot()),
                AnyHandle::FilterWheel(h) => snap.filter_wheel = Some(h.snapshot()),
                AnyHandle::Guider(h) => snap.guider = Some(h.snapshot()),
                AnyHandle::Rotator(h) => snap.rotator = Some(h.snapshot()),
                AnyHandle::CoverCalibrator(h) => snap.cover_calibrator = Some(h.snapshot()),
                AnyHandle::Switch(h) => snap.switch = Some(h.snapshot()),
                AnyHandle::Weather(h) => snap.weather = Some(h.snapshot()),
                AnyHandle::SafetyMonitor(h) => {
                    let s = h.snapshot();
                    monitor = Some((Arc::clone(&s), link, h.identity()));
                    snap.safety_monitor = Some(s);
                }
                AnyHandle::Dome(h) => snap.dome = Some(h.snapshot()),
            }
        }
        snap.camera = self.camera_summary();
        snap.site = *read(&self.site);
        snap.busy = self.busy.current();
        snap.resume_blocks = self.resume.active();
        snap.safety = match read(&self.safety_override).clone() {
            Some(summary) => summary,
            None => derive_safety_summary(
                monitor
                    .as_ref()
                    .map(|(s, link, identity)| (s.as_ref(), *link, identity.as_ref())),
                self.config.warning_is_unsafe,
            ),
        };
        snap
    }

    // -- refresh -----------------------------------------------------------------

    fn detach(&self, kind: DeviceKind, connection_id: Uuid) -> bool {
        let removed = {
            let mut devices = write(&self.devices);
            match devices.get(&kind) {
                Some(h) if h.connection_id() == connection_id => devices.remove(&kind),
                _ => None,
            }
        };
        self.bus.retire(kind, connection_id);
        removed.is_some()
    }

    fn refresh(&self) {
        let now_mono = self.clock.now_mono_ns();
        let handles = self.handles();
        // The registry first, so a retired handle's final `disconnected`
        // projection lands before it is detached.
        let bridge = read(&self.registry).clone();
        if let Some(bridge) = bridge {
            for handle in &handles {
                bridge.project(
                    &handle.identity(),
                    &handle.health(),
                    &handle.snapshot_json(),
                    handle.snapshot_age_ms(now_mono),
                );
            }
        }
        for handle in handles {
            handle.push_into(&self.rings);
            if handle.is_retired() {
                self.detach(handle.kind(), handle.connection_id());
                continue;
            }
            // A lost mount or guider blocks automatic resumption even when no
            // `device_lost` event reached the bus (once per connection).
            if handle.health().state == LinkState::Lost
                && lock(&self.lost_seen).insert(handle.connection_id())
            {
                let kind = match handle.kind() {
                    DeviceKind::Mount => Some(ResumeBlockKind::MountLost),
                    DeviceKind::Guider => Some(ResumeBlockKind::GuiderLost),
                    _ => None,
                };
                if let Some(kind) = kind {
                    self.raise(
                        BlockRequest::new(
                            kind,
                            handle
                                .health()
                                .reason
                                .unwrap_or_else(|| "device lost".into()),
                        )
                        .for_device(handle.kind(), Some(handle.connection_id())),
                    );
                }
            }
        }
        self.busy.set_derived(derive_busy(&self.build_snapshot()));
        self.watchdog.tick(now_mono);
    }

    // -- windows and quiescence ----------------------------------------------------

    fn kind_quiet<S: DeviceSnapshot>(
        &self,
        ring: &Ring<S>,
        kind: DeviceKind,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        window_events: &[Arc<EquipmentEvent>],
    ) -> KindQuiet {
        let name = kind_name(kind);
        if ring.is_empty() {
            return KindQuiet::Unknown(format!("{name}_no_data"));
        }
        if self.link(kind) == Some(LinkState::Lost) {
            return KindQuiet::Unknown(format!("{name}_lost"));
        }
        if let Some(event) = window_events.iter().find(|e| signals_motion_of(e, kind)) {
            return KindQuiet::Moved(event.seq);
        }
        let idle = chrono_of(default_cadence(kind).idle);
        // The guider is event-driven: a stopped guider publishes rarely, so
        // only motion matters for it, not sample coverage.
        let event_driven = kind == DeviceKind::Guider;
        let before = ring.at_or_before(start);
        if !event_driven {
            let Some(before) = &before else {
                return KindQuiet::Unknown(format!("{name}_no_sample_before_start"));
            };
            if start - before.at_utc() > idle * 3 {
                return KindQuiet::Unknown(format!("{name}_stale"));
            }
            if ring.at_or_after(end).is_none() {
                let fresh = ring
                    .latest()
                    .is_some_and(|latest| self.clock.now_utc() - latest.at_utc() <= idle * 2);
                if !fresh {
                    return KindQuiet::Unknown(format!("{name}_no_sample_after_end"));
                }
            }
        }
        let moving = before
            .iter()
            .chain(ring.range(start, end).iter())
            .any(|s| s.in_motion());
        if moving {
            let seq = self
                .bus
                .events_between(start - idle * 3, end)
                .iter()
                .rev()
                .find(|e| signals_motion_of(e, kind))
                .map_or(0, |e| e.seq);
            return KindQuiet::Moved(seq);
        }
        // Settled since: the first still sample after the last moving one,
        // and never before the last motion event.
        let mut since: Option<DateTime<Utc>> = None;
        for sample in ring.items().iter().filter(|s| s.at_utc() <= start) {
            if sample.in_motion() {
                since = None;
            } else if since.is_none() {
                since = Some(sample.at_utc());
            }
        }
        let oldest = ring.oldest().map(|s| s.at_utc()).unwrap_or(start);
        let last_motion_event = self
            .bus
            .events_between(oldest.min(start), start)
            .iter()
            .filter(|e| signals_motion_of(e, kind))
            .map(|e| e.at_utc)
            .max();
        let mut settled = since.unwrap_or(start);
        if let Some(at) = last_motion_event {
            settled = settled.max(at);
        }
        KindQuiet::Quiet(settled)
    }

    fn quiescent(
        &self,
        kinds: &[DeviceKind],
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Quiescence {
        let window_events = self.bus.events_between(start, end);
        let mut settled_since: Option<DateTime<Utc>> = None;
        for &kind in kinds {
            let result = match kind {
                DeviceKind::Mount => {
                    self.kind_quiet(&self.rings.mount, kind, start, end, &window_events)
                }
                DeviceKind::Focuser => {
                    self.kind_quiet(&self.rings.focuser, kind, start, end, &window_events)
                }
                DeviceKind::FilterWheel => {
                    self.kind_quiet(&self.rings.filter_wheel, kind, start, end, &window_events)
                }
                DeviceKind::Guider => {
                    self.kind_quiet(&self.rings.guider, kind, start, end, &window_events)
                }
                DeviceKind::Rotator => {
                    self.kind_quiet(&self.rings.rotator, kind, start, end, &window_events)
                }
                DeviceKind::CoverCalibrator => self.kind_quiet(
                    &self.rings.cover_calibrator,
                    kind,
                    start,
                    end,
                    &window_events,
                ),
                DeviceKind::Dome => {
                    self.kind_quiet(&self.rings.dome, kind, start, end, &window_events)
                }
                // The camera is OLS's; switches, weather and safety monitors
                // never move the optics.
                DeviceKind::Camera
                | DeviceKind::Switch
                | DeviceKind::Weather
                | DeviceKind::SafetyMonitor => continue,
            };
            match result {
                KindQuiet::Quiet(at) => {
                    settled_since = Some(settled_since.map_or(at, |s| s.max(at)));
                }
                KindQuiet::Moved(event_seq) => return Quiescence::Moved { kind, event_seq },
                KindQuiet::Unknown(reason) => return Quiescence::Unknown { reason },
            }
        }
        Quiescence::Quiescent {
            settled_since: settled_since.unwrap_or(start),
        }
    }

    /// Start/mid/end samples of one kind, applying D7: the window waits for a
    /// sample newer than the exposure end, for at most 2 x the idle cadence;
    /// after that the end state is marked stale.
    fn finalise<S: DeviceSnapshot>(
        &self,
        ring: &Ring<S>,
        kind: DeviceKind,
        pass: &mut WindowPass<'_>,
    ) -> super::snapshot::WindowSamples<S> {
        let pick = ring.window_samples(pass.start, pass.end);
        let mut samples = pick.samples;
        let relevant = pass.attached.contains(&kind) || pick.has_data;
        if relevant && pick.has_data && !pick.has_post_end {
            let grace = chrono_of(default_cadence(kind).idle) * 2;
            if self.clock.now_utc() >= pass.end + grace {
                samples.end_stale = true;
                pass.stale_kinds.push(kind);
            } else {
                pass.finalised = false;
            }
        }
        samples
    }

    fn guide_stats(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        events: &[Arc<EquipmentEvent>],
    ) -> GuideWindowStats {
        let steps = self.bus.guide_steps_between(start, end);
        let guiding: Vec<(f64, f64, Option<f64>)> = steps
            .iter()
            .filter(|e| e.step.phase == GuidePhase::Guiding)
            .filter_map(|e| Some((e.step.ra_raw?, e.step.dec_raw?, e.step.pixel_scale_arcsec)))
            .collect();
        let pulses_from_steps = steps
            .iter()
            .filter(|e| {
                e.step.ra_duration_ms.is_some_and(|ms| ms > 0)
                    || e.step.dec_duration_ms.is_some_and(|ms| ms > 0)
            })
            .count();
        let achernar_pulses = events
            .iter()
            .filter(|e| matches!(e.event, EventKind::PulseGuide { .. }))
            .count();
        GuideWindowStats {
            state: self
                .rings
                .guider
                .nearest(midpoint(start, end))
                .and_then(|s| s.app_state.known().copied()),
            rms: guide_rms(&guiding),
            dither_in_window: events
                .iter()
                .any(|e| matches!(e.event, EventKind::Dithered { .. }))
                || steps.iter().any(|e| e.step.phase == GuidePhase::Dithering),
            settle_in_window: events.iter().any(|e| {
                matches!(
                    e.event,
                    EventKind::SettleBegin
                        | EventKind::Settling { .. }
                        | EventKind::SettleDone { .. }
                )
            }) || steps.iter().any(|e| e.step.phase == GuidePhase::Settling),
            pulses_n: u32::try_from(pulses_from_steps + achernar_pulses).unwrap_or(u32::MAX),
        }
    }

    fn motion_intervals(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<MotionInterval> {
        let mut samples = Vec::new();
        if let Some(before) = self.rings.mount.at_or_before(start) {
            samples.push(before);
        }
        for sample in self.rings.mount.range(start, end) {
            if samples.last().map(|s| s.at_utc()) != Some(sample.at_utc()) {
                samples.push(sample);
            }
        }
        let mut intervals: Vec<MotionInterval> = Vec::new();
        for (i, sample) in samples.iter().enumerate() {
            let from = sample.at_utc().max(start);
            let to = samples
                .get(i + 1)
                .map(|next| next.at_utc().min(end))
                .unwrap_or(end);
            if to < from {
                continue;
            }
            let interval = MotionInterval {
                from,
                to,
                tracking: sample.state.tracking.fresh_value().copied(),
                slewing: sample.state.slewing.fresh_value().copied(),
                pulse_guiding: sample.state.is_pulse_guiding.fresh_value().copied(),
            };
            match intervals.last_mut() {
                Some(last)
                    if last.tracking == interval.tracking
                        && last.slewing == interval.slewing
                        && last.pulse_guiding == interval.pulse_guiding =>
                {
                    last.to = interval.to;
                }
                _ => intervals.push(interval),
            }
        }
        intervals
    }

    fn window(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> DeviceWindow {
        let attached: BTreeSet<DeviceKind> = read(&self.devices).keys().copied().collect();
        let mut pass = WindowPass {
            attached: &attached,
            start,
            end,
            finalised: true,
            stale_kinds: Vec::new(),
        };
        let events = self.bus.events_between(start, end);
        let mut window = DeviceWindow {
            start,
            end,
            ..DeviceWindow::default()
        };
        let r = &self.rings;
        window.mount = self.finalise(&r.mount, DeviceKind::Mount, &mut pass);
        window.focuser = self.finalise(&r.focuser, DeviceKind::Focuser, &mut pass);
        window.filter_wheel = self.finalise(&r.filter_wheel, DeviceKind::FilterWheel, &mut pass);
        window.rotator = self.finalise(&r.rotator, DeviceKind::Rotator, &mut pass);
        window.cover_calibrator =
            self.finalise(&r.cover_calibrator, DeviceKind::CoverCalibrator, &mut pass);
        window.switch = self.finalise(&r.switch, DeviceKind::Switch, &mut pass);
        window.dome = self.finalise(&r.dome, DeviceKind::Dome, &mut pass);
        let WindowPass {
            finalised,
            stale_kinds,
            ..
        } = pass;
        let mid = midpoint(start, end);
        window.mount_mid = mount_position_at(&r.mount, mid);
        window.weather = r.weather.nearest(mid).map(|s| s.reading.clone());
        window.safety = match r.safety_monitor.nearest(mid) {
            Some(sample) => {
                let identity = self
                    .device(DeviceKind::SafetyMonitor)
                    .map(|h| (*h.identity()).clone());
                let link = if sample.level.is_fresh() {
                    LinkState::Ready
                } else {
                    LinkState::Degraded
                };
                identity.map(|identity| {
                    derive_safety_summary(
                        Some((sample.as_ref(), link, &identity)),
                        self.config.warning_is_unsafe,
                    )
                })
            }
            None if !attached.contains(&DeviceKind::SafetyMonitor) => {
                Some(SafetySummary::default())
            }
            None => Some(SafetySummary {
                state: SafetyGate::Unknown,
                ..SafetySummary::default()
            }),
        };
        window.guide = self.guide_stats(start, end, &events);
        window.motion = self.motion_intervals(start, end);
        let moved = |kind: DeviceKind, ring_moving: bool| {
            ring_moving || events.iter().any(|e| signals_motion_of(e, kind))
        };
        let any_motion = |samples: Vec<bool>| samples.into_iter().any(|m| m);
        window.moved = MovedFlags {
            focuser: moved(
                DeviceKind::Focuser,
                any_motion(
                    r.focuser
                        .range(start, end)
                        .iter()
                        .map(|s| s.in_motion())
                        .collect(),
                ) || window
                    .focuser
                    .start
                    .as_ref()
                    .and_then(|s| s.logical_position)
                    != window.focuser.end.as_ref().and_then(|s| s.logical_position),
            ),
            wheel: moved(
                DeviceKind::FilterWheel,
                any_motion(
                    r.filter_wheel
                        .range(start, end)
                        .iter()
                        .map(|s| s.in_motion())
                        .collect(),
                ) || window
                    .filter_wheel
                    .start
                    .as_ref()
                    .and_then(|s| s.position.known().copied())
                    != window
                        .filter_wheel
                        .end
                        .as_ref()
                        .and_then(|s| s.position.known().copied()),
            ),
            mount: moved(
                DeviceKind::Mount,
                any_motion(
                    r.mount
                        .range(start, end)
                        .iter()
                        .map(|s| s.in_motion())
                        .collect(),
                ),
            ),
            rotator: moved(
                DeviceKind::Rotator,
                any_motion(
                    r.rotator
                        .range(start, end)
                        .iter()
                        .map(|s| s.in_motion())
                        .collect(),
                ),
            ),
        };
        window.resume_block_active = self.resume.active_during(start, end);
        let motion_kinds: Vec<DeviceKind> = [
            DeviceKind::Mount,
            DeviceKind::Focuser,
            DeviceKind::FilterWheel,
            DeviceKind::Rotator,
        ]
        .into_iter()
        .filter(|kind| attached.contains(kind))
        .collect();
        window.quiescence = self.quiescent(&motion_kinds, start, end);
        window.clock_quality = self.exposure_clock_quality(start);
        window.events = events.iter().map(|e| (**e).clone()).collect();
        window.finalised = finalised;
        window.stale_kinds = stale_kinds;
        window
    }

    /// The clock quality the camera hook reported for the exposure that
    /// started at `start` (`exposure_started`), else unknown.
    fn exposure_clock_quality(&self, start: DateTime<Utc>) -> ClockQuality {
        let slack = chrono::Duration::seconds(120);
        self.bus
            .events_between(start - slack, start + slack)
            .iter()
            .find_map(|event| match &event.event {
                EventKind::ExposureStarted { exposure } if exposure.started_at == Some(start) => {
                    Some(exposure.clock_quality)
                }
                _ => None,
            })
            .unwrap_or_default()
    }

    // -- H6 and H14 ------------------------------------------------------------------

    fn ambient_temperature(&self) -> Observed<TemperatureReading> {
        let snap = self.build_snapshot();
        let identity_of = |kind| self.device(kind).map(|h| h.identity());
        let mut candidates: Vec<Observed<TemperatureReading>> = Vec::new();
        if let Some(weather) = &snap.weather {
            let identity = identity_of(DeviceKind::Weather);
            candidates.push(weather.reading.temperature_c.clone().map(|celsius| {
                TemperatureReading {
                    celsius,
                    source: TemperatureSourceKind::Weather,
                    role: TemperatureRole::Ambient,
                    identity_key: identity.as_ref().map(|i| i.identity_key.clone()),
                    connection_id: identity.as_ref().map(|i| i.connection_id),
                    camera_sensor_fallback: false,
                }
            }));
        }
        if let Some(focuser) = &snap.focuser {
            if matches!(
                focuser.temperature_role,
                TemperatureRole::Ambient | TemperatureRole::Tube
            ) {
                let identity = identity_of(DeviceKind::Focuser);
                let role = focuser.temperature_role;
                candidates.push(focuser.state.temperature_c.clone().map(|celsius| {
                    TemperatureReading {
                        celsius,
                        source: TemperatureSourceKind::FocuserProbe,
                        role,
                        identity_key: identity.as_ref().map(|i| i.identity_key.clone()),
                        connection_id: identity.as_ref().map(|i| i.connection_id),
                        camera_sensor_fallback: false,
                    }
                }));
            }
        }
        if let Some(sensor) = read(&self.camera_sensor).clone() {
            let name = sensor.name.clone();
            candidates.push(sensor.celsius.map(|celsius| TemperatureReading {
                celsius,
                source: TemperatureSourceKind::CameraSensor,
                role: TemperatureRole::Unknown,
                identity_key: name.clone(),
                connection_id: None,
                camera_sensor_fallback: true,
            }));
        }
        // Fresh before stale among the ambient sources (weather, then the
        // probe); the camera sensor only when neither has a value at all.
        let usable = |o: &Observed<TemperatureReading>| {
            o.value.as_ref().is_some_and(|r| r.celsius.is_finite())
        };
        let ambient = |o: &&Observed<TemperatureReading>| {
            o.value.as_ref().is_some_and(|r| !r.camera_sensor_fallback)
        };
        candidates
            .iter()
            .filter(ambient)
            .find(|o| o.state == ObsState::Fresh && usable(o))
            .or_else(|| candidates.iter().filter(ambient).find(|o| usable(o)))
            .or_else(|| candidates.iter().find(|o| usable(o)))
            .cloned()
            .unwrap_or_else(|| Observed::not_set(Source::Derived))
    }

    fn coc_context(&self) -> CocContext {
        let snap = self.build_snapshot();
        let at = self.clock.now_utc();
        let mount = snap.mount.clone();
        let site = snap.site.or_else(|| {
            mount
                .as_ref()
                .and_then(|m| m.state.site.fresh_value().copied())
        });
        let mut context = CocContext {
            at,
            site,
            view_direction: *read(&self.view_direction),
            ..CocContext::default()
        };
        if let Some(mount) = &mount {
            context.mount_alt_deg = mount.state.alt_deg.clone();
            context.mount_az_deg = mount.state.az_deg.clone();
            context.tracking = mount.state.tracking.clone();
            context.slewing = mount.state.slewing.clone();
            context.pointing_j2000 = mount.j2000;
        }
        let ephemeris = read(&self.ephemeris).clone();
        if let (Some(site), Some(ephemeris)) = (site, ephemeris) {
            if let Some(sky) = ephemeris.sky(at, &site) {
                context.sun_alt_deg = Some(sky.sun_alt_deg);
                context.moon_alt_deg = Some(sky.moon_alt_deg);
                context.moon_illumination = Some(sky.moon_illumination);
                context.twilight = Some(twilight_for_sun_alt(sky.sun_alt_deg));
            }
        }
        let mut corroboration = CocCorroboration::default();
        if let Some(weather) = &snap.weather {
            let r = &weather.reading;
            corroboration.cloud_cover_pct = r.cloud_cover_pct.clone();
            corroboration.sky_temperature_c = r.sky_temperature_c.clone();
            corroboration.rain_rate_mm_h = r.rain_rate_mm_h.clone();
            corroboration.humidity_pct = r.humidity_pct.clone();
            corroboration.dew_point_c = r.dew_point_c.clone();
            corroboration.sky_quality_mag_arcsec2 = r.sky_quality_mag_arcsec2.clone();
        }
        if let Some(monitor) = &snap.safety_monitor {
            let warning_is_unsafe = self.config.warning_is_unsafe;
            corroboration.is_safe = monitor.level.clone().map(|level| match level {
                SafetyLevel::Safe => true,
                SafetyLevel::Unsafe => false,
                SafetyLevel::Warning => !warning_is_unsafe,
            });
        }
        let sky_fresh = [
            &corroboration.cloud_cover_pct,
            &corroboration.sky_temperature_c,
            &corroboration.sky_quality_mag_arcsec2,
        ]
        .iter()
        .any(|o| o.is_fresh());
        let other_fresh = [
            &corroboration.rain_rate_mm_h,
            &corroboration.humidity_pct,
            &corroboration.dew_point_c,
        ]
        .iter()
        .any(|o| o.is_fresh())
            || corroboration.is_safe.is_fresh();
        let any_known = [
            &corroboration.cloud_cover_pct,
            &corroboration.sky_temperature_c,
            &corroboration.sky_quality_mag_arcsec2,
            &corroboration.rain_rate_mm_h,
            &corroboration.humidity_pct,
            &corroboration.dew_point_c,
        ]
        .iter()
        .any(|o| o.known().is_some())
            || corroboration.is_safe.known().is_some();
        corroboration.level = if sky_fresh && other_fresh {
            CorroborationLevel::Full
        } else if any_known {
            CorroborationLevel::Partial
        } else {
            CorroborationLevel::None
        };
        context.corroboration = corroboration;
        if let Some(cover) = &snap.cover_calibrator {
            context.cover_state = cover.state.cover_state.clone();
        }
        if let Some(dome) = &snap.dome {
            context.dome_shutter = dome.state.shutter.clone();
        }
        if let Some(switch) = &snap.switch {
            for channel in &switch.channels {
                if channel.role.as_deref() == Some("dew_heater") {
                    context
                        .dew_heaters
                        .insert(channel.channel.name.clone(), channel.value.clone());
                }
            }
        }
        context
    }
}

impl GateContext for HubCore {
    fn clock(&self) -> Arc<dyn Clock> {
        Arc::clone(&self.clock)
    }

    fn snapshot(&self) -> Arc<EquipmentSnapshot> {
        Arc::new(self.build_snapshot())
    }

    fn active_blocks(&self) -> Vec<ResumeBlock> {
        self.resume.active()
    }

    fn raise_block(&self, request: BlockRequest) -> ResumeBlock {
        self.raise(request)
    }

    fn clear_block(&self, id: Uuid, by: &CommandSource) -> Result<ResumeBlock, ClearError> {
        let block = self.resume.clear(id, by)?;
        self.cleared(&block, by);
        Ok(block)
    }

    fn emit(&self, event: EquipmentEvent) {
        self.ingest(event);
    }

    fn record(&self, record: GroundTruthRecord) {
        self.recorder.record(record);
    }

    fn guider(&self) -> Option<GuiderHandle> {
        self.device(DeviceKind::Guider)
            .and_then(|h| GuiderHandle::unwrap_any(&h))
            .filter(|h| !h.is_retired())
    }
}

impl EventSink for HubCore {
    fn emit(&self, event: EquipmentEvent) -> Option<Arc<EquipmentEvent>> {
        self.ingest(event)
    }
}

// ---------------------------------------------------------------------------
// The hub
// ---------------------------------------------------------------------------

/// The equipment hub (see the module docs).
pub struct EquipmentHub {
    core: Arc<HubCore>,
    gate: Arc<CommandGate>,
    inert: bool,
    started: AtomicBool,
    tasks: Mutex<Vec<tokio::task::JoinHandle<()>>>,
    workers: Mutex<Vec<(DeviceKind, Uuid, WorkerHandle)>>,
}

impl std::fmt::Debug for EquipmentHub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EquipmentHub")
            .field("inert", &self.inert)
            .field("core", &self.core)
            .finish()
    }
}

impl EquipmentHub {
    /// The inert hub of `AppState::new` (D11): no task, no database writer,
    /// nothing attached. Every reader method answers "nothing connected".
    pub fn empty() -> Self {
        let mut hub = Self::new(
            HubConfig::default(),
            SystemClock::shared(),
            Arc::new(NullRecorder::new()),
        );
        hub.inert = true;
        hub
    }

    /// A configured hub. Nothing runs until [`EquipmentHub::start`].
    pub fn new(config: HubConfig, clock: Arc<dyn Clock>, recorder: Arc<dyn RecorderSink>) -> Self {
        let gate_config = config.gate.clone();
        let core = Arc::new(HubCore::new(config, clock, recorder));
        let gate = Arc::new(CommandGate::new(
            Arc::clone(&core) as Arc<dyn GateContext>,
            gate_config,
        ));
        Self {
            core,
            gate,
            inert: false,
            started: AtomicBool::new(false),
            tasks: Mutex::new(Vec::new()),
            workers: Mutex::new(Vec::new()),
        }
    }

    /// The hub for `with_equipment` (§2.10).
    pub fn from_configs(
        configs: &EquipmentConfigs,
        site: Option<Site>,
        recorder: Arc<dyn RecorderSink>,
    ) -> Self {
        let hub = Self::new(
            HubConfig::from_configs(configs),
            SystemClock::shared(),
            recorder,
        );
        hub.set_site(site);
        hub
    }

    pub fn is_inert(&self) -> bool {
        self.inert
    }

    /// Spawn the publisher (needs a tokio runtime). Returns false for the
    /// inert hub, outside a runtime, or when already started.
    pub fn start(self: &Arc<Self>) -> bool {
        if self.inert {
            return false;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return false;
        };
        if self.started.swap(true, Ordering::AcqRel) {
            return false;
        }
        let core = Arc::downgrade(&self.core);
        let period = self.core.config.publish_interval;
        let task = runtime.spawn(async move {
            let mut tick = tokio::time::interval(period);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tick.tick().await;
                let Some(core) = core.upgrade() else {
                    break;
                };
                core.refresh();
            }
        });
        lock(&self.tasks).push(task);
        true
    }

    /// Stop the publisher.
    pub fn stop(&self) {
        for task in lock(&self.tasks).drain(..) {
            task.abort();
        }
        self.started.store(false, Ordering::Release);
    }

    pub fn config(&self) -> &HubConfig {
        &self.core.config
    }

    pub fn clock(&self) -> Arc<dyn Clock> {
        Arc::clone(&self.core.clock)
    }

    pub fn gate(&self) -> &Arc<CommandGate> {
        &self.gate
    }

    pub fn recorder(&self) -> Arc<dyn RecorderSink> {
        Arc::clone(&self.core.recorder)
    }

    pub fn bus(&self) -> &EventBus {
        &self.core.bus
    }

    /// The sink workers and services publish into (connection-id filtered,
    /// with the resume-block side effects).
    pub fn event_sink(&self) -> Arc<dyn EventSink> {
        Arc::clone(&self.core) as Arc<dyn EventSink>
    }

    pub fn emit(&self, event: EquipmentEvent) -> Option<Arc<EquipmentEvent>> {
        self.core.ingest(event)
    }

    pub fn watchdog(&self) -> Arc<Watchdog> {
        Arc::clone(&self.core.watchdog)
    }

    pub fn rings(&self) -> &Rings {
        &self.core.rings
    }

    pub fn busy_flags(&self) -> &Arc<BusyPublisher> {
        &self.core.busy
    }

    pub fn resume(&self) -> &ResumeBlocks {
        &self.core.resume
    }

    // -- slots -------------------------------------------------------------------

    /// Attach a connected device's handle (replacing any previous handle of
    /// that kind) and return its ring, for the worker to push into.
    pub fn attach<H: HubDevice>(&self, handle: H) -> Arc<Ring<H::Snap>> {
        let any = handle.wrap();
        let kind = any.kind();
        let connection_id = any.connection_id();
        let previous = write(&self.core.devices).insert(kind, any.clone());
        if let Some(previous) = previous {
            if previous.connection_id() != connection_id {
                self.core.bus.retire(kind, previous.connection_id());
            }
        }
        self.core.bus.set_live(kind, connection_id);
        any.push_into(&self.core.rings);
        Arc::clone(H::ring(&self.core.rings))
    }

    /// Detach `kind` if `connection_id` is still its device. Late events from
    /// that connection are dropped from now on.
    pub fn detach(&self, kind: DeviceKind, connection_id: Uuid) -> bool {
        self.core.detach(kind, connection_id)
    }

    /// The attached handle of type `H`.
    pub fn handle<H: HubDevice>(&self) -> Option<H> {
        self.core
            .device(<H::Snap as DeviceSnapshot>::KIND)
            .and_then(|any| H::unwrap_any(&any))
    }

    pub fn any_handle(&self, kind: DeviceKind) -> Option<AnyHandle> {
        self.core.device(kind)
    }

    pub fn attached(&self) -> Vec<AnyHandle> {
        self.core.handles()
    }

    /// Keep a worker so [`EquipmentHub::shutdown`] can stop it.
    pub fn adopt_worker(&self, kind: DeviceKind, connection_id: Uuid, worker: WorkerHandle) {
        lock(&self.workers).push((kind, connection_id, worker));
    }

    // -- context setters -----------------------------------------------------------

    pub fn set_site(&self, site: Option<Site>) {
        *write(&self.core.site) = site;
    }

    pub fn site(&self) -> Option<Site> {
        *read(&self.core.site)
    }

    /// The `CameraActivity` adapter (WP1B-1).
    pub fn set_camera(&self, camera: Option<Arc<dyn CameraActivity>>) {
        *write(&self.core.camera) = camera;
    }

    pub fn camera(&self) -> Option<Arc<dyn CameraActivity>> {
        read(&self.core.camera).clone()
    }

    /// The camera sensor temperature, for the flagged H6 fallback only.
    pub fn set_camera_sensor_temperature(&self, celsius: Observed<f64>, camera: Option<String>) {
        *write(&self.core.camera_sensor) = Some(CameraSensor {
            celsius,
            name: camera,
        });
    }

    pub fn set_ephemeris(&self, ephemeris: Option<Arc<dyn SkyEphemeris>>) {
        *write(&self.core.ephemeris) = ephemeris;
    }

    /// A fixed all-sky camera's declared view direction (rig profile).
    pub fn set_view_direction(&self, direction: Option<AltAz>) {
        *write(&self.core.view_direction) = direction;
    }

    /// Replace the derived safety summary with the safety hub's (WP1A-8);
    /// `None` restores the derived one.
    pub fn set_safety_summary(&self, summary: Option<SafetySummary>) {
        *write(&self.core.safety_override) = summary;
    }

    /// Project health and snapshots into `state.devices` (WP1B-1).
    pub fn set_registry(&self, bridge: Option<RegistryBridge>) {
        *write(&self.core.registry) = bridge.map(Arc::new);
    }

    // -- blocks ----------------------------------------------------------------------

    /// Raise a resume block (handshake, safety, e-stop producers).
    pub fn raise_block(&self, request: BlockRequest) -> ResumeBlock {
        self.core.raise(request)
    }

    /// The safety resume policy (§3.10): a confirmed safe reading clears
    /// every `safety_*` block.
    pub fn clear_safety_blocks(&self, by: &CommandSource) -> Vec<ResumeBlock> {
        let cleared = self.core.resume.clear_safety(by);
        for block in &cleared {
            self.core.cleared(block, by);
        }
        cleared
    }

    // -- periodic work ---------------------------------------------------------------

    /// Pull the latest snapshots into the rings, detach retired handles,
    /// update the busy set, tick the watchdog and project the registry. The
    /// publisher calls this every 250 ms; tests call it directly.
    pub fn refresh(&self) {
        self.core.refresh();
    }

    /// Journal tail (`GET equipment/events?since_seq=&kinds=`).
    pub fn events_since(
        &self,
        since: u64,
        kinds: &[String],
        limit: usize,
    ) -> Vec<Arc<EquipmentEvent>> {
        self.core.bus.since(since, kinds, limit.min(500))
    }

    /// Daemon shutdown (§2.15): refuse new gated commands, stop every adopted
    /// worker (they halt moves, stop jogs, abort Achernar slews, never park)
    /// and wait up to `budget` for them.
    pub fn shutdown(&self, budget: Duration) {
        self.gate.begin_shutdown();
        let workers: Vec<_> = lock(&self.workers).drain(..).collect();
        for (_, _, worker) in &workers {
            worker.stop();
        }
        let deadline = std::time::Instant::now() + budget;
        for (_, _, worker) in workers {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            worker.join_timeout(left);
        }
        self.stop();
    }
}

impl Drop for EquipmentHub {
    fn drop(&mut self) {
        self.stop();
    }
}

impl EquipmentReader for EquipmentHub {
    fn snapshot(&self) -> Arc<EquipmentSnapshot> {
        Arc::new(self.core.build_snapshot())
    }

    fn subscribe(&self) -> broadcast::Receiver<Arc<EquipmentEvent>> {
        self.core.bus.subscribe()
    }

    fn window(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> DeviceWindow {
        self.core.window(start, end)
    }

    fn quiescent(
        &self,
        kinds: &[DeviceKind],
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Quiescence {
        self.core.quiescent(kinds, start, end)
    }

    fn busy(&self) -> watch::Receiver<BusySet> {
        self.core.busy.subscribe()
    }

    fn resume_blocks(&self) -> watch::Receiver<Vec<ResumeBlock>> {
        self.core.resume.subscribe()
    }

    fn ambient_temperature(&self) -> Observed<TemperatureReading> {
        self.core.ambient_temperature()
    }

    fn coc_context(&self) -> CocContext {
        self.core.coc_context()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::busy::BusyFlag;
    use crate::equipment::events::FlipPhase;
    use crate::equipment::ops::{ActiveOp, OpKind, OpPhase};
    use crate::equipment::snapshot::{
        ExposureRef, SwitchChannel, SwitchChannelState, WeatherReading,
    };
    use crate::equipment::testing::fake_handles::{
        fake_focuser, fake_guider, fake_mount, fake_safety_monitor, fake_switch, fake_weather,
        FakeCameraActivity, RecordingSink,
    };
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::types::{CommandId, EqCoord, EqFrame, FailureClass, SiteSource};

    fn hub_with_clock() -> (EquipmentHub, Arc<ManualClock>, Arc<RecordingSink>) {
        let clock = Arc::new(ManualClock::new());
        let sink = RecordingSink::new();
        let hub = EquipmentHub::new(HubConfig::default(), clock.clone(), sink.clone());
        (hub, clock, sink)
    }

    fn at(clock: &ManualClock) -> (DateTime<Utc>, u64) {
        (clock.now_utc(), clock.now_mono_ns())
    }

    fn mount_sample(clock: &ManualClock, ra: f64, dec: f64, slewing: bool) -> Arc<MountSnapshot> {
        let (utc, mono) = at(clock);
        let mut snap = MountSnapshot::default();
        snap.state.ra_h = Observed::fresh(ra, Source::DeviceState, utc, mono);
        snap.state.dec_deg = Observed::fresh(dec, Source::DeviceState, utc, mono);
        snap.state.slewing = Observed::fresh(slewing, Source::DeviceState, utc, mono);
        snap.state.tracking = Observed::fresh(true, Source::DeviceState, utc, mono);
        snap.capabilities.native_frame = EqFrame::JNowApparent;
        snap.set_published(utc, mono);
        Arc::new(snap)
    }

    fn focuser_sample(clock: &ManualClock, position: i32, moving: bool) -> Arc<FocuserSnapshot> {
        let (utc, mono) = at(clock);
        let mut snap = FocuserSnapshot::default();
        snap.state.position = Observed::fresh(position, Source::Property, utc, mono);
        snap.state.is_moving = Observed::fresh(moving, Source::Property, utc, mono);
        snap.logical_position = Some(position);
        snap.set_published(utc, mono);
        Arc::new(snap)
    }

    #[test]
    fn the_empty_hub_is_inert_and_reports_nothing_connected() {
        let hub = Arc::new(EquipmentHub::empty());
        assert!(hub.is_inert());
        assert!(!hub.start(), "the inert hub spawns nothing");
        let snap = hub.snapshot();
        assert!(snap.mount.is_none() && snap.links.is_empty());
        assert_eq!(snap.safety.state, SafetyGate::None);
        assert!(!snap.busy.any());
        let ambient = hub.ambient_temperature();
        assert!(ambient.value.is_none() && ambient.state == ObsState::NotSet);
        let coc = hub.coc_context();
        assert_eq!(coc.corroboration.level, CorroborationLevel::None);
        assert!(
            coc.sun_alt_deg.is_none(),
            "never assumed without an ephemeris"
        );
        let now = Utc::now();
        let window = hub.window(now - chrono::Duration::seconds(10), now);
        assert!(window.finalised);
        assert!(window.mount.start.is_none());
        assert_eq!(
            hub.quiescent(&[DeviceKind::Mount], now, now),
            Quiescence::Unknown {
                reason: "mount_no_data".into()
            }
        );
        assert!(hub.resume_blocks().borrow().is_empty());
    }

    #[tokio::test]
    async fn a_late_event_from_an_old_connection_id_is_ignored() {
        let (hub, clock, sink) = hub_with_clock();
        let mut rx = hub.subscribe();
        let (old, _old_fake) = fake_mount(MountSnapshot::default());
        hub.attach(old.clone());
        let event = |id| {
            EquipmentEvent::new(EventKind::SlewStarted, clock.as_ref())
                .for_device(DeviceKind::Mount, id)
        };
        assert!(hub.emit(event(old.connection_id())).is_some());
        // The mount reconnects with a new connection id.
        let (new, _new_fake) = fake_mount(MountSnapshot::default());
        hub.attach(new.clone());
        assert!(
            hub.emit(event(old.connection_id())).is_none(),
            "late event dropped"
        );
        assert!(hub.emit(event(new.connection_id())).is_some());
        assert_eq!(hub.bus().dropped_stale(), 1);
        // After detaching, the new connection's late events are dropped too.
        assert!(hub.detach(DeviceKind::Mount, new.connection_id()));
        assert!(hub.emit(event(new.connection_id())).is_none());
        // Gate-wide events (no connection id) always pass.
        assert!(hub
            .emit(EquipmentEvent::new(
                EventKind::Estop { results: vec![] },
                clock.as_ref()
            ))
            .is_some());
        let first = rx.recv().await.unwrap();
        let second = rx.recv().await.unwrap();
        assert_eq!(first.connection_id, Some(old.connection_id()));
        assert_eq!(second.seq, first.seq + 1);
        assert_eq!(
            sink.event_names(),
            vec!["slew_started", "slew_started", "estop"]
        );
    }

    #[test]
    fn sequence_numbers_continue_and_guide_steps_are_not_journaled() {
        let (hub, clock, sink) = hub_with_clock();
        hub.bus().continue_after(41);
        let e = hub
            .emit(EquipmentEvent::new(EventKind::Homed, clock.as_ref()))
            .unwrap();
        assert_eq!(e.seq, 42);
        hub.bus().continue_after(10);
        let step = hub
            .emit(EquipmentEvent::new(
                EventKind::GuideStep {
                    step: Box::default(),
                },
                clock.as_ref(),
            ))
            .unwrap();
        assert_eq!(step.seq, 43, "numbering never moves backwards");
        assert_eq!(
            sink.event_names(),
            vec!["homed"],
            "guide steps go to guide_steps"
        );
        assert_eq!(hub.events_since(0, &[], 500).len(), 1);
        assert_eq!(hub.events_since(42, &[], 500).len(), 0);
        let names = vec!["homed".to_string()];
        assert_eq!(hub.events_since(0, &names, 500).len(), 1);
    }

    #[test]
    fn unknown_outcomes_lost_devices_and_limits_raise_resume_blocks() {
        let (hub, clock, sink) = hub_with_clock();
        let (mount, _fake) = fake_mount(MountSnapshot::default());
        hub.attach(mount.clone());
        let mut blocks = hub.resume_blocks();
        hub.emit(
            EquipmentEvent::new(
                EventKind::OutcomeUnknown {
                    op: OpKind::Slew,
                    reason: "lost mid-slew".into(),
                },
                clock.as_ref(),
            )
            .for_device(DeviceKind::Mount, mount.connection_id()),
        );
        hub.emit(
            EquipmentEvent::new(
                EventKind::DeviceLost {
                    class: FailureClass::Timeout,
                    reason: Some("silent".into()),
                },
                clock.as_ref(),
            )
            .for_device(DeviceKind::Mount, mount.connection_id()),
        );
        hub.emit(EquipmentEvent::new(
            EventKind::LimitReached {
                kind: crate::equipment::events::LimitKind::Altitude,
                value: Some(9.0),
                limit: 10.0,
            },
            clock.as_ref(),
        ));
        assert!(blocks.has_changed().unwrap());
        let kinds: Vec<_> = blocks
            .borrow_and_update()
            .iter()
            .map(|b| b.kind.clone())
            .collect();
        assert_eq!(
            kinds,
            vec![
                ResumeBlockKind::OutcomeUnknown {
                    device: DeviceKind::Mount,
                    op: OpKind::Slew
                },
                ResumeBlockKind::MountLost,
                ResumeBlockKind::LimitReached
            ]
        );
        assert_eq!(
            sink.event_names()
                .into_iter()
                .filter(|n| *n == "resume_blocked")
                .count(),
            3
        );
        // The refresh fallback does not raise a second mount_lost.
        hub.refresh();
        assert_eq!(hub.resume().active().len(), 3);
        // Operator clear through the gate is audited and evented.
        let block = hub.resume().active()[1].clone();
        hub.gate()
            .clear_resume_block(block.id, CommandSource::Operator { overriding: false })
            .unwrap();
        assert!(sink.event_names().contains(&"resume_cleared"));
        assert_eq!(hub.resume().active().len(), 2);
    }

    #[test]
    fn a_lost_guider_without_an_event_is_blocked_once_by_refresh() {
        let (hub, _clock, _sink) = hub_with_clock();
        let (guider, fake) = fake_guider(GuiderSnapshot::default());
        hub.attach(guider);
        fake.set_link(LinkState::Lost);
        hub.refresh();
        hub.refresh();
        let active = hub.resume().active();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].kind, ResumeBlockKind::GuiderLost);
        // Cleared by the operator, it is not re-raised for the same connection.
        hub.gate()
            .clear_resume_block(active[0].id, CommandSource::Operator { overriding: false })
            .unwrap();
        hub.refresh();
        assert!(hub.resume().active().is_empty());
    }

    #[test]
    fn window_interpolates_mount_position_across_ra_wrap() {
        let (hub, clock, _sink) = hub_with_clock();
        let (mount, _fake) = fake_mount(MountSnapshot::default());
        let ring = hub.attach(mount);
        let t0 = clock.now_utc();
        for i in 0..12 {
            clock.advance_ms(1000);
            let ra = EqCoord::wrap_ra_h(23.8 + 0.05 * f64::from(i));
            ring.push(mount_sample(&clock, ra, 45.0, false));
        }
        // RA 23.90h at t0+3s, 23.95h at t0+4s, 0.00h at t0+5s, 0.05h at t0+6s.
        let start = t0 + chrono::Duration::seconds(3);
        let end = t0 + chrono::Duration::seconds(6);
        let window = hub.window(start, end);
        // Mid-exposure (t0+4.5s) lies between 23.95h and 0h: the shortest way
        // round gives 23.975h, not the 11.975h a naive average would.
        let mid = window.mount_mid.unwrap();
        assert!((mid.ra_h - 23.975).abs() < 1e-9, "{mid:?}");
        assert!((mid.dec_deg - 45.0).abs() < 1e-12);
        assert_eq!(mid.frame, EqFrame::JNowApparent);
        assert!(window.finalised, "a post-end sample exists");
        assert_eq!(window.mount.samples_n, 4);
        assert!(!window.moved.mount);
        assert_eq!(window.motion.len(), 1, "tracking throughout, no slew");
        assert_eq!(window.motion[0].tracking, Some(true));
        assert!(window.quiescence.is_quiescent());
        let quarter = mount_position_at(
            &hub.rings().mount,
            t0 + chrono::Duration::milliseconds(6500),
        )
        .unwrap();
        assert!((quarter.ra_h - 0.075).abs() < 1e-9, "{quarter:?}");
    }

    #[test]
    fn window_finalises_after_the_grace_and_marks_missing_end_states_stale() {
        let (hub, clock, _sink) = hub_with_clock();
        let (mount, _fake) = fake_mount(MountSnapshot::default());
        let ring = hub.attach(mount);
        let t0 = clock.now_utc();
        clock.advance_ms(500);
        ring.push(mount_sample(&clock, 1.0, 1.0, false));
        let start = t0 + chrono::Duration::seconds(1);
        let end = t0 + chrono::Duration::seconds(3);
        clock.advance_ms(3000);
        let open = hub.window(start, end);
        assert!(!open.finalised, "waiting for a post-end mount sample");
        assert!(open.stale_kinds.is_empty());
        clock.advance_ms(5000);
        let late = hub.window(start, end);
        assert!(late.finalised);
        assert!(late.mount.end_stale);
        assert_eq!(late.stale_kinds, vec![DeviceKind::Mount]);
    }

    #[test]
    fn quiescence_reports_motion_events_staleness_and_settle_time() {
        let (hub, clock, _sink) = hub_with_clock();
        let (focuser, focuser_fake) = fake_focuser(FocuserSnapshot::default());
        let ring = hub.attach(focuser.clone());
        let t0 = clock.now_utc();
        // Moving at 1 s and 2 s, still from 3 s.
        for (sec, moving) in [(1, true), (2, true), (3, false), (4, false), (5, false)] {
            clock.set_utc(t0 + chrono::Duration::seconds(sec));
            clock.advance_ms(0);
            let snap = {
                let (utc, _) = at(&clock);
                let mono = u64::try_from(sec).unwrap() * 1_000_000_000;
                let mut s = (*focuser_sample(&clock, 1000 + sec as i32, moving)).clone();
                s.set_published(utc, mono);
                Arc::new(s)
            };
            ring.push(snap);
        }
        let start = t0 + chrono::Duration::milliseconds(3500);
        let end = t0 + chrono::Duration::milliseconds(4500);
        assert_eq!(
            hub.quiescent(&[DeviceKind::Focuser], start, end),
            Quiescence::Quiescent {
                settled_since: t0 + chrono::Duration::seconds(3)
            }
        );
        // An exposure that overlaps the motion is not quiescent.
        let overlap = hub.quiescent(
            &[DeviceKind::Focuser],
            t0 + chrono::Duration::milliseconds(1500),
            t0 + chrono::Duration::milliseconds(2500),
        );
        assert!(matches!(
            overlap,
            Quiescence::Moved {
                kind: DeviceKind::Focuser,
                ..
            }
        ));
        // A motion event inside the window carries its sequence number.
        clock.set_utc(t0 + chrono::Duration::milliseconds(4000));
        let ev = hub
            .emit(
                EquipmentEvent::new(
                    EventKind::FocuserMoveStarted {
                        move_id: Uuid::new_v4(),
                        from: Some(1004),
                    },
                    clock.as_ref(),
                )
                .for_device(DeviceKind::Focuser, focuser.connection_id()),
            )
            .unwrap();
        assert_eq!(
            hub.quiescent(&[DeviceKind::Focuser], start, end),
            Quiescence::Moved {
                kind: DeviceKind::Focuser,
                event_seq: ev.seq
            }
        );
        // No sample covering the start: unknown, never assumed still.
        let early = hub.quiescent(
            &[DeviceKind::Focuser],
            t0 - chrono::Duration::seconds(30),
            t0,
        );
        assert!(matches!(early, Quiescence::Unknown { reason } if reason.contains("no_sample")));
        // A lost device is unknown.
        focuser_fake.set_link(LinkState::Lost);
        assert!(matches!(
            hub.quiescent(&[DeviceKind::Focuser], start, end),
            Quiescence::Unknown { reason } if reason == "focuser_lost"
        ));
        // Weather never blocks quiescence; an empty list is vacuously quiet.
        assert!(hub
            .quiescent(&[DeviceKind::Weather], start, end)
            .is_quiescent());
    }

    #[test]
    fn window_guide_statistics_exclude_dithers_and_match_a_reference() {
        let (hub, clock, _sink) = hub_with_clock();
        let start = clock.now_utc();
        let values = [(0.5, -0.2), (-0.3, 0.4), (0.1, 0.0), (0.2, -0.1)];
        for (ra, dec) in values {
            clock.advance_ms(1000);
            hub.emit(EquipmentEvent::new(
                EventKind::GuideStep {
                    step: Box::new(GuideStepSample {
                        ra_raw: Some(ra),
                        dec_raw: Some(dec),
                        ra_duration_ms: Some(120),
                        pixel_scale_arcsec: Some(2.0),
                        phase: GuidePhase::Guiding,
                        ..GuideStepSample::default()
                    }),
                },
                clock.as_ref(),
            ));
        }
        clock.advance_ms(1000);
        hub.emit(EquipmentEvent::new(
            EventKind::GuideStep {
                step: Box::new(GuideStepSample {
                    ra_raw: Some(9.0),
                    dec_raw: Some(9.0),
                    phase: GuidePhase::Dithering,
                    ..GuideStepSample::default()
                }),
            },
            clock.as_ref(),
        ));
        hub.emit(EquipmentEvent::new(
            EventKind::Dithered { dx: 3.0, dy: 1.0 },
            clock.as_ref(),
        ));
        let end = clock.now_utc();
        let window = hub.window(start, end);
        let rms = window.guide.rms.unwrap();
        // Reference: population standard deviation of the guiding steps only.
        let n = values.len() as f64;
        let mean_ra = values.iter().map(|v| v.0).sum::<f64>() / n;
        let sd_ra = (values.iter().map(|v| (v.0 - mean_ra).powi(2)).sum::<f64>() / n).sqrt();
        let rms0_dec = (values.iter().map(|v| v.1 * v.1).sum::<f64>() / n).sqrt();
        assert!((rms.ra_px - sd_ra).abs() < 1e-12);
        assert!((rms.rms0_dec_px - rms0_dec).abs() < 1e-12);
        assert!((rms.ra_arcsec.unwrap() - 2.0 * sd_ra).abs() < 1e-12);
        assert_eq!(rms.steps_n, 4);
        assert!((rms.peak_ra_px - 0.5).abs() < 1e-12);
        assert!(window.guide.dither_in_window);
        assert_eq!(window.guide.pulses_n, 4);
        assert!(guide_rms(&[]).is_none());
        let unscaled = guide_rms(&[(1.0, 1.0, None), (1.0, 1.0, Some(1.0))]).unwrap();
        assert!(unscaled.total_arcsec.is_none(), "arcsec needs every scale");
    }

    #[test]
    fn ambient_temperature_follows_the_h6_priority() {
        let (hub, clock, _sink) = hub_with_clock();
        assert!(hub.ambient_temperature().value.is_none());
        // Only the camera sensor: published, flagged as a fallback.
        hub.set_camera_sensor_temperature(
            Observed::fresh_at(-5.0, Source::Property, clock.as_ref()),
            Some("R50".into()),
        );
        let reading = hub.ambient_temperature().value.unwrap();
        assert_eq!(reading.source, TemperatureSourceKind::CameraSensor);
        assert!(reading.camera_sensor_fallback);
        // A focuser probe of unknown role is skipped; a tube probe wins.
        let (focuser, focuser_fake) = fake_focuser(FocuserSnapshot::default());
        hub.attach(focuser.clone());
        focuser_fake.update(|s| {
            s.state.temperature_c = Observed::fresh_at(4.0, Source::Property, clock.as_ref());
        });
        assert_eq!(
            hub.ambient_temperature().value.unwrap().source,
            TemperatureSourceKind::CameraSensor
        );
        focuser_fake.update(|s| s.temperature_role = TemperatureRole::Tube);
        let reading = hub.ambient_temperature();
        let value = reading.value.as_ref().unwrap();
        assert_eq!(value.source, TemperatureSourceKind::FocuserProbe);
        assert_eq!(value.role, TemperatureRole::Tube);
        assert_eq!(value.identity_key.as_deref(), Some("sim:focuser"));
        assert!(!value.camera_sensor_fallback);
        assert!(reading.age_ms(clock.now_mono_ns()).is_some());
        // Weather wins when fresh; a stale weather value loses to a fresh probe.
        let (weather, weather_fake) = fake_weather(WeatherSnapshot::default());
        hub.attach(weather);
        weather_fake.update(|s| {
            s.reading.temperature_c = Observed::fresh_at(2.5, Source::Property, clock.as_ref());
        });
        assert_eq!(
            hub.ambient_temperature().value.unwrap().source,
            TemperatureSourceKind::Weather
        );
        weather_fake.update(|s| {
            s.reading.temperature_c = s.reading.temperature_c.clone().into_stale();
        });
        assert_eq!(
            hub.ambient_temperature().value.unwrap().source,
            TemperatureSourceKind::FocuserProbe
        );
        // NaN is never a temperature.
        focuser_fake.update(|s| {
            s.state.temperature_c = Observed::fresh_at(f64::NAN, Source::Property, clock.as_ref());
        });
        assert_eq!(
            hub.ambient_temperature().value.unwrap().source,
            TemperatureSourceKind::Weather,
            "stale weather beats an invalid probe"
        );
    }

    struct FixedSky(SkyGeometry);

    impl SkyEphemeris for FixedSky {
        fn sky(&self, _at: DateTime<Utc>, _site: &Site) -> Option<SkyGeometry> {
            Some(self.0)
        }
    }

    #[test]
    fn coc_context_collects_pointing_sky_corroboration_and_dew_heaters() {
        let (hub, clock, _sink) = hub_with_clock();
        let site = Site {
            lat_deg: 40.0,
            lon_deg: -75.0,
            elev_m: 100.0,
            source: SiteSource::Config,
        };
        hub.set_site(Some(site));
        let (mount, mount_fake) = fake_mount(MountSnapshot::default());
        hub.attach(mount);
        mount_fake.update(|s| {
            s.state.alt_deg = Observed::fresh_at(55.0, Source::DeviceState, clock.as_ref());
            s.state.tracking = Observed::fresh_at(true, Source::DeviceState, clock.as_ref());
            s.j2000 = Some(EqCoord::new(5.5, 22.0, EqFrame::J2000));
        });
        // No ephemeris: the sky stays unknown.
        assert!(hub.coc_context().twilight.is_none());
        hub.set_ephemeris(Some(Arc::new(FixedSky(SkyGeometry {
            sun_alt_deg: -14.0,
            moon_alt_deg: 20.0,
            moon_illumination: 0.4,
        }))));
        let (weather, weather_fake) = fake_weather(WeatherSnapshot::default());
        hub.attach(weather);
        weather_fake.update(|s| {
            s.reading = WeatherReading {
                humidity_pct: Observed::fresh_at(80.0, Source::Property, clock.as_ref()),
                ..WeatherReading::default()
            };
        });
        let context = hub.coc_context();
        assert_eq!(context.site, Some(site));
        assert_eq!(context.mount_alt_deg.value, Some(55.0));
        assert_eq!(context.pointing_j2000.unwrap().frame, EqFrame::J2000);
        assert_eq!(context.twilight, Some(Twilight::Astronomical));
        assert_eq!(context.moon_illumination, Some(0.4));
        assert_eq!(context.corroboration.level, CorroborationLevel::Partial);
        // A sky sensor plus a safety monitor is full corroboration.
        weather_fake.update(|s| {
            s.reading.cloud_cover_pct = Observed::fresh_at(10.0, Source::Property, clock.as_ref());
        });
        let (monitor, monitor_fake) = fake_safety_monitor(SafetySnapshot::default());
        hub.attach(monitor);
        monitor_fake.update(|s| {
            s.level = Observed::fresh_at(SafetyLevel::Safe, Source::Property, clock.as_ref());
        });
        let context = hub.coc_context();
        assert_eq!(context.corroboration.level, CorroborationLevel::Full);
        assert_eq!(context.corroboration.is_safe.value, Some(true));
        let (switch, switch_fake) = fake_switch(SwitchSnapshot::default());
        hub.attach(switch);
        switch_fake.update(|s| {
            s.channels = vec![
                SwitchChannelState {
                    channel: SwitchChannel {
                        id: 0,
                        name: "Dew A".into(),
                        ..SwitchChannel::default()
                    },
                    value: Observed::fresh_at(0.6, Source::Property, clock.as_ref()),
                    role: Some("dew_heater".into()),
                },
                SwitchChannelState {
                    channel: SwitchChannel {
                        id: 1,
                        name: "12V".into(),
                        ..SwitchChannel::default()
                    },
                    value: Observed::fresh_at(1.0, Source::Property, clock.as_ref()),
                    role: Some("power".into()),
                },
            ];
        });
        let context = hub.coc_context();
        assert_eq!(context.dew_heaters.len(), 1);
        assert_eq!(context.dew_heaters["Dew A"].value, Some(0.6));
        assert_eq!(twilight_for_sun_alt(5.0), Twilight::Day);
        assert_eq!(twilight_for_sun_alt(-20.0), Twilight::Night);
    }

    #[test]
    fn safety_summary_is_fail_safe() {
        let identity =
            crate::equipment::testing::fake_handles::fake_identity(DeviceKind::SafetyMonitor);
        let mut snap = SafetySnapshot::default();
        assert_eq!(derive_safety_summary(None, true).state, SafetyGate::None);
        assert_eq!(
            derive_safety_summary(Some((&snap, LinkState::Ready, &identity)), true).state,
            SafetyGate::Unknown
        );
        snap.level = Observed::fresh(SafetyLevel::Safe, Source::Property, Utc::now(), 1);
        assert_eq!(
            derive_safety_summary(Some((&snap, LinkState::Ready, &identity)), true).state,
            SafetyGate::Safe
        );
        assert_eq!(
            derive_safety_summary(Some((&snap, LinkState::Degraded, &identity)), true).state,
            SafetyGate::Unknown,
            "a degraded monitor is not safe for automatic actions"
        );
        snap.level = Observed::fresh(SafetyLevel::Warning, Source::IndiPush, Utc::now(), 1);
        assert_eq!(
            derive_safety_summary(Some((&snap, LinkState::Degraded, &identity)), true).state,
            SafetyGate::Unsafe
        );
        assert_eq!(
            derive_safety_summary(Some((&snap, LinkState::Ready, &identity)), false).state,
            SafetyGate::Safe
        );
    }

    #[test]
    fn busy_flags_follow_snapshots_and_claims_after_refresh() {
        let (hub, clock, _sink) = hub_with_clock();
        let mut busy = hub.busy();
        let (mount, mount_fake) = fake_mount(MountSnapshot::default());
        hub.attach(mount);
        mount_fake.update(|s| {
            s.active_op = Some(ActiveOp {
                command_id: CommandId::new(),
                kind: OpKind::Park,
                phase: OpPhase::InProgress,
                started_at: clock.now_utc(),
                started_mono_ns: 0,
                deadline_mono_ns: 1,
            });
        });
        hub.refresh();
        assert!(busy.borrow_and_update().parking);
        let camera = FakeCameraActivity::new();
        hub.set_camera(Some(camera.clone()));
        camera.start_exposure(ExposureRef::default());
        hub.refresh();
        assert!(hub.snapshot().busy.exposing);
        let _af = hub.busy_flags().claim(BusyFlag::Autofocus);
        assert!(busy.borrow_and_update().autofocus);
        mount_fake.update(|s| s.active_op = None);
        camera.finish_exposure();
        hub.refresh();
        let now = *busy.borrow_and_update();
        assert!(!now.parking && !now.exposing && now.autofocus);
        // The flip phase event is a mount motion event for quiescence.
        let flip = EquipmentEvent::new(
            EventKind::MeridianFlip {
                phase: FlipPhase::Slew,
            },
            clock.as_ref(),
        );
        assert_eq!(motion_kinds(&flip), &[DeviceKind::Mount]);
    }

    #[test]
    fn exposure_clock_quality_comes_from_the_camera_hook() {
        let (hub, clock, _sink) = hub_with_clock();
        let start = clock.now_utc();
        hub.emit(EquipmentEvent::new(
            EventKind::ExposureStarted {
                exposure: ExposureRef {
                    started_at: Some(start),
                    clock_quality: ClockQuality::HostTriggerBracket,
                    ..ExposureRef::default()
                },
            },
            clock.as_ref(),
        ));
        clock.advance_ms(30_000);
        let window = hub.window(start, clock.now_utc());
        assert_eq!(window.clock_quality, ClockQuality::HostTriggerBracket);
        assert_eq!(window.events.len(), 1);
        let other = hub.window(start + chrono::Duration::seconds(1), clock.now_utc());
        assert_eq!(other.clock_quality, ClockQuality::Unknown);
    }

    #[test]
    fn refresh_projects_the_registry_and_detaches_retired_handles() {
        let (hub, _clock, _sink) = hub_with_clock();
        let registry = Arc::new(RwLock::new(
            crate::hardware::health::DeviceRegistry::default(),
        ));
        hub.set_registry(Some(RegistryBridge::new(Arc::clone(&registry))));
        let (focuser, fake) = fake_focuser(FocuserSnapshot::default());
        registry.write().unwrap().connected(
            DeviceKind::Focuser,
            focuser.connection_id().to_string(),
            None,
        );
        hub.attach(focuser.clone());
        fake.set_link(LinkState::Degraded);
        hub.refresh();
        let value = registry
            .read()
            .unwrap()
            .snapshot(DeviceKind::Focuser)
            .unwrap();
        assert_eq!(value["state"], serde_json::json!("degraded"));
        assert!(registry.read().unwrap().0[&DeviceKind::Focuser].connected);
        fake.retire();
        hub.refresh();
        assert!(
            hub.handle::<FocuserHandle>().is_none(),
            "retired handles are detached"
        );
        assert!(!registry.read().unwrap().0[&DeviceKind::Focuser].connected);
    }
}

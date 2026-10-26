//! Driver and integration traits (spec §2.4). Frozen by WP0.
//!
//! **Drivers are synchronous and are called only from their worker thread**
//! (D3). They never run on a tokio worker and never under a slot mutex.
//!
//! Poll contract: the worker calls [`Device::poll`] once per tick. `poll`
//! does that tick's I/O (one Alpaca `devicestate` GET, or the per-member
//! reads of a Platform 6 device, or a non-blocking INDI cache read) and keeps
//! the values. The typed accessors (`MountDriver::state`, `FocuserDriver::state`,
//! `WeatherDriver::read`, ...) then assemble [`Observed`] values from what
//! `poll` read and must not repeat that I/O. Members a plan did not cover
//! keep their previous value, marked `Stale` once overdue (C-2).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::commands::{GatedCommand, SafetyRejection};
use super::events::{EquipmentEvent, GroundTruthRecord};
use super::snapshot::{
    BusySet, CocContext, CoverCalState, DeviceWindow, DomeState, EquipmentSnapshot, ExposureRef,
    FocuserCapabilities, FocuserState, MountCapabilities, MountState, Quiescence, ResumeBlock,
    RotatorState, SafetyReading, SlotInfo, SwitchChannel, TemperatureReading, WeatherReading,
};
use super::types::{
    AltAz, DeviceIdentity, DeviceKind, DeviceResult, DriveRate, EqCoord, EqFrame, GuideDirection,
    ObsState, Observed, PierSide, Site,
};

// ---------------------------------------------------------------------------
// Connect and poll support types
// ---------------------------------------------------------------------------

/// Which connect path a device took (§2.6 step 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectPath {
    /// Platform 7: PUT `connect`, then poll `connecting`.
    Platform7,
    /// Platform 6: PUT `Connected=true`, then GET `connected`.
    #[default]
    Platform6,
    /// INDI `CONNECTION` switch (device-wide; D16).
    Indi,
    /// PHD2 event server.
    Phd2,
    /// Simulator or in-process driver.
    InProcess,
}

/// Result of [`Device::connect`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ConnectReport {
    pub path: ConnectPath,
    /// `interfaceversion`; a failed read is reported as `Some(1)` (ASCOM
    /// Library rule, §2.6 step 3).
    pub interface_version: Option<i32>,
    /// True when Platform 7 `devicestate` is available and non-empty.
    pub device_state: bool,
    /// Members memoised as unsupported during the connect probes.
    pub unsupported_members: Vec<String>,
    /// Non-fatal findings (name-based rig heuristics may warn, never refuse;
    /// D15).
    pub warnings: Vec<String>,
    pub duration_ms: u64,
}

/// What a worker asks one [`Device::poll`] to read.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PollPlan {
    /// Monotonically increasing tick number for this connection.
    pub tick: u64,
    /// Current cadence (§2.7), after any adaptive reduction.
    pub interval: Duration,
    /// An Achernar-owned operation is running (active cadence).
    pub active: bool,
    /// Use one `devicestate` GET when the device supports it.
    pub prefer_device_state: bool,
    /// Slow members due on this tick (guide rates every 30 s, TempComp
    /// every 10 s, `timesincelastupdate` every 60 s, ...), by member name.
    pub extra_members: Vec<String>,
}

/// One member read by a poll.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawMember {
    /// Lower-case member or property name (`rightascension`, `FILTER_SLOT`).
    pub name: String,
    /// The raw value; `None` when the read failed or was not set.
    pub value: Option<serde_json::Value>,
    pub state: ObsState,
    pub observed_at: Option<DateTime<Utc>>,
    pub mono_ns: Option<u64>,
    pub latency_ms: Option<u32>,
    /// Error text for a failed member read.
    pub error: Option<String>,
}

/// How a poll obtained its values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PollMethod {
    DeviceState,
    #[default]
    PerMember,
    /// INDI cache read; no I/O.
    PushCache,
    InProcess,
}

/// Raw result of one poll, used for latency accounting, adaptive cadence
/// (§2.7) and diagnostics. Typed values come from the kind accessors.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RawPoll {
    pub method: PollMethod,
    pub started_mono_ns: u64,
    pub finished_mono_ns: u64,
    /// DeviceState `TimeStamp`, parsed permissively (§2.14).
    pub device_time: Option<DateTime<Utc>>,
    pub members: Vec<RawMember>,
}

impl RawPoll {
    pub fn duration(&self) -> Duration {
        Duration::from_nanos(self.finished_mono_ns.saturating_sub(self.started_mono_ns))
    }

    /// True when any member read failed (the device counts as degraded for
    /// this tick; §2.8).
    pub fn any_failed(&self) -> bool {
        self.members.iter().any(|m| m.state == ObsState::Error)
    }
}

/// Alpaca timeout classes (C-3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeoutClass {
    /// 5 s: `connect`, `connecting`, `connected`, `interfaceversion`.
    Connect,
    /// 10 s: ordinary members.
    Ordinary,
    /// 100 s: `abortslew`, `findhome`, `moveaxis`, `park`, `unpark`,
    /// `pulseguide`, slews and syncs, focuser `move` and `halt`, rotator moves.
    Long,
}

impl TimeoutClass {
    pub fn duration(self) -> Duration {
        match self {
            TimeoutClass::Connect => Duration::from_secs(5),
            TimeoutClass::Ordinary => Duration::from_secs(10),
            TimeoutClass::Long => Duration::from_secs(100),
        }
    }
}

// ---------------------------------------------------------------------------
// Driver traits
// ---------------------------------------------------------------------------

/// Common device behaviour.
pub trait Device: Send {
    fn identity(&self) -> &DeviceIdentity;
    /// P7 `connect` + `connecting`, else P6 `Connected=true`. Connecting never
    /// moves hardware (D16).
    fn connect(&mut self, timeout: Duration) -> DeviceResult<ConnectReport>;
    /// Subject to the D16 sharing rules (never parks; INDI `CONNECTION` is
    /// left alone unless `indi_disconnect_device`).
    fn disconnect(&mut self) -> DeviceResult<()>;
    fn is_connected(&mut self) -> DeviceResult<bool>;
    /// One DeviceState, or per-member reads (see the module docs).
    fn poll(&mut self, plan: &PollPlan) -> DeviceResult<RawPoll>;
    fn supported_actions(&mut self) -> DeviceResult<Vec<String>>;
    fn action(&mut self, name: &str, params: &str) -> DeviceResult<String>;
    /// Out-of-band stop that never queues behind another command (D4).
    fn abort_path(&self) -> Option<Arc<dyn AbortPath>>;
}

/// Out-of-band abort (D4). Runs on a blocking thread, concurrently with the
/// worker; implementations use their own connection (Alpaca: a second
/// `ureq::Agent` with a 5 s timeout).
pub trait AbortPath: Send + Sync {
    fn abort(&self) -> DeviceResult<()>;
}

pub trait MountDriver: Device {
    /// Can* flags, AlignmentMode, EquatorialSystem, DoesRefraction,
    /// TrackingRates, AxisRates, focal length / aperture claims.
    fn capabilities(&mut self) -> DeviceResult<MountCapabilities>;
    /// RA/Dec (native frame), Alt/Az, LST, UTC, Slewing, Tracking, rate,
    /// AtPark, AtHome, SideOfPier, IsPulseGuiding, guide rates, site.
    fn state(&mut self) -> DeviceResult<MountState>;
    /// `target` is already in the native frame.
    fn slew_async(&mut self, target: EqCoord) -> DeviceResult<()>;
    fn slew_altaz_async(&mut self, target: AltAz) -> DeviceResult<()>;
    fn abort_slew(&mut self) -> DeviceResult<()>;
    fn sync(&mut self, target: EqCoord) -> DeviceResult<()>;
    fn set_tracking(&mut self, on: bool) -> DeviceResult<()>;
    fn set_tracking_rate(&mut self, rate: DriveRate) -> DeviceResult<()>;
    fn park(&mut self) -> DeviceResult<()>;
    fn unpark(&mut self) -> DeviceResult<()>;
    fn set_park(&mut self) -> DeviceResult<()>;
    fn find_home(&mut self) -> DeviceResult<()>;
    fn move_axis(&mut self, axis: u8, rate_deg_s: f64) -> DeviceResult<()>;
    fn pulse_guide(&mut self, dir: GuideDirection, ms: u32) -> DeviceResult<()>;
    /// Alpaca units (deg/s). Snapshots publish arcsec/s (§2.14).
    fn set_guide_rates(&mut self, ra_deg_s: f64, dec_deg_s: f64) -> DeviceResult<()>;
    fn set_site(&mut self, site: &Site) -> DeviceResult<()>;
    fn set_utc(&mut self, utc: DateTime<Utc>) -> DeviceResult<()>;
    fn destination_side_of_pier(&mut self, target: EqCoord) -> DeviceResult<PierSide>;
    fn set_side_of_pier(&mut self, side: PierSide) -> DeviceResult<()>;
}

pub trait FocuserDriver: Device {
    /// absolute, max_step, max_increment, step_size_um, temp_comp_available,
    /// can_sync.
    fn capabilities(&mut self) -> DeviceResult<FocuserCapabilities>;
    /// position, is_moving, temperature, temp_comp.
    fn state(&mut self) -> DeviceResult<FocuserState>;
    /// Only when absolute.
    fn move_absolute(&mut self, target: i32) -> DeviceResult<()>;
    /// Only when relative (Alpaca `Move` semantics).
    fn move_relative(&mut self, steps: i32) -> DeviceResult<()>;
    fn halt(&mut self) -> DeviceResult<()>;
    fn set_temp_comp(&mut self, on: bool) -> DeviceResult<()>;
    /// INDI `FOCUS_SYNC`; `Unsupported` elsewhere (Alpaca has none).
    fn sync_position(&mut self, position: i32) -> DeviceResult<()>;
}

pub trait FilterWheelDriver: Device {
    /// index0, device name, device focus offset.
    fn slots(&mut self) -> DeviceResult<Vec<SlotInfo>>;
    /// `None` = moving. Always 0-based (INDI's 1-based slot is mapped).
    fn position(&mut self) -> DeviceResult<Option<u32>>;
    fn set_position(&mut self, index0: u32) -> DeviceResult<()>;
}

pub trait WeatherDriver: Device {
    fn read(&mut self) -> DeviceResult<WeatherReading>;
    fn refresh(&mut self) -> DeviceResult<()>;
}

pub trait SafetyDriver: Device {
    /// Safe | Warning | Unsafe. Never cached (§3.10).
    fn is_safe(&mut self) -> DeviceResult<SafetyReading>;
}

pub trait RotatorDriver: Device {
    /// position (sky), mechanical, is_moving, reverse, target, can_reverse.
    fn state(&mut self) -> DeviceResult<RotatorState>;
    fn move_relative(&mut self, deg: f64) -> DeviceResult<()>;
    fn move_absolute(&mut self, sky_deg: f64) -> DeviceResult<()>;
    fn move_mechanical(&mut self, deg: f64) -> DeviceResult<()>;
    fn sync(&mut self, sky_deg: f64) -> DeviceResult<()>;
    fn halt(&mut self) -> DeviceResult<()>;
    fn set_reverse(&mut self, on: bool) -> DeviceResult<()>;
}

pub trait CoverCalibratorDriver: Device {
    /// cover_state, cover_moving, calibrator_state, changing, brightness, max.
    fn state(&mut self) -> DeviceResult<CoverCalState>;
    fn open(&mut self) -> DeviceResult<()>;
    fn close(&mut self) -> DeviceResult<()>;
    fn halt_cover(&mut self) -> DeviceResult<()>;
    fn calibrator_on(&mut self, brightness: u32) -> DeviceResult<()>;
    fn calibrator_off(&mut self) -> DeviceResult<()>;
}

pub trait SwitchDriver: Device {
    /// id, name, description, can_write, can_async, min, max, step.
    fn channels(&mut self) -> DeviceResult<Vec<SwitchChannel>>;
    fn read(&mut self, id: u32) -> DeviceResult<f64>;
    /// `setasyncvalue` when can_async, else `setswitchvalue`.
    fn set(&mut self, id: u32, value: f64) -> DeviceResult<()>;
    /// `None` when the channel is not async.
    fn state_change_complete(&mut self, id: u32) -> DeviceResult<Option<bool>>;
}

pub trait DomeDriver: Device {
    /// shutter, azimuth, altitude, slewing, at_park, at_home, slaved, Can*.
    fn state(&mut self) -> DeviceResult<DomeState>;
    fn open_shutter(&mut self) -> DeviceResult<()>;
    fn close_shutter(&mut self) -> DeviceResult<()>;
    fn park(&mut self) -> DeviceResult<()>;
    fn slew_azimuth(&mut self, az: f64) -> DeviceResult<()>;
    fn abort(&mut self) -> DeviceResult<()>;
    fn set_slaved(&mut self, on: bool) -> DeviceResult<()>;
}

// ---------------------------------------------------------------------------
// Transport, conversion, checks and hooks
// ---------------------------------------------------------------------------

/// One Alpaca device's member API. Implemented by `alpaca/client.rs` (WP1A-1)
/// and `testing/fake_transport.rs`.
///
/// `member` is the lower-case member name (`rightascension`). `query` and
/// `form` hold parameters in exact case (`RightAscension`, C-3); the client
/// adds `ClientID` and `ClientTransactionID`. On success the result is the
/// Alpaca `Value` member (`null` for a void PUT). Every failure is already
/// mapped to a [`DeviceError`](super::types::DeviceError) with
/// [`from_http_status`](super::types::DeviceError::from_http_status) or
/// [`from_alpaca_error`](super::types::DeviceError::from_alpaca_error), so
/// drivers see one error vocabulary from the real client and the fake.
pub trait AlpacaTransport: Send + Sync {
    fn get(
        &self,
        member: &str,
        query: &[(&str, String)],
        class: TimeoutClass,
    ) -> DeviceResult<serde_json::Value>;
    fn put(
        &self,
        member: &str,
        form: &[(&str, String)],
        class: TimeoutClass,
    ) -> DeviceResult<serde_json::Value>;
}

/// Equatorial frame conversion. Implemented by `astro/frames.rs` (WP1A-3);
/// [`IdentityConverter`] for tests.
pub trait FrameConverter: Send + Sync {
    fn convert(&self, c: EqCoord, to: EqFrame, at: DateTime<Utc>, site: Option<&Site>) -> EqCoord;
}

/// A gate safety check, registered by a device work package; the gate
/// (WP1A-2) only runs them (§2.9 step 3).
pub trait SafetyCheck: Send + Sync {
    /// Stable name, recorded as `safety_rejected{check}`.
    fn name(&self) -> &'static str;
    fn applies_to(&self, cmd: &GatedCommand) -> bool;
    fn check(&self, cmd: &GatedCommand, snap: &EquipmentSnapshot) -> Result<(), SafetyRejection>;
    /// Notes the gate audits on an accepted command, e.g. `limits_unchecked`
    /// for an operator slew whose altitude could not be computed.
    fn annotations(&self, _cmd: &GatedCommand, _snap: &EquipmentSnapshot) -> Vec<String> {
        Vec::new()
    }
}

/// The camera hook. Cameras stay on OLS (O1); implemented in WP1B-1 over
/// `camera_runtime` and the existing abort; faked in tests.
pub trait CameraActivity: Send + Sync {
    fn exposing(&self) -> Observed<bool>;
    /// started_at, duration, frame/operation id, clock quality.
    fn current_exposure(&self) -> Option<ExposureRef>;
    /// Runs on a blocking worker.
    fn abort_exposure(&self) -> DeviceResult<()>;
    /// For the guide-camera role check (§3.5 G-l).
    fn camera_name(&self) -> Option<String>;
}

/// Injectable time (D22). Production uses
/// [`SystemClock`](super::clock::SystemClock); tests use
/// `testing::manual_clock::ManualClock`.
pub trait Clock: Send + Sync {
    fn now_utc(&self) -> DateTime<Utc>;
    /// Nanoseconds since daemon start.
    fn now_mono_ns(&self) -> u64;
}

/// The model-facing read API (§4.1). Implemented by `hub.rs` (WP1A-2);
/// consumed by models and the API. Models never touch driver objects (D6).
pub trait EquipmentReader: Send + Sync {
    fn snapshot(&self) -> Arc<EquipmentSnapshot>;
    fn subscribe(&self) -> tokio::sync::broadcast::Receiver<Arc<EquipmentEvent>>;
    /// Built from ring buffers; never calls a device.
    fn window(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> DeviceWindow;
    fn quiescent(
        &self,
        kinds: &[DeviceKind],
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Quiescence;
    fn busy(&self) -> tokio::sync::watch::Receiver<BusySet>;
    fn resume_blocks(&self) -> tokio::sync::watch::Receiver<Vec<ResumeBlock>>;
    /// Value + source + identity + age (H6).
    fn ambient_temperature(&self) -> Observed<TemperatureReading>;
    /// H14.
    fn coc_context(&self) -> CocContext;
}

/// Ground-truth sink (WP1A-9). Never blocks; bounded; drops are counted.
pub trait RecorderSink: Send + Sync {
    fn record(&self, rec: GroundTruthRecord);
}

// ---------------------------------------------------------------------------
// Inert implementations
// ---------------------------------------------------------------------------

/// Relabels the coordinate without transforming it. For tests and for an
/// inert hub only: it must never feed a real pointing comparison.
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityConverter;

impl FrameConverter for IdentityConverter {
    fn convert(
        &self,
        c: EqCoord,
        to: EqFrame,
        _at: DateTime<Utc>,
        _site: Option<&Site>,
    ) -> EqCoord {
        EqCoord { frame: to, ..c }
    }
}

/// Discards every record and counts it. Used by an inert hub
/// (`EquipmentHub::empty()`, D11) and by tests that do not inspect records.
#[derive(Debug, Default)]
pub struct NullRecorder {
    dropped: AtomicU64,
}

impl NullRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

impl RecorderSink for NullRecorder {
    fn record(&self, _rec: GroundTruthRecord) {
        self.dropped.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::events::{EventKind, TelemetrySample};

    #[test]
    fn identity_converter_only_relabels() {
        let c = EqCoord::new(5.5, -20.0, EqFrame::JNowApparent);
        let out = IdentityConverter.convert(c, EqFrame::J2000, Utc::now(), None);
        assert_eq!(
            (out.ra_h, out.dec_deg, out.frame),
            (5.5, -20.0, EqFrame::J2000)
        );
    }

    #[test]
    fn null_recorder_counts_and_never_blocks() {
        let sink = NullRecorder::new();
        let clock = crate::equipment::testing::manual_clock::ManualClock::new();
        for _ in 0..3 {
            sink.record(GroundTruthRecord::Event(Arc::new(EquipmentEvent::new(
                EventKind::PollCadenceReduced {
                    interval_ms: 2000,
                    factor: 2,
                },
                &clock,
            ))));
        }
        sink.record(GroundTruthRecord::Telemetry(TelemetrySample {
            source: DeviceKind::Mount,
            at: Utc::now(),
            payload: serde_json::json!({}),
        }));
        assert_eq!(sink.dropped(), 4);
    }

    #[test]
    fn timeout_classes_match_c3() {
        assert_eq!(TimeoutClass::Connect.duration(), Duration::from_secs(5));
        assert_eq!(TimeoutClass::Ordinary.duration(), Duration::from_secs(10));
        assert_eq!(TimeoutClass::Long.duration(), Duration::from_secs(100));
    }
}

//! Capture integration with the equipment layer (equipment spec H1, H16,
//! H17, H22, D7, §3.12). Owner: WP1B-3.
//!
//! Cameras stay on OpenLiveStacker (O1). This module ties each exposure to
//! the equipment layer without adding device I/O to a capture (D7):
//!
//! - **Gate** ([`authorize_capture`]). A capture is a `GatedOp::Capture` on
//!   the camera domain: audited (H21), refused with HTTP 423 for an
//!   automatic source while a resume block is active, and with 422 while
//!   the safety state is not confirmed safe (H22, §2.8). Operator captures
//!   run and carry the active blocks into their provenance.
//! - **Exposure hooks** ([`CaptureHooks`]). The `exposing` busy flag is held
//!   from the trigger until the exposure returns (H20), with
//!   `exposure_started` before it and `exposure_completed{frame_id}` after
//!   the commit (H17; the recorder links pending dithers to that frame,
//!   H10). `stream_restarted` and `exposure_settings_changed` are emitted by
//!   the stream and control paths (H2 baselines).
//! - **Window** ([`assemble`]). After the exposure the hub's `window()` is
//!   read from the ring buffers. It gives the frame's
//!   `exposure_device_windows` row (written in the frame's own commit
//!   transaction by `capture_store`), its `acquisition.equipment` block and
//!   its FITS keywords. Rewritten once the window is final ([`finalise`]).
//! - **Clock quality per driver and path** ([`clock_quality`], §2.14): the
//!   OLS still path brackets the trigger with the host clock; every OLS
//!   stream driver stamps a frame with the host clock when it is delivered,
//!   so a stream frame's window is `arrival - (exposure + frame interval)`
//!   to its arrival, conservatively.
//! - **Filter** comes from the verified wheel when one is connected and
//!   verified the slot for the whole exposure; otherwise it is the client
//!   label, marked unverified. **Focuser position** comes from the window's
//!   start and end, never from a read after the commit, and only while the
//!   position is reliable (v6 §7.6.2).
//! - **End state (D7).** The end state of the focuser and the wheel is the
//!   last sample at or before the exposure end, confirmed by the first
//!   sample after it. When that sample differs and nothing shows the change
//!   began after the end, or when it never arrived (`Stale`), the end state
//!   is unknown: no verified filter, no end position, `focuser_moved` null.
//!   Motion at the start sample (a move already running) counts as motion.
//!
//! A value that is not known is omitted (FITS) or null (rows); it is never
//! written as 0 (D5).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::equipment::astro;
use crate::equipment::busy::{BusyFlag, BusyGuard};
use crate::equipment::commands::{GateError, GatedCommand, GatedOp};
use crate::equipment::events::{EquipmentEvent, EventKind, ExposureWindowRow, GroundTruthRecord};
use crate::equipment::gate::StepOwner;
use crate::equipment::guider::service::GuiderWindows;
use crate::equipment::handle::Permit;
use crate::equipment::hub::{max_read_age, motion_kinds, EquipmentHub};
use crate::equipment::ring::{midpoint, DeviceSnapshot, Ring, Timestamped};
use crate::equipment::snapshot::{
    CoverCalibratorSnapshot, DeviceWindow, ExposureRef, FilterWheelSnapshot, FocuserSnapshot,
    GuideWindowStats, MotionInterval, MountSnapshot, PositionSource, Quiescence, ResumeBlock,
    WindowSamples,
};
use crate::equipment::traits::EquipmentReader;
use crate::equipment::types::{
    kind_name, AutonomyMode, ClockQuality, CommandId, CommandSource, DeviceKind, EqCoord, EqFrame,
    Observed, PierSide, Priority, Site, SiteSource, TemperatureRole,
};
use crate::fits::FitsCard;
use crate::state::{AppState, CameraRuntimeState};

/// Schema tag of `frames.acquisition_json.equipment`.
pub(crate) const EQUIPMENT_SCHEMA: &str = "capture_equipment.v1";

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

fn utc_of(unix_seconds: f64) -> DateTime<Utc> {
    if !unix_seconds.is_finite() {
        return Utc::now();
    }
    let secs = unix_seconds.floor();
    let nanos = ((unix_seconds - secs) * 1e9)
        .round()
        .clamp(0.0, 999_999_999.0) as u32;
    DateTime::from_timestamp(secs as i64, nanos).unwrap_or_else(Utc::now)
}

fn seconds(value: f64) -> chrono::Duration {
    if !value.is_finite() || value <= 0.0 {
        return chrono::Duration::zero();
    }
    chrono::Duration::from_std(Duration::from_secs_f64(value.min(1.0e7)))
        .unwrap_or(chrono::Duration::zero())
}

fn read_age(kind: DeviceKind) -> chrono::Duration {
    chrono::Duration::from_std(max_read_age(kind)).unwrap_or(chrono::Duration::MAX)
}

/// `sample` when it was taken within `kind`'s read age of `at`: a ring
/// sample from a device that stopped publishing is not that device's state
/// during the exposure (D5).
fn near<S: Timestamped>(sample: Option<&S>, at: DateTime<Utc>, kind: DeviceKind) -> Option<&S> {
    sample.filter(|s| (s.at_utc() - at).abs() <= read_age(kind))
}

// ---------------------------------------------------------------------------
// Capture source and the gate (H21, H22)
// ---------------------------------------------------------------------------

/// Who asked for a capture (H22). The direct API and the relay are operator
/// sources and the default. A sequencer, trigger or policy names itself
/// (`{"kind": "trigger", "rule": "..."}`), so that a resume block refuses
/// it with HTTP 423 instead of letting it resume automatically.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CaptureSource {
    #[default]
    Operator,
    Trigger {
        rule: String,
    },
    Policy {
        model: String,
        version: String,
    },
}

impl CaptureSource {
    /// The gate's source for this capture when it arrives over the relay
    /// (`relay`) or the direct API: an operator request is audited as the
    /// transport it came through; a trigger or policy keeps its own
    /// (automatic) source, so a resume block still refuses it.
    pub(crate) fn command_source_over(&self, relay: bool) -> CommandSource {
        match self {
            CaptureSource::Operator if relay => CommandSource::Relay { overriding: false },
            CaptureSource::Operator => CommandSource::Operator { overriding: false },
            CaptureSource::Trigger { rule } => CommandSource::Trigger { rule: rule.clone() },
            CaptureSource::Policy { model, version } => CommandSource::Policy {
                model: model.clone(),
                version: version.clone(),
            },
        }
    }
}

/// An authorised capture: the gated command (for the executed and outcome
/// audit stages), the permit holding the camera domain until it drops, and
/// the resume blocks active when it was authorised.
pub(crate) struct CaptureAuthorization {
    pub command: GatedCommand,
    /// Held, never read: the camera domain is released when it drops. A
    /// step's permit holds nothing; its parent's holds the domain.
    _permit: Permit,
    pub blocks: Vec<ResumeBlock>,
}

/// Authorise a capture through the command gate. Automatic sources are
/// refused while a resume block is active (423) or while the safety state
/// is not confirmed safe (422); every stage is audited.
///
/// With `parent`, the capture is one step of a compound operation that
/// already owns the camera domain (a center loop's frame): it is authorised
/// as a child of that operation (O11 b), so its own checks run but it does
/// not contend with the parent for the camera. Without one it claims the
/// camera domain itself, so an operator capture while a center loop holds
/// the camera is refused with 409.
pub(crate) fn authorize_capture(
    hub: &EquipmentHub,
    source: &CommandSource,
    parent: Option<&StepOwner>,
    capture_type: &str,
    params: Value,
) -> Result<CaptureAuthorization, GateError> {
    let mode = if source.is_automatic() {
        AutonomyMode::Bounded
    } else {
        AutonomyMode::Manual
    };
    let command = GatedCommand::new(
        GatedOp::Capture {
            capture_type: Some(capture_type.to_string()),
        },
        source.clone(),
        mode,
        Priority::Exposure,
    )
    .with_params(params);
    let permit = match parent {
        Some(parent) => hub.gate().child_permit_of(parent, &command)?,
        None => hub.gate().authorize(&command)?,
    };
    let blocks = hub.resume().active();
    Ok(CaptureAuthorization {
        command,
        _permit: permit,
        blocks,
    })
}

// ---------------------------------------------------------------------------
// Clock quality and the exposure interval (§2.14, A06)
// ---------------------------------------------------------------------------

/// How the camera delivered the exposure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AcquisitionPath {
    /// A dedicated still (gphoto2): the driver brackets the trigger.
    Still,
    /// The stream is the exposure (ASI, SVBony, ToupTek, UVC, Alpaca, ...).
    Stream,
}

/// Exposure clock quality per OLS driver and path. The OLS still path reads
/// the host clock immediately around the trigger (a bracket, not the
/// shutter time). Every OLS stream driver stamps a frame with the host
/// clock when the SDK delivers it, so stream frames carry an arrival time,
/// never a driver exposure start. The OLS simulator replays recorded frames.
pub(crate) fn clock_quality(driver_id: Option<&str>, path: AcquisitionPath) -> ClockQuality {
    match (driver_id, path) {
        (None, _) => ClockQuality::Unknown,
        (Some("sim" | "simulator"), _) => ClockQuality::Simulator,
        (Some(_), AcquisitionPath::Still) => ClockQuality::HostTriggerBracket,
        (Some(_), AcquisitionPath::Stream) => ClockQuality::HostArrival,
    }
}

/// The span the exposure's window covers, and how it was obtained.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ExposureInterval {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub clock_quality: ClockQuality,
    pub basis: &'static str,
    /// The applied exposure, seconds.
    pub exposure_s: f64,
    /// The stream's measured frame period, when two consecutive frames
    /// were seen.
    pub frame_interval_s: Option<f64>,
}

impl ExposureInterval {
    /// A still: the driver's trigger bracket, which contains the exposure.
    pub fn still(
        started_unix: f64,
        completed_unix: f64,
        exposure_s: f64,
        clock_quality: ClockQuality,
    ) -> Self {
        let (a, b) = (utc_of(started_unix), utc_of(completed_unix));
        Self {
            start: a.min(b),
            end: a.max(b),
            clock_quality,
            basis: "driver_trigger_bracket",
            exposure_s: if exposure_s.is_finite() {
                exposure_s.max(0.0)
            } else {
                0.0
            },
            frame_interval_s: None,
        }
    }

    /// A stream frame that arrived at `arrived_unix` (OLS's host-clock
    /// delivery stamp, the frame's DATE-OBS). Its arrival can lag the end
    /// of its exposure by up to a frame interval, so the window reaches back
    /// `exposure + frame interval` from the arrival (the H3 rule for
    /// `host_arrival`). Without a measured interval the exposure stands in
    /// for it (a stream's frame period is at least its exposure).
    pub fn stream(
        arrived_unix: f64,
        exposure_s: f64,
        frame_interval_s: Option<f64>,
        clock_quality: ClockQuality,
    ) -> Self {
        let exposure_s = if exposure_s.is_finite() {
            exposure_s.max(0.0)
        } else {
            0.0
        };
        let frame_interval_s = frame_interval_s.filter(|s| s.is_finite() && *s > 0.0);
        let end = utc_of(arrived_unix);
        let lead = exposure_s + frame_interval_s.unwrap_or(exposure_s);
        Self {
            start: end - seconds(lead),
            end,
            clock_quality,
            basis: "arrival_minus_exposure_and_frame_interval",
            exposure_s,
            frame_interval_s,
        }
    }
}

// ---------------------------------------------------------------------------
// Camera events and the exposure hooks (H17, H20, H2)
// ---------------------------------------------------------------------------

fn camera_event(hub: &EquipmentHub, connection: Option<&str>, kind: EventKind) -> EquipmentEvent {
    let mut event = EquipmentEvent::new(kind, hub.clock().as_ref());
    event.device_kind = Some(DeviceKind::Camera);
    event.connection_id = connection.and_then(|id| Uuid::parse_str(id).ok());
    event
}

/// `stream_restarted{generation}`: a new stream generation began, so frame
/// baselines from the previous one no longer apply (H2).
pub(crate) fn emit_stream_restarted(hub: &EquipmentHub, connection: Option<&str>, generation: u64) {
    hub.emit(camera_event(
        hub,
        connection,
        EventKind::StreamRestarted { generation },
    ));
}

/// The exposure settings last applied per camera connection, so that
/// `exposure_settings_changed` is emitted on a change and not on every
/// capture. Keyed by connection id, so a reconnect starts from unknown.
static APPLIED_SETTINGS: Mutex<BTreeMap<String, BTreeMap<String, Value>>> =
    Mutex::new(BTreeMap::new());

/// Connections remembered; only one camera is connected at a time.
const REMEMBERED_CAMERAS: usize = 16;

fn same_setting(a: &Value, b: &Value) -> bool {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => (x - y).abs() <= 1e-9 * x.abs().max(y.abs()).max(1.0),
        _ => a == b,
    }
}

/// Merge `update` into what is known of `camera`'s exposure settings and
/// return the keys that changed (`{key: {from, to}}`, `from` null when it
/// was unknown), or `None` when nothing changed.
pub(crate) fn note_settings(camera: &str, update: &BTreeMap<String, Value>) -> Option<Value> {
    let mut all = lock(&APPLIED_SETTINGS);
    if !all.contains_key(camera) && all.len() >= REMEMBERED_CAMERAS {
        if let Some(oldest) = all.keys().next().cloned() {
            all.remove(&oldest);
        }
    }
    let known = all.entry(camera.to_string()).or_default();
    let mut changes = serde_json::Map::new();
    for (key, value) in update {
        if value.is_null() {
            continue;
        }
        let previous = known.get(key);
        if previous.is_some_and(|p| same_setting(p, value)) {
            continue;
        }
        changes.insert(
            key.clone(),
            json!({"from": previous.cloned().unwrap_or(Value::Null), "to": value}),
        );
        known.insert(key.clone(), value.clone());
    }
    (!changes.is_empty()).then_some(Value::Object(changes))
}

/// The settings `note_settings` knows for `camera`.
fn known_settings(camera: &str) -> Value {
    json!(lock(&APPLIED_SETTINGS)
        .get(camera)
        .cloned()
        .unwrap_or_default())
}

/// Record `update` for the connected camera and emit
/// `exposure_settings_changed` when it changed anything (H2 baselines).
pub(crate) fn settings_applied(
    hub: &EquipmentHub,
    runtime: &CameraRuntimeState,
    update: BTreeMap<String, Value>,
    cause: &str,
    capture_id: Option<&str>,
) {
    let Some(connection) = runtime.connection_id.as_deref() else {
        return;
    };
    let Some(changed) = note_settings(connection, &update) else {
        return;
    };
    let settings = json!({
        "cause": cause,
        "camera": runtime.camera_name,
        "driver": runtime.driver_id,
        "capture_id": capture_id,
        "changed": changed,
        "settings": known_settings(connection),
    });
    hub.emit(camera_event(
        hub,
        Some(connection),
        EventKind::ExposureSettingsChanged { settings },
    ));
}

/// One capture's exposure hooks.
pub(crate) struct CaptureHooks {
    hub: Arc<EquipmentHub>,
    connection: Option<String>,
    driver_id: Option<String>,
    capture_id: String,
    frame_id: String,
    exposing: Option<BusyGuard>,
    stream_generation: Option<u64>,
}

impl CaptureHooks {
    pub fn new(
        hub: Arc<EquipmentHub>,
        runtime: &CameraRuntimeState,
        capture_id: &str,
        frame_id: &str,
    ) -> Self {
        Self {
            hub,
            connection: runtime.connection_id.clone(),
            driver_id: runtime.driver_id.clone(),
            capture_id: capture_id.to_string(),
            frame_id: frame_id.to_string(),
            exposing: None,
            stream_generation: None,
        }
    }

    pub fn hub(&self) -> &Arc<EquipmentHub> {
        &self.hub
    }

    pub fn clock_quality(&self, path: AcquisitionPath) -> ClockQuality {
        clock_quality(self.driver_id.as_deref(), path)
    }

    /// The trigger is about to be sent: hold the `exposing` busy flag and
    /// emit `exposure_started`.
    pub fn exposure_started(
        &mut self,
        path: AcquisitionPath,
        duration_s: f64,
        stream_generation: Option<u64>,
    ) {
        self.stream_generation = stream_generation;
        if self.exposing.is_none() {
            self.exposing = Some(self.hub.busy_flags().claim(BusyFlag::Exposing));
        }
        let exposure = ExposureRef {
            started_at: Some(self.hub.clock().now_utc()),
            duration_s: Some(duration_s).filter(|d| d.is_finite()),
            frame_id: Some(self.frame_id.clone()),
            operation_id: Some(self.capture_id.clone()),
            capture_id: Some(self.capture_id.clone()),
            stream_generation,
            clock_quality: self.clock_quality(path),
        };
        self.hub.emit(camera_event(
            &self.hub,
            self.connection.as_deref(),
            EventKind::ExposureStarted { exposure },
        ));
    }

    /// The exposure returned: the camera is no longer integrating.
    pub fn exposure_ended(&mut self) {
        self.exposing = None;
    }

    /// `exposure_completed{frame_id}` after the commit (H17). The frame id is
    /// also the key of the frame's window row. `None` when the exposure
    /// ended but no frame was committed.
    pub fn exposure_completed(&self, frame_id: Option<&str>, interval: &ExposureInterval) {
        let exposure = ExposureRef {
            started_at: Some(interval.start),
            duration_s: Some(interval.exposure_s),
            frame_id: frame_id.map(str::to_string),
            operation_id: Some(self.capture_id.clone()),
            capture_id: Some(self.capture_id.clone()),
            stream_generation: self.stream_generation,
            clock_quality: interval.clock_quality,
        };
        self.hub.emit(camera_event(
            &self.hub,
            self.connection.as_deref(),
            EventKind::ExposureCompleted {
                exposure,
                completed_at: interval.end,
            },
        ));
    }
}

// ---------------------------------------------------------------------------
// The window and everything derived from it (H1, H16)
// ---------------------------------------------------------------------------

/// What the capture path knows that the window does not.
#[derive(Debug, Clone)]
pub(crate) struct AssemblyInput {
    pub frame_id: String,
    /// The client's `filter` label, if any.
    pub client_filter: Option<String>,
    pub blocks_at_start: Vec<ResumeBlock>,
    pub source: CommandSource,
    pub command_id: Option<CommandId>,
}

/// The hub's window plus the ring facts a frame needs and the window does
/// not hold: the last focuser and wheel samples at or before the exposure
/// end (the window's `end` is the first sample after it once one exists,
/// D7), whether a sample inside the exposure reported motion, and whether
/// a change that the first sample after the end shows is known to have
/// begun after the end.
#[derive(Debug, Clone, Default)]
pub(crate) struct WindowView {
    pub window: DeviceWindow,
    pub site: Option<Site>,
    pub devices: Value,
    pub guide: Option<GuideWindowStats>,
    pub focuser_end: Option<FocuserSnapshot>,
    pub focuser_moving_inside: bool,
    pub focuser_changed_after_end: bool,
    pub wheel_end: Option<FilterWheelSnapshot>,
    pub wheel_moving_inside: bool,
    pub wheel_changed_after_end: bool,
}

/// True when whatever the first sample after `end` shows is known to have
/// begun after `end`: the sample's operation, or a motion event of its
/// kind, started after the end and no later than the sample. Without
/// either, a difference between that sample and the last one inside the
/// exposure could have happened at any time between them.
fn began_after_end<S: DeviceSnapshot>(
    hub: &EquipmentHub,
    ring: &Ring<S>,
    end: DateTime<Utc>,
) -> bool {
    let Some(post) = ring.at_or_after(end) else {
        return false;
    };
    let post_at = post.at_utc();
    let after = |at: DateTime<Utc>| at > end && at <= post_at;
    post.active_op().is_some_and(|op| after(op.started_at))
        || hub
            .bus()
            .events_between(end, post_at)
            .iter()
            .any(|e| after(e.at_utc) && motion_kinds(e).contains(&S::KIND))
}

impl WindowView {
    /// Read the window of `interval` from the hub (rings and journal only;
    /// no device is called) and, when PHD2 is connected, its own guide
    /// statistics (G-i).
    pub fn read(
        hub: &EquipmentHub,
        guide: Option<&GuiderWindows>,
        interval: &ExposureInterval,
    ) -> Self {
        let (start, end) = (interval.start, interval.end);
        let rings = hub.rings();
        Self {
            window: hub.window(start, end),
            site: hub.site(),
            devices: device_summaries(hub),
            guide: guide.map(|g| g.window_stats(start, end)),
            focuser_end: rings.focuser.at_or_before(end).map(|s| (*s).clone()),
            focuser_moving_inside: rings
                .focuser
                .range(start, end)
                .iter()
                .any(|s| s.in_motion()),
            focuser_changed_after_end: began_after_end(hub, &rings.focuser, end),
            wheel_end: rings.filter_wheel.at_or_before(end).map(|s| (*s).clone()),
            wheel_moving_inside: rings
                .filter_wheel
                .range(start, end)
                .iter()
                .any(|s| s.in_motion()),
            wheel_changed_after_end: began_after_end(hub, &rings.filter_wheel, end),
        }
    }
}

/// How a device's state at the exposure end is known (D7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum EndState {
    /// The first sample after the end agrees with the last one at or before
    /// it.
    Confirmed,
    /// The first sample after the end differs, and the change is shown to
    /// have begun after the end (an operation or motion event started
    /// after it): the last sample at or before the end is the end state.
    ChangedAfterEnd,
    /// No sample after the end yet: the last one at or before it stands in
    /// until the window is final.
    Provisional,
    /// The first sample after the end differs and nothing shows when the
    /// change happened: the end state is unknown.
    ChangedAtOrAfterEnd,
    /// No sample after the end arrived within the D7 grace: the end state
    /// is `Stale`.
    Stale,
    /// No sample near the end.
    Missing,
}

impl EndState {
    /// The last sample at or before the end is the device's end state.
    fn known(self) -> bool {
        matches!(
            self,
            EndState::Confirmed | EndState::ChangedAfterEnd | EndState::Provisional
        )
    }
}

/// The D7 end state of one kind: `last` is its last sample at or before the
/// exposure end (within read age), `samples.end` the first one after it once
/// it exists, and `agrees(last, post)` whether the two describe the same
/// state.
fn end_state<S: Timestamped>(
    samples: &WindowSamples<S>,
    last: Option<&S>,
    interval_end: DateTime<Utc>,
    kind: DeviceKind,
    changed_after_end: bool,
    agrees: impl Fn(&S, &S) -> bool,
) -> EndState {
    let Some(last) = last else {
        return EndState::Missing;
    };
    if samples.end_stale {
        return EndState::Stale;
    }
    let Some(post) = samples.end.as_ref().filter(|s| s.at_utc() >= interval_end) else {
        return EndState::Provisional;
    };
    if near(Some(post), interval_end, kind).is_none() {
        return EndState::Stale;
    }
    if agrees(last, post) {
        EndState::Confirmed
    } else if changed_after_end {
        EndState::ChangedAfterEnd
    } else {
        EndState::ChangedAtOrAfterEnd
    }
}

/// Identity and link state of every attached device (per-frame device
/// provenance, §7.2.22).
fn device_summaries(hub: &EquipmentHub) -> Value {
    let mut devices = serde_json::Map::new();
    for handle in hub.attached() {
        let identity = handle.identity();
        devices.insert(
            kind_name(handle.kind()).to_string(),
            json!({
                "identity_key": identity.identity_key,
                "name": identity.name,
                "driver": identity.driver,
                "driver_version": identity.driver_version,
                "connection_id": identity.connection_id,
                "qualification": identity.qualification.as_str(),
                "state": handle.health().state,
            }),
        );
    }
    Value::Object(devices)
}

/// Where the telescope pointed at mid-exposure.
#[derive(Debug, Clone, Default, Serialize)]
struct Pointing {
    /// Mid-exposure position in the mount's native frame.
    native: Option<EqCoord>,
    j2000: Option<EqCoord>,
    frame_reason: Option<String>,
    alt_deg: Option<f64>,
    az_deg: Option<f64>,
    airmass: Option<f64>,
    ha_h: Option<f64>,
    lst_h: Option<f64>,
    /// `computed` (astro, from the position and the site) or `mount`.
    altaz_source: Option<&'static str>,
    pier_side: Option<PierSide>,
    site: Option<Site>,
    /// Where the site came from: the hub site's own source (`config` for
    /// `[site]`, `mount` when it was taken from the mount, `operator`), or
    /// `mount` for the mount's site when the hub has none.
    site_source: Option<&'static str>,
}

fn site_source_name(source: SiteSource) -> &'static str {
    match source {
        SiteSource::Config => "config",
        SiteSource::Mount => "mount",
        SiteSource::Operator => "operator",
    }
}

fn pointing(
    window: &DeviceWindow,
    mid_sample: Option<&MountSnapshot>,
    configured_site: Option<Site>,
    mid: DateTime<Utc>,
) -> Pointing {
    let mut p = Pointing::default();
    if let Some(site) = configured_site {
        p.site = Some(site);
        p.site_source = Some(site_source_name(site.source));
    } else if let Some(site) = mid_sample.and_then(|m| m.state.site.fresh_value().copied()) {
        p.site = Some(site);
        p.site_source = Some("mount");
    }
    let Some(sample) = mid_sample else {
        return p;
    };
    p.native = window
        .mount_mid
        .filter(|c| c.ra_h.is_finite() && c.dec_deg.is_finite())
        .or_else(|| {
            let ra = sample.state.ra_h.fresh_value()?;
            let dec = sample.state.dec_deg.fresh_value()?;
            Some(EqCoord::new(*ra, *dec, sample.capabilities.native_frame))
        });
    if let Some(native) = p.native {
        if native.frame == EqFrame::Unknown {
            p.frame_reason = Some(
                sample
                    .capabilities
                    .frame_reason
                    .clone()
                    .unwrap_or_else(|| "mount_frame_unknown".into()),
            );
        } else {
            p.j2000 = astro::frames::convert(native, EqFrame::J2000, mid).ok();
        }
    }
    let geometry = p.site.and_then(|site| {
        let position = p
            .j2000
            .or(p.native.filter(|c| c.frame != EqFrame::Unknown))?;
        astro::meridian::target_geometry(position, &site, mid, None)
    });
    if let Some(g) = geometry {
        p.alt_deg = Some(g.horizontal.alt_deg);
        p.az_deg = Some(g.horizontal.az_deg);
        p.airmass = g.airmass;
        p.ha_h = Some(g.horizontal.ha_h);
        p.lst_h = Some(g.horizontal.lst_h);
        p.altaz_source = Some("computed");
    } else if let (Some(alt), Some(az)) = (
        sample.state.alt_deg.fresh_value().copied(),
        sample.state.az_deg.fresh_value().copied(),
    ) {
        p.alt_deg = Some(alt);
        p.az_deg = Some(az);
        p.airmass = astro::refraction::airmass(alt);
        p.ha_h = sample.ha_h;
        p.lst_h = sample.state.sidereal_time_h.fresh_value().copied();
        p.altaz_source = Some("mount");
    }
    p.pier_side = sample
        .state
        .side_of_pier
        .fresh_value()
        .copied()
        .filter(|side| *side != PierSide::Unknown);
    p
}

/// The focuser over the exposure.
#[derive(Debug, Clone, Default, Serialize)]
struct FocuserFacts {
    present: bool,
    /// Logical positions at the start and end of the exposure.
    position_start: Option<i32>,
    position_end: Option<i32>,
    raw_start: Option<i32>,
    raw_end: Option<i32>,
    position_source: Option<PositionSource>,
    /// The start position is a reliable absolute position (v6 §7.6.2).
    reliable_start: bool,
    /// The end position is known (D7) and a reliable absolute position.
    reliable_end: bool,
    /// Both are.
    reliable: bool,
    /// `true` if motion was seen at the start or inside the exposure,
    /// `false` only when the start and end states are both known and
    /// nothing moved, else unknown.
    moved: Option<bool>,
    /// How the end state is known (D7).
    end_state: Option<EndState>,
    temperature_c: Option<f64>,
    temperature_role: Option<TemperatureRole>,
    /// What `frames.focuser_position` and `FOCPOS` carry: the position when
    /// it was reliable and did not change during the exposure.
    frame_position: Option<i32>,
}

/// The same focuser state: not moving, at the same logical position, from
/// the same position source.
fn focuser_agrees(last: &FocuserSnapshot, post: &FocuserSnapshot) -> bool {
    !post.in_motion()
        && post.logical_position == last.logical_position
        && post.position_source == last.position_source
}

fn focuser_facts(view: &WindowView, interval: &ExposureInterval) -> FocuserFacts {
    let w = &view.window;
    let kind = DeviceKind::Focuser;
    let start = near(w.focuser.start.as_ref(), interval.start, kind);
    let last = near(view.focuser_end.as_ref(), interval.end, kind);
    let post = near(
        w.focuser
            .end
            .as_ref()
            .filter(|s| s.at_utc() >= interval.end),
        interval.end,
        kind,
    );
    let mid = near(
        w.focuser.mid.as_ref(),
        midpoint(interval.start, interval.end),
        kind,
    );
    if start.is_none() && last.is_none() && mid.is_none() && post.is_none() {
        return FocuserFacts::default();
    }
    let end_state = end_state(
        &w.focuser,
        last,
        interval.end,
        kind,
        view.focuser_changed_after_end,
        focuser_agrees,
    );
    // The end state is the last sample at or before the end only while D7
    // says it is known; otherwise there is no end position.
    let end = last.filter(|_| end_state.known());
    let position_start = start.and_then(|s| s.logical_position);
    let position_end = end.and_then(|s| s.logical_position);
    let position_last = last.and_then(|s| s.logical_position);
    // Motion at the start (a move already running) or anywhere inside.
    let motion_seen = start.is_some_and(|s| s.in_motion())
        || last.is_some_and(|s| s.in_motion())
        || view.focuser_moving_inside
        || w.events.iter().any(|e| motion_kinds(e).contains(&kind))
        || matches!((position_start, position_last), (Some(a), Some(b)) if a != b);
    let moved = if motion_seen {
        Some(true)
    } else if start.is_some() && end.is_some() {
        Some(false)
    } else {
        None
    };
    let reliable_start = start.is_some_and(|s| s.position_source.is_reliable());
    let reliable_end = end.is_some_and(|s| s.position_source.is_reliable());
    let reliable = reliable_start && reliable_end;
    let temperature_c = [mid, start, last]
        .into_iter()
        .flatten()
        .find_map(|s| s.state.temperature_c.fresh_value().copied())
        .filter(|t| t.is_finite());
    FocuserFacts {
        present: true,
        position_start,
        position_end,
        raw_start: start.and_then(|s| s.state.position.known().copied()),
        raw_end: end.and_then(|s| s.state.position.known().copied()),
        position_source: start.or(last).map(|s| s.position_source),
        reliable_start,
        reliable_end,
        reliable,
        moved,
        end_state: Some(end_state),
        temperature_c,
        temperature_role: start.or(last).or(mid).map(|s| s.temperature_role),
        frame_position: (reliable && moved == Some(false) && position_start == position_end)
            .then_some(position_start)
            .flatten(),
    }
}

/// The filter the exposure was taken through.
#[derive(Debug, Clone, Default, Serialize)]
struct FilterFacts {
    name: Option<String>,
    slot: Option<u32>,
    verified: Option<bool>,
    /// `wheel` (verified), `client` (the request's label) or `none`.
    source: &'static str,
    client_label: Option<String>,
    wheel_present: bool,
    /// What the wheel reported, verified or not.
    wheel_slot: Option<u32>,
    wheel_slot_name: Option<String>,
    /// Why a connected wheel did not verify this exposure.
    unverified_reason: Option<&'static str>,
    /// How the wheel's end state is known (D7).
    end_state: Option<EndState>,
}

/// The same wheel state: not moving, at the same known slot, with the same
/// verification.
fn wheel_agrees(last: &FilterWheelSnapshot, post: &FilterWheelSnapshot) -> bool {
    let slot = post.position.known().copied();
    !post.in_motion()
        && slot.is_some()
        && slot == last.position.known().copied()
        && post.verified == last.verified
        && post.verified_slot == last.verified_slot
}

fn slot_name(wheel: &FilterWheelSnapshot, slot: u32) -> Option<String> {
    wheel
        .slots
        .iter()
        .find(|s| s.index == slot)
        .map(|s| s.name.clone())
}

fn filter_facts(
    view: &WindowView,
    client_label: Option<&str>,
    interval: &ExposureInterval,
) -> FilterFacts {
    let w = &view.window;
    let kind = DeviceKind::FilterWheel;
    let client_label = client_label
        .map(str::trim)
        .filter(|label| !label.is_empty())
        .map(str::to_string);
    let start = near(w.filter_wheel.start.as_ref(), interval.start, kind);
    let last = near(view.wheel_end.as_ref(), interval.end, kind);
    let post = near(
        w.filter_wheel
            .end
            .as_ref()
            .filter(|s| s.at_utc() >= interval.end),
        interval.end,
        kind,
    );
    let mut facts = FilterFacts {
        client_label: client_label.clone(),
        ..FilterFacts::default()
    };
    if let Some(wheel) = last.or(start).or(post) {
        facts.wheel_present = true;
        facts.wheel_slot = wheel.position.known().copied();
        facts.wheel_slot_name = facts.wheel_slot.and_then(|slot| slot_name(wheel, slot));
    }
    let end_state = end_state(
        &w.filter_wheel,
        last,
        interval.end,
        kind,
        view.wheel_changed_after_end,
        wheel_agrees,
    );
    if facts.wheel_present {
        facts.end_state = Some(end_state);
    }
    // The end state is the last sample at or before the end only while D7
    // says it is known.
    let end = last.filter(|_| end_state.known());
    let slot_of = |s: &FilterWheelSnapshot| s.position.known().copied();
    // Motion at the start (a change already running) or anywhere inside.
    let moved = start.is_some_and(|s| s.in_motion())
        || last.is_some_and(|s| s.in_motion())
        || view.wheel_moving_inside
        || w.events.iter().any(|e| motion_kinds(e).contains(&kind))
        || matches!(
            (start.and_then(slot_of), last.and_then(slot_of)),
            (Some(a), Some(b)) if a != b
        );
    let verified_slot = match (start, end) {
        (Some(s), Some(e))
            if s.verified
                && e.verified
                && s.verified_slot.is_some()
                && s.verified_slot == e.verified_slot
                && !moved =>
        {
            s.verified_slot
        }
        _ if moved => {
            facts.unverified_reason = Some("wheel_moved_during_exposure");
            None
        }
        _ if end_state == EndState::ChangedAtOrAfterEnd => {
            facts.unverified_reason = Some("changed_at_or_after_end");
            None
        }
        _ if end_state == EndState::Stale => {
            facts.unverified_reason = Some("end_state_stale");
            None
        }
        (Some(_), Some(_)) => {
            facts.unverified_reason = Some("slot_not_verified");
            None
        }
        _ if facts.wheel_present => {
            facts.unverified_reason = Some("wheel_state_missing_at_start_or_end");
            None
        }
        _ => None,
    };
    match (verified_slot, end) {
        (Some(slot), Some(wheel)) => {
            facts.name = slot_name(wheel, slot);
            facts.slot = Some(slot);
            facts.verified = Some(true);
            facts.source = "wheel";
        }
        _ => {
            facts.name = client_label;
            facts.verified = (facts.name.is_some() || facts.wheel_present).then_some(false);
            facts.source = if facts.name.is_some() {
                "client"
            } else {
                "none"
            };
        }
    }
    facts
}

/// `{field: value}` for fresh values, null otherwise, plus a `not_fresh`
/// map naming the state of every value that was not fresh.
#[derive(Default)]
struct Fields {
    map: serde_json::Map<String, Value>,
    not_fresh: serde_json::Map<String, Value>,
}

impl Fields {
    fn obs<T: Serialize>(&mut self, key: &str, observed: &Observed<T>) -> &mut Self {
        match observed.fresh_value() {
            Some(value) => {
                self.map.insert(key.into(), json!(value));
            }
            None => {
                self.map.insert(key.into(), Value::Null);
                self.not_fresh.insert(key.into(), json!(observed.state));
            }
        }
        self
    }

    fn put(&mut self, key: &str, value: impl Serialize) -> &mut Self {
        self.map.insert(key.into(), json!(value));
        self
    }

    fn finish(&mut self) -> Value {
        let mut map = std::mem::take(&mut self.map);
        if !self.not_fresh.is_empty() {
            map.insert(
                "not_fresh".into(),
                Value::Object(std::mem::take(&mut self.not_fresh)),
            );
        }
        Value::Object(map)
    }
}

/// A compact mount state for `mount_{start,mid,end}_json`.
fn mount_sample_json(s: &MountSnapshot) -> Value {
    Fields::default()
        .put("at", s.published_at)
        .put("connection_id", s.connection_id)
        .put("frame", s.capabilities.native_frame)
        .obs("ra_h", &s.state.ra_h)
        .obs("dec_deg", &s.state.dec_deg)
        .put("j2000", s.j2000)
        .obs("alt_deg", &s.state.alt_deg)
        .obs("az_deg", &s.state.az_deg)
        .obs("lst_h", &s.state.sidereal_time_h)
        .put("ha_h", s.ha_h)
        .obs("pier_side", &s.state.side_of_pier)
        .obs("tracking", &s.state.tracking)
        .obs("tracking_rate", &s.state.tracking_rate)
        .obs("slewing", &s.state.slewing)
        .obs("at_park", &s.state.at_park)
        .obs("is_pulse_guiding", &s.state.is_pulse_guiding)
        .put("does_refraction", s.capabilities.does_refraction)
        .put("sync_generation", s.sync_generation)
        .put("jog_active", s.jog_active)
        .put("active_op", s.active_op.as_ref().map(|op| op.kind))
        .finish()
}

/// Cover and calibrator state for `cover_json` (H16, BGE calibration
/// context).
fn cover_json(s: &CoverCalibratorSnapshot, moved: bool) -> Value {
    Fields::default()
        .put("at", s.published_at)
        .obs("cover_state", &s.state.cover_state)
        .obs("cover_moving", &s.state.cover_moving)
        .obs("calibrator_state", &s.state.calibrator_state)
        .obs("calibrator_changing", &s.state.calibrator_changing)
        .obs("brightness", &s.state.brightness)
        .put("max_brightness", s.state.max_brightness)
        .put("changed_during_exposure", moved)
        .finish()
}

/// `Some(true)` if `pick` is true anywhere, `Some(false)` if it is known
/// false over the whole (covered) exposure, else `None`.
fn any_known(
    intervals: &[MotionInterval],
    covered: bool,
    pick: impl Fn(&MotionInterval) -> Option<bool>,
) -> Option<bool> {
    if intervals.iter().any(|i| pick(i) == Some(true)) {
        return Some(true);
    }
    (covered && !intervals.is_empty() && intervals.iter().all(|i| pick(i) == Some(false)))
        .then_some(false)
}

/// Tracking "throughout": `Some(false)` if it was off at any point,
/// `Some(true)` only if it was known on over the whole exposure.
fn tracking_throughout(intervals: &[MotionInterval], covered: bool) -> Option<bool> {
    if intervals.iter().any(|i| i.tracking == Some(false)) {
        return Some(false);
    }
    (covered && !intervals.is_empty() && intervals.iter().all(|i| i.tracking == Some(true)))
        .then_some(true)
}

/// Everything a frame takes from the equipment layer.
#[derive(Debug, Clone)]
pub(crate) struct CaptureEquipment {
    /// The frame's `exposure_device_windows` row.
    pub row: ExposureWindowRow,
    /// `frames.acquisition_json.equipment` (the row is under `window`).
    pub block: Value,
    /// FITS keywords; a missing value has no card.
    pub cards: Vec<FitsCard>,
    pub filter_name: Option<String>,
    pub filter_verified: Option<bool>,
    pub focuser_position: Option<i64>,
    pub warnings: Vec<String>,
}

impl CaptureEquipment {
    /// The short summary returned with a capture.
    pub fn summary(&self) -> Value {
        json!({
            "filter": self.filter_name,
            "filter_verified": self.filter_verified,
            "focuser_position": self.focuser_position,
            "clock_quality": self.row.clock_quality,
            "window_finalised": self.row.finalised,
            "resume_block_active": self.row.resume_block_active,
            "fits_keywords": self.cards.iter().map(|c| c.key.as_str()).collect::<Vec<_>>(),
        })
    }
}

/// Read the window of `interval` and derive the frame's equipment record.
pub(crate) fn assemble(
    hub: &EquipmentHub,
    guide: Option<&GuiderWindows>,
    interval: &ExposureInterval,
    input: &AssemblyInput,
) -> CaptureEquipment {
    let view = WindowView::read(hub, guide, interval);
    derive(&view, interval, input, hub.clock().now_utc())
}

/// The frame's equipment record from a window (pure; no hub, no device).
pub(crate) fn derive(
    view: &WindowView,
    interval: &ExposureInterval,
    input: &AssemblyInput,
    now: DateTime<Utc>,
) -> CaptureEquipment {
    let w = &view.window;
    let mid = midpoint(interval.start, interval.end);
    let mut warnings = Vec::new();

    let mount_kind = DeviceKind::Mount;
    let mount_start = near(w.mount.start.as_ref(), interval.start, mount_kind);
    let mount_mid = near(w.mount.mid.as_ref(), mid, mount_kind);
    let mount_end = near(w.mount.end.as_ref(), interval.end, mount_kind);
    let pointing = pointing(w, mount_mid, view.site, mid);
    let focuser = focuser_facts(view, interval);
    let filter = filter_facts(view, input.client_filter.as_deref(), interval);
    if let (Some(true), Some(label), Some(name)) =
        (filter.verified, &filter.client_label, &filter.name)
    {
        if !label.eq_ignore_ascii_case(name) {
            warnings.push(format!(
                "Requested filter '{label}' differs from the verified wheel slot '{name}'; \
                 the wheel's filter is recorded"
            ));
        }
    }
    if filter.wheel_present && filter.verified != Some(true) {
        warnings.push(format!(
            "The filter wheel did not verify this exposure ({}); the filter is recorded as \
             unverified",
            filter.unverified_reason.unwrap_or("not verified")
        ));
    }
    if focuser.present && focuser.moved == Some(true) {
        warnings.push("The focuser moved during the exposure".into());
    } else if let Some(state @ (EndState::ChangedAtOrAfterEnd | EndState::Stale)) =
        focuser.end_state
    {
        let why = match state {
            EndState::Stale => "no focuser sample after the end arrived in time",
            _ => {
                "the first focuser sample after the end differs, and nothing shows when it changed"
            }
        };
        warnings.push(format!(
            "The focuser's state at the end of the exposure is unknown ({why}); no focuser \
             position is recorded"
        ));
    }
    let rotator_angle_deg = near(w.rotator.mid.as_ref(), mid, DeviceKind::Rotator)
        .and_then(|r| r.state.position_deg.fresh_value().copied());
    let cover = near(
        w.cover_calibrator.mid.as_ref(),
        mid,
        DeviceKind::CoverCalibrator,
    )
    .map(|c| {
        let changed = w.events.iter().any(|e| {
            matches!(
                e.event,
                EventKind::CoverChanged { .. } | EventKind::CalibratorChanged { .. }
            )
        });
        cover_json(c, changed)
    });
    let guide = view.guide.clone().unwrap_or_else(|| w.guide.clone());
    let guider_seen = guide.state.is_some() || guide.rms.is_some();

    let covered = mount_start.is_some()
        && mount_end.is_some()
        && w.motion.first().is_some_and(|i| i.from <= interval.start);
    // The mount moved under its own drive (a slew, jog, flip, park or home).
    // Guide pulses and dithers also move it, but they are not slews; they
    // are `pulse_guiding_any` and `dither_in_window`.
    let slewed = w.events.iter().any(|e| {
        matches!(
            e.event,
            EventKind::SlewStarted
                | EventKind::SlewArrived { .. }
                | EventKind::SlewEnded { .. }
                | EventKind::MoveAxis { .. }
                | EventKind::MeridianFlip { .. }
                | EventKind::Parked { .. }
                | EventKind::Unparked
                | EventKind::Homed
        )
    });
    let resume_block_active = w.resume_block_active || !input.blocks_at_start.is_empty();
    let (quiescent, quiescence_reason) = match &w.quiescence {
        Quiescence::Quiescent { .. } => (Some(true), None),
        Quiescence::Moved { kind, event_seq } => (
            Some(false),
            Some(format!("{}_moved (event {event_seq})", kind_name(*kind))),
        ),
        Quiescence::Unknown { reason } => (None, Some(reason.clone())),
    };
    let row = ExposureWindowRow {
        frame_id: input.frame_id.clone(),
        exposure_started_at: Some(interval.start),
        exposure_completed_at: Some(interval.end),
        clock_quality: interval.clock_quality,
        finalised: w.finalised,
        mount_start_json: mount_start.map(mount_sample_json),
        mount_mid_json: mount_mid.map(|s| {
            let mut value = mount_sample_json(s);
            value["interpolated"] = json!(pointing.native);
            value
        }),
        mount_end_json: mount_end.map(mount_sample_json),
        mount_frame: mount_mid
            .or(mount_start)
            .or(mount_end)
            .map(|s| s.capabilities.native_frame),
        pier_side: pointing.pier_side,
        tracking: tracking_throughout(&w.motion, covered),
        slewing_any: if slewed {
            Some(true)
        } else {
            any_known(&w.motion, covered, |i| i.slewing)
        },
        pulse_guiding_any: if guide.pulses_n > 0 {
            Some(true)
        } else {
            any_known(&w.motion, covered, |i| i.pulse_guiding)
        },
        focuser_pos_start: focuser
            .reliable_start
            .then_some(focuser.position_start)
            .flatten(),
        focuser_pos_end: focuser
            .reliable_end
            .then_some(focuser.position_end)
            .flatten(),
        focuser_moved: focuser.moved,
        focuser_temperature_c: focuser.temperature_c,
        filter_slot: filter.slot,
        filter_name: filter.name.clone(),
        filter_verified: filter.verified,
        rotator_angle_deg,
        guiding_state: guide.state,
        guide_rms_ra_px: guide.rms.map(|r| r.ra_px),
        guide_rms_dec_px: guide.rms.map(|r| r.dec_px),
        guide_rms_total_arcsec: guide.rms.and_then(|r| r.total_arcsec),
        guide_rms0_total_arcsec: guide.rms.and_then(|r| r.rms0_total_arcsec),
        guide_peak_ra_px: guide.rms.map(|r| r.peak_ra_px),
        guide_peak_dec_px: guide.rms.map(|r| r.peak_dec_px),
        guide_steps_n: guide.rms.map(|r| r.steps_n),
        dither_in_window: guider_seen.then_some(guide.dither_in_window),
        settle_in_window: guider_seen.then_some(guide.settle_in_window),
        safety_state: w.safety.as_ref().map(|s| s.state),
        weather_json: w.weather.as_ref().map(|reading| json!(reading)),
        cover_json: cover,
        resume_block_active,
        quiescent,
        quiescence_reason,
        events: w.events.iter().map(|e| e.seq).collect(),
    };
    let cards = fits_cards(&pointing, &focuser, &filter, rotator_angle_deg);
    let blocks: Vec<Value> = input
        .blocks_at_start
        .iter()
        .map(|b| json!({"id": b.id, "kind": b.kind, "reason": b.reason, "raised_at": b.raised_at}))
        .collect();
    let block = json!({
        "schema": EQUIPMENT_SCHEMA,
        "assembled_at": now,
        "finalised": w.finalised,
        "source": input.source,
        "command_id": input.command_id,
        "interval": interval,
        "devices": view.devices,
        "pointing": pointing,
        "focuser": focuser,
        "filter": filter,
        "rotator": {"angle_deg": rotator_angle_deg},
        "resume_blocks": blocks,
        "resume_block_active": resume_block_active,
        "stale_kinds": w.stale_kinds,
        "fits_keywords": cards.iter().map(|c| c.key.as_str()).collect::<Vec<_>>(),
        "warnings": warnings,
        "window": row,
    });
    CaptureEquipment {
        filter_name: row.filter_name.clone(),
        filter_verified: row.filter_verified,
        focuser_position: focuser.frame_position.map(i64::from),
        row,
        block,
        cards,
        warnings,
    }
}

/// The H1 FITS keywords. Each is written only from a known value: no mount
/// near mid-exposure means no pointing cards, an unknown mount frame means
/// no J2000 cards, and so on.
fn fits_cards(
    pointing: &Pointing,
    focuser: &FocuserFacts,
    filter: &FilterFacts,
    rotator_angle_deg: Option<f64>,
) -> Vec<FitsCard> {
    let mut cards = Vec::new();
    if let Some(j2000) = pointing.j2000 {
        cards.push(FitsCard::float(
            "RA",
            j2000.ra_h * 15.0,
            "[deg] telescope RA (J2000), mid-exposure",
        ));
        cards.push(FitsCard::float(
            "DEC",
            j2000.dec_deg,
            "[deg] telescope Dec (J2000), mid-exposure",
        ));
        if let Some(ra) = crate::fits::sexagesimal_ra(j2000.ra_h) {
            cards.push(FitsCard::string(
                "OBJCTRA",
                ra,
                "[hms] telescope RA (J2000), mid-exposure",
            ));
        }
        if let Some(dec) = crate::fits::sexagesimal_dec(j2000.dec_deg) {
            cards.push(FitsCard::string(
                "OBJCTDEC",
                dec,
                "[dms] telescope Dec (J2000), mid-exposure",
            ));
        }
        cards.push(FitsCard::float(
            "EQUINOX",
            2000.0,
            "[yr] equinox of RA, DEC, OBJCTRA, OBJCTDEC",
        ));
    }
    match pointing.pier_side {
        Some(PierSide::East) => cards.push(FitsCard::string(
            "PIERSIDE",
            "EAST",
            "telescope side of pier",
        )),
        Some(PierSide::West) => cards.push(FitsCard::string(
            "PIERSIDE",
            "WEST",
            "telescope side of pier",
        )),
        _ => {}
    }
    if let Some(alt) = pointing.alt_deg {
        cards.push(FitsCard::float(
            "CENTALT",
            alt,
            "[deg] altitude of the field centre",
        ));
    }
    if let Some(az) = pointing.az_deg {
        cards.push(FitsCard::float(
            "CENTAZ",
            az,
            "[deg] azimuth of the field centre",
        ));
    }
    if let Some(airmass) = pointing.airmass {
        cards.push(FitsCard::float(
            "AIRMASS",
            airmass,
            "relative optical path (Kasten-Young)",
        ));
    }
    if let Some(position) = focuser.frame_position {
        cards.push(FitsCard::int(
            "FOCPOS",
            i64::from(position),
            "[step] focuser position",
        ));
    }
    if let Some(celsius) = focuser.temperature_c {
        cards.push(FitsCard::float(
            "FOCTEMP",
            celsius,
            "[degC] focuser temperature",
        ));
    }
    if let Some(name) = &filter.name {
        let comment = if filter.verified == Some(true) {
            "filter, verified wheel slot"
        } else {
            "filter, operator label (unverified)"
        };
        cards.push(FitsCard::string("FILTER", name.clone(), comment));
    }
    if let Some(angle) = rotator_angle_deg {
        cards.push(FitsCard::float(
            "ROTATANG",
            angle,
            "[deg] rotator sky position angle",
        ));
    }
    if let Some(site) = pointing.site {
        cards.push(FitsCard::float(
            "SITELAT",
            site.lat_deg,
            "[deg] site latitude",
        ));
        cards.push(FitsCard::float(
            "SITELONG",
            site.lon_deg,
            "[deg] site longitude, east positive",
        ));
        cards.push(FitsCard::float(
            "SITEELEV",
            site.elev_m,
            "[m] site elevation",
        ));
    }
    cards
}

// ---------------------------------------------------------------------------
// Finalisation (D7)
// ---------------------------------------------------------------------------

/// D7 grace: the window waits at most `2 x` the slowest idle cadence (the
/// switch's 5 s) for a sample after the exposure end.
const FINALISE_GRACE: Duration = Duration::from_secs(12);
/// PHD2 delivers a guide step after its guide exposure; steps placed inside
/// the exposure may arrive this long after it ends.
const GUIDE_STEP_ALLOWANCE: Duration = Duration::from_secs(3);
const FINALISE_POLL: Duration = Duration::from_secs(1);
/// A bound on the finaliser's polls, whatever the clock does.
const FINALISE_MAX_POLLS: u32 = 32;

/// What the finaliser needs to rebuild a committed frame's window.
#[derive(Debug, Clone)]
pub(crate) struct FinalisePlan {
    pub input: AssemblyInput,
    pub interval: ExposureInterval,
    /// The window was already final when the frame was committed.
    pub finalised_at_commit: bool,
}

/// The final window, once every relevant device has a sample after the
/// exposure end and PHD2's steps for it have arrived, or once the grace has
/// passed (when the hub marks missing end states stale). `None` while it is
/// still provisional.
pub(crate) fn try_finalise(
    hub: &EquipmentHub,
    guide: Option<&GuiderWindows>,
    plan: &FinalisePlan,
) -> Option<CaptureEquipment> {
    let now = hub.clock().now_utc();
    let end = plan.interval.end;
    let allowance =
        chrono::Duration::from_std(GUIDE_STEP_ALLOWANCE).unwrap_or(chrono::Duration::zero());
    let grace = chrono::Duration::from_std(FINALISE_GRACE).unwrap_or(chrono::Duration::zero());
    let guide_ready = guide.is_none() || now >= end + allowance;
    let assembled = assemble(hub, guide, &plan.interval, &plan.input);
    let expired = now >= end + grace + allowance;
    ((assembled.row.finalised && guide_ready) || expired).then_some(assembled)
}

/// After the commit, rewrite the frame's window row once it is final, and
/// write PHD2's per-exposure `guider_state_log` row (H9). Nothing to do on
/// the inert hub (no recorder), or for a window that was final at commit
/// with no guider connected.
pub(crate) fn spawn_finaliser(state: &AppState, plan: FinalisePlan) {
    let hub = Arc::clone(&state.equipment);
    if hub.is_inert() {
        return;
    }
    let guide = state.equipment_ctx.guider_windows();
    if plan.finalised_at_commit && guide.is_none() {
        return;
    }
    let task = tokio::spawn(async move {
        for _ in 0..FINALISE_MAX_POLLS {
            if finalise(&hub, guide.as_ref(), &plan) {
                return;
            }
            tokio::time::sleep(FINALISE_POLL).await;
        }
        tracing::warn!(
            "Equipment window of frame {} was not finalised in time; the provisional row stands",
            plan.input.frame_id
        );
    });
    state.equipment_ctx.adopt_task(task);
}

/// One finalisation attempt; true once the final row was recorded.
pub(crate) fn finalise(
    hub: &EquipmentHub,
    guide: Option<&GuiderWindows>,
    plan: &FinalisePlan,
) -> bool {
    let Some(done) = try_finalise(hub, guide, plan) else {
        return false;
    };
    hub.recorder()
        .record(GroundTruthRecord::ExposureWindow(Box::new(done.row)));
    if let Some(guide) = guide {
        guide.record_exposure_window(
            Some(plan.input.frame_id.clone()),
            plan.interval.start,
            plan.interval.end,
        );
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::hub::HubConfig;
    use crate::equipment::resume::BlockRequest;
    use crate::equipment::ring::DeviceSnapshot;
    use crate::equipment::snapshot::{
        CoverCalState, CoverState, FilterSlot, ResumeBlockKind, SafetyGate,
    };
    use crate::equipment::testing::fake_handles::{
        fake_cover_calibrator, fake_filter_wheel, fake_focuser, fake_mount, RecordingSink,
    };
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::traits::Clock;
    use crate::equipment::types::{SiteSource, Source};
    use crate::fits::FitsValue;

    fn hub() -> (Arc<EquipmentHub>, Arc<ManualClock>, Arc<RecordingSink>) {
        let clock = Arc::new(ManualClock::new());
        let sink = RecordingSink::new();
        let hub = EquipmentHub::new(HubConfig::default(), clock.clone(), sink.clone());
        (Arc::new(hub), clock, sink)
    }

    fn site() -> Site {
        Site {
            lat_deg: 40.0,
            lon_deg: -75.0,
            elev_m: 100.0,
            source: SiteSource::Config,
        }
    }

    fn fresh<T>(value: T, clock: &ManualClock) -> Observed<T> {
        Observed::fresh_at(value, Source::DeviceState, clock)
    }

    fn mount_at(clock: &ManualClock, ra: f64, dec: f64, tracking: bool) -> Arc<MountSnapshot> {
        let mut snap = MountSnapshot::default();
        snap.state.ra_h = fresh(ra, clock);
        snap.state.dec_deg = fresh(dec, clock);
        snap.state.tracking = fresh(tracking, clock);
        snap.state.slewing = fresh(false, clock);
        snap.state.is_pulse_guiding = fresh(false, clock);
        snap.state.side_of_pier = fresh(PierSide::West, clock);
        snap.state.alt_deg = fresh(55.0, clock);
        snap.state.az_deg = fresh(120.0, clock);
        snap.capabilities.native_frame = EqFrame::JNowApparent;
        snap.set_published(clock.now_utc(), clock.now_mono_ns());
        Arc::new(snap)
    }

    fn focuser_at(
        clock: &ManualClock,
        position: i32,
        moving: bool,
        source: PositionSource,
    ) -> Arc<FocuserSnapshot> {
        let mut snap = FocuserSnapshot::default();
        snap.state.position = fresh(position, clock);
        snap.state.is_moving = fresh(moving, clock);
        snap.state.temperature_c = fresh(7.5, clock);
        snap.logical_position = Some(position);
        snap.position_source = source;
        snap.set_published(clock.now_utc(), clock.now_mono_ns());
        Arc::new(snap)
    }

    /// A focuser move Achernar started at `started_at`.
    fn focuser_op(started_at: DateTime<Utc>) -> crate::equipment::ops::ActiveOp {
        use crate::equipment::ops::{ActiveOp, OpKind, OpPhase};
        ActiveOp {
            command_id: CommandId::new(),
            kind: OpKind::FocuserMove,
            phase: OpPhase::InProgress,
            started_at,
            started_mono_ns: 0,
            deadline_mono_ns: u64::MAX,
        }
    }

    fn wheel_at(clock: &ManualClock, slot: u32, verified: bool) -> Arc<FilterWheelSnapshot> {
        let mut snap = FilterWheelSnapshot {
            position: fresh(slot, clock),
            slots: ["L", "R", "Ha"]
                .iter()
                .enumerate()
                .map(|(i, name)| FilterSlot {
                    index: i as u32,
                    name: (*name).into(),
                    ..FilterSlot::default()
                })
                .collect(),
            verified,
            verified_slot: verified.then_some(slot),
            ..FilterWheelSnapshot::default()
        };
        snap.set_published(clock.now_utc(), clock.now_mono_ns());
        Arc::new(snap)
    }

    fn input(frame_id: &str, client_filter: Option<&str>) -> AssemblyInput {
        AssemblyInput {
            frame_id: frame_id.into(),
            client_filter: client_filter.map(str::to_string),
            blocks_at_start: Vec::new(),
            source: CommandSource::Operator { overriding: false },
            command_id: None,
        }
    }

    /// A still exposure from now to `secs` later, by the hub clock.
    fn still(clock: &ManualClock, secs: f64) -> (ExposureInterval, DateTime<Utc>) {
        let start = clock.now_utc();
        let interval = ExposureInterval {
            start,
            end: start + seconds(secs),
            clock_quality: ClockQuality::HostTriggerBracket,
            basis: "driver_trigger_bracket",
            exposure_s: secs,
            frame_interval_s: None,
        };
        (interval, start)
    }

    fn card<'a>(cards: &'a [FitsCard], key: &str) -> Option<&'a FitsValue> {
        cards.iter().find(|c| c.key == key).map(|c| &c.value)
    }

    #[test]
    fn clock_quality_follows_the_driver_and_the_acquisition_path() {
        use AcquisitionPath::{Still, Stream};
        assert_eq!(
            clock_quality(Some("gphoto2"), Still),
            ClockQuality::HostTriggerBracket
        );
        for driver in ["asi", "svb", "toup", "uvc", "alpaca", "indi", "wdir"] {
            assert_eq!(
                clock_quality(Some(driver), Stream),
                ClockQuality::HostArrival,
                "{driver} stamps frames on delivery"
            );
        }
        assert_eq!(clock_quality(Some("sim"), Stream), ClockQuality::Simulator);
        assert_eq!(clock_quality(None, Stream), ClockQuality::Unknown);
    }

    #[test]
    fn a_stream_window_reaches_back_exposure_plus_a_frame_interval() {
        let seen = 1_800_000_000.0;
        let measured = ExposureInterval::stream(seen, 4.0, Some(4.2), ClockQuality::HostArrival);
        assert_eq!(measured.end, utc_of(seen));
        let lead = measured.end - measured.start;
        assert!(
            (lead - chrono::Duration::milliseconds(8_200)).abs()
                < chrono::Duration::microseconds(1),
            "{lead}"
        );
        assert_eq!(measured.frame_interval_s, Some(4.2));
        // Without a measured interval, the exposure stands in for it.
        let fallback = ExposureInterval::stream(seen, 4.0, None, ClockQuality::HostArrival);
        assert_eq!(fallback.start, utc_of(seen) - chrono::Duration::seconds(8));
        // A still is the driver's bracket, in order.
        let still = ExposureInterval::still(seen + 30.0, seen, 30.0, ClockQuality::Unknown);
        assert!(still.start < still.end);
        assert_eq!(still.basis, "driver_trigger_bracket");
    }

    #[test]
    fn the_window_gives_the_row_the_block_and_the_fits_keywords() {
        let (hub, clock, _sink) = hub();
        hub.set_site(Some(site()));
        let (mount, _m) = fake_mount(MountSnapshot::default());
        let (focuser, _f) = fake_focuser(FocuserSnapshot::default());
        let (wheel, _w) = fake_filter_wheel(FilterWheelSnapshot::default());
        let (cover, _c) = fake_cover_calibrator(CoverCalibratorSnapshot::default());
        let mount_ring = hub.attach(mount);
        let focuser_ring = hub.attach(focuser);
        let wheel_ring = hub.attach(wheel);
        let cover_ring = hub.attach(cover);
        // On the meridian, 40 degrees up: above the horizon at the site.
        let ra = astro::time::local_sidereal_time_h(clock.now_utc(), site().lon_deg);
        let push_all = |clock: &ManualClock| {
            mount_ring.push(mount_at(clock, ra, 40.0, true));
            focuser_ring.push(focuser_at(clock, 12_000, false, PositionSource::Device));
            wheel_ring.push(wheel_at(clock, 2, true));
            let mut cover = CoverCalibratorSnapshot {
                state: CoverCalState {
                    cover_state: fresh(CoverState::Open, clock),
                    brightness: fresh(0, clock),
                    ..CoverCalState::default()
                },
                ..CoverCalibratorSnapshot::default()
            };
            cover.set_published(clock.now_utc(), clock.now_mono_ns());
            cover_ring.push(Arc::new(cover));
        };
        push_all(&clock);
        clock.advance_ms(500);
        let (interval, _) = still(&clock, 10.0);
        for _ in 0..12 {
            clock.advance_ms(1000);
            push_all(&clock);
        }
        let equipment = assemble(&hub, None, &interval, &input("frame-1", Some("Ha")));
        let row = &equipment.row;
        assert!(row.finalised, "samples after the exposure end exist");
        assert_eq!(row.frame_id, "frame-1");
        assert_eq!(row.clock_quality, ClockQuality::HostTriggerBracket);
        assert_eq!(row.mount_frame, Some(EqFrame::JNowApparent));
        assert_eq!(row.pier_side, Some(PierSide::West));
        assert_eq!(row.tracking, Some(true), "tracking known on throughout");
        assert_eq!(row.slewing_any, Some(false));
        assert_eq!(row.focuser_pos_start, Some(12_000));
        assert_eq!(row.focuser_pos_end, Some(12_000));
        assert_eq!(row.focuser_moved, Some(false));
        assert_eq!(row.focuser_temperature_c, Some(7.5));
        assert_eq!(row.filter_slot, Some(2));
        assert_eq!(row.filter_name.as_deref(), Some("Ha"));
        assert_eq!(row.filter_verified, Some(true));
        assert_eq!(row.safety_state, Some(SafetyGate::None));
        assert_eq!(row.quiescent, Some(true));
        assert!(!row.resume_block_active);
        let cover = row.cover_json.as_ref().expect("H16 cover context");
        assert_eq!(cover["cover_state"], "open");
        assert_eq!(cover["brightness"], 0);
        assert_eq!(
            row.mount_mid_json.as_ref().unwrap()["interpolated"]["frame"],
            "jnow_apparent"
        );
        assert_eq!(equipment.focuser_position, Some(12_000));
        assert_eq!(equipment.filter_name.as_deref(), Some("Ha"));
        // The FITS keywords, all from known values.
        let cards = &equipment.cards;
        for key in [
            "RA", "DEC", "OBJCTRA", "OBJCTDEC", "EQUINOX", "PIERSIDE", "CENTALT", "CENTAZ",
            "AIRMASS", "FOCPOS", "FOCTEMP", "FILTER", "SITELAT", "SITELONG", "SITEELEV",
        ] {
            assert!(card(cards, key).is_some(), "missing {key}");
        }
        assert_eq!(card(cards, "FOCPOS"), Some(&FitsValue::Int(12_000)));
        assert_eq!(
            card(cards, "PIERSIDE"),
            Some(&FitsValue::Str("WEST".into()))
        );
        assert_eq!(card(cards, "EQUINOX"), Some(&FitsValue::Float(2000.0)));
        assert!(card(cards, "ROTATANG").is_none(), "no rotator: no card");
        // J2000 is the precessed mount position: within a degree of JNow
        // for 2026, never equal to it.
        let Some(FitsValue::Float(ra_deg)) = card(cards, "RA") else {
            panic!("RA")
        };
        let offset = (ra_deg - ra * 15.0 + 180.0).rem_euclid(360.0) - 180.0;
        assert!(offset.abs() < 1.0 && offset.abs() > 1e-6, "{offset}");
        let Some(FitsValue::Float(alt)) = card(cards, "CENTALT") else {
            panic!("CENTALT")
        };
        assert!(*alt > 0.0, "{alt}");
        // The altitude is computed from the position and the configured site.
        assert_eq!(equipment.block["pointing"]["altaz_source"], "computed");
        assert_eq!(equipment.block["pointing"]["site_source"], "config");
        assert_eq!(equipment.block["schema"], EQUIPMENT_SCHEMA);
        assert_eq!(equipment.block["devices"]["mount"]["driver"], "sim");
        assert_eq!(
            equipment.block["window"]["filter_verified"], true,
            "the row travels in the acquisition block"
        );
        // A label that disagrees with the verified wheel is reported, and the
        // wheel's filter wins.
        let disagreeing = assemble(&hub, None, &interval, &input("frame-2", Some("OIII")));
        assert_eq!(disagreeing.filter_name.as_deref(), Some("Ha"));
        assert!(disagreeing.warnings.iter().any(|w| w.contains("OIII")));
    }

    #[test]
    fn missing_equipment_writes_no_keywords_and_no_guesses() {
        let (hub, clock, _sink) = hub();
        let (interval, _) = still(&clock, 5.0);
        clock.advance_ms(10_000);
        let equipment = assemble(&hub, None, &interval, &input("frame-empty", None));
        assert!(equipment.cards.is_empty(), "{:?}", equipment.cards);
        let row = &equipment.row;
        assert!(row.mount_start_json.is_none() && row.mount_mid_json.is_none());
        assert_eq!(row.tracking, None);
        assert_eq!(row.slewing_any, None);
        assert_eq!(row.pulse_guiding_any, None);
        assert_eq!(row.focuser_pos_start, None);
        assert_eq!(row.focuser_moved, None);
        assert_eq!(row.filter_name, None);
        assert_eq!(row.filter_verified, None, "no wheel and no label");
        assert_eq!(row.dither_in_window, None);
        assert_eq!(equipment.focuser_position, None);
        // A client label alone is recorded, unverified.
        let labelled = assemble(&hub, None, &interval, &input("frame-l", Some("L")));
        assert_eq!(labelled.filter_name.as_deref(), Some("L"));
        assert_eq!(labelled.filter_verified, Some(false));
        assert_eq!(
            card(&labelled.cards, "FILTER"),
            Some(&FitsValue::Str("L".into()))
        );
    }

    #[test]
    fn an_unknown_mount_frame_omits_the_equatorial_keywords() {
        let (hub, clock, _sink) = hub();
        let (mount, _m) = fake_mount(MountSnapshot::default());
        let ring = hub.attach(mount);
        let unknown = |clock: &ManualClock| {
            let mut snap = (*mount_at(clock, 5.5, 20.0, true)).clone();
            snap.capabilities.native_frame = EqFrame::Unknown;
            snap.capabilities.frame_reason = Some("unsupported_epoch".into());
            ring.push(Arc::new(snap));
        };
        unknown(&clock);
        clock.advance_ms(200);
        let (interval, _) = still(&clock, 2.0);
        for _ in 0..4 {
            clock.advance_ms(1000);
            unknown(&clock);
        }
        let equipment = assemble(&hub, None, &interval, &input("frame-u", None));
        for key in ["RA", "DEC", "OBJCTRA", "OBJCTDEC", "EQUINOX"] {
            assert!(card(&equipment.cards, key).is_none(), "{key}");
        }
        // No site: the mount's own altitude is used and says so.
        assert_eq!(
            card(&equipment.cards, "CENTALT"),
            Some(&FitsValue::Float(55.0))
        );
        assert!(card(&equipment.cards, "AIRMASS").is_some());
        assert_eq!(equipment.block["pointing"]["altaz_source"], "mount");
        assert_eq!(
            equipment.block["pointing"]["frame_reason"],
            "unsupported_epoch"
        );
        assert_eq!(equipment.row.mount_frame, Some(EqFrame::Unknown));
    }

    #[test]
    fn the_focuser_position_is_the_windows_not_a_read_after_the_commit() {
        let (hub, clock, _sink) = hub();
        let (focuser, _f) = fake_focuser(FocuserSnapshot::default());
        let ring = hub.attach(focuser);
        ring.push(focuser_at(&clock, 1_000, false, PositionSource::Device));
        clock.advance_ms(100);
        let (interval, _) = still(&clock, 4.0);
        clock.advance_ms(2_000);
        ring.push(focuser_at(&clock, 1_000, false, PositionSource::Device));
        // After the exposure the focuser moves on (the next autofocus step,
        // an operation started after the end): the live snapshot says 2000,
        // the frame keeps 1000.
        clock.advance_ms(2_100);
        let op_started = clock.now_utc();
        clock.advance_ms(400);
        let mut stepping = (*focuser_at(&clock, 1_400, true, PositionSource::Device)).clone();
        stepping.active_op = Some(focuser_op(op_started));
        ring.push(Arc::new(stepping));
        clock.advance_ms(1_000);
        ring.push(focuser_at(&clock, 2_000, false, PositionSource::Device));
        let equipment = assemble(&hub, None, &interval, &input("frame-f", None));
        assert_eq!(equipment.focuser_position, Some(1_000));
        assert_eq!(equipment.row.focuser_moved, Some(false));
        assert_eq!(equipment.row.focuser_pos_end, Some(1_000));
        assert_eq!(equipment.block["focuser"]["end_state"], "changed_after_end");
        assert_eq!(
            card(&equipment.cards, "FOCPOS"),
            Some(&FitsValue::Int(1_000))
        );

        // A move inside the exposure: no single position, and it is said.
        let (hub, clock, _sink) = self::hub();
        let (focuser, _f) = fake_focuser(FocuserSnapshot::default());
        let ring = hub.attach(focuser);
        ring.push(focuser_at(&clock, 1_000, false, PositionSource::Device));
        clock.advance_ms(100);
        let (interval, _) = still(&clock, 4.0);
        clock.advance_ms(2_000);
        ring.push(focuser_at(&clock, 1_200, true, PositionSource::Device));
        clock.advance_ms(3_000);
        ring.push(focuser_at(&clock, 1_300, false, PositionSource::Device));
        let moved = assemble(&hub, None, &interval, &input("frame-m", None));
        assert_eq!(moved.focuser_position, None);
        assert_eq!(moved.row.focuser_moved, Some(true));
        assert_eq!(moved.row.focuser_pos_start, Some(1_000));
        assert!(card(&moved.cards, "FOCPOS").is_none());
        assert!(moved.warnings.iter().any(|w| w.contains("focuser moved")));

        // An emulated focuser with no known origin has no absolute position.
        let (hub, clock, _sink) = self::hub();
        let (focuser, _f) = fake_focuser(FocuserSnapshot::default());
        let ring = hub.attach(focuser);
        let unknown = PositionSource::EmulatedOriginUnknown;
        ring.push(focuser_at(&clock, 500, false, unknown));
        clock.advance_ms(100);
        let (interval, _) = still(&clock, 2.0);
        clock.advance_ms(3_000);
        ring.push(focuser_at(&clock, 500, false, unknown));
        let emulated = assemble(&hub, None, &interval, &input("frame-e", None));
        assert_eq!(emulated.focuser_position, None);
        assert_eq!(emulated.row.focuser_pos_start, None);
        assert_eq!(
            emulated.block["focuser"]["position_source"],
            "emulated_origin_unknown"
        );
        assert_eq!(emulated.block["focuser"]["position_start"], 500);
    }

    #[test]
    fn the_filter_is_verified_only_by_a_wheel_that_held_its_slot() {
        // The wheel moves during the exposure: the client label, unverified.
        let (hub, clock, _sink) = hub();
        let (wheel, _w) = fake_filter_wheel(FilterWheelSnapshot::default());
        let ring = hub.attach(wheel);
        ring.push(wheel_at(&clock, 0, true));
        clock.advance_ms(100);
        let (interval, _) = still(&clock, 6.0);
        clock.advance_ms(2_000);
        let mut moving = (*wheel_at(&clock, 0, false)).clone();
        moving.moving = true;
        moving.position = Observed::default();
        ring.push(Arc::new(moving));
        clock.advance_ms(2_000);
        ring.push(wheel_at(&clock, 1, true));
        clock.advance_ms(4_000);
        ring.push(wheel_at(&clock, 1, true));
        let equipment = assemble(&hub, None, &interval, &input("frame-w", Some("L")));
        assert_eq!(equipment.filter_name.as_deref(), Some("L"));
        assert_eq!(equipment.filter_verified, Some(false));
        assert_eq!(equipment.row.filter_slot, None);
        assert_eq!(
            equipment.block["filter"]["unverified_reason"],
            "wheel_moved_during_exposure"
        );
        assert_eq!(
            card(&equipment.cards, "FILTER"),
            Some(&FitsValue::Str("L".into()))
        );

        // A wheel that never verified its slot does not verify the frame.
        let (hub, clock, _sink) = self::hub();
        let (wheel, _w) = fake_filter_wheel(FilterWheelSnapshot::default());
        let ring = hub.attach(wheel);
        ring.push(wheel_at(&clock, 2, false));
        clock.advance_ms(100);
        let (interval, _) = still(&clock, 2.0);
        clock.advance_ms(3_000);
        ring.push(wheel_at(&clock, 2, false));
        let unverified = assemble(&hub, None, &interval, &input("frame-v", None));
        assert_eq!(unverified.filter_name, None);
        assert_eq!(unverified.filter_verified, Some(false));
        assert_eq!(unverified.block["filter"]["wheel_slot_name"], "Ha");
    }

    #[test]
    fn a_change_only_the_sample_after_a_short_exposure_shows_leaves_the_end_unknown() {
        // A 0.5 s stream exposure on a shared wheel and focuser publishing at
        // their 2 s idle cadence: the only samples before the end were taken
        // before the exposure began.
        let (hub, clock, _sink) = hub();
        let (wheel, _w) = fake_filter_wheel(FilterWheelSnapshot::default());
        let (focuser, _f) = fake_focuser(FocuserSnapshot::default());
        let wheel_ring = hub.attach(wheel);
        let focuser_ring = hub.attach(focuser);
        wheel_ring.push(wheel_at(&clock, 2, true));
        focuser_ring.push(focuser_at(&clock, 1_000, false, PositionSource::Device));
        clock.advance_ms(1_000);
        let (interval, _) = still(&clock, 0.5);
        clock.advance_ms(500);
        // At the commit no sample after the end exists yet: the window is
        // provisional and the last samples stand in.
        let provisional = assemble(&hub, None, &interval, &input("frame-s", Some("Ha")));
        assert!(!provisional.row.finalised);
        assert_eq!(provisional.filter_verified, Some(true));
        assert_eq!(provisional.block["filter"]["end_state"], "provisional");
        assert_eq!(provisional.focuser_position, Some(1_000));

        // 0.5 s after the end the next samples show both devices moving
        // (another client, a handset): nothing shows when that began.
        clock.advance_ms(500);
        let mut moving = (*wheel_at(&clock, 2, false)).clone();
        moving.moving = true;
        moving.position = Observed::default();
        wheel_ring.push(Arc::new(moving));
        focuser_ring.push(focuser_at(&clock, 1_000, true, PositionSource::Device));
        let plan = FinalisePlan {
            input: input("frame-s", Some("Ha")),
            interval: interval.clone(),
            finalised_at_commit: false,
        };
        let done = try_finalise(&hub, None, &plan).expect("a sample after the end exists");
        assert!(done.row.finalised);
        assert!(
            hub.window(interval.start, interval.end).moved.wheel,
            "the hub's window sees the change too"
        );
        assert_eq!(done.row.filter_verified, Some(false));
        assert_eq!(done.row.filter_slot, None);
        assert_eq!(done.filter_name.as_deref(), Some("Ha"), "the client label");
        assert_eq!(
            done.block["filter"]["unverified_reason"],
            "changed_at_or_after_end"
        );
        assert_eq!(done.block["filter"]["end_state"], "changed_at_or_after_end");
        assert_eq!(
            card(&done.cards, "FILTER"),
            Some(&FitsValue::Str("Ha".into()))
        );
        assert_eq!(done.row.focuser_moved, None, "unknown, not unmoved");
        assert_eq!(done.row.focuser_pos_start, Some(1_000));
        assert_eq!(done.row.focuser_pos_end, None);
        assert_eq!(done.focuser_position, None);
        assert!(card(&done.cards, "FOCPOS").is_none());
        assert!(done
            .warnings
            .iter()
            .any(|w| w.contains("end of the exposure is unknown")));

        // The same sample after a move shown to have begun after the end (a
        // filter change started then): the wheel held its slot throughout.
        let (hub, clock, _sink) = self::hub();
        let (wheel, _w) = fake_filter_wheel(FilterWheelSnapshot::default());
        let id = wheel.connection_id();
        let wheel_ring = hub.attach(wheel);
        wheel_ring.push(wheel_at(&clock, 2, true));
        clock.advance_ms(1_000);
        let (interval, _) = still(&clock, 0.5);
        clock.advance_ms(700);
        hub.emit(
            EquipmentEvent::new(EventKind::FilterMoving { to_slot: 0 }, clock.as_ref())
                .for_device(DeviceKind::FilterWheel, id),
        );
        clock.advance_ms(300);
        let mut moving = (*wheel_at(&clock, 2, false)).clone();
        moving.moving = true;
        moving.position = Observed::default();
        wheel_ring.push(Arc::new(moving));
        let after = assemble(&hub, None, &interval, &input("frame-a", None));
        assert!(after.row.finalised);
        assert_eq!(after.filter_verified, Some(true));
        assert_eq!(after.row.filter_slot, Some(2));
        assert_eq!(after.block["filter"]["end_state"], "changed_after_end");
    }

    #[test]
    fn a_move_already_running_at_the_start_is_a_move() {
        // The last focuser sample before the exposure shows it moving (a
        // TempComp step, another client); the next one, after the end, shows
        // it settled where it started.
        let (hub, clock, _sink) = hub();
        let (focuser, _f) = fake_focuser(FocuserSnapshot::default());
        let ring = hub.attach(focuser);
        ring.push(focuser_at(&clock, 1_000, true, PositionSource::Device));
        clock.advance_ms(500);
        let (interval, _) = still(&clock, 1.0);
        clock.advance_ms(1_500);
        ring.push(focuser_at(&clock, 1_000, false, PositionSource::Device));
        let equipment = assemble(&hub, None, &interval, &input("frame-r", None));
        assert!(equipment.row.finalised);
        assert_eq!(equipment.row.quiescent, Some(false));
        assert_eq!(
            equipment.row.focuser_moved,
            Some(true),
            "agrees with the hub's quiescence"
        );
        assert_eq!(equipment.focuser_position, None);
        assert!(card(&equipment.cards, "FOCPOS").is_none());

        // A wheel still changing slots at the start does not verify.
        let (hub, clock, _sink) = self::hub();
        let (wheel, _w) = fake_filter_wheel(FilterWheelSnapshot::default());
        let ring = hub.attach(wheel);
        let mut moving = (*wheel_at(&clock, 1, false)).clone();
        moving.moving = true;
        moving.position = Observed::default();
        ring.push(Arc::new(moving));
        clock.advance_ms(500);
        let (interval, _) = still(&clock, 1.0);
        clock.advance_ms(1_500);
        ring.push(wheel_at(&clock, 1, true));
        let wheel = assemble(&hub, None, &interval, &input("frame-q", Some("R")));
        assert_eq!(wheel.filter_verified, Some(false));
        assert_eq!(
            wheel.block["filter"]["unverified_reason"],
            "wheel_moved_during_exposure"
        );
    }

    #[test]
    fn a_device_silent_after_the_end_has_a_stale_end_state() {
        let (hub, clock, _sink) = hub();
        let (focuser, _f) = fake_focuser(FocuserSnapshot::default());
        let (wheel, _w) = fake_filter_wheel(FilterWheelSnapshot::default());
        let focuser_ring = hub.attach(focuser);
        let wheel_ring = hub.attach(wheel);
        focuser_ring.push(focuser_at(&clock, 1_000, false, PositionSource::Device));
        wheel_ring.push(wheel_at(&clock, 1, true));
        clock.advance_ms(100);
        let (interval, _) = still(&clock, 3.0);
        clock.advance_ms(1_000);
        focuser_ring.push(focuser_at(&clock, 1_000, false, PositionSource::Device));
        wheel_ring.push(wheel_at(&clock, 1, true));
        // Both fall silent. Past the D7 grace (2 x the 2 s idle cadence)
        // their end states are stale: no position, no verified filter.
        clock.advance_ms(6_100);
        let window = hub.window(interval.start, interval.end);
        assert!(window.focuser.end_stale && window.filter_wheel.end_stale);
        let stale = assemble(&hub, None, &interval, &input("frame-x", None));
        assert!(stale.row.finalised, "a stale end state is final");
        assert_eq!(stale.block["focuser"]["end_state"], "stale");
        assert_eq!(stale.row.focuser_moved, None);
        assert_eq!(stale.row.focuser_pos_start, Some(1_000));
        assert_eq!(stale.row.focuser_pos_end, None);
        assert_eq!(stale.focuser_position, None);
        assert!(card(&stale.cards, "FOCPOS").is_none());
        assert_eq!(stale.filter_verified, Some(false));
        assert_eq!(stale.row.filter_slot, None);
        assert_eq!(
            stale.block["filter"]["unverified_reason"],
            "end_state_stale"
        );
    }

    #[test]
    fn the_site_source_is_the_sites_own() {
        let now = Utc::now();
        let mut from_mount = site();
        from_mount.source = SiteSource::Mount;
        let p = pointing(&DeviceWindow::default(), None, Some(from_mount), now);
        assert_eq!(p.site_source, Some("mount"));
        assert_eq!(json!(p)["site"]["source"], "mount");
        let p = pointing(&DeviceWindow::default(), None, Some(site()), now);
        assert_eq!(p.site_source, Some("config"));
        let mut operator = site();
        operator.source = SiteSource::Operator;
        let p = pointing(&DeviceWindow::default(), None, Some(operator), now);
        assert_eq!(p.site_source, Some("operator"));
    }

    #[test]
    fn tracking_is_true_only_when_known_on_throughout() {
        let (hub, clock, _sink) = hub();
        let (mount, _m) = fake_mount(MountSnapshot::default());
        let ring = hub.attach(mount);
        ring.push(mount_at(&clock, 1.0, 10.0, true));
        clock.advance_ms(100);
        let (interval, _) = still(&clock, 6.0);
        clock.advance_ms(2_000);
        ring.push(mount_at(&clock, 1.0, 10.0, false));
        clock.advance_ms(5_000);
        ring.push(mount_at(&clock, 1.0, 10.0, true));
        let off = assemble(&hub, None, &interval, &input("frame-t", None));
        assert_eq!(off.row.tracking, Some(false), "off at some point");

        // No mount sample at the start: tracking is unknown, not "on".
        let (hub, clock, _sink) = self::hub();
        let (mount, _m) = fake_mount(MountSnapshot::default());
        let ring = hub.attach(mount);
        let (interval, _) = still(&clock, 6.0);
        clock.advance_ms(3_000);
        ring.push(mount_at(&clock, 1.0, 10.0, true));
        clock.advance_ms(5_000);
        ring.push(mount_at(&clock, 1.0, 10.0, true));
        let partial = assemble(&hub, None, &interval, &input("frame-p", None));
        assert_eq!(partial.row.tracking, None);
    }

    #[test]
    fn guide_pulses_are_not_slews_and_a_slew_is_not_quiescent() {
        use crate::equipment::types::GuideDirection;
        let scenario = |event: EventKind| {
            let (hub, clock, _sink) = hub();
            let (mount, _m) = fake_mount(MountSnapshot::default());
            let id = mount.connection_id();
            let ring = hub.attach(mount);
            ring.push(mount_at(&clock, 2.0, 30.0, true));
            clock.advance_ms(100);
            let (interval, _) = still(&clock, 6.0);
            clock.advance_ms(1_000);
            ring.push(mount_at(&clock, 2.0, 30.0, true));
            hub.emit(EquipmentEvent::new(event, clock.as_ref()).for_device(DeviceKind::Mount, id));
            clock.advance_ms(6_000);
            ring.push(mount_at(&clock, 2.0, 30.0, true));
            assemble(&hub, None, &interval, &input("frame-g", None)).row
        };
        let pulsed = scenario(EventKind::PulseGuide {
            dir: GuideDirection::North,
            ms: 300,
            source: CommandSource::Operator { overriding: false },
        });
        assert_eq!(pulsed.pulse_guiding_any, Some(true));
        assert_eq!(pulsed.slewing_any, Some(false), "a pulse is not a slew");
        assert_eq!(pulsed.tracking, Some(true));
        let slewed = scenario(EventKind::SlewStarted);
        assert_eq!(slewed.slewing_any, Some(true));
        assert_eq!(slewed.quiescent, Some(false));
        assert!(slewed
            .quiescence_reason
            .as_deref()
            .is_some_and(|r| r.starts_with("mount_moved")));
        assert_eq!(slewed.events.len(), 1);
    }

    #[test]
    fn a_mount_that_stopped_publishing_is_not_the_exposures_state() {
        let (hub, clock, _sink) = hub();
        hub.set_site(Some(site()));
        let (mount, _m) = fake_mount(MountSnapshot::default());
        let ring = hub.attach(mount);
        ring.push(mount_at(&clock, 1.0, 10.0, true));
        // Ten minutes of silence, then the exposure.
        clock.advance_ms(600_000);
        let (interval, _) = still(&clock, 5.0);
        clock.advance_ms(20_000);
        let equipment = assemble(&hub, None, &interval, &input("frame-s", None));
        for key in ["RA", "DEC", "PIERSIDE", "CENTALT", "AIRMASS"] {
            assert!(card(&equipment.cards, key).is_none(), "{key}");
        }
        assert!(equipment.row.mount_mid_json.is_none());
        assert_eq!(equipment.row.tracking, None);
        // The configured site is still the site.
        assert!(card(&equipment.cards, "SITELAT").is_some());
    }

    /// A capture that is a step of a compound operation holding the camera
    /// (a center loop's frame) is a child of that operation's permit: it
    /// passes while the parent holds the camera, claims nothing itself, and
    /// is audited under its own source. A plain capture is refused 409.
    #[test]
    fn a_step_capture_runs_under_its_parents_camera_claim() {
        use crate::equipment::commands::Domain;
        let (hub, _clock, sink) = hub();
        let relay = CommandSource::Relay { overriding: false };
        let center = GatedCommand::new(
            GatedOp::Center {
                target: EqCoord::new(5.5, -5.0, EqFrame::J2000),
                tolerance_arcmin: None,
                max_attempts: None,
            },
            relay.clone(),
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        let parent = hub.gate().authorize(&center).unwrap();
        let owner = StepOwner::of(&parent);
        let step = authorize_capture(&hub, &relay, Some(&owner), "center", json!({})).unwrap();
        assert_eq!(step.command.source, relay);
        assert_eq!(
            hub.gate()
                .holder(Domain::Device(DeviceKind::Camera))
                .map(|claim| claim.command_id),
            Some(center.command_id),
            "the parent keeps the camera; the step claims nothing"
        );
        let operator = CaptureSource::Operator.command_source_over(false);
        let plain = authorize_capture(&hub, &operator, None, "light", json!({})).err();
        assert_eq!(plain.as_ref().map(GateError::http_status), Some(409));
        assert_eq!(plain.map(|e| e.code()), Some("ownership_conflict"));
        drop(step);
        // Once the parent lets go of the camera, its steps are refused.
        drop(parent);
        assert!(hub.gate().is_idle());
        let orphan = authorize_capture(&hub, &relay, Some(&owner), "center", json!({})).err();
        assert_eq!(orphan.map(|e| e.code()), Some("ownership_conflict"));
        let audited_as_step = sink.records().into_iter().any(|r| match r {
            GroundTruthRecord::Audit(audit) => audit
                .reason
                .as_deref()
                .is_some_and(|reason| reason == format!("step of {}", center.command_id)),
            _ => false,
        });
        assert!(audited_as_step, "the step's proposal names its parent");
    }

    #[test]
    fn a_relayed_operator_capture_is_audited_as_the_relay_and_automatic_ones_stay_automatic() {
        assert_eq!(
            CaptureSource::Operator.command_source_over(true),
            CommandSource::Relay { overriding: false }
        );
        assert_eq!(
            CaptureSource::Operator.command_source_over(false),
            CommandSource::Operator { overriding: false }
        );
        let trigger = CaptureSource::Trigger {
            rule: "sequence".into(),
        };
        assert!(trigger.command_source_over(true).is_automatic());
        assert_eq!(
            trigger.command_source_over(true),
            trigger.command_source_over(false)
        );
    }

    #[test]
    fn automatic_captures_are_refused_while_a_resume_block_is_active() {
        let (hub, _clock, sink) = hub();
        let block = hub.raise_block(BlockRequest::new(
            ResumeBlockKind::FilterHandshakeIncomplete,
            "wheel timeout",
        ));
        let trigger = CaptureSource::Trigger {
            rule: "sequence".into(),
        }
        .command_source_over(false);
        let refused = authorize_capture(&hub, &trigger, None, "light", json!({})).err();
        assert_eq!(refused.as_ref().map(GateError::http_status), Some(423));
        assert_eq!(refused.map(|e| e.code()), Some("resume_block_active"));
        // The operator's capture runs, and knows which blocks were active.
        let operator = CaptureSource::Operator.command_source_over(false);
        let authorized = authorize_capture(&hub, &operator, None, "light", json!({})).unwrap();
        assert_eq!(authorized.blocks.len(), 1);
        assert_eq!(authorized.blocks[0].id, block.id);
        assert!(hub
            .gate()
            .holder(crate::equipment::commands::Domain::Device(
                DeviceKind::Camera
            ))
            .is_some());
        drop(authorized);
        assert!(
            hub.gate().is_idle(),
            "the camera domain is released with the permit"
        );
        // Every stage is audited.
        let audits = sink
            .records()
            .into_iter()
            .filter(|r| matches!(r, GroundTruthRecord::Audit(_)))
            .count();
        assert!(audits >= 3, "{audits}");
        // An operator capture's frame is flagged.
        let (clock_hub, clock, _) = self::hub();
        let (interval, _) = still(&clock, 1.0);
        let mut flagged = input("frame-b", None);
        flagged.blocks_at_start = vec![block];
        let equipment = assemble(&clock_hub, None, &interval, &flagged);
        assert!(equipment.row.resume_block_active);
        assert_eq!(
            equipment.block["resume_blocks"][0]["kind"]["kind"],
            "filter_handshake_incomplete"
        );
    }

    #[test]
    fn exposure_hooks_hold_the_busy_flag_and_name_the_frame() {
        let (hub, clock, sink) = hub();
        let camera = Uuid::new_v4();
        let runtime = CameraRuntimeState {
            connection_id: Some(camera.to_string()),
            driver_id: Some("asi".into()),
            ..CameraRuntimeState::default()
        };
        let mut hooks = CaptureHooks::new(Arc::clone(&hub), &runtime, "cap-1", "frame-1");
        let busy = hub.busy();
        hooks.exposure_started(AcquisitionPath::Stream, 2.0, Some(7));
        assert!(busy.borrow().exposing);
        clock.advance_ms(2_000);
        hooks.exposure_ended();
        assert!(!busy.borrow().exposing);
        let interval = ExposureInterval::stream(
            clock.now_utc().timestamp() as f64,
            2.0,
            None,
            hooks.clock_quality(AcquisitionPath::Stream),
        );
        hooks.exposure_completed(Some("frame-1"), &interval);
        let events: Vec<Arc<EquipmentEvent>> = sink
            .records()
            .into_iter()
            .filter_map(|r| match r {
                GroundTruthRecord::Event(e) => Some(e),
                _ => None,
            })
            .collect();
        assert_eq!(events.len(), 2);
        let EventKind::ExposureStarted { exposure } = &events[0].event else {
            panic!("{:?}", events[0].event)
        };
        assert_eq!(exposure.frame_id.as_deref(), Some("frame-1"));
        assert_eq!(exposure.stream_generation, Some(7));
        assert_eq!(exposure.clock_quality, ClockQuality::HostArrival);
        assert_eq!(events[0].device_kind, Some(DeviceKind::Camera));
        assert_eq!(events[0].connection_id, Some(camera));
        let EventKind::ExposureCompleted {
            exposure,
            completed_at,
        } = &events[1].event
        else {
            panic!("{:?}", events[1].event)
        };
        assert_eq!(exposure.frame_id.as_deref(), Some("frame-1"));
        assert_eq!(exposure.capture_id.as_deref(), Some("cap-1"));
        assert_eq!(exposure.started_at, Some(interval.start));
        assert_eq!(*completed_at, interval.end);
        // Dropping the hooks mid-exposure (an error path) clears the flag.
        let mut failing = CaptureHooks::new(Arc::clone(&hub), &runtime, "cap-2", "frame-2");
        failing.exposure_started(AcquisitionPath::Stream, 1.0, None);
        assert!(busy.borrow().exposing);
        drop(failing);
        assert!(!busy.borrow().exposing);
    }

    #[test]
    fn settings_changes_are_reported_once_per_change() {
        let camera = format!("camera-{}", Uuid::new_v4());
        let update = |exposure: f64, gain: f64| {
            BTreeMap::from([
                ("exposure_s".to_string(), json!(exposure)),
                ("gain".to_string(), json!(gain)),
            ])
        };
        let first = note_settings(&camera, &update(2.0, 100.0)).expect("unknown before");
        assert_eq!(first["exposure_s"]["from"], Value::Null);
        assert!(note_settings(&camera, &update(2.0, 100.0)).is_none());
        assert!(
            note_settings(&camera, &update(2.0 + 1e-12, 100.0)).is_none(),
            "readback jitter is not a change"
        );
        let changed = note_settings(&camera, &update(4.0, 100.0)).unwrap();
        assert_eq!(changed["exposure_s"]["from"], 2.0);
        assert_eq!(changed["exposure_s"]["to"], 4.0);
        assert!(changed.get("gain").is_none());
        // Emitted as an event with the cause and the full known settings.
        let (hub, _clock, sink) = hub();
        let runtime = CameraRuntimeState {
            connection_id: Some(camera.clone()),
            ..CameraRuntimeState::default()
        };
        settings_applied(&hub, &runtime, update(8.0, 100.0), "capture", Some("cap-9"));
        settings_applied(
            &hub,
            &runtime,
            update(8.0, 100.0),
            "capture",
            Some("cap-10"),
        );
        assert_eq!(sink.event_names(), vec!["exposure_settings_changed"]);
        emit_stream_restarted(&hub, runtime.connection_id.as_deref(), 3);
        assert_eq!(
            sink.event_names(),
            vec!["exposure_settings_changed", "stream_restarted"]
        );
    }

    #[test]
    fn the_window_is_rewritten_once_final_and_not_before() {
        let (hub, clock, sink) = hub();
        let (mount, _m) = fake_mount(MountSnapshot::default());
        let ring = hub.attach(mount);
        ring.push(mount_at(&clock, 3.0, 30.0, true));
        clock.advance_ms(100);
        let (interval, _) = still(&clock, 3.0);
        clock.advance_ms(2_000);
        ring.push(mount_at(&clock, 3.0, 30.0, true));
        // The exposure just ended and no mount sample after it exists yet:
        // the window at commit is provisional, and nothing is rewritten.
        clock.advance_ms(1_100);
        let plan = FinalisePlan {
            input: input("frame-final", None),
            interval: interval.clone(),
            finalised_at_commit: false,
        };
        assert!(!assemble(&hub, None, &interval, &plan.input).row.finalised);
        assert!(!finalise(&hub, None, &plan));
        // A sample after the end makes it final; the row is recorded once.
        clock.advance_ms(500);
        ring.push(mount_at(&clock, 3.0, 30.0, true));
        assert!(finalise(&hub, None, &plan));
        let rows: Vec<ExposureWindowRow> = sink
            .records()
            .into_iter()
            .filter_map(|r| match r {
                GroundTruthRecord::ExposureWindow(row) => Some(*row),
                _ => None,
            })
            .collect();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].finalised);
        assert_eq!(rows[0].frame_id, "frame-final");
        assert_eq!(rows[0].tracking, Some(true));
    }
}

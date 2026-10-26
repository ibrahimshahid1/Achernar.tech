//! `GuiderService`: the PHD2 tokio task behind a [`GuiderHandle`] (spec §3.5
//! G-a to G-j read side, G-l and G-m hooks; H9, H10; WP1A-7).
//!
//! The service is to PHD2 what a device worker (WP1A-2) is to a polled
//! driver: it owns the connection, publishes [`GuiderSnapshot`]s through the
//! handle's watch, keeps the handle's health, runs one operation at a time
//! and emits the §4.2 guider events. It is an independent implementation from
//! the specification; NINA's PHD2 guider is a behaviour reference only (§6).
//!
//! Sequence and rules:
//! - **G-a connect** ([`connect`], [`connect_with_id`]): consume `Version`
//!   and `AppState`; call `get_connected`, and `set_connected [true]` only
//!   when `[guider] connect_equipment = true`; read the session context
//!   (profile, equipment, pixel scale, calibration, Dec guide mode, exposure,
//!   algorithm parameter names and values). No profile switching.
//! - **§2.6 steps 9-10**: `connect` emits and records nothing. The caller
//!   publishes the slot and registers the `connection_id` with the hub, then
//!   calls [`Phd2Connection::start`], which emits `device_connected`, the
//!   `capability_unsupported` events and what the connect burst produced,
//!   inserts the `device_connections` row and starts the service task.
//! - **G-b state** follows events and is reconciled with `get_app_state`
//!   every 5 s, which wins.
//! - **G-c start**: `guide {settle, recalibrate}`, then exactly one
//!   `SettleDone` within `settle.timeout + 10 s`.
//! - **G-d stop**: `stop_capture`, then `get_app_state` every 1 s for up to
//!   10 s until Stopped.
//! - **G-e pause / resume**: `set_paused`, then the Paused event or state.
//! - **G-f dither**: refused unless Guiding; waits for a settle in progress;
//!   `dither {amount, raOnly, settle}`; records `GuidingDithered`, every
//!   `Settling` and `SettleDone` in a `dither_events` row (H10).
//! - **G-g / G-h**: every event NINA ignores is reported; every GuideStep is
//!   recorded unmodified with its phase tag (`guide_steps`).
//! - **G-i**: sliding 50-step RMS in the snapshot; per-exposure statistics
//!   and the `guider_state_log` row through [`GuiderWindows`] (H9).
//! - **G-j read side**: aggressiveness names are discovered, never assumed.
//!   `SetAlgoParam` (the M-1 actuator) is Tier 2 (WP2-7) and is refused here.
//! - **Cancel** (abort, e-stop, preemption): the abort path already sent
//!   `stop_capture` out of band (D4). The service sends it in band too
//!   (D19: always attempted), then resolves the operation `Aborted { by }`
//!   once PHD2 reports Stopped, or `Unknown` if it does not stop within the
//!   G-d stop timeout.
//! - **D16**: Achernar never sends `set_connected [false]`; disconnecting
//!   leaves PHD2 guiding as it is.
//! - **Loss**: EOF or a socket error makes the device `Lost` (`device_lost`,
//!   which raises the hub's `guider_lost` resume block). Tier 1 does not
//!   reconnect. An active operation resolves `Unknown`.
//!
//! Events go to the hub through [`EventEmitter`] (the hub assigns `seq` and
//! journals them). Ground truth that is not an event (`guide_steps`,
//! `dither_events`, `guider_state_log`, `device_connections`) goes straight to
//! the [`RecorderSink`].

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use uuid::Uuid;

use super::phd2_client::{
    until_deadline, ClientEvent, ClientTasks, Phd2AbortPath, Phd2Client, TICK,
};
use super::phd2_proto::{
    self as proto, method, EventBody, GuideStepEvent, Phd2Event, SettleDoneEvent,
};
use super::rms::{PhaseFilter, SlidingRms};
use super::steps::{
    sample_from_event, step_record, ClockOffset, Marker, PhaseTracker, StateRecordContext,
    StepHistory,
};
use crate::config::{EquipmentConfig, GuiderConfig};
use crate::equipment::clock::duration_ns;
use crate::equipment::commands::{Axis, GuiderCmd, PixelUnit, Settle};
use crate::equipment::events::{
    CalibrationPhase, ConnectionEnd, DeviceConnectionRecord, DitherBackend, DitherRecord,
    EquipmentEvent, EventKind, GroundTruthRecord, GuiderStateRecord,
};
use crate::equipment::handle::{
    DeviceHandle, Envelope, EnvelopeBody, GuiderHandle, Permit, WorkerEnd,
};
use crate::equipment::ops::{ActiveOp, OpKind, OpPhase};
use crate::equipment::snapshot::{
    AlgoParam, GuideStepSample, GuideWindowStats, GuiderAppState, GuiderSnapshot, LastDither,
    SettleProgress,
};
use crate::equipment::traits::{AbortPath, Clock, ConnectPath, ConnectReport, RecorderSink};
use crate::equipment::types::{
    identity_key, CommandId, CommandSource, DeviceError, DeviceIdentity, DeviceKind, DeviceResult,
    DriverId, Endpoint, ErrorClass, FailureClass, HealthState, LinkState, Observed, OpOutcome,
    Qualification, Source,
};

/// Delivers events to the hub (WP1A-2), which assigns `seq`, drops events
/// from stale connections and journals them.
pub type EventEmitter = Arc<dyn Fn(EquipmentEvent) + Send + Sync>;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

fn secs(value: f64) -> Duration {
    Duration::from_secs_f64(if value.is_finite() {
        value.clamp(0.0, 86_400.0)
    } else {
        0.0
    })
}

/// The settle timeout plus margin for a settle Achernar did not request
/// (the step tracker's bound, and the G-f wait for someone else's settle).
fn default_settle_bound(settings: &Phd2Settings) -> Duration {
    secs(proto::sent_settle(&settings.default_settle).timeout_s) + settings.settle_margin
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// One PHD2 connection's settings, from `[guider]` and `[equipment]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Phd2Settings {
    pub host: String,
    pub port: u16,
    /// TCP connect only (`[equipment] connect_timeout_s`).
    pub connect_timeout: Duration,
    /// Default RPC timeout (`[guider] rpc_timeout_s`); settle waits are
    /// separate.
    pub rpc_timeout: Duration,
    /// `[guider] connect_equipment`: call `set_connected [true]` when PHD2's
    /// equipment is not connected. Off by default (D16).
    pub connect_equipment: bool,
    /// Settle for dithers and starts that come without one, and the
    /// tolerance used to tag steps of a settle Achernar did not request.
    pub default_settle: Settle,
    pub aggressiveness_param_ra: String,
    pub aggressiveness_param_dec: String,
    /// Consecutive unanswered requests before the guider is `Lost`.
    pub lost_after_failures: u32,
    /// G-b reconciliation with `get_app_state`.
    pub reconcile_interval: Duration,
    /// G-d: how long `stop_capture` may take to reach Stopped.
    pub stop_timeout: Duration,
    /// G-d / G-e state polling while waiting.
    pub poll_interval: Duration,
    /// G-e: how long a pause or resume may take.
    pub pause_timeout: Duration,
    /// G-c / G-f: added to the settle timeout.
    pub settle_margin: Duration,
    /// The out-of-band `stop_capture` connection (D4).
    pub abort_timeout: Duration,
    /// How long guide steps stay available for exposure windows.
    pub history_retention: Duration,
}

impl Phd2Settings {
    /// Defaults for `host:port`.
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self::from_config(
            host,
            port,
            &GuiderConfig::default(),
            &EquipmentConfig::default(),
        )
    }

    pub fn from_config(
        host: impl Into<String>,
        port: u16,
        guider: &GuiderConfig,
        equipment: &EquipmentConfig,
    ) -> Self {
        Self {
            host: host.into(),
            port,
            connect_timeout: Duration::from_secs(equipment.connect_timeout_s.max(1)),
            rpc_timeout: Duration::from_secs(guider.rpc_timeout_s.max(1)),
            connect_equipment: guider.connect_equipment,
            default_settle: guider.settle(),
            aggressiveness_param_ra: guider.aggressiveness_param_ra.clone(),
            aggressiveness_param_dec: guider.aggressiveness_param_dec.clone(),
            lost_after_failures: equipment.lost_after_failures.max(1),
            reconcile_interval: Duration::from_secs(5),
            stop_timeout: Duration::from_secs(10),
            poll_interval: Duration::from_secs(1),
            pause_timeout: Duration::from_secs(60),
            settle_margin: Duration::from_secs(10),
            abort_timeout: Duration::from_secs(5),
            history_retention: Duration::from_secs(equipment.ring_buffer_minutes.max(1) * 60),
        }
    }

    pub fn endpoint(&self) -> Endpoint {
        Endpoint::Phd2 {
            host: self.host.clone(),
            port: self.port,
        }
    }
}

/// Parse a `/connect` option for driver `phd2` (§2.12): `host:port`,
/// `host` (port 4400), `:port` (localhost) or `[v6]:port`. An empty option
/// means `localhost:4400`, the first PHD2 instance.
pub fn parse_option(option: &str) -> DeviceResult<(String, u16)> {
    let text = option.trim();
    let text = text.strip_prefix("phd2://").unwrap_or(text);
    let invalid = || {
        DeviceError::invalid_value(format!(
            "PHD2 option {option:?} is not host:port (instance n listens on 4400 + n - 1)"
        ))
    };
    let (host, port) = if let Some(rest) = text.strip_prefix('[') {
        let (host, after) = rest.split_once(']').ok_or_else(invalid)?;
        let port = match after {
            "" => None,
            p => Some(p.strip_prefix(':').ok_or_else(invalid)?),
        };
        (host.to_string(), port)
    } else {
        match text.rsplit_once(':') {
            Some((host, _)) if host.contains(':') => return Err(invalid()),
            Some((host, port)) => (host.to_string(), Some(port)),
            None => (text.to_string(), None),
        }
    };
    let host = match host.trim() {
        "" => "localhost".to_string(),
        h => h.to_string(),
    };
    let port = match port {
        None => Endpoint::PHD2_BASE_PORT,
        Some(p) => p
            .trim()
            .parse::<u16>()
            .ok()
            .filter(|p| *p > 0)
            .ok_or_else(invalid)?,
    };
    Ok((host, port))
}

// ---------------------------------------------------------------------------
// Aggressiveness discovery (G-j)
// ---------------------------------------------------------------------------

/// Which algorithm parameter on each axis is the aggressiveness knob.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AggressivenessDiscovery {
    pub ra: Option<String>,
    pub dec: Option<String>,
    /// Why M-1 actuation is unavailable, e.g. `no_aggressiveness_param`.
    pub disabled_reason: Option<String>,
}

/// G-j: a configured name must exist in PHD2's list; with no configured
/// name, the first name equal to `aggression` or `aggressiveness`
/// (case-insensitive) is used. Names are never assumed.
pub fn discover_aggressiveness(
    ra_names: &[String],
    dec_names: &[String],
    configured_ra: &str,
    configured_dec: &str,
) -> AggressivenessDiscovery {
    fn pick(names: &[String], configured: &str) -> Result<String, &'static str> {
        let configured = configured.trim();
        if !configured.is_empty() {
            return names
                .iter()
                .find(|n| n.trim().eq_ignore_ascii_case(configured))
                .cloned()
                .ok_or("configured_aggressiveness_param_missing");
        }
        names
            .iter()
            .find(|n| {
                let lower = n.trim().to_ascii_lowercase();
                lower == "aggression" || lower == "aggressiveness"
            })
            .cloned()
            .ok_or("no_aggressiveness_param")
    }
    let ra = pick(ra_names, configured_ra);
    let dec = pick(dec_names, configured_dec);
    let disabled_reason = match (&ra, &dec) {
        (Ok(_), Ok(_)) => None,
        (Err(a), Err(b)) if a == b => Some((*a).to_string()),
        (Err(a), Err(b)) => Some(format!("ra:{a};dec:{b}")),
        (Err(a), Ok(_)) => Some(format!("{a}:ra")),
        (Ok(_), Err(b)) => Some(format!("{b}:dec")),
    };
    AggressivenessDiscovery {
        ra: ra.ok(),
        dec: dec.ok(),
        disabled_reason,
    }
}

// ---------------------------------------------------------------------------
// Public connection types
// ---------------------------------------------------------------------------

/// What the service needs from the rest of the layer.
#[derive(Clone)]
pub struct GuiderContext {
    pub clock: Arc<dyn Clock>,
    pub recorder: Arc<dyn RecorderSink>,
    pub events: EventEmitter,
}

/// A connected PHD2 guider, not yet started.
///
/// §2.6 steps 9-10: publish the slot and register `handle`'s
/// `connection_id` with the hub, then call [`Phd2Connection::start`]. Until
/// then nothing is emitted or recorded, PHD2's events wait in the client's
/// queue, and commands wait in the handle's queue (`abort` still runs its
/// out-of-band path at once). Dropping a connection that was never started
/// closes the socket.
pub struct Phd2Connection {
    pub handle: GuiderHandle,
    /// Per-exposure statistics and `guider_state_log` rows (H9).
    pub windows: GuiderWindows,
    pub report: ConnectReport,
    /// The `device_connected` capabilities payload.
    pub capabilities: Value,
    identity: DeviceIdentity,
    connected_at: DateTime<Utc>,
    recorder: Arc<dyn RecorderSink>,
    /// The service until [`Phd2Connection::start`] spawns it.
    pending: Mutex<Option<Service>>,
    task: Option<JoinHandle<()>>,
}

impl Phd2Connection {
    pub fn connection_id(&self) -> Uuid {
        self.identity.connection_id
    }

    /// The identity `device_connected` will carry.
    pub fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    pub fn is_started(&self) -> bool {
        self.task.is_some()
    }

    /// §2.6 steps 9-10, after the caller published the slot: emit
    /// `device_connected`, then `capability_unsupported` per unsupported
    /// member, then the events the connect burst produced (the initial
    /// `guider_state`); insert the `device_connections` row; start the
    /// service task. Must run inside a tokio runtime. Returns `false` when
    /// already started.
    pub fn start(&mut self) -> bool {
        let Some(mut service) = self
            .pending
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        else {
            return false;
        };
        let buffered = std::mem::take(&mut service.core.outbox);
        service.core.emit(
            EventKind::DeviceConnected {
                identity: Box::new(self.identity.clone()),
                capabilities: self.capabilities.clone(),
            },
            None,
            None,
        );
        for member in &self.report.unsupported_members {
            service.core.emit(
                EventKind::CapabilityUnsupported {
                    member: member.clone(),
                },
                None,
                None,
            );
        }
        service.core.outbox.extend(buffered);
        service.flush();
        self.recorder
            .record(GroundTruthRecord::ConnectionOpened(Box::new(
                DeviceConnectionRecord {
                    identity: self.identity.clone(),
                    capabilities: self.capabilities.clone(),
                    connected_at: self.connected_at,
                    session_id: None,
                    night_id: None,
                },
            )));
        self.task = Some(tokio::spawn(service.run()));
        true
    }

    /// The service task once started. It ends after a disconnect; aborting
    /// it is safe (the socket closes).
    pub fn task_mut(&mut self) -> Option<&mut JoinHandle<()>> {
        self.task.as_mut()
    }

    /// Abort the service task, or drop the service that was never started.
    pub fn abort_task(&mut self) {
        if let Some(task) = &self.task {
            task.abort();
        }
        drop(
            self.pending
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .take(),
        );
    }
}

/// Read side of the guide-step history, for exposure windows (G-i, H9).
/// Called by the window builder (WP1A-2) and on `exposure_completed`
/// (WP1B-3); never touches PHD2.
#[derive(Clone)]
pub struct GuiderWindows {
    history: Arc<Mutex<StepHistory>>,
    snapshot: watch::Receiver<Arc<GuiderSnapshot>>,
    recorder: Arc<dyn RecorderSink>,
    clock: Arc<dyn Clock>,
    connection_id: Uuid,
}

impl GuiderWindows {
    pub fn connection_id(&self) -> Uuid {
        self.connection_id
    }

    /// G-i statistics, dither and settle steps excluded.
    pub fn window_stats(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> GuideWindowStats {
        self.window_stats_with(start, end, &PhaseFilter::default())
    }

    pub fn window_stats_with(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        filter: &PhaseFilter,
    ) -> GuideWindowStats {
        lock(&self.history).window_stats(start, end, filter)
    }

    /// The raw steps placed inside `[start, end]`.
    pub fn steps(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<GuideStepSample> {
        lock(&self.history).steps_between(start, end)
    }

    /// The `guider_state_log` row for one exposure, not yet recorded.
    pub fn state_record(
        &self,
        frame_id: Option<String>,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> GuiderStateRecord {
        let snapshot = Arc::clone(&self.snapshot.borrow());
        let ctx = StateRecordContext {
            connection_id: self.connection_id,
            algo_params: snapshot.algo_params.clone(),
            aggressiveness_param_ra: snapshot.aggressiveness_param_ra.clone(),
            aggressiveness_param_dec: snapshot.aggressiveness_param_dec.clone(),
        };
        lock(&self.history).state_record(&ctx, frame_id, start, end, self.clock.now_utc())
    }

    /// Build and record the row (H9: one per exposure window).
    pub fn record_exposure_window(
        &self,
        frame_id: Option<String>,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> GuiderStateRecord {
        let record = self.state_record(frame_id, start, end);
        self.recorder
            .record(GroundTruthRecord::GuiderWindow(Box::new(record.clone())));
        record
    }
}

// ---------------------------------------------------------------------------
// Core: snapshot and event handling shared by connect and the task
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Stamp {
    at: DateTime<Utc>,
    mono: u64,
    device_time: Option<DateTime<Utc>>,
    window_at: DateTime<Utc>,
    latency_ms: Option<u32>,
    source: Source,
}

struct Core {
    connection_id: Uuid,
    settings: Phd2Settings,
    clock: Arc<dyn Clock>,
    recorder: Arc<dyn RecorderSink>,
    snap: GuiderSnapshot,
    tracker: PhaseTracker,
    sliding: SlidingRms,
    history: Arc<Mutex<StepHistory>>,
    offset: ClockOffset,
    outbox: Vec<EquipmentEvent>,
    malformed: u64,
    refresh_algo: bool,
    refresh_context: bool,
    algo_names_ra: Vec<String>,
    algo_names_dec: Vec<String>,
    unsupported: BTreeSet<String>,
    msg_version: Option<i64>,
    equipment_connected: Option<bool>,
    state_before_pause: Option<GuiderAppState>,
    dither_awaiting_settle: bool,
}

impl Core {
    fn new(
        settings: Phd2Settings,
        clock: Arc<dyn Clock>,
        recorder: Arc<dyn RecorderSink>,
        connection_id: Uuid,
    ) -> Self {
        let retention = chrono::Duration::from_std(settings.history_retention)
            .unwrap_or_else(|_| chrono::Duration::minutes(30));
        let tracker_bound = default_settle_bound(&settings);
        let snap = GuiderSnapshot {
            connection_id,
            app_state: Observed::not_set(Source::Phd2Event),
            published_at: clock.now_utc(),
            published_mono_ns: clock.now_mono_ns(),
            ..GuiderSnapshot::default()
        };
        Self {
            connection_id,
            settings,
            clock,
            recorder,
            snap,
            tracker: PhaseTracker::new(tracker_bound),
            sliding: SlidingRms::default(),
            history: Arc::new(Mutex::new(StepHistory::new(retention, 20_000))),
            offset: ClockOffset::new(),
            outbox: Vec::new(),
            malformed: 0,
            refresh_algo: false,
            refresh_context: false,
            algo_names_ra: Vec::new(),
            algo_names_dec: Vec::new(),
            unsupported: BTreeSet::new(),
            msg_version: None,
            equipment_connected: None,
            state_before_pause: None,
            dither_awaiting_settle: false,
        }
    }

    fn emit(
        &mut self,
        kind: EventKind,
        command_id: Option<CommandId>,
        device_time: Option<DateTime<Utc>>,
    ) {
        let mut event = EquipmentEvent::new(kind, self.clock.as_ref())
            .for_device(DeviceKind::Guider, self.connection_id);
        event.command_id = command_id;
        event.device_time = device_time;
        self.outbox.push(event);
    }

    fn state(&self) -> GuiderAppState {
        self.snap.app_state.value.unwrap_or_default()
    }

    /// Drop a dither or settle tag that ran out of time (a dither nobody
    /// settles, an unanswered dither, a lost `SettleDone`), with the
    /// snapshot's settle flags. Returns `true` when one was dropped.
    fn expire_settle_tags(&mut self, now_mono_ns: u64) -> bool {
        if !self.tracker.expire(now_mono_ns) {
            return false;
        }
        tracing::debug!(
            connection_id = %self.connection_id,
            "PHD2 dither/settle tag lapsed without SettleDone; steps are tagged guiding again"
        );
        self.snap.settling = false;
        self.snap.settle = None;
        true
    }

    /// A dither or settle is in progress (G-f step 1).
    fn settle_in_progress(&self, now_mono_ns: u64) -> bool {
        self.tracker.is_settling(now_mono_ns) || self.snap.settling
    }

    fn set_state(&mut self, to: GuiderAppState, stamp: Stamp) {
        let from = self.state();
        let mut observed = Observed::fresh(to, stamp.source, stamp.at, stamp.mono);
        observed.device_time = stamp.device_time;
        observed.latency_ms = stamp.latency_ms;
        self.snap.app_state = observed;
        if from == to {
            return;
        }
        lock(&self.history).push_marker(stamp.window_at, Marker::State(to));
        if matches!(to, GuiderAppState::Stopped | GuiderAppState::Selected) {
            self.tracker.on_guiding_stopped();
            self.snap.settling = false;
            self.snap.settle = None;
        }
        self.emit(EventKind::GuiderState { from, to }, None, stamp.device_time);
    }

    fn marker(&self, at: DateTime<Utc>, marker: Marker) {
        lock(&self.history).push_marker(at, marker);
    }

    fn on_event(&mut self, event: &Phd2Event, received_at: DateTime<Utc>, received_mono: u64) {
        let device_time = event.device_time();
        if let Some(ts) = event.timestamp {
            self.offset.update(ts, received_at);
            self.snap.clock_offset_s = self.offset.offset_s();
        }
        let stamp = Stamp {
            at: received_at,
            mono: received_mono,
            device_time,
            window_at: self.offset.step_time(event.timestamp, received_at),
            latency_ms: None,
            source: Source::Phd2Event,
        };
        let at = stamp.window_at;
        match &event.body {
            EventBody::Version(version) => {
                if let Some(text) = version.display() {
                    self.snap.phd2_version = Some(text);
                }
                self.msg_version = version.msg_version.or(self.msg_version);
            }
            EventBody::AppState { state } => {
                if let Some(state) = state {
                    self.set_state(GuiderAppState::from_phd2(state), stamp);
                }
            }
            EventBody::GuideStep(step) => self.on_guide_step(step, event.timestamp, stamp),
            EventBody::StarLost(lost) => {
                self.tracker.on_star_lost();
                self.marker(at, Marker::StarLost);
                self.set_state(GuiderAppState::LostLock, stamp);
                self.emit(
                    EventKind::StarLost {
                        frame: lost.frame,
                        star_mass: lost.star_mass,
                        snr: lost.snr,
                        avg_dist: lost.avg_dist,
                        error_code: lost.error_code.and_then(|c| i32::try_from(c).ok()),
                        status: lost.status.clone(),
                    },
                    None,
                    device_time,
                );
            }
            EventBody::Settling(settling) => {
                self.tracker.on_settling(
                    settling.distance,
                    self.settings.default_settle.pixels,
                    stamp.mono,
                );
                self.snap.settling = true;
                if let (Some(distance), Some(time), Some(settle_time)) =
                    (settling.distance, settling.time, settling.settle_time)
                {
                    let progress = SettleProgress {
                        distance_px: distance,
                        time_s: time,
                        settle_time_s: settle_time,
                        star_locked: settling.star_locked.unwrap_or(false),
                    };
                    self.snap.settle = Some(progress);
                    if settling.star_locked.is_some() {
                        self.emit(EventKind::Settling { progress }, None, device_time);
                    }
                }
            }
            EventBody::SettleBegin => {
                self.tracker.on_settle_begin(stamp.mono);
                self.snap.settling = true;
                self.marker(at, Marker::SettleBegin);
                self.emit(EventKind::SettleBegin, None, device_time);
            }
            EventBody::SettleDone(done) => {
                self.tracker.on_settle_done();
                self.snap.settling = false;
                self.snap.settle = None;
                self.marker(at, Marker::SettleDone);
                if self.dither_awaiting_settle {
                    self.dither_awaiting_settle = false;
                    if let Some(last) = self.snap.last_dither.as_mut() {
                        last.settle_status = Some(done.status_code());
                    }
                }
                self.emit(
                    EventKind::SettleDone {
                        status: done.status_code(),
                        error: done.error.clone().filter(|e| !e.trim().is_empty()),
                        total_frames: done.total_frames.and_then(|n| u32::try_from(n).ok()),
                        dropped_frames: done.dropped_frames.and_then(|n| u32::try_from(n).ok()),
                    },
                    None,
                    device_time,
                );
            }
            EventBody::GuidingDithered { dx, dy } => {
                let grace = PhaseTracker::unsolicited_grace(self.snap.exposure_ms.map(u64::from));
                self.tracker.on_dithered(stamp.mono, grace);
                self.marker(at, Marker::Dither);
                self.snap.last_dither = Some(LastDither {
                    at: received_at,
                    dx: *dx,
                    dy: *dy,
                    settle_status: None,
                });
                self.dither_awaiting_settle = true;
                if let (Some(dx), Some(dy)) = (dx, dy) {
                    self.emit(EventKind::Dithered { dx: *dx, dy: *dy }, None, device_time);
                }
            }
            EventBody::StartGuiding => {
                self.sliding.clear();
                self.set_state(GuiderAppState::Guiding, stamp);
                self.emit(EventKind::GuidingStarted, None, device_time);
            }
            EventBody::GuidingStopped => {
                self.tracker.on_guiding_stopped();
                self.snap.settling = false;
                self.snap.settle = None;
                self.set_state(GuiderAppState::Stopped, stamp);
                self.emit(EventKind::GuidingStopped, None, device_time);
            }
            EventBody::Paused => {
                self.tracker.on_paused();
                let before = self.state();
                if before != GuiderAppState::Paused {
                    self.state_before_pause = Some(before);
                }
                self.set_state(GuiderAppState::Paused, stamp);
                self.emit(EventKind::GuiderPaused, None, device_time);
            }
            EventBody::Resumed => {
                self.tracker.on_resumed();
                self.emit(EventKind::GuiderResumed, None, device_time);
                // PHD2 sends no state with Resumed; restore the state before
                // the pause, and let reconciliation correct it.
                if let Some(before) = self.state_before_pause.take() {
                    if before != GuiderAppState::Paused && before != GuiderAppState::Unknown {
                        self.set_state(before, stamp);
                    }
                }
            }
            EventBody::LoopingExposures { .. } => self.set_state(GuiderAppState::Looping, stamp),
            EventBody::LoopingExposuresStopped => self.set_state(GuiderAppState::Stopped, stamp),
            EventBody::StartCalibration { mount } => {
                self.set_state(GuiderAppState::Calibrating, stamp);
                self.emit(
                    EventKind::Calibration {
                        phase: CalibrationPhase::Started,
                        detail: mount.clone(),
                    },
                    None,
                    device_time,
                );
            }
            EventBody::Calibrating(step) => {
                self.set_state(GuiderAppState::Calibrating, stamp);
                let detail = format!(
                    "{} step {} dir {} dist {}",
                    step.mount.as_deref().unwrap_or("mount"),
                    step.step.map_or("?".into(), |s| s.to_string()),
                    step.dir.as_deref().unwrap_or("?"),
                    step.dist.map_or("?".into(), |d| format!("{d:.2}")),
                );
                self.emit(
                    EventKind::Calibration {
                        phase: CalibrationPhase::Step,
                        detail: Some(detail),
                    },
                    None,
                    device_time,
                );
            }
            EventBody::CalibrationComplete { mount } => {
                self.snap.calibrated = Some(true);
                self.refresh_context = true;
                self.emit(
                    EventKind::Calibration {
                        phase: CalibrationPhase::Complete,
                        detail: mount.clone(),
                    },
                    None,
                    device_time,
                );
            }
            EventBody::CalibrationFailed { reason } => {
                self.emit(
                    EventKind::Calibration {
                        phase: CalibrationPhase::Failed,
                        detail: reason.clone(),
                    },
                    None,
                    device_time,
                );
            }
            EventBody::CalibrationDataFlipped { mount } => {
                self.refresh_context = true;
                self.emit(
                    EventKind::Calibration {
                        phase: CalibrationPhase::DataFlipped,
                        detail: mount.clone(),
                    },
                    None,
                    device_time,
                );
            }
            EventBody::LockPositionLost => {
                self.tracker.on_star_lost();
                self.marker(at, Marker::LockPositionLost);
                self.emit(EventKind::LockPositionLost, None, device_time);
            }
            EventBody::LockPositionShiftLimitReached => self.emit(
                EventKind::GuiderAlert {
                    message: "PHD2 reached the lock-position shift limit".into(),
                    alert_type: Some("warning".into()),
                },
                None,
                device_time,
            ),
            EventBody::Alert {
                message,
                alert_type,
            } => self.emit(
                EventKind::GuiderAlert {
                    message: message.clone().unwrap_or_default(),
                    alert_type: alert_type.clone(),
                },
                None,
                device_time,
            ),
            EventBody::GuideParamChange { name, value } => {
                // Operator edits in PHD2 confound M-1 data: mark the window
                // and re-read the parameters.
                self.marker(at, Marker::ParamChanged);
                self.refresh_algo = true;
                self.emit(
                    EventKind::GuideParamChanged {
                        name: name.clone().unwrap_or_default(),
                        value: value.clone().unwrap_or(Value::Null),
                    },
                    None,
                    device_time,
                );
            }
            EventBody::ConfigurationChange => self.refresh_context = true,
            EventBody::LockPositionSet { .. }
            | EventBody::StarSelected { .. }
            | EventBody::Other(_) => {}
        }
    }

    fn on_guide_step(&mut self, step: &GuideStepEvent, timestamp: Option<f64>, stamp: Stamp) {
        let state = self.state();
        self.expire_settle_tags(stamp.mono);
        let phase = self.tracker.phase_for_step(state, stamp.mono);
        let Some(sample) = sample_from_event(
            step,
            timestamp,
            stamp.at,
            phase,
            self.snap.pixel_scale_arcsec,
        ) else {
            self.malformed += 1;
            tracing::warn!(
                frame = ?step.frame,
                "PHD2 GuideStep without Frame, Time, dx or dy; not recorded"
            );
            return;
        };
        if state != GuiderAppState::Paused {
            self.set_state(GuiderAppState::Guiding, stamp);
        }
        if PhaseFilter::default().includes(phase) {
            self.sliding.push_step(&sample);
        }
        self.snap.rms = self.sliding.rms();
        self.snap.last_step = Some(sample.clone());
        lock(&self.history).push_step(stamp.window_at, sample.clone());
        self.recorder
            .record(GroundTruthRecord::GuideStep(Box::new(step_record(
                self.connection_id,
                sample.clone(),
            ))));
        self.emit(
            EventKind::GuideStep {
                step: Box::new(sample),
            },
            None,
            stamp.device_time,
        );
    }

    fn apply_discovery(&mut self) {
        let found = discover_aggressiveness(
            &self.algo_names_ra,
            &self.algo_names_dec,
            &self.settings.aggressiveness_param_ra,
            &self.settings.aggressiveness_param_dec,
        );
        self.snap.aggressiveness_param_ra = found.ra;
        self.snap.aggressiveness_param_dec = found.dec;
        self.snap.actuation_disabled_reason = found.disabled_reason;
    }

    fn capabilities(&self, report: &ConnectReport) -> Value {
        json!({
            "driver": DriverId::Phd2.as_str(),
            "phd2_version": self.snap.phd2_version,
            "msg_version": self.msg_version,
            "profile": self.snap.profile,
            "equipment_connected": self.equipment_connected,
            "camera": self.snap.camera_name,
            "mount": self.snap.mount_name,
            "pixel_scale_arcsec": self.snap.pixel_scale_arcsec,
            "calibrated": self.snap.calibrated,
            "dec_guide_mode": self.snap.dec_guide_mode,
            "exposure_ms": self.snap.exposure_ms,
            "algo_param_names": {"ra": self.algo_names_ra, "dec": self.algo_names_dec},
            "aggressiveness_param_ra": self.snap.aggressiveness_param_ra,
            "aggressiveness_param_dec": self.snap.aggressiveness_param_dec,
            "actuation_disabled_reason": self.snap.actuation_disabled_reason,
            // The M-1 actuator is Tier 2 (WP2-7).
            "set_algo_param": false,
            "unsupported_methods": report.unsupported_members,
            "qualification": Qualification::ImplementedUnqualified.as_str(),
        })
    }
}

/// Reads PHD2's session context, tolerating every individual failure.
struct ContextReader<'a> {
    client: &'a Phd2Client,
    unsupported: &'a mut BTreeSet<String>,
    warnings: Vec<String>,
    first_error: Option<DeviceError>,
    gave_up: bool,
}

impl ContextReader<'_> {
    async fn read(&mut self, name: &str, params: Option<Value>) -> Option<Value> {
        if self.gave_up {
            return None;
        }
        match self.client.call(name, params).await {
            Ok(value) => Some(value),
            Err(err) if err.class == ErrorClass::Unsupported => {
                self.unsupported.insert(name.to_string());
                None
            }
            Err(err) => {
                if matches!(err.class, ErrorClass::Timeout | ErrorClass::Disconnected) {
                    // PHD2 stopped answering: do not wait out every read.
                    self.gave_up = true;
                }
                self.warnings.push(format!("{name}: {err}"));
                self.first_error.get_or_insert(err);
                None
            }
        }
    }
}

/// G-a step 3 (also `Refresh`, `ConfigurationChange` and calibration
/// changes). Returns the warnings and the first real failure.
async fn read_context(client: &Phd2Client, core: &mut Core) -> (Vec<String>, Option<DeviceError>) {
    let mut reader = ContextReader {
        client,
        unsupported: &mut core.unsupported,
        warnings: Vec::new(),
        first_error: None,
        gave_up: false,
    };
    if let Some(v) = reader.read(method::GET_PROFILE, None).await {
        core.snap.profile = proto::parse_profile_name(&v);
    }
    if let Some(v) = reader.read(method::GET_CURRENT_EQUIPMENT, None).await {
        let (camera, mount) = proto::parse_current_equipment(&v);
        core.snap.camera_name = camera;
        core.snap.mount_name = mount;
    }
    if let Some(v) = reader.read(method::GET_PIXEL_SCALE, None).await {
        core.snap.pixel_scale_arcsec = proto::parse_pixel_scale(&v);
    }
    if let Some(v) = reader.read(method::GET_CALIBRATED, None).await {
        core.snap.calibrated = proto::parse_bool(&v);
    }
    let calibrated = core.snap.calibrated == Some(true);
    let warnings_before = reader.warnings.len();
    match reader
        .read(method::GET_CALIBRATION_DATA, Some(json!(["Mount"])))
        .await
    {
        Some(v) => core.snap.calibration = (!v.is_null()).then_some(v),
        None => {
            core.snap.calibration = None;
            // An uncalibrated PHD2 refuses this read; that is not a problem.
            if !calibrated && reader.warnings.len() > warnings_before && !reader.gave_up {
                reader.warnings.pop();
                reader.first_error = None;
            }
        }
    }
    if let Some(v) = reader.read(method::GET_DEC_GUIDE_MODE, None).await {
        core.snap.dec_guide_mode = proto::parse_string(&v);
    }
    if let Some(v) = reader.read(method::GET_EXPOSURE, None).await {
        core.snap.exposure_ms = proto::parse_exposure_ms(&v);
    }
    let (params, names_ra, names_dec) = read_algo(&mut reader).await;
    let (warnings, first_error) = (reader.warnings, reader.first_error);
    if let Some(names) = names_ra {
        core.algo_names_ra = names;
    }
    if let Some(names) = names_dec {
        core.algo_names_dec = names;
    }
    if let Some(params) = params {
        core.snap.algo_params = params;
    }
    core.apply_discovery();
    (warnings, first_error)
}

type AlgoRead = (
    Option<Vec<AlgoParam>>,
    Option<Vec<String>>,
    Option<Vec<String>>,
);

/// `get_algo_param_names` per axis, then every `get_algo_param` (G-a, G-j).
async fn read_algo(reader: &mut ContextReader<'_>) -> AlgoRead {
    let mut names = [None, None];
    for (slot, axis) in names.iter_mut().zip([Axis::Ra, Axis::Dec]) {
        *slot = reader
            .read(
                method::GET_ALGO_PARAM_NAMES,
                Some(proto::algo_param_names_params(axis)),
            )
            .await
            .map(|v| proto::parse_string_list(&v));
    }
    if names.iter().all(Option::is_none) {
        return (None, None, None);
    }
    let mut params = Vec::new();
    for (axis, axis_names) in [(Axis::Ra, &names[0]), (Axis::Dec, &names[1])] {
        for name in axis_names.iter().flatten() {
            let value = reader
                .read(
                    method::GET_ALGO_PARAM,
                    Some(proto::algo_param_params(axis, name)),
                )
                .await;
            if let Some(value) = value.as_ref().and_then(proto::parse_f64) {
                params.push(AlgoParam {
                    axis,
                    name: name.clone(),
                    value,
                });
            }
        }
    }
    let [ra, dec] = names;
    (Some(params), ra, dec)
}

/// Re-read only the algorithm parameters (after `GuideParamChange`).
async fn read_algo_only(client: &Phd2Client, core: &mut Core) {
    let mut reader = ContextReader {
        client,
        unsupported: &mut core.unsupported,
        warnings: Vec::new(),
        first_error: None,
        gave_up: false,
    };
    let (params, names_ra, names_dec) = read_algo(&mut reader).await;
    for warning in reader.warnings {
        tracing::debug!(%warning, "PHD2 algorithm parameter re-read");
    }
    if let Some(names) = names_ra {
        core.algo_names_ra = names;
    }
    if let Some(names) = names_dec {
        core.algo_names_dec = names;
    }
    if let Some(params) = params {
        core.snap.algo_params = params;
    }
    core.apply_discovery();
}

// ---------------------------------------------------------------------------
// Connect (G-a)
// ---------------------------------------------------------------------------

/// Connect to PHD2 with a new `connection_id`. See [`connect_with_id`].
pub async fn connect(settings: Phd2Settings, ctx: GuiderContext) -> DeviceResult<Phd2Connection> {
    connect_with_id(settings, ctx, Uuid::new_v4()).await
}

/// Connect to PHD2 and run the G-a sequence. Must run inside a tokio
/// runtime. Never moves hardware; never switches PHD2's profile; connects
/// PHD2's equipment only when `connect_equipment` is set.
///
/// `connection_id` is the caller's, so the hub can register it before any
/// event is emitted (§2.8). Nothing is emitted or recorded and the service
/// does not run until [`Phd2Connection::start`] (§2.6 steps 9-10).
pub async fn connect_with_id(
    settings: Phd2Settings,
    ctx: GuiderContext,
    connection_id: Uuid,
) -> DeviceResult<Phd2Connection> {
    let started_mono = ctx.clock.now_mono_ns();
    let (client, mut events_rx, tasks) = Phd2Client::connect(
        &settings.host,
        settings.port,
        settings.connect_timeout,
        settings.rpc_timeout,
        Arc::clone(&ctx.clock),
    )
    .await?;
    let mut core = Core::new(
        settings.clone(),
        Arc::clone(&ctx.clock),
        Arc::clone(&ctx.recorder),
        connection_id,
    );
    let mut report = ConnectReport {
        path: ConnectPath::Phd2,
        ..ConnectReport::default()
    };
    if let Err(err) = startup(&client, &mut events_rx, &mut core, &mut report).await {
        client.close();
        tasks.abort();
        return Err(err);
    }
    report.interface_version = core.msg_version.and_then(|v| i32::try_from(v).ok());
    report.unsupported_members = core.unsupported.iter().cloned().collect();
    report.duration_ms = ctx.clock.now_mono_ns().saturating_sub(started_mono) / 1_000_000;

    let identity = build_identity(&settings, &core, connection_id);
    let capabilities = core.capabilities(&report);
    let now = ctx.clock.now_utc();
    let now_mono = ctx.clock.now_mono_ns();
    core.snap.published_at = now;
    core.snap.published_mono_ns = now_mono;
    // The abort path connects first to the address this connection reached
    // (`localhost` may resolve to `::1` first; PHD2 listens on IPv4 only).
    let abort_path: Arc<dyn AbortPath> = Arc::new(
        Phd2AbortPath::new(settings.host.clone(), settings.port, settings.abort_timeout)
            .with_peer_addr(client.peer_addr()),
    );
    let (handle, end) = DeviceHandle::new_pair(
        identity.clone(),
        Some(abort_path),
        core.snap.clone(),
        HealthState::ready(now, now_mono),
    );
    let windows = GuiderWindows {
        history: Arc::clone(&core.history),
        snapshot: handle.watch_snapshot(),
        recorder: Arc::clone(&ctx.recorder),
        clock: Arc::clone(&ctx.clock),
        connection_id,
    };
    // The connect burst's events stay in the outbox until start().
    let next_reconcile_ns = now_mono.saturating_add(duration_ns(settings.reconcile_interval));
    let service = Service {
        core,
        client,
        tasks,
        events_rx,
        events_open: true,
        end,
        emitter: Arc::clone(&ctx.events),
        op: None,
        pending_initiator: false,
        lost: false,
        closed_recorded: false,
        finished: false,
        next_reconcile_ns,
        failures: 0,
    };
    Ok(Phd2Connection {
        handle,
        windows,
        report,
        capabilities,
        identity,
        connected_at: now,
        recorder: Arc::clone(&ctx.recorder),
        pending: Mutex::new(Some(service)),
        task: None,
    })
}

async fn startup(
    client: &Phd2Client,
    events_rx: &mut mpsc::UnboundedReceiver<ClientEvent>,
    core: &mut Core,
    report: &mut ConnectReport,
) -> DeviceResult<()> {
    let clock = Arc::clone(&core.clock);
    let deadline = clock
        .now_mono_ns()
        .saturating_add(duration_ns(core.settings.rpc_timeout));
    let (mut seen_version, mut seen_state) = (false, false);
    while !(seen_version && seen_state) {
        match until_deadline(events_rx.recv(), clock.as_ref(), deadline).await {
            Some(Some(ClientEvent::Event {
                event,
                received_at,
                received_mono_ns,
            })) => {
                seen_version |= matches!(event.body, EventBody::Version(_));
                seen_state |= matches!(event.body, EventBody::AppState { .. });
                core.on_event(&event, received_at, received_mono_ns);
            }
            Some(Some(ClientEvent::Malformed { line, error })) => {
                core.malformed += 1;
                tracing::debug!(%line, %error, "malformed PHD2 line during connect");
            }
            Some(Some(ClientEvent::Closed { reason, .. })) => {
                return Err(DeviceError::disconnected(format!(
                    "PHD2 closed the connection during the connect sequence: {reason}"
                )))
            }
            Some(None) => {
                return Err(DeviceError::disconnected(
                    "PHD2 closed the connection during the connect sequence",
                ))
            }
            None => break,
        }
    }
    if !seen_version {
        report
            .warnings
            .push("PHD2 sent no Version event after connecting".into());
    }
    if !seen_state {
        report
            .warnings
            .push("PHD2 sent no AppState event; read it with get_app_state".into());
        let value = client.call(method::GET_APP_STATE, None).await?;
        if let Some(text) = proto::parse_app_state(&value) {
            let at = clock.now_utc();
            core.set_state(
                GuiderAppState::from_phd2(&text),
                Stamp {
                    at,
                    mono: clock.now_mono_ns(),
                    device_time: None,
                    window_at: at,
                    latency_ms: None,
                    source: Source::Property,
                },
            );
        }
    }
    // A PHD2 that does not answer this is not usable: fail the connect.
    let connected = proto::parse_bool(&client.call(method::GET_CONNECTED, None).await?);
    core.equipment_connected = connected;
    if connected != Some(true) {
        if core.settings.connect_equipment {
            match client
                .call(method::SET_CONNECTED, Some(json!([true])))
                .await
            {
                Ok(_) => {
                    let again = client.call(method::GET_CONNECTED, None).await;
                    core.equipment_connected = again.ok().as_ref().and_then(proto::parse_bool);
                    if core.equipment_connected != Some(true) {
                        report
                            .warnings
                            .push("PHD2 still reports its equipment disconnected".into());
                    }
                }
                Err(err) => report
                    .warnings
                    .push(format!("set_connected [true] failed: {err}")),
            }
        } else {
            report.warnings.push(
                "PHD2's equipment is not connected; Achernar connects it only when [guider] \
                 connect_equipment = true"
                    .into(),
            );
        }
    }
    let (warnings, _) = read_context(client, core).await;
    report.warnings.extend(warnings);
    Ok(())
}

fn build_identity(settings: &Phd2Settings, core: &Core, connection_id: Uuid) -> DeviceIdentity {
    let key = identity_key::phd2(&settings.host, settings.port, core.snap.profile.as_deref());
    let name = match core.snap.profile.as_deref() {
        Some(profile) => format!("PHD2 ({profile})"),
        None => "PHD2".to_string(),
    };
    let mut identity = DeviceIdentity::new(
        DeviceKind::Guider,
        DriverId::Phd2,
        settings.endpoint(),
        key,
        name,
    );
    identity.connection_id = connection_id;
    identity.description = Some("PHD2 guiding through its event server".into());
    identity.driver_info = Some("PHD2 event server (line-delimited JSON-RPC over TCP)".into());
    identity.driver_version = core.snap.phd2_version.clone();
    identity.interface_version = core.msg_version.and_then(|v| i32::try_from(v).ok());
    identity.qualification = Qualification::ImplementedUnqualified;
    identity
}

// ---------------------------------------------------------------------------
// The service task
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Target {
    /// Stopped or Selected (not capturing).
    Stopped,
    Paused,
    NotPaused,
}

impl Target {
    fn reached(self, state: GuiderAppState) -> bool {
        match self {
            Target::Stopped => matches!(state, GuiderAppState::Stopped | GuiderAppState::Selected),
            Target::Paused => state == GuiderAppState::Paused,
            Target::NotPaused => !matches!(state, GuiderAppState::Paused | GuiderAppState::Unknown),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// G-f step 1: a settle is in progress; PHD2 rejects re-entrant settles.
    WaitSettleClear,
    /// G-c / G-f: waiting for `SettleDone`.
    AwaitSettleDone,
    /// G-d / G-e: waiting for a state, polling `get_app_state`.
    AwaitState(Target),
}

struct GuiderOp {
    command_id: CommandId,
    kind: OpKind,
    cmd: GuiderCmd,
    started_at: DateTime<Utc>,
    started_mono_ns: u64,
    deadline_ns: u64,
    next_poll_ns: u64,
    phase: OpPhase,
    stage: Stage,
    /// The settle actually sent (whole seconds, §phd2_proto::sent_settle).
    settle: Settle,
    dither: Option<DitherRecord>,
    /// Set by `Cancel`: the op now waits for PHD2 to report Stopped and
    /// resolves `Aborted { by }` (or `Unknown` if PHD2 never stops).
    aborted_by: Option<CommandId>,
    done: Option<oneshot::Sender<OpOutcome>>,
    /// Held for the operation's lifetime; dropping it releases ownership.
    _permit: Permit,
}

impl GuiderOp {
    fn active(&self) -> ActiveOp {
        ActiveOp {
            command_id: self.command_id,
            kind: self.kind,
            phase: self.phase,
            started_at: self.started_at,
            started_mono_ns: self.started_mono_ns,
            deadline_mono_ns: self.deadline_ns,
        }
    }
}

enum Wake {
    Command(Box<Envelope<GuiderCmd>>),
    QueueClosed,
    Client(Option<ClientEvent>),
    Tick,
}

struct Service {
    core: Core,
    client: Phd2Client,
    tasks: ClientTasks,
    events_rx: mpsc::UnboundedReceiver<ClientEvent>,
    events_open: bool,
    end: WorkerEnd<GuiderCmd, GuiderSnapshot>,
    emitter: EventEmitter,
    op: Option<GuiderOp>,
    pending_initiator: bool,
    lost: bool,
    closed_recorded: bool,
    finished: bool,
    next_reconcile_ns: u64,
    failures: u32,
}

impl Drop for Service {
    /// However the task ends (disconnect, a closed queue, or an abort of
    /// [`Phd2Connection::task`]), the socket closes and waiters fail fast.
    /// PHD2 itself is left as it is (D16).
    fn drop(&mut self) {
        self.client.close();
        self.tasks.abort();
    }
}

impl Service {
    async fn run(mut self) {
        let mut ticker = tokio::time::interval(TICK);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
        while !self.finished {
            let wake = tokio::select! {
                biased;
                envelope = self.end.commands.pop() => match envelope {
                    Some(envelope) => Wake::Command(Box::new(envelope)),
                    None => Wake::QueueClosed,
                },
                event = self.events_rx.recv(), if self.events_open => Wake::Client(event),
                _ = ticker.tick() => Wake::Tick,
            };
            let changed = match wake {
                Wake::Command(envelope) => {
                    self.on_envelope(*envelope).await;
                    true
                }
                Wake::QueueClosed => {
                    self.finished = true;
                    false
                }
                Wake::Client(event) => {
                    self.on_client_event(event);
                    true
                }
                Wake::Tick => self.on_tick().await,
            };
            self.flush();
            if changed && !self.end.is_retired() {
                self.publish();
            }
        }
        self.client.close();
        self.tasks.abort();
    }

    fn now(&self) -> (DateTime<Utc>, u64) {
        (self.core.clock.now_utc(), self.core.clock.now_mono_ns())
    }

    fn flush(&mut self) {
        for event in self.core.outbox.drain(..) {
            (self.emitter)(event);
        }
    }

    fn publish(&mut self) {
        let (at, mono) = self.now();
        self.core.snap.published_at = at;
        self.core.snap.published_mono_ns = mono;
        self.core.snap.active_op = self.op.as_ref().map(GuiderOp::active);
        self.end.publish(self.core.snap.clone());
    }

    fn lost_error(&self) -> DeviceError {
        DeviceError::disconnected(
            "the PHD2 connection was lost; disconnect and reconnect the guider",
        )
    }

    // ---- commands ----

    async fn on_envelope(&mut self, envelope: Envelope<GuiderCmd>) {
        let command_id = envelope.command_id;
        match envelope.body {
            EnvelopeBody::Command {
                cmd,
                permit,
                accepted,
                done,
            } => {
                self.on_command(command_id, cmd, permit, accepted, done)
                    .await
            }
            EnvelopeBody::Cancel { by } => self.on_cancel(by).await,
            EnvelopeBody::Disconnect { done, .. } => self.on_disconnect(done),
        }
    }

    async fn on_command(
        &mut self,
        command_id: CommandId,
        cmd: GuiderCmd,
        permit: Permit,
        accepted: oneshot::Sender<DeviceResult<()>>,
        done: oneshot::Sender<OpOutcome>,
    ) {
        if self.lost {
            let _ = accepted.send(Err(self.lost_error()));
            return;
        }
        let kind = cmd.op_kind();
        let source = permit.source().clone();
        if matches!(cmd, GuiderCmd::Refresh) {
            // A re-read: no hardware, no operation slot.
            let _ = accepted.send(Ok(()));
            self.core.emit(
                EventKind::OperationCommanded { op: kind, source },
                Some(command_id),
                None,
            );
            let (_, started) = self.now();
            let (warnings, error) = read_context(&self.client, &mut self.core).await;
            for warning in warnings {
                tracing::debug!(%warning, "PHD2 refresh");
            }
            let outcome = error.map_or(OpOutcome::Succeeded, OpOutcome::Failed);
            let duration_ms = self.core.clock.now_mono_ns().saturating_sub(started) / 1_000_000;
            self.core.emit(
                EventKind::OperationEnded {
                    op: kind,
                    outcome: outcome.clone(),
                    duration_ms,
                },
                Some(command_id),
                None,
            );
            let _ = done.send(outcome);
            return;
        }
        if matches!(cmd, GuiderCmd::SetAlgoParam { .. }) {
            let _ = accepted.send(Err(DeviceError::unsupported(
                "set_algo_param (the M-1 actuator) is Tier 2 (WP2-7); Tier 1 reads algorithm \
                 parameters only",
            )));
            return;
        }
        if self.op.is_some() {
            let _ = accepted.send(Err(DeviceError::invalid_operation(
                "another guider operation is active",
            )));
            return;
        }
        if let Err(err) = self.validate(&cmd) {
            let _ = accepted.send(Err(err));
            return;
        }
        let _ = accepted.send(Ok(()));
        let (now, now_mono) = self.now();
        self.core.expire_settle_tags(now_mono);
        let settle_in_progress = self.core.settle_in_progress(now_mono);
        let settings = &self.core.settings;
        let settle = match &cmd {
            GuiderCmd::Start { settle, .. } | GuiderCmd::Dither { settle, .. } => {
                proto::sent_settle(settle)
            }
            _ => proto::sent_settle(&settings.default_settle),
        };
        let waits_for_settle =
            matches!(kind, OpKind::GuiderStart | OpKind::Dither) && settle_in_progress;
        let budget = match kind {
            OpKind::GuiderStart | OpKind::Dither if waits_for_settle => {
                default_settle_bound(settings)
            }
            OpKind::GuiderStart | OpKind::Dither => secs(settle.timeout_s) + settings.settle_margin,
            OpKind::GuiderStop => settings.stop_timeout,
            _ => settings.pause_timeout,
        };
        let dither = match &cmd {
            GuiderCmd::Dither {
                amount,
                unit,
                ra_only,
                ..
            } => Some(DitherRecord {
                id: Uuid::new_v4(),
                command_id: Some(command_id),
                policy_version: match &source {
                    CommandSource::Policy { version, .. } => Some(version.clone()),
                    _ => None,
                },
                source: source.clone(),
                backend: DitherBackend::Phd2,
                amount: *amount,
                amount_unit: *unit,
                ra_only: *ra_only,
                settle,
                requested_at: now,
                dx: None,
                dy: None,
                settle_done_at: None,
                settle_status: None,
                settle_error: None,
                total_frames: None,
                dropped_frames: None,
                max_distance_px: None,
                next_frame_id: None,
                next_frame_accepted: None,
            }),
            _ => None,
        };
        self.op = Some(GuiderOp {
            command_id,
            kind,
            cmd,
            started_at: now,
            started_mono_ns: now_mono,
            deadline_ns: now_mono.saturating_add(duration_ns(budget)),
            next_poll_ns: now_mono,
            phase: OpPhase::Starting,
            stage: if waits_for_settle {
                Stage::WaitSettleClear
            } else {
                Stage::AwaitSettleDone
            },
            settle,
            dither,
            aborted_by: None,
            done: Some(done),
            _permit: permit,
        });
        self.core.emit(
            EventKind::OperationCommanded { op: kind, source },
            Some(command_id),
            None,
        );
        if !waits_for_settle {
            self.send_initiator().await;
        }
    }

    fn validate(&self, cmd: &GuiderCmd) -> DeviceResult<()> {
        fn check_settle(settle: &Settle) -> DeviceResult<()> {
            let ok = settle.pixels.is_finite()
                && settle.pixels > 0.0
                && settle.time_s.is_finite()
                && settle.time_s >= 0.0
                && settle.timeout_s.is_finite()
                && settle.timeout_s > 0.0;
            if ok {
                Ok(())
            } else {
                Err(DeviceError::invalid_value(format!(
                    "settle needs pixels > 0, time >= 0 and timeout > 0; got {settle:?}"
                )))
            }
        }
        match cmd {
            GuiderCmd::Start { settle, .. } => check_settle(settle),
            GuiderCmd::Dither {
                amount,
                unit,
                settle,
                ..
            } => {
                if *unit != PixelUnit::GuidePx {
                    return Err(DeviceError::invalid_value(
                        "PHD2 dithers in guide-camera pixels (guide_px); convert a main_px \
                         amount with the rig's image scales first",
                    ));
                }
                if !(amount.is_finite() && *amount > 0.0) {
                    return Err(DeviceError::invalid_value(format!(
                        "dither amount must be a positive number of guide pixels; got {amount}"
                    )));
                }
                check_settle(settle)?;
                let state = self.core.state();
                if state != GuiderAppState::Guiding {
                    return Err(DeviceError::invalid_operation(format!(
                        "dither needs PHD2 to be guiding; PHD2 is {state:?}"
                    )));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Send the operation's initiator RPC (C-4).
    async fn send_initiator(&mut self) {
        let Some(op) = self.op.as_ref() else { return };
        let settle = op.settle;
        let kind = op.kind;
        let (name, params) = match &op.cmd {
            GuiderCmd::Start {
                settle,
                recalibrate,
            } => (
                method::GUIDE,
                Some(proto::guide_params(settle, *recalibrate)),
            ),
            GuiderCmd::Dither {
                amount,
                ra_only,
                settle,
                ..
            } => (
                method::DITHER,
                Some(proto::dither_params(*amount, *ra_only, settle)),
            ),
            GuiderCmd::Stop => (method::STOP_CAPTURE, None),
            GuiderCmd::Pause { full } => (
                method::SET_PAUSED,
                Some(proto::set_paused_params(true, *full)),
            ),
            GuiderCmd::Resume => (
                method::SET_PAUSED,
                Some(proto::set_paused_params(false, false)),
            ),
            GuiderCmd::SetAlgoParam { .. } | GuiderCmd::Refresh => return,
        };
        let settle_bound = secs(settle.timeout_s) + self.core.settings.settle_margin;
        let sent_mono = self.core.clock.now_mono_ns();
        match kind {
            OpKind::Dither => {
                self.core
                    .tracker
                    .on_dither_sent(settle.pixels, settle_bound, sent_mono)
            }
            OpKind::GuiderStart => self.core.tracker.on_guide_sent(settle.pixels, settle_bound),
            _ => {}
        }
        let result = self.client.call(name, params).await;
        self.note_rpc(&result);
        match result {
            Ok(_) => {
                let (_, now) = self.now();
                let margin = self.core.settings.settle_margin;
                let Some(op) = self.op.as_mut() else { return };
                op.stage = match op.kind {
                    OpKind::GuiderStop => Stage::AwaitState(Target::Stopped),
                    OpKind::GuiderPause => Stage::AwaitState(Target::Paused),
                    OpKind::GuiderResume => Stage::AwaitState(Target::NotPaused),
                    _ => Stage::AwaitSettleDone,
                };
                op.phase = match op.stage {
                    Stage::AwaitSettleDone => OpPhase::InProgress,
                    _ => OpPhase::Verifying,
                };
                if matches!(op.kind, OpKind::GuiderStart | OpKind::Dither) {
                    op.deadline_ns =
                        now.saturating_add(duration_ns(secs(settle.timeout_s) + margin));
                }
                op.next_poll_ns = now;
                let (command_id, kind) = (op.command_id, op.kind);
                self.core.emit(
                    EventKind::OperationStarted { op: kind },
                    Some(command_id),
                    None,
                );
                self.check_state_target();
            }
            Err(err) => {
                // A refusal is an answer: PHD2 did nothing. A request that got
                // no answer (timeout, or the socket dropped after sending) may
                // have been carried out, so its outcome is unknown (C-6).
                let unanswered = matches!(
                    err.class,
                    ErrorClass::Timeout | ErrorClass::Disconnected | ErrorClass::Transport
                );
                if kind == OpKind::Dither && !unanswered {
                    self.core.tracker.on_settle_done();
                }
                if let Some(record) = self.op.as_mut().and_then(|op| op.dither.as_mut()) {
                    record.settle_error = Some(err.message.clone());
                }
                self.finish(if unanswered {
                    OpOutcome::Unknown(err)
                } else {
                    OpOutcome::Failed(err)
                });
            }
        }
    }

    /// `Cancel` (abort, e-stop or preemption; the WP0 contract): the
    /// operation resolves `Aborted { by }` only once PHD2 reports Stopped.
    /// The abort path already sent `stop_capture` out of band (D4); it is
    /// sent in band as well, because the out-of-band connection can fail
    /// (D19: always attempted). PHD2 that does not stop within the G-d stop
    /// timeout leaves the outcome unknown.
    async fn on_cancel(&mut self, by: CommandId) {
        if self.lost {
            // The loss already resolved any operation; nothing to send on.
            return;
        }
        // A settle that ended before the cancel must not start the op now.
        self.pending_initiator = false;
        if let Some(op) = self.op.as_mut() {
            op.aborted_by = Some(by);
            op.stage = Stage::AwaitState(Target::Stopped);
            op.phase = OpPhase::Cancelling;
        }
        let result = self.client.call(method::STOP_CAPTURE, None).await;
        self.note_rpc(&result);
        if let Err(err) = &result {
            tracing::warn!(
                connection_id = %self.core.connection_id,
                error = %err,
                "in-band stop_capture after an abort failed; waiting for PHD2 to report Stopped"
            );
        }
        let (_, now) = self.now();
        let stop_timeout = self.core.settings.stop_timeout;
        if let Some(op) = self.op.as_mut() {
            op.deadline_ns = now.saturating_add(duration_ns(stop_timeout));
            op.next_poll_ns = now;
        }
        self.check_state_target();
    }

    fn on_disconnect(&mut self, done: oneshot::Sender<DeviceResult<()>>) {
        if self.op.is_some() {
            self.finish(OpOutcome::Unknown(DeviceError::disconnected(
                "the guider was disconnected during the operation",
            )));
        }
        // D16: never `set_connected [false]`; PHD2 keeps guiding.
        self.client.close();
        self.tasks.abort();
        let (now, now_mono) = self.now();
        if !self.closed_recorded {
            self.core
                .recorder
                .record(GroundTruthRecord::ConnectionClosed {
                    connection_id: self.core.connection_id,
                    disconnected_at: now,
                    end_class: ConnectionEnd::Operator,
                });
            self.closed_recorded = true;
        }
        let end_class = if self.lost {
            ConnectionEnd::Lost
        } else {
            ConnectionEnd::Operator
        };
        self.core.emit(
            EventKind::DeviceDisconnected {
                end_class,
                reason: None,
            },
            None,
            None,
        );
        self.flush();
        self.end.retire(now_mono);
        self.events_open = false;
        self.finished = true;
        let _ = done.send(Ok(()));
    }

    /// Resolve the active operation (C-6) and record its dither row (H10).
    fn finish(&mut self, outcome: OpOutcome) {
        let Some(mut op) = self.op.take() else { return };
        let (_, now) = self.now();
        let duration_ms = now.saturating_sub(op.started_mono_ns) / 1_000_000;
        if let Some(mut record) = op.dither.take() {
            if record.settle_error.is_none() && record.settle_status.is_none() {
                record.settle_error = match &outcome {
                    OpOutcome::Succeeded => None,
                    OpOutcome::Failed(err) | OpOutcome::Unknown(err) => Some(err.message.clone()),
                    OpOutcome::TimedOut => Some(format!(
                        "no SettleDone within the settle timeout plus {} s",
                        self.core.settings.settle_margin.as_secs()
                    )),
                    OpOutcome::Aborted { .. } => Some("aborted".into()),
                };
            }
            self.core
                .recorder
                .record(GroundTruthRecord::Dither(Box::new(record)));
        }
        self.core.emit(
            EventKind::OperationEnded {
                op: op.kind,
                outcome: outcome.clone(),
                duration_ms,
            },
            Some(op.command_id),
            None,
        );
        if let OpOutcome::Unknown(err) = &outcome {
            self.core.emit(
                EventKind::OutcomeUnknown {
                    op: op.kind,
                    reason: err.message.clone(),
                },
                Some(op.command_id),
                None,
            );
        }
        if let Some(done) = op.done.take() {
            let _ = done.send(outcome);
        }
        // `op` (and its permit) drops here, releasing ownership.
    }

    fn check_state_target(&mut self) {
        let reached = match self.op.as_ref().map(|op| op.stage) {
            Some(Stage::AwaitState(target)) => target.reached(self.core.state()),
            _ => false,
        };
        if reached {
            let outcome = match self.op.as_ref().and_then(|op| op.aborted_by) {
                Some(by) => OpOutcome::Aborted { by: Some(by) },
                None => OpOutcome::Succeeded,
            };
            self.finish(outcome);
        }
    }

    // ---- PHD2 events ----

    fn on_client_event(&mut self, event: Option<ClientEvent>) {
        match event {
            Some(ClientEvent::Event {
                event,
                received_at,
                received_mono_ns,
            }) => {
                self.core.on_event(&event, received_at, received_mono_ns);
                self.op_on_event(&event, received_at);
                self.check_state_target();
            }
            Some(ClientEvent::Malformed { line, error }) => {
                self.core.malformed += 1;
                tracing::debug!(%line, %error, "malformed PHD2 line");
            }
            Some(ClientEvent::Closed { reason, local }) => {
                self.events_open = false;
                if !local && !self.lost {
                    self.on_connection_lost(reason);
                }
            }
            None => {
                self.events_open = false;
                if !self.lost && !self.finished {
                    self.on_connection_lost("the PHD2 reader stopped".into());
                }
            }
        }
    }

    fn op_on_event(&mut self, event: &Phd2Event, received_at: DateTime<Utc>) {
        let now = self.core.clock.now_mono_ns();
        let margin = self.core.settings.settle_margin;
        let Some(op) = self.op.as_mut() else { return };
        let mut outcome = None;
        match (op.stage, &event.body) {
            (Stage::WaitSettleClear, EventBody::SettleDone(_)) => self.pending_initiator = true,
            // G-c with recalibrate: PHD2 calibrates before it settles, which
            // can take minutes. The settle timeout restarts with each sign of
            // calibration progress and once guiding starts.
            (
                Stage::AwaitSettleDone,
                EventBody::StartCalibration { .. }
                | EventBody::Calibrating(_)
                | EventBody::CalibrationComplete { .. }
                | EventBody::StartGuiding,
            ) if op.kind == OpKind::GuiderStart => {
                op.deadline_ns =
                    now.saturating_add(duration_ns(secs(op.settle.timeout_s) + margin));
            }
            (Stage::AwaitSettleDone, EventBody::CalibrationFailed { reason })
                if op.kind == OpKind::GuiderStart =>
            {
                outcome = Some(OpOutcome::Failed(DeviceError::driver(format!(
                    "PHD2 calibration failed: {}",
                    reason.as_deref().unwrap_or("no reason given")
                ))));
            }
            (Stage::AwaitSettleDone, EventBody::GuidingDithered { dx, dy }) => {
                if let Some(record) = op.dither.as_mut() {
                    record.dx = *dx;
                    record.dy = *dy;
                }
            }
            (Stage::AwaitSettleDone, EventBody::Settling(settling)) => {
                op.phase = OpPhase::Settling;
                if let (Some(record), Some(distance)) = (op.dither.as_mut(), settling.distance) {
                    record.max_distance_px =
                        Some(record.max_distance_px.map_or(distance, |m| m.max(distance)));
                }
            }
            (Stage::AwaitSettleDone, EventBody::SettleDone(done)) => {
                if let Some(record) = op.dither.as_mut() {
                    record.settle_done_at = Some(received_at);
                    record.settle_status = Some(done.status_code());
                    record.settle_error = done.error.clone().filter(|e| !e.trim().is_empty());
                    record.total_frames = done.total_frames.and_then(|n| u32::try_from(n).ok());
                    record.dropped_frames = done.dropped_frames.and_then(|n| u32::try_from(n).ok());
                }
                outcome = Some(settle_outcome(done));
            }
            _ => {}
        }
        if let Some(outcome) = outcome {
            self.finish(outcome);
        }
    }

    fn on_connection_lost(&mut self, reason: String) {
        self.lost = true;
        if self.op.is_some() {
            self.finish(OpOutcome::Unknown(DeviceError::disconnected(format!(
                "the PHD2 connection was lost during the operation: {reason}"
            ))));
        }
        let (now, now_mono) = self.now();
        self.end.update_health(|health| {
            health.state = LinkState::Lost;
            health.failure_class = Some(FailureClass::Disconnected);
            health.reason = Some(reason.clone());
            health.changed_mono_ns = now_mono;
        });
        self.core.emit(
            EventKind::DeviceLost {
                class: FailureClass::Disconnected,
                reason: Some(reason),
            },
            None,
            None,
        );
        self.core.snap.app_state = self.core.snap.app_state.clone().into_stale();
        self.core.snap.settling = false;
        self.core.snap.settle = None;
        if !self.closed_recorded {
            self.core
                .recorder
                .record(GroundTruthRecord::ConnectionClosed {
                    connection_id: self.core.connection_id,
                    disconnected_at: now,
                    end_class: ConnectionEnd::Lost,
                });
            self.closed_recorded = true;
        }
        self.tasks.abort();
    }

    // ---- timers ----

    async fn on_tick(&mut self) -> bool {
        if self.lost {
            return false;
        }
        let mut changed = false;
        let (_, now) = self.now();
        if self.core.expire_settle_tags(now) {
            changed = true;
        }
        // G-f step 1: go ahead once the settle waited for is over, whether
        // its SettleDone arrived or its tag lapsed (a dither PHD2 never
        // settles).
        let waiting = self
            .op
            .as_ref()
            .is_some_and(|op| op.stage == Stage::WaitSettleClear);
        if waiting && (self.pending_initiator || !self.core.settle_in_progress(now)) {
            self.pending_initiator = false;
            self.send_initiator().await;
            changed = true;
        }
        self.pending_initiator = false;
        let (_, now) = self.now();
        if let Some((stage, deadline, next_poll, aborted)) = self
            .op
            .as_ref()
            .map(|op| (op.stage, op.deadline_ns, op.next_poll_ns, op.aborted_by))
        {
            if now >= deadline {
                changed = true;
                if aborted.is_some() {
                    let waited = self.core.settings.stop_timeout;
                    self.finish(OpOutcome::Unknown(DeviceError::outcome_unknown(format!(
                        "PHD2 did not report Stopped within {waited:?} of the abort; it may \
                         still be guiding"
                    ))));
                } else if stage == Stage::WaitSettleClear {
                    // The settle we waited for never reported done; go ahead
                    // (PHD2 refuses the request if it is still settling).
                    self.core.tracker.on_settle_done();
                    self.core.snap.settling = false;
                    self.core.snap.settle = None;
                    self.send_initiator().await;
                } else {
                    self.finish(OpOutcome::TimedOut);
                }
            } else if matches!(stage, Stage::AwaitState(_)) && now >= next_poll {
                changed = true;
                self.read_app_state().await;
                let next = self
                    .core
                    .clock
                    .now_mono_ns()
                    .saturating_add(duration_ns(self.core.settings.poll_interval));
                if let Some(op) = self.op.as_mut() {
                    op.next_poll_ns = next;
                }
                self.check_state_target();
            }
        }
        if self.lost {
            return true;
        }
        let (_, now) = self.now();
        if now >= self.next_reconcile_ns {
            changed = true;
            self.read_app_state().await;
            self.next_reconcile_ns = self
                .core
                .clock
                .now_mono_ns()
                .saturating_add(duration_ns(self.core.settings.reconcile_interval));
            self.check_state_target();
        }
        if !self.lost && self.core.refresh_context {
            self.core.refresh_context = false;
            self.core.refresh_algo = false;
            let (warnings, _) = read_context(&self.client, &mut self.core).await;
            for warning in warnings {
                tracing::debug!(%warning, "PHD2 context re-read");
            }
            changed = true;
        } else if !self.lost && self.core.refresh_algo {
            self.core.refresh_algo = false;
            read_algo_only(&self.client, &mut self.core).await;
            changed = true;
        }
        changed
    }

    /// `get_app_state`, the source of truth (G-b).
    async fn read_app_state(&mut self) {
        let (t0, m0) = self.now();
        let result = self.client.call(method::GET_APP_STATE, None).await;
        self.note_rpc(&result);
        let (t1, m1) = self.now();
        match result {
            Ok(value) => {
                if let Some(text) = proto::parse_app_state(&value) {
                    let at = t0 + (t1 - t0) / 2;
                    let stamp = Stamp {
                        at,
                        mono: m0 + m1.saturating_sub(m0) / 2,
                        device_time: None,
                        window_at: at,
                        latency_ms: u32::try_from(m1.saturating_sub(m0) / 1_000_000).ok(),
                        source: Source::Property,
                    };
                    self.core.set_state(GuiderAppState::from_phd2(&text), stamp);
                }
            }
            Err(_) => {
                self.core.snap.app_state = self.core.snap.app_state.clone().into_stale();
            }
        }
    }

    /// Health from request outcomes (§2.8): an unanswered request degrades
    /// the guider; `lost_after_failures` in a row lose it. Any answer,
    /// including a refusal, proves the link.
    fn note_rpc<T>(&mut self, result: &DeviceResult<T>) {
        let (now, now_mono) = self.now();
        let previous = self.end.health.borrow().state;
        match result {
            Err(err) if err.class == ErrorClass::Timeout => {
                self.failures += 1;
                let failures = self.failures;
                let next = if failures >= self.core.settings.lost_after_failures {
                    LinkState::Lost
                } else {
                    LinkState::Degraded
                };
                let reason = err.message.clone();
                self.end.update_health(|health| {
                    health.state = next;
                    health.failure_class = Some(FailureClass::Timeout);
                    health.reason = Some(reason.clone());
                    health.consecutive_failures = failures;
                    if previous != next {
                        health.changed_mono_ns = now_mono;
                    }
                });
                if previous != next {
                    let kind = if next == LinkState::Lost {
                        EventKind::DeviceLost {
                            class: FailureClass::Timeout,
                            reason: Some(reason),
                        }
                    } else {
                        EventKind::DeviceDegraded {
                            class: FailureClass::Timeout,
                            reason: Some(reason),
                        }
                    };
                    self.core.emit(kind, None, None);
                }
            }
            Err(err) if err.class == ErrorClass::Disconnected => {}
            _ => {
                self.failures = 0;
                self.end.update_health(|health| {
                    health.state = LinkState::Ready;
                    health.failure_class = None;
                    health.reason = None;
                    health.consecutive_failures = 0;
                    health.last_ok_at = Some(now);
                    health.last_ok_mono_ns = Some(now_mono);
                    if previous != LinkState::Ready {
                        health.changed_mono_ns = now_mono;
                    }
                });
                if previous == LinkState::Lost {
                    self.core.emit(
                        EventKind::DeviceReconnected {
                            reason: Some("PHD2 answers requests again".into()),
                        },
                        None,
                        None,
                    );
                }
            }
        }
    }
}

/// `SettleDone` → the operation's outcome. A PHD2 settle timeout is a
/// `Timeout`-class failure; other settle failures are `Driver` failures.
fn settle_outcome(done: &SettleDoneEvent) -> OpOutcome {
    if done.succeeded() {
        return OpOutcome::Succeeded;
    }
    let status = done.status_code();
    let message = done
        .error
        .clone()
        .filter(|e| !e.trim().is_empty())
        .unwrap_or_else(|| format!("PHD2 settle failed with status {status}"));
    let lower = message.to_ascii_lowercase();
    let class = if lower.contains("timed-out")
        || lower.contains("timed out")
        || lower.contains("timeout")
    {
        ErrorClass::Timeout
    } else {
        ErrorClass::Driver
    };
    let err = if (100..=599).contains(&status) {
        DeviceError::new(class, message)
    } else {
        DeviceError::with_code(class, status, message)
    };
    OpOutcome::Failed(err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::events::GuideStepRecord;
    use crate::equipment::handle::OpTicket;
    use crate::equipment::snapshot::GuidePhase;
    use crate::equipment::testing::fake_handles::RecordingSink;
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::testing::phd2_mock::{MockState, Phd2Mock};
    use crate::equipment::types::{ObsState, Priority};

    const WAIT: Duration = Duration::from_secs(5);

    fn permit_from(source: CommandSource) -> Permit {
        Permit::unmanaged(CommandId::new(), Priority::Exposure, source)
    }

    fn operator() -> CommandSource {
        CommandSource::Operator { overriding: false }
    }

    fn settle(pixels: f64, time_s: f64, timeout_s: f64) -> Settle {
        Settle {
            pixels,
            time_s,
            timeout_s,
        }
    }

    fn dither_cmd(amount: f64) -> GuiderCmd {
        GuiderCmd::Dither {
            amount,
            unit: PixelUnit::GuidePx,
            ra_only: false,
            settle: settle(1.5, 10.0, 40.0),
        }
    }

    fn guiding() -> MockState {
        MockState {
            app_state: "Guiding".into(),
            ..MockState::default()
        }
    }

    struct Rig {
        mock: Phd2Mock,
        clock: Arc<ManualClock>,
        sink: Arc<RecordingSink>,
        events: Arc<Mutex<Vec<EquipmentEvent>>>,
        conn: Phd2Connection,
    }

    impl Rig {
        async fn start(state: MockState) -> Self {
            Self::start_with(state, |_| {}).await
        }

        async fn start_with(state: MockState, tweak: impl FnOnce(&mut Phd2Settings)) -> Self {
            let mock = Phd2Mock::start_with(state).await;
            let clock = Arc::new(ManualClock::new());
            let sink = RecordingSink::new();
            let events = Arc::new(Mutex::new(Vec::new()));
            let mut settings = Phd2Settings::new("127.0.0.1", mock.port());
            tweak(&mut settings);
            let ctx = context(&clock, &sink, &events);
            let mut conn = connect(settings, ctx)
                .await
                .expect("connect to the PHD2 mock");
            assert!(conn.start());
            Self {
                mock,
                clock,
                sink,
                events,
                conn,
            }
        }

        fn handle(&self) -> &GuiderHandle {
            &self.conn.handle
        }

        async fn send(&self, cmd: GuiderCmd) -> DeviceResult<OpTicket> {
            self.handle().send(cmd, permit_from(operator())).await
        }

        fn names(&self) -> Vec<&'static str> {
            lock(&self.events)
                .iter()
                .map(EquipmentEvent::name)
                .collect()
        }

        fn events_named(&self, name: &str) -> Vec<EquipmentEvent> {
            lock(&self.events)
                .iter()
                .filter(|e| e.name() == name)
                .cloned()
                .collect()
        }

        async fn wait_event(&self, name: &str, n: usize) -> bool {
            let deadline = tokio::time::Instant::now() + WAIT;
            while tokio::time::Instant::now() < deadline {
                if self.events_named(name).len() >= n {
                    return true;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            false
        }

        async fn wait_snapshot(&self, pred: impl Fn(&GuiderSnapshot) -> bool) -> bool {
            let mut rx = self.handle().watch_snapshot();
            let waited = tokio::time::timeout(WAIT, async {
                loop {
                    if pred(&rx.borrow_and_update()) {
                        return;
                    }
                    if rx.changed().await.is_err() {
                        return;
                    }
                }
            })
            .await;
            waited.is_ok() && pred(&self.handle().snapshot())
        }

        async fn wait_frame(&self, frame: u64) -> bool {
            self.wait_snapshot(|s| s.last_step.as_ref().is_some_and(|st| st.frame == frame))
                .await
        }

        async fn wait_state(&self, state: GuiderAppState) -> bool {
            self.wait_snapshot(|s| s.app_state.value == Some(state))
                .await
        }

        async fn wait_health(&self, state: LinkState) -> bool {
            let deadline = tokio::time::Instant::now() + WAIT;
            while tokio::time::Instant::now() < deadline {
                if self.handle().health().state == state {
                    return true;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            false
        }

        /// Move the injected clock and give the service real time to react.
        async fn advance(&self, by: Duration) {
            self.clock.advance(by);
            tokio::time::sleep(Duration::from_millis(80)).await;
        }

        fn phases(&self) -> Vec<GuidePhase> {
            self.step_records().iter().map(|r| r.step.phase).collect()
        }

        fn step_records(&self) -> Vec<GuideStepRecord> {
            self.sink
                .records()
                .into_iter()
                .filter_map(|r| match r {
                    GroundTruthRecord::GuideStep(step) => Some(*step),
                    _ => None,
                })
                .collect()
        }

        fn dither_records(&self) -> Vec<DitherRecord> {
            self.sink
                .records()
                .into_iter()
                .filter_map(|r| match r {
                    GroundTruthRecord::Dither(d) => Some(*d),
                    _ => None,
                })
                .collect()
        }

        fn closed_records(&self) -> Vec<ConnectionEnd> {
            self.sink
                .records()
                .into_iter()
                .filter_map(|r| match r {
                    GroundTruthRecord::ConnectionClosed { end_class, .. } => Some(end_class),
                    _ => None,
                })
                .collect()
        }
    }

    fn context(
        clock: &Arc<ManualClock>,
        sink: &Arc<RecordingSink>,
        events: &Arc<Mutex<Vec<EquipmentEvent>>>,
    ) -> GuiderContext {
        let collected = Arc::clone(events);
        GuiderContext {
            clock: Arc::clone(clock) as Arc<dyn Clock>,
            recorder: Arc::clone(sink) as Arc<dyn RecorderSink>,
            events: Arc::new(move |event| lock(&collected).push(event)),
        }
    }

    async fn outcome(ticket: OpTicket) -> OpOutcome {
        tokio::time::timeout(WAIT, ticket.wait())
            .await
            .expect("the operation resolved in time")
    }

    #[test]
    fn options_and_settings_follow_the_spec() {
        assert_eq!(parse_option("").unwrap(), ("localhost".into(), 4400));
        assert_eq!(
            parse_option("192.168.1.5:4401").unwrap(),
            ("192.168.1.5".into(), 4401)
        );
        assert_eq!(parse_option("pi.local").unwrap(), ("pi.local".into(), 4400));
        assert_eq!(parse_option(":4402").unwrap(), ("localhost".into(), 4402));
        assert_eq!(parse_option("[::1]:4400").unwrap(), ("::1".into(), 4400));
        assert_eq!(
            parse_option("phd2://rig:4400").unwrap(),
            ("rig".into(), 4400)
        );
        for bad in ["host:0", "host:abc", "::1", "[::1", "[::1]x"] {
            assert_eq!(
                parse_option(bad).unwrap_err().class,
                ErrorClass::InvalidValue,
                "{bad}"
            );
        }
        let guider = GuiderConfig {
            rpc_timeout_s: 7,
            connect_equipment: true,
            settle_pixels: 2.0,
            aggressiveness_param_ra: "Aggression".into(),
            ..GuiderConfig::default()
        };
        let equipment = EquipmentConfig {
            lost_after_failures: 4,
            ring_buffer_minutes: 45,
            ..EquipmentConfig::default()
        };
        let s = Phd2Settings::from_config("rig", 4401, &guider, &equipment);
        assert_eq!(s.rpc_timeout, Duration::from_secs(7));
        assert!(s.connect_equipment);
        assert_eq!(s.default_settle.pixels, 2.0);
        assert_eq!(s.aggressiveness_param_ra, "Aggression");
        assert_eq!(s.lost_after_failures, 4);
        assert_eq!(s.history_retention, Duration::from_secs(45 * 60));
        assert_eq!(s.reconcile_interval, Duration::from_secs(5));
        assert_eq!(
            s.endpoint(),
            Endpoint::Phd2 {
                host: "rig".into(),
                port: 4401
            }
        );
        let defaults = Phd2Settings::new("localhost", 4400);
        assert!(!defaults.connect_equipment, "D16: off by default");
        assert_eq!(defaults.default_settle, Settle::default());
    }

    #[test]
    fn aggressiveness_names_are_discovered_never_assumed() {
        let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        // Hysteresis and Resist-Switch use `aggression`, Lowpass2
        // `aggressiveness`.
        let found = discover_aggressiveness(
            &names(&["minMove", "hysteresis", "aggression"]),
            &names(&["minMove", "Aggressiveness"]),
            "",
            "",
        );
        assert_eq!(found.ra.as_deref(), Some("aggression"));
        assert_eq!(found.dec.as_deref(), Some("Aggressiveness"));
        assert_eq!(found.disabled_reason, None);
        // A predictive-PEC algorithm has neither.
        let none = discover_aggressiveness(
            &names(&["controlGain", "minMove", "predictionGain"]),
            &names(&["minMove"]),
            "",
            "",
        );
        assert_eq!((none.ra, none.dec), (None, None));
        assert_eq!(
            none.disabled_reason.as_deref(),
            Some("no_aggressiveness_param")
        );
        let half = discover_aggressiveness(&names(&["aggression"]), &names(&["minMove"]), "", "");
        assert_eq!(
            half.disabled_reason.as_deref(),
            Some("no_aggressiveness_param:dec")
        );
        // A configured name must exist; matching ignores case.
        let configured = discover_aggressiveness(
            &names(&["minMove", "Aggression"]),
            &names(&["aggressiveness"]),
            "aggression",
            "",
        );
        assert_eq!(configured.ra.as_deref(), Some("Aggression"));
        let missing = discover_aggressiveness(
            &names(&["minMove"]),
            &names(&["aggressiveness"]),
            "MaxAggr",
            "",
        );
        assert_eq!(missing.ra, None);
        assert_eq!(
            missing.disabled_reason.as_deref(),
            Some("configured_aggressiveness_param_missing:ra")
        );
    }

    #[test]
    fn settle_outcomes_keep_phd2_status_and_class() {
        let ok = SettleDoneEvent {
            status: Some(0),
            ..SettleDoneEvent::default()
        };
        assert_eq!(settle_outcome(&ok), OpOutcome::Succeeded);
        let timeout = SettleDoneEvent {
            status: Some(1),
            error: Some("timed-out waiting for guider to settle".into()),
            ..SettleDoneEvent::default()
        };
        let OpOutcome::Failed(err) = settle_outcome(&timeout) else {
            panic!("expected a failure")
        };
        assert_eq!((err.class, err.code), (ErrorClass::Timeout, Some(1)));
        let OpOutcome::Failed(err) = settle_outcome(&SettleDoneEvent::default()) else {
            panic!("expected a failure")
        };
        assert_eq!(err.class, ErrorClass::Driver);
        assert!(
            err.message.contains("status -1"),
            "a missing Status is not success"
        );
    }

    #[tokio::test]
    async fn the_connect_burst_builds_identity_snapshot_and_records() {
        let rig = Rig::start(MockState::default()).await;
        let port = rig.mock.port();
        let id = rig.handle().identity();
        assert_eq!(id.kind, DeviceKind::Guider);
        assert_eq!(id.driver, DriverId::Phd2);
        assert_eq!(
            id.identity_key,
            format!("phd2:127.0.0.1:{port}/Mock Profile")
        );
        assert_eq!(id.driver_version.as_deref(), Some("2.6.13dev1"));
        assert_eq!(id.interface_version, Some(1));
        assert_eq!(id.qualification, Qualification::ImplementedUnqualified);
        assert_eq!(
            id.endpoint,
            Endpoint::Phd2 {
                host: "127.0.0.1".into(),
                port
            }
        );
        let snap = rig.handle().snapshot();
        assert_eq!(snap.connection_id, id.connection_id);
        assert_eq!(snap.app_state.value, Some(GuiderAppState::Stopped));
        assert_eq!(snap.app_state.source, Source::Phd2Event);
        assert_eq!(snap.phd2_version.as_deref(), Some("2.6.13dev1"));
        assert_eq!(snap.profile.as_deref(), Some("Mock Profile"));
        assert_eq!(snap.camera_name.as_deref(), Some("Mock Guide Camera"));
        assert_eq!(snap.mount_name.as_deref(), Some("On-camera"));
        assert_eq!(snap.pixel_scale_arcsec, Some(1.5));
        assert_eq!(snap.calibrated, Some(true));
        assert!(snap.calibration.is_some());
        assert_eq!(snap.dec_guide_mode.as_deref(), Some("Auto"));
        assert_eq!(snap.exposure_ms, Some(2000));
        assert_eq!(snap.algo_params.len(), 5);
        assert_eq!(snap.aggressiveness_param_ra.as_deref(), Some("aggression"));
        assert_eq!(
            snap.aggressiveness_param_dec.as_deref(),
            Some("aggressiveness")
        );
        assert_eq!(snap.actuation_disabled_reason, None);
        // G-a: no equipment or profile changes; the documented read order.
        assert_eq!(rig.mock.method_count("set_connected"), 0);
        assert_eq!(rig.mock.method_count("set_profile"), 0);
        let methods: Vec<String> = rig.mock.requests().into_iter().map(|r| r.method).collect();
        assert_eq!(
            methods[..10],
            [
                "get_connected",
                "get_profile",
                "get_current_equipment",
                "get_pixel_scale",
                "get_calibrated",
                "get_calibration_data",
                "get_dec_guide_mode",
                "get_exposure",
                "get_algo_param_names",
                "get_algo_param_names",
            ]
        );
        assert_eq!(rig.mock.method_count("get_algo_param"), 5);
        assert_eq!(
            rig.mock.requests()[8].params,
            json!(["ra"]),
            "names are read per axis"
        );
        // device_connected first, then what the burst produced.
        let names = rig.names();
        assert_eq!(names[0], "device_connected");
        assert!(names.contains(&"guider_state"));
        let first = lock(&rig.events)[0].clone();
        assert_eq!(first.device_kind, Some(DeviceKind::Guider));
        assert_eq!(first.connection_id, Some(id.connection_id));
        assert!(matches!(
            rig.sink.records().first(),
            Some(GroundTruthRecord::ConnectionOpened(row))
                if row.identity.connection_id == id.connection_id
        ));
        let caps = &rig.conn.capabilities;
        assert_eq!(caps["aggressiveness_param_ra"], json!("aggression"));
        assert_eq!(caps["equipment_connected"], json!(true));
        assert_eq!(caps["qualification"], json!("implemented_unqualified"));
        assert_eq!(rig.conn.report.path, ConnectPath::Phd2);
        assert!(
            rig.conn.report.warnings.is_empty(),
            "{:?}",
            rig.conn.report.warnings
        );
        assert_eq!(rig.handle().health().state, LinkState::Ready);
    }

    #[tokio::test]
    async fn phd2_equipment_is_connected_only_when_configured() {
        let state = MockState {
            connected: false,
            ..MockState::default()
        };
        let rig = Rig::start(state.clone()).await;
        assert_eq!(rig.mock.method_count("set_connected"), 0);
        assert!(rig
            .conn
            .report
            .warnings
            .iter()
            .any(|w| w.contains("connect_equipment")));
        assert_eq!(rig.conn.capabilities["equipment_connected"], json!(false));

        let rig = Rig::start_with(state, |s| s.connect_equipment = true).await;
        assert_eq!(rig.mock.method_count("set_connected"), 1);
        assert_eq!(
            rig.mock.last_request("set_connected").unwrap().params,
            json!([true])
        );
        assert_eq!(rig.conn.capabilities["equipment_connected"], json!(true));
    }

    #[tokio::test]
    async fn both_line_endings_work_end_to_end() {
        for crlf in [false, true] {
            let state = MockState {
                crlf,
                ..MockState::default()
            };
            let rig = Rig::start(state).await;
            assert_eq!(
                rig.handle().snapshot().profile.as_deref(),
                Some("Mock Profile"),
                "crlf={crlf}"
            );
            rig.mock.send_event(Phd2Mock::guide_step(1, 0.1, 0.2));
            assert!(rig.wait_frame(1).await, "crlf={crlf}");
        }
    }

    #[tokio::test]
    async fn guide_steps_are_recorded_unmodified_and_fill_the_telemetry_block() {
        let rig = Rig::start(guiding()).await;
        let values = [(0.31, -0.12), (-0.25, 0.08), (0.12, 0.2), (-0.4, -0.05)];
        let mut sent = Vec::new();
        for (i, (ra, dec)) in values.iter().enumerate() {
            let mut step = Phd2Mock::guide_step(i as u64 + 1, *ra, *dec);
            if i == 2 {
                let obj = step.as_object_mut().unwrap();
                obj.remove("RALimited");
                obj.remove("DecLimited");
                obj.remove("DECDuration");
                obj.remove("DECDirection");
            }
            sent.push(step.clone());
            rig.mock.send_event(step);
        }
        assert!(rig.wait_frame(4).await);
        let records = rig.step_records();
        assert_eq!(records.len(), 4);
        for (record, json) in records.iter().zip(&sent) {
            let s = &record.step;
            assert_eq!(record.connection_id, rig.handle().connection_id());
            assert_eq!(s.frame, json["Frame"].as_u64().unwrap());
            assert_eq!(s.phd2_time_s, json["Time"].as_f64().unwrap());
            assert_eq!(s.dx, json["dx"].as_f64().unwrap(), "PHD2 sign kept");
            assert_eq!(s.dy, json["dy"].as_f64().unwrap());
            assert_eq!(s.ra_raw, json["RADistanceRaw"].as_f64());
            assert_eq!(s.dec_raw, json["DECDistanceRaw"].as_f64());
            assert_eq!(s.ra_guide, json["RADistanceGuide"].as_f64());
            assert_eq!(s.ra_duration_ms, Some(120));
            assert_eq!(s.ra_direction.as_deref(), json["RADirection"].as_str());
            assert_eq!(s.star_mass, Some(12345.0));
            assert_eq!(s.snr, Some(25.0));
            assert_eq!(s.hfd, Some(2.2));
            assert_eq!(s.avg_dist, Some(0.3));
            assert_eq!(s.error_code, Some(0));
            assert!(s.phd2_timestamp.is_some());
            assert_eq!(s.received_at_utc, rig.clock.now_utc());
            assert_eq!(s.pixel_scale_arcsec, Some(1.5));
            assert_eq!(s.phase, GuidePhase::Guiding);
        }
        assert_eq!(records[2].step.ra_limited, None, "absent, not false");
        assert_eq!(records[2].step.dec_duration_ms, None);
        assert_eq!(records[0].step.ra_limited, Some(false));
        assert_eq!(rig.events_named("guide_step").len(), 4);
        let snap = rig.handle().snapshot();
        assert_eq!(snap.rms.unwrap().steps_n, 4);
        assert!(snap.clock_offset_s.is_some());
        // Acceptance: `guiding` telemetry is non-null against the mock.
        let telemetry = super::super::guiding_telemetry(&snap);
        assert_eq!(telemetry["state"], json!("guiding"));
        assert!(telemetry["rms_total"].is_number());
        assert!(telemetry["rms_ra_arcsec"].is_number());
        assert_eq!(telemetry["dx"], sent[3]["dx"]);
        assert_eq!(telemetry["snr"], json!(25.0));
        assert_eq!(telemetry["pixel_scale"], json!(1.5));
        assert_eq!(telemetry["calibrated"], json!(true));
    }

    #[tokio::test]
    async fn per_exposure_rms_equals_a_reference_computation() {
        let rig = Rig::start(guiding()).await;
        let values = [
            (0.31, -0.12),
            (-0.25, 0.08),
            (0.12, 0.2),
            (-0.4, -0.05),
            (0.05, 0.33),
            (0.6, -0.21),
            (-0.11, 0.0),
            (0.27, 0.14),
            (-0.33, -0.27),
            (0.9, 0.05),
            (-0.75, 0.4),
            (0.02, -0.1),
        ];
        let t0 = rig.clock.now_utc();
        for (i, (ra, dec)) in values.iter().enumerate() {
            rig.clock.advance(Duration::from_secs(1));
            let frame = i as u64 + 1;
            let mut step = Phd2Mock::guide_step(frame, *ra, *dec);
            step["Timestamp"] = json!(1_900_000_000.0 + i as f64);
            rig.mock.send_event(step);
            assert!(rig.wait_frame(frame).await);
        }
        // An exposure from t0+3 s to t0+10 s covers frames 3..=10.
        let start = t0 + chrono::Duration::seconds(3);
        let end = t0 + chrono::Duration::seconds(10);
        let record = rig
            .conn
            .windows
            .record_exposure_window(Some("frame-42".into()), start, end);
        let window = &values[2..10];
        let reference = |axis: &dyn Fn(&(f64, f64)) -> f64| {
            let xs: Vec<f64> = window.iter().map(axis).collect();
            let n = xs.len() as f64;
            let mean = xs.iter().sum::<f64>() / n;
            let std = (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
            let rms0 = (xs.iter().map(|x| x * x).sum::<f64>() / n).sqrt();
            let peak = xs.iter().fold(0.0_f64, |p, x| p.max(x.abs()));
            (std, rms0, peak)
        };
        let (ra_std, ra_rms0, ra_peak) = reference(&|v| v.0);
        let (dec_std, dec_rms0, dec_peak) = reference(&|v| v.1);
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        let rms = record.rms.expect("RMS for the window");
        assert_eq!(rms.steps_n, 8);
        assert!(close(rms.ra_px, ra_std));
        assert!(close(rms.dec_px, dec_std));
        assert!(close(rms.total_px, ra_std.hypot(dec_std)));
        assert!(close(rms.rms0_ra_px, ra_rms0));
        assert!(close(rms.rms0_dec_px, dec_rms0));
        assert!(close(rms.peak_ra_px, ra_peak));
        assert!(close(rms.peak_dec_px, dec_peak));
        assert!(close(rms.ra_arcsec.unwrap(), ra_std * 1.5));
        assert!(close(rms.dec_arcsec.unwrap(), dec_std * 1.5));
        assert!(close(
            rms.total_arcsec.unwrap(),
            ra_std.hypot(dec_std) * 1.5
        ));
        assert!(close(
            rms.rms0_total_arcsec.unwrap(),
            ra_rms0.hypot(dec_rms0) * 1.5
        ));
        assert_eq!(record.frame_id.as_deref(), Some("frame-42"));
        assert_eq!((record.window_start, record.window_end), (start, end));
        assert_eq!(record.ra_aggressiveness, Some(0.7));
        assert_eq!(record.dec_aggressiveness, Some(0.5));
        assert_eq!(record.guide_star_snr, Some(25.0));
        assert_eq!(record.freeze_reason, None);
        assert!(!record.dither_active);
        assert_eq!(record.algorithm_params.len(), 5);
        let recorded = rig.sink.records().into_iter().find_map(|r| match r {
            GroundTruthRecord::GuiderWindow(w) => Some(*w),
            _ => None,
        });
        assert_eq!(recorded.as_ref(), Some(&record));
        let stats = rig.conn.windows.window_stats(start, end);
        assert_eq!(stats.rms, Some(rms));
        assert_eq!(stats.state, Some(GuiderAppState::Guiding));
        assert_eq!(stats.pulses_n, 16);
        assert_eq!(rig.conn.windows.steps(start, end).len(), 8);
    }

    #[tokio::test]
    async fn a_dither_settles_tags_its_steps_and_writes_a_dither_record() {
        let mut state = guiding();
        state.dither_script.auto = false;
        let rig = Rig::start(state).await;
        let ticket = rig.send(dither_cmd(5.0)).await.unwrap();
        let command_id = ticket.command_id;
        assert!(rig.mock.wait_for_request("dither", WAIT).await);
        assert_eq!(
            rig.mock.last_request("dither").unwrap().params,
            json!({"amount": 5.0, "raOnly": false,
                   "settle": {"pixels": 1.5, "time": 10, "timeout": 40}})
        );
        let settling = |distance: f64| {
            json!({"Event": "Settling", "Distance": distance, "Time": 1.0,
                   "SettleTime": 10.0, "StarLocked": true})
        };
        // The mock answers `dither` with GuidingDithered {2.4, -1.8}; the
        // test then plays the steps and settle events PHD2 sends.
        rig.mock.send_event(Phd2Mock::guide_step(10, 2.0, -1.5));
        rig.mock.send_event(settling(2.6));
        rig.mock.send_event(Phd2Mock::guide_step(11, 1.0, -0.6));
        rig.mock.send_event(settling(0.9));
        rig.mock.send_event(Phd2Mock::guide_step(12, 0.3, -0.2));
        rig.mock.send_event(
            json!({"Event": "SettleDone", "Status": 0, "TotalFrames": 3, "DroppedFrames": 0}),
        );
        assert_eq!(outcome(ticket).await, OpOutcome::Succeeded);
        rig.mock.send_event(Phd2Mock::guide_step(13, 0.1, 0.0));
        assert!(rig.wait_frame(13).await);
        let phases: Vec<GuidePhase> = rig.step_records().iter().map(|r| r.step.phase).collect();
        assert_eq!(
            phases,
            vec![
                GuidePhase::Dithering,
                GuidePhase::Dithering,
                GuidePhase::Settling,
                GuidePhase::Guiding
            ]
        );
        let records = rig.dither_records();
        assert_eq!(records.len(), 1);
        let d = &records[0];
        assert_eq!(d.command_id, Some(command_id));
        assert_eq!(d.source, operator());
        assert_eq!(d.backend, DitherBackend::Phd2);
        assert_eq!(
            (d.amount, d.amount_unit, d.ra_only),
            (5.0, PixelUnit::GuidePx, false)
        );
        assert_eq!(d.settle, settle(1.5, 10.0, 40.0));
        assert_eq!((d.dx, d.dy), (Some(2.4), Some(-1.8)));
        assert_eq!(d.settle_status, Some(0));
        assert_eq!(d.settle_error, None);
        assert_eq!((d.total_frames, d.dropped_frames), (Some(3), Some(0)));
        assert_eq!(d.max_distance_px, Some(2.6));
        assert!(d.settle_done_at.is_some());
        assert_eq!(
            (d.next_frame_id.clone(), d.next_frame_accepted),
            (None, None)
        );
        // The journal order of a dither.
        let tracked = [
            "operation_commanded",
            "operation_started",
            "dithered",
            "settling",
            "settle_done",
            "operation_ended",
        ];
        let order: Vec<&str> = rig
            .names()
            .into_iter()
            .filter(|n| tracked.contains(n))
            .collect();
        assert_eq!(
            order,
            vec![
                "operation_commanded",
                "operation_started",
                "dithered",
                "settling",
                "settling",
                "settle_done",
                "operation_ended"
            ]
        );
        let ended = rig.events_named("operation_ended");
        assert_eq!(ended[0].command_id, Some(command_id));
        let snap = rig.handle().snapshot();
        let last = snap.last_dither.clone().unwrap();
        assert_eq!(
            (last.dx, last.dy, last.settle_status),
            (Some(2.4), Some(-1.8), Some(0))
        );
        assert!(!snap.settling);
        assert!(snap.active_op.is_none());
    }

    #[tokio::test]
    async fn a_failed_settle_fails_the_dither_and_records_the_error() {
        let mut state = guiding();
        state.dither_script.done = Some((1, Some("timed-out waiting for guider to settle".into())));
        let rig = Rig::start(state).await;
        let ticket = rig.send(dither_cmd(3.0)).await.unwrap();
        let OpOutcome::Failed(err) = outcome(ticket).await else {
            panic!("expected a failure");
        };
        assert_eq!(err.class, ErrorClass::Timeout);
        assert_eq!(err.code, Some(1));
        let d = &rig.dither_records()[0];
        assert_eq!(d.settle_status, Some(1));
        assert_eq!(
            d.settle_error.as_deref(),
            Some("timed-out waiting for guider to settle")
        );
        assert_eq!(d.total_frames, Some(3));
        let done = rig.events_named("settle_done");
        assert!(matches!(
            &done[0].event,
            EventKind::SettleDone {
                status: 1,
                error: Some(_),
                ..
            }
        ));
        assert_eq!(
            rig.handle()
                .snapshot()
                .last_dither
                .clone()
                .unwrap()
                .settle_status,
            Some(1)
        );
    }

    #[tokio::test]
    async fn a_reentrant_dither_error_fails_without_hanging() {
        let mut state = guiding();
        state.dither_error = Some((1, "cannot dither while settling".into()));
        let rig = Rig::start(state).await;
        let ticket = rig.send(dither_cmd(3.0)).await.unwrap();
        let OpOutcome::Failed(err) = outcome(ticket).await else {
            panic!("expected a failure");
        };
        assert_eq!(err.class, ErrorClass::InvalidOperation);
        assert!(err.message.contains("cannot dither while settling"));
        let d = &rig.dither_records()[0];
        assert_eq!(d.settle_status, None);
        assert!(d
            .settle_error
            .as_deref()
            .unwrap()
            .contains("cannot dither while settling"));
        // Nothing is left behind: the next dither runs.
        rig.mock.with_state(|s| s.dither_error = None);
        let ticket = rig.send(dither_cmd(3.0)).await.unwrap();
        assert_eq!(outcome(ticket).await, OpOutcome::Succeeded);
        assert_eq!(rig.dither_records().len(), 2);
    }

    #[tokio::test]
    async fn an_unanswered_initiator_leaves_the_outcome_unknown() {
        let mut state = guiding();
        state.silent_methods.insert("dither".into());
        let rig = Rig::start(state).await;
        let ticket = rig.send(dither_cmd(3.0)).await.unwrap();
        assert!(rig.mock.wait_for_request("dither", WAIT).await);
        rig.advance(Duration::from_secs(10)).await;
        let OpOutcome::Unknown(err) = outcome(ticket).await else {
            panic!("PHD2 may have dithered: the outcome is unknown, not failed");
        };
        assert_eq!(err.class, ErrorClass::Timeout);
        assert_eq!(rig.events_named("outcome_unknown").len(), 1);
        // The unanswered request degraded the link; the next reconcile
        // answered and restored it.
        assert!(rig.wait_event("device_degraded", 1).await);
        assert!(rig.dither_records()[0]
            .settle_error
            .as_deref()
            .unwrap()
            .contains("did not answer dither"));
        // PHD2 may have dithered, so the next step is still tagged; but with
        // no GuidingDithered within the settle bound (40 s + 10 s from the
        // request) the tag lapses.
        rig.mock.send_event(Phd2Mock::guide_step(1, 0.1, 0.1));
        assert!(rig.wait_frame(1).await);
        rig.advance(Duration::from_secs(41)).await;
        rig.mock.send_event(Phd2Mock::guide_step(2, 0.1, 0.1));
        assert!(rig.wait_frame(2).await);
        assert_eq!(
            rig.phases(),
            vec![GuidePhase::Dithering, GuidePhase::Guiding]
        );
        assert_eq!(rig.handle().snapshot().rms.unwrap().steps_n, 1);
    }

    /// A dither from PHD2's Manual Guide dialog or legacy socket server:
    /// `GuidingDithered` and no settle events at all.
    #[tokio::test]
    async fn a_dither_phd2_never_settles_does_not_tag_every_later_step() {
        let rig = Rig::start(guiding()).await;
        rig.mock
            .send_event(json!({"Event": "GuidingDithered", "dx": 3.0, "dy": -2.0}));
        rig.mock.send_event(Phd2Mock::guide_step(1, 2.0, -1.5));
        assert!(rig.wait_frame(1).await);
        // An Achernar dither now waits for that dither's settle (G-f 1)...
        let ticket = rig.send(dither_cmd(3.0)).await.unwrap();
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(rig.mock.method_count("dither"), 0);
        // ...but only for the grace (5 s at a 2 s exposure), not 50 s.
        rig.advance(Duration::from_secs(6)).await;
        assert!(rig.mock.wait_for_request("dither", WAIT).await);
        assert_eq!(outcome(ticket).await, OpOutcome::Succeeded);
        assert!(!rig.handle().snapshot().settling);

        // Another unsettled dither: tagged within the grace, guiding after.
        rig.mock
            .send_event(json!({"Event": "GuidingDithered", "dx": -1.0, "dy": 2.5}));
        rig.mock.send_event(Phd2Mock::guide_step(2, -0.9, 2.0));
        assert!(rig.wait_frame(2).await);
        rig.advance(Duration::from_secs(6)).await;
        for frame in 3..=5 {
            rig.mock.send_event(Phd2Mock::guide_step(frame, 0.1, -0.1));
        }
        assert!(rig.wait_frame(5).await);
        assert_eq!(
            rig.phases(),
            vec![
                GuidePhase::Dithering,
                GuidePhase::Dithering,
                GuidePhase::Guiding,
                GuidePhase::Guiding,
                GuidePhase::Guiding
            ]
        );
        // The telemetry RMS is current, not frozen at the dither.
        let snap = rig.handle().snapshot();
        assert_eq!(snap.rms.unwrap().steps_n, 3);
        assert!(!snap.settling);
    }

    #[tokio::test]
    async fn a_dither_waits_for_a_settle_already_in_progress() {
        let rig = Rig::start(guiding()).await;
        // Somebody else's settle (a dither from the PHD2 GUI) is under way.
        rig.mock.send_event(json!({"Event": "SettleBegin"}));
        rig.mock
            .send_event(json!({"Event": "Settling", "Distance": 3.0, "Time": 1.0,
                                   "SettleTime": 10.0, "StarLocked": true}));
        assert!(rig.wait_snapshot(|s| s.settling).await);
        let ticket = rig.send(dither_cmd(4.0)).await.unwrap();
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(
            rig.mock.method_count("dither"),
            0,
            "PHD2 rejects re-entrant settles"
        );
        assert_eq!(
            rig.handle()
                .snapshot()
                .active_op
                .as_ref()
                .map(|op| op.phase),
            Some(OpPhase::Starting)
        );
        rig.mock
            .send_event(json!({"Event": "SettleDone", "Status": 0}));
        assert!(rig.mock.wait_for_request("dither", WAIT).await);
        assert_eq!(outcome(ticket).await, OpOutcome::Succeeded);
    }

    #[tokio::test]
    async fn dithers_are_refused_at_acceptance_when_they_cannot_run() {
        let mut state = MockState::default();
        state.dither_script.done = None;
        let rig = Rig::start(state).await;
        // Not guiding.
        let err = rig.send(dither_cmd(3.0)).await.unwrap_err();
        assert_eq!(err.class, ErrorClass::InvalidOperation);
        rig.mock.with_state(|s| s.app_state = "Guiding".into());
        rig.mock.send_event(Phd2Mock::guide_step(1, 0.1, 0.1));
        assert!(rig.wait_state(GuiderAppState::Guiding).await);
        // A main-camera amount needs a conversion the guider cannot make.
        let err = rig
            .send(GuiderCmd::Dither {
                amount: 3.0,
                unit: PixelUnit::MainPx,
                ra_only: true,
                settle: Settle::default(),
            })
            .await
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::InvalidValue);
        assert_eq!(
            rig.send(dither_cmd(-1.0)).await.unwrap_err().class,
            ErrorClass::InvalidValue
        );
        let bad_settle = GuiderCmd::Dither {
            amount: 3.0,
            unit: PixelUnit::GuidePx,
            ra_only: false,
            settle: settle(0.0, 10.0, 40.0),
        };
        assert_eq!(
            rig.send(bad_settle).await.unwrap_err().class,
            ErrorClass::InvalidValue
        );
        // One operation at a time.
        let held = rig.send(dither_cmd(3.0)).await.unwrap();
        let err = rig.send(GuiderCmd::Stop).await.unwrap_err();
        assert_eq!(err.class, ErrorClass::InvalidOperation);
        // The M-1 actuator is Tier 2.
        let err = rig
            .send(GuiderCmd::SetAlgoParam {
                axis: Axis::Ra,
                name: "aggression".into(),
                value: 0.6,
            })
            .await
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::Unsupported);
        assert_eq!(rig.mock.method_count("set_algo_param"), 0);
        drop(held);
    }

    #[tokio::test]
    async fn star_lost_moves_to_lost_lock_and_tags_the_recovery_step() {
        let rig = Rig::start(guiding()).await;
        rig.mock.send_event(Phd2Mock::guide_step(1, 0.1, 0.1));
        assert!(rig.wait_frame(1).await);
        rig.mock.send_event(json!({
            "Event": "StarLost", "Frame": 2, "Time": 4.0, "StarMass": 0.0, "SNR": 0.0,
            "AvgDist": 1.1, "ErrorCode": 2, "Status": "Star lost - low SNR"
        }));
        assert!(rig.wait_state(GuiderAppState::LostLock).await);
        let lost = rig.events_named("star_lost");
        let EventKind::StarLost {
            frame,
            avg_dist,
            error_code,
            status,
            ..
        } = &lost[0].event
        else {
            panic!("expected star_lost")
        };
        assert_eq!(
            (*frame, *avg_dist, *error_code),
            (Some(2), Some(1.1), Some(2))
        );
        assert_eq!(status.as_deref(), Some("Star lost - low SNR"));
        assert!(lost[0].device_time.is_some());
        rig.mock.send_event(Phd2Mock::guide_step(3, 0.8, -0.4));
        rig.mock.send_event(Phd2Mock::guide_step(4, 0.1, 0.0));
        assert!(rig.wait_frame(4).await);
        let phases: Vec<GuidePhase> = rig.step_records().iter().map(|r| r.step.phase).collect();
        assert_eq!(
            phases,
            vec![
                GuidePhase::Guiding,
                GuidePhase::Recovering,
                GuidePhase::Guiding
            ]
        );
        assert_eq!(
            rig.handle().snapshot().app_state.value,
            Some(GuiderAppState::Guiding)
        );
        let changes: Vec<(GuiderAppState, GuiderAppState)> = rig
            .events_named("guider_state")
            .iter()
            .filter_map(|e| match e.event {
                EventKind::GuiderState { from, to } => Some((from, to)),
                _ => None,
            })
            .collect();
        assert!(changes.contains(&(GuiderAppState::Guiding, GuiderAppState::LostLock)));
        assert!(changes.contains(&(GuiderAppState::LostLock, GuiderAppState::Guiding)));
    }

    #[tokio::test]
    async fn alerts_param_changes_and_guiding_stopped_are_reported() {
        let rig = Rig::start(guiding()).await;
        rig.mock.send_event(
            json!({"Event": "Alert", "Msg": "Guide star lost repeatedly", "Type": "warning"}),
        );
        assert!(rig.wait_event("guider_alert", 1).await);
        assert!(matches!(
            &rig.events_named("guider_alert")[0].event,
            EventKind::GuiderAlert { message, alert_type: Some(t) }
                if message == "Guide star lost repeatedly" && t == "warning"
        ));
        // An operator edit in PHD2: reported and re-read.
        rig.mock.with_state(|s| {
            s.algo_values
                .insert(("ra".into(), "aggression".into()), 0.55);
        });
        let reads = rig.mock.method_count("get_algo_param");
        rig.mock.send_event(
            json!({"Event": "GuideParamChange", "Name": "Aggressiveness", "Value": 0.55}),
        );
        assert!(rig.wait_event("guide_param_changed", 1).await);
        assert!(matches!(
            &rig.events_named("guide_param_changed")[0].event,
            EventKind::GuideParamChanged { name, value }
                if name == "Aggressiveness" && *value == json!(0.55)
        ));
        assert!(
            rig.wait_snapshot(|s| s
                .algo_params
                .iter()
                .any(|p| p.axis == Axis::Ra && p.name == "aggression" && p.value == 0.55))
                .await
        );
        assert!(rig.mock.method_count("get_algo_param") > reads);
        rig.mock.send_event(json!({"Event": "GuidingStopped"}));
        assert!(rig.wait_state(GuiderAppState::Stopped).await);
        assert_eq!(rig.events_named("guiding_stopped").len(), 1);
        rig.mock
            .send_event(json!({"Event": "CalibrationFailed", "Reason": "star did not move"}));
        assert!(rig.wait_event("calibration", 1).await);
        assert!(matches!(
            &rig.events_named("calibration")[0].event,
            EventKind::Calibration {
                phase: CalibrationPhase::Failed,
                detail: Some(d)
            } if d == "star did not move"
        ));
        rig.mock.send_event(json!({"Event": "LockPositionLost"}));
        assert!(rig.wait_event("lock_position_lost", 1).await);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_socket_drop_loses_the_guider_and_leaves_the_operation_unknown() {
        let mut state = guiding();
        state.dither_script.done = None;
        let rig = Rig::start(state).await;
        let ticket = rig.send(dither_cmd(3.0)).await.unwrap();
        assert!(rig.mock.wait_for_request("dither", WAIT).await);
        rig.mock.drop_clients();
        let OpOutcome::Unknown(err) = outcome(ticket).await else {
            panic!("an interrupted settle has an unknown outcome");
        };
        assert_eq!(err.class, ErrorClass::Disconnected);
        assert!(rig.wait_health(LinkState::Lost).await);
        let health = rig.handle().health();
        assert_eq!(health.failure_class, Some(FailureClass::Disconnected));
        assert!(rig.wait_event("device_lost", 1).await);
        assert!(matches!(
            rig.events_named("device_lost")[0].event,
            EventKind::DeviceLost {
                class: FailureClass::Disconnected,
                ..
            }
        ));
        assert_eq!(rig.events_named("outcome_unknown").len(), 1);
        assert_eq!(rig.closed_records(), vec![ConnectionEnd::Lost]);
        assert!(rig.dither_records()[0].settle_error.is_some());
        assert!(
            rig.wait_snapshot(|s| s.app_state.state == ObsState::Stale)
                .await
        );
        // Commands fail fast; the handle is not retired, so abort still runs
        // its out-of-band path (D19).
        let err = rig.send(GuiderCmd::Stop).await.unwrap_err();
        assert_eq!(err.class, ErrorClass::Disconnected);
        assert!(!rig.handle().is_retired());
        let abort = rig.handle().abort();
        assert!(abort.abort_path_ran);
        assert!(tokio::time::timeout(WAIT, abort.outcome())
            .await
            .unwrap()
            .is_ok());
        assert_eq!(rig.mock.method_count("stop_capture"), 1);
        rig.handle().disconnect(false).await.unwrap();
        assert!(rig.handle().is_retired());
        assert_eq!(
            rig.closed_records(),
            vec![ConnectionEnd::Lost],
            "closed once"
        );
        assert!(matches!(
            rig.events_named("device_disconnected")[0].event,
            EventKind::DeviceDisconnected {
                end_class: ConnectionEnd::Lost,
                ..
            }
        ));
        assert_eq!(rig.mock.method_count("set_connected"), 0);
    }

    #[tokio::test]
    async fn start_waits_for_exactly_one_settle_done_on_the_clock() {
        let rig = Rig::start(MockState::default()).await;
        let ticket = rig
            .send(GuiderCmd::Start {
                settle: settle(1.5, 10.0, 40.0),
                recalibrate: true,
            })
            .await
            .unwrap();
        assert_eq!(outcome(ticket).await, OpOutcome::Succeeded);
        assert_eq!(
            rig.mock.last_request("guide").unwrap().params,
            json!({"settle": {"pixels": 1.5, "time": 10, "timeout": 40}, "recalibrate": true})
        );
        assert!(rig.wait_state(GuiderAppState::Guiding).await);
        for name in ["guiding_started", "settle_begin", "settling", "settle_done"] {
            assert!(!rig.events_named(name).is_empty(), "{name}");
        }
        // PHD2 never reports the settle: TimedOut after timeout + 10 s.
        rig.mock.with_state(|s| s.guide_script.done = None);
        let mut ticket = rig
            .send(GuiderCmd::Start {
                settle: settle(1.5, 5.0, 20.0),
                recalibrate: false,
            })
            .await
            .unwrap();
        assert!(rig.mock.wait_for_count("guide", 2, WAIT).await);
        // The deadline runs from PHD2's acknowledgement of `guide`.
        assert!(
            rig.wait_snapshot(|s| s
                .active_op
                .as_ref()
                .is_some_and(|op| op.phase != OpPhase::Starting))
                .await
        );
        rig.advance(Duration::from_secs(29)).await;
        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut ticket.done)
                .await
                .is_err(),
            "still waiting before the deadline"
        );
        rig.advance(Duration::from_secs(2)).await;
        assert_eq!(outcome(ticket).await, OpOutcome::TimedOut);
    }

    #[tokio::test]
    async fn a_recalibrating_start_extends_the_settle_wait_and_fails_on_calibration_failure() {
        use crate::equipment::testing::phd2_mock::MockReply;
        let rig = Rig::start(MockState::default()).await;
        rig.mock.set_responder(|request, state| {
            (request.method == "guide").then(|| {
                state.app_state = "Calibrating".into();
                MockReply {
                    result: Ok(json!(0)),
                    events: vec![json!({"Event": "StartCalibration", "Mount": "On-camera"})],
                }
            })
        });
        let start = GuiderCmd::Start {
            settle: settle(1.5, 5.0, 20.0),
            recalibrate: true,
        };
        let mut ticket = rig.send(start.clone()).await.unwrap();
        assert!(rig.wait_state(GuiderAppState::Calibrating).await);
        // 25 s of calibration: past the plain 30 s budget once the next
        // 25 s pass, but every calibration step restarts it.
        rig.advance(Duration::from_secs(25)).await;
        rig.mock
            .send_event(json!({"Event": "Calibrating", "Mount": "On-camera",
                                   "dir": "West", "dist": 12.0, "step": 4}));
        assert!(rig.wait_event("calibration", 2).await);
        rig.advance(Duration::from_secs(25)).await;
        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut ticket.done)
                .await
                .is_err(),
            "calibration progress keeps the start alive"
        );
        rig.mock.with_state(|s| s.app_state = "Guiding".into());
        for event in [
            json!({"Event": "CalibrationComplete", "Mount": "On-camera"}),
            json!({"Event": "StartGuiding"}),
            json!({"Event": "SettleBegin"}),
            json!({"Event": "SettleDone", "Status": 0}),
        ] {
            rig.mock.send_event(event);
        }
        assert_eq!(outcome(ticket).await, OpOutcome::Succeeded);
        assert_eq!(rig.handle().snapshot().calibrated, Some(true));

        // A failed calibration fails the start at once.
        rig.mock.with_state(|s| s.app_state = "Stopped".into());
        rig.mock.send_event(json!({"Event": "GuidingStopped"}));
        assert!(rig.wait_state(GuiderAppState::Stopped).await);
        let ticket = rig.send(start).await.unwrap();
        assert!(rig.wait_state(GuiderAppState::Calibrating).await);
        rig.mock.send_event(
            json!({"Event": "CalibrationFailed", "Reason": "star did not move enough"}),
        );
        let OpOutcome::Failed(err) = outcome(ticket).await else {
            panic!("a failed calibration fails the start");
        };
        assert!(err.message.contains("star did not move enough"));
    }

    #[tokio::test]
    async fn stop_polls_until_stopped_and_times_out_when_phd2_keeps_guiding() {
        let rig = Rig::start(guiding()).await;
        let ticket = rig.send(GuiderCmd::Stop).await.unwrap();
        assert_eq!(outcome(ticket).await, OpOutcome::Succeeded);
        assert_eq!(
            rig.handle().snapshot().app_state.value,
            Some(GuiderAppState::Stopped)
        );
        assert_eq!(rig.events_named("guiding_stopped").len(), 1);
        // A PHD2 that acknowledges but keeps guiding.
        rig.mock.with_state(|s| {
            s.app_state = "Guiding".into();
            s.stop_ignored = true;
        });
        rig.mock.send_event(Phd2Mock::guide_step(1, 0.1, 0.1));
        assert!(rig.wait_state(GuiderAppState::Guiding).await);
        let polls = rig.mock.method_count("get_app_state");
        let mut ticket = rig.send(GuiderCmd::Stop).await.unwrap();
        assert!(rig.mock.wait_for_count("stop_capture", 2, WAIT).await);
        for _ in 0..9 {
            rig.advance(Duration::from_secs(1)).await;
        }
        assert!(
            tokio::time::timeout(Duration::from_millis(50), &mut ticket.done)
                .await
                .is_err()
        );
        assert!(
            rig.mock.method_count("get_app_state") >= polls + 5,
            "polled every second"
        );
        rig.advance(Duration::from_secs(2)).await;
        assert_eq!(outcome(ticket).await, OpOutcome::TimedOut);
    }

    #[tokio::test]
    async fn pause_and_resume_wait_for_the_state_and_tag_paused_steps() {
        let rig = Rig::start(guiding()).await;
        let ticket = rig.send(GuiderCmd::Pause { full: true }).await.unwrap();
        assert_eq!(outcome(ticket).await, OpOutcome::Succeeded);
        assert_eq!(
            rig.mock.last_request("set_paused").unwrap().params,
            json!([true, "full"])
        );
        assert_eq!(
            rig.handle().snapshot().app_state.value,
            Some(GuiderAppState::Paused)
        );
        assert_eq!(rig.events_named("guider_paused").len(), 1);
        rig.mock.send_event(Phd2Mock::guide_step(1, 0.1, 0.1));
        assert!(rig.wait_frame(1).await);
        let ticket = rig.send(GuiderCmd::Resume).await.unwrap();
        assert_eq!(outcome(ticket).await, OpOutcome::Succeeded);
        assert_eq!(
            rig.mock.last_request("set_paused").unwrap().params,
            json!([false])
        );
        assert_eq!(rig.events_named("guider_resumed").len(), 1);
        assert_eq!(
            rig.handle().snapshot().app_state.value,
            Some(GuiderAppState::Guiding)
        );
        rig.mock.send_event(Phd2Mock::guide_step(2, 0.1, 0.1));
        assert!(rig.wait_frame(2).await);
        let phases: Vec<GuidePhase> = rig.step_records().iter().map(|r| r.step.phase).collect();
        assert_eq!(phases, vec![GuidePhase::Paused, GuidePhase::Guiding]);
    }

    #[tokio::test]
    async fn refresh_rereads_the_session_context() {
        let rig = Rig::start(MockState::default()).await;
        rig.mock.with_state(|s| {
            s.pixel_scale = Some(2.25);
            s.exposure_ms = 1500;
        });
        let ticket = rig.send(GuiderCmd::Refresh).await.unwrap();
        assert_eq!(outcome(ticket).await, OpOutcome::Succeeded);
        assert!(
            rig.wait_snapshot(|s| s.pixel_scale_arcsec == Some(2.25) && s.exposure_ms == Some(1500))
                .await
        );
    }

    #[tokio::test]
    async fn get_app_state_reconciles_the_state_every_five_seconds() {
        let rig = Rig::start(MockState::default()).await;
        let before = rig.mock.method_count("get_app_state");
        // PHD2 changed state without an event Achernar saw.
        rig.mock.with_state(|s| s.app_state = "Looping".into());
        rig.advance(Duration::from_secs(4)).await;
        assert_eq!(rig.mock.method_count("get_app_state"), before);
        rig.advance(Duration::from_secs(1)).await;
        assert!(rig.wait_state(GuiderAppState::Looping).await);
        assert_eq!(rig.handle().snapshot().app_state.source, Source::Property);
        assert!(rig.events_named("guider_state").iter().any(|e| matches!(
            e.event,
            EventKind::GuiderState {
                from: GuiderAppState::Stopped,
                to: GuiderAppState::Looping
            }
        )));
    }

    #[tokio::test]
    async fn an_unresponsive_phd2_degrades_then_loses_the_link_and_recovers() {
        let rig = Rig::start(MockState::default()).await;
        rig.mock.with_state(|s| {
            s.silent_methods.insert("get_app_state".into());
        });
        for n in 1..=3 {
            rig.advance(Duration::from_secs(5)).await;
            assert!(rig.mock.wait_for_count("get_app_state", n, WAIT).await);
            rig.advance(Duration::from_secs(10)).await;
            let expected = if n < 3 {
                LinkState::Degraded
            } else {
                LinkState::Lost
            };
            assert!(
                rig.wait_health(expected).await,
                "after {n} unanswered requests"
            );
            assert_eq!(rig.handle().health().consecutive_failures, n as u32);
        }
        assert_eq!(rig.events_named("device_degraded").len(), 1);
        assert!(matches!(
            rig.events_named("device_lost")[0].event,
            EventKind::DeviceLost {
                class: FailureClass::Timeout,
                ..
            }
        ));
        assert_eq!(rig.handle().snapshot().app_state.state, ObsState::Stale);
        // PHD2 answers again.
        rig.mock.with_state(|s| s.silent_methods.clear());
        rig.advance(Duration::from_secs(5)).await;
        assert!(rig.wait_health(LinkState::Ready).await);
        assert!(rig.wait_event("device_reconnected", 1).await);
        assert!(
            rig.closed_records().is_empty(),
            "the TCP connection never closed"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn abort_stops_capture_out_of_band_and_resolves_the_operation() {
        let mut state = guiding();
        state.dither_script.done = None;
        let rig = Rig::start(state).await;
        let ticket = rig.send(dither_cmd(3.0)).await.unwrap();
        assert!(rig.mock.wait_for_request("dither", WAIT).await);
        let abort = rig.handle().abort();
        let abort_id = abort.command_id;
        assert!(abort.abort_path_ran && abort.cancel_sent);
        assert!(tokio::time::timeout(WAIT, abort.outcome())
            .await
            .unwrap()
            .is_ok());
        assert_eq!(
            outcome(ticket).await,
            OpOutcome::Aborted { by: Some(abort_id) }
        );
        // Out of band (the abort path) and in band (the service's cancel).
        assert_eq!(rig.mock.method_count("stop_capture"), 2);
        assert_eq!(rig.mock.state().app_state, "Stopped");
        assert_eq!(
            rig.dither_records()[0].settle_error.as_deref(),
            Some("aborted")
        );
        assert!(rig.wait_state(GuiderAppState::Stopped).await);
    }

    /// The default `/connect` host is `localhost`, which usually resolves to
    /// `::1` first; PHD2 (and the mock) listen on IPv4 only.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn abort_reaches_phd2_when_the_host_is_localhost() {
        let mut state = guiding();
        state.dither_script.done = None;
        let rig = Rig::start_with(state, |s| s.host = "localhost".into()).await;
        assert_eq!(
            rig.handle().identity().endpoint,
            Endpoint::Phd2 {
                host: "localhost".into(),
                port: rig.mock.port()
            }
        );
        let ticket = rig.send(dither_cmd(3.0)).await.unwrap();
        assert!(rig.mock.wait_for_request("dither", WAIT).await);
        let abort = rig.handle().abort();
        let abort_id = abort.command_id;
        let result = tokio::time::timeout(WAIT, abort.outcome()).await.unwrap();
        assert!(
            result.is_ok(),
            "the out-of-band stop reached PHD2: {result:?}"
        );
        assert_eq!(
            outcome(ticket).await,
            OpOutcome::Aborted { by: Some(abort_id) }
        );
        assert_eq!(rig.mock.method_count("stop_capture"), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_abort_resolves_only_once_phd2_reports_stopped() {
        let mut state = guiding();
        state.dither_script.done = None;
        state.stop_ignored = true;
        let rig = Rig::start(state).await;
        let ticket = rig.send(dither_cmd(3.0)).await.unwrap();
        assert!(rig.mock.wait_for_request("dither", WAIT).await);
        let abort = rig.handle().abort();
        let abort_id = abort.command_id;
        assert!(tokio::time::timeout(WAIT, abort.outcome()).await.is_ok());
        assert!(rig.mock.wait_for_count("stop_capture", 2, WAIT).await);
        assert!(
            rig.wait_snapshot(|s| s
                .active_op
                .as_ref()
                .is_some_and(|op| op.phase == OpPhase::Cancelling))
                .await
        );
        // PHD2 acknowledged both stops but is still guiding: not Aborted.
        let mut ticket = ticket;
        rig.advance(Duration::from_secs(3)).await;
        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut ticket.done)
                .await
                .is_err(),
            "no Aborted before PHD2 stops"
        );
        // PHD2 stops: now Aborted { by }.
        rig.mock.with_state(|s| s.app_state = "Stopped".into());
        rig.mock.send_event(json!({"Event": "GuidingStopped"}));
        assert_eq!(
            outcome(ticket).await,
            OpOutcome::Aborted { by: Some(abort_id) }
        );
        assert_eq!(
            rig.dither_records()[0].settle_error.as_deref(),
            Some("aborted")
        );

        // A PHD2 that never stops: the outcome is unknown after 10 s.
        rig.mock.with_state(|s| s.app_state = "Guiding".into());
        rig.mock.send_event(Phd2Mock::guide_step(1, 0.1, 0.1));
        assert!(rig.wait_state(GuiderAppState::Guiding).await);
        let ticket = rig.send(dither_cmd(3.0)).await.unwrap();
        assert!(rig.mock.wait_for_count("dither", 2, WAIT).await);
        let abort = rig.handle().abort();
        assert!(tokio::time::timeout(WAIT, abort.outcome()).await.is_ok());
        assert!(rig.mock.wait_for_count("stop_capture", 4, WAIT).await);
        for _ in 0..11 {
            rig.advance(Duration::from_secs(1)).await;
        }
        let OpOutcome::Unknown(err) = outcome(ticket).await else {
            panic!("PHD2 never stopped: the outcome is unknown, not aborted");
        };
        assert_eq!(err.class, ErrorClass::OutcomeUnknown);
        assert!(err.message.contains("did not report Stopped"), "{err:?}");
        assert_eq!(rig.events_named("outcome_unknown").len(), 1);
        assert!(rig.dither_records()[1]
            .settle_error
            .as_deref()
            .unwrap()
            .contains("did not report Stopped"));
    }

    #[tokio::test]
    async fn nothing_is_emitted_or_recorded_until_start() {
        let mock = Phd2Mock::start_with(guiding()).await;
        let clock = Arc::new(ManualClock::new());
        let sink = RecordingSink::new();
        let events = Arc::new(Mutex::new(Vec::new()));
        let id = Uuid::new_v4();
        let mut conn = connect_with_id(
            Phd2Settings::new("127.0.0.1", mock.port()),
            context(&clock, &sink, &events),
            id,
        )
        .await
        .unwrap();
        // The caller's id, so the hub can register it before any event.
        assert_eq!(conn.connection_id(), id);
        assert_eq!(conn.handle.connection_id(), id);
        assert_eq!(conn.handle.identity().connection_id, id);
        assert_eq!(conn.windows.connection_id(), id);
        assert!(!conn.is_started());
        // PHD2 events that arrive before start wait in the queue.
        mock.send_event(Phd2Mock::guide_step(1, 0.1, 0.1));
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(lock(&events).is_empty(), "nothing before start");
        assert!(sink.is_empty(), "no device_connections row before start");
        assert!(conn.start());
        assert!(!conn.start(), "start runs once");
        let deadline = tokio::time::Instant::now() + WAIT;
        while sink.records().len() < 2 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let names: Vec<&str> = lock(&events).iter().map(EquipmentEvent::name).collect();
        assert_eq!(names[0], "device_connected");
        assert_eq!(names[1], "guider_state", "the connect burst follows");
        assert!(names.contains(&"guide_step"), "{names:?}");
        assert!(lock(&events).iter().all(|e| e.connection_id == Some(id)));
        let records = sink.records();
        assert!(matches!(
            records.first(),
            Some(GroundTruthRecord::ConnectionOpened(row)) if row.identity.connection_id == id
        ));
        assert!(matches!(
            records.get(1),
            Some(GroundTruthRecord::GuideStep(_))
        ));

        // A connection that is never started closes its socket when dropped.
        let before = mock.client_count();
        let unstarted = connect(
            Phd2Settings::new("127.0.0.1", mock.port()),
            context(&clock, &sink, &events),
        )
        .await
        .unwrap();
        assert_eq!(mock.client_count(), before + 1);
        let recorded = sink.records().len();
        drop(unstarted);
        let deadline = tokio::time::Instant::now() + WAIT;
        while mock.client_count() > before && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(mock.client_count(), before);
        assert_eq!(sink.records().len(), recorded, "nothing recorded");
    }

    #[test]
    fn a_connection_can_move_between_threads_and_be_shared() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Phd2Connection>();
    }

    #[tokio::test]
    async fn disconnect_never_disconnects_phd2_equipment_and_retires_the_handle() {
        let mut rig = Rig::start(guiding()).await;
        rig.handle().disconnect(false).await.unwrap();
        assert!(rig.handle().is_retired());
        assert_eq!(rig.handle().health().state, LinkState::Disconnected);
        assert_eq!(rig.mock.method_count("set_connected"), 0);
        assert_eq!(
            rig.mock.method_count("stop_capture"),
            0,
            "PHD2 keeps guiding"
        );
        assert_eq!(rig.closed_records(), vec![ConnectionEnd::Operator]);
        assert!(matches!(
            rig.events_named("device_disconnected")[0].event,
            EventKind::DeviceDisconnected {
                end_class: ConnectionEnd::Operator,
                ..
            }
        ));
        let err = rig.send(GuiderCmd::Refresh).await.unwrap_err();
        assert_eq!(err.class, ErrorClass::NotConnected);
        let task = rig.conn.task_mut().expect("started");
        assert!(tokio::time::timeout(WAIT, task).await.is_ok());
    }

    #[tokio::test]
    async fn aborting_the_service_task_closes_the_socket() {
        let mut rig = Rig::start(MockState::default()).await;
        assert_eq!(rig.mock.client_count(), 1);
        rig.conn.abort_task();
        let deadline = tokio::time::Instant::now() + WAIT;
        while rig.mock.client_count() > 0 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(rig.mock.client_count(), 0, "no socket outlives the service");
        let err = rig.send(GuiderCmd::Refresh).await.unwrap_err();
        assert_eq!(err.class, ErrorClass::Disconnected);
        assert_eq!(rig.mock.method_count("set_connected"), 0);
    }

    #[tokio::test]
    async fn connect_fails_cleanly_without_a_usable_phd2() {
        let clock = Arc::new(ManualClock::new());
        let sink = RecordingSink::new();
        let events = Arc::new(Mutex::new(Vec::new()));
        let closed = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        };
        let err = connect(
            Phd2Settings::new("127.0.0.1", closed),
            context(&clock, &sink, &events),
        )
        .await
        .err()
        .unwrap();
        assert_eq!(err.class, ErrorClass::Disconnected);

        // Listening, but no burst and no answers.
        let mut state = MockState {
            send_connect_burst: false,
            ..MockState::default()
        };
        state.silent_methods.insert("get_app_state".into());
        let mock = Phd2Mock::start_with(state).await;
        let ctx = context(&clock, &sink, &events);
        let attempt = tokio::spawn(connect(Phd2Settings::new("127.0.0.1", mock.port()), ctx));
        // Step the clock until the burst wait expires and get_app_state is
        // sent, then until that request times out. Small steps keep the
        // test independent of how fast the connect task runs.
        let mut steps = 0;
        while mock.method_count("get_app_state") == 0 && steps < 400 {
            clock.advance(Duration::from_millis(500));
            tokio::time::sleep(Duration::from_millis(10)).await;
            steps += 1;
        }
        assert_eq!(mock.method_count("get_app_state"), 1);
        while !attempt.is_finished() && steps < 800 {
            clock.advance(Duration::from_millis(500));
            tokio::time::sleep(Duration::from_millis(10)).await;
            steps += 1;
        }
        let err = tokio::time::timeout(WAIT, attempt)
            .await
            .unwrap()
            .unwrap()
            .err()
            .unwrap();
        assert_eq!(err.class, ErrorClass::Timeout);
        assert!(sink.is_empty(), "no connection row for a failed connect");
        assert!(lock(&events).is_empty());
    }

    #[tokio::test]
    async fn missing_algorithm_parameters_disable_actuation_with_a_reason() {
        let mut state = MockState {
            algo_names_ra: vec!["controlGain".into(), "minMove".into()],
            algo_names_dec: vec!["minMove".into()],
            ..MockState::default()
        };
        state
            .algo_values
            .insert(("ra".into(), "controlGain".into()), 0.8);
        let rig = Rig::start(state).await;
        let snap = rig.handle().snapshot();
        assert_eq!(snap.aggressiveness_param_ra, None);
        assert_eq!(
            snap.actuation_disabled_reason.as_deref(),
            Some("no_aggressiveness_param")
        );
        assert_eq!(
            rig.conn.capabilities["actuation_disabled_reason"],
            json!("no_aggressiveness_param")
        );

        let mut state = MockState::default();
        state
            .unsupported_methods
            .insert("get_algo_param_names".into());
        let rig = Rig::start(state).await;
        assert!(rig
            .conn
            .report
            .unsupported_members
            .contains(&"get_algo_param_names".to_string()));
        assert!(rig
            .events_named("capability_unsupported")
            .iter()
            .any(|e| matches!(
                &e.event,
                EventKind::CapabilityUnsupported { member } if member == "get_algo_param_names"
            )));
        assert_eq!(
            rig.handle().snapshot().actuation_disabled_reason.as_deref(),
            Some("no_aggressiveness_param")
        );
    }
}

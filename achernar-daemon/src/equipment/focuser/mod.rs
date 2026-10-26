//! Focuser behaviour shared by every focuser driver (spec §3.3, F-a to F-i).
//!
//! Drivers (`alpaca/focuser.rs`, `sim/focuser.rs`, the legacy adapters and,
//! in Tier 2, the INDI focuser) implement
//! [`FocuserDriver`](crate::equipment::traits::FocuserDriver): the raw device
//! position, IsMoving, Temperature and TempComp. Everything above the raw
//! device lives here and is identical for every driver:
//!
//! - [`FocuserSession`]: one per connection. It owns the logical position
//!   (backlash compensation and relative emulation hidden), the
//!   [`PositionSource`], the identity profile, the temperature stream, and the
//!   snapshot the worker publishes after each poll. It builds the operations.
//! - [`motion`]: the move, halt, TempComp and sync operations the worker
//!   drives (F-a, F-b, F-c, F-h).
//! - [`backlash`]: the `none`, `overshoot` and `absolute` models (F-e).
//! - [`tempcomp`]: TempComp off during moves and restored afterwards (F-d).
//! - [`FocuserSafetyCheck`]: the gate plug-in (§2.9 step 3).
//!
//! # Integration (worker WP1A-2, spine WP1B-1)
//!
//! 1. Build the driver and call [`FocuserSession::connect`] on the worker
//!    thread. It connects the driver, probes capabilities, resolves the
//!    settings from `[focuser]`, `[focus.safety]`, `[focus.backlash]` and the
//!    identity profile (`device_profiles.profile_json`), restores a relative
//!    focuser's position when the profile row was closed cleanly (§2.15),
//!    and runs the F-d, F-g and F-i post-connect steps.
//! 2. Poll with [`FocuserSession::poll_plan`] (TempComp every 10 s) at
//!    [`IDLE_POLL_INTERVAL`] or, while an operation runs,
//!    [`ACTIVE_POLL_INTERVAL`]; after every poll publish
//!    [`FocuserSession::after_poll`].
//! 3. Turn each accepted [`FocuserCmd`] into an operation with
//!    [`FocuserSession::operation`] and drive it (`start`, `step` after each
//!    poll, `cancel` on abort, preemption or deadline).
//! 4. On a reconnect call [`FocuserSession::on_reconnect`]; on disconnect
//!    [`FocuserSession::on_disconnect`]. `GatedOp::FocuserSetPosition` is
//!    executed with [`FocuserSession::set_position`] (profile only).
//! 5. Register [`FocuserSafetyCheck`] with the gate once.
//!
//! Events and ground-truth records (`focuser_moves`, `environment_log` probe
//! rows, `device_profiles` patches) go to the session's
//! [`RecorderSink`](crate::equipment::traits::RecorderSink) as
//! [`GroundTruthRecord`]s. The hub's sink is expected to sequence and
//! broadcast `GroundTruthRecord::Event` as well as persist it, because the
//! frozen `Operation` trait has no event channel of its own.

pub mod backlash;
pub mod motion;
pub mod tempcomp;

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

use crate::config::TempCompDuringMoves;
use crate::config::{BacklashConfig, EquipmentConfigs, FocusSafetyConfig, FocuserConfig};
use crate::equipment::commands::{FocuserCmd, GatedCommand, GatedOp, SafetyRejection};
use crate::equipment::events::{
    ConnectionEnd, EnvironmentRecord, EquipmentEvent, EventKind, GroundTruthRecord, ProfileUpdate,
};
use crate::equipment::ops::{ActiveOp, Operation};
use crate::equipment::snapshot::{
    BacklashModel, BacklashOwner, EquipmentSnapshot, FocusDirection, FocuserCapabilities,
    FocuserSnapshot, PositionSource,
};
use crate::equipment::traits::{
    CameraActivity, Clock, ConnectReport, FocuserDriver, PollPlan, RawPoll, RecorderSink,
    SafetyCheck,
};
use crate::equipment::types::{
    CommandId, CommandSource, DeviceError, DeviceIdentity, DeviceKind, DeviceResult,
    TemperatureRole,
};

pub use backlash::BacklashState;

/// Idle cadence (§2.7: 0.5 Hz).
pub const IDLE_POLL_INTERVAL: Duration = Duration::from_secs(2);
/// Cadence while an Achernar-owned focuser operation runs (§2.7: 10 Hz).
pub const ACTIVE_POLL_INTERVAL: Duration = Duration::from_millis(100);
/// The extra member read on its own schedule (TempComp is not in
/// DeviceState).
pub const TEMPCOMP_MEMBER: &str = "tempcomp";
/// TempComp refresh period (§2.7).
pub const TEMPCOMP_REFRESH: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// Identity profile (device_profiles.profile_json, kind = focuser)
// ---------------------------------------------------------------------------

/// Whether the relative-focuser position in the profile can be trusted
/// after a restart (§2.15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CloseState {
    /// Written after a completed move, or at a clean disconnect.
    Clean,
    /// A move was in progress when the row was last written.
    Moving,
    /// The position was lost (interrupted move, reconnect, controller
    /// reset).
    #[default]
    Unknown,
}

/// The persisted logical position of a relative focuser.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelativePosition {
    pub logical: Option<i32>,
    pub state: CloseState,
    pub updated_at: Option<DateTime<Utc>>,
}

/// Operator travel bounds for this device, in logical steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TravelBounds {
    pub min: Option<i32>,
    pub max: Option<i32>,
}

/// The `absolute` backlash model's state, kept across restarts. It is
/// reused only when the device still reports the raw position it was
/// written with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BacklashMemory {
    pub offset: i32,
    pub last_direction: Option<FocusDirection>,
    pub raw: Option<i32>,
}

/// What the optional F-i position-source Action reported.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PositionSourceReport {
    pub action: String,
    /// `homed`, `restored` or `unknown`; anything else is kept verbatim.
    pub result: String,
    pub at: DateTime<Utc>,
}

/// The focuser part of an identity profile (D17). Every key is optional;
/// unknown keys (written by other packages) are ignored, and a malformed key
/// is dropped with a warning instead of discarding the rest.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FocuserProfile {
    pub backlash: Option<BacklashModel>,
    pub backlash_owner: Option<BacklashOwner>,
    pub travel: TravelBounds,
    pub settle_ms: Option<u64>,
    pub temperature_role: Option<TemperatureRole>,
    /// Alpaca `stepsize`, recorded as a driver claim (F-g). The V-curve
    /// `steps_per_micron` stays authoritative.
    pub step_size_um_claim: Option<f64>,
    pub relative_position: Option<RelativePosition>,
    pub backlash_state: Option<BacklashMemory>,
    pub position_source_action: Option<PositionSourceReport>,
}

fn profile_field<T: DeserializeOwned>(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    warnings: &mut Vec<String>,
) -> Option<T> {
    let value = object.get(key)?;
    if value.is_null() {
        return None;
    }
    match serde_json::from_value(value.clone()) {
        Ok(parsed) => Some(parsed),
        Err(err) => {
            warnings.push(format!("focuser profile key {key:?} ignored: {err}"));
            None
        }
    }
}

impl FocuserProfile {
    /// Parse `device_profiles.profile_json`. Returns the profile and any
    /// warnings about keys that could not be read.
    pub fn from_json(value: Option<&serde_json::Value>) -> (Self, Vec<String>) {
        let mut warnings = Vec::new();
        let Some(value) = value else {
            return (Self::default(), warnings);
        };
        let Some(object) = value.as_object() else {
            if !value.is_null() {
                warnings.push("focuser profile is not a JSON object; ignored".into());
            }
            return (Self::default(), warnings);
        };
        let profile = Self {
            backlash: profile_field(object, "backlash", &mut warnings),
            backlash_owner: profile_field(object, "backlash_owner", &mut warnings),
            travel: profile_field(object, "travel", &mut warnings).unwrap_or_default(),
            settle_ms: profile_field(object, "settle_ms", &mut warnings),
            temperature_role: profile_field(object, "temperature_role", &mut warnings),
            step_size_um_claim: profile_field(object, "step_size_um_claim", &mut warnings),
            relative_position: profile_field(object, "relative_position", &mut warnings),
            backlash_state: profile_field(object, "backlash_state", &mut warnings),
            position_source_action: profile_field(object, "position_source_action", &mut warnings),
        };
        (profile, warnings)
    }

    /// The restart rule (§2.15, F-b): a relative focuser's logical position
    /// is restored only from a row closed cleanly after a completed move.
    pub fn restored_relative_position(&self) -> Option<i32> {
        self.relative_position
            .as_ref()
            .filter(|rel| rel.state == CloseState::Clean)
            .and_then(|rel| rel.logical)
    }
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// Behaviour of one focuser connection, resolved from the TOML defaults and
/// the identity profile (the profile wins; D17).
#[derive(Debug, Clone, PartialEq)]
pub struct FocuserSettings {
    pub stuck_timeout: Duration,
    pub max_reissue: u32,
    pub tempcomp_during_moves: TempCompDuringMoves,
    pub temperature_role: TemperatureRole,
    pub position_source_action: Option<String>,
    /// Per-leg budget (`[focus.safety] move_timeout_ms`, C-4).
    pub move_timeout: Duration,
    pub settle: Duration,
    pub arrival_tolerance_steps: i32,
    /// `FocusSafetyGuard` travel bounds (`[focus.safety] min_position`,
    /// `max_position`).
    pub guard_min: i32,
    pub guard_max: i32,
    /// Largest relative move allowed while the origin is unknown.
    pub max_step_single: i32,
    pub profile_bounds: TravelBounds,
    pub backlash_model: BacklashModel,
    pub backlash_owner: BacklashOwner,
    /// `focuser_temperature` events: at least this often while readings
    /// arrive...
    pub temperature_event_interval: Duration,
    /// ...or sooner on a change of at least this much...
    pub temperature_event_delta_c: f64,
    /// ...but never more often than this.
    pub temperature_event_min_gap: Duration,
    /// `environment_log` probe rows (ambient role only).
    pub environment_interval: Duration,
}

impl Default for FocuserSettings {
    fn default() -> Self {
        Self::resolve(
            &FocuserConfig::default(),
            &FocusSafetyConfig::default(),
            &BacklashConfig::default(),
            &FocuserProfile::default(),
        )
    }
}

impl FocuserSettings {
    pub fn resolve(
        focuser: &FocuserConfig,
        safety: &FocusSafetyConfig,
        backlash_cfg: &BacklashConfig,
        profile: &FocuserProfile,
    ) -> Self {
        let action = focuser.position_source_action.trim();
        Self {
            stuck_timeout: Duration::from_secs(focuser.stuck_timeout_s.max(1)),
            max_reissue: focuser.max_reissue,
            tempcomp_during_moves: focuser.tempcomp_during_moves,
            temperature_role: profile.temperature_role.unwrap_or(focuser.temperature_role),
            position_source_action: (!action.is_empty()).then(|| action.to_string()),
            move_timeout: Duration::from_millis(safety.move_timeout_ms.max(100)),
            settle: Duration::from_millis(profile.settle_ms.unwrap_or(safety.settle_ms)),
            arrival_tolerance_steps: safety.arrival_tolerance_steps.max(0),
            guard_min: safety.min_position,
            guard_max: safety.max_position,
            max_step_single: safety.max_step_single.saturating_abs().max(1),
            profile_bounds: profile.travel,
            backlash_model: profile
                .backlash
                .unwrap_or_else(|| backlash::model_from_config(backlash_cfg)),
            backlash_owner: profile.backlash_owner.unwrap_or_default(),
            temperature_event_interval: Duration::from_secs(60),
            temperature_event_delta_c: 0.5,
            temperature_event_min_gap: Duration::from_secs(10),
            environment_interval: Duration::from_secs(60),
        }
    }

    pub fn from_configs(configs: &EquipmentConfigs, profile: &FocuserProfile) -> Self {
        let mut settings = Self::resolve(
            &configs.focuser,
            &configs.focus_safety,
            &configs.focus_backlash,
            profile,
        );
        settings.environment_interval = Duration::from_secs(configs.weather.log_interval_s.max(1));
        settings
    }

    /// Logical travel bounds: the guard bounds, the profile bounds, and for
    /// an absolute focuser `[0, MaxStep]`. `None` when they are empty.
    pub fn travel_bounds(&self, caps: &FocuserCapabilities) -> Option<(i32, i32)> {
        let mut lo = self.guard_min;
        let mut hi = self.guard_max;
        if let Some(min) = self.profile_bounds.min {
            lo = lo.max(min);
        }
        if let Some(max) = self.profile_bounds.max {
            hi = hi.min(max);
        }
        if caps.absolute {
            lo = lo.max(0);
            if let Some(max_step) = caps.max_step.filter(|m| *m > 0) {
                hi = hi.min(max_step);
            }
        }
        (lo <= hi).then_some((lo, hi))
    }

    /// The backlash model in force (`none` when the firmware owns it).
    pub fn effective_backlash(&self) -> BacklashModel {
        backlash::effective_model(self.backlash_model, self.backlash_owner)
    }
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// What a session needs from the hub.
#[derive(Clone)]
pub struct FocuserDeps {
    pub clock: Arc<dyn Clock>,
    pub sink: Arc<dyn RecorderSink>,
    /// The camera hook, re-checked when a move starts (A06).
    pub camera: Option<Arc<dyn CameraActivity>>,
}

#[derive(Debug, Clone, Default)]
struct LogicalState {
    source: PositionSource,
    /// Published logical position; `None` while the origin is unknown.
    logical: Option<i32>,
    backlash: BacklashState,
    settling: bool,
    op_active: bool,
    /// Raw position at the last idle poll, for external-change detection.
    last_idle_raw: Option<i32>,
    position_lost: bool,
    /// Mono time and value of the last `focuser_temperature` event.
    temperature_event: Option<(u64, f64)>,
    environment_mono: Option<u64>,
}

struct SessionInner {
    identity_key: String,
    connection_id: Uuid,
    settings: FocuserSettings,
    caps: FocuserCapabilities,
    clock: Arc<dyn Clock>,
    sink: Arc<dyn RecorderSink>,
    camera: Option<Arc<dyn CameraActivity>>,
    state: Mutex<LogicalState>,
}

impl SessionInner {
    fn lock(&self) -> MutexGuard<'_, LogicalState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn event(&self, kind: EventKind, command_id: Option<CommandId>) {
        let mut event = EquipmentEvent::new(kind, self.clock.as_ref())
            .for_device(DeviceKind::Focuser, self.connection_id);
        if let Some(command_id) = command_id {
            event = event.with_command(command_id);
        }
        self.sink.record(GroundTruthRecord::Event(Arc::new(event)));
    }

    fn patch_profile(&self, patch: serde_json::Value) {
        self.sink.record(GroundTruthRecord::Profile(ProfileUpdate {
            identity_key: self.identity_key.clone(),
            kind: DeviceKind::Focuser,
            patch,
            updated_at: self.clock.now_utc(),
        }));
    }

    fn persist_relative(&self, logical: Option<i32>, state: CloseState) {
        let rel = RelativePosition {
            logical,
            state,
            updated_at: Some(self.clock.now_utc()),
        };
        self.patch_profile(json!({ "relative_position": rel }));
    }

    fn persist_backlash(&self, backlash: BacklashState, raw: Option<i32>) {
        let memory = BacklashMemory {
            offset: backlash.offset,
            last_direction: backlash.last_direction,
            raw,
        };
        self.patch_profile(json!({ "backlash_state": memory }));
    }

    fn travel_bounds(&self) -> Option<(i32, i32)> {
        self.settings.travel_bounds(&self.caps)
    }

    /// A06 defence in depth: the gate already refused moves while exposing;
    /// this closes the gap between the gate and the worker.
    fn camera_exposing(&self) -> bool {
        self.camera
            .as_ref()
            .is_some_and(|camera| camera.exposing().known().copied() == Some(true))
    }
}

/// One focuser connection's behaviour (see the module docs).
#[derive(Clone)]
pub struct FocuserSession {
    inner: Arc<SessionInner>,
}

/// The result of [`FocuserSession::connect`].
pub struct FocuserConnection {
    pub report: ConnectReport,
    pub session: FocuserSession,
    pub capabilities: FocuserCapabilities,
    /// Profile and post-connect findings; none of them refuses the connect.
    pub warnings: Vec<String>,
}

impl FocuserSession {
    pub fn new(
        identity: &DeviceIdentity,
        capabilities: FocuserCapabilities,
        settings: FocuserSettings,
        profile: &FocuserProfile,
        deps: FocuserDeps,
    ) -> Self {
        let mut state = LogicalState::default();
        if capabilities.absolute {
            state.source = PositionSource::Device;
            if let Some(memory) = profile.backlash_state {
                // Verified against the raw position in `post_connect`.
                state.backlash = BacklashState {
                    offset: memory.offset,
                    last_direction: memory.last_direction,
                };
            }
        } else if let Some(position) = profile.restored_relative_position() {
            state.source = PositionSource::EmulatedRestored;
            state.logical = Some(position);
        } else {
            state.source = PositionSource::EmulatedOriginUnknown;
        }
        Self {
            inner: Arc::new(SessionInner {
                identity_key: identity.identity_key.clone(),
                connection_id: identity.connection_id,
                settings,
                caps: capabilities,
                clock: deps.clock,
                sink: deps.sink,
                camera: deps.camera,
                state: Mutex::new(state),
            }),
        }
    }

    /// The connect sequence for a focuser (§2.6 steps 4-8): connect,
    /// capabilities, settings and profile, the initial poll, then F-d, F-g and
    /// F-i. On a failure after the driver connected, the driver is
    /// disconnected again and the error returned.
    pub fn connect(
        drv: &mut dyn FocuserDriver,
        timeout: Duration,
        profile_json: Option<&serde_json::Value>,
        configs: &EquipmentConfigs,
        deps: FocuserDeps,
    ) -> DeviceResult<FocuserConnection> {
        let report = drv.connect(timeout)?;
        let capabilities = match drv.capabilities() {
            Ok(caps) => caps,
            Err(err) => {
                let _ = drv.disconnect();
                return Err(err);
            }
        };
        let (profile, mut warnings) = FocuserProfile::from_json(profile_json);
        let settings = FocuserSettings::from_configs(configs, &profile);
        if settings.travel_bounds(&capabilities).is_none() {
            warnings.push(
                "focuser travel bounds are empty ([focus.safety] and the profile disagree); \
                 every move will be refused"
                    .into(),
            );
        }
        let identity = drv.identity().clone();
        let session = Self::new(&identity, capabilities.clone(), settings, &profile, deps);
        warnings.extend(session.post_connect(drv, &profile));
        Ok(FocuserConnection {
            report,
            session,
            capabilities,
            warnings,
        })
    }

    /// The first poll after connect reads TempComp too.
    pub fn initial_poll_plan() -> PollPlan {
        PollPlan {
            tick: 0,
            interval: IDLE_POLL_INTERVAL,
            active: false,
            prefer_device_state: true,
            extra_members: vec![TEMPCOMP_MEMBER.to_string()],
        }
    }

    /// The plan for one tick: `active` while an operation runs (10 Hz);
    /// TempComp when `tempcomp_due` (every [`TEMPCOMP_REFRESH`]).
    pub fn poll_plan(tick: u64, active: bool, tempcomp_due: bool) -> PollPlan {
        PollPlan {
            tick,
            interval: if active {
                ACTIVE_POLL_INTERVAL
            } else {
                IDLE_POLL_INTERVAL
            },
            active,
            prefer_device_state: true,
            extra_members: if tempcomp_due {
                vec![TEMPCOMP_MEMBER.to_string()]
            } else {
                Vec::new()
            },
        }
    }

    /// Post-connect steps F-d (TempComp read), F-g (StepSize claim), F-i
    /// (optional position-source Action) and the backlash-memory check.
    /// Returns warnings; nothing here refuses the connection.
    pub fn post_connect(
        &self,
        drv: &mut dyn FocuserDriver,
        profile: &FocuserProfile,
    ) -> Vec<String> {
        let inner = &self.inner;
        let mut warnings = Vec::new();
        if let Err(err) = drv.poll(&Self::initial_poll_plan()) {
            warnings.push(format!("initial focuser poll failed: {err}"));
        }
        let state = drv.state().unwrap_or_default();

        // F-g: StepSize is a driver claim, recorded in the profile.
        if let Some(step) = inner
            .caps
            .step_size_um
            .filter(|s| s.is_finite() && *s > 0.0)
        {
            if profile.step_size_um_claim != Some(step) {
                inner.patch_profile(json!({ "step_size_um_claim": step }));
            }
        }

        // The absolute model's offset is reused only if the device still
        // reports the raw position it was written with.
        if inner.caps.absolute {
            let raw = state.position.fresh_value().copied();
            let mut reset = None;
            {
                let mut s = inner.lock();
                if let Some(memory) = profile.backlash_state {
                    let matches = memory.raw.is_some() && memory.raw == raw;
                    if !matches && s.backlash != BacklashState::default() {
                        reset = Some((raw.map(|r| r - s.backlash.offset), raw));
                        s.backlash = BacklashState::default();
                    }
                }
                s.logical = raw.map(|r| r - s.backlash.offset);
                s.last_idle_raw = raw;
            }
            if let Some((from, to)) = reset {
                inner.event(
                    EventKind::FocuserPositionReset {
                        from,
                        to,
                        cause: "backlash_offset_reset".into(),
                    },
                    None,
                );
                inner.persist_backlash(BacklashState::default(), raw);
            }
        }

        // F-i: an optional Alpaca Action that reports homed|restored|unknown.
        if let Some(action) = inner.settings.position_source_action.clone() {
            match drv.supported_actions() {
                Ok(actions) if actions.iter().any(|a| a.eq_ignore_ascii_case(&action)) => {
                    match drv.action(&action, "") {
                        Ok(reply) => {
                            let result = reply.trim().to_ascii_lowercase();
                            let report = PositionSourceReport {
                                action: action.clone(),
                                result: result.clone(),
                                at: inner.clock.now_utc(),
                            };
                            inner.patch_profile(json!({ "position_source_action": report }));
                            if !matches!(result.as_str(), "homed" | "restored") {
                                warnings.push(format!(
                                    "position source action {action} reports {result:?}; FDF \
                                     and FOP re-baseline"
                                ));
                                let position = inner.lock().logical;
                                inner.event(
                                    EventKind::FocuserPositionReset {
                                        from: None,
                                        to: position,
                                        cause: format!("position_source_action_{result}"),
                                    },
                                    None,
                                );
                            }
                        }
                        Err(err) => {
                            warnings.push(format!("position source action {action} failed: {err}"))
                        }
                    }
                }
                Ok(_) => warnings.push(format!(
                    "position source action {action} is not in SupportedActions"
                )),
                Err(err) => warnings.push(format!("SupportedActions could not be read: {err}")),
            }
        }

        if !inner.caps.absolute && inner.lock().source == PositionSource::EmulatedOriginUnknown {
            warnings.push(
                "relative focuser origin is unknown: only relative moves up to \
                 [focus.safety] max_step_single are allowed, and autofocus is refused until \
                 the position is set"
                    .into(),
            );
        }
        warnings
    }

    /// Update the logical position, temperature stream and change detection
    /// from the poll just made, and build the snapshot to publish. Never does
    /// device I/O (it reads `drv.state()`, which only assembles the poll).
    pub fn after_poll(
        &self,
        drv: &mut dyn FocuserDriver,
        poll: Result<&RawPoll, &DeviceError>,
        active_op: Option<ActiveOp>,
    ) -> FocuserSnapshot {
        let inner = &self.inner;
        let state = drv.state().unwrap_or_default();
        let now_mono = inner.clock.now_mono_ns();
        let now_utc = inner.clock.now_utc();
        let settings = &inner.settings;
        let mut events = Vec::new();
        let mut environment = None;

        let position_lost = poll.ok().is_some_and(|poll| {
            poll.members.iter().any(|m| {
                m.name == "position" && m.error.as_deref().is_some_and(announces_lost_position)
            })
        });

        let mut s = inner.lock();
        if inner.caps.absolute {
            if position_lost && !s.position_lost {
                s.position_lost = true;
                events.push(EventKind::FocuserPositionReset {
                    from: s.logical,
                    to: None,
                    cause: "driver_position_unknown".into(),
                });
                s.logical = None;
                s.last_idle_raw = None;
                s.backlash.last_direction = None;
            }
            if let Some(raw) = state.position.fresh_value().copied() {
                let tempcomp_on = state.temp_comp.known().copied() == Some(true);
                if !s.op_active && !tempcomp_on && !s.position_lost {
                    if let Some(prev) = s.last_idle_raw {
                        if (raw - prev).abs() > settings.arrival_tolerance_steps {
                            events.push(EventKind::FocuserPositionReset {
                                from: Some(prev - s.backlash.offset),
                                to: Some(raw - s.backlash.offset),
                                cause: "external_change".into(),
                            });
                            s.backlash.last_direction = None;
                        }
                    }
                }
                s.last_idle_raw = if s.op_active { None } else { Some(raw) };
                if !position_lost {
                    s.position_lost = false;
                    s.logical = Some(raw - s.backlash.offset);
                }
            }
        }

        // F-f: temperature stream (rate-limited events; ambient probe rows).
        if let Some(celsius) = state
            .temperature_c
            .fresh_value()
            .copied()
            .filter(|t| t.is_finite())
        {
            let due = match s.temperature_event {
                None => true,
                Some((at, last)) => {
                    let gap = Duration::from_nanos(now_mono.saturating_sub(at));
                    gap >= settings.temperature_event_interval
                        || (gap >= settings.temperature_event_min_gap
                            && (celsius - last).abs() >= settings.temperature_event_delta_c)
                }
            };
            if due {
                s.temperature_event = Some((now_mono, celsius));
                events.push(EventKind::FocuserTemperature {
                    celsius,
                    role: settings.temperature_role,
                });
            }
            let env_due = s.environment_mono.is_none_or(|at| {
                Duration::from_nanos(now_mono.saturating_sub(at)) >= settings.environment_interval
            });
            if settings.temperature_role == TemperatureRole::Ambient && env_due {
                s.environment_mono = Some(now_mono);
                environment = Some(EnvironmentRecord {
                    observed_at: state.temperature_c.observed_at.unwrap_or(now_utc),
                    device_identity_key: Some(inner.identity_key.clone()),
                    ambient_temp_c: Some(celsius),
                    ..EnvironmentRecord::default()
                });
            }
        }

        let snapshot = FocuserSnapshot {
            connection_id: inner.connection_id,
            state,
            capabilities: inner.caps.clone(),
            logical_position: s.logical,
            position_source: s.source,
            backlash_model: settings.effective_backlash(),
            backlash_owner: settings.backlash_owner,
            temperature_role: settings.temperature_role,
            settling: s.settling,
            active_op,
            published_at: now_utc,
            published_mono_ns: now_mono,
        };
        drop(s);
        for event in events {
            inner.event(event, None);
        }
        if let Some(row) = environment {
            inner
                .sink
                .record(GroundTruthRecord::Environment(Box::new(row)));
        }
        snapshot
    }

    /// The operation for an accepted command. `Sync` on a focuser without
    /// position sync is refused here (HTTP 501).
    pub fn operation(
        &self,
        cmd: &FocuserCmd,
        command_id: CommandId,
        source: CommandSource,
    ) -> DeviceResult<Box<dyn Operation<dyn FocuserDriver>>> {
        self.build_operation(cmd, command_id, source, None)
    }

    /// A move inside an autofocus run: TempComp is always off (F-d) and the
    /// run id is recorded on the `focuser_moves` row.
    pub fn autofocus_operation(
        &self,
        cmd: &FocuserCmd,
        command_id: CommandId,
        source: CommandSource,
        run_id: String,
    ) -> DeviceResult<Box<dyn Operation<dyn FocuserDriver>>> {
        if !self.position_source().is_reliable() {
            return Err(DeviceError::invalid_operation(
                "autofocus is refused while the focuser origin is unknown",
            ));
        }
        self.build_operation(cmd, command_id, source, Some(run_id))
    }

    fn build_operation(
        &self,
        cmd: &FocuserCmd,
        command_id: CommandId,
        source: CommandSource,
        run_id: Option<String>,
    ) -> DeviceResult<Box<dyn Operation<dyn FocuserDriver>>> {
        let inner = Arc::clone(&self.inner);
        Ok(match *cmd {
            FocuserCmd::MoveTo(target) => Box::new(motion::FocuserMoveOp::new(
                inner,
                motion::MoveRequest::To(target),
                command_id,
                source,
                run_id,
            )),
            FocuserCmd::MoveBy(steps) => Box::new(motion::FocuserMoveOp::new(
                inner,
                motion::MoveRequest::By(steps),
                command_id,
                source,
                run_id,
            )),
            FocuserCmd::Halt => Box::new(motion::FocuserHaltOp::new(inner, command_id)),
            FocuserCmd::SetTempComp(on) => {
                if !self.inner.caps.temp_comp_available {
                    return Err(DeviceError::unsupported(
                        "TempComp is not available on this focuser",
                    ));
                }
                Box::new(motion::FocuserTempCompOp::new(inner, command_id, on))
            }
            FocuserCmd::Sync(position) => {
                if !self.inner.caps.can_sync {
                    return Err(DeviceError::unsupported(
                        "this focuser has no position sync (INDI FOCUS_SYNC only); use \
                         focuser/set-position for an emulated relative focuser",
                    ));
                }
                Box::new(motion::FocuserSyncOp::new(inner, command_id, position))
            }
        })
    }

    /// The single reconnect attempt succeeded (§2.6): a relative focuser's
    /// origin is lost.
    pub fn on_reconnect(&self) {
        let inner = &self.inner;
        if inner.caps.absolute {
            let mut s = inner.lock();
            s.last_idle_raw = None;
            s.backlash.last_direction = None;
            return;
        }
        let from = {
            let mut s = inner.lock();
            let from = s.logical;
            s.logical = None;
            s.source = PositionSource::UnknownAfterReconnect;
            s.backlash = BacklashState::default();
            from
        };
        inner.event(
            EventKind::FocuserPositionReset {
                from,
                to: None,
                cause: "reconnect".into(),
            },
            None,
        );
        inner.persist_relative(None, CloseState::Unknown);
    }

    /// Persist what a clean disconnect may keep. A disconnect while a move
    /// is active leaves the relative position unknown.
    pub fn on_disconnect(&self, end: ConnectionEnd) {
        let inner = &self.inner;
        let s = inner.lock().clone();
        if inner.caps.absolute {
            inner.persist_backlash(s.backlash, s.last_idle_raw);
            return;
        }
        let clean = !s.op_active
            && s.source.is_reliable()
            && matches!(
                end,
                ConnectionEnd::Operator | ConnectionEnd::Shutdown | ConnectionEnd::Replaced
            );
        if clean {
            inner.persist_relative(s.logical, CloseState::Clean);
        } else {
            inner.persist_relative(None, CloseState::Unknown);
        }
    }

    /// `POST focuser/set-position`: the operator declares the logical
    /// position of an emulated relative focuser. Profile only; no hardware
    /// command (F-b). The gate audits it and checks the bounds.
    pub fn set_position(&self, position: i32) -> DeviceResult<()> {
        let inner = &self.inner;
        if inner.caps.absolute {
            return Err(DeviceError::invalid_operation(
                "an absolute focuser reports its own position; set-position applies only to \
                 emulated relative focusers",
            ));
        }
        if let Some((lo, hi)) = inner.travel_bounds() {
            if position < lo || position > hi {
                return Err(DeviceError::invalid_value(format!(
                    "position {position} is outside the travel bounds [{lo}, {hi}]"
                )));
            }
        }
        let from = {
            let mut s = inner.lock();
            if s.op_active {
                return Err(DeviceError::invalid_operation(
                    "the focuser is moving; halt it before setting its position",
                ));
            }
            let from = s.logical;
            s.logical = Some(position);
            s.source = PositionSource::Emulated;
            s.backlash = BacklashState::default();
            from
        };
        inner.event(
            EventKind::FocuserPositionReset {
                from,
                to: Some(position),
                cause: "operator_set_position".into(),
            },
            None,
        );
        inner.persist_relative(Some(position), CloseState::Clean);
        Ok(())
    }

    pub fn position_source(&self) -> PositionSource {
        self.inner.lock().source
    }

    pub fn logical_position(&self) -> Option<i32> {
        self.inner.lock().logical
    }

    pub fn backlash_state(&self) -> BacklashState {
        self.inner.lock().backlash
    }

    pub fn settings(&self) -> &FocuserSettings {
        &self.inner.settings
    }

    pub fn capabilities(&self) -> &FocuserCapabilities {
        &self.inner.caps
    }
}

/// Alpaca drivers report a lost position as a driver error (for example
/// `0x500` "position unknown"); F-h turns it into `focuser_position_reset`.
pub fn announces_lost_position(message: &str) -> bool {
    let text = message.to_ascii_lowercase();
    text.contains("position")
        && ["unknown", "lost", "not homed", "reset"]
            .iter()
            .any(|word| text.contains(word))
}

/// The focuser's camera-dependent reading for gate checks: true when an
/// exposure is known (fresh or stale) to be integrating.
fn camera_exposing(snap: &EquipmentSnapshot) -> bool {
    snap.camera.exposing.known().copied() == Some(true)
}

// ---------------------------------------------------------------------------
// Gate plug-in
// ---------------------------------------------------------------------------

/// The focuser `SafetyCheck` (§2.9 step 3):
///
/// - `FocusSafetyGuard` travel bounds (`[focus.safety] min_position`,
///   `max_position`, plus `[0, MaxStep]`) on **every** move, manual included;
/// - while the position source is `emulated_origin_unknown` or
///   `unknown_after_reconnect`, only relative moves up to
///   `[focus.safety] max_step_single`, and autofocus is refused;
/// - every move (and autofocus) is refused while the camera integrates an
///   exposure (A06).
#[derive(Debug, Clone, PartialEq)]
pub struct FocuserSafetyCheck {
    guard_min: i32,
    guard_max: i32,
    max_step_single: i32,
}

impl FocuserSafetyCheck {
    pub const NAME: &'static str = "focuser_safety";

    pub fn new(safety: &FocusSafetyConfig) -> Self {
        Self {
            guard_min: safety.min_position,
            guard_max: safety.max_position,
            max_step_single: safety.max_step_single.saturating_abs().max(1),
        }
    }

    fn reject(reason: impl Into<String>, details: serde_json::Value) -> SafetyRejection {
        SafetyRejection {
            check: Self::NAME.into(),
            reason: reason.into(),
            details: Some(details),
        }
    }

    fn bounds(&self, snap: &FocuserSnapshot) -> (i32, i32) {
        let mut lo = self.guard_min;
        let mut hi = self.guard_max;
        if snap.capabilities.absolute {
            lo = lo.max(0);
            if let Some(max_step) = snap.capabilities.max_step.filter(|m| *m > 0) {
                hi = hi.min(max_step);
            }
        }
        (lo, hi)
    }

    fn check_target(&self, snap: &FocuserSnapshot, target: i64) -> Result<(), SafetyRejection> {
        let (lo, hi) = self.bounds(snap);
        if target < i64::from(lo) || target > i64::from(hi) {
            return Err(Self::reject(
                format!("target {target} is outside the focuser travel bounds [{lo}, {hi}]"),
                json!({"target": target, "min": lo, "max": hi, "rule": "travel_bounds"}),
            ));
        }
        Ok(())
    }
}

impl SafetyCheck for FocuserSafetyCheck {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn applies_to(&self, cmd: &GatedCommand) -> bool {
        matches!(
            cmd.op,
            GatedOp::Focuser(FocuserCmd::MoveTo(_) | FocuserCmd::MoveBy(_) | FocuserCmd::Sync(_))
                | GatedOp::FocuserSetPosition { .. }
                | GatedOp::Autofocus { .. }
        )
    }

    fn check(&self, cmd: &GatedCommand, snap: &EquipmentSnapshot) -> Result<(), SafetyRejection> {
        let Some(focuser) = snap.focuser.as_deref() else {
            return Err(Self::reject(
                "no focuser state is available",
                json!({"rule": "focuser_state_unknown"}),
            ));
        };
        let reliable = focuser.position_source.is_reliable();
        let moves = matches!(
            cmd.op,
            GatedOp::Focuser(FocuserCmd::MoveTo(_) | FocuserCmd::MoveBy(_))
                | GatedOp::Autofocus { .. }
        );
        if moves && camera_exposing(snap) {
            return Err(Self::reject(
                "the camera is integrating an exposure; focuser moves wait for the exposure \
                 boundary (A06)",
                json!({"rule": "camera_exposing"}),
            ));
        }
        match &cmd.op {
            GatedOp::Autofocus { .. } if !reliable => Err(Self::reject(
                "autofocus is refused while the focuser origin is unknown",
                json!({"rule": "origin_unknown", "position_source": focuser.position_source}),
            )),
            GatedOp::Focuser(FocuserCmd::MoveTo(target)) => {
                if !reliable {
                    return Err(Self::reject(
                        "absolute moves are refused while the focuser origin is unknown; use a \
                         relative move or set the position",
                        json!({"rule": "origin_unknown", "position_source": focuser.position_source}),
                    ));
                }
                self.check_target(focuser, i64::from(*target))
            }
            GatedOp::Focuser(FocuserCmd::MoveBy(steps)) => {
                if !reliable {
                    if steps.unsigned_abs() > self.max_step_single.unsigned_abs() {
                        return Err(Self::reject(
                            format!(
                                "relative move of {steps} steps exceeds max_step_single {} while \
                                 the focuser origin is unknown",
                                self.max_step_single
                            ),
                            json!({"rule": "origin_unknown_step_limit", "steps": steps,
                                   "max_step_single": self.max_step_single}),
                        ));
                    }
                    return Ok(());
                }
                match focuser.logical_position {
                    Some(position) => {
                        self.check_target(focuser, i64::from(position) + i64::from(*steps))
                    }
                    None => Err(Self::reject(
                        "the focuser position is unknown, so the move cannot be bounds-checked",
                        json!({"rule": "position_unknown"}),
                    )),
                }
            }
            GatedOp::Focuser(FocuserCmd::Sync(position))
            | GatedOp::FocuserSetPosition { position } => {
                self.check_target(focuser, i64::from(*position))
            }
            _ => Ok(()),
        }
    }

    fn annotations(&self, cmd: &GatedCommand, snap: &EquipmentSnapshot) -> Vec<String> {
        let moves = matches!(
            cmd.op,
            GatedOp::Focuser(FocuserCmd::MoveTo(_) | FocuserCmd::MoveBy(_))
                | GatedOp::Autofocus { .. }
        );
        if moves && snap.camera.exposing.known().is_none() {
            vec!["camera_state_unknown".into()]
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    //! Session-level tests: every operation runs through a minimal stand-in
    //! for the WP1A-2 worker (poll, publish, step, honour `Continue` and
    //! `Settling` on a `ManualClock`, enforce the deadline).

    use super::*;
    use crate::config::AchernarConfig;
    use crate::equipment::alpaca::focuser::AlpacaFocuser;
    use crate::equipment::events::FocuserMoveRecord;
    use crate::equipment::legacy::sim_focuser::LegacySimFocuser;
    use crate::equipment::ops::OpProgress;
    use crate::equipment::sim::focuser::{
        SimCall, SimFocuser, SimFocuserConfig, SimFocuserControl,
    };
    use crate::equipment::snapshot::{CameraSummary, ExposureRef, MoveMode};
    use crate::equipment::testing::fake_handles::{FakeCameraActivity, RecordingSink};
    use crate::equipment::testing::fake_transport::{FakeAlpacaTransport, FakeReply};
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::types::{
        AutonomyMode, ErrorClass, ObsState, Observed, OpOutcome, Priority, Source,
    };
    use serde_json::Value;

    struct Rig {
        clock: Arc<ManualClock>,
        sink: Arc<RecordingSink>,
        camera: Arc<FakeCameraActivity>,
        drv: Box<dyn FocuserDriver>,
        control: Option<SimFocuserControl>,
        session: FocuserSession,
        warnings: Vec<String>,
        tick: u64,
    }

    fn configs(tune: impl FnOnce(&mut EquipmentConfigs)) -> EquipmentConfigs {
        let mut configs = AchernarConfig::default().equipment_configs();
        tune(&mut configs);
        configs
    }

    fn rig_with(
        drv: Box<dyn FocuserDriver>,
        control: Option<SimFocuserControl>,
        clock: Arc<ManualClock>,
        configs: &EquipmentConfigs,
        profile: Option<Value>,
    ) -> Rig {
        let mut drv = drv;
        let sink = RecordingSink::new();
        let camera = FakeCameraActivity::new();
        let deps = FocuserDeps {
            clock: clock.clone(),
            sink: sink.clone(),
            camera: Some(camera.clone()),
        };
        let connection = FocuserSession::connect(
            drv.as_mut(),
            Duration::from_secs(5),
            profile.as_ref(),
            configs,
            deps,
        )
        .expect("connect");
        let mut rig = Rig {
            clock,
            sink,
            camera,
            drv,
            control,
            session: connection.session,
            warnings: connection.warnings,
            tick: 0,
        };
        rig.poll(false);
        rig
    }

    fn sim_rig(cfg: SimFocuserConfig, configs: &EquipmentConfigs, profile: Option<Value>) -> Rig {
        let clock = Arc::new(ManualClock::new());
        let sim = SimFocuser::new(cfg, clock.clone());
        let control = sim.control();
        rig_with(Box::new(sim), Some(control), clock, configs, profile)
    }

    fn operator() -> CommandSource {
        CommandSource::Operator { overriding: false }
    }

    impl Rig {
        fn control(&self) -> &SimFocuserControl {
            self.control.as_ref().expect("a simulator rig")
        }

        fn poll(&mut self, active: bool) -> FocuserSnapshot {
            self.tick += 1;
            let plan = FocuserSession::poll_plan(self.tick, active, self.tick.is_multiple_of(5));
            let poll = self.drv.poll(&plan);
            self.session
                .after_poll(self.drv.as_mut(), poll.as_ref(), None)
        }

        fn op(&self, cmd: FocuserCmd) -> Box<dyn Operation<dyn FocuserDriver>> {
            self.session
                .operation(&cmd, CommandId::new(), operator())
                .expect("operation")
        }

        /// Start and drive an operation to its outcome.
        fn run_op(&mut self, mut op: Box<dyn Operation<dyn FocuserDriver>>) -> OpOutcome {
            self.run_op_with(&mut op, |_, _| {})
        }

        fn run_op_with(
            &mut self,
            op: &mut Box<dyn Operation<dyn FocuserDriver>>,
            mut each_step: impl FnMut(&mut Rig, u32),
        ) -> OpOutcome {
            self.poll(false);
            let clock = self.clock.clone();
            if let Err(err) = op.start(self.drv.as_mut(), clock.as_ref()) {
                return if err.class == ErrorClass::OutcomeUnknown {
                    OpOutcome::Unknown(err)
                } else {
                    OpOutcome::Failed(err)
                };
            }
            let deadline =
                clock.now_mono_ns() + crate::equipment::clock::duration_ns(op.deadline());
            for i in 0..100_000 {
                each_step(self, i);
                self.poll(true);
                match op.step(self.drv.as_mut(), clock.as_ref()) {
                    OpProgress::Continue { next_poll } => clock.advance(next_poll),
                    OpProgress::Settling { until_mono_ns } => {
                        let now = clock.now_mono_ns();
                        clock.advance(Duration::from_nanos(until_mono_ns.saturating_sub(now)));
                    }
                    OpProgress::Done(outcome) => {
                        self.poll(false);
                        return outcome;
                    }
                }
                if clock.now_mono_ns() >= deadline {
                    let _ = op.cancel(self.drv.as_mut());
                    return OpOutcome::TimedOut;
                }
            }
            panic!("operation never resolved");
        }

        fn run(&mut self, cmd: FocuserCmd) -> OpOutcome {
            let op = self.op(cmd);
            self.run_op(op)
        }

        fn moves(&self) -> Vec<FocuserMoveRecord> {
            self.sink
                .records()
                .into_iter()
                .filter_map(|r| match r {
                    GroundTruthRecord::FocuserMove(rec) => Some(*rec),
                    _ => None,
                })
                .collect()
        }

        fn events(&self) -> Vec<EventKind> {
            self.sink
                .records()
                .into_iter()
                .filter_map(|r| match r {
                    GroundTruthRecord::Event(event) => Some(event.event.clone()),
                    _ => None,
                })
                .collect()
        }

        fn focuser_event_names(&self) -> Vec<&'static str> {
            self.sink
                .event_names()
                .into_iter()
                .filter(|n| n.starts_with("focuser_") || *n == "tempcomp_changed")
                .collect()
        }

        fn resets(&self) -> Vec<(Option<i32>, Option<i32>, String)> {
            self.events()
                .into_iter()
                .filter_map(|e| match e {
                    EventKind::FocuserPositionReset { from, to, cause } => Some((from, to, cause)),
                    _ => None,
                })
                .collect()
        }

        fn patches(&self) -> Vec<Value> {
            self.sink
                .records()
                .into_iter()
                .filter_map(|r| match r {
                    GroundTruthRecord::Profile(update) => Some(update.patch),
                    _ => None,
                })
                .collect()
        }

        /// What a profile store would hold after merging every patch.
        fn profile_json(&self, base: Option<Value>) -> Value {
            let mut profile = base.unwrap_or_else(|| json!({}));
            for patch in self.patches() {
                merge(&mut profile, &patch);
            }
            profile
        }

        fn snapshot(&mut self) -> EquipmentSnapshot {
            let focuser = self.poll(false);
            EquipmentSnapshot {
                focuser: Some(Arc::new(focuser)),
                camera: CameraSummary {
                    exposing: self.camera.exposing(),
                    ..CameraSummary::default()
                },
                ..EquipmentSnapshot::default()
            }
        }
    }

    fn merge(target: &mut Value, patch: &Value) {
        let (Some(target), Some(patch)) = (target.as_object_mut(), patch.as_object()) else {
            return;
        };
        for (key, value) in patch {
            match target.get_mut(key) {
                Some(existing) if existing.is_object() && value.is_object() => {
                    merge(existing, value)
                }
                _ => {
                    target.insert(key.clone(), value.clone());
                }
            }
        }
    }

    fn gated(op: GatedOp) -> GatedCommand {
        GatedCommand::new(op, operator(), AutonomyMode::Manual, Priority::Exposure)
    }

    fn sim_moves(rig: &Rig) -> Vec<SimCall> {
        rig.control().moves()
    }

    // ---- F-a: completion, read-back and the one-record rule ----

    #[test]
    fn every_move_writes_one_record_with_read_back_and_ordered_events() {
        let mut rig = sim_rig(SimFocuserConfig::absolute(), &configs(|_| {}), None);
        assert_eq!(rig.session.position_source(), PositionSource::Device);
        assert_eq!(rig.run(FocuserCmd::MoveTo(25_400)), OpOutcome::Succeeded);
        let moves = rig.moves();
        assert_eq!(moves.len(), 1);
        let rec = &moves[0];
        assert_eq!(rec.mode, MoveMode::Absolute);
        assert_eq!(rec.commanded_target, 25_400);
        assert_eq!((rec.raw_start, rec.raw_end), (Some(25_000), Some(25_400)));
        assert_eq!(
            (rec.logical_start, rec.logical_end),
            (Some(25_000), Some(25_400))
        );
        assert_eq!(rec.position_source, PositionSource::Device);
        assert_eq!(rec.direction, Some(FocusDirection::Out));
        assert_eq!(rec.reversal, None);
        assert_eq!(rec.temperature_start_c, Some(12.0));
        assert!(rec.started_at.is_some() && rec.stopped_at.is_some());
        let settled = rec.settled_at.unwrap();
        // The settle (500 ms) separates the stop from the settled read.
        assert!(settled - rec.stopped_at.unwrap() >= chrono::Duration::milliseconds(500));
        assert_eq!(rec.outcome, OpOutcome::Succeeded);
        assert_eq!(
            rig.focuser_event_names(),
            vec![
                "focuser_temperature",
                "focuser_move_commanded",
                "focuser_move_started",
                "focuser_move_stopped",
                "focuser_settled",
            ]
        );
        assert_eq!(rig.poll(false).logical_position, Some(25_400));
        assert_eq!(
            rig.session.backlash_state().last_direction,
            Some(FocusDirection::Out)
        );
    }

    #[test]
    fn moves_are_chunked_by_max_increment() {
        let mut cfg = SimFocuserConfig::absolute();
        cfg.max_increment = 100;
        let mut rig = sim_rig(cfg, &configs(|_| {}), None);
        assert_eq!(rig.run(FocuserCmd::MoveTo(25_350)), OpOutcome::Succeeded);
        assert_eq!(
            sim_moves(&rig),
            vec![
                SimCall::MoveAbsolute(25_100),
                SimCall::MoveAbsolute(25_200),
                SimCall::MoveAbsolute(25_300),
                SimCall::MoveAbsolute(25_350),
            ]
        );
        assert_eq!(
            rig.moves().len(),
            1,
            "one record for the whole chunked move"
        );
        // Without a smaller MaxIncrement nothing is chunked.
        let mut rig = sim_rig(SimFocuserConfig::absolute(), &configs(|_| {}), None);
        rig.run(FocuserCmd::MoveBy(-3_000));
        assert_eq!(sim_moves(&rig), vec![SimCall::MoveAbsolute(22_000)]);
    }

    #[test]
    fn targets_are_clamped_to_the_profile_and_max_step() {
        let profile = json!({"travel": {"min": 1_000, "max": 30_000}});
        let mut rig = sim_rig(
            SimFocuserConfig::absolute(),
            &configs(|_| {}),
            Some(profile),
        );
        assert_eq!(rig.run(FocuserCmd::MoveTo(45_000)), OpOutcome::Succeeded);
        assert_eq!(rig.run(FocuserCmd::MoveBy(-40_000)), OpOutcome::Succeeded);
        let moves = rig.moves();
        assert_eq!(moves[0].commanded_target, 45_000);
        assert_eq!(moves[0].raw_end, Some(30_000));
        assert_eq!(moves[1].logical_end, Some(1_000));

        let mut cfg = SimFocuserConfig::absolute();
        cfg.max_step = 26_000;
        let mut rig = sim_rig(cfg, &configs(|_| {}), None);
        rig.run(FocuserCmd::MoveTo(40_000));
        assert_eq!(sim_moves(&rig), vec![SimCall::MoveAbsolute(26_000)]);
    }

    #[test]
    fn a_stuck_motor_is_halted_and_fails_as_stuck() {
        let cfgs = configs(|c| c.focuser.stuck_timeout_s = 2);
        let mut rig = sim_rig(SimFocuserConfig::absolute(), &cfgs, None);
        let mut op = rig.op(FocuserCmd::MoveTo(30_000));
        let outcome = rig.run_op_with(&mut op, |rig, i| {
            if i == 3 {
                rig.control().set_stuck(true);
            }
        });
        let OpOutcome::Failed(err) = outcome else {
            panic!("expected a stuck failure, got {outcome:?}");
        };
        assert!(err.message.starts_with("stuck"), "{err}");
        assert_eq!(rig.control().halts(), 1);
        assert!(!rig.control().is_moving());
        let rec = &rig.moves()[0];
        assert!(matches!(rec.outcome, OpOutcome::Failed(_)));
        assert_eq!(rec.raw_end, Some(25_200), "read-back of where it stalled");
        let names = rig.focuser_event_names();
        assert!(
            names.ends_with(&["focuser_halted", "focuser_move_stopped"]),
            "{names:?}"
        );
    }

    #[test]
    fn off_target_arrivals_are_reissued_up_to_the_cap() {
        let mut rig = sim_rig(SimFocuserConfig::absolute(), &configs(|_| {}), None);
        rig.control().push_arrival_error(7);
        assert_eq!(rig.run(FocuserCmd::MoveTo(25_400)), OpOutcome::Succeeded);
        assert_eq!(
            sim_moves(&rig),
            vec![SimCall::MoveAbsolute(25_400), SimCall::MoveAbsolute(25_400)]
        );
        assert_eq!(rig.moves()[0].raw_end, Some(25_400));

        // max_reissue = 1: a second miss fails instead of looping.
        rig.control().clear_calls();
        rig.control().push_arrival_error(9);
        rig.control().push_arrival_error(9);
        let outcome = rig.run(FocuserCmd::MoveTo(25_000));
        let OpOutcome::Failed(err) = outcome else {
            panic!("expected off-target failure, got {outcome:?}");
        };
        assert!(err.message.contains("off target"), "{err}");
        assert_eq!(sim_moves(&rig).len(), 2, "initial move plus one re-issue");
        assert_eq!(rig.moves()[1].raw_end, Some(25_009));
    }

    // ---- F-e: backlash ----

    #[test]
    fn overshoot_backlash_always_finishes_from_the_same_side() {
        let mut cfg = SimFocuserConfig::absolute();
        cfg.mechanical_backlash_steps = 20;
        let profile = json!({"backlash": {"model": "overshoot", "in_steps": 0, "out_steps": 40}});
        let mut rig = sim_rig(cfg, &configs(|_| {}), Some(profile));
        assert_eq!(rig.run(FocuserCmd::MoveTo(25_400)), OpOutcome::Succeeded);
        assert_eq!(
            sim_moves(&rig),
            vec![SimCall::MoveAbsolute(25_440), SimCall::MoveAbsolute(25_400)]
        );
        assert_eq!(rig.control().optical_position(), 25_400);
        rig.control().clear_calls();
        assert_eq!(rig.run(FocuserCmd::MoveTo(25_300)), OpOutcome::Succeeded);
        assert_eq!(sim_moves(&rig), vec![SimCall::MoveAbsolute(25_300)]);
        assert_eq!(rig.control().optical_position(), 25_300);
        let moves = rig.moves();
        assert_eq!(
            moves[0].backlash_model,
            BacklashModel::Overshoot {
                in_steps: 0,
                out_steps: 40
            }
        );
        assert_eq!(moves[1].reversal, Some(true));
        assert_eq!(moves[1].logical_end, Some(25_300));
    }

    #[test]
    fn absolute_backlash_compensates_reversals_and_hides_the_offset() {
        let mut cfg = SimFocuserConfig::absolute();
        cfg.mechanical_backlash_steps = 20;
        let profile = json!({"backlash": {"model": "absolute", "in_steps": 20, "out_steps": 20}});
        let mut rig = sim_rig(cfg, &configs(|_| {}), Some(profile));
        let mut lag = Vec::new();
        for target in [25_400, 25_300, 25_500] {
            assert_eq!(rig.run(FocuserCmd::MoveTo(target)), OpOutcome::Succeeded);
            let snap = rig.poll(false);
            assert_eq!(snap.logical_position, Some(target));
            lag.push(rig.control().optical_position() - target);
        }
        // The optical position keeps a constant relation to the logical one:
        // the slop never leaks into the position models see.
        assert_eq!(lag, vec![-20, -20, -20]);
        let raw: Vec<_> = rig.moves().iter().map(|m| m.raw_end).collect();
        assert_eq!(raw, vec![Some(25_400), Some(25_280), Some(25_500)]);
        let reversals: Vec<_> = rig.moves().iter().map(|m| m.reversal).collect();
        assert_eq!(reversals, vec![None, Some(true), Some(true)]);
        // The offset is persisted with the raw position it belongs to.
        let profile = rig.profile_json(None);
        assert_eq!(profile["backlash_state"]["raw"], json!(25_500));
    }

    #[test]
    fn firmware_owned_backlash_gets_no_compensation() {
        let cfgs = configs(|c| {
            c.focus_backlash.enabled = true;
            c.focus_backlash.compensation_steps = 50;
        });
        let profile = json!({"backlash_owner": "firmware"});
        let mut rig = sim_rig(SimFocuserConfig::absolute(), &cfgs, Some(profile));
        rig.run(FocuserCmd::MoveTo(24_000));
        assert_eq!(sim_moves(&rig), vec![SimCall::MoveAbsolute(24_000)]);
        let rec = &rig.moves()[0];
        assert_eq!(rec.backlash_model, BacklashModel::None);
        assert_eq!(rec.backlash_owner, BacklashOwner::Firmware);
        // Without the declaration the [focus.backlash] default overshoots.
        let mut rig = sim_rig(SimFocuserConfig::absolute(), &cfgs, None);
        rig.run(FocuserCmd::MoveTo(24_000));
        assert_eq!(
            sim_moves(&rig),
            vec![SimCall::MoveAbsolute(23_950), SimCall::MoveAbsolute(24_000)]
        );
    }

    // ---- F-d: TempComp ----

    #[test]
    fn tempcomp_is_turned_off_for_a_move_and_restored_on_every_path() {
        let mut cfg = SimFocuserConfig::absolute();
        cfg.temp_comp = true;
        let cfgs = configs(|c| c.focuser.stuck_timeout_s = 1);
        let mut rig = sim_rig(cfg, &cfgs, None);
        assert_eq!(rig.run(FocuserCmd::MoveTo(25_200)), OpOutcome::Succeeded);
        let calls: Vec<_> = rig
            .control()
            .calls()
            .into_iter()
            .filter(|c| !matches!(c, SimCall::Halt { .. }))
            .collect();
        assert_eq!(
            calls,
            vec![
                SimCall::SetTempComp(false),
                SimCall::MoveAbsolute(25_200),
                SimCall::SetTempComp(true),
            ]
        );
        let rec = &rig.moves()[0];
        assert_eq!(
            (rec.tempcomp_before, rec.tempcomp_after),
            (Some(true), Some(true))
        );
        let changes: Vec<_> = rig
            .events()
            .into_iter()
            .filter_map(|e| match e {
                EventKind::TempcompChanged { to, .. } => Some(to),
                _ => None,
            })
            .collect();
        assert_eq!(changes, vec![false, true]);
        assert!(rig.control().temp_comp());

        // A failed (stuck) move restores it too.
        let mut op = rig.op(FocuserCmd::MoveTo(26_000));
        rig.run_op_with(&mut op, |rig, i| {
            if i == 1 {
                rig.control().set_stuck(true);
            }
        });
        assert!(rig.control().temp_comp());
        assert_eq!(rig.moves()[1].tempcomp_after, Some(true));

        // `leave` writes nothing.
        let mut cfg = SimFocuserConfig::absolute();
        cfg.temp_comp = true;
        let cfgs = configs(|c| c.focuser.tempcomp_during_moves = TempCompDuringMoves::Leave);
        let mut rig = sim_rig(cfg, &cfgs, None);
        rig.run(FocuserCmd::MoveTo(25_100));
        assert!(rig
            .control()
            .calls()
            .iter()
            .all(|c| !matches!(c, SimCall::SetTempComp(_))));
    }

    #[test]
    fn autofocus_moves_run_with_tempcomp_off_and_carry_the_run_id() {
        let mut cfg = SimFocuserConfig::absolute();
        cfg.temp_comp = true;
        let cfgs = configs(|c| c.focuser.tempcomp_during_moves = TempCompDuringMoves::Leave);
        let mut rig = sim_rig(cfg, &cfgs, None);
        let op = rig
            .session
            .autofocus_operation(
                &FocuserCmd::MoveTo(25_100),
                CommandId::new(),
                operator(),
                "run-7".into(),
            )
            .unwrap();
        assert_eq!(rig.run_op(op), OpOutcome::Succeeded);
        assert_eq!(rig.control().calls()[0], SimCall::SetTempComp(false));
        assert_eq!(rig.moves()[0].run_id.as_deref(), Some("run-7"));
    }

    // ---- F-b and §2.15: relative emulation ----

    #[test]
    fn a_relative_position_is_restored_after_a_clean_close() {
        let base = json!({"relative_position": {"logical": 12_000, "state": "clean"}});
        let mut rig = sim_rig(
            SimFocuserConfig::relative(),
            &configs(|_| {}),
            Some(base.clone()),
        );
        assert_eq!(
            rig.session.position_source(),
            PositionSource::EmulatedRestored
        );
        assert_eq!(rig.session.logical_position(), Some(12_000));
        assert_eq!(rig.run(FocuserCmd::MoveBy(300)), OpOutcome::Succeeded);
        assert_eq!(rig.run(FocuserCmd::MoveTo(12_100)), OpOutcome::Succeeded);
        assert_eq!(
            sim_moves(&rig),
            vec![SimCall::MoveRelative(300), SimCall::MoveRelative(-200)]
        );
        assert_eq!(rig.session.logical_position(), Some(12_100));
        let rec = &rig.moves()[0];
        assert_eq!(rec.mode, MoveMode::Relative);
        assert_eq!(
            (rec.logical_start, rec.logical_end),
            (Some(12_000), Some(12_300))
        );
        assert_eq!(
            (rec.raw_start, rec.raw_end),
            (None, None),
            "no device position"
        );
        assert_eq!(rec.position_source, PositionSource::EmulatedRestored);
        // Marked moving while in flight, clean after each completed move.
        let states: Vec<_> = rig
            .patches()
            .into_iter()
            .filter_map(|p| p["relative_position"]["state"].as_str().map(str::to_string))
            .collect();
        assert_eq!(states, vec!["moving", "clean", "moving", "clean"]);
        rig.session.on_disconnect(ConnectionEnd::Operator);

        // Restart: a new connection restores the persisted position.
        let stored = rig.profile_json(Some(base));
        let restarted = sim_rig(SimFocuserConfig::relative(), &configs(|_| {}), Some(stored));
        assert_eq!(
            restarted.session.position_source(),
            PositionSource::EmulatedRestored
        );
        assert_eq!(restarted.session.logical_position(), Some(12_100));
    }

    #[test]
    fn a_relative_position_is_not_restored_after_an_unclean_close() {
        // The daemon died mid-move: the last row says `moving`.
        let base = json!({"relative_position": {"logical": 12_000, "state": "clean"}});
        let mut rig = sim_rig(
            SimFocuserConfig::relative(),
            &configs(|_| {}),
            Some(base.clone()),
        );
        let mut op = rig.op(FocuserCmd::MoveBy(400));
        rig.poll(false);
        op.start(rig.drv.as_mut(), rig.clock.as_ref()).unwrap();
        drop(op);
        let stored = rig.profile_json(Some(base.clone()));
        assert_eq!(stored["relative_position"]["state"], json!("moving"));
        let restarted = sim_rig(SimFocuserConfig::relative(), &configs(|_| {}), Some(stored));
        assert_eq!(
            restarted.session.position_source(),
            PositionSource::EmulatedOriginUnknown
        );
        assert_eq!(restarted.session.logical_position(), None);

        // A halted relative move loses its origin and closes the row unknown.
        let mut rig = sim_rig(
            SimFocuserConfig::relative(),
            &configs(|_| {}),
            Some(base.clone()),
        );
        let mut op = rig.op(FocuserCmd::MoveBy(400));
        rig.poll(false);
        op.start(rig.drv.as_mut(), rig.clock.as_ref()).unwrap();
        rig.clock.advance_ms(100);
        rig.poll(true);
        op.cancel(rig.drv.as_mut()).unwrap();
        assert_eq!(
            rig.session.position_source(),
            PositionSource::EmulatedOriginUnknown
        );
        assert_eq!(
            rig.resets().last().unwrap(),
            &(Some(12_000), None, "relative_move_interrupted".to_string())
        );
        assert!(matches!(rig.moves()[0].outcome, OpOutcome::Aborted { .. }));
        let stored = rig.profile_json(Some(base.clone()));
        assert_eq!(stored["relative_position"]["state"], json!("unknown"));
        let restarted = sim_rig(SimFocuserConfig::relative(), &configs(|_| {}), Some(stored));
        assert_eq!(
            restarted.session.position_source(),
            PositionSource::EmulatedOriginUnknown
        );

        // A reconnect loses it as well.
        let mut rig = sim_rig(SimFocuserConfig::relative(), &configs(|_| {}), Some(base));
        rig.session.on_reconnect();
        assert_eq!(
            rig.session.position_source(),
            PositionSource::UnknownAfterReconnect
        );
        assert_eq!(rig.resets()[0].2, "reconnect");
        assert_eq!(rig.poll(false).logical_position, None);
    }

    #[test]
    fn an_unknown_origin_allows_only_small_relative_moves_and_refuses_autofocus() {
        let mut rig = sim_rig(SimFocuserConfig::relative(), &configs(|_| {}), None);
        assert_eq!(
            rig.session.position_source(),
            PositionSource::EmulatedOriginUnknown
        );
        assert!(rig.warnings.iter().any(|w| w.contains("origin is unknown")));

        // The gate check.
        let check = FocuserSafetyCheck::new(&FocusSafetyConfig::default());
        let snap = rig.snapshot();
        let reject =
            |op| check.check(&gated(op), &snap).unwrap_err().details.unwrap()["rule"].clone();
        assert_eq!(
            reject(GatedOp::Focuser(FocuserCmd::MoveTo(100))),
            json!("origin_unknown")
        );
        assert_eq!(
            reject(GatedOp::Focuser(FocuserCmd::MoveBy(600))),
            json!("origin_unknown_step_limit")
        );
        assert_eq!(
            reject(GatedOp::Autofocus { run_id: None }),
            json!("origin_unknown")
        );
        assert!(check
            .check(&gated(GatedOp::Focuser(FocuserCmd::MoveBy(-500))), &snap)
            .is_ok());

        // The operation enforces the same rules.
        let outcome = rig.run(FocuserCmd::MoveTo(100));
        assert!(
            matches!(outcome, OpOutcome::Failed(ref e) if e.class == ErrorClass::InvalidOperation)
        );
        let outcome = rig.run(FocuserCmd::MoveBy(600));
        assert!(matches!(outcome, OpOutcome::Failed(ref e) if e.class == ErrorClass::InvalidValue));
        assert!(rig
            .session
            .autofocus_operation(
                &FocuserCmd::MoveBy(10),
                CommandId::new(),
                operator(),
                "r".into()
            )
            .is_err());
        assert_eq!(rig.run(FocuserCmd::MoveBy(-200)), OpOutcome::Succeeded);
        assert_eq!(sim_moves(&rig), vec![SimCall::MoveRelative(-200)]);
        // Every attempt is recorded; none invents a position.
        let moves = rig.moves();
        assert_eq!(moves.len(), 3);
        assert!(moves.iter().all(|m| m.logical_end.is_none()));
        assert_eq!(
            moves[2].commanded_target, -200,
            "step count when the origin is unknown"
        );
        assert!(!moves[2].position_source.is_reliable(), "no FOP label");

        // The operator declares the origin (profile only, no hardware).
        rig.control().clear_calls();
        rig.session.set_position(8_000).unwrap();
        assert!(rig.control().calls().is_empty());
        assert_eq!(rig.session.position_source(), PositionSource::Emulated);
        assert_eq!(rig.resets().last().unwrap().2, "operator_set_position");
        assert_eq!(rig.run(FocuserCmd::MoveTo(8_250)), OpOutcome::Succeeded);
        assert_eq!(rig.session.logical_position(), Some(8_250));
        let snap = rig.snapshot();
        assert!(check
            .check(&gated(GatedOp::Autofocus { run_id: None }), &snap)
            .is_ok());
    }

    // ---- A06 and the FocusSafetyGuard bounds ----

    #[test]
    fn moves_are_refused_while_the_camera_is_exposing() {
        let mut rig = sim_rig(SimFocuserConfig::absolute(), &configs(|_| {}), None);
        rig.camera.start_exposure(ExposureRef {
            duration_s: Some(30.0),
            ..ExposureRef::default()
        });
        let check = FocuserSafetyCheck::new(&FocusSafetyConfig::default());
        let snap = rig.snapshot();
        for op in [
            GatedOp::Focuser(FocuserCmd::MoveTo(25_100)),
            GatedOp::Focuser(FocuserCmd::MoveBy(10)),
            GatedOp::Autofocus { run_id: None },
        ] {
            let rejection = check.check(&gated(op), &snap).unwrap_err();
            assert_eq!(rejection.check, FocuserSafetyCheck::NAME);
            assert_eq!(rejection.details.unwrap()["rule"], json!("camera_exposing"));
        }
        // Halt is never gated, so it is not this check's business.
        assert!(!check.applies_to(&gated(GatedOp::Focuser(FocuserCmd::Halt))));

        // The worker re-checks at start: nothing moves, the attempt is recorded.
        let outcome = rig.run(FocuserCmd::MoveTo(25_100));
        assert!(
            matches!(outcome, OpOutcome::Failed(ref e) if e.class == ErrorClass::SafetyRejected)
        );
        assert!(sim_moves(&rig).is_empty());
        assert_eq!(rig.moves().len(), 1);
        rig.camera.finish_exposure();
        assert_eq!(rig.run(FocuserCmd::MoveTo(25_100)), OpOutcome::Succeeded);
    }

    #[test]
    fn focus_safety_guard_bounds_apply_to_manual_moves() {
        let mut rig = sim_rig(SimFocuserConfig::absolute(), &configs(|_| {}), None);
        let safety = FocusSafetyConfig {
            min_position: 1_000,
            max_position: 40_000,
            ..FocusSafetyConfig::default()
        };
        let check = FocuserSafetyCheck::new(&safety);
        let snap = rig.snapshot();
        let ok = |op| check.check(&gated(op), &snap).is_ok();
        assert!(!ok(GatedOp::Focuser(FocuserCmd::MoveTo(40_001))));
        assert!(!ok(GatedOp::Focuser(FocuserCmd::MoveTo(999))));
        assert!(!ok(GatedOp::Focuser(FocuserCmd::MoveBy(15_001))));
        assert!(!ok(GatedOp::Focuser(FocuserCmd::Sync(50))));
        assert!(!ok(GatedOp::FocuserSetPosition { position: 45_000 }));
        assert!(ok(GatedOp::Focuser(FocuserCmd::MoveTo(40_000))));
        assert!(ok(GatedOp::Focuser(FocuserCmd::MoveBy(-24_000))));
        // MaxStep tightens the guard.
        let wide = FocuserSafetyCheck::new(&FocusSafetyConfig {
            max_position: 90_000,
            ..FocusSafetyConfig::default()
        });
        assert!(wide
            .check(&gated(GatedOp::Focuser(FocuserCmd::MoveTo(50_001))), &snap)
            .is_err());
        // A move with no camera hook state is allowed but annotated.
        let blind = EquipmentSnapshot {
            focuser: snap.focuser.clone(),
            ..EquipmentSnapshot::default()
        };
        let cmd = gated(GatedOp::Focuser(FocuserCmd::MoveTo(20_000)));
        assert!(check.check(&cmd, &blind).is_ok());
        assert_eq!(
            check.annotations(&cmd, &blind),
            vec!["camera_state_unknown"]
        );
        assert!(check.annotations(&cmd, &snap).is_empty());
        assert!(
            check.check(&cmd, &EquipmentSnapshot::default()).is_err(),
            "no focuser state: fail safe"
        );
        // The operation applies the same guard by clamping.
        let cfgs = configs(|c| c.focus_safety = safety.clone());
        let mut rig = sim_rig(SimFocuserConfig::absolute(), &cfgs, None);
        rig.run(FocuserCmd::MoveTo(48_000));
        assert_eq!(rig.moves()[0].raw_end, Some(40_000));
    }

    // ---- F-c: halting ----

    #[test]
    fn cancel_halts_the_hardware_and_records_the_abort() {
        let mut rig = sim_rig(SimFocuserConfig::absolute(), &configs(|_| {}), None);
        let mut op = rig.op(FocuserCmd::MoveTo(30_000));
        rig.poll(false);
        op.start(rig.drv.as_mut(), rig.clock.as_ref()).unwrap();
        for _ in 0..5 {
            rig.clock.advance_ms(100);
            rig.poll(true);
            assert!(matches!(
                op.step(rig.drv.as_mut(), rig.clock.as_ref()),
                OpProgress::Continue { .. }
            ));
        }
        op.cancel(rig.drv.as_mut()).unwrap();
        assert_eq!(rig.control().halts(), 1);
        assert!(!rig.control().is_moving());
        assert!(matches!(
            op.step(rig.drv.as_mut(), rig.clock.as_ref()),
            OpProgress::Done(OpOutcome::Aborted { .. })
        ));
        let rec = &rig.moves()[0];
        assert!(matches!(rec.outcome, OpOutcome::Aborted { by: None }));
        assert_eq!(rec.raw_end, Some(25_500));
        // A second cancel is a no-op; there is still one record.
        op.cancel(rig.drv.as_mut()).unwrap();
        assert_eq!(rig.moves().len(), 1);
        let names = rig.focuser_event_names();
        assert!(
            names.ends_with(&["focuser_halted", "focuser_move_stopped"]),
            "{names:?}"
        );
        // The next poll publishes where it stopped.
        assert_eq!(rig.poll(false).logical_position, Some(25_500));

        // A halt command with no move active completes on IsMoving false.
        assert_eq!(rig.run(FocuserCmd::Halt), OpOutcome::Succeeded);
    }

    #[test]
    fn a_move_past_its_deadline_is_halted_as_timed_out() {
        let mut cfg = SimFocuserConfig::absolute();
        cfg.speed_steps_per_s = 10.0;
        let cfgs = configs(|c| c.focus_safety.move_timeout_ms = 2_000);
        let mut rig = sim_rig(cfg, &cfgs, None);
        assert_eq!(rig.run(FocuserCmd::MoveTo(26_000)), OpOutcome::TimedOut);
        assert_eq!(rig.control().halts(), 1);
        assert_eq!(rig.moves()[0].outcome, OpOutcome::TimedOut);
    }

    // ---- F-f, F-g, F-h, F-i ----

    #[test]
    fn temperature_events_are_rate_limited_and_ambient_probes_feed_environment_log() {
        let cfgs = configs(|c| c.focuser.temperature_role = TemperatureRole::Ambient);
        let mut rig = sim_rig(SimFocuserConfig::absolute(), &cfgs, None);
        for _ in 0..10 {
            rig.clock.advance_ms(2_000);
            rig.poll(false);
        }
        let temps = |rig: &Rig| {
            rig.events()
                .into_iter()
                .filter(|e| matches!(e, EventKind::FocuserTemperature { .. }))
                .count()
        };
        assert_eq!(temps(&rig), 1, "a steady temperature: one event per minute");
        rig.control().set_temperature(Some(13.0));
        rig.clock.advance_ms(10_000);
        rig.poll(false);
        assert_eq!(temps(&rig), 2, "a 1 degree change after the minimum gap");
        rig.clock.advance_ms(60_000);
        rig.poll(false);
        assert_eq!(temps(&rig), 3);
        let rows: Vec<_> = rig
            .sink
            .records()
            .into_iter()
            .filter_map(|r| match r {
                GroundTruthRecord::Environment(row) => Some(*row),
                _ => None,
            })
            .collect();
        // One probe row per [weather] log_interval_s (60 s): t = 0 and 90 s.
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].ambient_temp_c, Some(12.0));
        assert_eq!(rows[1].ambient_temp_c, Some(13.0));
        assert_eq!(rows[0].device_identity_key.as_deref(), Some("sim:focuser"));
        let snap = rig.poll(false);
        assert_eq!(snap.temperature_role, TemperatureRole::Ambient);
        assert!(snap.state.temperature_c.is_fresh());

        // A tube or unknown probe is journaled but never written as ambient.
        let mut rig = sim_rig(SimFocuserConfig::absolute(), &configs(|_| {}), None);
        rig.poll(false);
        assert!(rig
            .sink
            .records()
            .iter()
            .all(|r| !matches!(r, GroundTruthRecord::Environment(_))));
    }

    #[test]
    fn step_size_claims_and_the_position_source_action_are_recorded() {
        let mut cfg = SimFocuserConfig::absolute();
        cfg.supported_actions = vec!["PositionSource".into()];
        cfg.action_replies = vec![("PositionSource".into(), "Unknown\n".into())];
        let cfgs = configs(|c| c.focuser.position_source_action = "PositionSource".into());
        let rig = sim_rig(cfg, &cfgs, None);
        let profile = rig.profile_json(None);
        assert_eq!(profile["step_size_um_claim"], json!(4.5));
        assert_eq!(
            profile["position_source_action"]["result"],
            json!("unknown")
        );
        assert!(rig
            .warnings
            .iter()
            .any(|w| w.contains("reports \"unknown\"")));
        assert_eq!(
            rig.resets(),
            vec![(None, Some(25_000), "position_source_action_unknown".into())]
        );
        assert!(rig
            .control()
            .calls()
            .contains(&SimCall::Action("PositionSource".into())));

        // Not in SupportedActions: a warning, nothing called.
        let cfgs = configs(|c| c.focuser.position_source_action = "Homed".into());
        let rig = sim_rig(SimFocuserConfig::absolute(), &cfgs, None);
        assert!(rig
            .warnings
            .iter()
            .any(|w| w.contains("not in SupportedActions")));
        assert!(rig.resets().is_empty());
    }

    #[test]
    fn position_resets_are_published_for_sync_loss_and_external_moves() {
        let mut cfg = SimFocuserConfig::absolute();
        cfg.can_sync = true;
        let mut rig = sim_rig(cfg, &configs(|_| {}), None);
        assert_eq!(rig.run(FocuserCmd::Sync(10_000)), OpOutcome::Succeeded);
        assert_eq!(rig.resets()[0], (Some(25_000), Some(10_000), "sync".into()));
        assert_eq!(rig.poll(false).logical_position, Some(10_000));

        // A hand-controller move while idle.
        rig.control().external_move_to(10_200);
        rig.clock.advance_ms(2_000);
        rig.poll(false);
        assert_eq!(
            rig.resets()[1],
            (Some(10_000), Some(10_200), "external_change".into())
        );

        // The controller announces a lost position (0x500).
        rig.control().set_position_lost(true);
        rig.clock.advance_ms(2_000);
        rig.poll(false);
        rig.poll(false);
        let lost: Vec<_> = rig
            .resets()
            .into_iter()
            .filter(|r| r.2 == "driver_position_unknown")
            .collect();
        assert_eq!(
            lost,
            vec![(Some(10_200), None, "driver_position_unknown".into())]
        );
        assert_eq!(rig.poll(false).logical_position, None);
        rig.control().set_position_lost(false);
        assert_eq!(rig.poll(false).logical_position, Some(10_200));

        // Drivers without sync refuse it up front (HTTP 501).
        let rig = sim_rig(SimFocuserConfig::absolute(), &configs(|_| {}), None);
        let err = match rig
            .session
            .operation(&FocuserCmd::Sync(1), CommandId::new(), operator())
        {
            Err(err) => err,
            Ok(_) => panic!("sync must be refused"),
        };
        assert_eq!(err.class.http_status(), 501);
    }

    #[test]
    fn tempcomp_commands_are_verified_and_refused_when_unavailable() {
        let mut rig = sim_rig(SimFocuserConfig::absolute(), &configs(|_| {}), None);
        assert_eq!(rig.run(FocuserCmd::SetTempComp(true)), OpOutcome::Succeeded);
        assert!(rig.control().temp_comp());
        assert!(rig.events().iter().any(|e| matches!(
            e,
            EventKind::TempcompChanged { to: true, reason: Some(r), .. } if r == "command"
        )));
        let rig = sim_rig(SimFocuserConfig::relative(), &configs(|_| {}), None);
        assert!(rig
            .session
            .operation(&FocuserCmd::SetTempComp(true), CommandId::new(), operator())
            .is_err());
    }

    // ---- Other drivers through the same session ----

    #[test]
    fn a_platform6_alpaca_focuser_with_firmware_backlash_moves_through_the_session() {
        let fake = FakeAlpacaTransport::new().shared();
        fake.on_get("interfaceversion", FakeReply::value(3))
            .on_put("connected", FakeReply::ok())
            .on_get("connected", FakeReply::value(true))
            .on_get("absolute", FakeReply::value(true))
            .on_get("maxstep", FakeReply::value(40_000))
            .on_get("maxincrement", FakeReply::value(40_000))
            .on_get("tempcompavailable", FakeReply::value(false))
            .on_get("temperature", FakeReply::value(6.0))
            .on_put("move", FakeReply::ok())
            // Four idle polls (connect, rig, snapshot, pre-start), then one
            // poll while moving, then stopped at the target.
            .queue_get(
                "ismoving",
                [
                    FakeReply::value(false),
                    FakeReply::value(false),
                    FakeReply::value(false),
                    FakeReply::value(false),
                    FakeReply::value(true),
                ],
            )
            .on_get("ismoving", FakeReply::value(false))
            .queue_get(
                "position",
                [
                    FakeReply::value(12_000),
                    FakeReply::value(12_000),
                    FakeReply::value(12_000),
                    FakeReply::value(12_000),
                    FakeReply::value(12_250),
                ],
            )
            .on_get("position", FakeReply::value(12_500));
        let clock = Arc::new(ManualClock::new());
        let drv = AlpacaFocuser::new(
            "http://esp32.local:80",
            0,
            fake.clone(),
            fake.clone(),
            clock.clone(),
        )
        .with_connect_poll_interval(Duration::ZERO);
        let cfgs = configs(|c| {
            c.focus_backlash.enabled = true;
            c.focus_backlash.compensation_steps = 30;
        });
        let profile = json!({"backlash_owner": "firmware", "temperature_role": "ambient"});
        let mut rig = rig_with(Box::new(drv), None, clock, &cfgs, Some(profile));
        let snap = rig.poll(false);
        assert_eq!(snap.capabilities.interface_version, Some(3));
        assert!(!snap.capabilities.device_state);
        assert_eq!(snap.state.temperature_c.value, Some(6.0));
        assert_eq!(snap.backlash_owner, BacklashOwner::Firmware);
        assert_eq!(snap.backlash_model, BacklashModel::None);
        assert_eq!(rig.run(FocuserCmd::MoveTo(12_500)), OpOutcome::Succeeded);
        let rec = &rig.moves()[0];
        assert_eq!((rec.raw_start, rec.raw_end), (Some(12_000), Some(12_500)));
        assert_eq!(rec.temperature_start_c, Some(6.0));
        let puts: Vec<_> = fake
            .calls()
            .into_iter()
            .filter(|c| c.member == "move")
            .map(|c| c.params)
            .collect();
        assert_eq!(
            puts,
            vec![vec![("Position".to_string(), "12500".to_string())]]
        );
    }

    #[test]
    fn the_legacy_simulator_moves_through_the_session() {
        let clock = Arc::new(ManualClock::new());
        let drv = LegacySimFocuser::new(10_000, clock.clone());
        let mut rig = rig_with(Box::new(drv), None, clock, &configs(|_| {}), None);
        assert_eq!(rig.run(FocuserCmd::MoveTo(10_250)), OpOutcome::Succeeded);
        assert_eq!(rig.moves()[0].raw_end, Some(10_250));
        assert_eq!(
            rig.poll(false).state.temperature_c.state,
            ObsState::Unsupported
        );
    }

    #[test]
    fn profiles_parse_tolerantly_and_settings_resolve_with_the_profile_winning() {
        let (profile, warnings) = FocuserProfile::from_json(Some(&json!({
            "backlash": {"model": "sideways"},
            "settle_ms": 1200,
            "temperature_role": "tube",
            "relative_position": {"logical": 5, "state": "clean"},
            "last_used_option": "http://x"
        })));
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert_eq!(profile.backlash, None);
        assert_eq!(profile.restored_relative_position(), Some(5));
        let settings = FocuserSettings::from_configs(&configs(|_| {}), &profile);
        assert_eq!(settings.settle, Duration::from_millis(1200));
        assert_eq!(settings.temperature_role, TemperatureRole::Tube);
        assert_eq!(settings.backlash_model, BacklashModel::None);
        assert_eq!(FocuserProfile::from_json(Some(&json!(7))).1.len(), 1);
        assert_eq!(FocuserProfile::from_json(None).0, FocuserProfile::default());
        let empty = FocuserSettings {
            guard_min: 10,
            guard_max: 5,
            ..FocuserSettings::default()
        };
        assert_eq!(empty.travel_bounds(&FocuserCapabilities::default()), None);
        assert!(announces_lost_position(
            "Focuser position unknown after reset"
        ));
        assert!(!announces_lost_position("motor driver fault"));
        let observed: Observed<bool> = Observed::fresh(true, Source::Derived, Utc::now(), 1);
        assert!(observed.is_fresh());
    }
}

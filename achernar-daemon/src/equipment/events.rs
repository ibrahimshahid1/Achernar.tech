//! Equipment events and ground-truth records (spec §4.2, §4.3). Frozen by
//! WP0.
//!
//! Every event carries seq, event_id, UTC time, monotonic time, device kind,
//! connection id, an optional command id and an optional device time. The
//! hub assigns `seq` (continuing from `MAX(equipment_events.seq)` when a store
//! is attached) and drops events whose connection id does not match the live
//! connection (§2.8).
//!
//! JSON shape: the envelope fields plus `"kind"` (the catalogue name below)
//! and `"payload"` (the variant's fields), e.g.
//! `{"seq": 7, ..., "kind": "slew_settled", "payload": {"position": ...}}`.
//! The recorder stores `kind` and `payload` in `equipment_events.kind` and
//! `payload_json`.
//!
//! Additions to the §4.2 catalogue made by WP0: `operation_commanded`,
//! `operation_started` and `operation_ended`, the generic C-1 records every
//! operation emits. The `equipment_overheads` view (H18) reads
//! `operation_ended`.

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::commands::{Domain, GatedCommand, PixelUnit, Settle};
use super::ops::OpKind;
use super::snapshot::{
    AlgoParam, BacklashModel, BacklashOwner, CalibratorState, CoverState, ExposureRef,
    FocusDirection, GuideRms, GuideStepSample, GuiderAppState, MoveMode, PositionSource,
    ResumeBlock, SafetyGate, SafetySource, SettleProgress, ShutterState, WeatherReading,
};
use super::traits::Clock;
use super::types::{
    AltAz, AutonomyMode, ClockQuality, CommandId, CommandSource, DeviceError, DeviceIdentity,
    DeviceKind, DriveRate, EqCoord, EqFrame, ErrorClass, FailureClass, GuideDirection, JNowFlavour,
    MountSyncMode, OpOutcome, PierSide, Priority, Site, TemperatureRole,
};

// ---------------------------------------------------------------------------
// Payload helper types
// ---------------------------------------------------------------------------

/// Why a connection ended (`device_connections.end_class`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectionEnd {
    Operator,
    Lost,
    Shutdown,
    Replaced,
    Failed,
}

/// Per-device e-stop result (§2.9).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EstopDeviceResult {
    pub kind: DeviceKind,
    pub connection_id: Option<Uuid>,
    pub ok: bool,
    pub error: Option<DeviceError>,
    pub latency_ms: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitKind {
    Altitude,
    Horizon,
    HourAngle,
}

/// Meridian-flip workflow phases (M-l).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlipPhase {
    StopGuiding,
    WaitForHourAngle,
    Slew,
    Settle,
    Solve,
    Center,
    Autofocus,
    RestartGuiding,
    GuiderSettle,
    Done,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationPhase {
    Started,
    Step,
    Complete,
    Failed,
    DataFlipped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParkMethod {
    Driver,
    /// Tier 2 fallback park for mounts without CanPark.
    Fallback,
}

// ---------------------------------------------------------------------------
// Event catalogue
// ---------------------------------------------------------------------------

/// Defines [`EventKind`] and `EventKind::name`, so the serde tag and the
/// journal `kind` can never disagree.
macro_rules! event_catalogue {
    (
        $(
            $(#[$meta:meta])*
            $name:literal => $variant:ident $({ $($field:ident : $ty:ty),* $(,)? })?
        ),* $(,)?
    ) => {
        /// The frozen event catalogue (§4.2).
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        #[serde(tag = "kind", content = "payload")]
        pub enum EventKind {
            $(
                $(#[$meta])*
                #[serde(rename = $name)]
                $variant $({ $($field: $ty),* })?,
            )*
        }

        impl EventKind {
            /// Every catalogue name, in declaration order.
            pub const NAMES: &'static [&'static str] = &[$($name),*];

            /// The catalogue name (`equipment_events.kind`).
            pub fn name(&self) -> &'static str {
                match self {
                    $(EventKind::$variant { .. } => $name,)*
                }
            }
        }
    };
}

event_catalogue! {
    // ---- Lifecycle ----
    "device_connected" => DeviceConnected {
        identity: Box<DeviceIdentity>,
        capabilities: serde_json::Value,
    },
    "device_reconnected" => DeviceReconnected { reason: Option<String> },
    "device_degraded" => DeviceDegraded { class: FailureClass, reason: Option<String> },
    "device_lost" => DeviceLost { class: FailureClass, reason: Option<String> },
    "device_disconnected" => DeviceDisconnected { end_class: ConnectionEnd, reason: Option<String> },
    "capability_unsupported" => CapabilityUnsupported { member: String },
    "outcome_unknown" => OutcomeUnknown { op: OpKind, reason: String },
    "poll_cadence_reduced" => PollCadenceReduced { interval_ms: u64, factor: u32 },

    // ---- Generic operation records (C-1; added by WP0 for H18) ----
    "operation_commanded" => OperationCommanded { op: OpKind, source: CommandSource },
    "operation_started" => OperationStarted { op: OpKind },
    "operation_ended" => OperationEnded { op: OpKind, outcome: OpOutcome, duration_ms: u64 },

    // ---- Gate ----
    "ownership_conflict" => OwnershipConflict {
        domain: Domain,
        holder: Option<CommandId>,
        requested_by: CommandSource,
    },
    "safety_rejected" => SafetyRejected { check: String, reason: String },
    "override_applied" => OverrideApplied { domain: Domain, cancelled: Option<CommandId> },
    "resume_blocked" => ResumeBlocked { block: ResumeBlock },
    "resume_cleared" => ResumeCleared { block: ResumeBlock, by: CommandSource },
    "estop" => Estop { results: Vec<EstopDeviceResult> },

    // ---- Mount ----
    "slew_commanded" => SlewCommanded {
        target: EqCoord,
        native_target: Option<EqCoord>,
        altaz: Option<AltAz>,
    },
    "slew_started" => SlewStarted,
    "slew_arrived" => SlewArrived { position: Option<EqCoord> },
    /// Replaces the probe inference in `D/mount/plate_solve_trigger.rs`.
    "slew_settled" => SlewSettled { position: Option<EqCoord>, error_arcmin: Option<f64> },
    "slew_ended" => SlewEnded { outcome: OpOutcome, duration_ms: u64 },
    "arrived_off_target" => ArrivedOffTarget { error_arcmin: f64 },
    "tracking_changed" => TrackingChanged {
        on: Option<bool>,
        rate: Option<DriveRate>,
        source: Option<CommandSource>,
        reason: Option<String>,
    },
    "pier_side_changed" => PierSideChanged { from: PierSide, to: PierSide },
    "pier_side_static" => PierSideStatic { side: PierSide },
    "parked" => Parked { method: ParkMethod },
    "unparked" => Unparked,
    "homed" => Homed,
    "park_failed" => ParkFailed { class: ErrorClass, reason: String },
    "sync_applied" => SyncApplied {
        before: Option<EqCoord>,
        after: Option<EqCoord>,
        generation: u64,
    },
    "centered" => Centered { error_arcmin: f64, attempts: u32 },
    "center_failed" => CenterFailed { error_arcmin: Option<f64>, attempts: u32, reason: String },
    "pulse_guide" => PulseGuide { dir: GuideDirection, ms: u32, source: CommandSource },
    "move_axis" => MoveAxis { axis: u8, rate_deg_s: f64 },
    "limit_reached" => LimitReached { kind: LimitKind, value: Option<f64>, limit: f64 },
    "limit_watch_blind" => LimitWatchBlind { stale_ms: u64 },
    "site_mismatch" => SiteMismatch { configured: Option<Site>, mount: Option<Site> },
    "clock_mismatch" => ClockMismatch { delta_s: f64 },
    "meridian_flip" => MeridianFlip { phase: FlipPhase },

    // ---- Focuser ----
    "focuser_move_commanded" => FocuserMoveCommanded {
        move_id: Uuid,
        mode: MoveMode,
        target: i32,
        steps: Option<i32>,
    },
    "focuser_move_started" => FocuserMoveStarted { move_id: Uuid, from: Option<i32> },
    "focuser_move_stopped" => FocuserMoveStopped {
        move_id: Uuid,
        position: Option<i32>,
        outcome: OpOutcome,
    },
    "focuser_settled" => FocuserSettled {
        move_id: Option<Uuid>,
        position: Option<i32>,
        raw_position: Option<i32>,
    },
    "focuser_halted" => FocuserHalted { position: Option<i32> },
    "focuser_position_reset" => FocuserPositionReset {
        from: Option<i32>,
        to: Option<i32>,
        cause: String,
    },
    "tempcomp_changed" => TempcompChanged { from: Option<bool>, to: bool, reason: Option<String> },
    /// Rate-limited.
    "focuser_temperature" => FocuserTemperature { celsius: f64, role: TemperatureRole },

    // ---- Filter ----
    "filter_change_requested" => FilterChangeRequested {
        from_slot: Option<u32>,
        to_slot: u32,
        to_name: Option<String>,
    },
    "filter_moving" => FilterMoving { to_slot: u32 },
    "filter_arrived" => FilterArrived { slot: Option<u32>, verified: bool },
    "filter_settled" => FilterSettled { slot: u32, settle_ms: u64 },
    "filter_offset_applied" => FilterOffsetApplied { steps: i32, focuser_move_id: Option<Uuid> },
    "filter_change_ended" => FilterChangeEnded { outcome: OpOutcome, duration_ms: u64 },
    "filter_config_mismatch" => FilterConfigMismatch { differences: Vec<String> },
    "focus_verification_required" => FocusVerificationRequired { reason: String },

    // ---- Guider ----
    /// Not journaled in `equipment_events`; stored in `guide_steps`.
    "guide_step" => GuideStep { step: Box<GuideStepSample> },
    "guider_state" => GuiderState { from: GuiderAppState, to: GuiderAppState },
    "star_lost" => StarLost {
        frame: Option<u64>,
        star_mass: Option<f64>,
        snr: Option<f64>,
        avg_dist: Option<f64>,
        error_code: Option<i32>,
        status: Option<String>,
    },
    "lock_position_lost" => LockPositionLost,
    "settle_begin" => SettleBegin,
    "settling" => Settling { progress: SettleProgress },
    "settle_done" => SettleDone {
        status: i32,
        error: Option<String>,
        total_frames: Option<u32>,
        dropped_frames: Option<u32>,
    },
    "dithered" => Dithered { dx: f64, dy: f64 },
    "guiding_started" => GuidingStarted,
    "guiding_stopped" => GuidingStopped,
    "guider_paused" => GuiderPaused,
    "guider_resumed" => GuiderResumed,
    "calibration" => Calibration { phase: CalibrationPhase, detail: Option<String> },
    /// Operator edits in PHD2 confound M-1 data.
    "guide_param_changed" => GuideParamChanged { name: String, value: serde_json::Value },
    "guider_alert" => GuiderAlert { message: String, alert_type: Option<String> },
    "camera_role_conflict" => CameraRoleConflict { camera_name: String, guider_camera: String },

    // ---- Environment ----
    "weather_sample" => WeatherSample { reading: Box<WeatherReading> },
    "safety_changed" => SafetyChanged { state: SafetyGate, sources: Vec<SafetySource> },
    "cover_changed" => CoverChanged { from: CoverState, to: CoverState },
    "calibrator_changed" => CalibratorChanged { state: CalibratorState, brightness: Option<u32> },
    "rotator_moved" => RotatorMoved { from: Option<f64>, to: Option<f64> },
    "switch_changed" => SwitchChanged { id: u32, name: String, role: Option<String>, value: f64 },
    "dome_shutter_changed" => DomeShutterChanged { from: ShutterState, to: ShutterState },

    // ---- Acquisition (camera hook, autofocus) ----
    "exposure_started" => ExposureStarted { exposure: ExposureRef },
    "exposure_completed" => ExposureCompleted { exposure: ExposureRef, completed_at: DateTime<Utc> },
    "exposure_settings_changed" => ExposureSettingsChanged { settings: serde_json::Value },
    "stream_restarted" => StreamRestarted { generation: u64 },
    "autofocus_started" => AutofocusStarted { run_id: String },
    "autofocus_verified" => AutofocusVerified {
        run_id: String,
        position_readback: Option<i32>,
        hfr: Option<f64>,
        frame_id: Option<String>,
        measured_at: DateTime<Utc>,
    },
    "autofocus_failed" => AutofocusFailed { run_id: String, reason: String },
}

/// One journal entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EquipmentEvent {
    /// Assigned by the hub; 0 until then.
    pub seq: u64,
    pub event_id: Uuid,
    pub at_utc: DateTime<Utc>,
    pub mono_ns: u64,
    /// `None` for gate-wide events (e-stop, resume blocks).
    pub device_kind: Option<DeviceKind>,
    pub connection_id: Option<Uuid>,
    pub command_id: Option<CommandId>,
    pub device_time: Option<DateTime<Utc>>,
    #[serde(flatten)]
    pub event: EventKind,
}

impl EquipmentEvent {
    /// A new event timestamped from `clock`, not yet sequenced.
    pub fn new(event: EventKind, clock: &dyn Clock) -> Self {
        Self {
            seq: 0,
            event_id: Uuid::new_v4(),
            at_utc: clock.now_utc(),
            mono_ns: clock.now_mono_ns(),
            device_kind: None,
            connection_id: None,
            command_id: None,
            device_time: None,
            event,
        }
    }

    pub fn for_device(mut self, kind: DeviceKind, connection_id: Uuid) -> Self {
        self.device_kind = Some(kind);
        self.connection_id = Some(connection_id);
        self
    }

    pub fn with_command(mut self, command_id: CommandId) -> Self {
        self.command_id = Some(command_id);
        self
    }

    pub fn with_device_time(mut self, device_time: DateTime<Utc>) -> Self {
        self.device_time = Some(device_time);
        self
    }

    pub fn name(&self) -> &'static str {
        self.event.name()
    }

    /// Written to `equipment_events`. Guide steps go to `guide_steps`.
    pub fn journaled(&self) -> bool {
        !matches!(self.event, EventKind::GuideStep { .. })
    }
}

// ---------------------------------------------------------------------------
// Ground-truth records (§4.3). One struct per table row; the repositories
// (WP1A-9) map fields to columns of the same name unless noted.
// ---------------------------------------------------------------------------

/// Everything the recorder persists. `RecorderSink::record` never blocks.
#[derive(Debug, Clone)]
pub enum GroundTruthRecord {
    /// `equipment_events` (skipped when `!event.journaled()`).
    Event(Arc<EquipmentEvent>),
    /// Insert into `device_connections`.
    ConnectionOpened(Box<DeviceConnectionRecord>),
    /// Close a `device_connections` row.
    ConnectionClosed {
        connection_id: Uuid,
        disconnected_at: DateTime<Utc>,
        end_class: ConnectionEnd,
    },
    FocuserMove(Box<FocuserMoveRecord>),
    GuideStep(Box<GuideStepRecord>),
    Dither(Box<DitherRecord>),
    ExposureWindow(Box<ExposureWindowRow>),
    PointingSample(Box<PointingSampleRecord>),
    /// H23 decision record (`policy_decisions`).
    Decision(Box<PolicyDecisionRecord>),
    /// One `control_audit_logs` row per gate stage (H21).
    Audit(Box<AuditRecord>),
    /// `environment_log` with `source = 'probe'`.
    Environment(Box<EnvironmentRecord>),
    FilterSwitch(Box<FilterSwitchRecord>),
    /// One `guider_state_log` row per exposure window (H9).
    GuiderWindow(Box<GuiderStateRecord>),
    /// 1 Hz downsample to `telemetry_log` (H19).
    Telemetry(TelemetrySample),
    /// Merge into `device_profiles.profile_json`.
    Profile(ProfileUpdate),
}

/// `device_connections`. `id` is `identity.connection_id`; `endpoint` is
/// `identity.endpoint.describe()`; `qualification` is
/// `identity.qualification.as_str()`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceConnectionRecord {
    pub identity: DeviceIdentity,
    pub capabilities: serde_json::Value,
    pub connected_at: DateTime<Utc>,
    pub session_id: Option<String>,
    pub night_id: Option<String>,
}

/// `focuser_moves`: one row per move, with read-back (F-a .. F-i).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FocuserMoveRecord {
    pub id: Uuid,
    pub command_id: Option<CommandId>,
    pub connection_id: Uuid,
    pub mode: MoveMode,
    pub commanded_target: i32,
    pub logical_start: Option<i32>,
    pub logical_end: Option<i32>,
    pub raw_start: Option<i32>,
    pub raw_end: Option<i32>,
    pub position_source: PositionSource,
    pub direction: Option<FocusDirection>,
    pub reversal: Option<bool>,
    pub backlash_model: BacklashModel,
    pub backlash_owner: BacklashOwner,
    pub tempcomp_before: Option<bool>,
    pub tempcomp_after: Option<bool>,
    pub temperature_start_c: Option<f64>,
    pub temperature_end_c: Option<f64>,
    pub commanded_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub stopped_at: Option<DateTime<Utc>>,
    pub settled_at: Option<DateTime<Utc>>,
    pub outcome: OpOutcome,
    pub source: CommandSource,
    pub run_id: Option<String>,
}

/// `guide_steps`: one raw PHD2 GuideStep.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuideStepRecord {
    pub connection_id: Uuid,
    pub step: GuideStepSample,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DitherBackend {
    Phd2,
    Direct,
}

/// `dither_events` (H10). `next_frame_id` and `next_frame_accepted` are
/// linked after the next exposure by the recorder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DitherRecord {
    pub id: Uuid,
    pub command_id: Option<CommandId>,
    pub source: CommandSource,
    pub policy_version: Option<String>,
    pub backend: DitherBackend,
    pub amount: f64,
    pub amount_unit: PixelUnit,
    pub ra_only: bool,
    pub settle: Settle,
    pub requested_at: DateTime<Utc>,
    pub dx: Option<f64>,
    pub dy: Option<f64>,
    pub settle_done_at: Option<DateTime<Utc>>,
    pub settle_status: Option<i32>,
    pub settle_error: Option<String>,
    pub total_frames: Option<u32>,
    pub dropped_frames: Option<u32>,
    pub max_distance_px: Option<f64>,
    pub next_frame_id: Option<String>,
    pub next_frame_accepted: Option<bool>,
}

/// `exposure_device_windows` (H1), one row per committed frame. Built by the
/// capture integration (WP1B-3) from a `DeviceWindow`. Missing values stay
/// `None`; they are never written as 0.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ExposureWindowRow {
    pub frame_id: String,
    pub exposure_started_at: Option<DateTime<Utc>>,
    pub exposure_completed_at: Option<DateTime<Utc>>,
    pub clock_quality: ClockQuality,
    pub finalised: bool,
    pub mount_start_json: Option<serde_json::Value>,
    pub mount_mid_json: Option<serde_json::Value>,
    pub mount_end_json: Option<serde_json::Value>,
    pub mount_frame: Option<EqFrame>,
    pub pier_side: Option<PierSide>,
    /// True only if tracking was on throughout; false if it was off at any
    /// point; `None` if unknown.
    pub tracking: Option<bool>,
    pub slewing_any: Option<bool>,
    pub pulse_guiding_any: Option<bool>,
    pub focuser_pos_start: Option<i32>,
    pub focuser_pos_end: Option<i32>,
    pub focuser_moved: Option<bool>,
    pub focuser_temperature_c: Option<f64>,
    pub filter_slot: Option<u32>,
    pub filter_name: Option<String>,
    pub filter_verified: Option<bool>,
    pub rotator_angle_deg: Option<f64>,
    pub guiding_state: Option<GuiderAppState>,
    pub guide_rms_ra_px: Option<f64>,
    pub guide_rms_dec_px: Option<f64>,
    pub guide_rms_total_arcsec: Option<f64>,
    /// RMS about zero.
    pub guide_rms0_total_arcsec: Option<f64>,
    pub guide_peak_ra_px: Option<f64>,
    pub guide_peak_dec_px: Option<f64>,
    pub guide_steps_n: Option<u32>,
    pub dither_in_window: Option<bool>,
    pub settle_in_window: Option<bool>,
    pub safety_state: Option<SafetyGate>,
    pub weather_json: Option<serde_json::Value>,
    pub cover_json: Option<serde_json::Value>,
    pub resume_block_active: bool,
    pub quiescent: Option<bool>,
    pub quiescence_reason: Option<String>,
    /// Event `seq` values inside the window.
    pub events: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PointingPurpose {
    Center,
    AutoSolve,
    Manual,
}

/// `pointing_samples` (H12; M-4, M-3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PointingSampleRecord {
    pub id: Uuid,
    pub plate_solve_event_id: Option<String>,
    pub frame_id: Option<String>,
    pub mount_identity_key: String,
    pub mount_frame: EqFrame,
    pub jnow_flavour: JNowFlavour,
    pub does_refraction: Option<bool>,
    pub mount_ra_native_h: Option<f64>,
    pub mount_dec_native_deg: Option<f64>,
    pub mount_ra_j2000_h: Option<f64>,
    pub mount_dec_j2000_deg: Option<f64>,
    pub solved_ra_j2000_h: f64,
    pub solved_dec_j2000_deg: f64,
    /// Multiplied by cos(dec).
    pub err_ra_arcsec: Option<f64>,
    pub err_dec_arcsec: Option<f64>,
    pub err_total_arcsec: Option<f64>,
    pub pier_side: PierSide,
    pub ha_h: Option<f64>,
    pub lst_h: Option<f64>,
    pub alt_deg: Option<f64>,
    pub az_deg: Option<f64>,
    pub sync_generation: u64,
    pub sync_mode: MountSyncMode,
    pub coordinate_precision_arcsec: Option<f64>,
    pub temperature_c: Option<f64>,
    pub mid_exposure_utc: Option<DateTime<Utc>>,
    pub purpose: PointingPurpose,
}

/// `policy_decisions` (H23; DC §7 "Decision"). Propensities are present only
/// when the choice was randomised.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyDecisionRecord {
    pub decision_id: Uuid,
    pub policy: String,
    pub policy_version: String,
    pub mode: AutonomyMode,
    pub context: serde_json::Value,
    /// Age of each input measurement, ms, by input name.
    pub measurement_ages_ms: BTreeMap<String, u64>,
    pub safe_arms: Vec<String>,
    pub chosen_arm: Option<String>,
    pub propensities: Option<serde_json::Value>,
    pub requested_action: Option<serde_json::Value>,
    pub validated_action: Option<serde_json::Value>,
    pub executed_action: Option<serde_json::Value>,
    pub command_id: Option<CommandId>,
    pub reward_components: Option<serde_json::Value>,
    pub attribution_window_start: Option<DateTime<Utc>>,
    pub attribution_window_end: Option<DateTime<Utc>>,
    pub interrupted: bool,
    /// Column `override`.
    pub override_applied: bool,
    pub parent_frame_ids: Vec<String>,
    pub created_at: DateTime<Utc>,
}

/// Gate audit stage (v6 §7.2: proposed, then validated or rejected, then
/// executed, then outcome).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditStage {
    Proposed,
    Validated,
    Rejected,
    Executed,
    Outcome,
}

/// One `control_audit_logs` row. Columns: `actor = source.label()`,
/// `action = op` (serde name), `target` = the domains, `payload_json` = this
/// record, `accepted`, `reason`, plus `command_id` and `stage` (added by
/// migration 009).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditRecord {
    pub command_id: CommandId,
    pub stage: AuditStage,
    pub op: OpKind,
    pub source: CommandSource,
    pub mode: AutonomyMode,
    pub priority: Priority,
    pub domains: Vec<Domain>,
    pub params: serde_json::Value,
    pub policy_version: Option<String>,
    pub accepted: bool,
    pub reason: Option<String>,
    pub override_applied: bool,
    /// Accepted-with-note annotations from safety checks.
    pub annotations: Vec<String>,
    pub outcome: Option<OpOutcome>,
    pub at: DateTime<Utc>,
}

impl AuditRecord {
    pub fn for_command(
        cmd: &GatedCommand,
        stage: AuditStage,
        accepted: bool,
        reason: Option<String>,
        at: DateTime<Utc>,
    ) -> Self {
        let policy_version = match &cmd.source {
            CommandSource::Policy { version, .. } => Some(version.clone()),
            _ => None,
        };
        Self {
            command_id: cmd.command_id,
            stage,
            op: cmd.kind(),
            source: cmd.source.clone(),
            mode: cmd.mode,
            priority: cmd.priority,
            domains: cmd.domains.clone(),
            params: cmd.params.clone(),
            policy_version,
            accepted,
            reason,
            override_applied: cmd.overriding,
            annotations: Vec::new(),
            outcome: None,
            at,
        }
    }
}

/// `environment_log` probe row (§3.9, F-f).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct EnvironmentRecord {
    pub observed_at: DateTime<Utc>,
    pub session_id: Option<String>,
    pub night_id: Option<String>,
    pub rig_id: Option<String>,
    pub device_identity_key: Option<String>,
    pub ambient_temp_c: Option<f64>,
    pub humidity_pct: Option<f64>,
    pub dew_point_c: Option<f64>,
    pub pressure_hpa: Option<f64>,
    pub cloud_cover_pct: Option<f64>,
    pub sky_temp_c: Option<f64>,
    pub sky_quality: Option<f64>,
    pub sky_brightness_lux: Option<f64>,
    pub star_fwhm_arcsec: Option<f64>,
    pub rain_rate_mm_h: Option<f64>,
    pub wind_speed_ms: Option<f64>,
    pub wind_gust_ms: Option<f64>,
    pub wind_dir_deg: Option<f64>,
    /// Column `sensor_ages_json`.
    pub sensor_ages_s: BTreeMap<String, f64>,
}

impl EnvironmentRecord {
    /// A probe row from a weather reading. Only known values are copied.
    pub fn from_weather(
        observed_at: DateTime<Utc>,
        device_identity_key: Option<String>,
        reading: &WeatherReading,
    ) -> Self {
        let v = |o: &super::types::Observed<f64>| o.known().copied().filter(|x| x.is_finite());
        Self {
            observed_at,
            device_identity_key,
            ambient_temp_c: v(&reading.temperature_c),
            humidity_pct: v(&reading.humidity_pct),
            dew_point_c: v(&reading.dew_point_c),
            pressure_hpa: v(&reading.pressure_hpa),
            cloud_cover_pct: v(&reading.cloud_cover_pct),
            sky_temp_c: v(&reading.sky_temperature_c),
            sky_quality: v(&reading.sky_quality_mag_arcsec2),
            sky_brightness_lux: v(&reading.sky_brightness_lux),
            star_fwhm_arcsec: v(&reading.star_fwhm_arcsec),
            rain_rate_mm_h: v(&reading.rain_rate_mm_h),
            wind_speed_ms: v(&reading.wind_speed_ms),
            wind_gust_ms: v(&reading.wind_gust_ms),
            wind_dir_deg: v(&reading.wind_direction_deg),
            sensor_ages_s: reading.sensor_ages_s.clone(),
            ..Self::default()
        }
    }
}

/// `filter_switch_events` (existing columns plus the 009 extension; H8).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterSwitchRecord {
    pub id: Uuid,
    pub session_id: Option<String>,
    pub from_filter: Option<String>,
    pub to_filter: String,
    /// Column `focus_offset`: the offset of the new filter.
    pub focus_offset: Option<i32>,
    pub duration_ms: Option<u64>,
    pub success: bool,
    pub command_id: Option<CommandId>,
    pub from_slot: Option<u32>,
    pub to_slot: u32,
    pub requested_at: DateTime<Utc>,
    pub wheel_done_at: Option<DateTime<Utc>>,
    pub verified_at: Option<DateTime<Utc>>,
    pub settle_ms_actual: Option<u64>,
    pub offset_from: Option<i32>,
    pub offset_to: Option<i32>,
    pub offset_applied_steps: Option<i32>,
    pub focuser_move_id: Option<Uuid>,
    pub guiding_paused: bool,
    pub handshake_ok: bool,
    pub outcome: OpOutcome,
    pub trigger: Option<String>,
    pub policy_version: Option<String>,
    pub arm: Option<String>,
    pub propensity: Option<f64>,
    pub background_before: Option<f64>,
    pub background_after: Option<f64>,
}

/// `guider_state_log`, one row per exposure window (H9). The existing
/// `ra_rms`, `dec_rms` and `total_rms` columns take `rms.ra_arcsec`,
/// `rms.dec_arcsec` and `rms.total_arcsec`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuiderStateRecord {
    pub timestamp: DateTime<Utc>,
    pub session_id: Option<String>,
    pub connection_id: Uuid,
    pub frame_id: Option<String>,
    pub window_start: DateTime<Utc>,
    pub window_end: DateTime<Utc>,
    pub rms: Option<GuideRms>,
    pub ra_aggressiveness: Option<f64>,
    pub dec_aggressiveness: Option<f64>,
    pub guide_star_snr: Option<f64>,
    pub dither_active: bool,
    pub algorithm_params: Vec<AlgoParam>,
    pub executed_arm_ra: Option<String>,
    pub executed_arm_dec: Option<String>,
    pub policy_version: Option<String>,
    pub freeze_reason: Option<String>,
}

/// `telemetry_log`: `source` = device kind, `event_type = 'snapshot'`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetrySample {
    pub source: DeviceKind,
    pub at: DateTime<Utc>,
    pub payload: serde_json::Value,
}

/// A JSON merge patch for `device_profiles.profile_json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileUpdate {
    pub identity_key: String,
    pub kind: DeviceKind,
    pub patch: serde_json::Value,
    pub updated_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::commands::{GatedOp, MountCmd};
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::types::{DriverId, Endpoint, Observed, Source};
    use serde_json::json;

    fn round_trip(event: &EquipmentEvent) -> serde_json::Value {
        let value = serde_json::to_value(event).unwrap();
        let back: EquipmentEvent = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(&back, event, "{value}");
        value
    }

    #[test]
    fn events_round_trip_with_kind_and_payload() {
        let clock = ManualClock::new();
        let connection = Uuid::new_v4();
        let unit = EquipmentEvent::new(EventKind::SlewStarted, &clock)
            .for_device(DeviceKind::Mount, connection)
            .with_command(CommandId::new());
        let value = round_trip(&unit);
        assert_eq!(value["kind"], json!("slew_started"));
        assert_eq!(value["device_kind"], json!("mount"));

        let settled = EquipmentEvent::new(
            EventKind::SlewSettled {
                position: Some(EqCoord::new(5.0, 10.0, EqFrame::JNowApparent)),
                error_arcmin: Some(0.4),
            },
            &clock,
        );
        let value = round_trip(&settled);
        assert_eq!(
            value["payload"]["position"]["frame"],
            json!("jnow_apparent")
        );

        let pier = EquipmentEvent::new(
            EventKind::PierSideChanged {
                from: PierSide::East,
                to: PierSide::Unknown,
            },
            &clock,
        );
        let value = round_trip(&pier);
        assert_eq!(value["payload"], json!({"from": "east", "to": null}));

        let ended = EquipmentEvent::new(
            EventKind::OperationEnded {
                op: OpKind::Slew,
                outcome: OpOutcome::Failed(DeviceError::from_alpaca_error(0x408, "parked")),
                duration_ms: 1234,
            },
            &clock,
        );
        let value = round_trip(&ended);
        assert_eq!(value["payload"]["op"], json!("slew"));
        assert_eq!(value["payload"]["outcome"]["status"], json!("failed"));
        assert_eq!(value["payload"]["duration_ms"], json!(1234));

        let identity = DeviceIdentity::new(
            DeviceKind::Focuser,
            DriverId::Alpaca,
            Endpoint::AlpacaHttp {
                base: "http://esp32.local:80".into(),
                device_type: "focuser".into(),
                number: 0,
            },
            "alpaca:esp32.local:80/focuser/0",
            "ESP32 Focuser",
        );
        round_trip(&EquipmentEvent::new(
            EventKind::DeviceConnected {
                identity: Box::new(identity),
                capabilities: json!({"absolute": true}),
            },
            &clock,
        ));
        let reading = WeatherReading {
            temperature_c: Observed::fresh(4.5, Source::Property, clock.now_utc(), 1),
            ..WeatherReading::default()
        };
        round_trip(&EquipmentEvent::new(
            EventKind::WeatherSample {
                reading: Box::new(reading),
            },
            &clock,
        ));
        round_trip(&EquipmentEvent::new(
            EventKind::GuideStep {
                step: Box::new(GuideStepSample {
                    frame: 12,
                    dx: -0.25,
                    dy: 0.5,
                    ..GuideStepSample::default()
                }),
            },
            &clock,
        ));
    }

    #[test]
    fn catalogue_names_match_serde_tags_and_are_unique() {
        let clock = ManualClock::new();
        let samples = [
            EventKind::Unparked,
            EventKind::LockPositionLost,
            EventKind::Estop { results: vec![] },
            EventKind::FocusVerificationRequired {
                reason: "estimated offset".into(),
            },
        ];
        for kind in samples {
            let name = kind.name();
            let value = serde_json::to_value(EquipmentEvent::new(kind, &clock)).unwrap();
            assert_eq!(value["kind"], json!(name));
            assert!(EventKind::NAMES.contains(&name));
        }
        let mut names = EventKind::NAMES.to_vec();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "duplicate catalogue names");
        for required in [
            "device_connected",
            "slew_settled",
            "pier_side_changed",
            "focuser_settled",
            "filter_change_ended",
            "guide_step",
            "settle_done",
            "safety_changed",
            "exposure_completed",
            "autofocus_verified",
            "operation_ended",
        ] {
            assert!(EventKind::NAMES.contains(&required), "{required}");
        }
    }

    #[test]
    fn guide_steps_are_not_journaled() {
        let clock = ManualClock::new();
        let step = EquipmentEvent::new(
            EventKind::GuideStep {
                step: Box::default(),
            },
            &clock,
        );
        assert!(!step.journaled());
        assert!(EquipmentEvent::new(EventKind::Dithered { dx: 1.0, dy: 2.0 }, &clock).journaled());
    }

    #[test]
    fn audit_records_take_policy_version_from_the_source() {
        let cmd = GatedCommand::new(
            GatedOp::Mount(MountCmd::Park),
            CommandSource::Policy {
                model: "m3".into(),
                version: "0.2.0".into(),
            },
            AutonomyMode::Proposal,
            Priority::MountRecovery,
        );
        let rec = AuditRecord::for_command(&cmd, AuditStage::Proposed, true, None, Utc::now());
        assert_eq!(rec.policy_version.as_deref(), Some("0.2.0"));
        assert_eq!(rec.op, OpKind::Park);
        let text = serde_json::to_string(&rec).unwrap();
        assert_eq!(serde_json::from_str::<AuditRecord>(&text).unwrap(), rec);
    }

    #[test]
    fn environment_rows_copy_only_known_values() {
        let at = Utc::now();
        let reading = WeatherReading {
            humidity_pct: Observed::fresh(81.0, Source::Property, at, 1),
            temperature_c: Observed::fresh(f64::NAN, Source::Property, at, 1),
            sensor_ages_s: [("Humidity".to_string(), 4.0)].into_iter().collect(),
            ..WeatherReading::default()
        };
        let row = EnvironmentRecord::from_weather(at, Some("alpaca:oc".into()), &reading);
        assert_eq!(row.humidity_pct, Some(81.0));
        assert_eq!(row.ambient_temp_c, None, "NaN is unknown, never a number");
        assert_eq!(row.cloud_cover_pct, None);
        assert_eq!(row.sensor_ages_s["Humidity"], 4.0);
    }
}

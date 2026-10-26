//! Typed device states, snapshots, windows and model contexts (spec §2.4,
//! §2.13, §4.1). Frozen by WP0.
//!
//! Layering:
//! - `*State` / `*Reading` / `*Capabilities`: what a driver returns.
//! - `*Snapshot`: what a worker publishes after each poll (`watch` plus the
//!   ring buffer): the driver state plus derived and operation fields.
//! - [`EquipmentSnapshot`]: the hub's aggregate, read by models and gate
//!   safety checks.
//!
//! Units follow §2.14: RA in hours; Dec, Alt and Az in degrees; guide rates in
//! arcsec/s; temperatures in °C; focuser positions in integer steps.

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::events::EquipmentEvent;
use super::ops::{ActiveOp, OpKind};
use super::types::{
    AlignmentMode, AltAz, ClockQuality, CommandId, DeviceKind, DriveRate, EqCoord, EqFrame,
    LinkState, Observed, PierSide, Site, TemperatureRole,
};

/// Serde for `Option<Arc<T>>` without serde's `rc` feature.
mod arc_opt {
    use std::sync::Arc;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<T: Serialize, S: Serializer>(
        value: &Option<Arc<T>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value.as_deref().serialize(serializer)
    }

    pub fn deserialize<'de, T: Deserialize<'de>, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Arc<T>>, D::Error> {
        Ok(Option::<T>::deserialize(deserializer)?.map(Arc::new))
    }
}

// ---------------------------------------------------------------------------
// Mount
// ---------------------------------------------------------------------------

/// One `axisrates` interval, deg/s.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct RateRange {
    pub min_deg_s: f64,
    pub max_deg_s: f64,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct AxisRates {
    /// 0 primary (RA/Az), 1 secondary (Dec/Alt), 2 tertiary.
    pub axis: u8,
    pub ranges: Vec<RateRange>,
}

/// Mount capabilities, probed at connect (§2.6 step 6). An unknown Can* flag
/// is `false` and its member is listed in `unsupported_members`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct MountCapabilities {
    pub can_park: bool,
    pub can_unpark: bool,
    pub can_set_park: bool,
    pub can_find_home: bool,
    pub can_slew: bool,
    pub can_slew_async: bool,
    pub can_slew_altaz: bool,
    pub can_slew_altaz_async: bool,
    pub can_sync: bool,
    pub can_sync_altaz: bool,
    pub can_set_tracking: bool,
    pub can_set_pier_side: bool,
    pub can_pulse_guide: bool,
    pub can_set_guide_rates: bool,
    /// `canmoveaxis` for axes 0, 1 and 2.
    pub can_move_axis: [bool; 3],
    pub can_set_declination_rate: bool,
    pub can_set_right_ascension_rate: bool,
    pub alignment_mode: Option<AlignmentMode>,
    /// Raw ASCOM `EquatorialSystem`, when read (InterfaceVersion >= 2).
    pub equatorial_system: Option<i32>,
    /// Native frame after M-a (and any `equatorial_system_override`).
    pub native_frame: EqFrame,
    /// Why `native_frame` is `Unknown`, e.g. `unsupported_epoch`.
    pub frame_reason: Option<String>,
    pub does_refraction: Option<bool>,
    pub tracking_rates: Vec<DriveRate>,
    pub axis_rates: Vec<AxisRates>,
    /// Driver claims (M-m); the rig profile stays authoritative.
    pub focal_length_mm: Option<f64>,
    pub aperture_diameter_mm: Option<f64>,
    /// Coordinate precision, e.g. about 60" for low-precision LX200 through
    /// INDI (M-4 gap 2).
    pub coordinate_precision_arcsec: Option<f64>,
    pub interface_version: Option<i32>,
    pub device_state: bool,
    pub unsupported_members: Vec<String>,
}

/// One poll of a mount (R1-R10). Coordinates are in the native frame.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct MountState {
    pub ra_h: Observed<f64>,
    pub dec_deg: Observed<f64>,
    pub alt_deg: Observed<f64>,
    pub az_deg: Observed<f64>,
    pub sidereal_time_h: Observed<f64>,
    pub utc: Observed<DateTime<Utc>>,
    pub slewing: Observed<bool>,
    pub tracking: Observed<bool>,
    pub tracking_rate: Observed<DriveRate>,
    pub at_park: Observed<bool>,
    pub at_home: Observed<bool>,
    pub side_of_pier: Observed<PierSide>,
    pub is_pulse_guiding: Observed<bool>,
    pub guide_rate_ra_arcsec_s: Observed<f64>,
    pub guide_rate_dec_arcsec_s: Observed<f64>,
    pub site: Observed<Site>,
    /// TargetRightAscension / TargetDeclination, read on slew.
    pub target: Observed<EqCoord>,
}

/// Published mount snapshot (§2.13 `mount` block).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct MountSnapshot {
    pub connection_id: Uuid,
    pub state: MountState,
    pub capabilities: MountCapabilities,
    /// Mount position converted to J2000 by the `FrameConverter`; `None` when
    /// the native frame is unknown or the position is unknown.
    pub j2000: Option<EqCoord>,
    /// Hour angle, hours, from LST and RA (mount or astro).
    pub ha_h: Option<f64>,
    /// `can_flip` (D14): GermanPolar and a SideOfPier that actually changes.
    pub can_flip: bool,
    /// SideOfPier never changed on a GermanPolar mount (M-j).
    pub pier_side_static: bool,
    /// Mount UTC minus host UTC, seconds (M-b).
    pub clock_delta_s: Option<f64>,
    /// Incremented by every sync; segments M-4 residual series (M-g).
    pub sync_generation: u64,
    /// A MoveAxis jog is running and needs renewals (M-h).
    pub jog_active: bool,
    /// Answer to the latest `MountCmd::QueryDestinationSideOfPier` (M-j).
    pub destination_side_of_pier: Option<DestinationSideOfPier>,
    pub active_op: Option<ActiveOp>,
    pub published_at: DateTime<Utc>,
    pub published_mono_ns: u64,
}

/// `destinationsideofpier` for a planned target.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DestinationSideOfPier {
    pub target: EqCoord,
    pub side: PierSide,
    pub at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// Focuser
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct FocuserCapabilities {
    pub absolute: bool,
    pub max_step: Option<i32>,
    pub max_increment: Option<i32>,
    /// Alpaca `stepsize` (µm), a driver claim (F-g).
    pub step_size_um: Option<f64>,
    pub temp_comp_available: bool,
    pub can_sync: bool,
    pub interface_version: Option<i32>,
    pub device_state: bool,
    pub unsupported_members: Vec<String>,
}

/// One poll of a focuser (R12-R15). `position` is the raw device position.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct FocuserState {
    pub position: Observed<i32>,
    pub is_moving: Observed<bool>,
    pub temperature_c: Observed<f64>,
    pub temp_comp: Observed<bool>,
}

/// Where the focuser's logical position comes from (F-b, §2.15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionSource {
    /// An absolute focuser's own position.
    Device,
    /// Relative focuser, origin known this session (sync or operator
    /// declaration).
    Emulated,
    /// Relative focuser, restored from a cleanly closed profile row.
    EmulatedRestored,
    /// Relative focuser with no known origin: relative moves only, no
    /// autofocus, no FOP labels.
    #[default]
    EmulatedOriginUnknown,
    /// A relative focuser reconnected; its origin is lost.
    UnknownAfterReconnect,
}

impl PositionSource {
    /// v6 §7.6.2 "reliable absolute position".
    pub fn is_reliable(self) -> bool {
        matches!(
            self,
            PositionSource::Device | PositionSource::Emulated | PositionSource::EmulatedRestored
        )
    }
}

/// Backlash model (F-e).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "model", rename_all = "snake_case")]
pub enum BacklashModel {
    #[default]
    None,
    /// Overshoot past the target in the direction of travel, then return.
    Overshoot { in_steps: i32, out_steps: i32 },
    /// Compensate only on a reversal, hiding the offset in the logical
    /// position.
    Absolute { in_steps: i32, out_steps: i32 },
}

/// Who compensates backlash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BacklashOwner {
    #[default]
    Achernar,
    /// The device compensates internally; Achernar applies none.
    Firmware,
}

/// Focuser travel direction. `Out` is increasing step count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FocusDirection {
    In,
    Out,
}

/// How a focuser move was commanded (`focuser_moves.mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoveMode {
    #[serde(rename = "abs")]
    Absolute,
    #[serde(rename = "rel")]
    Relative,
}

/// Published focuser snapshot (§2.13 `focuser` block).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct FocuserSnapshot {
    pub connection_id: Uuid,
    pub state: FocuserState,
    pub capabilities: FocuserCapabilities,
    /// Logical position (backlash and relative emulation hidden).
    pub logical_position: Option<i32>,
    pub position_source: PositionSource,
    pub backlash_model: BacklashModel,
    pub backlash_owner: BacklashOwner,
    pub temperature_role: TemperatureRole,
    pub settling: bool,
    pub active_op: Option<ActiveOp>,
    pub published_at: DateTime<Utc>,
    pub published_mono_ns: u64,
}

// ---------------------------------------------------------------------------
// Filter wheel
// ---------------------------------------------------------------------------

/// What the device reports for one slot.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SlotInfo {
    pub index0: u32,
    pub device_name: Option<String>,
    /// Alpaca `focusoffsets`; INDI has none.
    pub device_focus_offset: Option<i32>,
}

/// Where a reconciled slot value came from (W-b).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotValueSource {
    Profile,
    Device,
    #[default]
    None,
}

/// A reconciled slot (W-b).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct FilterSlot {
    pub index: u32,
    /// Profile name, else device name, else `Slot n`.
    pub name: String,
    pub name_source: SlotValueSource,
    /// Profile offset, else device offset. TOML seeds are never applied (D17).
    pub offset: Option<i32>,
    pub offset_source: SlotValueSource,
}

/// Published filter-wheel snapshot (§2.13 `filter_wheel` block).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct FilterWheelSnapshot {
    pub connection_id: Uuid,
    /// 0-based slot. The value is `None` while the wheel moves (Alpaca -1,
    /// INDI Busy) or when it is unknown; `moving` says which.
    pub position: Observed<u32>,
    pub moving: bool,
    pub slots: Vec<FilterSlot>,
    /// The slot last verified by a handshake read (W-c step 7).
    pub verified_slot: Option<u32>,
    pub verified: bool,
    pub active_op: Option<ActiveOp>,
    pub published_at: DateTime<Utc>,
    pub published_mono_ns: u64,
}

// ---------------------------------------------------------------------------
// Guider
// ---------------------------------------------------------------------------

/// PHD2 `AppState` (G-b).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuiderAppState {
    Stopped,
    Selected,
    Calibrating,
    Guiding,
    LostLock,
    Paused,
    Looping,
    #[default]
    Unknown,
}

impl GuiderAppState {
    /// Parse PHD2's `AppState` strings (`"LostLock"`, `"Guiding"`, ...).
    pub fn from_phd2(value: &str) -> Self {
        match value {
            "Stopped" => GuiderAppState::Stopped,
            "Selected" => GuiderAppState::Selected,
            "Calibrating" => GuiderAppState::Calibrating,
            "Guiding" => GuiderAppState::Guiding,
            "LostLock" => GuiderAppState::LostLock,
            "Paused" => GuiderAppState::Paused,
            "Looping" => GuiderAppState::Looping,
            _ => GuiderAppState::Unknown,
        }
    }

    /// D20: Achernar's own pulses are refused in these states.
    pub fn owns_mount(self) -> bool {
        matches!(self, GuiderAppState::Guiding | GuiderAppState::Calibrating)
    }
}

/// Phase tag of a guide step (G-h).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuidePhase {
    #[default]
    Guiding,
    Dithering,
    Settling,
    Paused,
    Recovering,
}

/// One PHD2 `GuideStep`, stored unmodified (G-h): PHD2 signs and times.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct GuideStepSample {
    pub frame: u64,
    /// PHD2 `Time`: seconds since guiding started.
    pub phd2_time_s: f64,
    /// PHD2 event `Timestamp`: Unix seconds, PHD2's clock.
    pub phd2_timestamp: Option<f64>,
    pub received_at_utc: DateTime<Utc>,
    pub mount: Option<String>,
    pub dx: f64,
    pub dy: f64,
    pub ra_raw: Option<f64>,
    pub dec_raw: Option<f64>,
    pub ra_guide: Option<f64>,
    pub dec_guide: Option<f64>,
    pub ra_duration_ms: Option<i32>,
    pub ra_direction: Option<String>,
    pub dec_duration_ms: Option<i32>,
    pub dec_direction: Option<String>,
    pub star_mass: Option<f64>,
    pub snr: Option<f64>,
    pub hfd: Option<f64>,
    pub avg_dist: Option<f64>,
    pub ra_limited: Option<bool>,
    pub dec_limited: Option<bool>,
    pub error_code: Option<i32>,
    pub phase: GuidePhase,
    /// Pixel scale current when the step arrived.
    pub pixel_scale_arcsec: Option<f64>,
}

/// Guide statistics over a set of steps (G-i). Pixels are guide-camera
/// pixels; arcsec values need a known pixel scale.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct GuideRms {
    /// Standard deviation per axis (NINA-compatible Welford).
    pub ra_px: f64,
    pub dec_px: f64,
    pub total_px: f64,
    /// RMS about zero.
    pub rms0_ra_px: f64,
    pub rms0_dec_px: f64,
    pub rms0_total_px: f64,
    pub peak_ra_px: f64,
    pub peak_dec_px: f64,
    pub ra_arcsec: Option<f64>,
    pub dec_arcsec: Option<f64>,
    pub total_arcsec: Option<f64>,
    pub rms0_total_arcsec: Option<f64>,
    pub steps_n: u32,
}

/// Settle progress (`Settling` events).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct SettleProgress {
    pub distance_px: f64,
    pub time_s: f64,
    pub settle_time_s: f64,
    pub star_locked: bool,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct LastDither {
    pub at: DateTime<Utc>,
    pub dx: Option<f64>,
    pub dy: Option<f64>,
    /// PHD2 `SettleDone.Status` (0 = success).
    pub settle_status: Option<i32>,
}

/// One algorithm parameter (R25).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlgoParam {
    pub axis: super::commands::Axis,
    pub name: String,
    pub value: f64,
}

/// Published guider snapshot (§2.13 `guiding` block).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct GuiderSnapshot {
    pub connection_id: Uuid,
    pub app_state: Observed<GuiderAppState>,
    pub phd2_version: Option<String>,
    pub profile: Option<String>,
    /// From `get_current_equipment` (G-l role check).
    pub camera_name: Option<String>,
    pub mount_name: Option<String>,
    pub pixel_scale_arcsec: Option<f64>,
    pub calibrated: Option<bool>,
    pub calibration: Option<serde_json::Value>,
    pub dec_guide_mode: Option<String>,
    pub exposure_ms: Option<u32>,
    pub settling: bool,
    pub settle: Option<SettleProgress>,
    /// Sliding 50-step window for telemetry (G-i).
    pub rms: Option<GuideRms>,
    pub last_step: Option<GuideStepSample>,
    pub last_dither: Option<LastDither>,
    pub algo_params: Vec<AlgoParam>,
    /// Discovered aggressiveness parameter names (G-j).
    pub aggressiveness_param_ra: Option<String>,
    pub aggressiveness_param_dec: Option<String>,
    /// e.g. `no_aggressiveness_param`.
    pub actuation_disabled_reason: Option<String>,
    /// EWMA of `received - Timestamp`, seconds (§2.14).
    pub clock_offset_s: Option<f64>,
    pub active_op: Option<ActiveOp>,
    pub published_at: DateTime<Utc>,
    pub published_mono_ns: u64,
}

// ---------------------------------------------------------------------------
// Weather and safety
// ---------------------------------------------------------------------------

/// The 13 ObservingConditions fields (R18). NaN, ValueNotSet and
/// NotImplemented are unknown. Each field's `device_time` is the sensor's
/// last update (`now - timesincelastupdate`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct WeatherReading {
    pub cloud_cover_pct: Observed<f64>,
    pub dew_point_c: Observed<f64>,
    pub humidity_pct: Observed<f64>,
    pub pressure_hpa: Observed<f64>,
    pub rain_rate_mm_h: Observed<f64>,
    pub sky_brightness_lux: Observed<f64>,
    pub sky_quality_mag_arcsec2: Observed<f64>,
    pub sky_temperature_c: Observed<f64>,
    pub star_fwhm_arcsec: Observed<f64>,
    pub temperature_c: Observed<f64>,
    pub wind_direction_deg: Observed<f64>,
    pub wind_gust_ms: Observed<f64>,
    pub wind_speed_ms: Observed<f64>,
    /// `timesincelastupdate` per sensor, seconds, keyed by Alpaca sensor name.
    pub sensor_ages_s: BTreeMap<String, f64>,
    /// `sensordescription`, captured once.
    pub sensor_descriptions: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct WeatherSnapshot {
    pub connection_id: Uuid,
    pub reading: WeatherReading,
    pub published_at: DateTime<Utc>,
    pub published_mono_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SafetyLevel {
    Safe,
    /// INDI Busy; unsafe when `warning_is_unsafe`.
    Warning,
    Unsafe,
}

/// One `IsSafe` read (R19).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafetyReading {
    pub level: SafetyLevel,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SafetySnapshot {
    pub connection_id: Uuid,
    pub level: Observed<SafetyLevel>,
    pub detail: Option<String>,
    pub published_at: DateTime<Utc>,
    pub published_mono_ns: u64,
}

/// The combined safety state (§3.10 hub, §2.8 fail-safe rule).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SafetyGate {
    /// No monitor configured: nothing is triggered and learning that needs
    /// weather corroboration stays frozen.
    #[default]
    None,
    /// Every monitor connected, Ready and safe.
    Safe,
    /// A confirmed Unsafe reading (or Warning when `warning_is_unsafe`).
    Unsafe,
    /// A monitor is configured but not Ready, or its reading is stale.
    Unknown,
}

impl SafetyGate {
    /// Fail-safe for gating new automatic actions (§2.8).
    pub fn allows_automatic(self) -> bool {
        matches!(self, SafetyGate::Safe | SafetyGate::None)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafetySource {
    pub connection_id: Uuid,
    pub name: String,
    pub link: LinkState,
    pub level: Option<SafetyLevel>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SafetySummary {
    pub state: SafetyGate,
    pub sources: Vec<SafetySource>,
    pub changed_at: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// Rotator, cover calibrator, switch, dome
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RotatorState {
    /// Sky position angle (Interface V3 distinguishes it from mechanical).
    pub position_deg: Observed<f64>,
    pub mechanical_deg: Observed<f64>,
    pub is_moving: Observed<bool>,
    pub reverse: Observed<bool>,
    pub target_deg: Observed<f64>,
    pub can_reverse: bool,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RotatorSnapshot {
    pub connection_id: Uuid,
    pub state: RotatorState,
    pub sync_offset_deg: Option<f64>,
    pub active_op: Option<ActiveOp>,
    pub published_at: DateTime<Utc>,
    pub published_mono_ns: u64,
}

/// ASCOM `CoverStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverState {
    NotPresent,
    Closed,
    Moving,
    Open,
    #[default]
    Unknown,
    Error,
}

impl CoverState {
    pub fn from_ascom(value: i64) -> Self {
        match value {
            0 => CoverState::NotPresent,
            1 => CoverState::Closed,
            2 => CoverState::Moving,
            3 => CoverState::Open,
            5 => CoverState::Error,
            _ => CoverState::Unknown,
        }
    }
}

/// ASCOM `CalibratorStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibratorState {
    NotPresent,
    Off,
    NotReady,
    Ready,
    #[default]
    Unknown,
    Error,
}

impl CalibratorState {
    pub fn from_ascom(value: i64) -> Self {
        match value {
            0 => CalibratorState::NotPresent,
            1 => CalibratorState::Off,
            2 => CalibratorState::NotReady,
            3 => CalibratorState::Ready,
            5 => CalibratorState::Error,
            _ => CalibratorState::Unknown,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CoverCalState {
    pub cover_state: Observed<CoverState>,
    /// Interface V2 `covermoving`.
    pub cover_moving: Observed<bool>,
    pub calibrator_state: Observed<CalibratorState>,
    pub calibrator_changing: Observed<bool>,
    pub brightness: Observed<u32>,
    pub max_brightness: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CoverCalibratorSnapshot {
    pub connection_id: Uuid,
    pub state: CoverCalState,
    pub active_op: Option<ActiveOp>,
    pub published_at: DateTime<Utc>,
    pub published_mono_ns: u64,
}

/// Static description of one switch channel (R22).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SwitchChannel {
    pub id: u32,
    pub name: String,
    pub description: Option<String>,
    pub can_write: bool,
    pub can_async: bool,
    pub min: f64,
    pub max: f64,
    pub step: f64,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SwitchChannelState {
    pub channel: SwitchChannel,
    pub value: Observed<f64>,
    /// Operator-assigned role from the profile: `dew_heater`, `power`, `fan`.
    pub role: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SwitchSnapshot {
    pub connection_id: Uuid,
    pub channels: Vec<SwitchChannelState>,
    pub active_op: Option<ActiveOp>,
    pub published_at: DateTime<Utc>,
    pub published_mono_ns: u64,
}

/// ASCOM `ShutterState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShutterState {
    Open,
    Closed,
    Opening,
    Closing,
    Error,
    #[default]
    Unknown,
}

impl ShutterState {
    pub fn from_ascom(value: i64) -> Self {
        match value {
            0 => ShutterState::Open,
            1 => ShutterState::Closed,
            2 => ShutterState::Opening,
            3 => ShutterState::Closing,
            4 => ShutterState::Error,
            _ => ShutterState::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct DomeCapabilities {
    pub can_find_home: bool,
    pub can_park: bool,
    pub can_set_altitude: bool,
    pub can_set_azimuth: bool,
    pub can_set_park: bool,
    pub can_set_shutter: bool,
    pub can_slave: bool,
    pub can_sync_azimuth: bool,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct DomeState {
    pub shutter: Observed<ShutterState>,
    pub azimuth_deg: Observed<f64>,
    pub altitude_deg: Observed<f64>,
    pub slewing: Observed<bool>,
    pub at_park: Observed<bool>,
    pub at_home: Observed<bool>,
    pub slaved: Observed<bool>,
    pub capabilities: DomeCapabilities,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct DomeSnapshot {
    pub connection_id: Uuid,
    pub state: DomeState,
    pub active_op: Option<ActiveOp>,
    pub published_at: DateTime<Utc>,
    pub published_mono_ns: u64,
}

// ---------------------------------------------------------------------------
// Camera hook
// ---------------------------------------------------------------------------

/// An exposure as seen by the camera hook (`CameraActivity`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ExposureRef {
    pub started_at: Option<DateTime<Utc>>,
    pub duration_s: Option<f64>,
    pub frame_id: Option<String>,
    pub operation_id: Option<String>,
    pub capture_id: Option<String>,
    pub stream_generation: Option<u64>,
    pub clock_quality: ClockQuality,
}

/// Camera state copied into the aggregate snapshot for gate checks
/// (A06: no focuser or wheel move while an exposure integrates).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CameraSummary {
    pub exposing: Observed<bool>,
    pub exposure: Option<ExposureRef>,
    pub name: Option<String>,
}

// ---------------------------------------------------------------------------
// Busy set, resume blocks and the aggregate snapshot
// ---------------------------------------------------------------------------

/// Critical-transaction flags (C9, H20). The compute-aware runtime never
/// switches models while any flag is set (v6 §6.7.6, §6.7.9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BusySet {
    pub exposing: bool,
    pub autofocus: bool,
    pub slewing: bool,
    pub settling: bool,
    pub parking: bool,
    pub flipping: bool,
    pub centering: bool,
    pub dithering: bool,
    pub guider_settling: bool,
    pub filter_changing: bool,
    pub cover_moving: bool,
    pub calibrator_changing: bool,
}

impl BusySet {
    pub fn any(&self) -> bool {
        self.exposing
            || self.autofocus
            || self.slewing
            || self.settling
            || self.parking
            || self.flipping
            || self.centering
            || self.dithering
            || self.guider_settling
            || self.filter_changing
            || self.cover_moving
            || self.calibrator_changing
    }
}

/// Why automatic exposure resumption is blocked (H22, v6 §8.2.1).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResumeBlockKind {
    FilterHandshakeIncomplete,
    OutcomeUnknown {
        device: DeviceKind,
        op: OpKind,
    },
    SafetyUnsafe,
    /// Only when a monitor is configured but not Ready.
    SafetyUnknown,
    GuiderLost,
    MountLost,
    LimitReached,
    Estop,
}

impl ResumeBlockKind {
    /// `safety_*` clears on a confirmed safe reading under the resume policy;
    /// every other kind is cleared by the operator (audited).
    pub fn operator_clears(&self) -> bool {
        !matches!(
            self,
            ResumeBlockKind::SafetyUnsafe | ResumeBlockKind::SafetyUnknown
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResumeBlock {
    pub id: Uuid,
    pub kind: ResumeBlockKind,
    pub raised_at: DateTime<Utc>,
    pub raised_mono_ns: u64,
    pub device_kind: Option<DeviceKind>,
    pub connection_id: Option<Uuid>,
    pub command_id: Option<CommandId>,
    pub reason: String,
}

/// The hub's aggregate (§4.1 `snapshot()`): one typed snapshot per kind,
/// plus site, busy flags, resume blocks, safety and camera state.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct EquipmentSnapshot {
    pub taken_at: DateTime<Utc>,
    pub taken_mono_ns: u64,
    #[serde(with = "arc_opt")]
    pub mount: Option<Arc<MountSnapshot>>,
    #[serde(with = "arc_opt")]
    pub focuser: Option<Arc<FocuserSnapshot>>,
    #[serde(with = "arc_opt")]
    pub filter_wheel: Option<Arc<FilterWheelSnapshot>>,
    #[serde(with = "arc_opt")]
    pub guider: Option<Arc<GuiderSnapshot>>,
    #[serde(with = "arc_opt")]
    pub rotator: Option<Arc<RotatorSnapshot>>,
    #[serde(with = "arc_opt")]
    pub cover_calibrator: Option<Arc<CoverCalibratorSnapshot>>,
    #[serde(with = "arc_opt")]
    pub switch: Option<Arc<SwitchSnapshot>>,
    #[serde(with = "arc_opt")]
    pub weather: Option<Arc<WeatherSnapshot>>,
    #[serde(with = "arc_opt")]
    pub safety_monitor: Option<Arc<SafetySnapshot>>,
    #[serde(with = "arc_opt")]
    pub dome: Option<Arc<DomeSnapshot>>,
    /// Link state per connected kind.
    pub links: BTreeMap<DeviceKind, LinkState>,
    pub camera: CameraSummary,
    pub site: Option<Site>,
    pub busy: BusySet,
    pub resume_blocks: Vec<ResumeBlock>,
    pub safety: SafetySummary,
}

impl EquipmentSnapshot {
    /// An empty snapshot: nothing connected, safety `none`.
    pub fn empty(taken_at: DateTime<Utc>, taken_mono_ns: u64) -> Self {
        Self {
            taken_at,
            taken_mono_ns,
            ..Self::default()
        }
    }

    pub fn link(&self, kind: DeviceKind) -> Option<LinkState> {
        self.links.get(&kind).copied()
    }
}

// ---------------------------------------------------------------------------
// Windows and quiescence (§4.1, D7)
// ---------------------------------------------------------------------------

/// Snapshots nearest the start, middle and end of a window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindowSamples<T> {
    pub start: Option<T>,
    pub mid: Option<T>,
    pub end: Option<T>,
    /// No sample newer than the window end arrived in time (D7).
    pub end_stale: bool,
    pub samples_n: u32,
}

impl<T> Default for WindowSamples<T> {
    fn default() -> Self {
        Self {
            start: None,
            mid: None,
            end: None,
            end_stale: false,
            samples_n: 0,
        }
    }
}

/// Mount motion state over a sub-interval of a window.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MotionInterval {
    pub from: DateTime<Utc>,
    pub to: DateTime<Utc>,
    pub tracking: Option<bool>,
    pub slewing: Option<bool>,
    pub pulse_guiding: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct MovedFlags {
    pub focuser: bool,
    pub wheel: bool,
    pub mount: bool,
    pub rotator: bool,
}

/// Guide statistics for a window (G-i; dither and settle steps excluded by
/// default).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct GuideWindowStats {
    pub state: Option<GuiderAppState>,
    pub rms: Option<GuideRms>,
    pub dither_in_window: bool,
    pub settle_in_window: bool,
    pub pulses_n: u32,
}

/// Device state over an exposure (§4.1 `window()`), assembled after the
/// exposure from ring buffers and the journal (D7).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct DeviceWindow {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub finalised: bool,
    pub clock_quality: ClockQuality,
    pub mount: WindowSamples<MountSnapshot>,
    /// Native RA/Dec at mid-exposure, linearly interpolated with RA wrap.
    pub mount_mid: Option<EqCoord>,
    pub focuser: WindowSamples<FocuserSnapshot>,
    pub filter_wheel: WindowSamples<FilterWheelSnapshot>,
    pub rotator: WindowSamples<RotatorSnapshot>,
    pub cover_calibrator: WindowSamples<CoverCalibratorSnapshot>,
    pub switch: WindowSamples<SwitchSnapshot>,
    pub dome: WindowSamples<DomeSnapshot>,
    /// Weather reading nearest the middle of the window.
    pub weather: Option<WeatherReading>,
    pub safety: Option<SafetySummary>,
    pub guide: GuideWindowStats,
    pub motion: Vec<MotionInterval>,
    pub moved: MovedFlags,
    pub events: Vec<EquipmentEvent>,
    pub resume_block_active: bool,
    pub quiescence: Quiescence,
    /// Kinds whose end state is `Stale` (D7).
    pub stale_kinds: Vec<DeviceKind>,
}

/// Result of `quiescent(kinds, start, end)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Quiescence {
    Quiescent {
        settled_since: DateTime<Utc>,
    },
    Moved {
        kind: DeviceKind,
        event_seq: u64,
    },
    /// No data, stale data, or a lost device.
    Unknown {
        reason: String,
    },
}

impl Default for Quiescence {
    fn default() -> Self {
        Quiescence::Unknown {
            reason: "no_data".into(),
        }
    }
}

impl Quiescence {
    pub fn is_quiescent(&self) -> bool {
        matches!(self, Quiescence::Quiescent { .. })
    }
}

// ---------------------------------------------------------------------------
// Model contexts (H6, H14)
// ---------------------------------------------------------------------------

/// Where an ambient temperature came from (H6 priority order).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemperatureSourceKind {
    Weather,
    FocuserProbe,
    /// Never used for FDF learning (v6 §8.1, §10.7).
    CameraSensor,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemperatureReading {
    pub celsius: f64,
    pub source: TemperatureSourceKind,
    pub role: TemperatureRole,
    pub identity_key: Option<String>,
    pub connection_id: Option<Uuid>,
    pub camera_sensor_fallback: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Twilight {
    Day,
    Civil,
    Nautical,
    Astronomical,
    Night,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorroborationLevel {
    /// No sky sensor: COC infers nothing from corroboration.
    #[default]
    None,
    Partial,
    Full,
}

/// Independent sky corroboration with per-sensor ages (COC, R18-R19).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CocCorroboration {
    pub level: CorroborationLevel,
    pub cloud_cover_pct: Observed<f64>,
    pub sky_temperature_c: Observed<f64>,
    pub rain_rate_mm_h: Observed<f64>,
    pub humidity_pct: Observed<f64>,
    pub dew_point_c: Observed<f64>,
    pub sky_quality_mag_arcsec2: Observed<f64>,
    pub is_safe: Observed<bool>,
}

/// Context for the cloud/obstruction classifier (H14). Wiring it into the
/// user-owned `coc_shadow.rs` is left to the user.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CocContext {
    pub at: DateTime<Utc>,
    pub site: Option<Site>,
    pub mount_alt_deg: Observed<f64>,
    pub mount_az_deg: Observed<f64>,
    pub pointing_j2000: Option<EqCoord>,
    pub tracking: Observed<bool>,
    pub slewing: Observed<bool>,
    pub sun_alt_deg: Option<f64>,
    pub moon_alt_deg: Option<f64>,
    pub moon_illumination: Option<f64>,
    pub twilight: Option<Twilight>,
    pub corroboration: CocCorroboration,
    pub cover_state: Observed<CoverState>,
    pub dome_shutter: Observed<ShutterState>,
    /// Switch channels with role `dew_heater`, by name.
    pub dew_heaters: BTreeMap<String, Observed<f64>>,
    /// A fixed all-sky camera's declared view direction (rig profile).
    pub view_direction: Option<AltAz>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::types::Source;
    use serde_json::json;

    fn round_trip<T: Serialize + for<'de> Deserialize<'de> + PartialEq + std::fmt::Debug>(
        value: &T,
    ) {
        let text = serde_json::to_string(value).unwrap();
        let back: T = serde_json::from_str(&text).unwrap();
        assert_eq!(&back, value, "{text}");
    }

    fn mount_snapshot() -> MountSnapshot {
        let at = Utc::now();
        MountSnapshot {
            connection_id: Uuid::new_v4(),
            state: MountState {
                ra_h: Observed::fresh(5.5, Source::DeviceState, at, 10),
                dec_deg: Observed::fresh(-5.25, Source::DeviceState, at, 10),
                slewing: Observed::fresh(false, Source::DeviceState, at, 10),
                side_of_pier: Observed::fresh(PierSide::West, Source::DeviceState, at, 10),
                tracking_rate: Observed::fresh(DriveRate::Sidereal, Source::Property, at, 10),
                ..MountState::default()
            },
            capabilities: MountCapabilities {
                can_park: true,
                alignment_mode: Some(AlignmentMode::Polar),
                equatorial_system: Some(1),
                native_frame: EqFrame::JNowApparent,
                tracking_rates: vec![DriveRate::Sidereal, DriveRate::King],
                axis_rates: vec![AxisRates {
                    axis: 0,
                    ranges: vec![RateRange {
                        min_deg_s: 0.0,
                        max_deg_s: 4.0,
                    }],
                }],
                ..MountCapabilities::default()
            },
            j2000: Some(EqCoord::new(5.49, -5.3, EqFrame::J2000)),
            ha_h: Some(1.25),
            published_at: at,
            published_mono_ns: 11,
            ..MountSnapshot::default()
        }
    }

    #[test]
    fn device_snapshots_round_trip() {
        let mount = mount_snapshot();
        round_trip(&mount);
        let json = serde_json::to_value(&mount).unwrap();
        assert_eq!(json["state"]["side_of_pier"]["value"], json!("west"));
        assert_eq!(json["capabilities"]["native_frame"], json!("jnow_apparent"));
        assert_eq!(json["capabilities"]["alignment_mode"], json!("polar"));

        round_trip(&FocuserSnapshot {
            position_source: PositionSource::EmulatedRestored,
            backlash_model: BacklashModel::Overshoot {
                in_steps: 0,
                out_steps: 40,
            },
            backlash_owner: BacklashOwner::Firmware,
            ..FocuserSnapshot::default()
        });
        round_trip(&FilterWheelSnapshot {
            position: Observed::not_set(Source::Property),
            moving: true,
            slots: vec![FilterSlot {
                index: 0,
                name: "L".into(),
                name_source: SlotValueSource::Device,
                offset: Some(0),
                offset_source: SlotValueSource::Profile,
            }],
            ..FilterWheelSnapshot::default()
        });
        round_trip(&GuiderSnapshot {
            app_state: Observed::fresh(GuiderAppState::LostLock, Source::Phd2Event, Utc::now(), 4),
            rms: Some(GuideRms {
                ra_px: 0.4,
                steps_n: 50,
                ..GuideRms::default()
            }),
            algo_params: vec![AlgoParam {
                axis: super::super::commands::Axis::Ra,
                name: "aggression".into(),
                value: 0.7,
            }],
            ..GuiderSnapshot::default()
        });
        round_trip(&WeatherSnapshot::default());
        round_trip(&SafetySnapshot {
            level: Observed::fresh(SafetyLevel::Warning, Source::IndiPush, Utc::now(), 5),
            ..SafetySnapshot::default()
        });
        round_trip(&RotatorSnapshot::default());
        round_trip(&CoverCalibratorSnapshot::default());
        round_trip(&SwitchSnapshot::default());
        round_trip(&DomeSnapshot::default());
    }

    #[test]
    fn aggregate_snapshot_round_trips_with_shared_arcs() {
        let mut snap = EquipmentSnapshot::empty(Utc::now(), 99);
        snap.mount = Some(Arc::new(mount_snapshot()));
        snap.links.insert(DeviceKind::Mount, LinkState::Degraded);
        snap.busy.slewing = true;
        snap.resume_blocks.push(ResumeBlock {
            id: Uuid::new_v4(),
            kind: ResumeBlockKind::OutcomeUnknown {
                device: DeviceKind::FilterWheel,
                op: OpKind::FilterMove,
            },
            raised_at: Utc::now(),
            raised_mono_ns: 5,
            device_kind: Some(DeviceKind::FilterWheel),
            connection_id: None,
            command_id: Some(CommandId::new()),
            reason: "wheel timeout".into(),
        });
        round_trip(&snap);
        let json = serde_json::to_value(&snap).unwrap();
        assert_eq!(json["focuser"], json!(null));
        assert_eq!(json["links"]["mount"], json!("degraded"));
        assert_eq!(json["safety"]["state"], json!("none"));
        assert_eq!(
            json["resume_blocks"][0]["kind"],
            json!({"kind": "outcome_unknown", "device": "filter_wheel", "op": "filter_move"})
        );
        assert!(snap.busy.any());
        assert!(!BusySet::default().any());
    }

    #[test]
    fn windows_and_contexts_round_trip() {
        let window = DeviceWindow {
            mount: WindowSamples {
                start: Some(mount_snapshot()),
                end_stale: true,
                samples_n: 1,
                ..WindowSamples::default()
            },
            quiescence: Quiescence::Moved {
                kind: DeviceKind::Focuser,
                event_seq: 42,
            },
            stale_kinds: vec![DeviceKind::Mount],
            ..DeviceWindow::default()
        };
        round_trip(&window);
        assert_eq!(
            serde_json::to_value(&window.quiescence).unwrap(),
            json!({"status": "moved", "kind": "focuser", "event_seq": 42})
        );
        round_trip(&CocContext::default());
        round_trip(&TemperatureReading {
            celsius: 3.5,
            source: TemperatureSourceKind::FocuserProbe,
            role: TemperatureRole::Tube,
            identity_key: Some("alpaca:esp32-focuser".into()),
            connection_id: None,
            camera_sensor_fallback: false,
        });
    }

    #[test]
    fn safety_gate_is_fail_safe_for_automatic_actions() {
        assert!(SafetyGate::Safe.allows_automatic());
        assert!(SafetyGate::None.allows_automatic());
        assert!(!SafetyGate::Unknown.allows_automatic());
        assert!(!SafetyGate::Unsafe.allows_automatic());
        assert!(!ResumeBlockKind::SafetyUnsafe.operator_clears());
        assert!(ResumeBlockKind::Estop.operator_clears());
        assert!(GuiderAppState::Calibrating.owns_mount());
        assert!(!GuiderAppState::Paused.owns_mount());
        assert_eq!(
            GuiderAppState::from_phd2("LostLock"),
            GuiderAppState::LostLock
        );
        assert!(!PositionSource::EmulatedOriginUnknown.is_reliable());
        assert!(PositionSource::EmulatedRestored.is_reliable());
    }
}

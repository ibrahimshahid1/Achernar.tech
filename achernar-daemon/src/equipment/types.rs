//! Shared value types of the equipment layer (spec §2.4). Frozen by WP0.
//!
//! Every reading is an [`Observed`] value: a value or an explicit unknown, plus
//! when, how and from where it was observed (D5). Nothing in this module ever
//! synthesises a value: an unknown stays `None`, never `0` or "safe".
//!
//! Owner decision O4 removed the native LX200 and MoonLite drivers, so
//! [`DriverId`], [`Endpoint`] and [`Source`] carry no serial variants. Those
//! devices are reached through indiserver (`indi`) or ASCOM Remote (`alpaca`).

use std::fmt;

use chrono::{DateTime, Utc};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use uuid::Uuid;

pub use crate::hardware::health::DeviceKind;

/// The serde name of a [`DeviceKind`], e.g. `filter_wheel`.
pub fn kind_name(kind: DeviceKind) -> &'static str {
    match kind {
        DeviceKind::Camera => "camera",
        DeviceKind::Mount => "mount",
        DeviceKind::Focuser => "focuser",
        DeviceKind::FilterWheel => "filter_wheel",
        DeviceKind::Guider => "guider",
        DeviceKind::Rotator => "rotator",
        DeviceKind::CoverCalibrator => "cover_calibrator",
        DeviceKind::Switch => "switch",
        DeviceKind::Weather => "weather",
        DeviceKind::SafetyMonitor => "safety_monitor",
        DeviceKind::Dome => "dome",
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

pub type DeviceResult<T> = Result<T, DeviceError>;

/// Failure classes of a device call (v6 §4.1, spec §2.8 and §3.1 C-3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorClass {
    NotConnected,
    Disconnected,
    Unsupported,
    InvalidValue,
    InvalidWhileParked,
    InvalidOperation,
    Timeout,
    OutcomeUnknown,
    Transport,
    Driver,
    Cancelled,
    SafetyRejected,
    OwnershipConflict,
}

impl ErrorClass {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorClass::NotConnected => "not_connected",
            ErrorClass::Disconnected => "disconnected",
            ErrorClass::Unsupported => "unsupported",
            ErrorClass::InvalidValue => "invalid_value",
            ErrorClass::InvalidWhileParked => "invalid_while_parked",
            ErrorClass::InvalidOperation => "invalid_operation",
            ErrorClass::Timeout => "timeout",
            ErrorClass::OutcomeUnknown => "outcome_unknown",
            ErrorClass::Transport => "transport",
            ErrorClass::Driver => "driver",
            ErrorClass::Cancelled => "cancelled",
            ErrorClass::SafetyRejected => "safety_rejected",
            ErrorClass::OwnershipConflict => "ownership_conflict",
        }
    }

    /// HTTP status for the API error body (spec §2.12). Direct and relay
    /// routes use the same mapping.
    pub fn http_status(self) -> u16 {
        match self {
            ErrorClass::NotConnected | ErrorClass::Disconnected => 412,
            ErrorClass::Unsupported => 501,
            ErrorClass::InvalidValue => 400,
            ErrorClass::InvalidWhileParked
            | ErrorClass::InvalidOperation
            | ErrorClass::OwnershipConflict
            | ErrorClass::Cancelled => 409,
            ErrorClass::SafetyRejected => 422,
            ErrorClass::Timeout | ErrorClass::OutcomeUnknown => 504,
            ErrorClass::Transport | ErrorClass::Driver => 502,
        }
    }

    /// The registry `failure_class` this error implies, if it is a device
    /// failure at all. Refusals (cancelled, safety, ownership) are not.
    pub fn failure_class(self) -> Option<FailureClass> {
        match self {
            ErrorClass::NotConnected | ErrorClass::Disconnected | ErrorClass::Transport => {
                Some(FailureClass::Disconnected)
            }
            ErrorClass::Unsupported => Some(FailureClass::Unsupported),
            ErrorClass::Timeout => Some(FailureClass::Timeout),
            ErrorClass::OutcomeUnknown => Some(FailureClass::OutcomeUnknown),
            ErrorClass::Driver => Some(FailureClass::Driver),
            ErrorClass::InvalidValue
            | ErrorClass::InvalidWhileParked
            | ErrorClass::InvalidOperation => Some(FailureClass::Invalid),
            ErrorClass::Cancelled | ErrorClass::SafetyRejected | ErrorClass::OwnershipConflict => {
                None
            }
        }
    }
}

impl fmt::Display for ErrorClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// ASCOM error numbers used by the C-3 mapping.
pub mod ascom_error {
    pub const NOT_IMPLEMENTED: i32 = 0x400;
    pub const INVALID_VALUE: i32 = 0x401;
    pub const VALUE_NOT_SET: i32 = 0x402;
    pub const NOT_CONNECTED: i32 = 0x407;
    pub const INVALID_WHILE_PARKED: i32 = 0x408;
    pub const INVALID_WHILE_SLAVED: i32 = 0x409;
    pub const INVALID_OPERATION: i32 = 0x40B;
    pub const ACTION_NOT_IMPLEMENTED: i32 = 0x40C;
    pub const OPERATION_CANCELLED: i32 = 0x40E;
    pub const DRIVER_BASE: i32 = 0x500;
    pub const DRIVER_MAX: i32 = 0xFFF;
}

/// A failed device call.
///
/// `code` conventions, so one field can carry every protocol's number:
/// - `100..=599`: an HTTP status from the Alpaca transport (non-200 reply
///   with a plain-text body). Built only by [`DeviceError::from_http_status`].
/// - `>= 0x400`: an ASCOM `ErrorNumber`. Built by
///   [`DeviceError::from_alpaca_error`].
/// - anything else: a protocol-specific code (PHD2 JSON-RPC, INDI), which
///   must stay outside `100..=599`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceError {
    pub class: ErrorClass,
    pub code: Option<i32>,
    pub message: String,
}

impl DeviceError {
    pub fn new(class: ErrorClass, message: impl Into<String>) -> Self {
        Self {
            class,
            code: None,
            message: message.into(),
        }
    }

    pub fn with_code(class: ErrorClass, code: i32, message: impl Into<String>) -> Self {
        Self {
            class,
            code: Some(code),
            message: message.into(),
        }
    }

    pub fn not_connected(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::NotConnected, message)
    }

    pub fn disconnected(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::Disconnected, message)
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::Unsupported, message)
    }

    pub fn invalid_value(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::InvalidValue, message)
    }

    pub fn invalid_operation(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::InvalidOperation, message)
    }

    pub fn timeout(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::Timeout, message)
    }

    pub fn outcome_unknown(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::OutcomeUnknown, message)
    }

    pub fn transport(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::Transport, message)
    }

    pub fn driver(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::Driver, message)
    }

    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::Cancelled, message)
    }

    /// Map an Alpaca `ErrorNumber` (non-zero) per spec §3.1 C-3.
    ///
    /// `0x402` (ValueNotSet) is **not an error** to the layer: it means the
    /// reading is unknown. It is returned as class `Driver` with code `0x402`
    /// so the transport stays uniform; every reader must test
    /// [`DeviceError::is_not_set`] first and publish `ObsState::NotSet`.
    pub fn from_alpaca_error(error_number: i32, message: &str) -> Self {
        use ascom_error::*;
        let class = match error_number {
            NOT_IMPLEMENTED | ACTION_NOT_IMPLEMENTED => ErrorClass::Unsupported,
            INVALID_VALUE => ErrorClass::InvalidValue,
            NOT_CONNECTED => ErrorClass::NotConnected,
            INVALID_WHILE_PARKED => ErrorClass::InvalidWhileParked,
            INVALID_WHILE_SLAVED | INVALID_OPERATION => ErrorClass::InvalidOperation,
            OPERATION_CANCELLED => ErrorClass::Cancelled,
            _ => ErrorClass::Driver,
        };
        Self::with_code(class, error_number, message)
    }

    /// Map a non-200 Alpaca HTTP reply with its plain-text body (C-3 step 1).
    ///
    /// 400 is "bad request or unknown member": it maps to `InvalidValue` for
    /// commands, and capability probes test [`DeviceError::is_unknown_member`]
    /// to memoise the member as unsupported instead (§2.6 step 6).
    pub fn from_http_status(status: u16, body: &str) -> Self {
        let class = match status {
            400 => ErrorClass::InvalidValue,
            404 => ErrorClass::Unsupported,
            408 | 504 => ErrorClass::Timeout,
            500..=599 => ErrorClass::Driver,
            _ => ErrorClass::Transport,
        };
        let body = body.trim();
        let message = if body.is_empty() {
            format!("HTTP {status}")
        } else {
            format!("HTTP {status}: {body}")
        };
        Self::with_code(class, i32::from(status), message)
    }

    /// The HTTP status, when this error came from a non-200 Alpaca reply.
    pub fn http_status(&self) -> Option<u16> {
        self.code
            .filter(|code| (100..=599).contains(code))
            .and_then(|code| u16::try_from(code).ok())
    }

    /// The ASCOM `ErrorNumber`, when this error came from an Alpaca reply.
    pub fn alpaca_error_number(&self) -> Option<i32> {
        self.code
            .filter(|code| *code >= ascom_error::NOT_IMPLEMENTED)
    }

    /// ASCOM ValueNotSet: the value is unknown, not failed (C-2).
    pub fn is_not_set(&self) -> bool {
        self.alpaca_error_number() == Some(ascom_error::VALUE_NOT_SET)
    }

    /// True when a capability probe should memoise the member as
    /// unsupported for the session: ASCOM NotImplemented, or HTTP 400/404 on
    /// an older server that does not know the member (§2.6 step 6).
    pub fn is_unknown_member(&self) -> bool {
        matches!(self.http_status(), Some(400 | 404))
            || matches!(
                self.alpaca_error_number(),
                Some(ascom_error::NOT_IMPLEMENTED | ascom_error::ACTION_NOT_IMPLEMENTED)
            )
            || (self.class == ErrorClass::Unsupported && self.code.is_none())
    }

    /// API error code, e.g. `mount_outcome_unknown` (spec §2.12).
    pub fn api_code(&self, kind: DeviceKind) -> String {
        format!("{}_{}", kind_name(kind), self.class.as_str())
    }
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.code {
            Some(code) => write!(f, "{} ({code:#x}): {}", self.class, self.message),
            None => write!(f, "{}: {}", self.class, self.message),
        }
    }
}

impl std::error::Error for DeviceError {}

// ---------------------------------------------------------------------------
// Observations
// ---------------------------------------------------------------------------

/// State of one observed value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObsState {
    /// Read successfully on the latest poll.
    Fresh,
    /// The latest read failed or is overdue; `value` is the last good value.
    Stale,
    /// The device does not implement this member (memoised for the session).
    Unsupported,
    /// Nothing observed yet, or the device reports ValueNotSet.
    #[default]
    NotSet,
    /// The read failed and no earlier value exists.
    Error,
}

/// Where a reading came from. Serial query sources were removed by O4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    /// One Alpaca `devicestate` GET.
    DeviceState,
    /// A per-member read (Alpaca property, PHD2 RPC reply).
    Property,
    /// An INDI property pushed by indiserver.
    IndiPush,
    /// A PHD2 event-server notification.
    Phd2Event,
    /// Computed by Achernar from other readings (astro, interpolation).
    #[default]
    Derived,
    /// Taken from the identity profile or a configured fallback.
    Profile,
    /// Declared by the operator.
    Operator,
}

/// A value or an explicit unknown, with timing and provenance (D5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observed<T> {
    /// `None` = unknown. Never a synthesised 0 or "safe".
    pub value: Option<T>,
    pub state: ObsState,
    /// Midpoint of the request, host clock.
    pub observed_at: Option<DateTime<Utc>>,
    /// Nanoseconds since daemon start ([`crate::equipment::traits::Clock`]).
    pub mono_ns: Option<u64>,
    /// DeviceState TimeStamp, INDI timestamp, PHD2 Timestamp, or the sensor
    /// update time derived from `timesincelastupdate`.
    pub device_time: Option<DateTime<Utc>>,
    pub latency_ms: Option<u32>,
    pub source: Source,
}

impl<T> Default for Observed<T> {
    fn default() -> Self {
        Self::unknown(ObsState::NotSet, Source::Derived)
    }
}

impl<T> Observed<T> {
    /// A value read successfully now.
    pub fn fresh(value: T, source: Source, observed_at: DateTime<Utc>, mono_ns: u64) -> Self {
        Self {
            value: Some(value),
            state: ObsState::Fresh,
            observed_at: Some(observed_at),
            mono_ns: Some(mono_ns),
            device_time: None,
            latency_ms: None,
            source,
        }
    }

    /// A value read successfully, timestamped from `clock`.
    pub fn fresh_at(value: T, source: Source, clock: &dyn super::traits::Clock) -> Self {
        Self::fresh(value, source, clock.now_utc(), clock.now_mono_ns())
    }

    /// An unknown value in the given state.
    pub fn unknown(state: ObsState, source: Source) -> Self {
        Self {
            value: None,
            state,
            observed_at: None,
            mono_ns: None,
            device_time: None,
            latency_ms: None,
            source,
        }
    }

    pub fn unsupported(source: Source) -> Self {
        Self::unknown(ObsState::Unsupported, source)
    }

    pub fn not_set(source: Source) -> Self {
        Self::unknown(ObsState::NotSet, source)
    }

    pub fn error(source: Source) -> Self {
        Self::unknown(ObsState::Error, source)
    }

    pub fn with_latency_ms(mut self, latency_ms: u32) -> Self {
        self.latency_ms = Some(latency_ms);
        self
    }

    pub fn with_device_time(mut self, device_time: DateTime<Utc>) -> Self {
        self.device_time = Some(device_time);
        self
    }

    /// The read failed: keep the last value but never present it as fresh
    /// (C-2). A value that was never known becomes `Error`.
    pub fn into_stale(mut self) -> Self {
        self.state = match (self.state, self.value.is_some()) {
            (ObsState::Unsupported, _) => ObsState::Unsupported,
            (_, true) => ObsState::Stale,
            (_, false) => ObsState::Error,
        };
        self
    }

    pub fn is_fresh(&self) -> bool {
        self.state == ObsState::Fresh && self.value.is_some()
    }

    /// The value if one is known (fresh or stale).
    pub fn known(&self) -> Option<&T> {
        self.value.as_ref()
    }

    /// The value only if it is fresh.
    pub fn fresh_value(&self) -> Option<&T> {
        if self.state == ObsState::Fresh {
            self.value.as_ref()
        } else {
            None
        }
    }

    /// Age of the observation against a monotonic `now`.
    pub fn age_ms(&self, now_mono_ns: u64) -> Option<u64> {
        self.mono_ns
            .map(|at| now_mono_ns.saturating_sub(at) / 1_000_000)
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Observed<U> {
        Observed {
            value: self.value.map(f),
            state: self.state,
            observed_at: self.observed_at,
            mono_ns: self.mono_ns,
            device_time: self.device_time,
            latency_ms: self.latency_ms,
            source: self.source,
        }
    }
}

// ---------------------------------------------------------------------------
// Drivers, endpoints and identity
// ---------------------------------------------------------------------------

/// Driver implementations. `OlsAlpaca` lives until the end of Tier 1 (O2);
/// `OlsIndi` until the native INDI client is accepted (O3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriverId {
    Alpaca,
    Indi,
    Phd2,
    #[serde(rename = "direct")]
    DirectGuider,
    Sim,
    OlsAlpaca,
    OlsIndi,
}

impl DriverId {
    pub fn as_str(self) -> &'static str {
        match self {
            DriverId::Alpaca => "alpaca",
            DriverId::Indi => "indi",
            DriverId::Phd2 => "phd2",
            DriverId::DirectGuider => "direct",
            DriverId::Sim => "sim",
            DriverId::OlsAlpaca => "ols_alpaca",
            DriverId::OlsIndi => "ols_indi",
        }
    }

    /// Parse a `/connect` driver id, including the `simulator` alias.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "alpaca" => Some(DriverId::Alpaca),
            "indi" => Some(DriverId::Indi),
            "phd2" => Some(DriverId::Phd2),
            "direct" => Some(DriverId::DirectGuider),
            "sim" | "simulator" => Some(DriverId::Sim),
            "ols_alpaca" => Some(DriverId::OlsAlpaca),
            "ols_indi" => Some(DriverId::OlsIndi),
            _ => None,
        }
    }
}

impl fmt::Display for DriverId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How a device is reached.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "transport", rename_all = "snake_case")]
pub enum Endpoint {
    /// `base` is `http(s)://host:port`; `device_type` is the lower-case
    /// Alpaca type (`telescope`, `focuser`, ...).
    AlpacaHttp {
        base: String,
        device_type: String,
        number: u32,
    },
    Indi {
        host: String,
        port: u16,
        device: String,
    },
    /// PHD2 event server; instance n listens on `4400 + n - 1`.
    Phd2 { host: String, port: u16 },
    /// Simulators and the direct guider.
    InProcess { name: String },
}

impl Endpoint {
    pub const PHD2_BASE_PORT: u16 = 4400;

    /// The PHD2 endpoint for a 1-based instance number.
    pub fn phd2_instance(host: impl Into<String>, instance: u16) -> Self {
        Endpoint::Phd2 {
            host: host.into(),
            port: Self::PHD2_BASE_PORT + instance.max(1) - 1,
        }
    }

    /// Human-readable form for logs and the `device_connections.endpoint`
    /// column.
    pub fn describe(&self) -> String {
        match self {
            Endpoint::AlpacaHttp {
                base,
                device_type,
                number,
            } => format!(
                "{}/api/v1/{device_type}/{number}",
                base.trim_end_matches('/')
            ),
            Endpoint::Indi { host, port, device } => format!("indi://{host}:{port}/{device}"),
            Endpoint::Phd2 { host, port } => format!("phd2://{host}:{port}"),
            Endpoint::InProcess { name } => format!("inprocess:{name}"),
        }
    }
}

/// Qualification state of a driver or device (D21).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Qualification {
    Planned,
    /// What every work package reports for what it built (§5.1 rule 10).
    #[default]
    ImplementedUnqualified,
    /// Only the rig acceptance procedure (WP1B-6) records this.
    Qualified {
        record_id: String,
    },
}

impl Qualification {
    pub fn as_str(&self) -> &'static str {
        match self {
            Qualification::Planned => "planned",
            Qualification::ImplementedUnqualified => "implemented_unqualified",
            Qualification::Qualified { .. } => "qualified",
        }
    }
}

/// Who a connected device is (§2.6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceIdentity {
    pub kind: DeviceKind,
    pub driver: DriverId,
    pub endpoint: Endpoint,
    /// Stable profile and persistence key (§2.6); see [`identity_key`].
    pub identity_key: String,
    pub connection_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub driver_info: Option<String>,
    pub driver_version: Option<String>,
    pub interface_version: Option<i32>,
    pub unique_id: Option<String>,
    /// Alpaca `/setup/v1/{type}/{n}/setup`.
    pub setup_url: Option<String>,
    pub qualification: Qualification,
}

impl DeviceIdentity {
    /// A minimal identity: a new connection id, no driver claims yet.
    pub fn new(
        kind: DeviceKind,
        driver: DriverId,
        endpoint: Endpoint,
        identity_key: impl Into<String>,
        name: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            driver,
            endpoint,
            identity_key: identity_key.into(),
            connection_id: Uuid::new_v4(),
            name: name.into(),
            description: None,
            driver_info: None,
            driver_version: None,
            interface_version: None,
            unique_id: None,
            setup_url: None,
            qualification: Qualification::ImplementedUnqualified,
        }
    }
}

/// Identity-key builders (§2.6 table). Every driver uses these so profile
/// rows written by one package are found by another.
pub mod identity_key {
    use super::{kind_name, DeviceKind};

    /// `alpaca:{UniqueID}`, or `alpaca:{host}:{port}/{type}/{n}` without one.
    pub fn alpaca(
        unique_id: Option<&str>,
        host: &str,
        port: u16,
        device_type: &str,
        number: u32,
    ) -> String {
        match unique_id.map(str::trim).filter(|id| !id.is_empty()) {
            Some(id) => format!("alpaca:{id}"),
            None => format!(
                "alpaca:{}:{port}/{}/{number}",
                host.to_ascii_lowercase(),
                device_type.to_ascii_lowercase()
            ),
        }
    }

    /// `indi:{host}:{port}/{device}`.
    pub fn indi(host: &str, port: u16, device: &str) -> String {
        format!("indi:{}:{port}/{device}", host.to_ascii_lowercase())
    }

    /// `phd2:{host}:{port}`, plus `/{profile}` once the PHD2 profile is known.
    pub fn phd2(host: &str, port: u16, profile: Option<&str>) -> String {
        let base = format!("phd2:{}:{port}", host.to_ascii_lowercase());
        match profile.map(str::trim).filter(|p| !p.is_empty()) {
            Some(profile) => format!("{base}/{profile}"),
            None => base,
        }
    }

    /// `sim:{kind}`.
    pub fn sim(kind: DeviceKind) -> String {
        format!("sim:{}", kind_name(kind))
    }
}

// ---------------------------------------------------------------------------
// Coordinates and mount enums
// ---------------------------------------------------------------------------

/// Equatorial reference frame (D10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum EqFrame {
    #[serde(rename = "j2000")]
    J2000,
    #[serde(rename = "jnow_mean")]
    JNowMean,
    #[serde(rename = "jnow_apparent")]
    JNowApparent,
    #[default]
    #[serde(rename = "unknown")]
    Unknown,
}

impl EqFrame {
    pub fn as_str(self) -> &'static str {
        match self {
            EqFrame::J2000 => "j2000",
            EqFrame::JNowMean => "jnow_mean",
            EqFrame::JNowApparent => "jnow_apparent",
            EqFrame::Unknown => "unknown",
        }
    }

    /// Map ASCOM `EquatorialSystem` per §3.2 M-a. Returns the frame and, for
    /// `Unknown`, the published reason.
    pub fn from_equatorial_system(
        code: i32,
        flavour: JNowFlavour,
    ) -> (EqFrame, Option<&'static str>) {
        match code {
            1 => (flavour.frame(), None),
            2 => (EqFrame::J2000, None),
            3 | 4 => (EqFrame::Unknown, Some("unsupported_epoch")),
            0 => (EqFrame::Unknown, Some("equatorial_system_other")),
            _ => (EqFrame::Unknown, Some("equatorial_system_invalid")),
        }
    }
}

/// How "Topocentric"/EOD coordinates are interpreted; recorded with every
/// pointing sample (M-4 gap 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JNowFlavour {
    Mean,
    #[default]
    Apparent,
}

impl JNowFlavour {
    pub fn frame(self) -> EqFrame {
        match self {
            JNowFlavour::Mean => EqFrame::JNowMean,
            JNowFlavour::Apparent => EqFrame::JNowApparent,
        }
    }
}

/// RA in hours, Dec in degrees, with the frame they are expressed in.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct EqCoord {
    pub ra_h: f64,
    pub dec_deg: f64,
    pub frame: EqFrame,
}

impl EqCoord {
    pub fn new(ra_h: f64, dec_deg: f64, frame: EqFrame) -> Self {
        Self {
            ra_h,
            dec_deg,
            frame,
        }
    }

    /// RA wrapped into `[0, 24)` hours.
    pub fn wrap_ra_h(ra_h: f64) -> f64 {
        let wrapped = ra_h.rem_euclid(24.0);
        if wrapped >= 24.0 {
            0.0
        } else {
            wrapped
        }
    }

    pub fn normalized(self) -> Self {
        Self {
            ra_h: Self::wrap_ra_h(self.ra_h),
            ..self
        }
    }

    /// Great-circle separation in arcminutes, or `None` when the two
    /// coordinates are in different frames (convert first; D10).
    pub fn separation_arcmin(&self, other: &EqCoord) -> Option<f64> {
        if self.frame != other.frame {
            return None;
        }
        let (ra1, ra2) = (
            (self.ra_h * 15.0).to_radians(),
            (other.ra_h * 15.0).to_radians(),
        );
        let (d1, d2) = (self.dec_deg.to_radians(), other.dec_deg.to_radians());
        let a = ((d2 - d1) / 2.0).sin().powi(2)
            + d1.cos() * d2.cos() * ((ra2 - ra1) / 2.0).sin().powi(2);
        Some(2.0 * a.sqrt().min(1.0).asin().to_degrees() * 60.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct AltAz {
    pub alt_deg: f64,
    pub az_deg: f64,
}

/// ASCOM pointing state. JSON is `"east"`, `"west"` or `null` (the frontend
/// type, `F/types/achernar.ts:37`). Inside an `Observed<PierSide>` prefer a
/// `None` value for unknown: `Some(Unknown)` also serialises as `null`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PierSide {
    East,
    West,
    #[default]
    Unknown,
}

impl PierSide {
    /// ASCOM `PointingState`: 0 = pierEast, 1 = pierWest, -1 = pierUnknown.
    pub fn from_ascom(value: i64) -> Self {
        match value {
            0 => PierSide::East,
            1 => PierSide::West,
            _ => PierSide::Unknown,
        }
    }

    pub fn to_ascom(self) -> i32 {
        match self {
            PierSide::East => 0,
            PierSide::West => 1,
            PierSide::Unknown => -1,
        }
    }

    pub fn as_str(self) -> Option<&'static str> {
        match self {
            PierSide::East => Some("east"),
            PierSide::West => Some("west"),
            PierSide::Unknown => None,
        }
    }
}

impl Serialize for PierSide {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.as_str() {
            Some(side) => serializer.serialize_str(side),
            None => serializer.serialize_none(),
        }
    }
}

impl<'de> Deserialize<'de> for PierSide {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match Option::<String>::deserialize(deserializer)?.as_deref() {
            None | Some("unknown") => Ok(PierSide::Unknown),
            Some("east") => Ok(PierSide::East),
            Some("west") => Ok(PierSide::West),
            Some(other) => Err(de::Error::unknown_variant(other, &["east", "west"])),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentMode {
    AltAz,
    Polar,
    GermanPolar,
}

impl AlignmentMode {
    /// ASCOM `AlignmentModes`: 0 = algAltAz, 1 = algPolar, 2 = algGermanPolar.
    pub fn from_ascom(value: i64) -> Option<Self> {
        match value {
            0 => Some(AlignmentMode::AltAz),
            1 => Some(AlignmentMode::Polar),
            2 => Some(AlignmentMode::GermanPolar),
            _ => None,
        }
    }
}

/// ASCOM `DriveRates` 0..3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriveRate {
    Sidereal,
    Lunar,
    Solar,
    King,
}

impl DriveRate {
    pub fn from_ascom(value: i64) -> Option<Self> {
        match value {
            0 => Some(DriveRate::Sidereal),
            1 => Some(DriveRate::Lunar),
            2 => Some(DriveRate::Solar),
            3 => Some(DriveRate::King),
            _ => None,
        }
    }

    pub fn to_ascom(self) -> i32 {
        match self {
            DriveRate::Sidereal => 0,
            DriveRate::Lunar => 1,
            DriveRate::Solar => 2,
            DriveRate::King => 3,
        }
    }
}

/// ASCOM `GuideDirections` 0..3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuideDirection {
    North,
    South,
    East,
    West,
}

impl GuideDirection {
    pub fn from_ascom(value: i64) -> Option<Self> {
        match value {
            0 => Some(GuideDirection::North),
            1 => Some(GuideDirection::South),
            2 => Some(GuideDirection::East),
            3 => Some(GuideDirection::West),
            _ => None,
        }
    }

    pub fn to_ascom(self) -> i32 {
        match self {
            GuideDirection::North => 0,
            GuideDirection::South => 1,
            GuideDirection::East => 2,
            GuideDirection::West => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SiteSource {
    #[default]
    Config,
    Mount,
    Operator,
}

/// Observing site. Longitude is east-positive.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Site {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub elev_m: f64,
    pub source: SiteSource,
}

impl Site {
    pub fn validate(&self) -> Result<(), String> {
        if !(self.lat_deg.is_finite() && (-90.0..=90.0).contains(&self.lat_deg)) {
            return Err(format!(
                "site latitude {} is outside [-90, 90]",
                self.lat_deg
            ));
        }
        if !(self.lon_deg.is_finite() && (-180.0..=360.0).contains(&self.lon_deg)) {
            return Err(format!(
                "site longitude {} is outside [-180, 360]",
                self.lon_deg
            ));
        }
        if !self.elev_m.is_finite() {
            return Err("site elevation is not finite".into());
        }
        Ok(())
    }

    /// True when the two sites disagree beyond the tolerances (M-b).
    pub fn differs_from(
        &self,
        other: &Site,
        tolerance_deg: f64,
        elevation_tolerance_m: f64,
    ) -> bool {
        let dlon = (self.lon_deg - other.lon_deg + 540.0).rem_euclid(360.0) - 180.0;
        (self.lat_deg - other.lat_deg).abs() > tolerance_deg
            || dlon.abs() > tolerance_deg
            || (self.elev_m - other.elev_m).abs() > elevation_tolerance_m
    }
}

/// Timing quality of an exposure's start time (§2.14, A06).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClockQuality {
    DriverExposureStart,
    HostTriggerBracket,
    HostArrival,
    Simulator,
    #[default]
    Unknown,
}

/// `[mount] sync_mode`: sync the mount, or keep a solved offset (NINA NoSync).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MountSyncMode {
    #[default]
    Mount,
    Offset,
}

/// Physical role of a focuser temperature probe (F-f, H6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemperatureRole {
    Ambient,
    Tube,
    #[default]
    Unknown,
}

// ---------------------------------------------------------------------------
// Commands, sources, priorities and outcomes
// ---------------------------------------------------------------------------

/// Identity of one command across HTTP, relay, events and audit (v6 §15.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CommandId(pub Uuid);

impl CommandId {
    pub fn new() -> Self {
        CommandId(Uuid::new_v4())
    }
}

impl Default for CommandId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CommandId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Who issued a command. "Operator" is a source, not a priority (D9).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum CommandSource {
    Operator { overriding: bool },
    Relay { overriding: bool },
    Policy { model: String, version: String },
    Trigger { rule: String },
    Safety,
    System,
}

impl CommandSource {
    /// Direct UI or relay operator.
    pub fn is_operator(&self) -> bool {
        matches!(
            self,
            CommandSource::Operator { .. } | CommandSource::Relay { .. }
        )
    }

    /// Sequencing, triggers and policies: refused while a resume block is
    /// active (H22) and never preempt an operator-held domain (§2.9).
    pub fn is_automatic(&self) -> bool {
        matches!(
            self,
            CommandSource::Policy { .. } | CommandSource::Trigger { .. } | CommandSource::System
        )
    }

    pub fn overriding(&self) -> bool {
        matches!(
            self,
            CommandSource::Operator { overriding: true }
                | CommandSource::Relay { overriding: true }
        )
    }

    /// Short label for audit rows (`control_audit_logs.actor`).
    pub fn label(&self) -> String {
        match self {
            CommandSource::Operator { .. } => "operator".into(),
            CommandSource::Relay { .. } => "relay".into(),
            CommandSource::Policy { model, version } => format!("policy:{model}@{version}"),
            CommandSource::Trigger { rule } => format!("trigger:{rule}"),
            CommandSource::Safety => "safety".into(),
            CommandSource::System => "system".into(),
        }
    }
}

/// Autonomy mode of a request: v6 §7.5 request distinction, §7.11 L0-L3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutonomyMode {
    #[default]
    Manual,
    Advisory,
    Proposal,
    Bounded,
}

impl AutonomyMode {
    /// The v6 §7.11 ladder level, L0 to L3.
    pub fn level(self) -> u8 {
        match self {
            AutonomyMode::Manual => 0,
            AutonomyMode::Advisory => 1,
            AutonomyMode::Proposal => 2,
            AutonomyMode::Bounded => 3,
        }
    }
}

/// Arbiter priority, in v6 §7.3 order (highest first).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    #[serde(rename = "estop")]
    EStop,
    Safety,
    MountRecovery,
    Autofocus,
    Exposure,
    Recommendation,
}

impl Priority {
    /// Rank held by an operator command (§2.9 arbiter table).
    pub const OPERATOR_HELD_RANK: u8 = 5;

    /// Preemption rank of the priority alone (§2.9: 7 e-stop ... 1
    /// recommendation; 5 is reserved for operator-held domains).
    pub fn rank(self) -> u8 {
        match self {
            Priority::EStop => 7,
            Priority::Safety => 6,
            Priority::MountRecovery => 4,
            Priority::Autofocus => 3,
            Priority::Exposure => 2,
            Priority::Recommendation => 1,
        }
    }

    /// Rank of a permit: an operator command holds at least rank 5.
    pub fn preemption_rank(self, source: &CommandSource) -> u8 {
        if source.is_operator() {
            self.rank().max(Self::OPERATOR_HELD_RANK)
        } else {
            self.rank()
        }
    }
}

/// Final outcome of an operation (C-6). `Unknown` raises a resume block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", content = "detail", rename_all = "snake_case")]
pub enum OpOutcome {
    Succeeded,
    Failed(DeviceError),
    Aborted { by: Option<CommandId> },
    TimedOut,
    Unknown(DeviceError),
}

impl OpOutcome {
    pub fn is_success(&self) -> bool {
        matches!(self, OpOutcome::Succeeded)
    }

    pub fn label(&self) -> &'static str {
        match self {
            OpOutcome::Succeeded => "succeeded",
            OpOutcome::Failed(_) => "failed",
            OpOutcome::Aborted { .. } => "aborted",
            OpOutcome::TimedOut => "timed_out",
            OpOutcome::Unknown(_) => "unknown",
        }
    }
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

/// Registry `failure_class` values (§2.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureClass {
    Disconnected,
    Unsupported,
    Timeout,
    OutcomeUnknown,
    Driver,
    DriverPanic,
    Invalid,
}

/// Per-device link state: `Connecting -> Ready -> Degraded -> Lost ->
/// Disconnected` (§2.8). Published as the registry field `state`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkState {
    Connecting,
    Ready,
    Degraded,
    Lost,
    #[default]
    Disconnected,
}

impl LinkState {
    /// The registry `connected` flag. The registry `status` keeps its two
    /// values (D8): `Lost` is `connected: false, status: "disconnected"`.
    pub fn is_connected(self) -> bool {
        matches!(self, LinkState::Ready | LinkState::Degraded)
    }

    pub fn registry_status(self) -> &'static str {
        if self.is_connected() {
            "connected"
        } else {
            "disconnected"
        }
    }

    /// Abort still runs on these states, best effort (D19).
    pub fn abort_allowed(self) -> bool {
        !matches!(self, LinkState::Disconnected)
    }
}

/// Health of one connected device, published by its worker (§2.8).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct HealthState {
    pub state: LinkState,
    pub failure_class: Option<FailureClass>,
    /// Mirrors `last_error`; the frontend reads `device_health[x].reason`.
    pub reason: Option<String>,
    pub consecutive_failures: u32,
    pub last_ok_at: Option<DateTime<Utc>>,
    pub last_ok_mono_ns: Option<u64>,
    /// Monotonic time of the last state change.
    pub changed_mono_ns: u64,
}

impl HealthState {
    pub fn connecting(now_mono_ns: u64) -> Self {
        Self {
            state: LinkState::Connecting,
            changed_mono_ns: now_mono_ns,
            ..Self::default()
        }
    }

    pub fn ready(now_utc: DateTime<Utc>, now_mono_ns: u64) -> Self {
        Self {
            state: LinkState::Ready,
            last_ok_at: Some(now_utc),
            last_ok_mono_ns: Some(now_mono_ns),
            changed_mono_ns: now_mono_ns,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn pier_side_serialises_as_frontend_strings_or_null() {
        assert_eq!(serde_json::to_value(PierSide::East).unwrap(), json!("east"));
        assert_eq!(serde_json::to_value(PierSide::West).unwrap(), json!("west"));
        assert_eq!(
            serde_json::to_value(PierSide::Unknown).unwrap(),
            json!(null)
        );
        for side in [PierSide::East, PierSide::West, PierSide::Unknown] {
            let text = serde_json::to_string(&side).unwrap();
            assert_eq!(serde_json::from_str::<PierSide>(&text).unwrap(), side);
        }
        assert!(serde_json::from_str::<PierSide>("\"north\"").is_err());
        assert_eq!(PierSide::from_ascom(0), PierSide::East);
        assert_eq!(PierSide::from_ascom(1), PierSide::West);
        assert_eq!(PierSide::from_ascom(-1), PierSide::Unknown);
    }

    #[test]
    fn observed_round_trips_and_never_invents_values() {
        let at = Utc::now();
        let fresh = Observed::fresh(12.5_f64, Source::DeviceState, at, 5_000_000)
            .with_latency_ms(40)
            .with_device_time(at);
        let text = serde_json::to_string(&fresh).unwrap();
        assert_eq!(serde_json::from_str::<Observed<f64>>(&text).unwrap(), fresh);
        assert_eq!(fresh.age_ms(12_000_000), Some(7));

        let stale = fresh.clone().into_stale();
        assert_eq!(stale.state, ObsState::Stale);
        assert_eq!(stale.known(), Some(&12.5));
        assert_eq!(stale.fresh_value(), None);

        let never: Observed<f64> = Observed::not_set(Source::Property);
        assert_eq!(never.clone().into_stale().state, ObsState::Error);
        assert_eq!(never.known(), None);
        let unsupported: Observed<bool> = Observed::unsupported(Source::Property);
        assert_eq!(unsupported.into_stale().state, ObsState::Unsupported);
        let json = serde_json::to_value(Observed::<f64>::default()).unwrap();
        assert_eq!(json["value"], json!(null));
        assert_eq!(json["state"], json!("not_set"));
    }

    #[test]
    fn alpaca_error_numbers_follow_the_c3_table() {
        use ascom_error::*;
        let cases = [
            (NOT_IMPLEMENTED, ErrorClass::Unsupported),
            (INVALID_VALUE, ErrorClass::InvalidValue),
            (VALUE_NOT_SET, ErrorClass::Driver),
            (NOT_CONNECTED, ErrorClass::NotConnected),
            (INVALID_WHILE_PARKED, ErrorClass::InvalidWhileParked),
            (INVALID_WHILE_SLAVED, ErrorClass::InvalidOperation),
            (INVALID_OPERATION, ErrorClass::InvalidOperation),
            (ACTION_NOT_IMPLEMENTED, ErrorClass::Unsupported),
            (OPERATION_CANCELLED, ErrorClass::Cancelled),
            (0x500, ErrorClass::Driver),
            (0xFFF, ErrorClass::Driver),
        ];
        for (number, class) in cases {
            let err = DeviceError::from_alpaca_error(number, "x");
            assert_eq!(err.class, class, "{number:#x}");
            assert_eq!(err.alpaca_error_number(), Some(number));
            assert_eq!(err.http_status(), None);
        }
        assert!(DeviceError::from_alpaca_error(VALUE_NOT_SET, "").is_not_set());
        assert!(DeviceError::from_alpaca_error(NOT_IMPLEMENTED, "").is_unknown_member());
    }

    #[test]
    fn http_errors_keep_their_status_and_plain_text_body() {
        let err = DeviceError::from_http_status(400, "Unknown member 'devicestate'\n");
        assert_eq!(err.http_status(), Some(400));
        assert_eq!(err.alpaca_error_number(), None);
        assert_eq!(err.class, ErrorClass::InvalidValue);
        assert!(err.is_unknown_member());
        assert!(err.message.contains("Unknown member 'devicestate'"));
        assert!(DeviceError::from_http_status(404, "").is_unknown_member());
        let err = DeviceError::from_http_status(500, "driver exploded");
        assert_eq!(err.class, ErrorClass::Driver);
        assert!(!err.is_unknown_member());
    }

    #[test]
    fn error_classes_map_to_the_api_contract() {
        assert_eq!(ErrorClass::NotConnected.http_status(), 412);
        assert_eq!(ErrorClass::Unsupported.http_status(), 501);
        assert_eq!(ErrorClass::InvalidValue.http_status(), 400);
        assert_eq!(ErrorClass::InvalidWhileParked.http_status(), 409);
        assert_eq!(ErrorClass::OwnershipConflict.http_status(), 409);
        assert_eq!(ErrorClass::SafetyRejected.http_status(), 422);
        assert_eq!(ErrorClass::Timeout.http_status(), 504);
        assert_eq!(ErrorClass::Transport.http_status(), 502);
        let err = DeviceError::outcome_unknown("lost mid-slew");
        assert_eq!(err.api_code(DeviceKind::Mount), "mount_outcome_unknown");
        assert_eq!(
            serde_json::to_value(ErrorClass::InvalidWhileParked).unwrap(),
            json!("invalid_while_parked")
        );
    }

    #[test]
    fn enums_use_their_contract_names() {
        assert_eq!(
            serde_json::to_value(EqFrame::JNowApparent).unwrap(),
            json!("jnow_apparent")
        );
        assert_eq!(
            serde_json::to_value(EqFrame::JNowMean).unwrap(),
            json!("jnow_mean")
        );
        assert_eq!(
            serde_json::to_value(EqFrame::J2000).unwrap(),
            json!("j2000")
        );
        assert_eq!(
            serde_json::to_value(AlignmentMode::Polar).unwrap(),
            json!("polar")
        );
        assert_eq!(
            serde_json::to_value(AlignmentMode::GermanPolar).unwrap(),
            json!("german_polar")
        );
        assert_eq!(
            serde_json::to_value(DriverId::DirectGuider).unwrap(),
            json!("direct")
        );
        assert_eq!(DriverId::parse("simulator"), Some(DriverId::Sim));
        assert_eq!(DriverId::parse("lx200"), None, "O4 removed native LX200");
        assert_eq!(
            serde_json::to_value(ClockQuality::HostTriggerBracket).unwrap(),
            json!("host_trigger_bracket")
        );
        assert_eq!(
            serde_json::to_value(Qualification::Qualified {
                record_id: "r1".into()
            })
            .unwrap(),
            json!({"state": "qualified", "record_id": "r1"})
        );
        assert_eq!(
            serde_json::to_value(CommandSource::Operator { overriding: true }).unwrap(),
            json!({"source": "operator", "overriding": true})
        );
        for rate in [
            DriveRate::Sidereal,
            DriveRate::Lunar,
            DriveRate::Solar,
            DriveRate::King,
        ] {
            assert_eq!(
                DriveRate::from_ascom(i64::from(rate.to_ascom())),
                Some(rate)
            );
        }
        for dir in [
            GuideDirection::North,
            GuideDirection::South,
            GuideDirection::East,
            GuideDirection::West,
        ] {
            assert_eq!(
                GuideDirection::from_ascom(i64::from(dir.to_ascom())),
                Some(dir)
            );
        }
    }

    #[test]
    fn equatorial_system_maps_per_m_a() {
        let (frame, reason) = EqFrame::from_equatorial_system(1, JNowFlavour::Apparent);
        assert_eq!((frame, reason), (EqFrame::JNowApparent, None));
        assert_eq!(
            EqFrame::from_equatorial_system(1, JNowFlavour::Mean).0,
            EqFrame::JNowMean
        );
        assert_eq!(
            EqFrame::from_equatorial_system(2, JNowFlavour::Apparent).0,
            EqFrame::J2000
        );
        assert_eq!(
            EqFrame::from_equatorial_system(4, JNowFlavour::Apparent),
            (EqFrame::Unknown, Some("unsupported_epoch"))
        );
        assert_eq!(
            EqFrame::from_equatorial_system(0, JNowFlavour::Apparent).0,
            EqFrame::Unknown
        );
    }

    #[test]
    fn priorities_and_sources_follow_the_arbiter_table() {
        let operator = CommandSource::Operator { overriding: false };
        let policy = CommandSource::Policy {
            model: "fw2".into(),
            version: "1".into(),
        };
        assert_eq!(Priority::EStop.preemption_rank(&operator), 7);
        assert_eq!(Priority::Safety.preemption_rank(&CommandSource::Safety), 6);
        assert_eq!(Priority::Exposure.preemption_rank(&operator), 5);
        assert_eq!(Priority::MountRecovery.preemption_rank(&policy), 4);
        assert_eq!(Priority::Recommendation.preemption_rank(&policy), 1);
        assert!(policy.is_automatic() && !policy.is_operator());
        assert!(!CommandSource::Safety.is_automatic());
        assert!(CommandSource::Relay { overriding: true }.overriding());
    }

    #[test]
    fn coordinates_wrap_and_refuse_cross_frame_separation() {
        assert_eq!(EqCoord::wrap_ra_h(-1.0), 23.0);
        assert_eq!(EqCoord::wrap_ra_h(24.0), 0.0);
        let a = EqCoord::new(23.99, 10.0, EqFrame::J2000);
        let b = EqCoord::new(0.01, 10.0, EqFrame::J2000);
        let sep = a.separation_arcmin(&b).unwrap();
        // 0.02 h = 0.3 deg of RA at Dec 10: 18' * cos(10 deg) = 17.73'.
        assert!((sep - 17.73).abs() < 0.05, "{sep}");
        let c = EqCoord::new(0.01, 10.0, EqFrame::JNowApparent);
        assert_eq!(a.separation_arcmin(&c), None);
    }

    #[test]
    fn identity_keys_follow_section_2_6() {
        assert_eq!(
            identity_key::alpaca(Some("abc-123"), "10.0.0.5", 11111, "telescope", 0),
            "alpaca:abc-123"
        );
        assert_eq!(
            identity_key::alpaca(None, "ESP32.local", 80, "Focuser", 1),
            "alpaca:esp32.local:80/focuser/1"
        );
        assert_eq!(
            identity_key::indi("pi", 7624, "Telescope Simulator"),
            "indi:pi:7624/Telescope Simulator"
        );
        assert_eq!(
            identity_key::phd2("localhost", 4400, Some("SV105")),
            "phd2:localhost:4400/SV105"
        );
        assert_eq!(
            identity_key::sim(DeviceKind::FilterWheel),
            "sim:filter_wheel"
        );
        assert_eq!(
            Endpoint::phd2_instance("localhost", 2),
            Endpoint::Phd2 {
                host: "localhost".into(),
                port: 4401
            }
        );
    }

    #[test]
    fn site_tolerances_wrap_longitude() {
        let a = Site {
            lat_deg: 40.0,
            lon_deg: 179.9995,
            elev_m: 100.0,
            source: SiteSource::Config,
        };
        let b = Site {
            lon_deg: -179.9999,
            ..a
        };
        assert!(!a.differs_from(&b, 0.001, 10.0));
        assert!(a.differs_from(&Site { elev_m: 120.0, ..a }, 0.001, 10.0));
        assert!(Site { lat_deg: 91.0, ..a }.validate().is_err());
    }

    #[test]
    fn link_state_keeps_the_two_registry_statuses() {
        assert_eq!(LinkState::Ready.registry_status(), "connected");
        assert_eq!(LinkState::Degraded.registry_status(), "connected");
        assert_eq!(LinkState::Lost.registry_status(), "disconnected");
        assert!(LinkState::Lost.abort_allowed());
        assert!(!LinkState::Disconnected.abort_allowed());
    }
}

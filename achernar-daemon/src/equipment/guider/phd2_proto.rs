//! PHD2 event-server protocol types (spec §3.5 G-a, G-b; WP1A-7).
//!
//! The protocol facts here (line framing, the event and method names, field
//! names such as `Timestamp`, `RADistanceRaw` or `SettleDone.Status`) come
//! from the PHD2 EventMonitoring wiki and the method table in PHD2's
//! `event_server.cpp` (BSD-3), used as API facts only. No PHD2 or NINA code
//! is copied (§6).
//!
//! - Framing: one JSON object per line, terminated by LF or CRLF
//!   ([`LineFramer`]).
//! - A line with an `Event` key is an event ([`Phd2Event`]). A line with an
//!   `id` and `jsonrpc` (or `result` / `error`) is a reply to one of our
//!   requests ([`RpcReply`]). Request ids are integers.
//! - Every field is parsed leniently: unknown keys are ignored, a field of
//!   an unexpected type reads as absent, and optional fields such as
//!   `RALimited` and `DecLimited` may be missing. Nothing is synthesised: an
//!   absent value stays `None`.

use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};

use crate::equipment::commands::{Axis, Settle};
use crate::equipment::types::{DeviceError, ErrorClass};

/// Largest line accepted. PHD2 lines are small (Achernar never asks for star
/// images); a longer line is discarded rather than buffered without bound.
pub const MAX_LINE_BYTES: usize = 1 << 20;

/// PHD2 method names used by the client (event_server method table).
pub mod method {
    pub const GET_APP_STATE: &str = "get_app_state";
    pub const GET_CONNECTED: &str = "get_connected";
    pub const SET_CONNECTED: &str = "set_connected";
    pub const GET_PROFILE: &str = "get_profile";
    pub const GET_CURRENT_EQUIPMENT: &str = "get_current_equipment";
    pub const GET_PIXEL_SCALE: &str = "get_pixel_scale";
    pub const GET_CALIBRATED: &str = "get_calibrated";
    pub const GET_CALIBRATION_DATA: &str = "get_calibration_data";
    pub const GET_DEC_GUIDE_MODE: &str = "get_dec_guide_mode";
    pub const GET_EXPOSURE: &str = "get_exposure";
    pub const GET_ALGO_PARAM_NAMES: &str = "get_algo_param_names";
    pub const GET_ALGO_PARAM: &str = "get_algo_param";
    pub const SET_ALGO_PARAM: &str = "set_algo_param";
    pub const GUIDE: &str = "guide";
    pub const DITHER: &str = "dither";
    pub const STOP_CAPTURE: &str = "stop_capture";
    pub const SET_PAUSED: &str = "set_paused";
    pub const SET_PROFILE: &str = "set_profile";
    pub const FIND_STAR: &str = "find_star";
    pub const FLIP_CALIBRATION: &str = "flip_calibration";
}

/// JSON-RPC error codes PHD2 uses besides its own small positive codes.
pub mod rpc_code {
    pub const PARSE_ERROR: i64 = -32700;
    pub const INVALID_REQUEST: i64 = -32600;
    pub const METHOD_NOT_FOUND: i64 = -32601;
    pub const INVALID_PARAMS: i64 = -32602;
}

// ---------------------------------------------------------------------------
// Framing
// ---------------------------------------------------------------------------

/// Splits a byte stream into lines. LF ends a line; a CR before it is
/// stripped, so LF and CRLF framing are equivalent. Chunk boundaries may fall
/// anywhere, including inside a multi-byte UTF-8 sequence or between CR and
/// LF.
#[derive(Debug, Default)]
pub struct LineFramer {
    buf: Vec<u8>,
    discarding: bool,
    overflows: u64,
}

impl LineFramer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed bytes; returns every line they complete. Blank lines are skipped.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        let mut lines = Vec::new();
        for &byte in bytes {
            if byte == b'\n' {
                if self.discarding {
                    self.discarding = false;
                } else if let Some(line) = Self::take_line(&mut self.buf) {
                    lines.push(line);
                }
                self.buf.clear();
                continue;
            }
            if self.discarding {
                continue;
            }
            if self.buf.len() >= MAX_LINE_BYTES {
                self.buf.clear();
                self.discarding = true;
                self.overflows += 1;
                continue;
            }
            self.buf.push(byte);
        }
        lines
    }

    /// The final unterminated line at end of stream, if any.
    pub fn finish(&mut self) -> Option<String> {
        if self.discarding {
            self.discarding = false;
            self.buf.clear();
            return None;
        }
        let line = Self::take_line(&mut self.buf);
        self.buf.clear();
        line
    }

    /// Lines discarded for exceeding [`MAX_LINE_BYTES`].
    pub fn overflows(&self) -> u64 {
        self.overflows
    }

    fn take_line(buf: &mut Vec<u8>) -> Option<String> {
        while matches!(buf.last(), Some(b'\r')) {
            buf.pop();
        }
        let text = String::from_utf8_lossy(buf).trim().to_string();
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    }
}

// ---------------------------------------------------------------------------
// Lenient field readers
// ---------------------------------------------------------------------------

fn value_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64().filter(|x| x.is_finite()),
        Value::String(s) => s.trim().parse::<f64>().ok().filter(|x| x.is_finite()),
        _ => None,
    }
}

fn value_i64(value: &Value) -> Option<i64> {
    match value {
        Value::Number(n) => n.as_i64().or_else(|| {
            n.as_f64()
                .filter(|x| x.is_finite())
                .map(|x| x.round() as i64)
        }),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}

fn value_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(b) => Some(*b),
        Value::Number(n) => n.as_i64().map(|x| x != 0),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn value_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn lenient_f64<'de, D: Deserializer<'de>>(d: D) -> Result<Option<f64>, D::Error> {
    Ok(Option::<Value>::deserialize(d)?
        .as_ref()
        .and_then(value_f64))
}

fn lenient_i64<'de, D: Deserializer<'de>>(d: D) -> Result<Option<i64>, D::Error> {
    Ok(Option::<Value>::deserialize(d)?
        .as_ref()
        .and_then(value_i64))
}

fn lenient_u64<'de, D: Deserializer<'de>>(d: D) -> Result<Option<u64>, D::Error> {
    Ok(Option::<Value>::deserialize(d)?
        .as_ref()
        .and_then(value_i64)
        .and_then(|x| u64::try_from(x).ok()))
}

fn lenient_bool<'de, D: Deserializer<'de>>(d: D) -> Result<Option<bool>, D::Error> {
    Ok(Option::<Value>::deserialize(d)?
        .as_ref()
        .and_then(value_bool))
}

fn lenient_string<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
    Ok(Option::<Value>::deserialize(d)?
        .as_ref()
        .and_then(value_string))
}

/// A PHD2 `Timestamp` (Unix seconds, PHD2's clock) as UTC.
pub fn phd2_time(timestamp: f64) -> Option<DateTime<Utc>> {
    if !timestamp.is_finite() || timestamp < 0.0 {
        return None;
    }
    let secs = timestamp.floor();
    let nanos = ((timestamp - secs) * 1e9).round().clamp(0.0, 999_999_999.0) as u32;
    Utc.timestamp_opt(secs as i64, nanos).single()
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// `Version`: the first message after connecting.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct VersionEvent {
    #[serde(rename = "PHDVersion", default, deserialize_with = "lenient_string")]
    pub phd_version: Option<String>,
    #[serde(rename = "PHDSubver", default, deserialize_with = "lenient_string")]
    pub phd_subver: Option<String>,
    #[serde(rename = "MsgVersion", default, deserialize_with = "lenient_i64")]
    pub msg_version: Option<i64>,
    #[serde(rename = "OverlapSupport", default, deserialize_with = "lenient_bool")]
    pub overlap_support: Option<bool>,
}

impl VersionEvent {
    /// `2.6.11dev3`-style version text, or `None` when PHD2 sent none.
    pub fn display(&self) -> Option<String> {
        let version = self.phd_version.as_deref()?.trim();
        if version.is_empty() {
            return None;
        }
        Some(match self.phd_subver.as_deref().map(str::trim) {
            Some(sub) if !sub.is_empty() => format!("{version}{sub}"),
            _ => version.to_string(),
        })
    }
}

/// `GuideStep`, with PHD2's own names, signs and units (G-h). `Frame`,
/// `Time`, `dx` and `dy` are required for a usable step; everything else is
/// optional (PHD2 omits the pulse fields when no pulse was sent, and older
/// versions omit `RALimited` / `DecLimited`).
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct GuideStepEvent {
    #[serde(rename = "Frame", default, deserialize_with = "lenient_u64")]
    pub frame: Option<u64>,
    #[serde(rename = "Time", default, deserialize_with = "lenient_f64")]
    pub time: Option<f64>,
    #[serde(rename = "Mount", default, deserialize_with = "lenient_string")]
    pub mount: Option<String>,
    #[serde(
        rename = "dx",
        alias = "Dx",
        alias = "DX",
        default,
        deserialize_with = "lenient_f64"
    )]
    pub dx: Option<f64>,
    #[serde(
        rename = "dy",
        alias = "Dy",
        alias = "DY",
        default,
        deserialize_with = "lenient_f64"
    )]
    pub dy: Option<f64>,
    #[serde(rename = "RADistanceRaw", default, deserialize_with = "lenient_f64")]
    pub ra_distance_raw: Option<f64>,
    #[serde(
        rename = "DECDistanceRaw",
        alias = "DecDistanceRaw",
        default,
        deserialize_with = "lenient_f64"
    )]
    pub dec_distance_raw: Option<f64>,
    #[serde(rename = "RADistanceGuide", default, deserialize_with = "lenient_f64")]
    pub ra_distance_guide: Option<f64>,
    #[serde(
        rename = "DECDistanceGuide",
        alias = "DecDistanceGuide",
        default,
        deserialize_with = "lenient_f64"
    )]
    pub dec_distance_guide: Option<f64>,
    #[serde(rename = "RADuration", default, deserialize_with = "lenient_i64")]
    pub ra_duration_ms: Option<i64>,
    #[serde(rename = "RADirection", default, deserialize_with = "lenient_string")]
    pub ra_direction: Option<String>,
    #[serde(
        rename = "DECDuration",
        alias = "DecDuration",
        default,
        deserialize_with = "lenient_i64"
    )]
    pub dec_duration_ms: Option<i64>,
    #[serde(
        rename = "DECDirection",
        alias = "DecDirection",
        default,
        deserialize_with = "lenient_string"
    )]
    pub dec_direction: Option<String>,
    #[serde(rename = "StarMass", default, deserialize_with = "lenient_f64")]
    pub star_mass: Option<f64>,
    #[serde(rename = "SNR", default, deserialize_with = "lenient_f64")]
    pub snr: Option<f64>,
    #[serde(rename = "HFD", default, deserialize_with = "lenient_f64")]
    pub hfd: Option<f64>,
    #[serde(rename = "AvgDist", default, deserialize_with = "lenient_f64")]
    pub avg_dist: Option<f64>,
    #[serde(
        rename = "RALimited",
        alias = "RaLimited",
        default,
        deserialize_with = "lenient_bool"
    )]
    pub ra_limited: Option<bool>,
    #[serde(
        rename = "DecLimited",
        alias = "DECLimited",
        default,
        deserialize_with = "lenient_bool"
    )]
    pub dec_limited: Option<bool>,
    #[serde(rename = "ErrorCode", default, deserialize_with = "lenient_i64")]
    pub error_code: Option<i64>,
}

impl GuideStepEvent {
    /// True when the four required fields are present.
    pub fn is_complete(&self) -> bool {
        self.frame.is_some() && self.time.is_some() && self.dx.is_some() && self.dy.is_some()
    }
}

/// `StarLost`, with all its fields (G-g).
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct StarLostEvent {
    #[serde(rename = "Frame", default, deserialize_with = "lenient_u64")]
    pub frame: Option<u64>,
    #[serde(rename = "Time", default, deserialize_with = "lenient_f64")]
    pub time: Option<f64>,
    #[serde(rename = "StarMass", default, deserialize_with = "lenient_f64")]
    pub star_mass: Option<f64>,
    #[serde(rename = "SNR", default, deserialize_with = "lenient_f64")]
    pub snr: Option<f64>,
    #[serde(rename = "AvgDist", default, deserialize_with = "lenient_f64")]
    pub avg_dist: Option<f64>,
    #[serde(rename = "ErrorCode", default, deserialize_with = "lenient_i64")]
    pub error_code: Option<i64>,
    #[serde(rename = "Status", default, deserialize_with = "lenient_string")]
    pub status: Option<String>,
}

/// `Settling`: progress of a settle (G-f).
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct SettlingEvent {
    #[serde(rename = "Distance", default, deserialize_with = "lenient_f64")]
    pub distance: Option<f64>,
    #[serde(rename = "Time", default, deserialize_with = "lenient_f64")]
    pub time: Option<f64>,
    #[serde(rename = "SettleTime", default, deserialize_with = "lenient_f64")]
    pub settle_time: Option<f64>,
    #[serde(rename = "StarLocked", default, deserialize_with = "lenient_bool")]
    pub star_locked: Option<bool>,
}

/// `SettleDone`: `Status` 0 is success.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct SettleDoneEvent {
    #[serde(rename = "Status", default, deserialize_with = "lenient_i64")]
    pub status: Option<i64>,
    #[serde(rename = "Error", default, deserialize_with = "lenient_string")]
    pub error: Option<String>,
    #[serde(rename = "TotalFrames", default, deserialize_with = "lenient_u64")]
    pub total_frames: Option<u64>,
    #[serde(rename = "DroppedFrames", default, deserialize_with = "lenient_u64")]
    pub dropped_frames: Option<u64>,
}

impl SettleDoneEvent {
    /// A missing `Status` is not success: it is reported as an unknown
    /// status (`-1`), never as 0.
    pub fn status_code(&self) -> i32 {
        self.status
            .and_then(|s| i32::try_from(s).ok())
            .unwrap_or(-1)
    }

    pub fn succeeded(&self) -> bool {
        self.status == Some(0)
    }
}

/// `Calibrating`: one calibration step.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct CalibratingEvent {
    #[serde(rename = "Mount", default, deserialize_with = "lenient_string")]
    pub mount: Option<String>,
    #[serde(rename = "dir", default, deserialize_with = "lenient_string")]
    pub dir: Option<String>,
    #[serde(rename = "dist", default, deserialize_with = "lenient_f64")]
    pub dist: Option<f64>,
    #[serde(rename = "step", default, deserialize_with = "lenient_i64")]
    pub step: Option<i64>,
    #[serde(rename = "State", default, deserialize_with = "lenient_string")]
    pub state: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
struct MountOnly {
    #[serde(rename = "Mount", default, deserialize_with = "lenient_string")]
    mount: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
struct StateOnly {
    #[serde(rename = "State", default, deserialize_with = "lenient_string")]
    state: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
struct DitheredFields {
    #[serde(rename = "dx", alias = "Dx", default, deserialize_with = "lenient_f64")]
    dx: Option<f64>,
    #[serde(rename = "dy", alias = "Dy", default, deserialize_with = "lenient_f64")]
    dy: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
struct FrameOnly {
    #[serde(rename = "Frame", default, deserialize_with = "lenient_u64")]
    frame: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
struct ReasonOnly {
    #[serde(rename = "Reason", default, deserialize_with = "lenient_string")]
    reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
struct AlertFields {
    #[serde(rename = "Msg", default, deserialize_with = "lenient_string")]
    msg: Option<String>,
    #[serde(rename = "Type", default, deserialize_with = "lenient_string")]
    alert_type: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
struct ParamChangeFields {
    #[serde(rename = "Name", default, deserialize_with = "lenient_string")]
    name: Option<String>,
    #[serde(rename = "Value", default)]
    value: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
struct PositionFields {
    #[serde(rename = "X", default, deserialize_with = "lenient_f64")]
    x: Option<f64>,
    #[serde(rename = "Y", default, deserialize_with = "lenient_f64")]
    y: Option<f64>,
}

/// The typed body of a PHD2 event. Events Achernar does not know are kept
/// raw in [`EventBody::Other`].
#[derive(Debug, Clone, PartialEq)]
pub enum EventBody {
    Version(VersionEvent),
    AppState {
        state: Option<String>,
    },
    GuideStep(Box<GuideStepEvent>),
    StarLost(StarLostEvent),
    Settling(SettlingEvent),
    SettleBegin,
    SettleDone(SettleDoneEvent),
    GuidingDithered {
        dx: Option<f64>,
        dy: Option<f64>,
    },
    StartGuiding,
    GuidingStopped,
    Paused,
    Resumed,
    LoopingExposures {
        frame: Option<u64>,
    },
    LoopingExposuresStopped,
    StartCalibration {
        mount: Option<String>,
    },
    Calibrating(CalibratingEvent),
    CalibrationComplete {
        mount: Option<String>,
    },
    CalibrationFailed {
        reason: Option<String>,
    },
    CalibrationDataFlipped {
        mount: Option<String>,
    },
    LockPositionSet {
        x: Option<f64>,
        y: Option<f64>,
    },
    LockPositionLost,
    LockPositionShiftLimitReached,
    StarSelected {
        x: Option<f64>,
        y: Option<f64>,
    },
    Alert {
        message: Option<String>,
        alert_type: Option<String>,
    },
    GuideParamChange {
        name: Option<String>,
        value: Option<Value>,
    },
    ConfigurationChange,
    Other(Value),
}

/// One PHD2 event with its envelope.
#[derive(Debug, Clone, PartialEq)]
pub struct Phd2Event {
    /// The `Event` name, e.g. `GuideStep`.
    pub name: String,
    /// `Timestamp`: Unix seconds on PHD2's clock.
    pub timestamp: Option<f64>,
    pub host: Option<String>,
    pub inst: Option<u64>,
    pub body: EventBody,
}

impl Phd2Event {
    /// The PHD2 `Timestamp` as UTC (device time, §2.14).
    pub fn device_time(&self) -> Option<DateTime<Utc>> {
        self.timestamp.and_then(phd2_time)
    }

    fn from_value(name: String, value: Value) -> Self {
        let timestamp = value.get("Timestamp").and_then(value_f64);
        let host = value.get("Host").and_then(value_string);
        let inst = value
            .get("Inst")
            .and_then(value_i64)
            .and_then(|x| u64::try_from(x).ok());
        let body = parse_body(&name, value);
        Self {
            name,
            timestamp,
            host,
            inst,
            body,
        }
    }
}

fn fields<T: for<'de> Deserialize<'de> + Default>(value: Value) -> T {
    serde_json::from_value(value).unwrap_or_default()
}

fn parse_body(name: &str, value: Value) -> EventBody {
    match name {
        "Version" => EventBody::Version(fields(value)),
        "AppState" => EventBody::AppState {
            state: fields::<StateOnly>(value).state,
        },
        "GuideStep" => EventBody::GuideStep(Box::new(fields(value))),
        "StarLost" => EventBody::StarLost(fields(value)),
        "Settling" => EventBody::Settling(fields(value)),
        "SettleBegin" => EventBody::SettleBegin,
        "SettleDone" => EventBody::SettleDone(fields(value)),
        "GuidingDithered" => {
            let f: DitheredFields = fields(value);
            EventBody::GuidingDithered { dx: f.dx, dy: f.dy }
        }
        "StartGuiding" => EventBody::StartGuiding,
        "GuidingStopped" => EventBody::GuidingStopped,
        "Paused" => EventBody::Paused,
        "Resumed" => EventBody::Resumed,
        "LoopingExposures" => EventBody::LoopingExposures {
            frame: fields::<FrameOnly>(value).frame,
        },
        "LoopingExposuresStopped" => EventBody::LoopingExposuresStopped,
        "StartCalibration" => EventBody::StartCalibration {
            mount: fields::<MountOnly>(value).mount,
        },
        "Calibrating" => EventBody::Calibrating(fields(value)),
        "CalibrationComplete" => EventBody::CalibrationComplete {
            mount: fields::<MountOnly>(value).mount,
        },
        "CalibrationFailed" => EventBody::CalibrationFailed {
            reason: fields::<ReasonOnly>(value).reason,
        },
        "CalibrationDataFlipped" => EventBody::CalibrationDataFlipped {
            mount: fields::<MountOnly>(value).mount,
        },
        "LockPositionSet" => {
            let f: PositionFields = fields(value);
            EventBody::LockPositionSet { x: f.x, y: f.y }
        }
        "LockPositionLost" => EventBody::LockPositionLost,
        "LockPositionShiftLimitReached" => EventBody::LockPositionShiftLimitReached,
        "StarSelected" => {
            let f: PositionFields = fields(value);
            EventBody::StarSelected { x: f.x, y: f.y }
        }
        "Alert" => {
            let f: AlertFields = fields(value);
            EventBody::Alert {
                message: f.msg,
                alert_type: f.alert_type,
            }
        }
        "GuideParamChange" => {
            let f: ParamChangeFields = fields(value);
            EventBody::GuideParamChange {
                name: f.name,
                value: f.value,
            }
        }
        "ConfigurationChange" => EventBody::ConfigurationChange,
        _ => EventBody::Other(value),
    }
}

// ---------------------------------------------------------------------------
// RPC
// ---------------------------------------------------------------------------

/// A request. PHD2 ignores members it does not know, so only `method`,
/// `params` and an integer `id` are sent.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RpcRequest {
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
    pub id: u64,
}

impl RpcRequest {
    pub fn new(method: impl Into<String>, params: Option<Value>, id: u64) -> Self {
        Self {
            method: method.into(),
            params,
            id,
        }
    }

    /// The wire form: one JSON object terminated by CRLF.
    pub fn to_line(&self) -> String {
        let mut line = serde_json::to_string(self)
            .unwrap_or_else(|_| format!("{{\"method\":{:?},\"id\":{}}}", self.method, self.id));
        line.push_str("\r\n");
        line
    }
}

/// A JSON-RPC error object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
}

impl RpcError {
    /// The layer's error for a refused call. `method not found` means the
    /// PHD2 version lacks the method (`Unsupported`); bad parameters are
    /// `InvalidValue`; PHD2's own refusals ("cannot dither while settling")
    /// are `InvalidOperation`. The code is kept when it lies outside the
    /// HTTP range reserved by `DeviceError` (`100..=599`).
    pub fn to_device_error(&self, method: &str) -> DeviceError {
        let class = match self.code {
            rpc_code::METHOD_NOT_FOUND => ErrorClass::Unsupported,
            rpc_code::INVALID_PARAMS => ErrorClass::InvalidValue,
            rpc_code::PARSE_ERROR | rpc_code::INVALID_REQUEST => ErrorClass::Driver,
            _ => ErrorClass::InvalidOperation,
        };
        let message = format!(
            "PHD2 refused {method}: {} (code {})",
            self.message, self.code
        );
        match i32::try_from(self.code) {
            Ok(code) if !(100..=599).contains(&code) => {
                DeviceError::with_code(class, code, message)
            }
            _ => DeviceError::new(class, message),
        }
    }
}

/// A reply to one request.
#[derive(Debug, Clone, PartialEq)]
pub struct RpcReply {
    pub id: Option<u64>,
    pub outcome: Result<Value, RpcError>,
}

/// One classified line.
#[derive(Debug, Clone, PartialEq)]
pub enum Phd2Message {
    Event(Phd2Event),
    Reply(RpcReply),
    /// Valid JSON that is neither (ignored and counted).
    Other(Value),
}

/// Why a line could not be read.
#[derive(Debug, Clone, PartialEq)]
pub struct ProtoError {
    pub message: String,
}

impl std::fmt::Display for ProtoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// Classify one line (G-a): `Event` key → event; `id` with `jsonrpc`,
/// `result` or `error` → reply.
pub fn parse_line(line: &str) -> Result<Phd2Message, ProtoError> {
    let value: Value = serde_json::from_str(line.trim()).map_err(|e| ProtoError {
        message: format!("not JSON: {e}"),
    })?;
    if !value.is_object() {
        return Ok(Phd2Message::Other(value));
    }
    if let Some(name) = value.get("Event").and_then(value_string) {
        return Ok(Phd2Message::Event(Phd2Event::from_value(name, value)));
    }
    let has_rpc_shape = value.get("jsonrpc").is_some()
        || value.get("result").is_some()
        || value.get("error").is_some();
    if value.get("id").is_some() && has_rpc_shape {
        let id = value
            .get("id")
            .and_then(value_i64)
            .and_then(|x| u64::try_from(x).ok());
        let outcome = match value.get("error").filter(|e| !e.is_null()) {
            Some(error) => Err(RpcError {
                code: error.get("code").and_then(value_i64).unwrap_or(0),
                message: error
                    .get("message")
                    .and_then(value_string)
                    .unwrap_or_else(|| error.to_string()),
            }),
            None => Ok(value.get("result").cloned().unwrap_or(Value::Null)),
        };
        return Ok(Phd2Message::Reply(RpcReply { id, outcome }));
    }
    Ok(Phd2Message::Other(value))
}

// ---------------------------------------------------------------------------
// Parameters and results
// ---------------------------------------------------------------------------

/// PHD2 reads `time` and `timeout` as whole seconds. Fractions are rounded
/// up, so the settle Achernar asked for is never shortened.
pub fn sent_settle(settle: &Settle) -> Settle {
    Settle {
        pixels: settle.pixels,
        time_s: settle.time_s.max(0.0).ceil(),
        timeout_s: settle.timeout_s.max(0.0).ceil(),
    }
}

/// The `settle` object `{pixels, time, timeout}`.
pub fn settle_param(settle: &Settle) -> Value {
    let sent = sent_settle(settle);
    json!({
        "pixels": sent.pixels,
        "time": sent.time_s as i64,
        "timeout": sent.timeout_s as i64,
    })
}

/// `guide {settle, recalibrate}` (G-c).
pub fn guide_params(settle: &Settle, recalibrate: bool) -> Value {
    json!({"settle": settle_param(settle), "recalibrate": recalibrate})
}

/// `dither {amount, raOnly, settle}` (G-f). `amount` is in guide-camera
/// pixels.
pub fn dither_params(amount_px: f64, ra_only: bool, settle: &Settle) -> Value {
    json!({"amount": amount_px, "raOnly": ra_only, "settle": settle_param(settle)})
}

/// `set_paused [paused]`, or `[true, "full"]` for a full pause (G-e).
pub fn set_paused_params(paused: bool, full: bool) -> Value {
    if paused && full {
        json!([true, "full"])
    } else {
        json!([paused])
    }
}

pub fn axis_name(axis: Axis) -> &'static str {
    match axis {
        Axis::Ra => "ra",
        Axis::Dec => "dec",
    }
}

pub fn algo_param_names_params(axis: Axis) -> Value {
    json!([axis_name(axis)])
}

pub fn algo_param_params(axis: Axis, name: &str) -> Value {
    json!([axis_name(axis), name])
}

/// `get_profile` → the profile name.
pub fn parse_profile_name(result: &Value) -> Option<String> {
    result
        .get("name")
        .and_then(value_string)
        .or_else(|| result.as_str().map(str::to_string))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// `get_current_equipment` → (camera name, mount name).
pub fn parse_current_equipment(result: &Value) -> (Option<String>, Option<String>) {
    let name_of = |key: &str| {
        result
            .get(key)
            .and_then(|device| device.get("name").or(Some(device)))
            .and_then(Value::as_str)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };
    (name_of("camera"), name_of("mount"))
}

/// `get_pixel_scale` → arcsec per guide-camera pixel. A non-positive or null
/// scale is unknown.
pub fn parse_pixel_scale(result: &Value) -> Option<f64> {
    value_f64(result).filter(|x| *x > 0.0)
}

pub fn parse_bool(result: &Value) -> Option<bool> {
    value_bool(result)
}

pub fn parse_f64(result: &Value) -> Option<f64> {
    value_f64(result)
}

/// `get_exposure` → milliseconds.
pub fn parse_exposure_ms(result: &Value) -> Option<u32> {
    value_i64(result)
        .filter(|x| *x > 0)
        .and_then(|x| u32::try_from(x).ok())
}

pub fn parse_string(result: &Value) -> Option<String> {
    value_string(result)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// `get_algo_param_names` → names, in PHD2's order.
pub fn parse_string_list(result: &Value) -> Vec<String> {
    result
        .as_array()
        .map(|items| items.iter().filter_map(value_string).collect())
        .unwrap_or_default()
}

/// `get_app_state` → the state text.
pub fn parse_app_state(result: &Value) -> Option<String> {
    result
        .as_str()
        .map(str::to_string)
        .or_else(|| result.get("State").and_then(value_string))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framer_handles_lf_crlf_and_arbitrary_chunk_boundaries() {
        let wire = b"{\"a\":1}\r\n{\"b\":2}\n\r\n{\"c\":\"\xc3\xa9\"}\r\n{\"d\":4}";
        // Every split point, including between CR and LF and inside the
        // two-byte UTF-8 sequence.
        for split in 0..wire.len() {
            let mut framer = LineFramer::new();
            let mut lines = framer.push(&wire[..split]);
            lines.extend(framer.push(&wire[split..]));
            lines.extend(framer.finish());
            assert_eq!(
                lines,
                vec!["{\"a\":1}", "{\"b\":2}", "{\"c\":\"é\"}", "{\"d\":4}"],
                "split at {split}"
            );
        }
        // Byte-by-byte delivery.
        let mut framer = LineFramer::new();
        let mut lines = Vec::new();
        for byte in wire.iter() {
            lines.extend(framer.push(std::slice::from_ref(byte)));
        }
        assert_eq!(lines.len(), 3);
        assert_eq!(framer.finish().as_deref(), Some("{\"d\":4}"));
    }

    #[test]
    fn framer_discards_oversized_lines_and_recovers() {
        let mut framer = LineFramer::new();
        let big = vec![b'x'; MAX_LINE_BYTES + 10];
        assert!(framer.push(&big).is_empty());
        let lines = framer.push(b"tail\n{\"ok\":true}\n");
        assert_eq!(lines, vec!["{\"ok\":true}"]);
        assert_eq!(framer.overflows(), 1);
    }

    #[test]
    fn guide_steps_keep_phd2_names_signs_and_optional_fields() {
        let line = r#"{"Event":"GuideStep","Timestamp":1700000000.25,"Host":"rig","Inst":1,
            "Frame":42,"Time":84.5,"Mount":"On-camera","dx":-0.31,"dy":0.12,
            "RADistanceRaw":-0.27,"DECDistanceRaw":0.18,"RADistanceGuide":-0.2,
            "DECDistanceGuide":0.0,"RADuration":120,"RADirection":"East",
            "StarMass":15123.0,"SNR":37.2,"HFD":2.1,"AvgDist":0.33,
            "SomethingNew":{"nested":true}}"#
            .replace('\n', " ");
        let Phd2Message::Event(event) = parse_line(&line).unwrap() else {
            panic!("not an event");
        };
        assert_eq!(event.name, "GuideStep");
        assert_eq!(event.timestamp, Some(1700000000.25));
        assert_eq!(event.inst, Some(1));
        let device_time = event.device_time().unwrap();
        assert_eq!(device_time.timestamp(), 1_700_000_000);
        assert_eq!(device_time.timestamp_subsec_millis(), 250);
        let EventBody::GuideStep(step) = event.body else {
            panic!("not a step");
        };
        assert!(step.is_complete());
        assert_eq!(step.frame, Some(42));
        assert_eq!(step.time, Some(84.5));
        assert_eq!(step.dx, Some(-0.31), "sign kept");
        assert_eq!(step.ra_distance_raw, Some(-0.27));
        assert_eq!(step.dec_distance_guide, Some(0.0));
        assert_eq!(step.ra_duration_ms, Some(120));
        assert_eq!(step.ra_direction.as_deref(), Some("East"));
        assert_eq!(step.dec_duration_ms, None, "no Dec pulse sent");
        assert_eq!(step.ra_limited, None, "older PHD2 omits RALimited");
        assert_eq!(step.dec_limited, None);
        assert_eq!(step.error_code, None);

        let with_limits = r#"{"Event":"GuideStep","Frame":1,"Time":1,"dx":0,"dy":0,"RALimited":true,"DecLimited":false,"ErrorCode":3}"#;
        let Phd2Message::Event(Phd2Event {
            body: EventBody::GuideStep(step),
            ..
        }) = parse_line(with_limits).unwrap()
        else {
            panic!();
        };
        assert_eq!(step.ra_limited, Some(true));
        assert_eq!(step.dec_limited, Some(false));
        assert_eq!(step.error_code, Some(3));

        // A field of the wrong type reads as absent; it never fails the event.
        let odd = r#"{"Event":"GuideStep","Frame":"7","Time":2,"dx":0.1,"dy":"oops"}"#;
        let Phd2Message::Event(Phd2Event {
            body: EventBody::GuideStep(step),
            ..
        }) = parse_line(odd).unwrap()
        else {
            panic!();
        };
        assert_eq!(step.frame, Some(7));
        assert_eq!(step.dy, None);
        assert!(!step.is_complete());
    }

    #[test]
    fn settle_and_lifecycle_events_parse() {
        let parse = |line: &str| match parse_line(line).unwrap() {
            Phd2Message::Event(event) => event.body,
            other => panic!("{other:?}"),
        };
        assert_eq!(
            parse(
                r#"{"Event":"Settling","Distance":0.8,"Time":3.5,"SettleTime":10,"StarLocked":true}"#
            ),
            EventBody::Settling(SettlingEvent {
                distance: Some(0.8),
                time: Some(3.5),
                settle_time: Some(10.0),
                star_locked: Some(true),
            })
        );
        let EventBody::SettleDone(done) = parse(
            r#"{"Event":"SettleDone","Status":1,"Error":"timed-out waiting for guider to settle","TotalFrames":12,"DroppedFrames":2}"#,
        ) else {
            panic!();
        };
        assert!(!done.succeeded());
        assert_eq!(done.status_code(), 1);
        assert_eq!(done.total_frames, Some(12));
        assert_eq!(SettleDoneEvent::default().status_code(), -1);
        assert_eq!(
            parse(r#"{"Event":"GuidingDithered","dx":3.1,"dy":-2.4}"#),
            EventBody::GuidingDithered {
                dx: Some(3.1),
                dy: Some(-2.4)
            }
        );
        assert_eq!(
            parse(r#"{"Event":"AppState","State":"LostLock"}"#),
            EventBody::AppState {
                state: Some("LostLock".into())
            }
        );
        let EventBody::Version(version) = parse(
            r#"{"Event":"Version","PHDVersion":"2.6.11","PHDSubver":"dev3","MsgVersion":1,"OverlapSupport":true}"#,
        ) else {
            panic!();
        };
        assert_eq!(version.display().as_deref(), Some("2.6.11dev3"));
        assert_eq!(version.msg_version, Some(1));
        assert_eq!(
            parse(r#"{"Event":"Alert","Msg":"Star lost repeatedly","Type":"warning"}"#),
            EventBody::Alert {
                message: Some("Star lost repeatedly".into()),
                alert_type: Some("warning".into())
            }
        );
        assert_eq!(
            parse(r#"{"Event":"GuideParamChange","Name":"Aggressiveness","Value":0.65}"#),
            EventBody::GuideParamChange {
                name: Some("Aggressiveness".into()),
                value: Some(json!(0.65))
            }
        );
        assert_eq!(
            parse(r#"{"Event":"CalibrationFailed","Reason":"star did not move"}"#),
            EventBody::CalibrationFailed {
                reason: Some("star did not move".into())
            }
        );
        let EventBody::StarLost(lost) = parse(
            r#"{"Event":"StarLost","Frame":9,"Time":20.0,"StarMass":0,"SNR":0,"AvgDist":1.2,"ErrorCode":2,"Status":"star lost"}"#,
        ) else {
            panic!();
        };
        assert_eq!(lost.error_code, Some(2));
        assert_eq!(lost.status.as_deref(), Some("star lost"));
        assert!(matches!(
            parse(r#"{"Event":"SomeFutureEvent","X":1}"#),
            EventBody::Other(_)
        ));
        assert_eq!(
            parse(r#"{"Event":"GuidingStopped"}"#),
            EventBody::GuidingStopped
        );
        assert_eq!(
            parse(r#"{"Event":"LockPositionLost"}"#),
            EventBody::LockPositionLost
        );
    }

    #[test]
    fn replies_are_told_apart_from_events() {
        assert_eq!(
            parse_line(r#"{"jsonrpc":"2.0","result":0,"id":7}"#).unwrap(),
            Phd2Message::Reply(RpcReply {
                id: Some(7),
                outcome: Ok(json!(0))
            })
        );
        assert_eq!(
            parse_line(r#"{"jsonrpc":"2.0","error":{"code":1,"message":"cannot dither while settling"},"id":8}"#)
                .unwrap(),
            Phd2Message::Reply(RpcReply {
                id: Some(8),
                outcome: Err(RpcError {
                    code: 1,
                    message: "cannot dither while settling".into()
                })
            })
        );
        // A result of null is still a successful reply.
        assert!(matches!(
            parse_line(r#"{"jsonrpc":"2.0","result":null,"id":"9"}"#).unwrap(),
            Phd2Message::Reply(RpcReply {
                id: Some(9),
                outcome: Ok(Value::Null)
            })
        ));
        assert!(matches!(
            parse_line(r#"{"hello":1}"#).unwrap(),
            Phd2Message::Other(_)
        ));
        assert!(parse_line("not json").is_err());
    }

    #[test]
    fn rpc_errors_map_to_layer_classes() {
        let err = RpcError {
            code: rpc_code::METHOD_NOT_FOUND,
            message: "method not found".into(),
        };
        let mapped = err.to_device_error("get_algo_param_names");
        assert_eq!(mapped.class, ErrorClass::Unsupported);
        assert_eq!(mapped.code, Some(-32601));
        assert_eq!(mapped.http_status(), None);
        let refused = RpcError {
            code: 1,
            message: "cannot dither while settling".into(),
        }
        .to_device_error("dither");
        assert_eq!(refused.class, ErrorClass::InvalidOperation);
        assert!(refused.message.contains("cannot dither while settling"));
        // A code in the reserved HTTP range is dropped from `code`.
        let odd = RpcError {
            code: 404,
            message: "x".into(),
        }
        .to_device_error("m");
        assert_eq!(odd.code, None);
        assert!(odd.message.contains("code 404"));
    }

    #[test]
    fn requests_and_parameters_use_phd2_shapes() {
        let req = RpcRequest::new(method::GET_APP_STATE, None, 3);
        assert_eq!(req.to_line(), "{\"method\":\"get_app_state\",\"id\":3}\r\n");
        let settle = Settle {
            pixels: 1.5,
            time_s: 9.2,
            timeout_s: 40.0,
        };
        assert_eq!(
            settle_param(&settle),
            json!({"pixels": 1.5, "time": 10, "timeout": 40})
        );
        assert_eq!(
            dither_params(5.0, true, &settle),
            json!({"amount": 5.0, "raOnly": true, "settle": {"pixels": 1.5, "time": 10, "timeout": 40}})
        );
        assert_eq!(guide_params(&settle, false)["recalibrate"], json!(false));
        assert_eq!(set_paused_params(true, true), json!([true, "full"]));
        assert_eq!(set_paused_params(true, false), json!([true]));
        assert_eq!(set_paused_params(false, true), json!([false]));
        assert_eq!(
            algo_param_params(Axis::Dec, "Aggressiveness"),
            json!(["dec", "Aggressiveness"])
        );
    }

    #[test]
    fn results_parse_without_inventing_values() {
        assert_eq!(
            parse_profile_name(&json!({"id": 1, "name": "SV105 guide"})).as_deref(),
            Some("SV105 guide")
        );
        let (camera, mount) = parse_current_equipment(&json!({
            "camera": {"name": "SVBONY SV105", "connected": true},
            "mount": {"name": "On-camera", "connected": true},
            "aux_mount": null
        }));
        assert_eq!(camera.as_deref(), Some("SVBONY SV105"));
        assert_eq!(mount.as_deref(), Some("On-camera"));
        assert_eq!(parse_current_equipment(&json!({})), (None, None));
        assert_eq!(parse_pixel_scale(&json!(1.48)), Some(1.48));
        assert_eq!(parse_pixel_scale(&json!(0)), None, "0 is unknown");
        assert_eq!(parse_pixel_scale(&Value::Null), None);
        assert_eq!(parse_exposure_ms(&json!(2000)), Some(2000));
        assert_eq!(
            parse_string_list(&json!(["minMove", "aggression"])),
            vec!["minMove", "aggression"]
        );
        assert_eq!(
            parse_app_state(&json!("Guiding")).as_deref(),
            Some("Guiding")
        );
        assert_eq!(phd2_time(-1.0), None);
    }
}

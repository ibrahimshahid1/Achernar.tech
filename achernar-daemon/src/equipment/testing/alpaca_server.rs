//! In-process Alpaca servers for tests (spec §5.6). Owner: WP1A-1.
//!
//! - [`TestAlpacaServer`]: an axum 0.7 Alpaca server on `127.0.0.1:0` with
//!   its own thread and tokio runtime, so tests drive the blocking client
//!   from plain std threads (never from a tokio worker). It serves
//!   [`ScriptedDevice`]s: members with values, scripted replies (Alpaca
//!   errors, plain-text HTTP errors, raw bodies, delays), PUT and GET
//!   handlers with state, the Platform 6 and Platform 7 connect members,
//!   `devicestate` with a chosen TimeStamp form, and the management API.
//!   Two ready profiles mirror the user's rig without testing for it (D15):
//!   [`ScriptedDevice::esp32_like_mount`] (IV4, Polar, Topocentric, constant
//!   SideOfPier 0, park, no home, DeviceState) and
//!   [`ScriptedDevice::platform6_focuser`] (IV3: no Connect/Connecting, no
//!   DeviceState; absolute, temperature).
//! - [`HttpScript`]: a raw HTTP/1.1 responder on `127.0.0.1:0` with exact
//!   control over each reply, including a stall and a dropped connection,
//!   for transport and retry tests.
//! - The O11 (l) adapter: [`ScriptedDevice::sim_mount`] and
//!   [`ScriptedDevice::sim_focuser`] serve the WP1A-4 kinematic simulator
//!   mount and the WP1A-5 simulator focuser behind real Alpaca HTTP (a
//!   [`SimBackend`]), so the native Alpaca drivers and the daemon's routes
//!   can be run end to end against moving devices (WP1B-6). Scripted
//!   replies still win over the simulator, for fault injection.
//!
//! Every request is recorded (verb, member, decoded parameters, content
//! type, ClientID, ClientTransactionID) at the moment it arrives. Like a
//! real server, an unparsable ClientID or ClientTransactionID gets HTTP 400,
//! and a PUT whose body is not form-encoded gets HTTP 400.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::future::IntoFuture;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use axum::extract::{Path, RawQuery, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use serde_json::{json, Value};

use crate::equipment::alpaca::errors::Verb;
use crate::equipment::alpaca::form;
use crate::equipment::sim::focuser::SimFocuser;
use crate::equipment::sim::mount::SimMount;
use crate::equipment::snapshot::{
    FocuserCapabilities, FocuserState, MountCapabilities, MountState,
};
use crate::equipment::traits::{Device, FocuserDriver, MountDriver, PollPlan};
use crate::equipment::types::{
    ascom_error, AlignmentMode, AltAz, DeviceError, DriveRate, EqCoord, ErrorClass, GuideDirection,
    ObsState, Observed, PierSide, Site, SiteSource,
};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// Replies
// ---------------------------------------------------------------------------

/// One scripted server reply.
#[derive(Debug, Clone)]
pub enum ServerReply {
    /// HTTP 200 with this `Value` and ErrorNumber 0.
    Value(Value),
    /// A void PUT success: ErrorNumber 0, no `Value`.
    Ok,
    /// HTTP 200 with a non-zero ErrorNumber and no `Value`.
    AlpacaError { number: i32, message: String },
    /// Any status, content type and body, sent verbatim.
    Raw {
        status: u16,
        content_type: Option<String>,
        body: String,
    },
    /// Wait (without blocking other requests), then reply.
    Delay(Duration, Box<ServerReply>),
}

impl ServerReply {
    pub fn value(value: impl Into<Value>) -> Self {
        ServerReply::Value(value.into())
    }

    /// A non-200 reply with a plain-text body.
    pub fn text(status: u16, body: &str) -> Self {
        ServerReply::Raw {
            status,
            content_type: Some("text/plain; charset=utf-8".into()),
            body: body.into(),
        }
    }

    pub fn raw(status: u16, content_type: Option<&str>, body: &str) -> Self {
        ServerReply::Raw {
            status,
            content_type: content_type.map(str::to_string),
            body: body.into(),
        }
    }

    pub fn alpaca_error(number: i32, message: &str) -> Self {
        ServerReply::AlpacaError {
            number,
            message: message.into(),
        }
    }

    pub fn not_implemented() -> Self {
        Self::alpaca_error(ascom_error::NOT_IMPLEMENTED, "Not implemented")
    }

    pub fn invalid_value(message: &str) -> Self {
        Self::alpaca_error(ascom_error::INVALID_VALUE, message)
    }
}

/// How a device answers a member it does not know.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnknownMember {
    /// HTTP 400 with a plain-text body (older servers; the default).
    Http400,
    Http404,
    /// ASCOM NotImplemented (0x400).
    NotImplemented,
}

/// The form of the `TimeStamp` entry in `devicestate`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeStampStyle {
    /// `...Z`.
    Utc,
    /// An explicit offset, in minutes east of UTC.
    Offset(i32),
    /// No offset, in the host's local time.
    Local,
    /// No TimeStamp entry.
    Omit,
}

// ---------------------------------------------------------------------------
// Scripted devices
// ---------------------------------------------------------------------------

/// The mutable state a scripted device's handlers work on.
#[derive(Debug, Default)]
pub struct DeviceModel {
    /// Member values by lower-case name.
    pub props: BTreeMap<String, Value>,
    /// Members that answer ASCOM ValueNotSet until they are set.
    pub not_set: BTreeSet<String>,
    /// Free counters for handler state machines.
    pub counters: HashMap<String, i64>,
    pub connected: bool,
    /// Platform 7: `connecting` reads left before the device is connected.
    pub connecting_reads_left: u32,
}

impl DeviceModel {
    pub fn set(&mut self, member: &str, value: impl Into<Value>) {
        let member = member.to_ascii_lowercase();
        self.not_set.remove(&member);
        self.props.insert(member, value.into());
    }

    pub fn get(&self, member: &str) -> Option<&Value> {
        self.props.get(&member.to_ascii_lowercase())
    }

    pub fn f64(&self, member: &str) -> f64 {
        self.get(member).and_then(Value::as_f64).unwrap_or_default()
    }

    pub fn bool(&self, member: &str) -> bool {
        self.get(member)
            .and_then(Value::as_bool)
            .unwrap_or_default()
    }

    pub fn counter(&mut self, name: &str) -> &mut i64 {
        self.counters.entry(name.to_string()).or_insert(0)
    }
}

/// Decoded request parameters (GET query or PUT body).
pub type Params = [(String, String)];

/// A stateful member handler.
pub type Handler = Arc<dyn Fn(&mut DeviceModel, &Params) -> ServerReply + Send + Sync>;
/// Runs before every GET (time steps for simple state machines).
pub type GetHook = Arc<dyn Fn(&mut DeviceModel, &str) + Send + Sync>;

/// One device behind the test server.
pub struct ScriptedDevice {
    pub device_type: String,
    pub number: u32,
    pub name: String,
    pub unique_id: Option<String>,
    interface_version: i32,
    platform7: bool,
    device_state: Option<Vec<String>>,
    timestamp: TimeStampStyle,
    require_connected: bool,
    connect_reads: u32,
    unknown: UnknownMember,
    model: DeviceModel,
    queued: HashMap<(Verb, String), VecDeque<ServerReply>>,
    sticky: HashMap<(Verb, String), ServerReply>,
    put_handlers: HashMap<String, Handler>,
    get_handlers: HashMap<String, Handler>,
    get_hook: Option<GetHook>,
    /// A simulator driver answering the device members (O11 l).
    backend: Option<Box<dyn SimBackend>>,
}

/// Members every device answers even while disconnected.
const ALWAYS_AVAILABLE: [&str; 9] = [
    "connected",
    "connecting",
    "connect",
    "disconnect",
    "interfaceversion",
    "name",
    "description",
    "driverinfo",
    "driverversion",
];

impl ScriptedDevice {
    /// A minimal device: connected, no DeviceState, IV 1, unknown members
    /// answered with HTTP 400.
    pub fn new(device_type: &str, number: u32, name: &str) -> Self {
        let mut model = DeviceModel {
            connected: true,
            ..DeviceModel::default()
        };
        model.set("name", name);
        Self {
            device_type: device_type.to_ascii_lowercase(),
            number,
            name: name.into(),
            unique_id: None,
            interface_version: 1,
            platform7: false,
            device_state: None,
            timestamp: TimeStampStyle::Utc,
            require_connected: false,
            connect_reads: 0,
            unknown: UnknownMember::Http400,
            model,
            queued: HashMap::new(),
            sticky: HashMap::new(),
            put_handlers: HashMap::new(),
            get_handlers: HashMap::new(),
            get_hook: None,
            backend: None,
        }
    }

    pub fn unique_id(mut self, unique_id: &str) -> Self {
        self.unique_id = Some(unique_id.into());
        self
    }

    pub fn interface_version(mut self, version: i32) -> Self {
        self.interface_version = version;
        self
    }

    /// Serve Platform 7 `connect`, `connecting` and `disconnect`, starting
    /// disconnected.
    pub fn platform7(mut self, connecting_reads: u32) -> Self {
        self.platform7 = true;
        self.connect_reads = connecting_reads;
        self.model.connected = false;
        self
    }

    /// Start disconnected (a Platform 6 device waiting for `Connected=True`).
    pub fn starts_disconnected(mut self) -> Self {
        self.model.connected = false;
        self
    }

    /// Answer ASCOM NotConnected for device members while disconnected.
    pub fn require_connected(mut self) -> Self {
        self.require_connected = true;
        self
    }

    /// Serve `devicestate` with these members (device spelling).
    pub fn device_state(mut self, members: &[&str]) -> Self {
        self.device_state = Some(members.iter().map(|m| (*m).to_string()).collect());
        self
    }

    pub fn timestamp(mut self, style: TimeStampStyle) -> Self {
        self.timestamp = style;
        self
    }

    pub fn unknown_member(mut self, unknown: UnknownMember) -> Self {
        self.unknown = unknown;
        self
    }

    pub fn property(mut self, member: &str, value: impl Into<Value>) -> Self {
        self.model.set(member, value);
        self
    }

    /// A member that answers ValueNotSet until something sets it.
    pub fn not_set(mut self, member: &str) -> Self {
        self.model.not_set.insert(member.to_ascii_lowercase());
        self
    }

    /// A PUT that stores parameter `key` (exact case) into the member.
    pub fn writable(self, member: &str, key: &'static str) -> Self {
        let target = member.to_ascii_lowercase();
        self.on_put_with(member, move |model, params| {
            match form::find(params, key, true) {
                Some(text) => {
                    let value = parse_param(text);
                    model.set(&target, value);
                    ServerReply::Ok
                }
                None => ServerReply::text(400, &format!("Missing parameter {key}")),
            }
        })
    }

    /// Sticky scripted GET reply (wins over everything else).
    pub fn on_get(mut self, member: &str, reply: ServerReply) -> Self {
        self.sticky
            .insert((Verb::Get, member.to_ascii_lowercase()), reply);
        self
    }

    pub fn on_put(mut self, member: &str, reply: ServerReply) -> Self {
        self.sticky
            .insert((Verb::Put, member.to_ascii_lowercase()), reply);
        self
    }

    /// One-shot GET replies, consumed before anything else.
    pub fn queue_get(
        mut self,
        member: &str,
        replies: impl IntoIterator<Item = ServerReply>,
    ) -> Self {
        self.queued
            .entry((Verb::Get, member.to_ascii_lowercase()))
            .or_default()
            .extend(replies);
        self
    }

    pub fn queue_put(
        mut self,
        member: &str,
        replies: impl IntoIterator<Item = ServerReply>,
    ) -> Self {
        self.queued
            .entry((Verb::Put, member.to_ascii_lowercase()))
            .or_default()
            .extend(replies);
        self
    }

    pub fn on_put_with(
        mut self,
        member: &str,
        handler: impl Fn(&mut DeviceModel, &Params) -> ServerReply + Send + Sync + 'static,
    ) -> Self {
        self.put_handlers
            .insert(member.to_ascii_lowercase(), Arc::new(handler));
        self
    }

    pub fn on_get_with(
        mut self,
        member: &str,
        handler: impl Fn(&mut DeviceModel, &Params) -> ServerReply + Send + Sync + 'static,
    ) -> Self {
        self.get_handlers
            .insert(member.to_ascii_lowercase(), Arc::new(handler));
        self
    }

    pub fn get_hook(
        mut self,
        hook: impl Fn(&mut DeviceModel, &str) + Send + Sync + 'static,
    ) -> Self {
        self.get_hook = Some(Arc::new(hook));
        self
    }

    fn unknown_reply(&self, member: &str) -> ServerReply {
        match self.unknown {
            UnknownMember::Http400 => ServerReply::text(400, &format!("Unknown member '{member}'")),
            UnknownMember::Http404 => ServerReply::text(404, "Not Found"),
            UnknownMember::NotImplemented => ServerReply::not_implemented(),
        }
    }

    fn scripted(&mut self, verb: Verb, member: &str) -> Option<ServerReply> {
        let key = (verb, member.to_string());
        if let Some(reply) = self.queued.get_mut(&key).and_then(VecDeque::pop_front) {
            return Some(reply);
        }
        self.sticky.get(&key).cloned()
    }

    fn gated(&self, member: &str) -> bool {
        self.require_connected && !self.model.connected && !ALWAYS_AVAILABLE.contains(&member)
    }

    fn handle_get(&mut self, member: &str, params: &Params) -> ServerReply {
        if let Some(hook) = self.get_hook.clone() {
            hook(&mut self.model, member);
        }
        if let Some(reply) = self.scripted(Verb::Get, member) {
            return reply;
        }
        match member {
            "connected" => return ServerReply::value(self.model.connected),
            "interfaceversion" => return ServerReply::value(self.interface_version),
            "connecting" if self.platform7 => {
                let busy = self.model.connecting_reads_left > 0;
                if busy {
                    self.model.connecting_reads_left -= 1;
                    if self.model.connecting_reads_left == 0 {
                        self.model.connected = true;
                    }
                }
                return ServerReply::value(busy);
            }
            "supportedactions" if self.model.get(member).is_none() => {
                return ServerReply::value(json!([]))
            }
            "description" | "driverinfo" | "driverversion" if self.model.get(member).is_none() => {
                return ServerReply::value(format!("{} ({member})", self.name))
            }
            // An older server does not know the member at all.
            "devicestate" if self.device_state.is_none() => return self.unknown_reply(member),
            _ => {}
        }
        if self.gated(member) {
            return ServerReply::alpaca_error(ascom_error::NOT_CONNECTED, "Not connected");
        }
        if let Some(reply) = self.backend.as_mut().and_then(|b| b.get(member, params)) {
            return reply;
        }
        if member == "devicestate" {
            if let Some(names) = &self.device_state {
                return ServerReply::value(self.device_state_value(names));
            }
        }
        if let Some(handler) = self.get_handlers.get(member).cloned() {
            return handler(&mut self.model, params);
        }
        if self.model.not_set.contains(member) {
            return ServerReply::alpaca_error(ascom_error::VALUE_NOT_SET, "Value not set");
        }
        match self.model.props.get(member) {
            Some(value) => ServerReply::Value(value.clone()),
            None => self.unknown_reply(member),
        }
    }

    fn handle_put(&mut self, member: &str, params: &Params) -> ServerReply {
        if let Some(reply) = self.scripted(Verb::Put, member) {
            return reply;
        }
        match member {
            "connected" => {
                return match form::find(params, "Connected", true).and_then(form::parse_flag) {
                    Some(on) => {
                        self.model.connected = on;
                        ServerReply::Ok
                    }
                    None => ServerReply::text(400, "Missing or invalid Connected parameter"),
                }
            }
            "connect" if self.platform7 => {
                self.model.connecting_reads_left = self.connect_reads;
                if self.connect_reads == 0 {
                    self.model.connected = true;
                }
                return ServerReply::Ok;
            }
            "disconnect" if self.platform7 => {
                self.model.connected = false;
                self.model.connecting_reads_left = 0;
                return ServerReply::Ok;
            }
            _ => {}
        }
        if self.gated(member) {
            return ServerReply::alpaca_error(ascom_error::NOT_CONNECTED, "Not connected");
        }
        if let Some(handler) = self.put_handlers.get(member).cloned() {
            return handler(&mut self.model, params);
        }
        match self.backend.as_mut().and_then(|b| b.put(member, params)) {
            Some(reply) => reply,
            None => self.unknown_reply(member),
        }
    }

    fn device_state_value(&self, names: &[String]) -> Value {
        let mut entries: Vec<Value> = names
            .iter()
            .filter_map(|name| {
                self.model
                    .get(name)
                    .map(|value| json!({"Name": name, "Value": value}))
            })
            .collect();
        let now = chrono::Utc::now();
        let stamp = match self.timestamp {
            TimeStampStyle::Utc => Some(now.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()),
            TimeStampStyle::Offset(minutes) => {
                chrono::FixedOffset::east_opt(minutes * 60).map(|offset| {
                    now.with_timezone(&offset)
                        .format("%Y-%m-%dT%H:%M:%S%.7f%:z")
                        .to_string()
                })
            }
            TimeStampStyle::Local => Some(
                now.with_timezone(&chrono::Local)
                    .format("%Y-%m-%dT%H:%M:%S%.7f")
                    .to_string(),
            ),
            TimeStampStyle::Omit => None,
        };
        if let Some(stamp) = stamp {
            entries.push(json!({"Name": "TimeStamp", "Value": stamp}));
        }
        Value::Array(entries)
    }

    /// The Alpaca `configureddevices` entry.
    fn configured_entry(&self) -> Value {
        json!({
            "DeviceName": self.name,
            "DeviceType": pascal_device_type(&self.device_type),
            "DeviceNumber": self.number,
            "UniqueID": self.unique_id.clone().unwrap_or_default(),
        })
    }
}

/// A numeric, boolean or text value from a form parameter.
fn parse_param(text: &str) -> Value {
    if let Some(flag) = form::parse_flag(text) {
        return Value::Bool(flag);
    }
    if let Ok(int) = text.parse::<i64>() {
        return Value::from(int);
    }
    if let Ok(float) = text.parse::<f64>() {
        return Value::from(float);
    }
    Value::String(text.into())
}

fn param_f64(params: &Params, key: &str) -> Option<f64> {
    form::find(params, key, true).and_then(|v| v.trim().parse::<f64>().ok())
}

fn param_i64(params: &Params, key: &str) -> Option<i64> {
    form::find(params, key, true).and_then(|v| v.trim().parse::<i64>().ok())
}

fn pascal_device_type(device_type: &str) -> &str {
    match device_type {
        "telescope" => "Telescope",
        "focuser" => "Focuser",
        "filterwheel" => "FilterWheel",
        "rotator" => "Rotator",
        "covercalibrator" => "CoverCalibrator",
        "switch" => "Switch",
        "observingconditions" => "ObservingConditions",
        "safetymonitor" => "SafetyMonitor",
        "dome" => "Dome",
        "camera" => "Camera",
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Profiles
// ---------------------------------------------------------------------------

/// `devicestate` members of an ITelescopeV4.
pub const TELESCOPE_STATE: [&str; 12] = [
    "Altitude",
    "AtHome",
    "AtPark",
    "Azimuth",
    "Declination",
    "IsPulseGuiding",
    "RightAscension",
    "SideOfPier",
    "SiderealTime",
    "Slewing",
    "Tracking",
    "UTCDate",
];

/// Polls of `slewing` (or `devicestate`) a scripted GoTo or park takes.
pub const SCRIPTED_SLEW_READS: i64 = 3;
/// Polls of `ispulseguiding` (or `devicestate`) a scripted pulse takes.
pub const SCRIPTED_PULSE_READS: i64 = 2;
/// Polls of `ismoving` (or `position`) a scripted focuser move takes.
pub const SCRIPTED_MOVE_READS: i64 = 3;

impl ScriptedDevice {
    /// A telescope shaped like the user's ESP32 mount (spec §9.1 item 3),
    /// built only from standard Alpaca behaviour: InterfaceVersion 4 with
    /// Connect/Connecting and DeviceState, AlignmentMode Polar (1),
    /// EquatorialSystem Topocentric (1), a constant SideOfPier of 0, CanPark
    /// true, CanFindHome false, CanSetPierSide false, CanSetGuideRates true,
    /// TrackingRates [0,1,2,3], and per O10 CanSlew false with CanSlewAsync
    /// true. GoTo and park complete after [`SCRIPTED_SLEW_READS`] polls; a
    /// pulse after [`SCRIPTED_PULSE_READS`]. It starts parked at the pole.
    pub fn esp32_like_mount(number: u32) -> Self {
        let sidereal_deg_s = 15.041_067 / 3600.0;
        ScriptedDevice::new("telescope", number, "ESP32-like mount (test profile)")
            .interface_version(4)
            .platform7(1)
            .require_connected()
            .device_state(&TELESCOPE_STATE)
            .timestamp(TimeStampStyle::Utc)
            .unique_id("test-esp32-mount-0001")
            .property("description", "Scripted ITelescopeV4 for CI")
            .property("driverinfo", "Achernar test server")
            .property("driverversion", "1.0")
            .property("alignmentmode", 1)
            .property("equatorialsystem", 1)
            .property("doesrefraction", false)
            .property("canpark", true)
            .property("canunpark", true)
            .property("cansetpark", false)
            .property("canfindhome", false)
            .property("cansetpierside", false)
            .property("cansetguiderates", true)
            .property("cansettracking", true)
            .property("canslew", false)
            .property("canslewasync", true)
            .property("canslewaltaz", false)
            .property("canslewaltazasync", false)
            .property("cansync", true)
            .property("cansyncaltaz", false)
            .property("canpulseguide", true)
            .property("cansetdeclinationrate", false)
            .property("cansetrightascensionrate", false)
            .property("trackingrates", json!([0, 1, 2, 3]))
            .property("trackingrate", 0)
            .property("tracking", false)
            .property("guideraterightascension", 0.5 * sidereal_deg_s)
            .property("guideratedeclination", 0.5 * sidereal_deg_s)
            .property("rightascension", 0.0)
            .property("declination", 90.0)
            .property("altitude", 40.0)
            .property("azimuth", 0.0)
            .property("siderealtime", 0.0)
            .property("slewing", false)
            .property("atpark", true)
            .property("athome", false)
            .property("sideofpier", 0)
            .property("ispulseguiding", false)
            .property("utcdate", "2026-10-26T21:30:00.000Z")
            .property("sitelatitude", 40.0)
            .property("sitelongitude", -75.0)
            .property("siteelevation", 100.0)
            .property("focallength", 0.4)
            .property("aperturediameter", 0.08)
            .not_set("targetrightascension")
            .not_set("targetdeclination")
            .writable("tracking", "Tracking")
            .writable("trackingrate", "TrackingRate")
            .writable("guideraterightascension", "GuideRateRightAscension")
            .writable("guideratedeclination", "GuideRateDeclination")
            .writable("utcdate", "UTCDate")
            .writable("sitelatitude", "SiteLatitude")
            .writable("sitelongitude", "SiteLongitude")
            .writable("siteelevation", "SiteElevation")
            .on_put("sideofpier", ServerReply::not_implemented())
            .on_put("findhome", ServerReply::not_implemented())
            .on_put("setpark", ServerReply::not_implemented())
            .on_put("slewtocoordinates", ServerReply::not_implemented())
            .on_get_with("canmoveaxis", |_, params| {
                match form::find(params, "Axis", false).and_then(|a| a.parse::<i64>().ok()) {
                    Some(0 | 1) => ServerReply::value(true),
                    Some(_) => ServerReply::value(false),
                    None => ServerReply::invalid_value("Axis is required"),
                }
            })
            .on_get_with("axisrates", |_, _| {
                ServerReply::value(json!([{"Minimum": 0.0, "Maximum": 4.0}]))
            })
            .on_put_with("slewtocoordinatesasync", |model, params| {
                if model.bool("atpark") {
                    return ServerReply::alpaca_error(ascom_error::INVALID_WHILE_PARKED, "Parked");
                }
                let (Some(ra), Some(dec)) = (
                    param_f64(params, "RightAscension"),
                    param_f64(params, "Declination"),
                ) else {
                    return ServerReply::invalid_value(
                        "RightAscension and Declination are required",
                    );
                };
                if !(0.0..24.0).contains(&ra) || !(-90.0..=90.0).contains(&dec) {
                    return ServerReply::invalid_value("Target out of range");
                }
                model.set("targetrightascension", ra);
                model.set("targetdeclination", dec);
                model.set("slewing", true);
                *model.counter("slew") = SCRIPTED_SLEW_READS;
                ServerReply::Ok
            })
            .on_put_with("synctocoordinates", |model, params| {
                match (
                    param_f64(params, "RightAscension"),
                    param_f64(params, "Declination"),
                ) {
                    (Some(ra), Some(dec)) => {
                        model.set("rightascension", ra);
                        model.set("declination", dec);
                        ServerReply::Ok
                    }
                    _ => ServerReply::invalid_value("RightAscension and Declination are required"),
                }
            })
            .on_put_with("abortslew", |model, _| {
                *model.counter("slew") = 0;
                *model.counter("park") = 0;
                model.set("slewing", false);
                ServerReply::Ok
            })
            .on_put_with("park", |model, _| {
                if model.bool("atpark") {
                    return ServerReply::Ok;
                }
                model.set("slewing", true);
                model.set("tracking", false);
                *model.counter("park") = SCRIPTED_SLEW_READS;
                ServerReply::Ok
            })
            .on_put_with("unpark", |model, _| {
                model.set("atpark", false);
                ServerReply::Ok
            })
            .on_put_with("pulseguide", |model, params| {
                match (
                    param_i64(params, "Direction"),
                    param_i64(params, "Duration"),
                ) {
                    (Some(0..=3), Some(duration)) if duration >= 0 => {
                        model.set("ispulseguiding", duration > 0);
                        *model.counter("pulse") = if duration > 0 {
                            SCRIPTED_PULSE_READS
                        } else {
                            0
                        };
                        ServerReply::Ok
                    }
                    _ => {
                        ServerReply::invalid_value("Direction 0..3 and Duration >= 0 are required")
                    }
                }
            })
            .on_put_with("moveaxis", |model, params| {
                match (param_i64(params, "Axis"), param_f64(params, "Rate")) {
                    (Some(0 | 1), Some(rate)) if rate.abs() <= 4.0 => {
                        let goto_active = *model.counter("slew") > 0;
                        model.set("slewing", rate != 0.0 || goto_active);
                        ServerReply::Ok
                    }
                    _ => ServerReply::invalid_value("Axis 0 or 1 and |Rate| <= 4 are required"),
                }
            })
            .get_hook(|model, member| {
                if matches!(
                    member,
                    "slewing" | "devicestate" | "atpark" | "rightascension" | "declination"
                ) {
                    let slew = *model.counter("slew");
                    if slew > 0 {
                        *model.counter("slew") = slew - 1;
                        if slew == 1 {
                            let ra = model.f64("targetrightascension");
                            let dec = model.f64("targetdeclination");
                            model.set("rightascension", ra);
                            model.set("declination", dec);
                            model.set("slewing", false);
                        }
                    }
                    let park = *model.counter("park");
                    if park > 0 {
                        *model.counter("park") = park - 1;
                        if park == 1 {
                            model.set("rightascension", 0.0);
                            model.set("declination", 90.0);
                            model.set("slewing", false);
                            model.set("atpark", true);
                        }
                    }
                }
                if matches!(member, "ispulseguiding" | "devicestate") {
                    let pulse = *model.counter("pulse");
                    if pulse > 0 {
                        *model.counter("pulse") = pulse - 1;
                        if pulse == 1 {
                            model.set("ispulseguiding", false);
                        }
                    }
                }
            })
    }

    /// A focuser shaped like the user's planned ESP32 focuser (spec §8 item
    /// 1): InterfaceVersion 3, so no Connect/Connecting and no DeviceState
    /// (both answer HTTP 400 like an older server); absolute, MaxStep 20000,
    /// a temperature probe, TempComp unavailable. A move completes after
    /// [`SCRIPTED_MOVE_READS`] polls of `ismoving`.
    pub fn platform6_focuser(number: u32) -> Self {
        ScriptedDevice::new("focuser", number, "Platform 6 focuser (test profile)")
            .interface_version(3)
            .starts_disconnected()
            .require_connected()
            .property("description", "Scripted IFocuserV3 for CI")
            .property("driverinfo", "Achernar test server")
            .property("driverversion", "1.0")
            .property("absolute", true)
            .property("maxstep", 20_000)
            .property("maxincrement", 20_000)
            .property("stepsize", 2.5)
            .property("position", 10_000)
            .property("ismoving", false)
            .property("temperature", 12.5)
            .property("tempcompavailable", false)
            .property("tempcomp", false)
            .on_put("tempcomp", ServerReply::not_implemented())
            .on_put_with("move", |model, params| {
                let Some(target) = param_i64(params, "Position") else {
                    return ServerReply::invalid_value("Position is required");
                };
                let max = model.get("maxstep").and_then(Value::as_i64).unwrap_or(0);
                if !(0..=max).contains(&target) {
                    return ServerReply::invalid_value("Position outside 0..MaxStep");
                }
                model.set("target", target);
                model.set("ismoving", true);
                *model.counter("move") = SCRIPTED_MOVE_READS;
                ServerReply::Ok
            })
            .on_put_with("halt", |model, _| {
                *model.counter("move") = 0;
                model.set("ismoving", false);
                ServerReply::Ok
            })
            .get_hook(|model, member| {
                if member != "ismoving" {
                    return;
                }
                let left = *model.counter("move");
                if left > 0 {
                    *model.counter("move") = left - 1;
                    if left == 1 {
                        let target = model.get("target").cloned().unwrap_or(Value::Null);
                        model.set("position", target);
                        model.set("ismoving", false);
                    }
                }
            })
    }
}

// ---------------------------------------------------------------------------
// Simulator drivers behind Alpaca (O11 l)
// ---------------------------------------------------------------------------

/// A simulator driver behind a [`ScriptedDevice`] (O11 l). It answers the
/// device's own members; the scripted device keeps the connect members, the
/// identity strings, `connected` gating and scripted replies, which are
/// consulted first so a test can still inject faults and delays. `None`
/// means "not a member of this device" (the device's unknown-member reply).
pub trait SimBackend: Send {
    fn get(&mut self, member: &str, params: &Params) -> Option<ServerReply>;
    fn put(&mut self, member: &str, params: &Params) -> Option<ServerReply>;
}

/// The ASCOM error reply for a driver error: its own ErrorNumber when it
/// came with one, else the number of its class.
fn error_reply(err: &DeviceError) -> ServerReply {
    let number = err.alpaca_error_number().unwrap_or(match err.class {
        ErrorClass::Unsupported => ascom_error::NOT_IMPLEMENTED,
        ErrorClass::InvalidValue => ascom_error::INVALID_VALUE,
        ErrorClass::NotConnected | ErrorClass::Disconnected => ascom_error::NOT_CONNECTED,
        ErrorClass::InvalidWhileParked => ascom_error::INVALID_WHILE_PARKED,
        ErrorClass::InvalidOperation => ascom_error::INVALID_OPERATION,
        ErrorClass::Cancelled => ascom_error::OPERATION_CANCELLED,
        _ => ascom_error::DRIVER_BASE,
    });
    ServerReply::alpaca_error(number, &err.message)
}

fn void_reply(result: Result<(), DeviceError>) -> ServerReply {
    match result {
        Ok(()) => ServerReply::Ok,
        Err(err) => error_reply(&err),
    }
}

/// A reading as an Alpaca reply: its value, or NotImplemented, ValueNotSet
/// or a driver error. Never a synthesised value (D5).
fn observed_reply<T>(observed: &Observed<T>, to_json: impl Fn(&T) -> Value) -> ServerReply {
    match (&observed.value, observed.state) {
        (_, ObsState::Unsupported) => ServerReply::not_implemented(),
        (_, ObsState::NotSet) => {
            ServerReply::alpaca_error(ascom_error::VALUE_NOT_SET, "Value not set")
        }
        (Some(value), _) => ServerReply::Value(to_json(value)),
        (None, _) => ServerReply::alpaca_error(ascom_error::DRIVER_BASE, "reading unavailable"),
    }
}

fn missing(key: &str) -> ServerReply {
    ServerReply::invalid_value(&format!("{key} is required"))
}

/// Alpaca's UTCDate form (ISO 8601, `Z`).
fn alpaca_time(at: chrono::DateTime<chrono::Utc>) -> String {
    at.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()
}

fn alignment_ascom(mode: AlignmentMode) -> i64 {
    match mode {
        AlignmentMode::AltAz => 0,
        AlignmentMode::Polar => 1,
        AlignmentMode::GermanPolar => 2,
    }
}

/// A `devicestate` entry list plus the TimeStamp entry.
fn device_state_entries(entries: Vec<(&str, Option<Value>)>) -> Value {
    let mut list: Vec<Value> = entries
        .into_iter()
        .filter_map(|(name, value)| value.map(|value| json!({"Name": name, "Value": value})))
        .collect();
    list.push(json!({"Name": "TimeStamp", "Value": alpaca_time(chrono::Utc::now())}));
    Value::Array(list)
}

/// The telescope members the simulator's poll answers.
const SIM_MOUNT_STATE: [&str; 21] = [
    "rightascension",
    "declination",
    "altitude",
    "azimuth",
    "siderealtime",
    "utcdate",
    "slewing",
    "tracking",
    "trackingrate",
    "atpark",
    "athome",
    "sideofpier",
    "ispulseguiding",
    "guideraterightascension",
    "guideratedeclination",
    "sitelatitude",
    "sitelongitude",
    "siteelevation",
    "targetrightascension",
    "targetdeclination",
    "devicestate",
];

/// The WP1A-4 kinematic simulator mount as an ITelescope (O11 l). Every
/// member is a call on the simulator driver, so GoTo, park, pulses, jog and
/// tracking move the simulated axes on its clock. Per O10 the synchronous
/// slews answer NotImplemented.
pub struct SimMountBackend {
    sim: SimMount,
    caps: MountCapabilities,
}

impl SimMountBackend {
    fn state(&mut self) -> Result<MountState, ServerReply> {
        self.sim
            .poll(&PollPlan::default())
            .map_err(|err| error_reply(&err))?;
        self.sim.state().map_err(|err| error_reply(&err))
    }

    fn coord(&self, params: &Params) -> Option<EqCoord> {
        Some(EqCoord::new(
            param_f64(params, "RightAscension")?,
            param_f64(params, "Declination")?,
            self.caps.native_frame,
        ))
    }

    fn capability(&self, member: &str, params: &Params) -> Option<ServerReply> {
        let caps = &self.caps;
        let flag = |on: bool| Some(ServerReply::value(on));
        match member {
            "canpark" => flag(caps.can_park),
            "canunpark" => flag(caps.can_unpark),
            "cansetpark" => flag(caps.can_set_park),
            "canfindhome" => flag(caps.can_find_home),
            "canslew" => flag(caps.can_slew),
            "canslewasync" => flag(caps.can_slew_async),
            "canslewaltaz" => flag(caps.can_slew_altaz),
            "canslewaltazasync" => flag(caps.can_slew_altaz_async),
            "cansync" => flag(caps.can_sync),
            "cansyncaltaz" => flag(caps.can_sync_altaz),
            "cansettracking" => flag(caps.can_set_tracking),
            "cansetpierside" => flag(caps.can_set_pier_side),
            "canpulseguide" => flag(caps.can_pulse_guide),
            "cansetguiderates" => flag(caps.can_set_guide_rates),
            "cansetdeclinationrate" => flag(caps.can_set_declination_rate),
            "cansetrightascensionrate" => flag(caps.can_set_right_ascension_rate),
            "canmoveaxis" => Some(match param_i64(params, "Axis") {
                Some(axis @ 0..=2) => ServerReply::value(caps.can_move_axis[axis as usize]),
                Some(_) => ServerReply::invalid_value("Axis must be 0, 1 or 2"),
                None => missing("Axis"),
            }),
            "axisrates" => Some(match param_i64(params, "Axis") {
                Some(axis @ 0..=2) => {
                    let ranges: Vec<Value> = caps
                        .axis_rates
                        .iter()
                        .filter(|rates| i64::from(rates.axis) == axis)
                        .flat_map(|rates| rates.ranges.iter())
                        .map(|r| json!({"Minimum": r.min_deg_s, "Maximum": r.max_deg_s}))
                        .collect();
                    ServerReply::value(Value::Array(ranges))
                }
                Some(_) => ServerReply::invalid_value("Axis must be 0, 1 or 2"),
                None => missing("Axis"),
            }),
            "alignmentmode" => Some(match caps.alignment_mode {
                Some(mode) => ServerReply::value(alignment_ascom(mode)),
                None => ServerReply::not_implemented(),
            }),
            "equatorialsystem" => Some(match caps.equatorial_system {
                Some(system) => ServerReply::value(system),
                None => ServerReply::not_implemented(),
            }),
            "doesrefraction" => Some(match caps.does_refraction {
                Some(on) => ServerReply::value(on),
                None => ServerReply::not_implemented(),
            }),
            "trackingrates" => Some(ServerReply::value(Value::Array(
                caps.tracking_rates
                    .iter()
                    .map(|rate| json!(rate.to_ascom()))
                    .collect(),
            ))),
            "focallength" | "aperturediameter" => {
                let mm = if member == "focallength" {
                    caps.focal_length_mm
                } else {
                    caps.aperture_diameter_mm
                };
                Some(match mm {
                    Some(mm) => ServerReply::value(mm / 1000.0),
                    None => ServerReply::not_implemented(),
                })
            }
            _ => None,
        }
    }

    fn reading(&mut self, member: &str) -> Option<ServerReply> {
        if !SIM_MOUNT_STATE.contains(&member) {
            return None;
        }
        if member == "devicestate" && !self.caps.device_state {
            return None;
        }
        let state = match self.state() {
            Ok(state) => state,
            Err(reply) => return Some(reply),
        };
        let num = |v: &f64| json!(v);
        let flag = |v: &bool| json!(v);
        let site = |field: fn(&Site) -> f64| observed_reply(&state.site, move |s| json!(field(s)));
        Some(match member {
            "rightascension" => observed_reply(&state.ra_h, num),
            "declination" => observed_reply(&state.dec_deg, num),
            "altitude" => observed_reply(&state.alt_deg, num),
            "azimuth" => observed_reply(&state.az_deg, num),
            "siderealtime" => observed_reply(&state.sidereal_time_h, num),
            "utcdate" => observed_reply(&state.utc, |t| json!(alpaca_time(*t))),
            "slewing" => observed_reply(&state.slewing, flag),
            "tracking" => observed_reply(&state.tracking, flag),
            "trackingrate" => observed_reply(&state.tracking_rate, |r| json!(r.to_ascom())),
            "atpark" => observed_reply(&state.at_park, flag),
            "athome" => observed_reply(&state.at_home, flag),
            "sideofpier" => observed_reply(&state.side_of_pier, |s| json!(s.to_ascom())),
            "ispulseguiding" => observed_reply(&state.is_pulse_guiding, flag),
            "guideraterightascension" => {
                observed_reply(&state.guide_rate_ra_arcsec_s, |v| json!(v / 3600.0))
            }
            "guideratedeclination" => {
                observed_reply(&state.guide_rate_dec_arcsec_s, |v| json!(v / 3600.0))
            }
            "sitelatitude" => site(|s| s.lat_deg),
            "sitelongitude" => site(|s| s.lon_deg),
            "siteelevation" => site(|s| s.elev_m),
            "targetrightascension" => observed_reply(&state.target, |c| json!(c.ra_h)),
            "targetdeclination" => observed_reply(&state.target, |c| json!(c.dec_deg)),
            _ => ServerReply::value(device_state_entries(vec![
                ("Altitude", state.alt_deg.value.map(|v| json!(v))),
                ("AtHome", state.at_home.value.map(|v| json!(v))),
                ("AtPark", state.at_park.value.map(|v| json!(v))),
                ("Azimuth", state.az_deg.value.map(|v| json!(v))),
                ("Declination", state.dec_deg.value.map(|v| json!(v))),
                (
                    "IsPulseGuiding",
                    state.is_pulse_guiding.value.map(|v| json!(v)),
                ),
                ("RightAscension", state.ra_h.value.map(|v| json!(v))),
                (
                    "SideOfPier",
                    state.side_of_pier.value.map(|s| json!(s.to_ascom())),
                ),
                (
                    "SiderealTime",
                    state.sidereal_time_h.value.map(|v| json!(v)),
                ),
                ("Slewing", state.slewing.value.map(|v| json!(v))),
                ("Tracking", state.tracking.value.map(|v| json!(v))),
                ("UTCDate", state.utc.value.map(|t| json!(alpaca_time(t)))),
            ])),
        })
    }

    /// A site member write: the other two values from the simulator's site.
    fn set_site(&mut self, member: &str, value: f64) -> ServerReply {
        let current = match self.state() {
            Ok(state) => state.site.value,
            Err(reply) => return reply,
        };
        let Some(mut site) = current else {
            return ServerReply::not_implemented();
        };
        match member {
            "sitelatitude" => site.lat_deg = value,
            "sitelongitude" => site.lon_deg = value,
            _ => site.elev_m = value,
        }
        site.source = SiteSource::Mount;
        void_reply(self.sim.set_site(&site))
    }

    /// One guide-rate write: the other axis keeps the simulator's rate.
    fn set_guide_rate(&mut self, ra: bool, deg_s: f64) -> ServerReply {
        let state = match self.state() {
            Ok(state) => state,
            Err(reply) => return reply,
        };
        let other = if ra {
            state.guide_rate_dec_arcsec_s.value
        } else {
            state.guide_rate_ra_arcsec_s.value
        }
        .map(|arcsec| arcsec / 3600.0)
        .unwrap_or(deg_s);
        let (ra_deg_s, dec_deg_s) = if ra { (deg_s, other) } else { (other, deg_s) };
        void_reply(self.sim.set_guide_rates(ra_deg_s, dec_deg_s))
    }
}

impl SimBackend for SimMountBackend {
    fn get(&mut self, member: &str, params: &Params) -> Option<ServerReply> {
        if let Some(reply) = self.capability(member, params) {
            return Some(reply);
        }
        if member == "destinationsideofpier" {
            let Some(target) = self.coord(params) else {
                return Some(missing("RightAscension and Declination"));
            };
            return Some(match self.sim.destination_side_of_pier(target) {
                Ok(side) => ServerReply::value(side.to_ascom()),
                Err(err) => error_reply(&err),
            });
        }
        self.reading(member)
    }

    fn put(&mut self, member: &str, params: &Params) -> Option<ServerReply> {
        let sim = &mut self.sim;
        Some(match member {
            // O10: Alpaca devices raise MethodNotImplemented for the
            // synchronous slews.
            "slewtocoordinates" | "slewtoaltaz" | "slewtotarget" => ServerReply::not_implemented(),
            "slewtocoordinatesasync" => match self.coord(params) {
                Some(target) => void_reply(self.sim.slew_async(target)),
                None => missing("RightAscension and Declination"),
            },
            "synctocoordinates" => match self.coord(params) {
                Some(target) => void_reply(self.sim.sync(target)),
                None => missing("RightAscension and Declination"),
            },
            "slewtoaltazasync" => {
                match (param_f64(params, "Azimuth"), param_f64(params, "Altitude")) {
                    (Some(az_deg), Some(alt_deg)) => {
                        void_reply(sim.slew_altaz_async(AltAz { alt_deg, az_deg }))
                    }
                    _ => missing("Azimuth and Altitude"),
                }
            }
            "abortslew" => void_reply(sim.abort_slew()),
            "park" => void_reply(sim.park()),
            "unpark" => void_reply(sim.unpark()),
            "setpark" => void_reply(sim.set_park()),
            "findhome" => void_reply(sim.find_home()),
            "tracking" => match form::find(params, "Tracking", true).and_then(form::parse_flag) {
                Some(on) => void_reply(sim.set_tracking(on)),
                None => missing("Tracking"),
            },
            "trackingrate" => {
                match param_i64(params, "TrackingRate").and_then(DriveRate::from_ascom) {
                    Some(rate) => void_reply(sim.set_tracking_rate(rate)),
                    None => ServerReply::invalid_value("TrackingRate must be 0..3"),
                }
            }
            "moveaxis" => match (param_i64(params, "Axis"), param_f64(params, "Rate")) {
                (Some(axis @ 0..=2), Some(rate)) => void_reply(sim.move_axis(axis as u8, rate)),
                _ => missing("Axis (0..2) and Rate"),
            },
            "pulseguide" => match (
                param_i64(params, "Direction").and_then(GuideDirection::from_ascom),
                param_i64(params, "Duration").and_then(|d| u32::try_from(d).ok()),
            ) {
                (Some(direction), Some(ms)) => void_reply(sim.pulse_guide(direction, ms)),
                _ => missing("Direction (0..3) and Duration (ms >= 0)"),
            },
            "guideraterightascension" => match param_f64(params, "GuideRateRightAscension") {
                Some(rate) => self.set_guide_rate(true, rate),
                None => missing("GuideRateRightAscension"),
            },
            "guideratedeclination" => match param_f64(params, "GuideRateDeclination") {
                Some(rate) => self.set_guide_rate(false, rate),
                None => missing("GuideRateDeclination"),
            },
            "sitelatitude" | "sitelongitude" | "siteelevation" => {
                let key = match member {
                    "sitelatitude" => "SiteLatitude",
                    "sitelongitude" => "SiteLongitude",
                    _ => "SiteElevation",
                };
                match param_f64(params, key) {
                    Some(value) => self.set_site(member, value),
                    None => missing(key),
                }
            }
            "utcdate" => match form::find(params, "UTCDate", true)
                .and_then(|text| chrono::DateTime::parse_from_rfc3339(text.trim()).ok())
            {
                Some(at) => void_reply(sim.set_utc(at.with_timezone(&chrono::Utc))),
                None => ServerReply::invalid_value("UTCDate must be an ISO 8601 time"),
            },
            "sideofpier" => match param_i64(params, "SideOfPier") {
                Some(side @ 0..=1) => void_reply(sim.set_side_of_pier(PierSide::from_ascom(side))),
                _ => ServerReply::invalid_value("SideOfPier must be 0 or 1"),
            },
            "action" => {
                let name = form::find(params, "Action", true).unwrap_or_default();
                let parameters = form::find(params, "Parameters", true).unwrap_or_default();
                match sim.action(name, parameters) {
                    Ok(text) => ServerReply::value(text),
                    Err(err) => error_reply(&err),
                }
            }
            _ => return None,
        })
    }
}

/// The WP1A-5 simulator focuser as an IFocuser (O11 l): moves travel at the
/// simulator's speed on its clock, `ismoving` follows them, and its
/// temperature probe answers `temperature`.
pub struct SimFocuserBackend {
    sim: SimFocuser,
    caps: FocuserCapabilities,
}

impl SimFocuserBackend {
    fn state(&mut self) -> Result<FocuserState, ServerReply> {
        let plan = PollPlan {
            prefer_device_state: self.caps.device_state,
            ..PollPlan::default()
        };
        self.sim.poll(&plan).map_err(|err| error_reply(&err))?;
        self.sim.state().map_err(|err| error_reply(&err))
    }
}

impl SimBackend for SimFocuserBackend {
    fn get(&mut self, member: &str, _params: &Params) -> Option<ServerReply> {
        let caps = &self.caps;
        let optional = |value: Option<Value>| match value {
            Some(value) => ServerReply::value(value),
            None => ServerReply::not_implemented(),
        };
        match member {
            "absolute" => return Some(ServerReply::value(caps.absolute)),
            "maxstep" => return Some(optional(caps.max_step.map(Value::from))),
            "maxincrement" => return Some(optional(caps.max_increment.map(Value::from))),
            "stepsize" => return Some(optional(caps.step_size_um.map(Value::from))),
            "tempcompavailable" => return Some(ServerReply::value(caps.temp_comp_available)),
            "position" | "ismoving" | "temperature" | "tempcomp" => {}
            "devicestate" if caps.device_state => {}
            _ => return None,
        }
        let state = match self.state() {
            Ok(state) => state,
            Err(reply) => return Some(reply),
        };
        Some(match member {
            "position" => observed_reply(&state.position, |p| json!(p)),
            "ismoving" => observed_reply(&state.is_moving, |m| json!(m)),
            "temperature" => observed_reply(&state.temperature_c, |t| json!(t)),
            // ASCOM: TempComp reads false when compensation is unavailable.
            "tempcomp" if !self.caps.temp_comp_available => ServerReply::value(false),
            "tempcomp" => observed_reply(&state.temp_comp, |on| json!(on)),
            _ => ServerReply::value(device_state_entries(vec![
                ("IsMoving", state.is_moving.value.map(|v| json!(v))),
                ("Position", state.position.value.map(|v| json!(v))),
                ("Temperature", state.temperature_c.value.map(|v| json!(v))),
            ])),
        })
    }

    fn put(&mut self, member: &str, params: &Params) -> Option<ServerReply> {
        let sim = &mut self.sim;
        Some(match member {
            "move" => match param_i64(params, "Position").and_then(|p| i32::try_from(p).ok()) {
                Some(position) if self.caps.absolute => void_reply(sim.move_absolute(position)),
                Some(steps) => void_reply(sim.move_relative(steps)),
                None => missing("Position"),
            },
            "halt" => void_reply(sim.halt()),
            "tempcomp" => match form::find(params, "TempComp", true).and_then(form::parse_flag) {
                Some(on) => void_reply(sim.set_temp_comp(on)),
                None => missing("TempComp"),
            },
            "action" => {
                let name = form::find(params, "Action", true).unwrap_or_default();
                let parameters = form::find(params, "Parameters", true).unwrap_or_default();
                match sim.action(name, parameters) {
                    Ok(text) => ServerReply::value(text),
                    Err(err) => error_reply(&err),
                }
            }
            _ => return None,
        })
    }
}

impl ScriptedDevice {
    /// A device whose members come from `backend`, with the connect path of
    /// its interface version: Platform 7 `connect`/`connecting` at or above
    /// `platform7_version` (one `connecting` read), else Platform 6
    /// `Connected=true`; members answer NotConnected until connected.
    fn simulated(
        device_type: &str,
        number: u32,
        identity: &crate::equipment::types::DeviceIdentity,
        platform7_version: i32,
        device_state: bool,
        backend: Box<dyn SimBackend>,
    ) -> Self {
        let version = identity.interface_version.unwrap_or(1);
        let mut device = ScriptedDevice::new(device_type, number, &identity.name)
            .interface_version(version)
            .require_connected()
            .property("driverversion", "sim");
        if let Some(description) = &identity.description {
            device = device.property("description", description.as_str());
        }
        if let Some(info) = &identity.driver_info {
            device = device.property("driverinfo", info.as_str());
        }
        device = if version >= platform7_version {
            device.platform7(1)
        } else {
            device.starts_disconnected()
        };
        if device_state {
            // Served by the backend; an empty list only marks it present.
            device.device_state = Some(Vec::new());
        }
        device.backend = Some(backend);
        device
    }

    /// O11 (l): the WP1A-4 kinematic simulator mount behind Alpaca as
    /// telescope `number`, with the interface version, DeviceState and
    /// capabilities of its profile (for example
    /// [`SimMountProfile::esp32_like`](crate::equipment::sim::mount::SimMountProfile::esp32_like)).
    /// Keep [`SimMount::control`] before handing the simulator over to
    /// inject faults while the server runs.
    pub fn sim_mount(number: u32, mut sim: SimMount) -> Self {
        // The simulator itself stays connected; the Alpaca connection state
        // is the scripted device's.
        let _ = sim.connect(Duration::ZERO);
        let caps = sim.capabilities().unwrap_or_default();
        let identity = sim.identity().clone();
        let device_state = caps.device_state;
        Self::simulated(
            "telescope",
            number,
            &identity,
            4,
            device_state,
            Box::new(SimMountBackend { sim, caps }),
        )
    }

    /// O11 (l): the WP1A-5 simulator focuser behind Alpaca as focuser
    /// `number` (for example
    /// [`SimFocuserConfig::platform6_firmware_backlash`](crate::equipment::sim::focuser::SimFocuserConfig::platform6_firmware_backlash):
    /// InterfaceVersion 3, no DeviceState, absolute, a temperature probe).
    /// Firmware-owned backlash is not an Alpaca member: the daemon learns it
    /// from the identity profile (D17).
    pub fn sim_focuser(number: u32, mut sim: SimFocuser) -> Self {
        let _ = sim.connect(Duration::ZERO);
        let caps = sim.capabilities().unwrap_or_default();
        let identity = sim.identity().clone();
        let device_state = caps.device_state && caps.interface_version.unwrap_or(1) >= 4;
        Self::simulated(
            "focuser",
            number,
            &identity,
            4,
            device_state,
            Box::new(SimFocuserBackend { sim, caps }),
        )
    }
}

// ---------------------------------------------------------------------------
// The server
// ---------------------------------------------------------------------------

/// One request as the server received it.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordedRequest {
    pub verb: Verb,
    pub device_type: String,
    pub number: u32,
    /// Lower-case member, or the management path for management requests.
    pub member: String,
    pub params: Vec<(String, String)>,
    pub content_type: Option<String>,
    pub client_id: Option<u32>,
    pub client_transaction_id: Option<u32>,
}

struct DeviceSlot {
    device_type: String,
    number: u32,
    device: Mutex<ScriptedDevice>,
}

struct ServerState {
    devices: Vec<DeviceSlot>,
    log: Mutex<Vec<RecordedRequest>>,
    server_transaction: AtomicU32,
}

impl ServerState {
    fn record(&self, request: RecordedRequest) {
        lock(&self.log).push(request);
    }

    fn next_server_transaction(&self) -> u32 {
        self.server_transaction.fetch_add(1, Ordering::Relaxed)
    }

    fn device(&self, device_type: &str, number: u32) -> Option<&Mutex<ScriptedDevice>> {
        self.devices
            .iter()
            .find(|slot| slot.device_type == device_type && slot.number == number)
            .map(|slot| &slot.device)
    }
}

/// An Alpaca server on `127.0.0.1:0`, running on its own thread and tokio
/// runtime. Dropping it stops the server.
pub struct TestAlpacaServer {
    addr: SocketAddr,
    state: Arc<ServerState>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl TestAlpacaServer {
    pub fn start(devices: Vec<ScriptedDevice>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind the test Alpaca server");
        listener
            .set_nonblocking(true)
            .expect("non-blocking test listener");
        let addr = listener.local_addr().expect("test server address");
        let state = Arc::new(ServerState {
            devices: devices
                .into_iter()
                .map(|device| DeviceSlot {
                    device_type: device.device_type.clone(),
                    number: device.number,
                    device: Mutex::new(device),
                })
                .collect(),
            log: Mutex::new(Vec::new()),
            server_transaction: AtomicU32::new(1),
        });
        let router = Router::new()
            .route(
                "/api/v1/:device_type/:number/:member",
                get(device_get).put(device_put),
            )
            .route("/management/apiversions", get(management_versions))
            .route("/management/v1/description", get(management_description))
            .route(
                "/management/v1/configureddevices",
                get(management_configured),
            )
            .route("/setup/v1/:device_type/:number/setup", get(setup_page))
            .fallback(|| async { (StatusCode::NOT_FOUND, "Not Found") })
            .with_state(Arc::clone(&state));
        let (shutdown, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let thread = std::thread::Builder::new()
            .name("test-alpaca-server".into())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("test server runtime");
                runtime.block_on(async move {
                    let listener =
                        tokio::net::TcpListener::from_std(listener).expect("tokio listener");
                    let serve = axum::serve(listener, router).into_future();
                    tokio::select! {
                        _ = serve => {}
                        _ = shutdown_rx => {}
                    }
                });
            })
            .expect("spawn the test Alpaca server");
        Self {
            addr,
            state,
            shutdown: Some(shutdown),
            thread: Some(thread),
        }
    }

    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// `http://127.0.0.1:{port}`.
    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn requests(&self) -> Vec<RecordedRequest> {
        lock(&self.state.log).clone()
    }

    pub fn clear_requests(&self) {
        lock(&self.state.log).clear();
    }

    /// Inspect or change a device's state while the server runs.
    pub fn with_device<R>(
        &self,
        device_type: &str,
        number: u32,
        f: impl FnOnce(&mut DeviceModel) -> R,
    ) -> Option<R> {
        self.state
            .device(device_type, number)
            .map(|device| f(&mut lock(device).model))
    }

    /// Script further replies while the server runs.
    pub fn queue_reply(
        &self,
        device_type: &str,
        number: u32,
        verb: Verb,
        member: &str,
        reply: ServerReply,
    ) {
        if let Some(device) = self.state.device(device_type, number) {
            lock(device)
                .queued
                .entry((verb, member.to_ascii_lowercase()))
                .or_default()
                .push_back(reply);
        }
    }

    /// Stop serving. Further requests are refused.
    pub fn stop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for TestAlpacaServer {
    fn drop(&mut self) {
        self.stop();
    }
}

/// `None` when absent, `Some(Err(()))` when present but not a u32.
fn parse_id(params: &Params, key: &str) -> Option<Result<u32, ()>> {
    form::find(params, key, false).map(|text| text.trim().parse::<u32>().map_err(|_| ()))
}

async fn reply_to(state: &ServerState, mut reply: ServerReply, client_tx: Option<u32>) -> Response {
    loop {
        match reply {
            ServerReply::Delay(delay, inner) => {
                tokio::time::sleep(delay).await;
                reply = *inner;
            }
            ServerReply::Raw {
                status,
                content_type,
                body,
            } => {
                let status =
                    StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                let mut response = (status, body).into_response();
                response.headers_mut().remove(header::CONTENT_TYPE);
                if let Some(content_type) = content_type {
                    if let Ok(value) = content_type.parse() {
                        response.headers_mut().insert(header::CONTENT_TYPE, value);
                    }
                }
                return response;
            }
            ServerReply::Value(value) => {
                return alpaca_json(state, Some(value), 0, "", client_tx);
            }
            ServerReply::Ok => return alpaca_json(state, None, 0, "", client_tx),
            ServerReply::AlpacaError { number, message } => {
                return alpaca_json(state, None, number, &message, client_tx);
            }
        }
    }
}

fn alpaca_json(
    state: &ServerState,
    value: Option<Value>,
    error_number: i32,
    message: &str,
    client_tx: Option<u32>,
) -> Response {
    let mut body = json!({
        "ClientTransactionID": client_tx.unwrap_or(0),
        "ServerTransactionID": state.next_server_transaction(),
        "ErrorNumber": error_number,
        "ErrorMessage": message,
    });
    if let Some(value) = value {
        body["Value"] = value;
    }
    let mut response = (StatusCode::OK, body.to_string()).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/json; charset=utf-8"),
    );
    response
}

fn record(
    state: &ServerState,
    verb: Verb,
    device_type: &str,
    number: u32,
    member: &str,
    params: &Params,
    content_type: Option<String>,
) -> Result<Option<u32>, String> {
    let client_id = parse_id(params, form::CLIENT_ID_KEY);
    let client_tx = parse_id(params, form::CLIENT_TRANSACTION_ID_KEY);
    state.record(RecordedRequest {
        verb,
        device_type: device_type.into(),
        number,
        member: member.into(),
        params: params.to_vec(),
        content_type,
        client_id: client_id.and_then(Result::ok),
        client_transaction_id: client_tx.and_then(Result::ok),
    });
    for (key, parsed) in [
        (form::CLIENT_ID_KEY, client_id),
        (form::CLIENT_TRANSACTION_ID_KEY, client_tx),
    ] {
        if matches!(parsed, Some(Err(()))) {
            return Err(format!("Invalid {key}"));
        }
    }
    Ok(client_tx.and_then(Result::ok))
}

async fn device_get(
    State(state): State<Arc<ServerState>>,
    Path((device_type, number, member)): Path<(String, u32, String)>,
    RawQuery(query): RawQuery,
) -> Response {
    let device_type = device_type.to_ascii_lowercase();
    let member = member.to_ascii_lowercase();
    let params = form::decode_pairs(query.as_deref().unwrap_or_default());
    let client_tx = match record(
        &state,
        Verb::Get,
        &device_type,
        number,
        &member,
        &params,
        None,
    ) {
        Ok(tx) => tx,
        Err(message) => return (StatusCode::BAD_REQUEST, message).into_response(),
    };
    let Some(device) = state.device(&device_type, number) else {
        return (StatusCode::BAD_REQUEST, "Device not found").into_response();
    };
    let reply = lock(device).handle_get(&member, &params);
    reply_to(&state, reply, client_tx).await
}

async fn device_put(
    State(state): State<Arc<ServerState>>,
    Path((device_type, number, member)): Path<(String, u32, String)>,
    headers: HeaderMap,
    body: String,
) -> Response {
    let device_type = device_type.to_ascii_lowercase();
    let member = member.to_ascii_lowercase();
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let params = form::decode_pairs(&body);
    let client_tx = match record(
        &state,
        Verb::Put,
        &device_type,
        number,
        &member,
        &params,
        content_type.clone(),
    ) {
        Ok(tx) => tx,
        Err(message) => return (StatusCode::BAD_REQUEST, message).into_response(),
    };
    let form_encoded = content_type
        .as_deref()
        .and_then(|ct| ct.split(';').next())
        .is_some_and(|ct| ct.trim().eq_ignore_ascii_case(form::FORM_CONTENT_TYPE));
    if !form_encoded {
        return (StatusCode::BAD_REQUEST, "PUT body must be form-encoded").into_response();
    }
    let Some(device) = state.device(&device_type, number) else {
        return (StatusCode::BAD_REQUEST, "Device not found").into_response();
    };
    let reply = lock(device).handle_put(&member, &params);
    reply_to(&state, reply, client_tx).await
}

async fn management(
    state: &ServerState,
    path: &str,
    query: Option<String>,
    value: Value,
) -> Response {
    let params = form::decode_pairs(query.as_deref().unwrap_or_default());
    match record(state, Verb::Get, "management", 0, path, &params, None) {
        Ok(client_tx) => reply_to(state, ServerReply::Value(value), client_tx).await,
        Err(message) => (StatusCode::BAD_REQUEST, message).into_response(),
    }
}

async fn management_versions(
    State(state): State<Arc<ServerState>>,
    RawQuery(query): RawQuery,
) -> Response {
    management(&state, "apiversions", query, json!([1])).await
}

async fn management_description(
    State(state): State<Arc<ServerState>>,
    RawQuery(query): RawQuery,
) -> Response {
    let value = json!({
        "ServerName": "Achernar test Alpaca server",
        "Manufacturer": "Achernar",
        "ManufacturerVersion": env!("CARGO_PKG_VERSION"),
        "Location": "loopback",
    });
    management(&state, "description", query, value).await
}

async fn management_configured(
    State(state): State<Arc<ServerState>>,
    RawQuery(query): RawQuery,
) -> Response {
    let devices: Vec<Value> = state
        .devices
        .iter()
        .map(|slot| lock(&slot.device).configured_entry())
        .collect();
    management(&state, "configureddevices", query, Value::Array(devices)).await
}

async fn setup_page(Path((device_type, number)): Path<(String, u32)>) -> Response {
    let mut response =
        format!("<html><body>Setup for {device_type} {number}</body></html>").into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("text/html"),
    );
    response
}

// ---------------------------------------------------------------------------
// Raw HTTP script
// ---------------------------------------------------------------------------

/// What [`HttpScript`] does with one request (one connection each).
#[derive(Debug, Clone)]
pub enum RawAction {
    /// Reply and close.
    Respond {
        status: u16,
        content_type: Option<String>,
        body: String,
    },
    /// Read the request, then close the connection without replying.
    CloseWithoutReply,
    /// Read the request, hold the connection open this long, then close it
    /// without replying (a device that stopped answering).
    Stall(Duration),
}

/// A request [`HttpScript`] received.
#[derive(Debug, Clone, PartialEq)]
pub struct RawRequest {
    pub method: String,
    pub path: String,
    pub params: Vec<(String, String)>,
}

impl RawRequest {
    pub fn transaction_id(&self) -> Option<u32> {
        form::find(&self.params, form::CLIENT_TRANSACTION_ID_KEY, false)
            .and_then(|id| id.parse().ok())
    }
}

/// A raw HTTP/1.1 server on `127.0.0.1:0` that answers each request with
/// the next scripted [`RawAction`] (`Connection: close` on every reply, so
/// one request is one connection). Once the script is exhausted it answers
/// HTTP 500.
pub struct HttpScript {
    addr: SocketAddr,
    requests: Arc<Mutex<Vec<RawRequest>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl HttpScript {
    /// The Alpaca success body for `value` (`Null` for a void PUT).
    pub fn alpaca_ok(value: Value) -> RawAction {
        let mut body = json!({"ServerTransactionID": 1, "ErrorNumber": 0, "ErrorMessage": ""});
        if !value.is_null() {
            body["Value"] = value;
        }
        RawAction::Respond {
            status: 200,
            content_type: Some("application/json".into()),
            body: body.to_string(),
        }
    }

    pub fn start(actions: Vec<RawAction>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind the raw HTTP script");
        listener
            .set_nonblocking(true)
            .expect("non-blocking raw listener");
        let addr = listener.local_addr().expect("raw script address");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let actions = Arc::new(Mutex::new(VecDeque::from(actions)));
        let thread = {
            let requests = Arc::clone(&requests);
            let stop = Arc::clone(&stop);
            std::thread::Builder::new()
                .name("test-http-script".into())
                .spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        match listener.accept() {
                            Ok((stream, _)) => {
                                let requests = Arc::clone(&requests);
                                let actions = Arc::clone(&actions);
                                std::thread::spawn(move || serve_raw(stream, &requests, &actions));
                            }
                            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                                std::thread::sleep(Duration::from_millis(2));
                            }
                            Err(_) => std::thread::sleep(Duration::from_millis(2)),
                        }
                    }
                })
                .expect("spawn the raw HTTP script")
        };
        Self {
            addr,
            requests,
            stop,
            thread: Some(thread),
        }
    }

    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn requests(&self) -> Vec<RawRequest> {
        lock(&self.requests).clone()
    }
}

impl Drop for HttpScript {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn serve_raw(
    mut stream: TcpStream,
    requests: &Mutex<Vec<RawRequest>>,
    actions: &Mutex<VecDeque<RawAction>>,
) {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let Some(request) = read_raw_request(&mut stream) else {
        return;
    };
    let action = lock(actions).pop_front();
    lock(requests).push(request);
    match action {
        Some(RawAction::Respond {
            status,
            content_type,
            body,
        }) => {
            let mut head = format!(
                "HTTP/1.1 {status} {}\r\nContent-Length: {}\r\nConnection: close\r\n",
                reason_phrase(status),
                body.len()
            );
            if let Some(content_type) = content_type {
                head.push_str(&format!("Content-Type: {content_type}\r\n"));
            }
            head.push_str("\r\n");
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(body.as_bytes());
            let _ = stream.flush();
        }
        Some(RawAction::CloseWithoutReply) => {}
        Some(RawAction::Stall(duration)) => std::thread::sleep(duration),
        None => {
            let body = "script exhausted";
            let _ = write!(
                stream,
                "HTTP/1.1 500 Internal Server Error\r\nContent-Length: {}\r\nConnection: close\r\nContent-Type: text/plain\r\n\r\n{body}",
                body.len()
            );
        }
    }
    let _ = stream.shutdown(std::net::Shutdown::Both);
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Status",
    }
}

/// Read one request (headers and a Content-Length body).
fn read_raw_request(stream: &mut TcpStream) -> Option<RawRequest> {
    let mut data = Vec::new();
    let mut chunk = [0_u8; 4096];
    let header_end = loop {
        if let Some(end) = find_header_end(&data) {
            break end;
        }
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            return None;
        }
        data.extend_from_slice(&chunk[..read]);
    };
    let head = String::from_utf8_lossy(&data[..header_end]).into_owned();
    let mut lines = head.split("\r\n");
    let mut request_line = lines.next()?.split_whitespace();
    let method = request_line.next()?.to_string();
    let target = request_line.next()?.to_string();
    let content_length = lines
        .filter_map(|line| line.split_once(':'))
        .find(|(name, _)| name.trim().eq_ignore_ascii_case("content-length"))
        .and_then(|(_, value)| value.trim().parse::<usize>().ok())
        .unwrap_or(0);
    let body_start = header_end + 4;
    while data.len() < body_start + content_length {
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            break;
        }
        data.extend_from_slice(&chunk[..read]);
    }
    let body_end = (body_start + content_length).min(data.len());
    let body = String::from_utf8_lossy(&data[body_start..body_end]).into_owned();
    let (path, query) = target.split_once('?').unwrap_or((target.as_str(), ""));
    let mut params = form::decode_pairs(query);
    params.extend(form::decode_pairs(&body));
    Some(RawRequest {
        method,
        path: path.to_string(),
        params,
    })
}

fn find_header_end(data: &[u8]) -> Option<usize> {
    data.windows(4).position(|window| window == b"\r\n\r\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::alpaca::client::{AlpacaClient, ClientConfig, ClientId};
    use crate::equipment::traits::{AlpacaTransport, TimeoutClass};
    use crate::equipment::types::ErrorClass;

    fn client(server: &TestAlpacaServer, device_type: &str) -> AlpacaClient {
        AlpacaClient::new(
            &server.base_url(),
            device_type,
            0,
            ClientConfig::new(ClientId::new(321).unwrap()),
        )
        .unwrap()
    }

    #[test]
    fn the_esp32_like_mount_requires_connect_and_completes_a_goto_from_slewing() {
        let server = TestAlpacaServer::start(vec![ScriptedDevice::esp32_like_mount(0)]);
        let mount = client(&server, "telescope");
        let err = mount
            .get("rightascension", &[], TimeoutClass::Ordinary)
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::NotConnected);
        mount.put("connect", &[], TimeoutClass::Connect).unwrap();
        assert_eq!(
            mount.get("connecting", &[], TimeoutClass::Connect).unwrap(),
            json!(true)
        );
        assert_eq!(
            mount.get("connecting", &[], TimeoutClass::Connect).unwrap(),
            json!(false)
        );
        assert_eq!(
            mount.get("connected", &[], TimeoutClass::Connect).unwrap(),
            json!(true)
        );

        assert_eq!(
            mount.get("canslew", &[], TimeoutClass::Ordinary).unwrap(),
            json!(false)
        );
        assert_eq!(
            mount
                .get("canslewasync", &[], TimeoutClass::Ordinary)
                .unwrap(),
            json!(true)
        );
        assert_eq!(
            mount
                .get(
                    "canmoveaxis",
                    &[("Axis", "1".into())],
                    TimeoutClass::Ordinary
                )
                .unwrap(),
            json!(true)
        );
        assert!(mount
            .get("targetrightascension", &[], TimeoutClass::Ordinary)
            .unwrap_err()
            .is_not_set());
        let slew = [
            ("RightAscension", "5.5".to_string()),
            ("Declination", "-5.25".to_string()),
        ];
        let err = mount
            .put("slewtocoordinatesasync", &slew, TimeoutClass::Long)
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::InvalidWhileParked);
        mount.put("unpark", &[], TimeoutClass::Long).unwrap();
        mount
            .put("slewtocoordinatesasync", &slew, TimeoutClass::Long)
            .unwrap();
        let mut polls = 0;
        while mount.get("slewing", &[], TimeoutClass::Ordinary).unwrap() == json!(true) {
            polls += 1;
            assert!(polls < 10);
        }
        assert_eq!(polls, SCRIPTED_SLEW_READS - 1);
        assert_eq!(
            mount
                .get("rightascension", &[], TimeoutClass::Ordinary)
                .unwrap(),
            json!(5.5)
        );

        let state = mount
            .get("devicestate", &[], TimeoutClass::Ordinary)
            .unwrap();
        let parsed = crate::equipment::alpaca::devicestate::parse_device_state(&state).unwrap();
        assert!(parsed
            .missing(crate::equipment::alpaca::devicestate::expected_members(
                "telescope"
            ))
            .is_empty());
        assert_eq!(parsed.i64("SideOfPier").unwrap().unwrap(), 0);
        assert!(parsed.timestamp.is_some());
        assert_eq!(
            mount
                .put(
                    "sideofpier",
                    &[("SideOfPier", "1".into())],
                    TimeoutClass::Ordinary
                )
                .unwrap_err()
                .class,
            ErrorClass::Unsupported
        );

        mount
            .put(
                "pulseguide",
                &[("Direction", "0".into()), ("Duration", "300".into())],
                TimeoutClass::Long,
            )
            .unwrap();
        assert_eq!(
            mount
                .get("ispulseguiding", &[], TimeoutClass::Ordinary)
                .unwrap(),
            json!(true)
        );
        assert_eq!(
            mount
                .get("ispulseguiding", &[], TimeoutClass::Ordinary)
                .unwrap(),
            json!(false)
        );

        mount.put("park", &[], TimeoutClass::Long).unwrap();
        while mount.get("slewing", &[], TimeoutClass::Ordinary).unwrap() == json!(true) {}
        assert_eq!(
            mount.get("atpark", &[], TimeoutClass::Ordinary).unwrap(),
            json!(true)
        );
    }

    #[test]
    fn the_platform6_focuser_has_no_platform7_members_and_moves_absolutely() {
        let server = TestAlpacaServer::start(vec![ScriptedDevice::platform6_focuser(0)]);
        let focuser = client(&server, "focuser");
        assert_eq!(
            focuser
                .get("interfaceversion", &[], TimeoutClass::Connect)
                .unwrap(),
            json!(3)
        );
        assert!(focuser
            .put("connect", &[], TimeoutClass::Connect)
            .unwrap_err()
            .is_unknown_member());
        assert!(focuser
            .get("devicestate", &[], TimeoutClass::Ordinary)
            .unwrap_err()
            .is_unknown_member());
        focuser
            .put(
                "connected",
                &[("Connected", "True".into())],
                TimeoutClass::Connect,
            )
            .unwrap();
        assert_eq!(
            focuser
                .get("temperature", &[], TimeoutClass::Ordinary)
                .unwrap(),
            json!(12.5)
        );
        let err = focuser
            .put("move", &[("Position", "30000".into())], TimeoutClass::Long)
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::InvalidValue);
        focuser
            .put("move", &[("Position", "12000".into())], TimeoutClass::Long)
            .unwrap();
        while focuser
            .get("ismoving", &[], TimeoutClass::Ordinary)
            .unwrap()
            == json!(true)
        {}
        assert_eq!(
            focuser
                .get("position", &[], TimeoutClass::Ordinary)
                .unwrap(),
            json!(12000)
        );
        assert_eq!(
            focuser
                .put(
                    "tempcomp",
                    &[("TempComp", "True".into())],
                    TimeoutClass::Ordinary
                )
                .unwrap_err()
                .class,
            ErrorClass::Unsupported
        );
    }

    #[test]
    fn the_sim_mount_adapter_serves_the_kinematic_simulator_over_http() {
        use crate::equipment::sim::mount::{SimFaults, SimMountProfile};
        use crate::equipment::testing::manual_clock::ManualClock;
        use std::sync::Arc;

        let clock = Arc::new(ManualClock::new());
        let sim = SimMount::new(SimMountProfile::esp32_like(), clock.clone());
        let control = sim.control();
        let server = TestAlpacaServer::start(vec![
            ScriptedDevice::sim_mount(0, sim).unique_id("sim-mount-adapter")
        ]);
        let mount = client(&server, "telescope");
        // Platform 7 connect, gated members until then.
        assert_eq!(
            mount
                .get("tracking", &[], TimeoutClass::Ordinary)
                .unwrap_err()
                .class,
            ErrorClass::NotConnected
        );
        assert_eq!(
            mount
                .get("interfaceversion", &[], TimeoutClass::Connect)
                .unwrap(),
            json!(4)
        );
        mount.put("connect", &[], TimeoutClass::Connect).unwrap();
        while mount.get("connecting", &[], TimeoutClass::Connect).unwrap() == json!(true) {}
        assert_eq!(
            mount.get("connected", &[], TimeoutClass::Connect).unwrap(),
            json!(true)
        );
        // The profile's capabilities (O10) and frame.
        for (member, expected) in [
            ("canslew", json!(false)),
            ("canslewasync", json!(true)),
            ("canpark", json!(true)),
            ("canfindhome", json!(false)),
            ("alignmentmode", json!(1)),
            ("equatorialsystem", json!(1)),
            ("trackingrates", json!([0, 1, 2, 3])),
        ] {
            assert_eq!(
                mount.get(member, &[], TimeoutClass::Ordinary).unwrap(),
                expected,
                "{member}"
            );
        }
        assert_eq!(
            mount
                .get(
                    "canmoveaxis",
                    &[("Axis", "1".into())],
                    TimeoutClass::Ordinary
                )
                .unwrap(),
            json!(true)
        );
        assert_eq!(
            mount
                .get("axisrates", &[("Axis", "0".into())], TimeoutClass::Ordinary)
                .unwrap(),
            json!([{"Minimum": 0.0, "Maximum": 4.0}])
        );
        // DeviceState carries every ITelescopeV4 member; SideOfPier is the
        // ESP32's constant 0.
        let state = mount
            .get("devicestate", &[], TimeoutClass::Ordinary)
            .unwrap();
        let parsed = crate::equipment::alpaca::devicestate::parse_device_state(&state).unwrap();
        assert!(parsed
            .missing(crate::equipment::alpaca::devicestate::expected_members(
                "telescope"
            ))
            .is_empty());
        assert_eq!(parsed.i64("SideOfPier").unwrap().unwrap(), 0);
        // Synchronous slews are not implemented (O10); the async GoTo moves
        // the simulated axes on the simulator's clock.
        let target = [
            ("RightAscension", "3.25".to_string()),
            ("Declination", "35.5".to_string()),
        ];
        assert_eq!(
            mount
                .put("slewtocoordinates", &target, TimeoutClass::Long)
                .unwrap_err()
                .class,
            ErrorClass::Unsupported
        );
        mount
            .put("slewtocoordinatesasync", &target, TimeoutClass::Long)
            .unwrap();
        assert_eq!(
            mount.get("slewing", &[], TimeoutClass::Ordinary).unwrap(),
            json!(true)
        );
        clock.advance(Duration::from_secs(120));
        assert_eq!(
            mount.get("slewing", &[], TimeoutClass::Ordinary).unwrap(),
            json!(false)
        );
        let ra = mount
            .get("rightascension", &[], TimeoutClass::Ordinary)
            .unwrap()
            .as_f64()
            .unwrap();
        assert!((ra - 3.25).abs() < 1e-6, "{ra}");
        // A pulse is observed through IsPulseGuiding.
        mount
            .put(
                "pulseguide",
                &[("Direction", "0".into()), ("Duration", "300".into())],
                TimeoutClass::Long,
            )
            .unwrap();
        assert_eq!(
            mount
                .get("ispulseguiding", &[], TimeoutClass::Ordinary)
                .unwrap(),
            json!(true)
        );
        clock.advance(Duration::from_millis(400));
        assert_eq!(
            mount
                .get("ispulseguiding", &[], TimeoutClass::Ordinary)
                .unwrap(),
            json!(false)
        );
        // Park completes on AtPark; a slew while parked is refused.
        mount.put("park", &[], TimeoutClass::Long).unwrap();
        clock.advance(Duration::from_secs(120));
        assert_eq!(
            mount.get("atpark", &[], TimeoutClass::Ordinary).unwrap(),
            json!(true)
        );
        assert_eq!(
            mount
                .put("slewtocoordinatesasync", &target, TimeoutClass::Long)
                .unwrap_err()
                .class,
            ErrorClass::InvalidWhileParked
        );
        // Faults injected through the simulator's control still apply.
        control.set_faults(SimFaults {
            fail_polls: 1,
            ..SimFaults::default()
        });
        assert_eq!(
            mount
                .get("atpark", &[], TimeoutClass::Ordinary)
                .unwrap_err()
                .class,
            ErrorClass::Driver
        );
        // Scripted replies win over the simulator.
        server.queue_reply(
            "telescope",
            0,
            Verb::Get,
            "slewing",
            ServerReply::text(500, "glitch"),
        );
        assert_eq!(
            mount
                .get("slewing", &[], TimeoutClass::Ordinary)
                .unwrap_err()
                .class,
            ErrorClass::Driver
        );
        assert_eq!(
            server.requests()[0].member,
            "tracking",
            "every request is still recorded"
        );
    }

    #[test]
    fn the_sim_focuser_adapter_is_a_platform6_focuser_with_a_probe() {
        use crate::equipment::sim::focuser::SimFocuserConfig;
        use crate::equipment::testing::manual_clock::ManualClock;
        use std::sync::Arc;

        let clock = Arc::new(ManualClock::new());
        let sim = SimFocuser::new(
            SimFocuserConfig::platform6_firmware_backlash(),
            clock.clone(),
        );
        let server = TestAlpacaServer::start(vec![ScriptedDevice::sim_focuser(0, sim)]);
        let focuser = client(&server, "focuser");
        assert_eq!(
            focuser
                .get("interfaceversion", &[], TimeoutClass::Connect)
                .unwrap(),
            json!(3)
        );
        // An IFocuserV3 server knows neither Connect nor DeviceState.
        assert!(focuser
            .put("connect", &[], TimeoutClass::Connect)
            .unwrap_err()
            .is_unknown_member());
        assert!(focuser
            .get("devicestate", &[], TimeoutClass::Ordinary)
            .unwrap_err()
            .is_unknown_member());
        focuser
            .put(
                "connected",
                &[("Connected", "True".into())],
                TimeoutClass::Connect,
            )
            .unwrap();
        for (member, expected) in [
            ("absolute", json!(true)),
            ("maxstep", json!(50_000)),
            ("temperature", json!(12.0)),
            ("tempcompavailable", json!(false)),
            ("tempcomp", json!(false)),
            ("position", json!(25_000)),
            ("ismoving", json!(false)),
        ] {
            assert_eq!(
                focuser.get(member, &[], TimeoutClass::Ordinary).unwrap(),
                expected,
                "{member}"
            );
        }
        focuser
            .put("move", &[("Position", "25400".into())], TimeoutClass::Long)
            .unwrap();
        assert_eq!(
            focuser
                .get("ismoving", &[], TimeoutClass::Ordinary)
                .unwrap(),
            json!(true)
        );
        clock.advance(Duration::from_secs(2));
        assert_eq!(
            focuser
                .get("ismoving", &[], TimeoutClass::Ordinary)
                .unwrap(),
            json!(false)
        );
        assert_eq!(
            focuser
                .get("position", &[], TimeoutClass::Ordinary)
                .unwrap(),
            json!(25_400)
        );
        assert_eq!(
            focuser
                .put(
                    "tempcomp",
                    &[("TempComp", "True".into())],
                    TimeoutClass::Ordinary
                )
                .unwrap_err()
                .class,
            ErrorClass::Unsupported
        );
        focuser.put("halt", &[], TimeoutClass::Long).unwrap();
    }

    #[test]
    fn invalid_ids_and_non_form_bodies_get_http_400_like_a_real_server() {
        let server = TestAlpacaServer::start(vec![
            ScriptedDevice::new("focuser", 0, "F").property("position", 1)
        ]);
        let agent = ureq::Agent::config_builder()
            .proxy(None)
            .http_status_as_error(false)
            .build()
            .new_agent();
        let mut response = agent
            .get(format!(
                "{}/api/v1/focuser/0/position?ClientTransactionID=abc",
                server.base_url()
            ))
            .call()
            .unwrap();
        assert_eq!(response.status().as_u16(), 400);
        assert!(response
            .body_mut()
            .read_to_string()
            .unwrap()
            .contains("ClientTransactionID"));
        let response = agent
            .put(format!("{}/api/v1/focuser/0/move", server.base_url()))
            .content_type("application/json")
            .send("{\"Position\": 1}")
            .unwrap();
        assert_eq!(response.status().as_u16(), 400);
        let response = agent
            .get(format!("{}/api/v1/focuser/7/position", server.base_url()))
            .call()
            .unwrap();
        assert_eq!(response.status().as_u16(), 400);
        let mut response = agent
            .get(format!("{}/setup/v1/focuser/0/setup", server.base_url()))
            .call()
            .unwrap();
        assert_eq!(response.status().as_u16(), 200);
        assert!(response
            .body_mut()
            .read_to_string()
            .unwrap()
            .contains("Setup"));
    }

    #[test]
    fn a_stopped_server_refuses_connections() {
        let mut server = TestAlpacaServer::start(vec![
            ScriptedDevice::new("focuser", 0, "F").property("position", 1)
        ]);
        let focuser = client(&server, "focuser");
        assert!(focuser.get("position", &[], TimeoutClass::Ordinary).is_ok());
        server.stop();
        let err = focuser
            .get("position", &[], TimeoutClass::Ordinary)
            .unwrap_err();
        assert!(
            matches!(err.class, ErrorClass::Transport | ErrorClass::Timeout),
            "{err}"
        );
        assert!(server
            .with_device("focuser", 0, |model| model.f64("position"))
            .is_some());
    }

    #[test]
    fn queued_replies_can_be_added_while_the_server_runs() {
        let server = TestAlpacaServer::start(vec![
            ScriptedDevice::new("focuser", 0, "F").property("position", 1)
        ]);
        let focuser = client(&server, "focuser");
        server.queue_reply(
            "focuser",
            0,
            Verb::Get,
            "position",
            ServerReply::text(500, "glitch"),
        );
        assert_eq!(
            focuser
                .get("position", &[], TimeoutClass::Ordinary)
                .unwrap_err()
                .class,
            ErrorClass::Driver
        );
        assert_eq!(
            focuser
                .get("position", &[], TimeoutClass::Ordinary)
                .unwrap(),
            json!(1)
        );
        server.with_device("focuser", 0, |model| model.set("position", 99));
        assert_eq!(
            focuser
                .get("position", &[], TimeoutClass::Ordinary)
                .unwrap(),
            json!(99)
        );
    }
}

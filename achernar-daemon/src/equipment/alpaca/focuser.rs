//! ASCOM Alpaca focuser driver (`IFocuserV1` to `V4`; spec §3.3, §7.1
//! R12-R16), an independent implementation from the ASCOM Alpaca API
//! reference and Master Interfaces (MIT).
//!
//! It runs over any [`AlpacaTransport`]: the WP1A-1 client in production,
//! `FakeAlpacaTransport` in tests. Every member name is lower case and every
//! PUT parameter is in exact case (`Position`, `TempComp`, `Connected`,
//! `Action`, `Parameters`), as C-3 requires.
//!
//! Connect (§2.6): `interfaceversion` (a failed read counts as 1); at
//! InterfaceVersion 4 or higher, PUT `connect` and poll `connecting` every
//! 250 ms (Platform 7); otherwise, or when `connect` is unknown, PUT
//! `Connected=True` (Platform 6). Then `connected` must read true. The
//! identity, the capabilities (`absolute`, `maxstep`, `maxincrement`,
//! `stepsize`, `tempcompavailable`) and `devicestate` are probed; a member
//! that answers NotImplemented, HTTP 400 or HTTP 404 is memoised as
//! unsupported for the session (optional members only: `ismoving` and an
//! absolute focuser's `position` are never memoised).
//!
//! Poll (§2.7): one `devicestate` GET (IsMoving, Position, Temperature,
//! TimeStamp) when the device has it, else per-member reads of `ismoving`,
//! `position` (absolute only) and `temperature`. A member missing from
//! DeviceState is read on its own. `tempcomp` is not in DeviceState and is
//! read when the plan lists it (every 10 s), after a write, and on the first
//! poll. Every value is timestamped at the midpoint of its request with its
//! latency (D5). ValueNotSet is `NotSet`; a failed read keeps the last value
//! as `Stale` (C-2). A link failure (transport, timeout, not connected)
//! skips the rest of the tick and fails the poll, so a dead server costs one
//! timeout per tick, not one per member.
//!
//! Alpaca has no position sync, so `sync_position` is `Unsupported` (R16).
//! The abort path PUTs `halt` on a second transport (the client's abort
//! agent), so it never queues behind a slow request (D4).
//!
//! The DeviceState parsing here is local to the focuser; once the shared
//! parser in `alpaca/devicestate.rs` (WP1A-1) lands it can replace
//! [`parse_device_state`] and [`parse_timestamp`].

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Local, NaiveDateTime, TimeZone, Utc};
use serde_json::Value;

use crate::equipment::clock::duration_ns;
use crate::equipment::snapshot::{FocuserCapabilities, FocuserState};
use crate::equipment::traits::{
    AbortPath, AlpacaTransport, Clock, ConnectPath, ConnectReport, Device, FocuserDriver,
    PollMethod, PollPlan, RawMember, RawPoll, TimeoutClass,
};
use crate::equipment::types::{
    identity_key, DeviceError, DeviceIdentity, DeviceKind, DeviceResult, DriverId, Endpoint,
    ErrorClass, ObsState, Observed, Source,
};

/// Focuser InterfaceVersion at and above which Platform 7 `connect` and
/// `devicestate` exist.
pub const PLATFORM7_INTERFACE: i32 = 4;
/// `connecting` poll period (§2.6 step 4).
pub const CONNECT_POLL_INTERVAL: Duration = Duration::from_millis(250);
/// TempComp older than this without a refresh is published `Stale`.
const TEMPCOMP_STALE_AFTER: Duration = Duration::from_secs(30);

fn alpaca_bool(value: bool) -> String {
    if value { "True" } else { "False" }.to_string()
}

fn as_bool(value: &Value) -> Option<bool> {
    value.as_bool()
}

fn as_i32(value: &Value) -> Option<i32> {
    value
        .as_i64()
        .or_else(|| {
            value
                .as_f64()
                .filter(|f| f.fract() == 0.0)
                .map(|f| f as i64)
        })
        .and_then(|v| i32::try_from(v).ok())
}

fn as_f64(value: &Value) -> Option<f64> {
    value.as_f64().filter(|f| f.is_finite())
}

/// `{"Name": .., "Value": ..}` entries of a `devicestate` reply, keyed by
/// lower-case name. `None` when the reply is not such a list.
pub fn parse_device_state(value: &Value) -> Option<BTreeMap<String, Value>> {
    let entries = value.as_array()?;
    let mut out = BTreeMap::new();
    for entry in entries {
        let object = entry.as_object()?;
        let name = object
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("name"))
            .and_then(|(_, v)| v.as_str())?;
        let value = object
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("value"))
            .map(|(_, v)| v.clone())
            .unwrap_or(Value::Null);
        out.insert(name.to_ascii_lowercase(), value);
    }
    Some(out)
}

/// DeviceState `TimeStamp`, parsed permissively (§2.14): an explicit offset
/// or `Z`, else a local time without offset.
pub fn parse_timestamp(text: &str) -> Option<DateTime<Utc>> {
    let text = text.trim();
    if let Ok(parsed) = DateTime::parse_from_rfc3339(text) {
        return Some(parsed.with_timezone(&Utc));
    }
    ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f"]
        .iter()
        .find_map(|format| NaiveDateTime::parse_from_str(text, format).ok())
        .and_then(|naive| Local.from_local_datetime(&naive).earliest())
        .map(|local| local.with_timezone(&Utc))
}

/// `(host, port)` of an `http(s)://host:port` base URL.
fn host_port(base: &str) -> (String, u16) {
    let (scheme, rest) = base.split_once("://").unwrap_or(("http", base));
    let authority = rest.split(['/', '#', '?']).next().unwrap_or_default();
    let default_port = if scheme.eq_ignore_ascii_case("https") {
        443
    } else {
        80
    };
    match authority.rsplit_once(':') {
        Some((host, port)) if !host.is_empty() => {
            (host.to_string(), port.parse().unwrap_or(default_port))
        }
        _ => (authority.to_string(), default_port),
    }
}

#[derive(Debug, Clone, Copy)]
struct Timing {
    at: DateTime<Utc>,
    mono: u64,
    latency_ms: u32,
}

impl Timing {
    fn observed<T>(self, value: T, source: Source) -> Observed<T> {
        Observed::fresh(value, source, self.at, self.mono).with_latency_ms(self.latency_ms)
    }
}

/// One parsed `devicestate` reply.
struct DeviceStateSample {
    values: BTreeMap<String, Value>,
    timing: Timing,
    device_time: Option<DateTime<Utc>>,
}

/// True for failures that mean the link, not the member, is broken.
fn link_failure(err: &DeviceError) -> bool {
    matches!(
        err.class,
        ErrorClass::Transport
            | ErrorClass::Timeout
            | ErrorClass::NotConnected
            | ErrorClass::Disconnected
    )
}

/// The Alpaca focuser driver.
pub struct AlpacaFocuser {
    identity: DeviceIdentity,
    base: String,
    number: u32,
    transport: Arc<dyn AlpacaTransport>,
    abort_transport: Arc<dyn AlpacaTransport>,
    clock: Arc<dyn Clock>,
    connect_poll: Duration,
    connected: bool,
    path: ConnectPath,
    device_state: bool,
    unsupported: BTreeSet<String>,
    caps: Option<FocuserCapabilities>,
    state: FocuserState,
    tempcomp_dirty: bool,
}

impl AlpacaFocuser {
    /// `base` is `http(s)://host:port`; `number` the Alpaca device number.
    /// `abort_transport` must be a separate connection to the same device
    /// (the client's abort agent; D4).
    pub fn new(
        base: impl Into<String>,
        number: u32,
        transport: Arc<dyn AlpacaTransport>,
        abort_transport: Arc<dyn AlpacaTransport>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let base = base.into().trim_end_matches('/').to_string();
        let (host, port) = host_port(&base);
        let mut identity = DeviceIdentity::new(
            DeviceKind::Focuser,
            DriverId::Alpaca,
            Endpoint::AlpacaHttp {
                base: base.clone(),
                device_type: "focuser".into(),
                number,
            },
            identity_key::alpaca(None, &host, port, "focuser", number),
            "Alpaca focuser",
        );
        identity.setup_url = Some(format!("{base}/setup/v1/focuser/{number}/setup"));
        Self {
            identity,
            base,
            number,
            transport,
            abort_transport,
            clock,
            connect_poll: CONNECT_POLL_INTERVAL,
            connected: false,
            path: ConnectPath::Platform6,
            device_state: false,
            unsupported: BTreeSet::new(),
            caps: None,
            state: FocuserState::default(),
            tempcomp_dirty: true,
        }
    }

    /// The UniqueID from the management API's `configureddevices` (§2.6
    /// step 5), which makes the identity key survive an IP change.
    pub fn with_unique_id(mut self, unique_id: Option<&str>) -> Self {
        let (host, port) = host_port(&self.base);
        self.identity.unique_id = unique_id
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(str::to_string);
        self.identity.identity_key = identity_key::alpaca(
            self.identity.unique_id.as_deref(),
            &host,
            port,
            "focuser",
            self.number,
        );
        self
    }

    /// The `connecting` poll period (tests use zero).
    pub fn with_connect_poll_interval(mut self, interval: Duration) -> Self {
        self.connect_poll = interval;
        self
    }

    pub fn connect_path(&self) -> ConnectPath {
        self.path
    }

    fn timed<T>(&self, call: impl FnOnce() -> T) -> (T, Timing) {
        let utc0 = self.clock.now_utc();
        let mono0 = self.clock.now_mono_ns();
        let out = call();
        let utc1 = self.clock.now_utc();
        let mono1 = self.clock.now_mono_ns().max(mono0);
        let timing = Timing {
            at: utc0 + (utc1 - utc0) / 2,
            mono: mono0 + (mono1 - mono0) / 2,
            latency_ms: u32::try_from((mono1 - mono0) / 1_000_000).unwrap_or(u32::MAX),
        };
        (out, timing)
    }

    fn get(&self, member: &str, class: TimeoutClass) -> DeviceResult<Value> {
        self.transport.get(member, &[], class)
    }

    fn put(
        &self,
        member: &str,
        form: &[(&str, String)],
        class: TimeoutClass,
    ) -> DeviceResult<Value> {
        self.transport.put(member, form, class)
    }

    fn require_connected(&self) -> DeviceResult<()> {
        if self.connected {
            Ok(())
        } else {
            Err(DeviceError::not_connected(
                "the Alpaca focuser is not connected",
            ))
        }
    }

    fn caps(&self) -> DeviceResult<&FocuserCapabilities> {
        self.caps
            .as_ref()
            .ok_or_else(|| DeviceError::not_connected("the Alpaca focuser is not connected"))
    }

    /// Read an optional member once at connect; unknown members are
    /// memoised.
    fn probe<T>(
        &mut self,
        member: &str,
        parse: fn(&Value) -> Option<T>,
    ) -> DeviceResult<Option<T>> {
        match self.get(member, TimeoutClass::Ordinary) {
            Ok(value) => Ok(parse(&value)),
            Err(err) if err.is_unknown_member() => {
                self.unsupported.insert(member.to_string());
                Ok(None)
            }
            Err(err) if err.is_not_set() => Ok(None),
            Err(err) if link_failure(&err) => Err(err),
            Err(_) => Ok(None),
        }
    }

    /// Poll `connecting` until it reads false. The budget is elapsed clock
    /// time, request latency included: no GET starts once `timeout` has
    /// passed. (One GET in flight can still run to its own `Connect`
    /// timeout; the frozen transport takes a timeout class, not a budget.)
    fn wait_connecting(&self, timeout: Duration) -> DeviceResult<()> {
        let started = self.clock.now_mono_ns();
        let budget = duration_ns(timeout);
        // A backstop for a clock that does not advance (a test clock); with
        // a real clock the elapsed time ends the wait first.
        let max_attempts = if self.connect_poll.is_zero() {
            1_000
        } else {
            timeout.as_millis() / self.connect_poll.as_millis().max(1) + 1
        };
        let mut attempts = 0u128;
        loop {
            attempts += 1;
            match self.get("connecting", TimeoutClass::Connect) {
                Ok(value) if as_bool(&value) == Some(false) => return Ok(()),
                Ok(_) => {}
                // A server that knows `connect` but not `connecting` has
                // nothing to wait for.
                Err(err) if err.is_unknown_member() => return Ok(()),
                Err(err) => return Err(err),
            }
            let elapsed = self.clock.now_mono_ns().saturating_sub(started);
            if elapsed >= budget || attempts >= max_attempts {
                break;
            }
            if !self.connect_poll.is_zero() {
                let left = Duration::from_nanos(budget - elapsed);
                std::thread::sleep(self.connect_poll.min(left));
            }
        }
        Err(DeviceError::timeout(format!(
            "the focuser was still connecting after {} s",
            timeout.as_secs()
        )))
    }

    /// §2.6: any failure after the device may have accepted the connect
    /// disconnects it again (best effort), so no session is left dangling
    /// for this ClientID (D16: per-ClientID firmware stops motion only when
    /// its last client leaves). Returns `err`.
    fn abandon_connect(&mut self, path: ConnectPath, err: DeviceError) -> DeviceError {
        self.path = path;
        self.connected = false;
        self.caps = None;
        if let Err(cleanup) = self.disconnect() {
            tracing::warn!(
                error = %cleanup,
                "focuser disconnect after a failed connect failed"
            );
        }
        err
    }

    fn probe_capabilities(&mut self, interface_version: i32) -> DeviceResult<FocuserCapabilities> {
        let absolute = match self.get("absolute", TimeoutClass::Ordinary) {
            Ok(value) => as_bool(&value).ok_or_else(|| {
                DeviceError::driver(format!("absolute returned {value}, not a boolean"))
            })?,
            Err(err) if err.is_unknown_member() => {
                // Mandatory in every interface version; without it the
                // position cannot be trusted, so treat it as relative.
                self.unsupported.insert("absolute".into());
                false
            }
            Err(err) => return Err(err),
        };
        let max_step = self.probe("maxstep", as_i32)?;
        let max_increment = self.probe("maxincrement", as_i32)?;
        let step_size_um = self.probe("stepsize", as_f64)?.filter(|s| *s > 0.0);
        let temp_comp_available = self.probe("tempcompavailable", as_bool)?.unwrap_or(false);
        if !temp_comp_available {
            self.unsupported.insert("tempcomp".into());
        }
        if !absolute {
            // Relative focusers raise NotImplemented for Position.
            self.unsupported.insert("position".into());
        }
        if interface_version >= PLATFORM7_INTERFACE {
            match self.get("devicestate", TimeoutClass::Ordinary) {
                Ok(value) => {
                    self.device_state = parse_device_state(&value).is_some_and(|m| !m.is_empty());
                    if !self.device_state {
                        self.unsupported.insert("devicestate".into());
                    }
                }
                Err(err) if link_failure(&err) => return Err(err),
                Err(_) => {
                    self.device_state = false;
                    self.unsupported.insert("devicestate".into());
                }
            }
        } else {
            self.device_state = false;
            self.unsupported.insert("devicestate".into());
        }
        Ok(FocuserCapabilities {
            absolute,
            max_step: max_step.filter(|m| *m > 0),
            max_increment: max_increment.filter(|m| *m > 0),
            step_size_um,
            temp_comp_available,
            can_sync: false,
            interface_version: Some(interface_version),
            device_state: self.device_state,
            unsupported_members: self.unsupported.iter().cloned().collect(),
        })
    }

    /// One per-member read, applying C-2.
    fn read_member<T: Clone>(
        &mut self,
        name: &str,
        mandatory: bool,
        previous: &Observed<T>,
        parse: fn(&Value) -> Option<T>,
        members: &mut Vec<RawMember>,
        failure: &mut Option<DeviceError>,
    ) -> Observed<T> {
        if self.unsupported.contains(name) {
            return Observed::unsupported(Source::Property);
        }
        if failure.is_some() {
            return previous.clone().into_stale();
        }
        let (result, timing) = self.timed(|| self.get(name, TimeoutClass::Ordinary));
        let mut member = RawMember {
            name: name.to_string(),
            value: None,
            state: ObsState::Fresh,
            observed_at: Some(timing.at),
            mono_ns: Some(timing.mono),
            latency_ms: Some(timing.latency_ms),
            error: None,
        };
        let observed = match result {
            Ok(value) => match parse(&value) {
                Some(parsed) => {
                    member.value = Some(value);
                    timing.observed(parsed, Source::Property)
                }
                None => {
                    member.state = ObsState::Error;
                    member.error = Some(format!("unexpected value {value}"));
                    previous.clone().into_stale()
                }
            },
            Err(err) if err.is_not_set() => {
                member.state = ObsState::NotSet;
                Observed {
                    observed_at: Some(timing.at),
                    mono_ns: Some(timing.mono),
                    latency_ms: Some(timing.latency_ms),
                    ..Observed::not_set(Source::Property)
                }
            }
            Err(err) if !mandatory && err.is_unknown_member() => {
                self.unsupported.insert(name.to_string());
                member.state = ObsState::Unsupported;
                member.error = Some(err.to_string());
                Observed::unsupported(Source::Property)
            }
            Err(err) => {
                member.state = ObsState::Error;
                member.error = Some(err.to_string());
                if link_failure(&err) {
                    *failure = Some(err);
                }
                previous.clone().into_stale()
            }
        };
        members.push(member);
        observed
    }

    /// A DeviceState value, or a fallback per-member read when it is missing.
    #[allow(clippy::too_many_arguments)]
    fn state_or_member<T: Clone>(
        &mut self,
        name: &str,
        mandatory: bool,
        previous: &Observed<T>,
        parse: fn(&Value) -> Option<T>,
        device_state: Option<&DeviceStateSample>,
        members: &mut Vec<RawMember>,
        failure: &mut Option<DeviceError>,
    ) -> Observed<T> {
        if self.unsupported.contains(name) {
            return Observed::unsupported(Source::Property);
        }
        if let Some(sample) = device_state {
            if let Some(parsed) = sample.values.get(name).and_then(parse) {
                let timing = sample.timing;
                let mut observed = timing.observed(parsed, Source::DeviceState);
                if let Some(device_time) = sample.device_time {
                    observed = observed.with_device_time(device_time);
                }
                members.push(RawMember {
                    name: name.to_string(),
                    value: sample.values.get(name).cloned(),
                    state: ObsState::Fresh,
                    observed_at: Some(timing.at),
                    mono_ns: Some(timing.mono),
                    latency_ms: Some(timing.latency_ms),
                    error: None,
                });
                return observed;
            }
        }
        self.read_member(name, mandatory, previous, parse, members, failure)
    }
}

impl Device for AlpacaFocuser {
    fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    fn connect(&mut self, timeout: Duration) -> DeviceResult<ConnectReport> {
        let started = self.clock.now_mono_ns();
        let mut warnings = Vec::new();
        self.unsupported.clear();
        let interface_version = match self.get("interfaceversion", TimeoutClass::Connect) {
            Ok(value) => as_i32(&value).unwrap_or(1),
            Err(err) => {
                // ASCOM Library rule: a failed read is version 1.
                warnings.push(format!("interfaceversion unreadable ({err}); assuming 1"));
                1
            }
        };
        self.identity.interface_version = Some(interface_version);

        // A connect write that timed out may still have reached the device.
        let unanswered = |err: &DeviceError| {
            matches!(err.class, ErrorClass::Timeout | ErrorClass::OutcomeUnknown)
        };
        let mut path = ConnectPath::Platform6;
        if interface_version >= PLATFORM7_INTERFACE {
            match self.put("connect", &[], TimeoutClass::Connect) {
                Ok(_) => {
                    path = ConnectPath::Platform7;
                    if let Err(err) = self.wait_connecting(timeout) {
                        return Err(self.abandon_connect(path, err));
                    }
                }
                Err(err) if err.is_unknown_member() => warnings.push(format!(
                    "connect is not implemented ({err}); using Connected=True"
                )),
                Err(err) if unanswered(&err) => {
                    return Err(self.abandon_connect(ConnectPath::Platform7, err))
                }
                Err(err) => return Err(err),
            }
        }
        if path == ConnectPath::Platform6 {
            if let Err(err) = self.put(
                "connected",
                &[("Connected", alpaca_bool(true))],
                TimeoutClass::Connect,
            ) {
                return Err(if unanswered(&err) {
                    self.abandon_connect(path, err)
                } else {
                    err
                });
            }
        }
        let confirmed = match self.get("connected", TimeoutClass::Connect) {
            Ok(value) => value,
            Err(err) => return Err(self.abandon_connect(path, err)),
        };
        if as_bool(&confirmed) != Some(true) {
            let err = DeviceError::not_connected(format!(
                "the focuser reports connected = {confirmed} after connecting"
            ));
            return Err(self.abandon_connect(path, err));
        }

        // Identity (§2.6 step 5). Failures leave the field unknown.
        let text = |me: &Self, member: &str| {
            me.get(member, TimeoutClass::Ordinary)
                .ok()
                .and_then(|v| v.as_str().map(str::trim).map(str::to_string))
                .filter(|s| !s.is_empty())
        };
        if let Some(name) = text(self, "name") {
            self.identity.name = name;
        }
        self.identity.description = text(self, "description");
        self.identity.driver_info = text(self, "driverinfo");
        self.identity.driver_version = text(self, "driverversion");

        let caps = match self.probe_capabilities(interface_version) {
            Ok(caps) => caps,
            Err(err) => return Err(self.abandon_connect(path, err)),
        };
        self.caps = Some(caps.clone());
        self.path = path;
        self.connected = true;
        self.state = FocuserState::default();
        self.tempcomp_dirty = true;
        Ok(ConnectReport {
            path,
            interface_version: Some(interface_version),
            device_state: caps.device_state,
            unsupported_members: caps.unsupported_members.clone(),
            warnings,
            duration_ms: self.clock.now_mono_ns().saturating_sub(started) / 1_000_000,
        })
    }

    /// Ends this client's session only. With a per-ClientID device (the
    /// ESP32 firmware) motion stops only when the last client leaves (D16).
    fn disconnect(&mut self) -> DeviceResult<()> {
        let was = self.path;
        self.connected = false;
        if was == ConnectPath::Platform7 {
            match self.put("disconnect", &[], TimeoutClass::Connect) {
                Ok(_) => return Ok(()),
                Err(err) if !err.is_unknown_member() => return Err(err),
                Err(_) => {}
            }
        }
        self.put(
            "connected",
            &[("Connected", alpaca_bool(false))],
            TimeoutClass::Connect,
        )
        .map(|_| ())
    }

    fn is_connected(&mut self) -> DeviceResult<bool> {
        let value = self.get("connected", TimeoutClass::Connect)?;
        Ok(as_bool(&value) == Some(true))
    }

    fn poll(&mut self, plan: &PollPlan) -> DeviceResult<RawPoll> {
        self.require_connected()?;
        let started = self.clock.now_mono_ns();
        let absolute = self.caps()?.absolute;
        let temp_comp_available = self.caps()?.temp_comp_available;
        let mut members = Vec::new();
        let mut failure: Option<DeviceError> = None;
        let mut method = PollMethod::PerMember;
        let mut device_time = None;
        let mut device_state = None;

        if plan.prefer_device_state
            && self.device_state
            && !self.unsupported.contains("devicestate")
        {
            let (result, timing) = self.timed(|| self.get("devicestate", TimeoutClass::Ordinary));
            match result {
                Ok(value) => match parse_device_state(&value) {
                    Some(map) if !map.is_empty() => {
                        method = PollMethod::DeviceState;
                        device_time = map
                            .get("timestamp")
                            .and_then(Value::as_str)
                            .and_then(parse_timestamp);
                        device_state = Some(DeviceStateSample {
                            values: map,
                            timing,
                            device_time,
                        });
                    }
                    _ => members.push(RawMember {
                        name: "devicestate".into(),
                        value: Some(value),
                        state: ObsState::Error,
                        observed_at: Some(timing.at),
                        mono_ns: Some(timing.mono),
                        latency_ms: Some(timing.latency_ms),
                        error: Some("devicestate is not a Name/Value list".into()),
                    }),
                },
                Err(err) if err.is_unknown_member() => {
                    self.unsupported.insert("devicestate".into());
                    self.device_state = false;
                }
                Err(err) => {
                    members.push(RawMember {
                        name: "devicestate".into(),
                        value: None,
                        state: ObsState::Error,
                        observed_at: Some(timing.at),
                        mono_ns: Some(timing.mono),
                        latency_ms: Some(timing.latency_ms),
                        error: Some(err.to_string()),
                    });
                    if link_failure(&err) {
                        failure = Some(err);
                    }
                }
            }
        }

        let previous = self.state.clone();
        let is_moving = self.state_or_member(
            "ismoving",
            true,
            &previous.is_moving,
            as_bool,
            device_state.as_ref(),
            &mut members,
            &mut failure,
        );
        let position = if absolute {
            self.state_or_member(
                "position",
                true,
                &previous.position,
                as_i32,
                device_state.as_ref(),
                &mut members,
                &mut failure,
            )
        } else {
            Observed::unsupported(Source::Property)
        };
        let temperature_c = self.state_or_member(
            "temperature",
            false,
            &previous.temperature_c,
            as_f64,
            device_state.as_ref(),
            &mut members,
            &mut failure,
        );
        let tempcomp_due = self.tempcomp_dirty
            || previous.temp_comp.mono_ns.is_none()
            || plan
                .extra_members
                .iter()
                .any(|m| m.eq_ignore_ascii_case("tempcomp"));
        let temp_comp = if !temp_comp_available {
            Observed::unsupported(Source::Property)
        } else if tempcomp_due {
            let observed = self.read_member(
                "tempcomp",
                false,
                &previous.temp_comp,
                as_bool,
                &mut members,
                &mut failure,
            );
            if observed.state == ObsState::Fresh {
                self.tempcomp_dirty = false;
            }
            observed
        } else {
            let overdue = previous
                .temp_comp
                .age_ms(self.clock.now_mono_ns())
                .is_some_and(|age| age > TEMPCOMP_STALE_AFTER.as_millis() as u64);
            if overdue {
                previous.temp_comp.clone().into_stale()
            } else {
                previous.temp_comp.clone()
            }
        };

        self.state = FocuserState {
            position,
            is_moving,
            temperature_c,
            temp_comp,
        };
        if let Some(err) = failure {
            return Err(err);
        }
        Ok(RawPoll {
            method,
            started_mono_ns: started,
            finished_mono_ns: self.clock.now_mono_ns(),
            device_time,
            members,
        })
    }

    fn supported_actions(&mut self) -> DeviceResult<Vec<String>> {
        let value = self.get("supportedactions", TimeoutClass::Ordinary)?;
        Ok(value
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default())
    }

    fn action(&mut self, name: &str, params: &str) -> DeviceResult<String> {
        let value = self.put(
            "action",
            &[
                ("Action", name.to_string()),
                ("Parameters", params.to_string()),
            ],
            TimeoutClass::Ordinary,
        )?;
        Ok(match value {
            Value::String(text) => text,
            Value::Null => String::new(),
            other => other.to_string(),
        })
    }

    fn abort_path(&self) -> Option<Arc<dyn AbortPath>> {
        Some(Arc::new(AlpacaFocuserAbort {
            transport: Arc::clone(&self.abort_transport),
        }))
    }
}

impl FocuserDriver for AlpacaFocuser {
    fn capabilities(&mut self) -> DeviceResult<FocuserCapabilities> {
        self.caps().cloned()
    }

    fn state(&mut self) -> DeviceResult<FocuserState> {
        self.require_connected()?;
        Ok(self.state.clone())
    }

    fn move_absolute(&mut self, target: i32) -> DeviceResult<()> {
        self.require_connected()?;
        if !self.caps()?.absolute {
            return Err(DeviceError::invalid_operation(
                "a relative focuser takes step counts, not positions",
            ));
        }
        self.put(
            "move",
            &[("Position", target.to_string())],
            TimeoutClass::Long,
        )
        .map(|_| ())
    }

    fn move_relative(&mut self, steps: i32) -> DeviceResult<()> {
        self.require_connected()?;
        if self.caps()?.absolute {
            return Err(DeviceError::invalid_operation(
                "an absolute focuser takes positions, not step counts",
            ));
        }
        // Alpaca `Move` on a relative focuser takes a step count.
        self.put(
            "move",
            &[("Position", steps.to_string())],
            TimeoutClass::Long,
        )
        .map(|_| ())
    }

    fn halt(&mut self) -> DeviceResult<()> {
        self.put("halt", &[], TimeoutClass::Long).map(|_| ())
    }

    fn set_temp_comp(&mut self, on: bool) -> DeviceResult<()> {
        self.require_connected()?;
        if !self.caps()?.temp_comp_available {
            return Err(DeviceError::unsupported(
                "TempCompAvailable is false on this focuser",
            ));
        }
        self.tempcomp_dirty = true;
        self.put(
            "tempcomp",
            &[("TempComp", alpaca_bool(on))],
            TimeoutClass::Ordinary,
        )
        .map(|_| ())
    }

    fn sync_position(&mut self, _position: i32) -> DeviceResult<()> {
        Err(DeviceError::unsupported(
            "the Alpaca focuser interface has no position sync",
        ))
    }
}

/// `halt` on the abort agent (D4).
struct AlpacaFocuserAbort {
    transport: Arc<dyn AlpacaTransport>,
}

impl AbortPath for AlpacaFocuserAbort {
    fn abort(&self) -> DeviceResult<()> {
        self.transport
            .put("halt", &[], TimeoutClass::Long)
            .map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::testing::fake_transport::{FakeAlpacaTransport, FakeReply, Method};
    use crate::equipment::testing::manual_clock::ManualClock;
    use serde_json::json;

    /// A scripted Platform 6 focuser: InterfaceVersion 3, no DeviceState,
    /// absolute, temperature, no StepSize, no TempComp (the ESP32 focuser
    /// plan's profile).
    fn platform6() -> Arc<FakeAlpacaTransport> {
        let fake = FakeAlpacaTransport::new().shared();
        fake.on_get("interfaceversion", FakeReply::value(3))
            .on_put("connected", FakeReply::ok())
            .on_get("connected", FakeReply::value(true))
            .on_get("name", FakeReply::value("ESP32 Focuser"))
            .on_get("driverversion", FakeReply::value("0.3"))
            .on_get("absolute", FakeReply::value(true))
            .on_get("maxstep", FakeReply::value(40_000))
            .on_get("maxincrement", FakeReply::value(40_000))
            .on_get("stepsize", FakeReply::not_implemented())
            .on_get("tempcompavailable", FakeReply::value(false))
            .on_get("position", FakeReply::value(12_000))
            .on_get("ismoving", FakeReply::value(false))
            .on_get("temperature", FakeReply::value(8.5))
            .on_put("move", FakeReply::ok())
            .on_put("halt", FakeReply::ok());
        fake
    }

    fn driver(fake: &Arc<FakeAlpacaTransport>, clock: &Arc<ManualClock>) -> AlpacaFocuser {
        AlpacaFocuser::new(
            "http://esp32.local:80/",
            0,
            fake.clone(),
            fake.clone(),
            clock.clone(),
        )
        .with_connect_poll_interval(Duration::ZERO)
    }

    fn plan(extra: &[&str]) -> PollPlan {
        PollPlan {
            prefer_device_state: true,
            extra_members: extra.iter().map(|m| m.to_string()).collect(),
            ..PollPlan::default()
        }
    }

    #[test]
    fn a_platform6_focuser_connects_with_connected_true_and_polls_per_member() {
        let fake = platform6();
        let clock = Arc::new(ManualClock::new());
        let mut drv = driver(&fake, &clock);
        let report = drv.connect(Duration::from_secs(30)).unwrap();
        assert_eq!(report.path, ConnectPath::Platform6);
        assert_eq!(report.interface_version, Some(3));
        assert!(!report.device_state);
        assert!(report.unsupported_members.contains(&"devicestate".into()));
        assert!(report.unsupported_members.contains(&"stepsize".into()));
        assert_eq!(
            fake.calls_to(Method::Put, "connect"),
            0,
            "no Platform 7 connect"
        );
        assert_eq!(fake.calls_to(Method::Get, "devicestate"), 0);
        let put = fake
            .calls()
            .into_iter()
            .find(|c| c.method == Method::Put && c.member == "connected")
            .unwrap();
        assert_eq!(
            put.params,
            vec![("Connected".to_string(), "True".to_string())]
        );
        assert_eq!(put.class, TimeoutClass::Connect);
        assert_eq!(drv.identity().name, "ESP32 Focuser");
        assert_eq!(
            drv.identity().identity_key,
            "alpaca:esp32.local:80/focuser/0"
        );
        assert_eq!(
            drv.identity().setup_url.as_deref(),
            Some("http://esp32.local:80/setup/v1/focuser/0/setup")
        );
        let caps = drv.capabilities().unwrap();
        assert!(caps.absolute);
        assert_eq!(caps.max_step, Some(40_000));
        assert_eq!(caps.step_size_um, None);
        assert!(!caps.temp_comp_available && !caps.can_sync);

        fake.clear_calls();
        let poll = drv.poll(&plan(&["tempcomp"])).unwrap();
        assert_eq!(poll.method, PollMethod::PerMember);
        let members: Vec<_> = fake.calls().into_iter().map(|c| c.member).collect();
        assert_eq!(members, vec!["ismoving", "position", "temperature"]);
        let st = drv.state().unwrap();
        assert_eq!(st.position.value, Some(12_000));
        assert_eq!(st.position.source, Source::Property);
        assert_eq!(st.temperature_c.value, Some(8.5));
        assert_eq!(st.temp_comp.state, ObsState::Unsupported);
        assert_eq!(
            drv.sync_position(5).unwrap_err().class,
            ErrorClass::Unsupported,
            "R16: Alpaca has no sync"
        );
    }

    #[test]
    fn a_platform7_focuser_uses_connect_connecting_and_one_devicestate_get() {
        let fake = FakeAlpacaTransport::new().shared();
        fake.on_get("interfaceversion", FakeReply::value(4))
            .on_put("connect", FakeReply::ok())
            .queue_get(
                "connecting",
                [FakeReply::value(true), FakeReply::value(true)],
            )
            .on_get("connecting", FakeReply::value(false))
            .on_get("connected", FakeReply::value(true))
            .on_get("absolute", FakeReply::value(true))
            .on_get("maxstep", FakeReply::value(50_000))
            .on_get("maxincrement", FakeReply::value(1_000))
            .on_get("stepsize", FakeReply::value(4.5))
            .on_get("tempcompavailable", FakeReply::value(true))
            .on_get("tempcomp", FakeReply::value(true))
            .on_get(
                "devicestate",
                FakeReply::value(json!([
                    {"Name": "IsMoving", "Value": true},
                    {"Name": "Position", "Value": 20_100},
                    {"Name": "Temperature", "Value": 3.25},
                    {"Name": "TimeStamp", "Value": "2026-01-01T00:00:01.5Z"}
                ])),
            );
        let clock = Arc::new(ManualClock::new());
        let mut drv = driver(&fake, &clock);
        let report = drv.connect(Duration::from_secs(30)).unwrap();
        assert_eq!(report.path, ConnectPath::Platform7);
        assert!(report.device_state);
        assert_eq!(fake.calls_to(Method::Get, "connecting"), 3);
        assert_eq!(fake.calls_to(Method::Put, "connected"), 0);
        let caps = drv.capabilities().unwrap();
        assert_eq!(caps.max_increment, Some(1_000));
        assert_eq!(caps.step_size_um, Some(4.5));

        fake.clear_calls();
        let poll = drv.poll(&plan(&[])).unwrap();
        assert_eq!(poll.method, PollMethod::DeviceState);
        assert_eq!(
            poll.device_time,
            Some(
                Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 1).unwrap()
                    + chrono::Duration::milliseconds(500)
            )
        );
        // One devicestate GET, plus TempComp on the first poll only.
        let members: Vec<_> = fake.calls().into_iter().map(|c| c.member).collect();
        assert_eq!(members, vec!["devicestate", "tempcomp"]);
        let st = drv.state().unwrap();
        assert_eq!(st.is_moving.value, Some(true));
        assert_eq!(st.position.value, Some(20_100));
        assert_eq!(st.position.source, Source::DeviceState);
        assert_eq!(st.position.device_time, poll.device_time);
        assert_eq!(st.temp_comp.value, Some(true));
        fake.clear_calls();
        drv.poll(&plan(&[])).unwrap();
        let members: Vec<_> = fake.calls().into_iter().map(|c| c.member).collect();
        assert_eq!(
            members,
            vec!["devicestate"],
            "TempComp waits for its 10 s slot"
        );
        fake.clear_calls();
        drv.poll(&plan(&["tempcomp"])).unwrap();
        assert_eq!(fake.calls_to(Method::Get, "tempcomp"), 1);
    }

    #[test]
    fn missing_devicestate_entries_fall_back_to_member_reads() {
        let fake = FakeAlpacaTransport::new().shared();
        fake.on_get("interfaceversion", FakeReply::value(4))
            .on_put("connect", FakeReply::ok())
            .on_get("connecting", FakeReply::value(false))
            .on_get("connected", FakeReply::value(true))
            .on_get("absolute", FakeReply::value(true))
            .on_get("maxstep", FakeReply::value(50_000))
            .on_get("tempcompavailable", FakeReply::value(false))
            .on_get("position", FakeReply::value(77))
            .on_get("temperature", FakeReply::value_not_set())
            .on_get(
                "devicestate",
                FakeReply::value(json!([{"Name": "IsMoving", "Value": false}])),
            );
        let clock = Arc::new(ManualClock::new());
        let mut drv = driver(&fake, &clock);
        drv.connect(Duration::from_secs(30)).unwrap();
        assert_eq!(
            drv.capabilities().unwrap().max_increment,
            None,
            "unknown member"
        );
        fake.clear_calls();
        drv.poll(&plan(&[])).unwrap();
        let members: Vec<_> = fake.calls().into_iter().map(|c| c.member).collect();
        assert_eq!(members, vec!["devicestate", "position", "temperature"]);
        let st = drv.state().unwrap();
        assert_eq!(st.position.value, Some(77));
        assert_eq!(st.position.source, Source::Property);
        assert_eq!(
            st.temperature_c.state,
            ObsState::NotSet,
            "ValueNotSet is unknown"
        );
        assert_eq!(st.temperature_c.value, None);
    }

    #[test]
    fn temperature_observations_carry_ages_and_go_stale_on_failure() {
        let fake = platform6();
        let clock = Arc::new(ManualClock::new());
        let mut drv = driver(&fake, &clock);
        drv.connect(Duration::from_secs(30)).unwrap();
        clock.advance_ms(1_000);
        drv.poll(&plan(&[])).unwrap();
        let first = drv.state().unwrap().temperature_c;
        assert!(first.is_fresh());
        assert_eq!(first.mono_ns, Some(1_000_000_000));
        assert_eq!(first.latency_ms, Some(0));
        assert!(first.observed_at.is_some());
        clock.advance_ms(4_000);
        assert_eq!(first.age_ms(clock.now_mono_ns()), Some(4_000));

        // A failed read keeps the value but never presents it as fresh.
        fake.queue_get("temperature", [FakeReply::http(500, "sensor read failed")]);
        let poll = drv.poll(&plan(&[])).unwrap();
        assert!(poll.any_failed());
        let stale = drv.state().unwrap().temperature_c;
        assert_eq!(stale.state, ObsState::Stale);
        assert_eq!(stale.value, Some(8.5));
        assert_eq!(stale.mono_ns, Some(1_000_000_000), "age keeps growing");
        assert_eq!(stale.age_ms(clock.now_mono_ns()), Some(4_000));

        // NotImplemented is memoised: no more temperature GETs.
        fake.queue_get("temperature", [FakeReply::not_implemented()]);
        drv.poll(&plan(&[])).unwrap();
        assert_eq!(
            drv.state().unwrap().temperature_c.state,
            ObsState::Unsupported
        );
        fake.clear_calls();
        drv.poll(&plan(&[])).unwrap();
        assert_eq!(fake.calls_to(Method::Get, "temperature"), 0);
    }

    #[test]
    fn a_link_failure_skips_the_rest_of_the_tick_and_fails_the_poll() {
        let fake = platform6();
        let clock = Arc::new(ManualClock::new());
        let mut drv = driver(&fake, &clock);
        drv.connect(Duration::from_secs(30)).unwrap();
        drv.poll(&plan(&[])).unwrap();
        fake.clear_calls();
        fake.queue_get("ismoving", [FakeReply::Timeout]);
        let err = drv.poll(&plan(&[])).unwrap_err();
        assert_eq!(err.class, ErrorClass::Timeout);
        assert_eq!(
            fake.calls().len(),
            1,
            "one timeout per tick, not one per member"
        );
        let st = drv.state().unwrap();
        assert_eq!(st.position.state, ObsState::Stale);
        assert_eq!(st.position.value, Some(12_000));
        // IsMoving is mandatory: a 400 is an error, never "unsupported".
        fake.queue_get("ismoving", [FakeReply::http(400, "bad request")]);
        drv.poll(&plan(&[])).unwrap();
        assert_eq!(drv.state().unwrap().is_moving.state, ObsState::Stale);
        drv.poll(&plan(&[])).unwrap();
        assert!(drv.state().unwrap().is_moving.is_fresh());
    }

    #[test]
    fn moves_halts_and_tempcomp_use_exact_case_forms_and_timeout_classes() {
        let fake = FakeAlpacaTransport::new().shared();
        fake.on_get("interfaceversion", FakeReply::value(3))
            .on_put("connected", FakeReply::ok())
            .on_get("connected", FakeReply::value(true))
            .on_get("absolute", FakeReply::value(false))
            .on_get("maxstep", FakeReply::value(10_000))
            .on_get("maxincrement", FakeReply::value(500))
            .on_get("tempcompavailable", FakeReply::value(true))
            .on_put("move", FakeReply::ok())
            .on_put("halt", FakeReply::ok())
            .on_put("tempcomp", FakeReply::ok())
            .on_get(
                "supportedactions",
                FakeReply::value(json!(["PositionSource"])),
            )
            .on_put("action", FakeReply::value("restored"));
        let clock = Arc::new(ManualClock::new());
        let mut drv = driver(&fake, &clock);
        drv.connect(Duration::from_secs(30)).unwrap();
        assert!(!drv.capabilities().unwrap().absolute);
        assert!(drv.move_absolute(100).is_err(), "relative takes steps");
        drv.move_relative(-250).unwrap();
        drv.set_temp_comp(false).unwrap();
        drv.abort_path().unwrap().abort().unwrap();
        assert_eq!(drv.supported_actions().unwrap(), vec!["PositionSource"]);
        assert_eq!(drv.action("PositionSource", "").unwrap(), "restored");
        let calls: Vec<_> = fake
            .calls()
            .into_iter()
            .filter(|c| c.method == Method::Put && c.member != "connected")
            .map(|c| (c.member, c.params, c.class))
            .collect();
        assert_eq!(
            calls,
            vec![
                (
                    "move".to_string(),
                    vec![("Position".to_string(), "-250".to_string())],
                    TimeoutClass::Long
                ),
                (
                    "tempcomp".to_string(),
                    vec![("TempComp".to_string(), "False".to_string())],
                    TimeoutClass::Ordinary
                ),
                ("halt".to_string(), vec![], TimeoutClass::Long),
                (
                    "action".to_string(),
                    vec![
                        ("Action".to_string(), "PositionSource".to_string()),
                        ("Parameters".to_string(), String::new())
                    ],
                    TimeoutClass::Ordinary
                ),
            ]
        );
        // A relative focuser never reads Position.
        fake.clear_calls();
        fake.on_get("ismoving", FakeReply::value(false));
        drv.poll(&plan(&[])).unwrap();
        assert_eq!(fake.calls_to(Method::Get, "position"), 0);
        assert_eq!(drv.state().unwrap().position.state, ObsState::Unsupported);
    }

    #[test]
    fn connect_failures_are_reported_and_unique_ids_key_the_identity() {
        let fake = FakeAlpacaTransport::new().shared();
        fake.on_get("interfaceversion", FakeReply::Timeout)
            .on_put("connected", FakeReply::ok())
            .on_get("connected", FakeReply::value(false));
        let clock = Arc::new(ManualClock::new());
        let mut drv = driver(&fake, &clock);
        let err = drv.connect(Duration::from_secs(1)).unwrap_err();
        assert_eq!(err.class, ErrorClass::NotConnected);
        assert_eq!(drv.identity().interface_version, Some(1));
        let connected_puts: Vec<_> = fake
            .calls()
            .into_iter()
            .filter(|c| c.method == Method::Put && c.member == "connected")
            .map(|c| c.params[0].1.clone())
            .collect();
        assert_eq!(connected_puts, vec!["True", "False"], "undone (§2.6)");

        let stuck = FakeAlpacaTransport::new().shared();
        stuck
            .on_get("interfaceversion", FakeReply::value(4))
            .on_put("connect", FakeReply::ok())
            .on_get("connecting", FakeReply::value(true))
            .on_put("disconnect", FakeReply::ok());
        let mut drv = AlpacaFocuser::new("http://h:1", 0, stuck.clone(), stuck.clone(), clock)
            .with_connect_poll_interval(Duration::from_millis(1))
            .with_unique_id(Some(" 7f1e-uid "));
        assert_eq!(drv.identity().identity_key, "alpaca:7f1e-uid");
        let err = drv.connect(Duration::from_millis(5)).unwrap_err();
        assert_eq!(err.class, ErrorClass::Timeout);
        // The manual clock never advances, so the attempt backstop ends it.
        assert_eq!(stuck.calls_to(Method::Get, "connecting"), 6);
        assert_eq!(stuck.calls_to(Method::Put, "disconnect"), 1);
    }

    /// Every request takes `per_call` of clock time.
    struct SlowTransport {
        inner: Arc<FakeAlpacaTransport>,
        clock: Arc<ManualClock>,
        per_call: Duration,
    }

    impl AlpacaTransport for SlowTransport {
        fn get(
            &self,
            member: &str,
            query: &[(&str, String)],
            class: TimeoutClass,
        ) -> DeviceResult<Value> {
            self.clock.advance(self.per_call);
            self.inner.get(member, query, class)
        }

        fn put(
            &self,
            member: &str,
            form: &[(&str, String)],
            class: TimeoutClass,
        ) -> DeviceResult<Value> {
            self.clock.advance(self.per_call);
            self.inner.put(member, form, class)
        }
    }

    #[test]
    fn a_failed_connect_disconnects_and_the_wait_is_bounded_by_elapsed_time() {
        // Platform 7, `connecting` never clears, every request takes 2 s: a
        // 30 s budget allows 15 GETs (request latency counts), then the
        // accepted connect is undone with PUT disconnect.
        let fake = FakeAlpacaTransport::new().shared();
        fake.on_get("interfaceversion", FakeReply::value(4))
            .on_put("connect", FakeReply::ok())
            .on_get("connecting", FakeReply::value(true))
            .on_put("disconnect", FakeReply::ok());
        let clock = Arc::new(ManualClock::new());
        let slow = Arc::new(SlowTransport {
            inner: fake.clone(),
            clock: clock.clone(),
            per_call: Duration::from_secs(2),
        });
        let mut drv = AlpacaFocuser::new("http://h:1", 0, slow, fake.clone(), clock.clone())
            .with_connect_poll_interval(Duration::ZERO);
        let err = drv.connect(Duration::from_secs(30)).unwrap_err();
        assert_eq!(err.class, ErrorClass::Timeout);
        assert_eq!(fake.calls_to(Method::Get, "connecting"), 15);
        assert_eq!(fake.calls_to(Method::Put, "disconnect"), 1);
        assert_eq!(fake.calls_to(Method::Put, "connected"), 0);
        assert!(drv.poll(&plan(&[])).is_err(), "left disconnected");

        // A link failure while probing capabilities on Platform 7 is undone
        // with PUT disconnect, not Connected=False.
        let fake = FakeAlpacaTransport::new().shared();
        fake.on_get("interfaceversion", FakeReply::value(4))
            .on_put("connect", FakeReply::ok())
            .on_get("connecting", FakeReply::value(false))
            .on_get("connected", FakeReply::value(true))
            .on_get("absolute", FakeReply::Timeout)
            .on_put("disconnect", FakeReply::ok());
        let mut drv = driver(&fake, &clock);
        assert_eq!(
            drv.connect(Duration::from_secs(30)).unwrap_err().class,
            ErrorClass::Timeout
        );
        assert_eq!(fake.calls_to(Method::Put, "disconnect"), 1);
        assert_eq!(fake.calls_to(Method::Put, "connected"), 0);

        // A connect write that timed out may have reached the device.
        let fake = FakeAlpacaTransport::new().shared();
        fake.on_get("interfaceversion", FakeReply::value(4))
            .on_put("connect", FakeReply::Timeout)
            .on_put("disconnect", FakeReply::ok());
        let mut drv = driver(&fake, &clock);
        assert!(drv.connect(Duration::from_secs(30)).is_err());
        assert_eq!(fake.calls_to(Method::Put, "disconnect"), 1);

        // A connect the device refused has nothing to undo.
        let fake = FakeAlpacaTransport::new().shared();
        fake.on_get("interfaceversion", FakeReply::value(3))
            .on_put("connected", FakeReply::alpaca_error(0x500, "motor fault"));
        let mut drv = driver(&fake, &clock);
        assert!(drv.connect(Duration::from_secs(30)).is_err());
        assert_eq!(fake.calls_to(Method::Put, "connected"), 1);
    }

    #[test]
    fn timestamps_parse_permissively() {
        let z = parse_timestamp("2026-03-01T10:00:00Z").unwrap();
        assert_eq!(z, Utc.with_ymd_and_hms(2026, 3, 1, 10, 0, 0).unwrap());
        let offset = parse_timestamp("2026-03-01T12:00:00+02:00").unwrap();
        assert_eq!(offset, z);
        let local = parse_timestamp("2026-03-01T10:00:00.250").unwrap();
        let expected = Local
            .from_local_datetime(
                &NaiveDateTime::parse_from_str("2026-03-01T10:00:00.250", "%Y-%m-%dT%H:%M:%S%.f")
                    .unwrap(),
            )
            .earliest()
            .unwrap()
            .with_timezone(&Utc);
        assert_eq!(local, expected);
        assert_eq!(parse_timestamp("yesterday"), None);
        assert_eq!(parse_device_state(&json!({"Name": "x"})), None);
        assert_eq!(host_port("https://h"), ("h".to_string(), 443));
        assert_eq!(
            host_port("http://10.0.0.2:11111/"),
            ("10.0.0.2".to_string(), 11111)
        );
    }
}

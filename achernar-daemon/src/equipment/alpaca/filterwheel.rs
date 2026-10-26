//! Alpaca filter-wheel driver (spec §3.4, §2.6, §7.1 R17). Owner: WP1A-6.
//!
//! An independent implementation of the ASCOM `IFilterWheelV3` member set
//! over [`AlpacaTransport`]: `position` (0-based; -1 while moving), `names`,
//! `focusoffsets`, plus the common members. Behaviour facts come from the
//! ASCOM Alpaca API reference and Master Interfaces (MIT).
//!
//! Connect (§2.6): `interfaceversion` first (a failed read counts as 1).
//! FilterWheel is Platform 7 from interface version 3: PUT `connect`, then
//! poll `connecting` every 250 ms up to the connect timeout; a server that
//! does not know `connect` falls back to the Platform 6 path, PUT
//! `Connected=true` then GET `connected`. Identity and the slot table
//! (`names`, `focusoffsets`) are read once at connect; the slot count is the
//! device's.
//!
//! Polling (§2.7): one `devicestate` GET when the device offers it (the
//! `TimeStamp` becomes the reading's device time), else one `position` GET.
//! Every reading is timestamped at the midpoint of its request. ValueNotSet
//! is unknown, never an error; a failed read keeps the last value as
//! `Stale` (C-2).
//!
//! A filter wheel has no halt, so there is no abort path (C-4).
//!
//! The connect sequence and the `devicestate` parsing are local to this
//! driver because WP1A-1's shared helpers (`alpaca/client.rs`,
//! `alpaca/devicestate.rs`) are built in parallel; they should replace these
//! once available (change note).

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Local, NaiveDateTime, TimeZone, Utc};
use serde_json::Value;

use crate::equipment::clock::duration_ns;
use crate::equipment::snapshot::SlotInfo;
use crate::equipment::traits::{
    AbortPath, AlpacaTransport, Clock, ConnectPath, ConnectReport, Device, FilterWheelDriver,
    PollMethod, PollPlan, RawMember, RawPoll, TimeoutClass,
};
use crate::equipment::types::{
    ascom_error, DeviceError, DeviceIdentity, DeviceResult, Endpoint, ObsState, Observed, Source,
};

/// The Alpaca device type in URLs.
pub const DEVICE_TYPE: &str = "filterwheel";

/// FilterWheel interface version from which Platform 7 `connect` applies.
pub const PLATFORM7_INTERFACE_VERSION: i32 = 3;

/// `connecting` poll interval (§2.6 step 4).
pub const CONNECT_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// The Alpaca setup page for a filter wheel endpoint.
pub fn setup_url(endpoint: &Endpoint) -> Option<String> {
    match endpoint {
        Endpoint::AlpacaHttp { base, number, .. } => Some(format!(
            "{}/setup/v1/{DEVICE_TYPE}/{number}/setup",
            base.trim_end_matches('/')
        )),
        _ => None,
    }
}

/// Timing of one request.
#[derive(Debug, Clone, Copy)]
struct Timing {
    started_mono_ns: u64,
    finished_mono_ns: u64,
    observed_at: DateTime<Utc>,
    mono_ns: u64,
    latency_ms: u32,
}

fn as_bool(value: &Value, member: &str) -> DeviceResult<bool> {
    value
        .as_bool()
        .ok_or_else(|| DeviceError::driver(format!("{member} returned {value}, not a boolean")))
}

fn as_int(value: &Value) -> Option<i64> {
    value.as_i64().or_else(|| {
        value
            .as_f64()
            .filter(|v| v.is_finite() && v.fract() == 0.0)
            .map(|v| v as i64)
    })
}

fn as_text(value: &Value) -> Option<String> {
    value
        .as_str()
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
}

/// Parse a DeviceState `TimeStamp` permissively (§2.14): an explicit offset
/// or `Z`, else a zone-less local time.
pub fn parse_timestamp(text: &str) -> Option<DateTime<Utc>> {
    let text = text.trim();
    if let Ok(at) = DateTime::parse_from_rfc3339(text) {
        return Some(at.with_timezone(&Utc));
    }
    [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.f",
    ]
    .iter()
    .find_map(|format| NaiveDateTime::parse_from_str(text, format).ok())
    .and_then(|naive| Local.from_local_datetime(&naive).earliest())
    .map(|at| at.with_timezone(&Utc))
}

/// `devicestate` entries by lower-case name.
fn device_state_entries(value: &Value) -> DeviceResult<Vec<(String, Value)>> {
    let items = value
        .as_array()
        .ok_or_else(|| DeviceError::driver(format!("devicestate returned {value}, not a list")))?;
    Ok(items
        .iter()
        .filter_map(|item| {
            let name = item
                .get("Name")
                .or_else(|| item.get("name"))
                .and_then(Value::as_str)?;
            let value = item
                .get("Value")
                .or_else(|| item.get("value"))
                .cloned()
                .unwrap_or(Value::Null);
            Some((name.to_ascii_lowercase(), value))
        })
        .collect())
}

/// The Alpaca filter-wheel driver. Synchronous; called only from its worker
/// thread (D3).
pub struct AlpacaFilterWheel {
    identity: DeviceIdentity,
    transport: Arc<dyn AlpacaTransport>,
    clock: Arc<dyn Clock>,
    connect_poll: Duration,
    interface_version: Option<i32>,
    platform7: bool,
    device_state: bool,
    unsupported: BTreeSet<String>,
    supported_actions: Vec<String>,
    slots: Vec<SlotInfo>,
    position: Observed<u32>,
    last_error: Option<DeviceError>,
    connected: bool,
}

impl AlpacaFilterWheel {
    /// `identity` carries the endpoint and the identity key (§2.6); connect
    /// fills in the driver's claims.
    pub fn new(
        identity: DeviceIdentity,
        transport: Arc<dyn AlpacaTransport>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            identity,
            transport,
            clock,
            connect_poll: CONNECT_POLL_INTERVAL,
            interface_version: None,
            platform7: false,
            device_state: false,
            unsupported: BTreeSet::new(),
            supported_actions: Vec::new(),
            slots: Vec::new(),
            position: Observed::not_set(Source::Property),
            last_error: None,
            connected: false,
        }
    }

    /// Real-time pause between `connecting` polls (tests use zero; the poll
    /// count stays bounded by the timeout at 250 ms per poll).
    pub fn with_connect_poll(mut self, interval: Duration) -> Self {
        self.connect_poll = interval;
        self
    }

    pub fn interface_version(&self) -> Option<i32> {
        self.interface_version
    }

    /// True when polls use one `devicestate` GET.
    pub fn uses_device_state(&self) -> bool {
        self.device_state
    }

    pub fn unsupported_members(&self) -> Vec<String> {
        self.unsupported.iter().cloned().collect()
    }

    /// The latest position observation; `moving` is a fresh read of -1.
    pub fn observed_position(&self) -> (Observed<u32>, bool) {
        let moving = self.position.state == ObsState::Fresh && self.position.value.is_none();
        (self.position.clone(), moving)
    }

    fn timed<T>(&self, call: impl FnOnce() -> T) -> (T, Timing) {
        let started_utc = self.clock.now_utc();
        let started = self.clock.now_mono_ns();
        let result = call();
        let finished_utc = self.clock.now_utc();
        let finished = self.clock.now_mono_ns();
        let elapsed = finished.saturating_sub(started);
        let observed_at = started_utc + (finished_utc - started_utc) / 2;
        (
            result,
            Timing {
                started_mono_ns: started,
                finished_mono_ns: finished,
                observed_at,
                mono_ns: started + elapsed / 2,
                latency_ms: u32::try_from(elapsed / 1_000_000).unwrap_or(u32::MAX),
            },
        )
    }

    fn get(&self, member: &str, class: TimeoutClass) -> DeviceResult<Value> {
        self.transport.get(member, &[], class)
    }

    /// An optional member: unknown members are memoised as unsupported.
    fn optional(&mut self, member: &str, report: &mut ConnectReport) -> Option<Value> {
        match self.get(member, TimeoutClass::Ordinary) {
            Ok(value) => Some(value),
            Err(err) if err.is_unknown_member() => {
                self.unsupported.insert(member.to_string());
                None
            }
            Err(err) => {
                report
                    .warnings
                    .push(format!("{member} could not be read: {err}"));
                None
            }
        }
    }

    fn confirm_connected(&self) -> DeviceResult<()> {
        let value = self.get("connected", TimeoutClass::Connect)?;
        if as_bool(&value, "connected")? {
            Ok(())
        } else {
            Err(DeviceError::driver(
                "the device reports Connected=false after connecting",
            ))
        }
    }

    fn connect_platform6(&self) -> DeviceResult<()> {
        self.transport.put(
            "connected",
            &[("Connected", "true".to_string())],
            TimeoutClass::Connect,
        )?;
        self.confirm_connected()
    }

    /// The connect request and its confirmation (§2.6): Platform 7 when
    /// `platform7` is set and the server knows `connect`, else Platform 6
    /// (`platform7` is then cleared).
    fn open_connection(
        &mut self,
        timeout: Duration,
        started: u64,
        report: &mut ConnectReport,
    ) -> DeviceResult<()> {
        if self.platform7 {
            if self.connect_platform7(timeout, started)? {
                return Ok(());
            }
            report
                .warnings
                .push("the server does not implement connect; using Connected=true".into());
            self.unsupported.insert("connect".into());
            self.platform7 = false;
        }
        self.connect_platform6()
    }

    /// PUT `connect`, then poll `connecting`. `Ok(false)` means the server
    /// does not know `connect` and the Platform 6 path should be used.
    fn connect_platform7(&self, timeout: Duration, started: u64) -> DeviceResult<bool> {
        match self.transport.put("connect", &[], TimeoutClass::Connect) {
            Ok(_) => {}
            Err(err) if err.is_unknown_member() => return Ok(false),
            Err(err) => return Err(err),
        }
        let max_polls = (timeout.as_millis() / CONNECT_POLL_INTERVAL.as_millis()).max(1);
        let deadline = started.saturating_add(duration_ns(timeout));
        for _ in 0..max_polls {
            let connecting = match self.get("connecting", TimeoutClass::Connect) {
                Ok(value) => as_bool(&value, "connecting")?,
                Err(err) if err.is_unknown_member() => false,
                Err(err) => return Err(err),
            };
            if !connecting {
                self.confirm_connected()?;
                return Ok(true);
            }
            if self.clock.now_mono_ns() >= deadline {
                break;
            }
            if !self.connect_poll.is_zero() {
                std::thread::sleep(self.connect_poll);
            }
        }
        Err(DeviceError::timeout(format!(
            "the filter wheel was still connecting after {} ms",
            timeout.as_millis()
        )))
    }

    fn read_identity(&mut self, report: &mut ConnectReport) {
        if let Some(name) = self.optional("name", report).as_ref().and_then(as_text) {
            self.identity.name = name;
        }
        self.identity.description = self
            .optional("description", report)
            .as_ref()
            .and_then(as_text);
        self.identity.driver_info = self
            .optional("driverinfo", report)
            .as_ref()
            .and_then(as_text);
        self.identity.driver_version = self
            .optional("driverversion", report)
            .as_ref()
            .and_then(as_text);
        self.supported_actions = self
            .optional("supportedactions", report)
            .and_then(|value| value.as_array().cloned())
            .map(|items| items.iter().filter_map(as_text).collect())
            .unwrap_or_default();
        self.identity.interface_version = self.interface_version;
        if self.identity.setup_url.is_none() {
            self.identity.setup_url = setup_url(&self.identity.endpoint);
        }
    }

    fn probe_device_state(&mut self, report: &mut ConnectReport) {
        self.device_state = false;
        if !self.platform7 {
            return;
        }
        if let Some(value) = self.optional("devicestate", report) {
            self.device_state = value.as_array().is_some_and(|items| !items.is_empty());
        }
    }

    /// The slot table (names and offsets); the slot count is the device's.
    fn read_slots(&mut self, report: &mut ConnectReport) -> DeviceResult<()> {
        let mut table = |member: &str| -> DeviceResult<Option<Vec<Value>>> {
            match self.get(member, TimeoutClass::Ordinary) {
                Ok(Value::Array(items)) => Ok(Some(items)),
                Ok(other) => Err(DeviceError::driver(format!(
                    "{member} returned {other}, not a list"
                ))),
                Err(err) if err.is_unknown_member() => {
                    self.unsupported.insert(member.to_string());
                    Ok(None)
                }
                Err(err) => Err(err),
            }
        };
        let names = table("names")?;
        let offsets = table("focusoffsets")?;
        let count = match (&names, &offsets) {
            (Some(names), Some(offsets)) => {
                if names.len() != offsets.len() {
                    report.warnings.push(format!(
                        "names has {} entries and focusoffsets {}; the slot count follows names",
                        names.len(),
                        offsets.len()
                    ));
                }
                names.len()
            }
            (Some(names), None) => names.len(),
            (None, Some(offsets)) => offsets.len(),
            (None, None) => return Err(DeviceError::driver(
                "the wheel implements neither names nor focusoffsets, so its slot count is unknown",
            )),
        };
        if count == 0 {
            return Err(DeviceError::driver("the wheel reports zero slots"));
        }
        self.slots = (0..count)
            .map(|i| SlotInfo {
                index0: u32::try_from(i).unwrap_or(u32::MAX),
                device_name: names.as_ref().and_then(|n| n.get(i)).and_then(as_text),
                device_focus_offset: offsets
                    .as_ref()
                    .and_then(|o| o.get(i))
                    .and_then(as_int)
                    .and_then(|v| i32::try_from(v).ok()),
            })
            .collect();
        Ok(())
    }

    fn slot_count(&self) -> u32 {
        u32::try_from(self.slots.len()).unwrap_or(u32::MAX)
    }

    /// Apply one position value (-1 = moving).
    fn apply_position(
        &mut self,
        value: &Value,
        timing: &Timing,
        source: Source,
        device_time: Option<DateTime<Utc>>,
    ) -> DeviceResult<()> {
        let slot = match as_int(value) {
            Some(-1) => None,
            Some(v) if v >= 0 && v < i64::from(self.slot_count()) => {
                Some(u32::try_from(v).unwrap_or(u32::MAX))
            }
            _ => {
                return Err(DeviceError::driver(format!(
                    "position {value} is outside -1..{}",
                    self.slot_count()
                )))
            }
        };
        self.position = Observed {
            value: slot,
            state: ObsState::Fresh,
            observed_at: Some(timing.observed_at),
            mono_ns: Some(timing.mono_ns),
            device_time,
            latency_ms: Some(timing.latency_ms),
            source,
        };
        self.last_error = None;
        Ok(())
    }

    fn member(
        &self,
        value: Option<Value>,
        timing: &Timing,
        error: Option<&DeviceError>,
    ) -> RawMember {
        RawMember {
            name: "position".into(),
            value,
            state: match error {
                None => ObsState::Fresh,
                Some(err) if err.is_not_set() => ObsState::NotSet,
                Some(_) => ObsState::Error,
            },
            observed_at: Some(timing.observed_at),
            mono_ns: Some(timing.mono_ns),
            latency_ms: Some(timing.latency_ms),
            error: error.map(|err| err.to_string()),
        }
    }

    /// A failed position read: ValueNotSet is unknown and not a failure;
    /// anything else keeps the last value as stale (C-2) and fails the poll.
    fn fail_position(
        &mut self,
        err: DeviceError,
        method: PollMethod,
        timing: Timing,
    ) -> DeviceResult<RawPoll> {
        if err.is_not_set() {
            self.position = Observed {
                observed_at: Some(timing.observed_at),
                mono_ns: Some(timing.mono_ns),
                latency_ms: Some(timing.latency_ms),
                ..Observed::not_set(Source::Property)
            };
            self.last_error = Some(err.clone());
            return Ok(RawPoll {
                method,
                started_mono_ns: timing.started_mono_ns,
                finished_mono_ns: timing.finished_mono_ns,
                device_time: None,
                members: vec![self.member(None, &timing, Some(&err))],
            });
        }
        self.position = self.position.clone().into_stale();
        self.last_error = Some(err.clone());
        Err(err)
    }

    fn poll_member(&mut self) -> DeviceResult<RawPoll> {
        let (result, timing) = self.timed(|| self.get("position", TimeoutClass::Ordinary));
        let value = match result {
            Ok(value) => value,
            Err(err) => return self.fail_position(err, PollMethod::PerMember, timing),
        };
        if let Err(err) = self.apply_position(&value, &timing, Source::Property, None) {
            return self.fail_position(err, PollMethod::PerMember, timing);
        }
        Ok(RawPoll {
            method: PollMethod::PerMember,
            started_mono_ns: timing.started_mono_ns,
            finished_mono_ns: timing.finished_mono_ns,
            device_time: None,
            members: vec![self.member(Some(value), &timing, None)],
        })
    }

    fn poll_device_state(&mut self) -> DeviceResult<RawPoll> {
        let (result, timing) = self.timed(|| self.get("devicestate", TimeoutClass::Ordinary));
        let entries = match result.and_then(|value| device_state_entries(&value)) {
            Ok(entries) => entries,
            Err(err) if err.is_unknown_member() => {
                self.unsupported.insert("devicestate".into());
                self.device_state = false;
                return self.poll_member();
            }
            Err(err) => return self.fail_position(err, PollMethod::DeviceState, timing),
        };
        let device_time = entries
            .iter()
            .find(|(name, _)| name == "timestamp")
            .and_then(|(_, value)| value.as_str())
            .and_then(parse_timestamp);
        let Some((_, value)) = entries.into_iter().find(|(name, _)| name == "position") else {
            // DeviceState with a missing entry: read the member this tick.
            return self.poll_member();
        };
        if let Err(err) = self.apply_position(&value, &timing, Source::DeviceState, device_time) {
            return self.fail_position(err, PollMethod::DeviceState, timing);
        }
        Ok(RawPoll {
            method: PollMethod::DeviceState,
            started_mono_ns: timing.started_mono_ns,
            finished_mono_ns: timing.finished_mono_ns,
            device_time,
            members: vec![self.member(Some(value), &timing, None)],
        })
    }
}

impl Device for AlpacaFilterWheel {
    fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    fn connect(&mut self, timeout: Duration) -> DeviceResult<ConnectReport> {
        let started = self.clock.now_mono_ns();
        let mut report = ConnectReport::default();
        self.unsupported.clear();

        let version = match self.get("interfaceversion", TimeoutClass::Connect) {
            Ok(value) => as_int(&value)
                .and_then(|v| i32::try_from(v).ok())
                .unwrap_or_else(|| {
                    report
                        .warnings
                        .push(format!("interfaceversion returned {value}; treated as 1"));
                    1
                }),
            Err(err) => {
                report.warnings.push(format!(
                    "interfaceversion could not be read ({err}); treated as 1"
                ));
                1
            }
        };
        self.interface_version = Some(version);
        report.interface_version = Some(version);

        // From the first connect request on, any failure disconnects the
        // device (§2.6), so a server that finishes connecting late keeps no
        // stray connection for this ClientID. `platform7` is set before the
        // attempt so that cleanup uses PUT `disconnect` on Platform 7.
        self.platform7 = version >= PLATFORM7_INTERFACE_VERSION;
        if let Err(err) = self.open_connection(timeout, started, &mut report) {
            let _ = self.disconnect();
            return Err(err);
        }
        report.path = if self.platform7 {
            ConnectPath::Platform7
        } else {
            ConnectPath::Platform6
        };

        self.read_identity(&mut report);
        self.probe_device_state(&mut report);
        if let Err(err) = self.read_slots(&mut report) {
            // Any failure disconnects the device (§2.6).
            let _ = self.disconnect();
            return Err(err);
        }
        self.connected = true;
        if let Err(err) = self.poll(&PollPlan {
            prefer_device_state: true,
            ..PollPlan::default()
        }) {
            report
                .warnings
                .push(format!("the initial position read failed: {err}"));
        }

        report.device_state = self.device_state;
        report.unsupported_members = self.unsupported_members();
        report.duration_ms = self.clock.now_mono_ns().saturating_sub(started) / 1_000_000;
        Ok(report)
    }

    fn disconnect(&mut self) -> DeviceResult<()> {
        self.connected = false;
        let result = if self.platform7 {
            match self.transport.put("disconnect", &[], TimeoutClass::Connect) {
                Err(err) if err.is_unknown_member() => self.transport.put(
                    "connected",
                    &[("Connected", "false".to_string())],
                    TimeoutClass::Connect,
                ),
                other => other,
            }
        } else {
            self.transport.put(
                "connected",
                &[("Connected", "false".to_string())],
                TimeoutClass::Connect,
            )
        };
        result.map(|_| ())
    }

    fn is_connected(&mut self) -> DeviceResult<bool> {
        let value = self.get("connected", TimeoutClass::Connect)?;
        as_bool(&value, "connected")
    }

    fn poll(&mut self, plan: &PollPlan) -> DeviceResult<RawPoll> {
        if self.device_state && plan.prefer_device_state {
            self.poll_device_state()
        } else {
            self.poll_member()
        }
    }

    fn supported_actions(&mut self) -> DeviceResult<Vec<String>> {
        let value = self.get("supportedactions", TimeoutClass::Ordinary)?;
        let actions: Vec<String> = value
            .as_array()
            .map(|items| items.iter().filter_map(as_text).collect())
            .unwrap_or_default();
        self.supported_actions = actions.clone();
        Ok(actions)
    }

    fn action(&mut self, name: &str, params: &str) -> DeviceResult<String> {
        let value = self.transport.put(
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

    /// A filter wheel has no halt (C-4).
    fn abort_path(&self) -> Option<Arc<dyn AbortPath>> {
        None
    }
}

impl FilterWheelDriver for AlpacaFilterWheel {
    fn slots(&mut self) -> DeviceResult<Vec<SlotInfo>> {
        if self.slots.is_empty() {
            return Err(DeviceError::not_connected(
                "the filter wheel's slots are read at connect",
            ));
        }
        Ok(self.slots.clone())
    }

    fn position(&mut self) -> DeviceResult<Option<u32>> {
        match self.position.state {
            // `None` while fresh is a read of -1: moving.
            ObsState::Fresh => Ok(self.position.value),
            ObsState::NotSet => Err(self.last_error.clone().unwrap_or_else(|| {
                DeviceError::from_alpaca_error(
                    ascom_error::VALUE_NOT_SET,
                    "no position has been read yet",
                )
            })),
            _ => Err(self
                .last_error
                .clone()
                .unwrap_or_else(|| DeviceError::driver("the position is unknown"))),
        }
    }

    fn set_position(&mut self, index0: u32) -> DeviceResult<()> {
        if !self.connected {
            return Err(DeviceError::not_connected(
                "the filter wheel is not connected",
            ));
        }
        if index0 >= self.slot_count() {
            return Err(DeviceError::invalid_value(format!(
                "slot {index0} does not exist; the wheel has {} slots",
                self.slot_count()
            )));
        }
        self.transport.put(
            "position",
            &[("Position", index0.to_string())],
            TimeoutClass::Ordinary,
        )?;
        // A successful set invalidates the member; it is re-read on the next
        // poll (C-2), and until then the wheel counts as moving.
        self.position = Observed {
            value: None,
            state: ObsState::Fresh,
            observed_at: Some(self.clock.now_utc()),
            mono_ns: Some(self.clock.now_mono_ns()),
            device_time: None,
            latency_ms: None,
            source: Source::Property,
        };
        self.last_error = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::filter::motion::{is_wrong_slot, FilterMoveOp};
    use crate::equipment::filter::{FilterSettings, MOVE_POLL_INTERVAL};
    use crate::equipment::ops::{OpProgress, Operation};
    use crate::equipment::testing::fake_transport::{FakeAlpacaTransport, FakeReply, Method};
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::types::{identity_key, DeviceKind, DriverId, ErrorClass, OpOutcome};
    use serde_json::json;

    fn identity() -> DeviceIdentity {
        DeviceIdentity::new(
            DeviceKind::FilterWheel,
            DriverId::Alpaca,
            Endpoint::AlpacaHttp {
                base: "http://127.0.0.1:11111".into(),
                device_type: DEVICE_TYPE.into(),
                number: 0,
            },
            identity_key::alpaca(None, "127.0.0.1", 11111, DEVICE_TYPE, 0),
            "Alpaca filter wheel",
        )
    }

    /// A Platform 6 wheel (interface version 2) with five named slots.
    fn platform6() -> Arc<FakeAlpacaTransport> {
        let fake = FakeAlpacaTransport::new().shared();
        fake.on_get("interfaceversion", FakeReply::value(2))
            .on_put("connected", FakeReply::ok())
            .on_get("connected", FakeReply::value(true))
            .on_get("name", FakeReply::value("EFW 5"))
            .on_get("description", FakeReply::value("Five-slot wheel"))
            .on_get("driverinfo", FakeReply::value("vendor driver"))
            .on_get("driverversion", FakeReply::value("1.2"))
            .on_get("supportedactions", FakeReply::value(json!([])))
            .on_get("names", FakeReply::value(json!(["L", "R", "G", "B", ""])))
            .on_get(
                "focusoffsets",
                FakeReply::value(json!([0, 15, 10, 20, 100])),
            )
            .on_get("position", FakeReply::value(0));
        fake
    }

    fn driver(fake: &Arc<FakeAlpacaTransport>, clock: &Arc<ManualClock>) -> AlpacaFilterWheel {
        AlpacaFilterWheel::new(
            identity(),
            Arc::clone(fake) as Arc<dyn AlpacaTransport>,
            Arc::clone(clock) as Arc<dyn Clock>,
        )
        .with_connect_poll(Duration::ZERO)
    }

    #[test]
    fn platform6_connect_reads_identity_and_the_slot_table() {
        let fake = platform6();
        let clock = Arc::new(ManualClock::new());
        let mut wheel = driver(&fake, &clock);
        let report = wheel.connect(Duration::from_secs(30)).unwrap();
        assert_eq!(report.path, ConnectPath::Platform6);
        assert_eq!(report.interface_version, Some(2));
        assert!(!report.device_state);
        assert_eq!(fake.calls_to(Method::Put, "connect"), 0);
        let put = fake
            .calls()
            .into_iter()
            .find(|c| c.method == Method::Put && c.member == "connected")
            .unwrap();
        assert_eq!(put.params, vec![("Connected".into(), "true".into())]);
        assert_eq!(wheel.identity().name, "EFW 5");
        assert_eq!(wheel.identity().driver_version.as_deref(), Some("1.2"));
        assert_eq!(
            wheel.identity().setup_url.as_deref(),
            Some("http://127.0.0.1:11111/setup/v1/filterwheel/0/setup")
        );
        let slots = wheel.slots().unwrap();
        assert_eq!(slots.len(), 5);
        assert_eq!(slots[1].device_name.as_deref(), Some("R"));
        assert_eq!(slots[4].device_name, None, "an empty name is no name");
        assert_eq!(slots[4].device_focus_offset, Some(100));
        assert_eq!(
            wheel.position().unwrap(),
            Some(0),
            "initial poll at connect"
        );
    }

    #[test]
    fn platform7_connect_polls_connecting_and_uses_devicestate() {
        let fake = platform6();
        fake.on_get("interfaceversion", FakeReply::value(3))
            .on_put("connect", FakeReply::ok())
            .queue_get(
                "connecting",
                [FakeReply::value(true), FakeReply::value(true)],
            )
            .on_get("connecting", FakeReply::value(false))
            .on_get(
                "devicestate",
                FakeReply::value(json!([
                    {"Name": "Position", "Value": 2},
                    {"Name": "TimeStamp", "Value": "2026-01-01T00:00:01.5Z"}
                ])),
            );
        let clock = Arc::new(ManualClock::new());
        let mut wheel = driver(&fake, &clock);
        let report = wheel.connect(Duration::from_secs(30)).unwrap();
        assert_eq!(report.path, ConnectPath::Platform7);
        assert!(report.device_state);
        assert_eq!(fake.calls_to(Method::Get, "connecting"), 3);
        assert_eq!(fake.calls_to(Method::Put, "connected"), 0);
        let raw = wheel
            .poll(&PollPlan {
                prefer_device_state: true,
                ..PollPlan::default()
            })
            .unwrap();
        assert_eq!(raw.method, PollMethod::DeviceState);
        assert_eq!(
            raw.device_time,
            Some(
                DateTime::parse_from_rfc3339("2026-01-01T00:00:01.5Z")
                    .unwrap()
                    .with_timezone(&Utc)
            )
        );
        assert_eq!(wheel.position().unwrap(), Some(2));
        assert_eq!(wheel.observed_position().0.source, Source::DeviceState);
        // Without the preference the member is read directly.
        let before = fake.calls_to(Method::Get, "position");
        wheel.poll(&PollPlan::default()).unwrap();
        assert_eq!(fake.calls_to(Method::Get, "position"), before + 1);
    }

    #[test]
    fn a_failed_interfaceversion_is_version_1_and_a_slow_connect_times_out() {
        let fake = platform6();
        fake.on_get("interfaceversion", FakeReply::http(500, "boom"));
        let clock = Arc::new(ManualClock::new());
        let mut wheel = driver(&fake, &clock);
        let report = wheel.connect(Duration::from_secs(5)).unwrap();
        assert_eq!(report.interface_version, Some(1));
        assert_eq!(report.path, ConnectPath::Platform6);
        assert!(report.warnings[0].contains("treated as 1"));

        let fake = platform6();
        fake.on_get("interfaceversion", FakeReply::value(3))
            .on_put("connect", FakeReply::ok())
            .on_get("connecting", FakeReply::value(true));
        let mut wheel = driver(&fake, &clock);
        let err = wheel.connect(Duration::from_secs(1)).unwrap_err();
        assert_eq!(err.class, ErrorClass::Timeout);
        assert_eq!(fake.calls_to(Method::Get, "connecting"), 4, "1 s at 250 ms");

        // A Platform 7 version on a server without `connect` falls back.
        let fake = platform6();
        fake.on_get("interfaceversion", FakeReply::value(3));
        let mut wheel = driver(&fake, &clock);
        let report = wheel.connect(Duration::from_secs(1)).unwrap();
        assert_eq!(report.path, ConnectPath::Platform6);
        assert!(report.unsupported_members.contains(&"connect".to_string()));
    }

    /// Review: every failure after the first connect request disconnects
    /// (§2.6), with the platform's own member.
    #[test]
    fn every_connect_failure_disconnects_the_device() {
        let clock = Arc::new(ManualClock::new());
        let disconnects = |fake: &FakeAlpacaTransport| {
            (
                fake.calls_to(Method::Put, "disconnect"),
                fake.calls()
                    .into_iter()
                    .filter(|c| {
                        c.method == Method::Put
                            && c.member == "connected"
                            && c.params == vec![("Connected".into(), "false".into())]
                    })
                    .count(),
            )
        };

        // Platform 7: still connecting at the deadline.
        let fake = platform6();
        fake.on_get("interfaceversion", FakeReply::value(3))
            .on_put("connect", FakeReply::ok())
            .on_put("disconnect", FakeReply::ok())
            .on_get("connecting", FakeReply::value(true));
        let mut wheel = driver(&fake, &clock);
        let err = wheel.connect(Duration::from_secs(1)).unwrap_err();
        assert_eq!(err.class, ErrorClass::Timeout);
        assert_eq!(disconnects(&fake), (1, 0));

        // Platform 7: the connect request itself times out.
        let fake = platform6();
        fake.on_get("interfaceversion", FakeReply::value(3))
            .on_put("connect", FakeReply::Timeout)
            .on_put("disconnect", FakeReply::ok());
        let mut wheel = driver(&fake, &clock);
        assert_eq!(
            wheel.connect(Duration::from_secs(1)).unwrap_err().class,
            ErrorClass::Timeout
        );
        assert_eq!(disconnects(&fake), (1, 0));

        // Platform 7: connected, but Connected reads false.
        let fake = platform6();
        fake.on_get("interfaceversion", FakeReply::value(3))
            .on_put("connect", FakeReply::ok())
            .on_put("disconnect", FakeReply::ok())
            .on_get("connecting", FakeReply::value(false))
            .on_get("connected", FakeReply::value(false));
        let mut wheel = driver(&fake, &clock);
        assert!(wheel.connect(Duration::from_secs(1)).is_err());
        assert_eq!(disconnects(&fake), (1, 0));

        // Platform 6: Connected=true is not confirmed.
        let fake = platform6();
        fake.on_get("connected", FakeReply::value(false));
        let mut wheel = driver(&fake, &clock);
        assert_eq!(
            wheel.connect(Duration::from_secs(1)).unwrap_err().class,
            ErrorClass::Driver
        );
        assert_eq!(disconnects(&fake), (0, 1));

        // Platform 6: the Connected=true request fails.
        let fake = platform6();
        fake.queue_put("connected", [FakeReply::Transport("reset".into())]);
        let mut wheel = driver(&fake, &clock);
        assert_eq!(
            wheel.connect(Duration::from_secs(1)).unwrap_err().class,
            ErrorClass::Transport
        );
        assert_eq!(disconnects(&fake), (0, 1));

        // No connect request was sent yet: nothing to undo. A failed
        // interfaceversion read is version 1 and connects normally.
        let fake = platform6();
        fake.on_get("interfaceversion", FakeReply::Timeout);
        let mut wheel = driver(&fake, &clock);
        wheel.connect(Duration::from_secs(1)).unwrap();
        assert_eq!(disconnects(&fake), (0, 0));
    }

    #[test]
    fn a_wheel_without_a_slot_table_refuses_to_connect() {
        let fake = platform6();
        fake.on_get("names", FakeReply::not_implemented())
            .on_get("focusoffsets", FakeReply::http(400, "Unknown member"))
            .on_put("connected", FakeReply::ok());
        let clock = Arc::new(ManualClock::new());
        let mut wheel = driver(&fake, &clock);
        let err = wheel.connect(Duration::from_secs(5)).unwrap_err();
        assert_eq!(err.class, ErrorClass::Driver);
        let disconnect = fake
            .calls()
            .into_iter()
            .rev()
            .find(|c| c.method == Method::Put && c.member == "connected")
            .unwrap();
        assert_eq!(
            disconnect.params,
            vec![("Connected".into(), "false".into())]
        );

        // Offsets alone still give the slot count.
        let fake = platform6();
        fake.on_get("names", FakeReply::not_implemented());
        let mut wheel = driver(&fake, &clock);
        wheel.connect(Duration::from_secs(5)).unwrap();
        let slots = wheel.slots().unwrap();
        assert_eq!(slots.len(), 5);
        assert!(slots.iter().all(|s| s.device_name.is_none()));
        assert!(wheel.unsupported_members().contains(&"names".to_string()));
    }

    #[test]
    fn position_reads_follow_the_property_policy() {
        let fake = platform6();
        let clock = Arc::new(ManualClock::new());
        let mut wheel = driver(&fake, &clock);
        wheel.connect(Duration::from_secs(5)).unwrap();

        fake.on_get("position", FakeReply::value(-1));
        let raw = wheel.poll(&PollPlan::default()).unwrap();
        assert_eq!(wheel.position().unwrap(), None, "-1 is moving");
        assert!(wheel.observed_position().1);
        assert_eq!(raw.members[0].state, ObsState::Fresh);

        fake.on_get("position", FakeReply::value(3));
        wheel.poll(&PollPlan::default()).unwrap();
        assert_eq!(wheel.position().unwrap(), Some(3));

        fake.on_get("position", FakeReply::http(500, "fault"));
        assert_eq!(
            wheel.poll(&PollPlan::default()).unwrap_err().class,
            ErrorClass::Driver
        );
        let (observed, _) = wheel.observed_position();
        assert_eq!((observed.state, observed.value), (ObsState::Stale, Some(3)));
        assert!(
            wheel.position().is_err(),
            "a stale read is never presented as fresh"
        );

        fake.on_get("position", FakeReply::value_not_set());
        let raw = wheel.poll(&PollPlan::default()).unwrap();
        assert_eq!(raw.members[0].state, ObsState::NotSet);
        assert!(wheel.position().unwrap_err().is_not_set());

        fake.on_get("position", FakeReply::value(9));
        assert!(wheel.poll(&PollPlan::default()).is_err(), "out of range");
    }

    #[test]
    fn set_position_sends_the_exact_case_form_and_validates_the_slot() {
        let fake = platform6();
        fake.on_put("position", FakeReply::ok());
        let clock = Arc::new(ManualClock::new());
        let mut wheel = driver(&fake, &clock);
        assert_eq!(
            wheel.set_position(1).unwrap_err().class,
            ErrorClass::NotConnected
        );
        wheel.connect(Duration::from_secs(5)).unwrap();
        assert_eq!(
            wheel.set_position(5).unwrap_err().class,
            ErrorClass::InvalidValue
        );
        wheel.set_position(4).unwrap();
        let call = fake
            .calls()
            .into_iter()
            .find(|c| c.method == Method::Put && c.member == "position")
            .unwrap();
        assert_eq!(call.params, vec![("Position".into(), "4".into())]);
        assert_eq!(call.class, TimeoutClass::Ordinary);
        assert_eq!(
            wheel.position().unwrap(),
            None,
            "moving until the next read"
        );
        assert!(wheel.abort_path().is_none());
    }

    /// The wheel operation against the fake transport: -1 while moving,
    /// then arrival and verification, as the worker would drive it.
    #[test]
    fn the_wheel_operation_runs_over_alpaca() {
        let fake = platform6();
        fake.on_put("position", FakeReply::ok());
        let clock = Arc::new(ManualClock::new());
        let mut wheel = driver(&fake, &clock);
        wheel.connect(Duration::from_secs(5)).unwrap();
        fake.queue_get(
            "position",
            [
                FakeReply::value(-1),
                FakeReply::value(-1),
                FakeReply::value(2),
            ],
        )
        .on_get("position", FakeReply::value(2));
        let mut op = FilterMoveOp::new(2, &FilterSettings::default());
        op.start(&mut wheel, clock.as_ref()).unwrap();
        let mut outcome = None;
        for _ in 0..10 {
            clock.advance(MOVE_POLL_INTERVAL);
            let _ = wheel.poll(&PollPlan::default());
            if let OpProgress::Done(done) = op.step(&mut wheel, clock.as_ref()) {
                outcome = Some(done);
                break;
            }
        }
        assert_eq!(outcome, Some(OpOutcome::Succeeded));

        // A wheel that settles on another slot is a wrong slot.
        fake.on_get("position", FakeReply::value(3));
        let mut op = FilterMoveOp::new(1, &FilterSettings::default());
        op.start(&mut wheel, clock.as_ref()).unwrap();
        let mut outcome = None;
        for _ in 0..10 {
            clock.advance(MOVE_POLL_INTERVAL);
            let _ = wheel.poll(&PollPlan::default());
            if let OpProgress::Done(done) = op.step(&mut wheel, clock.as_ref()) {
                outcome = Some(done);
                break;
            }
        }
        assert!(is_wrong_slot(&outcome.unwrap()));
    }

    #[test]
    fn timestamps_parse_permissively() {
        let utc = parse_timestamp("2026-10-26T01:02:03.250Z").unwrap();
        assert_eq!(utc.to_rfc3339(), "2026-10-26T01:02:03.250+00:00");
        let offset = parse_timestamp("2026-10-26T03:02:03+02:00").unwrap();
        assert_eq!(offset, parse_timestamp("2026-10-26T01:02:03Z").unwrap());
        assert!(
            parse_timestamp("2026-10-26T01:02:03.5").is_some(),
            "local time"
        );
        assert!(parse_timestamp("yesterday").is_none());
    }

    #[test]
    fn disconnect_and_actions_use_the_platform_members() {
        let fake = platform6();
        fake.on_put("action", FakeReply::value("ok"));
        let clock = Arc::new(ManualClock::new());
        let mut wheel = driver(&fake, &clock);
        wheel.connect(Duration::from_secs(5)).unwrap();
        assert_eq!(wheel.action("ping", "x").unwrap(), "ok");
        let call = fake
            .calls()
            .into_iter()
            .find(|c| c.member == "action")
            .unwrap();
        assert_eq!(
            call.params,
            vec![
                ("Action".into(), "ping".into()),
                ("Parameters".into(), "x".into())
            ]
        );
        assert!(wheel.supported_actions().unwrap().is_empty());
        assert!(wheel.is_connected().unwrap());
        wheel.disconnect().unwrap();
        assert_eq!(
            wheel.set_position(0).unwrap_err().class,
            ErrorClass::NotConnected
        );
    }
}

//! Alpaca SafetyMonitor driver (spec §3.10, §7.1 R19). Owner: WP1A-8.
//!
//! An independent implementation of `SafetyDriver` from the ASCOM Alpaca API
//! reference and the `ISafetyMonitor` master interface (MIT). No NINA code is
//! used.
//!
//! - `IsSafe` is read on every poll (§2.7: every 2 s) and **never cached**:
//!   [`SafetyDriver::is_safe`] returns only what the latest poll read, and
//!   consumes it, so a later call without a new poll reads the device again.
//! - A failed read, including ValueNotSet, is an error. It is never reported
//!   as safe; the hub treats it as not safe for gating (§2.8 fail-safe rule).
//! - Alpaca `IsSafe` is a boolean: `true` is `Safe`, `false` is `Unsafe`.
//!   Alpaca has no warning level (INDI Busy maps to `Warning` in Tier 2).
//! - Platform 7 (InterfaceVersion >= 3) connects with `connect` and
//!   `connecting` and reads `IsSafe` from `devicestate` when the plan
//!   prefers it; Platform 6 uses `Connected=true` and `issafe`.

use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;

use super::observingconditions::{parse_device_state, AlpacaLink, Timed};
use crate::equipment::snapshot::{SafetyLevel, SafetyReading};
use crate::equipment::traits::{
    AbortPath, AlpacaTransport, Clock, ConnectReport, Device, PollMethod, PollPlan, RawPoll,
    SafetyDriver, TimeoutClass,
};
use crate::equipment::types::{DeviceError, DeviceIdentity, DeviceResult, ObsState};

/// SafetyMonitor interface version at which Platform 7 connect and
/// `devicestate` apply (ASCOM Library `DeviceCapabilities`).
pub const SAFETY_MONITOR_PLATFORM7: i32 = 3;

/// Poll cadence (§2.7 table: 0.5 Hz, never cached).
pub const SAFETY_POLL_INTERVAL: Duration = Duration::from_secs(2);

/// Map an `IsSafe` value. Booleans only; a tolerant `"true"`/`"false"`
/// string is accepted. Anything else is a driver error, never "safe".
pub fn parse_is_safe(value: &Value) -> DeviceResult<SafetyLevel> {
    match value {
        Value::Bool(true) => Ok(SafetyLevel::Safe),
        Value::Bool(false) => Ok(SafetyLevel::Unsafe),
        Value::String(text) if text.trim().eq_ignore_ascii_case("true") => Ok(SafetyLevel::Safe),
        Value::String(text) if text.trim().eq_ignore_ascii_case("false") => Ok(SafetyLevel::Unsafe),
        other => Err(DeviceError::driver(format!(
            "issafe returned a non-boolean value: {other}"
        ))),
    }
}

fn reading_from(result: &DeviceResult<Value>) -> DeviceResult<SafetyReading> {
    match result {
        Ok(value) => parse_is_safe(value).map(|level| SafetyReading {
            level,
            detail: None,
        }),
        Err(err) if err.is_not_set() => Err(DeviceError {
            message: format!("IsSafe is not set ({}); counted as not safe", err.message),
            ..err.clone()
        }),
        Err(err) => Err(err.clone()),
    }
}

/// Alpaca SafetyMonitor device (R19).
pub struct AlpacaSafetyMonitor {
    identity: DeviceIdentity,
    link: AlpacaLink,
    /// The latest poll's reading; taken by `is_safe` (never cached).
    latest: Option<DeviceResult<SafetyReading>>,
}

impl AlpacaSafetyMonitor {
    /// `identity.endpoint` is the device's `AlpacaHttp` endpoint; the spine
    /// builds the identity key with `identity_key::alpaca` (§2.6).
    pub fn new(
        identity: DeviceIdentity,
        transport: Arc<dyn AlpacaTransport>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            identity,
            link: AlpacaLink::new(transport, clock, SAFETY_MONITOR_PLATFORM7),
            latest: None,
        }
    }

    /// Wait between Platform 7 `connecting` reads (default 250 ms).
    pub fn with_connect_poll(mut self, interval: Duration) -> Self {
        self.link.connect_poll = interval;
        self
    }

    pub fn uses_device_state(&self) -> bool {
        self.link.device_state
    }

    fn read_member(&self) -> Timed {
        self.link.get("issafe", &[], TimeoutClass::Ordinary)
    }
}

impl Device for AlpacaSafetyMonitor {
    fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    fn connect(&mut self, timeout: Duration) -> DeviceResult<ConnectReport> {
        self.latest = None;
        let mut report = self.link.connect(&mut self.identity, timeout)?;
        // `issafe` is mandatory; a monitor without it is unusable.
        let probe = self.read_member();
        if let Err(err) = &probe.result {
            if err.is_unknown_member() {
                self.link.connected = false;
                return Err(DeviceError::unsupported(format!(
                    "the safety monitor does not implement IsSafe: {err}"
                )));
            }
            report
                .warnings
                .push(format!("first IsSafe read failed: {err}"));
        }
        self.link.probe_device_state(&mut report);
        report.device_state = self.link.device_state;
        Ok(report)
    }

    fn disconnect(&mut self) -> DeviceResult<()> {
        self.latest = None;
        self.link.disconnect()
    }

    fn is_connected(&mut self) -> DeviceResult<bool> {
        self.link.is_connected()
    }

    fn poll(&mut self, plan: &PollPlan) -> DeviceResult<RawPoll> {
        self.link.require_connected()?;
        let started_mono_ns = self.link.clock.now_mono_ns();
        let mut method = PollMethod::PerMember;
        let mut device_time = None;
        let mut timed = None;

        if self.link.device_state && plan.prefer_device_state {
            let state = self.link.get("devicestate", &[], TimeoutClass::Ordinary);
            match &state.result {
                Ok(value) => {
                    let values = parse_device_state(value).unwrap_or_default();
                    if let Some(is_safe) = values.get("IsSafe") {
                        method = PollMethod::DeviceState;
                        device_time = values.timestamp;
                        timed = Some(Timed {
                            result: Ok(is_safe.clone()),
                            observed_at: state.observed_at,
                            mono_ns: state.mono_ns,
                            latency_ms: state.latency_ms,
                        });
                    }
                }
                Err(err) if err.is_unknown_member() => self.link.device_state = false,
                Err(err) => {
                    self.latest = Some(Err(err.clone()));
                    return Err(err.clone());
                }
            }
        }
        // Per-member read, or DeviceState without an IsSafe entry.
        let timed = timed.unwrap_or_else(|| self.read_member());
        let reading = reading_from(&timed.result);
        let member = timed.raw_member(
            "issafe",
            if reading.is_ok() {
                ObsState::Fresh
            } else {
                ObsState::Error
            },
        );
        self.latest = Some(reading.clone());
        reading?;
        Ok(RawPoll {
            method,
            started_mono_ns,
            finished_mono_ns: self.link.clock.now_mono_ns(),
            device_time,
            members: vec![member],
        })
    }

    fn supported_actions(&mut self) -> DeviceResult<Vec<String>> {
        self.link.supported_actions()
    }

    fn action(&mut self, name: &str, params: &str) -> DeviceResult<String> {
        self.link.action(name, params)
    }

    /// A safety monitor never moves: no abort path.
    fn abort_path(&self) -> Option<Arc<dyn AbortPath>> {
        None
    }
}

impl SafetyDriver for AlpacaSafetyMonitor {
    /// The latest poll's reading, consumed; without a new poll, a fresh read.
    fn is_safe(&mut self) -> DeviceResult<SafetyReading> {
        self.link.require_connected()?;
        match self.latest.take() {
            Some(reading) => reading,
            None => reading_from(&self.read_member().result),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::testing::fake_transport::{FakeAlpacaTransport, FakeReply, Method};
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::traits::ConnectPath;
    use crate::equipment::types::{identity_key, DeviceKind, DriverId, Endpoint, ErrorClass};
    use serde_json::json;

    fn monitor(fake: &Arc<FakeAlpacaTransport>) -> AlpacaSafetyMonitor {
        let identity = DeviceIdentity::new(
            DeviceKind::SafetyMonitor,
            DriverId::Alpaca,
            Endpoint::AlpacaHttp {
                base: "http://127.0.0.1:11111".into(),
                device_type: "safetymonitor".into(),
                number: 0,
            },
            identity_key::alpaca(Some("sm-uuid-1"), "127.0.0.1", 11111, "safetymonitor", 0),
            "SafetyMonitor 0",
        );
        AlpacaSafetyMonitor::new(
            identity,
            Arc::clone(fake) as Arc<dyn AlpacaTransport>,
            Arc::new(ManualClock::new()),
        )
        .with_connect_poll(Duration::from_millis(1))
    }

    fn platform6_fake(is_safe: bool) -> Arc<FakeAlpacaTransport> {
        let fake = FakeAlpacaTransport::new().shared();
        fake.on_get("interfaceversion", FakeReply::value(1))
            .on_put("connected", FakeReply::ok())
            .on_get("connected", FakeReply::value(true))
            .on_get("name", FakeReply::value("Cloud Sensor"))
            .on_get("issafe", FakeReply::value(is_safe));
        fake
    }

    #[test]
    fn is_safe_maps_booleans_and_is_never_cached() {
        let fake = platform6_fake(true);
        let mut sm = monitor(&fake);
        let report = sm.connect(Duration::from_secs(5)).unwrap();
        assert_eq!(report.path, ConnectPath::Platform6);
        assert_eq!(sm.identity().name, "Cloud Sensor");
        assert_eq!(sm.identity().identity_key, "alpaca:sm-uuid-1");

        let raw = sm.poll(&PollPlan::default()).unwrap();
        assert_eq!(raw.members[0].name, "issafe");
        assert_eq!(raw.members[0].state, ObsState::Fresh);
        assert_eq!(sm.is_safe().unwrap().level, SafetyLevel::Safe);

        // The poll's reading was consumed: the next call reads again.
        fake.on_get("issafe", FakeReply::value(false));
        fake.clear_calls();
        assert_eq!(sm.is_safe().unwrap().level, SafetyLevel::Unsafe);
        assert_eq!(fake.calls_to(Method::Get, "issafe"), 1);
        sm.poll(&PollPlan::default()).unwrap();
        assert_eq!(sm.is_safe().unwrap().level, SafetyLevel::Unsafe);
        assert!(sm.abort_path().is_none());
    }

    #[test]
    fn read_errors_and_value_not_set_are_never_safe() {
        let fake = platform6_fake(true);
        let mut sm = monitor(&fake);
        sm.connect(Duration::from_secs(5)).unwrap();
        sm.poll(&PollPlan::default()).unwrap();
        assert_eq!(sm.is_safe().unwrap().level, SafetyLevel::Safe);

        // A timeout fails the poll, and is_safe reports the error rather
        // than the earlier "safe".
        fake.queue_get("issafe", [FakeReply::Timeout]);
        let err = sm.poll(&PollPlan::default()).unwrap_err();
        assert_eq!(err.class, ErrorClass::Timeout);
        assert_eq!(sm.is_safe().unwrap_err().class, ErrorClass::Timeout);

        fake.queue_get("issafe", [FakeReply::value_not_set()]);
        assert!(sm.poll(&PollPlan::default()).is_err());
        let err = sm.is_safe().unwrap_err();
        assert!(err.is_not_set());
        assert!(err.message.contains("not safe"));

        fake.queue_get("issafe", [FakeReply::value(json!(1))]);
        assert_eq!(
            sm.poll(&PollPlan::default()).unwrap_err().class,
            ErrorClass::Driver
        );
        assert_eq!(parse_is_safe(&json!("TRUE")).unwrap(), SafetyLevel::Safe);
        assert!(parse_is_safe(&Value::Null).is_err());

        sm.disconnect().unwrap();
        assert_eq!(
            sm.is_safe().unwrap_err().class,
            ErrorClass::NotConnected,
            "a disconnected monitor is not safe"
        );
    }

    #[test]
    fn platform7_reads_is_safe_from_device_state() {
        let fake = FakeAlpacaTransport::new().shared();
        fake.on_get("interfaceversion", FakeReply::value(3))
            .on_put("connect", FakeReply::ok())
            .on_get("connecting", FakeReply::value(false))
            .on_get("connected", FakeReply::value(true))
            .on_get("issafe", FakeReply::value(true))
            .on_get(
                "devicestate",
                FakeReply::value(json!([
                    {"Name": "IsSafe", "Value": false},
                    {"Name": "TimeStamp", "Value": "2026-01-01T00:00:01Z"},
                ])),
            );
        let mut sm = monitor(&fake);
        let report = sm.connect(Duration::from_secs(5)).unwrap();
        assert_eq!(report.path, ConnectPath::Platform7);
        assert!(report.device_state && sm.uses_device_state());
        fake.clear_calls();
        let plan = PollPlan {
            prefer_device_state: true,
            ..PollPlan::default()
        };
        let raw = sm.poll(&plan).unwrap();
        assert_eq!(raw.method, PollMethod::DeviceState);
        assert!(raw.device_time.is_some());
        assert_eq!(fake.calls_to(Method::Get, "issafe"), 0);
        assert_eq!(sm.is_safe().unwrap().level, SafetyLevel::Unsafe);

        // DeviceState without IsSafe falls back to the member.
        fake.on_get("devicestate", FakeReply::value(json!([])));
        sm.poll(&plan).unwrap();
        assert_eq!(fake.calls_to(Method::Get, "issafe"), 1);
        assert_eq!(sm.is_safe().unwrap().level, SafetyLevel::Safe);
    }

    #[test]
    fn a_monitor_without_is_safe_refuses_to_connect() {
        let fake = platform6_fake(true);
        fake.on_get("issafe", FakeReply::not_implemented());
        let mut sm = monitor(&fake);
        assert_eq!(
            sm.connect(Duration::from_secs(5)).unwrap_err().class,
            ErrorClass::Unsupported
        );
        assert!(sm.poll(&PollPlan::default()).is_err());
    }
}

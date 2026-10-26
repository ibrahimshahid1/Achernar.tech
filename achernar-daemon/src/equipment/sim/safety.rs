//! Simulated safety monitor (spec §5.6, §7.1 R19). Owner: WP1A-8.
//!
//! An in-process `SafetyDriver`, also selectable as driver `sim`. A
//! [`SimSafetyControl`] (cloneable) scripts it: safe, unsafe or warning (the
//! INDI Busy level, so the `warning_is_unsafe` rule is testable without
//! INDI), injected read failures, an `IsSafe` that is not set (ValueNotSet),
//! and a dropped connection. Like the Alpaca driver it never caches:
//! `is_safe` returns the latest poll's reading once, and a failed or unset
//! read is an error, never "safe". A not-set read does not fail the poll.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::equipment::snapshot::{SafetyLevel, SafetyReading};
use crate::equipment::traits::{
    AbortPath, Clock, ConnectPath, ConnectReport, Device, PollMethod, PollPlan, RawMember, RawPoll,
    SafetyDriver,
};
use crate::equipment::types::{
    ascom_error, identity_key, DeviceError, DeviceIdentity, DeviceKind, DeviceResult, DriverId,
    Endpoint, ObsState,
};

#[derive(Debug)]
struct SimSafetyState {
    level: SafetyLevel,
    detail: Option<String>,
    fail_reads: u32,
    failing: bool,
    not_set: bool,
    reachable: bool,
    reads: u64,
}

impl Default for SimSafetyState {
    fn default() -> Self {
        Self {
            level: SafetyLevel::Safe,
            detail: None,
            fail_reads: 0,
            failing: false,
            not_set: false,
            reachable: true,
            reads: 0,
        }
    }
}

/// Script handle for a [`SimSafetyMonitor`]; clones share one device.
#[derive(Debug, Clone, Default)]
pub struct SimSafetyControl {
    state: Arc<Mutex<SimSafetyState>>,
}

impl SimSafetyControl {
    pub fn new() -> Self {
        Self::default()
    }

    fn state(&self) -> MutexGuard<'_, SimSafetyState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn set_level(&self, level: SafetyLevel) {
        self.state().level = level;
    }

    pub fn set_safe(&self, safe: bool) {
        self.set_level(if safe {
            SafetyLevel::Safe
        } else {
            SafetyLevel::Unsafe
        });
    }

    pub fn set_detail(&self, detail: Option<&str>) {
        self.state().detail = detail.map(str::to_string);
    }

    /// The next `n` reads fail with a timeout (a Wi-Fi blip).
    pub fn fail_next_reads(&self, n: u32) {
        self.state().fail_reads = n;
    }

    /// Every read fails until cleared.
    pub fn set_failing(&self, failing: bool) {
        self.state().failing = failing;
    }

    /// True: `IsSafe` answers ValueNotSet (a monitor still warming up).
    pub fn set_not_set(&self, not_set: bool) {
        self.state().not_set = not_set;
    }

    /// False: the monitor drops the connection.
    pub fn set_reachable(&self, reachable: bool) {
        self.state().reachable = reachable;
    }

    pub fn reads(&self) -> u64 {
        self.state().reads
    }
}

/// The simulated safety monitor.
pub struct SimSafetyMonitor {
    identity: DeviceIdentity,
    clock: Arc<dyn Clock>,
    control: SimSafetyControl,
    connected: bool,
    latest: Option<DeviceResult<SafetyReading>>,
}

impl SimSafetyMonitor {
    pub fn new(clock: Arc<dyn Clock>, control: SimSafetyControl) -> Self {
        let identity = DeviceIdentity::new(
            DeviceKind::SafetyMonitor,
            DriverId::Sim,
            Endpoint::InProcess {
                name: "sim-safety".into(),
            },
            identity_key::sim(DeviceKind::SafetyMonitor),
            "Simulated safety monitor",
        );
        Self::with_identity(identity, clock, control)
    }

    /// A second simulated monitor needs its own name and identity key.
    pub fn with_identity(
        identity: DeviceIdentity,
        clock: Arc<dyn Clock>,
        control: SimSafetyControl,
    ) -> Self {
        Self {
            identity,
            clock,
            control,
            connected: false,
            latest: None,
        }
    }

    pub fn control(&self) -> SimSafetyControl {
        self.control.clone()
    }

    fn require_connected(&self) -> DeviceResult<()> {
        if self.connected {
            Ok(())
        } else {
            Err(DeviceError::not_connected(
                "the simulated safety monitor is not connected",
            ))
        }
    }

    fn read_now(&mut self) -> DeviceResult<SafetyReading> {
        let mut state = self.control.state();
        state.reads += 1;
        if !state.reachable {
            drop(state);
            self.connected = false;
            return Err(DeviceError::disconnected(
                "the simulated safety monitor dropped the connection",
            ));
        }
        if state.failing || state.fail_reads > 0 {
            state.fail_reads = state.fail_reads.saturating_sub(1);
            return Err(DeviceError::timeout(
                "simulated safety monitor: injected read failure",
            ));
        }
        if state.not_set {
            return Err(DeviceError::from_alpaca_error(
                ascom_error::VALUE_NOT_SET,
                "simulated safety monitor: IsSafe is not set",
            ));
        }
        Ok(SafetyReading {
            level: state.level,
            detail: state.detail.clone(),
        })
    }
}

impl Device for SimSafetyMonitor {
    fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    fn connect(&mut self, _timeout: Duration) -> DeviceResult<ConnectReport> {
        if !self.control.state().reachable {
            return Err(DeviceError::disconnected(
                "the simulated safety monitor is unreachable",
            ));
        }
        self.connected = true;
        self.latest = None;
        Ok(ConnectReport {
            path: ConnectPath::InProcess,
            interface_version: Some(3),
            ..ConnectReport::default()
        })
    }

    fn disconnect(&mut self) -> DeviceResult<()> {
        self.connected = false;
        self.latest = None;
        Ok(())
    }

    fn is_connected(&mut self) -> DeviceResult<bool> {
        if !self.control.state().reachable {
            self.connected = false;
        }
        Ok(self.connected)
    }

    fn poll(&mut self, _plan: &PollPlan) -> DeviceResult<RawPoll> {
        self.require_connected()?;
        let now = self.clock.now_utc();
        let mono = self.clock.now_mono_ns();
        let reading = self.read_now();
        let member = RawMember {
            name: "issafe".into(),
            value: reading
                .as_ref()
                .ok()
                .map(|r| serde_json::Value::Bool(r.level == SafetyLevel::Safe)),
            state: match &reading {
                Ok(_) => ObsState::Fresh,
                Err(err) if err.is_not_set() => ObsState::NotSet,
                Err(_) => ObsState::Error,
            },
            observed_at: Some(now),
            mono_ns: Some(mono),
            latency_ms: Some(0),
            error: reading.as_ref().err().map(|e| e.to_string()),
        };
        self.latest = Some(reading.clone());
        // ValueNotSet is an unknown level, not a link failure (C-2).
        if let Err(err) = reading {
            if !err.is_not_set() {
                return Err(err);
            }
        }
        Ok(RawPoll {
            method: PollMethod::InProcess,
            started_mono_ns: mono,
            finished_mono_ns: self.clock.now_mono_ns(),
            device_time: Some(now),
            members: vec![member],
        })
    }

    fn supported_actions(&mut self) -> DeviceResult<Vec<String>> {
        Ok(Vec::new())
    }

    fn action(&mut self, name: &str, _params: &str) -> DeviceResult<String> {
        Err(DeviceError::unsupported(format!(
            "the simulated safety monitor has no action {name}"
        )))
    }

    fn abort_path(&self) -> Option<Arc<dyn AbortPath>> {
        None
    }
}

impl SafetyDriver for SimSafetyMonitor {
    /// The latest poll's reading, consumed; without a new poll, a fresh read.
    fn is_safe(&mut self) -> DeviceResult<SafetyReading> {
        self.require_connected()?;
        match self.latest.take() {
            Some(reading) => reading,
            None => self.read_now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::types::ErrorClass;

    #[test]
    fn the_sim_monitor_scripts_levels_failures_and_drops() {
        let control = SimSafetyControl::new();
        let mut sim = SimSafetyMonitor::new(Arc::new(ManualClock::new()), control.clone());
        assert!(sim.is_safe().is_err(), "not connected is never safe");
        sim.connect(Duration::from_secs(1)).unwrap();
        sim.poll(&PollPlan::default()).unwrap();
        assert_eq!(sim.is_safe().unwrap().level, SafetyLevel::Safe);

        control.set_level(SafetyLevel::Warning);
        control.set_detail(Some("clouds"));
        let reading = sim.is_safe().unwrap();
        assert_eq!(reading.level, SafetyLevel::Warning);
        assert_eq!(reading.detail.as_deref(), Some("clouds"));

        control.fail_next_reads(1);
        assert_eq!(
            sim.poll(&PollPlan::default()).unwrap_err().class,
            ErrorClass::Timeout
        );
        assert_eq!(sim.is_safe().unwrap_err().class, ErrorClass::Timeout);
        control.set_safe(false);
        sim.poll(&PollPlan::default()).unwrap();
        assert_eq!(sim.is_safe().unwrap().level, SafetyLevel::Unsafe);

        control.set_not_set(true);
        let raw = sim.poll(&PollPlan::default()).unwrap();
        assert_eq!(raw.members[0].state, ObsState::NotSet);
        assert!(
            sim.is_safe().unwrap_err().is_not_set(),
            "not set is not safe"
        );
        control.set_not_set(false);

        control.set_reachable(false);
        assert_eq!(
            sim.poll(&PollPlan::default()).unwrap_err().class,
            ErrorClass::Disconnected
        );
        assert!(!sim.is_connected().unwrap());
        assert_eq!(control.reads(), 6);
    }
}

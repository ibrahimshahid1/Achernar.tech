//! In-process simulator focuser (spec §5.3 WP1A-5, §5.6), also selectable as
//! driver `sim`.
//!
//! A kinematic model on the injected [`Clock`]: a commanded move travels at
//! `speed_steps_per_s`, so `IsMoving` stays true until the position arrives,
//! exactly like a real device polled by its worker. Configurable profiles:
//!
//! - absolute or relative (a relative focuser reports no `Position`);
//! - `MaxStep`, `MaxIncrement` (a larger single move is refused, as ASCOM
//!   requires), `StepSize`;
//! - a temperature probe with drift, and TempComp;
//! - a Platform 6 profile (InterfaceVersion 3, no DeviceState), which is how
//!   the planned ESP32 focuser presents itself;
//! - declared firmware-owned backlash, or modelled mechanical slop, so
//!   backlash compensation can be checked against the "optical" position
//!   the slop produces.
//!
//! [`SimFocuserControl`] injects faults while the worker owns the driver: a
//! stalled motor that keeps reporting `IsMoving`, an off-target landing,
//! failing members, a lost position (`0x500` "position unknown"), a dropped
//! link, and external (hand-controller) moves. The sim's halt also runs on
//! its own [`AbortPath`], concurrently with the worker (D4).

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::equipment::snapshot::{FocuserCapabilities, FocuserState};
use crate::equipment::traits::{
    AbortPath, Clock, ConnectPath, ConnectReport, Device, FocuserDriver, PollMethod, PollPlan,
    RawMember, RawPoll,
};
use crate::equipment::types::{
    ascom_error, identity_key, DeviceError, DeviceIdentity, DeviceKind, DeviceResult, DriverId,
    Endpoint, ObsState, Observed, Source,
};

/// Simulator profile.
#[derive(Debug, Clone, PartialEq)]
pub struct SimFocuserConfig {
    pub name: String,
    pub absolute: bool,
    pub max_step: i32,
    pub max_increment: i32,
    pub step_size_um: Option<f64>,
    pub speed_steps_per_s: f64,
    pub start_position: i32,
    /// `None`: the probe is not implemented (`Unsupported`).
    pub temperature_c: Option<f64>,
    pub temperature_drift_c_per_hour: f64,
    pub temp_comp_available: bool,
    pub temp_comp: bool,
    pub interface_version: i32,
    /// Platform 7 `devicestate`.
    pub device_state: bool,
    /// INDI-style position sync (R16 "S: yes").
    pub can_sync: bool,
    /// The device declares that it compensates backlash itself.
    pub firmware_backlash: bool,
    /// Mechanical slop the gear train has (when the firmware does not take
    /// it up).
    pub mechanical_backlash_steps: i32,
    pub supported_actions: Vec<String>,
    /// Replies for `action`, by action name.
    pub action_replies: Vec<(String, String)>,
}

impl SimFocuserConfig {
    /// An absolute Platform 7 focuser with a probe and TempComp.
    pub fn absolute() -> Self {
        Self {
            name: "Simulated focuser".into(),
            absolute: true,
            max_step: 50_000,
            max_increment: 50_000,
            step_size_um: Some(4.5),
            speed_steps_per_s: 1_000.0,
            start_position: 25_000,
            temperature_c: Some(12.0),
            temperature_drift_c_per_hour: 0.0,
            temp_comp_available: true,
            temp_comp: false,
            interface_version: 4,
            device_state: true,
            can_sync: false,
            firmware_backlash: false,
            mechanical_backlash_steps: 0,
            supported_actions: Vec::new(),
            action_replies: Vec::new(),
        }
    }

    /// A relative focuser: no `Position`, moves are step counts.
    pub fn relative() -> Self {
        Self {
            name: "Simulated relative focuser".into(),
            absolute: false,
            max_increment: 10_000,
            step_size_um: None,
            temp_comp_available: false,
            ..Self::absolute()
        }
    }

    /// A Platform 6 focuser (InterfaceVersion 3, no DeviceState) with a
    /// temperature probe and firmware-owned backlash: the profile the
    /// planned ESP32 focuser declares. Nothing here is rig-specific code;
    /// it is one standard profile among others (D15).
    pub fn platform6_firmware_backlash() -> Self {
        Self {
            name: "Simulated Platform 6 focuser".into(),
            interface_version: 3,
            device_state: false,
            firmware_backlash: true,
            temp_comp_available: false,
            ..Self::absolute()
        }
    }
}

/// A driver call the sim received, for test assertions.
#[derive(Debug, Clone, PartialEq)]
pub enum SimCall {
    MoveAbsolute(i32),
    MoveRelative(i32),
    Halt { abort_path: bool },
    SetTempComp(bool),
    Sync(i32),
    Action(String),
}

#[derive(Debug)]
struct SimCore {
    cfg: SimFocuserConfig,
    raw: f64,
    optical: f64,
    target: Option<f64>,
    last_update_ns: u64,
    start_ns: u64,
    temperature: Option<f64>,
    temp_comp: bool,
    stuck: bool,
    arrival_errors: VecDeque<i32>,
    failures: HashMap<String, VecDeque<DeviceError>>,
    position_lost: bool,
    link_down: bool,
    connected: bool,
    calls: Vec<SimCall>,
}

impl SimCore {
    fn advance(&mut self, now_ns: u64) {
        let dt_s = now_ns.saturating_sub(self.last_update_ns) as f64 / 1e9;
        self.last_update_ns = self.last_update_ns.max(now_ns);
        let Some(target) = self.target else {
            return;
        };
        if self.stuck {
            return;
        }
        let travel = self.cfg.speed_steps_per_s.max(1.0) * dt_s;
        let remaining = target - self.raw;
        if remaining.abs() <= travel {
            self.raw = target;
            self.target = None;
        } else {
            self.raw += travel.copysign(remaining);
        }
        self.update_optical();
    }

    /// The draw tube follows the motor inside a dead band of
    /// `mechanical_backlash_steps`: optical is in `[raw - slop, raw]`.
    fn update_optical(&mut self) {
        let slop = if self.cfg.firmware_backlash {
            0.0
        } else {
            f64::from(self.cfg.mechanical_backlash_steps.max(0))
        };
        if self.optical > self.raw {
            self.optical = self.raw;
        }
        if self.optical < self.raw - slop {
            self.optical = self.raw - slop;
        }
    }

    fn take_failure(&mut self, member: &str) -> Option<DeviceError> {
        self.failures.get_mut(member).and_then(VecDeque::pop_front)
    }

    fn check_link(&self) -> DeviceResult<()> {
        if self.link_down {
            return Err(DeviceError::disconnected("simulated link is down"));
        }
        if !self.connected {
            return Err(DeviceError::not_connected(
                "simulated focuser is not connected",
            ));
        }
        Ok(())
    }

    fn temperature_now(&self, now_ns: u64) -> Option<f64> {
        let hours = now_ns.saturating_sub(self.start_ns) as f64 / 3.6e12;
        self.temperature
            .map(|t| t + self.cfg.temperature_drift_c_per_hour * hours)
    }

    fn start_move(&mut self, target: f64) {
        let error = self.arrival_errors.pop_front().unwrap_or(0);
        let max = f64::from(self.cfg.max_step.max(0));
        self.target = Some((target + f64::from(error)).clamp(0.0, max));
    }

    fn halt(&mut self) {
        self.target = None;
        self.stuck = false;
    }
}

fn lock(core: &Mutex<SimCore>) -> MutexGuard<'_, SimCore> {
    core.lock().unwrap_or_else(|e| e.into_inner())
}

/// The simulator driver.
pub struct SimFocuser {
    identity: DeviceIdentity,
    core: Arc<Mutex<SimCore>>,
    clock: Arc<dyn Clock>,
    state: FocuserState,
}

impl SimFocuser {
    pub fn new(cfg: SimFocuserConfig, clock: Arc<dyn Clock>) -> Self {
        let mut identity = DeviceIdentity::new(
            DeviceKind::Focuser,
            DriverId::Sim,
            Endpoint::InProcess {
                name: "sim-focuser".into(),
            },
            identity_key::sim(DeviceKind::Focuser),
            cfg.name.clone(),
        );
        identity.interface_version = Some(cfg.interface_version);
        identity.description = Some("Achernar in-process focuser simulator".into());
        identity.driver_info = Some("achernar sim/focuser".into());
        let now = clock.now_mono_ns();
        let start = f64::from(cfg.start_position.clamp(0, cfg.max_step.max(0)));
        let slop = if cfg.firmware_backlash {
            0.0
        } else {
            f64::from(cfg.mechanical_backlash_steps.max(0))
        };
        let core = SimCore {
            raw: start,
            // Start engaged from the outward side.
            optical: start - slop,
            target: None,
            last_update_ns: now,
            start_ns: now,
            temperature: cfg.temperature_c,
            temp_comp: cfg.temp_comp,
            stuck: false,
            arrival_errors: VecDeque::new(),
            failures: HashMap::new(),
            position_lost: false,
            link_down: false,
            connected: false,
            calls: Vec::new(),
            cfg,
        };
        Self {
            identity,
            core: Arc::new(Mutex::new(core)),
            clock,
            state: FocuserState::default(),
        }
    }

    /// Fault injection and inspection, usable while a worker owns the driver.
    pub fn control(&self) -> SimFocuserControl {
        SimFocuserControl {
            core: Arc::clone(&self.core),
        }
    }

    fn config(&self) -> SimFocuserConfig {
        lock(&self.core).cfg.clone()
    }

    fn source(cfg: &SimFocuserConfig) -> Source {
        if cfg.device_state {
            Source::DeviceState
        } else {
            Source::Property
        }
    }
}

fn member(name: &str, observed: &Observed<serde_json::Value>, error: Option<String>) -> RawMember {
    RawMember {
        name: name.into(),
        value: observed.value.clone(),
        state: observed.state,
        observed_at: observed.observed_at,
        mono_ns: observed.mono_ns,
        latency_ms: observed.latency_ms,
        error,
    }
}

impl Device for SimFocuser {
    fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    fn connect(&mut self, _timeout: Duration) -> DeviceResult<ConnectReport> {
        let mut core = lock(&self.core);
        if core.link_down {
            return Err(DeviceError::disconnected("simulated link is down"));
        }
        core.connected = true;
        let mut unsupported = Vec::new();
        if !core.cfg.absolute {
            unsupported.push("position".to_string());
        }
        if core.cfg.temperature_c.is_none() {
            unsupported.push("temperature".into());
        }
        if core.cfg.step_size_um.is_none() {
            unsupported.push("stepsize".into());
        }
        if !core.cfg.device_state {
            unsupported.push("devicestate".into());
        }
        Ok(ConnectReport {
            path: ConnectPath::InProcess,
            interface_version: Some(core.cfg.interface_version),
            device_state: core.cfg.device_state,
            unsupported_members: unsupported,
            warnings: Vec::new(),
            duration_ms: 0,
        })
    }

    fn disconnect(&mut self) -> DeviceResult<()> {
        lock(&self.core).connected = false;
        Ok(())
    }

    fn is_connected(&mut self) -> DeviceResult<bool> {
        let core = lock(&self.core);
        if core.link_down {
            return Err(DeviceError::disconnected("simulated link is down"));
        }
        Ok(core.connected)
    }

    fn poll(&mut self, plan: &PollPlan) -> DeviceResult<RawPoll> {
        let now_ns = self.clock.now_mono_ns();
        let now_utc = self.clock.now_utc();
        let mut core = lock(&self.core);
        if let Err(err) = core.check_link() {
            drop(core);
            self.state = FocuserState {
                position: self.state.position.clone().into_stale(),
                is_moving: self.state.is_moving.clone().into_stale(),
                temperature_c: self.state.temperature_c.clone().into_stale(),
                temp_comp: self.state.temp_comp.clone().into_stale(),
            };
            return Err(err);
        }
        core.advance(now_ns);
        let source = Self::source(&core.cfg);
        let mut members = Vec::new();
        let mut ok = |name: &str, value: serde_json::Value, source: Source| {
            members.push(member(
                name,
                &Observed::fresh(value, source, now_utc, now_ns),
                None,
            ));
        };

        // Position.
        let mut failed: Vec<(&str, DeviceError)> = Vec::new();
        let position = if !core.cfg.absolute {
            Observed::unsupported(source)
        } else if core.position_lost {
            failed.push((
                "position",
                DeviceError::from_alpaca_error(
                    ascom_error::DRIVER_BASE,
                    "focuser position unknown after a controller reset",
                ),
            ));
            self.state.position.clone().into_stale()
        } else if let Some(err) = core.take_failure("position") {
            failed.push(("position", err));
            self.state.position.clone().into_stale()
        } else {
            let value = core.raw.round() as i32;
            ok("position", serde_json::json!(value), source);
            Observed::fresh(value, source, now_utc, now_ns)
        };

        // IsMoving.
        let is_moving = if let Some(err) = core.take_failure("ismoving") {
            failed.push(("ismoving", err));
            self.state.is_moving.clone().into_stale()
        } else {
            let value = core.target.is_some();
            ok("ismoving", serde_json::json!(value), source);
            Observed::fresh(value, source, now_utc, now_ns)
        };

        // Temperature.
        let temperature_c = match core.temperature_now(now_ns) {
            None => Observed::unsupported(source),
            Some(t) => match core.take_failure("temperature") {
                Some(err) => {
                    failed.push(("temperature", err));
                    self.state.temperature_c.clone().into_stale()
                }
                None => {
                    ok("temperature", serde_json::json!(t), source);
                    Observed::fresh(t, source, now_utc, now_ns)
                }
            },
        };

        // TempComp is not part of DeviceState; read it as an extra member.
        let temp_comp = if !core.cfg.temp_comp_available {
            Observed::unsupported(Source::Property)
        } else {
            let value = core.temp_comp;
            ok("tempcomp", serde_json::json!(value), Source::Property);
            Observed::fresh(value, Source::Property, now_utc, now_ns)
        };
        for (name, err) in failed {
            members.push(member(
                name,
                &Observed::error(source),
                Some(err.to_string()),
            ));
        }

        let method = if core.cfg.device_state && plan.prefer_device_state {
            PollMethod::DeviceState
        } else {
            PollMethod::InProcess
        };
        drop(core);
        self.state = FocuserState {
            position,
            is_moving,
            temperature_c,
            temp_comp,
        };
        Ok(RawPoll {
            method,
            started_mono_ns: now_ns,
            finished_mono_ns: now_ns,
            device_time: None,
            members,
        })
    }

    fn supported_actions(&mut self) -> DeviceResult<Vec<String>> {
        let core = lock(&self.core);
        core.check_link()?;
        Ok(core.cfg.supported_actions.clone())
    }

    fn action(&mut self, name: &str, _params: &str) -> DeviceResult<String> {
        let mut core = lock(&self.core);
        core.check_link()?;
        core.calls.push(SimCall::Action(name.to_string()));
        core.cfg
            .action_replies
            .iter()
            .find(|(action, _)| action.eq_ignore_ascii_case(name))
            .map(|(_, reply)| reply.clone())
            .ok_or_else(|| {
                DeviceError::from_alpaca_error(
                    ascom_error::ACTION_NOT_IMPLEMENTED,
                    &format!("action {name} is not implemented"),
                )
            })
    }

    fn abort_path(&self) -> Option<Arc<dyn AbortPath>> {
        Some(Arc::new(SimFocuserAbort {
            core: Arc::clone(&self.core),
        }))
    }
}

impl FocuserDriver for SimFocuser {
    fn capabilities(&mut self) -> DeviceResult<FocuserCapabilities> {
        let cfg = self.config();
        let mut unsupported = Vec::new();
        if !cfg.absolute {
            unsupported.push("position".to_string());
        }
        if cfg.temperature_c.is_none() {
            unsupported.push("temperature".into());
        }
        if cfg.step_size_um.is_none() {
            unsupported.push("stepsize".into());
        }
        if !cfg.device_state {
            unsupported.push("devicestate".into());
        }
        Ok(FocuserCapabilities {
            absolute: cfg.absolute,
            max_step: Some(cfg.max_step),
            max_increment: Some(cfg.max_increment),
            step_size_um: cfg.step_size_um,
            temp_comp_available: cfg.temp_comp_available,
            can_sync: cfg.can_sync,
            interface_version: Some(cfg.interface_version),
            device_state: cfg.device_state,
            unsupported_members: unsupported,
        })
    }

    fn state(&mut self) -> DeviceResult<FocuserState> {
        Ok(self.state.clone())
    }

    fn move_absolute(&mut self, target: i32) -> DeviceResult<()> {
        let now = self.clock.now_mono_ns();
        let mut core = lock(&self.core);
        core.check_link()?;
        core.advance(now);
        if !core.cfg.absolute {
            return Err(DeviceError::invalid_operation(
                "a relative focuser takes step counts, not positions",
            ));
        }
        if let Some(err) = core.take_failure("move") {
            return Err(err);
        }
        let delta = i64::from(target) - core.raw.round() as i64;
        if delta.abs() > i64::from(core.cfg.max_increment) {
            return Err(DeviceError::from_alpaca_error(
                ascom_error::INVALID_VALUE,
                &format!(
                    "move of {delta} steps exceeds MaxIncrement {}",
                    core.cfg.max_increment
                ),
            ));
        }
        core.calls.push(SimCall::MoveAbsolute(target));
        core.start_move(f64::from(target));
        Ok(())
    }

    fn move_relative(&mut self, steps: i32) -> DeviceResult<()> {
        let now = self.clock.now_mono_ns();
        let mut core = lock(&self.core);
        core.check_link()?;
        core.advance(now);
        if core.cfg.absolute {
            return Err(DeviceError::invalid_operation(
                "an absolute focuser takes positions, not step counts",
            ));
        }
        if let Some(err) = core.take_failure("move") {
            return Err(err);
        }
        if steps.unsigned_abs() > core.cfg.max_increment.unsigned_abs() {
            return Err(DeviceError::from_alpaca_error(
                ascom_error::INVALID_VALUE,
                &format!(
                    "move of {steps} steps exceeds MaxIncrement {}",
                    core.cfg.max_increment
                ),
            ));
        }
        core.calls.push(SimCall::MoveRelative(steps));
        let target = core.raw + f64::from(steps);
        core.start_move(target);
        Ok(())
    }

    fn halt(&mut self) -> DeviceResult<()> {
        let now = self.clock.now_mono_ns();
        let mut core = lock(&self.core);
        core.check_link()?;
        core.advance(now);
        if let Some(err) = core.take_failure("halt") {
            return Err(err);
        }
        core.calls.push(SimCall::Halt { abort_path: false });
        core.halt();
        Ok(())
    }

    fn set_temp_comp(&mut self, on: bool) -> DeviceResult<()> {
        let mut core = lock(&self.core);
        core.check_link()?;
        if !core.cfg.temp_comp_available {
            return Err(DeviceError::from_alpaca_error(
                ascom_error::NOT_IMPLEMENTED,
                "TempComp is not available",
            ));
        }
        core.calls.push(SimCall::SetTempComp(on));
        core.temp_comp = on;
        Ok(())
    }

    fn sync_position(&mut self, position: i32) -> DeviceResult<()> {
        let mut core = lock(&self.core);
        core.check_link()?;
        if !core.cfg.can_sync {
            return Err(DeviceError::unsupported(
                "this focuser has no position sync",
            ));
        }
        core.calls.push(SimCall::Sync(position));
        let shift = f64::from(position) - core.raw;
        core.raw = f64::from(position);
        core.optical += shift;
        Ok(())
    }
}

/// Out-of-band halt for the sim (D4).
struct SimFocuserAbort {
    core: Arc<Mutex<SimCore>>,
}

impl AbortPath for SimFocuserAbort {
    fn abort(&self) -> DeviceResult<()> {
        let mut core = lock(&self.core);
        if core.link_down {
            return Err(DeviceError::disconnected("simulated link is down"));
        }
        core.calls.push(SimCall::Halt { abort_path: true });
        core.halt();
        Ok(())
    }
}

/// Test and operator-tool access to a running simulator.
#[derive(Clone)]
pub struct SimFocuserControl {
    core: Arc<Mutex<SimCore>>,
}

impl SimFocuserControl {
    /// The motor stalls: `IsMoving` stays true and the position freezes
    /// until a halt.
    pub fn set_stuck(&self, stuck: bool) {
        lock(&self.core).stuck = stuck;
    }

    /// The next moves land this many steps off their target, in order.
    pub fn push_arrival_error(&self, steps: i32) {
        lock(&self.core).arrival_errors.push_back(steps);
    }

    /// The next call to `member` (`position`, `ismoving`, `temperature`,
    /// `move`, `halt`) fails with `err`.
    pub fn fail_next(&self, member: &str, err: DeviceError) {
        lock(&self.core)
            .failures
            .entry(member.to_ascii_lowercase())
            .or_default()
            .push_back(err);
    }

    /// The controller lost its position: `Position` reads fail with a
    /// driver error (`0x500` "position unknown") until cleared.
    pub fn set_position_lost(&self, lost: bool) {
        lock(&self.core).position_lost = lost;
    }

    /// Every call fails as a transport loss while down.
    pub fn set_link_down(&self, down: bool) {
        lock(&self.core).link_down = down;
    }

    pub fn set_temperature(&self, celsius: Option<f64>) {
        lock(&self.core).temperature = celsius;
    }

    /// A hand-controller move Achernar did not command.
    pub fn external_move_to(&self, target: i32) {
        let mut core = lock(&self.core);
        let max = f64::from(core.cfg.max_step.max(0));
        core.target = Some(f64::from(target).clamp(0.0, max));
    }

    pub fn raw_position(&self) -> i32 {
        lock(&self.core).raw.round() as i32
    }

    /// Where the draw tube is, after mechanical slop.
    pub fn optical_position(&self) -> i32 {
        lock(&self.core).optical.round() as i32
    }

    pub fn is_moving(&self) -> bool {
        lock(&self.core).target.is_some()
    }

    pub fn temp_comp(&self) -> bool {
        lock(&self.core).temp_comp
    }

    pub fn calls(&self) -> Vec<SimCall> {
        lock(&self.core).calls.clone()
    }

    pub fn clear_calls(&self) {
        lock(&self.core).calls.clear();
    }

    /// Moves the device was commanded to make, in order (halts excluded).
    pub fn moves(&self) -> Vec<SimCall> {
        self.calls()
            .into_iter()
            .filter(|c| matches!(c, SimCall::MoveAbsolute(_) | SimCall::MoveRelative(_)))
            .collect()
    }

    pub fn halts(&self) -> usize {
        self.calls()
            .iter()
            .filter(|c| matches!(c, SimCall::Halt { .. }))
            .count()
    }
}

/// True when `observed` is a value the sim reported as unsupported.
pub fn is_unsupported<T>(observed: &Observed<T>) -> bool {
    observed.state == ObsState::Unsupported && observed.value.is_none()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::types::ErrorClass;

    fn connected(cfg: SimFocuserConfig) -> (SimFocuser, Arc<ManualClock>) {
        let clock = Arc::new(ManualClock::new());
        let mut drv = SimFocuser::new(cfg, clock.clone());
        drv.connect(Duration::from_secs(1)).unwrap();
        drv.poll(&PollPlan::default()).unwrap();
        (drv, clock)
    }

    #[test]
    fn moves_take_time_and_report_is_moving_until_arrival() {
        let (mut drv, clock) = connected(SimFocuserConfig::absolute());
        drv.move_absolute(25_500).unwrap();
        drv.poll(&PollPlan::default()).unwrap();
        assert_eq!(drv.state().unwrap().is_moving.value, Some(true));
        clock.advance_ms(250);
        drv.poll(&PollPlan::default()).unwrap();
        let st = drv.state().unwrap();
        assert_eq!(st.position.value, Some(25_250));
        assert_eq!(st.is_moving.value, Some(true));
        clock.advance_ms(300);
        drv.poll(&PollPlan::default()).unwrap();
        let st = drv.state().unwrap();
        assert_eq!(st.position.value, Some(25_500));
        assert_eq!(st.is_moving.value, Some(false));
        assert_eq!(st.position.mono_ns, Some(550_000_000));
    }

    #[test]
    fn moves_beyond_max_increment_are_refused_and_limits_stop_travel() {
        let mut cfg = SimFocuserConfig::absolute();
        cfg.max_increment = 100;
        let (mut drv, clock) = connected(cfg);
        let err = drv.move_absolute(25_101).unwrap_err();
        assert_eq!(err.class, ErrorClass::InvalidValue);
        drv.move_absolute(25_100).unwrap();
        clock.advance_ms(1_000);
        drv.poll(&PollPlan::default()).unwrap();
        assert_eq!(drv.state().unwrap().position.value, Some(25_100));
        assert!(drv.move_relative(10).is_err(), "absolute takes positions");
    }

    #[test]
    fn a_stuck_motor_keeps_moving_until_halted_by_the_abort_path() {
        let (mut drv, clock) = connected(SimFocuserConfig::absolute());
        let control = drv.control();
        drv.move_absolute(26_000).unwrap();
        control.set_stuck(true);
        clock.advance_ms(5_000);
        drv.poll(&PollPlan::default()).unwrap();
        let st = drv.state().unwrap();
        assert_eq!(st.is_moving.value, Some(true));
        assert_eq!(st.position.value, Some(25_000));
        drv.abort_path().unwrap().abort().unwrap();
        drv.poll(&PollPlan::default()).unwrap();
        assert_eq!(drv.state().unwrap().is_moving.value, Some(false));
        assert_eq!(
            control.calls().last(),
            Some(&SimCall::Halt { abort_path: true })
        );
    }

    #[test]
    fn mechanical_slop_lags_the_optical_position_on_reversal() {
        let mut cfg = SimFocuserConfig::absolute();
        cfg.mechanical_backlash_steps = 20;
        let (mut drv, clock) = connected(cfg);
        let control = drv.control();
        assert_eq!(control.optical_position(), 24_980);
        drv.move_absolute(25_100).unwrap();
        clock.advance_ms(1_000);
        drv.poll(&PollPlan::default()).unwrap();
        assert_eq!(control.optical_position(), 25_080);
        drv.move_absolute(25_090).unwrap();
        clock.advance_ms(1_000);
        drv.poll(&PollPlan::default()).unwrap();
        // 10 steps in after moving out: still inside the dead band.
        assert_eq!(control.optical_position(), 25_080);
        drv.move_absolute(25_050).unwrap();
        clock.advance_ms(1_000);
        drv.poll(&PollPlan::default()).unwrap();
        assert_eq!(control.optical_position(), 25_050);
    }

    #[test]
    fn relative_and_platform6_profiles_report_what_they_lack() {
        let (mut drv, clock) = connected(SimFocuserConfig::relative());
        let st = drv.state().unwrap();
        assert!(is_unsupported(&st.position));
        assert!(!drv.capabilities().unwrap().absolute);
        drv.move_relative(-300).unwrap();
        clock.advance_ms(1_000);
        drv.poll(&PollPlan::default()).unwrap();
        assert_eq!(drv.control().raw_position(), 24_700);
        assert!(drv.move_absolute(10).is_err());

        let (mut p6, _clock) = connected(SimFocuserConfig::platform6_firmware_backlash());
        let caps = p6.capabilities().unwrap();
        assert_eq!(caps.interface_version, Some(3));
        assert!(!caps.device_state);
        assert!(caps
            .unsupported_members
            .contains(&"devicestate".to_string()));
        let report = p6.connect(Duration::from_secs(1)).unwrap();
        assert!(!report.device_state);
        assert_eq!(p6.state().unwrap().temperature_c.value, Some(12.0));
    }

    #[test]
    fn injected_faults_surface_as_stale_or_errors() {
        let (mut drv, clock) = connected(SimFocuserConfig::absolute());
        let control = drv.control();
        control.fail_next("position", DeviceError::timeout("slow"));
        clock.advance_ms(100);
        let poll = drv.poll(&PollPlan::default()).unwrap();
        assert!(poll.any_failed());
        let st = drv.state().unwrap();
        assert_eq!(st.position.state, ObsState::Stale);
        assert_eq!(st.position.value, Some(25_000));
        control.set_link_down(true);
        assert_eq!(
            drv.poll(&PollPlan::default()).unwrap_err().class,
            ErrorClass::Disconnected
        );
        assert_eq!(drv.state().unwrap().is_moving.state, ObsState::Stale);
        control.set_link_down(false);
        control.set_position_lost(true);
        let poll = drv.poll(&PollPlan::default()).unwrap();
        let err = poll
            .members
            .iter()
            .find(|m| m.name == "position")
            .and_then(|m| m.error.clone())
            .unwrap();
        assert!(err.contains("position unknown"), "{err}");
    }
}

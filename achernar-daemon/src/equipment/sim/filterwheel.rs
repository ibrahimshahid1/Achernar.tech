//! Simulator filter wheel (spec §5.3 WP1A-6, §5.6). Owner: WP1A-6.
//!
//! An in-process `FilterWheelDriver`, also selectable as driver `sim`. Motion
//! is modelled on the injected [`Clock`], so tests drive it with a
//! `ManualClock` and no sleeps: a move takes `move_overhead_ms` plus
//! `ms_per_slot` for every slot passed (one direction only when
//! `unidirectional`, as most wheels turn). While moving it reads like Alpaca
//! Position -1 (`None`).
//!
//! The default wheel reports zero focus offsets for every slot, so a sim
//! wheel connected next to a real focuser never moves it by invented amounts
//! (D17). Tests inject faults through [`SimFilterWheelControl`]: a jammed move
//! that never arrives, a move that lands on the wrong slot, failed polls, a
//! dropped connection.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::equipment::clock::duration_ns;
use crate::equipment::snapshot::SlotInfo;
use crate::equipment::traits::{
    AbortPath, Clock, ConnectPath, ConnectReport, Device, FilterWheelDriver, PollMethod, PollPlan,
    RawMember, RawPoll,
};
use crate::equipment::types::{
    identity_key, DeviceError, DeviceIdentity, DeviceKind, DeviceResult, DriverId, Endpoint,
    ObsState, Observed, Source,
};

/// The standard sim slot names.
pub const DEFAULT_NAMES: [&str; 7] = ["L", "R", "G", "B", "Ha", "OIII", "SII"];

#[derive(Debug, Clone, PartialEq)]
pub struct SimFilterWheelConfig {
    pub names: Vec<String>,
    /// Device `FocusOffsets`; `None` behaves like a driver without offsets.
    pub offsets: Option<Vec<i32>>,
    pub ms_per_slot: u64,
    pub move_overhead_ms: u64,
    pub unidirectional: bool,
    pub initial_slot: u32,
}

impl Default for SimFilterWheelConfig {
    fn default() -> Self {
        Self {
            names: DEFAULT_NAMES.iter().map(|n| n.to_string()).collect(),
            offsets: Some(vec![0; DEFAULT_NAMES.len()]),
            ms_per_slot: 600,
            move_overhead_ms: 200,
            unidirectional: true,
            initial_slot: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fault {
    Jam,
    WrongSlot(u32),
}

#[derive(Debug, Clone, Copy)]
struct Move {
    lands_on: u32,
    arrive_mono_ns: Option<u64>,
}

#[derive(Debug, Default)]
struct SimState {
    slot: u32,
    moving: Option<Move>,
    next_fault: Option<Fault>,
    failing_polls: u32,
    link_down: bool,
    connected: bool,
    commanded: Vec<u32>,
}

fn lock(state: &Mutex<SimState>) -> MutexGuard<'_, SimState> {
    state.lock().unwrap_or_else(|e| e.into_inner())
}

/// Fault injection and inspection, shared with the driver.
#[derive(Debug, Clone)]
pub struct SimFilterWheelControl(Arc<Mutex<SimState>>);

impl SimFilterWheelControl {
    /// The next move never arrives (reads as moving forever).
    pub fn jam_next_move(&self) {
        lock(&self.0).next_fault = Some(Fault::Jam);
    }

    /// The next move lands on `slot` instead of its target.
    pub fn wrong_slot_next_move(&self, slot: u32) {
        lock(&self.0).next_fault = Some(Fault::WrongSlot(slot));
    }

    /// The next `n` polls fail with a transport error.
    pub fn fail_next_polls(&self, n: u32) {
        lock(&self.0).failing_polls = n;
    }

    /// The device stops answering as connected.
    pub fn drop_connection(&self) {
        lock(&self.0).link_down = true;
    }

    /// Every commanded target, in order.
    pub fn commanded(&self) -> Vec<u32> {
        lock(&self.0).commanded.clone()
    }

    /// The physical slot (the slot it will land on while moving).
    pub fn physical_slot(&self) -> u32 {
        let state = lock(&self.0);
        state.moving.map_or(state.slot, |m| m.lands_on)
    }
}

/// The simulator driver.
pub struct SimFilterWheel {
    identity: DeviceIdentity,
    clock: Arc<dyn Clock>,
    config: SimFilterWheelConfig,
    state: Arc<Mutex<SimState>>,
    position: Observed<u32>,
    last_error: Option<DeviceError>,
}

impl SimFilterWheel {
    pub fn new(config: SimFilterWheelConfig, clock: Arc<dyn Clock>) -> Self {
        let identity = DeviceIdentity::new(
            DeviceKind::FilterWheel,
            DriverId::Sim,
            Endpoint::InProcess {
                name: "sim-filter-wheel".into(),
            },
            identity_key::sim(DeviceKind::FilterWheel),
            "Simulated filter wheel",
        );
        let slot_count = u32::try_from(config.names.len()).unwrap_or(u32::MAX);
        let state = SimState {
            slot: config.initial_slot.min(slot_count.saturating_sub(1)),
            ..SimState::default()
        };
        Self {
            identity,
            clock,
            config,
            state: Arc::new(Mutex::new(state)),
            position: Observed::not_set(Source::Property),
            last_error: None,
        }
    }

    /// The standard seven-slot wheel.
    pub fn standard(clock: Arc<dyn Clock>) -> Self {
        Self::new(SimFilterWheelConfig::default(), clock)
    }

    pub fn control(&self) -> SimFilterWheelControl {
        SimFilterWheelControl(Arc::clone(&self.state))
    }

    fn slot_count(&self) -> u32 {
        u32::try_from(self.config.names.len()).unwrap_or(u32::MAX)
    }

    fn travel(&self, from: u32, to: u32) -> Duration {
        let n = self.slot_count().max(1);
        let distance = if self.config.unidirectional {
            (to + n - from % n) % n
        } else {
            from.abs_diff(to)
        };
        if distance == 0 {
            Duration::ZERO
        } else {
            Duration::from_millis(
                self.config.move_overhead_ms + u64::from(distance) * self.config.ms_per_slot,
            )
        }
    }

    fn require_connected(&self) -> DeviceResult<()> {
        let state = lock(&self.state);
        if state.connected && !state.link_down {
            Ok(())
        } else {
            Err(DeviceError::not_connected(
                "the simulated filter wheel is not connected",
            ))
        }
    }
}

impl Device for SimFilterWheel {
    fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    fn connect(&mut self, _timeout: Duration) -> DeviceResult<ConnectReport> {
        let started = self.clock.now_mono_ns();
        {
            let mut state = lock(&self.state);
            state.connected = true;
            state.link_down = false;
        }
        self.identity.description = Some("Achernar in-process filter wheel simulator".into());
        self.identity.driver_info = Some("achernar sim".into());
        self.identity.driver_version = Some(env!("CARGO_PKG_VERSION").into());
        let _ = self.poll(&PollPlan::default());
        Ok(ConnectReport {
            path: ConnectPath::InProcess,
            interface_version: None,
            device_state: false,
            unsupported_members: Vec::new(),
            warnings: Vec::new(),
            duration_ms: self.clock.now_mono_ns().saturating_sub(started) / 1_000_000,
        })
    }

    fn disconnect(&mut self) -> DeviceResult<()> {
        lock(&self.state).connected = false;
        Ok(())
    }

    fn is_connected(&mut self) -> DeviceResult<bool> {
        let state = lock(&self.state);
        Ok(state.connected && !state.link_down)
    }

    fn poll(&mut self, _plan: &PollPlan) -> DeviceResult<RawPoll> {
        let now = self.clock.now_mono_ns();
        let at = self.clock.now_utc();
        let read = {
            let mut state = lock(&self.state);
            if !state.connected || state.link_down {
                Err(DeviceError::not_connected(
                    "the simulated filter wheel is not connected",
                ))
            } else if state.failing_polls > 0 {
                state.failing_polls -= 1;
                Err(DeviceError::transport("simulated poll failure"))
            } else {
                if let Some(motion) = state.moving {
                    if motion.arrive_mono_ns.is_some_and(|t| now >= t) {
                        state.slot = motion.lands_on;
                        state.moving = None;
                    }
                }
                Ok(state.moving.is_none().then_some(state.slot))
            }
        };
        match read {
            Ok(value) => {
                self.position = Observed {
                    value,
                    state: ObsState::Fresh,
                    observed_at: Some(at),
                    mono_ns: Some(now),
                    device_time: None,
                    latency_ms: Some(0),
                    source: Source::Property,
                };
                self.last_error = None;
                Ok(RawPoll {
                    method: PollMethod::InProcess,
                    started_mono_ns: now,
                    finished_mono_ns: now,
                    device_time: None,
                    members: vec![RawMember {
                        name: "position".into(),
                        value: Some(serde_json::json!(value.map_or(-1, i64::from))),
                        state: ObsState::Fresh,
                        observed_at: Some(at),
                        mono_ns: Some(now),
                        latency_ms: Some(0),
                        error: None,
                    }],
                })
            }
            Err(err) => {
                self.position = self.position.clone().into_stale();
                self.last_error = Some(err.clone());
                Err(err)
            }
        }
    }

    fn supported_actions(&mut self) -> DeviceResult<Vec<String>> {
        Ok(Vec::new())
    }

    fn action(&mut self, name: &str, _params: &str) -> DeviceResult<String> {
        Err(DeviceError::unsupported(format!(
            "the simulated filter wheel has no action '{name}'"
        )))
    }

    /// A filter wheel has no halt.
    fn abort_path(&self) -> Option<Arc<dyn AbortPath>> {
        None
    }
}

impl FilterWheelDriver for SimFilterWheel {
    fn slots(&mut self) -> DeviceResult<Vec<SlotInfo>> {
        Ok(self
            .config
            .names
            .iter()
            .enumerate()
            .map(|(i, name)| SlotInfo {
                index0: u32::try_from(i).unwrap_or(u32::MAX),
                device_name: Some(name.clone()),
                device_focus_offset: self
                    .config
                    .offsets
                    .as_ref()
                    .and_then(|offsets| offsets.get(i).copied()),
            })
            .collect())
    }

    fn position(&mut self) -> DeviceResult<Option<u32>> {
        match self.position.state {
            ObsState::Fresh => Ok(self.position.value),
            _ => Err(self.last_error.clone().unwrap_or_else(|| {
                DeviceError::driver("no position has been read from the simulated wheel")
            })),
        }
    }

    fn set_position(&mut self, index0: u32) -> DeviceResult<()> {
        self.require_connected()?;
        if index0 >= self.slot_count() {
            return Err(DeviceError::invalid_value(format!(
                "slot {index0} does not exist; the wheel has {} slots",
                self.slot_count()
            )));
        }
        let now = self.clock.now_mono_ns();
        let mut state = lock(&self.state);
        if state.moving.is_some() {
            return Err(DeviceError::invalid_operation(
                "the simulated wheel is already moving",
            ));
        }
        let travel = self.travel(state.slot, index0);
        let fault = state.next_fault.take();
        let lands_on = match fault {
            Some(Fault::WrongSlot(slot)) => slot.min(self.slot_count() - 1),
            _ => index0,
        };
        let arrive_mono_ns = match fault {
            Some(Fault::Jam) => None,
            _ => Some(now.saturating_add(duration_ns(travel))),
        };
        state.commanded.push(index0);
        state.moving = Some(Move {
            lands_on,
            arrive_mono_ns,
        });
        drop(state);
        // The member is re-read on the next poll (C-2); until then the
        // wheel counts as moving.
        self.position = Observed {
            value: None,
            state: ObsState::Fresh,
            observed_at: Some(self.clock.now_utc()),
            mono_ns: Some(now),
            device_time: None,
            latency_ms: None,
            source: Source::Property,
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::types::ErrorClass;

    fn connected(config: SimFilterWheelConfig) -> (SimFilterWheel, Arc<ManualClock>) {
        let clock = Arc::new(ManualClock::new());
        let mut sim = SimFilterWheel::new(config, Arc::clone(&clock) as Arc<dyn Clock>);
        let report = sim.connect(Duration::from_secs(1)).unwrap();
        assert_eq!(report.path, ConnectPath::InProcess);
        (sim, clock)
    }

    #[test]
    fn the_standard_wheel_reports_seven_named_slots_with_zero_offsets() {
        let (mut sim, _clock) = connected(SimFilterWheelConfig::default());
        let slots = sim.slots().unwrap();
        assert_eq!(slots.len(), 7);
        assert_eq!(slots[4].device_name.as_deref(), Some("Ha"));
        assert!(slots.iter().all(|s| s.device_focus_offset == Some(0)));
        assert_eq!(sim.position().unwrap(), Some(0));
        assert_eq!(sim.identity().identity_key, "sim:filter_wheel");
        assert!(sim.abort_path().is_none(), "a wheel has no halt");
        assert!(sim.is_connected().unwrap());
    }

    #[test]
    fn moves_take_clock_time_and_read_as_moving() {
        let (mut sim, clock) = connected(SimFilterWheelConfig::default());
        sim.set_position(2).unwrap();
        assert_eq!(sim.position().unwrap(), None);
        clock.advance_ms(1000);
        sim.poll(&PollPlan::default()).unwrap();
        assert_eq!(sim.position().unwrap(), None, "1400 ms of travel");
        clock.advance_ms(400);
        let raw = sim.poll(&PollPlan::default()).unwrap();
        assert_eq!(sim.position().unwrap(), Some(2));
        assert_eq!(raw.members[0].value, Some(serde_json::json!(2)));
        // Unidirectional: from 2 back to 1 passes six slots.
        sim.set_position(1).unwrap();
        clock.advance_ms(200 + 5 * 600);
        sim.poll(&PollPlan::default()).unwrap();
        assert_eq!(sim.position().unwrap(), None);
        clock.advance_ms(600);
        sim.poll(&PollPlan::default()).unwrap();
        assert_eq!(sim.position().unwrap(), Some(1));
        // The same slot is immediate.
        sim.set_position(1).unwrap();
        sim.poll(&PollPlan::default()).unwrap();
        assert_eq!(sim.position().unwrap(), Some(1));
    }

    #[test]
    fn invalid_slots_and_overlapping_moves_are_refused() {
        let (mut sim, _clock) = connected(SimFilterWheelConfig {
            names: vec!["A".into(), "B".into()],
            offsets: None,
            unidirectional: false,
            ..SimFilterWheelConfig::default()
        });
        assert_eq!(
            sim.set_position(2).unwrap_err().class,
            ErrorClass::InvalidValue
        );
        sim.set_position(1).unwrap();
        assert_eq!(
            sim.set_position(0).unwrap_err().class,
            ErrorClass::InvalidOperation
        );
        assert!(sim
            .slots()
            .unwrap()
            .iter()
            .all(|s| s.device_focus_offset.is_none()));
    }

    #[test]
    fn faults_are_injected_through_the_control() {
        let (mut sim, clock) = connected(SimFilterWheelConfig::default());
        let control = sim.control();
        control.fail_next_polls(1);
        assert_eq!(
            sim.poll(&PollPlan::default()).unwrap_err().class,
            ErrorClass::Transport
        );
        assert!(sim.position().is_err());
        sim.poll(&PollPlan::default()).unwrap();
        control.wrong_slot_next_move(3);
        sim.set_position(1).unwrap();
        assert_eq!(control.physical_slot(), 3);
        clock.advance_ms(5000);
        sim.poll(&PollPlan::default()).unwrap();
        assert_eq!(sim.position().unwrap(), Some(3));
        control.jam_next_move();
        sim.set_position(4).unwrap();
        clock.advance(Duration::from_secs(600));
        sim.poll(&PollPlan::default()).unwrap();
        assert_eq!(sim.position().unwrap(), None);
        control.drop_connection();
        assert_eq!(
            sim.poll(&PollPlan::default()).unwrap_err().class,
            ErrorClass::NotConnected
        );
        assert!(!sim.is_connected().unwrap());
        assert_eq!(control.commanded(), vec![1, 4]);
    }
}

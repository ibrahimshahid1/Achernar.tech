//! The existing [`SimFocuser`](crate::hardware::focuser::SimFocuser) behind
//! the device worker (§2.3).
//!
//! It keeps the `health_tests.rs` semantics: an absolute focuser over
//! `0..=max_range` (50000 unless a test sets another range), moves that
//! complete at once, and the stalled-driver test hook that keeps `IsMoving`
//! true until a halt. It has no temperature probe, StepSize or TempComp, so
//! those members are `Unsupported`. The `sim` driver id runs the native
//! simulator with a kinematic model (`sim/focuser.rs`); this adapter keeps
//! the older simulator's tests (`health_tests.rs`, the focus fixtures)
//! running through the same worker.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::equipment::snapshot::{FocuserCapabilities, FocuserState};
use crate::equipment::traits::{
    AbortPath, Clock, ConnectPath, ConnectReport, Device, FocuserDriver, PollMethod, PollPlan,
    RawMember, RawPoll,
};
use crate::equipment::types::{
    ascom_error, identity_key, DeviceError, DeviceIdentity, DeviceKind, DeviceResult, DriverId,
    Endpoint, Observed, Source,
};
use crate::error::{AchernarError, HardwareError};
use crate::hardware::focuser::SimFocuser;

/// Map a daemon error from the simulator onto the device vocabulary.
fn map_error(err: AchernarError) -> DeviceError {
    match err {
        AchernarError::Hardware(HardwareError::Timeout(m)) => DeviceError::timeout(m),
        AchernarError::Hardware(HardwareError::Unsupported(m)) => DeviceError::unsupported(m),
        AchernarError::Hardware(HardwareError::DeviceNotFound(m)) => DeviceError::not_connected(m),
        AchernarError::Hardware(HardwareError::DeviceBusy(m)) => DeviceError::invalid_operation(m),
        AchernarError::Hardware(HardwareError::Communication(m)) => DeviceError::transport(m),
        other => DeviceError::driver(other.to_string()),
    }
}

fn lock(sim: &Mutex<SimFocuser>) -> MutexGuard<'_, SimFocuser> {
    sim.lock().unwrap_or_else(|e| e.into_inner())
}

/// The legacy simulator as a [`FocuserDriver`].
pub struct LegacySimFocuser {
    identity: DeviceIdentity,
    sim: Arc<Mutex<SimFocuser>>,
    clock: Arc<dyn Clock>,
    connected: bool,
    state: FocuserState,
}

impl LegacySimFocuser {
    /// A simulator at `position` (the `/connect` option of the `sim` driver).
    pub fn new(position: i32, clock: Arc<dyn Clock>) -> Self {
        Self::from_sim(SimFocuser::new(position), clock)
    }

    /// Wrap an existing simulator (tests build one with a custom range or
    /// the stalled-driver hook).
    pub fn from_sim(sim: SimFocuser, clock: Arc<dyn Clock>) -> Self {
        let identity = DeviceIdentity::new(
            DeviceKind::Focuser,
            DriverId::Sim,
            Endpoint::InProcess {
                name: "legacy-sim-focuser".into(),
            },
            identity_key::sim(DeviceKind::Focuser),
            "Simulator focuser",
        );
        Self {
            identity,
            sim: Arc::new(Mutex::new(sim)),
            clock,
            connected: false,
            state: FocuserState::default(),
        }
    }

    /// Run `f` on the wrapped simulator (test hooks).
    pub fn with_sim<R>(&self, f: impl FnOnce(&mut SimFocuser) -> R) -> R {
        f(&mut lock(&self.sim))
    }

    /// The wrapped simulator, shared (test hooks after the driver moved into
    /// its worker).
    pub fn shared_sim(&self) -> Arc<Mutex<SimFocuser>> {
        Arc::clone(&self.sim)
    }

    fn require_connected(&self) -> DeviceResult<()> {
        if self.connected {
            Ok(())
        } else {
            Err(DeviceError::not_connected(
                "the simulator focuser is not connected",
            ))
        }
    }
}

fn unsupported_members() -> Vec<String> {
    ["temperature", "stepsize", "tempcomp", "devicestate"]
        .iter()
        .map(|m| m.to_string())
        .collect()
}

impl Device for LegacySimFocuser {
    fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    fn connect(&mut self, _timeout: Duration) -> DeviceResult<ConnectReport> {
        self.connected = true;
        Ok(ConnectReport {
            path: ConnectPath::InProcess,
            interface_version: None,
            device_state: false,
            unsupported_members: unsupported_members(),
            warnings: Vec::new(),
            duration_ms: 0,
        })
    }

    fn disconnect(&mut self) -> DeviceResult<()> {
        self.connected = false;
        Ok(())
    }

    fn is_connected(&mut self) -> DeviceResult<bool> {
        Ok(self.connected)
    }

    fn poll(&mut self, _plan: &PollPlan) -> DeviceResult<RawPoll> {
        self.require_connected()?;
        let now_ns = self.clock.now_mono_ns();
        let now_utc = self.clock.now_utc();
        let (position, moving) = {
            let sim = lock(&self.sim);
            (
                sim.position().map_err(map_error)?,
                sim.is_moving().map_err(map_error)?,
            )
        };
        let position = Observed::fresh(position, Source::Property, now_utc, now_ns);
        let is_moving = Observed::fresh(moving, Source::Property, now_utc, now_ns);
        let members = vec![
            RawMember {
                name: "position".into(),
                value: position.value.map(serde_json::Value::from),
                state: position.state,
                observed_at: position.observed_at,
                mono_ns: position.mono_ns,
                latency_ms: None,
                error: None,
            },
            RawMember {
                name: "ismoving".into(),
                value: is_moving.value.map(serde_json::Value::from),
                state: is_moving.state,
                observed_at: is_moving.observed_at,
                mono_ns: is_moving.mono_ns,
                latency_ms: None,
                error: None,
            },
        ];
        self.state = FocuserState {
            position,
            is_moving,
            temperature_c: Observed::unsupported(Source::Property),
            temp_comp: Observed::unsupported(Source::Property),
        };
        Ok(RawPoll {
            method: PollMethod::InProcess,
            started_mono_ns: now_ns,
            finished_mono_ns: now_ns,
            device_time: None,
            members,
        })
    }

    fn supported_actions(&mut self) -> DeviceResult<Vec<String>> {
        Ok(Vec::new())
    }

    fn action(&mut self, name: &str, _params: &str) -> DeviceResult<String> {
        Err(DeviceError::from_alpaca_error(
            ascom_error::ACTION_NOT_IMPLEMENTED,
            &format!("the simulator focuser has no action {name}"),
        ))
    }

    fn abort_path(&self) -> Option<Arc<dyn AbortPath>> {
        Some(Arc::new(LegacySimAbort {
            sim: Arc::clone(&self.sim),
        }))
    }
}

impl FocuserDriver for LegacySimFocuser {
    fn capabilities(&mut self) -> DeviceResult<FocuserCapabilities> {
        let sim = lock(&self.sim);
        let max = sim.max_range().map_err(map_error)?;
        Ok(FocuserCapabilities {
            absolute: sim.is_absolute().map_err(map_error)?,
            max_step: Some(max),
            max_increment: Some(max),
            step_size_um: None,
            temp_comp_available: false,
            can_sync: false,
            interface_version: None,
            device_state: false,
            unsupported_members: unsupported_members(),
        })
    }

    fn state(&mut self) -> DeviceResult<FocuserState> {
        Ok(self.state.clone())
    }

    fn move_absolute(&mut self, target: i32) -> DeviceResult<()> {
        self.require_connected()?;
        lock(&self.sim).move_absolute(target).map_err(map_error)
    }

    fn move_relative(&mut self, _steps: i32) -> DeviceResult<()> {
        Err(DeviceError::invalid_operation(
            "the simulator focuser is absolute; it takes positions, not step counts",
        ))
    }

    fn halt(&mut self) -> DeviceResult<()> {
        lock(&self.sim).abort().map_err(map_error)
    }

    fn set_temp_comp(&mut self, _on: bool) -> DeviceResult<()> {
        Err(DeviceError::unsupported(
            "the simulator focuser has no TempComp",
        ))
    }

    fn sync_position(&mut self, _position: i32) -> DeviceResult<()> {
        Err(DeviceError::unsupported(
            "the simulator focuser has no position sync",
        ))
    }
}

struct LegacySimAbort {
    sim: Arc<Mutex<SimFocuser>>,
}

impl AbortPath for LegacySimAbort {
    fn abort(&self) -> DeviceResult<()> {
        lock(&self.sim).abort().map_err(map_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::types::{ErrorClass, ObsState};

    #[test]
    fn the_legacy_simulator_keeps_its_health_test_semantics() {
        let clock = Arc::new(ManualClock::new());
        let mut drv = LegacySimFocuser::new(12_345, clock);
        assert_eq!(drv.identity().identity_key, "sim:focuser");
        assert_eq!(
            drv.poll(&PollPlan::default()).unwrap_err().class,
            ErrorClass::NotConnected
        );
        drv.connect(Duration::from_secs(1)).unwrap();
        drv.poll(&PollPlan::default()).unwrap();
        let st = drv.state().unwrap();
        assert_eq!(st.position.value, Some(12_345));
        assert_eq!(st.is_moving.value, Some(false));
        assert_eq!(st.temperature_c.state, ObsState::Unsupported);
        let caps = drv.capabilities().unwrap();
        assert!(caps.absolute);
        assert_eq!(caps.max_step, Some(50_000));
        // Moves clamp to the range and complete at once.
        drv.move_absolute(60_000).unwrap();
        drv.poll(&PollPlan::default()).unwrap();
        assert_eq!(drv.state().unwrap().position.value, Some(50_000));
        assert!(drv.move_relative(5).is_err());
    }

    #[test]
    fn the_stalled_driver_hook_is_cleared_by_the_abort_path() {
        let clock = Arc::new(ManualClock::new());
        let mut sim = SimFocuser::with_max_range(100, 2_000);
        sim.set_stuck_moving(true);
        let mut drv = LegacySimFocuser::from_sim(sim, clock);
        drv.connect(Duration::from_secs(1)).unwrap();
        drv.poll(&PollPlan::default()).unwrap();
        assert_eq!(drv.state().unwrap().is_moving.value, Some(true));
        assert_eq!(drv.capabilities().unwrap().max_step, Some(2_000));
        drv.abort_path().unwrap().abort().unwrap();
        drv.poll(&PollPlan::default()).unwrap();
        assert_eq!(drv.state().unwrap().is_moving.value, Some(false));
        assert!(!drv.with_sim(|sim| sim.is_moving().unwrap()));
    }
}

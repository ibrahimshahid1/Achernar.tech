//! Legacy adapter: the OLS focuser behind the device worker (D1), until it
//! is deleted at the end of Tier 1 once the native Alpaca focuser passes
//! acceptance (O2). Its code is not kept as a fallback after that.
//!
//! What OLS can supply is limited (§7.1, column O): position, IsMoving,
//! absolute, MaxStep (`max_range`), move and halt. Temperature, StepSize and
//! TempComp are `Unsupported`, and a relative OLS focuser's counter is not
//! persisted, so its position is reported as unsupported and the session
//! emulates it like any relative focuser.
//!
//! OLS calls are blocking FFI. They run only on the worker thread, and the
//! abort path (OLS `abort`) shares the backend through a mutex; OLS
//! serialises calls internally, so an abort can wait behind one short call,
//! which the worker records as abort latency (D4).

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
use crate::hardware::focuser::{Focuser, FocuserDriver as OlsDriverKind};

/// Map a daemon error from the OLS wrappers onto the device vocabulary.
pub fn map_error(err: AchernarError) -> DeviceError {
    match err {
        AchernarError::Hardware(HardwareError::Timeout(m)) => DeviceError::timeout(m),
        AchernarError::Hardware(HardwareError::Unsupported(m)) => DeviceError::unsupported(m),
        AchernarError::Hardware(HardwareError::DeviceNotFound(m)) => DeviceError::not_connected(m),
        AchernarError::Hardware(HardwareError::DeviceBusy(m)) => DeviceError::invalid_operation(m),
        AchernarError::Hardware(HardwareError::Communication(m)) => DeviceError::transport(m),
        other => DeviceError::driver(other.to_string()),
    }
}

/// What the adapter needs from OLS. Implemented by the OLS
/// [`Focuser`](crate::hardware::focuser::Focuser); tests use a scripted
/// backend, because loading an OLS driver would reach real hardware.
pub trait OlsFocuserBackend: Send {
    fn load(&mut self, option: &str) -> crate::error::Result<()>;
    fn is_loaded(&self) -> bool;
    fn position(&self) -> crate::error::Result<i32>;
    fn is_moving(&self) -> crate::error::Result<bool>;
    fn is_absolute(&self) -> crate::error::Result<bool>;
    fn max_range(&self) -> crate::error::Result<i32>;
    fn move_absolute(&mut self, position: i32) -> crate::error::Result<()>;
    fn move_relative(&mut self, steps: i32) -> crate::error::Result<()>;
    fn abort(&mut self) -> crate::error::Result<()>;
}

impl OlsFocuserBackend for Focuser {
    fn load(&mut self, option: &str) -> crate::error::Result<()> {
        self.connect_with_option(option)
    }
    fn is_loaded(&self) -> bool {
        self.is_connected()
    }
    fn position(&self) -> crate::error::Result<i32> {
        Focuser::position(self)
    }
    fn is_moving(&self) -> crate::error::Result<bool> {
        Focuser::is_moving(self)
    }
    fn is_absolute(&self) -> crate::error::Result<bool> {
        Focuser::is_absolute(self)
    }
    fn max_range(&self) -> crate::error::Result<i32> {
        Focuser::max_range(self)
    }
    fn move_absolute(&mut self, position: i32) -> crate::error::Result<()> {
        Focuser::move_absolute(self, position)
    }
    fn move_relative(&mut self, steps: i32) -> crate::error::Result<()> {
        Focuser::move_relative(self, steps)
    }
    fn abort(&mut self) -> crate::error::Result<()> {
        Focuser::abort(self)
    }
}

fn lock<B>(backend: &Mutex<B>) -> MutexGuard<'_, B> {
    backend.lock().unwrap_or_else(|e| e.into_inner())
}

/// Split an OLS Alpaca option `http(s)://host:port#device=N` into the base
/// URL, host, port and device number.
pub fn parse_alpaca_option(option: &str) -> Option<(String, String, u16, u32)> {
    let option = option.trim();
    let (base, fragment) = match option.split_once('#') {
        Some((base, fragment)) => (base, Some(fragment)),
        None => (option, None),
    };
    let (scheme, rest) = base.split_once("://")?;
    let default_port = match scheme.to_ascii_lowercase().as_str() {
        "http" => 80,
        "https" => 443,
        _ => return None,
    };
    let authority = rest.split('/').next().unwrap_or_default();
    if authority.is_empty() {
        return None;
    }
    let (host, port) = match authority.rsplit_once(':') {
        Some((host, port)) if !host.is_empty() => (host, port.parse().ok()?),
        _ => (authority, default_port),
    };
    let number = fragment
        .and_then(|f| f.strip_prefix("device="))
        .map_or(Some(0), |n| n.trim().parse().ok())?;
    Some((
        format!("{scheme}://{authority}"),
        host.to_string(),
        port,
        number,
    ))
}

/// The OLS focuser as a [`FocuserDriver`].
pub struct OlsFocuserAdapter<B: OlsFocuserBackend + 'static> {
    identity: DeviceIdentity,
    backend: Arc<Mutex<B>>,
    option: String,
    clock: Arc<dyn Clock>,
    connected: bool,
    absolute: bool,
    max_range: Option<i32>,
    state: FocuserState,
}

impl OlsFocuserAdapter<Focuser> {
    /// Create the OLS interface for an Alpaca option. Does not connect.
    pub fn open(option: &str, clock: Arc<dyn Clock>) -> DeviceResult<Self> {
        let focuser = Focuser::new(OlsDriverKind::Alpaca).map_err(map_error)?;
        Ok(Self::with_backend(focuser, option, clock))
    }
}

impl<B: OlsFocuserBackend + 'static> OlsFocuserAdapter<B> {
    pub fn with_backend(backend: B, option: &str, clock: Arc<dyn Clock>) -> Self {
        let (endpoint, key) = match parse_alpaca_option(option) {
            Some((base, host, port, number)) => (
                Endpoint::AlpacaHttp {
                    base,
                    device_type: "focuser".into(),
                    number,
                },
                // The same key as the native driver, so the identity profile
                // carries over when the device moves to `alpaca`.
                identity_key::alpaca(None, &host, port, "focuser", number),
            ),
            None => (
                Endpoint::InProcess {
                    name: format!("ols:{option}"),
                },
                format!("ols_alpaca:{option}"),
            ),
        };
        let mut identity = DeviceIdentity::new(
            DeviceKind::Focuser,
            DriverId::OlsAlpaca,
            endpoint,
            key,
            "OLS focuser",
        );
        identity.driver_info = Some("OpenLiveStacker focuser (legacy adapter)".into());
        Self {
            identity,
            backend: Arc::new(Mutex::new(backend)),
            option: option.to_string(),
            clock,
            connected: false,
            absolute: false,
            max_range: None,
            state: FocuserState::default(),
        }
    }

    fn require_connected(&self) -> DeviceResult<()> {
        if self.connected {
            Ok(())
        } else {
            Err(DeviceError::not_connected(
                "the OLS focuser is not connected",
            ))
        }
    }
}

fn raw_member<T: serde::Serialize>(
    name: &str,
    observed: &Observed<T>,
    error: Option<String>,
) -> RawMember {
    RawMember {
        name: name.into(),
        value: observed
            .value
            .as_ref()
            .and_then(|v| serde_json::to_value(v).ok()),
        state: observed.state,
        observed_at: observed.observed_at,
        mono_ns: observed.mono_ns,
        latency_ms: observed.latency_ms,
        error,
    }
}

impl<B: OlsFocuserBackend + 'static> Device for OlsFocuserAdapter<B> {
    fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    fn connect(&mut self, _timeout: Duration) -> DeviceResult<ConnectReport> {
        let started = self.clock.now_mono_ns();
        let (absolute, max_range) = {
            let mut backend = lock(&self.backend);
            if !backend.is_loaded() {
                backend.load(&self.option).map_err(map_error)?;
            }
            let absolute = backend.is_absolute().map_err(map_error)?;
            (absolute, backend.max_range().ok().filter(|m| *m > 0))
        };
        self.absolute = absolute;
        self.max_range = max_range;
        self.connected = true;
        let mut unsupported = vec![
            "temperature".to_string(),
            "stepsize".into(),
            "tempcomp".into(),
            "devicestate".into(),
        ];
        let mut warnings = Vec::new();
        if !absolute {
            unsupported.push("position".into());
            warnings.push(
                "the OLS focuser is relative and its counter is not persisted; the position \
                 is emulated by Achernar"
                    .into(),
            );
        }
        Ok(ConnectReport {
            path: ConnectPath::InProcess,
            interface_version: None,
            device_state: false,
            unsupported_members: unsupported,
            warnings,
            duration_ms: self.clock.now_mono_ns().saturating_sub(started) / 1_000_000,
        })
    }

    /// OLS has no unload; dropping the adapter destroys the interface.
    fn disconnect(&mut self) -> DeviceResult<()> {
        self.connected = false;
        Ok(())
    }

    fn is_connected(&mut self) -> DeviceResult<bool> {
        Ok(self.connected && lock(&self.backend).is_loaded())
    }

    fn poll(&mut self, _plan: &PollPlan) -> DeviceResult<RawPoll> {
        self.require_connected()?;
        let started = self.clock.now_mono_ns();
        let now_utc = self.clock.now_utc();
        let (position, moving) = {
            let backend = lock(&self.backend);
            let position = self.absolute.then(|| backend.position());
            (position, backend.is_moving())
        };
        let finished = self.clock.now_mono_ns();
        let mut members = Vec::new();
        let mut transport_error = None;
        let position = match position {
            None => Observed::unsupported(Source::Property),
            Some(Ok(value)) => {
                let obs = Observed::fresh(value, Source::Property, now_utc, finished);
                members.push(raw_member("position", &obs, None));
                obs
            }
            Some(Err(err)) => {
                let err = map_error(err);
                members.push(raw_member(
                    "position",
                    &Observed::<i32>::error(Source::Property),
                    Some(err.to_string()),
                ));
                self.state.position.clone().into_stale()
            }
        };
        let is_moving = match moving {
            Ok(value) => {
                let obs = Observed::fresh(value, Source::Property, now_utc, finished);
                members.push(raw_member("ismoving", &obs, None));
                obs
            }
            Err(err) => {
                let err = map_error(err);
                members.push(raw_member(
                    "ismoving",
                    &Observed::<bool>::error(Source::Property),
                    Some(err.to_string()),
                ));
                transport_error = Some(err);
                self.state.is_moving.clone().into_stale()
            }
        };
        self.state = FocuserState {
            position,
            is_moving,
            temperature_c: Observed::unsupported(Source::Property),
            temp_comp: Observed::unsupported(Source::Property),
        };
        if let Some(err) = transport_error {
            return Err(err);
        }
        Ok(RawPoll {
            method: PollMethod::InProcess,
            started_mono_ns: started,
            finished_mono_ns: finished,
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
            &format!("the OLS focuser has no action {name}"),
        ))
    }

    fn abort_path(&self) -> Option<Arc<dyn AbortPath>> {
        Some(Arc::new(OlsFocuserAbort {
            backend: Arc::clone(&self.backend),
        }))
    }
}

impl<B: OlsFocuserBackend + 'static> FocuserDriver for OlsFocuserAdapter<B> {
    fn capabilities(&mut self) -> DeviceResult<FocuserCapabilities> {
        self.require_connected()?;
        let mut unsupported = vec![
            "temperature".to_string(),
            "stepsize".into(),
            "tempcomp".into(),
            "devicestate".into(),
        ];
        if !self.absolute {
            unsupported.push("position".into());
        }
        Ok(FocuserCapabilities {
            absolute: self.absolute,
            max_step: self.max_range,
            max_increment: self.max_range,
            step_size_um: None,
            temp_comp_available: false,
            can_sync: false,
            interface_version: None,
            device_state: false,
            unsupported_members: unsupported,
        })
    }

    fn state(&mut self) -> DeviceResult<FocuserState> {
        Ok(self.state.clone())
    }

    fn move_absolute(&mut self, target: i32) -> DeviceResult<()> {
        self.require_connected()?;
        if !self.absolute {
            return Err(DeviceError::invalid_operation(
                "a relative focuser takes step counts, not positions",
            ));
        }
        lock(&self.backend).move_absolute(target).map_err(map_error)
    }

    fn move_relative(&mut self, steps: i32) -> DeviceResult<()> {
        self.require_connected()?;
        if self.absolute {
            return Err(DeviceError::invalid_operation(
                "an absolute focuser takes positions, not step counts",
            ));
        }
        lock(&self.backend).move_relative(steps).map_err(map_error)
    }

    fn halt(&mut self) -> DeviceResult<()> {
        lock(&self.backend).abort().map_err(map_error)
    }

    fn set_temp_comp(&mut self, _on: bool) -> DeviceResult<()> {
        Err(DeviceError::unsupported("the OLS focuser has no TempComp"))
    }

    fn sync_position(&mut self, _position: i32) -> DeviceResult<()> {
        Err(DeviceError::unsupported(
            "the OLS focuser has no position sync",
        ))
    }
}

struct OlsFocuserAbort<B: OlsFocuserBackend> {
    backend: Arc<Mutex<B>>,
}

impl<B: OlsFocuserBackend> AbortPath for OlsFocuserAbort<B> {
    fn abort(&self) -> DeviceResult<()> {
        lock(&self.backend).abort().map_err(map_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::types::{ErrorClass, ObsState};

    fn all_known(state: &FocuserState) -> bool {
        [state.position.state, state.is_moving.state]
            .iter()
            .all(|s| matches!(s, ObsState::Fresh | ObsState::Unsupported))
    }

    /// A scripted OLS backend: instantaneous moves, optional failures.
    #[derive(Default)]
    struct ScriptedOls {
        loaded: bool,
        absolute: bool,
        position: i32,
        loads: Vec<String>,
        aborts: u32,
        fail_moving: bool,
    }

    impl OlsFocuserBackend for ScriptedOls {
        fn load(&mut self, option: &str) -> crate::error::Result<()> {
            self.loads.push(option.to_string());
            self.loaded = true;
            Ok(())
        }
        fn is_loaded(&self) -> bool {
            self.loaded
        }
        fn position(&self) -> crate::error::Result<i32> {
            Ok(self.position)
        }
        fn is_moving(&self) -> crate::error::Result<bool> {
            if self.fail_moving {
                Err(HardwareError::Communication("server gone".into()).into())
            } else {
                Ok(false)
            }
        }
        fn is_absolute(&self) -> crate::error::Result<bool> {
            Ok(self.absolute)
        }
        fn max_range(&self) -> crate::error::Result<i32> {
            Ok(30_000)
        }
        fn move_absolute(&mut self, position: i32) -> crate::error::Result<()> {
            self.position = position;
            Ok(())
        }
        fn move_relative(&mut self, steps: i32) -> crate::error::Result<()> {
            self.position += steps;
            Ok(())
        }
        fn abort(&mut self) -> crate::error::Result<()> {
            self.aborts += 1;
            Ok(())
        }
    }

    fn adapter(absolute: bool) -> OlsFocuserAdapter<ScriptedOls> {
        let clock = Arc::new(ManualClock::new());
        OlsFocuserAdapter::with_backend(
            ScriptedOls {
                absolute,
                position: 1_000,
                ..ScriptedOls::default()
            },
            "http://10.0.0.7:11111#device=1",
            clock,
        )
    }

    #[test]
    fn options_parse_into_the_native_identity_key() {
        assert_eq!(
            parse_alpaca_option("http://ESP32.local:80#device=2"),
            Some(("http://ESP32.local:80".into(), "ESP32.local".into(), 80, 2))
        );
        assert_eq!(
            parse_alpaca_option("https://host"),
            Some(("https://host".into(), "host".into(), 443, 0))
        );
        assert_eq!(parse_alpaca_option("auto"), None);
        assert_eq!(parse_alpaca_option("http://host:x#device=1"), None);
        let drv = adapter(true);
        assert_eq!(
            drv.identity().identity_key,
            "alpaca:10.0.0.7:11111/focuser/1"
        );
        assert_eq!(drv.identity().driver, DriverId::OlsAlpaca);
    }

    #[test]
    fn the_adapter_reports_what_ols_supplies_and_marks_the_rest_unsupported() {
        let mut drv = adapter(true);
        assert_eq!(
            drv.move_absolute(5).unwrap_err().class,
            ErrorClass::NotConnected
        );
        let report = drv.connect(Duration::from_secs(1)).unwrap();
        assert!(report
            .unsupported_members
            .contains(&"temperature".to_string()));
        let caps = drv.capabilities().unwrap();
        assert!(caps.absolute);
        assert_eq!(caps.max_step, Some(30_000));
        assert!(!caps.temp_comp_available);
        drv.poll(&PollPlan::default()).unwrap();
        let st = drv.state().unwrap();
        assert!(all_known(&st));
        assert_eq!(st.position.value, Some(1_000));
        assert_eq!(st.temperature_c.state, ObsState::Unsupported);
        drv.move_absolute(1_500).unwrap();
        drv.poll(&PollPlan::default()).unwrap();
        assert_eq!(drv.state().unwrap().position.value, Some(1_500));
        assert_eq!(
            drv.set_temp_comp(true).unwrap_err().class,
            ErrorClass::Unsupported
        );
        drv.abort_path().unwrap().abort().unwrap();
        assert_eq!(lock(&drv.backend).aborts, 1);
        assert_eq!(
            lock(&drv.backend).loads,
            vec!["http://10.0.0.7:11111#device=1"]
        );
    }

    #[test]
    fn a_relative_ols_focuser_has_no_position_and_failures_go_stale() {
        let mut drv = adapter(false);
        drv.connect(Duration::from_secs(1)).unwrap();
        drv.poll(&PollPlan::default()).unwrap();
        assert_eq!(drv.state().unwrap().position.state, ObsState::Unsupported);
        assert!(drv.move_absolute(10).is_err());
        drv.move_relative(-40).unwrap();
        lock(&drv.backend).fail_moving = true;
        let err = drv.poll(&PollPlan::default()).unwrap_err();
        assert_eq!(err.class, ErrorClass::Transport);
        assert_eq!(drv.state().unwrap().is_moving.state, ObsState::Stale);
        assert_eq!(
            map_error(HardwareError::Timeout("x".into()).into()).class,
            ErrorClass::Timeout
        );
    }
}

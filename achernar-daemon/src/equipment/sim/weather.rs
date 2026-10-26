//! Simulated ObservingConditions device (spec §5.6, §7.1 R18). Owner:
//! WP1A-8.
//!
//! An in-process `WeatherDriver`, also selectable as driver `sim`. A
//! [`SimWeatherControl`] (cloneable) scripts it from tests or a bench UI:
//! per-field values, NaN, ValueNotSet and NotImplemented, per-sensor ages,
//! injected poll failures and a dropped connection. It follows the same
//! `Observed` rules as the Alpaca driver: an unknown value is `None`, never a
//! number, and a failed poll keeps the last value as `Stale`.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::equipment::alpaca::observingconditions::WeatherField;
use crate::equipment::snapshot::WeatherReading;
use crate::equipment::traits::{
    AbortPath, Clock, ConnectPath, ConnectReport, Device, PollMethod, PollPlan, RawMember, RawPoll,
    WeatherDriver,
};
use crate::equipment::types::{
    identity_key, DeviceError, DeviceIdentity, DeviceKind, DeviceResult, DriverId, Endpoint,
    ObsState, Observed, Source,
};

/// What the simulator reports for one field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimValue {
    /// A reading; NaN is reported as unknown, exactly as a real device's NaN.
    Value(f64),
    /// ASCOM ValueNotSet.
    NotSet,
    /// ASCOM NotImplemented.
    Unsupported,
}

#[derive(Debug)]
struct SimWeatherState {
    fields: BTreeMap<WeatherField, SimValue>,
    /// Seconds since each sensor's last update.
    ages_s: BTreeMap<WeatherField, f64>,
    fail_polls: u32,
    failing: bool,
    reachable: bool,
    polls: u64,
    refreshes: u32,
}

impl Default for SimWeatherState {
    /// A plausible clear night: every field implemented, sensors current.
    fn default() -> Self {
        use WeatherField::*;
        let fields = [
            (CloudCover, 5.0),
            (DewPoint, 1.5),
            (Humidity, 72.0),
            (Pressure, 1013.25),
            (RainRate, 0.0),
            (SkyBrightness, 0.001),
            (SkyQuality, 20.8),
            (SkyTemperature, -18.0),
            (StarFwhm, 2.4),
            (Temperature, 6.5),
            (WindDirection, 225.0),
            (WindGust, 3.5),
            (WindSpeed, 1.8),
        ]
        .into_iter()
        .map(|(field, value)| (field, SimValue::Value(value)))
        .collect();
        Self {
            fields,
            ages_s: BTreeMap::new(),
            fail_polls: 0,
            failing: false,
            reachable: true,
            polls: 0,
            refreshes: 0,
        }
    }
}

/// Script handle for a [`SimWeather`]; clones share one device.
#[derive(Debug, Clone, Default)]
pub struct SimWeatherControl {
    state: Arc<Mutex<SimWeatherState>>,
}

impl SimWeatherControl {
    pub fn new() -> Self {
        Self::default()
    }

    fn state(&self) -> MutexGuard<'_, SimWeatherState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn set(&self, field: WeatherField, value: f64) {
        self.state().fields.insert(field, SimValue::Value(value));
    }

    pub fn set_value(&self, field: WeatherField, value: SimValue) {
        self.state().fields.insert(field, value);
    }

    pub fn set_not_set(&self, field: WeatherField) {
        self.set_value(field, SimValue::NotSet);
    }

    pub fn set_unsupported(&self, field: WeatherField) {
        self.set_value(field, SimValue::Unsupported);
    }

    /// Seconds since the sensor last updated (`timesincelastupdate`).
    pub fn set_age(&self, field: WeatherField, seconds: f64) {
        self.state().ages_s.insert(field, seconds);
    }

    /// The next `n` polls fail with a timeout.
    pub fn fail_next_polls(&self, n: u32) {
        self.state().fail_polls = n;
    }

    /// Every poll fails until cleared.
    pub fn set_failing(&self, failing: bool) {
        self.state().failing = failing;
    }

    /// False: the device drops the connection (`Connected` reads false).
    pub fn set_reachable(&self, reachable: bool) {
        self.state().reachable = reachable;
    }

    pub fn polls(&self) -> u64 {
        self.state().polls
    }

    pub fn refreshes(&self) -> u32 {
        self.state().refreshes
    }
}

/// The simulated weather device.
pub struct SimWeather {
    identity: DeviceIdentity,
    clock: Arc<dyn Clock>,
    control: SimWeatherControl,
    connected: bool,
    reading: WeatherReading,
}

impl SimWeather {
    pub fn new(clock: Arc<dyn Clock>, control: SimWeatherControl) -> Self {
        let identity = DeviceIdentity::new(
            DeviceKind::Weather,
            DriverId::Sim,
            Endpoint::InProcess {
                name: "sim-weather".into(),
            },
            identity_key::sim(DeviceKind::Weather),
            "Simulated weather station",
        );
        Self::with_identity(identity, clock, control)
    }

    pub fn with_identity(
        identity: DeviceIdentity,
        clock: Arc<dyn Clock>,
        control: SimWeatherControl,
    ) -> Self {
        Self {
            identity,
            clock,
            control,
            connected: false,
            reading: WeatherReading::default(),
        }
    }

    pub fn control(&self) -> SimWeatherControl {
        self.control.clone()
    }

    fn require_connected(&self) -> DeviceResult<()> {
        if self.connected {
            Ok(())
        } else {
            Err(DeviceError::not_connected(
                "the simulated weather station is not connected",
            ))
        }
    }

    fn mark_all_stale(&mut self) {
        for field in WeatherField::ALL {
            let stale = field.get(&self.reading).clone().into_stale();
            *field.get_mut(&mut self.reading) = stale;
        }
    }
}

impl Device for SimWeather {
    fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    fn connect(&mut self, _timeout: Duration) -> DeviceResult<ConnectReport> {
        let state = self.control.state();
        if !state.reachable {
            return Err(DeviceError::disconnected(
                "the simulated weather station is unreachable",
            ));
        }
        let unsupported_members = state
            .fields
            .iter()
            .filter(|(_, value)| **value == SimValue::Unsupported)
            .map(|(field, _)| field.member().to_string())
            .collect();
        drop(state);
        self.connected = true;
        self.reading = WeatherReading::default();
        for field in WeatherField::ALL {
            self.reading.sensor_descriptions.insert(
                field.sensor_name().to_string(),
                format!("Simulated {} sensor", field.sensor_name()),
            );
        }
        Ok(ConnectReport {
            path: ConnectPath::InProcess,
            interface_version: Some(2),
            unsupported_members,
            ..ConnectReport::default()
        })
    }

    fn disconnect(&mut self) -> DeviceResult<()> {
        self.connected = false;
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
        let mut state = self.control.state();
        state.polls += 1;
        if !state.reachable {
            drop(state);
            self.connected = false;
            self.mark_all_stale();
            return Err(DeviceError::disconnected(
                "the simulated weather station dropped the connection",
            ));
        }
        if state.failing || state.fail_polls > 0 {
            state.fail_polls = state.fail_polls.saturating_sub(1);
            drop(state);
            self.mark_all_stale();
            return Err(DeviceError::timeout(
                "simulated weather station: injected poll failure",
            ));
        }
        let fields = state.fields.clone();
        let ages = state.ages_s.clone();
        drop(state);

        let mut members = Vec::new();
        self.reading.sensor_ages_s.clear();
        for field in WeatherField::ALL {
            let value = fields.get(&field).copied().unwrap_or(SimValue::Unsupported);
            let (value, obs_state) = match value {
                SimValue::Value(v) if v.is_finite() => (Some(v), ObsState::Fresh),
                SimValue::Value(_) | SimValue::NotSet => (None, ObsState::NotSet),
                SimValue::Unsupported => (None, ObsState::Unsupported),
            };
            let mut observed = match obs_state {
                ObsState::Unsupported => Observed::unsupported(Source::Property),
                _ => Observed {
                    value,
                    state: obs_state,
                    observed_at: Some(now),
                    mono_ns: Some(mono),
                    device_time: None,
                    latency_ms: Some(0),
                    source: Source::Property,
                },
            };
            if obs_state != ObsState::Unsupported {
                let age = ages.get(&field).copied().unwrap_or(0.0).max(0.0);
                observed.device_time =
                    Some(now - chrono::Duration::milliseconds((age * 1000.0).round() as i64));
                self.reading
                    .sensor_ages_s
                    .insert(field.sensor_name().to_string(), age);
            }
            members.push(RawMember {
                name: field.member().to_string(),
                value: value.map(serde_json::Value::from),
                state: obs_state,
                observed_at: Some(now),
                mono_ns: Some(mono),
                latency_ms: Some(0),
                error: None,
            });
            *field.get_mut(&mut self.reading) = observed;
        }
        Ok(RawPoll {
            method: PollMethod::InProcess,
            started_mono_ns: mono,
            finished_mono_ns: self.clock.now_mono_ns(),
            device_time: Some(now),
            members,
        })
    }

    fn supported_actions(&mut self) -> DeviceResult<Vec<String>> {
        Ok(Vec::new())
    }

    fn action(&mut self, name: &str, _params: &str) -> DeviceResult<String> {
        Err(DeviceError::unsupported(format!(
            "the simulated weather station has no action {name}"
        )))
    }

    fn abort_path(&self) -> Option<Arc<dyn AbortPath>> {
        None
    }
}

impl WeatherDriver for SimWeather {
    fn read(&mut self) -> DeviceResult<WeatherReading> {
        self.require_connected()?;
        Ok(self.reading.clone())
    }

    fn refresh(&mut self) -> DeviceResult<()> {
        self.require_connected()?;
        self.control.state().refreshes += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::types::ErrorClass;

    fn sim() -> (SimWeather, SimWeatherControl, Arc<ManualClock>) {
        let clock = Arc::new(ManualClock::new());
        let control = SimWeatherControl::new();
        let sim = SimWeather::new(Arc::clone(&clock) as Arc<dyn Clock>, control.clone());
        (sim, control, clock)
    }

    #[test]
    fn the_sim_reports_every_field_with_ages_and_unknowns() {
        let (mut sim, control, clock) = sim();
        control.set(WeatherField::Temperature, f64::NAN);
        control.set_not_set(WeatherField::Humidity);
        control.set_unsupported(WeatherField::StarFwhm);
        control.set_age(WeatherField::CloudCover, 30.0);
        let report = sim.connect(Duration::from_secs(1)).unwrap();
        assert_eq!(report.unsupported_members, vec!["starfwhm".to_string()]);
        assert_eq!(sim.identity().identity_key, "sim:weather");

        let raw = sim.poll(&PollPlan::default()).unwrap();
        assert_eq!(raw.members.len(), 13);
        let reading = sim.read().unwrap();
        assert_eq!(reading.temperature_c.value, None, "NaN is unknown");
        assert_eq!(reading.temperature_c.state, ObsState::NotSet);
        assert_eq!(reading.humidity_pct.state, ObsState::NotSet);
        assert_eq!(reading.star_fwhm_arcsec.state, ObsState::Unsupported);
        assert_eq!(reading.cloud_cover_pct.value, Some(5.0));
        assert_eq!(reading.sensor_ages_s["CloudCover"], 30.0);
        assert_eq!(
            reading.cloud_cover_pct.device_time,
            Some(clock.now_utc() - chrono::Duration::seconds(30))
        );
        assert!(!reading.sensor_ages_s.contains_key("StarFWHM"));
        sim.refresh().unwrap();
        assert_eq!(control.refreshes(), 1);
    }

    #[test]
    fn injected_failures_keep_values_stale_and_a_drop_disconnects() {
        let (mut sim, control, _clock) = sim();
        sim.connect(Duration::from_secs(1)).unwrap();
        sim.poll(&PollPlan::default()).unwrap();
        control.fail_next_polls(1);
        assert_eq!(
            sim.poll(&PollPlan::default()).unwrap_err().class,
            ErrorClass::Timeout
        );
        let reading = sim.read().unwrap();
        assert_eq!(reading.wind_speed_ms.state, ObsState::Stale);
        assert_eq!(reading.wind_speed_ms.value, Some(1.8));
        assert!(sim.poll(&PollPlan::default()).is_ok());

        control.set_reachable(false);
        assert_eq!(
            sim.poll(&PollPlan::default()).unwrap_err().class,
            ErrorClass::Disconnected
        );
        assert!(!sim.is_connected().unwrap());
        assert!(sim.read().is_err());
        assert!(sim.connect(Duration::from_secs(1)).is_err());
        control.set_reachable(true);
        assert!(sim.connect(Duration::from_secs(1)).is_ok());
        assert_eq!(control.polls(), 4);
    }
}

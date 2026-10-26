//! Alpaca ObservingConditions driver (spec §3.9, §7.1 R18). Owner: WP1A-8.
//!
//! An independent implementation of `WeatherDriver` from the ASCOM Alpaca API
//! reference and the `IObservingConditions` master interface (MIT). No NINA
//! code is used.
//!
//! - All 13 fields are read. NaN, `null`, ValueNotSet (0x402) and
//!   NotImplemented (0x400, or HTTP 400/404 from an older server) become an
//!   unknown value, never a number (D5).
//! - A member that is not implemented is memoised as `Unsupported` for the
//!   session (§2.6 step 6) and never read again.
//! - Each sensor's age comes from `timesincelastupdate?SensorName=`, refreshed
//!   every [`SENSOR_AGE_INTERVAL`] (or when the poll plan asks for
//!   `timesincelastupdate`). The sensor's last update time becomes the
//!   field's `device_time`; between refreshes the estimate is conservative
//!   (it can only make a value look older). `sensordescription` is captured
//!   once, at connect.
//! - Platform 7 (InterfaceVersion >= 2) connects with `connect` and
//!   `connecting` and polls one `devicestate` GET when the plan prefers it;
//!   Platform 6 uses `Connected=true` and per-member reads.
//!
//! The file also holds [`AlpacaLink`], the connection-level half of an Alpaca
//! driver (connect paths, identity, actions, `devicestate` parsing), shared
//! with `safetymonitor.rs`. WP1A-1's client may later offer the same helpers;
//! until then these two drivers carry their own (§5.1 rule 9).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Local, NaiveDateTime, TimeZone, Utc};
use serde_json::Value;

use crate::equipment::snapshot::WeatherReading;
use crate::equipment::traits::{
    AbortPath, AlpacaTransport, Clock, ConnectPath, ConnectReport, Device, PollMethod, PollPlan,
    RawMember, RawPoll, TimeoutClass, WeatherDriver,
};
use crate::equipment::types::{
    DeviceError, DeviceIdentity, DeviceResult, ErrorClass, ObsState, Observed, Source,
};

/// ObservingConditions interface version at which Platform 7 connect and
/// `devicestate` apply (ASCOM Library `DeviceCapabilities`).
pub const OBSERVING_CONDITIONS_PLATFORM7: i32 = 2;

/// Idle poll cadence (§2.7 table: weather every 10 s).
pub const WEATHER_POLL_INTERVAL: Duration = Duration::from_secs(10);

/// How often `timesincelastupdate` is read per implemented sensor (§2.7).
pub const SENSOR_AGE_INTERVAL: Duration = Duration::from_secs(60);

/// The `PollPlan::extra_members` name that forces a sensor-age refresh.
pub const SENSOR_AGES_MEMBER: &str = "timesincelastupdate";

/// Default wait between `connecting` reads on the Platform 7 path (§2.6).
pub const CONNECT_POLL_INTERVAL: Duration = Duration::from_millis(250);

// ---------------------------------------------------------------------------
// The 13 ObservingConditions fields
// ---------------------------------------------------------------------------

/// One ObservingConditions field (R18). Units follow the ASCOM interface,
/// which matches [`WeatherReading`]: %, °C, hPa, mm/h, lux, mag/arcsec²,
/// arcsec, degrees and m/s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum WeatherField {
    CloudCover,
    DewPoint,
    Humidity,
    Pressure,
    RainRate,
    SkyBrightness,
    SkyQuality,
    SkyTemperature,
    StarFwhm,
    Temperature,
    WindDirection,
    WindGust,
    WindSpeed,
}

impl WeatherField {
    pub const ALL: [WeatherField; 13] = [
        WeatherField::CloudCover,
        WeatherField::DewPoint,
        WeatherField::Humidity,
        WeatherField::Pressure,
        WeatherField::RainRate,
        WeatherField::SkyBrightness,
        WeatherField::SkyQuality,
        WeatherField::SkyTemperature,
        WeatherField::StarFwhm,
        WeatherField::Temperature,
        WeatherField::WindDirection,
        WeatherField::WindGust,
        WeatherField::WindSpeed,
    ];

    /// The ASCOM property and sensor name, exact case (`StarFWHM`). It is
    /// also the `SensorName` parameter and the `devicestate` entry name.
    pub fn sensor_name(self) -> &'static str {
        match self {
            WeatherField::CloudCover => "CloudCover",
            WeatherField::DewPoint => "DewPoint",
            WeatherField::Humidity => "Humidity",
            WeatherField::Pressure => "Pressure",
            WeatherField::RainRate => "RainRate",
            WeatherField::SkyBrightness => "SkyBrightness",
            WeatherField::SkyQuality => "SkyQuality",
            WeatherField::SkyTemperature => "SkyTemperature",
            WeatherField::StarFwhm => "StarFWHM",
            WeatherField::Temperature => "Temperature",
            WeatherField::WindDirection => "WindDirection",
            WeatherField::WindGust => "WindGust",
            WeatherField::WindSpeed => "WindSpeed",
        }
    }

    /// The lower-case Alpaca member (`starfwhm`).
    pub fn member(self) -> &'static str {
        match self {
            WeatherField::CloudCover => "cloudcover",
            WeatherField::DewPoint => "dewpoint",
            WeatherField::Humidity => "humidity",
            WeatherField::Pressure => "pressure",
            WeatherField::RainRate => "rainrate",
            WeatherField::SkyBrightness => "skybrightness",
            WeatherField::SkyQuality => "skyquality",
            WeatherField::SkyTemperature => "skytemperature",
            WeatherField::StarFwhm => "starfwhm",
            WeatherField::Temperature => "temperature",
            WeatherField::WindDirection => "winddirection",
            WeatherField::WindGust => "windgust",
            WeatherField::WindSpeed => "windspeed",
        }
    }

    /// Match a member or sensor name, ignoring case.
    pub fn from_name(name: &str) -> Option<Self> {
        let name = name.trim();
        WeatherField::ALL
            .into_iter()
            .find(|field| field.sensor_name().eq_ignore_ascii_case(name))
    }

    pub fn get(self, reading: &WeatherReading) -> &Observed<f64> {
        match self {
            WeatherField::CloudCover => &reading.cloud_cover_pct,
            WeatherField::DewPoint => &reading.dew_point_c,
            WeatherField::Humidity => &reading.humidity_pct,
            WeatherField::Pressure => &reading.pressure_hpa,
            WeatherField::RainRate => &reading.rain_rate_mm_h,
            WeatherField::SkyBrightness => &reading.sky_brightness_lux,
            WeatherField::SkyQuality => &reading.sky_quality_mag_arcsec2,
            WeatherField::SkyTemperature => &reading.sky_temperature_c,
            WeatherField::StarFwhm => &reading.star_fwhm_arcsec,
            WeatherField::Temperature => &reading.temperature_c,
            WeatherField::WindDirection => &reading.wind_direction_deg,
            WeatherField::WindGust => &reading.wind_gust_ms,
            WeatherField::WindSpeed => &reading.wind_speed_ms,
        }
    }

    pub fn get_mut(self, reading: &mut WeatherReading) -> &mut Observed<f64> {
        match self {
            WeatherField::CloudCover => &mut reading.cloud_cover_pct,
            WeatherField::DewPoint => &mut reading.dew_point_c,
            WeatherField::Humidity => &mut reading.humidity_pct,
            WeatherField::Pressure => &mut reading.pressure_hpa,
            WeatherField::RainRate => &mut reading.rain_rate_mm_h,
            WeatherField::SkyBrightness => &mut reading.sky_brightness_lux,
            WeatherField::SkyQuality => &mut reading.sky_quality_mag_arcsec2,
            WeatherField::SkyTemperature => &mut reading.sky_temperature_c,
            WeatherField::StarFwhm => &mut reading.star_fwhm_arcsec,
            WeatherField::Temperature => &mut reading.temperature_c,
            WeatherField::WindDirection => &mut reading.wind_direction_deg,
            WeatherField::WindGust => &mut reading.wind_gust_ms,
            WeatherField::WindSpeed => &mut reading.wind_speed_ms,
        }
    }
}

/// An ObservingConditions value: a finite number, or unknown. NaN arrives as
/// JSON `null` (most serialisers) or as the string `"NaN"`; both are unknown.
pub fn parse_weather_value(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64().filter(|v| v.is_finite()),
        Value::String(text) => text.trim().parse::<f64>().ok().filter(|v| v.is_finite()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Shared Alpaca link (connect, identity, actions, devicestate)
// ---------------------------------------------------------------------------

/// One member call, timestamped at the midpoint of the request (§2.7).
pub(super) struct Timed {
    pub(super) result: DeviceResult<Value>,
    pub(super) observed_at: DateTime<Utc>,
    pub(super) mono_ns: u64,
    pub(super) latency_ms: u32,
}

impl Timed {
    /// An observation carrying this call's timing.
    pub(super) fn observed<T>(
        &self,
        value: Option<T>,
        state: ObsState,
        source: Source,
    ) -> Observed<T> {
        Observed {
            value,
            state,
            observed_at: Some(self.observed_at),
            mono_ns: Some(self.mono_ns),
            device_time: None,
            latency_ms: Some(self.latency_ms),
            source,
        }
    }

    pub(super) fn raw_member(&self, name: &str, state: ObsState) -> RawMember {
        RawMember {
            name: name.to_string(),
            value: self.result.as_ref().ok().cloned(),
            state,
            observed_at: Some(self.observed_at),
            mono_ns: Some(self.mono_ns),
            latency_ms: Some(self.latency_ms),
            error: self.result.as_ref().err().map(|e| e.to_string()),
        }
    }
}

/// A failure that says nothing about the link: the value is simply unknown,
/// or the member does not exist.
pub(super) fn is_soft_error(err: &DeviceError) -> bool {
    err.is_not_set() || err.is_unknown_member()
}

/// The values of one `devicestate` read.
#[derive(Debug, Clone, PartialEq, Default)]
pub(super) struct DeviceStateValues {
    /// Entry names in lower case, with their raw values.
    pub(super) entries: Vec<(String, Value)>,
    pub(super) timestamp: Option<DateTime<Utc>>,
}

impl DeviceStateValues {
    pub(super) fn get(&self, name: &str) -> Option<&Value> {
        self.entries
            .iter()
            .find(|(entry, _)| entry.eq_ignore_ascii_case(name))
            .map(|(_, value)| value)
    }
}

/// Parse an Alpaca `devicestate` value: an array of `{Name, Value}` objects.
/// Names are matched without regard to case; malformed entries are skipped.
pub(super) fn parse_device_state(value: &Value) -> Option<DeviceStateValues> {
    let items = value.as_array()?;
    let mut parsed = DeviceStateValues::default();
    for item in items {
        let Some(object) = item.as_object() else {
            continue;
        };
        let name = object
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case("name"))
            .and_then(|(_, v)| v.as_str());
        let Some(name) = name else {
            continue;
        };
        let value = object
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case("value"))
            .map(|(_, v)| v.clone())
            .unwrap_or(Value::Null);
        if name.eq_ignore_ascii_case("timestamp") {
            parsed.timestamp = value.as_str().and_then(parse_device_time);
        } else {
            parsed.entries.push((name.to_ascii_lowercase(), value));
        }
    }
    Some(parsed)
}

/// Parse a device timestamp permissively (§2.14): RFC 3339 with `Z` or an
/// offset, an offset without a colon, or a local time without any offset.
pub(super) fn parse_device_time(text: &str) -> Option<DateTime<Utc>> {
    let text = text.trim();
    if let Ok(at) = DateTime::parse_from_rfc3339(text) {
        return Some(at.with_timezone(&Utc));
    }
    for format in ["%Y-%m-%dT%H:%M:%S%.f%z", "%Y-%m-%d %H:%M:%S%.f%z"] {
        if let Ok(at) = DateTime::parse_from_str(text, format) {
            return Some(at.with_timezone(&Utc));
        }
    }
    for format in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f"] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(text, format) {
            return Local
                .from_local_datetime(&naive)
                .earliest()
                .map(|at| at.with_timezone(&Utc));
        }
    }
    None
}

/// The connection-level half of an Alpaca driver (§2.6 steps 3-5, C-3).
pub(super) struct AlpacaLink {
    pub(super) transport: Arc<dyn AlpacaTransport>,
    pub(super) clock: Arc<dyn Clock>,
    platform7_threshold: i32,
    pub(super) connect_poll: Duration,
    pub(super) interface_version: Option<i32>,
    pub(super) platform7: bool,
    /// Platform 7 `devicestate` answered with at least one entry.
    pub(super) device_state: bool,
    pub(super) connected: bool,
}

impl AlpacaLink {
    pub(super) fn new(
        transport: Arc<dyn AlpacaTransport>,
        clock: Arc<dyn Clock>,
        platform7_threshold: i32,
    ) -> Self {
        Self {
            transport,
            clock,
            platform7_threshold,
            connect_poll: CONNECT_POLL_INTERVAL,
            interface_version: None,
            platform7: false,
            device_state: false,
            connected: false,
        }
    }

    pub(super) fn get(&self, member: &str, query: &[(&str, String)], class: TimeoutClass) -> Timed {
        let start_utc = self.clock.now_utc();
        let start_mono = self.clock.now_mono_ns();
        let result = self.transport.get(member, query, class);
        let end_mono = self.clock.now_mono_ns().max(start_mono);
        let half = (end_mono - start_mono) / 2;
        Timed {
            result,
            observed_at: start_utc
                + chrono::Duration::nanoseconds(i64::try_from(half).unwrap_or(i64::MAX)),
            mono_ns: start_mono + half,
            latency_ms: u32::try_from((end_mono - start_mono) / 1_000_000).unwrap_or(u32::MAX),
        }
    }

    pub(super) fn put(
        &self,
        member: &str,
        form: &[(&str, String)],
        class: TimeoutClass,
    ) -> DeviceResult<Value> {
        self.transport.put(member, form, class)
    }

    pub(super) fn require_connected(&self) -> DeviceResult<()> {
        if self.connected {
            Ok(())
        } else {
            Err(DeviceError::not_connected("the device is not connected"))
        }
    }

    /// Connect and read the identity. Connecting never moves hardware
    /// (D16). Returns the report so far; the caller adds its probes.
    pub(super) fn connect(
        &mut self,
        identity: &mut DeviceIdentity,
        timeout: Duration,
    ) -> DeviceResult<ConnectReport> {
        let started = self.clock.now_mono_ns();
        let mut report = ConnectReport::default();

        // Step 3: a failed interfaceversion read counts as version 1.
        let version = match self
            .get("interfaceversion", &[], TimeoutClass::Connect)
            .result
        {
            Ok(value) => value
                .as_i64()
                .and_then(|v| i32::try_from(v).ok())
                .unwrap_or_else(|| {
                    report
                        .warnings
                        .push(format!("interfaceversion was not an integer: {value}"));
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
        identity.interface_version = Some(version);
        report.interface_version = Some(version);

        // Step 4: Platform 7 connect + connecting, else Platform 6.
        self.platform7 = version >= self.platform7_threshold;
        if self.platform7 {
            match self.put("connect", &[], TimeoutClass::Connect) {
                Ok(_) => {
                    self.wait_for_connecting(timeout)?;
                    report.path = ConnectPath::Platform7;
                }
                Err(err) if err.is_unknown_member() => {
                    report.warnings.push(format!(
                        "interface version {version} declares Platform 7 but `connect` is \
                         unknown ({err}); using Connected=true"
                    ));
                    self.platform7 = false;
                    self.connect_platform6()?;
                    report.path = ConnectPath::Platform6;
                }
                Err(err) => return Err(err),
            }
        } else {
            self.connect_platform6()?;
            report.path = ConnectPath::Platform6;
        }
        match self.get("connected", &[], TimeoutClass::Connect).result {
            Ok(Value::Bool(true)) => {}
            Ok(other) => {
                return Err(DeviceError::not_connected(format!(
                    "the device reports Connected={other} after connecting"
                )))
            }
            Err(err) => return Err(err),
        }
        self.connected = true;

        // Step 5: identity. Every member is optional; failures only warn.
        let text = |link: &Self, member: &str| match link
            .get(member, &[], TimeoutClass::Ordinary)
            .result
        {
            Ok(Value::String(text)) if !text.trim().is_empty() => Some(text.trim().to_string()),
            _ => None,
        };
        if let Some(name) = text(self, "name") {
            identity.name = name;
        }
        identity.description = text(self, "description");
        identity.driver_info = text(self, "driverinfo");
        identity.driver_version = text(self, "driverversion");

        report.duration_ms = self.clock.now_mono_ns().saturating_sub(started) / 1_000_000;
        Ok(report)
    }

    fn connect_platform6(&mut self) -> DeviceResult<()> {
        self.put(
            "connected",
            &[("Connected", "true".to_string())],
            TimeoutClass::Connect,
        )
        .map(|_| ())
    }

    /// Poll `connecting` every `connect_poll` until false, for at most
    /// `timeout` of elapsed time (§2.6), however long each GET takes. The
    /// budget is wall time from `std::time::Instant`, not the injectable
    /// clock, so a manual test clock cannot stall it; a poll count derived
    /// from the timeout is a second cap.
    fn wait_for_connecting(&self, timeout: Duration) -> DeviceResult<()> {
        let started = std::time::Instant::now();
        let step = self.connect_poll.max(Duration::from_millis(1));
        let max_polls = (timeout.as_millis() / step.as_millis()).max(1) + 1;
        for poll in 0..max_polls {
            match self.get("connecting", &[], TimeoutClass::Connect).result {
                Ok(Value::Bool(false)) => return Ok(()),
                Ok(Value::Bool(true)) => {}
                Ok(other) => {
                    return Err(DeviceError::driver(format!(
                        "connecting returned a non-boolean value: {other}"
                    )))
                }
                Err(err) => return Err(err),
            }
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() || poll + 1 >= max_polls {
                break;
            }
            std::thread::sleep(self.connect_poll.min(remaining));
        }
        Err(DeviceError::timeout(format!(
            "the device was still connecting after {} s",
            timeout.as_secs()
        )))
    }

    /// Probe `devicestate` (Platform 7 only). Unknown members are recorded
    /// as unsupported; the probe never fails the connect.
    pub(super) fn probe_device_state(
        &mut self,
        report: &mut ConnectReport,
    ) -> Option<DeviceStateValues> {
        self.device_state = false;
        if !self.platform7 {
            return None;
        }
        match self.get("devicestate", &[], TimeoutClass::Ordinary).result {
            Ok(value) => match parse_device_state(&value) {
                Some(values) if !values.entries.is_empty() => {
                    self.device_state = true;
                    Some(values)
                }
                _ => None,
            },
            Err(err) if err.is_unknown_member() => {
                report.unsupported_members.push("devicestate".into());
                None
            }
            Err(err) => {
                report
                    .warnings
                    .push(format!("devicestate probe failed: {err}"));
                None
            }
        }
    }

    pub(super) fn disconnect(&mut self) -> DeviceResult<()> {
        let result = if self.platform7 {
            self.put("disconnect", &[], TimeoutClass::Connect)
        } else {
            self.put(
                "connected",
                &[("Connected", "false".to_string())],
                TimeoutClass::Connect,
            )
        };
        self.connected = false;
        result.map(|_| ())
    }

    pub(super) fn is_connected(&mut self) -> DeviceResult<bool> {
        match self.get("connected", &[], TimeoutClass::Connect).result? {
            Value::Bool(connected) => {
                if !connected {
                    self.connected = false;
                }
                Ok(connected)
            }
            other => Err(DeviceError::driver(format!(
                "connected returned a non-boolean value: {other}"
            ))),
        }
    }

    pub(super) fn supported_actions(&self) -> DeviceResult<Vec<String>> {
        match self
            .get("supportedactions", &[], TimeoutClass::Ordinary)
            .result?
        {
            Value::Array(items) => Ok(items
                .into_iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect()),
            Value::Null => Ok(Vec::new()),
            other => Err(DeviceError::driver(format!(
                "supportedactions returned a non-array value: {other}"
            ))),
        }
    }

    pub(super) fn action(&self, name: &str, params: &str) -> DeviceResult<String> {
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
}

// ---------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------

/// Alpaca ObservingConditions device (R18).
pub struct AlpacaObservingConditions {
    identity: DeviceIdentity,
    link: AlpacaLink,
    /// Members that answered NotImplemented (or HTTP 400/404), per session.
    unsupported: BTreeSet<WeatherField>,
    /// Sensors whose `timesincelastupdate` is not implemented.
    ages_unsupported: BTreeSet<WeatherField>,
    reading: WeatherReading,
    /// Each sensor's last update on the host clock (`now - age`).
    last_update: BTreeMap<WeatherField, DateTime<Utc>>,
    ages_refreshed_mono: Option<u64>,
}

impl AlpacaObservingConditions {
    /// `identity.endpoint` is the device's `AlpacaHttp` endpoint; the spine
    /// builds the identity key with `identity_key::alpaca` (§2.6).
    pub fn new(
        identity: DeviceIdentity,
        transport: Arc<dyn AlpacaTransport>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            identity,
            link: AlpacaLink::new(transport, clock, OBSERVING_CONDITIONS_PLATFORM7),
            unsupported: BTreeSet::new(),
            ages_unsupported: BTreeSet::new(),
            reading: WeatherReading::default(),
            last_update: BTreeMap::new(),
            ages_refreshed_mono: None,
        }
    }

    /// Wait between Platform 7 `connecting` reads (default 250 ms).
    pub fn with_connect_poll(mut self, interval: Duration) -> Self {
        self.link.connect_poll = interval;
        self
    }

    /// Fields memoised as unsupported this session.
    pub fn unsupported_fields(&self) -> Vec<WeatherField> {
        self.unsupported.iter().copied().collect()
    }

    pub fn uses_device_state(&self) -> bool {
        self.link.device_state
    }

    fn supported_fields(&self) -> Vec<WeatherField> {
        WeatherField::ALL
            .into_iter()
            .filter(|field| !self.unsupported.contains(field))
            .collect()
    }

    /// Apply one field result. Returns `Some(error)` for a hard failure.
    fn apply(
        &mut self,
        field: WeatherField,
        timed: &Timed,
        source: Source,
        members: &mut Vec<RawMember>,
    ) -> Option<DeviceError> {
        let (observed, state, hard) = match &timed.result {
            Ok(value) => match parse_weather_value(value) {
                Some(number) => (
                    timed.observed(Some(number), ObsState::Fresh, source),
                    ObsState::Fresh,
                    None,
                ),
                None => (
                    timed.observed(None, ObsState::NotSet, source),
                    ObsState::NotSet,
                    None,
                ),
            },
            Err(err) if err.is_not_set() => (
                timed.observed(None, ObsState::NotSet, source),
                ObsState::NotSet,
                None,
            ),
            Err(err) if err.is_unknown_member() => {
                self.unsupported.insert(field);
                (Observed::unsupported(source), ObsState::Unsupported, None)
            }
            Err(err) => (
                field.get(&self.reading).clone().into_stale(),
                ObsState::Error,
                Some(err.clone()),
            ),
        };
        *field.get_mut(&mut self.reading) = observed;
        members.push(timed.raw_member(field.member(), state));
        hard
    }

    fn refresh_sensor_ages(&mut self, members: &mut Vec<RawMember>) {
        for field in self.supported_fields() {
            if self.ages_unsupported.contains(&field) {
                continue;
            }
            let timed = self.link.get(
                SENSOR_AGES_MEMBER,
                &[("SensorName", field.sensor_name().to_string())],
                TimeoutClass::Ordinary,
            );
            let state = match &timed.result {
                Ok(value) => match parse_weather_value(value).filter(|age| *age >= 0.0) {
                    Some(age) => {
                        let age = chrono::Duration::milliseconds((age * 1000.0).round() as i64);
                        self.last_update.insert(field, timed.observed_at - age);
                        ObsState::Fresh
                    }
                    None => {
                        self.last_update.remove(&field);
                        ObsState::NotSet
                    }
                },
                Err(err) if err.is_unknown_member() => {
                    self.ages_unsupported.insert(field);
                    self.last_update.remove(&field);
                    ObsState::Unsupported
                }
                Err(err) if err.is_not_set() => {
                    self.last_update.remove(&field);
                    ObsState::NotSet
                }
                Err(_) => ObsState::Error,
            };
            members.push(timed.raw_member(
                &format!("{SENSOR_AGES_MEMBER}:{}", field.sensor_name()),
                state,
            ));
        }
        self.ages_refreshed_mono = Some(self.link.clock.now_mono_ns());
    }

    fn ages_due(&self, plan: &PollPlan) -> bool {
        if plan
            .extra_members
            .iter()
            .any(|m| m.eq_ignore_ascii_case(SENSOR_AGES_MEMBER))
        {
            return true;
        }
        match self.ages_refreshed_mono {
            None => true,
            Some(at) => {
                self.link.clock.now_mono_ns().saturating_sub(at)
                    >= crate::equipment::clock::duration_ns(SENSOR_AGE_INTERVAL)
            }
        }
    }

    /// Stamp each field with its sensor's last update and publish the ages.
    fn apply_sensor_ages(&mut self) {
        let now = self.link.clock.now_utc();
        self.reading.sensor_ages_s.clear();
        for field in WeatherField::ALL {
            let last = if self.unsupported.contains(&field) {
                None
            } else {
                self.last_update.get(&field).copied()
            };
            let observed = field.get_mut(&mut self.reading);
            observed.device_time = last;
            if let Some(last) = last {
                let age_s = (now - last).num_milliseconds().max(0) as f64 / 1000.0;
                self.reading
                    .sensor_ages_s
                    .insert(field.sensor_name().to_string(), age_s);
            }
        }
    }

    fn mark_all_stale(&mut self) {
        for field in WeatherField::ALL {
            let observed = field.get(&self.reading).clone().into_stale();
            *field.get_mut(&mut self.reading) = observed;
        }
    }
}

impl Device for AlpacaObservingConditions {
    fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    fn connect(&mut self, timeout: Duration) -> DeviceResult<ConnectReport> {
        let mut report = self.link.connect(&mut self.identity, timeout)?;
        self.unsupported.clear();
        self.ages_unsupported.clear();
        self.last_update.clear();
        self.ages_refreshed_mono = None;
        self.reading = WeatherReading::default();

        // Step 6: which of the 13 members exist. A value, NaN or ValueNotSet
        // means the member exists; NotImplemented or HTTP 400/404 means it
        // does not (memoised, not a failure).
        let mut members = Vec::new();
        for field in WeatherField::ALL {
            let timed = self.link.get(field.member(), &[], TimeoutClass::Ordinary);
            if let Some(err) = self.apply(field, &timed, Source::Property, &mut members) {
                report
                    .warnings
                    .push(format!("{} probe failed: {err}", field.member()));
            }
        }
        for field in &self.unsupported {
            report.unsupported_members.push(field.member().to_string());
        }
        self.link.probe_device_state(&mut report);
        report.device_state = self.link.device_state;

        // `sensordescription` is captured once.
        for field in self.supported_fields() {
            let timed = self.link.get(
                "sensordescription",
                &[("SensorName", field.sensor_name().to_string())],
                TimeoutClass::Ordinary,
            );
            if let Ok(Value::String(text)) = timed.result {
                if !text.trim().is_empty() {
                    self.reading
                        .sensor_descriptions
                        .insert(field.sensor_name().to_string(), text.trim().to_string());
                }
            }
        }
        Ok(report)
    }

    fn disconnect(&mut self) -> DeviceResult<()> {
        self.link.disconnect()
    }

    fn is_connected(&mut self) -> DeviceResult<bool> {
        self.link.is_connected()
    }

    fn poll(&mut self, plan: &PollPlan) -> DeviceResult<RawPoll> {
        self.link.require_connected()?;
        let started_mono_ns = self.link.clock.now_mono_ns();
        let mut members = Vec::new();
        let mut attempted = 0usize;
        let mut hard_errors = Vec::new();
        let mut device_time = None;
        let mut method = PollMethod::PerMember;
        let mut pending: Vec<WeatherField> = self.supported_fields();

        if self.link.device_state && plan.prefer_device_state {
            let timed = self.link.get("devicestate", &[], TimeoutClass::Ordinary);
            match &timed.result {
                Ok(value) => {
                    if let Some(values) = parse_device_state(value) {
                        method = PollMethod::DeviceState;
                        device_time = values.timestamp;
                        let mut remaining = Vec::new();
                        for field in pending {
                            match values.get(field.sensor_name()) {
                                Some(value) => {
                                    let entry = Timed {
                                        result: Ok(value.clone()),
                                        observed_at: timed.observed_at,
                                        mono_ns: timed.mono_ns,
                                        latency_ms: timed.latency_ms,
                                    };
                                    self.apply(field, &entry, Source::DeviceState, &mut members);
                                }
                                // Missing from DeviceState: read it directly.
                                None => remaining.push(field),
                            }
                        }
                        pending = remaining;
                    }
                }
                Err(err) if err.is_unknown_member() => {
                    self.link.device_state = false;
                }
                Err(err) => {
                    let err = err.clone();
                    self.mark_all_stale();
                    self.apply_sensor_ages();
                    return Err(err);
                }
            }
        }

        for field in pending {
            attempted += 1;
            let timed = self.link.get(field.member(), &[], TimeoutClass::Ordinary);
            if let Some(err) = self.apply(field, &timed, Source::Property, &mut members) {
                hard_errors.push(err);
            }
        }
        // Every attempted read failed on the link itself: the poll failed.
        if attempted > 0 && hard_errors.len() == attempted {
            self.apply_sensor_ages();
            return Err(hard_errors.swap_remove(0));
        }

        if self.ages_due(plan) {
            self.refresh_sensor_ages(&mut members);
        }
        self.apply_sensor_ages();
        Ok(RawPoll {
            method,
            started_mono_ns,
            finished_mono_ns: self.link.clock.now_mono_ns(),
            device_time,
            members,
        })
    }

    fn supported_actions(&mut self) -> DeviceResult<Vec<String>> {
        self.link.supported_actions()
    }

    fn action(&mut self, name: &str, params: &str) -> DeviceResult<String> {
        self.link.action(name, params)
    }

    /// Weather devices never move: no abort path.
    fn abort_path(&self) -> Option<Arc<dyn AbortPath>> {
        None
    }
}

impl WeatherDriver for AlpacaObservingConditions {
    /// The reading assembled by the latest poll; no I/O.
    fn read(&mut self) -> DeviceResult<WeatherReading> {
        self.link.require_connected()?;
        Ok(self.reading.clone())
    }

    /// PUT `refresh`. Optional in ASCOM; NotImplemented maps to Unsupported.
    fn refresh(&mut self) -> DeviceResult<()> {
        self.link.require_connected()?;
        match self.link.put("refresh", &[], TimeoutClass::Ordinary) {
            Ok(_) => Ok(()),
            Err(mut err) if err.is_unknown_member() => {
                err.class = ErrorClass::Unsupported;
                err.message = format!("refresh is not implemented: {}", err.message);
                Err(err)
            }
            Err(err) => Err(err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::testing::fake_transport::{FakeAlpacaTransport, FakeReply, Method};
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::types::{identity_key, DeviceKind, DriverId, Endpoint};
    use serde_json::json;

    fn identity() -> DeviceIdentity {
        DeviceIdentity::new(
            DeviceKind::Weather,
            DriverId::Alpaca,
            Endpoint::AlpacaHttp {
                base: "http://127.0.0.1:11111".into(),
                device_type: "observingconditions".into(),
                number: 0,
            },
            identity_key::alpaca(None, "127.0.0.1", 11111, "observingconditions", 0),
            "ObservingConditions 0",
        )
    }

    /// A Platform 6 (IV1) device with temperature, humidity and dew point.
    fn platform6_fake() -> Arc<FakeAlpacaTransport> {
        let fake = FakeAlpacaTransport::new().shared();
        fake.on_get("interfaceversion", FakeReply::value(1))
            .on_put("connected", FakeReply::ok())
            .on_get("connected", FakeReply::value(true))
            .on_get("name", FakeReply::value("Backyard Station"))
            .on_get("driverversion", FakeReply::value("1.2"))
            .on_get("temperature", FakeReply::value(4.25))
            .on_get("humidity", FakeReply::value(81.5))
            .on_get("dewpoint", FakeReply::value(1.25));
        for field in WeatherField::ALL {
            if !matches!(
                field,
                WeatherField::Temperature | WeatherField::Humidity | WeatherField::DewPoint
            ) {
                fake.on_get(field.member(), FakeReply::not_implemented());
            }
        }
        fake
    }

    fn driver(
        fake: &Arc<FakeAlpacaTransport>,
        clock: &Arc<ManualClock>,
    ) -> AlpacaObservingConditions {
        AlpacaObservingConditions::new(
            identity(),
            Arc::clone(fake) as Arc<dyn AlpacaTransport>,
            Arc::clone(clock) as Arc<dyn Clock>,
        )
        .with_connect_poll(Duration::from_millis(1))
    }

    #[test]
    fn platform6_connect_reads_identity_and_memoises_unsupported_members() {
        let fake = platform6_fake();
        let clock = Arc::new(ManualClock::new());
        let mut oc = driver(&fake, &clock);
        let report = oc.connect(Duration::from_secs(5)).unwrap();
        assert_eq!(report.path, ConnectPath::Platform6);
        assert_eq!(report.interface_version, Some(1));
        assert!(!report.device_state);
        assert_eq!(oc.identity().name, "Backyard Station");
        assert_eq!(oc.identity().driver_version.as_deref(), Some("1.2"));
        assert_eq!(report.unsupported_members.len(), 10);
        assert!(report
            .unsupported_members
            .contains(&"cloudcover".to_string()));
        assert_eq!(
            fake.calls_to(Method::Put, "connected"),
            1,
            "Platform 6 uses Connected=true"
        );
        assert_eq!(fake.calls_to(Method::Put, "connect"), 0);

        // An unsupported member is never read again.
        fake.clear_calls();
        oc.poll(&PollPlan::default()).unwrap();
        assert_eq!(fake.calls_to(Method::Get, "cloudcover"), 0);
        assert_eq!(fake.calls_to(Method::Get, "temperature"), 1);
        let reading = oc.read().unwrap();
        assert_eq!(reading.temperature_c.value, Some(4.25));
        assert_eq!(reading.temperature_c.state, ObsState::Fresh);
        assert_eq!(reading.temperature_c.source, Source::Property);
        assert_eq!(reading.cloud_cover_pct.state, ObsState::Unsupported);
        assert_eq!(reading.cloud_cover_pct.value, None);
    }

    #[test]
    fn nan_value_not_set_and_not_implemented_become_unknown() {
        let fake = platform6_fake();
        // NaN serialises as JSON null in serde_json, and some servers send
        // the string "NaN"; ValueNotSet is ASCOM 0x402.
        fake.on_get("temperature", FakeReply::value(f64::NAN))
            .on_get("humidity", FakeReply::value("NaN"))
            .on_get("dewpoint", FakeReply::value_not_set());
        let clock = Arc::new(ManualClock::new());
        let mut oc = driver(&fake, &clock);
        oc.connect(Duration::from_secs(5)).unwrap();
        let raw = oc.poll(&PollPlan::default()).unwrap();
        assert!(!raw.any_failed(), "unknown values are not failures");
        let reading = oc.read().unwrap();
        for (observed, name) in [
            (&reading.temperature_c, "NaN as null"),
            (&reading.humidity_pct, "NaN as string"),
            (&reading.dew_point_c, "ValueNotSet"),
        ] {
            assert_eq!(observed.value, None, "{name}");
            assert_eq!(observed.state, ObsState::NotSet, "{name}");
        }
        assert_eq!(reading.wind_speed_ms.value, None);
        assert_eq!(reading.wind_speed_ms.state, ObsState::Unsupported);
        assert_eq!(parse_weather_value(&json!("12.5")), Some(12.5));
        assert_eq!(parse_weather_value(&json!("Infinity")), None);
        assert_eq!(parse_weather_value(&json!(true)), None);
    }

    #[test]
    fn http_400_from_an_older_server_is_unsupported_not_a_failure() {
        let fake = platform6_fake();
        fake.on_get(
            "skyquality",
            FakeReply::http(400, "Unknown member 'skyquality'"),
        );
        fake.on_get("pressure", FakeReply::http(404, ""));
        let clock = Arc::new(ManualClock::new());
        let mut oc = driver(&fake, &clock);
        let report = oc.connect(Duration::from_secs(5)).unwrap();
        assert!(report.unsupported_members.contains(&"skyquality".into()));
        assert!(report.unsupported_members.contains(&"pressure".into()));
        assert!(oc.unsupported_fields().contains(&WeatherField::SkyQuality));
    }

    #[test]
    fn sensor_ages_set_device_time_and_refresh_every_minute() {
        let fake = platform6_fake();
        fake.queue_get(SENSOR_AGES_MEMBER, [FakeReply::value(12.0)]);
        fake.on_get(SENSOR_AGES_MEMBER, FakeReply::value(3.0));
        let clock = Arc::new(ManualClock::new());
        let mut oc = driver(&fake, &clock);
        oc.connect(Duration::from_secs(5)).unwrap();
        let t0 = clock.now_utc();
        oc.poll(&PollPlan::default()).unwrap();
        // One age read per supported sensor on the first poll, with the
        // exact-case SensorName parameter.
        assert_eq!(fake.calls_to(Method::Get, SENSOR_AGES_MEMBER), 3);
        let first = fake
            .calls()
            .into_iter()
            .find(|call| call.member == SENSOR_AGES_MEMBER)
            .unwrap();
        assert_eq!(first.params, vec![("SensorName".into(), "DewPoint".into())]);
        let reading = oc.read().unwrap();
        assert_eq!(reading.sensor_ages_s["DewPoint"], 12.0);
        assert_eq!(reading.sensor_ages_s["Temperature"], 3.0);
        assert_eq!(
            reading.temperature_c.device_time,
            Some(t0 - chrono::Duration::seconds(3))
        );
        assert!(!reading.sensor_ages_s.contains_key("CloudCover"));

        // 10 s later: no age reads, but the published age grows.
        clock.advance(Duration::from_secs(10));
        fake.clear_calls();
        oc.poll(&PollPlan::default()).unwrap();
        assert_eq!(fake.calls_to(Method::Get, SENSOR_AGES_MEMBER), 0);
        let reading = oc.read().unwrap();
        assert_eq!(reading.sensor_ages_s["Temperature"], 13.0);

        // After 60 s the ages are read again.
        clock.advance(SENSOR_AGE_INTERVAL);
        oc.poll(&PollPlan::default()).unwrap();
        assert_eq!(fake.calls_to(Method::Get, SENSOR_AGES_MEMBER), 3);
        assert_eq!(oc.read().unwrap().sensor_ages_s["DewPoint"], 3.0);

        // A plan can force a refresh.
        fake.clear_calls();
        let plan = PollPlan {
            extra_members: vec![SENSOR_AGES_MEMBER.into()],
            ..PollPlan::default()
        };
        oc.poll(&plan).unwrap();
        assert_eq!(fake.calls_to(Method::Get, SENSOR_AGES_MEMBER), 3);
    }

    #[test]
    fn a_failed_read_keeps_the_last_value_as_stale_and_a_dead_link_fails_the_poll() {
        let fake = platform6_fake();
        let clock = Arc::new(ManualClock::new());
        let mut oc = driver(&fake, &clock);
        oc.connect(Duration::from_secs(5)).unwrap();
        oc.poll(&PollPlan::default()).unwrap();

        // One member times out: the poll succeeds but marks it degraded.
        fake.queue_get("temperature", [FakeReply::Timeout]);
        clock.advance(Duration::from_secs(10));
        let raw = oc.poll(&PollPlan::default()).unwrap();
        assert!(raw.any_failed());
        let reading = oc.read().unwrap();
        assert_eq!(reading.temperature_c.state, ObsState::Stale);
        assert_eq!(reading.temperature_c.value, Some(4.25), "last good value");
        assert_eq!(reading.humidity_pct.state, ObsState::Fresh);

        // Every read fails on the link: the poll fails.
        for field in [
            WeatherField::Temperature,
            WeatherField::Humidity,
            WeatherField::DewPoint,
        ] {
            fake.queue_get(field.member(), [FakeReply::Transport("reset".into())]);
        }
        let err = oc.poll(&PollPlan::default()).unwrap_err();
        assert_eq!(err.class, ErrorClass::Transport);
        assert_eq!(oc.read().unwrap().humidity_pct.state, ObsState::Stale);
    }

    #[test]
    fn platform7_connects_with_connecting_and_polls_device_state() {
        let fake = FakeAlpacaTransport::new().shared();
        fake.on_get("interfaceversion", FakeReply::value(2))
            .on_put("connect", FakeReply::ok())
            .queue_get("connecting", [FakeReply::value(true)])
            .on_get("connecting", FakeReply::value(false))
            .on_get("connected", FakeReply::value(true));
        for field in WeatherField::ALL {
            fake.on_get(field.member(), FakeReply::value(1.0));
        }
        fake.on_get(
            "devicestate",
            FakeReply::value(json!([
                {"Name": "CloudCover", "Value": 12.5},
                {"Name": "temperature", "Value": 3.5},
                {"Name": "Humidity", "Value": null},
                {"Name": "TimeStamp", "Value": "2026-01-01T00:00:00.5Z"},
            ])),
        );
        let clock = Arc::new(ManualClock::new());
        let mut oc = driver(&fake, &clock);
        let report = oc.connect(Duration::from_secs(5)).unwrap();
        assert_eq!(report.path, ConnectPath::Platform7);
        assert!(report.device_state && oc.uses_device_state());
        assert_eq!(fake.calls_to(Method::Get, "connecting"), 2);
        assert_eq!(fake.calls_to(Method::Put, "connected"), 0);

        fake.clear_calls();
        let plan = PollPlan {
            prefer_device_state: true,
            ..PollPlan::default()
        };
        let raw = oc.poll(&plan).unwrap();
        assert_eq!(raw.method, PollMethod::DeviceState);
        assert_eq!(
            raw.device_time,
            Some(
                DateTime::parse_from_rfc3339("2026-01-01T00:00:00.5Z")
                    .unwrap()
                    .into()
            )
        );
        let reading = oc.read().unwrap();
        assert_eq!(reading.cloud_cover_pct.value, Some(12.5));
        assert_eq!(reading.cloud_cover_pct.source, Source::DeviceState);
        assert_eq!(reading.temperature_c.value, Some(3.5));
        assert_eq!(reading.humidity_pct.value, None, "null in DeviceState");
        assert_eq!(reading.humidity_pct.state, ObsState::NotSet);
        // Fields missing from DeviceState are read per member.
        assert_eq!(fake.calls_to(Method::Get, "windspeed"), 1);
        assert_eq!(fake.calls_to(Method::Get, "cloudcover"), 0);
        assert_eq!(reading.wind_speed_ms.source, Source::Property);
    }

    #[test]
    fn platform7_connect_times_out_and_refresh_maps_not_implemented() {
        let fake = FakeAlpacaTransport::new().shared();
        fake.on_get("interfaceversion", FakeReply::value(2))
            .on_put("connect", FakeReply::ok())
            .on_get("connecting", FakeReply::value(true));
        let clock = Arc::new(ManualClock::new());
        let mut oc = driver(&fake, &clock);
        let err = oc.connect(Duration::from_millis(5)).unwrap_err();
        assert_eq!(err.class, ErrorClass::Timeout);
        assert!(oc.poll(&PollPlan::default()).is_err(), "not connected");

        // Slow `connecting` replies: the wait is bounded by elapsed time,
        // not by a poll count (61 polls of 25 ms here would take 1.5 s).
        let slow = FakeAlpacaTransport::new().shared();
        slow.on_get("interfaceversion", FakeReply::value(2))
            .on_put("connect", FakeReply::ok())
            .on_get(
                "connecting",
                FakeReply::Delay(Duration::from_millis(25), Box::new(FakeReply::value(true))),
            );
        let mut oc = driver(&slow, &clock);
        let err = oc.connect(Duration::from_millis(60)).unwrap_err();
        assert_eq!(err.class, ErrorClass::Timeout);
        let polls = slow.calls_to(Method::Get, "connecting");
        assert!((1..=3).contains(&polls), "{polls} connecting polls");

        let fake = platform6_fake();
        fake.on_put("refresh", FakeReply::not_implemented());
        let mut oc = driver(&fake, &clock);
        oc.connect(Duration::from_secs(5)).unwrap();
        assert_eq!(oc.refresh().unwrap_err().class, ErrorClass::Unsupported);
        fake.on_put("refresh", FakeReply::ok());
        assert!(oc.refresh().is_ok());
        assert!(oc.abort_path().is_none());
    }

    #[test]
    fn failed_interfaceversion_counts_as_one_and_connected_false_refuses() {
        let fake = platform6_fake();
        fake.on_get("interfaceversion", FakeReply::Timeout);
        let clock = Arc::new(ManualClock::new());
        let mut oc = driver(&fake, &clock);
        let report = oc.connect(Duration::from_secs(5)).unwrap();
        assert_eq!(report.interface_version, Some(1));
        assert!(!report.warnings.is_empty());

        fake.on_get("connected", FakeReply::value(false));
        let mut oc = driver(&fake, &clock);
        assert_eq!(
            oc.connect(Duration::from_secs(5)).unwrap_err().class,
            ErrorClass::NotConnected
        );
    }

    #[test]
    fn device_time_parsing_is_permissive() {
        let utc = parse_device_time("2026-03-01T12:00:00Z").unwrap();
        assert_eq!(parse_device_time("2026-03-01T14:00:00+02:00").unwrap(), utc);
        assert_eq!(parse_device_time("2026-03-01T14:00:00+0200").unwrap(), utc);
        let naive = NaiveDateTime::parse_from_str("2026-03-01T12:00:00.25", "%Y-%m-%dT%H:%M:%S%.f")
            .unwrap();
        let local = Local
            .from_local_datetime(&naive)
            .earliest()
            .unwrap()
            .with_timezone(&Utc);
        assert_eq!(parse_device_time("2026-03-01T12:00:00.25"), Some(local));
        assert_eq!(parse_device_time("yesterday"), None);
        assert_eq!(
            WeatherField::from_name("starfwhm"),
            Some(WeatherField::StarFwhm)
        );
        assert_eq!(WeatherField::from_name("Rain"), None);
    }
}

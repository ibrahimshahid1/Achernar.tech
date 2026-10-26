//! Alpaca `ITelescopeV4` driver implementing `MountDriver` (§3.2, §7.1
//! R1-R11). Owner: WP1A-4.
//!
//! Independent implementation from the ASCOM Alpaca API reference and the
//! ITelescopeV4 master interface (MIT); no code is taken from the ASCOM
//! Library or NINA.
//!
//! - **Connect** (§2.6): `interfaceversion` (a failed read counts as 1);
//!   from InterfaceVersion 4 (Platform 7) PUT `connect` then poll
//!   `connecting`; otherwise PUT `Connected=true` and GET `connected`. An
//!   older server that answers HTTP 400/404 or NotImplemented to `connect`
//!   falls back to the Platform 6 path. Identity, every Can* flag,
//!   AlignmentMode, EquatorialSystem (InterfaceVersion >= 2), DoesRefraction,
//!   TrackingRates, AxisRates, FocalLength and ApertureDiameter (M-m) are
//!   probed once; an unknown member is memoised as unsupported for the
//!   session, never treated as a driver failure.
//! - **Poll** (§2.7): one `devicestate` GET when the device offers a
//!   non-empty DeviceState, else the twelve per-member reads. Members
//!   missing from DeviceState are read individually. TrackingRate (10 s),
//!   guide rates (30 s, read whenever implemented, M-i) and the site (60 s)
//!   are read on their own schedule or when the plan names them;
//!   TargetRightAscension/Declination are read while an operation runs. A
//!   failed DeviceState marks the tick's members stale instead of retrying
//!   them one by one, so a slow device is not hammered. This driver always
//!   prefers DeviceState when it is available.
//! - **Abort** (D4): [`AlpacaMountAbort`] runs on its own transport (the
//!   client's second agent): AbortSlew, plus MoveAxis 0 on every axis this
//!   driver set moving. It is never gated on a Can* flag (O10, D19).
//! - **Units**: RA hours, Dec/Alt/Az degrees, guide rates deg/s on the wire
//!   and arcsec/s in [`MountState`], focal length and aperture metres on the
//!   wire and millimetres in [`MountCapabilities`].

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Local, NaiveDateTime, TimeZone, Utc};
use serde_json::Value;

use crate::equipment::mount::jog::JogFlags;
use crate::equipment::mount::{native_frame_policy, MountDriverOptions};
use crate::equipment::snapshot::{AxisRates, MountCapabilities, MountState, RateRange};
use crate::equipment::traits::{
    AbortPath, AlpacaTransport, Clock, ConnectPath, ConnectReport, Device, MountDriver, PollMethod,
    PollPlan, RawMember, RawPoll, TimeoutClass,
};
use crate::equipment::types::{
    AlignmentMode, AltAz, DeviceError, DeviceIdentity, DeviceResult, DriveRate, Endpoint, EqCoord,
    ErrorClass, GuideDirection, ObsState, Observed, PierSide, Site, SiteSource, Source,
};

/// Telescope InterfaceVersion at which Platform 7 connect and DeviceState
/// apply (§2.6 step 4).
pub const PLATFORM7_INTERFACE: i32 = 4;

/// Members read every tick: (member, DeviceState name).
const CORE: &[(&str, &str)] = &[
    ("rightascension", "RightAscension"),
    ("declination", "Declination"),
    ("altitude", "Altitude"),
    ("azimuth", "Azimuth"),
    ("siderealtime", "SiderealTime"),
    ("utcdate", "UTCDate"),
    ("slewing", "Slewing"),
    ("tracking", "Tracking"),
    ("atpark", "AtPark"),
    ("athome", "AtHome"),
    ("sideofpier", "SideOfPier"),
    ("ispulseguiding", "IsPulseGuiding"),
];

/// Slow members and their period in seconds (§2.7 "extra members").
const SLOW: &[(&str, u64)] = &[
    ("trackingrate", 10),
    ("guideraterightascension", 30),
    ("guideratedeclination", 30),
    ("sitelatitude", 60),
    ("sitelongitude", 60),
    ("siteelevation", 60),
];

const TARGET: &[&str] = &["targetrightascension", "targetdeclination"];

/// Every Can* flag probed at connect.
const CAN_FLAGS: &[&str] = &[
    "canpark",
    "canunpark",
    "cansetpark",
    "canfindhome",
    "canslew",
    "canslewasync",
    "canslewaltaz",
    "canslewaltazasync",
    "cansync",
    "cansyncaltaz",
    "cansettracking",
    "cansetpierside",
    "canpulseguide",
    "cansetguiderates",
    "cansetdeclinationrate",
    "cansetrightascensionrate",
];

fn number(value: f64) -> String {
    // Rust's `Display` never uses exponents or locale separators, which is
    // the invariant culture C-3 asks for.
    format!("{value}")
}

fn boolean(value: bool) -> String {
    if value { "True" } else { "False" }.to_string()
}

fn as_f64(value: &Value) -> Option<f64> {
    value.as_f64().filter(|v| v.is_finite())
}

fn as_i64(value: &Value) -> Option<i64> {
    value.as_i64().or_else(|| {
        value
            .as_f64()
            .filter(|v| v.fract() == 0.0)
            .map(|v| v as i64)
    })
}

fn as_bool(value: &Value) -> Option<bool> {
    value.as_bool().or_else(|| match value.as_str() {
        Some(s) if s.eq_ignore_ascii_case("true") => Some(true),
        Some(s) if s.eq_ignore_ascii_case("false") => Some(false),
        _ => None,
    })
}

fn get_ci<'a>(object: &'a serde_json::Map<String, Value>, key: &str) -> Option<&'a Value> {
    object
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(key))
        .map(|(_, v)| v)
}

/// Parse an Alpaca date-time permissively (§2.14): RFC 3339 with `Z` or an
/// explicit offset, or no zone at all, which is local time when
/// `naive_is_local` (DeviceState `TimeStamp`) and UTC otherwise (`UTCDate`).
pub fn parse_alpaca_datetime(text: &str, naive_is_local: bool) -> Option<DateTime<Utc>> {
    let text = text.trim();
    if let Ok(parsed) = DateTime::parse_from_rfc3339(text) {
        return Some(parsed.with_timezone(&Utc));
    }
    for format in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f"] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(text, format) {
            return if naive_is_local {
                Local
                    .from_local_datetime(&naive)
                    .earliest()
                    .map(|local| local.with_timezone(&Utc))
            } else {
                Some(Utc.from_utc_datetime(&naive))
            };
        }
    }
    None
}

/// DeviceState `[{Name, Value}, ...]` keyed by lower-case name.
pub fn parse_device_state(value: &Value) -> HashMap<String, Value> {
    let mut entries = HashMap::new();
    for item in value.as_array().into_iter().flatten() {
        let Some(object) = item.as_object() else {
            continue;
        };
        if let (Some(name), Some(value)) = (
            get_ci(object, "Name").and_then(Value::as_str),
            get_ci(object, "Value"),
        ) {
            entries.insert(name.to_ascii_lowercase(), value.clone());
        }
    }
    entries
}

/// One cached member read.
#[derive(Debug, Clone, Default)]
struct Slot {
    value: Option<Value>,
    state: ObsState,
    observed_at: Option<DateTime<Utc>>,
    mono_ns: Option<u64>,
    latency_ms: Option<u32>,
    device_time: Option<DateTime<Utc>>,
    source: Source,
}

#[derive(Debug, Clone, Copy)]
struct Timing {
    utc: DateTime<Utc>,
    mono_ns: u64,
    latency_ms: u32,
}

/// The out-of-band stop (D4, D19): AbortSlew, then MoveAxis 0 on every axis
/// this driver set moving. "Parked, nothing to abort" counts as success.
pub struct AlpacaMountAbort {
    transport: Arc<dyn AlpacaTransport>,
    jog: Arc<JogFlags>,
}

impl AbortPath for AlpacaMountAbort {
    fn abort(&self) -> DeviceResult<()> {
        let mut first_error = match self.transport.put("abortslew", &[], TimeoutClass::Long) {
            Ok(_) => None,
            Err(err) if err.class == ErrorClass::InvalidWhileParked => None,
            Err(err) => Some(err),
        };
        for axis in self.jog.moving_axes() {
            let form = [("Axis", axis.to_string()), ("Rate", "0".to_string())];
            match self.transport.put("moveaxis", &form, TimeoutClass::Long) {
                Ok(_) => self.jog.set(axis, 0.0),
                Err(err) => {
                    first_error.get_or_insert(err);
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

/// The Alpaca telescope driver.
pub struct AlpacaTelescope {
    identity: DeviceIdentity,
    transport: Arc<dyn AlpacaTransport>,
    clock: Arc<dyn Clock>,
    options: MountDriverOptions,
    abort: Arc<AlpacaMountAbort>,
    jog: Arc<JogFlags>,
    connected: bool,
    path: ConnectPath,
    interface_version: Option<i32>,
    device_state: bool,
    caps: MountCapabilities,
    unsupported: BTreeSet<String>,
    slots: HashMap<&'static str, Slot>,
    next_due: HashMap<&'static str, u64>,
    connect_poll_interval: Duration,
}

impl AlpacaTelescope {
    /// `transport` serves ordinary calls; `abort_transport` is the client's
    /// second agent (5 s timeout), used only by the abort path.
    pub fn new(
        identity: DeviceIdentity,
        transport: Arc<dyn AlpacaTransport>,
        abort_transport: Arc<dyn AlpacaTransport>,
        clock: Arc<dyn Clock>,
        options: MountDriverOptions,
    ) -> Self {
        let jog = JogFlags::new();
        let abort = Arc::new(AlpacaMountAbort {
            transport: abort_transport,
            jog: Arc::clone(&jog),
        });
        Self {
            identity,
            transport,
            clock,
            options,
            abort,
            jog,
            connected: false,
            path: ConnectPath::Platform6,
            interface_version: None,
            device_state: false,
            caps: MountCapabilities::default(),
            unsupported: BTreeSet::new(),
            slots: HashMap::new(),
            next_due: HashMap::new(),
            connect_poll_interval: Duration::from_millis(250),
        }
    }

    /// The `connecting` poll interval (250 ms per §2.6; tests shorten it).
    pub fn with_connect_poll_interval(mut self, interval: Duration) -> Self {
        self.connect_poll_interval = interval;
        self
    }

    pub fn connect_path(&self) -> ConnectPath {
        self.path
    }

    /// `{base}/setup/v1/{type}/{n}/setup` for an Alpaca endpoint.
    pub fn setup_url(endpoint: &Endpoint) -> Option<String> {
        match endpoint {
            Endpoint::AlpacaHttp {
                base,
                device_type,
                number,
            } => Some(format!(
                "{}/setup/v1/{device_type}/{number}/setup",
                base.trim_end_matches('/')
            )),
            _ => None,
        }
    }

    fn require_connected(&self) -> DeviceResult<()> {
        if self.connected {
            Ok(())
        } else {
            Err(DeviceError::not_connected("the telescope is not connected"))
        }
    }

    fn begin(&self) -> (DateTime<Utc>, u64) {
        (self.clock.now_utc(), self.clock.now_mono_ns())
    }

    /// The midpoint of a request and its latency (§2.7).
    fn finish(&self, (utc0, mono0): (DateTime<Utc>, u64)) -> Timing {
        let (utc1, mono1) = self.begin();
        let half = mono1.saturating_sub(mono0) / 2;
        Timing {
            utc: utc0 + (utc1 - utc0) / 2,
            mono_ns: mono0 + half,
            latency_ms: u32::try_from(mono1.saturating_sub(mono0) / 1_000_000).unwrap_or(u32::MAX),
        }
    }

    fn memoise_unsupported(&mut self, member: &str) {
        self.unsupported.insert(member.to_string());
    }

    fn is_unsupported(&self, member: &str) -> bool {
        self.unsupported.contains(member)
    }

    fn record_ok(
        &mut self,
        member: &'static str,
        value: Value,
        timing: Timing,
        source: Source,
        device_time: Option<DateTime<Utc>>,
    ) -> RawMember {
        let (value, state) = if value.is_null() {
            (None, ObsState::NotSet)
        } else {
            (Some(value), ObsState::Fresh)
        };
        self.slots.insert(
            member,
            Slot {
                value: value.clone(),
                state,
                observed_at: Some(timing.utc),
                mono_ns: Some(timing.mono_ns),
                latency_ms: Some(timing.latency_ms),
                device_time,
                source,
            },
        );
        RawMember {
            name: member.to_string(),
            value,
            state,
            observed_at: Some(timing.utc),
            mono_ns: Some(timing.mono_ns),
            latency_ms: Some(timing.latency_ms),
            error: None,
        }
    }

    /// C-2: ValueNotSet is unknown, an unknown member is unsupported for the
    /// session, anything else keeps the last value as stale.
    fn record_err(&mut self, member: &'static str, err: &DeviceError) -> RawMember {
        let slot = self.slots.entry(member).or_default();
        let state = if err.is_not_set() {
            slot.value = None;
            ObsState::NotSet
        } else if err.is_unknown_member() {
            ObsState::Unsupported
        } else if slot.value.is_some() {
            ObsState::Stale
        } else {
            ObsState::Error
        };
        slot.state = state;
        if state == ObsState::Unsupported {
            slot.value = None;
            self.memoise_unsupported(member);
        }
        RawMember {
            name: member.to_string(),
            value: None,
            state,
            observed_at: None,
            mono_ns: None,
            latency_ms: None,
            error: Some(err.to_string()),
        }
    }

    fn read_member(&mut self, member: &'static str, query: &[(&str, String)]) -> RawMember {
        let started = self.begin();
        let result = self.transport.get(member, query, TimeoutClass::Ordinary);
        let timing = self.finish(started);
        match result {
            Ok(value) => self.record_ok(member, value, timing, Source::Property, None),
            Err(err) => self.record_err(member, &err),
        }
    }

    /// An observed value from the cache, without I/O.
    fn obs<T>(
        &self,
        member: &str,
        overdue: Option<Duration>,
        parse: impl Fn(&Value) -> Option<T>,
    ) -> Observed<T> {
        if self.is_unsupported(member) {
            return Observed::unsupported(Source::Property);
        }
        let Some(slot) = self.slots.get(member) else {
            return Observed::not_set(Source::Property);
        };
        let mut state = slot.state;
        if let (Some(overdue), Some(at), ObsState::Fresh) = (overdue, slot.mono_ns, state) {
            if self.clock.now_mono_ns().saturating_sub(at)
                > crate::equipment::clock::duration_ns(overdue)
            {
                state = ObsState::Stale;
            }
        }
        Observed {
            value: slot.value.as_ref().and_then(parse),
            state,
            observed_at: slot.observed_at,
            mono_ns: slot.mono_ns,
            device_time: slot.device_time,
            latency_ms: slot.latency_ms,
            source: slot.source,
        }
    }

    fn slow_overdue(member: &str) -> Option<Duration> {
        SLOW.iter()
            .find(|(m, _)| *m == member)
            .map(|(_, every)| Duration::from_secs(every * 3))
    }

    fn put(
        &mut self,
        member: &str,
        form: &[(&str, String)],
        class: TimeoutClass,
    ) -> DeviceResult<()> {
        self.require_connected()?;
        self.transport.put(member, form, class).map(|_| ())
    }

    /// A set invalidates the member: it is re-read on the next poll (C-2).
    fn invalidate(&mut self, member: &'static str) {
        self.next_due.insert(member, 0);
    }

    fn write_site(&mut self, site: &Site) -> DeviceResult<()> {
        self.put(
            "sitelatitude",
            &[("SiteLatitude", number(site.lat_deg))],
            TimeoutClass::Ordinary,
        )?;
        self.put(
            "sitelongitude",
            &[("SiteLongitude", number(site.lon_deg))],
            TimeoutClass::Ordinary,
        )?;
        if !site.elev_m.is_finite() || self.is_unsupported("siteelevation") {
            return Ok(());
        }
        match self.put(
            "siteelevation",
            &[("SiteElevation", number(site.elev_m))],
            TimeoutClass::Ordinary,
        ) {
            Err(err) if err.is_unknown_member() => {
                self.memoise_unsupported("siteelevation");
                Ok(())
            }
            other => other,
        }
    }

    // ---- connect helpers ----

    /// Poll `connecting` until it reads false, for up to `timeout` of
    /// elapsed time (§2.6 step 4) whatever each GET's latency. This blocking
    /// connect path runs on the worker thread and sleeps in real time, so
    /// the deadline is a real-time `Instant`, not the injected clock (a
    /// manual clock would never reach it). One slow GET can overrun the
    /// deadline by at most its own connect-class timeout.
    fn wait_connecting(&mut self, timeout: Duration) -> DeviceResult<()> {
        let interval = self.connect_poll_interval.max(Duration::from_millis(1));
        let deadline = std::time::Instant::now() + timeout;
        loop {
            match self.transport.get("connecting", &[], TimeoutClass::Connect) {
                Ok(value) if as_bool(&value) == Some(false) => return Ok(()),
                Ok(_) => {}
                Err(err) if err.is_unknown_member() => return Ok(()),
                Err(err) => return Err(err),
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                break;
            }
            std::thread::sleep(interval.min(deadline - now));
        }
        Err(DeviceError::timeout(format!(
            "Connecting still reads true after {:.1} s",
            timeout.as_secs_f64()
        )))
    }

    fn probe(
        &mut self,
        member: &str,
        query: &[(&str, String)],
        report: &mut ConnectReport,
    ) -> Option<Value> {
        if self.is_unsupported(member) {
            return None;
        }
        match self.transport.get(member, query, TimeoutClass::Ordinary) {
            Ok(Value::Null) => None,
            Ok(value) => Some(value),
            Err(err) if err.is_unknown_member() => {
                self.memoise_unsupported(member);
                None
            }
            Err(err) if err.is_not_set() => None,
            Err(err) => {
                report.warnings.push(format!("{member}: {err}"));
                None
            }
        }
    }

    fn probe_bool(&mut self, member: &str, report: &mut ConnectReport) -> bool {
        self.probe(member, &[], report)
            .as_ref()
            .and_then(as_bool)
            .unwrap_or(false)
    }

    fn read_identity(&mut self, report: &mut ConnectReport) {
        let mut text = |this: &mut Self, member: &str| {
            this.probe(member, &[], report)
                .and_then(|v| v.as_str().map(str::to_string))
                .filter(|s| !s.trim().is_empty())
        };
        if let Some(name) = text(self, "name") {
            self.identity.name = name;
        }
        self.identity.description = text(self, "description");
        self.identity.driver_info = text(self, "driverinfo");
        self.identity.driver_version = text(self, "driverversion");
        self.identity.interface_version = self.interface_version;
        self.identity.setup_url = Self::setup_url(&self.identity.endpoint);
    }

    fn probe_capabilities(&mut self, report: &mut ConnectReport) {
        let iv = self.interface_version.unwrap_or(1);
        let mut flags = HashMap::new();
        for member in CAN_FLAGS {
            flags.insert(*member, self.probe_bool(member, report));
        }
        let flag = |name: &str| flags.get(name).copied().unwrap_or(false);
        let mut can_move_axis = [false; 3];
        for axis in 0..3u8 {
            can_move_axis[usize::from(axis)] = self
                .probe("canmoveaxis", &[("Axis", axis.to_string())], report)
                .as_ref()
                .and_then(as_bool)
                .unwrap_or(false);
        }
        let alignment_mode = self
            .probe("alignmentmode", &[], report)
            .as_ref()
            .and_then(as_i64)
            .and_then(AlignmentMode::from_ascom);
        let equatorial_system = if iv >= 2 {
            self.probe("equatorialsystem", &[], report)
                .as_ref()
                .and_then(as_i64)
                .and_then(|v| i32::try_from(v).ok())
        } else {
            None
        };
        let (native_frame, frame_reason) =
            native_frame_policy(equatorial_system, Some(iv), &self.options);
        let does_refraction = self
            .probe("doesrefraction", &[], report)
            .as_ref()
            .and_then(as_bool);
        let tracking_rates = self
            .probe("trackingrates", &[], report)
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default()
            .iter()
            .filter_map(as_i64)
            .filter_map(DriveRate::from_ascom)
            .collect();
        let mut axis_rates = Vec::new();
        for axis in 0..3u8 {
            if !can_move_axis[usize::from(axis)] {
                continue;
            }
            let ranges = self
                .probe("axisrates", &[("Axis", axis.to_string())], report)
                .and_then(|v| v.as_array().cloned())
                .unwrap_or_default()
                .iter()
                .filter_map(Value::as_object)
                .filter_map(|o| {
                    Some(RateRange {
                        min_deg_s: get_ci(o, "Minimum").and_then(as_f64)?,
                        max_deg_s: get_ci(o, "Maximum").and_then(as_f64)?,
                    })
                })
                .collect();
            axis_rates.push(AxisRates { axis, ranges });
        }
        let metres_to_mm = |v: Option<Value>| {
            v.as_ref()
                .and_then(as_f64)
                .filter(|m| *m > 0.0)
                .map(|m| m * 1000.0)
        };
        let focal_length_mm = metres_to_mm(self.probe("focallength", &[], report));
        let aperture_diameter_mm = metres_to_mm(self.probe("aperturediameter", &[], report));
        if iv >= PLATFORM7_INTERFACE {
            self.device_state = self
                .probe("devicestate", &[], report)
                .and_then(|v| v.as_array().map(|a| !a.is_empty()))
                .unwrap_or(false);
        } else {
            self.memoise_unsupported("devicestate");
        }
        self.caps = MountCapabilities {
            can_park: flag("canpark"),
            can_unpark: flag("canunpark"),
            can_set_park: flag("cansetpark"),
            can_find_home: flag("canfindhome"),
            can_slew: flag("canslew"),
            can_slew_async: flag("canslewasync"),
            can_slew_altaz: flag("canslewaltaz"),
            can_slew_altaz_async: flag("canslewaltazasync"),
            can_sync: flag("cansync"),
            can_sync_altaz: flag("cansyncaltaz"),
            can_set_tracking: flag("cansettracking"),
            can_set_pier_side: flag("cansetpierside"),
            can_pulse_guide: flag("canpulseguide"),
            can_set_guide_rates: flag("cansetguiderates"),
            can_move_axis,
            can_set_declination_rate: flag("cansetdeclinationrate"),
            can_set_right_ascension_rate: flag("cansetrightascensionrate"),
            alignment_mode,
            equatorial_system,
            native_frame,
            frame_reason,
            does_refraction,
            tracking_rates,
            axis_rates,
            focal_length_mm,
            aperture_diameter_mm,
            coordinate_precision_arcsec: None,
            interface_version: Some(iv),
            device_state: self.device_state,
            unsupported_members: self.unsupported.iter().cloned().collect(),
        };
    }

    /// The mount's site. Latitude and longitude are required; ASCOM lets
    /// SiteElevation raise NotImplemented, in which case the site is still
    /// reported with the elevation unknown: `f64::NAN`, which
    /// `Site::differs_from` ignores and JSON publishes as null.
    fn site_obs(&self) -> Observed<Site> {
        let overdue = Self::slow_overdue("sitelatitude");
        let lat = self.obs("sitelatitude", overdue, as_f64);
        let lon = self.obs("sitelongitude", overdue, as_f64);
        let elev = self.obs("siteelevation", overdue, as_f64);
        if lat.state == ObsState::Unsupported || lon.state == ObsState::Unsupported {
            return Observed::unsupported(Source::Property);
        }
        let elevation_unsupported = elev.state == ObsState::Unsupported;
        let elev_m = if elevation_unsupported {
            Some(f64::NAN)
        } else {
            elev.value
        };
        let (Some(lat_deg), Some(lon_deg), Some(elev_m)) = (lat.value, lon.value, elev_m) else {
            return Observed::not_set(Source::Property);
        };
        let mut parts = vec![lat, lon];
        if !elevation_unsupported {
            parts.push(elev);
        }
        let oldest = parts
            .iter()
            .min_by_key(|p| p.mono_ns.unwrap_or(0))
            .cloned()
            .unwrap_or_default();
        let state = if parts.iter().all(|p| p.state == ObsState::Fresh) {
            ObsState::Fresh
        } else {
            ObsState::Stale
        };
        Observed {
            value: Some(Site {
                lat_deg,
                lon_deg,
                elev_m,
                source: SiteSource::Mount,
            }),
            state,
            observed_at: oldest.observed_at,
            mono_ns: oldest.mono_ns,
            device_time: None,
            latency_ms: oldest.latency_ms,
            source: Source::Property,
        }
    }

    fn target_obs(&self) -> Observed<crate::equipment::types::EqCoord> {
        let ra = self.obs("targetrightascension", None, as_f64);
        let dec = self.obs("targetdeclination", None, as_f64);
        if ra.state == ObsState::Unsupported || dec.state == ObsState::Unsupported {
            return Observed::unsupported(Source::Property);
        }
        let frame = self.caps.native_frame;
        let state = if ra.state == ObsState::Fresh && dec.state == ObsState::Fresh {
            ObsState::Fresh
        } else if ra.state == ObsState::Fresh {
            dec.state
        } else {
            ra.state
        };
        let value = ra
            .value
            .zip(dec.value)
            .map(|(ra, dec)| EqCoord::new(ra, dec, frame));
        Observed {
            value,
            state,
            observed_at: ra.observed_at,
            mono_ns: ra.mono_ns,
            device_time: ra.device_time,
            latency_ms: ra.latency_ms,
            source: ra.source,
        }
    }
}

impl Device for AlpacaTelescope {
    fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    fn connect(&mut self, timeout: Duration) -> DeviceResult<ConnectReport> {
        let started = self.clock.now_mono_ns();
        let mut report = ConnectReport::default();
        let iv = self
            .transport
            .get("interfaceversion", &[], TimeoutClass::Connect)
            .ok()
            .as_ref()
            .and_then(as_i64)
            .and_then(|v| i32::try_from(v).ok())
            .unwrap_or(1);
        self.interface_version = Some(iv);
        report.interface_version = Some(iv);
        let mut platform7 = false;
        if iv >= PLATFORM7_INTERFACE {
            match self.transport.put("connect", &[], TimeoutClass::Connect) {
                Ok(_) => {
                    self.wait_connecting(timeout)?;
                    platform7 = true;
                }
                Err(err) if err.is_unknown_member() => self.memoise_unsupported("connect"),
                Err(err) => return Err(err),
            }
        }
        if !platform7 {
            self.transport.put(
                "connected",
                &[("Connected", boolean(true))],
                TimeoutClass::Connect,
            )?;
        }
        let connected = self
            .transport
            .get("connected", &[], TimeoutClass::Connect)?;
        if as_bool(&connected) != Some(true) {
            return Err(DeviceError::not_connected(
                "the telescope reports Connected = false after connecting",
            ));
        }
        self.path = if platform7 {
            ConnectPath::Platform7
        } else {
            ConnectPath::Platform6
        };
        self.connected = true;
        self.read_identity(&mut report);
        self.probe_capabilities(&mut report);
        for member in ["sitelatitude", "sitelongitude", "siteelevation"] {
            if !self.is_unsupported(member) {
                self.read_member(member, &[]);
            }
        }
        let now = self.clock.now_mono_ns();
        for (member, every) in SLOW {
            if member.starts_with("site") {
                self.next_due.insert(member, now + every * 1_000_000_000);
            }
        }
        report.path = self.path;
        report.device_state = self.device_state;
        report.unsupported_members = self.unsupported.iter().cloned().collect();
        report.duration_ms = self.clock.now_mono_ns().saturating_sub(started) / 1_000_000;
        Ok(report)
    }

    fn disconnect(&mut self) -> DeviceResult<()> {
        let result = if self.path == ConnectPath::Platform7 {
            self.transport
                .put("disconnect", &[], TimeoutClass::Connect)
                .and_then(|_| self.wait_connecting(Duration::from_secs(10)))
        } else {
            self.transport
                .put(
                    "connected",
                    &[("Connected", boolean(false))],
                    TimeoutClass::Connect,
                )
                .map(|_| ())
        };
        self.connected = false;
        result
    }

    fn is_connected(&mut self) -> DeviceResult<bool> {
        let value = self
            .transport
            .get("connected", &[], TimeoutClass::Connect)?;
        Ok(as_bool(&value) == Some(true))
    }

    fn poll(&mut self, plan: &PollPlan) -> DeviceResult<RawPoll> {
        self.require_connected()?;
        let started_mono_ns = self.clock.now_mono_ns();
        let mut members = Vec::new();
        let mut method = PollMethod::PerMember;
        let mut device_time = None;
        let mut pending: Vec<(&'static str, &'static str)> = CORE
            .iter()
            .copied()
            .filter(|(member, _)| !self.is_unsupported(member))
            .collect();
        if self.device_state {
            let started = self.begin();
            let result = self
                .transport
                .get("devicestate", &[], TimeoutClass::Ordinary);
            let timing = self.finish(started);
            match result {
                Ok(value) => {
                    method = PollMethod::DeviceState;
                    let entries = parse_device_state(&value);
                    device_time = entries
                        .get("timestamp")
                        .and_then(Value::as_str)
                        .and_then(|s| parse_alpaca_datetime(s, true));
                    let mut missing = Vec::new();
                    for (member, ds_name) in pending {
                        match entries.get(&ds_name.to_ascii_lowercase()) {
                            Some(value) => members.push(self.record_ok(
                                member,
                                value.clone(),
                                timing,
                                Source::DeviceState,
                                device_time,
                            )),
                            None => missing.push((member, ds_name)),
                        }
                    }
                    pending = missing;
                }
                Err(err) if err.is_unknown_member() => {
                    self.device_state = false;
                    self.caps.device_state = false;
                    self.memoise_unsupported("devicestate");
                }
                Err(err) => {
                    for (member, _) in pending.drain(..) {
                        members.push(self.record_err(member, &err));
                    }
                    return Err(err);
                }
            }
        }
        let per_member = !pending.is_empty();
        let mut failures = Vec::new();
        for (member, _) in &pending {
            let raw = self.read_member(member, &[]);
            if raw.state == ObsState::Error || raw.state == ObsState::Stale {
                failures.push(raw.error.clone().unwrap_or_default());
            }
            members.push(raw);
        }
        // Every per-member core read failed: the device is not answering, so
        // the slow members are not tried on this tick either.
        if per_member && method == PollMethod::PerMember && failures.len() == pending.len() {
            return Err(DeviceError::transport(format!(
                "every telescope member read failed: {}",
                failures.first().cloned().unwrap_or_default()
            )));
        }
        let now = self.clock.now_mono_ns();
        for (member, every) in SLOW {
            if self.is_unsupported(member) {
                continue;
            }
            let named = plan
                .extra_members
                .iter()
                .any(|m| m.eq_ignore_ascii_case(member));
            let due = self.next_due.get(member).is_none_or(|at| now >= *at);
            if named || due {
                members.push(self.read_member(member, &[]));
                self.next_due.insert(member, now + every * 1_000_000_000);
            }
        }
        if plan.active {
            for member in TARGET {
                if !self.is_unsupported(member) {
                    members.push(self.read_member(member, &[]));
                }
            }
        }
        Ok(RawPoll {
            method,
            started_mono_ns,
            finished_mono_ns: self.clock.now_mono_ns(),
            device_time,
            members,
        })
    }

    fn supported_actions(&mut self) -> DeviceResult<Vec<String>> {
        let value = self
            .transport
            .get("supportedactions", &[], TimeoutClass::Ordinary)?;
        Ok(value
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect())
    }

    fn action(&mut self, name: &str, params: &str) -> DeviceResult<String> {
        self.require_connected()?;
        let value = self.transport.put(
            "action",
            &[
                ("Action", name.to_string()),
                ("Parameters", params.to_string()),
            ],
            TimeoutClass::Long,
        )?;
        Ok(match value {
            Value::String(s) => s,
            Value::Null => String::new(),
            other => other.to_string(),
        })
    }

    fn abort_path(&self) -> Option<Arc<dyn AbortPath>> {
        Some(Arc::clone(&self.abort) as Arc<dyn AbortPath>)
    }
}

impl MountDriver for AlpacaTelescope {
    fn capabilities(&mut self) -> DeviceResult<MountCapabilities> {
        let mut caps = self.caps.clone();
        caps.device_state = self.device_state;
        caps.unsupported_members = self.unsupported.iter().cloned().collect();
        Ok(caps)
    }

    fn state(&mut self) -> DeviceResult<MountState> {
        let f = as_f64;
        let b = as_bool;
        let rate_overdue = Self::slow_overdue("trackingrate");
        let guide_overdue = Self::slow_overdue("guideraterightascension");
        let to_arcsec = |v: &Value| as_f64(v).map(|deg_s| deg_s * 3600.0);
        Ok(MountState {
            ra_h: self.obs("rightascension", None, f),
            dec_deg: self.obs("declination", None, f),
            alt_deg: self.obs("altitude", None, f),
            az_deg: self.obs("azimuth", None, f),
            sidereal_time_h: self.obs("siderealtime", None, f),
            utc: self.obs("utcdate", None, |v: &Value| {
                v.as_str().and_then(|s| parse_alpaca_datetime(s, false))
            }),
            slewing: self.obs("slewing", None, b),
            tracking: self.obs("tracking", None, b),
            tracking_rate: self.obs("trackingrate", rate_overdue, |v: &Value| {
                as_i64(v).and_then(DriveRate::from_ascom)
            }),
            at_park: self.obs("atpark", None, b),
            at_home: self.obs("athome", None, b),
            side_of_pier: self.obs("sideofpier", None, |v: &Value| {
                as_i64(v)
                    .map(PierSide::from_ascom)
                    .filter(|side| *side != PierSide::Unknown)
            }),
            is_pulse_guiding: self.obs("ispulseguiding", None, b),
            guide_rate_ra_arcsec_s: self.obs("guideraterightascension", guide_overdue, to_arcsec),
            guide_rate_dec_arcsec_s: self.obs("guideratedeclination", guide_overdue, to_arcsec),
            site: self.site_obs(),
            target: self.target_obs(),
        })
    }

    fn slew_async(&mut self, target: EqCoord) -> DeviceResult<()> {
        let target = target.normalized();
        self.put(
            "slewtocoordinatesasync",
            &[
                ("RightAscension", number(target.ra_h)),
                ("Declination", number(target.dec_deg)),
            ],
            TimeoutClass::Long,
        )
    }

    fn slew_altaz_async(&mut self, target: AltAz) -> DeviceResult<()> {
        self.put(
            "slewtoaltazasync",
            &[
                ("Azimuth", number(target.az_deg)),
                ("Altitude", number(target.alt_deg)),
            ],
            TimeoutClass::Long,
        )
    }

    fn abort_slew(&mut self) -> DeviceResult<()> {
        self.put("abortslew", &[], TimeoutClass::Long)
    }

    fn sync(&mut self, target: EqCoord) -> DeviceResult<()> {
        let target = target.normalized();
        self.put(
            "synctocoordinates",
            &[
                ("RightAscension", number(target.ra_h)),
                ("Declination", number(target.dec_deg)),
            ],
            TimeoutClass::Long,
        )
    }

    fn set_tracking(&mut self, on: bool) -> DeviceResult<()> {
        self.put(
            "tracking",
            &[("Tracking", boolean(on))],
            TimeoutClass::Ordinary,
        )
    }

    fn set_tracking_rate(&mut self, rate: DriveRate) -> DeviceResult<()> {
        self.put(
            "trackingrate",
            &[("TrackingRate", rate.to_ascom().to_string())],
            TimeoutClass::Ordinary,
        )?;
        self.invalidate("trackingrate");
        Ok(())
    }

    fn park(&mut self) -> DeviceResult<()> {
        self.put("park", &[], TimeoutClass::Long)
    }

    fn unpark(&mut self) -> DeviceResult<()> {
        self.put("unpark", &[], TimeoutClass::Long)
    }

    fn set_park(&mut self) -> DeviceResult<()> {
        self.put("setpark", &[], TimeoutClass::Ordinary)
    }

    fn find_home(&mut self) -> DeviceResult<()> {
        self.put("findhome", &[], TimeoutClass::Long)
    }

    fn move_axis(&mut self, axis: u8, rate_deg_s: f64) -> DeviceResult<()> {
        self.put(
            "moveaxis",
            &[("Axis", axis.to_string()), ("Rate", number(rate_deg_s))],
            TimeoutClass::Long,
        )?;
        self.jog.set(axis, rate_deg_s);
        Ok(())
    }

    fn pulse_guide(&mut self, dir: GuideDirection, ms: u32) -> DeviceResult<()> {
        self.put(
            "pulseguide",
            &[
                ("Direction", dir.to_ascom().to_string()),
                ("Duration", ms.to_string()),
            ],
            TimeoutClass::Long,
        )
    }

    fn set_guide_rates(&mut self, ra_deg_s: f64, dec_deg_s: f64) -> DeviceResult<()> {
        self.put(
            "guideraterightascension",
            &[("GuideRateRightAscension", number(ra_deg_s))],
            TimeoutClass::Ordinary,
        )?;
        self.put(
            "guideratedeclination",
            &[("GuideRateDeclination", number(dec_deg_s))],
            TimeoutClass::Ordinary,
        )?;
        self.invalidate("guideraterightascension");
        self.invalidate("guideratedeclination");
        Ok(())
    }

    /// Latitude and longitude are required. SiteElevation is written only
    /// when it is implemented (ASCOM allows NotImplemented); a NotImplemented
    /// answer is memoised and is not a failure. The members are re-read on
    /// the next poll even after a partial write.
    fn set_site(&mut self, site: &Site) -> DeviceResult<()> {
        let result = self.write_site(site);
        for member in ["sitelatitude", "sitelongitude", "siteelevation"] {
            self.invalidate(member);
        }
        result
    }

    fn set_utc(&mut self, utc: DateTime<Utc>) -> DeviceResult<()> {
        self.put(
            "utcdate",
            &[("UTCDate", utc.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string())],
            TimeoutClass::Ordinary,
        )
    }

    fn destination_side_of_pier(&mut self, target: EqCoord) -> DeviceResult<PierSide> {
        self.require_connected()?;
        let target = target.normalized();
        let value = self.transport.get(
            "destinationsideofpier",
            &[
                ("RightAscension", number(target.ra_h)),
                ("Declination", number(target.dec_deg)),
            ],
            TimeoutClass::Ordinary,
        )?;
        as_i64(&value)
            .map(PierSide::from_ascom)
            .ok_or_else(|| DeviceError::driver(format!("DestinationSideOfPier returned {value}")))
    }

    fn set_side_of_pier(&mut self, side: PierSide) -> DeviceResult<()> {
        self.put(
            "sideofpier",
            &[("SideOfPier", side.to_ascom().to_string())],
            TimeoutClass::Long,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::commands::MountCmd;
    use crate::equipment::mount::limits::MountLimits;
    use crate::equipment::mount::{plan, CommandMeta, MountPlan, MountRuntime, MountSettings};
    use crate::equipment::ops::OpProgress;
    use crate::equipment::testing::fake_transport::{FakeAlpacaTransport, FakeReply, Method};
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::traits::IdentityConverter;
    use crate::equipment::types::{
        identity_key, CommandId, CommandSource, DeviceKind, DriverId, EqFrame, OpOutcome,
    };
    use serde_json::json;

    fn identity() -> DeviceIdentity {
        DeviceIdentity::new(
            DeviceKind::Mount,
            DriverId::Alpaca,
            Endpoint::AlpacaHttp {
                base: "http://esp32.local:80".into(),
                device_type: "telescope".into(),
                number: 0,
            },
            identity_key::alpaca(None, "esp32.local", 80, "telescope", 0),
            "telescope 0",
        )
    }

    /// The ESP32-like telescope: IV4, DeviceState, Polar, Topocentric,
    /// CanSlew = false with CanSlewAsync = true (O10).
    fn esp32() -> Arc<FakeAlpacaTransport> {
        let fake = FakeAlpacaTransport::new();
        let ok = FakeReply::ok;
        let v = |value: Value| FakeReply::Value(value);
        fake.on_get("interfaceversion", v(json!(4)))
            .on_put("connect", ok())
            .on_get("connecting", v(json!(false)))
            .on_get("connected", v(json!(true)))
            .on_get("name", v(json!("ESP32 Mount")))
            .on_get("description", v(json!("ESP32 Alpaca telescope")))
            .on_get("driverinfo", v(json!("achernar-mount-esp32")))
            .on_get("driverversion", v(json!("1.0")));
        for (member, value) in [
            ("canpark", true),
            ("canunpark", true),
            ("cansetpark", false),
            ("canfindhome", false),
            ("canslew", false),
            ("canslewasync", true),
            ("canslewaltaz", false),
            ("canslewaltazasync", false),
            ("cansync", true),
            ("cansyncaltaz", false),
            ("cansettracking", true),
            ("cansetpierside", false),
            ("canpulseguide", true),
            ("cansetguiderates", true),
            ("cansetdeclinationrate", false),
            ("cansetrightascensionrate", false),
        ] {
            fake.on_get(member, v(json!(value)));
        }
        fake.queue_get(
            "canmoveaxis",
            [v(json!(true)), v(json!(true)), v(json!(false))],
        )
        .on_get("axisrates", v(json!([{"Minimum": 0.0, "Maximum": 4.0}])))
        .on_get("alignmentmode", v(json!(1)))
        .on_get("equatorialsystem", v(json!(1)))
        .on_get("doesrefraction", v(json!(false)))
        .on_get("trackingrates", v(json!([0, 1, 2, 3])))
        .on_get("focallength", FakeReply::not_implemented())
        .on_get("aperturediameter", FakeReply::not_implemented())
        .on_get("sitelatitude", v(json!(40.0)))
        .on_get("sitelongitude", v(json!(-75.0)))
        .on_get("siteelevation", v(json!(100.0)))
        .on_get("trackingrate", v(json!(0)))
        .on_get("guideraterightascension", v(json!(0.002089)))
        .on_get("guideratedeclination", v(json!(0.002089)))
        .on_get("targetrightascension", FakeReply::value_not_set())
        .on_get("targetdeclination", FakeReply::value_not_set())
        .on_get(
            "devicestate",
            v(device_state(
                5.5,
                20.0,
                false,
                "2026-01-01T00:00:00.1234567Z",
            )),
        );
        for member in [
            "slewtocoordinatesasync",
            "abortslew",
            "moveaxis",
            "pulseguide",
            "tracking",
            "trackingrate",
            "park",
            "unpark",
            "synctocoordinates",
            "guideraterightascension",
            "guideratedeclination",
            "sitelatitude",
            "sitelongitude",
            "siteelevation",
            "utcdate",
            "disconnect",
        ] {
            fake.on_put(member, ok());
        }
        fake.shared()
    }

    fn device_state(ra: f64, dec: f64, slewing: bool, timestamp: &str) -> Value {
        json!([
            {"Name": "Altitude", "Value": 45.0},
            {"Name": "AtHome", "Value": false},
            {"Name": "AtPark", "Value": false},
            {"Name": "Azimuth", "Value": 180.0},
            {"Name": "Declination", "Value": dec},
            {"Name": "IsPulseGuiding", "Value": false},
            {"Name": "RightAscension", "Value": ra},
            {"Name": "SideOfPier", "Value": 0},
            {"Name": "SiderealTime", "Value": 6.0},
            {"Name": "Slewing", "Value": slewing},
            {"Name": "Tracking", "Value": true},
            {"Name": "UTCDate", "Value": "2026-01-01T00:00:00.000Z"},
            {"Name": "TimeStamp", "Value": timestamp},
        ])
    }

    /// A Platform 6 German mount with per-member reads and
    /// CanSetGuideRates = false.
    fn platform6() -> Arc<FakeAlpacaTransport> {
        let fake = FakeAlpacaTransport::new();
        let v = |value: Value| FakeReply::Value(value);
        fake.on_get("interfaceversion", v(json!(3)))
            .on_put("connected", FakeReply::ok())
            .on_get("connected", v(json!(true)))
            .on_get("name", v(json!("GEM")))
            .on_get("canslewasync", v(json!(true)))
            .on_get("canslew", v(json!(true)))
            .on_get("cansettracking", v(json!(true)))
            .on_get("canpulseguide", v(json!(true)))
            .on_get("cansetguiderates", v(json!(false)))
            .on_get("canmoveaxis", v(json!(false)))
            .on_get("alignmentmode", v(json!(2)))
            .on_get("equatorialsystem", v(json!(2)))
            .on_get("trackingrates", v(json!([0, 2])))
            .on_get("focallength", v(json!(0.8)))
            .on_get("aperturediameter", v(json!(0.2)))
            .on_get("rightascension", v(json!(5.5)))
            .on_get("declination", v(json!(20.0)))
            .on_get("altitude", v(json!(50.0)))
            .on_get("azimuth", v(json!(170.0)))
            .on_get("siderealtime", v(json!(6.0)))
            .on_get("utcdate", v(json!("2026-01-01T00:00:00")))
            .on_get("slewing", v(json!(false)))
            .on_get("tracking", v(json!(true)))
            .on_get("atpark", v(json!(false)))
            .on_get("athome", v(json!(false)))
            .on_get("sideofpier", v(json!(1)))
            .on_get("ispulseguiding", v(json!(false)))
            .on_get("trackingrate", v(json!(2)))
            .on_get("guideraterightascension", v(json!(15.041 / 3600.0)))
            .on_get("guideratedeclination", v(json!(7.52 / 3600.0)))
            .on_get("sitelatitude", v(json!(40.0)))
            .on_get("sitelongitude", v(json!(-75.0)))
            .on_get("siteelevation", v(json!(100.0)))
            .on_get("targetrightascension", v(json!(5.5)))
            .on_get("targetdeclination", v(json!(20.0)))
            .on_put("slewtocoordinatesasync", FakeReply::ok())
            .on_put("abortslew", FakeReply::ok());
        fake.shared()
    }

    struct Setup {
        clock: Arc<ManualClock>,
        fake: Arc<FakeAlpacaTransport>,
        abort: Arc<FakeAlpacaTransport>,
        drv: AlpacaTelescope,
    }

    fn setup(fake: Arc<FakeAlpacaTransport>) -> Setup {
        let clock = Arc::new(ManualClock::new());
        let abort = FakeAlpacaTransport::new();
        abort
            .on_put("abortslew", FakeReply::ok())
            .on_put("moveaxis", FakeReply::ok());
        let abort = abort.shared();
        let drv = AlpacaTelescope::new(
            identity(),
            fake.clone(),
            abort.clone(),
            clock.clone(),
            MountDriverOptions::default(),
        )
        .with_connect_poll_interval(Duration::from_millis(1));
        Setup {
            clock,
            fake,
            abort,
            drv,
        }
    }

    fn connected(fake: Arc<FakeAlpacaTransport>) -> (Setup, ConnectReport) {
        let mut s = setup(fake);
        let report = s.drv.connect(Duration::from_secs(5)).unwrap();
        (s, report)
    }

    fn runtime(s: &mut Setup) -> MountRuntime {
        let mut runtime = MountRuntime::new(
            MountSettings::default(),
            MountLimits::default(),
            Arc::new(IdentityConverter),
            s.clock.clone(),
            s.drv.identity().connection_id,
            None,
        );
        runtime.on_connected(&mut s.drv).unwrap();
        runtime.after_poll(&mut s.drv, None);
        runtime
    }

    fn run(s: &mut Setup, runtime: &mut MountRuntime, cmd: MountCmd) -> OpOutcome {
        let meta = CommandMeta::new(
            CommandId::new(),
            CommandSource::Operator { overriding: false },
        );
        let MountPlan::Start(mut op) = runtime.plan(cmd, meta, None) else {
            panic!("not startable");
        };
        if let Err(err) = op.start(&mut s.drv, s.clock.as_ref()) {
            return OpOutcome::Failed(err);
        }
        for _ in 0..400 {
            s.clock.advance(Duration::from_millis(250));
            let _ = s.drv.poll(&plan(1, true, &[]));
            runtime.after_poll(&mut s.drv, None);
            if let OpProgress::Done(outcome) = op.step(&mut s.drv, s.clock.as_ref()) {
                return outcome;
            }
        }
        panic!("operation did not finish");
    }

    #[test]
    fn platform7_connect_uses_connect_and_connecting() {
        let (s, report) = connected(esp32());
        assert_eq!(report.path, ConnectPath::Platform7);
        assert_eq!(report.interface_version, Some(4));
        assert!(report.device_state);
        assert_eq!(s.fake.calls_to(Method::Put, "connect"), 1);
        assert_eq!(s.fake.calls_to(Method::Put, "connected"), 0);
        assert!(s.fake.calls_to(Method::Get, "connecting") >= 1);
        let identity = s.drv.identity();
        assert_eq!(identity.name, "ESP32 Mount");
        assert_eq!(identity.driver_version.as_deref(), Some("1.0"));
        assert_eq!(identity.interface_version, Some(4));
        assert_eq!(
            identity.setup_url.as_deref(),
            Some("http://esp32.local:80/setup/v1/telescope/0/setup")
        );
        assert!(report
            .unsupported_members
            .contains(&"focallength".to_string()));
    }

    #[test]
    fn platform6_connect_sets_connected_and_reads_it_back() {
        let (s, report) = connected(platform6());
        assert_eq!(report.path, ConnectPath::Platform6);
        assert!(!report.device_state);
        let puts: Vec<_> = s
            .fake
            .calls()
            .into_iter()
            .filter(|c| c.method == Method::Put && c.member == "connected")
            .collect();
        assert_eq!(puts.len(), 1);
        assert_eq!(
            puts[0].params,
            vec![("Connected".to_string(), "True".to_string())]
        );
        assert_eq!(s.fake.calls_to(Method::Put, "connect"), 0);
        assert_eq!(
            s.fake.calls_to(Method::Get, "devicestate"),
            0,
            "below Platform 7"
        );
    }

    #[test]
    fn an_older_server_without_connect_falls_back_to_platform6() {
        let fake = esp32();
        fake.on_put("connect", FakeReply::http(400, "Unknown method connect"))
            .on_put("connected", FakeReply::ok());
        let (_s, report) = connected(fake);
        assert_eq!(report.path, ConnectPath::Platform6);
        assert!(report.unsupported_members.contains(&"connect".to_string()));
    }

    #[test]
    fn an_unreadable_interface_version_counts_as_one() {
        let fake = platform6();
        fake.on_get("interfaceversion", FakeReply::Timeout);
        let (mut s, report) = connected(fake);
        assert_eq!(report.interface_version, Some(1));
        assert_eq!(s.fake.calls_to(Method::Get, "equatorialsystem"), 0);
        let caps = s.drv.capabilities().unwrap();
        assert_eq!(caps.native_frame, EqFrame::Unknown);
        assert_eq!(
            caps.frame_reason.as_deref(),
            Some("equatorial_system_needs_interface_v2")
        );
    }

    #[test]
    fn a_connect_that_stays_connecting_times_out() {
        let fake = esp32();
        fake.on_get("connecting", FakeReply::value(true));
        let mut s = setup(fake);
        let err = s.drv.connect(Duration::from_millis(5)).unwrap_err();
        assert_eq!(err.class, ErrorClass::Timeout);
    }

    #[test]
    fn equatorial_system_cases_zero_to_four_map_per_m_a() {
        let cases = [
            (0, EqFrame::Unknown, Some("equatorial_system_other")),
            (1, EqFrame::JNowApparent, None),
            (2, EqFrame::J2000, None),
            (3, EqFrame::Unknown, Some("unsupported_epoch")),
            (4, EqFrame::Unknown, Some("unsupported_epoch")),
        ];
        for (code, frame, reason) in cases {
            let fake = esp32();
            fake.on_get("equatorialsystem", FakeReply::value(code));
            let (mut s, _) = connected(fake);
            let caps = s.drv.capabilities().unwrap();
            assert_eq!(caps.equatorial_system, Some(code), "code {code}");
            assert_eq!(caps.native_frame, frame, "code {code}");
            assert_eq!(caps.frame_reason.as_deref(), reason, "code {code}");
        }
        // The override wins.
        let mut s = setup(esp32());
        s.drv.options = MountDriverOptions {
            frame_override: Some(EqFrame::J2000),
            ..MountDriverOptions::default()
        };
        s.drv.connect(Duration::from_secs(1)).unwrap();
        assert_eq!(s.drv.capabilities().unwrap().native_frame, EqFrame::J2000);
    }

    #[test]
    fn capabilities_and_optics_claims_are_probed_once() {
        let (mut s, report) = connected(platform6());
        let caps = s.drv.capabilities().unwrap();
        assert_eq!(caps.alignment_mode, Some(AlignmentMode::GermanPolar));
        assert_eq!(caps.native_frame, EqFrame::J2000);
        assert_eq!(
            caps.tracking_rates,
            vec![DriveRate::Sidereal, DriveRate::Solar]
        );
        assert_eq!(caps.focal_length_mm, Some(800.0));
        assert_eq!(caps.aperture_diameter_mm, Some(200.0));
        assert!(caps.can_slew_async && !caps.can_set_guide_rates);
        // Unscripted Can* members answer HTTP 400: unsupported, not failures.
        assert!(caps.unsupported_members.contains(&"canpark".to_string()));
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        let esp = connected(esp32()).0.drv.capabilities().unwrap();
        assert_eq!(esp.can_move_axis, [true, true, false]);
        assert_eq!(esp.axis_rates.len(), 2);
        assert_eq!(esp.axis_rates[0].ranges[0].max_deg_s, 4.0);
        assert!(!esp.can_slew && esp.can_slew_async);
    }

    #[test]
    fn device_state_is_one_get_per_tick_with_its_timestamp() {
        let (mut s, _) = connected(esp32());
        s.fake.clear_calls();
        let raw = s.drv.poll(&plan(1, false, &[])).unwrap();
        assert_eq!(raw.method, PollMethod::DeviceState);
        assert_eq!(s.fake.calls_to(Method::Get, "devicestate"), 1);
        assert_eq!(s.fake.calls_to(Method::Get, "rightascension"), 0);
        assert_eq!(
            raw.device_time,
            Some(
                chrono::Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap()
                    + chrono::Duration::nanoseconds(123_456_700)
            )
        );
        let state = s.drv.state().unwrap();
        assert_eq!(state.ra_h.value, Some(5.5));
        assert_eq!(state.ra_h.source, Source::DeviceState);
        assert_eq!(state.side_of_pier.value, Some(PierSide::East));
        assert!(state.ra_h.device_time.is_some());
        // Slow members follow their own schedule.
        s.fake.clear_calls();
        s.clock.advance(Duration::from_secs(1));
        s.drv.poll(&plan(2, false, &[])).unwrap();
        assert_eq!(s.fake.calls_to(Method::Get, "guideraterightascension"), 0);
        s.clock.advance(Duration::from_secs(30));
        s.drv.poll(&plan(3, false, &[])).unwrap();
        assert_eq!(s.fake.calls_to(Method::Get, "guideraterightascension"), 1);
    }

    #[test]
    fn members_missing_from_device_state_are_read_individually() {
        let fake = esp32();
        fake.on_get(
            "devicestate",
            FakeReply::value(json!([
                {"Name": "RightAscension", "Value": 1.0},
                {"Name": "Declination", "Value": 2.0},
                {"Name": "TimeStamp", "Value": "2026-01-01T02:00:00+02:00"},
            ])),
        )
        .on_get("slewing", FakeReply::value(false))
        .on_get("tracking", FakeReply::value(true));
        let (mut s, _) = connected(fake);
        let raw = s.drv.poll(&plan(1, false, &[])).unwrap();
        assert_eq!(
            raw.device_time,
            Some(chrono::Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap())
        );
        assert_eq!(s.fake.calls_to(Method::Get, "slewing"), 1);
        let state = s.drv.state().unwrap();
        assert_eq!(state.slewing.source, Source::Property);
        assert_eq!(state.ra_h.source, Source::DeviceState);
        // A member the server does not know is memoised as unsupported.
        assert_eq!(state.alt_deg.state, ObsState::Unsupported);
        s.fake.clear_calls();
        s.drv.poll(&plan(2, false, &[])).unwrap();
        assert_eq!(s.fake.calls_to(Method::Get, "altitude"), 0);
    }

    #[test]
    fn a_device_state_http_400_falls_back_to_per_member_reads() {
        let fake = esp32();
        let (mut s, _) = connected(fake);
        s.fake
            .on_get(
                "devicestate",
                FakeReply::http(400, "Unknown member devicestate"),
            )
            .on_get("rightascension", FakeReply::value(3.0))
            .on_get("declination", FakeReply::value(4.0));
        s.drv.poll(&plan(1, false, &[])).unwrap();
        assert!(!s.drv.capabilities().unwrap().device_state);
        s.fake.clear_calls();
        let raw = s.drv.poll(&plan(2, false, &[])).unwrap();
        assert_eq!(raw.method, PollMethod::PerMember);
        assert_eq!(s.fake.calls_to(Method::Get, "devicestate"), 0);
        assert_eq!(s.drv.state().unwrap().ra_h.value, Some(3.0));
    }

    #[test]
    fn a_failed_device_state_marks_values_stale_and_fails_the_poll() {
        let (mut s, _) = connected(esp32());
        s.drv.poll(&plan(1, false, &[])).unwrap();
        s.fake.on_get("devicestate", FakeReply::Timeout);
        assert_eq!(
            s.drv.poll(&plan(2, false, &[])).unwrap_err().class,
            ErrorClass::Timeout
        );
        let state = s.drv.state().unwrap();
        assert_eq!(state.ra_h.state, ObsState::Stale);
        assert_eq!(
            state.ra_h.value,
            Some(5.5),
            "the last value is kept, never fresh"
        );
    }

    #[test]
    fn a_silent_platform6_mount_fails_the_poll_without_more_reads() {
        let (mut s, _) = connected(platform6());
        s.drv.poll(&plan(1, false, &[])).unwrap();
        for (member, _) in CORE {
            s.fake.on_get(member, FakeReply::Timeout);
        }
        s.fake.clear_calls();
        s.clock.advance(Duration::from_secs(120));
        let err = s.drv.poll(&plan(2, false, &[])).unwrap_err();
        assert_eq!(err.class, ErrorClass::Transport);
        assert_eq!(s.fake.calls_to(Method::Get, "guideraterightascension"), 0);
        assert_eq!(s.drv.state().unwrap().ra_h.state, ObsState::Stale);
    }

    #[test]
    fn value_not_set_is_unknown_not_an_error() {
        let (mut s, _) = connected(esp32());
        s.drv.poll(&plan(1, true, &[])).unwrap();
        let state = s.drv.state().unwrap();
        assert_eq!(state.target.value, None);
        assert_eq!(state.target.state, ObsState::NotSet);
    }

    #[test]
    fn guide_rates_are_read_even_when_they_cannot_be_set() {
        let (mut s, _) = connected(platform6());
        assert!(!s.drv.capabilities().unwrap().can_set_guide_rates);
        s.drv.poll(&plan(1, false, &[])).unwrap();
        let state = s.drv.state().unwrap();
        assert!((state.guide_rate_ra_arcsec_s.value.unwrap() - 15.041).abs() < 1e-6);
        assert!((state.guide_rate_dec_arcsec_s.value.unwrap() - 7.52).abs() < 1e-6);
        assert!(state.guide_rate_ra_arcsec_s.is_fresh());
        assert_eq!(
            state.tracking_rate.value,
            Some(DriveRate::Solar),
            "ASCOM 2 is Solar"
        );
        assert_eq!(state.side_of_pier.value, Some(PierSide::West));
        let site = state.site.value.unwrap();
        assert_eq!(
            (site.lat_deg, site.lon_deg, site.source),
            (40.0, -75.0, SiteSource::Mount)
        );
        assert_eq!(
            state.utc.value,
            Some(chrono::Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap()),
            "a zoneless UTCDate is UTC"
        );
    }

    #[test]
    fn drive_rates_use_the_ascom_numbers() {
        let (mut s, _) = connected(esp32());
        for (rate, wire) in [
            (DriveRate::Sidereal, "0"),
            (DriveRate::Lunar, "1"),
            (DriveRate::Solar, "2"),
            (DriveRate::King, "3"),
        ] {
            s.fake.clear_calls();
            s.drv.set_tracking_rate(rate).unwrap();
            let call = &s.fake.calls()[0];
            assert_eq!(
                call.params,
                vec![("TrackingRate".to_string(), wire.to_string())]
            );
        }
        let caps = s.drv.capabilities().unwrap();
        assert_eq!(
            caps.tracking_rates,
            vec![
                DriveRate::Sidereal,
                DriveRate::Lunar,
                DriveRate::Solar,
                DriveRate::King
            ]
        );
    }

    #[test]
    fn commands_send_exact_case_parameters() {
        let (mut s, _) = connected(esp32());
        s.fake.clear_calls();
        s.drv
            .slew_async(EqCoord::new(-1.0, -5.25, EqFrame::JNowApparent))
            .unwrap();
        s.drv.pulse_guide(GuideDirection::West, 300).unwrap();
        s.drv.set_tracking(true).unwrap();
        s.drv
            .set_utc(chrono::Utc.with_ymd_and_hms(2026, 3, 1, 4, 5, 6).unwrap())
            .unwrap();
        s.drv.set_guide_rates(0.004, 0.002).unwrap();
        let calls = s.fake.calls();
        let params = |member: &str| {
            calls
                .iter()
                .find(|c| c.member == member)
                .map(|c| c.params.clone())
                .unwrap()
        };
        let pairs = |items: &[(&str, &str)]| {
            items
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            params("slewtocoordinatesasync"),
            pairs(&[("RightAscension", "23"), ("Declination", "-5.25")])
        );
        assert_eq!(
            params("pulseguide"),
            pairs(&[("Direction", "3"), ("Duration", "300")])
        );
        assert_eq!(params("tracking"), pairs(&[("Tracking", "True")]));
        assert_eq!(
            params("utcdate"),
            pairs(&[("UTCDate", "2026-03-01T04:05:06.000Z")])
        );
        assert_eq!(
            params("guideratedeclination"),
            pairs(&[("GuideRateDeclination", "0.002")])
        );
        assert!(calls.iter().all(|c| c.member != "slewtocoordinates"), "O10");
        let slew = calls
            .iter()
            .find(|c| c.member == "slewtocoordinatesasync")
            .unwrap();
        assert_eq!(slew.class, TimeoutClass::Long);
    }

    #[test]
    fn a_slew_completes_only_when_slewing_reads_false() {
        let (mut s, _) = connected(platform6());
        let mut runtime = runtime(&mut s);
        s.fake.queue_get(
            "slewing",
            [
                FakeReply::value(true),
                FakeReply::value(true),
                FakeReply::value(true),
            ],
        );
        s.fake.clear_calls();
        let target = EqCoord::new(5.5, 20.0, EqFrame::J2000);
        assert_eq!(
            run(&mut s, &mut runtime, MountCmd::Slew { target }),
            OpOutcome::Succeeded
        );
        assert!(s.fake.calls_to(Method::Get, "slewing") >= 4);
        assert_eq!(s.fake.calls_to(Method::Put, "slewtocoordinatesasync"), 1);
        let names: Vec<_> = runtime.take_events().iter().map(|e| e.name()).collect();
        assert!(names.contains(&"slew_settled"), "{names:?}");
    }

    #[test]
    fn the_abort_path_uses_its_own_transport_and_stops_jogs() {
        let (mut s, _) = connected(esp32());
        s.drv.move_axis(0, 2.0).unwrap();
        s.fake.clear_calls();
        s.drv.abort_path().unwrap().abort().unwrap();
        assert_eq!(s.abort.calls_to(Method::Put, "abortslew"), 1);
        let moves: Vec<_> = s
            .abort
            .calls()
            .into_iter()
            .filter(|c| c.member == "moveaxis")
            .collect();
        assert_eq!(moves.len(), 1);
        assert_eq!(
            moves[0].params,
            vec![
                ("Axis".to_string(), "0".to_string()),
                ("Rate".to_string(), "0".to_string())
            ]
        );
        assert!(s.fake.calls().is_empty(), "the busy agent is not used");
        // Without a jog, abort is AbortSlew only; parked is not a failure.
        s.abort.on_put(
            "abortslew",
            FakeReply::alpaca_error(
                crate::equipment::types::ascom_error::INVALID_WHILE_PARKED,
                "parked",
            ),
        );
        s.abort.clear_calls();
        assert!(s.drv.abort_path().unwrap().abort().is_ok());
        assert_eq!(s.abort.calls_to(Method::Put, "moveaxis"), 0);
    }

    #[test]
    fn can_slew_false_with_async_supports_goto_jog_and_abort() {
        let (mut s, _) = connected(esp32());
        let mut runtime = runtime(&mut s);
        let target = EqCoord::new(5.5, 20.0, EqFrame::Unknown);
        assert_eq!(
            run(&mut s, &mut runtime, MountCmd::Slew { target }),
            OpOutcome::Succeeded
        );
        assert_eq!(s.fake.calls_to(Method::Put, "slewtocoordinatesasync"), 1);
        let meta = CommandMeta::new(
            CommandId::new(),
            CommandSource::Operator { overriding: false },
        );
        let MountPlan::Start(mut jog) = runtime.plan(
            MountCmd::MoveAxis {
                axis: 0,
                rate_deg_s: 1.0,
            },
            meta,
            None,
        ) else {
            panic!("jog refused");
        };
        jog.start(&mut s.drv, s.clock.as_ref()).unwrap();
        assert_eq!(s.fake.calls_to(Method::Put, "moveaxis"), 1);
        s.drv.abort_path().unwrap().abort().unwrap();
        assert_eq!(s.abort.calls_to(Method::Put, "abortslew"), 1);
        assert_eq!(s.abort.calls_to(Method::Put, "moveaxis"), 1);
        jog.cancel(&mut s.drv).unwrap();
    }

    #[test]
    fn a_short_pulse_reading_false_at_once_succeeds() {
        let (mut s, _) = connected(esp32());
        let mut runtime = runtime(&mut s);
        let outcome = run(
            &mut s,
            &mut runtime,
            MountCmd::PulseGuide {
                dir: GuideDirection::North,
                ms: 30,
            },
        );
        assert_eq!(outcome, OpOutcome::Succeeded);
        assert_eq!(s.fake.calls_to(Method::Put, "pulseguide"), 1);
    }

    #[test]
    fn date_times_parse_permissively() {
        let utc = chrono::Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        assert_eq!(
            parse_alpaca_datetime("2026-01-01T00:00:00Z", true),
            Some(utc)
        );
        assert_eq!(
            parse_alpaca_datetime("2026-01-01T01:00:00+01:00", true),
            Some(utc)
        );
        assert_eq!(
            parse_alpaca_datetime("2026-01-01T00:00:00.0000000", false),
            Some(utc)
        );
        let local = parse_alpaca_datetime("2026-01-01T00:00:00", true).unwrap();
        let expected = Local
            .from_local_datetime(&utc.naive_utc())
            .earliest()
            .unwrap()
            .with_timezone(&Utc);
        assert_eq!(local, expected, "a zoneless TimeStamp is local time");
        assert_eq!(parse_alpaca_datetime("yesterday", true), None);
    }

    #[test]
    fn a_mount_without_site_elevation_keeps_its_site_and_verifies_lat_lon() {
        let fake = esp32();
        fake.on_get("siteelevation", FakeReply::not_implemented())
            .on_put("siteelevation", FakeReply::not_implemented());
        let (mut s, report) = connected(fake);
        assert!(report
            .unsupported_members
            .contains(&"siteelevation".to_string()));
        s.drv.poll(&plan(1, false, &[])).unwrap();
        let state = s.drv.state().unwrap();
        assert_ne!(state.site.state, ObsState::Unsupported);
        let site = state.site.value.expect("latitude and longitude suffice");
        assert_eq!((site.lat_deg, site.lon_deg), (40.0, -75.0));
        assert!(site.elev_m.is_nan(), "the elevation is unknown");
        assert_eq!(
            serde_json::to_value(site).unwrap()["elev_m"],
            Value::Null,
            "published as null"
        );
        // verify_only: a [site] that differs only in elevation matches.
        let configured = Site {
            lat_deg: 40.0,
            lon_deg: -75.0,
            elev_m: 350.0,
            source: SiteSource::Config,
        };
        let mut runtime = MountRuntime::new(
            MountSettings::default(),
            MountLimits::default(),
            Arc::new(IdentityConverter),
            s.clock.clone(),
            s.drv.identity().connection_id,
            Some(configured),
        );
        let post = runtime.on_connected(&mut s.drv).unwrap();
        assert!(!post.site.site_mismatch);
        assert_eq!(post.site.mount_site.map(|m| m.lat_deg), Some(40.0));
        let snapshot = runtime.after_poll(&mut s.drv, None);
        assert!(snapshot.ha_h.is_some(), "geometry keeps working");
        // SetSite writes and verifies latitude and longitude only.
        s.fake
            .on_get("sitelatitude", FakeReply::value(41.0))
            .on_get("sitelongitude", FakeReply::value(-74.0));
        s.fake.clear_calls();
        let target = Site {
            lat_deg: 41.0,
            lon_deg: -74.0,
            elev_m: 120.0,
            source: SiteSource::Operator,
        };
        let outcome = run(&mut s, &mut runtime, MountCmd::SetSite(target));
        assert_eq!(outcome, OpOutcome::Succeeded);
        assert_eq!(s.fake.calls_to(Method::Put, "sitelatitude"), 1);
        assert_eq!(s.fake.calls_to(Method::Put, "sitelongitude"), 1);
        assert_eq!(s.fake.calls_to(Method::Put, "siteelevation"), 0);
        // A mount that keeps a different latitude still fails verification.
        s.fake.on_get("sitelatitude", FakeReply::value(10.0));
        let outcome = run(&mut s, &mut runtime, MountCmd::SetSite(target));
        assert!(matches!(outcome, OpOutcome::Failed(_)), "{outcome:?}");
    }

    #[test]
    fn a_site_elevation_put_that_is_not_implemented_is_not_a_failure() {
        let (mut s, _) = connected(esp32());
        s.fake.on_put("siteelevation", FakeReply::not_implemented());
        let site = Site {
            lat_deg: 41.0,
            lon_deg: -74.0,
            elev_m: 120.0,
            source: SiteSource::Operator,
        };
        s.drv.set_site(&site).unwrap();
        s.drv.set_site(&site).unwrap();
        assert_eq!(
            s.fake.calls_to(Method::Put, "siteelevation"),
            1,
            "memoised after the first NotImplemented"
        );
        assert_eq!(s.fake.calls_to(Method::Put, "sitelatitude"), 2);
    }

    #[test]
    fn a_slow_connecting_read_does_not_stretch_the_connect_timeout() {
        let fake = esp32();
        fake.on_get(
            "connecting",
            FakeReply::Delay(Duration::from_millis(40), Box::new(FakeReply::value(true))),
        );
        let mut s = setup(fake);
        let started = std::time::Instant::now();
        let err = s.drv.connect(Duration::from_millis(100)).unwrap_err();
        let elapsed = started.elapsed();
        assert_eq!(err.class, ErrorClass::Timeout);
        // Counting 100 polls of 1 ms would have taken about 4 s.
        assert!(elapsed < Duration::from_millis(1500), "{elapsed:?}");
        assert!(s.fake.calls_to(Method::Get, "connecting") <= 5);
    }

    #[test]
    fn commands_need_a_connection() {
        let mut s = setup(esp32());
        assert_eq!(s.drv.park().unwrap_err().class, ErrorClass::NotConnected);
        assert_eq!(
            s.drv.poll(&PollPlan::default()).unwrap_err().class,
            ErrorClass::NotConnected
        );
        s.drv.connect(Duration::from_secs(1)).unwrap();
        assert!(s.drv.is_connected().unwrap());
        s.drv.disconnect().unwrap();
        assert_eq!(
            s.fake.calls_to(Method::Put, "disconnect"),
            1,
            "Platform 7 disconnect"
        );
        assert_eq!(s.drv.park().unwrap_err().class, ErrorClass::NotConnected);
    }
}

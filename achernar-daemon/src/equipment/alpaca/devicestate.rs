//! Platform 7 `devicestate` parser (spec §2.7, §2.14). Owner: WP1A-1.
//!
//! One `devicestate` GET returns every operational member of a device as an
//! array of `{"Name": ..., "Value": ...}` entries plus an optional
//! `TimeStamp`. The worker reads it once per tick (§2.7) instead of about ten
//! per-member GETs, which is what makes 4 Hz slew polling affordable on a
//! Wi-Fi ESP32.
//!
//! Rules:
//! - Entry names are matched case-insensitively; the first of two entries
//!   with the same name wins and the duplicate is counted.
//! - An entry without a string `Name` or without a `Value` is skipped and
//!   counted, never fatal. A `null` Value means "known member, no value"
//!   and becomes `ObsState::NotSet`.
//! - A member the device leaves out is simply absent ([`DeviceStateValues::missing`]
//!   lists them against the Platform 7 member list); the driver keeps the
//!   previous value as `Stale` or reads it individually (C-2). Nothing is
//!   synthesised.
//! - `TimeStamp` is parsed permissively: `Z`, an explicit offset (`+02:00`
//!   or `+0200`), or a local time without an offset, with up to nine
//!   fractional digits (.NET writes seven) and `T` or a space as separator.
//!   A local time is read in the host's zone, and the result records which
//!   form it came from so a consumer can discount an ambiguous clock. An
//!   unparsable TimeStamp leaves the device time unknown.

use chrono::{DateTime, FixedOffset, Local, NaiveDateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::errors::{expect_bool, expect_f64, expect_i64, expect_string, expect_utc_datetime};
use crate::equipment::traits::{PollMethod, RawMember, RawPoll};
use crate::equipment::types::{DeviceError, DeviceResult, ObsState};

/// Which form a DeviceState `TimeStamp` had.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeStampForm {
    /// Ends in `Z`.
    Utc,
    /// Carries an explicit UTC offset.
    Offset,
    /// No offset: read in the host's local zone (or the zone a caller gave).
    Local,
}

/// A parsed device time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceTimeStamp {
    pub utc: DateTime<Utc>,
    pub form: TimeStampForm,
}

/// One DeviceState entry, in the device's own spelling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceStateEntry {
    pub name: String,
    pub value: Value,
}

/// The parsed `devicestate` Value.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct DeviceStateValues {
    /// Operational members in device order, `TimeStamp` excluded.
    pub entries: Vec<DeviceStateEntry>,
    pub timestamp: Option<DeviceTimeStamp>,
    /// The raw TimeStamp when one was present but could not be parsed.
    pub unparsed_timestamp: Option<String>,
    /// Entries skipped because they had no string `Name` or no `Value`.
    pub malformed_entries: usize,
    /// Entries skipped because an earlier entry had the same name.
    pub duplicate_entries: usize,
}

/// Parse a `devicestate` Value, reading an offset-less TimeStamp in the
/// host's local zone.
pub fn parse_device_state(value: &Value) -> DeviceResult<DeviceStateValues> {
    parse_entries(value, &|text| parse_timestamp(text))
}

/// As [`parse_device_state`], reading an offset-less TimeStamp in `local`.
pub fn parse_device_state_with(
    value: &Value,
    local: FixedOffset,
) -> DeviceResult<DeviceStateValues> {
    parse_entries(value, &|text| parse_timestamp_with(text, local))
}

fn parse_entries(
    value: &Value,
    parse_stamp: &dyn Fn(&str) -> Option<DeviceTimeStamp>,
) -> DeviceResult<DeviceStateValues> {
    let items = value.as_array().ok_or_else(|| {
        DeviceError::driver(format!(
            "devicestate Value is not an array: {}",
            short(&value.to_string())
        ))
    })?;
    let mut out = DeviceStateValues::default();
    let mut seen_timestamp = false;
    for item in items {
        let Some(object) = item.as_object() else {
            out.malformed_entries += 1;
            continue;
        };
        let name = super::errors::field(object, "Name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty());
        let entry_value = super::errors::field(object, "Value");
        let (Some(name), Some(entry_value)) = (name, entry_value) else {
            out.malformed_entries += 1;
            continue;
        };
        if name.eq_ignore_ascii_case("TimeStamp") {
            if seen_timestamp {
                out.duplicate_entries += 1;
                continue;
            }
            seen_timestamp = true;
            match entry_value.as_str().and_then(parse_stamp) {
                Some(stamp) => out.timestamp = Some(stamp),
                None if entry_value.is_null() => {}
                None => {
                    let raw = entry_value
                        .as_str()
                        .map_or_else(|| entry_value.to_string(), str::to_string);
                    out.unparsed_timestamp = Some(short(&raw));
                }
            }
            continue;
        }
        if out.contains(name) {
            out.duplicate_entries += 1;
            continue;
        }
        out.entries.push(DeviceStateEntry {
            name: name.to_string(),
            value: entry_value.clone(),
        });
    }
    Ok(out)
}

fn short(text: &str) -> String {
    if text.chars().count() <= 200 {
        text.to_string()
    } else {
        let mut out: String = text.chars().take(200).collect();
        out.push_str("...");
        out
    }
}

impl DeviceStateValues {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when the device returned no operational members, which counts
    /// as "no DeviceState" for the poll plan (§2.6 step 4).
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|entry| entry.name.as_str())
    }

    /// The raw value of a member, matched case-insensitively.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.entries
            .iter()
            .find(|entry| entry.name.eq_ignore_ascii_case(name))
            .map(|entry| &entry.value)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// A typed member: `None` when absent or `null`, `Some(Err)` when the
    /// device sent the wrong type.
    pub fn typed<T>(
        &self,
        name: &str,
        parse: impl Fn(&str, &Value) -> DeviceResult<T>,
    ) -> Option<DeviceResult<T>> {
        self.get(name)
            .filter(|value| !value.is_null())
            .map(|value| parse(name, value))
    }

    pub fn f64(&self, name: &str) -> Option<DeviceResult<f64>> {
        self.typed(name, expect_f64)
    }

    pub fn bool(&self, name: &str) -> Option<DeviceResult<bool>> {
        self.typed(name, expect_bool)
    }

    pub fn i64(&self, name: &str) -> Option<DeviceResult<i64>> {
        self.typed(name, expect_i64)
    }

    pub fn string(&self, name: &str) -> Option<DeviceResult<String>> {
        self.typed(name, expect_string)
    }

    /// A UTC date-time member such as `UTCDate`.
    pub fn utc(&self, name: &str) -> Option<DeviceResult<DateTime<Utc>>> {
        self.typed(name, expect_utc_datetime)
    }

    /// The members of `expected` the device left out.
    pub fn missing<'a>(&self, expected: &[&'a str]) -> Vec<&'a str> {
        expected
            .iter()
            .copied()
            .filter(|name| !self.contains(name))
            .collect()
    }

    /// The entries as poll members: lower-case names, `Fresh` values, and
    /// `NotSet` for a `null` value. Every member shares the request's
    /// midpoint time and latency.
    pub fn to_raw_members(
        &self,
        observed_at: DateTime<Utc>,
        mono_ns: u64,
        latency_ms: u32,
    ) -> Vec<RawMember> {
        self.entries
            .iter()
            .map(|entry| RawMember {
                name: entry.name.to_ascii_lowercase(),
                value: (!entry.value.is_null()).then(|| entry.value.clone()),
                state: if entry.value.is_null() {
                    ObsState::NotSet
                } else {
                    ObsState::Fresh
                },
                observed_at: Some(observed_at),
                mono_ns: Some(mono_ns),
                latency_ms: Some(latency_ms),
                error: None,
            })
            .collect()
    }

    /// A [`RawPoll`] for one DeviceState GET that started and finished at
    /// the given monotonic times.
    pub fn to_raw_poll(
        &self,
        started_mono_ns: u64,
        finished_mono_ns: u64,
        observed_at: DateTime<Utc>,
    ) -> RawPoll {
        let latency_ms =
            u32::try_from(finished_mono_ns.saturating_sub(started_mono_ns) / 1_000_000)
                .unwrap_or(u32::MAX);
        let mid = started_mono_ns + finished_mono_ns.saturating_sub(started_mono_ns) / 2;
        RawPoll {
            method: PollMethod::DeviceState,
            started_mono_ns,
            finished_mono_ns,
            device_time: self.timestamp.map(|stamp| stamp.utc),
            members: self.to_raw_members(observed_at, mid, latency_ms),
        }
    }
}

/// The Platform 7 DeviceState members per Alpaca device type (TimeStamp
/// excluded). Switch members are per channel (`GetSwitch0`, ...), so its
/// list is empty.
pub fn expected_members(device_type: &str) -> &'static [&'static str] {
    match device_type.to_ascii_lowercase().as_str() {
        "telescope" => &[
            "Altitude",
            "AtHome",
            "AtPark",
            "Azimuth",
            "Declination",
            "IsPulseGuiding",
            "RightAscension",
            "SideOfPier",
            "SiderealTime",
            "Slewing",
            "Tracking",
            "UTCDate",
        ],
        "focuser" => &["IsMoving", "Position", "Temperature"],
        "filterwheel" => &["Position"],
        "rotator" => &["IsMoving", "MechanicalPosition", "Position"],
        "covercalibrator" => &[
            "Brightness",
            "CalibratorChanging",
            "CalibratorState",
            "CoverMoving",
            "CoverState",
        ],
        "observingconditions" => &[
            "CloudCover",
            "DewPoint",
            "Humidity",
            "Pressure",
            "RainRate",
            "SkyBrightness",
            "SkyQuality",
            "SkyTemperature",
            "StarFWHM",
            "Temperature",
            "WindDirection",
            "WindGust",
            "WindSpeed",
        ],
        "safetymonitor" => &["IsSafe"],
        "dome" => &[
            "Altitude",
            "AtHome",
            "AtPark",
            "Azimuth",
            "ShutterStatus",
            "Slewing",
        ],
        _ => &[],
    }
}

// ---------------------------------------------------------------------------
// TimeStamp parsing
// ---------------------------------------------------------------------------

/// Parse a device time, reading an offset-less time in the host's zone.
pub fn parse_timestamp(text: &str) -> Option<DeviceTimeStamp> {
    parse_with(text, &|naive| {
        Local
            .from_local_datetime(&naive)
            .earliest()
            .map(|local| local.with_timezone(&Utc))
    })
}

/// Parse a device time, reading an offset-less time in `local`.
pub fn parse_timestamp_with(text: &str, local: FixedOffset) -> Option<DeviceTimeStamp> {
    parse_with(text, &|naive| {
        local
            .from_local_datetime(&naive)
            .single()
            .map(|at| at.with_timezone(&Utc))
    })
}

const NAIVE_FORMATS: [&str; 4] = [
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
];

fn parse_with(
    text: &str,
    local: &dyn Fn(NaiveDateTime) -> Option<DateTime<Utc>>,
) -> Option<DeviceTimeStamp> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    if let Some(body) = text.strip_suffix(['Z', 'z']) {
        let naive = parse_naive(body)?;
        return Some(DeviceTimeStamp {
            utc: Utc.from_utc_datetime(&naive),
            form: TimeStampForm::Utc,
        });
    }
    if let Some((body, offset)) = split_offset(text) {
        let naive = parse_naive(body)?;
        let at = offset.from_local_datetime(&naive).single()?;
        return Some(DeviceTimeStamp {
            utc: at.with_timezone(&Utc),
            form: TimeStampForm::Offset,
        });
    }
    let naive = parse_naive(text)?;
    Some(DeviceTimeStamp {
        utc: local(naive)?,
        form: TimeStampForm::Local,
    })
}

fn parse_naive(text: &str) -> Option<NaiveDateTime> {
    // Accept a lower-case `t` separator as RFC 3339 does.
    let text = text.replacen('t', "T", 1);
    NAIVE_FORMATS
        .iter()
        .find_map(|format| NaiveDateTime::parse_from_str(&text, format).ok())
}

/// Split a trailing `+hh:mm`, `-hh:mm`, `+hhmm` or `+hh` offset that follows
/// the time part (after the `T` or space), so the date's own `-` signs are
/// never mistaken for one.
fn split_offset(text: &str) -> Option<(&str, FixedOffset)> {
    let time_start = text.find(['T', 't', ' '])? + 1;
    let sign_at = time_start + text[time_start..].rfind(['+', '-'])?;
    let (body, offset) = text.split_at(sign_at);
    let sign = if offset.starts_with('-') { -1 } else { 1 };
    let digits: String = offset[1..].chars().filter(|c| *c != ':').collect();
    if !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let (hours, minutes) = match digits.len() {
        2 => (digits.parse::<i32>().ok()?, 0),
        4 => (
            digits[..2].parse::<i32>().ok()?,
            digits[2..].parse::<i32>().ok()?,
        ),
        _ => return None,
    };
    if hours > 23 || minutes > 59 {
        return None;
    }
    let offset = FixedOffset::east_opt(sign * (hours * 3600 + minutes * 60))?;
    Some((body, offset))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn minus_five() -> FixedOffset {
        FixedOffset::west_opt(5 * 3600).unwrap()
    }

    fn rfc(stamp: &DeviceTimeStamp) -> String {
        stamp
            .utc
            .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
    }

    #[test]
    fn the_three_timestamp_forms_parse_to_utc() {
        let z = parse_timestamp_with("2026-10-26T21:30:00.1234567Z", minus_five()).unwrap();
        assert_eq!(z.form, TimeStampForm::Utc);
        assert_eq!(rfc(&z), "2026-10-26T21:30:00.123Z");

        let offset = parse_timestamp_with("2026-10-26T23:30:00.5+02:00", minus_five()).unwrap();
        assert_eq!(offset.form, TimeStampForm::Offset);
        assert_eq!(rfc(&offset), "2026-10-26T21:30:00.500Z");

        let local = parse_timestamp_with("2026-10-26T16:30:00", minus_five()).unwrap();
        assert_eq!(local.form, TimeStampForm::Local);
        assert_eq!(rfc(&local), "2026-10-26T21:30:00.000Z");
    }

    #[test]
    fn timestamp_variants_are_tolerated_and_garbage_is_unknown() {
        let utc = |text: &str| parse_timestamp_with(text, minus_five()).map(|s| rfc(&s));
        assert_eq!(
            utc("2026-10-26t21:30:00z").as_deref(),
            Some("2026-10-26T21:30:00.000Z")
        );
        assert_eq!(
            utc("2026-10-26 21:30:00Z").as_deref(),
            Some("2026-10-26T21:30:00.000Z")
        );
        assert_eq!(
            utc("2026-10-26T16:30:00-0500").as_deref(),
            Some("2026-10-26T21:30:00.000Z")
        );
        assert_eq!(
            utc("2026-10-26T23:30:00+02").as_deref(),
            Some("2026-10-26T21:30:00.000Z")
        );
        assert_eq!(
            utc("2026-10-26T16:30:00.123456789").as_deref(),
            Some("2026-10-26T21:30:00.123Z")
        );
        assert_eq!(
            utc("  2026-10-26T21:30Z ").as_deref(),
            Some("2026-10-26T21:30:00.000Z")
        );
        for bad in [
            "",
            "now",
            "2026-13-01T00:00:00Z",
            "2026-10-26T21:30:00+25:00",
            "26/10/2026 21:30",
            "2026-10-26",
        ] {
            assert_eq!(utc(bad), None, "{bad}");
        }
        // The host-zone variant parses the same three forms.
        assert_eq!(
            parse_timestamp("2026-10-26T21:30:00Z").unwrap().form,
            TimeStampForm::Utc
        );
        assert_eq!(
            parse_timestamp("2026-10-26T21:30:00").unwrap().form,
            TimeStampForm::Local
        );
    }

    fn telescope_state(skip: &[&str]) -> Value {
        let all = [
            ("Altitude", json!(45.0)),
            ("AtHome", json!(false)),
            ("AtPark", json!(false)),
            ("Azimuth", json!(180.5)),
            ("Declination", json!(-5.25)),
            ("IsPulseGuiding", json!(false)),
            ("RightAscension", json!(5.5)),
            ("SideOfPier", json!(0)),
            ("SiderealTime", json!(3.25)),
            ("Slewing", json!(true)),
            ("Tracking", json!(true)),
            ("UTCDate", json!("2026-10-26T21:30:00.000Z")),
            ("TimeStamp", json!("2026-10-26T21:30:00.250Z")),
        ];
        Value::Array(
            all.into_iter()
                .filter(|(name, _)| !skip.contains(name))
                .map(|(name, value)| json!({"Name": name, "Value": value}))
                .collect(),
        )
    }

    #[test]
    fn a_complete_telescope_state_parses_with_its_timestamp() {
        let state = parse_device_state(&telescope_state(&[])).unwrap();
        assert_eq!(state.len(), 12);
        assert!(state.missing(expected_members("telescope")).is_empty());
        assert_eq!(state.f64("rightascension").unwrap().unwrap(), 5.5);
        assert!(state.bool("SLEWING").unwrap().unwrap());
        assert_eq!(state.i64("SideOfPier").unwrap().unwrap(), 0);
        assert_eq!(
            state.utc("UTCDate").unwrap().unwrap().to_rfc3339(),
            "2026-10-26T21:30:00+00:00"
        );
        let stamp = state.timestamp.unwrap();
        assert_eq!(stamp.form, TimeStampForm::Utc);
        assert_eq!(rfc(&stamp), "2026-10-26T21:30:00.250Z");
        assert!(
            !state.contains("TimeStamp"),
            "TimeStamp is not an operational member"
        );
    }

    #[test]
    fn missing_null_malformed_and_duplicate_entries_are_reported_not_invented() {
        let mut value = telescope_state(&["Slewing", "SideOfPier", "TimeStamp"]);
        let items = value.as_array_mut().unwrap();
        items.push(json!({"Name": "Temperature", "Value": null}));
        items.push(json!({"Name": "RightAscension", "Value": 23.0}));
        items.push(json!({"Value": 1}));
        items.push(json!({"Name": "", "Value": 1}));
        items.push(json!({"Name": "NoValue"}));
        items.push(json!("not an object"));
        let state = parse_device_state_with(&value, minus_five()).unwrap();
        assert_eq!(
            state.missing(expected_members("telescope")),
            vec!["SideOfPier", "Slewing"]
        );
        assert_eq!(state.timestamp, None, "no TimeStamp means no device time");
        assert_eq!(state.malformed_entries, 4);
        assert_eq!(state.duplicate_entries, 1);
        assert_eq!(
            state.f64("RightAscension").unwrap().unwrap(),
            5.5,
            "first entry wins"
        );
        assert!(state.contains("Temperature"));
        assert!(state.f64("Temperature").is_none(), "null is not a value");
        assert!(state.f64("Slewing").is_none());
        let err = state.bool("Altitude").unwrap().unwrap_err();
        assert!(err.message.contains("Altitude"));

        let at = Utc::now();
        let members = state.to_raw_members(at, 42, 17);
        let temperature = members.iter().find(|m| m.name == "temperature").unwrap();
        assert_eq!(temperature.state, ObsState::NotSet);
        assert_eq!(temperature.value, None);
        let ra = members.iter().find(|m| m.name == "rightascension").unwrap();
        assert_eq!(ra.state, ObsState::Fresh);
        assert_eq!(ra.value, Some(json!(5.5)));
        assert_eq!((ra.mono_ns, ra.latency_ms), (Some(42), Some(17)));
    }

    #[test]
    fn an_unparsable_or_duplicated_timestamp_leaves_device_time_unknown() {
        let value = json!([
            {"Name": "Position", "Value": 3},
            {"Name": "TimeStamp", "Value": "about now"},
            {"Name": "timestamp", "Value": "2026-10-26T21:30:00Z"},
        ]);
        let state = parse_device_state(&value).unwrap();
        assert_eq!(state.timestamp, None);
        assert_eq!(state.unparsed_timestamp.as_deref(), Some("about now"));
        assert_eq!(state.duplicate_entries, 1);
        let poll = state.to_raw_poll(1_000_000, 41_000_000, Utc::now());
        assert_eq!(poll.method, PollMethod::DeviceState);
        assert_eq!(poll.device_time, None);
        assert_eq!(poll.members[0].latency_ms, Some(40));
        assert_eq!(
            poll.members[0].mono_ns,
            Some(21_000_000),
            "midpoint of the request"
        );
    }

    #[test]
    fn a_non_array_value_is_a_driver_error_and_an_empty_array_is_empty() {
        let err = parse_device_state(&json!({"Position": 3})).unwrap_err();
        assert_eq!(err.class, crate::equipment::types::ErrorClass::Driver);
        let empty = parse_device_state(&json!([])).unwrap();
        assert!(empty.is_empty());
        let only_stamp =
            parse_device_state(&json!([{"Name": "TimeStamp", "Value": "2026-10-26T21:30:00Z"}]))
                .unwrap();
        assert!(
            only_stamp.is_empty(),
            "a TimeStamp alone is not a usable DeviceState"
        );
        assert!(only_stamp.timestamp.is_some());
    }

    #[test]
    fn expected_member_lists_cover_every_platform7_type() {
        for device_type in [
            "telescope",
            "focuser",
            "filterwheel",
            "rotator",
            "covercalibrator",
            "observingconditions",
            "safetymonitor",
            "dome",
        ] {
            assert!(!expected_members(device_type).is_empty(), "{device_type}");
        }
        assert_eq!(
            expected_members("Focuser"),
            &["IsMoving", "Position", "Temperature"]
        );
        assert!(expected_members("switch").is_empty());
    }
}

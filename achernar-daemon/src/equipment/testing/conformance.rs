//! Generic per-trait conformance suites (spec §5.6, §5.7 Tier 2 item 1).
//! Owner: WP1A-2.
//!
//! Every driver of a kind (Alpaca, INDI, simulator, OLS legacy) must pass the
//! same suite, so a model sees one contract whatever the transport:
//!
//! ```ignore
//! let mut drv = SimFocuser::new(...);
//! assert_conforms("sim focuser", &focuser_suite(&mut drv, &clock));
//! ```
//!
//! Suites return [`Violation`]s instead of panicking, so a driver package
//! can assert that the list is empty, or that a deliberately broken fixture
//! is caught. Each violation names a rule:
//!
//! - `fresh_has_value_and_time`, `unknown_is_never_fresh`,
//!   `no_value_without_reading`, `stale_keeps_last_value`,
//!   `timestamps_not_in_the_future`, `nan_is_unknown`: the `Observed<T>`
//!   contract (D5: never a synthesised 0 or "safe");
//! - `identity_kind`, `identity_key`, `connection_id`,
//!   `only_rig_acceptance_qualifies` (D21);
//! - `connect`, `is_connected`, `poll_timestamps`, `failed_member_has_no_value`,
//!   `disconnect`;
//! - `motion_devices_have_an_abort_path` (D4), `unknown_action_is_unsupported`;
//! - per kind: ranges, capability consistency, `abort_slew_never_gated` (O10),
//!   `move_mode_matches_absolute`, `sync_only_when_supported`,
//!   `slots_contiguous`, and so on.
//!
//! The suites only read and issue harmless commands (halt, abort); they never
//! slew, move or park, so they also run against simulators of real rigs.

use std::fmt::Debug;
use std::time::Duration;

use uuid::Uuid;

use crate::equipment::ops::Operation;
use crate::equipment::traits::{
    Clock, CoverCalibratorDriver, Device, DomeDriver, FilterWheelDriver, FocuserDriver,
    MountDriver, PollPlan, RotatorDriver, SafetyDriver, SwitchDriver, WeatherDriver,
};
use crate::equipment::types::{
    kind_name, DeviceKind, DeviceResult, ErrorClass, ObsState, Observed, Qualification,
};

/// One broken rule.
#[derive(Debug, Clone, PartialEq)]
pub struct Violation {
    pub rule: &'static str,
    pub detail: String,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.rule, self.detail)
    }
}

fn violation(out: &mut Vec<Violation>, rule: &'static str, detail: impl Into<String>) {
    out.push(Violation {
        rule,
        detail: detail.into(),
    });
}

/// Panic with every violation listed.
pub fn assert_conforms(driver: &str, violations: &[Violation]) {
    if !violations.is_empty() {
        let list: Vec<String> = violations.iter().map(ToString::to_string).collect();
        panic!(
            "{driver} breaks {} conformance rule(s):\n  {}",
            violations.len(),
            list.join("\n  ")
        );
    }
}

/// True when `rule` was broken.
pub fn broke(violations: &[Violation], rule: &str) -> bool {
    violations.iter().any(|v| v.rule == rule)
}

// ---------------------------------------------------------------------------
// The Observed contract
// ---------------------------------------------------------------------------

/// Check one observed value (D5).
pub fn check_observed<T: Debug>(
    field: &str,
    value: &Observed<T>,
    now_mono_ns: u64,
    out: &mut Vec<Violation>,
) {
    match value.state {
        ObsState::Fresh => {
            if value.value.is_none() {
                violation(
                    out,
                    "unknown_is_never_fresh",
                    format!("{field} is Fresh with no value"),
                );
            }
            if value.observed_at.is_none() || value.mono_ns.is_none() {
                violation(
                    out,
                    "fresh_has_value_and_time",
                    format!("{field} is Fresh without observed_at/mono_ns"),
                );
            }
        }
        ObsState::Stale => {
            if value.value.is_none() {
                violation(
                    out,
                    "stale_keeps_last_value",
                    format!("{field} is Stale with no value (that is Error)"),
                );
            }
        }
        ObsState::Unsupported | ObsState::NotSet | ObsState::Error => {
            if value.value.is_some() {
                violation(
                    out,
                    "no_value_without_reading",
                    format!(
                        "{field} is {:?} but carries {:?} (a synthesised value)",
                        value.state, value.value
                    ),
                );
            }
        }
    }
    if value.mono_ns.is_some_and(|mono| mono > now_mono_ns) {
        violation(
            out,
            "timestamps_not_in_the_future",
            format!("{field} mono_ns {:?} > now {now_mono_ns}", value.mono_ns),
        );
    }
}

/// [`check_observed`] plus: a NaN or infinite reading must be unknown.
pub fn check_observed_f64(
    field: &str,
    value: &Observed<f64>,
    now_mono_ns: u64,
    out: &mut Vec<Violation>,
) {
    check_observed(field, value, now_mono_ns, out);
    if value.value.is_some_and(|v| !v.is_finite()) {
        violation(
            out,
            "nan_is_unknown",
            format!("{field} carries a non-finite value; it must be None"),
        );
    }
}

fn check_range(
    field: &str,
    value: &Observed<f64>,
    range: std::ops::Range<f64>,
    inclusive_end: bool,
    out: &mut Vec<Violation>,
) {
    if let Some(v) = value.value.filter(|v| v.is_finite()) {
        let inside = v >= range.start && (v < range.end || (inclusive_end && v == range.end));
        if !inside {
            violation(
                out,
                "value_in_range",
                format!("{field} = {v} is outside {range:?}"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Common device suite
// ---------------------------------------------------------------------------

fn motion_kind(kind: DeviceKind) -> bool {
    matches!(
        kind,
        DeviceKind::Mount
            | DeviceKind::Focuser
            | DeviceKind::Rotator
            | DeviceKind::CoverCalibrator
            | DeviceKind::Dome
    )
}

/// Identity, connect, poll and the abort path. Leaves the device connected.
pub fn device_suite<D: Device + ?Sized>(
    drv: &mut D,
    kind: DeviceKind,
    clock: &dyn Clock,
) -> Vec<Violation> {
    let mut out = Vec::new();
    {
        let identity = drv.identity();
        if identity.kind != kind {
            violation(
                &mut out,
                "identity_kind",
                format!("identity says {:?}, expected {kind:?}", identity.kind),
            );
        }
        if identity.identity_key.trim().is_empty() {
            violation(&mut out, "identity_key", "the identity key is empty");
        }
        if identity.connection_id == Uuid::nil() {
            violation(&mut out, "connection_id", "the connection id is nil");
        }
        if matches!(identity.qualification, Qualification::Qualified { .. }) {
            violation(
                &mut out,
                "only_rig_acceptance_qualifies",
                "a driver may report planned or implemented_unqualified only (D21)",
            );
        }
    }
    if let Err(err) = drv.connect(Duration::from_secs(5)) {
        violation(&mut out, "connect", format!("connect failed: {err}"));
        return out;
    }
    match drv.is_connected() {
        Ok(true) => {}
        other => violation(
            &mut out,
            "is_connected",
            format!("is_connected after connect returned {other:?}"),
        ),
    }
    let plan = PollPlan {
        tick: 1,
        interval: Duration::from_secs(1),
        prefer_device_state: true,
        ..PollPlan::default()
    };
    match drv.poll(&plan) {
        Ok(raw) => {
            if raw.finished_mono_ns < raw.started_mono_ns {
                violation(
                    &mut out,
                    "poll_timestamps",
                    "the poll finished before it started",
                );
            }
            if raw.finished_mono_ns > clock.now_mono_ns() {
                violation(
                    &mut out,
                    "timestamps_not_in_the_future",
                    "the poll finished in the future",
                );
            }
            for member in &raw.members {
                if member.state == ObsState::Error && member.value.is_some() {
                    violation(
                        &mut out,
                        "failed_member_has_no_value",
                        format!("member {} failed but carries a value", member.name),
                    );
                }
            }
        }
        Err(err) => violation(&mut out, "poll", format!("poll failed: {err}")),
    }
    if motion_kind(kind) && drv.abort_path().is_none() {
        violation(
            &mut out,
            "motion_devices_have_an_abort_path",
            format!(
                "a {} driver needs an out-of-band abort path (D4)",
                kind_name(kind)
            ),
        );
    }
    match drv.action("__achernar_conformance_unknown_action__", "") {
        Err(err) if err.class == ErrorClass::Unsupported => {}
        other => violation(
            &mut out,
            "unknown_action_is_unsupported",
            format!("an unknown action returned {other:?}"),
        ),
    }
    if let Err(err) = drv.supported_actions() {
        if err.class != ErrorClass::Unsupported {
            violation(
                &mut out,
                "supported_actions",
                format!("supportedactions failed: {err}"),
            );
        }
    }
    out
}

/// Disconnect, then the device must say so. Run last.
pub fn disconnect_suite<D: Device + ?Sized>(drv: &mut D) -> Vec<Violation> {
    let mut out = Vec::new();
    if let Err(err) = drv.disconnect() {
        violation(&mut out, "disconnect", format!("disconnect failed: {err}"));
    }
    match drv.is_connected() {
        Ok(false) => {}
        Err(err)
            if matches!(
                err.class,
                ErrorClass::NotConnected | ErrorClass::Disconnected
            ) => {}
        other => violation(
            &mut out,
            "disconnect",
            format!("is_connected after disconnect returned {other:?}"),
        ),
    }
    out
}

fn poll_once<D: Device + ?Sized>(drv: &mut D, out: &mut Vec<Violation>) {
    let plan = PollPlan {
        tick: 2,
        interval: Duration::from_secs(1),
        prefer_device_state: true,
        ..PollPlan::default()
    };
    if let Err(err) = drv.poll(&plan) {
        violation(out, "poll", format!("poll failed: {err}"));
    }
}

fn halt_ok(name: &str, result: DeviceResult<()>, out: &mut Vec<Violation>) {
    if let Err(err) = result {
        violation(
            out,
            "halt_always_available",
            format!("{name} failed while idle: {err}"),
        );
    }
}

// ---------------------------------------------------------------------------
// Per-kind suites
// ---------------------------------------------------------------------------

/// Mount (R1-R11, O10).
pub fn mount_suite<D: MountDriver + ?Sized>(drv: &mut D, clock: &dyn Clock) -> Vec<Violation> {
    let mut out = device_suite(drv, DeviceKind::Mount, clock);
    poll_once(drv, &mut out);
    match drv.capabilities() {
        Ok(caps) => {
            if caps.native_frame == crate::equipment::types::EqFrame::Unknown
                && caps.frame_reason.is_none()
            {
                violation(
                    &mut out,
                    "unknown_frame_has_a_reason",
                    "native_frame is unknown without a frame_reason",
                );
            }
            for axis in 0..3u8 {
                if caps.can_move_axis[usize::from(axis)]
                    && !caps.axis_rates.iter().any(|r| r.axis == axis)
                {
                    violation(
                        &mut out,
                        "moveaxis_has_rates",
                        format!("CanMoveAxis({axis}) without AxisRates"),
                    );
                }
            }
        }
        Err(err) => violation(
            &mut out,
            "capabilities",
            format!("capabilities failed: {err}"),
        ),
    }
    let now = clock.now_mono_ns();
    match drv.state() {
        Ok(state) => {
            check_observed_f64("ra_h", &state.ra_h, now, &mut out);
            check_observed_f64("dec_deg", &state.dec_deg, now, &mut out);
            check_observed_f64("alt_deg", &state.alt_deg, now, &mut out);
            check_observed_f64("az_deg", &state.az_deg, now, &mut out);
            check_observed_f64("sidereal_time_h", &state.sidereal_time_h, now, &mut out);
            check_observed("utc", &state.utc, now, &mut out);
            check_observed("slewing", &state.slewing, now, &mut out);
            check_observed("tracking", &state.tracking, now, &mut out);
            check_observed("tracking_rate", &state.tracking_rate, now, &mut out);
            check_observed("at_park", &state.at_park, now, &mut out);
            check_observed("at_home", &state.at_home, now, &mut out);
            check_observed("side_of_pier", &state.side_of_pier, now, &mut out);
            check_observed("is_pulse_guiding", &state.is_pulse_guiding, now, &mut out);
            check_observed_f64(
                "guide_rate_ra",
                &state.guide_rate_ra_arcsec_s,
                now,
                &mut out,
            );
            check_observed_f64(
                "guide_rate_dec",
                &state.guide_rate_dec_arcsec_s,
                now,
                &mut out,
            );
            check_observed("site", &state.site, now, &mut out);
            check_observed("target", &state.target, now, &mut out);
            check_range("ra_h", &state.ra_h, 0.0..24.0, false, &mut out);
            check_range("dec_deg", &state.dec_deg, -90.0..90.0, true, &mut out);
            check_range("alt_deg", &state.alt_deg, -90.0..90.0, true, &mut out);
            check_range("az_deg", &state.az_deg, 0.0..360.0, false, &mut out);
            check_range(
                "sidereal_time_h",
                &state.sidereal_time_h,
                0.0..24.0,
                false,
                &mut out,
            );
            if state.side_of_pier.value == Some(crate::equipment::types::PierSide::Unknown) {
                violation(
                    &mut out,
                    "unknown_is_never_fresh",
                    "side_of_pier carries Some(Unknown); use None for unknown",
                );
            }
        }
        Err(err) => violation(&mut out, "state", format!("state failed: {err}")),
    }
    // O10: AbortSlew is never gated on a Can* flag.
    if let Err(err) = drv.abort_slew() {
        if err.class == ErrorClass::Unsupported {
            violation(
                &mut out,
                "abort_slew_never_gated",
                format!("abort_slew returned Unsupported: {err}"),
            );
        }
    }
    out
}

/// Focuser (R12-R16).
pub fn focuser_suite<D: FocuserDriver + ?Sized>(drv: &mut D, clock: &dyn Clock) -> Vec<Violation> {
    let mut out = device_suite(drv, DeviceKind::Focuser, clock);
    poll_once(drv, &mut out);
    let caps = match drv.capabilities() {
        Ok(caps) => caps,
        Err(err) => {
            violation(
                &mut out,
                "capabilities",
                format!("capabilities failed: {err}"),
            );
            return out;
        }
    };
    if caps.max_step.is_some_and(|m| m <= 0) {
        violation(&mut out, "max_step_positive", "max_step must be positive");
    }
    if let (Some(inc), Some(max)) = (caps.max_increment, caps.max_step) {
        if inc <= 0 || inc > max {
            violation(
                &mut out,
                "max_increment_within_max_step",
                format!("max_increment {inc} outside 1..={max}"),
            );
        }
    }
    if caps
        .step_size_um
        .is_some_and(|s| !(s.is_finite() && s > 0.0))
    {
        violation(&mut out, "step_size_positive", "step_size_um must be > 0");
    }
    let now = clock.now_mono_ns();
    match drv.state() {
        Ok(state) => {
            check_observed("position", &state.position, now, &mut out);
            check_observed("is_moving", &state.is_moving, now, &mut out);
            check_observed_f64("temperature_c", &state.temperature_c, now, &mut out);
            check_observed("temp_comp", &state.temp_comp, now, &mut out);
            if caps.absolute && state.position.value.is_none() {
                violation(
                    &mut out,
                    "absolute_reports_position",
                    "an absolute focuser must report its position after a poll",
                );
            }
        }
        Err(err) => violation(&mut out, "state", format!("state failed: {err}")),
    }
    // The wrong move mode must be refused, never silently emulated.
    let wrong = if caps.absolute {
        drv.move_relative(0)
    } else {
        drv.move_absolute(0)
    };
    if wrong.is_ok() {
        violation(
            &mut out,
            "move_mode_matches_absolute",
            format!(
                "an {} focuser accepted the other move mode",
                if caps.absolute {
                    "absolute"
                } else {
                    "relative"
                }
            ),
        );
    }
    if !caps.can_sync {
        match drv.sync_position(0) {
            Err(err) if err.class == ErrorClass::Unsupported => {}
            other => violation(
                &mut out,
                "sync_only_when_supported",
                format!("sync_position without can_sync returned {other:?}"),
            ),
        }
    }
    if !caps.temp_comp_available && drv.set_temp_comp(true).is_ok() {
        violation(
            &mut out,
            "tempcomp_only_when_available",
            "set_temp_comp succeeded without temp_comp_available",
        );
    }
    halt_ok("halt", drv.halt(), &mut out);
    out
}

/// Filter wheel (R17).
pub fn filter_wheel_suite<D: FilterWheelDriver + ?Sized>(
    drv: &mut D,
    clock: &dyn Clock,
) -> Vec<Violation> {
    let mut out = device_suite(drv, DeviceKind::FilterWheel, clock);
    poll_once(drv, &mut out);
    let slots = match drv.slots() {
        Ok(slots) => slots,
        Err(err) => {
            violation(&mut out, "slots", format!("slots failed: {err}"));
            return out;
        }
    };
    if slots.is_empty() {
        violation(
            &mut out,
            "slots_contiguous",
            "a wheel has at least one slot",
        );
    }
    for (i, slot) in slots.iter().enumerate() {
        if usize::try_from(slot.index0).ok() != Some(i) {
            violation(
                &mut out,
                "slots_contiguous",
                format!("slot {i} reports index0 {}", slot.index0),
            );
        }
    }
    match drv.position() {
        Ok(Some(p)) if usize::try_from(p).map_or(true, |p| p >= slots.len()) => violation(
            &mut out,
            "position_in_range",
            format!("position {p} with {} slots", slots.len()),
        ),
        Ok(_) => {}
        Err(err) => violation(&mut out, "position", format!("position failed: {err}")),
    }
    out
}

/// ObservingConditions (R18).
pub fn weather_suite<D: WeatherDriver + ?Sized>(drv: &mut D, clock: &dyn Clock) -> Vec<Violation> {
    let mut out = device_suite(drv, DeviceKind::Weather, clock);
    poll_once(drv, &mut out);
    let now = clock.now_mono_ns();
    match drv.read() {
        Ok(r) => {
            for (name, value) in [
                ("cloud_cover_pct", &r.cloud_cover_pct),
                ("dew_point_c", &r.dew_point_c),
                ("humidity_pct", &r.humidity_pct),
                ("pressure_hpa", &r.pressure_hpa),
                ("rain_rate_mm_h", &r.rain_rate_mm_h),
                ("sky_brightness_lux", &r.sky_brightness_lux),
                ("sky_quality", &r.sky_quality_mag_arcsec2),
                ("sky_temperature_c", &r.sky_temperature_c),
                ("star_fwhm_arcsec", &r.star_fwhm_arcsec),
                ("temperature_c", &r.temperature_c),
                ("wind_direction_deg", &r.wind_direction_deg),
                ("wind_gust_ms", &r.wind_gust_ms),
                ("wind_speed_ms", &r.wind_speed_ms),
            ] {
                check_observed_f64(name, value, now, &mut out);
            }
            check_range(
                "cloud_cover_pct",
                &r.cloud_cover_pct,
                0.0..100.0,
                true,
                &mut out,
            );
            check_range("humidity_pct", &r.humidity_pct, 0.0..100.0, true, &mut out);
            for (sensor, age) in &r.sensor_ages_s {
                if !(age.is_finite() && *age >= 0.0) {
                    violation(
                        &mut out,
                        "sensor_age_non_negative",
                        format!("{sensor} age {age}"),
                    );
                }
            }
        }
        Err(err) => violation(&mut out, "read", format!("read failed: {err}")),
    }
    out
}

/// SafetyMonitor (R19).
pub fn safety_suite<D: SafetyDriver + ?Sized>(drv: &mut D, clock: &dyn Clock) -> Vec<Violation> {
    let mut out = device_suite(drv, DeviceKind::SafetyMonitor, clock);
    poll_once(drv, &mut out);
    if let Err(err) = drv.is_safe() {
        violation(&mut out, "is_safe", format!("is_safe failed: {err}"));
    }
    out
}

/// Rotator (R20).
pub fn rotator_suite<D: RotatorDriver + ?Sized>(drv: &mut D, clock: &dyn Clock) -> Vec<Violation> {
    let mut out = device_suite(drv, DeviceKind::Rotator, clock);
    poll_once(drv, &mut out);
    let now = clock.now_mono_ns();
    match drv.state() {
        Ok(state) => {
            check_observed_f64("position_deg", &state.position_deg, now, &mut out);
            check_observed_f64("mechanical_deg", &state.mechanical_deg, now, &mut out);
            check_observed("is_moving", &state.is_moving, now, &mut out);
            check_observed("reverse", &state.reverse, now, &mut out);
            check_observed_f64("target_deg", &state.target_deg, now, &mut out);
            check_range(
                "position_deg",
                &state.position_deg,
                0.0..360.0,
                false,
                &mut out,
            );
            check_range(
                "mechanical_deg",
                &state.mechanical_deg,
                0.0..360.0,
                false,
                &mut out,
            );
        }
        Err(err) => violation(&mut out, "state", format!("state failed: {err}")),
    }
    halt_ok("halt", drv.halt(), &mut out);
    out
}

/// CoverCalibrator (R21).
pub fn cover_calibrator_suite<D: CoverCalibratorDriver + ?Sized>(
    drv: &mut D,
    clock: &dyn Clock,
) -> Vec<Violation> {
    let mut out = device_suite(drv, DeviceKind::CoverCalibrator, clock);
    poll_once(drv, &mut out);
    let now = clock.now_mono_ns();
    match drv.state() {
        Ok(state) => {
            check_observed("cover_state", &state.cover_state, now, &mut out);
            check_observed("cover_moving", &state.cover_moving, now, &mut out);
            check_observed("calibrator_state", &state.calibrator_state, now, &mut out);
            check_observed(
                "calibrator_changing",
                &state.calibrator_changing,
                now,
                &mut out,
            );
            check_observed("brightness", &state.brightness, now, &mut out);
            if let (Some(b), Some(max)) = (state.brightness.value, state.max_brightness) {
                if b > max {
                    violation(
                        &mut out,
                        "brightness_within_max",
                        format!("brightness {b} > max {max}"),
                    );
                }
            }
        }
        Err(err) => violation(&mut out, "state", format!("state failed: {err}")),
    }
    halt_ok("halt_cover", drv.halt_cover(), &mut out);
    out
}

/// Switch (R22).
pub fn switch_suite<D: SwitchDriver + ?Sized>(drv: &mut D, clock: &dyn Clock) -> Vec<Violation> {
    let mut out = device_suite(drv, DeviceKind::Switch, clock);
    poll_once(drv, &mut out);
    let channels = match drv.channels() {
        Ok(channels) => channels,
        Err(err) => {
            violation(&mut out, "channels", format!("channels failed: {err}"));
            return out;
        }
    };
    let mut ids: Vec<u32> = channels.iter().map(|c| c.id).collect();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() != channels.len() {
        violation(
            &mut out,
            "channel_ids_unique",
            "duplicate switch channel ids",
        );
    }
    for channel in &channels {
        if !(channel.min <= channel.max && channel.step > 0.0) {
            violation(
                &mut out,
                "channel_range",
                format!(
                    "channel {} min {} max {} step {}",
                    channel.id, channel.min, channel.max, channel.step
                ),
            );
        }
        match drv.read(channel.id) {
            Ok(v) if !v.is_finite() || v < channel.min || v > channel.max => violation(
                &mut out,
                "channel_value_in_range",
                format!("channel {} reads {v}", channel.id),
            ),
            Ok(_) => {}
            Err(err) => violation(&mut out, "read", format!("channel {}: {err}", channel.id)),
        }
        if !channel.can_async {
            if let Ok(Some(_)) = drv.state_change_complete(channel.id) {
                violation(
                    &mut out,
                    "async_only_when_supported",
                    format!("channel {} reports async completion", channel.id),
                );
            }
        }
    }
    out
}

/// Dome (R23).
pub fn dome_suite<D: DomeDriver + ?Sized>(drv: &mut D, clock: &dyn Clock) -> Vec<Violation> {
    let mut out = device_suite(drv, DeviceKind::Dome, clock);
    poll_once(drv, &mut out);
    let now = clock.now_mono_ns();
    match drv.state() {
        Ok(state) => {
            check_observed("shutter", &state.shutter, now, &mut out);
            check_observed_f64("azimuth_deg", &state.azimuth_deg, now, &mut out);
            check_observed_f64("altitude_deg", &state.altitude_deg, now, &mut out);
            check_observed("slewing", &state.slewing, now, &mut out);
            check_observed("at_park", &state.at_park, now, &mut out);
            check_observed("at_home", &state.at_home, now, &mut out);
            check_observed("slaved", &state.slaved, now, &mut out);
            check_range(
                "azimuth_deg",
                &state.azimuth_deg,
                0.0..360.0,
                false,
                &mut out,
            );
        }
        Err(err) => violation(&mut out, "state", format!("state failed: {err}")),
    }
    halt_ok("abort", drv.abort(), &mut out);
    out
}

/// Static checks every `Operation` must pass before the worker drives it.
pub fn operation_suite<D: ?Sized>(op: &dyn Operation<D>) -> Vec<Violation> {
    let mut out = Vec::new();
    if op.deadline().is_zero() {
        violation(
            &mut out,
            "operation_has_a_deadline",
            format!("{:?} has a zero deadline (C-4)", op.kind()),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::ops::{OpKind, OpProgress};
    use crate::equipment::snapshot::{FocuserCapabilities, FocuserState, WeatherReading};
    use crate::equipment::testing::fake_handles::{fake_identity, FakeAbortPath};
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::traits::{AbortPath, ConnectReport, RawPoll};
    use crate::equipment::types::{DeviceError, DeviceIdentity, Source};
    use std::sync::Arc;

    /// A reference focuser: compliant unless `broken`.
    struct RefFocuser {
        identity: DeviceIdentity,
        connected: bool,
        broken: bool,
        clock: Arc<ManualClock>,
    }

    impl Device for RefFocuser {
        fn identity(&self) -> &DeviceIdentity {
            &self.identity
        }
        fn connect(&mut self, _timeout: Duration) -> DeviceResult<ConnectReport> {
            self.connected = true;
            Ok(ConnectReport::default())
        }
        fn disconnect(&mut self) -> DeviceResult<()> {
            self.connected = false;
            Ok(())
        }
        fn is_connected(&mut self) -> DeviceResult<bool> {
            Ok(self.connected)
        }
        fn poll(&mut self, _plan: &PollPlan) -> DeviceResult<RawPoll> {
            let now = self.clock.now_mono_ns();
            Ok(RawPoll {
                started_mono_ns: now,
                finished_mono_ns: now,
                ..RawPoll::default()
            })
        }
        fn supported_actions(&mut self) -> DeviceResult<Vec<String>> {
            Ok(Vec::new())
        }
        fn action(&mut self, _name: &str, _params: &str) -> DeviceResult<String> {
            if self.broken {
                Ok(String::new())
            } else {
                Err(DeviceError::unsupported("no actions"))
            }
        }
        fn abort_path(&self) -> Option<Arc<dyn AbortPath>> {
            if self.broken {
                None
            } else {
                Some(FakeAbortPath::new() as Arc<dyn AbortPath>)
            }
        }
    }

    impl FocuserDriver for RefFocuser {
        fn capabilities(&mut self) -> DeviceResult<FocuserCapabilities> {
            Ok(FocuserCapabilities {
                absolute: true,
                max_step: Some(10_000),
                max_increment: Some(if self.broken { 20_000 } else { 1_000 }),
                ..FocuserCapabilities::default()
            })
        }
        fn state(&mut self) -> DeviceResult<FocuserState> {
            let clock = self.clock.as_ref();
            if self.broken {
                // Synthesised values: a 0 C "temperature" that was never read
                // and a Fresh flag with no value.
                Ok(FocuserState {
                    position: Observed::fresh_at(100, Source::Property, clock),
                    is_moving: Observed {
                        value: None,
                        ..Observed::fresh_at(false, Source::Property, clock)
                    },
                    temperature_c: Observed {
                        value: Some(0.0),
                        ..Observed::not_set(Source::Property)
                    },
                    temp_comp: Observed::fresh_at(false, Source::Property, clock),
                })
            } else {
                Ok(FocuserState {
                    position: Observed::fresh_at(100, Source::Property, clock),
                    is_moving: Observed::fresh_at(false, Source::Property, clock),
                    temperature_c: Observed::unsupported(Source::Property),
                    temp_comp: Observed::unsupported(Source::Property),
                })
            }
        }
        fn move_absolute(&mut self, _target: i32) -> DeviceResult<()> {
            Ok(())
        }
        fn move_relative(&mut self, _steps: i32) -> DeviceResult<()> {
            if self.broken {
                Ok(())
            } else {
                Err(DeviceError::invalid_operation("absolute focuser"))
            }
        }
        fn halt(&mut self) -> DeviceResult<()> {
            Ok(())
        }
        fn set_temp_comp(&mut self, _on: bool) -> DeviceResult<()> {
            Err(DeviceError::unsupported("no tempcomp"))
        }
        fn sync_position(&mut self, _position: i32) -> DeviceResult<()> {
            Err(DeviceError::unsupported("Alpaca has no sync"))
        }
    }

    struct RefWeather {
        identity: DeviceIdentity,
        clock: Arc<ManualClock>,
        nan: bool,
    }

    impl Device for RefWeather {
        fn identity(&self) -> &DeviceIdentity {
            &self.identity
        }
        fn connect(&mut self, _timeout: Duration) -> DeviceResult<ConnectReport> {
            Ok(ConnectReport::default())
        }
        fn disconnect(&mut self) -> DeviceResult<()> {
            Ok(())
        }
        fn is_connected(&mut self) -> DeviceResult<bool> {
            Ok(true)
        }
        fn poll(&mut self, _plan: &PollPlan) -> DeviceResult<RawPoll> {
            Ok(RawPoll::default())
        }
        fn supported_actions(&mut self) -> DeviceResult<Vec<String>> {
            Err(DeviceError::unsupported("not implemented"))
        }
        fn action(&mut self, _name: &str, _params: &str) -> DeviceResult<String> {
            Err(DeviceError::unsupported("no actions"))
        }
        fn abort_path(&self) -> Option<Arc<dyn AbortPath>> {
            None
        }
    }

    impl WeatherDriver for RefWeather {
        fn read(&mut self) -> DeviceResult<WeatherReading> {
            let clock = self.clock.as_ref();
            Ok(WeatherReading {
                temperature_c: Observed::fresh_at(
                    if self.nan { f64::NAN } else { 4.0 },
                    Source::Property,
                    clock,
                ),
                humidity_pct: Observed::fresh_at(80.0, Source::Property, clock),
                ..WeatherReading::default()
            })
        }
        fn refresh(&mut self) -> DeviceResult<()> {
            Ok(())
        }
    }

    fn focuser(broken: bool) -> RefFocuser {
        let clock = Arc::new(ManualClock::new());
        clock.advance_ms(1000);
        RefFocuser {
            identity: fake_identity(DeviceKind::Focuser),
            connected: false,
            broken,
            clock,
        }
    }

    #[test]
    fn a_compliant_focuser_passes_and_is_disconnected_cleanly() {
        let mut drv = focuser(false);
        let clock = Arc::clone(&drv.clock);
        assert_conforms(
            "reference focuser",
            &focuser_suite(&mut drv, clock.as_ref()),
        );
        assert_conforms("reference focuser", &disconnect_suite(&mut drv));
        // Through a trait object, as the worker holds drivers.
        let mut boxed: Box<dyn FocuserDriver> = Box::new(focuser(false));
        assert_conforms(
            "boxed focuser",
            &focuser_suite(boxed.as_mut(), clock.as_ref()),
        );
    }

    #[test]
    fn a_broken_focuser_is_caught_rule_by_rule() {
        let mut drv = focuser(true);
        let clock = Arc::clone(&drv.clock);
        let violations = focuser_suite(&mut drv, clock.as_ref());
        for rule in [
            "unknown_is_never_fresh",
            "no_value_without_reading",
            "max_increment_within_max_step",
            "move_mode_matches_absolute",
            "motion_devices_have_an_abort_path",
            "unknown_action_is_unsupported",
        ] {
            assert!(
                broke(&violations, rule),
                "{rule} not caught: {violations:?}"
            );
        }
        let mut wrong_kind = focuser(false);
        wrong_kind.identity = fake_identity(DeviceKind::Rotator);
        assert!(broke(
            &device_suite(&mut wrong_kind, DeviceKind::Focuser, clock.as_ref()),
            "identity_kind"
        ));
        let mut qualified = focuser(false);
        qualified.identity.qualification = Qualification::Qualified {
            record_id: "self-declared".into(),
        };
        assert!(broke(
            &device_suite(&mut qualified, DeviceKind::Focuser, clock.as_ref()),
            "only_rig_acceptance_qualifies"
        ));
    }

    #[test]
    fn weather_nan_must_be_unknown() {
        let clock = Arc::new(ManualClock::new());
        let mut good = RefWeather {
            identity: fake_identity(DeviceKind::Weather),
            clock: Arc::clone(&clock),
            nan: false,
        };
        assert_conforms(
            "reference weather",
            &weather_suite(&mut good, clock.as_ref()),
        );
        let mut bad = RefWeather {
            identity: fake_identity(DeviceKind::Weather),
            clock: Arc::clone(&clock),
            nan: true,
        };
        assert!(broke(
            &weather_suite(&mut bad, clock.as_ref()),
            "nan_is_unknown"
        ));
    }

    #[test]
    fn observed_rules_and_operations() {
        let clock = ManualClock::new();
        let mut out = Vec::new();
        let future = Observed {
            mono_ns: Some(10),
            ..Observed::fresh_at(1.0_f64, Source::Derived, &clock)
        };
        check_observed_f64("x", &future, 0, &mut out);
        assert!(broke(&out, "timestamps_not_in_the_future"));
        let mut out = Vec::new();
        let stale_empty = Observed::<f64> {
            state: ObsState::Stale,
            ..Observed::not_set(Source::Derived)
        };
        check_observed_f64("x", &stale_empty, 0, &mut out);
        assert!(broke(&out, "stale_keeps_last_value"));

        struct NoDeadline;
        impl Operation<()> for NoDeadline {
            fn kind(&self) -> OpKind {
                OpKind::FocuserMove
            }
            fn deadline(&self) -> Duration {
                Duration::ZERO
            }
            fn start(&mut self, _d: &mut (), _c: &dyn Clock) -> DeviceResult<()> {
                Ok(())
            }
            fn step(&mut self, _d: &mut (), _c: &dyn Clock) -> OpProgress {
                OpProgress::Done(crate::equipment::types::OpOutcome::Succeeded)
            }
            fn cancel(&mut self, _d: &mut ()) -> DeviceResult<()> {
                Ok(())
            }
        }
        assert!(broke(
            &operation_suite(&NoDeadline),
            "operation_has_a_deadline"
        ));
    }
}

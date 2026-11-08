//! The `/status` and `TELEMETRY_UPDATE` device blocks (spec §2.13). Owner:
//! WP1B-1.
//!
//! Each block is built from a device's published snapshot, identity and
//! health only; nothing here touches a driver (`health_tests.rs`: `/status`
//! never enters a driver). Today's keys keep their meaning and their null
//! semantics; every new key is additive and nullable. Unknown values are
//! null, never a synthesised 0 (D5). Every device block names its
//! `identity_key` (§2.6), the stable key the frontend keys hardware on,
//! beside the per-connection `connection_id`.

use serde_json::{json, Map, Value};

use super::mount;
use super::snapshot::{
    FilterWheelSnapshot, FocuserSnapshot, GuiderSnapshot, SafetySnapshot, WeatherSnapshot,
};
use super::snapshot::{MountSnapshot, ResumeBlock};
use super::types::{DeviceIdentity, HealthState, Observed, Site};

/// Milliseconds between a monotonic reading and now.
fn age_ms(now_mono_ns: u64, mono_ns: Option<u64>) -> Option<u64> {
    mono_ns.map(|mono| now_mono_ns.saturating_sub(mono) / 1_000_000)
}

/// The additive health keys every block carries (§2.8): `link_state`,
/// `failure_class` and `reason` (which mirrors `last_error`), and `state`
/// too where the block has no `state` of its own (the guiding block's
/// `state` is PHD2's app state, §2.13).
fn with_health(mut value: Value, health: &HealthState) -> Value {
    if let Value::Object(map) = &mut value {
        let link = serde_json::to_value(health.state).unwrap_or(Value::Null);
        if !map.contains_key("state") {
            map.insert("state".into(), link.clone());
        }
        map.insert("link_state".into(), link);
        map.insert(
            "failure_class".into(),
            serde_json::to_value(health.failure_class).unwrap_or(Value::Null),
        );
        map.insert(
            "reason".into(),
            health.reason.clone().map_or(Value::Null, Value::String),
        );
    }
    value
}

/// An observed value with its age, for the weather and safety blocks.
fn observed_entry<T: serde::Serialize>(observed: &Observed<T>, now_mono_ns: u64) -> Value {
    json!({
        "value": observed.value,
        "state": observed.state,
        "observed_at": observed.observed_at,
        "age_ms": age_ms(now_mono_ns, observed.mono_ns),
        "source": observed.source,
    })
}

/// §2.13 `mount`: `mount::status_block` (today's keys plus every new one)
/// with the health keys.
pub fn mount_block(
    snapshot: &MountSnapshot,
    identity: &DeviceIdentity,
    health: &HealthState,
    session_site: Option<&Site>,
    now_mono_ns: u64,
) -> Value {
    let mut value = with_health(
        mount::status_block(snapshot, identity, health, session_site, now_mono_ns),
        health,
    );
    if let Value::Object(map) = &mut value {
        map.insert("driver".into(), json!(identity.driver));
        map.insert("identity_key".into(), json!(identity.identity_key));
        map.insert("connection_id".into(), json!(identity.connection_id));
        map.insert(
            "qualification".into(),
            serde_json::to_value(&identity.qualification).unwrap_or(Value::Null),
        );
    }
    value
}

/// §2.13 `focuser`: today's keys (`name`, `connected`, `position`,
/// `moving`, `absolute`, `max_range`) plus the new ones. `position` is the
/// logical position (backlash and relative emulation hidden); null while
/// the origin is unknown.
pub fn focuser_block(
    snapshot: &FocuserSnapshot,
    identity: &DeviceIdentity,
    health: &HealthState,
    now_mono_ns: u64,
) -> Value {
    let s = &snapshot.state;
    let caps = &snapshot.capabilities;
    let observed = s.position.mono_ns.or(s.is_moving.mono_ns);
    let value = json!({
        "name": identity.name,
        "connected": health.state.is_connected(),
        "position": snapshot.logical_position,
        "raw_position": s.position.value,
        "moving": s.is_moving.value,
        "absolute": caps.absolute,
        "max_range": caps.max_step,
        "max_increment": caps.max_increment,
        "temperature": s.temperature_c.value,
        "temperature_source": s.temperature_c.value.map(|_| snapshot.temperature_role),
        "temperature_age_ms": age_ms(now_mono_ns, s.temperature_c.mono_ns),
        "step_size_um": caps.step_size_um,
        "temp_comp": s.temp_comp.value,
        "temp_comp_available": caps.temp_comp_available,
        "position_source": snapshot.position_source,
        "backlash_model": snapshot.backlash_model,
        "backlash_owner": snapshot.backlash_owner,
        "settling": snapshot.settling,
        "active_op": snapshot.active_op.as_ref().map(|op| op.kind),
        "observed_at": s.position.observed_at.or(s.is_moving.observed_at),
        "age_ms": age_ms(now_mono_ns, observed),
        "driver": identity.driver,
        "identity_key": identity.identity_key,
        "connection_id": identity.connection_id,
        "qualification": identity.qualification,
    });
    with_health(value, health)
}

/// §2.13 `filter_wheel`: `{name, connected, filter, slot, moving, offset}`
/// (the frontend's shape) plus `slots` and `verified`. `slot` is 0-based
/// and null while moving or unknown; `filter` and `offset` are those of the
/// reported slot.
pub fn filter_wheel_block(
    snapshot: &FilterWheelSnapshot,
    identity: &DeviceIdentity,
    health: &HealthState,
    now_mono_ns: u64,
) -> Value {
    let slot = snapshot.position.value;
    let current = slot.and_then(|slot| snapshot.slots.iter().find(|s| s.index == slot));
    let slots: Vec<Value> = snapshot
        .slots
        .iter()
        .map(|s| json!({"index": s.index, "name": s.name, "offset": s.offset}))
        .collect();
    let value = json!({
        "name": identity.name,
        "connected": health.state.is_connected(),
        "filter": current.map(|s| s.name.clone()),
        "slot": slot,
        "position": slot,
        "moving": snapshot.moving,
        "offset": current.and_then(|s| s.offset),
        "slots": slots,
        "verified": snapshot.verified,
        "verified_slot": snapshot.verified_slot,
        "active_op": snapshot.active_op.as_ref().map(|op| op.kind),
        "observed_at": snapshot.position.observed_at,
        "age_ms": age_ms(now_mono_ns, snapshot.position.mono_ns),
        "driver": identity.driver,
        "identity_key": identity.identity_key,
        "connection_id": identity.connection_id,
        "qualification": identity.qualification,
    });
    with_health(value, health)
}

/// §2.13 `guiding`: `{state, rms_ra_arcsec, rms_dec_arcsec, rms_total, dx,
/// dy, snr, star_mass, hfd, settling, pixel_scale, calibrated,
/// last_dither{at, dx, dy, settle_status}, observed_at}`. `rms_total` is in
/// arcsec under the key the frontend expects; null without a pixel scale.
pub fn guiding_block(
    snapshot: &GuiderSnapshot,
    identity: &DeviceIdentity,
    health: &HealthState,
    now_mono_ns: u64,
) -> Value {
    let rms = snapshot.rms.as_ref();
    let step = snapshot.last_step.as_ref();
    let value = json!({
        "name": identity.name,
        "connected": health.state.is_connected(),
        "state": snapshot.app_state.value,
        "app_state_state": snapshot.app_state.state,
        "rms_ra_arcsec": rms.and_then(|r| r.ra_arcsec),
        "rms_dec_arcsec": rms.and_then(|r| r.dec_arcsec),
        "rms_total": rms.and_then(|r| r.total_arcsec),
        "rms_ra_px": rms.map(|r| r.ra_px),
        "rms_dec_px": rms.map(|r| r.dec_px),
        "rms_total_px": rms.map(|r| r.total_px),
        "steps_n": rms.map(|r| r.steps_n),
        "dx": step.map(|s| s.dx),
        "dy": step.map(|s| s.dy),
        "snr": step.and_then(|s| s.snr),
        "star_mass": step.and_then(|s| s.star_mass),
        "hfd": step.and_then(|s| s.hfd),
        "settling": snapshot.settling,
        "settle": snapshot.settle,
        "pixel_scale": snapshot.pixel_scale_arcsec,
        "calibrated": snapshot.calibrated,
        "last_dither": snapshot.last_dither.as_ref().map(|d| json!({
            "at": d.at,
            "dx": d.dx,
            "dy": d.dy,
            "settle_status": d.settle_status,
        })),
        "phd2_version": snapshot.phd2_version,
        "profile": snapshot.profile,
        "camera_name": snapshot.camera_name,
        "observed_at": snapshot.app_state.observed_at,
        "age_ms": age_ms(now_mono_ns, snapshot.app_state.mono_ns),
        "driver": identity.driver,
        "identity_key": identity.identity_key,
        "connection_id": identity.connection_id,
        "qualification": identity.qualification,
    });
    with_health(value, health)
}

/// §2.13 `weather`: each ObservingConditions field with its age and state,
/// plus the per-sensor ages the device reports.
pub fn weather_block(
    snapshot: &WeatherSnapshot,
    identity: &DeviceIdentity,
    health: &HealthState,
    now_mono_ns: u64,
) -> Value {
    let r = &snapshot.reading;
    let mut fields = Map::new();
    for (key, observed) in [
        ("cloud_cover_pct", &r.cloud_cover_pct),
        ("dew_point_c", &r.dew_point_c),
        ("humidity_pct", &r.humidity_pct),
        ("pressure_hpa", &r.pressure_hpa),
        ("rain_rate_mm_h", &r.rain_rate_mm_h),
        ("sky_brightness_lux", &r.sky_brightness_lux),
        ("sky_quality_mag_arcsec2", &r.sky_quality_mag_arcsec2),
        ("sky_temperature_c", &r.sky_temperature_c),
        ("star_fwhm_arcsec", &r.star_fwhm_arcsec),
        ("temperature_c", &r.temperature_c),
        ("wind_direction_deg", &r.wind_direction_deg),
        ("wind_gust_ms", &r.wind_gust_ms),
        ("wind_speed_ms", &r.wind_speed_ms),
    ] {
        fields.insert(key.into(), observed_entry(observed, now_mono_ns));
    }
    let value = json!({
        "name": identity.name,
        "connected": health.state.is_connected(),
        "fields": fields,
        "temperature": r.temperature_c.fresh_value(),
        "humidity": r.humidity_pct.fresh_value(),
        "dew_point": r.dew_point_c.fresh_value(),
        "sensor_ages_s": r.sensor_ages_s,
        "observed_at": snapshot.published_at,
        "age_ms": age_ms(now_mono_ns, Some(snapshot.published_mono_ns)),
        "driver": identity.driver,
        "identity_key": identity.identity_key,
        "connection_id": identity.connection_id,
        "qualification": identity.qualification,
    });
    with_health(value, health)
}

/// §2.13 `safety_monitor`: this monitor's level with its age and state.
/// The combined state of every monitor is the `safety` block.
pub fn safety_monitor_block(
    snapshot: &SafetySnapshot,
    identity: &DeviceIdentity,
    health: &HealthState,
    now_mono_ns: u64,
) -> Value {
    let value = json!({
        "name": identity.name,
        "connected": health.state.is_connected(),
        "level": snapshot.level.value,
        "is_safe": snapshot.level.fresh_value().map(|level| {
            *level == super::snapshot::SafetyLevel::Safe
        }),
        "reading": observed_entry(&snapshot.level, now_mono_ns),
        "detail": snapshot.detail,
        "observed_at": snapshot.level.observed_at,
        "age_ms": age_ms(now_mono_ns, snapshot.level.mono_ns),
        "driver": identity.driver,
        "identity_key": identity.identity_key,
        "connection_id": identity.connection_id,
        "qualification": identity.qualification,
    });
    with_health(value, health)
}

/// The `resume_blocks` block: the active blocks, newest last.
pub fn resume_blocks_block(blocks: &[ResumeBlock]) -> Value {
    serde_json::to_value(blocks).unwrap_or_else(|_| json!([]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::snapshot::{FilterSlot, GuideRms, LastDither};
    use crate::equipment::testing::fake_handles::fake_identity;
    use crate::equipment::types::{DeviceKind, LinkState, ObsState, PierSide, Source};
    use chrono::Utc;

    fn ready() -> HealthState {
        HealthState::ready(Utc::now(), 1)
    }

    /// Every device block carries the identity key beside the connection
    /// id, so a client can tell a reconnect of the same device from a new
    /// one.
    #[test]
    fn every_device_block_names_its_identity_key() {
        let blocks = [
            (
                DeviceKind::Mount,
                mount_block(
                    &MountSnapshot::default(),
                    &fake_identity(DeviceKind::Mount),
                    &ready(),
                    None,
                    0,
                ),
            ),
            (
                DeviceKind::Focuser,
                focuser_block(
                    &FocuserSnapshot::default(),
                    &fake_identity(DeviceKind::Focuser),
                    &ready(),
                    0,
                ),
            ),
            (
                DeviceKind::FilterWheel,
                filter_wheel_block(
                    &FilterWheelSnapshot::default(),
                    &fake_identity(DeviceKind::FilterWheel),
                    &ready(),
                    0,
                ),
            ),
            (
                DeviceKind::Guider,
                guiding_block(
                    &GuiderSnapshot::default(),
                    &fake_identity(DeviceKind::Guider),
                    &ready(),
                    0,
                ),
            ),
            (
                DeviceKind::Weather,
                weather_block(
                    &WeatherSnapshot::default(),
                    &fake_identity(DeviceKind::Weather),
                    &ready(),
                    0,
                ),
            ),
            (
                DeviceKind::SafetyMonitor,
                safety_monitor_block(
                    &SafetySnapshot::default(),
                    &fake_identity(DeviceKind::SafetyMonitor),
                    &ready(),
                    0,
                ),
            ),
        ];
        for (kind, block) in blocks {
            let identity = fake_identity(kind);
            assert_eq!(
                block["identity_key"],
                serde_json::json!(identity.identity_key),
                "{kind:?}"
            );
            assert!(block["connection_id"].is_string(), "{kind:?}");
        }
    }

    #[test]
    fn the_mount_block_has_every_section_2_13_key_and_pier_side_as_a_string() {
        let mut snapshot = MountSnapshot::default();
        snapshot.state.side_of_pier =
            Observed::fresh(PierSide::West, Source::DeviceState, Utc::now(), 1);
        let identity = fake_identity(DeviceKind::Mount);
        let block = mount_block(&snapshot, &identity, &ready(), None, 2_000_000);
        for key in mount::STATUS_KEYS {
            assert!(block.get(*key).is_some(), "missing {key}");
        }
        assert_eq!(block["pier_side"], "west");
        assert_eq!(block["state"], "ready");
        assert!(block["failure_class"].is_null() && block["reason"].is_null());
        // Unknown readings are null, never 0.
        assert!(block["ra"].is_null() && block["alt"].is_null());
    }

    #[test]
    fn the_focuser_block_keeps_todays_keys_and_never_invents_a_position() {
        let identity = fake_identity(DeviceKind::Focuser);
        let block = focuser_block(&FocuserSnapshot::default(), &identity, &ready(), 0);
        for key in [
            "name",
            "connected",
            "position",
            "moving",
            "absolute",
            "max_range",
            "temperature",
            "temperature_source",
            "step_size_um",
            "temp_comp",
            "temp_comp_available",
            "max_increment",
            "position_source",
            "backlash_model",
            "backlash_owner",
            "settling",
            "observed_at",
            "age_ms",
            "state",
            "reason",
        ] {
            assert!(block.get(key).is_some(), "missing {key}");
        }
        assert!(block["position"].is_null());
        assert!(block["temperature"].is_null());
    }

    #[test]
    fn the_filter_wheel_block_matches_the_frontend_shape() {
        let identity = fake_identity(DeviceKind::FilterWheel);
        let snapshot = FilterWheelSnapshot {
            position: Observed::fresh(1, Source::Property, Utc::now(), 1),
            slots: vec![
                FilterSlot {
                    index: 0,
                    name: "L".into(),
                    offset: Some(0),
                    ..FilterSlot::default()
                },
                FilterSlot {
                    index: 1,
                    name: "Ha".into(),
                    offset: Some(120),
                    ..FilterSlot::default()
                },
            ],
            verified: true,
            ..FilterWheelSnapshot::default()
        };
        let block = filter_wheel_block(&snapshot, &identity, &ready(), 0);
        for key in [
            "name",
            "connected",
            "filter",
            "slot",
            "moving",
            "offset",
            "slots",
            "verified",
        ] {
            assert!(block.get(key).is_some(), "missing {key}");
        }
        assert_eq!(block["filter"], "Ha");
        assert_eq!(block["slot"], 1);
        assert_eq!(block["offset"], 120);
        assert_eq!(block["slots"][0]["name"], "L");
    }

    #[test]
    fn the_guiding_block_reports_rms_total_in_arcsec_and_null_without_a_scale() {
        let identity = fake_identity(DeviceKind::Guider);
        let mut snapshot = GuiderSnapshot {
            rms: Some(GuideRms {
                ra_px: 0.3,
                dec_px: 0.4,
                total_px: 0.5,
                ..GuideRms::default()
            }),
            last_dither: Some(LastDither {
                at: Utc::now(),
                dx: Some(1.0),
                dy: Some(-1.0),
                settle_status: Some(0),
            }),
            ..GuiderSnapshot::default()
        };
        let block = guiding_block(&snapshot, &identity, &ready(), 0);
        for key in [
            "state",
            "rms_ra_arcsec",
            "rms_dec_arcsec",
            "rms_total",
            "dx",
            "dy",
            "snr",
            "star_mass",
            "hfd",
            "settling",
            "pixel_scale",
            "calibrated",
            "last_dither",
            "observed_at",
        ] {
            assert!(block.get(key).is_some(), "missing {key}");
        }
        assert!(block["rms_total"].is_null(), "no scale, no arcsec");
        assert_eq!(block["last_dither"]["settle_status"], 0);
        snapshot.rms.as_mut().unwrap().total_arcsec = Some(0.8);
        assert_eq!(
            guiding_block(&snapshot, &identity, &ready(), 0)["rms_total"],
            0.8
        );
    }

    #[test]
    fn weather_and_safety_blocks_carry_ages_and_never_cache_safety() {
        let identity = fake_identity(DeviceKind::Weather);
        let mut snapshot = WeatherSnapshot::default();
        snapshot.reading.temperature_c =
            Observed::fresh(4.5, Source::Property, Utc::now(), 1_000_000);
        let block = weather_block(&snapshot, &identity, &ready(), 3_000_000);
        assert_eq!(block["fields"]["temperature_c"]["value"], 4.5);
        assert_eq!(block["fields"]["temperature_c"]["age_ms"], 2);
        assert_eq!(block["temperature"], 4.5);
        assert!(block["fields"]["humidity_pct"]["value"].is_null());

        let identity = fake_identity(DeviceKind::SafetyMonitor);
        let unknown = SafetySnapshot {
            level: Observed::error(Source::Property),
            ..SafetySnapshot::default()
        };
        let block = safety_monitor_block(&unknown, &identity, &ready(), 0);
        assert!(block["is_safe"].is_null(), "an unknown level is never safe");
        assert_eq!(
            block["reading"]["state"],
            serde_json::json!(ObsState::Error)
        );
        let lost = HealthState {
            state: LinkState::Lost,
            reason: Some("timeout".into()),
            ..ready()
        };
        let block = safety_monitor_block(&unknown, &identity, &lost, 0);
        assert_eq!(block["connected"], false);
        assert_eq!(block["reason"], "timeout");
    }
}

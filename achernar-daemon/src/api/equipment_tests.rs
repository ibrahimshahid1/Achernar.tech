//! Route and integration tests for the WP1B-1 spine, against the in-process
//! simulators, the PHD2 mock and fake handles. No serial port, no real
//! hardware, no network beyond loopback; every timing rule that matters is
//! bounded, so the suite runs on Windows CI too.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};

use super::equipment as eq;
use crate::config::{AchernarConfig, CameraConfig, FocusConfig};
use crate::equipment::health_bridge::HealthPolicy;
use crate::equipment::sim::mount::SimMountProfile;
use crate::equipment::spine::{SpineContext, WorkerTuning};
use crate::equipment::worker::Cadence;
use crate::hardware::health::DeviceKind;
use crate::state::AppState;
use crate::storage::{
    capture_store::CaptureStore, frames_repo::FrameStore, local_store::LocalStore,
};

/// A fresh daemon state with the inert hub, quick sim cadences and quick
/// settles. Nothing is connected and nothing runs until a test connects.
pub(crate) async fn state() -> (PathBuf, AppState) {
    let directory =
        std::env::temp_dir().join(format!("achernar-spine-test-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&directory).unwrap();
    let store = LocalStore::open(directory.join("state.db").to_str().unwrap())
        .await
        .unwrap();
    let mut focus = FocusConfig::default();
    focus.models.quality_model_path =
        Some(directory.join("absent-fqe.onnx").to_string_lossy().into());
    let mut state = AppState::new(
        focus,
        CameraConfig::default(),
        store,
        FrameStore::new(directory.join("previews")),
        CaptureStore::new(directory.join("captures")),
        "spine-fixture-rig".into(),
        directory.join("data"),
        None,
    );
    let mut configs = AchernarConfig::default().equipment_configs();
    configs.mount.settle_s = 0.1;
    configs.focus_safety.settle_ms = 0;
    configs.filter_wheel.settle_ms = 0;
    let ctx = SpineContext::new(&state.equipment, configs, None, None);
    let mut profile = SimMountProfile::esp32_like();
    profile.slew_rate_deg_s = 60.0;
    ctx.set_tuning(WorkerTuning {
        cadence: Some(Cadence {
            idle: Duration::from_millis(100),
            active: Duration::from_millis(50),
        }),
        policy: Some(HealthPolicy::default()),
        sim_mount_profile: Some(profile),
    });
    state.equipment_ctx = Arc::new(ctx);
    (directory, state)
}

/// Disconnect everything (stopping the workers), then close the store.
pub(crate) async fn cleanup(directory: PathBuf, state: AppState) {
    for kind in eq::AUXILIARY_KINDS {
        let _ = eq::disconnect_device(&state, kind, true).await;
    }
    let pool = state.local_store.pool().clone();
    drop(state);
    pool.close().await;
    std::fs::remove_dir_all(directory).ok();
}

async fn connect(state: &AppState, kind: &str, driver: &str, option: &str) -> Value {
    super::connect_equipment(
        State(state.clone()),
        Json(super::ConnectRequest {
            equipment_type: Some(kind.into()),
            driver: driver.into(),
            option: Some(option.into()),
            index: None,
        }),
    )
    .await
    .unwrap_or_else(|err| panic!("connect {kind}/{driver}: {}: {}", err.code, err.message))
    .0
}

fn status(state: &AppState) -> Value {
    crate::telemetry::equipment_status(state)
}

/// Wait (real time, bounded) for `done` on the status payload.
async fn wait_status(state: &AppState, what: &str, done: impl Fn(&Value) -> bool) -> Value {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let value = status(state);
        if done(&value) {
            return value;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {what}: {value}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Wait until no command holds a domain: the previous operation resolved
/// and released its permit (an operator command on a held domain is 409).
async fn wait_idle(state: &AppState) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !state.equipment.gate().is_idle() {
        assert!(
            Instant::now() < deadline,
            "an operation never released its domain"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn event_names(state: &AppState) -> Vec<String> {
    eq::events_value(state, &eq::EventsQuery::default())["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["kind"].as_str().unwrap_or_default().to_string())
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_default_configuration_opens_nothing_and_broadcasts_nothing() {
    let (directory, state) = state().await;
    assert!(state.equipment.is_inert());
    assert!(state.equipment.attached().is_empty());
    let value = status(&state);
    assert_eq!(value["connected"], false);
    for key in [
        "mount",
        "focuser",
        "filter_wheel",
        "guiding",
        "weather",
        "safety_monitor",
        "rotator",
        "cover_calibrator",
        "switch",
        "dome",
    ] {
        assert!(
            value[key].is_null(),
            "{key} must be null with nothing connected"
        );
    }
    for kind in DeviceKind::ALL {
        let key = serde_json::to_value(kind).unwrap();
        assert_eq!(
            value["device_health"][key.as_str().unwrap()]["status"],
            "disconnected"
        );
    }
    assert_eq!(value["resume_blocks"], json!([]));
    // D13: an empty or "auto" Alpaca option is refused without the opt-in,
    // before any socket is opened.
    for option in ["", "auto"] {
        let err = super::connect_equipment(
            State(state.clone()),
            Json(super::ConnectRequest {
                equipment_type: Some("mount".into()),
                driver: "alpaca".into(),
                option: Some(option.into()),
                index: None,
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(
            err.status,
            axum::http::StatusCode::BAD_REQUEST,
            "{option:?}"
        );
        assert_eq!(err.code, "broadcast_discovery_disabled");
    }
    // Broadcast discovery is refused too.
    let err = crate::hardware::discovery::discover_equipment(
        &state,
        crate::hardware::discovery::DiscoveryRequest {
            equipment_type: Some("mount".into()),
            driver: Some("alpaca_broadcast".into()),
            option: None,
        },
    )
    .await
    .unwrap_err();
    assert!(err.contains("broadcast discovery is disabled"), "{err}");
    // A missing driver is a 400 on the direct route and the relay alike.
    let err = super::connect_equipment(
        State(state.clone()),
        Json(super::ConnectRequest {
            equipment_type: Some("focuser".into()),
            driver: String::new(),
            option: None,
            index: None,
        }),
    )
    .await
    .unwrap_err();
    assert_eq!(err.code, "missing_driver");
    let relayed = crate::relay_commands::execute(
        state.clone(),
        &crate::relay_commands::RelayCommand {
            command_id: "no-driver".into(),
            action: "equipment.connect".into(),
            payload: json!({"equipment_type": "focuser"}),
        },
    )
    .await
    .unwrap_err();
    assert!(relayed.starts_with("missing_driver"), "{relayed}");
    // Tier 2 and O4 drivers are refused with 501, never silently accepted.
    for (kind, driver) in [
        ("focuser", "moonlite"),
        ("focuser", "indi"),
        ("rotator", "alpaca"),
        ("guider", "direct"),
        ("mount", "lx200"),
    ] {
        let err = super::connect_equipment(
            State(state.clone()),
            Json(super::ConnectRequest {
                equipment_type: Some(kind.into()),
                driver: driver.into(),
                option: None,
                index: None,
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(err.status.as_u16(), 501, "{kind}/{driver}");
    }
    // Still nothing attached, no worker started.
    assert!(state.equipment.attached().is_empty());
    cleanup(directory, state).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_sim_mount_publishes_every_status_field_and_slews_through_the_gate() {
    let (directory, state) = state().await;
    let body = connect(&state, "mount", "sim", "").await;
    assert_eq!(body["status"], "connected");
    assert_eq!(
        body["identity"]["qualification"]["state"],
        "implemented_unqualified"
    );
    let value = wait_status(&state, "a polled mount", |v| {
        v["mount"]["lst_h"].is_number()
    })
    .await;
    let mount = &value["mount"];
    for key in crate::equipment::mount::STATUS_KEYS {
        assert!(mount.get(*key).is_some(), "mount block lacks {key}");
    }
    // The ESP32-like profile (§5.7 Tier 1 item 2).
    assert_eq!(mount["frame"], "jnow_apparent");
    assert_eq!(mount["alignment_mode"], "polar");
    assert_eq!(mount["can_flip"], false);
    assert_eq!(mount["pier_side"], "east", "pier_side is a string");
    assert_eq!(mount["parked"], false);
    assert_eq!(mount["state"], "ready");
    assert_eq!(value["device_health"]["mount"]["state"], "ready");

    // A short GoTo in the mount frame: commanded -> started -> arrived ->
    // settled, through the gate.
    let ra = mount["ra"].as_f64().unwrap();
    let dec = mount["dec"].as_f64().unwrap();
    let accepted = super::mount::slew_value(
        &state,
        super::mount::SlewRequest {
            ra,
            dec: dec + 2.0,
            frame: None,
            overriding: false,
        },
        false,
    )
    .await
    .unwrap();
    assert_eq!(accepted["status"], "slewing");
    assert!(accepted["command_id"].is_string());
    let deadline = Instant::now() + Duration::from_secs(10);
    while !event_names(&state).iter().any(|n| n == "slew_settled") {
        assert!(
            Instant::now() < deadline,
            "no slew_settled: {:?}",
            event_names(&state)
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let names = event_names(&state);
    let order: Vec<usize> = [
        "slew_commanded",
        "slew_started",
        "slew_arrived",
        "slew_settled",
    ]
    .iter()
    .map(|n| {
        names
            .iter()
            .position(|e| e == n)
            .unwrap_or_else(|| panic!("{n}: {names:?}"))
    })
    .collect();
    assert!(order.windows(2).all(|w| w[0] < w[1]), "{names:?}");

    // GET mount.
    let descriptor = super::mount::descriptor_value(&state).unwrap();
    assert_eq!(descriptor["frame"], "jnow_apparent");
    assert_eq!(descriptor["can_flip"], false);
    cleanup(directory, state).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mount_abort_stops_a_goto_and_park_unpark_are_verified() {
    let (directory, state) = state().await;
    connect(&state, "mount", "sim", "").await;
    let value = wait_status(&state, "a polled mount", |v| v["mount"]["ra"].is_number()).await;
    let ra = value["mount"]["ra"].as_f64().unwrap();
    // A long GoTo, then the abort: Slewing reads false within 2 s.
    super::mount::slew_value(
        &state,
        super::mount::SlewRequest {
            ra: (ra + 6.0) % 24.0,
            dec: 5.0,
            frame: None,
            overriding: false,
        },
        false,
    )
    .await
    .unwrap();
    wait_status(&state, "slewing", |v| v["mount"]["slewing"] == true).await;
    let aborted = super::mount::abort_value(&state, false).await.unwrap();
    assert_eq!(aborted["abort_path_ran"], true);
    let started = Instant::now();
    wait_status(&state, "stopped", |v| v["mount"]["slewing"] == false).await;
    assert!(started.elapsed() < Duration::from_secs(2));
    wait_idle(&state).await;

    // Park, verified through AtPark, then unpark.
    let parked = super::mount::park_value(&state, false, false)
        .await
        .unwrap();
    assert_eq!(parked["status"], "parking");
    wait_status(&state, "parked", |v| v["mount"]["parked"] == true).await;
    wait_idle(&state).await;
    super::mount::unpark_value(&state, false, false)
        .await
        .unwrap();
    wait_status(&state, "unparked", |v| v["mount"]["parked"] == false).await;
    wait_idle(&state).await;
    let names = event_names(&state);
    assert!(names.iter().any(|n| n == "parked"), "{names:?}");
    assert!(names.iter().any(|n| n == "unparked"), "{names:?}");

    // A slewing mount refuses a plain disconnect (409); force aborts first.
    super::mount::slew_value(
        &state,
        super::mount::SlewRequest {
            ra: (ra + 6.0) % 24.0,
            dec: 5.0,
            frame: None,
            overriding: false,
        },
        false,
    )
    .await
    .unwrap();
    wait_status(&state, "slewing again", |v| v["mount"]["slewing"] == true).await;
    let err = eq::disconnect_device(&state, DeviceKind::Mount, false)
        .await
        .unwrap_err();
    assert_eq!(err.status.as_u16(), 409);
    assert!(eq::disconnect_device(&state, DeviceKind::Mount, true)
        .await
        .unwrap());
    assert!(state.mount.lock().unwrap().is_none());
    assert!(status(&state)["mount"].is_null());
    cleanup(directory, state).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn abort_on_a_lost_handle_that_is_not_retired_reaches_its_abort_path() {
    use crate::equipment::snapshot::MountSnapshot;
    use crate::equipment::testing::fake_handles::fake_mount;
    use crate::equipment::types::LinkState;

    let (directory, state) = state().await;
    let (mount, fake) = fake_mount(MountSnapshot::default());
    *state.mount.lock().unwrap() = Some(mount.clone());
    state.equipment.attach(mount.clone());
    state.devices.write().unwrap().connected(
        DeviceKind::Mount,
        mount.connection_id().to_string(),
        Some(json!({"name": "fake"})),
    );
    fake.set_link(LinkState::Lost);
    // The registry also says it is gone: commands get 412...
    state
        .devices
        .write()
        .unwrap()
        .disconnected(DeviceKind::Mount, None, Some("lost".into()));
    let err = super::mount::set_tracking_value(
        &state,
        super::mount::TrackingRequest {
            enabled: false,
            overriding: false,
        },
        false,
    )
    .await
    .unwrap_err();
    assert_eq!(err.status.as_u16(), 412);
    // ...but the abort still reaches the out-of-band path (D19).
    let aborted = super::mount::abort_value(&state, false).await.unwrap();
    assert_eq!(aborted["abort_path_ran"], true);
    assert_eq!(fake.abort_path().calls(), 1);
    // Only a retired handle refuses the abort.
    fake.retire();
    let err = super::mount::abort_value(&state, false).await.unwrap_err();
    assert_eq!(err.status.as_u16(), 412);
    state.mount.lock().unwrap().take();
    cleanup(directory, state).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_sim_focuser_moves_through_the_gate_within_its_travel_bounds() {
    let (directory, state) = state().await;
    let body = connect(&state, "focuser", "sim", "12345").await;
    assert_eq!(body["position"], 12345);
    let value = status(&state);
    assert_eq!(value["focuser"]["position"], 12345);
    for key in [
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
    ] {
        assert!(
            value["focuser"].get(key).is_some(),
            "focuser block lacks {key}"
        );
    }
    let accepted = super::focuser::move_absolute_value(
        &state,
        super::focuser::MoveToRequest {
            position: 12_400,
            overriding: false,
        },
        false,
    )
    .await
    .unwrap();
    assert_eq!(accepted["status"], "moving");
    // Arrived, settled and verified: the move resolved, which releases the
    // focuser's domain (a later operator move would otherwise get 409).
    wait_status(&state, "arrival", |v| {
        v["focuser"]["position"] == 12_400 && v["focuser"]["active_op"].is_null()
    })
    .await;
    assert_eq!(super::focuser::position_value(&state).unwrap(), 12_400);
    let deadline = Instant::now() + Duration::from_secs(5);
    while !state.equipment.gate().is_idle() {
        assert!(
            Instant::now() < deadline,
            "the move never released its domain"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // Manual moves obey the travel bounds too (D9): a target past the
    // configured maximum is refused before anything moves.
    let max = state.equipment_ctx.configs.focus_safety.max_position;
    let err = super::focuser::move_absolute_value(
        &state,
        super::focuser::MoveToRequest {
            position: max + 1_000,
            overriding: false,
        },
        false,
    )
    .await
    .unwrap_err();
    assert_eq!(err.status.as_u16(), 422, "{}: {}", err.code, err.message);
    let descriptor = super::focuser::descriptor_value(&state).unwrap();
    assert_eq!(descriptor["position_source"], "device");
    // The halt route works on a connected focuser.
    let halted = super::focuser::abort_value(&state, false).await.unwrap();
    assert_eq!(halted["status"], "stopped");
    cleanup(directory, state).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_sim_filter_wheel_changes_through_the_handshake() {
    let (directory, state) = state().await;
    connect(&state, "filter_wheel", "sim", "").await;
    let value = wait_status(&state, "a polled wheel", |v| {
        v["filter_wheel"]["slot"].is_number()
    })
    .await;
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
        assert!(
            value["filter_wheel"].get(key).is_some(),
            "wheel block lacks {key}"
        );
    }
    let accepted = super::filter_wheel::change_value(
        &state,
        super::filter_wheel::ChangeRequest {
            slot: Some(2),
            name: None,
            overriding: false,
        },
        false,
    )
    .unwrap();
    assert_eq!(accepted["status"], "changing");
    wait_status(&state, "slot 2 verified", |v| {
        v["filter_wheel"]["slot"] == 2 && v["filter_wheel"]["verified"] == true
    })
    .await;
    let names = event_names(&state);
    for name in ["filter_change_requested", "filter_change_ended"] {
        assert!(names.iter().any(|n| n == name), "{name}: {names:?}");
    }
    // A change needs a target.
    let err = super::filter_wheel::change_value(
        &state,
        super::filter_wheel::ChangeRequest {
            slot: None,
            name: None,
            overriding: false,
        },
        false,
    )
    .unwrap_err();
    assert_eq!(err.code, "filter_target_required");
    let descriptor = super::filter_wheel::descriptor_value(&state).unwrap();
    assert!(descriptor["slots"].as_array().unwrap().len() >= 3);
    cleanup(directory, state).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn weather_and_safety_report_readings_and_safety_is_none_without_a_monitor() {
    let (directory, state) = state().await;
    // No monitor: the combined state is `none` (§5.7 Tier 1 item 6).
    assert_eq!(super::safety::status_value(&state)["state"], "none");

    connect(&state, "weather", "sim", "").await;
    connect(&state, "safety_monitor", "sim", "").await;
    let weather = super::weather::status_value(&state).unwrap();
    assert!(weather["fields"].is_object());
    assert_eq!(weather["identity"]["kind"], "weather");
    wait_status(&state, "a safe monitor", |v| {
        v["safety_monitor"]["is_safe"] == true
    })
    .await;
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let safety = super::safety::status_value(&state);
        if safety["state"] == "safe" {
            break;
        }
        assert!(Instant::now() < deadline, "{safety}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(status(&state)["weather"].is_object());
    cleanup(directory, state).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_phd2_mock_fills_the_guiding_block() {
    use crate::equipment::testing::phd2_mock::{MockState, Phd2Mock};

    let (directory, state) = state().await;
    let mock = Phd2Mock::start_with(MockState {
        app_state: "Guiding".into(),
        pixel_scale: Some(2.0),
        ..MockState::default()
    })
    .await;
    connect(
        &state,
        "guider",
        "phd2",
        &format!("127.0.0.1:{}", mock.port()),
    )
    .await;
    let value = wait_status(&state, "guiding", |v| v["guiding"]["state"] == "guiding").await;
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
        assert!(value["guiding"].get(key).is_some(), "guiding lacks {key}");
    }
    let telemetry = crate::telemetry::telemetry_event(&state);
    assert!(
        telemetry["payload"]["guiding"].is_object(),
        "guiding is non-null"
    );
    let descriptor = super::guider::descriptor_value(&state).unwrap();
    assert_eq!(descriptor["identity"]["driver"], "phd2");
    // Disconnecting never tells PHD2 to disconnect its equipment (D16).
    assert!(eq::disconnect_device(&state, DeviceKind::Guider, false)
        .await
        .unwrap());
    assert_eq!(mock.method_count("set_connected"), 0);
    assert!(status(&state)["guiding"].is_null());
    cleanup(directory, state).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn estop_reaches_every_device_raises_a_block_and_the_operator_clears_it() {
    let (directory, state) = state().await;
    connect(&state, "mount", "sim", "").await;
    connect(&state, "focuser", "sim", "20000").await;
    let report = eq::estop_value(&state, eq::EstopBody::default(), false).await;
    assert_eq!(report["status"], "stopped", "{report}");
    let kinds: Vec<&str> = report["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["kind"].as_str().unwrap())
        .collect();
    assert!(
        kinds.contains(&"mount") && kinds.contains(&"focuser"),
        "{kinds:?}"
    );
    assert!(report["elapsed_ms"].as_u64().unwrap() < 2000);
    let blocks = eq::resume_blocks_value(&state);
    let block = &blocks["blocks"][0];
    assert_eq!(block["kind"]["kind"], "estop");
    assert_eq!(status(&state)["resume_blocks"][0]["kind"]["kind"], "estop");
    // The e-stop turned tracking off (the default).
    wait_status(&state, "tracking off", |v| v["mount"]["tracking"] == false).await;
    let id: uuid::Uuid = serde_json::from_value(block["id"].clone()).unwrap();
    let cleared = eq::clear_resume_block_value(&state, eq::ClearBlockBody { id }, false).unwrap();
    assert_eq!(cleared["status"], "cleared");
    assert_eq!(eq::resume_blocks_value(&state)["blocks"], json!([]));
    let err = eq::clear_resume_block_value(&state, eq::ClearBlockBody { id }, false).unwrap_err();
    assert_eq!(err.status.as_u16(), 404);
    // The journal tail filters by kind and continues after a seq.
    let events = eq::events_value(
        &state,
        &eq::EventsQuery {
            since_seq: 0,
            kinds: Some("estop".into()),
            limit: None,
        },
    );
    assert_eq!(events["events"].as_array().unwrap().len(), 1);
    let next = events["next_seq"].as_u64().unwrap();
    let after = eq::events_value(
        &state,
        &eq::EventsQuery {
            since_seq: next,
            kinds: Some("estop".into()),
            limit: None,
        },
    );
    assert!(after["events"].as_array().unwrap().is_empty());
    cleanup(directory, state).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn jog_and_pulse_run_on_the_direct_api_and_relay_errors_carry_codes() {
    let (directory, state) = state().await;
    connect(&state, "mount", "sim", "").await;
    wait_status(&state, "a polled mount", |v| v["mount"]["ra"].is_number()).await;
    let moved = super::mount::move_axis_value(
        &state,
        super::mount::MoveAxisRequest {
            axis: 0,
            rate_deg_s: 1.0,
        },
        false,
    )
    .await
    .unwrap();
    assert_eq!(moved["status"], "moving");
    let stopped = super::mount::move_axis_value(
        &state,
        super::mount::MoveAxisRequest {
            axis: 0,
            rate_deg_s: 0.0,
        },
        false,
    )
    .await
    .unwrap();
    assert_eq!(stopped["status"], "stopped");
    // A relayed command on a disconnected focuser fails with the same code
    // as the direct route.
    let relayed = crate::relay_commands::execute(
        state.clone(),
        &crate::relay_commands::RelayCommand {
            command_id: "no-focuser".into(),
            action: "focuser.move_absolute".into(),
            payload: json!({"position": 1}),
        },
    )
    .await
    .unwrap_err();
    assert!(relayed.starts_with("focuser_not_connected"), "{relayed}");
    cleanup(directory, state).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn native_alpaca_mount_and_focuser_connect_through_the_spine() {
    use crate::equipment::testing::alpaca_server::{ScriptedDevice, TestAlpacaServer};
    let server = TestAlpacaServer::start(vec![
        ScriptedDevice::esp32_like_mount(0),
        ScriptedDevice::platform6_focuser(0),
    ]);
    let (directory, state) = state().await;
    let option = format!("{}#device=0", server.base_url());

    // The ESP32-shaped ITelescopeV4 over native Alpaca: the identity comes
    // from the management API's UniqueID (§2.6 step 5).
    let body = connect(&state, "mount", "alpaca", &option).await;
    assert_eq!(body["status"], "connected");
    assert_eq!(body["identity"]["driver"], "alpaca");
    assert_eq!(body["identity"]["unique_id"], "test-esp32-mount-0001");
    assert!(body["identity"]["setup_url"]
        .as_str()
        .unwrap()
        .ends_with("/setup/v1/telescope/0/setup"));
    let value = wait_status(&state, "a polled Alpaca mount", |v| {
        v["mount"]["lst_h"].is_number() && v["mount"]["parked"].is_boolean()
    })
    .await;
    assert_eq!(value["mount"]["driver"], "alpaca");
    assert_eq!(value["mount"]["parked"], true, "the profile starts parked");
    assert_eq!(value["mount"]["alignment_mode"], "polar");
    assert_eq!(value["device_health"]["mount"]["state"], "ready");

    // The Platform 6 focuser (no Connect/DeviceState, the ESP32 focuser
    // shape), moved through the gate.
    let body = connect(&state, "focuser", "alpaca", &option).await;
    assert_eq!(body["status"], "connected");
    assert_eq!(body["identity"]["driver"], "alpaca");
    wait_status(&state, "a polled Alpaca focuser", |v| {
        v["focuser"]["position"] == 10_000
    })
    .await;
    let accepted = super::focuser::move_absolute_value(
        &state,
        super::focuser::MoveToRequest {
            position: 10_200,
            overriding: false,
        },
        false,
    )
    .await
    .unwrap();
    assert_eq!(accepted["status"], "moving");
    wait_status(&state, "the Alpaca focuser arrival", |v| {
        v["focuser"]["position"] == 10_200 && v["focuser"]["active_op"].is_null()
    })
    .await;
    wait_idle(&state).await;

    // Both reached the device over HTTP.
    let requests = server.requests();
    assert!(requests.iter().any(|r| r.device_type == "telescope"));
    assert!(requests
        .iter()
        .any(|r| r.device_type == "focuser" && r.member == "move"));
    cleanup(directory, state).await;
    drop(server);
}

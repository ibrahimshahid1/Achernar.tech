//! Lifecycle tests use simulated devices behind the equipment layer's
//! workers and no native hardware handles. The invariants are the ones the
//! OLS-era tests protected, now for handles (spec §5.3 WP1B-1): telemetry
//! never blocks on a device slot, a late probe never revives a device, a
//! retired handle gives 412, a replacement is safe, and relay connect and
//! disconnect use the registry.
use super::*;
use crate::config::{CameraConfig, FocusConfig};
use crate::relay_commands::{execute, RelayCommand};
use crate::storage::{
    capture_store::CaptureStore, frames_repo::FrameStore, local_store::LocalStore,
};
use axum::{extract::State, http::StatusCode, Json};
use serde_json::json;
use std::path::PathBuf;

async fn state() -> (PathBuf, AppState) {
    let directory =
        std::env::temp_dir().join(format!("achernar-health-test-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&directory).unwrap();
    let store = LocalStore::open(directory.join("state.db").to_str().unwrap())
        .await
        .unwrap();
    let mut focus = FocusConfig::default();
    focus.models.quality_model_path =
        Some(directory.join("absent-fqe.onnx").to_string_lossy().into());
    let state = AppState::new(
        focus,
        CameraConfig::default(),
        store,
        FrameStore::new(directory.join("previews")),
        CaptureStore::new(directory.join("captures")),
        "health-fixture-rig".into(),
        directory.join("data"),
        None,
    );
    (directory, state)
}

async fn cleanup(directory: PathBuf, state: AppState) {
    for kind in crate::api::equipment::AUXILIARY_KINDS {
        let _ = crate::api::equipment::disconnect_device(&state, kind, true).await;
    }
    let pool = state.local_store.pool().clone();
    drop(state);
    pool.close().await;
    std::fs::remove_dir_all(directory).unwrap();
}

/// Connect the daemon's legacy simulator focuser behind its worker; returns
/// the registry connection id and the simulator, to inspect it without the
/// driver.
async fn install_focuser(
    state: &AppState,
    position: i32,
) -> (String, Arc<Mutex<crate::hardware::focuser::SimFocuser>>) {
    let driver = crate::equipment::legacy::sim_focuser::LegacySimFocuser::new(
        position,
        state.equipment.clock(),
    );
    let sim = driver.shared_sim();
    let body = crate::api::equipment::install_focuser(state, Box::new(driver), "sim", "")
        .await
        .unwrap();
    (body["connection_id"].as_str().unwrap().to_string(), sim)
}

fn sim_position(sim: &Arc<Mutex<crate::hardware::focuser::SimFocuser>>) -> i32 {
    sim.lock().unwrap().position().unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn old_probe_cannot_retire_a_replacement_handle() {
    let (directory, state) = state().await;
    let (id, sim) = install_focuser(&state, 23456).await;
    // A late probe and retire for an old connection leave the replacement.
    probe_device(&state, DeviceKind::Focuser, "retired-old-connection");
    retire_auxiliary(&state, DeviceKind::Focuser);
    let handle = state.focuser.lock().unwrap().clone().unwrap();
    assert!(!handle.is_retired());
    assert_eq!(handle.snapshot().logical_position, Some(23456));
    assert_eq!(sim_position(&sim), 23456);
    assert_eq!(
        state
            .devices
            .read()
            .unwrap()
            .connection_id(DeviceKind::Focuser)
            .as_deref(),
        Some(id.as_str())
    );
    assert_eq!(
        crate::telemetry::equipment_status(&state)["focuser"]["position"],
        23456
    );
    // A late projection from an old connection id never revives anything.
    let lost = crate::equipment::types::HealthState {
        state: crate::equipment::types::LinkState::Lost,
        ..crate::equipment::types::HealthState::default()
    };
    assert!(!state.devices.write().unwrap().lost(
        DeviceKind::Focuser,
        "retired-old-connection",
        &lost
    ));
    assert_eq!(
        crate::telemetry::equipment_status(&state)["focuser"]["position"],
        23456
    );
    cleanup(directory, state).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retired_public_device_is_hidden_and_commands_reject_retained_handle() {
    let (directory, state) = state().await;
    let (id, sim) = install_focuser(&state, 10000).await;
    lose_device(
        &state,
        DeviceKind::Focuser,
        &id,
        "fixture transport failure".into(),
    );
    // The handle outlives the public loss until it is retired; that must not
    // preserve public liveness.
    assert!(state.focuser.lock().unwrap().is_some());
    let status = crate::telemetry::equipment_status(&state);
    assert_eq!(status["focuser"], Value::Null);
    assert_eq!(status["connected"], false);
    assert_eq!(
        status["device_health"]["focuser"]["last_error"],
        "fixture transport failure"
    );
    assert_eq!(
        status["device_health"]["focuser"]["reason"], "fixture transport failure",
        "reason mirrors last_error (§2.8)"
    );
    assert!(state.focus_abort_requested.load(Ordering::Acquire));
    let direct = crate::api::focuser::move_absolute(
        State(state.clone()),
        Json(crate::api::focuser::MoveToRequest {
            position: 20000,
            overriding: false,
        }),
    )
    .await;
    assert_eq!(direct.unwrap_err().status, StatusCode::PRECONDITION_FAILED);
    let relayed = execute(
        state.clone(),
        &RelayCommand {
            command_id: "retired-command".into(),
            action: "focuser.move_absolute".into(),
            payload: json!({"position":20000}),
        },
    )
    .await;
    assert!(relayed.is_err());
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert_eq!(sim_position(&sim), 10000);
    // Retiring releases the slot and tells the worker to disconnect.
    let handle = state.focuser.lock().unwrap().clone().unwrap();
    retire_auxiliary(&state, DeviceKind::Focuser);
    assert!(state.focuser.lock().unwrap().is_none());
    assert!(state.focuser_name.lock().unwrap().is_none());
    let deadline = Instant::now() + Duration::from_secs(5);
    while !handle.is_retired() {
        assert!(Instant::now() < deadline, "the worker was never retired");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // A retired handle refuses even its abort (412).
    assert_eq!(
        crate::api::focuser::abort(State(state.clone()))
            .await
            .unwrap_err()
            .status,
        StatusCode::PRECONDITION_FAILED
    );
    cleanup(directory, state).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn relay_connect_and_disconnect_use_the_authoritative_device_registry() {
    let (directory, state) = state().await;
    execute(
        state.clone(),
        &RelayCommand {
            command_id: "connect-fixture".into(),
            action: "equipment.connect".into(),
            payload: json!({"equipment_type":"focuser","driver":"sim","option":"12345"}),
        },
    )
    .await
    .unwrap();
    assert!(state
        .devices
        .read()
        .unwrap()
        .connection_id(DeviceKind::Focuser)
        .is_some());
    let status = crate::telemetry::equipment_status(&state);
    assert_eq!(status["focuser"]["position"], 12345);
    assert_eq!(status["connected"], true);
    execute(
        state.clone(),
        &RelayCommand {
            command_id: "disconnect-fixture".into(),
            action: "equipment.disconnect".into(),
            payload: json!({"equipment_type":"focuser"}),
        },
    )
    .await
    .unwrap();
    assert_eq!(
        crate::telemetry::equipment_status(&state)["focuser"],
        Value::Null
    );
    assert!(state.focuser.lock().unwrap().is_none());
    cleanup(directory, state).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn telemetry_does_not_wait_for_any_native_device_slot() {
    let (directory, state) = state().await;
    install_focuser(&state, 34567).await;
    // A plain thread holds every native slot lock (no guard is held across
    // an await) until the test releases it.
    let (held_tx, held_rx) = tokio::sync::oneshot::channel::<()>();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let holder_state = state.clone();
    let holder = std::thread::spawn(move || {
        let _mount = holder_state.mount.lock().unwrap();
        let _focuser = holder_state.focuser.lock().unwrap();
        let _wheel = holder_state.filter_wheel.lock().unwrap();
        let _ = held_tx.send(());
        let _ = release_rx.recv_timeout(Duration::from_secs(10));
    });
    held_rx.await.unwrap();
    let worker_state = state.clone();
    let mut job =
        tokio::task::spawn_blocking(move || crate::telemetry::equipment_status(&worker_state));
    let result = tokio::time::timeout(Duration::from_millis(200), &mut job).await;
    // Always release locks before asserting, including a failing regression,
    // so a blocked worker cannot hang the test runtime during shutdown.
    let _ = release_tx.send(());
    tokio::task::spawn_blocking(move || holder.join())
        .await
        .unwrap()
        .unwrap();
    let timed_out = result.is_err();
    let status = match result {
        Ok(value) => value.unwrap(),
        Err(_) => job.await.unwrap(),
    };
    assert!(!timed_out, "Telemetry waited for a native device slot");
    assert_eq!(status["focuser"]["position"], 34567);
    cleanup(directory, state).await;
}

#[tokio::test]
async fn filter_wheel_snapshot_participates_in_all_hardware_liveness() {
    let (directory, state) = state().await;
    state.devices.write().unwrap().connected(
        DeviceKind::FilterWheel,
        "wheel-fixture".into(),
        Some(json!({"connected":true,"name":"fixture wheel","position":2})),
    );
    let status = crate::telemetry::equipment_status(&state);
    assert_eq!(status["connected"], true);
    assert_eq!(status["filter_wheel"]["position"], 2);
    assert_eq!(
        crate::telemetry::telemetry_event(&state)["payload"]["filter_wheel"]["position"],
        2
    );
    lose_device(
        &state,
        DeviceKind::FilterWheel,
        "wheel-fixture",
        "fixture loss".into(),
    );
    assert_eq!(
        crate::telemetry::equipment_status(&state)["filter_wheel"],
        Value::Null
    );
    cleanup(directory, state).await;
}

#[tokio::test]
async fn camera_loss_clears_pixels_and_diagnostics_without_native_teardown() {
    use crate::hardware::camera::FrameData;
    use crate::hardware::ols_ffi::{OlsBayerType, OlsStreamType};
    use crate::state::CameraActivity;
    use crate::vision::observed_frame::ObservedFrame;
    use crate::vision::shared_observation::SharedObservation;

    let (directory, state) = state().await;
    let mut updates = state.device_updates.subscribe();
    let generation = state.current_camera_generation();
    let frame = FrameData {
        stream_generation: generation,
        format: OlsStreamType::Rgb24,
        bayer: OlsBayerType::Na,
        frame_counter: 7,
        unix_timestamp: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
        width: 2,
        height: 2,
        data: vec![128; 12],
        source: None,
        saturation_adu: None,
        sensor_descriptor: None,
    };
    let mut observation = SharedObservation::empty_for_test();
    observation.hfr_px = Some(2.5);
    observation.star_count = 4;
    *state.latest_frame.lock().unwrap() = Some(frame.clone());
    *state.vision_observation.write().unwrap() = Some(observation.clone());
    *state.last_observed_frame.write().unwrap() = Some(ObservedFrame {
        frame_counter: frame.frame_counter,
        capture_unix_time: frame.unix_timestamp,
        frame: Arc::new(frame),
        observation,
    });
    *state.streaming.lock().unwrap() = true;
    *state.camera_runtime.write().unwrap() = CameraRuntimeState {
        connection_id: Some("old-camera".into()),
        camera_name: Some("old camera name".into()),
        activity: CameraActivity::Streaming,
        streaming: true,
        ..CameraRuntimeState::default()
    };
    state
        .devices
        .write()
        .unwrap()
        .connected(DeviceKind::Camera, "old-camera".into(), None);
    assert!(crate::telemetry::equipment_status(&state)["camera"].is_object());
    let direct = crate::api::camera::preview_snapshot_value(&state).unwrap();
    assert_eq!(direct["connection_id"], "old-camera");
    assert_eq!(direct["stream_generation"], generation);
    let relay = execute(
        state.clone(),
        &RelayCommand {
            command_id: "snapshot-fixture".into(),
            action: "camera.snapshot".into(),
            payload: Value::Null,
        },
    )
    .await
    .unwrap();
    assert_eq!(relay, direct);

    assert!(state
        .camera_loss_signals()
        .invalidate("old-camera", "fixture physical transport loss".into()));
    assert_eq!(state.current_camera_generation(), generation + 1);
    assert!(state.latest_frame.lock().unwrap().is_none());
    assert!(state.vision_observation.read().unwrap().is_none());
    assert!(state.last_observed_frame.read().unwrap().is_none());
    assert!(!*state.streaming.lock().unwrap());
    assert!(state.focus_abort_requested.load(Ordering::Acquire));
    let runtime = state.camera_runtime.read().unwrap().clone();
    assert!(runtime.connection_id.is_none());
    assert!(runtime.camera_name.is_none());
    assert_eq!(runtime.activity, CameraActivity::Disconnected);
    assert_eq!(
        runtime.last_error.as_deref(),
        Some("fixture physical transport loss")
    );
    let event = crate::telemetry::telemetry_event(&state);
    assert_eq!(event["payload"]["camera"], Value::Null);
    assert_eq!(event["payload"]["ai"]["focus_hfr"], Value::Null);
    assert_eq!(event["payload"]["ai"]["star_count"], Value::Null);
    assert_eq!(
        crate::api::camera::preview_snapshot_value(&state).unwrap_err(),
        StatusCode::PRECONDITION_FAILED
    );
    assert!(execute(
        state.clone(),
        &RelayCommand {
            command_id: "retired-snapshot-fixture".into(),
            action: "camera.snapshot".into(),
            payload: Value::Null,
        },
    )
    .await
    .is_err());
    assert!(updates.try_recv().is_ok());

    // A callback from a destroyed old camera cannot erase a replacement.
    *state.camera_runtime.write().unwrap() = CameraRuntimeState {
        connection_id: Some("new-camera".into()),
        camera_name: Some("new camera name".into()),
        activity: CameraActivity::Idle,
        ..CameraRuntimeState::default()
    };
    state
        .devices
        .write()
        .unwrap()
        .connected(DeviceKind::Camera, "new-camera".into(), None);
    assert!(!state
        .camera_loss_signals()
        .invalidate("old-camera", "late callback".into()));
    assert_eq!(state.current_camera_generation(), generation + 1);
    assert_eq!(
        crate::telemetry::equipment_status(&state)["camera"]["name"],
        "new camera name"
    );
    assert!(matches!(
        updates.try_recv(),
        Err(tokio::sync::broadcast::error::TryRecvError::Empty)
    ));
    cleanup(directory, state).await;
}

#[tokio::test]
async fn snapshot_rejects_stale_future_wrong_generation_and_stopped_frames() {
    use crate::hardware::camera::FrameData;
    use crate::hardware::ols_ffi::{OlsBayerType, OlsStreamType};
    use crate::state::CameraActivity;

    let (directory, state) = state().await;
    *state.camera_runtime.write().unwrap() = CameraRuntimeState {
        connection_id: Some("snapshot-camera".into()),
        camera_name: Some("fixture preview".into()),
        activity: CameraActivity::Streaming,
        streaming: true,
        ..CameraRuntimeState::default()
    };
    state
        .devices
        .write()
        .unwrap()
        .connected(DeviceKind::Camera, "snapshot-camera".into(), None);
    let fresh = FrameData {
        stream_generation: state.current_camera_generation(),
        format: OlsStreamType::Rgb24,
        bayer: OlsBayerType::Na,
        frame_counter: 1,
        unix_timestamp: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
        width: 2,
        height: 2,
        data: vec![128; 12],
        source: None,
        saturation_adu: None,
        sensor_descriptor: None,
    };
    let mut variants = vec![];
    for offset in [-6.0, 2.0, f64::NAN] {
        let mut frame = fresh.clone();
        frame.unix_timestamp += offset;
        variants.push(frame);
    }
    let mut retired_generation = fresh.clone();
    retired_generation.stream_generation += 1;
    variants.push(retired_generation);
    let mut empty_geometry = fresh.clone();
    empty_geometry.width = 0;
    variants.push(empty_geometry);
    for frame in variants {
        *state.latest_frame.lock().unwrap() = Some(frame);
        assert_eq!(
            crate::api::camera::preview_snapshot_value(&state).unwrap_err(),
            StatusCode::NOT_FOUND
        );
    }
    *state.latest_frame.lock().unwrap() = Some(fresh);
    state.camera_runtime.write().unwrap().streaming = false;
    assert_eq!(
        crate::api::camera::preview_snapshot_value(&state).unwrap_err(),
        StatusCode::PRECONDITION_FAILED
    );
    state.camera_runtime.write().unwrap().streaming = true;
    state.devices.write().unwrap().disconnected(
        DeviceKind::Camera,
        Some("snapshot-camera"),
        Some("fixture registry loss".into()),
    );
    assert_eq!(
        crate::api::camera::preview_snapshot_value(&state).unwrap_err(),
        StatusCode::NOT_FOUND
    );
    cleanup(directory, state).await;
}

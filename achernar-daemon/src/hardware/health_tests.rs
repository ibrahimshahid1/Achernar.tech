//! Lifecycle tests use simulated focuser values and no native hardware handles.
use super::*;
use crate::config::{CameraConfig, FocusConfig};
use crate::hardware::focuser::{ConnectedFocuser, SimFocuser};
use crate::relay_commands::{execute, RelayCommand};
use crate::storage::{
    capture_store::CaptureStore, frames_repo::FrameStore, local_store::LocalStore,
};
use axum::{extract::State, http::StatusCode, Json};
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
    let pool = state.local_store.pool().clone();
    drop(state);
    pool.close().await;
    std::fs::remove_dir_all(directory).unwrap();
}

fn install_focuser(state: &AppState, id: &str, position: i32) {
    let mut slot = state.focuser.lock().unwrap();
    let focuser = ConnectedFocuser::Simulator(SimFocuser::new(position));
    let snapshot = focuser_snapshot("fixture focuser".into(), &focuser.health_status().unwrap());
    *slot = Some(focuser);
    *state.focuser_name.lock().unwrap() = Some("fixture focuser".into());
    state
        .devices
        .write()
        .unwrap()
        .connected(DeviceKind::Focuser, id.into(), Some(snapshot));
}

#[tokio::test]
async fn old_probe_cannot_retire_a_replacement_handle() {
    let (directory, state) = state().await;
    install_focuser(&state, "replacement", 23456);
    probe_device(&state, DeviceKind::Focuser, "retired-old-connection");
    retire_auxiliary(&state, DeviceKind::Focuser);
    assert_eq!(
        state
            .focuser
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .position()
            .unwrap(),
        23456
    );
    assert_eq!(
        state
            .devices
            .read()
            .unwrap()
            .connection_id(DeviceKind::Focuser)
            .as_deref(),
        Some("replacement")
    );
    assert_eq!(
        crate::telemetry::equipment_status(&state)["focuser"]["position"],
        23456
    );
    cleanup(directory, state).await;
}

#[tokio::test]
async fn retired_public_device_is_hidden_and_commands_reject_retained_handle() {
    let (directory, state) = state().await;
    install_focuser(&state, "lost", 10000);
    lose_device(
        &state,
        DeviceKind::Focuser,
        "lost",
        "fixture transport failure".into(),
    );
    // Native destruction can be delayed; that must not preserve public liveness.
    assert!(state.focuser.lock().unwrap().is_some());
    let status = crate::telemetry::equipment_status(&state);
    assert_eq!(status["focuser"], Value::Null);
    assert_eq!(status["connected"], false);
    assert_eq!(
        status["device_health"]["focuser"]["last_error"],
        "fixture transport failure"
    );
    assert!(state.focus_abort_requested.load(Ordering::Acquire));
    let direct = crate::api::focuser::move_absolute(
        State(state.clone()),
        Json(crate::api::focuser::MoveToRequest { position: 20000 }),
    )
    .await;
    assert_eq!(direct.unwrap_err(), StatusCode::PRECONDITION_FAILED);
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
    assert_eq!(
        state
            .focuser
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .position()
            .unwrap(),
        10000
    );
    retire_auxiliary(&state, DeviceKind::Focuser);
    assert!(state.focuser.lock().unwrap().is_none());
    assert!(state.focuser_name.lock().unwrap().is_none());
    cleanup(directory, state).await;
}

#[tokio::test]
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
    install_focuser(&state, "cached", 34567);
    let mount = state.mount.lock().unwrap();
    let focuser = state.focuser.lock().unwrap();
    let wheel = state.filter_wheel.lock().unwrap();
    let worker_state = state.clone();
    let mut job =
        tokio::task::spawn_blocking(move || crate::telemetry::equipment_status(&worker_state));
    let result = tokio::time::timeout(Duration::from_millis(200), &mut job).await;
    // Always release locks before asserting, including a failing regression,
    // so a blocked worker cannot hang the test runtime during shutdown.
    drop(wheel);
    drop(focuser);
    drop(mount);
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

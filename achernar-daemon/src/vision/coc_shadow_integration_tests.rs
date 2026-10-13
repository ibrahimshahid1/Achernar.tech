//! Local worker/API integration with a real trained ONNX graph and injected
//! rendered pixels. No SDK camera, capture or control command is issued.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::api::coc::{configure_value, freeze_value, save_value, status_value};
use crate::config::{CameraConfig, FocusConfig};
use crate::hardware::camera::FrameData;
use crate::hardware::ols_ffi::{OlsBayerType, OlsStreamType};
use crate::state::{AppState, CameraActivity};
use crate::storage::capture_store::CaptureStore;
use crate::storage::frames_repo::FrameStore;
use crate::storage::local_store::LocalStore;
use crate::vision::coc_shadow::spawn;

async fn test_state() -> (PathBuf, AppState) {
    let directory =
        std::env::temp_dir().join(format!("achernar-coc-worker-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&directory).unwrap();
    let database = directory.join("achernar.db");
    let local_store = LocalStore::open(database.to_str().unwrap()).await.unwrap();
    let mut focus = FocusConfig::default();
    focus.models.quality_model_path =
        Some(directory.join("absent-fqe.onnx").to_string_lossy().into());
    // AppState constructs the production service from externally supplied
    // ACHERNAR_COC_BUNDLE. Tests never mutate process-wide environment.
    let state = AppState::new(
        focus,
        CameraConfig::default(),
        local_store,
        FrameStore::new(directory.join("preview")),
        CaptureStore::new(directory.join("captures")),
        "injected-preview-review-rig".into(),
        directory.join("data"),
        None,
    );
    (directory, state)
}

fn inject_preview(state: &AppState, rgb: &[u8], counter: i32) {
    let _barrier = state.camera_observation_barrier.lock().unwrap();
    *state.latest_frame.lock().unwrap() = Some(FrameData {
        stream_generation: state.current_camera_generation(),
        format: OlsStreamType::Rgb24,
        bayer: OlsBayerType::Na,
        frame_counter: counter,
        unix_timestamp: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
        width: 128,
        height: 128,
        data: rgb.to_vec(),
        source: None,
        saturation_adu: None,
        sensor_descriptor: None,
    });
    let mut runtime = state.camera_runtime.write().unwrap();
    runtime.connection_id = Some("injected-rendered-preview".into());
    runtime.streaming = true;
    runtime.activity = CameraActivity::Streaming;
}

async fn wait_for_sample(state: &AppState, roi_expected: bool, counter: i32) -> Value {
    let start = Instant::now();
    loop {
        let status = status_value(state);
        if status["status"] == "ready"
            && status["latest"].is_object()
            && status["latest"]["roi"].is_object() == roi_expected
            && status["latest"]["frame_counter"] == counter
        {
            return status;
        }
        assert!(
            start.elapsed() < Duration::from_secs(4),
            "No fresh model output: {status}"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires ACHERNAR_COC_BUNDLE, COC_TEST_BUNDLE and COC_TEST_RGB pointing to the local real-model fixture"]
async fn real_worker_api_roi_durable_save_and_disconnect() {
    let expected_bundle = std::env::var("COC_TEST_BUNDLE").expect("COC_TEST_BUNDLE");
    let service_bundle = std::env::var("ACHERNAR_COC_BUNDLE").expect("ACHERNAR_COC_BUNDLE");
    assert_eq!(
        Path::new(&expected_bundle).canonicalize().unwrap(),
        Path::new(&service_bundle).canonicalize().unwrap()
    );
    let rgb = std::fs::read(std::env::var("COC_TEST_RGB").expect("COC_TEST_RGB")).unwrap();
    assert_eq!(rgb.len(), 128 * 128 * 3);
    let (directory, state) = test_state().await;
    let mut control_events = state.vision_updates.subscribe();
    assert_eq!(status_value(&state)["status"], "disabled");
    assert!(status_value(&state)["models"]
        .as_array()
        .unwrap()
        .iter()
        .any(|m| m["id"] == "teacher" && m["loaded"] == true));
    inject_preview(&state, &rgb, 1);
    configure_value(
        &state,
        serde_json::from_value(json!({"enabled":true,"model":"teacher","roi":null,"context":null}))
            .unwrap(),
    )
    .unwrap();
    let worker = spawn(state.clone());
    let status = wait_for_sample(&state, false, 1).await;
    assert_eq!(
        status["latest"]["summary"]["reason"],
        "valid_region_required"
    );
    assert_eq!(
        status["latest"]["summary"]["cloud_area_fraction_of_roi"],
        Value::Null
    );
    let context = status["latest"]["context"].clone();
    let mut wrong_context = context.clone();
    wrong_context["generation"] = json!(state.current_camera_generation() + 1);
    let roi = json!({"left":0.1,"top":0.2,"right":0.9,"bottom":0.8});
    assert!(configure_value(
        &state,
        serde_json::from_value(
            json!({"enabled":true,"model":"teacher","roi":roi,"context":wrong_context})
        )
        .unwrap()
    )
    .is_err());
    inject_preview(&state, &rgb, 2);
    configure_value(
        &state,
        serde_json::from_value(
            json!({"enabled":true,"model":"teacher","roi":roi,"context":context}),
        )
        .unwrap(),
    )
    .unwrap();
    let status = wait_for_sample(&state, true, 2).await;
    assert_eq!(status["roi_valid_for_current_camera"], true);
    assert_eq!(status["latest"]["summary"]["runtime_eligible"], false);
    assert_eq!(status["latest"]["summary"]["probability_calibrated"], false);
    assert!(
        status["latest"]["summary"]["valid_region_pixels"]
            .as_u64()
            .unwrap()
            > 0
    );
    let sample_id = status["latest"]["sample_id"].as_str().unwrap().to_string();
    let frozen = freeze_value(
        &state,
        serde_json::from_value(json!({"sample_id":sample_id})).unwrap(),
    )
    .unwrap();
    assert_eq!(frozen["sample_id"], sample_id);
    // Updating the live preview must not retarget the pinned review/notes.
    inject_preview(&state, &vec![0; rgb.len()], 3);
    let newer = wait_for_sample(&state, true, 3).await;
    assert_ne!(newer["latest"]["sample_id"], sample_id);
    assert_eq!(
        newer["latest"]["summary"]["reason"],
        "uninformative_rgb_input"
    );
    let saved = save_value(state.clone(), serde_json::from_value(json!({"sample_id":sample_id,"notes":"Injected preview integration fixture, not physical sky validation."})).unwrap()).await.unwrap();
    let saved_path = Path::new(saved["path"].as_str().unwrap());
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(saved_path.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["sample_id"], sample_id);
    assert_eq!(manifest["training_approved"], false);
    assert_eq!(manifest["runtime_eligible"], false);
    assert_eq!(manifest["sample"]["context"], status["latest"]["context"]);
    assert_eq!(manifest["sample"]["roi"], roi);
    assert!(manifest["files"]["independent_roi.png"].is_object());
    assert!(manifest["files"]["source_preview.jpg"].is_null());
    for (name, identity) in manifest["files"].as_object().unwrap() {
        let path = saved_path.join(name);
        assert_eq!(
            super::sha256_file(&path).unwrap(),
            identity["sha256"].as_str().unwrap()
        );
        assert_eq!(
            std::fs::metadata(path).unwrap().len(),
            identity["bytes"].as_u64().unwrap()
        );
    }
    assert_eq!(
        std::fs::read(saved_path.join("logits.f32le"))
            .unwrap()
            .len(),
        3 * 128 * 128 * 4
    );
    assert_eq!(
        image::open(saved_path.join("source_rgb.png"))
            .unwrap()
            .to_rgb8()
            .into_raw(),
        rgb
    );
    assert!(save_value(
        state.clone(),
        serde_json::from_value(json!({"sample_id":sample_id,"notes":"duplicate"})).unwrap()
    )
    .await
    .is_err());
    assert!(state.camera.lock().unwrap().is_none());
    assert!(state.mount.lock().unwrap().is_none());
    assert!(state.focuser.lock().unwrap().is_none());
    assert!(state.vision_observation.read().unwrap().is_none());
    assert!(state.last_observed_frame.read().unwrap().is_none());
    assert!(matches!(
        control_events.try_recv(),
        Err(tokio::sync::broadcast::error::TryRecvError::Empty)
    ));
    let (captures,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM frames")
        .fetch_one(state.local_store.pool())
        .await
        .unwrap();
    assert_eq!(captures, 0);
    state.clear_camera_observations();
    {
        let mut runtime = state.camera_runtime.write().unwrap();
        runtime.connection_id = None;
        runtime.streaming = false;
        runtime.activity = CameraActivity::Disconnected;
    }
    let disconnected = status_value(&state);
    assert_eq!(disconnected["latest"], Value::Null);
    assert_eq!(disconnected["roi_valid_for_current_camera"], false);
    assert!(save_value(
        state.clone(),
        serde_json::from_value(json!({"sample_id":sample_id,"notes":"disconnected"})).unwrap()
    )
    .await
    .is_err());
    worker.abort();
    let _ = worker.await;
    let pool = state.local_store.pool().clone();
    drop(state);
    pool.close().await;
    std::fs::remove_dir_all(directory).unwrap();
}

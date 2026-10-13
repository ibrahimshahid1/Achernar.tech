//! API preflight checks use the production state and storage constructors.
//! No camera is attached: reaching camera lookup is observable as HTTP 412,
//! so HTTP 503 while archive capacity is reserved proves earlier rejection.

use std::path::PathBuf;

use axum::http::StatusCode;

use super::camera::{perform_capture, CaptureParams};
use crate::config::{CameraConfig, FocusConfig};
use crate::state::{AppState, CameraActivity};
use crate::storage::capture_archive::ArchiveReservation;
use crate::storage::capture_store::CaptureStore;
use crate::storage::frames_repo::FrameStore;
use crate::storage::local_store::LocalStore;

async fn test_state() -> (PathBuf, AppState) {
    let directory =
        std::env::temp_dir().join(format!("achernar-api-preflight-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&directory).unwrap();
    let database = directory.join("achernar.db");
    let local_store = LocalStore::open(database.to_str().unwrap()).await.unwrap();
    let mut focus = FocusConfig::default();
    // Override model discovery without changing process-wide environment or
    // opening an installed ONNX model during this storage/API test.
    focus.models.quality_model_path = Some(
        directory
            .join("absent-model.onnx")
            .to_string_lossy()
            .into_owned(),
    );
    let state = AppState::new(
        focus,
        CameraConfig::default(),
        local_store,
        FrameStore::new(directory.join("preview")),
        CaptureStore::new(directory.join("captures")),
        "preflight-rig".to_string(),
        directory.join("data"),
        None,
    );
    (directory, state)
}

fn fill_archive(state: &AppState) -> Vec<ArchiveReservation> {
    let mut reservations = Vec::new();
    while let Ok(reservation) = state.capture_archive.reserve() {
        reservations.push(reservation);
    }
    assert!(
        !reservations.is_empty(),
        "worker must initially have capacity"
    );
    reservations
}

async fn assert_no_archived_exposure(state: &AppState) {
    let (frames,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM frames")
        .fetch_one(state.local_store.pool())
        .await
        .unwrap();
    let (assets,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM frame_assets")
        .fetch_one(state.local_store.pool())
        .await
        .unwrap();
    assert_eq!(frames, 0);
    assert_eq!(assets, 0);
    assert!(!state.capture_store.root().exists());
    assert!(state.camera.lock().unwrap().is_none());
    let runtime = state.camera_runtime.read().unwrap();
    assert_eq!(runtime.activity, CameraActivity::Disconnected);
    assert!(runtime.operation_id.is_none());
    assert!(runtime.exposure_started_at.is_none());
}

async fn clean_up(directory: PathBuf, state: AppState) {
    let pool = state.local_store.pool().clone();
    drop(state);
    pool.close().await;
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn full_archive_rejects_before_camera_lookup_and_records_failed_attempt() {
    let (directory, state) = test_state().await;
    let reservations = fill_archive(&state);
    let params: CaptureParams = serde_json::from_value(serde_json::json!({
        "exposure": 1.0,
        "retention_policy": "retain_on_camera"
    }))
    .unwrap();

    let error = perform_capture(state.clone(), params.clone())
        .await
        .unwrap_err();
    assert_eq!(error.status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(error.message.contains("queue is full"), "{error:?}");
    let (status, detail): (String, String) =
        sqlx::query_as("SELECT status, error FROM capture_attempts")
            .fetch_one(state.local_store.pool())
            .await
            .unwrap();
    assert_eq!(status, "failed");
    assert!(detail.contains("queue is full"));
    assert_no_archived_exposure(&state).await;

    // Releasing unused reservations lets the identical request reach camera
    // lookup. The contrasting 412 verifies that 503 was the archival gate.
    drop(reservations);
    let error = perform_capture(state.clone(), params).await.unwrap_err();
    assert_eq!(error.status, StatusCode::PRECONDITION_FAILED);
    assert!(error.message.contains("No camera connected"), "{error:?}");
    let (failed_attempts,): (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM capture_attempts WHERE status = 'failed'")
            .fetch_one(state.local_store.pool())
            .await
            .unwrap();
    assert_eq!(failed_attempts, 2);
    assert_no_archived_exposure(&state).await;
    clean_up(directory, state).await;
}

#[tokio::test]
async fn acknowledged_deletion_is_rejected_before_archive_or_camera_and_preserves_request() {
    let (directory, state) = test_state().await;
    let reservations = fill_archive(&state);
    let params: CaptureParams = serde_json::from_value(serde_json::json!({
        "exposure": 1.0,
        "retention_policy": "delete_after_verified_local_commit",
        "retention_acknowledgement": {
            "policy": "delete_after_verified_local_commit",
            "policy_version": 1
        }
    }))
    .unwrap();

    let error = perform_capture(state.clone(), params).await.unwrap_err();
    assert_eq!(error.status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(error.message.contains("A12"), "{error:?}");
    let (status, detail, requested): (String, String, String) =
        sqlx::query_as("SELECT status, error, requested_json FROM capture_attempts")
            .fetch_one(state.local_store.pool())
            .await
            .unwrap();
    assert_eq!(status, "failed");
    assert!(detail.contains("A12"));
    let requested: serde_json::Value = serde_json::from_str(&requested).unwrap();
    assert_eq!(
        requested["retention_policy"],
        "delete_after_verified_local_commit"
    );
    assert_eq!(
        requested["retention"]["requested_policy"],
        requested["retention_policy"]
    );
    assert_eq!(
        requested["retention_acknowledgement"]["policy"],
        requested["retention_policy"]
    );
    assert_eq!(requested["retention_acknowledgement"]["policy_version"], 1);
    assert_eq!(
        requested["retention"]["effective_policy"],
        "retain_on_camera"
    );
    assert_eq!(requested["retention"]["camera_files_deleted"], false);
    assert_eq!(requested["retention"]["deletion_eligible"], false);
    assert_no_archived_exposure(&state).await;

    drop(reservations);
    clean_up(directory, state).await;
}

#[tokio::test]
async fn unknown_camera_outcome_requires_reconnection_before_new_capture() {
    let (directory, state) = test_state().await;
    {
        let mut runtime = state.camera_runtime.write().unwrap();
        runtime.connection_id = Some("uncertain-connection".into());
        runtime.activity = CameraActivity::Error;
        runtime.last_error = Some("Previous exposure did not establish event completion".into());
    }
    let params: CaptureParams =
        serde_json::from_value(serde_json::json!({"exposure": 1.0})).unwrap();
    let error = perform_capture(state.clone(), params).await.unwrap_err();
    assert_eq!(error.status, StatusCode::CONFLICT);
    assert_eq!(error.code, "camera_reconnect_required");
    assert_eq!(
        state.camera_runtime.read().unwrap().activity,
        CameraActivity::Error
    );
    let (count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM capture_attempts")
        .fetch_one(state.local_store.pool())
        .await
        .unwrap();
    assert_eq!(count, 0);
    clean_up(directory, state).await;
}

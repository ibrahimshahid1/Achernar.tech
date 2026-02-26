use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};

use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct CaptureParams {
    pub exposure: f64,
    pub gain: Option<u32>,
    pub offset: Option<u32>,
    pub binning: Option<u32>,
    pub filter: Option<String>,
    pub temperature: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct ImageData {
    pub id: String,
    pub width: u32,
    pub height: u32,
    pub metadata: ImageMetadata,
}

#[derive(Debug, Serialize)]
pub struct ImageMetadata {
    pub exposure: f64,
    pub gain: u32,
    pub temperature: f64,
    pub timestamp: String,
}

#[derive(Debug, Deserialize)]
pub struct StreamStartRequest {
    pub format_index: Option<i32>,
}

/// Start the camera stream
pub async fn start_stream(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let frame_store = state.latest_frame.clone();

    let mut cam_lock = state.camera.lock().unwrap();
    let camera = cam_lock.as_mut().ok_or_else(|| {
        tracing::error!("No camera connected");
        StatusCode::PRECONDITION_FAILED
    })?;

    // Get first format
    let formats = camera.formats().map_err(|e| {
        tracing::error!("Failed to get formats: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    if formats.is_empty() {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    let fmt = &formats[0];
    tracing::info!(
        "Starting stream: {:?} {}x{} bin={}",
        fmt.format, fmt.width, fmt.height, fmt.bin
    );

    camera.start_stream(fmt, frame_store).map_err(|e| {
        tracing::error!("Failed to start stream: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    *state.streaming.lock().unwrap() = true;

    Ok(Json(serde_json::json!({
        "status": "streaming",
        "format": format!("{:?}", fmt.format),
        "width": fmt.width,
        "height": fmt.height,
    })))
}

/// High-resolution capture (same as before but tries to use real exposure if connected)
pub async fn capture(
    State(state): State<AppState>,
    Json(params): Json<CaptureParams>,
) -> Result<Json<ImageData>, StatusCode> {
    tracing::info!("Capture requested: {} seconds", params.exposure);

    // Set exposure if camera is connected
    {
        let cam_lock = state.camera.lock().unwrap();
        if let Some(camera) = cam_lock.as_ref() {
            let exp_ms = params.exposure * 1000.0;
            let _ = camera.set_parameter(
                crate::hardware::ols_ffi::OlsCamOptionId::Exp,
                exp_ms,
            );
        }
    }

    let (width, height) = {
        let frame = state.latest_frame.lock().unwrap();
        match frame.as_ref() {
            Some(f) => (f.width as u32, f.height as u32),
            None => (640, 480),
        }
    };

    let image = ImageData {
        id: uuid::Uuid::new_v4().to_string(),
        width,
        height,
        metadata: ImageMetadata {
            exposure: params.exposure,
            gain: params.gain.unwrap_or(100),
            temperature: -10.0,
            timestamp: chrono::Utc::now().to_rfc3339(),
        },
    };

    Ok(Json(image))
}

pub async fn abort(
    State(state): State<AppState>,
) -> Result<StatusCode, StatusCode> {
    tracing::info!("Capture/stream abort requested");

    let mut cam_lock = state.camera.lock().unwrap();
    if let Some(camera) = cam_lock.as_mut() {
        let _ = camera.stop_stream();
    }
    *state.streaming.lock().unwrap() = false;

    Ok(StatusCode::OK)
}

/// MJPEG Stream placeholder
pub async fn stream_mjpeg() -> Result<String, StatusCode> {
    tracing::info!("MJPEG stream requested");
    Ok("Use /api/v1/camera/stream/start to start WebSocket stream".to_string())
}

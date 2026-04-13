pub mod camera;
pub mod mount;
pub mod focuser;
pub mod focus;
pub mod websocket;

use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};

use crate::state::AppState;

#[derive(Debug, Serialize, Deserialize)]
pub struct EquipmentStatus {
    pub connected: bool,
    pub camera: Option<CameraStatus>,
    pub mount: Option<MountStatus>,
    pub focuser: Option<FocuserStatus>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CameraStatus {
    pub name: String,
    pub temperature: f64,
    pub cooling: bool,
    pub exposing: bool,
    pub streaming: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MountStatus {
    pub name: String,
    pub ra: f64,
    pub dec: f64,
    pub tracking: bool,
    pub parked: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FocuserStatus {
    pub name: String,
    pub position: i32,
    pub moving: bool,
}

#[derive(Debug, Deserialize)]
pub struct ConnectRequest {
    /// Driver type: "sim", "alpaca", "asi", "toup", etc.
    pub driver: String,
    /// Driver-specific option (e.g., sim data directory path)
    pub option: Option<String>,
}

pub async fn get_status(
    State(state): State<AppState>,
) -> Result<Json<EquipmentStatus>, StatusCode> {
    let camera_status = {
        let cam_name = state.camera_name.lock().unwrap();
        let streaming = state.streaming.lock().unwrap();
        cam_name.as_ref().map(|name| {
            // Try to get temperature from camera
            let temperature = {
                let cam = state.camera.lock().unwrap();
                cam.as_ref()
                    .and_then(|c| {
                        c.get_parameter(
                            crate::hardware::ols_ffi::OlsCamOptionId::Temperature,
                        )
                        .ok()
                    })
                    .map(|p| p.cur_val)
                    .unwrap_or(-10.0)
            };
            CameraStatus {
                name: name.clone(),
                temperature,
                cooling: false,
                exposing: *streaming,
                streaming: *streaming,
            }
        })
    };

    let connected = camera_status.is_some();

    let status = EquipmentStatus {
        connected,
        camera: camera_status,
        mount: None,
        focuser: None,
    };

    Ok(Json(status))
}

pub async fn connect_equipment(
    State(state): State<AppState>,
    Json(req): Json<ConnectRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    tracing::info!("Connect request: driver={}, option={:?}", req.driver, req.option);

    let driver_result = match req.driver.as_str() {
        "sim" => {
            let sim_dir = req.option.unwrap_or_else(|| "./sim".to_string());
            crate::hardware::camera::OlsCameraDriver::new_sim(&sim_dir)
        }
        "asi" => crate::hardware::camera::OlsCameraDriver::new_asi(),
        "svb" => crate::hardware::camera::OlsCameraDriver::new_svb(),
        "toup" => crate::hardware::camera::OlsCameraDriver::new_toup(),
        _ => {
            tracing::warn!("Unsupported driver: {}", req.driver);
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    match driver_result {
        Ok(mut driver) => {
            // List cameras
            let cameras = driver.list_cameras().map_err(|e| {
                tracing::error!("Failed to list cameras: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;

            tracing::info!("Found cameras: {:?}", cameras);

            if cameras.is_empty() {
                return Err(StatusCode::NOT_FOUND);
            }

            // Open first camera
            let camera = driver.open_camera(0).map_err(|e| {
                tracing::error!("Failed to open camera: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;

            let cam_name = camera.name().unwrap_or_else(|_| "Unknown".to_string());

            // Store in state
            *state.camera_driver.lock().unwrap() = Some(driver);
            *state.camera_name.lock().unwrap() = Some(cam_name.clone());
            *state.camera.lock().unwrap() = Some(camera);

            Ok(Json(serde_json::json!({
                "status": "connected",
                "driver": req.driver,
                "camera": cam_name,
                "cameras_found": cameras,
            })))
        }
        Err(e) => {
            tracing::error!("Failed to create {} driver: {}", req.driver, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

pub async fn disconnect_equipment(
    State(state): State<AppState>,
) -> Result<StatusCode, StatusCode> {
    tracing::info!("Disconnect requested");

    // Stop stream if active
    {
        let mut cam = state.camera.lock().unwrap();
        if let Some(camera) = cam.as_mut() {
            let _ = camera.stop_stream();
        }
    }

    // Clear state
    *state.streaming.lock().unwrap() = false;
    *state.camera.lock().unwrap() = None;
    *state.camera_name.lock().unwrap() = None;
    *state.camera_driver.lock().unwrap() = None;

    Ok(StatusCode::OK)
}

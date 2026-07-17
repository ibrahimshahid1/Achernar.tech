pub mod camera;
pub mod focus;
pub mod focuser;
pub mod mount;
pub mod websocket;

use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};

use crate::hardware::{
    camera::OlsCameraDriver,
    capabilities::{driver_capability_report, DriverCapabilityReport},
    focuser::{ConnectedFocuser, Focuser, FocuserDriver, SimFocuser},
    mount::{Mount, MountDriver},
};
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
    /// Equipment kind: "camera", "mount", or "focuser". Defaults to "camera"
    /// for compatibility with the original camera-only endpoint.
    pub equipment_type: Option<String>,
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
            // Read real sensor temperature and cooler state from the camera.
            let (temperature, cooling) = {
                let cam = state.camera.lock().unwrap();
                let temperature = cam
                    .as_ref()
                    .and_then(|c| {
                        c.get_parameter(crate::hardware::ols_ffi::OlsCamOptionId::Temperature)
                            .ok()
                    })
                    .map(|p| p.cur_val)
                    .unwrap_or(-10.0);
                let cooling = cam
                    .as_ref()
                    .and_then(|c| {
                        c.get_parameter(crate::hardware::ols_ffi::OlsCamOptionId::CoolerOn)
                            .ok()
                    })
                    .map(|p| p.cur_val > 0.5)
                    .unwrap_or(false);
                (temperature, cooling)
            };
            CameraStatus {
                name: name.clone(),
                temperature,
                cooling,
                exposing: *streaming,
                streaming: *streaming,
            }
        })
    };

    let mount_status = {
        let mount_name = state.mount_name.lock().unwrap().clone();
        let mount = state.mount.lock().unwrap();
        mount.as_ref().map(|m| {
            let hw = m.status();
            let (ra, dec) = hw
                .position
                .map(|pos| (pos.ra, pos.dec))
                .unwrap_or((0.0, 0.0));
            MountStatus {
                name: mount_name.unwrap_or_else(|| "OpenLiveStacker mount".to_string()),
                ra,
                dec,
                tracking: hw.tracking.unwrap_or(false),
                parked: false,
            }
        })
    };

    let focuser_status = {
        let focuser_name = state.focuser_name.lock().unwrap().clone();
        let focuser = state.focuser.lock().unwrap();
        focuser.as_ref().map(|f| {
            let hw = f.status();
            FocuserStatus {
                name: focuser_name.unwrap_or_else(|| "OpenLiveStacker focuser".to_string()),
                position: hw.position.unwrap_or(0),
                moving: hw.is_moving,
            }
        })
    };

    let connected = camera_status.is_some() || mount_status.is_some() || focuser_status.is_some();

    let status = EquipmentStatus {
        connected,
        camera: camera_status,
        mount: mount_status,
        focuser: focuser_status,
    };

    Ok(Json(status))
}

pub async fn get_driver_capabilities() -> Json<DriverCapabilityReport> {
    Json(driver_capability_report())
}

pub async fn connect_equipment(
    State(state): State<AppState>,
    Json(req): Json<ConnectRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    tracing::info!(
        "Connect request: driver={}, option={:?}",
        req.driver,
        req.option
    );

    let equipment_type = req
        .equipment_type
        .as_deref()
        .unwrap_or("camera")
        .to_ascii_lowercase();

    match equipment_type.as_str() {
        "camera" => connect_camera(state, req).await,
        "mount" => connect_mount(state, req).await,
        "focuser" => connect_focuser(state, req).await,
        _ => {
            tracing::warn!("Unsupported equipment type: {}", equipment_type);
            Err(StatusCode::BAD_REQUEST)
        }
    }
}

async fn connect_camera(
    state: AppState,
    req: ConnectRequest,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let driver_result = match req.driver.as_str() {
        "sim" => {
            let sim_dir = req.option.unwrap_or_else(|| "./sim".to_string());
            OlsCameraDriver::new_sim(&sim_dir)
        }
        "alpaca" => {
            let url = req
                .option
                .unwrap_or_else(|| "http://localhost:11111".to_string());
            OlsCameraDriver::new_alpaca(&url)
        }
        "asi" => OlsCameraDriver::new_asi(),
        "svb" => OlsCameraDriver::new_svb(),
        "toup" => OlsCameraDriver::new_toup(),
        "wdir" | "watchdir" => {
            let dir = req.option.unwrap_or_else(|| "./watch".to_string());
            OlsCameraDriver::new_wdir(&dir)
        }
        "gphoto2" | "dslr" => OlsCameraDriver::new_gphoto2(),
        "uvc" | "webcam" => OlsCameraDriver::new_uvc(),
        "indi" => OlsCameraDriver::new_indi(),
        "indigo" => OlsCameraDriver::new_indigo(),
        _ => {
            tracing::warn!("Unsupported camera driver: {}", req.driver);
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

async fn connect_mount(
    state: AppState,
    req: ConnectRequest,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let driver_type = match req.driver.as_str() {
        "alpaca" => MountDriver::Alpaca,
        "indi" => MountDriver::Indi,
        "sim" | "simulator" => MountDriver::Simulator,
        _ => {
            tracing::warn!("Unsupported mount driver: {}", req.driver);
            return Err(StatusCode::BAD_REQUEST);
        }
    };
    let option = req.option.unwrap_or_else(|| "auto".to_string());

    let mut mount = Mount::new(driver_type).map_err(|e| {
        tracing::error!("Failed to create mount interface: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    mount.connect_with_option(&option).map_err(|e| {
        tracing::error!("Failed to connect mount: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let drivers = mount.list_drivers().unwrap_or_default();
    *state.mount_name.lock().unwrap() = Some(format!("{driver_type:?} mount"));
    *state.mount.lock().unwrap() = Some(mount);

    Ok(Json(serde_json::json!({
        "status": "connected",
        "equipment_type": "mount",
        "driver": req.driver,
        "option": option,
        "available_drivers": drivers,
    })))
}

async fn connect_focuser(
    state: AppState,
    req: ConnectRequest,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if matches!(req.driver.as_str(), "sim" | "simulator") {
        let position = req
            .option
            .as_deref()
            .and_then(|option| option.parse::<i32>().ok())
            .unwrap_or(10000);
        *state.focuser_name.lock().unwrap() = Some("Simulator focuser".to_string());
        *state.focuser.lock().unwrap() =
            Some(ConnectedFocuser::Simulator(SimFocuser::new(position)));

        return Ok(Json(serde_json::json!({
            "status": "connected",
            "equipment_type": "focuser",
            "driver": req.driver,
            "position": position,
            "available_drivers": [{"name": "sim", "option": "initial_position"}],
        })));
    }

    let driver_type = match req.driver.as_str() {
        "alpaca" => FocuserDriver::Alpaca,
        "indi" => FocuserDriver::Indi,
        "moonlite" => FocuserDriver::Moonlite,
        "usb" | "usb_serial" => FocuserDriver::UsbSerial,
        _ => {
            tracing::warn!("Unsupported focuser driver: {}", req.driver);
            return Err(StatusCode::BAD_REQUEST);
        }
    };
    let option = req.option.unwrap_or_else(|| "auto".to_string());

    let mut focuser = Focuser::new(driver_type).map_err(|e| {
        tracing::error!("Failed to create focuser interface: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    focuser.connect_with_option(&option).map_err(|e| {
        tracing::error!("Failed to connect focuser: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let drivers = focuser.list_drivers().unwrap_or_default();
    *state.focuser_name.lock().unwrap() = Some(format!("{driver_type:?} focuser"));
    *state.focuser.lock().unwrap() = Some(ConnectedFocuser::Hardware(focuser));

    Ok(Json(serde_json::json!({
        "status": "connected",
        "equipment_type": "focuser",
        "driver": req.driver,
        "option": option,
        "available_drivers": drivers,
    })))
}

pub async fn disconnect_equipment(State(state): State<AppState>) -> Result<StatusCode, StatusCode> {
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
    *state.mount.lock().unwrap() = None;
    *state.mount_name.lock().unwrap() = None;
    *state.focuser.lock().unwrap() = None;
    *state.focuser_name.lock().unwrap() = None;

    Ok(StatusCode::OK)
}

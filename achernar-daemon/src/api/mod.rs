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
    discovery::{discover_cameras, DiscoveryReport},
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
    /// Which camera within this driver's enumeration, as reported by
    /// `/discover`. Defaults to 0.
    ///
    /// `/discover` has always returned indexes and this handler has always
    /// ignored them, so with two cameras attached both "connect" requests
    /// opened the same device. That is exactly the guiding + imaging setup, and
    /// the failure is silent — you get a working camera, just not the one you
    /// picked.
    pub index: Option<i32>,
}

pub async fn get_status(
    State(state): State<AppState>,
) -> Result<Json<EquipmentStatus>, StatusCode> {
    let camera_status = {
        // Take the locks in the order documented on AppState — camera, then
        // camera_name, then streaming — and hold no two of them at once.
        //
        // This used to hold `camera_name` and `streaming` while reaching for
        // `camera`, while `start_stream` held `camera` while reaching for
        // `streaming`. Opposite orders, so a status poll arriving during a
        // stream start deadlocked both tasks and, with them, the daemon.
        // Copying each value out and dropping the guard removes the nesting
        // entirely rather than relying on everyone remembering the order.
        let camera_readings = {
            let cam = state.camera.lock().unwrap();
            cam.as_ref().map(|c| {
                let temperature = c
                    .get_parameter(crate::hardware::ols_ffi::OlsCamOptionId::Temperature)
                    .map(|p| p.cur_val)
                    .unwrap_or(-10.0);
                let cooling = c
                    .get_parameter(crate::hardware::ols_ffi::OlsCamOptionId::CoolerOn)
                    .map(|p| p.cur_val > 0.5)
                    .unwrap_or(false);
                (temperature, cooling)
            })
        };
        let cam_name = state.camera_name.lock().unwrap().clone();
        let streaming = *state.streaming.lock().unwrap();

        cam_name.map(|name| {
            let (temperature, cooling) = camera_readings.unwrap_or((-10.0, false));
            CameraStatus {
                name,
                temperature,
                cooling,
                exposing: streaming,
                streaming,
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

/// Scan locally-attached buses for cameras — the "Scan for devices" button.
///
/// POST rather than GET: this instantiates vendor SDK drivers and touches real
/// hardware, so it must never be prefetched, cached, or retried automatically
/// by a browser. It is an operator-initiated action with side effects.
///
/// Returns every device found plus a per-driver outcome, so the UI can
/// distinguish "no ZWO camera attached" from "the ZWO probe failed".
pub async fn discover_devices(State(state): State<AppState>) -> Json<DiscoveryReport> {
    tracing::info!("Device discovery scan requested");
    let report = discover_cameras(&state).await;
    tracing::info!(
        "Device discovery complete: {} camera(s) across {} probe(s)",
        report.cameras.len(),
        report.probes.len()
    );
    Json(report)
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

            // Open the requested camera. An out-of-range index is rejected
            // rather than clamped to 0: silently opening a different camera
            // than the operator selected is how a guider ends up taking the
            // light frames.
            let index = req.index.unwrap_or(0);
            if index < 0 || index as usize >= cameras.len() {
                tracing::warn!(
                    "Camera index {index} out of range; driver {} enumerated {} camera(s)",
                    req.driver,
                    cameras.len()
                );
                return Err(StatusCode::BAD_REQUEST);
            }

            let camera = driver.open_camera(index).map_err(|e| {
                tracing::error!("Failed to open camera {index}: {}", e);
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
                "index": index,
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

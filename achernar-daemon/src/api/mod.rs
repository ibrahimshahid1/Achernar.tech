pub mod api_error;
pub mod camera;
pub mod coc;
pub mod focus;
pub mod focuser;
pub mod mount;
pub mod plate_solve;
pub mod websocket;

#[cfg(test)]
mod capture_preflight_tests;

use axum::{body::Bytes, extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};

use crate::hardware::{
    camera::OlsCameraDriver,
    capabilities::{driver_capability_report, DriverCapabilityReport},
    discovery::{discover_equipment, DiscoveryReport, DiscoveryRequest},
    focuser::{ConnectedFocuser, Focuser, FocuserDriver, SimFocuser},
    health::{self, DeviceKind},
    mount::{Mount, MountDriver},
};
use crate::state::AppState;
use crate::state::CameraActivity;
use api_error::ApiError;

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
    pub connected: bool,
    pub status: CameraActivity,
    pub temperature: Option<f64>,
    pub cooling: Option<bool>,
    pub exposing: bool,
    pub streaming: bool,
    pub progress: Option<f64>,
    pub driver_id: Option<String>,
    pub device_index: Option<i32>,
    pub connection_id: Option<String>,
    pub last_error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MountStatus {
    pub name: String,
    pub connected: bool,
    /// `None` when the mount has not reported a position. RA 0h / Dec +0deg
    /// is a real coordinate in Pisces, so it cannot stand in for "unknown" —
    /// these were `f64` and the non-nullable type is what forced the
    /// `unwrap_or((0.0, 0.0))` that produced a plausible wrong pointing.
    pub ra: Option<f64>,
    pub dec: Option<f64>,
    pub tracking: Option<bool>,
    /// `None` always, for now: MountStatus carries no park field from the
    /// hardware layer, so the honest answer is "not measured" rather than
    /// asserting the mount is unparked.
    pub parked: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FocuserStatus {
    pub name: String,
    pub connected: bool,
    /// `None` when the encoder has not been read. Step 0 is a legitimate
    /// focuser position.
    pub position: Option<i32>,
    pub moving: bool,
    pub absolute: bool,
    pub max_range: Option<i32>,
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

#[derive(Debug, Default, Deserialize)]
pub struct DisconnectRequest {
    /// camera | mount | focuser | all. Defaults to all for compatibility with
    /// the original body-less endpoint.
    pub equipment_type: Option<String>,
}

pub async fn get_status(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // The direct endpoint and Django command relay consume one full snapshot.
    // Neither path performs blocking vendor calls while reporting status.
    Ok(Json(crate::telemetry::equipment_status(&state)))
}

pub async fn get_driver_capabilities() -> Json<DriverCapabilityReport> {
    Json(driver_capability_report())
}

/// GET /api/v1/commands — the daemon's own command registry.
///
/// The same pattern as `GET /drivers`: rather than every client hard-coding
/// what this build can do, the build says so. Django can relay anything on
/// this list without a whitelist, and the frontend can map an action to a
/// direct route without a hand-written table.
///
/// Generated from the single declaration in `relay_commands.rs` that the
/// dispatcher also matches on, so it cannot drift from what actually runs.
pub async fn get_commands() -> Json<&'static [crate::relay_commands::CommandSpec]> {
    Json(crate::relay_commands::COMMANDS)
}

/// Scan local cameras or list devices on one explicitly addressed Alpaca server.
///
/// POST rather than GET: this instantiates vendor SDK drivers and touches real
/// hardware, so it must never be prefetched, cached, or retried automatically
/// by a browser. It is an operator-initiated action with side effects.
///
/// Returns every device found plus a per-driver outcome, so the UI can
/// distinguish "no ZWO camera attached" from "the ZWO probe failed".
pub async fn discover_devices(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Json<DiscoveryReport>, ApiError> {
    // Bodyless requests retain the original camera-discovery API. Malformed
    // JSON must fail before discovery, never silently fall back to a USB scan.
    if body.len() > 16 * 1024 {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_discovery_request",
            "Discovery request exceeds 16 KiB",
        ));
    }
    let request = if body.is_empty() {
        DiscoveryRequest::default()
    } else {
        serde_json::from_slice(&body).map_err(|error| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_discovery_request",
                error.to_string(),
            )
        })?
    };
    tracing::info!("Device discovery scan requested");
    let report = discover_equipment(&state, request).await.map_err(|message| {
        ApiError::new(StatusCode::BAD_REQUEST, "invalid_discovery_request", message)
    })?;
    tracing::info!(
        "Device discovery complete: {} camera(s) across {} probe(s)",
        report.cameras.len(),
        report.probes.len()
    );
    Ok(Json(report))
}

pub async fn connect_equipment(
    State(state): State<AppState>,
    Json(req): Json<ConnectRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
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
        "mount" => connect_mount(state, req).await.map_err(|status| {
            ApiError::new(
                status,
                "mount_connect_failed",
                "Could not connect the mount",
            )
        }),
        "focuser" => connect_focuser(state, req).await.map_err(|status| {
            ApiError::new(
                status,
                "focuser_connect_failed",
                "Could not connect the focuser",
            )
        }),
        _ => {
            tracing::warn!("Unsupported equipment type: {}", equipment_type);
            Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "unsupported_equipment_type",
                format!("Unsupported equipment type '{equipment_type}'"),
            ))
        }
    }
}

pub(crate) async fn connect_camera(
    state: AppState,
    req: ConnectRequest,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _acquisition = state.acquisition.lock().await;

    if state.camera.lock().unwrap().is_some() {
        tracing::warn!("Connect rejected: a camera is already connected");
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "camera_already_connected",
            "Disconnect the current camera before connecting another one",
        ));
    }
    state.clear_camera_observations();
    {
        let mut runtime = state.camera_runtime.write().unwrap();
        *runtime = crate::state::CameraRuntimeState::default();
        runtime.activity = CameraActivity::Connecting;
        runtime.driver_id = Some(req.driver.clone());
        runtime.device_index = Some(req.index.unwrap_or(0));
    }

    let driver_id = req.driver.clone();
    let driver_option = req.option.clone();
    let requested_index = req.index.unwrap_or(0);
    type ConnectedCamera = (
        OlsCameraDriver,
        crate::hardware::camera::OlsCamera,
        String,
        Vec<String>,
        i32,
    );
    type ConnectFailure = (StatusCode, String);

    // Vendor driver construction, enumeration and open are blocking FFI. Keep
    // the acquisition ownership guard, but run the SDK work off Axum's async
    // worker so `/status` and telemetry remain responsive during a slow open.
    let worker_driver_id = driver_id.clone();
    let connect_worker = tokio::task::spawn_blocking(move || {
        let driver_result = match worker_driver_id.as_str() {
            "sim" => {
                OlsCameraDriver::new_sim(&driver_option.unwrap_or_else(|| "./sim".to_string()))
            }
            "alpaca" => OlsCameraDriver::new_alpaca(
                &driver_option.unwrap_or_else(|| "http://localhost:11111".to_string()),
            ),
            "asi" => OlsCameraDriver::new_asi(),
            "svb" => OlsCameraDriver::new_svb(),
            "toup" => OlsCameraDriver::new_toup(),
            "wdir" | "watchdir" => {
                OlsCameraDriver::new_wdir(&driver_option.unwrap_or_else(|| "./watch".to_string()))
            }
            "gphoto2" | "dslr" => OlsCameraDriver::new_gphoto2(),
            "uvc" | "webcam" => OlsCameraDriver::new_uvc(),
            "indi" => OlsCameraDriver::new_indi(),
            "indigo" => OlsCameraDriver::new_indigo(),
            other => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("Unsupported camera driver '{other}'"),
                ));
            }
        };

        let mut driver = driver_result.map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to create {worker_driver_id} driver: {error}"),
            )
        })?;
        let cameras = driver.list_cameras().map_err(|error| {
            (
                StatusCode::BAD_GATEWAY,
                format!("Failed to enumerate {worker_driver_id} cameras: {error}"),
            )
        })?;
        if cameras.is_empty() {
            return Err((
                StatusCode::NOT_FOUND,
                "No cameras found for the selected driver".to_string(),
            ));
        }
        if requested_index < 0 || requested_index as usize >= cameras.len() {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "Camera index {requested_index} is out of range; driver reports {} camera(s)",
                    cameras.len()
                ),
            ));
        }

        let camera = driver.open_camera(requested_index).map_err(|error| {
            (
                StatusCode::BAD_GATEWAY,
                format!("Failed to open camera {requested_index}: {error}"),
            )
        })?;
        let camera_name = camera.name().unwrap_or_else(|_| "Unknown".to_string());
        Ok((driver, camera, camera_name, cameras, requested_index))
    });
    let connect_result: Result<ConnectedCamera, ConnectFailure> = match connect_worker.await {
        Ok(result) => result,
        Err(error) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Camera connection worker failed: {error}"),
        )),
    };

    let (driver, mut camera, camera_name, cameras, index) = match connect_result {
        Ok(connected) => connected,
        Err((status, message)) => {
            tracing::error!("{message}");
            let mut runtime = state.camera_runtime.write().unwrap();
            runtime.activity = CameraActivity::Error;
            runtime.last_error = Some(message.clone());
            return Err(ApiError::new(status, connect_error_code(status), message));
        }
    };

    tracing::info!("Found cameras: {:?}", cameras);
    let connection_id = uuid::Uuid::new_v4().to_string();
    let signals = state.camera_loss_signals();
    let callback_connection_id = connection_id.clone();
    camera.set_fault_handler(std::sync::Arc::new(move |reason| {
        signals.invalidate(&callback_connection_id, reason);
    }));
    *state.camera_driver.lock().unwrap() = Some(driver);
    *state.camera_name.lock().unwrap() = Some(camera_name.clone());
    *state.camera.lock().unwrap() = Some(camera);
    *state.streaming.lock().unwrap() = false;
    {
        let mut runtime = state.camera_runtime.write().unwrap();
        runtime.connection_id = Some(connection_id.clone());
        runtime.activity = CameraActivity::Idle;
        runtime.streaming = false;
        runtime.camera_name = Some(camera_name.clone());
        runtime.driver_id = Some(driver_id.clone());
        runtime.device_index = Some(index);
        runtime.last_error = None;
    }
    state
        .devices
        .write()
        .unwrap()
        .connected(DeviceKind::Camera, connection_id.clone(), None);
    let _ = state.device_updates.send(());

    Ok(Json(serde_json::json!({
        "status": "connected",
        "connection_id": connection_id,
        "driver": driver_id,
        "camera": camera_name,
        "index": index,
        "cameras_found": cameras,
    })))
}

fn connect_error_code(status: StatusCode) -> &'static str {
    match status {
        StatusCode::BAD_REQUEST => "invalid_camera_selection",
        StatusCode::NOT_FOUND => "camera_not_found",
        StatusCode::BAD_GATEWAY => "camera_hardware_error",
        _ => "camera_connect_failed",
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
    let hw = mount.health_status().map_err(|error| {
        tracing::error!("Mount did not answer passive status query after connect: {error}");
        StatusCode::BAD_GATEWAY
    })?;
    let snapshot = health::mount_snapshot(format!("{driver_type:?} mount"), &hw);
    let retired = {
        let mut slot = state.mount.lock().unwrap();
        let retired = slot.replace(mount);
        *state.mount_name.lock().unwrap() = Some(format!("{driver_type:?} mount"));
        state.devices.write().unwrap().connected(
            DeviceKind::Mount,
            uuid::Uuid::new_v4().to_string(),
            Some(snapshot),
        );
        retired
    };
    drop(retired);
    let _ = state.device_updates.send(());

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
        let focuser = ConnectedFocuser::Simulator(SimFocuser::new(position));
        let snapshot = health::focuser_snapshot(
            "Simulator focuser".into(),
            &focuser.health_status().unwrap(),
        );
        let retired = {
            let mut slot = state.focuser.lock().unwrap();
            let retired = slot.replace(focuser);
            *state.focuser_name.lock().unwrap() = Some("Simulator focuser".into());
            state.devices.write().unwrap().connected(
                DeviceKind::Focuser,
                uuid::Uuid::new_v4().to_string(),
                Some(snapshot),
            );
            retired
        };
        drop(retired);
        let _ = state.device_updates.send(());

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
    let focuser = ConnectedFocuser::Hardware(focuser);
    let hw = focuser.health_status().map_err(|error| {
        tracing::error!("Focuser did not answer passive position query after connect: {error}");
        StatusCode::BAD_GATEWAY
    })?;
    let snapshot = health::focuser_snapshot(format!("{driver_type:?} focuser"), &hw);
    let retired = {
        let mut slot = state.focuser.lock().unwrap();
        let retired = slot.replace(focuser);
        *state.focuser_name.lock().unwrap() = Some(format!("{driver_type:?} focuser"));
        state.devices.write().unwrap().connected(
            DeviceKind::Focuser,
            uuid::Uuid::new_v4().to_string(),
            Some(snapshot),
        );
        retired
    };
    drop(retired);
    let _ = state.device_updates.send(());

    Ok(Json(serde_json::json!({
        "status": "connected",
        "equipment_type": "focuser",
        "driver": req.driver,
        "option": option,
        "available_drivers": drivers,
    })))
}

pub(crate) async fn disconnect_camera(state: &AppState) -> Result<(), ApiError> {
    let ownership = state.acquisition.clone().lock_owned().await;
    let id = state.camera_runtime.read().unwrap().connection_id.clone();
    if let Some(id) = id {
        state
            .camera_loss_signals()
            .invalidate(&id, "Disconnected by request".into());
    }
    *state.streaming.lock().unwrap() = false;
    state.clear_camera_observations();
    *state.camera_runtime.write().unwrap() = crate::state::CameraRuntimeState::default();
    state
        .devices
        .write()
        .unwrap()
        .disconnected(DeviceKind::Camera, None, None);
    let _ = state.device_updates.send(());
    let worker_state = state.clone();
    tokio::task::spawn_blocking(move || {
        let _ownership = ownership;
        health::retire_camera(&worker_state);
    })
    .await
    .map_err(|error| ApiError::hardware(format!("Camera disconnect worker failed: {error}")))?;
    Ok(())
}

async fn disconnect_auxiliary(state: &AppState, kind: DeviceKind) -> Result<(), ApiError> {
    let id = state.devices.read().unwrap().connection_id(kind);
    if let Some(id) = id {
        state
            .devices
            .write()
            .unwrap()
            .disconnected(kind, Some(&id), None);
    }
    if kind == DeviceKind::Focuser {
        state
            .focus_abort_requested
            .store(true, std::sync::atomic::Ordering::Release);
    }
    let _ = state.device_updates.send(());
    let worker_state = state.clone();
    tokio::task::spawn_blocking(move || health::retire_auxiliary(&worker_state, kind))
        .await
        .map_err(|error| {
            ApiError::hardware(format!("{kind:?} disconnect worker failed: {error}"))
        })?;
    Ok(())
}

pub async fn disconnect_equipment(
    State(state): State<AppState>,
    request: Option<Json<DisconnectRequest>>,
) -> Result<Json<serde_json::Value>, ApiError> {
    tracing::info!("Disconnect requested");
    let equipment_type = request
        .and_then(|Json(req)| req.equipment_type)
        .unwrap_or_else(|| "all".to_string())
        .to_ascii_lowercase();

    if matches!(equipment_type.as_str(), "camera" | "all") {
        disconnect_camera(&state).await?;
    }
    if matches!(equipment_type.as_str(), "mount" | "all") {
        disconnect_auxiliary(&state, DeviceKind::Mount).await?;
    }
    if matches!(equipment_type.as_str(), "focuser" | "all") {
        disconnect_auxiliary(&state, DeviceKind::Focuser).await?;
    }
    if matches!(equipment_type.as_str(), "filter_wheel" | "all") {
        disconnect_auxiliary(&state, DeviceKind::FilterWheel).await?;
    }
    if !matches!(
        equipment_type.as_str(),
        "camera" | "mount" | "focuser" | "filter_wheel" | "all"
    ) {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "unsupported_equipment_type",
            format!("Unsupported equipment type '{equipment_type}'"),
        ));
    }
    let _ = state.device_updates.send(());

    Ok(Json(serde_json::json!({
        "status": "disconnected",
        "equipment_type": equipment_type,
    })))
}

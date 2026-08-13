use base64::{engine::general_purpose, Engine as _};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::hardware::{
    capabilities::driver_capability_report,
    focuser::{ConnectedFocuser, Focuser, FocuserDriver, SimFocuser},
    mount::{EqCoord, Mount, MountDriver},
};
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct RelayCommand {
    pub command_id: String,
    pub action: String,
    #[serde(default)]
    pub payload: Value,
}

fn payload_str(payload: &Value, key: &str, default: &str) -> String {
    payload
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_string()
}

fn payload_i32(payload: &Value, key: &str) -> Result<i32, String> {
    payload
        .get(key)
        .and_then(Value::as_i64)
        .map(|v| v as i32)
        .ok_or_else(|| format!("payload.{key} is required"))
}

fn payload_f64(payload: &Value, key: &str) -> Result<f64, String> {
    payload
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("payload.{key} is required"))
}

fn payload_bool(payload: &Value, key: &str) -> Result<bool, String> {
    payload
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("payload.{key} is required"))
}

pub async fn execute(state: AppState, command: &RelayCommand) -> Result<Value, String> {
    match command.action.as_str() {
        "equipment.connect" => connect_equipment(state, &command.payload).await,
        "equipment.disconnect" => disconnect_equipment(state, &command.payload).await,
        "equipment.status" => Ok(crate::telemetry::equipment_status(&state)),
        "drivers.capabilities" => Ok(json!(driver_capability_report())),
        "camera.descriptor" => camera_descriptor(state).await,
        "camera.parameters" => camera_parameters(state).await,
        "camera.formats" => camera_formats(state).await,
        "camera.set_parameter" => camera_set_parameter(state, &command.payload).await,
        "camera.start_stream" => start_camera_stream(state, &command.payload).await,
        "camera.stop_stream" | "camera.abort" => abort_camera(state).await,
        "camera.capture" => camera_capture(state, &command.payload).await,
        "camera.snapshot" => camera_snapshot(state),
        "mount.slew" => mount_slew(state, &command.payload),
        "mount.sync" => mount_sync(state, &command.payload),
        "mount.set_tracking" => mount_set_tracking(state, &command.payload),
        "mount.abort" => mount_abort(state),
        "focuser.move_relative" => focuser_move_relative(state, &command.payload),
        "focuser.move_absolute" => focuser_move_absolute(state, &command.payload),
        "focuser.position" => focuser_position(state),
        "focuser.abort" => focuser_abort(state),
        other => Err(format!("Unsupported relay action: {other}")),
    }
}

async fn camera_descriptor(state: AppState) -> Result<Value, String> {
    let axum::Json(descriptor) =
        crate::api::camera::get_camera_descriptor(axum::extract::State(state))
            .await
            .map_err(|error| error.message)?;
    serde_json::to_value(descriptor).map_err(|error| error.to_string())
}

async fn camera_parameters(state: AppState) -> Result<Value, String> {
    let axum::Json(parameters) = crate::api::camera::get_parameters(axum::extract::State(state))
        .await
        .map_err(|error| error.message)?;
    serde_json::to_value(parameters).map_err(|error| error.to_string())
}

async fn camera_formats(state: AppState) -> Result<Value, String> {
    let axum::Json(formats) = crate::api::camera::get_formats(axum::extract::State(state))
        .await
        .map_err(|error| error.message)?;
    serde_json::to_value(formats).map_err(|error| error.to_string())
}

async fn camera_set_parameter(state: AppState, payload: &Value) -> Result<Value, String> {
    let name = payload
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "payload.name is required".to_string())?;
    let req = crate::api::camera::SetParamRequest {
        name: name.to_string(),
        value: payload_f64(payload, "value")?,
    };
    let axum::Json(value) =
        crate::api::camera::set_parameter(axum::extract::State(state), axum::Json(req))
            .await
            .map_err(|error| error.message)?;
    Ok(value)
}

/// Cloud-relay capture.
///
/// Delegates to the single shared implementation in `api::camera` rather than
/// keeping a second copy. The duplicate that used to live here carried the
/// same defects as the HTTP path — it wrote `Exp`/`Gain` to bodies that only
/// understand `shutterspeed`/`iso`, discarded the resulting errors with
/// `let _ =`, and then recorded the *requested* exposure in the FITS header —
/// so a remote capture could disagree with the photograph it claimed to
/// describe. One implementation means one place to be correct.
async fn camera_capture(state: AppState, payload: &Value) -> Result<Value, String> {
    let params: crate::api::camera::CaptureParams =
        serde_json::from_value(payload.clone()).map_err(|e| e.to_string())?;

    let image = crate::api::camera::perform_capture(state, params)
        .await
        .map_err(|error| error.message)?;

    serde_json::to_value(image).map_err(|e| e.to_string())
}

async fn connect_equipment(state: AppState, payload: &Value) -> Result<Value, String> {
    let equipment_type = payload_str(payload, "equipment_type", "camera").to_ascii_lowercase();
    let driver = payload_str(payload, "driver", "sim");
    let option = payload
        .get("option")
        .and_then(Value::as_str)
        .map(str::to_string);

    let index = payload
        .get("index")
        .and_then(Value::as_i64)
        .map(|value| value as i32);

    match equipment_type.as_str() {
        "camera" => {
            let req = crate::api::ConnectRequest {
                equipment_type: Some("camera".to_string()),
                driver,
                option,
                index,
            };
            crate::api::connect_camera(state, req)
                .await
                .map(|axum::Json(value)| value)
                .map_err(|error| error.message)
        }
        "mount" => connect_mount(state, &driver, option),
        "focuser" => connect_focuser(state, &driver, option),
        other => Err(format!("Unsupported equipment_type: {other}")),
    }
}

async fn disconnect_equipment(state: AppState, payload: &Value) -> Result<Value, String> {
    let equipment_type = payload_str(payload, "equipment_type", "all");
    let axum::Json(response) = crate::api::disconnect_equipment(
        axum::extract::State(state),
        Some(axum::Json(crate::api::DisconnectRequest {
            equipment_type: Some(equipment_type.clone()),
        })),
    )
    .await
    .map_err(|error| error.message)?;
    Ok(response)
}

fn connect_mount(state: AppState, driver: &str, option: Option<String>) -> Result<Value, String> {
    let driver_type = match driver {
        "alpaca" => MountDriver::Alpaca,
        "indi" => MountDriver::Indi,
        "sim" | "simulator" => MountDriver::Simulator,
        other => return Err(format!("Unsupported mount driver: {other}")),
    };
    let option = option.unwrap_or_else(|| "auto".to_string());
    let mut mount = Mount::new(driver_type).map_err(|e| e.to_string())?;
    mount
        .connect_with_option(&option)
        .map_err(|e| e.to_string())?;
    let drivers = mount.list_drivers().unwrap_or_default();

    *state.mount_name.lock().unwrap() = Some(format!("{driver_type:?} mount"));
    *state.mount.lock().unwrap() = Some(mount);

    Ok(json!({
        "status": "connected",
        "equipment_type": "mount",
        "driver": driver,
        "option": option,
        "available_drivers": drivers,
    }))
}

fn connect_focuser(state: AppState, driver: &str, option: Option<String>) -> Result<Value, String> {
    if matches!(driver, "sim" | "simulator") {
        let position = option
            .as_deref()
            .and_then(|value| value.parse::<i32>().ok())
            .unwrap_or(10000);
        *state.focuser_name.lock().unwrap() = Some("Simulator focuser".to_string());
        *state.focuser.lock().unwrap() =
            Some(ConnectedFocuser::Simulator(SimFocuser::new(position)));

        return Ok(json!({
            "status": "connected",
            "equipment_type": "focuser",
            "driver": driver,
            "position": position,
            "available_drivers": [{"name": "sim", "option": "initial_position"}],
        }));
    }

    let driver_type = match driver {
        "alpaca" => FocuserDriver::Alpaca,
        "indi" => FocuserDriver::Indi,
        "moonlite" => FocuserDriver::Moonlite,
        "usb" | "serial" => FocuserDriver::UsbSerial,
        other => return Err(format!("Unsupported focuser driver: {other}")),
    };
    let option = option.unwrap_or_else(|| "auto".to_string());
    let mut focuser = Focuser::new(driver_type).map_err(|e| e.to_string())?;
    focuser
        .connect_with_option(&option)
        .map_err(|e| e.to_string())?;
    let drivers = focuser.list_drivers().unwrap_or_default();

    *state.focuser_name.lock().unwrap() = Some(format!("{driver_type:?} focuser"));
    *state.focuser.lock().unwrap() = Some(ConnectedFocuser::Hardware(focuser));

    Ok(json!({
        "status": "connected",
        "equipment_type": "focuser",
        "driver": driver,
        "option": option,
        "available_drivers": drivers,
    }))
}

async fn start_camera_stream(state: AppState, payload: &Value) -> Result<Value, String> {
    let format_index = match payload.get("format_index").and_then(Value::as_i64) {
        Some(value) if value < 0 => return Err("format_index cannot be negative".to_string()),
        Some(value) => Some(value as usize),
        None => None,
    };
    crate::api::camera::start_stream_control(&state, format_index)
        .await
        .map_err(|error| error.message)
}

async fn abort_camera(state: AppState) -> Result<Value, String> {
    crate::api::camera::stop_stream_control(&state)
        .await
        .map_err(|error| error.message)
}

fn camera_snapshot(state: AppState) -> Result<Value, String> {
    let frame = state.latest_frame.lock().unwrap().clone();
    let frame = frame.ok_or_else(|| "No camera frame is available yet".to_string())?;
    let jpeg = crate::api::camera::encode_frame_jpeg(&frame)
        .ok_or_else(|| "Latest frame cannot be encoded as a browser preview".to_string())?;
    Ok(json!({
        "status": "ok",
        "format": "JPEG",
        "mime_type": "image/jpeg",
        "width": frame.width,
        "height": frame.height,
        "frame_counter": frame.frame_counter,
        "timestamp": frame.unix_timestamp,
        "data_base64": general_purpose::STANDARD.encode(jpeg),
    }))
}

fn mount_slew(state: AppState, payload: &Value) -> Result<Value, String> {
    let target = EqCoord {
        ra: payload_f64(payload, "ra")?,
        dec: payload_f64(payload, "dec")?,
    };
    let mount = state.mount.lock().unwrap();
    let mount = mount
        .as_ref()
        .ok_or_else(|| "No mount connected".to_string())?;
    mount.goto(target).map_err(|e| e.to_string())?;
    Ok(json!({ "status": "slewing", "ra": target.ra, "dec": target.dec }))
}

fn mount_sync(state: AppState, payload: &Value) -> Result<Value, String> {
    let coord = EqCoord {
        ra: payload_f64(payload, "ra")?,
        dec: payload_f64(payload, "dec")?,
    };
    let mount = state.mount.lock().unwrap();
    let mount = mount
        .as_ref()
        .ok_or_else(|| "No mount connected".to_string())?;
    mount.sync(coord).map_err(|e| e.to_string())?;
    Ok(json!({ "status": "synced", "ra": coord.ra, "dec": coord.dec }))
}

fn mount_set_tracking(state: AppState, payload: &Value) -> Result<Value, String> {
    let enabled = payload_bool(payload, "enabled")?;
    let mount = state.mount.lock().unwrap();
    let mount = mount
        .as_ref()
        .ok_or_else(|| "No mount connected".to_string())?;
    mount.set_tracking(enabled).map_err(|e| e.to_string())?;
    Ok(json!({ "status": "tracking_updated", "enabled": enabled }))
}

fn mount_abort(state: AppState) -> Result<Value, String> {
    let mount = state.mount.lock().unwrap();
    let mount = mount
        .as_ref()
        .ok_or_else(|| "No mount connected".to_string())?;
    mount.stop().map_err(|e| e.to_string())?;
    Ok(json!({ "status": "stopped" }))
}

fn focuser_move_relative(state: AppState, payload: &Value) -> Result<Value, String> {
    let steps = payload_i32(payload, "steps")?;
    let mut focuser = state.focuser.lock().unwrap();
    let focuser = focuser
        .as_mut()
        .ok_or_else(|| "No focuser connected".to_string())?;
    focuser.move_relative(steps).map_err(|e| e.to_string())?;
    Ok(json!({ "status": "moving", "steps": steps }))
}

fn focuser_move_absolute(state: AppState, payload: &Value) -> Result<Value, String> {
    let position = payload_i32(payload, "position")?;
    let mut focuser = state.focuser.lock().unwrap();
    let focuser = focuser
        .as_mut()
        .ok_or_else(|| "No focuser connected".to_string())?;
    focuser.move_absolute(position).map_err(|e| e.to_string())?;
    Ok(json!({ "status": "moving", "position": position }))
}

fn focuser_position(state: AppState) -> Result<Value, String> {
    let focuser = state.focuser.lock().unwrap();
    let focuser = focuser
        .as_ref()
        .ok_or_else(|| "No focuser connected".to_string())?;
    let position = focuser.position().map_err(|e| e.to_string())?;
    Ok(json!({ "position": position }))
}

fn focuser_abort(state: AppState) -> Result<Value, String> {
    let mut focuser = state.focuser.lock().unwrap();
    let focuser = focuser
        .as_mut()
        .ok_or_else(|| "No focuser connected".to_string())?;
    focuser.abort().map_err(|e| e.to_string())?;
    Ok(json!({ "status": "stopped" }))
}

use base64::{engine::general_purpose, Engine as _};
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::{Duration, Instant};

use crate::api::camera::{option_name, option_type_name, parse_option, CamParamInfo, FormatInfo};
use crate::fits::FitsMeta;
use crate::hardware::{
    camera::OlsCameraDriver,
    capabilities::driver_capability_report,
    focuser::{ConnectedFocuser, Focuser, FocuserDriver, SimFocuser},
    mount::{EqCoord, Mount, MountDriver},
    ols_ffi::OlsCamOptionId,
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
        "equipment.status" => Ok(crate::telemetry::equipment_status(&state)),
        "drivers.capabilities" => Ok(json!(driver_capability_report())),
        "camera.parameters" => camera_parameters(state),
        "camera.formats" => camera_formats(state),
        "camera.set_parameter" => camera_set_parameter(state, &command.payload),
        "camera.start_stream" => start_camera_stream(state, &command.payload),
        "camera.abort" => abort_camera(state),
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

fn camera_parameters(state: AppState) -> Result<Value, String> {
    let cam_lock = state.camera.lock().unwrap();
    let camera = cam_lock
        .as_ref()
        .ok_or_else(|| "No camera connected".to_string())?;
    let options = camera.supported_options().map_err(|e| e.to_string())?;
    let mut params = Vec::with_capacity(options.len());

    for id in options {
        if let Ok(p) = camera.get_parameter(id) {
            params.push(CamParamInfo {
                name: option_name(id).to_string(),
                kind: option_type_name(p.option_type).to_string(),
                read_only: p.read_only,
                min: p.min_val,
                max: p.max_val,
                step: p.step_size,
                default: p.def_val,
                current: p.cur_val,
            });
        }
    }

    Ok(json!(params))
}

fn camera_formats(state: AppState) -> Result<Value, String> {
    let cam_lock = state.camera.lock().unwrap();
    let camera = cam_lock
        .as_ref()
        .ok_or_else(|| "No camera connected".to_string())?;
    let formats = camera.formats().map_err(|e| e.to_string())?;
    let list: Vec<FormatInfo> = formats
        .iter()
        .enumerate()
        .map(|(i, f)| FormatInfo {
            index: i,
            format: format!("{:?}", f.format),
            width: f.width,
            height: f.height,
            bin: f.bin,
            framerate: f.framerate,
        })
        .collect();

    Ok(json!(list))
}

fn camera_set_parameter(state: AppState, payload: &Value) -> Result<Value, String> {
    let name = payload
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "payload.name is required".to_string())?;
    let value = payload_f64(payload, "value")?;
    let id = parse_option(name).ok_or_else(|| format!("Unknown camera parameter: {name}"))?;

    let cam_lock = state.camera.lock().unwrap();
    let camera = cam_lock
        .as_ref()
        .ok_or_else(|| "No camera connected".to_string())?;
    camera.set_parameter(id, value).map_err(|e| e.to_string())?;
    let current = camera.get_parameter(id).map(|p| p.cur_val).unwrap_or(value);

    Ok(json!({
        "status": "ok",
        "name": name,
        "value": current,
    }))
}

async fn camera_capture(state: AppState, payload: &Value) -> Result<Value, String> {
    let params: crate::api::camera::CaptureParams =
        serde_json::from_value(payload.clone()).map_err(|e| e.to_string())?;

    {
        let cam_lock = state.camera.lock().unwrap();
        let camera = cam_lock
            .as_ref()
            .ok_or_else(|| "No camera connected".to_string())?;
        let _ = camera.set_parameter(OlsCamOptionId::Exp, params.exposure * 1000.0);
        if let Some(gain) = params.gain {
            let _ = camera.set_parameter(OlsCamOptionId::Gain, gain as f64);
        }
        if let Some(temp) = params.temperature {
            let _ = camera.set_parameter(OlsCamOptionId::CoolerTarget, temp);
        }
    }

    if !*state.streaming.lock().unwrap() {
        return Err("Capture requires an active camera stream".to_string());
    }

    let settle = Duration::from_secs_f64(params.exposure.clamp(0.0, 120.0));
    tokio::time::sleep(settle).await;

    let baseline = state
        .latest_frame
        .lock()
        .unwrap()
        .as_ref()
        .map(|f| f.frame_counter);
    let deadline = Instant::now() + Duration::from_secs(5);
    let frame = loop {
        {
            let guard = state.latest_frame.lock().unwrap();
            if let Some(frame) = guard.as_ref() {
                if Some(frame.frame_counter) != baseline {
                    break frame.clone();
                }
            }
        }
        if Instant::now() >= deadline {
            let guard = state.latest_frame.lock().unwrap();
            match guard.as_ref() {
                Some(frame) => break frame.clone(),
                None => return Err("No camera frame is available yet".to_string()),
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    };

    let (temperature, gain_val) = {
        let cam_lock = state.camera.lock().unwrap();
        let camera = cam_lock.as_ref();
        let temperature = camera
            .and_then(|c| c.get_parameter(OlsCamOptionId::Temperature).ok())
            .map(|p| p.cur_val)
            .unwrap_or(f64::NAN);
        let gain_val = camera
            .and_then(|c| c.get_parameter(OlsCamOptionId::Gain).ok())
            .map(|p| p.cur_val as u32)
            .unwrap_or_else(|| params.gain.unwrap_or(0));
        (temperature, gain_val)
    };

    let id = uuid::Uuid::new_v4().to_string();
    let timestamp = chrono::Utc::now().to_rfc3339();
    let meta = FitsMeta {
        exposure_s: params.exposure,
        gain: gain_val,
        temperature_c: temperature,
        binning: params.binning.unwrap_or(1),
        instrument: state
            .camera_name
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_else(|| "Achernar".to_string()),
        date_obs: timestamp.clone(),
    };
    let path = crate::fits::write_capture(&frame, &meta, &id).map_err(|e| e.to_string())?;

    Ok(json!({
        "id": id,
        "width": frame.width,
        "height": frame.height,
        "path": path,
        "metadata": {
            "exposure": params.exposure,
            "gain": gain_val,
            "temperature": temperature,
            "timestamp": timestamp,
        }
    }))
}

async fn connect_equipment(state: AppState, payload: &Value) -> Result<Value, String> {
    let equipment_type = payload_str(payload, "equipment_type", "camera").to_ascii_lowercase();
    let driver = payload_str(payload, "driver", "sim");
    let option = payload
        .get("option")
        .and_then(Value::as_str)
        .map(str::to_string);

    match equipment_type.as_str() {
        "camera" => connect_camera(state, &driver, option),
        "mount" => connect_mount(state, &driver, option),
        "focuser" => connect_focuser(state, &driver, option),
        other => Err(format!("Unsupported equipment_type: {other}")),
    }
}

fn connect_camera(state: AppState, driver: &str, option: Option<String>) -> Result<Value, String> {
    let driver_result = match driver {
        "sim" => OlsCameraDriver::new_sim(option.as_deref().unwrap_or("./sim")),
        "alpaca" => {
            OlsCameraDriver::new_alpaca(option.as_deref().unwrap_or("http://localhost:11111"))
        }
        "asi" => OlsCameraDriver::new_asi(),
        "svb" => OlsCameraDriver::new_svb(),
        "toup" => OlsCameraDriver::new_toup(),
        "wdir" | "watchdir" => OlsCameraDriver::new_wdir(option.as_deref().unwrap_or("./watch")),
        "gphoto2" | "dslr" => OlsCameraDriver::new_gphoto2(),
        "uvc" | "webcam" => OlsCameraDriver::new_uvc(),
        "indi" => OlsCameraDriver::new_indi(),
        "indigo" => OlsCameraDriver::new_indigo(),
        other => return Err(format!("Unsupported camera driver: {other}")),
    };

    let mut camera_driver = driver_result.map_err(|e| e.to_string())?;
    let cameras = camera_driver.list_cameras().map_err(|e| e.to_string())?;
    if cameras.is_empty() {
        return Err("No cameras found for selected driver".to_string());
    }
    let camera = camera_driver.open_camera(0).map_err(|e| e.to_string())?;
    let camera_name = camera
        .name()
        .unwrap_or_else(|_| "Unknown camera".to_string());

    *state.camera_driver.lock().unwrap() = Some(camera_driver);
    *state.camera_name.lock().unwrap() = Some(camera_name.clone());
    *state.camera.lock().unwrap() = Some(camera);

    Ok(json!({
        "status": "connected",
        "equipment_type": "camera",
        "driver": driver,
        "camera": camera_name,
        "cameras_found": cameras,
    }))
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

fn start_camera_stream(state: AppState, payload: &Value) -> Result<Value, String> {
    let format_index = payload
        .get("format_index")
        .and_then(Value::as_i64)
        .unwrap_or(0)
        .max(0) as usize;

    let mut camera_lock = state.camera.lock().unwrap();
    let camera = camera_lock
        .as_mut()
        .ok_or_else(|| "No camera connected".to_string())?;
    let formats = camera.formats().map_err(|e| e.to_string())?;
    let fmt = formats
        .get(format_index)
        .or_else(|| formats.first())
        .ok_or_else(|| "Camera reports no stream formats".to_string())?;

    camera
        .start_stream(fmt, state.latest_frame.clone())
        .map_err(|e| e.to_string())?;
    *state.streaming.lock().unwrap() = true;

    Ok(json!({
        "status": "streaming",
        "format": format!("{:?}", fmt.format),
        "width": fmt.width,
        "height": fmt.height,
    }))
}

fn abort_camera(state: AppState) -> Result<Value, String> {
    let mut camera = state.camera.lock().unwrap();
    if let Some(camera) = camera.as_mut() {
        let _ = camera.stop_stream();
    }
    *state.streaming.lock().unwrap() = false;
    Ok(json!({ "status": "stopped" }))
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

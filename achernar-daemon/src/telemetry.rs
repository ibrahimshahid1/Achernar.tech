use serde_json::{json, Value};

use crate::state::AppState;

pub fn equipment_status(state: &AppState) -> Value {
    use crate::hardware::health::DeviceKind;
    // Never enter a native SDK on a telemetry or HTTP status path. Health
    // workers own passive probes; their registry is the connection authority.
    let devices = state.devices.read().unwrap().clone();
    let camera_status = {
        let _barrier = state.camera_observation_barrier.lock().unwrap();
        let runtime = state.camera_runtime.read().unwrap().clone();
        let generation = state.current_camera_generation();
        let frame = state.latest_frame.lock().unwrap().clone();
        let frame_info = frame
            .filter(|f| f.stream_generation == generation && runtime.streaming)
            .map(|f| {
                json!({"stream_generation":f.stream_generation, "frame_counter":f.frame_counter,
                "width":f.width, "height":f.height, "format":format!("{:?}",f.format),
                "timestamp":f.unix_timestamp})
            });
        runtime.connection_id.as_ref().filter(|id| devices.connection_id(DeviceKind::Camera).as_ref()==Some(id)).map(|_| {
            json!({"connected":true, "connection_id":runtime.connection_id, "name":runtime.camera_name,
                "driver_id":runtime.driver_id,"device_index":runtime.device_index,"status":runtime.activity,
                "temperature":runtime.temperature_c,"cooling":runtime.cooler_on,
                "exposing":runtime.activity==crate::state::CameraActivity::Exposing,
                "streaming":runtime.streaming,"progress":runtime.exposure_progress(chrono::Utc::now()),
                "last_error":runtime.last_error,"latest_frame":frame_info})
        })
    };
    let mount_status = devices.snapshot(DeviceKind::Mount);
    let focuser_status = devices.snapshot(DeviceKind::Focuser);
    let filter_status = devices.snapshot(DeviceKind::FilterWheel);
    json!({"connected":camera_status.is_some() || mount_status.is_some() || focuser_status.is_some() || filter_status.is_some(),
        "camera":camera_status,"mount":mount_status,"focuser":focuser_status,"filter_wheel":filter_status,
        "device_health":devices.0})
}

pub fn telemetry_event(state: &AppState) -> Value {
    let status = equipment_status(state);
    let observation = state
        .vision_observation
        .read()
        .ok()
        .and_then(|obs| obs.clone());
    json!({
        "type": "TELEMETRY_UPDATE",
        "timestamp": chrono::Utc::now().timestamp(),
        "payload": {
            "camera": status.get("camera").cloned().unwrap_or(Value::Null),
            "mount": status.get("mount").cloned().unwrap_or(Value::Null),
            "focuser": status.get("focuser").cloned().unwrap_or(Value::Null),
            "filter_wheel": status.get("filter_wheel").cloned().unwrap_or(Value::Null),
            "device_health": status.get("device_health").cloned().unwrap_or_else(||json!({})),
            // There is no guider in this system. Reporting 0.00" RMS every
            // second is a measurement of a subsystem that does not exist, and
            // it reads as perfect guiding. Null until a guider reports.
            "guiding": Value::Null,
            "ai": {
                // Null (not 0.0) when there is no valid measurement — 0 HFR would
                // read as perfect focus on a blank frame. The same reasoning
                // applies to every field below: an absent observation is not a
                // clear sky, not an empty star field, and not a zero exposure.
                "focus_hfr": observation.as_ref().and_then(|obs| obs.hfr_px),
                "cloud_prob": observation.as_ref().map(|obs| obs.cloud_probability),
                "star_count": observation.as_ref().map(|obs| obs.star_count),
                "star_count_used": observation.as_ref().map(|obs| obs.star_count_used),
                // False here means "we looked and the observation was not
                // usable". Null means we never looked at all.
                "observation_valid": observation.as_ref().map(|obs| obs.observation_valid),
                // Not implemented. Null rather than a plausible exposure time.
                "recommended_exposure": Value::Null,
            },
        }
    })
}

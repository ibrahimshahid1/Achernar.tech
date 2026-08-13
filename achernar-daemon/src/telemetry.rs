use serde_json::{json, Value};

use crate::state::AppState;

pub fn equipment_status(state: &AppState) -> Value {
    let camera_status = {
        // Copy every slot independently. In particular, never hold the latest
        // frame while reaching for the camera mutex: start/stop callbacks take
        // those in the opposite direction and used to deadlock telemetry.
        let runtime = state.camera_runtime.read().unwrap().clone();
        let frame = state.latest_frame.lock().unwrap().clone();
        let frame_info = frame.map(|f| {
            json!({
                "stream_generation": f.stream_generation,
                "frame_counter": f.frame_counter,
                "width": f.width,
                "height": f.height,
                "format": format!("{:?}", f.format),
            })
        });
        runtime.connection_id.as_ref().map(|_| {
            json!({
                "connected": true,
                "connection_id": runtime.connection_id,
                "name": runtime.camera_name,
                "driver_id": runtime.driver_id,
                "device_index": runtime.device_index,
                "status": runtime.activity,
                "temperature": runtime.temperature_c,
                "cooling": runtime.cooler_on,
                "exposing": runtime.activity == crate::state::CameraActivity::Exposing,
                "streaming": runtime.streaming,
                "progress": runtime.exposure_progress(chrono::Utc::now()),
                "last_error": runtime.last_error,
                "latest_frame": frame_info,
            })
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
            json!({
                "connected": hw.connected,
                "name": mount_name.unwrap_or_else(|| "OpenLiveStacker mount".to_string()),
                "ra": ra,
                "dec": dec,
                "pier_side": "unknown",
                "tracking": hw.tracking.unwrap_or(false),
                "parked": false,
            })
        })
    };

    let focuser_status = {
        let focuser_name = state.focuser_name.lock().unwrap().clone();
        let focuser = state.focuser.lock().unwrap();
        focuser.as_ref().map(|f| {
            let hw = f.status();
            json!({
                "connected": hw.connected,
                "name": focuser_name.unwrap_or_else(|| "OpenLiveStacker focuser".to_string()),
                "position": hw.position.unwrap_or(0),
                "moving": hw.is_moving,
                "absolute": hw.is_absolute,
                "max_range": hw.max_range,
            })
        })
    };

    json!({
        "connected": camera_status.is_some() || mount_status.is_some() || focuser_status.is_some(),
        "camera": camera_status,
        "mount": mount_status,
        "focuser": focuser_status,
    })
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
            "guiding": {
                "rms_total": 0.0,
                "dx": 0.0,
                "dy": 0.0,
            },
            "ai": {
                // Null (not 0.0) when there is no valid measurement — 0 HFR would
                // read as perfect focus on a blank frame.
                "focus_hfr": observation.as_ref().and_then(|obs| obs.hfr_px),
                "cloud_prob": observation.as_ref().map(|obs| obs.cloud_probability).unwrap_or(0.0),
                "star_count": observation.as_ref().map(|obs| obs.star_count).unwrap_or(0),
                "star_count_used": observation.as_ref().map(|obs| obs.star_count_used).unwrap_or(0),
                "observation_valid": observation.as_ref().map(|obs| obs.observation_valid).unwrap_or(false),
                "recommended_exposure": 0.0,
            },
        }
    })
}

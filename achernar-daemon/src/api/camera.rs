use axum::{extract::Query, extract::State, http::StatusCode, Json};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use crate::fits::FitsMeta;
use crate::hardware::camera::FrameData;
use crate::hardware::ols_ffi::{OlsCamOptionId, OlsCamOptionType, OlsStreamType};
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
    /// Filesystem path of the saved FITS frame.
    pub path: Option<String>,
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
pub struct StreamStartQuery {
    /// Optional index into GET /camera/formats (selects resolution / binning).
    pub format_index: Option<usize>,
}

/// Start the camera stream. Optionally pick a stream format / binning mode by
/// index (see GET /camera/formats); defaults to the first available format.
pub async fn start_stream(
    State(state): State<AppState>,
    Query(q): Query<StreamStartQuery>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let frame_store = state.latest_frame.clone();

    let mut cam_lock = state.camera.lock().unwrap();
    let camera = cam_lock.as_mut().ok_or_else(|| {
        tracing::error!("No camera connected");
        StatusCode::PRECONDITION_FAILED
    })?;

    let formats = camera.formats().map_err(|e| {
        tracing::error!("Failed to get formats: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    if formats.is_empty() {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    let idx = q.format_index.unwrap_or(0).min(formats.len() - 1);
    let fmt = formats[idx];
    tracing::info!(
        "Starting stream: idx={} {:?} {}x{} bin={}",
        idx,
        fmt.format,
        fmt.width,
        fmt.height,
        fmt.bin
    );

    camera.start_stream(&fmt, frame_store).map_err(|e| {
        tracing::error!("Failed to start stream: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    *state.streaming.lock().unwrap() = true;

    Ok(Json(serde_json::json!({
        "status": "streaming",
        "format": format!("{:?}", fmt.format),
        "width": fmt.width,
        "height": fmt.height,
        "bin": fmt.bin,
    })))
}

/// Single-shot capture: apply exposure/gain/cooling, wait one exposure for a
/// fresh frame from the active stream, save it to FITS, and return the metadata
/// plus the saved file path.
///
/// Requires an active stream (OLS produces frames through the streaming
/// callback). Choose a Raw16/Mono16 stream format for true 16-bit science
/// frames; an MJPEG/RGB preview yields an 8-bit RGB FITS.
pub async fn capture(
    State(state): State<AppState>,
    Json(params): Json<CaptureParams>,
) -> Result<Json<ImageData>, StatusCode> {
    tracing::info!("Capture requested: {}s", params.exposure);

    // 1) Apply exposure / gain / optional cooler setpoint.
    {
        let cam_lock = state.camera.lock().unwrap();
        let camera = cam_lock.as_ref().ok_or(StatusCode::PRECONDITION_FAILED)?;
        let _ = camera.set_parameter(OlsCamOptionId::Exp, params.exposure * 1000.0);
        if let Some(gain) = params.gain {
            let _ = camera.set_parameter(OlsCamOptionId::Gain, gain as f64);
        }
        if let Some(temp) = params.temperature {
            let _ = camera.set_parameter(OlsCamOptionId::CoolerTarget, temp);
        }
    }

    // 2) The capture reads from the streaming pipeline; require it to be active.
    if !*state.streaming.lock().unwrap() {
        tracing::warn!("Capture requested with no active stream");
        return Err(StatusCode::PRECONDITION_FAILED);
    }

    // 3) Let the new exposure settle, then wait for a *fresh* frame.
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
            if let Some(f) = guard.as_ref() {
                if Some(f.frame_counter) != baseline {
                    break f.clone();
                }
            }
        }
        if Instant::now() >= deadline {
            // Fall back to the most recent frame, if any.
            let guard = state.latest_frame.lock().unwrap();
            match guard.as_ref() {
                Some(f) => break f.clone(),
                None => return Err(StatusCode::GATEWAY_TIMEOUT),
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    };

    // 4) Read back the values the hardware actually applied.
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

    // 5) Persist to FITS with acquisition metadata.
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

    let path = crate::fits::write_capture(&frame, &meta, &id).map_err(|e| {
        tracing::error!("FITS write failed: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    tracing::info!("Capture saved: {}", path);

    Ok(Json(ImageData {
        id,
        width: frame.width as u32,
        height: frame.height as u32,
        path: Some(path),
        metadata: ImageMetadata {
            exposure: params.exposure,
            gain: gain_val,
            temperature,
            timestamp,
        },
    }))
}

pub async fn abort(State(state): State<AppState>) -> Result<StatusCode, StatusCode> {
    tracing::info!("Capture/stream abort requested");

    let mut cam_lock = state.camera.lock().unwrap();
    if let Some(camera) = cam_lock.as_mut() {
        let _ = camera.stop_stream();
    }
    *state.streaming.lock().unwrap() = false;

    Ok(StatusCode::OK)
}

/// POST /camera/snapshot — return the latest preview frame as a JPEG payload
/// for the Django relay/browser path.
pub async fn snapshot(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let frame = state.latest_frame.lock().unwrap().clone();
    let frame = frame.ok_or(StatusCode::NOT_FOUND)?;
    let jpeg = encode_frame_jpeg(&frame).ok_or(StatusCode::UNSUPPORTED_MEDIA_TYPE)?;

    Ok(Json(serde_json::json!({
        "status": "ok",
        "format": "JPEG",
        "mime_type": "image/jpeg",
        "width": frame.width,
        "height": frame.height,
        "frame_counter": frame.frame_counter,
        "timestamp": frame.unix_timestamp,
        "data_base64": base64::engine::general_purpose::STANDARD.encode(jpeg),
    })))
}

/// MJPEG Stream placeholder
pub async fn stream_mjpeg() -> Result<String, StatusCode> {
    tracing::info!("MJPEG stream requested");
    Ok("Use /api/v1/camera/stream/start to start WebSocket stream".to_string())
}

// ───────────────────────── Manual parameter control ─────────────────────────
// The daemon exposes the camera's *entire* control surface so the UI can offer
// traditional, NINA/SharpCap-style manual control of every supported option.
// AI/autonomy is layered on top of these same primitives — it never replaces
// the operator's ability to drive each parameter directly.

#[derive(Debug, Serialize)]
pub struct CamParamInfo {
    /// Stable snake_case key (e.g. "exposure", "gain", "cooler_target").
    pub name: String,
    /// Value semantics: bool, number, msec, percent, kelvin, celsius, selection.
    pub kind: String,
    pub read_only: bool,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub default: f64,
    pub current: f64,
}

/// GET /camera/parameters — introspect every supported option with its
/// min/max/step/current so the UI can render adaptive manual controls.
pub async fn get_parameters(
    State(state): State<AppState>,
) -> Result<Json<Vec<CamParamInfo>>, StatusCode> {
    let cam_lock = state.camera.lock().unwrap();
    let camera = cam_lock.as_ref().ok_or(StatusCode::PRECONDITION_FAILED)?;

    let options = camera.supported_options().map_err(|e| {
        tracing::error!("Failed to list camera options: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

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
    Ok(Json(params))
}

#[derive(Debug, Deserialize)]
pub struct SetParamRequest {
    pub name: String,
    pub value: f64,
}

/// POST /camera/parameters — set a single option by name
/// (exposure, gain, gamma, cooler_target, cooler_on, fan_on, ...).
pub async fn set_parameter(
    State(state): State<AppState>,
    Json(req): Json<SetParamRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let id = parse_option(&req.name).ok_or_else(|| {
        tracing::warn!("Unknown camera parameter: {}", req.name);
        StatusCode::BAD_REQUEST
    })?;

    let cam_lock = state.camera.lock().unwrap();
    let camera = cam_lock.as_ref().ok_or(StatusCode::PRECONDITION_FAILED)?;

    camera.set_parameter(id, req.value).map_err(|e| {
        tracing::error!("Failed to set {}: {}", req.name, e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Read back the value the hardware actually applied (it may clamp/round).
    let current = camera
        .get_parameter(id)
        .map(|p| p.cur_val)
        .unwrap_or(req.value);

    Ok(Json(serde_json::json!({
        "status": "ok",
        "name": req.name,
        "value": current,
    })))
}

#[derive(Debug, Serialize)]
pub struct FormatInfo {
    pub index: usize,
    pub format: String,
    pub width: i32,
    pub height: i32,
    pub bin: i32,
    pub framerate: f32,
}

/// GET /camera/formats — list selectable stream formats / binning modes.
pub async fn get_formats(
    State(state): State<AppState>,
) -> Result<Json<Vec<FormatInfo>>, StatusCode> {
    let cam_lock = state.camera.lock().unwrap();
    let camera = cam_lock.as_ref().ok_or(StatusCode::PRECONDITION_FAILED)?;

    let formats = camera.formats().map_err(|e| {
        tracing::error!("Failed to get formats: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let list = formats
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
    Ok(Json(list))
}

/// Encode a raw camera frame to JPEG bytes for the live-view stream.
/// MJPEG frames pass through untouched; RGB24 and 8-bit mono are re-encoded.
/// Returns `None` for formats we don't preview-encode yet (e.g. 16-bit raw).
pub fn encode_frame_jpeg(f: &FrameData) -> Option<Vec<u8>> {
    if f.format == OlsStreamType::Mjpeg {
        return Some(f.data.clone());
    }
    let (w, h) = (f.width as u32, f.height as u32);
    let dynimg = match f.format {
        OlsStreamType::Rgb24 => {
            image::DynamicImage::ImageRgb8(image::RgbImage::from_raw(w, h, f.data.clone())?)
        }
        OlsStreamType::Mono8 | OlsStreamType::Raw8 => {
            image::DynamicImage::ImageLuma8(image::GrayImage::from_raw(w, h, f.data.clone())?)
        }
        _ => return None,
    };
    let mut out = Vec::new();
    dynimg
        .write_to(
            &mut std::io::Cursor::new(&mut out),
            image::ImageOutputFormat::Jpeg(80),
        )
        .ok()?;
    Some(out)
}

/// snake_case key for each option id (stable contract with the UI).
pub(crate) fn option_name(id: OlsCamOptionId) -> &'static str {
    use OlsCamOptionId::*;
    match id {
        AutoExp => "auto_exposure",
        AutoWb => "auto_wb",
        AutoFocus => "auto_focus",
        Exp => "exposure",
        Wb => "white_balance",
        WbR => "wb_red",
        WbB => "wb_blue",
        Focus => "focus",
        Gain => "gain",
        Gamma => "gamma",
        Brightness => "brightness",
        Contrast => "contrast",
        Temperature => "temperature",
        CoolerTarget => "cooler_target",
        CoolerOn => "cooler_on",
        FanOn => "fan_on",
        CoolerPowerPerc => "cooler_power",
        AverageBin => "average_bin",
        BlackLevel => "black_level",
        ConvGainHcg => "conv_gain_hcg",
        ConvGainHdr => "conv_gain_hdr",
        LowNoise => "low_noise",
        HighFullwell => "high_fullwell",
        Iso => "iso",
        Shutter => "shutter",
        Viewfinder => "viewfinder",
        CaptureTarget => "capture_target",
        KeepImages => "keep_images",
        CaptureDelay => "capture_delay",
        Zoom => "zoom",
        LiveStretch => "live_stretch",
    }
}

pub(crate) fn parse_option(name: &str) -> Option<OlsCamOptionId> {
    use OlsCamOptionId::*;
    Some(match name {
        "auto_exposure" => AutoExp,
        "auto_wb" => AutoWb,
        "auto_focus" => AutoFocus,
        "exposure" => Exp,
        "white_balance" => Wb,
        "wb_red" => WbR,
        "wb_blue" => WbB,
        "focus" => Focus,
        "gain" => Gain,
        "gamma" => Gamma,
        "brightness" => Brightness,
        "contrast" => Contrast,
        "temperature" => Temperature,
        "cooler_target" => CoolerTarget,
        "cooler_on" => CoolerOn,
        "fan_on" => FanOn,
        "cooler_power" => CoolerPowerPerc,
        "average_bin" => AverageBin,
        "black_level" => BlackLevel,
        "conv_gain_hcg" => ConvGainHcg,
        "conv_gain_hdr" => ConvGainHdr,
        "low_noise" => LowNoise,
        "high_fullwell" => HighFullwell,
        "iso" => Iso,
        "shutter" => Shutter,
        "viewfinder" => Viewfinder,
        "capture_target" | "capturetarget" => CaptureTarget,
        "keep_images" => KeepImages,
        "capture_delay" => CaptureDelay,
        "zoom" => Zoom,
        "live_stretch" => LiveStretch,
        _ => return None,
    })
}

pub(crate) fn option_type_name(t: OlsCamOptionType) -> &'static str {
    use OlsCamOptionType::*;
    match t {
        Bool => "bool",
        Number => "number",
        Msec => "msec",
        Percent => "percent",
        Kelvin => "kelvin",
        Celsius => "celsius",
        Selection => "selection",
    }
}

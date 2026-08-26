use axum::{extract::Query, extract::State, http::StatusCode, Json};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use crate::api::api_error::ApiError;
use crate::hardware::camera::{CamParam, FrameData, OlsCamera, StreamFormat};
use crate::hardware::ols_ffi::{OlsCamOptionId, OlsCamOptionType, OlsStreamType};
use crate::state::{AppState, CameraActivity};

#[derive(Debug, Clone, Deserialize)]
pub struct CaptureParams {
    pub exposure: f64,
    pub gain: Option<u32>,
    pub offset: Option<u32>,
    pub binning: Option<u32>,
    pub filter: Option<String>,
    pub temperature: Option<f64>,
    /// Session this exposure belongs to. Captures grouped by session are what
    /// make leakage-safe train/test splits possible later.
    #[serde(default)]
    pub session_id: Option<String>,
    /// light | dark | flat | bias | focus | preview | verification
    #[serde(default)]
    pub capture_type: Option<String>,
}

/// One artifact of a committed capture.
#[derive(Debug, Serialize)]
pub struct CapturedAssetInfo {
    /// native_raw | camera_jpeg | sensor_fits | analysis_luma | thumbnail
    pub role: String,
    pub media_type: String,
    /// Relative to the capture root, so the archive can be relocated.
    pub relative_path: String,
    pub byte_length: i64,
    pub sha256: String,
}

/// Result of a capture.
///
/// Reports the whole bundle rather than one ambiguous `path`: a mirrorless
/// body set to RAW+JPEG produces two native files plus a derived FITS, and
/// collapsing that to a single path throws away everything a consumer needs to
/// find the canonical bytes.
#[derive(Debug, Serialize)]
pub struct ImageData {
    /// Identifies the capture attempt, allocated before the camera is
    /// commanded so a failure is still traceable.
    pub capture_id: String,
    /// The `frames` row. Every asset below belongs to it.
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub status: &'static str,
    pub width: u32,
    pub height: u32,
    /// Directory holding the bundle, relative to the capture root.
    pub relative_dir: String,
    pub assets: Vec<CapturedAssetInfo>,
    /// Deprecated: relative path of the canonical asset.
    ///
    /// Kept so existing clients that read a single `path` keep working while
    /// they migrate to `assets`. It cannot express a RAW+JPEG capture, which is
    /// the whole reason `assets` exists — new code should not use it.
    pub path: Option<String>,
    pub metadata: ImageMetadata,
}

/// Canonical asset for the legacy `path` field: the native RAW if there is
/// one, else the first asset recorded.
fn legacy_path(assets: &[CapturedAssetInfo]) -> Option<String> {
    assets
        .iter()
        .find(|a| a.role == "native_raw")
        .or_else(|| assets.first())
        .map(|a| a.relative_path.clone())
}

#[derive(Debug, Serialize)]
pub struct ImageMetadata {
    /// Exposure the hardware actually delivered, in seconds, read back from
    /// the camera. On a DSLR this is the shutter-ladder step that was selected,
    /// which is not necessarily what was requested.
    pub exposure: f64,
    /// Exposure that was asked for, kept so the two can be compared.
    pub requested_exposure: f64,
    /// Signed difference between request and delivery, in stops.
    pub exposure_error_stops: f64,
    pub gain: u32,
    /// ISO speed, for bodies whose gain control is an ISO enumeration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iso: Option<f64>,
    /// The camera's own shutter label ("1/125", "30"), when it has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shutter: Option<String>,
    pub temperature: f64,
    /// Whether `temperature` was measured, estimated, or absent.
    pub temperature_state: String,
    /// Start of exposure, from the driver's own clock reading taken
    /// immediately before the trigger — this is the FITS `DATE-OBS`, not a
    /// timestamp generated after the download finished.
    pub timestamp: String,
    pub completed_at: String,
    /// Non-fatal problems the operator should see.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// Outcome of pushing one requested setting onto the camera.
struct Applied {
    /// The physical value the hardware ended up at.
    value: f64,
    /// The camera's own label for it, when the control is an enumeration.
    label: Option<String>,
}

/// Requires two completed frame counters after a settings write. The first
/// post-write callback may belong to an exposure that was already in flight.
/// This small pure gate keeps that provenance rule testable without hardware.
#[derive(Debug)]
struct FreshFrameGate {
    baseline: Option<i32>,
    first_seen_without_baseline: Option<i32>,
}

impl FreshFrameGate {
    fn new(baseline: Option<i32>) -> Self {
        Self {
            baseline,
            first_seen_without_baseline: None,
        }
    }

    fn accepts(&mut self, counter: i32) -> bool {
        match self.baseline {
            Some(baseline) => i64::from(counter) - i64::from(baseline) >= 2,
            None => match self.first_seen_without_baseline {
                Some(first) => i64::from(counter) - i64::from(first) >= 1,
                None => {
                    self.first_seen_without_baseline = Some(counter);
                    false
                }
            },
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct StreamStartQuery {
    /// Optional index into GET /camera/formats (selects resolution / binning).
    pub format_index: Option<usize>,
}

fn select_format(
    formats: &[StreamFormat],
    requested: Option<usize>,
) -> Result<(usize, StreamFormat), ApiError> {
    if formats.is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_GATEWAY,
            "camera_has_no_formats",
            "The connected camera reports no acquisition formats",
        ));
    }
    // UVC descriptor order is device-defined.  In particular, inexpensive
    // cameras often advertise YUY2 before MJPEG; YUY2 is useful to the vision
    // loop but the browser preview encoder cannot display it yet.  An omitted
    // index means "give me a usable preview", while an explicit index remains
    // exact and is never silently substituted.
    let index = requested.unwrap_or_else(|| {
        formats
            .iter()
            .position(|format| format.format == OlsStreamType::Mjpeg)
            .or_else(|| {
                formats
                    .iter()
                    .position(|format| stream_type_previewable(format.format))
            })
            .unwrap_or(0)
    });
    let format = formats.get(index).copied().ok_or_else(|| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_camera_format",
            format!(
                "Format index {index} is invalid; the camera reports {} format(s)",
                formats.len()
            ),
        )
    })?;
    Ok((index, format))
}

/// Shared implementation used by direct HTTP, the Django relay and autofocus.
/// All camera commands pass through the same acquisition gate so behavior
/// cannot diverge between transports.
pub(crate) async fn start_stream_control(
    state: &AppState,
    format_index: Option<usize>,
) -> Result<serde_json::Value, ApiError> {
    let _acquisition = state.acquisition.lock().await;

    let runtime = state.camera_runtime.read().unwrap().clone();
    if runtime.connection_id.is_some() && runtime.activity == CameraActivity::Error {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "camera_reconnect_required",
            runtime.last_error.unwrap_or_else(|| {
                "Camera state is uncertain; disconnect and reconnect before streaming".to_string()
            }),
        ));
    }

    if *state.streaming.lock().unwrap() {
        let active = state.camera_runtime.read().unwrap().active_format_index;
        if format_index.is_none() || format_index == active {
            return Ok(serde_json::json!({
                "status": "streaming",
                "already_running": true,
                "format_index": active,
            }));
        }
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "camera_stream_already_running",
            "Stop the current stream before selecting another format",
        ));
    }

    let frame_store = state.latest_frame.clone();
    let (index, format) = {
        let mut cam_lock = state.camera.lock().unwrap();
        let camera = cam_lock.as_mut().ok_or_else(ApiError::no_camera)?;
        let formats = camera
            .formats()
            .map_err(|e| ApiError::hardware(format!("Could not read camera formats: {e}")))?;
        let (index, format) = select_format(&formats, format_index)?;

        // This is a new stream generation. Old preview/vision values must be
        // gone before a callback from the new device is allowed to arrive.
        state.clear_camera_observations();
        let generation = state.current_camera_generation();
        if let Err(error) = camera.start_stream(
            &format,
            frame_store,
            state.camera_config.saturation_adu,
            generation,
            state.camera_generation.clone(),
        ) {
            let message = format!(
                "Could not start camera stream: {error}. Disconnect and reconnect before retrying"
            );
            let mut runtime = state.camera_runtime.write().unwrap();
            runtime.activity = CameraActivity::Error;
            runtime.active_format_index = None;
            runtime.last_error = Some(message.clone());
            return Err(ApiError::hardware(message));
        }
        (index, format)
    };

    *state.streaming.lock().unwrap() = true;
    {
        let mut runtime = state.camera_runtime.write().unwrap();
        runtime.activity = CameraActivity::Streaming;
        runtime.streaming = true;
        runtime.active_format_index = Some(index);
        runtime.operation_id = None;
        runtime.exposure_started_at = None;
        runtime.exposure_duration_s = None;
        runtime.last_error = None;
    }

    Ok(serde_json::json!({
        "status": "streaming",
        "already_running": false,
        "format_index": index,
        "format": format!("{:?}", format.format),
        "width": format.width,
        "height": format.height,
        "bin": format.bin,
        "roi_num": format.roi_num,
        "roi_den": format.roi_den,
        "mode": if format.is_liveview() { "liveview" } else { "still" },
    }))
}

/// Start the camera stream. Optionally pick a stream format / binning mode by
/// index (see GET /camera/formats); defaults to the first available format.
pub async fn start_stream(
    State(state): State<AppState>,
    Query(q): Query<StreamStartQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    start_stream_control(&state, q.format_index).await.map(Json)
}

/// Single-shot capture.
///
/// Takes one exposure through the driver's dedicated still path and commits it
/// to the canonical store. It does **not** read `state.latest_frame`.
///
/// That buffer is the preview bus: whatever frame most recently arrived from a
/// free-running stream. Sampling it for a science exposure meant the default
/// DSLR capture saved a low-resolution 8-bit live-view JPEG, stamped it with a
/// still-camera exposure, and produced no RAW at all — and even on a still
/// stream, a setting written mid-flight could land while an exposure was
/// already underway, so the frame returned carried the *previous* settings
/// under the new metadata. Neither is recoverable after the fact, because
/// nothing in the record says it happened.
pub async fn capture(
    State(state): State<AppState>,
    Json(params): Json<CaptureParams>,
) -> Result<Json<ImageData>, ApiError> {
    perform_capture(state, params).await.map(Json)
}

/// The capture implementation, shared by the HTTP handler above and the cloud
/// relay in `relay_commands.rs`.
///
/// One implementation on purpose. The relay used to carry its own copy, so
/// every fix had to be made twice and stayed correct only until someone forgot.
pub(crate) async fn perform_capture(
    state: AppState,
    params: CaptureParams,
) -> Result<ImageData, ApiError> {
    // Detach the hardware transaction from the HTTP request lifetime. If a
    // browser tab closes mid-exposure, Tokio keeps this task (and its ownership
    // gate) alive until the camera and atomic capture-store commit finish.
    let cleanup_state = state.clone();
    tokio::spawn(async move { perform_capture_owned(state, params).await })
        .await
        .map_err(|error| {
            let message = format!("Capture task failed: {error}");
            finish_capture_runtime(&cleanup_state, Some(message.clone()));
            ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "capture_task_failed",
                message,
            )
        })?
}

async fn perform_capture_owned(
    state: AppState,
    params: CaptureParams,
) -> Result<ImageData, ApiError> {
    // Serialise everything that commands the camera. Held across the whole
    // exposure — which can be minutes — so a second request cannot change the
    // shutter out from under one already in flight.
    let _acquisition = state.acquisition.lock().await;

    let capture_id = uuid::Uuid::new_v4().to_string();
    let frame_id = uuid::Uuid::new_v4().to_string();
    let capture_type = params
        .capture_type
        .clone()
        .unwrap_or_else(|| "light".to_string());

    {
        let mut runtime = state.camera_runtime.write().unwrap();
        runtime.activity = CameraActivity::Exposing;
        runtime.operation_id = Some(capture_id.clone());
        runtime.exposure_started_at = Some(chrono::Utc::now());
        runtime.exposure_duration_s = Some(params.exposure);
        runtime.last_error = None;
    }

    tracing::info!(
        "Capture {capture_id}: {}s, gain {:?}, type {capture_type}",
        params.exposure,
        params.gain
    );

    // Record the attempt before touching hardware, so a failure leaves a trace
    // instead of being indistinguishable from "nobody asked".
    let requested_json = serde_json::json!({
        "exposure": params.exposure,
        "gain": params.gain,
        "offset": params.offset,
        "binning": params.binning,
        "filter": params.filter,
        "temperature": params.temperature,
        "capture_type": capture_type,
    });
    if let Err(e) = state
        .capture_store
        .begin_attempt(
            state.local_store.pool(),
            &capture_id,
            params.session_id.as_deref(),
            &state.rig_id,
            &requested_json,
        )
        .await
    {
        tracing::error!("Could not record capture attempt: {e}");
        finish_capture_runtime(&state, Some(e.to_string()));
        return Err(ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "capture_attempt_store_failed",
            format!("Could not record capture attempt: {e}"),
        ));
    }

    match run_capture(&state, &capture_id, &frame_id, &capture_type, &params).await {
        Ok(image) => {
            finish_capture_runtime(&state, None);
            Ok(image)
        }
        Err(err) => {
            tracing::error!("Capture {capture_id} failed: {}", err.detail);
            state
                .capture_store
                .fail_attempt(state.local_store.pool(), &capture_id, &err.detail)
                .await;
            finish_capture_runtime(&state, Some(err.detail.clone()));
            Err(ApiError::new(
                err.status,
                capture_error_code(err.status),
                err.detail,
            ))
        }
    }
}

fn capture_error_code(status: StatusCode) -> &'static str {
    match status {
        StatusCode::PRECONDITION_FAILED => "camera_not_ready",
        StatusCode::UNPROCESSABLE_ENTITY | StatusCode::BAD_REQUEST => "invalid_capture_settings",
        StatusCode::CONFLICT => "camera_state_conflict",
        StatusCode::GATEWAY_TIMEOUT => "capture_timeout",
        StatusCode::UNSUPPORTED_MEDIA_TYPE => "capture_format_unsupported",
        StatusCode::BAD_GATEWAY => "camera_hardware_error",
        _ => "capture_failed",
    }
}

fn finish_capture_runtime(state: &AppState, error: Option<String>) {
    let streaming = *state.streaming.lock().unwrap();
    let mut runtime = state.camera_runtime.write().unwrap();
    runtime.activity = if runtime.connection_id.is_none() {
        CameraActivity::Disconnected
    } else if streaming {
        CameraActivity::Streaming
    } else {
        CameraActivity::Idle
    };
    runtime.operation_id = None;
    runtime.exposure_started_at = None;
    runtime.exposure_duration_s = None;
    runtime.last_error = error;
}

fn mark_capture_processing(state: &AppState) {
    let mut runtime = state.camera_runtime.write().unwrap();
    if runtime.activity == CameraActivity::Exposing {
        runtime.activity = CameraActivity::Processing;
        runtime.exposure_started_at = None;
        runtime.exposure_duration_s = None;
    }
}

/// Internal error carrying both the HTTP status and a message worth recording.
struct CaptureError {
    status: StatusCode,
    detail: String,
}

impl CaptureError {
    fn new(status: StatusCode, detail: impl Into<String>) -> Self {
        Self {
            status,
            detail: detail.into(),
        }
    }
}

async fn run_capture(
    state: &AppState,
    capture_id: &str,
    frame_id: &str,
    capture_type: &str,
    params: &CaptureParams,
) -> Result<ImageData, CaptureError> {
    let mut warnings: Vec<String> = Vec::new();

    if !params.exposure.is_finite() || params.exposure <= 0.0 {
        return Err(CaptureError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Exposure must be a finite value greater than zero seconds",
        ));
    }

    // ─── 1. Confirm the camera can take a real still ───
    let (supports_still, camera_name) = {
        let cam_lock = state.camera.lock().unwrap();
        let camera = cam_lock.as_ref().ok_or_else(|| {
            CaptureError::new(StatusCode::PRECONDITION_FAILED, "No camera connected")
        })?;
        (
            camera.supports_still_capture(),
            camera.name().unwrap_or_else(|_| "Unknown".to_string()),
        )
    };

    if !supports_still {
        // Sensor cameras (ASI/SVBony/ToupTek) have no separate still path; the
        // stream *is* the acquisition. Handled separately below rather than
        // pretending the DSLR path applies.
        return capture_from_stream(state, capture_id, frame_id, capture_type, params, warnings)
            .await;
    }

    if params.binning.unwrap_or(1) != 1 {
        return Err(CaptureError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "This dedicated-still camera does not expose hardware binning",
        ));
    }
    let applied_offset = {
        let cam_lock = state.camera.lock().unwrap();
        let camera = cam_lock.as_ref().ok_or_else(|| {
            CaptureError::new(StatusCode::PRECONDITION_FAILED, "No camera connected")
        })?;
        let options = camera.supported_options().map_err(|error| {
            CaptureError::new(
                StatusCode::BAD_GATEWAY,
                format!("Could not list camera controls: {error}"),
            )
        })?;
        apply_offset(camera, &options, params.offset).map_err(|status| {
            CaptureError::new(
                status,
                "Requested offset is not supported or could not be verified",
            )
        })?
    };

    // ─── 2. Take the exposure ───
    //
    // libgphoto2 is blocking and a full-frame download over USB 2.0 takes
    // seconds. Running it on an Axum worker would stall every other request
    // for the duration.
    let camera_arc = state.camera.clone();
    let exposure = params.exposure;
    let sensitivity = params.gain.map(|g| g as f64);
    let native = tokio::task::spawn_blocking(move || {
        let cam_lock = camera_arc.lock().unwrap();
        let camera = cam_lock
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Camera disconnected before the exposure started"))?;
        camera.capture_still(exposure, sensitivity)
    })
    .await
    .map_err(|e| {
        CaptureError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("capture task panicked: {e}"),
        )
    })?
    .map_err(|e| CaptureError::new(StatusCode::BAD_GATEWAY, e.to_string()))?;

    let (capture, handle) = native;
    mark_capture_processing(state);

    let exposure_error_stops =
        crate::hardware::dslr_controls::stops_from(params.exposure, capture.applied_exposure_s);
    if exposure_error_stops.is_finite() && exposure_error_stops.abs() >= 0.25 {
        warnings.push(format!(
            "Camera delivered {:.4}s for a {:.4}s request ({:+.2} stops)",
            capture.applied_exposure_s, params.exposure, exposure_error_stops
        ));
    }
    if capture.applied_sensitivity <= 0.0 && params.gain.is_some() {
        warnings.push("Camera reported no numeric ISO; gain was not applied".to_string());
    }

    // ─── 3. Sensor temperature, with its provenance ───
    let (temperature, temperature_state, temperature_source) = read_temperature(state);

    // ─── 4. Assemble the bundle ───
    let mut assets: Vec<crate::storage::capture_store::PendingAsset> = Vec::new();
    let mut width = 0u32;
    let mut height = 0u32;

    for native_asset in &capture.assets {
        let role = match native_asset.role() {
            "camera_jpeg" => crate::storage::capture_store::AssetRole::CameraJpeg,
            _ => crate::storage::capture_store::AssetRole::NativeRaw,
        };
        let filename = format!("image.{}", native_asset.extension);
        assets.push(
            crate::storage::capture_store::PendingAsset::new(
                role,
                filename,
                native_asset.media_type(),
                native_asset.bytes.clone(),
            )
            .with_original_filename(
                std::path::Path::new(&native_asset.camera_path)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| native_asset.camera_path.clone()),
            ),
        );
    }

    // Derive a FITS from the primary asset. Deliberately non-fatal: a durably
    // saved CR3 *is* the exposure, and losing a derivation we can redo later
    // must never discard the original.
    if let Some(primary) = capture.primary() {
        match crate::fits::fits_from_source(
            &primary.bytes,
            &primary.extension,
            &crate::fits::FitsMeta {
                exposure_s: capture.applied_exposure_s,
                gain: capture.applied_sensitivity.max(0.0) as u32,
                temperature_c: temperature.unwrap_or(f64::NAN),
                binning: params.binning.unwrap_or(1),
                instrument: camera_name.clone(),
                date_obs: unix_to_rfc3339(capture.started_at),
                iso: (capture.applied_sensitivity > 0.0).then_some(capture.applied_sensitivity),
                shutter_label: (!capture.shutter_label.is_empty())
                    .then(|| capture.shutter_label.clone()),
                source_file: Some(format!("image.{}", primary.extension)),
                // Filled in by the decoder, which is the only thing that knows
                // the body's black and saturation levels.
                calibration: None,
            },
        ) {
            Ok(Some(fits)) => {
                width = fits.width;
                height = fits.height;

                // Everything the decoder chose not to apply. Recorded so a
                // calibration step can apply it later, and so a frame can be
                // reproduced from the archive alone — none of it is recoverable
                // from the FITS pixels, and cam_mul is not recoverable at all.
                let mut decode_provenance = serde_json::json!({
                    "producer": concat!("achernar-daemon ", env!("CARGO_PKG_VERSION")),
                    "decoder": fits.decoder,
                });
                if let Some(info) = fits.raw_info.as_ref() {
                    decode_provenance["raw"] = serde_json::json!({
                        "make": info.make,
                        "model": info.model,
                        "black_level": info.black_level,
                        "cblack": info.cblack,
                        "cblack_pattern": info.cblack_pattern,
                        "black_level_complete": info.black_level_is_complete(),
                        "white_level": info.white_level,
                        "cam_mul": info.cam_mul,
                        "bayer": format!("{:?}", info.bayer),
                        // Stated explicitly so a reader never has to infer it
                        // from the absence of a field.
                        "black_subtracted": false,
                        "scaled": false,
                        "white_balance_applied": false,
                    });
                    if !info.black_level_is_complete() {
                        warnings.push(format!(
                            "{} {} reports a {}x{} black-level table; only the four \
                             per-channel offsets were recorded. Calibrate from bias \
                             frames rather than the header.",
                            info.make, info.model, info.cblack_pattern[0], info.cblack_pattern[1]
                        ));
                    }
                    if info.bayer == crate::hardware::ols_ffi::OlsBayerType::Na {
                        warnings.push(
                            "Decoder found no Bayer pattern; the frame is being filed as \
                             mono. If this is a colour body, BAYERPAT is missing and \
                             debayering downstream will fail."
                                .to_string(),
                        );
                    }
                }

                assets.push(
                    crate::storage::capture_store::PendingAsset::new(
                        crate::storage::capture_store::AssetRole::SensorFits,
                        "image.fits",
                        "application/fits",
                        fits.bytes,
                    )
                    .derived_from(if primary.role() == "camera_jpeg" {
                        crate::storage::capture_store::AssetRole::CameraJpeg
                    } else {
                        crate::storage::capture_store::AssetRole::NativeRaw
                    })
                    .with_dimensions(fits.width as i64, fits.height as i64, fits.bit_depth as i64)
                    .with_provenance(decode_provenance),
                );
            }
            Ok(None) => {
                warnings.push(format!(
                    "No FITS derived: .{} needs a decoder this build does not have. \
                     The native file is preserved and can be converted later.",
                    primary.extension
                ));
            }
            Err(e) => {
                warnings.push(format!(
                    "FITS derivation failed ({e}); native file preserved"
                ));
            }
        }
    }

    // ─── 5. Commit ───
    let pending = crate::storage::capture_store::PendingCapture {
        capture_id: capture_id.to_string(),
        frame_id: frame_id.to_string(),
        session_id: params.session_id.clone(),
        rig_id: state.rig_id.clone(),
        camera_name: Some(camera_name),
        driver_name: state.camera_runtime.read().unwrap().driver_id.clone(),
        capture_type: capture_type.to_string(),
        exposure_started_at: capture.started_at,
        exposure_completed_at: capture.completed_at,
        requested_exposure_s: params.exposure,
        requested_gain: params.gain.map(|g| g as f64),
        applied_exposure_s: capture.applied_exposure_s,
        iso: (capture.applied_sensitivity > 0.0).then_some(capture.applied_sensitivity),
        shutter_label: (!capture.shutter_label.is_empty()).then(|| capture.shutter_label.clone()),
        offset_val: applied_offset.map(|value| value.round() as i64),
        binning: 1,
        filter_name: params.filter.clone(),
        focuser_position: read_focuser_position(state),
        temperature,
        temperature_source,
        temperature_state,
        acquisition: serde_json::json!({
            "shutter_label": capture.shutter_label,
            "iso_label": capture.iso_label,
            "measured_duration_s": capture.exposure_duration_s(),
            "asset_count": capture.assets.len(),
        }),
        assets,
    };

    let committed = state
        .capture_store
        .persist_capture(state.local_store.pool(), pending)
        .await
        .map_err(|e| {
            CaptureError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("persist capture: {e}"),
            )
        })?;

    // ─── 6. Only now is it safe to reclaim the card ───
    // The bytes are on disk, hashed, and referenced by a committed row.
    {
        let cam_lock = state.camera.lock().unwrap();
        if let Some(camera) = cam_lock.as_ref() {
            if let Err(e) = camera.delete_captured(&handle) {
                // Not fatal: wasted card space, versus destroying an exposure.
                tracing::warn!("Could not delete captured files from the camera: {e}");
                warnings.push(format!("Files left on the camera: {e}"));
            }
        }
    }
    drop(handle);

    // Build the asset list once; `legacy_path` needs it before ImageData is
    // constructed, and `committed.assets` is consumed by the conversion.
    let asset_infos: Vec<CapturedAssetInfo> = committed
        .assets
        .into_iter()
        .map(|a| CapturedAssetInfo {
            role: a.role,
            media_type: a.media_type,
            relative_path: a.relative_path,
            byte_length: a.byte_length,
            sha256: a.sha256,
        })
        .collect();
    let legacy = legacy_path(&asset_infos);

    Ok(ImageData {
        capture_id: committed.capture_id,
        id: committed.frame_id,
        session_id: committed.session_id,
        status: committed.status,
        width,
        height,
        relative_dir: committed.relative_dir,
        assets: asset_infos,
        path: legacy,
        metadata: ImageMetadata {
            exposure: capture.applied_exposure_s,
            requested_exposure: params.exposure,
            exposure_error_stops: if exposure_error_stops.is_finite() {
                exposure_error_stops
            } else {
                0.0
            },
            gain: capture.applied_sensitivity.max(0.0) as u32,
            iso: (capture.applied_sensitivity > 0.0).then_some(capture.applied_sensitivity),
            shutter: (!capture.shutter_label.is_empty()).then(|| capture.shutter_label.clone()),
            temperature: temperature.unwrap_or(f64::NAN),
            temperature_state: temperature_state_name(temperature_state).to_string(),
            timestamp: unix_to_rfc3339(capture.started_at),
            completed_at: unix_to_rfc3339(capture.completed_at),
            warnings,
        },
    })
}

/// Acquisition path for sensor cameras that have no separate still operation
/// (ASI, SVBony, ToupTek, the SV105): the stream *is* the exposure.
///
/// Still avoids the old defect — it applies the settings, discards the first
/// post-write frame as potentially already in flight, then waits for the next
/// frame in the same stream generation. It fails rather than substituting a
/// stale frame.
/// A capture that cannot prove which exposure it returned is not a capture.
async fn capture_from_stream(
    state: &AppState,
    capture_id: &str,
    frame_id: &str,
    capture_type: &str,
    params: &CaptureParams,
    mut warnings: Vec<String>,
) -> Result<ImageData, CaptureError> {
    if !*state.streaming.lock().unwrap() {
        return Err(CaptureError::new(
            StatusCode::PRECONDITION_FAILED,
            "This camera captures through its stream; start the stream first",
        ));
    }

    let stream_generation = state.current_camera_generation();
    let runtime = state.camera_runtime.read().unwrap().clone();
    let active_format_index = runtime.active_format_index.ok_or_else(|| {
        CaptureError::new(
            StatusCode::PRECONDITION_FAILED,
            "The active stream has no recorded format; stop and restart it",
        )
    })?;

    // Baseline before settings are applied. The gate below rejects both this
    // counter and the first counter after it, covering an exposure already in
    // flight while the hardware accepted the writes.
    let baseline = state
        .latest_frame
        .lock()
        .unwrap()
        .as_ref()
        .map(|f| f.frame_counter);

    let (applied_exposure, applied_gain, applied_offset, camera_name, actual_binning) = {
        let cam_lock = state.camera.lock().unwrap();
        let camera = cam_lock.as_ref().ok_or_else(|| {
            CaptureError::new(StatusCode::PRECONDITION_FAILED, "No camera connected")
        })?;
        let formats = camera.formats().map_err(|error| {
            CaptureError::new(
                StatusCode::BAD_GATEWAY,
                format!("Could not read camera formats: {error}"),
            )
        })?;
        let active_format = formats.get(active_format_index).ok_or_else(|| {
            CaptureError::new(
                StatusCode::CONFLICT,
                "The active stream format is no longer reported by the camera",
            )
        })?;
        let actual_binning = active_format.bin.max(1) as u32;
        if let Some(requested) = params.binning {
            if requested != actual_binning {
                return Err(CaptureError::new(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!(
                        "Capture requested {requested}x binning, but active format \
                         {active_format_index} is {actual_binning}x; select the matching \
                         format before capture"
                    ),
                ));
            }
        }
        let options = camera.supported_options().map_err(|e| {
            CaptureError::new(StatusCode::BAD_GATEWAY, format!("list options: {e}"))
        })?;
        let exposure = apply_exposure(camera, &options, params.exposure, &mut warnings)
            .map_err(|s| CaptureError::new(s, "Failed to apply the requested exposure"))?;
        let gain = apply_gain(camera, &options, params.gain, &mut warnings)
            .map_err(|s| CaptureError::new(s, "Failed to apply the requested gain"))?;
        let offset = apply_offset(camera, &options, params.offset).map_err(|status| {
            CaptureError::new(
                status,
                "Requested offset is not supported or could not be verified",
            )
        })?;
        if let Some(temp) = params.temperature {
            if options.contains(&OlsCamOptionId::CoolerTarget) {
                if let Err(e) = camera.set_parameter(OlsCamOptionId::CoolerTarget, temp) {
                    warnings.push(format!("Cooler setpoint rejected: {e}"));
                }
            }
        }
        (
            exposure,
            gain,
            offset,
            camera.name().unwrap_or_else(|_| "Unknown".to_string()),
            actual_binning,
        )
    };

    let actual_exposure_s = applied_exposure.value;

    // Two exposures plus slack: the in-flight exposure must finish and be
    // discarded before a frame taken under the new settings can arrive.
    let timeout = Duration::from_secs_f64((actual_exposure_s * 3.0 + 20.0).clamp(20.0, 900.0));
    let deadline = Instant::now() + timeout;
    let mut fresh_gate = FreshFrameGate::new(baseline);
    let frame = loop {
        {
            let guard = state.latest_frame.lock().unwrap();
            if let Some(f) = guard.as_ref() {
                if f.stream_generation == stream_generation && fresh_gate.accepts(f.frame_counter) {
                    break f.clone();
                }
            }
        }
        if Instant::now() >= deadline {
            return Err(CaptureError::new(
                StatusCode::GATEWAY_TIMEOUT,
                format!(
                    "No frame taken under the requested settings arrived within {:.0}s",
                    timeout.as_secs_f64()
                ),
            ));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    let completed_at = unix_now();
    mark_capture_processing(state);

    let (temperature, temperature_state, temperature_source) = read_temperature(state);

    let meta = crate::fits::FitsMeta {
        exposure_s: actual_exposure_s,
        gain: applied_gain.value.max(0.0) as u32,
        temperature_c: temperature.unwrap_or(f64::NAN),
        binning: actual_binning,
        instrument: camera_name.clone(),
        date_obs: unix_to_rfc3339(frame.unix_timestamp),
        iso: applied_gain.label.as_ref().map(|_| applied_gain.value),
        shutter_label: applied_exposure.label.clone(),
        source_file: None,
        // Stream frames come through the driver's own pipeline, which does not
        // report a black or saturation level. Asserting one here would be a
        // guess written into a file that reads like a measurement.
        calibration: None,
    };
    let fits_bytes = crate::fits::encode_capture(&frame, &meta).map_err(|e| {
        CaptureError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("encode FITS: {e}"),
        )
    })?;

    let bit_depth = stream_frame_bit_depth(frame.format).ok_or_else(|| {
        CaptureError::new(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            format!("Stream format {:?} cannot be persisted", frame.format),
        )
    })?;

    // This FITS is produced by the daemon from the driver's stream bytes. It is
    // canonical for this sensor capture, but it is not a camera-authored RAW
    // container and must not be labelled/uploaded as one.
    let assets = vec![crate::storage::capture_store::PendingAsset::new(
        crate::storage::capture_store::AssetRole::SensorFits,
        "image.fits",
        "application/fits",
        fits_bytes,
    )
    .with_dimensions(frame.width as i64, frame.height as i64, bit_depth)
    .with_provenance(serde_json::json!({
        "producer": concat!("achernar-daemon ", env!("CARGO_PKG_VERSION")),
        "stream_format": format!("{:?}", frame.format),
        "source": "driver_stream",
        "note": "sensor camera: no separate still path; daemon encoded the selected stream frame as FITS",
    }))];

    let pending = crate::storage::capture_store::PendingCapture {
        capture_id: capture_id.to_string(),
        frame_id: frame_id.to_string(),
        session_id: params.session_id.clone(),
        rig_id: state.rig_id.clone(),
        camera_name: Some(camera_name),
        driver_name: runtime.driver_id.clone(),
        capture_type: capture_type.to_string(),
        exposure_started_at: frame.unix_timestamp,
        exposure_completed_at: completed_at,
        requested_exposure_s: params.exposure,
        requested_gain: params.gain.map(|g| g as f64),
        applied_exposure_s: actual_exposure_s,
        iso: applied_gain.label.as_ref().map(|_| applied_gain.value),
        shutter_label: applied_exposure.label.clone(),
        offset_val: applied_offset.map(|value| value.round() as i64),
        binning: actual_binning as i64,
        filter_name: params.filter.clone(),
        focuser_position: read_focuser_position(state),
        temperature,
        temperature_source,
        temperature_state,
        acquisition: serde_json::json!({
            "stream_generation": frame.stream_generation,
            "frame_counter": frame.frame_counter,
            "baseline_frame_counter": baseline,
            "stream_format": format!("{:?}", frame.format),
            "active_format_index": active_format_index,
            "requested_binning": params.binning,
            "applied_binning": actual_binning,
            "requested_offset": params.offset,
            "applied_offset": applied_offset,
        }),
        assets,
    };

    let committed = state
        .capture_store
        .persist_capture(state.local_store.pool(), pending)
        .await
        .map_err(|e| {
            CaptureError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("persist capture: {e}"),
            )
        })?;

    // Build the asset list once; `legacy_path` needs it before ImageData is
    // constructed, and `committed.assets` is consumed by the conversion.
    let asset_infos: Vec<CapturedAssetInfo> = committed
        .assets
        .into_iter()
        .map(|a| CapturedAssetInfo {
            role: a.role,
            media_type: a.media_type,
            relative_path: a.relative_path,
            byte_length: a.byte_length,
            sha256: a.sha256,
        })
        .collect();
    let legacy = legacy_path(&asset_infos);

    Ok(ImageData {
        capture_id: committed.capture_id,
        id: committed.frame_id,
        session_id: committed.session_id,
        status: committed.status,
        width: frame.width as u32,
        height: frame.height as u32,
        relative_dir: committed.relative_dir,
        assets: asset_infos,
        path: legacy,
        metadata: ImageMetadata {
            exposure: actual_exposure_s,
            requested_exposure: params.exposure,
            exposure_error_stops: 0.0,
            gain: applied_gain.value.max(0.0) as u32,
            iso: applied_gain.label.as_ref().map(|_| applied_gain.value),
            shutter: applied_exposure.label,
            temperature: temperature.unwrap_or(f64::NAN),
            temperature_state: temperature_state_name(temperature_state).to_string(),
            timestamp: unix_to_rfc3339(frame.unix_timestamp),
            completed_at: unix_to_rfc3339(completed_at),
            warnings,
        },
    })
}

fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn unix_to_rfc3339(unix_seconds: f64) -> String {
    let secs = unix_seconds.trunc() as i64;
    let nanos = ((unix_seconds - unix_seconds.trunc()) * 1e9).round() as u32;
    chrono::DateTime::from_timestamp(secs, nanos.min(999_999_999))
        .unwrap_or_else(chrono::Utc::now)
        .to_rfc3339()
}

fn temperature_state_name(state: crate::storage::capture_store::TemperatureState) -> &'static str {
    use crate::storage::capture_store::TemperatureState::*;
    match state {
        Measured => "measured",
        Estimated => "estimated",
        Missing => "missing",
    }
}

/// Read the sensor temperature along with how it was obtained.
///
/// A DSLR exposes no sensor thermometer, so a capture from one records
/// `missing` rather than a plausible-looking number. Recording an invented
/// value is worse than recording none: a drift model cannot tell the two apart.
pub(crate) fn read_temperature(
    state: &AppState,
) -> (
    Option<f64>,
    crate::storage::capture_store::TemperatureState,
    Option<String>,
) {
    use crate::storage::capture_store::TemperatureState;
    let cam_lock = state.camera.lock().unwrap();
    match cam_lock
        .as_ref()
        .and_then(|c| c.get_parameter(OlsCamOptionId::Temperature).ok())
        .map(|p| p.cur_val)
    {
        Some(v) if v.is_finite() => (
            Some(v),
            TemperatureState::Measured,
            Some("camera_sensor".to_string()),
        ),
        _ => (None, TemperatureState::Missing, None),
    }
}

fn read_focuser_position(state: &AppState) -> Option<i64> {
    let focuser = state.focuser.lock().ok()?;
    focuser.as_ref()?.position().ok().map(|p| p as i64)
}

fn disable_auto_exposure(camera: &OlsCamera, options: &[OlsCamOptionId]) -> Result<(), StatusCode> {
    if !options.contains(&OlsCamOptionId::AutoExp) {
        return Ok(());
    }
    let auto = camera
        .get_parameter(OlsCamOptionId::AutoExp)
        .map_err(|error| {
            tracing::error!("Failed to read auto-exposure state: {error}");
            StatusCode::BAD_GATEWAY
        })?;
    if auto.cur_val <= 0.5 {
        return Ok(());
    }
    if auto.read_only {
        tracing::error!("Camera auto exposure is active and read-only");
        return Err(StatusCode::CONFLICT);
    }
    camera
        .set_parameter(OlsCamOptionId::AutoExp, 0.0)
        .map_err(|error| {
            tracing::error!("Failed to disable auto exposure: {error}");
            StatusCode::BAD_GATEWAY
        })?;
    let applied = camera
        .get_parameter(OlsCamOptionId::AutoExp)
        .map_err(|error| {
            tracing::error!("Failed to verify manual exposure mode: {error}");
            StatusCode::BAD_GATEWAY
        })?;
    if applied.cur_val > 0.5 {
        tracing::error!("Camera remained in auto-exposure mode after a manual request");
        return Err(StatusCode::CONFLICT);
    }
    Ok(())
}

/// Apply the sensor pedestal/offset and return the camera's readback. A value
/// supplied in a capture request is never metadata-only: unsupported hardware
/// rejects it so the archive cannot claim a setting the camera did not use.
fn apply_offset(
    camera: &OlsCamera,
    options: &[OlsCamOptionId],
    requested: Option<u32>,
) -> Result<Option<f64>, StatusCode> {
    let Some(requested) = requested else {
        return Ok(None);
    };
    if !options.contains(&OlsCamOptionId::BlackLevel) {
        tracing::warn!(
            "Capture requested offset {requested}, but camera exposes no offset control"
        );
        return Err(StatusCode::UNPROCESSABLE_ENTITY);
    }
    let parameter = camera
        .get_parameter(OlsCamOptionId::BlackLevel)
        .map_err(|_| StatusCode::BAD_GATEWAY)?;
    let requested = requested as f64;
    if parameter.read_only || requested < parameter.min_val || requested > parameter.max_val {
        return Err(StatusCode::UNPROCESSABLE_ENTITY);
    }
    camera
        .set_parameter(OlsCamOptionId::BlackLevel, requested)
        .map_err(|_| StatusCode::BAD_GATEWAY)?;
    let applied = camera
        .get_parameter(OlsCamOptionId::BlackLevel)
        .map_err(|_| StatusCode::BAD_GATEWAY)?
        .cur_val;
    Ok(Some(applied))
}

fn stream_frame_bit_depth(format: OlsStreamType) -> Option<i64> {
    match format {
        OlsStreamType::Mjpeg
        | OlsStreamType::Rgb24
        | OlsStreamType::Mono8
        | OlsStreamType::Raw8 => Some(8),
        OlsStreamType::Mono16 | OlsStreamType::Raw16 => Some(16),
        _ => None,
    }
}

/// Push the requested exposure onto whichever control this camera has.
///
/// Returns the exposure the hardware settled on. Errors are surfaced, never
/// swallowed — a capture whose exposure could not be set is not a capture.
fn apply_exposure(
    camera: &crate::hardware::camera::OlsCamera,
    options: &[OlsCamOptionId],
    requested_s: f64,
    warnings: &mut Vec<String>,
) -> Result<Applied, StatusCode> {
    use crate::hardware::dslr_controls::nearest_shutter;

    if options.contains(&OlsCamOptionId::Exp) || options.contains(&OlsCamOptionId::Shutter) {
        disable_auto_exposure(camera, options)?;
    }

    // Continuous exposure control (astronomy cameras): milliseconds.
    if options.contains(&OlsCamOptionId::Exp) {
        camera
            .set_parameter(OlsCamOptionId::Exp, requested_s * 1000.0)
            .map_err(|e| {
                tracing::error!("Failed to set exposure: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;
        // Read back: drivers clamp to their own min/max and round to a step.
        let applied_ms = camera
            .get_parameter(OlsCamOptionId::Exp)
            .map(|p| p.cur_val)
            .map_err(|error| {
                tracing::error!("Failed to read back exposure: {error}");
                StatusCode::BAD_GATEWAY
            })?;
        return Ok(Applied {
            value: applied_ms / 1000.0,
            label: None,
        });
    }

    // Enumerated shutter ladder (DSLR / mirrorless).
    if options.contains(&OlsCamOptionId::Shutter) {
        let labels = camera
            .parameter_choices(OlsCamOptionId::Shutter)
            .map_err(|e| {
                tracing::error!("Failed to read shutter choices: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;

        match nearest_shutter(&labels, requested_s) {
            Some(choice) => {
                tracing::info!(
                    "Shutter: requested {:.4}s -> choice {} \"{}\" ({:.4}s)",
                    requested_s,
                    choice.index,
                    choice.label,
                    choice.value
                );
                camera
                    .set_parameter(OlsCamOptionId::Shutter, choice.index as f64)
                    .map_err(|e| {
                        tracing::error!("Failed to set shutter to {}: {}", choice.label, e);
                        StatusCode::INTERNAL_SERVER_ERROR
                    })?;
                return Ok(Applied {
                    value: choice.value,
                    label: Some(choice.label),
                });
            }
            None => {
                // Body is in bulb/auto only, or reported no usable labels.
                warnings.push(format!(
                    "Camera exposed no numeric shutter speed (choices: {}); \
                     exposure left as the camera has it",
                    if labels.is_empty() {
                        "none".to_string()
                    } else {
                        labels.join(", ")
                    }
                ));
                return Ok(Applied {
                    value: requested_s,
                    label: None,
                });
            }
        }
    }

    warnings.push(
        "Camera exposes no exposure control; the value on the body was used as-is".to_string(),
    );
    Ok(Applied {
        value: requested_s,
        label: None,
    })
}

/// Push the requested gain onto whichever control this camera has (continuous
/// `Gain`, or an enumerated `Iso` ladder on a DSLR).
fn apply_gain(
    camera: &crate::hardware::camera::OlsCamera,
    options: &[OlsCamOptionId],
    requested: Option<u32>,
    warnings: &mut Vec<String>,
) -> Result<Applied, StatusCode> {
    use crate::hardware::dslr_controls::nearest_iso;

    if options.contains(&OlsCamOptionId::Gain) {
        if let Some(gain) = requested {
            camera
                .set_parameter(OlsCamOptionId::Gain, gain as f64)
                .map_err(|e| {
                    tracing::error!("Failed to set gain: {}", e);
                    StatusCode::INTERNAL_SERVER_ERROR
                })?;
        }
        let applied = camera
            .get_parameter(OlsCamOptionId::Gain)
            .map(|p| p.cur_val)
            .map_err(|error| {
                tracing::error!("Failed to read back gain: {error}");
                StatusCode::BAD_GATEWAY
            })?;
        return Ok(Applied {
            value: applied,
            label: None,
        });
    }

    if options.contains(&OlsCamOptionId::Iso) {
        let labels = camera.parameter_choices(OlsCamOptionId::Iso).map_err(|e| {
            tracing::error!("Failed to read ISO choices: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        // No gain requested: report what the body is already set to rather
        // than changing it. Manual control stays the operator's.
        let current_index = camera
            .get_parameter(OlsCamOptionId::Iso)
            .map(|p| p.cur_val as usize)
            .unwrap_or(usize::MAX);

        let Some(gain) = requested else {
            let (value, label) = labels
                .get(current_index)
                .map(|l| {
                    (
                        crate::hardware::dslr_controls::parse_iso(l).unwrap_or(0.0),
                        Some(l.clone()),
                    )
                })
                .unwrap_or((0.0, None));
            return Ok(Applied { value, label });
        };

        match nearest_iso(&labels, gain as f64) {
            Some(choice) => {
                tracing::info!(
                    "ISO: requested {} -> choice {} \"{}\" ({})",
                    gain,
                    choice.index,
                    choice.label,
                    choice.value
                );
                camera
                    .set_parameter(OlsCamOptionId::Iso, choice.index as f64)
                    .map_err(|e| {
                        tracing::error!("Failed to set ISO to {}: {}", choice.label, e);
                        StatusCode::INTERNAL_SERVER_ERROR
                    })?;
                return Ok(Applied {
                    value: choice.value,
                    label: Some(choice.label),
                });
            }
            None => {
                warnings.push("Camera exposed no numeric ISO setting; gain ignored".to_string());
                return Ok(Applied {
                    value: 0.0,
                    label: None,
                });
            }
        }
    }

    if requested.is_some() {
        warnings.push("Camera exposes no gain or ISO control; gain ignored".to_string());
    }
    Ok(Applied {
        value: 0.0,
        label: None,
    })
}

/// Stop the preview/acquisition stream.
///
/// This is intentionally not advertised as exposure abort: the current bridge
/// cannot interrupt a blocking DSLR exposure. A future `abort_exposure` route
/// will only be exposed when a driver reports that capability truthfully.
pub(crate) async fn stop_stream_control(state: &AppState) -> Result<serde_json::Value, ApiError> {
    let _acquisition = state.acquisition.try_lock().map_err(|_| {
        let activity = state.camera_runtime.read().unwrap().activity;
        if activity == CameraActivity::Exposing {
            ApiError::new(
                StatusCode::CONFLICT,
                "exposure_abort_unsupported",
                "This camera backend cannot interrupt the active exposure; wait for it to finish",
            )
        } else {
            ApiError::new(
                StatusCode::CONFLICT,
                "camera_busy",
                "Camera control is busy with another operation",
            )
        }
    })?;
    let was_streaming = *state.streaming.lock().unwrap();

    let stop_error = {
        let mut cam_lock = state.camera.lock().unwrap();
        let camera = cam_lock.as_mut().ok_or_else(ApiError::no_camera)?;
        if was_streaming {
            camera.stop_stream().err().map(|e| e.to_string())
        } else {
            None
        }
    };

    *state.streaming.lock().unwrap() = false;
    state.clear_camera_observations();
    {
        let mut runtime = state.camera_runtime.write().unwrap();
        runtime.activity = if stop_error.is_some() {
            CameraActivity::Error
        } else {
            CameraActivity::Idle
        };
        runtime.streaming = false;
        runtime.active_format_index = None;
        runtime.operation_id = None;
        runtime.exposure_started_at = None;
        runtime.exposure_duration_s = None;
        runtime.last_error = stop_error.clone();
    }

    if let Some(error) = stop_error {
        return Err(ApiError::hardware(format!(
            "Camera stream stopped with a driver error: {error}"
        )));
    }

    Ok(serde_json::json!({
        "status": "stopped",
        "was_streaming": was_streaming,
        "exposure_aborted": false,
    }))
}

pub async fn abort(State(state): State<AppState>) -> Result<Json<serde_json::Value>, ApiError> {
    tracing::info!("Camera stream stop requested");
    match stop_stream_control(&state).await {
        Ok(value) => Ok(Json(value)),
        // The original `/camera/abort` route was idempotent. Keep that
        // compatibility while `/camera/stream/stop` remains the strict API.
        Err(error) if error.code == "camera_not_connected" => Ok(Json(serde_json::json!({
            "status": "stopped",
            "was_streaming": false,
            "exposure_aborted": false,
        }))),
        Err(error) => Err(error),
    }
}

pub async fn stop_stream(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    stop_stream_control(&state).await.map(Json)
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
    /// Driver-independent human label suitable for a generated control panel.
    pub label: String,
    /// acquisition | cooling | image | advanced
    pub group: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<&'static str>,
    /// Value semantics: bool, number, msec, percent, kelvin, celsius, selection.
    pub kind: String,
    pub read_only: bool,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub default: f64,
    pub current: f64,
    /// For `kind == "selection"`, the camera's own labels in index order, so
    /// `choices[current as usize]` is what the body is set to. Empty for
    /// continuous controls.
    ///
    /// Without these the UI can only render a bare integer slider for `iso`
    /// and `shutter` — the operator would be choosing "17" with no way to know
    /// it means 1/125 s. Manual control is only real if the labels are visible.
    pub choices: Vec<String>,
    /// Parsed physical value of the current choice: seconds for `shutter`,
    /// ISO for `iso`. `None` for non-numeric labels ("bulb", "Auto").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_value: Option<f64>,
}

fn parameter_group(id: OlsCamOptionId) -> &'static str {
    use OlsCamOptionId::*;
    match id {
        AutoExp | Exp | Gain | Iso | Shutter | BlackLevel | AverageBin => "acquisition",
        Temperature | CoolerTarget | CoolerOn | FanOn | CoolerPowerPerc => "cooling",
        AutoWb | Wb | WbR | WbB | Gamma | Brightness | Contrast | LiveStretch => "image",
        _ => "advanced",
    }
}

fn parameter_unit(id: OlsCamOptionId, option_type: OlsCamOptionType) -> Option<&'static str> {
    match option_type {
        OlsCamOptionType::Msec => return Some("ms"),
        OlsCamOptionType::Percent => return Some("percent"),
        OlsCamOptionType::Kelvin => return Some("K"),
        OlsCamOptionType::Celsius => return Some("degC"),
        _ => {}
    }

    use OlsCamOptionId::*;
    match id {
        Exp | CaptureDelay => Some("ms"),
        Temperature | CoolerTarget => Some("degC"),
        CoolerPowerPerc => Some("percent"),
        Gain | BlackLevel | Brightness | Contrast | Gamma | Wb | WbR | WbB => Some("driver_units"),
        Iso => Some("ISO"),
        _ => None,
    }
}

pub(crate) fn parameter_info(camera: &OlsCamera, id: OlsCamOptionId, p: CamParam) -> CamParamInfo {
    let choices = if p.option_type == OlsCamOptionType::Selection {
        camera.parameter_choices(id).unwrap_or_else(|e| {
            tracing::warn!("Could not read choices for {:?}: {}", id, e);
            Vec::new()
        })
    } else {
        Vec::new()
    };
    let current_label = choices.get(p.cur_val as usize);
    let current_value = current_label.and_then(|label| match id {
        OlsCamOptionId::Shutter => crate::hardware::dslr_controls::parse_shutter_seconds(label),
        OlsCamOptionId::Iso => crate::hardware::dslr_controls::parse_iso(label),
        _ => None,
    });

    CamParamInfo {
        name: option_name(id).to_string(),
        label: option_label(id).to_string(),
        group: parameter_group(id),
        unit: parameter_unit(id, p.option_type),
        kind: option_type_name(p.option_type).to_string(),
        read_only: p.read_only,
        min: p.min_val,
        max: p.max_val,
        step: p.step_size,
        default: p.def_val,
        current: p.cur_val,
        choices,
        current_value,
    }
}

fn read_parameters(camera: &OlsCamera) -> Result<Vec<CamParamInfo>, ApiError> {
    let options = camera
        .supported_options()
        .map_err(|e| ApiError::hardware(format!("Could not list camera controls: {e}")))?;
    options
        .into_iter()
        .map(|id| {
            camera
                .get_parameter(id)
                .map(|parameter| parameter_info(camera, id, parameter))
                .map_err(|error| {
                    ApiError::hardware(format!(
                        "Could not read camera control '{}': {error}",
                        option_name(id)
                    ))
                })
        })
        .collect()
}

fn update_runtime_telemetry(state: &AppState, controls: &[CamParamInfo]) {
    let mut runtime = state.camera_runtime.write().unwrap();
    runtime.temperature_c = controls
        .iter()
        .find(|control| control.name == "temperature")
        .map(|control| control.current);
    runtime.cooler_on = controls
        .iter()
        .find(|control| control.name == "cooler_on")
        .map(|control| control.current > 0.5);
}

/// GET /camera/parameters — introspect every supported option with its
/// min/max/step/current so the UI can render adaptive manual controls.
pub async fn get_parameters(
    State(state): State<AppState>,
) -> Result<Json<Vec<CamParamInfo>>, ApiError> {
    let _acquisition = state.acquisition.try_lock().map_err(|_| {
        ApiError::new(
            StatusCode::CONFLICT,
            "camera_busy",
            "Camera controls cannot be refreshed during another camera operation",
        )
    })?;
    let cam_lock = state.camera.lock().unwrap();
    let camera = cam_lock.as_ref().ok_or_else(ApiError::no_camera)?;
    let controls = read_parameters(camera)?;
    drop(cam_lock);
    update_runtime_telemetry(&state, &controls);
    Ok(Json(controls))
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
) -> Result<Json<serde_json::Value>, ApiError> {
    let _acquisition = state.acquisition.lock().await;
    let id = parse_option(&req.name).ok_or_else(|| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "unknown_camera_control",
            format!("Unknown camera control '{}'", req.name),
        )
    })?;

    let cam_lock = state.camera.lock().unwrap();
    let camera = cam_lock.as_ref().ok_or_else(ApiError::no_camera)?;

    let parameter = camera.get_parameter(id).map_err(|_| {
        ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "unsupported_camera_control",
            format!("The connected camera does not expose '{}'", req.name),
        )
    })?;
    if parameter.read_only {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "camera_control_read_only",
            format!("Camera control '{}' is read-only", req.name),
        ));
    }
    if !req.value.is_finite() || req.value < parameter.min_val || req.value > parameter.max_val {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "camera_control_out_of_range",
            format!(
                "Value {} is outside the supported range {}..={} for '{}'",
                req.value, parameter.min_val, parameter.max_val, req.name
            ),
        ));
    }
    if parameter.option_type == OlsCamOptionType::Bool && !matches!(req.value, 0.0 | 1.0) {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_camera_control_value",
            format!("Boolean camera control '{}' accepts only 0 or 1", req.name),
        ));
    }
    if parameter.option_type == OlsCamOptionType::Selection && req.value.fract() != 0.0 {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_camera_control_value",
            format!(
                "Selection camera control '{}' requires an integer choice index",
                req.name
            ),
        ));
    }

    camera
        .set_parameter(id, req.value)
        .map_err(|e| ApiError::hardware(format!("Could not set '{}': {e}", req.name)))?;

    let current = camera
        .get_parameter(id)
        .map(|p| p.cur_val)
        .map_err(|e| ApiError::hardware(format!("Could not read back '{}': {e}", req.name)))?;

    if id == OlsCamOptionId::Temperature {
        state.camera_runtime.write().unwrap().temperature_c = Some(current);
    } else if id == OlsCamOptionId::CoolerOn {
        state.camera_runtime.write().unwrap().cooler_on = Some(current > 0.5);
    }

    Ok(Json(serde_json::json!({
        "status": "ok",
        "name": req.name,
        "requested_value": req.value,
        "value": current,
        "applied_value": current,
    })))
}

#[derive(Debug, Serialize)]
pub struct FormatInfo {
    pub index: usize,
    /// Stable for this camera connection; clients should send this index back
    /// unchanged rather than rebuilding a format from width/height alone.
    pub id: String,
    pub format: String,
    pub width: i32,
    pub height: i32,
    pub bin: i32,
    pub roi_num: i32,
    pub roi_den: i32,
    pub framerate: f32,
    pub previewable: bool,
    /// Whether the daemon can persist this format as a capture. This does not
    /// claim the encoding is linear, lossless, or suitable for calibration.
    pub capture_supported: bool,
    /// Acquisition mode: `"liveview"` costs no shutter actuations and is the
    /// right choice for framing and focusing; `"still"` fires the shutter once
    /// per frame and is the right choice for science frames. On a DSLR the
    /// driver lists liveview first, so index 0 is always the free one.
    pub mode: &'static str,
}

fn stream_type_previewable(format: OlsStreamType) -> bool {
    matches!(
        format,
        OlsStreamType::Mjpeg | OlsStreamType::Rgb24 | OlsStreamType::Mono8 | OlsStreamType::Raw8
    )
}

fn stream_type_capture_supported(format: OlsStreamType) -> bool {
    matches!(
        format,
        OlsStreamType::Mjpeg
            | OlsStreamType::Rgb24
            | OlsStreamType::Mono8
            | OlsStreamType::Raw8
            | OlsStreamType::Mono16
            | OlsStreamType::Raw16
    )
}

pub(crate) fn format_info(index: usize, f: &StreamFormat) -> FormatInfo {
    FormatInfo {
        index,
        id: format!("format-{index}"),
        format: format!("{:?}", f.format),
        width: f.width,
        height: f.height,
        bin: f.bin,
        roi_num: f.roi_num,
        roi_den: f.roi_den,
        framerate: f.framerate,
        previewable: stream_type_previewable(f.format),
        capture_supported: stream_type_capture_supported(f.format),
        mode: if f.is_liveview() { "liveview" } else { "still" },
    }
}

/// GET /camera/formats — list selectable stream formats / binning modes.
pub async fn get_formats(State(state): State<AppState>) -> Result<Json<Vec<FormatInfo>>, ApiError> {
    let _acquisition = state.acquisition.try_lock().map_err(|_| {
        ApiError::new(
            StatusCode::CONFLICT,
            "camera_busy",
            "Camera formats cannot be refreshed during another camera operation",
        )
    })?;
    let cam_lock = state.camera.lock().unwrap();
    let camera = cam_lock.as_ref().ok_or_else(ApiError::no_camera)?;

    let formats = camera.formats().map_err(|e| {
        tracing::error!("Failed to get formats: {}", e);
        ApiError::hardware(format!("Could not read camera formats: {e}"))
    })?;

    let list = formats
        .iter()
        .enumerate()
        .map(|(i, f)| format_info(i, f))
        .collect();
    Ok(Json(list))
}

#[derive(Debug, Serialize)]
pub struct CameraCapabilityInfo {
    pub supports_stream: bool,
    pub supports_dedicated_still: bool,
    pub supports_exposure_abort: bool,
    pub supports_cooling: bool,
    pub supports_temperature: bool,
    pub supports_hardware_binning: bool,
    pub supports_roi: bool,
}

#[derive(Debug, Serialize)]
pub struct CameraFrameInfo {
    pub stream_generation: u64,
    pub counter: i32,
    pub timestamp: f64,
    pub width: i32,
    pub height: i32,
    pub format: String,
}

#[derive(Debug, Serialize)]
pub struct CameraDescriptor {
    pub connected: bool,
    /// Unique for this connection. Reconnect produces a new id.
    pub connection_id: Option<String>,
    pub name: Option<String>,
    pub driver_id: Option<String>,
    /// Driver-local, scan-scoped selector. It is not a permanent hardware id.
    pub device_index: Option<i32>,
    /// Null until the native bridge exposes a vendor serial/VID/PID identity.
    pub hardware_id: Option<String>,
    pub activity: CameraActivity,
    pub streaming: bool,
    pub active_format_index: Option<usize>,
    pub operation_id: Option<String>,
    pub exposure_progress: Option<f64>,
    pub temperature_c: Option<f64>,
    pub cooler_on: Option<bool>,
    pub last_error: Option<String>,
    pub capabilities: CameraCapabilityInfo,
    pub formats: Vec<FormatInfo>,
    pub controls: Vec<CamParamInfo>,
    pub latest_frame: Option<CameraFrameInfo>,
}

/// GET /camera — one frontend-ready description of the connected camera.
///
/// A generated Equipment screen should not need to race separate status,
/// parameter and format requests just to decide which controls to render.
pub async fn get_camera_descriptor(
    State(state): State<AppState>,
) -> Result<Json<CameraDescriptor>, ApiError> {
    // Reading every control calls into the vendor SDK. Do not mix camera A's
    // identity with camera B's controls during reconnect, or compete with a
    // long exposure. `/status` remains the non-blocking endpoint to poll while
    // this returns `camera_busy`.
    let _acquisition = state.acquisition.try_lock().map_err(|_| {
        ApiError::new(
            StatusCode::CONFLICT,
            "camera_busy",
            "Camera descriptor cannot be refreshed during another camera operation",
        )
    })?;
    let connection_before = state.camera_runtime.read().unwrap().connection_id.clone();

    let (formats, controls, supports_still) = {
        let cam_lock = state.camera.lock().unwrap();
        match cam_lock.as_ref() {
            Some(camera) => {
                let formats = camera.formats().map_err(|e| {
                    ApiError::hardware(format!("Could not read camera formats: {e}"))
                })?;
                let controls = read_parameters(camera)?;
                (formats, controls, camera.supports_still_capture())
            }
            None => (Vec::new(), Vec::new(), false),
        }
    };
    update_runtime_telemetry(&state, &controls);

    let runtime = state.camera_runtime.read().unwrap().clone();
    if runtime.connection_id != connection_before {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "camera_changed_during_read",
            "Camera changed while its descriptor was being read; refresh it again",
        ));
    }
    let latest = state.latest_frame.lock().unwrap().clone();

    let control_present = |name: &str| controls.iter().any(|control| control.name == name);
    let supports_roi = formats
        .iter()
        .any(|format| format.roi_num != 1 || format.roi_den != 1);
    let supports_hardware_binning = formats.iter().any(|format| format.bin > 1);

    Ok(Json(CameraDescriptor {
        connected: runtime.connection_id.is_some(),
        connection_id: runtime.connection_id.clone(),
        name: runtime.camera_name.clone(),
        driver_id: runtime.driver_id.clone(),
        device_index: runtime.device_index,
        hardware_id: None,
        activity: runtime.activity,
        streaming: runtime.streaming,
        active_format_index: runtime.active_format_index,
        operation_id: runtime.operation_id.clone(),
        exposure_progress: runtime.exposure_progress(chrono::Utc::now()),
        temperature_c: runtime.temperature_c,
        cooler_on: runtime.cooler_on,
        last_error: runtime.last_error.clone(),
        capabilities: CameraCapabilityInfo {
            supports_stream: !formats.is_empty(),
            supports_dedicated_still: supports_still,
            supports_exposure_abort: false,
            supports_cooling: control_present("cooler_on") || control_present("cooler_target"),
            supports_temperature: control_present("temperature"),
            supports_hardware_binning,
            supports_roi,
        },
        formats: formats
            .iter()
            .enumerate()
            .map(|(index, format)| format_info(index, format))
            .collect(),
        controls,
        latest_frame: latest.map(|frame| CameraFrameInfo {
            stream_generation: frame.stream_generation,
            counter: frame.frame_counter,
            timestamp: frame.unix_timestamp,
            width: frame.width,
            height: frame.height,
            format: format!("{:?}", frame.format),
        }),
    }))
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

pub(crate) fn option_label(id: OlsCamOptionId) -> &'static str {
    use OlsCamOptionId::*;
    match id {
        AutoExp => "Auto exposure",
        AutoWb => "Auto white balance",
        AutoFocus => "Camera autofocus",
        Exp => "Exposure",
        Wb => "White balance",
        WbR => "White balance red",
        WbB => "White balance blue",
        Focus => "Camera focus",
        Gain => "Gain",
        Gamma => "Gamma",
        Brightness => "Brightness",
        Contrast => "Contrast",
        Temperature => "Sensor temperature",
        CoolerTarget => "Cooler target",
        CoolerOn => "Cooler",
        FanOn => "Fan",
        CoolerPowerPerc => "Cooler power",
        AverageBin => "Average binning",
        BlackLevel => "Black level / offset",
        ConvGainHcg => "High conversion gain",
        ConvGainHdr => "Conversion gain mode",
        LowNoise => "Low noise mode",
        HighFullwell => "High full-well mode",
        Iso => "ISO",
        Shutter => "Shutter speed",
        Viewfinder => "Live view",
        CaptureTarget => "Capture target",
        KeepImages => "Keep images on camera",
        CaptureDelay => "Inter-shot delay",
        Zoom => "Live-view zoom",
        LiveStretch => "Automatic preview stretch",
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

#[cfg(test)]
mod tests {
    use super::*;

    fn stream_format(
        format: OlsStreamType,
        width: i32,
        height: i32,
        bin: i32,
        framerate: f32,
    ) -> StreamFormat {
        StreamFormat {
            format,
            width,
            height,
            bin,
            roi_num: 1,
            roi_den: 1,
            framerate,
        }
    }

    #[test]
    fn phase1_camera_select_format_defaults_to_previewable_and_accepts_exact_index() {
        let formats = [
            stream_format(OlsStreamType::Yuv2, 1920, 1080, 1, 5.0),
            stream_format(OlsStreamType::Mjpeg, 1920, 1080, 1, 30.0),
            stream_format(OlsStreamType::Mono16, 1280, 960, 2, 0.0),
        ];

        let (default_index, default_format) = select_format(&formats, None).unwrap();
        assert_eq!(default_index, 1);
        assert_eq!(default_format.format, OlsStreamType::Mjpeg);
        assert_eq!((default_format.width, default_format.height), (1920, 1080));

        let (selected_index, selected_format) = select_format(&formats, Some(2)).unwrap();
        assert_eq!(selected_index, 2);
        assert_eq!(selected_format.format, OlsStreamType::Mono16);
        assert_eq!(selected_format.bin, 2);
    }

    #[test]
    fn phase1_camera_select_format_rejects_empty_and_out_of_range_requests() {
        let empty = select_format(&[], None).unwrap_err();
        assert_eq!(empty.status, StatusCode::BAD_GATEWAY);
        assert_eq!(empty.code, "camera_has_no_formats");

        let formats = [stream_format(OlsStreamType::Mono8, 640, 480, 1, 15.0)];
        for requested in [1, usize::MAX] {
            let error = select_format(&formats, Some(requested)).unwrap_err();
            assert_eq!(error.status, StatusCode::BAD_REQUEST);
            assert_eq!(error.code, "invalid_camera_format");
            assert!(error.message.contains(&format!("Format index {requested}")));
            assert!(error.message.contains("1 format(s)"));
        }
    }

    #[test]
    fn phase1_camera_fresh_frame_gate_discards_first_post_setting_frame() {
        let mut gate = FreshFrameGate::new(Some(41));

        assert!(!gate.accepts(41), "the baseline frame is stale");
        assert!(
            !gate.accepts(42),
            "the first callback may already be in flight"
        );
        assert!(
            gate.accepts(43),
            "the second completed frame is attributable"
        );
        assert!(
            gate.accepts(44),
            "later monotonic frames remain attributable"
        );
    }

    #[test]
    fn phase1_camera_fresh_frame_gate_learns_a_baseline_when_mailbox_is_empty() {
        let mut gate = FreshFrameGate::new(None);

        assert!(
            !gate.accepts(7),
            "the first observed counter establishes a baseline"
        );
        assert!(
            !gate.accepts(7),
            "a duplicate callback is not a second frame"
        );
        assert!(
            gate.accepts(8),
            "one later counter proves a second completed frame"
        );
    }

    #[test]
    fn phase1_camera_fresh_frame_gate_rejects_counter_reset_and_does_not_overflow() {
        let mut reset = FreshFrameGate::new(Some(41));
        assert!(!reset.accepts(0));
        assert!(!reset.accepts(1));

        let mut upper_edge = FreshFrameGate::new(Some(i32::MAX - 1));
        assert!(!upper_edge.accepts(i32::MAX));
        assert!(
            !upper_edge.accepts(i32::MIN),
            "a wrapped counter must not look newer than the old stream"
        );

        let mut exactly_two = FreshFrameGate::new(Some(i32::MAX - 2));
        assert!(exactly_two.accepts(i32::MAX));
    }

    #[test]
    fn phase1_camera_format_info_reports_preview_capture_and_mode_truthfully() {
        struct Case {
            format: OlsStreamType,
            framerate: f32,
            previewable: bool,
            capture_supported: bool,
            mode: &'static str,
        }

        let cases = [
            Case {
                format: OlsStreamType::Mjpeg,
                framerate: 30.0,
                previewable: true,
                capture_supported: true,
                mode: "liveview",
            },
            Case {
                format: OlsStreamType::Mono8,
                framerate: -1.0,
                previewable: true,
                capture_supported: true,
                mode: "liveview",
            },
            Case {
                format: OlsStreamType::Mono16,
                framerate: 0.0,
                previewable: false,
                capture_supported: true,
                mode: "still",
            },
            Case {
                format: OlsStreamType::Yuv2,
                framerate: 30.0,
                previewable: false,
                capture_supported: false,
                mode: "liveview",
            },
            Case {
                format: OlsStreamType::Rgb48,
                framerate: 0.0,
                previewable: false,
                capture_supported: false,
                mode: "still",
            },
        ];

        for (index, case) in cases.iter().enumerate() {
            let mut format = stream_format(case.format, 1024, 768, 2, case.framerate);
            format.roi_num = 3;
            format.roi_den = 4;
            let info = format_info(index, &format);

            assert_eq!(info.index, index);
            assert_eq!(info.id, format!("format-{index}"));
            assert_eq!(info.width, 1024);
            assert_eq!(info.height, 768);
            assert_eq!(info.bin, 2);
            assert_eq!(info.roi_num, 3);
            assert_eq!(info.roi_den, 4);
            assert_eq!(info.previewable, case.previewable, "{:?}", case.format);
            assert_eq!(
                info.capture_supported, case.capture_supported,
                "{:?}",
                case.format
            );
            assert_eq!(info.mode, case.mode, "{:?}", case.format);
        }
    }
}

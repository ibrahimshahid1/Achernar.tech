use axum::{extract::Query, extract::State, http::StatusCode, Json};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use crate::fits::FitsMeta;
use crate::hardware::camera::FrameData;
use crate::hardware::ols_ffi::{OlsCamOptionId, OlsCamOptionType, OlsStreamType};
use crate::state::AppState;

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

    camera
        .start_stream(&fmt, frame_store, state.camera_config.saturation_adu)
        .map_err(|e| {
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
        // Make the cost of the chosen mode explicit in the response: on a DSLR
        // a "still" stream is one shutter actuation per frame.
        "mode": if fmt.is_liveview() { "liveview" } else { "still" },
    })))
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
) -> Result<Json<ImageData>, StatusCode> {
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
) -> Result<ImageData, StatusCode> {
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
        "binning": params.binning,
        "filter": params.filter,
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
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    match run_capture(&state, &capture_id, &frame_id, &capture_type, &params).await {
        Ok(image) => Ok(image),
        Err(err) => {
            tracing::error!("Capture {capture_id} failed: {}", err.detail);
            state
                .capture_store
                .fail_attempt(state.local_store.pool(), &capture_id, &err.detail)
                .await;
            Err(err.status)
        }
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
    .map_err(|e| CaptureError::new(StatusCode::INTERNAL_SERVER_ERROR, format!("capture task panicked: {e}")))?
    .map_err(|e| CaptureError::new(StatusCode::BAD_GATEWAY, e.to_string()))?;

    let (capture, handle) = native;

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
                            info.make,
                            info.model,
                            info.cblack_pattern[0],
                            info.cblack_pattern[1]
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
                warnings.push(format!("FITS derivation failed ({e}); native file preserved"));
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
        driver_name: Some("gphoto2".to_string()),
        capture_type: capture_type.to_string(),
        exposure_started_at: capture.started_at,
        exposure_completed_at: capture.completed_at,
        requested_exposure_s: params.exposure,
        requested_gain: params.gain.map(|g| g as f64),
        applied_exposure_s: capture.applied_exposure_s,
        iso: (capture.applied_sensitivity > 0.0).then_some(capture.applied_sensitivity),
        shutter_label: (!capture.shutter_label.is_empty()).then(|| capture.shutter_label.clone()),
        offset_val: params.offset.map(|o| o as i64),
        binning: params.binning.unwrap_or(1) as i64,
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
/// Still avoids the old defect — it applies the settings, then waits for a
/// frame whose counter is strictly newer than the one present *before* the
/// settings were written, and fails rather than substituting a stale frame.
/// A capture that cannot prove which exposure it returned is not a capture.
async fn capture_from_stream(
    state: &AppState,
    capture_id: &str,
    frame_id: &str,
    capture_type: &str,
    params: &CaptureParams,
    mut warnings: Vec<String>,
) -> Result<ImageData, CaptureError> {
    // Baseline BEFORE the settings are applied, so any frame we accept was
    // started after the new settings were in place.
    let baseline = state
        .latest_frame
        .lock()
        .unwrap()
        .as_ref()
        .map(|f| f.frame_counter);

    let (applied_exposure, applied_gain, camera_name) = {
        let cam_lock = state.camera.lock().unwrap();
        let camera = cam_lock.as_ref().ok_or_else(|| {
            CaptureError::new(StatusCode::PRECONDITION_FAILED, "No camera connected")
        })?;
        let options = camera.supported_options().map_err(|e| {
            CaptureError::new(StatusCode::INTERNAL_SERVER_ERROR, format!("list options: {e}"))
        })?;
        let exposure = apply_exposure(camera, &options, params.exposure, &mut warnings)
            .map_err(|s| CaptureError::new(s, "Failed to apply the requested exposure"))?;
        let gain = apply_gain(camera, &options, params.gain, &mut warnings)
            .map_err(|s| CaptureError::new(s, "Failed to apply the requested gain"))?;
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
            camera.name().unwrap_or_else(|_| "Unknown".to_string()),
        )
    };

    if !*state.streaming.lock().unwrap() {
        return Err(CaptureError::new(
            StatusCode::PRECONDITION_FAILED,
            "This camera captures through its stream; start the stream first",
        ));
    }

    let actual_exposure_s = applied_exposure.value;
    let started_at = unix_now();

    // Two exposures plus slack: the in-flight exposure must finish and be
    // discarded before a frame taken under the new settings can arrive.
    let timeout =
        Duration::from_secs_f64((actual_exposure_s * 3.0 + 20.0).clamp(20.0, 900.0));
    let deadline = Instant::now() + timeout;
    let frame = loop {
        {
            let guard = state.latest_frame.lock().unwrap();
            if let Some(f) = guard.as_ref() {
                let fresh = match baseline {
                    // Strictly newer, and not merely different: a counter that
                    // wrapped or reset must not read as a new exposure.
                    Some(b) => f.frame_counter > b,
                    None => true,
                };
                if fresh {
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

    let (temperature, temperature_state, temperature_source) = read_temperature(state);

    let meta = crate::fits::FitsMeta {
        exposure_s: actual_exposure_s,
        gain: applied_gain.value.max(0.0) as u32,
        temperature_c: temperature.unwrap_or(f64::NAN),
        binning: params.binning.unwrap_or(1),
        instrument: camera_name.clone(),
        date_obs: unix_to_rfc3339(started_at),
        iso: applied_gain.label.as_ref().map(|_| applied_gain.value),
        shutter_label: applied_exposure.label.clone(),
        source_file: None,
        // Stream frames come through the driver's own pipeline, which does not
        // report a black or saturation level. Asserting one here would be a
        // guess written into a file that reads like a measurement.
        calibration: None,
    };
    let fits_bytes = crate::fits::encode_capture(&frame, &meta).map_err(|e| {
        CaptureError::new(StatusCode::INTERNAL_SERVER_ERROR, format!("encode FITS: {e}"))
    })?;

    // For a sensor camera the FITS is the canonical record — there is no
    // camera-authored file behind it — so it is filed as a native asset.
    let assets = vec![crate::storage::capture_store::PendingAsset::new(
        crate::storage::capture_store::AssetRole::NativeRaw,
        "image.fits",
        "application/fits",
        fits_bytes,
    )
    .with_dimensions(frame.width as i64, frame.height as i64, 16)
    .with_provenance(serde_json::json!({
        "producer": concat!("achernar-daemon ", env!("CARGO_PKG_VERSION")),
        "stream_format": format!("{:?}", frame.format),
        "note": "sensor camera: no separate still path, frame taken from the stream",
    }))];

    let pending = crate::storage::capture_store::PendingCapture {
        capture_id: capture_id.to_string(),
        frame_id: frame_id.to_string(),
        session_id: params.session_id.clone(),
        rig_id: state.rig_id.clone(),
        camera_name: Some(camera_name),
        driver_name: None,
        capture_type: capture_type.to_string(),
        exposure_started_at: started_at,
        exposure_completed_at: completed_at,
        requested_exposure_s: params.exposure,
        requested_gain: params.gain.map(|g| g as f64),
        applied_exposure_s: actual_exposure_s,
        iso: applied_gain.label.as_ref().map(|_| applied_gain.value),
        shutter_label: applied_exposure.label.clone(),
        offset_val: params.offset.map(|o| o as i64),
        binning: params.binning.unwrap_or(1) as i64,
        filter_name: params.filter.clone(),
        focuser_position: read_focuser_position(state),
        temperature,
        temperature_source,
        temperature_state,
        acquisition: serde_json::json!({
            "frame_counter": frame.frame_counter,
            "baseline_frame_counter": baseline,
            "stream_format": format!("{:?}", frame.format),
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
            timestamp: unix_to_rfc3339(started_at),
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
            .unwrap_or(requested_s * 1000.0);
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
            .unwrap_or_else(|_| requested.unwrap_or(0) as f64);
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
    /// For `kind == "selection"`, the camera's own labels in index order, so
    /// `choices[current as usize]` is what the body is set to. Empty for
    /// continuous controls.
    ///
    /// Without these the UI can only render a bare integer slider for `iso`
    /// and `shutter` — the operator would be choosing "17" with no way to know
    /// it means 1/125 s. Manual control is only real if the labels are visible.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub choices: Vec<String>,
    /// Parsed physical value of the current choice: seconds for `shutter`,
    /// ISO for `iso`. `None` for non-numeric labels ("bulb", "Auto").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_value: Option<f64>,
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
            // Selection options are meaningless as bare indices; fetch the
            // labels so the UI can show "1/125" instead of "17".
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
                OlsCamOptionId::Shutter => {
                    crate::hardware::dslr_controls::parse_shutter_seconds(label)
                }
                OlsCamOptionId::Iso => crate::hardware::dslr_controls::parse_iso(label),
                _ => None,
            });

            params.push(CamParamInfo {
                name: option_name(id).to_string(),
                kind: option_type_name(p.option_type).to_string(),
                read_only: p.read_only,
                min: p.min_val,
                max: p.max_val,
                step: p.step_size,
                default: p.def_val,
                current: p.cur_val,
                choices,
                current_value,
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
    /// Acquisition mode: `"liveview"` costs no shutter actuations and is the
    /// right choice for framing and focusing; `"still"` fires the shutter once
    /// per frame and is the right choice for science frames. On a DSLR the
    /// driver lists liveview first, so index 0 is always the free one.
    pub mode: &'static str,
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
            mode: if f.is_liveview() { "liveview" } else { "still" },
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

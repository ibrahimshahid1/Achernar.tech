//! Plate-solve routes — Section 8.3.
//!
//! `POST /platesolve/solve` solves one archived exposure and records the
//! attempt against it; `GET /platesolve/status` says whether a solver is
//! installed, why not, and what the automatic trigger is waiting for. Solving
//! reads the archive and never commands hardware, so it takes no acquisition
//! lock and cannot disturb an exposure. It does use every core while it runs,
//! which is why only one runs at a time.
//!
//! A solve that finds no position is a 200 with `success: false` and a
//! failure reason — clouds are an observation, not a server error. Non-200
//! responses mean the solver could not be asked at all.
//!
//! **Pointing samples (H12, M-4, M-3).** A successful solve of a frame whose
//! exposure has a device window becomes a `pointing_samples` row: the
//! mount's native RA/Dec at mid-exposure (interpolated from the hub's ring
//! buffer, never a read after the fact), its frame and JNow flavour,
//! DoesRefraction, the same position in J2000, the solved J2000 truth, the
//! residual (mount minus truth, RA scaled by cos δ), pier side, hour angle,
//! LST, Alt/Az, sync generation and sync mode. The residual also fills
//! `plate_solve_events.pointing_error_arcmin`, and the pointing model of
//! that mount, generation and pier side is refitted. No sample is written
//! when the window has no mount data, the mount slewed or was synced during
//! the exposure, or its native frame is unknown (M-a); the response says
//! why.

use axum::{extract::State, http::StatusCode, Json};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::api_error::ApiError;
use crate::config::RigOptics;
use crate::equipment::events::{PointingPurpose, PointingSampleRecord};
use crate::equipment::handle::MountHandle;
use crate::equipment::snapshot::{DeviceWindow, MountSnapshot};
use crate::equipment::traits::{EquipmentReader, FrameConverter};
use crate::equipment::types::{
    EqCoord, EqFrame, JNowFlavour, MountSyncMode, PierSide, Site, TemperatureRole,
};
use crate::mount::plate_solve_trigger::TriggerStatus;
use crate::processing::plate_solver::{
    PlateSolution, PointingHint, SolveError, SolveFailure, SolveRequest, SolverStatus,
};
use crate::state::AppState;
use crate::storage::plate_solve_repo::{self, LatestSolve, PlateSolveRecord};

/// Arcseconds per radian divided by 1000: image scale in arcsec per pixel is
/// this times pixel size in microns over focal length in millimetres.
const ARCSEC_PER_UM_PER_MM: f64 = 206.264_806;

#[derive(Debug, Deserialize)]
pub struct SolveFrameRequest {
    pub frame_id: String,
    /// Where to start looking. Omit for a blind, all-sky solve.
    #[serde(default)]
    pub hint: Option<HintRequest>,
    /// Expected arcsec per binned pixel. Overrides the rig optics.
    #[serde(default)]
    pub pixel_scale_arcsec: Option<f64>,
    /// Let the solver search for the scale even when the rig optics are known
    /// — after fitting a reducer the `rigs` row does not know about, say.
    #[serde(default)]
    pub ignore_rig_optics: bool,
}

#[derive(Debug, Deserialize)]
pub struct HintRequest {
    pub ra_hours: f64,
    pub dec_deg: f64,
    /// Defaults to `[plate_solve] search_radius_deg`.
    #[serde(default)]
    pub search_radius_deg: Option<f64>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ScaleHint {
    pub arcsec_per_px: f64,
    /// `request` | `rig_optics`
    pub source: &'static str,
}

#[derive(Debug, Serialize)]
pub struct SolvedAsset {
    pub id: String,
    pub role: String,
    pub relative_path: String,
    pub sha256: String,
}

#[derive(Debug, Serialize)]
pub struct SolveFrameResponse {
    pub frame_id: String,
    /// The `plate_solve_events` row. `None` only if recording failed — see
    /// `record_error`; the solution itself is still valid.
    pub event_id: Option<String>,
    pub success: bool,
    pub solution: Option<PlateSolution>,
    pub failure: Option<SolveFailure>,
    /// `manual` | `startup` | `post_slew`
    pub trigger: String,
    /// `hinted` | `blind`
    pub mode: &'static str,
    pub scale_hint: Option<ScaleHint>,
    /// Field height handed to the solver; `None` when it searched for one.
    pub fov_height_deg: Option<f64>,
    pub solve_time_ms: u64,
    pub solver_command: String,
    pub solved_asset: SolvedAsset,
    /// Shape of the solved file as its own header states it.
    pub image: crate::fits::FitsImageGeometry,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub record_error: Option<String>,
    /// The H12 pointing sample of a successful solve, or why none was
    /// written. `None` when the solve failed.
    pub pointing: Option<PointingOutcome>,
}

#[derive(Debug, Serialize)]
pub struct PlateSolveStatus {
    #[serde(flatten)]
    pub solver: SolverStatus,
    pub auto_solve: TriggerStatus,
    /// The newest successful solve, from `plate_solve_events` (v6 §8.1: the
    /// plate-solve state carries its own timestamp).
    pub last_solve: Option<LatestSolve>,
}

/// GET /api/v1/platesolve/status
pub async fn status(State(state): State<AppState>) -> Json<PlateSolveStatus> {
    crate::mount::auto_solve::attach(&state);
    let last_solve = plate_solve_repo::latest_successful_solve(state.local_store.pool())
        .await
        .unwrap_or_else(|error| {
            tracing::warn!("Could not read the latest plate solve: {error}");
            None
        });
    Json(PlateSolveStatus {
        solver: state.plate_solver.status(),
        auto_solve: state.plate_solve_trigger.lock().unwrap().status(),
        last_solve,
    })
}

/// POST /api/v1/platesolve/solve
pub async fn solve_frame(
    State(state): State<AppState>,
    Json(request): Json<SolveFrameRequest>,
) -> Result<Json<SolveFrameResponse>, ApiError> {
    solve_and_record(&state, request, "manual").await.map(Json)
}

/// Solve one archived frame and record the attempt, whatever asked for it.
/// `trigger` is stored with the record: `manual`, or the automatic rule.
pub(crate) async fn solve_and_record(
    state: &AppState,
    request: SolveFrameRequest,
    trigger: &str,
) -> Result<SolveFrameResponse, ApiError> {
    let hint = request.hint.as_ref().map(|hint| PointingHint {
        ra_hours: hint.ra_hours,
        dec_deg: hint.dec_deg,
        search_radius_deg: hint
            .search_radius_deg
            .unwrap_or_else(|| state.plate_solver.default_search_radius_deg()),
    });
    // Validate what the caller typed before touching the archive.
    SolveRequest {
        hint,
        pixel_scale_arcsec: request.pixel_scale_arcsec,
    }
    .validate()
    .map_err(|message| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_plate_solve_request",
            message,
        )
    })?;

    let frame = plate_solve_repo::load_solvable_frame(state.local_store.pool(), &request.frame_id)
        .await
        .map_err(|e| {
            ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "storage_error",
                e.to_string(),
            )
        })?
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::NOT_FOUND,
                "frame_not_found",
                format!("No frame {}", request.frame_id),
            )
        })?;
    let asset = frame.asset.clone().ok_or_else(|| {
        ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "no_solvable_asset",
            "This frame has no FITS a solver can read. A frame archived as camera RAW alone \
             needs a build with LibRaw to derive one.",
        )
    })?;

    let scale_hint = scale_hint(&request, state.rig_optics.as_ref(), frame.binning);
    let solve_request = SolveRequest {
        hint,
        pixel_scale_arcsec: scale_hint.map(|s| s.arcsec_per_px),
    };
    let path = state.capture_store.root().join(&asset.relative_path);
    let attempt = state
        .plate_solver
        .solve_archived_fits(&path, Some(&asset.sha256), &solve_request)
        .await
        .map_err(solve_error)?;

    let mode = if hint.is_some() { "hinted" } else { "blind" };
    let solver_command = attempt.command.join(" ");
    tracing::debug!(
        "Plate solve of {} frame {} from {}",
        frame.capture_type.as_deref().unwrap_or("unknown-type"),
        frame.frame_id,
        frame.camera_name.as_deref().unwrap_or("an unnamed camera")
    );
    match &attempt.outcome {
        Ok(solution) => tracing::info!(
            "Plate solve {}: RA {:.4}h Dec {:+.4}° {:.3}\"/px PA {:.1}° ({} ms, {mode}, {trigger})",
            frame.frame_id,
            solution.ra_hours,
            solution.dec_deg,
            solution.pixel_scale_arcsec,
            solution.position_angle_deg,
            attempt.elapsed_ms
        ),
        Err(failure) => tracing::warn!(
            "Plate solve {} found no position ({}): {} ({} ms, {mode}, {trigger})",
            frame.frame_id,
            failure.kind.as_str(),
            failure.detail,
            attempt.elapsed_ms
        ),
    }

    let record = PlateSolveRecord {
        session_id: frame.session_id.clone(),
        frame_id: frame.frame_id.clone(),
        asset_id: asset.id.clone(),
        asset_sha256: asset.sha256.clone(),
        solver: "astap".into(),
        trigger: trigger.into(),
        mode: mode.into(),
        hint_ra_deg: hint.map(|h| h.ra_hours * 15.0),
        hint_dec_deg: hint.map(|h| h.dec_deg),
        search_radius_deg: hint.map(|h| h.search_radius_deg),
        scale_hint_arcsec: scale_hint.map(|s| s.arcsec_per_px),
        scale_hint_source: scale_hint.map(|s| s.source.to_string()),
        success: attempt.outcome.is_ok(),
        ra_deg: attempt.outcome.as_ref().ok().map(|s| s.ra_deg),
        dec_deg: attempt.outcome.as_ref().ok().map(|s| s.dec_deg),
        position_angle_deg: attempt.outcome.as_ref().ok().map(|s| s.position_angle_deg),
        pixel_scale_arcsec: attempt.outcome.as_ref().ok().map(|s| s.pixel_scale_arcsec),
        field_width_deg: attempt.outcome.as_ref().ok().map(|s| s.field_width_deg),
        field_height_deg: attempt.outcome.as_ref().ok().map(|s| s.field_height_deg),
        mirrored: attempt.outcome.as_ref().ok().map(|s| s.mirrored),
        wcs_json: attempt
            .outcome
            .as_ref()
            .ok()
            .and_then(|s| serde_json::to_string(&s.wcs).ok()),
        failure_kind: attempt
            .outcome
            .as_ref()
            .err()
            .map(|f| f.kind.as_str().to_string()),
        solver_message: match &attempt.outcome {
            Ok(solution) => solution.warning.clone(),
            Err(failure) => Some(failure.detail.clone()),
        },
        solver_command: solver_command.clone(),
        solve_time_ms: i64::try_from(attempt.elapsed_ms).unwrap_or(i64::MAX),
    };
    let (event_id, record_error) =
        match plate_solve_repo::record_solve(state.local_store.pool(), &record).await {
            Ok(id) => (Some(id), None),
            Err(error) => {
                tracing::error!(
                    "Could not record plate solve of {}: {error}",
                    frame.frame_id
                );
                (None, Some(error.to_string()))
            }
        };

    let (solution, failure) = match attempt.outcome {
        Ok(solution) => (Some(solution), None),
        Err(failure) => (None, Some(failure)),
    };
    let pointing = match &solution {
        Some(solution) => Some(
            match (frame.exposure_started_at, frame.exposure_completed_at) {
                (Some(start), Some(end)) => {
                    record_pointing_sample(
                        state,
                        PointingInput {
                            plate_solve_event_id: event_id.clone(),
                            frame_id: Some(frame.frame_id.clone()),
                            exposure_start: start,
                            exposure_end: end.max(start),
                            solved_j2000: EqCoord::new(
                                solution.ra_hours,
                                solution.dec_deg,
                                EqFrame::J2000,
                            ),
                            purpose: pointing_purpose(trigger),
                        },
                    )
                    .await
                }
                _ => PointingOutcome::skipped("exposure_time_unknown"),
            },
        ),
        None => None,
    };
    Ok(SolveFrameResponse {
        frame_id: frame.frame_id,
        event_id,
        success: solution.is_some(),
        solution,
        failure,
        trigger: trigger.into(),
        mode,
        scale_hint,
        fov_height_deg: attempt.fov_height_deg,
        solve_time_ms: attempt.elapsed_ms,
        solver_command,
        solved_asset: SolvedAsset {
            id: asset.id,
            role: asset.role,
            relative_path: asset.relative_path,
            sha256: asset.sha256,
        },
        image: attempt.geometry,
        record_error,
        pointing,
    })
}

// ═══════════════════════════════════════════════════════════════════
//  Pointing samples (H12)
// ═══════════════════════════════════════════════════════════════════

/// `pointing_samples.purpose` for a solve trigger.
pub fn pointing_purpose(trigger: &str) -> PointingPurpose {
    match trigger {
        "manual" => PointingPurpose::Manual,
        "center" => PointingPurpose::Center,
        _ => PointingPurpose::AutoSolve,
    }
}

/// One solved exposure, as H12 needs it.
#[derive(Debug, Clone)]
pub struct PointingInput {
    pub plate_solve_event_id: Option<String>,
    pub frame_id: Option<String>,
    /// The driver's exposure start and end.
    pub exposure_start: DateTime<Utc>,
    pub exposure_end: DateTime<Utc>,
    /// The solved centre (ICRS, i.e. J2000).
    pub solved_j2000: EqCoord,
    pub purpose: PointingPurpose,
}

/// What H12 did for one solve.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PointingOutcome {
    pub sample_id: Option<uuid::Uuid>,
    /// Mount minus truth, arcmin (also `plate_solve_events.pointing_error_arcmin`).
    pub error_arcmin: Option<f64>,
    pub err_ra_arcsec: Option<f64>,
    pub err_dec_arcsec: Option<f64>,
    pub mount_frame: Option<EqFrame>,
    pub pier_side: Option<PierSide>,
    pub sync_generation: Option<u64>,
    /// The `pointing_model_snapshots` row the refit wrote, if any.
    pub model_snapshot_id: Option<String>,
    /// Why no sample was written.
    pub skipped: Option<String>,
}

impl PointingOutcome {
    pub fn skipped(reason: impl Into<String>) -> Self {
        Self {
            sample_id: None,
            error_arcmin: None,
            err_ra_arcsec: None,
            err_dec_arcsec: None,
            mount_frame: None,
            pier_side: None,
            sync_generation: None,
            model_snapshot_id: None,
            skipped: Some(reason.into()),
        }
    }
}

/// What the sample builder needs besides the window.
#[derive(Debug, Clone)]
pub struct PointingContext {
    pub mount_identity_key: String,
    pub mount_connection_id: uuid::Uuid,
    pub sync_mode: MountSyncMode,
    pub jnow_flavour: JNowFlavour,
    pub site: Option<Site>,
    /// Ambient temperature at the exposure, when one is known.
    pub temperature_c: Option<f64>,
}

fn signed_ra_delta_h(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(24.0);
    if d >= 12.0 {
        d - 24.0
    } else {
        d
    }
}

/// The mount snapshot nearest mid-exposure.
fn window_mid_snapshot(window: &DeviceWindow) -> Option<&MountSnapshot> {
    window
        .mount
        .mid
        .as_ref()
        .or(window.mount.start.as_ref())
        .or(window.mount.end.as_ref())
}

/// Build the H12 sample for one solve from the exposure's device window.
/// `Err` is the reason no sample can be written.
pub fn pointing_sample_from_window(
    window: &DeviceWindow,
    ctx: &PointingContext,
    converter: &dyn FrameConverter,
    input: &PointingInput,
) -> Result<PointingSampleRecord, &'static str> {
    let snapshot = window_mid_snapshot(window).ok_or("no_mount_samples_in_window")?;
    if snapshot.connection_id != ctx.mount_connection_id {
        return Err("mount_reconnected_since_exposure");
    }
    if window.motion.iter().any(|m| m.slewing == Some(true))
        || [&window.mount.start, &window.mount.mid, &window.mount.end]
            .iter()
            .filter_map(|s| s.as_ref())
            .any(|s| s.state.slewing.known() == Some(&true))
    {
        return Err("mount_slewing_during_exposure");
    }
    let generations: Vec<u64> = [&window.mount.start, &window.mount.mid, &window.mount.end]
        .iter()
        .filter_map(|s| s.as_ref().map(|s| s.sync_generation))
        .collect();
    if generations.windows(2).any(|pair| pair[0] != pair[1]) {
        return Err("sync_during_exposure");
    }
    let caps = &snapshot.capabilities;
    let frame = caps.native_frame;
    if frame == EqFrame::Unknown {
        return Err("mount_frame_unknown");
    }
    let native = window
        .mount_mid
        .map(|c| EqCoord { frame, ..c })
        .or_else(|| crate::equipment::mount::position(&snapshot.state, frame))
        .ok_or("mount_position_unknown")?;
    let mid = input.exposure_start + (input.exposure_end - input.exposure_start) / 2;
    let mount_j2000 = if frame == EqFrame::J2000 {
        native
    } else {
        converter.convert(native, EqFrame::J2000, mid, ctx.site.as_ref())
    };
    let truth = input.solved_j2000;
    let err_ra = signed_ra_delta_h(mount_j2000.ra_h, truth.ra_h)
        * 15.0
        * 3600.0
        * truth.dec_deg.to_radians().cos();
    let err_dec = (mount_j2000.dec_deg - truth.dec_deg) * 3600.0;
    let err_total = mount_j2000
        .separation_arcmin(&truth)
        .map(|arcmin| arcmin * 60.0);
    let lst_h = snapshot.state.sidereal_time_h.known().copied().or_else(|| {
        ctx.site
            .map(|site| crate::equipment::astro::time::local_sidereal_time_h(mid, site.lon_deg))
    });
    let ha_h = lst_h.map(|lst| signed_ra_delta_h(lst, native.ra_h));
    let (alt_deg, az_deg) = match (
        snapshot.state.alt_deg.known(),
        snapshot.state.az_deg.known(),
    ) {
        (Some(alt), Some(az)) => (Some(*alt), Some(*az)),
        _ => match (ha_h, ctx.site) {
            (Some(ha), Some(site)) => {
                let altaz =
                    crate::equipment::mount::limits::alt_az(ha, native.dec_deg, site.lat_deg);
                (Some(altaz.alt_deg), Some(altaz.az_deg))
            }
            _ => (None, None),
        },
    };
    Ok(PointingSampleRecord {
        id: uuid::Uuid::new_v4(),
        plate_solve_event_id: input.plate_solve_event_id.clone(),
        frame_id: input.frame_id.clone(),
        mount_identity_key: ctx.mount_identity_key.clone(),
        mount_frame: frame,
        jnow_flavour: ctx.jnow_flavour,
        does_refraction: caps.does_refraction,
        mount_ra_native_h: Some(native.ra_h),
        mount_dec_native_deg: Some(native.dec_deg),
        mount_ra_j2000_h: Some(mount_j2000.ra_h),
        mount_dec_j2000_deg: Some(mount_j2000.dec_deg),
        solved_ra_j2000_h: truth.ra_h,
        solved_dec_j2000_deg: truth.dec_deg,
        err_ra_arcsec: err_ra.is_finite().then_some(err_ra),
        err_dec_arcsec: err_dec.is_finite().then_some(err_dec),
        err_total_arcsec: err_total.filter(|v| v.is_finite()),
        pier_side: snapshot
            .state
            .side_of_pier
            .known()
            .copied()
            .unwrap_or(PierSide::Unknown),
        ha_h,
        lst_h,
        alt_deg,
        az_deg,
        sync_generation: snapshot.sync_generation,
        sync_mode: ctx.sync_mode,
        coordinate_precision_arcsec: caps.coordinate_precision_arcsec,
        temperature_c: ctx.temperature_c,
        mid_exposure_utc: Some(mid),
        purpose: input.purpose,
    })
}

/// The connected mount's handle: the hub's slot, else the state slot.
fn mount_handle(state: &AppState) -> Option<MountHandle> {
    state
        .equipment
        .handle::<MountHandle>()
        .or_else(|| super::equipment::mount_slot(state))
}

/// H12 for one solve: build the sample from the hub's device window, write
/// it, fill `plate_solve_events.pointing_error_arcmin`, and refit the
/// pointing model. Never fails the solve: a problem is reported in the
/// outcome and logged.
pub async fn record_pointing_sample(state: &AppState, input: PointingInput) -> PointingOutcome {
    let Some(handle) = mount_handle(state) else {
        return PointingOutcome::skipped("no_mount_connected");
    };
    let identity = handle.identity();
    let window = state
        .equipment
        .window(input.exposure_start, input.exposure_end);
    let configs = &state.equipment_ctx.configs;
    let site = state
        .equipment_ctx
        .mount_shared()
        .and_then(|shared| shared.geometry_site())
        .or_else(|| state.equipment.site());
    let ambient = state.equipment.ambient_temperature();
    let temperature_c = window
        .weather
        .as_ref()
        .and_then(|w| w.temperature_c.known().copied())
        .or_else(|| {
            ambient
                .known()
                .filter(|t| !t.camera_sensor_fallback && t.role != TemperatureRole::Unknown)
                .map(|t| t.celsius)
        });
    let ctx = PointingContext {
        mount_identity_key: identity.identity_key.clone(),
        mount_connection_id: identity.connection_id,
        sync_mode: configs.mount.sync_mode,
        jnow_flavour: configs.mount.jnow_flavour,
        site,
        temperature_c,
    };
    let converter = crate::equipment::astro::frame_converter();
    let record = match pointing_sample_from_window(&window, &ctx, converter.as_ref(), &input) {
        Ok(record) => record,
        Err(reason) => {
            tracing::info!(
                "No pointing sample for {}: {reason}",
                input.frame_id.as_deref().unwrap_or("this solve")
            );
            return PointingOutcome::skipped(reason);
        }
    };
    let pool = state.local_store.pool();
    if let Err(error) = crate::storage::pointing_samples_repo::upsert(pool, &record).await {
        tracing::error!("Could not record a pointing sample: {error}");
        return PointingOutcome::skipped("pointing_sample_store_failed");
    }
    let error_arcmin = record.err_total_arcsec.map(|arcsec| arcsec / 60.0);
    if let (Some(event_id), Some(arcmin)) = (&record.plate_solve_event_id, error_arcmin) {
        if let Err(error) = plate_solve_repo::set_pointing_error(pool, event_id, arcmin).await {
            tracing::warn!("Could not fill pointing_error_arcmin for {event_id}: {error}");
        }
    }
    let model_snapshot_id = match crate::mount::pointing_model::refit_snapshot(
        pool,
        Some(&state.rig_id),
        &record,
    )
    .await
    {
        Ok(snapshot) => snapshot.map(|s| s.id),
        Err(error) => {
            tracing::warn!("Could not refit the pointing model: {error}");
            None
        }
    };
    PointingOutcome {
        sample_id: Some(record.id),
        error_arcmin,
        err_ra_arcsec: record.err_ra_arcsec,
        err_dec_arcsec: record.err_dec_arcsec,
        mount_frame: Some(record.mount_frame),
        pier_side: Some(record.pier_side),
        sync_generation: Some(record.sync_generation),
        model_snapshot_id,
        skipped: None,
    }
}

fn solve_error(error: SolveError) -> ApiError {
    let message = error.to_string();
    let (status, code) = match error {
        SolveError::Unavailable(_) => (StatusCode::SERVICE_UNAVAILABLE, "plate_solver_unavailable"),
        SolveError::Busy => (StatusCode::CONFLICT, "plate_solver_busy"),
        SolveError::ChecksumMismatch { .. } => (StatusCode::CONFLICT, "asset_checksum_mismatch"),
        SolveError::Input(_) => (StatusCode::UNPROCESSABLE_ENTITY, "asset_unreadable"),
        SolveError::WorkingDirectory(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "plate_solve_workspace_error",
        ),
    };
    ApiError::new(status, code, message)
}

/// Arcsec per binned pixel: the request's value first, then the rig optics.
///
/// Unknown binning means an unknown scale. A 2x2 frame read as 1x1 halves the
/// field height, which sends the solver looking for star patterns at the wrong
/// size; letting it search for the scale is slower but cannot be misled.
fn scale_hint(
    request: &SolveFrameRequest,
    optics: Option<&RigOptics>,
    binning: Option<i64>,
) -> Option<ScaleHint> {
    if let Some(arcsec_per_px) = request.pixel_scale_arcsec {
        return Some(ScaleHint {
            arcsec_per_px,
            source: "request",
        });
    }
    if request.ignore_rig_optics {
        return None;
    }
    let optics = optics?;
    let binning = binning.filter(|b| *b >= 1)? as f64;
    let arcsec_per_px =
        ARCSEC_PER_UM_PER_MM * optics.pixel_size_um? * binning / optics.focal_length_mm?;
    Some(ScaleHint {
        arcsec_per_px,
        source: "rig_optics",
    })
}

/// A daemon state over a real archive and database, shared with the
/// automatic-solve tests.
#[cfg(test)]
pub(crate) mod test_support {
    use crate::config::RigOptics;
    use crate::processing::plate_solver::PlateSolver;
    use crate::state::AppState;
    use std::path::{Path, PathBuf};

    /// A fresh archive holding one light frame whose sensor FITS is `fits`,
    /// solved by `solver`.
    pub(crate) async fn archive_with_frame(
        directory: &Path,
        solver: PlateSolver,
        fits: Vec<u8>,
        binning: i64,
        rig_optics: Option<RigOptics>,
    ) -> (AppState, String) {
        use crate::config::{CameraConfig, FocusConfig};
        use crate::storage::capture_store::CaptureStore;

        let store = crate::storage::local_store::LocalStore::open(
            directory.join("achernar.db").to_str().unwrap(),
        )
        .await
        .unwrap();
        let mut focus = FocusConfig::default();
        focus.models.quality_model_path =
            Some(directory.join("absent.onnx").to_string_lossy().into_owned());
        let capture_store = CaptureStore::new(directory.join("captures"));
        let state = AppState::new(
            focus,
            CameraConfig::default(),
            store,
            crate::storage::frames_repo::FrameStore::new(directory.join("preview")),
            capture_store,
            "solve-rig".into(),
            directory.join("data"),
            rig_optics,
        )
        .with_plate_solver(solver);

        let frame_id = persist_frame(&state, fits, binning, "light", 1_792_029_600.0, 10.0).await;
        (state, frame_id)
    }

    /// Commit one frame of `capture_type` to `state`'s archive, exposed from
    /// `started_unix` for `exposure_s`, and return its id.
    pub(crate) async fn persist_frame(
        state: &AppState,
        fits: Vec<u8>,
        binning: i64,
        capture_type: &str,
        started_unix: f64,
        exposure_s: f64,
    ) -> String {
        use crate::storage::capture_store::{
            AssetRole, PendingAsset, PendingCapture, TemperatureState,
        };
        let frame_id = uuid::Uuid::new_v4().to_string();
        state
            .capture_store
            .persist_capture(
                state.local_store.pool(),
                PendingCapture {
                    capture_id: format!("capture-{frame_id}"),
                    frame_id: frame_id.clone(),
                    session_id: None,
                    rig_id: state.rig_id.clone(),
                    camera_name: Some("fixture".into()),
                    driver_name: Some("test".into()),
                    capture_type: capture_type.into(),
                    exposure_started_at: started_unix,
                    exposure_completed_at: started_unix + exposure_s,
                    requested_exposure_s: exposure_s,
                    requested_gain: Some(100.0),
                    applied_exposure_s: exposure_s,
                    iso: None,
                    shutter_label: None,
                    offset_val: None,
                    binning,
                    filter_name: None,
                    focuser_position: None,
                    temperature: None,
                    temperature_source: None,
                    temperature_state: TemperatureState::Missing,
                    acquisition: serde_json::json!({}),
                    assets: vec![PendingAsset::new(
                        AssetRole::SensorFits,
                        "image.fits",
                        "application/fits",
                        fits,
                    )],
                },
            )
            .await
            .unwrap();
        frame_id
    }

    pub(crate) fn scratch() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("achernar-solve-api-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    pub(crate) async fn close(state: AppState, directory: PathBuf) {
        let pool = state.local_store.pool().clone();
        drop(state);
        pool.close().await;
        std::fs::remove_dir_all(directory).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(pixel_scale_arcsec: Option<f64>, ignore_rig_optics: bool) -> SolveFrameRequest {
        SolveFrameRequest {
            frame_id: "f".into(),
            hint: None,
            pixel_scale_arcsec,
            ignore_rig_optics,
        }
    }

    const OPTICS: RigOptics = RigOptics {
        focal_ratio: Some(5.6),
        focal_length_mm: Some(448.0),
        pixel_size_um: Some(3.76),
    };

    #[test]
    fn rig_optics_give_the_binned_scale() {
        // 206.265 * 3.76 / 448 = 1.731"/px; binned 2x2, 3.462"/px.
        let hint = scale_hint(&request(None, false), Some(&OPTICS), Some(2)).unwrap();
        assert_eq!(hint.source, "rig_optics");
        assert!(
            (hint.arcsec_per_px - 3.4622).abs() < 1e-3,
            "{}",
            hint.arcsec_per_px
        );
    }

    #[test]
    fn the_request_outranks_the_rig_row() {
        let hint = scale_hint(&request(Some(0.9), true), Some(&OPTICS), Some(1)).unwrap();
        assert_eq!((hint.arcsec_per_px, hint.source), (0.9, "request"));
    }

    #[test]
    fn missing_facts_leave_the_scale_to_the_solver() {
        assert!(scale_hint(&request(None, false), None, Some(1)).is_none());
        assert!(
            scale_hint(&request(None, false), Some(&OPTICS), None).is_none(),
            "binning unknown"
        );
        assert!(scale_hint(&request(None, true), Some(&OPTICS), Some(1)).is_none());
        let no_focal_length = RigOptics {
            focal_length_mm: None,
            ..OPTICS
        };
        assert!(scale_hint(&request(None, false), Some(&no_focal_length), Some(1)).is_none());
    }

    use super::test_support::{archive_with_frame, close, scratch};
    use crate::processing::plate_solver::PlateSolver;
    use std::path::Path;

    fn solve(frame_id: &str, hint: Option<HintRequest>) -> SolveFrameRequest {
        SolveFrameRequest {
            frame_id: frame_id.into(),
            hint,
            pixel_scale_arcsec: None,
            ignore_rig_optics: false,
        }
    }

    /// Through the real handler, archive and database, with a stand-in ASTAP.
    #[cfg(unix)]
    mod end_to_end {
        use super::*;
        use crate::config::PlateSolveConfig;
        use crate::processing::astap::test_support::{fake_astap, SOLVED_INI};
        use crate::processing::astap::AstapSolver;
        use crate::processing::plate_solver::SolveFailureKind;

        /// One 2x2-binned 8 x 6 mono capture, solved by a script that answers
        /// with `ini` and `code`.
        async fn archive_with_one_frame(
            directory: &Path,
            ini: Option<&str>,
            code: i32,
        ) -> (AppState, String) {
            let solver = PlateSolver::with_backend_for_test(
                AstapSolver::new(
                    fake_astap(directory, ini, code, 0.0),
                    &PlateSolveConfig::default(),
                ),
                directory.join("work"),
            );
            let meta = crate::fits::FitsMeta {
                exposure_s: 10.0,
                gain: 100,
                temperature_c: f64::NAN,
                binning: 2,
                instrument: "fixture".into(),
                date_obs: "2026-10-15T02:00:00Z".into(),
                iso: None,
                shutter_label: None,
                source_file: None,
                calibration: None,
            };
            let fits = crate::fits::encode_bayer_u16(&[0u8; 8 * 6 * 2], 8, 6, &meta, None).unwrap();
            archive_with_frame(directory, solver, fits, 2, Some(OPTICS)).await
        }

        #[tokio::test]
        async fn a_solved_frame_is_answered_and_recorded_against_its_exposure() {
            let directory = scratch();
            let (state, frame_id) = archive_with_one_frame(&directory, Some(SOLVED_INI), 0).await;
            let hint = HintRequest {
                ra_hours: 5.5,
                dec_deg: -5.0,
                search_radius_deg: None,
            };

            let Json(response) =
                solve_frame(State(state.clone()), Json(solve(&frame_id, Some(hint))))
                    .await
                    .unwrap();
            assert!(response.success);
            assert_eq!(response.mode, "hinted");
            assert_eq!(response.solved_asset.role, "sensor_fits");
            assert_eq!((response.image.width, response.image.height), (8, 6));
            assert_eq!(response.image.binning, Some(2));
            // Binning comes from the frames row, optics from the rigs row.
            let scale = response.scale_hint.unwrap();
            assert_eq!(scale.source, "rig_optics");
            assert!((scale.arcsec_per_px - 3.4622).abs() < 1e-3);
            assert!(
                (response.fov_height_deg.unwrap() - scale.arcsec_per_px * 6.0 / 3600.0).abs()
                    < 1e-12
            );

            // The solver searched where it was told, at the default radius.
            let args = std::fs::read_to_string(directory.join("args.txt")).unwrap();
            let args: Vec<&str> = args.lines().collect();
            let after = |flag: &str| args[args.iter().position(|a| *a == flag).unwrap() + 1];
            assert_eq!(after("-ra"), "5.5");
            assert_eq!(after("-spd"), "85");
            assert_eq!(after("-r"), "30");

            let row: (String, String, i64, f64, f64, String, String) = sqlx::query_as(
                "SELECT frame_id, asset_sha256, success, hint_ra_deg, solved_ra, mode, scale_hint_source \
                 FROM plate_solve_events WHERE id = ?",
            )
            .bind(response.event_id.as_deref().unwrap())
            .fetch_one(state.local_store.pool())
            .await
            .unwrap();
            assert_eq!(row.0, frame_id);
            assert_eq!(row.1, response.solved_asset.sha256);
            assert_eq!(
                (row.2, row.3, row.5.as_str(), row.6.as_str()),
                (1, 82.5, "hinted", "rig_optics")
            );
            // The canned answer was written for a 4000 x 3000 frame, so the
            // 8 x 6 archive's centre is projected well away from CRVAL. The
            // record must carry the projected centre, exactly as answered.
            let solution = response.solution.unwrap();
            assert_eq!(row.4, solution.ra_deg);
            assert!((solution.ra_deg - 83.822).abs() > 0.1);

            close(state, directory).await;
        }

        #[tokio::test]
        async fn no_stars_is_a_recorded_answer_not_an_http_error() {
            let directory = scratch();
            let unsolved = "PLTSOLVD=F\nERROR=Not enough stars\nWARNING=";
            let (state, frame_id) = archive_with_one_frame(&directory, Some(unsolved), 2).await;

            let Json(response) = solve_frame(State(state.clone()), Json(solve(&frame_id, None)))
                .await
                .unwrap();
            assert!(!response.success);
            assert_eq!(response.mode, "blind");
            assert_eq!(
                response.failure.unwrap().kind,
                SolveFailureKind::NotEnoughStars
            );

            let (success, kind, message): (i64, String, String) = sqlx::query_as(
                "SELECT success, failure_kind, solver_message FROM plate_solve_events WHERE frame_id = ?",
            )
            .bind(&frame_id)
            .fetch_one(state.local_store.pool())
            .await
            .unwrap();
            assert_eq!(
                (success, kind.as_str(), message.as_str()),
                (0, "not_enough_stars", "Not enough stars")
            );

            close(state, directory).await;
        }

        #[tokio::test]
        async fn bad_requests_are_refused_before_the_solver_runs() {
            let directory = scratch();
            let (state, frame_id) = archive_with_one_frame(&directory, Some(SOLVED_INI), 0).await;

            let unknown = solve_frame(State(state.clone()), Json(solve("no-such-frame", None)))
                .await
                .unwrap_err();
            assert_eq!(
                (unknown.status, unknown.code),
                (StatusCode::NOT_FOUND, "frame_not_found")
            );

            let off_sphere = HintRequest {
                ra_hours: 25.0,
                dec_deg: 0.0,
                search_radius_deg: None,
            };
            let invalid = solve_frame(
                State(state.clone()),
                Json(solve(&frame_id, Some(off_sphere))),
            )
            .await
            .unwrap_err();
            assert_eq!(invalid.status, StatusCode::BAD_REQUEST);

            assert!(
                !directory.join("args.txt").exists(),
                "solver must not have run"
            );
            let (rows,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM plate_solve_events")
                .fetch_one(state.local_store.pool())
                .await
                .unwrap();
            assert_eq!(rows, 0);

            close(state, directory).await;
        }
    }

    /// H12: the pointing sample built from a device window.
    mod pointing {
        use super::*;
        use crate::equipment::snapshot::{
            MotionInterval, MountCapabilities, MountState, WindowSamples,
        };
        use crate::equipment::traits::IdentityConverter;
        use crate::equipment::types::{Observed, Source};

        fn at(ms: i64) -> DateTime<Utc> {
            DateTime::from_timestamp(1_792_029_600, 0).unwrap() + chrono::Duration::milliseconds(ms)
        }

        fn snapshot(connection: uuid::Uuid, ms: i64, generation: u64) -> MountSnapshot {
            let t = at(ms);
            let fresh = |v| Observed::fresh(v, Source::DeviceState, t, 0);
            MountSnapshot {
                connection_id: connection,
                state: MountState {
                    ra_h: fresh(5.62),
                    dec_deg: fresh(-5.30),
                    alt_deg: fresh(41.0),
                    az_deg: fresh(171.0),
                    sidereal_time_h: fresh(6.12),
                    slewing: Observed::fresh(false, Source::DeviceState, t, 0),
                    side_of_pier: Observed::fresh(PierSide::West, Source::DeviceState, t, 0),
                    ..MountState::default()
                },
                capabilities: MountCapabilities {
                    native_frame: EqFrame::JNowApparent,
                    does_refraction: Some(false),
                    coordinate_precision_arcsec: Some(1.0),
                    ..MountCapabilities::default()
                },
                sync_generation: generation,
                published_at: t,
                ..MountSnapshot::default()
            }
        }

        fn window(connection: uuid::Uuid, generations: [u64; 2]) -> DeviceWindow {
            DeviceWindow {
                start: at(0),
                end: at(4000),
                mount: WindowSamples {
                    start: Some(snapshot(connection, 0, generations[0])),
                    mid: Some(snapshot(connection, 2000, generations[0])),
                    end: Some(snapshot(connection, 4000, generations[1])),
                    end_stale: false,
                    samples_n: 3,
                },
                mount_mid: Some(EqCoord::new(5.62, -5.30, EqFrame::JNowApparent)),
                motion: vec![MotionInterval {
                    from: at(0),
                    to: at(4000),
                    tracking: Some(true),
                    slewing: Some(false),
                    pulse_guiding: Some(false),
                }],
                ..DeviceWindow::default()
            }
        }

        fn context(connection: uuid::Uuid) -> PointingContext {
            PointingContext {
                mount_identity_key: "alpaca:esp32-mount".into(),
                mount_connection_id: connection,
                sync_mode: MountSyncMode::Offset,
                jnow_flavour: JNowFlavour::Apparent,
                site: None,
                temperature_c: Some(6.5),
            }
        }

        fn input() -> PointingInput {
            PointingInput {
                plate_solve_event_id: Some("solve-1".into()),
                frame_id: Some("frame-1".into()),
                exposure_start: at(0),
                exposure_end: at(4000),
                solved_j2000: EqCoord::new(5.60, -5.32, EqFrame::J2000),
                purpose: PointingPurpose::Center,
            }
        }

        #[test]
        fn a_sample_stores_epoch_pier_side_and_sync_generation() {
            let connection = uuid::Uuid::new_v4();
            let record = pointing_sample_from_window(
                &window(connection, [4, 4]),
                &context(connection),
                &IdentityConverter,
                &input(),
            )
            .unwrap();
            assert_eq!(record.mount_frame, EqFrame::JNowApparent);
            assert_eq!(record.jnow_flavour, JNowFlavour::Apparent);
            assert_eq!(record.pier_side, PierSide::West);
            assert_eq!(record.sync_generation, 4);
            assert_eq!(record.sync_mode, MountSyncMode::Offset);
            assert_eq!(record.purpose, PointingPurpose::Center);
            assert_eq!(record.does_refraction, Some(false));
            assert_eq!(record.mid_exposure_utc, Some(at(2000)));
            assert_eq!(record.plate_solve_event_id.as_deref(), Some("solve-1"));
            assert_eq!(record.mount_ra_native_h, Some(5.62));
            // Mount minus truth: +0.02 h in RA at Dec -5.32, +0.02 deg in Dec.
            let cos_dec = (-5.32_f64).to_radians().cos();
            assert!((record.err_ra_arcsec.unwrap() - 0.02 * 15.0 * 3600.0 * cos_dec).abs() < 1e-6);
            assert!((record.err_dec_arcsec.unwrap() - 72.0).abs() < 1e-6);
            assert!(record.err_total_arcsec.unwrap() > 72.0);
            // HA from the mount's own LST and native RA.
            assert!((record.ha_h.unwrap() - 0.5).abs() < 1e-9);
            assert_eq!((record.alt_deg, record.az_deg), (Some(41.0), Some(171.0)));
            assert_eq!(record.temperature_c, Some(6.5));
        }

        #[test]
        fn no_sample_is_written_without_a_trustworthy_window() {
            let connection = uuid::Uuid::new_v4();
            let ctx = context(connection);
            let check = |window: DeviceWindow, ctx: &PointingContext| {
                pointing_sample_from_window(&window, ctx, &IdentityConverter, &input()).unwrap_err()
            };
            assert_eq!(
                check(DeviceWindow::default(), &ctx),
                "no_mount_samples_in_window"
            );
            assert_eq!(
                check(window(connection, [4, 5]), &ctx),
                "sync_during_exposure"
            );
            let mut slewing = window(connection, [4, 4]);
            slewing.motion[0].slewing = Some(true);
            assert_eq!(check(slewing, &ctx), "mount_slewing_during_exposure");
            let mut unknown = window(connection, [4, 4]);
            for sample in [
                &mut unknown.mount.start,
                &mut unknown.mount.mid,
                &mut unknown.mount.end,
            ] {
                sample.as_mut().unwrap().capabilities.native_frame = EqFrame::Unknown;
            }
            assert_eq!(check(unknown, &ctx), "mount_frame_unknown");
            assert_eq!(
                check(window(connection, [4, 4]), &context(uuid::Uuid::new_v4())),
                "mount_reconnected_since_exposure"
            );
        }

        #[test]
        fn the_trigger_names_the_purpose() {
            assert_eq!(pointing_purpose("manual"), PointingPurpose::Manual);
            assert_eq!(pointing_purpose("center"), PointingPurpose::Center);
            assert_eq!(pointing_purpose("post_slew"), PointingPurpose::AutoSolve);
            assert_eq!(pointing_purpose("recovery"), PointingPurpose::AutoSolve);
        }
    }

    /// H12 through the real handler, a sim mount behind its worker and a
    /// stand-in ASTAP.
    #[cfg(unix)]
    mod pointing_end_to_end {
        use super::super::test_support::persist_frame;
        use super::*;
        use crate::config::PlateSolveConfig;
        use crate::equipment::commands::{GatedOp, MountCmd};
        use crate::equipment::types::Priority;
        use crate::processing::astap::test_support::{fake_astap, SOLVED_INI};
        use crate::processing::astap::AstapSolver;

        fn unix(at: DateTime<Utc>) -> f64 {
            at.timestamp_micros() as f64 / 1e6
        }

        /// Expose for 300 ms now, with mount samples either side.
        async fn solved_frame(state: &AppState) -> SolveFrameResponse {
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
            let start = Utc::now();
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
            let fits =
                crate::fits::encode_linear_analysis(&[1.0; 48], 8, 6, "ADU", "test").unwrap();
            let frame_id = persist_frame(state, fits, 1, "light", unix(start), 0.3).await;
            solve_and_record(state, solve(&frame_id, None), "manual")
                .await
                .unwrap()
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn a_solve_with_a_mount_window_records_a_sample_and_the_pointing_error() {
            let (directory, state) = crate::api::equipment_tests::state().await;
            let solver = PlateSolver::with_backend_for_test(
                AstapSolver::new(
                    fake_astap(&directory, Some(SOLVED_INI), 0, 0.0),
                    &PlateSolveConfig::default(),
                ),
                directory.join("work"),
            );
            let state = state.with_plate_solver(solver);
            let _connected = crate::api::connect_equipment(
                State(state.clone()),
                Json(crate::api::ConnectRequest {
                    equipment_type: Some("mount".into()),
                    driver: "sim".into(),
                    option: None,
                    index: None,
                }),
            )
            .await
            .unwrap_or_else(|err| panic!("{}: {}", err.code, err.message));

            let response = solved_frame(&state).await;
            assert!(response.success);
            let pointing = response.pointing.clone().unwrap();
            assert_eq!(pointing.skipped, None, "{pointing:?}");
            assert_eq!(pointing.mount_frame, Some(EqFrame::JNowApparent));
            assert_eq!(pointing.pier_side, Some(PierSide::East));
            assert_eq!(pointing.sync_generation, Some(0));
            let event_id = response.event_id.clone().unwrap();
            let row: (String, String, Option<String>, i64, String, String) = sqlx::query_as(
                "SELECT mount_frame, jnow_flavour, pier_side, sync_generation, purpose, \
                 mount_identity_key FROM pointing_samples WHERE plate_solve_event_id = ?",
            )
            .bind(&event_id)
            .fetch_one(state.local_store.pool())
            .await
            .unwrap();
            assert_eq!(
                row,
                (
                    "jnow_apparent".into(),
                    "apparent".into(),
                    Some("east".into()),
                    0,
                    "manual".into(),
                    "sim:mount".into()
                )
            );
            let (stored,): (Option<f64>,) =
                sqlx::query_as("SELECT pointing_error_arcmin FROM plate_solve_events WHERE id = ?")
                    .bind(&event_id)
                    .fetch_one(state.local_store.pool())
                    .await
                    .unwrap();
            assert_eq!(stored, pointing.error_arcmin);
            assert!(stored.unwrap() > 0.0);

            // A sync starts a new generation; the next sample carries it.
            let handle = crate::api::equipment::mount_handle(&state).unwrap();
            let snapshot = handle.snapshot();
            let here = snapshot.j2000.unwrap();
            let submitted = crate::api::equipment::submit(
                &state,
                &handle,
                GatedOp::Mount(MountCmd::Sync(EqCoord::new(
                    here.ra_h,
                    here.dec_deg + 0.01,
                    EqFrame::J2000,
                ))),
                crate::api::equipment::source(false, false),
                Priority::Exposure,
            )
            .await
            .unwrap();
            assert!(submitted.ticket.wait().await.is_success());
            let response = solved_frame(&state).await;
            let pointing = response.pointing.unwrap();
            assert_eq!(pointing.skipped, None, "{pointing:?}");
            assert_eq!(pointing.sync_generation, Some(1));

            crate::api::equipment_tests::cleanup(directory, state).await;
        }
    }

    /// The real solver, through the handler, archive and database. The fixture
    /// is any FITS whose header carries a TAN WCS — a SkyView cutout, say. That
    /// header is the truth, and it cannot leak into the solve: every hint is
    /// passed explicitly on the command line, which ASTAP prefers to headers.
    mod real_astap {
        use super::*;
        use crate::config::PlateSolveConfig;
        use crate::processing::plate_solver::TanWcs;

        /// The TAN WCS a fixture's header declares.
        fn header_wcs(bytes: &[u8]) -> TanWcs {
            let mut values = std::collections::HashMap::new();
            for card in bytes.chunks_exact(80) {
                let card = std::str::from_utf8(card).unwrap_or("");
                let key = card.get(..8).unwrap_or("").trim();
                if key == "END" {
                    break;
                }
                if card.get(8..10) == Some("= ") {
                    let value = card[10..].split('/').next().unwrap_or("").trim();
                    if let Ok(number) = value.parse::<f64>() {
                        values.insert(key.to_string(), number);
                    }
                }
            }
            let get = |key: &str| {
                *values
                    .get(key)
                    .unwrap_or_else(|| panic!("fixture header has no {key}"))
            };
            TanWcs {
                crpix: [get("CRPIX1"), get("CRPIX2")],
                crval: [get("CRVAL1"), get("CRVAL2")],
                cd: [[get("CD1_1"), get("CD1_2")], [get("CD2_1"), get("CD2_2")]],
                naxis: [get("NAXIS1") as u32, get("NAXIS2") as u32],
            }
        }

        fn assert_matches_truth(found: &PlateSolution, truth: &PlateSolution, label: &str) {
            let (ra1, dec1) = (found.ra_deg.to_radians(), found.dec_deg.to_radians());
            let (ra2, dec2) = (truth.ra_deg.to_radians(), truth.dec_deg.to_radians());
            let cosine = dec1.sin() * dec2.sin() + dec1.cos() * dec2.cos() * (ra1 - ra2).cos();
            let centre_error_arcsec = cosine.clamp(-1.0, 1.0).acos().to_degrees() * 3600.0;
            let scale_error = (found.pixel_scale_arcsec / truth.pixel_scale_arcsec - 1.0).abs();
            let pa_error = ((found.position_angle_deg - truth.position_angle_deg + 540.0)
                .rem_euclid(360.0)
                - 180.0)
                .abs();
            eprintln!(
                "{label}: centre off by {centre_error_arcsec:.2}\", scale {:.4}\"/px (truth {:.4}), \
                 PA {:.3} deg (truth {:.3}), mirrored {}",
                found.pixel_scale_arcsec,
                truth.pixel_scale_arcsec,
                found.position_angle_deg,
                truth.position_angle_deg,
                found.mirrored
            );
            assert!(
                centre_error_arcsec < 5.0,
                "{label}: centre {centre_error_arcsec}\""
            );
            assert!(scale_error < 0.005, "{label}: scale error {scale_error}");
            assert!(
                pa_error < 0.1,
                "{label}: position angle error {pa_error} deg"
            );
            assert_eq!(found.mirrored, truth.mirrored, "{label}: parity");
        }

        #[tokio::test]
        #[ignore = "set ACHERNAR_ASTAP to an astap_cli/astap executable and ACHERNAR_SOLVE_FIXTURE \
                    to a FITS with a TAN WCS header (e.g. a SkyView cutout); ACHERNAR_ASTAP_DB optional"]
        async fn real_astap_solves_an_archived_star_field_hinted_and_blind() {
            let astap = std::env::var("ACHERNAR_ASTAP").expect("ACHERNAR_ASTAP");
            let fixture = std::env::var("ACHERNAR_SOLVE_FIXTURE").expect("ACHERNAR_SOLVE_FIXTURE");
            let bytes = std::fs::read(&fixture).unwrap();
            let truth = PlateSolution::from_wcs(header_wcs(&bytes), None).unwrap();

            let config = PlateSolveConfig {
                astap_path: Some(astap),
                database_dir: std::env::var("ACHERNAR_ASTAP_DB").ok(),
                timeout_s: 600,
                ..PlateSolveConfig::default()
            };
            let solver = PlateSolver::from_config(&config);
            let status = solver.status();
            assert!(status.available, "{:?}", status.unavailable_reason);
            let directory = scratch();
            let (state, frame_id) = archive_with_frame(&directory, solver, bytes, 1, None).await;

            // Hinted the way a mount would be: half a degree out, scale known.
            let hinted = SolveFrameRequest {
                frame_id: frame_id.clone(),
                hint: Some(HintRequest {
                    ra_hours: (truth.ra_hours + 0.5 / 15.0).rem_euclid(24.0),
                    dec_deg: (truth.dec_deg - 0.5).clamp(-90.0, 90.0),
                    search_radius_deg: Some(5.0),
                }),
                pixel_scale_arcsec: Some(truth.pixel_scale_arcsec),
                ignore_rig_optics: true,
            };
            let Json(response) = solve_frame(State(state.clone()), Json(hinted))
                .await
                .unwrap();
            assert!(response.success, "{:?}", response.failure);
            assert_eq!(response.mode, "hinted");
            eprintln!("hinted: {} ms", response.solve_time_ms);
            assert_matches_truth(response.solution.as_ref().unwrap(), &truth, "hinted");

            // Blind: no position and no scale.
            let blind = SolveFrameRequest {
                frame_id: frame_id.clone(),
                hint: None,
                pixel_scale_arcsec: None,
                ignore_rig_optics: true,
            };
            let Json(response) = solve_frame(State(state.clone()), Json(blind))
                .await
                .unwrap();
            assert!(response.success, "{:?}", response.failure);
            assert_eq!(response.mode, "blind");
            eprintln!("blind: {} ms", response.solve_time_ms);
            assert_matches_truth(response.solution.as_ref().unwrap(), &truth, "blind");

            let (solved,): (i64,) = sqlx::query_as(
                "SELECT COUNT(*) FROM plate_solve_events WHERE frame_id = ? AND success = 1",
            )
            .bind(&frame_id)
            .fetch_one(state.local_store.pool())
            .await
            .unwrap();
            assert_eq!(solved, 2);

            close(state, directory).await;
        }
    }
}

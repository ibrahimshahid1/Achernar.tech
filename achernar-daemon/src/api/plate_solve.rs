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

use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};

use super::api_error::ApiError;
use crate::config::RigOptics;
use crate::mount::plate_solve_trigger::TriggerStatus;
use crate::processing::plate_solver::{
    PlateSolution, PointingHint, SolveError, SolveFailure, SolveRequest, SolverStatus,
};
use crate::state::AppState;
use crate::storage::plate_solve_repo::{self, PlateSolveRecord};

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
}

#[derive(Debug, Serialize)]
pub struct PlateSolveStatus {
    #[serde(flatten)]
    pub solver: SolverStatus,
    pub auto_solve: TriggerStatus,
}

/// GET /api/v1/platesolve/status
pub async fn status(State(state): State<AppState>) -> Json<PlateSolveStatus> {
    Json(PlateSolveStatus {
        solver: state.plate_solver.status(),
        auto_solve: state.plate_solve_trigger.lock().unwrap().status(),
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
    })
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
        use crate::storage::capture_store::{
            AssetRole, CaptureStore, PendingAsset, PendingCapture, TemperatureState,
        };

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
            capture_store.clone(),
            "solve-rig".into(),
            directory.join("data"),
            rig_optics,
        )
        .with_plate_solver(solver);

        let frame_id = uuid::Uuid::new_v4().to_string();
        capture_store
            .persist_capture(
                state.local_store.pool(),
                PendingCapture {
                    capture_id: format!("capture-{frame_id}"),
                    frame_id: frame_id.clone(),
                    session_id: None,
                    rig_id: "solve-rig".into(),
                    camera_name: Some("fixture".into()),
                    driver_name: Some("test".into()),
                    capture_type: "light".into(),
                    exposure_started_at: 1_792_029_600.0,
                    exposure_completed_at: 1_792_029_610.0,
                    requested_exposure_s: 10.0,
                    requested_gain: Some(100.0),
                    applied_exposure_s: 10.0,
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
        (state, frame_id)
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

//! The center loop (§3.2 M-n; WP1B-4). NINA's Center, reimplemented against
//! the equipment layer:
//!
//! 1. Slew to the target (J2000).
//! 2. Capture with the plate-solve exposure settings.
//! 3. Solve.
//! 4. Measure the error against the target. Both are J2000; the correction
//!    is sent in J2000 and converted to the mount's native frame by its
//!    worker (D10), so the loop never compares coordinates of different
//!    epochs.
//! 5. Correct: with `sync_mode = "mount"` sync the mount to the solved
//!    position, with `sync_mode = "offset"` record the solved offset (the
//!    mount's own model is left alone), then re-slew.
//! 6. Repeat until the error is within `tolerance_arcmin` or `max_attempts`
//!    solves have been made.
//!
//! Every solve writes a `pointing_samples` row with purpose `center` (H12),
//! through the same path as any other solve. The outcome is
//! `centered{error_arcmin, attempts}` or `center_failed`. The mount and camera
//! domains are owned for the whole loop by one gate permit, and every slew
//! and sync is a child step of it ([`CommandGate::submit_step`]): each
//! step's own safety checks run, PHD2 is stopped before each slew (D20), and
//! an abort or e-stop stops the loop's later motion. The busy set shows
//! `centering` while it runs.
//!
//! [`CommandGate::submit_step`]: crate::equipment::gate::CommandGate::submit_step

// `start` is the body of `POST mount/center` (`api/mount.rs::center_value`,
// spine-owned): wired by the WP1B-4 change note for that file. Until it is
// applied only tests call the loop.
#![allow(dead_code)]

use std::future::Future;
use std::pin::Pin;

use chrono::Utc;
use serde::Serialize;
use serde_json::{json, Value};

use crate::api::api_error::ApiError;
use crate::api::plate_solve::{solve_and_record, HintRequest, PointingOutcome, SolveFrameRequest};
use crate::config::MountCenterConfig;
use crate::equipment::busy::BusyFlag;
use crate::equipment::commands::{GatedCommand, GatedOp, MountCmd};
use crate::equipment::events::{EquipmentEvent, EventKind};
use crate::equipment::gate::StepOwner;
use crate::equipment::handle::{MountHandle, Permit};
use crate::equipment::hub::EquipmentHub;
use crate::equipment::types::{
    AutonomyMode, CommandId, CommandSource, DeviceError, DeviceKind, EqCoord, EqFrame,
    MountSyncMode, OpOutcome, Priority,
};
use crate::state::AppState;

/// Exposure used when neither the request nor an earlier solved frame says.
pub const DEFAULT_EXPOSURE_S: f64 = 5.0;

/// How long a center frame waits for a solve already in flight (an
/// automatic or manual one) beyond the solver's own timeout.
const SOLVER_BUSY_MARGIN_S: u64 = 30;

/// `[mount.center]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CenterSettings {
    pub tolerance_arcmin: f64,
    pub max_attempts: u32,
}

impl CenterSettings {
    pub fn from_config(config: &MountCenterConfig) -> Self {
        Self {
            tolerance_arcmin: config.tolerance_arcmin,
            max_attempts: config.max_attempts,
        }
    }

    /// Request overrides on top of the configuration; non-positive or
    /// non-finite values are ignored.
    pub fn with_overrides(self, tolerance_arcmin: Option<f64>, max_attempts: Option<u32>) -> Self {
        Self {
            tolerance_arcmin: tolerance_arcmin
                .filter(|t| t.is_finite() && *t > 0.0)
                .unwrap_or(self.tolerance_arcmin),
            max_attempts: max_attempts
                .filter(|n| *n > 0)
                .unwrap_or(self.max_attempts)
                .max(1),
        }
    }
}

/// The exposure of each center frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CenterExposure {
    pub exposure_s: f64,
    pub gain: Option<u32>,
    pub binning: Option<u32>,
    /// `request`, `last_solved_frame` or `default`.
    pub source: &'static str,
}

/// What a request may set.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CenterOverrides {
    pub tolerance_arcmin: Option<f64>,
    pub max_attempts: Option<u32>,
    pub exposure_s: Option<f64>,
    pub gain: Option<u32>,
    pub binning: Option<u32>,
}

/// One solved center frame.
#[derive(Debug, Clone, PartialEq)]
pub struct CenterSolve {
    pub solved_j2000: EqCoord,
    pub frame_id: Option<String>,
    pub event_id: Option<String>,
    pub pointing: Option<PointingOutcome>,
}

/// Why a center frame produced no position.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum CenterImagingError {
    /// The capture failed, or its frame or file cannot be solved.
    Capture(String),
    /// The solver ran and found nothing (cloud, defocus): another attempt
    /// may succeed.
    NoSolution(String),
    /// Another solve held the solver for longer than this frame waited:
    /// another attempt may find it free.
    SolverBusy(String),
    /// The solver could not be asked; no further attempt can succeed.
    SolverUnavailable(String),
}

impl CenterImagingError {
    fn retryable(&self) -> bool {
        matches!(
            self,
            CenterImagingError::NoSolution(_) | CenterImagingError::SolverBusy(_)
        )
    }

    /// A refused `solve_and_record`, by what refused it.
    fn from_solve_error(error: &ApiError) -> Self {
        let detail = format!("{}: {}", error.code, error.message);
        match error.code {
            "plate_solver_busy" => CenterImagingError::SolverBusy(detail),
            "frame_not_found"
            | "no_solvable_asset"
            | "asset_unreadable"
            | "asset_checksum_mismatch" => CenterImagingError::Capture(detail),
            _ => CenterImagingError::SolverUnavailable(detail),
        }
    }
}

/// Capture and solve one center frame near `hint` (J2000).
pub trait CenterImaging: Send + Sync {
    fn capture_and_solve<'a>(
        &'a self,
        hint: EqCoord,
        attempt: u32,
    ) -> Pin<Box<dyn Future<Output = Result<CenterSolve, CenterImagingError>> + Send + 'a>>;
}

/// The daemon's camera and ASTAP: a `center` capture through the capture
/// path, then a hinted solve recorded with trigger `center`.
///
/// The center permit owns the camera for the whole loop, so each frame is
/// a child step of it (O11 b): `owner` is that permit and `source` the
/// loop's own source (the operator or the relay). A plain capture would
/// claim the camera again and be refused with 409.
pub struct DaemonImaging {
    pub state: AppState,
    pub exposure: CenterExposure,
    pub owner: StepOwner,
    pub source: CommandSource,
}

impl DaemonImaging {
    /// The imaging of `job`: its frames are steps of its permit.
    pub fn for_job(state: &AppState, job: &CenterJob, exposure: CenterExposure) -> Self {
        Self {
            state: state.clone(),
            exposure,
            owner: StepOwner::of(&job.permit),
            source: job.command.source.clone(),
        }
    }
}

impl CenterImaging for DaemonImaging {
    fn capture_and_solve<'a>(
        &'a self,
        hint: EqCoord,
        _attempt: u32,
    ) -> Pin<Box<dyn Future<Output = Result<CenterSolve, CenterImagingError>> + Send + 'a>> {
        Box::pin(async move {
            if !self.state.plate_solver.is_available() {
                return Err(CenterImagingError::SolverUnavailable(
                    self.state
                        .plate_solver
                        .status()
                        .unavailable_reason
                        .unwrap_or_else(|| "no plate solver is configured".into()),
                ));
            }
            let params = crate::api::camera::CaptureParams {
                exposure: self.exposure.exposure_s,
                gain: self.exposure.gain,
                offset: None,
                binning: self.exposure.binning,
                filter: None,
                temperature: None,
                session_id: None,
                capture_type: Some("center".into()),
                retention_policy: Default::default(),
                retention_acknowledgement: None,
                source: Default::default(),
            };
            let origin =
                crate::api::camera::CaptureOrigin::step_of(self.owner.clone(), self.source.clone());
            let image = crate::api::camera::perform_capture_as(self.state.clone(), params, origin)
                .await
                .map_err(|e| CenterImagingError::Capture(format!("{}: {}", e.code, e.message)))?;
            let request = SolveFrameRequest {
                frame_id: image.id.clone(),
                hint: Some(HintRequest {
                    ra_hours: hint.ra_h,
                    dec_deg: hint.dec_deg,
                    search_radius_deg: None,
                }),
                pixel_scale_arcsec: None,
                ignore_rig_optics: false,
            };
            let trigger = crate::mount::plate_solve_trigger::SolveReason::Center.as_str();
            // One solve runs at a time: an automatic or manual solve begun
            // just before Center holds the solver for up to its timeout.
            let busy_wait = std::time::Duration::from_secs(
                self.state
                    .plate_solver
                    .status()
                    .timeout_s
                    .saturating_add(SOLVER_BUSY_MARGIN_S),
            );
            let deadline = tokio::time::Instant::now() + busy_wait;
            let response = loop {
                match solve_and_record(&self.state, request.clone(), trigger).await {
                    Ok(response) => break response,
                    Err(error)
                        if error.code == "plate_solver_busy"
                            && tokio::time::Instant::now() < deadline =>
                    {
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    }
                    Err(error) => return Err(CenterImagingError::from_solve_error(&error)),
                }
            };
            match response.solution {
                Some(solution) => Ok(CenterSolve {
                    solved_j2000: EqCoord::new(solution.ra_hours, solution.dec_deg, EqFrame::J2000),
                    frame_id: Some(response.frame_id),
                    event_id: response.event_id,
                    pointing: response.pointing,
                }),
                None => Err(CenterImagingError::NoSolution(
                    response
                        .failure
                        .map(|f| format!("{}: {}", f.kind.as_str(), f.detail))
                        .unwrap_or_else(|| "no solution".into()),
                )),
            }
        })
    }
}

/// One solve of the loop.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CenterAttempt {
    pub attempt: u32,
    pub solved_j2000: Option<EqCoord>,
    pub error_arcmin: Option<f64>,
    pub frame_id: Option<String>,
    pub plate_solve_event_id: Option<String>,
    pub pointing_sample_id: Option<uuid::Uuid>,
    /// `sync` or `offset`, when a correction followed.
    pub correction: Option<&'static str>,
    pub failure: Option<CenterImagingError>,
}

/// How the loop ended.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CenterOutcome {
    Centered {
        error_arcmin: f64,
        attempts: u32,
    },
    Failed {
        error_arcmin: Option<f64>,
        attempts: u32,
        reason: String,
    },
}

/// The whole run, for logs and tests.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CenterReport {
    pub command_id: CommandId,
    pub target: EqCoord,
    pub settings: CenterSettings,
    pub outcome: CenterOutcome,
    pub attempts: Vec<CenterAttempt>,
}

/// An authorised center loop, not yet running.
pub struct CenterJob {
    pub command: GatedCommand,
    pub permit: Permit,
    pub mount: MountHandle,
    /// J2000.
    pub target: EqCoord,
    pub settings: CenterSettings,
    /// What a correction does: the mount's worker applies `[mount]
    /// sync_mode` as it was at connect.
    pub sync_mode: MountSyncMode,
}

fn conflict(code: &'static str, message: impl Into<String>) -> ApiError {
    ApiError::new(axum::http::StatusCode::CONFLICT, code, message)
}

/// The target in J2000: a J2000 target as given; a JNow or mount-frame
/// target converted from the mount's native frame (D10). An unknown native
/// frame refuses a mount-frame target (M-a).
pub fn target_j2000(
    state: &AppState,
    mount: &MountHandle,
    target: EqCoord,
) -> Result<EqCoord, ApiError> {
    crate::equipment::mount::validate_coord(&target)
        .map_err(|err| crate::api::equipment::device_error(DeviceKind::Mount, &err))?;
    let target = target.normalized();
    let frame = match target.frame {
        EqFrame::J2000 => return Ok(target),
        EqFrame::Unknown => mount.snapshot().capabilities.native_frame,
        other => other,
    };
    if frame == EqFrame::Unknown {
        return Err(conflict(
            "mount_frame_unknown",
            "the mount's native frame is unknown, so a mount-frame target cannot be compared \
             with a J2000 solve; send the target as j2000 or set [mount] \
             equatorial_system_override",
        ));
    }
    let site = state
        .equipment_ctx
        .mount_shared()
        .and_then(|shared| shared.geometry_site())
        .or_else(|| state.equipment.site());
    let converted = crate::equipment::astro::frame_converter().convert(
        EqCoord { frame, ..target },
        EqFrame::J2000,
        Utc::now(),
        site.as_ref(),
    );
    Ok(EqCoord {
        frame: EqFrame::J2000,
        ..converted
    }
    .normalized())
}

/// Authorise a center loop: the mount and camera domains, through the gate
/// (ownership, arbiter, safety checks, audit). Nothing moves yet.
pub fn authorize(
    state: &AppState,
    target: EqCoord,
    overrides: CenterOverrides,
    source: CommandSource,
) -> Result<CenterJob, ApiError> {
    let mount = crate::api::equipment::mount_handle(state)?;
    let target = target_j2000(state, &mount, target)?;
    let settings = CenterSettings::from_config(&state.equipment_ctx.configs.mount.center)
        .with_overrides(overrides.tolerance_arcmin, overrides.max_attempts);
    let command = GatedCommand::new(
        GatedOp::Center {
            target,
            tolerance_arcmin: Some(settings.tolerance_arcmin),
            max_attempts: Some(settings.max_attempts),
        },
        source,
        AutonomyMode::Manual,
        Priority::Exposure,
    );
    let permit = state
        .equipment
        .gate()
        .authorize(&command)
        .map_err(|err| crate::api::equipment::submit_error(DeviceKind::Mount, &err.into()))?;
    let sync_mode = state
        .equipment_ctx
        .mount_shared()
        .map(|shared| shared.settings().sync_mode)
        .unwrap_or(state.equipment_ctx.configs.mount.sync_mode);
    Ok(CenterJob {
        command,
        permit,
        mount,
        target,
        settings,
        sync_mode,
    })
}

/// The exposure of each center frame: the request's, else the settings of
/// the newest frame that solved, else [`DEFAULT_EXPOSURE_S`].
pub async fn exposure(state: &AppState, overrides: &CenterOverrides) -> CenterExposure {
    if let Some(exposure_s) = overrides.exposure_s.filter(|s| s.is_finite() && *s > 0.0) {
        return CenterExposure {
            exposure_s,
            gain: overrides.gain,
            binning: overrides.binning,
            source: "request",
        };
    }
    let solved =
        crate::storage::plate_solve_repo::latest_successful_solve(state.local_store.pool())
            .await
            .ok()
            .flatten()
            .and_then(|solve| solve.exposure);
    match solved {
        Some(solved) => CenterExposure {
            exposure_s: solved.exposure_s,
            gain: overrides.gain.or_else(|| {
                solved
                    .gain
                    .filter(|g| g.is_finite() && *g >= 0.0)
                    .map(|g| g.round() as u32)
            }),
            binning: overrides
                .binning
                .or_else(|| solved.binning.and_then(|b| u32::try_from(b).ok())),
            source: "last_solved_frame",
        },
        None => CenterExposure {
            exposure_s: DEFAULT_EXPOSURE_S,
            gain: overrides.gain,
            binning: overrides.binning,
            source: "default",
        },
    }
}

/// M-4 in shadow: what the pointing model of this mount, connection, sync
/// generation and pier side would correct for `target`. Reported beside
/// the first solve's measured error so the model can be evaluated; never
/// applied (see `pointing_model`'s module docs).
pub async fn m4_shadow(state: &AppState, mount: &MountHandle, target: EqCoord) -> Value {
    use crate::equipment::types::PierSide;
    use crate::mount::pointing_model::{connection_segment_start, ModelKey, PointingModel};
    let snapshot = mount.snapshot();
    let pier_side = snapshot
        .state
        .side_of_pier
        .known()
        .copied()
        .unwrap_or(PierSide::Unknown);
    let native_frame = snapshot.capabilities.native_frame;
    if native_frame == EqFrame::Unknown {
        return json!({"applied": false, "refused": {"reason": "mount_frame_unknown"}});
    }
    // The generation counter restarts with every connection: only this
    // connection's samples describe the mount's model now.
    let Some(segment_start) = connection_segment_start(mount.connection_id()) else {
        return json!({
            "applied": false,
            "refused": {"reason": "no_samples_on_this_connection"},
        });
    };
    let key = ModelKey {
        mount_identity_key: mount.identity().identity_key.clone(),
        sync_generation: snapshot.sync_generation,
        pier_side,
        mount_frame: native_frame,
        jnow_flavour: state.equipment_ctx.configs.mount.jnow_flavour,
        segment_start,
    };
    let records = match crate::storage::pointing_samples_repo::for_mount(
        state.local_store.pool(),
        &key.mount_identity_key,
        Some(key.sync_generation),
        500,
    )
    .await
    {
        Ok(records) => records,
        Err(error) => return json!({"applied": false, "error": error.to_string()}),
    };
    let model = PointingModel::from_records(key, &records);
    let Some(lst) = snapshot.state.sidereal_time_h.known().copied() else {
        return json!({"applied": false, "refused": {"reason": "lst_unknown"}});
    };
    // The samples' hour angle is LST minus the mount's native RA (H12), so
    // the target's is taken in the native frame too.
    let native = if native_frame == EqFrame::J2000 {
        target
    } else {
        let site = state
            .equipment_ctx
            .mount_shared()
            .and_then(|shared| shared.geometry_site())
            .or_else(|| state.equipment.site());
        crate::equipment::astro::frame_converter().convert(
            target,
            native_frame,
            Utc::now(),
            site.as_ref(),
        )
    };
    let ha_h = (lst - native.ra_h + 12.0).rem_euclid(24.0) - 12.0;
    match model.correct(target, ha_h, pier_side) {
        Ok(correction) => json!({"applied": false, "correction": correction}),
        Err(refused) => json!({"applied": false, "refused": refused}),
    }
}

/// `POST mount/center`: authorise, start the loop, return on accept with
/// `{status: "centering", command_id, ...}`. Completion arrives as the
/// `centered` or `center_failed` event.
pub async fn start(
    state: &AppState,
    target: EqCoord,
    overrides: CenterOverrides,
    source: CommandSource,
) -> Result<Value, ApiError> {
    let job = authorize(state, target, overrides, source)?;
    let exposure = exposure(state, &overrides).await;
    let m4 = m4_shadow(state, &job.mount, job.target).await;
    let body = json!({
        "status": "centering",
        "command_id": job.command.command_id,
        "target": job.target,
        "tolerance_arcmin": job.settings.tolerance_arcmin,
        "max_attempts": job.settings.max_attempts,
        "exposure": exposure,
        "m4_shadow": m4,
    });
    let hub = std::sync::Arc::clone(&state.equipment);
    let imaging = DaemonImaging::for_job(state, &job, exposure);
    tokio::spawn(async move {
        let report = run(&hub, &imaging, job).await;
        tracing::info!(
            "Center {} finished: {}",
            report.command_id,
            serde_json::to_string(&report.outcome).unwrap_or_default()
        );
    });
    Ok(body)
}

/// One slew or sync step of the loop: a child of the center permit,
/// awaited to its outcome. The permit is only read while the step is
/// authorised, so the returned future holds no reference to it.
fn step<'a>(
    hub: &'a EquipmentHub,
    job: &'a CenterJob,
    cmd: MountCmd,
) -> impl Future<Output = Result<(), String>> + Send + 'a {
    let step = GatedCommand::new(
        GatedOp::Mount(cmd),
        job.command.source.clone(),
        job.command.mode,
        job.command.priority,
    );
    let what = format!("{:?}", step.kind());
    let submit = hub.gate().submit_step(&job.mount, &job.permit, step);
    async move {
        let submitted = submit
            .await
            .map_err(|err| format!("{what} refused: {err}"))?;
        match submitted.ticket.wait().await {
            OpOutcome::Succeeded => Ok(()),
            other => Err(format!("{what} ended {}: {other:?}", other.label())),
        }
    }
}

/// Run an authorised loop to its end (see the module docs).
pub async fn run(hub: &EquipmentHub, imaging: &dyn CenterImaging, job: CenterJob) -> CenterReport {
    let _busy = hub.busy_flags().claim(BusyFlag::Centering);
    let gate = hub.gate();
    gate.record_executed(
        &job.command,
        vec![format!(
            "tolerance_arcmin={} max_attempts={}",
            job.settings.tolerance_arcmin, job.settings.max_attempts
        )],
    );
    let sync_label = match job.sync_mode {
        MountSyncMode::Mount => "sync",
        MountSyncMode::Offset => "offset",
    };
    let mut attempts: Vec<CenterAttempt> = Vec::new();
    let outcome = 'run: {
        if let Err(reason) = step(hub, &job, MountCmd::Slew { target: job.target }).await {
            break 'run CenterOutcome::Failed {
                error_arcmin: None,
                attempts: 0,
                reason,
            };
        }
        let mut last_error = None;
        for attempt in 1..=job.settings.max_attempts {
            let solve = imaging.capture_and_solve(job.target, attempt).await;
            let mut record = CenterAttempt {
                attempt,
                solved_j2000: None,
                error_arcmin: None,
                frame_id: None,
                plate_solve_event_id: None,
                pointing_sample_id: None,
                correction: None,
                failure: None,
            };
            let solved = match solve {
                Ok(solve) => solve,
                Err(failure) => {
                    let retryable = failure.retryable();
                    let reason = format!("attempt {attempt}: {failure:?}");
                    record.failure = Some(failure);
                    attempts.push(record);
                    if retryable && attempt < job.settings.max_attempts {
                        continue;
                    }
                    break 'run CenterOutcome::Failed {
                        error_arcmin: last_error,
                        attempts: attempt,
                        reason,
                    };
                }
            };
            let error = solved
                .solved_j2000
                .separation_arcmin(&job.target)
                .filter(|e| e.is_finite());
            record.solved_j2000 = Some(solved.solved_j2000);
            record.error_arcmin = error;
            record.frame_id = solved.frame_id.clone();
            record.plate_solve_event_id = solved.event_id.clone();
            record.pointing_sample_id = solved.pointing.as_ref().and_then(|p| p.sample_id);
            last_error = error.or(last_error);
            let Some(error) = error else {
                attempts.push(record);
                break 'run CenterOutcome::Failed {
                    error_arcmin: None,
                    attempts: attempt,
                    reason: "the solved position is not a J2000 coordinate".into(),
                };
            };
            if error <= job.settings.tolerance_arcmin {
                attempts.push(record);
                break 'run CenterOutcome::Centered {
                    error_arcmin: error,
                    attempts: attempt,
                };
            }
            if attempt == job.settings.max_attempts {
                attempts.push(record);
                break 'run CenterOutcome::Failed {
                    error_arcmin: Some(error),
                    attempts: attempt,
                    reason: format!(
                        "{error:.2}' from the target after {attempt} attempts (tolerance {}')",
                        job.settings.tolerance_arcmin
                    ),
                };
            }
            record.correction = Some(sync_label);
            attempts.push(record);
            if let Err(reason) = step(hub, &job, MountCmd::Sync(solved.solved_j2000)).await {
                break 'run CenterOutcome::Failed {
                    error_arcmin: Some(error),
                    attempts: attempt,
                    reason,
                };
            }
            if let Err(reason) = step(hub, &job, MountCmd::Slew { target: job.target }).await {
                break 'run CenterOutcome::Failed {
                    error_arcmin: Some(error),
                    attempts: attempt,
                    reason,
                };
            }
        }
        CenterOutcome::Failed {
            error_arcmin: last_error,
            attempts: job.settings.max_attempts,
            reason: "no attempt solved".into(),
        }
    };

    let event = match &outcome {
        CenterOutcome::Centered {
            error_arcmin,
            attempts,
        } => EventKind::Centered {
            error_arcmin: *error_arcmin,
            attempts: *attempts,
        },
        CenterOutcome::Failed {
            error_arcmin,
            attempts,
            reason,
        } => EventKind::CenterFailed {
            error_arcmin: *error_arcmin,
            attempts: *attempts,
            reason: reason.clone(),
        },
    };
    hub.emit(
        EquipmentEvent::new(event, hub.clock().as_ref())
            .for_device(DeviceKind::Mount, job.mount.connection_id())
            .with_command(job.command.command_id),
    );
    gate.record_outcome(
        &job.command,
        &match &outcome {
            CenterOutcome::Centered { .. } => OpOutcome::Succeeded,
            CenterOutcome::Failed { reason, .. } => {
                OpOutcome::Failed(DeviceError::invalid_operation(reason.clone()))
            }
        },
    );
    CenterReport {
        command_id: job.command.command_id,
        target: job.target,
        settings: job.settings,
        outcome,
        attempts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::plate_solve::{record_pointing_sample, PointingInput};
    use crate::config::AchernarConfig;
    use crate::equipment::events::PointingPurpose;
    use crate::equipment::health_bridge::HealthPolicy;
    use crate::equipment::sim::mount::{SimMountControl, SimMountProfile};
    use crate::equipment::spine::{SpineContext, WorkerTuning};
    use crate::equipment::traits::{EquipmentReader, FrameConverter};
    use crate::equipment::types::{MountSyncMode, Site};
    use crate::equipment::worker::Cadence;
    use crate::mount::pointing_model::gnomonic_inverse;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    /// A daemon state with fast sim cadences and `sync_mode`, like the
    /// spine's route tests.
    async fn state(sync_mode: MountSyncMode) -> (std::path::PathBuf, AppState) {
        let (directory, mut state) = crate::api::equipment_tests::state().await;
        let mut configs = AchernarConfig::default().equipment_configs();
        configs.mount.settle_s = 0.1;
        configs.mount.sync_mode = sync_mode;
        configs.mount.center.max_attempts = 3;
        let ctx = SpineContext::new(&state.equipment, configs, None, None);
        let mut profile = SimMountProfile::esp32_like();
        profile.slew_rate_deg_s = 60.0;
        ctx.set_tuning(WorkerTuning {
            cadence: Some(Cadence {
                idle: Duration::from_millis(100),
                active: Duration::from_millis(50),
            }),
            policy: Some(HealthPolicy::default()),
            sim_mount_profile: Some(profile),
        });
        state.equipment_ctx = Arc::new(ctx);
        let _connected = crate::api::connect_equipment(
            axum::extract::State(state.clone()),
            axum::Json(crate::api::ConnectRequest {
                equipment_type: Some("mount".into()),
                driver: "sim".into(),
                option: None,
                index: None,
            }),
        )
        .await
        .unwrap_or_else(|err| panic!("connect: {}: {}", err.code, err.message));
        (directory, state)
    }

    /// A solver whose answer is the sim's reported position minus a pointing
    /// error. A mount sync re-labels the sim's coordinates, after which only
    /// `residual` of the error is left; an offset leaves the error but moves
    /// the commanded target.
    struct ScriptedImaging {
        state: AppState,
        sim: SimMountControl,
        site: Option<Site>,
        /// Mount reading minus truth, arcsec (xi east, eta north), native.
        error: Mutex<(f64, f64)>,
        residual: f64,
        syncs_seen: Mutex<usize>,
        converter: Arc<dyn FrameConverter>,
    }

    impl ScriptedImaging {
        fn new(state: &AppState, error: (f64, f64), residual: f64) -> Self {
            Self {
                sim: state.equipment_ctx.sims().mount.unwrap(),
                site: state
                    .equipment_ctx
                    .mount_shared()
                    .and_then(|shared| shared.geometry_site()),
                state: state.clone(),
                error: Mutex::new(error),
                residual,
                syncs_seen: Mutex::new(0),
                converter: crate::equipment::astro::frame_converter(),
            }
        }
    }

    impl CenterImaging for ScriptedImaging {
        fn capture_and_solve<'a>(
            &'a self,
            _hint: EqCoord,
            _attempt: u32,
        ) -> Pin<Box<dyn Future<Output = Result<CenterSolve, CenterImagingError>> + Send + 'a>>
        {
            Box::pin(async move {
                let start = Utc::now();
                tokio::time::sleep(Duration::from_millis(250)).await;
                let end = Utc::now();
                let syncs = self.sim.calls().iter().filter(|c| *c == "sync").count();
                let (xi, eta) = {
                    let mut seen = self.syncs_seen.lock().unwrap();
                    let mut error = self.error.lock().unwrap();
                    for _ in *seen..syncs {
                        error.0 *= self.residual;
                        error.1 *= self.residual;
                    }
                    *seen = syncs;
                    *error
                };
                let (ra, dec) = self.sim.position();
                let reported = EqCoord::new(ra, dec, EqFrame::JNowApparent);
                let truth_native = gnomonic_inverse(reported, -xi, -eta);
                let truth =
                    self.converter
                        .convert(truth_native, EqFrame::J2000, end, self.site.as_ref());
                let truth = EqCoord {
                    frame: EqFrame::J2000,
                    ..truth
                };
                // Let a mount sample after the exposure end reach the ring.
                tokio::time::sleep(Duration::from_millis(250)).await;
                let pointing = record_pointing_sample(
                    &self.state,
                    PointingInput {
                        plate_solve_event_id: None,
                        frame_id: None,
                        exposure_start: start,
                        exposure_end: end,
                        solved_j2000: truth,
                        purpose: PointingPurpose::Center,
                    },
                )
                .await;
                Ok(CenterSolve {
                    solved_j2000: truth,
                    frame_id: None,
                    event_id: None,
                    pointing: Some(pointing),
                })
            })
        }
    }

    /// A J2000 target offset from where the mount points, once its worker
    /// has published a converted position.
    async fn near(state: &AppState, d_ra_h: f64, d_dec: f64) -> EqCoord {
        let handle = crate::api::equipment::mount_handle(state).unwrap();
        for _ in 0..200 {
            if let Some(here) = handle.snapshot().j2000 {
                return EqCoord::new(here.ra_h + d_ra_h, here.dec_deg + d_dec, EqFrame::J2000);
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("the mount never published a J2000 position");
    }

    fn operator() -> CommandSource {
        CommandSource::Operator { overriding: false }
    }

    fn center_events(state: &AppState) -> Vec<String> {
        state
            .equipment
            .events_since(0, &["centered".into(), "center_failed".into()], 50)
            .iter()
            .map(|e| e.name().to_string())
            .collect()
    }

    async fn run_job(state: &AppState, imaging: &ScriptedImaging, target: EqCoord) -> CenterReport {
        let job = authorize(state, target, CenterOverrides::default(), operator()).unwrap();
        let report = run(&state.equipment, imaging, job).await;
        // The permit was released with the job.
        assert!(state.equipment.gate().is_idle());
        report
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_center_loop_converges_in_the_sim_by_syncing_the_mount() {
        let (directory, state) = state(MountSyncMode::Mount).await;
        let imaging = ScriptedImaging::new(&state, (420.0, -300.0), 0.05);
        let target = near(&state, 0.15, 1.5).await;
        let report = run_job(&state, &imaging, target).await;
        match &report.outcome {
            CenterOutcome::Centered {
                error_arcmin,
                attempts,
            } => {
                assert_eq!(*attempts, 2, "{report:#?}");
                assert!(*error_arcmin <= 1.0);
            }
            other => panic!("{other:?}: {report:#?}"),
        }
        assert!(
            (report.attempts[0].error_arcmin.unwrap() - (420f64.hypot(300.0) / 60.0)).abs() < 0.2
        );
        assert_eq!(report.attempts[0].correction, Some("sync"));
        assert_eq!(center_events(&state), vec!["centered".to_string()]);
        // Every solve wrote a pointing sample, and the sync started a new
        // generation between them.
        let samples: Vec<(i64, String)> =
            sqlx::query_as("SELECT sync_generation, purpose FROM pointing_samples ORDER BY rowid")
                .fetch_all(state.local_store.pool())
                .await
                .unwrap();
        assert_eq!(samples, vec![(0, "center".into()), (1, "center".into())]);
        crate::api::equipment_tests::cleanup(directory, state).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_center_loop_converges_with_a_solved_offset_and_never_syncs_the_mount() {
        let (directory, state) = state(MountSyncMode::Offset).await;
        let imaging = ScriptedImaging::new(&state, (-360.0, 240.0), 1.0);
        let target = near(&state, -0.1, 1.0).await;
        let report = run_job(&state, &imaging, target).await;
        assert!(
            matches!(report.outcome, CenterOutcome::Centered { attempts: 2, .. }),
            "{report:#?}"
        );
        assert_eq!(report.attempts[0].correction, Some("offset"));
        assert!(
            !imaging.sim.calls().iter().any(|c| c == "sync"),
            "offset mode leaves the mount's own model alone"
        );
        let offsets = state
            .equipment_ctx
            .mount_shared()
            .unwrap()
            .with_offsets(|offsets| offsets.entries().to_vec());
        assert_eq!(offsets.len(), 1);
        // The recorded offset is the scripted pointing error (mount minus
        // truth): +240" in Dec.
        assert!(
            (offsets[0].d_dec_deg - 240.0 / 3600.0).abs() < 0.005,
            "{:?}",
            offsets[0]
        );
        // A sample of the same mount and generation number from an earlier
        // connection (whose counter also started at 0) is another model's.
        let pool = state.local_store.pool();
        let handle = crate::api::equipment::mount_handle(&state).unwrap();
        let stored = crate::storage::pointing_samples_repo::for_mount(
            pool,
            &handle.identity().identity_key,
            Some(0),
            10,
        )
        .await
        .unwrap();
        assert_eq!(stored.len(), 2);
        let mut earlier = stored[0].clone();
        earlier.id = uuid::Uuid::new_v4();
        earlier.mid_exposure_utc = Some(Utc::now() - chrono::Duration::hours(2));
        crate::storage::pointing_samples_repo::upsert(pool, &earlier)
            .await
            .unwrap();
        // M-4 in shadow: two samples of this connection and generation are
        // too few to predict from, and nothing is ever applied.
        let shadow = m4_shadow(&state, &handle, target).await;
        assert_eq!(shadow["applied"], json!(false));
        assert_eq!(
            shadow["refused"]["reason"],
            json!("too_few_samples"),
            "{shadow}"
        );
        assert_eq!(shadow["refused"]["have"], json!(2), "{shadow}");
        crate::api::equipment_tests::cleanup(directory, state).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_correction_that_does_not_take_fails_after_max_attempts() {
        let (directory, state) = state(MountSyncMode::Mount).await;
        // Every sync leaves the whole error: the loop gives up.
        let imaging = ScriptedImaging::new(&state, (600.0, 0.0), 1.0);
        let target = near(&state, 0.05, 0.5).await;
        let report = run_job(&state, &imaging, target).await;
        match &report.outcome {
            CenterOutcome::Failed {
                attempts,
                error_arcmin,
                ..
            } => {
                assert_eq!(*attempts, 3);
                assert!(error_arcmin.unwrap() > 9.0);
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(report.attempts.len(), 3);
        assert_eq!(center_events(&state), vec!["center_failed".to_string()]);
        crate::api::equipment_tests::cleanup(directory, state).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_loop_owns_the_mount_and_camera_and_shows_centering() {
        let (directory, state) = state(MountSyncMode::Mount).await;
        let job = authorize(
            &state,
            near(&state, 0.02, 0.2).await,
            CenterOverrides::default(),
            operator(),
        )
        .unwrap();
        // While it is held, an operator slew without override is a conflict.
        let handle = crate::api::equipment::mount_handle(&state).unwrap();
        let conflict = crate::api::equipment::submit(
            &state,
            &handle,
            GatedOp::Mount(MountCmd::Slew {
                target: near(&state, 0.0, 0.0).await,
            }),
            operator(),
            Priority::Exposure,
        )
        .await
        .unwrap_err();
        assert_eq!(conflict.status, axum::http::StatusCode::CONFLICT);
        assert!(state
            .equipment
            .gate()
            .holder(crate::equipment::commands::Domain::Device(
                DeviceKind::Camera
            ))
            .is_some());
        let imaging = ScriptedImaging::new(&state, (30.0, 0.0), 1.0);
        let busy = state.equipment.busy();
        let run = tokio::spawn({
            let state = state.clone();
            async move { run(&state.equipment, &imaging, job).await }
        });
        let mut saw_centering = false;
        for _ in 0..100 {
            if busy.borrow().centering {
                saw_centering = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let report = run.await.unwrap();
        assert!(saw_centering);
        assert!(matches!(
            report.outcome,
            CenterOutcome::Centered { attempts: 1, .. }
        ));
        assert!(!state.equipment.busy().borrow().centering);
        crate::api::equipment_tests::cleanup(directory, state).await;
    }

    fn center_frame() -> crate::api::camera::CaptureParams {
        serde_json::from_value(json!({"exposure": 1.0, "capture_type": "center"})).unwrap()
    }

    /// The center permit owns the camera, so its own frame is a child step
    /// of it and passes the gate (no camera is connected here, so it stops
    /// at the camera lookup with 412, never 409), while a plain operator
    /// capture during the loop is still refused with 409.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_center_frame_is_a_step_of_the_center_permit_and_an_operator_capture_is_refused() {
        use crate::api::camera::{perform_capture, perform_capture_as, CaptureOrigin};
        use crate::equipment::commands::Domain;
        let (directory, state) = state(MountSyncMode::Mount).await;
        let job = authorize(
            &state,
            near(&state, 0.02, 0.2).await,
            CenterOverrides::default(),
            CommandSource::Relay { overriding: false },
        )
        .unwrap();
        let origin = CaptureOrigin::step_of(StepOwner::of(&job.permit), job.command.source.clone());
        let step = perform_capture_as(state.clone(), center_frame(), origin)
            .await
            .unwrap_err();
        assert_eq!(
            step.status,
            axum::http::StatusCode::PRECONDITION_FAILED,
            "{}: {}",
            step.code,
            step.message
        );
        assert_eq!(step.code, "camera_not_ready");
        let plain = perform_capture(state.clone(), center_frame())
            .await
            .unwrap_err();
        assert_eq!(plain.status, axum::http::StatusCode::CONFLICT);
        assert_eq!(plain.code, "ownership_conflict", "{}", plain.message);
        // The step held nothing of its own: the loop still owns the camera.
        assert_eq!(
            state
                .equipment
                .gate()
                .holder(Domain::Device(DeviceKind::Camera))
                .map(|claim| claim.command_id),
            Some(job.command.command_id)
        );
        // The step was recorded under the loop's (relay) source.
        let sources: Vec<(String,)> =
            sqlx::query_as("SELECT requested_json FROM capture_attempts ORDER BY rowid")
                .fetch_all(state.local_store.pool())
                .await
                .unwrap();
        let first: Value = serde_json::from_str(&sources[0].0).unwrap();
        assert_eq!(first["command_source"], json!("relay"));
        assert_eq!(
            first["parent_command_id"],
            json!(job.command.command_id),
            "{first}"
        );
        drop(job);
        assert!(state.equipment.gate().is_idle());
        crate::api::equipment_tests::cleanup(directory, state).await;
    }

    /// The daemon's own imaging path takes its frame under the center
    /// permit: with a solver available it reaches the camera (none here),
    /// not an ownership conflict.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_daemon_imaging_captures_under_the_center_permit() {
        use crate::config::PlateSolveConfig;
        use crate::processing::astap::test_support::{fake_astap, SOLVED_INI};
        use crate::processing::astap::AstapSolver;
        use crate::processing::plate_solver::PlateSolver;
        let (directory, state) = state(MountSyncMode::Mount).await;
        let solver = PlateSolver::with_backend_for_test(
            AstapSolver::new(
                fake_astap(&directory, Some(SOLVED_INI), 0, 0.0),
                &PlateSolveConfig::default(),
            ),
            directory.join("work"),
        );
        let state = state.with_plate_solver(solver);
        let target = near(&state, 0.02, 0.2).await;
        let job = authorize(&state, target, CenterOverrides::default(), operator()).unwrap();
        let imaging = DaemonImaging::for_job(
            &state,
            &job,
            exposure(&state, &CenterOverrides::default()).await,
        );
        let failure = imaging.capture_and_solve(target, 1).await.unwrap_err();
        match &failure {
            CenterImagingError::Capture(detail) => {
                assert!(detail.starts_with("camera_not_ready"), "{detail}");
            }
            other => panic!("{other:?}"),
        }
        drop(job);
        crate::api::equipment_tests::cleanup(directory, state).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_mount_frame_target_is_converted_and_an_invalid_one_refused() {
        let (directory, state) = state(MountSyncMode::Mount).await;
        near(&state, 0.0, 0.0).await;
        let handle = crate::api::equipment::mount_handle(&state).unwrap();
        let native = handle.snapshot();
        let (ra, dec) = (
            native.state.ra_h.value.unwrap(),
            native.state.dec_deg.value.unwrap(),
        );
        let j2000 = target_j2000(&state, &handle, EqCoord::new(ra, dec, EqFrame::Unknown)).unwrap();
        let reference = native.j2000.unwrap();
        assert_eq!(j2000.frame, EqFrame::J2000);
        assert!(j2000.separation_arcmin(&reference).unwrap() < 0.1);
        let bad = target_j2000(&state, &handle, EqCoord::new(1.0, 95.0, EqFrame::J2000));
        assert_eq!(bad.unwrap_err().status, axum::http::StatusCode::BAD_REQUEST);
        crate::api::equipment_tests::cleanup(directory, state).await;
    }

    #[test]
    fn a_busy_solver_is_retried_and_only_an_unavailable_one_is_final() {
        let error =
            |status, code| CenterImagingError::from_solve_error(&ApiError::new(status, code, "x"));
        use axum::http::StatusCode;
        let busy = error(StatusCode::CONFLICT, "plate_solver_busy");
        assert!(matches!(busy, CenterImagingError::SolverBusy(_)));
        assert!(busy.retryable());
        for code in ["frame_not_found", "no_solvable_asset", "asset_unreadable"] {
            let failure = error(StatusCode::UNPROCESSABLE_ENTITY, code);
            assert!(matches!(failure, CenterImagingError::Capture(_)), "{code}");
            assert!(!failure.retryable());
        }
        let unavailable = error(StatusCode::SERVICE_UNAVAILABLE, "plate_solver_unavailable");
        assert!(matches!(
            unavailable,
            CenterImagingError::SolverUnavailable(_)
        ));
        assert!(!unavailable.retryable());
        assert!(CenterImagingError::NoSolution("clouds".into()).retryable());
    }

    #[test]
    fn overrides_never_loosen_into_nonsense() {
        let base = CenterSettings {
            tolerance_arcmin: 1.0,
            max_attempts: 3,
        };
        assert_eq!(base.with_overrides(Some(0.5), Some(5)).max_attempts, 5);
        assert_eq!(base.with_overrides(Some(-1.0), Some(0)), base);
        assert_eq!(base.with_overrides(Some(f64::NAN), None), base);
    }
}

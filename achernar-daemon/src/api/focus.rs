//! Focus API endpoints — Section 7.5.2–7.5.4.
//! All /api/v1/focus/* endpoints.
//!
//! # The equipment layer (equipment spec WP1B-2)
//!
//! An autofocus run is one compound operation of the command gate. It is
//! authorised once (`GatedOp::Autofocus`, Autofocus priority) and owns the
//! focuser and camera domains for the whole run, plus the guider when
//! `[guider] guiding_during_autofocus` pauses or stops PHD2 and the wheel when
//! an AF filter is configured (§2.9). Every focuser command of the run is a
//! step of that operation (`CommandGate::submit_step`, O11 b): the focuser
//! safety check (travel bounds, the A06 exposure check, an unknown origin)
//! and the audit apply to each move, an abort stops the run's later moves
//! (D19), and the moves run on the focuser's worker, so no driver call
//! happens on an async thread (D3). The operation's own records are the
//! audit stages, `operation_commanded`/`operation_ended` (H18) and the
//! `autofocus_*` events.
//!
//! Hooks implemented here:
//!
//! - **H3** sample freshness (A06): [`frame_freshness`] plus the hub's
//!   quiescence over the exposure window and a focuser read-back from a poll
//!   at or after the window's end, in [`judge_frame`]. Samples record the
//!   timing basis their window came from as their clock quality.
//! - **H4** a verification frame after the move to best focus
//!   ([`verify_focus`]), recorded on the `focus_actions` row.
//! - **H5** the FDF learns only from a verified, non-simulator run, at the
//!   verification frame's time (D18), after re-baselining on any focuser
//!   position reset since its last update (F-h).
//! - **H6** the hub's ambient temperature for the FDF and the drift forecast;
//!   the camera sensor is a flagged fallback that never teaches the FDF.
//! - **H7** one FOP feature vector per sample (`focus::fop_features`),
//!   written into the frame's `quality_json`.
//! - **G-m** `guiding_during_autofocus`, and **F-d** TempComp off for the
//!   whole run.
//!
//! The physical-autofocus gate (`validate_simulation_focus_run`) is
//! unchanged (D18): a run needs the simulator camera and a simulator focuser
//! until the A03/A06/A13/A15/A23 qualification exists. Everything above runs
//! for those simulator runs, and their records stay out of training sets
//! (`training_label_eligible: false`).

use crate::api::api_error::ApiError;
use crate::config::GuidingDuringAutofocus;
use crate::control::safety_guard::{FocusSafetyGuard, SafetyVerdict};
use crate::equipment::busy::BusyFlag;
use crate::equipment::commands::{Domain, FocuserCmd, GateError, GatedCommand, GatedOp, GuiderCmd};
use crate::equipment::events::{EquipmentEvent, EventKind};
use crate::equipment::gate::{GatedDeviceCommand, SubmitError, Submitted};
use crate::equipment::handle::{DeviceHandle, FocuserHandle, GuiderHandle, Permit};
use crate::equipment::ops::OpKind;
use crate::equipment::snapshot::{FocuserSnapshot, GuiderAppState, Quiescence, TemperatureReading};
use crate::equipment::traits::EquipmentReader;
use crate::equipment::types::{
    AutonomyMode, ClockQuality, CommandId, CommandSource, DeviceKind, DriverId, Observed,
    OpOutcome, Priority,
};
use crate::error::{AchernarError, FocusError, HardwareError};
use crate::focus::focus_agent::{FocusRunContext, FocusRunState, FocusSample};
use crate::focus::fop_features::{self, CfzInputs, FopInputs, HistorySample, VerifiedFilter};
use crate::hardware::ols_ffi::{OlsBayerType, OlsCamOptionId, OlsStreamType};
use crate::state::{AppState, CameraActivity};
use crate::storage::capture_archive::ArchiveReservation;
use crate::storage::capture_store::{
    AssetRole, CommittedCapture, PendingAsset, PendingCapture, TemperatureState,
};
use crate::storage::focus_actions_repo::{FocusActionRow, FocusVerification};
use crate::storage::focus_samples_repo::{FocusSampleProvenance, FocusSampleRow};
use crate::storage::local_store::LocalStore;
use crate::vision::observed_frame::ObservedFrame;
use crate::vision::shared_observation::SharedObservation;
use axum::{extract::State, http::StatusCode, Json};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};
use std::sync::{atomic::Ordering, Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use uuid::Uuid;

#[derive(Debug, Serialize)]
pub struct FocusRunResponse {
    pub run_id: String,
    pub state: String,
    pub best_position: Option<i32>,
    pub best_hfr: Option<f64>,
    pub samples_count: u32,
    /// H4: whether a fresh frame after the move to best focus confirmed it.
    /// Present only in the response of the run that just finished.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verified: Option<bool>,
    /// Things that went wrong around a run that still finished, e.g. guiding
    /// that could not be resumed.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct StartFocusRunRequest {
    pub method: Option<String>,
    pub filter_name: Option<String>,
    pub step_size: Option<i32>,
    /// `"monotonic"` (default) or `"reversal"`. See `focus_agent::SweepPattern`.
    ///
    /// Per-run rather than per-installation: the observing plan wants roughly
    /// one run in three to reverse, which is a decision a night's script makes,
    /// not one an operator edits into a config file between sweeps.
    pub sweep_pattern: Option<String>,
    /// Positions either side of the start, overriding `sweep_radius_points`.
    /// 3 gives the seven-point sweep; campaign nights want 12 (25 points).
    pub sweep_radius_points: Option<u32>,
}

/// POST /api/v1/focus/run — start a new autofocus run
pub async fn start_focus_run(
    State(state): State<AppState>,
    Json(req): Json<StartFocusRunRequest>,
) -> Result<Json<FocusRunResponse>, ApiError> {
    // Like a camera exposure, an autofocus sweep is a hardware transaction,
    // not a browser-request lifetime. Closing the page must not drop the
    // acquisition guard halfway through a focuser move and let another camera
    // command interleave. The explicit `/focus/abort` signal remains active.
    tokio::spawn(async move { start_focus_run_owned(state, req).await })
        .await
        .map_err(|error| {
            tracing::error!("Autofocus task failed: {error}");
            ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "focus_task_failed",
                error.to_string(),
            )
        })?
}

async fn start_focus_run_owned(
    state: AppState,
    req: StartFocusRunRequest,
) -> Result<Json<FocusRunResponse>, ApiError> {
    let method = req.method.as_deref().unwrap_or("classical");
    if !matches!(method, "classical" | "deterministic" | "hfr") {
        tracing::warn!("Unsupported focus method requested: {method}");
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_focus_method",
            format!("Unsupported focus method: {method}"),
        ));
    }

    let mut config = state.focus_config.clone();
    if let Some(step_size) = req.step_size {
        config.initial_step_size = step_size.abs().max(1);
    }
    if let Some(radius) = req.sweep_radius_points {
        config.sweep_radius_points = radius.max(1);
    }
    if let Some(requested) = req.sweep_pattern.as_deref() {
        // Reject an unrecognised pattern rather than silently sweeping
        // monotonically: a night's script that asks for reversals and quietly
        // gets none produces a dataset whose direction features look collected
        // and are not, which is only discoverable months later at training time.
        match crate::focus::focus_agent::SweepPattern::parse(requested) {
            Some(pattern) => config.sweep_pattern = pattern,
            None => {
                tracing::warn!("Unsupported sweep pattern requested: {requested}");
                return Err(ApiError::new(
                    StatusCode::BAD_REQUEST,
                    "invalid_sweep_pattern",
                    format!("Unsupported sweep pattern: {requested}"),
                ));
            }
        }
    }

    // Autofocus owns camera acquisition for the complete sweep. Parameter
    // writes, stream stop/restart, capture and disconnect must not interleave
    // with frames whose HFR labels are being collected. The abort endpoint is
    // intentionally out-of-band and signals through `focus_abort_requested`.
    let _acquisition = state.acquisition.lock().await;
    state.focus_abort_requested.store(false, Ordering::SeqCst);

    // Both read the OLS camera (blocking FFI under the camera mutex): off
    // the async threads (D3).
    let camera_state = state.clone();
    tokio::task::spawn_blocking(move || {
        validate_simulation_focus_run(&camera_state)?;
        ensure_streaming(&camera_state)
    })
    .await
    .map_err(|error| {
        focus_api_error(
            HardwareError::Communication(format!("the camera check did not finish: {error}"))
                .into(),
        )
    })?
    .map_err(focus_api_error)?;
    let camera: Arc<dyn FocusCamera> = Arc::new(OlsFocusCamera {
        state: state.clone(),
    });
    let guiding = state.equipment_ctx.configs.guider.guiding_during_autofocus;
    run_autofocus_session(&state, config, req.filter_name.clone(), camera, guiding)
        .await
        .map(Json)
}

/// POST /api/v1/focus/abort — abort current focus run
pub async fn abort_focus_run(State(state): State<AppState>) -> Result<StatusCode, StatusCode> {
    state.focus_abort_requested.store(true, Ordering::SeqCst);
    if let Ok(mut agent) = state.focus_agent.try_lock() {
        agent.abort();
    }

    // The focuser's audited abort (D19): its out-of-band halt plus a cancel
    // of the running move. The gate's abort epoch also stops the run's later
    // moves. No slot guard is held across the call.
    if let Ok(handle) = crate::api::equipment::focuser_abort_handle(&state) {
        let _ = state
            .equipment
            .gate()
            .abort(&handle, CommandSource::Operator { overriding: false });
    }

    Ok(StatusCode::OK)
}

/// GET /api/v1/focus/status — get current focus state
pub async fn get_focus_status(
    State(state): State<AppState>,
) -> Result<Json<FocusRunResponse>, StatusCode> {
    let agent = state.focus_agent.lock().await;
    if let Some(run) = agent.current_run() {
        Ok(Json(response_from_run(run)))
    } else {
        Ok(Json(FocusRunResponse {
            run_id: String::new(),
            state: format!("{:?}", FocusRunState::Idle),
            best_position: None,
            best_hfr: None,
            samples_count: 0,
            verified: None,
            warnings: Vec::new(),
        }))
    }
}

/// GET /api/v1/focus/history — list past focus runs
pub async fn get_focus_history(
    State(state): State<AppState>,
) -> Result<Json<Vec<FocusRunResponse>>, StatusCode> {
    let agent = state.focus_agent.lock().await;
    Ok(Json(
        agent.history().iter().map(response_from_run).collect(),
    ))
}

fn response_from_run(run: &FocusRunContext) -> FocusRunResponse {
    FocusRunResponse {
        run_id: run.run_id.to_string(),
        state: format!("{:?}", run.state),
        best_position: run.best_position,
        best_hfr: run.best_hfr,
        samples_count: run.samples.len() as u32,
        verified: None,
        warnings: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// The camera stream autofocus measures from
// ---------------------------------------------------------------------------

/// What one autofocus run needs from the camera stream. The daemon's
/// implementation reads the vision loop's output ([`OlsFocusCamera`]); tests
/// drive a run with a scripted stream. `stream` and `sensor_temperature` may
/// block (the OLS camera is FFI): the run calls them on a blocking thread
/// ([`stream_info`], [`sensor_temperature`]).
pub(crate) trait FocusCamera: Send + Sync {
    /// The newest frame the vision loop has observed.
    fn latest(&self) -> Option<ObservedFrame>;
    /// The live stream generation.
    fn generation(&self) -> u64;
    /// The stream's timing and identity.
    fn stream(&self) -> crate::error::Result<StreamInfo>;
    /// The camera-reported sensor temperature (FOP #13).
    fn sensor_temperature(&self) -> (Option<f64>, TemperatureState, Option<String>);
}

/// The timing a frame's exposure window is derived from (H3).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StreamInfo {
    /// Exposure read back from the camera.
    pub exposure_s: f64,
    /// One frame interval of the stream.
    pub frame_interval_s: f64,
    pub camera_name: Option<String>,
    pub driver_id: Option<String>,
    pub clock_quality: ClockQuality,
}

/// The OLS camera behind the vision loop.
struct OlsFocusCamera {
    state: AppState,
}

impl FocusCamera for OlsFocusCamera {
    fn latest(&self) -> Option<ObservedFrame> {
        self.state
            .last_observed_frame
            .read()
            .ok()
            .and_then(|slot| slot.clone())
    }

    fn generation(&self) -> u64 {
        self.state.current_camera_generation()
    }

    /// A stream delivers one frame per exposure, so its frame interval is
    /// the exposure read back from the camera.
    fn stream(&self) -> crate::error::Result<StreamInfo> {
        let driver_id = self
            .state
            .camera_runtime
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .driver_id
            .clone();
        let camera = self.state.camera.lock().unwrap();
        let camera = camera.as_ref().ok_or_else(|| {
            HardwareError::DeviceNotFound("No camera connected for autofocus".to_string())
        })?;
        let parameter = camera
            .get_parameter(OlsCamOptionId::Exp)
            .map_err(|error| HardwareError::Communication(error.to_string()))?;
        if parameter.option_type != crate::hardware::ols_ffi::OlsCamOptionType::Msec
            || !parameter.cur_val.is_finite()
            || parameter.cur_val <= 0.0
        {
            return Err(HardwareError::Unsupported(
                "Simulator frame cadence could not be read back in milliseconds".to_string(),
            )
            .into());
        }
        let exposure_s = parameter.cur_val / 1000.0;
        let camera_name = Some(
            camera
                .name()
                .map_err(|error| HardwareError::Communication(error.to_string()))?,
        );
        Ok(StreamInfo {
            exposure_s,
            frame_interval_s: exposure_s,
            camera_name,
            clock_quality: crate::equipment::camera_hook::clock_quality(driver_id.as_deref()),
            driver_id,
        })
    }

    fn sensor_temperature(&self) -> (Option<f64>, TemperatureState, Option<String>) {
        crate::api::camera::read_temperature(&self.state)
    }
}

/// The stream's timing and identity, read on a blocking thread: the OLS
/// camera calls behind it are blocking FFI under the camera mutex, which
/// must not run on an async thread (D3).
async fn stream_info(camera: &Arc<dyn FocusCamera>) -> crate::error::Result<StreamInfo> {
    let camera = Arc::clone(camera);
    tokio::task::spawn_blocking(move || camera.stream())
        .await
        .map_err(|error| {
            HardwareError::Communication(format!("the camera read did not finish: {error}"))
        })?
}

/// The camera-reported sensor temperature, read on a blocking thread (D3).
async fn sensor_temperature(
    camera: &Arc<dyn FocusCamera>,
) -> (Option<f64>, TemperatureState, Option<String>) {
    let camera = Arc::clone(camera);
    tokio::task::spawn_blocking(move || camera.sensor_temperature())
        .await
        .unwrap_or((None, TemperatureState::Missing, None))
}

// ---------------------------------------------------------------------------
// The autofocus operation (gate parent permit)
// ---------------------------------------------------------------------------

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// One authorised autofocus run: the gate command, the permit holding its
/// domains, and the devices it steps.
struct AutofocusOp {
    command: GatedCommand,
    /// The `focus_runs` id, minted before the run is authorised so that
    /// `GatedOp::Autofocus`, `autofocus_started`, `autofocus_verified` and
    /// `autofocus_failed` all carry it.
    run_id: Uuid,
    /// Released (dropped) by [`AutofocusOp::finish`], or on drop.
    permit: Mutex<Option<Permit>>,
    focuser: FocuserHandle,
    /// The guider, when the run holds it (G-m).
    guider: Option<GuiderHandle>,
    started_at: DateTime<Utc>,
    started_mono_ns: u64,
}

impl AutofocusOp {
    /// Authorise the run (§2.9). The guider domain is claimed only when the
    /// policy acts on a guiding PHD2; the wheel only when an AF filter is
    /// configured and a wheel is connected (§2.9 step 1 table).
    fn authorize(
        state: &AppState,
        focuser: FocuserHandle,
        guiding: GuidingDuringAutofocus,
    ) -> Result<Self, ApiError> {
        let guider = state
            .equipment
            .handle::<GuiderHandle>()
            .filter(|_| guiding != GuidingDuringAutofocus::Keep)
            .filter(|guider| {
                !guider.is_retired()
                    && guider.snapshot().app_state.known() == Some(&GuiderAppState::Guiding)
            });
        let af_filter = !state
            .equipment_ctx
            .configs
            .filter_wheel
            .af_filter
            .trim()
            .is_empty()
            && crate::api::equipment::wheel_slot(state).is_some_and(|w| !w.is_retired());
        let run_id = Uuid::new_v4();
        let op = GatedOp::Autofocus {
            run_id: Some(run_id.to_string()),
        };
        let mut domains = op.default_domains(af_filter);
        if guider.is_some() {
            domains.push(Domain::Device(DeviceKind::Guider));
        }
        let command = GatedCommand::new(
            op,
            CommandSource::Operator { overriding: false },
            AutonomyMode::Manual,
            Priority::Autofocus,
        )
        .with_domains(domains);
        let permit = state
            .equipment
            .gate()
            .authorize(&command)
            .map_err(gate_api_error)?;
        let clock = state.equipment.clock();
        Ok(Self {
            command,
            run_id,
            permit: Mutex::new(Some(permit)),
            focuser,
            guider,
            started_at: clock.now_utc(),
            started_mono_ns: clock.now_mono_ns(),
        })
    }

    fn command_id(&self) -> CommandId {
        self.command.command_id
    }

    /// An event of this run: the focuser's connection, the run's command.
    fn emit(&self, state: &AppState, event: EventKind) {
        let clock = state.equipment.clock();
        state.equipment.emit(
            EquipmentEvent::new(event, clock.as_ref())
                .for_device(DeviceKind::Focuser, self.focuser.connection_id())
                .with_command(self.command.command_id),
        );
    }

    /// The run started: the executed audit stage and the C-1 records.
    fn started(&self, state: &AppState, annotations: Vec<String>) {
        state
            .equipment
            .gate()
            .record_executed(&self.command, annotations);
        self.emit(
            state,
            EventKind::OperationCommanded {
                op: OpKind::Autofocus,
                source: self.command.source.clone(),
            },
        );
        self.emit(
            state,
            EventKind::OperationStarted {
                op: OpKind::Autofocus,
            },
        );
        // The run's `focus_runs` id: the agent runs under it, and the
        // verification and failure events carry it too.
        self.emit(
            state,
            EventKind::AutofocusStarted {
                run_id: self.run_id.to_string(),
            },
        );
    }

    /// The run ended on any path: the outcome audit stage, `operation_ended`
    /// (H18), and the release of every domain.
    fn finish(&self, state: &AppState, outcome: OpOutcome) {
        state
            .equipment
            .gate()
            .record_outcome(&self.command, &outcome);
        let duration_ms = state
            .equipment
            .clock()
            .now_mono_ns()
            .saturating_sub(self.started_mono_ns)
            / 1_000_000;
        self.emit(
            state,
            EventKind::OperationEnded {
                op: OpKind::Autofocus,
                outcome,
                duration_ms,
            },
        );
        lock(&self.permit).take();
    }

    /// Send one step of the run through the gate (a child permit of the
    /// run's, with the step's own checks). Returns on accept.
    async fn submit_step<C, S>(
        &self,
        state: &AppState,
        handle: &DeviceHandle<C, S>,
        op: GatedOp,
    ) -> Result<Submitted, SubmitError>
    where
        C: GatedDeviceCommand,
        S: Send + Sync + 'static,
    {
        let step = GatedCommand::new(
            op,
            self.command.source.clone(),
            AutonomyMode::Manual,
            Priority::Autofocus,
        );
        let gate = state.equipment.gate();
        let pending = {
            let permit = lock(&self.permit);
            match permit.as_ref() {
                Some(parent) => gate.submit_step(handle, parent, step),
                None => {
                    return Err(SubmitError::Gate(GateError::Refused {
                        reason: "the autofocus operation has ended".into(),
                    }))
                }
            }
        };
        pending.await
    }

    /// A step, then its outcome.
    async fn step_and_wait<C, S>(
        &self,
        state: &AppState,
        handle: &DeviceHandle<C, S>,
        op: GatedOp,
    ) -> Result<OpOutcome, SubmitError>
    where
        C: GatedDeviceCommand,
        S: Send + Sync + 'static,
    {
        let submitted = self.submit_step(state, handle, op).await?;
        Ok(submitted.ticket.done.await.unwrap_or_else(|_| {
            OpOutcome::Unknown(crate::equipment::types::DeviceError::outcome_unknown(
                "the worker stopped before the step resolved",
            ))
        }))
    }
}

fn gate_api_error(err: GateError) -> ApiError {
    crate::api::equipment::submit_error(DeviceKind::Focuser, &SubmitError::Gate(err))
}

/// A refused or failed step as the run's error: a safety refusal stays a
/// focus safety violation (400), a missing device 412, anything else 502.
fn step_error(kind: DeviceKind, err: &SubmitError) -> AchernarError {
    let api = crate::api::equipment::submit_error(kind, err);
    match api.status {
        StatusCode::UNPROCESSABLE_ENTITY => FocusError::SafetyViolation(api.message).into(),
        StatusCode::PRECONDITION_FAILED if kind == DeviceKind::Focuser => {
            FocusError::NoFocuser.into()
        }
        _ => HardwareError::Communication(format!("{}: {}", api.code, api.message)).into(),
    }
}

// ---------------------------------------------------------------------------
// TempComp (F-d) and guiding (G-m) around the run
// ---------------------------------------------------------------------------

/// Switch TempComp off for the whole run when it is on (F-d: "Autofocus
/// always runs with TempComp off"). Returns whether the run must restore it.
async fn hold_tempcomp_off(state: &AppState, op: &AutofocusOp) -> crate::error::Result<bool> {
    let snapshot = op.focuser.snapshot();
    if !(snapshot.capabilities.temp_comp_available
        && snapshot.state.temp_comp.known() == Some(&true))
    {
        return Ok(false);
    }
    match op
        .step_and_wait(
            state,
            &op.focuser,
            GatedOp::Focuser(FocuserCmd::SetTempComp(false)),
        )
        .await
    {
        Ok(OpOutcome::Succeeded) => Ok(true),
        Ok(other) => Err(HardwareError::Communication(format!(
            "TempComp could not be switched off for autofocus: the step ended {}",
            other.label()
        ))
        .into()),
        Err(err) => Err(step_error(DeviceKind::Focuser, &err)),
    }
}

/// Restore TempComp after the run, on every path.
async fn release_tempcomp(
    state: &AppState,
    op: &AutofocusOp,
    held: bool,
    warnings: &mut Vec<String>,
) {
    if !held {
        return;
    }
    let result = op
        .step_and_wait(
            state,
            &op.focuser,
            GatedOp::Focuser(FocuserCmd::SetTempComp(true)),
        )
        .await;
    match result {
        Ok(OpOutcome::Succeeded) => {}
        Ok(other) => warnings.push(format!(
            "TempComp was not restored after autofocus: the step ended {}",
            other.label()
        )),
        Err(err) => warnings.push(format!(
            "TempComp was not restored after autofocus: {}",
            step_error(DeviceKind::Focuser, &err)
        )),
    }
}

/// What the run did to PHD2 before the sweep (G-m).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GuidingHeld {
    Untouched,
    Paused,
    Stopped,
}

/// `[guider] guiding_during_autofocus`, after NINA's
/// `AutoFocusDisableGuiding`: `keep` leaves PHD2 alone; `pause` and `stop`
/// act only on a guiding PHD2, and a failure refuses the run.
async fn hold_guiding(
    state: &AppState,
    op: &AutofocusOp,
    policy: GuidingDuringAutofocus,
) -> crate::error::Result<GuidingHeld> {
    let Some(guider) = op.guider.as_ref() else {
        return Ok(GuidingHeld::Untouched);
    };
    let (cmd, held) = match policy {
        GuidingDuringAutofocus::Keep => return Ok(GuidingHeld::Untouched),
        GuidingDuringAutofocus::Pause => (GuiderCmd::Pause { full: false }, GuidingHeld::Paused),
        GuidingDuringAutofocus::Stop => (GuiderCmd::Stop, GuidingHeld::Stopped),
    };
    match op.step_and_wait(state, guider, GatedOp::Guider(cmd)).await {
        Ok(OpOutcome::Succeeded) => Ok(held),
        Ok(other) => Err(HardwareError::Communication(format!(
            "guiding could not be {} for autofocus: the guider step ended {}",
            if held == GuidingHeld::Paused {
                "paused"
            } else {
                "stopped"
            },
            other.label()
        ))
        .into()),
        Err(err) => Err(step_error(DeviceKind::Guider, &err)),
    }
}

/// After the run: `pause` resumes and settles, `stop` restarts and settles
/// (PHD2's `guide` with the `[guider]` settle, which on a guiding PHD2 only
/// settles). Failures are reported, never fatal: the focus result stands.
async fn release_guiding(
    state: &AppState,
    op: &AutofocusOp,
    held: GuidingHeld,
    warnings: &mut Vec<String>,
) {
    let Some(guider) = op.guider.as_ref() else {
        return;
    };
    let settle = state.equipment_ctx.configs.guider.settle();
    let mut steps = Vec::new();
    match held {
        GuidingHeld::Untouched => return,
        GuidingHeld::Paused => steps.push(GuiderCmd::Resume),
        GuidingHeld::Stopped => {}
    }
    steps.push(GuiderCmd::Start {
        settle,
        recalibrate: false,
    });
    for cmd in steps {
        let label = match cmd {
            GuiderCmd::Resume => "resume",
            _ => "restart and settle",
        };
        match op.step_and_wait(state, guider, GatedOp::Guider(cmd)).await {
            Ok(OpOutcome::Succeeded) => {}
            Ok(other) => {
                warnings.push(format!(
                    "guiding did not {label} after autofocus: the guider step ended {}",
                    other.label()
                ));
                return;
            }
            Err(err) => {
                warnings.push(format!(
                    "guiding did not {label} after autofocus: {}",
                    step_error(DeviceKind::Guider, &err)
                ));
                return;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// One run
// ---------------------------------------------------------------------------

/// The filter a run is measured through: a verified wheel slot when a wheel
/// is connected and verified (H8), otherwise the client's label, unverified.
#[derive(Debug, Clone, PartialEq)]
struct RunFilter {
    name: Option<String>,
    verified: Option<VerifiedFilter>,
    missing_reason: &'static str,
}

fn run_filter(state: &AppState, label: Option<String>, warnings: &mut Vec<String>) -> RunFilter {
    let Some(wheel) = crate::api::equipment::wheel_slot(state).filter(|w| !w.is_retired()) else {
        return RunFilter {
            name: label,
            verified: None,
            missing_reason: "no_filter_wheel",
        };
    };
    let snapshot = wheel.snapshot();
    let slot = snapshot
        .verified_slot
        .filter(|_| snapshot.verified)
        .and_then(|index| snapshot.slots.iter().find(|slot| slot.index == index));
    match slot {
        Some(slot) => {
            if let Some(label) = label.as_deref().filter(|l| *l != slot.name) {
                warnings.push(format!(
                    "the requested filter label {label:?} differs from the wheel's verified \
                     filter {:?}; the verified filter is recorded",
                    slot.name
                ));
            }
            RunFilter {
                name: Some(slot.name.clone()),
                verified: Some(VerifiedFilter {
                    name: slot.name.clone(),
                    offset_steps: slot.offset,
                }),
                missing_reason: "",
            }
        }
        None => RunFilter {
            name: label,
            verified: None,
            missing_reason: "filter_unverified",
        },
    }
}

/// What accumulates over one run, outside the agent: the FOP history and
/// the A06 provenance of every sample, keyed by frame id.
#[derive(Debug, Default)]
struct RunLedger {
    history: Vec<HistorySample>,
    provenance: HashMap<String, FocusSampleProvenance>,
}

/// Everything one run's steps share.
struct RunContext {
    state: AppState,
    config: crate::config::FocusConfig,
    camera: Arc<dyn FocusCamera>,
    op: AutofocusOp,
    focus_safety: Mutex<FocusSafetyGuard>,
    rig_id: String,
    filter: RunFilter,
    /// A simulator camera or focuser: the run teaches no rig model (D18).
    simulation: bool,
    start_position: i32,
    cfz: Option<CfzInputs>,
    ledger: Mutex<RunLedger>,
}

/// The run's optics for the CFZ (FOP #33): the focal ratio from `[focus.drift]`
/// or the rig row, and a configured or V-curve-measured focuser scale.
async fn run_cfz(state: &AppState) -> Option<CfzInputs> {
    let drift = &state.focus_config.drift;
    let f_ratio = drift
        .f_ratio
        .or_else(|| state.rig_optics.as_ref().and_then(|r| r.focal_ratio))?;
    let steps_per_micron = match drift.steps_per_micron {
        Some(spm) => Some(spm),
        None => {
            let focuser = state.focuser_name.lock().ok().and_then(|n| n.clone())?;
            crate::storage::device_profiles_repo::load_focuser_scale(
                state.local_store.pool(),
                &state.rig_id,
                &focuser,
            )
            .await
            .ok()
            .flatten()
        }
    }?;
    Some(CfzInputs {
        f_ratio,
        wavelength_nm: drift.wavelength_nm_or_default(),
        steps_per_micron,
    })
}

/// Run one autofocus sweep end to end: authorise, hold TempComp and guiding,
/// sweep, move to best, verify (H4), teach the FDF (H5), release, record.
async fn run_autofocus_session(
    state: &AppState,
    config: crate::config::FocusConfig,
    filter_label: Option<String>,
    camera: Arc<dyn FocusCamera>,
    guiding: GuidingDuringAutofocus,
) -> Result<FocusRunResponse, ApiError> {
    let focuser = crate::api::equipment::focuser_handle(state)
        .map_err(|_| focus_api_error(FocusError::NoFocuser.into()))?;
    let start_position = current_focuser_position(&focuser).map_err(focus_api_error)?;
    let stream = stream_info(&camera).await.map_err(focus_api_error)?;
    let simulation =
        stream.driver_id.as_deref() == Some("sim") || focuser.identity().driver == DriverId::Sim;
    let mut warnings = Vec::new();
    let filter = run_filter(state, filter_label, &mut warnings);
    let wheel_connected =
        crate::api::equipment::wheel_slot(state).is_some_and(|wheel| !wheel.is_retired());
    if let Some(warning) = af_filter_warning(
        &state.equipment_ctx.configs.filter_wheel.af_filter,
        &filter,
        wheel_connected,
    ) {
        warnings.push(warning);
    }
    let cfz = run_cfz(state).await;

    let op = AutofocusOp::authorize(state, focuser, guiding)?;
    // H20: no model switch during autofocus (v6 §6.7.6).
    let _busy = state.equipment.busy_flags().claim(BusyFlag::Autofocus);
    let mut annotations = vec![
        format!("guiding_during_autofocus={guiding:?}"),
        format!("simulation={simulation}"),
    ];
    for warning in &warnings {
        tracing::warn!("autofocus: {warning}");
        annotations.push(format!("warning: {warning}"));
    }
    let logged = warnings.len();
    op.started(state, annotations);
    let ctx = Arc::new(RunContext {
        state: state.clone(),
        focus_safety: Mutex::new(FocusSafetyGuard::new(config.safety.clone())),
        config,
        camera,
        op,
        rig_id: connected_rig_id(state),
        filter,
        simulation,
        start_position,
        cfz,
        ledger: Mutex::new(RunLedger::default()),
    });

    let tempcomp = match hold_tempcomp_off(state, &ctx.op).await {
        Ok(held) => held,
        Err(err) => {
            let reason = err.to_string();
            ctx.op.emit(
                state,
                EventKind::AutofocusFailed {
                    run_id: ctx.op.run_id.to_string(),
                    reason: reason.clone(),
                },
            );
            ctx.op.finish(state, failed_outcome(&reason));
            return Err(focus_api_error(err));
        }
    };
    let held = match hold_guiding(state, &ctx.op, guiding).await {
        Ok(held) => held,
        Err(err) => {
            release_tempcomp(state, &ctx.op, tempcomp, &mut warnings).await;
            let reason = err.to_string();
            ctx.op.emit(
                state,
                EventKind::AutofocusFailed {
                    run_id: ctx.op.run_id.to_string(),
                    reason: reason.clone(),
                },
            );
            ctx.op.finish(state, failed_outcome(&reason));
            return Err(focus_api_error(err));
        }
    };

    let result = sweep_and_verify(&ctx, &mut warnings).await;

    release_guiding(state, &ctx.op, held, &mut warnings).await;
    release_tempcomp(state, &ctx.op, tempcomp, &mut warnings).await;
    for warning in &warnings[logged..] {
        tracing::warn!("autofocus: {warning}");
    }
    match result {
        Ok(SweepResult {
            mut response,
            not_verified,
        }) => {
            // H4: a run that converged but whose verification frame did not
            // confirm focus has not succeeded. Accepted completion is not
            // confirmed completion (v6 §7.8): the audit outcome and
            // `operation_ended` agree with the run's `autofocus_failed`. The
            // body still returns the converged result, with `verified: false`.
            let outcome = match &not_verified {
                Some(reason) => failed_outcome(&format!("focus_not_verified: {reason}")),
                None => OpOutcome::Succeeded,
            };
            ctx.op.finish(state, outcome);
            response.warnings = warnings;
            Ok(response)
        }
        Err(err) => {
            let outcome = if state.focus_abort_requested.load(Ordering::SeqCst) {
                OpOutcome::Aborted { by: None }
            } else {
                failed_outcome(&err.to_string())
            };
            ctx.op.finish(state, outcome);
            Err(focus_api_error(err))
        }
    }
}

fn failed_outcome(reason: &str) -> OpOutcome {
    OpOutcome::Failed(crate::equipment::types::DeviceError::driver(reason))
}

/// `[filter_wheel] af_filter` is configured, but the run does not switch to
/// it: switching goes through the filter handshake as a child step, and that
/// contract does not exist yet. Say so in the response, the audit and the
/// log, rather than let the operator believe the run measured through it.
/// `None` when no AF filter is configured or the verified slot is already it
/// (matched like the wheel matches names: trimmed, ASCII case-insensitive).
fn af_filter_warning(af_filter: &str, filter: &RunFilter, wheel_connected: bool) -> Option<String> {
    let wanted = af_filter.trim();
    if wanted.is_empty() {
        return None;
    }
    if let Some(verified) = &filter.verified {
        if verified.name.trim().eq_ignore_ascii_case(wanted) {
            return None;
        }
        return Some(format!(
            "the configured AF filter {wanted:?} was not applied (autofocus does not switch \
             filters yet); the run measured through the wheel's verified filter {:?}",
            verified.name
        ));
    }
    let through = match &filter.name {
        Some(label) => format!("through the unverified filter label {label:?}"),
        None => "through an unknown filter".to_string(),
    };
    Some(if wheel_connected {
        format!(
            "the configured AF filter {wanted:?} was not applied (autofocus does not switch \
             filters yet) and the wheel's filter is not verified; the run measured {through}"
        )
    } else {
        format!(
            "the configured AF filter {wanted:?} was not applied: no filter wheel is \
             connected; the run measured {through}"
        )
    })
}

/// A sweep that ran to its end: the response, and why focus was not
/// verified when the H4 frame did not confirm it.
struct SweepResult {
    response: FocusRunResponse,
    not_verified: Option<String>,
}

/// The sweep, the move to best focus, the verification (H4) and the FDF
/// update (H5).
async fn sweep_and_verify(
    ctx: &Arc<RunContext>,
    warnings: &mut Vec<String>,
) -> crate::error::Result<SweepResult> {
    let state = &ctx.state;
    let mut agent = state.focus_agent.lock().await;
    let measure_ctx = Arc::clone(ctx);
    let run_result = agent
        .run_autofocus_with_id(
            ctx.op.run_id,
            &ctx.config,
            ctx.rig_id.clone(),
            ctx.filter.name.clone(),
            ctx.start_position,
            move |position| {
                let ctx = Arc::clone(&measure_ctx);
                async move { measure_at_position(&ctx, position).await }
            },
        )
        .await;
    let run = match run_result {
        Ok(run) => run,
        Err(err) => {
            // Preserve the partial sweep and every sample collected before the
            // failure. Simulator samples explain the failed run while remaining
            // explicitly excluded from scientific training datasets.
            let failed_run = agent.current_run().cloned();
            drop(agent);
            let run_id = match failed_run {
                Some(failed_run) => {
                    persist_run_logged(ctx, &failed_run).await;
                    failed_run.run_id.to_string()
                }
                None => ctx.op.run_id.to_string(),
            };
            ctx.op.emit(
                state,
                EventKind::AutofocusFailed {
                    run_id,
                    reason: err.to_string(),
                },
            );
            return Err(err);
        }
    };
    drop(agent);

    // Provenance persistence is important, but a local database fault must not
    // strand the focuser at the last sweep position. Preserve what we can, then
    // complete the hardware action and record that action only after it succeeds.
    persist_run_logged(ctx, &run).await;

    let Some(best_position) = run.best_position else {
        // The agent converges only with a best position; nothing was moved
        // to, so there is nothing to verify.
        return Ok(SweepResult {
            response: response_from_run(&run),
            not_verified: Some("the run converged without a best-focus position".into()),
        });
    };
    let from_position = current_focuser_position(&ctx.op.focuser).unwrap_or(run.start_position);
    let final_move = match move_focuser_to_position(ctx, best_position).await {
        Ok(report) => report,
        Err(err) => {
            let failed_run = {
                let mut agent = state.focus_agent.lock().await;
                agent
                    .mark_current_run_failed(run.run_id)
                    .unwrap_or_else(|| {
                        let mut failed = run.clone();
                        failed.state = FocusRunState::Failed;
                        failed
                    })
            };
            persist_run_logged(ctx, &failed_run).await;
            let action = FocusActionRow {
                run_id: Some(failed_run.run_id.to_string()),
                action: "move_to_best".into(),
                from_position: Some(from_position),
                to_position: Some(best_position),
                steps: Some(best_position - from_position),
                reason: Some("move_to_best_failed".into()),
                focuser_move_id: None,
            };
            if let Err(store_err) = persist_focus_action(&state.local_store, &action, None).await {
                tracing::error!("Failed to persist unsuccessful focus action: {store_err}");
            }
            ctx.op.emit(
                state,
                EventKind::AutofocusFailed {
                    run_id: failed_run.run_id.to_string(),
                    reason: format!("move to best focus failed: {err}"),
                },
            );
            return Err(err);
        }
    };

    // H4: a fresh optical sample at the chosen optimum before focus is
    // declared verified; motor-command completion alone is insufficient
    // (v6 §7.8).
    let verification = verify_focus(ctx, &run, best_position, &final_move).await;
    let action = FocusActionRow {
        run_id: Some(run.run_id.to_string()),
        action: "move_to_best".into(),
        from_position: Some(from_position),
        to_position: Some(best_position),
        steps: Some(best_position - from_position),
        reason: Some("classical_hfr_curve_fit".into()),
        focuser_move_id: final_move.move_id,
    };
    let record = FocusVerification {
        verification_frame_id: verification.frame_id.clone(),
        verification_hfr: verification.hfr,
        verified: verification.verified,
    };
    if let Err(err) = persist_focus_action(&state.local_store, &action, Some(&record)).await {
        tracing::error!("Focused successfully but failed to persist focus action: {err}");
    }
    let not_verified = if verification.verified {
        ctx.op.emit(
            state,
            EventKind::AutofocusVerified {
                run_id: run.run_id.to_string(),
                position_readback: verification.position_readback,
                hfr: verification.hfr,
                frame_id: verification.frame_id.clone(),
                measured_at: verification.measured_at.unwrap_or_else(Utc::now),
            },
        );
        None
    } else {
        let detail = verification
            .reason
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        let reason = format!("focus not verified: {detail}");
        warnings.push(reason.clone());
        ctx.op.emit(
            state,
            EventKind::AutofocusFailed {
                run_id: run.run_id.to_string(),
                reason,
            },
        );
        Some(detail)
    };
    if verification.aborted {
        return Err(FocusError::RunFailed("Autofocus aborted".to_string()).into());
    }

    // H5: only a verified, non-simulator run teaches the FDF, at the
    // verification frame's time and the hub's ambient temperature (H6).
    let ambient = state.equipment.ambient_temperature();
    match drift_learning_temperature(ctx.simulation, verification.verified, &ambient) {
        Ok(temperature_c) => {
            let position = verification.position_readback.unwrap_or(best_position);
            let measured_at = verification.measured_at.unwrap_or_else(Utc::now);
            record_drift_measurement(state, position, temperature_c, measured_at, &run.samples)
                .await;
        }
        Err(reason) => {
            tracing::debug!("Focus drift not updated from run {}: {reason}", run.run_id);
        }
    }

    let mut response = response_from_run(&run);
    response.verified = Some(verification.verified);
    Ok(SweepResult {
        response,
        not_verified,
    })
}

/// The focuser's logical position from its latest snapshot (the worker
/// polls it; nothing here enters the driver).
fn current_focuser_position(handle: &FocuserHandle) -> crate::error::Result<i32> {
    handle.snapshot().logical_position.ok_or_else(|| {
        HardwareError::Communication("The focuser has not reported a position".to_string()).into()
    })
}

// ---------------------------------------------------------------------------
// Moves
// ---------------------------------------------------------------------------

/// The last move of a positioning: its `focuser_moves` id and when the
/// worker reported it settled.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct MoveReport {
    command_id: Option<CommandId>,
    move_id: Option<Uuid>,
    settled_event_at: Option<DateTime<Utc>>,
}

impl MoveReport {
    /// Find the move's `focuser_settled` event, published by the worker
    /// before the move's outcome resolves.
    fn find(state: &AppState, seq_before: u64, command_id: CommandId) -> Self {
        let settled = state.equipment.events_since(
            seq_before.saturating_sub(1),
            &["focuser_settled".to_string()],
            500,
        );
        let found = settled.iter().rev().find_map(|event| match &event.event {
            EventKind::FocuserSettled { move_id, .. } if event.command_id == Some(command_id) => {
                Some((*move_id, event.at_utc))
            }
            _ => None,
        });
        Self {
            command_id: Some(command_id),
            move_id: found.and_then(|(move_id, _)| move_id),
            settled_event_at: found.map(|(_, at)| at),
        }
    }
}

async fn move_focuser_to_position(
    ctx: &RunContext,
    target_position: i32,
) -> crate::error::Result<MoveReport> {
    let state = &ctx.state;
    let config = &ctx.config;
    let target = target_position.clamp(config.safety.min_position, config.safety.max_position);
    if target != target_position {
        return Err(FocusError::SafetyViolation(format!(
            "Target position {target_position} is outside [{}, {}]",
            config.safety.min_position, config.safety.max_position
        ))
        .into());
    }

    // Remaining distance after the previous completed move. On an absolute
    // focuser, a completed move that makes no progress (missed steps reported
    // by the device, a firmware clamp, a stall) fails instead of retrying
    // forever. A relative focuser's position is its emulated logical
    // position, kept by the focuser session.
    let mut previous_remaining: Option<i32> = None;
    let tolerance = config.safety.arrival_tolerance_steps.max(0);
    let mut report = MoveReport::default();
    loop {
        abort_if_requested(state)?;
        let handle = &ctx.op.focuser;
        if handle.is_retired() {
            return Err(FocusError::NoFocuser.into());
        }
        let snapshot = handle.snapshot();
        let current = snapshot.logical_position.ok_or_else(|| {
            HardwareError::Communication("The focuser has not reported a position".to_string())
        })?;
        let delta = target - current;
        if delta.abs() <= tolerance {
            return Ok(report);
        }
        if previous_remaining.is_some_and(|previous| delta.abs() >= previous.abs()) {
            return Err(HardwareError::Communication(format!(
                "Focuser stopped at {current} without progressing toward {target}"
            ))
            .into());
        }
        previous_remaining = Some(delta);

        let verdict = lock(&ctx.focus_safety).validate_focus_action(current, delta, 1.0)?;
        let step = match verdict {
            SafetyVerdict::Approved => delta,
            SafetyVerdict::Modified { clamped, .. } => clamped,
            SafetyVerdict::Rejected { reason } => {
                return Err(FocusError::SafetyViolation(reason).into());
            }
        };
        let cmd = if snapshot.capabilities.absolute {
            FocuserCmd::MoveTo(current + step)
        } else {
            FocuserCmd::MoveBy(step)
        };
        // A step of the run's operation: the focuser check (travel bounds,
        // A06 exposure check, origin) and the audit apply to it (D9, O11 b).
        let seq_before = state.equipment.bus().next_seq();
        let submitted = ctx
            .op
            .submit_step(state, handle, GatedOp::Focuser(cmd))
            .await
            .map_err(|err| step_error(DeviceKind::Focuser, &err))?;
        let command_id = submitted.ticket.command_id;
        let mut done = submitted.ticket.done;
        let mut outcome: Option<OpOutcome> = None;

        // A sleep alone is not completion: a slow or belt-reduced focuser can
        // still be moving, which would put the next exposure mid-move. Wait
        // for the move operation to resolve (arrival, then settle), then
        // settle, then re-read the position at the top of the loop.
        let stopped = wait_until_stopped(
            || abort_if_requested(state),
            || match done.try_recv() {
                Ok(result) => {
                    outcome = Some(result);
                    Ok(false)
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => Ok(true),
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    Err(HardwareError::Communication(
                        "The focuser worker stopped before the move resolved".to_string(),
                    )
                    .into())
                }
            },
            // The move operation has its own deadline on the worker; this
            // bound only covers polling and settling on top of it.
            Duration::from_millis(config.safety.move_timeout_ms.max(100)) + MOVE_WAIT_MARGIN,
            Duration::from_millis(100),
        )
        .await;
        if let Err(err) = stopped {
            let _ = state
                .equipment
                .gate()
                .abort(handle, CommandSource::Operator { overriding: false });
            return Err(err);
        }
        match outcome {
            Some(OpOutcome::Succeeded) | None => {}
            Some(OpOutcome::TimedOut) => {
                return Err(HardwareError::Timeout(format!(
                    "Focuser still moving after {} ms",
                    config.safety.move_timeout_ms.max(100)
                ))
                .into());
            }
            Some(other) => {
                return Err(HardwareError::Communication(format!(
                    "The focuser move ended {}",
                    other.label()
                ))
                .into());
            }
        }
        report = MoveReport::find(state, seq_before, command_id);
        tokio::time::sleep(Duration::from_millis(config.safety.settle_ms.max(50))).await;
    }
}

/// How long past `[focus.safety] move_timeout_ms` a move may take to resolve
/// on the worker (polling at 10 Hz, settle, verification read).
const MOVE_WAIT_MARGIN: Duration = Duration::from_secs(2);

/// Poll `is_moving` until it reports false, failing with a timeout after
/// `timeout`. The first poll waits one interval so a driver that raises
/// IsMoving just after accepting the move is not read as already stopped.
/// `check_abort` errors propagate at once; up to `MAX_POLL_ERRORS`
/// consecutive poll errors are tolerated (one dropped Wi-Fi request must not
/// kill a sweep), after which the last error propagates.
async fn wait_until_stopped<A, F>(
    mut check_abort: A,
    mut is_moving: F,
    timeout: Duration,
    poll_interval: Duration,
) -> crate::error::Result<()>
where
    A: FnMut() -> crate::error::Result<()>,
    F: FnMut() -> crate::error::Result<bool>,
{
    const MAX_POLL_ERRORS: u32 = 3;
    let deadline = tokio::time::Instant::now() + timeout;
    let mut consecutive_errors = 0;
    loop {
        tokio::time::sleep(poll_interval).await;
        check_abort()?;
        match is_moving() {
            Ok(false) => return Ok(()),
            Ok(true) => consecutive_errors = 0,
            Err(err) => {
                consecutive_errors += 1;
                if consecutive_errors >= MAX_POLL_ERRORS {
                    return Err(err);
                }
            }
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(HardwareError::Timeout(format!(
                "Focuser still moving after {} ms",
                timeout.as_millis()
            ))
            .into());
        }
    }
}

fn abort_if_requested(state: &AppState) -> crate::error::Result<()> {
    if state.focus_abort_requested.load(Ordering::SeqCst) {
        Err(FocusError::RunFailed("Autofocus aborted".to_string()).into())
    } else {
        Ok(())
    }
}

/// A `focuser_position_reset` (F-h).
#[derive(Debug, Clone, PartialEq)]
struct PositionReset {
    at: DateTime<Utc>,
    from: Option<i32>,
    to: Option<i32>,
    cause: String,
}

impl PositionReset {
    /// The coordinate shift, when the reset reported both ends.
    fn shift(&self) -> Option<f64> {
        match (self.from, self.to) {
            (Some(from), Some(to)) => Some(f64::from(to) - f64::from(from)),
            _ => None,
        }
    }
}

/// Focuser position resets in the journal since `since`.
fn position_resets_since(state: &AppState, since: DateTime<Utc>) -> Vec<PositionReset> {
    let now = state.equipment.clock().now_utc();
    state
        .equipment
        .bus()
        .events_between(since, now)
        .iter()
        .filter_map(|event| match &event.event {
            EventKind::FocuserPositionReset { from, to, cause } => Some(PositionReset {
                at: event.at_utc,
                from: *from,
                to: *to,
                cause: cause.clone(),
            }),
            _ => None,
        })
        .collect()
}

/// F-h: samples on both sides of a position reset are not comparable, so a
/// reset during the run ends it.
fn check_no_position_reset(ctx: &RunContext) -> crate::error::Result<()> {
    match position_resets_since(&ctx.state, ctx.op.started_at).first() {
        Some(reset) => Err(FocusError::RunFailed(format!(
            "The focuser position was reset during autofocus ({}); samples before and after \
             the reset are not comparable",
            reset.cause
        ))
        .into()),
        None => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// H3: sample freshness (A06)
// ---------------------------------------------------------------------------

/// The state a frame is judged against: when the focuser settled, and the
/// newest frame counter and stream generation at that moment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FreshnessBaseline {
    pub settled_at: DateTime<Utc>,
    pub frame_counter: Option<i32>,
    pub generation: u64,
}

/// One frame's timing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FrameTiming {
    pub frame_counter: i32,
    pub generation: u64,
    /// When the frame was delivered (host clock).
    pub arrival: Option<DateTime<Utc>>,
    pub exposure_s: Option<f64>,
    pub frame_interval_s: Option<f64>,
    pub clock_quality: ClockQuality,
    /// The driver's own exposure start, for a dedicated exposure (§3.12).
    /// Stream frames carry none.
    pub exposure_start: Option<DateTime<Utc>>,
}

/// Why a frame was passed over, waiting for a later one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameSkip {
    /// The frame that was newest when the focuser settled.
    SameFrame,
    /// The exposure ended before the focuser settled.
    PreSettle,
    /// The exposure was already integrating when the focuser settled.
    InFlight,
}

/// Why no later frame can help: the measurement fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FrameRefusal {
    /// The counter went backwards within one stream generation.
    CounterReset { baseline: i32, observed: i32 },
    /// The stream restarted.
    GenerationChanged { baseline: u64, observed: u64 },
    /// The exposure's start cannot be bounded.
    UnknownStart(&'static str),
}

impl std::fmt::Display for FrameRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameRefusal::CounterReset { baseline, observed } => write!(
                f,
                "the camera frame counter reset ({observed} after {baseline})"
            ),
            FrameRefusal::GenerationChanged { baseline, observed } => write!(
                f,
                "the camera stream restarted (generation {observed} after {baseline})"
            ),
            FrameRefusal::UnknownStart(reason) => {
                write!(f, "the exposure start time is unknown ({reason})")
            }
        }
    }
}

/// The verdict on one frame's timing (H3).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Freshness {
    /// The exposure window lies wholly after the settle; quiescence over it
    /// is checked next.
    Fresh {
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        /// What the window was derived from, which is what a sample records
        /// as its clock quality: `driver_exposure_start` only when a driver
        /// start was present and used; the conservative arrival bound is
        /// `host_arrival` (`simulator` for simulator frames), whatever the
        /// camera's own clock quality.
        basis: ClockQuality,
    },
    Skip(FrameSkip),
    Refused(FrameRefusal),
}

fn chrono_secs(seconds: f64) -> chrono::Duration {
    chrono::Duration::microseconds((seconds * 1_000_000.0).round() as i64)
}

/// H3 (A06): a sample is accepted only if its exposure started at or after
/// the focuser settled. A dedicated exposure with a driver start time is
/// judged by that start. A stream frame carries only its delivery time, so
/// its exposure is bounded conservatively as starting no earlier than
/// `arrival - (exposure + one frame interval)` (host_arrival, simulator and
/// trigger-bracket timing alike). A frame whose start cannot be bounded, a
/// counter reset and a stream restart are refused; a pre-settle frame and an
/// exposure already in flight are passed over for a later frame.
pub(crate) fn frame_freshness(baseline: &FreshnessBaseline, frame: &FrameTiming) -> Freshness {
    if frame.generation != baseline.generation {
        return Freshness::Refused(FrameRefusal::GenerationChanged {
            baseline: baseline.generation,
            observed: frame.generation,
        });
    }
    if let Some(counter) = baseline.frame_counter {
        if frame.frame_counter == counter {
            return Freshness::Skip(FrameSkip::SameFrame);
        }
        if frame.frame_counter < counter {
            return Freshness::Refused(FrameRefusal::CounterReset {
                baseline: counter,
                observed: frame.frame_counter,
            });
        }
    }
    if frame.clock_quality == ClockQuality::Unknown {
        return Freshness::Refused(FrameRefusal::UnknownStart("clock_quality_unknown"));
    }
    let Some(arrival) = frame.arrival else {
        return Freshness::Refused(FrameRefusal::UnknownStart("no_frame_timestamp"));
    };
    let Some(exposure_s) = frame.exposure_s.filter(|e| e.is_finite() && *e > 0.0) else {
        return Freshness::Refused(FrameRefusal::UnknownStart("exposure_unknown"));
    };
    let (start, end, basis) = match frame.exposure_start {
        Some(start) if frame.clock_quality == ClockQuality::DriverExposureStart => (
            start,
            start + chrono_secs(exposure_s),
            ClockQuality::DriverExposureStart,
        ),
        _ => {
            let Some(interval_s) = frame
                .frame_interval_s
                .filter(|i| i.is_finite() && *i >= 0.0)
            else {
                return Freshness::Refused(FrameRefusal::UnknownStart("frame_interval_unknown"));
            };
            let basis = match frame.clock_quality {
                ClockQuality::Simulator => ClockQuality::Simulator,
                _ => ClockQuality::HostArrival,
            };
            (
                arrival - chrono_secs(exposure_s + interval_s),
                arrival,
                basis,
            )
        }
    };
    if end <= baseline.settled_at {
        return Freshness::Skip(FrameSkip::PreSettle);
    }
    if start < baseline.settled_at {
        return Freshness::Skip(FrameSkip::InFlight);
    }
    Freshness::Fresh { start, end, basis }
}

fn unix_time(seconds: f64) -> Option<DateTime<Utc>> {
    let micros = seconds * 1_000_000.0;
    (micros.is_finite() && seconds > 0.0 && micros <= i64::MAX as f64)
        .then(|| DateTime::<Utc>::from_timestamp_micros(micros.round() as i64))
        .flatten()
}

fn frame_timing(observed: &ObservedFrame, stream: &StreamInfo) -> FrameTiming {
    FrameTiming {
        frame_counter: observed.frame_counter,
        generation: observed.frame.stream_generation,
        arrival: unix_time(observed.capture_unix_time),
        exposure_s: Some(stream.exposure_s),
        frame_interval_s: Some(stream.frame_interval_s),
        clock_quality: stream.clock_quality,
        exposure_start: None,
    }
}

/// The device kinds whose motion invalidates a focus sample: the optics, the
/// pointing, the filter and the light path. Only connected kinds count.
const QUIESCENCE_KINDS: [DeviceKind; 7] = [
    DeviceKind::Focuser,
    DeviceKind::FilterWheel,
    DeviceKind::Mount,
    DeviceKind::Guider,
    DeviceKind::Rotator,
    DeviceKind::CoverCalibrator,
    DeviceKind::Dome,
];

fn quiescence_kinds(state: &AppState) -> Vec<DeviceKind> {
    let attached: BTreeSet<DeviceKind> = state
        .equipment
        .attached()
        .iter()
        .filter(|handle| !handle.is_retired())
        .map(|handle| handle.kind())
        .collect();
    QUIESCENCE_KINDS
        .iter()
        .copied()
        .filter(|kind| attached.contains(kind))
        .collect()
}

/// A frame that passed H3, with the evidence.
#[derive(Clone)]
struct AcceptedFrame {
    observed: ObservedFrame,
    window_start: DateTime<Utc>,
    window_end: DateTime<Utc>,
    /// The timing basis the window was derived from (see [`Freshness`]).
    timing_basis: ClockQuality,
    readback_after: Option<i32>,
    /// When the after read-back was taken: at or after `window_end`.
    readback_after_at: DateTime<Utc>,
    focuser: Arc<FocuserSnapshot>,
    kinds: Vec<DeviceKind>,
}

/// When the focuser read-back in `snapshot` was taken. An absolute
/// focuser's position is the device's own reading, timed at its request; a
/// relative focuser's is the session's count, current as of the snapshot's
/// publication. Never later than the publication.
fn readback_taken_at(snapshot: &FocuserSnapshot) -> DateTime<Utc> {
    let read_at = snapshot
        .state
        .position
        .observed_at
        .filter(|_| snapshot.capabilities.absolute);
    match read_at {
        Some(at) => at.min(snapshot.published_at),
        None => snapshot.published_at,
    }
}

/// The verdict on one frame, timing and quiescence together.
enum FrameVerdict {
    Accept(Box<AcceptedFrame>),
    /// Passed over; a later frame may do.
    Skip(String),
    /// Not yet decidable (e.g. no device sample after the window yet): ask
    /// again about the same frame.
    Undecided(String),
    Refuse(FrameRefusal),
}

/// H3 for one frame: [`frame_freshness`], then no motion of any connected
/// optics, pointing or light-path device inside the exposure window (the
/// hub's quiescence over its ring buffers and journal), then a focuser
/// read-back taken at or after the window's end that is unchanged since the
/// settle and not in motion.
///
/// The hub's quiescence accepts a window with no focuser sample after its
/// end while the latest sample is recent, so it cannot see a move that an
/// outside client (another Alpaca or ASCOM Remote client, a handset) starts
/// during the exposure. The read-back after must therefore come from a poll
/// that covers the window: until the worker publishes one, the frame is
/// undecided and is asked about again.
fn judge_frame(
    state: &AppState,
    focuser: &FocuserHandle,
    baseline: &FreshnessBaseline,
    readback_before: Option<i32>,
    stream: &StreamInfo,
    observed: &ObservedFrame,
) -> FrameVerdict {
    use crate::equipment::ring::DeviceSnapshot as _;

    let (start, end, timing_basis) =
        match frame_freshness(baseline, &frame_timing(observed, stream)) {
            Freshness::Fresh { start, end, basis } => (start, end, basis),
            Freshness::Skip(skip) => return FrameVerdict::Skip(format!("{skip:?}")),
            Freshness::Refused(refusal) => return FrameVerdict::Refuse(refusal),
        };
    let kinds = quiescence_kinds(state);
    match state.equipment.quiescent(&kinds, start, end) {
        Quiescence::Quiescent { .. } => {}
        Quiescence::Moved { kind, event_seq } => {
            return FrameVerdict::Skip(format!(
                "the {} moved during the exposure (event {event_seq})",
                crate::equipment::types::kind_name(kind)
            ));
        }
        Quiescence::Unknown { reason } => return FrameVerdict::Undecided(reason),
    }
    let snapshot = focuser.snapshot();
    let readback_at = readback_taken_at(&snapshot);
    if readback_at < end {
        return FrameVerdict::Undecided(format!(
            "focuser_no_readback_after_end (latest read-back {readback_at}, exposure end {end})"
        ));
    }
    if snapshot.logical_position != readback_before {
        return FrameVerdict::Skip(format!(
            "the focuser read back {:?} after the exposure, {readback_before:?} before",
            snapshot.logical_position
        ));
    }
    if snapshot.in_motion() {
        return FrameVerdict::Skip(
            "the focuser was moving at the read-back after the exposure".into(),
        );
    }
    FrameVerdict::Accept(Box::new(AcceptedFrame {
        observed: observed.clone(),
        window_start: start,
        window_end: end,
        timing_basis,
        readback_after: snapshot.logical_position,
        readback_after_at: readback_at,
        focuser: snapshot,
        kinds,
    }))
}

/// The longest a measurement waits for a frame that passes H3: the frame
/// window three times over, two focuser idle polls (a settled focuser's
/// first clean sample), and a fixed margin.
fn frame_deadline(stream: &StreamInfo) -> Duration {
    let window = stream.exposure_s + stream.frame_interval_s;
    let window = if window.is_finite() && window > 0.0 {
        Duration::from_secs_f64(window.min(3_600.0))
    } else {
        Duration::ZERO
    };
    Duration::from_secs(5) + window * 3 + crate::equipment::focuser::IDLE_POLL_INTERVAL * 2
}

/// Wait for a genuinely new frame that passes H3 (replacing the old
/// counter-only `wait_for_fresh_frame`). On the deadline it fails with the
/// last reason rather than accepting a stale observation.
async fn wait_for_valid_frame(
    ctx: &RunContext,
    baseline: &FreshnessBaseline,
    readback_before: Option<i32>,
    stream: &StreamInfo,
) -> crate::error::Result<AcceptedFrame> {
    let deadline = Instant::now() + frame_deadline(stream);
    let mut examined: Option<i32> = None;
    let mut last_reason = "no frame arrived".to_string();
    loop {
        abort_if_requested(&ctx.state)?;
        if let Some(observed) = ctx.camera.latest() {
            if examined != Some(observed.frame_counter) {
                match judge_frame(
                    &ctx.state,
                    &ctx.op.focuser,
                    baseline,
                    readback_before,
                    stream,
                    &observed,
                ) {
                    FrameVerdict::Accept(accepted) => return Ok(*accepted),
                    FrameVerdict::Skip(reason) => {
                        examined = Some(observed.frame_counter);
                        last_reason = reason;
                    }
                    FrameVerdict::Undecided(reason) => last_reason = reason,
                    FrameVerdict::Refuse(refusal) => {
                        return Err(FocusError::RunFailed(format!(
                            "Autofocus frame refused (A06): {refusal}"
                        ))
                        .into());
                    }
                }
            }
        }
        if Instant::now() >= deadline {
            return Err(HardwareError::Timeout(format!(
                "No frame passed the A06 freshness rule within {:.1} s (last: {last_reason})",
                frame_deadline(stream).as_secs_f64()
            ))
            .into());
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// The settle point a position's samples are judged from.
struct SettlePoint {
    baseline: FreshnessBaseline,
    readback: Option<i32>,
}

fn settle_point(ctx: &RunContext) -> SettlePoint {
    SettlePoint {
        baseline: FreshnessBaseline {
            settled_at: ctx.state.equipment.clock().now_utc(),
            frame_counter: ctx.camera.latest().map(|observed| observed.frame_counter),
            generation: ctx.camera.generation(),
        },
        readback: ctx.op.focuser.snapshot().logical_position,
    }
}

// ---------------------------------------------------------------------------
// Samples
// ---------------------------------------------------------------------------

async fn measure_at_position(
    ctx: &RunContext,
    position: i32,
) -> crate::error::Result<Vec<FocusSample>> {
    let state = &ctx.state;
    abort_if_requested(state)?;
    check_no_position_reset(ctx)?;
    // Storage backpressure must be visible before even a simulator focuser
    // moves. The reservation is released if movement or settling fails.
    let mut first_archive = Some(state.capture_archive.reserve()?);
    let moved = move_focuser_to_position(ctx, position).await?;
    abort_if_requested(state)?;
    tokio::time::sleep(Duration::from_millis(
        ctx.config.settle_ms.max(ctx.config.safety.settle_ms),
    ))
    .await;
    check_no_position_reset(ctx)?;
    let stream = stream_info(&ctx.camera).await?;
    let mut settle = settle_point(ctx);

    // Keep every frame at this position as its own sample with its own preserved
    // frame. Averaging here would destroy the per-position scatter that the FQE
    // learns from (and that the curve fit weights) — Section 7.6.1 / 7.7.5.
    let samples_per_position = ctx.config.samples_per_position.max(1);
    let mut samples = Vec::with_capacity(samples_per_position as usize);
    for _ in 0..samples_per_position {
        let archive = match first_archive.take() {
            Some(archive) => archive,
            None => state.capture_archive.reserve()?,
        };
        let accepted =
            wait_for_valid_frame(ctx, &settle.baseline, settle.readback, &stream).await?;
        settle.baseline.frame_counter = Some(accepted.observed.frame_counter);
        let (sample, _) = record_sample(
            ctx,
            archive,
            position,
            &accepted,
            &settle,
            &moved,
            &stream,
            SampleRole::Sweep,
        )
        .await?;
        samples.push(sample);
    }

    Ok(samples)
}

/// Whether a frame is a sweep sample or the H4 verification frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SampleRole {
    Sweep,
    Verification,
}

impl SampleRole {
    fn as_str(self) -> &'static str {
        match self {
            SampleRole::Sweep => "sweep",
            SampleRole::Verification => "verification",
        }
    }
}

fn f64_of(value: Option<f32>) -> Option<f64> {
    value.map(f64::from).filter(|v| v.is_finite())
}

/// FOP #35 input. Absent unless measured, so FOP masks it as
/// `background_unavailable` instead of training on a stand-in 0. FOP contract
/// 1.0.0 defines #35 on the legacy global estimator; another method's level
/// is a different population and is masked rather than mixed in.
fn fop_background_input(observation: &SharedObservation) -> Option<f64> {
    observation
        .background
        .as_ref()
        .filter(|background| {
            background.method == crate::vision::background_model::BackgroundMethod::LegacyGlobalClip
        })
        .and_then(|background| background.level())
        .map(f64::from)
        .filter(|value| value.is_finite())
}

/// The tempcomp state recorded with a sample: `on`, `off` or `unavailable`.
fn tempcomp_state(snapshot: &FocuserSnapshot) -> Option<String> {
    if !snapshot.capabilities.temp_comp_available {
        return Some("unavailable".into());
    }
    snapshot
        .state
        .temp_comp
        .known()
        .map(|on| if *on { "on" } else { "off" }.to_string())
}

fn ambient_json(ambient: &Observed<TemperatureReading>, now_mono_ns: u64) -> Value {
    match ambient.known() {
        Some(reading) => json!({
            "celsius": reading.celsius,
            "source": reading.source,
            "role": reading.role,
            "identity_key": reading.identity_key,
            "camera_sensor_fallback": reading.camera_sensor_fallback,
            "state": ambient.state,
            "age_ms": ambient.age_ms(now_mono_ns),
        }),
        None => Value::Null,
    }
}

/// Archive one accepted frame and turn it into a sample, with its A06
/// provenance, the H6 ambient reading and its FOP vector (H7).
#[allow(clippy::too_many_arguments)]
async fn record_sample(
    ctx: &RunContext,
    archive: ArchiveReservation,
    position: i32,
    accepted: &AcceptedFrame,
    settle: &SettlePoint,
    moved: &MoveReport,
    stream: &StreamInfo,
    role: SampleRole,
) -> crate::error::Result<(FocusSample, CommittedCapture)> {
    let state = &ctx.state;
    let observation = &accepted.observed.observation;
    let temperature = sensor_temperature(&ctx.camera).await;
    let ambient = state.equipment.ambient_temperature();
    let now_mono_ns = state.equipment.clock().now_mono_ns();
    let sensor_measured = temperature
        .0
        .filter(|_| temperature.1 == TemperatureState::Measured);
    let humidity = state
        .equipment
        .coc_context()
        .corroboration
        .humidity_pct
        .fresh_value()
        .copied();
    let temperature_coefficient = match (&state.focus_drift, ctx.simulation) {
        (Some(forecaster), false) => {
            let guard = forecaster.lock().await;
            guard
                .is_initialized()
                .then(|| guard.temperature_coefficient().0)
        }
        _ => None,
    };
    let fop = f64_of(observation.hfr_px).filter(|h| *h > 0.0).map(|hfr| {
        let ledger = lock(&ctx.ledger);
        fop_features::build(&FopInputs {
            position,
            run_start_position: ctx.start_position,
            history: &ledger.history,
            classical_hfr: hfr,
            classical_fwhm: f64_of(observation.fwhm_px),
            star_count: observation.star_count,
            fqe_hfr_px: f64_of(observation.fqe_hfr_px),
            fqe_sigma_log_hfr: f64_of(observation.fqe_sigma_log_hfr),
            quality_uncertainty: f64_of(observation.quality_uncertainty),
            background_adu: fop_background_input(observation),
            saturation_fraction: f64_of(observation.saturation_fraction),
            saturation_gate_applied: observation.saturation_gate.was_applied(),
            sensor_temperature_c: sensor_measured,
            ambient: &ambient,
            humidity_pct: humidity,
            temperature_coefficient,
            focuser: Some(&accepted.focuser),
            filter: ctx.filter.verified.as_ref(),
            filter_missing_reason: ctx.filter.missing_reason,
            cfz: ctx.cfz,
        })
    });

    let mut extra = serde_json::Map::new();
    extra.insert("autofocus_role".into(), json!(role.as_str()));
    extra.insert(
        "a06".into(),
        json!({
            "rule": "exposure_start_after_settle_and_quiescent_v1",
            "settled_at": settle.baseline.settled_at,
            "focuser_settled_event_at": moved.settled_event_at,
            "exposure_window_start": accepted.window_start,
            "exposure_window_end": accepted.window_end,
            // The basis the window was derived from; the camera's own
            // clock quality beside it.
            "clock_quality": accepted.timing_basis,
            "stream_clock_quality": stream.clock_quality,
            "frame_counter_baseline": settle.baseline.frame_counter,
            "position_readback_before": settle.readback,
            "position_readback_after": accepted.readback_after,
            "position_readback_after_at": accepted.readback_after_at,
            "focuser_snapshot_published_at": accepted.focuser.published_at,
            "position_source": accepted.focuser.position_source,
            "focuser_move_id": moved.move_id,
            "focuser_move_command_id": moved.command_id,
            "quiescent_kinds": accepted.kinds,
            "autofocus_command_id": ctx.op.command_id(),
        }),
    );
    extra.insert(
        "ambient_temperature".into(),
        ambient_json(&ambient, now_mono_ns),
    );
    extra.insert(
        "filter_verified".into(),
        json!(ctx.filter.verified.is_some()),
    );
    extra.insert(
        "fop_features".into(),
        fop.as_ref().map_or(Value::Null, |v| v.to_json()),
    );
    // A sweep sample's label exists only once the run converges (training
    // derives it from focus_runs.best_position). The verification frame is
    // taken at the chosen optimum, so its label is known now: 0 steps, when
    // the origin is reliable.
    if role == SampleRole::Verification {
        extra.insert(
            "fop_label_steps".into(),
            json!(fop.as_ref().and_then(|v| v.label(position, Some(position)))),
        );
    }

    let (equipment, finalise) = focus_frame_equipment(ctx, accepted, stream);
    let committed = persist_observed_frame(
        state,
        ctx.camera.as_ref(),
        archive,
        position,
        &ctx.rig_id,
        ctx.filter.name.as_deref(),
        &accepted.observed,
        stream,
        temperature.clone(),
        extra,
        Some(equipment),
    )
    .await?;
    // D7: rewrite the frame's window row once every device has reported
    // after the exposure (and write PHD2's per-exposure row, H9).
    let mut plan = finalise;
    plan.input.frame_id = committed.frame_id.clone();
    crate::api::camera::equipment::spawn_finaliser(state, plan);
    let sample = sample_from_observation(position, observation, &committed, temperature.0)?;

    let age_ms = temperature
        .0
        .map(|_| (Utc::now() - accepted.window_end).num_milliseconds().max(0) as u64);
    let provenance = FocusSampleProvenance {
        position_readback_before: settle.readback,
        position_readback_after: accepted.readback_after,
        position_source: Some(accepted.focuser.position_source),
        focuser_move_id: moved.move_id,
        settled_at: Some(settle.baseline.settled_at),
        exposure_started_at: Some(accepted.window_start),
        capture_id: Some(committed.capture_id.clone()),
        // What `exposure_started_at` was derived from, not the camera's
        // nominal quality: a stream frame's start is the arrival bound.
        clock_quality: Some(accepted.timing_basis),
        moved_during_exposure: Some(false),
        filter_verified: Some(ctx.filter.verified.is_some()),
        // These describe `focus_samples.temperature`, which is the camera's
        // own reading (FOP contract #13). The H6 ambient reading travels in
        // the frame's quality_json.
        temperature_source: temperature.2.clone(),
        temperature_age_ms: age_ms,
        tempcomp_state: tempcomp_state(&accepted.focuser),
        backlash_model: Some(accepted.focuser.backlash_model),
    };
    if role == SampleRole::Sweep {
        let mut ledger = lock(&ctx.ledger);
        ledger.history.push(HistorySample {
            position,
            hfr: sample.hfr,
            sensor_temperature_c: sensor_measured,
        });
        if let Some(frame_id) = sample.frame_id.clone() {
            ledger.provenance.insert(frame_id, provenance);
        }
    }
    Ok((sample, committed))
}

// ---------------------------------------------------------------------------
// H4: verification
// ---------------------------------------------------------------------------

/// The verification frame must not be worse than the best measured sweep
/// position by more than this fraction of its HFR. The fitted optimum is
/// normally at least as sharp as the best sampled position; the margin
/// absorbs seeing and measurement scatter. Both HFRs are recorded, so a
/// stricter rule can be applied offline.
const VERIFY_HFR_TOLERANCE: f64 = 0.10;

/// The H4 result.
#[derive(Debug, Clone, PartialEq, Default)]
struct Verification {
    verified: bool,
    frame_id: Option<String>,
    hfr: Option<f64>,
    position_readback: Option<i32>,
    measured_at: Option<DateTime<Utc>>,
    reason: Option<String>,
    aborted: bool,
}

/// The best (smallest) per-position median HFR of the sweep.
fn best_sampled_hfr(samples: &[FocusSample]) -> Option<f64> {
    let mut by_position: HashMap<i32, Vec<f64>> = HashMap::new();
    for sample in samples {
        if sample.hfr.is_finite() && sample.hfr > 0.0 {
            by_position
                .entry(sample.position)
                .or_default()
                .push(sample.hfr);
        }
    }
    by_position
        .into_values()
        .filter_map(|mut values| {
            values.sort_by(f64::total_cmp);
            let n = values.len();
            match n {
                0 => None,
                _ if n % 2 == 1 => Some(values[n / 2]),
                _ => Some(0.5 * (values[n / 2 - 1] + values[n / 2])),
            }
        })
        .min_by(f64::total_cmp)
}

/// The H4 acceptance rule: a valid HFR at the commanded position, read back
/// with a reliable origin, no worse than the best sampled position by more
/// than [`VERIFY_HFR_TOLERANCE`].
fn verification_verdict(
    hfr: f64,
    reference_hfr: Option<f64>,
    readback: Option<i32>,
    best_position: i32,
    tolerance_steps: i32,
    reliable_origin: bool,
) -> Result<(), String> {
    if !reliable_origin {
        return Err("the focuser origin is not reliable".into());
    }
    match readback {
        Some(position) if (position - best_position).abs() <= tolerance_steps.max(0) => {}
        other => {
            return Err(format!(
                "the focuser read back {other:?}, not the best-focus position {best_position}"
            ));
        }
    }
    let Some(reference) = reference_hfr else {
        return Err("the sweep has no valid HFR to compare against".into());
    };
    if hfr <= reference * (1.0 + VERIFY_HFR_TOLERANCE) {
        Ok(())
    } else {
        Err(format!(
            "verification HFR {hfr:.3} px is worse than the best sampled {reference:.3} px"
        ))
    }
}

/// H4: settle at the fitted optimum, take a fresh frame that passes H3,
/// archive it, and judge it.
async fn verify_focus(
    ctx: &RunContext,
    run: &FocusRunContext,
    best_position: i32,
    final_move: &MoveReport,
) -> Verification {
    let failed = |reason: String| Verification {
        reason: Some(reason),
        aborted: ctx.state.focus_abort_requested.load(Ordering::SeqCst),
        ..Verification::default()
    };
    if abort_if_requested(&ctx.state).is_err() {
        return failed("aborted".into());
    }
    tokio::time::sleep(Duration::from_millis(
        ctx.config.settle_ms.max(ctx.config.safety.settle_ms),
    ))
    .await;
    if let Err(err) = check_no_position_reset(ctx) {
        return failed(err.to_string());
    }
    let archive = match ctx.state.capture_archive.reserve() {
        Ok(archive) => archive,
        Err(err) => return failed(format!("no archive slot for the verification frame: {err}")),
    };
    let stream = match stream_info(&ctx.camera).await {
        Ok(stream) => stream,
        Err(err) => return failed(err.to_string()),
    };
    let settle = settle_point(ctx);
    let accepted = match wait_for_valid_frame(ctx, &settle.baseline, settle.readback, &stream).await
    {
        Ok(accepted) => accepted,
        Err(err) => return failed(err.to_string()),
    };
    let (sample, committed) = match record_sample(
        ctx,
        archive,
        best_position,
        &accepted,
        &settle,
        final_move,
        &stream,
        SampleRole::Verification,
    )
    .await
    {
        Ok(recorded) => recorded,
        Err(err) => return failed(err.to_string()),
    };
    let verdict = verification_verdict(
        sample.hfr,
        best_sampled_hfr(&run.samples),
        accepted.readback_after,
        best_position,
        ctx.config.safety.arrival_tolerance_steps,
        accepted.focuser.position_source.is_reliable(),
    );
    Verification {
        verified: verdict.is_ok(),
        frame_id: Some(committed.frame_id),
        hfr: Some(sample.hfr),
        position_readback: accepted.readback_after,
        measured_at: Some(accepted.window_end),
        reason: verdict.err(),
        aborted: false,
    }
}

// ---------------------------------------------------------------------------
// H5 and H6: the drift forecaster
// ---------------------------------------------------------------------------

/// The temperature a verified run may teach the FDF (H5, H6), or why it may
/// not: simulator runs never update a rig's state (D18); an unverified run,
/// a missing or stale reading, and the camera-sensor fallback teach nothing
/// (v6 §8.1, §10.7).
fn drift_learning_temperature(
    simulation: bool,
    verified: bool,
    ambient: &Observed<TemperatureReading>,
) -> Result<f64, &'static str> {
    if simulation {
        return Err("simulator_run");
    }
    if !verified {
        return Err("not_verified");
    }
    let Some(reading) = ambient.known() else {
        return Err("no_ambient_temperature");
    };
    if reading.camera_sensor_fallback {
        return Err("camera_sensor_fallback");
    }
    if !ambient.is_fresh() {
        return Err("temperature_stale");
    }
    if !reading.celsius.is_finite() {
        return Err("no_ambient_temperature");
    }
    Ok(reading.celsius)
}

/// Fold a verified best-focus read-back into `forecaster` at `measured_at`,
/// first re-baselining on every focuser position reset since its last
/// update (F-h), and snapshot it to `state_path`.
async fn fold_verified_focus(
    forecaster: &tokio::sync::Mutex<crate::focus::focus_drift::FocusDriftForecaster>,
    state_path: Option<&std::path::Path>,
    position: i32,
    temperature_c: f64,
    measured_at: DateTime<Utc>,
    resets: &[PositionReset],
) {
    let snapshot = {
        let mut guard = forecaster.lock().await;
        if let Some(last) = guard.last_update() {
            for reset in resets
                .iter()
                .filter(|reset| reset.at > last && reset.at <= measured_at)
            {
                guard.rebaseline(reset.shift());
            }
        }
        guard.update(f64::from(position), temperature_c, measured_at);
        guard.to_state()
    };
    if let Some(path) = state_path {
        if let Err(err) = crate::storage::focus_drift_repo::save_state(path, &snapshot) {
            tracing::warn!(
                "Failed to snapshot focus drift state to {}: {err}",
                path.display()
            );
        }
    }
}

/// Fold a verified, non-simulator autofocus result into the drift filter
/// and snapshot it (H5); measure the focuser scale from the run's V-curve.
///
/// Called only after verification: the filter models where best focus *is*,
/// so feeding it the endpoint of a run that never converged, or a position
/// no fresh frame confirmed, would teach it that best focus is wherever the
/// sweep gave up.
///
/// Every failure here is logged and swallowed. The forecaster is advisory; an
/// autofocus run that physically converged has succeeded whether or not a
/// prediction model got updated.
async fn record_drift_measurement(
    state: &AppState,
    position: i32,
    temperature_c: f64,
    measured_at: DateTime<Utc>,
    samples: &[FocusSample],
) {
    let Some(forecaster) = state.focus_drift.clone() else {
        return;
    };

    // Independent of the temperature: the V-curve measures the focuser's
    // mechanical scale, which has nothing to do with thermometers.
    measure_focuser_scale(state, &forecaster, samples).await;

    let since = {
        let guard = forecaster.lock().await;
        guard.last_update()
    };
    let resets = since
        .map(|since| position_resets_since(state, since))
        .unwrap_or_default();
    fold_verified_focus(
        &forecaster,
        state.focus_drift_state_path.as_deref(),
        position,
        temperature_c,
        measured_at,
        &resets,
    )
    .await;
}

/// The temperature the drift forecast is evaluated at (H6).
#[derive(Debug, Clone, PartialEq)]
struct DriftTemperature {
    celsius: Option<f64>,
    state: TemperatureState,
    source: Option<String>,
    /// The camera sensor stood in for an ambient or tube reading: the
    /// forecast is published as such, and nothing learns from it.
    camera_sensor_fallback: bool,
    age_ms: Option<u64>,
}

/// H6: the hub's ambient selection (weather, then a focuser probe whose role
/// is ambient or tube) when it has a fresh reading; otherwise the camera
/// sensor, flagged `camera_sensor_fallback`. The hub's own camera-sensor
/// fallback is fed only when something reads the camera, so it may be old:
/// the fallback is read live here, on a blocking thread (D3), which also
/// refreshes the hub's.
async fn drift_temperature(state: &AppState) -> DriftTemperature {
    let ambient = state.equipment.ambient_temperature();
    let now = state.equipment.clock().now_mono_ns();
    if let Some(reading) = ambient
        .fresh_value()
        .filter(|r| !r.camera_sensor_fallback && r.celsius.is_finite())
    {
        return DriftTemperature {
            celsius: Some(reading.celsius),
            state: TemperatureState::Measured,
            source: crate::storage::columns::name(&reading.source),
            camera_sensor_fallback: false,
            age_ms: ambient.age_ms(now),
        };
    }
    let (celsius, temperature_state, source) =
        crate::api::camera::read_temperature_blocking(state).await;
    DriftTemperature {
        camera_sensor_fallback: celsius.is_some(),
        celsius,
        state: temperature_state,
        source,
        age_ms: None,
    }
}

#[derive(Debug, Deserialize)]
pub struct DriftQuery {
    /// How far ahead to forecast, in seconds. Ten minutes by default — roughly
    /// a sub-exposure, which is the interval over which "will focus still be
    /// good?" is a question worth asking.
    pub horizon_s: Option<f64>,
}

/// GET /api/v1/focus/drift — current drift prediction.
///
/// Three distinct outcomes, kept distinct on purpose:
///
/// * `disabled` — `[focus.drift]` is off or misconfigured. The operator has to
///   do something.
/// * `uninitialized` — configured, but the filter has not seen enough measured
///   focus positions to say anything. Nothing to do but observe.
/// * `ok` — a real prediction with its uncertainty.
///
/// Collapsing these into one "unavailable" (which is what this endpoint used to
/// return unconditionally) tells the operator nothing about which of the three
/// they are in, and they call for different responses.
pub async fn get_drift_prediction(
    State(state): State<AppState>,
    axum::extract::Query(query): axum::extract::Query<DriftQuery>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let Some(forecaster) = state.focus_drift.clone() else {
        return Ok(Json(serde_json::json!({
            "status": "disabled",
            "detail": "Focus drift forecasting is off. Set [focus.drift] enabled = true in \
                       achernar.toml. It will start estimating immediately — optics are \
                       needed only to turn its estimate into refocus advice.",
        })));
    };

    let horizon_s = query
        .horizon_s
        .filter(|h| h.is_finite() && *h >= 0.0)
        .unwrap_or(600.0);

    // Temperature drives the forecast, so its provenance travels with the
    // answer (H6). An ambient or tube reading comes first; the camera sensor
    // is only a flagged fallback. A DSLR has no sensor thermometer: `missing`
    // is reported as missing rather than substituted, because a prediction
    // built on an invented temperature is indistinguishable from one built on
    // a real reading once it has been written down.
    let reading = drift_temperature(&state).await;
    let temperature_state = reading.state;
    let temperature_source = reading.source.clone();

    let Some(temperature_c) = reading.celsius else {
        let guard = forecaster.lock().await;
        let update_count = guard.update_count();
        drop(guard);
        return Ok(Json(serde_json::json!({
            "status": "unavailable",
            "detail": "No temperature reading, and drift is predicted from temperature. \
                       Attach a focuser or environment sensor that reports one.",
            "temperature_state": temperature_state,
            "update_count": update_count,
        })));
    };

    // Lock, predict, copy out, release — the SQLite write below must not hold
    // the filter (see the lock-order note on AppState).
    let (prediction, update_count, coefficient, drift_rate, threshold) = {
        let guard = forecaster.lock().await;
        (
            guard.predict(temperature_c, horizon_s),
            guard.update_count(),
            guard.temperature_coefficient(),
            guard.drift_rate(),
            guard.threshold(),
        )
    };

    if !prediction.initialized {
        return Ok(Json(serde_json::json!({
            "status": "uninitialized",
            "detail": "The filter has not seen enough measured focus positions yet. It learns \
                       from verified autofocus runs; run one or two more.",
            "update_count": update_count,
            "temperature_c": temperature_c,
            "temperature_state": temperature_state,
            "temperature_source": temperature_source,
            "camera_sensor_fallback": reading.camera_sensor_fallback,
            "refocus_threshold_steps": threshold.map(|(steps, _)| steps),
        })));
    }

    // Audit the evaluation. A failed write must not fail the request: the
    // operator asked what focus is doing, not to write a row.
    //
    // `focus_drift_events` is FDF ground truth, and its `temperature` column
    // has no source or fallback flag beside it. An evaluation at the camera
    // sensor's temperature (the H6 fallback) is therefore not recorded: a row
    // would be indistinguishable from one at an ambient reading. It is still
    // published, flagged, in the body.
    let event_recorded = if reading.camera_sensor_fallback {
        tracing::debug!(
            "focus_drift_event not recorded: evaluated at the camera-sensor fallback temperature"
        );
        false
    } else {
        match crate::storage::focus_drift_repo::record_event(
            state.local_store.pool(),
            None,
            &prediction,
            Some(temperature_c),
        )
        .await
        {
            Ok(_) => true,
            Err(err) => {
                tracing::warn!("Failed to record focus_drift_event: {err}");
                false
            }
        }
    };

    let (k, k_sigma) = coefficient;
    let mut body = serde_json::json!({
        "status": "ok",

        // ── Estimation. Always present once the filter has data; none of it
        //    needs to know anything about the optics. ────────────────────────
        "predicted_drift_steps": prediction.predicted_drift_steps,
        // Half-width of the 95% interval. Published beside the point estimate
        // because a drift of 40 +/- 200 steps and one of 40 +/- 5 justify very
        // different actions, and a bare point estimate hides which you have.
        "confidence_interval_steps": prediction.confidence_interval,
        "temperature_coefficient_steps_per_c": k,
        "temperature_coefficient_sigma": k_sigma,
        "drift_rate_steps_per_s": drift_rate,
        "update_count": update_count,
        "horizon_s": horizon_s,
        "temperature_c": temperature_c,
        "temperature_state": temperature_state,
        "temperature_source": temperature_source,
        "temperature_age_ms": reading.age_ms,
        "camera_sensor_fallback": reading.camera_sensor_fallback,
        "event_recorded": event_recorded,

        // ── Decision. Null when no refocus threshold is known. ─────────────
        "exceed_probability": prediction.exceed_probability,
        "refocus_recommended": prediction.refocus_recommended,
        "time_to_refocus_s": prediction.time_to_refocus_s,
        "refocus_threshold_steps": threshold.map(|(steps, _)| steps),
        "threshold_source": threshold.map(|(_, source)| source),
        "advice": prediction.advice,
    });

    // When advice is unavailable, say what to do about it — next to the
    // estimate, which is the evidence that doing it is worth the trouble.
    // Silence would be a weaker prompt than "here is what I already know".
    if matches!(
        prediction.advice,
        crate::focus::focus_drift::AdviceStatus::ThresholdUnknown
    ) {
        body["advice_detail"] = serde_json::json!(
            "Refocus advice unavailable: no refocus threshold. Estimation is unaffected — \
             the drift and coefficient above are real. Supply [focus.drift] \
             steps_per_micron (with f_ratio here or focal_ratio on the rigs row), or set \
             rigs.pixel_size_um and let the next autofocus V-curve measure the focuser \
             scale for you."
        );
    }

    Ok(Json(body))
}

/// Measure the focuser's steps-per-micron from the run that just finished, and
/// adopt the refocus threshold that falls out of it.
///
/// `steps_per_micron` is mechanical — turns of a knob against a drawtube. It is
/// printed on nothing, few operators know it for their own focuser, and
/// requiring it by hand is what kept the refocus threshold unset. The sweep the
/// daemon has just run measured it: the slope of a V-curve wing, against the
/// focal ratio and the sensor pixel pitch, gives it directly. See
/// [`crate::focus::focus_drift::steps_per_micron_from_vcurve`] for the
/// derivation and the error terms it neglects.
///
/// Everything here is best-effort. A rig that has not filled in
/// `rigs.pixel_size_um` simply gets no measurement, and the forecaster carries
/// on estimating without a threshold — which is the whole point of the split.
/// Simulator runs never reach this (D18).
async fn measure_focuser_scale(
    state: &AppState,
    forecaster: &Arc<tokio::sync::Mutex<crate::focus::focus_drift::FocusDriftForecaster>>,
    samples: &[FocusSample],
) {
    use crate::focus::classical_focus_solver::ClassicalFocusSolver;
    use crate::focus::focus_drift::{steps_per_micron_from_vcurve, DriftConfig};

    // A configured threshold outranks any measurement, so do not spend the
    // fit — but do keep measuring if the only threshold we have is a previous
    // measurement, since a later sweep is a better one.
    {
        let guard = forecaster.lock().await;
        if let Some((_, source)) = guard.threshold() {
            if source != crate::focus::focus_drift::ThresholdSource::MeasuredVCurve {
                return;
            }
        }
    }

    let Some((f_ratio, pixel_size_um)) = state
        .focus_config
        .drift
        .vcurve_inputs(state.rig_optics.as_ref())
    else {
        tracing::debug!(
            "Focuser scale not measured: needs a focal ratio ([focus.drift] f_ratio or \
             rigs.focal_ratio) and rigs.pixel_size_um"
        );
        return;
    };

    // Samples come from the run that just finished, passed in directly rather
    // than re-read from `agent.history()` — that would depend on whether the
    // agent has pushed the run yet, which is not this function's business.
    if samples.len() < 5 {
        return;
    }
    let (positions, hfrs): (Vec<i32>, Vec<f64>) =
        samples.iter().map(|s| (s.position, s.hfr)).unzip();

    // The V fit gives both wing slopes. Average their magnitudes: they measure
    // the same mechanical scale from opposite sides, and averaging halves the
    // effect of an asymmetric sweep or a little backlash on one side.
    let Some(fit) = ClassicalFocusSolver::fit_v_curve(&positions, &hfrs) else {
        return;
    };
    let (Some(left), Some(right)) = (fit.coefficients.first(), fit.coefficients.get(2)) else {
        return;
    };
    let slope = 0.5 * (left.abs() + right.abs());

    let Some(steps_per_micron) = steps_per_micron_from_vcurve(slope, f_ratio, pixel_size_um) else {
        return;
    };

    let threshold = DriftConfig::threshold_from_optics(
        f_ratio,
        state.focus_config.drift.wavelength_nm_or_default(),
        steps_per_micron,
    );

    let adopted = {
        let mut guard = forecaster.lock().await;
        guard.adopt_measured_threshold(threshold)
    };
    if !adopted {
        return;
    }

    tracing::info!(
        "Focuser scale measured from the V-curve: {steps_per_micron:.3} steps/µm \
         (wing slope {slope:.6} px/step, f/{f_ratio}, {pixel_size_um} µm pixels) \
         → refocus threshold {threshold:.1} steps. An estimate; a value in \
         [focus.drift] would override it."
    );

    // Persist it against the focuser, so a restart does not re-measure and the
    // operator never has to enter it.
    let focuser_name = state
        .focuser_name
        .lock()
        .ok()
        .and_then(|n| n.clone())
        .unwrap_or_else(|| "unknown".to_string());
    if let Err(err) = crate::storage::device_profiles_repo::record_focuser_scale(
        state.local_store.pool(),
        &state.rig_id,
        &focuser_name,
        steps_per_micron,
    )
    .await
    {
        tracing::warn!("Failed to persist measured focuser scale: {err}");
    }
}

/// GET /api/v1/focus/drift/history — recent forecaster evaluations.
pub async fn get_drift_history(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match crate::storage::focus_drift_repo::recent_events(state.local_store.pool(), 100).await {
        Ok(rows) => Ok(Json(serde_json::json!({ "events": rows }))),
        Err(err) => {
            tracing::error!("Failed to read focus_drift_events: {err}");
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

// ---------------------------------------------------------------------------
// The physical-autofocus gate (D18) and the stream
// ---------------------------------------------------------------------------

/// Physical autofocus remains unavailable until the analysis, timing,
/// settings, stellar-sweep and driver qualification gates have passed.
fn simulation_focus_allowed(camera_driver: Option<&str>, simulator_focuser: bool) -> bool {
    camera_driver == Some("sim") && simulator_focuser
}

fn validate_simulation_focus_run(state: &AppState) -> crate::error::Result<()> {
    let simulator_focuser = {
        let handle = crate::api::equipment::focuser_slot(state).ok_or(FocusError::NoFocuser)?;
        handle.identity().driver == DriverId::Sim
    };
    let camera_driver = state.camera_runtime.read().unwrap().driver_id.clone();
    if !simulation_focus_allowed(camera_driver.as_deref(), simulator_focuser) {
        return Err(HardwareError::Unsupported(
            "Automatic physical autofocus is unavailable pending A03/A06/A13/A15/A23 qualification; this stage permits a simulator camera with a simulator focuser only"
                .to_string(),
        )
        .into());
    }
    let camera = state.camera.lock().unwrap();
    let camera = camera.as_ref().ok_or_else(|| {
        HardwareError::DeviceNotFound("No camera connected for autofocus".to_string())
    })?;
    if camera.supports_still_capture() {
        return Err(HardwareError::Unsupported(
            "DSLR focus requires a dedicated archived still exposure; live-view and still-stream callbacks cannot supply a complete focus bundle"
                .to_string(),
        )
        .into());
    }
    Ok(())
}

fn is_focus_sensor_format(format: OlsStreamType) -> bool {
    matches!(format, OlsStreamType::Mono8 | OlsStreamType::Mono16)
}

/// Blocking: OLS stream queries and start (call from a blocking thread).
fn ensure_streaming(state: &AppState) -> crate::error::Result<()> {
    let mut camera = state.camera.lock().unwrap();
    let camera = camera.as_mut().ok_or_else(|| {
        HardwareError::DeviceNotFound("No camera connected for autofocus".to_string())
    })?;
    let formats = camera
        .formats()
        .map_err(|err| HardwareError::Communication(err.to_string()))?;
    if *state.streaming.lock().unwrap() {
        let active = state.camera_runtime.read().unwrap().active_format_index;
        if active
            .and_then(|index| formats.get(index))
            .is_some_and(|format| is_focus_sensor_format(format.format))
        {
            return Ok(());
        }
        return Err(HardwareError::Unsupported(
            "Autofocus requires an exact mono sensor stream; preview, RGB and CFA analysis remain unqualified"
                .to_string(),
        )
        .into());
    }
    let (format_index, format) = formats
        .iter()
        .enumerate()
        .find(|(_, format)| is_focus_sensor_format(format.format))
        .ok_or_else(|| {
            HardwareError::Unsupported(
                "Autofocus requires an exact mono sensor stream; preview, RGB and CFA analysis remain unqualified"
                    .to_string(),
            )
        })?;
    state.clear_camera_observations();
    let generation = state.current_camera_generation();
    camera
        .start_stream(
            format,
            state.latest_frame.clone(),
            state.camera_config.saturation_adu,
            generation,
            state.camera_generation.clone(),
        )
        .map_err(|err| HardwareError::Communication(err.to_string()))?;
    *state.streaming.lock().unwrap() = true;
    {
        let mut runtime = state.camera_runtime.write().unwrap();
        runtime.activity = CameraActivity::Streaming;
        runtime.streaming = true;
        runtime.active_format_index = Some(format_index);
        runtime.last_error = None;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Archiving a focus frame
// ---------------------------------------------------------------------------

/// Archive the exact simulator sensor buffer and its analysis derivative.
/// A missing local commit aborts measurement before a sample is accepted.
#[allow(clippy::too_many_arguments)]
async fn persist_observed_frame(
    state: &AppState,
    camera: &dyn FocusCamera,
    archive: ArchiveReservation,
    position: i32,
    rig_id: &str,
    filter_name: Option<&str>,
    observed: &ObservedFrame,
    stream: &StreamInfo,
    temperature: (Option<f64>, TemperatureState, Option<String>),
    extra_quality: serde_json::Map<String, Value>,
    equipment: Option<FocusFrameEquipment>,
) -> crate::error::Result<CommittedCapture> {
    if observed.frame.stream_generation != camera.generation() {
        return Err(FocusError::RunFailed(
            "Autofocus observation belongs to an inactive camera stream".to_string(),
        )
        .into());
    }
    let focuser_name = state.focuser_name.lock().ok().and_then(|name| name.clone());
    let (temperature, temperature_state, temperature_source) = temperature;
    let pending = simulation_focus_bundle(
        observed,
        SimulationFocusMetadata {
            position,
            rig_id: rig_id.to_string(),
            filter_name: filter_name.map(str::to_string),
            camera_name: stream.camera_name.clone(),
            focuser_name,
            cadence_s: stream.exposure_s,
            temperature,
            temperature_state,
            temperature_source,
            extra_quality,
            equipment,
        },
    )?;
    let capture_id = pending.capture_id.clone();
    let frame_id = pending.frame_id.clone();
    state
        .capture_store
        .begin_attempt(
            state.local_store.pool(),
            &capture_id,
            None,
            rig_id,
            &serde_json::json!({
                "capture_type": "focus",
                "acquisition_domain": "simulation",
                "training_label_eligible": false,
                "focuser_position": position,
                "stream_generation": observed.frame.stream_generation,
                "frame_counter": observed.frame_counter,
            }),
        )
        .await?;
    let committed = archive.persist(pending).await?;
    if committed.capture_id != capture_id || committed.frame_id != frame_id {
        return Err(FocusError::RunFailed(
            "Autofocus archive acknowledgement belongs to a different capture".to_string(),
        )
        .into());
    }
    Ok(committed)
}

struct SimulationFocusMetadata {
    position: i32,
    rig_id: String,
    filter_name: Option<String>,
    camera_name: Option<String>,
    focuser_name: Option<String>,
    cadence_s: f64,
    temperature: Option<f64>,
    temperature_state: TemperatureState,
    temperature_source: Option<String>,
    /// Keys merged into `quality_json`: the A06 evidence, the H6 ambient
    /// reading and the FOP vector (H7).
    extra_quality: serde_json::Map<String, Value>,
    /// The frame's equipment record from the hub's window (H1, D7); `None`
    /// keeps the bare record (unit tests of the bundle itself).
    equipment: Option<FocusFrameEquipment>,
}

/// A focus frame's equipment record: the capture path's assembly (H1, D7)
/// over the A06 interval the frame was accepted on. Nothing here calls a
/// device; it reads the hub's rings and journal.
struct FocusFrameEquipment {
    frame_id: String,
    /// `acquisition_json.equipment`; its `window` is the frame's
    /// `exposure_device_windows` row.
    block: Value,
    cards: Vec<crate::fits::FitsCard>,
    filter_name: Option<String>,
    /// The window's reliable, unmoved position, else the A06 read-back
    /// (taken after the exposure and equal to the one before it).
    focuser_position: Option<i64>,
}

/// Assemble a focus frame's equipment record (FQE and FOP ground truth),
/// and the finaliser's plan for its window row.
fn focus_frame_equipment(
    ctx: &RunContext,
    accepted: &AcceptedFrame,
    stream: &StreamInfo,
) -> (
    FocusFrameEquipment,
    crate::api::camera::equipment::FinalisePlan,
) {
    use crate::api::camera::equipment::{assemble, AssemblyInput, ExposureInterval, FinalisePlan};
    let state = &ctx.state;
    let interval = ExposureInterval {
        start: accepted.window_start,
        end: accepted.window_end,
        clock_quality: accepted.timing_basis,
        basis: "a06_accepted_window",
        exposure_s: stream.exposure_s,
        frame_interval_s: Some(stream.frame_interval_s).filter(|s| s.is_finite() && *s > 0.0),
    };
    let input = AssemblyInput {
        frame_id: Uuid::new_v4().to_string(),
        client_filter: ctx.filter.name.clone(),
        blocks_at_start: state.equipment.resume().active(),
        source: ctx.op.command.source.clone(),
        command_id: Some(ctx.op.command_id()),
    };
    let recorded = assemble(
        &state.equipment,
        state.equipment_ctx.guider_windows().as_ref(),
        &interval,
        &input,
    );
    let equipment = FocusFrameEquipment {
        frame_id: input.frame_id.clone(),
        block: recorded.block,
        cards: recorded.cards,
        filter_name: recorded.filter_name,
        focuser_position: recorded
            .focuser_position
            .or(accepted.readback_after.map(i64::from)),
    };
    let plan = FinalisePlan {
        input,
        interval,
        finalised_at_commit: recorded.row.finalised,
    };
    (equipment, plan)
}

/// The permitted analysis domain is deliberately small while A03 is pending:
/// mono identity conversion from an exact, uncompressed simulator buffer.
fn simulation_focus_bundle(
    observed: &ObservedFrame,
    meta: SimulationFocusMetadata,
) -> crate::error::Result<PendingCapture> {
    if !meta.cadence_s.is_finite() || meta.cadence_s <= 0.0 {
        return Err(FocusError::RunFailed("Invalid simulator frame cadence".to_string()).into());
    }
    let frame = &observed.frame;
    if !is_focus_sensor_format(frame.format)
        || frame.bayer != OlsBayerType::Na
        || frame.source.is_some()
    {
        return Err(HardwareError::Unsupported(
            "Autofocus cannot accept preview, selected-file or CFA data as a complete qualified sensor exposure"
                .to_string(),
        )
        .into());
    }
    let bit_depth = if frame.format == OlsStreamType::Mono8 {
        8
    } else {
        16
    };
    let expected_bytes = usize::try_from(frame.width)
        .ok()
        .zip(usize::try_from(frame.height).ok())
        .and_then(|(width, height)| width.checked_mul(height))
        .and_then(|pixels| pixels.checked_mul(bit_depth as usize / 8));
    if frame.width <= 0
        || frame.height <= 0
        || expected_bytes != Some(frame.data.len())
        || observed.frame_counter != frame.frame_counter
        || observed.capture_unix_time != frame.unix_timestamp
    {
        return Err(FocusError::RunFailed(
            "Autofocus sensor buffer does not match its observation envelope".to_string(),
        )
        .into());
    }
    let capture_micros = observed.capture_unix_time * 1_000_000.0;
    let captured_at = if capture_micros.is_finite()
        && capture_micros >= i64::MIN as f64
        && capture_micros <= i64::MAX as f64
    {
        chrono::DateTime::<chrono::Utc>::from_timestamp_micros(capture_micros.round() as i64)
    } else {
        None
    }
    .ok_or_else(|| FocusError::RunFailed("Invalid simulator frame timestamp".to_string()))?;
    let luma = crate::vision::quality::luma_from_frame(frame)?;
    let mut quality = autofocus_quality_json(
        meta.position,
        &meta.rig_id,
        meta.filter_name.as_deref(),
        meta.camera_name.as_deref(),
        meta.focuser_name.as_deref(),
        observed.frame_counter,
        observed.capture_unix_time,
        &observed.observation,
        &luma,
        meta.temperature_state,
        meta.temperature_source.clone(),
    );
    if let Value::Object(map) = &mut quality {
        for (key, value) in meta.extra_quality {
            map.entry(key).or_insert(value);
        }
    }
    let fits_meta = crate::fits::FitsMeta {
        exposure_s: meta.cadence_s,
        gain: 0,
        temperature_c: meta.temperature.unwrap_or(f64::NAN),
        binning: 1,
        instrument: meta
            .camera_name
            .clone()
            .unwrap_or_else(|| "Simulation".into()),
        date_obs: captured_at.to_rfc3339(),
        iso: None,
        shutter_label: None,
        source_file: None,
        calibration: None,
    };
    let cards = meta
        .equipment
        .as_ref()
        .map(|equipment| equipment.cards.as_slice())
        .unwrap_or_default();
    let fits_bytes =
        crate::fits::encode_capture_with(frame, &fits_meta, cards).map_err(|error| {
            crate::error::StorageError::Database(format!("encode autofocus sensor FITS: {error}"))
        })?;
    let analysis_bytes = encode_simulation_luma(&luma)?;
    let descriptor = frame.analysis_sensor_descriptor()?;
    let linear = crate::processing::debayer::analyze(&crate::processing::debayer::SensorFrame {
        descriptor,
        data: &frame.data,
        masks: Default::default(),
    })
    .map_err(|error| FocusError::RunFailed(format!("Simulation analysis failed: {error}")))?;
    if observed.observation.analysis_descriptor.as_ref() != Some(&linear.descriptor) {
        return Err(FocusError::RunFailed("Simulator focus label does not match the exact archived analysis source and configuration".into()).into());
    }
    let linear_assets = crate::storage::analysis_assets::analysis_assets(
        &linear,
        "sensor.fits",
        &fits_bytes,
        &frame.data,
    )
    .map_err(|error| {
        FocusError::RunFailed(format!("Simulation analysis archive failed: {error}"))
    })?;
    let (frame_id, filter_name, focuser_position, equipment_block) = match meta.equipment {
        Some(equipment) => (
            equipment.frame_id,
            equipment.filter_name,
            equipment.focuser_position,
            Some(equipment.block),
        ),
        None => (
            uuid::Uuid::new_v4().to_string(),
            meta.filter_name,
            Some(meta.position as i64),
            None,
        ),
    };
    let mut acquisition = serde_json::json!({
        "acquisition_domain": "simulation",
        "training_label_eligible": false,
        "analysis_version": crate::processing::debayer::PROCESSING_VERSION,
        "analysis_descriptor": "analysis/analysis.json",
        "stream_generation": frame.stream_generation,
        "frame_counter": frame.frame_counter,
        "stream_format": format!("{:?}", frame.format),
        "timing_state": "simulator_delivery_timestamp_only",
        "exposure_state": "simulator_cadence_readback_without_settings_write",
        "gain_state": "unavailable",
        "retention": crate::storage::capture_retention::RetentionDecision::retaining(false),
        "quality": quality,
    });
    if let Some(block) = equipment_block {
        acquisition["equipment"] = block;
    }
    Ok(PendingCapture {
        capture_id: uuid::Uuid::new_v4().to_string(),
        frame_id,
        session_id: None,
        rig_id: meta.rig_id,
        camera_name: meta.camera_name,
        driver_name: Some("sim".into()),
        capture_type: "focus".into(),
        // The simulator supplies a delivery timestamp, not a physical shutter
        // bracket. Preserve it without manufacturing an earlier start time.
        exposure_started_at: observed.capture_unix_time,
        exposure_completed_at: observed.capture_unix_time,
        requested_exposure_s: meta.cadence_s,
        requested_gain: None,
        applied_exposure_s: meta.cadence_s,
        iso: None,
        shutter_label: None,
        offset_val: None,
        binning: 1,
        filter_name,
        focuser_position,
        temperature: meta.temperature,
        temperature_source: meta.temperature_source,
        temperature_state: meta.temperature_state,
        acquisition,
        assets: {
            let mut assets = vec![
                PendingAsset::new(
                    AssetRole::SensorFits,
                    "sensor.fits",
                    "application/fits",
                    fits_bytes,
                )
                .with_dimensions(frame.width as i64, frame.height as i64, bit_depth)
                .with_provenance(serde_json::json!({
                    "source": "exact_simulator_sensor_buffer",
                    "acquisition_domain": "simulation",
                    "training_label_eligible": false,
                    "stream_format": format!("{:?}", frame.format),
                })),
                PendingAsset::new(
                    AssetRole::AnalysisLuma,
                    "luma.png",
                    "image/png",
                    analysis_bytes,
                )
                .derived_from(AssetRole::SensorFits)
                .with_dimensions(frame.width as i64, frame.height as i64, 16)
                .with_provenance(serde_json::json!({
                    "analysis_version": "mono_identity_v1",
                    "encoding": "uint16 PNG; mono sample values preserved without scaling",
                    "acquisition_domain": "simulation",
                    "training_label_eligible": false,
                })),
            ];
            assets.extend(linear_assets);
            assets
        },
    })
}

fn encode_simulation_luma(
    luma: &crate::vision::quality::LumaFrame,
) -> crate::error::Result<Vec<u8>> {
    let pixels: Option<Vec<u16>> = luma
        .pixels
        .iter()
        .map(|value| {
            (value.is_finite() && *value >= 0.0 && *value <= u16::MAX as f32)
                .then(|| value.round() as u16)
        })
        .collect();
    let buffer = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(
        luma.width as u32,
        luma.height as u32,
        pixels.ok_or_else(|| {
            FocusError::RunFailed("Simulator luma contains invalid sample values".to_string())
        })?,
    )
    .ok_or_else(|| FocusError::RunFailed("Invalid simulator luma dimensions".to_string()))?;
    let mut encoded = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageLuma16(buffer)
        .write_to(&mut encoded, image::ImageOutputFormat::Png)
        .map_err(|error| {
            crate::error::StorageError::Database(format!("encode autofocus analysis PNG: {error}"))
        })?;
    Ok(encoded.into_inner())
}

/// Everything about an autofocus frame that is not a column of its own.
///
/// This is the provenance record two dataset builders read. `frames.quality_json`
/// is where `ml/data_pipeline/build_focus_quality_dataset.py` finds the
/// saturation point it gates on, and where the FOP's builder will find the
/// Quality and Sky feature groups declared in `ml/contracts/fop_features.py`.
///
/// Split out of `persist_observed_frame` so it can be tested without an
/// `AppState`, a camera, or a database — the keys emitted here are a contract
/// with the Python side, and a contract nothing asserts is a contract that
/// drifts.
#[allow(clippy::too_many_arguments)]
fn autofocus_quality_json(
    position: i32,
    rig_id: &str,
    filter_name: Option<&str>,
    camera_name: Option<&str>,
    focuser_name: Option<&str>,
    frame_counter: i32,
    capture_unix_time: f64,
    observation: &SharedObservation,
    luma: &crate::vision::quality::LumaFrame,
    temperature_state: crate::storage::capture_store::TemperatureState,
    temperature_source: Option<String>,
) -> serde_json::Value {
    serde_json::json!({
            "source": "autofocus_simulation",
            "acquisition_domain": "simulation",
            "training_label_eligible": false,
            "analysis": luma.analysis_descriptor,
            "rig_id": rig_id,
            "camera_id": camera_name,
            "focuser_id": focuser_name,
            "filter_name": filter_name,
            "focuser_position": position,
            // Provenance: the camera frame these pixels and this label came from.
            "frame_counter": frame_counter,
            "capture_unix_time": capture_unix_time,
            "hfr_px": observation.hfr_px,
            // 2*HFR (HFD), not a fitted FWHM — named so the dataset builder never
            // mistakes it for a PSF FWHM.
            "hfd_px": observation.hfd_px,
            "fwhm_px": observation.fwhm_px,
            "star_count": observation.star_count,
            "star_count_used": observation.star_count_used,
            "status": observation.status,
            "quality_uncertainty": observation.quality_uncertainty,
            "hfr_log_sigma": observation.hfr_log_sigma,
            // Deprecated top-level mirrors (the FOP contract names them);
            // `null` when not measured. `background` is authoritative: method,
            // configuration hash, level/noise validity, units and support.
            // Rows without it are legacy-unverified.
            "background_adu": observation.background_adu,
            "background_sigma": observation.background_sigma,
            "saturation_fraction": observation.saturation_fraction,
            "background": observation.background,
            "detection": observation.detection,
            // The FQE's own reading of this frame — the Quality(3) group of the
            // Focus Offset Predictor's input (ml/contracts/fop_features.py).
            //
            // The vision loop computes these on every gated frame and, until
            // now, this function dropped them: the shadow prediction lived only
            // as long as the in-memory SharedObservation. That is survivable for
            // the FQE, whose pixels are on disk and can be re-scored at any
            // time, and fatal for the FOP, whose input *is* this derived state.
            // A sweep recorded without it can never be turned into FOP rows.
            //
            // Recording the version and calibration flag beside the values is
            // not bookkeeping for its own sake: two runs a week apart may have
            // been scored by different model versions, and a shadow evaluation
            // that pools them without knowing which is measuring nothing.
            "fqe_hfr_px": observation.fqe_hfr_px,
            "fqe_sigma_log_hfr": observation.fqe_sigma_log_hfr,
            "fqe_star_count": observation.fqe_star_count,
            "fqe_model_version": observation.fqe_model_version,
            "fqe_calibrated": observation.fqe_calibrated,
            "max_adu": luma.max_adu,
            // Saturation provenance, written so the dataset builder can filter
            // on it. `saturation_fraction: 0.0` and `saturated: 0` mean "clean
            // frame" only when the gate actually ran; on a body with no declared
            // saturation point they mean nothing was tested, and every clipped
            // core is in the HFR this row labels.
            "saturation_adu": luma.saturation_adu,
            "saturation_gate_applied": observation.saturation_gate.was_applied(),
            // Provenance for the temperature in the column beside this blob.
            // `measured` / `estimated` / `missing` — a drift model trained on a
            // mixture of the three, unable to tell them apart, is learning noise.
            "temperature_state": temperature_state,
            "temperature_source": temperature_source,
            "observation_valid": observation.observation_valid,
    })
}

fn connected_rig_id(state: &AppState) -> String {
    let camera = state
        .camera_name
        .lock()
        .ok()
        .and_then(|name| name.clone())
        .unwrap_or_else(|| "unknown-camera".to_string());
    let focuser = state
        .focuser_name
        .lock()
        .ok()
        .and_then(|name| name.clone())
        .unwrap_or_else(|| "unknown-focuser".to_string());
    format!("{camera}::{focuser}")
}

/// `temperature` is the camera's own reading (FOP contract #13), `None` when
/// nothing could measure one. Left NULL rather than filled with a default: a
/// column that mixes readings with stand-ins is worse than a column with holes
/// in it — the holes are at least visible.
fn sample_from_observation(
    position: i32,
    observation: &SharedObservation,
    committed: &CommittedCapture,
    temperature: Option<f64>,
) -> crate::error::Result<FocusSample> {
    if committed.status != "committed"
        || committed.capture_id.trim().is_empty()
        || committed.frame_id.trim().is_empty()
        || !committed
            .assets
            .iter()
            .any(|asset| asset.role == "sensor_fits" && asset.byte_length > 0)
    {
        return Err(FocusError::RunFailed(
            "Autofocus requires a committed native sensor bundle before accepting a sample"
                .to_string(),
        )
        .into());
    }
    let hfr = match observation.hfr_px {
        Some(hfr) if observation.observation_valid && (hfr as f64).is_finite() && hfr > 0.0 => {
            hfr as f64
        }
        _ => {
            return Err(FocusError::RunFailed(
                "Vision observation did not contain a valid focus metric".to_string(),
            )
            .into());
        }
    };

    Ok(FocusSample {
        position,
        hfr,
        hfd: observation.hfd_px.map(|value| value as f64),
        fwhm: observation.fwhm_px.map(|value| value as f64),
        star_count: observation.star_count,
        // The reciprocal focus score is gone: the curve fit uses raw HFR, not a
        // display score, so we no longer fabricate a per-sample quality_score.
        quality_score: None,
        temperature,
        timestamp: observation.timestamp,
        frame_id: Some(committed.frame_id.clone()),
    })
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

/// Persist a run, logging rather than failing: a database fault must not
/// strand the focuser mid-run.
async fn persist_run_logged(ctx: &RunContext, run: &FocusRunContext) {
    let provenance = lock(&ctx.ledger).provenance.clone();
    if let Err(err) = persist_focus_run(&ctx.state.local_store, run, &provenance).await {
        tracing::error!("Failed to persist autofocus run {}: {err}", run.run_id);
    }
}

/// Write a run and its samples. Each sample carries its H3 provenance from
/// `provenance` (keyed by frame id); a sample without one keeps NULLs.
async fn persist_focus_run(
    store: &LocalStore,
    run: &FocusRunContext,
    provenance: &HashMap<String, FocusSampleProvenance>,
) -> crate::error::Result<()> {
    let run_id = run.run_id.to_string();
    let mut transaction = store.pool().begin().await.map_err(storage_error)?;
    // The night this run belongs to, derived from when it started rather than
    // when it finished: a sweep that begins at 23:55 and ends at 00:05 is one
    // run on one night, and taking the end time would file it under the next.
    let night_id = crate::storage::night::night_id(run.started_at);
    sqlx::query(
        r#"
        INSERT INTO focus_runs (
            id, rig_id, filter_name, start_position, best_position, best_hfr,
            state, temperature, started_at, completed_at, night_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            rig_id = excluded.rig_id,
            filter_name = excluded.filter_name,
            start_position = excluded.start_position,
            best_position = excluded.best_position,
            best_hfr = excluded.best_hfr,
            state = excluded.state,
            temperature = excluded.temperature,
            started_at = excluded.started_at,
            completed_at = excluded.completed_at,
            night_id = excluded.night_id
        "#,
    )
    .bind(&run_id)
    .bind(&run.rig_id)
    .bind(&run.filter_name)
    .bind(run.start_position)
    .bind(run.best_position)
    .bind(run.best_hfr)
    .bind(format!("{:?}", run.state))
    .bind(run.temperature)
    .bind(run.started_at.to_rfc3339())
    .bind(chrono::Utc::now().to_rfc3339())
    .bind(&night_id)
    // `session_id` is deliberately not bound. Nothing in the daemon opens a
    // session yet — `storage::sessions_repo` is an empty stub — so binding one
    // would mean inventing an id that references no row. The column is in the
    // schema because `environment_log` and the sidecar importer both carry a
    // session_id, and a run recorded during an imported session needs somewhere
    // to say so. Until sessions exist, night_id is the join key that works.
    .execute(&mut *transaction)
    .await
    .map_err(storage_error)?;

    // Make retries/idempotent state updates exact. This delete and all inserts
    // share the transaction, so an individual sample failure cannot leave a
    // half-written run or duplicate the samples from an earlier attempt.
    sqlx::query("DELETE FROM focus_samples WHERE run_id = ?")
        .bind(&run_id)
        .execute(&mut *transaction)
        .await
        .map_err(storage_error)?;

    for sample in &run.samples {
        let row = FocusSampleRow {
            run_id: run_id.clone(),
            position: sample.position,
            hfr: sample.hfr,
            hfd: sample.hfd,
            fwhm: sample.fwhm,
            star_count: sample.star_count,
            quality_score: sample.quality_score,
            temperature: sample.temperature,
            captured_at: sample.timestamp,
            frame_id: sample.frame_id.clone(),
            provenance: sample
                .frame_id
                .as_ref()
                .and_then(|frame_id| provenance.get(frame_id))
                .cloned()
                .unwrap_or_default(),
        };
        crate::storage::focus_samples_repo::insert(&mut *transaction, &row)
            .await
            .map_err(storage_error)?;
    }

    transaction.commit().await.map_err(storage_error)?;
    Ok(())
}

/// Write a `focus_actions` row and, for the move to best focus, its H4
/// verification. Returns the action id.
async fn persist_focus_action(
    store: &LocalStore,
    action: &FocusActionRow,
    verification: Option<&FocusVerification>,
) -> crate::error::Result<i64> {
    let id = crate::storage::focus_actions_repo::insert(store.pool(), action)
        .await
        .map_err(storage_error)?;
    if let Some(verification) = verification {
        crate::storage::focus_actions_repo::record_verification(store.pool(), id, verification)
            .await
            .map_err(storage_error)?;
    }
    Ok(id)
}

fn storage_error(err: sqlx::Error) -> AchernarError {
    crate::error::StorageError::Database(err.to_string()).into()
}

fn focus_api_error(err: AchernarError) -> ApiError {
    let (status, code) = match &err {
        AchernarError::Focus(FocusError::NoFocuser) => {
            (StatusCode::PRECONDITION_FAILED, "focuser_not_connected")
        }
        AchernarError::Focus(FocusError::SafetyViolation(_)) => {
            (StatusCode::BAD_REQUEST, "focus_safety_violation")
        }
        AchernarError::Focus(_) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            "focus_measurement_unavailable",
        ),
        AchernarError::Hardware(HardwareError::DeviceNotFound(_)) => (
            StatusCode::PRECONDITION_FAILED,
            "focus_device_not_connected",
        ),
        AchernarError::Hardware(HardwareError::Unsupported(_)) => {
            (StatusCode::PRECONDITION_FAILED, "focus_unqualified")
        }
        AchernarError::Hardware(HardwareError::Timeout(_)) => {
            (StatusCode::GATEWAY_TIMEOUT, "focus_timeout")
        }
        AchernarError::Hardware(_) => (StatusCode::BAD_GATEWAY, "focus_hardware_error"),
        AchernarError::Storage(_) => (StatusCode::INTERNAL_SERVER_ERROR, "focus_archive_error"),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, "focus_internal_error"),
    };
    ApiError::new(status, code, err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::commands::GuiderCmd;
    use crate::equipment::snapshot::{GuiderSnapshot, PositionSource, TemperatureSourceKind};
    use crate::equipment::testing::fake_handles::{fake_focuser, fake_guider, fake_weather};
    use crate::equipment::types::{ObsState, Source, TemperatureRole};
    use crate::storage::capture_store::TemperatureState;
    use crate::vision::quality::LumaFrame;

    fn simulator_observation() -> ObservedFrame {
        let mut observation = SharedObservation::empty_for_test();
        observation.observation_valid = true;
        observation.hfr_px = Some(3.0);
        observation.star_count = 1;
        let mut observed = ObservedFrame {
            frame_counter: 7,
            capture_unix_time: 1_785_636_000.0,
            frame: Arc::new(crate::hardware::camera::FrameData {
                stream_generation: 1,
                format: OlsStreamType::Mono16,
                bayer: OlsBayerType::Na,
                frame_counter: 7,
                unix_timestamp: 1_785_636_000.0,
                width: 2,
                height: 2,
                data: [0u16, 128, 16383, 65535]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect(),
                source: None,
                saturation_adu: Some(16383.0),
                sensor_descriptor: Some(crate::processing::debayer::SensorDescriptor {
                    schema_version: 1,
                    width: 2,
                    height: 2,
                    row_stride_bytes: 4,
                    sample_type: crate::processing::debayer::SampleType::U16,
                    byte_order: crate::processing::debayer::ByteOrder::Little,
                    color: crate::processing::debayer::SensorColor::Mono,
                    domain: crate::processing::debayer::SampleDomain::NativeAdu,
                    geometry: crate::processing::debayer::PixelGeometry::identity(
                        "simulator_sensor_plane",
                    ),
                    saturation: None,
                    calibration_ids: Vec::new(),
                }),
            }),
            observation,
        };
        let frame = &observed.frame;
        observed.observation.analysis_descriptor = Some(
            crate::processing::debayer::analyze(&crate::processing::debayer::SensorFrame {
                descriptor: frame.sensor_descriptor.as_ref().unwrap(),
                data: &frame.data,
                masks: Default::default(),
            })
            .unwrap()
            .descriptor,
        );
        observed
    }

    fn simulator_metadata() -> SimulationFocusMetadata {
        SimulationFocusMetadata {
            position: 1200,
            rig_id: "sim-rig".into(),
            filter_name: None,
            camera_name: Some("Simulation".into()),
            focuser_name: Some("SimFocuser".into()),
            cadence_s: 1.0,
            temperature: None,
            temperature_state: TemperatureState::Missing,
            temperature_source: None,
            extra_quality: serde_json::Map::new(),
            equipment: None,
        }
    }

    #[test]
    fn physical_focus_cannot_use_the_simulation_gate() {
        assert!(simulation_focus_allowed(Some("sim"), true));
        for driver in [None, Some("gphoto2"), Some("asi"), Some("uvc")] {
            assert!(!simulation_focus_allowed(driver, true));
        }
        assert!(!simulation_focus_allowed(Some("sim"), false));
    }

    #[test]
    fn focus_qualification_failure_preserves_the_actionable_reason() {
        let error = focus_api_error(HardwareError::Unsupported(
            "Automatic physical autofocus is unavailable pending A03/A06/A13/A15/A23 qualification".into(),
        ).into());
        assert_eq!(error.status, StatusCode::PRECONDITION_FAILED);
        assert_eq!(error.code, "focus_unqualified");
        assert!(error.message.contains("A03/A06/A13/A15/A23"));
    }

    #[test]
    fn simulation_bundle_preserves_exact_sensor_and_analysis_samples() {
        let observed = simulator_observation();
        let pending = simulation_focus_bundle(&observed, simulator_metadata()).unwrap();
        let sensor = pending
            .assets
            .iter()
            .find(|a| a.role == AssetRole::SensorFits)
            .unwrap();
        let luma = pending
            .assets
            .iter()
            .find(|a| a.role == AssetRole::AnalysisLuma)
            .unwrap();
        let expected = [0u16, 128, 16383, 65535];
        let data: Vec<u16> = sensor.bytes[2880..2888]
            .chunks_exact(2)
            .map(|pair| (i16::from_be_bytes([pair[0], pair[1]]) as i32 + 32768) as u16)
            .collect();
        assert_eq!(data, expected);
        let linear = pending
            .assets
            .iter()
            .find(|asset| asset.role == AssetRole::AnalysisFits)
            .unwrap();
        let linear_samples: Vec<f32> = linear.bytes[2880..2896]
            .chunks_exact(4)
            .map(|sample| f32::from_be_bytes(sample.try_into().unwrap()))
            .collect();
        assert_eq!(linear_samples, expected.map(|value| value as f32));
        let descriptor = pending
            .assets
            .iter()
            .find(|asset| asset.role == AssetRole::AnalysisDescriptor)
            .unwrap();
        let descriptor: serde_json::Value = serde_json::from_slice(&descriptor.bytes).unwrap();
        assert_eq!(descriptor["parent"]["filename"], "sensor.fits");
        assert_eq!(descriptor["masks"].as_array().unwrap().len(), 4);
        assert_eq!(
            image::load_from_memory(&luma.bytes)
                .unwrap()
                .to_luma16()
                .into_raw(),
            expected
        );
        assert_eq!(
            pending.acquisition["quality"]["training_label_eligible"],
            false
        );
        assert_eq!(
            pending.acquisition["quality"]["acquisition_domain"],
            "simulation"
        );
    }

    #[test]
    fn extra_quality_keys_are_merged_without_overwriting_the_contract_keys() {
        let observed = simulator_observation();
        let mut meta = simulator_metadata();
        meta.extra_quality
            .insert("a06".into(), json!({"rule": "x"}));
        meta.extra_quality
            .insert("hfr_px".into(), json!("must not replace the measurement"));
        let pending = simulation_focus_bundle(&observed, meta).unwrap();
        let quality = &pending.acquisition["quality"];
        assert_eq!(quality["a06"]["rule"], "x");
        assert_eq!(quality["hfr_px"].as_f64(), Some(3.0));
    }

    #[test]
    fn preview_cfa_and_mismatched_envelopes_cannot_be_archived_as_focus_sensor_data() {
        for format in [
            OlsStreamType::Mjpeg,
            OlsStreamType::Raw16,
            OlsStreamType::Rgb48,
        ] {
            let mut observed = simulator_observation();
            Arc::make_mut(&mut observed.frame).format = format;
            assert!(simulation_focus_bundle(&observed, simulator_metadata()).is_err());
        }
        let mut observed = simulator_observation();
        Arc::make_mut(&mut observed.frame).source = Some(crate::hardware::camera::SourceImage {
            ext: "cr3".into(),
            bytes: vec![1, 2, 3],
        });
        assert!(simulation_focus_bundle(&observed, simulator_metadata()).is_err());
        let mut observed = simulator_observation();
        observed.frame_counter += 1;
        assert!(simulation_focus_bundle(&observed, simulator_metadata()).is_err());
        let mut observed = simulator_observation();
        Arc::make_mut(&mut observed.frame).data.pop();
        assert!(simulation_focus_bundle(&observed, simulator_metadata()).is_err());
    }

    #[test]
    fn simulator_label_requires_exact_analysis_source_and_configuration() {
        let mut observed = simulator_observation();
        observed.observation.analysis_descriptor = None;
        assert!(simulation_focus_bundle(&observed, simulator_metadata()).is_err());
        let mut observed = simulator_observation();
        observed
            .observation
            .analysis_descriptor
            .as_mut()
            .unwrap()
            .source_sha256 = "older-frame".into();
        assert!(simulation_focus_bundle(&observed, simulator_metadata()).is_err());
        let mut observed = simulator_observation();
        observed
            .observation
            .analysis_descriptor
            .as_mut()
            .unwrap()
            .config_sha256 = "older-pipeline".into();
        assert!(simulation_focus_bundle(&observed, simulator_metadata()).is_err());
    }

    #[tokio::test]
    async fn focus_samples_require_distinct_committed_sensor_bundles() {
        let dir =
            std::env::temp_dir().join(format!("achernar-focus-archive-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let local = LocalStore::open(dir.join("achernar.db").to_str().unwrap())
            .await
            .unwrap();
        let store = crate::storage::capture_store::CaptureStore::new(dir.join("captures"));
        let archive = crate::storage::capture_archive::CaptureArchive::spawn(
            store.clone(),
            local.pool().clone(),
            1,
        );
        let mut ids = std::collections::BTreeSet::new();
        for counter in [7, 8] {
            let mut observed = simulator_observation();
            observed.frame_counter = counter;
            Arc::make_mut(&mut observed.frame).frame_counter = counter;
            let pending = simulation_focus_bundle(&observed, simulator_metadata()).unwrap();
            let quality = pending.acquisition["quality"].clone();
            store
                .begin_attempt(
                    local.pool(),
                    &pending.capture_id,
                    None,
                    "sim-rig",
                    &serde_json::json!({}),
                )
                .await
                .unwrap();
            let committed = archive.reserve().unwrap().persist(pending).await.unwrap();
            let sensor_asset = committed
                .assets
                .iter()
                .find(|asset| asset.role == "sensor_fits")
                .unwrap();
            let canonical: Vec<_> = committed
                .assets
                .iter()
                .filter(|asset| {
                    matches!(
                        asset.role.as_str(),
                        "analysis_fits" | "analysis_mask" | "analysis_descriptor"
                    )
                })
                .collect();
            assert_eq!(canonical.len(), 6);
            assert!(canonical
                .iter()
                .all(|asset| asset.derived_from_asset_id.as_deref()
                    == Some(sensor_asset.id.as_str())));
            assert!(ids.insert(committed.capture_id.clone()));
            let sample =
                sample_from_observation(1200, &observed.observation, &committed, None).unwrap();
            assert_eq!(
                sample.frame_id.as_deref(),
                Some(committed.frame_id.as_str())
            );
            let (stored_quality,): (String,) =
                sqlx::query_as("SELECT quality_json FROM frames WHERE id = ?")
                    .bind(&committed.frame_id)
                    .fetch_one(local.pool())
                    .await
                    .unwrap();
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&stored_quality).unwrap(),
                quality
            );
            let mut invalid = committed.clone();
            invalid.status = "native_incomplete";
            assert!(sample_from_observation(1200, &observed.observation, &invalid, None).is_err());
            invalid.status = "committed";
            invalid.assets.retain(|a| a.role != "sensor_fits");
            assert!(sample_from_observation(1200, &observed.observation, &invalid, None).is_err());
        }
        drop(archive);
        local.pool().close().await;
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn luma() -> LumaFrame {
        LumaFrame {
            pixels: vec![100.0; 4],
            width: 2,
            height: 2,
            max_adu: 65535.0,
            saturation_adu: Some(16383.0),
            masks: crate::processing::debayer::AnalysisMasks::empty(0),
            analysis_descriptor: None,
        }
    }

    fn quality_json_for(observation: &SharedObservation) -> serde_json::Value {
        autofocus_quality_json(
            12_345,
            "rig-1",
            Some("L"),
            Some("Canon EOS T6S"),
            Some("myFP2ESP32"),
            7,
            1_770_000_000.0,
            observation,
            &luma(),
            TemperatureState::Measured,
            Some("camera".to_string()),
        )
    }

    /// The FQE's shadow prediction must survive the trip to disk.
    ///
    /// It did not before: `SharedObservation` carried it, `persist_observed_frame`
    /// did not copy it, and the value died with the in-memory observation. For
    /// the FQE that is recoverable — the crops are on disk and can be re-scored.
    /// For the Focus Offset Predictor it is not: these three numbers are its
    /// Quality feature group, they are derived state rather than pixels, and a
    /// sweep recorded without them can never be turned into FOP rows.
    #[test]
    fn fqe_shadow_prediction_reaches_quality_json() {
        let mut observation = SharedObservation::empty_for_test();
        observation.fqe_hfr_px = Some(3.25);
        observation.fqe_sigma_log_hfr = Some(0.11);
        observation.fqe_star_count = Some(42);
        observation.fqe_model_version = Some("1.1.0".to_string());
        observation.fqe_calibrated = Some(true);

        let json = quality_json_for(&observation);

        assert_eq!(json["fqe_hfr_px"].as_f64().unwrap(), 3.25);
        assert!((json["fqe_sigma_log_hfr"].as_f64().unwrap() - 0.11).abs() < 1e-6);
        assert_eq!(json["fqe_star_count"].as_u64().unwrap(), 42);
        assert_eq!(json["fqe_model_version"].as_str().unwrap(), "1.1.0");
        assert_eq!(json["fqe_calibrated"].as_bool().unwrap(), true);
    }

    /// A frame the FQE could not score records nulls, not zeros.
    ///
    /// Same discipline as `temperature_state`: a zero HFR is a *measurement*,
    /// and a training set that cannot tell it from "no model was loaded" will
    /// learn from the difference between them.
    #[test]
    fn absent_fqe_prediction_is_null_not_zero() {
        let json = quality_json_for(&SharedObservation::empty_for_test());

        for key in [
            "fqe_hfr_px",
            "fqe_sigma_log_hfr",
            "fqe_star_count",
            "fqe_model_version",
            "fqe_calibrated",
        ] {
            assert!(json[key].is_null(), "{key} should be null when unmeasured");
        }
    }

    /// quality_json carries the versioned background and detection objects;
    /// the deprecated top-level mirrors are null, not 0, when unmeasured.
    #[test]
    fn quality_json_records_versioned_background_and_null_mirrors() {
        let empty = quality_json_for(&SharedObservation::empty_for_test());
        for key in [
            "background_adu",
            "background_sigma",
            "saturation_fraction",
            "background",
        ] {
            assert!(empty[key].is_null(), "{key} should be null when unmeasured");
        }

        let analysis = crate::vision::quality::background_from_luma(&luma());
        let mut observation = SharedObservation::empty_for_test();
        let detection = crate::vision::quality::estimate_quality(&luma(), &analysis).detection;
        observation.set_background(analysis.result.clone(), detection);
        let json = quality_json_for(&observation);
        assert_eq!(json["background"]["schema_version"], 1);
        assert_eq!(json["background"]["method"], "legacy_global_clip");
        assert_eq!(
            json["background"]["config_sha256"],
            analysis.result.config_sha256.as_str()
        );
        assert_eq!(
            json["background_adu"].as_f64().map(|v| v as f32),
            analysis.result.level()
        );
    }

    /// FOP #35 is defined on the legacy estimator: a SEP level is masked, and
    /// an unmeasured level is absent.
    #[test]
    fn fop_background_feature_only_takes_the_legacy_method() {
        let analysis = crate::vision::quality::background_from_luma(&luma());
        let mut observation = SharedObservation::empty_for_test();
        assert_eq!(fop_background_input(&observation), None);
        observation.set_background(analysis.result.clone(), None);
        assert_eq!(fop_background_input(&observation), Some(100.0));
        let mut sep = analysis.result;
        sep.method = crate::vision::background_model::BackgroundMethod::SepSpatialMesh;
        observation.set_background(sep, None);
        assert!(observation.background_adu.is_some());
        assert_eq!(fop_background_input(&observation), None);
    }

    /// The saturation provenance the FQE dataset builder gates on (defect D1).
    #[test]
    fn saturation_provenance_is_recorded() {
        let json = quality_json_for(&SharedObservation::empty_for_test());

        // The declared sensor clip point, not the container ceiling. Both are
        // written; the builder must prefer the former.
        assert_eq!(json["saturation_adu"].as_f64().unwrap(), 16383.0);
        assert_eq!(json["max_adu"].as_f64().unwrap(), 65535.0);
        assert!(json["saturation_gate_applied"].is_boolean());
    }

    /// Every `frames.quality_json` key the FOP contract claims to read must
    /// actually be emitted here.
    ///
    /// This is the cross-language half of the contract. `ml/contracts/
    /// fop_features.py` records, per feature, the exact column or JSON key that
    /// supplies it; that file is generated into `fop_features_v1.json`, and this
    /// test reads it back and checks the Rust writer against it. Without it the
    /// two sides drift the way the FQE's three descriptions did — silently,
    /// until a model trains on a column of nulls.
    #[test]
    fn every_quality_json_source_named_by_the_fop_contract_is_emitted() {
        let contract_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../ml/contracts/fop_features_v1.json");
        let contract: serde_json::Value = match std::fs::read_to_string(&contract_path) {
            Ok(text) => serde_json::from_str(&text).expect("contract JSON parses"),
            // The daemon must remain buildable without the ML tree beside it.
            Err(_) => return,
        };

        // A fully-populated observation: the question is whether the key is
        // written at all, not whether this particular frame had a value.
        let mut observation = SharedObservation::empty_for_test();
        observation.fqe_hfr_px = Some(3.25);
        observation.fqe_sigma_log_hfr = Some(0.11);
        let json = quality_json_for(&observation);

        let features = contract["features"].as_array().expect("features array");
        let mut checked = 0;
        for feature in features {
            let source = feature["source"].as_str().unwrap_or_default();
            // Sources are written as `frames.quality_json -> key` , optionally
            // with a ", else fallback" clause naming a second key.
            let Some(rest) = source.strip_prefix("frames.quality_json -> ") else {
                continue;
            };
            for key in rest
                .split(", else ")
                .map(|k| k.trim().trim_end_matches(" (preferred)").trim())
            {
                assert!(
                    json.get(key).is_some(),
                    "FOP contract feature {:?} reads frames.quality_json -> {key}, \
                     but autofocus_quality_json never writes that key",
                    feature["name"].as_str().unwrap_or("?"),
                );
                checked += 1;
            }
        }
        assert!(
            checked >= 4,
            "expected the contract to name several quality_json sources, found {checked} \
             — the source strings may have been reworded, which would make this test vacuous"
        );
    }

    /// A converged run records the night it happened on (migration 007).
    ///
    /// This is the join key between a focus run and `environment_log` — the
    /// hand-logged ambient temperature the drift forecaster needs and the sky
    /// class the cloud classifier needs. Before it existed the observer's
    /// readings went into a table that nothing could reach from a focus run.
    #[tokio::test]
    async fn persisted_run_records_its_night() {
        let dir = std::env::temp_dir().join(format!("achernar-focus-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let db = dir.join("achernar.db");
        let store = LocalStore::open(db.to_str().unwrap()).await.unwrap();

        // 23:55 local on one day and 00:05 the next are the same observing
        // night; a run started at either must file under the earlier date.
        let started_at = chrono::Utc::now();
        let run = FocusRunContext {
            run_id: uuid::Uuid::new_v4(),
            rig_id: "rig-1".to_string(),
            filter_name: Some("L".to_string()),
            temperature: Some(8.5),
            start_position: 10_000,
            state: FocusRunState::Converged,
            started_at,
            samples: Vec::new(),
            best_position: Some(10_120),
            best_hfr: Some(2.4),
        };

        persist_focus_run(&store, &run, &HashMap::new())
            .await
            .unwrap();

        let night: (Option<String>,) =
            sqlx::query_as("SELECT night_id FROM focus_runs WHERE id = ?")
                .bind(run.run_id.to_string())
                .fetch_one(store.pool())
                .await
                .unwrap();
        assert_eq!(
            night.0.as_deref(),
            Some(crate::storage::night::night_id(started_at).as_str()),
            "focus_runs.night_id must come from started_at under the local-noon rule"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn wait_until_stopped_waits_for_the_device_to_report_stopped() {
        let mut polls = 0;
        wait_until_stopped(
            || Ok(()),
            || {
                polls += 1;
                Ok(polls < 4)
            },
            Duration::from_secs(5),
            Duration::from_millis(1),
        )
        .await
        .unwrap();
        assert_eq!(polls, 4, "returned before the focuser reported stopped");
    }

    #[tokio::test]
    async fn wait_until_stopped_times_out_on_a_focuser_that_never_stops() {
        let err = wait_until_stopped(
            || Ok(()),
            || Ok(true),
            Duration::from_millis(20),
            Duration::from_millis(2),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("still moving"), "{err}");
    }

    #[tokio::test]
    async fn wait_until_stopped_tolerates_transient_poll_errors_but_not_persistent_ones() {
        let mut polls = 0;
        wait_until_stopped(
            || Ok(()),
            || {
                polls += 1;
                match polls {
                    1 | 2 => Err(HardwareError::Communication("dropped".into()).into()),
                    3 => Ok(true),
                    _ => Ok(false),
                }
            },
            Duration::from_secs(5),
            Duration::from_millis(1),
        )
        .await
        .unwrap();

        let err = wait_until_stopped(
            || Ok(()),
            || Err(HardwareError::Communication("link down".into()).into()),
            Duration::from_secs(5),
            Duration::from_millis(1),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("link down"), "{err}");
    }

    #[tokio::test]
    async fn wait_until_stopped_propagates_abort_immediately() {
        let mut polls = 0;
        let err = wait_until_stopped(
            || Err(FocusError::RunFailed("Autofocus aborted".to_string()).into()),
            || {
                polls += 1;
                Ok(true)
            },
            Duration::from_secs(5),
            Duration::from_millis(1),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("aborted"), "{err}");
        assert_eq!(polls, 0);
    }

    // -----------------------------------------------------------------------
    // H3: A06 mocked timelines
    // -----------------------------------------------------------------------

    fn t(ms: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_785_636_000, 0).unwrap() + chrono::Duration::milliseconds(ms)
    }

    /// The focuser settled at t = 10 s; the newest frame then was #40.
    fn settled() -> FreshnessBaseline {
        FreshnessBaseline {
            settled_at: t(10_000),
            frame_counter: Some(40),
            generation: 3,
        }
    }

    /// A 1 s stream frame (exposure 1 s, interval 1 s) delivered at `arrival`.
    fn stream_frame(counter: i32, arrival_ms: i64) -> FrameTiming {
        FrameTiming {
            frame_counter: counter,
            generation: 3,
            arrival: Some(t(arrival_ms)),
            exposure_s: Some(1.0),
            frame_interval_s: Some(1.0),
            clock_quality: ClockQuality::HostArrival,
            exposure_start: None,
        }
    }

    #[test]
    fn a06_accepts_a_stream_frame_whose_whole_window_follows_the_settle() {
        // Delivered at 12.5 s: it began no earlier than 10.5 s.
        assert_eq!(
            frame_freshness(&settled(), &stream_frame(43, 12_500)),
            Freshness::Fresh {
                start: t(10_500),
                end: t(12_500),
                basis: ClockQuality::HostArrival,
            }
        );
        // Exactly at the settle is after it.
        assert!(matches!(
            frame_freshness(&settled(), &stream_frame(43, 12_000)),
            Freshness::Fresh { .. }
        ));
    }

    #[test]
    fn a06_rejects_a_pre_settle_exposure() {
        // A newer counter, but it was delivered before the settle (late
        // vision processing of an old frame).
        assert_eq!(
            frame_freshness(&settled(), &stream_frame(41, 9_800)),
            Freshness::Skip(FrameSkip::PreSettle)
        );
        // The frame that was newest at the settle is never a sample.
        assert_eq!(
            frame_freshness(&settled(), &stream_frame(40, 11_000)),
            Freshness::Skip(FrameSkip::SameFrame)
        );
    }

    #[test]
    fn a06_rejects_an_exposure_already_in_flight() {
        // Delivered at 11.2 s: it may have started at 9.2 s, before the
        // settle, so the focuser could have been moving during it.
        assert_eq!(
            frame_freshness(&settled(), &stream_frame(41, 11_200)),
            Freshness::Skip(FrameSkip::InFlight)
        );
        // A dedicated exposure that the driver says started before the
        // settle is in flight too, even though it ended after it.
        let mut dedicated = stream_frame(41, 12_000);
        dedicated.clock_quality = ClockQuality::DriverExposureStart;
        dedicated.exposure_start = Some(t(9_900));
        assert_eq!(
            frame_freshness(&settled(), &dedicated),
            Freshness::Skip(FrameSkip::InFlight)
        );
        // Started after the settle: judged by its own start, not the
        // conservative stream bound.
        dedicated.exposure_start = Some(t(10_100));
        assert_eq!(
            frame_freshness(&settled(), &dedicated),
            Freshness::Fresh {
                start: t(10_100),
                end: t(11_100),
                basis: ClockQuality::DriverExposureStart,
            }
        );
    }

    #[test]
    fn a06_records_the_timing_basis_it_used_not_the_cameras_nominal_quality() {
        // A camera whose driver could time its exposures (asi, alpaca, ...)
        // delivering a stream frame without a driver start: the window is
        // the arrival bound, and that is what the sample records.
        let mut stream = stream_frame(43, 12_500);
        stream.clock_quality = ClockQuality::DriverExposureStart;
        assert_eq!(
            frame_freshness(&settled(), &stream),
            Freshness::Fresh {
                start: t(10_500),
                end: t(12_500),
                basis: ClockQuality::HostArrival,
            }
        );
        // A trigger-bracketed camera's stream frame is the arrival bound too.
        stream.clock_quality = ClockQuality::HostTriggerBracket;
        assert!(matches!(
            frame_freshness(&settled(), &stream),
            Freshness::Fresh {
                basis: ClockQuality::HostArrival,
                ..
            }
        ));
        // Simulator frames stay marked as such.
        stream.clock_quality = ClockQuality::Simulator;
        assert!(matches!(
            frame_freshness(&settled(), &stream),
            Freshness::Fresh {
                basis: ClockQuality::Simulator,
                ..
            }
        ));
    }

    #[test]
    fn a06_refuses_a_counter_reset_and_a_stream_restart() {
        assert_eq!(
            frame_freshness(&settled(), &stream_frame(2, 15_000)),
            Freshness::Refused(FrameRefusal::CounterReset {
                baseline: 40,
                observed: 2,
            })
        );
        let mut restarted = stream_frame(41, 15_000);
        restarted.generation = 4;
        assert_eq!(
            frame_freshness(&settled(), &restarted),
            Freshness::Refused(FrameRefusal::GenerationChanged {
                baseline: 3,
                observed: 4,
            })
        );
    }

    #[test]
    fn a06_refuses_an_unknown_start_time() {
        let mut unknown_clock = stream_frame(41, 15_000);
        unknown_clock.clock_quality = ClockQuality::Unknown;
        let mut no_timestamp = stream_frame(41, 15_000);
        no_timestamp.arrival = None;
        let mut no_exposure = stream_frame(41, 15_000);
        no_exposure.exposure_s = None;
        let mut bad_exposure = stream_frame(41, 15_000);
        bad_exposure.exposure_s = Some(f64::NAN);
        let mut no_interval = stream_frame(41, 15_000);
        no_interval.frame_interval_s = None;
        for (frame, reason) in [
            (unknown_clock, "clock_quality_unknown"),
            (no_timestamp, "no_frame_timestamp"),
            (no_exposure, "exposure_unknown"),
            (bad_exposure, "exposure_unknown"),
            (no_interval, "frame_interval_unknown"),
        ] {
            assert_eq!(
                frame_freshness(&settled(), &frame),
                Freshness::Refused(FrameRefusal::UnknownStart(reason))
            );
        }
        assert_eq!(unix_time(0.0), None);
        assert_eq!(unix_time(f64::NAN), None);
    }

    // -----------------------------------------------------------------------
    // Fixtures: the daemon's simulator focuser behind its worker, and a
    // scripted camera stream
    // -----------------------------------------------------------------------

    /// The scripted stream: a frame every `interval_s`, delivered on the wall
    /// clock, its HFR a V-curve around `best` at the focuser's position.
    struct ScriptedCamera {
        sim: Arc<Mutex<crate::hardware::focuser::SimFocuser>>,
        template: ObservedFrame,
        started: Instant,
        started_unix: f64,
        interval_s: f64,
        best: i32,
        /// Triple the HFR of frames taken off the 100-step sweep grid (the
        /// fitted optimum's verification frame), so H4 fails.
        degrade_off_grid: bool,
    }

    fn v_curve_hfr(position: i32, best: i32) -> f64 {
        2.0 + f64::from((position - best).abs()) / 100.0
    }

    impl ScriptedCamera {
        fn new(sim: Arc<Mutex<crate::hardware::focuser::SimFocuser>>, best: i32) -> Self {
            Self {
                sim,
                template: simulator_observation(),
                started: Instant::now(),
                started_unix: Utc::now().timestamp_micros() as f64 / 1e6,
                interval_s: 0.03,
                best,
                degrade_off_grid: false,
            }
        }
    }

    impl FocusCamera for ScriptedCamera {
        fn latest(&self) -> Option<ObservedFrame> {
            let k = (self.started.elapsed().as_secs_f64() / self.interval_s).floor() as i32;
            let arrival = self.started_unix + f64::from(k) * self.interval_s;
            let position = self.sim.lock().unwrap().position().unwrap();
            let mut observed = self.template.clone();
            observed.frame_counter = k + 1;
            observed.capture_unix_time = arrival;
            {
                let frame = Arc::make_mut(&mut observed.frame);
                frame.frame_counter = k + 1;
                frame.unix_timestamp = arrival;
            }
            let mut hfr = v_curve_hfr(position, self.best);
            if self.degrade_off_grid && position % 100 != 0 {
                hfr *= 3.0;
            }
            observed.observation.hfr_px = Some(hfr as f32);
            observed.observation.star_count = 30;
            observed.observation.timestamp = Utc::now();
            Some(observed)
        }

        fn generation(&self) -> u64 {
            self.template.frame.stream_generation
        }

        fn stream(&self) -> crate::error::Result<StreamInfo> {
            Ok(StreamInfo {
                exposure_s: self.interval_s,
                frame_interval_s: self.interval_s,
                camera_name: Some("Simulation".into()),
                driver_id: Some("sim".into()),
                clock_quality: ClockQuality::Simulator,
            })
        }

        fn sensor_temperature(&self) -> (Option<f64>, TemperatureState, Option<String>) {
            (None, TemperatureState::Missing, None)
        }
    }

    struct Fixture {
        directory: std::path::PathBuf,
        state: AppState,
        config: crate::config::FocusConfig,
        sim: Arc<Mutex<crate::hardware::focuser::SimFocuser>>,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.directory).ok();
        }
    }

    /// An `AppState` with the legacy simulator focuser connected through the
    /// spine like any focuser. `fast` tunes the worker cadence for runs.
    async fn fixture(focuser: crate::hardware::focuser::SimFocuser, fast: bool) -> Fixture {
        let directory =
            std::env::temp_dir().join(format!("achernar-focus-move-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let store = LocalStore::open(directory.join("state.db").to_str().unwrap())
            .await
            .unwrap();
        let mut focus = crate::config::FocusConfig::default();
        focus.models.quality_model_path =
            Some(directory.join("absent-fqe.onnx").to_string_lossy().into());
        focus.safety.settle_ms = 0;
        focus.safety.move_timeout_ms = if fast { 2_000 } else { 100 };
        focus.settle_ms = 0;
        let state = AppState::new(
            focus.clone(),
            crate::config::CameraConfig::default(),
            store,
            crate::storage::frames_repo::FrameStore::new(directory.join("previews")),
            crate::storage::capture_store::CaptureStore::new(directory.join("captures")),
            "move-fixture-rig".into(),
            directory.join("data"),
            None,
        );
        if fast {
            state
                .equipment_ctx
                .set_tuning(crate::equipment::spine::WorkerTuning {
                    cadence: Some(crate::equipment::worker::Cadence {
                        idle: Duration::from_millis(20),
                        active: Duration::from_millis(10),
                    }),
                    ..Default::default()
                });
        }
        let driver = crate::equipment::legacy::sim_focuser::LegacySimFocuser::from_sim(
            focuser,
            state.equipment.clock(),
        );
        let sim = driver.shared_sim();
        crate::api::equipment::install_focuser(&state, Box::new(driver), "sim", "")
            .await
            .unwrap();
        Fixture {
            directory,
            state,
            config: focus,
            sim,
        }
    }

    /// A run context over the fixture's focuser, authorised like a run.
    fn run_ctx(fx: &Fixture, camera: Arc<dyn FocusCamera>) -> Arc<RunContext> {
        let focuser = crate::api::equipment::focuser_handle(&fx.state).unwrap();
        let start_position = focuser.snapshot().logical_position.unwrap_or(0);
        let op = AutofocusOp::authorize(&fx.state, focuser, GuidingDuringAutofocus::Keep).unwrap();
        Arc::new(RunContext {
            state: fx.state.clone(),
            config: fx.config.clone(),
            camera,
            op,
            focus_safety: Mutex::new(FocusSafetyGuard::new(fx.config.safety.clone())),
            rig_id: "move-fixture-rig".into(),
            filter: RunFilter {
                name: None,
                verified: None,
                missing_reason: "no_filter_wheel",
            },
            simulation: true,
            start_position,
            cfz: None,
            ledger: Mutex::new(RunLedger::default()),
        })
    }

    fn sim_position(sim: &Arc<Mutex<crate::hardware::focuser::SimFocuser>>) -> i32 {
        sim.lock().unwrap().position().unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn move_reaches_a_distant_target_through_clamped_steps() {
        let fx = fixture(crate::hardware::focuser::SimFocuser::new(0), false).await;
        let camera = Arc::new(ScriptedCamera::new(fx.sim.clone(), 0));
        let ctx = run_ctx(&fx, camera);
        let report = move_focuser_to_position(&ctx, 1_700).await.unwrap();
        assert_eq!(sim_position(&fx.sim), 1_700);
        // The last move's `focuser_moves` id and settle time are found.
        assert!(report.move_id.is_some());
        assert!(report.settled_event_at.is_some());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn move_fails_when_the_focuser_stops_short_of_the_target() {
        // The device's travel ends at 1000: the gate's travel bounds refuse
        // the step past it, or the move stops short; either way the run
        // fails at 1000 instead of retrying forever.
        let fx = fixture(
            crate::hardware::focuser::SimFocuser::with_max_range(0, 1_000),
            false,
        )
        .await;
        let camera = Arc::new(ScriptedCamera::new(fx.sim.clone(), 0));
        let ctx = run_ctx(&fx, camera);
        let err = move_focuser_to_position(&ctx, 2_000).await.unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("without progressing") || message.contains("travel"),
            "{message}"
        );
        assert_eq!(sim_position(&fx.sim), 1_000);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn move_halts_a_focuser_that_never_reports_stopped() {
        let mut sim = crate::hardware::focuser::SimFocuser::new(0);
        sim.set_stuck_moving(true);
        let fx = fixture(sim, false).await;
        let camera = Arc::new(ScriptedCamera::new(fx.sim.clone(), 0));
        let ctx = run_ctx(&fx, camera);
        let err = move_focuser_to_position(&ctx, 300).await.unwrap_err();
        assert!(err.to_string().contains("still moving"), "{err}");
        // The halt runs out of band (the abort path) or as the worker's
        // deadline cancel; give the out-of-band thread a moment.
        let deadline = Instant::now() + Duration::from_secs(5);
        while fx.sim.lock().unwrap().is_moving().unwrap() && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(
            !fx.sim.lock().unwrap().is_moving().unwrap(),
            "the timed-out focuser must have been halted"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_abort_stops_the_runs_later_moves() {
        let fx = fixture(crate::hardware::focuser::SimFocuser::new(0), false).await;
        let camera = Arc::new(ScriptedCamera::new(fx.sim.clone(), 0));
        let ctx = run_ctx(&fx, camera);
        move_focuser_to_position(&ctx, 200).await.unwrap();
        // An abort issued through the gate (as /focus/abort does) bumps the
        // focuser's abort epoch: the run's operation may no longer move it
        // (D19), even with the abort flag cleared.
        let handle = crate::api::equipment::focuser_abort_handle(&fx.state).unwrap();
        let _ = fx
            .state
            .equipment
            .gate()
            .abort(&handle, CommandSource::Operator { overriding: false });
        let err = move_focuser_to_position(&ctx, 400).await.unwrap_err();
        assert!(err.to_string().contains("aborted"), "{err}");
        assert_eq!(sim_position(&fx.sim), 200);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_runs_permit_holds_the_focuser_until_it_finishes() {
        let fx = fixture(crate::hardware::focuser::SimFocuser::new(0), false).await;
        let camera = Arc::new(ScriptedCamera::new(fx.sim.clone(), 0));
        let ctx = run_ctx(&fx, camera);
        // A manual move without override meets the run's ownership (409).
        let handle = crate::api::equipment::focuser_handle(&fx.state).unwrap();
        let err = crate::api::equipment::submit(
            &fx.state,
            &handle,
            GatedOp::Focuser(FocuserCmd::MoveTo(50)),
            CommandSource::Operator { overriding: false },
            Priority::Exposure,
        )
        .await
        .unwrap_err();
        assert_eq!(err.status, StatusCode::CONFLICT, "{}", err.message);
        ctx.op.finish(&fx.state, OpOutcome::Succeeded);
        assert!(fx.state.equipment.gate().is_idle());
        // The finished operation takes no more steps.
        assert!(move_focuser_to_position(&ctx, 100).await.is_err());
    }

    // -----------------------------------------------------------------------
    // H3 against the hub: motion inside the exposure window
    // -----------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_motion_event_inside_the_window_rejects_the_frame() {
        let fx = fixture(crate::hardware::focuser::SimFocuser::new(500), true).await;
        let focuser = crate::api::equipment::focuser_handle(&fx.state).unwrap();
        // Let the worker publish a few still samples.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let now = Utc::now();
        let baseline = FreshnessBaseline {
            settled_at: now - chrono::Duration::seconds(1),
            frame_counter: None,
            generation: 1,
        };
        let stream = StreamInfo {
            exposure_s: 0.1,
            frame_interval_s: 0.1,
            camera_name: None,
            driver_id: Some("sim".into()),
            clock_quality: ClockQuality::Simulator,
        };
        let mut observed = simulator_observation();
        let at = |when: DateTime<Utc>| when.timestamp_micros() as f64 / 1e6;
        // A frame whose window covers `now`, when something moved the focuser.
        observed.capture_unix_time = at(now + chrono::Duration::milliseconds(50));
        fx.state.equipment.emit(
            EquipmentEvent::new(
                EventKind::FocuserHalted {
                    position: Some(500),
                },
                fx.state.equipment.clock().as_ref(),
            )
            .for_device(DeviceKind::Focuser, focuser.connection_id()),
        );
        match judge_frame(
            &fx.state,
            &focuser,
            &baseline,
            Some(500),
            &stream,
            &observed,
        ) {
            FrameVerdict::Skip(reason) => assert!(reason.contains("moved"), "{reason}"),
            _ => panic!("a frame with focuser motion in its window must be skipped"),
        }
        // A later frame, after the event, is accepted once a poll covers it.
        tokio::time::sleep(Duration::from_millis(400)).await;
        observed.capture_unix_time = at(Utc::now());
        observed.frame_counter += 1;
        let end = unix_time(observed.capture_unix_time).unwrap();
        match judge_until_decided(
            &fx.state,
            &focuser,
            &baseline,
            Some(500),
            &stream,
            &observed,
        )
        .await
        {
            FrameVerdict::Accept(accepted) => {
                assert_eq!(accepted.readback_after, Some(500));
                assert!(accepted.readback_after_at >= end);
                assert!(accepted.kinds.contains(&DeviceKind::Focuser));
            }
            FrameVerdict::Skip(reason) | FrameVerdict::Undecided(reason) => {
                panic!("a quiescent frame must be accepted: {reason}")
            }
            FrameVerdict::Refuse(refusal) => panic!("refused: {refusal}"),
        }
        // A focuser read back elsewhere than at the settle is passed over.
        match judge_frame(
            &fx.state,
            &focuser,
            &baseline,
            Some(499),
            &stream,
            &observed,
        ) {
            FrameVerdict::Skip(reason) => assert!(reason.contains("read back"), "{reason}"),
            _ => panic!("a changed read-back must be skipped"),
        }
    }

    /// Judge a frame until the verdict is no longer undecided (5 s at most).
    async fn judge_until_decided(
        state: &AppState,
        focuser: &FocuserHandle,
        baseline: &FreshnessBaseline,
        readback_before: Option<i32>,
        stream: &StreamInfo,
        observed: &ObservedFrame,
    ) -> FrameVerdict {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let verdict = judge_frame(state, focuser, baseline, readback_before, stream, observed);
            if !matches!(verdict, FrameVerdict::Undecided(_)) || Instant::now() >= deadline {
                return verdict;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    /// H3: with the default 2 s idle cadence a sub-second exposure usually
    /// ends between two focuser polls. The latest read-back then predates the
    /// exposure, so the frame is not accepted until a poll taken after the
    /// window's end confirms the position (the hub's quiescence alone would
    /// accept it on a fresh pre-window sample).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_frame_waits_for_a_focuser_poll_after_its_exposure() {
        let fx = fixture(crate::hardware::focuser::SimFocuser::new(500), false).await;
        let focuser = crate::api::equipment::focuser_handle(&fx.state).unwrap();
        // The worker's first idle poll.
        let deadline = Instant::now() + Duration::from_secs(5);
        while focuser.snapshot().logical_position.is_none() && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let first = focuser.snapshot();
        assert_eq!(first.logical_position, Some(500));
        let stream = StreamInfo {
            exposure_s: 0.1,
            frame_interval_s: 0.1,
            camera_name: None,
            driver_id: Some("sim".into()),
            clock_quality: ClockQuality::Simulator,
        };
        // A 0.2 s window ending 1 s from now: no poll can have covered it
        // yet, and the next idle poll is up to 2 s away.
        let end = Utc::now() + chrono::Duration::seconds(1);
        let baseline = FreshnessBaseline {
            settled_at: end - chrono::Duration::seconds(1),
            frame_counter: None,
            generation: 1,
        };
        let mut observed = simulator_observation();
        observed.capture_unix_time = end.timestamp_micros() as f64 / 1e6;
        // The hub alone calls the window quiescent on the pre-window sample.
        let start = end - chrono::Duration::milliseconds(200);
        let deadline = Instant::now() + Duration::from_secs(2);
        while !matches!(
            fx.state
                .equipment
                .quiescent(&[DeviceKind::Focuser], start, end),
            Quiescence::Quiescent { .. }
        ) && Instant::now() < deadline
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(
            Utc::now() < end,
            "the test machine is too slow for this window"
        );
        match judge_frame(
            &fx.state,
            &focuser,
            &baseline,
            Some(500),
            &stream,
            &observed,
        ) {
            FrameVerdict::Undecided(reason) => {
                assert!(reason.contains("focuser_no_readback_after_end"), "{reason}")
            }
            FrameVerdict::Accept(accepted) => panic!(
                "accepted on a read-back taken at {} for an exposure ending {end}",
                accepted.readback_after_at
            ),
            FrameVerdict::Skip(reason) => panic!("skipped: {reason}"),
            FrameVerdict::Refuse(refusal) => panic!("refused: {refusal}"),
        }
        // The next idle poll after the window's end decides it.
        match judge_until_decided(
            &fx.state,
            &focuser,
            &baseline,
            Some(500),
            &stream,
            &observed,
        )
        .await
        {
            FrameVerdict::Accept(accepted) => {
                assert!(accepted.readback_after_at >= end);
                assert!(accepted.focuser.published_at >= end);
                assert_eq!(accepted.readback_after, Some(500));
            }
            FrameVerdict::Undecided(reason) | FrameVerdict::Skip(reason) => {
                panic!("a poll after the exposure must decide it: {reason}")
            }
            FrameVerdict::Refuse(refusal) => panic!("refused: {refusal}"),
        }
    }

    #[test]
    fn a_configured_af_filter_that_is_not_in_place_is_reported() {
        let unverified = RunFilter {
            name: Some("L".into()),
            verified: None,
            missing_reason: "filter_unverified",
        };
        let verified = |name: &str| RunFilter {
            name: Some(name.into()),
            verified: Some(VerifiedFilter {
                name: name.into(),
                offset_steps: None,
            }),
            missing_reason: "",
        };
        // Nothing configured: nothing to say.
        assert_eq!(af_filter_warning("  ", &verified("Ha"), true), None);
        // Already in place (matched like the wheel matches names).
        assert_eq!(af_filter_warning(" l ", &verified("L"), true), None);
        let other = af_filter_warning("L", &verified("Ha"), true).unwrap();
        assert!(
            other.contains("not applied") && other.contains("\"Ha\""),
            "{other}"
        );
        let unknown = af_filter_warning("L", &unverified, true).unwrap();
        assert!(unknown.contains("not verified"), "{unknown}");
        let no_wheel = af_filter_warning("L", &unverified, false).unwrap();
        assert!(no_wheel.contains("no filter wheel"), "{no_wheel}");
    }

    // -----------------------------------------------------------------------
    // The end-to-end simulator run (H3, H4, H5, H7)
    // -----------------------------------------------------------------------

    /// The WP1B-2 acceptance run: a simulator sweep through the gate and the
    /// worker produces samples with read-back and capture identity, a
    /// verified action, the FOP vectors, and leaves the rig's drift state
    /// untouched.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_simulator_run_records_verified_provenance_and_leaves_the_rig_drift_state() {
        let mut fx = fixture(crate::hardware::focuser::SimFocuser::new(1_000), true).await;
        fx.config.initial_step_size = 100;
        fx.config.samples_per_position = 1;
        fx.config.sweep_radius_points = 3;
        // The rig's drift model and its state file: a simulator run must
        // leave both exactly as they were (D18).
        let drift_path = fx.directory.join("focus_drift_state.json");
        std::fs::write(&drift_path, b"{\"sentinel\": true}").unwrap();
        let forecaster = Arc::new(tokio::sync::Mutex::new(
            crate::focus::focus_drift::FocusDriftForecaster::new(),
        ));
        fx.state.focus_drift = Some(forecaster.clone());
        fx.state.focus_drift_state_path = Some(drift_path.clone());

        let camera = Arc::new(ScriptedCamera::new(fx.sim.clone(), 1_050));
        let response = run_autofocus_session(
            &fx.state,
            fx.config.clone(),
            Some("L".into()),
            camera,
            GuidingDuringAutofocus::Keep,
        )
        .await
        .unwrap_or_else(|err| panic!("{}: {}", err.code, err.message));
        assert_eq!(response.state, "Converged");
        assert_eq!(response.samples_count, 7);
        // The curve fit (best r² of hyperbola, V and parabola) lands near the
        // true optimum of the scripted V-curve.
        let best = response.best_position.unwrap();
        assert!((best - 1_050).abs() <= 10, "best focus {best}");
        assert_eq!(response.verified, Some(true), "{:?}", response.warnings);
        assert_eq!(sim_position(&fx.sim), best);

        let pool = fx.state.local_store.pool();
        let rows = sqlx::query(
            "SELECT position, position_readback_before, position_readback_after, capture_id, \
             focuser_move_id, settled_at, exposure_started_at, clock_quality, position_source, \
             filter_verified, moved_during_exposure, backlash_model, frame_id \
             FROM focus_samples WHERE run_id = ? ORDER BY id",
        )
        .bind(&response.run_id)
        .fetch_all(pool)
        .await
        .unwrap();
        assert_eq!(rows.len(), 7);
        use sqlx::Row;
        for row in &rows {
            let position: i64 = row.get("position");
            assert_eq!(
                row.get::<Option<i64>, _>("position_readback_before"),
                Some(position)
            );
            assert_eq!(
                row.get::<Option<i64>, _>("position_readback_after"),
                Some(position)
            );
            assert!(row.get::<Option<String>, _>("capture_id").is_some());
            assert!(row.get::<Option<String>, _>("focuser_move_id").is_some());
            assert_eq!(
                row.get::<Option<String>, _>("clock_quality").as_deref(),
                Some("simulator")
            );
            assert_eq!(
                row.get::<Option<String>, _>("position_source").as_deref(),
                Some("device")
            );
            assert_eq!(row.get::<Option<bool>, _>("filter_verified"), Some(false));
            assert_eq!(
                row.get::<Option<bool>, _>("moved_during_exposure"),
                Some(false)
            );
            assert!(row.get::<Option<String>, _>("backlash_model").is_some());
            // A06: the exposure began at or after the settle.
            let settled = crate::storage::columns::parse_time(
                &row.get::<Option<String>, _>("settled_at").unwrap(),
            )
            .unwrap();
            let started = crate::storage::columns::parse_time(
                &row.get::<Option<String>, _>("exposure_started_at").unwrap(),
            )
            .unwrap();
            assert!(started >= settled, "{started} before {settled}");
        }

        // H1/D7: every focus frame carries the hub's window of the A06
        // interval it was accepted on: the equipment block, a window row
        // assembled from the rings (not the bare "not assembled" row) and
        // the window's focuser position, the same one FITS FOCPOS carries.
        for row in &rows {
            let frame_id: String = row.get("frame_id");
            let position: i64 = row.get("position");
            let (acquisition, focuser_position): (String, Option<i64>) = sqlx::query_as(
                "SELECT acquisition_json, focuser_position FROM frames WHERE id = ?",
            )
            .bind(&frame_id)
            .fetch_one(pool)
            .await
            .unwrap();
            let acquisition: Value = serde_json::from_str(&acquisition).unwrap();
            let block = &acquisition["equipment"];
            assert_eq!(
                block["schema"],
                crate::api::camera::equipment::EQUIPMENT_SCHEMA,
                "{acquisition}"
            );
            assert_eq!(block["window"]["clock_quality"], "simulator");
            assert_eq!(focuser_position, Some(position));
            let (reason, start, pos_start): (Option<String>, Option<String>, Option<i64>) =
                sqlx::query_as(
                    "SELECT quiescence_reason, exposure_started_at, focuser_pos_start \
                     FROM exposure_device_windows WHERE frame_id = ?",
                )
                .bind(&frame_id)
                .fetch_one(pool)
                .await
                .unwrap();
            assert_ne!(reason.as_deref(), Some("equipment_window_not_assembled"));
            assert_eq!(pos_start, Some(position));
            let start = crate::storage::columns::parse_time(&start.unwrap()).unwrap();
            let a06_start = crate::storage::columns::parse_time(
                &row.get::<Option<String>, _>("exposure_started_at").unwrap(),
            )
            .unwrap();
            assert_eq!(start, a06_start, "the window is the A06 interval");
        }

        // H7: every sample frame carries its FOP vector and A06 evidence.
        let frame_id: String = rows[3].get("frame_id");
        let (quality,): (String,) = sqlx::query_as("SELECT quality_json FROM frames WHERE id = ?")
            .bind(&frame_id)
            .fetch_one(pool)
            .await
            .unwrap();
        let quality: Value = serde_json::from_str(&quality).unwrap();
        assert_eq!(quality["autofocus_role"], "sweep");
        let fop = &quality["fop_features"];
        assert_eq!(fop["contract_version"], fop_features::CONTRACT_VERSION);
        assert_eq!(
            fop["values"].as_array().unwrap().len(),
            fop_features::N_FEATURES
        );
        assert_eq!(fop["values"][9], 3.0, "the fourth sample's ordinal");
        assert_eq!(fop["label_eligible"], true);
        // No ambient sensor, no wheel: missing, never fabricated.
        assert!(fop["values"][12].is_null());
        assert!(fop["values"][16].is_null());
        assert_eq!(quality["a06"]["clock_quality"], "simulator");
        assert_eq!(quality["a06"]["stream_clock_quality"], "simulator");
        assert!(quality["a06"]["focuser_move_id"].is_string());
        // H3: the read-back after comes from a poll at or after the window.
        let a06_time = |key: &str| {
            quality["a06"][key]
                .as_str()
                .and_then(|text| DateTime::parse_from_rfc3339(text).ok())
                .unwrap_or_else(|| panic!("a06.{key}: {}", quality["a06"][key]))
        };
        assert!(a06_time("position_readback_after_at") >= a06_time("exposure_window_end"));
        assert!(a06_time("focuser_snapshot_published_at") >= a06_time("exposure_window_end"));

        // H4: the verified action with its verification frame.
        let action = sqlx::query(
            "SELECT verified, verification_frame_id, verification_hfr, focuser_move_id \
             FROM focus_actions WHERE run_id = ?",
        )
        .bind(&response.run_id)
        .fetch_one(pool)
        .await
        .unwrap();
        assert_eq!(action.get::<Option<bool>, _>("verified"), Some(true));
        assert!(action.get::<Option<String>, _>("focuser_move_id").is_some());
        let hfr: f64 = action.get::<Option<f64>, _>("verification_hfr").unwrap();
        assert!((hfr - v_curve_hfr(best, 1_050)).abs() < 1e-6, "{hfr}");
        let verification_frame: String = action
            .get::<Option<String>, _>("verification_frame_id")
            .unwrap();
        let (quality,): (String,) = sqlx::query_as("SELECT quality_json FROM frames WHERE id = ?")
            .bind(&verification_frame)
            .fetch_one(pool)
            .await
            .unwrap();
        let quality: Value = serde_json::from_str(&quality).unwrap();
        assert_eq!(quality["autofocus_role"], "verification");
        assert_eq!(quality["fop_label_steps"], 0.0);

        // The journal: started, verified (with the focus run's id), and the
        // operation's end for the overheads view (H18).
        let events = fx.state.equipment.events_since(0, &[], 500);
        let names: Vec<&str> = events.iter().map(|e| e.name()).collect();
        assert!(names.contains(&"autofocus_started"), "{names:?}");
        let verified = events
            .iter()
            .find_map(|e| match &e.event {
                EventKind::AutofocusVerified {
                    run_id, frame_id, ..
                } => Some((run_id.clone(), frame_id.clone())),
                _ => None,
            })
            .expect("autofocus_verified");
        assert_eq!(verified.0, response.run_id);
        assert_eq!(verified.1.as_deref(), Some(verification_frame.as_str()));
        // autofocus_started names the same focus_runs id as the verification
        // and the stored run, and both share the operation's command id.
        let started = events
            .iter()
            .find(|e| matches!(&e.event, EventKind::AutofocusStarted { .. }))
            .expect("autofocus_started");
        match &started.event {
            EventKind::AutofocusStarted { run_id } => assert_eq!(run_id, &response.run_id),
            _ => unreachable!(),
        }
        let verified_event = events
            .iter()
            .find(|e| matches!(&e.event, EventKind::AutofocusVerified { .. }))
            .unwrap();
        assert!(started.command_id.is_some());
        assert_eq!(started.command_id, verified_event.command_id);
        let (stored_runs,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM focus_runs WHERE id = ?")
            .bind(&response.run_id)
            .fetch_one(pool)
            .await
            .unwrap();
        assert_eq!(stored_runs, 1);
        assert!(events.iter().any(|e| matches!(
            &e.event,
            EventKind::OperationEnded {
                op: OpKind::Autofocus,
                outcome: OpOutcome::Succeeded,
                ..
            }
        )));

        // Ownership and the busy flag are released.
        assert!(fx.state.equipment.gate().is_idle());
        assert!(!fx.state.equipment.busy_flags().current().autofocus);

        // H5 / D18: the rig's drift model and its file are untouched.
        assert_eq!(forecaster.lock().await.update_count(), 0);
        assert_eq!(
            std::fs::read(&drift_path).unwrap(),
            b"{\"sentinel\": true}".to_vec()
        );
    }

    /// H4: a run that converges but whose verification frame does not
    /// confirm focus returns the converged result with `verified: false`, and
    /// its audit outcome and `operation_ended` say failed
    /// (`focus_not_verified`), in agreement with its `autofocus_failed`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn an_unverified_run_is_recorded_as_failed_not_succeeded() {
        let mut fx = fixture(crate::hardware::focuser::SimFocuser::new(1_000), true).await;
        fx.config.initial_step_size = 100;
        fx.config.samples_per_position = 1;
        fx.config.sweep_radius_points = 3;
        let mut camera = ScriptedCamera::new(fx.sim.clone(), 1_050);
        camera.degrade_off_grid = true;
        let response = run_autofocus_session(
            &fx.state,
            fx.config.clone(),
            None,
            Arc::new(camera),
            GuidingDuringAutofocus::Keep,
        )
        .await
        .unwrap_or_else(|err| panic!("{}: {}", err.code, err.message));
        assert_eq!(response.state, "Converged");
        let best = response.best_position.unwrap();
        assert_ne!(
            best % 100,
            0,
            "the fitted optimum {best} must be off the grid"
        );
        assert_eq!(response.verified, Some(false));
        assert!(
            response
                .warnings
                .iter()
                .any(|w| w.starts_with("focus not verified")),
            "{:?}",
            response.warnings
        );

        let events = fx.state.equipment.events_since(0, &[], 500);
        assert!(events
            .iter()
            .any(|e| matches!(&e.event, EventKind::AutofocusFailed { .. })));
        let ended = events
            .iter()
            .find_map(|e| match &e.event {
                EventKind::OperationEnded {
                    op: OpKind::Autofocus,
                    outcome,
                    ..
                } => Some(outcome.clone()),
                _ => None,
            })
            .expect("operation_ended");
        match ended {
            OpOutcome::Failed(err) => {
                assert!(err.to_string().contains("focus_not_verified"), "{err}")
            }
            other => panic!("an unverified run ended {}", other.label()),
        }
        assert!(!events.iter().any(|e| matches!(
            &e.event,
            EventKind::OperationEnded {
                op: OpKind::Autofocus,
                outcome: OpOutcome::Succeeded,
                ..
            }
        )));
        assert!(fx.state.equipment.gate().is_idle());
    }

    // -----------------------------------------------------------------------
    // H4 rule
    // -----------------------------------------------------------------------

    fn sample(position: i32, hfr: f64) -> FocusSample {
        FocusSample {
            position,
            hfr,
            hfd: None,
            fwhm: None,
            star_count: 10,
            quality_score: None,
            temperature: None,
            timestamp: Utc::now(),
            frame_id: None,
        }
    }

    #[test]
    fn the_best_sampled_hfr_is_the_smallest_per_position_median() {
        let samples = [
            sample(100, 3.0),
            sample(200, 2.2),
            sample(200, 9.0),
            sample(200, 2.4),
            sample(300, 2.5),
        ];
        // 200's median is 2.4: one bad frame does not make it the worst.
        assert_eq!(best_sampled_hfr(&samples), Some(2.4));
        assert_eq!(best_sampled_hfr(&[]), None);
    }

    #[test]
    fn verification_needs_a_reliable_readback_at_best_and_an_hfr_no_worse_than_the_sweep() {
        assert!(verification_verdict(2.0, Some(2.5), Some(1_050), 1_050, 0, true).is_ok());
        assert!(verification_verdict(2.7, Some(2.5), Some(1_050), 1_050, 0, true).is_ok());
        let worse = verification_verdict(2.9, Some(2.5), Some(1_050), 1_050, 0, true);
        assert!(worse.unwrap_err().contains("worse"));
        let elsewhere = verification_verdict(2.0, Some(2.5), Some(1_052), 1_050, 0, true);
        assert!(elsewhere.unwrap_err().contains("read back"));
        assert!(verification_verdict(2.0, Some(2.5), Some(1_052), 1_050, 5, true).is_ok());
        assert!(verification_verdict(2.0, Some(2.5), Some(1_050), 1_050, 0, false).is_err());
        assert!(verification_verdict(2.0, None, Some(1_050), 1_050, 0, true).is_err());
    }

    // -----------------------------------------------------------------------
    // H5 and H6: an isolated forecaster
    // -----------------------------------------------------------------------

    fn reading(
        source: TemperatureSourceKind,
        celsius: f64,
        state: ObsState,
    ) -> Observed<TemperatureReading> {
        let mut observed = Observed::fresh(
            TemperatureReading {
                celsius,
                source,
                role: TemperatureRole::Ambient,
                identity_key: None,
                connection_id: None,
                camera_sensor_fallback: source == TemperatureSourceKind::CameraSensor,
            },
            Source::Derived,
            Utc::now(),
            1,
        );
        observed.state = state;
        observed
    }

    #[test]
    fn only_a_verified_non_simulator_run_with_a_fresh_ambient_reading_teaches_the_fdf() {
        let weather = reading(TemperatureSourceKind::Weather, 4.5, ObsState::Fresh);
        assert_eq!(drift_learning_temperature(false, true, &weather), Ok(4.5));
        assert_eq!(
            drift_learning_temperature(true, true, &weather),
            Err("simulator_run")
        );
        assert_eq!(
            drift_learning_temperature(false, false, &weather),
            Err("not_verified")
        );
        let camera = reading(TemperatureSourceKind::CameraSensor, 20.0, ObsState::Fresh);
        assert_eq!(
            drift_learning_temperature(false, true, &camera),
            Err("camera_sensor_fallback")
        );
        let stale = reading(TemperatureSourceKind::FocuserProbe, 4.5, ObsState::Stale);
        assert_eq!(
            drift_learning_temperature(false, true, &stale),
            Err("temperature_stale")
        );
        assert_eq!(
            drift_learning_temperature(false, true, &Observed::not_set(Source::Derived)),
            Err("no_ambient_temperature")
        );
    }

    #[tokio::test]
    async fn a_verified_measurement_updates_an_isolated_forecaster_at_the_frame_time() {
        let directory = std::env::temp_dir().join(format!("achernar-fdf-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("drift.json");
        let forecaster =
            tokio::sync::Mutex::new(crate::focus::focus_drift::FocusDriftForecaster::new());
        let frame_time = t(0);
        fold_verified_focus(&forecaster, Some(&path), 10_000, 8.0, frame_time, &[]).await;
        {
            let guard = forecaster.lock().await;
            assert_eq!(guard.update_count(), 1);
            assert_eq!(guard.last_update(), Some(frame_time));
        }
        let saved = crate::storage::focus_drift_repo::load_state(&path).unwrap();
        assert_eq!(saved.update_count, 1);
        assert_eq!(saved.last_update, Some(frame_time));

        // A reset with known ends since the last update translates the
        // position before the next measurement is folded in (F-h).
        let resets = [PositionReset {
            at: t(60_000),
            from: Some(10_000),
            to: Some(0),
            cause: "sync".into(),
        }];
        fold_verified_focus(&forecaster, None, 20, 8.0, t(120_000), &resets).await;
        let guard = forecaster.lock().await;
        assert_eq!(guard.update_count(), 2);
        assert!(guard.position().abs() < 100.0, "{}", guard.position());
        std::fs::remove_dir_all(&directory).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_drift_forecast_prefers_an_ambient_reading_over_the_camera() {
        let fx = fixture(crate::hardware::focuser::SimFocuser::new(0), false).await;
        // No ambient source and no camera: nothing, and not a fallback.
        let none = drift_temperature(&fx.state).await;
        assert_eq!(none.celsius, None);
        assert!(!none.camera_sensor_fallback);
        // A weather device with a fresh temperature is the source.
        let (weather, weather_fake) =
            fake_weather(crate::equipment::snapshot::WeatherSnapshot::default());
        fx.state.equipment.attach(weather);
        weather_fake.update(|s| {
            s.reading.temperature_c =
                Observed::fresh_at(3.5, Source::Property, fx.state.equipment.clock().as_ref());
        });
        let ambient = drift_temperature(&fx.state).await;
        assert_eq!(ambient.celsius, Some(3.5));
        assert_eq!(ambient.source.as_deref(), Some("weather"));
        assert!(!ambient.camera_sensor_fallback);
    }

    // -----------------------------------------------------------------------
    // G-m and F-d around a run
    // -----------------------------------------------------------------------

    fn guiding_snapshot() -> GuiderSnapshot {
        GuiderSnapshot {
            app_state: Observed::fresh(GuiderAppState::Guiding, Source::Phd2Event, Utc::now(), 1),
            ..GuiderSnapshot::default()
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn guiding_is_paused_then_resumed_and_settled_or_stopped_then_restarted() {
        for (policy, before, after) in [
            (GuidingDuringAutofocus::Pause, 1, 2),
            (GuidingDuringAutofocus::Stop, 1, 1),
        ] {
            let fx = fixture(crate::hardware::focuser::SimFocuser::new(0), false).await;
            let (guider, guider_fake) = fake_guider(guiding_snapshot());
            fx.state.equipment.attach(guider);
            let focuser = crate::api::equipment::focuser_handle(&fx.state).unwrap();
            let op = AutofocusOp::authorize(&fx.state, focuser, policy).unwrap();
            assert!(op
                .command
                .domains
                .contains(&Domain::Device(DeviceKind::Guider)));
            let held = hold_guiding(&fx.state, &op, policy).await.unwrap();
            let commands = guider_fake.commands();
            assert_eq!(commands.len(), before);
            match policy {
                GuidingDuringAutofocus::Pause => {
                    assert_eq!(held, GuidingHeld::Paused);
                    assert!(matches!(commands[0], GuiderCmd::Pause { full: false }));
                }
                _ => {
                    assert_eq!(held, GuidingHeld::Stopped);
                    assert!(matches!(commands[0], GuiderCmd::Stop));
                }
            }
            let mut warnings = Vec::new();
            release_guiding(&fx.state, &op, held, &mut warnings).await;
            assert!(warnings.is_empty(), "{warnings:?}");
            let commands = guider_fake.commands();
            assert_eq!(commands.len(), before + after);
            if policy == GuidingDuringAutofocus::Pause {
                assert!(matches!(commands[1], GuiderCmd::Resume));
            }
            assert!(matches!(
                commands.last().unwrap(),
                GuiderCmd::Start {
                    recalibrate: false,
                    ..
                }
            ));
            op.finish(&fx.state, OpOutcome::Succeeded);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn keep_and_a_guider_that_is_not_guiding_are_left_alone() {
        let fx = fixture(crate::hardware::focuser::SimFocuser::new(0), false).await;
        let (guider, guider_fake) = fake_guider(guiding_snapshot());
        fx.state.equipment.attach(guider);
        let focuser = crate::api::equipment::focuser_handle(&fx.state).unwrap();
        let op = AutofocusOp::authorize(&fx.state, focuser.clone(), GuidingDuringAutofocus::Keep)
            .unwrap();
        assert!(!op
            .command
            .domains
            .contains(&Domain::Device(DeviceKind::Guider)));
        let held = hold_guiding(&fx.state, &op, GuidingDuringAutofocus::Keep)
            .await
            .unwrap();
        assert_eq!(held, GuidingHeld::Untouched);
        op.finish(&fx.state, OpOutcome::Succeeded);
        // A looping (not guiding) PHD2 is not paused either.
        guider_fake.update(|s| {
            s.app_state = Observed::fresh(GuiderAppState::Looping, Source::Phd2Event, Utc::now(), 2)
        });
        let op = AutofocusOp::authorize(&fx.state, focuser, GuidingDuringAutofocus::Pause).unwrap();
        let held = hold_guiding(&fx.state, &op, GuidingDuringAutofocus::Pause)
            .await
            .unwrap();
        assert_eq!(held, GuidingHeld::Untouched);
        assert!(guider_fake.commands().is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tempcomp_is_off_for_the_run_and_restored_after_it() {
        let directory =
            std::env::temp_dir().join(format!("achernar-focus-tc-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let store = LocalStore::open(directory.join("state.db").to_str().unwrap())
            .await
            .unwrap();
        let state = AppState::new(
            crate::config::FocusConfig::default(),
            crate::config::CameraConfig::default(),
            store,
            crate::storage::frames_repo::FrameStore::new(directory.join("previews")),
            crate::storage::capture_store::CaptureStore::new(directory.join("captures")),
            "tc-rig".into(),
            directory.join("data"),
            None,
        );
        let mut snapshot = FocuserSnapshot {
            logical_position: Some(1_000),
            position_source: PositionSource::Device,
            ..FocuserSnapshot::default()
        };
        snapshot.capabilities.absolute = true;
        snapshot.capabilities.temp_comp_available = true;
        snapshot.state.temp_comp = Observed::fresh(true, Source::Property, Utc::now(), 1);
        let (focuser, focuser_fake) = fake_focuser(snapshot);
        state.equipment.attach(focuser.clone());
        let op = AutofocusOp::authorize(&state, focuser, GuidingDuringAutofocus::Keep).unwrap();
        assert!(hold_tempcomp_off(&state, &op).await.unwrap());
        assert_eq!(
            focuser_fake.commands(),
            vec![FocuserCmd::SetTempComp(false)]
        );
        let mut warnings = Vec::new();
        release_tempcomp(&state, &op, true, &mut warnings).await;
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(
            focuser_fake.commands(),
            vec![
                FocuserCmd::SetTempComp(false),
                FocuserCmd::SetTempComp(true)
            ]
        );
        // Already off, or not available: nothing to hold.
        focuser_fake.update(|s| {
            s.state.temp_comp = Observed::fresh(false, Source::Property, Utc::now(), 2)
        });
        assert!(!hold_tempcomp_off(&state, &op).await.unwrap());
        op.finish(&state, OpOutcome::Succeeded);
        std::fs::remove_dir_all(&directory).ok();
    }
}

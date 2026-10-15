//! Focus API endpoints — Section 7.5.2–7.5.4.
//! All /api/v1/focus/* endpoints.

use crate::api::api_error::ApiError;
use crate::control::safety_guard::{FocusSafetyGuard, SafetyVerdict};
use crate::error::{AchernarError, FocusError, HardwareError};
use crate::focus::focus_agent::{FocusRunContext, FocusRunState, FocusSample};
use crate::hardware::focuser::ConnectedFocuser;
use crate::hardware::ols_ffi::{OlsBayerType, OlsCamOptionId, OlsStreamType};
use crate::state::{AppState, CameraActivity};
use crate::storage::capture_archive::ArchiveReservation;
use crate::storage::capture_store::{
    AssetRole, CommittedCapture, PendingAsset, PendingCapture, TemperatureState,
};
use crate::storage::local_store::LocalStore;
use crate::vision::observed_frame::ObservedFrame;
use crate::vision::shared_observation::SharedObservation;
use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use std::sync::{atomic::Ordering, Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Serialize)]
pub struct FocusRunResponse {
    pub run_id: String,
    pub state: String,
    pub best_position: Option<i32>,
    pub best_hfr: Option<f64>,
    pub samples_count: u32,
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

    validate_simulation_focus_run(&state).map_err(focus_api_error)?;
    ensure_streaming(&state).await.map_err(focus_api_error)?;
    let start_position = current_focuser_position(&state).map_err(focus_api_error)?;
    let rig_id = connected_rig_id(&state);
    let filter_name = req.filter_name.clone();

    let measure_state = state.clone();
    let measure_config = config.clone();
    let measure_rig_id = rig_id.clone();
    let measure_filter_name = filter_name.clone();
    let focus_safety = Arc::new(Mutex::new(FocusSafetyGuard::new(config.safety.clone())));
    let measure_focus_safety = focus_safety.clone();
    let mut agent = state.focus_agent.lock().await;
    let run_result = agent
        .run_autofocus(
            &config,
            rig_id,
            filter_name,
            start_position,
            move |position| {
                let state = measure_state.clone();
                let config = measure_config.clone();
                let rig_id = measure_rig_id.clone();
                let filter_name = measure_filter_name.clone();
                let focus_safety = measure_focus_safety.clone();
                async move {
                    measure_at_position(
                        &state,
                        &config,
                        &focus_safety,
                        position,
                        &rig_id,
                        filter_name.as_deref(),
                    )
                    .await
                }
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
            if let Some(failed_run) = failed_run {
                if let Err(store_err) = persist_focus_run(&state.local_store, &failed_run).await {
                    tracing::error!(
                        "Failed to persist failed autofocus run {}: {store_err}",
                        failed_run.run_id
                    );
                }
            }
            return Err(focus_api_error(err));
        }
    };
    drop(agent);

    // Provenance persistence is important, but a local database fault must not
    // strand the focuser at the last sweep position. Preserve what we can, then
    // complete the hardware action and record that action only after it succeeds.
    if let Err(err) = persist_focus_run(&state.local_store, &run).await {
        tracing::error!("Failed to persist autofocus run {}: {err}", run.run_id);
    }

    if let Some(best_position) = run.best_position {
        let from_position = current_focuser_position(&state).unwrap_or(run.start_position);
        if let Err(err) =
            move_focuser_to_position(&state, &config, &focus_safety, best_position).await
        {
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
            if let Err(store_err) = persist_focus_run(&state.local_store, &failed_run).await {
                tracing::error!(
                    "Failed to update autofocus run {} after move failure: {store_err}",
                    failed_run.run_id
                );
            }
            if let Err(store_err) = persist_focus_action(
                &state.local_store,
                &failed_run,
                from_position,
                best_position,
                "move_to_best_failed",
            )
            .await
            {
                tracing::error!("Failed to persist unsuccessful focus action: {store_err}");
            }
            return Err(focus_api_error(err));
        }
        if let Err(err) = persist_focus_action(
            &state.local_store,
            &run,
            from_position,
            best_position,
            "classical_hfr_curve_fit",
        )
        .await
        {
            tracing::error!("Focused successfully but failed to persist focus action: {err}");
        }

        // This stage permits simulator sweeps only. Their curve fit must not
        // update a real rig's drift model or focuser scale calibration.
    }

    Ok(Json(response_from_run(&run)))
}

/// POST /api/v1/focus/abort — abort current focus run
pub async fn abort_focus_run(State(state): State<AppState>) -> Result<StatusCode, StatusCode> {
    state.focus_abort_requested.store(true, Ordering::SeqCst);
    if let Ok(mut agent) = state.focus_agent.try_lock() {
        agent.abort();
    }

    if let Ok(mut focuser) = state.focuser.lock() {
        if crate::hardware::health::require_connected(
            &state,
            crate::hardware::health::DeviceKind::Focuser,
        )
        .is_ok()
        {
            if let Some(focuser) = focuser.as_mut() {
                let _ = focuser.abort();
            }
        }
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
    // answer. A DSLR has no sensor thermometer: `missing` is reported as
    // missing rather than substituted, because a prediction built on an
    // invented temperature is indistinguishable from one built on a real
    // reading once it has been written down.
    let (temperature, temperature_state, temperature_source) =
        crate::api::camera::read_temperature(&state);

    let Some(temperature_c) = temperature else {
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
                       from converged autofocus runs; run one or two more.",
            "update_count": update_count,
            "temperature_c": temperature_c,
            "temperature_state": temperature_state,
            "refocus_threshold_steps": threshold.map(|(steps, _)| steps),
        })));
    }

    // Audit the evaluation. A failed write must not fail the request: the
    // operator asked what focus is doing, not to write a row.
    if let Err(err) = crate::storage::focus_drift_repo::record_event(
        state.local_store.pool(),
        None,
        &prediction,
        Some(temperature_c),
    )
    .await
    {
        tracing::warn!("Failed to record focus_drift_event: {err}");
    }

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

/// Fold a converged autofocus result into the drift filter and snapshot it.
///
/// Called only on success: the filter models where best focus *is*, so feeding
/// it the endpoint of a run that never converged would teach it that best focus
/// is wherever the sweep gave up.
///
/// Every failure here is logged and swallowed. The forecaster is advisory; an
/// autofocus run that physically converged has succeeded whether or not a
/// prediction model got updated.
async fn record_drift_measurement(
    state: &AppState,
    position: i32,
    temperature: Option<f64>,
    samples: &[FocusSample],
) {
    let Some(forecaster) = state.focus_drift.clone() else {
        return;
    };

    // Independent of the temperature check below: the V-curve measures the
    // focuser's mechanical scale, which has nothing to do with thermometers.
    // A run that converged in a rig with no thermometer can still teach the
    // system what a focuser step is worth.
    measure_focuser_scale(state, &forecaster, samples).await;

    let Some(temperature_c) = temperature else {
        tracing::debug!(
            "Focus drift: autofocus converged at {position} but no temperature was available; \
             not folding it in (a position without its temperature teaches the filter nothing \
             about thermal drift)"
        );
        return;
    };

    let snapshot = {
        let mut guard = forecaster.lock().await;
        guard.update(position as f64, temperature_c, chrono::Utc::now());
        guard.to_state()
    };

    if let Some(path) = state.focus_drift_state_path.as_ref() {
        if let Err(err) = crate::storage::focus_drift_repo::save_state(path, &snapshot) {
            tracing::warn!(
                "Failed to snapshot focus drift state to {}: {err}",
                path.display()
            );
        }
    }
}

fn response_from_run(run: &FocusRunContext) -> FocusRunResponse {
    FocusRunResponse {
        run_id: run.run_id.to_string(),
        state: format!("{:?}", run.state),
        best_position: run.best_position,
        best_hfr: run.best_hfr,
        samples_count: run.samples.len() as u32,
    }
}

/// Physical autofocus remains unavailable until the analysis, timing,
/// settings, stellar-sweep and driver qualification gates have passed.
fn simulation_focus_allowed(camera_driver: Option<&str>, simulator_focuser: bool) -> bool {
    camera_driver == Some("sim") && simulator_focuser
}

fn validate_simulation_focus_run(state: &AppState) -> crate::error::Result<()> {
    let simulator_focuser = {
        let focuser = state.focuser.lock().unwrap();
        match focuser.as_ref().ok_or(FocusError::NoFocuser)? {
            ConnectedFocuser::Simulator(_) => true,
            ConnectedFocuser::Hardware(_) => false,
        }
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

async fn ensure_streaming(state: &AppState) -> crate::error::Result<()> {
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

fn current_focuser_position(state: &AppState) -> crate::error::Result<i32> {
    let focuser = state.focuser.lock().unwrap();
    crate::hardware::health::require_connected(state, crate::hardware::health::DeviceKind::Focuser)
        .map_err(|_| FocusError::NoFocuser)?;
    let focuser = focuser.as_ref().ok_or(FocusError::NoFocuser)?;
    focuser
        .position()
        .map_err(|err| HardwareError::Communication(err.to_string()).into())
}

async fn measure_at_position(
    state: &AppState,
    config: &crate::config::FocusConfig,
    focus_safety: &Arc<Mutex<FocusSafetyGuard>>,
    position: i32,
    rig_id: &str,
    filter_name: Option<&str>,
) -> crate::error::Result<Vec<FocusSample>> {
    abort_if_requested(state)?;
    // Storage backpressure must be visible before even a simulator focuser
    // moves. The reservation is released if movement or settling fails.
    let mut first_archive = Some(state.capture_archive.reserve()?);
    move_focuser_to_position(state, config, focus_safety, position).await?;
    abort_if_requested(state)?;
    tokio::time::sleep(Duration::from_millis(
        config.settle_ms.max(config.safety.settle_ms),
    ))
    .await;

    // Keep every frame at this position as its own sample with its own preserved
    // frame. Averaging here would destroy the per-position scatter that the FQE
    // learns from (and that the curve fit weights) — Section 7.6.1 / 7.7.5.
    let samples_per_position = config.samples_per_position.max(1);
    let mut samples = Vec::with_capacity(samples_per_position as usize);
    for _ in 0..samples_per_position {
        let archive = match first_archive.take() {
            Some(archive) => archive,
            None => state.capture_archive.reserve()?,
        };
        let observed = wait_for_fresh_frame(state).await?;
        let (committed, temperature) =
            persist_observed_frame(state, archive, position, rig_id, filter_name, &observed)
                .await?;
        samples.push(sample_from_observation(
            position,
            &observed.observation,
            &committed,
            temperature,
        )?);
    }

    Ok(samples)
}

/// Archive the exact simulator sensor buffer and its analysis derivative.
/// A missing local commit aborts measurement before a sample is accepted.
async fn persist_observed_frame(
    state: &AppState,
    archive: ArchiveReservation,
    position: i32,
    rig_id: &str,
    filter_name: Option<&str>,
    observed: &ObservedFrame,
) -> crate::error::Result<(CommittedCapture, Option<f64>)> {
    if observed.frame.stream_generation != state.current_camera_generation() {
        return Err(FocusError::RunFailed(
            "Autofocus observation belongs to an inactive camera stream".to_string(),
        )
        .into());
    }
    let (cadence_s, camera_name) = {
        let camera = state.camera.lock().unwrap();
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
        (
            parameter.cur_val / 1000.0,
            Some(
                camera
                    .name()
                    .map_err(|error| HardwareError::Communication(error.to_string()))?,
            ),
        )
    };
    let focuser_name = state.focuser_name.lock().ok().and_then(|name| name.clone());
    let (temperature, temperature_state, temperature_source) =
        crate::api::camera::read_temperature(state);
    let pending = simulation_focus_bundle(
        observed,
        SimulationFocusMetadata {
            position,
            rig_id: rig_id.to_string(),
            filter_name: filter_name.map(str::to_string),
            camera_name,
            focuser_name,
            cadence_s,
            temperature,
            temperature_state,
            temperature_source,
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
    Ok((committed, temperature))
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
    let quality = autofocus_quality_json(
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
    let fits_bytes = crate::fits::encode_capture(frame, &fits_meta).map_err(|error| {
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
    Ok(PendingCapture {
        capture_id: uuid::Uuid::new_v4().to_string(),
        frame_id: uuid::Uuid::new_v4().to_string(),
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
        filter_name: meta.filter_name,
        focuser_position: Some(meta.position as i64),
        temperature: meta.temperature,
        temperature_source: meta.temperature_source,
        temperature_state: meta.temperature_state,
        acquisition: serde_json::json!({
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
        }),
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
            "background_adu": observation.background_adu,
            "background_sigma": observation.background_sigma,
            "saturation_fraction": observation.saturation_fraction,
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

async fn move_focuser_to_position(
    state: &AppState,
    config: &crate::config::FocusConfig,
    focus_safety: &Arc<Mutex<FocusSafetyGuard>>,
    target_position: i32,
) -> crate::error::Result<()> {
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
    // forever. A relative focuser's position is OLS's accumulated command count,
    // so arrival there is assumed, not verified.
    let mut previous_remaining: Option<i32> = None;
    let tolerance = config.safety.arrival_tolerance_steps.max(0);
    loop {
        abort_if_requested(state)?;
        {
            let mut focuser = state.focuser.lock().unwrap();
            crate::hardware::health::require_connected(
                state,
                crate::hardware::health::DeviceKind::Focuser,
            )
            .map_err(|_| FocusError::NoFocuser)?;
            let focuser = focuser.as_mut().ok_or(FocusError::NoFocuser)?;
            let current = focuser
                .position()
                .map_err(|err| HardwareError::Communication(err.to_string()))?;
            let delta = target - current;
            if delta.abs() <= tolerance {
                return Ok(());
            }
            if previous_remaining.is_some_and(|previous| delta.abs() >= previous.abs()) {
                return Err(HardwareError::Communication(format!(
                    "Focuser stopped at {current} without progressing toward {target}"
                ))
                .into());
            }
            previous_remaining = Some(delta);

            let verdict = focus_safety
                .lock()
                .map_err(|_| {
                    FocusError::SafetyViolation("Focus safety guard lock was poisoned".to_string())
                })?
                .validate_focus_action(current, delta, 1.0)?;
            let step = match verdict {
                SafetyVerdict::Approved => delta,
                SafetyVerdict::Modified { clamped, .. } => clamped,
                SafetyVerdict::Rejected { reason } => {
                    return Err(FocusError::SafetyViolation(reason).into());
                }
            };
            let absolute = focuser
                .is_absolute()
                .map_err(|err| HardwareError::Communication(err.to_string()))?;
            if absolute {
                focuser
                    .move_absolute(current + step)
                    .map_err(|err| HardwareError::Communication(err.to_string()))?;
            } else {
                focuser
                    .move_relative(step)
                    .map_err(|err| HardwareError::Communication(err.to_string()))?;
            }
        }

        // A sleep alone is not completion: a slow or belt-reduced focuser can
        // still be moving, which would put the next exposure mid-move. Wait
        // for the device to report stopped, then settle, then re-read the
        // position at the top of the loop to check arrival.
        let stopped = wait_until_stopped(
            || abort_if_requested(state),
            || {
                let focuser = state.focuser.lock().unwrap();
                let focuser = focuser.as_ref().ok_or(FocusError::NoFocuser)?;
                focuser
                    .is_moving()
                    .map_err(|err| HardwareError::Communication(err.to_string()).into())
            },
            Duration::from_millis(config.safety.move_timeout_ms.max(100)),
            Duration::from_millis(100),
        )
        .await;
        if let Err(err) = stopped {
            if let Some(focuser) = state.focuser.lock().unwrap().as_mut() {
                let _ = focuser.abort();
            }
            return Err(err);
        }
        tokio::time::sleep(Duration::from_millis(config.safety.settle_ms.max(50))).await;
    }
}

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

/// Wait for a genuinely new camera frame to be observed, keyed on the camera
/// `frame_counter` (not the vision loop's processing timestamp). This guarantees
/// each autofocus sample comes from a distinct exposure — reprocessing one
/// camera frame cannot masquerade as several samples — and returns the exact
/// frame+observation envelope. On the deadline it returns a Timeout error rather
/// than silently accepting the stale previous observation as a success.
async fn wait_for_fresh_frame(state: &AppState) -> crate::error::Result<ObservedFrame> {
    let baseline = state
        .last_observed_frame
        .read()
        .ok()
        .and_then(|slot| slot.as_ref().map(|observed| observed.frame_counter));
    let deadline = Instant::now() + Duration::from_secs(5);

    loop {
        abort_if_requested(state)?;
        if let Some(observed) = state
            .last_observed_frame
            .read()
            .ok()
            .and_then(|slot| slot.clone())
        {
            if Some(observed.frame_counter) != baseline {
                return Ok(observed);
            }
        }

        if Instant::now() >= deadline {
            return Err(HardwareError::Timeout(
                "No fresh vision frame arrived for autofocus measurement".to_string(),
            )
            .into());
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn abort_if_requested(state: &AppState) -> crate::error::Result<()> {
    if state.focus_abort_requested.load(Ordering::SeqCst) {
        Err(FocusError::RunFailed("Autofocus aborted".to_string()).into())
    } else {
        Ok(())
    }
}

/// `temperature` is `None` when nothing could measure one. Left NULL rather
/// than filled with a default: `focus_samples.temperature` feeds the drift
/// forecaster, and a column that mixes readings with stand-ins is worse than a
/// column with holes in it — the holes are at least visible.
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

async fn persist_focus_run(store: &LocalStore, run: &FocusRunContext) -> crate::error::Result<()> {
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
        sqlx::query(
            r#"
            INSERT INTO focus_samples (
                run_id, position, hfr, hfd, fwhm, star_count, quality_score,
                temperature, captured_at, frame_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&run_id)
        .bind(sample.position)
        .bind(sample.hfr)
        .bind(sample.hfd)
        .bind(sample.fwhm)
        .bind(sample.star_count as i64)
        .bind(sample.quality_score)
        .bind(sample.temperature)
        .bind(sample.timestamp.to_rfc3339())
        .bind(&sample.frame_id)
        .execute(&mut *transaction)
        .await
        .map_err(storage_error)?;
    }

    transaction.commit().await.map_err(storage_error)?;
    Ok(())
}

async fn persist_focus_action(
    store: &LocalStore,
    run: &FocusRunContext,
    from_position: i32,
    to_position: i32,
    reason: &str,
) -> crate::error::Result<()> {
    sqlx::query(
        r#"
        INSERT INTO focus_actions (run_id, action, from_position, to_position, steps, reason)
        VALUES (?, 'move_to_best', ?, ?, ?, ?)
        "#,
    )
    .bind(run.run_id.to_string())
    .bind(from_position)
    .bind(to_position)
    .bind(to_position - from_position)
    .bind(reason)
    .execute(store.pool())
    .await
    .map_err(storage_error)?;
    Ok(())
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

        persist_focus_run(&store, &run).await.unwrap();

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

    async fn move_fixture(
        focuser: crate::hardware::focuser::SimFocuser,
    ) -> (
        std::path::PathBuf,
        AppState,
        crate::config::FocusConfig,
        Arc<Mutex<FocusSafetyGuard>>,
    ) {
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
        focus.safety.move_timeout_ms = 100;
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
        let connected = ConnectedFocuser::Simulator(focuser);
        let snapshot = crate::hardware::health::focuser_snapshot(
            "fixture focuser".into(),
            &connected.health_status().unwrap(),
        );
        *state.focuser.lock().unwrap() = Some(connected);
        state.devices.write().unwrap().connected(
            crate::hardware::health::DeviceKind::Focuser,
            "fixture".into(),
            Some(snapshot),
        );
        let guard = Arc::new(Mutex::new(FocusSafetyGuard::new(focus.safety.clone())));
        (directory, state, focus, guard)
    }

    fn sim_position(state: &AppState) -> i32 {
        state
            .focuser
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .position()
            .unwrap()
    }

    #[tokio::test]
    async fn move_reaches_a_distant_target_through_clamped_steps() {
        let sim = crate::hardware::focuser::SimFocuser::new(0);
        let (directory, state, config, guard) = move_fixture(sim).await;
        move_focuser_to_position(&state, &config, &guard, 1_700)
            .await
            .unwrap();
        assert_eq!(sim_position(&state), 1_700);
        std::fs::remove_dir_all(directory).ok();
    }

    #[tokio::test]
    async fn move_fails_when_the_focuser_stops_short_of_the_target() {
        let sim = crate::hardware::focuser::SimFocuser::with_max_range(0, 1_000);
        let (directory, state, config, guard) = move_fixture(sim).await;
        let err = move_focuser_to_position(&state, &config, &guard, 2_000)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("without progressing"), "{err}");
        assert_eq!(sim_position(&state), 1_000);
        std::fs::remove_dir_all(directory).ok();
    }

    #[tokio::test]
    async fn move_halts_a_focuser_that_never_reports_stopped() {
        let mut sim = crate::hardware::focuser::SimFocuser::new(0);
        sim.set_stuck_moving(true);
        let (directory, state, config, guard) = move_fixture(sim).await;
        let err = move_focuser_to_position(&state, &config, &guard, 300)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("still moving"), "{err}");
        let moving = state
            .focuser
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .is_moving()
            .unwrap();
        assert!(!moving, "the timed-out focuser must have been halted");
        std::fs::remove_dir_all(directory).ok();
    }
}

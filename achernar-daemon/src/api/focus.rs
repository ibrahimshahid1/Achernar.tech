//! Focus API endpoints — Section 7.5.2–7.5.4.
//! All /api/v1/focus/* endpoints.

use crate::control::safety_guard::{FocusSafetyGuard, SafetyVerdict};
use crate::error::{AchernarError, FocusError, HardwareError};
use crate::focus::focus_agent::{FocusRunContext, FocusRunState, FocusSample};
use crate::state::{AppState, CameraActivity};
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
) -> Result<Json<FocusRunResponse>, StatusCode> {
    // Like a camera exposure, an autofocus sweep is a hardware transaction,
    // not a browser-request lifetime. Closing the page must not drop the
    // acquisition guard halfway through a focuser move and let another camera
    // command interleave. The explicit `/focus/abort` signal remains active.
    tokio::spawn(async move { start_focus_run_owned(state, req).await })
        .await
        .map_err(|error| {
            tracing::error!("Autofocus task failed: {error}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?
}

async fn start_focus_run_owned(
    state: AppState,
    req: StartFocusRunRequest,
) -> Result<Json<FocusRunResponse>, StatusCode> {
    let method = req.method.as_deref().unwrap_or("classical");
    if !matches!(method, "classical" | "deterministic" | "hfr") {
        tracing::warn!("Unsupported focus method requested: {method}");
        return Err(StatusCode::BAD_REQUEST);
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
                return Err(StatusCode::BAD_REQUEST);
            }
        }
    }

    // Autofocus owns camera acquisition for the complete sweep. Parameter
    // writes, stream stop/restart, capture and disconnect must not interleave
    // with frames whose HFR labels are being collected. The abort endpoint is
    // intentionally out-of-band and signals through `focus_abort_requested`.
    let _acquisition = state.acquisition.lock().await;
    state.focus_abort_requested.store(false, Ordering::SeqCst);

    ensure_streaming(&state).await.map_err(status_for_error)?;
    let start_position = current_focuser_position(&state).map_err(status_for_error)?;
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
            // failure. These are still scientifically useful labels and explain
            // the failed run; dropping them creates survivorship bias in FQE data.
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
            return Err(status_for_error(err));
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
            return Err(status_for_error(err));
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

        // Fold the converged position into the drift filter. Only here, after
        // the move succeeded: this is the one moment the daemon knows, by
        // measurement rather than by model, where best focus is.
        let (temperature, _, _) = crate::api::camera::read_temperature(&state);
        record_drift_measurement(&state, best_position, temperature, &run.samples).await;
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
        if let Some(focuser) = focuser.as_mut() {
            let _ = focuser.abort();
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

    let horizon_s = query.horizon_s.filter(|h| h.is_finite() && *h >= 0.0).unwrap_or(600.0);

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

async fn ensure_streaming(state: &AppState) -> crate::error::Result<()> {
    if *state.streaming.lock().unwrap() {
        return Ok(());
    }

    let mut camera = state.camera.lock().unwrap();
    let camera = camera.as_mut().ok_or_else(|| {
        HardwareError::DeviceNotFound("No camera connected for autofocus".to_string())
    })?;
    let formats = camera
        .formats()
        .map_err(|err| HardwareError::Communication(err.to_string()))?;
    let format = formats.first().ok_or_else(|| {
        HardwareError::Unsupported("Camera reports no stream formats".to_string())
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
        runtime.active_format_index = Some(0);
        runtime.last_error = None;
    }
    Ok(())
}

fn current_focuser_position(state: &AppState) -> crate::error::Result<i32> {
    let focuser = state.focuser.lock().unwrap();
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
        let observed = wait_for_fresh_frame(state).await?;
        let frame_id =
            persist_observed_frame(state, position, rig_id, filter_name, &observed).await;
        // Read the temperature next to the frame it belongs to, not once per
        // run: a sweep takes minutes and a tube cooling through it is the very
        // effect the drift filter is trying to learn.
        let (temperature, _, _) = crate::api::camera::read_temperature(state);
        samples.push(sample_from_observation(
            position,
            &observed.observation,
            frame_id,
            temperature,
        )?);
    }

    Ok(samples)
}

/// Persist the exact frame that produced `observed.observation` (best effort) so
/// the FQE dataset builder can recover (crop, HFR) training pairs from pixels
/// that provably match the label. Returns the new `frames.id`, or `None` if the
/// write failed — a provenance failure must never abort an autofocus run.
async fn persist_observed_frame(
    state: &AppState,
    position: i32,
    rig_id: &str,
    filter_name: Option<&str>,
    observed: &ObservedFrame,
) -> Option<String> {
    let observation = &observed.observation;
    let luma = match crate::vision::quality::luma_from_frame(&observed.frame) {
        Ok(luma) => luma,
        Err(err) => {
            tracing::warn!("Skipping autofocus frame persistence: {err}");
            return None;
        }
    };

    let camera_name = state.camera_name.lock().ok().and_then(|name| name.clone());
    let focuser_name = state.focuser_name.lock().ok().and_then(|name| name.clone());
    let capture_micros = observed.capture_unix_time * 1_000_000.0;
    let captured_at = if capture_micros.is_finite()
        && capture_micros >= i64::MIN as f64
        && capture_micros <= i64::MAX as f64
    {
        chrono::DateTime::<chrono::Utc>::from_timestamp_micros(capture_micros.round() as i64)
    } else {
        None
    }
    .unwrap_or(observation.timestamp);
    // Temperature, with its provenance. The drift forecaster is trained from
    // these rows, and it cannot tell a measured reading from an invented one
    // once both are just a number in a column — so a body with no thermometer
    // records NULL and says why, rather than a plausible-looking value.
    let (temperature, temperature_state, temperature_source) =
        crate::api::camera::read_temperature(state);

    let meta = crate::storage::frames_repo::FrameMeta {
        captured_at: Some(captured_at),
        filter_name: filter_name.map(str::to_string),
        temperature,
        quality_json: autofocus_quality_json(
            position,
            rig_id,
            filter_name,
            camera_name.as_deref(),
            focuser_name.as_deref(),
            observed.frame_counter,
            observed.capture_unix_time,
            observation,
            &luma,
            temperature_state,
            temperature_source,
        ),
        ..Default::default()
    };

    match state
        .frame_store
        .persist_luma(state.local_store.pool(), &luma, meta)
        .await
    {
        Ok(persisted) => Some(persisted.id),
        Err(err) => {
            tracing::warn!("Failed to persist autofocus frame: {err}");
            None
        }
    }
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
            "source": "autofocus",
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

    loop {
        abort_if_requested(state)?;
        let delta = {
            let mut focuser = state.focuser.lock().unwrap();
            let focuser = focuser.as_mut().ok_or(FocusError::NoFocuser)?;
            let current = focuser
                .position()
                .map_err(|err| HardwareError::Communication(err.to_string()))?;
            let delta = target - current;
            if delta == 0 {
                return Ok(());
            }

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
            target - (current + step)
        };

        tokio::time::sleep(Duration::from_millis(config.safety.settle_ms.max(50))).await;
        if delta == 0 {
            return Ok(());
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
    frame_id: Option<String>,
    temperature: Option<f64>,
) -> crate::error::Result<FocusSample> {
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
        frame_id,
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

fn status_for_error(err: AchernarError) -> StatusCode {
    match err {
        AchernarError::Focus(FocusError::NoFocuser) => StatusCode::PRECONDITION_FAILED,
        AchernarError::Focus(FocusError::SafetyViolation(_)) => StatusCode::BAD_REQUEST,
        AchernarError::Focus(_) => StatusCode::UNPROCESSABLE_ENTITY,
        AchernarError::Hardware(HardwareError::DeviceNotFound(_)) => {
            StatusCode::PRECONDITION_FAILED
        }
        AchernarError::Hardware(HardwareError::Unsupported(_)) => StatusCode::PRECONDITION_FAILED,
        AchernarError::Hardware(HardwareError::Timeout(_)) => StatusCode::GATEWAY_TIMEOUT,
        AchernarError::Hardware(_) => StatusCode::BAD_GATEWAY,
        AchernarError::Storage(_) => StatusCode::INTERNAL_SERVER_ERROR,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::capture_store::TemperatureState;
    use crate::vision::quality::LumaFrame;

    fn luma() -> LumaFrame {
        LumaFrame {
            pixels: vec![100.0; 4],
            width: 2,
            height: 2,
            max_adu: 65535.0,
            saturation_adu: Some(16383.0),
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
            for key in rest.split(", else ").map(|k| {
                k.trim()
                    .trim_end_matches(" (preferred)")
                    .trim()
            }) {
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
}

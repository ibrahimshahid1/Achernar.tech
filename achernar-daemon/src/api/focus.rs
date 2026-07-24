//! Focus API endpoints — Section 7.5.2–7.5.4.
//! All /api/v1/focus/* endpoints.

use crate::control::safety_guard::{FocusSafetyGuard, SafetyVerdict};
use crate::error::{AchernarError, FocusError, HardwareError};
use crate::focus::focus_agent::{FocusRunContext, FocusRunState, FocusSample};
use crate::state::AppState;
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
}

/// POST /api/v1/focus/run — start a new autofocus run
pub async fn start_focus_run(
    State(state): State<AppState>,
    Json(req): Json<StartFocusRunRequest>,
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

/// GET /api/v1/focus/drift — get current drift prediction
pub async fn get_drift_prediction(
    State(_state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    Ok(Json(serde_json::json!({
        "status": "unavailable",
        "detail": "Deterministic autofocus is enabled; drift prediction model is not active.",
    })))
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
    camera
        .start_stream(format, state.latest_frame.clone())
        .map_err(|err| HardwareError::Communication(err.to_string()))?;
    *state.streaming.lock().unwrap() = true;
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
        samples.push(sample_from_observation(
            position,
            &observed.observation,
            frame_id,
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
    let meta = crate::storage::frames_repo::FrameMeta {
        captured_at: Some(captured_at),
        filter_name: filter_name.map(str::to_string),
        temperature: None,
        quality_json: serde_json::json!({
            "source": "autofocus",
            "rig_id": rig_id,
            "camera_id": camera_name,
            "focuser_id": focuser_name,
            "filter_name": filter_name,
            "focuser_position": position,
            // Provenance: the camera frame these pixels and this label came from.
            "frame_counter": observed.frame_counter,
            "capture_unix_time": observed.capture_unix_time,
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
            "max_adu": luma.max_adu,
            "observation_valid": observation.observation_valid,
        }),
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

fn sample_from_observation(
    position: i32,
    observation: &SharedObservation,
    frame_id: Option<String>,
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
        temperature: None,
        timestamp: observation.timestamp,
        frame_id,
    })
}

async fn persist_focus_run(store: &LocalStore, run: &FocusRunContext) -> crate::error::Result<()> {
    let run_id = run.run_id.to_string();
    let mut transaction = store.pool().begin().await.map_err(storage_error)?;
    sqlx::query(
        r#"
        INSERT INTO focus_runs (
            id, rig_id, filter_name, start_position, best_position, best_hfr,
            state, temperature, started_at, completed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            rig_id = excluded.rig_id,
            filter_name = excluded.filter_name,
            start_position = excluded.start_position,
            best_position = excluded.best_position,
            best_hfr = excluded.best_hfr,
            state = excluded.state,
            temperature = excluded.temperature,
            started_at = excluded.started_at,
            completed_at = excluded.completed_at
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

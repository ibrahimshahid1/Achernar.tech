//! Focus API endpoints — Section 7.5.2–7.5.4.
//! All /api/v1/focus/* endpoints.

use crate::error::{AchernarError, FocusError, HardwareError};
use crate::focus::focus_agent::{FocusRunContext, FocusRunState, FocusSample};
use crate::state::AppState;
use crate::storage::local_store::LocalStore;
use crate::vision::shared_observation::SharedObservation;
use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use std::sync::atomic::Ordering;
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

    let measure_state = state.clone();
    let measure_config = config.clone();
    let mut agent = state.focus_agent.lock().await;
    let run = agent
        .run_autofocus(&config, req.filter_name, start_position, move |position| {
            let state = measure_state.clone();
            let config = measure_config.clone();
            async move { measure_at_position(&state, &config, position).await }
        })
        .await
        .map_err(status_for_error)?;
    drop(agent);

    persist_focus_run(&state.local_store, &run)
        .await
        .map_err(status_for_error)?;

    if let Some(best_position) = run.best_position {
        move_focuser_to_position(&state, &config, best_position)
            .await
            .map_err(status_for_error)?;
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
    position: i32,
) -> crate::error::Result<FocusSample> {
    abort_if_requested(state)?;
    move_focuser_to_position(state, config, position).await?;
    abort_if_requested(state)?;
    tokio::time::sleep(Duration::from_millis(
        config.settle_ms.max(config.safety.settle_ms),
    ))
    .await;

    let samples_per_position = config.samples_per_position.max(1);
    let mut hfr_sum = 0.0;
    let mut hfd_sum = 0.0;
    let mut star_count_sum = 0_u32;
    let mut temperature = None;

    for _ in 0..samples_per_position {
        let observation = wait_for_fresh_observation(state).await?;
        let sample = sample_from_observation(position, &observation)?;
        hfr_sum += sample.hfr;
        hfd_sum += sample.fwhm.unwrap_or(sample.hfr * 2.0);
        star_count_sum += sample.star_count;
        temperature = sample.temperature;
    }

    let n = samples_per_position as f64;
    Ok(FocusSample {
        position,
        hfr: hfr_sum / n,
        fwhm: Some(hfd_sum / n),
        star_count: (star_count_sum as f64 / n).round() as u32,
        quality_score: None,
        temperature,
        timestamp: chrono::Utc::now(),
    })
}

async fn move_focuser_to_position(
    state: &AppState,
    config: &crate::config::FocusConfig,
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

            let max_step = config.safety.max_step_single.abs().max(1);
            let step = delta.clamp(-max_step, max_step);
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

async fn wait_for_fresh_observation(state: &AppState) -> crate::error::Result<SharedObservation> {
    let baseline = state
        .vision_observation
        .read()
        .ok()
        .and_then(|obs| obs.as_ref().map(|obs| obs.timestamp));
    let deadline = Instant::now() + Duration::from_secs(5);

    loop {
        abort_if_requested(state)?;
        if let Some(observation) = state
            .vision_observation
            .read()
            .ok()
            .and_then(|obs| obs.clone())
        {
            if Some(observation.timestamp) != baseline || Instant::now() >= deadline {
                return Ok(observation);
            }
        }

        if Instant::now() >= deadline {
            return Err(HardwareError::Timeout(
                "No vision observation arrived for autofocus measurement".to_string(),
            )
            .into());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
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
) -> crate::error::Result<FocusSample> {
    let hfr = observation.hfr as f64;
    if !observation.observation_valid || !hfr.is_finite() || hfr <= 0.0 {
        return Err(FocusError::RunFailed(
            "Vision observation did not contain a valid focus metric".to_string(),
        )
        .into());
    }

    Ok(FocusSample {
        position,
        hfr,
        fwhm: Some(observation.fwhm as f64),
        star_count: observation.star_count,
        quality_score: Some(observation.focus_quality as f64),
        temperature: None,
        timestamp: observation.timestamp,
    })
}

async fn persist_focus_run(store: &LocalStore, run: &FocusRunContext) -> crate::error::Result<()> {
    let run_id = run.run_id.to_string();
    sqlx::query(
        r#"
        INSERT OR REPLACE INTO focus_runs (
            id, rig_id, filter_name, start_position, best_position, best_hfr,
            state, temperature, started_at, completed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    .execute(store.pool())
    .await
    .map_err(storage_error)?;

    for sample in &run.samples {
        sqlx::query(
            r#"
            INSERT INTO focus_samples (
                run_id, position, hfr, fwhm, star_count, quality_score,
                temperature, captured_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&run_id)
        .bind(sample.position)
        .bind(sample.hfr)
        .bind(sample.fwhm)
        .bind(sample.star_count as i64)
        .bind(sample.quality_score)
        .bind(sample.temperature)
        .bind(sample.timestamp.to_rfc3339())
        .execute(store.pool())
        .await
        .map_err(storage_error)?;
    }

    if let Some(best_position) = run.best_position {
        sqlx::query(
            r#"
            INSERT INTO focus_actions (run_id, action, from_position, to_position, steps, reason)
            VALUES (?, 'move_to_best', ?, ?, ?, 'classical_hfr_curve_fit')
            "#,
        )
        .bind(&run_id)
        .bind(run.start_position)
        .bind(best_position)
        .bind(best_position - run.start_position)
        .execute(store.pool())
        .await
        .map_err(storage_error)?;
    }

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

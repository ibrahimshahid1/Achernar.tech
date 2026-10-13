//! Read-only model output and explicit local review configuration, no hardware commands.
use super::api_error::ApiError;
use crate::state::AppState;
use crate::vision::coc_shadow::{current_context, ConfigureRequest, SaveRequest};
use axum::{extract::State, http::StatusCode, Json};
use serde_json::Value;

pub fn status_value(state: &AppState) -> Value {
    let _barrier = state.camera_observation_barrier.lock().unwrap();
    state.coc_shadow.snapshot(
        current_context(state).as_ref(),
        chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
    )
}

pub async fn status(State(state): State<AppState>) -> Json<Value> {
    Json(status_value(&state))
}

pub fn configure_value(state: &AppState, request: ConfigureRequest) -> Result<Value, String> {
    {
        let _barrier = state.camera_observation_barrier.lock().unwrap();
        state
            .coc_shadow
            .configure(request, current_context(state).as_ref())?;
    }
    Ok(status_value(state))
}

pub async fn configure(
    State(state): State<AppState>,
    Json(request): Json<ConfigureRequest>,
) -> Result<Json<Value>, ApiError> {
    configure_value(&state, request)
        .map(Json)
        .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, "coc_configuration", e))
}

pub async fn save_value(state: AppState, request: SaveRequest) -> Result<Value, String> {
    tokio::task::spawn_blocking(move || state.coc_shadow.save(request, &state))
        .await
        .map_err(|e| e.to_string())?
}

pub async fn save(
    State(state): State<AppState>,
    Json(request): Json<SaveRequest>,
) -> Result<Json<Value>, ApiError> {
    save_value(state, request)
        .await
        .map(Json)
        .map_err(|e| ApiError::new(StatusCode::CONFLICT, "coc_sample_save", e))
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FreezeRequest {
    pub sample_id: String,
}

pub fn freeze_value(state: &AppState, request: FreezeRequest) -> Result<Value, String> {
    let _barrier = state.camera_observation_barrier.lock().unwrap();
    state
        .coc_shadow
        .freeze(&request.sample_id, current_context(state).as_ref())
}

pub async fn freeze(
    State(state): State<AppState>,
    Json(request): Json<FreezeRequest>,
) -> Result<Json<Value>, ApiError> {
    freeze_value(&state, request)
        .map(Json)
        .map_err(|e| ApiError::new(StatusCode::CONFLICT, "coc_review_freeze", e))
}

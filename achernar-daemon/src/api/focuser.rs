use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};

use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct MoveRequest {
    pub steps: i32,
}

#[derive(Debug, Deserialize)]
pub struct MoveToRequest {
    pub position: i32,
}

#[derive(Debug, Serialize)]
pub struct PositionResponse {
    pub position: i32,
}

pub async fn move_relative(
    State(state): State<AppState>,
    Json(req): Json<MoveRequest>,
) -> Result<StatusCode, StatusCode> {
    tracing::info!("Focuser move: {} steps", req.steps);
    let mut focuser = state.focuser.lock().unwrap();
    crate::hardware::health::require_connected(
        &state,
        crate::hardware::health::DeviceKind::Focuser,
    )
    .map_err(|_| StatusCode::PRECONDITION_FAILED)?;
    let focuser = focuser.as_mut().ok_or(StatusCode::PRECONDITION_FAILED)?;
    focuser.move_relative(req.steps).map_err(|e| {
        tracing::error!("Focuser relative move failed: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    Ok(StatusCode::OK)
}

pub async fn move_absolute(
    State(state): State<AppState>,
    Json(req): Json<MoveToRequest>,
) -> Result<StatusCode, StatusCode> {
    tracing::info!("Focuser move to: {}", req.position);
    let mut focuser = state.focuser.lock().unwrap();
    crate::hardware::health::require_connected(
        &state,
        crate::hardware::health::DeviceKind::Focuser,
    )
    .map_err(|_| StatusCode::PRECONDITION_FAILED)?;
    let focuser = focuser.as_mut().ok_or(StatusCode::PRECONDITION_FAILED)?;
    focuser.move_absolute(req.position).map_err(|e| {
        tracing::error!("Focuser absolute move failed: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    Ok(StatusCode::OK)
}

pub async fn get_position(
    State(state): State<AppState>,
) -> Result<Json<PositionResponse>, StatusCode> {
    let focuser = state.focuser.lock().unwrap();
    crate::hardware::health::require_connected(
        &state,
        crate::hardware::health::DeviceKind::Focuser,
    )
    .map_err(|_| StatusCode::PRECONDITION_FAILED)?;
    let focuser = focuser.as_ref().ok_or(StatusCode::PRECONDITION_FAILED)?;
    let position = focuser.position().map_err(|e| {
        tracing::error!("Focuser position failed: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    Ok(Json(PositionResponse { position }))
}

pub async fn abort(State(state): State<AppState>) -> Result<StatusCode, StatusCode> {
    tracing::info!("Focuser abort requested");
    let mut focuser = state.focuser.lock().unwrap();
    crate::hardware::health::require_connected(
        &state,
        crate::hardware::health::DeviceKind::Focuser,
    )
    .map_err(|_| StatusCode::PRECONDITION_FAILED)?;
    let focuser = focuser.as_mut().ok_or(StatusCode::PRECONDITION_FAILED)?;
    focuser.abort().map_err(|e| {
        tracing::error!("Focuser abort failed: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    Ok(StatusCode::OK)
}

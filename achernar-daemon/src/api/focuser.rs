use axum::{http::StatusCode, Json};
use serde::{Deserialize, Serialize};

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

pub async fn move_relative(Json(req): Json<MoveRequest>) -> Result<StatusCode, StatusCode> {
    tracing::info!("Focuser move: {} steps", req.steps);
    // TODO: Wire to OLS focuser driver when focuser bridge is implemented
    Ok(StatusCode::OK)
}

pub async fn move_absolute(Json(req): Json<MoveToRequest>) -> Result<StatusCode, StatusCode> {
    tracing::info!("Focuser move to: {}", req.position);
    Ok(StatusCode::OK)
}

pub async fn get_position() -> Result<Json<PositionResponse>, StatusCode> {
    Ok(Json(PositionResponse { position: 5000 }))
}

pub async fn abort() -> Result<StatusCode, StatusCode> {
    tracing::info!("Focuser abort requested");
    Ok(StatusCode::OK)
}

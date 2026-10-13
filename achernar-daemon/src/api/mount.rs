use axum::{extract::State, http::StatusCode, Json};
use serde::Deserialize;

use crate::{hardware::mount::EqCoord, state::AppState};

#[derive(Debug, Deserialize)]
pub struct SlewRequest {
    pub ra: f64,
    pub dec: f64,
}

#[derive(Debug, Deserialize)]
pub struct SlewTargetRequest {
    pub target: String,
}

#[derive(Debug, Deserialize)]
pub struct TrackingRequest {
    pub enabled: bool,
}

pub async fn slew(
    State(state): State<AppState>,
    Json(req): Json<SlewRequest>,
) -> Result<StatusCode, StatusCode> {
    tracing::info!("Slew requested: RA={}, Dec={}", req.ra, req.dec);
    let mount = state.mount.lock().unwrap();
    crate::hardware::health::require_connected(&state, crate::hardware::health::DeviceKind::Mount)
        .map_err(|_| StatusCode::PRECONDITION_FAILED)?;
    let mount = mount.as_ref().ok_or(StatusCode::PRECONDITION_FAILED)?;
    mount
        .goto(EqCoord {
            ra: req.ra,
            dec: req.dec,
        })
        .map_err(|e| {
            tracing::error!("Mount slew failed: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    Ok(StatusCode::OK)
}

pub async fn slew_target(Json(req): Json<SlewTargetRequest>) -> Result<StatusCode, StatusCode> {
    tracing::info!("Slew to target: {}", req.target);
    Err(StatusCode::NOT_IMPLEMENTED)
}

pub async fn sync(
    State(state): State<AppState>,
    Json(req): Json<SlewRequest>,
) -> Result<StatusCode, StatusCode> {
    tracing::info!("Mount sync: RA={}, Dec={}", req.ra, req.dec);
    let mount = state.mount.lock().unwrap();
    crate::hardware::health::require_connected(&state, crate::hardware::health::DeviceKind::Mount)
        .map_err(|_| StatusCode::PRECONDITION_FAILED)?;
    let mount = mount.as_ref().ok_or(StatusCode::PRECONDITION_FAILED)?;
    mount
        .sync(EqCoord {
            ra: req.ra,
            dec: req.dec,
        })
        .map_err(|e| {
            tracing::error!("Mount sync failed: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    Ok(StatusCode::OK)
}

pub async fn park() -> Result<StatusCode, StatusCode> {
    tracing::info!("Park requested");
    Err(StatusCode::NOT_IMPLEMENTED)
}

pub async fn unpark() -> Result<StatusCode, StatusCode> {
    tracing::info!("Unpark requested");
    Err(StatusCode::NOT_IMPLEMENTED)
}

pub async fn abort(State(state): State<AppState>) -> Result<StatusCode, StatusCode> {
    tracing::info!("Mount abort requested");
    let mount = state.mount.lock().unwrap();
    crate::hardware::health::require_connected(&state, crate::hardware::health::DeviceKind::Mount)
        .map_err(|_| StatusCode::PRECONDITION_FAILED)?;
    let mount = mount.as_ref().ok_or(StatusCode::PRECONDITION_FAILED)?;
    mount.stop().map_err(|e| {
        tracing::error!("Mount abort failed: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    Ok(StatusCode::OK)
}

pub async fn set_tracking(
    State(state): State<AppState>,
    Json(req): Json<TrackingRequest>,
) -> Result<StatusCode, StatusCode> {
    tracing::info!("Set tracking: {}", req.enabled);
    let mount = state.mount.lock().unwrap();
    crate::hardware::health::require_connected(&state, crate::hardware::health::DeviceKind::Mount)
        .map_err(|_| StatusCode::PRECONDITION_FAILED)?;
    let mount = mount.as_ref().ok_or(StatusCode::PRECONDITION_FAILED)?;
    mount.set_tracking(req.enabled).map_err(|e| {
        tracing::error!("Set tracking failed: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    Ok(StatusCode::OK)
}

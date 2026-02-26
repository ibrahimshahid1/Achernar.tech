use axum::{http::StatusCode, Json};
use serde::Deserialize;

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

pub async fn slew(Json(req): Json<SlewRequest>) -> Result<StatusCode, StatusCode> {
    tracing::info!("Slew requested: RA={}, Dec={}", req.ra, req.dec);
    // TODO: Wire to OLS mount driver when mount bridge is implemented
    Ok(StatusCode::OK)
}

pub async fn slew_target(Json(req): Json<SlewTargetRequest>) -> Result<StatusCode, StatusCode> {
    tracing::info!("Slew to target: {}", req.target);
    Ok(StatusCode::OK)
}

pub async fn sync(Json(req): Json<SlewRequest>) -> Result<StatusCode, StatusCode> {
    tracing::info!("Mount sync: RA={}, Dec={}", req.ra, req.dec);
    Ok(StatusCode::OK)
}

pub async fn park() -> Result<StatusCode, StatusCode> {
    tracing::info!("Park requested");
    Ok(StatusCode::OK)
}

pub async fn unpark() -> Result<StatusCode, StatusCode> {
    tracing::info!("Unpark requested");
    Ok(StatusCode::OK)
}

pub async fn abort() -> Result<StatusCode, StatusCode> {
    tracing::info!("Mount abort requested");
    Ok(StatusCode::OK)
}

pub async fn set_tracking(Json(req): Json<TrackingRequest>) -> Result<StatusCode, StatusCode> {
    tracing::info!("Set tracking: {}", req.enabled);
    Ok(StatusCode::OK)
}

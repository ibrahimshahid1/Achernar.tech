//! Focus API endpoints — Section 7.5.2–7.5.4.
//! All /api/v1/focus/* endpoints.

use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use crate::state::AppState;

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
    State(_state): State<AppState>,
    Json(_req): Json<StartFocusRunRequest>,
) -> Result<Json<FocusRunResponse>, StatusCode> {
    todo!("Implement start focus run endpoint — Section 7.5.2")
}

/// POST /api/v1/focus/abort — abort current focus run
pub async fn abort_focus_run(
    State(_state): State<AppState>,
) -> Result<StatusCode, StatusCode> {
    todo!("Implement abort focus run endpoint")
}

/// GET /api/v1/focus/status — get current focus state
pub async fn get_focus_status(
    State(_state): State<AppState>,
) -> Result<Json<FocusRunResponse>, StatusCode> {
    todo!("Implement focus status endpoint — Section 7.5.3")
}

/// GET /api/v1/focus/history — list past focus runs
pub async fn get_focus_history(
    State(_state): State<AppState>,
) -> Result<Json<Vec<FocusRunResponse>>, StatusCode> {
    todo!("Implement focus history endpoint — Section 7.5.4")
}

/// GET /api/v1/focus/drift — get current drift prediction
pub async fn get_drift_prediction(
    State(_state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    todo!("Implement drift prediction endpoint")
}

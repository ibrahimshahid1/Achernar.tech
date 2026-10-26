//! Focuser routes (spec §2.12). Every move goes through the command gate,
//! so `FocusSafetyGuard`'s travel bounds apply to manual moves too and a
//! move is refused while an exposure integrates (A06, D9). Halt runs on a
//! degraded or lost focuser until it is retired (D19). Each route's
//! `*_value` function is shared with the relay arm (D12).

use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::api_error::ApiError;
use super::equipment::{self as eq, accepted, source};
use crate::equipment::commands::{FocuserCmd, GatedCommand, GatedOp};
use crate::equipment::types::{AutonomyMode, DeviceKind, OpOutcome, Priority};
use crate::state::AppState;

const OPERATOR_PRIORITY: Priority = Priority::Exposure;

#[derive(Debug, Deserialize)]
pub struct MoveRequest {
    pub steps: i32,
    #[serde(default, rename = "override")]
    pub overriding: bool,
}

#[derive(Debug, Deserialize)]
pub struct MoveToRequest {
    pub position: i32,
    #[serde(default, rename = "override")]
    pub overriding: bool,
}

#[derive(Debug, Serialize)]
pub struct PositionResponse {
    pub position: i32,
}

#[derive(Debug, Deserialize)]
pub struct TempCompRequest {
    pub enabled: bool,
}

#[derive(Debug, Deserialize)]
pub struct PositionRequest {
    pub position: i32,
}

async fn focuser_command(
    state: &AppState,
    cmd: FocuserCmd,
    relay: bool,
    overriding: bool,
    status: &str,
    extra: Value,
) -> Result<Value, ApiError> {
    let handle = eq::focuser_handle(state)?;
    let submitted = eq::submit(
        state,
        &handle,
        GatedOp::Focuser(cmd),
        source(relay, overriding),
        OPERATOR_PRIORITY,
    )
    .await?;
    Ok(accepted(status, &submitted, extra))
}

pub async fn move_relative_value(
    state: &AppState,
    req: MoveRequest,
    relay: bool,
) -> Result<Value, ApiError> {
    tracing::info!("Focuser move: {} steps", req.steps);
    focuser_command(
        state,
        FocuserCmd::MoveBy(req.steps),
        relay,
        req.overriding,
        "moving",
        json!({"steps": req.steps}),
    )
    .await
}

pub async fn move_relative(
    State(state): State<AppState>,
    Json(req): Json<MoveRequest>,
) -> Result<Json<Value>, ApiError> {
    move_relative_value(&state, req, false).await.map(Json)
}

pub async fn move_absolute_value(
    state: &AppState,
    req: MoveToRequest,
    relay: bool,
) -> Result<Value, ApiError> {
    tracing::info!("Focuser move to: {}", req.position);
    focuser_command(
        state,
        FocuserCmd::MoveTo(req.position),
        relay,
        req.overriding,
        "moving",
        json!({"position": req.position}),
    )
    .await
}

pub async fn move_absolute(
    State(state): State<AppState>,
    Json(req): Json<MoveToRequest>,
) -> Result<Json<Value>, ApiError> {
    move_absolute_value(&state, req, false).await.map(Json)
}

/// The logical position from the latest snapshot (never a driver call).
/// 404 while the position is unknown (an emulated focuser with no origin):
/// step 0 is a real position, so nothing is invented.
pub fn position_value(state: &AppState) -> Result<i32, ApiError> {
    let handle = eq::focuser_handle(state)?;
    handle.snapshot().logical_position.ok_or_else(|| {
        ApiError::new(
            StatusCode::NOT_FOUND,
            "focuser_position_unknown",
            "the focuser has not reported a position (or its origin is unknown)",
        )
    })
}

pub async fn get_position(
    State(state): State<AppState>,
) -> Result<Json<PositionResponse>, ApiError> {
    position_value(&state).map(|position| Json(PositionResponse { position }))
}

/// `/focuser/abort`: Halt through the out-of-band path and a cancel of the
/// running move (D19).
pub async fn abort_value(state: &AppState, relay: bool) -> Result<Value, ApiError> {
    tracing::info!("Focuser abort requested");
    let handle = eq::focuser_abort_handle(state)?;
    Ok(eq::abort(state, &handle, source(relay, false)).await)
}

pub async fn abort(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    abort_value(&state, false).await.map(Json)
}

pub async fn set_tempcomp_value(
    state: &AppState,
    req: TempCompRequest,
    relay: bool,
) -> Result<Value, ApiError> {
    focuser_command(
        state,
        FocuserCmd::SetTempComp(req.enabled),
        relay,
        false,
        "tempcomp_updated",
        json!({"enabled": req.enabled}),
    )
    .await
}

pub async fn set_tempcomp(
    State(state): State<AppState>,
    Json(req): Json<TempCompRequest>,
) -> Result<Json<Value>, ApiError> {
    set_tempcomp_value(&state, req, false).await.map(Json)
}

/// `POST focuser/sync {position}`: position sync where the device has it
/// (INDI `FOCUS_SYNC`, Tier 2); 501 otherwise.
pub async fn sync_value(
    state: &AppState,
    req: PositionRequest,
    relay: bool,
) -> Result<Value, ApiError> {
    focuser_command(
        state,
        FocuserCmd::Sync(req.position),
        relay,
        false,
        "synced",
        json!({"position": req.position}),
    )
    .await
}

pub async fn sync(
    State(state): State<AppState>,
    Json(req): Json<PositionRequest>,
) -> Result<Json<Value>, ApiError> {
    sync_value(&state, req, false).await.map(Json)
}

/// `POST focuser/set-position {position}`: the operator declares the
/// origin of an emulated relative focuser (F-b). Profile only, no hardware
/// command; the gate checks the travel bounds and audits it.
pub fn set_position_value(
    state: &AppState,
    req: PositionRequest,
    relay: bool,
) -> Result<Value, ApiError> {
    eq::focuser_handle(state)?;
    let session = state
        .equipment_ctx
        .focuser_session()
        .ok_or_else(|| eq::not_connected(DeviceKind::Focuser))?;
    let cmd = GatedCommand::new(
        GatedOp::FocuserSetPosition {
            position: req.position,
        },
        source(relay, false),
        AutonomyMode::Manual,
        OPERATOR_PRIORITY,
    );
    let gate = state.equipment.gate();
    let _permit = gate
        .authorize(&cmd)
        .map_err(|err| eq::submit_error(DeviceKind::Focuser, &err.into()))?;
    match session.set_position(req.position) {
        Ok(()) => {
            gate.record_executed(&cmd, Vec::new());
            gate.record_outcome(&cmd, &OpOutcome::Succeeded);
            Ok(json!({
                "status": "position_set",
                "command_id": cmd.command_id,
                "position": req.position,
                "position_source": session.position_source(),
            }))
        }
        Err(err) => {
            gate.record_outcome(&cmd, &OpOutcome::Failed(err.clone()));
            Err(eq::device_error(DeviceKind::Focuser, &err))
        }
    }
}

pub async fn set_position(
    State(state): State<AppState>,
    Json(req): Json<PositionRequest>,
) -> Result<Json<Value>, ApiError> {
    set_position_value(&state, req, false).map(Json)
}

/// `GET focuser` (§2.12): capabilities, the backlash model and owner, and
/// the position source.
pub fn descriptor_value(state: &AppState) -> Result<Value, ApiError> {
    let handle = eq::focuser_handle(state)?;
    let identity = handle.identity();
    let snapshot = handle.snapshot();
    let session = state.equipment_ctx.focuser_session();
    Ok(json!({
        "identity": crate::equipment::spine::identity_json(&identity),
        "capabilities": snapshot.capabilities,
        "backlash_model": snapshot.backlash_model,
        "backlash_owner": snapshot.backlash_owner,
        "position_source": snapshot.position_source,
        "position": snapshot.logical_position,
        "temperature_role": snapshot.temperature_role,
        "travel_bounds": session.as_ref().and_then(|s| s.settings().travel_bounds(s.capabilities())),
        "qualification": identity.qualification,
        "health": handle.health(),
    }))
}

pub async fn descriptor(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    descriptor_value(&state).map(Json)
}

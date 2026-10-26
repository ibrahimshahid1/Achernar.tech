//! Filter-wheel routes (spec §2.12, §3.4). Owner: WP1B-1.
//!
//! A change runs the v6 §8.2.1 handshake (W-c) behind the gate: the
//! request is authorised once for the wheel, focuser and guider domains,
//! the route returns on accept with the command id, and the handshake's
//! events, resume blocks and `filter_switch_events` row record the rest.
//! `abort` cancels the handshake (the wheel has no halt). `import-seed`
//! copies the `[filter_wheel]` TOML seeds into the wheel's identity profile
//! (D17), audited.

use axum::{extract::State, http::StatusCode, Json};
use serde::Deserialize;
use serde_json::{json, Value};

use super::api_error::ApiError;
use super::equipment::{self as eq, source};
use crate::equipment::commands::{FilterTarget, GatedCommand, GatedOp};
use crate::equipment::filter::change::{HandshakeRequest, Peers};
use crate::equipment::handle::GuiderHandle;
use crate::equipment::types::{AutonomyMode, DeviceKind, OpOutcome, Priority};
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct ChangeRequest {
    /// 0-based slot.
    #[serde(default)]
    pub slot: Option<u32>,
    /// Filter name (the reconciled name, case-insensitive).
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default, rename = "override")]
    pub overriding: bool,
}

fn changer(
    state: &AppState,
) -> Result<std::sync::Arc<crate::equipment::filter::change::FilterChanger>, ApiError> {
    eq::wheel_handle(state)?;
    state
        .equipment_ctx
        .filter_changer()
        .ok_or_else(|| eq::not_connected(DeviceKind::FilterWheel))
}

/// `GET filter-wheel`.
pub fn descriptor_value(state: &AppState) -> Result<Value, ApiError> {
    let handle = eq::wheel_handle(state)?;
    let identity = handle.identity();
    let snapshot = handle.snapshot();
    let changer = state.equipment_ctx.filter_changer();
    Ok(json!({
        "identity": crate::equipment::spine::identity_json(&identity),
        "slots": snapshot.slots,
        "position": snapshot.position.value,
        "moving": snapshot.moving,
        "verified": snapshot.verified,
        "verified_slot": snapshot.verified_slot,
        "settings": changer.as_ref().map(|c| json!({
            "settle_ms": c.settings().settle_ms,
            "use_focus_offsets": c.settings().use_focus_offsets,
            "pause_guiding_on_offset": c.settings().pause_guiding_on_offset,
            "max_automatic_switches_per_hour": c.settings().max_automatic_switches_per_hour,
        })),
        "qualification": identity.qualification,
        "health": handle.health(),
    }))
}

pub async fn descriptor(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    descriptor_value(&state).map(Json)
}

/// `POST filter-wheel/change {slot | name}`: authorise the handshake, start
/// it, and return on accept.
pub fn change_value(state: &AppState, req: ChangeRequest, relay: bool) -> Result<Value, ApiError> {
    let target = match (req.slot, req.name.as_deref().map(str::trim)) {
        (Some(slot), _) => FilterTarget::Slot(slot),
        (None, Some(name)) if !name.is_empty() => FilterTarget::Name(name.to_string()),
        _ => {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "filter_target_required",
                "give slot (0-based) or name",
            ))
        }
    };
    let changer = changer(state)?;
    let cmd = GatedCommand::new(
        GatedOp::FilterHandshake {
            target: target.clone(),
        },
        source(relay, req.overriding),
        AutonomyMode::Manual,
        Priority::Exposure,
    );
    let permit = state
        .equipment
        .gate()
        .authorize(&cmd)
        .map_err(|err| eq::submit_error(DeviceKind::FilterWheel, &err.into()))?;
    let peers = Peers {
        focuser: eq::focuser_handle(state).ok(),
        guider: state
            .equipment
            .handle::<GuiderHandle>()
            .filter(|g| !g.is_retired()),
        camera: state.equipment.camera(),
    };
    let gate = std::sync::Arc::clone(state.equipment.gate());
    let audited = cmd.clone();
    tokio::spawn(async move {
        let report = changer
            .change(peers, HandshakeRequest::new(target), permit)
            .await;
        gate.record_outcome(&audited, &report.outcome);
    });
    gate_executed(state, &cmd);
    Ok(json!({
        "status": "changing",
        "command_id": cmd.command_id,
    }))
}

fn gate_executed(state: &AppState, cmd: &GatedCommand) {
    state
        .equipment
        .gate()
        .record_executed(cmd, vec!["handshake=started".into()]);
}

pub async fn change(
    State(state): State<AppState>,
    Json(req): Json<ChangeRequest>,
) -> Result<Json<Value>, ApiError> {
    change_value(&state, req, false).map(Json)
}

/// `POST filter-wheel/abort`: cancel the handshake (D19: never refused;
/// the wheel's position becomes unknown, which the handshake reports with a
/// resume block).
pub fn abort_value(state: &AppState, relay: bool) -> Result<Value, ApiError> {
    let changer = state
        .equipment_ctx
        .filter_changer()
        .ok_or_else(|| eq::not_connected(DeviceKind::FilterWheel))?;
    let cmd = GatedCommand::new(
        GatedOp::Abort {
            kind: DeviceKind::FilterWheel,
        },
        source(relay, false),
        AutonomyMode::Manual,
        Priority::Safety,
    );
    let gate = state.equipment.gate();
    let _ = gate.authorize(&cmd);
    gate.abort_epochs().bump(DeviceKind::FilterWheel);
    let by = changer.abort();
    gate.record_executed(&cmd, vec![format!("cancelled_by={by}")]);
    gate.record_outcome(&cmd, &OpOutcome::Succeeded);
    Ok(json!({"status": "aborting", "command_id": cmd.command_id}))
}

pub async fn abort(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    abort_value(&state, false).map(Json)
}

/// `POST filter-wheel/import-seed` (D17): the TOML seeds become the
/// connected wheel's profile, audited.
pub fn import_seed_value(state: &AppState, relay: bool) -> Result<Value, ApiError> {
    let changer = changer(state)?;
    let cmd = GatedCommand::new(
        GatedOp::FilterImportSeed,
        source(relay, false),
        AutonomyMode::Manual,
        Priority::Exposure,
    );
    let gate = state.equipment.gate();
    let permit = gate
        .authorize(&cmd)
        .map_err(|err| eq::submit_error(DeviceKind::FilterWheel, &err.into()))?;
    match changer.import_seed(&state.equipment_ctx.configs.filter_wheel, &permit) {
        Ok(result) => {
            gate.record_executed(&cmd, result.warnings.clone());
            gate.record_outcome(&cmd, &OpOutcome::Succeeded);
            Ok(json!({
                "status": "imported",
                "command_id": cmd.command_id,
                "applied_slots": result.applied_slots,
                "warnings": result.warnings,
                "profile": result.profile,
            }))
        }
        Err(err) => {
            gate.record_outcome(&cmd, &OpOutcome::Failed(err.clone()));
            Err(eq::device_error(DeviceKind::FilterWheel, &err))
        }
    }
}

pub async fn import_seed(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    import_seed_value(&state, false).map(Json)
}

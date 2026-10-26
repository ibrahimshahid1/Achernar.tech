//! Guider routes (spec §2.12, §3.5). Owner: WP1B-1.
//!
//! PHD2 owns the guide camera and the guide loop (O6); these routes are thin
//! commands to its event server through the gate: start, stop, pause,
//! resume and dither (dither only between exposures and while guiding, the
//! WP1A-7 check). Stop is the guider's abort and is never refused (D19).

use axum::{extract::State, Json};
use serde::Deserialize;
use serde_json::{json, Value};

use super::api_error::ApiError;
use super::equipment::{self as eq, accepted, source};
use crate::equipment::commands::{GatedOp, GuiderCmd, PixelUnit, Settle};
use crate::equipment::types::Priority;
use crate::state::AppState;

#[derive(Debug, Default, Deserialize)]
pub struct SettleBody {
    #[serde(default)]
    pub pixels: Option<f64>,
    #[serde(default)]
    pub time_s: Option<f64>,
    #[serde(default)]
    pub timeout_s: Option<f64>,
}

impl SettleBody {
    fn resolve(&self, default: Settle) -> Settle {
        Settle {
            pixels: self.pixels.unwrap_or(default.pixels),
            time_s: self.time_s.unwrap_or(default.time_s),
            timeout_s: self.timeout_s.unwrap_or(default.timeout_s),
        }
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct StartRequest {
    #[serde(default)]
    pub settle: Option<SettleBody>,
    #[serde(default)]
    pub recalibrate: bool,
}

#[derive(Debug, Default, Deserialize)]
pub struct PauseRequest {
    /// Also pause looping exposures (PHD2 `set_paused [true, "full"]`).
    #[serde(default)]
    pub full: bool,
}

#[derive(Debug, Default, Deserialize)]
pub struct DitherRequest {
    /// Guide-camera pixels (`[guider] dither_amount_px` by default).
    #[serde(default)]
    pub amount_px: Option<f64>,
    #[serde(default)]
    pub ra_only: Option<bool>,
    #[serde(default)]
    pub settle: Option<SettleBody>,
}

fn default_settle(state: &AppState) -> Settle {
    state.equipment_ctx.configs.guider.settle()
}

async fn guider_command(
    state: &AppState,
    cmd: GuiderCmd,
    relay: bool,
    status: &str,
    extra: Value,
) -> Result<Value, ApiError> {
    let handle = eq::guider_handle(state)?;
    let submitted = eq::submit(
        state,
        &handle,
        GatedOp::Guider(cmd),
        source(relay, false),
        Priority::Exposure,
    )
    .await?;
    Ok(accepted(status, &submitted, extra))
}

pub fn descriptor_value(state: &AppState) -> Result<Value, ApiError> {
    let handle = eq::guider_handle(state)?;
    let identity = handle.identity();
    let snapshot = handle.snapshot();
    Ok(json!({
        "identity": crate::equipment::spine::identity_json(&identity),
        "guiding": crate::equipment::status::guiding_block(
            &snapshot,
            &identity,
            &handle.health(),
            state.equipment.clock().now_mono_ns(),
        ),
        "algo_params": snapshot.algo_params,
        "aggressiveness_param_ra": snapshot.aggressiveness_param_ra,
        "aggressiveness_param_dec": snapshot.aggressiveness_param_dec,
        "actuation_disabled_reason": snapshot.actuation_disabled_reason,
        "dec_guide_mode": snapshot.dec_guide_mode,
        "qualification": identity.qualification,
        "health": handle.health(),
    }))
}

pub async fn descriptor(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    descriptor_value(&state).map(Json)
}

pub async fn start_value(
    state: &AppState,
    req: StartRequest,
    relay: bool,
) -> Result<Value, ApiError> {
    let settle = req
        .settle
        .unwrap_or_default()
        .resolve(default_settle(state));
    guider_command(
        state,
        GuiderCmd::Start {
            settle,
            recalibrate: req.recalibrate,
        },
        relay,
        "starting",
        json!({"settle": settle, "recalibrate": req.recalibrate}),
    )
    .await
}

pub async fn start(
    State(state): State<AppState>,
    body: Option<Json<StartRequest>>,
) -> Result<Json<Value>, ApiError> {
    let req = body.map(|Json(b)| b).unwrap_or_default();
    start_value(&state, req, false).await.map(Json)
}

/// Stop guiding (`stop_capture`). The guider's abort (D19): it runs on a
/// degraded or lost guider until it is retired.
pub async fn stop_value(state: &AppState, relay: bool) -> Result<Value, ApiError> {
    let handle = eq::guider_abort_handle(state)?;
    Ok(eq::abort(state, &handle, source(relay, false)).await)
}

pub async fn stop(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    stop_value(&state, false).await.map(Json)
}

pub async fn pause_value(
    state: &AppState,
    req: PauseRequest,
    relay: bool,
) -> Result<Value, ApiError> {
    guider_command(
        state,
        GuiderCmd::Pause { full: req.full },
        relay,
        "pausing",
        json!({"full": req.full}),
    )
    .await
}

pub async fn pause(
    State(state): State<AppState>,
    body: Option<Json<PauseRequest>>,
) -> Result<Json<Value>, ApiError> {
    let req = body.map(|Json(b)| b).unwrap_or_default();
    pause_value(&state, req, false).await.map(Json)
}

pub async fn resume_value(state: &AppState, relay: bool) -> Result<Value, ApiError> {
    guider_command(state, GuiderCmd::Resume, relay, "resuming", json!({})).await
}

pub async fn resume(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    resume_value(&state, false).await.map(Json)
}

pub async fn dither_value(
    state: &AppState,
    req: DitherRequest,
    relay: bool,
) -> Result<Value, ApiError> {
    let guider = &state.equipment_ctx.configs.guider;
    let amount = req.amount_px.unwrap_or(guider.dither_amount_px);
    let ra_only = req.ra_only.unwrap_or(guider.dither_ra_only);
    let settle = req
        .settle
        .unwrap_or_default()
        .resolve(default_settle(state));
    guider_command(
        state,
        GuiderCmd::Dither {
            amount,
            unit: PixelUnit::GuidePx,
            ra_only,
            settle,
        },
        relay,
        "dithering",
        json!({"amount": amount, "unit": PixelUnit::GuidePx, "ra_only": ra_only, "settle": settle}),
    )
    .await
}

pub async fn dither(
    State(state): State<AppState>,
    body: Option<Json<DitherRequest>>,
) -> Result<Json<Value>, ApiError> {
    let req = body.map(|Json(b)| b).unwrap_or_default();
    dither_value(&state, req, false).await.map(Json)
}

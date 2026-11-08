//! Mount routes (spec §2.12). Every motion goes through the command gate
//! (D9): ownership, arbiter, the mount limits (H13), the D20 guider stop
//! before a slew or park, and the audit. Routes return on accept with
//! `{status, command_id, ...}`; completion arrives as events and status.
//! Abort is never gated on a capability flag (O10) and runs on a degraded
//! or lost mount until it is retired (D19). Each route's `*_value`
//! function is shared with the relay arm, so both transports return the
//! same body and the same error codes (D12).

use axum::{extract::State, http::StatusCode, Json};
use serde::Deserialize;
use serde_json::{json, Value};

use super::api_error::ApiError;
use super::equipment::{self as eq, accepted, source};
use crate::equipment::commands::{GatedCommand, GatedOp, MountCmd, SiteSyncDirection};
use crate::equipment::types::{
    AutonomyMode, DeviceKind, DriveRate, EqCoord, EqFrame, GuideDirection, Priority, Site,
};
use crate::state::AppState;

/// Operator commands acquire a free domain at rank 5 (D9); the priority
/// only matters against automatic holders.
const OPERATOR_PRIORITY: Priority = Priority::Exposure;

fn bad_request(code: &'static str, message: impl Into<String>) -> ApiError {
    ApiError::new(StatusCode::BAD_REQUEST, code, message)
}

/// `frame`: `"mount"` (the default, legacy semantics: the coordinates are
/// in the mount's own frame), `"j2000"` or `"jnow"`.
pub fn parse_frame(frame: Option<&str>) -> Result<EqFrame, ApiError> {
    match frame.map(|f| f.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") | Some("mount") | Some("native") => Ok(EqFrame::Unknown),
        Some("j2000") => Ok(EqFrame::J2000),
        Some("jnow") | Some("jnow_apparent") => Ok(EqFrame::JNowApparent),
        Some("jnow_mean") => Ok(EqFrame::JNowMean),
        Some(other) => Err(bad_request(
            "invalid_frame",
            format!("frame must be mount, j2000 or jnow, not {other:?}"),
        )),
    }
}

#[derive(Debug, Deserialize)]
pub struct SlewRequest {
    pub ra: f64,
    pub dec: f64,
    #[serde(default)]
    pub frame: Option<String>,
    #[serde(default, rename = "override")]
    pub overriding: bool,
}

#[derive(Debug, Deserialize)]
pub struct SlewTargetRequest {
    pub target: String,
}

#[derive(Debug, Deserialize)]
pub struct TrackingRequest {
    pub enabled: bool,
    #[serde(default, rename = "override")]
    pub overriding: bool,
}

#[derive(Debug, Default, Deserialize)]
pub struct OverrideRequest {
    #[serde(default, rename = "override")]
    pub overriding: bool,
}

#[derive(Debug, Deserialize)]
pub struct MoveAxisRequest {
    pub axis: u8,
    pub rate_deg_s: f64,
}

#[derive(Debug, Deserialize)]
pub struct PulseGuideRequest {
    pub direction: GuideDirection,
    pub duration_ms: u32,
}

#[derive(Debug, Deserialize)]
pub struct TrackingRateRequest {
    pub rate: DriveRate,
}

#[derive(Debug, Deserialize)]
pub struct GuideRatesRequest {
    pub ra_arcsec_s: f64,
    pub dec_arcsec_s: f64,
}

#[derive(Debug, Deserialize)]
pub struct SiteSyncRequest {
    pub direction: SiteSyncDirection,
}

#[derive(Debug, Deserialize)]
pub struct CenterRequest {
    pub ra: f64,
    pub dec: f64,
    #[serde(default)]
    pub frame: Option<String>,
    /// Overrides of `[mount.center]` for this run; a non-positive or
    /// non-finite value is ignored.
    #[serde(default)]
    pub tolerance_arcmin: Option<f64>,
    #[serde(default)]
    pub max_attempts: Option<u32>,
    /// The exposure, gain and binning of each center frame.
    #[serde(default)]
    pub exposure_s: Option<f64>,
    #[serde(default)]
    pub gain: Option<u32>,
    #[serde(default)]
    pub binning: Option<u32>,
}

impl CenterRequest {
    fn overrides(&self) -> crate::mount::center::CenterOverrides {
        crate::mount::center::CenterOverrides {
            tolerance_arcmin: self.tolerance_arcmin,
            max_attempts: self.max_attempts,
            exposure_s: self.exposure_s,
            gain: self.gain,
            binning: self.binning,
        }
    }
}

/// Gate, send and audit one mount command.
async fn mount_command(
    state: &AppState,
    cmd: MountCmd,
    relay: bool,
    overriding: bool,
    status: &str,
    extra: Value,
) -> Result<Value, ApiError> {
    let handle = eq::mount_handle(state)?;
    let submitted = eq::submit(
        state,
        &handle,
        GatedOp::Mount(cmd),
        source(relay, overriding),
        OPERATOR_PRIORITY,
    )
    .await?;
    Ok(accepted(status, &submitted, extra))
}

// -- slew, sync, tracking, abort (existing paths; JSON bodies now) -----------

pub async fn slew_value(
    state: &AppState,
    req: SlewRequest,
    relay: bool,
) -> Result<Value, ApiError> {
    tracing::info!("Slew requested: RA={}, Dec={}", req.ra, req.dec);
    let frame = parse_frame(req.frame.as_deref())?;
    let target = EqCoord::new(req.ra, req.dec, frame);
    let body = mount_command(
        state,
        MountCmd::Slew { target },
        relay,
        req.overriding,
        "slewing",
        json!({"ra": req.ra, "dec": req.dec, "frame": frame.as_str()}),
    )
    .await?;
    // The post-slew plate solve waits for the mount to report arriving.
    crate::mount::auto_solve::slew_commanded(
        state,
        crate::hardware::mount::EqCoord {
            ra: req.ra,
            dec: req.dec,
        },
    );
    Ok(body)
}

pub async fn slew(
    State(state): State<AppState>,
    Json(req): Json<SlewRequest>,
) -> Result<Json<Value>, ApiError> {
    slew_value(&state, req, false).await.map(Json)
}

pub async fn slew_target(Json(req): Json<SlewTargetRequest>) -> Result<StatusCode, StatusCode> {
    tracing::info!("Slew to target: {}", req.target);
    Err(StatusCode::NOT_IMPLEMENTED)
}

pub async fn sync_value(
    state: &AppState,
    req: SlewRequest,
    relay: bool,
) -> Result<Value, ApiError> {
    tracing::info!("Mount sync: RA={}, Dec={}", req.ra, req.dec);
    let frame = parse_frame(req.frame.as_deref())?;
    mount_command(
        state,
        MountCmd::Sync(EqCoord::new(req.ra, req.dec, frame)),
        relay,
        req.overriding,
        "synced",
        json!({"ra": req.ra, "dec": req.dec, "frame": frame.as_str()}),
    )
    .await
}

pub async fn sync(
    State(state): State<AppState>,
    Json(req): Json<SlewRequest>,
) -> Result<Json<Value>, ApiError> {
    sync_value(&state, req, false).await.map(Json)
}

pub async fn set_tracking_value(
    state: &AppState,
    req: TrackingRequest,
    relay: bool,
) -> Result<Value, ApiError> {
    tracing::info!("Set tracking: {}", req.enabled);
    mount_command(
        state,
        MountCmd::SetTracking(req.enabled),
        relay,
        req.overriding,
        "tracking_updated",
        json!({"enabled": req.enabled}),
    )
    .await
}

pub async fn set_tracking(
    State(state): State<AppState>,
    Json(req): Json<TrackingRequest>,
) -> Result<Json<Value>, ApiError> {
    set_tracking_value(&state, req, false).await.map(Json)
}

/// `/mount/abort` (D19, O10): the out-of-band AbortSlew plus MoveAxis 0 on
/// jogging axes, and a cancel of whatever operation runs. Works on a
/// degraded or lost mount; only a retired one gets 412.
pub async fn abort_value(state: &AppState, relay: bool) -> Result<Value, ApiError> {
    tracing::info!("Mount abort requested");
    let handle = eq::mount_abort_handle(state)?;
    Ok(eq::abort(state, &handle, source(relay, false)).await)
}

pub async fn abort(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    abort_value(&state, false).await.map(Json)
}

// -- park, unpark, home ---------------------------------------------------------

pub async fn park_value(
    state: &AppState,
    relay: bool,
    overriding: bool,
) -> Result<Value, ApiError> {
    tracing::info!("Park requested");
    mount_command(
        state,
        MountCmd::Park,
        relay,
        overriding,
        "parking",
        json!({}),
    )
    .await
}

pub async fn park(
    State(state): State<AppState>,
    body: Option<Json<OverrideRequest>>,
) -> Result<Json<Value>, ApiError> {
    let overriding = body.is_some_and(|Json(b)| b.overriding);
    park_value(&state, false, overriding).await.map(Json)
}

pub async fn unpark_value(
    state: &AppState,
    relay: bool,
    overriding: bool,
) -> Result<Value, ApiError> {
    tracing::info!("Unpark requested");
    mount_command(
        state,
        MountCmd::Unpark,
        relay,
        overriding,
        "unparking",
        json!({}),
    )
    .await
}

pub async fn unpark(
    State(state): State<AppState>,
    body: Option<Json<OverrideRequest>>,
) -> Result<Json<Value>, ApiError> {
    let overriding = body.is_some_and(|Json(b)| b.overriding);
    unpark_value(&state, false, overriding).await.map(Json)
}

pub async fn home_value(state: &AppState, relay: bool) -> Result<Value, ApiError> {
    mount_command(state, MountCmd::FindHome, relay, false, "homing", json!({})).await
}

pub async fn home(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    home_value(&state, false).await.map(Json)
}

// -- jog and pulses (direct transport only, D12) -------------------------------

pub async fn move_axis_value(
    state: &AppState,
    req: MoveAxisRequest,
    relay: bool,
) -> Result<Value, ApiError> {
    let handle = eq::mount_handle(state)?;
    let jogging = handle
        .snapshot()
        .active_op
        .as_ref()
        .is_some_and(|op| op.kind == crate::equipment::ops::OpKind::MoveAxis);
    if jogging && req.rate_deg_s == 0.0 {
        // Stopping the operator's own jog: the abort path stops every
        // jogging axis at once and is never refused (D19).
        let mut body = eq::abort(state, &handle, source(relay, false)).await;
        body["axis"] = json!(req.axis);
        body["rate_deg_s"] = json!(0.0);
        return Ok(body);
    }
    let cmd = MountCmd::MoveAxis {
        axis: req.axis,
        rate_deg_s: req.rate_deg_s,
    };
    // A rate change while the operator jogs takes over the running jog
    // (an audited override of the operator's own operation, D9).
    let submitted = eq::submit(
        state,
        &handle,
        GatedOp::Mount(cmd),
        source(relay, jogging),
        OPERATOR_PRIORITY,
    )
    .await?;
    let deadman_ms = state.equipment_ctx.configs.mount.jog_deadman_ms;
    Ok(accepted(
        if req.rate_deg_s == 0.0 {
            "stopped"
        } else {
            "moving"
        },
        &submitted,
        json!({
            "axis": req.axis,
            "rate_deg_s": req.rate_deg_s,
            "renew_within_ms": deadman_ms,
            "was_jogging": jogging,
        }),
    ))
}

pub async fn move_axis(
    State(state): State<AppState>,
    Json(req): Json<MoveAxisRequest>,
) -> Result<Json<Value>, ApiError> {
    move_axis_value(&state, req, false).await.map(Json)
}

pub async fn pulse_guide_value(
    state: &AppState,
    req: PulseGuideRequest,
    relay: bool,
) -> Result<Value, ApiError> {
    mount_command(
        state,
        MountCmd::PulseGuide {
            dir: req.direction,
            ms: req.duration_ms,
        },
        relay,
        false,
        "pulsing",
        json!({"direction": req.direction, "duration_ms": req.duration_ms}),
    )
    .await
}

pub async fn pulse_guide(
    State(state): State<AppState>,
    Json(req): Json<PulseGuideRequest>,
) -> Result<Json<Value>, ApiError> {
    pulse_guide_value(&state, req, false).await.map(Json)
}

// -- rates and site -------------------------------------------------------------

pub async fn tracking_rate_value(
    state: &AppState,
    req: TrackingRateRequest,
    relay: bool,
) -> Result<Value, ApiError> {
    mount_command(
        state,
        MountCmd::SetTrackingRate(req.rate),
        relay,
        false,
        "tracking_rate_updated",
        json!({"rate": req.rate}),
    )
    .await
}

pub async fn tracking_rate(
    State(state): State<AppState>,
    Json(req): Json<TrackingRateRequest>,
) -> Result<Json<Value>, ApiError> {
    tracking_rate_value(&state, req, false).await.map(Json)
}

pub async fn guide_rates_value(
    state: &AppState,
    req: GuideRatesRequest,
    relay: bool,
) -> Result<Value, ApiError> {
    mount_command(
        state,
        MountCmd::SetGuideRates {
            ra_arcsec_s: req.ra_arcsec_s,
            dec_arcsec_s: req.dec_arcsec_s,
        },
        relay,
        false,
        "guide_rates_updated",
        json!({"ra_arcsec_s": req.ra_arcsec_s, "dec_arcsec_s": req.dec_arcsec_s}),
    )
    .await
}

pub async fn guide_rates(
    State(state): State<AppState>,
    Json(req): Json<GuideRatesRequest>,
) -> Result<Json<Value>, ApiError> {
    guide_rates_value(&state, req, false).await.map(Json)
}

/// `POST mount/site-sync {direction}` (M-b): `to_mount` writes `[site]` to
/// the mount through the gate (refused while slewing or parking);
/// `from_mount` adopts the mount's site for the session. Both are audited.
pub async fn site_sync_value(
    state: &AppState,
    req: SiteSyncRequest,
    relay: bool,
) -> Result<Value, ApiError> {
    let handle = eq::mount_handle(state)?;
    let snapshot = handle.snapshot();
    if snapshot.state.slewing.known().copied() == Some(true)
        || snapshot
            .active_op
            .as_ref()
            .is_some_and(|op| op.kind.moves_hardware())
    {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "mount_invalid_operation",
            "site sync is refused while the mount is slewing, parking or moving",
        ));
    }
    match req.direction {
        SiteSyncDirection::ToMount => {
            let site: Site = state
                .equipment_ctx
                .configs
                .site
                .site()
                .ok()
                .flatten()
                .ok_or_else(|| {
                    bad_request(
                        "site_unknown",
                        "[site] has no latitude, longitude and elevation to write",
                    )
                })?;
            mount_command(
                state,
                MountCmd::SetSite(site),
                relay,
                false,
                "site_synced",
                json!({"direction": "to_mount", "site": site}),
            )
            .await
        }
        SiteSyncDirection::FromMount => {
            let site = snapshot.state.site.fresh_value().copied().ok_or_else(|| {
                ApiError::new(
                    StatusCode::CONFLICT,
                    "mount_site_unknown",
                    "the mount has not reported a site",
                )
            })?;
            let cmd = GatedCommand::new(
                GatedOp::SiteSync {
                    direction: SiteSyncDirection::FromMount,
                },
                source(relay, false),
                AutonomyMode::Manual,
                OPERATOR_PRIORITY,
            );
            let gate = state.equipment.gate();
            let _permit = gate
                .authorize(&cmd)
                .map_err(|err| eq::submit_error(DeviceKind::Mount, &err.into()))?;
            if let Some(shared) = state.equipment_ctx.mount_shared() {
                shared.set_session_site(Some(site));
            }
            state.equipment.set_site(Some(site));
            gate.record_executed(&cmd, vec!["session_site=mount".into()]);
            gate.record_outcome(&cmd, &crate::equipment::types::OpOutcome::Succeeded);
            Ok(json!({
                "status": "site_synced",
                "command_id": cmd.command_id,
                "direction": "from_mount",
                "site": site,
            }))
        }
    }
}

pub async fn site_sync(
    State(state): State<AppState>,
    Json(req): Json<SiteSyncRequest>,
) -> Result<Json<Value>, ApiError> {
    site_sync_value(&state, req, false).await.map(Json)
}

/// `POST mount/center` (M-n): authorise the center loop (`D/mount/center.rs`)
/// for the mount and camera domains, start it, and return on accept with
/// `{status: "centering", command_id, ...}`. Completion arrives as the
/// `centered` or `center_failed` event.
pub async fn center_value(
    state: &AppState,
    req: CenterRequest,
    relay: bool,
) -> Result<Value, ApiError> {
    tracing::info!("Center requested: RA={}, Dec={}", req.ra, req.dec);
    let frame = parse_frame(req.frame.as_deref())?;
    crate::mount::center::start(
        state,
        EqCoord::new(req.ra, req.dec, frame),
        req.overrides(),
        source(relay, false),
    )
    .await
}

pub async fn center(
    State(state): State<AppState>,
    Json(req): Json<CenterRequest>,
) -> Result<Json<Value>, ApiError> {
    center_value(&state, req, false).await.map(Json)
}

// -- descriptor -----------------------------------------------------------------

/// `GET mount` (§2.12): identity, capabilities, frame, site, `can_flip`
/// and qualification.
pub fn descriptor_value(state: &AppState) -> Result<Value, ApiError> {
    let handle = eq::mount_handle(state)?;
    let identity = handle.identity();
    let snapshot = handle.snapshot();
    let site = state
        .equipment_ctx
        .mount_shared()
        .and_then(|shared| shared.session_site())
        .or(snapshot.state.site.value);
    Ok(json!({
        "identity": crate::equipment::spine::identity_json(&identity),
        "capabilities": snapshot.capabilities,
        "frame": snapshot.capabilities.native_frame.as_str(),
        "site": site,
        "can_flip": snapshot.can_flip,
        "qualification": identity.qualification,
        "health": handle.health(),
        "jog_deadman_ms": state.equipment_ctx.configs.mount.jog_deadman_ms,
        "max_pulse_ms": state.equipment_ctx.configs.mount.max_pulse_ms,
    }))
}

pub async fn descriptor(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    descriptor_value(&state).map(Json)
}

//! The integration spine's API half (spec §2.6, §2.10, §2.12, D12, D19).
//! Owner: WP1B-1.
//!
//! - **Connect and disconnect** for every non-camera kind: resolve the
//!   driver id, run the connect sequence on a blocking thread, start the
//!   device's worker (`equipment::spine`), then publish the slot, the name
//!   and the registry entry, in the existing order (§2.6 step 9). A
//!   per-device projector task keeps the registry's `/status` block current
//!   from the handle's snapshot and health watches, so status and telemetry
//!   never touch a driver (`health_tests.rs`).
//! - **Commands** go through the gate ([`submit`]): ownership, arbiter,
//!   safety checks, audit, and the D20 guider stop before a slew or park.
//!   Direct and relay routes share one implementation each, so they return
//!   the same JSON body and the same error codes (D12).
//! - **Errors** keep the `ApiError` shape `{error: {code, message}}`; the
//!   code carries the failure class, e.g. `mount_outcome_unknown` (§2.12).
//! - **E-stop, resume blocks and the event journal** (`equipment/*`).

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use super::api_error::ApiError;
use super::ConnectRequest;
use crate::equipment::commands::{GatedCommand, GatedOp};
use crate::equipment::estop::{run_estop, EstopRequest};
use crate::equipment::focuser::FocuserDeps;
use crate::equipment::gate::{GatedDeviceCommand, SubmitError, Submitted};
use crate::equipment::guider::service::{self as phd2, GuiderContext, Phd2Settings};
use crate::equipment::handle::{
    DeviceHandle, FilterWheelHandle, FocuserHandle, GuiderHandle, MountHandle, WeatherHandle,
};
use crate::equipment::hub::AnyHandle;
use crate::equipment::spine::{self, DriverChoice, DriverRefusal};
use crate::equipment::status;
use crate::equipment::traits::EquipmentReader;
use crate::equipment::types::{
    kind_name, AutonomyMode, CommandSource, DeviceError, DeviceIdentity, DeviceKind, ErrorClass,
    HealthState, LinkState, Priority,
};
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Error codes
// ---------------------------------------------------------------------------

/// Error codes are `&'static str` in `ApiError`. The `{kind}_{class}` codes
/// form a small closed set (kinds x classes), so each distinct one is
/// interned once.
pub fn static_code(code: String) -> &'static str {
    static CODES: Mutex<BTreeSet<&'static str>> = Mutex::new(BTreeSet::new());
    let mut codes = CODES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(existing) = codes.get(code.as_str()) {
        return existing;
    }
    let leaked: &'static str = Box::leak(code.into_boxed_str());
    codes.insert(leaked);
    leaked
}

fn status_of(code: u16) -> StatusCode {
    StatusCode::from_u16(code).unwrap_or(StatusCode::BAD_GATEWAY)
}

/// A device error as an API error: the class's HTTP status (§2.12) and the
/// code `{kind}_{class}`.
pub fn device_error(kind: DeviceKind, err: &DeviceError) -> ApiError {
    ApiError::new(
        status_of(err.class.http_status()),
        static_code(err.api_code(kind)),
        err.message.clone(),
    )
}

/// A gate or coordination refusal, or a device refusal, as an API error.
pub fn submit_error(kind: DeviceKind, err: &SubmitError) -> ApiError {
    ApiError::new(
        status_of(err.http_status()),
        static_code(err.code(kind)),
        err.to_string(),
    )
}

/// HTTP 412 for a device that is not connected (kept, §2.12).
pub fn not_connected(kind: DeviceKind) -> ApiError {
    ApiError::new(
        StatusCode::PRECONDITION_FAILED,
        static_code(format!("{}_not_connected", kind_name(kind))),
        format!("No {} is connected", kind_name(kind).replace('_', " ")),
    )
}

fn refusal(refusal: DriverRefusal) -> ApiError {
    ApiError::new(status_of(refusal.status), refusal.code, refusal.message)
}

/// The relay's error string for an API error: the code and the message, so
/// both transports carry the same error meaning (D12).
pub fn relay_error(err: ApiError) -> String {
    format!("{}: {}", err.code, err.message)
}

// ---------------------------------------------------------------------------
// Kinds and sources
// ---------------------------------------------------------------------------

/// `equipment_type` names (§2.12). The serde names, plus the spellings the
/// frontend and Alpaca use.
pub fn kind_from_str(name: &str) -> Option<DeviceKind> {
    Some(match name.trim().to_ascii_lowercase().as_str() {
        "camera" => DeviceKind::Camera,
        "mount" | "telescope" => DeviceKind::Mount,
        "focuser" => DeviceKind::Focuser,
        "filter_wheel" | "filterwheel" => DeviceKind::FilterWheel,
        "guider" => DeviceKind::Guider,
        "rotator" => DeviceKind::Rotator,
        "cover_calibrator" | "covercalibrator" | "cover" => DeviceKind::CoverCalibrator,
        "switch" => DeviceKind::Switch,
        "weather" | "observingconditions" => DeviceKind::Weather,
        "safety_monitor" | "safetymonitor" | "safety" => DeviceKind::SafetyMonitor,
        "dome" => DeviceKind::Dome,
        _ => return None,
    })
}

/// Who issued a request: the operator on the direct API, the relay
/// otherwise (D12). Override is explicit and audited (D9).
pub fn source(relay: bool, overriding: bool) -> CommandSource {
    if relay {
        CommandSource::Relay { overriding }
    } else {
        CommandSource::Operator { overriding }
    }
}

// ---------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------

/// The registry's current connection for `kind` is `handle`: a handle the
/// registry has retired (lost, or replaced) takes no new commands (412).
fn registered<C, S>(state: &AppState, handle: &DeviceHandle<C, S>) -> bool {
    state
        .devices
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .connection_id(handle.kind())
        .as_deref()
        == Some(handle.connection_id().to_string().as_str())
}

fn live<C, S>(
    state: &AppState,
    kind: DeviceKind,
    handle: Option<DeviceHandle<C, S>>,
) -> Result<DeviceHandle<C, S>, ApiError> {
    match handle {
        Some(handle) if !handle.is_retired() && registered(state, &handle) => Ok(handle),
        _ => Err(not_connected(kind)),
    }
}

/// For aborts (D19): any handle that is not retired, whatever the registry
/// says (a lost device still gets its abort path).
fn abortable<C, S>(
    kind: DeviceKind,
    handle: Option<DeviceHandle<C, S>>,
) -> Result<DeviceHandle<C, S>, ApiError> {
    match handle {
        Some(handle) if !handle.is_retired() => Ok(handle),
        _ => Err(not_connected(kind)),
    }
}

pub fn mount_slot(state: &AppState) -> Option<MountHandle> {
    state
        .mount
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

pub fn focuser_slot(state: &AppState) -> Option<FocuserHandle> {
    state
        .focuser
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

pub fn wheel_slot(state: &AppState) -> Option<FilterWheelHandle> {
    state
        .filter_wheel
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

pub fn mount_handle(state: &AppState) -> Result<MountHandle, ApiError> {
    live(state, DeviceKind::Mount, mount_slot(state))
}

pub fn mount_abort_handle(state: &AppState) -> Result<MountHandle, ApiError> {
    abortable(DeviceKind::Mount, mount_slot(state))
}

pub fn focuser_handle(state: &AppState) -> Result<FocuserHandle, ApiError> {
    live(state, DeviceKind::Focuser, focuser_slot(state))
}

pub fn focuser_abort_handle(state: &AppState) -> Result<FocuserHandle, ApiError> {
    abortable(DeviceKind::Focuser, focuser_slot(state))
}

pub fn wheel_handle(state: &AppState) -> Result<FilterWheelHandle, ApiError> {
    live(state, DeviceKind::FilterWheel, wheel_slot(state))
}

pub fn guider_handle(state: &AppState) -> Result<GuiderHandle, ApiError> {
    live(
        state,
        DeviceKind::Guider,
        state.equipment.handle::<GuiderHandle>(),
    )
}

pub fn guider_abort_handle(state: &AppState) -> Result<GuiderHandle, ApiError> {
    abortable(DeviceKind::Guider, state.equipment.handle::<GuiderHandle>())
}

pub fn weather_handle(state: &AppState) -> Result<WeatherHandle, ApiError> {
    live(
        state,
        DeviceKind::Weather,
        state.equipment.handle::<WeatherHandle>(),
    )
}

// ---------------------------------------------------------------------------
// Commands through the gate
// ---------------------------------------------------------------------------

/// Authorise, coordinate, send and audit one device command (§2.9). Returns
/// on accept (HTTP routes do not wait for completion).
pub async fn submit<C, S>(
    state: &AppState,
    handle: &DeviceHandle<C, S>,
    op: GatedOp,
    source: CommandSource,
    priority: Priority,
) -> Result<Submitted, ApiError>
where
    C: GatedDeviceCommand,
    S: Send + Sync + 'static,
{
    let cmd = GatedCommand::new(op, source, AutonomyMode::Manual, priority);
    state
        .equipment
        .gate()
        .submit(handle, cmd)
        .await
        .map_err(|err| submit_error(handle.kind(), &err))
}

/// The accepted-command body shared by the direct and relay routes:
/// `{status, command_id, ...}` (§2.12).
pub fn accepted(status: &str, submitted: &Submitted, extra: Value) -> Value {
    let mut body = json!({
        "status": status,
        "command_id": submitted.ticket.command_id,
    });
    if let Some(pre) = &submitted.pre_motion {
        body["was_guiding"] = json!(pre.was_guiding);
    }
    if let (Value::Object(body), Value::Object(extra)) = (&mut body, extra) {
        body.extend(extra);
    }
    body
}

/// An audited abort (D19): runs on a ready, degraded or lost handle; waits
/// up to 2 s for the abort path's own result.
pub async fn abort<C, S>(
    state: &AppState,
    handle: &DeviceHandle<C, S>,
    source: CommandSource,
) -> Value {
    let ticket = state.equipment.gate().abort(handle, source);
    let command_id = ticket.command_id;
    let (cancel_sent, path_ran) = (ticket.cancel_sent, ticket.abort_path_ran);
    let result = tokio::time::timeout(Duration::from_secs(2), ticket.outcome()).await;
    let (abort_ok, error) = match result {
        Ok(Ok(())) => (Some(true), None),
        Ok(Err(err)) => (Some(false), Some(err.to_string())),
        Err(_) => (
            None,
            Some("the abort path had not reported within 2 s".into()),
        ),
    };
    json!({
        "status": "stopped",
        "command_id": command_id,
        "abort_path_ran": path_ran,
        "cancel_sent": cancel_sent,
        "abort_ok": abort_ok,
        "abort_error": error,
    })
}

// ---------------------------------------------------------------------------
// Registry projection
// ---------------------------------------------------------------------------

type BlockFn<S> = dyn Fn(&S, &DeviceIdentity, &HealthState, u64) -> Value + Send + Sync;

/// Keep the registry's entry for one device current from its watches (§2.5
/// "Registry projection"): Ready and Degraded refresh the `/status` block
/// through `checked` (refused for any other connection id); Lost marks the
/// entry disconnected with its reason, guarded by the connection id, so a
/// late update can neither revive the device nor touch its replacement. The
/// task ends when the worker stops or the handle is retired.
fn spawn_projector<C, S>(state: &AppState, handle: DeviceHandle<C, S>, block: Arc<BlockFn<S>>)
where
    C: Send + 'static,
    S: Send + Sync + 'static,
{
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return;
    };
    let registry = state.devices.clone();
    let updates = state.device_updates.clone();
    let focus_abort = state.focus_abort_requested.clone();
    let clock = state.equipment.clock();
    let task = runtime.spawn(async move {
        let identity = handle.identity();
        let id = identity.connection_id.to_string();
        let kind = identity.kind;
        let mut snapshots = handle.watch_snapshot();
        let mut healths = handle.watch_health();
        let mut last_state: Option<LinkState> = None;
        loop {
            if handle.is_retired() {
                break;
            }
            let health = healths.borrow_and_update().clone();
            let snapshot = Arc::clone(&snapshots.borrow_and_update());
            match health.state {
                LinkState::Ready | LinkState::Degraded => {
                    let value = block(&snapshot, &identity, &health, clock.now_mono_ns());
                    let mut registry = registry.write().unwrap_or_else(|e| e.into_inner());
                    if registry.checked(kind, &id, true, Some(value)) {
                        registry.set_link(kind, &id, &health);
                    }
                }
                LinkState::Lost | LinkState::Disconnected => {
                    let lost = registry
                        .write()
                        .unwrap_or_else(|e| e.into_inner())
                        .lost(kind, &id, &health);
                    if lost && kind == DeviceKind::Focuser {
                        focus_abort.store(true, std::sync::atomic::Ordering::Release);
                    }
                }
                LinkState::Connecting => {}
            }
            if last_state != Some(health.state) {
                last_state = Some(health.state);
                let _ = updates.send(());
            }
            tokio::select! {
                changed = snapshots.changed() => if changed.is_err() { break },
                changed = healths.changed() => if changed.is_err() { break },
            }
        }
    });
    state.equipment_ctx.adopt_task(task);
}

/// Publish a started device (§2.6 step 9): the registry entry with its first
/// block, the device update, and the projector.
fn publish<C, S>(state: &AppState, handle: &DeviceHandle<C, S>, block: Arc<BlockFn<S>>)
where
    C: Send + 'static,
    S: Send + Sync + 'static,
{
    let identity = handle.identity();
    let first = block(
        &handle.snapshot(),
        &identity,
        &handle.health(),
        state.equipment.clock().now_mono_ns(),
    );
    state
        .devices
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .connected(
            identity.kind,
            identity.connection_id.to_string(),
            Some(first),
        );
    let _ = state.device_updates.send(());
    spawn_projector(state, handle.clone(), block);
}

fn connected_body(
    kind: DeviceKind,
    driver: &str,
    option: &str,
    identity: &DeviceIdentity,
    capabilities: Value,
    warnings: &[String],
) -> Value {
    json!({
        "status": "connected",
        "equipment_type": kind_name(kind),
        "driver": driver,
        "option": option,
        "connection_id": identity.connection_id,
        "identity": spine::identity_json(identity),
        "capabilities": capabilities,
        "warnings": warnings,
    })
}

/// Run a connect sequence on a blocking thread (§2.6 steps 2-8; D3: no
/// blocking I/O on an async worker).
async fn blocking<T, F>(kind: DeviceKind, f: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, DeviceError> + Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(err)) => Err(connect_error(kind, &err)),
        Err(err) => Err(ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            static_code(format!("{}_connect_failed", kind_name(kind))),
            format!("the connect worker failed: {err}"),
        )),
    }
}

fn connect_error(kind: DeviceKind, err: &DeviceError) -> ApiError {
    let mut api = device_error(kind, err);
    if err.class == ErrorClass::InvalidOperation
        && err.message.contains("broadcast discovery is disabled")
    {
        // D13: an empty or "auto" Alpaca option without the opt-in.
        api.status = StatusCode::BAD_REQUEST;
        api.code = "broadcast_discovery_disabled";
    }
    api
}

// ---------------------------------------------------------------------------
// Connect (§2.6, §2.12)
// ---------------------------------------------------------------------------

/// `/connect` for every kind but the camera.
pub async fn connect_device(
    state: &AppState,
    kind: DeviceKind,
    req: ConnectRequest,
) -> Result<Value, ApiError> {
    let driver = req.driver.trim().to_ascii_lowercase();
    if driver.is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "missing_driver",
            "driver is required (GET /api/v1/drivers lists them)",
        ));
    }
    let choice = spine::resolve_driver(kind, &driver).map_err(refusal)?;
    let option = req.option.clone().unwrap_or_default();
    // Replacing a device disconnects the old one first (never forced: a
    // slewing mount refuses, 409).
    disconnect_device(state, kind, false).await?;
    match kind {
        DeviceKind::Mount => connect_mount(state, choice, &driver, option).await,
        DeviceKind::Focuser => connect_focuser(state, choice, &driver, option).await,
        DeviceKind::FilterWheel => connect_wheel(state, choice, &driver, option).await,
        DeviceKind::Weather => connect_weather(state, choice, &driver, option).await,
        DeviceKind::SafetyMonitor => connect_safety(state, choice, &driver, option).await,
        DeviceKind::Guider => connect_guider(state, &driver, option).await,
        _ => Err(not_connected(kind)),
    }
}

async fn connect_mount(
    state: &AppState,
    choice: DriverChoice,
    driver: &str,
    option: String,
) -> Result<Value, ApiError> {
    let ctx = Arc::clone(&state.equipment_ctx);
    let blocking_option = option.clone();
    let prepared = blocking(DeviceKind::Mount, move || {
        spine::prepare_mount(&ctx, choice, &blocking_option)
    })
    .await?;
    let warnings = prepared.warnings.clone();
    let shared = prepared.runtime.shared();
    let handle = spine::start_mount(&state.equipment, &state.equipment_ctx, prepared)
        .map_err(|err| device_error(DeviceKind::Mount, &err))?;
    let identity = handle.identity();
    *state.mount.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle.clone());
    *state.mount_name.lock().unwrap_or_else(|e| e.into_inner()) = Some(identity.name.clone());
    let block_shared = Arc::clone(&shared);
    publish(
        state,
        &handle,
        Arc::new(move |snapshot, identity, health, now| {
            status::mount_block(
                snapshot,
                identity,
                health,
                block_shared.session_site().as_ref(),
                now,
            )
        }),
    );
    spine::record_last_used(&state.equipment, &identity, &option);
    let capabilities = serde_json::to_value(shared.capabilities()).unwrap_or_default();
    Ok(connected_body(
        DeviceKind::Mount,
        driver,
        &option,
        &identity,
        capabilities,
        &warnings,
    ))
}

/// Connect and publish a focuser from an existing driver (tests with a
/// scripted or legacy simulator).
#[cfg(test)]
pub async fn install_focuser(
    state: &AppState,
    driver: Box<dyn crate::equipment::traits::FocuserDriver>,
    driver_id: &str,
    option: &str,
) -> Result<Value, ApiError> {
    let ctx = Arc::clone(&state.equipment_ctx);
    let deps = focuser_deps(state);
    let prepared = blocking(DeviceKind::Focuser, move || {
        spine::prepare_focuser_with(&ctx, driver, deps)
    })
    .await?;
    finish_focuser(state, prepared, driver_id, option)
}

fn focuser_deps(state: &AppState) -> FocuserDeps {
    FocuserDeps {
        clock: state.equipment.clock(),
        sink: state.equipment.sequencing_recorder(),
        camera: state.equipment.camera(),
    }
}

fn finish_focuser(
    state: &AppState,
    prepared: spine::PreparedFocuser,
    driver_id: &str,
    option: &str,
) -> Result<Value, ApiError> {
    let warnings = prepared.warnings.clone();
    let capabilities = serde_json::to_value(prepared.session.capabilities()).unwrap_or_default();
    let handle = spine::start_focuser(&state.equipment, &state.equipment_ctx, prepared)
        .map_err(|err| device_error(DeviceKind::Focuser, &err))?;
    let identity = handle.identity();
    *state.focuser.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle.clone());
    *state.focuser_name.lock().unwrap_or_else(|e| e.into_inner()) = Some(identity.name.clone());
    publish(
        state,
        &handle,
        Arc::new(|snapshot, identity, health, now| {
            status::focuser_block(snapshot, identity, health, now)
        }),
    );
    spine::record_last_used(&state.equipment, &identity, option);
    let mut body = connected_body(
        DeviceKind::Focuser,
        driver_id,
        option,
        &identity,
        capabilities,
        &warnings,
    );
    body["position"] = json!(handle.snapshot().logical_position);
    Ok(body)
}

async fn connect_focuser(
    state: &AppState,
    choice: DriverChoice,
    driver: &str,
    option: String,
) -> Result<Value, ApiError> {
    let ctx = Arc::clone(&state.equipment_ctx);
    let deps = focuser_deps(state);
    let blocking_option = option.clone();
    let prepared = blocking(DeviceKind::Focuser, move || {
        spine::prepare_focuser(&ctx, choice, &blocking_option, deps)
    })
    .await?;
    finish_focuser(state, prepared, driver, &option)
}

async fn connect_wheel(
    state: &AppState,
    choice: DriverChoice,
    driver: &str,
    option: String,
) -> Result<Value, ApiError> {
    let ctx = Arc::clone(&state.equipment_ctx);
    let hub = Arc::clone(&state.equipment);
    let blocking_option = option.clone();
    let prepared = blocking(DeviceKind::FilterWheel, move || {
        spine::prepare_wheel(&ctx, &hub, choice, &blocking_option)
    })
    .await?;
    let handle = spine::start_wheel(&state.equipment, &state.equipment_ctx, prepared)
        .map_err(|err| device_error(DeviceKind::FilterWheel, &err))?;
    let identity = handle.identity();
    *state.filter_wheel.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle.clone());
    publish(
        state,
        &handle,
        Arc::new(|snapshot, identity, health, now| {
            status::filter_wheel_block(snapshot, identity, health, now)
        }),
    );
    spine::record_last_used(&state.equipment, &identity, &option);
    let capabilities = json!({"slots": handle.snapshot().slots});
    Ok(connected_body(
        DeviceKind::FilterWheel,
        driver,
        &option,
        &identity,
        capabilities,
        &[],
    ))
}

async fn connect_weather(
    state: &AppState,
    choice: DriverChoice,
    driver: &str,
    option: String,
) -> Result<Value, ApiError> {
    let ctx = Arc::clone(&state.equipment_ctx);
    let blocking_option = option.clone();
    let prepared = blocking(DeviceKind::Weather, move || {
        spine::prepare_weather(&ctx, choice, &blocking_option)
    })
    .await?;
    let handle = spine::start_weather(&state.equipment, &state.equipment_ctx, prepared)
        .map_err(|err| device_error(DeviceKind::Weather, &err))?;
    let identity = handle.identity();
    publish(
        state,
        &handle,
        Arc::new(|snapshot, identity, health, now| {
            status::weather_block(snapshot, identity, health, now)
        }),
    );
    spine::record_last_used(&state.equipment, &identity, &option);
    Ok(connected_body(
        DeviceKind::Weather,
        driver,
        &option,
        &identity,
        Value::Null,
        &[],
    ))
}

async fn connect_safety(
    state: &AppState,
    choice: DriverChoice,
    driver: &str,
    option: String,
) -> Result<Value, ApiError> {
    let ctx = Arc::clone(&state.equipment_ctx);
    let blocking_option = option.clone();
    let prepared = blocking(DeviceKind::SafetyMonitor, move || {
        spine::prepare_safety(&ctx, choice, &blocking_option)
    })
    .await?;
    let handle = spine::start_safety(&state.equipment, &state.equipment_ctx, prepared)
        .map_err(|err| device_error(DeviceKind::SafetyMonitor, &err))?;
    let identity = handle.identity();
    publish(
        state,
        &handle,
        Arc::new(|snapshot, identity, health, now| {
            status::safety_monitor_block(snapshot, identity, health, now)
        }),
    );
    spine::record_last_used(&state.equipment, &identity, &option);
    Ok(connected_body(
        DeviceKind::SafetyMonitor,
        driver,
        &option,
        &identity,
        Value::Null,
        &[],
    ))
}

async fn connect_guider(state: &AppState, driver: &str, option: String) -> Result<Value, ApiError> {
    let (host, port) =
        phd2::parse_option(&option).map_err(|err| device_error(DeviceKind::Guider, &err))?;
    let configs = &state.equipment_ctx.configs;
    let settings = Phd2Settings::from_config(host, port, &configs.guider, &configs.equipment);
    let hub = Arc::clone(&state.equipment);
    let emitter = Arc::clone(&hub);
    let ctx = GuiderContext {
        clock: hub.clock(),
        recorder: hub.recorder(),
        events: Arc::new(move |event| {
            emitter.emit(event);
        }),
    };
    let mut connection = phd2::connect(settings, ctx)
        .await
        .map_err(|err| connect_error(DeviceKind::Guider, &err))?;
    let handle = connection.handle.clone();
    // §2.6 steps 9-10: attach (the hub's connection filter), publish, then
    // start the service, which emits `device_connected`.
    hub.attach(handle.clone());
    let identity = handle.identity();
    let capabilities = connection.capabilities.clone();
    publish(
        state,
        &handle,
        Arc::new(|snapshot, identity, health, now| {
            status::guiding_block(snapshot, identity, health, now)
        }),
    );
    connection.start();
    state.equipment_ctx.set_guider(connection);
    spine::record_last_used(&state.equipment, &identity, &option);
    Ok(connected_body(
        DeviceKind::Guider,
        driver,
        &option,
        &identity,
        capabilities,
        &[],
    ))
}

// ---------------------------------------------------------------------------
// Disconnect (§2.6)
// ---------------------------------------------------------------------------

/// The handle of `kind`, from its `AppState` slot or the hub.
pub fn any_handle(state: &AppState, kind: DeviceKind) -> Option<AnyHandle> {
    match kind {
        DeviceKind::Mount => mount_slot(state).map(AnyHandle::Mount),
        DeviceKind::Focuser => focuser_slot(state).map(AnyHandle::Focuser),
        DeviceKind::FilterWheel => wheel_slot(state).map(AnyHandle::FilterWheel),
        other => state.equipment.any_handle(other),
    }
}

fn clear_slot(state: &AppState, kind: DeviceKind, connection_id: uuid::Uuid) {
    fn take_if<C, S>(
        slot: &std::sync::Mutex<Option<DeviceHandle<C, S>>>,
        connection_id: uuid::Uuid,
    ) -> bool {
        let mut slot = slot.lock().unwrap_or_else(|e| e.into_inner());
        if slot
            .as_ref()
            .is_some_and(|h| h.connection_id() == connection_id)
        {
            slot.take();
            true
        } else {
            false
        }
    }
    match kind {
        DeviceKind::Mount => {
            if take_if(&state.mount, connection_id) {
                state
                    .mount_name
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take();
            }
        }
        DeviceKind::Focuser => {
            if take_if(&state.focuser, connection_id) {
                state
                    .focuser_name
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take();
            }
        }
        DeviceKind::FilterWheel => {
            take_if(&state.filter_wheel, connection_id);
        }
        _ => {}
    }
}

/// Disconnect `kind` (§2.6): a mount slewing refuses unless `force`, which
/// aborts first; disconnect never parks, and PHD2 is never told to
/// disconnect its equipment (D16). The handle is retired, the slot, the
/// registry and the hub are cleared, and the per-connection runtimes are
/// forgotten. True when a device was disconnected.
pub async fn disconnect_device(
    state: &AppState,
    kind: DeviceKind,
    force: bool,
) -> Result<bool, ApiError> {
    let Some(handle) = any_handle(state, kind) else {
        return Ok(false);
    };
    if let Err(err) = handle.disconnect(force).await {
        if err.class == ErrorClass::InvalidOperation {
            return Err(device_error(kind, &err));
        }
        tracing::warn!(
            ?kind,
            "disconnect reported an error; retiring anyway: {err}"
        );
    }
    let connection_id = handle.connection_id();
    let id = connection_id.to_string();
    state
        .devices
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .disconnected(kind, Some(&id), None);
    clear_slot(state, kind, connection_id);
    state.equipment.detach(kind, connection_id);
    if kind == DeviceKind::SafetyMonitor {
        if let Some(supervisor) = state.equipment_ctx.safety_supervisor() {
            supervisor.remove_monitor(connection_id);
        }
    }
    state.equipment_ctx.forget(kind);
    if kind == DeviceKind::Focuser {
        state
            .focus_abort_requested
            .store(true, std::sync::atomic::Ordering::Release);
    }
    let _ = state.device_updates.send(());
    Ok(true)
}

/// Every non-camera kind, in the order `disconnect all` takes them.
pub const AUXILIARY_KINDS: [DeviceKind; 10] = [
    DeviceKind::Guider,
    DeviceKind::Mount,
    DeviceKind::Focuser,
    DeviceKind::FilterWheel,
    DeviceKind::Rotator,
    DeviceKind::CoverCalibrator,
    DeviceKind::Switch,
    DeviceKind::Weather,
    DeviceKind::SafetyMonitor,
    DeviceKind::Dome,
];

// ---------------------------------------------------------------------------
// E-stop, resume blocks and the journal
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
pub struct EstopBody {
    #[serde(default)]
    pub abort_exposure: bool,
}

/// `POST equipment/estop` (§2.9): every device's abort within 2 s,
/// including degraded and lost ones; the `estop` resume block.
pub async fn estop_value(state: &AppState, body: EstopBody, relay: bool) -> Value {
    let report = run_estop(
        &state.equipment,
        EstopRequest {
            abort_exposure: body.abort_exposure,
            source: source(relay, false),
        },
    )
    .await;
    json!({
        "status": if report.all_ok() { "stopped" } else { "partial" },
        "command_id": report.command_id,
        "results": report.results,
        "exposure": report.exposure,
        "block": report.block,
        "elapsed_ms": report.elapsed_ms,
    })
}

pub async fn estop(State(state): State<AppState>, body: Option<Json<EstopBody>>) -> Json<Value> {
    let body = body.map(|Json(body)| body).unwrap_or_default();
    Json(estop_value(&state, body, false).await)
}

/// `GET equipment/resume-blocks`.
pub fn resume_blocks_value(state: &AppState) -> Value {
    let blocks = state.equipment.resume_blocks().borrow().clone();
    json!({ "blocks": status::resume_blocks_block(&blocks) })
}

pub async fn resume_blocks(State(state): State<AppState>) -> Json<Value> {
    Json(resume_blocks_value(&state))
}

#[derive(Debug, Deserialize)]
pub struct ClearBlockBody {
    pub id: uuid::Uuid,
}

/// `POST equipment/resume-blocks/clear {id}`: the operator clear, audited;
/// `safety_*` blocks clear only on a confirmed safe reading (409).
pub fn clear_resume_block_value(
    state: &AppState,
    body: ClearBlockBody,
    relay: bool,
) -> Result<Value, ApiError> {
    let block = state
        .equipment
        .gate()
        .clear_resume_block(body.id, source(relay, false))
        .map_err(|err| {
            let mut api = submit_error(DeviceKind::Camera, &SubmitError::Gate(err));
            if api.message.contains("no active resume block") {
                api.status = StatusCode::NOT_FOUND;
                api.code = "resume_block_not_found";
            }
            api
        })?;
    Ok(json!({ "status": "cleared", "block": block }))
}

pub async fn clear_resume_block(
    State(state): State<AppState>,
    Json(body): Json<ClearBlockBody>,
) -> Result<Json<Value>, ApiError> {
    clear_resume_block_value(&state, body, false).map(Json)
}

#[derive(Debug, Default, Deserialize)]
pub struct EventsQuery {
    #[serde(default)]
    pub since_seq: u64,
    /// Comma-separated event names; empty means every kind.
    #[serde(default)]
    pub kinds: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

/// `GET equipment/events?since_seq=&kinds=`: the journal tail, at most 500.
pub fn events_value(state: &AppState, query: &EventsQuery) -> Value {
    let kinds: Vec<String> = query
        .kinds
        .as_deref()
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|k| !k.is_empty())
        .map(str::to_string)
        .collect();
    let events = state
        .equipment
        .events_since(query.since_seq, &kinds, query.limit.unwrap_or(500));
    let next = events.last().map_or(query.since_seq, |e| e.seq);
    json!({ "events": events, "next_seq": next })
}

pub async fn events(
    State(state): State<AppState>,
    Query(query): Query<EventsQuery>,
) -> Json<Value> {
    Json(events_value(&state, &query))
}

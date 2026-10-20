use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::hardware::{capabilities::driver_capability_report, mount::EqCoord};
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct RelayCommand {
    pub command_id: String,
    pub action: String,
    #[serde(default)]
    pub payload: Value,
}

fn payload_str(payload: &Value, key: &str, default: &str) -> String {
    payload
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_string()
}

fn payload_i32(payload: &Value, key: &str) -> Result<i32, String> {
    payload
        .get(key)
        .and_then(Value::as_i64)
        .map(|v| v as i32)
        .ok_or_else(|| format!("payload.{key} is required"))
}

fn payload_f64(payload: &Value, key: &str) -> Result<f64, String> {
    payload
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("payload.{key} is required"))
}

fn payload_bool(payload: &Value, key: &str) -> Result<bool, String> {
    payload
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("payload.{key} is required"))
}

/// One entry in the daemon's published command registry.
///
/// `method` and `path` describe the daemon's *own* HTTP route for the same
/// operation, so a client that cannot use the relay can reach it directly.
/// Paths are relative to `/api/v1`.
#[derive(Debug, Clone, Serialize)]
pub struct CommandSpec {
    pub name: &'static str,
    pub method: &'static str,
    pub path: &'static str,
    pub description: &'static str,
}

/// Declares the relay commands exactly once.
///
/// This exists because the command list was previously transcribed by hand in
/// four places — this dispatcher, Django's fallback whitelist, and both
/// frontend clients — with no test that they agreed, and they did not: five
/// commands were missing from Django, and `camera.abort` meant one thing over
/// the relay and another over Django's localhost fallback.
///
/// Publishing a `const COMMANDS` table *next to* a hand-written `match` would
/// have made a fifth copy that drifts the first time someone adds an arm and
/// forgets the list. Emitting both from a single declaration is the only shape
/// where drift is impossible rather than merely tested for.
///
/// Handler bodies see `state: AppState` and `payload: &Value` in scope. Both
/// binding names are passed in from the call site: `macro_rules!` hygiene means
/// an identifier introduced inside the macro is invisible to expressions
/// supplied by the caller.
macro_rules! command_registry {
    ($state:ident, $payload:ident; $(
        $name:literal $method:ident $path:literal
            $description:literal
            => $body:expr
    ),* $(,)?) => {
        /// Every relay command this build handles, published at
        /// `GET /api/v1/commands`. Generated from the same declaration the
        /// dispatcher matches on, so it cannot disagree with it.
        pub const COMMANDS: &[CommandSpec] = &[
            $(CommandSpec {
                name: $name,
                method: stringify!($method),
                path: $path,
                description: $description,
            },)*
        ];

        pub async fn execute($state: AppState, command: &RelayCommand) -> Result<Value, String> {
            // Bound once so handler bodies read the same whether or not they
            // use the payload; unused in the arms that take no arguments.
            let $payload = &command.payload;
            let _ = $payload;

            match command.action.as_str() {
                $($name => $body,)*
                other => Err(format!(
                    "Unsupported relay action: {other}. See GET /api/v1/commands for the {} this daemon handles.",
                    COMMANDS.len()
                )),
            }
        }
    };
}

// ── Why `focus.*` is not here ────────────────────────────────────────────
//
// This is a decision, not an oversight. The daemon has six focus endpoints
// (`/focus/run`, `/abort`, `/status`, `/history`, `/drift`, `/drift/history`)
// and none of them are relayed, so autofocus cannot be driven through
// achernar.com. Three things would have to change together, not one:
//
//   1. Six arms here, wrapping handlers that already exist. Thin.
//   2. `cloud_client.rs` would have to forward `FOCUS_*` events upstream.
//      It sends only HELLO, HEARTBEAT, TELEMETRY_UPDATE and COMMAND_RESULT.
//   3. Django's `DaemonConsumer` would have to forward them to the browser.
//      It handles those same four plus FRAME_PREVIEW and drops the rest.
//
// Without 2 and 3 you get a sweep you can start from the cloud and cannot
// watch: a run takes minutes, the relay returns `202 pending` after three
// seconds, and the V-curve samples never cross the boundary. A start button
// with no telemetry is worse than no button.
//
// The case for doing it anyway was "start a sweep from a laptop indoors while
// the rig is outside" — but that is a LAN scenario, and the direct API already
// serves it: the laptop reaches `/focus/run` and `ws://rig:3030/ws/telemetry`
// without the cloud involved. The relay is only required for *off-site*
// control, which is not a current requirement.
//
// So: focus stays direct-daemon-only until off-site operation is actually
// needed. Revisit all three layers together when it is.

command_registry! {
    state, payload;

    "equipment.discover" POST "discover"
        "Scan local cameras or query an explicitly addressed Alpaca server for mount/focuser devices."
        => {
            let request = if payload.is_null() {
                crate::hardware::discovery::DiscoveryRequest::default()
            } else {
                serde_json::from_value(payload.clone()).map_err(|error| error.to_string())?
            };
            let report = crate::hardware::discovery::discover_equipment(&state, request).await?;
            serde_json::to_value(report).map_err(|error| error.to_string())
        },

    "equipment.connect" POST "connect"
        "Connect a camera, mount or focuser by driver id and device index."
        => connect_equipment(state, payload).await,

    "equipment.disconnect" POST "disconnect"
        "Disconnect a device and release its driver."
        => disconnect_equipment(state, payload).await,

    "equipment.status" GET "status"
        "Connection state and current readings for every attached device. \
         Unmeasured values are null, never a plausible zero."
        => Ok(crate::telemetry::equipment_status(&state)),

    "coc.status" GET "coc/status"
        "Research COC preview maps and freshness; never a control signal."
        => Ok(crate::api::coc::status_value(&state)),

    "coc.configure" POST "coc/configure"
        "Enable or disable shadow inference and set an independent sky region."
        => crate::api::coc::configure_value(&state, serde_json::from_value(payload.clone()).map_err(|e|e.to_string())?),

    "coc.freeze_sample" POST "coc/freeze-sample"
        "Freeze one current rendered preview for a bounded operator review."
        => crate::api::coc::freeze_value(&state, serde_json::from_value(payload.clone()).map_err(|e|e.to_string())?),

    "coc.save_sample" POST "coc/save-sample"
        "Save the displayed preview and model outputs with unadjudicated review notes."
        => crate::api::coc::save_value(state, serde_json::from_value(payload.clone()).map_err(|e|e.to_string())?).await,

    "drivers.capabilities" GET "drivers"
        "Which drivers this build actually contains, and which are usable."
        => Ok(json!(driver_capability_report())),

    "camera.descriptor" GET "camera"
        "Sensor geometry, pixel size and bit depth for the connected camera."
        => camera_descriptor(state).await,

    "camera.parameters" GET "camera/parameters"
        "Every controllable camera parameter with its range and current value."
        => camera_parameters(state).await,

    "camera.formats" GET "camera/formats"
        "Stream formats the connected camera offers."
        => camera_formats(state).await,

    "camera.set_parameter" POST "camera/parameters"
        "Set one camera parameter. Requires `name` and `value`."
        => camera_set_parameter(state, payload).await,

    "camera.start_stream" POST "camera/stream/start"
        "Start the live view. Optional `format_index` selects a stream format."
        => start_camera_stream(state, payload).await,

    "camera.stop_stream" POST "camera/stream/stop"
        "Stop the live view. Errors when no camera is connected."
        => stop_camera_stream(state).await,

    "camera.abort" POST "camera/abort"
        "Stop the live view and abort any exposure. Idempotent: succeeds even \
         when no camera is connected."
        => abort_camera(state).await,

    "camera.capture" POST "camera/capture"
        "Take one exposure and write a FITS frame."
        => camera_capture(state, payload).await,

    "camera.snapshot" POST "camera/snapshot"
        "Latest preview frame as a JPEG payload, for the browser path."
        => camera_snapshot(state),

    "mount.slew" POST "mount/slew"
        "Slew to equatorial coordinates. Requires `ra` and `dec`."
        => mount_slew(state, payload),

    "mount.sync" POST "mount/sync"
        "Tell the mount its current position. Requires `ra` and `dec`; a wrong \
         value here corrupts the pointing model for the rest of the session."
        => mount_sync(state, payload),

    "mount.set_tracking" POST "mount/tracking"
        "Start or stop sidereal tracking. Requires `enabled`."
        => mount_set_tracking(state, payload),

    "mount.abort" POST "mount/abort"
        "Stop any slew immediately."
        => mount_abort(state),

    "focuser.move_relative" POST "focuser/move"
        "Move the focuser by `steps`, signed."
        => focuser_move_relative(state, payload),

    "focuser.move_absolute" POST "focuser/moveto"
        "Move the focuser to absolute `position`. Step 0 is a real position, \
         so a missing argument must not default to it."
        => focuser_move_absolute(state, payload),

    "focuser.position" GET "focuser/position"
        "Current focuser position, or null when the encoder has not been read."
        => focuser_position(state),

    "focuser.abort" POST "focuser/abort"
        "Stop focuser motion immediately."
        => focuser_abort(state),
}

async fn camera_descriptor(state: AppState) -> Result<Value, String> {
    let axum::Json(descriptor) =
        crate::api::camera::get_camera_descriptor(axum::extract::State(state))
            .await
            .map_err(|error| error.message)?;
    serde_json::to_value(descriptor).map_err(|error| error.to_string())
}

async fn camera_parameters(state: AppState) -> Result<Value, String> {
    let axum::Json(parameters) = crate::api::camera::get_parameters(axum::extract::State(state))
        .await
        .map_err(|error| error.message)?;
    serde_json::to_value(parameters).map_err(|error| error.to_string())
}

async fn camera_formats(state: AppState) -> Result<Value, String> {
    let axum::Json(formats) = crate::api::camera::get_formats(axum::extract::State(state))
        .await
        .map_err(|error| error.message)?;
    serde_json::to_value(formats).map_err(|error| error.to_string())
}

async fn camera_set_parameter(state: AppState, payload: &Value) -> Result<Value, String> {
    let name = payload
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "payload.name is required".to_string())?;
    let req = crate::api::camera::SetParamRequest {
        name: name.to_string(),
        value: payload_f64(payload, "value")?,
    };
    let axum::Json(value) =
        crate::api::camera::set_parameter(axum::extract::State(state), axum::Json(req))
            .await
            .map_err(|error| error.message)?;
    Ok(value)
}

/// Cloud-relay capture.
///
/// Delegates to the single shared implementation in `api::camera` rather than
/// keeping a second copy. The duplicate that used to live here carried the
/// same defects as the HTTP path — it wrote `Exp`/`Gain` to bodies that only
/// understand `shutterspeed`/`iso`, discarded the resulting errors with
/// `let _ =`, and then recorded the *requested* exposure in the FITS header —
/// so a remote capture could disagree with the photograph it claimed to
/// describe. One implementation means one place to be correct.
async fn camera_capture(state: AppState, payload: &Value) -> Result<Value, String> {
    let params: crate::api::camera::CaptureParams =
        serde_json::from_value(payload.clone()).map_err(|e| e.to_string())?;

    let image = crate::api::camera::perform_capture(state, params)
        .await
        .map_err(|error| error.message)?;

    serde_json::to_value(image).map_err(|e| e.to_string())
}

async fn connect_equipment(state: AppState, payload: &Value) -> Result<Value, String> {
    let equipment_type = payload_str(payload, "equipment_type", "camera").to_ascii_lowercase();
    let driver = payload_str(payload, "driver", "sim");
    let option = payload
        .get("option")
        .and_then(Value::as_str)
        .map(str::to_string);

    let index = payload
        .get("index")
        .and_then(Value::as_i64)
        .map(|value| value as i32);

    let request = crate::api::ConnectRequest {
        equipment_type: Some(equipment_type),
        driver,
        option,
        index,
    };
    crate::api::connect_equipment(axum::extract::State(state), axum::Json(request))
        .await
        .map(|axum::Json(value)| value)
        .map_err(|error| error.message)
}

async fn disconnect_equipment(state: AppState, payload: &Value) -> Result<Value, String> {
    let equipment_type = payload_str(payload, "equipment_type", "all");
    let axum::Json(response) = crate::api::disconnect_equipment(
        axum::extract::State(state),
        Some(axum::Json(crate::api::DisconnectRequest {
            equipment_type: Some(equipment_type.clone()),
        })),
    )
    .await
    .map_err(|error| error.message)?;
    Ok(response)
}

async fn start_camera_stream(state: AppState, payload: &Value) -> Result<Value, String> {
    let format_index = match payload.get("format_index").and_then(Value::as_i64) {
        Some(value) if value < 0 => return Err("format_index cannot be negative".to_string()),
        Some(value) => Some(value as usize),
        None => None,
    };
    crate::api::camera::start_stream_control(&state, format_index)
        .await
        .map_err(|error| error.message)
}

/// Strict stop, mirroring `POST /camera/stream/stop`: errors when no camera
/// is connected.
async fn stop_camera_stream(state: AppState) -> Result<Value, String> {
    crate::api::camera::stop_stream_control(&state)
        .await
        .map_err(|error| error.message)
}

/// Idempotent stop, mirroring `POST /camera/abort`: succeeds even when no
/// camera is connected.
///
/// These two names used to share one arm and both got the strict behaviour,
/// so `camera.abort` meant one thing over the relay and another over Django's
/// localhost fallback — which posts to `/camera/abort` and gets the lenient
/// handler. Same action name, opposite behaviour, chosen by whichever
/// transport happened to be up. They agree now.
async fn abort_camera(state: AppState) -> Result<Value, String> {
    let axum::Json(value) = crate::api::camera::abort(axum::extract::State(state))
        .await
        .map_err(|error| error.message)?;
    Ok(value)
}

fn camera_snapshot(state: AppState) -> Result<Value, String> {
    crate::api::camera::preview_snapshot_value(&state)
        .map_err(|status| format!("No current camera preview is available: {status}"))
}

fn mount_slew(state: AppState, payload: &Value) -> Result<Value, String> {
    let target = EqCoord {
        ra: payload_f64(payload, "ra")?,
        dec: payload_f64(payload, "dec")?,
    };
    {
        let mount = state.mount.lock().unwrap();
        crate::hardware::health::require_connected(
            &state,
            crate::hardware::health::DeviceKind::Mount,
        )?;
        let mount = mount
            .as_ref()
            .ok_or_else(|| "No mount connected".to_string())?;
        mount.goto(target).map_err(|e| e.to_string())?;
    }
    // Same rule as the local route: the post-slew solve waits for arrival.
    crate::mount::auto_solve::slew_commanded(&state, target);
    Ok(json!({ "status": "slewing", "ra": target.ra, "dec": target.dec }))
}

fn mount_sync(state: AppState, payload: &Value) -> Result<Value, String> {
    let coord = EqCoord {
        ra: payload_f64(payload, "ra")?,
        dec: payload_f64(payload, "dec")?,
    };
    let mount = state.mount.lock().unwrap();
    crate::hardware::health::require_connected(&state, crate::hardware::health::DeviceKind::Mount)?;
    let mount = mount
        .as_ref()
        .ok_or_else(|| "No mount connected".to_string())?;
    mount.sync(coord).map_err(|e| e.to_string())?;
    Ok(json!({ "status": "synced", "ra": coord.ra, "dec": coord.dec }))
}

fn mount_set_tracking(state: AppState, payload: &Value) -> Result<Value, String> {
    let enabled = payload_bool(payload, "enabled")?;
    let mount = state.mount.lock().unwrap();
    crate::hardware::health::require_connected(&state, crate::hardware::health::DeviceKind::Mount)?;
    let mount = mount
        .as_ref()
        .ok_or_else(|| "No mount connected".to_string())?;
    mount.set_tracking(enabled).map_err(|e| e.to_string())?;
    Ok(json!({ "status": "tracking_updated", "enabled": enabled }))
}

fn mount_abort(state: AppState) -> Result<Value, String> {
    let mount = state.mount.lock().unwrap();
    crate::hardware::health::require_connected(&state, crate::hardware::health::DeviceKind::Mount)?;
    let mount = mount
        .as_ref()
        .ok_or_else(|| "No mount connected".to_string())?;
    mount.stop().map_err(|e| e.to_string())?;
    Ok(json!({ "status": "stopped" }))
}

fn focuser_move_relative(state: AppState, payload: &Value) -> Result<Value, String> {
    let steps = payload_i32(payload, "steps")?;
    let mut focuser = state.focuser.lock().unwrap();
    crate::hardware::health::require_connected(
        &state,
        crate::hardware::health::DeviceKind::Focuser,
    )?;
    let focuser = focuser
        .as_mut()
        .ok_or_else(|| "No focuser connected".to_string())?;
    focuser.move_relative(steps).map_err(|e| e.to_string())?;
    Ok(json!({ "status": "moving", "steps": steps }))
}

fn focuser_move_absolute(state: AppState, payload: &Value) -> Result<Value, String> {
    let position = payload_i32(payload, "position")?;
    let mut focuser = state.focuser.lock().unwrap();
    crate::hardware::health::require_connected(
        &state,
        crate::hardware::health::DeviceKind::Focuser,
    )?;
    let focuser = focuser
        .as_mut()
        .ok_or_else(|| "No focuser connected".to_string())?;
    focuser.move_absolute(position).map_err(|e| e.to_string())?;
    Ok(json!({ "status": "moving", "position": position }))
}

fn focuser_position(state: AppState) -> Result<Value, String> {
    let focuser = state.focuser.lock().unwrap();
    crate::hardware::health::require_connected(
        &state,
        crate::hardware::health::DeviceKind::Focuser,
    )?;
    let focuser = focuser
        .as_ref()
        .ok_or_else(|| "No focuser connected".to_string())?;
    let position = focuser.position().map_err(|e| e.to_string())?;
    Ok(json!({ "position": position }))
}

fn focuser_abort(state: AppState) -> Result<Value, String> {
    let mut focuser = state.focuser.lock().unwrap();
    crate::hardware::health::require_connected(
        &state,
        crate::hardware::health::DeviceKind::Focuser,
    )?;
    let focuser = focuser
        .as_mut()
        .ok_or_else(|| "No focuser connected".to_string())?;
    focuser.abort().map_err(|e| e.to_string())?;
    Ok(json!({ "status": "stopped" }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// The dispatcher and this table come from one declaration, so they cannot
    /// disagree about *which* commands exist. These guard the parts the macro
    /// cannot check: that the published routes are well formed, and that the
    /// declaration was not accidentally emptied.
    #[test]
    fn registry_is_populated() {
        assert!(
            COMMANDS.len() >= 20,
            "only {} commands declared — the registry looks truncated",
            COMMANDS.len()
        );
    }

    #[test]
    fn command_names_are_unique() {
        let mut seen = HashSet::new();
        for spec in COMMANDS {
            assert!(
                seen.insert(spec.name),
                "duplicate command name: {}. The later arm would be unreachable.",
                spec.name
            );
        }
    }

    #[test]
    fn commands_are_namespaced() {
        for spec in COMMANDS {
            assert!(
                spec.name.contains('.'),
                "{} is not namespaced as `category.action`",
                spec.name
            );
        }
    }

    #[test]
    fn published_paths_are_relative_to_api_v1() {
        for spec in COMMANDS {
            assert!(
                !spec.path.starts_with('/'),
                "{}: path {:?} should be relative to /api/v1, with no leading slash",
                spec.name,
                spec.path
            );
            assert!(!spec.path.is_empty(), "{}: empty path", spec.name);
        }
    }

    #[test]
    fn published_methods_are_valid() {
        for spec in COMMANDS {
            assert!(
                matches!(spec.method, "GET" | "POST"),
                "{}: unexpected method {:?}",
                spec.name,
                spec.method
            );
        }
    }

    #[test]
    fn every_command_is_described() {
        for spec in COMMANDS {
            assert!(
                spec.description.len() > 20,
                "{}: description is too thin to help a client author",
                spec.name
            );
        }
    }

    /// Autofocus is deliberately absent — see the note above the registry.
    /// If someone adds `focus.*` arms, they must also wire FOCUS_* forwarding
    /// through cloud_client.rs and Django's consumer, so this fails loudly to
    /// make them read why.
    #[test]
    fn focus_is_not_relayed() {
        let focus: Vec<_> = COMMANDS
            .iter()
            .filter(|s| s.name.starts_with("focus."))
            .map(|s| s.name)
            .collect();
        assert!(
            focus.is_empty(),
            "focus commands were added to the relay: {focus:?}. Adding the arms \
             is only step 1 of 3 — cloud_client.rs must forward FOCUS_* upstream \
             and Django's DaemonConsumer must forward them to the browser, or a \
             sweep can be started from the cloud and not watched. See the note \
             above command_registry!."
        );
    }
}

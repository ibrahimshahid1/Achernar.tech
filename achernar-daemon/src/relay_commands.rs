use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::hardware::capabilities::driver_capability_report;
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

fn payload_f64(payload: &Value, key: &str) -> Result<f64, String> {
    payload
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("payload.{key} is required"))
}

/// One entry in the daemon's published command registry.
///
/// `method` and `path` describe the daemon's *own* HTTP route for the same
/// operation, so a client that cannot use the relay can reach it directly.
/// Paths are relative to `/api/v1`.
///
/// `relay_allowed` is false for commands that are unsafe at relay latency
/// (jog and pulse guiding, D12): they are published for the direct
/// transport and refused by the relay dispatcher. Django reads the registry
/// as a name-keyed dict and ignores the extra key.
#[derive(Debug, Clone, Serialize)]
pub struct CommandSpec {
    pub name: &'static str,
    pub method: &'static str,
    pub path: &'static str,
    pub description: &'static str,
    pub relay_allowed: bool,
}

/// `direct_only` in a registry entry: published for the direct API,
/// refused over the relay (D12).
macro_rules! relay_allowed {
    () => {
        true
    };
    (direct_only) => {
        false
    };
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
        $name:literal $method:ident $path:literal $($flag:ident)?
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
                relay_allowed: relay_allowed!($($flag)?),
            },)*
        ];

        pub async fn execute($state: AppState, command: &RelayCommand) -> Result<Value, String> {
            // Bound once so handler bodies read the same whether or not they
            // use the payload; unused in the arms that take no arguments.
            let $payload = &command.payload;
            let _ = $payload;

            // D12: availability on the direct API does not imply relay
            // availability.
            if spec(&command.action).is_some_and(|spec| !spec.relay_allowed) {
                return Err(format!(
                    "{} is not available over the relay; use the direct daemon API",
                    command.action
                ));
            }

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
        "Connect a camera, mount, focuser, filter wheel, guider, weather station or safety \
         monitor by driver id and option. `driver` is required."
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

    "mount.descriptor" GET "mount"
        "Mount identity, capabilities, native frame, site, can_flip and qualification."
        => relay(crate::api::mount::descriptor_value(&state)),

    "mount.slew" POST "mount/slew"
        "Slew to equatorial coordinates through the gate. Requires `ra` and `dec`; \
         optional `frame` (mount, j2000 or jnow)."
        => relay(crate::api::mount::slew_value(&state, body(payload)?, true).await),

    "mount.sync" POST "mount/sync"
        "Tell the mount its current position. Requires `ra` and `dec`; a wrong \
         value here corrupts the pointing model for the rest of the session."
        => relay(crate::api::mount::sync_value(&state, body(payload)?, true).await),

    "mount.set_tracking" POST "mount/tracking"
        "Start or stop sidereal tracking. Requires `enabled`."
        => relay(crate::api::mount::set_tracking_value(&state, body(payload)?, true).await),

    "mount.abort" POST "mount/abort"
        "Stop any slew or jog immediately: AbortSlew out of band, also on a degraded \
         or lost mount."
        => relay(crate::api::mount::abort_value(&state, true).await),

    "mount.park" POST "mount/park"
        "Park the mount, verified through AtPark."
        => relay(crate::api::mount::park_value(&state, true, false).await),

    "mount.unpark" POST "mount/unpark"
        "Unpark the mount, verified through AtPark."
        => relay(crate::api::mount::unpark_value(&state, true, false).await),

    "mount.find_home" POST "mount/home"
        "Find the mount's home position, verified through AtHome."
        => relay(crate::api::mount::home_value(&state, true).await),

    "mount.center" POST "mount/center"
        "Center a target by slew, solve and sync or offset, then re-slew, until within \
         tolerance. Requires `ra` and `dec`; optional `frame`."
        => relay(crate::api::mount::center_value(&state, body(payload)?, true).await),

    "mount.move_axis" POST "mount/move-axis" direct_only
        "Jog one axis at `rate_deg_s`; renew within jog_deadman_ms. Direct API only: a \
         dead-man is not safe at relay latency."
        => relay(crate::api::mount::move_axis_value(&state, body(payload)?, true).await),

    "mount.pulse_guide" POST "mount/pulse-guide" direct_only
        "One guide pulse `{direction, duration_ms}`; refused while PHD2 guides. Direct \
         API only."
        => relay(crate::api::mount::pulse_guide_value(&state, body(payload)?, true).await),

    "mount.set_tracking_rate" POST "mount/tracking-rate"
        "Set the drive rate: sidereal, lunar, solar or king."
        => relay(crate::api::mount::tracking_rate_value(&state, body(payload)?, true).await),

    "mount.set_guide_rates" POST "mount/guide-rates"
        "Set the RA and Dec guide rates in arcsec/s."
        => relay(crate::api::mount::guide_rates_value(&state, body(payload)?, true).await),

    "mount.site_sync" POST "mount/site-sync"
        "Write [site] to the mount (to_mount) or adopt the mount's site (from_mount)."
        => relay(crate::api::mount::site_sync_value(&state, body(payload)?, true).await),

    "focuser.descriptor" GET "focuser"
        "Focuser capabilities, backlash model and owner, and position source."
        => relay(crate::api::focuser::descriptor_value(&state)),

    "focuser.move_relative" POST "focuser/move"
        "Move the focuser by `steps`, signed, within the travel bounds."
        => relay(crate::api::focuser::move_relative_value(&state, body(payload)?, true).await),

    "focuser.move_absolute" POST "focuser/moveto"
        "Move the focuser to absolute `position`. Step 0 is a real position, \
         so a missing argument must not default to it."
        => relay(crate::api::focuser::move_absolute_value(&state, body(payload)?, true).await),

    "focuser.position" GET "focuser/position"
        "Current logical focuser position; an error when it has not been read."
        => relay(crate::api::focuser::position_value(&state).map(|position| json!({ "position": position }))),

    "focuser.abort" POST "focuser/abort"
        "Stop focuser motion immediately (Halt out of band)."
        => relay(crate::api::focuser::abort_value(&state, true).await),

    "focuser.set_tempcomp" POST "focuser/tempcomp"
        "Turn the focuser's temperature compensation on or off. Requires `enabled`."
        => relay(crate::api::focuser::set_tempcomp_value(&state, body(payload)?, true).await),

    "focuser.sync" POST "focuser/sync"
        "Sync the focuser position where the device supports it (INDI FOCUS_SYNC)."
        => relay(crate::api::focuser::sync_value(&state, body(payload)?, true).await),

    "focuser.set_position" POST "focuser/set-position"
        "Declare the origin of an emulated relative focuser; profile only, audited."
        => relay(crate::api::focuser::set_position_value(&state, body(payload)?, true)),

    "filter_wheel.descriptor" GET "filter-wheel"
        "Filter-wheel identity, reconciled slots, position and verification."
        => relay(crate::api::filter_wheel::descriptor_value(&state)),

    "filter_wheel.change" POST "filter-wheel/change"
        "Change filter by `slot` (0-based) or `name` through the v6 handshake."
        => relay(crate::api::filter_wheel::change_value(&state, body(payload)?, true)),

    "filter_wheel.abort" POST "filter-wheel/abort"
        "Cancel the running filter handshake; the wheel itself has no halt."
        => relay(crate::api::filter_wheel::abort_value(&state, true)),

    "filter_wheel.import_seed" POST "filter-wheel/import-seed"
        "Copy the [filter_wheel] TOML seeds into the connected wheel's profile, audited."
        => relay(crate::api::filter_wheel::import_seed_value(&state, true)),

    "guider.descriptor" GET "guider"
        "Guider state, RMS, settle, calibration and algorithm parameters."
        => relay(crate::api::guider::descriptor_value(&state)),

    "guider.start" POST "guider/start"
        "Start guiding with optional `settle` and `recalibrate`."
        => relay(crate::api::guider::start_value(&state, body(payload)?, true).await),

    "guider.stop" POST "guider/stop"
        "Stop guiding (stop_capture); never refused while the guider is reachable."
        => relay(crate::api::guider::stop_value(&state, true).await),

    "guider.pause" POST "guider/pause"
        "Pause guiding; `full` also pauses looping exposures."
        => relay(crate::api::guider::pause_value(&state, body(payload)?, true).await),

    "guider.resume" POST "guider/resume"
        "Resume guiding after a pause."
        => relay(crate::api::guider::resume_value(&state, true).await),

    "guider.dither" POST "guider/dither"
        "Dither by `amount_px` guide pixels between exposures, with settle verification."
        => relay(crate::api::guider::dither_value(&state, body(payload)?, true).await),

    "weather.status" GET "weather"
        "Weather reading with every field's state and age; null when unknown."
        => relay(crate::api::weather::status_value(&state)),

    "safety.status" GET "safety"
        "Combined safety state of every monitor (none without one) and safety blocks."
        => Ok(crate::api::safety::status_value(&state)),

    "equipment.estop" POST "equipment/estop"
        "Emergency stop: every device's abort within 2 s, including lost ones; \
         `abort_exposure` optional. Never waits for a relay slot."
        => Ok(crate::api::equipment::estop_value(&state, estop_body(payload), true).await),

    "equipment.resume_blocks" GET "equipment/resume-blocks"
        "Active resume blocks that refuse automatic captures and sequencing."
        => Ok(crate::api::equipment::resume_blocks_value(&state)),

    "equipment.clear_resume_block" POST "equipment/resume-blocks/clear"
        "Clear one resume block by `id`, audited; safety blocks clear only when safe."
        => relay(crate::api::equipment::clear_resume_block_value(&state, body(payload)?, true)),

    "equipment.events" GET "equipment/events"
        "Equipment event journal tail after `since_seq`, at most 500, optionally by `kinds`."
        => Ok(crate::api::equipment::events_value(&state, &body(payload)?)),
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
///
/// The gate audits it as the relay (H21): an operator request arrives as
/// `CommandSource::Relay`; a trigger or policy keeps its own source.
async fn camera_capture(state: AppState, payload: &Value) -> Result<Value, String> {
    let params: crate::api::camera::CaptureParams =
        serde_json::from_value(payload.clone()).map_err(|e| e.to_string())?;

    let origin = crate::api::camera::CaptureOrigin::relay(&params);
    let image = crate::api::camera::perform_capture_as(state, params, origin)
        .await
        .map_err(|error| error.message)?;

    serde_json::to_value(image).map_err(|e| e.to_string())
}

async fn connect_equipment(state: AppState, payload: &Value) -> Result<Value, String> {
    let equipment_type = payload_str(payload, "equipment_type", "camera").to_ascii_lowercase();
    // No default driver (§2.12): a missing driver gets the direct route's 400.
    let driver = payload_str(payload, "driver", "");
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
        .map_err(crate::api::equipment::relay_error)
}

async fn disconnect_equipment(state: AppState, payload: &Value) -> Result<Value, String> {
    let equipment_type = payload_str(payload, "equipment_type", "all");
    let axum::Json(response) = crate::api::disconnect_equipment(
        axum::extract::State(state),
        Some(axum::Json(crate::api::DisconnectRequest {
            force: payload
                .get("force")
                .and_then(Value::as_bool)
                .unwrap_or(false),
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

/// Decode a relay payload into a route's request body; a null payload is an
/// empty object, so optional bodies work over the relay too.
fn body<T: serde::de::DeserializeOwned>(payload: &Value) -> Result<T, String> {
    let value = if payload.is_null() {
        json!({})
    } else {
        payload.clone()
    };
    serde_json::from_value(value).map_err(|error| format!("invalid payload: {error}"))
}

/// The e-stop's body over the relay. The e-stop is never refused (§2.9): a
/// missing, null or mistyped `abort_exposure` reads as false, as the direct
/// route treats a body it cannot read.
fn estop_body(payload: &Value) -> crate::api::equipment::EstopBody {
    crate::api::equipment::EstopBody {
        abort_exposure: payload
            .get("abort_exposure")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    }
}

/// The relay's error string for a route error: `code: message`, so the
/// relay carries the same error meaning as the direct route (D12).
fn relay(result: Result<Value, crate::api::api_error::ApiError>) -> Result<Value, String> {
    result.map_err(crate::api::equipment::relay_error)
}

/// True for actions that must never wait for a relay command slot: aborts,
/// halts, the guider's stop (its abort, D19) and the e-stop.
pub fn bypasses_slots(action: &str) -> bool {
    matches!(action, "equipment.estop" | "guider.stop")
        || action.ends_with(".abort")
        || action.ends_with(".halt")
}

/// The published spec of `action`, if any.
pub fn spec(action: &str) -> Option<&'static CommandSpec> {
    COMMANDS.iter().find(|spec| spec.name == action)
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

    /// D12: jog and pulse guiding are published for the direct transport
    /// and refused by the relay dispatcher; aborts, halts and the e-stop are
    /// always relayable and skip the command slots (D19).
    #[test]
    fn jog_and_pulse_are_not_relayed() {
        let direct_only: Vec<_> = COMMANDS
            .iter()
            .filter(|s| !s.relay_allowed)
            .map(|s| s.name)
            .collect();
        assert_eq!(direct_only, vec!["mount.move_axis", "mount.pulse_guide"]);
        for name in [
            "mount.abort",
            "focuser.abort",
            "filter_wheel.abort",
            "guider.stop",
            "equipment.estop",
        ] {
            let spec = spec(name).unwrap_or_else(|| panic!("{name} is not registered"));
            assert!(spec.relay_allowed, "{name} must be relayable");
            assert!(bypasses_slots(name), "{name} must bypass the relay slots");
        }
        assert!(!bypasses_slots("mount.slew"));
        assert!(!bypasses_slots("guider.start"));
        let json = serde_json::to_value(spec("mount.move_axis").unwrap()).unwrap();
        assert_eq!(json["relay_allowed"], false);
    }

    #[tokio::test]
    async fn the_relay_refuses_direct_only_commands_before_running_them() {
        let (directory, state) = crate::api::equipment_tests::state().await;
        for action in ["mount.move_axis", "mount.pulse_guide"] {
            let err = execute(
                state.clone(),
                &RelayCommand {
                    command_id: "direct-only".into(),
                    action: action.into(),
                    payload: json!({"axis": 0, "rate_deg_s": 1.0, "direction": "north", "duration_ms": 100}),
                },
            )
            .await
            .unwrap_err();
            assert!(
                err.contains("not available over the relay; use the direct daemon API"),
                "{err}"
            );
        }
        crate::api::equipment_tests::cleanup(directory, state).await;
    }

    /// H21: a relayed capture reaches the gate as the relay, not as the
    /// direct operator; the attempt row records which.
    #[tokio::test]
    async fn a_relayed_capture_is_gated_as_the_relay() {
        let (directory, state) = crate::api::equipment_tests::state().await;
        let relayed = execute(
            state.clone(),
            &RelayCommand {
                command_id: "relay-capture".into(),
                action: "camera.capture".into(),
                payload: json!({"exposure": 1.0}),
            },
        )
        .await
        .unwrap_err();
        // Past the gate, refused at the camera lookup (no camera here).
        assert!(relayed.contains("No camera connected"), "{relayed}");
        let direct: crate::api::camera::CaptureParams =
            serde_json::from_value(json!({"exposure": 1.0})).unwrap();
        crate::api::camera::perform_capture(state.clone(), direct)
            .await
            .unwrap_err();
        let sources: Vec<(String,)> =
            sqlx::query_as("SELECT requested_json FROM capture_attempts ORDER BY rowid")
                .fetch_all(state.local_store.pool())
                .await
                .unwrap();
        let sources: Vec<String> = sources
            .iter()
            .map(|(json,)| {
                let value: Value = serde_json::from_str(json).unwrap();
                value["command_source"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string()
            })
            .collect();
        assert_eq!(sources, vec!["relay".to_string(), "operator".to_string()]);
        assert!(state.equipment.gate().is_idle());
        crate::api::equipment_tests::cleanup(directory, state).await;
    }

    /// §2.9: the e-stop is always accepted. A malformed optional field
    /// never stops it from running over the relay, as on the direct route.
    #[tokio::test]
    async fn a_relayed_estop_runs_whatever_its_optional_field_holds() {
        let (directory, state) = crate::api::equipment_tests::state().await;
        for payload in [
            json!({"abort_exposure": "true"}),
            json!({"abort_exposure": null}),
            json!({"abort_exposure": 1}),
            json!("not an object"),
            Value::Null,
            json!({}),
        ] {
            let stopped = execute(
                state.clone(),
                &RelayCommand {
                    command_id: "relay-estop".into(),
                    action: "equipment.estop".into(),
                    payload: payload.clone(),
                },
            )
            .await
            .unwrap_or_else(|err| panic!("{payload}: {err}"));
            assert_eq!(stopped["status"], "stopped", "{payload}: {stopped}");
            assert!(stopped["command_id"].is_string(), "{payload}: {stopped}");
            assert!(stopped["exposure"].is_null(), "{payload}: {stopped}");
        }
        assert!(estop_body(&json!({"abort_exposure": true})).abort_exposure);
        crate::api::equipment_tests::cleanup(directory, state).await;
    }

    #[test]
    fn every_new_equipment_route_is_registered_once() {
        for name in [
            "mount.descriptor",
            "mount.park",
            "mount.unpark",
            "mount.find_home",
            "mount.center",
            "mount.set_tracking_rate",
            "mount.set_guide_rates",
            "mount.site_sync",
            "focuser.descriptor",
            "focuser.set_tempcomp",
            "focuser.sync",
            "focuser.set_position",
            "filter_wheel.descriptor",
            "filter_wheel.change",
            "filter_wheel.import_seed",
            "guider.descriptor",
            "guider.start",
            "guider.stop",
            "guider.pause",
            "guider.resume",
            "guider.dither",
            "weather.status",
            "safety.status",
            "equipment.resume_blocks",
            "equipment.clear_resume_block",
            "equipment.events",
        ] {
            assert!(spec(name).is_some(), "{name} is not registered");
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

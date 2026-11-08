//! Interop and acceptance tests (spec §5.3 WP1B-6, §5.7 Tier 1, O9, O10,
//! O11 l). Owner: WP1B-6.
//!
//! **CI-runnable** (every OS, no external process, loopback only):
//!
//! - The scenario
//!   ([`the_tier1_rig_runs_end_to_end_through_the_daemon_routes`]) is one
//!   observing session through the daemon's real HTTP routes (the `main.rs`
//!   route builders, served on `127.0.0.1:0`), against the WP1A-4 kinematic
//!   mount in its ESP32-like profile and the WP1A-5 Platform 6 focuser
//!   (temperature probe, absolute, firmware-owned backlash), both behind real
//!   Alpaca HTTP through the `testing/alpaca_server.rs` adapter (O11 l); the
//!   PHD2 mock; and the filter-wheel, weather and safety simulators. It
//!   checks that connecting moves nothing, a J2000 slew settles, tracking,
//!   a focuser move, a filter change, guiding and a dither through the mock,
//!   the D20 guider stop before a slew, unsafe weather driving the safety
//!   policy in order, abort and e-stop reaching the hardware, resume blocks
//!   refusing an automatic capture, park verification (success and
//!   failure), and that every v6 ground-truth table the session feeds gets
//!   rows (`telemetry_log` is reported, not required: nothing writes the H19
//!   series yet).
//! - The conformance run
//!   ([`native_alpaca_drivers_conform_against_the_served_simulators`]) holds
//!   the native Alpaca telescope and focuser drivers to the WP1A-2
//!   conformance suites, over HTTP to the same simulators.
//!
//! **Local only** (`#[ignore]`, named `interop_*`; run with
//! `cargo test --locked -- --ignored interop`). Each starts its external
//! process on free ports and kills it on every exit path, including a
//! failed assertion ([`ChildGuard`]). Without its environment variable a
//! test prints why and passes, so `--ignored` never needs the fixtures.
//!
//! - `ACHERNAR_MOUNT_SIM=<FW/host_sim/build/achernar-mount-sim>` (unix): the
//!   ESP32 firmware's host simulator, started with
//!   `--alpaca-port 0 --discovery-port 0 --lx200-port 0` plus `--lat`,
//!   `--lon`, `--elev` and `--unique-id`; the port comes from its banner.
//!   The native Alpaca mount runs end to end through the daemon (DEC
//!   present), the DEC-absent firmware refuses an out-of-window GoTo
//!   cleanly, and the telescope conformance suite passes against it.
//! - `ACHERNAR_OMNISIM=<launcher honouring OMNISIM_PORT>` (O9): ASCOM
//!   OmniSim on a free port. Every native Alpaca driver (Telescope,
//!   Focuser, FilterWheel, ObservingConditions, SafetyMonitor) passes its
//!   conformance suite against its OmniSim device, the daemon drives each of
//!   them through its routes, and Rotator, CoverCalibrator, Switch and Dome,
//!   which have no native driver until Tier 2 (WP2-5), get a client-level
//!   connect and read while the daemon refuses them with 501. OmniSim runs
//!   are serialised: every instance also binds the discovery port UDP 32227.
//!
//! Nothing here records `qualified` (D21): simulator success is not rig
//! acceptance. `achernar-daemon/EQUIPMENT_RUNBOOK.md` is the procedure that
//! does.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::api::camera::equipment::{
    finalise, AcquisitionPath, AssemblyInput, CaptureHooks, ExposureInterval, FinalisePlan,
};
use crate::api::equipment as eq;
use crate::config::{AchernarConfig, CameraConfig, EquipmentConfigs, FocusConfig};
use crate::equipment::alpaca::client::{AlpacaClient, ClientConfig, ClientId};
use crate::equipment::alpaca::discovery::{parse_option, AlpacaAddress, AlpacaOption};
use crate::equipment::alpaca::filterwheel::AlpacaFilterWheel;
use crate::equipment::alpaca::focuser::AlpacaFocuser;
use crate::equipment::alpaca::observingconditions::{AlpacaObservingConditions, WeatherField};
use crate::equipment::alpaca::safetymonitor::AlpacaSafetyMonitor;
use crate::equipment::alpaca::telescope::AlpacaTelescope;
use crate::equipment::clock::SystemClock;
use crate::equipment::events::{PointingPurpose, ProfileUpdate};
use crate::equipment::health_bridge::HealthPolicy;
use crate::equipment::mount::MountDriverOptions;
use crate::equipment::safety::policy::PolicyAction;
use crate::equipment::sim::focuser::{SimFocuser, SimFocuserConfig};
use crate::equipment::sim::mount::{SimFaults, SimMount, SimMountProfile};
use crate::equipment::snapshot::SafetyLevel;
use crate::equipment::spine::WorkerTuning;
use crate::equipment::testing::alpaca_server::{ScriptedDevice, TestAlpacaServer};
use crate::equipment::testing::conformance::{
    assert_conforms, disconnect_suite, filter_wheel_suite, focuser_suite, mount_suite,
    safety_suite, weather_suite, Violation,
};
use crate::equipment::testing::phd2_mock::Phd2Mock;
use crate::equipment::traits::{AlpacaTransport, Clock, TimeoutClass};
use crate::equipment::types::{
    CommandSource, DeviceIdentity, DeviceKind, DriverId, EqCoord, EqFrame,
};
use crate::equipment::worker::Cadence;
use crate::state::{AppState, CameraRuntimeState};
use crate::storage::{
    capture_store::CaptureStore, frames_repo::FrameStore, local_store::LocalStore,
};

// ---------------------------------------------------------------------------
// The daemon under test
// ---------------------------------------------------------------------------

/// Generous bound for anything that waits on a device in real time, so the
/// scenario holds on a slow Windows CI runner.
const WAIT: Duration = Duration::from_secs(30);

/// The daemon's `/api/v1` router, built from the same handlers and the same
/// `main.rs` route builders the binary serves. The COC routes are left out
/// (they belong to the user's COC work and take no equipment).
fn daemon_router(state: AppState) -> axum::Router {
    use crate::api;
    use axum::routing::{get, post};
    let api_v1 = axum::Router::new()
        .route("/status", get(api::get_status))
        .route("/drivers", get(api::get_driver_capabilities))
        .route("/commands", get(api::get_commands))
        .route("/discover", post(api::discover_devices))
        .route("/connect", post(api::connect_equipment))
        .route("/disconnect", post(api::disconnect_equipment))
        .nest("/camera", crate::camera_routes())
        .nest("/mount", crate::mount_routes())
        .nest("/focuser", crate::focuser_routes())
        .nest("/filter-wheel", crate::filter_wheel_routes())
        .nest("/guider", crate::guider_routes())
        .route("/weather", get(api::weather::status))
        .route("/safety", get(api::safety::status))
        .nest("/equipment", crate::equipment_routes())
        .nest("/focus", crate::focus_routes())
        .nest("/platesolve", crate::platesolve_routes());
    axum::Router::new()
        .nest("/api/v1", api_v1)
        .with_state(state)
}

/// The equipment configuration of the scenario: a known site, quick
/// settles, and the global backlash defaults switched on (so a focuser that
/// owns its backlash visibly overrides them, D17).
fn scenario_configs(lat_deg: f64, lon_deg: f64, elev_m: f64) -> EquipmentConfigs {
    let mut configs = AchernarConfig::default().equipment_configs();
    configs.site.latitude_deg = Some(lat_deg);
    configs.site.longitude_deg = Some(lon_deg);
    configs.site.elevation_m = Some(elev_m);
    configs.mount.settle_s = 0.2;
    configs.focus_safety.settle_ms = 0;
    configs.filter_wheel.settle_ms = 0;
    configs.focus_backlash.enabled = true;
    configs.focus_backlash.compensation_steps = 80;
    configs
}

/// One daemon: `AppState` with a live hub, recorder and profile store over a
/// fresh database (`with_equipment`, as `main.rs` builds it), its router on a
/// loopback port, and a blocking HTTP client run off the async workers.
struct Daemon {
    directory: PathBuf,
    state: AppState,
    base: String,
    agent: ureq::Agent,
    server: tokio::task::JoinHandle<()>,
}

impl Daemon {
    async fn start(configs: EquipmentConfigs) -> Self {
        let directory = std::env::temp_dir().join(format!("achernar-interop-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let store = LocalStore::open(directory.join("state.db").to_str().unwrap())
            .await
            .unwrap();
        let mut focus = FocusConfig::default();
        focus.models.quality_model_path =
            Some(directory.join("absent-fqe.onnx").to_string_lossy().into());
        focus.safety = configs.focus_safety.clone();
        focus.backlash = configs.focus_backlash.clone();
        let site = configs.site.site().unwrap();
        let state = AppState::new(
            focus,
            CameraConfig::default(),
            store.clone(),
            FrameStore::new(directory.join("previews")),
            CaptureStore::new(directory.join("captures")),
            "interop-rig".into(),
            directory.join("data"),
            None,
        )
        .with_equipment(&configs, site, Some(store));
        assert!(!state.equipment.is_inert(), "the hub records (D11)");
        state.equipment_ctx.set_tuning(WorkerTuning {
            cadence: Some(Cadence {
                idle: Duration::from_millis(100),
                active: Duration::from_millis(50),
            }),
            policy: Some(HealthPolicy::default()),
            sim_mount_profile: None,
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let app = daemon_router(state.clone());
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        let agent = ureq::Agent::config_builder()
            .proxy(None)
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(60)))
            .build()
            .new_agent();
        Self {
            directory,
            state,
            base,
            agent,
            server,
        }
    }

    /// One HTTP request; the reply's status and JSON body (a non-JSON body
    /// comes back as a string).
    async fn call(&self, method: &'static str, path: &str, body: Option<Value>) -> (u16, Value) {
        let url = format!("{}/api/v1{path}", self.base);
        let agent = self.agent.clone();
        tokio::task::spawn_blocking(move || {
            let result = match (method, body) {
                ("GET", _) => agent.get(&url).call(),
                (_, Some(body)) => agent
                    .post(&url)
                    .content_type("application/json")
                    .send(body.to_string()),
                (_, None) => agent.post(&url).send_empty(),
            };
            let mut response = result.unwrap_or_else(|err| panic!("{method} {url}: {err}"));
            let status = response.status().as_u16();
            let text = response.body_mut().read_to_string().unwrap_or_default();
            let value = serde_json::from_str(&text).unwrap_or(Value::String(text));
            (status, value)
        })
        .await
        .expect("the HTTP client thread")
    }

    async fn get(&self, path: &str) -> Value {
        let (status, body) = self.call("GET", path, None).await;
        assert_eq!(status, 200, "GET {path}: {body}");
        body
    }

    /// POST with a JSON body; the status and body.
    async fn post(&self, path: &str, body: Value) -> (u16, Value) {
        self.call("POST", path, Some(body)).await
    }

    /// POST that must be accepted (2xx).
    async fn post_ok(&self, path: &str, body: Value) -> Value {
        let (status, reply) = self.post(path, body).await;
        assert!(
            (200..300).contains(&status),
            "POST {path}: {status} {reply}"
        );
        reply
    }

    async fn status(&self) -> Value {
        self.get("/status").await
    }

    /// Poll `/status` until `done` holds.
    async fn wait_status(&self, what: &str, done: impl Fn(&Value) -> bool) -> Value {
        let deadline = Instant::now() + WAIT;
        loop {
            let value = self.status().await;
            if done(&value) {
                return value;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {what}: {value}"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    /// The journal position now: later events have a larger seq.
    fn mark(&self) -> u64 {
        self.state.equipment.bus().next_seq().saturating_sub(1)
    }

    /// Every journal event after `after` (through `GET equipment/events`,
    /// paged), optionally only of `kinds` (comma-separated).
    async fn events_after(&self, after: u64, kinds: &str) -> Vec<Value> {
        let mut out = Vec::new();
        let mut since = after;
        loop {
            let page = self
                .get(&format!(
                    "/equipment/events?since_seq={since}&kinds={kinds}&limit=500"
                ))
                .await;
            let events = page["events"].as_array().cloned().unwrap_or_default();
            if events.is_empty() {
                return out;
            }
            since = page["next_seq"].as_u64().unwrap();
            out.extend(events);
        }
    }

    /// The first `kind` event after `after`.
    async fn wait_event(&self, kind: &str, after: u64) -> Value {
        let deadline = Instant::now() + WAIT;
        loop {
            if let Some(event) = self.events_after(after, kind).await.into_iter().next() {
                return event;
            }
            assert!(
                Instant::now() < deadline,
                "no {kind} event after seq {after}; journal: {:?}",
                names(&self.events_after(after, "").await)
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    /// Wait until no command holds a domain (the last operation resolved);
    /// an operator command on a held domain gets 409.
    async fn wait_idle(&self) {
        let deadline = Instant::now() + WAIT;
        while !self.state.equipment.gate().is_idle() {
            assert!(
                Instant::now() < deadline,
                "an operation never released its domain"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    async fn connect(&self, kind: &str, driver: &str, option: &str) -> Value {
        let body = self
            .post_ok(
                "/connect",
                json!({"equipment_type": kind, "driver": driver, "option": option}),
            )
            .await;
        assert_eq!(body["status"], "connected", "{kind}: {body}");
        // D21: what this layer reports for what it built; only the rig
        // acceptance procedure records `qualified`.
        assert_eq!(
            body["identity"]["qualification"]["state"], "implemented_unqualified",
            "{kind}: {body}"
        );
        body
    }

    /// Store an identity profile before the device connects (an operator's
    /// declaration, D17).
    async fn seed_profile(&self, identity_key: &str, kind: DeviceKind, patch: Value) {
        self.state
            .equipment_ctx
            .profiles()
            .expect("with_equipment opens the profile store")
            .merge(&ProfileUpdate {
                identity_key: identity_key.to_string(),
                kind,
                patch,
                updated_at: Utc::now(),
            })
            .await
            .unwrap();
    }

    /// Everything recorded so far is on disk.
    async fn flush(&self) {
        self.state
            .equipment_ctx
            .recorder()
            .expect("with_equipment starts the recorder")
            .flush(Duration::from_secs(10))
            .await
            .expect("the recorder flushes");
    }

    async fn count(&self, sql: &str) -> i64 {
        sqlx::query_scalar::<_, i64>(sql)
            .fetch_one(self.state.local_store.pool())
            .await
            .unwrap_or_else(|err| panic!("{sql}: {err}"))
    }

    /// Stop every worker, the recorder and the server; remove the files.
    async fn close(self) {
        for kind in eq::AUXILIARY_KINDS {
            let _ = eq::disconnect_device(&self.state, kind, true).await;
        }
        self.server.abort();
        let hub = Arc::clone(&self.state.equipment);
        let _ = tokio::task::spawn_blocking(move || hub.shutdown(Duration::from_secs(2))).await;
        if let Some(recorder) = self.state.equipment_ctx.recorder() {
            let _ = recorder.shutdown(Utc::now(), Duration::from_secs(2)).await;
        }
        let pool = self.state.local_store.pool().clone();
        drop(self.state);
        pool.close().await;
        std::fs::remove_dir_all(&self.directory).ok();
    }
}

fn names(events: &[Value]) -> Vec<String> {
    events
        .iter()
        .map(|e| e["kind"].as_str().unwrap_or_default().to_string())
        .collect()
}

/// Assert that `events` contains the names of `order` as a subsequence.
fn assert_in_order(events: &[Value], order: &[&str]) {
    let all = names(events);
    let mut from = 0;
    for name in order {
        match all[from..].iter().position(|e| e == name) {
            Some(offset) => from += offset + 1,
            None => panic!("expected {order:?} in order; no {name} after #{from} in {all:?}"),
        }
    }
}

/// Angular separation in arcminutes.
fn separation_arcmin(ra1_h: f64, dec1: f64, ra2_h: f64, dec2: f64) -> f64 {
    let (a1, d1, a2, d2) = (
        (ra1_h * 15.0).to_radians(),
        dec1.to_radians(),
        (ra2_h * 15.0).to_radians(),
        dec2.to_radians(),
    );
    let cos = d1.sin() * d2.sin() + d1.cos() * d2.cos() * (a1 - a2).cos();
    cos.clamp(-1.0, 1.0).acos().to_degrees() * 60.0
}

fn unix(at: DateTime<Utc>) -> f64 {
    at.timestamp_micros() as f64 / 1e6
}

/// A J2000 target `ha_h` hours west of the meridian at `dec_deg`, from the
/// mount's reported LST (the J2000/JNow difference is irrelevant for the
/// hour angle at this precision).
fn target_from(status: &Value, ha_h: f64, dec_deg: f64) -> (f64, f64) {
    let lst = status["mount"]["lst_h"].as_f64().expect("the mount's LST");
    ((lst - ha_h).rem_euclid(24.0), dec_deg)
}

// ---------------------------------------------------------------------------
// CI: the Tier 1 rig scenario (§5.7 items 2-8)
// ---------------------------------------------------------------------------

const MOUNT_UID: &str = "interop-esp32-like-mount";
const FOCUSER_UID: &str = "interop-platform6-focuser";

/// The simulator's mechanical position: (hour angle h, Dec deg) at the
/// scenario site (longitude -75).
fn mechanical(control: &crate::equipment::sim::mount::SimMountControl) -> (f64, f64) {
    let (ra, dec) = control.position();
    let lst = crate::equipment::mount::limits::lst_hours(Utc::now(), -75.0);
    ((lst - ra).rem_euclid(24.0), dec)
}

/// The ESP32-like simulator profile, starting parked like the firmware
/// (power-on at the park marks), slewing fast enough for CI.
fn esp32_like_parked() -> SimMountProfile {
    let mut profile = SimMountProfile::esp32_like();
    profile.start_parked = true;
    profile.slew_rate_deg_s = 30.0;
    profile
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_tier1_rig_runs_end_to_end_through_the_daemon_routes() {
    let clock = SystemClock::shared();
    let mount_sim = SimMount::new(esp32_like_parked(), Arc::clone(&clock));
    let mount_control = mount_sim.control();
    let focuser_sim = SimFocuser::new(
        SimFocuserConfig::platform6_firmware_backlash(),
        Arc::clone(&clock),
    );
    let focuser_control = focuser_sim.control();
    let server = TestAlpacaServer::start(vec![
        ScriptedDevice::sim_mount(0, mount_sim).unique_id(MOUNT_UID),
        ScriptedDevice::sim_focuser(0, focuser_sim).unique_id(FOCUSER_UID),
    ]);
    let alpaca = format!("{}#device=0", server.base_url());
    let phd2 = Phd2Mock::start().await;
    let daemon = Daemon::start(scenario_configs(40.0, -75.0, 100.0)).await;
    let put_members = |device_type: &str| -> Vec<String> {
        server
            .requests()
            .into_iter()
            .filter(|r| {
                r.device_type == device_type
                    && r.verb == crate::equipment::alpaca::errors::Verb::Put
            })
            .map(|r| r.member)
            .collect()
    };

    // ---- 1. The default configuration opens nothing (D13) ----
    let status = daemon.status().await;
    assert_eq!(status["connected"], false);
    for key in ["mount", "focuser", "filter_wheel", "guiding", "weather"] {
        assert!(status[key].is_null(), "{key} before any connect: {status}");
    }
    assert!(daemon.get("/drivers").await["mount"].is_array());
    assert!(server.requests().is_empty(), "nothing touched the devices");

    // ---- 2. Connect everything; nothing moves (D16) ----
    // The focuser declares that its firmware owns backlash and that its
    // probe reads ambient air: identity-profile values (D17), not TOML. The
    // focuser session reads them at the top level of the profile document,
    // while `profile.rs` documents a `focuser` section (change note filed);
    // the declaration is written under both, as the runbook does.
    let declaration = json!({"backlash_owner": "firmware", "temperature_role": "ambient"});
    daemon
        .seed_profile(
            &format!("alpaca:{FOCUSER_UID}"),
            DeviceKind::Focuser,
            json!({
                "backlash_owner": "firmware",
                "temperature_role": "ambient",
                "focuser": declaration,
            }),
        )
        .await;
    let position_before = mechanical(&mount_control);
    let body = daemon.connect("mount", "alpaca", &alpaca).await;
    let connected_at = Instant::now();
    assert_eq!(body["identity"]["driver"], "alpaca");
    assert_eq!(body["identity"]["unique_id"], MOUNT_UID);
    // §5.7 item 2: every mount field within 2 s of the connect.
    let value = daemon
        .wait_status("a polled Alpaca mount", |v| {
            v["mount"]["lst_h"].is_number() && v["mount"]["parked"].is_boolean()
        })
        .await;
    let lag = connected_at.elapsed();
    assert!(
        lag < Duration::from_secs(2),
        "status complete after {lag:?}"
    );
    let mount = &value["mount"];
    for key in crate::equipment::mount::STATUS_KEYS {
        assert!(mount.get(*key).is_some(), "mount block lacks {key}");
    }
    assert_eq!(mount["driver"], "alpaca");
    assert_eq!(mount["frame"], "jnow_apparent");
    assert_eq!(mount["alignment_mode"], "polar");
    assert_eq!(mount["can_flip"], false, "a Polar mount never flips (D14)");
    assert_eq!(mount["pier_side"], "east", "pier_side is a string");
    assert_eq!(mount["parked"], true);
    assert_eq!(mount["tracking"], false);
    assert_eq!(mount["slewing"], false);
    assert_eq!(mount["state"], "ready");
    let descriptor = daemon.get("/mount").await;
    assert_eq!(descriptor["can_flip"], false);

    let body = daemon.connect("focuser", "alpaca", &alpaca).await;
    assert_eq!(body["identity"]["unique_id"], FOCUSER_UID);
    let value = daemon
        .wait_status("a polled Platform 6 focuser", |v| {
            v["focuser"]["position"] == 25_000 && v["focuser"]["temperature"].is_number()
        })
        .await;
    let focuser = &value["focuser"];
    assert_eq!(focuser["absolute"], true);
    assert_eq!(focuser["temperature"], 12.0);
    assert_eq!(focuser["temperature_source"], "ambient");
    assert_eq!(focuser["backlash_owner"], "firmware");
    assert_eq!(
        focuser["backlash_model"]["model"], "none",
        "the profile wins over [focus.backlash] (D17)"
    );

    daemon.connect("filter_wheel", "sim", "").await;
    daemon.connect("weather", "sim", "").await;
    daemon.connect("safety_monitor", "sim", "").await;
    daemon
        .connect("guider", "phd2", &format!("127.0.0.1:{}", phd2.port()))
        .await;
    let value = daemon
        .wait_status("every device ready", |v| {
            [
                "mount",
                "focuser",
                "filter_wheel",
                "weather",
                "safety_monitor",
                "guider",
            ]
            .iter()
            .all(|k| v["device_health"][*k]["state"] == "ready")
                && v["filter_wheel"]["slot"].is_number()
                && v["guiding"].is_object()
        })
        .await;
    assert_eq!(value["filter_wheel"]["verified"], true);
    let sims = daemon.state.equipment_ctx.sims();
    let wheel = sims.filter_wheel.clone().expect("the sim wheel's control");
    let weather = sims.weather.clone().expect("the sim weather's control");
    let safety = sims.safety.clone().expect("the sim monitor's control");
    // Connecting moved nothing: only `connect` (or `Connected=true`) was
    // written to the Alpaca devices, the simulators received no motion
    // command, and PHD2 was not told to connect its equipment.
    assert_eq!(put_members("telescope"), vec!["connect".to_string()]);
    assert_eq!(put_members("focuser"), vec!["connected".to_string()]);
    assert_eq!(mount_control.calls(), vec!["connect".to_string()]);
    assert!(focuser_control.moves().is_empty());
    assert!(wheel.commanded().is_empty());
    assert_eq!(phd2.method_count("set_connected"), 0, "D16");
    // A parked mount with tracking off keeps its hour angle (its RA drifts
    // with the sky); compare the mechanical position.
    let (ha_before, dec_before) = position_before;
    let (ha_after, dec_after) = mechanical(&mount_control);
    assert!(
        (ha_after - ha_before).abs() < 1e-3 && (dec_after - dec_before).abs() < 1e-9,
        "the mount moved on connect: {:?} -> {:?}",
        position_before,
        (ha_after, dec_after)
    );
    let safety_status = daemon.get("/safety").await;
    assert_eq!(safety_status["state"], "safe", "{safety_status}");

    // ---- 3. Unpark, then a J2000 slew that settles; tracking ----
    let mark = daemon.mark();
    daemon.post_ok("/mount/unpark", json!({})).await;
    daemon.wait_event("unparked", mark).await;
    daemon.wait_idle().await;
    let status = daemon.status().await;
    let (ra_j2000, dec_j2000) = target_from(&status, 1.0, 50.0);
    let mark = daemon.mark();
    let accepted = daemon
        .post_ok(
            "/mount/slew",
            json!({"ra": ra_j2000, "dec": dec_j2000, "frame": "j2000"}),
        )
        .await;
    assert_eq!(accepted["status"], "slewing");
    assert!(accepted["command_id"].is_string());
    daemon.wait_event("slew_settled", mark).await;
    daemon.wait_idle().await;
    let events = daemon.events_after(mark, "").await;
    assert_in_order(
        &events,
        &[
            "slew_commanded",
            "slew_started",
            "slew_arrived",
            "slew_settled",
        ],
    );
    // M-c step 4: tracking was off after the unpark, so the slew turned it
    // on first and recorded why.
    assert!(
        events
            .iter()
            .any(|e| e["kind"] == "tracking_changed"
                && e["payload"]["reason"] == "slew_precondition"),
        "{:?}",
        names(&events)
    );
    // The J2000 target was converted to the mount's JNow frame (D10): the
    // native RA/Dec written to the device are not the J2000 numbers.
    let slew_put = server
        .requests()
        .into_iter()
        .rev()
        .find(|r| r.member == "slewtocoordinatesasync")
        .expect("the GoTo reached the device");
    let param = |key: &str| -> f64 {
        crate::equipment::alpaca::form::find(&slew_put.params, key, true)
            .and_then(|v| v.parse().ok())
            .unwrap()
    };
    let native_shift = separation_arcmin(
        param("RightAscension"),
        param("Declination"),
        ra_j2000,
        dec_j2000,
    );
    assert!(
        native_shift > 5.0,
        "J2000 sent unconverted ({native_shift:.2}')"
    );
    let value = daemon
        .wait_status("tracking at the target", |v| v["mount"]["tracking"] == true)
        .await;
    let arrived = separation_arcmin(
        value["mount"]["ra_j2000"].as_f64().unwrap(),
        value["mount"]["dec_j2000"].as_f64().unwrap(),
        ra_j2000,
        dec_j2000,
    );
    assert!(arrived < 2.0, "arrived {arrived:.2}' from the J2000 target");
    daemon
        .post_ok("/mount/tracking", json!({"enabled": false}))
        .await;
    daemon
        .wait_status("tracking off", |v| v["mount"]["tracking"] == false)
        .await;
    daemon.wait_idle().await;
    daemon
        .post_ok("/mount/tracking", json!({"enabled": true}))
        .await;
    daemon
        .wait_status("tracking on", |v| v["mount"]["tracking"] == true)
        .await;
    daemon.wait_idle().await;

    // ---- 4. A focuser move: one device move, read back (F-a, F-e) ----
    let mark = daemon.mark();
    let accepted = daemon
        .post_ok("/focuser/moveto", json!({"position": 25_400}))
        .await;
    assert_eq!(accepted["status"], "moving");
    daemon
        .wait_status("the focuser at 25400", |v| {
            v["focuser"]["position"] == 25_400 && v["focuser"]["active_op"].is_null()
        })
        .await;
    daemon.wait_idle().await;
    daemon.wait_event("focuser_settled", mark).await;
    let moves: Vec<String> = server
        .requests()
        .into_iter()
        .filter(|r| r.device_type == "focuser" && r.member == "move")
        .map(|r| {
            crate::equipment::alpaca::form::find(&r.params, "Position", true)
                .unwrap_or_default()
                .to_string()
        })
        .collect();
    // Firmware-owned backlash: no overshoot or compensation move even though
    // [focus.backlash] is enabled with 80 steps.
    assert_eq!(moves, vec!["25400".to_string()]);
    assert_eq!(focuser_control.raw_position(), 25_400);

    // ---- 5. A filter change through the handshake (W-c) ----
    let mark = daemon.mark();
    let accepted = daemon
        .post_ok("/filter-wheel/change", json!({"slot": 2}))
        .await;
    assert_eq!(accepted["status"], "changing");
    daemon
        .wait_status("slot 2 verified", |v| {
            v["filter_wheel"]["slot"] == 2 && v["filter_wheel"]["verified"] == true
        })
        .await;
    daemon.wait_event("filter_change_ended", mark).await;
    daemon.wait_idle().await;
    assert_in_order(
        &daemon.events_after(mark, "").await,
        &[
            "filter_change_requested",
            "filter_arrived",
            "filter_change_ended",
        ],
    );
    assert_eq!(wheel.commanded(), vec![2]);

    // ---- 6. Guiding, guide steps and a dither through the PHD2 mock ----
    let mark = daemon.mark();
    daemon.post_ok("/guider/start", json!({})).await;
    daemon
        .wait_status("guiding", |v| v["guiding"]["state"] == "guiding")
        .await;
    daemon.wait_idle().await;
    for frame in 1..=8 {
        phd2.send_event(Phd2Mock::guide_step(frame, 0.3, -0.2));
    }
    daemon
        .wait_status("guide RMS", |v| v["guiding"]["rms_total"].is_number())
        .await;
    let dither_mark = daemon.mark();
    let accepted = daemon.post_ok("/guider/dither", json!({})).await;
    assert!(accepted["command_id"].is_string(), "{accepted}");
    daemon.wait_event("settle_done", dither_mark).await;
    daemon.wait_idle().await;
    assert_in_order(
        &daemon.events_after(mark, "").await,
        &["guiding_started", "settle_done", "dithered", "settle_done"],
    );
    assert_eq!(phd2.method_count("dither"), 1);
    // D20: Achernar never pulses while PHD2 guides.
    let (code, refused) = daemon
        .post(
            "/mount/pulse-guide",
            json!({"direction": "north", "duration_ms": 300}),
        )
        .await;
    assert_eq!(code, 422, "{refused}");
    assert!(!mount_control
        .calls()
        .iter()
        .any(|c| c.starts_with("pulse_guide")));

    // ---- 7. One exposure's ground truth from the live rings (H1, H9, H12) ----
    // The camera stays on OLS (O1) and no CI fixture drives an OLS capture,
    // so the capture path's own hooks stand in for the camera: the window is
    // assembled from the rings the real devices filled, finalised (D7), and
    // a pointing sample is built from it with a known solution standing in
    // for ASTAP.
    let frame_id = exposure_with_ground_truth(&daemon, &phd2, ra_j2000, dec_j2000).await;

    // ---- 8. D20: a slew stops guiding first; then a pulse runs ----
    let stops = phd2.method_count("stop_capture");
    let status = daemon.status().await;
    let (ra2, dec2) = target_from(&status, 0.8, 52.0);
    let mark = daemon.mark();
    let accepted = daemon
        .post_ok(
            "/mount/slew",
            json!({"ra": ra2, "dec": dec2, "frame": "j2000"}),
        )
        .await;
    assert_eq!(accepted["was_guiding"], true, "{accepted}");
    assert!(phd2.method_count("stop_capture") > stops);
    daemon.wait_event("slew_settled", mark).await;
    daemon.wait_idle().await;
    daemon
        .wait_status("PHD2 stopped", |v| v["guiding"]["state"] == "stopped")
        .await;
    let mark = daemon.mark();
    let accepted = daemon
        .post_ok(
            "/mount/pulse-guide",
            json!({"direction": "north", "duration_ms": 300}),
        )
        .await;
    assert!(accepted["command_id"].is_string(), "{accepted}");
    daemon.wait_event("pulse_guide", mark).await;
    daemon.wait_idle().await;
    assert!(mount_control
        .calls()
        .iter()
        .any(|c| c == "pulse_guide:North:300"));
    assert!(put_members("telescope").iter().any(|m| m == "pulseguide"));

    // ---- 9. Unsafe weather drives the safety policy (§3.10, O11 h) ----
    // A GoTo that never arrives keeps the mount slewing, and PHD2 guides,
    // so every policy step has something to do.
    mount_control.update_faults(|f| f.slew_hangs = true);
    let status = daemon.status().await;
    let (ra3, dec3) = target_from(&status, -2.0, 20.0);
    daemon
        .post_ok(
            "/mount/slew",
            json!({"ra": ra3, "dec": dec3, "frame": "j2000"}),
        )
        .await;
    daemon
        .wait_status("slewing", |v| v["mount"]["slewing"] == true)
        .await;
    daemon.post_ok("/guider/start", json!({})).await;
    daemon
        .wait_status("guiding again", |v| v["guiding"]["state"] == "guiding")
        .await;
    let stops = phd2.method_count("stop_capture");
    let aborts_before = put_members("telescope")
        .iter()
        .filter(|m| *m == "abortslew")
        .count();
    let mark = daemon.mark();
    weather.set(WeatherField::RainRate, 3.0);
    weather.set(WeatherField::CloudCover, 100.0);
    weather.set(WeatherField::Humidity, 97.0);
    safety.set_detail(Some("rain"));
    safety.set_level(SafetyLevel::Unsafe);
    daemon.wait_event("safety_changed", mark).await;
    daemon.wait_event("parked", mark).await;
    let supervisor = daemon
        .state
        .equipment_ctx
        .safety_supervisor()
        .expect("the supervisor started with the monitor");
    let deadline = Instant::now() + WAIT;
    while supervisor.policy_running() || supervisor.runs().is_empty() {
        assert!(Instant::now() < deadline, "the policy never finished");
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    let run = supervisor.runs().last().cloned().unwrap();
    assert_eq!(
        run.done(),
        vec![
            PolicyAction::RaiseResumeBlock,
            PolicyAction::StopGuiding,
            PolicyAction::AbortSlew,
            PolicyAction::Park,
        ],
        "§5.7 item 6 order: {:?}",
        run.steps
    );
    assert!(phd2.method_count("stop_capture") > stops, "guiding stopped");
    let telescope_puts = put_members("telescope");
    let first_abort = telescope_puts
        .iter()
        .enumerate()
        .filter(|(_, m)| *m == "abortslew")
        .nth(aborts_before)
        .map(|(i, _)| i)
        .expect("the abort reached the mount");
    let park_put = telescope_puts
        .iter()
        .rposition(|m| m == "park")
        .expect("the park reached the mount");
    assert!(first_abort < park_put, "{telescope_puts:?}");
    let value = daemon
        .wait_status("parked and blocked", |v| {
            v["mount"]["parked"] == true
                && v["resume_blocks"]
                    .as_array()
                    .is_some_and(|b| b.iter().any(|b| b["kind"]["kind"] == "safety_unsafe"))
        })
        .await;
    let weather_now = daemon.get("/weather").await;
    assert_eq!(
        weather_now["fields"]["rain_rate_mm_h"]["value"], 3.0,
        "{weather_now}"
    );
    assert!(value["mount"]["parked"] == true);
    assert_eq!(daemon.get("/safety").await["state"], "unsafe");
    mount_control.update_faults(|f| f.slew_hangs = false);
    daemon.wait_idle().await;

    // ---- 10. Resume blocks refuse automatic work (H22) ----
    let automatic_capture = json!({
        "exposure": 1.0,
        "capture_type": "light",
        "source": {"kind": "trigger", "rule": "interop"},
    });
    let (code, refused) = daemon
        .post("/camera/capture", automatic_capture.clone())
        .await;
    assert_eq!(code, 423, "{refused}");
    let blocks = daemon.get("/equipment/resume-blocks").await;
    let block_id = |kind: &str| -> String {
        blocks["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|b| b["kind"]["kind"] == kind)
            .and_then(|b| b["id"].as_str())
            .unwrap_or_else(|| panic!("no {kind} block in {blocks}"))
            .to_string()
    };
    let unsafe_block = block_id("safety_unsafe");
    // A safety block clears only on a confirmed safe reading.
    let (code, refused) = daemon
        .post(
            "/equipment/resume-blocks/clear",
            json!({"id": unsafe_block}),
        )
        .await;
    assert_eq!(code, 409, "{refused}");
    safety.set_level(SafetyLevel::Safe);
    weather.set(WeatherField::RainRate, 0.0);
    let deadline = Instant::now() + WAIT;
    while daemon.get("/safety").await["state"] != "safe" {
        assert!(Instant::now() < deadline, "safety never read safe again");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    // The operator's manual resume (Tier 1) once safety reads safe. The
    // clear route refuses every safety block today (it calls the generic
    // clear, not `SafetySupervisor::request_resume`; change note filed), so
    // the supervisor's resume stands in until the route reaches it.
    let (code, reply) = daemon
        .post(
            "/equipment/resume-blocks/clear",
            json!({"id": unsafe_block}),
        )
        .await;
    if !(200..300).contains(&code) {
        eprintln!("resume-blocks/clear refused a safety block while safe: {code} {reply}");
        let cleared = supervisor
            .request_resume(&CommandSource::Operator { overriding: false })
            .expect("the operator resumes once safety reads safe");
        assert!(!cleared.is_empty());
    }
    let blocks = daemon.get("/equipment/resume-blocks").await;
    assert!(
        !blocks["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|b| b["kind"]["kind"] == "safety_unsafe"),
        "{blocks}"
    );

    // ---- 11. Abort and e-stop reach the hardware (D4, D19, §2.9) ----
    let mark = daemon.mark();
    daemon.post_ok("/mount/unpark", json!({})).await;
    daemon.wait_event("unparked", mark).await;
    daemon.wait_idle().await;
    mount_control.update_faults(|f| f.slew_hangs = true);
    let status = daemon.status().await;
    let (ra4, dec4) = target_from(&status, -1.5, 30.0);
    daemon
        .post_ok(
            "/mount/slew",
            json!({"ra": ra4, "dec": dec4, "frame": "j2000"}),
        )
        .await;
    daemon
        .wait_status("slewing", |v| v["mount"]["slewing"] == true)
        .await;
    let aborts = put_members("telescope")
        .iter()
        .filter(|m| *m == "abortslew")
        .count();
    let abort_started = Instant::now();
    let aborted = daemon.post_ok("/mount/abort", json!({})).await;
    assert_eq!(aborted["abort_path_ran"], true, "{aborted}");
    daemon
        .wait_status("stopped", |v| v["mount"]["slewing"] == false)
        .await;
    assert!(
        abort_started.elapsed() < Duration::from_secs(2),
        "§5.7: Slewing false within 2 s"
    );
    assert!(
        put_members("telescope")
            .iter()
            .filter(|m| *m == "abortslew")
            .count()
            > aborts,
        "AbortSlew reached the device"
    );
    daemon.wait_idle().await;
    // E-stop with the mount slewing and the focuser stuck mid-move.
    daemon
        .post_ok(
            "/mount/slew",
            json!({"ra": ra4, "dec": dec4, "frame": "j2000"}),
        )
        .await;
    daemon
        .wait_status("slewing", |v| v["mount"]["slewing"] == true)
        .await;
    focuser_control.set_stuck(true);
    daemon
        .post_ok("/focuser/moveto", json!({"position": 25_200}))
        .await;
    daemon
        .wait_status("the focuser moving", |v| v["focuser"]["moving"] == true)
        .await;
    daemon.post_ok("/guider/start", json!({})).await;
    let stops = phd2.method_count("stop_capture");
    server.clear_requests();
    let mark = daemon.mark();
    let report = daemon.post_ok("/equipment/estop", json!({})).await;
    assert!(
        report["elapsed_ms"].as_u64().unwrap() < 2000,
        "every abort within 2 s: {report}"
    );
    let kinds: Vec<&str> = report["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["kind"].as_str().unwrap_or_default())
        .collect();
    for kind in ["mount", "focuser", "guider"] {
        assert!(kinds.contains(&kind), "e-stop reached {kinds:?}");
    }
    daemon.wait_event("estop", mark).await;
    daemon
        .wait_status("everything stopped", |v| {
            v["mount"]["slewing"] == false
                && v["mount"]["tracking"] == false
                && v["focuser"]["moving"] == false
        })
        .await;
    let puts = |device_type: &str| put_members(device_type);
    assert!(puts("telescope").iter().any(|m| m == "abortslew"));
    assert!(
        puts("telescope").iter().any(|m| m == "tracking"),
        "estop_stops_tracking (default)"
    );
    assert!(puts("focuser").iter().any(|m| m == "halt"));
    assert!(phd2.method_count("stop_capture") > stops);
    assert!(!mount_control.slewing());
    assert!(!focuser_control.is_moving());
    focuser_control.set_stuck(false);
    mount_control.update_faults(|f| f.slew_hangs = false);
    // The estop block refuses an automatic capture until the operator
    // clears it.
    let (code, refused) = daemon.post("/camera/capture", automatic_capture).await;
    assert_eq!(code, 423, "{refused}");
    let blocks = daemon.get("/equipment/resume-blocks").await;
    let estop_block = blocks["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["kind"]["kind"] == "estop")
        .and_then(|b| b["id"].as_str())
        .unwrap_or_else(|| panic!("no estop block: {blocks}"))
        .to_string();
    daemon
        .post_ok("/equipment/resume-blocks/clear", json!({"id": estop_block}))
        .await;
    daemon.wait_idle().await;

    // ---- 12. Park verification: verified through AtPark, and a park that
    // never sets AtPark fails (M-f) ----
    let mark = daemon.mark();
    let accepted = daemon.post_ok("/mount/park", json!({})).await;
    assert_eq!(accepted["status"], "parking");
    daemon.wait_event("parked", mark).await;
    daemon.wait_idle().await;
    daemon
        .wait_status("parked", |v| v["mount"]["parked"] == true)
        .await;
    let mark = daemon.mark();
    daemon.post_ok("/mount/unpark", json!({})).await;
    daemon.wait_event("unparked", mark).await;
    daemon.wait_idle().await;
    mount_control.set_faults(SimFaults {
        park_never_sets_atpark: true,
        ..SimFaults::default()
    });
    let mark = daemon.mark();
    daemon.post_ok("/mount/park", json!({})).await;
    let failed = daemon.wait_event("park_failed", mark).await;
    assert!(
        failed["payload"]["reason"]
            .as_str()
            .is_some_and(|r| r.contains("park_not_verified")),
        "{failed}"
    );
    daemon.wait_idle().await;
    assert_eq!(daemon.status().await["mount"]["parked"], false);
    mount_control.set_faults(SimFaults::default());

    // ---- 13. Disconnect; the v6 ground truth is on disk (§4.3) ----
    let body = daemon.post_ok("/disconnect", json!({})).await;
    assert_eq!(body["status"], "disconnected");
    let value = daemon.status().await;
    assert!(
        value["mount"].is_null() && value["focuser"].is_null(),
        "{value}"
    );
    assert_eq!(phd2.method_count("set_connected"), 0, "D16");
    daemon.flush().await;
    let rows = |table: &'static str| {
        let daemon = &daemon;
        async move { daemon.count(&format!("SELECT COUNT(*) FROM {table}")).await }
    };
    // Lifecycle (§2.6 step 10): one row per connection, every one closed.
    assert!(rows("device_connections").await >= 6);
    assert_eq!(
        daemon
            .count("SELECT COUNT(*) FROM device_connections WHERE disconnected_at IS NULL")
            .await,
        0
    );
    assert_eq!(
        daemon
            .count(&format!(
                "SELECT COUNT(*) FROM device_connections WHERE identity_key = 'alpaca:{MOUNT_UID}' \
                 AND qualification = 'implemented_unqualified'"
            ))
            .await,
        1,
        "the mount's identity and qualification are recorded"
    );
    // The journal (D6), without guide steps (they have their own table).
    for kind in [
        "device_connected",
        "slew_settled",
        "tracking_changed",
        "unparked",
        "parked",
        "park_failed",
        "focuser_settled",
        "filter_change_ended",
        "dithered",
        "safety_changed",
        "estop",
        "resume_blocked",
        "resume_cleared",
        "exposure_completed",
        "device_disconnected",
    ] {
        assert!(
            daemon
                .count(&format!(
                    "SELECT COUNT(*) FROM equipment_events WHERE kind = '{kind}'"
                ))
                .await
                >= 1,
            "no {kind} in equipment_events"
        );
    }
    assert_eq!(
        daemon
            .count("SELECT COUNT(*) FROM equipment_events WHERE kind = 'guide_step'")
            .await,
        0
    );
    // H21: every gated command audited, the policy under the safety source.
    assert!(rows("control_audit_logs").await >= 10);
    // F-a: the move with its read-back.
    assert!(
        daemon
            .count("SELECT COUNT(*) FROM focuser_moves WHERE raw_start = 25000 AND raw_end = 25400")
            .await
            >= 1
    );
    // W-c, H9, H10.
    assert!(
        daemon
            .count("SELECT COUNT(*) FROM filter_switch_events WHERE to_slot = 2")
            .await
            >= 1
    );
    assert!(rows("guide_steps").await >= 8);
    assert!(
        daemon
            .count("SELECT COUNT(*) FROM dither_events WHERE settle_status IS NOT NULL")
            .await
            >= 1
    );
    // H1, H9 (per exposure), H12.
    let window =
        format!("SELECT COUNT(*) FROM exposure_device_windows WHERE frame_id = '{frame_id}'");
    assert_eq!(daemon.count(&window).await, 1);
    assert_eq!(
        daemon
            .count(&format!(
                "SELECT COUNT(*) FROM exposure_device_windows WHERE frame_id = '{frame_id}' \
                 AND finalised = 1 AND mount_mid_json IS NOT NULL AND filter_slot = 2 \
                 AND focuser_pos_start = 25400 AND guide_steps_n > 0"
            ))
            .await,
        1,
        "the window carries the mount, wheel, focuser and guide state"
    );
    assert!(
        daemon
            .count(&format!(
                "SELECT COUNT(*) FROM guider_state_log WHERE frame_id = '{frame_id}'"
            ))
            .await
            >= 1
    );
    assert_eq!(
        daemon
            .count(&format!(
                "SELECT COUNT(*) FROM pointing_samples WHERE frame_id = '{frame_id}' \
                 AND mount_frame = 'jnow_apparent' AND pier_side = 'east'"
            ))
            .await,
        1
    );
    // Weather and the focuser probe (§3.9, F-f) and the identity profiles
    // (last used, D17). The 1 Hz `telemetry_log` series (H19) has no
    // production writer yet (change note filed), so it is reported, not
    // required.
    assert!(
        daemon
            .count("SELECT COUNT(*) FROM environment_log WHERE source = 'probe'")
            .await
            >= 1
    );
    let telemetry = rows("telemetry_log").await;
    if telemetry == 0 {
        eprintln!("telemetry_log has no rows: nothing records the H19 series yet");
    }
    assert_eq!(
        daemon
            .count(&format!(
                "SELECT COUNT(*) FROM device_profiles WHERE identity_key IN \
                 ('alpaca:{MOUNT_UID}', 'alpaca:{FOCUSER_UID}') \
                 AND json_extract(profile_json, '$.last_used.driver') = 'alpaca'"
            ))
            .await,
        2
    );
    daemon.close().await;
    drop(phd2);
    drop(server);
}

/// One 2 s exposure while the mount tracks and PHD2 guides, through the
/// capture path's hooks (`api/camera/equipment.rs`): `exposure_started`,
/// the `exposing` busy flag, `exposure_completed{frame_id}`, the window
/// finalised from the rings (D7) with PHD2's steps (G-i, H9), then a
/// pointing sample (H12) for a solve a few arcseconds from the target.
/// Returns the frame id.
async fn exposure_with_ground_truth(
    daemon: &Daemon,
    phd2: &Phd2Mock,
    ra_j2000: f64,
    dec_j2000: f64,
) -> String {
    let state = &daemon.state;
    let hub = Arc::clone(&state.equipment);
    let frame_id = Uuid::new_v4().to_string();
    let capture_id = Uuid::new_v4().to_string();
    let runtime = CameraRuntimeState {
        driver_id: Some("interop".into()),
        ..CameraRuntimeState::default()
    };
    let mut hooks = CaptureHooks::new(Arc::clone(&hub), &runtime, &capture_id, &frame_id);
    let started = Utc::now();
    hooks.exposure_started(AcquisitionPath::Still, 2.0, None);
    for frame in 20..24 {
        phd2.send_event(Phd2Mock::guide_step(frame, 0.2, 0.1));
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    hooks.exposure_ended();
    let ended = Utc::now();
    let interval = ExposureInterval::still(
        unix(started),
        unix(ended),
        2.0,
        hooks.clock_quality(AcquisitionPath::Still),
    );
    sqlx::query("INSERT INTO frames (id, local_path, captured_at) VALUES (?, ?, ?)")
        .bind(&frame_id)
        .bind("interop://hooked-exposure")
        .bind(crate::storage::columns::time(&ended))
        .execute(state.local_store.pool())
        .await
        .unwrap();
    hooks.exposure_completed(Some(&frame_id), &interval);
    let plan = FinalisePlan {
        input: AssemblyInput {
            frame_id: frame_id.clone(),
            client_filter: None,
            blocks_at_start: Vec::new(),
            source: CommandSource::Operator { overriding: false },
            command_id: None,
        },
        interval: interval.clone(),
        finalised_at_commit: false,
    };
    let guide = state.equipment_ctx.guider_windows();
    let deadline = Instant::now() + WAIT;
    while !finalise(&hub, guide.as_ref(), &plan) {
        assert!(Instant::now() < deadline, "the window never finalised");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let outcome = crate::api::plate_solve::record_pointing_sample(
        state,
        crate::api::plate_solve::PointingInput {
            plate_solve_event_id: None,
            frame_id: Some(frame_id.clone()),
            exposure_start: interval.start,
            exposure_end: interval.end,
            solved_j2000: EqCoord::new(ra_j2000 + 0.0005, dec_j2000 - 0.002, EqFrame::J2000),
            purpose: PointingPurpose::Manual,
        },
    )
    .await;
    assert!(outcome.sample_id.is_some(), "{outcome:?}");
    let error = outcome.error_arcmin.expect("a pointing error");
    assert!(error < 3.0, "pointing error {error:.2}'");
    frame_id
}

// ---------------------------------------------------------------------------
// Driver-level helpers
// ---------------------------------------------------------------------------

/// An address, a transport and an abort channel for one Alpaca device.
fn alpaca_parts(
    base: &str,
    device_type: &str,
    number: u32,
) -> (
    AlpacaAddress,
    Arc<dyn AlpacaTransport>,
    Arc<dyn AlpacaTransport>,
) {
    let address = match parse_option(&format!("{base}#device={number}"), false).unwrap() {
        AlpacaOption::Address(address) => address,
        AlpacaOption::Broadcast => unreachable!("an explicit address"),
    };
    let client = AlpacaClient::new(
        &address.base,
        device_type,
        number,
        ClientConfig::new(ClientId::new(4321).unwrap()),
    )
    .unwrap();
    let abort = Arc::new(client.abort_channel()) as Arc<dyn AlpacaTransport>;
    (address, Arc::new(client), abort)
}

fn alpaca_identity(kind: DeviceKind, address: &AlpacaAddress, device_type: &str) -> DeviceIdentity {
    DeviceIdentity::new(
        kind,
        DriverId::Alpaca,
        address.endpoint(device_type),
        address.identity_key(device_type, None),
        format!("Alpaca {device_type}"),
    )
}

fn telescope_driver(base: &str, clock: &Arc<dyn Clock>) -> AlpacaTelescope {
    let (address, transport, abort) = alpaca_parts(base, "telescope", 0);
    AlpacaTelescope::new(
        alpaca_identity(DeviceKind::Mount, &address, "telescope"),
        transport,
        abort,
        Arc::clone(clock),
        MountDriverOptions::default(),
    )
}

fn focuser_driver(base: &str, clock: &Arc<dyn Clock>) -> AlpacaFocuser {
    let (address, transport, abort) = alpaca_parts(base, "focuser", 0);
    AlpacaFocuser::new(address.base, 0, transport, abort, Arc::clone(clock))
}

#[test]
fn native_alpaca_drivers_conform_against_the_served_simulators() {
    // Blocking drivers on a plain test thread (D3), against the WP1A-4/5
    // simulators behind HTTP (O11 l).
    let clock = SystemClock::shared();
    let mut profile = SimMountProfile::esp32_like();
    profile.slew_rate_deg_s = 30.0;
    let server = TestAlpacaServer::start(vec![
        ScriptedDevice::sim_mount(0, SimMount::new(profile, Arc::clone(&clock)))
            .unique_id(MOUNT_UID),
        ScriptedDevice::sim_focuser(
            0,
            SimFocuser::new(
                SimFocuserConfig::platform6_firmware_backlash(),
                Arc::clone(&clock),
            ),
        ),
    ]);
    let mut telescope = telescope_driver(&server.base_url(), &clock);
    assert_conforms(
        "Alpaca telescope vs the ESP32-like simulator",
        &mount_suite(&mut telescope, clock.as_ref()),
    );
    assert_conforms(
        "Alpaca telescope disconnect",
        &disconnect_suite(&mut telescope),
    );
    let mut focuser = focuser_driver(&server.base_url(), &clock);
    assert_conforms(
        "Alpaca focuser vs the Platform 6 simulator",
        &focuser_suite(&mut focuser, clock.as_ref()),
    );
    assert_conforms("Alpaca focuser disconnect", &disconnect_suite(&mut focuser));
}

// ---------------------------------------------------------------------------
// Local-only fixtures: external processes
// ---------------------------------------------------------------------------

/// The value of a fixture variable naming an existing file, or `None` with
/// the reason printed (the test then passes without running).
fn fixture(var: &str) -> Option<PathBuf> {
    match std::env::var_os(var).map(PathBuf::from) {
        Some(path) if path.is_file() => Some(path),
        Some(path) => {
            eprintln!("{var}={} is not a file; skipping", path.display());
            None
        }
        None => {
            eprintln!("{var} is not set; skipping this local-only interop test");
            None
        }
    }
}

/// A free loopback TCP port (released for the child to bind).
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// A child process that is killed and reaped when this guard drops: at the
/// end of the test and while a failed assertion unwinds.
struct ChildGuard {
    name: &'static str,
    child: std::process::Child,
}

impl ChildGuard {
    fn spawn(name: &'static str, command: &mut std::process::Command) -> Self {
        let child = command
            .spawn()
            .unwrap_or_else(|err| panic!("cannot start {name}: {err}"));
        Self { name, child }
    }

    /// Fail if the child already exited (a port clash, a bad argument).
    fn assert_running(&mut self) {
        if let Ok(Some(status)) = self.child.try_wait() {
            panic!("{} exited early: {status}", self.name);
        }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Wait until `url` answers HTTP 200.
fn wait_http(url: &str, timeout: Duration, child: &mut ChildGuard) {
    let agent = ureq::Agent::config_builder()
        .proxy(None)
        .http_status_as_error(false)
        .timeout_global(Some(Duration::from_secs(2)))
        .build()
        .new_agent();
    let deadline = Instant::now() + timeout;
    loop {
        child.assert_running();
        if agent
            .get(url)
            .call()
            .is_ok_and(|r| r.status().as_u16() == 200)
        {
            return;
        }
        assert!(Instant::now() < deadline, "{url} never answered");
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// Raw member access for checks outside the drivers.
fn raw_client(base: &str, device_type: &str) -> AlpacaClient {
    AlpacaClient::new(
        base,
        device_type,
        0,
        ClientConfig::new(ClientId::new(4322).unwrap()),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Local only: the ESP32 firmware's host simulator (ACHERNAR_MOUNT_SIM)
// ---------------------------------------------------------------------------

/// `achernar-mount-sim` on an ephemeral Alpaca port, with discovery and the
/// LX200 port off. The banner names the port:
/// `achernar-mount-sim 0.1.0 alpaca=127.0.0.1:PORT discovery=0 lx200=0 ...`.
#[cfg(unix)]
struct HostSim {
    base: String,
    _child: ChildGuard,
}

#[cfg(unix)]
impl HostSim {
    fn start(path: &Path, extra: &[&str]) -> Self {
        use std::io::BufRead;
        let mut command = std::process::Command::new(path);
        command
            .args([
                "--bind",
                "127.0.0.1",
                "--alpaca-port",
                "0",
                "--discovery-port",
                "0",
                "--lx200-port",
                "0",
            ])
            .args(extra)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit());
        let mut child = ChildGuard::spawn("achernar-mount-sim", &mut command);
        let stdout = child.child.stdout.take().expect("piped stdout");
        let (tx, rx) = std::sync::mpsc::channel();
        // Read the banner, then keep draining so the simulator never blocks
        // on a full pipe; the thread ends when the process does.
        std::thread::spawn(move || {
            let mut lines = std::io::BufReader::new(stdout).lines();
            if let Some(Ok(banner)) = lines.next() {
                let _ = tx.send(banner);
            }
            for _ in lines {}
        });
        let banner = rx
            .recv_timeout(Duration::from_secs(10))
            .unwrap_or_else(|_| panic!("achernar-mount-sim printed no banner"));
        let address = banner
            .split_whitespace()
            .find_map(|word| word.strip_prefix("alpaca="))
            .unwrap_or_else(|| panic!("no alpaca= in the banner {banner:?}"))
            .to_string();
        let base = format!("http://{address}");
        wait_http(
            &format!("{base}/management/v1/configureddevices"),
            Duration::from_secs(10),
            &mut child,
        );
        Self {
            base,
            _child: child,
        }
    }
}

#[cfg(unix)]
const HOST_SIM_SITE: [&str; 6] = ["--lat", "40", "--lon", "-75", "--elev", "100"];

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "local only: set ACHERNAR_MOUNT_SIM to FW/host_sim/build/achernar-mount-sim"]
async fn interop_esp32_host_sim_runs_the_native_alpaca_mount_end_to_end() {
    let Some(path) = fixture("ACHERNAR_MOUNT_SIM") else {
        return;
    };
    let mut args = HOST_SIM_SITE.to_vec();
    args.extend(["--dec", "present", "--unique-id", "interop-host-sim-dec"]);
    let sim = HostSim::start(&path, &args);
    let daemon = Daemon::start(scenario_configs(40.0, -75.0, 100.0)).await;
    let option = format!("{}#device=0", sim.base);

    // Connect: the firmware's identity, nothing moving (D16).
    let connected_at = Instant::now();
    let body = daemon.connect("mount", "alpaca", &option).await;
    assert_eq!(body["identity"]["unique_id"], "interop-host-sim-dec");
    let value = daemon
        .wait_status("a polled firmware mount", |v| {
            v["mount"]["lst_h"].is_number() && v["mount"]["slewing"].is_boolean()
        })
        .await;
    eprintln!(
        "host sim: status complete {} ms after connect",
        connected_at.elapsed().as_millis()
    );
    let mount = &value["mount"];
    assert_eq!(mount["slewing"], false, "connecting moved the mount");
    assert_eq!(mount["frame"], "jnow_apparent", "EquatorialSystem 1");
    assert_eq!(mount["alignment_mode"], "polar");
    assert_eq!(mount["can_flip"], false, "D14");
    assert_eq!(mount["pier_side"], "east", "the firmware's constant 0");
    assert_eq!(mount["state"], "ready");
    for key in crate::equipment::mount::STATUS_KEYS {
        assert!(mount.get(*key).is_some(), "mount block lacks {key}");
    }

    if mount["parked"] == true {
        let mark = daemon.mark();
        daemon.post_ok("/mount/unpark", json!({})).await;
        daemon.wait_event("unparked", mark).await;
        daemon.wait_idle().await;
    }

    // A short J2000 GoTo near the pole (the firmware slews at 1 deg/s).
    let status = daemon.status().await;
    let (ra, dec) = target_from(&status, 0.3, 86.0);
    let mark = daemon.mark();
    daemon
        .post_ok(
            "/mount/slew",
            json!({"ra": ra, "dec": dec, "frame": "j2000"}),
        )
        .await;
    let settled = daemon.wait_event("slew_settled", mark).await;
    daemon.wait_idle().await;
    assert_in_order(
        &daemon.events_after(mark, "").await,
        &[
            "slew_commanded",
            "slew_started",
            "slew_arrived",
            "slew_settled",
        ],
    );
    eprintln!("host sim: settled {}", settled["payload"]);
    let value = daemon
        .wait_status("tracking at the target", |v| v["mount"]["tracking"] == true)
        .await;
    let arrived = separation_arcmin(
        value["mount"]["ra_j2000"].as_f64().unwrap(),
        value["mount"]["dec_j2000"].as_f64().unwrap(),
        ra,
        dec,
    );
    assert!(arrived < 5.0, "arrived {arrived:.2}' from the target");

    // Abort mid-GoTo (D19, O10: never gated on a Can* flag).
    let (ra_far, dec_far) = target_from(&daemon.status().await, -0.8, 78.0);
    let mark = daemon.mark();
    daemon
        .post_ok(
            "/mount/slew",
            json!({"ra": ra_far, "dec": dec_far, "frame": "j2000"}),
        )
        .await;
    daemon
        .wait_status("slewing", |v| v["mount"]["slewing"] == true)
        .await;
    let started = Instant::now();
    let aborted = daemon.post_ok("/mount/abort", json!({})).await;
    assert_eq!(aborted["abort_path_ran"], true, "{aborted}");
    daemon
        .wait_status("stopped", |v| v["mount"]["slewing"] == false)
        .await;
    let latency = started.elapsed();
    eprintln!(
        "host sim: Slewing false {} ms after the abort",
        latency.as_millis()
    );
    assert!(latency < Duration::from_secs(4), "{latency:?}");
    let ended = daemon.wait_event("slew_ended", mark).await;
    assert_eq!(
        ended["payload"]["outcome"]["status"], "aborted",
        "the abort interrupted the GoTo: {ended}"
    );
    let value = daemon.status().await;
    let short = separation_arcmin(
        value["mount"]["ra_j2000"].as_f64().unwrap(),
        value["mount"]["dec_j2000"].as_f64().unwrap(),
        ra_far,
        dec_far,
    );
    assert!(short > 30.0, "the GoTo was not interrupted ({short:.1}')");
    daemon.wait_idle().await;

    // Tracking off and on; a pulse.
    daemon
        .post_ok("/mount/tracking", json!({"enabled": false}))
        .await;
    daemon
        .wait_status("tracking off", |v| v["mount"]["tracking"] == false)
        .await;
    daemon.wait_idle().await;
    daemon
        .post_ok("/mount/tracking", json!({"enabled": true}))
        .await;
    daemon
        .wait_status("tracking on", |v| v["mount"]["tracking"] == true)
        .await;
    daemon.wait_idle().await;
    let mark = daemon.mark();
    daemon
        .post_ok(
            "/mount/pulse-guide",
            json!({"direction": "east", "duration_ms": 300}),
        )
        .await;
    daemon.wait_event("pulse_guide", mark).await;
    daemon.wait_idle().await;

    // Park, verified through AtPark (M-f), and unpark.
    let mark = daemon.mark();
    daemon.post_ok("/mount/park", json!({})).await;
    daemon.wait_event("parked", mark).await;
    daemon.wait_idle().await;
    assert_eq!(daemon.status().await["mount"]["parked"], true);
    let mark = daemon.mark();
    daemon.post_ok("/mount/unpark", json!({})).await;
    daemon.wait_event("unparked", mark).await;
    daemon.wait_idle().await;

    // E-stop: AbortSlew, MoveAxis 0 and tracking off on the firmware.
    let report = daemon.post_ok("/equipment/estop", json!({})).await;
    assert!(report["elapsed_ms"].as_u64().unwrap() < 2000, "{report}");
    daemon
        .wait_status("tracking off", |v| v["mount"]["tracking"] == false)
        .await;
    daemon
        .post_ok("/disconnect", json!({"equipment_type": "mount"}))
        .await;
    daemon.flush().await;
    assert!(
        daemon
            .count(
                "SELECT COUNT(*) FROM device_connections WHERE \
                 identity_key = 'alpaca:interop-host-sim-dec' AND disconnected_at IS NOT NULL"
            )
            .await
            == 1
    );
    daemon.close().await;

    // The telescope conformance suite against the same firmware.
    let base = sim.base.clone();
    tokio::task::spawn_blocking(move || {
        let clock = SystemClock::shared();
        let mut telescope = telescope_driver(&base, &clock);
        assert_conforms(
            "Alpaca telescope vs the ESP32 firmware",
            &mount_suite(&mut telescope, clock.as_ref()),
        );
        assert_conforms("disconnect", &disconnect_suite(&mut telescope));
    })
    .await
    .unwrap();
    drop(sim);
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "local only: set ACHERNAR_MOUNT_SIM to FW/host_sim/build/achernar-mount-sim"]
async fn interop_esp32_host_sim_without_dec_refuses_an_out_of_window_goto_cleanly() {
    // The user's current mount has no Dec motor (`CFG_DEC_INSTALLED 0`): the
    // firmware refuses GoTos more than 1 deg from the synced Dec. The daemon
    // must surface that as an error and leave the mount still, not slew,
    // hang or retry.
    let Some(path) = fixture("ACHERNAR_MOUNT_SIM") else {
        return;
    };
    let mut args = HOST_SIM_SITE.to_vec();
    args.extend(["--dec", "absent", "--unique-id", "interop-host-sim-nodec"]);
    let sim = HostSim::start(&path, &args);
    let daemon = Daemon::start(scenario_configs(40.0, -75.0, 100.0)).await;
    daemon
        .connect("mount", "alpaca", &format!("{}#device=0", sim.base))
        .await;
    let value = daemon
        .wait_status("a polled firmware mount", |v| {
            v["mount"]["lst_h"].is_number() && v["mount"]["parked"].is_boolean()
        })
        .await;
    if value["mount"]["parked"] == true {
        let mark = daemon.mark();
        daemon.post_ok("/mount/unpark", json!({})).await;
        daemon.wait_event("unparked", mark).await;
        daemon.wait_idle().await;
    }
    let status = daemon.status().await;
    let dec_now = status["mount"]["dec"].as_f64().unwrap();
    let (ra_far, _) = target_from(&status, 0.2, dec_now);
    let mark = daemon.mark();
    let (code, body) = daemon
        .post(
            "/mount/slew",
            json!({"ra": ra_far, "dec": dec_now - 8.0, "frame": "mount"}),
        )
        .await;
    eprintln!("host sim (no Dec): an 8 deg Dec GoTo answered {code} {body}");
    if (200..300).contains(&code) {
        // Accepted by the gate, refused by the device: the operation fails
        // (it does not report success or keep the mount slewing).
        let ended = daemon.wait_event("slew_ended", mark).await;
        eprintln!("host sim (no Dec): {}", ended["payload"]);
        let outcome = &ended["payload"]["outcome"]["status"];
        assert!(outcome.is_string() && outcome != "succeeded", "{ended}");
    } else {
        assert!((400..500).contains(&code), "{code}: {body}");
        assert!(body["error"]["code"].is_string(), "{body}");
    }
    daemon.wait_idle().await;
    let value = daemon.status().await;
    assert_eq!(value["mount"]["slewing"], false);
    let moved = (value["mount"]["dec"].as_f64().unwrap() - dec_now).abs();
    assert!(moved < 0.1, "the Dec moved by {moved} deg");
    assert_eq!(value["device_health"]["mount"]["state"], "ready");
    daemon.close().await;
    drop(sim);
}

// ---------------------------------------------------------------------------
// Local only: ASCOM OmniSim (ACHERNAR_OMNISIM, O9)
// ---------------------------------------------------------------------------

/// One OmniSim at a time: every instance also binds UDP 32227 for discovery.
static OMNISIM_SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// OmniSim bound to `127.0.0.1:{OMNISIM_PORT}` on a free port, with its
/// output in a log under the temporary directory (named on stderr).
struct OmniSim {
    base: String,
    _child: ChildGuard,
}

impl OmniSim {
    fn start(launcher: &Path) -> Self {
        let port = free_port();
        let log = std::env::temp_dir().join(format!("achernar-interop-omnisim-{port}.log"));
        eprintln!("OmniSim on port {port}, log {}", log.display());
        let out = std::fs::File::create(&log).unwrap();
        let err = out.try_clone().unwrap();
        let mut command = std::process::Command::new(launcher);
        command
            .env("OMNISIM_PORT", port.to_string())
            .stdin(std::process::Stdio::null())
            .stdout(out)
            .stderr(err);
        let mut child = ChildGuard::spawn("OmniSim", &mut command);
        let base = format!("http://127.0.0.1:{port}");
        wait_http(
            &format!("{base}/management/v1/configureddevices"),
            Duration::from_secs(60),
            &mut child,
        );
        Self {
            base,
            _child: child,
        }
    }
}

/// Known OmniSim v0.5.0 departures from the ASCOM interface standards that
/// a native driver passes through faithfully. Each one is printed when it
/// shows up, never silently dropped; anything else fails the test.
fn omnisim_deviation(violation: &Violation) -> Option<&'static str> {
    // ASCOM `Action`: an unsupported action name raises
    // ActionNotImplemented (0x40C). OmniSim's telescope answers
    // InvalidOperation (0x40B, ErrorNumber 1035) instead.
    (violation.rule == "unknown_action_is_unsupported"
        && violation.detail.contains("code: Some(1035)"))
    .then_some("OmniSim answers an unknown Action with 0x40B, not ActionNotImplemented 0x40C")
}

#[test]
#[ignore = "local only: set ACHERNAR_OMNISIM to the OmniSim launcher (O9)"]
fn interop_omnisim_every_native_alpaca_driver_conforms() {
    let Some(launcher) = fixture("ACHERNAR_OMNISIM") else {
        return;
    };
    let _serial = OMNISIM_SERIAL.blocking_lock();
    let omni = OmniSim::start(&launcher);
    let clock = SystemClock::shared();
    let base = omni.base.clone();

    // Every suite runs before anything is asserted, so one report names
    // every driver's violations.
    let mut reports: Vec<(&str, Vec<Violation>)> = Vec::new();
    let mut telescope = telescope_driver(&base, &clock);
    let mut violations = mount_suite(&mut telescope, clock.as_ref());
    violations.extend(disconnect_suite(&mut telescope));
    reports.push(("Alpaca telescope", violations));

    let mut focuser = focuser_driver(&base, &clock);
    let mut violations = focuser_suite(&mut focuser, clock.as_ref());
    violations.extend(disconnect_suite(&mut focuser));
    reports.push(("Alpaca focuser", violations));

    let (address, transport, _) = alpaca_parts(&base, "filterwheel", 0);
    let mut wheel = AlpacaFilterWheel::new(
        alpaca_identity(DeviceKind::FilterWheel, &address, "filterwheel"),
        transport,
        Arc::clone(&clock),
    );
    let mut violations = filter_wheel_suite(&mut wheel, clock.as_ref());
    violations.extend(disconnect_suite(&mut wheel));
    reports.push(("Alpaca filter wheel", violations));

    let (address, transport, _) = alpaca_parts(&base, "observingconditions", 0);
    let mut weather = AlpacaObservingConditions::new(
        alpaca_identity(DeviceKind::Weather, &address, "observingconditions"),
        transport,
        Arc::clone(&clock),
    );
    let mut violations = weather_suite(&mut weather, clock.as_ref());
    violations.extend(disconnect_suite(&mut weather));
    reports.push(("Alpaca observing conditions", violations));

    let (address, transport, _) = alpaca_parts(&base, "safetymonitor", 0);
    let mut monitor = AlpacaSafetyMonitor::new(
        alpaca_identity(DeviceKind::SafetyMonitor, &address, "safetymonitor"),
        transport,
        Arc::clone(&clock),
    );
    let mut violations = safety_suite(&mut monitor, clock.as_ref());
    violations.extend(disconnect_suite(&mut monitor));
    reports.push(("Alpaca safety monitor", violations));

    let mut failures = Vec::new();
    for (driver, violations) in reports {
        let (deviations, broken): (Vec<_>, Vec<_>) = violations
            .into_iter()
            .partition(|v| omnisim_deviation(v).is_some());
        for deviation in &deviations {
            eprintln!(
                "{driver} vs OmniSim: {deviation} ({})",
                omnisim_deviation(deviation).unwrap_or_default()
            );
        }
        if !broken.is_empty() {
            failures.push(format!(
                "{driver} vs OmniSim:\n    {}",
                broken
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("\n    ")
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));

    // Rotator, CoverCalibrator, Switch and Dome have no native driver until
    // Tier 2 (WP2-5): the WP1A-1 client connects and reads each one.
    for (device_type, member) in [
        ("rotator", "position"),
        ("covercalibrator", "coverstate"),
        ("switch", "maxswitch"),
        ("dome", "shutterstatus"),
    ] {
        let client = raw_client(&base, device_type);
        let version = client
            .get("interfaceversion", &[], TimeoutClass::Connect)
            .unwrap();
        assert!(version.as_i64().is_some_and(|v| v >= 1), "{device_type}");
        let platform7 = match device_type {
            "rotator" => 4,
            "covercalibrator" => 2,
            "switch" => 3,
            _ => 3,
        };
        if version.as_i64().unwrap() >= platform7 {
            client.put("connect", &[], TimeoutClass::Connect).unwrap();
            let deadline = Instant::now() + Duration::from_secs(10);
            while client
                .get("connecting", &[], TimeoutClass::Connect)
                .unwrap()
                == json!(true)
            {
                assert!(Instant::now() < deadline, "{device_type} never connected");
                std::thread::sleep(Duration::from_millis(100));
            }
        } else {
            client
                .put(
                    "connected",
                    &[("Connected", "True".into())],
                    TimeoutClass::Connect,
                )
                .unwrap();
        }
        assert_eq!(
            client.get("connected", &[], TimeoutClass::Connect).unwrap(),
            json!(true),
            "{device_type}"
        );
        assert!(client.get("name", &[], TimeoutClass::Ordinary).is_ok());
        let value = client
            .get(member, &[], TimeoutClass::Ordinary)
            .unwrap_or_else(|err| panic!("{device_type} {member}: {err}"));
        assert!(!value.is_null(), "{device_type} {member}");
        client
            .put(
                "connected",
                &[("Connected", "False".into())],
                TimeoutClass::Connect,
            )
            .unwrap();
    }
    drop(omni);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "local only: set ACHERNAR_OMNISIM to the OmniSim launcher (O9)"]
async fn interop_omnisim_daemon_drives_every_tier1_alpaca_device() {
    let Some(launcher) = fixture("ACHERNAR_OMNISIM") else {
        return;
    };
    let _serial = OMNISIM_SERIAL.lock().await;
    let omni = tokio::task::spawn_blocking(move || OmniSim::start(&launcher))
        .await
        .unwrap();
    // OmniSim's own site (51.07861 N, 0.29444 W, 845 m) matches [site], so
    // verify_only finds no mismatch.
    let daemon = Daemon::start(scenario_configs(51.07861, -0.29444, 845.0)).await;
    let option = format!("{}#device=0", omni.base);
    for kind in [
        "mount",
        "focuser",
        "filter_wheel",
        "weather",
        "safety_monitor",
    ] {
        let body = daemon.connect(kind, "alpaca", &option).await;
        assert_eq!(body["identity"]["driver"], "alpaca", "{kind}");
        assert!(body["identity"]["unique_id"].is_string(), "{kind}: {body}");
    }
    // No native driver yet for these (Tier 2, WP2-5): refused, never faked.
    for kind in ["rotator", "cover_calibrator", "switch", "dome"] {
        let (code, body) = daemon
            .post(
                "/connect",
                json!({"equipment_type": kind, "driver": "alpaca", "option": option}),
            )
            .await;
        assert_eq!(code, 501, "{kind}: {body}");
    }
    let value = daemon
        .wait_status("every OmniSim device polled", |v| {
            v["mount"]["lst_h"].is_number()
                && v["focuser"]["position"].is_number()
                && v["filter_wheel"]["slot"].is_number()
                && v["weather"].is_object()
                && v["safety_monitor"]["is_safe"].is_boolean()
        })
        .await;
    assert_eq!(
        value["mount"]["frame"], "jnow_apparent",
        "EquatorialSystem 1"
    );
    assert_eq!(value["mount"]["alignment_mode"], "german_polar");
    assert_eq!(value["safety_monitor"]["is_safe"], true);
    assert!(
        value["weather"].to_string().contains("temperature"),
        "{}",
        value["weather"]
    );

    // Mount: a J2000 GoTo that settles, abort, a pulse, park and unpark.
    if value["mount"]["parked"] == true {
        let mark = daemon.mark();
        daemon.post_ok("/mount/unpark", json!({})).await;
        daemon.wait_event("unparked", mark).await;
        daemon.wait_idle().await;
    }
    let status = daemon.status().await;
    let (ra, dec) = target_from(&status, 1.0, 40.0);
    let mark = daemon.mark();
    daemon
        .post_ok(
            "/mount/slew",
            json!({"ra": ra, "dec": dec, "frame": "j2000"}),
        )
        .await;
    daemon.wait_event("slew_settled", mark).await;
    daemon.wait_idle().await;
    let value = daemon
        .wait_status("tracking", |v| v["mount"]["tracking"] == true)
        .await;
    let arrived = separation_arcmin(
        value["mount"]["ra_j2000"].as_f64().unwrap(),
        value["mount"]["dec_j2000"].as_f64().unwrap(),
        ra,
        dec,
    );
    assert!(arrived < 2.0, "arrived {arrived:.2}' from the target");
    let (ra_far, dec_far) = target_from(&status, -3.0, 10.0);
    daemon
        .post_ok(
            "/mount/slew",
            json!({"ra": ra_far, "dec": dec_far, "frame": "j2000"}),
        )
        .await;
    daemon
        .wait_status("slewing", |v| v["mount"]["slewing"] == true)
        .await;
    let started = Instant::now();
    let aborted = daemon.post_ok("/mount/abort", json!({})).await;
    assert_eq!(aborted["abort_path_ran"], true);
    daemon
        .wait_status("stopped", |v| v["mount"]["slewing"] == false)
        .await;
    assert!(started.elapsed() < Duration::from_secs(2), "§5.7 item 2");
    daemon.wait_idle().await;
    let mark = daemon.mark();
    daemon
        .post_ok(
            "/mount/pulse-guide",
            json!({"direction": "north", "duration_ms": 300}),
        )
        .await;
    daemon.wait_event("pulse_guide", mark).await;
    daemon.wait_idle().await;
    let mark = daemon.mark();
    daemon.post_ok("/mount/park", json!({})).await;
    daemon.wait_event("parked", mark).await;
    daemon.wait_idle().await;
    let mark = daemon.mark();
    daemon.post_ok("/mount/unpark", json!({})).await;
    daemon.wait_event("unparked", mark).await;
    daemon.wait_idle().await;

    // Focuser: a move with read-back.
    let position = daemon.status().await["focuser"]["position"]
        .as_i64()
        .unwrap();
    let target = if position > 25_000 {
        position - 300
    } else {
        position + 300
    };
    daemon
        .post_ok("/focuser/moveto", json!({"position": target}))
        .await;
    daemon
        .wait_status("the focuser arrived", |v| {
            v["focuser"]["position"] == target && v["focuser"]["active_op"].is_null()
        })
        .await;
    daemon.wait_idle().await;

    // Filter wheel: a change to the next slot through the handshake.
    let slot = daemon.status().await["filter_wheel"]["slot"]
        .as_u64()
        .unwrap();
    let next = (slot + 1) % 2;
    let mark = daemon.mark();
    daemon
        .post_ok("/filter-wheel/change", json!({"slot": next}))
        .await;
    let ended = daemon.wait_event("filter_change_ended", mark).await;
    eprintln!("OmniSim filter change: {}", ended["payload"]);
    daemon
        .wait_status("the wheel verified", |v| {
            v["filter_wheel"]["slot"] == next && v["filter_wheel"]["verified"] == true
        })
        .await;
    daemon.wait_idle().await;

    // Weather and safety read through their routes.
    let weather = daemon.get("/weather").await;
    assert!(weather["fields"].is_object(), "{weather}");
    assert_eq!(daemon.get("/safety").await["state"], "safe");

    // E-stop reaches every motion device; the operator clears the block.
    let report = daemon.post_ok("/equipment/estop", json!({})).await;
    assert!(report["elapsed_ms"].as_u64().unwrap() < 2000, "{report}");
    let blocks = daemon.get("/equipment/resume-blocks").await;
    let id = blocks["blocks"][0]["id"].as_str().unwrap().to_string();
    daemon
        .post_ok("/equipment/resume-blocks/clear", json!({"id": id}))
        .await;

    daemon.post_ok("/disconnect", json!({})).await;
    daemon.flush().await;
    assert_eq!(
        daemon
            .count("SELECT COUNT(*) FROM device_connections WHERE disconnected_at IS NULL")
            .await,
        0
    );
    assert!(
        daemon
            .count("SELECT COUNT(*) FROM focuser_moves WHERE raw_end IS NOT NULL")
            .await
            >= 1
    );
    daemon.close().await;
    drop(omni);
}

//! Passive device liveness and one authoritative disconnect lifecycle.
//!
//! Probes never expose, slew, focus, or move a filter. Each device has at most
//! one blocking SDK call in flight. A deadline invalidates its public state;
//! it does not cancel native code or free a handle while native code uses it.
//!
//! Only the camera (OLS) is probed here. Every other kind runs behind an
//! equipment-layer worker, and the worker is the probe (spec §2.8): its
//! health reaches this registry through the spine's projector, which keeps
//! the `status` values `"connected"`/`"disconnected"` (D8) and adds `state`,
//! `failure_class` and `reason`.

use std::collections::BTreeMap;
use std::sync::{atomic::Ordering, Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::state::{AppState, CameraRuntimeState};

const POLL_INTERVAL: Duration = Duration::from_secs(2);
const PROBE_DEADLINE: Duration = Duration::from_secs(3);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind {
    Camera,
    Mount,
    Focuser,
    FilterWheel,
    // Equipment-layer kinds (spec §2.4). Their handles live in the equipment
    // hub; the passive probe loop below never probes them.
    Guider,
    Rotator,
    CoverCalibrator,
    Switch,
    Weather,
    SafetyMonitor,
    Dome,
}

impl DeviceKind {
    /// Every kind, in registry (and `device_health` key) order.
    pub const ALL: [DeviceKind; 11] = [
        DeviceKind::Camera,
        DeviceKind::Mount,
        DeviceKind::Focuser,
        DeviceKind::FilterWheel,
        DeviceKind::Guider,
        DeviceKind::Rotator,
        DeviceKind::CoverCalibrator,
        DeviceKind::Switch,
        DeviceKind::Weather,
        DeviceKind::SafetyMonitor,
        DeviceKind::Dome,
    ];
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceHealth {
    pub connected: bool,
    pub status: String,
    pub connection_id: Option<String>,
    /// None until a real probe has run; false means this driver cannot probe.
    pub probe_supported: Option<bool>,
    pub last_checked_at: Option<chrono::DateTime<chrono::Utc>>,
    pub last_error: Option<String>,
    /// `connecting | ready | degraded | lost | disconnected` (§2.8, D8).
    /// Null for a device no worker reports on (the camera).
    #[serde(default)]
    pub state: Option<String>,
    /// `disconnected | unsupported | timeout | outcome_unknown | driver |
    /// driver_panic | invalid`, while the device is degraded or lost.
    #[serde(default)]
    pub failure_class: Option<String>,
    /// Mirrors `last_error`; the frontend reads `device_health[x].reason`.
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(skip)]
    pub snapshot: Option<Value>,
}

impl Default for DeviceHealth {
    fn default() -> Self {
        Self {
            connected: false,
            status: "disconnected".into(),
            connection_id: None,
            probe_supported: None,
            last_checked_at: None,
            last_error: None,
            state: None,
            failure_class: None,
            reason: None,
            snapshot: None,
        }
    }
}

fn label<T: Serialize>(value: &T) -> Option<String> {
    serde_json::to_value(value)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
}

#[derive(Debug, Clone)]
pub struct DeviceRegistry(pub BTreeMap<DeviceKind, DeviceHealth>);

impl Default for DeviceRegistry {
    /// Every kind starts disconnected and unprobed (`probe_supported: None`).
    fn default() -> Self {
        Self(
            DeviceKind::ALL
                .into_iter()
                .map(|kind| (kind, DeviceHealth::default()))
                .collect(),
        )
    }
}

impl DeviceRegistry {
    pub fn connected(&mut self, kind: DeviceKind, connection_id: String, snapshot: Option<Value>) {
        self.0.insert(
            kind,
            DeviceHealth {
                connected: true,
                status: "connected".into(),
                connection_id: Some(connection_id),
                snapshot,
                ..DeviceHealth::default()
            },
        );
    }

    pub fn connection_id(&self, kind: DeviceKind) -> Option<String> {
        self.0
            .get(&kind)
            .filter(|health| health.connected)?
            .connection_id
            .clone()
    }

    /// A late result from an old connection can never revive a retired device.
    pub fn checked(
        &mut self,
        kind: DeviceKind,
        id: &str,
        supported: bool,
        snapshot: Option<Value>,
    ) -> bool {
        let Some(health) = self.0.get_mut(&kind) else {
            return false;
        };
        if !health.connected || health.connection_id.as_deref() != Some(id) {
            return false;
        }
        health.probe_supported = Some(supported);
        health.last_checked_at = Some(chrono::Utc::now());
        if snapshot.is_some() {
            health.snapshot = snapshot;
        }
        true
    }

    pub fn disconnected(
        &mut self,
        kind: DeviceKind,
        expected: Option<&str>,
        error: Option<String>,
    ) -> bool {
        let health = self.0.entry(kind).or_default();
        if expected.is_some() && health.connection_id.as_deref() != expected {
            return false;
        }
        health.connected = false;
        health.status = "disconnected".into();
        health.connection_id = None;
        health.snapshot = None;
        health.last_checked_at = Some(chrono::Utc::now());
        health.reason = error.clone();
        health.last_error = error;
        if health.state.is_some() {
            health.state = Some("disconnected".into());
            health.failure_class = None;
        }
        true
    }

    /// The worker's link state for the current connection `id` (§2.8).
    pub fn set_link(
        &mut self,
        kind: DeviceKind,
        id: &str,
        link: &crate::equipment::types::HealthState,
    ) -> bool {
        let Some(health) = self.0.get_mut(&kind) else {
            return false;
        };
        if !health.connected || health.connection_id.as_deref() != Some(id) {
            return false;
        }
        health.state = label(&link.state);
        health.failure_class = link.failure_class.as_ref().and_then(label);
        health.reason = link.reason.clone();
        health.last_error = link.reason.clone();
        true
    }

    /// A worker declared connection `id` lost: `connected: false`, `status:
    /// "disconnected"` (D8), `state: "lost"`, with the failure class and
    /// reason. Guarded by the connection id, so it never touches a
    /// replacement. True when the entry changed.
    pub fn lost(
        &mut self,
        kind: DeviceKind,
        id: &str,
        link: &crate::equipment::types::HealthState,
    ) -> bool {
        if !self.disconnected(kind, Some(id), link.reason.clone()) {
            return false;
        }
        if let Some(health) = self.0.get_mut(&kind) {
            health.state = label(&link.state);
            health.failure_class = link.failure_class.as_ref().and_then(label);
        }
        true
    }

    pub fn snapshot(&self, kind: DeviceKind) -> Option<Value> {
        self.0
            .get(&kind)
            .filter(|health| health.connected)?
            .snapshot
            .clone()
    }
}

/// Only publication state is captured by a native callback. No camera/driver
/// handle is captured, and no SDK calls or thread joins run on that callback.
#[derive(Clone)]
pub struct CameraLossSignals {
    pub runtime: Arc<RwLock<CameraRuntimeState>>,
    pub generation: Arc<std::sync::atomic::AtomicU64>,
    pub barrier: Arc<Mutex<()>>,
    pub latest_frame: Arc<Mutex<Option<super::camera::FrameData>>>,
    pub observation: Arc<RwLock<Option<crate::vision::shared_observation::SharedObservation>>>,
    pub observed_frame: Arc<RwLock<Option<crate::vision::observed_frame::ObservedFrame>>>,
    pub streaming: Arc<Mutex<bool>>,
    pub focus_abort: Arc<std::sync::atomic::AtomicBool>,
    pub devices: Arc<RwLock<DeviceRegistry>>,
    pub updates: tokio::sync::broadcast::Sender<()>,
}

impl CameraLossSignals {
    pub fn invalidate(&self, expected_id: &str, reason: String) -> bool {
        let _barrier = self.barrier.lock().unwrap();
        {
            let mut runtime = self.runtime.write().unwrap();
            if runtime.connection_id.as_deref() != Some(expected_id) {
                return false;
            }
            *runtime = CameraRuntimeState {
                last_error: Some(reason.clone()),
                ..CameraRuntimeState::default()
            };
        }
        self.generation.fetch_add(1, Ordering::AcqRel);
        *self.latest_frame.lock().unwrap() = None;
        *self.observation.write().unwrap() = None;
        *self.observed_frame.write().unwrap() = None;
        *self.streaming.lock().unwrap() = false;
        self.focus_abort.store(true, Ordering::Release);
        self.devices.write().unwrap().disconnected(
            DeviceKind::Camera,
            Some(expected_id),
            Some(reason),
        );
        let _ = self.updates.send(());
        true
    }
}

pub fn spawn(state: AppState) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_millis(100));
        let mut next_probe = Instant::now();
        let mut pending: BTreeMap<DeviceKind, PendingProbe> = BTreeMap::new();
        loop {
            ticker.tick().await;
            pending.retain(|_, probe| !probe.worker.is_finished());
            for (kind, probe) in pending.iter_mut() {
                if !probe.timed_out && probe.started.elapsed() >= PROBE_DEADLINE {
                    probe.timed_out = true;
                    lose_device(
                        &state,
                        *kind,
                        &probe.connection_id,
                        "Passive device health check timed out".into(),
                    );
                }
            }
            // Callback faults clear public state immediately. Reap their
            // handles separately, only after command ownership is available.
            let camera_handle_present = state
                .camera
                .try_lock()
                .map(|slot| slot.is_some())
                .unwrap_or(false);
            if camera_handle_present
                && state.camera_runtime.read().unwrap().connection_id.is_none()
                && !pending.contains_key(&DeviceKind::Camera)
            {
                if let Ok(ownership) = state.acquisition.clone().try_lock_owned() {
                    let worker_state = state.clone();
                    let worker = tokio::task::spawn_blocking(move || {
                        let _ownership = ownership;
                        retire_camera(&worker_state);
                    });
                    pending.insert(
                        DeviceKind::Camera,
                        PendingProbe {
                            connection_id: String::new(),
                            started: Instant::now(),
                            worker,
                            timed_out: true,
                        },
                    );
                }
            }
            if Instant::now() < next_probe {
                continue;
            }
            next_probe = Instant::now() + POLL_INTERVAL;
            // Handle-based kinds are not probed: their workers are the probe
            // (§2.8), and their health reaches the registry through the
            // spine's projector.
            for kind in [DeviceKind::Camera] {
                if pending.contains_key(&kind) {
                    continue;
                }
                let Some(id) = state.devices.read().unwrap().connection_id(kind) else {
                    continue;
                };
                let ownership = if kind == DeviceKind::Camera {
                    match state.acquisition.clone().try_lock_owned() {
                        Ok(guard) => Some(guard),
                        Err(_) => continue, // Exposures/commands are not failed health checks.
                    }
                } else {
                    None
                };
                let worker_state = state.clone();
                let worker_id = id.clone();
                let worker = tokio::task::spawn_blocking(move || {
                    let _ownership = ownership;
                    probe_device(&worker_state, kind, &worker_id);
                });
                pending.insert(
                    kind,
                    PendingProbe {
                        connection_id: id,
                        started: Instant::now(),
                        worker,
                        timed_out: false,
                    },
                );
            }
        }
    });
}

struct PendingProbe {
    connection_id: String,
    started: Instant,
    worker: tokio::task::JoinHandle<()>,
    timed_out: bool,
}

fn lose_device(state: &AppState, kind: DeviceKind, id: &str, reason: String) {
    tracing::warn!(?kind, %reason, "Device connection lost");
    if kind == DeviceKind::Camera {
        state.camera_loss_signals().invalidate(id, reason);
    } else if state
        .devices
        .write()
        .unwrap()
        .disconnected(kind, Some(id), Some(reason))
    {
        if kind == DeviceKind::Focuser {
            state.focus_abort_requested.store(true, Ordering::Release);
        }
        let _ = state.device_updates.send(());
    }
}

fn probe_device(state: &AppState, kind: DeviceKind, id: &str) {
    let result: Result<(bool, Option<Value>), String> = match kind {
        DeviceKind::Camera => {
            let result = {
                let Ok(camera) = state.camera.try_lock() else {
                    return;
                };
                if state.devices.read().unwrap().connection_id(kind).as_deref() != Some(id) {
                    drop(camera);
                    retire_camera(state);
                    return;
                }
                let Some(camera) = camera.as_ref() else {
                    return;
                };
                camera
                    .stream_health()
                    .and_then(|_| camera.check_connection())
                    .map(|supported| (supported, None))
                    .map_err(|e| e.to_string())
            };
            result
        }
        // Every other kind runs behind an equipment-layer worker, which is
        // its probe (spec §2.8). A stale call for a connection the registry
        // has retired still releases the retained handle.
        DeviceKind::Mount | DeviceKind::Focuser | DeviceKind::FilterWheel => {
            if state.devices.read().unwrap().connection_id(kind).as_deref() != Some(id) {
                retire_auxiliary(state, kind);
            }
            return;
        }
        DeviceKind::Guider
        | DeviceKind::Rotator
        | DeviceKind::CoverCalibrator
        | DeviceKind::Switch
        | DeviceKind::Weather
        | DeviceKind::SafetyMonitor
        | DeviceKind::Dome => return,
    };
    match result {
        Ok((supported, snapshot)) => {
            state
                .devices
                .write()
                .unwrap()
                .checked(kind, id, supported, snapshot);
        }
        Err(error) => lose_device(state, kind, id, error),
    }
    // This includes a timeout that occurred while native code was returning.
    // A stale success does not revive it and no live pointer is retained.
    if state.devices.read().unwrap().connection_id(kind).as_deref() != Some(id) {
        match kind {
            DeviceKind::Camera => retire_camera(state),
            DeviceKind::Mount | DeviceKind::Focuser | DeviceKind::FilterWheel => {
                retire_auxiliary(state, kind)
            }
            DeviceKind::Guider
            | DeviceKind::Rotator
            | DeviceKind::CoverCalibrator
            | DeviceKind::Switch
            | DeviceKind::Weather
            | DeviceKind::SafetyMonitor
            | DeviceKind::Dome => {}
        }
    }
}

/// Ask a retired device's worker to disconnect (forced: the registry already
/// says it is gone, so nothing may refuse) and detach it from the hub.
fn retire_handle<C, S>(state: &AppState, handle: crate::equipment::handle::DeviceHandle<C, S>)
where
    C: Send + 'static,
    S: Send + Sync + 'static,
{
    state
        .equipment
        .detach(handle.kind(), handle.connection_id());
    state.equipment_ctx.forget(handle.kind());
    if let Ok(runtime) = tokio::runtime::Handle::try_current() {
        runtime.spawn(async move {
            if let Err(err) = handle.disconnect(true).await {
                tracing::warn!("retiring a lost device's worker failed: {err}");
            }
        });
    }
}

/// Auxiliary handles are published with their registry connection under the
/// slot lock (slot -> registry). Recheck after acquiring that same lock: a late
/// old probe must never destroy a newly connected replacement. The handle's
/// worker is told to disconnect off the slot and registry locks.
pub(crate) fn retire_auxiliary(state: &AppState, kind: DeviceKind) {
    match kind {
        DeviceKind::Mount => {
            let retired = {
                let mut slot = state.mount.lock().unwrap();
                if state.devices.read().unwrap().connection_id(kind).is_some() {
                    return;
                }
                let retired = slot.take();
                state.mount_name.lock().unwrap().take();
                retired
            };
            if let Some(handle) = retired {
                retire_handle(state, handle);
            }
        }
        DeviceKind::Focuser => {
            let retired = {
                let mut slot = state.focuser.lock().unwrap();
                if state.devices.read().unwrap().connection_id(kind).is_some() {
                    return;
                }
                let retired = slot.take();
                state.focuser_name.lock().unwrap().take();
                retired
            };
            if let Some(handle) = retired {
                retire_handle(state, handle);
            }
        }
        DeviceKind::FilterWheel => {
            let retired = {
                let mut slot = state.filter_wheel.lock().unwrap();
                if state.devices.read().unwrap().connection_id(kind).is_some() {
                    return;
                }
                slot.take()
            };
            if let Some(handle) = retired {
                retire_handle(state, handle);
            }
        }
        DeviceKind::Camera => {}
        // Equipment-layer handles live in the hub, which retires them itself.
        DeviceKind::Guider
        | DeviceKind::Rotator
        | DeviceKind::CoverCalibrator
        | DeviceKind::Switch
        | DeviceKind::Weather
        | DeviceKind::SafetyMonitor
        | DeviceKind::Dome => {}
    }
}

/// Called only on a blocking worker holding camera acquisition ownership.
pub(crate) fn retire_camera(state: &AppState) {
    if state.camera_runtime.read().unwrap().connection_id.is_some() {
        return;
    }
    let (camera, driver) = {
        let mut driver = state.camera_driver.lock().unwrap();
        let mut camera = state.camera.lock().unwrap();
        (camera.take(), driver.take())
    };
    state.camera_name.lock().unwrap().take();
    // Camera destruction quiesces callbacks before releasing their Rust data;
    // retain its driver until destruction has finished.
    drop(camera);
    drop(driver);
}

#[cfg(test)]
#[path = "health_tests.rs"]
mod lifecycle_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn late_probe_cannot_revive_old_device_or_replace_reconnection() {
        for kind in [
            DeviceKind::Camera,
            DeviceKind::Mount,
            DeviceKind::Focuser,
            DeviceKind::FilterWheel,
        ] {
            let mut devices = DeviceRegistry::default();
            devices.connected(kind, "old".into(), Some(json!({"name":"old"})));
            assert!(devices.disconnected(kind, Some("old"), Some("unplugged".into())));
            assert!(devices.snapshot(kind).is_none());
            assert!(!devices.checked(kind, "old", true, Some(json!({"name":"stale"}))));
            devices.connected(kind, "new".into(), Some(json!({"name":"new"})));
            assert!(!devices.disconnected(kind, Some("old"), Some("late timeout".into())));
            assert!(!devices.checked(kind, "old", true, None));
            assert_eq!(devices.snapshot(kind).unwrap()["name"], "new");
        }
    }

    #[test]
    fn device_kind_serde_names_are_the_contract_names() {
        assert_eq!(
            serde_json::to_value(DeviceKind::ALL).unwrap(),
            json!([
                "camera",
                "mount",
                "focuser",
                "filter_wheel",
                "guider",
                "rotator",
                "cover_calibrator",
                "switch",
                "weather",
                "safety_monitor",
                "dome",
            ])
        );
        for kind in DeviceKind::ALL {
            let text = serde_json::to_string(&kind).unwrap();
            assert_eq!(serde_json::from_str::<DeviceKind>(&text).unwrap(), kind);
        }
    }

    #[test]
    fn new_kinds_start_disconnected_unprobed_and_keep_the_connection_guard() {
        let devices = DeviceRegistry::default();
        assert_eq!(devices.0.len(), DeviceKind::ALL.len());
        let json = serde_json::to_value(&devices.0).unwrap();
        for kind in [
            DeviceKind::Guider,
            DeviceKind::Rotator,
            DeviceKind::CoverCalibrator,
            DeviceKind::Switch,
            DeviceKind::Weather,
            DeviceKind::SafetyMonitor,
            DeviceKind::Dome,
        ] {
            let health = &devices.0[&kind];
            assert!(!health.connected);
            assert_eq!(health.status, "disconnected");
            assert_eq!(health.probe_supported, None);
            let key = serde_json::to_value(kind).unwrap();
            assert_eq!(json[key.as_str().unwrap()]["status"], "disconnected");

            let mut devices = DeviceRegistry::default();
            devices.connected(kind, "old".into(), None);
            assert!(devices.disconnected(kind, Some("old"), None));
            assert!(!devices.checked(kind, "old", true, Some(json!({"name":"stale"}))));
            devices.connected(kind, "new".into(), Some(json!({"name":"new"})));
            assert!(!devices.disconnected(kind, Some("old"), None));
            assert_eq!(devices.snapshot(kind).unwrap()["name"], "new");
        }
        // The existing keys keep their order and shape.
        let keys: Vec<&String> = json.as_object().unwrap().keys().collect();
        assert!(keys.contains(&&"filter_wheel".to_string()));
    }

    #[test]
    fn unsupported_probe_is_explicit_and_never_fabricates_a_read() {
        let mut devices = DeviceRegistry::default();
        devices.connected(DeviceKind::Camera, "camera".into(), None);
        assert!(devices.checked(DeviceKind::Camera, "camera", false, None));
        let health = &devices.0[&DeviceKind::Camera];
        assert!(health.connected);
        assert_eq!(health.probe_supported, Some(false));
    }
}

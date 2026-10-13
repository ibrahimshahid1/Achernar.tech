//! Passive device liveness and one authoritative disconnect lifecycle.
//!
//! Probes never expose, slew, focus, or move a filter. Each device has at most
//! one blocking SDK call in flight. A deadline invalidates its public state;
//! it does not cancel native code or free a handle while native code uses it.

use std::collections::BTreeMap;
use std::sync::{atomic::Ordering, Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::state::{AppState, CameraRuntimeState};

const POLL_INTERVAL: Duration = Duration::from_secs(2);
const PROBE_DEADLINE: Duration = Duration::from_secs(3);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind {
    Camera,
    Mount,
    Focuser,
    FilterWheel,
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
            snapshot: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeviceRegistry(pub BTreeMap<DeviceKind, DeviceHealth>);

impl Default for DeviceRegistry {
    fn default() -> Self {
        Self(
            [
                DeviceKind::Camera,
                DeviceKind::Mount,
                DeviceKind::Focuser,
                DeviceKind::FilterWheel,
            ]
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
        health.last_error = error;
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

pub fn mount_snapshot(name: String, hw: &super::mount::MountStatus) -> Value {
    json!({"connected": true, "name": name, "ra": hw.position.map(|p| p.ra),
        "dec": hw.position.map(|p| p.dec), "tracking": hw.tracking,
        "parked": null, "pier_side": null})
}

pub fn focuser_snapshot(name: String, hw: &super::focuser::FocuserStatus) -> Value {
    json!({"connected": true, "name": name, "position": hw.position,
        "moving": hw.is_moving, "absolute": hw.is_absolute, "max_range": hw.max_range})
}

/// Call while holding the corresponding device slot, immediately before a
/// command. A retained native handle is not a current connection after loss.
pub fn require_connected(state: &AppState, kind: DeviceKind) -> Result<(), String> {
    if state.devices.read().unwrap().connection_id(kind).is_none() {
        return Err(format!(
            "{kind:?} is disconnected; reconnect before issuing commands"
        ));
    }
    Ok(())
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
            for kind in [
                DeviceKind::Camera,
                DeviceKind::Mount,
                DeviceKind::Focuser,
                DeviceKind::FilterWheel,
            ] {
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
        DeviceKind::Mount => {
            let Ok(mount) = state.mount.try_lock() else {
                return;
            };
            if state.devices.read().unwrap().connection_id(kind).as_deref() != Some(id) {
                drop(mount);
                retire_auxiliary(state, kind);
                return;
            }
            let Some(mount) = mount.as_ref() else { return };
            mount
                .health_status()
                .map(|hw| {
                    (
                        true,
                        Some(mount_snapshot(
                            state
                                .mount_name
                                .lock()
                                .unwrap()
                                .clone()
                                .unwrap_or_else(|| "Mount".into()),
                            &hw,
                        )),
                    )
                })
                .map_err(|e| e.to_string())
        }
        DeviceKind::Focuser => {
            let Ok(focuser) = state.focuser.try_lock() else {
                return;
            };
            if state.devices.read().unwrap().connection_id(kind).as_deref() != Some(id) {
                drop(focuser);
                retire_auxiliary(state, kind);
                return;
            }
            let Some(focuser) = focuser.as_ref() else {
                return;
            };
            focuser
                .health_status()
                .map(|hw| {
                    (
                        true,
                        Some(focuser_snapshot(
                            state
                                .focuser_name
                                .lock()
                                .unwrap()
                                .clone()
                                .unwrap_or_else(|| "Focuser".into()),
                            &hw,
                        )),
                    )
                })
                .map_err(|e| e.to_string())
        }
        DeviceKind::FilterWheel => {
            let Ok(wheel) = state.filter_wheel.try_lock() else {
                return;
            };
            if state.devices.read().unwrap().connection_id(kind).as_deref() != Some(id) {
                drop(wheel);
                retire_auxiliary(state, kind);
                return;
            }
            let Some(wheel) = wheel.as_ref() else { return };
            wheel
                .health_status()
                .map(|snapshot| (true, Some(snapshot)))
                .map_err(|e| e.to_string())
        }
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
        }
    }
}

/// Auxiliary handles are published with their registry connection under the
/// slot lock (slot -> registry). Recheck after acquiring that same lock: a late
/// old probe must never destroy a newly connected replacement. Drop native
/// handles outside registry/slot locks, on this blocking worker.
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
            drop(retired);
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
            drop(retired);
        }
        DeviceKind::FilterWheel => {
            let retired = {
                let mut slot = state.filter_wheel.lock().unwrap();
                if state.devices.read().unwrap().connection_id(kind).is_some() {
                    return;
                }
                slot.take()
            };
            drop(retired);
        }
        DeviceKind::Camera => {}
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
    fn unsupported_probe_is_explicit_and_never_fabricates_a_read() {
        let mut devices = DeviceRegistry::default();
        devices.connected(DeviceKind::Camera, "camera".into(), None);
        assert!(devices.checked(DeviceKind::Camera, "camera", false, None));
        let health = &devices.0[&DeviceKind::Camera];
        assert!(health.connected);
        assert_eq!(health.probe_supported, Some(false));
    }
}

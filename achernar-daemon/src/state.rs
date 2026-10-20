//! Shared application state for the Achernar daemon.
//!
//! This module provides `AppState` which is shared across all Axum route handlers
//! via `axum::extract::State`. It holds references to hardware (camera, mount, focuser)
//! and the latest captured frame for streaming.

use crate::hardware::{
    camera::{FrameData, OlsCamera, OlsCameraDriver},
    focuser::ConnectedFocuser,
    mount::Mount,
};
use crate::vision::observed_frame::ObservedFrame;
use crate::vision::shared_observation::SharedObservation;
use crate::{
    config::FocusConfig, focus::focus_agent::FocusAgent, focus::focus_drift::FocusDriftForecaster,
    storage::frames_repo::FrameStore, storage::local_store::LocalStore,
};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

/// Frontend-facing lifecycle of the one camera currently owned by the daemon.
///
/// Keep this separate from the legacy `streaming` boolean. A preview stream is
/// not an exposure, and treating the two as synonyms makes it impossible for a
/// client to show truthful progress or decide whether an operation can be
/// cancelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraActivity {
    Disconnected,
    Connecting,
    Idle,
    Streaming,
    Exposing,
    Downloading,
    Processing,
    Disconnecting,
    Error,
}

/// Small, lock-independent snapshot of camera control state.
///
/// Hardware calls may block inside a vendor SDK for seconds or minutes. Status
/// and telemetry endpoints read this snapshot instead of waiting on the camera
/// mutex, so a web client remains responsive during a long exposure.
#[derive(Debug, Clone, Serialize)]
pub struct CameraRuntimeState {
    pub connection_id: Option<String>,
    pub activity: CameraActivity,
    pub streaming: bool,
    pub driver_id: Option<String>,
    pub device_index: Option<i32>,
    pub camera_name: Option<String>,
    pub active_format_index: Option<usize>,
    pub operation_id: Option<String>,
    pub exposure_started_at: Option<chrono::DateTime<chrono::Utc>>,
    pub exposure_duration_s: Option<f64>,
    pub temperature_c: Option<f64>,
    pub cooler_on: Option<bool>,
    pub last_error: Option<String>,
}

impl Default for CameraRuntimeState {
    fn default() -> Self {
        Self {
            connection_id: None,
            activity: CameraActivity::Disconnected,
            streaming: false,
            driver_id: None,
            device_index: None,
            camera_name: None,
            active_format_index: None,
            operation_id: None,
            exposure_started_at: None,
            exposure_duration_s: None,
            temperature_c: None,
            cooler_on: None,
            last_error: None,
        }
    }
}

impl CameraRuntimeState {
    /// Progress is deliberately an estimate: many camera SDKs report no
    /// download progress. It is still useful during the timed exposure, and is
    /// omitted for every state where a percentage would be invented.
    pub fn exposure_progress(&self, now: chrono::DateTime<chrono::Utc>) -> Option<f64> {
        if self.activity != CameraActivity::Exposing {
            return None;
        }
        let started = self.exposure_started_at?;
        let duration = self.exposure_duration_s?;
        if !duration.is_finite() || duration <= 0.0 {
            return None;
        }
        let elapsed_ms = now.signed_duration_since(started).num_milliseconds().max(0) as f64;
        Some((elapsed_ms / 1000.0 / duration).clamp(0.0, 1.0))
    }
}

/// Shared application state injected into all Axum route handlers.
///
/// # Lock order
///
/// Several handlers need more than one of these mutexes at once. Take them in
/// this order and never in any other:
///
/// ```text
/// acquisition → camera_driver → camera → camera_name → streaming
///             → camera_observation_barrier → latest_frame → camera_runtime
///             → focus_agent → focus_drift
/// ```
///
/// `focus_drift` is deliberately last. The forecaster is a leaf: it needs no
/// camera, no focuser and no agent to do its work, so putting it at the tail
/// means any handler may take it while already holding something else, and it
/// can never be the outer lock in a cycle. The one path that holds two —
/// autofocus completion, which has `focus_agent` and then folds the converged
/// position into the filter — satisfies the order without doing anything
/// special.
///
/// `plate_solve_trigger` is a leaf in the same sense: nothing is acquired
/// while it is held, so it may be taken under any of the above.
///
/// This is not a style preference. `get_status` used to hold `camera_name` and
/// `streaming` while reaching for `camera`, at the same time as `start_stream`
/// held `camera` while reaching for `streaming` — a textbook ABBA deadlock that
/// wedged the whole daemon whenever a status poll landed during a stream start.
///
/// Prefer not to nest at all: copy the value out and drop the guard.
/// `*state.streaming.lock().unwrap() = true;` creates a temporary guard that is
/// released at the end of the statement and is always safe; a `let` binding is
/// what holds a lock open across other acquisitions.
///
/// Never hold any of these `std::sync::Mutex` guards across an `.await`.
#[derive(Clone)]
pub struct AppState {
    /// Serialises anything that commands the camera: still capture, stream
    /// start/stop, parameter writes that must not interleave, disconnect.
    ///
    /// A `tokio::sync::Mutex` because it is deliberately held across `.await`
    /// — a single exposure can take minutes, and two callers racing to set the
    /// shutter and trigger would otherwise produce a frame whose metadata
    /// belongs to the other request.
    pub acquisition: Arc<tokio::sync::Mutex<()>>,

    /// Camera driver (manages enumeration and camera lifecycle)
    pub camera_driver: Arc<Mutex<Option<OlsCameraDriver>>>,

    /// Currently active camera
    pub camera: Arc<Mutex<Option<OlsCamera>>>,

    /// Currently active telescope mount
    pub mount: Arc<Mutex<Option<Mount>>>,

    /// Human-readable mount name/driver for status reporting
    pub mount_name: Arc<Mutex<Option<String>>>,

    /// Currently active focuser
    pub focuser: Arc<Mutex<Option<ConnectedFocuser>>>,

    /// Human-readable focuser name/driver for status reporting
    pub focuser_name: Arc<Mutex<Option<String>>>,

    /// Filter-wheel drivers are not implemented yet. The lifecycle slot still
    /// participates in the same null-inclusive status/health protocol.
    pub filter_wheel: Arc<Mutex<Option<crate::hardware::filter_wheel::FilterWheel>>>,

    /// Passive liveness and cached auxiliary-device measurements. Telemetry
    /// never enters a vendor SDK or waits for an unresponsive device.
    pub devices: Arc<RwLock<crate::hardware::health::DeviceRegistry>>,
    pub device_updates: tokio::sync::broadcast::Sender<()>,

    /// Latest frame received from the camera stream
    pub latest_frame: Arc<Mutex<Option<FrameData>>>,

    /// Monotonic identity of the active camera stream. Late callbacks and
    /// vision work from an invalidated stream are discarded by generation.
    pub camera_generation: Arc<AtomicU64>,

    /// Makes stream invalidation atomic with publication of derived vision
    /// state. Generation checks alone leave a small check-then-publish window:
    /// a disconnect can clear the buses after the check and vision can then
    /// republish the old camera's result. Only lifecycle clearing and the
    /// vision publication commit take this lock; frame callbacks do not.
    pub camera_observation_barrier: Arc<Mutex<()>>,

    /// Latest shared perception bus observation. Controllers read this, not raw pixels.
    pub vision_observation: Arc<RwLock<Option<SharedObservation>>>,

    /// Latest frame+observation envelope. Autofocus persistence reads this so the
    /// saved pixels provably match their label, and keys freshness on the camera
    /// frame counter rather than processing time.
    pub last_observed_frame: Arc<RwLock<Option<ObservedFrame>>>,

    /// Broadcast stream for VISION_UPDATE websocket events.
    pub vision_updates: tokio::sync::broadcast::Sender<SharedObservation>,

    /// Camera name (for status reporting)
    pub camera_name: Arc<Mutex<Option<String>>>,

    /// Whether the camera stream is active
    pub streaming: Arc<Mutex<bool>>,

    /// Non-blocking identity/activity snapshot for API and telemetry clients.
    /// Never hold this lock while calling into a camera driver.
    pub camera_runtime: Arc<RwLock<CameraRuntimeState>>,

    /// Deterministic autofocus state machine
    pub focus_agent: Arc<tokio::sync::Mutex<FocusAgent>>,

    /// Cross-task abort signal for a running deterministic autofocus sweep
    pub focus_abort_requested: Arc<AtomicBool>,

    /// Temperature-compensated focus drift forecaster (Section 7.6.3).
    ///
    /// `None` only when `[focus.drift] enabled = false`, or when a config value
    /// is present and invalid. Missing optics do **not** land here: the filter
    /// runs and estimates without them and reports
    /// `AdviceStatus::ThresholdUnknown`, because refusing to construct it would
    /// substitute "off" for "unknown" and throw away the expensive half of the
    /// model to save the cheap half.
    ///
    /// Three states the operator can be in, all reported distinctly by
    /// `/focus/drift`: not enabled, enabled but untrained (`update_count == 0`),
    /// and enabled and estimating but with no threshold to advise against.
    ///
    /// A `tokio::sync::Mutex` for consistency with `focus_agent`, though the
    /// filter maths itself is synchronous: take the lock, update or predict,
    /// copy the result out, drop the guard, and only then await the SQLite
    /// write. Nothing needs the guard held across an `.await`.
    pub focus_drift: Option<Arc<tokio::sync::Mutex<FocusDriftForecaster>>>,

    /// Where the learned filter state is snapshotted, when persistence is on.
    pub focus_drift_state_path: Option<std::path::PathBuf>,

    /// Optics from this rig's `rigs` row, read once at startup.
    ///
    /// `pixel_size_um` in particular has no `[focus.drift]` key: it is a camera
    /// fact this column already holds, and a second place to type it in would
    /// be the friction the V-curve measurement exists to remove.
    pub rig_optics: Option<crate::config::RigOptics>,

    /// Focus configuration loaded from achernar.toml
    pub focus_config: FocusConfig,

    /// Camera configuration loaded from achernar.toml.
    ///
    /// Carries `saturation_adu`, which nothing in the pipeline can derive and
    /// only the operator knows. Every stream start stamps it onto the frames it
    /// produces rather than consulting this at measurement time, so a rig with
    /// two cameras can hold two different answers at once.
    pub camera_config: crate::config::CameraConfig,

    /// Durable offline-first SQLite store
    pub local_store: LocalStore,

    /// Preserves autofocus/preview frames on disk + in the `frames` table so the
    /// ML pipeline can recover (crop, label) training pairs.
    pub frame_store: FrameStore,

    /// Canonical destination for physical exposures: one `frames` row plus
    /// immutable `frame_assets`, staged and committed transactionally.
    /// `frame_store` above handles the analysis-only luma path; anything the
    /// shutter produced goes here.
    pub capture_store: crate::storage::capture_store::CaptureStore,

    /// Owns submitted exposure bundles independently of request lifetimes.
    /// Acquisition must reserve a slot before issuing new science commands.
    pub capture_archive: crate::storage::capture_archive::CaptureArchive,

    /// Identifies this rig in the archive. Every capture is filed under it, and
    /// dataset splits group by it, so a wrong value silently merges two rigs'
    /// data into one training set.
    pub rig_id: String,

    /// FQE ONNX runner (shadow mode). Loaded from `focus.models.quality_model_path`
    /// or the newest `models/focus_quality/<semver>/model.onnx`; disabled when
    /// neither exists. The classical measurement stays the control authority.
    pub fqe_runner: Arc<crate::vision::fqe_runner::FqeRunner>,
    /// Rendered-preview research diagnostics; never supplies control observations.
    pub coc_shadow: Arc<crate::vision::coc_shadow::CocShadowService>,

    /// ASTAP plate solver for archived frames. Disabled until
    /// [`AppState::with_plate_solver`] installs one built from `[plate_solve]`.
    pub plate_solver: Arc<crate::processing::plate_solver::PlateSolver>,

    /// Which archived frames get solved without being asked (startup and
    /// post-slew rules). Disabled until [`AppState::with_auto_solve`].
    pub plate_solve_trigger: Arc<Mutex<crate::mount::plate_solve_trigger::PlateSolveTrigger>>,
}

impl AppState {
    /// `rig_optics` comes from the `rigs` row and is consulted for the refocus
    /// threshold before the operator is asked to repeat those facts in TOML.
    ///
    /// `data_dir` is where mutable, rig-specific runtime state is snapshotted —
    /// currently the learned focus-drift filter. Deliberately not the models
    /// directory: what one telescope learned last night is not a shipped
    /// artefact and must never end up in version control.
    pub fn new(
        focus_config: FocusConfig,
        camera_config: crate::config::CameraConfig,
        local_store: LocalStore,
        frame_store: FrameStore,
        capture_store: crate::storage::capture_store::CaptureStore,
        rig_id: String,
        data_dir: std::path::PathBuf,
        rig_optics: Option<crate::config::RigOptics>,
    ) -> Self {
        let (vision_updates, _) = tokio::sync::broadcast::channel(64);
        let (device_updates, _) = tokio::sync::broadcast::channel(16);
        let capture_archive = crate::storage::capture_archive::CaptureArchive::spawn(
            capture_store.clone(),
            local_store.pool().clone(),
            2,
        );

        // Resolve the FQE shadow model: explicit config path wins; otherwise the
        // newest semver under <models_root>/focus_quality/. Any load failure
        // degrades to a disabled runner — never a crashed daemon.
        let models_root =
            std::env::var("ACHERNAR_MODELS_DIR").unwrap_or_else(|_| "models".to_string());
        let fqe_path = focus_config.models.quality_model_path.clone().or_else(|| {
            crate::vision::fqe_runner::resolve_latest_model(&models_root)
                .map(|p| p.to_string_lossy().to_string())
        });
        let fqe_runner = Arc::new(
            crate::vision::fqe_runner::FqeRunner::new(fqe_path).unwrap_or_else(|err| {
                tracing::warn!("FQE runner failed to load ({err}); shadow inference disabled");
                crate::vision::fqe_runner::FqeRunner::disabled()
            }),
        );

        // Resolve the focus-drift forecaster. Any misconfiguration disables it
        // with an explanation rather than substituting defaults: a threshold
        // nobody chose produces refocus advice nobody can interpret.
        let (focus_drift, focus_drift_state_path) =
            Self::build_focus_drift(&focus_config.drift, rig_optics.as_ref(), &data_dir);
        let coc_shadow = Arc::new(crate::vision::coc_shadow::CocShadowService::from_env(&data_dir));

        Self {
            acquisition: Arc::new(tokio::sync::Mutex::new(())),
            camera_driver: Arc::new(Mutex::new(None)),
            camera: Arc::new(Mutex::new(None)),
            mount: Arc::new(Mutex::new(None)),
            mount_name: Arc::new(Mutex::new(None)),
            focuser: Arc::new(Mutex::new(None)),
            focuser_name: Arc::new(Mutex::new(None)),
            filter_wheel: Arc::new(Mutex::new(None)),
            devices: Arc::new(RwLock::new(crate::hardware::health::DeviceRegistry::default())),
            device_updates,
            latest_frame: Arc::new(Mutex::new(None)),
            camera_generation: Arc::new(AtomicU64::new(0)),
            camera_observation_barrier: Arc::new(Mutex::new(())),
            vision_observation: Arc::new(RwLock::new(None)),
            last_observed_frame: Arc::new(RwLock::new(None)),
            vision_updates,
            camera_name: Arc::new(Mutex::new(None)),
            streaming: Arc::new(Mutex::new(false)),
            camera_runtime: Arc::new(RwLock::new(CameraRuntimeState::default())),
            focus_agent: Arc::new(tokio::sync::Mutex::new(FocusAgent::new())),
            focus_abort_requested: Arc::new(AtomicBool::new(false)),
            focus_drift,
            focus_drift_state_path,
            rig_optics,
            focus_config,
            camera_config,
            local_store,
            frame_store,
            capture_store,
            capture_archive,
            rig_id,
            fqe_runner,
            coc_shadow,
            plate_solver: Arc::new(crate::processing::plate_solver::PlateSolver::disabled(
                "No plate solver was configured for this daemon instance",
            )),
            plate_solve_trigger: Arc::new(Mutex::new(
                crate::mount::plate_solve_trigger::PlateSolveTrigger::disabled(),
            )),
        }
    }

    /// Arm the automatic solve rules from `[plate_solve]`, with the startup
    /// rule pending.
    pub fn with_auto_solve(self, config: &crate::config::PlateSolveConfig) -> Self {
        if config.auto_solve {
            *self.plate_solve_trigger.lock().unwrap() =
                crate::mount::plate_solve_trigger::PlateSolveTrigger::new(
                    std::time::Duration::from_secs(config.auto_solve_retry_s),
                );
        }
        self
    }

    /// Install the plate solver resolved from `[plate_solve]`.
    pub fn with_plate_solver(
        mut self,
        solver: crate::processing::plate_solver::PlateSolver,
    ) -> Self {
        self.plate_solver = Arc::new(solver);
        self
    }

    /// Remove every value derived from a previous stream generation.
    ///
    /// Camera frame counters commonly restart at zero after reconnect. Leaving
    /// an older frame in the mailbox can therefore make a fresh capture wait
    /// forever for a counter greater than the previous camera's value, while
    /// preview and vision clients briefly display pixels from the wrong device.
    pub fn clear_camera_observations(&self) {
        let _barrier = self.camera_observation_barrier.lock().unwrap();
        self.camera_generation.fetch_add(1, Ordering::AcqRel);
        *self.latest_frame.lock().unwrap() = None;
        *self.vision_observation.write().unwrap() = None;
        *self.last_observed_frame.write().unwrap() = None;
    }

    pub fn current_camera_generation(&self) -> u64 {
        self.camera_generation.load(Ordering::Acquire)
    }

    /// Native callbacks receive only these publication signals, never an
    /// owning camera handle. They may invalidate state but never join or free
    /// their own SDK worker.
    pub fn camera_loss_signals(&self) -> crate::hardware::health::CameraLossSignals {
        crate::hardware::health::CameraLossSignals {
            runtime: self.camera_runtime.clone(),
            generation: self.camera_generation.clone(),
            barrier: self.camera_observation_barrier.clone(),
            latest_frame: self.latest_frame.clone(),
            observation: self.vision_observation.clone(),
            observed_frame: self.last_observed_frame.clone(),
            streaming: self.streaming.clone(),
            focus_abort: self.focus_abort_requested.clone(),
            devices: self.devices.clone(),
            updates: self.device_updates.clone(),
        }
    }

    /// Construct the drift forecaster from config, restoring any snapshot.
    ///
    /// Missing optics do **not** disable it. Estimation — learning this rig's
    /// drift rate and temperature coefficient — needs no optical constants, and
    /// it is the half that costs sky: the coefficient is only identifiable
    /// across varied thermal conditions over several sessions. Switching the
    /// filter off until someone edits a TOML file would throw that away and
    /// then start from the diffuse prior on the night they finally did.
    ///
    /// So an unconfigured threshold yields a running filter that reports
    /// `AdviceStatus::ThresholdUnknown`. Only a *wrong* value (present and
    /// invalid) disables it, because that is a mistake to surface rather than
    /// an absence to model.
    fn build_focus_drift(
        cfg: &crate::config::FocusDriftConfig,
        rig_optics: Option<&crate::config::RigOptics>,
        data_dir: &std::path::Path,
    ) -> (
        Option<Arc<tokio::sync::Mutex<FocusDriftForecaster>>>,
        Option<std::path::PathBuf>,
    ) {
        if !cfg.enabled {
            return (None, None);
        }

        let (drift_config, threshold_source) = match cfg.to_drift_config(rig_optics) {
            Ok(c) => c,
            Err(err) => {
                tracing::warn!(
                    "Focus drift forecaster disabled: {err}. This is a value that is \
                     present and invalid, not one that is missing — fix it in \
                     [focus.drift] and the forecaster will start."
                );
                return (None, None);
            }
        };
        let threshold_source =
            threshold_source.unwrap_or(crate::focus::focus_drift::ThresholdSource::Configured);

        let state_path = cfg.state_path.as_ref().map(|p| {
            let candidate = std::path::PathBuf::from(p);
            if candidate.is_absolute() {
                candidate
            } else {
                data_dir.join(candidate)
            }
        });

        // Restore a snapshot when one exists. A corrupt or stale file starts a
        // fresh filter rather than aborting startup — the cost of relearning is
        // one night of measurements, the cost of refusing to boot is the night.
        let forecaster = match state_path.as_ref().filter(|p| p.exists()) {
            Some(path) => match std::fs::read_to_string(path)
                .map_err(|e| e.to_string())
                .and_then(|text| {
                    serde_json::from_str::<crate::focus::focus_drift::DriftState>(&text)
                        .map_err(|e| e.to_string())
                })
                .and_then(|snapshot| {
                    FocusDriftForecaster::from_state(snapshot, drift_config.clone())
                }) {
                Ok(f) => {
                    tracing::info!(
                        "Focus drift forecaster restored from {} ({} update(s))",
                        path.display(),
                        f.update_count()
                    );
                    f
                }
                Err(err) => {
                    tracing::warn!(
                        "Focus drift snapshot at {} unusable ({err}); starting a fresh filter",
                        path.display()
                    );
                    FocusDriftForecaster::with_config_and_source(drift_config, threshold_source)
                }
            },
            None => FocusDriftForecaster::with_config_and_source(drift_config, threshold_source),
        };

        match forecaster.threshold() {
            Some((steps, source)) => tracing::info!(
                "Focus drift forecaster enabled: refocus threshold {steps:.1} steps ({source:?}), \
                 {} prior update(s)",
                forecaster.update_count()
            ),
            None => tracing::info!(
                "Focus drift forecaster enabled and estimating, but with NO refocus threshold: \
                 drift and its uncertainty will be reported, refocus advice will not. Set \
                 [focus.drift] steps_per_micron (with f_ratio here or focal_ratio on the rigs \
                 row), or let an autofocus V-curve measure the focuser scale — that needs \
                 rigs.pixel_size_um to be set."
            ),
        }

        (
            Some(Arc::new(tokio::sync::Mutex::new(forecaster))),
            state_path,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase1_camera_activity_has_stable_snake_case_json_values() {
        let cases = [
            (CameraActivity::Disconnected, "disconnected"),
            (CameraActivity::Connecting, "connecting"),
            (CameraActivity::Idle, "idle"),
            (CameraActivity::Streaming, "streaming"),
            (CameraActivity::Exposing, "exposing"),
            (CameraActivity::Downloading, "downloading"),
            (CameraActivity::Processing, "processing"),
            (CameraActivity::Disconnecting, "disconnecting"),
            (CameraActivity::Error, "error"),
        ];

        for (activity, expected) in cases {
            assert_eq!(serde_json::to_value(activity).unwrap(), expected);
            assert_eq!(
                serde_json::from_value::<CameraActivity>(serde_json::json!(expected)).unwrap(),
                activity
            );
        }
    }

    #[test]
    fn phase1_camera_exposure_progress_is_clamped_and_only_exists_while_exposing() {
        let start = chrono::Utc::now();
        let mut runtime = CameraRuntimeState {
            activity: CameraActivity::Exposing,
            exposure_started_at: Some(start),
            exposure_duration_s: Some(10.0),
            ..CameraRuntimeState::default()
        };

        assert_eq!(
            runtime.exposure_progress(start - chrono::Duration::seconds(2)),
            Some(0.0)
        );
        assert_eq!(
            runtime.exposure_progress(start + chrono::Duration::milliseconds(2500)),
            Some(0.25)
        );
        assert_eq!(
            runtime.exposure_progress(start + chrono::Duration::seconds(20)),
            Some(1.0)
        );

        runtime.activity = CameraActivity::Downloading;
        assert_eq!(
            runtime.exposure_progress(start + chrono::Duration::seconds(5)),
            None
        );
    }

    #[test]
    fn phase1_camera_exposure_progress_rejects_missing_or_invalid_timing() {
        let now = chrono::Utc::now();
        let mut runtime = CameraRuntimeState {
            activity: CameraActivity::Exposing,
            ..CameraRuntimeState::default()
        };

        assert_eq!(runtime.exposure_progress(now), None);

        runtime.exposure_started_at = Some(now);
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            runtime.exposure_duration_s = Some(invalid);
            assert_eq!(
                runtime.exposure_progress(now),
                None,
                "duration {invalid:?} must not produce invented progress"
            );
        }
    }
}

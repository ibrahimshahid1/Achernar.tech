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
    config::FocusConfig, focus::focus_agent::FocusAgent, storage::frames_repo::FrameStore,
    storage::local_store::LocalStore,
};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, RwLock};

/// Shared application state injected into all Axum route handlers.
#[derive(Clone)]
pub struct AppState {
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

    /// Latest frame received from the camera stream
    pub latest_frame: Arc<Mutex<Option<FrameData>>>,

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

    /// Deterministic autofocus state machine
    pub focus_agent: Arc<tokio::sync::Mutex<FocusAgent>>,

    /// Cross-task abort signal for a running deterministic autofocus sweep
    pub focus_abort_requested: Arc<AtomicBool>,

    /// Focus configuration loaded from achernar.toml
    pub focus_config: FocusConfig,

    /// Durable offline-first SQLite store
    pub local_store: LocalStore,

    /// Preserves autofocus/preview frames on disk + in the `frames` table so the
    /// ML pipeline can recover (crop, label) training pairs.
    pub frame_store: FrameStore,

    /// FQE ONNX runner (shadow mode). Loaded from `focus.models.quality_model_path`
    /// or the newest `models/focus_quality/<semver>/model.onnx`; disabled when
    /// neither exists. The classical measurement stays the control authority.
    pub fqe_runner: Arc<crate::vision::fqe_runner::FqeRunner>,
}

impl AppState {
    pub fn new(
        focus_config: FocusConfig,
        local_store: LocalStore,
        frame_store: FrameStore,
    ) -> Self {
        let (vision_updates, _) = tokio::sync::broadcast::channel(64);

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

        Self {
            camera_driver: Arc::new(Mutex::new(None)),
            camera: Arc::new(Mutex::new(None)),
            mount: Arc::new(Mutex::new(None)),
            mount_name: Arc::new(Mutex::new(None)),
            focuser: Arc::new(Mutex::new(None)),
            focuser_name: Arc::new(Mutex::new(None)),
            latest_frame: Arc::new(Mutex::new(None)),
            vision_observation: Arc::new(RwLock::new(None)),
            last_observed_frame: Arc::new(RwLock::new(None)),
            vision_updates,
            camera_name: Arc::new(Mutex::new(None)),
            streaming: Arc::new(Mutex::new(false)),
            focus_agent: Arc::new(tokio::sync::Mutex::new(FocusAgent::new())),
            focus_abort_requested: Arc::new(AtomicBool::new(false)),
            focus_config,
            local_store,
            frame_store,
            fqe_runner,
        }
    }
}

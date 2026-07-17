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
use crate::vision::shared_observation::SharedObservation;
use crate::{
    config::FocusConfig, focus::focus_agent::FocusAgent, storage::local_store::LocalStore,
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
}

impl AppState {
    pub fn new(focus_config: FocusConfig, local_store: LocalStore) -> Self {
        let (vision_updates, _) = tokio::sync::broadcast::channel(64);
        Self {
            camera_driver: Arc::new(Mutex::new(None)),
            camera: Arc::new(Mutex::new(None)),
            mount: Arc::new(Mutex::new(None)),
            mount_name: Arc::new(Mutex::new(None)),
            focuser: Arc::new(Mutex::new(None)),
            focuser_name: Arc::new(Mutex::new(None)),
            latest_frame: Arc::new(Mutex::new(None)),
            vision_observation: Arc::new(RwLock::new(None)),
            vision_updates,
            camera_name: Arc::new(Mutex::new(None)),
            streaming: Arc::new(Mutex::new(false)),
            focus_agent: Arc::new(tokio::sync::Mutex::new(FocusAgent::new())),
            focus_abort_requested: Arc::new(AtomicBool::new(false)),
            focus_config,
            local_store,
        }
    }
}

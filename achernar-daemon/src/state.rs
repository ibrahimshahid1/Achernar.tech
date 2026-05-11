//! Shared application state for the Achernar daemon.
//!
//! This module provides `AppState` which is shared across all Axum route handlers
//! via `axum::extract::State`. It holds references to hardware (camera, mount, focuser)
//! and the latest captured frame for streaming.

use crate::hardware::{
    camera::{FrameData, OlsCamera, OlsCameraDriver},
    focuser::Focuser,
    mount::Mount,
};
use std::sync::{Arc, Mutex};

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
    pub focuser: Arc<Mutex<Option<Focuser>>>,

    /// Human-readable focuser name/driver for status reporting
    pub focuser_name: Arc<Mutex<Option<String>>>,

    /// Latest frame received from the camera stream
    pub latest_frame: Arc<Mutex<Option<FrameData>>>,

    /// Camera name (for status reporting)
    pub camera_name: Arc<Mutex<Option<String>>>,

    /// Whether the camera stream is active
    pub streaming: Arc<Mutex<bool>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            camera_driver: Arc::new(Mutex::new(None)),
            camera: Arc::new(Mutex::new(None)),
            mount: Arc::new(Mutex::new(None)),
            mount_name: Arc::new(Mutex::new(None)),
            focuser: Arc::new(Mutex::new(None)),
            focuser_name: Arc::new(Mutex::new(None)),
            latest_frame: Arc::new(Mutex::new(None)),
            camera_name: Arc::new(Mutex::new(None)),
            streaming: Arc::new(Mutex::new(false)),
        }
    }
}

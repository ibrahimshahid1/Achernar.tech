//! The `CameraActivity` adapter over the OLS camera state (spec §2.4, §3.12,
//! A06). Owner: WP1B-1.
//!
//! Cameras stay on OpenLiveStacker (O1). The equipment layer only needs to
//! know whether an exposure is integrating (the focuser and filter-wheel
//! checks refuse motion then), what that exposure is (windows, clock
//! quality), how to abort it (e-stop, the safety policy) and the camera's
//! name (the PHD2 guide-camera role check). All of it is read from
//! `camera_runtime`, the lock-independent snapshot the camera API keeps; the
//! adapter never enters a vendor SDK and never takes a camera lock.
//!
//! Aborting: the current camera backends cannot interrupt an exposure in
//! progress (`/camera/abort` stops the live stream and reports
//! `exposure_abort_unsupported` while exposing). The adapter says so
//! honestly instead of reporting an abort that did not happen; with no
//! exposure running there is nothing to abort and it succeeds.

use std::sync::{Arc, RwLock};

use super::snapshot::ExposureRef;
use super::traits::{CameraActivity, Clock};
use super::types::{ClockQuality, DeviceError, DeviceResult, Observed, Source};
use crate::hardware::health::{DeviceKind, DeviceRegistry};
use crate::state::{CameraActivity as CameraState, CameraRuntimeState};

/// Exposure clock quality per OLS driver id (§2.14).
pub fn clock_quality(driver_id: Option<&str>) -> ClockQuality {
    match driver_id {
        Some("sim") => ClockQuality::Simulator,
        Some("gphoto2" | "dslr") => ClockQuality::HostTriggerBracket,
        Some("uvc" | "webcam" | "wdir" | "watchdir") => ClockQuality::HostArrival,
        Some("asi" | "svb" | "toup" | "alpaca" | "indi" | "indigo") => {
            ClockQuality::DriverExposureStart
        }
        _ => ClockQuality::Unknown,
    }
}

/// The camera hook installed on the hub by `AppState`.
pub struct OlsCameraHook {
    runtime: Arc<RwLock<CameraRuntimeState>>,
    devices: Arc<RwLock<DeviceRegistry>>,
    clock: Arc<dyn Clock>,
}

impl OlsCameraHook {
    pub fn new(
        runtime: Arc<RwLock<CameraRuntimeState>>,
        devices: Arc<RwLock<DeviceRegistry>>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            runtime,
            devices,
            clock,
        }
    }

    fn runtime(&self) -> CameraRuntimeState {
        self.runtime
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// The camera the registry still calls connected.
    fn connected(&self, runtime: &CameraRuntimeState) -> bool {
        let registered = self
            .devices
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .connection_id(DeviceKind::Camera);
        registered.is_some() && registered == runtime.connection_id
    }
}

impl CameraActivity for OlsCameraHook {
    /// Known false with no camera (nothing can integrate); known from the
    /// runtime activity otherwise. A camera in an error or transitional
    /// state is unknown, never assumed idle.
    fn exposing(&self) -> Observed<bool> {
        let runtime = self.runtime();
        let at = |value: bool| Observed::fresh_at(value, Source::Derived, self.clock.as_ref());
        if !self.connected(&runtime) {
            return at(false);
        }
        match runtime.activity {
            CameraState::Exposing => at(true),
            CameraState::Idle
            | CameraState::Streaming
            | CameraState::Downloading
            | CameraState::Processing
            | CameraState::Disconnected => at(false),
            CameraState::Connecting | CameraState::Disconnecting | CameraState::Error => {
                Observed::error(Source::Derived)
            }
        }
    }

    fn current_exposure(&self) -> Option<ExposureRef> {
        let runtime = self.runtime();
        if !self.connected(&runtime) || runtime.activity != CameraState::Exposing {
            return None;
        }
        Some(ExposureRef {
            started_at: runtime.exposure_started_at,
            duration_s: runtime.exposure_duration_s,
            frame_id: None,
            operation_id: runtime.operation_id.clone(),
            capture_id: None,
            stream_generation: None,
            clock_quality: clock_quality(runtime.driver_id.as_deref()),
        })
    }

    fn abort_exposure(&self) -> DeviceResult<()> {
        let runtime = self.runtime();
        if !self.connected(&runtime) || runtime.activity != CameraState::Exposing {
            return Ok(());
        }
        Err(DeviceError::unsupported(
            "this camera backend cannot interrupt the active exposure; it ends when the \
             exposure completes",
        ))
    }

    fn camera_name(&self) -> Option<String> {
        let runtime = self.runtime();
        self.connected(&runtime)
            .then_some(runtime.camera_name)
            .flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::testing::manual_clock::ManualClock;

    fn hook() -> (
        OlsCameraHook,
        Arc<RwLock<CameraRuntimeState>>,
        Arc<RwLock<DeviceRegistry>>,
    ) {
        let runtime = Arc::new(RwLock::new(CameraRuntimeState::default()));
        let devices = Arc::new(RwLock::new(DeviceRegistry::default()));
        let hook = OlsCameraHook::new(
            Arc::clone(&runtime),
            Arc::clone(&devices),
            Arc::new(ManualClock::new()),
        );
        (hook, runtime, devices)
    }

    #[test]
    fn no_camera_is_known_not_exposing_and_has_nothing_to_abort() {
        let (hook, _, _) = hook();
        assert_eq!(hook.exposing().known(), Some(&false));
        assert!(hook.current_exposure().is_none());
        assert!(hook.abort_exposure().is_ok());
        assert!(hook.camera_name().is_none());
    }

    #[test]
    fn an_exposing_camera_is_reported_with_its_clock_quality_and_no_fake_abort() {
        let (hook, runtime, devices) = hook();
        devices
            .write()
            .unwrap()
            .connected(DeviceKind::Camera, "cam".into(), None);
        *runtime.write().unwrap() = CameraRuntimeState {
            connection_id: Some("cam".into()),
            activity: CameraState::Exposing,
            driver_id: Some("gphoto2".into()),
            camera_name: Some("R50".into()),
            exposure_duration_s: Some(30.0),
            ..CameraRuntimeState::default()
        };
        assert_eq!(hook.exposing().known(), Some(&true));
        let exposure = hook.current_exposure().unwrap();
        assert_eq!(exposure.clock_quality, ClockQuality::HostTriggerBracket);
        assert_eq!(exposure.duration_s, Some(30.0));
        assert_eq!(hook.camera_name().as_deref(), Some("R50"));
        let err = hook.abort_exposure().unwrap_err();
        assert_eq!(err.class, crate::equipment::types::ErrorClass::Unsupported);
        // An error state is unknown, never assumed idle.
        runtime.write().unwrap().activity = CameraState::Error;
        assert!(hook.exposing().known().is_none());
        // A runtime from a retired connection is not the camera.
        devices
            .write()
            .unwrap()
            .disconnected(DeviceKind::Camera, Some("cam"), None);
        assert_eq!(hook.exposing().known(), Some(&false));
    }
}

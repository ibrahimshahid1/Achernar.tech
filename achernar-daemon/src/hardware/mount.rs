//! Mount hardware abstraction via OLS bridge.
//!
//! Provides a safe Rust wrapper around the OLS C++ mount drivers
//! (currently ASCOM Alpaca, with INDI as a conditional compile target).
//! All methods translate to FFI calls through the OLS bridge.

use serde::{Deserialize, Serialize};
use std::ffi::CString;
use tracing::{debug, info};

use super::ols_ffi;

// ═══════════════════════════════════════════════════════════════════
//  Public Types
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MountDriver {
    Alpaca,
    Indi,
    Simulator,
}

impl MountDriver {
    /// Name string used by OLS driver loading.
    pub fn ols_name(&self) -> &str {
        match self {
            MountDriver::Alpaca => "alpaca",
            MountDriver::Indi => "indi:indi_simulator_telescope",
            MountDriver::Simulator => "alpaca",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EqCoord {
    /// Right ascension in hours (0..24)
    pub ra: f64,
    /// Declination in degrees (-90..+90)
    pub dec: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AltAzCoord {
    /// Altitude in degrees (-90..+90)
    pub alt: f64,
    /// Azimuth in degrees (0..360)
    pub az: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackingMode {
    Sidereal,
    Solar,
    Lunar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlewDirection {
    Stop,
    North,
    South,
    West,
    East,
}

impl SlewDirection {
    fn to_ols(self) -> ols_ffi::OlsMountSlew {
        match self {
            SlewDirection::Stop => ols_ffi::OlsMountSlew::Stop,
            SlewDirection::North => ols_ffi::OlsMountSlew::N,
            SlewDirection::South => ols_ffi::OlsMountSlew::S,
            SlewDirection::West => ols_ffi::OlsMountSlew::W,
            SlewDirection::East => ols_ffi::OlsMountSlew::E,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeridianFlip {
    Unsupported,
    Flip,
    Stop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MountDriverInfo {
    pub name: String,
    pub option: String,
    pub has_option: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MountStatus {
    pub connected: bool,
    pub driver_loaded: bool,
    pub mount_loaded: bool,
    pub position: Option<EqCoord>,
    pub tracking: Option<bool>,
    pub tracking_mode: Option<TrackingMode>,
}

// ═══════════════════════════════════════════════════════════════════
//  Mount Implementation
// ═══════════════════════════════════════════════════════════════════

/// Hardware mount interface backed by the OLS bridge.
///
/// Manages the lifecycle of the OLS `MountInterface` opaque handle.
/// Thread-safe: The OLS MountInterface uses internal mutexes.
pub struct Mount {
    handle: *mut ols_ffi::OlsMountInterface,
    driver_type: MountDriver,
}

// OLS MountInterface uses internal locking
unsafe impl Send for Mount {}
unsafe impl Sync for Mount {}

impl Mount {
    /// Create a new mount interface (does not connect).
    pub fn new(driver: MountDriver) -> crate::error::Result<Self> {
        let mut handle: *mut ols_ffi::OlsMountInterface = std::ptr::null_mut();
        let rc = unsafe { ols_ffi::ols_mount_interface_create(&mut handle) };
        ols_ffi::check_ols_error(rc).map_err(|e| {
            crate::error::HardwareError::Communication(format!(
                "Failed to create mount interface: {e}"
            ))
        })?;

        info!(driver = ?driver, "Created mount interface");
        Ok(Self {
            handle,
            driver_type: driver,
        })
    }

    /// List available mount drivers.
    pub fn list_drivers(&self) -> crate::error::Result<Vec<MountDriverInfo>> {
        let mut infos = vec![
            ols_ffi::OlsMountDriverInfo {
                name: [0i8; 128],
                option: [0i8; 128],
                has_option: 0,
            };
            16
        ];
        let mut count: i32 = 0;
        let rc = unsafe {
            ols_ffi::ols_mount_driver_list(self.handle, infos.as_mut_ptr(), 16, &mut count)
        };
        ols_ffi::check_ols_error(rc).map_err(|e| {
            crate::error::HardwareError::Communication(format!("Failed to list mount drivers: {e}"))
        })?;

        let result: Vec<MountDriverInfo> = infos[..count as usize]
            .iter()
            .map(|info| {
                let name = unsafe { std::ffi::CStr::from_ptr(info.name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                let option = unsafe { std::ffi::CStr::from_ptr(info.option.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                MountDriverInfo {
                    name,
                    option,
                    has_option: info.has_option != 0,
                }
            })
            .collect();

        debug!(drivers = ?result, "Listed mount drivers");
        Ok(result)
    }

    /// Connect to the mount using the configured driver.
    pub fn connect(&mut self) -> crate::error::Result<()> {
        self.connect_with_option("auto")
    }

    /// Connect to the mount with a specific driver option (e.g., Alpaca URL).
    pub fn connect_with_option(&mut self, option: &str) -> crate::error::Result<()> {
        let driver_name = CString::new(self.driver_type.ols_name()).map_err(|_| {
            crate::error::HardwareError::Communication("Invalid driver name".into())
        })?;
        let option_c = CString::new(option).map_err(|_| {
            crate::error::HardwareError::Communication("Invalid option string".into())
        })?;

        // Load driver
        info!(driver = ?self.driver_type, option = option, "Loading mount driver");
        let rc = unsafe {
            ols_ffi::ols_mount_load_driver(self.handle, driver_name.as_ptr(), option_c.as_ptr())
        };
        ols_ffi::check_ols_error(rc).map_err(|e| {
            crate::error::HardwareError::Communication(format!("Failed to load mount driver: {e}"))
        })?;

        // Load mount
        let rc = unsafe { ols_ffi::ols_mount_load_mount(self.handle) };
        ols_ffi::check_ols_error(rc).map_err(|e| {
            crate::error::HardwareError::Communication(format!("Failed to load mount: {e}"))
        })?;

        // Connect
        let rc = unsafe { ols_ffi::ols_mount_connect(self.handle, 0) };
        ols_ffi::check_ols_error(rc).map_err(|e| {
            crate::error::HardwareError::Communication(format!("Failed to connect mount: {e}"))
        })?;

        info!("Mount connected successfully");
        Ok(())
    }

    /// Check if the mount is connected.
    pub fn is_connected(&self) -> bool {
        unsafe { ols_ffi::ols_mount_connected(self.handle) != 0 }
    }

    /// Get current equatorial coordinates (RA/DEC).
    pub fn get_position(&self) -> crate::error::Result<EqCoord> {
        let mut coord = ols_ffi::OlsEqCoord { ra: 0.0, dec: 0.0 };
        let rc = unsafe { ols_ffi::ols_mount_get_current(self.handle, &mut coord) };
        ols_ffi::check_ols_error(rc).map_err(|e| {
            crate::error::HardwareError::Communication(format!("Failed to get position: {e}"))
        })?;
        Ok(EqCoord {
            ra: coord.ra,
            dec: coord.dec,
        })
    }

    /// Slew to target equatorial coordinates (GoTo).
    pub fn goto(&self, target: EqCoord) -> crate::error::Result<()> {
        info!(ra = target.ra, dec = target.dec, "Mount GoTo");
        let rc = unsafe { ols_ffi::ols_mount_goto(self.handle, target.ra, target.dec) };
        ols_ffi::check_ols_error(rc).map_err(|e| {
            crate::error::HardwareError::Communication(format!("Failed to goto: {e}"))
        })?;
        Ok(())
    }

    /// Sync mount position to given coordinates.
    pub fn sync(&self, coord: EqCoord) -> crate::error::Result<()> {
        info!(ra = coord.ra, dec = coord.dec, "Mount Sync");
        let rc = unsafe { ols_ffi::ols_mount_sync(self.handle, coord.ra, coord.dec) };
        ols_ffi::check_ols_error(rc).map_err(|e| {
            crate::error::HardwareError::Communication(format!("Failed to sync: {e}"))
        })?;
        Ok(())
    }

    /// Manual slew in a direction at a given speed.
    pub fn slew(&self, direction: SlewDirection, speed: i32) -> crate::error::Result<()> {
        let rc = unsafe { ols_ffi::ols_mount_slew(self.handle, direction.to_ols(), speed) };
        ols_ffi::check_ols_error(rc).map_err(|e| {
            crate::error::HardwareError::Communication(format!("Failed to slew: {e}"))
        })?;
        Ok(())
    }

    /// Stop all mount motion.
    pub fn stop(&self) -> crate::error::Result<()> {
        self.slew(SlewDirection::Stop, 0)
    }

    /// Get current tracking mode.
    pub fn get_tracking_mode(&self) -> crate::error::Result<TrackingMode> {
        let mut mode = ols_ffi::OlsMountTracking::Sidereal;
        let rc = unsafe { ols_ffi::ols_mount_get_tracking(self.handle, &mut mode) };
        ols_ffi::check_ols_error(rc).map_err(|e| {
            crate::error::HardwareError::Communication(format!("Failed to get tracking: {e}"))
        })?;
        Ok(match mode {
            ols_ffi::OlsMountTracking::Sidereal => TrackingMode::Sidereal,
            ols_ffi::OlsMountTracking::Solar => TrackingMode::Solar,
            ols_ffi::OlsMountTracking::Lunar => TrackingMode::Lunar,
        })
    }

    /// Check if tracking is enabled (true/false).
    pub fn is_tracking(&self) -> crate::error::Result<bool> {
        let mut state: i32 = 0;
        let rc = unsafe { ols_ffi::ols_mount_get_tracking_state(self.handle, &mut state) };
        ols_ffi::check_ols_error(rc).map_err(|e| {
            crate::error::HardwareError::Communication(format!("Failed to get tracking state: {e}"))
        })?;
        Ok(state > 0)
    }

    /// Enable or disable tracking.
    pub fn set_tracking(&self, enabled: bool) -> crate::error::Result<()> {
        let rc = unsafe {
            ols_ffi::ols_mount_set_tracking_state(self.handle, if enabled { 1 } else { 0 })
        };
        ols_ffi::check_ols_error(rc).map_err(|e| {
            crate::error::HardwareError::Communication(format!("Failed to set tracking: {e}"))
        })?;
        Ok(())
    }

    /// Set tracking mode (sidereal, solar, lunar).
    pub fn set_tracking_mode(&self, mode: TrackingMode) -> crate::error::Result<()> {
        let ols_mode = match mode {
            TrackingMode::Sidereal => ols_ffi::OlsMountTracking::Sidereal,
            TrackingMode::Solar => ols_ffi::OlsMountTracking::Solar,
            TrackingMode::Lunar => ols_ffi::OlsMountTracking::Lunar,
        };
        let rc = unsafe { ols_ffi::ols_mount_set_tracking_mode(self.handle, ols_mode) };
        ols_ffi::check_ols_error(rc).map_err(|e| {
            crate::error::HardwareError::Communication(format!("Failed to set tracking mode: {e}"))
        })?;
        Ok(())
    }

    /// Set site geographic coordinates.
    pub fn set_location(&self, lat: f64, lon: f64) -> crate::error::Result<()> {
        let rc = unsafe { ols_ffi::ols_mount_set_lat_lon(self.handle, lat, lon) };
        ols_ffi::check_ols_error(rc).map_err(|e| {
            crate::error::HardwareError::Communication(format!("Failed to set location: {e}"))
        })?;
        Ok(())
    }

    /// Get site geographic coordinates.
    pub fn get_location(&self) -> crate::error::Result<(f64, f64)> {
        let mut lat: f64 = 0.0;
        let mut lon: f64 = 0.0;
        let rc = unsafe { ols_ffi::ols_mount_get_lat_lon(self.handle, &mut lat, &mut lon) };
        ols_ffi::check_ols_error(rc).map_err(|e| {
            crate::error::HardwareError::Communication(format!("Failed to get location: {e}"))
        })?;
        Ok((lat, lon))
    }

    /// Send pulse guide corrections (in milliseconds).
    pub fn pulse_guide(&self, ns_ms: f32, we_ms: f32) -> crate::error::Result<()> {
        let rc = unsafe { ols_ffi::ols_mount_pulse_guide(self.handle, ns_ms, we_ms) };
        ols_ffi::check_ols_error(rc).map_err(|e| {
            crate::error::HardwareError::Communication(format!("Failed to pulse guide: {e}"))
        })?;
        Ok(())
    }

    /// Check if pulse guiding is supported.
    pub fn pulse_guide_supported(&self) -> crate::error::Result<(bool, f32, f32)> {
        let mut supported: i32 = 0;
        let mut ns_rate: f32 = 0.0;
        let mut we_rate: f32 = 0.0;
        let rc = unsafe {
            ols_ffi::ols_mount_pulse_guide_supported(
                self.handle,
                &mut supported,
                &mut ns_rate,
                &mut we_rate,
            )
        };
        ols_ffi::check_ols_error(rc).map_err(|e| {
            crate::error::HardwareError::Communication(format!("Failed to check pulse guide: {e}"))
        })?;
        Ok((supported != 0, ns_rate, we_rate))
    }

    /// Get current slew rate and total number of rates.
    pub fn get_slew_rate(&self) -> crate::error::Result<(i32, i32)> {
        let mut current: i32 = 0;
        let mut count: i32 = 0;
        let rc = unsafe { ols_ffi::ols_mount_get_slew_rate(self.handle, &mut current, &mut count) };
        ols_ffi::check_ols_error(rc).map_err(|e| {
            crate::error::HardwareError::Communication(format!("Failed to get slew rate: {e}"))
        })?;
        Ok((current, count))
    }

    /// Get full mount status (combined query).
    pub fn status(&self) -> MountStatus {
        let connected = self.is_connected();
        let driver_loaded = unsafe { ols_ffi::ols_mount_driver_loaded(self.handle) != 0 };
        let mount_loaded = unsafe { ols_ffi::ols_mount_is_loaded(self.handle) != 0 };

        let position = if connected {
            self.get_position().ok()
        } else {
            None
        };
        let tracking = if connected {
            self.is_tracking().ok()
        } else {
            None
        };
        let tracking_mode = if connected {
            self.get_tracking_mode().ok()
        } else {
            None
        };

        MountStatus {
            connected,
            driver_loaded,
            mount_loaded,
            position,
            tracking,
            tracking_mode,
        }
    }

    /// Shutdown and disconnect the mount.
    pub fn shutdown(&self) -> crate::error::Result<()> {
        info!("Shutting down mount");
        let rc = unsafe { ols_ffi::ols_mount_shutdown(self.handle) };
        ols_ffi::check_ols_error(rc).map_err(|e| {
            crate::error::HardwareError::Communication(format!("Failed to shutdown mount: {e}"))
        })?;
        Ok(())
    }
}

impl Drop for Mount {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            debug!("Destroying mount interface");
            unsafe {
                ols_ffi::ols_mount_interface_destroy(self.handle);
            }
            self.handle = std::ptr::null_mut();
        }
    }
}

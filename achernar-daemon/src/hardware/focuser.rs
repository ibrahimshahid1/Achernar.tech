//! Focuser hardware abstraction via OLS bridge.
//!
//! Provides a safe Rust wrapper around the OLS C++ focuser drivers
//! (currently ASCOM Alpaca). All methods translate to FFI calls
//! through the OLS bridge.

use serde::{Deserialize, Serialize};
use std::ffi::CString;
use tracing::{debug, info};

use super::ols_ffi;

// ═══════════════════════════════════════════════════════════════════
//  Public Types
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FocuserDriver {
    Alpaca,
    Indi,
    Moonlite,
    UsbSerial,
    Simulator,
}

impl FocuserDriver {
    /// Name string used by OLS driver loading.
    pub fn ols_name(&self) -> &str {
        match self {
            FocuserDriver::Alpaca => "alpaca:remote",
            FocuserDriver::Indi => "indi:focuser",
            FocuserDriver::Moonlite => "moonlite",
            FocuserDriver::UsbSerial => "usb",
            FocuserDriver::Simulator => "alpaca:remote",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocuserDriverInfo {
    pub name: String,
    pub option: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocuserStatus {
    pub connected: bool,
    pub position: Option<i32>,
    pub is_moving: bool,
    pub is_absolute: bool,
    pub max_range: Option<i32>,
}

// ═══════════════════════════════════════════════════════════════════
//  Focuser Implementation
// ═══════════════════════════════════════════════════════════════════

/// Hardware focuser interface backed by the OLS bridge.
///
/// Manages the lifecycle of the OLS `FocuserInterface` opaque handle.
pub struct Focuser {
    handle: *mut ols_ffi::OlsFocuserInterface,
    driver_type: FocuserDriver,
    connected: bool,
}

// OLS FocuserInterface uses internal locking
unsafe impl Send for Focuser {}
unsafe impl Sync for Focuser {}

impl Focuser {
    /// Create a new focuser interface (does not connect).
    pub fn new(driver: FocuserDriver) -> crate::error::Result<Self> {
        let mut handle: *mut ols_ffi::OlsFocuserInterface = std::ptr::null_mut();
        let rc = unsafe { ols_ffi::ols_focuser_interface_create(&mut handle) };
        ols_ffi::check_ols_error(rc)
            .map_err(|e| crate::error::HardwareError::Communication(format!("Failed to create focuser interface: {e}")))?;

        info!(driver = ?driver, "Created focuser interface");
        Ok(Self {
            handle,
            driver_type: driver,
            connected: false,
        })
    }

    /// List available focuser drivers.
    pub fn list_drivers(&self) -> crate::error::Result<Vec<FocuserDriverInfo>> {
        let mut infos = vec![
            ols_ffi::OlsFocuserDriverInfo {
                name: [0i8; 128],
                option: [0i8; 128],
            };
            16
        ];
        let mut count: i32 = 0;
        let rc = unsafe {
            ols_ffi::ols_focuser_driver_list(self.handle, infos.as_mut_ptr(), 16, &mut count)
        };
        ols_ffi::check_ols_error(rc)
            .map_err(|e| crate::error::HardwareError::Communication(format!("Failed to list focuser drivers: {e}")))?;

        let result: Vec<FocuserDriverInfo> = infos[..count as usize]
            .iter()
            .map(|info| {
                let name = unsafe { std::ffi::CStr::from_ptr(info.name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                let option = unsafe { std::ffi::CStr::from_ptr(info.option.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                FocuserDriverInfo { name, option }
            })
            .collect();

        debug!(drivers = ?result, "Listed focuser drivers");
        Ok(result)
    }

    /// Connect to the focuser using the configured driver.
    pub fn connect(&mut self) -> crate::error::Result<()> {
        self.connect_with_option("auto")
    }

    /// Connect to the focuser with a specific option (e.g., Alpaca URL).
    pub fn connect_with_option(&mut self, option: &str) -> crate::error::Result<()> {
        let driver_name = CString::new(self.driver_type.ols_name())
            .map_err(|_| crate::error::HardwareError::Communication("Invalid driver name".into()))?;
        let option_c = CString::new(option)
            .map_err(|_| crate::error::HardwareError::Communication("Invalid option string".into()))?;

        info!(driver = ?self.driver_type, option = option, "Loading focuser driver");
        let rc = unsafe {
            ols_ffi::ols_focuser_load_driver(self.handle, driver_name.as_ptr(), option_c.as_ptr())
        };
        ols_ffi::check_ols_error(rc)
            .map_err(|e| crate::error::HardwareError::Communication(format!("Failed to load focuser driver: {e}")))?;

        self.connected = true;
        info!("Focuser connected successfully");
        Ok(())
    }

    /// Check if the focuser is connected.
    pub fn is_connected(&self) -> bool {
        self.connected && unsafe { ols_ffi::ols_focuser_is_loaded(self.handle) != 0 }
    }

    /// Get current focuser position.
    pub fn position(&self) -> crate::error::Result<i32> {
        let mut pos: i32 = 0;
        let rc = unsafe { ols_ffi::ols_focuser_position(self.handle, &mut pos) };
        ols_ffi::check_ols_error(rc)
            .map_err(|e| crate::error::HardwareError::Communication(format!("Failed to get position: {e}")))?;
        Ok(pos)
    }

    /// Check if the focuser is currently moving.
    pub fn is_moving(&self) -> crate::error::Result<bool> {
        let mut moving: i32 = 0;
        let rc = unsafe { ols_ffi::ols_focuser_is_moving(self.handle, &mut moving) };
        ols_ffi::check_ols_error(rc)
            .map_err(|e| crate::error::HardwareError::Communication(format!("Failed to check moving: {e}")))?;
        Ok(moving != 0)
    }

    /// Check if the focuser supports absolute positioning.
    pub fn is_absolute(&self) -> crate::error::Result<bool> {
        let mut absolute: i32 = 0;
        let rc = unsafe { ols_ffi::ols_focuser_is_absolute(self.handle, &mut absolute) };
        ols_ffi::check_ols_error(rc)
            .map_err(|e| crate::error::HardwareError::Communication(format!("Failed to check absolute: {e}")))?;
        Ok(absolute != 0)
    }

    /// Get the maximum focus range.
    pub fn max_range(&self) -> crate::error::Result<i32> {
        let mut range: i32 = 0;
        let rc = unsafe { ols_ffi::ols_focuser_max_range(self.handle, &mut range) };
        ols_ffi::check_ols_error(rc)
            .map_err(|e| crate::error::HardwareError::Communication(format!("Failed to get max range: {e}")))?;
        Ok(range)
    }

    /// Move the focuser to an absolute position.
    pub fn move_absolute(&mut self, position: i32) -> crate::error::Result<()> {
        info!(position, "Moving focuser to absolute position");
        let rc = unsafe { ols_ffi::ols_focuser_move_to(self.handle, position) };
        ols_ffi::check_ols_error(rc)
            .map_err(|e| crate::error::HardwareError::Communication(format!("Failed to move to position: {e}")))?;
        Ok(())
    }

    /// Move the focuser by a relative number of steps.
    pub fn move_relative(&mut self, steps: i32) -> crate::error::Result<()> {
        info!(steps, "Moving focuser by relative steps");
        let rc = unsafe { ols_ffi::ols_focuser_move_by(self.handle, steps) };
        ols_ffi::check_ols_error(rc)
            .map_err(|e| crate::error::HardwareError::Communication(format!("Failed to move by steps: {e}")))?;
        Ok(())
    }

    /// Abort focuser motion.
    pub fn abort(&mut self) -> crate::error::Result<()> {
        info!("Aborting focuser motion");
        let rc = unsafe { ols_ffi::ols_focuser_abort(self.handle) };
        ols_ffi::check_ols_error(rc)
            .map_err(|e| crate::error::HardwareError::Communication(format!("Failed to abort focuser: {e}")))?;
        Ok(())
    }

    /// Get full focuser status (combined query).
    pub fn status(&self) -> FocuserStatus {
        let connected = self.is_connected();
        if !connected {
            return FocuserStatus {
                connected: false,
                position: None,
                is_moving: false,
                is_absolute: false,
                max_range: None,
            };
        }

        FocuserStatus {
            connected,
            position: self.position().ok(),
            is_moving: self.is_moving().unwrap_or(false),
            is_absolute: self.is_absolute().unwrap_or(false),
            max_range: self.max_range().ok(),
        }
    }
}

impl Drop for Focuser {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            debug!("Destroying focuser interface");
            unsafe {
                ols_ffi::ols_focuser_interface_destroy(self.handle);
            }
            self.handle = std::ptr::null_mut();
        }
    }
}

//! Filter wheel hardware abstraction.
//! INDI + USB HID filter wheel drivers.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FilterWheelDriver {
    Indi,
    UsbHid,
    Simulator,
}

pub struct FilterWheel {
    driver: FilterWheelDriver,
    position: u32,
    filter_count: u32,
    connected: bool,
    is_moving: bool,
}

impl FilterWheel {
    pub fn new(driver: FilterWheelDriver, filter_count: u32) -> Self {
        Self {
            driver,
            position: 0,
            filter_count,
            connected: false,
            is_moving: false,
        }
    }

    pub fn position(&self) -> u32 {
        self.position
    }
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    pub async fn connect(&mut self) -> crate::error::Result<()> {
        if !matches!(self.driver, FilterWheelDriver::Simulator) {
            return Err(crate::error::HardwareError::Communication(
                "Physical filter-wheel drivers are not implemented".into()).into());
        }
        self.connected = true;
        Ok(())
    }

    pub async fn set_position(&mut self, pos: u32) -> crate::error::Result<()> {
        if !self.connected || !matches!(self.driver, FilterWheelDriver::Simulator) {
            return Err(crate::error::HardwareError::Communication("Filter wheel is disconnected".into()).into());
        }
        if pos >= self.filter_count {
            return Err(crate::error::HardwareError::Communication("Filter position is out of range".into()).into());
        }
        self.position = pos;
        Ok(())
    }

    pub fn health_status(&self) -> crate::error::Result<serde_json::Value> {
        if !self.connected || !matches!(self.driver, FilterWheelDriver::Simulator) {
            return Err(crate::error::HardwareError::Communication("Filter wheel is disconnected".into()).into());
        }
        Ok(serde_json::json!({"connected": true, "name": "Simulator filter wheel",
            "position": self.position, "moving": self.is_moving, "filter_count": self.filter_count}))
    }
}

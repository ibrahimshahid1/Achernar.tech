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
        Self { driver, position: 0, filter_count, connected: false, is_moving: false }
    }

    pub fn position(&self) -> u32 { self.position }
    pub fn is_connected(&self) -> bool { self.connected }

    pub async fn connect(&mut self) -> crate::error::Result<()> {
        todo!("Connect to filter wheel via {:?}", self.driver)
    }

    pub async fn set_position(&mut self, _pos: u32) -> crate::error::Result<()> {
        todo!("Move filter wheel to position")
    }
}

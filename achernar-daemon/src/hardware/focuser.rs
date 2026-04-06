//! Focuser hardware abstraction.
//! INDI + Moonlite + USB/Serial focuser drivers.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FocuserDriver {
    Indi,
    Moonlite,
    UsbSerial,
    Simulator,
}

/// Hardware focuser interface.
pub struct Focuser {
    driver: FocuserDriver,
    position: i32,
    connected: bool,
    is_moving: bool,
}

impl Focuser {
    pub fn new(driver: FocuserDriver) -> Self {
        Self { driver, position: 0, connected: false, is_moving: false }
    }

    pub fn position(&self) -> i32 { self.position }
    pub fn is_connected(&self) -> bool { self.connected }
    pub fn is_moving(&self) -> bool { self.is_moving }

    pub async fn connect(&mut self) -> crate::error::Result<()> {
        todo!("Connect to focuser via {:?}", self.driver)
    }

    pub async fn move_absolute(&mut self, _position: i32) -> crate::error::Result<()> {
        todo!("Move focuser to absolute position")
    }

    pub async fn move_relative(&mut self, _steps: i32) -> crate::error::Result<()> {
        todo!("Move focuser relative steps")
    }

    pub async fn abort(&mut self) -> crate::error::Result<()> {
        self.is_moving = false;
        Ok(())
    }
}

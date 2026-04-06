//! Mount hardware abstraction.
//! INDI + ASCOM Alpaca + Serial/TCP mount drivers.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MountDriver {
    Indi,
    AscomAlpaca,
    SerialTcp,
    Simulator,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PierSide {
    East,
    West,
    Unknown,
}

pub struct Mount {
    driver: MountDriver,
    ra: f64,
    dec: f64,
    tracking: bool,
    parked: bool,
    pier_side: PierSide,
    connected: bool,
}

impl Mount {
    pub fn new(driver: MountDriver) -> Self {
        Self {
            driver, ra: 0.0, dec: 0.0, tracking: false,
            parked: false, pier_side: PierSide::Unknown, connected: false,
        }
    }

    pub fn ra(&self) -> f64 { self.ra }
    pub fn dec(&self) -> f64 { self.dec }
    pub fn is_tracking(&self) -> bool { self.tracking }
    pub fn is_connected(&self) -> bool { self.connected }

    pub async fn connect(&mut self) -> crate::error::Result<()> {
        todo!("Connect to mount via {:?}", self.driver)
    }

    pub async fn slew_to(&mut self, _ra: f64, _dec: f64) -> crate::error::Result<()> {
        todo!("Slew mount to coordinates")
    }

    pub async fn park(&mut self) -> crate::error::Result<()> {
        todo!("Park mount")
    }

    pub async fn set_tracking(&mut self, _enabled: bool) -> crate::error::Result<()> {
        todo!("Set tracking state")
    }

    pub async fn pulse_guide(&mut self, _direction: u8, _duration_ms: u32) -> crate::error::Result<()> {
        todo!("Send pulse guide command")
    }
}

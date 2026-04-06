//! Mount controller — Section 8.3.
//! Slew execution and safety enforcement.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MountState {
    Idle,
    Slewing,
    Tracking,
    Guiding,
    Parking,
    Parked,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlewTarget {
    pub ra_hours: f64,
    pub dec_degrees: f64,
    pub name: Option<String>,
}

pub struct MountController {
    state: MountState,
}

impl MountController {
    pub fn new() -> Self {
        Self { state: MountState::Idle }
    }

    pub fn state(&self) -> MountState { self.state }

    pub async fn slew(&mut self, _target: SlewTarget) -> crate::error::Result<()> {
        todo!("Implement slew with safety enforcement — Section 8.3")
    }

    pub async fn park(&mut self) -> crate::error::Result<()> {
        todo!("Implement park")
    }

    pub async fn unpark(&mut self) -> crate::error::Result<()> {
        todo!("Implement unpark")
    }

    pub fn abort_slew(&mut self) {
        self.state = MountState::Idle;
        tracing::warn!("Mount slew aborted");
    }
}

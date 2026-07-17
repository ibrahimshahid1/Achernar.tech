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
        Self {
            state: MountState::Idle,
        }
    }

    pub fn state(&self) -> MountState {
        self.state
    }

    pub async fn slew(&mut self, target: SlewTarget) -> crate::error::Result<()> {
        if !(0.0..24.0).contains(&target.ra_hours) || !(-90.0..=90.0).contains(&target.dec_degrees)
        {
            return Err(crate::error::ControlError::SafetyBlocked(format!(
                "Unsafe slew target ra={}h dec={}deg",
                target.ra_hours, target.dec_degrees
            ))
            .into());
        }
        if matches!(self.state, MountState::Parking | MountState::Parked) {
            return Err(crate::error::ControlError::ModeNotAllowed(
                "Cannot slew while mount is parking or parked".to_string(),
            )
            .into());
        }

        self.state = MountState::Slewing;
        tracing::info!(
            "Mount controller accepted slew target: {}h {}deg",
            target.ra_hours,
            target.dec_degrees
        );
        self.state = MountState::Tracking;
        Ok(())
    }

    pub async fn park(&mut self) -> crate::error::Result<()> {
        self.state = MountState::Parking;
        tracing::info!("Mount controller park requested");
        self.state = MountState::Parked;
        Ok(())
    }

    pub async fn unpark(&mut self) -> crate::error::Result<()> {
        if !matches!(self.state, MountState::Parked | MountState::Parking) {
            return Ok(());
        }
        tracing::info!("Mount controller unpark requested");
        self.state = MountState::Idle;
        Ok(())
    }

    pub fn abort_slew(&mut self) {
        self.state = MountState::Idle;
        tracing::warn!("Mount slew aborted");
    }
}

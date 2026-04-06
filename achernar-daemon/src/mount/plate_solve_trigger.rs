//! Plate solve trigger — Section 8.3 (M-3).
//! Determines when to trigger a plate solve and calls the solver.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlateSolveResult {
    pub ra_hours: f64,
    pub dec_degrees: f64,
    pub rotation_degrees: f64,
    pub pixel_scale: f64,
    pub solve_time_ms: u64,
    pub success: bool,
}

pub struct PlateSolveTrigger {
    max_pointing_error_arcmin: f64,
    cooldown_seconds: f64,
    last_solve: Option<chrono::DateTime<chrono::Utc>>,
}

impl PlateSolveTrigger {
    pub fn new(max_error_arcmin: f64, cooldown_seconds: f64) -> Self {
        Self {
            max_pointing_error_arcmin: max_error_arcmin,
            cooldown_seconds,
            last_solve: None,
        }
    }

    pub fn should_solve(&self, _pointing_error_arcmin: f64) -> bool {
        todo!("Implement plate solve trigger logic — Section 8.3 M-3")
    }

    pub async fn trigger_solve(&mut self, _image_data: &[u8]) -> crate::error::Result<PlateSolveResult> {
        todo!("Call plate solver and return result")
    }
}

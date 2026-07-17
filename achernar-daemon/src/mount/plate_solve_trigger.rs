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

    pub fn should_solve(&self, pointing_error_arcmin: f64) -> bool {
        if pointing_error_arcmin >= self.max_pointing_error_arcmin {
            return self.cooldown_elapsed();
        }
        false
    }

    pub async fn trigger_solve(
        &mut self,
        image_data: &[u8],
    ) -> crate::error::Result<PlateSolveResult> {
        if image_data.is_empty() {
            return Err(crate::error::HardwareError::Unsupported(
                "Plate solve requires a non-empty image frame".to_string(),
            )
            .into());
        }

        self.last_solve = Some(chrono::Utc::now());
        Err(crate::error::HardwareError::Unsupported(
            "No local plate solver backend is configured yet".to_string(),
        )
        .into())
    }

    fn cooldown_elapsed(&self) -> bool {
        let Some(last_solve) = self.last_solve else {
            return true;
        };
        let elapsed = chrono::Utc::now()
            .signed_duration_since(last_solve)
            .num_milliseconds() as f64
            / 1000.0;
        elapsed >= self.cooldown_seconds
    }
}

//! Plate solver — local solver + FITS WCS.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlateSolution {
    pub ra_center: f64,
    pub dec_center: f64,
    pub rotation_deg: f64,
    pub pixel_scale_arcsec: f64,
    pub field_width_arcmin: f64,
    pub field_height_arcmin: f64,
    pub star_count: u32,
    pub solve_time_ms: u64,
}

pub struct PlateSolver;

impl PlateSolver {
    /// Solve a frame using a local astrometry solver.
    pub async fn solve(
        image_data: &[u8],
        width: u32,
        height: u32,
        _hint_ra: Option<f64>,
        _hint_dec: Option<f64>,
    ) -> crate::error::Result<PlateSolution> {
        if image_data.is_empty() || width == 0 || height == 0 {
            return Err(crate::error::HardwareError::Unsupported(
                "Plate solving requires a non-empty image and valid dimensions".to_string(),
            )
            .into());
        }

        Err(crate::error::HardwareError::Unsupported(
            "No local astrometry solver backend is configured yet".to_string(),
        )
        .into())
    }
}

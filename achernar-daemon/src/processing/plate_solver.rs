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
        _image_data: &[u8],
        _width: u32,
        _height: u32,
        _hint_ra: Option<f64>,
        _hint_dec: Option<f64>,
    ) -> crate::error::Result<PlateSolution> {
        todo!("Implement local plate solving (astrometry.net or ASTAP)")
    }
}

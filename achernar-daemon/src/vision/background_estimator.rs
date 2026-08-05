//! Background estimator — Section 8.1.
//! BGE classical SEP wrapper for sky background estimation.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundEstimate {
    pub mean_adu: f64,
    pub rms_adu: f64,
    pub median_adu: f64,
    pub gradient_x: f64,
    pub gradient_y: f64,
}

pub struct BackgroundEstimator;

impl BackgroundEstimator {
    /// Estimate background from an image using SEP-like sigma-clipped statistics.
    pub fn estimate(image: &ndarray::Array2<f32>) -> BackgroundEstimate {
        let pixels: Vec<f32> = image.iter().copied().collect();
        // No saturation point: this entry point takes a bare ndarray with no
        // camera provenance. The previous `observed_max.max(65535.0)` was
        // always ≥ 65535 by construction, so the saturation threshold it
        // produced was never reachable — the same never-fires bug, spelled
        // differently. Excluding nothing is safe for a median/MAD background.
        let stats = super::background::estimate_background(&pixels, None);

        BackgroundEstimate {
            mean_adu: stats.background_adu as f64,
            rms_adu: stats.background_sigma as f64,
            median_adu: stats.background_adu as f64,
            gradient_x: 0.0,
            gradient_y: 0.0,
        }
    }
}

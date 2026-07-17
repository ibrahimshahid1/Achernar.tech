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
        let max_adu = pixels
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(0.0_f32, f32::max)
            .max(65535.0);
        let stats = super::background::estimate_background(&pixels, max_adu);

        BackgroundEstimate {
            mean_adu: stats.background_adu as f64,
            rms_adu: stats.background_sigma as f64,
            median_adu: stats.background_adu as f64,
            gradient_x: 0.0,
            gradient_y: 0.0,
        }
    }
}

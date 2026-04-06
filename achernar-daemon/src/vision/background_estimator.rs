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
    pub fn estimate(_image: &ndarray::Array2<f32>) -> BackgroundEstimate {
        todo!("Implement background estimation with sigma-clipped stats — Section 8.1")
    }
}

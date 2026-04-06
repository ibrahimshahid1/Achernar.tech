//! TED runner — Section 8.1.
//! Trailing/elongation detection: classical SEP + CNN wrapper.

use super::shared_observation::TrailingAxis;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TedOutput {
    pub trailing_detected: bool,
    pub trailing_axis: TrailingAxis,
    pub trailing_magnitude: f64,
    pub elongation_ratio: f64,
}

pub struct TedRunner {
    model_path: Option<String>,
    elongation_threshold: f64,
}

impl TedRunner {
    pub fn new(model_path: Option<String>) -> Self {
        Self { model_path, elongation_threshold: 1.3 }
    }

    /// Classical SEP-based trailing detection.
    pub fn detect_classical(&self, _image: &ndarray::Array2<f32>) -> TedOutput {
        todo!("Implement classical SEP trailing detection — Section 8.1")
    }

    /// CNN-based trailing detection (if model available).
    pub fn detect_cnn(&self, _image: &ndarray::Array3<f32>) -> crate::error::Result<TedOutput> {
        todo!("Implement TED CNN inference — Section 8.1")
    }
}

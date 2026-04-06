//! Focus inference — Section 7.3.5.
//! ONNX runner for focus quality estimation and offset prediction.

use serde::{Deserialize, Serialize};

/// Output from the FocusQualityEstimator (FQE) ONNX model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusQualityOutput {
    /// Quality score in [0, 1] — 1 = perfectly focused
    pub quality_score: f64,
    /// Confidence of the quality estimate
    pub confidence: f64,
    /// Predicted defocus direction: -1 (intra), 0 (in focus), +1 (extra)
    pub defocus_direction: i8,
}

/// Output from the FocusOffsetMLP ONNX model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusOffsetOutput {
    /// Predicted optimal focus offset in steps
    pub predicted_offset: f64,
    /// Uncertainty (std dev) of the prediction
    pub uncertainty: f64,
    /// Model confidence
    pub confidence: f64,
}

/// ONNX inference wrapper for focus models.
pub struct FocusInference {
    quality_model_path: Option<String>,
    offset_model_path: Option<String>,
}

impl FocusInference {
    pub fn new(config: &crate::config::FocusModelsConfig) -> Self {
        Self {
            quality_model_path: config.quality_model_path.clone(),
            offset_model_path: config.offset_model_path.clone(),
        }
    }

    /// Run the focus quality estimator on an image crop.
    pub fn predict_quality(
        &self,
        _image_crop: &ndarray::Array3<f32>,
    ) -> crate::error::Result<FocusQualityOutput> {
        todo!("Implement FQE ONNX inference — spec Section 7.6.1")
    }

    /// Run the focus offset predictor on extracted features.
    pub fn predict_offset(
        &self,
        _features: &super::focus_features::FocusFeatureVector,
    ) -> crate::error::Result<FocusOffsetOutput> {
        todo!("Implement FocusOffsetMLP ONNX inference — spec Section 7.6.2")
    }
}

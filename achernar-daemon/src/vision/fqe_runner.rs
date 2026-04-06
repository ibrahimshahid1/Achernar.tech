//! FQE runner — Section 8.1.
//! Focus Quality Estimator ONNX inference wrapper.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FqeOutput {
    pub quality_score: f64,
    pub confidence: f64,
}

pub struct FqeRunner {
    model_path: Option<String>,
}

impl FqeRunner {
    pub fn new(model_path: Option<String>) -> Self {
        Self { model_path }
    }

    pub fn predict(&self, _image_crop: &ndarray::Array3<f32>) -> crate::error::Result<FqeOutput> {
        todo!("Implement FQE ONNX inference — Section 8.1")
    }
}

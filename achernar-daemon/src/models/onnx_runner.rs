//! Generic ONNX session — Section 6.7.
//! Wraps ort (ONNX Runtime) with typed inference helpers.

use serde::{Deserialize, Serialize};

/// Metadata for a loaded ONNX model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxModelInfo {
    pub name: String,
    pub version: String,
    pub input_shapes: Vec<Vec<i64>>,
    pub output_shapes: Vec<Vec<i64>>,
    pub sha256: String,
}

/// Generic ONNX inference session wrapper.
pub struct OnnxRunner {
    model_path: String,
    // session: Option<ort::Session>,
}

impl OnnxRunner {
    pub fn new(model_path: &str) -> crate::error::Result<Self> {
        // TODO: Initialize ONNX Runtime session
        Ok(Self {
            model_path: model_path.to_string(),
        })
    }

    /// Run inference with a single f32 input tensor.
    pub fn run_f32(
        &self,
        _input: &ndarray::ArrayD<f32>,
    ) -> crate::error::Result<Vec<ndarray::ArrayD<f32>>> {
        todo!("Implement generic ONNX f32 inference — Section 6.7")
    }

    /// Get model info.
    pub fn info(&self) -> crate::error::Result<OnnxModelInfo> {
        todo!("Extract model metadata from ONNX session")
    }

    pub fn model_path(&self) -> &str {
        &self.model_path
    }
}

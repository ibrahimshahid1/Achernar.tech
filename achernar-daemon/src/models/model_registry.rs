//! Model registry — Section 6.7.
//! Loads models from filesystem, checks versions and SHA-256 hashes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Model metadata as stored in metadata.json per Section 7.6.10.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_name: String,
    pub version: String,
    pub task: String,
    pub framework: String,
    pub input_format: serde_json::Value,
    pub output_format: serde_json::Value,
    pub sha256: String,
    pub created_at: String,
    pub training_info: Option<serde_json::Value>,
}

/// The model registry tracks available models on the local filesystem.
pub struct ModelRegistry {
    base_path: PathBuf,
    models: HashMap<String, ModelMetadata>,
}

impl ModelRegistry {
    pub fn new(base_path: &str) -> Self {
        Self {
            base_path: PathBuf::from(base_path),
            models: HashMap::new(),
        }
    }

    /// Scan the models directory and load all metadata.json files.
    pub fn scan(&mut self) -> crate::error::Result<()> {
        todo!("Scan models/ directory for metadata.json files")
    }

    /// Get metadata for a specific model.
    pub fn get(&self, model_name: &str) -> Option<&ModelMetadata> {
        self.models.get(model_name)
    }

    /// Get the ONNX model path for a specific model.
    pub fn model_path(&self, model_name: &str) -> Option<PathBuf> {
        self.models.get(model_name).map(|m| {
            self.base_path.join(model_name).join(&m.version).join("model.onnx")
        })
    }

    /// Verify SHA-256 hash of a model file.
    pub fn verify_hash(&self, _model_name: &str) -> crate::error::Result<bool> {
        todo!("Verify SHA-256 hash of model.onnx against metadata")
    }
}

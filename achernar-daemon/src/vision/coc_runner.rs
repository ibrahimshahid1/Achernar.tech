//! COC runner — Section 8.1.
//! Cloud condition classifier with hysteresis state machine.

use super::shared_observation::CloudCondition;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CocOutput {
    pub condition: CloudCondition,
    pub probability: f64,
    pub raw_scores: Vec<f64>,
}

pub struct CocRunner {
    model_path: Option<String>,
    current_condition: CloudCondition,
    hysteresis_threshold: f64,
    consecutive_count: u32,
    required_consecutive: u32,
}

impl CocRunner {
    pub fn new(model_path: Option<String>) -> Self {
        Self {
            model_path,
            current_condition: CloudCondition::Clear,
            hysteresis_threshold: 0.3,
            consecutive_count: 0,
            required_consecutive: 3,
        }
    }

    pub fn predict(&mut self, _image: &ndarray::Array3<f32>) -> crate::error::Result<CocOutput> {
        todo!("Implement COC ONNX inference with hysteresis — Section 8.1")
    }

    pub fn current_condition(&self) -> CloudCondition {
        self.current_condition
    }
}

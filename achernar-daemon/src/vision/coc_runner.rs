//! COC runner — Section 8.1.
//! Cloud condition classifier with hysteresis state machine.

use serde::{Deserialize, Serialize};

/// Cloud condition classification used by the future COC model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CloudCondition {
    Clear,
    Thin,
    Moderate,
    Heavy,
    Overcast,
}

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

    pub fn predict(&mut self, image: &ndarray::Array3<f32>) -> crate::error::Result<CocOutput> {
        let values: Vec<f32> = image
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        if values.is_empty() {
            return Ok(CocOutput {
                condition: self.current_condition,
                probability: 0.0,
                raw_scores: vec![1.0, 0.0, 0.0, 0.0, 0.0],
            });
        }

        let max_value = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = if max_value > 1.0 { max_value } else { 1.0 };
        let normalized: Vec<f32> = values
            .iter()
            .map(|value| (*value / scale).clamp(0.0, 1.0))
            .collect();
        let mean = normalized.iter().sum::<f32>() / normalized.len() as f32;
        let stddev = (normalized
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .sum::<f32>()
            / normalized.len() as f32)
            .sqrt();
        let contrast_loss = (1.0 - stddev * 6.0).clamp(0.0, 1.0);
        let brightness = (mean * 1.5).clamp(0.0, 1.0);
        let _model_configured = self.model_path.as_ref().is_some();
        let probability = (0.65 * contrast_loss + 0.35 * brightness).clamp(0.0, 1.0);
        let proposed = condition_for_probability(probability as f64);

        if proposed as u8 != self.current_condition as u8 {
            self.consecutive_count = self.consecutive_count.saturating_add(1);
            if self.consecutive_count >= self.required_consecutive
                && (probability as f64 - self.hysteresis_threshold).abs() > 0.0
            {
                self.current_condition = proposed;
                self.consecutive_count = 0;
            }
        } else {
            self.consecutive_count = 0;
        }

        Ok(CocOutput {
            condition: self.current_condition,
            probability: probability as f64,
            raw_scores: class_scores(probability as f64),
        })
    }

    pub fn current_condition(&self) -> CloudCondition {
        self.current_condition
    }
}

fn condition_for_probability(probability: f64) -> CloudCondition {
    if probability >= 0.85 {
        CloudCondition::Overcast
    } else if probability >= 0.70 {
        CloudCondition::Heavy
    } else if probability >= 0.50 {
        CloudCondition::Moderate
    } else if probability >= 0.30 {
        CloudCondition::Thin
    } else {
        CloudCondition::Clear
    }
}

fn class_scores(probability: f64) -> Vec<f64> {
    let centers = [0.05, 0.35, 0.55, 0.75, 0.95];
    let mut scores: Vec<f64> = centers
        .iter()
        .map(|center| (1.0 - (probability - center).abs() / 0.30).max(0.0))
        .collect();
    let sum = scores.iter().sum::<f64>().max(f64::EPSILON);
    for score in &mut scores {
        *score /= sum;
    }
    scores
}

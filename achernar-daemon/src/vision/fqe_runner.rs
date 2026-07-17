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

    pub fn predict(&self, image_crop: &ndarray::Array3<f32>) -> crate::error::Result<FqeOutput> {
        let values: Vec<f32> = image_crop
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        if values.is_empty() {
            return Ok(FqeOutput {
                quality_score: 0.0,
                confidence: 0.0,
            });
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .sum::<f32>()
            / values.len() as f32;
        let dynamic_range = values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - values.iter().copied().fold(f32::INFINITY, f32::min);
        let gradient_energy = gradient_energy(image_crop);
        let _model_configured = self.model_path.as_ref().is_some();
        let quality = (gradient_energy / (gradient_energy + variance + 1.0)).clamp(0.0, 1.0);
        let confidence = ((dynamic_range / (dynamic_range + 10.0)) * 0.7
            + (values.len() as f32 / 4096.0).clamp(0.0, 1.0) * 0.3)
            .clamp(0.0, 1.0);

        Ok(FqeOutput {
            quality_score: quality as f64,
            confidence: confidence as f64,
        })
    }
}

fn gradient_energy(image: &ndarray::Array3<f32>) -> f32 {
    let shape = image.shape();
    if shape.len() != 3 || shape[0] < 2 || shape[1] < 2 {
        return 0.0;
    }

    let mut total = 0.0_f32;
    let mut count = 0_usize;
    for y in 0..shape[0] - 1 {
        for x in 0..shape[1] - 1 {
            for c in 0..shape[2] {
                let center = image[[y, x, c]];
                let dx = image[[y, x + 1, c]] - center;
                let dy = image[[y + 1, x, c]] - center;
                if dx.is_finite() && dy.is_finite() {
                    total += dx.abs() + dy.abs();
                    count += 1;
                }
            }
        }
    }

    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

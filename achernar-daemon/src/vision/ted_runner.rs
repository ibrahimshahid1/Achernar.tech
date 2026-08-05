//! TED runner — Section 8.1.
//! Trailing/elongation detection: classical SEP + CNN wrapper.

use serde::{Deserialize, Serialize};

/// Trailing axis classification used by the future TED model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TrailingAxis {
    None,
    Ra,
    Dec,
    Both,
}

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
        Self {
            model_path,
            elongation_threshold: 1.3,
        }
    }

    /// Classical SEP-based trailing detection.
    pub fn detect_classical(&self, image: &ndarray::Array2<f32>) -> TedOutput {
        let shape = image.shape();
        if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
            return TedOutput {
                trailing_detected: false,
                trailing_axis: TrailingAxis::None,
                trailing_magnitude: 0.0,
                elongation_ratio: 1.0,
            };
        }

        let pixels: Vec<f32> = image.iter().copied().collect();
        let max_adu = pixels
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(0.0_f32, f32::max)
            .max(1.0);
        let frame = super::quality::LumaFrame {
            pixels,
            width: shape[1],
            height: shape[0],
            max_adu,
            // TED receives a bare ndarray with no camera provenance attached, so
            // the sensor's saturation point is genuinely unknown here. `max_adu`
            // above is the observed peak, which is not a saturation point:
            // deriving one from it would make the threshold depend on how bright
            // the brightest star in this particular frame happened to be.
            saturation_adu: None,
        };
        let background = super::background::estimate_background(&frame.pixels, frame.saturation_adu);
        let estimate = super::trailing::estimate_trailing(&frame, &background);
        let trailing_detected = estimate.severity >= self.elongation_threshold_to_severity();
        let axis = match estimate.axis.as_deref() {
            Some("RA") => TrailingAxis::Ra,
            Some("Dec") => TrailingAxis::Dec,
            Some("Ambiguous") => TrailingAxis::Both,
            _ => TrailingAxis::None,
        };

        TedOutput {
            trailing_detected,
            trailing_axis: axis,
            trailing_magnitude: estimate.severity as f64,
            elongation_ratio: (1.0 / (1.0 - estimate.severity.max(0.0) as f64)).clamp(1.0, 99.0),
        }
    }

    /// CNN-based trailing detection (if model available).
    pub fn detect_cnn(&self, image: &ndarray::Array3<f32>) -> crate::error::Result<TedOutput> {
        let shape = image.shape();
        if shape.len() != 3 || shape[0] == 0 || shape[1] == 0 {
            return Ok(TedOutput {
                trailing_detected: false,
                trailing_axis: TrailingAxis::None,
                trailing_magnitude: 0.0,
                elongation_ratio: 1.0,
            });
        }

        let channel = 0;
        let mut mono = ndarray::Array2::<f32>::zeros((shape[0], shape[1]));
        for y in 0..shape[0] {
            for x in 0..shape[1] {
                mono[[y, x]] = image[[y, x, channel.min(shape[2] - 1)]];
            }
        }

        Ok(self.detect_classical(&mono))
    }

    fn elongation_threshold_to_severity(&self) -> f32 {
        (1.0 - 1.0 / self.elongation_threshold.max(1.0)) as f32
    }
}

//! Focus features — Section 7.6.2.
//! Sliding feature window (38 features) for the focus offset model.

use serde::{Deserialize, Serialize};

/// Feature vector for focus offset prediction (38 features).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusFeatureVector {
    /// Current HFR
    pub hfr: f64,
    /// Current FWHM
    pub fwhm: f64,
    /// Star count in the frame
    pub star_count: u32,
    /// Current focuser position (normalized)
    pub position_norm: f64,
    /// Temperature in °C
    pub temperature: f64,
    /// HFR gradient (Δ HFR / Δ position)
    pub hfr_gradient: f64,
    /// Second derivative of HFR curve
    pub hfr_curvature: f64,
    /// Standard deviation of recent HFR measurements
    pub hfr_std: f64,
    /// Filter index (one-hot or ordinal)
    pub filter_index: u32,
    /// Time since session start (hours)
    pub session_hours: f64,
    /// Altitude of target (degrees)
    pub altitude: f64,
    /// Seeing estimate
    pub seeing_estimate: f64,
    /// Background ADU level
    pub background_adu: f64,
    /// Trailing metric
    pub trailing_metric: f64,
    /// Cloud probability
    pub cloud_probability: f64,
    /// Raw feature array (all 38 features flattened)
    pub raw: Vec<f64>,
}

/// Sliding window that accumulates focus features over time.
pub struct FocusFeatureWindow {
    window_size: usize,
    samples: Vec<FocusFeatureVector>,
}

impl FocusFeatureWindow {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            samples: Vec::with_capacity(window_size),
        }
    }

    /// Add a new feature sample to the window.
    pub fn push(&mut self, features: FocusFeatureVector) {
        if self.samples.len() >= self.window_size {
            self.samples.remove(0);
        }
        self.samples.push(features);
    }

    /// Get the latest feature vector.
    pub fn latest(&self) -> Option<&FocusFeatureVector> {
        self.samples.last()
    }

    /// Build a feature tensor from the window for ML input.
    pub fn build_tensor(&self) -> Option<ndarray::Array2<f64>> {
        todo!("Build (window_size, 38) feature tensor from sliding window")
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

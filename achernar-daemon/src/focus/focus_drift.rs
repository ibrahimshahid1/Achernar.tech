//! Focus drift forecaster — Section 7.6.3.
//! Kalman filter-based temperature-compensated drift prediction.

use serde::{Deserialize, Serialize};

/// Drift prediction output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftPrediction {
    /// Predicted focus drift in steps
    pub predicted_drift_steps: f64,
    /// Confidence interval (95%)
    pub confidence_interval: f64,
    /// Whether a refocus is recommended
    pub refocus_recommended: bool,
    /// Estimated time until next refocus needed (seconds)
    pub time_to_refocus_s: Option<f64>,
}

/// Kalman drift forecaster state.
pub struct FocusDriftForecaster {
    /// State vector [position_offset, drift_rate, temp_coefficient]
    state: [f64; 3],
    /// Covariance matrix (3x3 flattened)
    covariance: [f64; 9],
    /// Process noise
    q: f64,
    /// Measurement noise
    r: f64,
    /// Last known temperature
    last_temperature: Option<f64>,
    /// Last update timestamp
    last_update: Option<chrono::DateTime<chrono::Utc>>,
}

impl FocusDriftForecaster {
    pub fn new() -> Self {
        Self {
            state: [0.0; 3],
            covariance: [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ],
            q: 0.01,
            r: 1.0,
            last_temperature: None,
            last_update: None,
        }
    }

    /// Update the Kalman filter with a new focus measurement.
    pub fn update(
        &mut self,
        _measured_offset: f64,
        _temperature: f64,
        _timestamp: chrono::DateTime<chrono::Utc>,
    ) {
        todo!("Implement Kalman filter update — spec Section 7.6.3")
    }

    /// Predict drift at a future time.
    pub fn predict(
        &self,
        _temperature: f64,
        _horizon_seconds: f64,
    ) -> DriftPrediction {
        todo!("Implement Kalman prediction — spec Section 7.6.3")
    }

    /// Reset the filter state.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

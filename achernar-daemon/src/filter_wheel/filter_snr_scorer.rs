//! Filter SNR scorer — Section 8.2 (FW-1).
//! Classical SNR potential scorer for filter recommendation.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterSnrScore {
    pub filter_name: String,
    pub estimated_snr: f64,
    pub sky_brightness: f64,
    pub target_flux: f64,
}

pub struct FilterSnrScorer;

impl FilterSnrScorer {
    /// Score each filter by its expected SNR potential given current conditions.
    pub fn score_filters(
        _filter_names: &[String],
        _moon_altitude: f64,
        _moon_phase: f64,
        _target_type: &str,
        _exposure_seconds: f64,
    ) -> Vec<FilterSnrScore> {
        todo!("Implement classical SNR potential scoring — Section 8.2 FW-1")
    }
}

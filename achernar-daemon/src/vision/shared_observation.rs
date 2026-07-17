//! SharedObservation — Section 8.1.
//! The central data structure published by the vision loop.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CloudEvent {
    None,
    CloudDetected,
    ClearedUp,
}

/// The shared observation struct published by the vision loop at 1–5 Hz.
/// All subsystem controllers consume this as their view of the sky.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedObservation {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub focus_quality: f32,
    pub predicted_hfr: f32,
    pub sharpness_score: f32,
    pub quality_uncertainty: f32,
    pub hfr: f32,
    pub fwhm: f32,
    pub seeing_estimate_arcsec: f32,
    pub star_count: u32,
    pub cloud_probability: f32,
    pub cloud_event: CloudEvent,
    pub trailing_severity: f32,
    pub trailing_axis: Option<String>,
    pub trailing_direction_pa: Option<f32>,
    pub background_adu: f32,
    pub sky_background_adu: f32,
    pub background_sigma: f32,
    pub snr_proxy_median: f32,
    pub saturation_fraction: f32,
    pub pointing_error_arcmin: Option<f32>,
    pub wcs_residual_arcsec: Option<f32>,
    pub last_solve_confidence: Option<f32>,
    pub last_solve_at: Option<chrono::DateTime<chrono::Utc>>,
    pub frame_quality_score: f32,
    pub observation_valid: bool,
}

/// CloudEvent for publishing observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionCloudEvent {
    pub source: String,
    pub event_type: String,
    pub data: SharedObservation,
}

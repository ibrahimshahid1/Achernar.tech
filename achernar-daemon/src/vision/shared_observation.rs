//! SharedObservation — Section 8.1.
//! The central data structure published by the vision loop.

use serde::{Deserialize, Serialize};

/// Trailing axis classification.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TrailingAxis {
    None,
    Ra,
    Dec,
    Both,
}

/// Cloud condition classification from COC.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CloudCondition {
    Clear,
    Thin,
    Moderate,
    Heavy,
    Overcast,
}

/// The shared observation struct published by the vision loop at 1–5 Hz.
/// All subsystems consume this as their view of the sky.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedObservation {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub frame_id: uuid::Uuid,

    // Focus quality (FQE)
    pub focus_quality: f64,
    pub focus_quality_confidence: f64,
    pub hfr: f64,
    pub fwhm: f64,
    pub star_count: u32,

    // Cloud condition (COC)
    pub cloud_condition: CloudCondition,
    pub cloud_probability: f64,

    // Trailing detection (TED)
    pub trailing_detected: bool,
    pub trailing_axis: TrailingAxis,
    pub trailing_magnitude: f64,

    // Background estimation (BGE)
    pub background_adu: f64,
    pub background_rms: f64,

    // Environment
    pub temperature: Option<f64>,
    pub humidity: Option<f64>,
    pub altitude: Option<f64>,
    pub azimuth: Option<f64>,
}

/// CloudEvent for publishing observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionCloudEvent {
    pub source: String,
    pub event_type: String,
    pub data: SharedObservation,
}

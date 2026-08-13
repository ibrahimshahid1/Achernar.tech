//! SharedObservation — Section 8.1.
//! The central data structure published by the vision loop.

use serde::{Deserialize, Serialize};

use super::quality::{QualityRejectionReason, QualityStatus, StarRejectionTallies};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CloudEvent {
    None,
    CloudDetected,
    ClearedUp,
}

/// The shared observation struct published by the vision loop at 1–5 Hz.
/// All subsystem controllers consume this as their view of the sky.
///
/// Width/quality fields are `Option`: a missing measurement is `null`, never a
/// spuriously good `0`. The former `focus_quality` / `sharpness_score` /
/// `frame_quality_score` reciprocal transforms are gone — they were three views
/// of one HFR value with no calibrated meaning. Downstream consumers derive
/// task-specific scores (calibrated focus probability, stack percentile) from
/// `hfr_px` + `quality_uncertainty` + `status`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedObservation {
    pub timestamp: chrono::DateTime<chrono::Utc>,

    // --- From FocusQualityEstimator (classical fallback for now) ---
    pub status: QualityStatus,
    pub rejection_reasons: Vec<QualityRejectionReason>,
    /// Half-flux radius (px). `None` when no valid stellar measurement.
    pub hfr_px: Option<f32>,
    /// Half-flux diameter = `2 * HFR` (px). This is what the old `fwhm` field held.
    pub hfd_px: Option<f32>,
    /// Fitted PSF FWHM (px). `None` until a PSF fit lands (`FWHM = 2*HFR` only
    /// in the Gaussian limit, so it is *not* derived from HFR).
    pub fwhm_px: Option<f32>,
    /// Angular image FWHM (arcsec). `None` until a pixel scale and a fitted FWHM
    /// exist; the old `seeing_estimate_arcsec = 2*HFR` was invalid (no pixel
    /// scale, and not calibrated seeing per Law/Tokovinin).
    pub image_fwhm_arcsec: Option<f32>,
    /// Reserved for a future calibrated task probability. The classical path is
    /// `None`; it does not compress star count/scatter into an arbitrary score.
    pub quality_uncertainty: Option<f32>,
    /// Robust std-dev of `ln(HFR)` across the stars used. This is between-star
    /// dispersion, not a calibrated standard error of the frame median.
    pub hfr_log_sigma: Option<f32>,
    /// Stars detected in the frame.
    pub star_count: u32,
    /// Stars that produced a finite HFR.
    pub star_count_used: u32,
    /// Per-reason counts of stars rejected by the source-validity gates
    /// (saturated / edge / low-SNR / elongated / blended...). Explains the gap
    /// between `star_count` and `star_count_used`. `#[serde(default)]` so
    /// observations persisted before this field deserialize cleanly.
    #[serde(default)]
    pub star_rejections: StarRejectionTallies,

    // --- FQE shadow predictions (Section 7.6.1, published alongside — never
    // replacing — the classical measurement until the promotion gate passes) ---
    /// FQE frame HFR (px): per-star predictions on the classically gated stars
    /// that also admit centered 64 px crops, pooled per `aggregate_stars`.
    #[serde(default)]
    pub fqe_hfr_px: Option<f32>,
    /// FQE frame log-HFR sigma (between-star scatter + mean calibrated aleatoric
    /// variance), followed by a post-pooling frame scale when available. Same
    /// units as `hfr_log_sigma`, though the classical value is dispersion rather
    /// than a predictive interval.
    #[serde(default)]
    pub fqe_sigma_log_hfr: Option<f32>,
    /// Number of centered, gate-surviving star crops pooled into this FQE result.
    #[serde(default)]
    pub fqe_star_count: Option<u32>,
    /// Model version directory the prediction came from (e.g. "1.1.0") —
    /// shadow evaluation needs exact provenance per observation.
    #[serde(default)]
    pub fqe_model_version: Option<String>,
    /// Whether both the per-star and post-pooling frame scales were fitted on
    /// validation groups and hash-bound to this exact model artifact.
    #[serde(default)]
    pub fqe_calibrated: Option<bool>,

    // --- From CloudObstructionClassifier ---
    pub cloud_probability: f32,
    pub cloud_event: CloudEvent,

    // --- From TrailingElongationDetector ---
    pub trailing_severity: f32,
    pub trailing_axis: Option<String>,
    pub trailing_direction_pa: Option<f32>,

    // --- From BackgroundEstimator ---
    pub background_adu: f32,
    pub sky_background_adu: f32,
    pub background_sigma: f32,
    pub snr_proxy_median: f32,
    pub saturation_fraction: f32,
    /// Whether the star-saturation gate could run on this frame.
    ///
    /// Without it, `saturation_fraction: 0.0` and `star_rejections.saturated: 0`
    /// are ambiguous — they read as "clean frame" whether the gate found nothing
    /// or was never able to test. On a 14-bit body with no declared saturation
    /// point it is always the second. `#[serde(default)]` so observations
    /// persisted before this field deserialize as `NotPerformed`, which is the
    /// truthful reading of them.
    #[serde(default)]
    pub saturation_gate: crate::vision::quality::SaturationGate,

    // --- From PlateSolver (lower cadence) ---
    pub pointing_error_arcmin: Option<f32>,
    pub wcs_residual_arcsec: Option<f32>,
    pub last_solve_confidence: Option<f32>,
    pub last_solve_at: Option<chrono::DateTime<chrono::Utc>>,

    /// Derived from `status == Valid`. Retained for backward compatibility.
    pub observation_valid: bool,
}

#[cfg(test)]
impl SharedObservation {
    /// A structurally valid observation with no measurement in it.
    ///
    /// Every width and quality field is `None` and every count is zero, which
    /// is what the vision loop publishes for a frame it could not measure.
    /// Tests set only the fields they are about, so adding a field to the
    /// struct does not require editing every test that builds one.
    pub fn empty_for_test() -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            status: super::quality::QualityStatus::NoUsableStars,
            rejection_reasons: Vec::new(),
            hfr_px: None,
            hfd_px: None,
            fwhm_px: None,
            image_fwhm_arcsec: None,
            quality_uncertainty: None,
            hfr_log_sigma: None,
            star_count: 0,
            star_count_used: 0,
            star_rejections: StarRejectionTallies::default(),
            fqe_hfr_px: None,
            fqe_sigma_log_hfr: None,
            fqe_star_count: None,
            fqe_model_version: None,
            fqe_calibrated: None,
            cloud_probability: 0.0,
            cloud_event: CloudEvent::None,
            trailing_severity: 0.0,
            trailing_axis: None,
            trailing_direction_pa: None,
            background_adu: 0.0,
            sky_background_adu: 0.0,
            background_sigma: 0.0,
            snr_proxy_median: 0.0,
            saturation_fraction: 0.0,
            saturation_gate: crate::vision::quality::SaturationGate::default(),
            pointing_error_arcmin: None,
            wcs_residual_arcsec: None,
            last_solve_confidence: None,
            last_solve_at: None,
            observation_valid: false,
        }
    }
}

/// CloudEvent for publishing observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionCloudEvent {
    pub source: String,
    pub event_type: String,
    pub data: SharedObservation,
}

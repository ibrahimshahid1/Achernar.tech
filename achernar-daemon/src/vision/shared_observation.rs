//! SharedObservation — Section 8.1.
//! The central data structure published by the vision loop.
//!
//! Besides the pixel measurements, every observation carries what the
//! equipment layer knows about the frame (equipment spec WP1B-4):
//!
//! - `equipment` (H2): whether every connected motion device was quiescent
//!   over the frame's exposure window, and if not why, with device ages. FQE,
//!   BGE, TED and COC discard inference made before the equipment settled.
//! - `coc_context` (H14): mount pointing and motion, site, Sun and Moon,
//!   weather corroboration with ages, cover, dome and dew heaters, from
//!   `EquipmentReader::coc_context()`. Wiring the user-owned COC shadow
//!   service to it is left to the user.
//! - `ted` (H15): the trailing evidence with the exposure window's tracking,
//!   pulses and wind, and a sky axis only with a WCS.
//! - `last_solve_at` and `pointing_error_arcmin` come from
//!   `plate_solve_events` (the plate-solve state carries its own timestamp),
//!   never from the previous observation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::quality::{QualityRejectionReason, QualityStatus, StarRejectionTallies};
use super::ted_runner::TedEvidence;
use crate::equipment::snapshot::{CocContext, GuiderAppState};
use crate::equipment::types::{ClockQuality, DeviceKind};

/// H2: the equipment's validity for one observed frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EquipmentValidity {
    /// `Some(true)` when every connected motion device was quiescent over
    /// the window; `Some(false)` when one moved; `None` when unknown (no
    /// data, stale data, a lost device, no window, or no motion device).
    pub quiescent: Option<bool>,
    /// Why the frame is not known to be quiescent.
    pub reason: Option<String>,
    pub settled_since: Option<DateTime<Utc>>,
    pub moved: Option<DeviceKind>,
    pub moved_event_seq: Option<u64>,
    /// The conservative exposure window of a stream frame: from two frame
    /// intervals before its arrival to its arrival.
    pub window_start: Option<DateTime<Utc>>,
    pub window_end: Option<DateTime<Utc>>,
    pub clock_quality: ClockQuality,
    /// The motion devices that were checked.
    pub kinds_checked: Vec<DeviceKind>,
    pub mount_age_ms: Option<u64>,
    pub focuser_age_ms: Option<u64>,
    pub filter_wheel_age_ms: Option<u64>,
    pub guider_age_ms: Option<u64>,
    pub mount_slewing: Option<bool>,
    pub mount_tracking: Option<bool>,
    pub pulse_guiding: Option<bool>,
    pub guider_state: Option<GuiderAppState>,
    pub resume_block_active: bool,
    /// Bumped by every segmentation event (`slew_*`, `filter_*`,
    /// `focuser_move_*`, `exposure_settings_changed`, `stream_restarted`);
    /// the cloud heuristic's baselines restart with it.
    pub segment: u64,
    pub segment_reset_by: Option<String>,
}

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
    /// Exact scientific transform/domain used for this observation, if any.
    #[serde(default)]
    pub analysis_descriptor: Option<crate::processing::debayer::AnalysisDescriptor>,
    #[serde(default)]
    pub analysis_unavailable_reason: Option<String>,
    #[serde(default)]
    pub fqe_unavailable_reason: Option<String>,

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
    /// `RA`, `Dec` or `Ambiguous` only when a WCS still describes the frame
    /// (H15); otherwise the sensor axis: `sensor_x`, `sensor_y` or
    /// `sensor_diagonal`.
    pub trailing_axis: Option<String>,
    /// Position angle of the trail on the sensor, degrees in [0, 180).
    pub trailing_direction_pa: Option<f32>,
    /// The full TED evidence, with the exposure window's equipment state.
    #[serde(default)]
    pub ted: Option<TedEvidence>,
    /// Set when trailing could not be measured on this frame; then
    /// `trailing_severity` is not a measurement.
    #[serde(default)]
    pub trailing_unavailable_reason: Option<String>,
    /// Source-selection policy behind the trailing measurement.
    #[serde(default)]
    pub trailing_policy: Option<String>,

    // --- Background (vision::background_model) ---
    /// Authoritative versioned background result: method, configuration
    /// hash, level and noise validity, units, support, and reasons. `None`
    /// on observations that predate it (legacy-unverified) and when no frame
    /// was analysed.
    #[serde(default)]
    pub background: Option<crate::vision::background_model::BackgroundResult>,
    /// Detection backbone behind `star_count` and the HFR sample.
    #[serde(default)]
    pub detection: Option<crate::vision::background_model::DetectionProvenance>,
    /// Deprecated mirror of `background.level` (only when measured). Its
    /// population depends on `background.method`; read `background` for
    /// what it means. `null` when unavailable, never a stand-in 0.
    #[serde(default)]
    pub background_adu: Option<f32>,
    /// Deprecated duplicate of `background_adu`.
    #[serde(default)]
    pub sky_background_adu: Option<f32>,
    /// Deprecated mirror of `background.rms` (only when measured and
    /// positive). `null` for unresolved or unavailable noise.
    #[serde(default)]
    pub background_sigma: Option<f32>,
    /// Deprecated name of `background.bright_pixel_significance`, a legacy
    /// pixel-significance heuristic. Not aperture or PSF SNR.
    #[serde(default)]
    pub snr_proxy_median: Option<f32>,
    /// Mirror of `background.saturation_fraction`; `null` without
    /// saturation evidence (`saturation_gate` says why).
    #[serde(default)]
    pub saturation_fraction: Option<f32>,
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

    // --- From PlateSolver (lower cadence), read from plate_solve_events ---
    /// Of the newest successful solve (H12 fills it).
    pub pointing_error_arcmin: Option<f32>,
    /// Not reported by ASTAP; stays `None`.
    pub wcs_residual_arcsec: Option<f32>,
    /// Not reported by ASTAP; stays `None`.
    pub last_solve_confidence: Option<f32>,
    pub last_solve_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    pub last_solve_event_id: Option<String>,

    // --- From the equipment layer ---
    /// H2: equipment validity of this frame.
    #[serde(default)]
    pub equipment: Option<EquipmentValidity>,
    /// H14: the COC context at this frame.
    #[serde(default)]
    pub coc_context: Option<CocContext>,

    /// Derived from `status == Valid`. Retained for backward compatibility.
    pub observation_valid: bool,
}

impl SharedObservation {
    #[cfg(test)]
    pub fn empty_for_test() -> Self {
        Self::empty()
    }

    /// Attach the authoritative background result and refresh the deprecated
    /// top-level mirrors from it. Mirrors carry only measured values.
    pub fn set_background(
        &mut self,
        result: crate::vision::background_model::BackgroundResult,
        detection: Option<crate::vision::background_model::DetectionProvenance>,
    ) {
        self.background_adu = result.level();
        self.sky_background_adu = result.level();
        self.background_sigma = result.rms();
        self.snr_proxy_median = result.bright_pixel_significance;
        self.saturation_fraction = result.saturation_fraction;
        self.background = Some(result);
        self.detection = detection;
    }

    /// How to read the background fields of this observation:
    /// `current` (a versioned `background` result is attached),
    /// `legacy_unverified` (only the old top-level numbers exist; a stored 0
    /// cannot be told apart from an unmeasured frame), or `absent`.
    pub fn background_provenance_state(&self) -> &'static str {
        match (&self.background, self.background_adu) {
            (Some(_), _) => "current",
            (None, Some(_)) => "legacy_unverified",
            (None, None) => "absent",
        }
    }

    pub fn analysis_unavailable(reason: String) -> Self {
        let mut observation = Self::empty();
        observation.status = QualityStatus::AnalysisUnavailable;
        observation.rejection_reasons = vec![QualityRejectionReason::UnqualifiedPixelDomain];
        observation.analysis_unavailable_reason = Some(reason);
        observation.fqe_unavailable_reason =
            Some("FQE disabled: scientific analysis is unavailable".to_string());
        observation
    }

    /// A structurally valid observation with no measurement in it.
    ///
    /// Every width and quality field is `None` and every count is zero, which
    /// is what the vision loop publishes for a frame it could not measure.
    /// Tests set only the fields they are about, so adding a field to the
    /// struct does not require editing every test that builds one.
    pub fn empty() -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            analysis_descriptor: None,
            analysis_unavailable_reason: None,
            fqe_unavailable_reason: None,
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
            ted: None,
            trailing_unavailable_reason: None,
            trailing_policy: None,
            background: None,
            detection: None,
            background_adu: None,
            sky_background_adu: None,
            background_sigma: None,
            snr_proxy_median: None,
            saturation_fraction: None,
            saturation_gate: crate::vision::quality::SaturationGate::default(),
            pointing_error_arcmin: None,
            wcs_residual_arcsec: None,
            last_solve_confidence: None,
            last_solve_at: None,
            last_solve_event_id: None,
            equipment: None,
            coc_context: None,
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

#[cfg(test)]
mod tests {
    use super::SharedObservation;
    use crate::vision::quality::{background_from_luma, LumaFrame};

    fn analysed() -> SharedObservation {
        let pixels: Vec<f32> = (0..64 * 64).map(|i| 100.0 + (i % 5) as f32).collect();
        let frame = LumaFrame {
            pixels,
            width: 64,
            height: 64,
            max_adu: 65535.0,
            saturation_adu: Some(65535.0),
            masks: Default::default(),
            analysis_descriptor: None,
        };
        let analysis = background_from_luma(&frame);
        let mut observation = SharedObservation::empty();
        observation.set_background(analysis.result, None);
        observation
    }

    #[test]
    fn empty_observation_publishes_no_background_numbers() {
        let json = serde_json::to_value(SharedObservation::empty()).unwrap();
        for key in [
            "background_adu",
            "sky_background_adu",
            "background_sigma",
            "snr_proxy_median",
            "saturation_fraction",
            "background",
        ] {
            assert!(json[key].is_null(), "{key} must be null, got {}", json[key]);
        }
        assert_eq!(
            SharedObservation::empty().background_provenance_state(),
            "absent"
        );
    }

    #[test]
    fn versioned_background_round_trips_and_mirrors_only_measured_values() {
        let observation = analysed();
        let json = serde_json::to_string(&observation).unwrap();
        let back: SharedObservation = serde_json::from_str(&json).unwrap();
        assert_eq!(back.background, observation.background);
        assert_eq!(
            back.background_adu,
            observation.background.as_ref().unwrap().level()
        );
        assert_eq!(back.background_provenance_state(), "current");
        assert_eq!(
            back.background.as_ref().unwrap().method.id(),
            "legacy_global_clip"
        );
    }

    #[test]
    fn legacy_payload_keeps_its_numbers_but_reads_as_unverified() {
        // An observation serialized before the versioned result existed:
        // plain numbers, possibly the old 0.0 sentinel, no `background`.
        let mut json = serde_json::to_value(SharedObservation::empty()).unwrap();
        let object = json.as_object_mut().unwrap();
        object.remove("background");
        object.remove("detection");
        object.remove("trailing_unavailable_reason");
        object.remove("trailing_policy");
        for key in [
            "background_adu",
            "sky_background_adu",
            "background_sigma",
            "snr_proxy_median",
            "saturation_fraction",
        ] {
            object.insert(key.into(), serde_json::json!(0.0));
        }
        let legacy: SharedObservation = serde_json::from_value(json).unwrap();
        assert_eq!(legacy.background_adu, Some(0.0));
        assert!(legacy.background.is_none());
        assert_eq!(legacy.background_provenance_state(), "legacy_unverified");
    }
}

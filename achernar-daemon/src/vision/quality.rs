//! Classical frame quality metrics for the shared vision bus.

use serde::{Deserialize, Serialize};

use crate::focus::classical_focus_solver::{ClassicalFocusSolver, CropSky};
use crate::hardware::camera::FrameData;
use crate::processing::debayer::{
    analyze, AnalysisDescriptor, AnalysisMasks, NativeMasks, SampleType, SensorFrame,
};

use super::background::{estimate_background, legacy_result};
use super::background_model::{
    config_sha256, BackgroundAnalysis, BackgroundField, BackgroundReason, CatalogSource,
    CropSkySource, DetectionMethod, DetectionProvenance, NoiseStatus, SourceCatalog,
};

#[derive(Debug, Clone)]
pub struct LumaFrame {
    pub pixels: Vec<f32>,
    pub width: usize,
    pub height: usize,
    /// Full scale of the *container* the samples arrived in: 255 for an 8-bit
    /// stream, 65535 for a 16-bit one. Not the sensor's saturation point, and
    /// using it as one is the bug [`LumaFrame::saturation_adu`] exists to fix.
    pub max_adu: f32,
    /// Native clipping/nonlinearity evidence, when explicitly declared. The
    /// scalar is provenance only for reconstructed intensity: source rejection
    /// uses the propagated native-saturation mask, never a proxy peak.
    /// Missing thresholds remain unknown at every container depth.
    pub saturation_adu: Option<f32>,
    /// Masks refer to the linear intensity plane, including the reconstruction
    /// footprint of invalid or clipped native samples.
    pub masks: AnalysisMasks,
    pub analysis_descriptor: Option<AnalysisDescriptor>,
}

impl LumaFrame {
    pub fn measurement_pixel_valid(&self, index: usize) -> bool {
        [
            self.masks.bad_pixel.get(index),
            self.masks.invalid.get(index),
            self.masks.edge.get(index),
        ]
        .iter()
        .all(|value| value.copied().unwrap_or(0) == 0)
    }

    pub fn native_saturation_at(&self, index: usize) -> bool {
        self.masks
            .native_saturation
            .get(index)
            .copied()
            .unwrap_or(0)
            != 0
    }

    /// Which masks constrain measurements on this frame (`provided` or
    /// `unavailable`), for result provenance. Versioned analysis frames take
    /// the native-mask states from their descriptor; the reconstruction edge
    /// mask always exists there. Frames without a descriptor only have the
    /// masks a caller attached.
    pub fn mask_provenance(&self) -> std::collections::BTreeMap<String, String> {
        let attached = |mask: &Vec<u8>| {
            if mask.len() == self.pixels.len() && !mask.is_empty() {
                "provided"
            } else {
                "unavailable"
            }
        };
        let mut states = std::collections::BTreeMap::new();
        match &self.analysis_descriptor {
            Some(descriptor) => {
                for name in ["bad_pixel", "invalid", "native_saturation"] {
                    let state = descriptor
                        .mask_states
                        .get(name)
                        .map(String::as_str)
                        .unwrap_or("unavailable");
                    states.insert(name.to_string(), state.to_string());
                }
                states.insert("edge".to_string(), "provided".to_string());
            }
            None => {
                states.insert("bad_pixel".into(), attached(&self.masks.bad_pixel).into());
                states.insert("invalid".into(), attached(&self.masks.invalid).into());
                states.insert("edge".into(), attached(&self.masks.edge).into());
                states.insert(
                    "native_saturation".into(),
                    attached(&self.masks.native_saturation).into(),
                );
            }
        }
        states
    }

    /// The evidence a saturation fraction would rest on. `unknown` means a
    /// fraction of 0 would only say that nothing was tested.
    pub fn saturation_evidence(&self) -> &'static str {
        match saturation_gate(self) {
            SaturationGate::Applied { .. } if self.analysis_descriptor.is_some() => {
                "native_threshold"
            }
            SaturationGate::Applied { .. } => "legacy_peak_threshold",
            SaturationGate::NativeMaskApplied => "native_mask",
            SaturationGate::NotPerformed => "unknown",
        }
    }

    pub fn masks_have_valid_shape(&self) -> bool {
        [
            &self.masks.bad_pixel,
            &self.masks.invalid,
            &self.masks.edge,
            &self.masks.native_saturation,
        ]
        .iter()
        .all(|mask| {
            (mask.len() == self.pixels.len()
                || (self.analysis_descriptor.is_none() && mask.is_empty()))
                && mask.iter().all(|value| *value <= 1)
        })
    }
}

/// Legacy background (`legacy_global_clip`) over usable linear samples:
/// finite pixels outside the bad/invalid/edge and native-saturation masks.
/// Malformed geometry or masks give an unavailable result, never zeros.
pub fn background_from_luma(frame: &LumaFrame) -> BackgroundAnalysis {
    if !frame_geometry_valid(frame) {
        let mut analysis = legacy_result(frame, None);
        analysis.result.reasons = vec![BackgroundReason::MalformedGeometry];
        analysis.result.saturation_fraction = None;
        return analysis;
    }
    let valid: Vec<f32> = frame
        .pixels
        .iter()
        .enumerate()
        .filter(|(index, _)| {
            frame.measurement_pixel_valid(*index) && !frame.native_saturation_at(*index)
        })
        .map(|(_, value)| *value)
        .collect();
    let stats = estimate_background(
        &valid,
        if frame.analysis_descriptor.is_some() {
            None
        } else {
            frame.saturation_adu
        },
    );
    legacy_result(frame, stats.as_ref())
}

/// Pixel count matches the geometry and every mask is a 0/1 plane of the
/// right size (or absent on a frame without a descriptor).
pub fn frame_geometry_valid(frame: &LumaFrame) -> bool {
    !frame.pixels.is_empty()
        && frame.width > 0
        && frame.height > 0
        && frame.pixels.len() == frame.width.saturating_mul(frame.height)
        && frame.masks_have_valid_shape()
}

/// Whether the star-saturation gate could be applied to a frame.
///
/// Published so a consumer — the FQE labeler above all — can tell "no stars
/// were saturated" from "we never checked". Only the first is evidence.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SaturationGate {
    /// Applied from this native threshold. Versioned analysis uses a native
    /// clipping mask; legacy independent mono callers retain the 98% gate.
    Applied { threshold_adu: f32 },
    /// Not applied: the sensor's saturation point is unknown for this frame.
    /// Measurements are still produced, but any star with a clipped, flat core
    /// is in them, biasing HFR low exactly where the V-curve needs resolution.
    NotPerformed,
    /// An explicit native clipping mask was supplied without a numeric native
    /// threshold (for example for signed, calibrated residual samples).
    NativeMaskApplied,
}

impl SaturationGate {
    pub fn was_applied(&self) -> bool {
        matches!(
            self,
            SaturationGate::Applied { .. } | SaturationGate::NativeMaskApplied
        )
    }
}

impl Default for SaturationGate {
    /// `NotPerformed`. Chosen so that an observation persisted before this
    /// field existed deserializes to the truth about itself — no gate ran on
    /// those frames either.
    fn default() -> Self {
        SaturationGate::NotPerformed
    }
}

/// Why a frame did (not) yield a usable focus measurement. `Valid` means at
/// least one detected star produced a finite HFR. Anything else means no width
/// could be measured — which must serialize as `null`, never a spuriously good 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityStatus {
    Valid,
    EmptyFrame,
    AnalysisUnavailable,
    NoSources,
    NoUsableStars,
    /// No background level exists, so detection did not run.
    BackgroundUnavailable,
    /// A background level exists but detection could not run (unresolved
    /// noise for a relative threshold, or a backend extraction failure).
    DetectionUnavailable,
}

/// Specific reasons a frame was rejected, for diagnostics and dashboards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityRejectionReason {
    NoFinitePixels,
    UnqualifiedPixelDomain,
    NoStellarSources,
    NoValidHfr,
    BackgroundUnavailable,
    NoiseUnresolved,
    DetectionFailed,
}

/// Classical per-frame quality measurements.
///
/// This deliberately publishes *physical measurements* plus validity, not a
/// single opaque scalar. In particular there is no `focus_quality`/`sharpness`
/// reciprocal transform of HFR anymore: those carried no calibrated meaning and
/// presented one HFR value as if it were independent evidence. Consumers derive
/// their own task-specific scores from `hfr_px` + uncertainty + `status`.
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    pub status: QualityStatus,
    pub rejection_reasons: Vec<QualityRejectionReason>,
    /// Half-flux radius (px). `None` when no star produced a finite HFR — never
    /// `0.0`, which is the mathematically *best* value for a width metric and
    /// would let a blank frame win a lower-is-better comparison.
    pub hfr_px: Option<f32>,
    /// Half-flux diameter = `2 * HFR` (px). This is the value the old `fwhm`
    /// field actually held while mislabeling itself as a fitted FWHM.
    pub hfd_px: Option<f32>,
    /// Fitted PSF FWHM (px). Unavailable in the classical path: a Moffat/Gaussian
    /// fit is required, and `FWHM = 2*HFR` holds only in the Gaussian limit.
    pub fwhm_px: Option<f32>,
    /// Stars detected as bright connected components.
    pub star_count_detected: u32,
    /// Stars that produced a finite HFR (the sample the median is built from).
    pub star_count_used: u32,
    /// Reserved compatibility field for a future calibrated task probability.
    /// The classical path leaves it `None`: squashing scatter or star count into
    /// `[0,1]` would create another arbitrary score with no coverage semantics.
    pub quality_uncertainty: Option<f32>,
    /// Robust standard deviation of `ln(HFR)` across the stars used. This is an
    /// empirical between-star dispersion diagnostic, not a calibrated standard
    /// error of the frame median. `None` when fewer than three stars are usable.
    pub hfr_log_sigma: Option<f32>,
    /// Per-reason counts of examined-but-rejected stars (see
    /// [`StarRejectionTallies`]). Diagnostic: "14 detected / 5 used" becomes
    /// explainable ("6 saturated, 2 at the edge, 1 blended").
    pub star_rejections: StarRejectionTallies,
    /// The gate-surviving stars (centroid + HFR) the frame median was built
    /// from. Consumed by the FQE shadow path; not serialized onto the bus.
    pub used_stars: Vec<UsedStar>,
    /// Whether the saturation gate ran at all. `star_rejections.saturated == 0`
    /// is only meaningful when this is `Applied`; under `NotPerformed` it means
    /// nothing was checked, and a training label built from the frame inherits
    /// whatever clipped cores were present.
    pub saturation_gate: SaturationGate,
    /// Detection backbone, configuration and sky source behind the sample.
    /// `None` when detection did not run.
    pub detection: Option<DetectionProvenance>,
    /// Sources the detector handed to the star gates (diagnostics and
    /// evaluation; not serialized onto the bus).
    pub candidates: Vec<StarCandidate>,
    /// Accepted stars that needed a v2 rule (not rejections).
    pub gate_diagnostics: GateDiagnostics,
    /// sha256 of the canonical star-gate configuration record
    /// ([`FocusSourcePolicy::config_json`]).
    pub star_gates_config_sha256: String,
}

impl QualityMetrics {
    /// Legacy convenience: the observation is valid iff a width was measured.
    pub fn observation_valid(&self) -> bool {
        matches!(self.status, QualityStatus::Valid)
    }
}

/// A detected source handed to the star gates.
#[derive(Debug, Clone, PartialEq)]
pub struct StarCandidate {
    pub x: f32,
    pub y: f32,
    pub area: usize,
    /// Legacy: sum of `max(I-B,0)` over the component. SEP: unfiltered
    /// residual flux over the footprint. Used for ranking only.
    pub flux: f32,
    /// Raw input value at the brightest pixel (legacy peak saturation gate).
    pub peak: f32,
    /// Bounding box of the detection footprint (inclusive, frame coords).
    /// Drives the adaptive crop size and the edge-truncation gate.
    pub min_x: usize,
    pub max_x: usize,
    pub min_y: usize,
    pub max_y: usize,
    /// The detector split this source out of a blend (SEP deblending).
    pub deblended: bool,
    /// Catalog objects behind this candidate: 1 for an ordinary source;
    /// `n >= 2` for a regrouped family of deblended children
    /// (`regroup_deblended`), whose footprint, flux and box are the union.
    pub family_size: u32,
}

/// Which gate-5 blend test runs (`[vision.star_gates] blend_test`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlendTest {
    /// Two or more distinct core peaks above 5 sigma (the original gate;
    /// it rejects every donut whose ring breaks into maxima).
    PeakCountV1,
    /// `docs/bge-v2-spec.md` §1: peaks must be separated by a significant
    /// saddle (bottleneck merge over above-threshold pixels), and a
    /// blend-flagged source that is one annulus (spider vanes, central
    /// obstruction) is exempt.
    #[default]
    DipAnnulusV2,
}

impl BlendTest {
    pub fn id(self) -> &'static str {
        match self {
            BlendTest::PeakCountV1 => "peak_count_v1",
            BlendTest::DipAnnulusV2 => "dip_annulus_v2",
        }
    }
}

/// `[vision.star_gates]` in `achernar.toml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StarGateSettings {
    #[serde(default)]
    pub blend_test: BlendTest,
    /// Re-merge SEP deblended children into their parent family and judge
    /// the family with the blend gate (`docs/bge-v2-spec.md` §2). Has no
    /// effect when `accept_deblended_for_focus` is on.
    #[serde(default = "default_regroup_deblended")]
    pub regroup_deblended: bool,
}

fn default_regroup_deblended() -> bool {
    true
}

impl Default for StarGateSettings {
    fn default() -> Self {
        Self {
            blend_test: BlendTest::default(),
            regroup_deblended: default_regroup_deblended(),
        }
    }
}

/// Focus-star acceptance policy that is independent of the detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FocusSourcePolicy {
    /// Accept deblended children as focus stars. Off by default: a split
    /// blend's wings are shared with its neighbour.
    pub accept_deblended: bool,
    pub blend_test: BlendTest,
    /// See [`StarGateSettings::regroup_deblended`].
    pub regroup_deblended: bool,
}

impl Default for FocusSourcePolicy {
    fn default() -> Self {
        Self::new(&StarGateSettings::default(), false)
    }
}

impl FocusSourcePolicy {
    /// `accept_deblended` comes from `[vision.background.detection]
    /// accept_deblended_for_focus`.
    pub fn new(star_gates: &StarGateSettings, accept_deblended: bool) -> Self {
        Self {
            accept_deblended,
            blend_test: star_gates.blend_test,
            regroup_deblended: star_gates.regroup_deblended,
        }
    }

    /// Regrouping runs only when deblended children are not accepted
    /// individually.
    pub fn regroups(&self) -> bool {
        self.regroup_deblended && !self.accept_deblended
    }

    /// Canonical star-gate configuration record (`docs/bge-v2-spec.md`
    /// §1.6), hashed with [`config_sha256`].
    pub fn config_json(&self) -> serde_json::Value {
        match self.blend_test {
            BlendTest::PeakCountV1 => serde_json::json!({
                "blend_test": BlendTest::PeakCountV1.id(),
                "peak_window_half": BLEND_MIN_SEPARATION,
                "threshold_sigma": BLEND_THRESHOLD_SIGMA,
                "regroup_deblended": self.regroup_deblended,
                "accept_deblended": self.accept_deblended,
            }),
            BlendTest::DipAnnulusV2 => serde_json::json!({
                "blend_test": BlendTest::DipAnnulusV2.id(),
                "peak_window_half": BLEND_MIN_SEPARATION,
                "threshold_sigma": BLEND_THRESHOLD_SIGMA,
                "dip_fraction": DIP_FRACTION,
                "dip_min_sigma": DIP_MIN_SIGMA,
                "annulus": {
                    "bins": ANN_BINS,
                    "bin_fraction": ANN_BIN_FRACTION,
                    "min_coverage": ANN_MIN_COVERAGE,
                    "half_width": ANN_HALF_WIDTH,
                    "hole_radius": ANN_HOLE_RADIUS,
                    "hole_fraction": ANN_HOLE_FRACTION,
                    "min_radius_px": ANN_MIN_RADIUS_PX,
                    "radius_tol": ANN_RADIUS_TOL,
                },
                "regroup_deblended": self.regroup_deblended,
                "accept_deblended": self.accept_deblended,
            }),
        }
    }

    pub fn config_sha256(&self) -> String {
        config_sha256(&self.config_json())
    }
}

/// Accepted stars (gate survivors with a finite HFR) that needed a v2 rule.
/// Diagnostics, not rejections (`docs/bge-v2-spec.md` §1.7).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateDiagnostics {
    /// Blend-flagged by the saddle test and accepted as one annulus (§1.4).
    pub annulus_exempt: u32,
    /// Regrouped families of deblended children that were accepted (§2).
    pub regrouped_accepted: u32,
}

/// Per-reason counts of stars examined but rejected by the source-validity
/// gates (ports of `ml/labels/psf_measure.py::measure_star`, so the live path
/// and the FQE label path reject the same stars for the same reasons).
///
/// A single star may increment several tallies (all triggered reasons are
/// recorded, matching the Python labeler), so the sum can exceed the number of
/// rejected stars.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StarRejectionTallies {
    /// Measurement support contains clipped native samples (legacy mono uses
    /// a peak at/above 98% of the declared limit): flat cores bias HFR low and would
    /// flatten the V-curve exactly near best focus.
    pub saturated: u32,
    /// Above-threshold pixels reach the frame border: flux is truncated, so the
    /// half-flux radius of what remains is meaningless.
    pub edge_truncated: u32,
    /// Aperture SNR below `MIN_APERTURE_SNR`: the width of a noise-dominated
    /// blob measures the noise, not the PSF.
    pub low_snr: u32,
    /// Fewer than `MIN_CORE_PIXELS` pixels above 3 sigma inside the aperture.
    pub too_few_pixels: u32,
    /// Second-moment ellipticity above `MAX_ELLIPTICITY`: trailed/astigmatic
    /// sources broaden the radial profile and poison a *focus* width label
    /// (trailing is measured separately by `vision::trailing`).
    pub elongated: u32,
    /// Two or more distinct core peaks: a close double whose merged profile
    /// reads spuriously broad (SEP deblends these; the crop path drops them).
    pub blended: u32,
    /// Passed every gate but the curve-of-growth returned no finite width.
    pub invalid_hfr: u32,
    /// Source measurement support touches bad, invalid or reconstruction-edge pixels.
    #[serde(default)]
    pub masked_support: u32,
    /// The detector deblended the source and the focus policy does not
    /// accept deblended children; or (`regroup_deblended`) the regrouped
    /// family failed the blend gate, confirming the detector's split.
    #[serde(default)]
    pub deblended: u32,
    /// `masked_local_v2`: the measurement crop contains a pixel whose RMS is
    /// locally invalid, so no noise supports the gates there.
    #[serde(default)]
    pub invalid_noise_support: u32,
}

/// A star that survived every validity gate and produced a finite HFR. The
/// centroid lets the FQE shadow path cut its 64 px crops from exactly the
/// stars the classical measurement used — same sample, comparable numbers.
#[derive(Debug, Clone, Copy)]
pub struct UsedStar {
    pub x: f32,
    pub y: f32,
    pub hfr_px: f32,
}

// --- Source-validity gate thresholds (parity with ml/labels/psf_measure.py) ---
/// Stop once this many stars produced a usable HFR.
const MAX_STARS_USED: usize = 16;
/// Never examine more candidates than this per frame (1-5 Hz budget).
const MAX_STARS_EXAMINED: usize = 48;
/// Peak >= this fraction of `max_adu` counts as saturated (matches
/// `background::saturation_threshold`).
const SATURATION_FRACTION_OF_MAX: f32 = 0.98;
/// Above-threshold pixels within this many px of the frame border => truncated.
const EDGE_MARGIN_PX: usize = 2;
/// Minimum aperture SNR for a width measurement.
const MIN_APERTURE_SNR: f32 = 5.0;
/// Minimum pixels above 3 sigma inside the aperture.
const MIN_CORE_PIXELS: usize = 5;
/// Maximum second-moment ellipticity (1 - b/a).
const MAX_ELLIPTICITY: f32 = 0.35;
/// Local-maximum window half-size for the blend (double-star) test.
const BLEND_MIN_SEPARATION: usize = 3;
/// Blend-test threshold: residual above this many noise sigma.
const BLEND_THRESHOLD_SIGMA: f64 = 5.0;

// `dip_annulus_v2` parameters (docs/bge-v2-spec.md §1.5): provisional
// engineering choices from preliminary simulations, versioned through the
// star-gate configuration hash.
/// A resolved peak is significant when its dip exceeds
/// `max(DIP_FRACTION * p, DIP_MIN_SIGMA * sigma_p)`.
pub const DIP_FRACTION: f64 = 0.15;
pub const DIP_MIN_SIGMA: f64 = 4.0;
/// Annulus exemption: smallest common radius of the peaks (px).
pub const ANN_MIN_RADIUS_PX: f64 = 2.5;
/// Every peak within `ANN_RADIUS_TOL * r + 1` px of the common radius.
pub const ANN_RADIUS_TOL: f64 = 0.35;
/// Ring half-width `max(1.5, ANN_HALF_WIDTH * r)` px.
pub const ANN_HALF_WIDTH: f64 = 0.25;
/// Azimuthal bins of the ring.
pub const ANN_BINS: usize = 16;
/// A bin covers when its mean reaches this fraction of the top-4 mean.
pub const ANN_BIN_FRACTION: f64 = 0.5;
/// Fraction of covering bins required.
pub const ANN_MIN_COVERAGE: f64 = 0.625;
/// Hole radius `max(1, ANN_HOLE_RADIUS * r)` px.
pub const ANN_HOLE_RADIUS: f64 = 0.35;
/// The hole mean must be at most this fraction of the top-4 bin mean.
pub const ANN_HOLE_FRACTION: f64 = 0.6;
/// Adaptive crop bounds. 32 px caps measurable HFR at ~15 px, which truncates
/// defocused stars at the V-curve wings; the crop now scales with the source
/// bounding box up to 128 px (HFR cap ~63 px).
const CROP_MIN_PX: usize = 32;
const CROP_MAX_PX: usize = 128;

pub fn luma_from_frame(frame: &FrameData) -> crate::error::Result<LumaFrame> {
    let descriptor = frame.analysis_sensor_descriptor()?;
    let analysis = analyze(&SensorFrame {
        descriptor,
        data: &frame.data,
        masks: NativeMasks::default(),
    })
    .map_err(|error| crate::error::HardwareError::Communication(error.to_string()))?;
    Ok(luma_from_analysis(analysis))
}

pub fn luma_from_analysis(analysis: crate::processing::debayer::AnalysisFrame) -> LumaFrame {
    let sensor = &analysis.descriptor.sensor;
    let max_adu = match sensor.sample_type {
        SampleType::U8 => 255.0,
        SampleType::U16 => 65535.0,
        SampleType::F32 => f32::MAX,
    };
    LumaFrame {
        width: sensor.width,
        height: sensor.height,
        saturation_adu: sensor.saturation.as_ref().map(|threshold| threshold.adu),
        max_adu,
        pixels: analysis.pixels,
        masks: analysis.masks,
        analysis_descriptor: Some(analysis.descriptor),
    }
}

/// [`estimate_quality_with_policy`] with the default policy
/// (`dip_annulus_v2`, `regroup_deblended`, deblended children not accepted).
pub fn estimate_quality(frame: &LumaFrame, background: &BackgroundAnalysis) -> QualityMetrics {
    estimate_quality_with_policy(frame, background, FocusSourcePolicy::default())
}

/// Detect and measure focus stars against the frame's background analysis.
///
/// Detection uses the SEP catalog when the backend produced one, otherwise
/// the legacy global-threshold components of a uniform background. The star
/// gates and HFR use the same background: the parent spatial map evaluated in
/// each crop, or (legacy) the crop's own median sky.
pub fn estimate_quality_with_policy(
    frame: &LumaFrame,
    background: &BackgroundAnalysis,
    policy: FocusSourcePolicy,
) -> QualityMetrics {
    // Every result names the gate configuration it was measured under.
    let mut metrics = measure_quality(frame, background, policy);
    metrics.star_gates_config_sha256 = policy.config_sha256();
    metrics
}

fn measure_quality(
    frame: &LumaFrame,
    background: &BackgroundAnalysis,
    policy: FocusSourcePolicy,
) -> QualityMetrics {
    let gate = saturation_gate(frame);

    if !frame_geometry_valid(frame) {
        return invalid_metrics(
            QualityStatus::EmptyFrame,
            QualityRejectionReason::NoFinitePixels,
            0,
            StarRejectionTallies::default(),
            gate,
            None,
        );
    }

    let Some(field) = background.field.as_ref() else {
        return invalid_metrics(
            QualityStatus::BackgroundUnavailable,
            QualityRejectionReason::BackgroundUnavailable,
            0,
            StarRejectionTallies::default(),
            gate,
            None,
        );
    };

    let (mut stars, detection) = match (&background.catalog, field) {
        (Some(catalog), _) => (
            candidates_from_catalog(catalog, policy, frame.width, frame.height),
            catalog.provenance.clone(),
        ),
        (None, BackgroundField::Uniform { level, sigma }) => {
            detect_star_candidates(frame, *level, *sigma)
        }
        (None, BackgroundField::Map { .. }) => {
            let reason = if background.result.noise_status == NoiseStatus::Measured {
                QualityRejectionReason::DetectionFailed
            } else {
                QualityRejectionReason::NoiseUnresolved
            };
            return invalid_metrics(
                QualityStatus::DetectionUnavailable,
                reason,
                0,
                StarRejectionTallies::default(),
                gate,
                None,
            );
        }
    };
    stars.sort_by(|a, b| {
        b.flux
            .total_cmp(&a.flux)
            .then_with(|| b.peak.total_cmp(&a.peak))
            .then_with(|| b.area.cmp(&a.area))
    });

    let star_count_detected = stars.len() as u32;
    if star_count_detected == 0 {
        return invalid_metrics(
            QualityStatus::NoSources,
            QualityRejectionReason::NoStellarSources,
            0,
            StarRejectionTallies::default(),
            gate,
            Some(detection),
        );
    }

    let (used_stars, star_rejections, gate_diagnostics) =
        measure_stars(frame, &stars, field, policy);
    let hfrs: Vec<f32> = used_stars.iter().map(|star| star.hfr_px).collect();
    let star_count_used = hfrs.len() as u32;
    let hfr_px = median(&hfrs);

    match hfr_px {
        Some(hfr) if hfr.is_finite() && hfr > 0.0 => {
            // Preserve the physical between-star log-HFR scatter. It is useful
            // diagnostics, but it is not predictive interval calibration and is
            // therefore not converted into a unitless "quality uncertainty".
            let hfr_log_sigma = robust_log_sigma(&hfrs);
            QualityMetrics {
                status: QualityStatus::Valid,
                rejection_reasons: Vec::new(),
                hfr_px: Some(hfr),
                hfd_px: Some(hfr * 2.0),
                // A fitted FWHM needs a PSF fit (not implemented in the classical
                // path); `2*HFR` is HFD, not FWHM, so we leave this None.
                fwhm_px: None,
                star_count_detected,
                star_count_used,
                quality_uncertainty: None,
                hfr_log_sigma,
                star_rejections,
                used_stars,
                saturation_gate: gate,
                detection: Some(detection),
                candidates: stars,
                gate_diagnostics,
                star_gates_config_sha256: String::new(),
            }
        }
        // Stars were detected but none survived the validity gates with a usable
        // HFR. `star_count_detected` and the per-reason tallies stay as real
        // diagnostics; widths remain None.
        _ => {
            let mut metrics = invalid_metrics(
                QualityStatus::NoUsableStars,
                QualityRejectionReason::NoValidHfr,
                star_count_detected,
                star_rejections,
                gate,
                Some(detection),
            );
            metrics.candidates = stars;
            metrics.gate_diagnostics = gate_diagnostics;
            metrics
        }
    }
}

fn invalid_metrics(
    status: QualityStatus,
    reason: QualityRejectionReason,
    star_count_detected: u32,
    star_rejections: StarRejectionTallies,
    saturation_gate: SaturationGate,
    detection: Option<DetectionProvenance>,
) -> QualityMetrics {
    QualityMetrics {
        status,
        rejection_reasons: vec![reason],
        hfr_px: None,
        hfd_px: None,
        fwhm_px: None,
        star_count_detected,
        star_count_used: 0,
        quality_uncertainty: None,
        hfr_log_sigma: None,
        star_rejections,
        used_stars: Vec::new(),
        saturation_gate,
        detection,
        candidates: Vec::new(),
        gate_diagnostics: GateDiagnostics::default(),
        star_gates_config_sha256: String::new(),
    }
}

fn median(values: &[f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    Some(median_sorted(&sorted))
}

fn median_sorted(sorted: &[f32]) -> f32 {
    if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        0.5 * (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2])
    }
}

/// Robust standard deviation of `ln(HFR)` over the stars used, via the median
/// absolute deviation (MAD x 1.4826). This is the empirical dispersion of the
/// per-star measurements — the classical analogue of the FQE's `sigma_log_hfr`.
/// Returns `None` for fewer than three positive HFRs (too few to estimate
/// scatter — reporting 0 would be falsely confident).
fn robust_log_sigma(hfrs: &[f32]) -> Option<f32> {
    let mut logs: Vec<f32> = hfrs
        .iter()
        .filter(|hfr| **hfr > 0.0 && hfr.is_finite())
        .map(|hfr| hfr.ln())
        .collect();
    if logs.len() < 3 {
        return None;
    }
    logs.sort_by(|a, b| a.total_cmp(b));
    let median = median_sorted(&logs);
    let mut deviations: Vec<f32> = logs.iter().map(|value| (value - median).abs()).collect();
    deviations.sort_by(|a, b| a.total_cmp(b));
    let mad = median_sorted(&deviations) * 1.4826;
    Some(mad)
}

/// Canonical configuration of the legacy detector (its floors are labelled
/// heuristics, not a camera calibration).
pub fn legacy_detection_config_json() -> serde_json::Value {
    serde_json::json!({
        "method": DetectionMethod::LegacyConnectedComponents.id(),
        "method_version": 1,
        "threshold": "max(B + 5*max(sigma,1), B + 8) with the global level B and scale sigma",
        "thresh_sigma": 5.0,
        "sigma_floor": 1.0,
        "min_excess": 8.0,
        "connectivity": 8,
        "min_area": 2,
        "max_area": MAX_SOURCE_AREA,
        "centroid_weight": "max(I-B,0)",
    })
}

/// Largest detection footprint treated as a candidate star (both detectors).
const MAX_SOURCE_AREA: usize = 5_000;

/// Legacy detector: 8-connected components above a global threshold.
fn detect_star_candidates(
    frame: &LumaFrame,
    level: f32,
    measured_sigma: f32,
) -> (Vec<StarCandidate>, DetectionProvenance) {
    let sigma = measured_sigma.max(1.0);
    let noise_floor_applied = !(measured_sigma >= 1.0);
    let threshold = (level + sigma * 5.0).max(level + 8.0);
    let mut visited = vec![false; frame.pixels.len()];
    let mut stars = Vec::new();

    for idx in 0..frame.pixels.len() {
        let value = frame.pixels[idx];
        if visited[idx]
            || !frame.measurement_pixel_valid(idx)
            || !value.is_finite()
            || value < threshold
        {
            continue;
        }

        let mut stack = vec![idx];
        visited[idx] = true;
        let mut area = 0_usize;
        let mut flux = 0.0_f32;
        let mut weighted_x = 0.0_f32;
        let mut weighted_y = 0.0_f32;
        let mut peak = value;
        let mut bb_min_x = usize::MAX;
        let mut bb_max_x = 0_usize;
        let mut bb_min_y = usize::MAX;
        let mut bb_max_y = 0_usize;

        while let Some(current) = stack.pop() {
            let current_value = frame.pixels[current];
            let signal = (current_value - level).max(0.0);
            let x = current % frame.width;
            let y = current / frame.width;

            area += 1;
            flux += signal;
            weighted_x += x as f32 * signal;
            weighted_y += y as f32 * signal;
            peak = peak.max(current_value);
            bb_min_x = bb_min_x.min(x);
            bb_max_x = bb_max_x.max(x);
            bb_min_y = bb_min_y.min(y);
            bb_max_y = bb_max_y.max(y);

            let min_x = x.saturating_sub(1);
            let max_x = (x + 1).min(frame.width - 1);
            let min_y = y.saturating_sub(1);
            let max_y = (y + 1).min(frame.height - 1);

            for ny in min_y..=max_y {
                for nx in min_x..=max_x {
                    if nx == x && ny == y {
                        continue;
                    }
                    let neighbor = ny * frame.width + nx;
                    let neighbor_value = frame.pixels[neighbor];
                    if visited[neighbor]
                        || !frame.measurement_pixel_valid(neighbor)
                        || !neighbor_value.is_finite()
                        || neighbor_value < threshold
                    {
                        continue;
                    }
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }

        if area >= 2 && area <= MAX_SOURCE_AREA && flux > 0.0 {
            stars.push(StarCandidate {
                x: weighted_x / flux,
                y: weighted_y / flux,
                area,
                flux,
                peak,
                min_x: bb_min_x,
                max_x: bb_max_x,
                min_y: bb_min_y,
                max_y: bb_max_y,
                deblended: false,
                family_size: 1,
            });
        }
    }

    let provenance = DetectionProvenance {
        method: DetectionMethod::LegacyConnectedComponents,
        config_sha256: config_sha256(&legacy_detection_config_json()),
        crop_sky_source: CropSkySource::LegacyCropMedian,
        noise_floor_applied,
        objects_detected: stars.len() as u32,
        objects_deblended: 0,
        objects_truncated: 0,
    };
    (stars, provenance)
}

/// SEP catalog objects as star candidates, with the same footprint-size and
/// positive-flux limits as the legacy detector.
///
/// With [`FocusSourcePolicy::regroups`] (`docs/bge-v2-spec.md` §2), the
/// deblended children of one parent detection (`achernar_sep::
/// deblend_families`, over all catalog children) become one candidate,
/// emitted at the position of the family's first member: summed flux and
/// npix, flux-weighted position, the largest raw peak and the union box. The
/// size and flux limits then apply to the family. A family of one child
/// stays an ordinary deblended child.
fn candidates_from_catalog(
    catalog: &SourceCatalog,
    policy: FocusSourcePolicy,
    width: usize,
    height: usize,
) -> Vec<StarCandidate> {
    let keep = |area: usize, flux: f32| area <= MAX_SOURCE_AREA && flux > 0.0 && flux.is_finite();
    let single = |object: &CatalogSource| StarCandidate {
        x: object.x,
        y: object.y,
        area: object.npix as usize,
        flux: object.flux,
        peak: object.peak_raw,
        min_x: object.xmin,
        max_x: object.xmax,
        min_y: object.ymin,
        max_y: object.ymax,
        deblended: object.deblended,
        family_size: 1,
    };
    if !policy.regroups() {
        return catalog
            .objects
            .iter()
            .filter(|object| keep(object.npix as usize, object.flux))
            .map(single)
            .collect();
    }
    let families = achernar_sep::deblend_families(
        catalog
            .objects
            .iter()
            .map(|o| (o.pixels.as_slice(), o.deblended)),
        width,
        height,
    );
    let mut family_of = vec![None; catalog.objects.len()];
    for (f, members) in families.iter().enumerate() {
        if members.len() >= 2 {
            for &i in members {
                family_of[i] = Some(f);
            }
        }
    }
    let mut candidates = Vec::new();
    for (index, object) in catalog.objects.iter().enumerate() {
        let candidate = match family_of[index] {
            None => single(object),
            Some(f) if families[f][0] == index => {
                family_candidate(families[f].iter().map(|&i| &catalog.objects[i]))
            }
            Some(_) => continue,
        };
        if keep(candidate.area, candidate.flux) {
            candidates.push(candidate);
        }
    }
    candidates
}

/// One candidate for a family of deblended children (§2.2).
fn family_candidate<'a>(members: impl Iterator<Item = &'a CatalogSource>) -> StarCandidate {
    let mut flux = 0.0f64;
    let (mut wsum, mut wx, mut wy) = (0.0f64, 0.0f64, 0.0f64);
    let (mut ux, mut uy) = (0.0f64, 0.0f64);
    let mut n = 0u32;
    let mut area = 0usize;
    let mut peak = f32::NEG_INFINITY;
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (usize::MAX, 0, usize::MAX, 0);
    for child in members {
        n += 1;
        flux += child.flux as f64;
        let weight = if child.flux > 0.0 {
            child.flux as f64
        } else {
            0.0
        };
        wsum += weight;
        wx += weight * child.x as f64;
        wy += weight * child.y as f64;
        ux += child.x as f64;
        uy += child.y as f64;
        area += child.npix as usize;
        peak = peak.max(child.peak_raw);
        min_x = min_x.min(child.xmin);
        max_x = max_x.max(child.xmax);
        min_y = min_y.min(child.ymin);
        max_y = max_y.max(child.ymax);
    }
    let (x, y) = if wsum > 0.0 {
        (wx / wsum, wy / wsum)
    } else {
        (ux / n as f64, uy / n as f64)
    };
    StarCandidate {
        x: x as f32,
        y: y as f32,
        area,
        flux: flux as f32,
        peak,
        min_x,
        max_x,
        min_y,
        max_y,
        deblended: true,
        family_size: n,
    }
}

/// The sky the star gates subtract inside one measurement crop.
enum GateSky {
    /// Legacy: the crop's median and 1.4826*MAD (with the legacy 1e-6 noise
    /// floor), one value for the crop.
    Crop { sky: f32, noise: f32 },
    /// The parent spatial background and RMS maps evaluated over the crop.
    Map { level: Vec<f32>, rms: Vec<f32> },
}

impl GateSky {
    fn level(&self, index: usize) -> f32 {
        match self {
            GateSky::Crop { sky, .. } => *sky,
            GateSky::Map { level, .. } => level[index],
        }
    }

    fn noise(&self, index: usize) -> f32 {
        match self {
            GateSky::Crop { noise, .. } => *noise,
            GateSky::Map { rms, .. } => rms[index],
        }
    }
}

fn window(
    values: &[f32],
    frame_width: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(w * h);
    for y in y0..y0 + h {
        let row = y * frame_width + x0;
        out.extend_from_slice(&values[row..row + w]);
    }
    out
}

/// Measure HFRs for the brightest stars that pass the source-validity gates.
///
/// Candidates are examined in flux order; each must clear, in order of cost:
/// saturation (component peak), edge truncation (component bbox), then on an
/// adaptively-sized crop with a *local* sky estimate: aperture SNR, core pixel
/// count, second-moment ellipticity, and the blend test (`policy.blend_test`).
/// Only then is the curve-of-growth HFR computed. Without these gates, ranking
/// by flux preferentially selects saturated stars, whose flat cores bias HFR
/// low right where the autofocus curve needs discrimination.
fn measure_stars(
    frame: &LumaFrame,
    stars: &[StarCandidate],
    field: &BackgroundField,
    policy: FocusSourcePolicy,
) -> (Vec<UsedStar>, StarRejectionTallies, GateDiagnostics) {
    let saturation_adu = match saturation_gate(frame) {
        SaturationGate::Applied { threshold_adu } => Some(threshold_adu),
        SaturationGate::NotPerformed | SaturationGate::NativeMaskApplied => None,
    };
    let invalid_noise = match field {
        BackgroundField::Map {
            invalid_noise: Some(invalid),
            ..
        } => Some(invalid.as_slice()),
        _ => None,
    };
    let mut used = Vec::new();
    let mut tallies = StarRejectionTallies::default();
    let mut diagnostics = GateDiagnostics::default();

    for star in stars.iter().take(MAX_STARS_EXAMINED) {
        if used.len() >= MAX_STARS_USED {
            break;
        }

        let mut rejected = false;

        // Legacy independent mono callers retain their peak gate. Versioned
        // analysis below tests native support across the measurement crop.
        //
        // Skipped when the sensor's saturation point is unknown. It is not
        // replaced by a container-derived threshold: on a 14-bit body that
        // threshold sits above anything the ADC can produce, so it would pass
        // every star while looking like a gate that ran. `QualityMetrics::
        // saturation_gate` carries the distinction to consumers.
        if frame.analysis_descriptor.is_none() {
            if let Some(threshold) = saturation_adu {
                if star.peak >= threshold {
                    tallies.saturated += 1;
                    rejected = true;
                }
            }
        }

        // Gate 2 — edge truncation: above-threshold pixels reach the border.
        if star.min_x <= EDGE_MARGIN_PX
            || star.min_y <= EDGE_MARGIN_PX
            || star.max_x + EDGE_MARGIN_PX >= frame.width.saturating_sub(1)
            || star.max_y + EDGE_MARGIN_PX >= frame.height.saturating_sub(1)
        {
            tallies.edge_truncated += 1;
            rejected = true;
        }

        // Gate 2b — deblending policy. The detector splitting a blend is not
        // the same decision as accepting its children for a focus width. A
        // regrouped family is judged by the blend gate instead.
        let family = star.family_size >= 2;
        if star.deblended && !policy.accept_deblended && !family {
            tallies.deblended += 1;
            rejected = true;
        }

        if rejected {
            continue;
        }

        // Adaptive crop: 2x the component bounding box, clamped. A defocused
        // star's >=5-sigma footprint scales with its width, so the crop (and
        // therefore the measurable HFR ceiling of min(crop)/2 - 1) follows it.
        let bbox_extent = (star.max_x - star.min_x + 1).max(star.max_y - star.min_y + 1);
        let crop_size = (2 * bbox_extent).clamp(CROP_MIN_PX, CROP_MAX_PX);
        let (crop, cw, ch, start_x, start_y) = extract_crop(frame, star.x, star.y, crop_size);
        // Never estimate a width using an interpolated replacement for missing
        // detector evidence. Reject the complete measurement footprint.
        let indices = (start_y..start_y + ch)
            .flat_map(|y| (start_x..start_x + cw).map(move |x| y * frame.width + x));
        let mut masked = false;
        let mut saturated_support = false;
        let mut invalid_support = false;
        for index in indices {
            masked |= !frame.measurement_pixel_valid(index);
            saturated_support |= frame.native_saturation_at(index);
            invalid_support |= invalid_noise.is_some_and(|invalid| invalid[index] != 0);
        }
        if masked {
            tallies.masked_support += 1;
        }
        if saturated_support {
            tallies.saturated += 1;
        }
        // masked_local_v2: no valid noise supports gates or HFR there.
        if invalid_support {
            tallies.invalid_noise_support += 1;
        }
        if masked || saturated_support || invalid_support {
            continue;
        }

        // Sky for the gates. Legacy: the crop median + MAD. Spatial: the
        // parent maps over exactly this crop, so gates and HFR share the
        // detection's background (BGE-13).
        let sky = match field {
            BackgroundField::Uniform { .. } => {
                let (sky, rms) = local_sky_stats(&crop);
                GateSky::Crop {
                    sky,
                    noise: rms.max(1e-6),
                }
            }
            BackgroundField::Map { level, rms, .. } => GateSky::Map {
                level: window(level, frame.width, start_x, start_y, cw, ch),
                rms: window(rms, frame.width, start_x, start_y, cw, ch),
            },
        };
        let cx = (star.x - start_x as f32).clamp(0.0, (cw - 1) as f32);
        let cy = (star.y - start_y as f32).clamp(0.0, (ch - 1) as f32);

        // Gate 3 — aperture SNR + core pixel count. The sky-subtracted sum is
        // UNCLIPPED so zero-mean noise cancels; clipping first would rectify
        // noise and inflate the SNR of a blank patch. The noise is the
        // background scale only (no source Poisson term: gain is unknown).
        let r_ap = (cw.min(ch) as f32 / 8.0).max(4.0);
        let mut ap_signal = 0.0_f32;
        let mut ap_variance = 0.0_f32;
        let mut n_ap = 0_usize;
        let mut core_pixels = 0_usize;
        for y in 0..ch {
            for x in 0..cw {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if dx * dx + dy * dy <= r_ap * r_ap {
                    let index = y * cw + x;
                    let noise = sky.noise(index);
                    let resid = crop[index] - sky.level(index);
                    ap_signal += resid;
                    ap_variance += noise * noise;
                    n_ap += 1;
                    if resid > 3.0 * noise {
                        core_pixels += 1;
                    }
                }
            }
        }
        let snr = match &sky {
            GateSky::Crop { noise, .. } => ap_signal / (noise * (n_ap.max(1) as f32).sqrt()),
            GateSky::Map { .. } => ap_signal / ap_variance.max(f32::MIN_POSITIVE).sqrt(),
        };
        if snr < MIN_APERTURE_SNR {
            tallies.low_snr += 1;
            rejected = true;
        }
        if core_pixels < MIN_CORE_PIXELS {
            tallies.too_few_pixels += 1;
            rejected = true;
        }

        // Gate 4 — second-moment ellipticity (trailed/astigmatic source).
        let ellipticity = second_moment_ellipticity(&crop, cw, ch, |index| sky.level(index));
        if ellipticity > MAX_ELLIPTICITY {
            tallies.elongated += 1;
            rejected = true;
        }

        // Gate 5 — blend test. Runs unconditionally (like the Python
        // labeler) so every triggered reason is tallied, not just the first.
        // peak_count_v1: >=2 distinct core peaks (local maxima of the residual
        // above 5 noise; with a uniform sky that is the raw crop).
        // dip_annulus_v2: peaks separated by a significant saddle, unless the
        // source is one annulus.
        let (blend_flagged, annulus_exempt) = match policy.blend_test {
            BlendTest::PeakCountV1 => {
                let peaks = match &sky {
                    GateSky::Crop { sky, noise } => {
                        let threshold = sky + 5.0 * noise;
                        count_core_peaks(&crop, cw, ch, |_| threshold)
                    }
                    GateSky::Map { level, rms } => {
                        let residual: Vec<f32> =
                            crop.iter().zip(level).map(|(v, b)| v - b).collect();
                        count_core_peaks(&residual, cw, ch, |index| 5.0 * rms[index])
                    }
                };
                (peaks >= 2, false)
            }
            BlendTest::DipAnnulusV2 => {
                let verdict = match &sky {
                    GateSky::Crop { sky, noise } => {
                        let residual: Vec<f32> = crop.iter().map(|v| v - sky).collect();
                        dip_annulus_verdict(&residual, cw, ch, |_| *noise)
                    }
                    GateSky::Map { level, rms } => {
                        let residual: Vec<f32> =
                            crop.iter().zip(level).map(|(v, b)| v - b).collect();
                        dip_annulus_verdict(&residual, cw, ch, |index| rms[index])
                    }
                };
                (
                    verdict == BlendVerdict::Blended,
                    verdict == BlendVerdict::AnnulusExempt,
                )
            }
        };
        if blend_flagged {
            // A family the gate still sees as several sources confirms the
            // detector's split (§2.3).
            if family {
                tallies.deblended += 1;
            } else {
                tallies.blended += 1;
            }
            rejected = true;
        }

        if rejected {
            continue;
        }

        // HFR integrates against the explicitly chosen sky: the legacy crop
        // median (named, not silent) or the parent maps over the crop.
        let hfr = match &sky {
            GateSky::Crop { .. } => {
                let (level, sigma) = ClassicalFocusSolver::legacy_crop_sky(&crop);
                ClassicalFocusSolver::compute_hfr_with_sky(
                    &crop,
                    cw,
                    ch,
                    &CropSky::Uniform { level, sigma },
                )
            }
            GateSky::Map { level, rms } => ClassicalFocusSolver::compute_hfr_with_sky(
                &crop,
                cw,
                ch,
                &CropSky::PerPixel { level, sigma: rms },
            ),
        };
        if hfr.is_finite() && hfr > 0.0 {
            used.push(UsedStar {
                x: star.x,
                y: star.y,
                hfr_px: hfr as f32,
            });
            if annulus_exempt {
                diagnostics.annulus_exempt += 1;
            }
            if family {
                diagnostics.regrouped_accepted += 1;
            }
        } else {
            tallies.invalid_hfr += 1;
        }
    }

    (used, tallies, diagnostics)
}

/// Resolve the star-saturation gate for a frame.
///
/// Returns [`SaturationGate::NotPerformed`] rather than an unreachable
/// threshold when the sensor's saturation point is unknown, so "no stars were
/// saturated" and "we never checked" stay distinguishable all the way out to
/// the dataset. Mirrors `background::saturation_threshold`.
fn saturation_gate(frame: &LumaFrame) -> SaturationGate {
    match frame.saturation_adu {
        Some(sat) if sat.is_finite() && sat > 0.0 => SaturationGate::Applied {
            threshold_adu: if frame.analysis_descriptor.is_some() {
                sat
            } else {
                sat * SATURATION_FRACTION_OF_MAX
            },
        },
        _ if frame
            .analysis_descriptor
            .as_ref()
            .and_then(|descriptor| descriptor.mask_states.get("native_saturation"))
            .map(String::as_str)
            == Some("provided") =>
        {
            SaturationGate::NativeMaskApplied
        }
        // A caller-attached native clipping mask on a frame without a
        // descriptor is applied by the footprint veto too.
        _ if frame.analysis_descriptor.is_none()
            && !frame.masks.native_saturation.is_empty()
            && frame.masks.native_saturation.len() == frame.pixels.len() =>
        {
            SaturationGate::NativeMaskApplied
        }
        _ => SaturationGate::NotPerformed,
    }
}

/// Median sky and MAD-based robust sigma (x1.4826) of a crop.
fn local_sky_stats(crop: &[f32]) -> (f32, f32) {
    let mut finite: Vec<f32> = crop.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return (0.0, 0.0);
    }
    finite.sort_by(|a, b| a.total_cmp(b));
    let sky = median_sorted(&finite);
    let mut deviations: Vec<f32> = finite.iter().map(|v| (v - sky).abs()).collect();
    deviations.sort_by(|a, b| a.total_cmp(b));
    let rms = median_sorted(&deviations) * 1.4826;
    (sky, rms)
}

/// Ellipticity `1 - b/a` from flux-weighted second central moments of the
/// sky-subtracted (clipped) signal — the SExtractor A/B formalism, identical to
/// `psf_measure.second_moment_shape`. Clipping is fine here: this locates and
/// shapes the star; only flux *integration* must stay unclipped.
fn second_moment_ellipticity(
    crop: &[f32],
    width: usize,
    height: usize,
    sky: impl Fn(usize) -> f32,
) -> f32 {
    let mut total = 0.0_f32;
    let mut cx = 0.0_f32;
    let mut cy = 0.0_f32;
    for y in 0..height {
        for x in 0..width {
            let f = (crop[y * width + x] - sky(y * width + x)).max(0.0);
            total += f;
            cx += x as f32 * f;
            cy += y as f32 * f;
        }
    }
    if total <= f32::EPSILON {
        return 0.0;
    }
    cx /= total;
    cy /= total;

    let mut mxx = 0.0_f32;
    let mut myy = 0.0_f32;
    let mut mxy = 0.0_f32;
    for y in 0..height {
        for x in 0..width {
            let f = (crop[y * width + x] - sky(y * width + x)).max(0.0);
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            mxx += f * dx * dx;
            myy += f * dy * dy;
            mxy += f * dx * dy;
        }
    }
    mxx /= total;
    myy /= total;
    mxy /= total;

    let common = (((mxx - myy) * 0.5).powi(2) + mxy * mxy).sqrt();
    let a2 = ((mxx + myy) * 0.5 + common).max(0.0);
    let b2 = ((mxx + myy) * 0.5 - common).max(0.0);
    let a = a2.sqrt();
    if a <= f32::EPSILON {
        return 0.0;
    }
    (1.0 - b2.sqrt() / a).clamp(0.0, 1.0)
}

/// Count distinct local maxima above a per-pixel threshold (legacy: `sky +
/// 5*noise` on the raw crop; spatial: `5*R` on the residual) — a cheap
/// rejection heuristic, not a deblender (port of
/// `psf_measure._count_core_peaks`). A pixel is a peak when it equals the
/// maximum of its `(2*BLEND_MIN_SEPARATION+1)^2` window; touching plateau
/// peaks collapse into one via connected components.
fn count_core_peaks(
    crop: &[f32],
    width: usize,
    height: usize,
    threshold: impl Fn(usize) -> f32,
) -> usize {
    let win = BLEND_MIN_SEPARATION;
    let mut is_peak = vec![false; crop.len()];

    for y in 0..height {
        for x in 0..width {
            let value = crop[y * width + x];
            if !value.is_finite() || value <= threshold(y * width + x) {
                continue;
            }
            let mut window_max = f32::NEG_INFINITY;
            for ny in y.saturating_sub(win)..=(y + win).min(height - 1) {
                for nx in x.saturating_sub(win)..=(x + win).min(width - 1) {
                    window_max = window_max.max(crop[ny * width + nx]);
                }
            }
            if value >= window_max {
                is_peak[y * width + x] = true;
            }
        }
    }

    // Collapse touching plateau peaks (8-connectivity) into single detections.
    let mut visited = vec![false; crop.len()];
    let mut peaks = 0_usize;
    for idx in 0..crop.len() {
        if !is_peak[idx] || visited[idx] {
            continue;
        }
        peaks += 1;
        let mut stack = vec![idx];
        visited[idx] = true;
        while let Some(current) = stack.pop() {
            let x = current % width;
            let y = current / width;
            for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    let neighbor = ny * width + nx;
                    if is_peak[neighbor] && !visited[neighbor] {
                        visited[neighbor] = true;
                        stack.push(neighbor);
                    }
                }
            }
        }
    }
    peaks
}

/// Outcome of the `dip_annulus_v2` blend test on one crop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlendVerdict {
    /// At most one significant component.
    NotBlended,
    /// Blend-flagged by the saddle test, accepted as one annulus (§1.4).
    AnnulusExempt,
    Blended,
}

/// A plateau of 8-connected peak pixels (§1.2).
#[derive(Debug, Clone, Copy)]
struct CorePeak {
    value: f32,
    /// Lowest linear index holding `value`.
    rep: usize,
}

impl CorePeak {
    /// Higher value, then lower representative index.
    fn brighter_than(&self, other: &CorePeak) -> bool {
        self.value > other.value || (self.value == other.value && self.rep < other.rep)
    }
}

/// Neighbour visiting order of the merge (dx, dy), fixed by §1.3.
const MERGE_NEIGHBOURS: [(isize, isize); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

fn uf_root(parent: &mut [u32], mut i: usize) -> usize {
    while parent[i] as usize != i {
        let up = parent[i] as usize;
        parent[i] = parent[up];
        i = up;
    }
    i
}

/// `dip_annulus_v2` (`docs/bge-v2-spec.md` §1) on a crop residual `res`
/// (row-major, `width x height`) with noise `sig`. The threshold is
/// `t = 5 * sig` (float32, as v1); dips, centroid, distances and means are
/// double precision.
///
/// Peaks are v1's (window maxima above `t`, plateaus collapsed). Pixels above
/// `t` are merged in descending order (ties by index); when two components
/// holding peaks meet at level `v`, the fainter peak resolves with dip
/// `p - v`. Peaks never joined to the brightest resolve with `p - t_rep`.
/// A resolved peak is significant when its dip exceeds
/// `max(DIP_FRACTION * p, DIP_MIN_SIGMA * sig_p)`; one or more significant
/// peaks blend-flag the star, which the annulus exemption may then accept.
fn dip_annulus_verdict(
    res: &[f32],
    width: usize,
    height: usize,
    sig: impl Fn(usize) -> f32,
) -> BlendVerdict {
    let n = res.len();
    let threshold = |i: usize| 5.0 * sig(i);
    let above = |i: usize| res[i].is_finite() && res[i] > threshold(i);

    // §1.2 peaks, identical to v1 on the residual.
    let win = BLEND_MIN_SEPARATION;
    let mut is_peak = vec![false; n];
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            if !above(i) {
                continue;
            }
            let mut window_max = f32::NEG_INFINITY;
            for ny in y.saturating_sub(win)..=(y + win).min(height - 1) {
                for nx in x.saturating_sub(win)..=(x + win).min(width - 1) {
                    window_max = window_max.max(res[ny * width + nx]);
                }
            }
            if res[i] >= window_max {
                is_peak[i] = true;
            }
        }
    }
    const NONE: u32 = u32::MAX;
    let mut peak_at = vec![NONE; n];
    let mut peaks: Vec<CorePeak> = Vec::new();
    let mut visited = vec![false; n];
    let mut stack = Vec::new();
    for start in 0..n {
        if !is_peak[start] || visited[start] {
            continue;
        }
        visited[start] = true;
        stack.push(start);
        let mut peak = CorePeak {
            value: res[start],
            rep: start,
        };
        while let Some(current) = stack.pop() {
            if res[current] > peak.value || (res[current] == peak.value && current < peak.rep) {
                peak = CorePeak {
                    value: res[current],
                    rep: current,
                };
            }
            let (x, y) = (current % width, current / width);
            for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    let neighbour = ny * width + nx;
                    if is_peak[neighbour] && !visited[neighbour] {
                        visited[neighbour] = true;
                        stack.push(neighbour);
                    }
                }
            }
        }
        peak_at[peak.rep] = peaks.len() as u32;
        peaks.push(peak);
    }
    if peaks.len() <= 1 {
        return BlendVerdict::NotBlended;
    }

    // §1.3 bottleneck merge.
    let mut order: Vec<usize> = (0..n).filter(|&i| above(i)).collect();
    order.sort_by(|&a, &b| {
        res[b]
            .partial_cmp(&res[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut parent = vec![NONE; n];
    let mut best = vec![NONE; n];
    let mut dip: Vec<Option<f64>> = vec![None; peaks.len()];
    for &q in &order {
        parent[q] = q as u32;
        best[q] = peak_at[q];
        let level = res[q] as f64;
        let (x, y) = (q % width, q / width);
        for (dx, dy) in MERGE_NEIGHBOURS {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                continue;
            }
            let neighbour = ny as usize * width + nx as usize;
            if parent[neighbour] == NONE {
                continue;
            }
            let (rq, rn) = (uf_root(&mut parent, q), uf_root(&mut parent, neighbour));
            if rq == rn {
                continue;
            }
            let merged = match (best[rq], best[rn]) {
                (NONE, other) | (other, NONE) => other,
                (a, b) => {
                    let (bright, faint) = if peaks[a as usize].brighter_than(&peaks[b as usize]) {
                        (a, b)
                    } else {
                        (b, a)
                    };
                    dip[faint as usize] = Some(peaks[faint as usize].value as f64 - level);
                    bright
                }
            };
            parent[rn] = rq as u32;
            best[rq] = merged;
        }
    }
    let brightest = (0..peaks.len()).fold(0, |b, k| {
        if peaks[k].brighter_than(&peaks[b]) {
            k
        } else {
            b
        }
    });
    // Disjoint components: dip down to the threshold at the peak.
    for (k, peak) in peaks.iter().enumerate() {
        if k != brightest && dip[k].is_none() {
            dip[k] = Some(peak.value as f64 - threshold(peak.rep) as f64);
        }
    }
    let significant: Vec<usize> = (0..peaks.len())
        .filter(|&k| k != brightest)
        .filter(|&k| {
            let p = peaks[k].value as f64;
            let sig_p = sig(peaks[k].rep) as f64;
            dip[k].is_some_and(|d| d > (DIP_FRACTION * p).max(DIP_MIN_SIGMA * sig_p))
        })
        .collect();
    if significant.is_empty() {
        return BlendVerdict::NotBlended;
    }

    // §1.4 annulus exemption.
    let members: Vec<usize> = std::iter::once(brightest).chain(significant).collect();
    let roots: Vec<usize> = members
        .iter()
        .map(|&k| uf_root(&mut parent, peaks[k].rep))
        .collect();
    let (mut sw, mut sx, mut sy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        if parent[i] == NONE || !roots.contains(&uf_root(&mut parent, i)) {
            continue;
        }
        let w = res[i] as f64;
        sw += w;
        sx += w * (i % width) as f64;
        sy += w * (i / width) as f64;
    }
    let (cx, cy) = (sx / sw, sy / sw);
    let distance = |i: usize| {
        let (dx, dy) = ((i % width) as f64 - cx, (i / width) as f64 - cy);
        (dx * dx + dy * dy).sqrt()
    };
    let mut radii: Vec<f64> = members.iter().map(|&k| distance(peaks[k].rep)).collect();
    radii.sort_by(f64::total_cmp);
    let m = radii.len();
    let r = if m % 2 == 1 {
        radii[m / 2]
    } else {
        0.5 * (radii[m / 2 - 1] + radii[m / 2])
    };
    if !(r >= ANN_MIN_RADIUS_PX) {
        return BlendVerdict::Blended;
    }
    if radii
        .iter()
        .any(|d| !((d - r).abs() <= ANN_RADIUS_TOL * r + 1.0))
    {
        return BlendVerdict::Blended;
    }
    let half = (ANN_HALF_WIDTH * r).max(1.5);
    let hole_radius = (ANN_HOLE_RADIUS * r).max(1.0);
    let mut bin_sum = [0.0f64; ANN_BINS];
    let mut bin_count = [0usize; ANN_BINS];
    let (mut hole_sum, mut hole_count) = (0.0f64, 0usize);
    for (i, &value) in res.iter().enumerate() {
        let (dx, dy) = ((i % width) as f64 - cx, (i / width) as f64 - cy);
        let d = (dx * dx + dy * dy).sqrt();
        if (d - r).abs() <= half {
            let angle = dy.atan2(dx);
            let bin = ((angle + std::f64::consts::PI) / (2.0 * std::f64::consts::PI)
                * ANN_BINS as f64)
                .floor()
                .clamp(0.0, (ANN_BINS - 1) as f64) as usize;
            bin_sum[bin] += value as f64;
            bin_count[bin] += 1;
        }
        if d <= hole_radius {
            hole_sum += value as f64;
            hole_count += 1;
        }
    }
    let mut means = [f64::NEG_INFINITY; ANN_BINS];
    for k in 0..ANN_BINS {
        if bin_count[k] > 0 {
            means[k] = bin_sum[k] / bin_count[k] as f64;
        }
    }
    let mut sorted = means;
    sorted.sort_by(|a, b| b.total_cmp(a));
    let top = sorted[..4].iter().sum::<f64>() / 4.0;
    let covered = means
        .iter()
        .filter(|&&mean| mean >= ANN_BIN_FRACTION * top)
        .count();
    if !(covered as f64 / ANN_BINS as f64 >= ANN_MIN_COVERAGE) {
        return BlendVerdict::Blended;
    }
    let hole_mean = hole_sum / hole_count as f64;
    if !(hole_mean <= ANN_HOLE_FRACTION * top) {
        return BlendVerdict::Blended;
    }
    BlendVerdict::AnnulusExempt
}

fn extract_crop(
    frame: &LumaFrame,
    center_x: f32,
    center_y: f32,
    crop_size: usize,
) -> (Vec<f32>, usize, usize, usize, usize) {
    let crop_width = crop_size.min(frame.width);
    let crop_height = crop_size.min(frame.height);
    let center_x = center_x
        .round()
        .clamp(0.0, frame.width.saturating_sub(1) as f32) as usize;
    let center_y = center_y
        .round()
        .clamp(0.0, frame.height.saturating_sub(1) as f32) as usize;
    let start_x = center_x
        .saturating_sub(crop_width / 2)
        .min(frame.width - crop_width);
    let start_y = center_y
        .saturating_sub(crop_height / 2)
        .min(frame.height - crop_height);
    let mut crop = Vec::with_capacity(crop_width * crop_height);

    for y in start_y..start_y + crop_height {
        let row = y * frame.width + start_x;
        crop.extend_from_slice(&frame.pixels[row..row + crop_width]);
    }

    (crop, crop_width, crop_height, start_x, start_y)
}

#[cfg(test)]
mod tests {
    use super::{background_from_luma, estimate_quality, local_sky_stats, LumaFrame};

    #[test]
    fn local_sky_matches_numpy_median_for_even_samples() {
        let (sky, rms) = local_sky_stats(&[1.0, 2.0, 3.0, 4.0]);
        assert!((sky - 2.5).abs() < 1e-6);
        assert!((rms - 1.4826).abs() < 1e-4);
    }

    #[test]
    fn quality_finds_synthetic_star() {
        let width = 64;
        let height = 64;
        let mut pixels = vec![100.0_f32; width * height];
        let cx = 32_i32;
        let cy = 32_i32;
        for y in -3..=3 {
            for x in -3..=3 {
                let distance = ((x * x + y * y) as f32).sqrt();
                let value = 500.0 - distance * 80.0;
                let idx = ((cy + y) as usize) * width + (cx + x) as usize;
                pixels[idx] = pixels[idx].max(value);
            }
        }
        let frame = LumaFrame {
            pixels,
            width,
            height,
            max_adu: 65535.0,
            saturation_adu: Some(65535.0),
            masks: Default::default(),
            analysis_descriptor: None,
        };
        let background = background_from_luma(&frame);
        let quality = estimate_quality(&frame, &background);

        assert!(quality.observation_valid());
        assert_eq!(quality.status, super::QualityStatus::Valid);
        assert_eq!(quality.star_count_detected, 1);
        assert_eq!(quality.star_count_used, 1);
        assert!(quality.hfr_px.unwrap() > 0.0);
        // HFD is exactly twice HFR; a fitted FWHM is not available classically.
        assert!((quality.hfd_px.unwrap() - 2.0 * quality.hfr_px.unwrap()).abs() < 1e-4);
        assert!(quality.fwhm_px.is_none());
        // A clean star trips no validity gate.
        assert_eq!(
            quality.star_rejections,
            super::StarRejectionTallies::default()
        );
    }

    fn add_gaussian(
        pixels: &mut [f32],
        width: usize,
        cx: f32,
        cy: f32,
        amp: f32,
        sigma: f32,
        extent: i32,
    ) {
        let height = pixels.len() / width;
        for dy in -extent..=extent {
            for dx in -extent..=extent {
                let x = cx as i32 + dx;
                let y = cy as i32 + dy;
                if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
                    continue;
                }
                let fx = x as f32 - cx;
                let fy = y as f32 - cy;
                let value = amp * (-(fx * fx + fy * fy) / (2.0 * sigma * sigma)).exp();
                let idx = y as usize * width + x as usize;
                pixels[idx] += value;
            }
        }
    }

    #[test]
    fn saturated_star_is_rejected() {
        let width = 64;
        let height = 64;
        let mut pixels = vec![100.0_f32; width * height];
        // Peak at 99.2% of full scale. The old accidental max(max*0.98,
        // max-1) threshold missed this nonlinear/clipped core.
        for dy in -3_i32..=3 {
            for dx in -3_i32..=3 {
                let d = ((dx * dx + dy * dy) as f32).sqrt();
                let idx = ((32 + dy) as usize) * width + (32 + dx) as usize;
                pixels[idx] = (65_000.0 - d * 3_000.0).max(pixels[idx]);
            }
        }
        let frame = LumaFrame {
            pixels,
            width,
            height,
            max_adu: 65_535.0,
            saturation_adu: Some(65_535.0),
            masks: Default::default(),
            analysis_descriptor: None,
        };
        let background = background_from_luma(&frame);
        let quality = estimate_quality(&frame, &background);

        assert_eq!(quality.star_count_detected, 1);
        assert_eq!(quality.star_count_used, 0);
        assert_eq!(quality.status, super::QualityStatus::NoUsableStars);
        assert!(
            quality.hfr_px.is_none(),
            "saturated star must not yield a width"
        );
        assert_eq!(quality.star_rejections.saturated, 1);
        assert!(quality.saturation_gate.was_applied());
    }

    /// A 14-bit body clipped at 16383 inside a 16-bit container.
    ///
    /// This is the frame the old code got wrong: the gate was computed from the
    /// container (0.98 * 65535 = 64224), the star peaks at 16383, so it sailed
    /// through as a clean measurement and its flat core biased HFR low. The two
    /// halves below feed byte-identical pixels and differ only in whether the
    /// sensor's saturation point is known.
    fn clipped_14bit_star_frame(saturation_adu: Option<f32>) -> LumaFrame {
        let width = 64;
        let height = 64;
        let mut pixels = vec![100.0_f32; width * height];
        for dy in -4_i32..=4 {
            for dx in -4_i32..=4 {
                let d = ((dx * dx + dy * dy) as f32).sqrt();
                let idx = ((32 + dy) as usize) * width + (32 + dx) as usize;
                // Flat, clipped core at the 14-bit ceiling, falling off outside.
                pixels[idx] = (16_383.0 - (d - 1.5).max(0.0) * 3_000.0).max(pixels[idx]);
            }
        }
        LumaFrame {
            pixels,
            width,
            height,
            max_adu: 65_535.0,
            saturation_adu,
            masks: Default::default(),
            analysis_descriptor: None,
        }
    }

    #[test]
    fn clipped_14bit_star_is_rejected_when_saturation_is_declared() {
        let frame = clipped_14bit_star_frame(Some(16_383.0));
        let background = background_from_luma(&frame);
        let quality = estimate_quality(&frame, &background);

        assert!(quality.saturation_gate.was_applied());
        assert_eq!(
            quality.star_rejections.saturated, 1,
            "a core clipped at the 14-bit ceiling must be caught once the ceiling is known"
        );
    }

    #[test]
    fn unknown_saturation_reports_not_performed_rather_than_passing() {
        let frame = clipped_14bit_star_frame(None);
        let background = background_from_luma(&frame);
        let quality = estimate_quality(&frame, &background);

        // The star is NOT rejected — we cannot tell it is clipped. That is the
        // honest outcome, and the point of the change is that it is now
        // *labelled* as such instead of being indistinguishable from a clean
        // frame.
        assert_eq!(quality.star_rejections.saturated, 0);
        assert!(
            !quality.saturation_gate.was_applied(),
            "gate must report NotPerformed so a dataset builder can exclude this frame"
        );
        assert_eq!(quality.saturation_gate, super::SaturationGate::NotPerformed);
        // This frame holds a clipped 14-bit core, so a saturation fraction of
        // 0.0 would be false; with no saturation evidence it is absent. The
        // background still excludes no pixel for an unknown saturation point.
        assert_eq!(background.result.saturation_fraction, None);
        assert_eq!(background.result.support.saturation_evidence, "unknown");
        assert_eq!(
            background.result.support.usable_pixels, background.result.support.total_pixels,
            "unknown saturation must exclude no pixels from the background"
        );
    }

    #[test]
    fn edge_truncated_star_is_rejected() {
        let width = 64;
        let height = 64;
        let mut pixels = vec![100.0_f32; width * height];
        add_gaussian(&mut pixels, width, 2.0, 32.0, 800.0, 1.5, 4);
        let frame = LumaFrame {
            pixels,
            width,
            height,
            max_adu: 65_535.0,
            saturation_adu: Some(65_535.0),
            masks: Default::default(),
            analysis_descriptor: None,
        };
        let background = background_from_luma(&frame);
        let quality = estimate_quality(&frame, &background);

        assert_eq!(quality.star_count_used, 0);
        assert!(quality.star_rejections.edge_truncated >= 1);
        assert!(quality.hfr_px.is_none());
    }

    #[test]
    fn blended_double_is_rejected() {
        let width = 64;
        let height = 64;
        let mut pixels = vec![100.0_f32; width * height];
        // Two equal stars 8 px apart: merged into one component by the detection
        // threshold, but two distinct core peaks (and an elongated joint shape).
        add_gaussian(&mut pixels, width, 28.0, 32.0, 800.0, 1.5, 5);
        add_gaussian(&mut pixels, width, 36.0, 32.0, 800.0, 1.5, 5);
        let frame = LumaFrame {
            pixels,
            width,
            height,
            max_adu: 65_535.0,
            saturation_adu: Some(65_535.0),
            masks: Default::default(),
            analysis_descriptor: None,
        };
        let background = background_from_luma(&frame);
        let quality = estimate_quality(&frame, &background);

        assert_eq!(quality.star_count_used, 0);
        assert!(
            quality.star_rejections.blended >= 1,
            "double must trip the blend gate"
        );
        assert!(quality.hfr_px.is_none());
    }

    #[test]
    fn elongated_streak_is_rejected_for_focus() {
        let width = 64;
        let height = 64;
        let mut pixels = vec![100.0_f32; width * height];
        for x in 24..40 {
            pixels[32 * width + x] = 600.0;
        }
        for x in 27..37 {
            pixels[31 * width + x] = 350.0;
            pixels[33 * width + x] = 350.0;
        }
        let frame = LumaFrame {
            pixels,
            width,
            height,
            max_adu: 65_535.0,
            saturation_adu: Some(65_535.0),
            masks: Default::default(),
            analysis_descriptor: None,
        };
        let background = background_from_luma(&frame);
        let quality = estimate_quality(&frame, &background);

        // Trailed sources are measured by vision::trailing; they must not
        // contribute a focus width.
        assert_eq!(quality.star_count_used, 0);
        assert!(quality.star_rejections.elongated >= 1);
        assert!(quality.hfr_px.is_none());
    }

    #[test]
    fn defocused_wide_star_measures_true_hfr_via_adaptive_crop() {
        // HFR 12 px star (sigma ~10.2). The old fixed 32 px crop capped the
        // measurable HFR at ~15 px and its bounded aperture at 15 px, biasing
        // this star to ~9 px; the bbox-scaled crop must recover ~12 px.
        let width = 192;
        let height = 192;
        let truth_hfr = 12.0_f32;
        let sigma = truth_hfr / (2.0_f32 * std::f32::consts::LN_2).sqrt();
        let mut pixels = vec![0.0_f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - 96.0;
                let dy = y as f32 - 96.0;
                pixels[y * width + x] =
                    100.0 + 3_000.0 * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
            }
        }
        let frame = LumaFrame {
            pixels,
            width,
            height,
            max_adu: 65_535.0,
            saturation_adu: Some(65_535.0),
            masks: Default::default(),
            analysis_descriptor: None,
        };
        let background = background_from_luma(&frame);
        let quality = estimate_quality(&frame, &background);

        assert!(quality.observation_valid());
        let hfr = quality.hfr_px.unwrap();
        assert!(
            (hfr - truth_hfr).abs() < 0.8,
            "adaptive crop should recover ~{truth_hfr} px, got {hfr}"
        );
    }

    #[test]
    fn blank_frame_reports_null_widths_not_zero() {
        let frame = LumaFrame {
            pixels: vec![100.0_f32; 64 * 64],
            width: 64,
            height: 64,
            max_adu: 65535.0,
            saturation_adu: Some(65535.0),
            masks: Default::default(),
            analysis_descriptor: None,
        };
        let background = background_from_luma(&frame);
        let quality = estimate_quality(&frame, &background);

        assert!(!quality.observation_valid());
        assert_eq!(quality.status, super::QualityStatus::NoSources);
        assert!(quality.hfr_px.is_none());
        assert!(quality.hfd_px.is_none());
        assert!(quality.quality_uncertainty.is_none());
        assert!(quality.hfr_log_sigma.is_none());
    }

    #[test]
    fn multiple_stars_report_real_between_star_sigma() {
        // Four stars of deliberately different widths on a 128x128 frame. The
        // between-star HFR scatter must surface as a real, positive log-sigma —
        // not the arbitrary-constant heuristic.
        let width = 128;
        let height = 128;
        let mut pixels = vec![100.0_f32; width * height];
        let stars = [
            (24, 24, 3.0_f32),
            (24, 96, 5.0),
            (96, 24, 2.0),
            (96, 96, 6.0),
        ];
        for (cx, cy, spread) in stars {
            for y in -8..=8 {
                for x in -8..=8 {
                    let d2 = (x * x + y * y) as f32;
                    let value = 100.0 + 900.0 * (-d2 / (2.0 * spread * spread)).exp();
                    let idx = ((cy + y) as usize) * width + (cx + x) as usize;
                    pixels[idx] = pixels[idx].max(value);
                }
            }
        }
        let frame = LumaFrame {
            pixels,
            width,
            height,
            max_adu: 65535.0,
            saturation_adu: Some(65535.0),
            masks: Default::default(),
            analysis_descriptor: None,
        };
        let background = background_from_luma(&frame);
        let quality = estimate_quality(&frame, &background);

        assert!(quality.observation_valid());
        assert!(quality.star_count_used >= 3);
        let sigma = quality
            .hfr_log_sigma
            .expect(">=3 stars should yield a sigma");
        assert!(
            sigma > 0.0,
            "stars of differing width must produce nonzero scatter"
        );
        // Scatter remains in its physical log-HFR domain. No arbitrary [0,1]
        // transform is presented as calibrated uncertainty.
        assert!(quality.quality_uncertainty.is_none());
    }
    #[test]
    fn demosaic_dilution_cannot_hide_native_clipping_from_source_gate() {
        use crate::processing::debayer::*;
        let width = 72;
        let height = 72;
        let mut native = vec![100_u16; width * height];
        for y in 0..height {
            for x in 0..width {
                let r2 = (x as f32 - 36.0).powi(2) + (y as f32 - 36.0).powi(2);
                native[y * width + x] += (2500.0 * (-r2 / 12.0).exp()) as u16;
            }
        }
        native[36 * width + 36] = 5000;
        let data: Vec<u8> = native
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let descriptor = SensorDescriptor {
            schema_version: 1,
            width,
            height,
            row_stride_bytes: width * 2,
            sample_type: SampleType::U16,
            byte_order: ByteOrder::Little,
            color: SensorColor::Bayer {
                pattern: CfaPattern::Rggb,
                origin_x: 0,
                origin_y: 0,
            },
            domain: SampleDomain::NativeAdu,
            geometry: PixelGeometry::identity("test_pixels"),
            saturation: Some(NativeThreshold {
                adu: 5000.0,
                source: "synthetic_known_limit".into(),
            }),
            calibration_ids: vec![],
        };
        let analysis = analyze(&SensorFrame {
            descriptor: &descriptor,
            data: &data,
            masks: NativeMasks::default(),
        })
        .unwrap();
        let frame = super::luma_from_analysis(analysis);
        assert!(
            frame.pixels.iter().copied().fold(0.0, f32::max) < 5000.0,
            "interpolated proxy hides raw peak"
        );
        let quality = estimate_quality(&frame, &super::background_from_luma(&frame));
        assert!(quality.star_count_detected > 0);
        assert_eq!(quality.star_count_used, 0);
        assert!(quality.star_rejections.saturated > 0);
        assert_eq!(
            quality.saturation_gate,
            super::SaturationGate::Applied {
                threshold_adu: 5000.0
            }
        );
    }

    #[test]
    fn masks_exclude_background_contamination_and_refuse_source_measurement_support() {
        let mut frame = clipped_14bit_star_frame(None);
        // A detector defect outside the source must not affect sky statistics.
        let before = super::background_from_luma(&frame);
        frame.pixels[0] = 1e9;
        frame.masks.bad_pixel = vec![0; frame.pixels.len()];
        frame.masks.bad_pixel[0] = 1;
        let after = super::background_from_luma(&frame);
        assert_eq!(before.result.level(), after.result.level());
        frame.masks.bad_pixel[32 * frame.width + 32] = 1;
        let quality = estimate_quality(&frame, &after);
        assert_eq!(quality.star_count_used, 0);
        assert!(quality.star_rejections.masked_support > 0);
        frame.masks.invalid = vec![1];
        assert_eq!(
            estimate_quality(&frame, &after).status,
            super::QualityStatus::EmptyFrame
        );
    }

    // --- Spatial background path (synthetic maps and catalog; the SEP
    // backend itself is tested in sep_backend.rs) ---

    use crate::vision::background_model::{
        BackgroundAnalysis, BackgroundField, CatalogSource, CropSkySource, DetectionMethod,
        DetectionProvenance, NoiseStatus, SourceCatalog,
    };

    /// Deterministic zero-mean noise in [-amp, amp).
    fn noise(index: usize, amp: f32) -> f32 {
        let mut x = (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
        x ^= x >> 31;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 29;
        ((x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amp
    }

    /// A 3 px-HFR Gaussian star on a steep linear ramp, with the true maps.
    fn star_on_ramp() -> (LumaFrame, Vec<f32>, Vec<f32>) {
        let (width, height) = (160usize, 128usize);
        let level: Vec<f32> = (0..width * height)
            .map(|i| 1000.0 + 25.0 * (i % width) as f32)
            .collect();
        let mut pixels: Vec<f32> = level
            .iter()
            .enumerate()
            .map(|(i, b)| b + noise(i, 10.0))
            .collect();
        let sigma = 3.0 / (2.0 * std::f32::consts::LN_2).sqrt();
        add_gaussian(&mut pixels, width, 80.0, 64.0, 3000.0, sigma, 20);
        // Uniform noise on [-a, a) has standard deviation a/sqrt(3).
        let rms = vec![10.0 / 3.0_f32.sqrt(); width * height];
        let frame = LumaFrame {
            pixels,
            width,
            height,
            max_adu: 65535.0,
            saturation_adu: Some(65535.0),
            masks: Default::default(),
            analysis_descriptor: None,
        };
        (frame, level, rms)
    }

    fn spatial_analysis(
        frame: &LumaFrame,
        level: Vec<f32>,
        rms: Vec<f32>,
        deblended: bool,
    ) -> BackgroundAnalysis {
        let mut analysis = background_from_luma(frame);
        let (x0, x1, y0, y1) = (74usize, 86usize, 58usize, 70usize);
        let pixels: Vec<u32> = (y0..=y1)
            .flat_map(|y| (x0..=x1).map(move |x| (y * frame.width + x) as u32))
            .collect();
        let object = CatalogSource {
            x: 80.0,
            y: 64.0,
            xmin: x0,
            xmax: x1,
            ymin: y0,
            ymax: y1,
            npix: pixels.len() as u32,
            flux: 1.0e5,
            peak_residual: 3000.0,
            peak_raw: frame.pixels[64 * frame.width + 80],
            a: 2.5,
            b: 2.5,
            theta: 0.0,
            flags: if deblended { 1 } else { 0 },
            deblended,
            truncated: false,
            pixels,
        };
        analysis.catalog = Some(SourceCatalog {
            provenance: DetectionProvenance {
                method: DetectionMethod::SepExtract,
                config_sha256: "test".into(),
                crop_sky_source: CropSkySource::ParentBackgroundMap,
                noise_floor_applied: false,
                objects_detected: 1,
                objects_deblended: deblended as u32,
                objects_truncated: 0,
            },
            objects: vec![object],
            segmentation: None,
        });
        analysis.field = Some(BackgroundField::Map {
            level,
            rms,
            width: frame.width,
            height: frame.height,
            invalid_noise: None,
        });
        analysis
    }

    #[test]
    fn parent_map_drives_gates_and_hfr_on_a_gradient() {
        let (frame, level, rms) = star_on_ramp();
        let spatial = estimate_quality(&frame, &spatial_analysis(&frame, level, rms, false));
        assert_eq!(spatial.status, super::QualityStatus::Valid, "{spatial:?}");
        let hfr = spatial.hfr_px.unwrap();
        assert!((hfr - 3.0).abs() < 0.3, "spatial HFR {hfr}");
        assert_eq!(
            spatial.detection.as_ref().unwrap().crop_sky_source,
            CropSkySource::ParentBackgroundMap
        );

        // The legacy global threshold and crop median on the same pixels do
        // not recover the width (no star, or a worse one).
        let legacy = estimate_quality(&frame, &background_from_luma(&frame));
        assert!(legacy
            .hfr_px
            .map_or(true, |legacy| (legacy - 3.0).abs() > (hfr - 3.0).abs()));
    }

    #[test]
    fn deblended_children_follow_the_focus_policy_not_the_detector() {
        let (frame, level, rms) = star_on_ramp();
        let analysis = spatial_analysis(&frame, level, rms, true);
        let rejected = estimate_quality(&frame, &analysis);
        assert_eq!(rejected.status, super::QualityStatus::NoUsableStars);
        assert_eq!(rejected.star_rejections.deblended, 1);
        let accepted = super::estimate_quality_with_policy(
            &frame,
            &analysis,
            super::FocusSourcePolicy {
                accept_deblended: true,
                ..Default::default()
            },
        );
        assert_eq!(accepted.status, super::QualityStatus::Valid);
    }

    #[test]
    fn missing_background_or_detection_is_reported_not_measured() {
        let (frame, level, rms) = star_on_ramp();
        let mut no_catalog = spatial_analysis(&frame, level, rms, false);
        no_catalog.catalog = None;
        no_catalog.result.noise_status = NoiseStatus::Unresolved;
        let quality = estimate_quality(&frame, &no_catalog);
        assert_eq!(quality.status, super::QualityStatus::DetectionUnavailable);
        assert_eq!(
            quality.rejection_reasons,
            vec![super::QualityRejectionReason::NoiseUnresolved]
        );
        assert!(quality.hfr_px.is_none() && quality.detection.is_none());

        let mut no_level = no_catalog;
        no_level.field = None;
        let quality = estimate_quality(&frame, &no_level);
        assert_eq!(quality.status, super::QualityStatus::BackgroundUnavailable);
        assert!(quality.hfr_px.is_none());
    }

    #[test]
    fn background_excludes_native_saturation_and_edge_support() {
        // A block of mid-level masked pixels would move the median if used.
        let width = 64;
        let mut pixels: Vec<f32> = (0..width * width).map(|i| 100.0 + noise(i, 2.0)).collect();
        let mut native_saturation = vec![0u8; pixels.len()];
        let mut edge = vec![0u8; pixels.len()];
        for index in 0..pixels.len() / 2 + 200 {
            pixels[index] = 900.0;
            if index % 2 == 0 {
                native_saturation[index] = 1;
            } else {
                edge[index] = 1;
            }
        }
        let frame = LumaFrame {
            pixels,
            width,
            height: width,
            max_adu: 65535.0,
            saturation_adu: None,
            masks: crate::processing::debayer::AnalysisMasks {
                bad_pixel: vec![],
                invalid: vec![],
                edge,
                native_saturation,
            },
            analysis_descriptor: None,
        };
        let level = background_from_luma(&frame).result.level().unwrap();
        assert!((level - 100.0).abs() < 3.0, "level {level}");
    }

    #[test]
    fn faint_star_tallies_low_snr_and_too_few_core_pixels() {
        // Characterization of the legacy gates: a source just above the
        // detection threshold but with almost no aperture signal.
        let width = 64;
        let mut pixels: Vec<f32> = (0..width * width).map(|i| 100.0 + noise(i, 3.0)).collect();
        for (dx, dy) in [(0, 0), (1, 0), (0, 1)] {
            pixels[(32 + dy) * width + 32 + dx] = 112.0;
        }
        let frame = LumaFrame {
            pixels,
            width,
            height: width,
            max_adu: 65535.0,
            saturation_adu: Some(65535.0),
            masks: Default::default(),
            analysis_descriptor: None,
        };
        let quality = estimate_quality(&frame, &background_from_luma(&frame));
        assert!(quality.star_count_detected >= 1);
        assert_eq!(quality.star_count_used, 0);
        assert!(quality.star_rejections.low_snr >= 1);
        assert!(quality.star_rejections.too_few_pixels >= 1);
    }

    #[test]
    fn attached_native_mask_without_descriptor_is_an_applied_gate() {
        // Before: the footprint veto used the mask while the gate said
        // NotPerformed and the fraction was absent.
        let mut frame = clipped_14bit_star_frame(None);
        let mut mask = vec![0u8; frame.pixels.len()];
        for index in 0..frame.pixels.len() {
            if frame.pixels[index] >= 16_000.0 {
                mask[index] = 1;
            }
        }
        let clipped = mask.iter().filter(|m| **m == 1).count();
        assert!(clipped > 0);
        frame.masks.native_saturation = mask;
        let background = background_from_luma(&frame);
        let quality = estimate_quality(&frame, &background);
        assert_eq!(
            quality.saturation_gate,
            super::SaturationGate::NativeMaskApplied
        );
        assert!(quality.star_rejections.saturated > 0);
        assert_eq!(background.result.support.saturation_evidence, "native_mask");
        let fraction = background.result.saturation_fraction.unwrap();
        assert!((fraction - clipped as f32 / frame.pixels.len() as f32).abs() < 1e-6);
    }

    // --- BGE v2 star gates (docs/bge-v2-spec.md §1-2, §5) ---

    use super::{BlendTest, FocusSourcePolicy, GateDiagnostics, StarGateSettings};

    fn v1_policy() -> FocusSourcePolicy {
        FocusSourcePolicy {
            accept_deblended: false,
            blend_test: BlendTest::PeakCountV1,
            regroup_deblended: false,
        }
    }

    fn mono(pixels: Vec<f32>, width: usize) -> LumaFrame {
        LumaFrame {
            height: pixels.len() / width,
            pixels,
            width,
            max_adu: 65535.0,
            saturation_adu: Some(65535.0),
            masks: Default::default(),
            analysis_descriptor: None,
        }
    }

    /// A defocused donut: Gaussian ring of radius `r0` and width `w` on a
    /// 100 ADU sky with +-3 ADU uniform noise, cut by `vanes` radial spider
    /// shadows (70% deep, Gaussian profile of 0.8 px).
    fn donut(vanes: usize) -> (Vec<f32>, usize) {
        let width = 96;
        let (cx, cy, r0, w, amp) = (48.3f32, 47.6f32, 8.0f32, 1.3f32, 600.0f32);
        let pixels = (0..width * width)
            .map(|i| {
                let (dx, dy) = ((i % width) as f32 - cx, (i / width) as f32 - cy);
                let r = (dx * dx + dy * dy).sqrt();
                let mut signal = amp * (-(r - r0).powi(2) / (2.0 * w * w)).exp();
                for j in 0..vanes {
                    let phi = 0.4 + j as f32 * std::f32::consts::TAU / vanes as f32;
                    let (ux, uy) = (phi.cos(), phi.sin());
                    if dx * ux + dy * uy > 0.0 {
                        let perp = (dx * uy - dy * ux).abs();
                        signal *= 1.0 - 0.7 * (-perp * perp / (2.0 * 0.8 * 0.8)).exp();
                    }
                }
                100.0 + noise(i, 3.0) + signal
            })
            .collect();
        (pixels, width)
    }

    #[test]
    fn star_gate_config_records_and_hashes_are_pinned() {
        // Python: hashlib.sha256(json.dumps(record, sort_keys=True,
        // separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
        let v2 = FocusSourcePolicy::default();
        assert_eq!(v2.blend_test, BlendTest::DipAnnulusV2);
        assert!(v2.regroup_deblended && !v2.accept_deblended);
        assert_eq!(
            crate::vision::background_model::canonical_json(&v2.config_json()),
            r#"{"accept_deblended":false,"annulus":{"bin_fraction":0.5,"bins":16,"half_width":0.25,"hole_fraction":0.6,"hole_radius":0.35,"min_coverage":0.625,"min_radius_px":2.5,"radius_tol":0.35},"blend_test":"dip_annulus_v2","dip_fraction":0.15,"dip_min_sigma":4.0,"peak_window_half":3,"regroup_deblended":true,"threshold_sigma":5.0}"#
        );
        assert_eq!(
            v2.config_sha256(),
            "77034b3d3e0fc0dc9661a822a8ae7baa0ef644eeaa2273ddd2fcf8ec60f93c6b"
        );
        assert_eq!(
            crate::vision::background_model::canonical_json(&v1_policy().config_json()),
            r#"{"accept_deblended":false,"blend_test":"peak_count_v1","peak_window_half":3,"regroup_deblended":false,"threshold_sigma":5.0}"#
        );
        assert_eq!(
            v1_policy().config_sha256(),
            "b517dd32a276a9e645c00d584cf9386bef2817584db91bf2087322d8e39e4460"
        );
        let v1_regroup = FocusSourcePolicy {
            regroup_deblended: true,
            ..v1_policy()
        };
        assert_eq!(
            v1_regroup.config_sha256(),
            "bb8329236a23e7337e8766d9f7032fa722d4c6535ad32a8c535adf918348b9ad"
        );
        assert_eq!(
            FocusSourcePolicy::new(&StarGateSettings::default(), false),
            v2
        );
        // Every result carries the hash, measured or not.
        let blank = mono(vec![100.0; 64 * 64], 64);
        let quality = estimate_quality(&blank, &background_from_luma(&blank));
        assert_eq!(quality.status, super::QualityStatus::NoSources);
        assert_eq!(quality.star_gates_config_sha256, v2.config_sha256());
    }

    #[test]
    fn donuts_with_zero_three_and_four_vanes_pass_v2_and_fail_v1() {
        for vanes in [0, 3, 4] {
            let (pixels, width) = donut(vanes);
            let frame = mono(pixels, width);
            let background = background_from_luma(&frame);
            let v2 = estimate_quality(&frame, &background);
            println!(
                "{vanes} vanes: v2 {:?} {:?} {:?}",
                v2.status, v2.star_rejections, v2.gate_diagnostics
            );
            assert_eq!(v2.star_count_detected, 1, "{vanes} vanes: one source");
            assert_eq!(v2.star_count_used, 1, "{vanes} vanes accepted by v2");
            assert_eq!(v2.star_rejections, super::StarRejectionTallies::default());
            if vanes > 0 {
                // The vanes cut the ridge into arcs with real dips; the
                // annulus exemption accepts them as one source.
                assert_eq!(
                    v2.gate_diagnostics,
                    GateDiagnostics {
                        annulus_exempt: 1,
                        regrouped_accepted: 0
                    },
                    "{vanes} vanes"
                );
            }
            let v1 = super::estimate_quality_with_policy(&frame, &background, v1_policy());
            assert_eq!(v1.star_count_used, 0, "{vanes} vanes: v1 rejects the donut");
            assert_eq!(v1.star_rejections.blended, 1);
            assert_eq!(v1.gate_diagnostics, GateDiagnostics::default());
        }
    }

    #[test]
    fn close_double_at_six_px_is_rejected_by_v2() {
        let width = 96;
        let mut pixels: Vec<f32> = (0..width * width).map(|i| 100.0 + noise(i, 3.0)).collect();
        add_gaussian(&mut pixels, width, 45.0, 48.0, 800.0, 1.5, 8);
        add_gaussian(&mut pixels, width, 51.0, 48.0, 800.0, 1.5, 8);
        let frame = mono(pixels, width);
        let background = background_from_luma(&frame);
        let v2 = estimate_quality(&frame, &background);
        assert_eq!(v2.star_count_used, 0);
        assert_eq!(v2.star_rejections.blended, 1);
        assert_eq!(v2.gate_diagnostics, GateDiagnostics::default());
        let v1 = super::estimate_quality_with_policy(&frame, &background, v1_policy());
        assert_eq!(v1.star_rejections.blended, 1);
    }

    #[test]
    fn neighbouring_star_in_the_crop_is_rejected_by_v2() {
        // Two separate detections 12 px apart: each crop holds the other.
        let width = 96;
        let mut pixels: Vec<f32> = (0..width * width).map(|i| 100.0 + noise(i, 3.0)).collect();
        add_gaussian(&mut pixels, width, 42.0, 48.0, 800.0, 1.5, 8);
        add_gaussian(&mut pixels, width, 54.0, 48.0, 500.0, 1.5, 8);
        let frame = mono(pixels, width);
        let background = background_from_luma(&frame);
        let v2 = estimate_quality(&frame, &background);
        assert_eq!(v2.star_count_detected, 2);
        assert_eq!(v2.star_count_used, 0);
        assert_eq!(v2.star_rejections.blended, 2);
        let v1 = super::estimate_quality_with_policy(&frame, &background, v1_policy());
        assert_eq!(v1.star_rejections.blended, 2);
    }

    #[test]
    fn a_disjoint_noise_bump_blends_under_v1_only() {
        let width = 96;
        let mut pixels: Vec<f32> = (0..width * width).map(|i| 100.0 + noise(i, 3.0)).collect();
        add_gaussian(&mut pixels, width, 48.0, 48.0, 800.0, 1.5, 8);
        // One pixel ~15 ADU up (5 sigma is ~11 ADU here): a peak, but its dip
        // to the threshold is below DIP_MIN_SIGMA sigma.
        pixels[44 * width + 56] = 115.0;
        let frame = mono(pixels, width);
        let background = background_from_luma(&frame);
        let v1 = super::estimate_quality_with_policy(&frame, &background, v1_policy());
        assert_eq!(
            v1.star_rejections.blended, 1,
            "v1 counts the bump as a peak"
        );
        let v2 = estimate_quality(&frame, &background);
        assert_eq!(v2.star_count_used, 1);
        assert_eq!(v2.star_rejections, super::StarRejectionTallies::default());
        assert_eq!(v2.gate_diagnostics, GateDiagnostics::default());
    }

    /// A catalog object from explicit footprint pixels, as SEP would report
    /// it (flux-weighted barycentre of the residual).
    fn object_from(
        frame: &LumaFrame,
        level: f32,
        pixels: Vec<u32>,
        deblended: bool,
    ) -> CatalogSource {
        let (mut flux, mut sx, mut sy) = (0.0f64, 0.0f64, 0.0f64);
        let (mut peak, mut peak_raw) = (f32::NEG_INFINITY, 0.0f32);
        let (mut x0, mut x1, mut y0, mut y1) = (usize::MAX, 0, usize::MAX, 0);
        for &p in &pixels {
            let (x, y) = (p as usize % frame.width, p as usize / frame.width);
            let r = frame.pixels[p as usize] - level;
            flux += r as f64;
            sx += r as f64 * x as f64;
            sy += r as f64 * y as f64;
            if r > peak {
                peak = r;
                peak_raw = frame.pixels[p as usize];
            }
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
        }
        CatalogSource {
            x: (sx / flux) as f32,
            y: (sy / flux) as f32,
            xmin: x0,
            xmax: x1,
            ymin: y0,
            ymax: y1,
            npix: pixels.len() as u32,
            flux: flux as f32,
            peak_residual: peak,
            peak_raw,
            a: 2.0,
            b: 2.0,
            theta: 0.0,
            flags: u16::from(deblended),
            deblended,
            truncated: false,
            pixels,
        }
    }

    /// The pixels above `5 * rms` of `frame` split into children by `key`
    /// (deblended), on uniform maps.
    fn split_catalog(
        frame: &LumaFrame,
        level: f32,
        rms: f32,
        key: impl Fn(usize, usize) -> usize,
        children: usize,
    ) -> BackgroundAnalysis {
        let mut parts = vec![Vec::new(); children];
        for (i, &v) in frame.pixels.iter().enumerate() {
            if v - level > 5.0 * rms {
                parts[key(i % frame.width, i / frame.width)].push(i as u32);
            }
        }
        let objects: Vec<CatalogSource> = parts
            .into_iter()
            .filter(|p| !p.is_empty())
            .map(|p| object_from(frame, level, p, true))
            .collect();
        let mut analysis = background_from_luma(frame);
        analysis.catalog = Some(SourceCatalog {
            provenance: DetectionProvenance {
                method: DetectionMethod::SepExtract,
                config_sha256: "test".into(),
                crop_sky_source: CropSkySource::ParentBackgroundMap,
                noise_floor_applied: false,
                objects_detected: objects.len() as u32,
                objects_deblended: objects.len() as u32,
                objects_truncated: 0,
            },
            objects,
            segmentation: None,
        });
        analysis.field = Some(BackgroundField::Map {
            level: vec![level; frame.pixels.len()],
            rms: vec![rms; frame.pixels.len()],
            width: frame.width,
            height: frame.width,
            invalid_noise: None,
        });
        analysis
    }

    #[test]
    fn a_ring_split_by_the_deblender_is_regrouped_and_accepted() {
        // SEP cuts a vane donut at its dips; the family is the donut.
        let (pixels, width) = donut(4);
        let frame = mono(pixels, width);
        let rms = 3.0 / 3.0_f32.sqrt();
        let quadrant = |x: usize, y: usize| {
            let angle = (y as f32 - 47.6).atan2(x as f32 - 48.3) - 0.4;
            (angle.rem_euclid(std::f32::consts::TAU) / std::f32::consts::FRAC_PI_2) as usize % 4
        };
        let analysis = split_catalog(&frame, 100.0, rms, quadrant, 4);
        assert_eq!(analysis.catalog.as_ref().unwrap().objects.len(), 4);

        let v2 = estimate_quality(&frame, &analysis);
        assert_eq!(v2.candidates.len(), 1, "one family candidate");
        let family = &v2.candidates[0];
        assert_eq!(family.family_size, 4);
        assert!(family.deblended);
        assert_eq!(
            family.area,
            analysis
                .catalog
                .as_ref()
                .unwrap()
                .objects
                .iter()
                .map(|o| o.npix as usize)
                .sum::<usize>()
        );
        assert!((family.x - 48.3).abs() < 0.5 && (family.y - 47.6).abs() < 0.5);
        assert_eq!(v2.star_count_used, 1);
        assert_eq!(v2.star_rejections, super::StarRejectionTallies::default());
        assert_eq!(
            v2.gate_diagnostics,
            GateDiagnostics {
                annulus_exempt: 1,
                regrouped_accepted: 1
            }
        );

        // v1: every child is rejected by the deblending policy.
        let v1 = super::estimate_quality_with_policy(&frame, &analysis, v1_policy());
        assert_eq!(v1.candidates.len(), 4);
        assert_eq!(v1.star_rejections.deblended, 4);
        assert_eq!(v1.star_count_used, 0);
        // Regrouped but judged by peak_count_v1: the split is confirmed.
        let v1_regroup = FocusSourcePolicy {
            regroup_deblended: true,
            ..v1_policy()
        };
        let judged = super::estimate_quality_with_policy(&frame, &analysis, v1_regroup);
        assert_eq!(judged.star_rejections.deblended, 1);
        assert_eq!(judged.star_rejections.blended, 0);
        // accept_deblended takes the children one by one, without regrouping.
        let accepted = super::estimate_quality_with_policy(
            &frame,
            &analysis,
            FocusSourcePolicy {
                accept_deblended: true,
                ..Default::default()
            },
        );
        assert_eq!(accepted.candidates.len(), 4);
        assert!(accepted.candidates.iter().all(|c| c.family_size == 1));
    }

    #[test]
    fn a_family_of_two_stars_is_rejected_as_deblended() {
        let width = 96;
        let mut pixels: Vec<f32> = (0..width * width).map(|i| 100.0 + noise(i, 3.0)).collect();
        add_gaussian(&mut pixels, width, 44.0, 48.0, 800.0, 1.5, 8);
        add_gaussian(&mut pixels, width, 52.0, 48.0, 700.0, 1.5, 8);
        let frame = mono(pixels, width);
        let rms = 3.0 / 3.0_f32.sqrt();
        let analysis = split_catalog(&frame, 100.0, rms, |x, _| usize::from(x >= 48), 2);
        let v2 = estimate_quality(&frame, &analysis);
        assert_eq!(v2.candidates.len(), 1);
        assert_eq!(v2.candidates[0].family_size, 2);
        assert_eq!(v2.star_count_used, 0);
        assert_eq!(v2.star_rejections.deblended, 1);
        assert_eq!(v2.star_rejections.blended, 0);
        assert_eq!(v2.gate_diagnostics, GateDiagnostics::default());
        let v1 = super::estimate_quality_with_policy(&frame, &analysis, v1_policy());
        assert_eq!(v1.star_rejections.deblended, 2);
    }

    #[test]
    fn a_crop_touching_locally_invalid_noise_is_rejected() {
        let (frame, level, rms) = star_on_ramp();
        let mut analysis = spatial_analysis(&frame, level, rms, false);
        let mut invalid = vec![0u8; frame.pixels.len()];
        // Inside the 32 px crop around (80, 64), clear of the star core.
        invalid[70 * frame.width + 90] = 1;
        if let Some(BackgroundField::Map { invalid_noise, .. }) = analysis.field.as_mut() {
            *invalid_noise = Some(invalid.clone());
        }
        let quality = estimate_quality(&frame, &analysis);
        assert_eq!(quality.status, super::QualityStatus::NoUsableStars);
        assert_eq!(quality.star_rejections.invalid_noise_support, 1);
        assert_eq!(quality.star_rejections.masked_support, 0);
        // Outside the crop it is irrelevant.
        invalid[70 * frame.width + 90] = 0;
        invalid[5 * frame.width + 5] = 1;
        if let Some(BackgroundField::Map { invalid_noise, .. }) = analysis.field.as_mut() {
            *invalid_noise = Some(invalid);
        }
        assert_eq!(
            estimate_quality(&frame, &analysis).status,
            super::QualityStatus::Valid
        );
    }
}

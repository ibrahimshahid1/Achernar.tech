//! Classical frame quality metrics for the shared vision bus.

use serde::{Deserialize, Serialize};

use crate::error::HardwareError;
use crate::focus::classical_focus_solver::ClassicalFocusSolver;
use crate::hardware::{camera::FrameData, ols_ffi::OlsStreamType};

use super::background::BackgroundStats;

#[derive(Debug, Clone)]
pub struct LumaFrame {
    pub pixels: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub max_adu: f32,
}

/// Why a frame did (not) yield a usable focus measurement. `Valid` means at
/// least one detected star produced a finite HFR. Anything else means no width
/// could be measured — which must serialize as `null`, never a spuriously good 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityStatus {
    Valid,
    EmptyFrame,
    NoSources,
    NoUsableStars,
}

/// Specific reasons a frame was rejected, for diagnostics and dashboards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityRejectionReason {
    NoFinitePixels,
    NoStellarSources,
    NoValidHfr,
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
}

impl QualityMetrics {
    /// Legacy convenience: the observation is valid iff a width was measured.
    pub fn observation_valid(&self) -> bool {
        matches!(self.status, QualityStatus::Valid)
    }
}

#[derive(Debug, Clone)]
struct StarCandidate {
    x: f32,
    y: f32,
    area: usize,
    flux: f32,
    peak: f32,
    /// Bounding box of the above-threshold pixels (inclusive, frame coords).
    /// Drives the adaptive crop size and the edge-truncation gate.
    min_x: usize,
    max_x: usize,
    min_y: usize,
    max_y: usize,
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
    /// Peak at/above ~98% of full scale: flat cores bias HFR low and would
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
/// Adaptive crop bounds. 32 px caps measurable HFR at ~15 px, which truncates
/// defocused stars at the V-curve wings; the crop now scales with the source
/// bounding box up to 128 px (HFR cap ~63 px).
const CROP_MIN_PX: usize = 32;
const CROP_MAX_PX: usize = 128;

pub fn luma_from_frame(frame: &FrameData) -> crate::error::Result<LumaFrame> {
    let width = frame.width.max(0) as usize;
    let height = frame.height.max(0) as usize;
    let expected = width
        .checked_mul(height)
        .ok_or_else(|| HardwareError::Communication("Frame dimensions overflowed".to_string()))?;
    if expected == 0 {
        return Err(
            HardwareError::Communication("Frame has invalid dimensions".to_string()).into(),
        );
    }

    let (mut pixels, max_adu) = match frame.format {
        OlsStreamType::Mjpeg => {
            let image = image::load_from_memory(&frame.data)
                .map_err(|err| HardwareError::Communication(err.to_string()))?
                .to_luma8();
            (
                image
                    .pixels()
                    .map(|pixel| pixel[0] as f32)
                    .collect::<Vec<_>>(),
                255.0,
            )
        }
        OlsStreamType::Rgb24 => (
            frame
                .data
                .chunks_exact(3)
                .map(|px| 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32)
                .collect(),
            255.0,
        ),
        OlsStreamType::Rgb48 => (
            frame
                .data
                .chunks_exact(6)
                .map(|px| {
                    let r = u16::from_le_bytes([px[0], px[1]]) as f32;
                    let g = u16::from_le_bytes([px[2], px[3]]) as f32;
                    let b = u16::from_le_bytes([px[4], px[5]]) as f32;
                    0.299 * r + 0.587 * g + 0.114 * b
                })
                .collect(),
            65535.0,
        ),
        OlsStreamType::Mono8 | OlsStreamType::Raw8 => (
            frame.data.iter().map(|value| *value as f32).collect(),
            255.0,
        ),
        OlsStreamType::Mono16 | OlsStreamType::Raw16 => (
            frame
                .data
                .chunks_exact(2)
                .map(|px| u16::from_le_bytes([px[0], px[1]]) as f32)
                .collect(),
            65535.0,
        ),
        OlsStreamType::Yuv2 => (
            frame
                .data
                .chunks_exact(4)
                .flat_map(|px| [px[0] as f32, px[2] as f32])
                .collect(),
            255.0,
        ),
        OlsStreamType::Error => {
            return Err(
                HardwareError::Communication("Camera returned an error frame".to_string()).into(),
            )
        }
    };

    if pixels.len() < expected {
        return Err(HardwareError::Communication(
            "Frame payload is shorter than expected".to_string(),
        )
        .into());
    }
    pixels.truncate(expected);

    Ok(LumaFrame {
        pixels,
        width,
        height,
        max_adu,
    })
}

pub fn estimate_quality(frame: &LumaFrame, background: &BackgroundStats) -> QualityMetrics {
    if frame.pixels.is_empty() || frame.width == 0 || frame.height == 0 {
        return invalid_metrics(
            QualityStatus::EmptyFrame,
            QualityRejectionReason::NoFinitePixels,
            0,
            StarRejectionTallies::default(),
        );
    }

    let mut stars = detect_star_candidates(frame, background);
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
        );
    }

    let (used_stars, star_rejections) = measure_stars(frame, &stars);
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
            }
        }
        // Stars were detected but none survived the validity gates with a usable
        // HFR. `star_count_detected` and the per-reason tallies stay as real
        // diagnostics; widths remain None.
        _ => invalid_metrics(
            QualityStatus::NoUsableStars,
            QualityRejectionReason::NoValidHfr,
            star_count_detected,
            star_rejections,
        ),
    }
}

fn invalid_metrics(
    status: QualityStatus,
    reason: QualityRejectionReason,
    star_count_detected: u32,
    star_rejections: StarRejectionTallies,
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

fn detect_star_candidates(frame: &LumaFrame, background: &BackgroundStats) -> Vec<StarCandidate> {
    let sigma = background.background_sigma.max(1.0);
    let threshold = (background.background_adu + sigma * 5.0).max(background.background_adu + 8.0);
    let mut visited = vec![false; frame.pixels.len()];
    let mut stars = Vec::new();

    for idx in 0..frame.pixels.len() {
        let value = frame.pixels[idx];
        if visited[idx] || !value.is_finite() || value < threshold {
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
            let signal = (current_value - background.background_adu).max(0.0);
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

        if area >= 2 && area <= 5_000 && flux > 0.0 {
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
            });
        }
    }

    stars
}

/// Measure HFRs for the brightest stars that pass the source-validity gates.
///
/// Candidates are examined in flux order; each must clear, in order of cost:
/// saturation (component peak), edge truncation (component bbox), then on an
/// adaptively-sized crop with a *local* sky estimate: aperture SNR, core pixel
/// count, second-moment ellipticity, and the two-peak blend test. Only then is
/// the curve-of-growth HFR computed. Without these gates, ranking by flux
/// preferentially selects saturated stars, whose flat cores bias HFR low right
/// where the autofocus curve needs discrimination.
fn measure_stars(
    frame: &LumaFrame,
    stars: &[StarCandidate],
) -> (Vec<UsedStar>, StarRejectionTallies) {
    let saturation_adu = saturation_threshold_adu(frame.max_adu);
    let mut used = Vec::new();
    let mut tallies = StarRejectionTallies::default();

    for star in stars.iter().take(MAX_STARS_EXAMINED) {
        if used.len() >= MAX_STARS_USED {
            break;
        }

        let mut rejected = false;

        // Gate 1 — saturation, from the component peak (no crop needed).
        if star.peak >= saturation_adu {
            tallies.saturated += 1;
            rejected = true;
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

        if rejected {
            continue;
        }

        // Adaptive crop: 2x the component bounding box, clamped. A defocused
        // star's >=5-sigma footprint scales with its width, so the crop (and
        // therefore the measurable HFR ceiling of min(crop)/2 - 1) follows it.
        let bbox_extent = (star.max_x - star.min_x + 1).max(star.max_y - star.min_y + 1);
        let crop_size = (2 * bbox_extent).clamp(CROP_MIN_PX, CROP_MAX_PX);
        let (crop, cw, ch, start_x, start_y) = extract_crop(frame, star.x, star.y, crop_size);

        // Local sky for the gates (median + MAD), matching what compute_hfr
        // uses internally and what the Python labeler does per crop.
        let (sky, rms) = local_sky_stats(&crop);
        let noise = rms.max(1e-6);
        let cx = (star.x - start_x as f32).clamp(0.0, (cw - 1) as f32);
        let cy = (star.y - start_y as f32).clamp(0.0, (ch - 1) as f32);

        // Gate 3 — aperture SNR + core pixel count. The sky-subtracted sum is
        // UNCLIPPED so zero-mean noise cancels; clipping first would rectify
        // noise and inflate the SNR of a blank patch.
        let r_ap = (cw.min(ch) as f32 / 8.0).max(4.0);
        let mut ap_signal = 0.0_f32;
        let mut n_ap = 0_usize;
        let mut core_pixels = 0_usize;
        for y in 0..ch {
            for x in 0..cw {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if dx * dx + dy * dy <= r_ap * r_ap {
                    let resid = crop[y * cw + x] - sky;
                    ap_signal += resid;
                    n_ap += 1;
                    if resid > 3.0 * noise {
                        core_pixels += 1;
                    }
                }
            }
        }
        let snr = ap_signal / (noise * (n_ap.max(1) as f32).sqrt());
        if snr < MIN_APERTURE_SNR {
            tallies.low_snr += 1;
            rejected = true;
        }
        if core_pixels < MIN_CORE_PIXELS {
            tallies.too_few_pixels += 1;
            rejected = true;
        }

        // Gate 4 — second-moment ellipticity (trailed/astigmatic source).
        let ellipticity = second_moment_ellipticity(&crop, cw, ch, sky);
        if ellipticity > MAX_ELLIPTICITY {
            tallies.elongated += 1;
            rejected = true;
        }

        // Gate 5 — blend test: >=2 distinct core peaks means a close double.
        // Runs unconditionally (like the Python labeler) so every triggered
        // reason is tallied, not just the first.
        if count_core_peaks(&crop, cw, ch, sky, noise) >= 2 {
            tallies.blended += 1;
            rejected = true;
        }

        if rejected {
            continue;
        }

        let hfr = ClassicalFocusSolver::compute_hfr(&crop, cw, ch);
        if hfr.is_finite() && hfr > 0.0 {
            used.push(UsedStar {
                x: star.x,
                y: star.y,
                hfr_px: hfr as f32,
            });
        } else {
            tallies.invalid_hfr += 1;
        }
    }

    (used, tallies)
}

/// Saturation threshold in ADU (mirrors `background::saturation_threshold`).
fn saturation_threshold_adu(max_adu: f32) -> f32 {
    if max_adu <= 0.0 {
        f32::INFINITY
    } else {
        max_adu * SATURATION_FRACTION_OF_MAX
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
fn second_moment_ellipticity(crop: &[f32], width: usize, height: usize, sky: f32) -> f32 {
    let mut total = 0.0_f32;
    let mut cx = 0.0_f32;
    let mut cy = 0.0_f32;
    for y in 0..height {
        for x in 0..width {
            let f = (crop[y * width + x] - sky).max(0.0);
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
            let f = (crop[y * width + x] - sky).max(0.0);
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

/// Count distinct local maxima above `sky + 5*noise` — a cheap deblend test
/// (port of `psf_measure._count_core_peaks`). A pixel is a peak when it equals
/// the maximum of its `(2*BLEND_MIN_SEPARATION+1)^2` window; touching plateau
/// peaks collapse into one via connected components.
fn count_core_peaks(crop: &[f32], width: usize, height: usize, sky: f32, noise: f32) -> usize {
    let threshold = sky + 5.0 * noise;
    let win = BLEND_MIN_SEPARATION;
    let mut is_peak = vec![false; crop.len()];

    for y in 0..height {
        for x in 0..width {
            let value = crop[y * width + x];
            if !value.is_finite() || value <= threshold {
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
    use super::{estimate_quality, local_sky_stats, LumaFrame};
    use crate::vision::background::estimate_background;

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
        };
        let background = estimate_background(&frame.pixels, frame.max_adu);
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
        };
        let background = estimate_background(&frame.pixels, frame.max_adu);
        let quality = estimate_quality(&frame, &background);

        assert_eq!(quality.star_count_detected, 1);
        assert_eq!(quality.star_count_used, 0);
        assert_eq!(quality.status, super::QualityStatus::NoUsableStars);
        assert!(
            quality.hfr_px.is_none(),
            "saturated star must not yield a width"
        );
        assert_eq!(quality.star_rejections.saturated, 1);
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
        };
        let background = estimate_background(&frame.pixels, frame.max_adu);
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
        };
        let background = estimate_background(&frame.pixels, frame.max_adu);
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
        };
        let background = estimate_background(&frame.pixels, frame.max_adu);
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
        };
        let background = estimate_background(&frame.pixels, frame.max_adu);
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
        };
        let background = estimate_background(&frame.pixels, frame.max_adu);
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
        };
        let background = estimate_background(&frame.pixels, frame.max_adu);
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
}

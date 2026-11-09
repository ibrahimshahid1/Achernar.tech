//! Legacy global background estimator (`legacy_global_clip`, method v1).
//!
//! A frame median, a robust scale seed, one ±3-seed-sigma clipping pass, then
//! the clipped median and the clipped population standard deviation. It has
//! no spatial model and is not SEP or Source Extractor: Bertin & Arnouts
//! (1996, §2) clip each mesh iteratively and interpolate a map. It is kept,
//! named and versioned, as the default live path and as the diagnostic
//! comparator for the opt-in SEP backend.

use super::background_model::{
    config_sha256, legacy_config_json, BackgroundAnalysis, BackgroundField, BackgroundMethod,
    BackgroundReason, BackgroundResult, BackgroundSupport, LegacyGlobalClipConfig, LevelStatus,
    NoiseSource, NoiseStatus, BACKGROUND_SCHEMA_VERSION,
};
use super::quality::LumaFrame;

/// Statistics of the legacy estimator over the usable pixels it was given.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LegacyGlobalStats {
    /// Clipped median (input units).
    pub level: f32,
    /// Population standard deviation of the clipped sample (input units).
    /// Zero on a quantized plateau: zero *scatter*, not zero detector noise.
    pub sigma: f32,
    pub finite_pixels: usize,
    pub clipped_pixels: usize,
    /// Fraction of finite pixels at/above 98% of `saturation_adu`; 0 when the
    /// saturation point is unknown (callers decide whether that is evidence).
    pub saturation_fraction: f32,
    /// Legacy pixel-significance heuristic (formerly `snr_proxy_median`).
    pub bright_pixel_significance: Option<f32>,
}

/// Robust global sky statistics, or `None` without any finite pixel.
///
/// `saturation_adu` is the sensor's linear saturation point, or `None` when it
/// is unknown — see [`crate::vision::quality::LumaFrame::saturation_adu`].
/// Unknown means no pixel is excluded and the fraction is 0, which is safe for
/// this median/MAD estimate (robust to a few clipped pixels) but is not
/// evidence of an unclipped frame; [`legacy_result`] reports it as absent.
pub fn estimate_background(
    pixels: &[f32],
    saturation_adu: Option<f32>,
) -> Option<LegacyGlobalStats> {
    let mut finite: Vec<f32> = pixels.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return None;
    }

    finite.sort_by(|a, b| a.total_cmp(b));
    let median = percentile_sorted(&finite, 0.5);
    let mad_sigma = robust_sigma(&finite, median);
    let interquantile_sigma =
        0.5 * (percentile_sorted(&finite, 0.84) - percentile_sorted(&finite, 0.16));
    // Never seed source clipping from the global standard deviation: stars are
    // exactly the high-side outliers being removed and can inflate it enough to
    // survive. MAD and the central 16-84% span are robust sky-noise scales.
    let sigma_seed = mad_sigma.max(interquantile_sigma).max(0.0);
    let clipped: Vec<f32> = if sigma_seed > f32::EPSILON {
        let radius = 3.0 * sigma_seed;
        finite
            .iter()
            .copied()
            .filter(|value| (*value - median).abs() <= radius)
            .collect()
    } else {
        // Quantized flat backgrounds often have MAD=IQR=0. Keep the median
        // plateau rather than allowing sparse bright sources into the sky RMS.
        finite
            .iter()
            .copied()
            .filter(|value| (*value - median).abs() <= f32::EPSILON)
            .collect()
    };

    let level = if clipped.is_empty() {
        median
    } else {
        let mut clipped_sorted = clipped.clone();
        clipped_sorted.sort_by(|a, b| a.total_cmp(b));
        percentile_sorted(&clipped_sorted, 0.5)
    };
    let sigma = if clipped.is_empty() {
        sigma_seed
    } else {
        stddev(&clipped, mean(&clipped))
    };

    let saturation_threshold = saturation_threshold(saturation_adu);
    let saturated = finite
        .iter()
        .filter(|value| **value >= saturation_threshold)
        .count();

    Some(LegacyGlobalStats {
        level,
        sigma,
        finite_pixels: finite.len(),
        clipped_pixels: clipped.len(),
        saturation_fraction: saturated as f32 / finite.len() as f32,
        bright_pixel_significance: bright_pixel_significance(pixels, level, sigma),
    })
}

/// The authoritative legacy result for `frame`, from the statistics of its
/// finite measurement-valid, non-clipped samples (`None` when there are none).
pub fn legacy_result(frame: &LumaFrame, stats: Option<&LegacyGlobalStats>) -> BackgroundAnalysis {
    let config = legacy_config_json(&LegacyGlobalClipConfig::default());
    let units = frame
        .analysis_descriptor
        .as_ref()
        .map(|descriptor| descriptor.units.clone())
        .unwrap_or_else(|| "unverified".to_string());
    let saturation_evidence = frame.saturation_evidence();
    let saturation_fraction = if saturation_evidence == "unknown" {
        None
    } else if frame.analysis_descriptor.is_some() {
        // Versioned analysis: the fraction of analysis pixels touched by the
        // reconstruction footprint of clipped native samples.
        (!frame.pixels.is_empty()).then(|| {
            frame
                .masks
                .native_saturation
                .iter()
                .filter(|value| **value != 0)
                .count() as f32
                / frame.pixels.len() as f32
        })
    } else {
        stats.map(|stats| stats.saturation_fraction)
    };

    let mut reasons = vec![];
    let (level_status, noise_status, level, rms) = match stats {
        None => {
            reasons.push(BackgroundReason::NoUsablePixels);
            (
                LevelStatus::Unavailable,
                NoiseStatus::Unavailable,
                None,
                None,
            )
        }
        Some(stats) if stats.sigma > 0.0 && stats.sigma.is_finite() => (
            LevelStatus::Measured,
            NoiseStatus::Measured,
            Some(stats.level),
            Some(stats.sigma),
        ),
        Some(stats) => {
            reasons.push(BackgroundReason::ZeroEmpiricalScatter);
            (
                LevelStatus::Measured,
                NoiseStatus::Unresolved,
                Some(stats.level),
                None,
            )
        }
    };

    let result = BackgroundResult {
        schema_version: BACKGROUND_SCHEMA_VERSION,
        method: BackgroundMethod::LegacyGlobalClip,
        method_version: BackgroundMethod::LegacyGlobalClip.method_version(),
        library: None,
        library_version: None,
        requested_method: BackgroundMethod::LegacyGlobalClip,
        config_sha256: config_sha256(&config),
        config,
        level_status,
        noise_status,
        reasons,
        backend_error: None,
        units,
        coordinate_frame: "analysis_pixel".to_string(),
        width: frame.width as u32,
        height: frame.height as u32,
        noise_source: if stats.is_some() {
            NoiseSource::EmpiricalGlobalClip
        } else {
            NoiseSource::Unavailable
        },
        support: BackgroundSupport {
            total_pixels: frame.pixels.len() as u64,
            usable_pixels: stats.map_or(0, |stats| stats.finite_pixels as u64),
            clipped_sample_pixels: stats.map(|stats| stats.clipped_pixels as u64),
            masks: frame.mask_provenance(),
            saturation_evidence: saturation_evidence.to_string(),
        },
        level,
        level_definition: "median of the usable pixels kept by one pass of |v - median| <= \
                           3*max(1.4826*MAD, 0.5*(p84 - p16)) around the usable-pixel median \
                           (the median plateau when that scale is 0)"
            .to_string(),
        rms,
        rms_definition: "population standard deviation of that clipped sample (no truncation \
                         correction); absent when it is zero"
            .to_string(),
        mesh: None,
        plane: None,
        plane_unavailable_reason: Some(BackgroundReason::NoSpatialModel),
        bright_pixel_significance: stats.and_then(|stats| stats.bright_pixel_significance),
        saturation_fraction,
    };
    let field = stats.map(|stats| BackgroundField::Uniform {
        level: stats.level,
        sigma: stats.sigma,
    });
    BackgroundAnalysis {
        result,
        field,
        catalog: None,
        catalog_error: None,
    }
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len().max(1) as f32
}

fn stddev(values: &[f32], center: f32) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|value| {
            let delta = *value - center;
            delta * delta
        })
        .sum::<f32>()
        / values.len() as f32;
    variance.sqrt()
}

fn robust_sigma(sorted_values: &[f32], median: f32) -> f32 {
    let mut deviations: Vec<f32> = sorted_values
        .iter()
        .map(|value| (*value - median).abs())
        .collect();
    deviations.sort_by(|a, b| a.total_cmp(b));
    percentile_sorted(&deviations, 0.5) * 1.4826
}

fn percentile_sorted(sorted_values: &[f32], quantile: f32) -> f32 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    if sorted_values.len() == 1 {
        return sorted_values[0];
    }
    let position = (sorted_values.len() - 1) as f64 * quantile.clamp(0.0, 1.0) as f64;
    let lower = position.floor() as usize;
    let upper = (lower + 1).min(sorted_values.len() - 1);
    let fraction = (position - lower as f64) as f32;
    sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
}

fn saturation_threshold(saturation_adu: Option<f32>) -> f32 {
    match saturation_adu {
        // Unknown: exclude nothing. An infinite threshold is the honest
        // encoding of "we cannot test this" — the alternative, deriving one
        // from the container depth, produces a finite-looking threshold that
        // no sample on a 14-bit body can ever reach.
        Some(sat) if sat > 0.0 => {
            // 0.98 rather than full scale: sensor response goes non-linear
            // below the hard clip, and the previous `max(..., max_adu - 1)`
            // accidentally made this an almost-100% threshold that missed
            // clipped cores between 98% and full scale.
            sat * 0.98
        }
        _ => f32::INFINITY,
    }
}

/// Legacy heuristic, unchanged: median of `(v-B)/sqrt(sigma^2+9)` over pixels
/// with `v-B > 3*max(sigma,1)`. The fixed 9 ADU^2 term and the 1 ADU floor
/// are not a camera calibration; this is a pixel-significance proxy, not
/// aperture or PSF SNR. `None` when no pixel qualifies.
fn bright_pixel_significance(pixels: &[f32], level: f32, sigma: f32) -> Option<f32> {
    let noise = (sigma * sigma + 9.0).sqrt().max(1.0);
    let mut values: Vec<f32> = pixels
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .filter_map(|value| {
            let signal = value - level;
            (signal > 3.0 * sigma.max(1.0)).then_some(signal / noise)
        })
        .collect();

    if values.is_empty() {
        return None;
    }

    values.sort_by(|a, b| a.total_cmp(b));
    Some(percentile_sorted(&values, 0.5))
}

#[cfg(test)]
mod tests {
    use super::{estimate_background, legacy_result, percentile_sorted};
    use crate::vision::background_model::{BackgroundReason, LevelStatus, NoiseStatus};
    use crate::vision::quality::LumaFrame;

    fn frame(pixels: Vec<f32>, saturation_adu: Option<f32>) -> LumaFrame {
        LumaFrame {
            width: pixels.len(),
            height: 1,
            pixels,
            max_adu: 65535.0,
            saturation_adu,
            masks: Default::default(),
            analysis_descriptor: None,
        }
    }

    #[test]
    fn percentile_uses_linear_interpolation_for_even_samples() {
        assert!((percentile_sorted(&[1.0, 2.0, 3.0, 4.0], 0.5) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn background_uses_clipped_median_and_counts_saturation() {
        let mut pixels = vec![100.0_f32; 100];
        pixels.extend([102.0, 98.0, 65535.0, 65534.0]);

        let stats = estimate_background(&pixels, Some(65535.0)).unwrap();

        assert!((98.0..=102.0).contains(&stats.level));
        assert!(stats.sigma < 10.0);
        assert!(stats.saturation_fraction > 0.0);
    }

    #[test]
    fn bright_sources_do_not_inflate_flat_sky_noise() {
        let mut pixels = vec![100.0_f32; 90];
        pixels.extend(vec![10_000.0_f32; 10]);

        let stats = estimate_background(&pixels, Some(65_535.0)).unwrap();

        assert_eq!(stats.level, 100.0);
        assert_eq!(stats.sigma, 0.0);
    }

    #[test]
    fn no_finite_pixel_is_no_estimate_rather_than_zero_sky() {
        assert!(estimate_background(&[f32::NAN, f32::INFINITY], Some(1000.0)).is_none());
        assert!(estimate_background(&[], None).is_none());

        let analysis = legacy_result(&frame(vec![f32::NAN; 4], Some(1000.0)), None);
        assert_eq!(analysis.result.level_status, LevelStatus::Unavailable);
        assert_eq!(analysis.result.noise_status, NoiseStatus::Unavailable);
        assert_eq!(analysis.result.level, None);
        assert_eq!(analysis.result.rms, None);
        assert_eq!(analysis.result.saturation_fraction, None);
        assert_eq!(
            analysis.result.reasons,
            vec![BackgroundReason::NoUsablePixels]
        );
        assert!(analysis.field.is_none());
        let json = serde_json::to_value(&analysis.result).unwrap();
        assert!(json["level"].is_null() && json["rms"].is_null());
    }

    #[test]
    fn zero_scatter_plateau_has_a_level_but_unresolved_noise() {
        // Quantized flat sky: a level exists, but zero empirical scatter is
        // not zero detector noise (BGE-07).
        let pixels = vec![100.0_f32; 64];
        let stats = estimate_background(&pixels, None).unwrap();
        let analysis = legacy_result(&frame(pixels, None), Some(&stats));
        assert_eq!(analysis.result.level(), Some(100.0));
        assert_eq!(analysis.result.noise_status, NoiseStatus::Unresolved);
        assert_eq!(analysis.result.rms(), None);
        assert!(analysis
            .result
            .reasons
            .contains(&BackgroundReason::ZeroEmpiricalScatter));
    }

    #[test]
    fn measured_zero_sky_is_a_value_and_unknown_saturation_is_not_zero() {
        // Calibrated residual frames can legitimately sit at 0.
        let pixels: Vec<f32> = (0..256).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
        let stats = estimate_background(&pixels, None).unwrap();
        let analysis = legacy_result(&frame(pixels, None), Some(&stats));
        assert_eq!(analysis.result.level(), Some(0.0));
        assert!(analysis.result.rms().unwrap() > 0.0);
        assert_eq!(analysis.result.support.saturation_evidence, "unknown");
        assert_eq!(analysis.result.saturation_fraction, None);
        assert_eq!(analysis.result.units, "unverified");
        assert_eq!(
            analysis
                .result
                .support
                .masks
                .get("native_saturation")
                .map(String::as_str),
            Some("unavailable")
        );
        assert!(analysis.result.plane.is_none());
        assert_eq!(
            analysis.result.plane_unavailable_reason,
            Some(BackgroundReason::NoSpatialModel)
        );
    }
}

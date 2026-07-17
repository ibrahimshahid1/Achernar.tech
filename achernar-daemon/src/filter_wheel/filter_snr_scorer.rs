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
        filter_names: &[String],
        moon_altitude: f64,
        moon_phase: f64,
        target_type: &str,
        exposure_seconds: f64,
    ) -> Vec<FilterSnrScore> {
        let moon_visible = if moon_altitude > 0.0 { 1.0 } else { 0.25 };
        let moon_brightness = moon_phase.clamp(0.0, 1.0) * moon_visible;
        let exposure = exposure_seconds.max(1.0);

        let mut scores: Vec<FilterSnrScore> = filter_names
            .iter()
            .map(|filter_name| {
                let target_flux = target_flux_proxy(filter_name, target_type);
                let sky_brightness = sky_brightness_proxy(filter_name, moon_brightness);
                let read_noise = 5.0_f64;
                let dark_noise = 0.05_f64;
                let aperture_pixels = 50.0_f64;
                let signal = target_flux * exposure;
                let noise = (signal
                    + sky_brightness * exposure * aperture_pixels
                    + aperture_pixels * dark_noise * exposure
                    + read_noise * read_noise)
                    .sqrt()
                    .max(f64::EPSILON);

                FilterSnrScore {
                    filter_name: filter_name.clone(),
                    estimated_snr: signal / noise,
                    sky_brightness,
                    target_flux,
                }
            })
            .collect();

        scores.sort_by(|a, b| b.estimated_snr.total_cmp(&a.estimated_snr));
        scores
    }
}

fn target_flux_proxy(filter_name: &str, target_type: &str) -> f64 {
    let filter = filter_name.to_ascii_lowercase();
    let target = target_type.to_ascii_lowercase();
    let base = match filter.as_str() {
        "l" | "lum" | "luminance" => 1.0,
        "r" | "g" | "b" => 0.55,
        "ha" | "h-alpha" => {
            if target.contains("nebula") {
                0.85
            } else {
                0.18
            }
        }
        "oiii" | "o_iii" => {
            if target.contains("nebula") || target.contains("planetary") {
                0.65
            } else {
                0.12
            }
        }
        "sii" | "s_ii" => {
            if target.contains("nebula") {
                0.45
            } else {
                0.08
            }
        }
        _ => 0.4,
    };

    if target.contains("galaxy") && matches!(filter.as_str(), "l" | "lum" | "luminance") {
        base * 1.2
    } else {
        base
    }
}

fn sky_brightness_proxy(filter_name: &str, moon_brightness: f64) -> f64 {
    let filter = filter_name.to_ascii_lowercase();
    let moon_sensitivity = match filter.as_str() {
        "ha" | "h-alpha" | "oiii" | "o_iii" | "sii" | "s_ii" => 0.18,
        "r" => 0.65,
        "g" => 0.85,
        "b" => 1.0,
        _ => 0.90,
    };

    8.0 + 80.0 * moon_brightness * moon_sensitivity
}

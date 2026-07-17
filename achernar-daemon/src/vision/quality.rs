//! Classical frame quality metrics for the shared vision bus.

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

#[derive(Debug, Clone, Copy)]
pub struct QualityMetrics {
    pub focus_quality: f32,
    pub sharpness_score: f32,
    pub quality_uncertainty: f32,
    pub hfr: f32,
    pub fwhm: f32,
    pub seeing_estimate_arcsec: f32,
    pub star_count: u32,
    pub frame_quality_score: f32,
    pub observation_valid: bool,
}

#[derive(Debug, Clone)]
struct StarCandidate {
    x: f32,
    y: f32,
    area: usize,
    flux: f32,
    peak: f32,
}

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
        return invalid_metrics();
    }

    let mut stars = detect_star_candidates(frame, background);
    stars.sort_by(|a, b| {
        b.flux
            .total_cmp(&a.flux)
            .then_with(|| b.peak.total_cmp(&a.peak))
            .then_with(|| b.area.cmp(&a.area))
    });

    let hfr = median_hfr(frame, &stars).unwrap_or(0.0);
    let star_count = stars.len() as u32;
    let observation_valid = hfr.is_finite() && hfr > 0.0 && star_count > 0;
    let focus_quality = if observation_valid {
        (1.0 / (1.0 + hfr / 3.0)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let sharpness_score = if observation_valid {
        (1.0 / (1.0 + hfr)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let quality_uncertainty = quality_uncertainty(
        star_count,
        background.background_sigma,
        background.saturation_fraction,
    );
    let star_factor = (star_count as f32 / 20.0).clamp(0.0, 1.0);
    let saturation_penalty = (1.0 - background.saturation_fraction * 4.0).clamp(0.0, 1.0);
    let frame_quality_score = if observation_valid {
        ((focus_quality * 0.55) + (sharpness_score * 0.15) + (star_factor * 0.3))
            * saturation_penalty
    } else {
        0.0
    };

    QualityMetrics {
        focus_quality,
        sharpness_score,
        quality_uncertainty,
        hfr,
        fwhm: hfr * 2.0,
        seeing_estimate_arcsec: hfr * 2.0,
        star_count,
        frame_quality_score,
        observation_valid,
    }
}

fn invalid_metrics() -> QualityMetrics {
    QualityMetrics {
        focus_quality: 0.0,
        sharpness_score: 0.0,
        quality_uncertainty: 1.0,
        hfr: 0.0,
        fwhm: 0.0,
        seeing_estimate_arcsec: 0.0,
        star_count: 0,
        frame_quality_score: 0.0,
        observation_valid: false,
    }
}

fn quality_uncertainty(star_count: u32, background_sigma: f32, saturation_fraction: f32) -> f32 {
    let star_term = 1.0 / (1.0 + star_count as f32 / 8.0);
    let noise_term = (background_sigma / 500.0).clamp(0.0, 0.4);
    let saturation_term = (saturation_fraction * 5.0).clamp(0.0, 0.4);
    (star_term + noise_term + saturation_term).clamp(0.0, 1.0)
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
            });
        }
    }

    stars
}

fn median_hfr(frame: &LumaFrame, stars: &[StarCandidate]) -> Option<f32> {
    let mut hfrs: Vec<f32> = stars
        .iter()
        .take(16)
        .filter_map(|star| {
            let (crop, width, height) = extract_crop(frame, star.x, star.y, 32);
            let hfr = ClassicalFocusSolver::compute_hfr(&crop, width, height);
            (hfr.is_finite() && hfr > 0.0).then_some(hfr as f32)
        })
        .collect();

    if hfrs.is_empty() {
        return None;
    }

    hfrs.sort_by(|a, b| a.total_cmp(b));
    Some(hfrs[hfrs.len() / 2])
}

fn extract_crop(
    frame: &LumaFrame,
    center_x: f32,
    center_y: f32,
    crop_size: usize,
) -> (Vec<f32>, usize, usize) {
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

    (crop, crop_width, crop_height)
}

#[cfg(test)]
mod tests {
    use super::{estimate_quality, LumaFrame};
    use crate::vision::background::estimate_background;

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

        assert!(quality.observation_valid);
        assert_eq!(quality.star_count, 1);
        assert!(quality.hfr > 0.0);
        assert!(quality.frame_quality_score > 0.0);
    }
}

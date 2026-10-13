//! Classical trailing metrics for the shared vision bus.

use super::background::BackgroundStats;
use super::quality::LumaFrame;

#[derive(Debug, Clone)]
pub struct TrailingEstimate {
    pub severity: f32,
    pub axis: Option<String>,
    pub direction_pa: Option<f32>,
}

#[derive(Debug, Clone)]
struct Ellipse {
    ellipticity: f32,
    pa_deg: f32,
    flux: f32,
}

pub fn estimate_trailing(frame: &LumaFrame, background: &BackgroundStats) -> TrailingEstimate {
    let mut ellipses = source_ellipses(frame, background);
    if ellipses.is_empty() {
        return TrailingEstimate {
            severity: 0.0,
            axis: None,
            direction_pa: None,
        };
    }

    ellipses.sort_by(|a, b| a.ellipticity.total_cmp(&b.ellipticity));
    let ellipticity = trimmed_mean_ellipticity(&ellipses, 0.2);
    let pa_deg = weighted_mean_pa(&ellipses);
    let severity = if ellipses.len() >= 3 {
        ellipticity.clamp(0.0, 1.0)
    } else {
        (ellipticity * 0.5).clamp(0.0, 1.0)
    };
    let axis = (severity > 0.15).then(|| classify_axis(pa_deg));

    TrailingEstimate {
        severity,
        axis,
        direction_pa: (severity > 0.10).then_some(pa_deg),
    }
}

fn source_ellipses(frame: &LumaFrame, background: &BackgroundStats) -> Vec<Ellipse> {
    let sigma = background.background_sigma.max(1.0);
    let threshold = (background.background_adu + sigma * 3.0).max(background.background_adu + 6.0);
    let mut visited = vec![false; frame.pixels.len()];
    let mut ellipses = Vec::new();

    for idx in 0..frame.pixels.len() {
        let value = frame.pixels[idx];
        if visited[idx]
            || !frame.measurement_pixel_valid(idx)
            || frame.native_saturation_at(idx)
            || !value.is_finite()
            || value < threshold
        {
            continue;
        }

        let mut stack = vec![idx];
        visited[idx] = true;
        let mut points = Vec::new();
        let mut flux = 0.0_f32;
        let mut weighted_x = 0.0_f32;
        let mut weighted_y = 0.0_f32;
        let mut touches_masked_support = false;

        while let Some(current) = stack.pop() {
            let current_value = frame.pixels[current];
            let signal = (current_value - background.background_adu).max(0.0);
            let x = current % frame.width;
            let y = current / frame.width;

            points.push((x as f32, y as f32, signal));
            flux += signal;
            weighted_x += x as f32 * signal;
            weighted_y += y as f32 * signal;

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
                    touches_masked_support |= !frame.measurement_pixel_valid(neighbor)
                        || frame.native_saturation_at(neighbor);
                    if visited[neighbor]
                        || !frame.measurement_pixel_valid(neighbor)
                        || frame.native_saturation_at(neighbor)
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

        if touches_masked_support
            || points.len() < 5
            || points.len() > 5_000
            || flux <= f32::EPSILON
        {
            continue;
        }

        let cx = weighted_x / flux;
        let cy = weighted_y / flux;
        let mut mxx = 0.0_f32;
        let mut myy = 0.0_f32;
        let mut mxy = 0.0_f32;

        for (x, y, signal) in points {
            let dx = x - cx;
            let dy = y - cy;
            mxx += signal * dx * dx;
            myy += signal * dy * dy;
            mxy += signal * dx * dy;
        }

        mxx /= flux;
        myy /= flux;
        mxy /= flux;

        let trace = mxx + myy;
        let det_term = ((mxx - myy).powi(2) + 4.0 * mxy * mxy).sqrt();
        let major = ((trace + det_term) * 0.5).max(0.0).sqrt();
        let minor = ((trace - det_term) * 0.5).max(0.0).sqrt();
        if major <= f32::EPSILON {
            continue;
        }

        let ellipticity = (1.0 - minor / major).clamp(0.0, 1.0);
        let pa_deg = normalize_pa(0.5 * (2.0 * mxy).atan2(mxx - myy).to_degrees());
        ellipses.push(Ellipse {
            ellipticity,
            pa_deg,
            flux,
        });
    }

    ellipses
}

fn trimmed_mean_ellipticity(ellipses: &[Ellipse], trim_fraction: f32) -> f32 {
    if ellipses.is_empty() {
        return 0.0;
    }
    let trim = ((ellipses.len() as f32 * trim_fraction).floor() as usize)
        .min(ellipses.len().saturating_sub(1) / 2);
    let slice = &ellipses[trim..ellipses.len() - trim];
    slice.iter().map(|ellipse| ellipse.ellipticity).sum::<f32>() / slice.len() as f32
}

fn weighted_mean_pa(ellipses: &[Ellipse]) -> f32 {
    let mut x = 0.0_f32;
    let mut y = 0.0_f32;
    let mut weight_sum = 0.0_f32;

    for ellipse in ellipses {
        let weight = ellipse.flux.max(1.0) * ellipse.ellipticity.max(0.01);
        let radians = (ellipse.pa_deg * 2.0).to_radians();
        x += weight * radians.cos();
        y += weight * radians.sin();
        weight_sum += weight;
    }

    if weight_sum <= f32::EPSILON {
        return 0.0;
    }

    normalize_pa(0.5 * y.atan2(x).to_degrees())
}

fn classify_axis(pa_deg: f32) -> String {
    let pa = normalize_pa(pa_deg);
    let horizontal = pa.min((pa - 180.0).abs());
    let vertical = (pa - 90.0).abs();

    if horizontal < 25.0 {
        "RA".to_string()
    } else if vertical < 25.0 {
        "Dec".to_string()
    } else {
        "Ambiguous".to_string()
    }
}

fn normalize_pa(pa_deg: f32) -> f32 {
    pa_deg.rem_euclid(180.0)
}

#[cfg(test)]
mod tests {
    use super::estimate_trailing;
    use crate::vision::background::estimate_background;
    use crate::vision::quality::LumaFrame;

    #[test]
    fn elongated_star_produces_trailing_signal() {
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
            max_adu: 65535.0,
            saturation_adu: Some(65535.0),
            masks: Default::default(),
            analysis_descriptor: None,
        };
        let background = estimate_background(&frame.pixels, frame.saturation_adu);
        let trailing = estimate_trailing(&frame, &background);

        assert!(trailing.severity > 0.15);
        assert_eq!(trailing.axis.as_deref(), Some("RA"));
        assert!(trailing.direction_pa.is_some());
    }
}

//! Classical background and saturation metrics for the shared vision bus.

#[derive(Debug, Clone, Copy)]
pub struct BackgroundStats {
    pub background_adu: f32,
    pub background_sigma: f32,
    pub snr_proxy_median: f32,
    pub saturation_fraction: f32,
}

pub fn estimate_background(pixels: &[f32], max_adu: f32) -> BackgroundStats {
    let mut finite: Vec<f32> = pixels.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return BackgroundStats {
            background_adu: 0.0,
            background_sigma: 0.0,
            snr_proxy_median: 0.0,
            saturation_fraction: 0.0,
        };
    }

    finite.sort_by(|a, b| a.total_cmp(b));
    let median = percentile_sorted(&finite, 0.5);
    let sigma_seed = robust_sigma(&finite, median).max(stddev(&finite, median));
    let clipped: Vec<f32> = if sigma_seed > f32::EPSILON {
        let radius = 3.0 * sigma_seed;
        finite
            .iter()
            .copied()
            .filter(|value| (*value - median).abs() <= radius)
            .collect()
    } else {
        finite.clone()
    };

    let background_adu = if clipped.is_empty() {
        median
    } else {
        let mut clipped_sorted = clipped.clone();
        clipped_sorted.sort_by(|a, b| a.total_cmp(b));
        percentile_sorted(&clipped_sorted, 0.5)
    };
    let background_sigma = if clipped.is_empty() {
        sigma_seed
    } else {
        stddev(&clipped, mean(&clipped))
    };

    let saturation_threshold = saturation_threshold(max_adu);
    let saturated = finite
        .iter()
        .filter(|value| **value >= saturation_threshold)
        .count();
    let snr_proxy_median = snr_proxy_median(pixels, background_adu, background_sigma);

    BackgroundStats {
        background_adu,
        background_sigma,
        snr_proxy_median,
        saturation_fraction: saturated as f32 / finite.len() as f32,
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
    let idx = ((sorted_values.len() - 1) as f32 * quantile.clamp(0.0, 1.0)).round() as usize;
    sorted_values[idx]
}

fn saturation_threshold(max_adu: f32) -> f32 {
    if max_adu <= 0.0 {
        f32::INFINITY
    } else {
        (max_adu * 0.98).max(max_adu - 1.0)
    }
}

fn snr_proxy_median(pixels: &[f32], background_adu: f32, background_sigma: f32) -> f32 {
    let noise = (background_sigma * background_sigma + 9.0).sqrt().max(1.0);
    let mut snr_values: Vec<f32> = pixels
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .filter_map(|value| {
            let signal = value - background_adu;
            (signal > 3.0 * background_sigma.max(1.0)).then_some(signal / noise)
        })
        .collect();

    if snr_values.is_empty() {
        return 0.0;
    }

    snr_values.sort_by(|a, b| a.total_cmp(b));
    percentile_sorted(&snr_values, 0.5)
}

#[cfg(test)]
mod tests {
    use super::estimate_background;

    #[test]
    fn background_uses_clipped_median_and_counts_saturation() {
        let mut pixels = vec![100.0_f32; 100];
        pixels.extend([102.0, 98.0, 65535.0, 65534.0]);

        let stats = estimate_background(&pixels, 65535.0);

        assert!((98.0..=102.0).contains(&stats.background_adu));
        assert!(stats.background_sigma < 10.0);
        assert!(stats.saturation_fraction > 0.0);
    }
}

//! FQE edge preprocessing — must be numerically identical to the training
//! path (`ml/preprocess/normalize.py::robust_normalize`), because the model
//! only ever saw crops normalized exactly that way.
//!
//! Contract (single source of truth = normalize.py):
//!   1. Extract a fixed `FQE_CROP_SIZE` square crop centered on the star.
//!   2. Compute the crop's own 1st/99th percentiles over finite pixels using
//!      NumPy's default *linear-interpolation* percentile (`position =
//!      (n-1) * q/100`, interpolate between order statistics) — NOT
//!      nearest-rank, which differs enough to shift model inputs.
//!   3. `clip((x - lo) / (hi - lo), 0, 1)`; if `hi <= lo`, use `hi = lo + 1`.
//!
//! Training currently normalizes per crop and discards the parameters, so the
//! edge must also normalize per crop (per-rig stored p1/p99 only becomes the
//! contract if training switches to fixed params — keep the two in lockstep).

use ndarray::Array2;

use super::fqe_runner::FQE_CROP_SIZE;
use super::quality::LumaFrame;

/// Extract a `FQE_CROP_SIZE` crop centered on `(cx, cy)` and robust-normalize
/// it to `[0, 1]`. Returns `None` when the frame is smaller than the crop or the
/// star is too close to an edge for a genuinely centered crop. The training
/// builder applies the same rule; shifting an edge star off-center would create
/// an input distribution the model never saw.
pub fn fqe_input_from_frame(frame: &LumaFrame, cx: f32, cy: f32) -> Option<Array2<f32>> {
    if frame.width < FQE_CROP_SIZE
        || frame.height < FQE_CROP_SIZE
        || !cx.is_finite()
        || !cy.is_finite()
    {
        return None;
    }

    let center_x = cx.round() as isize;
    let center_y = cy.round() as isize;
    let half = (FQE_CROP_SIZE / 2) as isize;
    let start_x = center_x - half;
    let start_y = center_y - half;
    if start_x < 0
        || start_y < 0
        || start_x + FQE_CROP_SIZE as isize > frame.width as isize
        || start_y + FQE_CROP_SIZE as isize > frame.height as isize
    {
        return None;
    }
    let start_x = start_x as usize;
    let start_y = start_y as usize;

    let mut crop = Vec::with_capacity(FQE_CROP_SIZE * FQE_CROP_SIZE);
    for y in start_y..start_y + FQE_CROP_SIZE {
        let row = y * frame.width + start_x;
        crop.extend_from_slice(&frame.pixels[row..row + FQE_CROP_SIZE]);
    }

    let normalized = robust_normalize(&crop);
    Array2::from_shape_vec((FQE_CROP_SIZE, FQE_CROP_SIZE), normalized).ok()
}

/// Per-crop percentile clip + scale, numerically matching
/// `robust_normalize(crop)` in `ml/preprocess/normalize.py` (p_lo=1, p_hi=99).
/// Non-finite pixels map to 0 (they cannot occur from u8/u16 camera sources;
/// NumPy would propagate NaN, which must never reach the model).
pub fn robust_normalize(crop: &[f32]) -> Vec<f32> {
    let mut finite: Vec<f32> = crop.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return vec![0.0; crop.len()];
    }
    finite.sort_by(|a, b| a.total_cmp(b));

    let lo = numpy_percentile_sorted(&finite, 1.0);
    let mut hi = numpy_percentile_sorted(&finite, 99.0);
    if hi <= lo {
        hi = lo + 1.0;
    }
    let range = hi - lo;

    crop.iter()
        .map(|v| {
            if v.is_finite() {
                ((v - lo) / range).clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .collect()
}

/// NumPy default (`linear`) percentile on a pre-sorted slice:
/// `pos = (n - 1) * q / 100`, linearly interpolated between order statistics.
fn numpy_percentile_sorted(sorted: &[f32], q: f32) -> f32 {
    debug_assert!(!sorted.is_empty());
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = (sorted.len() - 1) as f64 * (q as f64 / 100.0);
    let lower = pos.floor() as usize;
    let upper = (lower + 1).min(sorted.len() - 1);
    let frac = (pos - lower as f64) as f32;
    sorted[lower] * (1.0 - frac) + sorted[upper] * frac
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_matches_numpy_linear_interpolation() {
        // np.percentile([0,10,...,90], 1)  = 0.9   (pos = 9*0.01 = 0.09)
        // np.percentile([0,10,...,90], 99) = 89.1  (pos = 9*0.99 = 8.91)
        let values: Vec<f32> = (0..10).map(|i| (i * 10) as f32).collect();
        let lo = numpy_percentile_sorted(&values, 1.0);
        let hi = numpy_percentile_sorted(&values, 99.0);
        assert!((lo - 0.9).abs() < 1e-4, "lo={lo}");
        assert!((hi - 89.1).abs() < 1e-4, "hi={hi}");
    }

    #[test]
    fn normalize_clips_to_unit_range_and_handles_flat_crops() {
        let crop = vec![100.0_f32; 16];
        let out = robust_normalize(&crop);
        // hi <= lo -> hi = lo + 1: flat crop maps to exactly 0, not NaN.
        assert!(out.iter().all(|v| *v == 0.0));

        let ramp: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let out = robust_normalize(&ramp);
        assert!(out.iter().all(|v| (0.0..=1.0).contains(v)));
        assert_eq!(out[0], 0.0, "below p1 must clip to 0");
        assert_eq!(*out.last().unwrap(), 1.0, "above p99 must clip to 1");
    }

    #[test]
    fn crop_is_fixed_size_and_centered() {
        let width = 200;
        let height = 200;
        let mut pixels = vec![10.0_f32; width * height];
        pixels[100 * width + 100] = 500.0;
        let frame = LumaFrame {
            pixels,
            width,
            height,
            max_adu: 65_535.0,
        };

        let crop = fqe_input_from_frame(&frame, 100.0, 100.0).unwrap();
        assert_eq!(crop.shape(), &[FQE_CROP_SIZE, FQE_CROP_SIZE]);
        // The bright pixel lands at the crop center and normalizes to 1.0.
        assert_eq!(crop[[FQE_CROP_SIZE / 2, FQE_CROP_SIZE / 2]], 1.0);
    }

    #[test]
    fn crop_rejects_stars_that_cannot_be_centered() {
        let frame = LumaFrame {
            pixels: vec![10.0_f32; 200 * 200],
            width: 200,
            height: 200,
            max_adu: 65_535.0,
        };

        assert!(fqe_input_from_frame(&frame, 31.0, 100.0).is_none());
        assert!(fqe_input_from_frame(&frame, 32.0, 100.0).is_some());
        assert!(fqe_input_from_frame(&frame, 168.0, 100.0).is_some());
        assert!(fqe_input_from_frame(&frame, 169.0, 100.0).is_none());
    }
}

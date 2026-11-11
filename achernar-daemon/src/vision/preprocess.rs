//! FQE edge preprocessing — must be numerically identical to the training
//! path (`ml/preprocess/normalize.py::robust_normalize`), because the model
//! only ever saw crops normalized exactly that way.
//!
//! Contract (single source of truth = normalize.py; the frozen copy
//! `stellar_sim_v1/normalization_source.py` has the same sha256 b15ab052...):
//!   1. Extract a fixed `FQE_CROP_SIZE` square crop centered on the star.
//!   2. Compute the crop's own 1st/99th percentiles over finite pixels using
//!      NumPy's default *linear-interpolation* percentile (`position =
//!      (n-1) * q/100`, interpolate between order statistics) — NOT
//!      nearest-rank, which differs enough to shift model inputs.
//!   3. `clip((x - lo) / (hi - lo), 0, 1)`; if `hi <= lo`, use `hi = lo + 1`.
//!
//! [`robust_normalize`] reproduces NumPy >= 2 bit for bit on float32 crops
//! (NumPy's `_lerp` in the array dtype, `lo`/`hi` widened to Python floats,
//! `hi - lo` in double). Checked against numpy 2.5.4 on 407 crops (integer
//! ADU, RGB-mean thirds, float, log-normal and 1e5-3e7 magnitudes): 0
//! differing bits. The previous `a*(1-t) + b*t` interpolation differed by
//! <= 1.2e-7 on 16-bit-range crops but by up to 0.083 on large-magnitude
//! float crops.
//!
//! Training currently normalizes per crop and discards the parameters, so the
//! edge must also normalize per crop (per-rig stored p1/p99 only becomes the
//! contract if training switches to fixed params — keep the two in lockstep).
//!
//! **Prototype geometry** ([`prototype_inputs`]): the crop origin is
//! `floor(x + 0.5) - 32` with pixel centres at integer coordinates (the
//! daemon's centroid convention), exactly the HST archive importer's
//! `crop = image[cy-32:cy+32, cx-32:cx+32]` with `cx = floor(x + 0.5)`, so the
//! star sits at 32 +/- 0.5 px. The synthetic training source centres stars at
//! 31.5 +/- 0.45 px (contract-v1 known confound: a 0.5 px offset between the
//! two training sources). Crops are never shifted to fit, padded or resized;
//! any masked (bad/invalid/edge/native-saturation) or non-finite pixel
//! rejects the crop (the synthetic source rejected such rows; the HST source
//! imputed DQ pixels with its background map, which the daemon does not).
use ndarray::Array2;

use super::fqe_runner::FQE_CROP_SIZE;
use super::quality::{LumaFrame, UsedStar};

/// Why the FQE is off for every frame unless a prototype runs: no bundle is
/// bound to the versioned linear analysis domain (A03).
pub const FQE_A03_DISABLED_REASON: &str =
    "FQE disabled: no model bundle is qualified for the A03 analysis domain/version";

/// Extract a `FQE_CROP_SIZE` crop centered on `(cx, cy)` and robust-normalize
/// it to `[0, 1]`. Returns `None` when the frame is smaller than the crop or the
/// star is too close to an edge for a genuinely centered crop. The training
/// builder applies the same rule; shifting an edge star off-center would create
/// an input distribution the model never saw.
pub fn fqe_input_from_frame(frame: &LumaFrame, cx: f32, cy: f32) -> Option<Array2<f32>> {
    // A03 changes the input domain. Existing model exports have no qualification
    // binding for the versioned linear intensity, masks or geometry, including
    // shadow inference. Shape compatibility alone cannot enable this path.
    if fqe_unavailable_reason(frame).is_some() {
        return None;
    }
    centered_normalized_crop(frame, cx, cy)
}

pub fn fqe_unavailable_reason(_frame: &LumaFrame) -> Option<&'static str> {
    Some(FQE_A03_DISABLED_REASON)
}

/// The prototype's inputs for one frame: normalized crops of the accepted
/// stars, plus what happened to the stars that gave none.
#[derive(Debug, Default)]
pub struct PrototypeInputs {
    /// One `FQE_CROP_SIZE` crop per fed star, in `used_stars` order.
    pub crops: Vec<Array2<f32>>,
    /// Accepted stars offered (at most the classical used-star cap).
    pub offered: usize,
    /// No centred crop fits inside the frame.
    pub skipped_edge: usize,
    /// A masked (bad/invalid/edge/native-saturation) pixel in the crop.
    pub skipped_masked: usize,
    /// A non-finite pixel in the crop.
    pub skipped_non_finite: usize,
}

/// `None` when the frame is in the versioned linear analysis domain the
/// prototype runs on (an analysis descriptor of the current
/// [`crate::processing::debayer::PROCESSING_VERSION`]); otherwise why not.
pub fn prototype_domain_reason(frame: &LumaFrame) -> Option<String> {
    match &frame.analysis_descriptor {
        Some(descriptor)
            if descriptor.processing_version == crate::processing::debayer::PROCESSING_VERSION =>
        {
            None
        }
        Some(descriptor) => Some(format!(
            "FQE prototype: analysis version {} is not {}",
            descriptor.processing_version,
            crate::processing::debayer::PROCESSING_VERSION
        )),
        None => Some("FQE prototype: frame has no versioned analysis descriptor".to_string()),
    }
}

/// Crops for the prototype FQE from the classical measurement's accepted
/// stars only (`QualityMetrics::used_stars`: every validity gate passed and a
/// finite HFR, capped at the classical used-star limit, brightest first).
/// Candidates the gates rejected are never fed. See the module docs for the
/// geometry and mask policy.
pub fn prototype_inputs(frame: &LumaFrame, stars: &[UsedStar]) -> PrototypeInputs {
    let mut inputs = PrototypeInputs {
        offered: stars.len(),
        ..Default::default()
    };
    for star in stars {
        match prototype_crop(frame, star.x, star.y) {
            Ok(crop) => inputs.crops.push(crop),
            Err(CropSkip::Edge) => inputs.skipped_edge += 1,
            Err(CropSkip::Masked) => inputs.skipped_masked += 1,
            Err(CropSkip::NonFinite) => inputs.skipped_non_finite += 1,
        }
    }
    inputs
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CropSkip {
    Edge,
    Masked,
    NonFinite,
}

fn prototype_crop(frame: &LumaFrame, x: f32, y: f32) -> Result<Array2<f32>, CropSkip> {
    if !x.is_finite()
        || !y.is_finite()
        || frame.width < FQE_CROP_SIZE
        || frame.height < FQE_CROP_SIZE
    {
        return Err(CropSkip::Edge);
    }
    let half = (FQE_CROP_SIZE / 2) as i64;
    // floor(x + 0.5) in double, as the archive importer computes it.
    let start_x = (f64::from(x) + 0.5).floor() as i64 - half;
    let start_y = (f64::from(y) + 0.5).floor() as i64 - half;
    if start_x < 0
        || start_y < 0
        || start_x + FQE_CROP_SIZE as i64 > frame.width as i64
        || start_y + FQE_CROP_SIZE as i64 > frame.height as i64
    {
        return Err(CropSkip::Edge);
    }
    let (start_x, start_y) = (start_x as usize, start_y as usize);
    let mut crop = Vec::with_capacity(FQE_CROP_SIZE * FQE_CROP_SIZE);
    let mut masked = false;
    for row in start_y..start_y + FQE_CROP_SIZE {
        for index in row * frame.width + start_x..row * frame.width + start_x + FQE_CROP_SIZE {
            masked |= !frame.measurement_pixel_valid(index) || frame.native_saturation_at(index);
            crop.push(frame.pixels[index]);
        }
    }
    if masked {
        return Err(CropSkip::Masked);
    }
    if crop.iter().any(|value| !value.is_finite()) {
        return Err(CropSkip::NonFinite);
    }
    Array2::from_shape_vec((FQE_CROP_SIZE, FQE_CROP_SIZE), robust_normalize(&crop))
        .map_err(|_| CropSkip::Edge)
}

fn centered_normalized_crop(frame: &LumaFrame, cx: f32, cy: f32) -> Option<Array2<f32>> {
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

/// Per-crop percentile clip + scale, bit-identical to
/// `robust_normalize(crop)` in `ml/preprocess/normalize.py` (p_lo=1, p_hi=99)
/// under NumPy >= 2 for float32 input:
///
/// * `lo`/`hi` are NumPy's float32 percentiles ([`numpy_percentile_sorted`]),
///   then widened (`float(...)`); a flat crop gets `hi = lo + 1` in double;
/// * `(img - lo)` subtracts `lo` rounded to float32, and `hi - lo` is formed
///   in double then rounded to float32 (NEP 50 weak Python scalars);
/// * the float32 quotient is clipped to `[0, 1]`.
///
/// Non-finite pixels map to 0 (NumPy's `out[~finite] = 0`). A NaN quotient
/// (impossible unless `hi - lo` underflows float32) also maps to 0 rather
/// than reaching the model.
pub fn robust_normalize(crop: &[f32]) -> Vec<f32> {
    let mut finite: Vec<f32> = crop.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return vec![0.0; crop.len()];
    }
    finite.sort_by(|a, b| a.total_cmp(b));

    let lo = f64::from(numpy_percentile_sorted(&finite, 1.0));
    let mut hi = f64::from(numpy_percentile_sorted(&finite, 99.0));
    if hi <= lo {
        hi = lo + 1.0;
    }
    let lo32 = lo as f32;
    let range32 = (hi - lo) as f32;

    crop.iter()
        .map(|v| {
            if v.is_finite() {
                let z = ((v - lo32) / range32).clamp(0.0, 1.0);
                if z.is_nan() {
                    0.0
                } else {
                    z
                }
            } else {
                0.0
            }
        })
        .collect()
}

/// NumPy >= 2 `np.percentile(a_float32, q)` (method "linear", Python-float
/// `q`) on a pre-sorted slice: virtual index `(n - 1) * (q / 100)` in double,
/// `gamma` its fractional part, then NumPy's `_lerp` in float32 (the
/// Python-float `gamma` is a weak scalar):
/// `a + (b - a) * gamma` below 0.5, `b - (b - a) * (1 - gamma)` from 0.5.
fn numpy_percentile_sorted(sorted: &[f32], q: f64) -> f32 {
    debug_assert!(!sorted.is_empty());
    let n = sorted.len();
    let virtual_index = (n - 1) as f64 * (q / 100.0);
    let previous = virtual_index.floor();
    let gamma = virtual_index - previous;
    let lower = (previous as usize).min(n - 1);
    let upper = (lower + 1).min(n - 1);
    let (a, b) = (sorted[lower], sorted[upper]);
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * ((1.0 - gamma) as f32)
    } else {
        a + diff * (gamma as f32)
    }
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
            saturation_adu: Some(65_535.0),
            masks: Default::default(),
            analysis_descriptor: None,
        };

        let crop = centered_normalized_crop(&frame, 100.0, 100.0).unwrap();
        assert_eq!(crop.shape(), &[FQE_CROP_SIZE, FQE_CROP_SIZE]);
        // The bright pixel lands at the crop center and normalizes to 1.0.
        assert_eq!(crop[[FQE_CROP_SIZE / 2, FQE_CROP_SIZE / 2]], 1.0);
    }

    #[test]
    fn unqualified_analysis_cannot_reach_even_shadow_preprocessing() {
        let frame = LumaFrame {
            pixels: vec![10.0; 200 * 200],
            width: 200,
            height: 200,
            max_adu: 65535.0,
            saturation_adu: Some(16383.0),
            masks: Default::default(),
            analysis_descriptor: None,
        };
        assert!(fqe_input_from_frame(&frame, 100.0, 100.0).is_none());
        assert!(fqe_unavailable_reason(&frame).unwrap().contains("A03"));
    }

    /// The crops `achernar-daemon/tests/fixtures/fqe_norm_parity_gen.py`
    /// builds with the same integer LCG: integer ADU, RGB-mean thirds, a
    /// noise-only crop, an exactly flat crop and a 5e4-ADU sky. Keep the two
    /// in step; the script prints the expected values of both parity tests
    /// below and `--check` compares them with this file.
    fn shared_fixture_crop(k: i64) -> Vec<f32> {
        let mut state = 0x5EED_u64 + k as u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state >> 33
        };
        let sky: i64 = if k == 6 { 50_000 } else { 100 + 37 * k };
        let amp: i64 = match k {
            3 | 4 => 0,
            6 => 15_000,
            _ => 400 * (k + 1),
        };
        let noise_mod: u64 = if k == 4 { 1 } else { 41 };
        let (cx, cy) = (32 + (k % 3) - 1, 32 - (k % 2));
        let width2 = 2 + 3 * k;
        let mut crop = Vec::with_capacity(FQE_CROP_SIZE * FQE_CROP_SIZE);
        for y in 0..FQE_CROP_SIZE as i64 {
            for x in 0..FQE_CROP_SIZE as i64 {
                let r2 = (x - cx).pow(2) + (y - cy).pow(2);
                let star = (amp * width2) / (width2 + r2);
                crop.push((sky + (next() % noise_mod) as i64 + star) as f32);
            }
        }
        if k % 2 == 1 {
            crop.iter_mut().for_each(|value| *value /= 3.0);
        }
        crop
    }

    fn sha256_f32_le(values: &[f32]) -> String {
        use sha2::{Digest, Sha256};
        let mut digest = Sha256::new();
        for value in values {
            digest.update(value.to_le_bytes());
        }
        format!("{:x}", digest.finalize())
    }

    /// Bit parity with the frozen training normalization: sha256 of the
    /// float32 little-endian output of `ml/preprocess/normalize.py::
    /// robust_normalize` (numpy 2.5.4) on the same crops, as printed by
    /// `achernar-daemon/tests/fixtures/fqe_norm_parity_gen.py`.
    #[test]
    fn normalization_is_bit_identical_to_the_training_normalization() {
        let expected = [
            (
                "4405bc1437784cddf02a7931d4a43c8a926c49f2e48c8d3b51f2a36203e0d7dc",
                "b2b12b11bb9262976bf2d495b200ac07b665f2ece97b129e84af35afbde7c77c",
            ),
            (
                "fd03c3a96347d624c09776a473109cf969bf78f3a3e11924c42b6fe66cd1debc",
                "4c006479c0bbe19603c51774886a46b80b5d0a118d9b939491c037641c33d997",
            ),
            (
                "6626688b2807edc250847a65c14f7f59fc16a95b6ab3a15e433b2aabcd2d9973",
                "cdc6f2e92f19da70e7aff55b3100c030ee9d2a4e5a828074214d1a296460fbca",
            ),
            (
                "e0dbf817cdd9d7b440c826c350bdaaf2426e7baa165f3b0870d02554ce305e31",
                "49d42c33eaa2a3bd71025a980f735ab316be6e222df66b698ff39a7adcdf2510",
            ),
            (
                "2c99dbdf7e586b9760c88672df9b2a0b05f0b4765af6675ee53c508decd77873",
                "4fe7b59af6de3b665b67788cc2f99892ab827efae3a467342b3bb4e3bc8e5bfe",
            ),
            (
                "580e39cd5dab7345d4b063de93b6c4c629bf066f1a779eb70c51894b10363012",
                "d8607b6d637ed4ea353d91fda05e400c46eaa6ddfe4bad54141401293c037b59",
            ),
            (
                "b12220d397d64ad771020eee2bea200bb1e16b44376eaaabc1b78d5e4003159d",
                "ceb4696ae2689bd6dc1ad44410c5e420636b7058aaad65d56dcbaf6577325e8e",
            ),
        ];
        for (k, (raw_sha, norm_sha)) in expected.iter().enumerate() {
            let crop = shared_fixture_crop(k as i64);
            assert_eq!(
                &sha256_f32_le(&crop),
                raw_sha,
                "fixture crop {k} generation"
            );
            assert_eq!(
                &sha256_f32_le(&robust_normalize(&crop)),
                norm_sha,
                "crop {k}: normalization differs from numpy"
            );
        }
    }

    /// Normalization plus inference end to end: the daemon's runner on the
    /// shared crops against Python onnxruntime 1.31.0 (CPU provider) running
    /// the same checked-in random-weight graph on `robust_normalize` output
    /// (`achernar-daemon/tests/fixtures/fqe_norm_parity_gen.py`, default
    /// `--graph`). The F-006 torch/ONNX tolerance is 1e-4.
    #[test]
    fn reference_graph_inference_matches_python_onnxruntime() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../models/focus_quality/1.0.0/model.onnx");
        let runner =
            crate::vision::fqe_runner::FqeRunner::new(Some(path.to_string_lossy().into_owned()))
                .unwrap();
        let crops: Vec<Array2<f32>> = (0..7)
            .map(|k| {
                Array2::from_shape_vec(
                    (FQE_CROP_SIZE, FQE_CROP_SIZE),
                    robust_normalize(&shared_fixture_crop(k)),
                )
                .unwrap()
            })
            .collect();
        let python: [(f64, f64); 7] = [
            (-0.21554423868656158, -0.49063754081726074),
            (-0.03157650679349899, -0.27696534991264343),
            (0.007628770545125008, -0.23740142583847046),
            (-0.9684083461761475, -0.6424286365509033),
            (0.0, 0.0),
            (0.02684958092868328, -0.2364780753850937),
            (-0.02996467985212803, -0.2342822551727295),
        ];
        let outputs = runner.predict_batch(&crops).unwrap();
        for (k, (output, (mu, log_var))) in outputs.iter().zip(python).enumerate() {
            assert!((output.mu_log_hfr - mu).abs() <= 1e-5, "crop {k}: mu");
            // No calibration: sigma = exp(0.5 * log_var).
            let sigma = (0.5 * log_var).exp();
            assert!(
                (output.sigma_log_hfr - sigma).abs() <= 1e-5,
                "crop {k}: sigma"
            );
        }
    }

    fn versioned_frame(width: usize, height: usize, pixels: Vec<f32>) -> LumaFrame {
        let descriptor = crate::processing::debayer::SensorDescriptor {
            schema_version: 1,
            width,
            height,
            row_stride_bytes: width * 2,
            sample_type: crate::processing::debayer::SampleType::U16,
            byte_order: crate::processing::debayer::ByteOrder::Little,
            color: crate::processing::debayer::SensorColor::Mono,
            domain: crate::processing::debayer::SampleDomain::NativeAdu,
            geometry: crate::processing::debayer::PixelGeometry::identity("test"),
            saturation: None,
            calibration_ids: vec![],
        };
        let data: Vec<u8> = pixels
            .iter()
            .flat_map(|value| (value.round().clamp(0.0, 65_535.0) as u16).to_le_bytes())
            .collect();
        let analysis =
            crate::processing::debayer::analyze(&crate::processing::debayer::SensorFrame {
                descriptor: &descriptor,
                data: &data,
                masks: Default::default(),
            })
            .unwrap();
        crate::vision::quality::luma_from_analysis(analysis)
    }

    #[test]
    fn prototype_crops_are_centred_like_the_archive_importer() {
        let (width, height) = (160, 120);
        let mut pixels = vec![100.0_f32; width * height];
        pixels[60 * width + 80] = 5_000.0;
        let frame = versioned_frame(width, height, pixels);
        assert!(prototype_domain_reason(&frame).is_none());
        // floor(x + 0.5): 80.49 -> 80, 80.5 -> 81; the bright pixel lands at
        // index 32 or 31 of the crop accordingly.
        for (x, column) in [(80.49_f32, 32), (79.5, 32), (80.5, 31)] {
            let inputs = prototype_inputs(
                &frame,
                &[UsedStar {
                    x,
                    y: 60.0,
                    hfr_px: 1.0,
                }],
            );
            assert_eq!(inputs.crops.len(), 1);
            assert_eq!(inputs.crops[0][[32, column]], 1.0, "x={x}");
        }
    }

    #[test]
    fn prototype_feeds_accepted_stars_and_skips_edge_masked_and_non_finite_crops() {
        let (width, height) = (200, 120);
        let mut frame = versioned_frame(width, height, vec![100.0_f32; width * height]);
        let star = |x: f32, y: f32| UsedStar { x, y, hfr_px: 1.5 };
        // A bad pixel inside the second star's crop, a NaN inside the third.
        frame.masks.bad_pixel[60 * width + 100] = 1;
        frame.pixels[60 * width + 160] = f32::NAN;
        let inputs = prototype_inputs(
            &frame,
            &[
                star(40.0, 60.0),
                star(100.0, 60.0),
                star(160.0, 60.0),
                star(31.4, 60.0),
                star(40.0, 88.6),
            ],
        );
        assert_eq!(inputs.offered, 5);
        assert_eq!(inputs.crops.len(), 1);
        assert_eq!(inputs.skipped_masked, 1);
        assert_eq!(inputs.skipped_non_finite, 1);
        assert_eq!(inputs.skipped_edge, 2);
        assert!(inputs
            .crops
            .iter()
            .all(|crop| crop.shape() == [FQE_CROP_SIZE, FQE_CROP_SIZE]));
    }

    #[test]
    fn prototype_needs_the_versioned_analysis_domain() {
        let frame = LumaFrame {
            pixels: vec![10.0; 200 * 200],
            width: 200,
            height: 200,
            max_adu: 65535.0,
            saturation_adu: None,
            masks: Default::default(),
            analysis_descriptor: None,
        };
        assert!(prototype_domain_reason(&frame)
            .unwrap()
            .contains("descriptor"));
        // The non-prototype reason is unchanged.
        assert_eq!(
            fqe_unavailable_reason(&frame),
            Some(FQE_A03_DISABLED_REASON)
        );
    }

    #[test]
    fn crop_rejects_stars_that_cannot_be_centered() {
        let frame = LumaFrame {
            pixels: vec![10.0_f32; 200 * 200],
            width: 200,
            height: 200,
            max_adu: 65_535.0,
            saturation_adu: Some(65_535.0),
            masks: Default::default(),
            analysis_descriptor: None,
        };

        assert!(centered_normalized_crop(&frame, 31.0, 100.0).is_none());
        assert!(centered_normalized_crop(&frame, 32.0, 100.0).is_some());
        assert!(centered_normalized_crop(&frame, 168.0, 100.0).is_some());
        assert!(centered_normalized_crop(&frame, 169.0, 100.0).is_none());
    }
}

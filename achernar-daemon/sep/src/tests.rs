//! Unit tests of the safe API against the vendored C library.

use super::*;

/// Deterministic N(0,1) noise (xorshift64* + Box-Muller) so the tests need no
/// dependencies.
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn noisy_sky(width: usize, height: usize, level: f32, sigma: f32, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..width * height)
        .map(|_| level + sigma * rng.normal() as f32)
        .collect()
}

fn add_gaussian(image: &mut [f32], width: usize, x0: f64, y0: f64, sigma: f64, peak: f64) {
    let height = image.len() / width;
    for y in 0..height {
        for x in 0..width {
            let r2 = (x as f64 - x0).powi(2) + (y as f64 - y0).powi(2);
            image[y * width + x] += (peak * (-r2 / (2.0 * sigma * sigma)).exp()) as f32;
        }
    }
}

/// Background-subtracts with SEP itself and extracts with the RMS map, CONV
/// filter, as the daemon adapter does.
fn extract_with_maps(
    image: &[f32],
    width: usize,
    height: usize,
    params: &ExtractParams,
) -> Catalog {
    let background = Background::estimate(
        image,
        width,
        height,
        None,
        &BackgroundParams {
            bw: 32,
            bh: 32,
            ..BackgroundParams::default()
        },
    )
    .unwrap();
    let back = background.back_array().unwrap();
    let rms = background.rms_array().unwrap();
    let residual: Vec<f32> = image.iter().zip(&back).map(|(i, b)| i - b).collect();
    extract(&residual, width, height, None, Noise::Array(&rms), params).unwrap()
}

fn conv_params(thresh: f32) -> ExtractParams {
    ExtractParams {
        filter_type: FilterType::Conv,
        ..ExtractParams::new(thresh)
    }
}

#[test]
fn compiled_version_string_is_the_vendored_release() {
    assert_eq!(library_version(), SEP_VERSION);
    assert!(BUILD_FLAGS.contains("SEP_VERSION_STRING"));
}

#[test]
fn shape_errors_are_typed_and_precede_any_call() {
    let params = BackgroundParams::default();
    for (len, width, height) in [(0, 0, 4), (4, 4, 0), (15, 4, 4), (17, 4, 4)] {
        let data = vec![0.0; len];
        assert!(matches!(
            Background::estimate(&data, width, height, None, &params),
            Err(SepError::InvalidShape { .. })
        ));
        assert!(matches!(
            extract(&data, width, height, None, Noise::None, &conv_params(1.0)),
            Err(SepError::InvalidShape { .. })
        ));
    }
    let wide = vec![0.0; 2];
    assert!(matches!(
        Background::estimate(&wide, MAX_DIMENSION + 1, 1, None, &params),
        Err(SepError::InvalidShape { .. })
    ));
    assert!(matches!(
        Background::estimate(&wide, MAX_DIMENSION, MAX_DIMENSION, None, &params),
        Err(SepError::FrameTooLarge { .. })
    ));
}

#[test]
fn mask_and_noise_lengths_are_checked() {
    let data = vec![1.0; 16];
    assert_eq!(
        Background::estimate(&data, 4, 4, Some(&[0; 15]), &BackgroundParams::default())
            .unwrap_err(),
        SepError::MaskLengthMismatch {
            expected: 16,
            actual: 15
        }
    );
    assert_eq!(
        extract(
            &data,
            4,
            4,
            None,
            Noise::Array(&[1.0; 17]),
            &conv_params(1.0)
        )
        .unwrap_err(),
        SepError::NoiseLengthMismatch {
            expected: 16,
            actual: 17
        }
    );
}

#[test]
fn kernel_and_parameter_validation() {
    assert!(Kernel::new(vec![1.0; 4], 2, 2).is_err(), "even kernel");
    assert!(Kernel::new(vec![1.0; 8], 3, 3).is_err(), "length");
    assert!(Kernel::new(vec![0.0; 9], 3, 3).is_err(), "zero sum");
    assert!(Kernel::new(vec![f32::NAN; 9], 3, 3).is_err(), "NaN");
    assert!(Kernel::from_rows(&[vec![1.0, 2.0, 1.0], vec![1.0]]).is_err());
    assert!(Kernel::from_rows(&[]).is_err());
    assert_eq!(
        Kernel::from_rows(&[
            vec![1.0, 2.0, 1.0],
            vec![2.0, 4.0, 2.0],
            vec![1.0, 2.0, 1.0]
        ])
        .unwrap(),
        Kernel::sep_default()
    );

    let data = vec![1.0; 64];
    for params in [
        BackgroundParams {
            bw: 0,
            ..Default::default()
        },
        BackgroundParams {
            fh: 0,
            ..Default::default()
        },
        BackgroundParams {
            fw: 4,
            ..Default::default()
        },
        BackgroundParams {
            fthresh: f64::NAN,
            ..Default::default()
        },
    ] {
        assert!(matches!(
            Background::estimate(&data, 8, 8, None, &params),
            Err(SepError::InvalidParameter { .. })
        ));
    }
    for params in [
        ExtractParams {
            minarea: 0,
            ..conv_params(1.0)
        },
        ExtractParams {
            deblend_nthresh: 0,
            ..conv_params(1.0)
        },
        ExtractParams {
            deblend_cont: -0.1,
            ..conv_params(1.0)
        },
        ExtractParams {
            clean_param: 0.0,
            ..conv_params(1.0)
        },
        ExtractParams {
            pixstack: 0,
            ..conv_params(1.0)
        },
        ExtractParams {
            sub_object_limit: MAX_SUB_OBJECT_LIMIT + 1,
            ..conv_params(1.0)
        },
        conv_params(0.0),
    ] {
        assert!(matches!(
            extract(&data, 8, 8, None, Noise::Scalar(1.0), &params),
            Err(SepError::InvalidParameter { .. })
        ));
    }
    assert!(matches!(
        extract(&data, 8, 8, None, Noise::Scalar(0.0), &conv_params(1.0)),
        Err(SepError::InvalidNoise { .. })
    ));
}

#[test]
fn unmasked_non_finite_pixels_are_rejected_and_masked_ones_tolerated() {
    let (width, height) = (48, 40);
    let mut data = noisy_sky(width, height, 100.0, 3.0, 7);
    data[5 * width + 9] = f32::NAN;
    assert!(matches!(
        Background::estimate(&data, width, height, None, &BackgroundParams::default()),
        Err(SepError::NonFinitePixel { x: 9, y: 5, value }) if value.is_nan()
    ));
    let mut mask = vec![0u8; width * height];
    mask[5 * width + 9] = 1;
    data[6 * width + 9] = f32::INFINITY;
    mask[6 * width + 9] = 1;
    let background = Background::estimate(
        &data,
        width,
        height,
        Some(&mask),
        &BackgroundParams::default(),
    )
    .unwrap();
    assert!(background.global_level().is_finite());
    assert!((background.global_level() - 100.0).abs() < 1.0);

    let residual: Vec<f32> = data.iter().map(|v| v - 100.0).collect();
    assert!(matches!(
        extract(
            &residual,
            width,
            height,
            None,
            Noise::Scalar(3.0),
            &conv_params(5.0)
        ),
        Err(SepError::NonFinitePixel { x: 9, y: 5, .. })
    ));
    extract(
        &residual,
        width,
        height,
        Some(&mask),
        Noise::Scalar(3.0),
        &conv_params(5.0),
    )
    .unwrap();

    let mut rms = vec![3.0f32; width * height];
    rms[100] = 0.0;
    assert!(matches!(
        extract(
            &residual,
            width,
            height,
            Some(&mask),
            Noise::Array(&rms),
            &conv_params(5.0)
        ),
        Err(SepError::InvalidNoise {
            x: Some(4),
            y: Some(2),
            ..
        })
    ));
}

#[test]
fn constant_image_has_zero_mesh_rms_and_seps_fabricated_global_rms() {
    let (width, height) = (96, 80);
    let data = vec![250.0f32; width * height];
    let background = Background::estimate(
        &data,
        width,
        height,
        None,
        &BackgroundParams {
            bw: 32,
            bh: 32,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(background.global_level(), 250.0);
    assert!(background.mesh_rms().iter().all(|&s| s == 0.0));
    assert!(background.mesh_level().iter().all(|&b| b == 250.0));
    assert!(background.rms_array().unwrap().iter().all(|&r| r == 0.0));
    assert!(background.back_array().unwrap().iter().all(|&b| b == 250.0));
    // filterback: median sigma 0 -> no positive sigma -> globalrms = 1.0.
    assert_eq!(background.global_rms(), 1.0);
}

#[test]
fn frame_smaller_than_one_mesh_is_a_single_node() {
    let (width, height) = (20, 15);
    let data = noisy_sky(width, height, 40.0, 2.0, 3);
    let background =
        Background::estimate(&data, width, height, None, &BackgroundParams::default()).unwrap();
    assert_eq!(
        background.mesh_grid(),
        MeshGrid {
            nx: 1,
            ny: 1,
            bw: 64,
            bh: 64
        }
    );
    let back = background.back_array().unwrap();
    assert!(back.iter().all(|&b| b == background.mesh_level()[0]));
    assert!((background.global_level() - 40.0).abs() < 1.0);
}

#[test]
fn line_evaluators_match_the_array_evaluators() {
    let (width, height) = (130, 70);
    let mut data = noisy_sky(width, height, 500.0, 4.0, 11);
    for (i, v) in data.iter_mut().enumerate() {
        *v += 0.3 * (i % width) as f32;
    }
    let background = Background::estimate(
        &data,
        width,
        height,
        None,
        &BackgroundParams {
            bw: 32,
            bh: 32,
            ..Default::default()
        },
    )
    .unwrap();
    let back = background.back_array().unwrap();
    let rms = background.rms_array().unwrap();
    let mut line = vec![0.0; width];
    for y in [0, 17, height - 1] {
        background.back_line(y, &mut line).unwrap();
        assert_eq!(line, back[y * width..(y + 1) * width]);
        background.rms_line(y, &mut line).unwrap();
        assert_eq!(line, rms[y * width..(y + 1) * width]);
    }
    assert!(background.back_line(height, &mut line).is_err());
    assert!(background.back_line(0, &mut line[1..]).is_err());
}

#[test]
fn masked_region_is_excluded_from_the_background() {
    let (width, height) = (128, 128);
    let mut data = noisy_sky(width, height, 100.0, 2.0, 5);
    let mut mask = vec![0u8; width * height];
    // A bright 40x40 patch covering most of one 32x32 mesh.
    for y in 40..80 {
        for x in 40..80 {
            data[y * width + x] += 5000.0;
            mask[y * width + x] = 1;
        }
    }
    let params = BackgroundParams {
        bw: 32,
        bh: 32,
        fw: 1,
        fh: 1,
        fthresh: 0.0,
    };
    let unmasked = Background::estimate(&data, width, height, None, &params).unwrap();
    let masked = Background::estimate(&data, width, height, Some(&mask), &params).unwrap();
    // Mesh (1,1) covers x,y in 32..64: 24x24 of its 1024 pixels are bright.
    let mesh = width / 32 + 1;
    assert!(unmasked.mesh_level()[mesh] > 150.0);
    assert!((masked.mesh_level()[mesh] - 100.0).abs() < 2.0);
    // Mesh (2,2) is 75% masked: below SEP's 50% support, interpolated from
    // its nearest good meshes, so still near the sky.
    let bad = 2 * (width / 32) + 2;
    assert!((masked.mesh_level()[bad] - 100.0).abs() < 2.0);
    let back = masked.back_array().unwrap();
    assert!(back.iter().all(|b| (b - 100.0).abs() < 3.0));
}

#[test]
fn all_masked_frame_gets_seps_placeholder_model() {
    // filterback with no good mesh: back 0, sigma 1, RETURN_OK. The safe API
    // reports what SEP returned; support checks belong to the caller.
    let data = vec![10.0f32; 64 * 64];
    let mask = vec![1u8; 64 * 64];
    let background = Background::estimate(
        &data,
        64,
        64,
        Some(&mask),
        &BackgroundParams {
            bw: 32,
            bh: 32,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(background.mesh_level().iter().all(|&b| b == 0.0));
    assert!(background.mesh_rms().iter().all(|&s| s == 1.0));
}

#[test]
fn synthetic_star_centroid_is_recovered() {
    let (width, height) = (96, 96);
    let mut image = noisy_sky(width, height, 200.0, 2.0, 21);
    add_gaussian(&mut image, width, 40.3, 51.7, 1.8, 400.0);
    let catalog = extract_with_maps(&image, width, height, &conv_params(5.0));
    assert_eq!(catalog.objects.len(), 1, "{:?}", catalog.objects);
    let star = &catalog.objects[0];
    assert!((star.x - 40.3).abs() < 0.05, "x {}", star.x);
    assert!((star.y - 51.7).abs() < 0.05, "y {}", star.y);
    assert_eq!((star.xpeak, star.ypeak), (40, 52));
    assert_eq!(star.flag, 0);
    assert_eq!(star.pixels.len(), star.npix);
    assert_eq!(catalog.filter_applied, Some(FilterType::Conv));
    assert!(catalog.thresh_relative);
    // SEP's flux is the unfiltered footprint sum; the full Gaussian integral
    // is 2*pi*sigma^2*peak ~ 8143, the isophotal footprint holds most of it.
    assert!(
        star.flux > 7000.0 && star.flux < 8300.0,
        "flux {}",
        star.flux
    );
}

#[test]
fn close_double_is_deblended_and_flagged_merged() {
    let (width, height) = (96, 96);
    let mut image = noisy_sky(width, height, 200.0, 2.0, 22);
    add_gaussian(&mut image, width, 45.0, 48.0, 1.6, 600.0);
    add_gaussian(&mut image, width, 51.0, 48.0, 1.6, 450.0);
    let mut params = conv_params(5.0);
    params.segmentation_map = true;
    let catalog = extract_with_maps(&image, width, height, &params);
    assert_eq!(catalog.objects.len(), 2, "{:?}", catalog.objects);
    for object in &catalog.objects {
        assert!(object.has_flag(flags::OBJ_MERGED));
        assert!(!object.has_flag(flags::OBJ_TRUNC));
    }
    let mut xs: Vec<f64> = catalog.objects.iter().map(|o| o.x).collect();
    xs.sort_by(f64::total_cmp);
    assert!(
        (xs[0] - 45.0).abs() < 0.6 && (xs[1] - 51.0).abs() < 0.6,
        "{xs:?}"
    );
    let segmap = catalog.segmentation.unwrap();
    for (i, object) in catalog.objects.iter().enumerate() {
        assert!(object
            .pixels
            .iter()
            .all(|&p| segmap[p as usize] == i as i32 + 1));
    }
    assert_eq!(
        segmap.iter().filter(|&&id| id != 0).count(),
        catalog.objects.iter().map(|o| o.npix).sum::<usize>()
    );

    // deblend_cont = 1 disables deblending: one merged-into-one object.
    params.deblend_cont = 1.0;
    let single = extract_with_maps(&image, width, height, &params);
    assert_eq!(single.objects.len(), 1);
    assert!(!single.objects[0].has_flag(flags::OBJ_MERGED));
}

#[test]
fn border_object_is_flagged_truncated() {
    let (width, height) = (80, 64);
    let mut image = noisy_sky(width, height, 200.0, 2.0, 23);
    add_gaussian(&mut image, width, 0.5, 30.0, 1.8, 500.0);
    let catalog = extract_with_maps(&image, width, height, &conv_params(5.0));
    assert_eq!(catalog.objects.len(), 1);
    assert!(catalog.objects[0].has_flag(flags::OBJ_TRUNC));
    assert_eq!(catalog.objects[0].xmin, 0);
}

#[test]
fn matched_filter_falls_back_to_conv_with_scalar_noise() {
    let (width, height) = (64, 64);
    let mut image = noisy_sky(width, height, 0.0, 2.0, 24);
    add_gaussian(&mut image, width, 30.0, 30.0, 1.8, 300.0);
    let params = ExtractParams::new(5.0);
    let scalar = extract(&image, width, height, None, Noise::Scalar(2.0), &params).unwrap();
    assert_eq!(scalar.filter_applied, Some(FilterType::Conv));
    let rms = vec![2.0f32; width * height];
    let array = extract(&image, width, height, None, Noise::Array(&rms), &params).unwrap();
    assert_eq!(array.filter_applied, Some(FilterType::Matched));
    let unfiltered = ExtractParams {
        filter: None,
        ..params
    };
    let none = extract(&image, width, height, None, Noise::None, &unfiltered).unwrap();
    assert_eq!(none.filter_applied, None);
    assert!(!none.thresh_relative);
}

#[test]
fn pixel_stack_overflow_is_a_library_error_and_does_not_persist() {
    let (width, height) = (64, 64);
    let mut image = noisy_sky(width, height, 0.0, 1.0, 25);
    add_gaussian(&mut image, width, 32.0, 32.0, 4.0, 200.0);
    let small = ExtractParams {
        pixstack: 10,
        ..conv_params(3.0)
    };
    match extract(&image, width, height, None, Noise::Scalar(1.0), &small) {
        Err(SepError::Library {
            code,
            message,
            detail,
        }) => {
            assert_eq!(code, status::PIXSTACK_FULL);
            assert_eq!(message, "internal pixel buffer full");
            assert!(
                detail.contains("The limit of 10 active object pixels"),
                "{detail}"
            );
        }
        other => panic!("expected PIXSTACK_FULL, got {other:?}"),
    }
    // The next call sets its own pixstack, and the stale detail was consumed.
    let catalog = extract(
        &image,
        width,
        height,
        None,
        Noise::Scalar(1.0),
        &conv_params(3.0),
    )
    .unwrap();
    assert_eq!(catalog.objects.len(), 1);
}

#[test]
fn repeated_estimate_extract_and_drop_is_stable() {
    // Every exit path frees: Background::drop -> sep_bkg_free, CatalogGuard
    // -> sep_catalog_free (success and error). Run under `leaks --atExit`
    // (macOS) to see zero leaked bytes; here it checks determinism and that
    // nothing crashes when models are freed in a loop.
    let (width, height) = (96, 80);
    let mut image = noisy_sky(width, height, 120.0, 3.0, 26);
    add_gaussian(&mut image, width, 30.0, 30.0, 1.7, 300.0);
    add_gaussian(&mut image, width, 60.0, 50.0, 1.7, 200.0);
    let first = extract_with_maps(&image, width, height, &conv_params(5.0));
    for _ in 0..200 {
        let again = extract_with_maps(&image, width, height, &conv_params(5.0));
        assert_eq!(again, first);
        let _ = extract(
            &image,
            width,
            height,
            None,
            Noise::Scalar(3.0),
            &ExtractParams {
                pixstack: 5,
                ..conv_params(1.0)
            },
        )
        .unwrap_err();
    }
}

#[test]
fn concurrent_calls_are_serialized_and_deterministic() {
    let (width, height) = (96, 80);
    let mut image = noisy_sky(width, height, 120.0, 3.0, 27);
    add_gaussian(&mut image, width, 30.0, 30.0, 1.7, 300.0);
    add_gaussian(&mut image, width, 34.0, 31.0, 1.7, 250.0);
    let reference = extract_with_maps(&image, width, height, &conv_params(5.0));
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|t| {
                let image = &image;
                let reference = &reference;
                scope.spawn(move || {
                    for i in 0..25 {
                        // Alternate pixstack values that would break a
                        // concurrent call if the global leaked between them.
                        if (t + i) % 2 == 0 {
                            assert!(extract(
                                image,
                                width,
                                height,
                                None,
                                Noise::Scalar(3.0),
                                &ExtractParams {
                                    pixstack: 3,
                                    ..conv_params(1.0)
                                },
                            )
                            .is_err());
                        } else {
                            assert_eq!(
                                &extract_with_maps(image, width, height, &conv_params(5.0)),
                                reference
                            );
                        }
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
    });
}

/// Host timing of `filterback`'s bad-mesh search (O(bad x meshes)); the
/// measurement behind `MAX_BAD_MESH_WORK`. Run with
/// `cargo test --release -p achernar-sep -- --ignored bad_mesh_work --nocapture`.
#[test]
#[ignore]
fn bad_mesh_work_host_timing() {
    for (width, height, mesh) in [
        (2000, 1000, 8),
        (2000, 2000, 8),
        (2800, 2800, 8),
        (4000, 2000, 8),
        (6000, 4000, 16),
    ] {
        let data = noisy_sky(width, height, 100.0, 3.0, 31);
        // Left half masked: every mesh there is below support.
        let mask: Vec<u8> = (0..width * height)
            .map(|i| u8::from(i % width < width / 2))
            .collect();
        let grid = MeshGrid::for_frame(width, height, mesh, mesh);
        let meshes = grid.nx * grid.ny;
        let bad = mesh_support(Some(&mask), width, height, grid)
            .iter()
            .filter(|ok| !**ok)
            .count();
        let params = BackgroundParams {
            bw: mesh,
            bh: mesh,
            ..Default::default()
        };
        let work = bad as u64 * meshes as u64;
        let start = std::time::Instant::now();
        let result = Background::estimate_unbounded(&data, width, height, Some(&mask), &params);
        let elapsed = start.elapsed().as_secs_f64();
        assert!(result.is_ok());
        println!(
            "{width}x{height} mesh {mesh}: {meshes} meshes, {bad} bad, work {work:.3e}: {elapsed:.3} s ({:.3} ns per unit)",
            elapsed * 1e9 / work as f64
        );
    }
}

#[test]
fn bad_mesh_search_above_the_work_bound_is_refused_before_sep() {
    // 2000x2000 with 8 px meshes, left half masked: 31,250 of 62,500 meshes
    // bad, 1.95e9 search units (about 1 s on the development host).
    let (width, height) = (2000, 2000);
    let data = vec![100.0f32; width * height];
    let mask: Vec<u8> = (0..width * height)
        .map(|i| u8::from(i % width < width / 2))
        .collect();
    let params = BackgroundParams {
        bw: 8,
        bh: 8,
        ..Default::default()
    };
    let error = Background::estimate(&data, width, height, Some(&mask), &params).unwrap_err();
    assert_eq!(
        error,
        SepError::WorkBound {
            bad_meshes: 31_250,
            meshes: 62_500,
            limit: MAX_BAD_MESH_WORK
        }
    );
    assert!(error.to_string().starts_with("work bound:"), "{error}");
    // Without bad meshes the same grid is linear work and allowed.
    Background::estimate(&data, width, height, None, &params).unwrap();
}

#[test]
fn mesh_support_counts_actual_partial_mesh_sizes() {
    // 20x10 with 8 px meshes: columns 0..8, 8..16, 16..20 (4 px wide);
    // rows 0..8, 8..10 (2 px high).
    let (width, height) = (20, 10);
    let grid = MeshGrid::for_frame(width, height, 8, 8);
    assert_eq!((grid.nx, grid.ny), (3, 2));
    let mut mask = vec![0u8; width * height];
    // Mesh (2,0) is 4x8 = 32 px: 16 masked is exactly half -> supported.
    for y in 0..8 {
        for x in 16..18 {
            mask[y * width + x] = 1;
        }
    }
    // Mesh (0,1) is 8x2 = 16 px: 9 masked -> 7 < 8 -> bad.
    for i in 0..9 {
        mask[8 * width + (i % 8) + (i / 8) * width] = 1;
    }
    assert_eq!(
        mesh_support(Some(&mask), width, height, grid),
        vec![true, true, true, false, true, true]
    );
    assert!(mesh_support(None, width, height, grid).iter().all(|&ok| ok));
    assert!(usable_data_value(9.9e29) && !usable_data_value(-1e30));
    assert!(!usable_data_value(f32::NAN) && !usable_data_value(f32::INFINITY));
}

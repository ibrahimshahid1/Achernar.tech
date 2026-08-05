# Astronomical quality metrics research

## Scope

This note audits Achernar's current `QualityMetrics` and `SharedObservation` fields against the astronomy and focus-measure literature supplied in `astronomy_texts/`, the architecture specification, and the vendored OpenLiveStacker implementation.

The central recommendation is to stop treating frame quality as one universal scalar. Achernar should publish a quality vector containing physical measurements, validity diagnostics, and uncertainty. Task-specific consumers can then derive a calibrated focus probability, stacking percentile, or autofocus-run decision without changing the meaning of the underlying measurements.

## Executive findings

1. The current `focus_quality` and `sharpness_score` are two reciprocal transforms of the same HFR value. They are not independent evidence and have no calibrated meaning.
2. The current `fwhm` field is actually HFD, because it is assigned `2 * HFR`. HFD is twice HFR by definition; FWHM is omg PSF-width measurement.
3. `seeing_estimate_arcsec = 2 * HFR` is invalid without pixel scale and is not a defensible seeing measurement even after converting pixels to arcseconds.
4. Returning zero for unavailable HFR/FWHM/seeing is dangerous because zero is the mathematically best value for lower-is-better width metrics. Missing measurements should be `None`/JSON `null` with an explicit status and rejection reasons.
5. `quality_uncertainty` is a unitless heuristic, not predictive uncertainty. FQE uncertainty should be reported in the same domain as its prediction, for example `sigma_log_hfr` or `sigma_hfr_px`.
6. `frame_quality_score` combines two versions of HFR with star count and saturation using arbitrary weights. It is neither a focus score nor a general astronomy quality score.
7. The repository already computes useful ingredients, including sky level, sky noise, saturation, source moments, ellipticity, position angle, cloud trend, and registration state. Those should be exposed with explicit definitions rather than compressed prematurely.

## Repository audit

### `achernar-daemon/src/vision/quality.rs`

- Bright connected components are thresholded at approximately `background + 5 sigma` and sorted by flux, peak, and area.
- Median HFR is computed from at most the 16 brightest 32x32 crops.
- Candidates are not rejected for saturation, cosmic rays, elongation, blending, crop truncation, or poor SNR before HFR aggregation.
- `star_count` counts detected components, while the number producing valid HFR values is not reported.
- The HFR routine clips negative background residuals to zero over the whole crop. Positive noise therefore contributes to total flux and can bias HFR outward, especially at low SNR or when the aperture is too large.
- There is no per-star HFR scatter, field-position summary, label uncertainty, or common-star tracking.

### `achernar-daemon/src/vision/background.rs`

- The intended sigma-clipped background idea is consistent with SExtractor, but `sigma_seed` takes the maximum of robust MAD sigma and full-frame standard deviation. Bright sources can therefore enlarge the clipping radius and contaminate the sky RMS.
- `snr_proxy_median` is the median of selected bright-pixel values divided by `sqrt(background_sigma^2 + 9)`. It is not a stellar aperture or PSF-fit SNR, and the constant `9` has no camera-noise interpretation.
- Saturation is measured as a frame-wide pixel fraction. Achernar also needs the fraction of selected stars that are saturated, because a tiny pixel fraction can still invalidate the stars used for focus.
- The generic saturation threshold effectively approaches `max_adu - 1`. A per-camera linearity/saturation level is preferable.

### `achernar-daemon/src/vision/trailing.rs`

- The second-moment ellipticity `1 - b/a` and axial position angle are standard and useful raw measurements.
- `trailing_severity` applies fixed thresholds and a star-count penalty without reporting median ellipticity, scatter, or position-angle coherence.
- Coherent trailing should elongate many stars in a common direction. Random position angles are more consistent with noise or field-dependent aberration. A doubled-angle circular concentration statistic should therefore accompany the median ellipticity.

### Consumers and contracts

- `SharedObservation` duplicates `focus_quality`, `predicted_hfr`, `hfr`, and `sharpness_score` even though they currently derive from one HFR measurement.
- The WebSocket TypeScript adapter turns missing numeric fields into zero, preserving the invalid-as-perfect-width problem.
- Telemetry also emits zero HFR when no observation exists.
- The cloud database model already makes HFR, FWHM, and quality score nullable, so nullable measurements are compatible with the durable backend.
- `focus_inference.rs` says FQE will output a signed defocus direction, while the architecture's FQE section specifies HFR and log variance. A single crop cannot reliably identify signed direction; that belongs to the position-indexed focus controller.
- The architecture's richer focus-sample schema already anticipates `star_count_used`, classical ellipticity, fit quality, and measurement validity. The current runtime schema implements only part of that design.

## Metrics traditionally used in astronomical imaging

### 1. Half-flux radius and diameter

`HFR` is the radius containing half of a background-subtracted star's measured flux. `HFD = 2 * HFR`. SEP explicitly provides half-light radius and local-background-aware aperture measurements. Law et al. use the equivalent full width at half enclosed flux and note that it is especially relevant to crowded-field photometry.

Recommended fields:

- `hfr_px_median`
- `hfr_px_mad`
- `hfd_px_median`, only when a consumer explicitly wants diameter
- `ee80_radius_px`, optional, to characterize broad PSF halos
- `star_count_detected`
- `star_count_used`

HFR should be measured inside a controlled aperture or curve of growth after local background subtraction. Crop-edge truncation must invalidate the star. Interpolate between cumulative-flux samples to avoid unnecessary whole-pixel quantization.

### 2. Fitted FWHM and PSF shape

Astronomical FWHM is commonly obtained from a 2D Gaussian or, more realistically for ground-based stellar images, a 2D Moffat fit. Law et al. measure image FWHM from 2D Moffat fits. SExtractor simulations likewise use a Moffat PSF with both FWHM and beta.

For a circular Moffat profile

```text
I(r) = I0 * [1 + (r / alpha)^2]^(-beta)
FWHM = 2 * alpha * sqrt(2^(1 / beta) - 1)
HFR  = alpha * sqrt(2^(1 / (beta - 1)) - 1), beta > 1
```

Therefore `FWHM = 2 * HFR` holds only in the circular Gaussian limit, not for a general Moffat PSF.

Recommended fields:

- `fwhm_px_median: Option<f32>`
- `fwhm_px_mad: Option<f32>`
- `psf_model: Option<Gaussian | Moffat | Empirical>`
- `moffat_beta_median: Option<f32>`
- `psf_fit_reduced_chi2_median: Option<f32>` or a normalized fit residual
- `image_fwhm_arcsec: Option<f32>`, computed only when pixel scale is known

Call the last field image FWHM, not seeing. The observed PSF also includes focus, guiding, optics, sampling, dome effects, wavelength, and exposure integration.

### 3. Encircled energy and concentration

Encircled-energy radii describe how much light remains in the PSF halo. They complement FWHM, which mostly describes core width. Law et al. distinguish FWHM from full width at half enclosed flux because the latter better represents light concentration for crowded observations.

Recommended diagnostics:

- `ee50_radius_px`, equivalent to HFR
- `ee80_radius_px`
- `concentration_50_80 = ee50_radius_px / ee80_radius_px`
- `peak_flux_fraction`, the background-subtracted peak divided by aperture flux

Peak concentration is useful for relative short-exposure ranking but is sensitive to subpixel centering, saturation, SNR, and sampling.

### 4. Strehl ratio

Strehl ratio is the measured PSF peak divided by the theoretical diffraction-limited peak under matched flux and sampling. It is a standard high-resolution imaging metric in the Law, Garrel, and Tokovinin papers.

Strehl should only be reported when aperture, obstruction, wavelength/filter response, pixel scale, detector integration, and an appropriate reference PSF are known. Otherwise expose `peak_flux_fraction` or label the result `strehl_proxy`, never `strehl_ratio`.

### 5. Ellipticity, elongation, and position-angle coherence

Second moments provide major/minor axes and position angle. Common definitions include:

```text
ellipticity = 1 - b / a
elongation  = a / b
```

Recommended frame summaries:

- `ellipticity_median`
- `ellipticity_mad`
- `major_fwhm_px_median`
- `minor_fwhm_px_median`
- `position_angle_deg`
- `position_angle_coherence`, in `[0, 1]`

For axial angles, compute coherence from doubled angles:

```text
R = abs(sum_i w_i * exp(j * 2 * theta_i)) / sum_i w_i
```

Large ellipticity with high `R` supports coherent tracking/trailing. Large ellipticity with low `R` suggests field aberration, blends, undersampling, or noisy moment estimates.

### 6. Stellar SNR and source validity

DAOPHOT explicitly models gain, read noise, sky level, nonlinearity, bad pixels, expected FWHM, and detection significance. A useful star SNR is aperture or PSF-fit flux divided by its propagated uncertainty, not a statistic over arbitrary bright pixels.

Recommended fields:

- `star_snr_median`
- `star_snr_p10`
- `saturated_star_fraction`
- `blended_star_fraction`
- `edge_truncated_star_fraction`
- `cosmic_or_hot_pixel_rejection_count`
- `source_acceptance_fraction`

If gain and read noise are unavailable, keep an explicitly named `star_snr_proxy` and include its exact formula in model metadata.

### 7. Sky and detector metrics

SExtractor estimates a spatial background map from meshes with iterative clipping and interpolation. A single full-frame median cannot represent gradients, vignetting, moonlight, nebulosity, or amp glow.

Recommended fields:

- `sky_median_adu`
- `sky_rms_adu`
- `sky_gradient_x_adu_per_px`
- `sky_gradient_y_adu_per_px`
- `sky_mesh_rms_adu`, measuring spatial background nonuniformity
- `saturated_pixel_fraction`
- `nonfinite_pixel_fraction`
- `bad_pixel_fraction`, when a detector mask exists

### 8. Classical focus-operator diagnostics

Pertuz et al. compare gradient, Laplacian, wavelet, and statistical focus measures under noise, contrast, saturation, and window-size changes. Their results support a small diagnostic bank rather than one unnamed sharpness scalar:

- modified Laplacian or variance of Laplacian
- Tenengrad variance
- normalized gray-level variance
- wavelet detail energy

Laplacian operators perform strongly on clean images but are noise- and saturation-sensitive. Statistical operators are more robust to noise. Wavelets retain useful behavior in small windows. All must be background-subtracted, noise-normalized, and benchmarked on Achernar camera strata.

Van Dokkum demonstrates that Laplacian response is also strong on cosmic rays. A Laplacian focus metric must therefore run after artifact rejection or alongside a stellar fine-structure/PSF gate.

Recommended fields use explicit names and raw or standardized units, for example:

- `modified_laplacian_snr`
- `tenengrad_variance_snr`
- `normalized_variance`
- `wavelet_detail_snr`

Do not publish their weighted sum as a physical measurement.

### 9. Registration and stacking quality

Guizar-Sicairos et al. use an upsampled cross-correlation peak for subpixel registration and a translation-invariant normalized RMS error for image comparison. OpenLiveStacker computes phase correlation, ranks its raw registration peak against session history, and independently ranks a central-ROI squared-gradient sharpness score.

Recommended fields:

- `registration_shift_x_px`
- `registration_shift_y_px`
- `registration_nrmse`
- `registration_peak_ratio` or a clearly defined normalized correlation statistic
- `registration_quality_percentile`
- `sharpness_quality_percentile`
- `transparency_flux_ratio`, preferably from common unsaturated stars
- `background_ratio`

Percentiles are appropriate for stack selection because they answer a contextual question: how does this frame compare with recent comparable frames? They are not absolute physical quality.

### 10. Autofocus-run quality

Curve quality belongs to the focus-run result, not the single-frame `QualityMetrics`:

- `fit_method`
- `fit_reduced_chi2` or robust residual scale
- `r_squared`, retained only as a secondary diagnostic
- `best_position_sigma_steps`
- `minimum_hfr_px`
- `curve_curvature`
- `left_slope` and `right_slope`
- `minimum_bracketed`
- `monotonic_left` and `monotonic_right`
- `outlier_count`
- `common_star_fraction`

A high `R^2` alone is insufficient. A shallow or unbracketed curve can fit well while leaving the optimum poorly constrained.

## Recommended contract

The exact nesting can follow repository conventions, but the semantic split should look like this:

```rust
pub enum QualityStatus {
    Valid,
    NoFinitePixels,
    NoSources,
    NoUsableStars,
    InsufficientSignal,
    Saturated,
    CropTruncated,
    ModelUnavailable,
}

pub struct PsfSummary {
    pub hfr_px_median: f32,
    pub hfr_px_mad: f32,
    pub fwhm_px_median: Option<f32>,
    pub fwhm_px_mad: Option<f32>,
    pub image_fwhm_arcsec: Option<f32>,
    pub ellipticity_median: f32,
    pub ellipticity_mad: f32,
    pub position_angle_deg: Option<f32>,
    pub position_angle_coherence: f32,
    pub star_snr_median: f32,
    pub star_count_detected: u32,
    pub star_count_used: u32,
}

pub struct FocusPrediction {
    pub hfr_pred_px: f32,
    pub sigma_hfr_px: f32,
    pub model_version: String,
}

pub struct DerivedQuality {
    pub focus_within_tolerance_probability: Option<f32>,
    pub stack_quality_percentile: Option<f32>,
}

pub struct QualityMetrics {
    pub status: QualityStatus,
    pub rejection_reasons: Vec<QualityRejectionReason>,
    pub psf: Option<PsfSummary>,
    pub prediction: Option<FocusPrediction>,
    pub operators: Option<FocusOperatorDiagnostics>,
    pub derived: DerivedQuality,
}
```

Background, cloud, registration, and pointing can remain sibling components of `SharedObservation`. The key requirement is that missing measurements stay missing and that every score names its task and reference population.

For backward compatibility, `observation_valid` can be derived from `status == Valid`, and a legacy adapter can temporarily emit old scalar fields. New control code should consume the typed measurements directly.

## Invalid-observation semantics

The current invalid block should not encode unavailable widths as zero:

```rust
QualityMetrics {
    status: QualityStatus::NoUsableStars,
    rejection_reasons: vec![QualityRejectionReason::NoStellarSources],
    psf: None,
    prediction: None,
    operators: None,
    derived: DerivedQuality {
        focus_within_tolerance_probability: None,
        stack_quality_percentile: None,
    },
}
```

`star_count_detected = 0` is a real diagnostic and may still live in a source-detection summary. `None` means no width or probability was measured. This distinction prevents an invalid frame from winning a lower-HFR comparison and lets the frontend render unavailable data as `--` rather than `0.00`.

## Task-specific derived scores

### Focus control

Use raw `hfr_px`, its uncertainty, source validity, and the focuser-position sequence. Do not use a display score to fit the focus curve.

### Focus display

After FQE calibration, expose an event with explicit semantics:

```text
focus_within_tolerance_probability
    = P(HFR <= context_tolerance | frame, calibrated model)
```

The tolerance must identify its context, such as the current run minimum or a rig/filter/binning baseline.

### Live stacking

Use a rolling or bounded-session percentile over comparable frames, based on separate gates for:

- registration residual/correlation
- stellar width or concentration
- coherent trailing
- transparency/common-star flux
- background level and noise
- saturation and source validity

This follows OpenLiveStacker's useful relative-ranking design while retaining the individual reasons a frame was rejected.

### Photometric suitability

Do not collapse photometric quality into focus quality. Publish stellar SNR, sky RMS, saturation, PSF fit residual, crowding/blending, and transparency so the photometry workflow can make its own decision.

## Implementation order

1. Introduce nullable width/prediction fields, `QualityStatus`, and rejection reasons while retaining a temporary legacy serializer.
2. Correct the semantic bug by renaming the current `2 * HFR` value to HFD and leaving FWHM unavailable until it is fitted.
3. Replace global background/noise with a SExtractor-style mesh map and iterative clipping; make saturation camera-profile aware.
4. Build one shared source catalog for quality and trailing calculations. Add saturation, edge, blend, cosmic/hot-pixel, SNR, and shape gates.
5. Report per-star measurements and robust frame summaries: median, MAD, detected count, and used count.
6. Add fitted Moffat FWHM and fit residual, then angular image FWHM when pixel scale is known.
7. Expose named classical focus-operator diagnostics and benchmark them by noise, sampling, saturation, rig, and field type.
8. Implement task-specific derived products: calibrated FQE probability for focus display and contextual percentile gates for stacking.
9. Add autofocus-run fit diagnostics and propagate measurement uncertainty into the fitted optimum.

## Verification plan

- Gaussian synthetic PSFs: verify `FWHM = 2 * HFR` only for the Gaussian model within sampling tolerance.
- Moffat synthetic PSFs across beta: verify fitted FWHM and curve-of-growth HFR against analytic values.
- Noise sweep: test HFR bias, uncertainty coverage, source acceptance, and operator robustness.
- Saturation/cosmic/hot-pixel cases: ensure artifacts cannot improve sharpness or become selected focus stars.
- Crop and field-edge cases: ensure truncated flux yields an explicit rejection reason.
- Trailing cases: test ellipticity and position-angle coherence separately.
- Empty/invalid frames: assert all unavailable widths and scores serialize as `null`, never zero.
- Consumer tests: ensure autofocus, telemetry, storage, and the frontend cannot rank an invalid observation as optimal.

## Primary literature from the supplied corpus

- Pertuz, Puig, and Garcia, focus-measure comparison: [10.1016/j.patcog.2012.11.011](https://doi.org/10.1016/j.patcog.2012.11.011)
- Subbarao and Tyan, focus-measure selection and uncertainty: [10.1109/34.709612](https://doi.org/10.1109/34.709612)
- Stetson, DAOPHOT source/PSF diagnostics: [10.1086/131977](https://doi.org/10.1086/131977)
- Bertin and Arnouts, SExtractor background/source measurements: [10.1051/aas:1996164](https://doi.org/10.1051/aas:1996164)
- Barbary, SEP library and half-light radius: [10.21105/joss.00058](https://doi.org/10.21105/joss.00058)
- van Dokkum, Laplacian cosmic-ray discrimination: [10.1086/323894](https://doi.org/10.1086/323894)
- Law, Mackay, and Baldwin, FWHM/FWHEF/Strehl frame selection: [10.1051/0004-6361:20053695](https://doi.org/10.1051/0004-6361:20053695)
- Garrel, Guyon, and Baudoz, Fourier-amplitude selection: [10.1086/667399](https://doi.org/10.1086/667399)
- Tokovinin, calibrated seeing and instrumental biases: [10.1086/342683](https://doi.org/10.1086/342683)
- Guizar-Sicairos, Thurman, and Fienup, registration error: [10.1364/OL.33.000156](https://doi.org/10.1364/OL.33.000156)
- Bosse et al., patch-quality aggregation: [10.1109/TIP.2017.2760518](https://doi.org/10.1109/TIP.2017.2760518)
- Nix and Weigend, predictive mean/variance: [10.1109/ICNN.1994.374138](https://doi.org/10.1109/ICNN.1994.374138)
- Kendall and Gal, aleatoric and epistemic uncertainty: [10.48550/arXiv.1703.04977](https://doi.org/10.48550/arXiv.1703.04977)

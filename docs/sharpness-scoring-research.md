# Sharpness scoring research notes

Context: `achernar-daemon/src/vision/quality.rs` currently computes median HFR over detected star crops and maps it to `focus_quality = 1 / (1 + hfr / 3)` and `sharpness_score = 1 / (1 + hfr)`. That is a reasonable placeholder, but it is not calibrated, pixel-scale aware, seeing aware, target aware, or robust to star-set changes.

## Local findings

- Achernar already intends a stronger path: calibrated quality scoring, focus-quality CNN/ONNX, HFR/FWHM features, uncertainty calibration, and live-stacking IQ weighting by `1/FWHM^2`.
- `quality.rs` detects bright connected components, sorts by flux/peak/area, takes median HFR over up to 16 crops, and then hardcodes reciprocal HFR-to-score transforms.
- `classical_focus_solver.rs` computes HFR/HFD as a physical measurement. This should stay as a raw metric, not become the only score.
- `fqe_runner.rs` has a placeholder gradient-energy/variance score. That is closer to a generic sharpness operator, but it still needs normalization and calibration.
- OpenLiveStacker filters live-stacking frames with rolling percentile gates: registration score, central-ROI sharpness, and brightness sigma. Its sharpness score is a sum of squared 2-pixel horizontal/vertical differences over the quality ROI.
- KStars/Ekos supports HFR, adjusted HFR, FWHM/PSF, Fourier power, blurriness, curve fits, weighted fitting, R2 limits, outlier rejection, multi-frame averaging, and relative HFR over common stars. The relative-HFR idea is especially relevant because Achernar's top-N star set can change frame to frame.

## Source map

| # | Source | Type | Why it matters for Achernar |
|---|---|---|---|
| 1 | Pertuz, Puig, Garcia, "Analysis of focus measure operators for shape-from-focus" ([DOI](https://doi.org/10.1016/j.patcog.2012.11.011)) | Paper | Broad benchmark of focus operators: gradients, Laplacian, wavelets, statistics. Useful for choosing non-stellar fallback metrics. |
| 2 | Pech-Pacheco et al., "Diatom autofocusing in brightfield microscopy: a comparative study" ([DOI](https://doi.org/10.1109/ICPR.2000.903548)) | Paper | Comparative autofocus metrics under real imaging noise; good operator-selection reference. |
| 3 | Subbarao and Tyan, "Selecting the optimal focus measure for autofocusing and depth-from-focus" ([DOI](https://doi.org/10.1109/34.709612)) | Paper | Focus metric selection and failure modes; supports benchmarking rather than hardcoding one metric. |
| 4 | Nayar and Nakagawa, "Shape from focus" ([DOI](https://doi.org/10.1109/34.308479)) | Paper | Foundational focus-as-measurement framing. |
| 5 | Nayar and Nakagawa, "Shape from focus: an effective approach for rough surfaces" ([DOI](https://doi.org/10.1109/ROBOT.1990.125976)) | Paper | Earlier applied focus-measure work; helpful for focus curves and local maxima concerns. |
| 6 | Aydin and Akgul, "A New Adaptive Focus Measure for Shape From Focus" ([DOI](https://doi.org/10.5244/C.22.8)) | Paper | Adaptive focus measures may help normalize across targets/fields. |
| 7 | Groen et al., "A comparison of different focus functions for use in autofocus algorithms" ([DOI](https://doi.org/10.1002/cyto.990060202)) | Paper | Classic focus-function comparison; reinforces multi-metric evaluation. |
| 8 | Firestone et al., "Comparison of autofocus methods for automated microscopy" ([DOI](https://doi.org/10.1002/cyto.990120302)) | Paper | Practical autofocus function tradeoffs under automation constraints. |
| 9 | Santos et al., "Evaluation of autofocus functions in molecular cytogenetic analysis" ([DOI](https://doi.org/10.1046/j.1365-2818.1997.2630819.x)) | Paper | Another applied metric comparison; useful for noise sensitivity. |
| 10 | Marziliano et al., "A no-reference perceptual blur metric" ([DOI](https://doi.org/10.1109/ICIP.2002.1038902)) | Paper | Edge-width blur measurement; possible fallback when star fitting fails. |
| 11 | Crete-Roffet et al., "The blur effect..." ([DOI](https://doi.org/10.1117/12.702790)) | Paper | Perceptual blur metric; useful for global sharpness scoring. |
| 12 | Ferzli and Karam, "Just Noticeable Blur" sharpness metric ([DOI](https://doi.org/10.1109/TIP.2008.2011760)) | Paper | No-reference sharpness metric with perceptual thresholding. |
| 13 | Narvekar and Karam, CPBD blur metric ([DOI](https://doi.org/10.1109/TIP.2011.2131660)) | Paper | Maps blur evidence to a probability-like score; better conceptual fit than raw reciprocal HFR. |
| 14 | Vu and Chandler, wavelet sharpness estimation ([DOI](https://doi.org/10.1109/LSP.2012.2199980)) | Paper | Fast local/global sharpness metric; plausible edge fallback. |
| 15 | Mittal, Moorthy, Bovik, BRISQUE ([DOI](https://doi.org/10.1109/TIP.2012.2214050)) | Paper | Blind image-quality features; potential cloud/blur/noise composite inputs. |
| 16 | Mittal, Soundararajan, Bovik, NIQE ([DOI](https://doi.org/10.1109/LSP.2012.2227726)) | Paper | Unsupervised quality model idea; less astronomy-specific but useful for anomaly scoring. |
| 17 | Saad, Bovik, Charrier, BLIINDS-II ([DOI](https://doi.org/10.1109/TIP.2012.2191563)) | Paper | DCT-domain blind quality features. |
| 18 | Moorthy and Bovik, blind IQA from natural scene statistics ([DOI](https://doi.org/10.1109/TIP.2011.2147325)) | Paper | General no-reference IQA baseline. |
| 19 | Bosse et al., deep no-reference/full-reference IQA ([DOI](https://doi.org/10.1109/TIP.2017.2760518)) | Paper | Neural quality-estimation pattern for the planned FQE. |
| 20 | Stetson, "DAOPHOT" ([DOI](https://doi.org/10.1086/131977)) | Paper/software | Stellar detection and photometry foundation; relevant to robust star candidates. |
| 21 | Bertin and Arnouts, "SExtractor" ([DOI](https://doi.org/10.1051/aas:1996164)) | Paper/software | Source extraction and shape metrics; useful for replacing connected-component blobs. |
| 22 | Barbary, "SEP: Source Extractor as a library" ([DOI](https://doi.org/10.21105/joss.00058)) | Paper/software | C/Python library version of SExtractor concepts; implementation reference. |
| 23 | Astropy 2013 core package ([DOI](https://doi.org/10.1051/0004-6361/201322068)) | Paper/software | Astronomy data model and ecosystem baseline. |
| 24 | Astropy v2.0 status ([DOI](https://doi.org/10.3847/1538-3881/aabc4f)) | Paper/software | Ecosystem reference for interoperable astronomy tooling. |
| 25 | Astropy v5.0 community/project paper ([DOI](https://doi.org/10.3847/1538-4357/ac7c74)) | Paper/software | Current-ish ecosystem reference for package selection. |
| 26 | Lang et al., Astrometry.net ([DOI](https://doi.org/10.1088/0004-6256/139/5/1782)) | Paper/software | Star-field recognition and calibration context; not a sharpness metric, but important for frame validity. |
| 27 | Collins et al., AstroImageJ ([DOI](https://doi.org/10.3847/1538-3881/153/2/77)) | Paper/software | Astronomical image processing and photometric extraction in practice. |
| 28 | Tokovinin, "From Differential Image Motion to Seeing" ([DOI](https://doi.org/10.1086/342683)) | Paper | Seeing estimation context; HFR/FWHM scores must not confuse seeing with focus. |
| 29 | Fried, atmospheric optical resolution ([DOI](https://doi.org/10.1364/JOSA.56.001372)) | Paper | Physical basis for seeing-limited image quality. |
| 30 | Noll, Zernike polynomials and turbulence ([DOI](https://doi.org/10.1364/JOSA.66.000207)) | Paper | Aberration/turbulence basis; useful if later decomposing PSF quality. |
| 31 | Fried, lucky short-exposure probability ([DOI](https://doi.org/10.1364/JOSA.68.001651)) | Paper | Frame selection theory; supports percentile/ranking over absolute hard scores. |
| 32 | Law, Mackay, Baldwin, lucky imaging ([DOI](https://doi.org/10.1051/0004-6361:20053695)) | Paper | Astronomy frame selection under turbulence. |
| 33 | Garrel, Guyon, Baudoz, Fourier-amplitude lucky imaging ([DOI](https://doi.org/10.1086/667399)) | Paper | Fourier-domain frame-quality/selection idea. |
| 34 | Guizar-Sicairos, Thurman, Fienup, subpixel registration ([DOI](https://doi.org/10.1364/OL.33.000156)) | Paper | Registration quality should be part of frame quality, especially for stacking. |
| 35 | Thevenaz, Ruttimann, Unser, pyramid registration ([DOI](https://doi.org/10.1109/83.650848)) | Paper | Robust registration implementation reference. |
| 36 | Zitova and Flusser, image registration survey ([DOI](https://doi.org/10.1016/S0262-8856(03)00137-9)) | Paper | Registration method taxonomy for frame scoring. |
| 37 | van Dokkum, cosmic-ray rejection by Laplacian edge detection ([DOI](https://doi.org/10.1086/323894)) | Paper | Laplacian edges can detect artifacts as well as sharpness; avoid mistaking hot pixels/cosmic rays for sharp stars. |
| 38 | Beroiz et al., Astroalign ([DOI](https://doi.org/10.1016/j.ascom.2020.100384)) | Paper/software | Practical astronomical registration package. |
| 39 | van der Walt et al., scikit-image ([DOI](https://doi.org/10.7717/peerj.453)) | Paper/software | Implementation reference for Laplacian, gradients, filters, morphology. |
| 40 | OpenLiveStacker ([GitHub](https://github.com/artyom-beilis/OpenLiveStacker)) | Project | Rolling percentile filters for sharpness, registration score, brightness; directly relevant because it is in this repo. |
| 41 | KStars/Ekos focus module ([GitHub](https://github.com/KDE/kstars/tree/master/kstars/ekos/focus)) | Project | Production autofocus implementation with HFR/FWHM/Fourier/blurriness, relative HFR, curve fitting, weights, and outlier controls. |
| 42 | N.I.N.A. ([GitHub](https://github.com/isbeorn/nina)) | Project | Mature automation app; useful for autofocus workflow patterns and quality gates. |
| 43 | NINA Hocus Focus plugin ([GitHub](https://github.com/ghilios/NINA.Joko.Plugin.HocusFocus)) | Project | Specialized autofocus plugin; useful for focus curve and star-measure UX expectations. |
| 44 | PHD2 Guiding ([GitHub](https://github.com/OpenPHDGuiding/phd2)) | Project | Mature star-measurement/guiding codebase; useful for star quality and SNR handling. |
| 45 | Siril ([Website](https://www.siril.org/), [arXiv](https://arxiv.org/abs/2408.03346), [GitLab](https://gitlab.com/free-astro/siril)) | Project/paper | Open astronomical preprocessing, registration, stacking, photometry workflows. |
| 46 | DeepSkyStacker ([GitHub](https://github.com/deepskystacker/DSS)) | Project | Open stacking implementation; useful for star score/frame acceptance comparisons. |
| 47 | Photutils ([GitHub](https://github.com/astropy/photutils)) | Project | Source detection/segmentation/photometry package; likely best Python reference for Achernar data tooling. |
| 48 | ccdproc ([GitHub](https://github.com/astropy/ccdproc)) | Project | Calibration/reduction reference; frame quality should be measured after sane calibration. |
| 49 | SExtractor source ([GitHub](https://github.com/astromatic/sextractor)) | Project | Production source extraction implementation. |
| 50 | OpenCV ([GitHub](https://github.com/opencv/opencv)) | Project | Practical implementations for Laplacian, Sobel/Tenengrad, phase correlation, and filtering; OLS already depends on OpenCV. |

## Implementation direction

Do not replace `1 / (1 + hfr)` with a different magic curve. Instead:

1. Keep physical measurements separate: `hfr_px`, `fwhm_px`, `fwhm_arcsec`, `star_count`, `star_snr`, `background_sigma`, `saturation_fraction`, eccentricity/trailing, registration score, and gradient/Fourier sharpness.
2. Add relative HFR for focus sequences: track common stars between frames and compare HFR ratios against the best/reference star set, following the Ekos idea.
3. Add a rolling percentile gate for live stacking: accept/reject based on recent field-specific distributions of sharpness, registration score, and brightness, following OLS.
4. Calibrate display scores: fit Platt/isotonic/logistic calibration from historical labels or proxy labels such as "kept in stack", "autofocus improved", "user rejected", and "best focus run sample".
5. Normalize by context: pixel scale, binning, filter, exposure, target altitude/airmass, seeing estimate, and rig profile. A raw HFR of 2.5 px can mean different things on different equipment.
6. Use a learned FQE when enough data exists: predict HFR plus uncertainty, then compute a calibrated quality probability from model output and classical features. This matches the Achernar v4 architecture more closely than the current reciprocal rule.

## Minimal near-term replacement

For a low-risk interim change, compute raw metrics first and make the score explicitly relative:

```text
sharpness_rank = percentile_rank(-log(hfr_px), recent_valid_frames_for_same_rig_filter_bin)
focus_quality = calibrated_or_sigmoid(z_score(log(hfr_px), rig_filter_baseline_median_mad))
frame_quality = weighted blend of focus_quality, star_count confidence, saturation penalty,
                registration confidence, brightness stability, and trailing penalty
```

This keeps the current UI behavior possible while making the score explainable and tunable. The reciprocal HFR formula can remain as a fallback only when there is no baseline/history/model.

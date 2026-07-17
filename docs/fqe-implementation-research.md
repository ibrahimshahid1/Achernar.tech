# FQE implementation research notes

Context: the proposed Focus Quality Estimator (FQE) should become a calibrated focus state estimator, not just a replacement curve for `1 / (1 + hfr)`. This bibliography is organized around implementation decisions for Achernar: labels, preprocessing, model architecture, uncertainty, calibration, validation, and edge deployment.

For papers that are primarily distributed through arXiv/PMLR rather than publisher proceedings, the DOI link uses the stable arXiv DOI form, `10.48550/arXiv.<id>`.

## Highest-impact reads

Start with these before writing code:

1. Pertuz et al. focus-measure operator survey: pick robust classical baselines before training.
2. SExtractor/SEP/DAOPHOT: improve source detection and star measurements used as labels.
3. Nix and Weigend plus Kendall and Gal: implement `hfr_pred, log_sigma_sq` correctly.
4. Guo et al. calibration and Niculescu-Mizil/Caruana: calibrate quality/confidence after training.
5. Kaufman/Rosset leakage paper: split by focus run, date, and rig; never row-random split.
6. MobileNetV3 plus Jacob et al. quantization: keep edge inference fast and stable.

## DOI map

| # | DOI | Source | Implementation use |
|---|---|---|---|
| 1 | [10.1016/j.patcog.2012.11.011](https://doi.org/10.1016/j.patcog.2012.11.011) | Pertuz, Puig, Garcia, "Analysis of focus measure operators for shape-from-focus" | Benchmark classical focus operators: Tenengrad, Laplacian, wavelets, variance, statistics. Use to choose fallback metrics and ablations. |
| 2 | [10.1109/34.709612](https://doi.org/10.1109/34.709612) | Subbarao and Tyan, "Selecting the optimal focus measure..." | Criteria for autofocus focus-measure selection; good framing for metric monotonicity, noise sensitivity, and unimodality. |
| 3 | [10.1109/34.308479](https://doi.org/10.1109/34.308479) | Nayar and Nakagawa, "Shape from focus" | Foundation for focus-as-measurement and focus curves. |
| 4 | [10.1109/ICPR.2000.903548](https://doi.org/10.1109/ICPR.2000.903548) | Pech-Pacheco et al., "Diatom autofocusing..." | Comparative focus-function behavior in noisy imaging. Useful for robust fallback scoring. |
| 5 | [10.1002/cyto.990060202](https://doi.org/10.1002/cyto.990060202) | Groen et al., "A comparison of different focus functions..." | Classic focus-function comparison; useful for selecting classical metrics to log beside FQE. |
| 6 | [10.1002/cyto.990120302](https://doi.org/10.1002/cyto.990120302) | Firestone et al., "Comparison of autofocus methods..." | Practical autofocus evaluation and automation constraints. |
| 7 | [10.1046/j.1365-2818.1997.2630819.x](https://doi.org/10.1046/j.1365-2818.1997.2630819.x) | Santos et al., "Evaluation of autofocus functions..." | Noise and image-content sensitivity in focus functions. |
| 8 | [10.5244/C.22.8](https://doi.org/10.5244/C.22.8) | Aydin and Akgul, adaptive focus measure | Adaptive focus measures; informs field/target-dependent normalization. |
| 9 | [10.1086/131977](https://doi.org/10.1086/131977) | Stetson, "DAOPHOT" | Better stellar-source measurement and PSF/photometry lineage for label generation. |
| 10 | [10.1086/132719](https://doi.org/10.1086/132719) | Stetson, growth-curve calibration | Aperture growth curves and flux measurement stability; useful for HFR/HFD label quality. |
| 11 | [10.1051/aas:1996164](https://doi.org/10.1051/aas:1996164) | Bertin and Arnouts, "SExtractor" | Source extraction and shape features; replace naive connected-component star candidates. |
| 12 | [10.21105/joss.00058](https://doi.org/10.21105/joss.00058) | Barbary, "SEP: Source Extractor as a library" | Practical C/Python implementation path for source extraction and labels. |
| 13 | [10.1051/0004-6361/201322068](https://doi.org/10.1051/0004-6361/201322068) | Astropy core package | FITS/data handling ecosystem for dataset builder and offline preprocessing. |
| 14 | [10.3847/1538-3881/aabc4f](https://doi.org/10.3847/1538-3881/aabc4f) | Astropy v2.0 project paper | Modern astronomy Python stack reference. |
| 15 | [10.3847/1538-4357/ac7c74](https://doi.org/10.3847/1538-4357/ac7c74) | Astropy v5.0 project paper | Current ecosystem baseline and reproducibility practices. |
| 16 | [10.1088/0004-6256/139/5/1782](https://doi.org/10.1088/0004-6256/139/5/1782) | Lang et al., Astrometry.net | Star-field validation and plate-solve context; useful to mark unreliable frames. |
| 17 | [10.3847/1538-3881/153/2/77](https://doi.org/10.3847/1538-3881/153/2/77) | Collins et al., AstroImageJ | Practical astronomical image processing and photometry workflow. |
| 18 | [10.1086/342683](https://doi.org/10.1086/342683) | Tokovinin, "From Differential Image Motion to Seeing" | Seeing estimation; avoid labeling seeing changes as focus errors. |
| 19 | [10.1364/JOSA.56.001372](https://doi.org/10.1364/JOSA.56.001372) | Fried, atmospheric optical resolution | Physical context for seeing-limited FWHM/HFR. |
| 20 | [10.1364/JOSA.66.000207](https://doi.org/10.1364/JOSA.66.000207) | Noll, Zernike/turbulence | Aberration and turbulence basis for future PSF features. |
| 21 | [10.1051/0004-6361:20053695](https://doi.org/10.1051/0004-6361:20053695) | Law, Mackay, Baldwin, lucky imaging | Frame selection under atmospheric variation; supports percentile/ranking labels. |
| 22 | [10.1086/667399](https://doi.org/10.1086/667399) | Garrel, Guyon, Baudoz, Fourier-amplitude lucky imaging | Fourier-domain frame quality and frame selection. |
| 23 | [10.1364/OL.33.000156](https://doi.org/10.1364/OL.33.000156) | Guizar-Sicairos et al., subpixel registration | Registration score and alignment confidence as FQE features or gates. |
| 24 | [10.1016/j.ascom.2020.100384](https://doi.org/10.1016/j.ascom.2020.100384) | Beroiz et al., Astroalign | Practical astronomical image registration reference. |
| 25 | [10.1086/323894](https://doi.org/10.1086/323894) | van Dokkum, Laplacian cosmic-ray rejection | Prevent hot pixels/cosmic rays from being interpreted as "sharpness." |
| 26 | [10.7717/peerj.453](https://doi.org/10.7717/peerj.453) | scikit-image | Reference implementations for filters, morphology, Laplacian/Sobel features. |
| 27 | [10.1109/TIP.2008.2011760](https://doi.org/10.1109/TIP.2008.2011760) | Ferzli and Karam, JNB sharpness metric | No-reference blur/sharpness fallback when star labels are unavailable. |
| 28 | [10.1109/TIP.2011.2131660](https://doi.org/10.1109/TIP.2011.2131660) | Narvekar and Karam, CPBD blur metric | Probability-like blur score; useful for calibration design. |
| 29 | [10.1109/LSP.2012.2199980](https://doi.org/10.1109/LSP.2012.2199980) | Vu and Chandler, wavelet sharpness | Fast local/global sharpness features for ablation. |
| 30 | [10.1109/TIP.2012.2214050](https://doi.org/10.1109/TIP.2012.2214050) | Mittal et al., BRISQUE | Blind image-quality features; potential auxiliary diagnostics. |
| 31 | [10.1109/TIP.2017.2760518](https://doi.org/10.1109/TIP.2017.2760518) | Bosse et al., deep IQA | Patch-based learned image-quality estimation pattern. |
| 32 | [10.1109/ICCV.2019.00140](https://doi.org/10.1109/ICCV.2019.00140) | Howard et al., MobileNetV3 | Backbone choice for low-latency edge FQE. |
| 33 | [10.48550/arXiv.1905.11946](https://doi.org/10.48550/arXiv.1905.11946) | Tan and Le, EfficientNet | Alternative efficient backbone if MobileNetV3 underfits. |
| 34 | [10.1109/CVPR.2018.00286](https://doi.org/10.1109/CVPR.2018.00286) | Jacob et al., integer-only quantization | INT8/QAT export strategy for edge inference. |
| 35 | [10.48550/arXiv.2008.08272](https://doi.org/10.48550/arXiv.2008.08272) | ONNX-MLIR paper | ONNX deployment/compilation context. |
| 36 | [10.48550/arXiv.2206.07527](https://doi.org/10.48550/arXiv.2206.07527) | QONNX | Quantized ONNX representation ideas. |
| 37 | [10.1109/ICNN.1994.374138](https://doi.org/10.1109/ICNN.1994.374138) | Nix and Weigend, mean/variance regression | Core basis for outputting `hfr_pred` plus variance. |
| 38 | [10.48550/arXiv.1703.04977](https://doi.org/10.48550/arXiv.1703.04977) | Kendall and Gal, aleatoric/epistemic uncertainty | How to separate image-noise uncertainty from model uncertainty. |
| 39 | [10.48550/arXiv.1506.02142](https://doi.org/10.48550/arXiv.1506.02142) | Gal and Ghahramani, MC dropout | Optional epistemic uncertainty estimate for shadow mode. |
| 40 | [10.48550/arXiv.1612.01474](https://doi.org/10.48550/arXiv.1612.01474) | Lakshminarayanan et al., deep ensembles | Strong uncertainty baseline; useful offline, maybe too heavy for edge. |
| 41 | [10.48550/arXiv.1806.01768](https://doi.org/10.48550/arXiv.1806.01768) | Sensoy et al., evidential deep learning | Alternative uncertainty formulation if variance head is unstable. |
| 42 | [10.48550/arXiv.1706.04599](https://doi.org/10.48550/arXiv.1706.04599) | Guo et al., calibration of modern neural nets | Temperature scaling, ECE, and calibration diagnostics. |
| 43 | [10.1145/1102351.1102430](https://doi.org/10.1145/1102351.1102430) | Niculescu-Mizil and Caruana, probability calibration | Platt/isotonic calibration evidence; useful for quality probability. |
| 44 | [10.1145/775047.775151](https://doi.org/10.1145/775047.775151) | Zadrozny and Elkan, multiclass probability calibration | Score-to-probability calibration methods. |
| 45 | [10.1145/1102351.1102363](https://doi.org/10.1145/1102351.1102363) | Burges et al., RankNet | Ranking loss for "which focus sample is better" within a run. |
| 46 | [10.1145/1571941.1572021](https://doi.org/10.1145/1571941.1572021) | LambdaRank local optimality | Ranking objective ideas for focus-run ordering. |
| 47 | [10.1007/978-3-319-24261-3_7](https://doi.org/10.1007/978-3-319-24261-3_7) | Hoffer and Ailon, triplet network | Triplet loss idea: best-focus crop should embed closer to other good crops. |
| 48 | [10.1109/ICCV.2017.324](https://doi.org/10.1109/ICCV.2017.324) | Lin et al., focal loss | Useful if training a bad-frame/valid-frame auxiliary head with imbalanced labels. |
| 49 | [10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101) | Loshchilov and Hutter, AdamW | Optimizer choice for stable CNN training. |
| 50 | [10.48550/arXiv.1608.03983](https://doi.org/10.48550/arXiv.1608.03983) | Loshchilov and Hutter, SGDR | Cosine/warm-restart learning rate schedule option. |
| 51 | [10.1145/2382577.2382579](https://doi.org/10.1145/2382577.2382579) | Kaufman and Rosset, leakage in data mining | Justifies run/date/rig-level splits instead of row-level splits. |
| 52 | [10.1145/3368555.3384468](https://doi.org/10.1145/3368555.3384468) | Oakden-Rayner et al., hidden stratification | Evaluate by rig/filter/seeing/target strata, not just aggregate MAE. |

## How this maps to the implementation

### Dataset and labels

Use SExtractor/SEP/DAOPHOT-style measurements to build robust labels:

- `hfr_px`, `hfd_px`, `fwhm_px`, flux, peak, SNR, eccentricity/elongation.
- `background_adu`, `background_sigma`, saturation fraction, hot-pixel/cosmic-ray flags.
- `run_id`, `sample_index`, focuser position, filter, binning, exposure, temperature, rig id.
- derived labels: best-in-run rank, relative-HFR ratio, valid-frame flag, focus-improved flag.

Avoid row-level random splits. Split by run/date, and hold out one rig or rig-filter combination when possible.

### Model contract

Prefer the architecture-spec contract:

```text
inputs:
  image: [B, 1, H, W]
  meta:  [B, M]

outputs:
  hfr_pred:      [B, 1]
  log_sigma_sq:  [B, 1]
  optional_valid_logit: [B, 1]
  optional_embedding: penultimate features for offset model
```

Use MobileNetV3-Small as the first production backbone. EfficientNet is worth an ablation, but MobileNetV3 is the better first fit for daemon latency.

### Losses

Baseline loss:

```text
L_hfr = exp(-log_sigma_sq) * smooth_l1(hfr_pred, hfr_true) + log_sigma_sq
```

Add one ranking term inside each focus run:

```text
L_rank = pairwise_rank_loss(score_i, score_j, label_i_better_than_j)
```

Add optional auxiliary heads only if labels are clean:

- valid observation / invalid observation classification.
- bad frame due to saturation/cloud/trailing/hot pixels.

### Calibration and promotion

After training:

- calibrate `quality_score` from predicted HFR, uncertainty, and classical metrics.
- report ECE, reliability curves, MAE in HFR, rank correlation within runs, and calibration by rig/filter/binning.
- run shadow mode before promotion: compare FQE against classical HFR on real sessions without controlling the focuser.

### Edge export

Export:

```text
models/focus_quality/<version>/model.onnx
models/focus_quality/<version>/metadata.json
models/focus_quality/<version>/norm_params.json
models/focus_quality/<version>/calibration.json
```

Use QAT or post-training INT8 only after the FP32 model passes rank/MAE/calibration gates. Quantization should be checked against calibration drift, not just raw output closeness.


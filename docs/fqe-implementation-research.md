# FQE implementation research notes

Context: the proposed Focus Quality Estimator (FQE) should become a calibrated focus state estimator, not just a replacement curve for `1 / (1 + hfr)`. This bibliography is organized around implementation decisions for Achernar: labels, preprocessing, model architecture, uncertainty, calibration, validation, and edge deployment.

For papers that are primarily distributed through arXiv/PMLR rather than publisher proceedings, the DOI link uses the stable arXiv DOI form, `10.48550/arXiv.<id>`.

## Supplied corpus audit

`astronomy_texts/` contains 51 readable PDFs. All 51 were opened, text-extracted, and classified by their relevance to focus measurement, astronomical source extraction, atmospheric/registration confounds, learned image quality, uncertainty, calibration, validation, or edge deployment. The 52-entry DOI map below has one paper that is not currently present in the directory: Stetson's aperture-growth calibration paper, DOI `10.1086/132719`. The supplied `stetson1987.pdf` is the more immediately useful DAOPHOT paper, DOI `10.1086/131977`.

The close-reading set used to make the implementation decisions below was:

| Area | Primary papers read closely | Decision supported |
|---|---|---|
| Focus operators | Pertuz; Subbarao and Tyan; Groen; Firestone; Santos | There is no universally best scalar focus operator. Laplacian measures are strong on clean data, statistics/variance are more noise-tolerant, and wavelets retain useful behavior on small windows. Benchmark a feature bank by Achernar camera and scene strata. |
| Stellar labels and artifact rejection | SExtractor; SEP; DAOPHOT; van Dokkum | Build labels from local-background-aware, deblended stellar measurements. Reject saturation, cosmic rays/hot pixels, elongated detections, blends, and crop-edge truncation before calling a detection focus evidence. |
| Astronomical confounds | Law et al.; Garrel et al.; Tokovinin; Guizar-Sicairos et al. | A single PSF width mixes focus, seeing, guiding, exposure, sampling, and optics. FQE estimates observable width; a position-indexed controller estimates focus direction. Registration and seeing remain separate diagnostics/gates. |
| Patch aggregation | Bosse et al. | Start with deterministic per-star selection and robust pooling. Learned patch weights are an ablation after enough diverse data exists, because no-reference weighting can generalize worse when only a few patches are available. |
| Efficient model | MobileNetV3; Jacob et al. | MobileNetV3-Small is a reasonable edge backbone, but preserve pixel-scale PSF information and establish FP32 behavior before QAT/INT8. |
| Predictive uncertainty | Nix and Weigend; Kendall and Gal; Deep Ensembles | Train a proper mean/variance regression objective for aleatoric uncertainty, calibrate interval coverage, and use multiple seeds offline to audit epistemic/OOD risk. |
| Ranking and calibration | RankNet; Guo et al.; Niculescu-Mizil and Caruana; Zadrozny and Elkan | Add within-run pairwise ordering supervision. Calibrate regression intervals as regression; use Platt/isotonic only for a separately defined binary quality event. |
| Validation | Kaufman and Rosset; Oakden-Rayner et al. | Split by focus run/date/rig and report operationally important slices. Crops or repeated frames from one run must never cross splits. |

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

## FQE design decision

FQE v1 will not directly emit an arbitrary `[0, 1]` sharpness number. It will estimate a physical, observable quantity for each accepted stellar crop:

```text
inputs
  image:        float32 [B, 1, 64, 64]   # compare 96x96 in an ablation
  measurement:  float32 [B, M]           # optional observable context only

raw outputs
  mu_log_hfr:       float32 [B, 1]
  log_var_log_hfr:  float32 [B, 1]

decoded outputs
  hfr_px = exp(mu_log_hfr)
  sigma_hfr_px = propagated and validation-calibrated uncertainty
```

Useful measurement context is `log_snr`, saturation fraction, field radius, and sampling/binning or pixel scale. Absolute focuser position is excluded from FQE because it is an easy rig-specific shortcut and belongs in the Focus Offset Predictor/controller. Temperature and filter identity also stay in the controller initially; they can enter FQE only after an ablation shows out-of-rig benefit without shortcut learning.

Predicting log-HFR guarantees positivity and makes relative width errors easier to model. The exported API can still expose `hfr_pred` and `log_sigma_sq`, but metadata must specify the units and transform. HFR remains in pixels inside FQE. Angular conversion requires pixel scale. `FWHM = 2 * HFR` is exact only for the circular Gaussian convention, not a general stellar PSF or Moffat profile, and a PSF width in arcseconds is not by itself a seeing estimate.

One image cannot reliably determine which way to move a focuser. The FQE produces per-image state evidence. The controller consumes a short sequence of `(position, hfr_px, sigma_hfr_px)` observations and obtains signed direction from a robust V-curve/parabola/hyperbola fit. Seeing, trailing, registration, background, and artifacts remain explicit diagnostics rather than being hidden inside one score.

## Dataset and label pipeline

### Capture contract

Before model work, retain every underlying autofocus exposure instead of only its position-level average. Each observation needs a stable `run_id`, `frame_id`, file path/checksum, focuser position, timestamp, rig/camera/optics identifiers, filter, exposure, gain, binning, temperatures, and pixel scale when known. Preserve repeated frames at the same position so label variability and aleatoric noise can be measured.

The current local schema already has an optional `focus_samples.frame_id`, but the runtime sample structure and insert path do not populate it. That provenance gap is the first implementation blocker.

### Offline stellar labels

Build the first labels with SEP/SExtractor/DAOPHOT-style processing:

1. Estimate a spatially varying local background with robust clipping.
2. Detect sources through a PSF-like matched filter and deblend connected sources.
3. Reject edge-truncated, saturated, blended, elongated, cosmic-ray/hot-pixel, and low-SNR candidates using DAOPHOT-like `sharp`/`round` and detector flags.
4. Measure flux, half-light radius/HFR, SNR, centroid, ellipticity, and saturation for every accepted star.
5. Fit a Moffat or empirical PSF for an auxiliary FWHM label; do not infer it unconditionally as `2 * HFR`.
6. Select up to 8-16 unsaturated stars distributed across an image grid, not simply the brightest 16.
7. Store per-star rows plus a robust position-level median and MAD. Fit the run's focus curve from those position-level observations.

Synthetic Gaussian/Moffat PSFs with Poisson/read noise, background gradients, undersampling, trailing, hot pixels, and cosmic rays are useful for unit tests and pretraining. They are not the final validation set. Label-changing blur must never be used as an invariant augmentation. Safe initial augmentations are flips, 90-degree rotations, subpixel translations, and physically parameterized noise/background changes whose parameters are recorded.

### Leakage-proof splits

All stars, crops, and repeated exposures from one focus run stay in one split. Use chronological train/validation/test blocks and add at least one held-out rig or rig-filter combination. Fit normalization and calibration only on train/validation. Record a split registry so future dataset rebuilds cannot silently move a run across sets.

## Model and aggregation

Use a grayscale MobileNetV3-Small as the first production backbone, with two astronomy-specific changes:

- Compare a stride-1 stem against the standard stride-2 stem. A stellar PSF may be only a few pixels wide, so early downsampling can erase the quantity being predicted.
- Compare native 64x64 and 96x96 single-star crops. Do not tile unrelated stars into one 256x256 tensor in v1; per-star inference makes artifact rejection, spatial diagnostics, and uncertainty aggregation inspectable.

Start with image-only inference and add the small measurement-context MLP only if grouped validation improves. EfficientNet is a backup ablation, not the first deployment target.

At frame level, apply deterministic source gates and aggregate accepted per-star predictions with a median or trimmed mean. Combine calibrated predicted aleatoric variance with robust between-star scatter (MAD), then recalibrate the resulting frame interval on held-out runs. Keep per-star outputs available so field tilt, aberration, and one bad source do not disappear inside a frame scalar. Learned Bosse-style patch weights are a later ablation after deterministic pooling is well characterized.

## Training objective

Let `z = log(hfr_px)`, `mu` be `mu_log_hfr`, and `s` be `log_var_log_hfr`. Begin with the Gaussian heteroscedastic negative log likelihood used by the mean/variance regression literature:

```text
L_nll = 0.5 * exp(-s) * (z - mu)^2 + 0.5 * s
```

Compare Laplace NLL if held-out residuals are clearly heavy-tailed. The architecture document's uncertainty-weighted Huber expression is a useful robust heuristic, but it is not a standard Gaussian likelihood and its stated `0.5 arc-sec` delta conflicts with the current pixel-valued HFR. Do not use it as the probabilistic default without an explicit residual model and consistent units.

Add a small RankNet-style pairwise term for meaningful pairs from the same focus run:

```text
P(i better than j) = sigmoid((mu_j - mu_i) / temperature)
L_total = L_nll + lambda_rank * binary_cross_entropy(P, target)
```

Lower HFR means better focus. Ignore ties whose HFR difference is below combined label uncertainty, and choose `lambda_rank` on grouped validation. Warm up the mean head before enabling variance learning, as the variance head can otherwise explain early residuals instead of learning calibrated noise. Train 3-5 independent seeds offline as an epistemic audit; deploy one model initially unless shadow evidence justifies conditional ensemble inference.

Use AdamW with linear warmup and cosine decay. Add an invalid-observation head only after the artifact taxonomy has enough clean positive and negative labels; focal loss is justified only if measured class imbalance makes ordinary BCE inadequate.

## Calibration and score semantics

Aleatoric regression uncertainty is calibrated on the validation split by fitting a small scale correction and checking empirical 50%, 80%, 90%, and 95% interval coverage, interval width, NLL, and residuals by stratum. Classification temperature scaling does not directly calibrate a regression variance head.

The UI/stacking `quality_score` is derived after regression and has an explicit meaning. Two supported products are:

- `P(HFR <= tolerance)` under the calibrated predictive distribution, where tolerance is defined relative to a focus-run optimum or a rig/filter/binning baseline.
- A bounded rolling percentile among comparable recent frames, following lucky-imaging and OpenLiveStacker-style frame selection.

Platt scaling or isotonic regression may calibrate the first product as a binary event. Use Platt scaling when the calibration set is small; compare isotonic only with enough held-out data. The controller consumes raw HFR and uncertainty, not the display score, and the API reports uncertainty rather than calling `1 - uncertainty` confidence.

## Evaluation and promotion

Benchmark these baselines before choosing the neural model:

1. Existing classical HFR, with corrected source selection and units.
2. Pertuz-inspired feature bank: modified Laplacian, Tenengrad/gradient energy, normalized variance, and wavelet energy.
3. OpenLiveStacker's squared two-pixel gradient score plus within-session percentile.
4. The current compact depthwise CNN.
5. MobileNetV3-Small with stride/crop/metadata ablations.

Report per-star and frame-level HFR MAE/RMSE, within-run Spearman/Kendall correlation, pairwise ordering accuracy, best-focus position error in focuser steps, predictive NLL and interval coverage, invalid-frame false acceptance, latency, and memory. Slice every result by rig, camera/optics, filter, binning/pixel scale, exposure/SNR, star count/density, saturation, field radius, seeing proxy, defocus severity, trailing, and crop truncation.

Promotion requires all of the following:

- Dataset provenance and group-split leakage tests pass.
- The model beats the strongest frozen classical baseline on held-out-run HFR and ordering metrics.
- Replaying complete autofocus runs is non-inferior on best-position error, move count, fallback rate, and convergence failures.
- Nominal 90% regression intervals achieve 87-93% coverage overall, with no severe hidden-stratum failure.
- PyTorch and FP32 ONNX outputs pass numerical parity tests; INT8 is evaluated separately for task and calibration drift.
- Raspberry Pi 4 latency meets the architecture target of `<10 ms` p50 and `<20 ms` p95 for the final input contract.
- A shadow deployment logs enough real sessions without controlling the focuser before authority is enabled.

## Implementation sequence

### Phase 0: Data provenance and contract

- Extend `FocusSample` or introduce a per-frame observation type carrying `frame_id` and image provenance.
- Save each autofocus frame/crop and its metadata; preserve repeated measurements before position-level aggregation.
- Reconcile the runtime SQLite schema with `002_focus_schema.sql` and bind `frame_id` in persistence.
- Version the label schema, crop geometry, normalization, model inputs/outputs, and units.

Primary files: `achernar-daemon/src/focus/focus_agent.rs`, `achernar-daemon/src/api/focus.rs`, `achernar-daemon/src/storage/local_store.rs`, and `achernar-daemon/migrations/002_focus_schema.sql`.

### Phase 1: Literature-backed labels and baselines

- Implement the SEP-based offline builder, per-star quality gates, HFR/Moffat measurements, grouped split registry, and synthetic PSF tests.
- Implement and freeze the classical operator benchmark, including the OpenLiveStacker metric.
- Generate a versioned Parquet manifest that points to immutable frames/crops instead of embedding accidental train/test state.

Primary files: `ml/datasets/build_focus_quality_dataset.py`, `ml/preprocess/crops.py`, `ml/preprocess/normalize.py`, `ml/datasets/split_registry.py`, and `ml/requirements.txt`.

### Phase 2: FP32 model and training

- Replace the scalar sigmoid CNN with the two-head MobileNetV3-Small regression model.
- Implement Gaussian NLL, within-run RankNet pairs, mean-head warmup, AdamW/cosine training, and 3-5 seed runs.
- Track every ablation against identical split and label versions.

Primary files: `ml/models/focus_quality_cnn.py` and `ml/train/train_focus_quality.py`.

### Phase 3: Evaluation and calibration

- Implement frame pooling, all operational metrics/slices, interval coverage calibration, and the derived quality probability/percentile.
- Freeze promotion thresholds only after the classical baseline distribution is recorded, before comparing candidate models on the final test set.

Primary files: `ml/eval/eval_quality.py` and `ml/train/calibrate_uncertainty.py`.

### Phase 4: ONNX and Rust integration

- Export and validate FP32 ONNX first; add QAT/INT8 only after semantics and calibration pass.
- Implement real ONNX Runtime inference, preprocessing parity, per-star batching, robust frame aggregation, and explicit raw outputs in Rust.
- Keep classical HFR and validity gates available as diagnostics and fallback.

Primary files: `ml/export/export_quality_onnx.py`, `achernar-daemon/src/models/onnx_runner.rs`, `achernar-daemon/src/vision/fqe_runner.rs`, `achernar-daemon/src/focus/focus_inference.rs`, and the versioned model metadata/calibration files.

### Phase 5: Shadow mode and controller promotion

- Replay archived runs, then run FQE in shadow mode while the classical solver remains authoritative.
- Feed calibrated `(position, hfr, sigma)` sequences to the controller and compare full autofocus outcomes.
- Replace the reciprocal quality formulas only after the replay and shadow gates pass; retain bounded classical fallback behavior.

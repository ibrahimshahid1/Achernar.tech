# Background-estimation benchmark (`ml/eval/bge`)

Seeded simulations with known truth, a Python SEP 1.4.1 reference, the Rust
daemon through its `bge_eval_bridge` test, and one command that scores them
(audit BGE-09, BGE-14, BGE-22, part of BGE-23).

## The command

From the repository root, with the Python environment that has the pinned
`ml/requirements-public-data.lock.txt` (numpy 2.5.4, sep 1.4.1; Python >= 3.12):

```sh
python -m ml.eval.bge.run_benchmark --out <dir> [--rust --repo <checkout with achernar-daemon>] \
    [--cargo-target-dir <dir>] [--mesh-study] [--timing] [--save-maps]
```

- Without `--rust` every Rust row is `skipped: --rust not requested`.
- `--rust` runs `cargo test --release bge_eval_bridge -- --ignored --nocapture`
  with `ACHERNAR_BGE_EVAL_MANIFEST`/`ACHERNAR_BGE_EVAL_OUT`. A missing test or a
  failed build is reported as `skipped: rust bridge unavailable (...)`, never as
  a pass; a revision whose SEP backend is not built is reported as
  `skipped: rust sep backend not integrated at <rev>`.
- `--mesh-study` adds the SEP mesh-size / source-footprint study; `--timing`
  adds 6000x4000 host timing and peak memory.

Outputs in `<dir>`: `results.json` (schema `bge-eval-results-v1`; every
scenario x method has `status` in completed | unsupported | skipped | failed
with a `reason`), `summary.md`, `configs/` (effective configurations and
sha256), `input_hashes.json`, `bridge_inputs/` (manifest + raw pixels/masks),
`bridge_outputs/`, `catalogs/`, optional `maps/`, `mesh_study/`, `timing/`, `logs/`.

Tests: `python -m pytest -q ml/tests/test_bge_fixtures.py`.

## Three comparisons, three interpretations

1. **Simulated truth** — correctness under controlled assumptions; not
   real-camera accuracy. Sensitive to how well the simulation matches real
   sensors and skies.
2. **Cross-implementation agreement** (Python SEP vs Rust SEP on identical
   input bytes and configuration) — implementation consistency, not physical
   validation. Reported against the predeclared D-006 bounds as diagnostics.
3. **Real frames** — not run. `real_frames.py` loads a real-frame manifest for
   a future run, emits observations only, refuses accuracy or pass/fail
   without a predeclared thresholds file, and never opens hold-out frames
   unless explicitly requested with such a file.

## Methods

| id | kind | what it is |
| --- | --- | --- |
| `python_sep` | reference | `sep.Background` (bw=bh=64, fw=fh=3, fthresh=0) with mask = invalid, bad_pixel, edge, native_saturation, non-finite (non-finite pixels also set to 0); `sep.extract` on `I - B(x,y)` with masked pixels 0 and the same mask passed (as the daemon adapter does), `err = R(x,y)`, thresh 5, minarea 5, 3x3 `[[1,2,1],[2,4,2],[1,2,1]]` kernel, `filter_type="conv"`, deblend 32 / 0.005, clean 1.0, pixstack 300000. Config hashes equal the daemon defaults (`80eab0fb...`, `4f4499cd...`). |
| `python_global_scalar` | comparator | One iteratively 3-sigma-clipped median and std over usable pixels, used as constant B and R. Not the Rust legacy method. |
| `oracle_true_background` | oracle | The truth B and sky noise R fed to the same detection/HFR path. Separates background error from detector/estimator effects. Not a method. |
| `rust_legacy_global_clip` | Rust | Daemon default path through the bridge. |
| `rust_sep` | Rust | Daemon opt-in SEP path through the bridge. |

All three Python rows share one detection, acceptance and HFR path
(`ACCEPTANCE_POLICY` in `python_methods.py`, an evaluation-harness policy
modelled on the daemon's gates, not the daemon's policy): reject SEP-truncated
or border-touching objects, deblended children, any masked/non-finite pixel
within the HFR aperture (native saturation tallied separately), aperture SNR
< 5, ellipticity 1 - b/a > 0.35, a neighbour centroid inside the aperture.
HFR = `sep.flux_radius(frac=0.5, subpix=5)` on `I - B`, two passes:
`rmax1 = 20`, `rmax2 = clip(4*hfr1, 6, 63)` px.

SEP fabricates numbers in some cases; the Python reference detects them and
reports instead of scoring: no mesh with >= 50% usable pixels (SEP returns
back 0, rms 1 without error) -> `insufficient_support`; frame below
`min_meshes_per_axis * bw` -> `frame_smaller_than_mesh_grid`; any usable
`R == 0` -> `zero_empirical_scatter`; any usable `R < 0` or non-finite (SEP's
bicubic RMS spline can undershoot) -> `invalid_rms_map`. Unresolved noise
means detection does not run.

## Scenarios (`scenarios.py`, generator `bge-fixtures-v1`)

Frames are 512x384 float32 ADU unless stated. Each generator uses
`numpy.random.default_rng` child streams of `SeedSequence(seed)`; seeds and
parameters are recorded in `results.json`. Parameters were chosen for physical
plausibility, not tuned to any method.

| id | content |
| --- | --- |
| `uniform_gaussian_sky` | 1000 ADU, sigma 10, no sources |
| `poisson_read_low_counts` | pedestal 100 ADU + 4 e-/px, Poisson + 1.8 e- read noise, gain 1; 25 faint stars |
| `quantization_coarse` | 200.3 ADU, analog sigma 0.5 rounded to 1 ADU |
| `quantization_plateau` | 200.2 ADU, analog sigma 0.05 -> exactly 200 off-source (zero scatter) |
| `gradient_linear_xy` | 0.4 / 0.25 ADU/px gradient |
| `vignetting_radial` | 1000(1 - 0.25 rho^2 - 0.05 rho^4) |
| `curved_bowl` | stress: 1000 + 0.02 r^2 |
| `steep_ridge_background` | 512x512, 1000 + 800 exp(-((x-256)/60)^2), sigma 5 (negative-RMS-spline reproduction) |
| `amp_glow_corner` | +600 e- exponential corner glow, Poisson + read noise |
| `heteroscedastic_noise` | R(x,y) = 6 + 12 x/(W-1) + 4 sin(pi y/(H-1)) |
| `sparse_gaussian_stars`, `moffat_stars` | 40 isolated stars (Gaussian sigma 2; Moffat FWHM 5, beta 3) |
| `defocused_donut_stars` | annulus r_out 14, r_in 5, 1.2 px seeing |
| `crowded_field` | 300 Moffat stars + 20 close doubles (3-8 px); blend groups link at 8 px |
| `extended_emission` | exponential-disk galaxy + broad nebula, declared as sources |
| `native_clipping` | 14-bit clipping at 16383, clipped pixels in `native_saturation` |
| `bad_pixels_invalid` | hot/dead clusters and columns (`bad_pixel`), NaN/Inf (`invalid`), 4 px border (`edge`) |
| `tiny_frame` | 100x80 (below two 64 px meshes per axis) |
| `all_masked`, `all_nonfinite`, `zero_size` | no usable pixels / no finite pixel / 0x0 (zero_size is Python-only) |
| `sweep_moffat_fixed` | 13 positions 4000..5200 step 100, true p0 = 4630; Moffat FWHM sqrt(3.2^2 + (0.040 (p-p0))^2); gradient sky constant over the sweep; 24 bright stars |
| `sweep_moffat_changing_set` | as above plus 40 faint stars that drop out of detection away from focus |
| `sweep_donut` | annulus r_out = 0.028 |p-p0|, r_in = 0.35 r_out, 1.3 px seeing |
| mesh study | one 20-star jittered grid (flux 3e5) on the vignetting sky, footprints Moffat FWHM 3 / 8, donuts r_out 12 / 24 / 40; SEP bw = bh in {16, 32, 64, 128} |
| `timing_6000x4000` | 2000 Gaussian stars, gradient sky; timing only |

**Truth.** `background` is B(x,y) (including pedestal); `noise_std` is the
expected std of the analog noise process (sources included), `noise_std_sky`
the same without source photons (what an RMS map estimates). Quantized
scenarios keep the analog noise as truth and record the step. Sources are
pixel-integrated (Gaussian exactly via erf; Moffat/annulus by midpoint
super-sampling). `hfr_true_px` is computed from the isolated noiseless stamp:
the radius enclosing half of the stamp flux inside the render radius
(`hfr_aperture_px`; Gaussian 6 sigma, Moffat 99.9% of model flux), with each
pixel's flux uniform over the pixel and exact circle-pixel overlap areas.
Validation (tests): for a Gaussian, pixel integration and the uniform-pixel
assumption each add 1/12 px^2 per axis, so `HFR_num = 1.17741 sqrt(sigma^2 + 1/6)`
to leading order; the test requires `0 <= HFR_num - 1.17741 sigma <= 2 eps *
1.17741 sigma`, `eps = sqrt(1 + 1/(6 sigma^2)) - 1`; the factor 2 covers the
pixel-phase term of the same order. A source is `saturated` when noiseless
B + S reaches `saturation_adu` within 2 HFR of its centre.

## Metric definitions (units)

Usable pixels: finite and outside every mask. Scalar methods are evaluated as
constant maps. e = B_hat - B_true.

- **bias** = median(e) [ADU]; mean, MAE = mean|e|, RMSE = sqrt(mean e^2), p99|e| [ADU].
- **residual structure**: RMS of e - median(e) [ADU]; **large-scale** = RMS of
  32x32-block means of e - median(e) over blocks >= 50% usable [ADU];
  residual plane slopes from least squares on e [ADU/px].
- **under sources / far**: median e where the noiseless source image exceeds
  1 x truth sky noise / farther than 16 px from any such pixel [ADU];
  e at truth star centres (median, max |e|) [ADU].
- **R error** = median(R_hat / R_true - 1) over usable pixels with R_true > 0
  (dimensionless); p05/p95 of the ratio.
- **matching**: one-to-one greedy (closest pairs first), d <= max(2 px,
  0.5 HFR_true) (10 px for extended sources). **completeness** = matched truth
  / truth (all sources; stars; bright stars with noiseless peak >= 20 x sky
  noise). **purity** = matched / candidates. **false** = unmatched candidates
  off every source footprint (per quadrant too); **fragments** = unmatched
  candidates on a source footprint. Rust detection uses the bridge
  `candidates` (focus detector before star gates).
- **blends** (multi-member truth groups): **merged** = group with a matched
  member and an unmatched member inside that detection's bounding box;
  **split** = truth star with >= 2 candidates within max(3, 1.5 HFR_true) px
  whose nearest truth it is (counts).
- **HFR** of accepted stars matched to truth stars [px]: signed bias
  median/mean, MAE, relative bias median((HFR - HFR_true)/HFR_true), dispersion,
  outlier fraction (|rel| > 0.2), all on *clean* matches (not saturated, not in
  a multi-member blend group); saturated and blend leakage counts.
  Bounded-aperture estimators measure a slightly smaller HFR than the
  total-flux truth for wide-winged PSFs; the oracle row shows that
  definitional part.
- **focus**: per frame, median HFR of accepted stars (Rust: `quality.hfr_px`).
  Hyperbola `a sqrt(1 + ((p - p0)/b)^2)` (Levenberg-Marquardt from the exact
  linearised HFR^2 quadratic) and parabola (numpy polyfit) [focuser steps].
  **shift** = fitted p0 - true p0. Failure: < 4 (hyperbola) / < 3 (parabola)
  valid frames, no convergence, downward parabola, minimum outside the sampled
  range, or fewer than 2 valid frames on either side of the fitted minimum. **Jaccard** = overlap of each frame's accepted truth-star set with the
  frame nearest p0.
- **honest absence** (inputs without usable pixels): true iff no level was
  reported as measured.
- **timing**: wall ms (median, p95 of repeats after one warm-up) and peak RSS
  (getrusage in a fresh subprocess; `/usr/bin/time -l` around the Rust test
  binary). Available-host software timing, not a deployment target.

## Known limitations

Simulations use Gaussian/Poisson noise, ideal flat fields, analytic PSFs and
no nonlinearity; diffuse emission is declared a source by convention. The
harness acceptance policy differs from the daemon's; compare Rust and Python
HFR rows with that in mind. No real-camera accuracy is implied.

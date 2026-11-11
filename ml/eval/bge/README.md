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
  failed build (no test binary ran) is reported as `skipped: rust bridge
  unavailable (...)`, never as a pass. A test that compiled and then failed
  (panic, "test failed", a signal) is `failed` with the reason. A revision
  whose SEP backend is not built is reported as
  `skipped: rust sep backend not integrated at <rev>`.
- `--mesh-study` adds the SEP mesh-size / source-footprint study; `--timing`
  adds 6000x4000 host timing and peak memory.
- `--split dev` (default) runs the development set: the 22 pre-existing
  scenarios, the 3 pre-existing sweeps and the BGE v2 development variants
  (`dev_*`). `--split holdout` adds the held-out sets (`ho_*`) and refuses to
  start (exit 2, before any held-out frame is rendered) unless
  `ml/eval/bge/acceptance/synthetic-v2.json` is tracked in git
  (`git ls-files --error-unmatch`) and unchanged versus HEAD
  (`git diff --quiet HEAD --`); results record that file's last commit and
  sha256 under `acceptance_criteria`. The held-out run includes the
  development set so one `results.json` serves every criterion;
  `--holdout-only` drops it.
- `--methods` takes method ids or groups: `default` (the five pre-v2 rows;
  the default for `dev`), `matrix` (every row of the BGE v2 matrix below; the
  default for `holdout`), `python`, `python_variants`, `rust`,
  `rust_variants`.
- `python -m ml.eval.bge.acceptance.evaluate --results <results.json> [...]`
  applies the predeclared criteria of `acceptance/synthetic-v2.json` (read
  only) and prints `pass` / `fail` / `not_evaluable` per criterion and
  implementation (`--out` writes JSON). Several results files are merged; the
  interpretations it fixes are listed in its module docstring.

Outputs in `<dir>`: `results.json` (schema `bge-eval-results-v1`; every
scenario x method has `status` in completed | unsupported | skipped | failed
with a `reason`), `summary.md`, `configs/` (effective configurations and
sha256), `input_hashes.json`, `bridge_inputs/` (manifest + raw pixels/masks),
`bridge_outputs/`, `catalogs/`, optional `maps/`, `mesh_study/`, `timing/`, `logs/`.

**Statuses.** A row has four stages: background, noise, detection, hfr
(width measured on at least one accepted star; `not_applicable` when the
scenario has no stars). `completed` = every applicable stage ran and is
scored. `unsupported` = at least one stage could not measure; if other stages
completed, the row is `partial` (`results.json`: `partial: true`,
`scored_stages`, `unscored_stages`, and `scored` per stage; `summary.md`:
`partial: <reason>`). Its numbers cover only the scored stages and must not be
aggregated with completed rows. `skipped` and `failed` rows carry no numbers.
A focus sweep is `completed` only when the fit succeeded and every frame has
an HFR; `partial` when the fit succeeded over a subset (frame counts given);
`failed` otherwise.

Tests: `python -m pytest -q ml/tests/test_bge_fixtures.py`. Tests that run SEP
ERROR with "missing prerequisite: sep==1.4.1" when the pinned wheel is
absent; the others (truth, metrics, bridge, statuses) run without it. The
benchmark command itself requires SEP.

## Three comparisons, three interpretations

1. **Simulated truth** — correctness under controlled assumptions; not
   real-camera accuracy. Sensitive to how well the simulation matches real
   sensors and skies.
2. **Cross-implementation agreement** (Python SEP vs Rust SEP on identical
   input bytes and configuration) — implementation consistency, not physical
   validation. Reported against the predeclared D-006 bounds as diagnostics:
   B maps, raw R maps (compared whatever the noise status, e.g. the negative
   spline of `steep_ridge_background`), globalback, globalrms (only when the
   daemon reports a measured rms) and catalogs, for the benchmark frames and
   every mesh-study run. Segmentation maps are not compared (the bridge does
   not export them).
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
| `rust_legacy_global_clip` | Rust | Daemon legacy path through the bridge; alias of `rust_legacy__g1` (peak_count_v1 sent explicitly). |
| `rust_sep` | Rust | Daemon opt-in SEP path through the bridge; alias of `rust_sep__n1g1` (frame_wide_v1, peak_count_v1, no regrouping sent explicitly). |

### BGE v2 method matrix (`docs/bge-v2-spec.md` §4)

`n1` = noise policy `frame_wide_v1` (D-018), `n2` = `masked_local_v2`;
`g1` = blend gate `peak_count_v1` without SEP deblend regrouping, `g2` =
`dip_annulus_v2` with regrouping.

| id | what it is |
| --- | --- |
| `python_sep__n1g1`, `__n2g1`, `__n1g2`, `__n2g2` | The Python reference of the daemon's SEP path: the labeller's `psf_measure.estimate_spatial_background` (noise policy), `sep.extract` with the daemon settings plus segmentation map, then the labeller's Rust-parity focus gates `psf_measure.focus_quality` (a port of quality.rs `measure_stars`: 48 examined / 16 used, saturation, edge, deblending policy or regrouped family, crop masked/saturated/`invalid_noise_support`, aperture SNR, core pixels, ellipticity, gate 5, HFR by the daemon's curve of growth). |
| `rust_legacy__g1`, `rust_legacy__g2` | Daemon legacy path through the bridge with the stated gate. |
| `rust_sep__n1g1`, `__n2g1`, `__n1g2`, `__n2g2` | Daemon SEP path through the bridge. |

Rust variants are sent to the bridge as method-id backends (one bridge run;
each id forces its backend, noise policy, blend test and regrouping); the
plain `legacy_global_clip` / `sep` strings would run the daemon's current
defaults (now v2) and are never used to label a row. `python_sep` keeps its
harness acceptance policy (`ACCEPTANCE_POLICY`) for continuity: its blend
rule rejects any star with another catalog centroid inside its HFR aperture
(a "neighbour" rule), not the daemon's gate 5, so `python_sep` and the
`python_sep__*` rows are not interchangeable. Rows of every SEP variant
record `refinement` (spec §3.7), `gate_diagnostics` (`annulus_exempt`,
`regrouped_accepted`, counted over accepted stars), `star_gates_config_sha256`
and `accepted_stars` (x, y, hfr).

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
`R == 0` -> `zero_empirical_scatter`; any usable `R < 0`, non-finite, or (D-018)
below 0.5 x the smallest positive filtered mesh sigma (SEP's bicubic RMS spline
undershooting, possibly while still positive) -> `invalid_rms_map`. These
rules come from `ml.labels.psf_measure.rms_map_validity`, shared with the
labeler and matching the daemon; its diagnostics are recorded under
`background_result.diagnostics.rms_map_validity`. Unresolved noise means
detection does not run.

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
| `steep_ridge_amp600` | as above with a 600 ADU ridge: the RMS spline undershoots while staying positive (D-018; expects `invalid_rms_map`) |
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

### BGE v2 sets (`dev_*` development, `ho_*` held-out)

Seeds: development 8101-8108, held-out 9101-9108, disjoint from every
pre-existing seed (101-122, 201-203, 301, 401; checked by the tests). Frames
of each set are `<set>_fNN` (sweeps `<sweep>_p<position>`); each frame draws
from `SeedSequence([seed, frame])`. Donuts are geometric-defocus pupil
shadows (`render_pupil_shadow`): a disk of radius r_out minus the obstruction
disk minus radial vane strips of width `w_img = (vane width / D) * 2 r_out`,
midpoint super-sampled with exact 1-D anti-aliasing of the strips, then
convolved with Gaussian seeing; their truth HFR is taken about the stamp's
flux centroid. Frames whose acceptance is a criterion hold at most 16
acceptable sources (the daemon's used-star cap), so the cap never decides a
truth source's fate. A source is `isolated` when every other source k
satisfies d > 2 L_j + L_k (L = r_out + 3 sigma for donuts, 1.5 FWHM Moffat).

| id | content |
| --- | --- |
| `*_newtonian_spider_donuts` | Z114-like: per frame one r_out (stratified over 6-32 px), obstruction 0.25-0.35, 4 vanes (3-vane variant on 2 of 8 frames; dev 1 of 4) of 0.5-1.5% D at one angle, seeing sigma 1.0-1.8 px; flux 5e4-1e6; 800x600 |
| `*_refractor_disks` | unobstructed disks, r_out 6-24 px |
| `*_comatic_donuts` | obstruction 0.25-0.35, brightness 1 + 0.4 cos(phi - phi0), obstruction offset <= 0.2 r_out, r_out 6-32 px |
| `*_close_doubles` | Moffat beta 3, FWHM 3-6 px per frame; 10 pairs (separation 3-10 px, flux ratio 0.2-1.0, primary 2e4-3e5) + 4 singles per frame; 640x480 (6 frames held-out, 3 dev) |
| `*_sweep_newtonian` | 13 positions 4000..5200 step 100, r_out = 0.03 abs(p - p0), obstruction 0.3, 4 vanes of 1% D, seeing 1.4 px, 20 bright stars, gradient sky (p0 4585 held-out, 4615 dev) |
| `ho_sweep_defocused_wings` | `sweep_moffat_fixed`'s generator (D-020 class) with seed 9106 and p0 4655 |
| `*_amp_glow_ridge` | Gaussian sky ridges +150..900 ADU (45-75 px, vertical or horizontal) on 1000 ADU, sigma 5; each amplitude once starless and once with 30 Moffat stars |
| `*_dense_defocused_field` | 80 spider donuts r_out 8-16 px per 640x480 frame, overlaps allowed |

**Truth.** `background` is B(x,y) (including pedestal); `noise_std` is the
expected std of the analog noise process (sources included), `noise_std_sky`
the same without source photons (what an RMS map estimates). In digitized
(quantized) scenarios the primary noise truth is the effective std of the
digitized values, computed exactly for `step*round(X/step)` with
`X ~ N(B, sigma)` (`digitized_moments`); the analog sigma is kept as a
labelled secondary truth (`R err analog`). Where the expected digitized level
differs from the analog level by more than 0.01 step (`quantization_plateau`),
the level is quantization-limited: bias is scored against the expected
digitized level and the analog bias is shown as secondary. An effective noise
below 0.01 step is not scoreable, and the oracle then reports the noise as
unresolved. Source footprints and the bright-star cut use the analog sigma in
digitized scenarios. Sources are
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
  candidates on a source footprint. Rust detection uses the bridge `catalog`
  when the backend emits one (SEP), else `candidates` (legacy detector).
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
- **BGE v2 acceptance metrics** (`metrics.py`, `v2_set_metrics` in
  results.json, used by `acceptance/evaluate.py`): *donut acceptance* =
  truth stars of a population (e.g. isolated with r_out >= 6 px) matched by an
  accepted star (standard matching), pooled over the frames; a frame whose
  detection did not run accepted nothing; skipped frames make the set not
  evaluable. *Close-double rejection* = fraction of truth pairs (e.g.
  separation >= 5 px and flux ratio >= 0.5) with no accepted star within
  separation/2 + max(2, 0.5 HFR_true) of the pair midpoint. *Accepted-donut
  HFR bias* = median (HFR - HFR_true)/HFR_true over accepted stars matched to
  donut truth. *False detections per frame* on starless frames; a frame whose
  detection did not run contributes 0. Sweeps add `frames_noise_measured`,
  `valid_frames_left_right` and the hyperbola p0 shift as a fraction of the
  sweep step.
- **BGE v2 parity** (`cross_implementation.v2`, per SEP variant, and the
  mesh study's `parity_v2`): statuses, reasons (as sets), refinement (exact
  except `invalid_rms_pixels`, within 0.1% of usable pixels), catalog sizes
  (exact and platform-aware: non-deblended objects), gate tallies, gate
  diagnostics, accepted-star sets (positions within 0.01 px).
- **timing**: Python: one fresh process per method, one discarded in-process
  warm-up call, then repeats in that process (warm); wall ms median/p95 and
  peak RSS from getrusage. Rust: one discarded warm-up process, then each
  repeat is a fresh test-binary process (cold process) under `/usr/bin/time`;
  `timing_ms` covers only the in-process background/quality/trailing stages,
  while process wall time and peak RSS include process start, frame read and
  result write. A repeat with a non-zero return code or no fresh output (output
  directories are cleared first) fails the timing row. The host label carries
  the CPU model; the Rust build profile ([profile.release]) is recorded.
  Available-host software timing, not a deployment target.

## Known limitations

Simulations use Gaussian/Poisson noise, ideal flat fields, analytic PSFs and
no nonlinearity; diffuse emission is declared a source by convention. The
harness acceptance policy differs from the daemon's; compare Rust and Python
HFR rows with that in mind (the `python_sep__*` rows use the daemon's gates).
SEP 1.4.1 thresholds masked pixels of the first image row against their own
noise value, so a negative R there is "detected" (zero flux, NaN centroid);
under `masked_local_v2` the extraction noise of every masked pixel is the
cell floor (spec amendment A-1). No real-camera accuracy is implied.

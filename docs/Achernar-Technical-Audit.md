# Achernar — Technical Project Audit

**Audit date:** 2026-08-24 · **Repository:** `ibrahimshahid1/Achernar.tech`, branch `main`, HEAD `56d15464` (2026-08-13), plus uncommitted Windows-port work
**Method:** direct inspection of source, schema, models, docs, git history, plus live execution — daemon built and run, test suites executed, a full autofocus sweep driven against the simulator, and the resulting database fed through the Python dataset builder.

Everything marked **[verified]** below was executed during this audit, not read off a README.

---

## 0. Verification performed during this audit

| Check | Result |
|---|---|
| `cargo check` (daemon + C++ bridge via `build.rs`) | **Pass**, exit 0 |
| `cargo test` (daemon) | **120 passed, 0 failed** |
| `python -m unittest discover -s ml/tests` | **54 passed** |
| `python -m unittest discover -s tools` | **20 passed** |
| `tsc --noEmit -p tsconfig.app.json` (frontend) | **Pass**, exit 0 |
| Daemon binary launch, 7 migrations applied, 25-table schema created | **Pass** |
| `GET /api/v1/drivers` — runtime driver capability report | **Pass**; sim/alpaca/asi/svb/toup/uvc/gphoto2 compiled on this host, indi/indigo/wdir not |
| `POST /connect` sim camera + sim focuser, `POST /camera/stream/start`, `POST /camera/snapshot` | **Pass** |
| `POST /focus/run` — full 7-position × 3-frame autofocus sweep | **Ran to completion**, 21 samples, 21 preserved frames, run persisted, correctly reported `Failed` (see §3) |
| `ml/data_pipeline/build_focus_quality_dataset.py` against that database | **Pass** — 296 labeled star crops from 21 frames across 1 run |
| `ml/eval/baselines.py --synthetic` | **Pass** — produced frozen baseline numbers |
| `ml/pipelines/rebuild_fqe.py --synthetic --epochs 5` — full train → calibrate → baseline → export → publish → parity | **Pass**, exit 0 — trained a model, fit `sigma_scale=1.308`, exported ONNX with FP32 parity `max\|Δmu\|=1.79e-07`, published a hash-bound bundle, and **correctly refused it**: `beats_constant_mae: false`, `runtime_eligible: false`, `ready_for_promotion_review: false` |
| Cross-language ONNX parity on that freshly trained model | **Exact.** Python: `PARITY_PY mu_log_hfr=0.792735 hfr_px=2.209430 sigma_log_hfr=0.861523`. Rust daemon: `PARITY_RUST mu_log_hfr=0.792735 hfr_px=2.209430 sigma_log_hfr=0.861523` |

**Two of the repository's own status documents are now stale in the same direction.** `HANDOFF-next-session.md` (2026-08-05) says the RAW-decode and still-capture work was "never compiled"; `MARKET-AND-NEXT-STEPS.md` (2026-08-07) lists "Compile" as step 1. The 2026-08-10 and 2026-08-13 commits closed that gap: the tree compiles clean, the test suite runs, and a 66 MB debug binary dated 2026-08-13 exists. The software prerequisite those documents name as blocking is met — the acquisition chain now runs end to end, from a focuser sweep through to a labeled `.npz`, which is what the observing campaign in §4 needs in order to start producing data.

---

## 1. What Achernar is

Achernar is a **telescope control and astronomical-imaging system** built around a Rust daemon that runs on a computer at the telescope. It is a hybrid project spanning four categories, and does not collapse cleanly into one:

- **Systems software** — a hardware-control daemon with a C++ FFI bridge to camera/mount/focuser drivers, an HTTP + WebSocket API, a durable local capture archive, and a cloud reverse tunnel.
- **Scientific instrumentation software** — star detection, curve-of-growth half-flux-radius measurement, background/noise estimation, saturation gating, FITS writing with sensor-ADU calibration headers, and a closed-loop autofocus controller.
- **Machine-learning research infrastructure** — a specified nine-model intelligence layer, of which the training/evaluation/export/serving *chain* for one model (the Focus Quality Estimator) is built end to end, with no trained weights in it yet.
- **A data-collection program** — a specified observing protocol, a hardware build plan, and implemented ingest tooling for producing, from a $150 tabletop Dobsonian and a DIY motorized focuser, the training data those models require. This is a designed subsystem rather than an intention; §4 covers it.

### The problem it addresses

Every existing amateur/prosumer astrophotography automation product (NINA, Ekos/KStars, Voyager, Sequence Generator Pro, ASIAIR, ACP Expert) automates through **deterministic scripts and hand-tuned heuristics**. Achernar's thesis, stated in the v4 architecture specification and in `MARKET-AND-NEXT-STEPS.md`, is that the gap in that market is **learned closed-loop control with calibrated uncertainty, bounded by a deterministic safety supervisor, promoted only after shadow-mode evaluation against frozen classical baselines**.

The narrower, concrete problem the implemented work attacks is **autofocus**: a telescope's focus drifts as the tube cools, and classical autofocus spends a multi-minute sweep every time to re-find it. A model that can read focus quality from a single star crop — with a calibrated uncertainty — and a second model that can predict the signed step offset to best focus, would collapse that sweep.

### What goes in, what comes out (as built today)

```
frames from a camera driver
  → LumaFrame (grayscale, container max_adu + sensor saturation_adu)
  → background/noise estimate  → star detection → per-star validity gates
  → curve-of-growth HFR per star → robust median + between-star log-sigma
  → SharedObservation published on a broadcast bus at 1 Hz
      ├→ WebSocket telemetry to the UI
      ├→ autofocus sweep: (position, HFR) samples → parabola/V-curve fit → move
      └→ SQLite rows + preserved 16-bit PNG pixels
              → Python dataset builder → (crop, log-HFR) training pairs
              → train → calibrate → baseline → ONNX export → hash-bound bundle
              → daemon loads it and publishes SHADOW predictions beside the classical ones
```

### What distinguishes it from a simpler implementation of the same idea

1. **It refuses to fabricate measurements.** Missing width metrics serialize as `null`, never `0.0` — because 0 is the mathematically best value for a lower-is-better metric and would let a blank frame win a focus comparison. The saturation gate has three states, not two: `Applied{threshold}`, and `NotPerformed`, so "no stars were saturated" and "we never checked" are distinguishable in the training set (`vision/quality.rs:36-61`).
2. **It separates container depth from sensor saturation.** `LumaFrame.max_adu` (255 or 65535, the container) is a different field from `saturation_adu: Option<f32>` (the sensor's linear clip, 16383 on a 14-bit body). Conflating them is a real bug the project found and fixed in itself; the old code hardcoded 65535 so the 0.98 gate sat at 64224 and *never fired* on 14-bit data.
3. **The ML input contract is a versioned, tested artifact.** `ml/contracts/fop_features.py` declares all 38 Focus Offset Predictor features with per-feature *provenance* (the exact SQLite column or JSON key) and an *availability verdict* — 26 `available`, 8 `degenerate`, 3 `missing`, 1 `rejected` [verified by parsing `fop_features_v1.json`]. A Python test fails if module and JSON drift; a Rust test in `api/focus.rs` fails if the daemon stops writing a `quality_json` key the contract claims to read.
4. **Splits are grouped, and there are two of them.** `ml/data_pipeline/split_registry.py` implements both random-grouped-by-run and *temporal* grouped splits, on the argument that they answer different questions ("generalize to an unseen run?" vs "still work next month?") and a gap between them is itself a finding.
5. **No model can promote itself.** `models/model_registry.rs:50` and `vision/fqe_runner.rs` both gate on `training_info.runtime_eligible == true` and `promotion_status ∈ {shadow, promoted}`. The checked-in reference model declares `"trained": false, "weights": "random", "runtime_eligible": false`, so the daemon will not load it into the live path. `pipelines/rebuild_fqe.py` *always* stamps `shadow`; promotion is a separate human decision.
6. **The C++ bridge is not a thin wrapper.** It extracts OpenLiveStacker's driver classes out of their cppcms/booster web-framework host by supplying 307 lines of shim headers (`shims/cppcms/json.h`, `shims/booster/aio/*`), so the drivers link into a standalone static library that Rust calls through a flat C ABI with an explicit version number the daemon checks at startup and refuses to run on mismatch.

---

## 2. The technical workflow, as built

### 2.1 Hardware layer

`hardware/ols_ffi.rs` (670 lines) declares the C ABI. `hardware/camera.rs` (879) wraps it in a Rust `OlsCamera`/`OlsCameraDriver` with RAII handles. The C side is `ols-bridge/` — `ols_bridge.cpp` (1,799), `camera_base.cpp` (238), `windows_mf_camera.cpp` (1,144, uncommitted), `ols_bridge.h` (712), plus shims: **4,200 lines of C/C++ authored for this project**.

Below that sits vendored OpenLiveStacker (~34k lines of its own src+include). The bridge's CMakeLists selects only the files it needs: the Alpaca camera/mount/focuser drivers, plus vendor-SDK drivers guarded by `find_library` on `sdks/` (ASI 311 MB, SVBony 163 MB, ToupTek 1.1 GB, all gitignored).

**Driver availability is reported at runtime, not assumed.** `hardware/capabilities.rs` reads a bitmask from the compiled bridge and returns a three-way status per driver: *compiled in*, *compiled out*, and *exposed in Rust but never wired to the bridge* (the `moonlite` and `usb` focuser variants). [verified: `GET /api/v1/drivers` on this Mac returns sim/alpaca/asi/svb/toup/uvc/gphoto2 enabled; indi/indigo/wdir disabled; moonlite/usb focusers disabled.]

`hardware/discovery.rs` (390) answers the different question "what is plugged in *right now*", by instantiating each local-bus driver and enumerating. Deliberately scoped to USB/SDK only — Alpaca UDP broadcast and INDI port-scanning are excluded with written reasons — probes are `spawn_blocking` with a 10 s timeout that abandons rather than cancels (an FFI call inside a vendor SDK cannot be interrupted), and discovery refuses to run while a camera is connected.

### 2.2 Two acquisition paths

**Path A — streaming.** `POST /camera/stream/start` opens a format and pushes frames into a single-slot `latest_frame` mailbox tagged with `stream_generation` and the driver's `frame_counter`. This is what autofocus uses.

**Path B — still capture.** `POST /camera/capture` calls `OlsCamera::capture_still` → `ols_camera_capture_still` → the gphoto2 driver, which pauses live view, applies **and reads back** shutter and ISO, timestamps, triggers once, drains events to `CAPTURE_COMPLETE`, downloads every companion asset, and resumes. Path B explicitly does **not** read `latest_frame` — a documented earlier defect where a "science exposure" silently returned a low-resolution 8-bit live-view JPEG.

For sensor cameras with no separate still operation (ASI/SVBony/ToupTek/SV105), `capture_from_stream` (`api/camera.rs:820`) applies settings, **discards the first post-write frame** as potentially already in flight, and waits for the next frame in the same stream generation — failing rather than substituting a stale one. The provenance rule is factored into a pure, unit-tested gate (`api/camera.rs:118`).

### 2.3 Vision loop

`vision/vision_loop.rs` runs at 1 Hz (configurable 1–5), reads `latest_frame`, and produces a `SharedObservation`:

- `vision/preprocess.rs` — 64×64 star crops, percentile-clipped normalization matching NumPy's linear interpolation exactly (there is a test asserting this).
- `vision/background.rs` (195) — clipped-median sky and MAD sigma, saturation-aware.
- `vision/quality.rs` (1,254) — connected-component star detection, then per-star rejection gates: saturated, edge-truncated, low-SNR, elongated, blended. It publishes *tallies* per reason, so "14 detected / 5 used" is explainable. HFR is the robust median over surviving stars; `hfr_log_sigma` is the robust std-dev of ln(HFR) across them, and is explicitly documented as between-star dispersion, **not** a calibrated standard error.
- `focus/classical_focus_solver.rs` (549) — the HFR itself, by curve of growth. Two properties it guarantees, both fixes to earlier code: the sky-subtracted flux is integrated **without clipping negatives to zero** (clipping rectified positive noise excursions into a pedestal that inflated HFR with the noise level), and the aperture is bounded to ~4× a first-pass estimate so a larger crop cannot accumulate more noise-only annuli.
- `vision/trailing.rs` (223) — classical elongation/trailing signal.
- `vision/fqe_runner.rs` (541) — ONNX inference, when a runtime-eligible bundle exists.

Fields that cannot be honestly produced are `None` and documented as such: `fwhm_px` (needs a Moffat/Gaussian fit; `FWHM = 2·HFR` holds only in the Gaussian limit), `image_fwhm_arcsec` (needs a pixel scale), `quality_uncertainty` (reserved for a calibrated probability).

### 2.4 Autofocus

`POST /focus/run` → `api/focus.rs:58` → `focus/focus_agent.rs` state machine: `Idle → Initializing → Sweeping → (Moving ↔ Measuring)ⁿ → Evaluating → Converged | Failed | Aborted`.

Per position: `move_focuser_to_position` (safety-checked, backlash-compensated) → settle → `wait_for_fresh_frame` keyed on the camera's own frame counter → sample from the observation → persist the exact pixels that produced the label. Then `ClassicalFocusSolver::fit_parabola` over (position, HFR); on success, move to the fitted minimum, write a `focus_actions` row, and fold the converged position into the drift filter.

Design decisions embodied here:

- **`SweepPattern::{Monotonic, Reversal}`** with a written rationale: monotone sweeps approach every position from the same direction so backlash enters identically, but 9 of the FOP's 38 features (History and Direction groups) are *constant on every row of every monotone run* — the direction head has nothing to learn. Reversal is a data-collection control, settable per request because it is a per-night decision.
- **Failed runs are still persisted, with all samples collected before the failure**, explicitly to avoid survivorship bias in the FQE dataset.
- **A local database fault must not strand the focuser** at the last sweep position: persistence failures are logged, then the hardware move completes anyway, and the action row is written only after the move succeeds.
- **The drift filter is updated on autofocus completion only, after the move succeeds** — "the one moment the daemon knows by measurement rather than by model where best focus is." Feeding it a failed run would teach it that best focus is wherever the sweep gave up.

### 2.5 Focus drift forecaster

`focus/focus_drift.rs` (1,205 lines) is a linear time-varying Kalman filter with state `x = [p, v, k]ᵀ` — best-focus position, thermal-independent drift rate, and temperature coefficient in steps/°C — transitioning as `p' = p + v·dt + k·ΔT`. Because ΔT is a *measured exogenous regressor*, the system stays linear and needs no EKF/UKF; this is the standard augmented-state joint state/parameter estimation, cited to Ljung and Simon in the module docs. Process noise uses the discrete white-noise-acceleration model for the (p,v) block (Bar-Shalom §6.2.2) with a slow random walk on k. The covariance update uses the **Joseph stabilized form** for symmetry and positive-definiteness over an all-night run.

The most interesting decision here is the **estimation/decision split**. The filter does two separable jobs: *identify* [p, v, k] from (temperature, position) pairs, which needs no optics at all; and *decide* whether predicted drift warrants a refocus, which needs the critical focus zone. An earlier version refused to construct the forecaster when optics were unspecified — killing the estimation half to protect the decision half, and, worse, substituting `off` for `unknown` without recording it. It now always constructs; `predicted_drift_steps`, `confidence_interval` and `initialized` are unconditional, while `exceed_probability`, `refocus_recommended` and `time_to_refocus_s` are `Option` with a `DriftPrediction::advice` carrying the reason. `Some(false)` and `None` are different claims.

To make the missing threshold cheap to acquire, `steps_per_micron_from_vcurve` derives it from the sweep the system already runs: for a uniform blur disc of radius R, half the flux lies inside R/√2, so along a V-curve wing `dHFR/dδ = 1/(2√2·N)` in µm, giving

```
steps_per_micron = 1 / (2√2 · N · slope[px/step] · pixel_size[µm])
```

Neglected terms are named in the function: central obstruction (~4% low for a 30% secondary), seeing and diffraction (excluded by fitting the wing rather than the vertex), and focuser non-linearity — which is why the result is tagged `ThresholdSource::MeasuredVCurve`, an estimate that always loses to a configured value.

[verified: `GET /focus/drift` on a fresh install returns `{"status":"disabled","detail":"…Set [focus.drift] enabled = true… It will start estimating immediately — optics are needed only to turn its estimate into refocus advice."}` — the three-status design is live.]

### 2.6 Persistence

7 SQL migrations define **25 tables**. `migrations/` is the single schema authority — it previously was decorative, with nothing executing it and a `frames` definition that disagreed with the live one in `local_store.rs`; a `schema_migrations` ledger and a benign-error path now handle re-runs. [verified: all 7 applied cleanly on a fresh database.]

`storage/capture_store.rs` (1,175) commits one physical exposure as one `frames` row plus N immutable `frame_assets`, with an ordering chosen so the *recoverable* failure is the likely one:

1. stage under `.<frame-id>.partial`, hashing while writing
2. write `manifest.v1.json`
3. fsync files and staging directory
4. open the SQLite transaction, insert frames + assets + outbox
5. rename staging into place
6. commit

A crash before 5 leaves a `.partial` and no rows; between 5 and 6, a complete self-describing bundle with no rows, recoverable from its manifest. What cannot happen is a committed row pointing at an unfinished file. `reconcile()` runs at startup and handles both. [verified: startup logged `Capture store reconciled: clean`.]

`storage/night.rs` defines `night_id` as the local date of sunset (before local noon belongs to the previous day) — because a sweep at 23:50 and one at 00:10 are the same night, same seeing, same tube temperature, and grouping on calendar date would let them land on opposite sides of a train/test split. It is documented as needing to stay identical to `night_id_for()` in `tools/import_session.py`.

### 2.7 ML pipeline

`ml/labels/psf_measure.py` (694) is the reference labeler: SExtractor-style sigma-clipped background, source gates matching the daemon's, curve-of-growth HFR, and a **fitted Moffat FWHM reported separately from 2·HFR**, with a test asserting the two differ.

`ml/data_pipeline/build_focus_quality_dataset.py` (401) joins `focus_samples → frames` on `frame_id`, loads the preserved 16-bit PNGs, detects stars, labels each crop, and emits an `.npz` grouped by run. Its saturation filter reads `saturation_adu` and `saturation_gate_applied` — with a deliberate asymmetry against the daemon that is spelled out in the docstring: at runtime an unknown saturation point means reject nothing and report the gate as not performed (you must keep observing); at training time it means **drop the frame**, because an ungated label is indistinguishable from a clean one once it is in the dataset.

`ml/models/focus_quality_cnn.py` — grayscale-adapted MobileNetV3-Small, two heads (`mu = E[ln HFR_px]`, `log_var`). Three deviations from the specification, each with a reason: per-star rather than whole-frame; log-HFR in pixels rather than HFR in arcseconds (multiplicative scatter, no pixel scale needed); and **pixels only, no focuser position** — because within a run HFR is monotonic in |position − focus|, so feeding position lets the network regress HFR from a rig-specific shortcut instead of measuring PSF width.

`ml/models/fqe_losses.py` — heteroscedastic Gaussian NLL (Nix & Weigend 1994; Kendall & Gal 2017) plus a RankNet within-run pairwise ordering loss (Burges 2005), and `aggregate_stars`, a transparent robust pooling whose behavior is asserted to match the Rust implementation by a test on both sides.

`ml/eval/baselines.py` freezes four classical baselines the model must beat. [verified on synthetic data: `constant` MAE 2.219 px / ρ 0.000; `direct_hfr` MAE 0.503 px / ρ +0.963 at 2.23 ms/crop; `tenengrad` ρ +0.963; `norm_variance` ρ −0.490.] Note `direct_hfr` is *executed and timed*, not assumed perfect — the file records that an earlier hard-coded 0-MAE entry was flagged in review.

`ml/pipelines/rebuild_fqe.py` chains train → calibrate → baseline → export → publish → **parity check**, where step 6 prints a `PARITY_PY` line for the same deterministic ramp crop that the Rust test prints `PARITY_RUST` for, so Rust == Python is a diffable claim rather than an assumption.

### 2.8 Cloud path

`cloud_client.rs` (230) registers with a Django backend over HTTP, then opens an outbound WebSocket to `/ws/gateway/<daemon_id>/` and holds it — a reverse tunnel, so the rig needs no inbound port. `relay_commands.rs` (365) dispatches **21 command names** (`camera.*`, `mount.*`, `focuser.*`, `equipment.*`, `drivers.capabilities`) into *the same implementation functions* the HTTP handlers call. That single-implementation rule is explicit in the code: the relay used to carry its own copy of the capture path, so every fix had to be made twice.

Django side: `apps/realtime/consumers.py` (346) has a `DaemonConsumer` for the tunnel and a client consumer for telemetry subscription, with Redis-cached command results and frame previews; `apps/equipment/views.py` (516) is the REST surface; `apps/sync/` handles registration.

---

## 3. What it can actually do today

### A. Implemented and demonstrated

- **Compiles and runs.** [verified] Rust + C++ bridge build clean; 120 Rust tests pass; binary starts, applies 7 migrations, serves 34 HTTP routes and a telemetry WebSocket.
- **Runtime driver capability reporting** distinguishing compiled/not-compiled/not-wired. [verified]
- **Camera connect, format enumeration, streaming, MJPEG and snapshot** through the OLS bridge. [verified against the sim driver: connected "Simulation", enumerated a 1304×976 MJPEG live-view format, streamed, and returned a 746 KB snapshot.]
- **Star detection and classical HFR on real sky images.** [verified: 34–39 stars detected per simulator frame, 11 surviving the validity gates, HFR ≈ 2.98–3.36 px with `hfr_log_sigma` ≈ 0.023.]
- **Full autofocus sweep, closed loop over the focuser.** [verified: `Idle→Initializing→Sweeping→(Moving↔Measuring)×7→Evaluating`, 21 samples at 7 positions, 21 preserved 16-bit PNGs, `focus_runs` + `focus_samples` + `frames` rows written with `night_id` and `frame_id` linkage.]
- **Correct refusal to converge on a degenerate curve.** [verified: the run terminated `Failed` with `best_position: null`. This is right — the simulator replays fixed frames, so HFR is flat noise with no V-curve, and the solver declined to name a minimum. It also persisted all 21 samples anyway, per the survivorship-bias rule.]
- **Provenance record sufficient for the ML contract.** [verified: `frames.quality_json` contained `hfr_px`, `hfd_px`, `hfr_log_sigma`, `star_count`, `star_count_used`, `background_adu`, `background_sigma`, `max_adu`, `saturation_adu`, `saturation_gate_applied: true`, `focuser_position`, `temperature_state: "missing"`, `fqe_*: null`, `status: "Valid"`.]
- **The daemon → dataset handoff.** [verified: the builder read that database and produced 296 crops × 64×64 float32 in [0,1], with `log_hfr`, `hfr_px`, `star_snr`, `run_key`, `frame_id`, `frame_checksum`, `night_id`, and `split_group` columns.]
- **Real ONNX inference inside the daemon.** [verified: loading the checked-in reference bundle and running a crop produced `mu_log_hfr=0.165106, hfr_px=1.179518, sigma_log_hfr=0.857477`. The graph, the runtime, the preprocessing and the output contract all work — the weights are random.]
- **The complete train → calibrate → baseline → export → publish → parity lap, run end to end.** [verified: `rebuild_fqe.py --synthetic --epochs 5` trained a MobileNetV3-Small two-head model over 48 synthetic runs with grouped splits, fit a per-star `sigma_scale = 1.308` and a post-pooling `frame_sigma_scale = 2.427` **on validation groups only** and reported coverage on test (0.829 → 1.000 at nominal 95%), ran the four frozen baselines, exported ONNX with verified FP32 parity (`max|Δmu| = 1.79e-07`), and published `metadata.json` + `calibration.json` + `norm_params.json` with a SHA-256 binding the calibration to that exact graph.]
- **The promotion gate refusing a bad model, with a structured verdict.** [verified: the published `training_info.promotion_gate` block recorded `beats_constant_mae: false` (test MAE 2.823 px against a constant baseline of 2.219 px), `meets_classical_rank_floor: false` (within-run ρ = −0.419 against a tenengrad floor of +0.913), `real_multi_run_data: false`, `runtime_eligible: false`, `ready_for_promotion_review: false` — and stamped `promotion_status: "shadow"` regardless, because the pipeline cannot promote. A five-epoch run on synthetic data was never going to beat the baseline; the point is that the gate **said so in machine-readable form and blocked the artifact**, rather than publishing a number that looked fine.]
- **Exact cross-language parity between the training pipeline and the daemon.** [verified: on the model trained minutes earlier, Python printed `PARITY_PY mu_log_hfr=0.792735 hfr_px=2.209430 sigma_log_hfr=0.861523` and the Rust daemon's ONNX runner printed the identical `PARITY_RUST` line. Preprocessing, graph execution and the calibrated sigma path agree to six decimals across two languages and two runtimes.]
- **Hash-bound model bundle verification.** `model_registry.rs` and `fqe_runner.rs` verify SHA-256 over `model.onnx` and bind `calibration.json` to the matching model hash; tests cover tampering and mismatch.
- **Durable capture archive with startup reconciliation.** [verified: `Capture store reconciled: clean`; three unit tests cover partials, missing files, and rejecting a capture with no native asset.]
- **Frontend typechecks** under `strict` and drives the daemon through two transports (direct and Django relay). [verified]

### B. Implemented but preliminary or approximate

- **`hfr_log_sigma`** is between-star dispersion, not a calibrated standard error — the code says so.
- **`steps_per_micron` from the V-curve** is explicitly an estimate; three neglected physical terms are named.
- **`mount/pointing_model.rs`** implements thin-plate-spline correction with inverse-distance-weighting fallback below 5 samples, and clamps corrections to 0.5°. No measured pointing data exists, and it is called from nothing.
- **`mount/guiding_bandit.rs` and `filter_wheel/filter_bandit.rs`** are working LinUCB implementations over fixed arm sets — unexercised, and there is no guider or filter wheel.
- **`filter_wheel/filter_snr_scorer.rs`** uses hardcoded flux/sky "proxy" constants and a fixed read noise of 5 e⁻ and aperture of 50 px. It is a plausible shape, not a calibrated photometric model.
- **`fits.rs`** is a hand-written single primary HDU. It writes `BUNIT`/`BLKLEVEL`/`CBLACK0..3`/`SATURATE`/`XBAYROFF`/`YBAYROFF`/`BAYERPAT` correctly by construction, but no file it produced has been read back by astropy or DS9 in evidence available here.
- **The Bayer phase of the RAW decoder** was verified only against a *stub* LibRaw in a harness. The repository's own handoff names this as the one thing stub testing cannot settle, and specifies the test: shoot a red-dominant target, confirm the red plane is red. LibRaw is not compiled into the current macOS build [verified: startup logs `RAW decode: LibRaw NOT compiled into the bridge`], so Path B currently archives RAW containers without deriving FITS.

### C. Partially implemented

- **The storage repository layer.** 11 of 20 files in `storage/` are 11-line empty stubs (`sessions_repo`, `focus_runs_repo`, `focus_samples_repo`, `focus_actions_repo`, `focus_predictions_repo`, `plate_solve_repo`, `pointing_model_repo`, `guider_state_repo`, `filter_switch_repo`, `focus_model_registry_repo`, `sync_client`). The persistence that actually runs is written inline in `api/focus.rs` and `capture_store.rs`. The schema for these tables exists; nothing writes most of them.
- **`models/model_registry.rs`** (323 lines, 3 passing tests including hash-tamper detection) has **zero external references** — the FQE runner carries its own resolver. It is built, tested, and unused.
- **`processing/plate_solver.rs`** validates its inputs and then always returns `Unsupported("No local astrometry solver backend is configured yet")`.
- **`control/`** — `safety_guard.rs` (168) is live, called from `api/focus.rs` on every focuser move. `action_arbiter.rs`, `control_supervisor.rs` and `ownership.rs` are referenced only within `control/`.
- **The Django AI engine.** `apps/ai_engine/{quality,recommender,anomaly,embeddings}/` are 11–12 line files containing a docstring and a TODO list. Only `embeddings/models.py` (79 lines, Django ORM models for 512-D AstroM3 embeddings) has content, and `migrations/0002_pgvector_search_column.py` exists for it.
- **The frontend.** `RunRoom.tsx`, `Dashboard.tsx`, `CameraControls.tsx` (543 lines) and `MjpegLiveView.tsx` are real and wired. `Queue.tsx`, `Equipment.tsx`, and the `mount/`, `imaging/`, `cloud/`, `settings/` component modules are labeled placeholders returning `"— stub"`.

### D. Currently under active implementation

**The Windows rig-agent port** — uncommitted as of this audit, files dated 2026-08-13:

- `ols-bridge/src/windows_mf_camera.cpp` (1,144 lines, untracked) — a **new Media Foundation camera backend** implementing the same OLS `CameraDriver`/`Camera` contract as the libuvc backend, so the Rust and web layers continue to see `driver_id: "uvc"`. Explicitly avoids requiring a Zadig/WinUSB driver swap, so it coexists with SharpCap and other Windows astronomy software.
- A Windows portability fix to vendored `alpaca_client.cpp` (+41/−14): Winsock wants `SO_RCVTIMEO` as a `DWORD` of milliseconds where POSIX wants a `timeval` — which compiles on Windows and produces an invalid timeout — plus `WSAGetLastError()` in place of `errno`.
- `achernar-daemon/vcpkg.json` with a **pinned registry baseline** so a later port update cannot silently change what CI and a bootstrapped rig build.
- `scripts/windows/` — 827 lines of PowerShell: `bootstrap.ps1`, `build.ps1`, `run-daemon.ps1`, and `smoke-api.ps1` (444 lines), an eight-stage hardware acceptance test covering discovery, exact device-index selection, capability-driven control write-readback, stream start, REST snapshot, binary WebSocket JPEG delivery, a persisted capture with SHA-256 provenance, and restoration of original controls on teardown.
- `.github/workflows/windows-daemon.yml` — MSVC x64 build + full Rust test suite + a simulator API smoke test on `windows-2022`.
- `docs/windows-rig-agent-testing.md` (217 lines), whose status section states plainly: *"Until that new job has its first green run, it is an intended gate rather than proof that the port builds."*

**What exists:** the driver, the build system, the scripts, the CI definition, the runbook. **What remains:** the workflow is untracked, so it has never run; nobody has built the daemon on Windows; no SV105 has been opened through Media Foundation.

**Why:** it moves the target deployment from the developer's Mac to a Windows rig computer sitting at the telescope, and unblocks the SV105 as the primary imaging camera — which the build plan argues for over the DSLR on three grounds (it runs on the mature streaming Path A, a 25-position sweep takes 2–3 minutes instead of 8–10, and 100 g will not sag a plastic rack-and-pinion focuser the way a 700 g DSLR would).

### E. Planned or documented but not implemented

- `hardware/filter_wheel.rs` — two `todo!()` calls. Unreachable: `POST /connect` accepts only `camera`/`mount`/`focuser`.
- `hardware/sensors.rs` — `todo!()`. No environment sensor exists.
- `hardware/indi_client.rs`, `hardware/serial_mount.rs`, `hardware/usb_camera.rs`, `hardware/v4l_camera.rs` — 12–14 line stubs.
- `processing/{debayer,stacking,calibration}.rs` — 4 lines each: `pub struct` + TODO.
- `ai/{focus_score,cloud_detect,recommendations}.rs` — 4 lines each, dead, superseded by `vision/`.
- `models/onnx_runner.rs` — the generic ONNX path is `todo!()`; only the FQE-specific runner works.
- `focus/focus_policy.rs` — `todo!("Implement hybrid decision logic")`, unreferenced.
- ML training entry points for eight of nine models: `train_focus_offset.py`, `eval_offset.py`, `export_offset_onnx.py`, `build_focus_offset_dataset.py`, `cloud_classifier.py`, `trailing_cnn.py`, `filter_recommender.py`, `exposure_bandit.py`, `training/{focus,cloud,embeddings}/train.py` all raise `NotImplementedError` or are TODO shells.
- Model bundles: `cloud_classifier/1.2.0/` and `trailing_detector/0.8.0/` contain a `metadata.json` with `"sha256": null, "dataset_size": null` and **no `model.onnx` at all**. `focus_offset/` contains only a README explaining that its previous `0.3.0/` directory was deleted because it declared a ten-feature input against a 38-feature specification, three of which named quantities the daemon does not produce, and encoded filter as an ordinal — "an ordinal filter index asserts that Ha sits between B and OIII on some scale, which is not a thing."

### Model and data state

No model has been trained on sky data yet, and the observing campaign that produces it begins now. Concretely [verified]: `models/focus_quality/1.0.0/metadata.json` declares `"status": "reference"`, `"trained": false`, `"weights": "random"`, `"runtime_eligible": false`; the working `achernar.db` holds one simulator focus run from 2026-07-07; the 5 AstroM3 pretrained checkpoint sets are downloaded.

The daemon behaves correctly under that condition rather than degrading: the promotion gate keeps untrained bundles out of the live path, the `fqe_*` fields on the observation bus publish as `null`, and the classical measurement remains the control authority. **How the data gets collected is itself a designed and partly-implemented subsystem — §4.**

---

## 4. The data-collection system

A nine-model intelligence layer needs nine different kinds of labeled data, almost none of which can be bought or downloaded. Achernar treats data acquisition as a subsystem with the same specification discipline as the daemon: a 2,703-line protocol document (`achernar_data_collection.tex`), a set of database tables and ingest tools built specifically to serve it, and daemon features that exist only because a runbook asked for them.

**A meaningful part of this subsystem is implemented and tested**, and the end-to-end ingest path was exercised during this audit.

### 4.1 The protocol document

`achernar_data_collection.tex` is organized in five parts plus five appendices:

| Part | Contents |
|---|---|
| I | Capability audit — what the daemon can capture today, "written by reading the tree, not the previous handoff notes"; a runtime driver matrix; seven numbered blocking defects D1–D7 ordered by how much data each spoils |
| II | Instrument inventory — what each rig can physically measure; sampling analysis per optical configuration |
| III | The path bytes take, table by table; the commit protocol; grouping and leakage; per-model data requirements |
| IV | Priority tiers, critical path, risk register, campaign schedule |
| V | Sourcing routes — own gear, the UNCG track, remote networks, purchased hardware, each with an activation trigger |
| A–E | **6 bench procedures** (BP-0…BP-5), **9 observing runbooks** (RB-1…RB-9), an observation log schema with a machine-readable JSON form, a post-session ingest checklist, and a table of standing decisions with rationales |

### 4.2 Per-model data requirements

Every model gets the same eight-field template — *Predicts / Label source / Unit and group / Covariates / Rigs / Budget / Blocked by / Runbook* — and three budget tiers with stated meanings:

- **v0 — plumbing.** Enough to prove the dataset builder, split registry, training loop and ONNX export run end to end on real bytes. "Expect the model to be bad; the deliverable is a green pipeline."
- **v1 — shadow-eligible.** Enough to beat the classical baseline on a held-out group split and run in shadow. Not promotable.
- **v2 — promotion candidate.** Enough groups, rigs, nights and covariate span that a held-out *rig* evaluation is meaningful.

Budgets are stated in **groups first, rows second**, because the binding constraint on a grouped split is the group count: "Twenty thousand star crops from three focus runs is three data points. This is the single most common way an astro-ML dataset that looks large turns out to be worthless."

| Model | Row / group | v1 budget | Runbook |
|---|---|---|---|
| 1 — FQE | star crop / focus run | 60 runs, ≥2 rigs, ≥4 nights | RB-2 (manual), RB-3 (motorized) |
| 2 — FOP | focus sample / run | 200 runs (~4,000 rows) | RB-3 + passive harvesting |
| 3 — FDF | consecutive run pair / night×rig | 3 nights per rig, ≥6 °C span each | RB-4 |
| 4 — COC | sky frame / night, in 10-min label blocks | 8 nights; ≥2,000 frames for {clear, thin}, ≥800 each for {moderate, heavy, overcast} | RB-5 |
| 5 — TED | 32×32 star cutout / (field, exposure step, night) | 3 nights, 5 fields, 12 exposure steps, 20 repeats (~100k cutouts) | RB-6 |
| 6 — Background/SNR | *no rows — constants* | 5 bodies × 4 ISOs × (20 bias + 12 flat pairs) ≈ 1,400 frames | BP-2 |

### 4.3 Label sources: four of the five labeled models bootstrap without human annotation

Five of the in-scope models produce labeled rows (FQE, FOP, FDF, COC, TED); the sixth, Background/SNR, needs measured constants rather than labels. Of the five, only one requires a person to supply ground truth. That is the design property that makes the campaign tractable for one person with a $150 telescope.

- **FQE — automatic.** SEP/DAOPHOT-style HFR from `ml/labels/psf_measure.py`. No human labeling at all.
- **TED — physically computed, and exactly.** An untracked exposure trails by `15.041·cos δ` arcsec per second. The plan draws the consequence explicitly: *"The absence of tracking is an asset, not only a limitation… a physically computed, exactly known elongation label, available in unlimited quantity, on any clear night, with no rental."* The Dobsonian's biggest limitation is turned into the one instrument advantage the project has. An exposure ladder then gives a continuous label axis rather than a binary trailed/not-trailed split.
- **FOP — automatic but retrospective.** The label is `p_best − p_i`, and `p_best` is known only once the run converges, so an aborted run yields no rows. The plan's response is **passive harvesting**: instrument the daemon so *every* autofocus run, for any purpose, writes FOP rows, making the dataset a free byproduct of every other campaign.
- **FDF — automatic from consecutive converged runs.** `α_new = (p_k − p_{k−1}) / (T_k − T_{k−1})`, EWMA'd at 0.8/0.2 into `focuser_profiles.temp_coefficient_steps_per_c`. Because this is a Kalman filter with one fitted scalar per rig rather than a learned model, it wants *few, well-separated, well-thermometered* measurements: "Eight clean autofocus runs spread over a 6 °C night beat two hundred runs at constant temperature." The plan also notes α is a property of the optical tube, not the camera, so a loaner scope's coefficient does not transfer.
- **COC — the one model a human must label**, and the labels are irrecoverable, which is why it starts first and runs every night regardless.
- **Model 6 — not labels at all.** It needs *constants* (`READNOISE`, `SENSOR_SATURATION`), which is why it is bench work and weather-independent.

### 4.4 Rig assignment

Five rigs are defined from owned plus loanable gear, each matched to the campaigns it can actually serve.

**SV105 (primary imaging camera)** — Sony IMX307, 1920×1080, 2.9 µm, USB 2.0. The inventory analysis draws three consequences that shape the whole plan: it is a **UVC class device, not an SVBony-SDK device**, so it must come in through the `uvc` driver and not `svb`; its **8-bit output makes the saturation gate load-bearing and self-satisfying**, since `resolve_saturation_adu` returns `Some(max_adu)` for an 8-bit container so the gate fires by construction; and its **500 ms exposure ceiling rules out deep sky entirely** while making it excellent for high-cadence focus sweeps, where dozens of frames per focuser position arrive in seconds.

**Z114** — 114 mm, 450 mm, f/3.95 tabletop Dobsonian, manual rack-and-pinion focuser, no tracking. A 2× Barlow is a **standing decision**, marked mandatory because native focal length is undersampled at best focus for every available camera.

**Five loanable bodies** (Canon T3i, T6S, M50, R50, Lumix G95) come with a **per-body gphoto2 tethering risk table**, assessed on the operation that matters — remote shutter release, not file download. T3i/T6S are treated as the primary tethered bodies on mature Canon PTP support; M50 is medium risk; R50 and G95 are high risk. The plan then makes the risk non-blocking: *"Bodies that cannot be tethered are still useful"* on an intervalometer for COC, sky background and trailing, none of which need the daemon to hold the shutter — but they cannot participate in a closed loop, so they are assigned to open-loop campaigns and never to a rental night.

The bill of materials for the whole primary path is under $100, and the plan carries a **fallback matrix defined as triggers rather than options** — LCO public archive (activate: now, free, no application), UNCG/TCO (now), LCO Global Sky Clubs, Skynet, a ZWO EAF if the DIY focuser fails twice, a Seestar if the build slips past October, iTelescope only after the focus stack is done — including one marked **Never**: a GoTo mount, because it solves tracking and tracking is not the blocker.

### 4.5 What is already built to serve the campaign

The ingest half of the campaign is not a plan. It is code, with tests.

| Artifact | Lines | What it does for the campaign |
|---|---|---|
| `tools/import_session.py` | 664 | The **sidecar importer**, closing gaps G2/G4/G5. Ingests frames the daemon did not capture — intervalometer runs, untethered bodies, hand-shot calibration — mirroring `capture_store.rs` exactly: hash while staging to `.partial`, manifest, fsync, transaction, atomic rename, commit. A RAW and its in-camera JPEG are recognized as **one exposure with two assets**, not two frames. Idempotent by content hash, so re-importing a card is safe. Metadata falls back EXIF → session log → file mtime and records which in `acquisition_json.metadata_source`, because "a timestamp from mtime is not the same observation as one from the shutter." |
| `tools/photon_transfer.py` | 347 | **All of Model 6**, plus the `saturation_adu` constant that D1 needs. Paired-flat photon transfer curve; saturation from the variance turnover, not the largest value in the file; refuses demosaiced input; measures the four CFA phases separately. Emits a device-profile JSON, a PTC plot, and the exact TOML line to paste. |
| `tools/` test suites | 414 | 20 tests over synthetic fixtures. [verified passing] |
| Migration 006 | 72 | `environment_log` and `disturbance_log` — schema for the hand-logged ambient temperature, humidity, sky class and disturbance records that Appendix C specifies. Sky labels land in `frame_measurements` beside where a COC model's predictions will go. |
| Migration 007 | 62 | `night_id` and `session_id` on `focus_runs`, `night_id` on `frames`, with indexes — the grouping keys the split registry needs. |
| `frame_measurements` | — | Append-only, one row per (frame, measurement type), stamped with `pipeline_version`, `model_version`, `preprocessing_version`. "Measurements are opinions produced by a specific version of a specific pipeline; overwriting them inside `frames` would make it impossible to re-evaluate later without destroying what the previous model said." |
| `capture_attempts` | — | Written **before** the camera is commanded, so "the mount was slewing and the capture silently did nothing" is distinguishable from "no capture was requested." This is the denominator in every yield calculation in the plan. |
| `SweepPattern::Reversal` | — | The per-request sweep control that exists solely so RB-2/RB-3's "one run in three should be a reversal run" is executable — the only source of variation for the FOP's 9 History and Direction features. |
| `sweep_radius_points` | — | Per-request sweep width: 3 for an ordinary 7-point sweep, 12 for the 25-point campaign sweep, still truncated by `max_total_travel_per_run` so raising it cannot overrun the safety budget. |
| `night_id` (Rust + Python) | 107 | The noon-boundary observing-night rule, implemented identically on both sides with a comment requiring they stay in sync, so daemon-captured and imported rows group into the same night. |
| `--allow-missing-saturation` | — | An explicit escape hatch on the dataset builder for synthetic fixtures, which stamps the manifest `permissive` so a fixture-derived dataset can never be mistaken for a real one. |

Two of the importer's design choices are worth recording because they are refusals rather than features. It writes **no `capture_attempts` rows** — "that table records what the daemon *commanded*. An offline import commanded nothing, and inventing attempts would corrupt the denominator in every yield calculation." And it stamps `temperature_state = 'missing'` rather than borrowing a plausible ambient value, putting ambient into `environment_log` where its provenance is explicit.

**The schema is the protocol made executable, not a generic log table.** [verified in `006_environment_log.sql`] `environment_log` carries exactly Appendix C.2's fields — `ambient_temp_c`, `humidity_pct`, `dew_point_c`, `wind_beaufort`, `transparency_nelm`, `seeing_impression` — plus a `sky_class` column constrained to `BETWEEN 0 AND 4`, with the D7 resolution written into the column comment: five levels are collected because the specification's head is binary while the code is five-class, and "five collapses to two; two can never be expanded to five, and the sky is not re-observable." It also carries a `source` column distinguishing `manual_log | probe | forecast`, because "a forecast temperature and a measured one are not the same observation." The importer's own `--log` argument documents itself as taking the "Session JSON log (Appendix C.5 schema)" — the tool, the table and the paper field card are one design.

**[verified during this audit: the full ingest→dataset path runs.]** A daemon-driven autofocus sweep produced 21 linked frames and samples, and `build_focus_quality_dataset.py` turned that database into 296 labeled star crops with run, night, rig, camera and frame-checksum provenance attached — which is exactly the operation Appendix D step 8 asks the observer to perform after every session.

### 4.6 Field discipline, designed in

The protocol treats the observer as part of the instrument, and the parts of the record only a person present can supply are enumerated rather than assumed.

**The irrecoverable-metadata table** lists 11 quantities that cannot be reconstructed from pixels afterwards — sky-condition class, ambient temperature, humidity and dew point, focuser absolute position, last focuser direction, wind, tracking state, moon geometry, transparency, dew on the optics, and *anything you touched mid-run*. Five are marked **"You, at the time."**

**The disturbance block** gets its own argument: "If you bump the tripod, nudge the focuser to check something, swap an eyepiece, or a car's headlights sweep the field — write the timestamp. One un-annotated disturbance in the middle of a focus run turns a clean V-curve into a bimodal one, and the fitter will silently converge on the wrong minimum." There is no way to detect it from the pixels alone, so there is a dedicated block on the field card and a `disturbance_log` table behind it.

**Absence is a value.** "Every one of these may be absent. Write `null`, not a guess" — explicitly so the log does not undermine the `temperature_state` distinction the daemon already models.

**Each runbook carries abort and acceptance criteria.** RB-2 aborts on cloud crossing the field, fewer than 8 stars at sweep centre, or visible wind shake, and flags rather than keeps a run whose ambient drifts more than 0.5 °C — "keep the data but flag it; it belongs to RB-4, not here." Acceptance is a clean V or U, monotonic on each side, minimum interior to the sweep, ≥15 stars at the minimum and ≥8 at both ends. The standing preference is to **drop a contaminated run rather than clean it**.

**The post-session checklist** is ten steps with runnable SQL, and two of them are the ones that matter: count what you got against `capture_attempts` (what you *attempted*), and dry-run the dataset builder checking **the group count, not the row count**. Step 9 verifies the saturation gate actually ran that night — "if it reports *not performed*, D1 or D2 is still open and the night's focus labels are suspect." Step 1 is "do not format the card," which is tied directly to the capture-store commit protocol: if the laptop dies mid-download nothing is half-recorded, but the frame may still be on the card.

### 4.7 Yields, tiers, and sequencing

Yields are computed rather than hoped for. RB-2 runs ~15 minutes per sweep at ~10 runs on a productive night, so **v1 (60 runs) is six clear nights and v2 (150 runs) is fifteen**. A 21-position sweep with 150 usable stars per frame yields ~3,000 crops — but one group, which is why the schedule counts nights, not frames.

Work is tiered by what it needs, and the two cheapest tiers need no clear sky at all:

| Tier | Precondition | Yields |
|---|---|---|
| **T0** | **None — indoors, any weather** | Sensor constants (Model 6 complete), gphoto2 capability truth, saturation values, focus-reach answer. **Unblocks D1.** |
| **T1b** | **Any night, even cloudy** | COC alone. "The cheapest data in the plan." |
| T1 | One clear night, own gear + Barlow | TED, FQE without positions, COC in parallel |
| T2 | An electronic focuser | FOP, FDF, FQE with full provenance |
| T3 | Rental window | Exposure bandit, tracked TED negatives, long-exposure FQE |

Two sequencing decisions follow from this and are worth stating on their own:

**BP-2 is the highest-value work available at any moment**, because it requires no clear sky, no rental, no focuser and no working daemon, and every other model in the document consumes its output. The plan says so directly: "If there is one thing to do before reading further, it is BP-2."

**The COC campaign inverts the usual weather logic.** "Go outside when it is cloudy." Clear nights are over-represented in any naturally collected set, and focal loss can reweight a class that exists but cannot create one — so broken, marginal, scudding-cumulus nights are the *most* valuable nights for COC and the least valuable for everything else, and therefore cost nothing to spend that way. The plan also lists the hard negatives this specific site will produce: thin high cirrus, moon glow near a bright moon, the Durham/Chapel Hill light dome, contrails fresh and spread, a tree branch or power line intruding (that is *obstruction*, which the model is also named for and which is trivially easy to collect and easy to forget), and dew on the lens — "photograph it on purpose once before wiping it off."

The rental window is treated as the scarcest resource and deliberately spent **last**, after the software path is proven and the field procedures have been shaken out on gear that can be re-shot for free: "Arriving at a 14-day loan with untested capture code is how the window gets spent on debugging."

### 4.8 What the campaign is waiting on

Three T0 items and one build, none of which need a clear night:

1. **BP-2 photon transfer** on real bias and flat ladders → `[camera] saturation_adu` → D1 closed and Model 6 complete.
2. **BP-1 camera capability probe** — confirm libuvc opens the SV105 and reports an 8-bit container. `scripts/dslr_bench.sh` does the DSLR half, enumerating each body's actual controls rather than trusting the supported-camera list.
3. **BP-3/BP-4** — prime-focus reach in daylight, and plate scale by solving.
4. **The ESP32 focuser** — build, flash, and BP-5 bench calibration against a dial indicator.

And one interim measure that removes the hardware blocker for FQE entirely: **the dial-indicator modification**. A $15 dial indicator clamped to the focuser body with its plunger on the drawtube gives absolute position to 10 µm. At f/3.95 the CFZ is ≈42 µm, so 10 µm resolves the focus zone into four steps — coarse, but *real absolute position*, which is the difference between an FQE-only run and an FOP-eligible one. RB-2 is written around it, and it works tonight.

---

## 5. What was built here

Sole author across all 34 commits: **Ibrahim Shahid** (three git identity variants, same person). Four early commits in Aug 2025 came through Codex-branch PRs on his own repository; everything from Dec 2025 onward is direct.

### Systems and FFI engineering

- **Designed and implemented the C ABI bridge** (`ols_bridge.h`, 712 lines) with an explicit `OLS_BRIDGE_ABI_VERSION`, checked at daemon startup with a refusal on mismatch — because "a bridge built from one revision and a daemon built from another will not fail to link, they will silently read each other's structs at the wrong offsets and produce garbage frames." The version history in the header documents what each bump changed and why v2 must still be refused even though no existing struct changed.
- **Extracted OpenLiveStacker's drivers from their web-framework host** by writing 307 lines of shim headers replacing cppcms/booster, so the drivers link into a standalone library.
- **Extended the vendored gphoto2 driver by ~2,775 lines net** across `gphoto2_camera.cpp` and `camera.h`: shutter/ISO ladder parsing with nearest-choice selection, live-view probing and acquisition, separate still vs live-view streaming paths, pause/resume of the stream around a still, `drain_exposure` to collect every file from one exposure, delivery of original camera bytes alongside frames, and `capture_still`/`delete_captured` with non-destructive failure semantics.
- **Wrote a Windows Media Foundation camera backend** (1,144 lines) against the same driver contract.
- **Diagnosed and fixed an ABBA deadlock** between `get_status` and `start_stream` (one held `camera_name`+`streaming` while reaching for `camera`; the other held `camera` while reaching for `streaming`), then wrote a documented total lock order onto `AppState` with `focus_drift` deliberately placed last as a leaf, plus the rule "never hold a `std::sync::Mutex` guard across an `.await`."
- **Built a `CallbackTrampoline`** with deactivate/wait semantics and `stream_state_uncertain` tracking, so SDK worker threads calling back across the FFI boundary during teardown cannot use freed state (commit `56d15464`).

### Measurement and numerical work

- **Rewrote HFR measurement** to eliminate a noise pedestal (removing `(pixel − sky).max(0.0)` from the flux integral) and to bound the aperture adaptively, with a test asserting stability across noise levels and crop sizes.
- **Derived and implemented the V-curve → steps/µm conversion**, including the `R/√2` half-flux-of-a-uniform-disc step, and documented the three neglected terms.
- **Implemented the drift Kalman filter** including the Joseph-form covariance update and the argument for why the augmented-state formulation stays linear.
- **Separated container depth from sensor saturation** throughout the stack — config, camera, frame, quality, background estimator, both API paths, the relay, and the dataset builder.

### ML infrastructure and research judgment

- **Wrote the 38-feature FOP contract** with per-feature provenance and availability, generated to JSON, tested from both the Python and Rust sides.
- **Designed the FQE around leakage avoidance**: per-star crops, log-HFR in pixels, and no focuser position as input, citing Oakden-Rayner 2020 and Kaufman & Rosset 2012.
- **Froze classical baselines before training anything**, and made `direct_hfr` an executed-and-timed row after review flagged a hard-coded 0-MAE entry.
- **Built a promotion gate that publishes its own verdict.** The bundle's `training_info.promotion_gate` block records six named criteria — beats-constant-MAE, meets-classical-rank-floor, real-multi-run-data, and two coverage checks — with the numbers behind each, so a model's ineligibility is a readable fact rather than an absence. [verified rejecting a model during this audit.]
- **Built grouped and temporal split machinery** and argued for reporting both.
- **Retrofitted AstroM3** (Rizhko & Bloom 2024): reused the published `Informer` and `MetaModel` encoders and `CLIPLoss` unmodified, mapping on-mount telemetry onto the photometry branch and capture metadata onto the metadata branch, and wrote a **new `ImageEncoder`** for the third modality — because an imaging rig produces no spectra. Kept the shared space at 512-D so edge and cloud embeddings remain comparable.

### Written technical work

- **`achernar_v4.pdf`** (108 KB of extracted text) — the system architecture, AI/ML design, and autonomous-control specification, authored Dec 2025, revised Mar 2026, listing him as Lead Architect.
- **`achernar_data_collection.tex`** (2,703 lines) — the data-collection protocol analyzed in §4: a capability audit written by reading the tree, an instrument inventory with per-body tethering risk, seven numbered blocking defects (D1–D7) ordered by how much data each spoils, model-by-model budgets expressed in *runs* rather than rows, and five appendices holding six bench procedures, nine observing runbooks with abort and acceptance criteria, a machine-readable session-log schema, a post-session ingest checklist, and a table of standing decisions with rationales. It is the second-largest piece of technical writing in the project and the one that governs what happens next.
- **`achernar_audit.tex`** (463 lines) — an end-to-end wiring audit with five numbered fixes and a verification table naming what was and was not run.
- **`BUILD-PLAN-fall-2026.md`** — the hardware and observing plan (see §4 and §8).
- **`docs/fqe-implementation-research.md`** and **`docs/astronomical-quality-metrics-research.md`** — a corpus audit of 51 astronomy/ML papers in `astronomy_texts/`, classified by relevance, with a DOI map and a close-reading set mapped to specific implementation decisions.

### Where AI assistance appears

The repository is transparent that parts of the implementation were produced with AI assistance in a sandbox without a Rust toolchain — `HANDOFF-next-session.md` names exactly which tasks (A, B, C, D), what was verified by porting logic to Python and running it, and what could not be. The technical substance that is unambiguously the author's: the problem definition, the ABI and lock-order design, the physical models (drift filter, V-curve derivation, saturation semantics), the ML contracts and leakage arguments, the defect diagnoses, the decision to descope three of the nine models, and every judgment about what the evidence does and does not support.

---

## 6. Technically difficult parts

**1. Making a measurement that a machine-learning label can rest on.** The problem: HFR is a physical width, and every shortcut that makes it easier to compute biases it. Clipping negatives rectifies noise into a pedestal; an unbounded aperture accumulates noise-only annuli; a clipped star core is flat and therefore measures *smaller* than the truth — biasing the brightest, highest-weight samples toward "sharper than reality". Nontrivial because each of these fails *silently* and produces plausible numbers. Addressed by: unclipped integration, adaptive aperture bounding, and a three-state saturation gate whose "not performed" value propagates into the dataset builder as a *drop* rule. **Solution status: solid for 8-bit streams** (where the container is the clip point, verified live); **unresolved for 16-bit** until `[camera] saturation_adu` is measured per body — which is exactly what `tools/photon_transfer.py` exists to do, and it has not been run on a real sensor.

**2. Preserving derived state that cannot be recovered later.** The FQE's input is pixels — if the contract drifts, the crops are on disk and the dataset can be rebuilt. The FOP's input is derived state that exists only at the moment of capture; a feature not logged during a sweep cannot be recovered from that sweep afterwards, at any cost. Nontrivial because the cost of getting it wrong is invisible until training, months later, on data that cannot be re-collected. Addressed by settling the 38-feature contract *before* the first run, with provenance and availability per feature, and cross-language drift tests. **Solution status: the contract is complete and tested; 12 of 38 features are not yet producible**, and the reasons are recorded rather than discovered later.

**3. Concurrency across an FFI boundary into vendor SDKs.** A vendor SDK's worker thread calls back into Rust; a disconnect can land mid-callback; an FFI call already inside an SDK cannot be cancelled. Addressed by: `stream_generation` tagging on every frame so post-invalidation frames are dropped, a `camera_observation_barrier` around publication, a `CallbackTrampoline` with deactivate-and-wait, discovery probes that *abandon* rather than cancel, and a documented total lock order. **Solution status: the specific ABBA deadlock is fixed and the ordering documented; the trampoline work is compiled and unit-tested but not exercised against a real vendor SDK under teardown.**

**4. Proving one exposure's metadata belongs to that exposure.** A settings write can land while an exposure is already in flight, so the returned frame carries the previous settings under the new metadata — unrecoverable after the fact, because nothing in the record says it happened. Addressed by requiring two completed frame counters after a settings write, factored into a pure testable gate, and by failing rather than substituting a stale frame. **Solution status: implemented and unit-tested; the failure mode it prevents has not been reproduced against real hardware.**

**5. Joint state/parameter estimation under collinearity.** The temperature coefficient k is only identifiable across varied thermal conditions, because under steady cooling `k·ΔT` and `v·dt` are collinear. Nontrivial: an unlucky first night can fit a confident and wrong k. Addressed by the augmented-state filter with a slow random walk on k so it adapts rather than freezing, the Joseph form to survive an all-night run numerically, and the estimation/decision split so the filter accumulates history from the first session even with no threshold configured. **Solution status: the filter is implemented and tested (17 unit tests), and a Python port reportedly recovered k = −19.98 against a true −20.0 with no threshold set. It has never seen a real thermal night.**

**6. Getting usable focus data from an untracked telescope.** The Z114 Dobsonian has no tracking, so at 0.66″/px on the celestial equator a star trails 22.8 px/s and any exposure long enough to measure would smear. The plan's solution is a genuine piece of observational reasoning: point at the north celestial pole, where cos δ = 0.011 and the trail rate falls to 0.26 px/s — 0.13 px in a 500 ms exposure. An untracked tabletop Dob becomes, for focus-sweep purposes, a tracked telescope. Recentering every 30–45 minutes as Polaris walks its 39′ circle. The plan further notes to use *field stars around* Polaris rather than Polaris itself, since mag 2.0 clips an 8-bit sensor at best focus and a clipped core reads as artificially sharp. **Solution status: reasoned, arithmetically checked, never executed.**

**7. Choosing the focuser interface to avoid writing a driver.** The plan records reading `OpenLiveStacker/src/focuser.cpp` and finding it is 46 lines whose entire driver list is `{{"alpaca:remote", "auto"}}` — while the Rust side has enum variants naming `moonlite` and `usb` drivers that do not exist, which is precisely what `capabilities.rs` reports as `disabled`. The conclusion: an Arduino + Moonlite protocol focuser would require writing a new C++ driver and wiring it through the ABI, in the riskiest part of the codebase; an ESP32 running myFP2ESP32 speaks Alpaca natively, and `alpaca_focuser.cpp` (178 lines) is already compiled. **Solution status: a decision made from reading the code, not yet built.** The plan also notes the Alpaca focuser interface carries `Temperature` and `TempComp` as standard properties, so one build closes both D3 (no focuser position) and D5 (no environment sensor).

---

## 7. Technical maturity

Achernar is not one kind of project, so it needs three answers.

**As a telescope-control application:** it works against a simulator and is architecturally ready for hardware, but no one has driven a real camera or focuser with it in evidence available here. The camera path is mature enough that a driver-level failure would be reported honestly rather than silently — the capability report, the discovery timeout, the stream-generation tagging and the fresh-frame gate all exist to make hardware failures legible. What a second person could do with it today: run the daemon, connect the simulator, watch a live view, run an autofocus sweep, and get a database. What they could not: use it on their own telescope without first proving their driver opens, since none has been.

**As scientific-instrument software:** the measurement layer is the strongest part of the project and is close to defensible. HFR is computed by a documented method with tested numerical properties, star rejection is itemized, and invalid measurements are `null`. What is missing before its output could support a claim: no comparison against an independent measurement of the same frames (SExtractor/SEP on the same data, or a Bahtinov mask as human-readable ground truth), and no FITS file read back by an external reader. The classical path is the *control authority* today and would remain so under the project's own promotion rules.

**As machine-learning research:** the infrastructure is built, tested and executable ahead of the data it will consume. The full lap runs [verified end to end on synthetic data during this audit] — dataset builder, grouped/temporal splits, heteroscedastic two-head model with a RankNet auxiliary, post-hoc calibration fit on validation and reported on test, frozen baselines, ONNX export with numerical parity, hash-bound bundle, runtime-eligibility gate, exact Rust↔Python agreement, and a machine-readable promotion verdict that blocked the artifact it had just produced. That is working research infrastructure, and it is unusually complete for a project that has not yet trained on real data.

Nothing has yet been *demonstrated about a model*: no held-out sky results, no evidence that a learned estimator beats classical HFR at anything. The lap has been built and run once on synthetic data, where it failed its own gate — machinery working, not a result. What that buys is that the first real dataset has somewhere to go the night it exists: §4's v0 tier is defined as exactly this handoff, and the plumbing it is meant to prove is already proven.

**As a data-acquisition program:** the protocol is complete and the ingest half is implemented (§4). The importer, the photon-transfer tool, the environment and disturbance tables, the grouping keys and the reversal-sweep control all exist and are tested; the field procedures, budgets and acceptance criteria are written to the level of a runnable checklist. What has not happened is the observing itself.

**As hardware:** nothing physical has been built yet. The focuser exists as a sized design — CFZ at f/3.95 computed as 4.88·λ·N² ≈ 42 µm, a step-size spec of ≤ CFZ/5 ≈ 8 µm, and a 3:1 GT2 reduction with 1/16 microstepping landing at 2–3 µm/step — plus a print list, a material decision (PETG/ASA over PLA, because PLA creeps under belt tension and will not survive a car trunk in the sun), and a bill of materials under $100. The dial-indicator interim (§4.8) makes FQE and coarse FOP runs collectable for $15 before any of it is printed.

---

## 8. Intended system, direction, and what "done" means

### A. What exists now, as part of the eventual system

The daemon's acquisition, vision, autofocus, persistence and API layers are not scaffolding for the target system — they *are* its bottom half, and the ML layer is designed to attach at points that already exist: the `SharedObservation` bus carries `fqe_*` shadow fields today; `fqe_runner` resolves the highest runtime-eligible semver from `models/`, so a trained `1.1.0` supersedes the reference `1.0.0` with no code change; `model_registry` generalizes that resolution for the models that do not exist yet; `frames.quality_json` already carries the keys the FOP contract claims to read.

### B. What is being built now

Three tracks, running in parallel:

1. **The Windows rig-agent port** (§3D) — new camera backend, build tooling and CI, none of it yet run.
2. **The DIY motorized focuser** — designed and sized, parts under $100, chosen so it connects over Alpaca with no new driver code.
3. **The observing campaign** (§4) — the piece that turns the built pipeline into results. Its protocol is complete, its ingest tooling is implemented and tested, its per-model budgets and yields are computed, and its first nights are the immediate next step. The two tiers that need no clear sky (T0 bench calibration, T1b cloudy-night COC) are available at any time; the focus-sweep tiers need clear nights, of which the plan budgets ~12 by Nov 1 and ~25 by Jan 1.

Per `BUILD-PLAN-fall-2026.md`, dated 2026-08-07, the campaign runs Aug 2026 → Jan 2027 with a portfolio freeze at Nov 1.

### C. What remains

The plan is explicit about scope, and about what it cuts. Of nine specified models:

| Model | Status per plan | Blocker |
|---|---|---|
| 6 — Background/SNR | Target: **complete by Nov 1** | Bench only, weather-independent; needs `tools/photon_transfer.py` run on real bias/flat ladders |
| 1 — FQE | Target: trained by Nov 1, 60–80 runs | D1 (saturation undeclared), D2, and a motorized focuser to make sweeps fast enough |
| 5 — TED | Target: dataset complete by Nov 1 | 2 untracked nights at three declinations |
| 4 — COC | Target: 15+ labelled nights by Nov 1, trained by Jan 1 | Needs the most *distinct* nights, so it starts first; runs on cloudy nights |
| 2 — FOP | Target: Nov–Jan, ~200 runs | **D3** — no focuser with readable absolute position. Hard blocker; the label *is* a signed step count |
| 3 — FDF | Target: Nov–Jan, 1–2 thermal nights | D3 + D5 (no ambient sensor) |
| 7 — Exposure bandit | **Cut** | Needs a tracked mount and 300 s exposures |
| 8 — Filter Order Recommender | **Cut** | Needs a filter wheel and seven filters; `hardware/filter_wheel.rs` is `todo!()` |
| 9 — Tri-Modal Embedding | **Cut** | Needs plate solving (D6) plus ~10k solved frames |

Three models are cut (7, 8, 9), five are kept, and Model 6 is the sensor-constant foundation the other five sit on. The stated reasoning: "Five of nine models, and they're the five that form one coherent story: the complete focus stack, plus the two sky-condition models, sitting on sensor constants you measured yourself. That reads far better than nine half-finished models."

Independent of the models, reaching the intended system also requires: a real INDI client (the vendor-neutrality argument against ASIAIR depends on it, and it is a 12-line stub); debayer and calibration application (without them an OSC or DSLR frame yields HFR measured on a Bayer mosaic — garbage labels); plate solving (the plan's recommendation is to shell out to a local `solve-field` rather than implement one); and per-user authentication replacing the shared dev relay token.

### D. What "done" looks like, concretely

A completed Achernar would let someone do this, start to finish:

1. Install a signed **Achernar Rig Agent** on a computer at the telescope. It enumerates what is physically attached — camera, mount, focuser, filter wheel — through vendor SDKs, INDI, or Alpaca, and reports which drivers its build actually contains.
2. Open **achernar.com** in a browser anywhere. The rig has an outbound WebSocket held open, so no port is forwarded; the same command set reaches it whether the user is on the LAN through the desktop app or across the internet through the relay.
3. Pick a target. The system plate-solves, slews, syncs, and confirms pointing from a *solved* position rather than a claimed one.
4. Run autofocus. Instead of a 25-position sweep, the **Focus Offset Predictor** reads the current frame plus rig and thermal state and proposes a signed step count with an uncertainty; the **safety supervisor** checks it against per-rig position limits, per-move and per-run travel budgets, and a minimum confidence; the **action arbiter** resolves it against anything else wanting the focuser. If confidence is low, it falls back to the classical sweep. Every decision is logged with the model version and hash that produced it.
5. Image. The **Focus Drift Forecaster** tracks the tube's cooling curve and says *when* to refocus rather than refocusing on a timer. The **Cloud/Obstruction Classifier** notices a cloud bank and pauses. The **Trailing Detector** flags a guiding excursion. Each publishes a calibrated confidence, and each is bounded by the same supervisor.
6. Every exposure lands in the durable archive as native bytes plus derived FITS with real calibration headers, catalogued in SQLite, and syncs to the cloud when connectivity returns.
7. New data flows back into training. A retrained model is published as a hash-bound bundle, runs in **shadow** beside the classical measurement, and is promoted only when it beats the frozen baselines on grouped, held-out real data — a decision a person makes and the registry records.

**The concrete near-term definition of done**, and the one the plan actually optimizes for, is much smaller and much sharper: *a plot of predicted best-focus position against true best-focus position, on data collected by a focuser that was designed, printed, wired and flashed for the purpose.* That closes CAD → mechanics → firmware → a standard protocol → a Rust daemon → a trained network, and it is measured rather than asserted.

### E. Is the trajectory coherent?

Yes, with one structural caveat.

The existing components are pieces of the target system, not experiments: the ONNX serving path, the shadow fields on the bus, the model-bundle format and its gate, the grouped-split machinery, the durable archive, and the relay all exist *because* the intended system needs them, and reaching the endpoint is mostly a matter of adding missing components rather than redesigning existing ones. The extension points are real and identifiable: `fqe_runner::resolve_latest_model`, `ModelRegistry::scan`, `SharedObservation`'s `fqe_*` block, `FocusModelsConfig::offset_model_path`, `frames.quality_json`, and `frame_measurements` (append-only, versioned by pipeline/model/preprocessing).

The caveat is **inversion of effort**. The control and serving layers for nine models were built before any model existed, and four of the nine are now cut. `filter_wheel/` (466 lines, five modules, unreferenced), `mount/`'s bandit and dither policy, `models/model_registry.rs`, and the eleven empty storage repos are all consequences: 65 `struct is never constructed` and 22 `function is never used` warnings on a `cargo check --tests` [verified]. None of it is wrong, and none of it is on the critical path. The project's own most recent documents (`MARKET-AND-NEXT-STEPS.md`, `BUILD-PLAN-fall-2026.md`) diagnose exactly this and reorder toward data, which is the right correction.

---

## 9. Where the chain stops today

The progression this project follows:

```
specification → daemon + hardware abstraction → measurement → closed-loop control
   → durable provenance → dataset construction → training → calibration → export
   → shadow serving → promotion → autonomous operation
                                    ↑
                    data-acquisition program (§4)
     sensor calibration → focuser → runbooks → observing nights → ingest
```

| Stage | Status |
|---|---|
| Specification | **Exists** — v4 architecture doc, data-collection protocol, wiring audit, build plan |
| Daemon + hardware abstraction | **Exists**, compiled and running; real hardware unproven |
| Measurement (background, stars, HFR, trailing) | **Exists and demonstrated** on real sky images |
| Closed-loop autofocus control | **Exists and demonstrated** against the simulator |
| Durable provenance (frames + assets + reconcile) | **Exists and demonstrated** |
| Dataset construction | **Exists and demonstrated** — 296 crops built from a daemon-produced database |
| Training / calibration / export | **Built and verified on synthetic data** — trains, calibrates, exports with FP32 parity, publishes a hash-bound bundle. Never run on real data |
| Shadow serving | **Built and exercised** — the daemon loads a bundle and runs real inference, matching the training pipeline's output exactly. Predictions are meaningless because the weights are |
| Promotion | **Gate implemented and demonstrated rejecting a model.** Never reached by a real one |
| Autonomous operation | **Specified only** |
| Data-acquisition program | **Protocol complete; ingest tooling built and tested; observing not yet begun** (§4) |
| Physical instrument | **Designed and sized; not built.** A $15 dial-indicator interim makes FQE runs collectable without it |

**The functioning chain stops between "dataset construction" and "training on real data."** It stops there for instrumental reasons rather than software ones — a focuser with readable absolute position and a measured sensor saturation point are what turn a sweep into trustworthy labels — and both are T0/T2 items in §4 with written procedures behind them, not open research questions.

**The intended chain ends at** a rig that focuses, tracks conditions, and adapts its own settings under learned control bounded by a deterministic supervisor, with every model promotion earned against a frozen classical baseline on held-out real nights.

---

## 10. Gaps between here and there

**Gap 1 — Sensor saturation is undeclared (D1).**
*Missing:* a measured `saturation_adu` per body per ISO, plus read noise and gain.
*Why it matters:* until it is set, every 16-bit frame carries `saturation_gate = not_performed`, and the dataset builder correctly *drops* those frames. No FQE training set can be built from them. It is also the whole of Model 6.
*Status:* **actively addressed** — `tools/photon_transfer.py` (347 lines, 20 passing tests) implements the paired-flat photon transfer curve, `g = [(S₁+S₂) − (B₁+B₂)] / [var(S₁−S₂) − var(B₁−B₂)]`, taking saturation from where the variance-vs-signal curve turns over rather than from the largest value in the file, refusing demosaiced input, and measuring the four CFA phases separately. It has never been run on real frames.
*Where it belongs:* bench measurement, indoors, weather-independent.
*Closure:* a device-profile JSON and a PTC plot showing the knee, plus `saturation_gate_applied: true` in a persisted 16-bit frame's `quality_json`.

**Gap 2 — No focuser with readable absolute position (D3).**
*Missing:* a motorized focuser reporting absolute position, with measured steps/µm, backlash, and repeatability.
*Why it matters:* the FOP's label *is* a signed step count, and the FDF's coefficient is Δposition/Δtemperature. Both are unfittable without it. The FQE can still be trained (it is deliberately pixels-only) but each run's provenance is degraded — without positions you cannot order samples within a run or verify the sweep was monotone.
*Status:* **actively being built** — ESP32 + myFP2ESP32 + DRV8825/TMC2209 + NEMA 17 + DS18B20 over Alpaca, chosen specifically so that no new C++ or Rust is required. A stated success criterion is that `POST /connect` with `driver: alpaca` and `POST /focuser/moveto` require **no code changes** — "if they are, that's the finding."
*Where it belongs:* fabrication (3D printing, wiring, firmware flashing) plus bench calibration against a dial indicator.
*Closure:* a `focuser_profiles` row with measured steps/µm, a backlash figure before and after compensation, and 20 round-trip repeatability measurements — then a converged autofocus run on the sky whose fitted minimum is interior to the sweep.

**Gap 3 — No debayering or calibration application.**
*Missing:* `processing/debayer.rs` and `processing/calibration.rs` (4 lines each).
*Why it matters:* HFR measured on an undebayered Bayer mosaic from an OSC or DSLR is measured on a checkerboard, not a star. This corrupts labels at the source and blocks any DSLR-based dataset.
*Status:* **implied by the architecture, not yet planned in detail.** Not blocking for the SV105 path, which the plan makes primary.
*Closure:* a debayered frame whose HFR matches the same star measured on the extracted green plane, within noise.

**Gap 4 — No plate solving (D6).**
*Missing:* a solver backend; `plate_solver.rs` always returns `Unsupported`.
*Why it matters:* blocks verified pixel scale, target identity per frame, airmass and moon-separation covariates from a *solved* rather than claimed pointing, the `plate_solve_events` table, and Model 9 entirely.
*Status:* **planned**; the recommendation is to shell out to a local `solve-field` install rather than implement a solver.
*Closure:* a `plate_solve_events` row with a WCS whose pixel scale agrees with the optical prediction.

**Gap 5 — No independent validation of the measurement layer.**
*Missing:* comparison of the daemon's HFR against SEP/SExtractor on the same frames, and against a Bahtinov mask as human-readable ground truth.
*Why it matters:* the entire label chain rests on this measurement. It is internally consistent and tested against synthetic PSFs, but has never been checked against an independent implementation on real data.
*Status:* **partly planned** — the build plan lists printing a Bahtinov mask as "independent, human-readable best-focus ground truth" and calls the mask agreeing or disagreeing with the model a thing worth capturing.
*Closure:* a plot of daemon HFR against SEP HFR on the same frames, and a Bahtinov-mask best-focus position compared against the fitted V-curve minimum.

**Gap 6 — The Windows port has never been built or run.**
*Missing:* one green CI run and one SV105 opened through Media Foundation.
*Why it matters:* the whole observing campaign assumes the SV105 on a Windows rig computer.
*Status:* **actively being implemented**; the workflow file is untracked, so it has never executed.
*Closure:* a green `windows-daemon` run, and a `smoke-api.ps1` artifact directory showing all eight stages passing against a real SV105 with SHA-256 capture provenance.

**Gap 7 — Observing time on the sky.**
*Missing:* the runs themselves. §4.7 puts v1 FQE at 60 runs ≈ six productive clear nights at ~10 runs each, and v2 at fifteen.
*Status:* **actively opening.** This is the one gap in the list that is not a piece of engineering but a schedule, and it is downstream of Gaps 1, 2 and 6, all of which have written procedures. The plan's realistic estimate for Piedmont NC is ~12 usable sessions by Nov 1 and ~25 by Jan 1; the COC campaign (§4.7) additionally converts cloudy nights into data rather than losing them, and the T0 bench work proceeds in any weather.
*Closure:* `models/focus_quality/1.1.0/` with `"trained": true`, `"runtime_eligible": true`, `"promotion_status": "shadow"`, a non-null SHA-256, a fitted `calibration.json`, and a held-out-by-night evaluation showing the model beating `constant` on MAE and approaching `direct_hfr`/`tenengrad` on within-run Spearman ρ. The plan states the honest alternative in advance: "If the model doesn't beat classical HFR on a held-out night, say so — that's a real result and it's more credible than a claim."

---

## 11. Validation and evidence

**Automated tests — 194 passing across three suites [all verified].**

| Suite | Count | What it validates | What it does not |
|---|---|---|---|
| Rust (`cargo test`) | 120 | HFR stability across noise and crop size; blank frames returning null not zero; saturated/edge/blended/elongated star rejection; a clipped 14-bit star rejected *when saturation is declared* and the gate reporting `NotPerformed` when it is not; defocused wide stars measured via adaptive crop; local sky matching NumPy's median for even samples; night-boundary handling including UTC leakage; capture-store partials, missing files, and rejection of a capture with no native asset; model-registry semver selection and hash-tamper detection; FQE aggregation matching the Python `aggregate_stars`; calibration refusing a mismatched model hash | Any behavior against real hardware; any vendor SDK; the RAW decode path against a real LibRaw |
| Python ML | 54 | Moffat fitted FWHM against the analytic value, and that 2·HFR is *not* it; HFR invariance to noise level and crop size; near-Nyquist PSFs measured rather than exploding; source gates; the FOP contract's 38 features matching the committed JSON; dataset-builder saturation gating and the permissive-mode stamp; grouped and temporal split disjointness | Whether the labels are correct on real sky data |
| Python tools | 20 | Photon-transfer gain/read-noise recovery on synthetic ladders; session-import idempotency, RAW+JPEG pairing as one exposure, EXIF→log→mtime metadata fallback | Whether the method works on a real sensor |

**Cross-implementation parity.** Three separate mechanisms exist because the same quantity is computed on both sides of a language boundary: `aggregate_stars` is asserted identical in Rust and Python; `PARITY_RUST`/`PARITY_PY` lines are printed for the same deterministic crop so the ONNX graph's behavior is diffable; and `night_id` carries a comment requiring it stay identical to the importer's copy. **[verified: both halves were run against a model trained during this audit and agree exactly — `mu_log_hfr=0.792735 hfr_px=2.209430 sigma_log_hfr=0.861523` from both. The ONNX export also self-checks against the PyTorch model at `max|Δmu| = 1.79e-07`.]** This is the right kind of check and it passes — but note that "identical to each other" is not "correct".

**The promotion gate, demonstrated.** [verified] The published bundle's `training_info.promotion_gate` block carries six named boolean criteria plus the numbers behind them, and the pipeline recorded `ready_for_promotion_review: false` on a model it had just trained. This validates that a bad model cannot enter the live path; it does not validate that the criteria are the *right* criteria, which only real held-out nights can settle.

**Sandbox verification recorded in `HANDOFF-next-session.md`.** 61 SQL statements run against real SQLite (fresh apply, idempotent re-run, FK rejection, NULL round-trips, upsert-refines-in-place); the Kalman filter ported to Python and run; the V-curve derivation round-tripped across four scope/camera combinations; the C++ decode block run against a stub LibRaw covering crop offsets, an odd left margin correctly reporting GRBG rather than mislabelling RGGB, sub-region phase preservation, and every rejection path. This is careful work, and the document is explicit about its limit: none of it compiled, and a stub LibRaw cannot settle Bayer phase on a real frame.

**What has no independent validation at all:** every physical measurement the system produces. HFR has never been compared to another implementation on real data. No FITS file has been read by astropy or DS9 in evidence. No sensor constant has been measured. No focus curve from a real telescope exists. The drift filter has never seen a real cooling night. The pointing model, both bandits, and the filter SNR scorer have never been exercised at all.

**Validation that is not yet relevant:** held-out model evaluation, uncertainty calibration coverage, and shadow-vs-classical comparison all require a trained model on real data. Their absence is expected, not a defect — the machinery for all three exists and runs.

---

## 12. Technical evolution

**Feb 2025 – Sep 2025 — Django scaffold.** Initial requirements, a Django project, a "Hello" endpoint, two Codex PRs, then a Tauri desktop shell.

**Dec 2025 – Jan 2026 — the daemon becomes the center.** The v4 architecture specification is written (dated Dec 19, 2025). OpenLiveStacker is vendored (`91b8283b`, `6c4b3f62`), and the repository is reorganized around a Rust daemon rather than a Django app. *Effect:* the project stops being a web application with a hardware story and becomes a hardware system with a web story.

**Feb – Apr 2026 — the bridge and the schema.** `8b03617e` adds the OLS bridge and FFI; `8e917b9d` (109 files) adds the daemon core and DB migrations; `26979c3e` adds ML stubs; `0a29eb40` (32,562 insertions) lands the C++ bridge properly with CMake discovery, rpath handling and vendor SDK drivers. *Effect:* a working FFI boundary, and the ABI-version discipline that follows from it.

**May – Jul 2026 — control surfaces and cloud.** Mount and focuser integration (`56995be8`), live stream and status (`226d40a9`), the cloud relay client (`c62480d8`), daemon control (`9a117357`), then `3b67964c` (84 files, +7,466) adding autofocus, the filter-wheel control layer, and the API expansion. *Effect:* the daemon becomes reachable from a browser through a reverse tunnel, and the nine-model control scaffolding is laid.

**Jun 13, 2026 — the wiring audit.** `achernar_audit.tex` finds five issues (a broken cache backend for the relay handoff, a split-brain frontend API layer, an untracked `src/lib` directory) and fixes them. *Effect:* the audit habit starts, and the verification table naming what was *not* run becomes a pattern.

**Jul 2026 — three disagreeing FQE contracts.** The audit finds one input described three incompatible ways with no test comparing them. `rebuild_fqe.py` is written to resolve it, and `models/focus_offset/0.3.0/` is **deleted** — its version number implied a trained artifact that never existed, and its ten-feature `metadata.json` contradicted a 38-feature specification. Replaced by `ml/contracts/fop_features.py` plus tests on both sides. *Effect:* the project's central discipline — that a contract must be one artifact with a test defending it — is established by having been burned.

**Aug 5, 2026 — RAW decode, and refusing a convenient function.** `54ce3313` adds still capture and live view to the gphoto2 driver (+894 lines to that file alone). The RAW decoder deliberately does *not* call OLS's existing `load_libraw()`, because that function black-subtracts and multiplies by `65535/(maximum − black)` in integer arithmetic — 4 on a 14-bit Canon where the exact factor is 4.572. The written reasoning goes past the truncation: the multiply itself severs ADU from electrons (signal scales by k, shot-noise variance by k², so any gain fitted from the frame is wrong by k), k is body-dependent so two cameras stop being comparable, and subtracting black into an unsigned type clips the negative half of the noise distribution around the pedestal, making read noise unfittable. `load_libraw()` is left untouched to serve the preview stream, "which wants a stretched image and is welcome to one." *Effect:* pixels stay in sensor ADU and calibration constants travel in the FITS header.

**Aug 5, 2026 — the `max_adu` bug (D-series).** `quality.rs` derived `max_adu` from the stream format and hardcoded 65535 for 16-bit, so the 0.98 gate sat at 64224 on a body whose data stops at 16383 and **never fired**. Every clipped, flat-cored star flowed into HFR. Fixed by splitting container full-scale from sensor saturation and adding `SaturationGate::NotPerformed`. The consequence is recorded rather than papered over: *"the FQE frames already collected carry the broken labels. They are not retroactively fixable — the gate genuinely never ran — but they are now identifiable."* *Effect:* the strongest single instance of the project's habit of preferring an honest unknown to a plausible default.

**Aug 5, 2026 — the drift-filter estimation/decision split.** A first version refused to construct the forecaster without optics. It was revised after review on three grounds, all recorded: it killed the estimation half to protect the decision half; the expensive half was the collateral damage, since k needs varied thermal conditions accumulated over sessions; and it violated its own principle by substituting `off` for `unknown` without recording it. Two stale artifacts were deleted in the same pass — `kalman_state.json`, which tried to be tuning constants and learned per-rig state at once and could serve neither, and `drift_model_path` from the config, since FDF is a Kalman filter and there is no ONNX file. *Effect:* the "`Some(false)` and `None` are different claims" pattern generalizes across the API.

**Aug 7–10, 2026 — pivot from architecture to data.** `MARKET-AND-NEXT-STEPS.md` and `BUILD-PLAN-fall-2026.md` are written. The diagnosis at the time: the specification's §7/§8 control scaffolding is essentially complete while nothing has been captured, so priority should invert. Four of nine models are cut, and the remaining five are ordered by data cost. `8c72ff11` immediately lands `tools/` — the photon-transfer calibrator, the sidecar importer, and their tests — plus migration 006's environment and disturbance tables. *Effect:* the project reorganizes around acquiring data, and the ingest subsystem in §4 is built over the following week.

**Aug 13, 2026 — compile and harden.** `56d15464` (+1,312 lines in `api/camera.rs` alone) adds `ApiError`, the `FreshFrameGate`, the `CallbackTrampoline`, stream-generation handling, and migration 007 adding `night_id`/`session_id` for grouping. It also deletes `focus/focus_features.rs` and `focus/focus_inference.rs` — the 15-field struct with a `raw: Vec<f64>` escape hatch and a `todo!()` tensor builder that the FOP contract replaced. *Effect:* the tree compiles, the tests run, and the dead FOP implementation is gone rather than left to mislead.

**Aug 13, 2026 onward (uncommitted) — the Windows port.** See §3D.

---

## 13. Limitations and unresolved problems

### Major — these bound what the implemented portion can claim

1. **No model has been trained on sky data.** Any statement about the system's *intelligence* currently describes the specification and the pipeline, not a result. The pipeline itself is demonstrated (§3A).
2. **No hardware has been driven.** Every camera, mount and focuser path is unproven against real devices. The FFI concurrency work is a defense against failure modes that have not been observed here.
3. **`saturation_adu` unset.** [verified at startup: the daemon warns that the gate cannot run on 16-bit streams.] Blocks FQE labels from any 16-bit source until BP-2 runs. 8-bit sources — including the SV105, the plan's primary camera — are unaffected, because the container is the clip point and the gate fires by construction [verified: `saturation_gate_applied: true` on every frame of the audit's test sweep].
4. **LibRaw not compiled into the current build.** [verified] Path B archives RAW containers and derives no FITS on this host.
5. **Bayer phase unverified on a real frame.** The one thing a stub LibRaw cannot settle, and the project says so.
6. **No independent check on the measurement layer.** Internally consistent and untested against another implementation on real data.
7. **The Windows port has never been compiled or run.**

### Secondary — worth fixing, but they do not change the interpretation

- **Eleven empty storage repositories.** The persistence that runs is inline; the repo layer is a naming skeleton with a lowercase-struct-name warning attached to each.
- **`model_registry.rs` is unused.** Built and tested, zero external references; `fqe_runner` has its own resolver.
- **Substantial dead code.** 65 `struct is never constructed` + 22 `function is never used` warnings [verified]. `ai/` (12 lines across three files) is superseded by `vision/` and the project's own notes say to delete it.
- **README drift.** The README describes a repository layout (`packages/web/`, `achernar_backend/`, `docs/API.md`, `docs/HARDWARE.md`, `docs/AI_ARCHITECTURE.md`) that does not exist, lists a daemon port of 3030 against a default that is also 3030 but a documented example of 8080 elsewhere, and presents nine models as features. It is the least reliable document in the repository — which matters mainly because it is the first one a reader sees.
- **CI references a non-existent module.** The advisory ML import check lists `ml.models.focus_drift_kalman`, which is not in `ml/models/`. Advisory, so it does not fail.
- **Repository weight.** 4.5 GB working tree, 895 MB `.git`, driven by 1.6 GB of vendor SDKs and a vendored OpenLiveStacker of ~1.7 M lines including its externals. The audit document recommends a submodule and Git LFS.
- **Security posture is development-grade.** A shared `achernar-dev-token` relay secret with a hardcoded default, `AllowAny` equipment endpoints, and a null CSP — all identified in the audit's Phase 4.
- **Commit hygiene early on.** `"stuff"`, `"f"`, `"Commit"`, `"models"`. Recent messages are detailed and specific.

### Expected incompleteness — absent because development has not reached them

Filter wheel hardware layer, environment sensors, INDI client, debayer/stacking/calibration, plate solving, generic ONNX runner, eight of nine model training paths, the AstroM3 cloud service, and the Django AI engine. Three of these belong to models that were deliberately cut. None is a failure; all are unstarted stages of a project mid-build.

---

## 14. Actual technical ambition

Achernar is trying to build **the first amateur-astronomy control system whose focusing, condition monitoring, and exposure decisions are made by learned models with calibrated uncertainty, bounded by a deterministic safety supervisor, and promoted only after beating frozen classical baselines on grouped held-out data.**

Every existing product in the space — free, prosumer, and the $1,495 institutional tier — automates by script and heuristic. The four things Achernar's specification asks for that none of them ships are learned closed-loop control with calibrated uncertainty, a deterministic supervisor bounding model actions, shadow-mode evaluation with explicit promotion criteria, and one shared perception bus feeding focus, filter and mount subsystems from a single vision loop. Of those four, **the supervisor, the shadow path, the promotion gate and the shared bus are implemented** — `control/safety_guard.rs` runs on every focuser move, `SharedObservation` carries `fqe_*` fields beside the classical measurement, `runtime_eligible`/`promotion_status` gate every bundle, and `vision_loop` is the single producer. **The learned control is the part that does not exist**, because it requires data that requires an instrument that has not been built.

The existing components are being built in the order that makes the last one possible. The daemon exists to drive hardware; hardware exists to collect labeled sweeps; the durable archive and `night_id`/`run_id` grouping exist because a leaked split would make the eventual model's numbers meaningless; the FOP contract exists because features not logged during a sweep can never be recovered from it. When the chain closes, the system can focus a telescope from a single frame instead of a multi-minute sweep, know when to refocus from a thermal model rather than a timer, and prove it earned the right to do so.

**How much of that ambition is in the repository today:** the whole lower half; the whole serving, evaluation and promotion apparatus of the upper half; and the ingest half of the data-acquisition program that feeds it. What is not there yet is the trained content of the upper half — which is what the campaign in §4 exists to produce.

**Is current development moving toward it:** yes, and the direction changed to make it so. The Aug 2026 pivot — cutting three models, ordering the remaining five by data cost, moving the target platform to Windows to make the SV105 usable, designing a focuser specifically so no new driver code is required, and writing a per-model collection protocol with computed yields — is a plan built by reading the code and counting clear nights. It also front-loads the parts that need no sky: sensor calibration, capability probes, the importer, and a COC campaign that runs on the nights everything else is lost to. Whether it arrives depends on things outside the codebase: a $40 focuser working, and roughly 25 clear nights between now and January.

---

## 15. Scale

| Measure | Value |
|---|---|
| Rust daemon | **21,269 lines**, 97 files, 12 top-level modules |
| C/C++ bridge (own code: bridge, shims, Windows MF driver) | **4,200 lines**, 6 files |
| Extensions to vendored OLS drivers (gphoto2, camera.h, uvc, alpaca) | **+2,775 / −61 lines** |
| Python ML | **5,710 lines**, 44 files |
| Python tools | **1,425 lines**, 4 files |
| Django backend | **3,269 lines**, 151 files (78 near-empty scaffold files) |
| TypeScript frontend + shared UI | **3,772 lines** |
| SQL migrations | **592 lines**, 7 files, **25 tables** |
| PowerShell + shell scripts | **1,487 lines**, 10 files |
| Technical Markdown (own, excl. README) | **~3,080 lines** |
| LaTeX technical documents | **3,893 lines** across 3 documents; v4 spec ~108 KB of text |
| **Automated tests** | **194 passing** — 120 Rust, 54 Python ML, 20 Python tools |
| HTTP routes | 34, plus 1 telemetry WebSocket |
| Relay command names | 21 |
| Camera drivers reachable | 10 declared; 7 compiled on this host |
| Models specified / with a bundle / trained | **9 / 4 / 0** |
| FOP feature contract | 38 features, 9 groups; 26 available, 8 degenerate, 3 missing, 1 rejected |
| Literature corpus incorporated | **51 PDFs**, all opened and classified; 52-entry DOI map |
| Git history | 34 commits, Feb 2025 → Aug 2026, one author |
| Repository size | 4.5 GB working tree (1.6 GB vendor SDKs), 895 MB `.git` |
| Dataset built during this audit | 296 crops, 21 frames, 1 run |

---

## 16. External dependencies vs. project work

| Dependency | What it does | What Achernar supplies |
|---|---|---|
| **OpenLiveStacker** (vendored, ~34k own lines) | Camera/mount/focuser driver implementations for ASI, SVBony, ToupTek, UVC, gphoto2, INDI, INDIGO, Alpaca | The C ABI that exposes them to Rust; 307 lines of shim headers replacing their cppcms/booster host so they link standalone; the versioned ABI contract and startup check; **+2,775 lines extending the gphoto2 driver** with still capture, live view, ladder parsing and exposure draining; a new Windows Media Foundation backend against the same contract; a Windows Winsock fix |
| **ONNX Runtime** (`ort`) | Executes the graph | The model, its two-head output contract, the crop preprocessing that must match training exactly, the SHA-256 bundle verification, the calibration binding, the runtime-eligibility gate, and the shadow-vs-classical publication rule |
| **PyTorch / torchvision** | MobileNetV3-Small backbone, autograd, optimizers | The grayscale adaptation, the two-head heteroscedastic architecture, the NLL + RankNet objective, the decision to exclude focuser position as a leakage source, the grouped/temporal split design, and the post-hoc uncertainty calibration |
| **AstroM3** (Rizhko & Bloom 2024, vendored + 5 HF checkpoints) | Informer, GalSpecNet, MetaModel encoders and CLIPLoss, pretrained | The retrofit itself: mapping on-mount telemetry onto the photometry branch and capture metadata onto the metadata branch, writing a **new `ImageEncoder`** for a modality AstroM3 has none of, and holding the shared space at 512-D so edge and cloud embeddings stay comparable |
| **SEP / DAOPHOT / SExtractor methods** (as literature) | Published algorithms for background, detection, curve-of-growth | Two independent implementations — Rust for the daemon, Python for the labeler — the rejection taxonomy, the "HFD = 2·HFR, FWHM is fitted separately" distinction, and the tests asserting the two implementations agree |
| **axum / tokio / sqlx** | HTTP, async runtime, SQLite | The API surface, the lock ordering, the acquisition mutex held across `.await` for minutes-long exposures, the migration ledger, and the stage→hash→manifest→fsync→transaction→rename→commit protocol |
| **LibRaw** | Decodes CR2/CR3/RW2/NEF/ARW | The decision to bypass its convenience wrapper and take raw sensor ADU; the `cblack` table handling (LibRaw's `cblack[4]`/`[5]` are the *dimensions* of a 2-D black table, not black values — a larger table cannot be represented in four offsets, so `black_level_is_complete()` reports it); CFA phase from `idata.filters` rather than `raw.COLOR()` because the latter's indexing origin has moved between releases; and the FITS calibration cards that let a downstream pipeline do the calibration correctly |
| **vcpkg / CMake / Homebrew** | Native dependency resolution | The pinned vcpkg baseline; the CMake logic probing brew keg prefixes for `opencv@4` (keg-only because the unversioned formula now tracks OpenCV 5 and ships `opencv5.pc`), which otherwise fails configure on any Mac with a perfectly good install |
| **AI-assisted development** | Implementation of several tasks, documented in `HANDOFF-next-session.md` | The problem definition, the ABI and lock-order design, the physical models, the ML contracts and leakage arguments, the defect diagnoses, the scope decisions, and — visibly, in the revision history of the drift-filter split and the deleted `focus_offset/0.3.0` bundle — the judgment about which proposed implementations were wrong and why |

---

## 17. Factual evidence for later use

### Technical independence

- Sole author of all 34 commits over 18 months.
- Wrote the system architecture specification (`achernar_v4.pdf`, Dec 2025, revised Mar 2026) listing himself as Lead Architect, then built to it — and departed from it deliberately in at least four documented places (per-star instead of whole-frame FQE; log-HFR in pixels instead of arcseconds; excluding focuser position as an input; 5-class instead of binary cloud labels in the field), each with written reasoning.
- Audited his own repository three times and wrote the audits up as technical documents: the wiring audit (June), the capability audit inside the data-collection protocol (August, explicitly "written by reading the tree, not the previous handoff notes"), and the market/repo-state review (August, "verified against code, not inferred from docs").
- Chose the focuser architecture by reading `OpenLiveStacker/src/focuser.cpp`, finding it is 46 lines with one driver, and concluding that ESP32 + Alpaca requires zero new C++ or Rust where Arduino + Moonlite requires a new driver in the riskiest part of the codebase.

### Technical depth

- Linear time-varying Kalman filter with augmented-state joint parameter estimation, Joseph-form covariance update, DWNA process noise — with the collinearity of `k·ΔT` and `v·dt` under steady cooling identified as the reason `k` needs varied thermal conditions.
- Derivation of `steps_per_micron = 1/(2√2·N·slope·pixel_size)` from the half-flux radius of a uniform blur disc, with three neglected physical terms named.
- Trail-rate arithmetic selecting the north celestial pole as an observing target: 15.041·cos(δ) ″/s gives 22.8 px/s at the equator against 0.26 px/s at δ = +89.35°, turning an untracked Dobsonian into a tracked telescope for focus-sweep purposes.
- Focuser sizing from the critical focus zone: 4.88·λ·N² ≈ 42 µm at f/3.95, step size ≤ CFZ/5 ≈ 8 µm, met by 3:1 GT2 reduction at 1/16 microstepping.
- Photon transfer curve method with paired flats, taking saturation from the variance turnover rather than the maximum sample value, refusing demosaiced input, and measuring the four CFA phases separately.
- ABBA deadlock diagnosis and a documented total lock order with a written rationale for placing the drift forecaster last.
- The `cblack` table subtlety in LibRaw, and the CFA-phase-source argument (`idata.filters` vs `raw.COLOR()`).

### Persistence and debugging

- **The `max_adu` bug.** Found that a hardcoded 65535 put the 0.98 saturation gate at 64224 on 14-bit data, so it never fired and every clipped star flowed into HFR. Traced it through six modules, split the concept in two, added a third gate state, and then recorded that the frames already collected carry unfixable labels but are now identifiable.
- **The HFR noise pedestal.** Found that `(pixel − sky).max(0.0)` in the flux integral rectified positive noise excursions into a pedestal that scaled HFR with the noise level, and that an unbounded aperture let a larger crop accumulate more noise-only annuli. Fixed both; added a test asserting stability across noise levels and crop sizes.
- **The default-capture defect.** Found that `/capture` read the preview mailbox, so a "science exposure" on a DSLR silently returned a low-resolution 8-bit live-view JPEG stamped with still-camera exposure metadata and no RAW.
- **Three disagreeing FQE input contracts,** resolved by deleting a versioned model directory whose number implied a trained artifact that never existed.
- **The ABBA deadlock** that wedged the daemon whenever a status poll landed during a stream start.
- **The Winsock `SO_RCVTIMEO` type mismatch** — `timeval` compiles on Windows and produces an invalid timeout.

### Research and engineering judgment

- **Rejected a convenient decode path on physical grounds.** `load_libraw()` was available and would have worked; the reasoning for not using it goes past integer truncation to why the multiply itself severs ADU from electrons (shot-noise variance scales by k² where signal scales by k) and why subtracting black into an unsigned type makes read noise unfittable.
- **Reversed his own design after review**, and wrote down the three reasons the first version was wrong — including that it "violated the principle it invoked" by substituting `off` for `unknown` without recording it.
- **Made a baseline honest at his own expense.** The `direct_hfr` row was a hard-coded 0-MAE entry; it is now executed and timed, and the file records that it runs on normalized crops whose p1/p99 clip slightly flattens bright cores, so it is "the deployable-baseline number, not the label oracle."
- **Cut three of the nine models,** with the reasoning written down for each, rather than leaving nine partially built.
- **Committed to reporting a negative result:** "If the model doesn't beat classical HFR on a held-out night, say so — that's a real result and it's more credible than a claim."
- **Resolved a spec/code disagreement in the direction that preserves information.** Section 7.6.4 specifies a binary cloud classifier; `cloud_classifier.py` declares 5 classes. Adopted 5-class field labels with a documented collapse to binary, because "five-class labels can always be collapsed to two; two can never be expanded to five" — and because the sky is gone by the time you would want to relabel.
- **Identified where the system's own output was insufficient** and said so in the daemon's warning text at startup, in the `/focus/drift` response body, and in the model metadata.

### Initiative

- Extended a third-party C++ codebase by ~2,775 lines rather than working around its limits.
- Wrote a Windows camera backend from scratch to move the project onto its target deployment platform, plus 827 lines of PowerShell tooling and a CI job to gate it.
- Built the ingest half of the observing campaign before the observing started: a 664-line sidecar importer mirroring the daemon's own commit protocol, a 347-line photon-transfer calibration tool that constitutes an entire model on its own, 20 tests over synthetic fixtures, and two migrations adding the environment, disturbance and night-grouping tables the runbooks write into — all of it usable with no telescope, no clear sky and no Rust toolchain.
- Wrote a per-model data-collection protocol that computes its own yields (≈15 min per focus sweep, ~10 sweeps per productive night, therefore 6 clear nights for v1 and 15 for v2) and states budgets in groups rather than rows, on the argument that "twenty thousand star crops from three focus runs is three data points."
- Turned the project's biggest instrumental limitation into its one label advantage: because the Dobsonian does not track, every exposure carries an exactly computed trailing label at `15.041·cos δ` arcsec/s, giving the trailing detector unlimited free ground truth on any clear night.
- Read and classified 51 astronomy and ML papers, and mapped specific implementation decisions to specific sources.
- Designed a fallback matrix (LCO public archive, LCO Global Sky Clubs, Skynet, ZWO EAF, Seestar, iTelescope, a GoTo mount) as *triggers* rather than options — "so you're not re-deciding under pressure in October" — including the one marked **Never**, with the reasoning that a GoTo mount solves tracking and tracking is not the blocker.

### Collaboration

No collaborator commits appear. An outreach track is documented (`email-miroshnichenko-followup.md`, and §5 of the build plan) toward Prof. Miroshnichenko, Director of the Three College Observatory, and Dr. Alicia Aarnio at UNCG — with three ranked asks, an explicit note that the second one "serves your project, not his," and a standing rule that "this track may not block the build." Its outcome is not visible in the repository.

### Tangible outputs

Software: a 21,269-line Rust daemon that compiles and runs; a 4,200-line C/C++ bridge; a Windows camera driver; a React control UI; a Tauri shell; a Django relay backend. Data infrastructure: a 25-table schema with 7 migrations, a durable capture archive with startup reconciliation, and three standalone Python tools. ML infrastructure: a complete FQE train→calibrate→baseline→export→serve chain, a 38-feature versioned input contract, and an AstroM3 edge retrofit. Documents: a system architecture specification, a 2,703-line data-collection protocol, a 463-line wiring audit, a build plan, three research notes, and a camera-control API reference. Tests: 194 passing across three languages. Verified during this audit: a 296-crop labeled dataset generated end to end from the daemon's own output.

---

## 18. Factual characterizations

1. Built a 21,000-line Rust telescope-control daemon with a 4,200-line C/C++ FFI bridge to camera, mount and focuser drivers, exposing 34 HTTP endpoints, a telemetry WebSocket, and a cloud reverse tunnel that shares one implementation with the direct HTTP path.
2. Implemented the measurement layer an autofocus system rests on — background estimation, star detection with itemized rejection, and curve-of-growth half-flux radius — and found and fixed two bugs in his own earlier version of it: a noise pedestal from clipping negative flux, and a saturation gate that was hardcoded to a value it could never reach on 14-bit data.
3. Built and demonstrated a closed-loop autofocus controller that drives a focuser through a sweep, fits a V-curve, moves to the fitted minimum, and persists each sample beside the exact pixels that produced it — verified running end to end against a simulator, where it correctly declined to converge on a degenerate curve rather than reporting a best position.
4. Built the complete training-to-serving chain for a Focus Quality Estimator — dataset builder, grouped and temporal splits, a two-head heteroscedastic CNN with a RankNet auxiliary loss, post-hoc uncertainty calibration fit on validation and reported on test, frozen classical baselines, ONNX export with numerical parity checks, and hash-bound runtime loading behind a promotion gate — verified running end to end, with the daemon's Rust inference matching the Python pipeline's output to six decimals on a model trained minutes earlier, and the gate correctly refusing that model for failing to beat the constant baseline.
5. Extended a vendored third-party C++ driver codebase by roughly 2,775 lines to add DSLR still capture and live view, and wrote a new 1,144-line Windows Media Foundation camera backend against the same driver contract.
6. Wrote the system architecture specification the project implements, then audited his own repository against it three times and wrote up each audit as a technical document — including a capability audit that names seven blocking defects in his own work by their effect on data quality.
7. Wrote a 2,703-line data-collection protocol specifying, per model, the label source, the split group, the covariates that cannot be recovered afterwards, and sample budgets stated in groups rather than rows — with six bench procedures, nine observing runbooks carrying abort and acceptance criteria, a machine-readable session-log schema, and a post-session ingest checklist — and designed four of the five in-scope models to bootstrap their labels from classical measurement or physics rather than hand annotation.
8. Built the ingest half of that campaign before observing began: a 664-line sidecar importer that mirrors the daemon's stage-hash-manifest-fsync-transaction-rename commit protocol and is idempotent by content hash, a 347-line photon-transfer tool that constitutes one of the nine models on its own, environment/disturbance/night-grouping schema, and 20 tests — with the daemon→dataset handoff verified producing 296 labeled crops from a live autofocus sweep.
9. Designed a $40 ESP32-based motorized focuser and an observing campaign for autumn 2026, choosing the Alpaca protocol specifically because reading `focuser.cpp` showed it was the one path requiring no new driver code, and choosing the north celestial pole as the observing target because at δ = +89.35° an untracked Dobsonian trails 0.26 px/s instead of 22.8.
10. Cut three of the nine specified models with written reasoning for each, on the grounds that five models forming one coherent focus-plus-conditions story — the focus stack, the two sky-condition models, and the measured sensor constants underneath them — are worth more than nine half-finished ones.
11. Is currently porting the daemon to Windows as its target rig platform — new Media Foundation camera backend, pinned vcpkg dependencies, 827 lines of PowerShell bootstrap/build/acceptance tooling, and an MSVC CI job — which has not yet had its first run, a fact the accompanying runbook states in its own status section.
12. Maintains 194 automated tests across Rust, Python and TypeScript, including cross-language parity checks asserting that the Rust and Python implementations of the same aggregation and the same ONNX graph produce identical numbers.

---

## 19. Canonical fact sheet

**Project** — Achernar (`Achernar.tech`)

**Type** — Hybrid: systems software (Rust daemon + C++ FFI), scientific instrumentation software (star detection, HFR, FITS), ML research infrastructure (nine-model spec, one chain built), and a hardware + observing data-collection program. Solo project, Feb 2025 – present.

**Primary objective** — A telescope control system whose focus, sky-condition and exposure decisions are made by learned models with calibrated uncertainty, bounded by a deterministic safety supervisor, and promoted only after beating frozen classical baselines on grouped held-out real data.

**Current state** — Acquisition, measurement, control, persistence and the full ML pipeline are built and running; the observing campaign that supplies their input begins now. Compiles clean, 194 tests pass, a full autofocus sweep runs end to end against the simulator, its output feeds the Python dataset builder, and the resulting train→calibrate→export→shadow lap completes with exact Rust↔Python parity. No model trained on sky data yet; no hardware driven yet.

**What exists** — Rust daemon (21,269 lines, 97 files, 12 modules): OLS FFI bridge with versioned ABI + startup check; camera/mount/focuser drivers with runtime capability reporting and user-triggered discovery; two acquisition paths (streaming, gphoto2 still capture); 1 Hz vision loop → `SharedObservation` bus; classical HFR with itemized star rejection and a three-state saturation gate; autofocus state machine with backlash compensation, safety guard and V-curve fitting; a Kalman focus-drift forecaster with online temperature-coefficient estimation; durable capture archive with stage→hash→manifest→fsync→transaction→rename→commit and startup reconcile; 25-table schema across 7 migrations; ONNX serving with SHA-256-bound bundles and a runtime-eligibility gate; cloud reverse tunnel sharing one implementation with HTTP. Plus 4,200 lines of own C/C++, 5,710 lines of Python ML, 1,425 lines of Python tools, 3,772 lines of TypeScript, a Django relay backend, and a Tauri shell.

**Under implementation** — Windows rig-agent port (Media Foundation camera backend, pinned vcpkg, PowerShell bootstrap/build/acceptance scripts, MSVC CI job — never run); ESP32 + Alpaca motorized focuser (designed and sized, not built); **the autumn 2026 observing campaign** — protocol complete, ingest tooling built and tested, first nights imminent.

**Intended system** — Rig agent at the telescope + hosted UI at achernar.com over a reverse tunnel. Plate-solved pointing; autofocus by predicted step offset with classical sweep as fallback; refocus timing from a thermal model; cloud and trailing detection; every model decision logged with its version and hash, bounded by the safety supervisor, and promoted only against frozen baselines.

**Definition of done** — Near-term: a predicted-vs-true best-focus plot on data collected by a focuser the author designed, printed, wired and flashed. Full: a second person installs the agent, connects their own hardware, and gets a night of imaging in which focus, refocus timing and condition response are made by models that earned promotion on held-out real nights.

**Inputs** — Camera frames (MJPEG/RAW8/RAW16/mono/RGB, DSLR RAW); focuser position; mount coordinates; temperature where available; `achernar.toml`; operator commands over HTTP/WebSocket or the relay.

**Core process** — frame → luma + saturation resolution → background/noise → star detection → per-star validity gates → curve-of-growth HFR → robust median + log-sigma → `SharedObservation` → {telemetry, autofocus sweep + curve fit + move, SQLite + preserved pixels} → Python dataset builder → grouped splits → train/calibrate/baseline/export → hash-bound bundle → shadow inference.

**Outputs** — Live view (MJPEG/WebSocket); JSON telemetry at 1 Hz; FITS with `BUNIT`/`BLKLEVEL`/`CBLACK0..3`/`SATURATE`/`BAYERPAT`/`XBAYROFF`/`YBAYROFF`; 16-bit PNG focus frames; SQLite rows across 25 tables with SHA-256 provenance; `.npz` training sets; ONNX bundles with metadata + calibration + norm params.

**Major components** — `hardware/`, `vision/`, `focus/`, `api/`, `storage/`, `models/`, `control/`, `mount/`, `filter_wheel/`, `processing/` (daemon); `ols-bridge/` (C/C++); `ml/` (contracts, data_pipeline, labels, models, train, eval, export, pipelines, astrom3); `tools/` (sidecar importer, photon transfer); `achernar-frontend/`; `achernar-desktop/`; `backend/`. **Data-acquisition subsystem** (§4): the 2,703-line protocol, 6 bench procedures, 9 runbooks, the `environment_log`/`disturbance_log` schema, `night_id`/`session_id` grouping keys, `capture_attempts` as the yield denominator, and the append-only `frame_measurements` table.

**Key technologies** — Rust (tokio, axum, sqlx, ort), C++17 + FFI, OpenLiveStacker (vendored, extended), LibRaw, OpenCV, libgphoto2, Windows Media Foundation, ASCOM Alpaca, ONNX Runtime, PyTorch/torchvision, MobileNetV3-Small, AstroM3, SQLite, React/TypeScript, Tauri, Django Channels, vcpkg/CMake. Methods: curve-of-growth HFR, SExtractor/SEP/DAOPHOT source gates, Moffat PSF fitting, heteroscedastic Gaussian NLL, RankNet, post-hoc uncertainty calibration, LinUCB, thin-plate splines, Kalman filtering with Joseph-form update, photon transfer curves, grouped and temporal splits.

**Major contributions** — The C ABI and its version discipline; the vision/measurement layer and its two bug fixes; the autofocus controller; the drift filter and its estimation/decision split; the durable capture protocol; the 38-feature FOP contract with provenance and availability; the FQE design decisions (per-star, log-pixels, no position); the frozen-baseline promotion gate; ~2,775 lines extending the gphoto2 driver; the Windows Media Foundation backend; the AstroM3 edge retrofit; three self-audits; the observing plan and focuser design.

**Strongest demonstrated capabilities** — Compiles and runs with 194 passing tests; drives a complete autofocus sweep end to end and persists linked samples and pixels; detects 34–39 stars per frame and measures HFR with tested numerical properties; produces a 296-crop labeled dataset from its own database; runs the full train→calibrate→export→publish→shadow lap with exact Rust↔Python parity; refuses to converge on a degenerate curve, refuses to promote a model that fails its gate, and refuses to invent a measurement it did not make.

**Major limitations** — No model trained on sky data; no hardware ever driven; `saturation_adu` unset for 16-bit sources (8-bit is unaffected — gate fires by construction); LibRaw not compiled in the current build; Bayer phase unverified on a real frame; no independent validation of the measurement layer; Windows port never built.

**Major unfinished components** — Filter wheel hardware layer, environment sensors, INDI client, debayer/stacking/calibration, plate solving, generic ONNX runner, 8 of 9 model training paths, 11 empty storage repos, Django AI engine, the physical focuser.

**Validation performed** — 194 automated tests; exact Rust↔Python ONNX parity on a freshly trained model; ONNX-vs-PyTorch FP32 parity at 1.79e-07; a demonstrated promotion-gate rejection with a machine-readable verdict; 61 SQL statements against real SQLite; the Kalman filter ported to Python and run; the V-curve derivation round-tripped across four scope/camera combinations; the C++ decode block run against a stub LibRaw including an odd-margin Bayer-phase case; frontend strict typecheck; live daemon smoke test with a complete autofocus sweep and dataset build. **Not performed:** any comparison against an independent measurement on real data, any hardware test, any model evaluation on real sky.

**Development frontier** — The T0 bench tier (sensor calibration, camera capability probes, plate scale) which needs no clear sky; Windows port bring-up; the focuser build; and the opening nights of the observing campaign — COC on any night, focus sweeps on the pole when clear.

**Endpoint** — Learned, uncertainty-bounded closed-loop telescope control, promoted against classical baselines.

**Next technical steps** — (1) One green Windows CI run and an SV105 accepted through Media Foundation. (2) Photon transfer on real bias/flat ladders → `[camera] saturation_adu` → Model 6 complete. (3) Build, flash and bench-calibrate the ESP32 focuser; verify `POST /focuser/moveto` works with no code change. (4) Artificial-star sweeps indoors to debug the sweep→capture→HFR→database loop with no clear nights consumed (engineering data, explicitly excluded from training). (5) 60–80 real focus runs on the pole, one in three reversed. (6) Train the FQE, calibrate, evaluate against frozen baselines on a held-out night, publish as shadow.

**Tangible artifacts** — A running daemon binary; a 25-table populated schema; a tested sidecar importer and photon-transfer tool; 296 labeled crops built during this audit; a complete FQE bundle (ONNX + metadata + calibration + norm params) produced end to end during this audit and correctly refused by its own gate; 194 passing tests; 4 checked-in model bundle directories (1 with a loadable graph, 0 trained); 5 AstroM3 checkpoints; a system architecture specification, a 2,703-line data-collection protocol with 6 bench procedures and 9 observing runbooks, a wiring audit, a build plan, 3 research notes, and a camera-control API reference; CI definitions for Linux and Windows.

**Quantitative context** — ~38,000 lines of own code across Rust/C++/Python/TypeScript/SQL/PowerShell; 34 commits over 18 months, one author; 9 models specified, 5 in scope, 4 with bundles; 51 papers incorporated; campaign budgets of 60 focus runs (v1, ≈6 clear nights) and 150 (v2, ≈15); 4.5 GB working tree.

---

## 20. Concise descriptions

**One sentence**

Achernar is a telescope control system under development for learned, uncertainty-bounded closed-loop imaging control, with a 21,000-line Rust daemon, a C++ driver bridge, classical star measurement, a working autofocus controller, a complete model training-and-serving pipeline, and a specified observing campaign with its ingest tooling already built — now entering the data-collection phase those models require.

**Three sentences**

Achernar is intended to be a telescope control system in which focus, refocus timing, and sky-condition response are decided by learned models with calibrated uncertainty, bounded by a deterministic safety supervisor and promoted only after beating frozen classical baselines on held-out real nights. I have built the daemon that makes that possible — a Rust hardware-control service with a C++ FFI bridge to camera, mount and focuser drivers, a 1 Hz vision loop that detects stars and measures half-flux radius with itemized rejection accounting, a closed-loop autofocus controller that sweeps, fits a V-curve and persists each sample beside the pixels that produced it, and a durable capture archive — together with the complete train→calibrate→baseline→export→shadow-serve pipeline, which runs end to end, produces a labeled dataset from the daemon's own output, and whose Rust inference matches its Python training code exactly. What remains is the data: a motorized focuser I am building, sensor calibration constants I need to measure, a Windows port I am finishing so the imaging camera can be used, and roughly 25 clear nights of focus sweeps — after which the model training that the entire pipeline exists to serve can happen for the first time.

**Technical paragraph**

Achernar is a telescope control and imaging system built around a Rust daemon that runs on a computer at the telescope, interfacing with cameras, mounts and focusers through a C ABI bridge into OpenLiveStacker's driver layer. The implemented core acquires frames over two paths — a streaming path for live view and autofocus, and a gphoto2 still-capture path that applies and reads back shutter and ISO before triggering — and runs a 1 Hz vision loop that estimates sky background, detects stars, applies saturation/edge/blend/elongation gates, and measures half-flux radius by curve of growth, publishing the result with explicit validity status and per-reason rejection tallies on a shared observation bus. An autofocus controller consumes that bus to sweep the focuser, fit a V-curve, and move to the fitted minimum, persisting each sample beside the exact preserved pixels through a durable archive that stages, hashes, manifests, fsyncs and atomically commits across SQLite and the filesystem. A Kalman filter estimates focus drift and its per-rig temperature coefficient online. On top of this sits a complete machine-learning pipeline for a Focus Quality Estimator — grouped and temporal splits, a two-head heteroscedastic CNN, post-hoc uncertainty calibration, frozen classical baselines, ONNX export, and hash-bound loading behind a runtime-eligibility gate the daemon enforces — every stage of which executes end to end, with the daemon's Rust inference matching the Python training code exactly. Feeding it is a designed data-acquisition program: a 2,703-line collection protocol specifying per-model label sources, group-based sample budgets, six bench procedures and nine observing runbooks, backed by an implemented and tested sidecar importer, a photon-transfer calibration tool, environment and disturbance logging tables, and observing-night grouping keys shared between the daemon and the importer. Current work is a Windows port with a new Media Foundation camera backend, a DIY ESP32 focuser chosen so it connects over Alpaca with no new driver code, and the opening nights of that campaign.

---

*Compiled 2026-08-24 by direct inspection and live execution. Sections marked **[verified]** were run during the audit; everything else is cited to a file, a commit, or a project document.*

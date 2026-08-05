# Handoff — Achernar, next session

Paste this as your first message in a fresh conversation. Delete this file once
the work lands.

---

## Working agreement

Reference `achernar_v4.pdf` — the app is based on it. I review all your code and
work alongside you, so explain every model and concept you introduce and the
reasoning behind each step, in enough detail that I can shape my own work
around it. I'll describe what I'm doing so you can do the same.

**Do not trust this document over the code.** It was written by a previous
session from a partial reading. Verify every line reference before acting on it.

---

## Where things stand

A previous session rebuilt the DSLR/mirrorless capture path so models can be
trained on real data. Landed and verified:

- LibRaw detected and compiled into `ols-bridge`; `link_optional_libraw()` in
  `build.rs`.
- `capture_still()` on the OLS `Camera` interface + gphoto2 implementation
  (pause live view → apply and read back shutter/ISO → timestamp → trigger once
  → drain to `CAPTURE_COMPLETE` → download every companion → resume).
- Opaque `OlsStillCapture` FFI handle; bridge ABI bumped to **v2** with a
  startup check in `main.rs`.
- `/capture` no longer reads `state.latest_frame`. Runs under a new
  `AppState::acquisition` async mutex, on `spawn_blocking`.
- `migrations/` is now the single schema authority (it used to be decorative —
  nothing executed it, and its `frames` definition disagreed with the live one
  in `local_store.rs`). `004_frame_assets.sql` adds `frame_assets`,
  `capture_attempts`, `frame_measurements` and the capture columns.
- `storage/capture_store.rs`: stage to `.partial` → hash while writing →
  `manifest.v1.json` → fsync → SQLite transaction → atomic rename → commit →
  only then delete from the camera. `reconcile()` runs at startup.
- ABBA deadlock between `get_status` and `start_stream` fixed; lock order
  documented on `AppState`.

Verified in a Linux sandbox: 58 SQL statements against real SQLite (fresh,
legacy, idempotent re-run, UNIQUE enforcement); C++ syntax-clean with and
without `WITH_LIBRAW`; `ols_bridge.h` valid as pure C.

**Never compiled.** The sandbox has no Rust toolchain.

---

## Step 0 — before anything else

```bash
brew install libraw libgphoto2      # if not already present
cd achernar-daemon && cargo build 2>&1 | tee /tmp/build.log
cargo test
```

Expect signature mismatches from the capture work. Fix those against a clean
tree first — do not tangle them with new changes. Watch the CMake summary for
`Decode: ✅ LibRaw` and the startup line `OLS Bridge ABI: v3`.

---

## Task A — RAW decode — WRITTEN, NEVER COMPILED

Landed. Not built: the sandbox has no Rust toolchain, no LibRaw and no OpenCV.
`cargo build` on the Mac is the first real test.

### What it does

`ols_decode_raw` in the bridge produces **raw sensor ADU**. No black
subtraction, no scaling, no white balance, no left-shift. Calibration constants
travel in the header instead of being baked into the pixels.

It deliberately does **not** call `ols::load_libraw()` from
`libraw_wrapper.h`. That function black-subtracts and multiplies by
`65535 / (maximum - black)` in *integer* arithmetic — 4 on a 14-bit Canon where
the exact factor is 4.572. Fixing the truncation would not help: the multiply
itself severs ADU from electrons (signal scales by k, shot-noise variance by
k², so any gain fitted from the frame is wrong by k), and k is body-dependent,
so two cameras stop being comparable. Subtracting black into an unsigned type
clips the negative half of the noise distribution around the pedestal, which
makes read noise unfittable. `load_libraw()` is untouched and still serves the
preview stream, which wants a stretched image and is welcome to one.

### Files

| File | Change |
|---|---|
| `ols-bridge/include/ols_bridge.h` | ABI → **3**; `OlsRawImage` + `OlsRawImageInfo`; `ols_decode_raw` / `_info` / `_pixels` / `_destroy`; `OLS_CAP_DECODE_LIBRAW` (bit 16) |
| `ols-bridge/src/ols_bridge.cpp` | `#include <libraw/libraw.h>` under `WITH_LIBRAW`; own `open_buffer → unpack → crop → copy` path; CFA phase from `idata.filters` in raw coordinates |
| `src/hardware/ols_ffi.rs` | ABI → 3; extern decls; `OlsRawImageInfo` |
| `src/hardware/raw_decode.rs` | **new** — `RawImage` (Drop-releases the handle), `RawInfo`, `libraw_available()` |
| `src/hardware/camera.rs` | `check` / `last_error` → `pub(crate)` |
| `src/fits.rs` | `SensorCalibration`; `encode_bayer_u16(&[u8], …)`; RAW arm in `fits_from_source`; `BUNIT`/`BLKLEVEL`/`CBLACK0..3`/`SATURATE`/`XBAYROFF`/`YBAYROFF` |
| `src/api/camera.rs` | decode provenance into `acquisition_json`; two new warnings |
| `src/main.rs` | startup line for LibRaw presence |

### Deviations from the sketch, and why

- **`bin` must be 1.** Accepted in the signature for ABI stability, rejected at
  runtime. A 2×2 sum saturates at 4× `white_level`, which `OlsRawImageInfo`
  cannot express — it would be a lossy pre-calibration step of exactly the kind
  this API refuses. Bin after calibration.
- **`cblack[4]` and `cblack_pattern[2]` added to the info struct.** LibRaw's
  `color.cblack[4]`/`[5]` are the *dimensions* of a 2-D black table stored from
  `cblack[6]`, not black values. A table larger than 2×2 is not representable in
  four per-channel offsets, so recording only those would silently understate
  the black level. `RawInfo::black_level_is_complete()` reports it and the
  capture path raises a warning.
- **CFA phase from `idata.filters`, not `raw.COLOR()`.** Whether `COLOR()`
  indexes from the raw-array origin or the visible-image origin has moved
  between LibRaw releases, and choosing wrong rotates RGGB→GRBG only on bodies
  with an odd margin. `filters` is unambiguously the raw array's pattern, and
  the pixel copy indexes the raw array too, so they agree by construction.
- **One copy, in C++.** The handle owns a tightly-packed plane so
  `ols_rawimage_pixels` can honestly promise `w*h*2` contiguous bytes; Rust does
  zero copies from there to the FITS encoder. Keeping the `LibRaw` instance
  alive instead would save that copy but hold its larger margin-inclusive
  allocation for the life of the handle and make every consumer stride-aware.

### What was actually verified

`ols_bridge.h` compiles clean as C99 and C++17. The decode block was extracted
into a harness against a stub LibRaw and run: full-frame crop offsets and pixel
values, an **odd left margin correctly reporting GRBG rather than mislabelling
RGGB**, a centred sub-region preserving phase, and the rejection paths (bin≠1,
null, oversized region, X-Trans, null accessors). All pass. See the "known
unknowns" below for what a stub cannot tell you.

### Known unknowns — settle these on the Mac

1. **Bayer phase on a real frame.** Shoot a red-dominant target and confirm the
   red plane is red. `scripts/dslr_bench.sh` is where this belongs.
2. **`open_buffer` signature.** Cast to `char*` so it works whether the local
   LibRaw takes `char*` or `void*`. If it fails to compile, that assumption is
   the first place to look.
3. **`raw_pitch == 0`** on older LibRaw is handled by falling back to
   `raw_width * 2`. Unverified against a real build.

---

## Task D — `max_adu` was hardcoded — DONE, unbuilt

`quality.rs` derived `max_adu` from the stream format and hardcoded `65535.0`
for Raw16/Mono16, so the `0.98` gate sat at 64224 on a body whose data stops at
16383 and **never fired**. Every clipped, flat-cored star flowed into HFR.

`LumaFrame` now carries `max_adu` (container full scale) and `saturation_adu`
(the sensor's linear limit, `Option`) as separate things, because they are.

- 8-bit streams: the container *is* the clip point; nothing to declare.
- 16-bit streams: must be declared via `[camera] saturation_adu`, or come from
  the RAW decoder's `white_level`. Undeclared is `None`, not 65535.
- `None` produces `SaturationGate::NotPerformed` rather than a silent pass.
  `QualityMetrics`, `SharedObservation` and the autofocus `quality_json` all
  carry it, so a dataset builder can exclude frames whose gate never ran.
  "No stars were saturated" and "we never checked" are now different answers.

`estimate_background` takes `Option<f32>`; `None` means exclude nothing, which
is safe for a median/MAD estimate in a way it is not for HFR (the background is
robust to a few clipped pixels; HFR is measured *on* the clipped star).

`background_estimator.rs` had the same bug spelled differently —
`observed_max.max(65535.0)` is ≥ 65535 by construction — and now passes `None`.

**Files:** `config.rs` (`[camera] saturation_adu`), `hardware/camera.rs`
(`FrameData.saturation_adu`, `start_stream` takes it), `state.rs`
(`camera_config`), `vision/{quality,background,vision_loop,ted_runner,
background_estimator}.rs`, `api/{camera,focus}.rs`, `relay_commands.rs`.

**Still wrong until you act:** the FQE frames already collected carry the broken
labels. They are not retroactively fixable — the gate genuinely never ran — but
they are now *identifiable*, because rows written from here on record
`saturation_gate_applied` and older ones deserialize to `NotPerformed`.

---

## Task B — FDF wired in — DONE, unbuilt

`/focus/drift` returns a real `DriftPrediction`. Three distinct statuses, kept
distinct because they call for different actions: `disabled` (you have not
configured it), `uninitialized` (configured, not enough measurements yet), `ok`.
Plus `unavailable` when there is no temperature at all.

- `AppState.focus_drift: Option<Arc<tokio::sync::Mutex<FocusDriftForecaster>>>`.
  `Option`, so "not configured" and "configured but untrained" stay separable.
  Placed **last** in the documented lock order: the forecaster is a leaf, so at
  the tail it can never be the outer lock in a cycle.
- Temperature persisted at both sites, read per frame (a sweep takes minutes and
  a cooling tube is the effect being modelled), `NULL` when unmeasurable, with
  `temperature_state` / `temperature_source` in `quality_json`.
- `update()` is fed on **autofocus completion only**, after the move succeeds —
  the one moment the daemon knows by measurement where best focus is. Feeding it
  a failed run teaches it that best focus is wherever the sweep gave up.
- `focus_drift_events` rows written on every evaluation; new
  `/focus/drift/history`. `focus_drift_repo.rs` was an empty stub and is now
  real, with an atomic (`.tmp` + rename) state snapshot.

### The two stale artifacts

`kalman_state.json` is **deleted**, and `models/focus_drift/0.1.0/README.md`
explains why. It was trying to be two things with opposite lifecycles — tuning
constants that ship with the code, and a filter state learned nightly per rig —
so it could serve neither, and `from_state()` could not parse it.

- Tuning → `[focus.drift]` in `achernar.toml`.
- Physics → already in `DriftConfig`'s doc comments; the old `_note` about
  `temp_coeff_var = 400` was a duplicate of what the code already says.
- Learned state → `<data_dir>/focus_drift_state.json`, git-ignored.

`drift_model_path` is **removed** from `FocusModelsConfig`, not just uncommented
out of the example. FDF is a Kalman filter; there is no ONNX file and there
should not be a key implying one.

### The estimation / decision split (revised after review)

The first version of this refused to construct the forecaster when the optics
were unspecified. That was wrong, for three reasons:

1. **It killed the wrong half.** The filter does two separable jobs.
   *Estimation* — identify `[position, drift_rate, temp_coefficient]` from
   (temperature, position) pairs — needs no optics at all. *Decision* — turn
   predicted drift into `refocus_recommended` — needs the CFZ.
   `refocus_threshold_steps` enters only the second.
2. **The expensive half was the collateral damage.** The temperature
   coefficient is only identifiable across varied thermal conditions over
   several sessions, because under steady cooling `k·ΔT` and `v·dt` are
   collinear. Staying dark until someone edits TOML means the night they finally
   do, you start from the diffuse prior with no history.
3. **It violated the principle it invoked.** Refusing to construct substitutes
   `off` for `unknown`. `off` is a decision too — it just wasn't recorded as
   one, and was invisible at `/focus/drift`.

**What it does now.** `DriftConfig::refocus_threshold_steps` is `Option<f64>`
with **no default** (the placeholder 50 is gone). The forecaster is always
built. `predict()` returns `predicted_drift_steps`, `confidence_interval` and
`initialized` unconditionally; `exceed_probability`, `refocus_recommended` and
`time_to_refocus_s` are `Option`, and `DriftPrediction::advice` carries the
reason — `Available { threshold_steps, source }`, `NotInitialized`, or
`ThresholdUnknown`. `Some(false)` and `None` are now different claims.

`to_drift_config()` errs only on values that are **present and wrong** (a
negative sigma, a confidence outside `[0,1]`). Absence is modelled, not
rejected.

`/focus/drift` therefore answers with the evidence *and* the prompt:

```
drift -37 ± 9 steps · temp coeff -18.2 ± 0.6 steps/°C · 340 updates
advice unavailable: no refocus threshold — set [focus.drift] steps_per_micron,
or rigs.pixel_size_um and let the next V-curve measure it
```

### Making a threshold cheap to acquire

Two paths, both aimed at never asking for a number the system can find itself.

**Rig profile.** `rigs.focal_ratio` has existed since migration 002.
`to_drift_config()` now falls back to it before demanding `[focus.drift]
f_ratio`, and tags the result `ThresholdSource::RigProfile`.

**V-curve measurement.** `steps_per_micron` is mechanical, printed on nothing,
and the friction that guaranteed this stayed unset — and the sweep already
measures it. Defocus `δ` at focal ratio `N` gives a blur diameter `δ/N`; a
uniform disc of radius `R` has half its flux inside `R/√2`; so along a V-curve
wing `dHFR/dδ = 1/(2√2·N)` in µm. Converting the solver's slope (px per step)
to those units:

```
    steps_per_micron = 1 / (2√2 · N · slope[px/step] · pixel_size[µm])
```

`focus_drift::steps_per_micron_from_vcurve` implements it. At autofocus
completion, `measure_focuser_scale` averages the two wing slopes from
`fit_v_curve`, derives a threshold, and adopts it as
`ThresholdSource::MeasuredVCurve` — which always loses to a configured value,
and is persisted to the new `focuser_profiles.steps_per_micron` (migration
005) so it is measured once, not every restart.

Inputs come from `rigs`: `focal_ratio` and `pixel_size_um`. There is
deliberately no `[focus.drift] pixel_size_um` key — a second place to type the
same fact is exactly the friction being removed.

Neglected, and documented in the function: central obstruction (a 30% secondary
makes the estimate ~4% low, against a threshold convention that is itself
ambiguous by 2×), seeing and diffraction (excluded by fitting the wing, not the
vertex), and focuser non-linearity. All three are why this is tagged as an
estimate rather than treated as calibration.

---

---

## Task C — Frontend adopts the DSLR contract — DONE, typechecks

`tsc -p tsconfig.app.json --noEmit` passes.

1. `CamParam` gained `choices: string[]` and `current_value?: number`;
   `CamFormat` gained `mode: "liveview" | "still"`. The daemon was already
   returning all three.
2. `ParamRow` renders `kind === "selection"` as a `<select>` of labels — "1/125",
   not 17.
3. Capture sends `gain: undefined` when the body has no numeric gain option.
   `CaptureParams.gain` is now optional; `0` was overriding the operator's ISO.
4. `metadata.warnings[]` rendered as an amber list (not red — these are
   conditions on a capture that *succeeded*). The message line also shows the
   applied shutter when it differs from the request.
5. Format picker labels each entry `live view` / `still (1 shutter/frame)`, the
   stream defaults to a live-view format when one exists, and selecting a
   `still` format shows a standing note about shutter actuations.
6. Camera index plumbed end to end: `ConnectRequest.index` on the daemon (out of
   range is a 400, not a silent clamp to 0), `connectCamera(driver, option,
   index)` on both transports, and the discovery list now selects a *device*
   rather than just a driver.

---

## What was verified, and how

No Rust compiler in the sandbox, so none of this is compiled. What was checked:

- **61 SQL statements** against real SQLite: fresh apply, then a re-run through
  a port of `local_store.rs`'s benign-error path (19/29 of `004` skip as
  `duplicate column name`, which is why the `schema_migrations` ledger exists).
  `focus_drift_events` INSERT/SELECT, `NULL` temperature round-trip, FK
  rejection of a dangling `session_id`, the new `focuser_profiles` columns and
  the scale upsert (a second measurement refines in place, no duplicate rows),
  and a threshold-free evaluation storing drift with `NULL` confidence — all
  pass.
- **The Kalman filter after the split**, ported to Python and run against the
  new tests' scenarios. The headline result: with **no threshold at all**, the
  temperature coefficient is still identified to −19.98 against a true −20.0.
  That is exactly what the previous `Err` would have thrown away. Adopting a
  threshold afterwards changes the drift estimate by 0 to 12 decimal places.
- **The V-curve scale derivation** round-trips exactly across four
  scope/camera combinations, and the implied V-curves are realistic (3.4 px of
  HFR change over a 500-step sweep at f/5.6 with 3.76 µm pixels).
- **The C++ decode block** compiled against a stub LibRaw and run: crop offsets,
  pixel values, odd-margin Bayer phase, sub-region phase preservation, and every
  rejection path. `ols_bridge.h` is valid C99 and C++17.
- **The saturation logic** ported to Python and run against the byte-identical
  pixel arrays the new Rust tests build, confirming the new assertions hold and
  the pre-existing `saturated_star_is_rejected` still passes.
- **The CFZ arithmetic**: f/5.6, 550 nm, 2.5 steps/µm → 86.24 steps; f/10 is
  exactly 4x f/5.
- **`tsc --noEmit`** clean on the frontend.
- Brace/paren balance on all 16 changed Rust files.

Not checked, because it cannot be here: that any of the Rust compiles.

---

## Ordering — all four are written; none are compiled

A, D, B and C are all landed as code. The remaining work is a build, and it can
only happen on the Mac.

```bash
brew install libraw libgphoto2      # if not already present
cd achernar-daemon && cargo build 2>&1 | tee /tmp/build.log
cargo test
```

Fix in this order, because the errors cascade that way:

1. **`ols-bridge` / CMake.** Watch for `✅ LibRaw` in the summary. If
   `open_buffer` fails to compile, that is the `char*` vs `void*` cast in
   `ols_decode_raw` — the one assumption a stub LibRaw could not test.
2. **`AppState::new` signature.** It gained `camera_config` and `data_dir`.
   Only `main.rs` calls it.
3. **`start_stream` signature.** It gained `saturation_adu`. Three call sites:
   `api/camera.rs`, `api/focus.rs`, `relay_commands.rs`.
4. **Struct literals.** `FrameData`, `LumaFrame`, `SharedObservation`,
   `QualityMetrics` and `FitsMeta` all gained fields; every literal in `src/`
   was updated, but the compiler is the judge.

Then, at the telescope:

- `OLS Bridge ABI: v3` and `RAW decode: LibRaw available` on startup.
- A real CR3 through `/capture`: confirm a `.fits` asset appears with
  `BLKLEVEL`, `SATURATE` and `BAYERPAT`, and that its pixel values are in
  sensor ADU (max ≈ 16383 on a 14-bit body, **not** ≈ 57340 or 65535).
- **The Bayer phase.** Shoot a red-dominant target and confirm the red plane is
  actually red. This is the one thing no amount of stub testing settles, and it
  belongs in `scripts/dslr_bench.sh`.
- Set `[camera] saturation_adu` for your body, then check that
  `saturation_gate_applied` is `true` in a persisted autofocus frame's
  `quality_json`. If it is `false`, the gate is still not running.

Do them in separate sessions if context gets tight — say so rather than writing
edits against files you have only partly read.

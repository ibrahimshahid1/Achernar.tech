# tools/

Three standalone utilities that close the gaps between "the daemon exists" and
"data is being collected." All are pure Python + numpy (+ Pillow); none needs
the Rust toolchain, a telescope, or a clear sky.

Run the tests first — they use synthetic fixtures and take about a second:

```bash
python3 -m unittest discover -s tools -p 'test_*.py' -v
```

---

## `photon_transfer.py` — measure your sensor (BP-2)

**Do this first. It works indoors, in any weather, and everything else depends
on it.**

Spec §7.6.6 (Background / SNR Estimator) is not a trained model — it needs
constants. `READNOISE` and `SENSOR_SATURATION` appear literally in the spec's
own listing and nothing in the repo measured them. This does.

It also produces the number for `[camera] saturation_adu`, which is the other
half of defect D1: until it is set, every DSLR frame carries
`saturation_gate = not_performed`, and the FQE dataset builder will refuse the
frames (correctly — see below).

```bash
python3 tools/photon_transfer.py \
    --bias  ~/ptc/760d-iso800/bias/ \
    --flats ~/ptc/760d-iso800/flats/ \
    --camera "Canon EOS 760D" --iso 800 \
    --out docs/device-profiles/ --all-phases
```

**What to shoot** (RB-1): 20 bias frames at the shortest shutter with the cap
on, then 2 flats at each of ≥6 illumination levels spanning ~5 %–105 % of full
scale. Vary the *shutter*, not screen brightness — screen brightness controls
are non-linear and PWM-modulated. Name flats so sorted order pairs them
(`lvl00_a`, `lvl00_b`, `lvl01_a`, …).

**Method.** For a pair of flats at the same illumination the mean carries signal
and the *difference* carries only noise, which cancels fixed-pattern
non-uniformity that would otherwise dominate the variance and make gain come out
far too small:

```
g = [(S1 + S2) − (B1 + B2)] / [var(S1 − S2) − var(B1 − B2)]     e−/ADU
read_noise = g · std(B1 − B2) / √2                              e−
```

Saturation is taken from where the variance-vs-signal curve **turns over**, not
from the largest value in the file — that is the container ceiling, and
conflating the two is the bug this whole exercise exists to prevent. If your
ladder never reached the knee, the tool says so and reports a lower bound
instead of a number.

RAW input needs `rawpy` (`pip install rawpy`); otherwise convert with
`dcraw -D -4 -T` first. It refuses demosaiced input, because interpolation mixes
neighbouring pixels and destroys the independence the method assumes. The four
CFA phases are measured separately — they have different sensitivities and
mixing them inflates the variance.

Outputs a device-profile JSON, a PTC plot if matplotlib is present (good
portfolio artifact), and the exact TOML line to paste.

---

## `import_session.py` — ingest frames the daemon didn't capture (G2/G4/G5)

**This is what unblocks the cloud campaign tonight.** Every open-loop campaign —
intervalometer runs, untethered bodies, hand-shot calibration frames — produces
files with no database row. Without this they're bytes on a card with no
provenance and no way to join them to the conditions they were taken under.

```bash
python3 tools/import_session.py \
    --db achernar-daemon/achernar.db \
    --files ~/coc/2026-08-07/ \
    --log   ~/coc/2026-08-07/session.json \
    --capture-root ~/achernar-frames \
    --migrations achernar-daemon/migrations \
    --capture-type light --dry-run
```

Mirrors `storage/capture_store.rs` exactly: hash while staging to `.partial`,
write a manifest, fsync, open a transaction, insert `frames` + `frame_assets`,
atomically rename, commit. A crash mid-import leaves no half-recorded frame.

**Deliberate omissions**, both of which are the honest choice:

- **No `capture_attempts` rows.** That table records what the daemon
  *commanded*. An offline import commanded nothing, and inventing attempts would
  corrupt the denominator in every yield calculation.
- **`temperature_state = 'missing'`.** There's no thermometer we can trust in
  this path. Ambient goes in `environment_log` instead, where its provenance is
  explicit.

A RAW and its in-camera JPEG are recognised as **one exposure with two assets**,
not two frames. Re-running is idempotent (duplicate detection by content hash),
so it's safe to re-import a card.

Metadata falls back EXIF → session log → file mtime, and records which in
`acquisition_json.metadata_source`. A timestamp from mtime is not the same
observation as one from the shutter.

**Sky labels.** Entries in the log's `environment` array with a `sky_class` get
matched to frames by timestamp (within 15 min) and written to
`frame_measurements` as `measurement_type='cloud'`,
`pipeline_version='manual_log_v1'`. That table is append-only and versioned, so
your label and a future COC model's prediction sit side by side on the same
frame — which is what makes the model checkable against the person.

Session log schema is Appendix C.5 of `achernar_data_collection.pdf`.

---

## `migrations/006_environment_log.sql`

Two new tables, applied automatically by the importer's `--migrations` flag:

- **`environment_log`** — ambient temperature, humidity, wind, sky class (0–4),
  transparency, seeing impression, with a `source` column separating
  `manual_log` from `probe` from `forecast`. Every column but the timestamp is
  nullable on purpose; a night where humidity wasn't recorded must be
  representable as *not recorded*.
- **`disturbance_log`** — bumps, headlights, dew, cloud, battery swaps. One
  unlogged bump in a focus sweep produces a bimodal V-curve that the fitter
  resolves to the wrong minimum with no outward sign. This is how a run gets
  dropped on purpose instead of quietly poisoning a model.

---

## What changed in `ml/`

`data_pipeline/build_focus_quality_dataset.py` — **defect D1 fixed.**

It read `max_adu`, the *container* ceiling, and fell back to 64000. On a 14-bit
Canon in a 16-bit container the sensor clips near 16383 while that rule put the
gate at 64230, so no star was ever rejected. Clipped cores have flat profiles and
measure *smaller* HFR than truth, so the bias landed on the brightest,
highest-weight samples in the set.

It now reads `saturation_adu` — which `api/focus.rs` has been writing into
`quality_json` all along, with a comment saying it's there "so the dataset
builder can filter on it."

Note the deliberate asymmetry with the daemon. At runtime an unknown saturation
point yields an *infinite* threshold: reject nothing, report the gate as not
performed, keep observing. At training time the opposite is correct: **drop the
frame**, because once an ungated label is in the dataset it can't be
distinguished from a clean one. `--allow-missing-saturation` restores the old
behaviour for synthetic fixtures and stamps the manifest `permissive` so such a
build can never be mistaken for a real one.

---

## Still open, and needing a Rust compiler

I couldn't verify these — this sandbox has no toolchain.

| ID | Status |
|---|---|
| **D2** | **Downgraded from blocking to latent.** `saturation_adu` reaches `start_stream` (`api/camera.rs:159`) and the autofocus path (`api/focus.rs:561`), but not the still-capture handler. It turns out `capture()` computes no quality at all — no `LumaFrame`, no `estimate_quality` — so FQE labels come exclusively from the autofocus path, which *is* correctly plumbed. Fix it when still-capture starts measuring, not before. |
| **D6** | Plate solving is a 40-line stub. Cheapest fix is shelling out to a local `solve-field`. |
| **coc_runner** | Written but never invoked — `vision/mod.rs` declares the module and nothing calls it. Decide its input domain before wiring: a model trained on 60° all-sky frames will fail silently on 40-arcmin telescope frames. |
| **D7** | Resolved in favour of **5 classes**: `cloud_classifier.py` and `coc_runner.rs` both use five, against the spec's binary head. The importer stores both `sky_class` and a derived `sky_class_binary`, so either head can be trained from the same labels. |

## Test status

| Suite | Result |
|---|---|
| `tools/test_import_session.py` | **10/10 pass**, including applying migrations 001–006 to a fresh database |
| `tools/test_photon_transfer.py` | **10/10 pass**, recovering gain, read noise, black level and saturation from simulated sensors with known ground truth |
| `ml/tests/test_dataset_builder.py::TestSaturationGate` | 5/6 pass here; the sixth needs `sep`, which wouldn't install (sandbox disk full). **Run it on your Mac.** |

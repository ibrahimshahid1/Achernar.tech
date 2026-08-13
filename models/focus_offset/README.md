# focus_offset

No bundle here yet. The Focus Offset Predictor (Section 7.6.2) has never been
trained, because its label is a signed focuser step count and no focuser with a
readable absolute position exists yet (defect D3).

## What used to be here

A `0.3.0/` directory holding a `metadata.json` and a `norm_params.json`, both
all-null, and neither with a `model.onnx` beside it. The version number implied
a trained artifact that never existed.

Worse than empty, it was *wrong*. It declared a ten-feature input:

```
hfr, fwhm, star_count, temperature, temperature_delta,
filter_index, altitude, current_position, background_adu, quality_score
```

against a specification that defines **thirty-eight** features in nine groups.
Three of those ten name quantities the daemon does not produce at all
(`altitude`, `quality_score`), and `filter_index` was an ordinal encoding where
the spec calls for a seven-way one-hot — an ordinal filter index asserts that
Ha sits between B and OIII on some scale, which is not a thing.

It was one of three disagreeing descriptions of the FOP input, and the least
correct of them.

## Where the input contract lives now

**`ml/contracts/fop_features.py`**, generated into `fop_features_v1.json`
beside it. That file is the single authority for feature names, vector order,
normalization, and — the part that matters before any of this is trained — the
database column or JSON key behind each feature, plus an honest verdict on
whether the system can produce it today.

```bash
python -m contracts.fop_features   # regenerate the JSON, print what is usable
```

`ml/tests/test_fop_contract.py` fails if the module and the JSON drift apart,
and a test in `achernar-daemon/src/api/focus.rs` fails if the daemon stops
writing a `frames.quality_json` key the contract claims to read.

## What a real bundle here will contain

Per Section 7.6.10, once there is data to train on:

- `model.onnx` — the distilled MLP of Listing 11, INT8, three heads
  (`delta_steps`, `log_sigma`, `direction_logits`)
- `metadata.json` — with a non-null `sha256`, `created_at`, and a
  `training_info` block; the daemon's `model_registry` ignores bundles without
  one, which is why the old directory was inert rather than dangerous
- `norm_params.json` — real fitted statistics, and an explicit record of which
  feature blocks were **dropped** as degenerate. The filter one-hot will be
  among them until a wheel exists; leaving seven dead inputs in the graph is how
  a future filtered rig silently misuses them.

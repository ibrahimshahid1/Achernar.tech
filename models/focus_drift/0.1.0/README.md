# focus_drift 0.1.0

The focus-drift forecaster (Section 7.6.3) is a 3-state Kalman filter
implemented in `achernar-daemon/src/focus/focus_drift.rs`. It is not a neural
network, so this directory holds **no** `model.onnx` and never will. The
`drift_model_path` key that used to point at one has been removed from
`FocusModelsConfig` — it was conceptually wrong, not merely unset.

## What used to be here

`kalman_state.json` was removed. It was trying to be two things at once:

|                 | Tuning constants     | Learned filter state |
| --------------- | -------------------- | -------------------- |
| Lifecycle       | ships with the code  | changes every night  |
| Scope           | same for everyone    | one specific telescope |
| Version control | yes                  | never                |

Because it mixed them it could serve neither, and
`FocusDriftForecaster::from_state()` could not read it: the file's nested shape
(`state_vector`, `process_noise_Q`, `measurement_noise_R`,
`initial_covariance_P`) did not match the flat `DriftState` the code actually
serialises (`focus_position`, `drift_rate`, `temperature_coefficient`,
`covariance`, …).

## Where each half lives now

- **Tuning** → `[focus.drift]` in `achernar.toml`. The optics-dependent values
  (`f_ratio`, `steps_per_micron`) are per-rig facts the operator knows and the
  daemon cannot derive. Absent them the forecaster stays disabled and says so,
  rather than falling back to a 50-step threshold nobody chose.
- **Defaults and their physics** → `DriftConfig` in `focus_drift.rs`. The doc
  comment on `initial_temp_coeff_var` carries the argument the old file's
  `_note` made: a 900 mm aluminium tube shifts ≈21 µm/°C, which is tens of
  focuser steps, so σ = 20 steps/°C (variance 400) is the physically honest
  prior. An earlier value of 0.1 (σ ≈ 0.32 steps/°C) asserted the coefficient
  was essentially zero and pinned it there no matter what the data said.
- **Learned state** → `<data_dir>/focus_drift_state.json`, written atomically by
  `storage::focus_drift_repo::save_state` and git-ignored. Path configurable via
  `[focus.drift] state_path`.

## profile_registry.json

Still a placeholder (`"profiles": {}`). Multi-rig drift profiles are not
implemented: the current forecaster holds one filter for the running daemon, and
its state file is already per-installation. Left in place because the schema
block documents the intended shape.

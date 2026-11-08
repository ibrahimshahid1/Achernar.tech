//! FOP features (equipment spec H7, v6 §7.6.2). Owner: WP1B-2.
//!
//! Builds the Focus Offset Predictor's input vector online, one vector per
//! autofocus sample, from what the equipment layer and the vision loop know
//! at the moment of capture. The authoritative description of the vector
//! (names, order, groups, normalisation and the database column behind each
//! feature) is `ml/contracts/fop_features.py`, generated into
//! `fop_features_v1.json`; [`FEATURE_NAMES`] mirrors it and a test checks the
//! two against each other whenever the ML tree is present.
//!
//! # What a value means
//!
//! Every value is the feature's **raw quantity in contract units, before the
//! contract's normalisation** (`z-score`, `p / p_max`, `/ max step`, `log`,
//! ...). The z-scores need the training run's `norm_params.json`, which only a
//! model bundle carries, so normalisation is applied by the FOP runtime that
//! loads the model, never here. [`FopFeatureVector::context`] carries the
//! device constants a parameter-free normalisation needs (`p_max`).
//!
//! # Missing stays missing
//!
//! A feature the system cannot supply is `None`, with a reason in
//! [`FopFeatureVector::missing`]. Nothing is imputed:
//!
//! - `sharpness_score` (#1) is rejected by the contract; `fit_quality` (#5)
//!   needs the partial-fit rule training has not fixed yet (the contract's
//!   parity hazard), so it is never guessed online.
//! - Ambient temperature (#12) comes from the H6 selection (weather, then a
//!   focuser probe whose role is ambient or tube). The camera sensor is never
//!   substituted for it; a stale reading is not presented as current.
//! - The sensor temperature (#13, #14) is used only when it was measured.
//! - The filter block (#16-#22) and its offset (#34) need a *verified* wheel
//!   slot (H8); a client label is not verification.
//! - Backlash (#32) is the compensation Achernar applies: `0` when its model
//!   is `none` (the contract's truthful default), the steps of the travel
//!   direction for `overshoot` and `absolute`, and missing when the firmware
//!   owns backlash, because Achernar then does not know the amount. The
//!   travel direction is that of the move that brought the focuser to the
//!   sample: the last non-zero `dp`, or for the run's first position the
//!   move from the run's start position. With no move at all and unequal in
//!   and out steps it is missing (`backlash_direction_unknown`).
//!
//!   This differs from the contract's training source,
//!   `focuser_profiles.backlash_steps` (one value, defaulting to 0). ML
//!   change note (e) for the contract owner: the online #32 is the
//!   direction-dependent compensation above, missing when the firmware owns
//!   backlash; the training builder must mirror it (from
//!   `focus_samples.backlash_model` and the sample's move direction) or the
//!   contract must change.
//! - The CFZ (#33) needs the focal ratio and a V-curve or configured
//!   steps-per-micron scale.
//!
//! # Labels
//!
//! The label (`delta_steps = best_position - position`) is retrospective: it
//! exists once the run converges. A sample whose focuser origin is not
//! reliable (`emulated_origin_unknown`, `unknown_after_reconnect`) never gets
//! one (v6 §7.6.2 "reliable absolute position", F-b).
//!
//! # History features
//!
//! For the run's samples `s_0 .. s_n` in capture order (`s_n` is this
//! sample), with positions `p_i`, HFRs `h_i`, `dp_i = p_i - p_(i-1)` and
//! `dh_i = h_i - h_(i-1)`:
//!
//! - `sample_idx_norm` is `n`; `dir_encoded` is `sign(dp_n)`, 0 for `s_0`;
//! - `prev_delta_k` is `dp_(n-k+1)` and `prev_hfr_delta_k` is `dh_(n-k+1)`
//!   (k = 1 is the move that brought the focuser to this sample), missing
//!   while the run has fewer than `k` moves;
//! - `dir_changed_recently` is 1.0 when two consecutive non-zero signs among
//!   the last three `dp` differ (repeated samples at one position have
//!   `dp = 0` and are skipped);
//! - `steps_since_flip_norm` sums `|dp|` back to the last reversal (the whole
//!   run when there is none);
//! - `consec_improvements_norm` counts the trailing run of `dh < 0`.

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::{json, Value};

use crate::equipment::snapshot::{
    BacklashModel, BacklashOwner, FocuserSnapshot, PositionSource, TemperatureReading,
};
use crate::equipment::types::{ObsState, Observed};

/// `ml/contracts/fop_features.py` `CONTRACT_VERSION`.
pub const CONTRACT_VERSION: &str = "1.0.0";

/// `ml/contracts/fop_features.py` `N_FEATURES`.
pub const N_FEATURES: usize = 38;

/// Feature names in contract order (index = position in the vector).
pub const FEATURE_NAMES: [&str; N_FEATURES] = [
    "predicted_hfr",
    "sharpness_score",
    "quality_uncertainty",
    "classical_hfr",
    "classical_fwhm",
    "fit_quality",
    "star_count",
    "pos_norm",
    "dist_from_start_norm",
    "sample_idx_norm",
    "dir_encoded",
    "run_initial_hfr",
    "ambient_temp_norm",
    "sensor_temp_norm",
    "temp_delta_from_run_start",
    "temp_coeff_norm",
    "filter_l",
    "filter_r",
    "filter_g",
    "filter_b",
    "filter_ha",
    "filter_oiii",
    "filter_sii",
    "prev_delta_1",
    "prev_delta_2",
    "prev_delta_3",
    "prev_hfr_delta_1",
    "prev_hfr_delta_2",
    "prev_hfr_delta_3",
    "dir_changed_recently",
    "steps_since_flip_norm",
    "consec_improvements_norm",
    "backlash_norm",
    "cfz_norm",
    "filter_offset_norm",
    "background_norm",
    "saturation_frac",
    "humidity_norm",
];

/// The filters of the one-hot block (#16-#22), matched case-insensitively.
const FILTER_ONE_HOT: [&str; 7] = ["l", "r", "g", "b", "ha", "oiii", "sii"];
const FILTER_BLOCK_START: usize = 16;

/// The contract's CFZ factor (`4.88 * lambda * N^2`, #33).
const CFZ_FACTOR: f64 = 4.88;

/// One earlier sample of the same run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HistorySample {
    /// `focus_samples.position` (the commanded sweep position).
    pub position: i32,
    /// `focus_samples.hfr`.
    pub hfr: f64,
    /// The camera-reported temperature, only when it was measured.
    pub sensor_temperature_c: Option<f64>,
}

/// A wheel slot verified by the filter handshake (H8).
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedFilter {
    pub name: String,
    /// The slot's reconciled focus offset (profile, else device).
    pub offset_steps: Option<i32>,
}

/// What the CFZ needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CfzInputs {
    pub f_ratio: f64,
    pub wavelength_nm: f64,
    pub steps_per_micron: f64,
}

/// Everything one vector is built from. Nothing here is a default: a value
/// the caller does not have is `None`.
#[derive(Debug, Clone)]
pub struct FopInputs<'a> {
    /// This sample's commanded position.
    pub position: i32,
    pub run_start_position: i32,
    /// The run's earlier samples, in capture order.
    pub history: &'a [HistorySample],
    pub classical_hfr: f64,
    pub classical_fwhm: Option<f64>,
    pub star_count: u32,
    pub fqe_hfr_px: Option<f64>,
    pub fqe_sigma_log_hfr: Option<f64>,
    pub quality_uncertainty: Option<f64>,
    pub background_adu: Option<f64>,
    pub saturation_fraction: Option<f64>,
    pub saturation_gate_applied: bool,
    /// Camera-reported temperature, only when measured.
    pub sensor_temperature_c: Option<f64>,
    /// The H6 ambient selection at the sample.
    pub ambient: &'a Observed<TemperatureReading>,
    /// Relative humidity from a weather device, only when fresh.
    pub humidity_pct: Option<f64>,
    /// The FDF's learned coefficient (steps/°C), once it has one.
    pub temperature_coefficient: Option<f64>,
    /// The focuser at the sample.
    pub focuser: Option<&'a FocuserSnapshot>,
    pub filter: Option<&'a VerifiedFilter>,
    /// Why there is no verified filter, when there is none.
    pub filter_missing_reason: &'static str,
    pub cfz: Option<CfzInputs>,
}

/// Device constants for the parameter-free normalisations.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize)]
pub struct FopContext {
    /// The focuser's `MaxStep` (the contract's `p_max`), when it reports one.
    pub p_max: Option<i32>,
    pub position_source: Option<PositionSource>,
}

/// The FOP input vector of one sample.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FopFeatureVector {
    pub contract_version: &'static str,
    /// [`N_FEATURES`] raw values in contract order; `None` is missing.
    pub values: Vec<Option<f64>>,
    /// Feature name to the reason it is missing.
    pub missing: BTreeMap<&'static str, &'static str>,
    /// Whether this sample may carry a label (reliable focuser origin).
    pub label_eligible: bool,
    pub context: FopContext,
}

impl FopFeatureVector {
    /// The value of the feature called `name`.
    #[cfg(test)]
    pub fn get(&self, name: &str) -> Option<f64> {
        let index = FEATURE_NAMES.iter().position(|n| *n == name)?;
        self.values.get(index).copied().flatten()
    }

    /// `delta_steps = best_position - position` once the run converged, and
    /// only for a sample with a reliable focuser origin.
    pub fn label(&self, position: i32, best_position: Option<i32>) -> Option<f64> {
        if !self.label_eligible {
            return None;
        }
        best_position.map(|best| f64::from(best) - f64::from(position))
    }

    /// The JSON written beside the frame (`frames.quality_json ->
    /// fop_features`).
    pub fn to_json(&self) -> Value {
        json!({
            "contract_version": self.contract_version,
            "values": self.values,
            "missing": self.missing,
            "label_eligible": self.label_eligible,
            "context": self.context,
        })
    }
}

struct Builder {
    values: Vec<Option<f64>>,
    missing: BTreeMap<&'static str, &'static str>,
}

impl Builder {
    fn new() -> Self {
        Self {
            values: vec![None; N_FEATURES],
            missing: BTreeMap::new(),
        }
    }

    /// Set feature `index`; a `None` (or non-finite) value is recorded as
    /// missing with `reason`.
    fn set(&mut self, index: usize, value: Option<f64>, reason: &'static str) {
        match value.filter(|v| v.is_finite()) {
            Some(v) => self.values[index] = Some(v),
            None => {
                self.values[index] = None;
                self.missing.insert(FEATURE_NAMES[index], reason);
            }
        }
    }
}

fn sign(value: i64) -> f64 {
    match value.cmp(&0) {
        std::cmp::Ordering::Greater => 1.0,
        std::cmp::Ordering::Less => -1.0,
        std::cmp::Ordering::Equal => 0.0,
    }
}

/// The ambient temperature for #12: a fresh H6 reading that is not the
/// camera-sensor fallback.
fn ambient_celsius(ambient: &Observed<TemperatureReading>) -> Result<f64, &'static str> {
    let Some(reading) = ambient.known() else {
        return Err("no_ambient_temperature");
    };
    if reading.camera_sensor_fallback {
        return Err("camera_sensor_is_not_ambient");
    }
    if ambient.state != ObsState::Fresh {
        return Err("ambient_temperature_stale");
    }
    if reading.celsius.is_finite() {
        Ok(reading.celsius)
    } else {
        Err("no_ambient_temperature")
    }
}

/// The backlash compensation Achernar applies for a move in `direction`
/// (#32): `Some(+1.0)` outward, `Some(-1.0)` inward, `None` when no move of
/// the run brought the focuser to the sample. Without a direction the value
/// is known only when both directions compensate alike.
fn backlash_steps(focuser: &FocuserSnapshot, direction: Option<f64>) -> Result<f64, &'static str> {
    if focuser.backlash_owner == BacklashOwner::Firmware {
        return Err("backlash_firmware_owned");
    }
    let (in_steps, out_steps) = match focuser.backlash_model {
        BacklashModel::None => return Ok(0.0),
        BacklashModel::Overshoot {
            in_steps,
            out_steps,
        }
        | BacklashModel::Absolute {
            in_steps,
            out_steps,
        } => (in_steps, out_steps),
    };
    let steps = match direction {
        Some(d) if d > 0.0 => out_steps,
        Some(d) if d < 0.0 => in_steps,
        _ if in_steps == out_steps => in_steps,
        _ => return Err("backlash_direction_unknown"),
    };
    Ok(f64::from(steps.max(0)))
}

/// Build one sample's vector.
pub fn build(inputs: &FopInputs<'_>) -> FopFeatureVector {
    let mut b = Builder::new();

    // The run so far, this sample last.
    let mut positions: Vec<i64> = inputs
        .history
        .iter()
        .map(|s| i64::from(s.position))
        .collect();
    positions.push(i64::from(inputs.position));
    let mut hfrs: Vec<f64> = inputs.history.iter().map(|s| s.hfr).collect();
    hfrs.push(inputs.classical_hfr);
    let n = positions.len() - 1;
    let dp: Vec<i64> = positions.windows(2).map(|w| w[1] - w[0]).collect();
    let dh: Vec<f64> = hfrs.windows(2).map(|w| w[1] - w[0]).collect();

    // Quality (#0-#2).
    b.set(0, inputs.fqe_hfr_px, "fqe_not_loaded");
    b.set(1, None, "rejected_by_contract");
    b.set(
        2,
        inputs.fqe_sigma_log_hfr.or(inputs.quality_uncertainty),
        "uncertainty_unavailable",
    );

    // Classical (#3-#6).
    b.set(3, Some(inputs.classical_hfr), "hfr_unavailable");
    b.set(4, inputs.classical_fwhm, "no_psf_fit");
    b.set(5, None, "partial_fit_rule_undefined");
    b.set(
        6,
        Some(f64::from(inputs.star_count)),
        "star_count_unavailable",
    );

    // Position (#7-#11).
    b.set(7, Some(f64::from(inputs.position)), "position_unknown");
    b.set(
        8,
        Some((i64::from(inputs.position) - i64::from(inputs.run_start_position)) as f64),
        "position_unknown",
    );
    b.set(9, Some(n as f64), "position_unknown");
    b.set(
        10,
        Some(dp.last().copied().map_or(0.0, sign)),
        "position_unknown",
    );
    b.set(11, hfrs.first().copied(), "hfr_unavailable");

    // Temperature (#12-#15).
    match ambient_celsius(inputs.ambient) {
        Ok(celsius) => b.set(12, Some(celsius), "no_ambient_temperature"),
        Err(reason) => b.set(12, None, reason),
    }
    b.set(
        13,
        inputs.sensor_temperature_c,
        "sensor_temperature_not_measured",
    );
    let first_sensor = inputs
        .history
        .first()
        .map_or(inputs.sensor_temperature_c, |s| s.sensor_temperature_c);
    let delta = match (inputs.sensor_temperature_c, first_sensor) {
        (Some(now), Some(first)) => Some(now - first),
        _ => None,
    };
    b.set(14, delta, "sensor_temperature_not_measured");
    b.set(
        15,
        inputs.temperature_coefficient,
        "fdf_coefficient_not_learned",
    );

    // Filter (#16-#22) and its offset (#34): a verified slot only.
    match inputs.filter {
        Some(filter) => {
            let name = filter.name.trim().to_ascii_lowercase();
            for (i, candidate) in FILTER_ONE_HOT.iter().enumerate() {
                let hot = if name == *candidate { 1.0 } else { 0.0 };
                b.set(FILTER_BLOCK_START + i, Some(hot), "filter_unverified");
            }
            b.set(
                34,
                filter.offset_steps.map(f64::from),
                "filter_offset_unknown",
            );
        }
        None => {
            for i in 0..FILTER_ONE_HOT.len() {
                b.set(FILTER_BLOCK_START + i, None, inputs.filter_missing_reason);
            }
            b.set(34, None, inputs.filter_missing_reason);
        }
    }

    // History (#23-#28).
    for k in 1..=3usize {
        let index = dp.len().checked_sub(k);
        b.set(
            22 + k,
            index.map(|i| dp[i] as f64),
            "insufficient_run_history",
        );
        b.set(25 + k, index.map(|i| dh[i]), "insufficient_run_history");
    }

    // Direction (#29-#31).
    let recent: Vec<f64> = dp
        .iter()
        .rev()
        .take(3)
        .map(|d| sign(*d))
        .filter(|s| *s != 0.0)
        .collect();
    let reversed = recent.windows(2).any(|w| w[0] != w[1]);
    b.set(
        29,
        Some(if reversed { 1.0 } else { 0.0 }),
        "position_unknown",
    );
    let direction = dp.iter().rev().map(|d| sign(*d)).find(|s| *s != 0.0);
    let mut since_flip = 0i64;
    if let Some(direction) = direction {
        for d in dp.iter().rev() {
            if sign(*d) == -direction {
                break;
            }
            since_flip += d.abs();
        }
    }
    b.set(30, Some(since_flip as f64), "position_unknown");
    let improvements = dh.iter().rev().take_while(|d| **d < 0.0).count();
    b.set(31, Some(improvements as f64), "hfr_unavailable");

    // Rig (#32, #33). The move that brought the focuser to this sample: the
    // last non-zero `dp`, else (the run's first position, and repeats at it)
    // the move from the run's start position.
    let move_direction = direction.or_else(|| {
        let d = sign(i64::from(inputs.position) - i64::from(inputs.run_start_position));
        (d != 0.0).then_some(d)
    });
    match inputs.focuser {
        Some(focuser) => match backlash_steps(focuser, move_direction) {
            Ok(steps) => b.set(32, Some(steps), "backlash_unknown"),
            Err(reason) => b.set(32, None, reason),
        },
        None => b.set(32, None, "no_focuser_state"),
    }
    let cfz = inputs.cfz.and_then(|c| {
        let ok = c.f_ratio > 0.0 && c.wavelength_nm > 0.0 && c.steps_per_micron > 0.0;
        ok.then(|| CFZ_FACTOR * (c.wavelength_nm / 1000.0) * c.f_ratio.powi(2) * c.steps_per_micron)
    });
    b.set(33, cfz, "optics_unknown");

    // Sky (#35-#37).
    b.set(35, inputs.background_adu, "background_unavailable");
    let saturation = inputs
        .saturation_fraction
        .filter(|_| inputs.saturation_gate_applied);
    b.set(36, saturation, "saturation_gate_not_applied");
    b.set(37, inputs.humidity_pct, "no_humidity_reading");

    let position_source = inputs.focuser.map(|f| f.position_source);
    FopFeatureVector {
        contract_version: CONTRACT_VERSION,
        values: b.values,
        missing: b.missing,
        label_eligible: position_source.is_some_and(PositionSource::is_reliable),
        context: FopContext {
            p_max: inputs
                .focuser
                .and_then(|f| f.capabilities.max_step)
                .filter(|m| *m > 0),
            position_source,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::snapshot::TemperatureSourceKind;
    use crate::equipment::types::{Source, TemperatureRole};

    fn ambient(celsius: f64, state: ObsState, camera: bool) -> Observed<TemperatureReading> {
        let mut observed = Observed::fresh(
            TemperatureReading {
                celsius,
                source: if camera {
                    TemperatureSourceKind::CameraSensor
                } else {
                    TemperatureSourceKind::FocuserProbe
                },
                role: TemperatureRole::Tube,
                identity_key: Some("alpaca:focuser".into()),
                connection_id: None,
                camera_sensor_fallback: camera,
            },
            Source::Derived,
            chrono::Utc::now(),
            0,
        );
        observed.state = state;
        observed
    }

    fn focuser(
        source: PositionSource,
        model: BacklashModel,
        owner: BacklashOwner,
    ) -> FocuserSnapshot {
        let mut snap = FocuserSnapshot {
            position_source: source,
            backlash_model: model,
            backlash_owner: owner,
            ..FocuserSnapshot::default()
        };
        snap.capabilities.absolute = true;
        snap.capabilities.max_step = Some(60_000);
        snap
    }

    fn inputs<'a>(
        history: &'a [HistorySample],
        ambient: &'a Observed<TemperatureReading>,
        focuser: Option<&'a FocuserSnapshot>,
    ) -> FopInputs<'a> {
        FopInputs {
            position: 1_200,
            run_start_position: 1_000,
            history,
            classical_hfr: 2.5,
            classical_fwhm: None,
            star_count: 40,
            fqe_hfr_px: None,
            fqe_sigma_log_hfr: None,
            quality_uncertainty: None,
            background_adu: Some(812.0),
            saturation_fraction: Some(0.0),
            saturation_gate_applied: false,
            sensor_temperature_c: None,
            ambient,
            humidity_pct: None,
            temperature_coefficient: None,
            focuser,
            filter: None,
            filter_missing_reason: "no_filter_wheel",
            cfz: None,
        }
    }

    #[test]
    fn the_names_match_the_committed_contract() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../ml/contracts/fop_features_v1.json");
        let Ok(text) = std::fs::read_to_string(&path) else {
            // The daemon builds without the ML tree beside it.
            return;
        };
        let contract: Value = serde_json::from_str(&text).expect("contract JSON parses");
        assert_eq!(contract["contract_version"], CONTRACT_VERSION);
        assert_eq!(contract["n_features"], N_FEATURES);
        let features = contract["features"].as_array().expect("features");
        assert_eq!(features.len(), N_FEATURES);
        for feature in features {
            let index = feature["index"].as_u64().unwrap() as usize;
            assert_eq!(
                feature["name"].as_str().unwrap(),
                FEATURE_NAMES[index],
                "feature {index} differs from the contract"
            );
        }
    }

    /// The acceptance rule "FOP vectors contain no fabricated values": with
    /// nothing but a classical measurement, every feature the inputs do not
    /// support is missing with a reason, and none is filled with a stand-in.
    #[test]
    fn a_bare_sample_fabricates_nothing() {
        let ambient = Observed::not_set(Source::Derived);
        let v = build(&inputs(&[], &ambient, None));
        assert_eq!(v.values.len(), N_FEATURES);
        for name in [
            "predicted_hfr",
            "sharpness_score",
            "quality_uncertainty",
            "classical_fwhm",
            "fit_quality",
            "ambient_temp_norm",
            "sensor_temp_norm",
            "temp_delta_from_run_start",
            "temp_coeff_norm",
            "filter_l",
            "filter_sii",
            "prev_delta_1",
            "prev_hfr_delta_3",
            "backlash_norm",
            "cfz_norm",
            "filter_offset_norm",
            "saturation_frac",
            "humidity_norm",
        ] {
            assert_eq!(v.get(name), None, "{name} must be missing");
            assert!(v.missing.contains_key(name), "{name} needs a reason");
        }
        // What the sample itself establishes is present.
        assert_eq!(v.get("classical_hfr"), Some(2.5));
        assert_eq!(v.get("star_count"), Some(40.0));
        assert_eq!(v.get("pos_norm"), Some(1_200.0));
        assert_eq!(v.get("dist_from_start_norm"), Some(200.0));
        assert_eq!(v.get("sample_idx_norm"), Some(0.0));
        assert_eq!(v.get("dir_encoded"), Some(0.0));
        assert_eq!(v.get("run_initial_hfr"), Some(2.5));
        assert_eq!(v.get("background_norm"), Some(812.0));
        // Unknown focuser state: no label.
        assert!(!v.label_eligible);
        assert_eq!(v.label(1_200, Some(1_250)), None);
    }

    #[test]
    fn the_camera_sensor_and_stale_readings_never_stand_in_for_ambient() {
        let snap = focuser(
            PositionSource::Device,
            BacklashModel::None,
            BacklashOwner::Achernar,
        );
        let camera = ambient(30.0, ObsState::Fresh, true);
        let v = build(&inputs(&[], &camera, Some(&snap)));
        assert_eq!(v.get("ambient_temp_norm"), None);
        assert_eq!(
            v.missing.get("ambient_temp_norm"),
            Some(&"camera_sensor_is_not_ambient")
        );
        let stale = ambient(8.0, ObsState::Stale, false);
        let v = build(&inputs(&[], &stale, Some(&snap)));
        assert_eq!(v.get("ambient_temp_norm"), None);
        let fresh = ambient(8.0, ObsState::Fresh, false);
        let v = build(&inputs(&[], &fresh, Some(&snap)));
        assert_eq!(v.get("ambient_temp_norm"), Some(8.0));
    }

    #[test]
    fn history_features_follow_the_documented_definitions() {
        // 1000 -> 1100 -> 1200 -> 1100 (a reversal), HFR 4.0, 3.0, 2.0, 2.4.
        let history = [
            HistorySample {
                position: 1_000,
                hfr: 4.0,
                sensor_temperature_c: Some(10.0),
            },
            HistorySample {
                position: 1_100,
                hfr: 3.0,
                sensor_temperature_c: Some(9.5),
            },
            HistorySample {
                position: 1_200,
                hfr: 2.0,
                sensor_temperature_c: Some(9.0),
            },
        ];
        let ambient = Observed::not_set(Source::Derived);
        let snap = focuser(
            PositionSource::Device,
            BacklashModel::Overshoot {
                in_steps: 30,
                out_steps: 80,
            },
            BacklashOwner::Achernar,
        );
        let mut i = inputs(&history, &ambient, Some(&snap));
        i.position = 1_100;
        i.classical_hfr = 2.4;
        i.sensor_temperature_c = Some(8.5);
        let v = build(&i);
        assert_eq!(v.get("sample_idx_norm"), Some(3.0));
        assert_eq!(v.get("dir_encoded"), Some(-1.0));
        assert_eq!(v.get("prev_delta_1"), Some(-100.0));
        assert_eq!(v.get("prev_delta_2"), Some(100.0));
        assert_eq!(v.get("prev_delta_3"), Some(100.0));
        assert!((v.get("prev_hfr_delta_1").unwrap() - 0.4).abs() < 1e-12);
        assert!((v.get("prev_hfr_delta_3").unwrap() + 1.0).abs() < 1e-12);
        assert_eq!(v.get("dir_changed_recently"), Some(1.0));
        assert_eq!(v.get("steps_since_flip_norm"), Some(100.0));
        assert_eq!(v.get("consec_improvements_norm"), Some(0.0));
        assert_eq!(v.get("run_initial_hfr"), Some(4.0));
        assert_eq!(v.get("sensor_temp_norm"), Some(8.5));
        assert!((v.get("temp_delta_from_run_start").unwrap() + 1.5).abs() < 1e-12);
        // Moving in: the inward overshoot is the applied compensation.
        assert_eq!(v.get("backlash_norm"), Some(30.0));
        assert!(v.label_eligible);
        assert_eq!(v.label(1_100, Some(1_180)), Some(80.0));
        assert_eq!(v.context.p_max, Some(60_000));
    }

    #[test]
    fn an_unreliable_origin_never_gets_a_label() {
        let ambient = Observed::not_set(Source::Derived);
        for source in [
            PositionSource::EmulatedOriginUnknown,
            PositionSource::UnknownAfterReconnect,
        ] {
            let snap = focuser(source, BacklashModel::None, BacklashOwner::Achernar);
            let v = build(&inputs(&[], &ambient, Some(&snap)));
            assert!(!v.label_eligible);
            assert_eq!(v.label(1_200, Some(1_300)), None);
        }
    }

    #[test]
    fn firmware_backlash_is_missing_and_achernar_none_is_zero() {
        let ambient = Observed::not_set(Source::Derived);
        let firmware = focuser(
            PositionSource::Device,
            BacklashModel::None,
            BacklashOwner::Firmware,
        );
        let v = build(&inputs(&[], &ambient, Some(&firmware)));
        assert_eq!(v.get("backlash_norm"), None);
        assert_eq!(
            v.missing.get("backlash_norm"),
            Some(&"backlash_firmware_owned")
        );
        let none = focuser(
            PositionSource::Device,
            BacklashModel::None,
            BacklashOwner::Achernar,
        );
        let v = build(&inputs(&[], &ambient, Some(&none)));
        assert_eq!(v.get("backlash_norm"), Some(0.0));
    }

    #[test]
    fn the_first_samples_backlash_follows_the_move_from_the_run_start() {
        let ambient = Observed::not_set(Source::Derived);
        let snap = focuser(
            PositionSource::Device,
            BacklashModel::Overshoot {
                in_steps: 30,
                out_steps: 80,
            },
            BacklashOwner::Achernar,
        );
        // s_0 at 1200 from a start of 1000: an outward move.
        let mut i = inputs(&[], &ambient, Some(&snap));
        assert_eq!(build(&i).get("backlash_norm"), Some(80.0));
        // s_0 at 900: inward.
        i.position = 900;
        assert_eq!(build(&i).get("backlash_norm"), Some(30.0));
        // A repeat at s_0 keeps the direction of the move to s_0.
        let history = [HistorySample {
            position: 900,
            hfr: 3.0,
            sensor_temperature_c: None,
        }];
        let mut repeat = inputs(&history, &ambient, Some(&snap));
        repeat.position = 900;
        assert_eq!(build(&repeat).get("backlash_norm"), Some(30.0));
        // No move at all: unknown for unequal steps, never max(in, out).
        i.position = 1_000;
        let v = build(&i);
        assert_eq!(v.get("backlash_norm"), None);
        assert_eq!(
            v.missing.get("backlash_norm"),
            Some(&"backlash_direction_unknown")
        );
        let symmetric = focuser(
            PositionSource::Device,
            BacklashModel::Absolute {
                in_steps: 50,
                out_steps: 50,
            },
            BacklashOwner::Achernar,
        );
        let mut s = inputs(&[], &ambient, Some(&symmetric));
        s.position = 1_000;
        assert_eq!(build(&s).get("backlash_norm"), Some(50.0));
    }

    #[test]
    fn a_verified_filter_fills_the_block_and_its_offset() {
        let ambient = Observed::not_set(Source::Derived);
        let filter = VerifiedFilter {
            name: "Ha".into(),
            offset_steps: Some(120),
        };
        let mut i = inputs(&[], &ambient, None);
        i.filter = Some(&filter);
        let v = build(&i);
        let block: Vec<Option<f64>> = (16..23).map(|k| v.values[k]).collect();
        assert_eq!(
            block,
            vec![
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(1.0),
                Some(0.0),
                Some(0.0)
            ]
        );
        assert_eq!(v.get("filter_offset_norm"), Some(120.0));
    }

    #[test]
    fn cfz_needs_every_optical_input_and_uses_the_contract_factor() {
        let ambient = Observed::not_set(Source::Derived);
        let mut i = inputs(&[], &ambient, None);
        i.cfz = Some(CfzInputs {
            f_ratio: 5.0,
            wavelength_nm: 550.0,
            steps_per_micron: 2.0,
        });
        let v = build(&i);
        let expected = 4.88 * 0.55 * 25.0 * 2.0;
        assert!((v.get("cfz_norm").unwrap() - expected).abs() < 1e-9);
        i.cfz = Some(CfzInputs {
            f_ratio: 5.0,
            wavelength_nm: 550.0,
            steps_per_micron: 0.0,
        });
        assert_eq!(build(&i).get("cfz_norm"), None);
    }

    #[test]
    fn the_saturation_fraction_needs_the_gate() {
        let ambient = Observed::not_set(Source::Derived);
        let mut i = inputs(&[], &ambient, None);
        i.saturation_fraction = Some(0.02);
        assert_eq!(build(&i).get("saturation_frac"), None);
        i.saturation_gate_applied = true;
        assert_eq!(build(&i).get("saturation_frac"), Some(0.02));
    }
}

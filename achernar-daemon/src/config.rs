//! Configuration parsing for achernar.toml.
//! Implements all config sections including [focus], [focus.safety],
//! [focus.backlash], [focus.models], [mount], [filter_wheel].

use serde::Deserialize;

/// Top-level daemon configuration parsed from achernar.toml.
#[derive(Debug, Clone, Deserialize)]
pub struct AchernarConfig {
    #[serde(default)]
    pub daemon: DaemonConfig,
    #[serde(default)]
    pub focus: FocusConfig,
    #[serde(default)]
    pub mount: MountConfig,
    #[serde(default)]
    pub filter_wheel: FilterWheelConfig,
    #[serde(default)]
    pub camera: CameraConfig,
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DaemonConfig {
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default)]
    pub data_dir: Option<String>,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            port: default_port(),
            log_level: default_log_level(),
            data_dir: None,
        }
    }
}

/// Focus subsystem configuration — Section 7.3, 7.4.
#[derive(Debug, Clone, Deserialize)]
pub struct FocusConfig {
    #[serde(default)]
    pub safety: FocusSafetyConfig,
    #[serde(default)]
    pub backlash: BacklashConfig,
    #[serde(default)]
    pub models: FocusModelsConfig,
    #[serde(default)]
    pub drift: FocusDriftConfig,
    /// Autofocus step size for initial sweep
    #[serde(default = "default_step_size")]
    pub initial_step_size: i32,
    /// Number of samples per focus position
    #[serde(default = "default_samples_per_position")]
    pub samples_per_position: u32,
    /// Settle time in ms after each focus move
    #[serde(default = "default_settle_ms")]
    pub settle_ms: u64,

    /// Half-width of the sweep, in sample points either side of the start.
    ///
    /// The sweep visits `2 * sweep_radius_points + 1` positions, so the default
    /// of 3 gives the seven-point sweep this agent has always run. It used to be
    /// a hard-coded `(-3..=3)`, which is enough to fit a curve and not enough
    /// for the data campaign: the observing plan calls for 25-position sweeps,
    /// and the Focus Offset Predictor's balanced sampler buckets |delta| out to
    /// 120+ steps, which a +/-3 sweep at a 50-step size can only just reach at
    /// its extremes. Raise it to 12 for campaign nights.
    ///
    /// Still bounded by `safety.max_total_travel_per_run`, which silently
    /// truncates the sweep rather than overrunning the budget.
    #[serde(default = "default_sweep_radius_points")]
    pub sweep_radius_points: u32,

    /// The order positions are visited in. See `focus_agent::SweepPattern`.
    ///
    /// Overridable per run through `POST /focus/run`, because the mix is an
    /// observing decision rather than an installation one — the plan calls for
    /// roughly one run in three to be a reversal.
    #[serde(default)]
    pub sweep_pattern: crate::focus::focus_agent::SweepPattern,
}

impl Default for FocusConfig {
    fn default() -> Self {
        Self {
            safety: FocusSafetyConfig::default(),
            backlash: BacklashConfig::default(),
            models: FocusModelsConfig::default(),
            drift: FocusDriftConfig::default(),
            initial_step_size: default_step_size(),
            samples_per_position: default_samples_per_position(),
            settle_ms: default_settle_ms(),
            sweep_radius_points: default_sweep_radius_points(),
            sweep_pattern: crate::focus::focus_agent::SweepPattern::default(),
        }
    }
}

/// Focus-drift forecaster configuration — Section 7.6.3.
///
/// Lives here rather than in `models/focus_drift/<version>/kalman_state.json`,
/// which used to hold a confused mixture of two different things: tuning
/// constants that ship with the code, and a filter state learned at runtime.
/// They have opposite lifecycles — one is version-controlled and read-only, the
/// other is per-rig, changes every night, and must never be committed — so they
/// are now separated. Tuning is here; the learned state is a snapshot in the
/// data directory (see `FocusDriftConfig::state_path`).
///
/// Everything is optional. Defaults come from `focus::focus_drift::DriftConfig`,
/// whose doc comments carry the physical justification for each prior.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct FocusDriftConfig {
    /// Master switch. The forecaster is advisory even when on: it publishes a
    /// prediction on `/focus/drift`, it does not move the focuser.
    #[serde(default)]
    pub enabled: bool,

    // ---- Refocus threshold ------------------------------------------------
    // Section 7.6.3 sets this to half the critical focus zone,
    // CFZ = 4 * lambda * N^2. A step count guessed without the optics has no
    // physical meaning, so there is no default.
    //
    // Absence disables the ADVICE, not the FILTER. Estimation of position,
    // drift rate and temperature coefficient needs no optics at all, and it is
    // the half that costs nights of sky to train — so the forecaster runs
    // regardless and reports `AdviceStatus::ThresholdUnknown` until it has a
    // threshold from one of, in order: `refocus_threshold_steps`, f_ratio +
    // steps_per_micron here, `rigs.focal_ratio` + steps_per_micron, or a
    // measured autofocus V-curve.
    /// Focal ratio (f/N) of the imaging train. Falls back to
    /// `rigs.focal_ratio` when unset.
    #[serde(default)]
    pub f_ratio: Option<f64>,
    /// Wavelength for the CFZ calculation, nm. 550 (visual peak) if unset.
    #[serde(default)]
    pub wavelength_nm: Option<f64>,
    /// Focuser steps per micron of travel — a mechanical property of your
    /// focuser and reducer, not something the daemon can infer.
    #[serde(default)]
    pub steps_per_micron: Option<f64>,
    /// Direct override, in focuser steps.
    #[serde(default)]
    pub refocus_threshold_steps: Option<f64>,

    // ---- Decision and noise tuning ----------------------------------------
    /// Minimum P(|drift| > threshold) before a refocus is recommended.
    #[serde(default)]
    pub min_confidence_to_act: Option<f64>,
    /// DWNA acceleration noise, steps/s².
    #[serde(default)]
    pub sigma_accel_steps_per_s2: Option<f64>,
    /// Random-walk noise on the temperature coefficient, steps/°C per √s.
    #[serde(default)]
    pub sigma_temp_coeff: Option<f64>,
    /// Measurement noise on an observed best-focus position, steps.
    #[serde(default)]
    pub sigma_measurement_steps: Option<f64>,

    /// Where to keep the learned filter state across restarts. Relative paths
    /// resolve against the data directory. `None` disables persistence, which
    /// means every daemon restart re-learns from zero.
    #[serde(default)]
    pub state_path: Option<String>,
}

/// Optical constants read from the `rigs` row for the running rig.
///
/// Consulted before the operator is asked to repeat themselves in TOML — these
/// columns have existed since migration 002 and describe exactly the same
/// physical facts `[focus.drift]` asks for.
#[derive(Debug, Clone, Copy, Default)]
pub struct RigOptics {
    pub focal_ratio: Option<f64>,
    pub focal_length_mm: Option<f64>,
    pub pixel_size_um: Option<f64>,
}

impl FocusDriftConfig {
    /// Build a [`crate::focus::focus_drift::DriftConfig`].
    ///
    /// **Missing optics is not an error.** A threshold is needed only to turn a
    /// predicted drift into advice; the filter estimates position, drift rate
    /// and temperature coefficient without one. Returning `Err` here would
    /// disable the estimator as collateral damage, and the estimator is the
    /// half that costs sky — identifying the temperature coefficient needs
    /// varied thermal conditions across several sessions, and starting that
    /// from the diffuse prior because two numbers were missing from a config
    /// file is a bad trade.
    ///
    /// So an absent threshold yields `refocus_threshold_steps: None` and the
    /// forecaster reports `AdviceStatus::ThresholdUnknown` — which the operator
    /// can see, alongside the drift estimate that argues for fixing it.
    ///
    /// `Err` is reserved for values that are *present and wrong*. A negative
    /// sigma or a confidence outside `[0,1]` is a mistake to surface, not an
    /// absence to model.
    ///
    /// Threshold precedence: explicit config → config optics → rig profile.
    pub fn to_drift_config(
        &self,
        rig: Option<&RigOptics>,
    ) -> Result<
        (
            crate::focus::focus_drift::DriftConfig,
            Option<crate::focus::focus_drift::ThresholdSource>,
        ),
        String,
    > {
        use crate::focus::focus_drift::{DriftConfig, ThresholdSource};
        let mut cfg = DriftConfig::default();
        let wavelength = self.wavelength_nm.unwrap_or(550.0);

        let mut source = None;
        cfg.refocus_threshold_steps = match self.refocus_threshold_steps {
            Some(v) if v.is_finite() && v > 0.0 => {
                source = Some(ThresholdSource::Configured);
                Some(v)
            }
            Some(v) => return Err(format!("refocus_threshold_steps must be > 0, got {v}")),
            None => {
                // steps_per_micron is mechanical and nothing else knows it, so
                // it must come from config; f_ratio can come from the rig row.
                let f_ratio = self.f_ratio.or_else(|| rig.and_then(|r| r.focal_ratio));
                let from_rig = self.f_ratio.is_none() && f_ratio.is_some();

                match (f_ratio, self.steps_per_micron) {
                    (Some(n), Some(spm)) if n > 0.0 && spm > 0.0 => {
                        source = Some(if from_rig {
                            ThresholdSource::RigProfile
                        } else {
                            ThresholdSource::ConfiguredOptics
                        });
                        Some(DriftConfig::threshold_from_optics(n, wavelength, spm))
                    }
                    (Some(n), Some(spm)) => {
                        return Err(format!(
                            "f_ratio ({n}) and steps_per_micron ({spm}) must both be > 0"
                        ))
                    }
                    // Absent, not wrong. The filter still runs.
                    _ => None,
                }
            }
        };

        if let Some(v) = self.min_confidence_to_act {
            if !(0.0..=1.0).contains(&v) {
                return Err(format!("min_confidence_to_act must be in [0,1], got {v}"));
            }
            cfg.min_confidence_to_act = v;
        }
        for (value, slot, name) in [
            (
                self.sigma_accel_steps_per_s2,
                &mut cfg.sigma_accel_steps_per_s2,
                "sigma_accel_steps_per_s2",
            ),
            (
                self.sigma_temp_coeff,
                &mut cfg.sigma_temp_coeff,
                "sigma_temp_coeff",
            ),
            (
                self.sigma_measurement_steps,
                &mut cfg.sigma_measurement_steps,
                "sigma_measurement_steps",
            ),
        ] {
            if let Some(v) = value {
                if !v.is_finite() || v <= 0.0 {
                    return Err(format!("{name} must be finite and > 0, got {v}"));
                }
                *slot = v;
            }
        }

        Ok((cfg, source))
    }

    /// Optics for the V-curve scale measurement: `(f_ratio, pixel_size_um)`.
    ///
    /// Both may come from the rig row. `pixel_size_um` has no `[focus.drift]`
    /// key at all — it is a camera fact, `rigs.pixel_size_um` already holds it,
    /// and adding a second place to type it in would be the friction this whole
    /// path exists to remove.
    pub fn vcurve_inputs(&self, rig: Option<&RigOptics>) -> Option<(f64, f64)> {
        let f_ratio = self.f_ratio.or_else(|| rig.and_then(|r| r.focal_ratio))?;
        let pixel_size_um = rig.and_then(|r| r.pixel_size_um)?;
        (f_ratio > 0.0 && pixel_size_um > 0.0).then_some((f_ratio, pixel_size_um))
    }

    /// Wavelength for the CFZ, defaulting to the visual peak.
    pub fn wavelength_nm_or_default(&self) -> f64 {
        self.wavelength_nm.unwrap_or(550.0)
    }
}

#[cfg(test)]
mod focus_drift_config_tests {
    use super::{FocusDriftConfig, RigOptics};
    use crate::focus::focus_drift::ThresholdSource;

    fn optics(f_ratio: f64, steps_per_micron: f64) -> FocusDriftConfig {
        FocusDriftConfig {
            enabled: true,
            f_ratio: Some(f_ratio),
            steps_per_micron: Some(steps_per_micron),
            ..Default::default()
        }
    }

    #[test]
    fn threshold_comes_from_the_critical_focus_zone() {
        // CFZ = 4 * lambda * N^2 = 4 * 0.55 um * 5.6^2 = 68.99 um.
        // Threshold is half of that, in steps: 0.5 * 68.99 * 2.5 = 86.24.
        let (cfg, source) = optics(5.6, 2.5).to_drift_config(None).expect("valid optics");
        let t = cfg.refocus_threshold_steps.expect("threshold derived");
        assert!((t - 86.24).abs() < 0.05, "got {t}");
        assert_eq!(source, Some(ThresholdSource::ConfiguredOptics));
    }

    #[test]
    fn threshold_scales_with_the_square_of_focal_ratio() {
        // Why a fixed step count cannot serve two scopes: f/10 has a CFZ almost
        // four times f/5's.
        let slow = optics(10.0, 2.5).to_drift_config(None).unwrap().0;
        let fast = optics(5.0, 2.5).to_drift_config(None).unwrap().0;
        let ratio =
            slow.refocus_threshold_steps.unwrap() / fast.refocus_threshold_steps.unwrap();
        assert!((ratio - 4.0).abs() < 1e-6, "expected 4x, got {ratio}");
    }

    #[test]
    fn missing_optics_yields_no_threshold_but_still_builds() {
        // The correction: absence disables the ADVICE, not the FILTER.
        // Returning Err here would take the estimator down with it, and the
        // estimator is what costs nights of sky to train.
        let cfg = FocusDriftConfig {
            enabled: true,
            ..Default::default()
        };
        let (built, source) = cfg
            .to_drift_config(None)
            .expect("absent optics must not be an error");
        assert_eq!(built.refocus_threshold_steps, None);
        assert_eq!(source, None);
        // Everything the estimator needs is still present and sane.
        assert!(built.sigma_measurement_steps > 0.0);
        assert_eq!(built.initial_temp_coeff_var, 400.0);

        // Half the optics is still not enough for a threshold, and still fine.
        let half = FocusDriftConfig {
            enabled: true,
            f_ratio: Some(5.6),
            ..Default::default()
        };
        assert_eq!(
            half.to_drift_config(None).unwrap().0.refocus_threshold_steps,
            None
        );
    }

    #[test]
    fn rig_profile_supplies_f_ratio_before_toml_is_demanded() {
        // rigs.focal_ratio has existed since migration 002. Asking the operator
        // to retype it in achernar.toml is the friction that guarantees this
        // stays unset.
        let cfg = FocusDriftConfig {
            enabled: true,
            steps_per_micron: Some(2.5),
            ..Default::default()
        };
        let rig = RigOptics {
            focal_ratio: Some(5.6),
            focal_length_mm: Some(900.0),
            pixel_size_um: Some(3.76),
        };
        let (built, source) = cfg.to_drift_config(Some(&rig)).unwrap();
        assert!((built.refocus_threshold_steps.unwrap() - 86.24).abs() < 0.05);
        assert_eq!(source, Some(ThresholdSource::RigProfile));

        // An explicit config value still outranks the rig row.
        let explicit = FocusDriftConfig {
            f_ratio: Some(10.0),
            ..cfg.clone()
        };
        let (built, source) = explicit.to_drift_config(Some(&rig)).unwrap();
        assert!(built.refocus_threshold_steps.unwrap() > 200.0);
        assert_eq!(source, Some(ThresholdSource::ConfiguredOptics));
    }

    #[test]
    fn explicit_threshold_overrides_every_derivation() {
        let rig = RigOptics {
            focal_ratio: Some(5.6),
            ..Default::default()
        };
        let (cfg, source) = FocusDriftConfig {
            enabled: true,
            refocus_threshold_steps: Some(42.0),
            steps_per_micron: Some(2.5),
            ..Default::default()
        }
        .to_drift_config(Some(&rig))
        .expect("an explicit threshold needs no optics");
        assert_eq!(cfg.refocus_threshold_steps, Some(42.0));
        assert_eq!(source, Some(ThresholdSource::Configured));
    }

    #[test]
    fn present_but_wrong_is_an_error_absent_is_not() {
        // The distinction that matters: a value someone typed incorrectly is a
        // mistake to surface; a value nobody typed is a state to model.
        for bad in [
            FocusDriftConfig {
                enabled: true,
                refocus_threshold_steps: Some(-1.0),
                ..Default::default()
            },
            FocusDriftConfig {
                enabled: true,
                min_confidence_to_act: Some(1.4),
                ..optics(5.6, 2.5)
            },
            FocusDriftConfig {
                enabled: true,
                sigma_measurement_steps: Some(0.0),
                ..optics(5.6, 2.5)
            },
            FocusDriftConfig {
                enabled: true,
                f_ratio: Some(-5.6),
                steps_per_micron: Some(2.5),
                ..Default::default()
            },
        ] {
            assert!(
                bad.to_drift_config(None).is_err(),
                "a wrong value must be surfaced, not silently corrected"
            );
        }
    }

    #[test]
    fn defaults_are_preserved_when_not_overridden() {
        use crate::focus::focus_drift::DriftConfig;
        let (cfg, _) = optics(5.6, 2.5).to_drift_config(None).unwrap();
        let d = DriftConfig::default();
        assert_eq!(cfg.min_confidence_to_act, d.min_confidence_to_act);
        assert_eq!(cfg.sigma_measurement_steps, d.sigma_measurement_steps);
        // The prior that matters: sigma = 20 steps/degC, not ~0.
        assert_eq!(cfg.initial_temp_coeff_var, 400.0);
        // And the one that must NOT have a default any more.
        assert_eq!(d.refocus_threshold_steps, None);
    }

    #[test]
    fn vcurve_inputs_need_a_pixel_size_from_the_rig_row() {
        let cfg = optics(5.6, 2.5);
        assert_eq!(cfg.vcurve_inputs(None), None, "no rig row, no pixel size");

        let rig = RigOptics {
            focal_ratio: Some(4.0),
            focal_length_mm: None,
            pixel_size_um: Some(3.76),
        };
        // Config f_ratio wins; pixel size can only come from the rig.
        assert_eq!(cfg.vcurve_inputs(Some(&rig)), Some((5.6, 3.76)));

        let no_pixels = RigOptics {
            focal_ratio: Some(4.0),
            ..Default::default()
        };
        assert_eq!(cfg.vcurve_inputs(Some(&no_pixels)), None);
    }
}

/// Focus safety guard configuration — Section 7.3.3.
#[derive(Debug, Clone, Deserialize)]
pub struct FocusSafetyConfig {
    #[serde(default)]
    pub min_position: i32,
    #[serde(default = "default_max_position")]
    pub max_position: i32,
    #[serde(default = "default_max_step_single")]
    pub max_step_single: i32,
    #[serde(default = "default_max_total_travel")]
    pub max_total_travel_per_run: i32,
    #[serde(default = "default_min_confidence")]
    pub min_confidence_to_act: f64,
    #[serde(default = "default_max_failed")]
    pub max_failed_improvements: u32,
    #[serde(default = "default_settle_ms")]
    pub settle_ms: u64,
    #[serde(default)]
    pub backlash_steps: i32,
    /// Longest wait for a commanded focuser move to report stopped before the
    /// move is halted and treated as failed.
    #[serde(default = "default_move_timeout_ms")]
    pub move_timeout_ms: u64,
    /// A stopped focuser within this many steps of the target counts as
    /// arrived. Zero requires an exact match.
    #[serde(default)]
    pub arrival_tolerance_steps: i32,
}

impl Default for FocusSafetyConfig {
    fn default() -> Self {
        Self {
            min_position: 0,
            max_position: default_max_position(),
            max_step_single: default_max_step_single(),
            max_total_travel_per_run: default_max_total_travel(),
            min_confidence_to_act: default_min_confidence(),
            max_failed_improvements: default_max_failed(),
            settle_ms: default_settle_ms(),
            backlash_steps: 0,
            move_timeout_ms: default_move_timeout_ms(),
            arrival_tolerance_steps: 0,
        }
    }
}

/// Backlash compensation configuration — Section 7.3.5.
#[derive(Debug, Clone, Deserialize)]
pub struct BacklashConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub compensation_steps: i32,
    #[serde(default)]
    pub always_approach_from: Option<String>,
}

impl Default for BacklashConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            compensation_steps: 0,
            always_approach_from: None,
        }
    }
}

/// Paths to edge ONNX models.
///
/// There is deliberately no `drift_model_path`. The focus-drift forecaster is a
/// 3-state Kalman filter, not a network — there is no ONNX file to load and the
/// key was conceptually wrong rather than merely unset. Its tuning is
/// [`FocusDriftConfig`] and its learned state is a JSON snapshot in the data
/// directory.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct FocusModelsConfig {
    #[serde(default)]
    pub quality_model_path: Option<String>,
    #[serde(default)]
    pub offset_model_path: Option<String>,
}

/// Mount configuration — Section 8.3.
#[derive(Debug, Clone, Deserialize)]
pub struct MountConfig {
    #[serde(default)]
    pub safety: MountSafetyConfig,
    #[serde(default)]
    pub guide_rate: f64,
    #[serde(default)]
    pub max_slew_rate: f64,
}

impl Default for MountConfig {
    fn default() -> Self {
        Self {
            safety: MountSafetyConfig::default(),
            guide_rate: 0.5,
            max_slew_rate: 8.0,
        }
    }
}

/// Mount safety configuration — Section 8.3.
#[derive(Debug, Clone, Deserialize)]
pub struct MountSafetyConfig {
    #[serde(default)]
    pub min_altitude: f64,
    #[serde(default = "default_max_ha")]
    pub max_hour_angle: f64,
    #[serde(default)]
    pub horizon_model_path: Option<String>,
    #[serde(default)]
    pub park_on_safety_violation: bool,
}

impl Default for MountSafetyConfig {
    fn default() -> Self {
        Self {
            min_altitude: 0.0,
            max_hour_angle: default_max_ha(),
            horizon_model_path: None,
            park_on_safety_violation: true,
        }
    }
}

/// Filter wheel configuration — Section 8.2.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct FilterWheelConfig {
    #[serde(default)]
    pub filter_names: Vec<String>,
    #[serde(default)]
    pub focus_offsets: Vec<i32>,
}

/// Camera configuration.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct CameraConfig {
    #[serde(default)]
    pub default_gain: Option<u32>,
    #[serde(default)]
    pub default_offset: Option<u32>,
    #[serde(default)]
    pub cooling_target: Option<f64>,

    /// Linear saturation point of the sensor, in the ADU of the stream this
    /// camera produces. A 14-bit body is 16383, a 12-bit body 4095.
    ///
    /// Declared by the operator because nothing in the pipeline can derive it.
    /// The container depth is not the sensor depth: a 14-bit Canon delivers
    /// 16-bit samples that top out at 16383, so a saturation gate computed from
    /// the container sits at 64224 and can never fire. That was the state of
    /// this code until now, and it means every star with a clipped, flat core
    /// passed the gate and biased HFR low — precisely the effect the gate was
    /// written to prevent.
    ///
    /// Guessing it from the brightest pixel in a frame is not an option: a
    /// frame with no bright stars would produce a low estimate and a threshold
    /// that rejects perfectly good stars. Better to know it is unknown, which
    /// is what `None` means here — the gate is then reported as not performed
    /// rather than silently passed.
    ///
    /// Ignored for 8-bit streams (MJPEG, YUV, RGB24): those pixels have been
    /// through a non-linear transform and 255 is their real clip point.
    #[serde(default)]
    pub saturation_adu: Option<f32>,
}

/// Storage configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct StorageConfig {
    #[serde(default = "default_db_path")]
    pub database_path: String,
    #[serde(default)]
    pub frames_dir: Option<String>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            database_path: default_db_path(),
            frames_dir: None,
        }
    }
}

impl StorageConfig {
    /// Directory for preserved frame files. Defaults to a `frames` directory
    /// beside the SQLite database when not explicitly configured.
    pub fn frames_dir(&self) -> String {
        if let Some(dir) = &self.frames_dir {
            return dir.clone();
        }
        match std::path::Path::new(&self.database_path).parent() {
            Some(parent) if !parent.as_os_str().is_empty() => {
                parent.join("frames").to_string_lossy().into_owned()
            }
            _ => "frames".to_string(),
        }
    }

    /// Directory for mutable, rig-specific runtime state — the directory the
    /// database lives in. Distinct from the models directory, which holds
    /// version-controlled artefacts that ship with the code.
    pub fn data_dir(&self) -> std::path::PathBuf {
        match std::path::Path::new(&self.database_path).parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
            _ => std::path::PathBuf::from("."),
        }
    }
}

/// Telemetry/logging configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct TelemetryConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
    #[serde(default)]
    pub json_output: bool,
    #[serde(default)]
    pub log_dir: Option<String>,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            json_output: false,
            log_dir: None,
        }
    }
}

impl AchernarConfig {
    /// Load config from achernar.toml, falling back to defaults.
    pub fn load() -> crate::error::Result<Self> {
        let config_path =
            std::env::var("ACHERNAR_CONFIG").unwrap_or_else(|_| "achernar.toml".to_string());

        if std::path::Path::new(&config_path).exists() {
            let content = std::fs::read_to_string(&config_path).map_err(|e| {
                crate::error::AchernarError::Other(format!("Failed to read config: {}", e))
            })?;
            toml::from_str(&content).map_err(|e| {
                crate::error::AchernarError::Other(format!("Failed to parse config: {}", e))
            })
        } else {
            tracing::info!("No config file found at {}, using defaults", config_path);
            Ok(Self::default())
        }
    }
}

impl Default for AchernarConfig {
    fn default() -> Self {
        Self {
            daemon: DaemonConfig::default(),
            focus: FocusConfig::default(),
            mount: MountConfig::default(),
            filter_wheel: FilterWheelConfig::default(),
            camera: CameraConfig::default(),
            storage: StorageConfig::default(),
            telemetry: TelemetryConfig::default(),
        }
    }
}

// Default value functions
fn default_port() -> u16 {
    3030
}
fn default_log_level() -> String {
    "info".to_string()
}
fn default_step_size() -> i32 {
    50
}
fn default_samples_per_position() -> u32 {
    3
}
fn default_sweep_radius_points() -> u32 {
    3
}
fn default_settle_ms() -> u64 {
    500
}
fn default_move_timeout_ms() -> u64 {
    60_000
}
fn default_max_position() -> i32 {
    50000
}
fn default_max_step_single() -> i32 {
    500
}
fn default_max_total_travel() -> i32 {
    5000
}
fn default_min_confidence() -> f64 {
    0.6
}
fn default_max_failed() -> u32 {
    3
}
fn default_max_ha() -> f64 {
    5.5
}
fn default_db_path() -> String {
    "achernar.db".to_string()
}

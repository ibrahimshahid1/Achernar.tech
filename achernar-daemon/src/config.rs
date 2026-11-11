//! Configuration parsing for achernar.toml.
//! Implements all config sections including [focus], [focus.safety],
//! [focus.backlash], [focus.models], [mount], [filter_wheel], and the
//! equipment-layer sections [equipment], [site], [focuser], [guider],
//! [safety] and [weather] (equipment spec §2.11).

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
    pub plate_solve: PlateSolveConfig,
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    #[serde(default)]
    pub equipment: EquipmentConfig,
    #[serde(default)]
    pub site: SiteConfig,
    #[serde(default)]
    pub focuser: FocuserConfig,
    #[serde(default)]
    pub guider: GuiderConfig,
    #[serde(default)]
    pub safety: SafetyConfig,
    #[serde(default)]
    pub weather: WeatherConfig,
    /// `[vision]`: image-analysis settings of the vision loop.
    #[serde(default)]
    pub vision: VisionConfig,
}

/// `[vision]` in `achernar.toml`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct VisionConfig {
    /// `[vision.background]`: `backend = "legacy_global_clip" | "sep"`,
    /// `fallback = "unavailable" | "legacy_global_clip"`, `[vision.background.sep]`
    /// mesh parameters and `[vision.background.detection]` extraction
    /// parameters. `sep` is an opt-in diagnostic backend, not a qualified
    /// default.
    #[serde(default)]
    pub background: crate::vision::background_model::BackgroundSettings,
    /// `[vision.star_gates]`: `blend_test = "dip_annulus_v2" |
    /// "peak_count_v1"` and `regroup_deblended` (docs/bge-v2-spec.md §1-2).
    /// Applies to every background backend.
    #[serde(default)]
    pub star_gates: crate::vision::quality::StarGateSettings,
    /// `[vision.fqe]`: the opt-in FQE prototype switch.
    #[serde(default)]
    pub fqe: FqeConfig,
}

/// `[vision.fqe]` in `achernar.toml`.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub struct FqeConfig {
    /// PROTOTYPE, not qualified for R50/Z114, off by default (owner
    /// authorization F-006, FQE run 20261111T0038). When true, a
    /// `models/focus_quality/<semver>` bundle whose metadata carries the
    /// prototype provenance block `rebuild_fqe.py --prototype-runtime-eligible`
    /// writes (`fqe_runner::verify_prototype_graph`, which also binds the
    /// loaded graph's sha256 to that metadata) runs in the vision loop
    /// despite the A03 gate; its predictions are labelled `fqe_prototype` and
    /// classical HFR stays the authority. A bundle without that block is never
    /// enabled by this flag. False: behaviour is unchanged.
    #[serde(default)]
    pub prototype: bool,
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

    /// The HFR the autofocus curve fit uses: `"classical"` (default) or
    /// `"fqe_prototype"` (PROTOTYPE, not qualified for R50/Z114; requires
    /// `[vision.fqe] prototype = true`, which [`AchernarConfig::validate`]
    /// enforces). See `focus_agent::HfrSource`.
    #[serde(default)]
    pub hfr_source: crate::focus::focus_agent::HfrSource,
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
            hfr_source: crate::focus::focus_agent::HfrSource::default(),
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
    // Equipment layer (spec §2.11). `guide_rate` above is now a fallback,
    // used only when the mount reports no guide rate (source `Profile`).
    /// Settle after a slew, in addition to the driver's own SlewSettleTime.
    #[serde(default = "default_mount_settle_s")]
    pub settle_s: f64,
    #[serde(default = "default_slew_timeout_s")]
    pub slew_timeout_s: u64,
    #[serde(default = "default_arrival_tolerance_arcmin")]
    pub arrival_tolerance_arcmin: f64,
    /// `mount` syncs the mount; `offset` keeps a solved pointing offset
    /// (NINA "NoSync"), recommended while M-4 data is collected.
    #[serde(default)]
    pub sync_mode: crate::equipment::types::MountSyncMode,
    /// Write UTC at connect, only when |delta| >= `time_tolerance_s`. Off by
    /// default: connect writes nothing to a mount (D16).
    #[serde(default)]
    pub time_sync: bool,
    #[serde(default = "default_time_tolerance_s")]
    pub time_tolerance_s: f64,
    #[serde(default)]
    pub equatorial_system_override: EquatorialSystemOverride,
    /// How Topocentric/EOD is interpreted; recorded with every sample.
    #[serde(default)]
    pub jnow_flavour: crate::equipment::types::JNowFlavour,
    #[serde(default = "default_max_pulse_ms")]
    pub max_pulse_ms: u32,
    #[serde(default)]
    pub guide_dec_reverse_on_pier_west: bool,
    #[serde(default = "default_jog_deadman_ms")]
    pub jog_deadman_ms: u64,
    /// Tier 2 fallback park for mounts without CanPark.
    #[serde(default)]
    pub fallback_park: FallbackPark,
    #[serde(default)]
    pub center: MountCenterConfig,
    #[serde(default)]
    pub flip: MountFlipConfig,
}

impl Default for MountConfig {
    fn default() -> Self {
        Self {
            safety: MountSafetyConfig::default(),
            guide_rate: 0.5,
            max_slew_rate: 8.0,
            settle_s: default_mount_settle_s(),
            slew_timeout_s: default_slew_timeout_s(),
            arrival_tolerance_arcmin: default_arrival_tolerance_arcmin(),
            sync_mode: crate::equipment::types::MountSyncMode::default(),
            time_sync: false,
            time_tolerance_s: default_time_tolerance_s(),
            equatorial_system_override: EquatorialSystemOverride::default(),
            jnow_flavour: crate::equipment::types::JNowFlavour::default(),
            max_pulse_ms: default_max_pulse_ms(),
            guide_dec_reverse_on_pier_west: false,
            jog_deadman_ms: default_jog_deadman_ms(),
            fallback_park: FallbackPark::default(),
            center: MountCenterConfig::default(),
            flip: MountFlipConfig::default(),
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
    /// CSV of `az_deg,alt_deg`; empty = no horizon profile. Replaces
    /// `horizon_model_path`, which is ignored with a warning.
    #[serde(default)]
    pub horizon_profile_path: String,
    #[serde(default = "default_limit_hysteresis_deg")]
    pub limit_hysteresis_deg: f64,
}

impl Default for MountSafetyConfig {
    fn default() -> Self {
        Self {
            min_altitude: 0.0,
            max_hour_angle: default_max_ha(),
            horizon_model_path: None,
            park_on_safety_violation: true,
            horizon_profile_path: String::new(),
            limit_hysteresis_deg: default_limit_hysteresis_deg(),
        }
    }
}

#[allow(dead_code)] // read by equipment/mount/limits.rs (WP1A-4)
impl MountSafetyConfig {
    pub fn horizon_profile(&self) -> Option<&str> {
        Some(self.horizon_profile_path.trim()).filter(|path| !path.is_empty())
    }
}

/// Filter wheel configuration — Section 8.2.
///
/// `filter_names` and `focus_offsets` are SEEDS only (equipment spec D17):
/// they are never applied to a wheel automatically. The operator imports
/// them into the connected wheel's identity profile explicitly.
#[derive(Debug, Clone, Deserialize)]
pub struct FilterWheelConfig {
    #[serde(default)]
    pub filter_names: Vec<String>,
    #[serde(default)]
    pub focus_offsets: Vec<i32>,
    /// Settle after a verified slot change; the profile value wins.
    #[serde(default = "default_filter_settle_ms")]
    pub settle_ms: u64,
    #[serde(default = "default_max_switch_timeout_ms")]
    pub max_switch_timeout_ms: u64,
    #[serde(default)]
    pub min_switch_interval_s: f64,
    /// Applies profile offsets only; never the TOML seeds.
    #[serde(default = "default_true")]
    pub use_focus_offsets: bool,
    #[serde(default = "default_true")]
    pub pause_guiding_on_offset: bool,
    #[serde(default)]
    pub parallel_offset_move: bool,
    /// Filter to switch to for autofocus; empty = none.
    #[serde(default)]
    pub af_filter: String,
    /// Rolling cap on switches requested by automatic sources within one
    /// hour (O11 g). Operators are bound only by `min_switch_interval_s`.
    #[serde(default = "default_max_switches_per_hour")]
    pub max_switches_per_hour: u32,
}

impl Default for FilterWheelConfig {
    fn default() -> Self {
        Self {
            filter_names: Vec::new(),
            focus_offsets: Vec::new(),
            settle_ms: default_filter_settle_ms(),
            max_switch_timeout_ms: default_max_switch_timeout_ms(),
            min_switch_interval_s: 0.0,
            use_focus_offsets: true,
            pause_guiding_on_offset: true,
            parallel_offset_move: false,
            af_filter: String::new(),
            max_switches_per_hour: default_max_switches_per_hour(),
        }
    }
}

fn default_max_switches_per_hour() -> u32 {
    60
}

// ─── Equipment layer (equipment spec §2.11) ─────────────────────────────────
//
// Unknown keys are ignored and every default is safe: nothing auto-connects,
// nothing is broadcast, connect writes nothing to a mount, and policy
// actuation stays off. Rig-specific values (backlash, offsets, settle times,
// filter metadata, focuser origin) live in the identity profile, never here
// (D17). These sections are read by the equipment layer, which later work
// packages wire in; until then nothing outside tests reads them.

/// `[equipment]`: behaviour shared by every device.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EquipmentConfig {
    /// A device with no good poll for this long is `Lost`.
    #[serde(default = "default_lost_after_s")]
    pub lost_after_s: u64,
    /// Consecutive failed polls before a device is `Lost`.
    #[serde(default = "default_lost_after_failures")]
    pub lost_after_failures: u32,
    /// Platform 7 `connecting` wait; generous for slow ESP32 Wi-Fi.
    #[serde(default = "default_connect_timeout_s")]
    pub connect_timeout_s: u64,
    /// Gates `"auto"`/empty Alpaca options and UDP discovery (D13).
    #[serde(default)]
    pub allow_broadcast_discovery: bool,
    #[serde(default = "default_alpaca_discovery_port")]
    pub alpaca_discovery_port: u16,
    /// 0 = generate once (random 1..65535) and persist in the data directory.
    #[serde(default)]
    pub alpaca_client_id: u32,
    /// INDI `CONNECTION` is device-wide: switching it off could cut off PHD2
    /// or KStars, so Achernar leaves it on unless this is set (D16).
    #[serde(default)]
    pub indi_disconnect_device: bool,
    #[serde(default = "default_true")]
    pub estop_stops_tracking: bool,
    /// Downsampled device series for TME/replay; 0 disables.
    #[serde(default = "default_telemetry_log_hz")]
    pub telemetry_log_hz: f64,
    #[serde(default = "default_ring_buffer_minutes")]
    pub ring_buffer_minutes: u64,
}

impl Default for EquipmentConfig {
    fn default() -> Self {
        Self {
            lost_after_s: default_lost_after_s(),
            lost_after_failures: default_lost_after_failures(),
            connect_timeout_s: default_connect_timeout_s(),
            allow_broadcast_discovery: false,
            alpaca_discovery_port: default_alpaca_discovery_port(),
            alpaca_client_id: 0,
            indi_disconnect_device: false,
            estop_stops_tracking: true,
            telemetry_log_hz: default_telemetry_log_hz(),
            ring_buffer_minutes: default_ring_buffer_minutes(),
        }
    }
}

/// `[site] sync`: what connect does when the mount's site differs (M-b).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SiteSyncMode {
    /// Emit `site_mismatch` and change nothing.
    #[default]
    VerifyOnly,
    /// Write the mount (only while it is not slewing), then verify.
    ToMount,
    /// Adopt the mount's site for the session.
    FromMount,
    None,
}

/// `[site]`. Omitted coordinates mean "site unknown": geometry features then
/// report so, rather than assume a location.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SiteConfig {
    #[serde(default)]
    pub latitude_deg: Option<f64>,
    /// East positive.
    #[serde(default)]
    pub longitude_deg: Option<f64>,
    #[serde(default)]
    pub elevation_m: Option<f64>,
    #[serde(default)]
    pub sync: SiteSyncMode,
    #[serde(default = "default_site_tolerance_deg")]
    pub tolerance_deg: f64,
    #[serde(default = "default_elevation_tolerance_m")]
    pub elevation_tolerance_m: f64,
    /// UT1 - UTC in seconds for sidereal time (O11 g); |DUT1| <= 0.9 s.
    #[serde(default)]
    pub dut1_s: f64,
}

impl Default for SiteConfig {
    fn default() -> Self {
        Self {
            latitude_deg: None,
            longitude_deg: None,
            elevation_m: None,
            sync: SiteSyncMode::default(),
            tolerance_deg: default_site_tolerance_deg(),
            elevation_tolerance_m: default_elevation_tolerance_m(),
            dut1_s: 0.0,
        }
    }
}

#[allow(dead_code)]
impl SiteConfig {
    /// The configured site. `Ok(None)` when no coordinate is set; an error
    /// when only some are set or a value is out of range, so a half-typed
    /// site is surfaced rather than completed with invented numbers.
    pub fn site(&self) -> Result<Option<crate::equipment::types::Site>, String> {
        match (self.latitude_deg, self.longitude_deg, self.elevation_m) {
            (None, None, None) => Ok(None),
            (Some(lat_deg), Some(lon_deg), Some(elev_m)) => {
                let site = crate::equipment::types::Site {
                    lat_deg,
                    lon_deg,
                    elev_m,
                    source: crate::equipment::types::SiteSource::Config,
                };
                site.validate().map(|()| Some(site))
            }
            _ => Err("[site] needs latitude_deg, longitude_deg and elevation_m together".into()),
        }
    }
}

/// `[mount] equatorial_system_override`: overrides the frame the mount
/// reports (M-a). Empty means no override.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
pub enum EquatorialSystemOverride {
    #[default]
    #[serde(rename = "")]
    None,
    #[serde(rename = "j2000")]
    J2000,
    #[serde(rename = "jnow_mean")]
    JNowMean,
    #[serde(rename = "jnow_apparent")]
    JNowApparent,
}

#[allow(dead_code)]
impl EquatorialSystemOverride {
    pub fn frame(self) -> Option<crate::equipment::types::EqFrame> {
        use crate::equipment::types::EqFrame;
        match self {
            EquatorialSystemOverride::None => None,
            EquatorialSystemOverride::J2000 => Some(EqFrame::J2000),
            EquatorialSystemOverride::JNowMean => Some(EqFrame::JNowMean),
            EquatorialSystemOverride::JNowApparent => Some(EqFrame::JNowApparent),
        }
    }
}

/// `[mount] fallback_park` (Tier 2).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackPark {
    #[default]
    None,
    Pole,
}

/// `[mount.center]`: the slew-solve-correct loop (M-n, Tier 1).
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct MountCenterConfig {
    #[serde(default = "default_center_tolerance_arcmin")]
    pub tolerance_arcmin: f64,
    #[serde(default = "default_center_max_attempts")]
    pub max_attempts: u32,
    /// The plate-solve exposure of each center frame (M-n step 2), seconds.
    /// A request may override it; it never comes from the newest solved
    /// frame.
    #[serde(default = "default_center_exposure_s")]
    pub exposure_s: f64,
    /// Gain and binning of each center frame; unset keeps the camera's own.
    #[serde(default)]
    pub gain: Option<u32>,
    #[serde(default)]
    pub binning: Option<u32>,
}

impl Default for MountCenterConfig {
    fn default() -> Self {
        Self {
            tolerance_arcmin: default_center_tolerance_arcmin(),
            max_attempts: default_center_max_attempts(),
            exposure_s: default_center_exposure_s(),
            gain: None,
            binning: None,
        }
    }
}

/// `[mount.flip]` (Tier 2). Inert unless the mount `can_flip` (D14).
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct MountFlipConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_flip_minutes_after")]
    pub minutes_after_meridian: f64,
    #[serde(default = "default_flip_max_minutes_after")]
    pub max_minutes_after_meridian: f64,
    #[serde(default = "default_flip_settle_s")]
    pub settle_s: f64,
    #[serde(default = "default_true")]
    pub recenter: bool,
    #[serde(default)]
    pub autofocus_after: bool,
}

impl Default for MountFlipConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            minutes_after_meridian: default_flip_minutes_after(),
            max_minutes_after_meridian: default_flip_max_minutes_after(),
            settle_s: default_flip_settle_s(),
            recenter: true,
            autofocus_after: false,
        }
    }
}

/// `[focuser] tempcomp_during_moves` (F-d).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TempCompDuringMoves {
    /// Turn TempComp off for the move and restore it afterwards.
    #[default]
    Off,
    Leave,
}

/// `[focuser]`: device behaviour only. Backlash stays in `[focus.backlash]`
/// plus the identity profile.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct FocuserConfig {
    #[serde(default = "default_stuck_timeout_s")]
    pub stuck_timeout_s: u64,
    #[serde(default = "default_max_reissue")]
    pub max_reissue: u32,
    #[serde(default)]
    pub tempcomp_during_moves: TempCompDuringMoves,
    /// A per-identity profile value wins.
    #[serde(default)]
    pub temperature_role: crate::equipment::types::TemperatureRole,
    /// Optional Alpaca Action name reporting homed|restored|unknown (F-i).
    #[serde(default)]
    pub position_source_action: String,
}

impl Default for FocuserConfig {
    fn default() -> Self {
        Self {
            stuck_timeout_s: default_stuck_timeout_s(),
            max_reissue: default_max_reissue(),
            tempcomp_during_moves: TempCompDuringMoves::default(),
            temperature_role: crate::equipment::types::TemperatureRole::default(),
            position_source_action: String::new(),
        }
    }
}

/// `[guider] guiding_during_autofocus` (G-m).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuidingDuringAutofocus {
    #[default]
    Keep,
    Pause,
    Stop,
}

/// `[guider]`: the PHD2 client (§3.5).
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GuiderConfig {
    /// Never call PHD2 `set_connected` unless asked.
    #[serde(default)]
    pub connect_equipment: bool,
    #[serde(default = "default_rpc_timeout_s")]
    pub rpc_timeout_s: u64,
    #[serde(default = "default_settle_pixels")]
    pub settle_pixels: f64,
    #[serde(default = "default_settle_time_s")]
    pub settle_time_s: f64,
    #[serde(default = "default_settle_timeout_s")]
    pub settle_timeout_s: f64,
    /// Guide-camera pixels.
    #[serde(default = "default_dither_amount_px")]
    pub dither_amount_px: f64,
    #[serde(default)]
    pub dither_ra_only: bool,
    /// Empty = discover via `get_algo_param_names` (G-j).
    #[serde(default)]
    pub aggressiveness_param_ra: String,
    #[serde(default)]
    pub aggressiveness_param_dec: String,
    #[serde(default = "default_algo_param_min")]
    pub algo_param_min: f64,
    #[serde(default = "default_algo_param_max")]
    pub algo_param_max: f64,
    /// M-1 `set_algo_param` by a policy (Tier 2). Off by default (D18).
    #[serde(default)]
    pub allow_policy_actuation: bool,
    #[serde(default)]
    pub guiding_during_autofocus: GuidingDuringAutofocus,
}

impl Default for GuiderConfig {
    fn default() -> Self {
        Self {
            connect_equipment: false,
            rpc_timeout_s: default_rpc_timeout_s(),
            settle_pixels: default_settle_pixels(),
            settle_time_s: default_settle_time_s(),
            settle_timeout_s: default_settle_timeout_s(),
            dither_amount_px: default_dither_amount_px(),
            dither_ra_only: false,
            aggressiveness_param_ra: String::new(),
            aggressiveness_param_dec: String::new(),
            algo_param_min: default_algo_param_min(),
            algo_param_max: default_algo_param_max(),
            allow_policy_actuation: false,
            guiding_during_autofocus: GuidingDuringAutofocus::default(),
        }
    }
}

#[allow(dead_code)]
impl GuiderConfig {
    /// The configured PHD2 settle parameters.
    pub fn settle(&self) -> crate::equipment::commands::Settle {
        crate::equipment::commands::Settle {
            pixels: self.settle_pixels,
            time_s: self.settle_time_s,
            timeout_s: self.settle_timeout_s,
        }
    }
}

/// `[safety] on_unsafe` (§3.10).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnUnsafe {
    /// Park if `[mount.safety] park_on_safety_violation`, else stop tracking.
    #[default]
    FromMountConfig,
    Park,
    StopTracking,
    Notify,
}

/// `[safety] resume`.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SafetyResume {
    /// Tier 1.
    #[default]
    Manual,
    /// Tier 2; off by default (D18).
    Auto,
}

/// `[safety]`: the on-unsafe policy (§3.10).
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SafetyConfig {
    #[serde(default)]
    pub on_unsafe: OnUnsafe,
    #[serde(default = "default_true")]
    pub abort_exposure_on_unsafe: bool,
    /// INDI Busy and ASCOM-less "warning" states count as unsafe.
    #[serde(default = "default_true")]
    pub warning_is_unsafe: bool,
    #[serde(default)]
    pub resume: SafetyResume,
    #[serde(default = "default_resume_after_safe_s")]
    pub resume_after_safe_s: u64,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            on_unsafe: OnUnsafe::default(),
            abort_exposure_on_unsafe: true,
            warning_is_unsafe: true,
            resume: SafetyResume::default(),
            resume_after_safe_s: default_resume_after_safe_s(),
        }
    }
}

#[allow(dead_code)]
impl SafetyConfig {
    /// `on_unsafe` with `from_mount_config` resolved against
    /// `[mount.safety] park_on_safety_violation`.
    pub fn effective_on_unsafe(&self, mount_safety: &MountSafetyConfig) -> OnUnsafe {
        match self.on_unsafe {
            OnUnsafe::FromMountConfig if mount_safety.park_on_safety_violation => OnUnsafe::Park,
            OnUnsafe::FromMountConfig => OnUnsafe::StopTracking,
            other => other,
        }
    }
}

/// `[weather]`.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct WeatherConfig {
    /// One `environment_log` probe row per interval.
    #[serde(default = "default_weather_log_interval_s")]
    pub log_interval_s: u64,
}

impl Default for WeatherConfig {
    fn default() -> Self {
        Self {
            log_interval_s: default_weather_log_interval_s(),
        }
    }
}

/// Everything the equipment layer reads, handed to
/// `AppState::with_equipment` (spec §2.10).
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct EquipmentConfigs {
    pub equipment: EquipmentConfig,
    pub site: SiteConfig,
    pub mount: MountConfig,
    pub focuser: FocuserConfig,
    pub filter_wheel: FilterWheelConfig,
    pub guider: GuiderConfig,
    pub safety: SafetyConfig,
    pub weather: WeatherConfig,
    pub focus_safety: FocusSafetyConfig,
    pub focus_backlash: BacklashConfig,
}

#[allow(dead_code)]
impl AchernarConfig {
    pub fn equipment_configs(&self) -> EquipmentConfigs {
        EquipmentConfigs {
            equipment: self.equipment.clone(),
            site: self.site.clone(),
            mount: self.mount.clone(),
            focuser: self.focuser.clone(),
            filter_wheel: self.filter_wheel.clone(),
            guider: self.guider.clone(),
            safety: self.safety.clone(),
            weather: self.weather.clone(),
            focus_safety: self.focus.safety.clone(),
            focus_backlash: self.focus.backlash.clone(),
        }
    }

    /// Problems worth a startup warning. None of them stops the daemon.
    pub fn equipment_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if self.mount.safety.horizon_model_path.is_some() {
            warnings.push(
                "[mount.safety] horizon_model_path is ignored; set horizon_profile_path to a \
                 CSV of az_deg,alt_deg instead"
                    .into(),
            );
        }
        if let Err(err) = self.site.site() {
            warnings.push(format!("{err}; the site is treated as unknown"));
        }
        if self.equipment.alpaca_client_id > 65_535 {
            warnings.push(format!(
                "[equipment] alpaca_client_id {} is outside 1..65535",
                self.equipment.alpaca_client_id
            ));
        }
        let center_exposure = self.mount.center.exposure_s;
        if !center_exposure.is_finite() || center_exposure <= 0.0 {
            warnings.push(format!(
                "[mount.center] exposure_s {center_exposure} is not a positive number of \
                 seconds; center frames use {} s",
                default_center_exposure_s()
            ));
        }
        if self
            .guider
            .algo_param_min
            .partial_cmp(&self.guider.algo_param_max)
            != Some(std::cmp::Ordering::Less)
        {
            warnings.push("[guider] algo_param_min must be below algo_param_max".into());
        }
        if !self.filter_wheel.focus_offsets.is_empty() || !self.filter_wheel.filter_names.is_empty()
        {
            warnings.push(
                "[filter_wheel] filter_names and focus_offsets are seeds only; they are never \
                 applied to a wheel until imported with POST filter-wheel/import-seed"
                    .into(),
            );
        }
        warnings
    }
}

fn default_true() -> bool {
    true
}
fn default_lost_after_s() -> u64 {
    10
}
fn default_lost_after_failures() -> u32 {
    3
}
fn default_connect_timeout_s() -> u64 {
    30
}
fn default_alpaca_discovery_port() -> u16 {
    32227
}
fn default_telemetry_log_hz() -> f64 {
    1.0
}
fn default_ring_buffer_minutes() -> u64 {
    30
}
fn default_site_tolerance_deg() -> f64 {
    0.001
}
fn default_elevation_tolerance_m() -> f64 {
    10.0
}
fn default_mount_settle_s() -> f64 {
    5.0
}
fn default_slew_timeout_s() -> u64 {
    600
}
fn default_arrival_tolerance_arcmin() -> f64 {
    1.0
}
fn default_time_tolerance_s() -> f64 {
    10.0
}
fn default_max_pulse_ms() -> u32 {
    5000
}
fn default_jog_deadman_ms() -> u64 {
    1000
}
fn default_limit_hysteresis_deg() -> f64 {
    0.5
}
fn default_center_tolerance_arcmin() -> f64 {
    1.0
}
fn default_center_max_attempts() -> u32 {
    3
}
fn default_center_exposure_s() -> f64 {
    5.0
}
fn default_flip_minutes_after() -> f64 {
    5.0
}
fn default_flip_max_minutes_after() -> f64 {
    10.0
}
fn default_flip_settle_s() -> f64 {
    30.0
}
fn default_stuck_timeout_s() -> u64 {
    30
}
fn default_max_reissue() -> u32 {
    1
}
fn default_filter_settle_ms() -> u64 {
    500
}
fn default_max_switch_timeout_ms() -> u64 {
    10_000
}
fn default_rpc_timeout_s() -> u64 {
    10
}
fn default_settle_pixels() -> f64 {
    1.5
}
fn default_settle_time_s() -> f64 {
    10.0
}
fn default_settle_timeout_s() -> f64 {
    40.0
}
fn default_dither_amount_px() -> f64 {
    5.0
}
fn default_algo_param_min() -> f64 {
    0.1
}
fn default_algo_param_max() -> f64 {
    1.0
}
fn default_resume_after_safe_s() -> u64 {
    300
}
fn default_weather_log_interval_s() -> u64 {
    60
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

/// Plate solving — Section 8.3. See `processing::plate_solver`.
///
/// Nothing here is required. With no `astap_path` the daemon probes the
/// standard ASTAP install locations and reports what it found (or did not) on
/// `GET /platesolve/status`; solving stays unavailable, never faked, until an
/// executable is found.
#[derive(Debug, Clone, Deserialize)]
pub struct PlateSolveConfig {
    /// ASTAP executable — `astap_cli` (preferred: no GUI libraries) or the
    /// GUI `astap` binary, which accepts the same arguments.
    #[serde(default)]
    pub astap_path: Option<String>,
    /// Star database directory (ASTAP `-d`). Unset: ASTAP's own default.
    #[serde(default)]
    pub database_dir: Option<String>,
    /// Star database to use, e.g. "d50" (ASTAP `-D`). Unset: ASTAP picks.
    #[serde(default)]
    pub database: Option<String>,
    /// Search radius around a pointing hint, degrees. 30 matches the ASTAP and
    /// NINA defaults: generous enough for a mount that has lost alignment.
    #[serde(default = "default_search_radius_deg")]
    pub search_radius_deg: f64,
    /// Downsample factor before solving (ASTAP `-z`). 0 lets ASTAP choose.
    #[serde(default)]
    pub downsample: u32,
    /// Maximum stars used for the solution (ASTAP `-s`).
    #[serde(default = "default_max_stars")]
    pub max_stars: u32,
    /// Longest a single solve may run before the process is killed. A hinted
    /// solve with a known scale takes seconds; a blind solve with an unknown
    /// scale can take minutes.
    #[serde(default = "default_solve_timeout_s")]
    pub timeout_s: u64,
    /// Solve without being asked: the first light frame after startup, and
    /// the first begun after each slew arrives (v6 §8.3.3 deterministic
    /// rules). Does nothing while no solver is available.
    #[serde(default = "default_auto_solve")]
    pub auto_solve: bool,
    /// After an automatic solve fails, the wait before trying another frame,
    /// so a cloudy hour does not become a queue of minutes-long blind solves.
    #[serde(default = "default_auto_solve_retry_s")]
    pub auto_solve_retry_s: u64,
}

impl Default for PlateSolveConfig {
    fn default() -> Self {
        Self {
            astap_path: None,
            database_dir: None,
            database: None,
            search_radius_deg: default_search_radius_deg(),
            downsample: 0,
            max_stars: default_max_stars(),
            timeout_s: default_solve_timeout_s(),
            auto_solve: default_auto_solve(),
            auto_solve_retry_s: default_auto_solve_retry_s(),
        }
    }
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

impl AchernarConfig {
    /// Cross-section rules serde cannot express. The daemon refuses to start
    /// on an error rather than run a configuration nobody chose.
    pub fn validate(&self) -> Result<(), String> {
        if self.focus.hfr_source == crate::focus::focus_agent::HfrSource::FqePrototype
            && !self.vision.fqe.prototype
        {
            return Err(
                "[focus] hfr_source = \"fqe_prototype\" requires [vision.fqe] \
                 prototype = true (the FQE prototype is opt-in, not qualified for \
                 R50/Z114)"
                    .to_string(),
            );
        }
        Ok(())
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
            plate_solve: PlateSolveConfig::default(),
            storage: StorageConfig::default(),
            telemetry: TelemetryConfig::default(),
            equipment: EquipmentConfig::default(),
            site: SiteConfig::default(),
            focuser: FocuserConfig::default(),
            guider: GuiderConfig::default(),
            safety: SafetyConfig::default(),
            weather: WeatherConfig::default(),
            vision: VisionConfig::default(),
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
fn default_search_radius_deg() -> f64 {
    30.0
}

#[cfg(test)]
mod example_config_tests {
    /// The example is what people copy; if it stops parsing, a new
    /// installation's first experience is a config error.
    #[test]
    fn the_example_config_parses() {
        let config: super::AchernarConfig =
            toml::from_str(include_str!("../achernar.example.toml"))
                .expect("achernar.example.toml must parse");
        assert_eq!(config.plate_solve.search_radius_deg, 30.0);
        assert_eq!(config.plate_solve.timeout_s, 300);
        assert_eq!(config.plate_solve.astap_path, None);
    }

    /// The FQE prototype keys default off/classical, parse when given, and
    /// `hfr_source = "fqe_prototype"` is refused without the prototype.
    #[test]
    fn the_fqe_prototype_keys_default_off_and_validate() {
        use super::*;
        use crate::focus::focus_agent::HfrSource;
        let empty: AchernarConfig = toml::from_str("").unwrap();
        assert!(!empty.vision.fqe.prototype);
        assert_eq!(empty.focus.hfr_source, HfrSource::Classical);
        assert!(empty.validate().is_ok());
        assert!(AchernarConfig::default().validate().is_ok());

        let example: AchernarConfig = toml::from_str(include_str!("../achernar.example.toml"))
            .expect("achernar.example.toml must parse");
        assert!(!example.vision.fqe.prototype);
        assert_eq!(example.focus.hfr_source, HfrSource::Classical);
        assert!(example.validate().is_ok());

        let refused: AchernarConfig =
            toml::from_str("[focus]\nhfr_source = \"fqe_prototype\"\n").unwrap();
        assert_eq!(refused.focus.hfr_source, HfrSource::FqePrototype);
        let error = refused.validate().unwrap_err();
        assert!(error.contains("[vision.fqe] prototype = true"), "{error}");

        let both: AchernarConfig = toml::from_str(
            "[focus]\nhfr_source = \"fqe_prototype\"\n[vision.fqe]\nprototype = true\n",
        )
        .unwrap();
        assert!(both.vision.fqe.prototype);
        assert!(both.validate().is_ok());
        let prototype_only: AchernarConfig =
            toml::from_str("[vision.fqe]\nprototype = true\n").unwrap();
        assert_eq!(prototype_only.focus.hfr_source, HfrSource::Classical);
        assert!(prototype_only.validate().is_ok());
        assert!(toml::from_str::<AchernarConfig>("[focus]\nhfr_source = \"fqe\"\n").is_err());
    }

    /// O11 g: `[filter_wheel] max_switches_per_hour` (60) and `[site]
    /// dut1_s` (0.0) have serde defaults and parse when given.
    #[test]
    fn the_wave_b_config_keys_default_and_parse() {
        use super::*;
        let empty: AchernarConfig = toml::from_str("").unwrap();
        assert_eq!(empty.filter_wheel.max_switches_per_hour, 60);
        assert_eq!(empty.site.dut1_s, 0.0);
        let set: AchernarConfig =
            toml::from_str("[filter_wheel]\nmax_switches_per_hour = 12\n[site]\ndut1_s = -0.2\n")
                .unwrap();
        assert_eq!(set.filter_wheel.max_switches_per_hour, 12);
        assert_eq!(set.site.dut1_s, -0.2);
        assert_eq!(set.equipment_configs().site.dut1_s, -0.2);
    }

    /// `[mount.center]` exposure settings default to 5 s at the camera's
    /// own gain and binning, parse when given, and a nonsensical exposure
    /// is reported.
    #[test]
    fn the_center_exposure_keys_default_and_parse() {
        use super::*;
        let empty: AchernarConfig = toml::from_str("").unwrap();
        assert_eq!(empty.mount.center.exposure_s, 5.0);
        assert_eq!(empty.mount.center.gain, None);
        assert_eq!(empty.mount.center.binning, None);
        let set: AchernarConfig =
            toml::from_str("[mount.center]\nexposure_s = 8.0\ngain = 120\nbinning = 2\n").unwrap();
        assert_eq!(set.mount.center.exposure_s, 8.0);
        assert_eq!(set.mount.center.gain, Some(120));
        assert_eq!(set.mount.center.binning, Some(2));
        assert_eq!(set.mount.center.tolerance_arcmin, 1.0);
        assert!(set
            .equipment_warnings()
            .iter()
            .all(|w| !w.contains("[mount.center]")));
        let bad: AchernarConfig = toml::from_str("[mount.center]\nexposure_s = 0\n").unwrap();
        assert!(bad
            .equipment_warnings()
            .iter()
            .any(|w| w.contains("[mount.center] exposure_s")));
    }

    /// The example's `[vision.background]` documents the defaults: legacy
    /// backend, unavailable-on-failure, and the SEP starting values. A
    /// missing section is the legacy path.
    #[test]
    fn the_example_vision_background_section_is_the_default() {
        use super::*;
        let config: AchernarConfig = toml::from_str(include_str!("../achernar.example.toml"))
            .expect("achernar.example.toml must parse");
        assert_eq!(
            config.vision.background,
            crate::vision::background_model::BackgroundSettings::default()
        );
        let empty: AchernarConfig = toml::from_str("").unwrap();
        assert_eq!(
            empty.vision.background.backend,
            crate::vision::background_model::BackgroundBackend::LegacyGlobalClip
        );
        let sep: AchernarConfig = toml::from_str(
            "[vision.background]\nbackend = \"sep\"\n[vision.background.sep]\nbw = 32\n",
        )
        .unwrap();
        assert_eq!(
            sep.vision.background.backend,
            crate::vision::background_model::BackgroundBackend::Sep
        );
        assert_eq!(sep.vision.background.sep.bw, 32);
        assert_eq!(sep.vision.background.sep.bh, 64);
    }

    /// `[vision.star_gates]` and the v2 `[vision.background.sep]` keys: the
    /// example documents the defaults, a missing section is the default, and
    /// each key parses on its own.
    #[test]
    fn the_star_gate_and_noise_policy_keys_default_and_parse() {
        use super::*;
        use crate::vision::background_model::NoisePolicy;
        use crate::vision::quality::{BlendTest, StarGateSettings};
        let example: AchernarConfig = toml::from_str(include_str!("../achernar.example.toml"))
            .expect("achernar.example.toml must parse");
        assert_eq!(example.vision.star_gates, StarGateSettings::default());
        let empty: AchernarConfig = toml::from_str("").unwrap();
        assert_eq!(empty.vision.star_gates.blend_test, BlendTest::DipAnnulusV2);
        assert!(empty.vision.star_gates.regroup_deblended);
        assert_eq!(
            empty.vision.background.sep.noise_policy,
            NoisePolicy::MaskedLocalV2
        );
        let set: AchernarConfig = toml::from_str(
            "[vision.star_gates]\nblend_test = \"peak_count_v1\"\nregroup_deblended = false\n\
             [vision.background.sep]\nnoise_policy = \"frame_wide_v1\"\n\
             rms_max_invalid_fraction = 0.1\nmask_minarea = 9\n",
        )
        .unwrap();
        assert_eq!(set.vision.star_gates.blend_test, BlendTest::PeakCountV1);
        assert!(!set.vision.star_gates.regroup_deblended);
        assert_eq!(
            set.vision.background.sep.noise_policy,
            NoisePolicy::FrameWideV1
        );
        assert_eq!(set.vision.background.sep.rms_max_invalid_fraction, 0.1);
        assert_eq!(set.vision.background.sep.mask_minarea, 9);
        assert_eq!(set.vision.background.sep.mask_thresh_sigma, 2.0);
        let only_regroup: AchernarConfig =
            toml::from_str("[vision.star_gates]\nregroup_deblended = false\n").unwrap();
        assert_eq!(
            only_regroup.vision.star_gates.blend_test,
            BlendTest::DipAnnulusV2
        );
        assert!(
            toml::from_str::<AchernarConfig>("[vision.star_gates]\nblend_test = \"v3\"\n").is_err()
        );
    }

    /// The example documents the equipment defaults, so a copied example
    /// behaves exactly like no file at all.
    #[test]
    fn the_example_equipment_sections_are_the_defaults() {
        use super::*;
        let config: AchernarConfig = toml::from_str(include_str!("../achernar.example.toml"))
            .expect("achernar.example.toml must parse");
        assert_eq!(config.equipment, EquipmentConfig::default());
        assert_eq!(config.site, SiteConfig::default());
        assert_eq!(config.focuser, FocuserConfig::default());
        assert_eq!(config.guider, GuiderConfig::default());
        assert_eq!(config.safety, SafetyConfig::default());
        assert_eq!(config.weather, WeatherConfig::default());
        assert_eq!(config.mount.center, MountCenterConfig::default());
        assert_eq!(config.mount.flip, MountFlipConfig::default());
        let defaults = MountConfig::default();
        assert_eq!(config.mount.settle_s, defaults.settle_s);
        assert_eq!(config.mount.slew_timeout_s, defaults.slew_timeout_s);
        assert_eq!(config.mount.sync_mode, defaults.sync_mode);
        assert_eq!(config.mount.time_sync, defaults.time_sync);
        assert_eq!(config.mount.jnow_flavour, defaults.jnow_flavour);
        assert_eq!(
            config.mount.equatorial_system_override,
            EquatorialSystemOverride::None
        );
        assert_eq!(config.mount.max_pulse_ms, defaults.max_pulse_ms);
        assert_eq!(config.mount.jog_deadman_ms, defaults.jog_deadman_ms);
        assert_eq!(config.mount.fallback_park, FallbackPark::None);
        assert_eq!(config.mount.safety.horizon_profile(), None);
        assert_eq!(config.mount.safety.limit_hysteresis_deg, 0.5);
        let wheel = FilterWheelConfig::default();
        assert_eq!(config.filter_wheel.settle_ms, wheel.settle_ms);
        assert_eq!(
            config.filter_wheel.max_switch_timeout_ms,
            wheel.max_switch_timeout_ms
        );
        assert_eq!(
            config.filter_wheel.use_focus_offsets,
            wheel.use_focus_offsets
        );
        assert_eq!(config.filter_wheel.af_filter, "");
        // The seeds are still parsed, and flagged as seeds.
        assert_eq!(config.filter_wheel.focus_offsets.len(), 7);
        assert!(config
            .equipment_warnings()
            .iter()
            .any(|w| w.contains("seeds only")));
    }

    #[test]
    fn equipment_defaults_open_nothing_and_write_nothing() {
        use super::*;
        let config = AchernarConfig::default();
        assert!(!config.equipment.allow_broadcast_discovery);
        assert!(!config.equipment.indi_disconnect_device);
        assert_eq!(config.equipment.alpaca_client_id, 0);
        assert!(!config.mount.time_sync);
        assert_eq!(config.site.sync, SiteSyncMode::VerifyOnly);
        assert_eq!(config.site.site(), Ok(None));
        assert!(!config.guider.connect_equipment);
        assert!(!config.guider.allow_policy_actuation);
        assert_eq!(config.safety.resume, SafetyResume::Manual);
        assert!(!config.mount.flip.enabled);
        assert_eq!(
            config.safety.effective_on_unsafe(&config.mount.safety),
            OnUnsafe::Park
        );
        let mut no_park = config.mount.safety.clone();
        no_park.park_on_safety_violation = false;
        assert_eq!(
            config.safety.effective_on_unsafe(&no_park),
            OnUnsafe::StopTracking
        );
        assert_eq!(config.guider.settle().timeout_s, 40.0);
        assert!(config.equipment_warnings().is_empty());
        let bundle = config.equipment_configs();
        assert_eq!(bundle.focus_safety.max_step_single, 500);
    }

    #[test]
    fn unknown_equipment_keys_are_ignored() {
        let config: super::AchernarConfig = toml::from_str(
            r#"
            [equipment]
            lost_after_s = 20
            some_future_key = "x"

            [site]
            latitude_deg = 40.0
            longitude_deg = -75.0
            elevation_m = 100
            observatory_code = "Z99"

            [mount]
            settle_s = 2.5
            unknown_mount_key = 1

            [mount.flip]
            enabled = true
            pause_before_flip = true

            [focuser]
            tempcomp_during_moves = "leave"
            focuser_colour = "red"

            [guider]
            dither_amount_px = 3
            phd2_flavour = "dev"

            [safety]
            on_unsafe = "stop_tracking"
            siren = true

            [weather]
            log_interval_s = 30
            wind_units = "kts"

            [filter_wheel]
            settle_ms = 750
            wheel_colour = "black"

            [a_section_from_the_future]
            anything = 1
            "#,
        )
        .expect("unknown keys must be ignored");
        assert_eq!(config.equipment.lost_after_s, 20);
        let site = config.site.site().unwrap().unwrap();
        assert_eq!(
            (site.lat_deg, site.lon_deg, site.elev_m),
            (40.0, -75.0, 100.0)
        );
        assert_eq!(config.mount.settle_s, 2.5);
        assert!(config.mount.flip.enabled);
        assert_eq!(
            config.focuser.tempcomp_during_moves,
            super::TempCompDuringMoves::Leave
        );
        assert_eq!(config.guider.dither_amount_px, 3.0);
        assert_eq!(config.safety.on_unsafe, super::OnUnsafe::StopTracking);
        assert_eq!(config.weather.log_interval_s, 30);
        assert_eq!(config.filter_wheel.settle_ms, 750);
        // Untouched sections keep their defaults.
        assert_eq!(config.mount.slew_timeout_s, 600);
        assert_eq!(config.mount.safety.max_hour_angle, 5.5);
    }

    #[test]
    fn wrong_equipment_values_are_errors_and_half_a_site_is_flagged() {
        for bad in [
            "[mount]\nsync_mode = \"sometimes\"\n",
            "[mount]\nequatorial_system_override = \"b1950\"\n",
            "[site]\nsync = \"always\"\n",
            "[guider]\nguiding_during_autofocus = \"maybe\"\n",
            "[safety]\nresume = \"eventually\"\n",
            "[equipment]\nalpaca_discovery_port = 70000\n",
        ] {
            assert!(
                toml::from_str::<super::AchernarConfig>(bad).is_err(),
                "{bad} must not parse"
            );
        }
        let config: super::AchernarConfig =
            toml::from_str("[site]\nlatitude_deg = 40.0\n").unwrap();
        assert!(config.site.site().is_err());
        assert!(config
            .equipment_warnings()
            .iter()
            .any(|w| w.contains("[site]")));
        let config: super::AchernarConfig =
            toml::from_str("[site]\nlatitude_deg = 95.0\nlongitude_deg = 0.0\nelevation_m = 0.0\n")
                .unwrap();
        assert!(config.site.site().is_err());
        let config: super::AchernarConfig =
            toml::from_str("[mount]\nequatorial_system_override = \"jnow_mean\"\n").unwrap();
        assert_eq!(
            config.mount.equatorial_system_override.frame(),
            Some(crate::equipment::types::EqFrame::JNowMean)
        );
    }
}
fn default_max_stars() -> u32 {
    500
}
fn default_solve_timeout_s() -> u64 {
    300
}
fn default_auto_solve() -> bool {
    true
}
fn default_auto_solve_retry_s() -> u64 {
    120
}

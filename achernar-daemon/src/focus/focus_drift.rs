//! Focus drift forecaster (FDF) — Section 7.6.3.
//!
//! Kalman-filter tracking of best-focus position with an online-estimated,
//! per-rig temperature coefficient. This is a classical state-space model, NOT
//! a neural network — there is no ONNX file and there should not be.
//!
//! Tuning lives in `[focus.drift]` in `achernar.toml`; the learned state is a
//! [`DriftState`] snapshot in the data directory. They are separate because
//! they have opposite lifecycles: one ships with the code, the other is
//! per-rig and changes nightly. See `models/focus_drift/0.1.0/README.md`.
//!
//! # Model
//!
//! State (matches the persisted [`DriftState`] schema):
//!
//! ```text
//!     x = [ p, v, k ]ᵀ
//!         p : best-focus position           (focuser steps)
//!         v : thermal-independent drift rate (steps / s)  — flexure, seeing trend
//!         k : temperature coefficient        (steps / °C)
//! ```
//!
//! Transition over `dt` seconds with a *measured* temperature change `ΔT`:
//!
//! ```text
//!     p' = p + v·dt + k·ΔT
//!     v' = v
//!     k' = k
//!
//!     F = | 1  dt  ΔT |
//!         | 0   1   0 |
//!         | 0   0   1 |
//! ```
//!
//! Note `ΔT` is an *exogenous, measured* input, so although `k` is an unknown
//! parameter being estimated jointly with the state, it enters multiplied by a
//! known regressor. The system therefore stays **linear** (a linear
//! time-varying KF) and needs no EKF/UKF. This is the standard augmented-state
//! approach to joint state/parameter estimation — equivalent to embedding a
//! recursive least-squares fit of `k` inside the filter (Ljung, *System
//! Identification*, 2nd ed., §11; Simon, *Optimal State Estimation*, §10).
//!
//! Process noise uses the **discrete white-noise acceleration (DWNA)** /
//! nearly-constant-velocity model for the `(p, v)` block — the canonical choice
//! for tracking a smoothly drifting quantity (Bar-Shalom, Li & Kirubarajan,
//! *Estimation with Applications to Tracking and Navigation*, 2001, §6.2.2):
//!
//! ```text
//!     Q_pv = σ_a² · | dt⁴/4  dt³/2 |
//!                   | dt³/2  dt²   |
//! ```
//!
//! `k` gets a slow random walk (`σ_k²·dt`) so the coefficient can adapt as the
//! tube approaches thermal equilibrium, rather than being frozen after the
//! first night.
//!
//! The covariance update uses the **Joseph stabilized form**, which preserves
//! symmetry and positive-definiteness under finite precision far better than
//! the textbook `(I − KH)P` shortcut (Bucy & Joseph 1968; Bar-Shalom §5.2.4).
//! An autofocus filter that runs all night must not accumulate asymmetry.
//!
//! # Physical grounding of `k`
//!
//! For a reflector the dominant term is tube expansion moving the primary
//! relative to the focal plane: `Δf ≈ α · L · ΔT`, with α the tube CTE
//! (aluminium ≈ 23×10⁻⁶ /°C, steel ≈ 11×10⁻⁶, carbon fibre ≈ 0). A 900 mm
//! aluminium tube therefore shifts ≈ 21 µm/°C, which the focuser's
//! steps-per-micron converts into `k`. We do **not** hard-code this: the sign
//! depends on focuser orientation and the magnitude on construction, so `k` is
//! learned per rig and this note exists only to sanity-check the fitted value.

use serde::{Deserialize, Serialize};

/// Row-major 3×3 matrix.
type Mat3 = [f64; 9];
/// 3-vector.
type Vec3 = [f64; 3];

#[inline]
fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            let mut sum = 0.0;
            for i in 0..3 {
                sum += a[r * 3 + i] * b[i * 3 + c];
            }
            out[r * 3 + c] = sum;
        }
    }
    out
}

#[inline]
fn mat_mul_transpose(a: &Mat3, b: &Mat3) -> Mat3 {
    // a · bᵀ
    let mut out = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            let mut sum = 0.0;
            for i in 0..3 {
                sum += a[r * 3 + i] * b[c * 3 + i];
            }
            out[r * 3 + c] = sum;
        }
    }
    out
}

#[inline]
fn mat_add(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [0.0; 9];
    for i in 0..9 {
        out[i] = a[i] + b[i];
    }
    out
}

#[inline]
fn mat_vec(a: &Mat3, x: &Vec3) -> Vec3 {
    [
        a[0] * x[0] + a[1] * x[1] + a[2] * x[2],
        a[3] * x[0] + a[4] * x[1] + a[5] * x[2],
        a[6] * x[0] + a[7] * x[1] + a[8] * x[2],
    ]
}

/// Force exact symmetry. Covariances drift asymmetric through rounding; the
/// Joseph form limits this but re-symmetrizing is free insurance.
#[inline]
fn symmetrize(p: &mut Mat3) {
    for r in 0..3 {
        for c in (r + 1)..3 {
            let mean = 0.5 * (p[r * 3 + c] + p[c * 3 + r]);
            p[r * 3 + c] = mean;
            p[c * 3 + r] = mean;
        }
    }
}

/// Error function, Abramowitz & Stegun 7.1.26 (|ε| ≤ 1.5×10⁻⁷).
/// Rust's std has no `erf`, and the drift decision only needs ~6 digits.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    const P: f64 = 0.327_591_1;
    const A: [f64; 5] = [
        0.254_829_592,
        -0.284_496_736,
        1.421_413_741,
        -1.453_152_027,
        1.061_405_429,
    ];
    let t = 1.0 / (1.0 + P * x);
    let poly = t * (A[0] + t * (A[1] + t * (A[2] + t * (A[3] + t * A[4]))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Standard normal CDF.
#[inline]
fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Total critical focus zone in micrometres.
///
/// A longitudinal defocus δ produces a peak wavefront error `W = δ / (8·N²)`
/// at the pupil edge (Smith, *Modern Optical Engineering*; Mahajan, *Optical
/// Imaging and Aberrations*). Applying the Rayleigh quarter-wave criterion
/// `W = λ/4` gives a one-sided tolerance `δ = 2·λ·N²`, so the full zone either
/// side of focus is:
///
/// ```text
///     CFZ = 4 · λ · N²
/// ```
///
/// e.g. λ = 550 nm, N = 8 → CFZ ≈ 141 µm.
///
/// Amateur tools sometimes quote `2λN²` or `4.88λN²` instead; those are the
/// same relation under a different sharpness criterion, which is why the
/// operating threshold is configurable rather than derived here.
pub fn critical_focus_zone_um(f_ratio: f64, wavelength_nm: f64) -> f64 {
    4.0 * (wavelength_nm * 1e-3) * f_ratio * f_ratio
}

/// Tuning for the forecaster.
///
/// # Estimation and decision are separable
///
/// The filter does two jobs, and only the second needs to know anything about
/// the optics:
///
/// * **Estimation** — identify `[position, drift_rate, temp_coefficient]` from
///   (temperature, position) pairs. Pure system identification. Needs no
///   optical constants whatsoever.
/// * **Decision** — turn a predicted drift into "should we refocus?". Needs a
///   threshold, which is half the critical focus zone, which needs the optics.
///
/// `refocus_threshold_steps` enters only the second. Refusing to construct the
/// filter without it would take out the estimator as collateral damage — and
/// the estimator is the expensive half: identifying `k` needs varied thermal
/// conditions across multiple sessions, because under steady cooling `k·ΔT`
/// and `v·dt` are collinear (see `initial_temp_coeff_var` below). Weeks of sky
/// would be thrown away for want of two numbers that were never needed to
/// collect it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftConfig {
    /// Refocus threshold in steps — spec Section 7.6.3 sets this to *half* the
    /// critical focus zone. Use [`DriftConfig::threshold_from_optics`].
    ///
    /// `None` means the optics are unknown. The filter still estimates and
    /// still predicts drift; it declines to convert that drift into advice, and
    /// says so via [`AdviceStatus`]. There is deliberately no default: a fixed
    /// step count is several times the critical focus zone on a fast scope and
    /// a fraction of it on a slow one, so any placeholder produces
    /// `refocus_recommended` values the system cannot support.
    pub refocus_threshold_steps: Option<f64>,
    /// Minimum P(|drift| > threshold) before recommending a refocus. Guards
    /// against a noisy filter triggering constantly early in a session.
    pub min_confidence_to_act: f64,
    /// DWNA acceleration noise σ_a (steps/s²): how abruptly drift rate changes.
    pub sigma_accel_steps_per_s2: f64,
    /// Random-walk noise on the temperature coefficient (steps/°C per √s).
    pub sigma_temp_coeff: f64,
    /// Measurement noise σ (steps) on an observed best-focus position.
    pub sigma_measurement_steps: f64,

    // ---- Initial covariance (prior) ------------------------------------
    // These matter more than they look. Under a steadily falling temperature
    // the two drift terms are *collinear*: ΔT = (dT/dt)·dt, so `k·ΔT` and
    // `v·dt` explain a measurement equally well and the data alone cannot
    // split them (the input is not persistently exciting — Ljung, *System
    // Identification*, §13.2). What the filter reports is then decided by the
    // ratio of these priors, so they must be physically honest.
    /// Prior variance on best-focus position (steps²). Diffuse: the first
    /// measurement effectively sets it.
    pub initial_position_var: f64,
    /// Prior variance on mechanical drift rate ((steps/s)²).
    pub initial_drift_rate_var: f64,
    /// Prior variance on the temperature coefficient ((steps/°C)²).
    ///
    /// σ = 20 steps/°C: a 900 mm aluminium tube shifts ≈21 µm/°C, which is
    /// tens of steps on a typical focuser. An earlier value of 0.1 (σ ≈ 0.32
    /// steps/°C) asserted the coefficient was essentially zero and pinned `k`
    /// at its initial value no matter what the data said.
    pub initial_temp_coeff_var: f64,
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self {
            // Not 50.0. That was a number someone typed, and every
            // `refocus_recommended` derived from it was an assertion the system
            // could not support. Unknown is modelled, not defaulted.
            refocus_threshold_steps: None,
            min_confidence_to_act: 0.7,
            sigma_accel_steps_per_s2: 1e-4,
            sigma_temp_coeff: 1e-3,
            sigma_measurement_steps: 10.0,
            initial_position_var: 100.0,
            initial_drift_rate_var: 1.0,
            initial_temp_coeff_var: 400.0,
        }
    }
}

impl DriftConfig {
    /// Half the critical focus zone, expressed in focuser steps.
    pub fn threshold_from_optics(
        f_ratio: f64,
        wavelength_nm: f64,
        steps_per_micron: f64,
    ) -> f64 {
        0.5 * critical_focus_zone_um(f_ratio, wavelength_nm) * steps_per_micron
    }
}

/// Measure the focuser's scale (steps per micron of travel) from the slope of a
/// measured autofocus V-curve.
///
/// # Why this exists
///
/// `steps_per_micron` is mechanical: turns of a knob against a drawtube. It is
/// not printed on anything and few people know it for their focuser. Requiring
/// it by hand is the friction that guarantees the refocus threshold stays unset
/// — and the sweep the daemon already runs has measured it.
///
/// # Derivation
///
/// A longitudinal defocus `δ` (µm) at focal ratio `N` produces a blur circle of
/// diameter `δ/N` at the sensor — similar triangles on the converging cone,
/// whose half-angle has tangent `1/(2N)`.
///
/// For a uniformly illuminated disc of radius `R`, half the flux falls inside
/// `R/√2`, so the half-flux radius of a defocused star is
///
/// ```text
///     HFR_defocus = (δ / 2N) / √2  =  δ / (2√2 · N)          [µm]
/// ```
///
/// Far from focus this term dominates the in-focus PSF, which is the regime the
/// V-curve's straight wings are fitted in. So along a wing,
///
/// ```text
///     dHFR[µm] / dδ[µm] = 1 / (2√2 · N)
/// ```
///
/// The solver reports the wing slope in *pixels of HFR per focuser step*.
/// Converting the left side to those units introduces the pixel pitch and the
/// unknown we want:
///
/// ```text
///     dHFR[µm]/dx[step] = slope[px/step] · pixel_size[µm/px]
///     dδ[µm]  /dx[step] = 1 / steps_per_micron
///
///  ⇒  slope · pixel_size · steps_per_micron = 1 / (2√2 · N)
///
///  ⇒  steps_per_micron = 1 / (2√2 · N · slope · pixel_size)
/// ```
///
/// # What this ignores
///
/// * **Central obstruction.** A defocused star from an obstructed aperture is
///   an annulus, not a disc. For obstruction ratio `ε` the half-flux radius is
///   `√((1+ε²)/2)·R` rather than `R/√2`, so a 30% obstruction makes this
///   estimate ~4% low. Below the noise on a threshold whose own convention
///   (CFZ = 2λN² vs 4λN²) is a factor of two.
/// * **Seeing and diffraction**, which add in quadrature and matter only near
///   focus — excluded by fitting the wing, not the vertex.
/// * **Focuser non-linearity and backlash.** Both are real; both argue for
///   treating this as an estimate, which is why the result is tagged
///   [`ThresholdSource::MeasuredVCurve`] and always loses to a configured value.
///
/// Returns `None` rather than a number when any input is missing or
/// non-physical. A wrong scale would silently mis-size every refocus decision
/// that follows.
pub fn steps_per_micron_from_vcurve(
    wing_slope_px_per_step: f64,
    f_ratio: f64,
    pixel_size_um: f64,
) -> Option<f64> {
    let slope = wing_slope_px_per_step.abs();
    if !slope.is_finite() || slope <= 0.0 {
        return None;
    }
    if !f_ratio.is_finite() || f_ratio <= 0.0 {
        return None;
    }
    if !pixel_size_um.is_finite() || pixel_size_um <= 0.0 {
        return None;
    }

    let steps_per_micron = 1.0 / (2.0 * std::f64::consts::SQRT_2 * f_ratio * slope * pixel_size_um);
    (steps_per_micron.is_finite() && steps_per_micron > 0.0).then_some(steps_per_micron)
}

/// Why the decision half of a prediction is or is not available.
///
/// Separate from `initialized` because they fail for unrelated reasons and are
/// fixed by unrelated actions: `NotInitialized` is cured by observing, and
/// `ThresholdUnknown` is cured by telling the daemon about the optics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum AdviceStatus {
    /// A threshold is known; the decision fields are populated.
    Available {
        threshold_steps: f64,
        source: ThresholdSource,
    },
    /// The filter has not seen enough measurements to predict anything, so
    /// there is nothing to compare against a threshold.
    NotInitialized,
    /// The optics are unknown, so half the critical focus zone cannot be
    /// computed. Estimation is unaffected — drift and its interval are still
    /// in this prediction.
    ThresholdUnknown,
}

/// Where a refocus threshold came from. Recorded for the same reason
/// `TemperatureState` is: a threshold the operator declared, one read from a
/// rig profile, and one inferred from a V-curve are not the same kind of claim,
/// and a number in a column cannot be told apart from any of the others later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThresholdSource {
    /// `[focus.drift] refocus_threshold_steps` — declared outright.
    Configured,
    /// Derived from `[focus.drift]` f_ratio + steps_per_micron.
    ConfiguredOptics,
    /// Derived from the `rigs` row for this rig.
    RigProfile,
    /// Derived from a measured autofocus V-curve. An estimate: see
    /// [`steps_per_micron_from_vcurve`] for its assumptions and error terms.
    MeasuredVCurve,
}

/// Drift prediction output.
///
/// The estimation fields (`predicted_drift_steps`, `confidence_interval`,
/// `initialized`) are threshold-free and always meaningful once the filter has
/// data. The decision fields are `Option` because they require a refocus
/// threshold, which requires optics the daemon may not have been told about.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftPrediction {
    // ── Estimation ────────────────────────────────────────────────────────
    /// Predicted focus drift in steps over the horizon (signed).
    pub predicted_drift_steps: f64,
    /// Half-width of the 95% interval on that drift (steps).
    pub confidence_interval: f64,
    /// False until the filter has seen enough measurements to be meaningful.
    pub initialized: bool,

    // ── Decision ──────────────────────────────────────────────────────────
    /// P(|drift| > threshold) — the quantity the decision is actually made on.
    /// `None` when no threshold is known.
    pub exceed_probability: Option<f64>,
    /// Whether a refocus is recommended. `None` when no threshold is known —
    /// which is distinct from `Some(false)`, "we checked and it is fine".
    pub refocus_recommended: Option<bool>,
    /// Estimated time until the threshold is crossed (seconds).
    ///
    /// `None` for two different reasons, disambiguated by `advice`: no
    /// threshold at all, or a threshold plus an effective drift rate too small
    /// to ever cross it.
    pub time_to_refocus_s: Option<f64>,
    /// Whether the decision fields above are populated, and why not if not.
    pub advice: AdviceStatus,
}

/// Serializable filter state — what gets snapshotted to
/// `<data_dir>/focus_drift_state.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftState {
    pub focus_position: f64,
    pub drift_rate: f64,
    pub temperature_coefficient: f64,
    pub covariance: Vec<f64>,
    pub last_temperature: Option<f64>,
    pub last_update: Option<chrono::DateTime<chrono::Utc>>,
    pub temperature_rate_c_per_s: f64,
    pub update_count: u32,
}

/// Kalman drift forecaster.
pub struct FocusDriftForecaster {
    /// State vector [position, drift_rate, temp_coefficient].
    state: Vec3,
    /// Covariance (3×3 row-major).
    covariance: Mat3,
    config: DriftConfig,
    /// Refocus threshold in force, with where it came from.
    ///
    /// Separate from `config.refocus_threshold_steps` because it can be learned
    /// after construction: an autofocus V-curve measures the focuser's scale
    /// (see [`steps_per_micron_from_vcurve`]), so a rig that started with no
    /// threshold can acquire one without the operator editing anything. The
    /// configured value always wins over a measured one.
    threshold: Option<(f64, ThresholdSource)>,
    last_temperature: Option<f64>,
    last_update: Option<chrono::DateTime<chrono::Utc>>,
    /// Smoothed dT/dt (°C per second), used to extrapolate cooling.
    temperature_rate: f64,
    update_count: u32,
}

impl Default for FocusDriftForecaster {
    fn default() -> Self {
        Self::new()
    }
}

impl FocusDriftForecaster {
    pub fn new() -> Self {
        Self::with_config(DriftConfig::default())
    }

    pub fn with_config(config: DriftConfig) -> Self {
        Self::with_config_and_source(config, ThresholdSource::Configured)
    }

    /// As [`with_config`], recording where the configured threshold came from.
    pub fn with_config_and_source(config: DriftConfig, source: ThresholdSource) -> Self {
        // Diagonal prior: position is unknown until the first measurement (which
        // effectively sets it); rate and coefficient start at zero but must be
        // free enough to move to their true magnitudes. See the field docs on
        // `DriftConfig` — the coefficient prior is in (steps/°C)² and a prior
        // that is too tight silently pins `k` regardless of the data.
        let covariance = [
            config.initial_position_var,
            0.0,
            0.0,
            0.0,
            config.initial_drift_rate_var,
            0.0,
            0.0,
            0.0,
            config.initial_temp_coeff_var,
        ];
        let threshold = config
            .refocus_threshold_steps
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|v| (v, source));
        Self {
            state: [0.0; 3],
            covariance,
            config,
            threshold,
            last_temperature: None,
            last_update: None,
            temperature_rate: 0.0,
            update_count: 0,
        }
    }

    pub fn config(&self) -> &DriftConfig {
        &self.config
    }

    /// The refocus threshold in force, and where it came from.
    pub fn threshold(&self) -> Option<(f64, ThresholdSource)> {
        self.threshold
    }

    /// Adopt a threshold measured from an autofocus V-curve.
    ///
    /// Ignored when a configured threshold is already in force: a number the
    /// operator wrote down beats one the software inferred, always. Returns
    /// whether it was adopted, so the caller can say which happened rather than
    /// leaving the operator to guess why their setting did or did not stick.
    pub fn adopt_measured_threshold(&mut self, threshold_steps: f64) -> bool {
        if !threshold_steps.is_finite() || threshold_steps <= 0.0 {
            return false;
        }
        match self.threshold {
            Some((_, ThresholdSource::MeasuredVCurve)) | None => {
                self.threshold = Some((threshold_steps, ThresholdSource::MeasuredVCurve));
                true
            }
            // Configured / ConfiguredOptics / RigProfile all outrank a
            // measurement.
            Some(_) => false,
        }
    }

    /// Steps/°C currently estimated for this rig, with its 1σ uncertainty.
    /// Feeds `profile_registry.json`.
    pub fn temperature_coefficient(&self) -> (f64, f64) {
        (self.state[2], self.covariance[8].max(0.0).sqrt())
    }

    pub fn drift_rate(&self) -> f64 {
        self.state[1]
    }

    pub fn position(&self) -> f64 {
        self.state[0]
    }

    pub fn update_count(&self) -> u32 {
        self.update_count
    }

    /// True once the filter has enough measurements to forecast. One point
    /// fixes position but says nothing about rate or coefficient.
    pub fn is_initialized(&self) -> bool {
        self.update_count >= 2
    }

    /// Fold in a newly measured best-focus position (e.g. from a converged
    /// autofocus run) taken at `temperature` and `timestamp`.
    ///
    /// The first call seeds the position directly rather than running a full
    /// update against a meaningless prior.
    pub fn update(
        &mut self,
        measured_position: f64,
        temperature: f64,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) {
        if !measured_position.is_finite() || !temperature.is_finite() {
            tracing::warn!("FDF: ignoring non-finite focus/temperature measurement");
            return;
        }

        if self.update_count == 0 {
            self.state[0] = measured_position;
            self.covariance[0] = self.config.sigma_measurement_steps.powi(2);
            self.last_temperature = Some(temperature);
            self.last_update = Some(timestamp);
            self.update_count = 1;
            return;
        }

        let dt = self
            .last_update
            .map(|previous| (timestamp - previous).num_milliseconds() as f64 / 1000.0)
            .unwrap_or(0.0);
        if dt <= 0.0 {
            // Out-of-order or duplicate sample: a negative dt would corrupt the
            // covariance. Drop it rather than integrating backwards.
            tracing::warn!("FDF: non-positive dt ({dt:.3}s); measurement ignored");
            return;
        }

        let delta_temperature = temperature - self.last_temperature.unwrap_or(temperature);

        // Track dT/dt with a light EMA so the horizon extrapolation is not
        // dominated by a single noisy thermometer reading.
        let instantaneous_rate = delta_temperature / dt;
        if instantaneous_rate.is_finite() {
            const ALPHA: f64 = 0.3;
            self.temperature_rate = if self.update_count <= 1 {
                instantaneous_rate
            } else {
                ALPHA * instantaneous_rate + (1.0 - ALPHA) * self.temperature_rate
            };
        }

        // ---- Predict -------------------------------------------------------
        let f: Mat3 = [
            1.0, dt, delta_temperature, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0,
        ];
        let variance_accel = self.config.sigma_accel_steps_per_s2.powi(2);
        let q: Mat3 = [
            variance_accel * dt.powi(4) / 4.0,
            variance_accel * dt.powi(3) / 2.0,
            0.0,
            variance_accel * dt.powi(3) / 2.0,
            variance_accel * dt * dt,
            0.0,
            0.0,
            0.0,
            self.config.sigma_temp_coeff.powi(2) * dt,
        ];

        let predicted_state = mat_vec(&f, &self.state);
        let mut predicted_cov = mat_add(&mat_mul_transpose(&mat_mul(&f, &self.covariance), &f), &q);
        symmetrize(&mut predicted_cov);

        // ---- Update (H = [1, 0, 0]) ----------------------------------------
        let r = self.config.sigma_measurement_steps.powi(2);
        let innovation = measured_position - predicted_state[0];
        let s = predicted_cov[0] + r;
        if !(s > 0.0) || !s.is_finite() {
            tracing::error!("FDF: non-positive innovation covariance; resetting filter");
            self.reset();
            return;
        }
        // K = P Hᵀ S⁻¹ → first column of P scaled by 1/S.
        let gain: Vec3 = [
            predicted_cov[0] / s,
            predicted_cov[3] / s,
            predicted_cov[6] / s,
        ];

        self.state = [
            predicted_state[0] + gain[0] * innovation,
            predicted_state[1] + gain[1] * innovation,
            predicted_state[2] + gain[2] * innovation,
        ];

        // Joseph form: P = (I − KH) P (I − KH)ᵀ + K R Kᵀ.
        let i_kh: Mat3 = [
            1.0 - gain[0],
            0.0,
            0.0,
            -gain[1],
            1.0,
            0.0,
            -gain[2],
            0.0,
            1.0,
        ];
        let mut updated = mat_mul_transpose(&mat_mul(&i_kh, &predicted_cov), &i_kh);
        for r_idx in 0..3 {
            for c_idx in 0..3 {
                updated[r_idx * 3 + c_idx] += gain[r_idx] * r * gain[c_idx];
            }
        }
        symmetrize(&mut updated);
        self.covariance = updated;

        self.last_temperature = Some(temperature);
        self.last_update = Some(timestamp);
        self.update_count = self.update_count.saturating_add(1);
    }

    /// Forecast drift from the last measured focus position over `horizon_seconds`,
    /// given the current `temperature`.
    ///
    /// Drift accrues from three terms:
    ///
    /// ```text
    ///     ΔT_total = (T_now − T_at_last_update) + (dT/dt)·h
    ///     drift(h) = v·h + k·ΔT_total
    /// ```
    ///
    /// i.e. temperature change already realized since the last update, plus the
    /// change expected over the horizon at the observed cooling rate.
    ///
    /// Variance propagates through the Jacobian `J = [0, h, ΔT_total]ᵀ`
    /// (position drops out — this is a *change*, not an absolute), plus the
    /// DWNA process noise accumulated over the horizon:
    ///
    /// ```text
    ///     Var = h²·P_vv + ΔT²·P_kk + 2·h·ΔT·P_vk + σ_a²·h⁴/4
    /// ```
    pub fn predict(&self, temperature: f64, horizon_seconds: f64) -> DriftPrediction {
        let horizon = horizon_seconds.max(0.0);
        if !self.is_initialized() || !temperature.is_finite() {
            return DriftPrediction {
                predicted_drift_steps: 0.0,
                confidence_interval: f64::INFINITY,
                initialized: false,
                exceed_probability: None,
                refocus_recommended: None,
                time_to_refocus_s: None,
                advice: AdviceStatus::NotInitialized,
            };
        }

        let realized = temperature - self.last_temperature.unwrap_or(temperature);
        let delta_temperature = realized + self.temperature_rate * horizon;

        let velocity = self.state[1];
        let coefficient = self.state[2];
        let drift = velocity * horizon + coefficient * delta_temperature;

        let p_vv = self.covariance[4];
        let p_kk = self.covariance[8];
        let p_vk = self.covariance[5];
        let variance = (horizon * horizon * p_vv
            + delta_temperature * delta_temperature * p_kk
            + 2.0 * horizon * delta_temperature * p_vk
            + self.config.sigma_accel_steps_per_s2.powi(2) * horizon.powi(4) / 4.0)
            .max(0.0);
        let sigma = variance.sqrt();

        // ---- Estimation is complete; everything above is threshold-free ----

        // Without a threshold there is nothing to compare `drift` against. The
        // drift and its interval are still real measurements and are returned.
        let Some((threshold_steps, source)) = self.threshold else {
            return DriftPrediction {
                predicted_drift_steps: drift,
                confidence_interval: 1.96 * sigma,
                initialized: true,
                exceed_probability: None,
                refocus_recommended: None,
                time_to_refocus_s: None,
                advice: AdviceStatus::ThresholdUnknown,
            };
        };

        let threshold = threshold_steps.abs();
        // P(|drift| > θ) for drift ~ N(µ, σ²).
        let exceed_probability = if sigma > 0.0 {
            let upper = 1.0 - normal_cdf((threshold - drift) / sigma);
            let lower = normal_cdf((-threshold - drift) / sigma);
            (upper + lower).clamp(0.0, 1.0)
        } else if drift.abs() > threshold {
            1.0
        } else {
            0.0
        };

        // Effective rate combines mechanical drift and thermally-driven motion.
        let effective_rate = velocity + coefficient * self.temperature_rate;
        let time_to_refocus_s = if effective_rate.abs() > 1e-9 {
            let remaining = threshold - realized.abs() * coefficient.abs();
            let seconds = (remaining.max(0.0)) / effective_rate.abs();
            seconds.is_finite().then_some(seconds)
        } else {
            None
        };

        DriftPrediction {
            predicted_drift_steps: drift,
            confidence_interval: 1.96 * sigma,
            initialized: true,
            exceed_probability: Some(exceed_probability),
            refocus_recommended: Some(exceed_probability >= self.config.min_confidence_to_act),
            time_to_refocus_s,
            advice: AdviceStatus::Available {
                threshold_steps: threshold,
                source,
            },
        }
    }

    /// Snapshot for persistence. Drift knowledge must survive a daemon restart
    /// mid-session — a night of thermal identification is expensive to redo.
    ///
    /// Deliberately excludes the threshold: that describes the optics and comes
    /// from config or a fresh measurement, not from what the filter learned.
    pub fn to_state(&self) -> DriftState {
        DriftState {
            focus_position: self.state[0],
            drift_rate: self.state[1],
            temperature_coefficient: self.state[2],
            covariance: self.covariance.to_vec(),
            last_temperature: self.last_temperature,
            last_update: self.last_update,
            temperature_rate_c_per_s: self.temperature_rate,
            update_count: self.update_count,
        }
    }

    /// Restore from a persisted snapshot. A covariance of the wrong length is
    /// rejected rather than silently zero-filled.
    pub fn from_state(state: DriftState, config: DriftConfig) -> Result<Self, String> {
        if state.covariance.len() != 9 {
            return Err(format!(
                "expected a 3x3 covariance (9 values), got {}",
                state.covariance.len()
            ));
        }
        let mut covariance = [0.0; 9];
        covariance.copy_from_slice(&state.covariance);
        let threshold = config
            .refocus_threshold_steps
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|v| (v, ThresholdSource::Configured));
        Ok(Self {
            state: [
                state.focus_position,
                state.drift_rate,
                state.temperature_coefficient,
            ],
            covariance,
            config,
            threshold,
            last_temperature: state.last_temperature,
            last_update: state.last_update,
            temperature_rate: state.temperature_rate_c_per_s,
            update_count: state.update_count,
        })
    }

    /// Reset the filter state (new rig, new night, or numerical failure).
    ///
    /// Keeps the threshold. It describes the optics, not the thermal history —
    /// resetting what the filter learned about the tube says nothing about the
    /// focal ratio.
    pub fn reset(&mut self) {
        let config = self.config.clone();
        let threshold = self.threshold;
        *self = Self::with_config(config);
        self.threshold = threshold;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(seconds: i64) -> chrono::DateTime<chrono::Utc> {
        chrono::DateTime::from_timestamp(1_700_000_000 + seconds, 0).unwrap()
    }

    #[test]
    fn critical_focus_zone_matches_rayleigh_quarter_wave() {
        // CFZ = 4λN²; λ=550nm, N=8 -> 4 * 0.55µm * 64 = 140.8 µm.
        let cfz = critical_focus_zone_um(8.0, 550.0);
        assert!((cfz - 140.8).abs() < 0.1, "cfz={cfz}");
        // f/4 should be 4x tighter than f/8 (quadratic in N).
        let fast = critical_focus_zone_um(4.0, 550.0);
        assert!((cfz / fast - 4.0).abs() < 1e-9);
        // Half-CFZ in steps at 2 steps/µm.
        let threshold = DriftConfig::threshold_from_optics(8.0, 550.0, 2.0);
        assert!((threshold - 140.8).abs() < 0.2, "threshold={threshold}");
    }

    /// A forecaster with a threshold, for tests about the DECISION half.
    /// Tests about the ESTIMATION half use `FocusDriftForecaster::new()`, which
    /// now has no threshold at all — that is the point of the split.
    fn with_threshold(steps: f64) -> FocusDriftForecaster {
        let mut config = DriftConfig::default();
        config.refocus_threshold_steps = Some(steps);
        FocusDriftForecaster::with_config(config)
    }

    #[test]
    fn learns_constant_mechanical_drift_rate() {
        let mut fdf = with_threshold(50.0);
        // 0.5 steps/s at constant temperature -> pure velocity term.
        for i in 0..40 {
            let t = i as i64 * 30;
            fdf.update(1000.0 + 0.5 * t as f64, 10.0, at(t));
        }
        assert!(fdf.is_initialized());
        assert!(
            (fdf.drift_rate() - 0.5).abs() < 0.05,
            "drift_rate={}",
            fdf.drift_rate()
        );

        // Over 600 s that is ~300 steps of drift.
        let prediction = fdf.predict(10.0, 600.0);
        assert!(
            (prediction.predicted_drift_steps - 300.0).abs() < 30.0,
            "drift={}",
            prediction.predicted_drift_steps
        );
        assert_eq!(prediction.refocus_recommended, Some(true));
    }

    #[test]
    fn learns_temperature_coefficient_sign_and_magnitude() {
        // The temperature must be *persistently exciting* for `k` to be
        // identifiable: under a monotonic ramp, ΔT is proportional to dt and
        // `k·ΔT` is indistinguishable from `v·dt` (see the collinearity test
        // below). An oscillating profile breaks that proportionality, so the
        // data — not the prior — determines the split. Physically this is a
        // tube warming and cooling as cloud passes over, which is exactly when
        // the coefficient is learnable.
        let mut fdf = FocusDriftForecaster::new();
        let k_true = -20.0;
        let reference_temperature = 15.0;
        for i in 0..60 {
            let t = i as i64 * 30;
            let temperature = reference_temperature
                + 3.0 * (2.0 * std::f64::consts::PI * t as f64 / 600.0).sin();
            let position = 1000.0 + k_true * (temperature - reference_temperature);
            fdf.update(position, temperature, at(t));
        }
        let (k, sigma) = fdf.temperature_coefficient();
        assert!((k - k_true).abs() < 2.0, "k={k} (sigma={sigma})");
        // With the thermal term identified, mechanical drift stays near zero.
        assert!(fdf.drift_rate().abs() < 0.05, "v={}", fdf.drift_rate());
    }

    #[test]
    fn constant_ramp_is_collinear_but_the_forecast_still_holds() {
        // Documents a real limitation rather than hiding it. Cooling at a
        // constant 0.01 °C/s with k = -20 steps/°C moves focus +0.2 steps/s.
        // Per step: 30·v + (-0.3)·k = 6 — a single equation in two unknowns,
        // so the v/k split is set by the prior ratio, not the data.
        //
        // What IS invariant is the forecast: over a horizon h the prediction is
        // 300·v - 3·k = 60 for h = 300 s, which holds for *every* (v, k) pair
        // satisfying the constraint. Focus control consumes the forecast, not
        // the decomposition, so the filter is fit for purpose here — but the
        // reported coefficient should not be trusted as a rig calibration
        // until the temperature has varied non-monotonically.
        let mut fdf = FocusDriftForecaster::new();
        let start_temperature = 15.0;
        for i in 0..60 {
            let t = i as i64 * 30;
            let temperature = start_temperature - 0.01 * t as f64;
            let position = 1000.0 + 0.2 * t as f64;
            fdf.update(position, temperature, at(t));
        }
        let final_temperature = start_temperature - 0.01 * (59 * 30) as f64;
        let prediction = fdf.predict(final_temperature, 300.0);
        assert!(
            (prediction.predicted_drift_steps - 60.0).abs() < 6.0,
            "forecast must be split-invariant, got {}",
            prediction.predicted_drift_steps
        );
    }

    #[test]
    fn uninitialized_filter_never_recommends_refocus() {
        let fdf = with_threshold(50.0);
        let prediction = fdf.predict(10.0, 600.0);
        assert!(!prediction.initialized);
        // None, not Some(false): nothing was compared against the threshold.
        assert_eq!(prediction.refocus_recommended, None);
        assert_eq!(prediction.advice, AdviceStatus::NotInitialized);

        let mut fdf = with_threshold(50.0);
        fdf.update(1000.0, 10.0, at(0));
        assert!(!fdf.predict(10.0, 600.0).initialized, "one sample is not enough");
    }

    #[test]
    fn stable_focus_does_not_trigger_and_covariance_shrinks() {
        let mut fdf = with_threshold(50.0);
        let initial_position_variance = fdf.covariance[0];
        for i in 0..30 {
            let t = i as i64 * 30;
            fdf.update(1000.0, 10.0, at(t));
        }
        assert!(fdf.covariance[0] < initial_position_variance);
        let prediction = fdf.predict(10.0, 600.0);
        assert!(
            prediction.predicted_drift_steps.abs() < 10.0,
            "drift={}",
            prediction.predicted_drift_steps
        );
        assert_eq!(prediction.refocus_recommended, Some(false));
        // Covariance must stay symmetric and positive on the diagonal.
        for r in 0..3 {
            assert!(fdf.covariance[r * 3 + r] > 0.0);
            for c in 0..3 {
                let asymmetry = (fdf.covariance[r * 3 + c] - fdf.covariance[c * 3 + r]).abs();
                assert!(asymmetry < 1e-12, "covariance asymmetric at {r},{c}");
            }
        }
    }

    #[test]
    fn rejects_out_of_order_and_non_finite_samples() {
        let mut fdf = FocusDriftForecaster::new();
        fdf.update(1000.0, 10.0, at(0));
        fdf.update(1015.0, 10.0, at(30));
        let count = fdf.update_count();
        fdf.update(1020.0, 10.0, at(10)); // travels backwards in time
        fdf.update(f64::NAN, 10.0, at(60));
        fdf.update(1020.0, f64::INFINITY, at(90));
        assert_eq!(fdf.update_count(), count, "bad samples must not be folded in");
    }

    #[test]
    fn confidence_gate_suppresses_marginal_recommendations() {
        let mut config = DriftConfig::default();
        config.refocus_threshold_steps = Some(50.0);
        config.min_confidence_to_act = 0.99;
        let mut fdf = FocusDriftForecaster::with_config(config);
        for i in 0..10 {
            let t = i as i64 * 30;
            fdf.update(1000.0 + 0.18 * t as f64, 10.0, at(t));
        }
        // ~54 steps over 300s: just past the threshold, but a short history
        // leaves enough uncertainty that a 0.99 gate should not fire.
        let prediction = fdf.predict(10.0, 300.0);
        assert!(prediction.exceed_probability.unwrap() < 0.99);
        assert_eq!(prediction.refocus_recommended, Some(false));
    }

    // ── The estimation / decision split ──────────────────────────────────

    #[test]
    fn estimation_works_with_no_threshold_at_all() {
        // The whole point: a rig whose optics have never been declared still
        // learns its thermal behaviour. Refusing to construct the filter would
        // throw this away, and this is the half that costs sky to collect.
        let mut fdf = FocusDriftForecaster::new();
        assert_eq!(fdf.threshold(), None);

        let k_true = -20.0;
        let reference = 15.0;
        for i in 0..60 {
            let t = i as i64 * 30;
            let temperature =
                reference + 3.0 * (2.0 * std::f64::consts::PI * t as f64 / 600.0).sin();
            fdf.update(1000.0 + k_true * (temperature - reference), temperature, at(t));
        }

        let (k, _) = fdf.temperature_coefficient();
        assert!((k - k_true).abs() < 2.0, "coefficient must still be identified: k={k}");

        let prediction = fdf.predict(reference, 600.0);
        assert!(prediction.initialized);
        assert!(prediction.predicted_drift_steps.is_finite());
        assert!(prediction.confidence_interval.is_finite());
        // ...but no advice, and it says why.
        assert_eq!(prediction.advice, AdviceStatus::ThresholdUnknown);
        assert_eq!(prediction.refocus_recommended, None);
        assert_eq!(prediction.exceed_probability, None);
        assert_eq!(prediction.time_to_refocus_s, None);
    }

    #[test]
    fn a_threshold_learned_later_turns_on_advice_without_relearning() {
        let mut fdf = FocusDriftForecaster::new();
        for i in 0..40 {
            let t = i as i64 * 30;
            fdf.update(1000.0 + 0.5 * t as f64, 10.0, at(t));
        }
        let before = fdf.predict(10.0, 600.0);
        assert_eq!(before.advice, AdviceStatus::ThresholdUnknown);

        assert!(fdf.adopt_measured_threshold(50.0));
        let after = fdf.predict(10.0, 600.0);

        // The estimate is untouched — adopting a threshold is not new evidence.
        assert!(
            (before.predicted_drift_steps - after.predicted_drift_steps).abs() < 1e-12,
            "adopting a threshold must not perturb the estimator"
        );
        assert_eq!(after.refocus_recommended, Some(true));
        assert_eq!(
            after.advice,
            AdviceStatus::Available {
                threshold_steps: 50.0,
                source: ThresholdSource::MeasuredVCurve,
            }
        );
    }

    #[test]
    fn a_configured_threshold_outranks_a_measured_one() {
        let mut fdf = with_threshold(50.0);
        assert!(!fdf.adopt_measured_threshold(999.0), "config must win");
        assert_eq!(fdf.threshold(), Some((50.0, ThresholdSource::Configured)));

        // A measured value may be refined by a later measurement, though.
        let mut fdf = FocusDriftForecaster::new();
        assert!(fdf.adopt_measured_threshold(80.0));
        assert!(fdf.adopt_measured_threshold(86.0));
        assert_eq!(fdf.threshold(), Some((86.0, ThresholdSource::MeasuredVCurve)));

        // Nonsense is refused rather than adopted.
        assert!(!fdf.adopt_measured_threshold(0.0));
        assert!(!fdf.adopt_measured_threshold(f64::NAN));
        assert_eq!(fdf.threshold(), Some((86.0, ThresholdSource::MeasuredVCurve)));
    }

    #[test]
    fn reset_clears_learning_but_keeps_the_threshold() {
        let mut fdf = with_threshold(50.0);
        for i in 0..10 {
            fdf.update(1000.0 + i as f64, 10.0, at(i as i64 * 30));
        }
        fdf.reset();
        assert_eq!(fdf.update_count(), 0);
        // The threshold describes the optics, not the thermal history.
        assert_eq!(fdf.threshold(), Some((50.0, ThresholdSource::Configured)));
    }

    // ── V-curve scale measurement ────────────────────────────────────────

    #[test]
    fn vcurve_scale_round_trips() {
        // Pick a true scale, compute the wing slope it implies, recover it.
        for (spm_true, f_ratio, pixel_um) in
            [(2.5, 5.6, 3.76), (1.0, 10.0, 9.0), (12.0, 4.0, 2.4), (0.4, 2.8, 5.94)]
        {
            let slope = 1.0
                / (2.0 * std::f64::consts::SQRT_2 * f_ratio * spm_true * pixel_um);
            let recovered =
                steps_per_micron_from_vcurve(slope, f_ratio, pixel_um).expect("physical inputs");
            assert!(
                (recovered - spm_true).abs() < 1e-9,
                "f/{f_ratio} px={pixel_um}: expected {spm_true}, got {recovered}"
            );
        }
    }

    #[test]
    fn vcurve_scale_sign_is_irrelevant() {
        // The left wing has a negative slope and the right a positive one; both
        // measure the same mechanical scale.
        let left = steps_per_micron_from_vcurve(-0.006716, 5.6, 3.76).unwrap();
        let right = steps_per_micron_from_vcurve(0.006716, 5.6, 3.76).unwrap();
        assert!((left - right).abs() < 1e-12);
    }

    #[test]
    fn vcurve_scale_refuses_unphysical_inputs() {
        for (slope, n, px) in [
            (0.0, 5.6, 3.76),           // a flat wing measures nothing
            (0.006, 0.0, 3.76),         // f/0
            (0.006, 5.6, 0.0),          // zero-size pixels
            (f64::NAN, 5.6, 3.76),
            (0.006, f64::INFINITY, 3.76),
        ] {
            assert!(
                steps_per_micron_from_vcurve(slope, n, px).is_none(),
                "must refuse ({slope}, {n}, {px}) rather than invent a scale"
            );
        }
    }

    #[test]
    fn vcurve_scale_produces_a_believable_threshold() {
        // f/5.6, 3.76 µm pixels, a V-curve wing rising 3.36 px over 500 steps.
        let slope = 3.36 / 500.0;
        let spm = steps_per_micron_from_vcurve(slope, 5.6, 3.76).unwrap();
        assert!((spm - 2.5).abs() < 0.05, "spm={spm}");

        let threshold = DriftConfig::threshold_from_optics(5.6, 550.0, spm);
        assert!(
            (threshold - 86.2).abs() < 2.0,
            "half-CFZ should be ~86 steps, got {threshold}"
        );
    }

    #[test]
    fn state_round_trips_through_json() {
        let mut fdf = FocusDriftForecaster::new();
        for i in 0..10 {
            let t = i as i64 * 30;
            fdf.update(1000.0 + 0.4 * t as f64, 12.0 - 0.001 * t as f64, at(t));
        }
        let json = serde_json::to_string(&fdf.to_state()).unwrap();
        let restored: DriftState = serde_json::from_str(&json).unwrap();
        let reloaded = FocusDriftForecaster::from_state(restored, DriftConfig::default()).unwrap();

        let before = fdf.predict(11.9, 600.0);
        let after = reloaded.predict(11.9, 600.0);
        assert!((before.predicted_drift_steps - after.predicted_drift_steps).abs() < 1e-9);
        assert_eq!(before.refocus_recommended, after.refocus_recommended);

        let bad = DriftState {
            covariance: vec![1.0, 2.0],
            ..fdf.to_state()
        };
        assert!(FocusDriftForecaster::from_state(bad, DriftConfig::default()).is_err());
    }
}

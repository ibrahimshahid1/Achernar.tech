//! Pointing model — Section 8.3 (M-4, v6 §8.3.4).
//!
//! Learns where a mount actually points from plate-solve residuals
//! (`pointing_samples`, H12) and predicts the correction for a new target.
//!
//! - **Epoch.** Every residual is mount reading minus sky truth with both in
//!   J2000: the solve is ICRS, and the mount's native (usually JNow)
//!   reading is converted at mid-exposure before the sample is written. The
//!   mount's native frame and its JNow flavour are part of the model key, so
//!   a "mean" and an "apparent" series are never mixed (M-4 gap 1).
//! - **Pier side, connection and sync generation.** A model is fitted for
//!   one mount identity, one connection, one sync generation (a sync changes
//!   the mount's own model) and one pier side. The mount runtime's
//!   `sync_generation` restarts at 0 with every connection and every daemon
//!   start, so a generation number alone does not name a mount model: the
//!   key also carries the connection's segment start
//!   ([`note_connection_sample`]), and a sample exposed before it belongs to
//!   an earlier connection. A sample from another side, connection or
//!   generation never enters the fit; every reconnect starts a new model.
//! - **RA in hours, with wrap.** Positions are hour angle (hours) and
//!   declination; the fit runs on unit vectors of (HA, Dec), so RA and HA
//!   wrap at 24 h and the pole are no special cases. The hour angle is LST
//!   minus the mount's native-frame RA (as H12 stores it) and the Dec is
//!   J2000; a prediction is asked at the target's hour angle in the mount's
//!   native frame.
//! - **Tangent plane.** A residual is the gnomonic offset (xi east, eta
//!   north, arcsec) of the mount reading on the plane tangent to the sky at
//!   the truth, not a raw RA difference that blows up near the pole. A
//!   correction is applied with the inverse projection at the target.
//! - **Clamped.** A correction never exceeds [`MAX_CORRECTION_ARCMIN`]; a
//!   residual beyond [`OUTLIER_ARCMIN`] (a wrong field, a sync mismatch) is
//!   left out of the fit. A target far from every sample is refused rather
//!   than extrapolated.
//!
//! The interpolant is a thin-plate spline on the sphere (`r² ln r` on chord
//! distance, with an affine term in the unit vector) once there are
//! [`MIN_SPLINE_SAMPLES`] samples, and inverse-distance weighting before
//! that. Each refit writes a `pointing_model_snapshots` row (§4.3).
//!
//! This module fits and predicts; it does not move the mount. A corrected
//! slew is an ordinary gated slew (the limits are checked on the corrected
//! target). The center loop does not apply it: under `sync_mode = "offset"`
//! the mount runtime already applies the solved offset, and applying both
//! would correct twice.

use std::collections::BTreeMap;
use std::sync::{Mutex, PoisonError};

use chrono::{DateTime, Timelike, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::SqlitePool;
use uuid::Uuid;

use crate::equipment::events::PointingSampleRecord;
use crate::equipment::types::{EqCoord, EqFrame, JNowFlavour, PierSide};
use crate::storage::{pointing_model_repo, pointing_samples_repo};

const ARCSEC_PER_RAD: f64 = 206_264.806_247_096_36;

/// Largest correction ever applied, arcmin.
pub const MAX_CORRECTION_ARCMIN: f64 = 30.0;
/// Residuals beyond this are not pointing error, arcmin.
pub const OUTLIER_ARCMIN: f64 = 120.0;
/// Below this the model interpolates by inverse distance.
pub const MIN_SPLINE_SAMPLES: usize = 6;
/// Fewer samples than this give no correction at all.
pub const MIN_CORRECTION_SAMPLES: usize = 3;
/// A target farther than this from every sample is not corrected, degrees.
pub const MAX_EXTRAPOLATION_DEG: f64 = 30.0;
/// Samples read for one refit.
const REFIT_LIMIT: u32 = 500;
/// Spline smoothing: plate-solve residuals carry arcsecond noise.
const SMOOTHING: f64 = 1.0e-4;

/// Where each mount connection's pointing samples begin: the earliest
/// mid-exposure of a sample written under it in this process.
///
/// `pointing_samples` has no connection column, and the generation numbers
/// of different connections collide. A sample of an earlier connection was
/// exposed before this one opened: H12 writes a sample only when every
/// mount reading of its window comes from the connection it is filed under.
/// So a sample exposed before the segment start is never this connection's.
static CONNECTION_SEGMENTS: Mutex<BTreeMap<Uuid, DateTime<Utc>>> = Mutex::new(BTreeMap::new());

/// Note a sample of `connection_id` exposed at `mid_exposure`; returns the
/// connection's segment start (the earliest such instant).
///
/// The start is floored to whole milliseconds, the precision
/// `pointing_samples.mid_exposure_utc` is stored at, so the stored copy of
/// the sample that set it is never earlier than the start itself.
pub fn note_connection_sample(connection_id: Uuid, mid_exposure: DateTime<Utc>) -> DateTime<Utc> {
    let mid_exposure = floor_to_millis(mid_exposure);
    let mut segments = CONNECTION_SEGMENTS
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let start = segments.entry(connection_id).or_insert(mid_exposure);
    if mid_exposure < *start {
        *start = mid_exposure;
    }
    *start
}

/// `at` without its sub-millisecond part (storage precision, see
/// `storage::columns::time`).
fn floor_to_millis(at: DateTime<Utc>) -> DateTime<Utc> {
    let nanos = at.nanosecond();
    at.with_nanosecond(nanos - nanos % 1_000_000).unwrap_or(at)
}

/// The segment start of `connection_id`; `None` when no sample has been
/// written under it, so it has no model yet.
pub fn connection_segment_start(connection_id: Uuid) -> Option<DateTime<Utc>> {
    CONNECTION_SEGMENTS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&connection_id)
        .copied()
}

/// What one model is fitted for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelKey {
    pub mount_identity_key: String,
    pub sync_generation: u64,
    pub pier_side: PierSide,
    /// The mount's native frame and how its JNow is read; residuals are J2000.
    pub mount_frame: EqFrame,
    pub jnow_flavour: JNowFlavour,
    /// The connection's segment start ([`connection_segment_start`]). A
    /// sample exposed earlier, or with no exposure time, is another
    /// connection's, whatever its generation number.
    pub segment_start: DateTime<Utc>,
}

impl ModelKey {
    /// The key of `record`, taken under the connection whose segment starts
    /// at `segment_start`.
    pub fn of(record: &PointingSampleRecord, segment_start: DateTime<Utc>) -> Self {
        Self {
            mount_identity_key: record.mount_identity_key.clone(),
            sync_generation: record.sync_generation,
            pier_side: record.pier_side,
            mount_frame: record.mount_frame,
            jnow_flavour: record.jnow_flavour,
            segment_start,
        }
    }

    fn admits(&self, record: &PointingSampleRecord) -> bool {
        record.mount_identity_key == self.mount_identity_key
            && record.sync_generation == self.sync_generation
            && record.pier_side == self.pier_side
            && record.mount_frame == self.mount_frame
            && record.jnow_flavour == self.jnow_flavour
            && record
                .mid_exposure_utc
                .is_some_and(|mid| mid >= self.segment_start)
    }
}

/// One residual: the mount's reading and the solved truth, both J2000, at
/// the hour angle of the exposure.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PointingSample {
    pub truth: EqCoord,
    pub mount: EqCoord,
    /// Hour angle at mid-exposure, hours: LST minus the mount's native-frame
    /// RA, as H12 stores it (`pointing_samples.ha_h`).
    pub ha_h: f64,
    pub at: Option<DateTime<Utc>>,
}

impl PointingSample {
    /// A sample from a stored record; `None` without the mount's J2000
    /// position or an hour angle.
    pub fn from_record(record: &PointingSampleRecord) -> Option<Self> {
        let mount = EqCoord::new(
            record.mount_ra_j2000_h?,
            record.mount_dec_j2000_deg?,
            EqFrame::J2000,
        );
        let truth = EqCoord::new(
            record.solved_ra_j2000_h,
            record.solved_dec_j2000_deg,
            EqFrame::J2000,
        );
        let ha_h = record.ha_h?;
        let finite = [mount.ra_h, mount.dec_deg, truth.ra_h, truth.dec_deg, ha_h]
            .iter()
            .all(|v| v.is_finite());
        finite.then_some(Self {
            truth,
            mount,
            ha_h,
            at: record.mid_exposure_utc,
        })
    }

    /// Mount reading minus truth on the tangent plane at the truth, arcsec
    /// (xi east, eta north). `None` when the reading is on the far
    /// hemisphere.
    pub fn residual_arcsec(&self) -> Option<(f64, f64)> {
        gnomonic(self.truth, self.mount)
    }
}

/// Gnomonic projection of `point` on the plane tangent at `centre`, arcsec.
pub fn gnomonic(centre: EqCoord, point: EqCoord) -> Option<(f64, f64)> {
    let (a0, d0) = (
        (centre.ra_h * 15.0).to_radians(),
        centre.dec_deg.to_radians(),
    );
    let (a, d) = ((point.ra_h * 15.0).to_radians(), point.dec_deg.to_radians());
    let da = a - a0;
    let cos_c = d0.sin() * d.sin() + d0.cos() * d.cos() * da.cos();
    if cos_c <= 1.0e-9 {
        return None;
    }
    let xi = d.cos() * da.sin() / cos_c;
    let eta = (d0.cos() * d.sin() - d0.sin() * d.cos() * da.cos()) / cos_c;
    Some((xi * ARCSEC_PER_RAD, eta * ARCSEC_PER_RAD))
}

/// The sky point at tangent-plane offset `(xi, eta)` arcsec from `centre`:
/// the inverse of [`gnomonic`]. RA is wrapped into [0, 24) hours.
pub fn gnomonic_inverse(centre: EqCoord, xi_arcsec: f64, eta_arcsec: f64) -> EqCoord {
    let (xi, eta) = (xi_arcsec / ARCSEC_PER_RAD, eta_arcsec / ARCSEC_PER_RAD);
    let rho = xi.hypot(eta);
    if rho <= f64::EPSILON {
        return centre.normalized();
    }
    let (a0, d0) = (
        (centre.ra_h * 15.0).to_radians(),
        centre.dec_deg.to_radians(),
    );
    let c = rho.atan();
    let (sin_c, cos_c) = c.sin_cos();
    let dec = (cos_c * d0.sin() + eta * sin_c * d0.cos() / rho)
        .clamp(-1.0, 1.0)
        .asin();
    let ra = a0 + (xi * sin_c).atan2(rho * d0.cos() * cos_c - eta * d0.sin() * sin_c);
    EqCoord::new(
        EqCoord::wrap_ra_h(ra.to_degrees() / 15.0),
        dec.to_degrees(),
        centre.frame,
    )
}

/// Unit vector of (hour angle h, Dec deg): the fit's coordinates.
fn unit(ha_h: f64, dec_deg: f64) -> [f64; 3] {
    let (h, d) = ((ha_h * 15.0).to_radians(), dec_deg.to_radians());
    [d.cos() * h.cos(), d.cos() * h.sin(), d.sin()]
}

fn chord(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Great-circle angle for a chord of the unit sphere, degrees.
fn chord_to_deg(r: f64) -> f64 {
    (2.0 * (r / 2.0).clamp(-1.0, 1.0).asin()).to_degrees()
}

fn tps_phi(r: f64) -> f64 {
    if r <= f64::EPSILON {
        0.0
    } else {
        r * r * r.ln()
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Spline {
    weights_xi: Vec<f64>,
    weights_eta: Vec<f64>,
    affine_xi: [f64; 4],
    affine_eta: [f64; 4],
}

/// A correction to apply to a slew target.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Correction {
    /// Command this (J2000) so the sky truth is the target.
    pub commanded: EqCoord,
    /// The applied tangent-plane offset, arcsec, after the clamp.
    pub xi_arcsec: f64,
    pub eta_arcsec: f64,
    pub clamped: bool,
    pub samples: usize,
}

/// Why no correction is offered.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum CorrectionRefused {
    TooFewSamples { have: usize, need: usize },
    PierSideMismatch { model: PierSide, target: PierSide },
    NotJ2000,
    OutOfCoverage { nearest_deg: f64 },
}

/// A fitted model for one [`ModelKey`].
#[derive(Debug, Clone, PartialEq)]
pub struct PointingModel {
    key: ModelKey,
    samples: Vec<PointingSample>,
    residuals: Vec<(f64, f64)>,
    points: Vec<[f64; 3]>,
    spline: Option<Spline>,
}

impl PointingModel {
    /// Fit `samples` (already filtered to `key`). Outliers and samples whose
    /// residual cannot be projected are dropped.
    pub fn fit(key: ModelKey, samples: Vec<PointingSample>) -> Self {
        let mut kept = Vec::new();
        let mut residuals = Vec::new();
        let mut points = Vec::new();
        for sample in samples {
            let Some((xi, eta)) = sample.residual_arcsec() else {
                continue;
            };
            if xi.hypot(eta) > OUTLIER_ARCMIN * 60.0 {
                continue;
            }
            points.push(unit(sample.ha_h, sample.truth.dec_deg));
            residuals.push((xi, eta));
            kept.push(sample);
        }
        let spline = (kept.len() >= MIN_SPLINE_SAMPLES)
            .then(|| fit_spline(&points, &residuals))
            .flatten();
        Self {
            key,
            samples: kept,
            residuals,
            points,
            spline,
        }
    }

    /// Fit from stored records, keeping only those that match `key`.
    pub fn from_records(key: ModelKey, records: &[PointingSampleRecord]) -> Self {
        let samples = records
            .iter()
            .filter(|record| key.admits(record))
            .filter_map(PointingSample::from_record)
            .collect();
        Self::fit(key, samples)
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    pub fn is_spline(&self) -> bool {
        self.spline.is_some()
    }

    /// Predicted mount error at (hour angle, Dec), arcsec (xi, eta), before
    /// any clamp. `None` with no samples.
    pub fn predict_arcsec(&self, ha_h: f64, dec_deg: f64) -> Option<(f64, f64)> {
        if self.samples.is_empty() || !ha_h.is_finite() || !dec_deg.is_finite() {
            return None;
        }
        let p = unit(ha_h, dec_deg);
        Some(match &self.spline {
            Some(spline) => {
                let base = |a: &[f64; 4]| a[0] + a[1] * p[0] + a[2] * p[1] + a[3] * p[2];
                let mut xi = base(&spline.affine_xi);
                let mut eta = base(&spline.affine_eta);
                for (i, centre) in self.points.iter().enumerate() {
                    let phi = tps_phi(chord(&p, centre));
                    xi += spline.weights_xi[i] * phi;
                    eta += spline.weights_eta[i] * phi;
                }
                (xi, eta)
            }
            None => idw(&self.points, &self.residuals, &p),
        })
    }

    /// Angular distance from (hour angle, Dec) to the nearest sample, degrees.
    pub fn nearest_sample_deg(&self, ha_h: f64, dec_deg: f64) -> Option<f64> {
        let p = unit(ha_h, dec_deg);
        self.points
            .iter()
            .map(|c| chord_to_deg(chord(&p, c)))
            .min_by(|a, b| a.total_cmp(b))
    }

    /// The command for sky truth `target` (J2000) at hour angle `ha_h` on
    /// `pier_side`: the target moved by the predicted mount error on the
    /// tangent plane, clamped to [`MAX_CORRECTION_ARCMIN`]. `ha_h` follows
    /// the samples' convention: LST minus the target's RA in the mount's
    /// native frame.
    pub fn correct(
        &self,
        target: EqCoord,
        ha_h: f64,
        pier_side: PierSide,
    ) -> Result<Correction, CorrectionRefused> {
        if target.frame != EqFrame::J2000 {
            return Err(CorrectionRefused::NotJ2000);
        }
        if pier_side != self.key.pier_side {
            return Err(CorrectionRefused::PierSideMismatch {
                model: self.key.pier_side,
                target: pier_side,
            });
        }
        let have = self.sample_count();
        if have < MIN_CORRECTION_SAMPLES {
            return Err(CorrectionRefused::TooFewSamples {
                have,
                need: MIN_CORRECTION_SAMPLES,
            });
        }
        let nearest = self
            .nearest_sample_deg(ha_h, target.dec_deg)
            .unwrap_or(180.0);
        if nearest > MAX_EXTRAPOLATION_DEG {
            return Err(CorrectionRefused::OutOfCoverage {
                nearest_deg: nearest,
            });
        }
        let (mut xi, mut eta) =
            self.predict_arcsec(ha_h, target.dec_deg)
                .ok_or(CorrectionRefused::TooFewSamples {
                    have,
                    need: MIN_CORRECTION_SAMPLES,
                })?;
        let limit = MAX_CORRECTION_ARCMIN * 60.0;
        let magnitude = xi.hypot(eta);
        let clamped = magnitude > limit;
        if clamped {
            xi *= limit / magnitude;
            eta *= limit / magnitude;
        }
        Ok(Correction {
            commanded: gnomonic_inverse(target, xi, eta),
            xi_arcsec: xi,
            eta_arcsec: eta,
            clamped,
            samples: have,
        })
    }

    /// Leave-one-out RMS of the residual prediction, arcmin; `None` below
    /// [`MIN_CORRECTION_SAMPLES`].
    pub fn loo_rms_arcmin(&self) -> Option<f64> {
        let n = self.sample_count();
        if n < MIN_CORRECTION_SAMPLES {
            return None;
        }
        let mut sum = 0.0;
        for i in 0..n {
            let mut samples = self.samples.clone();
            let held = samples.remove(i);
            let model = Self::fit(self.key.clone(), samples);
            let (pxi, peta) = model.predict_arcsec(held.ha_h, held.truth.dec_deg)?;
            let (xi, eta) = self.residuals[i];
            sum += (xi - pxi).powi(2) + (eta - peta).powi(2);
        }
        Some((sum / n as f64).sqrt() / 60.0)
    }

    /// `pointing_model_snapshots.model_data`: the key (identity, generation,
    /// pier side, frame and epoch), the samples the fit is rebuilt from, and
    /// the clamps.
    pub fn snapshot_json(&self) -> serde_json::Value {
        json!({
            "model": "tps_sphere_v1",
            "interpolant": if self.is_spline() { "thin_plate_spline" } else { "inverse_distance" },
            "key": self.key,
            "residual_frame": "j2000",
            "residual": "mount_minus_truth_tangent_plane_arcsec",
            "coordinates": "hour_angle_h_dec_deg",
            "hour_angle": "lst_minus_native_frame_ra",
            "dec": "j2000",
            "segment": "mount_connection",
            "max_correction_arcmin": MAX_CORRECTION_ARCMIN,
            "outlier_arcmin": OUTLIER_ARCMIN,
            "max_extrapolation_deg": MAX_EXTRAPOLATION_DEG,
            "samples": self.samples,
        })
    }
}

fn idw(points: &[[f64; 3]], residuals: &[(f64, f64)], p: &[f64; 3]) -> (f64, f64) {
    let (mut xi, mut eta, mut total) = (0.0, 0.0, 0.0);
    for (centre, (rx, ry)) in points.iter().zip(residuals) {
        let r = chord(p, centre);
        let weight = 1.0 / (r * r + 1.0e-8);
        xi += weight * rx;
        eta += weight * ry;
        total += weight;
    }
    if total <= f64::EPSILON {
        (0.0, 0.0)
    } else {
        (xi / total, eta / total)
    }
}

fn fit_spline(points: &[[f64; 3]], residuals: &[(f64, f64)]) -> Option<Spline> {
    let n = points.len();
    let size = n + 4;
    let mut matrix = vec![vec![0.0_f64; size]; size];
    for i in 0..n {
        for j in 0..n {
            matrix[i][j] = tps_phi(chord(&points[i], &points[j]));
        }
        matrix[i][i] += SMOOTHING;
        let p = points[i];
        let row = [1.0, p[0], p[1], p[2]];
        for (k, value) in row.iter().enumerate() {
            matrix[i][n + k] = *value;
            matrix[n + k][i] = *value;
        }
    }
    let solve_axis = |pick: fn(&(f64, f64)) -> f64| {
        let mut rhs = vec![0.0; size];
        for (i, residual) in residuals.iter().enumerate() {
            rhs[i] = pick(residual);
        }
        solve_linear(&matrix, &rhs)
    };
    let xi = solve_axis(|r| r.0)?;
    let eta = solve_axis(|r| r.1)?;
    if xi.iter().chain(eta.iter()).any(|v| !v.is_finite()) {
        return None;
    }
    let affine = |v: &[f64]| [v[n], v[n + 1], v[n + 2], v[n + 3]];
    Some(Spline {
        weights_xi: xi[..n].to_vec(),
        weights_eta: eta[..n].to_vec(),
        affine_xi: affine(&xi),
        affine_eta: affine(&eta),
    })
}

/// Gaussian elimination with partial pivoting; `None` when singular.
fn solve_linear(matrix: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = rhs.len();
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    let mut a = matrix.to_vec();
    let mut b = rhs.to_vec();
    for col in 0..n {
        let pivot = (col..n).max_by(|x, y| a[*x][col].abs().total_cmp(&a[*y][col].abs()))?;
        if a[pivot][col].abs() <= 1.0e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let divisor = a[col][col];
        for value in &mut a[col][col..] {
            *value /= divisor;
        }
        b[col] /= divisor;
        let pivot_row = a[col][col..].to_vec();
        let pivot_rhs = b[col];
        for (row, (values, rhs)) in a.iter_mut().zip(b.iter_mut()).enumerate() {
            let factor = values[col];
            if row == col || factor == 0.0 {
                continue;
            }
            for (value, pivot) in values[col..].iter_mut().zip(&pivot_row) {
                *value -= factor * pivot;
            }
            *rhs -= factor * pivot_rhs;
        }
    }
    Some(b)
}

/// Refit the model the new `record` belongs to and write a
/// `pointing_model_snapshots` row once it has [`MIN_CORRECTION_SAMPLES`]
/// samples. `segment_start` is the connection's
/// ([`note_connection_sample`]). `rig_id` is kept only when that `rigs` row
/// exists. Returns the snapshot written, if any.
pub async fn refit_snapshot(
    pool: &SqlitePool,
    rig_id: Option<&str>,
    record: &PointingSampleRecord,
    segment_start: DateTime<Utc>,
) -> Result<Option<pointing_model_repo::PointingModelSnapshot>, sqlx::Error> {
    let key = ModelKey::of(record, segment_start);
    let records = pointing_samples_repo::for_mount(
        pool,
        &key.mount_identity_key,
        Some(key.sync_generation),
        REFIT_LIMIT,
    )
    .await?;
    let model = PointingModel::from_records(key, &records);
    if model.sample_count() < MIN_CORRECTION_SAMPLES {
        return Ok(None);
    }
    let rig_id = match rig_id {
        Some(id) => {
            sqlx::query_scalar::<_, String>("SELECT id FROM rigs WHERE id = ?")
                .bind(id)
                .fetch_optional(pool)
                .await?
        }
        None => None,
    };
    let snapshot = pointing_model_repo::PointingModelSnapshot {
        id: uuid::Uuid::new_v4().to_string(),
        rig_id,
        sample_count: u32::try_from(model.sample_count()).unwrap_or(u32::MAX),
        rms_error_arcmin: model.loo_rms_arcmin(),
        model_data: model.snapshot_json(),
        created_at: Utc::now(),
    };
    pointing_model_repo::insert(pool, &snapshot).await?;
    Ok(Some(snapshot))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::events::PointingPurpose;
    use crate::equipment::types::MountSyncMode;

    fn key(side: PierSide) -> ModelKey {
        ModelKey {
            mount_identity_key: "alpaca:esp32-mount".into(),
            sync_generation: 2,
            pier_side: side,
            mount_frame: EqFrame::JNowApparent,
            jnow_flavour: JNowFlavour::Apparent,
            segment_start: Utc::now() - chrono::Duration::hours(1),
        }
    }

    fn j2000(ra_h: f64, dec_deg: f64) -> EqCoord {
        EqCoord::new(ra_h, dec_deg, EqFrame::J2000)
    }

    /// A sample whose mount reading is the truth moved by (xi, eta) arcsec.
    fn sample(ra_h: f64, ha_h: f64, dec_deg: f64, xi: f64, eta: f64) -> PointingSample {
        let truth = j2000(ra_h, dec_deg);
        PointingSample {
            truth,
            mount: gnomonic_inverse(truth, xi, eta),
            ha_h,
            at: None,
        }
    }

    #[test]
    fn the_tangent_plane_round_trips_across_the_ra_wrap_and_near_the_pole() {
        for centre in [j2000(23.99, 10.0), j2000(0.01, -30.0), j2000(6.0, 89.5)] {
            let moved = gnomonic_inverse(centre, 120.0, -45.0);
            let (xi, eta) = gnomonic(centre, moved).unwrap();
            assert!((xi - 120.0).abs() < 1e-6 && (eta + 45.0).abs() < 1e-6);
            assert!((0.0..24.0).contains(&moved.ra_h));
        }
        // 60" east of RA 23h59m59s lands just after 0h, not near 24h.
        let east = gnomonic_inverse(j2000(23.9999, 0.0), 60.0, 0.0);
        assert!(east.ra_h < 0.01, "{east:?}");
        // Near the pole a raw RA difference is huge; the projection is not.
        let a = j2000(0.0, 89.9);
        let b = j2000(12.0, 89.9);
        let (xi, eta) = gnomonic(a, b).unwrap();
        assert!((xi.hypot(eta) - 720.0).abs() < 1.0, "{xi} {eta}");
    }

    #[test]
    fn a_constant_mount_error_is_learned_and_corrected_on_the_tangent_plane() {
        let samples: Vec<_> = (0..8)
            .map(|i| {
                let ha = -3.0 + i as f64 * 0.8;
                sample(
                    (10.0 + ha).rem_euclid(24.0),
                    ha,
                    20.0 + i as f64,
                    300.0,
                    -120.0,
                )
            })
            .collect();
        let model = PointingModel::fit(key(PierSide::East), samples);
        assert!(model.is_spline());
        let (xi, eta) = model.predict_arcsec(0.3, 24.0).unwrap();
        assert!(
            (xi - 300.0).abs() < 2.0 && (eta + 120.0).abs() < 2.0,
            "{xi} {eta}"
        );
        let target = j2000(10.3, 24.0);
        let correction = model.correct(target, 0.3, PierSide::East).unwrap();
        assert!(!correction.clamped);
        // Commanding the corrected position makes the mount read target + error.
        let (cx, cy) = gnomonic(target, correction.commanded).unwrap();
        assert!((cx - 300.0).abs() < 2.0 && (cy + 120.0).abs() < 2.0);
        assert!(model.loo_rms_arcmin().unwrap() < 0.1);
    }

    #[test]
    fn the_hour_angle_wraps_so_samples_either_side_of_12h_are_neighbours() {
        let samples = vec![
            sample(1.0, 11.9, 0.0, 60.0, 0.0),
            sample(1.2, -11.9, 0.0, 60.0, 0.0),
            sample(1.1, 11.95, 1.0, 60.0, 0.0),
        ];
        let model = PointingModel::fit(key(PierSide::West), samples);
        assert!(!model.is_spline());
        assert!(model.nearest_sample_deg(-11.95, 0.5).unwrap() < 2.0);
        let (xi, _) = model.predict_arcsec(-11.95, 0.5).unwrap();
        assert!((xi - 60.0).abs() < 1e-6);
    }

    #[test]
    fn corrections_are_clamped_and_refused_outside_their_evidence() {
        let samples: Vec<_> = (0..4)
            .map(|i| sample(5.0 + i as f64 * 0.1, i as f64 * 0.1, 10.0, 3000.0, 0.0))
            .collect();
        let model = PointingModel::fit(key(PierSide::East), samples);
        let correction = model
            .correct(j2000(5.1, 10.0), 0.1, PierSide::East)
            .unwrap();
        assert!(correction.clamped);
        assert!((correction.xi_arcsec.hypot(correction.eta_arcsec) - 1800.0).abs() < 1e-6);

        assert!(matches!(
            model.correct(j2000(5.1, 10.0), 0.1, PierSide::West),
            Err(CorrectionRefused::PierSideMismatch { .. })
        ));
        assert!(matches!(
            model.correct(
                EqCoord::new(5.1, 10.0, EqFrame::JNowApparent),
                0.1,
                PierSide::East
            ),
            Err(CorrectionRefused::NotJ2000)
        ));
        assert!(matches!(
            model.correct(j2000(17.0, -60.0), 6.0, PierSide::East),
            Err(CorrectionRefused::OutOfCoverage { .. })
        ));
        let thin = PointingModel::fit(key(PierSide::East), vec![sample(5.0, 0.0, 10.0, 1.0, 1.0)]);
        assert!(matches!(
            thin.correct(j2000(5.0, 10.0), 0.0, PierSide::East),
            Err(CorrectionRefused::TooFewSamples { have: 1, need: 3 })
        ));
    }

    #[test]
    fn outliers_never_enter_the_fit() {
        let mut samples: Vec<_> = (0..3)
            .map(|i| sample(5.0, i as f64 * 0.2, 10.0, 30.0, 0.0))
            .collect();
        samples.push(sample(5.0, 0.3, 10.0, OUTLIER_ARCMIN * 60.0 + 600.0, 0.0));
        let model = PointingModel::fit(key(PierSide::East), samples);
        assert_eq!(model.sample_count(), 3);
    }

    fn record(generation: u64, side: PierSide, ra_h: f64, ha_h: f64) -> PointingSampleRecord {
        let truth = j2000(ra_h, 15.0);
        let mount = gnomonic_inverse(truth, 90.0, 30.0);
        PointingSampleRecord {
            id: uuid::Uuid::new_v4(),
            plate_solve_event_id: None,
            frame_id: None,
            mount_identity_key: "alpaca:esp32-mount".into(),
            mount_frame: EqFrame::JNowApparent,
            jnow_flavour: JNowFlavour::Apparent,
            does_refraction: Some(false),
            mount_ra_native_h: Some(mount.ra_h + 0.01),
            mount_dec_native_deg: Some(mount.dec_deg),
            mount_ra_j2000_h: Some(mount.ra_h),
            mount_dec_j2000_deg: Some(mount.dec_deg),
            solved_ra_j2000_h: truth.ra_h,
            solved_dec_j2000_deg: truth.dec_deg,
            err_ra_arcsec: None,
            err_dec_arcsec: None,
            err_total_arcsec: None,
            pier_side: side,
            ha_h: Some(ha_h),
            lst_h: None,
            alt_deg: None,
            az_deg: None,
            sync_generation: generation,
            sync_mode: MountSyncMode::Offset,
            coordinate_precision_arcsec: None,
            temperature_c: None,
            mid_exposure_utc: Some(Utc::now()),
            purpose: PointingPurpose::Center,
        }
    }

    #[test]
    fn a_model_never_mixes_generations_pier_sides_or_jnow_flavours() {
        let mut records = vec![
            record(2, PierSide::East, 5.0, 0.1),
            record(2, PierSide::East, 5.2, 0.3),
            record(3, PierSide::East, 5.4, 0.5),
            record(2, PierSide::West, 5.6, 0.7),
        ];
        let mut mean = record(2, PierSide::East, 5.8, 0.9);
        mean.jnow_flavour = JNowFlavour::Mean;
        records.push(mean);
        // Generation 2 of an earlier connection (its counter restarted): a
        // different mount model under the same number.
        let mut earlier = record(2, PierSide::East, 6.0, 1.1);
        earlier.mid_exposure_utc = Some(Utc::now() - chrono::Duration::hours(3));
        records.push(earlier);
        let mut untimed = record(2, PierSide::East, 6.2, 1.3);
        untimed.mid_exposure_utc = None;
        records.push(untimed);
        let model = PointingModel::from_records(key(PierSide::East), &records);
        assert_eq!(model.sample_count(), 2);
    }

    #[test]
    fn a_connection_segment_starts_at_its_earliest_sample() {
        let connection = Uuid::new_v4();
        assert_eq!(connection_segment_start(connection), None);
        // A mid-exposure with sub-millisecond digits, as a live clock gives.
        let exact = Utc::now()
            .with_nanosecond(123_456_789)
            .expect("valid nanosecond");
        let t0 = note_connection_sample(connection, exact);
        assert_eq!(t0, exact.with_nanosecond(123_000_000).unwrap());
        // The stored copy of that sample (millisecond text) is in the
        // segment it started, not before it.
        let stored = crate::storage::columns::parse_time(&crate::storage::columns::time(&exact))
            .expect("stored time parses");
        assert!(stored >= t0);
        // An older frame of the same connection solved later moves it back.
        let earlier = t0 - chrono::Duration::minutes(5);
        assert_eq!(note_connection_sample(connection, earlier), earlier);
        assert_eq!(
            note_connection_sample(connection, t0 + chrono::Duration::minutes(5)),
            earlier
        );
        assert_eq!(connection_segment_start(connection), Some(earlier));
        assert_eq!(connection_segment_start(Uuid::new_v4()), None);
    }

    #[tokio::test]
    async fn each_refit_writes_a_snapshot_with_its_key_and_epoch() {
        let pool = crate::storage::test_pool().await;
        // Generation 2 of an earlier connection, stored first: never fitted.
        let mut earlier = record(2, PierSide::East, 4.0, -0.5);
        earlier.mid_exposure_utc = Some(Utc::now() - chrono::Duration::hours(6));
        pointing_samples_repo::upsert(&pool, &earlier)
            .await
            .unwrap();
        let segment_start = Utc::now() - chrono::Duration::seconds(1);
        let mut last = None;
        for (i, ha) in [0.1, 0.4, 0.8].iter().enumerate() {
            let sample = record(2, PierSide::East, 5.0 + i as f64 * 0.3, *ha);
            pointing_samples_repo::upsert(&pool, &sample).await.unwrap();
            last = refit_snapshot(&pool, Some("no-such-rig"), &sample, segment_start)
                .await
                .unwrap();
            if i < 2 {
                assert!(last.is_none(), "too few samples for a model yet");
            }
        }
        let snapshot = last.unwrap();
        assert_eq!(snapshot.sample_count, 3);
        assert_eq!(snapshot.rig_id, None);
        assert_eq!(snapshot.model_data["key"]["pier_side"], json!("east"));
        assert_eq!(snapshot.model_data["key"]["sync_generation"], json!(2));
        assert_eq!(
            snapshot.model_data["key"]["segment_start"],
            serde_json::to_value(segment_start).unwrap()
        );
        assert_eq!(
            snapshot.model_data["key"]["mount_frame"],
            json!("jnow_apparent")
        );
        assert_eq!(snapshot.model_data["residual_frame"], json!("j2000"));
        let stored = pointing_model_repo::latest(&pool, None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.id, snapshot.id);
    }
}

//! Independent astronomy math for the equipment layer (WP1A-3, spec §5.3).
//!
//! Every routine here is a fresh implementation from published formula
//! definitions: the IAU 2006 precession and IAU 2000B nutation as given in
//! the IERS Conventions (2010, ch. 5) and McCarthy & Luzum (2003), the
//! Astronomical Almanac, and Meeus, *Astronomical Algorithms* (2nd ed.).
//! **No SOFA code is ported** (§6). ERFA (BSD) and the books were used only
//! to produce the numeric test vectors in the unit tests.
//!
//! | Module | Provides | v6 consumers (§7.2) |
//! |---|---|---|
//! | [`time`] | Julian dates, leap seconds and ΔT (UTC→TT), ERA, GMST/GAST/LST | every geometry consumer; R2 "LST derived" |
//! | [`frames`] | J2000 ↔ JNow (mean and apparent), the production [`FrameConverter`] | D10, M-3, M-4 pointing samples |
//! | [`meridian`] | hour angle, alt/az, meridian timing, rise/transit/set, visibility | mount limits (H13), M-3, M-4, flip (M-l), FOR, FW-3, planning |
//! | [`refraction`] | Bennett/Saemundsson refraction, Kasten-Young airmass | FITS `AIRMASS` (H1), BGE, FW-1, Model 7, M-3 |
//! | [`ephem`] | low-precision Sun and Moon, Moon phase, twilight, Moon separation, sky context | COC (H14), BGE, FW-1, FW-2, Model 7, planning |
//! | [`horizon`] | the `[mount.safety] horizon_profile_path` loader | mount limits (M-k), FW-3, planning |
//!
//! **Conventions.**
//!
//! - Right ascension in hours, every other angle in degrees at the API;
//!   radians inside.
//! - Longitude east-positive (the [`Site`](super::types::Site) contract and
//!   ASCOM `SiteLongitude`). Azimuth from north through east.
//! - Inputs are UTC `DateTime`s. TT comes from the leap-second table, UT1 is
//!   taken as UTC unless a DUT1 is supplied ([`time::TimeScales`]); |DUT1| <
//!   0.9 s by definition of UTC. Polar motion (< 0.5″) is neglected.
//! - "J2000" means the ICRS/GCRS directions a plate solver or catalogue
//!   reports; the precession includes the frame bias.
//! - An answer that cannot be computed is `None` or an explicit
//!   [`EqFrame::Unknown`](super::types::EqFrame::Unknown), never a guessed
//!   value (D5).
//!
//! All functions are pure and synchronous: they take the time as an
//! argument and never read a clock, so they are deterministic in tests and
//! safe to call from any thread.

pub mod ephem;
pub mod frames;
pub mod horizon;
pub mod meridian;
pub mod refraction;
pub mod time;

use std::f64::consts::{PI, TAU};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::traits::FrameConverter;

/// The production [`FrameConverter`] as a shareable trait object (D10).
pub fn frame_converter() -> Arc<dyn FrameConverter> {
    Arc::new(frames::AstroFrameConverter)
}

/// One crossing of a threshold found by a time search.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Crossing {
    pub at: DateTime<Utc>,
    /// True when the value rises through the threshold.
    pub rising: bool,
}

// ---------------------------------------------------------------------------
// Shared numeric helpers (private to the astro tree)
// ---------------------------------------------------------------------------

/// A Cartesian 3-vector.
pub type Vec3 = [f64; 3];
/// A 3×3 rotation matrix, row-major; `m · v` rotates a column vector.
pub type Mat3 = [[f64; 3]; 3];

const ARCSEC_TO_RAD: f64 = PI / 648_000.0;
const HOURS_TO_RAD: f64 = PI / 12.0;

const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Angle in `[0, 360)` degrees.
fn norm_deg(deg: f64) -> f64 {
    let wrapped = deg.rem_euclid(360.0);
    if wrapped >= 360.0 {
        0.0
    } else {
        wrapped
    }
}

/// Angle in `[0, 2π)` radians.
fn norm_rad(rad: f64) -> f64 {
    let wrapped = rad.rem_euclid(TAU);
    if wrapped >= TAU {
        0.0
    } else {
        wrapped
    }
}

/// Hours wrapped into `[-12, 12)`.
fn wrap_hours_pm12(hours: f64) -> f64 {
    let wrapped = (hours + 12.0).rem_euclid(24.0) - 12.0;
    if wrapped >= 12.0 {
        -12.0
    } else {
        wrapped
    }
}

/// Unit vector for a right ascension and declination (radians).
fn radec_to_vec(ra_rad: f64, dec_rad: f64) -> Vec3 {
    let (sd, cd) = dec_rad.sin_cos();
    let (sa, ca) = ra_rad.sin_cos();
    [cd * ca, cd * sa, sd]
}

/// Right ascension in `[0, 2π)` and declination (radians) of a vector.
fn vec_to_radec(v: Vec3) -> (f64, f64) {
    let xy = v[0].hypot(v[1]);
    let ra = if xy == 0.0 {
        0.0
    } else {
        norm_rad(v[1].atan2(v[0]))
    };
    (ra, v[2].atan2(xy))
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn scale(a: Vec3, k: f64) -> Vec3 {
    [a[0] * k, a[1] * k, a[2] * k]
}

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn normalize(a: Vec3) -> Vec3 {
    let n = dot(a, a).sqrt();
    if n == 0.0 {
        a
    } else {
        scale(a, 1.0 / n)
    }
}

fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn mat_vec(m: &Mat3, v: Vec3) -> Vec3 {
    std::array::from_fn(|i| dot(m[i], v))
}

/// `mᵀ · v` (the inverse of a rotation matrix applied to `v`).
fn mat_t_vec(m: &Mat3, v: Vec3) -> Vec3 {
    std::array::from_fn(|i| m[0][i] * v[0] + m[1][i] * v[1] + m[2][i] * v[2])
}

/// `R1(angle) · m`: rotation of the coordinate frame about the x axis.
fn rot_x(angle: f64, m: &Mat3) -> Mat3 {
    let (s, c) = angle.sin_cos();
    let r = [[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]];
    mat_mul(&r, m)
}

/// `R3(angle) · m`: rotation of the coordinate frame about the z axis.
fn rot_z(angle: f64, m: &Mat3) -> Mat3 {
    let (s, c) = angle.sin_cos();
    let r = [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]];
    mat_mul(&r, m)
}

/// Finds every crossing of `threshold` by `f` in `[from, to]`.
///
/// `f` is sampled every `step` and each sign change is refined by bisection
/// to better than one second. A sample where `f` is `None` (not computable)
/// breaks the bracket, so no crossing is reported across it. Two crossings
/// closer together than `step` (a value grazing the threshold) can be
/// missed; callers choose `step` for the rate of change they expect.
fn find_crossings<F>(
    f: F,
    threshold: f64,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    step: chrono::Duration,
) -> Vec<Crossing>
where
    F: Fn(DateTime<Utc>) -> Option<f64>,
{
    let mut out = Vec::new();
    if to <= from || step <= chrono::Duration::zero() {
        return out;
    }
    let mut prev_t = from;
    let mut prev = f(from).map(|v| v - threshold);
    loop {
        let next_t = (prev_t + step).min(to);
        let next = f(next_t).map(|v| v - threshold);
        if let (Some(a), Some(b)) = (prev, next) {
            if (a < 0.0) != (b < 0.0) {
                let rising = b >= 0.0;
                let (mut lo, mut hi) = (prev_t, next_t);
                let mut lo_below = a < 0.0;
                while hi - lo > chrono::Duration::milliseconds(500) {
                    let mid = lo + (hi - lo) / 2;
                    match f(mid).map(|v| v - threshold) {
                        Some(m) if (m < 0.0) == lo_below => {
                            lo = mid;
                            lo_below = m < 0.0;
                        }
                        Some(_) => hi = mid,
                        None => break,
                    }
                }
                out.push(Crossing {
                    at: lo + (hi - lo) / 2,
                    rising,
                });
            }
        }
        if next_t >= to {
            break;
        }
        prev_t = next_t;
        prev = next;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::types::{EqCoord, EqFrame};
    use chrono::TimeZone;

    #[test]
    fn angle_wrapping() {
        assert_eq!(norm_deg(-30.0), 330.0);
        assert_eq!(norm_deg(720.0), 0.0);
        assert!((norm_rad(-0.5) - (TAU - 0.5)).abs() < 1e-15);
        assert_eq!(wrap_hours_pm12(13.0), -11.0);
        assert_eq!(wrap_hours_pm12(-12.0), -12.0);
        assert_eq!(wrap_hours_pm12(12.0), -12.0);
        assert!((wrap_hours_pm12(23.5) + 0.5).abs() < 1e-12);
    }

    #[test]
    fn vectors_round_trip_and_rotations_are_orthonormal() {
        for (ra, dec) in [(0.0, 0.0), (1.0, 0.5), (6.0, -1.2), (3.0, 1.55)] {
            let (r, d) = vec_to_radec(radec_to_vec(ra, dec));
            assert!((r - norm_rad(ra)).abs() < 1e-12 && (d - dec).abs() < 1e-12);
        }
        let m = rot_z(0.3, &rot_x(-0.4, &rot_z(1.1, &IDENTITY)));
        let mt_m: Mat3 = std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..3).map(|k| m[k][i] * m[k][j]).sum())
        });
        for (i, row) in mt_m.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((value - expected).abs() < 1e-14);
            }
        }
        let v = [0.2, -0.5, 0.84];
        let back = mat_t_vec(&m, mat_vec(&m, v));
        assert!(sub(back, v).iter().all(|c| c.abs() < 1e-15));
    }

    #[test]
    fn crossing_finder_brackets_and_refines() {
        let from = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let to = from + chrono::Duration::hours(24);
        // A sine with a 24 h period crossing 0.5 twice.
        let f = |t: DateTime<Utc>| {
            let h = (t - from).num_milliseconds() as f64 / 3_600_000.0;
            Some((h / 24.0 * TAU).sin())
        };
        let found = find_crossings(f, 0.5, from, to, chrono::Duration::minutes(10));
        assert_eq!(found.len(), 2);
        assert!(found[0].rising && !found[1].rising);
        // sin(x) = 0.5 at x = π/6 and 5π/6, i.e. 2 h and 10 h.
        let hours = |c: &Crossing| (c.at - from).num_milliseconds() as f64 / 3_600_000.0;
        assert!((hours(&found[0]) - 2.0).abs() < 1.0 / 3600.0);
        assert!((hours(&found[1]) - 10.0).abs() < 1.0 / 3600.0);
        // An uncomputable stretch never yields a crossing.
        let none = find_crossings(|_| None, 0.0, from, to, chrono::Duration::minutes(10));
        assert!(none.is_empty());
        assert!(find_crossings(f, 0.5, to, from, chrono::Duration::minutes(10)).is_empty());
    }

    #[test]
    fn frame_converter_is_a_shareable_trait_object() {
        let converter = frame_converter();
        let at = Utc.with_ymd_and_hms(2026, 10, 26, 0, 0, 0).unwrap();
        let c = EqCoord::new(5.5, 20.0, EqFrame::J2000);
        let out = converter.convert(c, EqFrame::JNowApparent, at, None);
        assert_eq!(out.frame, EqFrame::JNowApparent);
        // About 22' of precession since J2000 (D10).
        let back = EqCoord::new(out.ra_h, out.dec_deg, EqFrame::J2000);
        let moved = c.separation_arcmin(&back).unwrap();
        assert!((15.0..30.0).contains(&moved), "{moved}");
    }
}

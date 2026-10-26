//! Equatorial frames (WP1A-3; D10, §3.2 M-a): J2000 ↔ JNow, mean and
//! apparent, and the production [`FrameConverter`].
//!
//! | Frame | Meaning here |
//! |---|---|
//! | [`EqFrame::J2000`] | ICRS/GCRS directions (what ASTAP and catalogues report) |
//! | [`EqFrame::JNowMean`] | mean equator and equinox of date: frame bias and precession only |
//! | [`EqFrame::JNowApparent`] | true equator and equinox of date with annual aberration: the geocentric apparent place, i.e. ASCOM `equTopocentric` ("allowed for annual aberration, precession and nutation") |
//!
//! Models, from their published definitions (no SOFA code, §6):
//!
//! - **Precession and frame bias:** IAU 2006, the Fukushima-Williams angles
//!   γ̄, φ̄, ψ̄ and the obliquity εA (Hilton et al. 2006; IERS Conventions
//!   (2010) eq. 5.39-5.40). The bias-precession matrix is
//!   `R1(−εA)·R3(−ψ̄)·R1(φ̄)·R3(γ̄)`; nutation enters as `εA + Δε` and
//!   `ψ̄ + Δψ`.
//! - **Nutation:** IAU 2000B (McCarthy & Luzum 2003; IERS Conventions
//!   (2003) §5.5.2): the 77 largest luni-solar terms of IAU 2000A with
//!   linear fundamental arguments, plus the fixed offsets that stand in for
//!   the planetary terms. About 1 mas from IAU 2000A. The 2006 rescaling of
//!   the nutation amplitudes (about 10 µas) is not applied.
//! - **Annual aberration:** the special-relativistic transformation of the
//!   direction with the Earth's velocity from its Keplerian orbit (mean
//!   solar elements of Meeus ch. 25, κ = 20.49552″). Neglecting the
//!   Earth-Moon barycentre wobble and the Sun's barycentric motion costs
//!   about 0.02″.
//!
//! **Measured accuracy** (unit tests, 2000-2045): the nutation matches
//! ERFA `nut00b` below 1 µas and the mean and true places match ERFA to
//! 3 µas; the apparent place is within 0.03″ of ERFA `atci13`; Meeus
//! examples 21.b and 23.a are met within 0.5″; J2000 → JNow → J2000 round
//! trips close below 1e-9″.
//!
//! Not modelled (all below the coordinate precision mounts report):
//! gravitational light deflection by the Sun (a few mas away from the
//! Sun), diurnal aberration (≤ 0.32″, excluded by the ASCOM topocentric
//! definition), polar motion, proper motion and parallax (catalogue
//! concerns, not frame concerns).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::time::TimeScales;
use super::{
    add, dot, mat_mul, mat_t_vec, mat_vec, normalize, radec_to_vec, rot_x, rot_z, scale, sub,
    vec_to_radec, Mat3, Vec3, ARCSEC_TO_RAD, HOURS_TO_RAD, IDENTITY,
};
use crate::equipment::traits::FrameConverter;
use crate::equipment::types::{EqCoord, EqFrame, Site};

/// The constant of aberration κ (radians): the Earth's mean orbital speed
/// over c, for the Keplerian velocity below.
const ABERRATION_CONSTANT: f64 = 20.495_52 * ARCSEC_TO_RAD;

/// Fixed offsets for the omitted planetary nutation terms (IAU 2000B).
const DPSI_PLANETARY: f64 = -0.000_135 * ARCSEC_TO_RAD;
const DEPS_PLANETARY: f64 = 0.000_388 * ARCSEC_TO_RAD;

/// Nutation in longitude and obliquity, radians.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Nutation {
    pub dpsi_rad: f64,
    pub deps_rad: f64,
}

/// IAU 2006 Fukushima-Williams angles and mean obliquity, radians.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FukushimaWilliams {
    pub gamma_bar: f64,
    pub phi_bar: f64,
    pub psi_bar: f64,
    pub eps_a: f64,
}

/// Why a conversion could not be done.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[serde(rename_all = "snake_case")]
pub enum FrameError {
    /// The source or the target frame is [`EqFrame::Unknown`].
    #[error("the coordinate frame is unknown")]
    UnknownFrame,
    /// RA or Dec is NaN or infinite.
    #[error("the coordinate is not finite")]
    NonFinite,
}

// ---------------------------------------------------------------------------
// IAU 2000B nutation
// ---------------------------------------------------------------------------

/// One luni-solar term: multipliers of (l, l′, F, D, Ω), then the
/// longitude coefficients (sin, sin·t, cos) and the obliquity coefficients
/// (cos, cos·t, sin), in units of 0.1 µas.
type NutationTerm = ([i8; 5], [f64; 3], [f64; 3]);

/// The 77 luni-solar terms of IAU 2000B (IERS Conventions (2003) table 5.3a,
/// first 77 rows), largest first.
#[rustfmt::skip]
const NUTATION_2000B: [NutationTerm; 77] = [
    ([ 0, 0, 0, 0, 1], [-172_064_161.0, -174_666.0,  33_386.0], [92_052_331.0,  9_086.0,  15_377.0]),
    ([ 0, 0, 2,-2, 2], [ -13_170_906.0,   -1_675.0, -13_696.0], [ 5_730_336.0, -3_015.0,  -4_587.0]),
    ([ 0, 0, 2, 0, 2], [  -2_276_413.0,     -234.0,   2_796.0], [   978_459.0,   -485.0,   1_374.0]),
    ([ 0, 0, 0, 0, 2], [   2_074_554.0,      207.0,    -698.0], [  -897_492.0,    470.0,    -291.0]),
    ([ 0, 1, 0, 0, 0], [   1_475_877.0,   -3_633.0,  11_817.0], [    73_871.0,   -184.0,  -1_924.0]),
    ([ 0, 1, 2,-2, 2], [    -516_821.0,    1_226.0,    -524.0], [   224_386.0,   -677.0,    -174.0]),
    ([ 1, 0, 0, 0, 0], [     711_159.0,       73.0,    -872.0], [    -6_750.0,      0.0,     358.0]),
    ([ 0, 0, 2, 0, 1], [    -387_298.0,     -367.0,     380.0], [   200_728.0,     18.0,     318.0]),
    ([ 1, 0, 2, 0, 2], [    -301_461.0,      -36.0,     816.0], [   129_025.0,    -63.0,     367.0]),
    ([ 0,-1, 2,-2, 2], [     215_829.0,     -494.0,     111.0], [   -95_929.0,    299.0,     132.0]),
    ([ 0, 0, 2,-2, 1], [     128_227.0,      137.0,     181.0], [   -68_982.0,     -9.0,      39.0]),
    ([-1, 0, 2, 0, 2], [     123_457.0,       11.0,      19.0], [   -53_311.0,     32.0,      -4.0]),
    ([-1, 0, 0, 2, 0], [     156_994.0,       10.0,    -168.0], [    -1_235.0,      0.0,      82.0]),
    ([ 1, 0, 0, 0, 1], [      63_110.0,       63.0,      27.0], [   -33_228.0,      0.0,      -9.0]),
    ([-1, 0, 0, 0, 1], [     -57_976.0,      -63.0,    -189.0], [    31_429.0,      0.0,     -75.0]),
    ([-1, 0, 2, 2, 2], [     -59_641.0,      -11.0,     149.0], [    25_543.0,    -11.0,      66.0]),
    ([ 1, 0, 2, 0, 1], [     -51_613.0,      -42.0,     129.0], [    26_366.0,      0.0,      78.0]),
    ([-2, 0, 2, 0, 1], [      45_893.0,       50.0,      31.0], [   -24_236.0,    -10.0,      20.0]),
    ([ 0, 0, 0, 2, 0], [      63_384.0,       11.0,    -150.0], [    -1_220.0,      0.0,      29.0]),
    ([ 0, 0, 2, 2, 2], [     -38_571.0,       -1.0,     158.0], [    16_452.0,    -11.0,      68.0]),
    ([ 0,-2, 2,-2, 2], [      32_481.0,        0.0,       0.0], [   -13_870.0,      0.0,       0.0]),
    ([-2, 0, 0, 2, 0], [     -47_722.0,        0.0,     -18.0], [       477.0,      0.0,     -25.0]),
    ([ 2, 0, 2, 0, 2], [     -31_046.0,       -1.0,     131.0], [    13_238.0,    -11.0,      59.0]),
    ([ 1, 0, 2,-2, 2], [      28_593.0,        0.0,      -1.0], [   -12_338.0,     10.0,      -3.0]),
    ([-1, 0, 2, 0, 1], [      20_441.0,       21.0,      10.0], [   -10_758.0,      0.0,      -3.0]),
    ([ 2, 0, 0, 0, 0], [      29_243.0,        0.0,     -74.0], [      -609.0,      0.0,      13.0]),
    ([ 0, 0, 2, 0, 0], [      25_887.0,        0.0,     -66.0], [      -550.0,      0.0,      11.0]),
    ([ 0, 1, 0, 0, 1], [     -14_053.0,      -25.0,      79.0], [     8_551.0,     -2.0,     -45.0]),
    ([-1, 0, 0, 2, 1], [      15_164.0,       10.0,      11.0], [    -8_001.0,      0.0,      -1.0]),
    ([ 0, 2, 2,-2, 2], [     -15_794.0,       72.0,     -16.0], [     6_850.0,    -42.0,      -5.0]),
    ([ 0, 0,-2, 2, 0], [      21_783.0,        0.0,      13.0], [      -167.0,      0.0,      13.0]),
    ([ 1, 0, 0,-2, 1], [     -12_873.0,      -10.0,     -37.0], [     6_953.0,      0.0,     -14.0]),
    ([ 0,-1, 0, 0, 1], [     -12_654.0,       11.0,      63.0], [     6_415.0,      0.0,      26.0]),
    ([-1, 0, 2, 2, 1], [     -10_204.0,        0.0,      25.0], [     5_222.0,      0.0,      15.0]),
    ([ 0, 2, 0, 0, 0], [      16_707.0,      -85.0,     -10.0], [       168.0,     -1.0,      10.0]),
    ([ 1, 0, 2, 2, 2], [      -7_691.0,        0.0,      44.0], [     3_268.0,      0.0,      19.0]),
    ([-2, 0, 2, 0, 0], [     -11_024.0,        0.0,     -14.0], [       104.0,      0.0,       2.0]),
    ([ 0, 1, 2, 0, 2], [       7_566.0,      -21.0,     -11.0], [    -3_250.0,      0.0,      -5.0]),
    ([ 0, 0, 2, 2, 1], [      -6_637.0,      -11.0,      25.0], [     3_353.0,      0.0,      14.0]),
    ([ 0,-1, 2, 0, 2], [      -7_141.0,       21.0,       8.0], [     3_070.0,      0.0,       4.0]),
    ([ 0, 0, 0, 2, 1], [      -6_302.0,      -11.0,       2.0], [     3_272.0,      0.0,       4.0]),
    ([ 1, 0, 2,-2, 1], [       5_800.0,       10.0,       2.0], [    -3_045.0,      0.0,      -1.0]),
    ([ 2, 0, 2,-2, 2], [       6_443.0,        0.0,      -7.0], [    -2_768.0,      0.0,      -4.0]),
    ([-2, 0, 0, 2, 1], [      -5_774.0,      -11.0,     -15.0], [     3_041.0,      0.0,      -5.0]),
    ([ 2, 0, 2, 0, 1], [      -5_350.0,        0.0,      21.0], [     2_695.0,      0.0,      12.0]),
    ([ 0,-1, 2,-2, 1], [      -4_752.0,      -11.0,      -3.0], [     2_719.0,      0.0,      -3.0]),
    ([ 0, 0, 0,-2, 1], [      -4_940.0,      -11.0,     -21.0], [     2_720.0,      0.0,      -9.0]),
    ([-1,-1, 0, 2, 0], [       7_350.0,        0.0,      -8.0], [       -51.0,      0.0,       4.0]),
    ([ 2, 0, 0,-2, 1], [       4_065.0,        0.0,       6.0], [    -2_206.0,      0.0,       1.0]),
    ([ 1, 0, 0, 2, 0], [       6_579.0,        0.0,     -24.0], [      -199.0,      0.0,       2.0]),
    ([ 0, 1, 2,-2, 1], [       3_579.0,        0.0,       5.0], [    -1_900.0,      0.0,       1.0]),
    ([ 1,-1, 0, 0, 0], [       4_725.0,        0.0,      -6.0], [       -41.0,      0.0,       3.0]),
    ([-2, 0, 2, 0, 2], [      -3_075.0,        0.0,      -2.0], [     1_313.0,      0.0,      -1.0]),
    ([ 3, 0, 2, 0, 2], [      -2_904.0,        0.0,      15.0], [     1_233.0,      0.0,       7.0]),
    ([ 0,-1, 0, 2, 0], [       4_348.0,        0.0,     -10.0], [       -81.0,      0.0,       2.0]),
    ([ 1,-1, 2, 0, 2], [      -2_878.0,        0.0,       8.0], [     1_232.0,      0.0,       4.0]),
    ([ 0, 0, 0, 1, 0], [      -4_230.0,        0.0,       5.0], [       -20.0,      0.0,      -2.0]),
    ([-1,-1, 2, 2, 2], [      -2_819.0,        0.0,       7.0], [     1_207.0,      0.0,       3.0]),
    ([-1, 0, 2, 0, 0], [      -4_056.0,        0.0,       5.0], [        40.0,      0.0,      -2.0]),
    ([ 0,-1, 2, 2, 2], [      -2_647.0,        0.0,      11.0], [     1_129.0,      0.0,       5.0]),
    ([-2, 0, 0, 0, 1], [      -2_294.0,        0.0,     -10.0], [     1_266.0,      0.0,      -4.0]),
    ([ 1, 1, 2, 0, 2], [       2_481.0,        0.0,      -7.0], [    -1_062.0,      0.0,      -3.0]),
    ([ 2, 0, 0, 0, 1], [       2_179.0,        0.0,      -2.0], [    -1_129.0,      0.0,      -2.0]),
    ([-1, 1, 0, 1, 0], [       3_276.0,        0.0,       1.0], [        -9.0,      0.0,       0.0]),
    ([ 1, 1, 0, 0, 0], [      -3_389.0,        0.0,       5.0], [        35.0,      0.0,      -2.0]),
    ([ 1, 0, 2, 0, 0], [       3_339.0,        0.0,     -13.0], [      -107.0,      0.0,       1.0]),
    ([-1, 0, 2,-2, 1], [      -1_987.0,        0.0,      -6.0], [     1_073.0,      0.0,      -2.0]),
    ([ 1, 0, 0, 0, 2], [      -1_981.0,        0.0,       0.0], [       854.0,      0.0,       0.0]),
    ([-1, 0, 0, 1, 0], [       4_026.0,        0.0,    -353.0], [      -553.0,      0.0,    -139.0]),
    ([ 0, 0, 2, 1, 2], [       1_660.0,        0.0,      -5.0], [      -710.0,      0.0,      -2.0]),
    ([-1, 0, 2, 4, 2], [      -1_521.0,        0.0,       9.0], [       647.0,      0.0,       4.0]),
    ([-1, 1, 0, 1, 1], [       1_314.0,        0.0,       0.0], [      -700.0,      0.0,       0.0]),
    ([ 0,-2, 2,-2, 1], [      -1_283.0,        0.0,       0.0], [       672.0,      0.0,       0.0]),
    ([ 1, 0, 2, 2, 1], [      -1_331.0,        0.0,       8.0], [       663.0,      0.0,       4.0]),
    ([-2, 0, 2, 2, 2], [       1_383.0,        0.0,      -2.0], [      -594.0,      0.0,      -2.0]),
    ([-1, 0, 0, 0, 2], [       1_405.0,        0.0,       4.0], [      -610.0,      0.0,       2.0]),
    ([ 1, 1, 2,-2, 2], [       1_290.0,        0.0,       0.0], [      -556.0,      0.0,       0.0]),
];

/// IAU 2000B nutation at `t` Julian centuries of TT since J2000.
pub fn nutation_2000b(t: f64) -> Nutation {
    // Delaunay arguments, arcseconds, linear in t (the 2000B definition).
    let arg = |c0: f64, c1: f64| ((c0 + c1 * t) % 1_296_000.0) * ARCSEC_TO_RAD;
    let l = arg(485_868.249_036, 1_717_915_923.217_8);
    let lp = arg(1_287_104.793_05, 129_596_581.048_1);
    let f = arg(335_779.526_232, 1_739_527_262.847_8);
    let d = arg(1_072_260.703_69, 1_602_961_601.209_0);
    let om = arg(450_160.398_036, -6_962_890.543_1);

    // Sum the smallest terms first.
    let (mut dpsi, mut deps) = (0.0_f64, 0.0_f64);
    for (m, lon, obl) in NUTATION_2000B.iter().rev() {
        let a = f64::from(m[0]) * l
            + f64::from(m[1]) * lp
            + f64::from(m[2]) * f
            + f64::from(m[3]) * d
            + f64::from(m[4]) * om;
        let (s, c) = a.sin_cos();
        dpsi += (lon[0] + lon[1] * t) * s + lon[2] * c;
        deps += (obl[0] + obl[1] * t) * c + obl[2] * s;
    }
    // 0.1 µas → radians.
    let unit = 1e-7 * ARCSEC_TO_RAD;
    Nutation {
        dpsi_rad: dpsi * unit + DPSI_PLANETARY,
        deps_rad: deps * unit + DEPS_PLANETARY,
    }
}

// ---------------------------------------------------------------------------
// IAU 2006 precession
// ---------------------------------------------------------------------------

/// Mean obliquity of the ecliptic (IAU 2006), radians.
pub fn mean_obliquity_rad(t: f64) -> f64 {
    let arcsec = 84_381.406
        + t * (-46.836_769
            + t * (-0.000_183_1
                + t * (0.002_003_40 + t * (-0.000_000_576 + t * -0.000_000_043_4))));
    arcsec * ARCSEC_TO_RAD
}

/// The IAU 2006 Fukushima-Williams angles at `t` centuries of TT.
pub fn fukushima_williams(t: f64) -> FukushimaWilliams {
    let poly = |c: [f64; 6]| {
        (c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * (c[4] + t * c[5]))))) * ARCSEC_TO_RAD
    };
    FukushimaWilliams {
        gamma_bar: poly([
            -0.052_928,
            10.556_378,
            0.493_204_4,
            -0.000_312_38,
            -0.000_002_788,
            0.000_000_026_0,
        ]),
        phi_bar: poly([
            84_381.412_819,
            -46.811_016,
            0.051_126_8,
            0.000_532_89,
            -0.000_000_440,
            -0.000_000_017_6,
        ]),
        psi_bar: poly([
            -0.041_775,
            5_038.481_484,
            1.558_417_5,
            -0.000_185_22,
            -0.000_026_452,
            -0.000_000_014_8,
        ]),
        eps_a: mean_obliquity_rad(t),
    }
}

fn fw_matrix(gamma_bar: f64, phi_bar: f64, psi: f64, eps: f64) -> Mat3 {
    let m = rot_z(gamma_bar, &IDENTITY);
    let m = rot_x(phi_bar, &m);
    let m = rot_z(-psi, &m);
    rot_x(-eps, &m)
}

/// Frame bias and precession: GCRS (J2000) → mean equator and equinox of
/// date.
pub fn bias_precession_matrix(t: f64) -> Mat3 {
    let fw = fukushima_williams(t);
    fw_matrix(fw.gamma_bar, fw.phi_bar, fw.psi_bar, fw.eps_a)
}

/// Frame bias, precession and nutation: GCRS → true equator and equinox of
/// date.
pub fn bias_precession_nutation_matrix(t: f64, nutation: Nutation) -> Mat3 {
    let fw = fukushima_williams(t);
    fw_matrix(
        fw.gamma_bar,
        fw.phi_bar,
        fw.psi_bar + nutation.dpsi_rad,
        fw.eps_a + nutation.deps_rad,
    )
}

// ---------------------------------------------------------------------------
// Annual aberration
// ---------------------------------------------------------------------------

/// The Earth's orbital velocity over c, in mean-of-date equatorial axes,
/// from the Keplerian orbit with the mean solar elements of Meeus ch. 25.
fn earth_velocity_mean_of_date(t: f64) -> Vec3 {
    let l0 = 280.466_46 + t * (36_000.769_83 + t * 0.000_303_2);
    let m = (357.529_11 + t * (35_999.050_29 - t * 0.000_153_7)).to_radians();
    let e = 0.016_708_634 - t * (0.000_042_037 + t * 0.000_000_126_7);
    let c = (1.914_602 - t * (0.004_817 + t * 0.000_014)) * m.sin()
        + (0.019_993 - t * 0.000_101) * (2.0 * m).sin()
        + 0.000_289 * (3.0 * m).sin();
    // Geometric longitude of the Sun and the Earth's perihelion longitude.
    let sun = (l0 + c).to_radians();
    let perihelion = (102.937_35 + t * (1.719_46 + t * 0.000_46)).to_radians();
    let (ss, cs) = sun.sin_cos();
    let (sp, cp) = perihelion.sin_cos();
    // Ecliptic of date: v/c = κ·(sin☉ − e sin ϖ, −cos☉ + e cos ϖ, 0).
    let vx = ABERRATION_CONSTANT * (ss - e * sp);
    let vy = ABERRATION_CONSTANT * (-cs + e * cp);
    let (se, ce) = mean_obliquity_rad(t).sin_cos();
    [vx, vy * ce, vy * se]
}

/// The Earth's orbital velocity over c in GCRS (J2000) axes.
pub fn earth_velocity_gcrs(t: f64) -> Vec3 {
    mat_t_vec(&bias_precession_matrix(t), earth_velocity_mean_of_date(t))
}

/// Apparent direction seen by an observer moving with velocity `v` (units
/// of c) of a source in unit direction `p` (special-relativistic
/// aberration).
fn aberrate(p: Vec3, v: Vec3) -> Vec3 {
    let inv_gamma = (1.0 - dot(v, v)).sqrt();
    let pv = dot(p, v);
    let num = add(scale(p, inv_gamma), scale(v, 1.0 + pv / (1.0 + inv_gamma)));
    normalize(scale(num, 1.0 / (1.0 + pv)))
}

/// Inverse of [`aberrate`]: the boost by −v undoes the boost by v, and two
/// fixed-point steps remove the rounding left by it.
fn unaberrate(apparent: Vec3, v: Vec3) -> Vec3 {
    let mut p = aberrate(apparent, scale(v, -1.0));
    for _ in 0..2 {
        p = normalize(sub(p, sub(aberrate(p, v), apparent)));
    }
    p
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

/// Everything a conversion at one instant needs, computed once.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameContext {
    /// GCRS → mean of date.
    pub precession: Mat3,
    /// GCRS → true of date.
    pub precession_nutation: Mat3,
    /// Earth velocity over c, GCRS axes.
    pub velocity: Vec3,
    pub nutation: Nutation,
}

impl FrameContext {
    pub fn new(ts: &TimeScales) -> Self {
        let t = ts.tt_centuries();
        let nutation = nutation_2000b(t);
        let precession = bias_precession_matrix(t);
        Self {
            precession,
            precession_nutation: bias_precession_nutation_matrix(t, nutation),
            velocity: mat_t_vec(&precession, earth_velocity_mean_of_date(t)),
            nutation,
        }
    }

    pub fn at(utc: DateTime<Utc>) -> Self {
        Self::new(&TimeScales::from_utc(utc))
    }

    /// Unit vector in `frame` → GCRS unit vector.
    fn gcrs_of(&self, v: Vec3, frame: EqFrame) -> Option<Vec3> {
        match frame {
            EqFrame::J2000 => Some(v),
            EqFrame::JNowMean => Some(mat_t_vec(&self.precession, v)),
            EqFrame::JNowApparent => Some(unaberrate(
                mat_t_vec(&self.precession_nutation, v),
                self.velocity,
            )),
            EqFrame::Unknown => None,
        }
    }

    /// GCRS unit vector → unit vector in `frame`.
    fn expressed_in(&self, v: Vec3, frame: EqFrame) -> Option<Vec3> {
        match frame {
            EqFrame::J2000 => Some(v),
            EqFrame::JNowMean => Some(mat_vec(&self.precession, v)),
            EqFrame::JNowApparent => Some(mat_vec(
                &self.precession_nutation,
                aberrate(v, self.velocity),
            )),
            EqFrame::Unknown => None,
        }
    }

    /// Converts `c` into `to` at this context's instant.
    pub fn convert(&self, c: EqCoord, to: EqFrame) -> Result<EqCoord, FrameError> {
        if !(c.ra_h.is_finite() && c.dec_deg.is_finite()) {
            return Err(FrameError::NonFinite);
        }
        if c.frame == EqFrame::Unknown || to == EqFrame::Unknown {
            return Err(FrameError::UnknownFrame);
        }
        if c.frame == to {
            return Ok(c.normalized());
        }
        let v = radec_to_vec(c.ra_h * HOURS_TO_RAD, c.dec_deg.to_radians());
        let gcrs = self.gcrs_of(v, c.frame).ok_or(FrameError::UnknownFrame)?;
        let out = self
            .expressed_in(gcrs, to)
            .ok_or(FrameError::UnknownFrame)?;
        let (ra, dec) = vec_to_radec(out);
        Ok(EqCoord::new(ra / HOURS_TO_RAD, dec.to_degrees(), to).normalized())
    }

    /// The true-of-date matrix composed after the mean-of-date one, i.e.
    /// the nutation matrix alone (mean of date → true of date).
    pub fn nutation_matrix(&self) -> Mat3 {
        let p = self.precession;
        let pt: Mat3 = std::array::from_fn(|i| std::array::from_fn(|j| p[j][i]));
        mat_mul(&self.precession_nutation, &pt)
    }
}

/// Converts between equatorial frames, or says why it cannot.
pub fn convert(c: EqCoord, to: EqFrame, at: DateTime<Utc>) -> Result<EqCoord, FrameError> {
    FrameContext::at(at).convert(c, to)
}

/// The production [`FrameConverter`] (D10). Stateless and cheap to clone.
///
/// A conversion that cannot be done (an `Unknown` source or target frame,
/// or a non-finite coordinate) returns the input with
/// [`EqFrame::Unknown`]: it is never relabelled as converted, so
/// frame-sensitive features stay disabled (§3.2 M-a). The site is not used:
/// the apparent frame is geocentric, per the ASCOM topocentric definition.
#[derive(Debug, Clone, Copy, Default)]
pub struct AstroFrameConverter;

impl FrameConverter for AstroFrameConverter {
    fn convert(&self, c: EqCoord, to: EqFrame, at: DateTime<Utc>, _site: Option<&Site>) -> EqCoord {
        convert(c, to, at).unwrap_or(EqCoord {
            frame: EqFrame::Unknown,
            ..c
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::time::tests::utc;
    use super::*;

    const AS: f64 = ARCSEC_TO_RAD;

    fn sep_arcsec(a: EqCoord, b: EqCoord) -> f64 {
        let a = EqCoord {
            frame: b.frame,
            ..a
        };
        a.separation_arcmin(&b).unwrap() * 60.0
    }

    fn deg_of_hms(h: f64, m: f64, s: f64) -> f64 {
        (h + m / 60.0 + s / 3600.0) * 15.0
    }

    fn deg_of_dms(d: f64, m: f64, s: f64) -> f64 {
        d.signum() * (d.abs() + m / 60.0 + s / 3600.0)
    }

    #[test]
    fn nutation_matches_meeus_example_22a() {
        // 1987 April 10, 0h TD: Δψ = −3.788″, Δε = +9.443″ (IAU 1980; IAU
        // 2000B differs by a few hundredths of an arcsecond).
        let ts = TimeScales::from_jd_tt(2_446_895.5, 55.184);
        let n = nutation_2000b(ts.tt_centuries());
        assert!(
            (n.dpsi_rad / AS + 3.788).abs() < 0.03,
            "{}",
            n.dpsi_rad / AS
        );
        assert!(
            (n.deps_rad / AS - 9.443).abs() < 0.03,
            "{}",
            n.deps_rad / AS
        );
        // Mean obliquity 23°26′27.407″ (IAU 1980 value 23°26′27.407″).
        let eps0 = mean_obliquity_rad(ts.tt_centuries()).to_degrees();
        assert!(
            (eps0 - deg_of_dms(23.0, 26.0, 27.407)).abs() * 3600.0 < 0.1,
            "{eps0}"
        );
    }

    #[test]
    fn aberration_inverse_is_exact() {
        let v = earth_velocity_gcrs(0.27);
        for (ra, dec) in [(0.1, 0.2), (2.0, -1.4), (4.5, 1.5), (6.0, 0.0)] {
            let p = radec_to_vec(ra, dec);
            let back = unaberrate(aberrate(p, v), v);
            assert!(sub(back, p).iter().all(|c| c.abs() < 1e-15));
        }
        // |v| is the constant of aberration within the eccentricity.
        let speed = dot(v, v).sqrt() / AS;
        assert!((20.1..20.9).contains(&speed), "{speed}");
    }

    #[test]
    fn precession_matches_meeus_example_21b() {
        // θ Persei, J2000 α = 2h44m11.986s, δ = +49°13′42.48″, proper
        // motion +0.0342 s/yr, −0.083″/yr; mean place for 2028 Nov 13.19 TD
        // (JDE 2462088.69): α = 2h46m11.331s, δ = +49°20′54.54″.
        let years = (2_462_088.69 - 2_451_545.0) / 365.25;
        let ra0 = deg_of_hms(2.0, 44.0, 11.986 + 0.0342 * years) / 15.0;
        let dec0 = deg_of_dms(49.0, 13.0, 42.48 - 0.083 * years);
        let ctx = FrameContext::new(&TimeScales::from_jd_tt(2_462_088.69, 69.184));
        let mean = ctx
            .convert(EqCoord::new(ra0, dec0, EqFrame::J2000), EqFrame::JNowMean)
            .unwrap();
        let expected = EqCoord::new(
            deg_of_hms(2.0, 46.0, 11.331) / 15.0,
            deg_of_dms(49.0, 20.0, 54.54),
            EqFrame::JNowMean,
        );
        let err = sep_arcsec(mean, expected);
        assert!(err < 0.5, "precession error {err}\"");
    }

    #[test]
    fn apparent_place_matches_meeus_example_23a() {
        // The same star, apparent place: α = 2h46m14.390s, δ = +49°21′07.45″.
        let years = (2_462_088.69 - 2_451_545.0) / 365.25;
        let ra0 = deg_of_hms(2.0, 44.0, 11.986 + 0.0342 * years) / 15.0;
        let dec0 = deg_of_dms(49.0, 13.0, 42.48 - 0.083 * years);
        let ctx = FrameContext::new(&TimeScales::from_jd_tt(2_462_088.69, 69.184));
        let apparent = ctx
            .convert(
                EqCoord::new(ra0, dec0, EqFrame::J2000),
                EqFrame::JNowApparent,
            )
            .unwrap();
        let expected = EqCoord::new(
            deg_of_hms(2.0, 46.0, 14.390) / 15.0,
            deg_of_dms(49.0, 21.0, 7.45),
            EqFrame::JNowApparent,
        );
        let err = sep_arcsec(apparent, expected);
        assert!(err < 0.5, "apparent place error {err}\"");
    }

    #[test]
    fn j2000_jnow_round_trips_below_a_hundredth_of_an_arcsecond() {
        let instants = [
            utc(2026, 10, 26, 3, 15, 42.5),
            utc(2000, 1, 1, 12, 0, 0.0),
            utc(2045, 6, 30, 23, 59, 59.0),
        ];
        let mut worst: f64 = 0.0;
        for at in instants {
            let ctx = FrameContext::at(at);
            for ra in [0.0, 0.01, 5.5, 11.99, 12.0, 18.25, 23.999] {
                for dec in [-89.99, -60.0, -0.5, 0.0, 33.3, 75.0, 89.99] {
                    let c = EqCoord::new(ra, dec, EqFrame::J2000);
                    for flavour in [EqFrame::JNowMean, EqFrame::JNowApparent] {
                        let now = ctx.convert(c, flavour).unwrap();
                        let back = ctx.convert(now, EqFrame::J2000).unwrap();
                        worst = worst.max(sep_arcsec(back, c));
                        assert!((0.0..24.0).contains(&back.ra_h));
                    }
                }
            }
        }
        assert!(worst < 0.01, "worst round trip {worst}\"");
    }

    #[test]
    fn mean_and_apparent_of_date_differ_by_about_twenty_arcseconds() {
        // M-4 gap 1: the two JNow flavours are roughly 20″ apart.
        let ctx = FrameContext::at(utc(2026, 10, 26, 0, 0, 0.0));
        let c = EqCoord::new(5.5, 20.0, EqFrame::J2000);
        let mean = ctx.convert(c, EqFrame::JNowMean).unwrap();
        let apparent = ctx.convert(c, EqFrame::JNowApparent).unwrap();
        let d = sep_arcsec(mean, apparent);
        assert!((3.0..45.0).contains(&d), "{d}");
        // Converting between the two flavours directly agrees with going
        // through J2000.
        let direct = ctx.convert(mean, EqFrame::JNowApparent).unwrap();
        assert!(sep_arcsec(direct, apparent) < 1e-6);
    }

    #[test]
    fn converter_never_relabels_an_unknown_frame() {
        let at = utc(2026, 10, 26, 0, 0, 0.0);
        let unknown = EqCoord::new(1.0, 2.0, EqFrame::Unknown);
        let out = AstroFrameConverter.convert(unknown, EqFrame::J2000, at, None);
        assert_eq!(out, unknown);
        let known = EqCoord::new(1.0, 2.0, EqFrame::J2000);
        let out = AstroFrameConverter.convert(known, EqFrame::Unknown, at, None);
        assert_eq!(out.frame, EqFrame::Unknown);
        assert_eq!(
            convert(unknown, EqFrame::J2000, at),
            Err(FrameError::UnknownFrame)
        );
        let nan = EqCoord::new(f64::NAN, 2.0, EqFrame::J2000);
        assert_eq!(
            convert(nan, EqFrame::JNowMean, at),
            Err(FrameError::NonFinite)
        );
        // Same frame: a no-op apart from RA normalisation.
        let wrapped = EqCoord::new(24.5, 10.0, EqFrame::JNowMean);
        let same = AstroFrameConverter.convert(wrapped, EqFrame::JNowMean, at, None);
        assert!((same.ra_h - 0.5).abs() < 1e-12 && same.dec_deg == 10.0);
    }

    #[test]
    fn nutation_matrix_is_the_ratio_of_true_and_mean() {
        let ctx = FrameContext::at(utc(2026, 1, 1, 0, 0, 0.0));
        let n = ctx.nutation_matrix();
        let composed = mat_mul(&n, &ctx.precession);
        for (row, expected) in composed.iter().zip(ctx.precession_nutation.iter()) {
            for (value, want) in row.iter().zip(expected) {
                assert!((value - want).abs() < 1e-15);
            }
        }
    }

    // Reference values computed with ERFA 2.0.1 (BSD), used only as numeric
    // test vectors (§6): `nut00b`, `pfw06`, `pmat06`, `pn06` with the 2000B
    // nutation, and `atci13` minus the equation of the origins. UT1 = UTC.

    type Instant = (i32, u32, u32, u32, u32, f64);

    #[rustfmt::skip]
    const NUTATION_AND_FW: [(Instant, f64, f64, [f64; 4]); 6] = [
        ((2026, 10, 26, 3, 15, 42.5), 7.674_949_440, 7.913_952_161,
         [1.363_954_436_098_252e-5, 4.090_317_930_627_592e-1, 6.550_813_476_631_514e-3, 4.090_317_087_711_753e-1]),
        ((2000, 1, 1, 12, 0, 0.0), -13.931_658_618, -5.769_432_340,
         [-2.566_011_442_290_179e-7, 4.090_926_336_554_12e-1, -2.020_340_972_564_551e-7, 4.090_926_005_959_645e-1]),
        ((2045, 6, 30, 23, 59, 59.0), 10.298_631_524, 6.932_974_757,
         [2.352_260_730_026_733e-5, 4.089_894_332_230_896e-1, 1.111_483_987_187_743e-2, 4.089_892_925_406_048e-1]),
        ((1987, 4, 10, 19, 21, 0.0), -3.860_431_235, 9.468_315_334,
         [-6.731_606_375_890_41e-6, 4.091_215_220_855_472e-1, -3.109_040_508_604_202e-3, 4.091_215_008_725_581e-1]),
        ((2012, 3, 20, 6, 0, 0.0), 16.549_368_899, -2.621_601_032,
         [6.030_872_842_424_442e-6, 4.090_649_145_044_589e-1, 2.983_846_979_247_92e-3, 4.090_648_624_943_41e-1]),
        ((2031, 12, 31, 18, 30, 0.0), 13.235_038_806, -6.915_363_271,
         [1.636_435_261_801_507e-5, 4.090_200_408_618_596e-1, 7.816_803_116_784_107e-3, 4.090_199_426_155_78e-1]),
    ];

    #[test]
    fn nutation_and_precession_angles_match_erfa() {
        for ((y, mo, d, h, mi, s), dpsi, deps, fw_ref) in NUTATION_AND_FW {
            let t = TimeScales::from_utc(utc(y, mo, d, h, mi, s)).tt_centuries();
            let n = nutation_2000b(t);
            // The same IAU 2000B series: agreement to well under 1 µas.
            assert!(
                (n.dpsi_rad / AS - dpsi).abs() < 1e-6,
                "{y}: Δψ {}",
                n.dpsi_rad / AS
            );
            assert!(
                (n.deps_rad / AS - deps).abs() < 1e-6,
                "{y}: Δε {}",
                n.deps_rad / AS
            );
            let fw = fukushima_williams(t);
            let ours = [fw.gamma_bar, fw.phi_bar, fw.psi_bar, fw.eps_a];
            for (a, b) in ours.iter().zip(fw_ref) {
                assert!((a - b).abs() < 1e-14, "{y}: {a} vs {b}");
            }
        }
    }

    /// RA in hours, Dec in degrees.
    type RaDec = (f64, f64);
    /// (instant, J2000 star, mean of date, true of date without aberration,
    /// apparent).
    type PlaceRow = (Instant, RaDec, RaDec, RaDec, RaDec);

    #[rustfmt::skip]
    const PLACES: [PlaceRow; 18] = [
        ((2026, 10, 26, 3, 15, 42.5), (2.736_662, 49.228_467), (2.767_197_167_4, 49.340_647_997_0), (2.767_243_407_7, 49.342_739_883_1), (2.767_793_251_6, 49.343_574_871_7)),
        ((2026, 10, 26, 3, 15, 42.5), (18.615_649, 38.783_689), (18.630_660_945_7, 38.807_938_357_1), (18.630_727_131_8, 38.805_909_309_9), (18.630_521_465_9, 38.810_416_602_0)),
        ((2026, 10, 26, 3, 15, 42.5), (6.752_481, -16.716_116), (6.772_454_857_8, -16.745_720_191_9), (6.772_559_742_6, -16.743_737_001_1), (6.772_714_514_7, -16.740_504_248_8)),
        ((2026, 10, 26, 3, 15, 42.5), (12.0, 89.5), (12.020_250_452_5, 89.350_722_498_0), (12.033_269_256_2, 89.349_859_165_7), (12.007_536_244_5, 89.346_738_061_7)),
        ((2026, 10, 26, 3, 15, 42.5), (23.99, -88.0), (0.012_798_353_6, -87.850_721_983_8), (0.016_827_152_4, -87.849_865_513_9), (0.024_634_590_8, -87.852_934_213_0)),
        ((2026, 10, 26, 3, 15, 42.5), (14.660_765, -60.833_976), (14.695_200_658_1, -60.948_042_530_1), (14.695_196_218_2, -60.950_113_601_3), (14.694_457_315_8, -60.949_566_645_3)),
        ((2000, 1, 1, 12, 0, 0.0), (2.736_662, 49.228_467), (2.736_662_148_8, 49.228_462_283_6), (2.736_440_732_1, 49.226_248_971_2), (2.736_748_420_3, 49.229_705_896_2)),
        ((2000, 1, 1, 12, 0, 0.0), (0.0, 0.0), (0.000_000_272_1, -0.000_004_604_5), (23.999_763_564_6, -0.001_543_864_5), (23.999_699_549_6, -0.001_960_137_0)),
        ((2000, 1, 1, 12, 0, 0.0), (12.0, 89.5), (11.999_985_802_0, 89.500_004_604_5), (11.987_468_004_2, 89.501_541_282_0), (11.994_885_046_2, 89.495_847_475_5)),
        ((2000, 1, 1, 12, 0, 0.0), (14.660_765, -60.833_976), (14.660_765_092_8, -60.833_971_252_8), (14.660_557_257_3, -60.831_762_306_3), (14.660_158_298_4, -60.827_614_593_7)),
        ((2045, 6, 30, 23, 59, 59.0), (2.736_662, 49.228_467), (2.788_528_236_7, 49.418_320_530_5), (2.788_650_584_3, 49.420_452_629_4), (2.788_338_007_6, 49.417_139_901_8)),
        ((2045, 6, 30, 23, 59, 59.0), (18.615_649, 38.783_689), (18.641_119_468_1, 38.825_169_808_2), (18.641_217_013_7, 38.823_461_126_3), (18.641_694_643_0, 38.823_692_539_7)),
        ((2045, 6, 30, 23, 59, 59.0), (0.0, 0.0), (0.038_864_366_6, 0.253_252_943_6), (0.039_038_785_8, 0.254_410_228_6), (0.039_091_482_5, 0.254_752_643_6)),
        ((2045, 6, 30, 23, 59, 59.0), (23.99, -88.0), (0.027_795_897_9, -87.746_744_449_8), (0.031_218_040_6, -87.745_591_901_1), (0.032_573_742_6, -87.740_057_145_5)),
        ((2031, 12, 31, 18, 30, 0.0), (6.752_481, -16.716_116), (6.776_314_559_3, -16.751_528_266_0), (6.776_518_481_9, -16.753_704_802_3), (6.776_920_133_6, -16.754_010_984_9)),
        ((2031, 12, 31, 18, 30, 0.0), (12.0, 89.5), (12.023_719_446_0, 89.321_878_767_0), (12.013_096_653_6, 89.320_425_954_4), (12.018_088_091_4, 89.314_716_232_6)),
        ((2031, 12, 31, 18, 30, 0.0), (23.99, -88.0), (0.017_025_543_2, -87.821_877_994_4), (0.013_874_138_6, -87.820_423_690_4), (0.012_304_212_8, -87.826_139_583_4)),
        ((2031, 12, 31, 18, 30, 0.0), (14.660_765, -60.833_976), (14.701_875_420_3, -60.969_982_768_7), (14.702_389_809_0, -60.969_845_654_5), (14.701_976_120_5, -60.965_729_615_8)),
    ];

    #[test]
    fn mean_true_and_apparent_places_match_erfa() {
        let mut worst = [0.0_f64; 3];
        for ((y, mo, d, h, mi, s), star, mean, true_no_ab, apparent) in PLACES {
            let ctx = FrameContext::at(utc(y, mo, d, h, mi, s));
            let c = EqCoord::new(star.0, star.1, EqFrame::J2000);
            let ours_mean = ctx.convert(c, EqFrame::JNowMean).unwrap();
            let v = radec_to_vec(c.ra_h * HOURS_TO_RAD, c.dec_deg.to_radians());
            let (ra, dec) = vec_to_radec(mat_vec(&ctx.precession_nutation, v));
            let ours_true = EqCoord::new(ra / HOURS_TO_RAD, dec.to_degrees(), EqFrame::Unknown);
            let ours_app = ctx.convert(c, EqFrame::JNowApparent).unwrap();
            let pairs = [
                (ours_mean, mean),
                (ours_true, true_no_ab),
                (ours_app, apparent),
            ];
            for (slot, (ours, (ra_ref, dec_ref))) in pairs.into_iter().enumerate() {
                let reference = EqCoord::new(ra_ref, dec_ref, ours.frame);
                worst[slot] = worst[slot].max(sep_arcsec(ours, reference));
            }
        }
        // The precession and nutation are the same models: µas agreement.
        assert!(worst[0] < 1e-4, "mean of date {}\"", worst[0]);
        assert!(worst[1] < 1e-4, "true of date {}\"", worst[1]);
        // The apparent place differs from ERFA only by the Keplerian Earth
        // velocity and the omitted light deflection.
        assert!(worst[2] < 0.05, "apparent {}\"", worst[2]);
    }
}

//! Sun, Moon, Moon phase and twilight (WP1A-3; H14 §7.2.4 COC, BGE §7.2.6,
//! Model 7 §7.2.7, FW-1 §7.2.12, FW-2 §7.2.13, planning §7.2.21).
//!
//! - **Sun:** Meeus ch. 25 (geometric longitude from the mean elements and
//!   the equation of centre), about 0.01°. The apparent longitude adds the
//!   IAU 2000B nutation and the annual aberration −20.4898″/R.
//! - **Moon:** Meeus ch. 47, the truncated ELP-2000/82 series (60 terms in
//!   longitude and distance, 60 in latitude), about 10″ in longitude and 4″
//!   in latitude. The apparent longitude adds the nutation in longitude
//!   (the series' mean longitude already includes the light time).
//! - **Topocentric positions** subtract the observer's WGS84 geocentric
//!   vector, so the Moon's parallax (up to about 1°) is exact to the series
//!   accuracy. Altitudes are geometric (no refraction).
//! - **Moon phase:** Meeus ch. 48 from the geocentric elongation.
//! - **Twilight:** the Sun's topocentric geometric altitude against the
//!   standard thresholds: −0.833° (sunrise and sunset: 34′ of refraction
//!   plus the 16′ semi-diameter), −6°, −12° and −18°.
//!
//! **Measured accuracy** (unit tests, 2000-2045, four sites from 35° S to
//! 70° N): topocentric altitude within 0.006° for the Sun and 0.0003° for
//! the Moon of an ERFA reference (`epv00`, `moon98`, `c2t06a`); the
//! acceptance bound is 0.1° (§5.3).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::frames::{mean_obliquity_rad, nutation_2000b};
use super::meridian::{altaz_from_hadec, hour_angle_h, Horizontal};
use super::time::{self, TimeScales};
use super::{find_crossings, norm_deg, radec_to_vec, sub, vec_to_radec, Crossing, HOURS_TO_RAD};
use crate::equipment::snapshot::{CocContext, Twilight};
use crate::equipment::types::{EqCoord, EqFrame, Site};

/// Astronomical unit, km.
pub const AU_KM: f64 = 149_597_870.7;
/// WGS84 equatorial radius, km, and flattening.
const EARTH_RADIUS_KM: f64 = 6_378.137;
const EARTH_FLATTENING: f64 = 1.0 / 298.257_223_563;

/// Sun altitude of sunrise and sunset (centre, geometric).
pub const SUNRISE_ALT_DEG: f64 = -0.833;
pub const CIVIL_TWILIGHT_ALT_DEG: f64 = -6.0;
pub const NAUTICAL_TWILIGHT_ALT_DEG: f64 = -12.0;
pub const ASTRONOMICAL_TWILIGHT_ALT_DEG: f64 = -18.0;

/// A body's geocentric apparent place.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BodyPosition {
    /// Geocentric apparent RA/Dec, [`EqFrame::JNowApparent`].
    pub apparent: EqCoord,
    /// Apparent ecliptic longitude and latitude of date, degrees.
    pub ecliptic_lon_deg: f64,
    pub ecliptic_lat_deg: f64,
    pub distance_km: f64,
}

/// A body seen from a site.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TopocentricPosition {
    /// Topocentric alt/az, HA, LST and RA/Dec (in `apparent`).
    pub horizontal: Horizontal,
    pub distance_km: f64,
}

/// The Moon's illumination as seen from the Earth's centre.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MoonPhase {
    /// Illuminated fraction of the disc, 0..1.
    pub illuminated_fraction: f64,
    /// Sun-Moon-Earth angle, degrees (0 = full, 180 = new).
    pub phase_angle_deg: f64,
    /// Geocentric Sun-Moon elongation, degrees.
    pub elongation_deg: f64,
    pub waxing: bool,
}

fn ecliptic_to_apparent(lon_deg: f64, lat_deg: f64, true_obliquity: f64) -> EqCoord {
    let (sl, cl) = lon_deg.to_radians().sin_cos();
    let (sb, cb) = lat_deg.to_radians().sin_cos();
    let (se, ce) = true_obliquity.sin_cos();
    let v = [cb * cl, cb * sl * ce - sb * se, cb * sl * se + sb * ce];
    let (ra, dec) = vec_to_radec(v);
    EqCoord::new(ra / HOURS_TO_RAD, dec.to_degrees(), EqFrame::JNowApparent).normalized()
}

// ---------------------------------------------------------------------------
// Sun
// ---------------------------------------------------------------------------

/// The Sun's geocentric apparent place at `ts`.
pub fn sun_at(ts: &TimeScales) -> BodyPosition {
    let t = ts.tt_centuries();
    let l0 = 280.466_46 + t * (36_000.769_83 + t * 0.000_303_2);
    let m_deg = 357.529_11 + t * (35_999.050_29 - t * 0.000_153_7);
    let m = m_deg.to_radians();
    let e = 0.016_708_634 - t * (0.000_042_037 + t * 0.000_000_126_7);
    let c = (1.914_602 - t * (0.004_817 + t * 0.000_014)) * m.sin()
        + (0.019_993 - t * 0.000_101) * (2.0 * m).sin()
        + 0.000_289 * (3.0 * m).sin();
    let true_lon = l0 + c;
    let anomaly = (m_deg + c).to_radians();
    let r_au = 1.000_001_018 * (1.0 - e * e) / (1.0 + e * anomaly.cos());
    let nutation = nutation_2000b(t);
    let lon = norm_deg(true_lon + nutation.dpsi_rad.to_degrees() - 20.489_8 / 3600.0 / r_au);
    let eps = mean_obliquity_rad(t) + nutation.deps_rad;
    BodyPosition {
        apparent: ecliptic_to_apparent(lon, 0.0, eps),
        ecliptic_lon_deg: lon,
        ecliptic_lat_deg: 0.0,
        distance_km: r_au * AU_KM,
    }
}

/// The Sun's geocentric apparent place at a UTC instant.
pub fn sun(at: DateTime<Utc>) -> BodyPosition {
    sun_at(&TimeScales::from_utc(at))
}

// ---------------------------------------------------------------------------
// Moon
// ---------------------------------------------------------------------------

/// Multipliers of (D, M, M′, F) and the longitude (1e-6°) and distance
/// (1e-3 km) coefficients: Meeus table 47.A.
#[rustfmt::skip]
const MOON_LON_DIST: [([i8; 4], f64, f64); 60] = [
    ([0, 0, 1, 0], 6_288_774.0, -20_905_355.0),
    ([2, 0,-1, 0], 1_274_027.0,  -3_699_111.0),
    ([2, 0, 0, 0],   658_314.0,  -2_955_968.0),
    ([0, 0, 2, 0],   213_618.0,    -569_925.0),
    ([0, 1, 0, 0],  -185_116.0,      48_888.0),
    ([0, 0, 0, 2],  -114_332.0,      -3_149.0),
    ([2, 0,-2, 0],    58_793.0,     246_158.0),
    ([2,-1,-1, 0],    57_066.0,    -152_138.0),
    ([2, 0, 1, 0],    53_322.0,    -170_733.0),
    ([2,-1, 0, 0],    45_758.0,    -204_586.0),
    ([0, 1,-1, 0],   -40_923.0,    -129_620.0),
    ([1, 0, 0, 0],   -34_720.0,     108_743.0),
    ([0, 1, 1, 0],   -30_383.0,     104_755.0),
    ([2, 0, 0,-2],    15_327.0,      10_321.0),
    ([0, 0, 1, 2],   -12_528.0,           0.0),
    ([0, 0, 1,-2],    10_980.0,      79_661.0),
    ([4, 0,-1, 0],    10_675.0,     -34_782.0),
    ([0, 0, 3, 0],    10_034.0,     -23_210.0),
    ([4, 0,-2, 0],     8_548.0,     -21_636.0),
    ([2, 1,-1, 0],    -7_888.0,      24_208.0),
    ([2, 1, 0, 0],    -6_766.0,      30_824.0),
    ([1, 0,-1, 0],    -5_163.0,      -8_379.0),
    ([1, 1, 0, 0],     4_987.0,     -16_675.0),
    ([2,-1, 1, 0],     4_036.0,     -12_831.0),
    ([2, 0, 2, 0],     3_994.0,     -10_445.0),
    ([4, 0, 0, 0],     3_861.0,     -11_650.0),
    ([2, 0,-3, 0],     3_665.0,      14_403.0),
    ([0, 1,-2, 0],    -2_689.0,      -7_003.0),
    ([2, 0,-1, 2],    -2_602.0,           0.0),
    ([2,-1,-2, 0],     2_390.0,      10_056.0),
    ([1, 0, 1, 0],    -2_348.0,       6_322.0),
    ([2,-2, 0, 0],     2_236.0,      -9_884.0),
    ([0, 1, 2, 0],    -2_120.0,       5_751.0),
    ([0, 2, 0, 0],    -2_069.0,           0.0),
    ([2,-2,-1, 0],     2_048.0,      -4_950.0),
    ([2, 0, 1,-2],    -1_773.0,       4_130.0),
    ([2, 0, 0, 2],    -1_595.0,           0.0),
    ([4,-1,-1, 0],     1_215.0,      -3_958.0),
    ([0, 0, 2, 2],    -1_110.0,           0.0),
    ([3, 0,-1, 0],      -892.0,       3_258.0),
    ([2, 1, 1, 0],      -810.0,       2_616.0),
    ([4,-1,-2, 0],       759.0,      -1_897.0),
    ([0, 2,-1, 0],      -713.0,      -2_117.0),
    ([2, 2,-1, 0],      -700.0,       2_354.0),
    ([2, 1,-2, 0],       691.0,           0.0),
    ([2,-1, 0,-2],       596.0,           0.0),
    ([4, 0, 1, 0],       549.0,      -1_423.0),
    ([0, 0, 4, 0],       537.0,      -1_117.0),
    ([4,-1, 0, 0],       520.0,      -1_571.0),
    ([1, 0,-2, 0],      -487.0,      -1_739.0),
    ([2, 1, 0,-2],      -399.0,           0.0),
    ([0, 0, 2,-2],      -381.0,      -4_421.0),
    ([1, 1, 1, 0],       351.0,           0.0),
    ([3, 0,-2, 0],      -340.0,           0.0),
    ([4, 0,-3, 0],       330.0,           0.0),
    ([2,-1, 2, 0],       327.0,           0.0),
    ([0, 2, 1, 0],      -323.0,       1_165.0),
    ([1, 1,-1, 0],       299.0,           0.0),
    ([2, 0, 3, 0],       294.0,           0.0),
    ([2, 0,-1,-2],         0.0,       8_752.0),
];

/// Multipliers of (D, M, M′, F) and the latitude coefficient (1e-6°):
/// Meeus table 47.B.
#[rustfmt::skip]
const MOON_LAT: [([i8; 4], f64); 60] = [
    ([0, 0, 0, 1], 5_128_122.0),
    ([0, 0, 1, 1],   280_602.0),
    ([0, 0, 1,-1],   277_693.0),
    ([2, 0, 0,-1],   173_237.0),
    ([2, 0,-1, 1],    55_413.0),
    ([2, 0,-1,-1],    46_271.0),
    ([2, 0, 0, 1],    32_573.0),
    ([0, 0, 2, 1],    17_198.0),
    ([2, 0, 1,-1],     9_266.0),
    ([0, 0, 2,-1],     8_822.0),
    ([2,-1, 0,-1],     8_216.0),
    ([2, 0,-2,-1],     4_324.0),
    ([2, 0, 1, 1],     4_200.0),
    ([2, 1, 0,-1],    -3_359.0),
    ([2,-1,-1, 1],     2_463.0),
    ([2,-1, 0, 1],     2_211.0),
    ([2,-1,-1,-1],     2_065.0),
    ([0, 1,-1,-1],    -1_870.0),
    ([4, 0,-1,-1],     1_828.0),
    ([0, 1, 0, 1],    -1_794.0),
    ([0, 0, 0, 3],    -1_749.0),
    ([0, 1,-1, 1],    -1_565.0),
    ([1, 0, 0, 1],    -1_491.0),
    ([0, 1, 1, 1],    -1_475.0),
    ([0, 1, 1,-1],    -1_410.0),
    ([0, 1, 0,-1],    -1_344.0),
    ([1, 0, 0,-1],    -1_335.0),
    ([0, 0, 3, 1],     1_107.0),
    ([4, 0, 0,-1],     1_021.0),
    ([4, 0,-1, 1],       833.0),
    ([0, 0, 1,-3],       777.0),
    ([4, 0,-2, 1],       671.0),
    ([2, 0, 0,-3],       607.0),
    ([2, 0, 2,-1],       596.0),
    ([2,-1, 1,-1],       491.0),
    ([2, 0,-2, 1],      -451.0),
    ([0, 0, 3,-1],       439.0),
    ([2, 0, 2, 1],       422.0),
    ([2, 0,-3,-1],       421.0),
    ([2, 1,-1, 1],      -366.0),
    ([2, 1, 0, 1],      -351.0),
    ([4, 0, 0, 1],       331.0),
    ([2,-1, 1, 1],       315.0),
    ([2,-2, 0,-1],       302.0),
    ([0, 0, 1, 3],      -283.0),
    ([2, 1, 1,-1],      -229.0),
    ([1, 1, 0,-1],       223.0),
    ([1, 1, 0, 1],       223.0),
    ([0, 1,-2,-1],      -220.0),
    ([2, 1,-1,-1],      -220.0),
    ([1, 0, 1, 1],      -185.0),
    ([2,-1,-2,-1],       181.0),
    ([0, 1, 2, 1],      -177.0),
    ([4, 0,-2,-1],       176.0),
    ([4,-1,-1,-1],       166.0),
    ([1, 0, 1,-1],      -164.0),
    ([4, 0, 1,-1],       132.0),
    ([1, 0,-1,-1],      -119.0),
    ([4,-1, 0,-1],       115.0),
    ([2,-2, 0, 1],       107.0),
];

/// The series sums Σl, Σb (1e-6°) and Σr (1e-3 km), and the Moon's mean
/// longitude L′ (degrees), at `t` centuries of TT.
fn moon_sums(t: f64) -> (f64, f64, f64, f64) {
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let lp = 218.316_447_7 + 481_267.881_234_21 * t - 0.001_578_6 * t2 + t3 / 538_841.0
        - t4 / 65_194_000.0;
    let d = 297.850_192_1 + 445_267.111_403_4 * t - 0.001_881_9 * t2 + t3 / 545_868.0
        - t4 / 113_065_000.0;
    let m = 357.529_109_2 + 35_999.050_290_9 * t - 0.000_153_6 * t2 + t3 / 24_490_000.0;
    let mp = 134.963_396_4 + 477_198.867_505_5 * t + 0.008_741_4 * t2 + t3 / 69_699.0
        - t4 / 14_712_000.0;
    let f = 93.272_095_0 + 483_202.017_523_3 * t - 0.003_653_9 * t2 - t3 / 3_526_000.0
        + t4 / 863_310_000.0;
    let a1 = 119.75 + 131.849 * t;
    let a2 = 53.09 + 479_264.290 * t;
    let a3 = 313.45 + 481_266.484 * t;
    let e = 1.0 - 0.002_516 * t - 0.000_007_4 * t2;

    let arg = |k: &[i8; 4]| {
        (f64::from(k[0]) * d + f64::from(k[1]) * m + f64::from(k[2]) * mp + f64::from(k[3]) * f)
            .to_radians()
    };
    let ecc = |k: &[i8; 4]| match k[1].abs() {
        1 => e,
        2 => e * e,
        _ => 1.0,
    };

    let (mut sl, mut sr) = (0.0_f64, 0.0_f64);
    for (k, cl, cr) in MOON_LON_DIST.iter().rev() {
        let (s, c) = arg(k).sin_cos();
        sl += cl * ecc(k) * s;
        sr += cr * ecc(k) * c;
    }
    let mut sb = 0.0_f64;
    for (k, cb) in MOON_LAT.iter().rev() {
        sb += cb * ecc(k) * arg(k).sin();
    }
    let sin_deg = |x: f64| x.to_radians().sin();
    sl += 3958.0 * sin_deg(a1) + 1962.0 * sin_deg(lp - f) + 318.0 * sin_deg(a2);
    sb += -2235.0 * sin_deg(lp)
        + 382.0 * sin_deg(a3)
        + 175.0 * sin_deg(a1 - f)
        + 175.0 * sin_deg(a1 + f)
        + 127.0 * sin_deg(lp - mp)
        - 115.0 * sin_deg(lp + mp);
    (sl, sb, sr, lp)
}

/// The Moon's geocentric apparent place at `ts`.
pub fn moon_at(ts: &TimeScales) -> BodyPosition {
    let t = ts.tt_centuries();
    let (sl, sb, sr, lp) = moon_sums(t);
    let nutation = nutation_2000b(t);
    let lon = norm_deg(lp + sl / 1e6 + nutation.dpsi_rad.to_degrees());
    let lat = sb / 1e6;
    let eps = mean_obliquity_rad(t) + nutation.deps_rad;
    BodyPosition {
        apparent: ecliptic_to_apparent(lon, lat, eps),
        ecliptic_lon_deg: lon,
        ecliptic_lat_deg: lat,
        distance_km: 385_000.56 + sr / 1000.0,
    }
}

/// The Moon's geocentric apparent place at a UTC instant.
pub fn moon(at: DateTime<Utc>) -> BodyPosition {
    moon_at(&TimeScales::from_utc(at))
}

// ---------------------------------------------------------------------------
// Topocentric positions
// ---------------------------------------------------------------------------

/// The observer's geocentric position, km, in true-of-date equatorial axes
/// at local apparent sidereal time `lst_h` (WGS84; polar motion neglected).
fn observer_vector_km(site: &Site, lst_h: f64) -> [f64; 3] {
    let phi = site.lat_deg.to_radians();
    let (sp, cp) = phi.sin_cos();
    let b = 1.0 - EARTH_FLATTENING;
    let c = 1.0 / (cp * cp + b * b * sp * sp).sqrt();
    let s = b * b * c;
    let h_km = site.elev_m / 1000.0;
    let rho_cos = (EARTH_RADIUS_KM * c + h_km) * cp;
    let rho_sin = (EARTH_RADIUS_KM * s + h_km) * sp;
    let (st, ct) = (lst_h * HOURS_TO_RAD).sin_cos();
    [rho_cos * ct, rho_cos * st, rho_sin]
}

/// A body's position seen from a site; `None` for an invalid site.
pub fn topocentric_at(
    body: &BodyPosition,
    site: &Site,
    ts: &TimeScales,
) -> Option<TopocentricPosition> {
    site.validate().ok()?;
    let lst = time::local_sidereal_time_from(ts, site.lon_deg);
    let geo = radec_to_vec(
        body.apparent.ra_h * HOURS_TO_RAD,
        body.apparent.dec_deg.to_radians(),
    );
    let topo = sub(
        [
            geo[0] * body.distance_km,
            geo[1] * body.distance_km,
            geo[2] * body.distance_km,
        ],
        observer_vector_km(site, lst),
    );
    let distance_km = (topo[0] * topo[0] + topo[1] * topo[1] + topo[2] * topo[2]).sqrt();
    let (ra, dec) = vec_to_radec(topo);
    let apparent =
        EqCoord::new(ra / HOURS_TO_RAD, dec.to_degrees(), EqFrame::JNowApparent).normalized();
    let ha = hour_angle_h(lst, apparent.ra_h);
    let position = altaz_from_hadec(ha, apparent.dec_deg, site.lat_deg);
    Some(TopocentricPosition {
        horizontal: Horizontal {
            alt_deg: position.alt_deg,
            az_deg: position.az_deg,
            ha_h: ha,
            lst_h: lst,
            apparent,
        },
        distance_km,
    })
}

/// The Sun seen from a site.
pub fn sun_topocentric(site: &Site, at: DateTime<Utc>) -> Option<TopocentricPosition> {
    let ts = TimeScales::from_utc(at);
    topocentric_at(&sun_at(&ts), site, &ts)
}

/// The Moon seen from a site.
pub fn moon_topocentric(site: &Site, at: DateTime<Utc>) -> Option<TopocentricPosition> {
    let ts = TimeScales::from_utc(at);
    topocentric_at(&moon_at(&ts), site, &ts)
}

// ---------------------------------------------------------------------------
// Moon phase and separation
// ---------------------------------------------------------------------------

fn phase_from(sun: &BodyPosition, moon: &BodyPosition) -> MoonPhase {
    let s = EqCoord {
        frame: EqFrame::JNowApparent,
        ..sun.apparent
    };
    let elongation = s
        .separation_arcmin(&moon.apparent)
        .map(|arcmin| arcmin / 60.0)
        .unwrap_or(0.0);
    let psi = elongation.to_radians();
    let i = (sun.distance_km * psi.sin()).atan2(moon.distance_km - sun.distance_km * psi.cos());
    MoonPhase {
        illuminated_fraction: (1.0 + i.cos()) / 2.0,
        phase_angle_deg: i.to_degrees(),
        elongation_deg: elongation,
        waxing: norm_deg(moon.ecliptic_lon_deg - sun.ecliptic_lon_deg) < 180.0,
    }
}

/// The Moon's phase at `ts`.
pub fn moon_phase_at(ts: &TimeScales) -> MoonPhase {
    phase_from(&sun_at(ts), &moon_at(ts))
}

/// The Moon's phase at a UTC instant (geocentric; no site needed).
pub fn moon_phase(at: DateTime<Utc>) -> MoonPhase {
    moon_phase_at(&TimeScales::from_utc(at))
}

/// Angular distance from a target (any known frame) to the Moon, degrees:
/// topocentric when a site is given (the parallax moves the Moon by up to
/// about 1°), geocentric otherwise. `None` for an unknown frame.
pub fn moon_separation_deg(target: EqCoord, site: Option<&Site>, at: DateTime<Utc>) -> Option<f64> {
    let apparent = super::frames::convert(target, EqFrame::JNowApparent, at).ok()?;
    let ts = TimeScales::from_utc(at);
    let body = moon_at(&ts);
    let moon = match site {
        Some(s) => topocentric_at(&body, s, &ts)?.horizontal.apparent,
        None => body.apparent,
    };
    apparent
        .separation_arcmin(&moon)
        .map(|arcmin| arcmin / 60.0)
}

// ---------------------------------------------------------------------------
// Twilight
// ---------------------------------------------------------------------------

/// The twilight class for a geometric Sun-centre altitude.
pub fn twilight_for_sun_altitude(sun_alt_deg: f64) -> Twilight {
    if sun_alt_deg >= SUNRISE_ALT_DEG {
        Twilight::Day
    } else if sun_alt_deg >= CIVIL_TWILIGHT_ALT_DEG {
        Twilight::Civil
    } else if sun_alt_deg >= NAUTICAL_TWILIGHT_ALT_DEG {
        Twilight::Nautical
    } else if sun_alt_deg >= ASTRONOMICAL_TWILIGHT_ALT_DEG {
        Twilight::Astronomical
    } else {
        Twilight::Night
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TwilightEventKind {
    Sunset,
    CivilDusk,
    NauticalDusk,
    AstronomicalDusk,
    AstronomicalDawn,
    NauticalDawn,
    CivilDawn,
    Sunrise,
}

impl TwilightEventKind {
    fn threshold_and_rising(self) -> (f64, bool) {
        match self {
            TwilightEventKind::Sunset => (SUNRISE_ALT_DEG, false),
            TwilightEventKind::CivilDusk => (CIVIL_TWILIGHT_ALT_DEG, false),
            TwilightEventKind::NauticalDusk => (NAUTICAL_TWILIGHT_ALT_DEG, false),
            TwilightEventKind::AstronomicalDusk => (ASTRONOMICAL_TWILIGHT_ALT_DEG, false),
            TwilightEventKind::AstronomicalDawn => (ASTRONOMICAL_TWILIGHT_ALT_DEG, true),
            TwilightEventKind::NauticalDawn => (NAUTICAL_TWILIGHT_ALT_DEG, true),
            TwilightEventKind::CivilDawn => (CIVIL_TWILIGHT_ALT_DEG, true),
            TwilightEventKind::Sunrise => (SUNRISE_ALT_DEG, true),
        }
    }

    /// The kinds in the order they happen through one night.
    pub const NIGHT_ORDER: [TwilightEventKind; 8] = [
        TwilightEventKind::Sunset,
        TwilightEventKind::CivilDusk,
        TwilightEventKind::NauticalDusk,
        TwilightEventKind::AstronomicalDusk,
        TwilightEventKind::AstronomicalDawn,
        TwilightEventKind::NauticalDawn,
        TwilightEventKind::CivilDawn,
        TwilightEventKind::Sunrise,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TwilightEvent {
    pub kind: TwilightEventKind,
    pub at: DateTime<Utc>,
}

fn sun_alt(site: &Site, t: DateTime<Utc>) -> Option<f64> {
    sun_topocentric(site, t).map(|p| p.horizontal.alt_deg)
}

/// Crossings of a Sun altitude in `[from, to]`, sampled every 10 minutes.
pub fn sun_altitude_crossings(
    site: &Site,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    threshold_deg: f64,
) -> Vec<Crossing> {
    find_crossings(
        |t| sun_alt(site, t),
        threshold_deg,
        from,
        to,
        chrono::Duration::minutes(10),
    )
}

/// Every sunrise, sunset, dusk and dawn in `[from, to]`, in time order.
pub fn twilight_events(site: &Site, from: DateTime<Utc>, to: DateTime<Utc>) -> Vec<TwilightEvent> {
    let mut out = Vec::new();
    for threshold in [
        SUNRISE_ALT_DEG,
        CIVIL_TWILIGHT_ALT_DEG,
        NAUTICAL_TWILIGHT_ALT_DEG,
        ASTRONOMICAL_TWILIGHT_ALT_DEG,
    ] {
        for x in sun_altitude_crossings(site, from, to, threshold) {
            let kind = TwilightEventKind::NIGHT_ORDER
                .iter()
                .copied()
                .find(|k| k.threshold_and_rising() == (threshold, x.rising));
            if let Some(kind) = kind {
                out.push(TwilightEvent { kind, at: x.at });
            }
        }
    }
    out.sort_by_key(|e| e.at);
    out
}

/// The events of one night; a field is `None` when that event does not
/// happen that night (high latitudes near the solstices).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct NightTimes {
    pub sunset: Option<DateTime<Utc>>,
    pub civil_dusk: Option<DateTime<Utc>>,
    pub nautical_dusk: Option<DateTime<Utc>>,
    pub astronomical_dusk: Option<DateTime<Utc>>,
    pub astronomical_dawn: Option<DateTime<Utc>>,
    pub nautical_dawn: Option<DateTime<Utc>>,
    pub civil_dawn: Option<DateTime<Utc>>,
    pub sunrise: Option<DateTime<Utc>>,
}

impl NightTimes {
    fn slot(&mut self, kind: TwilightEventKind) -> &mut Option<DateTime<Utc>> {
        match kind {
            TwilightEventKind::Sunset => &mut self.sunset,
            TwilightEventKind::CivilDusk => &mut self.civil_dusk,
            TwilightEventKind::NauticalDusk => &mut self.nautical_dusk,
            TwilightEventKind::AstronomicalDusk => &mut self.astronomical_dusk,
            TwilightEventKind::AstronomicalDawn => &mut self.astronomical_dawn,
            TwilightEventKind::NauticalDawn => &mut self.nautical_dawn,
            TwilightEventKind::CivilDawn => &mut self.civil_dawn,
            TwilightEventKind::Sunrise => &mut self.sunrise,
        }
    }
}

/// The night in progress at `from` (when the Sun is down) or the next one.
///
/// The night starts at its sunset (the last one before `from` when the Sun
/// is already down, else the next one; `from` itself when there is none, as
/// in polar night) and ends at the first sunrise after that (or 24 h
/// later). Each other event is the first of its kind inside that span after
/// the previous event found. `None` for an invalid site.
pub fn night_times(site: &Site, from: DateTime<Utc>) -> Option<NightTimes> {
    let sun_down = sun_alt(site, from)? < SUNRISE_ALT_DEG;
    let events = twilight_events(
        site,
        from - chrono::Duration::hours(26),
        from + chrono::Duration::hours(50),
    );
    let sunset = if sun_down {
        events
            .iter()
            .rev()
            .find(|e| e.kind == TwilightEventKind::Sunset && e.at <= from)
    } else {
        events
            .iter()
            .find(|e| e.kind == TwilightEventKind::Sunset && e.at > from)
    }
    .map(|e| e.at);
    let start = sunset.unwrap_or(from);
    let end = events
        .iter()
        .find(|e| e.kind == TwilightEventKind::Sunrise && e.at > start)
        .map(|e| e.at)
        .filter(|at| *at - start <= chrono::Duration::hours(24));
    let limit = end.unwrap_or(start + chrono::Duration::hours(24));

    let mut night = NightTimes {
        sunset,
        sunrise: end,
        ..NightTimes::default()
    };
    let mut cursor = start;
    for kind in &TwilightEventKind::NIGHT_ORDER[1..7] {
        let found = events
            .iter()
            .find(|e| e.kind == *kind && e.at > cursor && e.at < limit)
            .map(|e| e.at);
        if let Some(at) = found {
            cursor = at;
        }
        *night.slot(*kind) = found;
    }
    Some(night)
}

// ---------------------------------------------------------------------------
// Sky context
// ---------------------------------------------------------------------------

/// Sun, Moon and twilight at a site and instant.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SkyConditions {
    pub at: DateTime<Utc>,
    /// Topocentric geometric altitudes and azimuths, degrees.
    pub sun_alt_deg: f64,
    pub sun_az_deg: f64,
    pub moon_alt_deg: f64,
    pub moon_az_deg: f64,
    pub moon: MoonPhase,
    pub twilight: Twilight,
}

/// Sun, Moon and twilight at a site; `None` for an invalid site.
pub fn sky_conditions(site: &Site, at: DateTime<Utc>) -> Option<SkyConditions> {
    let ts = TimeScales::from_utc(at);
    let sun_body = sun_at(&ts);
    let moon_body = moon_at(&ts);
    let sun = topocentric_at(&sun_body, site, &ts)?;
    let moon = topocentric_at(&moon_body, site, &ts)?;
    Some(SkyConditions {
        at,
        sun_alt_deg: sun.horizontal.alt_deg,
        sun_az_deg: sun.horizontal.az_deg,
        moon_alt_deg: moon.horizontal.alt_deg,
        moon_az_deg: moon.horizontal.az_deg,
        moon: phase_from(&sun_body, &moon_body),
        twilight: twilight_for_sun_altitude(sun.horizontal.alt_deg),
    })
}

/// Fills the astronomical fields of a COC context (H14) from its `at` and
/// `site`. The Moon's illumination is geocentric and always filled; the Sun
/// and Moon altitudes and the twilight class need a valid site and stay
/// `None` without one ("site unknown").
pub fn fill_coc_context(ctx: &mut CocContext) {
    let at = ctx.at;
    ctx.moon_illumination = Some(moon_phase(at).illuminated_fraction);
    match ctx.site.as_ref().and_then(|site| sky_conditions(site, at)) {
        Some(sky) => {
            ctx.sun_alt_deg = Some(sky.sun_alt_deg);
            ctx.moon_alt_deg = Some(sky.moon_alt_deg);
            ctx.twilight = Some(sky.twilight);
        }
        None => {
            ctx.sun_alt_deg = None;
            ctx.moon_alt_deg = None;
            ctx.twilight = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::time::tests::utc;
    use super::*;
    use crate::equipment::types::SiteSource;

    fn site(lat: f64, lon: f64, elev_m: f64) -> Site {
        Site {
            lat_deg: lat,
            lon_deg: lon,
            elev_m,
            source: SiteSource::Config,
        }
    }

    #[test]
    fn sun_matches_meeus_example_25a() {
        // 1992 October 13.0 TD: α = 13h13m31.4s (198.38083°),
        // δ = −7°47′06″ (−7.78507°), R = 0.99766 AU, apparent λ = 199.90895°.
        let ts = TimeScales::from_jd_tt(2_448_908.5, 59.184);
        let s = sun_at(&ts);
        assert!(
            (s.apparent.ra_h * 15.0 - 198.380_83).abs() < 0.001,
            "{}",
            s.apparent.ra_h * 15.0
        );
        assert!(
            (s.apparent.dec_deg + 7.785_07).abs() < 0.001,
            "{}",
            s.apparent.dec_deg
        );
        assert!((s.ecliptic_lon_deg - 199.908_95).abs() < 0.001);
        assert!((s.distance_km / AU_KM - 0.997_66).abs() < 1e-5);
    }

    #[test]
    fn moon_series_matches_meeus_example_47a() {
        // 1992 April 12, 0h TD: Σl = −1127527, Σb = −3229126, Σr = −16590875.
        let ts = TimeScales::from_jd_tt(2_448_724.5, 58.184);
        let (sl, sb, sr, _) = moon_sums(ts.tt_centuries());
        assert!((sl + 1_127_527.0).abs() < 2.0, "{sl}");
        assert!((sb + 3_229_126.0).abs() < 2.0, "{sb}");
        assert!((sr + 16_590_875.0).abs() < 2.0, "{sr}");
        // λ = 133.162655° + Δψ, β = −3.229126°, Δ = 368409.7 km,
        // apparent α = 134.688470°, δ = 13.768368°.
        let m = moon_at(&ts);
        assert!((m.ecliptic_lat_deg + 3.229_126).abs() < 1e-5);
        assert!((m.distance_km - 368_409.7).abs() < 0.2);
        assert!(
            (m.ecliptic_lon_deg - 133.167_265).abs() < 1e-4,
            "{}",
            m.ecliptic_lon_deg
        );
        assert!(
            (m.apparent.ra_h * 15.0 - 134.688_470).abs() < 1e-4,
            "{}",
            m.apparent.ra_h * 15.0
        );
        assert!(
            (m.apparent.dec_deg - 13.768_368).abs() < 1e-4,
            "{}",
            m.apparent.dec_deg
        );
    }

    #[test]
    fn moon_phase_matches_meeus_example_48a() {
        // 1992 April 12, 0h TD: k = 0.6786, waxing gibbous.
        let ts = TimeScales::from_jd_tt(2_448_724.5, 58.184);
        let p = moon_phase_at(&ts);
        assert!(
            (p.illuminated_fraction - 0.6786).abs() < 0.001,
            "{}",
            p.illuminated_fraction
        );
        assert!(p.waxing);
        assert!(
            (p.phase_angle_deg - 69.0756).abs() < 0.1,
            "{}",
            p.phase_angle_deg
        );
    }

    #[test]
    fn twilight_classes_follow_the_thresholds() {
        assert_eq!(twilight_for_sun_altitude(10.0), Twilight::Day);
        assert_eq!(twilight_for_sun_altitude(-0.833), Twilight::Day);
        assert_eq!(twilight_for_sun_altitude(-0.9), Twilight::Civil);
        assert_eq!(twilight_for_sun_altitude(-6.0), Twilight::Civil);
        assert_eq!(twilight_for_sun_altitude(-6.1), Twilight::Nautical);
        assert_eq!(twilight_for_sun_altitude(-12.5), Twilight::Astronomical);
        assert_eq!(twilight_for_sun_altitude(-18.01), Twilight::Night);
    }

    #[test]
    fn night_times_are_ordered_and_hit_their_thresholds() {
        let s = site(40.0, -75.0, 100.0);
        let from = utc(2026, 10, 26, 18, 0, 0.0);
        let night = night_times(&s, from).unwrap();
        let order = [
            (night.sunset, SUNRISE_ALT_DEG),
            (night.civil_dusk, CIVIL_TWILIGHT_ALT_DEG),
            (night.nautical_dusk, NAUTICAL_TWILIGHT_ALT_DEG),
            (night.astronomical_dusk, ASTRONOMICAL_TWILIGHT_ALT_DEG),
            (night.astronomical_dawn, ASTRONOMICAL_TWILIGHT_ALT_DEG),
            (night.nautical_dawn, NAUTICAL_TWILIGHT_ALT_DEG),
            (night.civil_dawn, CIVIL_TWILIGHT_ALT_DEG),
            (night.sunrise, SUNRISE_ALT_DEG),
        ];
        let mut previous = None;
        for (at, threshold) in order {
            let at = at.expect("every event happens at 40° N in October");
            let alt = sun_alt(&s, at).unwrap();
            assert!((alt - threshold).abs() < 0.01, "{at}: {alt} vs {threshold}");
            if let Some(p) = previous {
                assert!(at > p);
            }
            previous = Some(at);
        }
        // Sunset near 22:00 UTC (18:00 EDT); sunrise near 11:30 UTC.
        let sunset = night.sunset.unwrap();
        assert!(
            (sunset - utc(2026, 10, 26, 22, 0, 0.0)).num_minutes().abs() < 30,
            "{sunset}"
        );
        // Asking from the middle of that night returns the same night (the
        // searches sample different grids, so the bisected instants agree
        // to about a second).
        let mid = night.astronomical_dusk.unwrap() + chrono::Duration::hours(2);
        let again = night_times(&s, mid).unwrap();
        let same = |a: Option<DateTime<Utc>>, b: Option<DateTime<Utc>>| {
            (a.unwrap() - b.unwrap()).num_milliseconds().abs() <= 2_000
        };
        assert!(same(again.sunset, night.sunset));
        assert!(same(again.astronomical_dawn, night.astronomical_dawn));
        assert!(same(again.sunrise, night.sunrise));
    }

    #[test]
    fn polar_summer_has_no_dark_night() {
        // 70° N at the June solstice: the Sun never sets.
        let s = site(70.0, 20.0, 0.0);
        let night = night_times(&s, utc(2026, 6, 21, 12, 0, 0.0)).unwrap();
        assert_eq!(night.sunset, None);
        assert_eq!(night.astronomical_dusk, None);
        assert_eq!(night.sunrise, None);
        // 60° N in June: sunset and sunrise, but no astronomical darkness.
        let s = site(60.0, 25.0, 0.0);
        let night = night_times(&s, utc(2026, 6, 21, 12, 0, 0.0)).unwrap();
        assert!(night.sunset.is_some() && night.sunrise.is_some());
        assert_eq!(night.astronomical_dusk, None);
        assert_eq!(night.astronomical_dawn, None);
    }

    #[test]
    fn moon_separation_and_parallax() {
        let at = utc(2026, 10, 26, 3, 0, 0.0);
        let ts = TimeScales::from_utc(at);
        let geo = moon_at(&ts);
        // The Moon is zero degrees from itself (geocentric).
        let sep = moon_separation_deg(geo.apparent, None, at).unwrap();
        assert!(sep < 1e-6, "{sep}");
        // From a site the topocentric Moon is displaced by the parallax,
        // at most asin(R⊕/Δ) ≈ 0.9-1.0°.
        let s = site(40.0, -75.0, 0.0);
        let topo = moon_separation_deg(geo.apparent, Some(&s), at).unwrap();
        let max_parallax = (EARTH_RADIUS_KM / geo.distance_km).asin().to_degrees();
        assert!(
            topo > 0.0 && topo <= max_parallax + 1e-6,
            "{topo} vs {max_parallax}"
        );
        assert!(moon_separation_deg(EqCoord::new(1.0, 1.0, EqFrame::Unknown), None, at).is_none());
    }

    #[test]
    fn coc_context_is_filled_only_with_a_site() {
        let at = utc(2026, 10, 26, 3, 0, 0.0);
        let mut ctx = CocContext {
            at,
            ..CocContext::default()
        };
        fill_coc_context(&mut ctx);
        assert!(ctx.moon_illumination.is_some());
        assert_eq!(
            (ctx.sun_alt_deg, ctx.moon_alt_deg, ctx.twilight),
            (None, None, None)
        );
        ctx.site = Some(site(40.0, -75.0, 100.0));
        fill_coc_context(&mut ctx);
        let sky = sky_conditions(ctx.site.as_ref().unwrap(), at).unwrap();
        assert_eq!(ctx.sun_alt_deg, Some(sky.sun_alt_deg));
        assert_eq!(ctx.moon_alt_deg, Some(sky.moon_alt_deg));
        assert_eq!(ctx.twilight, Some(sky.twilight));
        // 03:00 UTC is 23:00 EDT: night at 40° N.
        assert_eq!(ctx.twilight, Some(Twilight::Night));
        ctx.site = Some(site(f64::NAN, 0.0, 0.0));
        fill_coc_context(&mut ctx);
        assert_eq!(ctx.sun_alt_deg, None);
    }

    // Reference values computed with ERFA 2.0.1 (BSD), used only as numeric
    // test vectors (§6). Sun: `epv00` with `ab` aberration; Moon: `moon98`;
    // both rotated with `pnm06a`. Topocentric: the WGS84 observer from
    // `gd2gc`, the celestial-to-terrestrial matrix `c2t06a` (UT1 = UTC, no
    // polar motion), geometric altitude about the geodetic vertical.

    type Instant = (i32, u32, u32, u32, u32, f64);

    /// (instant, Sun RA h, Dec °, R au, Moon RA h, Dec °, Δ km).
    #[rustfmt::skip]
    const GEOCENTRIC: [(Instant, [f64; 3], [f64; 3]); 6] = [
        ((2026, 10, 26, 3, 15, 42.5), [14.034_741_968, -12.416_457_113, 0.994_216_262], [1.892_386_072, 16.510_911_244, 369_005.554]),
        ((2000, 1, 1, 12, 0, 0.0), [18.751_892_639, -23.032_429_624, 0.983_327_666], [14.830_167_323, -10.900_315_855, 402_446.874]),
        ((2045, 6, 30, 23, 59, 59.0), [6.695_823_527, 23.088_414_088, 1.016_625_360], [20.064_550_950, -22.619_366_898, 404_334.057]),
        ((1987, 4, 10, 19, 21, 0.0), [1.255_444_412, 7.968_243_345, 1.001_892_583], [10.860_655_073, 10.222_872_872, 392_535.433]),
        ((2012, 3, 20, 6, 0, 0.0), [0.001_918_905, 0.012_675_922, 0.995_970_648], [22.215_633_708, -5.673_363_282, 394_073.254]),
        ((2031, 12, 31, 18, 30, 0.0), [18.715_938_455, -23.066_231_442, 0.983_264_553], [8.933_843_858, 12.033_306_628, 406_185.245]),
    ];

    fn sep_deg(ra_h: f64, dec: f64, reference_ra_h: f64, reference_dec: f64) -> f64 {
        let a = EqCoord::new(ra_h, dec, EqFrame::JNowApparent);
        let b = EqCoord::new(reference_ra_h, reference_dec, EqFrame::JNowApparent);
        a.separation_arcmin(&b).unwrap() / 60.0
    }

    #[test]
    fn sun_and_moon_match_erfa_geocentric_places() {
        let (mut sun_worst, mut moon_worst) = (0.0_f64, 0.0_f64);
        for ((y, mo, d, h, mi, s), sun_ref, moon_ref) in GEOCENTRIC {
            let at = utc(y, mo, d, h, mi, s);
            let sun = sun(at);
            sun_worst = sun_worst.max(sep_deg(
                sun.apparent.ra_h,
                sun.apparent.dec_deg,
                sun_ref[0],
                sun_ref[1],
            ));
            assert!((sun.distance_km / AU_KM - sun_ref[2]).abs() < 1e-4, "{at}");
            let moon = moon(at);
            moon_worst = moon_worst.max(sep_deg(
                moon.apparent.ra_h,
                moon.apparent.dec_deg,
                moon_ref[0],
                moon_ref[1],
            ));
            assert!((moon.distance_km - moon_ref[2]).abs() < 30.0, "{at}");
        }
        // Meeus ch. 25 is good to about 0.01°; the ch. 47 series is the one
        // `moon98` also implements.
        assert!(sun_worst < 0.01, "Sun {sun_worst}°");
        assert!(moon_worst < 0.003, "Moon {moon_worst}°");
    }

    /// (instant, site, Sun alt/az, Moon alt/az), degrees.
    #[rustfmt::skip]
    const TOPOCENTRIC: [(Instant, &str, [f64; 2], [f64; 2]); 20] = [
        ((2026, 10, 26, 3, 15, 42.5), "philadelphia", [-56.205_755, 318.354_424], [60.283_790, 137.547_397]),
        ((2026, 10, 26, 3, 15, 42.5), "canberra", [59.663_484, 313.439_520], [-61.877_286, 124.813_253]),
        ((2026, 10, 26, 3, 15, 42.5), "tromso", [-17.893_997, 77.252_383], [20.066_739, 260.837_836]),
        ((2026, 10, 26, 3, 15, 42.5), "mauna_kea", [7.271_365, 253.953_659], [-4.803_893, 70.922_619]),
        ((2000, 1, 1, 12, 0, 0.0), "philadelphia", [-4.614_888, 116.389_157], [35.980_474, 158.811_358]),
        ((2000, 1, 1, 12, 0, 0.0), "canberra", [-24.411_704, 212.067_651], [-37.882_492, 145.806_638]),
        ((2000, 1, 1, 12, 0, 0.0), "tromso", [-3.596_263, 196.678_051], [-6.652_671, 254.047_023]),
        ((2000, 1, 1, 12, 0, 0.0), "mauna_kea", [-67.721_846, 102.548_027], [-11.499_606, 97.871_794]),
        ((2045, 6, 30, 23, 59, 59.0), "philadelphia", [4.758_417, 296.331_711], [-20.073_086, 103.837_887]),
        ((2045, 6, 30, 23, 59, 59.0), "canberra", [24.287_657, 32.182_755], [-14.629_330, 228.829_769]),
        ((2045, 6, 30, 23, 59, 59.0), "tromso", [3.630_979, 16.523_683], [-3.186_825, 177.636_227]),
        ((2045, 6, 30, 23, 59, 59.0), "mauna_kea", [67.864_742, 282.722_541], [-86.094_961, 135.367_639]),
        ((2012, 3, 20, 6, 0, 0.0), "philadelphia", [-48.318_814, 19.744_961], [-41.217_361, 56.878_968]),
        ((2012, 3, 20, 6, 0, 0.0), "canberra", [26.174_963, 290.371_391], [7.185_882, 268.761_646]),
        ((2012, 3, 20, 6, 0, 0.0), "tromso", [5.879_865, 106.088_591], [7.552_002, 133.533_610]),
        ((2012, 3, 20, 6, 0, 0.0), "mauna_kea", [-21.266_497, 278.077_883], [-48.778_883, 284.191_179]),
        ((2031, 12, 31, 18, 30, 0.0), "philadelphia", [23.840_400, 201.721_200], [-37.603_453, 345.665_610]),
        ((2031, 12, 31, 18, 30, 0.0), "canberra", [-4.625_294, 122.596_822], [32.755_625, 320.783_929]),
        ((2031, 12, 31, 18, 30, 0.0), "tromso", [-30.408_845, 286.014_827], [7.779_489, 78.730_760]),
        ((2031, 12, 31, 18, 30, 0.0), "mauna_kea", [18.456_352, 124.011_902], [4.989_074, 280.699_612]),
    ];

    fn named_site(name: &str) -> Site {
        match name {
            "philadelphia" => site(39.9526, -75.1652, 12.0),
            "canberra" => site(-35.2809, 149.13, 580.0),
            "tromso" => site(69.6492, 18.9553, 50.0),
            "mauna_kea" => site(19.8207, -155.468, 4205.0),
            other => panic!("unknown test site {other}"),
        }
    }

    #[test]
    fn sun_and_moon_altitudes_are_within_a_tenth_of_a_degree() {
        let (mut sun_worst, mut moon_worst) = (0.0_f64, 0.0_f64);
        for ((y, mo, d, h, mi, s), name, sun_ref, moon_ref) in TOPOCENTRIC {
            let at = utc(y, mo, d, h, mi, s);
            let place = named_site(name);
            let sun = sun_topocentric(&place, at).unwrap().horizontal;
            let moon = moon_topocentric(&place, at).unwrap().horizontal;
            sun_worst = sun_worst.max((sun.alt_deg - sun_ref[0]).abs());
            moon_worst = moon_worst.max((moon.alt_deg - moon_ref[0]).abs());
            // Azimuth too, on the sky (scaled by cos alt).
            for (ours, reference) in [(sun, sun_ref), (moon, moon_ref)] {
                let daz = (ours.az_deg - reference[1] + 540.0).rem_euclid(360.0) - 180.0;
                let on_sky = daz.abs() * reference[0].to_radians().cos();
                assert!(on_sky < 0.1, "{at} {name}: azimuth off by {daz}°");
            }
        }
        // Acceptance (§5.3): Sun and Moon altitude < 0.1°.
        assert!(sun_worst < 0.1, "Sun altitude {sun_worst}°");
        assert!(moon_worst < 0.1, "Moon altitude {moon_worst}°");
        // In practice the low-precision series do much better.
        assert!(
            sun_worst < 0.01 && moon_worst < 0.01,
            "{sun_worst} {moon_worst}"
        );
    }
}

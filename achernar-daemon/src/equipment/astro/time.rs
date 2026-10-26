//! Time scales (WP1A-3): Julian dates, the leap-second table and ΔT
//! (UTC → TT), the Earth rotation angle and IAU 2006 sidereal time (GMST,
//! GAST, LST).
//!
//! Sources (formula definitions only):
//!
//! - ERA: IERS Conventions (2010) eq. 5.15.
//! - GMST: the IAU 2006 polynomial of Capitaine, Wallace & Chapront (2003),
//!   IERS Conventions (2010) eq. 5.32, as ERA plus a polynomial in TT.
//! - GAST: GMST plus the equation of the equinoxes, Δψ·cos εA plus the two
//!   largest complementary terms (IERS Conventions (2010) eq. 5.30 and
//!   table 5.2e; the omitted terms total below 0.05 mas, about 3 µs).
//! - TAI − UTC: the IERS Bulletin C leap-second history. No future leap
//!   second is guessed; a new announcement needs a new table row.
//! - ΔT before 1972: the historical values published in the Astronomical
//!   Almanac (Meeus, *Astronomical Algorithms*, table 10.A), interpolated
//!   linearly, and the Morrison & Stephenson parabola before 1900 (coarse).

use chrono::{DateTime, Datelike, NaiveDate, TimeZone, Utc};

use super::frames;
use super::{norm_rad, ARCSEC_TO_RAD};
use std::f64::consts::TAU;

/// JD of the J2000.0 epoch (2000-01-01 12:00 TT, or UT1 for UT1 dates).
pub const J2000_JD: f64 = 2_451_545.0;
pub const DAYS_PER_JULIAN_CENTURY: f64 = 36_525.0;
pub const SECONDS_PER_DAY: f64 = 86_400.0;
/// TT − TAI, exact by definition.
pub const TT_MINUS_TAI_S: f64 = 32.184;
/// Ratio of the sidereal to the solar (UT1) rate: one SI second of UT1 is
/// this many seconds of sidereal time.
pub const SIDEREAL_RATE: f64 = 1.002_737_909_350_795;

/// Unix time of 2000-01-01T12:00:00Z.
const UNIX_AT_J2000: i64 = 946_728_000;

/// `(year, month, TAI − UTC)`: the offset applies from 0h UTC on the first
/// day of that month (IERS Bulletin C).
const LEAP_SECONDS: [(i32, u32, f64); 28] = [
    (1972, 1, 10.0),
    (1972, 7, 11.0),
    (1973, 1, 12.0),
    (1974, 1, 13.0),
    (1975, 1, 14.0),
    (1976, 1, 15.0),
    (1977, 1, 16.0),
    (1978, 1, 17.0),
    (1979, 1, 18.0),
    (1980, 1, 19.0),
    (1981, 7, 20.0),
    (1982, 7, 21.0),
    (1983, 7, 22.0),
    (1985, 7, 23.0),
    (1988, 1, 24.0),
    (1990, 1, 25.0),
    (1991, 1, 26.0),
    (1992, 7, 27.0),
    (1993, 7, 28.0),
    (1994, 7, 29.0),
    (1996, 1, 30.0),
    (1997, 7, 31.0),
    (1999, 1, 32.0),
    (2006, 1, 33.0),
    (2009, 1, 34.0),
    (2012, 7, 35.0),
    (2015, 7, 36.0),
    (2017, 1, 37.0),
];

/// `(decimal year, ΔT = TT − UT in seconds)` for 1900-1972.
const DELTA_T_TABLE: [(f64, f64); 16] = [
    (1900.0, -2.72),
    (1905.0, 3.86),
    (1910.0, 10.46),
    (1915.0, 17.20),
    (1920.0, 21.16),
    (1925.0, 23.62),
    (1930.0, 24.02),
    (1935.0, 23.93),
    (1940.0, 24.33),
    (1945.0, 26.77),
    (1950.0, 29.15),
    (1955.0, 31.07),
    (1960.0, 33.15),
    (1965.0, 35.73),
    (1970.0, 40.18),
    (1972.0, 42.23),
];

// ---------------------------------------------------------------------------
// Julian dates
// ---------------------------------------------------------------------------

/// Days since 2000-01-01T12:00:00 UTC (a UTC-scale offset from J2000).
pub fn days_since_j2000_utc(utc: DateTime<Utc>) -> f64 {
    let secs = (utc.timestamp() - UNIX_AT_J2000) as f64;
    let nanos = f64::from(utc.timestamp_subsec_nanos());
    (secs + nanos * 1e-9) / SECONDS_PER_DAY
}

/// Julian date of a UTC instant (on the UTC scale).
pub fn julian_date(utc: DateTime<Utc>) -> f64 {
    J2000_JD + days_since_j2000_utc(utc)
}

/// The UTC instant of a UTC-scale Julian date; `None` when out of range.
pub fn utc_from_julian_date(jd: f64) -> Option<DateTime<Utc>> {
    if !jd.is_finite() {
        return None;
    }
    let secs = (jd - J2000_JD) * SECONDS_PER_DAY;
    if secs.abs() > 1e13 {
        return None;
    }
    let mut whole = secs.floor();
    let mut nanos = ((secs - whole) * 1e9).round();
    if nanos >= 1e9 {
        whole += 1.0;
        nanos = 0.0;
    }
    Utc.timestamp_opt(UNIX_AT_J2000 + whole as i64, nanos as u32)
        .single()
}

/// Year with fraction, e.g. 2026.82 (used only to index the ΔT table).
pub fn decimal_year(utc: DateTime<Utc>) -> f64 {
    let year = utc.year();
    let start = |y: i32| {
        NaiveDate::from_ymd_opt(y, 1, 1)
            .and_then(|d| d.and_hms_opt(0, 0, 0))
            .map(|d| d.and_utc())
    };
    match (start(year), start(year + 1)) {
        (Some(a), Some(b)) => {
            let span = (b - a).num_milliseconds() as f64;
            f64::from(year) + (utc - a).num_milliseconds() as f64 / span
        }
        _ => f64::from(year),
    }
}

// ---------------------------------------------------------------------------
// UTC, TAI, TT and ΔT
// ---------------------------------------------------------------------------

/// TAI − UTC in seconds from the leap-second table, or `None` before 1972
/// (UTC before 1972 used rate offsets that this module does not model).
pub fn tai_minus_utc_s(utc: DateTime<Utc>) -> Option<f64> {
    let key = (utc.year(), utc.month());
    LEAP_SECONDS
        .iter()
        .rev()
        .find(|(y, m, _)| (*y, *m) <= key)
        .map(|(_, _, offset)| *offset)
}

/// ΔT = TT − UT from the historical table, for a decimal year before 1972.
/// After 1972 use [`tt_minus_utc_s`], which is exact.
pub fn delta_t_historical_s(year: f64) -> f64 {
    let first = DELTA_T_TABLE[0];
    let last = DELTA_T_TABLE[DELTA_T_TABLE.len() - 1];
    if year < first.0 {
        // Morrison & Stephenson (2004) long-term parabola; coarse.
        let u = (year - 1820.0) / 100.0;
        return -20.0 + 32.0 * u * u;
    }
    if year >= last.0 {
        return last.1;
    }
    DELTA_T_TABLE
        .windows(2)
        .find(|w| year >= w[0].0 && year < w[1].0)
        .map(|w| {
            let f = (year - w[0].0) / (w[1].0 - w[0].0);
            w[0].1 + f * (w[1].1 - w[0].1)
        })
        .unwrap_or(last.1)
}

/// TT − UTC in seconds: 32.184 s plus the leap seconds from 1972 on, and
/// the historical ΔT (with UT taken as UTC) before.
pub fn tt_minus_utc_s(utc: DateTime<Utc>) -> f64 {
    match tai_minus_utc_s(utc) {
        Some(leap) => TT_MINUS_TAI_S + leap,
        None => delta_t_historical_s(decimal_year(utc)),
    }
}

/// ΔT = TT − UT1 in seconds, taking DUT1 = UT1 − UTC as zero.
pub fn delta_t_s(utc: DateTime<Utc>) -> f64 {
    tt_minus_utc_s(utc)
}

/// One instant on the TT and UT1 scales, as day offsets from JD 2451545.0.
///
/// Two offsets keep full `f64` precision (about 1 µs over centuries)
/// instead of a single Julian date near 2.46 million.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeScales {
    /// Days since J2000.0 on the TT scale.
    pub tt_days: f64,
    /// Days since JD 2451545.0 on the UT1 scale.
    pub ut1_days: f64,
}

impl TimeScales {
    /// From UTC with DUT1 = 0 (UT1 taken as UTC; at most 0.9 s, about
    /// 13.5″ of hour angle).
    pub fn from_utc(utc: DateTime<Utc>) -> Self {
        Self::from_utc_with_dut1(utc, 0.0)
    }

    /// From UTC with a known DUT1 = UT1 − UTC (IERS Bulletin A), seconds.
    pub fn from_utc_with_dut1(utc: DateTime<Utc>, dut1_s: f64) -> Self {
        let utc_days = days_since_j2000_utc(utc);
        Self {
            tt_days: utc_days + tt_minus_utc_s(utc) / SECONDS_PER_DAY,
            ut1_days: utc_days + dut1_s / SECONDS_PER_DAY,
        }
    }

    /// From a TT (TD) Julian date and ΔT = TT − UT1 (worked examples are
    /// usually given in TD).
    pub fn from_jd_tt(jd_tt: f64, delta_t_s: f64) -> Self {
        let tt_days = jd_tt - J2000_JD;
        Self {
            tt_days,
            ut1_days: tt_days - delta_t_s / SECONDS_PER_DAY,
        }
    }

    /// From a UT1 Julian date and ΔT = TT − UT1.
    pub fn from_jd_ut1(jd_ut1: f64, delta_t_s: f64) -> Self {
        let ut1_days = jd_ut1 - J2000_JD;
        Self {
            tt_days: ut1_days + delta_t_s / SECONDS_PER_DAY,
            ut1_days,
        }
    }

    /// Julian centuries of TT since J2000.0 (the `t` of every IAU series).
    pub fn tt_centuries(&self) -> f64 {
        self.tt_days / DAYS_PER_JULIAN_CENTURY
    }

    pub fn jd_tt(&self) -> f64 {
        J2000_JD + self.tt_days
    }

    pub fn jd_ut1(&self) -> f64 {
        J2000_JD + self.ut1_days
    }

    /// TT − UT1, seconds.
    pub fn delta_t_s(&self) -> f64 {
        (self.tt_days - self.ut1_days) * SECONDS_PER_DAY
    }
}

// ---------------------------------------------------------------------------
// Earth rotation and sidereal time
// ---------------------------------------------------------------------------

/// Earth rotation angle in `[0, 2π)`, IERS Conventions (2010) eq. 5.15.
pub fn earth_rotation_angle_rad(ts: &TimeScales) -> f64 {
    let du = ts.ut1_days;
    // 1.00273781191135448·Du split so that whole days drop out exactly.
    let turns = du.rem_euclid(1.0) + 0.779_057_273_264_0 + 0.002_737_811_911_354_48 * du;
    norm_rad(TAU * turns)
}

/// Greenwich mean sidereal time (IAU 2006), radians in `[0, 2π)`.
pub fn gmst_rad(ts: &TimeScales) -> f64 {
    let t = ts.tt_centuries();
    let poly_arcsec = 0.014_506
        + t * (4_612.156_534
            + t * (1.391_581_7
                + t * (-0.000_000_44 + t * (-0.000_029_956 + t * -0.000_000_036_8))));
    norm_rad(earth_rotation_angle_rad(ts) + poly_arcsec * ARCSEC_TO_RAD)
}

/// The equation of the equinoxes (GAST − GMST), radians, consistent with the
/// IAU 2000B nutation and the IAU 2006 obliquity.
pub fn equation_of_equinoxes_rad(ts: &TimeScales) -> f64 {
    let t = ts.tt_centuries();
    let nutation = frames::nutation_2000b(t);
    let eps_a = frames::mean_obliquity_rad(t);
    // Mean longitude of the Moon's ascending node (linear part).
    let om = ((450_160.398_036 - 6_962_890.543_1 * t) % 1_296_000.0) * ARCSEC_TO_RAD;
    // Table 5.2e lists these in µas: 2640.96 µas sin Ω + 63.52 µas sin 2Ω.
    let complementary_arcsec = 0.002_640_96 * om.sin() + 0.000_063_52 * (2.0 * om).sin();
    nutation.dpsi_rad * eps_a.cos() + complementary_arcsec * ARCSEC_TO_RAD
}

/// Greenwich apparent sidereal time, radians in `[0, 2π)`.
pub fn gast_rad(ts: &TimeScales) -> f64 {
    norm_rad(gmst_rad(ts) + equation_of_equinoxes_rad(ts))
}

/// Local apparent sidereal time in hours `[0, 24)` (ASCOM `SiderealTime`)
/// for an east-positive longitude.
pub fn local_sidereal_time_from(ts: &TimeScales, lon_deg: f64) -> f64 {
    norm_rad(gast_rad(ts) + lon_deg.to_radians()).to_degrees() / 15.0
}

/// Local apparent sidereal time in hours `[0, 24)`, with DUT1 = 0.
pub fn local_sidereal_time_h(utc: DateTime<Utc>, lon_deg: f64) -> f64 {
    local_sidereal_time_from(&TimeScales::from_utc(utc), lon_deg)
}

/// Local mean sidereal time in hours `[0, 24)`, with DUT1 = 0.
pub fn local_mean_sidereal_time_h(utc: DateTime<Utc>, lon_deg: f64) -> f64 {
    let ts = TimeScales::from_utc(utc);
    norm_rad(gmst_rad(&ts) + lon_deg.to_radians()).to_degrees() / 15.0
}

/// GMST in hours `[0, 24)`, with DUT1 = 0.
pub fn gmst_hours(utc: DateTime<Utc>) -> f64 {
    gmst_rad(&TimeScales::from_utc(utc)).to_degrees() / 15.0
}

/// GAST in hours `[0, 24)`, with DUT1 = 0.
pub fn gast_hours(utc: DateTime<Utc>) -> f64 {
    gast_rad(&TimeScales::from_utc(utc)).to_degrees() / 15.0
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A UTC instant from calendar fields and fractional seconds.
    pub(crate) fn utc(y: i32, mo: u32, d: u32, h: u32, mi: u32, s: f64) -> DateTime<Utc> {
        let whole = s.floor();
        let nanos = ((s - whole) * 1e9).round() as u32;
        Utc.with_ymd_and_hms(y, mo, d, h, mi, whole as u32).unwrap()
            + chrono::Duration::nanoseconds(i64::from(nanos))
    }

    /// Seconds of time between two angles in hours, wrapped to ±12 h.
    fn dsec(a_h: f64, b_h: f64) -> f64 {
        super::super::wrap_hours_pm12(a_h - b_h) * 3600.0
    }

    fn hms(h: f64, m: f64, s: f64) -> f64 {
        h + m / 60.0 + s / 3600.0
    }

    #[test]
    fn julian_dates_match_meeus_chapter_7() {
        // Meeus example 7.a and the table after it (Gregorian dates).
        let cases = [
            (utc(2000, 1, 1, 12, 0, 0.0), 2_451_545.0),
            (utc(1999, 1, 1, 0, 0, 0.0), 2_451_179.5),
            (utc(1987, 1, 27, 0, 0, 0.0), 2_446_822.5),
            (utc(1987, 6, 19, 12, 0, 0.0), 2_446_966.0),
            (utc(1988, 1, 27, 0, 0, 0.0), 2_447_187.5),
            (utc(1988, 6, 19, 12, 0, 0.0), 2_447_332.0),
            (utc(1900, 1, 1, 0, 0, 0.0), 2_415_020.5),
            (utc(1600, 1, 1, 0, 0, 0.0), 2_305_447.5),
            (utc(1600, 12, 31, 0, 0, 0.0), 2_305_812.5),
            // 1957 October 4.81 (Sputnik 1).
            (utc(1957, 10, 4, 19, 26, 24.0), 2_436_116.31),
        ];
        for (at, jd) in cases {
            assert!(
                (julian_date(at) - jd).abs() < 1e-8,
                "{at}: {}",
                julian_date(at)
            );
            let back = utc_from_julian_date(jd).unwrap();
            assert!((back - at).num_milliseconds().abs() <= 1, "{back} vs {at}");
        }
        assert_eq!(utc_from_julian_date(f64::NAN), None);
        assert_eq!(utc_from_julian_date(1e12), None);
    }

    #[test]
    fn leap_seconds_follow_bulletin_c() {
        assert_eq!(tai_minus_utc_s(utc(1971, 12, 31, 23, 59, 59.0)), None);
        assert_eq!(tai_minus_utc_s(utc(1972, 1, 1, 0, 0, 0.0)), Some(10.0));
        assert_eq!(tai_minus_utc_s(utc(1972, 6, 30, 23, 59, 59.0)), Some(10.0));
        assert_eq!(tai_minus_utc_s(utc(1972, 7, 1, 0, 0, 0.0)), Some(11.0));
        assert_eq!(tai_minus_utc_s(utc(1987, 4, 10, 0, 0, 0.0)), Some(23.0));
        assert_eq!(tai_minus_utc_s(utc(2016, 12, 31, 23, 59, 59.0)), Some(36.0));
        assert_eq!(tai_minus_utc_s(utc(2017, 1, 1, 0, 0, 0.0)), Some(37.0));
        assert_eq!(tai_minus_utc_s(utc(2026, 10, 26, 0, 0, 0.0)), Some(37.0));
        assert_eq!(tt_minus_utc_s(utc(2026, 10, 26, 0, 0, 0.0)), 69.184);
        assert_eq!(tt_minus_utc_s(utc(1992, 10, 13, 0, 0, 0.0)), 59.184);
    }

    #[test]
    fn delta_t_table_is_continuous_with_leap_seconds() {
        assert!((delta_t_historical_s(1950.0) - 29.15).abs() < 1e-12);
        assert!((delta_t_historical_s(1952.5) - 30.11).abs() < 1e-9);
        // The table's last row meets TT − UTC = 42.184 s at 1972 within the
        // size of DUT1.
        let before = tt_minus_utc_s(utc(1971, 12, 31, 23, 0, 0.0));
        let after = tt_minus_utc_s(utc(1972, 1, 1, 0, 0, 0.0));
        assert!((before - after).abs() < 0.9, "{before} vs {after}");
        // Coarse parabola before 1900, still the right size.
        assert!((-20.0..20.0).contains(&delta_t_historical_s(1850.0)));
        assert!((decimal_year(utc(2026, 7, 2, 12, 0, 0.0)) - 2026.5).abs() < 0.01);
    }

    #[test]
    fn time_scales_offsets_are_consistent() {
        let at = utc(2026, 10, 26, 3, 15, 42.5);
        let ts = TimeScales::from_utc(at);
        assert!((ts.delta_t_s() - 69.184).abs() < 1e-6);
        assert!((ts.jd_ut1() - julian_date(at)).abs() < 1e-9);
        let with = TimeScales::from_utc_with_dut1(at, 0.25);
        assert!((with.delta_t_s() - 68.934).abs() < 1e-6);
        let tt = TimeScales::from_jd_tt(2_451_545.0, 64.0);
        assert_eq!(tt.tt_centuries(), 0.0);
        assert!((tt.jd_ut1() - (2_451_545.0 - 64.0 / 86_400.0)).abs() < 1e-12);
    }

    #[test]
    fn earth_rotation_angle_at_j2000() {
        let ts = TimeScales::from_jd_ut1(J2000_JD, 64.184);
        let era = earth_rotation_angle_rad(&ts).to_degrees();
        assert!((era - 280.460_618_375_04).abs() < 1e-9, "{era}");
    }

    #[test]
    fn gmst_matches_meeus_examples_12a_and_12b() {
        // 1987 April 10, 0h UT: 13h10m46.3668s (IAU 1982 GMST; IAU 2006
        // agrees to a few ms).
        let a = TimeScales::from_jd_ut1(2_446_895.5, 55.184);
        let gmst_a = gmst_rad(&a).to_degrees() / 15.0;
        assert!(
            dsec(gmst_a, hms(13.0, 10.0, 46.3668)).abs() < 0.01,
            "{gmst_a}"
        );
        // 1987 April 10, 19h21m00s UT: 8h34m57.0896s.
        let b = TimeScales::from_utc(utc(1987, 4, 10, 19, 21, 0.0));
        let gmst_b = gmst_rad(&b).to_degrees() / 15.0;
        assert!(
            dsec(gmst_b, hms(8.0, 34.0, 57.0896)).abs() < 0.01,
            "{gmst_b}"
        );
    }

    #[test]
    fn gast_matches_meeus_example_12a() {
        // Apparent sidereal time, 1987 April 10, 0h UT: 13h10m46.1351s.
        let a = TimeScales::from_jd_ut1(2_446_895.5, 55.184);
        let gast = gast_rad(&a).to_degrees() / 15.0;
        assert!(dsec(gast, hms(13.0, 10.0, 46.1351)).abs() < 0.01, "{gast}");
        // Equation of the equinoxes: Δψ cos ε = −3.788″ cos 23.44° = −0.2317 s.
        let ee_s = equation_of_equinoxes_rad(&a).to_degrees() / 15.0 * 3600.0;
        assert!((ee_s + 0.2317).abs() < 0.003, "{ee_s}");
    }

    /// A UTC instant as (year, month, day, hour, minute, second).
    type Instant = (i32, u32, u32, u32, u32, f64);

    /// GMST (`gmst06`) and GAST (`gst06a`) in hours, computed with ERFA 2.0.1
    /// (BSD) as numeric test vectors only (§6), UT1 = UTC.
    #[rustfmt::skip]
    const SIDEREAL: [(Instant, f64, f64); 6] = [
        ((2026, 10, 26, 3, 15, 42.5), 5.562_986_668_547, 5.563_117_053_203),
        ((2000, 1, 1, 12, 0, 0.0), 18.697_374_828_703, 18.697_138_157_369),
        ((2045, 6, 30, 23, 59, 59.0), 18.630_106_585_063, 18.630_281_542_295),
        ((1987, 4, 10, 19, 21, 0.0), 8.582_525_805_364, 8.582_460_229_081),
        ((2012, 3, 20, 6, 0, 0.0), 17.878_183_691_233, 17.878_464_827_139),
        ((2031, 12, 31, 18, 30, 0.0), 1.165_890_229_665, 1.166_115_074_267),
    ];

    #[test]
    fn sidereal_time_matches_erfa() {
        for ((y, mo, d, h, mi, s), gmst_ref, gast_ref) in SIDEREAL {
            let at = utc(y, mo, d, h, mi, s);
            // The same IAU 2006 GMST definition: microsecond agreement.
            assert!(dsec(gmst_hours(at), gmst_ref).abs() < 1e-5, "{at}");
            // GAST: IAU 2000B against 2000A nutation, a few tens of µs.
            // The complementary terms alone are worth up to 0.18 ms, so
            // this bound catches them going missing.
            assert!(dsec(gast_hours(at), gast_ref).abs() < 1e-4, "{at}");
            // LST for an east-positive longitude (acceptance: < 0.1 s).
            for lon in [-75.1652, 0.0, 149.13, 179.9] {
                let expected = (gast_ref + lon / 15.0).rem_euclid(24.0);
                let lst = local_sidereal_time_h(at, lon);
                assert!((0.0..24.0).contains(&lst));
                assert!(dsec(lst, expected).abs() < 0.1, "{at} {lon}");
                assert!(dsec(lst, expected).abs() < 1e-3, "{at} {lon}");
                let lmst = local_mean_sidereal_time_h(at, lon);
                assert!(dsec(lmst, gmst_ref + lon / 15.0).abs() < 1e-5);
            }
        }
    }
}

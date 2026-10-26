//! Hour angle, alt/az and meridian timing (WP1A-3).
//!
//! Serves the mount limits and the tracking watchdog (H13, M-k), pointing
//! samples (M-4: HA, LST, alt/az), the meridian flip timing (M-l: "time to
//! flip = (HA_limit − HA) in sidereal time converted to SI"), and the
//! planners' visibility inputs (FOR §7.2.8, FW-3 §7.2.14, planning
//! §7.2.21).
//!
//! Positions are geometric (no refraction) unless a function says
//! otherwise; [`Horizontal::refracted_alt_deg`] adds standard refraction.
//! Azimuth runs from north through east. Hour angles are in hours, positive
//! west of the meridian, wrapped to `[-12, 12)`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::frames::FrameContext;
use super::horizon::HorizonProfile;
use super::refraction::{self, Atmosphere};
use super::time::{self, TimeScales, SIDEREAL_RATE};
use super::{find_crossings, norm_deg, wrap_hours_pm12};
use crate::equipment::types::{AltAz, EqCoord, EqFrame, PierSide, Site};

/// Sample spacing for rise, set and visibility searches.
pub const SEARCH_STEP_MINUTES: i64 = 10;

/// Hour angle `LST − RA`, hours in `[-12, 12)`.
pub fn hour_angle_h(lst_h: f64, ra_h: f64) -> f64 {
    wrap_hours_pm12(lst_h - ra_h)
}

/// Geometric altitude and azimuth of an hour angle and declination at a
/// latitude.
pub fn altaz_from_hadec(ha_h: f64, dec_deg: f64, lat_deg: f64) -> AltAz {
    let h = (ha_h * 15.0).to_radians();
    let (sd, cd) = dec_deg.to_radians().sin_cos();
    let (sp, cp) = lat_deg.to_radians().sin_cos();
    let (sh, ch) = h.sin_cos();
    let x = -cd * sh;
    let y = sd * cp - cd * ch * sp;
    let z = sd * sp + cd * ch * cp;
    AltAz {
        alt_deg: z.atan2(x.hypot(y)).to_degrees(),
        az_deg: norm_deg(x.atan2(y).to_degrees()),
    }
}

/// Hour angle (hours) and declination (degrees) of an alt/az position at a
/// latitude: the inverse of [`altaz_from_hadec`].
pub fn hadec_from_altaz(position: AltAz, lat_deg: f64) -> (f64, f64) {
    let (sa, ca) = position.alt_deg.to_radians().sin_cos();
    let (sz, cz) = position.az_deg.to_radians().sin_cos();
    let (sp, cp) = lat_deg.to_radians().sin_cos();
    let x = -sz * ca;
    let y = cp * sa - sp * ca * cz;
    let z = sp * sa + cp * ca * cz;
    let ha = wrap_hours_pm12(x.atan2(y).to_degrees() / 15.0);
    (ha, z.atan2(x.hypot(y)).to_degrees())
}

/// SI seconds in a span of sidereal hours.
pub fn sidereal_hours_to_si_seconds(sidereal_h: f64) -> f64 {
    sidereal_h * 3600.0 / SIDEREAL_RATE
}

/// The hour angle `elapsed_s` SI seconds after a sample at `ha_h`: the
/// prediction the limit watchdog uses when mount data goes stale (M-k).
pub fn predict_hour_angle_h(ha_h: f64, elapsed_s: f64) -> f64 {
    wrap_hours_pm12(ha_h + elapsed_s * SIDEREAL_RATE / 3600.0)
}

/// SI seconds until the hour angle next reaches `target_ha_h`, in
/// `[0, one sidereal day)`.
pub fn seconds_until_hour_angle(ha_h: f64, target_ha_h: f64) -> f64 {
    sidereal_hours_to_si_seconds((target_ha_h - ha_h).rem_euclid(24.0))
}

/// Signed SI minutes since the upper meridian transit: negative before it,
/// positive after, in `(-718, 718]`.
pub fn minutes_past_meridian(ha_h: f64) -> f64 {
    sidereal_hours_to_si_seconds(wrap_hours_pm12(ha_h)) / 60.0
}

/// The ASCOM pointing state a German equatorial mount takes for a target at
/// `ha_h` in its normal (counterweight-down) configuration: `East` (looking
/// west) when the target is west of the meridian, `West` (looking east)
/// when it is east. INDI mounts derive their destination pier side from
/// this (§7.1 R6); for other mount types the answer means nothing. A
/// non-finite hour angle gives `Unknown`, never a guessed side (D5).
pub fn expected_pier_side(ha_h: f64) -> PierSide {
    let ha = wrap_hours_pm12(ha_h);
    if !ha.is_finite() {
        PierSide::Unknown
    } else if ha >= 0.0 {
        PierSide::East
    } else {
        PierSide::West
    }
}

/// Meridian timing for one hour angle.
///
/// A non-finite hour angle leaves the three time fields NaN; with a limit
/// it also counts as past the limit (see [`meridian_timing`]).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeridianTiming {
    pub ha_h: f64,
    pub minutes_past_meridian: f64,
    /// SI seconds to the next upper transit.
    pub seconds_to_meridian: f64,
    /// SI seconds until the hour angle reaches the limit, `None` without
    /// one or when the hour angle or the limit is not finite. Zero or more;
    /// a target already past the limit wraps to the next sidereal day, so
    /// check [`Self::past_limit`] first.
    pub seconds_to_limit: Option<f64>,
    /// The hour angle (in `[-12, 12)`) is beyond the limit, or a limit is
    /// set and the hour angle or the limit is not finite (fail closed).
    pub past_limit: bool,
}

/// Meridian timing at `ha_h`, with an optional hour-angle limit (east of
/// the meridian is negative), e.g. a flip limit in hours past the meridian.
/// An hour angle or limit that is not finite is unknown: with a limit the
/// result reads past it with no time to it, never "within the limit".
pub fn meridian_timing(ha_h: f64, ha_limit_h: Option<f64>) -> MeridianTiming {
    let ha = wrap_hours_pm12(ha_h);
    let known_limit = ha_limit_h.filter(|limit| ha.is_finite() && limit.is_finite());
    MeridianTiming {
        ha_h: ha,
        minutes_past_meridian: minutes_past_meridian(ha),
        seconds_to_meridian: seconds_until_hour_angle(ha, 0.0),
        seconds_to_limit: known_limit.map(|limit| seconds_until_hour_angle(ha, limit)),
        past_limit: match (ha_limit_h, known_limit) {
            (None, _) => false,
            (Some(_), None) => true,
            (Some(_), Some(limit)) => ha > limit,
        },
    }
}

// ---------------------------------------------------------------------------
// Positions of a coordinate at a site
// ---------------------------------------------------------------------------

/// A coordinate's position at a site and instant.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Horizontal {
    /// Geometric altitude, degrees.
    pub alt_deg: f64,
    pub az_deg: f64,
    pub ha_h: f64,
    /// Local apparent sidereal time, hours.
    pub lst_h: f64,
    /// The coordinate the position was computed from, in
    /// [`EqFrame::JNowApparent`].
    pub apparent: EqCoord,
}

impl Horizontal {
    pub fn altaz(&self) -> AltAz {
        AltAz {
            alt_deg: self.alt_deg,
            az_deg: self.az_deg,
        }
    }

    /// Apparent (refracted) altitude; standard atmosphere when `None`.
    pub fn refracted_alt_deg(&self, atmosphere: Option<Atmosphere>) -> f64 {
        refraction::apparent_from_true_deg(self.alt_deg, atmosphere)
    }

    /// Kasten-Young airmass, `None` below the horizon.
    pub fn airmass(&self) -> Option<f64> {
        refraction::airmass(self.alt_deg)
    }
}

fn site_usable(site: &Site) -> bool {
    site.validate().is_ok()
}

/// Position of a coordinate already in the apparent frame, given the LST.
fn horizontal_from_apparent(apparent: EqCoord, site: &Site, lst_h: f64) -> Horizontal {
    let ha = hour_angle_h(lst_h, apparent.ra_h);
    let position = altaz_from_hadec(ha, apparent.dec_deg, site.lat_deg);
    Horizontal {
        alt_deg: position.alt_deg,
        az_deg: position.az_deg,
        ha_h: ha,
        lst_h,
        apparent,
    }
}

/// Position at one instant with a precomputed frame context.
pub fn horizontal_at(
    c: EqCoord,
    site: &Site,
    ts: &TimeScales,
    ctx: &FrameContext,
) -> Option<Horizontal> {
    if !site_usable(site) {
        return None;
    }
    let apparent = ctx.convert(c, EqFrame::JNowApparent).ok()?;
    let lst = time::local_sidereal_time_from(ts, site.lon_deg);
    Some(horizontal_from_apparent(apparent, site, lst))
}

/// Position of `c` (any known frame) at a site; `None` for an unknown frame,
/// a non-finite coordinate or an invalid site.
pub fn horizontal(c: EqCoord, site: &Site, at: DateTime<Utc>) -> Option<Horizontal> {
    let ts = TimeScales::from_utc(at);
    horizontal_at(c, site, &ts, &FrameContext::new(&ts))
}

/// Everything a model or a FITS header wants about one target at one
/// instant (H1 `CENTALT`, `CENTAZ`, `AIRMASS`; M-3 airmass and HA; M-4).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TargetGeometry {
    pub horizontal: Horizontal,
    /// Altitude with standard refraction.
    pub refracted_alt_deg: f64,
    pub airmass: Option<f64>,
    pub timing: MeridianTiming,
    /// For German equatorial mounts only (see [`expected_pier_side`]).
    pub expected_pier_side: PierSide,
    /// Altitude above the horizon profile, when one is given.
    pub horizon_clearance_deg: Option<f64>,
}

/// Geometry of a target; `None` when the position cannot be computed.
pub fn target_geometry(
    c: EqCoord,
    site: &Site,
    at: DateTime<Utc>,
    horizon: Option<&HorizonProfile>,
) -> Option<TargetGeometry> {
    let h = horizontal(c, site, at)?;
    Some(TargetGeometry {
        horizontal: h,
        refracted_alt_deg: h.refracted_alt_deg(None),
        airmass: h.airmass(),
        timing: meridian_timing(h.ha_h, None),
        expected_pier_side: expected_pier_side(h.ha_h),
        horizon_clearance_deg: horizon.map(|p| p.clearance_deg(h.altaz())),
    })
}

// ---------------------------------------------------------------------------
// Time searches
// ---------------------------------------------------------------------------

/// A target's apparent coordinate and an altitude function over a window.
/// The apparent place is computed once: it moves by well under 1″ per day
/// for a fixed target.
struct Tracker {
    apparent: EqCoord,
    site: Site,
}

impl Tracker {
    fn new(c: EqCoord, site: &Site, at: DateTime<Utc>) -> Option<Self> {
        if !site_usable(site) {
            return None;
        }
        let apparent = FrameContext::at(at)
            .convert(c, EqFrame::JNowApparent)
            .ok()?;
        Some(Self {
            apparent,
            site: *site,
        })
    }

    fn at(&self, t: DateTime<Utc>) -> Horizontal {
        let lst = time::local_sidereal_time_h(t, self.site.lon_deg);
        horizontal_from_apparent(self.apparent, &self.site, lst)
    }
}

fn step() -> chrono::Duration {
    chrono::Duration::minutes(SEARCH_STEP_MINUTES)
}

fn si_duration(seconds: f64) -> chrono::Duration {
    chrono::Duration::milliseconds((seconds * 1000.0).round() as i64)
}

/// The first upper transit (HA = 0) at or after `from`.
pub fn next_transit(c: EqCoord, site: &Site, from: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let tracker = Tracker::new(c, site, from)?;
    let mut t = from + si_duration(seconds_until_hour_angle(tracker.at(from).ha_h, 0.0));
    // One correction absorbs the sidereal-time polynomial; HA then sits
    // within milliseconds of zero.
    let ha = tracker.at(t).ha_h;
    t -= si_duration(sidereal_hours_to_si_seconds(ha));
    Some(t)
}

/// Rise, transit and set of a target around one sidereal day.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RiseTransitSet {
    /// First rise through the threshold after `from`.
    pub rise: Option<DateTime<Utc>>,
    pub transit: DateTime<Utc>,
    /// Geometric altitude at that transit.
    pub transit_alt_deg: f64,
    /// First set through the threshold after `from`.
    pub set: Option<DateTime<Utc>>,
    /// Above the threshold for the whole day.
    pub always_above: bool,
    /// Below the threshold for the whole day.
    pub never_above: bool,
}

/// Rise, transit and set within one sidereal day of `from`, against a
/// geometric altitude threshold (0° for the horizon, or a minimum
/// altitude).
pub fn rise_transit_set(
    c: EqCoord,
    site: &Site,
    from: DateTime<Utc>,
    threshold_alt_deg: f64,
) -> Option<RiseTransitSet> {
    let tracker = Tracker::new(c, site, from)?;
    let to = from + si_duration(sidereal_hours_to_si_seconds(24.0));
    let crossings = find_crossings(
        |t| Some(tracker.at(t).alt_deg),
        threshold_alt_deg,
        from,
        to,
        step(),
    );
    let transit = next_transit(c, site, from)?;
    let transit_alt_deg = tracker.at(transit).alt_deg;
    let above_now = tracker.at(from).alt_deg >= threshold_alt_deg;
    Some(RiseTransitSet {
        rise: crossings.iter().find(|x| x.rising).map(|x| x.at),
        transit,
        transit_alt_deg,
        set: crossings.iter().find(|x| !x.rising).map(|x| x.at),
        always_above: crossings.is_empty() && above_now,
        never_above: crossings.is_empty() && !above_now,
    })
}

/// Intervals in `[from, to]` during which the target is at or above both
/// `min_alt_deg` and the horizon profile (FW-3 and planning visibility).
pub fn visibility_intervals(
    c: EqCoord,
    site: &Site,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    min_alt_deg: f64,
    horizon: Option<&HorizonProfile>,
) -> Vec<(DateTime<Utc>, DateTime<Utc>)> {
    let Some(tracker) = Tracker::new(c, site, from) else {
        return Vec::new();
    };
    let clearance = |t: DateTime<Utc>| {
        let h = tracker.at(t);
        let above_min = h.alt_deg - min_alt_deg;
        Some(match horizon {
            Some(p) => above_min.min(p.clearance_deg(h.altaz())),
            None => above_min,
        })
    };
    let mut out = Vec::new();
    let mut open = clearance(from).is_some_and(|v| v >= 0.0).then_some(from);
    for x in find_crossings(clearance, 0.0, from, to, step()) {
        match (x.rising, open) {
            (true, None) => open = Some(x.at),
            (false, Some(start)) => {
                out.push((start, x.at));
                open = None;
            }
            _ => {}
        }
    }
    if let Some(start) = open {
        out.push((start, to));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::time::tests::utc;
    use super::*;
    use crate::equipment::types::SiteSource;

    fn site(lat: f64, lon: f64) -> Site {
        Site {
            lat_deg: lat,
            lon_deg: lon,
            elev_m: 0.0,
            source: SiteSource::Config,
        }
    }

    #[test]
    fn hour_angle_wraps() {
        assert_eq!(hour_angle_h(1.0, 23.0), 2.0);
        assert_eq!(hour_angle_h(23.0, 1.0), -2.0);
        assert_eq!(hour_angle_h(12.0, 0.0), -12.0);
    }

    #[test]
    fn altaz_matches_meeus_example_13b() {
        // Venus from the USNO (φ = +38°55′17″), 1987 April 10, 19h21m UT:
        // H = 64.352133°, δ = −6°43′11.61″ → A = 68.0337° from the south,
        // h = 15.1249°.
        let lat = 38.0 + 55.0 / 60.0 + 17.0 / 3600.0;
        let dec = -(6.0 + 43.0 / 60.0 + 11.61 / 3600.0);
        let p = altaz_from_hadec(64.352_133 / 15.0, dec, lat);
        assert!((p.alt_deg - 15.1249).abs() < 1e-4, "{}", p.alt_deg);
        assert!((p.az_deg - (68.0337 + 180.0)).abs() < 1e-4, "{}", p.az_deg);
        let (ha, d) = hadec_from_altaz(p, lat);
        assert!((ha * 15.0 - 64.352_133).abs() < 1e-9 && (d - dec).abs() < 1e-9);
    }

    #[test]
    fn altaz_cardinal_points() {
        // On the meridian south of the zenith: due south.
        let p = altaz_from_hadec(0.0, 10.0, 40.0);
        assert!((p.az_deg - 180.0).abs() < 1e-9 && (p.alt_deg - 60.0).abs() < 1e-9);
        // Six hours west on the equator: due west on the horizon.
        let p = altaz_from_hadec(6.0, 0.0, 40.0);
        assert!((p.az_deg - 270.0).abs() < 1e-9 && p.alt_deg.abs() < 1e-9);
        // Southern hemisphere: a southern circumpolar star at lower transit.
        let p = altaz_from_hadec(-12.0, -80.0, -35.0);
        assert!((p.az_deg - 180.0).abs() < 1e-9 && (p.alt_deg - 25.0).abs() < 1e-9);
        // The celestial pole sits at the latitude, due north.
        let p = altaz_from_hadec(3.0, 90.0, 52.0);
        assert!((p.alt_deg - 52.0).abs() < 1e-9);
    }

    #[test]
    fn meridian_timing_converts_sidereal_to_si() {
        // One sidereal hour before transit is 3590.17 SI seconds.
        let t = meridian_timing(-1.0, Some(0.25));
        assert!((t.seconds_to_meridian - 3590.170).abs() < 0.01);
        assert!((t.minutes_past_meridian + 59.836).abs() < 0.01);
        let to_limit = t.seconds_to_limit.unwrap();
        assert!((to_limit - sidereal_hours_to_si_seconds(1.25)).abs() < 1e-9);
        assert!(!t.past_limit);
        let past = meridian_timing(0.5, Some(0.25));
        assert!(past.past_limit && past.minutes_past_meridian > 0.0);
        // The watchdog's blind prediction advances at the sidereal rate.
        let predicted = predict_hour_angle_h(-0.5, 1800.0);
        assert!((predicted - (-0.5 + 0.5 * SIDEREAL_RATE)).abs() < 1e-12);
        assert!(
            (predict_hour_angle_h(11.9, 3600.0) - (-12.0 + 0.9 + SIDEREAL_RATE - 1.0)).abs() < 1e-9
        );
    }

    #[test]
    fn pier_side_follows_the_hour_angle() {
        assert_eq!(expected_pier_side(1.0), PierSide::East);
        assert_eq!(expected_pier_side(0.0), PierSide::East);
        assert_eq!(expected_pier_side(-0.01), PierSide::West);
        assert_eq!(expected_pier_side(13.0), PierSide::West);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(expected_pier_side(bad), PierSide::Unknown);
        }
    }

    #[test]
    fn a_non_finite_hour_angle_or_limit_fails_closed() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let t = meridian_timing(bad, Some(0.25));
            assert!(t.past_limit, "{bad}");
            assert_eq!(t.seconds_to_limit, None);
            assert!(t.ha_h.is_nan() && t.seconds_to_meridian.is_nan());
            assert!(t.minutes_past_meridian.is_nan());
            // Without a limit there is nothing to be past.
            let free = meridian_timing(bad, None);
            assert!(!free.past_limit && free.seconds_to_limit.is_none());
            // A finite hour angle against a non-finite limit.
            let t = meridian_timing(-3.0, Some(bad));
            assert!(t.past_limit, "limit {bad}");
            assert_eq!(t.seconds_to_limit, None);
            assert!((t.seconds_to_meridian - sidereal_hours_to_si_seconds(3.0)).abs() < 1e-9);
        }
    }

    #[test]
    fn horizontal_needs_a_known_frame_and_a_valid_site() {
        let at = utc(2026, 10, 26, 22, 0, 0.0);
        let s = site(40.0, -75.0);
        let unknown = EqCoord::new(1.0, 20.0, EqFrame::Unknown);
        assert!(horizontal(unknown, &s, at).is_none());
        assert!(horizontal(
            EqCoord::new(1.0, 20.0, EqFrame::J2000),
            &site(f64::NAN, 0.0),
            at
        )
        .is_none());
        let h = horizontal(EqCoord::new(1.0, 20.0, EqFrame::J2000), &s, at).unwrap();
        assert_eq!(h.apparent.frame, EqFrame::JNowApparent);
        assert!((-90.0..=90.0).contains(&h.alt_deg) && (0.0..360.0).contains(&h.az_deg));
        assert!(h.refracted_alt_deg(None) >= h.alt_deg);
    }

    #[test]
    fn transit_rise_and_set_are_consistent() {
        let s = site(40.0, -75.0);
        let from = utc(2026, 10, 26, 0, 0, 0.0);
        // Vega-like target.
        let c = EqCoord::new(18.6156, 38.7837, EqFrame::J2000);
        let transit = next_transit(c, &s, from).unwrap();
        assert!(transit >= from && transit - from < chrono::Duration::hours(24));
        let h = horizontal(c, &s, transit).unwrap();
        assert!(
            h.ha_h.abs() * 3600.0 < 0.1,
            "HA at transit {} s",
            h.ha_h * 3600.0
        );
        // Transit altitude is 90 − |φ − δ|.
        assert!((h.alt_deg - (90.0 - (40.0 - h.apparent.dec_deg).abs())).abs() < 1e-6);
        let rts = rise_transit_set(c, &s, from, 0.0).unwrap();
        assert!(!rts.always_above && !rts.never_above);
        for t in [rts.rise.unwrap(), rts.set.unwrap()] {
            let alt = horizontal(c, &s, t).unwrap().alt_deg;
            assert!(alt.abs() < 0.01, "altitude at crossing {alt}");
        }
        // Polaris never sets at 40° N; a −60° star never rises.
        let polaris = EqCoord::new(2.53, 89.26, EqFrame::J2000);
        assert!(
            rise_transit_set(polaris, &s, from, 0.0)
                .unwrap()
                .always_above
        );
        let south = EqCoord::new(8.0, -60.0, EqFrame::J2000);
        assert!(rise_transit_set(south, &s, from, 0.0).unwrap().never_above);
    }

    #[test]
    fn visibility_respects_min_altitude_and_horizon() {
        let s = site(40.0, -75.0);
        let from = utc(2026, 10, 26, 0, 0, 0.0);
        let to = from + chrono::Duration::hours(24);
        let c = EqCoord::new(18.6156, 38.7837, EqFrame::J2000);
        let open = visibility_intervals(c, &s, from, to, 30.0, None);
        assert!(!open.is_empty());
        for (a, b) in &open {
            assert!(b > a);
            let mid = *a + (*b - *a) / 2;
            assert!(horizontal(c, &s, mid).unwrap().alt_deg >= 30.0);
        }
        // A 60° wall everywhere leaves less time than a 30° limit alone.
        let wall = HorizonProfile::flat(60.0);
        let walled = visibility_intervals(c, &s, from, to, 30.0, Some(&wall));
        let total = |v: &[(DateTime<Utc>, DateTime<Utc>)]| {
            v.iter().map(|(a, b)| (*b - *a).num_seconds()).sum::<i64>()
        };
        assert!(total(&walled) < total(&open));
        // An invalid site or an unknown frame yields nothing.
        assert!(visibility_intervals(
            EqCoord::new(1.0, 1.0, EqFrame::Unknown),
            &s,
            from,
            to,
            0.0,
            None
        )
        .is_empty());
    }

    #[test]
    fn target_geometry_bundles_the_pieces() {
        let s = site(40.0, -75.0);
        let at = utc(2026, 10, 27, 1, 0, 0.0);
        let c = EqCoord::new(18.6156, 38.7837, EqFrame::J2000);
        let wall = HorizonProfile::flat(10.0);
        let g = target_geometry(c, &s, at, Some(&wall)).unwrap();
        assert_eq!(g.expected_pier_side, expected_pier_side(g.horizontal.ha_h));
        assert_eq!(g.horizon_clearance_deg, Some(g.horizontal.alt_deg - 10.0));
        assert_eq!(g.airmass, refraction::airmass(g.horizontal.alt_deg));
        assert!(g.refracted_alt_deg >= g.horizontal.alt_deg);
    }
}

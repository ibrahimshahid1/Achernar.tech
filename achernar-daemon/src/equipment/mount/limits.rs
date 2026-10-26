//! Mount limits (§3.2 M-k, H13): the gate [`SafetyCheck`] for every mount
//! command (§2.9 step 3) and the tracking watchdog. Replaces the hidden OLS
//! limits file; `D/mount/mount_safety.rs` is deleted by WP1B-4.
//!
//! Owner: WP1A-4.
//!
//! The geometry here is the small amount needed for limits (hour angle,
//! Alt/Az from HA/Dec/latitude, a GMST of about 0.1 s accuracy). The
//! authoritative astronomy (precession, nutation, ΔT, refraction) is
//! `astro/*` (WP1A-3), reached through the [`FrameConverter`]. The horizon
//! profile loader is also `astro/horizon.rs`; it plugs in through
//! [`HorizonLimit`].

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde_json::json;

use crate::config::{EquipmentConfig, MountConfig, MountSafetyConfig};
use crate::equipment::commands::{GatedCommand, GatedOp, MountCmd, SafetyRejection};
use crate::equipment::events::{EventKind, LimitKind};
use crate::equipment::ops::OpKind;
use crate::equipment::snapshot::{EquipmentSnapshot, MountSnapshot, SafetyGate};
use crate::equipment::traits::{FrameConverter, SafetyCheck};
use crate::equipment::types::{
    AlignmentMode, AltAz, CommandSource, EqCoord, EqFrame, PierSide, Site,
};

use super::{is_parked, SIDEREAL_PER_SI};

/// The check name recorded as `safety_rejected{check}`.
pub const CHECK_NAME: &str = "mount_limits";

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// Days since J2000.0 (2000-01-01T12:00:00), UT taken as UTC.
pub fn days_since_j2000(utc: DateTime<Utc>) -> f64 {
    const J2000_UNIX_MS: i64 = 946_728_000_000;
    (utc.timestamp_millis() - J2000_UNIX_MS) as f64 / 86_400_000.0
}

/// Greenwich mean sidereal time, hours, from the IAU 1982 expression
/// (Meeus, *Astronomical Algorithms*, eq. 12.4), with UTC standing in for
/// UT1 (|UT1 - UTC| < 0.9 s). Enough for limits; WP1A-3 owns the precise
/// sidereal time.
pub fn gmst_hours(utc: DateTime<Utc>) -> f64 {
    let d = days_since_j2000(utc);
    let t = d / 36_525.0;
    let deg =
        280.460_618_37 + 360.985_647_366_29 * d + 0.000_387_933 * t * t - t * t * t / 38_710_000.0;
    deg.rem_euclid(360.0) / 15.0
}

/// Local mean sidereal time, hours. Longitude is east-positive.
pub fn lst_hours(utc: DateTime<Utc>, lon_east_deg: f64) -> f64 {
    (gmst_hours(utc) + lon_east_deg / 15.0).rem_euclid(24.0)
}

/// Hour angle in (-12, 12] hours.
pub fn hour_angle_h(lst_h: f64, ra_h: f64) -> f64 {
    let h = (lst_h - ra_h).rem_euclid(24.0);
    if h > 12.0 {
        h - 24.0
    } else {
        h
    }
}

/// Altitude and azimuth (north 0, east 90) from hour angle, declination and
/// latitude. Geometric: no refraction.
pub fn alt_az(ha_h: f64, dec_deg: f64, lat_deg: f64) -> AltAz {
    let h = (ha_h * 15.0).to_radians();
    let d = dec_deg.to_radians();
    let p = lat_deg.to_radians();
    let up = p.sin() * d.sin() + p.cos() * d.cos() * h.cos();
    let north = p.cos() * d.sin() - p.sin() * d.cos() * h.cos();
    let east = -d.cos() * h.sin();
    // atan2 keeps full precision near the zenith, where asin does not.
    AltAz {
        alt_deg: up.atan2(north.hypot(east)).to_degrees(),
        az_deg: east.atan2(north).to_degrees().rem_euclid(360.0),
    }
}

/// Inverse of [`alt_az`]: hour angle (hours) and declination (degrees).
pub fn ha_dec_from_alt_az(target: AltAz, lat_deg: f64) -> (f64, f64) {
    let a = target.alt_deg.to_radians();
    let z = target.az_deg.to_radians();
    let p = lat_deg.to_radians();
    let sin_dec = p.sin() * a.sin() + p.cos() * a.cos() * z.cos();
    let y = -a.cos() * z.sin();
    let x = p.cos() * a.sin() - p.sin() * a.cos() * z.cos();
    let ha = (y.atan2(x).to_degrees() / 15.0 + 24.0).rem_euclid(24.0);
    let ha = if ha > 12.0 { ha - 24.0 } else { ha };
    (ha, sin_dec.clamp(-1.0, 1.0).asin().to_degrees())
}

/// Great-circle separation in degrees between two (longitude, latitude)
/// points in degrees.
pub fn angular_separation_deg(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
    let (l1, l2) = (lon1.to_radians(), lon2.to_radians());
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let a = ((p2 - p1) / 2.0).sin().powi(2) + p1.cos() * p2.cos() * ((l2 - l1) / 2.0).sin().powi(2);
    2.0 * a.sqrt().min(1.0).asin().to_degrees()
}

// ---------------------------------------------------------------------------
// Horizon
// ---------------------------------------------------------------------------

/// A local horizon: the lowest usable altitude at an azimuth. Implemented by
/// [`PiecewiseHorizon`] and by the `astro/horizon.rs` loader (WP1A-3).
pub trait HorizonLimit: Send + Sync {
    /// `None` when the horizon says nothing about this azimuth.
    fn min_altitude_deg(&self, az_deg: f64) -> Option<f64>;
}

/// A horizon from `(az_deg, alt_deg)` points, linearly interpolated in
/// azimuth with wrap-around at 360.
#[derive(Debug, Clone, PartialEq)]
pub struct PiecewiseHorizon {
    points: Vec<(f64, f64)>,
}

impl PiecewiseHorizon {
    pub fn from_points(points: Vec<(f64, f64)>) -> Result<Self, String> {
        if points.is_empty() {
            return Err("a horizon profile needs at least one point".into());
        }
        let mut cleaned = Vec::with_capacity(points.len());
        for (az, alt) in points {
            if !az.is_finite() || !alt.is_finite() || !(-90.0..=90.0).contains(&alt) {
                return Err(format!("invalid horizon point ({az}, {alt})"));
            }
            cleaned.push((az.rem_euclid(360.0), alt));
        }
        cleaned.sort_by(|a, b| a.0.total_cmp(&b.0));
        Ok(Self { points: cleaned })
    }
}

impl HorizonLimit for PiecewiseHorizon {
    fn min_altitude_deg(&self, az_deg: f64) -> Option<f64> {
        if !az_deg.is_finite() {
            return None;
        }
        let az = az_deg.rem_euclid(360.0);
        let n = self.points.len();
        if n == 1 {
            return Some(self.points[0].1);
        }
        let upper = self.points.iter().position(|(a, _)| *a >= az);
        let (lo, hi) = match upper {
            Some(0) | None => (self.points[n - 1], self.points[0]),
            Some(i) => (self.points[i - 1], self.points[i]),
        };
        let span = (hi.0 - lo.0).rem_euclid(360.0);
        if span == 0.0 {
            return Some(lo.1.max(hi.1));
        }
        let t = (az - lo.0).rem_euclid(360.0) / span;
        Some(lo.1 + (hi.1 - lo.1) * t)
    }
}

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------

/// `[mount.safety]` plus the pulse cap and the staleness horizon.
#[derive(Clone)]
pub struct MountLimits {
    pub min_altitude_deg: f64,
    /// Applied to GermanPolar mounts only.
    pub max_hour_angle_h: f64,
    /// The watchdog trips at `min_altitude_deg - hysteresis_deg`.
    pub hysteresis_deg: f64,
    pub horizon: Option<Arc<dyn HorizonLimit>>,
    pub max_pulse_ms: u32,
    /// `[equipment] lost_after_s`: announce `limit_watch_blind` past this.
    pub lost_after: Duration,
}

impl fmt::Debug for MountLimits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MountLimits")
            .field("min_altitude_deg", &self.min_altitude_deg)
            .field("max_hour_angle_h", &self.max_hour_angle_h)
            .field("hysteresis_deg", &self.hysteresis_deg)
            .field("horizon", &self.horizon.is_some())
            .field("max_pulse_ms", &self.max_pulse_ms)
            .field("lost_after", &self.lost_after)
            .finish()
    }
}

impl MountLimits {
    pub fn new(
        safety: &MountSafetyConfig,
        mount: &MountConfig,
        equipment: &EquipmentConfig,
    ) -> Self {
        Self {
            min_altitude_deg: safety.min_altitude,
            max_hour_angle_h: safety.max_hour_angle.abs(),
            hysteresis_deg: safety.limit_hysteresis_deg.max(0.0),
            horizon: None,
            max_pulse_ms: mount.max_pulse_ms.max(1),
            lost_after: Duration::from_secs(equipment.lost_after_s.max(1)),
        }
    }

    /// The horizon from `[mount.safety] horizon_profile_path`, as loaded by
    /// `astro/horizon.rs`.
    pub fn with_horizon(mut self, horizon: Arc<dyn HorizonLimit>) -> Self {
        self.horizon = Some(horizon);
        self
    }

    fn horizon_min(&self, az_deg: f64) -> Option<f64> {
        self.horizon
            .as_ref()
            .and_then(|h| h.min_altitude_deg(az_deg))
    }
}

impl Default for MountLimits {
    fn default() -> Self {
        Self::new(
            &MountSafetyConfig::default(),
            &MountConfig::default(),
            &EquipmentConfig::default(),
        )
    }
}

/// Where a target would put the mount.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetGeometry {
    pub alt_deg: f64,
    pub az_deg: f64,
    /// `None` for an Alt/Az target when the latitude is unknown.
    pub ha_h: Option<f64>,
}

/// The latitude source for geometry: the session site, else the mount's.
fn geometry_site(snap: &EquipmentSnapshot) -> Option<Site> {
    snap.site.or_else(|| {
        snap.mount
            .as_deref()
            .and_then(|m| m.state.site.known().copied())
    })
}

/// LST at the snapshot time: the mount's sidereal time advanced by its age,
/// else the site longitude and the snapshot clock.
fn lst_at(snap: &EquipmentSnapshot, site: Option<&Site>) -> Option<f64> {
    if let Some(mount) = snap.mount.as_deref() {
        let lst = &mount.state.sidereal_time_h;
        if let (Some(value), Some(mono)) = (lst.known(), lst.mono_ns) {
            let age_s = snap.taken_mono_ns.saturating_sub(mono) as f64 / 1e9;
            return Some((value + age_s * SIDEREAL_PER_SI / 3600.0).rem_euclid(24.0));
        }
    }
    site.map(|s| lst_hours(snap.taken_at, s.lon_deg))
}

/// The gate's mount check (§2.9 step 3 table, owner WP1A-4).
pub struct MountLimitsCheck {
    limits: MountLimits,
    converter: Arc<dyn FrameConverter>,
}

impl fmt::Debug for MountLimitsCheck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MountLimitsCheck")
            .field("limits", &self.limits)
            .finish()
    }
}

fn reject(reason: &str, details: serde_json::Value) -> SafetyRejection {
    SafetyRejection {
        check: CHECK_NAME.into(),
        reason: reason.into(),
        details: Some(details),
    }
}

impl MountLimitsCheck {
    pub fn new(limits: MountLimits, converter: Arc<dyn FrameConverter>) -> Self {
        Self { limits, converter }
    }

    pub fn limits(&self) -> &MountLimits {
        &self.limits
    }

    /// Geometry of an equatorial target at the snapshot time, or `None` when
    /// the latitude or the sidereal time is unknown.
    pub fn equatorial_geometry(
        &self,
        target: &EqCoord,
        snap: &EquipmentSnapshot,
    ) -> Option<TargetGeometry> {
        let site = geometry_site(snap);
        let lat = site?.lat_deg;
        let lst = lst_at(snap, site.as_ref())?;
        let of_date = match target.frame {
            EqFrame::J2000 => {
                self.converter
                    .convert(*target, EqFrame::JNowApparent, snap.taken_at, site.as_ref())
            }
            _ => *target,
        };
        let ha = hour_angle_h(lst, of_date.ra_h);
        let altaz = alt_az(ha, of_date.dec_deg, lat);
        Some(TargetGeometry {
            alt_deg: altaz.alt_deg,
            az_deg: altaz.az_deg,
            ha_h: Some(ha),
        })
    }

    fn altaz_geometry(&self, target: &AltAz, snap: &EquipmentSnapshot) -> TargetGeometry {
        let ha_h = geometry_site(snap).map(|site| ha_dec_from_alt_az(*target, site.lat_deg).0);
        TargetGeometry {
            alt_deg: target.alt_deg,
            az_deg: target.az_deg.rem_euclid(360.0),
            ha_h,
        }
    }

    fn check_geometry(
        &self,
        geometry: TargetGeometry,
        alignment: Option<AlignmentMode>,
    ) -> Result<(), SafetyRejection> {
        let limits = &self.limits;
        if geometry.alt_deg < limits.min_altitude_deg {
            return Err(reject(
                "below_min_altitude",
                json!({"alt_deg": geometry.alt_deg, "limit_deg": limits.min_altitude_deg}),
            ));
        }
        if let Some(min) = limits.horizon_min(geometry.az_deg) {
            if geometry.alt_deg < min {
                return Err(reject(
                    "below_horizon",
                    json!({"alt_deg": geometry.alt_deg, "az_deg": geometry.az_deg, "horizon_deg": min}),
                ));
            }
        }
        if alignment == Some(AlignmentMode::GermanPolar) {
            if let Some(ha) = geometry.ha_h {
                if ha.abs() > limits.max_hour_angle_h {
                    return Err(reject(
                        "hour_angle_limit",
                        json!({"ha_h": ha, "limit_h": limits.max_hour_angle_h}),
                    ));
                }
            }
        }
        Ok(())
    }

    /// An equatorial target: limits when computable; otherwise automatic
    /// sources are refused and operator slews are annotated.
    fn check_equatorial(
        &self,
        target: &EqCoord,
        cmd: &GatedCommand,
        snap: &EquipmentSnapshot,
        notes: &mut Vec<String>,
    ) -> Result<(), SafetyRejection> {
        let mount = snap.mount.as_deref();
        match self.equatorial_geometry(target, snap) {
            Some(geometry) => {
                self.check_geometry(geometry, mount.and_then(|m| m.capabilities.alignment_mode))?
            }
            None if cmd.source.is_automatic() => {
                return Err(reject(
                    "limits_unchecked",
                    json!({"message": "the target altitude cannot be computed: no site and no mount sidereal time"}),
                ))
            }
            None => notes.push("limits_unchecked".into()),
        }
        if let Some(answer) = mount.and_then(|m| m.destination_side_of_pier) {
            let same_target = answer
                .target
                .separation_arcmin(target)
                .is_some_and(|sep| sep < 1.0);
            if same_target && answer.side == PierSide::Unknown {
                return Err(reject(
                    "destination_side_of_pier_unknown",
                    json!({"ra_h": target.ra_h, "dec_deg": target.dec_deg}),
                ));
            }
        }
        Ok(())
    }

    /// Accepted-with-note annotations, or the rejection.
    pub fn evaluate(
        &self,
        cmd: &GatedCommand,
        snap: &EquipmentSnapshot,
    ) -> Result<Vec<String>, SafetyRejection> {
        let mount = snap.mount.as_deref();
        let parked = mount.is_some_and(|m| is_parked(&m.state));
        let unsafe_now = snap.safety.state == SafetyGate::Unsafe;
        let relay = matches!(cmd.source, CommandSource::Relay { .. });
        let not_parked = || {
            if parked {
                Err(reject(
                    "mount_parked",
                    json!({"message": "the mount is parked"}),
                ))
            } else {
                Ok(())
            }
        };
        let not_unsafe = || {
            if unsafe_now {
                Err(reject(
                    "safety_unsafe",
                    json!({"message": "the safety monitor reports unsafe; only park and abort are allowed"}),
                ))
            } else {
                Ok(())
            }
        };
        let direct_only = || {
            if relay {
                Err(reject(
                    "relay_not_allowed",
                    json!({"message": "not available over the relay; use the direct daemon API"}),
                ))
            } else {
                Ok(())
            }
        };
        let mut notes = Vec::new();
        match &cmd.op {
            GatedOp::Mount(mount_cmd) => match mount_cmd {
                MountCmd::Slew { target } => {
                    not_parked()?;
                    not_unsafe()?;
                    self.check_equatorial(target, cmd, snap, &mut notes)?;
                }
                MountCmd::SlewAltAz(target) => {
                    not_parked()?;
                    not_unsafe()?;
                    let geometry = self.altaz_geometry(target, snap);
                    self.check_geometry(
                        geometry,
                        mount.and_then(|m| m.capabilities.alignment_mode),
                    )?;
                }
                MountCmd::MoveAxis { .. } | MountCmd::RenewJog => {
                    direct_only()?;
                    not_parked()?;
                    not_unsafe()?;
                }
                MountCmd::PulseGuide { ms, .. } => {
                    direct_only()?;
                    let guiding = snap
                        .guider
                        .as_deref()
                        .and_then(|g| g.app_state.known().copied())
                        .is_some_and(|state| state.owns_mount());
                    if guiding {
                        return Err(reject(
                            "guider_active",
                            json!({"message": "PHD2 is guiding or calibrating; one guiding authority per mount (D20)"}),
                        ));
                    }
                    not_parked()?;
                    not_unsafe()?;
                    if *ms > self.limits.max_pulse_ms {
                        notes.push("pulse_clamped".into());
                    }
                }
                MountCmd::FindHome | MountCmd::SetSideOfPier(_) | MountCmd::SetTracking(true) => {
                    not_parked()?;
                    not_unsafe()?;
                }
                MountCmd::Unpark => not_unsafe()?,
                MountCmd::Sync(_) | MountCmd::SetPark => not_parked()?,
                MountCmd::Park
                | MountCmd::AbortSlew
                | MountCmd::SetTracking(false)
                | MountCmd::SetTrackingRate(_)
                | MountCmd::SetGuideRates { .. }
                | MountCmd::SetSite(_)
                | MountCmd::SetUtc(_)
                | MountCmd::QueryDestinationSideOfPier(_) => {}
            },
            GatedOp::Center { target, .. } => {
                not_parked()?;
                not_unsafe()?;
                self.check_equatorial(target, cmd, snap, &mut notes)?;
            }
            GatedOp::Flip => {
                not_parked()?;
                not_unsafe()?;
            }
            _ => {}
        }
        Ok(notes)
    }
}

impl SafetyCheck for MountLimitsCheck {
    fn name(&self) -> &'static str {
        CHECK_NAME
    }

    fn applies_to(&self, cmd: &GatedCommand) -> bool {
        matches!(
            cmd.op,
            GatedOp::Mount(_) | GatedOp::Center { .. } | GatedOp::Flip
        )
    }

    fn check(&self, cmd: &GatedCommand, snap: &EquipmentSnapshot) -> Result<(), SafetyRejection> {
        self.evaluate(cmd, snap).map(|_| ())
    }

    fn annotations(&self, cmd: &GatedCommand, snap: &EquipmentSnapshot) -> Vec<String> {
        self.evaluate(cmd, snap).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Tracking watchdog
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
struct GoodSample {
    mono_ns: u64,
    ha_h: Option<f64>,
    dec_deg: f64,
}

/// What one watchdog evaluation found.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct WatchVerdict {
    /// `limit_reached` and `limit_watch_blind`. The `limit_reached` event is
    /// what raises the `limit_reached` resume block (WP1A-2's `resume.rs`).
    pub events: Vec<EventKind>,
    /// A crossing: stop tracking, never slew away (M-k).
    pub stop_tracking: bool,
    /// The evaluation used a prediction from the last good sample.
    pub predicted: bool,
    pub alt_deg: Option<f64>,
    pub ha_h: Option<f64>,
}

/// While tracking, watches altitude (with hysteresis), the horizon and, on
/// GermanPolar mounts, the hour angle. When the mount data goes stale it
/// predicts the hour angle from the last good sample plus elapsed sidereal
/// time, and announces `limit_watch_blind` once the data is older than
/// `lost_after`.
#[derive(Debug, Clone)]
pub struct LimitWatchdog {
    limits: MountLimits,
    last_good: Option<GoodSample>,
    blind_announced: bool,
    tripped: bool,
}

impl LimitWatchdog {
    pub fn new(limits: MountLimits) -> Self {
        Self {
            limits,
            last_good: None,
            blind_announced: false,
            tripped: false,
        }
    }

    pub fn limits(&self) -> &MountLimits {
        &self.limits
    }

    pub fn evaluate(
        &mut self,
        snap: &MountSnapshot,
        site: Option<&Site>,
        now_mono_ns: u64,
    ) -> WatchVerdict {
        let mut verdict = WatchVerdict::default();
        let state = &snap.state;
        if state.tracking.known() != Some(&true) {
            self.tripped = false;
            self.blind_announced = false;
            return verdict;
        }
        let moving_op = snap.active_op.as_ref().is_some_and(|op| {
            matches!(
                op.kind,
                OpKind::Slew
                    | OpKind::SlewAltAz
                    | OpKind::Park
                    | OpKind::Unpark
                    | OpKind::Home
                    | OpKind::SetSideOfPier
                    | OpKind::MoveAxis
            )
        });
        if moving_op || state.slewing.known() == Some(&true) || is_parked(state) {
            // The position is changing on purpose; a prediction from before
            // the motion would be wrong.
            self.last_good = None;
            return verdict;
        }
        let lat = site.map(|s| s.lat_deg);
        let fresh = state.ra_h.is_fresh() && state.dec_deg.is_fresh();
        let (alt, az, ha) = if fresh {
            let dec = *state.dec_deg.known().unwrap_or(&0.0);
            let ha = snap.ha_h;
            let reported = state
                .alt_deg
                .fresh_value()
                .copied()
                .zip(state.az_deg.fresh_value().copied());
            let (alt, az) = match (reported, ha, lat) {
                (Some((alt, az)), _, _) => (Some(alt), Some(az)),
                (None, Some(ha), Some(lat)) => {
                    let altaz = alt_az(ha, dec, lat);
                    (Some(altaz.alt_deg), Some(altaz.az_deg))
                }
                _ => (None, None),
            };
            self.last_good = Some(GoodSample {
                mono_ns: state.ra_h.mono_ns.unwrap_or(now_mono_ns),
                ha_h: ha,
                dec_deg: dec,
            });
            self.blind_announced = false;
            (alt, az, ha)
        } else if let Some(good) = self.last_good {
            let stale_ms = now_mono_ns.saturating_sub(good.mono_ns) / 1_000_000;
            if Duration::from_millis(stale_ms) > self.limits.lost_after && !self.blind_announced {
                verdict.events.push(EventKind::LimitWatchBlind { stale_ms });
                self.blind_announced = true;
            }
            verdict.predicted = true;
            let ha = good
                .ha_h
                .map(|ha| ha + (stale_ms as f64 / 1000.0) * SIDEREAL_PER_SI / 3600.0);
            let altaz = ha.zip(lat).map(|(ha, lat)| alt_az(ha, good.dec_deg, lat));
            (altaz.map(|a| a.alt_deg), altaz.map(|a| a.az_deg), ha)
        } else {
            return verdict;
        };
        verdict.alt_deg = alt;
        verdict.ha_h = ha;

        let threshold = self.limits.min_altitude_deg - self.limits.hysteresis_deg;
        let mut crossing = None;
        if let Some(alt) = alt {
            if alt < threshold {
                crossing = Some((LimitKind::Altitude, Some(alt), threshold));
            } else if let Some(min) = az.and_then(|az| self.limits.horizon_min(az)) {
                let horizon_threshold = min - self.limits.hysteresis_deg;
                if alt < horizon_threshold {
                    crossing = Some((LimitKind::Horizon, Some(alt), horizon_threshold));
                }
            }
        }
        if crossing.is_none()
            && snap.capabilities.alignment_mode == Some(AlignmentMode::GermanPolar)
        {
            if let Some(ha) = ha {
                if ha.abs() > self.limits.max_hour_angle_h {
                    crossing = Some((LimitKind::HourAngle, Some(ha), self.limits.max_hour_angle_h));
                }
            }
        }
        match crossing {
            Some((kind, value, limit)) if !self.tripped => {
                verdict
                    .events
                    .push(EventKind::LimitReached { kind, value, limit });
                verdict.stop_tracking = true;
                self.tripped = true;
            }
            Some(_) => {}
            None => self.tripped = false,
        }
        verdict
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;
    use crate::equipment::commands::{GatedCommand, GatedOp};
    use crate::equipment::sim::mount::SimMountProfile;
    use crate::equipment::snapshot::{GuiderAppState, GuiderSnapshot, MountState, SafetySummary};
    use crate::equipment::traits::{Device, IdentityConverter};
    use crate::equipment::types::{AutonomyMode, GuideDirection, Observed, Priority, Source};
    use chrono::TimeZone;

    fn limits(min_alt: f64) -> MountLimits {
        MountLimits {
            min_altitude_deg: min_alt,
            ..MountLimits::default()
        }
    }

    fn check(min_alt: f64) -> MountLimitsCheck {
        MountLimitsCheck::new(limits(min_alt), Arc::new(IdentityConverter))
    }

    fn command(op: GatedOp, source: CommandSource) -> GatedCommand {
        GatedCommand::new(op, source, AutonomyMode::Manual, Priority::Exposure)
    }

    fn operator() -> CommandSource {
        CommandSource::Operator { overriding: false }
    }

    fn policy() -> CommandSource {
        CommandSource::Policy {
            model: "m3".into(),
            version: "1".into(),
        }
    }

    /// A snapshot at a fixed time with a site and a GEM at LST = 6 h.
    fn snapshot(alignment: AlignmentMode) -> EquipmentSnapshot {
        let at = Utc.with_ymd_and_hms(2026, 3, 1, 3, 0, 0).unwrap();
        let mut snap = EquipmentSnapshot::empty(at, 1_000_000_000);
        snap.site = Some(site());
        let mut mount = MountSnapshot::default();
        mount.capabilities.alignment_mode = Some(alignment);
        mount.state.sidereal_time_h = Observed::fresh(6.0, Source::DeviceState, at, 1_000_000_000);
        mount.state.at_park = Observed::fresh(false, Source::DeviceState, at, 1_000_000_000);
        snap.mount = Some(Arc::new(mount));
        snap
    }

    fn slew(ra_h: f64, dec_deg: f64) -> GatedOp {
        GatedOp::Mount(MountCmd::Slew {
            target: EqCoord::new(ra_h, dec_deg, EqFrame::JNowApparent),
        })
    }

    #[test]
    fn gmst_matches_meeus_examples_12a_and_12b() {
        // 1987-04-10 0h UT: 13h10m46.3668s; 19h21m UT: 8h34m57.0896s.
        let a = Utc.with_ymd_and_hms(1987, 4, 10, 0, 0, 0).unwrap();
        let expected_a = 13.0 + 10.0 / 60.0 + 46.3668 / 3600.0;
        assert!(
            (gmst_hours(a) - expected_a).abs() * 3600.0 < 0.01,
            "{}",
            gmst_hours(a)
        );
        let b = Utc.with_ymd_and_hms(1987, 4, 10, 19, 21, 0).unwrap();
        let expected_b = 8.0 + 34.0 / 60.0 + 57.0896 / 3600.0;
        assert!(
            (gmst_hours(b) - expected_b).abs() * 3600.0 < 0.01,
            "{}",
            gmst_hours(b)
        );
        assert!((lst_hours(b, -75.0) - (expected_b - 5.0)).abs() * 3600.0 < 0.01);
    }

    #[test]
    fn alt_az_geometry_is_consistent() {
        let zenith = alt_az(0.0, 40.0, 40.0);
        assert!((zenith.alt_deg - 90.0).abs() < 1e-9);
        let meridian = alt_az(0.0, 0.0, 40.0);
        assert!((meridian.alt_deg - 50.0).abs() < 1e-9);
        assert!((meridian.az_deg - 180.0).abs() < 1e-9);
        let pole = alt_az(3.0, 90.0, 40.0);
        assert!((pole.alt_deg - 40.0).abs() < 1e-9);
        let east = alt_az(-3.0, 10.0, 40.0);
        assert!(
            east.az_deg > 0.0 && east.az_deg < 180.0,
            "rising objects are east"
        );
        for (ha, dec) in [(-5.0, -20.0), (2.5, 35.0), (11.0, 70.0)] {
            let altaz = alt_az(ha, dec, 40.0);
            let (back_ha, back_dec) = ha_dec_from_alt_az(altaz, 40.0);
            assert!((back_ha - ha).abs() < 1e-9 && (back_dec - dec).abs() < 1e-9);
        }
        assert_eq!(hour_angle_h(1.0, 23.0), 2.0);
        assert_eq!(hour_angle_h(23.0, 1.0), -2.0);
        assert!((angular_separation_deg(0.0, 0.0, 90.0, 0.0) - 90.0).abs() < 1e-9);
    }

    #[test]
    fn horizon_profiles_interpolate_with_wrap() {
        let horizon =
            PiecewiseHorizon::from_points(vec![(90.0, 20.0), (270.0, 10.0), (350.0, 30.0)])
                .unwrap();
        assert!((horizon.min_altitude_deg(180.0).unwrap() - 15.0).abs() < 1e-9);
        // 350 -> 90 wraps through north.
        let at_north = horizon.min_altitude_deg(10.0).unwrap();
        assert!((at_north - 28.0).abs() < 1e-9, "{at_north}");
        assert!(PiecewiseHorizon::from_points(vec![]).is_err());
        assert!(PiecewiseHorizon::from_points(vec![(0.0, 95.0)]).is_err());
        let flat = PiecewiseHorizon::from_points(vec![(0.0, 5.0)]).unwrap();
        assert_eq!(flat.min_altitude_deg(123.0), Some(5.0));
    }

    #[test]
    fn slews_below_the_altitude_limit_or_horizon_are_rejected() {
        let snap = snapshot(AlignmentMode::Polar);
        let check = check(20.0);
        // HA 0, Dec 0 at lat 40: altitude 50.
        assert!(check
            .check(&command(slew(6.0, 0.0), operator()), &snap)
            .is_ok());
        // Dec -45 on the meridian: altitude 5.
        let err = check
            .check(&command(slew(6.0, -45.0), operator()), &snap)
            .unwrap_err();
        assert_eq!(
            (err.check.as_str(), err.reason.as_str()),
            (CHECK_NAME, "below_min_altitude")
        );
        let horizon = PiecewiseHorizon::from_points(vec![(0.0, 60.0)]).unwrap();
        let check = MountLimitsCheck::new(
            limits(0.0).with_horizon(Arc::new(horizon)),
            Arc::new(IdentityConverter),
        );
        let err = check
            .check(&command(slew(6.0, 0.0), operator()), &snap)
            .unwrap_err();
        assert_eq!(err.reason, "below_horizon");
    }

    #[test]
    fn hour_angle_limits_apply_to_german_mounts_only() {
        let check = check(-90.0);
        // LST 6 h, RA 0: HA +6 h > 5.5 h.
        let op = slew(0.0, 60.0);
        let err = check
            .check(
                &command(op.clone(), operator()),
                &snapshot(AlignmentMode::GermanPolar),
            )
            .unwrap_err();
        assert_eq!(err.reason, "hour_angle_limit");
        assert!(check
            .check(&command(op, operator()), &snapshot(AlignmentMode::Polar))
            .is_ok());
    }

    #[test]
    fn without_geometry_automatic_slews_are_refused_and_operator_slews_noted() {
        let mut snap = snapshot(AlignmentMode::Polar);
        snap.site = None;
        Arc::make_mut(snap.mount.as_mut().unwrap()).state = MountState::default();
        let check = check(10.0);
        let err = check
            .check(&command(slew(6.0, 0.0), policy()), &snap)
            .unwrap_err();
        assert_eq!(err.reason, "limits_unchecked");
        let operator_cmd = command(slew(6.0, 0.0), operator());
        assert!(check.check(&operator_cmd, &snap).is_ok());
        assert_eq!(
            check.annotations(&operator_cmd, &snap),
            vec!["limits_unchecked".to_string()]
        );
        // The mount's own site is enough when [site] is unset.
        let mut snap = snapshot(AlignmentMode::Polar);
        snap.site = None;
        Arc::make_mut(snap.mount.as_mut().unwrap()).state.site =
            Observed::fresh(site(), Source::Property, snap.taken_at, 1);
        assert!(check
            .check(&command(slew(6.0, 0.0), policy()), &snap)
            .is_ok());
    }

    #[test]
    fn parked_and_unsafe_mounts_refuse_motion_but_allow_park_and_abort() {
        let check = check(-90.0);
        let mut snap = snapshot(AlignmentMode::Polar);
        Arc::make_mut(snap.mount.as_mut().unwrap()).state.at_park =
            Observed::fresh(true, Source::DeviceState, snap.taken_at, 1);
        let err = check
            .check(&command(slew(6.0, 0.0), operator()), &snap)
            .unwrap_err();
        assert_eq!(err.reason, "mount_parked");
        for allowed in [MountCmd::Park, MountCmd::Unpark, MountCmd::AbortSlew] {
            assert!(check
                .check(&command(GatedOp::Mount(allowed), operator()), &snap)
                .is_ok());
        }
        let mut snap = snapshot(AlignmentMode::Polar);
        snap.safety = SafetySummary {
            state: SafetyGate::Unsafe,
            ..SafetySummary::default()
        };
        let err = check
            .check(&command(slew(6.0, 0.0), operator()), &snap)
            .unwrap_err();
        assert_eq!(err.reason, "safety_unsafe");
        let err = check
            .check(
                &command(GatedOp::Mount(MountCmd::Unpark), operator()),
                &snap,
            )
            .unwrap_err();
        assert_eq!(err.reason, "safety_unsafe");
        for allowed in [
            MountCmd::Park,
            MountCmd::AbortSlew,
            MountCmd::SetTracking(false),
        ] {
            assert!(check
                .check(
                    &command(GatedOp::Mount(allowed), CommandSource::Safety),
                    &snap
                )
                .is_ok());
        }
    }

    #[test]
    fn pulses_are_refused_while_the_guider_owns_the_mount_and_over_the_relay() {
        let check = check(-90.0);
        let mut snap = snapshot(AlignmentMode::Polar);
        let pulse = GatedOp::Mount(MountCmd::PulseGuide {
            dir: GuideDirection::North,
            ms: 300,
        });
        assert!(check
            .check(&command(pulse.clone(), operator()), &snap)
            .is_ok());
        for state in [GuiderAppState::Guiding, GuiderAppState::Calibrating] {
            snap.guider = Some(Arc::new(GuiderSnapshot {
                app_state: Observed::fresh(state, Source::Phd2Event, snap.taken_at, 1),
                ..GuiderSnapshot::default()
            }));
            let err = check
                .check(&command(pulse.clone(), operator()), &snap)
                .unwrap_err();
            assert_eq!(err.reason, "guider_active");
        }
        snap.guider = Some(Arc::new(GuiderSnapshot {
            app_state: Observed::fresh(GuiderAppState::Paused, Source::Phd2Event, snap.taken_at, 1),
            ..GuiderSnapshot::default()
        }));
        assert!(check
            .check(&command(pulse.clone(), operator()), &snap)
            .is_ok());
        let err = check
            .check(
                &command(pulse, CommandSource::Relay { overriding: false }),
                &snap,
            )
            .unwrap_err();
        assert_eq!(err.reason, "relay_not_allowed");
        let jog = GatedOp::Mount(MountCmd::MoveAxis {
            axis: 0,
            rate_deg_s: 1.0,
        });
        let err = check
            .check(
                &command(jog.clone(), CommandSource::Relay { overriding: true }),
                &snap,
            )
            .unwrap_err();
        assert_eq!(err.reason, "relay_not_allowed");
        assert!(check.check(&command(jog, operator()), &snap).is_ok());
        let long = GatedOp::Mount(MountCmd::PulseGuide {
            dir: GuideDirection::East,
            ms: 60_000,
        });
        snap.guider = None;
        assert_eq!(
            check.annotations(&command(long, operator()), &snap),
            vec!["pulse_clamped".to_string()]
        );
    }

    #[test]
    fn an_unknown_destination_side_refuses_that_target() {
        let check = check(-90.0);
        let mut snap = snapshot(AlignmentMode::GermanPolar);
        let target = EqCoord::new(5.0, 20.0, EqFrame::JNowApparent);
        Arc::make_mut(snap.mount.as_mut().unwrap()).destination_side_of_pier =
            Some(crate::equipment::snapshot::DestinationSideOfPier {
                target,
                side: PierSide::Unknown,
                at: snap.taken_at,
            });
        let err = check
            .check(
                &command(GatedOp::Mount(MountCmd::Slew { target }), operator()),
                &snap,
            )
            .unwrap_err();
        assert_eq!(err.reason, "destination_side_of_pier_unknown");
        assert!(check
            .check(&command(slew(9.0, 20.0), operator()), &snap)
            .is_ok());
    }

    #[test]
    fn the_check_covers_mount_center_and_flip_commands_only() {
        let check = check(0.0);
        assert_eq!(check.name(), CHECK_NAME);
        assert!(check.applies_to(&command(GatedOp::Mount(MountCmd::Park), operator())));
        assert!(check.applies_to(&command(GatedOp::Flip, operator())));
        assert!(check.applies_to(&command(
            GatedOp::Center {
                target: EqCoord::new(1.0, 2.0, EqFrame::J2000),
                tolerance_arcmin: None,
                max_attempts: None
            },
            operator()
        )));
        assert!(!check.applies_to(&command(
            GatedOp::Focuser(crate::equipment::commands::FocuserCmd::MoveTo(10)),
            operator()
        )));
        let alt_az_cmd = command(
            GatedOp::Mount(MountCmd::SlewAltAz(AltAz {
                alt_deg: -5.0,
                az_deg: 10.0,
            })),
            operator(),
        );
        assert_eq!(
            check
                .check(&alt_az_cmd, &snapshot(AlignmentMode::AltAz))
                .unwrap_err()
                .reason,
            "below_min_altitude"
        );
    }

    fn tracking_snapshot(
        alignment: AlignmentMode,
        ha_h: f64,
        dec: f64,
        mono_ns: u64,
    ) -> MountSnapshot {
        let at = Utc.with_ymd_and_hms(2026, 3, 1, 3, 0, 0).unwrap();
        let mut snap = MountSnapshot::default();
        snap.capabilities.alignment_mode = Some(alignment);
        snap.state.tracking = Observed::fresh(true, Source::DeviceState, at, mono_ns);
        snap.state.slewing = Observed::fresh(false, Source::DeviceState, at, mono_ns);
        snap.state.ra_h = Observed::fresh(0.0, Source::DeviceState, at, mono_ns);
        snap.state.dec_deg = Observed::fresh(dec, Source::DeviceState, at, mono_ns);
        snap.ha_h = Some(ha_h);
        snap
    }

    #[test]
    fn the_watchdog_stops_tracking_on_an_altitude_crossing_with_hysteresis() {
        let mut dog = LimitWatchdog::new(limits(20.0));
        let site = site();
        // Dec 0 at lat 40: HA 4.0 h is 22.5 deg, HA 4.25 h is 19.8 deg
        // (below 20 but inside the 0.5 deg hysteresis), HA 4.5 h is 17.0.
        let above = tracking_snapshot(AlignmentMode::Polar, 4.0, 0.0, 1);
        let verdict = dog.evaluate(&above, Some(&site), 1);
        assert!(!verdict.stop_tracking, "{verdict:?}");
        let inside_hysteresis = tracking_snapshot(AlignmentMode::Polar, 4.25, 0.0, 1);
        let verdict = dog.evaluate(&inside_hysteresis, Some(&site), 1);
        assert!(
            verdict.alt_deg.unwrap() < 20.0 && !verdict.stop_tracking,
            "{verdict:?}"
        );
        let below = tracking_snapshot(AlignmentMode::Polar, 5.0, 0.0, 2);
        let verdict = dog.evaluate(&below, Some(&site), 2);
        assert!(verdict.stop_tracking);
        assert!(matches!(
            verdict.events[0],
            EventKind::LimitReached { kind: LimitKind::Altitude, limit, .. } if (limit - 19.5).abs() < 1e-9
        ));
        // Once tripped it does not repeat while still tracking below.
        assert!(!dog.evaluate(&below, Some(&site), 3).stop_tracking);
        // A slew in progress is not watched.
        let mut slewing = below.clone();
        slewing.state.slewing = Observed::fresh(true, Source::DeviceState, Utc::now(), 4);
        assert!(dog.evaluate(&slewing, Some(&site), 4).events.is_empty());
    }

    #[test]
    fn the_watchdog_checks_hour_angle_on_german_mounts() {
        let mut dog = LimitWatchdog::new(limits(-90.0));
        let snap = tracking_snapshot(AlignmentMode::GermanPolar, 5.6, 60.0, 1);
        let verdict = dog.evaluate(&snap, Some(&site()), 1);
        assert!(verdict.stop_tracking);
        assert!(matches!(
            verdict.events[0],
            EventKind::LimitReached {
                kind: LimitKind::HourAngle,
                ..
            }
        ));
        let mut dog = LimitWatchdog::new(limits(-90.0));
        let snap = tracking_snapshot(AlignmentMode::Polar, 5.6, 60.0, 1);
        assert!(!dog.evaluate(&snap, Some(&site()), 1).stop_tracking);
    }

    #[test]
    fn the_blind_watchdog_predicts_hour_angle_and_announces_blindness() {
        let mut dog = LimitWatchdog::new(limits(-90.0));
        let site = site();
        let s = 1_000_000_000u64;
        let good = tracking_snapshot(AlignmentMode::GermanPolar, 5.4, 60.0, s);
        assert!(!dog.evaluate(&good, Some(&site), s).stop_tracking);
        // The data goes stale: same values, no longer fresh.
        let mut stale = good.clone();
        stale.state = super::super::stale_state(&good.state);
        // 5 s later: still inside, not yet blind.
        let verdict = dog.evaluate(&stale, Some(&site), 6 * s);
        assert!(verdict.predicted && verdict.events.is_empty() && !verdict.stop_tracking);
        // 11 s stale: blind (lost_after 10 s), still inside 5.5 h.
        let verdict = dog.evaluate(&stale, Some(&site), 12 * s);
        assert_eq!(
            verdict.events,
            vec![EventKind::LimitWatchBlind { stale_ms: 11_000 }]
        );
        assert!(!verdict.stop_tracking);
        // Announced once per blind episode.
        assert!(dog.evaluate(&stale, Some(&site), 13 * s).events.is_empty());
        // 0.1 h of sidereal time later the prediction crosses 5.5 h.
        let later = s + 361 * s;
        let verdict = dog.evaluate(&stale, Some(&site), later);
        assert!(verdict.stop_tracking, "{verdict:?}");
        assert!(verdict.ha_h.unwrap() > 5.5);
        assert!(matches!(
            verdict.events[0],
            EventKind::LimitReached {
                kind: LimitKind::HourAngle,
                ..
            }
        ));
    }

    #[test]
    fn the_runtime_stops_tracking_and_records_why() {
        let mut profile = SimMountProfile::esp32_like();
        profile.site = Some(site());
        let settings = super::super::MountSettings::default();
        let mut rig = rig_with(profile, settings, Some(site()));
        rig.runtime.take_events();
        let lst = rig
            .tick(Duration::from_secs(1), None)
            .state
            .sidereal_time_h
            .value
            .unwrap();
        // Put the mount 1 deg above the horizon in the west while tracking.
        let ha = 5.95;
        rig.sim.control().place(EqCoord::new(
            EqCoord::wrap_ra_h(lst - ha),
            0.0,
            EqFrame::JNowApparent,
        ));
        let runtime_limits = MountLimits {
            min_altitude_deg: 10.0,
            ..MountLimits::default()
        };
        rig.runtime = super::super::MountRuntime::new(
            super::super::MountSettings::default(),
            runtime_limits,
            Arc::new(IdentityConverter),
            rig.clock.clone(),
            rig.sim.identity().connection_id,
            Some(site()),
        );
        rig.runtime.on_connected(&mut rig.sim).unwrap();
        // The poll that sees the crossing publishes tracking still on and
        // stops it; the next poll reads it off.
        let snap = rig.tick(Duration::from_secs(1), None);
        assert_eq!(snap.state.tracking.value, Some(true));
        let snap = rig.tick(Duration::from_secs(1), None);
        assert_eq!(snap.state.tracking.value, Some(false));
        let events = rig.runtime.take_events();
        let names: Vec<_> = events.iter().map(|e| e.name()).collect();
        assert!(names.contains(&"limit_reached"), "{names:?}");
        assert!(events.iter().any(|e| matches!(
            &e.event,
            EventKind::TrackingChanged { on: Some(false), source: Some(CommandSource::Safety), reason: Some(r), .. } if r == "limit_reached"
        )));
        assert_eq!(
            names.iter().filter(|n| **n == "tracking_changed").count(),
            1,
            "the observed change is not announced twice: {names:?}"
        );
        // Never an automatic slew from a limit position.
        assert!(!rig.sim.control().slewing());
    }
}

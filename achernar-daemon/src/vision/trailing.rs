//! Classical trailing metrics for the shared vision bus (TED evidence, H15).
//!
//! A trail is measured on the **sensor**: the flux-weighted position angle of
//! the elongated sources, in pixel coordinates (from +x, the column axis,
//! toward +y, the row axis), and the sensor axis it lies along. That is
//! always reported.
//!
//! A trail is attributed to a **sky** axis (RA or Dec) only when a plate
//! solve's WCS says how this pixel grid sits on the sky: the CD matrix
//! carries the rotation and the parity. The attribution is withheld, with
//! the reason, unless all of these hold:
//!
//! - the solve is of the same camera and its pixel grid has the same
//!   orientation;
//! - it was taken under the mount connection now in use, and less than
//!   [`MAX_SOLVE_AGE`] ago (a camera turned by hand between sessions is not
//!   otherwise seen);
//! - on a mount that is not equatorial (Alt/Az, or an alignment mode that is
//!   unknown) the field rotates with time and with every repointing: the
//!   solve is less than [`FIELD_ROTATION_BUDGET`] old and the mount has not
//!   slewed, jogged, flipped, parked or homed since;
//! - the pier side and rotator angle are known and unchanged since that
//!   solve.
//!
//! Before this, any trail along the sensor's x axis was labelled "RA",
//! whatever the camera's rotation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::background_model::{BackgroundAnalysis, BackgroundField, SourceCatalog};
use super::quality::LumaFrame;
use crate::equipment::types::{AlignmentMode, PierSide};

/// Sky-axis attribution tolerance: a trail within this of an axis is
/// labelled with it.
const AXIS_TOLERANCE_DEG: f32 = 25.0;
/// A rotator that moved more than this since the solve invalidates its WCS.
const ROTATOR_TOLERANCE_DEG: f64 = 0.5;
/// The oldest solve whose orientation is trusted at all: about one night.
pub const MAX_SOLVE_AGE: chrono::Duration = chrono::Duration::hours(12);
/// The oldest solve trusted on a mount that is not equatorial. Field
/// rotation is about 15°/h·cos(lat)·cos(az)/cos(alt): a degree or two in
/// five minutes away from the zenith, well inside the axis tolerance.
pub const FIELD_ROTATION_BUDGET: chrono::Duration = chrono::Duration::minutes(5);

/// The sensor axis a trail lies along.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorAxis {
    /// Along the columns (+x).
    X,
    /// Along the rows (+y).
    Y,
    Diagonal,
}

impl SensorAxis {
    pub fn label(self) -> &'static str {
        match self {
            SensorAxis::X => "sensor_x",
            SensorAxis::Y => "sensor_y",
            SensorAxis::Diagonal => "sensor_diagonal",
        }
    }
}

/// The sky axis a trail lies along.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkyAxis {
    #[serde(rename = "RA")]
    Ra,
    #[serde(rename = "Dec")]
    Dec,
    Ambiguous,
}

impl SkyAxis {
    pub fn label(self) -> &'static str {
        match self {
            SkyAxis::Ra => "RA",
            SkyAxis::Dec => "Dec",
            SkyAxis::Ambiguous => "Ambiguous",
        }
    }
}

/// A trail placed on the sky.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkyAttribution {
    pub axis: SkyAxis,
    /// Position angle on the sky, degrees east of north, in [0, 180).
    pub sky_pa_deg: f32,
    pub solve_event_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TrailingEstimate {
    pub severity: f32,
    /// Position angle on the sensor, degrees in [0, 180); `None` for a
    /// round field.
    pub sensor_pa_deg: Option<f32>,
    pub sensor_axis: Option<SensorAxis>,
    /// Only with a WCS that still describes this frame.
    pub sky: Option<SkyAttribution>,
    /// Why `sky` is `None` when there is a trail to attribute.
    pub withheld: Option<String>,
    /// Source-selection policy that produced the ellipses.
    pub policy: &'static str,
    /// Set when trailing could not be measured (no background, or no
    /// detection on the spatial path). `severity` is then not a measurement.
    pub unavailable_reason: Option<String>,
}

/// Legacy trailing selection: 8-connected components above the global
/// `max(B + 3*max(sigma,1), B + 6)`, 5..=5000 pixels, no masked or clipped
/// neighbour, moments of `max(I-B,0)`.
pub const LEGACY_TRAILING_POLICY: &str = "legacy_components_3sigma_v1";
/// SEP-path trailing selection: the shared SEP catalog's detection
/// footprints (the focus detection threshold, not a separate 3-sigma pass),
/// 5..=5000 pixels, no masked or clipped neighbour, moments of the
/// *unfiltered* residual `max(I-B(x,y),0)` (SEP's own a/b/theta are moments
/// of the filtered image). Focus-star gates are not applied.
pub const SEP_TRAILING_POLICY: &str = "sep_footprint_unfiltered_moments_v1";

impl TrailingEstimate {
    /// The bus's `trailing_axis`: the sky axis when attributed, else the
    /// sensor axis (`sensor_x`, `sensor_y`, `sensor_diagonal`). Never RA or
    /// Dec without a WCS.
    pub fn axis_label(&self) -> Option<String> {
        match (&self.sky, self.sensor_axis) {
            (Some(sky), _) => Some(sky.axis.label().to_string()),
            (None, Some(axis)) => Some(axis.label().to_string()),
            (None, None) => None,
        }
    }
}

/// What a plate solve says about the pixel grid, and the conditions it was
/// taken under.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveOrientation<'a> {
    /// `CDi_j` of the solved file: pixel offset to (east, north), degrees.
    pub cd: Option<[[f64; 2]; 2]>,
    pub naxis: Option<[u32; 2]>,
    pub solve_event_id: Option<&'a str>,
    pub camera: Option<&'a str>,
    pub pier_side: Option<PierSide>,
    pub rotator_angle_deg: Option<f64>,
    /// When the solved frame was exposed (else recorded).
    pub exposed_at: Option<DateTime<Utc>>,
}

/// The frame being measured, and the conditions now.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameGeometry<'a> {
    pub camera: Option<&'a str>,
    pub width: usize,
    pub height: usize,
    pub pier_side: Option<PierSide>,
    pub rotator_angle_deg: Option<f64>,
    /// When this frame was seen.
    pub at: Option<DateTime<Utc>>,
    /// The mount's alignment mode; `None` when unknown.
    pub alignment_mode: Option<AlignmentMode>,
    /// The solve was taken under the mount connection now in use; `None`
    /// when that cannot be told.
    pub solve_on_this_connection: Option<bool>,
    /// The mount slewed, jogged, flipped, parked or homed after the solved
    /// exposure; `None` when not known.
    pub mount_repointed_since_solve: Option<bool>,
}

#[derive(Debug, Clone)]
struct Ellipse {
    ellipticity: f32,
    pa_deg: f32,
    flux: f32,
}

/// Measure the trail on the sensor. No sky axis is claimed: see [`attribute`].
pub fn estimate_trailing(frame: &LumaFrame, background: &BackgroundAnalysis) -> TrailingEstimate {
    let (mut ellipses, policy) = match (&background.catalog, background.field.as_ref()) {
        (_, None) => return unavailable(LEGACY_TRAILING_POLICY, "background_unavailable"),
        (Some(catalog), Some(field)) => {
            (catalog_ellipses(frame, field, catalog), SEP_TRAILING_POLICY)
        }
        (None, Some(BackgroundField::Uniform { level, sigma })) => (
            source_ellipses(frame, *level, *sigma),
            LEGACY_TRAILING_POLICY,
        ),
        (None, Some(BackgroundField::Map { .. })) => {
            let reason = background
                .catalog_error
                .clone()
                .unwrap_or_else(|| "detection_unavailable".to_string());
            return unavailable(SEP_TRAILING_POLICY, &reason);
        }
    };
    if ellipses.is_empty() {
        return TrailingEstimate {
            severity: 0.0,
            sensor_pa_deg: None,
            sensor_axis: None,
            sky: None,
            withheld: None,
            policy,
            unavailable_reason: None,
        };
    }

    ellipses.sort_by(|a, b| a.ellipticity.total_cmp(&b.ellipticity));
    let ellipticity = trimmed_mean_ellipticity(&ellipses, 0.2);
    let pa_deg = weighted_mean_pa(&ellipses);
    let severity = if ellipses.len() >= 3 {
        ellipticity.clamp(0.0, 1.0)
    } else {
        (ellipticity * 0.5).clamp(0.0, 1.0)
    };
    let sensor_axis = (severity > 0.15).then(|| classify_sensor_axis(pa_deg));

    TrailingEstimate {
        severity,
        sensor_pa_deg: (severity > 0.10).then_some(pa_deg),
        sensor_axis,
        sky: None,
        withheld: sensor_axis.map(|_| "no_plate_solve".to_string()),
        policy,
        unavailable_reason: None,
    }
}

fn unavailable(policy: &'static str, reason: &str) -> TrailingEstimate {
    TrailingEstimate {
        severity: 0.0,
        sensor_pa_deg: None,
        sensor_axis: None,
        sky: None,
        withheld: None,
        policy,
        unavailable_reason: Some(reason.to_string()),
    }
}

/// Why `solve` cannot place this frame's trails on the sky, if it cannot.
fn orientation_problem(solve: &SolveOrientation, frame: &FrameGeometry) -> Option<&'static str> {
    if solve.cd.is_none() {
        return Some("solve_has_no_wcs");
    }
    match (solve.camera, frame.camera) {
        (Some(a), Some(b)) if a != b => return Some("solved_with_another_camera"),
        (Some(_), Some(_)) => {}
        _ => return Some("camera_identity_unknown"),
    }
    if let Some([w, h]) = solve.naxis {
        if (w >= h) != (frame.width >= frame.height) {
            return Some("pixel_grid_differs_from_solve");
        }
    }
    match frame.solve_on_this_connection {
        Some(true) => {}
        Some(false) => return Some("solve_from_another_mount_connection"),
        None => return Some("solve_connection_unknown"),
    }
    let age = match (solve.exposed_at, frame.at) {
        (Some(solved), Some(now)) => (now - solved).abs(),
        _ => return Some("solve_time_unknown"),
    };
    if age > MAX_SOLVE_AGE {
        return Some("solve_older_than_session");
    }
    let equatorial = matches!(
        frame.alignment_mode,
        Some(AlignmentMode::Polar | AlignmentMode::GermanPolar)
    );
    if !equatorial {
        if age > FIELD_ROTATION_BUDGET {
            return Some("field_rotation_since_solve");
        }
        match frame.mount_repointed_since_solve {
            Some(false) => {}
            Some(true) => return Some("mount_repointed_since_solve"),
            None => return Some("mount_motion_since_solve_unknown"),
        }
    }
    match (solve.pier_side, frame.pier_side) {
        (Some(PierSide::Unknown) | None, _) => return Some("pier_side_unknown_at_solve"),
        (_, Some(PierSide::Unknown) | None) => return Some("pier_side_unknown"),
        (Some(a), Some(b)) if a != b => return Some("pier_side_changed_since_solve"),
        _ => {}
    }
    match (solve.rotator_angle_deg, frame.rotator_angle_deg) {
        (None, None) => None,
        (None, Some(_)) => Some("rotator_angle_unknown_at_solve"),
        (Some(_), None) => Some("rotator_angle_unknown"),
        (Some(a), Some(b)) => {
            let d = ((a - b + 540.0).rem_euclid(360.0) - 180.0).abs();
            (d > ROTATOR_TOLERANCE_DEG).then_some("rotator_moved_since_solve")
        }
    }
}

/// The sky position angle (east of north, [0, 180)) of a trail at sensor
/// angle `sensor_pa_deg`, through the CD matrix.
pub fn sky_pa_deg(cd: &[[f64; 2]; 2], sensor_pa_deg: f32) -> f32 {
    let (s, c) = f64::from(sensor_pa_deg).to_radians().sin_cos();
    let east = cd[0][0] * c + cd[0][1] * s;
    let north = cd[1][0] * c + cd[1][1] * s;
    (east.atan2(north).to_degrees().rem_euclid(180.0)) as f32
}

/// Attribute `estimate` to a sky axis when `solve` still describes `frame`;
/// otherwise keep the sensor-axis result and say why.
pub fn attribute(
    mut estimate: TrailingEstimate,
    solve: Option<&SolveOrientation>,
    frame: &FrameGeometry,
) -> TrailingEstimate {
    let Some(pa) = estimate
        .sensor_pa_deg
        .filter(|_| estimate.sensor_axis.is_some())
    else {
        estimate.sky = None;
        estimate.withheld = None;
        return estimate;
    };
    let Some(solve) = solve else {
        estimate.sky = None;
        estimate.withheld = Some("no_plate_solve".into());
        return estimate;
    };
    if let Some(problem) = orientation_problem(solve, frame) {
        estimate.sky = None;
        estimate.withheld = Some(problem.into());
        return estimate;
    }
    let Some(cd) = solve.cd else {
        estimate.withheld = Some("solve_has_no_wcs".into());
        return estimate;
    };
    let sky_pa = sky_pa_deg(&cd, pa);
    estimate.sky = Some(SkyAttribution {
        axis: classify_sky_axis(sky_pa),
        sky_pa_deg: sky_pa,
        solve_event_id: solve.solve_event_id.map(str::to_string),
    });
    estimate.withheld = None;
    estimate
}

fn source_ellipses(frame: &LumaFrame, level: f32, measured_sigma: f32) -> Vec<Ellipse> {
    let sigma = measured_sigma.max(1.0);
    let threshold = (level + sigma * 3.0).max(level + 6.0);
    let mut visited = vec![false; frame.pixels.len()];
    let mut ellipses = Vec::new();

    for idx in 0..frame.pixels.len() {
        let value = frame.pixels[idx];
        if visited[idx]
            || !frame.measurement_pixel_valid(idx)
            || frame.native_saturation_at(idx)
            || !value.is_finite()
            || value < threshold
        {
            continue;
        }

        let mut stack = vec![idx];
        visited[idx] = true;
        let mut points = Vec::new();
        let mut flux = 0.0_f32;
        let mut weighted_x = 0.0_f32;
        let mut weighted_y = 0.0_f32;
        let mut touches_masked_support = false;

        while let Some(current) = stack.pop() {
            let current_value = frame.pixels[current];
            let signal = (current_value - level).max(0.0);
            let x = current % frame.width;
            let y = current / frame.width;

            points.push((x as f32, y as f32, signal));
            flux += signal;
            weighted_x += x as f32 * signal;
            weighted_y += y as f32 * signal;

            let min_x = x.saturating_sub(1);
            let max_x = (x + 1).min(frame.width - 1);
            let min_y = y.saturating_sub(1);
            let max_y = (y + 1).min(frame.height - 1);

            for ny in min_y..=max_y {
                for nx in min_x..=max_x {
                    if nx == x && ny == y {
                        continue;
                    }
                    let neighbor = ny * frame.width + nx;
                    let neighbor_value = frame.pixels[neighbor];
                    touches_masked_support |= !frame.measurement_pixel_valid(neighbor)
                        || frame.native_saturation_at(neighbor);
                    if visited[neighbor]
                        || !frame.measurement_pixel_valid(neighbor)
                        || frame.native_saturation_at(neighbor)
                        || !neighbor_value.is_finite()
                        || neighbor_value < threshold
                    {
                        continue;
                    }
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }

        if touches_masked_support || points.len() < 5 || points.len() > 5_000 {
            continue;
        }
        if let Some(ellipse) = ellipse_of(&points, flux, weighted_x, weighted_y) {
            ellipses.push(ellipse);
        }
    }

    ellipses
}

/// Shape of each shared-catalog footprint from the unfiltered residual.
fn catalog_ellipses(
    frame: &LumaFrame,
    field: &BackgroundField,
    catalog: &SourceCatalog,
) -> Vec<Ellipse> {
    let mut ellipses = Vec::new();
    for object in &catalog.objects {
        let n = object.pixels.len();
        if !(5..=5_000).contains(&n) {
            continue;
        }
        // Same veto as the legacy policy: no footprint pixel may touch
        // masked or clipped support in its 8-neighbourhood.
        let touches_masked_support = object.pixels.iter().any(|&index| {
            let index = index as usize;
            let x = index % frame.width;
            let y = index / frame.width;
            (y.saturating_sub(1)..=(y + 1).min(frame.height - 1)).any(|ny| {
                (x.saturating_sub(1)..=(x + 1).min(frame.width - 1)).any(|nx| {
                    let neighbor = ny * frame.width + nx;
                    !frame.measurement_pixel_valid(neighbor) || frame.native_saturation_at(neighbor)
                })
            })
        });
        if touches_masked_support {
            continue;
        }
        let mut points = Vec::with_capacity(n);
        let mut flux = 0.0_f32;
        let mut weighted_x = 0.0_f32;
        let mut weighted_y = 0.0_f32;
        for &index in &object.pixels {
            let index = index as usize;
            let (x, y) = ((index % frame.width) as f32, (index / frame.width) as f32);
            let signal = (frame.pixels[index] - field.level_at(index)).max(0.0);
            points.push((x, y, signal));
            flux += signal;
            weighted_x += x * signal;
            weighted_y += y * signal;
        }
        if let Some(ellipse) = ellipse_of(&points, flux, weighted_x, weighted_y) {
            ellipses.push(ellipse);
        }
    }
    ellipses
}

/// Flux-weighted second moments of `(x, y, signal)` points (SExtractor A/B).
fn ellipse_of(
    points: &[(f32, f32, f32)],
    flux: f32,
    weighted_x: f32,
    weighted_y: f32,
) -> Option<Ellipse> {
    if flux <= f32::EPSILON {
        return None;
    }
    let cx = weighted_x / flux;
    let cy = weighted_y / flux;
    let mut mxx = 0.0_f32;
    let mut myy = 0.0_f32;
    let mut mxy = 0.0_f32;

    for &(x, y, signal) in points {
        let dx = x - cx;
        let dy = y - cy;
        mxx += signal * dx * dx;
        myy += signal * dy * dy;
        mxy += signal * dx * dy;
    }

    mxx /= flux;
    myy /= flux;
    mxy /= flux;

    let trace = mxx + myy;
    let det_term = ((mxx - myy).powi(2) + 4.0 * mxy * mxy).sqrt();
    let major = ((trace + det_term) * 0.5).max(0.0).sqrt();
    let minor = ((trace - det_term) * 0.5).max(0.0).sqrt();
    if major <= f32::EPSILON {
        return None;
    }

    let ellipticity = (1.0 - minor / major).clamp(0.0, 1.0);
    let pa_deg = normalize_pa(0.5 * (2.0 * mxy).atan2(mxx - myy).to_degrees());
    Some(Ellipse {
        ellipticity,
        pa_deg,
        flux,
    })
}

fn trimmed_mean_ellipticity(ellipses: &[Ellipse], trim_fraction: f32) -> f32 {
    if ellipses.is_empty() {
        return 0.0;
    }
    let trim = ((ellipses.len() as f32 * trim_fraction).floor() as usize)
        .min(ellipses.len().saturating_sub(1) / 2);
    let slice = &ellipses[trim..ellipses.len() - trim];
    slice.iter().map(|ellipse| ellipse.ellipticity).sum::<f32>() / slice.len() as f32
}

fn weighted_mean_pa(ellipses: &[Ellipse]) -> f32 {
    let mut x = 0.0_f32;
    let mut y = 0.0_f32;
    let mut weight_sum = 0.0_f32;

    for ellipse in ellipses {
        let weight = ellipse.flux.max(1.0) * ellipse.ellipticity.max(0.01);
        let radians = (ellipse.pa_deg * 2.0).to_radians();
        x += weight * radians.cos();
        y += weight * radians.sin();
        weight_sum += weight;
    }

    if weight_sum <= f32::EPSILON {
        return 0.0;
    }

    normalize_pa(0.5 * y.atan2(x).to_degrees())
}

/// Distance of an axis-free angle from `axis`, both in [0, 180).
fn axis_distance(pa: f32, axis: f32) -> f32 {
    let d = (pa - axis).rem_euclid(180.0);
    d.min(180.0 - d)
}

fn classify_sensor_axis(pa_deg: f32) -> SensorAxis {
    let pa = normalize_pa(pa_deg);
    if axis_distance(pa, 0.0) < AXIS_TOLERANCE_DEG {
        SensorAxis::X
    } else if axis_distance(pa, 90.0) < AXIS_TOLERANCE_DEG {
        SensorAxis::Y
    } else {
        SensorAxis::Diagonal
    }
}

/// East-west trails are along RA (PA 90), north-south along Dec (PA 0).
fn classify_sky_axis(sky_pa_deg: f32) -> SkyAxis {
    let pa = normalize_pa(sky_pa_deg);
    if axis_distance(pa, 90.0) < AXIS_TOLERANCE_DEG {
        SkyAxis::Ra
    } else if axis_distance(pa, 0.0) < AXIS_TOLERANCE_DEG {
        SkyAxis::Dec
    } else {
        SkyAxis::Ambiguous
    }
}

fn normalize_pa(pa_deg: f32) -> f32 {
    pa_deg.rem_euclid(180.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::quality::background_from_luma;

    /// One star trailed along the sensor's x axis.
    fn trailed_along_x() -> LumaFrame {
        let width = 64;
        let height = 64;
        let mut pixels = vec![100.0_f32; width * height];
        for x in 24..40 {
            pixels[32 * width + x] = 600.0;
        }
        for x in 27..37 {
            pixels[31 * width + x] = 350.0;
            pixels[33 * width + x] = 350.0;
        }
        LumaFrame {
            pixels,
            width,
            height,
            max_adu: 65535.0,
            saturation_adu: Some(65535.0),
            masks: Default::default(),
            analysis_descriptor: None,
        }
    }

    fn measured() -> TrailingEstimate {
        let frame = trailed_along_x();
        let background = background_from_luma(&frame);
        estimate_trailing(&frame, &background)
    }

    fn at(minutes: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_792_029_600, 0).unwrap() + chrono::Duration::minutes(minutes)
    }

    /// A frame on a German equatorial mount, an hour after the solve.
    fn frame(pier: Option<PierSide>) -> FrameGeometry<'static> {
        FrameGeometry {
            camera: Some("SV105"),
            width: 64,
            height: 64,
            pier_side: pier,
            rotator_angle_deg: None,
            at: Some(at(60)),
            alignment_mode: Some(AlignmentMode::GermanPolar),
            solve_on_this_connection: Some(true),
            mount_repointed_since_solve: Some(true),
        }
    }

    /// A solve with the given CD matrix, same camera and pier side.
    fn solve(cd: [[f64; 2]; 2]) -> SolveOrientation<'static> {
        SolveOrientation {
            cd: Some(cd),
            naxis: Some([4000, 3000]),
            solve_event_id: Some("solve-7"),
            camera: Some("SV105"),
            pier_side: Some(PierSide::East),
            rotator_angle_deg: None,
            exposed_at: Some(at(0)),
        }
    }

    #[test]
    fn elongated_star_produces_a_sensor_axis_signal_and_never_a_sky_label() {
        let trailing = measured();
        assert!(trailing.severity > 0.15);
        assert_eq!(trailing.sensor_axis, Some(SensorAxis::X));
        assert!(trailing.sensor_pa_deg.is_some());
        assert!(trailing.sky.is_none());
        assert_eq!(trailing.withheld.as_deref(), Some("no_plate_solve"));
        assert_eq!(trailing.axis_label().as_deref(), Some("sensor_x"));
    }

    #[test]
    fn trailing_output_never_labels_ra_or_dec_without_a_wcs_that_still_applies() {
        let north_up_east_left = [[-3.0e-4, 0.0], [0.0, 3.0e-4]];
        let no_wcs = SolveOrientation {
            cd: None,
            ..solve(north_up_east_left)
        };
        let other_camera = SolveOrientation {
            camera: Some("R50"),
            ..solve(north_up_east_left)
        };
        let unknown_camera = SolveOrientation {
            camera: None,
            ..solve(north_up_east_left)
        };
        let unknown_side_at_solve = SolveOrientation {
            pier_side: None,
            ..solve(north_up_east_left)
        };
        let rotated = SolveOrientation {
            rotator_angle_deg: Some(12.0),
            ..solve(north_up_east_left)
        };
        let portrait = SolveOrientation {
            naxis: Some([3000, 4000]),
            ..solve(north_up_east_left)
        };
        let good = solve(north_up_east_left);
        let east = frame(Some(PierSide::East));
        let cases: [(Option<&SolveOrientation>, FrameGeometry, &str); 8] = [
            (None, east, "no_plate_solve"),
            (Some(&no_wcs), east, "solve_has_no_wcs"),
            (Some(&other_camera), east, "solved_with_another_camera"),
            (Some(&unknown_camera), east, "camera_identity_unknown"),
            (
                Some(&unknown_side_at_solve),
                east,
                "pier_side_unknown_at_solve",
            ),
            (Some(&good), frame(None), "pier_side_unknown"),
            (
                Some(&good),
                frame(Some(PierSide::West)),
                "pier_side_changed_since_solve",
            ),
            (Some(&rotated), east, "rotator_angle_unknown"),
        ];
        for (orientation, geometry, reason) in cases {
            let trailing = attribute(measured(), orientation, &geometry);
            assert!(trailing.sky.is_none(), "{reason}");
            assert_eq!(trailing.withheld.as_deref(), Some(reason));
            let label = trailing.axis_label().unwrap();
            assert!(label.starts_with("sensor_"), "{reason}: {label}");
        }
        let trailing = attribute(measured(), Some(&portrait), &frame(Some(PierSide::East)));
        assert_eq!(
            trailing.withheld.as_deref(),
            Some("pixel_grid_differs_from_solve")
        );
    }

    /// The WCS of an earlier target, hour, connection or session.
    #[test]
    fn a_solve_from_another_time_or_connection_is_not_this_frames_orientation() {
        let good = solve([[-3.0e-4, 0.0], [0.0, 3.0e-4]]);
        let east = frame(Some(PierSide::East));
        let check = |solve: &SolveOrientation, frame: FrameGeometry| {
            attribute(measured(), Some(solve), &frame).withheld
        };
        // An equatorial mount: an hour and a slew later the WCS still holds.
        assert_eq!(check(&good, east), None);

        let reconnected = FrameGeometry {
            solve_on_this_connection: Some(false),
            ..east
        };
        assert_eq!(
            check(&good, reconnected).as_deref(),
            Some("solve_from_another_mount_connection")
        );
        let unknown_connection = FrameGeometry {
            solve_on_this_connection: None,
            ..east
        };
        assert_eq!(
            check(&good, unknown_connection).as_deref(),
            Some("solve_connection_unknown")
        );
        let last_night = FrameGeometry {
            at: Some(at(13 * 60)),
            ..east
        };
        assert_eq!(
            check(&good, last_night).as_deref(),
            Some("solve_older_than_session")
        );
        let untimed = SolveOrientation {
            exposed_at: None,
            ..good
        };
        assert_eq!(check(&untimed, east).as_deref(), Some("solve_time_unknown"));

        // Alt/Az (or unknown): the field rotates with time and every slew.
        for mode in [Some(AlignmentMode::AltAz), None] {
            let altaz = FrameGeometry {
                alignment_mode: mode,
                ..east
            };
            assert_eq!(
                check(&good, altaz).as_deref(),
                Some("field_rotation_since_solve"),
                "{mode:?}"
            );
            let recent = FrameGeometry {
                at: Some(at(2)),
                ..altaz
            };
            assert_eq!(
                check(&good, recent).as_deref(),
                Some("mount_repointed_since_solve"),
                "{mode:?}"
            );
            let unknown_motion = FrameGeometry {
                mount_repointed_since_solve: None,
                ..recent
            };
            assert_eq!(
                check(&good, unknown_motion).as_deref(),
                Some("mount_motion_since_solve_unknown")
            );
            let still = FrameGeometry {
                mount_repointed_since_solve: Some(false),
                ..recent
            };
            assert_eq!(check(&good, still), None, "{mode:?}");
        }
    }

    #[test]
    fn a_wcs_places_the_same_sensor_trail_on_ra_or_dec_by_the_cameras_rotation() {
        // North up, east left: the sensor's x axis runs east-west, i.e. RA.
        let unrotated = attribute(
            measured(),
            Some(&solve([[-3.0e-4, 0.0], [0.0, 3.0e-4]])),
            &frame(Some(PierSide::East)),
        );
        let sky = unrotated.sky.as_ref().unwrap();
        assert_eq!(sky.axis, SkyAxis::Ra);
        assert!((sky.sky_pa_deg - 90.0).abs() < 5.0, "{}", sky.sky_pa_deg);
        assert_eq!(sky.solve_event_id.as_deref(), Some("solve-7"));
        assert_eq!(unrotated.axis_label().as_deref(), Some("RA"));

        // The camera turned 90 degrees: the same sensor trail is along Dec.
        let rotated = attribute(
            measured(),
            Some(&solve([[0.0, 3.0e-4], [3.0e-4, 0.0]])),
            &frame(Some(PierSide::East)),
        );
        assert_eq!(rotated.sky.unwrap().axis, SkyAxis::Dec);

        // Mirrored (parity flipped) leaves an x trail on RA.
        let mirrored = attribute(
            measured(),
            Some(&solve([[3.0e-4, 0.0], [0.0, 3.0e-4]])),
            &frame(Some(PierSide::East)),
        );
        assert_eq!(mirrored.sky.unwrap().axis, SkyAxis::Ra);

        // Diagonal on the sky is ambiguous.
        let diagonal = attribute(
            measured(),
            Some(&solve([[-2.1e-4, 2.1e-4], [2.1e-4, 2.1e-4]])),
            &frame(Some(PierSide::East)),
        );
        assert_eq!(diagonal.sky.unwrap().axis, SkyAxis::Ambiguous);
    }

    #[test]
    fn a_round_field_has_nothing_to_attribute() {
        let mut round = measured();
        round.sensor_axis = None;
        round.sensor_pa_deg = None;
        let attributed = attribute(round, None, &frame(Some(PierSide::East)));
        assert!(attributed.sky.is_none() && attributed.withheld.is_none());
        assert_eq!(attributed.axis_label(), None);
    }

    #[test]
    fn shared_sep_catalog_trails_use_unfiltered_footprint_moments() {
        use crate::vision::background_model::{
            BackgroundField, CatalogSource, CropSkySource, DetectionMethod, DetectionProvenance,
            SourceCatalog,
        };
        let frame = trailed_along_x();
        let mut analysis = background_from_luma(&frame);
        let pixels: Vec<u32> = frame
            .pixels
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > 100.0)
            .map(|(i, _)| i as u32)
            .collect();
        analysis.field = Some(BackgroundField::Map {
            level: vec![100.0; frame.pixels.len()],
            rms: vec![1.0; frame.pixels.len()],
            width: frame.width,
            height: frame.height,
        });
        analysis.catalog = Some(SourceCatalog {
            provenance: DetectionProvenance {
                method: DetectionMethod::SepExtract,
                config_sha256: "test".into(),
                crop_sky_source: CropSkySource::ParentBackgroundMap,
                noise_floor_applied: false,
                objects_detected: 1,
                objects_deblended: 0,
                objects_truncated: 0,
            },
            objects: vec![CatalogSource {
                x: 31.5,
                y: 32.0,
                xmin: 24,
                xmax: 39,
                ymin: 31,
                ymax: 33,
                npix: pixels.len() as u32,
                flux: 1.0e4,
                peak_residual: 500.0,
                peak_raw: 600.0,
                // SEP's own (filtered-image) shape is deliberately round here:
                // the policy must not use it.
                a: 3.0,
                b: 3.0,
                theta: 0.0,
                flags: 0,
                deblended: false,
                truncated: false,
                pixels,
            }],
            segmentation: None,
        });
        let sep = estimate_trailing(&frame, &analysis);
        let legacy = measured();
        assert_eq!(sep.policy, SEP_TRAILING_POLICY);
        assert_eq!(legacy.policy, LEGACY_TRAILING_POLICY);
        assert!(sep.unavailable_reason.is_none());
        assert!((sep.severity - legacy.severity).abs() < 1e-5);
        assert_eq!(sep.sensor_axis, Some(SensorAxis::X));
    }

    #[test]
    fn trailing_without_a_background_is_unavailable_not_round() {
        let frame = trailed_along_x();
        let mut analysis = background_from_luma(&frame);
        analysis.field = None;
        let estimate = estimate_trailing(&frame, &analysis);
        assert_eq!(
            estimate.unavailable_reason.as_deref(),
            Some("background_unavailable")
        );
        assert!(estimate.sensor_axis.is_none());
    }
}

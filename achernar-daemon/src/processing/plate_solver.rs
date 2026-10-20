//! Plate solving — Section 8.3 (pointing verification; M-4 pointing evidence).
//!
//! Turns an archived exposure into a sky position. The solver itself is an
//! external program — [`super::astap`] — the same way it is for NINA, which
//! contains no solver of its own and calls ASTAP or astrometry.net. This module
//! owns the contract around that program: what a hint means, what a solution
//! claims, and which failures are a result rather than an error.
//!
//! # What a solution claims
//!
//! The centre is in the star catalogue's frame: ICRS, which is J2000 to well
//! under an arcsecond. Mount drivers usually report JNow, and by 2026 the two
//! differ by about 22 arcmin of precession. Subtracting a mount position from a
//! solved one without converting epochs therefore manufactures a pointing
//! error of that size — which is why nothing here computes
//! `pointing_error_arcmin`. That needs the mount's coordinate at exposure time
//! and an epoch conversion, and neither lives here yet. As a search *hint*, a
//! JNow position is fine: 22 arcmin is far inside any useful search radius.
//!
//! Orientation is reported in the FITS pixel frame of the file that was
//! solved. Whether a `mirrored` frame means a diagonal in the optical train or
//! just a driver that writes rows top-down is a fact about the camera path,
//! not something a solve can know.
//!
//! # Failure is an outcome
//!
//! Clouds, a defocused frame or a wrong scale hint produce no solution. That is
//! an observation about the sky and the frame, recorded with its reason like
//! any other — never an invented position. Errors are reserved for the cases
//! where the solver could not be asked at all.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::astap::{self, AstapSolver};
use crate::config::PlateSolveConfig;
use crate::fits::FitsImageGeometry;

// ═══════════════════════════════════════════════════════════════════
//  Request
// ═══════════════════════════════════════════════════════════════════

/// Where to start looking, and how far to look.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PointingHint {
    /// Right ascension in hours — the unit `/mount/slew` takes.
    pub ra_hours: f64,
    pub dec_deg: f64,
    pub search_radius_deg: f64,
}

impl PointingHint {
    /// A hint outside the sphere is a caller mistake to surface, not a value
    /// to clamp: a clamped hint silently searches somewhere nobody asked for.
    pub fn validate(&self) -> Result<(), String> {
        if !(self.ra_hours.is_finite() && (0.0..24.0).contains(&self.ra_hours)) {
            return Err(format!(
                "ra_hours must be in [0, 24), got {}",
                self.ra_hours
            ));
        }
        if !(self.dec_deg.is_finite() && (-90.0..=90.0).contains(&self.dec_deg)) {
            return Err(format!(
                "dec_deg must be in [-90, 90], got {}",
                self.dec_deg
            ));
        }
        if !(self.search_radius_deg.is_finite()
            && self.search_radius_deg > 0.0
            && self.search_radius_deg <= 180.0)
        {
            return Err(format!(
                "search_radius_deg must be in (0, 180], got {}",
                self.search_radius_deg
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SolveRequest {
    /// `None` is a blind, all-sky solve.
    pub hint: Option<PointingHint>,
    /// Expected image scale, arcsec per (binned) pixel. Becomes the field
    /// height the solver searches for once the file's true height is known.
    /// `None` makes the solver search a range of scales, which is slower.
    pub pixel_scale_arcsec: Option<f64>,
}

impl SolveRequest {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(hint) = &self.hint {
            hint.validate()?;
        }
        match self.pixel_scale_arcsec {
            Some(scale) if !(scale.is_finite() && scale > 0.0) => Err(format!(
                "pixel_scale_arcsec must be finite and > 0, got {scale}"
            )),
            _ => Ok(()),
        }
    }

    /// Field height the scale implies for an image `height_px` tall, degrees.
    pub fn fov_height_deg(&self, height_px: u32) -> Option<f64> {
        self.pixel_scale_arcsec
            .map(|scale| scale * f64::from(height_px) / 3600.0)
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Solution
// ═══════════════════════════════════════════════════════════════════

/// A gnomonic (`RA---TAN` / `DEC--TAN`) world coordinate system.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TanWcs {
    /// Reference pixel in the FITS convention: 1-based, pixel centres at
    /// integers, so the centre of an N-pixel axis is `(N + 1) / 2`.
    pub crpix: [f64; 2],
    /// Sky position of the reference pixel, degrees.
    pub crval: [f64; 2],
    /// `CDi_j`, row-major: pixel offset to intermediate world coordinates
    /// (east, north), degrees per pixel.
    pub cd: [[f64; 2]; 2],
    /// Image size in pixels (`NAXIS1`, `NAXIS2`).
    pub naxis: [u32; 2],
}

impl TanWcs {
    pub fn determinant(&self) -> f64 {
        self.cd[0][0] * self.cd[1][1] - self.cd[0][1] * self.cd[1][0]
    }

    /// Sky position (RA, Dec) of FITS pixel `(x, y)`, degrees.
    pub fn pixel_to_sky(&self, x: f64, y: f64) -> (f64, f64) {
        let dx = x - self.crpix[0];
        let dy = y - self.crpix[1];
        // Standard coordinates on the tangent plane, radians.
        let xi = (self.cd[0][0] * dx + self.cd[0][1] * dy).to_radians();
        let eta = (self.cd[1][0] * dx + self.cd[1][1] * dy).to_radians();
        let ra0 = self.crval[0].to_radians();
        let dec0 = self.crval[1].to_radians();
        let denominator = dec0.cos() - eta * dec0.sin();
        let ra = ra0 + xi.atan2(denominator);
        let dec = (dec0.sin() + eta * dec0.cos()).atan2(xi.hypot(denominator));
        (normalize_degrees(ra.to_degrees()), dec.to_degrees())
    }

    fn centre_pixel(&self) -> (f64, f64) {
        (
            (f64::from(self.naxis[0]) + 1.0) / 2.0,
            (f64::from(self.naxis[1]) + 1.0) / 2.0,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlateSolution {
    /// Image centre, ICRS (J2000), degrees. Not JNow — see the module docs.
    pub ra_deg: f64,
    pub dec_deg: f64,
    /// The same centre in hours, for `/mount/slew` and `/mount/sync`.
    pub ra_hours: f64,
    /// Image scale, arcsec per pixel: `sqrt(|det CD|)`, the geometric mean of
    /// the two axes, so it stays meaningful if the pixels are not square.
    pub pixel_scale_arcsec: f64,
    /// Position angle of the image +Y axis (increasing FITS row), degrees east
    /// of north, in `[0, 360)`.
    pub position_angle_deg: f64,
    /// `det CD > 0`: the pixel frame has the handedness of a mirror image of
    /// the sky. Relative to the file, not necessarily to the optics.
    pub mirrored: bool,
    /// Angular distance between opposite edge midpoints, degrees.
    pub field_width_deg: f64,
    pub field_height_deg: f64,
    pub wcs: TanWcs,
    /// Text the solver attached to an otherwise valid solution.
    pub warning: Option<String>,
}

impl PlateSolution {
    /// Derive every reported quantity from the WCS itself, so the numbers
    /// cannot disagree with the transform they claim to summarise. Solver
    /// conveniences such as `CROTA2` and `CDELT` are deliberately not read:
    /// their sign conventions differ between programs, and the CD matrix is
    /// the one thing every solver writes the same way.
    pub fn from_wcs(wcs: TanWcs, warning: Option<String>) -> Result<Self, String> {
        let finite = wcs
            .crpix
            .iter()
            .chain(wcs.crval.iter())
            .chain(wcs.cd.iter().flatten())
            .all(|v| v.is_finite());
        if !finite {
            return Err("WCS contains a non-finite value".into());
        }
        if wcs.naxis.contains(&0) {
            return Err("WCS image size is zero".into());
        }
        if !(-90.0..=90.0).contains(&wcs.crval[1]) {
            return Err(format!(
                "WCS declination {} is off the sphere",
                wcs.crval[1]
            ));
        }
        let determinant = wcs.determinant();
        if !(determinant.is_finite() && determinant != 0.0) {
            return Err("WCS CD matrix is singular".into());
        }

        let (cx, cy) = wcs.centre_pixel();
        let (ra_deg, dec_deg) = wcs.pixel_to_sky(cx, cy);
        let width_px = f64::from(wcs.naxis[0]);
        let height_px = f64::from(wcs.naxis[1]);
        let field_width_deg = separation_degrees(
            wcs.pixel_to_sky(0.5, cy),
            wcs.pixel_to_sky(width_px + 0.5, cy),
        );
        let field_height_deg = separation_degrees(
            wcs.pixel_to_sky(cx, 0.5),
            wcs.pixel_to_sky(cx, height_px + 0.5),
        );

        Ok(Self {
            ra_deg,
            dec_deg,
            ra_hours: ra_deg / 15.0,
            pixel_scale_arcsec: determinant.abs().sqrt() * 3600.0,
            // +Y on the sky is the CD matrix's second column, (east, north).
            position_angle_deg: normalize_degrees(wcs.cd[0][1].atan2(wcs.cd[1][1]).to_degrees()),
            mirrored: determinant > 0.0,
            field_width_deg,
            field_height_deg,
            wcs,
            warning,
        })
    }
}

/// Why a solve produced no position. The serialized names are stable API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolveFailureKind {
    /// Stars were found but matched nothing within the search: clouds, wrong
    /// scale, wrong hint, or a field outside the installed star database.
    NoSolution,
    /// Too few stars to try: cloud, dew, a closed roof, a bad focus.
    NotEnoughStars,
    ImageUnreadable,
    StarDatabaseMissing,
    StarDatabaseUnreadable,
    Timeout,
    /// The executable could not be started.
    SolverUnavailable,
    /// Anything else: an unexpected exit, a malformed or inconsistent result.
    SolverError,
}

impl SolveFailureKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoSolution => "no_solution",
            Self::NotEnoughStars => "not_enough_stars",
            Self::ImageUnreadable => "image_unreadable",
            Self::StarDatabaseMissing => "star_database_missing",
            Self::StarDatabaseUnreadable => "star_database_unreadable",
            Self::Timeout => "timeout",
            Self::SolverUnavailable => "solver_unavailable",
            Self::SolverError => "solver_error",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SolveFailure {
    pub kind: SolveFailureKind,
    pub detail: String,
}

impl SolveFailure {
    pub fn new(kind: SolveFailureKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
        }
    }
}

/// One run of the solver, successful or not, with what it was given.
#[derive(Debug, Clone)]
pub struct SolveAttempt {
    pub outcome: Result<PlateSolution, SolveFailure>,
    /// Wall time of the solver process, including startup.
    pub elapsed_ms: u64,
    /// Executable name and arguments, with the private working path reduced
    /// to its file name so the record is reproducible from the archive.
    pub command: Vec<String>,
    pub geometry: FitsImageGeometry,
    /// Field height handed to the solver; `None` means it searched for one.
    pub fov_height_deg: Option<f64>,
}

/// The solver could not be asked. Unlike [`SolveFailure`], nothing about the
/// sky was learned, so nothing is recorded against the frame.
#[derive(Debug, thiserror::Error)]
pub enum SolveError {
    #[error("Plate solver unavailable: {0}")]
    Unavailable(String),
    #[error("A plate solve is already running")]
    Busy,
    #[error("Cannot read the frame to solve: {0}")]
    Input(String),
    #[error(
        "Archived frame does not match its recorded checksum (expected {expected}, found {actual})"
    )]
    ChecksumMismatch { expected: String, actual: String },
    #[error("Cannot prepare the solver's working directory: {0}")]
    WorkingDirectory(String),
}

// ═══════════════════════════════════════════════════════════════════
//  Solver
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutableSource {
    /// `[plate_solve] astap_path`.
    Configured,
    /// Found at a standard install location or on `PATH`.
    Discovered,
}

#[derive(Debug, Clone, Serialize)]
pub struct SolverStatus {
    pub solver: &'static str,
    pub available: bool,
    pub executable: Option<String>,
    pub executable_source: Option<ExecutableSource>,
    /// Star database directory passed to the solver; `None` leaves the choice
    /// to ASTAP's built-in default.
    pub database_dir: Option<String>,
    pub unavailable_reason: Option<String>,
    pub busy: bool,
    pub default_search_radius_deg: f64,
    pub timeout_s: u64,
}

pub struct PlateSolver {
    backend: Result<AstapSolver, String>,
    source: Option<ExecutableSource>,
    default_search_radius_deg: f64,
    timeout: Duration,
    /// One solve at a time. ASTAP already uses every core; a second process
    /// only slows both and competes with acquisition on a small computer.
    in_flight: tokio::sync::Semaphore,
    /// Parent of the per-solve working directories.
    work_root: PathBuf,
}

impl PlateSolver {
    /// Resolve the executable and validate the configuration.
    ///
    /// Never fails: an unusable configuration yields a solver that reports
    /// why it is unavailable. Present-but-wrong values are named in that
    /// reason rather than replaced with defaults nobody chose.
    pub fn from_config(config: &PlateSolveConfig) -> Self {
        let mut solver = Self::disabled("");
        solver.default_search_radius_deg = config.search_radius_deg;
        solver.timeout = Duration::from_secs(config.timeout_s);

        if let Err(reason) = validate_config(config) {
            solver.backend = Err(reason);
            return solver;
        }
        let (executable, source) = match config.astap_path.as_deref() {
            Some(path) => {
                let path = PathBuf::from(path);
                if !path.is_file() {
                    solver.backend = Err(format!(
                        "[plate_solve] astap_path {} does not exist or is not a file",
                        path.display()
                    ));
                    return solver;
                }
                (path, ExecutableSource::Configured)
            }
            None => match astap::discover() {
                Some(path) => (path, ExecutableSource::Discovered),
                None => {
                    solver.backend = Err(format!(
                        "ASTAP was not found. Install it with a star database such as D50 \
                         (https://www.hnsky.org/astap.htm), or set [plate_solve] astap_path. \
                         Looked in {} and on PATH",
                        astap::searched_locations()
                            .iter()
                            .map(|p| p.display().to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                    return solver;
                }
            },
        };
        solver.backend = Ok(AstapSolver::new(executable, config));
        solver.source = Some(source);
        solver
    }

    pub fn disabled(reason: impl Into<String>) -> Self {
        let defaults = PlateSolveConfig::default();
        Self {
            backend: Err(reason.into()),
            source: None,
            default_search_radius_deg: defaults.search_radius_deg,
            timeout: Duration::from_secs(defaults.timeout_s),
            in_flight: tokio::sync::Semaphore::new(1),
            work_root: std::env::temp_dir().join("achernar-platesolve"),
        }
    }

    pub fn default_search_radius_deg(&self) -> f64 {
        self.default_search_radius_deg
    }

    pub fn is_available(&self) -> bool {
        self.backend.is_ok()
    }

    pub fn status(&self) -> SolverStatus {
        SolverStatus {
            solver: "astap",
            available: self.backend.is_ok(),
            executable: self
                .backend
                .as_ref()
                .ok()
                .map(|b| b.executable().display().to_string()),
            executable_source: self.source,
            database_dir: self
                .backend
                .as_ref()
                .ok()
                .and_then(|b| b.database_dir())
                .map(|d| d.display().to_string()),
            unavailable_reason: self.backend.as_ref().err().cloned(),
            busy: self.in_flight.available_permits() == 0,
            default_search_radius_deg: self.default_search_radius_deg,
            timeout_s: self.timeout.as_secs(),
        }
    }

    /// Solve an archived FITS file without touching it.
    ///
    /// The bytes are read once, checked against `expected_sha256` when the
    /// archive recorded one, and copied into a private working directory: the
    /// solver writes its result files beside its input, and an archived bundle
    /// is immutable. The copy is what is solved, so the hash names exactly the
    /// bytes the solution describes.
    pub async fn solve_archived_fits(
        &self,
        path: &Path,
        expected_sha256: Option<&str>,
        request: &SolveRequest,
    ) -> Result<SolveAttempt, SolveError> {
        let backend = self
            .backend
            .as_ref()
            .map_err(|reason| SolveError::Unavailable(reason.clone()))?;
        let _permit = self.in_flight.try_acquire().map_err(|_| SolveError::Busy)?;

        let work = WorkingDir::create(&self.work_root)
            .map_err(|e| SolveError::WorkingDirectory(e.to_string()))?;
        let input = work.path().join(format!("{}.fits", astap::INPUT_STEM));
        let geometry = {
            let source = path.to_path_buf();
            let destination = input.clone();
            let expected = expected_sha256.map(str::to_ascii_lowercase);
            tokio::task::spawn_blocking(move || prepare_input(&source, &destination, expected))
                .await
                .map_err(|e| SolveError::Input(format!("preparation task failed: {e}")))??
        };

        let fov_height_deg = request.fov_height_deg(geometry.height);
        let (outcome, elapsed_ms, command) = backend
            .solve(&input, work.path(), request, &geometry, fov_height_deg)
            .await;
        Ok(SolveAttempt {
            outcome,
            elapsed_ms,
            command,
            geometry,
            fov_height_deg,
        })
    }

    #[cfg(test)]
    pub(crate) fn with_backend_for_test(backend: AstapSolver, work_root: PathBuf) -> Self {
        let mut solver = Self::disabled("");
        solver.backend = Ok(backend);
        solver.source = Some(ExecutableSource::Configured);
        solver.work_root = work_root;
        solver
    }
}

fn validate_config(config: &PlateSolveConfig) -> Result<(), String> {
    if !(config.search_radius_deg.is_finite()
        && config.search_radius_deg > 0.0
        && config.search_radius_deg <= 180.0)
    {
        return Err(format!(
            "[plate_solve] search_radius_deg must be in (0, 180], got {}",
            config.search_radius_deg
        ));
    }
    if config.timeout_s == 0 {
        return Err("[plate_solve] timeout_s must be > 0".into());
    }
    if config.max_stars == 0 {
        return Err("[plate_solve] max_stars must be > 0".into());
    }
    Ok(())
}

/// Read, verify and copy the frame; return its geometry.
fn prepare_input(
    source: &Path,
    destination: &Path,
    expected_sha256: Option<String>,
) -> Result<FitsImageGeometry, SolveError> {
    let bytes = std::fs::read(source)
        .map_err(|e| SolveError::Input(format!("{}: {e}", source.display())))?;
    if let Some(expected) = expected_sha256 {
        let actual = format!("{:x}", Sha256::digest(&bytes));
        if actual != expected {
            return Err(SolveError::ChecksumMismatch { expected, actual });
        }
    }
    let geometry = crate::fits::image_geometry(&bytes)
        .map_err(|e| SolveError::Input(format!("{}: {e}", source.display())))?;
    std::fs::write(destination, &bytes)
        .map_err(|e| SolveError::WorkingDirectory(format!("{}: {e}", destination.display())))?;
    Ok(geometry)
}

/// A per-solve directory, removed with everything in it when dropped —
/// including when the request is cancelled mid-solve.
struct WorkingDir(PathBuf);

impl WorkingDir {
    fn create(root: &Path) -> std::io::Result<Self> {
        let path = root.join(uuid::Uuid::new_v4().to_string());
        std::fs::create_dir_all(&path)?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for WorkingDir {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.0) {
            tracing::warn!(
                "Could not remove plate-solve directory {}: {error}",
                self.0.display()
            );
        }
    }
}

fn normalize_degrees(angle: f64) -> f64 {
    let wrapped = angle.rem_euclid(360.0);
    // rem_euclid can return exactly 360.0 for tiny negative inputs.
    if wrapped >= 360.0 {
        0.0
    } else {
        wrapped
    }
}

/// Great-circle distance, degrees. The Vincenty form stays accurate for both
/// tiny and near-antipodal separations, where the plain cosine formula does not.
fn separation_degrees((ra1, dec1): (f64, f64), (ra2, dec2): (f64, f64)) -> f64 {
    let (ra1, dec1, ra2, dec2) = (
        ra1.to_radians(),
        dec1.to_radians(),
        ra2.to_radians(),
        dec2.to_radians(),
    );
    let dra = ra2 - ra1;
    let numerator = (dec2.cos() * dra.sin())
        .hypot(dec1.cos() * dec2.sin() - dec1.sin() * dec2.cos() * dra.cos());
    let denominator = dec1.sin() * dec2.sin() + dec1.cos() * dec2.cos() * dra.cos();
    numerator.atan2(denominator).to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Forward gnomonic projection: the inverse of `pixel_to_sky`, written
    /// independently from the textbook formulas so the test does not grade
    /// the implementation against itself.
    fn sky_to_pixel(wcs: &TanWcs, ra: f64, dec: f64) -> (f64, f64) {
        let (ra, dec) = (ra.to_radians(), dec.to_radians());
        let (ra0, dec0) = (wcs.crval[0].to_radians(), wcs.crval[1].to_radians());
        let cos_c = dec0.sin() * dec.sin() + dec0.cos() * dec.cos() * (ra - ra0).cos();
        let xi = (dec.cos() * (ra - ra0).sin() / cos_c).to_degrees();
        let eta = ((dec0.cos() * dec.sin() - dec0.sin() * dec.cos() * (ra - ra0).cos()) / cos_c)
            .to_degrees();
        let det = wcs.determinant();
        let dx = (wcs.cd[1][1] * xi - wcs.cd[0][1] * eta) / det;
        let dy = (-wcs.cd[1][0] * xi + wcs.cd[0][0] * eta) / det;
        (dx + wcs.crpix[0], dy + wcs.crpix[1])
    }

    /// Position angle of `to` as seen from `from`, degrees east of north.
    fn position_angle((ra1, dec1): (f64, f64), (ra2, dec2): (f64, f64)) -> f64 {
        let (ra1, dec1, ra2, dec2) = (
            ra1.to_radians(),
            dec1.to_radians(),
            ra2.to_radians(),
            dec2.to_radians(),
        );
        let dra = ra2 - ra1;
        let pa = (dra.sin() * dec2.cos())
            .atan2(dec1.cos() * dec2.sin() - dec1.sin() * dec2.cos() * dra.cos());
        normalize_degrees(pa.to_degrees())
    }

    /// A camera whose +Y axis points at position angle `pa_deg`, built from
    /// the geometry rather than from the formula under test: +Y is the unit
    /// vector (sin pa, cos pa) in (east, north); for an unmirrored view +X is
    /// +Y turned 90 degrees toward west, and a mirror flips +X.
    fn camera(pa_deg: f64, scale_arcsec: f64, mirrored: bool) -> TanWcs {
        let s = scale_arcsec / 3600.0;
        let (sin, cos) = pa_deg.to_radians().sin_cos();
        let (x_east, x_north) = if mirrored { (cos, -sin) } else { (-cos, sin) };
        TanWcs {
            crpix: [2000.5, 1500.5],
            crval: [83.822, -5.391],
            cd: [[s * x_east, s * sin], [s * x_north, s * cos]],
            naxis: [4000, 3000],
        }
    }

    #[test]
    fn centre_is_the_reference_value_when_crpix_is_the_image_centre() {
        let solution = PlateSolution::from_wcs(camera(0.0, 1.2, false), None).unwrap();
        assert!((solution.ra_deg - 83.822).abs() < 1e-9);
        assert!((solution.dec_deg + 5.391).abs() < 1e-9);
        assert!((solution.ra_hours - 83.822 / 15.0).abs() < 1e-9);
    }

    #[test]
    fn centre_is_projected_from_the_centre_pixel_when_crpix_is_elsewhere() {
        // Reference pixel near a corner: CRVAL is no longer the image centre,
        // and reporting it as one would be wrong by most of a degree.
        let mut wcs = camera(25.0, 2.0, false);
        wcs.crpix = [100.0, 2900.0];
        let solution = PlateSolution::from_wcs(wcs, None).unwrap();
        let (x, y) = sky_to_pixel(&wcs, solution.ra_deg, solution.dec_deg);
        assert!(
            (x - 2000.5).abs() < 1e-6 && (y - 1500.5).abs() < 1e-6,
            "({x}, {y})"
        );
        let crval_distance = separation_degrees(
            (solution.ra_deg, solution.dec_deg),
            (wcs.crval[0], wcs.crval[1]),
        );
        assert!(crval_distance > 0.5, "{crval_distance}");
    }

    #[test]
    fn pixel_to_sky_inverts_the_gnomonic_projection() {
        for (crval, pa) in [
            ([83.822, -5.391], 0.0),
            ([359.9, 12.0], 137.0),     // field straddling RA 0h
            ([10.68, 89.2], 301.0),     // near the pole
            ([201.365, -43.019], 90.0), // southern sky
        ] {
            let mut wcs = camera(pa, 3.5, pa > 200.0);
            wcs.crval = crval;
            for (x, y) in [
                (1.0, 1.0),
                (4000.0, 3000.0),
                (17.25, 2988.5),
                (2000.5, 1500.5),
            ] {
                let (ra, dec) = wcs.pixel_to_sky(x, y);
                let (bx, by) = sky_to_pixel(&wcs, ra, dec);
                assert!(
                    (bx - x).abs() < 1e-6 && (by - y).abs() < 1e-6,
                    "{crval:?} pixel ({x}, {y}) came back as ({bx}, {by})"
                );
            }
        }
    }

    #[test]
    fn position_angle_is_where_the_y_axis_points_on_the_sky() {
        for pa in [0.0, 30.0, 90.0, 179.0, 200.0, 359.0] {
            for mirrored in [false, true] {
                let wcs = camera(pa, 1.5, mirrored);
                let solution = PlateSolution::from_wcs(wcs, None).unwrap();
                // Walk up the image from the centre and measure the direction
                // on the sky. This is what a rotator or framing tool means by
                // position angle, independent of any matrix convention.
                let centre = wcs.pixel_to_sky(2000.5, 1500.5);
                let above = wcs.pixel_to_sky(2000.5, 1600.5);
                let measured = position_angle(centre, above);
                let reported = solution.position_angle_deg;
                let difference = (measured - reported + 540.0).rem_euclid(360.0) - 180.0;
                assert!(
                    difference.abs() < 1e-6,
                    "pa {pa} mirrored {mirrored}: reported {reported}, measured {measured}"
                );
                assert_eq!(solution.mirrored, mirrored);
            }
        }
    }

    #[test]
    fn east_lies_counterclockwise_from_north_in_an_unmirrored_frame() {
        // North up: an unmirrored sky has east on the left, so RA increases
        // toward -X. A frame where it increases toward +X is a mirror image.
        let wcs = camera(0.0, 2.0, false);
        let left = wcs.pixel_to_sky(1000.0, 1500.5);
        let right = wcs.pixel_to_sky(3000.0, 1500.5);
        assert!(left.0 > right.0);
        assert!(wcs.determinant() < 0.0);
    }

    #[test]
    fn scale_and_field_size_come_from_the_cd_matrix() {
        let solution = PlateSolution::from_wcs(camera(42.0, 1.25, false), None).unwrap();
        assert!((solution.pixel_scale_arcsec - 1.25).abs() < 1e-9);
        // 4000 x 1.25" = 1.3889 deg, 3000 x 1.25" = 1.0417 deg. The tangent
        // plane stretch over a degree is a few parts per million.
        assert!((solution.field_width_deg - 4000.0 * 1.25 / 3600.0).abs() < 1e-4);
        assert!((solution.field_height_deg - 3000.0 * 1.25 / 3600.0).abs() < 1e-4);
    }

    #[test]
    fn a_wcs_that_cannot_describe_the_sky_is_refused() {
        let mut singular = camera(0.0, 1.0, false);
        singular.cd = [[1e-4, 1e-4], [1e-4, 1e-4]];
        assert!(PlateSolution::from_wcs(singular, None).is_err());

        let mut nan = camera(0.0, 1.0, false);
        nan.crval[0] = f64::NAN;
        assert!(PlateSolution::from_wcs(nan, None).is_err());

        let mut off_sphere = camera(0.0, 1.0, false);
        off_sphere.crval[1] = 91.0;
        assert!(PlateSolution::from_wcs(off_sphere, None).is_err());

        let mut empty = camera(0.0, 1.0, false);
        empty.naxis = [0, 3000];
        assert!(PlateSolution::from_wcs(empty, None).is_err());
    }

    #[test]
    fn hints_outside_the_sphere_are_refused_not_clamped() {
        let good = PointingHint {
            ra_hours: 5.5,
            dec_deg: -5.4,
            search_radius_deg: 10.0,
        };
        assert!(good.validate().is_ok());
        for bad in [
            PointingHint {
                ra_hours: 24.0,
                ..good
            },
            PointingHint {
                ra_hours: -0.1,
                ..good
            },
            PointingHint {
                dec_deg: 90.5,
                ..good
            },
            PointingHint {
                search_radius_deg: 0.0,
                ..good
            },
            PointingHint {
                search_radius_deg: 181.0,
                ..good
            },
            PointingHint {
                ra_hours: f64::NAN,
                ..good
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?} should be refused");
        }
        let request = SolveRequest {
            hint: None,
            pixel_scale_arcsec: Some(-1.0),
        };
        assert!(request.validate().is_err());
    }

    #[test]
    fn scale_becomes_the_field_height_of_the_actual_image() {
        let request = SolveRequest {
            hint: None,
            pixel_scale_arcsec: Some(1.8),
        };
        assert!((request.fov_height_deg(3000).unwrap() - 1.5).abs() < 1e-12);
        assert_eq!(SolveRequest::default().fov_height_deg(3000), None);
    }

    #[test]
    fn configuration_mistakes_disable_the_solver_with_a_reason() {
        let wrong_radius = PlateSolveConfig {
            search_radius_deg: 0.0,
            ..PlateSolveConfig::default()
        };
        let status = PlateSolver::from_config(&wrong_radius).status();
        assert!(!status.available);
        assert!(status
            .unavailable_reason
            .unwrap()
            .contains("search_radius_deg"));

        let missing = PlateSolveConfig {
            astap_path: Some("/definitely/not/an/astap/binary".into()),
            ..PlateSolveConfig::default()
        };
        let status = PlateSolver::from_config(&missing).status();
        assert!(!status.available);
        assert!(status
            .unavailable_reason
            .unwrap()
            .contains("does not exist"));
    }

    #[tokio::test]
    async fn a_disabled_solver_refuses_before_reading_anything() {
        let solver = PlateSolver::disabled("not configured");
        let error = solver
            .solve_archived_fits(
                Path::new("/nonexistent.fits"),
                None,
                &SolveRequest::default(),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, SolveError::Unavailable(reason) if reason == "not configured"));
    }
}

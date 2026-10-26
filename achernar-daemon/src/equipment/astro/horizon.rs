//! Horizon profile (WP1A-3; §2.11 `[mount.safety] horizon_profile_path`,
//! M-k): the local horizon altitude as a function of azimuth.
//!
//! **File format.** One `az_deg,alt_deg` pair per line: azimuth from north
//! through east in `[0, 360]`, altitude in `[-90, 90]`. Commas, semicolons,
//! tabs or spaces separate the two numbers, so NINA-style space-separated
//! files load too. Blank lines, `#` comments and a non-numeric header line
//! (`az_deg,alt_deg`) are skipped. Points may come in any order; 360 is the
//! same azimuth as 0.
//!
//! **Interpolation.** Linear in azimuth between neighbouring points,
//! wrapping across north. One point gives a flat horizon.
//!
//! A configured file that cannot be read or parsed is an error, never an
//! empty (flat 0°) horizon: callers must surface it (D5).

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::norm_deg;
use crate::equipment::types::AltAz;

/// Largest horizon file accepted, bytes.
pub const MAX_FILE_BYTES: u64 = 1024 * 1024;
/// Most points accepted.
pub const MAX_POINTS: usize = 20_000;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HorizonPoint {
    pub az_deg: f64,
    pub alt_deg: f64,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum HorizonError {
    #[error("cannot read horizon profile {path}: {message}")]
    Io { path: String, message: String },
    #[error("horizon profile {path} is larger than {} bytes", MAX_FILE_BYTES)]
    TooLarge { path: String },
    #[error("horizon profile line {line}: {message}")]
    Parse { line: usize, message: String },
    #[error("horizon profile has no points")]
    Empty,
    #[error("horizon profile has more than {} points", MAX_POINTS)]
    TooManyPoints,
    #[error("horizon profile lists azimuth {az_deg}° twice")]
    DuplicateAzimuth { az_deg: f64 },
}

/// A validated horizon profile, sorted by azimuth in `[0, 360)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HorizonProfile {
    points: Vec<HorizonPoint>,
}

impl HorizonProfile {
    /// A flat horizon at `alt_deg`.
    pub fn flat(alt_deg: f64) -> Self {
        Self {
            points: vec![HorizonPoint {
                az_deg: 0.0,
                alt_deg: alt_deg.clamp(-90.0, 90.0),
            }],
        }
    }

    /// Validates and sorts the points.
    pub fn from_points(points: Vec<HorizonPoint>) -> Result<Self, HorizonError> {
        if points.is_empty() {
            return Err(HorizonError::Empty);
        }
        if points.len() > MAX_POINTS {
            return Err(HorizonError::TooManyPoints);
        }
        let mut out = Vec::with_capacity(points.len());
        for (index, p) in points.into_iter().enumerate() {
            validate(p).map_err(|message| HorizonError::Parse {
                line: index + 1,
                message,
            })?;
            out.push(HorizonPoint {
                az_deg: norm_deg(p.az_deg),
                alt_deg: p.alt_deg,
            });
        }
        out.sort_by(|a, b| a.az_deg.total_cmp(&b.az_deg));
        if let Some(pair) = out.windows(2).find(|w| w[0].az_deg == w[1].az_deg) {
            return Err(HorizonError::DuplicateAzimuth {
                az_deg: pair[0].az_deg,
            });
        }
        Ok(Self { points: out })
    }

    /// Parses the file format described in the module docs.
    pub fn parse(text: &str) -> Result<Self, HorizonError> {
        let mut points = Vec::new();
        let mut seen_data = false;
        for (index, raw) in text.lines().enumerate() {
            let line_no = index + 1;
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line
                .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
                .filter(|f| !f.is_empty())
                .collect();
            let numbers: Option<Vec<f64>> = fields.iter().map(|f| f.parse().ok()).collect();
            match numbers {
                Some(n) if n.len() == 2 => {
                    let p = HorizonPoint {
                        az_deg: n[0],
                        alt_deg: n[1],
                    };
                    validate(p).map_err(|message| HorizonError::Parse {
                        line: line_no,
                        message,
                    })?;
                    points.push(p);
                    seen_data = true;
                }
                // One header line is allowed before the data.
                None if !seen_data && points.is_empty() => {
                    seen_data = true;
                }
                _ => {
                    return Err(HorizonError::Parse {
                        line: line_no,
                        message: format!("expected `az_deg,alt_deg`, got `{line}`"),
                    })
                }
            }
        }
        Self::from_points(points)
    }

    /// Reads and parses a horizon file.
    pub fn load(path: &Path) -> Result<Self, HorizonError> {
        let shown = path.display().to_string();
        let io = |e: std::io::Error| HorizonError::Io {
            path: shown.clone(),
            message: e.to_string(),
        };
        let meta = std::fs::metadata(path).map_err(io)?;
        if meta.len() > MAX_FILE_BYTES {
            return Err(HorizonError::TooLarge { path: shown });
        }
        let text = std::fs::read_to_string(path).map_err(io)?;
        Self::parse(&text)
    }

    /// The points, sorted by azimuth.
    pub fn points(&self) -> &[HorizonPoint] {
        &self.points
    }

    /// Horizon altitude at `az_deg`, interpolated linearly with wrap. A
    /// non-finite azimuth gets the highest point of the profile
    /// (conservative for a limit check).
    pub fn altitude_at(&self, az_deg: f64) -> f64 {
        let pts = &self.points;
        if pts.len() == 1 || !az_deg.is_finite() {
            return pts.iter().map(|p| p.alt_deg).fold(f64::MIN, f64::max);
        }
        let az = norm_deg(az_deg);
        let upper = pts.partition_point(|p| p.az_deg <= az);
        let (a, b, span_a) = if upper == 0 || upper == pts.len() {
            // Between the last point and the first, across north.
            let last = pts[pts.len() - 1];
            let first = pts[0];
            let from = last.az_deg;
            let to = first.az_deg + 360.0;
            let x = if az >= from { az } else { az + 360.0 };
            (last, first, (x - from) / (to - from))
        } else {
            let a = pts[upper - 1];
            let b = pts[upper];
            (a, b, (az - a.az_deg) / (b.az_deg - a.az_deg))
        };
        a.alt_deg + span_a * (b.alt_deg - a.alt_deg)
    }

    /// Altitude above the horizon profile (negative = below it).
    pub fn clearance_deg(&self, position: AltAz) -> f64 {
        position.alt_deg - self.altitude_at(position.az_deg)
    }

    /// True when `position` is at least `margin_deg` above the profile.
    pub fn is_clear(&self, position: AltAz, margin_deg: f64) -> bool {
        self.clearance_deg(position) >= margin_deg
    }
}

fn validate(p: HorizonPoint) -> Result<(), String> {
    if !(p.az_deg.is_finite() && (0.0..=360.0).contains(&p.az_deg)) {
        return Err(format!("azimuth {} is outside [0, 360]", p.az_deg));
    }
    if !(p.alt_deg.is_finite() && (-90.0..=90.0).contains(&p.alt_deg)) {
        return Err(format!("altitude {} is outside [-90, 90]", p.alt_deg));
    }
    Ok(())
}

/// Loads the profile named by `[mount.safety] horizon_profile_path`
/// (`MountSafetyConfig::horizon_profile()`): `Ok(None)` when none is
/// configured, an error when one is configured but unusable.
pub fn load_configured(path: Option<&str>) -> Result<Option<HorizonProfile>, HorizonError> {
    match path.map(str::trim).filter(|p| !p.is_empty()) {
        None => Ok(None),
        Some(p) => HorizonProfile::load(Path::new(p)).map(Some),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
# horizon for the back garden
az_deg,alt_deg
0, 10
90 20
180;5   # trees
270\t30
";

    #[test]
    fn parses_mixed_separators_header_and_comments() {
        let h = HorizonProfile::parse(SAMPLE).unwrap();
        let azimuths: Vec<f64> = h.points().iter().map(|p| p.az_deg).collect();
        assert_eq!(azimuths, vec![0.0, 90.0, 180.0, 270.0]);
        assert_eq!(h.altitude_at(90.0), 20.0);
        assert_eq!(h.altitude_at(45.0), 15.0);
        assert_eq!(h.altitude_at(135.0), 12.5);
    }

    #[test]
    fn interpolates_across_north() {
        let h = HorizonProfile::parse("350,20\n10,40\n180,0\n").unwrap();
        assert!((h.altitude_at(0.0) - 30.0).abs() < 1e-12);
        assert!((h.altitude_at(360.0) - 30.0).abs() < 1e-12);
        assert!((h.altitude_at(355.0) - 25.0).abs() < 1e-12);
        assert!((h.altitude_at(5.0) - 35.0).abs() < 1e-12);
        assert!((h.altitude_at(-5.0) - 25.0).abs() < 1e-12);
        // 360 is the same azimuth as 0 and sorts first.
        let w = HorizonProfile::parse("360,12\n180,2\n").unwrap();
        assert_eq!(w.points()[0].az_deg, 0.0);
        assert!((w.altitude_at(90.0) - 7.0).abs() < 1e-12);
        assert!((w.altitude_at(270.0) - 7.0).abs() < 1e-12);
    }

    #[test]
    fn single_point_and_flat_profiles_are_constant() {
        let h = HorizonProfile::parse("123,7.5").unwrap();
        for az in [0.0, 123.0, 300.0] {
            assert_eq!(h.altitude_at(az), 7.5);
        }
        assert_eq!(HorizonProfile::flat(15.0).altitude_at(200.0), 15.0);
    }

    #[test]
    fn clearance_and_visibility() {
        let h = HorizonProfile::parse(SAMPLE).unwrap();
        let star = AltAz {
            alt_deg: 25.0,
            az_deg: 90.0,
        };
        assert_eq!(h.clearance_deg(star), 5.0);
        assert!(h.is_clear(star, 5.0));
        assert!(!h.is_clear(star, 5.1));
        let low = AltAz {
            alt_deg: 25.0,
            az_deg: 270.0,
        };
        assert!(!h.is_clear(low, 0.0));
    }

    #[test]
    fn bad_input_is_an_error_never_a_flat_horizon() {
        assert_eq!(HorizonProfile::parse(""), Err(HorizonError::Empty));
        assert_eq!(
            HorizonProfile::parse("# only a comment\n"),
            Err(HorizonError::Empty)
        );
        assert!(matches!(
            HorizonProfile::parse("0,10\nfoo,bar\n"),
            Err(HorizonError::Parse { line: 2, .. })
        ));
        assert!(matches!(
            HorizonProfile::parse("0,10,3\n"),
            Err(HorizonError::Parse { line: 1, .. })
        ));
        assert!(matches!(
            HorizonProfile::parse("400,10\n"),
            Err(HorizonError::Parse { line: 1, .. })
        ));
        assert!(matches!(
            HorizonProfile::parse("10,95\n"),
            Err(HorizonError::Parse { line: 1, .. })
        ));
        assert!(matches!(
            HorizonProfile::parse("10,NaN\n"),
            Err(HorizonError::Parse { line: 1, .. })
        ));
        assert_eq!(
            HorizonProfile::parse("0,10\n360,12\n"),
            Err(HorizonError::DuplicateAzimuth { az_deg: 0.0 })
        );
    }

    #[test]
    fn loads_files_and_reports_missing_ones() {
        let dir = std::env::temp_dir().join(format!("achernar-horizon-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("horizon.csv");
        std::fs::write(&path, SAMPLE).unwrap();
        let loaded = load_configured(Some(path.to_str().unwrap()))
            .unwrap()
            .unwrap();
        assert_eq!(loaded.points().len(), 4);
        assert_eq!(load_configured(None), Ok(None));
        assert_eq!(load_configured(Some("  ")), Ok(None));
        let missing = dir.join("missing.csv");
        assert!(matches!(
            load_configured(Some(missing.to_str().unwrap())),
            Err(HorizonError::Io { .. })
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

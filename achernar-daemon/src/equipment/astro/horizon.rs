//! Horizon profile (WP1A-3; §2.11 `[mount.safety] horizon_profile_path`,
//! M-k): the local horizon altitude as a function of azimuth.
//!
//! **File format.** One `az_deg,alt_deg` pair per line: azimuth from north
//! through east in `[0, 360]`, altitude in `[-90, 90]`. Commas, semicolons,
//! tabs or spaces separate the two numbers, so NINA-style space-separated
//! files load too. Blank lines, `#` comments and a leading UTF-8 byte order
//! mark are skipped. One header line (`az_deg,alt_deg`) is allowed before
//! the first point, and only when it contains no digit and no field reads
//! as a number; any other line that is not two numbers is an error, so a
//! typo on the first line is never mistaken for a header. Points may come
//! in any order. 360 is the same azimuth as 0: a closed loop that lists
//! both (the usual NINA layout) loads when the two altitudes agree, and is
//! an error ([`HorizonError::NorthMismatch`]) when they differ.
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
    /// Azimuths 0 and 360 are the same direction but were given different
    /// altitudes.
    #[error(
        "horizon profile gives azimuth 0° altitude {alt_at_0_deg}° and azimuth 360° \
         altitude {alt_at_360_deg}°; they are the same direction, so make them equal \
         or remove one"
    )]
    NorthMismatch {
        alt_at_0_deg: f64,
        alt_at_360_deg: f64,
    },
}

/// A validated horizon profile, sorted by azimuth in `[0, 360)`.
///
/// It deserializes through [`HorizonProfile::from_points`], so a stored or
/// received profile gets the same validation as a loaded file.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HorizonProfile {
    points: Vec<HorizonPoint>,
}

impl<'de> Deserialize<'de> for HorizonProfile {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            points: Vec<HorizonPoint>,
        }
        let raw = Raw::deserialize(deserializer)?;
        Self::from_points(raw.points).map_err(serde::de::Error::custom)
    }
}

/// Two altitudes listed for azimuths 0 and 360 count as equal within this.
const NORTH_MATCH_DEG: f64 = 1e-9;

impl HorizonProfile {
    /// A flat horizon at `alt_deg`, clamped to `[-90, 90]`. A non-finite
    /// altitude gives a 90° wall, so nothing passes a limit check against
    /// it.
    pub fn flat(alt_deg: f64) -> Self {
        let alt_deg = if alt_deg.is_finite() {
            alt_deg.clamp(-90.0, 90.0)
        } else {
            90.0
        };
        Self {
            points: vec![HorizonPoint {
                az_deg: 0.0,
                alt_deg,
            }],
        }
    }

    /// Validates and sorts the points. A point at azimuth 360 is the same
    /// as one at 0: when both are listed with equal altitudes they merge,
    /// and when the altitudes differ the profile is rejected.
    pub fn from_points(points: Vec<HorizonPoint>) -> Result<Self, HorizonError> {
        if points.is_empty() {
            return Err(HorizonError::Empty);
        }
        if points.len() > MAX_POINTS {
            return Err(HorizonError::TooManyPoints);
        }
        let mut out = Vec::with_capacity(points.len());
        let mut at_360: Option<f64> = None;
        for (index, p) in points.into_iter().enumerate() {
            validate(p).map_err(|message| HorizonError::Parse {
                line: index + 1,
                message,
            })?;
            if p.az_deg == 360.0 {
                if at_360.is_some() {
                    return Err(HorizonError::DuplicateAzimuth { az_deg: 360.0 });
                }
                at_360 = Some(p.alt_deg);
                continue;
            }
            out.push(HorizonPoint {
                // `+ 0.0` turns a `-0` azimuth into `0`.
                az_deg: norm_deg(p.az_deg) + 0.0,
                alt_deg: p.alt_deg,
            });
        }
        out.sort_by(|a, b| a.az_deg.total_cmp(&b.az_deg));
        if let Some(pair) = out.windows(2).find(|w| w[0].az_deg == w[1].az_deg) {
            return Err(HorizonError::DuplicateAzimuth {
                az_deg: pair[0].az_deg,
            });
        }
        if let Some(alt_at_360_deg) = at_360 {
            match out.first() {
                Some(first) if first.az_deg == 0.0 => {
                    if (first.alt_deg - alt_at_360_deg).abs() > NORTH_MATCH_DEG {
                        return Err(HorizonError::NorthMismatch {
                            alt_at_0_deg: first.alt_deg,
                            alt_at_360_deg,
                        });
                    }
                }
                _ => out.insert(
                    0,
                    HorizonPoint {
                        az_deg: 0.0,
                        alt_deg: alt_at_360_deg,
                    },
                ),
            }
        }
        Ok(Self { points: out })
    }

    /// Parses the file format described in the module docs.
    pub fn parse(text: &str) -> Result<Self, HorizonError> {
        let text = text.strip_prefix('\u{feff}').unwrap_or(text);
        let mut points = Vec::new();
        let mut header_allowed = true;
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
                    header_allowed = false;
                }
                // One header line is allowed before the data, and only a
                // line with no number-like field and no digit at all.
                None if header_allowed && is_header(line, &fields) => {
                    header_allowed = false;
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
        if pts.is_empty() {
            // Unreachable through the constructors; a wall if it happens.
            return 90.0;
        }
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

/// A header line: no field reads as a number (`nan` and `inf` included)
/// and no digit appears anywhere, so `0°,40°` or `0,4O` is never a header.
fn is_header(line: &str, fields: &[&str]) -> bool {
    !line.chars().any(|c| c.is_ascii_digit()) && fields.iter().all(|f| f.parse::<f64>().is_err())
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
            Err(HorizonError::NorthMismatch {
                alt_at_0_deg: 10.0,
                alt_at_360_deg: 12.0
            })
        );
        assert_eq!(
            HorizonProfile::parse("90,10\n90,12\n"),
            Err(HorizonError::DuplicateAzimuth { az_deg: 90.0 })
        );
        assert_eq!(
            HorizonProfile::parse("360,10\n360,10\n"),
            Err(HorizonError::DuplicateAzimuth { az_deg: 360.0 })
        );
    }

    #[test]
    fn a_bad_first_line_is_an_error_not_a_header() {
        // Typos, units and stray characters on the first data line of a
        // file without a header must not drop the point.
        for first in ["0,4O", "0°,40°", "0,40°", "O,40", "NaN,NaN", "inf 40", "0"] {
            let text = format!("{first}\n90,10\n180,10\n270,10\n");
            assert!(
                matches!(
                    HorizonProfile::parse(&text),
                    Err(HorizonError::Parse { line: 1, .. })
                ),
                "{first}: {:?}",
                HorizonProfile::parse(&text)
            );
        }
        // Only one header line, and only before the data.
        assert!(matches!(
            HorizonProfile::parse("az,alt\nazimuth altitude\n0,10\n"),
            Err(HorizonError::Parse { line: 2, .. })
        ));
        assert!(matches!(
            HorizonProfile::parse("0,10\naz,alt\n"),
            Err(HorizonError::Parse { line: 2, .. })
        ));
        // Accepted header spellings.
        for header in ["az_deg,alt_deg", "Azimuth Altitude", "az (deg); alt (deg)"] {
            let text = format!("{header}\n0,40\n180,10\n");
            let h = HorizonProfile::parse(&text).unwrap();
            assert_eq!(h.points().len(), 2, "{header}");
            assert_eq!(h.altitude_at(0.0), 40.0, "{header}");
        }
    }

    #[test]
    fn a_byte_order_mark_is_skipped() {
        // Windows Notepad and Excel "CSV UTF-8" start the file with U+FEFF.
        let h = HorizonProfile::parse("\u{feff}0,40\n90,10\n180,10\n270,10\n").unwrap();
        assert_eq!(h.points().len(), 4);
        assert_eq!(h.altitude_at(0.0), 40.0);
        let with_header = HorizonProfile::parse("\u{feff}az_deg,alt_deg\n0,40\n180,10\n").unwrap();
        assert_eq!(with_header.altitude_at(0.0), 40.0);
        // A BOM anywhere else is not whitespace and stays an error.
        assert!(matches!(
            HorizonProfile::parse("0,40\n\u{feff}90,10\n"),
            Err(HorizonError::Parse { line: 2, .. })
        ));
    }

    #[test]
    fn a_closed_nina_loop_loads() {
        // NINA-standard horizon files usually repeat the north point at 360.
        let h = HorizonProfile::parse("0 20\n90 15\n180 10\n270 15\n360 20\n").unwrap();
        let azimuths: Vec<f64> = h.points().iter().map(|p| p.az_deg).collect();
        assert_eq!(azimuths, vec![0.0, 90.0, 180.0, 270.0]);
        assert_eq!(h.altitude_at(0.0), 20.0);
        assert_eq!(h.altitude_at(360.0), 20.0);
        assert!((h.altitude_at(315.0) - 17.5).abs() < 1e-12);
        // 360 listed first, or with a trailing `.0`, is the same loop.
        let reordered = HorizonProfile::parse("360 20.0\n180 10\n0 20\n").unwrap();
        assert_eq!(reordered.points().len(), 2);
        assert_eq!(reordered.altitude_at(0.0), 20.0);
    }

    #[test]
    fn deserializing_validates_like_loading() {
        let h = HorizonProfile::parse(SAMPLE).unwrap();
        let json = serde_json::to_string(&h).unwrap();
        assert_eq!(serde_json::from_str::<HorizonProfile>(&json).unwrap(), h);
        // Unsorted input comes back sorted, a closing 360 merges.
        let unsorted: HorizonProfile = serde_json::from_str(
            r#"{"points":[{"az_deg":180.0,"alt_deg":5.0},{"az_deg":0.0,"alt_deg":10.0},
                {"az_deg":360.0,"alt_deg":10.0}]}"#,
        )
        .unwrap();
        let azimuths: Vec<f64> = unsorted.points().iter().map(|p| p.az_deg).collect();
        assert_eq!(azimuths, vec![0.0, 180.0]);
        for bad in [
            r#"{"points":[]}"#,
            r#"{"points":[{"az_deg":10.0,"alt_deg":95.0}]}"#,
            r#"{"points":[{"az_deg":400.0,"alt_deg":5.0}]}"#,
            r#"{"points":[{"az_deg":10.0,"alt_deg":5.0},{"az_deg":10.0,"alt_deg":6.0}]}"#,
            r#"{"points":[{"az_deg":0.0,"alt_deg":5.0},{"az_deg":360.0,"alt_deg":6.0}]}"#,
            r#"{}"#,
        ] {
            assert!(
                serde_json::from_str::<HorizonProfile>(bad).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn a_non_finite_flat_horizon_is_a_wall() {
        let h = HorizonProfile::flat(f64::NAN);
        assert_eq!(h.altitude_at(123.0), 90.0);
        let zenith_ish = AltAz {
            alt_deg: 89.0,
            az_deg: 0.0,
        };
        assert!(!h.is_clear(zenith_ish, 0.0));
        assert_eq!(HorizonProfile::flat(120.0).altitude_at(0.0), 90.0);
        assert_eq!(
            HorizonProfile::flat(f64::NEG_INFINITY).altitude_at(0.0),
            90.0
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

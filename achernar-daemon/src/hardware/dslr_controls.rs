//! Translation between physical exposure settings and DSLR selection indices.
//!
//! Astronomy cameras expose exposure as a continuous number of milliseconds and
//! gain as a continuous integer. DSLR and mirrorless bodies do not. Over PTP,
//! libgphoto2 presents `shutterspeed` and `iso` as *enumerations* whose labels
//! carry the only physical meaning:
//!
//! ```text
//! shutterspeed: ["bulb", "30", "25", "20", ..., "1/125", ..., "1/4000"]
//! iso:          ["Auto", "100", "200", "400", ..., "25600"]
//! ```
//!
//! Setting `shutterspeed` to `17.0` does not request 17 milliseconds — it
//! selects choice 17, which might be anything. This module turns a requested
//! exposure in seconds into the index of the closest choice the body actually
//! offers, and just as importantly reports back what that choice *is*, so the
//! FITS header records the exposure that was taken rather than the one that
//! was asked for.
//!
//! Verified against the label vocabularies of the Canon EOS R50, M50, 760D
//! (T6s), 600D (T3i) and Panasonic DC-G95 as exposed by libgphoto2 2.5.34.
//! Per the gPhoto project's own guidance, the authoritative list for any given
//! body must still be read off that body — which is exactly why this maps
//! whatever labels the camera reports instead of hardcoding a ladder.

/// Parse a `shutterspeed` choice label into seconds.
///
/// Handles every spelling these bodies use:
/// - fractional: `"1/125"`, `"1/4000"`
/// - whole seconds: `"30"`, `"2"`
/// - quoted seconds (Panasonic style): `"30\""`, `"2.5s"`
/// - decimal: `"0.3"`
///
/// Returns `None` for non-numeric modes (`"bulb"`, `"Auto"`, `"Unknown"`),
/// which cannot be expressed as a fixed duration and must not be silently
/// treated as one.
pub fn parse_shutter_seconds(label: &str) -> Option<f64> {
    // Strip the unit suffixes these bodies append: Panasonic writes 30" and
    // 2.5s, Canon writes bare numbers. Slice-of-char pattern rather than an
    // array literal, which only became a valid pattern in newer Rust.
    let cleaned = label
        .trim()
        .trim_end_matches(&['"', 's', 'S', '\''][..])
        .replace(' ', "");
    if cleaned.is_empty() {
        return None;
    }

    if let Some((num, den)) = cleaned.split_once('/') {
        let num: f64 = num.parse().ok()?;
        let den: f64 = den.parse().ok()?;
        if den == 0.0 {
            return None;
        }
        return finite_positive(num / den);
    }

    finite_positive(cleaned.parse().ok()?)
}

/// Parse an `iso` choice label into an ISO value.
///
/// Accepts `"800"` and `"ISO 800"`. Returns `None` for `"Auto"` and the
/// extended `"Hi"`/`"Lo"` pseudo-stops, which have no defined numeric gain.
pub fn parse_iso(label: &str) -> Option<f64> {
    let cleaned = label
        .trim()
        .trim_start_matches("ISO")
        .trim_start_matches("iso")
        .replace(' ', "");
    if cleaned.is_empty() {
        return None;
    }
    finite_positive(cleaned.parse().ok()?)
}

fn finite_positive(v: f64) -> Option<f64> {
    (v.is_finite() && v > 0.0).then_some(v)
}

/// The choice this camera can actually deliver for a requested value.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedChoice {
    /// Index to pass to `set_parameter` — the value the driver understands.
    pub index: usize,
    /// The label at that index, e.g. `"1/125"` or `"800"`.
    pub label: String,
    /// The label's parsed physical value: seconds for shutter, ISO for gain.
    /// This — not the requested value — is what belongs in the FITS header.
    pub value: f64,
}

/// Pick the choice closest to `target`, comparing in log space.
///
/// Log space is the right metric because both ladders are geometric: between
/// 1/125 s and 1/60 s, a request for 1/100 s is nearly equidistant in *stops*,
/// while a linear comparison would be biased hard toward the longer exposure.
/// The same reasoning applies to the ISO ladder.
///
/// Labels that do not parse (`"bulb"`, `"Auto"`) are skipped rather than
/// treated as zero. Returns `None` when no label in the list is numeric.
pub fn nearest_choice<F>(labels: &[String], parse: F, target: f64) -> Option<ResolvedChoice>
where
    F: Fn(&str) -> Option<f64>,
{
    if !(target.is_finite() && target > 0.0) {
        return None;
    }
    let log_target = target.ln();

    labels
        .iter()
        .enumerate()
        .filter_map(|(index, label)| {
            parse(label).map(|value| ResolvedChoice {
                index,
                label: label.clone(),
                value,
            })
        })
        .min_by(|a, b| {
            let da = (a.value.ln() - log_target).abs();
            let db = (b.value.ln() - log_target).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// Nearest shutter-speed choice to `seconds`.
pub fn nearest_shutter(labels: &[String], seconds: f64) -> Option<ResolvedChoice> {
    nearest_choice(labels, parse_shutter_seconds, seconds)
}

/// Nearest ISO choice to `iso`.
pub fn nearest_iso(labels: &[String], iso: f64) -> Option<ResolvedChoice> {
    nearest_choice(labels, parse_iso, iso)
}

/// How far off the delivered choice is from the request, in stops.
///
/// Callers use this to warn the operator: asking a body whose longest
/// non-bulb speed is 30 s for a 300 s sub yields a frame 3.3 stops short, and
/// that must be visible rather than buried.
pub fn stops_from(requested: f64, delivered: f64) -> f64 {
    if !(requested.is_finite() && requested > 0.0 && delivered.is_finite() && delivered > 0.0) {
        return f64::NAN;
    }
    (delivered / requested).log2()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parses_canon_shutter_labels() {
        assert_eq!(parse_shutter_seconds("1/125"), Some(0.008));
        assert_eq!(parse_shutter_seconds("1/4000"), Some(0.00025));
        assert_eq!(parse_shutter_seconds("30"), Some(30.0));
        assert_eq!(parse_shutter_seconds("0.3"), Some(0.3));
    }

    #[test]
    fn parses_panasonic_quoted_seconds() {
        assert_eq!(parse_shutter_seconds("30\""), Some(30.0));
        assert_eq!(parse_shutter_seconds("2.5s"), Some(2.5));
    }

    #[test]
    fn rejects_non_numeric_shutter_modes() {
        // The important case: "bulb" must never be read as a duration.
        assert_eq!(parse_shutter_seconds("bulb"), None);
        assert_eq!(parse_shutter_seconds("Bulb"), None);
        assert_eq!(parse_shutter_seconds("Auto"), None);
        assert_eq!(parse_shutter_seconds("Unknown value 0000"), None);
        assert_eq!(parse_shutter_seconds(""), None);
        assert_eq!(parse_shutter_seconds("1/0"), None);
    }

    #[test]
    fn parses_iso_labels() {
        assert_eq!(parse_iso("800"), Some(800.0));
        assert_eq!(parse_iso("ISO 800"), Some(800.0));
        assert_eq!(parse_iso("Auto"), None);
        assert_eq!(parse_iso("Hi 1"), None);
    }

    #[test]
    fn picks_nearest_shutter_skipping_bulb() {
        let l = labels(&["bulb", "30", "15", "8", "4", "2", "1", "1/2", "1/4"]);
        let r = nearest_shutter(&l, 3.5).unwrap();
        assert_eq!(r.label, "4");
        assert_eq!(r.index, 4);
        assert_eq!(r.value, 4.0);
    }

    #[test]
    fn nearest_is_geometric_not_linear() {
        // 1/100 s sits between 1/125 (0.008) and 1/60 (0.0167).
        // Linear distance favours 1/60 (0.0067 away vs 0.002)... no: linear
        // favours 1/125. Use a case where the two disagree: request 0.012 s.
        // Linear: |0.012-0.008| = 0.004, |0.0167-0.012| = 0.0047 -> 1/125.
        // Log:    ln ratios 0.405 vs 0.330 -> 1/60. Geometric is correct.
        let l = labels(&["1/60", "1/125"]);
        let r = nearest_shutter(&l, 0.012).unwrap();
        assert_eq!(r.label, "1/60");
    }

    #[test]
    fn clamps_to_longest_available_exposure() {
        // A body topping out at 30 s asked for a 300 s sub must return 30 s,
        // so the caller records 30 s and can warn about the shortfall.
        let l = labels(&["bulb", "30", "15", "8"]);
        let r = nearest_shutter(&l, 300.0).unwrap();
        assert_eq!(r.value, 30.0);
        assert!((stops_from(300.0, r.value) + 3.3219).abs() < 1e-3);
    }

    #[test]
    fn picks_nearest_iso_skipping_auto() {
        let l = labels(&["Auto", "100", "200", "400", "800", "1600"]);
        let r = nearest_iso(&l, 500.0).unwrap();
        // 500 is geometrically closer to 400 (ratio 1.25) than 800 (1.6).
        assert_eq!(r.label, "400");
        assert_eq!(r.index, 3);
    }

    #[test]
    fn returns_none_when_no_label_is_numeric() {
        let l = labels(&["bulb", "Auto", "Unknown"]);
        assert!(nearest_shutter(&l, 1.0).is_none());
        assert!(nearest_iso(&l, 800.0).is_none());
    }

    #[test]
    fn rejects_nonsense_targets() {
        let l = labels(&["1", "2"]);
        assert!(nearest_shutter(&l, 0.0).is_none());
        assert!(nearest_shutter(&l, -1.0).is_none());
        assert!(nearest_shutter(&l, f64::NAN).is_none());
    }

    #[test]
    fn stops_from_is_signed_and_symmetric() {
        assert!((stops_from(1.0, 2.0) - 1.0).abs() < 1e-9);
        assert!((stops_from(2.0, 1.0) + 1.0).abs() < 1e-9);
        assert!(stops_from(1.0, 0.0).is_nan());
    }
}

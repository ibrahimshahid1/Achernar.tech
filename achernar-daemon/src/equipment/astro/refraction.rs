//! Atmospheric refraction and airmass (WP1A-3).
//!
//! - Refraction from an apparent (observed) altitude: Bennett (1982),
//!   accurate to about 0.07′ above the horizon.
//! - Refraction from a true (geometric) altitude: Saemundsson (1986),
//!   consistent with Bennett to about 0.1′.
//! - Both take an optional pressure and temperature, scaled by
//!   `(P / 1010 hPa) · (283 K / (273 K + T))` (Meeus ch. 16). With no
//!   atmosphere given, the standard 1010 hPa and 10 °C apply. A pressure of
//!   zero means no refraction (an airless calculation).
//! - Airmass: Kasten & Young (1989) from the true altitude; about 38 at the
//!   horizon, `None` below it.
//!
//! The formulas are not meant for altitudes below about −1°; inputs below
//! [`MIN_ALTITUDE_DEG`] are clamped there so the result stays finite and
//! monotone.

use serde::{Deserialize, Serialize};

/// Lowest altitude the refraction formulas are evaluated at.
pub const MIN_ALTITUDE_DEG: f64 = -1.0;

/// Pressure and temperature for the refraction scale factor.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Atmosphere {
    pub pressure_hpa: f64,
    pub temperature_c: f64,
}

impl Atmosphere {
    /// 1010 hPa and 10 °C: the conditions both formulas are tabulated for.
    pub const STANDARD: Atmosphere = Atmosphere {
        pressure_hpa: 1010.0,
        temperature_c: 10.0,
    };

    /// No air: every refraction is zero.
    pub const NONE: Atmosphere = Atmosphere {
        pressure_hpa: 0.0,
        temperature_c: 10.0,
    };

    /// The multiplier applied to the standard refraction. Non-finite or
    /// non-physical values fall back to the standard atmosphere.
    pub fn factor(&self) -> f64 {
        let kelvin = 273.0 + self.temperature_c;
        if !(self.pressure_hpa.is_finite() && self.pressure_hpa >= 0.0)
            || !(kelvin.is_finite() && kelvin > 150.0)
        {
            return 1.0;
        }
        (self.pressure_hpa / 1010.0) * (283.0 / kelvin)
    }
}

impl Default for Atmosphere {
    fn default() -> Self {
        Self::STANDARD
    }
}

fn cot_deg(deg: f64) -> f64 {
    1.0 / deg.to_radians().tan()
}

/// Refraction in degrees for an **apparent** altitude (Bennett).
pub fn refraction_from_apparent_deg(apparent_alt_deg: f64, atmosphere: Option<Atmosphere>) -> f64 {
    if !apparent_alt_deg.is_finite() {
        return 0.0;
    }
    let h = apparent_alt_deg.clamp(MIN_ALTITUDE_DEG, 90.0);
    // +0.0013515′ makes R(90°) = 0 (Meeus ch. 16).
    let arcmin = cot_deg(h + 7.31 / (h + 4.4)) + 0.001_351_5;
    let factor = atmosphere.unwrap_or_default().factor();
    (arcmin.max(0.0) / 60.0) * factor
}

/// Refraction in degrees for a **true** (geometric) altitude
/// (Saemundsson).
pub fn refraction_from_true_deg(true_alt_deg: f64, atmosphere: Option<Atmosphere>) -> f64 {
    if !true_alt_deg.is_finite() {
        return 0.0;
    }
    let h = true_alt_deg.clamp(MIN_ALTITUDE_DEG, 90.0);
    // +0.0019279′ makes R(90°) = 0 (Meeus ch. 16).
    let arcmin = 1.02 * cot_deg(h + 10.3 / (h + 5.11)) + 0.001_927_9;
    let factor = atmosphere.unwrap_or_default().factor();
    (arcmin.max(0.0) / 60.0) * factor
}

/// Apparent altitude of an object at a true altitude.
pub fn apparent_from_true_deg(true_alt_deg: f64, atmosphere: Option<Atmosphere>) -> f64 {
    true_alt_deg + refraction_from_true_deg(true_alt_deg, atmosphere)
}

/// True altitude of an object seen at an apparent altitude.
pub fn true_from_apparent_deg(apparent_alt_deg: f64, atmosphere: Option<Atmosphere>) -> f64 {
    apparent_alt_deg - refraction_from_apparent_deg(apparent_alt_deg, atmosphere)
}

/// Relative airmass (Kasten & Young 1989) for a true altitude in degrees;
/// `None` below the horizon or for a non-finite input.
pub fn airmass(true_alt_deg: f64) -> Option<f64> {
    if !true_alt_deg.is_finite() || true_alt_deg < 0.0 {
        return None;
    }
    let z = 90.0 - true_alt_deg.min(90.0);
    Some(1.0 / (z.to_radians().cos() + 0.505_72 * (96.079_95 - z).powf(-1.636_4)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arcmin(deg: f64) -> f64 {
        deg * 60.0
    }

    #[test]
    fn bennett_matches_the_standard_values() {
        // About 34.5′ at the apparent horizon, about 1′ at 45°, 0 at zenith.
        assert!((arcmin(refraction_from_apparent_deg(0.0, None)) - 34.48).abs() < 0.05);
        assert!((arcmin(refraction_from_apparent_deg(45.0, None)) - 0.996).abs() < 0.01);
        assert!(arcmin(refraction_from_apparent_deg(90.0, None)).abs() < 1e-6);
        // Saemundsson: about 29′ at the true horizon, 0 at zenith.
        assert!((arcmin(refraction_from_true_deg(0.0, None)) - 28.99).abs() < 0.05);
        assert!(arcmin(refraction_from_true_deg(90.0, None)).abs() < 1e-6);
    }

    #[test]
    fn bennett_and_saemundsson_agree() {
        // Meeus: the two formulas are consistent to about 0.1′.
        let mut h0 = 0.0;
        while h0 <= 90.0 {
            let h = true_from_apparent_deg(h0, None);
            let back = apparent_from_true_deg(h, None);
            assert!(
                arcmin((back - h0).abs()) < 0.1,
                "{h0}: {}",
                arcmin(back - h0)
            );
            h0 += 0.5;
        }
    }

    #[test]
    fn refraction_is_monotone_and_finite_down_to_the_clamp() {
        let mut previous = f64::INFINITY;
        let mut h = -5.0;
        while h <= 90.0 {
            let r = refraction_from_apparent_deg(h, None);
            assert!(r.is_finite() && r >= 0.0 && r <= previous + 1e-12, "{h}");
            previous = r;
            h += 0.25;
        }
        assert_eq!(refraction_from_apparent_deg(f64::NAN, None), 0.0);
    }

    #[test]
    fn pressure_and_temperature_scale_refraction() {
        let base = refraction_from_apparent_deg(10.0, None);
        let thin = Atmosphere {
            pressure_hpa: 505.0,
            temperature_c: 10.0,
        };
        assert!((refraction_from_apparent_deg(10.0, Some(thin)) - base / 2.0).abs() < 1e-12);
        let cold = Atmosphere {
            pressure_hpa: 1010.0,
            temperature_c: -10.0,
        };
        assert!(refraction_from_true_deg(10.0, Some(cold)) > refraction_from_true_deg(10.0, None));
        assert_eq!(
            refraction_from_apparent_deg(10.0, Some(Atmosphere::NONE)),
            0.0
        );
        let broken = Atmosphere {
            pressure_hpa: f64::NAN,
            temperature_c: 10.0,
        };
        assert_eq!(broken.factor(), 1.0);
    }

    #[test]
    fn kasten_young_airmass() {
        assert!((airmass(90.0).unwrap() - 0.9997).abs() < 1e-3);
        assert!((airmass(30.0).unwrap() - 1.9943).abs() < 1e-3);
        assert!((airmass(0.0).unwrap() - 37.92).abs() < 0.02);
        assert_eq!(airmass(-0.1), None);
        assert_eq!(airmass(f64::NAN), None);
        let mut previous = 0.0;
        for alt in (0..=90).rev() {
            let x = airmass(f64::from(alt)).unwrap();
            assert!(x > previous || alt == 90);
            previous = x;
        }
    }
}

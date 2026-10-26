//! Filter SNR scorer — Section 8.2.2 (FW-1).
//!
//! Classical per-filter signal-to-noise in consistent electron units (v6
//! §8.2.2):
//!
//! ```text
//! SNR(f, t) = S_f t / sqrt(S_f t + n_pix (B_f t + D t + R^2))
//! ```
//!
//! `S_f` is source electrons per second inside the aperture, `B_f` and `D`
//! are the per-pixel background and dark rates (e-/s/px) and `R` is the
//! per-pixel read noise (e-). There are no built-in detector, sky or target
//! constants. Gain, read noise and dark rate come from the RB-C
//! photon-transfer calibration in the rig profile; the per-filter rates come
//! from BGE plus that gain, or from throughput and Moon geometry. Without
//! them a filter is reported `uncalibrated`, never scored with invented
//! numbers. ADU rates enter only through [`AduPerSecond::to_electrons`],
//! which needs a gain, so ADU and electron quantities cannot be mixed.

use serde::{Deserialize, Serialize};

/// A rate in electrons per second.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ElectronsPerSecond(pub f64);

/// A rate in ADU per second; convert with a gain before scoring.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AduPerSecond(pub f64);

/// Detector gain, electrons per ADU.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GainElectronsPerAdu(pub f64);

impl AduPerSecond {
    /// `None` for a non-finite rate or a gain that is not positive.
    pub fn to_electrons(self, gain: GainElectronsPerAdu) -> Option<ElectronsPerSecond> {
        let (rate, gain) = (self.0, gain.0);
        (rate.is_finite() && gain.is_finite() && gain > 0.0)
            .then_some(ElectronsPerSecond(rate * gain))
    }
}

/// The RB-C sensor calibration the score needs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorCalibration {
    pub gain: GainElectronsPerAdu,
    /// Per-pixel read noise, e-.
    pub read_noise_e: f64,
    /// Dark rate at the current sensor temperature, e-/s/px.
    pub dark_rate_e_per_px_s: f64,
    /// Where the numbers came from, e.g. the photon-transfer run id.
    pub provenance: String,
}

impl SensorCalibration {
    fn problem(&self) -> Option<String> {
        if !(self.gain.0.is_finite() && self.gain.0 > 0.0) {
            return Some("gain must be a positive number of e-/ADU".into());
        }
        if !(self.read_noise_e.is_finite() && self.read_noise_e >= 0.0) {
            return Some("read noise must be a non-negative number of electrons".into());
        }
        if !(self.dark_rate_e_per_px_s.is_finite() && self.dark_rate_e_per_px_s >= 0.0) {
            return Some("dark rate must be a non-negative e-/s/px".into());
        }
        None
    }
}

/// The electron rates of one filter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterRates {
    pub filter_name: String,
    /// `S_f`: source electrons per second inside the aperture.
    pub source_rate: Option<ElectronsPerSecond>,
    /// `B_f`: sky background electrons per second per pixel.
    pub background_rate_per_px: Option<ElectronsPerSecond>,
}

/// The exposure and aperture the filters are compared at.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnrSetup {
    pub exposure_s: f64,
    /// `n_pix`: pixels in the photometric aperture.
    pub aperture_pixels: f64,
    pub calibration: Option<SensorCalibration>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ScoreStatus {
    Calibrated,
    Uncalibrated { reason: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterSnrScore {
    pub filter_name: String,
    pub status: ScoreStatus,
    pub estimated_snr: Option<f64>,
    /// `S_f t`, electrons.
    pub signal_e: Option<f64>,
    /// The noise term, electrons.
    pub noise_e: Option<f64>,
}

impl FilterSnrScore {
    fn uncalibrated(filter_name: &str, reason: impl Into<String>) -> Self {
        Self {
            filter_name: filter_name.to_string(),
            status: ScoreStatus::Uncalibrated {
                reason: reason.into(),
            },
            estimated_snr: None,
            signal_e: None,
            noise_e: None,
        }
    }
}

pub struct FilterSnrScorer;

impl FilterSnrScorer {
    /// The aperture SNR, or `None` for inputs that are not physical.
    pub fn snr(
        source: ElectronsPerSecond,
        background_per_px: ElectronsPerSecond,
        dark_rate_e_per_px_s: f64,
        read_noise_e: f64,
        aperture_pixels: f64,
        exposure_s: f64,
    ) -> Option<(f64, f64, f64)> {
        let inputs = [
            source.0,
            background_per_px.0,
            dark_rate_e_per_px_s,
            read_noise_e,
            aperture_pixels,
            exposure_s,
        ];
        if inputs.iter().any(|x| !x.is_finite() || *x < 0.0)
            || aperture_pixels <= 0.0
            || exposure_s <= 0.0
        {
            return None;
        }
        let signal = source.0 * exposure_s;
        let variance = signal
            + aperture_pixels
                * (background_per_px.0 * exposure_s
                    + dark_rate_e_per_px_s * exposure_s
                    + read_noise_e * read_noise_e);
        if variance <= 0.0 {
            return None;
        }
        let noise = variance.sqrt();
        Some((signal / noise, signal, noise))
    }

    /// Score each filter. Calibrated scores come first, by SNR; filters that
    /// cannot be scored follow with the reason, in input order.
    pub fn score_filters(setup: &SnrSetup, filters: &[FilterRates]) -> Vec<FilterSnrScore> {
        let calibration = match &setup.calibration {
            None => {
                return filters
                    .iter()
                    .map(|f| {
                        FilterSnrScore::uncalibrated(
                            &f.filter_name,
                            "no RB-C sensor calibration (gain, read noise, dark rate)",
                        )
                    })
                    .collect()
            }
            Some(calibration) => calibration,
        };
        if let Some(problem) = calibration.problem() {
            return filters
                .iter()
                .map(|f| FilterSnrScore::uncalibrated(&f.filter_name, problem.clone()))
                .collect();
        }
        let mut scored: Vec<FilterSnrScore> = Vec::new();
        let mut unscored: Vec<FilterSnrScore> = Vec::new();
        for filter in filters {
            let (Some(source), Some(background)) =
                (filter.source_rate, filter.background_rate_per_px)
            else {
                unscored.push(FilterSnrScore::uncalibrated(
                    &filter.filter_name,
                    "no electron-unit source or background rate for this filter",
                ));
                continue;
            };
            match Self::snr(
                source,
                background,
                calibration.dark_rate_e_per_px_s,
                calibration.read_noise_e,
                setup.aperture_pixels,
                setup.exposure_s,
            ) {
                Some((snr, signal, noise)) => scored.push(FilterSnrScore {
                    filter_name: filter.filter_name.clone(),
                    status: ScoreStatus::Calibrated,
                    estimated_snr: Some(snr),
                    signal_e: Some(signal),
                    noise_e: Some(noise),
                }),
                None => unscored.push(FilterSnrScore::uncalibrated(
                    &filter.filter_name,
                    "the rates, exposure or aperture are not physical",
                )),
            }
        }
        scored.sort_by(|a, b| {
            b.estimated_snr
                .unwrap_or(f64::NEG_INFINITY)
                .total_cmp(&a.estimated_snr.unwrap_or(f64::NEG_INFINITY))
        });
        scored.extend(unscored);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibration(read_noise_e: f64) -> SensorCalibration {
        SensorCalibration {
            gain: GainElectronsPerAdu(0.8),
            read_noise_e,
            dark_rate_e_per_px_s: 0.1,
            provenance: "photon-transfer test".into(),
        }
    }

    fn rates(name: &str, source: f64, background: f64) -> FilterRates {
        FilterRates {
            filter_name: name.into(),
            source_rate: Some(ElectronsPerSecond(source)),
            background_rate_per_px: Some(ElectronsPerSecond(background)),
        }
    }

    #[test]
    fn the_aperture_formula_uses_electrons() {
        // S t = 1000; n_pix (B t + D t + R^2) = 20 (20 + 1 + 9) = 600.
        let (snr, signal, noise) = FilterSnrScorer::snr(
            ElectronsPerSecond(100.0),
            ElectronsPerSecond(2.0),
            0.1,
            3.0,
            20.0,
            10.0,
        )
        .unwrap();
        assert_eq!(signal, 1000.0);
        assert!((noise - 40.0).abs() < 1e-9);
        assert!((snr - 25.0).abs() < 1e-9);
        assert!(FilterSnrScorer::snr(
            ElectronsPerSecond(-1.0),
            ElectronsPerSecond(2.0),
            0.1,
            3.0,
            20.0,
            10.0
        )
        .is_none());
        assert!(FilterSnrScorer::snr(
            ElectronsPerSecond(1.0),
            ElectronsPerSecond(2.0),
            0.1,
            3.0,
            20.0,
            0.0
        )
        .is_none());
    }

    #[test]
    fn without_rb_c_calibration_every_filter_is_uncalibrated() {
        let setup = SnrSetup {
            exposure_s: 60.0,
            aperture_pixels: 25.0,
            calibration: None,
        };
        let scores = FilterSnrScorer::score_filters(&setup, &[rates("Ha", 5.0, 0.5)]);
        assert!(matches!(scores[0].status, ScoreStatus::Uncalibrated { .. }));
        assert_eq!(scores[0].estimated_snr, None);

        let bad = SnrSetup {
            calibration: Some(SensorCalibration {
                gain: GainElectronsPerAdu(0.0),
                ..calibration(3.0)
            }),
            ..setup
        };
        let scores = FilterSnrScorer::score_filters(&bad, &[rates("Ha", 5.0, 0.5)]);
        assert!(
            matches!(&scores[0].status, ScoreStatus::Uncalibrated { reason } if reason.contains("gain"))
        );
    }

    #[test]
    fn scores_rank_calibrated_filters_and_report_missing_rates() {
        let setup = SnrSetup {
            exposure_s: 120.0,
            aperture_pixels: 30.0,
            calibration: Some(calibration(3.0)),
        };
        let filters = [
            rates("L", 40.0, 25.0),
            rates("Ha", 6.0, 0.4),
            FilterRates {
                filter_name: "OIII".into(),
                source_rate: Some(ElectronsPerSecond(4.0)),
                background_rate_per_px: None,
            },
        ];
        let scores = FilterSnrScorer::score_filters(&setup, &filters);
        assert_eq!(scores.len(), 3);
        assert!(scores[0].estimated_snr >= scores[1].estimated_snr);
        assert_eq!(scores[2].filter_name, "OIII");
        assert!(matches!(scores[2].status, ScoreStatus::Uncalibrated { .. }));
        let serialised = serde_json::to_value(&scores[2]).unwrap();
        assert_eq!(serialised["status"]["status"], "uncalibrated");
    }

    #[test]
    fn the_detector_numbers_come_from_the_calibration_not_constants() {
        let filters = [rates("Ha", 2.0, 0.3)];
        let low = SnrSetup {
            exposure_s: 300.0,
            aperture_pixels: 30.0,
            calibration: Some(calibration(1.0)),
        };
        let high = SnrSetup {
            calibration: Some(calibration(12.0)),
            ..low.clone()
        };
        let a = FilterSnrScorer::score_filters(&low, &filters)[0]
            .estimated_snr
            .unwrap();
        let b = FilterSnrScorer::score_filters(&high, &filters)[0]
            .estimated_snr
            .unwrap();
        assert!(a > b, "more read noise must lower the SNR: {a} vs {b}");
    }

    #[test]
    fn adu_rates_need_a_gain_to_become_electrons() {
        let rate = AduPerSecond(50.0);
        assert_eq!(
            rate.to_electrons(GainElectronsPerAdu(1.5)),
            Some(ElectronsPerSecond(75.0))
        );
        assert_eq!(rate.to_electrons(GainElectronsPerAdu(0.0)), None);
        assert_eq!(
            AduPerSecond(f64::NAN).to_electrons(GainElectronsPerAdu(1.0)),
            None
        );
    }
}

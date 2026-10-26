//! Guide RMS accumulators (spec §3.5 G-i; WP1A-7).
//!
//! Per axis: the population standard deviation (Welford's online algorithm,
//! public-domain mathematics, §6; the same definition NINA displays), the
//! **RMS about zero**, the peak absolute error and the step count. The axes
//! are PHD2's mount axes (`RADistanceRaw`, `DECDistanceRaw`), in guide-camera
//! pixels; camera-axis `dx`/`dy` are never substituted for them. Arcsec values
//! use the pixel scale that was current for each step, and are reported only
//! when every counted step had one.
//!
//! Dither and settle steps are excluded by default ([`PhaseFilter`]); a
//! sliding 50-step window ([`SlidingRms`]) feeds telemetry.

use std::collections::VecDeque;

use crate::equipment::snapshot::{GuidePhase, GuideRms, GuideStepSample};

/// Telemetry window length (G-i).
pub const SLIDING_STEPS: usize = 50;

/// One axis of an RMS accumulator.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AxisStats {
    n: u64,
    mean: f64,
    m2: f64,
    sum_sq: f64,
    peak: f64,
}

impl AxisStats {
    pub fn push(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        self.m2 += delta * (x - self.mean);
        self.sum_sq += x * x;
        self.peak = self.peak.max(x.abs());
    }

    pub fn count(&self) -> u64 {
        self.n
    }

    pub fn mean(&self) -> Option<f64> {
        (self.n > 0).then_some(self.mean)
    }

    /// Population standard deviation (divide by n).
    pub fn std_dev(&self) -> Option<f64> {
        (self.n > 0).then(|| (self.m2 / self.n as f64).max(0.0).sqrt())
    }

    /// RMS about zero: `sqrt(sum(x^2) / n)`.
    pub fn rms0(&self) -> Option<f64> {
        (self.n > 0).then(|| (self.sum_sq / self.n as f64).sqrt())
    }

    /// Largest absolute value.
    pub fn peak(&self) -> Option<f64> {
        (self.n > 0).then_some(self.peak)
    }
}

/// RA and Dec accumulators, in pixels and (when known) arcsec.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RmsAccumulator {
    ra: AxisStats,
    dec: AxisStats,
    ra_arcsec: AxisStats,
    dec_arcsec: AxisStats,
    missing_scale: bool,
}

impl RmsAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one pair of mount-axis errors in guide pixels.
    pub fn push(&mut self, ra_px: f64, dec_px: f64, scale_arcsec: Option<f64>) {
        if !(ra_px.is_finite() && dec_px.is_finite()) {
            return;
        }
        self.ra.push(ra_px);
        self.dec.push(dec_px);
        match scale_arcsec.filter(|s| s.is_finite() && *s > 0.0) {
            Some(scale) => {
                self.ra_arcsec.push(ra_px * scale);
                self.dec_arcsec.push(dec_px * scale);
            }
            None => self.missing_scale = true,
        }
    }

    /// Add a guide step. Returns false (and counts nothing) when the step
    /// lacks a mount-axis error.
    pub fn push_step(&mut self, step: &GuideStepSample) -> bool {
        match (step.ra_raw, step.dec_raw) {
            (Some(ra), Some(dec)) if ra.is_finite() && dec.is_finite() => {
                self.push(ra, dec, step.pixel_scale_arcsec);
                true
            }
            _ => false,
        }
    }

    pub fn count(&self) -> u64 {
        self.ra.count()
    }

    /// The statistics, or `None` when no step was counted (never a 0 RMS).
    pub fn result(&self) -> Option<GuideRms> {
        let ra = self.ra.std_dev()?;
        let dec = self.dec.std_dev()?;
        let rms0_ra = self.ra.rms0()?;
        let rms0_dec = self.dec.rms0()?;
        let arcsec = !self.missing_scale && self.ra_arcsec.count() == self.ra.count();
        let ra_as = arcsec.then(|| self.ra_arcsec.std_dev()).flatten();
        let dec_as = arcsec.then(|| self.dec_arcsec.std_dev()).flatten();
        let rms0_ra_as = arcsec.then(|| self.ra_arcsec.rms0()).flatten();
        let rms0_dec_as = arcsec.then(|| self.dec_arcsec.rms0()).flatten();
        Some(GuideRms {
            ra_px: ra,
            dec_px: dec,
            total_px: ra.hypot(dec),
            rms0_ra_px: rms0_ra,
            rms0_dec_px: rms0_dec,
            rms0_total_px: rms0_ra.hypot(rms0_dec),
            peak_ra_px: self.ra.peak().unwrap_or_default(),
            peak_dec_px: self.dec.peak().unwrap_or_default(),
            ra_arcsec: ra_as,
            dec_arcsec: dec_as,
            total_arcsec: ra_as.zip(dec_as).map(|(a, b)| a.hypot(b)),
            rms0_total_arcsec: rms0_ra_as.zip(rms0_dec_as).map(|(a, b)| a.hypot(b)),
            steps_n: u32::try_from(self.ra.count()).unwrap_or(u32::MAX),
        })
    }
}

/// Which step phases count towards an RMS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseFilter {
    pub guiding: bool,
    pub dithering: bool,
    pub settling: bool,
    pub paused: bool,
    pub recovering: bool,
}

impl Default for PhaseFilter {
    /// G-i: dither and settle steps are excluded by default.
    fn default() -> Self {
        Self {
            guiding: true,
            dithering: false,
            settling: false,
            paused: true,
            recovering: true,
        }
    }
}

impl PhaseFilter {
    pub fn all() -> Self {
        Self {
            guiding: true,
            dithering: true,
            settling: true,
            paused: true,
            recovering: true,
        }
    }

    pub fn includes(&self, phase: GuidePhase) -> bool {
        match phase {
            GuidePhase::Guiding => self.guiding,
            GuidePhase::Dithering => self.dithering,
            GuidePhase::Settling => self.settling,
            GuidePhase::Paused => self.paused,
            GuidePhase::Recovering => self.recovering,
        }
    }
}

/// RMS over the steps the filter includes.
pub fn window_rms<'a>(
    steps: impl IntoIterator<Item = &'a GuideStepSample>,
    filter: &PhaseFilter,
) -> Option<GuideRms> {
    let mut acc = RmsAccumulator::new();
    for step in steps {
        if filter.includes(step.phase) {
            acc.push_step(step);
        }
    }
    acc.result()
}

/// The last [`SLIDING_STEPS`] counted steps, for telemetry.
#[derive(Debug, Clone)]
pub struct SlidingRms {
    capacity: usize,
    steps: VecDeque<(f64, f64, Option<f64>)>,
}

impl Default for SlidingRms {
    fn default() -> Self {
        Self::new(SLIDING_STEPS)
    }
}

impl SlidingRms {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            steps: VecDeque::with_capacity(capacity.max(1)),
        }
    }

    /// Add a step if it has mount-axis errors. Returns whether it counted.
    pub fn push_step(&mut self, step: &GuideStepSample) -> bool {
        match (step.ra_raw, step.dec_raw) {
            (Some(ra), Some(dec)) if ra.is_finite() && dec.is_finite() => {
                if self.steps.len() == self.capacity {
                    self.steps.pop_front();
                }
                self.steps.push_back((ra, dec, step.pixel_scale_arcsec));
                true
            }
            _ => false,
        }
    }

    pub fn clear(&mut self) {
        self.steps.clear();
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    pub fn rms(&self) -> Option<GuideRms> {
        let mut acc = RmsAccumulator::new();
        for &(ra, dec, scale) in &self.steps {
            acc.push(ra, dec, scale);
        }
        acc.result()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    /// A plain two-pass reference: population std, RMS about zero, peak.
    fn reference(values: &[f64]) -> (f64, f64, f64) {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let rms0 = (values.iter().map(|x| x * x).sum::<f64>() / n).sqrt();
        let peak = values.iter().fold(0.0_f64, |p, x| p.max(x.abs()));
        (var.sqrt(), rms0, peak)
    }

    fn step(
        ra: Option<f64>,
        dec: Option<f64>,
        phase: GuidePhase,
        scale: Option<f64>,
    ) -> GuideStepSample {
        GuideStepSample {
            ra_raw: ra,
            dec_raw: dec,
            phase,
            pixel_scale_arcsec: scale,
            ..GuideStepSample::default()
        }
    }

    #[test]
    fn welford_matches_a_two_pass_reference() {
        let ra = [0.31, -0.42, 0.05, 0.6, -0.11, 0.27, -0.33, 0.9, -0.75, 0.02];
        let dec = [-0.1, 0.15, 0.2, -0.05, 0.0, 0.31, -0.22, 0.12, 0.08, -0.4];
        let mut acc = RmsAccumulator::new();
        for (r, d) in ra.iter().zip(dec.iter()) {
            acc.push(*r, *d, Some(1.5));
        }
        let rms = acc.result().unwrap();
        let (ra_std, ra_rms0, ra_peak) = reference(&ra);
        let (dec_std, dec_rms0, dec_peak) = reference(&dec);
        assert!(close(rms.ra_px, ra_std));
        assert!(close(rms.dec_px, dec_std));
        assert!(close(rms.total_px, ra_std.hypot(dec_std)));
        assert!(close(rms.rms0_ra_px, ra_rms0));
        assert!(close(rms.rms0_dec_px, dec_rms0));
        assert!(close(rms.rms0_total_px, ra_rms0.hypot(dec_rms0)));
        assert!(close(rms.peak_ra_px, ra_peak));
        assert!(close(rms.peak_dec_px, dec_peak));
        assert!(close(rms.ra_arcsec.unwrap(), ra_std * 1.5));
        assert!(close(
            rms.total_arcsec.unwrap(),
            ra_std.hypot(dec_std) * 1.5
        ));
        assert!(close(
            rms.rms0_total_arcsec.unwrap(),
            ra_rms0.hypot(dec_rms0) * 1.5
        ));
        assert_eq!(rms.steps_n, 10);
        // RMS about zero differs from the standard deviation when the mean
        // is offset.
        let mut offset = RmsAccumulator::new();
        for _ in 0..4 {
            offset.push(1.0, 1.0, None);
        }
        let r = offset.result().unwrap();
        assert!(close(r.ra_px, 0.0));
        assert!(close(r.rms0_ra_px, 1.0));
        assert_eq!(r.ra_arcsec, None, "no pixel scale, no arcsec");
    }

    #[test]
    fn no_steps_is_unknown_not_zero() {
        assert_eq!(RmsAccumulator::new().result(), None);
        assert_eq!(SlidingRms::default().rms(), None);
        assert_eq!(AxisStats::default().std_dev(), None);
    }

    #[test]
    fn steps_without_mount_axis_errors_are_not_counted() {
        let mut acc = RmsAccumulator::new();
        assert!(!acc.push_step(&step(None, Some(0.1), GuidePhase::Guiding, None)));
        assert!(acc.push_step(&step(Some(0.2), Some(0.1), GuidePhase::Guiding, None)));
        assert_eq!(acc.count(), 1);
    }

    #[test]
    fn arcsec_needs_a_scale_on_every_counted_step() {
        let mut acc = RmsAccumulator::new();
        acc.push(0.1, 0.1, Some(2.0));
        acc.push(0.2, 0.2, None);
        let r = acc.result().unwrap();
        assert!(r.total_arcsec.is_none());
        assert!(r.total_px > 0.0);
    }

    #[test]
    fn the_default_filter_drops_dither_and_settle_steps() {
        let steps = vec![
            step(Some(0.1), Some(0.1), GuidePhase::Guiding, Some(1.0)),
            step(Some(5.0), Some(5.0), GuidePhase::Dithering, Some(1.0)),
            step(Some(1.0), Some(1.0), GuidePhase::Settling, Some(1.0)),
            step(Some(-0.1), Some(-0.1), GuidePhase::Guiding, Some(1.0)),
            step(Some(0.3), Some(0.0), GuidePhase::Recovering, Some(1.0)),
        ];
        let default = window_rms(&steps, &PhaseFilter::default()).unwrap();
        assert_eq!(default.steps_n, 3);
        assert!(default.peak_ra_px < 1.0);
        let all = window_rms(&steps, &PhaseFilter::all()).unwrap();
        assert_eq!(all.steps_n, 5);
        assert!(close(all.peak_ra_px, 5.0));
    }

    #[test]
    fn the_sliding_window_keeps_the_last_fifty_steps() {
        let mut sliding = SlidingRms::default();
        for i in 0..80 {
            let v = if i < 30 {
                10.0
            } else {
                0.5 * if i % 2 == 0 { 1.0 } else { -1.0 }
            };
            sliding.push_step(&step(Some(v), Some(0.0), GuidePhase::Guiding, Some(1.2)));
        }
        assert_eq!(sliding.len(), SLIDING_STEPS);
        let r = sliding.rms().unwrap();
        assert_eq!(r.steps_n, 50);
        assert!(close(r.peak_ra_px, 0.5), "the 10 px steps aged out");
        assert!(close(r.ra_px, 0.5));
        assert!(close(r.ra_arcsec.unwrap(), 0.6));
    }
}

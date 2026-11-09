//! TED runner — Section 8.1 (v6 §7.6.5; equipment spec H15, §7.2.5).
//!
//! The trailing/elongation evidence for one frame:
//!
//! - **Pixels:** the classical measurement (`trailing.rs`): severity, the
//!   sensor position angle and sensor axis, always; the sky axis only when a
//!   plate solve's WCS still describes the frame.
//! - **Equipment, bracketed by the exposure window:** tracking and rate,
//!   slewing, pulse guiding and the guide pulses, the guider state,
//!   declination, pier side and alignment mode (field rotation), rotator
//!   angle, focuser motion and wind. These come from the hub's device window
//!   (`EquipmentReader::window`), never from a read after the frame.
//! - **The RB-E untracked label** (15.041·cos δ ″/s) only with a verified
//!   `tracking = false` across the whole window: tracking read off in every
//!   motion interval, and the intervals covering the exposure from start to
//!   end. An exposure that tracked for part of its length gets no label.
//!   The declination must be known too.
//!
//! Missing values stay missing. There is no learned TED model yet; the
//! classical measurement is the evidence, and the runner refuses to dress a
//! bare image up as a sky-axis result.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::trailing::{SensorAxis, SkyAxis, TrailingEstimate};
use crate::equipment::mount::SIDEREAL_RATE_ARCSEC_S;
use crate::equipment::snapshot::{DeviceWindow, GuiderAppState};
use crate::equipment::types::{AlignmentMode, DriveRate, PierSide};

/// TED evidence for one frame (H15).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct TedEvidence {
    /// `classical`, or the learned model's path once one exists.
    pub model: String,
    pub severity: f32,
    pub trailing_detected: bool,
    pub elongation_ratio: f64,
    pub sensor_pa_deg: Option<f32>,
    pub sensor_axis: Option<SensorAxis>,
    pub sky_axis: Option<SkyAxis>,
    pub sky_pa_deg: Option<f32>,
    pub solve_event_id: Option<String>,
    /// Why the sky axis is withheld.
    pub sky_withheld: Option<String>,
    pub window_start: Option<DateTime<Utc>>,
    pub window_end: Option<DateTime<Utc>>,
    /// True only if tracking was on throughout; false if it was off at any
    /// point; `None` if unknown anywhere in the window.
    pub tracking: Option<bool>,
    pub tracking_rate: Option<DriveRate>,
    pub slewing_any: Option<bool>,
    pub pulse_guiding_any: Option<bool>,
    pub guide_pulses_n: Option<u32>,
    pub guiding_state: Option<GuiderAppState>,
    pub dec_deg: Option<f64>,
    pub pier_side: Option<PierSide>,
    pub alignment_mode: Option<AlignmentMode>,
    pub rotator_angle_deg: Option<f64>,
    pub focuser_moved: Option<bool>,
    pub wind_speed_ms: Option<f64>,
    /// RB-E: the untracked drift rate, only with tracking verified off over
    /// the whole window (not merely off at some point).
    pub untracked_rate_arcsec_s: Option<f64>,
}

pub struct TedRunner {
    model_path: Option<String>,
    elongation_threshold: f64,
}

impl TedRunner {
    pub fn new(model_path: Option<String>) -> Self {
        Self {
            model_path,
            elongation_threshold: 1.3,
        }
    }

    fn detected(&self, estimate: &TrailingEstimate) -> bool {
        estimate.severity >= self.elongation_threshold_to_severity()
    }

    fn elongation_threshold_to_severity(&self) -> f32 {
        (1.0 - 1.0 / self.elongation_threshold.max(1.0)) as f32
    }

    /// The H15 evidence: the measurement, plus the equipment state over the
    /// exposure `window` when one is known.
    pub fn evidence(
        &self,
        estimate: &TrailingEstimate,
        window: Option<&DeviceWindow>,
    ) -> TedEvidence {
        let mut evidence = TedEvidence {
            model: self
                .model_path
                .clone()
                .unwrap_or_else(|| "classical".to_string()),
            severity: estimate.severity,
            trailing_detected: self.detected(estimate),
            elongation_ratio: elongation_ratio(estimate.severity),
            sensor_pa_deg: estimate.sensor_pa_deg,
            sensor_axis: estimate.sensor_axis,
            sky_axis: estimate.sky.as_ref().map(|s| s.axis),
            sky_pa_deg: estimate.sky.as_ref().map(|s| s.sky_pa_deg),
            solve_event_id: estimate.sky.as_ref().and_then(|s| s.solve_event_id.clone()),
            sky_withheld: estimate.withheld.clone(),
            ..TedEvidence::default()
        };
        let Some(window) = window else {
            return evidence;
        };
        evidence.window_start = Some(window.start);
        evidence.window_end = Some(window.end);
        let all_known = |pick: fn(&crate::equipment::snapshot::MotionInterval) -> Option<bool>| {
            if window.motion.is_empty() {
                return None;
            }
            window
                .motion
                .iter()
                .map(pick)
                .collect::<Option<Vec<bool>>>()
        };
        evidence.tracking = all_known(|m| m.tracking).map(|v| v.iter().all(|on| *on));
        if evidence.tracking.is_none() && window.motion.iter().any(|m| m.tracking == Some(false)) {
            evidence.tracking = Some(false);
        }
        // RB-E needs more than "off at some point": off in every interval,
        // and intervals that cover the exposure.
        let tracking_off_throughout = motion_covers_window(window)
            && all_known(|m| m.tracking).is_some_and(|v| v.iter().all(|on| !on));
        evidence.slewing_any = all_known(|m| m.slewing)
            .map(|v| v.iter().any(|on| *on))
            .or_else(|| {
                window
                    .motion
                    .iter()
                    .any(|m| m.slewing == Some(true))
                    .then_some(true)
            });
        evidence.pulse_guiding_any = all_known(|m| m.pulse_guiding)
            .map(|v| v.iter().any(|on| *on))
            .or_else(|| {
                window
                    .motion
                    .iter()
                    .any(|m| m.pulse_guiding == Some(true))
                    .then_some(true)
            });
        evidence.guide_pulses_n = Some(window.guide.pulses_n);
        evidence.guiding_state = window.guide.state;
        let mount = window
            .mount
            .mid
            .as_ref()
            .or(window.mount.start.as_ref())
            .or(window.mount.end.as_ref());
        if let Some(mount) = mount {
            evidence.tracking_rate = mount.state.tracking_rate.known().copied();
            evidence.dec_deg = window
                .mount_mid
                .map(|c| c.dec_deg)
                .or_else(|| mount.state.dec_deg.known().copied());
            evidence.pier_side = mount
                .state
                .side_of_pier
                .known()
                .copied()
                .filter(|side| *side != PierSide::Unknown);
            evidence.alignment_mode = mount.capabilities.alignment_mode;
        }
        evidence.rotator_angle_deg = window
            .rotator
            .mid
            .as_ref()
            .and_then(|r| r.state.position_deg.known().copied());
        evidence.focuser_moved = (window.focuser.samples_n > 0).then_some(window.moved.focuser);
        evidence.wind_speed_ms = window
            .weather
            .as_ref()
            .and_then(|w| w.wind_speed_ms.known().copied());
        if tracking_off_throughout {
            evidence.untracked_rate_arcsec_s = evidence
                .dec_deg
                .map(|dec| SIDEREAL_RATE_ARCSEC_S * dec.to_radians().cos());
        }
        evidence
    }
}

/// The window's motion intervals run without a gap from its start to its
/// end. The hub starts them at the last mount sample before the exposure;
/// with none, the first interval begins inside the window and the start of
/// the exposure is unobserved.
fn motion_covers_window(window: &DeviceWindow) -> bool {
    let (Some(first), Some(last)) = (window.motion.first(), window.motion.last()) else {
        return false;
    };
    first.from <= window.start
        && last.to >= window.end
        && window
            .motion
            .windows(2)
            .all(|pair| pair[1].from <= pair[0].to)
}

fn elongation_ratio(severity: f32) -> f64 {
    (1.0 / (1.0 - severity.max(0.0) as f64)).clamp(1.0, 99.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::snapshot::{
        GuideWindowStats, MotionInterval, MountCapabilities, MountSnapshot, MountState,
        WindowSamples,
    };
    use crate::equipment::types::{EqCoord, EqFrame, Observed, Source};

    fn t(s: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_792_029_600 + s, 0).unwrap()
    }

    fn trailed() -> TrailingEstimate {
        TrailingEstimate {
            severity: 0.6,
            sensor_pa_deg: Some(3.0),
            sensor_axis: Some(SensorAxis::X),
            sky: None,
            withheld: Some("no_plate_solve".into()),
            policy: crate::vision::trailing::LEGACY_TRAILING_POLICY,
            unavailable_reason: None,
        }
    }

    fn window(tracking: [Option<bool>; 2]) -> DeviceWindow {
        let fresh = |v| Observed::fresh(v, Source::DeviceState, t(5), 0);
        DeviceWindow {
            start: t(0),
            end: t(10),
            mount: WindowSamples {
                mid: Some(MountSnapshot {
                    state: MountState {
                        dec_deg: fresh(60.0),
                        tracking_rate: Observed::fresh(
                            DriveRate::Sidereal,
                            Source::Property,
                            t(5),
                            0,
                        ),
                        side_of_pier: Observed::fresh(PierSide::East, Source::DeviceState, t(5), 0),
                        ..MountState::default()
                    },
                    capabilities: MountCapabilities {
                        alignment_mode: Some(AlignmentMode::Polar),
                        ..MountCapabilities::default()
                    },
                    ..MountSnapshot::default()
                }),
                samples_n: 1,
                ..WindowSamples::default()
            },
            mount_mid: Some(EqCoord::new(5.0, 60.0, EqFrame::JNowApparent)),
            motion: vec![
                MotionInterval {
                    from: t(0),
                    to: t(5),
                    tracking: tracking[0],
                    slewing: Some(false),
                    pulse_guiding: Some(false),
                },
                MotionInterval {
                    from: t(5),
                    to: t(10),
                    tracking: tracking[1],
                    slewing: Some(false),
                    pulse_guiding: Some(true),
                },
            ],
            guide: GuideWindowStats {
                state: Some(GuiderAppState::Guiding),
                pulses_n: 4,
                ..GuideWindowStats::default()
            },
            ..DeviceWindow::default()
        }
    }

    #[test]
    fn evidence_brackets_the_exposure_with_the_window() {
        let evidence =
            TedRunner::new(None).evidence(&trailed(), Some(&window([Some(true), Some(true)])));
        assert_eq!(evidence.model, "classical");
        assert!(evidence.trailing_detected);
        assert_eq!(evidence.tracking, Some(true));
        assert_eq!(evidence.slewing_any, Some(false));
        assert_eq!(evidence.pulse_guiding_any, Some(true));
        assert_eq!(evidence.guide_pulses_n, Some(4));
        assert_eq!(evidence.guiding_state, Some(GuiderAppState::Guiding));
        assert_eq!(evidence.tracking_rate, Some(DriveRate::Sidereal));
        assert_eq!(evidence.dec_deg, Some(60.0));
        assert_eq!(evidence.pier_side, Some(PierSide::East));
        assert_eq!(evidence.alignment_mode, Some(AlignmentMode::Polar));
        assert_eq!(evidence.sky_axis, None);
        assert_eq!(evidence.sky_withheld.as_deref(), Some("no_plate_solve"));
        assert_eq!(evidence.untracked_rate_arcsec_s, None, "tracking was on");
    }

    #[test]
    fn the_untracked_label_needs_tracking_verified_off() {
        let runner = TedRunner::new(None);
        let off = runner.evidence(&trailed(), Some(&window([Some(false), Some(false)])));
        assert_eq!(off.tracking, Some(false));
        let rate = off.untracked_rate_arcsec_s.unwrap();
        assert!((rate - SIDEREAL_RATE_ARCSEC_S * 0.5).abs() < 1e-6);

        // Off for the first half only: tracking was off at some point, but
        // the exposure did not drift at the full sidereal rate.
        let half = runner.evidence(&trailed(), Some(&window([Some(false), Some(true)])));
        assert_eq!(half.tracking, Some(false));
        assert_eq!(half.untracked_rate_arcsec_s, None);

        // Off where known, unknown elsewhere: not verified.
        let partly = runner.evidence(&trailed(), Some(&window([Some(false), None])));
        assert_eq!(partly.tracking, Some(false));
        assert_eq!(partly.untracked_rate_arcsec_s, None);

        // Off throughout the samples, but no sample covers the start of the
        // exposure.
        let mut late = window([Some(false), Some(false)]);
        late.motion[0].from = t(2);
        let late = runner.evidence(&trailed(), Some(&late));
        assert_eq!(late.tracking, Some(false));
        assert_eq!(late.untracked_rate_arcsec_s, None);

        let unknown = runner.evidence(&trailed(), Some(&window([None, Some(true)])));
        assert_eq!(unknown.tracking, None);
        assert_eq!(unknown.untracked_rate_arcsec_s, None);

        let no_window = TedRunner::new(None).evidence(&trailed(), None);
        assert_eq!((no_window.tracking, no_window.window_start), (None, None));
    }
}

//! Vision loop — Section 8.1.
//! Runs at 1-5 Hz and publishes SharedObservation to all subsystems.

use std::time::Duration;

use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};

use crate::hardware::camera::FrameData;
use crate::state::AppState;

use super::shared_observation::{CloudEvent, SharedObservation};
use super::{background, quality, trailing};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CloudStatus {
    Clear,
    Cloudy,
}

#[derive(Debug, Clone)]
struct VisionTrendState {
    clear_star_baseline: Option<f32>,
    clear_background_baseline: Option<f32>,
    cloud_status: CloudStatus,
    consecutive_cloudy: u8,
    consecutive_clear: u8,
}

impl Default for VisionTrendState {
    fn default() -> Self {
        Self {
            clear_star_baseline: None,
            clear_background_baseline: None,
            cloud_status: CloudStatus::Clear,
            consecutive_cloudy: 0,
            consecutive_clear: 0,
        }
    }
}

pub fn spawn(state: AppState, rate_hz: f64) -> JoinHandle<()> {
    tokio::spawn(async move {
        run(state, rate_hz).await;
    })
}

pub async fn run(state: AppState, rate_hz: f64) {
    let period = period_for_rate(rate_hz);
    let mut ticker = interval(period);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    tracing::info!(
        "Vision loop started at {:.2} Hz",
        1.0 / period.as_secs_f64()
    );
    let mut trends = VisionTrendState::default();

    loop {
        ticker.tick().await;

        let frame = state.latest_frame.lock().unwrap().clone();
        let Some(frame) = frame else {
            continue;
        };

        let previous = state
            .vision_observation
            .read()
            .ok()
            .and_then(|bus| bus.clone());

        match observation_from_frame_with_trends(&frame, &mut trends, previous.as_ref()) {
            Ok(observation) => {
                if let Ok(mut bus) = state.vision_observation.write() {
                    *bus = Some(observation.clone());
                }
                let _ = state.vision_updates.send(observation);
            }
            Err(err) => {
                tracing::debug!("Vision loop skipped frame: {}", err);
            }
        }
    }
}

pub fn observation_from_frame(frame: &FrameData) -> crate::error::Result<SharedObservation> {
    observation_from_frame_with_trends(frame, &mut VisionTrendState::default(), None)
}

fn observation_from_frame_with_trends(
    frame: &FrameData,
    trends: &mut VisionTrendState,
    previous: Option<&SharedObservation>,
) -> crate::error::Result<SharedObservation> {
    let luma = quality::luma_from_frame(frame)?;
    let background = background::estimate_background(&luma.pixels, luma.max_adu);
    let quality = quality::estimate_quality(&luma, &background);
    let trailing = trailing::estimate_trailing(&luma, &background);
    let cloud_probability = update_cloud_probability(
        trends,
        quality.star_count,
        background.background_adu,
        quality.observation_valid,
    );
    let cloud_event = update_cloud_event(trends, cloud_probability);

    Ok(SharedObservation {
        timestamp: chrono::Utc::now(),
        focus_quality: quality.focus_quality,
        predicted_hfr: quality.hfr,
        sharpness_score: quality.sharpness_score,
        quality_uncertainty: quality.quality_uncertainty,
        hfr: quality.hfr,
        fwhm: quality.fwhm,
        seeing_estimate_arcsec: quality.seeing_estimate_arcsec,
        star_count: quality.star_count,
        cloud_probability,
        cloud_event,
        trailing_severity: trailing.severity,
        trailing_axis: trailing.axis,
        trailing_direction_pa: trailing.direction_pa,
        background_adu: background.background_adu,
        sky_background_adu: background.background_adu,
        background_sigma: background.background_sigma,
        snr_proxy_median: background.snr_proxy_median,
        saturation_fraction: background.saturation_fraction,
        pointing_error_arcmin: previous.and_then(|obs| obs.pointing_error_arcmin),
        wcs_residual_arcsec: previous.and_then(|obs| obs.wcs_residual_arcsec),
        last_solve_confidence: previous.and_then(|obs| obs.last_solve_confidence),
        last_solve_at: previous.and_then(|obs| obs.last_solve_at.clone()),
        frame_quality_score: quality.frame_quality_score,
        observation_valid: quality.observation_valid,
    })
}

fn update_cloud_probability(
    trends: &mut VisionTrendState,
    star_count: u32,
    background_adu: f32,
    observation_valid: bool,
) -> f32 {
    let stars = star_count as f32;
    if trends.clear_star_baseline.is_none() && star_count > 0 {
        trends.clear_star_baseline = Some(stars.max(1.0));
    }
    if trends.clear_background_baseline.is_none() && background_adu.is_finite() {
        trends.clear_background_baseline = Some(background_adu.max(1.0));
    }

    let star_baseline = trends
        .clear_star_baseline
        .unwrap_or(stars.max(1.0))
        .max(1.0);
    let background_baseline = trends
        .clear_background_baseline
        .unwrap_or(background_adu.max(1.0))
        .max(1.0);

    let star_drop = (1.0 - stars / star_baseline).clamp(0.0, 1.0);
    let background_rise =
        ((background_adu - background_baseline) / background_baseline).clamp(0.0, 1.0);
    let invalid_penalty = if observation_valid { 0.0 } else { 0.15 };
    let probability = (0.70 * star_drop + 0.25 * background_rise + invalid_penalty).clamp(0.0, 1.0);

    if probability < 0.40 && observation_valid && star_count > 0 {
        trends.clear_star_baseline = Some(exp_smooth(star_baseline, stars, 0.05).max(1.0));
        trends.clear_background_baseline = Some(exp_smooth(
            background_baseline,
            background_adu.max(1.0),
            0.05,
        ));
    }

    probability
}

fn update_cloud_event(trends: &mut VisionTrendState, probability: f32) -> CloudEvent {
    const CLOUD_THRESHOLD: f32 = 0.80;
    const CLEAR_THRESHOLD: f32 = 0.30;
    const CONSEC_REQUIRED: u8 = 3;

    match trends.cloud_status {
        CloudStatus::Clear => {
            if probability > CLOUD_THRESHOLD {
                trends.consecutive_cloudy = trends.consecutive_cloudy.saturating_add(1);
                if trends.consecutive_cloudy >= CONSEC_REQUIRED {
                    trends.cloud_status = CloudStatus::Cloudy;
                    trends.consecutive_clear = 0;
                    return CloudEvent::CloudDetected;
                }
            } else {
                trends.consecutive_cloudy = 0;
            }
        }
        CloudStatus::Cloudy => {
            if probability < CLEAR_THRESHOLD {
                trends.consecutive_clear = trends.consecutive_clear.saturating_add(1);
                if trends.consecutive_clear >= CONSEC_REQUIRED {
                    trends.cloud_status = CloudStatus::Clear;
                    trends.consecutive_cloudy = 0;
                    return CloudEvent::ClearedUp;
                }
            } else {
                trends.consecutive_clear = 0;
            }
        }
    }

    CloudEvent::None
}

fn exp_smooth(old: f32, new: f32, alpha: f32) -> f32 {
    old * (1.0 - alpha) + new * alpha
}

fn period_for_rate(rate_hz: f64) -> Duration {
    let rate_hz = rate_hz.clamp(1.0, 5.0);
    Duration::from_secs_f64(1.0 / rate_hz)
}

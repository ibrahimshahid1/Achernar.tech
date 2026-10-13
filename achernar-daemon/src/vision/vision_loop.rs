//! Vision loop — Section 8.1.
//! Runs at 1-5 Hz and publishes SharedObservation to all subsystems.

use std::sync::Arc;
use std::time::Duration;

use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};

use crate::hardware::camera::FrameData;
use crate::state::AppState;

use super::fqe_runner::{aggregate_fqe, FqeRunner};
use super::observed_frame::ObservedFrame;
use super::shared_observation::{CloudEvent, SharedObservation};
use super::{preprocess, quality, trailing};

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
    let mut trend_generation = state.current_camera_generation();

    loop {
        ticker.tick().await;

        let generation = state.current_camera_generation();
        if generation != trend_generation {
            trends = VisionTrendState::default();
            trend_generation = generation;
        }
        let frame = state.latest_frame.lock().unwrap().clone();
        let Some(frame) = frame else {
            continue;
        };
        if frame.stream_generation != generation {
            continue;
        }

        let previous = state
            .vision_observation
            .read()
            .ok()
            .and_then(|bus| bus.clone());

        let observation = match observation_from_frame_with_trends(
            &frame,
            &mut trends,
            previous.as_ref(),
            Some(state.fqe_runner.as_ref()),
        ) {
            Ok(observation) => observation,
            Err(error) => {
                // A preview/unknown domain must replace the old measurement;
                // leaving the previous bus entry would falsely label new pixels.
                trends = VisionTrendState::default();
                SharedObservation::analysis_unavailable(error.to_string())
            }
        };
        let _publication = state.camera_observation_barrier.lock().unwrap();
        if frame.stream_generation != generation || state.current_camera_generation() != generation
        {
            trends = VisionTrendState::default();
            trend_generation = state.current_camera_generation();
            continue;
        }
        if let Ok(mut bus) = state.vision_observation.write() {
            *bus = Some(observation.clone());
        }
        if let Ok(mut slot) = state.last_observed_frame.write() {
            *slot = Some(ObservedFrame {
                frame_counter: frame.frame_counter,
                capture_unix_time: frame.unix_timestamp,
                frame: Arc::new(frame),
                observation: observation.clone(),
            });
        }
        let _ = state.vision_updates.send(observation);
    }
}

pub fn observation_from_frame(frame: &FrameData) -> crate::error::Result<SharedObservation> {
    observation_from_frame_with_trends(frame, &mut VisionTrendState::default(), None, None)
}

fn observation_from_frame_with_trends(
    frame: &FrameData,
    trends: &mut VisionTrendState,
    previous: Option<&SharedObservation>,
    fqe: Option<&FqeRunner>,
) -> crate::error::Result<SharedObservation> {
    let luma = match quality::luma_from_frame(frame) {
        Ok(luma) => luma,
        Err(error) => {
            *trends = VisionTrendState::default();
            return Ok(SharedObservation::analysis_unavailable(error.to_string()));
        }
    };
    let previous = previous.filter(|observation| {
        observation
            .analysis_descriptor
            .as_ref()
            .zip(luma.analysis_descriptor.as_ref())
            .map(|(old, current)| old.config_sha256 == current.config_sha256)
            .unwrap_or(false)
    });
    if previous.is_none() {
        *trends = VisionTrendState::default();
    }
    let background = quality::background_from_luma(&luma);
    let quality = quality::estimate_quality(&luma, &background);
    let trailing = trailing::estimate_trailing(&luma, &background);

    // FQE shadow inference on the gate-surviving classical stars that also admit
    // a centered 64 px crop. Published alongside the classical fields; any failure logs
    // and degrades to None — it can never affect the classical measurement.
    let mut fqe_hfr_px = None;
    let mut fqe_sigma_log_hfr = None;
    let mut fqe_star_count = None;
    let mut fqe_model_version = None;
    let mut fqe_calibrated = None;
    let fqe_unavailable_reason = preprocess::fqe_unavailable_reason(&luma).map(str::to_string);
    if let Some(runner) = fqe.filter(|r| r.is_loaded() && fqe_unavailable_reason.is_none()) {
        if !quality.used_stars.is_empty() {
            let crops: Vec<ndarray::Array2<f32>> = quality
                .used_stars
                .iter()
                .filter_map(|star| preprocess::fqe_input_from_frame(&luma, star.x, star.y))
                .collect();
            if !crops.is_empty() {
                match runner.predict_batch(&crops) {
                    Ok(outputs) => {
                        if let Some((hfr, sigma)) = aggregate_fqe(&outputs) {
                            fqe_hfr_px = Some(hfr as f32);
                            fqe_sigma_log_hfr = Some((sigma * runner.frame_sigma_scale()) as f32);
                            fqe_star_count = Some(crops.len() as u32);
                            fqe_model_version = runner.model_version().map(str::to_string);
                            fqe_calibrated = Some(runner.is_calibrated());
                            if let Some(classical) = quality.hfr_px {
                                tracing::debug!(
                                    "FQE shadow: hfr={hfr:.2}px sigma_log={sigma:.3} \
                                     classical={classical:.2}px delta={:+.2}px ({} stars)",
                                    hfr as f32 - classical,
                                    crops.len()
                                );
                            }
                        }
                    }
                    Err(err) => {
                        tracing::debug!("FQE shadow inference skipped: {err}");
                    }
                }
            }
        }
    }
    let cloud_probability = update_cloud_probability(
        trends,
        quality.star_count_detected,
        background.background_adu,
        quality.observation_valid(),
    );
    let cloud_event = update_cloud_event(trends, cloud_probability);

    let observation_valid = quality.observation_valid();
    Ok(SharedObservation {
        timestamp: chrono::Utc::now(),
        analysis_descriptor: luma.analysis_descriptor.clone(),
        analysis_unavailable_reason: None,
        fqe_unavailable_reason,
        status: quality.status,
        rejection_reasons: quality.rejection_reasons,
        hfr_px: quality.hfr_px,
        hfd_px: quality.hfd_px,
        fwhm_px: quality.fwhm_px,
        // Requires a pixel scale and a fitted FWHM — neither is available in the
        // classical path, so it stays null rather than reporting `2*HFR`.
        image_fwhm_arcsec: None,
        quality_uncertainty: quality.quality_uncertainty,
        hfr_log_sigma: quality.hfr_log_sigma,
        star_count: quality.star_count_detected,
        star_count_used: quality.star_count_used,
        star_rejections: quality.star_rejections,
        fqe_hfr_px,
        fqe_sigma_log_hfr,
        fqe_star_count,
        fqe_model_version,
        fqe_calibrated,
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
        saturation_gate: quality.saturation_gate,
        pointing_error_arcmin: previous.and_then(|obs| obs.pointing_error_arcmin),
        wcs_residual_arcsec: previous.and_then(|obs| obs.wcs_residual_arcsec),
        last_solve_confidence: previous.and_then(|obs| obs.last_solve_confidence),
        last_solve_at: previous.and_then(|obs| obs.last_solve_at.clone()),
        observation_valid,
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

#[cfg(test)]
mod analysis_domain_tests {
    use super::*;
    use crate::hardware::{
        camera::SourceImage,
        ols_ffi::{OlsBayerType, OlsStreamType},
    };
    use crate::processing::debayer::*;

    fn mono_frame() -> FrameData {
        FrameData {
            stream_generation: 7,
            format: OlsStreamType::Mono8,
            bayer: OlsBayerType::Na,
            frame_counter: 4,
            unix_timestamp: 1.0,
            width: 64,
            height: 64,
            data: vec![100; 4096],
            source: None,
            saturation_adu: None,
            sensor_descriptor: Some(SensorDescriptor {
                schema_version: 1,
                width: 64,
                height: 64,
                row_stride_bytes: 64,
                sample_type: SampleType::U8,
                byte_order: ByteOrder::Little,
                color: SensorColor::Mono,
                domain: SampleDomain::NativeAdu,
                geometry: PixelGeometry::identity("test_buffer_pixels"),
                saturation: None,
                calibration_ids: vec![],
            }),
        }
    }

    #[test]
    fn observation_identifies_transform_and_disables_unqualified_shadow() {
        let observation = observation_from_frame(&mono_frame()).unwrap();
        assert!(observation.analysis_descriptor.is_some());
        assert!(observation.analysis_unavailable_reason.is_none());
        assert!(observation
            .fqe_unavailable_reason
            .as_ref()
            .unwrap()
            .contains("A03"));
        assert!(observation.fqe_hfr_px.is_none());
        assert!(!observation.analysis_descriptor.unwrap().training_eligible);
        assert_eq!(
            observation.saturation_gate,
            quality::SaturationGate::NotPerformed
        );
    }

    #[test]
    fn previews_unknown_cfa_and_legacy_decodes_publish_current_unavailability() {
        let mut frame = mono_frame();
        for format in [
            OlsStreamType::Mjpeg,
            OlsStreamType::Rgb24,
            OlsStreamType::Yuv2,
            OlsStreamType::Raw8,
        ] {
            frame.format = format;
            let observation = observation_from_frame(&frame).unwrap();
            assert_eq!(
                observation.status,
                quality::QualityStatus::AnalysisUnavailable
            );
            assert!(!observation.observation_valid);
            assert!(observation.hfr_px.is_none());
            assert!(observation.analysis_unavailable_reason.is_some());
        }
        frame.format = OlsStreamType::Mono8;
        frame.source = Some(SourceImage {
            ext: "cr3".into(),
            bytes: vec![1],
        });
        assert!(observation_from_frame(&frame)
            .unwrap()
            .analysis_unavailable_reason
            .unwrap()
            .contains("A04"));
        frame.source = None;
        frame.sensor_descriptor.as_mut().unwrap().width = 63;
        assert!(observation_from_frame(&frame)
            .unwrap()
            .analysis_unavailable_reason
            .unwrap()
            .contains("identity"));
    }

    #[test]
    fn unavailable_new_frame_replaces_old_width_and_resets_trends() {
        let mut old = SharedObservation::empty();
        old.hfr_px = Some(2.0);
        old.observation_valid = true;
        let mut trends = VisionTrendState::default();
        trends.clear_star_baseline = Some(30.0);
        let mut frame = mono_frame();
        frame.sensor_descriptor = None;
        let observation =
            observation_from_frame_with_trends(&frame, &mut trends, Some(&old), None).unwrap();
        assert!(observation.hfr_px.is_none());
        assert!(!observation.observation_valid);
        assert!(trends.clear_star_baseline.is_none());
        assert!(observation.analysis_unavailable_reason.is_some());
    }
}

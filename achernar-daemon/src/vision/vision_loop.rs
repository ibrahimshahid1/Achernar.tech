//! Vision loop — Section 8.1.
//! Runs at 1-5 Hz and publishes SharedObservation to all subsystems.
//!
//! Each observation also carries what the equipment layer knows about the
//! frame (equipment spec WP1B-4):
//!
//! - **H2, preview-frame validity.** The loop subscribes to the hub's event
//!   bus. A segmentation event (`slew_*`, `filter_*`, `focuser_move_*`,
//!   `exposure_settings_changed`, `stream_restarted`) restarts the cloud
//!   heuristic's clear-sky baselines, and a frame that moved never adapts
//!   them. Each frame's validity comes from `EquipmentReader::quiescent`
//!   over a conservative window: stream frames carry only their host arrival
//!   time, so the window runs from two frame intervals before the arrival
//!   to the arrival (`clock_quality = host_arrival`).
//! - **H14, COC context** from `EquipmentReader::coc_context()`.
//! - **H15, TED evidence:** the trail on the sensor always; on the sky only
//!   with the newest solve's WCS, of the same camera, taken under the mount
//!   connection now in use within the session, with the pier side and
//!   rotator unchanged, and on a mount that is not equatorial only while the
//!   field cannot have rotated (a recent solve, no repointing since); plus
//!   the window's tracking, pulses and wind.
//! - **Plate-solve state** (`last_solve_at`, `pointing_error_arcmin`) is read
//!   from `plate_solve_events`, never carried forward.
//! - H2, H14 and the plate-solve state do not depend on the pixels: a frame
//!   whose pixels cannot be analysed still carries them.
//! - **M-3 Recovery:** a cloud-cleared event asks for a recovery solve, and
//!   the loop subscribes the plate-solve trigger to the bus at startup.

use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use tokio::sync::broadcast::{self, error::TryRecvError};
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};

use crate::equipment::events::{EquipmentEvent, EventKind};
use crate::equipment::hub::EquipmentHub;
use crate::equipment::snapshot::{CocContext, DeviceWindow, EquipmentSnapshot, Quiescence};
use crate::equipment::traits::EquipmentReader;
use crate::equipment::types::{AlignmentMode, ClockQuality, DeviceKind, PierSide};
use crate::hardware::camera::FrameData;
use crate::state::AppState;
use crate::storage::plate_solve_repo::LatestSolve;

use super::fqe_runner::{aggregate_fqe, FqeRunner};
use super::observed_frame::ObservedFrame;
use super::shared_observation::{CloudEvent, EquipmentValidity, SharedObservation};
use super::ted_runner::TedRunner;
use super::{preprocess, quality, trailing};

/// Motion devices whose quiescence a frame needs (H2). The camera is OLS's;
/// switches, weather and safety monitors never move the optics.
const MOTION_KINDS: [DeviceKind; 7] = [
    DeviceKind::Mount,
    DeviceKind::Focuser,
    DeviceKind::FilterWheel,
    DeviceKind::Guider,
    DeviceKind::Rotator,
    DeviceKind::CoverCalibrator,
    DeviceKind::Dome,
];

/// Longest frame interval a window is built from.
const MAX_FRAME_INTERVAL: chrono::Duration = chrono::Duration::minutes(10);

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
    /// Camera frame counter last used to adapt the references and advance the
    /// event counters. The loop re-analyses the latest frame on every tick, so
    /// a slow camera would otherwise count one exposure several times.
    last_frame_counter: Option<i32>,
}

impl Default for VisionTrendState {
    fn default() -> Self {
        Self {
            clear_star_baseline: None,
            clear_background_baseline: None,
            cloud_status: CloudStatus::Clear,
            consecutive_cloudy: 0,
            consecutive_clear: 0,
            last_frame_counter: None,
        }
    }
}

impl VisionTrendState {
    /// Restart the baselines after a segmentation event, without counting
    /// the current frame again.
    fn segment(&mut self) {
        *self = Self {
            last_frame_counter: self.last_frame_counter,
            ..Self::default()
        };
    }
}

/// Whether an equipment event restarts the cloud baselines (H2).
pub fn segments_baselines(name: &str) -> bool {
    name.starts_with("slew_")
        || name.starts_with("filter_")
        || name.starts_with("focuser_move_")
        || name == "exposure_settings_changed"
        || name == "stream_restarted"
}

/// Drain the bus: the name of the first segmenting event since the last
/// tick. A lagged subscription may have missed one, so it segments too.
fn drain_segmentation(events: &mut broadcast::Receiver<Arc<EquipmentEvent>>) -> Option<String> {
    let mut first = None;
    loop {
        match events.try_recv() {
            Ok(event) => {
                if first.is_none() && segments_baselines(event.name()) {
                    first = Some(event.name().to_string());
                }
            }
            Err(TryRecvError::Lagged(_)) => {
                if first.is_none() {
                    first = Some("bus_lagged".to_string());
                }
            }
            Err(TryRecvError::Empty | TryRecvError::Closed) => return first,
        }
    }
}

/// Segmentation count and cause, published with every observation.
#[derive(Debug, Clone, Default)]
struct Segments {
    count: u64,
    last: Option<String>,
}

/// Host arrival times of stream frames, as the loop first sees them.
#[derive(Debug, Clone, Default)]
struct FrameTiming {
    frame: Option<(u64, i32)>,
    seen_at: Option<DateTime<Utc>>,
    interval: Option<chrono::Duration>,
}

impl FrameTiming {
    fn observe(&mut self, generation: u64, counter: i32, now: DateTime<Utc>) {
        if self.frame == Some((generation, counter)) {
            return;
        }
        if self.frame.map(|(g, _)| g) != Some(generation) {
            self.seen_at = None;
        }
        self.interval = self.seen_at.map(|previous| now - previous);
        self.seen_at = Some(now);
        self.frame = Some((generation, counter));
    }

    /// From two frame intervals before the arrival to the arrival; `None`
    /// until an interval is known.
    fn window(&self) -> Option<(DateTime<Utc>, DateTime<Utc>)> {
        let end = self.seen_at?;
        let interval = self.interval?.min(MAX_FRAME_INTERVAL);
        Some((end - interval * 2, end))
    }
}

fn attached(snapshot: &EquipmentSnapshot, kind: DeviceKind) -> bool {
    match kind {
        DeviceKind::Mount => snapshot.mount.is_some(),
        DeviceKind::Focuser => snapshot.focuser.is_some(),
        DeviceKind::FilterWheel => snapshot.filter_wheel.is_some(),
        DeviceKind::Guider => snapshot.guider.is_some(),
        DeviceKind::Rotator => snapshot.rotator.is_some(),
        DeviceKind::CoverCalibrator => snapshot.cover_calibrator.is_some(),
        DeviceKind::Dome => snapshot.dome.is_some(),
        _ => false,
    }
}

/// H2 for one frame over `window`.
fn equipment_validity(
    hub: &EquipmentHub,
    snapshot: &EquipmentSnapshot,
    window: Option<(DateTime<Utc>, DateTime<Utc>)>,
    segments: &Segments,
) -> EquipmentValidity {
    let now_mono = hub.clock().now_mono_ns();
    let age = |published: u64| now_mono.checked_sub(published).map(|ns| ns / 1_000_000);
    let kinds: Vec<DeviceKind> = MOTION_KINDS
        .iter()
        .copied()
        .filter(|kind| attached(snapshot, *kind))
        .collect();
    let mut validity = EquipmentValidity {
        quiescent: None,
        reason: None,
        settled_since: None,
        moved: None,
        moved_event_seq: None,
        window_start: window.map(|(start, _)| start),
        window_end: window.map(|(_, end)| end),
        clock_quality: ClockQuality::HostArrival,
        kinds_checked: kinds.clone(),
        mount_age_ms: snapshot
            .mount
            .as_ref()
            .and_then(|m| age(m.published_mono_ns)),
        focuser_age_ms: snapshot
            .focuser
            .as_ref()
            .and_then(|f| age(f.published_mono_ns)),
        filter_wheel_age_ms: snapshot
            .filter_wheel
            .as_ref()
            .and_then(|w| age(w.published_mono_ns)),
        guider_age_ms: snapshot
            .guider
            .as_ref()
            .and_then(|g| age(g.published_mono_ns)),
        mount_slewing: snapshot
            .mount
            .as_ref()
            .and_then(|m| m.state.slewing.known().copied()),
        mount_tracking: snapshot
            .mount
            .as_ref()
            .and_then(|m| m.state.tracking.known().copied()),
        pulse_guiding: snapshot
            .mount
            .as_ref()
            .and_then(|m| m.state.is_pulse_guiding.known().copied()),
        guider_state: snapshot
            .guider
            .as_ref()
            .and_then(|g| g.app_state.known().copied()),
        resume_block_active: !snapshot.resume_blocks.is_empty(),
        segment: segments.count,
        segment_reset_by: segments.last.clone(),
    };
    if kinds.is_empty() {
        validity.reason = Some("no_motion_devices".into());
        return validity;
    }
    let Some((start, end)) = window else {
        validity.reason = Some("frame_interval_unknown".into());
        return validity;
    };
    match hub.quiescent(&kinds, start, end) {
        Quiescence::Quiescent { settled_since } => {
            validity.quiescent = Some(true);
            validity.settled_since = Some(settled_since);
        }
        Quiescence::Moved { kind, event_seq } => {
            validity.quiescent = Some(false);
            validity.reason = Some(format!(
                "moved:{}",
                crate::equipment::types::kind_name(kind)
            ));
            validity.moved = Some(kind);
            validity.moved_event_seq = (event_seq > 0).then_some(event_seq);
        }
        Quiescence::Unknown { reason } => {
            validity.reason = Some(reason);
        }
    }
    validity
}

/// An event after which the field of a mount that is not equatorial sits
/// at another angle: a slew, a jog, a flip, a park or a home. Guide pulses
/// and dithers move it too little to matter; a center loop's own slews are
/// `slew_*` events, and its `centered` follows its last solve.
fn repoints_mount(event: &EquipmentEvent) -> bool {
    event.name().starts_with("slew_")
        || matches!(
            event.event,
            EventKind::MoveAxis { .. }
                | EventKind::MeridianFlip { .. }
                | EventKind::ArrivedOffTarget { .. }
                | EventKind::Parked { .. }
                | EventKind::Unparked
                | EventKind::Homed
        )
}

/// Whether `solve` was taken under the mount connection now in use: its
/// pointing sample is of this mount, and was exposed inside this
/// connection's segment. `None` when either side is unknown.
fn solve_on_connection(
    solve: &LatestSolve,
    mount_identity_key: Option<&str>,
    segment_start: Option<DateTime<Utc>>,
    mount_connected: bool,
) -> Option<bool> {
    if !mount_connected {
        return None;
    }
    let sample_mid = solve.sample_mid_exposure_utc?;
    if let (Some(solved), Some(now)) = (solve.mount_identity_key.as_deref(), mount_identity_key) {
        if solved != now {
            return Some(false);
        }
    }
    // No segment: nothing has been sampled under this connection yet, so
    // the solve's sample is an earlier connection's.
    Some(segment_start.is_some_and(|start| sample_mid >= start))
}

/// What the equipment layer and the archive say about one frame.
#[derive(Debug, Clone, Default)]
pub struct ObservationContext {
    pub equipment: Option<EquipmentValidity>,
    pub coc: Option<CocContext>,
    /// The newest successful plate solve.
    pub solve: Option<LatestSolve>,
    /// The streaming camera's name.
    pub camera: Option<String>,
    pub pier_side: Option<PierSide>,
    pub rotator_angle_deg: Option<f64>,
    /// The exposure window's device state, for TED.
    pub window: Option<DeviceWindow>,
    /// When the context was gathered.
    pub at: Option<DateTime<Utc>>,
    pub alignment_mode: Option<AlignmentMode>,
    /// The solve belongs to the mount connection now in use.
    pub solve_on_this_connection: Option<bool>,
    /// The mount repointed after the solved exposure ([`repoints_mount`]).
    pub mount_repointed_since_solve: Option<bool>,
}

impl ObservationContext {
    fn gather(
        state: &AppState,
        window: Option<(DateTime<Utc>, DateTime<Utc>)>,
        segments: &Segments,
        solve: Option<LatestSolve>,
    ) -> Self {
        let hub = &state.equipment;
        let snapshot = hub.snapshot();
        let equipment = equipment_validity(hub, &snapshot, window, segments);
        let any_device = MOTION_KINDS.iter().any(|kind| attached(&snapshot, *kind));
        let device_window = window
            .filter(|_| any_device)
            .map(|(start, end)| hub.window(start, end));
        let camera = state
            .camera_runtime
            .read()
            .ok()
            .and_then(|runtime| runtime.camera_name.clone());
        let at = Utc::now();
        let alignment_mode = snapshot
            .mount
            .as_ref()
            .and_then(|m| m.capabilities.alignment_mode);
        let mount_identity_key = crate::api::plate_solve::mount_handle(state)
            .map(|handle| handle.identity().identity_key.clone());
        let segment_start = snapshot
            .mount
            .as_ref()
            .and_then(|m| crate::mount::pointing_model::connection_segment_start(m.connection_id));
        let solve_on_this_connection = solve.as_ref().and_then(|solve| {
            solve_on_connection(
                solve,
                mount_identity_key.as_deref(),
                segment_start,
                snapshot.mount.is_some(),
            )
        });
        // Only a mount that is not equatorial needs the journal: there a
        // repointing turns the field.
        let equatorial = matches!(
            alignment_mode,
            Some(AlignmentMode::Polar | AlignmentMode::GermanPolar)
        );
        let mount_repointed_since_solve = solve
            .as_ref()
            .and_then(|solve| solve.exposure_started_at.or(solve.solved_at))
            .filter(|_| !equatorial)
            .filter(|from| at - *from <= trailing::FIELD_ROTATION_BUDGET)
            .map(|from| {
                hub.bus()
                    .events_between(from, at)
                    .iter()
                    .any(|event| repoints_mount(event))
            });
        Self {
            equipment: Some(equipment),
            coc: Some(hub.coc_context()),
            solve,
            camera,
            pier_side: snapshot
                .mount
                .as_ref()
                .and_then(|m| m.state.side_of_pier.known().copied()),
            rotator_angle_deg: snapshot
                .rotator
                .as_ref()
                .and_then(|r| r.state.position_deg.known().copied()),
            window: device_window,
            at: Some(at),
            alignment_mode,
            solve_on_this_connection,
            mount_repointed_since_solve,
        }
    }

    /// The parts of an observation that do not depend on the pixels: H2
    /// validity, the H14 COC context and the plate-solve state. Applied to
    /// every observation of the tick, measured or not.
    fn apply(&self, mut observation: SharedObservation) -> SharedObservation {
        let solve = self.solve.as_ref();
        // From plate_solve_events, never carried forward from a previous
        // observation.
        observation.pointing_error_arcmin = solve
            .and_then(|s| s.pointing_error_arcmin)
            .map(|e| e as f32);
        observation.last_solve_at = solve.and_then(|s| s.solved_at);
        observation.last_solve_event_id = solve.map(|s| s.event_id.clone());
        observation.equipment = self.equipment.clone();
        observation.coc_context = self.coc.clone();
        observation
    }

    /// A frame that moved never adapts the cloud baselines.
    fn moved(&self) -> bool {
        self.equipment
            .as_ref()
            .is_some_and(|e| e.quiescent == Some(false))
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
    // M-3: the plate-solve trigger follows the bus from startup.
    crate::mount::auto_solve::attach(&state);
    let mut events = state.equipment.subscribe();
    let mut segments = Segments::default();
    let mut timing = FrameTiming::default();
    let mut trends = VisionTrendState::default();
    let mut trend_generation = state.current_camera_generation();

    loop {
        ticker.tick().await;

        if let Some(cause) = drain_segmentation(&mut events) {
            trends.segment();
            segments.count += 1;
            segments.last = Some(cause);
        }
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
        timing.observe(frame.stream_generation, frame.frame_counter, Utc::now());

        let previous = state
            .vision_observation
            .read()
            .ok()
            .and_then(|bus| bus.clone());
        let solve = match crate::storage::plate_solve_repo::latest_successful_solve(
            state.local_store.pool(),
        )
        .await
        {
            Ok(solve) => solve,
            Err(error) => {
                tracing::debug!("Vision loop could not read the latest plate solve: {error}");
                None
            }
        };
        let context = ObservationContext::gather(&state, timing.window(), &segments, solve);

        let observation = match observation_with_context(
            &frame,
            &mut trends,
            previous.as_ref(),
            Some(state.fqe_runner.as_ref()),
            &context,
        ) {
            Ok(observation) => observation,
            Err(error) => {
                // A preview/unknown domain must replace the old measurement;
                // leaving the previous bus entry would falsely label new pixels.
                trends = VisionTrendState::default();
                context.apply(SharedObservation::analysis_unavailable(error.to_string()))
            }
        };
        let cleared = observation.cloud_event == CloudEvent::ClearedUp;
        {
            let _publication = state.camera_observation_barrier.lock().unwrap();
            if frame.stream_generation != generation
                || state.current_camera_generation() != generation
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
        if cleared {
            crate::mount::auto_solve::cloud_cleared(&state);
        }
    }
}

pub fn observation_from_frame(frame: &FrameData) -> crate::error::Result<SharedObservation> {
    observation_with_context(
        frame,
        &mut VisionTrendState::default(),
        None,
        None,
        &ObservationContext::default(),
    )
}

#[cfg(test)]
fn observation_from_frame_with_trends(
    frame: &FrameData,
    trends: &mut VisionTrendState,
    previous: Option<&SharedObservation>,
    fqe: Option<&FqeRunner>,
) -> crate::error::Result<SharedObservation> {
    observation_with_context(frame, trends, previous, fqe, &ObservationContext::default())
}

fn observation_with_context(
    frame: &FrameData,
    trends: &mut VisionTrendState,
    previous: Option<&SharedObservation>,
    fqe: Option<&FqeRunner>,
    context: &ObservationContext,
) -> crate::error::Result<SharedObservation> {
    let luma = match quality::luma_from_frame(frame) {
        Ok(luma) => luma,
        Err(error) => {
            *trends = VisionTrendState::default();
            return Ok(context.apply(SharedObservation::analysis_unavailable(error.to_string())));
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

    // H15: the trail on the sensor; on the sky only with a WCS that still
    // describes this frame.
    let solve = context.solve.as_ref();
    let orientation = solve.map(|solve| trailing::SolveOrientation {
        cd: solve.wcs.map(|wcs| wcs.cd),
        naxis: solve.wcs.map(|wcs| wcs.naxis),
        solve_event_id: Some(solve.event_id.as_str()),
        camera: solve.camera_name.as_deref(),
        pier_side: solve.pier_side,
        rotator_angle_deg: solve.rotator_angle_deg,
        exposed_at: solve.exposure_started_at.or(solve.solved_at),
    });
    let geometry = trailing::FrameGeometry {
        camera: context.camera.as_deref(),
        width: luma.width,
        height: luma.height,
        pier_side: context.pier_side,
        rotator_angle_deg: context.rotator_angle_deg,
        at: context.at,
        alignment_mode: context.alignment_mode,
        solve_on_this_connection: context.solve_on_this_connection,
        mount_repointed_since_solve: context.mount_repointed_since_solve,
    };
    let trailing = trailing::attribute(
        trailing::estimate_trailing(&luma, &background),
        orientation.as_ref(),
        &geometry,
    );
    let ted = TedRunner::new(None).evidence(&trailing, context.window.as_ref());

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
    let new_frame = trends.last_frame_counter != Some(frame.frame_counter);
    trends.last_frame_counter = Some(frame.frame_counter);
    // H2: a frame exposed while equipment moved is not evidence about cloud.
    let counts = new_frame && !context.moved();
    let cloud_probability = update_cloud_probability(
        trends,
        quality.star_count_detected,
        background.background_adu,
        quality.observation_valid(),
        counts,
    );
    let cloud_event = if counts {
        update_cloud_event(trends, cloud_probability)
    } else {
        CloudEvent::None
    };

    let observation_valid = quality.observation_valid();
    Ok(context.apply(SharedObservation {
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
        trailing_axis: trailing.axis_label(),
        trailing_direction_pa: trailing.sensor_pa_deg,
        ted: Some(ted),
        background_adu: background.background_adu,
        sky_background_adu: background.background_adu,
        background_sigma: background.background_sigma,
        snr_proxy_median: background.snr_proxy_median,
        saturation_fraction: background.saturation_fraction,
        saturation_gate: quality.saturation_gate,
        // Filled by `context.apply`: from plate_solve_events and the hub,
        // never carried forward from `previous`.
        pointing_error_arcmin: None,
        wcs_residual_arcsec: None,
        last_solve_confidence: None,
        last_solve_at: None,
        last_solve_event_id: None,
        equipment: None,
        coc_context: None,
        observation_valid,
    }))
}

fn update_cloud_probability(
    trends: &mut VisionTrendState,
    star_count: u32,
    background_adu: f32,
    observation_valid: bool,
    new_frame: bool,
) -> f32 {
    // Clear-sky references adapt asymmetrically, once per distinct frame.
    // - Clearer evidence (more stars; a darker background while stars also look
    //   clear) is adopted quickly but by at most RECOVER_MAX_RATIO per frame, so
    //   a cloudy start recovers within a few frames while one odd frame (a long
    //   test exposure, a gain change) cannot inflate the reference much.
    // - Worse evidence is adopted slowly while the frame looks clear, so
    //   gradually thickening cloud cannot drag the reference down with it.
    // - While the reading sits elevated without a declared cloud event, the
    //   references relax very slowly, so no inflated reference can persist.
    // Star counts alone cannot tell a cloudy start from a sparse field; a
    // catalog-referenced transparency measurement is the intended successor.
    const RECOVER_ALPHA: f32 = 0.5;
    const RECOVER_MAX_RATIO: f32 = 1.5;
    const DECAY_ALPHA: f32 = 0.02;
    const RELAX_ALPHA: f32 = 0.005;
    const CLEAR_STAR_DROP: f32 = 0.2;

    let stars = star_count as f32;
    if new_frame && observation_valid {
        if trends.clear_star_baseline.is_none() && star_count > 0 {
            trends.clear_star_baseline = Some(stars.max(1.0));
        }
        if trends.clear_background_baseline.is_none() && background_adu.is_finite() {
            trends.clear_background_baseline = Some(background_adu.max(1.0));
        }
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

    if !new_frame || !observation_valid {
        return probability;
    }
    let looks_clear = probability < 0.40 && star_count > 0;
    let elevated_without_event =
        !looks_clear && star_count > 0 && trends.cloud_status == CloudStatus::Clear;

    if star_count > 0 && stars > star_baseline {
        trends.clear_star_baseline = Some(
            exp_smooth(star_baseline, stars, RECOVER_ALPHA).min(star_baseline * RECOVER_MAX_RATIO),
        );
    } else if looks_clear {
        trends.clear_star_baseline = Some(exp_smooth(star_baseline, stars, DECAY_ALPHA).max(1.0));
    } else if elevated_without_event {
        trends.clear_star_baseline = Some(exp_smooth(star_baseline, stars, RELAX_ALPHA).max(1.0));
    }

    if background_adu.is_finite() {
        let background = background_adu.max(1.0);
        if background < background_baseline && star_drop < CLEAR_STAR_DROP {
            trends.clear_background_baseline = Some(
                exp_smooth(background_baseline, background, RECOVER_ALPHA)
                    .max(background_baseline / RECOVER_MAX_RATIO),
            );
        } else if looks_clear {
            trends.clear_background_baseline =
                Some(exp_smooth(background_baseline, background, DECAY_ALPHA));
        } else if elevated_without_event && background > background_baseline {
            trends.clear_background_baseline =
                Some(exp_smooth(background_baseline, background, RELAX_ALPHA));
        }
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

    fn feed(trends: &mut VisionTrendState, stars: u32, background: f32, frames: usize) -> f32 {
        let mut probability = 0.0;
        for _ in 0..frames {
            probability = update_cloud_probability(trends, stars, background, true, true);
        }
        probability
    }

    #[test]
    fn cloudy_start_recovers_quickly_once_the_sky_clears() {
        let mut trends = VisionTrendState::default();
        feed(&mut trends, 5, 100.0, 10);
        assert_eq!(trends.clear_star_baseline, Some(5.0));
        // The previous symmetric 5% smoothing needed about 42 frames here; the
        // capped recovery climbs 5 -> 7.5 -> 11 -> 17 -> 25 -> 38 -> 44 -> 47.
        feed(&mut trends, 50, 100.0, 7);
        assert!(trends.clear_star_baseline.unwrap() >= 45.0);
        // Losing those stars again is now recognised as cloud.
        assert!(feed(&mut trends, 5, 100.0, 1) > 0.55);
    }

    #[test]
    fn gradual_thickening_does_not_drag_the_clear_reference_down() {
        let mut trends = VisionTrendState::default();
        feed(&mut trends, 50, 100.0, 10);
        let mut probability = 0.0;
        for step in 0..=40 {
            let stars = (50.0 - step as f32).round() as u32;
            probability = feed(&mut trends, stars, 100.0, 1);
        }
        // Symmetric 5% smoothing let the reference sag to about 29 stars.
        assert!(trends.clear_star_baseline.unwrap() > 40.0);
        assert!(probability > 0.5);
    }

    #[test]
    fn darker_background_is_adopted_and_invalid_frames_do_not_move_references() {
        let mut trends = VisionTrendState::default();
        // An invalid first frame must not seed either reference.
        update_cloud_probability(&mut trends, 20, 400.0, false, true);
        assert!(trends.clear_star_baseline.is_none());
        assert!(trends.clear_background_baseline.is_none());

        feed(&mut trends, 30, 400.0, 1);
        feed(&mut trends, 30, 100.0, 6);
        assert!(trends.clear_background_baseline.unwrap() < 125.0);

        let star_before = trends.clear_star_baseline;
        let background_before = trends.clear_background_baseline;
        update_cloud_probability(&mut trends, 300, 10.0, false, true);
        assert_eq!(trends.clear_star_baseline, star_before);
        assert_eq!(trends.clear_background_baseline, background_before);
    }
    #[test]
    fn one_inflated_frame_cannot_lock_in_a_false_cloud_reading() {
        let mut trends = VisionTrendState::default();
        feed(&mut trends, 50, 100.0, 20);
        // One long test exposure: three times the stars, five times the sky.
        feed(&mut trends, 150, 500.0, 1);
        assert!(trends.clear_star_baseline.unwrap() <= 75.0 + 1e-3);
        let back = feed(&mut trends, 50, 100.0, 1);
        assert!(back < 0.40, "normal sky must still look clear, got {back}");
        assert!(feed(&mut trends, 50, 100.0, 100) < 0.1);
    }
    #[test]
    fn sustained_inflation_relaxes_instead_of_sticking_forever() {
        let mut trends = VisionTrendState::default();
        feed(&mut trends, 50, 100.0, 20);
        feed(&mut trends, 150, 500.0, 10);
        assert!(feed(&mut trends, 50, 100.0, 1) >= 0.40);
        // A symmetric reference frozen above about 2.3x the normal count never
        // recovered; the slow relaxation bounds how long it can persist.
        assert!(feed(&mut trends, 50, 100.0, 300) < 0.1);
    }

    #[test]
    fn dark_excursion_without_clear_stars_does_not_poison_the_background() {
        let mut trends = VisionTrendState::default();
        feed(&mut trends, 50, 100.0, 10);
        // Short exposures to frame a bright star: few stars and a dark sky.
        feed(&mut trends, 10, 25.0, 10);
        assert!((trends.clear_background_baseline.unwrap() - 100.0).abs() < 1.0);
        assert!(feed(&mut trends, 30, 100.0, 1) < 0.40);
    }

    #[test]
    fn reanalysing_the_same_frame_neither_adapts_nor_advances_events() {
        let mut trends = VisionTrendState::default();
        feed(&mut trends, 50, 100.0, 10);
        let before = trends.clear_star_baseline;
        for _ in 0..10 {
            update_cloud_probability(&mut trends, 150, 500.0, true, false);
        }
        assert_eq!(trends.clear_star_baseline, before);

        let mut frame = mono_frame();
        frame.frame_counter = 99;
        let mut trends = VisionTrendState::default();
        trends.cloud_status = CloudStatus::Clear;
        for _ in 0..5 {
            observation_from_frame_with_trends(&frame, &mut trends, None, None).unwrap();
        }
        assert_eq!(trends.last_frame_counter, Some(99));
        assert!(trends.consecutive_cloudy <= 1);
    }

    // ── Equipment hooks (H2, H14, H15, plate-solve state) ──────────────────

    use crate::equipment::events::{EquipmentEvent, EventKind};
    use crate::equipment::hub::{EquipmentHub, HubConfig};
    use crate::equipment::ring::DeviceSnapshot;
    use crate::equipment::snapshot::MountSnapshot;
    use crate::equipment::testing::fake_handles::fake_mount;
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::traits::{Clock, NullRecorder};
    use crate::equipment::types::{EqFrame, Observed, Source};
    use crate::processing::plate_solver::TanWcs;

    #[test]
    fn the_segmenting_events_are_the_ones_the_spec_names() {
        for name in [
            "slew_commanded",
            "slew_settled",
            "filter_moving",
            "filter_change_ended",
            "focuser_move_started",
            "focuser_move_stopped",
            "exposure_settings_changed",
            "stream_restarted",
        ] {
            assert!(segments_baselines(name), "{name}");
        }
        for name in [
            "guide_step",
            "focuser_settled",
            "weather_sample",
            "pulse_guide",
        ] {
            assert!(!segments_baselines(name), "{name}");
        }
    }

    #[test]
    fn draining_the_bus_reports_the_first_segmenting_event_and_a_lag() {
        let clock = ManualClock::new();
        let (tx, mut rx) = tokio::sync::broadcast::channel(4);
        let send = |kind| {
            tx.send(Arc::new(EquipmentEvent::new(kind, &clock)))
                .unwrap();
        };
        assert_eq!(drain_segmentation(&mut rx), None);
        send(EventKind::GuiderPaused);
        send(EventKind::SlewStarted);
        send(EventKind::FilterMoving { to_slot: 1 });
        assert_eq!(drain_segmentation(&mut rx).as_deref(), Some("slew_started"));
        assert_eq!(drain_segmentation(&mut rx), None, "drained");
        for _ in 0..6 {
            send(EventKind::GuiderPaused);
        }
        assert_eq!(drain_segmentation(&mut rx).as_deref(), Some("bus_lagged"));

        // A segment restarts the baselines but never recounts the frame.
        let mut trends = VisionTrendState {
            clear_star_baseline: Some(40.0),
            consecutive_cloudy: 2,
            last_frame_counter: Some(9),
            ..VisionTrendState::default()
        };
        trends.segment();
        assert_eq!(trends.clear_star_baseline, None);
        assert_eq!(trends.consecutive_cloudy, 0);
        assert_eq!(trends.last_frame_counter, Some(9));
    }

    #[test]
    fn a_stream_frame_window_is_two_intervals_before_its_arrival() {
        let t0 = DateTime::from_timestamp(1_792_029_600, 0).unwrap();
        let mut timing = FrameTiming::default();
        timing.observe(1, 10, t0);
        assert_eq!(timing.window(), None, "no interval yet");
        timing.observe(1, 10, t0 + chrono::Duration::seconds(1));
        assert_eq!(timing.window(), None, "the same frame again");
        timing.observe(1, 11, t0 + chrono::Duration::seconds(2));
        assert_eq!(
            timing.window(),
            Some((
                t0 - chrono::Duration::seconds(2),
                t0 + chrono::Duration::seconds(2)
            ))
        );
        // A new stream starts over.
        timing.observe(2, 0, t0 + chrono::Duration::seconds(3));
        assert_eq!(timing.window(), None);
    }

    fn mount_sample(clock: &ManualClock, slewing: bool) -> Arc<MountSnapshot> {
        let (utc, mono) = (clock.now_utc(), clock.now_mono_ns());
        let mut snap = MountSnapshot::default();
        snap.state.ra_h = Observed::fresh(5.5, Source::DeviceState, utc, mono);
        snap.state.dec_deg = Observed::fresh(20.0, Source::DeviceState, utc, mono);
        snap.state.slewing = Observed::fresh(slewing, Source::DeviceState, utc, mono);
        snap.state.tracking = Observed::fresh(true, Source::DeviceState, utc, mono);
        snap.capabilities.native_frame = EqFrame::JNowApparent;
        snap.set_published(utc, mono);
        Arc::new(snap)
    }

    #[test]
    fn equipment_validity_reports_quiescence_motion_and_unknowns() {
        let clock = Arc::new(ManualClock::new());
        let hub = EquipmentHub::new(
            HubConfig::default(),
            clock.clone(),
            Arc::new(NullRecorder::new()),
        );
        let segments = Segments {
            count: 3,
            last: Some("slew_settled".into()),
        };
        let t0 = clock.now_utc();
        let window = (
            t0 + chrono::Duration::milliseconds(3500),
            t0 + chrono::Duration::milliseconds(4500),
        );

        let none = equipment_validity(&hub, &hub.snapshot(), Some(window), &segments);
        assert_eq!(none.quiescent, None);
        assert_eq!(none.reason.as_deref(), Some("no_motion_devices"));
        assert_eq!(none.clock_quality, ClockQuality::HostArrival);

        let (mount, _fake) = fake_mount(MountSnapshot::default());
        let ring = hub.attach(mount.clone());
        for _ in 0..5 {
            clock.advance_ms(1000);
            ring.push(mount_sample(&clock, false));
        }
        let unknown = equipment_validity(&hub, &hub.snapshot(), None, &segments);
        assert_eq!(unknown.reason.as_deref(), Some("frame_interval_unknown"));

        let quiet = equipment_validity(&hub, &hub.snapshot(), Some(window), &segments);
        assert_eq!(quiet.quiescent, Some(true), "{quiet:?}");
        assert_eq!(quiet.kinds_checked, vec![DeviceKind::Mount]);
        assert_eq!(
            (quiet.segment, quiet.segment_reset_by.as_deref()),
            (3, Some("slew_settled"))
        );

        clock.set_utc(t0 + chrono::Duration::milliseconds(4000));
        let slew = hub
            .emit(
                EquipmentEvent::new(EventKind::SlewStarted, clock.as_ref())
                    .for_device(DeviceKind::Mount, mount.connection_id()),
            )
            .unwrap();
        let moved = equipment_validity(&hub, &hub.snapshot(), Some(window), &segments);
        assert_eq!(moved.quiescent, Some(false));
        assert_eq!(moved.moved, Some(DeviceKind::Mount));
        assert_eq!(moved.moved_event_seq, Some(slew.seq));
        assert_eq!(moved.reason.as_deref(), Some("moved:mount"));
    }

    fn validity(quiescent: Option<bool>) -> EquipmentValidity {
        EquipmentValidity {
            quiescent,
            reason: None,
            settled_since: None,
            moved: None,
            moved_event_seq: None,
            window_start: None,
            window_end: None,
            clock_quality: ClockQuality::HostArrival,
            kinds_checked: vec![DeviceKind::Mount],
            mount_age_ms: None,
            focuser_age_ms: None,
            filter_wheel_age_ms: None,
            guider_age_ms: None,
            mount_slewing: None,
            mount_tracking: None,
            pulse_guiding: None,
            guider_state: None,
            resume_block_active: false,
            segment: 0,
            segment_reset_by: None,
        }
    }

    #[test]
    fn a_frame_that_moved_never_counts_toward_a_cloud_event() {
        // No stars on an empty sky: a high cloud reading on every frame, so
        // three counted frames declare cloud.
        let quiet = ObservationContext {
            equipment: Some(validity(Some(true))),
            ..ObservationContext::default()
        };
        let moving = ObservationContext {
            equipment: Some(validity(Some(false))),
            ..ObservationContext::default()
        };
        let run = |context: &ObservationContext| {
            let mut trends = VisionTrendState::default();
            let mut events = Vec::new();
            for counter in 0..3 {
                let mut frame = mono_frame();
                frame.frame_counter = counter;
                let previous = observation_from_frame(&frame).unwrap();
                let observation =
                    observation_with_context(&frame, &mut trends, Some(&previous), None, context)
                        .unwrap();
                events.push(observation.cloud_event);
            }
            (events, trends.consecutive_cloudy)
        };
        let (events, _) = run(&quiet);
        assert_eq!(events[2], CloudEvent::CloudDetected);
        let (events, cloudy) = run(&moving);
        assert!(events.iter().all(|e| *e == CloudEvent::None));
        assert_eq!(cloudy, 0);
    }

    fn latest_solve(camera: &str, pier: PierSide) -> LatestSolve {
        LatestSolve {
            event_id: "solve-9".into(),
            frame_id: Some("frame-9".into()),
            solved_at: Some(DateTime::from_timestamp(1_792_029_600, 0).unwrap()),
            exposure_started_at: None,
            trigger: Some("post_slew".into()),
            ra_deg: Some(83.8),
            dec_deg: Some(-5.4),
            position_angle_deg: Some(0.0),
            pixel_scale_arcsec: Some(1.2),
            mirrored: Some(false),
            wcs: Some(TanWcs {
                crpix: [2000.5, 1500.5],
                crval: [83.8, -5.4],
                cd: [[-3.0e-4, 0.0], [0.0, 3.0e-4]],
                naxis: [4000, 3000],
            }),
            pointing_error_arcmin: Some(2.5),
            camera_name: Some(camera.into()),
            pier_side: Some(pier),
            mount_identity_key: Some("sim:mount".into()),
            sample_mid_exposure_utc: Some(DateTime::from_timestamp(1_792_029_590, 0).unwrap()),
            rotator_angle_deg: None,
            exposure: None,
        }
    }

    /// The frame an hour after `latest_solve`, on the same German
    /// equatorial mount connection.
    fn same_rig_context() -> ObservationContext {
        ObservationContext {
            solve: Some(latest_solve("SV105", PierSide::East)),
            camera: Some("SV105".into()),
            pier_side: Some(PierSide::East),
            at: Some(DateTime::from_timestamp(1_792_033_200, 0).unwrap()),
            alignment_mode: Some(AlignmentMode::GermanPolar),
            solve_on_this_connection: Some(true),
            ..ObservationContext::default()
        }
    }

    #[test]
    fn the_plate_solve_state_comes_from_the_table_never_the_previous_observation() {
        let frame = mono_frame();
        let mut previous = observation_from_frame(&frame).unwrap();
        previous.last_solve_at = Some(Utc::now());
        previous.pointing_error_arcmin = Some(9.0);
        let carried = observation_with_context(
            &frame,
            &mut VisionTrendState::default(),
            Some(&previous),
            None,
            &ObservationContext::default(),
        )
        .unwrap();
        assert_eq!(carried.last_solve_at, None, "nothing is carried forward");
        assert_eq!(carried.pointing_error_arcmin, None);

        let solved = ObservationContext {
            solve: Some(latest_solve("SV105", PierSide::East)),
            ..ObservationContext::default()
        };
        let observation = observation_with_context(
            &frame,
            &mut VisionTrendState::default(),
            Some(&previous),
            None,
            &solved,
        )
        .unwrap();
        assert_eq!(
            observation.last_solve_at,
            Some(DateTime::from_timestamp(1_792_029_600, 0).unwrap())
        );
        assert_eq!(observation.pointing_error_arcmin, Some(2.5));
        assert_eq!(observation.last_solve_event_id.as_deref(), Some("solve-9"));
        assert_eq!(observation.last_solve_confidence, None);
    }

    /// One star trailed along the sensor's x axis.
    fn trailed_frame() -> FrameData {
        let mut frame = mono_frame();
        for x in 24..40 {
            frame.data[32 * 64 + x] = 250;
        }
        for x in 27..37 {
            frame.data[31 * 64 + x] = 180;
            frame.data[33 * 64 + x] = 180;
        }
        frame
    }

    #[test]
    fn the_bus_trailing_axis_is_ra_or_dec_only_with_a_wcs_that_still_applies() {
        let analyse = |context: &ObservationContext| {
            observation_with_context(
                &trailed_frame(),
                &mut VisionTrendState::default(),
                None,
                None,
                context,
            )
            .unwrap()
        };
        let unsolved = analyse(&ObservationContext::default());
        assert!(unsolved.trailing_severity > 0.15);
        assert_eq!(unsolved.trailing_axis.as_deref(), Some("sensor_x"));
        let ted = unsolved.ted.unwrap();
        assert_eq!(ted.sky_axis, None);
        assert_eq!(ted.sky_withheld.as_deref(), Some("no_plate_solve"));

        let same_rig = same_rig_context();
        let solved = analyse(&same_rig);
        assert_eq!(solved.trailing_axis.as_deref(), Some("RA"));
        assert_eq!(
            solved.ted.unwrap().solve_event_id.as_deref(),
            Some("solve-9")
        );

        let other_camera = ObservationContext {
            camera: Some("R50".into()),
            ..same_rig.clone()
        };
        let withheld = analyse(&other_camera);
        assert_eq!(withheld.trailing_axis.as_deref(), Some("sensor_x"));
        assert_eq!(
            withheld.ted.unwrap().sky_withheld.as_deref(),
            Some("solved_with_another_camera")
        );

        // An Alt/Az mount an hour later: the field has rotated.
        let altaz = ObservationContext {
            alignment_mode: Some(AlignmentMode::AltAz),
            ..same_rig.clone()
        };
        let rotated = analyse(&altaz);
        assert_eq!(rotated.trailing_axis.as_deref(), Some("sensor_x"));
        assert_eq!(
            rotated.ted.unwrap().sky_withheld.as_deref(),
            Some("field_rotation_since_solve")
        );
        // The mount reconnected since the solve.
        let reconnected = ObservationContext {
            solve_on_this_connection: Some(false),
            ..same_rig
        };
        assert_eq!(
            analyse(&reconnected).ted.unwrap().sky_withheld.as_deref(),
            Some("solve_from_another_mount_connection")
        );
    }

    #[test]
    fn a_solve_belongs_to_the_connection_whose_segment_holds_its_sample() {
        let solve = latest_solve("SV105", PierSide::East);
        let sample = solve.sample_mid_exposure_utc.unwrap();
        let before = sample - chrono::Duration::seconds(1);
        let after = sample + chrono::Duration::seconds(1);
        assert_eq!(
            solve_on_connection(&solve, Some("sim:mount"), Some(before), true),
            Some(true)
        );
        // This connection's samples all began after the solve's.
        assert_eq!(
            solve_on_connection(&solve, Some("sim:mount"), Some(after), true),
            Some(false)
        );
        assert_eq!(
            solve_on_connection(&solve, Some("sim:mount"), None, true),
            Some(false),
            "nothing sampled on this connection yet"
        );
        assert_eq!(
            solve_on_connection(&solve, Some("alpaca:other"), Some(before), true),
            Some(false)
        );
        assert_eq!(solve_on_connection(&solve, None, Some(before), false), None);
        let unsampled = LatestSolve {
            sample_mid_exposure_utc: None,
            ..solve
        };
        assert_eq!(
            solve_on_connection(&unsampled, Some("sim:mount"), Some(before), true),
            None
        );
    }

    #[test]
    fn only_repointing_events_turn_an_alt_az_field() {
        let clock = ManualClock::new();
        let event = |kind| EquipmentEvent::new(kind, &clock);
        assert!(repoints_mount(&event(EventKind::SlewStarted)));
        assert!(repoints_mount(&event(EventKind::Homed)));
        assert!(!repoints_mount(&event(EventKind::SettleBegin)));
        assert!(!repoints_mount(&event(EventKind::Centered {
            error_arcmin: 0.5,
            attempts: 2,
        })));
    }

    /// H2, H14 and the plate-solve state do not need the pixels.
    #[test]
    fn an_unmeasurable_frame_still_carries_its_equipment_and_solve_context() {
        let mut frame = mono_frame();
        frame.format = OlsStreamType::Mjpeg;
        let context = ObservationContext {
            equipment: Some(validity(Some(true))),
            coc: Some(CocContext::default()),
            ..same_rig_context()
        };
        let observation = observation_with_context(
            &frame,
            &mut VisionTrendState::default(),
            None,
            None,
            &context,
        )
        .unwrap();
        assert!(observation.analysis_unavailable_reason.is_some());
        assert_eq!(
            observation.equipment.as_ref().and_then(|e| e.quiescent),
            Some(true)
        );
        assert!(observation.coc_context.is_some());
        assert_eq!(observation.last_solve_event_id.as_deref(), Some("solve-9"));
        assert_eq!(observation.pointing_error_arcmin, Some(2.5));
        assert_eq!(
            observation.last_solve_at,
            Some(DateTime::from_timestamp(1_792_029_600, 0).unwrap())
        );
        let applied = context.apply(SharedObservation::analysis_unavailable("x".into()));
        assert_eq!(applied.last_solve_event_id.as_deref(), Some("solve-9"));
        assert!(applied.equipment.is_some() && applied.coc_context.is_some());
    }
}

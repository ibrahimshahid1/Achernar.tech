//! Automatic plate solving — the runtime half of `plate_solve_trigger` (M-3,
//! equipment spec H11).
//!
//! [`attach`] subscribes the trigger to the equipment hub's event bus: slews,
//! settles, pier-side changes, flips, reconnects, star loss and safety are
//! read from the bus, never inferred from the registry's position probes. A
//! lagging subscription catches up from the hub's journal, and events are
//! applied once each, in sequence order. The subscription holds no strong
//! reference to the hub; it ends when the hub's bus closes.
//!
//! [`frame_archived`] asks the trigger about each committed frame and runs
//! the solves it wants through the same path as `POST /platesolve/solve`, so
//! an automatic solve is recorded exactly like a manual one, with its reason
//! in `trigger` and its H12 pointing sample. Every decision about a
//! candidate frame is an H23 record, updated with the solve's result.
//!
//! Nothing here calls a device driver: mount state comes from the hub's
//! snapshot, which the mount's worker publishes.

use std::sync::{Arc, Mutex, Weak};

use chrono::{DateTime, Utc};
use serde_json::json;
use tokio::sync::broadcast::{self, error::RecvError};

use super::plate_solve_trigger::{
    ArchivedFrame, FrameDecision, PlateSolveTrigger, RuleChange, TriggerReadings,
};
use crate::api::plate_solve::{solve_and_record, HintRequest, SolveFrameRequest};
use crate::equipment::events::{EquipmentEvent, GroundTruthRecord};
use crate::equipment::hub::EquipmentHub;
use crate::equipment::traits::EquipmentReader;
use crate::equipment::types::{EqCoord, EqFrame};
use crate::state::AppState;

/// Journal entries read per catch-up request (the route's own cap).
const CATCH_UP_BATCH: usize = 500;

fn lock(trigger: &Mutex<PlateSolveTrigger>) -> std::sync::MutexGuard<'_, PlateSolveTrigger> {
    trigger.lock().unwrap_or_else(|e| e.into_inner())
}

/// Subscribe the trigger to `state`'s equipment bus, once per hub. Returns
/// true when this call started the subscription. A disabled trigger, or a
/// caller outside a tokio runtime, starts nothing.
///
/// The vision loop calls this at startup, so the subscription runs from
/// boot; every other entry point calls it too, so a state built without the
/// vision loop (tests, tools) still follows the bus.
pub fn attach(state: &AppState) -> bool {
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return false;
    };
    if !lock(&state.plate_solve_trigger).is_enabled() {
        return false;
    }
    let hub = Arc::clone(&state.equipment);
    // Read the next sequence number before subscribing: an event published
    // in between reaches the subscription and the catch-up both, and is
    // applied once.
    let from_seq = hub.bus().next_seq();
    let receiver = hub.subscribe();
    {
        let mut trigger = lock(&state.plate_solve_trigger);
        if trigger.attached_to(&hub) {
            return false;
        }
        trigger.mark_attached(&hub, from_seq);
    }
    runtime.spawn(feed(
        Arc::clone(&state.plate_solve_trigger),
        receiver,
        Arc::downgrade(&hub),
        from_seq.saturating_sub(1),
    ));
    true
}

async fn feed(
    trigger: Arc<Mutex<PlateSolveTrigger>>,
    mut receiver: broadcast::Receiver<Arc<EquipmentEvent>>,
    hub: Weak<EquipmentHub>,
    since: u64,
) {
    catch_up(&trigger, &hub, since);
    loop {
        match receiver.recv().await {
            Ok(event) => apply(&trigger, &event),
            Err(RecvError::Lagged(missed)) => {
                tracing::warn!("Plate-solve trigger lagged {missed} events; catching up");
                let since = lock(&trigger).status().last_event_seq;
                catch_up(&trigger, &hub, since);
            }
            Err(RecvError::Closed) => return,
        }
    }
}

/// Apply the journal after `since` (guide steps are not journaled and do not
/// matter here).
fn catch_up(trigger: &Mutex<PlateSolveTrigger>, hub: &Weak<EquipmentHub>, mut since: u64) {
    let Some(hub) = hub.upgrade() else {
        return;
    };
    loop {
        let events = hub.events_since(since, &[], CATCH_UP_BATCH);
        let Some(last) = events.last() else {
            return;
        };
        since = last.seq;
        for event in &events {
            apply(trigger, event);
        }
        if events.len() < CATCH_UP_BATCH {
            return;
        }
    }
}

fn apply(trigger: &Mutex<PlateSolveTrigger>, event: &EquipmentEvent) {
    let change = lock(trigger).observe(event);
    match change {
        Some(RuleChange::Armed(reason)) => tracing::info!(
            "Plate-solve rule {} armed by {} (seq {})",
            reason.as_str(),
            event.name(),
            event.seq
        ),
        Some(RuleChange::Raised(reason)) => tracing::info!(
            "Plate-solve rule {} raised by {} (seq {})",
            reason.as_str(),
            event.name(),
            event.seq
        ),
        Some(RuleChange::Relabelled(reason)) => tracing::info!(
            "Plate-solve rule is now {} after {}",
            reason.as_str(),
            event.name()
        ),
        Some(RuleChange::Dropped(reason)) => tracing::warn!(
            "Plate-solve rule {} dropped: {} (seq {})",
            reason.as_str(),
            event.name(),
            event.seq
        ),
        Some(RuleChange::Satisfied) => {
            tracing::info!("Plate-solve rule satisfied by {}", event.name())
        }
        None => {}
    }
}

/// A GoTo was accepted through the API. The rule itself is armed by the
/// `slew_commanded` event the mount's worker publishes and fired by its
/// `slew_settled`; this only makes sure the trigger is subscribed.
pub fn slew_commanded(state: &AppState, target: crate::hardware::mount::EqCoord) {
    if attach(state) {
        tracing::debug!(
            "Plate-solve trigger subscribed at a slew to RA {:.4}h Dec {:+.4}°",
            target.ra,
            target.dec
        );
    }
}

/// The vision loop saw the sky clear: a Recovery solve (v6 §8.3.3).
pub fn cloud_cleared(state: &AppState) {
    attach(state);
    if lock(&state.plate_solve_trigger)
        .cloud_cleared(Utc::now())
        .is_some()
    {
        tracing::info!("Plate-solve rule recovery raised: the sky cleared");
    }
}

/// The M-3 readings available now (v6 §8.3.3), each with its age.
fn readings(state: &AppState) -> TriggerReadings {
    let snapshot = state.equipment.snapshot();
    let now_mono = state.equipment.clock().now_mono_ns();
    let mut readings = TriggerReadings::default();
    if let Some(mount) = &snapshot.mount {
        readings.ha_h = mount.ha_h;
        readings.alt_deg = mount.state.alt_deg.known().copied();
        readings.airmass = readings
            .alt_deg
            .filter(|alt| *alt > 0.0)
            .map(kasten_young_airmass);
        readings.pier_side = mount.state.side_of_pier.known().copied();
        if let Some(age) = now_mono.checked_sub(mount.published_mono_ns) {
            readings
                .measurement_ages_ms
                .insert("mount".into(), age / 1_000_000);
        }
    }
    if let Some(guider) = &snapshot.guider {
        readings.guide_rms_arcsec = guider.rms.as_ref().and_then(|rms| rms.total_arcsec);
        if let Some(age) = now_mono.checked_sub(guider.published_mono_ns) {
            readings
                .measurement_ages_ms
                .insert("guider".into(), age / 1_000_000);
        }
    }
    let ambient = state.equipment.ambient_temperature();
    if let Some(reading) = ambient.known().filter(|t| !t.camera_sensor_fallback) {
        readings.temperature_c = Some(reading.celsius);
        if let Some(age) = ambient.age_ms(now_mono) {
            readings
                .measurement_ages_ms
                .insert("temperature".into(), age);
        }
    }
    let observation = state
        .vision_observation
        .read()
        .ok()
        .and_then(|bus| bus.clone());
    if let Some(observation) = observation {
        readings.star_count = Some(observation.star_count);
        readings.cloud_probability = Some(f64::from(observation.cloud_probability));
        let age = (Utc::now() - observation.timestamp)
            .num_milliseconds()
            .max(0);
        readings
            .measurement_ages_ms
            .insert("vision".into(), u64::try_from(age).unwrap_or(0));
    }
    readings
}

/// Kasten and Young (1989) relative airmass at apparent altitude `alt_deg`.
fn kasten_young_airmass(alt_deg: f64) -> f64 {
    1.0 / (alt_deg.to_radians().sin() + 0.50572 * (alt_deg + 6.07995).powf(-1.6364))
}

/// Where the mount says it points, J2000 when the converter knows its frame.
/// A reading off the sphere is no position at all: solving blind beats a
/// hint the request validation would refuse on every retry.
fn mount_hint(state: &AppState) -> Option<EqCoord> {
    let snapshot = state.equipment.snapshot();
    let mount = snapshot.mount.as_ref()?;
    let position = mount.j2000.or_else(|| {
        crate::equipment::mount::position(&mount.state, mount.capabilities.native_frame)
    })?;
    on_sphere(position).then_some(position)
}

fn on_sphere(position: EqCoord) -> bool {
    (0.0..24.0).contains(&position.ra_h) && (-90.0..=90.0).contains(&position.dec_deg)
}

/// The archive committed frame `frame_id`. `exposure_started_at` is the
/// RFC 3339 exposure start the capture reported.
pub fn frame_archived(
    state: &AppState,
    frame_id: &str,
    capture_type: &str,
    exposure_started_at: &str,
) {
    attach(state);
    if !state.plate_solver.is_available() {
        return;
    }
    // An exposure time that cannot be read cannot be shown to follow a slew.
    let Ok(started) = DateTime::parse_from_rfc3339(exposure_started_at) else {
        return;
    };
    let frame = ArchivedFrame {
        frame_id,
        capture_type,
        exposure_started_at: started.with_timezone(&Utc),
    };
    let readings = readings(state);
    let now = Utc::now();
    let (decision, record) = {
        let mut trigger = lock(&state.plate_solve_trigger);
        let decision = trigger.frame_archived(frame, now);
        let record = trigger.decision_record(&frame, &decision, &readings, now);
        (decision, record)
    };
    let recorder = state.equipment.recorder();
    if let Some(record) = &record {
        recorder.record(GroundTruthRecord::Decision(Box::new(record.clone())));
    }
    let FrameDecision::Solve(job) = decision else {
        return;
    };

    // A slew target is the best hint there is; then the mount's own report
    // of where it points; without a mount, blind.
    let hint = job
        .hint
        .filter(|hint| on_sphere(*hint))
        .or_else(|| mount_hint(state));
    let request = SolveFrameRequest {
        frame_id: job.frame_id.clone(),
        hint: hint.map(|coord| HintRequest {
            ra_hours: coord.ra_h,
            dec_deg: coord.dec_deg,
            search_radius_deg: None,
        }),
        pixel_scale_arcsec: None,
        ignore_rig_optics: false,
    };
    tracing::info!(
        "Plate solving frame {} automatically ({})",
        job.frame_id,
        job.reason.as_str()
    );
    let state = state.clone();
    tokio::spawn(async move {
        let response = solve_and_record(&state, request, job.reason.as_str()).await;
        let (solved, executed, reward) = match &response {
            Ok(response) => {
                let solved = response
                    .solution
                    .as_ref()
                    .map(|s| EqCoord::new(s.ra_hours, s.dec_deg, EqFrame::J2000));
                let error_arcmin = response.pointing.as_ref().and_then(|p| p.error_arcmin);
                (
                    solved,
                    json!({
                        "arm": "solve",
                        "event_id": response.event_id,
                        "success": response.success,
                        "failure": response.failure.as_ref().map(|f| f.kind.as_str()),
                    }),
                    json!({
                        "solved": response.success,
                        "pointing_error_arcmin": error_arcmin,
                        "solve_time_ms": response.solve_time_ms,
                    }),
                )
            }
            Err(error) => {
                tracing::warn!(
                    "Automatic plate solve of {} could not run: {}",
                    job.frame_id,
                    error.message
                );
                (
                    None,
                    json!({"arm": "solve", "error": error.code}),
                    json!({"solved": false}),
                )
            }
        };
        let finished = Utc::now();
        lock(&state.plate_solve_trigger).job_finished(
            &job,
            solved,
            readings.temperature_c,
            finished,
        );
        if let Some(mut record) = record {
            record.executed_action = Some(executed);
            record.reward_components = Some(reward);
            record.attribution_window_start = Some(now);
            record.attribution_window_end = Some(finished);
            state
                .equipment
                .recorder()
                .record(GroundTruthRecord::Decision(Box::new(record)));
        }
    });
}

/// The real archive, database and trigger, with a stand-in ASTAP, and mount
/// events published on the hub's bus.
#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::api::plate_solve::test_support::{archive_with_frame, close, scratch};
    use crate::config::PlateSolveConfig;
    use crate::equipment::events::EventKind;
    use crate::equipment::types::{CommandId, DeviceKind};
    use crate::processing::astap::test_support::{fake_astap, SOLVED_INI};
    use crate::processing::astap::AstapSolver;
    use crate::processing::plate_solver::PlateSolver;
    use std::time::Duration;

    async fn armed(directory: &std::path::Path) -> (AppState, String) {
        let solver = PlateSolver::with_backend_for_test(
            AstapSolver::new(
                fake_astap(directory, Some(SOLVED_INI), 0, 0.0),
                &PlateSolveConfig::default(),
            ),
            directory.join("work"),
        );
        let fits = crate::fits::encode_linear_analysis(&[1.0; 48], 8, 6, "ADU", "test").unwrap();
        let (state, frame_id) = archive_with_frame(directory, solver, fits, 1, None).await;
        (
            state.with_auto_solve(&PlateSolveConfig::default()),
            frame_id,
        )
    }

    /// `(trigger, mode, success, hint_ra_deg)` once the solve is recorded and
    /// the trigger has been told how it went.
    async fn recorded(state: &AppState, frame_id: &str) -> (String, String, i64, Option<f64>) {
        for _ in 0..200 {
            let row: Option<(String, String, i64, Option<f64>)> = sqlx::query_as(
                "SELECT trigger, mode, success, hint_ra_deg FROM plate_solve_events \
                 WHERE frame_id = ?",
            )
            .bind(frame_id)
            .fetch_optional(state.local_store.pool())
            .await
            .unwrap();
            let finished = !state.plate_solve_trigger.lock().unwrap().status().in_flight;
            if let (Some(row), true) = (row, finished) {
                return row;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("no automatic plate solve was recorded for {frame_id}");
    }

    async fn solves(state: &AppState) -> i64 {
        let (rows,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM plate_solve_events")
            .fetch_one(state.local_store.pool())
            .await
            .unwrap();
        rows
    }

    /// Wait (bounded) until the subscription has applied every event so far.
    async fn applied(state: &AppState) {
        let last = state.equipment.bus().next_seq().saturating_sub(1);
        for _ in 0..200 {
            if state
                .plate_solve_trigger
                .lock()
                .unwrap()
                .status()
                .last_event_seq
                >= last
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("the trigger never saw event {last}");
    }

    fn mount_event(state: &AppState, connection: uuid::Uuid, command: CommandId, event: EventKind) {
        let event = EquipmentEvent::new(event, state.equipment.clock().as_ref())
            .for_device(DeviceKind::Mount, connection)
            .with_command(command);
        assert!(state.equipment.emit(event).is_some());
    }

    #[tokio::test]
    async fn the_first_light_frame_after_startup_is_solved_blind_without_a_mount() {
        let directory = scratch();
        let (state, frame_id) = armed(&directory).await;

        frame_archived(&state, &frame_id, "flat", &Utc::now().to_rfc3339());
        frame_archived(&state, &frame_id, "light", &Utc::now().to_rfc3339());

        let row = recorded(&state, &frame_id).await;
        assert_eq!(row, ("startup".into(), "blind".into(), 1, None));
        assert!(
            state
                .plate_solve_trigger
                .lock()
                .unwrap()
                .status()
                .pending
                .is_none(),
            "a successful startup solve satisfies the rule"
        );
        close(state, directory).await;
    }

    #[tokio::test]
    async fn post_slew_fires_from_slew_settled_on_the_bus_not_from_position_probes() {
        let directory = scratch();
        let (state, frame_id) = armed(&directory).await;
        assert!(attach(&state), "the first call subscribes");
        assert!(!attach(&state), "once per hub");

        let connection = uuid::Uuid::new_v4();
        let command = CommandId::new();
        let target = EqCoord::new(8.855, 11.8, EqFrame::J2000);
        mount_event(
            &state,
            connection,
            command,
            EventKind::SlewCommanded {
                target,
                native_target: Some(EqCoord::new(8.87, 11.75, EqFrame::JNowApparent)),
                altaz: None,
            },
        );
        applied(&state).await;
        assert!(
            state
                .plate_solve_trigger
                .lock()
                .unwrap()
                .status()
                .pending
                .unwrap()
                .awaiting_arrival
        );

        // The registry's passive probe reports the mount holding at the
        // target: the old inference would call this arrival. It is not.
        state.devices.write().unwrap().connected(
            DeviceKind::Mount,
            connection.to_string(),
            Some(serde_json::json!({"connected": true, "ra": target.ra_h, "dec": target.dec_deg})),
        );
        for _ in 0..3 {
            state.devices.write().unwrap().checked(
                DeviceKind::Mount,
                &connection.to_string(),
                true,
                Some(serde_json::json!({"connected": true, "ra": target.ra_h, "dec": target.dec_deg})),
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        frame_archived(&state, &frame_id, "light", &Utc::now().to_rfc3339());
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(solves(&state).await, 0, "no solve before slew_settled");
        let status = state.plate_solve_trigger.lock().unwrap().status();
        assert!(status.pending.unwrap().awaiting_arrival);

        mount_event(
            &state,
            connection,
            command,
            EventKind::SlewSettled {
                position: None,
                error_arcmin: Some(0.3),
            },
        );
        applied(&state).await;
        let settled_at = state
            .plate_solve_trigger
            .lock()
            .unwrap()
            .status()
            .pending
            .and_then(|rule| rule.not_before)
            .expect("slew_settled arms the rule");

        // Began before the settle: trailed, never solved.
        let early = settled_at - chrono::Duration::seconds(1);
        frame_archived(&state, &frame_id, "light", &early.to_rfc3339());
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(solves(&state).await, 0);

        let begun = settled_at + chrono::Duration::seconds(1);
        frame_archived(&state, &frame_id, "light", &begun.to_rfc3339());
        let (trigger, mode, success, hint_ra_deg) = recorded(&state, &frame_id).await;
        assert_eq!(
            (trigger.as_str(), mode.as_str(), success),
            ("post_slew", "hinted", 1)
        );
        assert!((hint_ra_deg.unwrap() - target.ra_h * 15.0).abs() < 1e-9);
        assert!(state
            .plate_solve_trigger
            .lock()
            .unwrap()
            .status()
            .pending
            .is_none());

        close(state, directory).await;
    }

    #[tokio::test]
    async fn events_published_before_the_subscription_are_caught_up_from_the_journal() {
        let directory = scratch();
        let (state, _) = armed(&directory).await;
        let connection = uuid::Uuid::new_v4();
        let command = CommandId::new();
        let from = state.equipment.bus().next_seq();
        // Published while nothing was subscribed: the trigger reads it from
        // the journal when it attaches at or before this sequence number.
        mount_event(
            &state,
            connection,
            command,
            EventKind::SlewCommanded {
                target: EqCoord::new(5.588, -5.39, EqFrame::J2000),
                native_target: None,
                altaz: None,
            },
        );
        catch_up(
            &state.plate_solve_trigger,
            &Arc::downgrade(&state.equipment),
            from.saturating_sub(1),
        );
        let status = state.plate_solve_trigger.lock().unwrap().status();
        assert_eq!(status.last_event_seq, from);
        assert!(status.pending.unwrap().awaiting_arrival);
        close(state, directory).await;
    }
}

//! Automatic plate solving — the runtime half of `plate_solve_trigger`.
//!
//! Feeds the trigger the events it decides on (commanded slews, the mount's
//! passive position probes, archived frames) and runs the solves it asks for
//! through the same path as `POST /platesolve/solve`, so an automatic solve is
//! recorded exactly like a manual one, with its reason in `trigger`.
//!
//! Mount position comes only from the snapshot `hardware::health` publishes
//! after each passive probe. Nothing here calls the mount driver: that module
//! keeps at most one blocking SDK call in flight per device, and a second
//! poller would break that.

use std::time::Duration;

use chrono::{DateTime, Utc};

use super::plate_solve_trigger::{ArchivedFrame, SlewSettle};
use crate::api::plate_solve::{solve_and_record, HintRequest, SolveFrameRequest};
use crate::hardware::health::DeviceKind;
use crate::hardware::mount::EqCoord;
use crate::state::AppState;

/// How often the slew watcher reads the published probe. The probe itself
/// runs on its own schedule; this only notices new results promptly.
const WATCH_INTERVAL: Duration = Duration::from_millis(500);

/// Longest a slew may take before the watcher stops waiting for it. Without
/// an arrival there is no moment after which frames are known to be still.
const SLEW_TIMEOUT: Duration = Duration::from_secs(600);

/// A GoTo to `target` was accepted by the mount.
pub fn slew_commanded(state: &AppState, target: EqCoord) {
    let generation = {
        let mut trigger = state.plate_solve_trigger.lock().unwrap();
        if !trigger.status().enabled {
            return;
        }
        trigger.slew_commanded(target)
    };
    let commanded_at = Utc::now();
    let state = state.clone();
    tokio::spawn(async move { watch_slew(state, generation, target, commanded_at).await });
}

async fn watch_slew(
    state: AppState,
    generation: u64,
    target: EqCoord,
    commanded_at: DateTime<Utc>,
) {
    let mut settle = SlewSettle::new(target, commanded_at);
    let mut last_probe: Option<DateTime<Utc>> = None;
    let deadline = tokio::time::Instant::now() + SLEW_TIMEOUT;
    loop {
        tokio::time::sleep(WATCH_INTERVAL).await;
        if state.plate_solve_trigger.lock().unwrap().generation() != generation {
            return; // A later slew owns the trigger now.
        }
        let probe = {
            let devices = state.devices.read().unwrap();
            devices
                .0
                .get(&DeviceKind::Mount)
                .filter(|health| health.connected)
                .map(|health| (health.last_checked_at, health.snapshot.clone()))
        };
        let reason = match probe {
            None => Some("the mount disconnected"),
            Some(_) if tokio::time::Instant::now() >= deadline => {
                Some("it never reported arriving")
            }
            _ => None,
        };
        if let Some(reason) = reason {
            tracing::warn!(
                "No post-slew plate solve for RA {:.4}h Dec {:+.4}°: {reason}",
                target.ra,
                target.dec
            );
            state
                .plate_solve_trigger
                .lock()
                .unwrap()
                .slew_abandoned(generation);
            return;
        }
        let Some((Some(probed_at), Some(snapshot))) = probe else {
            continue;
        };
        if last_probe == Some(probed_at) {
            continue;
        }
        last_probe = Some(probed_at);
        let (Some(ra), Some(dec)) = (snapshot["ra"].as_f64(), snapshot["dec"].as_f64()) else {
            continue;
        };
        if let Some(arrived_at) = settle.observe(probed_at, EqCoord { ra, dec }) {
            if state
                .plate_solve_trigger
                .lock()
                .unwrap()
                .slew_settled(generation, arrived_at)
            {
                tracing::info!(
                    "Mount reported arriving at RA {:.4}h Dec {:+.4}°; the next light frame \
                     begun after {arrived_at} will be plate solved",
                    target.ra,
                    target.dec
                );
            }
            return;
        }
    }
}

/// The archive committed frame `frame_id`. `exposure_started_at` is the
/// RFC 3339 exposure start the capture reported.
pub fn frame_archived(
    state: &AppState,
    frame_id: &str,
    capture_type: &str,
    exposure_started_at: &str,
) {
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
    let Some(job) = state
        .plate_solve_trigger
        .lock()
        .unwrap()
        .frame_archived(frame, Utc::now())
    else {
        return;
    };

    // A slew target is the best hint there is. At startup, the mount's own
    // report of where it points is the next best; without a mount, blind.
    let hint = job.target.or_else(|| mount_position(state));
    let request = SolveFrameRequest {
        frame_id: job.frame_id.clone(),
        hint: hint.map(|coord| HintRequest {
            ra_hours: coord.ra,
            dec_deg: coord.dec,
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
        let solved = match solve_and_record(&state, request, job.reason.as_str()).await {
            Ok(response) => response.success,
            Err(error) => {
                tracing::warn!(
                    "Automatic plate solve of {} could not run: {}",
                    job.frame_id,
                    error.message
                );
                false
            }
        };
        state
            .plate_solve_trigger
            .lock()
            .unwrap()
            .job_finished(&job, solved);
    });
}

/// Where the mount last said it points, from the passive probe. A reading off
/// the sphere is no position at all: solving blind beats a hint the request
/// validation would refuse on every retry.
fn mount_position(state: &AppState) -> Option<EqCoord> {
    let snapshot = state.devices.read().unwrap().snapshot(DeviceKind::Mount)?;
    let position = EqCoord {
        ra: snapshot["ra"].as_f64()?,
        dec: snapshot["dec"].as_f64()?,
    };
    ((0.0..24.0).contains(&position.ra) && (-90.0..=90.0).contains(&position.dec))
        .then_some(position)
}

/// The real archive, database and trigger, with a stand-in ASTAP and mount
/// probes published the way `hardware::health` publishes them.
#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::api::plate_solve::test_support::{archive_with_frame, close, scratch};
    use crate::config::PlateSolveConfig;
    use crate::processing::astap::test_support::{fake_astap, SOLVED_INI};
    use crate::processing::astap::AstapSolver;
    use crate::processing::plate_solver::PlateSolver;

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

    fn mount_probe(ra: f64, dec: f64) -> Option<serde_json::Value> {
        Some(serde_json::json!({"connected": true, "name": "test mount", "ra": ra, "dec": dec}))
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
    async fn after_a_slew_the_first_frame_begun_once_the_mount_arrives_is_solved_at_the_target() {
        let directory = scratch();
        let (state, frame_id) = armed(&directory).await;
        state.devices.write().unwrap().connected(
            DeviceKind::Mount,
            "mount-1".into(),
            mount_probe(5.588, -5.39),
        );

        let target = EqCoord {
            ra: 8.855,
            dec: 11.8,
        };
        slew_commanded(&state, target);

        // Archived while the mount is still on its way: not solved.
        frame_archived(&state, &frame_id, "light", &Utc::now().to_rfc3339());
        let (rows,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM plate_solve_events")
            .fetch_one(state.local_store.pool())
            .await
            .unwrap();
        assert_eq!(rows, 0);

        // The passive probe sees it moving, then holding at the target.
        let publish = |ra, dec| {
            state.devices.write().unwrap().checked(
                DeviceKind::Mount,
                "mount-1",
                true,
                mount_probe(ra, dec),
            )
        };
        tokio::time::sleep(Duration::from_millis(600)).await;
        publish(7.0, 3.0);
        let mut arrived = None;
        for _ in 0..20 {
            tokio::time::sleep(Duration::from_millis(600)).await;
            publish(target.ra, target.dec);
            arrived = state
                .plate_solve_trigger
                .lock()
                .unwrap()
                .status()
                .pending
                .and_then(|rule| rule.not_before);
            if arrived.is_some() {
                break;
            }
        }
        let arrived = arrived.expect("the watcher saw the mount arrive");

        let begun = arrived + chrono::Duration::seconds(1);
        frame_archived(&state, &frame_id, "light", &begun.to_rfc3339());
        let (trigger, mode, success, hint_ra_deg) = recorded(&state, &frame_id).await;
        assert_eq!(
            (trigger.as_str(), mode.as_str(), success),
            ("post_slew", "hinted", 1)
        );
        assert!((hint_ra_deg.unwrap() - target.ra * 15.0).abs() < 1e-9);
        assert!(state
            .plate_solve_trigger
            .lock()
            .unwrap()
            .status()
            .pending
            .is_none());

        close(state, directory).await;
    }
}

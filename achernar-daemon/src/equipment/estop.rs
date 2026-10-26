//! Emergency stop (spec §2.9 "Emergency stop", D19, v6 §7.3). Owner:
//! WP1A-2.
//!
//! `POST /api/v1/equipment/estop {abort_exposure?: bool}` (relay action
//! `equipment.estop`, which bypasses the relay's command slots):
//!
//! 1. raises the `estop` resume block first, so the gate refuses new
//!    automatic commands while the stops fan out;
//! 2. starts a new abort epoch for every attached kind, so a motion command
//!    authorised before the e-stop never starts, even one already waiting
//!    in a worker's queue (D19, `gate::AbortEpochs`);
//! 3. runs every attached handle's out-of-band abort path in parallel,
//!    including degraded and lost handles. Each driver's `AbortPath` does
//!    its kind's stop: mount AbortSlew then MoveAxis 0 on both axes; focuser
//!    and rotator Halt; cover HaltCover; dome AbortSlew; guider
//!    `stop_capture`. The mount goes through
//!    [`estop_mount`](super::mount::estop_mount) (O11 j): its abort path,
//!    then MoveAxis 0 and, with `[equipment] estop_stops_tracking`
//!    (default), tracking off as e-stop-rank steps that outrank any running
//!    operation. When the worker refuses tracking off (a lost mount takes no
//!    commands) or has not confirmed it within half the budget (a worker
//!    blocked in a driver call), tracking off goes through the mount's
//!    out-of-band abort path instead, best effort; a failure there is
//!    audited and fails the mount's entry;
//! 4. aborts the camera exposure only when `abort_exposure` is true;
//! 5. returns per-device results within the 2 s budget. It never waits for
//!    motion to stop; confirmation arrives as events.
//!
//! Only a retired (disconnected) handle fails, with `NotConnected`. E-stop is
//! not a park. It skips ownership, the arbiter and the safety checks, and is
//! always audited.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;
use uuid::Uuid;

use super::commands::{GatedCommand, GatedOp, MountCmd};
use super::events::{EquipmentEvent, EstopDeviceResult, EventKind};
use super::gate::CommandGate;
use super::handle::{AbortTicket, MountHandle};
use super::hub::{AnyHandle, EquipmentHub};
use super::mount::estop_mount_reporting;
use super::resume::BlockRequest;
use super::snapshot::{ResumeBlock, ResumeBlockKind};
use super::types::{
    AutonomyMode, CommandId, CommandSource, DeviceError, DeviceKind, DeviceResult, OpOutcome,
    Priority,
};

/// Real-time slice between result checks.
const POLL_SLICE: Duration = Duration::from_millis(5);

/// An e-stop request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EstopRequest {
    #[serde(default)]
    pub abort_exposure: bool,
    pub source: CommandSource,
}

/// What the e-stop did, per device, within the budget.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EstopReport {
    pub command_id: CommandId,
    pub results: Vec<EstopDeviceResult>,
    /// The camera exposure abort, when requested.
    pub exposure: Option<EstopDeviceResult>,
    pub block: ResumeBlock,
    pub elapsed_ms: u64,
}

impl EstopReport {
    /// True when every device (and the exposure, if requested) confirmed
    /// its abort within the budget.
    pub fn all_ok(&self) -> bool {
        self.results
            .iter()
            .chain(self.exposure.iter())
            .all(|r| r.ok)
    }
}

struct Pending {
    kind: DeviceKind,
    connection_id: Option<Uuid>,
    started: Instant,
    ticket: PendingResult,
    /// The mount's tracking-off step, when the e-stop sends one.
    tracking: Option<oneshot::Receiver<DeviceResult<()>>>,
    abort_result: Option<DeviceResult<()>>,
    tracking_result: Option<DeviceResult<()>>,
    finished_at: Option<Instant>,
}

impl Pending {
    fn new(kind: DeviceKind, connection_id: Option<Uuid>, ticket: PendingResult) -> Self {
        Self {
            kind,
            connection_id,
            started: Instant::now(),
            ticket,
            tracking: None,
            abort_result: None,
            tracking_result: None,
            finished_at: None,
        }
    }

    /// Collect whatever has arrived; true once every part has.
    fn poll(&mut self) -> bool {
        if self.abort_result.is_none() {
            self.abort_result = self.ticket.try_result();
        }
        if self.tracking_result.is_none() {
            if let Some(rx) = self.tracking.as_mut() {
                self.tracking_result = match rx.try_recv() {
                    Ok(result) => Some(result),
                    Err(oneshot::error::TryRecvError::Empty) => None,
                    Err(oneshot::error::TryRecvError::Closed) => Some(Err(
                        DeviceError::outcome_unknown("tracking off stopped without a result"),
                    )),
                };
            }
        }
        let done = self.abort_result.is_some()
            && (self.tracking.is_none() || self.tracking_result.is_some());
        if done && self.finished_at.is_none() {
            self.finished_at = Some(Instant::now());
        }
        done
    }

    /// The device's entry: its abort path and, for the mount, tracking off.
    /// A part still missing when the budget ran out is a failure.
    fn result(&self, budget: Duration) -> EstopDeviceResult {
        let latency = self.finished_at.map(|at| {
            u64::try_from(at.duration_since(self.started).as_millis()).unwrap_or(u64::MAX)
        });
        let budget_ms = budget.as_millis();
        let abort = self.abort_result.clone().unwrap_or_else(|| {
            Err(DeviceError::timeout(format!(
                "the abort path was still running after {budget_ms} ms"
            )))
        });
        let tracking = self.tracking.as_ref().map(|_| {
            self.tracking_result.clone().unwrap_or_else(|| {
                Err(DeviceError::timeout(format!(
                    "tracking off was not confirmed within {budget_ms} ms (the mount worker \
                     may be blocked in a driver call)"
                )))
            })
        });
        let error = match (abort, tracking) {
            (Ok(()), None | Some(Ok(()))) => None,
            (Err(err), None | Some(Ok(()))) => Some(err),
            (Ok(()), Some(Err(err))) => Some(DeviceError {
                message: format!("tracking off: {}", err.message),
                ..err
            }),
            (Err(err), Some(Err(tracking))) => Some(DeviceError {
                message: format!("{}; tracking off: {}", err.message, tracking.message),
                ..err
            }),
        };
        EstopDeviceResult {
            kind: self.kind,
            connection_id: self.connection_id,
            ok: error.is_none(),
            error,
            latency_ms: latency,
        }
    }
}

enum PendingResult {
    Abort(AbortTicket),
    /// A result delivered by a task: the exposure abort, or the mount's
    /// abort path inside `estop_mount`.
    Channel(oneshot::Receiver<DeviceResult<()>>),
}

impl PendingResult {
    fn try_result(&mut self) -> Option<DeviceResult<()>> {
        match self {
            PendingResult::Abort(ticket) => ticket.try_outcome(),
            PendingResult::Channel(rx) => match rx.try_recv() {
                Ok(result) => Some(result),
                Err(oneshot::error::TryRecvError::Empty) => None,
                Err(oneshot::error::TryRecvError::Closed) => Some(Err(
                    DeviceError::outcome_unknown("the abort stopped without a result"),
                )),
            },
        }
    }
}

/// Tracking off through the mount's out-of-band abort path (O11 j), on a
/// blocking thread and bounded by `budget`.
async fn tracking_off_out_of_band(mount: &MountHandle, budget: Duration) -> DeviceResult<()> {
    let Some(path) = mount.abort_path() else {
        return Err(DeviceError::unsupported(
            "the mount has no out-of-band abort path to stop tracking",
        ));
    };
    let task = tokio::task::spawn_blocking(move || path.tracking_off());
    match tokio::time::timeout(budget, task).await {
        Ok(Ok(result)) => result,
        Ok(Err(_)) => Err(DeviceError::driver("the out-of-band tracking off panicked")),
        Err(_) => Err(DeviceError::timeout(format!(
            "the out-of-band tracking off did not answer within {} ms",
            budget.as_millis()
        ))),
    }
}

/// The mount's e-stop (O11 j): `estop_mount`, with its abort path reported on
/// `abort_tx` as soon as it is known, and tracking off on `tracking_tx`.
/// Tracking off that the worker refused or did not confirm within
/// `fallback_after` is sent through the abort path; that attempt is audited
/// as a step of the e-stop, and its failure fails the mount's entry.
async fn mount_estop(
    mount: MountHandle,
    stops_tracking: bool,
    gate: Arc<CommandGate>,
    estop: GatedCommand,
    fallback_after: Duration,
    abort_tx: oneshot::Sender<DeviceResult<()>>,
    tracking_tx: Option<oneshot::Sender<DeviceResult<()>>>,
) {
    // `estop_mount` runs to its end on its own task, so the abort path's
    // result is still delivered when tracking off falls back below.
    let (report_tx, report_rx) = oneshot::channel();
    let task_mount = mount.clone();
    tokio::spawn(async move {
        let report = estop_mount_reporting(&task_mount, stops_tracking, Some(abort_tx)).await;
        let _ = report_tx.send(report);
    });
    let Some(tracking_tx) = tracking_tx else {
        return;
    };
    let in_band = match tokio::time::timeout(fallback_after, report_rx).await {
        Ok(Ok(report)) => report
            .tracking_off()
            .cloned()
            .unwrap_or_else(|| Err(DeviceError::outcome_unknown("tracking off was not sent"))),
        Ok(Err(_)) => Err(DeviceError::outcome_unknown(
            "the mount e-stop stopped without a report",
        )),
        Err(_) => Err(DeviceError::timeout(format!(
            "the mount worker did not confirm tracking off within {} ms",
            fallback_after.as_millis()
        ))),
    };
    let result = match in_band {
        Ok(()) => Ok(()),
        Err(worker_err) => {
            let step = GatedCommand::new(
                GatedOp::Mount(MountCmd::SetTracking(false)),
                CommandSource::Safety,
                AutonomyMode::Manual,
                Priority::EStop,
            )
            .with_params(serde_json::json!({
                "estop": estop.command_id,
                "path": "abort_path",
                "worker_error": worker_err.to_string(),
            }));
            gate.record_executed(
                &step,
                vec![
                    format!("estop={}", estop.command_id),
                    "via=abort_path".into(),
                ],
            );
            let out_of_band = tracking_off_out_of_band(&mount, fallback_after).await;
            match out_of_band {
                Ok(()) => {
                    gate.record_outcome(&step, &OpOutcome::Succeeded);
                    Ok(())
                }
                Err(path_err) => {
                    let err = DeviceError {
                        message: format!("{}; abort path: {}", worker_err.message, path_err),
                        ..worker_err
                    };
                    gate.record_outcome(&step, &OpOutcome::Failed(err.clone()));
                    tracing::warn!("e-stop could not turn the mount's tracking off: {err}");
                    Err(err)
                }
            }
        }
    };
    let _ = tracking_tx.send(result);
}

/// Run the emergency stop. Must run inside a tokio runtime.
pub async fn run_estop(hub: &EquipmentHub, request: EstopRequest) -> EstopReport {
    let started = Instant::now();
    let budget = hub.config().estop_budget;
    let cmd = GatedCommand::new(
        GatedOp::EStop {
            abort_exposure: request.abort_exposure,
        },
        request.source.clone(),
        AutonomyMode::Manual,
        Priority::EStop,
    );
    // E-stop bypasses the checks: this audits the proposed and validated
    // stages and always succeeds.
    let _ = hub.gate().authorize(&cmd);
    // The block goes up before anything is stopped, so the gate refuses new
    // automatic commands for the whole fan-out.
    let block = hub.raise_block(
        BlockRequest::new(ResumeBlockKind::Estop, "emergency stop")
            .with_command(Some(cmd.command_id)),
    );

    let epochs = hub.gate().abort_epochs();
    let mut pending = Vec::new();
    for handle in hub.attached() {
        // Before the cancel is queued: every permit issued so far is now
        // older than the abort, and its worker refuses it (D19).
        epochs.bump(handle.kind());
        let entry = match &handle {
            AnyHandle::Mount(mount) if !mount.is_retired() => {
                // `estop_mount` (O11 j): the abort path plus e-stop-rank
                // stops that outrank any running operation, so they are never
                // refused as a conflict; tracking off is not a motion command,
                // so the abort leaves it queued behind the cancel.
                let stops_tracking = hub.config().estop_stops_tracking;
                let (abort_tx, abort_rx) = oneshot::channel();
                let (tracking_tx, tracking_rx) = if stops_tracking {
                    let (tx, rx) = oneshot::channel();
                    (Some(tx), Some(rx))
                } else {
                    (None, None)
                };
                tokio::spawn(mount_estop(
                    mount.clone(),
                    stops_tracking,
                    Arc::clone(hub.gate()),
                    cmd.clone(),
                    budget / 2,
                    abort_tx,
                    tracking_tx,
                ));
                let mut entry = Pending::new(
                    handle.kind(),
                    Some(handle.connection_id()),
                    PendingResult::Channel(abort_rx),
                );
                entry.tracking = tracking_rx;
                entry
            }
            _ => Pending::new(
                handle.kind(),
                Some(handle.connection_id()),
                PendingResult::Abort(handle.abort()),
            ),
        };
        pending.push(entry);
    }

    let mut exposure = None;
    if request.abort_exposure {
        if let Some(camera) = hub.camera() {
            let (tx, rx) = oneshot::channel();
            tokio::task::spawn_blocking(move || {
                let _ = tx.send(camera.abort_exposure());
            });
            exposure = Some(Pending::new(
                DeviceKind::Camera,
                None,
                PendingResult::Channel(rx),
            ));
        }
    }

    // Collect within the budget.
    let deadline = started + budget;
    loop {
        let mut done = true;
        for p in pending.iter_mut().chain(exposure.iter_mut()) {
            done &= p.poll();
        }
        if done || Instant::now() >= deadline {
            break;
        }
        tokio::time::sleep(POLL_SLICE).await;
    }
    let results: Vec<EstopDeviceResult> = pending.iter().map(|p| p.result(budget)).collect();
    let exposure = exposure.map(|p| p.result(budget));

    let mut evented = results.clone();
    evented.extend(exposure.clone());
    hub.emit(
        EquipmentEvent::new(
            EventKind::Estop {
                results: evented.clone(),
            },
            hub.clock().as_ref(),
        )
        .with_command(cmd.command_id),
    );
    let failed = evented.iter().filter(|r| !r.ok).count();
    hub.gate().record_executed(
        &cmd,
        evented
            .iter()
            .map(|r| format!("{:?}: {}", r.kind, if r.ok { "ok" } else { "failed" }))
            .collect(),
    );
    let outcome = if failed == 0 {
        OpOutcome::Succeeded
    } else {
        OpOutcome::Failed(DeviceError::driver(format!(
            "{failed} device(s) did not confirm the abort within the budget"
        )))
    };
    hub.gate().record_outcome(&cmd, &outcome);
    EstopReport {
        command_id: cmd.command_id,
        results,
        exposure,
        block,
        elapsed_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::events::{AuditStage, GroundTruthRecord};
    use crate::equipment::hub::HubConfig;
    use crate::equipment::snapshot::{
        DomeSnapshot, ExposureRef, FocuserSnapshot, GuiderSnapshot, MountSnapshot,
    };
    use crate::equipment::testing::fake_handles::{
        fake_dome, fake_focuser, fake_guider, fake_mount, FakeCameraActivity, FakeResponse,
        RecordingSink,
    };
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::types::{ErrorClass, LinkState};
    use std::sync::Arc;

    fn hub(budget: Duration) -> (EquipmentHub, Arc<RecordingSink>) {
        let sink = RecordingSink::new();
        let hub = EquipmentHub::new(
            HubConfig {
                estop_budget: budget,
                ..HubConfig::default()
            },
            Arc::new(ManualClock::new()),
            sink.clone(),
        );
        (hub, sink)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn estop_fans_out_to_every_handle_including_a_lost_one_within_2_s() {
        let (hub, sink) = hub(Duration::from_secs(2));
        let (mount, mount_fake) = fake_mount(MountSnapshot::default());
        let (focuser, focuser_fake) = fake_focuser(FocuserSnapshot::default());
        let (guider, guider_fake) = fake_guider(GuiderSnapshot::default());
        let (dome, dome_fake) = fake_dome(DomeSnapshot::default());
        hub.attach(mount);
        hub.attach(focuser);
        hub.attach(guider);
        hub.attach(dome);
        focuser_fake.set_link(LinkState::Lost);
        guider_fake.set_link(LinkState::Degraded);
        // Slow abort paths run in parallel, not one after another.
        for path in [
            mount_fake.abort_path(),
            focuser_fake.abort_path(),
            guider_fake.abort_path(),
            dome_fake.abort_path(),
        ] {
            path.set_delay(Some(Duration::from_millis(300)));
        }
        let camera = FakeCameraActivity::new();
        camera.start_exposure(ExposureRef::default());
        hub.set_camera(Some(camera.clone()));
        let report = run_estop(
            &hub,
            EstopRequest {
                abort_exposure: true,
                source: CommandSource::Operator { overriding: false },
            },
        )
        .await;
        assert!(report.elapsed_ms < 2000, "{} ms", report.elapsed_ms);
        assert!(
            report.elapsed_ms < 1000,
            "parallel, not serial: {} ms",
            report.elapsed_ms
        );
        assert!(report.all_ok(), "{report:?}");
        assert_eq!(report.results.len(), 4);
        for fake_calls in [
            mount_fake.abort_path().calls(),
            focuser_fake.abort_path().calls(),
            guider_fake.abort_path().calls(),
            dome_fake.abort_path().calls(),
        ] {
            assert_eq!(fake_calls, 1);
        }
        assert_eq!(camera.aborts(), 1);
        assert_eq!(report.block.kind, ResumeBlockKind::Estop);
        assert_eq!(hub.resume().active()[0].kind, ResumeBlockKind::Estop);
        // The mount also got tracking off, inside the reported result.
        assert!(mount_fake
            .commands()
            .contains(&MountCmd::SetTracking(false)));
        // Every attached kind starts a new abort epoch (D19).
        let epochs = hub.gate().abort_epochs();
        for kind in [
            DeviceKind::Mount,
            DeviceKind::Focuser,
            DeviceKind::Guider,
            DeviceKind::Dome,
        ] {
            assert_eq!(epochs.epoch(kind), 1, "{kind:?}");
        }
        let names = sink.event_names();
        assert!(names.contains(&"estop") && names.contains(&"resume_blocked"));
        let stages: Vec<AuditStage> = sink
            .records()
            .iter()
            .filter_map(|r| match r {
                GroundTruthRecord::Audit(a) if a.command_id == report.command_id => Some(a.stage),
                _ => None,
            })
            .collect();
        assert_eq!(
            stages,
            vec![
                AuditStage::Proposed,
                AuditStage::Validated,
                AuditStage::Executed,
                AuditStage::Outcome
            ]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_stuck_abort_path_is_reported_within_the_budget_and_retired_handles_fail() {
        let (hub, _sink) = hub(Duration::from_millis(300));
        let (mount, mount_fake) = fake_mount(MountSnapshot::default());
        let (focuser, focuser_fake) = fake_focuser(FocuserSnapshot::default());
        hub.attach(mount);
        hub.attach(focuser);
        mount_fake
            .abort_path()
            .set_delay(Some(Duration::from_secs(3)));
        focuser_fake.retire();
        let started = Instant::now();
        let report = run_estop(
            &hub,
            EstopRequest {
                abort_exposure: false,
                source: CommandSource::Safety,
            },
        )
        .await;
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(!report.all_ok());
        let mount_result = report
            .results
            .iter()
            .find(|r| r.kind == DeviceKind::Mount)
            .unwrap();
        assert_eq!(
            mount_result.error.as_ref().unwrap().class,
            ErrorClass::Timeout
        );
        let focuser_result = report
            .results
            .iter()
            .find(|r| r.kind == DeviceKind::Focuser)
            .unwrap();
        assert_eq!(
            focuser_result.error.as_ref().unwrap().class,
            ErrorClass::NotConnected,
            "only a retired handle refuses"
        );
        assert!(
            report.exposure.is_none(),
            "the exposure is left alone unless asked"
        );
    }

    fn lost_mount_refusing_tracking_off() -> (
        EquipmentHub,
        Arc<RecordingSink>,
        crate::equipment::testing::fake_handles::FakeDevice<MountCmd, MountSnapshot>,
    ) {
        let (hub, sink) = hub(Duration::from_secs(2));
        let (mount, mount_fake) = fake_mount(MountSnapshot::default());
        hub.attach(mount);
        mount_fake.set_link(LinkState::Lost);
        mount_fake.set_responder(|cmd, _snap: &mut MountSnapshot| match cmd {
            MountCmd::SetTracking(false) => {
                FakeResponse::reject(DeviceError::disconnected("the mount is lost"))
            }
            _ => FakeResponse::succeed(),
        });
        (hub, sink, mount_fake)
    }

    fn abort_path_steps(sink: &RecordingSink) -> Vec<(AuditStage, bool)> {
        sink.records()
            .iter()
            .filter_map(|r| match r {
                GroundTruthRecord::Audit(a)
                    if a.annotations.iter().any(|n| n == "via=abort_path")
                        || (a.stage == AuditStage::Outcome
                            && a.params.get("path").and_then(|p| p.as_str())
                                == Some("abort_path")) =>
                {
                    Some((a.stage, a.accepted))
                }
                _ => None,
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_lost_mount_gets_tracking_off_through_its_abort_path() {
        // O11 j: a lost mount's worker refuses the in-band tracking off, so
        // the e-stop sends it through the out-of-band abort path.
        let (hub, sink, mount_fake) = lost_mount_refusing_tracking_off();
        let report = run_estop(
            &hub,
            EstopRequest {
                abort_exposure: false,
                source: CommandSource::Operator { overriding: false },
            },
        )
        .await;
        assert!(report.all_ok(), "{report:?}");
        assert_eq!(mount_fake.abort_path().calls(), 1, "the abort path ran");
        assert_eq!(mount_fake.abort_path().tracking_calls(), 1);
        assert_eq!(report.block.kind, ResumeBlockKind::Estop);
        assert_eq!(
            abort_path_steps(&sink),
            vec![(AuditStage::Executed, true), (AuditStage::Outcome, true)]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_failed_out_of_band_tracking_off_fails_the_entry_and_is_audited() {
        let (hub, sink, mount_fake) = lost_mount_refusing_tracking_off();
        mount_fake
            .abort_path()
            .set_tracking_result(Err(DeviceError::transport("socket closed")));
        let report = run_estop(
            &hub,
            EstopRequest {
                abort_exposure: false,
                source: CommandSource::Operator { overriding: false },
            },
        )
        .await;
        assert!(!report.all_ok());
        let entry = &report.results[0];
        assert_eq!(entry.kind, DeviceKind::Mount);
        let err = entry.error.as_ref().unwrap();
        assert_eq!(err.class, ErrorClass::Disconnected);
        assert!(err.message.starts_with("tracking off:"), "{err}");
        assert!(err.message.contains("socket closed"), "{err}");
        assert_eq!(
            abort_path_steps(&sink),
            vec![(AuditStage::Executed, true), (AuditStage::Outcome, false)]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_blocked_mount_worker_falls_back_to_the_abort_path_within_the_budget() {
        // The worker never answers (blocked in a driver call): the in-band
        // tracking off is not waited for past half the budget.
        let (stuck_hub, _sink) = hub(Duration::from_millis(300));
        let (mount, mount_fake) = fake_mount(MountSnapshot::default());
        stuck_hub.attach(mount);
        mount_fake.set_responder(|_, _snap: &mut MountSnapshot| {
            std::thread::sleep(Duration::from_millis(600));
            FakeResponse::succeed()
        });
        let started = Instant::now();
        let report = run_estop(
            &stuck_hub,
            EstopRequest {
                abort_exposure: false,
                source: CommandSource::Safety,
            },
        )
        .await;
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(report.results[0].ok, "{report:?}");
        assert_eq!(mount_fake.abort_path().tracking_calls(), 1);

        // Neither the worker nor the abort path can stop tracking: reported
        // within the budget, never assumed.
        let (stuck_hub, _sink) = hub(Duration::from_millis(300));
        let (mount, mount_fake) = fake_mount(MountSnapshot::default());
        stuck_hub.attach(mount);
        mount_fake.set_responder(|_, _snap: &mut MountSnapshot| {
            std::thread::sleep(Duration::from_millis(600));
            FakeResponse::hold()
        });
        mount_fake
            .abort_path()
            .set_tracking_result(Err(DeviceError::unsupported("no tracking off")));
        let report = run_estop(
            &stuck_hub,
            EstopRequest {
                abort_exposure: false,
                source: CommandSource::Safety,
            },
        )
        .await;
        let err = report.results[0].error.as_ref().unwrap();
        assert_eq!(err.class, ErrorClass::Timeout);
        assert!(err.message.contains("tracking off"), "{err}");
        mount_fake.complete_pending(OpOutcome::Succeeded);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn estops_stop_tracking_only_when_configured() {
        let sink = RecordingSink::new();
        let hub = EquipmentHub::new(
            HubConfig {
                estop_stops_tracking: false,
                ..HubConfig::default()
            },
            Arc::new(ManualClock::new()),
            sink,
        );
        let (mount, mount_fake) = fake_mount(MountSnapshot::default());
        hub.attach(mount);
        let report = run_estop(
            &hub,
            EstopRequest {
                abort_exposure: false,
                source: CommandSource::Safety,
            },
        )
        .await;
        assert!(report.all_ok(), "{report:?}");
        assert!(mount_fake.commands().is_empty());
    }
}

//! Emergency stop (spec §2.9 "Emergency stop", D19, v6 §7.3). Owner:
//! WP1A-2.
//!
//! `POST /api/v1/equipment/estop {abort_exposure?: bool}` (relay action
//! `equipment.estop`, which bypasses the relay's command slots):
//!
//! 1. runs every attached handle's out-of-band abort path in parallel,
//!    including degraded and lost handles. Each driver's `AbortPath` does
//!    its kind's stop: mount AbortSlew then MoveAxis 0 on both axes; focuser
//!    and rotator Halt; cover HaltCover; dome AbortSlew; guider
//!    `stop_capture`. With `[equipment] estop_stops_tracking` (default) the
//!    mount also gets tracking off, as a step of the e-stop that outranks
//!    any running operation;
//! 2. aborts the camera exposure only when `abort_exposure` is true;
//! 3. raises the `estop` resume block;
//! 4. returns per-device results within the 2 s budget. It never waits for
//!    motion to stop; confirmation arrives as events.
//!
//! Only a retired (disconnected) handle fails, with `NotConnected`. E-stop is
//! not a park. It skips ownership, the arbiter and the safety checks, and is
//! always audited.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;
use uuid::Uuid;

use super::commands::{GatedCommand, GatedOp, MountCmd};
use super::events::{EquipmentEvent, EstopDeviceResult, EventKind};
use super::handle::{AbortTicket, Permit};
use super::hub::{AnyHandle, EquipmentHub};
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
}

enum PendingResult {
    Abort(AbortTicket),
    Exposure(oneshot::Receiver<DeviceResult<()>>),
}

impl PendingResult {
    fn try_result(&mut self) -> Option<DeviceResult<()>> {
        match self {
            PendingResult::Abort(ticket) => ticket.try_outcome(),
            PendingResult::Exposure(rx) => match rx.try_recv() {
                Ok(result) => Some(result),
                Err(oneshot::error::TryRecvError::Empty) => None,
                Err(oneshot::error::TryRecvError::Closed) => Some(Err(
                    DeviceError::outcome_unknown("the exposure abort stopped without a result"),
                )),
            },
        }
    }
}

fn result_of(pending: &Pending, result: DeviceResult<()>) -> EstopDeviceResult {
    let latency = u64::try_from(pending.started.elapsed().as_millis()).unwrap_or(u64::MAX);
    match result {
        Ok(()) => EstopDeviceResult {
            kind: pending.kind,
            connection_id: pending.connection_id,
            ok: true,
            error: None,
            latency_ms: Some(latency),
        },
        Err(err) => EstopDeviceResult {
            kind: pending.kind,
            connection_id: pending.connection_id,
            ok: false,
            error: Some(err),
            latency_ms: Some(latency),
        },
    }
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

    let mut pending = Vec::new();
    for handle in hub.attached() {
        let at = Instant::now();
        let ticket = handle.abort();
        if let AnyHandle::Mount(mount) = &handle {
            if hub.config().estop_stops_tracking && !mount.is_retired() {
                // A safety-source step outranks any running operation, so it
                // is never refused as a conflict; it is queued behind the
                // cancel and never delays the response.
                let mount = mount.clone();
                let permit =
                    Permit::unmanaged(cmd.command_id, Priority::EStop, CommandSource::Safety);
                tokio::spawn(async move {
                    if let Err(err) = mount.send(MountCmd::SetTracking(false), permit).await {
                        tracing::warn!("e-stop could not turn tracking off: {err}");
                    }
                });
            }
        }
        pending.push(Pending {
            kind: handle.kind(),
            connection_id: Some(handle.connection_id()),
            started: at,
            ticket: PendingResult::Abort(ticket),
        });
    }

    let mut exposure = None;
    if request.abort_exposure {
        if let Some(camera) = hub.camera() {
            let (tx, rx) = oneshot::channel();
            tokio::task::spawn_blocking(move || {
                let _ = tx.send(camera.abort_exposure());
            });
            exposure = Some(Pending {
                kind: DeviceKind::Camera,
                connection_id: None,
                started: Instant::now(),
                ticket: PendingResult::Exposure(rx),
            });
        }
    }

    // Collect within the budget.
    let deadline = started + budget;
    let mut results: Vec<Option<EstopDeviceResult>> = vec![None; pending.len()];
    let mut exposure_result: Option<EstopDeviceResult> = None;
    loop {
        for (slot, p) in results.iter_mut().zip(pending.iter_mut()) {
            if slot.is_none() {
                if let Some(result) = p.ticket.try_result() {
                    *slot = Some(result_of(p, result));
                }
            }
        }
        if let Some(p) = exposure.as_mut() {
            if exposure_result.is_none() {
                if let Some(result) = p.ticket.try_result() {
                    exposure_result = Some(result_of(p, result));
                }
            }
        }
        let done = results.iter().all(Option::is_some)
            && (exposure.is_none() || exposure_result.is_some());
        if done || Instant::now() >= deadline {
            break;
        }
        tokio::time::sleep(POLL_SLICE).await;
    }
    let timed_out = |p: &Pending| EstopDeviceResult {
        kind: p.kind,
        connection_id: p.connection_id,
        ok: false,
        error: Some(DeviceError::timeout(format!(
            "the abort path was still running after {} ms",
            budget.as_millis()
        ))),
        latency_ms: None,
    };
    let results: Vec<EstopDeviceResult> = results
        .into_iter()
        .zip(pending.iter())
        .map(|(result, p)| result.unwrap_or_else(|| timed_out(p)))
        .collect();
    let exposure = exposure.map(|p| exposure_result.unwrap_or_else(|| timed_out(&p)));

    let block = hub.raise_block(
        BlockRequest::new(ResumeBlockKind::Estop, "emergency stop")
            .with_command(Some(cmd.command_id)),
    );
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
        fake_dome, fake_focuser, fake_guider, fake_mount, FakeCameraActivity, RecordingSink,
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
        // The mount also gets tracking off.
        let deadline = Instant::now() + Duration::from_secs(5);
        while !mount_fake
            .commands()
            .contains(&MountCmd::SetTracking(false))
        {
            assert!(Instant::now() < deadline, "tracking was never turned off");
            tokio::time::sleep(Duration::from_millis(5)).await;
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
}

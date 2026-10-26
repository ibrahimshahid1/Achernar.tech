//! Pre-motion coordination (spec §2.9, D20). Owner: WP1A-2.
//!
//! One guiding authority per mount: before an Achernar slew, park, home,
//! flip or center, if the guider reports `Guiding`, `Calibrating` or
//! `LostLock`, the gate's executor
//!
//! 1. sends `GuiderCmd::Stop` (PHD2 `stop_capture`) through the gate, with
//!    the motion command's source and priority. When the motion's owner
//!    (the command itself, or the compound operation a step belongs to,
//!    such as a flip) already claims the guider, the stop runs as a child
//!    of that claim instead of competing with it;
//! 2. waits up to 10 s (by the injected clock) for the guider to report
//!    `Stopped`;
//! 3. records `was_guiding` on the operation, and whether the stop was
//!    actually sent.
//!
//! A guider that will not stop fails the motion with `guider_stop_failed`,
//! unless the source is safety: a safety park must never wait on PHD2, so it
//! proceeds and the failure is kept as a warning. Tier 1 leaves restarting
//! guiding to the operator; WP2-8 adds the deterministic restore.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::commands::{Domain, GatedCommand, GatedOp, GuiderCmd, MountCmd};
use super::gate::{CommandGate, StepOwner};
use super::handle::GuiderHandle;
use super::snapshot::GuiderAppState;
use super::traits::Clock;
use super::types::{CommandId, CommandSource, DeviceError, DeviceKind, OpOutcome};

/// How long the stop waits for `Stopped` (§2.9).
pub const STOP_GUIDING_TIMEOUT: Duration = Duration::from_secs(10);

/// Real-time slice between state checks while waiting.
const WAIT_SLICE: Duration = Duration::from_millis(20);

/// Motion that needs the guider stopped first.
pub fn needs_guider_stop(op: &GatedOp) -> bool {
    matches!(
        op,
        GatedOp::Mount(
            MountCmd::Slew { .. } | MountCmd::SlewAltAz(_) | MountCmd::Park | MountCmd::FindHome
        ) | GatedOp::Flip
            | GatedOp::Center { .. }
    )
}

/// Guider states in which PHD2 owns the mount's motion.
pub fn guider_active(state: GuiderAppState) -> bool {
    matches!(
        state,
        GuiderAppState::Guiding | GuiderAppState::Calibrating | GuiderAppState::LostLock
    )
}

/// What the pre-motion stop did, recorded on the operation.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct PreMotion {
    /// The guider was guiding, calibrating or had lost lock before the
    /// motion.
    pub was_guiding: bool,
    /// The guider state seen before the motion.
    pub guider_state: Option<GuiderAppState>,
    /// True only when the stop reached the guider.
    #[serde(default)]
    pub stop_sent: bool,
    /// The stop command, only when it was sent.
    pub stop_command_id: Option<CommandId>,
    pub waited_ms: u64,
    /// A stop that failed but did not block a safety source, or an unknown
    /// guider state.
    pub warning: Option<String>,
}

/// Why the stop failed, and whether it reached the guider at all.
#[derive(Debug, Clone, PartialEq)]
struct StopFailure {
    reason: String,
    sent: bool,
}

impl StopFailure {
    fn before_send(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
            sent: false,
        }
    }

    fn after_send(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
            sent: true,
        }
    }
}

/// The guider would not stop (`guider_stop_failed`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoordError {
    pub reason: String,
}

impl CoordError {
    pub fn code(&self) -> &'static str {
        "guider_stop_failed"
    }

    pub fn http_status(&self) -> u16 {
        409
    }

    pub fn to_device_error(&self) -> DeviceError {
        DeviceError::invalid_operation(format!("guider_stop_failed: {}", self.reason))
    }
}

impl std::fmt::Display for CoordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the guider did not stop before the motion: {}",
            self.reason
        )
    }
}

impl std::error::Error for CoordError {}

/// Stop guiding before `motion` (D20). `owner` is the permit holder the
/// motion runs under (the motion itself, or the compound operation it is a
/// step of); when it claims the guider the stop is authorised as its child.
/// Returns what was done, or the failure that must fail the motion.
pub async fn stop_guiding_before_motion(
    gate: &CommandGate,
    guider: Option<&GuiderHandle>,
    motion: &GatedCommand,
    owner: Option<&StepOwner>,
    clock: &dyn Clock,
    timeout: Duration,
) -> Result<PreMotion, CoordError> {
    if !needs_guider_stop(&motion.op) {
        return Ok(PreMotion::default());
    }
    let Some(guider) = guider.filter(|g| !g.is_retired()) else {
        return Ok(PreMotion::default());
    };
    let state = guider.snapshot().app_state.value;
    match state {
        Some(state) if guider_active(state) => {}
        Some(GuiderAppState::Unknown) | None => {
            return Ok(PreMotion {
                guider_state: state,
                warning: Some("guider_state_unknown".into()),
                ..PreMotion::default()
            })
        }
        Some(other) => {
            return Ok(PreMotion {
                guider_state: Some(other),
                ..PreMotion::default()
            })
        }
    }

    let stop = GatedCommand::new(
        GatedOp::Guider(GuiderCmd::Stop),
        motion.source.clone(),
        motion.mode,
        motion.priority,
    )
    .with_params(serde_json::json!({
        "op": "guider",
        "args": {"cmd": "stop"},
        "reason": "pre_motion_stop",
        "motion_command_id": motion.command_id,
    }));
    let started = clock.now_mono_ns();
    let result = run_stop(gate, guider, &stop, owner, clock, timeout).await;
    let waited_ms = clock.now_mono_ns().saturating_sub(started) / 1_000_000;
    match result {
        Ok(()) => Ok(PreMotion {
            was_guiding: true,
            guider_state: state,
            stop_sent: true,
            stop_command_id: Some(stop.command_id),
            waited_ms,
            warning: None,
        }),
        Err(failure) if motion.source == CommandSource::Safety => Ok(PreMotion {
            was_guiding: true,
            guider_state: state,
            stop_sent: failure.sent,
            stop_command_id: failure.sent.then_some(stop.command_id),
            waited_ms,
            warning: Some(failure.reason),
        }),
        Err(failure) => Err(CoordError {
            reason: failure.reason,
        }),
    }
}

async fn run_stop(
    gate: &CommandGate,
    guider: &GuiderHandle,
    stop: &GatedCommand,
    owner: Option<&StepOwner>,
    clock: &dyn Clock,
    timeout: Duration,
) -> Result<(), StopFailure> {
    let guider_domain = Domain::Device(DeviceKind::Guider);
    let permit = match owner.filter(|owner| owner.holds(guider_domain)) {
        // A flip (or a step of one) already owns the guider: stopping it is
        // part of the same operation, not a competing command.
        Some(owner) => gate.child_permit_of(owner, stop),
        None => gate.authorize(stop),
    }
    .map_err(|err| StopFailure::before_send(err.to_string()))?;
    let ticket = match guider.send(GuiderCmd::Stop, permit).await {
        Ok(ticket) => ticket,
        Err(err) => {
            gate.record_outcome(stop, &OpOutcome::Failed(err.clone()));
            return Err(StopFailure::before_send(format!(
                "the guider refused stop: {err}"
            )));
        }
    };
    gate.record_executed(stop, Vec::new());
    let mut done = ticket.done;
    let mut finished: Option<OpOutcome> = None;
    let mut snapshots = guider.watch_snapshot();
    let deadline = clock
        .now_mono_ns()
        .saturating_add(super::clock::duration_ns(timeout));
    loop {
        if finished.is_none() {
            match done.try_recv() {
                Ok(outcome) => {
                    if !outcome.is_success() {
                        gate.record_outcome(stop, &outcome);
                        return Err(StopFailure::after_send(format!(
                            "stop ended {}",
                            outcome.label()
                        )));
                    }
                    finished = Some(outcome);
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {}
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    let outcome = OpOutcome::Unknown(DeviceError::outcome_unknown(
                        "the guider dropped the stop command",
                    ));
                    gate.record_outcome(stop, &outcome);
                    return Err(StopFailure::after_send(
                        "the guider dropped the stop command",
                    ));
                }
            }
        }
        let state = snapshots.borrow_and_update().app_state.value;
        if state == Some(GuiderAppState::Stopped) {
            gate.record_outcome(stop, &finished.unwrap_or(OpOutcome::Succeeded));
            return Ok(());
        }
        if clock.now_mono_ns() >= deadline {
            let outcome = OpOutcome::TimedOut;
            gate.record_outcome(stop, &outcome);
            return Err(StopFailure::after_send(format!(
                "the guider did not report Stopped within {:.0} s (state {:?})",
                timeout.as_secs_f64(),
                state
            )));
        }
        tokio::select! {
            changed = snapshots.changed() => {
                if changed.is_err() {
                    // The guider's worker went away; keep checking the clock.
                    tokio::time::sleep(WAIT_SLICE).await;
                }
            }
            _ = tokio::time::sleep(WAIT_SLICE) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::commands::GateError;
    use crate::equipment::gate::tests::gate_with_ctx;
    use crate::equipment::gate::SubmitError;
    use crate::equipment::snapshot::{GuiderSnapshot, MountSnapshot};
    use crate::equipment::testing::fake_handles::{
        fake_guider, fake_mount, FakeDevice, FakeResponse,
    };
    use crate::equipment::types::{AutonomyMode, EqCoord, EqFrame, Observed, Priority, Source};
    use std::sync::Arc;

    fn guiding(state: GuiderAppState) -> GuiderSnapshot {
        GuiderSnapshot {
            app_state: Observed::fresh(state, Source::Phd2Event, chrono::Utc::now(), 1),
            ..GuiderSnapshot::default()
        }
    }

    fn slew(source: CommandSource) -> GatedCommand {
        GatedCommand::new(
            GatedOp::Mount(MountCmd::Slew {
                target: EqCoord::new(5.0, 20.0, EqFrame::J2000),
            }),
            source,
            AutonomyMode::Manual,
            Priority::Exposure,
        )
    }

    #[tokio::test]
    async fn a_guiding_guider_is_stopped_before_a_slew() {
        let (gate, ctx) = gate_with_ctx();
        let (guider, guider_fake) = fake_guider(guiding(GuiderAppState::Guiding));
        // PHD2 reports Stopped once stop_capture is processed.
        guider_fake.set_responder(|cmd, snap: &mut GuiderSnapshot| {
            if *cmd == GuiderCmd::Stop {
                snap.app_state = Observed::fresh(
                    GuiderAppState::Stopped,
                    Source::Phd2Event,
                    chrono::Utc::now(),
                    2,
                );
            }
            FakeResponse::succeed()
        });
        *ctx.guider.lock().unwrap() = Some(guider.clone());
        let (mount, mount_fake) = fake_mount(MountSnapshot::default());
        let cmd = slew(CommandSource::Operator { overriding: false });
        let submitted = gate.submit(&mount, cmd.clone()).await.unwrap();
        let report = submitted.pre_motion.unwrap();
        assert!(report.was_guiding);
        assert!(report.stop_sent);
        assert_eq!(report.guider_state, Some(GuiderAppState::Guiding));
        // The guider got its stop before the mount got its slew, and the
        // mount got the target the checks validated.
        assert_eq!(guider_fake.commands(), vec![GuiderCmd::Stop]);
        assert_eq!(
            mount_fake.commands(),
            vec![MountCmd::Slew {
                target: EqCoord::new(5.0, 20.0, EqFrame::J2000)
            }]
        );
        assert_eq!(submitted.ticket.wait().await, OpOutcome::Succeeded);
        let stop_id = report.stop_command_id.unwrap();
        assert_eq!(
            ctx.stages(stop_id),
            vec![
                crate::equipment::events::AuditStage::Proposed,
                crate::equipment::events::AuditStage::Validated,
                crate::equipment::events::AuditStage::Executed,
                crate::equipment::events::AuditStage::Outcome,
            ]
        );
        // A stopped guider is left alone.
        let again = gate
            .submit(&mount, slew(CommandSource::Operator { overriding: false }))
            .await
            .unwrap();
        assert!(!again.pre_motion.unwrap().was_guiding);
        assert_eq!(guider_fake.commands().len(), 1);
    }

    fn stopping_guider(
        state: GuiderAppState,
    ) -> (GuiderHandle, FakeDevice<GuiderCmd, GuiderSnapshot>) {
        let (guider, guider_fake) = fake_guider(guiding(state));
        guider_fake.set_responder(|cmd, snap: &mut GuiderSnapshot| {
            if *cmd == GuiderCmd::Stop {
                snap.app_state = Observed::fresh(
                    GuiderAppState::Stopped,
                    Source::Phd2Event,
                    chrono::Utc::now(),
                    2,
                );
            }
            FakeResponse::succeed()
        });
        (guider, guider_fake)
    }

    #[tokio::test]
    async fn a_flip_stops_guiding_under_its_own_guider_claim() {
        for (source, priority) in [
            (
                CommandSource::Operator { overriding: false },
                Priority::MountRecovery,
            ),
            (
                CommandSource::Trigger {
                    rule: "meridian".into(),
                },
                Priority::MountRecovery,
            ),
            (CommandSource::Safety, Priority::Safety),
        ] {
            let (gate, ctx) = gate_with_ctx();
            let (guider, guider_fake) = stopping_guider(GuiderAppState::Guiding);
            *ctx.guider.lock().unwrap() = Some(guider.clone());
            let (mount, mount_fake) = fake_mount(MountSnapshot::default());
            // The flip claims mount, guider and the camera boundary.
            let flip = GatedCommand::new(
                GatedOp::Flip,
                source.clone(),
                AutonomyMode::Manual,
                priority,
            );
            let flip_permit = gate.authorize(&flip).unwrap();
            // Its slew is a step: checked on its own target, sent as checked.
            let target = EqCoord::new(5.0, 20.0, EqFrame::J2000);
            let step = GatedCommand::new(
                GatedOp::Mount(MountCmd::Slew { target }),
                source.clone(),
                AutonomyMode::Manual,
                priority,
            );
            let submitted = gate
                .submit_step(&mount, &flip_permit, step)
                .await
                .unwrap_or_else(|e| panic!("{source:?}: {e}"));
            let report = submitted.pre_motion.unwrap();
            assert!(report.was_guiding && report.stop_sent, "{source:?}");
            assert!(report.warning.is_none(), "{source:?}: {report:?}");
            assert_eq!(guider_fake.commands(), vec![GuiderCmd::Stop], "{source:?}");
            assert_eq!(mount_fake.commands(), vec![MountCmd::Slew { target }]);
            let stop_id = report.stop_command_id.unwrap();
            assert_eq!(
                ctx.stages(stop_id),
                vec![
                    crate::equipment::events::AuditStage::Proposed,
                    crate::equipment::events::AuditStage::Validated,
                    crate::equipment::events::AuditStage::Executed,
                    crate::equipment::events::AuditStage::Outcome,
                ]
            );
            // The flip still holds the guider; its child stop never took it.
            assert_eq!(
                gate.holder(Domain::Device(DeviceKind::Guider))
                    .map(|c| c.command_id),
                Some(flip.command_id)
            );
        }
    }

    #[tokio::test]
    async fn compound_operations_are_not_sent_as_one_device_command() {
        let (gate, ctx) = gate_with_ctx();
        let (mount, mount_fake) = fake_mount(MountSnapshot::default());
        let flip = GatedCommand::new(
            GatedOp::Flip,
            CommandSource::Operator { overriding: false },
            AutonomyMode::Manual,
            Priority::MountRecovery,
        );
        let err = gate.submit(&mount, flip.clone()).await.unwrap_err();
        assert!(
            matches!(&err, SubmitError::Gate(GateError::Refused { .. })),
            "{err}"
        );
        assert!(mount_fake.commands().is_empty());
        assert_eq!(
            ctx.stages(flip.command_id),
            vec![
                crate::equipment::events::AuditStage::Proposed,
                crate::equipment::events::AuditStage::Rejected,
            ]
        );
        assert!(gate.is_idle(), "nothing was claimed");
    }

    #[tokio::test]
    async fn a_safety_motion_whose_stop_never_reached_the_guider_says_so() {
        let (gate, ctx) = gate_with_ctx();
        let (guider, guider_fake) = fake_guider(guiding(GuiderAppState::Guiding));
        guider_fake.set_responder(|_, _| {
            FakeResponse::reject(DeviceError::transport("PHD2 socket closed"))
        });
        *ctx.guider.lock().unwrap() = Some(guider.clone());
        let (mount, mount_fake) = fake_mount(MountSnapshot::default());
        let park = GatedCommand::new(
            GatedOp::Mount(MountCmd::Park),
            CommandSource::Safety,
            AutonomyMode::Manual,
            Priority::Safety,
        );
        let submitted = gate.submit(&mount, park).await.unwrap();
        let report = submitted.pre_motion.unwrap();
        assert!(report.was_guiding, "the guider was guiding");
        assert!(!report.stop_sent);
        assert!(report.stop_command_id.is_none());
        assert!(report.warning.unwrap().contains("refused"));
        assert_eq!(mount_fake.commands(), vec![MountCmd::Park]);
    }

    #[tokio::test]
    async fn a_guider_that_will_not_stop_fails_the_slew_unless_safety() {
        let (gate, ctx) = gate_with_ctx();
        let (guider, guider_fake) = fake_guider(guiding(GuiderAppState::Calibrating));
        // Accepts the stop but never leaves Calibrating.
        guider_fake.set_responder(|_, _| FakeResponse::succeed());
        *ctx.guider.lock().unwrap() = Some(guider.clone());
        let (mount, mount_fake) = fake_mount(MountSnapshot::default());

        // Drive the manual clock past the 10 s timeouts from another thread.
        let clock = Arc::clone(&ctx.clock);
        let stop_advancing = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_flag = Arc::clone(&stop_advancing);
        let advancer = std::thread::spawn(move || {
            for _ in 0..20_000 {
                if stop_flag.load(std::sync::atomic::Ordering::SeqCst) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(2));
                clock.advance_ms(100);
            }
        });
        let err = gate
            .submit(&mount, slew(CommandSource::Operator { overriding: false }))
            .await
            .unwrap_err();
        assert!(matches!(&err, SubmitError::Coord(e) if e.code() == "guider_stop_failed"));
        assert_eq!(err.http_status(), 409);
        assert!(
            mount_fake.commands().is_empty(),
            "the slew never reached the mount"
        );
        assert!(
            gate.holder(crate::equipment::commands::Domain::Device(
                crate::equipment::types::DeviceKind::Mount
            ))
            .is_none(),
            "the mount domain was released"
        );

        // A safety park proceeds anyway, with the failure as a warning.
        let park = GatedCommand::new(
            GatedOp::Mount(MountCmd::Park),
            CommandSource::Safety,
            AutonomyMode::Manual,
            Priority::Safety,
        );
        let submitted = gate.submit(&mount, park).await.unwrap();
        let report = submitted.pre_motion.unwrap();
        assert!(report.was_guiding);
        assert!(
            report.stop_sent,
            "the stop was accepted, PHD2 never stopped"
        );
        assert!(report.warning.unwrap().contains("Stopped"));
        assert_eq!(mount_fake.commands(), vec![MountCmd::Park]);
        stop_advancing.store(true, std::sync::atomic::Ordering::SeqCst);
        advancer.join().unwrap();
    }

    #[tokio::test]
    async fn refused_stops_fail_fast_and_unknown_states_warn() {
        let (gate, ctx) = gate_with_ctx();
        let (guider, guider_fake) = fake_guider(guiding(GuiderAppState::LostLock));
        guider_fake.set_responder(|_, _| {
            FakeResponse::reject(DeviceError::transport("PHD2 socket closed"))
        });
        *ctx.guider.lock().unwrap() = Some(guider.clone());
        let motion = slew(CommandSource::Operator { overriding: false });
        let err = stop_guiding_before_motion(
            &gate,
            Some(&guider),
            &motion,
            None,
            ctx.clock.as_ref(),
            STOP_GUIDING_TIMEOUT,
        )
        .await
        .unwrap_err();
        assert!(err.reason.contains("refused"));
        let (unknown, _f) = fake_guider(GuiderSnapshot::default());
        let report = stop_guiding_before_motion(
            &gate,
            Some(&unknown),
            &motion,
            None,
            ctx.clock.as_ref(),
            STOP_GUIDING_TIMEOUT,
        )
        .await
        .unwrap();
        assert!(!report.was_guiding);
        assert_eq!(report.warning.as_deref(), Some("guider_state_unknown"));
        // Focuser moves and tracking changes never stop the guider.
        assert!(!needs_guider_stop(&GatedOp::Mount(MountCmd::SetTracking(
            true
        ))));
        assert!(needs_guider_stop(&GatedOp::Flip));
    }
}

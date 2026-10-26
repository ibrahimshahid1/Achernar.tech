//! The on-unsafe policy (spec §3.10) and the safety supervisor. Owner:
//! WP1A-8.
//!
//! **When it runs.** Only on a confirmed unsafe reading or on a monitor
//! going `Lost` (see [`super::monitor_hub`]). One `Degraded` poll, a failed
//! read or a stale value never runs it; they only make safety `Unknown`,
//! which refuses new automatic actions. The supervisor latches the policy so
//! it runs once per unsafe episode while the rig stays secured, and re-arms
//! it on a confirmed safe state (or once no monitor is configured). A new
//! confirmed edge in the same episode runs it again when the rig is no
//! longer secured (for example, unparked by the operator while safety was
//! unknown), and a trigger that arrives during a run is looked at again
//! once that run ends.
//!
//! **What it does, in order** (the §5.3 and §5.7 acceptance order; §3.10
//! lists the exposure abort before stopping the guider, see the change notes):
//!
//! 1. Raise the `safety_unsafe` resume block, which stops automatic
//!    sequencing (H22).
//! 2. Stop guiding (`GuiderCmd::Stop`), unless the guider is known stopped.
//! 3. Abort the exposure through the camera hook, if
//!    `[safety] abort_exposure_on_unsafe` and an exposure may be running.
//! 4. Abort any slew in progress through the mount's out-of-band abort path
//!    (D4), and wait until the mount reports it stopped.
//! 5. Park, or stop tracking, according to the effective `on_unsafe`. A
//!    mount without CanPark stops tracking (Tier 1; the fallback park is
//!    Tier 2). A failed park emits `park_failed`, raises `mount_lost`, and
//!    then stops tracking as a fallback. `notify` leaves the mount alone. A
//!    park already in progress, or a parked mount, is not disturbed.
//!
//! Every action goes through the gate at [`Priority::Safety`] with
//! [`CommandSource::Safety`], so it preempts every holder below safety, and
//! is audited per stage: `proposed` and `validated`/`rejected` by the gate,
//! `executed` and `outcome` here. A failed step never stops the sequence;
//! aborts are attempted even when the gate refuses them (D19). If a device
//! refuses a command because another operation is active (no preemption),
//! the policy cancels that operation out of band and retries once.
//!
//! **Resume** is manual in Tier 1: the `safety_unsafe` block is cleared by
//! [`SafetySupervisor::request_resume`] only while safety reads confirmed
//! safe (or no monitor is configured). `resume = "auto"` is Tier 2 (WP2-9)
//! and is treated as manual here (D18).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::watch;
use uuid::Uuid;

use super::monitor_hub::{Evaluation, HubSettings, MonitorInput, PolicyTrigger, SafetyMonitorHub};
use super::{
    safety_block_clear_allowed, BlockRequest, EventSink, ResumeBlockStore, SafetyGatePort,
};
use crate::config::{EquipmentConfigs, MountConfig, OnUnsafe, SafetyConfig, SafetyResume};
use crate::equipment::commands::{GatedCommand, GatedOp, GuiderCmd, MountCmd};
use crate::equipment::events::{AuditRecord, AuditStage, EquipmentEvent, EventKind};
use crate::equipment::handle::{DeviceHandle, GuiderHandle, MountHandle, SafetyMonitorHandle};
use crate::equipment::ops::OpKind;
use crate::equipment::snapshot::{
    GuiderAppState, GuiderSnapshot, MountSnapshot, ResumeBlock, ResumeBlockKind, SafetyGate,
    SafetySummary,
};
use crate::equipment::traits::{CameraActivity, Clock};
use crate::equipment::types::{
    AutonomyMode, CommandId, CommandSource, DeviceError, DeviceKind, ErrorClass, OpOutcome,
    Priority,
};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Policy name recorded in every command's audited parameters.
pub const POLICY_NAME: &str = "safety.on_unsafe";

// ---------------------------------------------------------------------------
// Settings and targets
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct PolicySettings {
    /// Resolved against `[mount.safety] park_on_safety_violation`; never
    /// `FromMountConfig`.
    pub on_unsafe: OnUnsafe,
    /// `[safety] abort_exposure_on_unsafe`.
    pub abort_exposure: bool,
    /// Wait for the guider to stop (the §2.9 pre-motion rule uses 10 s).
    pub guider_stop_timeout: Duration,
    pub camera_abort_timeout: Duration,
    /// Wait for an abort path to report.
    pub abort_timeout: Duration,
    /// Wait for the mount to report it stopped after an abort.
    pub stop_wait: Duration,
    /// `[mount] slew_timeout_s` plus a margin; the park operation has its
    /// own deadline, this only guards against a worker that never answers.
    pub park_timeout: Duration,
    /// Tracking off and other short commands.
    pub command_timeout: Duration,
}

impl Default for PolicySettings {
    fn default() -> Self {
        Self::from_config(&SafetyConfig::default(), &MountConfig::default())
    }
}

impl PolicySettings {
    pub fn from_config(safety: &SafetyConfig, mount: &MountConfig) -> Self {
        Self {
            on_unsafe: safety.effective_on_unsafe(&mount.safety),
            abort_exposure: safety.abort_exposure_on_unsafe,
            guider_stop_timeout: Duration::from_secs(10),
            camera_abort_timeout: Duration::from_secs(10),
            abort_timeout: Duration::from_secs(10),
            stop_wait: Duration::from_secs(10),
            park_timeout: Duration::from_secs(mount.slew_timeout_s.saturating_add(30)),
            command_timeout: Duration::from_secs(30),
        }
    }
}

fn on_unsafe_label(on_unsafe: OnUnsafe) -> &'static str {
    match on_unsafe {
        OnUnsafe::FromMountConfig => "from_mount_config",
        OnUnsafe::Park => "park",
        OnUnsafe::StopTracking => "stop_tracking",
        OnUnsafe::Notify => "notify",
    }
}

/// The devices the policy acts on, read when it runs.
#[derive(Clone, Default)]
pub struct PolicyTargets {
    pub mount: Option<MountHandle>,
    pub guider: Option<GuiderHandle>,
    /// The camera hook (`CameraActivity`, WP1B-1).
    pub camera: Option<Arc<dyn CameraActivity>>,
}

impl std::fmt::Debug for PolicyTargets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyTargets")
            .field("mount", &self.mount)
            .field("guider", &self.guider)
            .field("camera", &self.camera.is_some())
            .finish()
    }
}

/// Where the supervisor finds the current devices. The spine implements it
/// over the `AppState` slots and the hub.
pub trait PolicyTargetSource: Send + Sync {
    fn targets(&self) -> PolicyTargets;
}

impl PolicyTargetSource for PolicyTargets {
    fn targets(&self) -> PolicyTargets {
        self.clone()
    }
}

/// Targets that can be swapped as devices connect and disconnect.
#[derive(Default)]
pub struct SharedTargets {
    targets: Mutex<PolicyTargets>,
}

impl SharedTargets {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn set_mount(&self, mount: Option<MountHandle>) {
        lock(&self.targets).mount = mount;
    }

    pub fn set_guider(&self, guider: Option<GuiderHandle>) {
        lock(&self.targets).guider = guider;
    }

    pub fn set_camera(&self, camera: Option<Arc<dyn CameraActivity>>) {
        lock(&self.targets).camera = camera;
    }
}

impl PolicyTargetSource for SharedTargets {
    fn targets(&self) -> PolicyTargets {
        lock(&self.targets).clone()
    }
}

// ---------------------------------------------------------------------------
// Run report
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyAction {
    RaiseResumeBlock,
    StopGuiding,
    AbortExposure,
    AbortSlew,
    Park,
    StopTracking,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum StepStatus {
    Done,
    Skipped { reason: String },
    Failed { error: DeviceError },
}

impl StepStatus {
    fn skipped(reason: impl Into<String>) -> Self {
        StepStatus::Skipped {
            reason: reason.into(),
        }
    }

    fn from_result(result: Result<(), DeviceError>) -> Self {
        match result {
            Ok(()) => StepStatus::Done,
            Err(error) => StepStatus::Failed { error },
        }
    }

    pub fn is_done(&self) -> bool {
        matches!(self, StepStatus::Done)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PolicyStep {
    pub action: PolicyAction,
    pub command_id: Option<CommandId>,
    pub status: StepStatus,
    pub at: DateTime<Utc>,
}

/// What one run did, step by step.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PolicyRun {
    pub trigger: PolicyTrigger,
    pub on_unsafe: &'static str,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub steps: Vec<PolicyStep>,
    /// The `safety_unsafe` block this run raised (or found active).
    pub block_id: Option<Uuid>,
}

impl PolicyRun {
    pub fn step(&self, action: PolicyAction) -> Option<&PolicyStep> {
        self.steps.iter().find(|step| step.action == action)
    }

    /// Actions that were carried out, in order.
    pub fn done(&self) -> Vec<PolicyAction> {
        self.steps
            .iter()
            .filter(|step| step.status.is_done())
            .map(|step| step.action)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// The policy executor
// ---------------------------------------------------------------------------

/// Operations that move the mount, so an abort is needed before a park.
fn mount_motion_op(kind: OpKind) -> bool {
    matches!(
        kind,
        OpKind::Slew
            | OpKind::SlewAltAz
            | OpKind::Home
            | OpKind::MoveAxis
            | OpKind::Unpark
            | OpKind::PulseGuide
            | OpKind::Center
            | OpKind::Flip
    )
}

/// Conservative: unknown slewing counts as moving.
fn mount_may_be_moving(snap: &MountSnapshot) -> bool {
    snap.jog_active
        || snap
            .active_op
            .as_ref()
            .is_some_and(|op| mount_motion_op(op.kind))
        || snap.state.slewing.fresh_value() != Some(&false)
}

/// After an abort: nothing reports motion any more.
fn mount_stopped(snap: &MountSnapshot) -> bool {
    !snap.jog_active
        && !snap
            .active_op
            .as_ref()
            .is_some_and(|op| mount_motion_op(op.kind))
        && snap.state.slewing.fresh_value() != Some(&true)
}

fn outcome_result(outcome: OpOutcome) -> Result<(), DeviceError> {
    match outcome {
        OpOutcome::Succeeded => Ok(()),
        OpOutcome::Failed(err) | OpOutcome::Unknown(err) => Err(err),
        OpOutcome::Aborted { by } => Err(DeviceError::cancelled(match by {
            Some(by) => format!("the operation was aborted by command {by}"),
            None => "the operation was aborted".to_string(),
        })),
        OpOutcome::TimedOut => Err(DeviceError::timeout("the operation timed out")),
    }
}

/// Runs the on-unsafe sequence. Holds no lock across an `.await`.
pub struct UnsafePolicy {
    settings: PolicySettings,
    gate: Arc<dyn SafetyGatePort>,
    blocks: Arc<dyn ResumeBlockStore>,
    events: Arc<dyn EventSink>,
    clock: Arc<dyn Clock>,
}

impl UnsafePolicy {
    pub fn new(
        settings: PolicySettings,
        gate: Arc<dyn SafetyGatePort>,
        blocks: Arc<dyn ResumeBlockStore>,
        events: Arc<dyn EventSink>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            settings,
            gate,
            blocks,
            events,
            clock,
        }
    }

    pub fn settings(&self) -> &PolicySettings {
        &self.settings
    }

    fn command(&self, op: GatedOp, trigger: &PolicyTrigger) -> GatedCommand {
        let mut cmd = GatedCommand::new(
            op,
            CommandSource::Safety,
            AutonomyMode::Bounded,
            Priority::Safety,
        );
        if let Value::Object(params) = &mut cmd.params {
            params.insert("policy".into(), json!(POLICY_NAME));
            params.insert(
                "trigger".into(),
                serde_json::to_value(trigger).unwrap_or(Value::Null),
            );
        }
        cmd
    }

    fn audit(
        &self,
        cmd: &GatedCommand,
        stage: AuditStage,
        accepted: bool,
        reason: Option<String>,
        outcome: Option<OpOutcome>,
    ) {
        let mut record =
            AuditRecord::for_command(cmd, stage, accepted, reason, self.clock.now_utc());
        record.outcome = outcome;
        self.gate.audit(record);
    }

    fn push(
        &self,
        run: &mut PolicyRun,
        action: PolicyAction,
        id: Option<CommandId>,
        status: StepStatus,
    ) {
        if let StepStatus::Failed { error } = &status {
            tracing::warn!("safety policy: {action:?} failed: {error}");
        }
        run.steps.push(PolicyStep {
            action,
            command_id: id,
            status,
            at: self.clock.now_utc(),
        });
    }

    /// Wait (bounded) until `done` holds for the handle's snapshot.
    async fn wait_until<C, S>(
        &self,
        handle: &DeviceHandle<C, S>,
        done: impl Fn(&S) -> bool,
        limit: Duration,
    ) -> bool {
        let mut snapshots = handle.watch_snapshot();
        let wait = async {
            loop {
                let satisfied = {
                    let snapshot = snapshots.borrow_and_update();
                    done(&snapshot)
                };
                if satisfied {
                    return true;
                }
                if snapshots.changed().await.is_err() {
                    return false;
                }
            }
        };
        tokio::time::timeout(limit, wait).await.unwrap_or(false)
    }

    /// Authorise, send, await and audit one device command. A device that
    /// refuses because another operation is active (the gate did not
    /// preempt) has that operation cancelled out of band, then one retry.
    async fn execute<C, S>(
        &self,
        handle: &DeviceHandle<C, S>,
        device_cmd: C,
        op: GatedOp,
        trigger: &PolicyTrigger,
        timeout: Duration,
        busy: fn(&S) -> bool,
    ) -> (CommandId, Result<(), DeviceError>)
    where
        C: Clone,
    {
        let (id, result, _started) = self
            .execute_tracked(handle, device_cmd, op, trigger, timeout, busy)
            .await;
        (id, result)
    }

    /// [`Self::execute`], also saying whether the worker accepted the
    /// command, i.e. whether the operation itself ran (O11 c: a park that
    /// ran reports its own `park_failed`).
    async fn execute_tracked<C, S>(
        &self,
        handle: &DeviceHandle<C, S>,
        device_cmd: C,
        op: GatedOp,
        trigger: &PolicyTrigger,
        timeout: Duration,
        busy: fn(&S) -> bool,
    ) -> (CommandId, Result<(), DeviceError>, bool)
    where
        C: Clone,
    {
        let cmd = self.command(op, trigger);
        let id = cmd.command_id;
        let mut retried = false;
        loop {
            let permit = match self.gate.authorize(&cmd).await {
                Ok(permit) => permit,
                // The gate audited the rejection.
                Err(err) => return (id, Err(err.to_device_error()), false),
            };
            match handle.send(device_cmd.clone(), permit).await {
                Ok(ticket) => {
                    self.audit(&cmd, AuditStage::Executed, true, None, None);
                    let outcome = match tokio::time::timeout(timeout, ticket.wait()).await {
                        Ok(outcome) => outcome,
                        Err(_) => OpOutcome::Unknown(DeviceError::outcome_unknown(format!(
                            "no outcome within {} s",
                            timeout.as_secs()
                        ))),
                    };
                    self.audit(
                        &cmd,
                        AuditStage::Outcome,
                        outcome.is_success(),
                        None,
                        Some(outcome.clone()),
                    );
                    return (id, outcome_result(outcome), true);
                }
                Err(err) if err.class == ErrorClass::InvalidOperation && !retried => {
                    retried = true;
                    self.audit(
                        &cmd,
                        AuditStage::Executed,
                        false,
                        Some(format!(
                            "refused while another operation was active ({err}); cancelling it \
                             and retrying once"
                        )),
                        None,
                    );
                    let abort = handle.abort();
                    let _ =
                        tokio::time::timeout(self.settings.abort_timeout, abort.outcome()).await;
                    self.wait_until(handle, |s| !busy(s), self.settings.stop_wait)
                        .await;
                }
                Err(err) => {
                    self.audit(
                        &cmd,
                        AuditStage::Outcome,
                        false,
                        Some(err.to_string()),
                        Some(OpOutcome::Failed(err.clone())),
                    );
                    return (id, Err(err), false);
                }
            }
        }
    }

    /// Abort-type commands skip the gate's checks but are audited; a
    /// refusal is noted and the abort is attempted anyway (D19).
    async fn authorize_abort(&self, cmd: &GatedCommand) -> Option<String> {
        match self.gate.authorize(cmd).await {
            Ok(_permit) => None,
            Err(err) => Some(format!(
                "the gate refused the abort ({err}); aborts are always attempted"
            )),
        }
    }

    /// True when a run now would leave every device alone: it reads the
    /// same snapshots and applies the same skip rules as [`Self::run`]. The
    /// supervisor uses it to re-run the policy on a new confirmed edge when
    /// the rig was unsecured after an earlier run (for example, unparked by
    /// the operator while safety was unknown). Reads handles only.
    pub fn rig_secured(&self, targets: &PolicyTargets) -> bool {
        let guider_secured = match targets.guider.as_ref().filter(|g| !g.is_retired()) {
            None => true,
            Some(guider) => {
                guider.snapshot().app_state.fresh_value() == Some(&GuiderAppState::Stopped)
            }
        };
        let camera_secured = !self.settings.abort_exposure
            || targets
                .camera
                .as_ref()
                .is_none_or(|camera| camera.exposing().fresh_value() == Some(&false));
        guider_secured && camera_secured && self.mount_secured(targets.mount.as_ref())
    }

    fn mount_secured(&self, mount: Option<&MountHandle>) -> bool {
        let Some(mount) = mount.filter(|m| !m.is_retired()) else {
            return true;
        };
        if self.settings.on_unsafe == OnUnsafe::Notify {
            return true;
        }
        let snap = mount.snapshot();
        if snap
            .active_op
            .as_ref()
            .is_some_and(|op| op.kind == OpKind::Park)
        {
            return true;
        }
        if mount_may_be_moving(&snap) {
            return false;
        }
        if snap.state.at_park.fresh_value() == Some(&true) {
            return true;
        }
        let parks = matches!(
            self.settings.on_unsafe,
            OnUnsafe::Park | OnUnsafe::FromMountConfig
        ) && snap.capabilities.can_park;
        !parks && snap.state.tracking.fresh_value() == Some(&false)
    }

    /// Run the sequence for `trigger` against `targets`.
    pub async fn run(&self, trigger: &PolicyTrigger, targets: &PolicyTargets) -> PolicyRun {
        let started_at = self.clock.now_utc();
        tracing::warn!(
            "safety policy: {} (on_unsafe = {})",
            trigger.reason(),
            on_unsafe_label(self.settings.on_unsafe)
        );
        let mut run = PolicyRun {
            trigger: trigger.clone(),
            on_unsafe: on_unsafe_label(self.settings.on_unsafe),
            started_at,
            finished_at: started_at,
            steps: Vec::new(),
            block_id: None,
        };

        // 1. Stop automatic sequencing (H22).
        let block = self.blocks.raise(
            BlockRequest::new(ResumeBlockKind::SafetyUnsafe, trigger.reason())
                .for_device(DeviceKind::SafetyMonitor, trigger.connection_id()),
        );
        run.block_id = Some(block.id);
        self.push(
            &mut run,
            PolicyAction::RaiseResumeBlock,
            None,
            StepStatus::Done,
        );

        // 2-5.
        self.stop_guiding(trigger, targets.guider.as_ref(), &mut run)
            .await;
        self.abort_exposure(trigger, targets.camera.as_ref(), &mut run)
            .await;
        self.secure_mount(trigger, targets.mount.as_ref(), &mut run)
            .await;

        run.finished_at = self.clock.now_utc();
        run
    }

    async fn stop_guiding(
        &self,
        trigger: &PolicyTrigger,
        guider: Option<&GuiderHandle>,
        run: &mut PolicyRun,
    ) {
        let Some(guider) = guider.filter(|g| !g.is_retired()) else {
            self.push(
                run,
                PolicyAction::StopGuiding,
                None,
                StepStatus::skipped("no guider is connected"),
            );
            return;
        };
        if guider.snapshot().app_state.fresh_value() == Some(&GuiderAppState::Stopped) {
            self.push(
                run,
                PolicyAction::StopGuiding,
                None,
                StepStatus::skipped("the guider is already stopped"),
            );
            return;
        }
        let (id, result) = self
            .execute(
                guider,
                GuiderCmd::Stop,
                GatedOp::Guider(GuiderCmd::Stop),
                trigger,
                self.settings.guider_stop_timeout,
                |s: &GuiderSnapshot| s.active_op.is_some(),
            )
            .await;
        self.push(
            run,
            PolicyAction::StopGuiding,
            Some(id),
            StepStatus::from_result(result),
        );
    }

    async fn abort_exposure(
        &self,
        trigger: &PolicyTrigger,
        camera: Option<&Arc<dyn CameraActivity>>,
        run: &mut PolicyRun,
    ) {
        if !self.settings.abort_exposure {
            self.push(
                run,
                PolicyAction::AbortExposure,
                None,
                StepStatus::skipped("[safety] abort_exposure_on_unsafe is false"),
            );
            return;
        }
        let Some(camera) = camera else {
            self.push(
                run,
                PolicyAction::AbortExposure,
                None,
                StepStatus::skipped("no camera hook is installed"),
            );
            return;
        };
        if camera.exposing().fresh_value() == Some(&false) {
            self.push(
                run,
                PolicyAction::AbortExposure,
                None,
                StepStatus::skipped("no exposure is in progress"),
            );
            return;
        }
        let cmd = self.command(
            GatedOp::Abort {
                kind: DeviceKind::Camera,
            },
            trigger,
        );
        let note = self.authorize_abort(&cmd).await;
        self.audit(&cmd, AuditStage::Executed, true, note, None);
        let camera = Arc::clone(camera);
        let blocking = tokio::task::spawn_blocking(move || camera.abort_exposure());
        let result = match tokio::time::timeout(self.settings.camera_abort_timeout, blocking).await
        {
            Ok(Ok(result)) => result,
            Ok(Err(join)) => Err(DeviceError::driver(format!(
                "the camera abort did not complete: {join}"
            ))),
            Err(_) => Err(DeviceError::outcome_unknown(format!(
                "the camera abort did not return within {} s",
                self.settings.camera_abort_timeout.as_secs()
            ))),
        };
        let outcome = match &result {
            Ok(()) => OpOutcome::Succeeded,
            Err(err) => OpOutcome::Failed(err.clone()),
        };
        self.audit(
            &cmd,
            AuditStage::Outcome,
            result.is_ok(),
            None,
            Some(outcome),
        );
        self.push(
            run,
            PolicyAction::AbortExposure,
            Some(cmd.command_id),
            StepStatus::from_result(result),
        );
    }

    async fn secure_mount(
        &self,
        trigger: &PolicyTrigger,
        mount: Option<&MountHandle>,
        run: &mut PolicyRun,
    ) {
        let Some(mount) = mount.filter(|m| !m.is_retired()) else {
            for action in [PolicyAction::AbortSlew, PolicyAction::Park] {
                self.push(
                    run,
                    action,
                    None,
                    StepStatus::skipped("no mount is connected"),
                );
            }
            return;
        };
        if self.settings.on_unsafe == OnUnsafe::Notify {
            for action in [PolicyAction::AbortSlew, PolicyAction::Park] {
                self.push(
                    run,
                    action,
                    None,
                    StepStatus::skipped("on_unsafe = notify: the mount is left alone"),
                );
            }
            return;
        }
        let snap = mount.snapshot();
        if let Some(op) = snap.active_op.as_ref().filter(|op| op.kind == OpKind::Park) {
            let reason = format!("a park is already in progress (command {})", op.command_id);
            self.push(
                run,
                PolicyAction::AbortSlew,
                None,
                StepStatus::skipped(reason.clone()),
            );
            self.push(run, PolicyAction::Park, None, StepStatus::skipped(reason));
            return;
        }

        // 4. Abort any slew in progress, out of band.
        if mount_may_be_moving(&snap) {
            self.abort_slew(trigger, mount, run).await;
        } else {
            self.push(
                run,
                PolicyAction::AbortSlew,
                None,
                StepStatus::skipped("the mount is not moving"),
            );
        }

        // 5. Park or stop tracking.
        let snap = mount.snapshot();
        if snap.state.at_park.fresh_value() == Some(&true) {
            self.push(
                run,
                PolicyAction::Park,
                None,
                StepStatus::skipped("the mount is already parked"),
            );
            return;
        }
        match self.settings.on_unsafe {
            OnUnsafe::Park | OnUnsafe::FromMountConfig if snap.capabilities.can_park => {
                self.park(trigger, mount, run).await;
            }
            OnUnsafe::Park | OnUnsafe::FromMountConfig => {
                self.push(
                    run,
                    PolicyAction::Park,
                    None,
                    StepStatus::skipped(
                        "the mount cannot park (CanPark is false); stopping tracking instead \
                         (the fallback park is Tier 2)",
                    ),
                );
                self.stop_tracking(trigger, mount, run).await;
            }
            OnUnsafe::StopTracking => {
                self.push(
                    run,
                    PolicyAction::Park,
                    None,
                    StepStatus::skipped("on_unsafe = stop_tracking"),
                );
                self.stop_tracking(trigger, mount, run).await;
            }
            OnUnsafe::Notify => {}
        }
    }

    async fn abort_slew(&self, trigger: &PolicyTrigger, mount: &MountHandle, run: &mut PolicyRun) {
        let cmd = self.command(GatedOp::Mount(MountCmd::AbortSlew), trigger);
        let note = self.authorize_abort(&cmd).await;
        let ticket = mount.abort();
        let path = format!(
            "abort path {}, cancel {}",
            if ticket.abort_path_ran {
                "ran"
            } else {
                "absent"
            },
            if ticket.cancel_sent {
                "queued"
            } else {
                "not queued"
            }
        );
        let note = Some(match note {
            Some(note) => format!("{note}; {path}"),
            None => path,
        });
        self.audit(&cmd, AuditStage::Executed, true, note, None);
        let result = match tokio::time::timeout(self.settings.abort_timeout, ticket.outcome()).await
        {
            Ok(result) => result,
            Err(_) => Err(DeviceError::outcome_unknown(format!(
                "the abort path did not report within {} s",
                self.settings.abort_timeout.as_secs()
            ))),
        };
        let result = match result {
            Ok(()) => {
                if self
                    .wait_until(mount, mount_stopped, self.settings.stop_wait)
                    .await
                {
                    Ok(())
                } else {
                    Err(DeviceError::outcome_unknown(format!(
                        "the mount still reports motion {} s after the abort",
                        self.settings.stop_wait.as_secs()
                    )))
                }
            }
            Err(err) => Err(err),
        };
        let outcome = match &result {
            Ok(()) => OpOutcome::Succeeded,
            Err(err) if err.class == ErrorClass::OutcomeUnknown => OpOutcome::Unknown(err.clone()),
            Err(err) => OpOutcome::Failed(err.clone()),
        };
        self.audit(
            &cmd,
            AuditStage::Outcome,
            result.is_ok(),
            None,
            Some(outcome),
        );
        self.push(
            run,
            PolicyAction::AbortSlew,
            Some(cmd.command_id),
            StepStatus::from_result(result),
        );
    }

    async fn park(&self, trigger: &PolicyTrigger, mount: &MountHandle, run: &mut PolicyRun) {
        let (id, result, started) = self
            .execute_tracked(
                mount,
                MountCmd::Park,
                GatedOp::Mount(MountCmd::Park),
                trigger,
                self.settings.park_timeout,
                |s: &MountSnapshot| s.active_op.is_some(),
            )
            .await;
        let failure = result.as_ref().err().cloned();
        self.push(
            run,
            PolicyAction::Park,
            Some(id),
            StepStatus::from_result(result),
        );
        let Some(err) = failure else {
            return;
        };
        // A failed park is recorded and raises `mount_lost`; then tracking
        // is stopped as a fallback so the mount does not drive on.
        // `park_failed` is emitted once (O11 c): by the park operation when it
        // ran, and here only when it never started (refused by the gate or
        // the worker).
        if !started {
            self.events.emit(
                EquipmentEvent::new(
                    EventKind::ParkFailed {
                        class: err.class,
                        reason: format!("safety policy park: {}", err.message),
                    },
                    self.clock.as_ref(),
                )
                .for_device(DeviceKind::Mount, mount.connection_id())
                .with_command(id),
            );
        }
        self.blocks.raise(
            BlockRequest::new(
                ResumeBlockKind::MountLost,
                format!("the safety park failed: {err}"),
            )
            .for_device(DeviceKind::Mount, mount.connection_id())
            .with_command(id),
        );
        self.stop_tracking(trigger, mount, run).await;
    }

    async fn stop_tracking(
        &self,
        trigger: &PolicyTrigger,
        mount: &MountHandle,
        run: &mut PolicyRun,
    ) {
        if mount.snapshot().state.tracking.fresh_value() == Some(&false) {
            self.push(
                run,
                PolicyAction::StopTracking,
                None,
                StepStatus::skipped("tracking is already off"),
            );
            return;
        }
        let (id, result) = self
            .execute(
                mount,
                MountCmd::SetTracking(false),
                GatedOp::Mount(MountCmd::SetTracking(false)),
                trigger,
                self.settings.command_timeout,
                |s: &MountSnapshot| s.active_op.is_some(),
            )
            .await;
        self.push(
            run,
            PolicyAction::StopTracking,
            Some(id),
            StepStatus::from_result(result),
        );
    }
}

// ---------------------------------------------------------------------------
// The supervisor
// ---------------------------------------------------------------------------

/// What one supervisor step saw and did.
#[derive(Debug, Clone)]
pub struct StepReport {
    pub evaluation: Evaluation,
    pub run: Option<PolicyRun>,
    /// Triggers not run now: the policy already ran for this episode and
    /// the rig is still secured, or a run is in progress. The latest trigger
    /// that arrived during a run is retried once the run ends, if the rig is
    /// then still not secured.
    pub suppressed: Vec<PolicyTrigger>,
}

enum BlockAction {
    None,
    RaiseUnknown(String),
    Clear(Uuid),
}

struct SupervisorState {
    hub: SafetyMonitorHub,
    monitors: Vec<SafetyMonitorHandle>,
    /// No run yet in this unsafe episode: the next trigger runs the policy
    /// even on a secured rig (it still raises the block).
    armed: bool,
    /// A trigger that arrived while a run was in progress.
    pending: Option<PolicyTrigger>,
    /// The `safety_unknown` block this supervisor raised.
    unknown_block: Option<Uuid>,
}

struct RunningGuard<'a>(&'a AtomicBool);

impl Drop for RunningGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

const MAX_RUNS_KEPT: usize = 32;

/// Ties monitors, the hub and the policy together. Reads only handles and
/// watches; never touches a device.
pub struct SafetySupervisor {
    state: Mutex<SupervisorState>,
    policy: UnsafePolicy,
    targets: Arc<dyn PolicyTargetSource>,
    blocks: Arc<dyn ResumeBlockStore>,
    events: Arc<dyn EventSink>,
    clock: Arc<dyn Clock>,
    summary: watch::Sender<SafetySummary>,
    runs: Mutex<Vec<PolicyRun>>,
    running: AtomicBool,
}

impl SafetySupervisor {
    pub fn new(
        hub: HubSettings,
        policy: UnsafePolicy,
        targets: Arc<dyn PolicyTargetSource>,
        blocks: Arc<dyn ResumeBlockStore>,
        events: Arc<dyn EventSink>,
        clock: Arc<dyn Clock>,
    ) -> Arc<Self> {
        let (summary, _) = watch::channel(SafetySummary::default());
        Arc::new(Self {
            state: Mutex::new(SupervisorState {
                hub: SafetyMonitorHub::new(hub),
                monitors: Vec::new(),
                armed: true,
                pending: None,
                unknown_block: None,
            }),
            policy,
            targets,
            blocks,
            events,
            clock,
            summary,
            runs: Mutex::new(Vec::new()),
            running: AtomicBool::new(false),
        })
    }

    /// Build from the parsed configuration (`[safety]`, `[mount]`).
    pub fn from_config(
        configs: &EquipmentConfigs,
        gate: Arc<dyn SafetyGatePort>,
        blocks: Arc<dyn ResumeBlockStore>,
        events: Arc<dyn EventSink>,
        targets: Arc<dyn PolicyTargetSource>,
        clock: Arc<dyn Clock>,
    ) -> Arc<Self> {
        if configs.safety.resume == SafetyResume::Auto {
            tracing::warn!(
                "[safety] resume = \"auto\" is a Tier 2 feature (WP2-9); resuming after unsafe \
                 stays manual"
            );
        }
        let policy = UnsafePolicy::new(
            PolicySettings::from_config(&configs.safety, &configs.mount),
            gate,
            Arc::clone(&blocks),
            Arc::clone(&events),
            Arc::clone(&clock),
        );
        Self::new(
            HubSettings::from_config(&configs.safety),
            policy,
            targets,
            blocks,
            events,
            clock,
        )
    }

    pub fn policy(&self) -> &UnsafePolicy {
        &self.policy
    }

    /// Add a monitor (replacing one with the same connection id).
    pub fn add_monitor(&self, handle: SafetyMonitorHandle) {
        let mut state = lock(&self.state);
        let id = handle.connection_id();
        state.monitors.retain(|h| h.connection_id() != id);
        state.monitors.push(handle);
    }

    /// Remove a monitor (an operator disconnect). Retired handles are also
    /// dropped automatically on the next evaluation.
    pub fn remove_monitor(&self, connection_id: Uuid) -> bool {
        let mut state = lock(&self.state);
        let before = state.monitors.len();
        state
            .monitors
            .retain(|h| h.connection_id() != connection_id);
        state.monitors.len() != before
    }

    pub fn monitor_count(&self) -> usize {
        lock(&self.state).monitors.len()
    }

    /// The latest combined safety (for `EquipmentSnapshot.safety`).
    pub fn summary(&self) -> SafetySummary {
        self.summary.borrow().clone()
    }

    pub fn watch_summary(&self) -> watch::Receiver<SafetySummary> {
        self.summary.subscribe()
    }

    pub fn policy_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// The most recent policy runs, oldest first.
    pub fn runs(&self) -> Vec<PolicyRun> {
        lock(&self.runs).clone()
    }

    /// Set the running flag if no run is in progress.
    fn try_start(&self) -> bool {
        self.running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Evaluate the monitors; emit `safety_changed`; manage the
    /// `safety_unknown` block; pick at most one trigger to run. A trigger
    /// that is returned has the running flag set, and the caller must run
    /// it ([`Self::execute_run`] clears the flag).
    ///
    /// Trigger selection. The first trigger of an unsafe episode always
    /// runs. A later one in the same episode runs only if the rig is no
    /// longer secured ([`UnsafePolicy::rig_secured`]): an edge that finds
    /// the mount unparked or tracking again (say, after an operator unpark
    /// while safety was unknown) secures it again, while an edge on a still
    /// secured rig is latched. A trigger that arrives during a run is kept
    /// and looked at again once the run has ended.
    fn assess(&self) -> (Evaluation, Option<PolicyTrigger>, Vec<PolicyTrigger>) {
        // Read the rig before taking the state lock: the target source and
        // the handles take their own locks. `running` is read first, so a
        // run that had ended by then is reflected in `secured`.
        let was_running = self.running.load(Ordering::Acquire);
        let secured = self.policy.rig_secured(&self.targets.targets());
        let (evaluation, trigger, suppressed, action) = {
            let mut state = lock(&self.state);
            state.monitors.retain(|h| !h.is_retired());
            let inputs: Vec<MonitorInput> = state
                .monitors
                .iter()
                .map(MonitorInput::from_handle)
                .collect();
            let evaluation = state.hub.evaluate(&inputs, self.clock.as_ref());
            if matches!(
                evaluation.summary.state,
                SafetyGate::Safe | SafetyGate::None
            ) {
                // The episode is over.
                state.armed = true;
                state.pending = None;
            }
            let mut trigger = None;
            let mut suppressed = Vec::new();
            for candidate in &evaluation.triggers {
                if trigger.is_none() && (state.armed || !secured) {
                    if self.try_start() {
                        state.armed = false;
                        state.pending = None;
                        trigger = Some(candidate.clone());
                        continue;
                    }
                    // A run is in progress: look again once it ends.
                    state.pending = Some(candidate.clone());
                }
                suppressed.push(candidate.clone());
            }
            if trigger.is_none() && !was_running {
                // Dropped when the run that was in progress secured the rig.
                if let Some(pending) = state.pending.take() {
                    if state.armed || !secured {
                        if self.try_start() {
                            state.armed = false;
                            trigger = Some(pending);
                        } else {
                            state.pending = Some(pending);
                        }
                    }
                }
            }
            let action = match evaluation.summary.state {
                SafetyGate::Unknown
                    if !evaluation.startup
                        && state.unknown_block.is_none()
                        && trigger.is_none() =>
                {
                    let names: Vec<String> = evaluation
                        .summary
                        .sources
                        .iter()
                        .filter(|s| s.level.is_none())
                        .map(|s| format!("{} ({:?})", s.name, s.link).to_lowercase())
                        .collect();
                    BlockAction::RaiseUnknown(format!(
                        "safety is unknown: {} not ready or stale",
                        names.join(", ")
                    ))
                }
                SafetyGate::Safe | SafetyGate::None => match state.unknown_block.take() {
                    Some(id) => BlockAction::Clear(id),
                    None => BlockAction::None,
                },
                _ => BlockAction::None,
            };
            (evaluation, trigger, suppressed, action)
        };

        // Outside the state lock: the store and the event sink may take
        // their own locks.
        match action {
            BlockAction::RaiseUnknown(reason) => {
                let unsafe_active = self
                    .blocks
                    .active()
                    .iter()
                    .any(|b| b.kind == ResumeBlockKind::SafetyUnsafe);
                if !unsafe_active {
                    let block = self
                        .blocks
                        .raise(BlockRequest::new(ResumeBlockKind::SafetyUnknown, reason));
                    lock(&self.state).unknown_block = Some(block.id);
                }
            }
            BlockAction::Clear(id) => {
                self.blocks.clear(id, &CommandSource::Safety);
            }
            BlockAction::None => {}
        }
        if let Some(event) = &evaluation.event {
            self.events.emit(event.clone());
        }
        self.summary.send_replace(evaluation.summary.clone());
        (evaluation, trigger, suppressed)
    }

    async fn execute_run(&self, trigger: PolicyTrigger) -> PolicyRun {
        let _running = RunningGuard(&self.running);
        let targets = self.targets.targets();
        let run = self.policy.run(&trigger, &targets).await;
        let mut runs = lock(&self.runs);
        runs.push(run.clone());
        if runs.len() > MAX_RUNS_KEPT {
            runs.remove(0);
        }
        run
    }

    /// One evaluation, and the policy run it triggers (awaited).
    pub async fn step(&self) -> StepReport {
        let (evaluation, trigger, suppressed) = self.assess();
        let run = match trigger {
            Some(trigger) => Some(self.execute_run(trigger).await),
            None => None,
        };
        StepReport {
            evaluation,
            run,
            suppressed,
        }
    }

    /// Evaluate every `period` on a tokio task; a triggered run is spawned
    /// so evaluation continues while it executes. The task ends once the
    /// supervisor is dropped.
    pub fn spawn(self: &Arc<Self>, period: Duration) -> tokio::task::JoinHandle<()> {
        let weak = Arc::downgrade(self);
        let period = period.max(Duration::from_millis(10));
        tokio::spawn(async move {
            let mut ticks = tokio::time::interval(period);
            ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                ticks.tick().await;
                let Some(supervisor) = weak.upgrade() else {
                    break;
                };
                let (_, trigger, _) = supervisor.assess();
                if let Some(trigger) = trigger {
                    tokio::spawn(async move {
                        supervisor.execute_run(trigger).await;
                    });
                }
            }
        })
    }

    /// The operator's manual resume (Tier 1): clears the safety blocks, but
    /// only while safety is confirmed safe or no monitor is configured, and
    /// not while the policy is still running. The route that calls it goes
    /// through the gate (`ClearResumeBlock`), which audits the request.
    pub fn request_resume(&self, by: &CommandSource) -> Result<Vec<ResumeBlock>, String> {
        let summary = self.summary();
        if !safety_block_clear_allowed(&ResumeBlockKind::SafetyUnsafe, &summary) {
            let state = serde_json::to_value(summary.state)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default();
            return Err(format!(
                "safety is {state}; safety resume blocks clear only on a confirmed safe reading"
            ));
        }
        if self.policy_running() {
            return Err("the on-unsafe policy is still running".into());
        }
        let cleared = self
            .blocks
            .active()
            .into_iter()
            .filter(|b| {
                matches!(
                    b.kind,
                    ResumeBlockKind::SafetyUnsafe | ResumeBlockKind::SafetyUnknown
                )
            })
            .filter_map(|b| self.blocks.clear(b.id, by))
            .collect();
        lock(&self.state).unknown_block = None;
        Ok(cleared)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AchernarConfig;
    use crate::equipment::alpaca::safetymonitor::SAFETY_POLL_INTERVAL;
    use crate::equipment::clock::SystemClock;
    use crate::equipment::commands::{GateError, PixelUnit, SafetyCmd, Settle};
    use crate::equipment::events::GroundTruthRecord;
    use crate::equipment::handle::{Permit, WorkerEnd};
    use crate::equipment::safety::{
        safety_snapshot, AuditingSafetyGate, MemoryResumeBlocks, RecorderEventSink,
    };
    use crate::equipment::sim::safety::{SimSafetyControl, SimSafetyMonitor};
    use crate::equipment::snapshot::{ExposureRef, MountCapabilities, MountState, SafetySnapshot};
    use crate::equipment::testing::fake_handles::{
        fake_guider, fake_mount, FakeCameraActivity, FakeDevice, FakeResponse, RecordingSink,
    };
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::traits::{Device, PollPlan, RecorderSink};
    use crate::equipment::types::{
        DeviceIdentity, DeviceResult, DriverId, Endpoint, HealthState, LinkState, ObsState,
        Observed, Source,
    };
    use futures::future::BoxFuture;

    type Order = Arc<Mutex<Vec<String>>>;

    /// A camera hook that logs its abort into the shared order.
    struct OrderedCamera {
        inner: Arc<FakeCameraActivity>,
        order: Order,
    }

    impl CameraActivity for OrderedCamera {
        fn exposing(&self) -> Observed<bool> {
            self.inner.exposing()
        }
        fn current_exposure(&self) -> Option<ExposureRef> {
            self.inner.current_exposure()
        }
        fn abort_exposure(&self) -> DeviceResult<()> {
            lock(&self.order).push("camera_abort".into());
            self.inner.abort_exposure()
        }
        fn camera_name(&self) -> Option<String> {
            self.inner.camera_name()
        }
    }

    struct Rig {
        clock: Arc<dyn Clock>,
        recorder: Arc<RecordingSink>,
        events: Arc<dyn EventSink>,
        blocks: Arc<MemoryResumeBlocks>,
        order: Order,
        mount: MountHandle,
        mount_fake: Arc<FakeDevice<MountCmd, MountSnapshot>>,
        guider: GuiderHandle,
        guider_fake: FakeDevice<GuiderCmd, GuiderSnapshot>,
        camera: Arc<FakeCameraActivity>,
        camera_hook: Arc<dyn CameraActivity>,
    }

    impl Rig {
        /// A guiding guider, an exposing camera, and a slewing, tracking,
        /// unparked mount that can park.
        fn new(clock: Arc<dyn Clock>) -> Self {
            let recorder = RecordingSink::new();
            let events: Arc<dyn EventSink> = Arc::new(RecorderEventSink::new(recorder.clone()));
            let blocks = Arc::new(MemoryResumeBlocks::new(
                Arc::clone(&events),
                Arc::clone(&clock),
            ));
            let order: Order = Arc::new(Mutex::new(Vec::new()));
            let fresh_bool = |v: bool| Observed::fresh_at(v, Source::DeviceState, clock.as_ref());

            let (mount, mount_fake) = fake_mount(MountSnapshot {
                state: MountState {
                    slewing: fresh_bool(true),
                    tracking: fresh_bool(true),
                    at_park: fresh_bool(false),
                    ..MountState::default()
                },
                capabilities: MountCapabilities {
                    can_park: true,
                    can_set_tracking: true,
                    ..MountCapabilities::default()
                },
                ..MountSnapshot::default()
            });
            let mount_fake = Arc::new(mount_fake);
            {
                let order = Arc::clone(&order);
                let abort = mount_fake.abort_path();
                let clock = Arc::clone(&clock);
                mount_fake.set_responder(move |cmd, snap: &mut MountSnapshot| {
                    match cmd {
                        MountCmd::Park => {
                            lock(&order).push(format!("mount_park(aborts={})", abort.calls()));
                            snap.state.at_park =
                                Observed::fresh_at(true, Source::DeviceState, clock.as_ref());
                            snap.state.tracking =
                                Observed::fresh_at(false, Source::DeviceState, clock.as_ref());
                        }
                        MountCmd::SetTracking(on) => {
                            lock(&order).push(format!("mount_tracking({on})"));
                            snap.state.tracking =
                                Observed::fresh_at(*on, Source::DeviceState, clock.as_ref());
                        }
                        other => lock(&order).push(format!("mount_{other:?}")),
                    }
                    FakeResponse::succeed()
                });
            }

            let (guider, guider_fake) = fake_guider(GuiderSnapshot {
                app_state: Observed::fresh_at(
                    GuiderAppState::Guiding,
                    Source::Phd2Event,
                    clock.as_ref(),
                ),
                ..GuiderSnapshot::default()
            });
            {
                let order = Arc::clone(&order);
                let clock = Arc::clone(&clock);
                guider_fake.set_responder(move |cmd, snap: &mut GuiderSnapshot| {
                    match cmd {
                        GuiderCmd::Stop => {
                            lock(&order).push("guider_stop".into());
                            snap.app_state = Observed::fresh_at(
                                GuiderAppState::Stopped,
                                Source::Phd2Event,
                                clock.as_ref(),
                            );
                        }
                        GuiderCmd::Dither { .. } => return FakeResponse::hold(),
                        other => lock(&order).push(format!("guider_{other:?}")),
                    }
                    FakeResponse::succeed()
                });
            }

            let camera = FakeCameraActivity::new();
            camera.start_exposure(ExposureRef {
                duration_s: Some(300.0),
                ..ExposureRef::default()
            });
            let camera_hook: Arc<dyn CameraActivity> = Arc::new(OrderedCamera {
                inner: Arc::clone(&camera),
                order: Arc::clone(&order),
            });
            Self {
                clock,
                recorder,
                events,
                blocks,
                order,
                mount,
                mount_fake,
                guider,
                guider_fake,
                camera,
                camera_hook,
            }
        }

        fn targets(&self) -> PolicyTargets {
            PolicyTargets {
                mount: Some(self.mount.clone()),
                guider: Some(self.guider.clone()),
                camera: Some(Arc::clone(&self.camera_hook)),
            }
        }

        fn gate(&self) -> Arc<dyn SafetyGatePort> {
            Arc::new(AuditingSafetyGate::new(
                self.recorder.clone(),
                Arc::clone(&self.clock),
            ))
        }

        fn policy(&self, settings: PolicySettings) -> UnsafePolicy {
            self.policy_with_gate(settings, self.gate())
        }

        fn policy_with_gate(
            &self,
            settings: PolicySettings,
            gate: Arc<dyn SafetyGatePort>,
        ) -> UnsafePolicy {
            UnsafePolicy::new(
                settings,
                gate,
                Arc::clone(&self.blocks) as Arc<dyn ResumeBlockStore>,
                Arc::clone(&self.events),
                Arc::clone(&self.clock),
            )
        }

        fn supervisor(&self, settings: PolicySettings) -> Arc<SafetySupervisor> {
            SafetySupervisor::new(
                HubSettings::default(),
                self.policy(settings),
                Arc::new(self.targets()),
                Arc::clone(&self.blocks) as Arc<dyn ResumeBlockStore>,
                Arc::clone(&self.events),
                Arc::clone(&self.clock),
            )
        }

        fn order(&self) -> Vec<String> {
            lock(&self.order).clone()
        }

        /// `(op, stage, accepted)` of every audit row, in order.
        fn audits(&self) -> Vec<(OpKind, AuditStage, bool)> {
            self.recorder
                .records()
                .into_iter()
                .filter_map(|record| match record {
                    GroundTruthRecord::Audit(a) => Some((a.op, a.stage, a.accepted)),
                    _ => None,
                })
                .collect()
        }

        fn block_kinds(&self) -> Vec<ResumeBlockKind> {
            self.blocks.active().into_iter().map(|b| b.kind).collect()
        }

        /// The fake mount does not model motion: once the abort's cancel
        /// reaches it, report the slew stopped.
        fn stop_mount_on_cancel(&self) -> std::thread::JoinHandle<()> {
            let fake = Arc::clone(&self.mount_fake);
            let clock = Arc::clone(&self.clock);
            let before = fake.cancels().len();
            std::thread::spawn(move || {
                if fake.wait_for_cancels(before + 1, Duration::from_secs(10)) {
                    fake.update(|snap| {
                        snap.state.slewing =
                            Observed::fresh_at(false, Source::DeviceState, clock.as_ref());
                    });
                }
            })
        }
    }

    fn unsafe_trigger() -> PolicyTrigger {
        PolicyTrigger::ConfirmedUnsafe {
            connection_id: Uuid::new_v4(),
            name: "roof sensor".into(),
            level: crate::equipment::snapshot::SafetyLevel::Unsafe,
            detail: Some("rain".into()),
        }
    }

    fn stages(op: OpKind) -> Vec<(OpKind, AuditStage, bool)> {
        [
            AuditStage::Proposed,
            AuditStage::Validated,
            AuditStage::Executed,
            AuditStage::Outcome,
        ]
        .into_iter()
        .map(|stage| (op, stage, true))
        .collect()
    }

    #[tokio::test]
    async fn the_unsafe_edge_stops_guiding_aborts_the_exposure_and_slew_then_parks_in_order() {
        let rig = Rig::new(Arc::new(ManualClock::new()));
        let stopper = rig.stop_mount_on_cancel();
        let run = rig
            .policy(PolicySettings::default())
            .run(&unsafe_trigger(), &rig.targets())
            .await;
        stopper.join().unwrap();

        // Device-side order: guider, camera, then the park, which arrives
        // after the out-of-band abort path ran once.
        assert_eq!(
            rig.order(),
            vec!["guider_stop", "camera_abort", "mount_park(aborts=1)"]
        );
        assert_eq!(rig.mount_fake.cancels().len(), 1);
        assert_eq!(rig.mount_fake.commands(), vec![MountCmd::Park]);
        assert_eq!(rig.guider_fake.commands(), vec![GuiderCmd::Stop]);
        assert_eq!(rig.camera.aborts(), 1);
        assert_eq!(
            run.done(),
            vec![
                PolicyAction::RaiseResumeBlock,
                PolicyAction::StopGuiding,
                PolicyAction::AbortExposure,
                PolicyAction::AbortSlew,
                PolicyAction::Park,
            ]
        );
        assert!(run.step(PolicyAction::StopTracking).is_none());
        assert_eq!(run.on_unsafe, "park");

        // Audited per stage, in the same order.
        let expected: Vec<_> = [
            OpKind::GuiderStop,
            OpKind::Abort,
            OpKind::AbortSlew,
            OpKind::Park,
        ]
        .into_iter()
        .flat_map(stages)
        .collect();
        assert_eq!(rig.audits(), expected);
        for record in rig.recorder.records() {
            if let GroundTruthRecord::Audit(a) = record {
                assert_eq!(a.source, CommandSource::Safety);
                assert_eq!(a.priority, Priority::Safety);
                assert_eq!(a.params["policy"], json!(POLICY_NAME));
                assert_eq!(a.params["trigger"]["trigger"], json!("confirmed_unsafe"));
            }
        }
        assert_eq!(rig.block_kinds(), vec![ResumeBlockKind::SafetyUnsafe]);
        assert_eq!(run.block_id, Some(rig.blocks.active()[0].id));
    }

    #[tokio::test]
    async fn a_mount_without_can_park_stops_tracking_instead() {
        let rig = Rig::new(Arc::new(ManualClock::new()));
        rig.mount_fake.update(|snap| {
            snap.capabilities.can_park = false;
            snap.state.slewing = Observed::fresh_at(false, Source::DeviceState, rig.clock.as_ref());
        });
        let run = rig
            .policy(PolicySettings::default())
            .run(&unsafe_trigger(), &rig.targets())
            .await;
        assert_eq!(
            rig.mount_fake.commands(),
            vec![MountCmd::SetTracking(false)]
        );
        assert!(rig.mount_fake.cancels().is_empty(), "not slewing: no abort");
        assert!(matches!(
            run.step(PolicyAction::Park).unwrap().status,
            StepStatus::Skipped { ref reason } if reason.contains("CanPark is false")
        ));
        assert!(run
            .step(PolicyAction::StopTracking)
            .unwrap()
            .status
            .is_done());
        assert_eq!(
            rig.order(),
            vec!["guider_stop", "camera_abort", "mount_tracking(false)"]
        );
    }

    #[tokio::test]
    async fn on_unsafe_stop_tracking_and_notify_follow_the_setting() {
        let rig = Rig::new(Arc::new(ManualClock::new()));
        let stopper = rig.stop_mount_on_cancel();
        let settings = PolicySettings {
            on_unsafe: OnUnsafe::StopTracking,
            abort_exposure: false,
            ..PolicySettings::default()
        };
        let run = rig
            .policy(settings)
            .run(&unsafe_trigger(), &rig.targets())
            .await;
        stopper.join().unwrap();
        assert_eq!(
            rig.mount_fake.commands(),
            vec![MountCmd::SetTracking(false)]
        );
        assert_eq!(rig.camera.aborts(), 0, "abort_exposure_on_unsafe = false");
        assert!(run.step(PolicyAction::AbortSlew).unwrap().status.is_done());

        let rig = Rig::new(Arc::new(ManualClock::new()));
        let settings = PolicySettings {
            on_unsafe: OnUnsafe::Notify,
            ..PolicySettings::default()
        };
        let run = rig
            .policy(settings)
            .run(&unsafe_trigger(), &rig.targets())
            .await;
        assert!(rig.mount_fake.commands().is_empty());
        assert!(rig.mount_fake.cancels().is_empty());
        assert_eq!(rig.mount_fake.abort_path().calls(), 0);
        assert_eq!(rig.guider_fake.commands(), vec![GuiderCmd::Stop]);
        assert_eq!(rig.camera.aborts(), 1);
        assert!(!run.step(PolicyAction::Park).unwrap().status.is_done());
        assert_eq!(rig.block_kinds(), vec![ResumeBlockKind::SafetyUnsafe]);
    }

    fn park_failed_events(rig: &Rig) -> Vec<Arc<EquipmentEvent>> {
        rig.recorder
            .records()
            .into_iter()
            .filter_map(|r| match r {
                GroundTruthRecord::Event(e) if e.name() == "park_failed" => Some(e),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn a_park_the_worker_refused_emits_park_failed_once_from_the_policy() {
        // O11 c: the park operation never started, so the policy reports it.
        let rig = Rig::new(Arc::new(ManualClock::new()));
        rig.mount_fake
            .set_responder(|cmd, _snap: &mut MountSnapshot| match cmd {
                MountCmd::Park => {
                    FakeResponse::reject(DeviceError::driver("the park initiator was refused"))
                }
                _ => FakeResponse::succeed(),
            });
        let run = rig
            .policy(PolicySettings::default())
            .run(&unsafe_trigger(), &rig.targets())
            .await;
        assert!(matches!(
            run.step(PolicyAction::Park).unwrap().status,
            StepStatus::Failed { .. }
        ));
        let events = park_failed_events(&rig);
        assert_eq!(events.len(), 1, "park_failed is emitted exactly once");
        assert_eq!(
            events[0].command_id,
            run.step(PolicyAction::Park).unwrap().command_id
        );
        assert_eq!(
            rig.block_kinds(),
            vec![ResumeBlockKind::SafetyUnsafe, ResumeBlockKind::MountLost]
        );
    }

    #[tokio::test]
    async fn a_failed_park_raises_mount_lost_stops_tracking_and_leaves_park_failed_to_the_op() {
        let rig = Rig::new(Arc::new(ManualClock::new()));
        rig.mount_fake
            .set_responder(|cmd, _snap: &mut MountSnapshot| match cmd {
                MountCmd::Park => FakeResponse::finish(OpOutcome::Failed(
                    DeviceError::from_alpaca_error(0x500, "park sensor fault"),
                )),
                _ => FakeResponse::succeed(),
            });
        let stopper = rig.stop_mount_on_cancel();
        let run = rig
            .policy(PolicySettings::default())
            .run(&unsafe_trigger(), &rig.targets())
            .await;
        stopper.join().unwrap();
        assert!(matches!(
            run.step(PolicyAction::Park).unwrap().status,
            StepStatus::Failed { .. }
        ));
        assert!(run
            .step(PolicyAction::StopTracking)
            .unwrap()
            .status
            .is_done());
        assert_eq!(
            rig.mount_fake.commands(),
            vec![MountCmd::Park, MountCmd::SetTracking(false)]
        );
        assert_eq!(
            rig.block_kinds(),
            vec![ResumeBlockKind::SafetyUnsafe, ResumeBlockKind::MountLost]
        );
        // The park operation ran (the fake stands in for it), so the
        // operation reports `park_failed`, never the policy too (O11 c).
        assert!(park_failed_events(&rig).is_empty());
        // The park's outcome row records the failure.
        assert!(rig
            .audits()
            .contains(&(OpKind::Park, AuditStage::Outcome, false)));
    }

    #[tokio::test]
    async fn a_parked_or_parking_mount_and_a_stopped_guider_are_left_alone() {
        let rig = Rig::new(Arc::new(ManualClock::new()));
        let at = rig.clock.as_ref();
        rig.mount_fake.update(|snap| {
            snap.state.slewing = Observed::fresh_at(false, Source::DeviceState, at);
            snap.state.at_park = Observed::fresh_at(true, Source::DeviceState, at);
        });
        rig.guider_fake.update(|snap| {
            snap.app_state = Observed::fresh_at(GuiderAppState::Stopped, Source::Phd2Event, at);
        });
        rig.camera.finish_exposure();
        let run = rig
            .policy(PolicySettings::default())
            .run(&unsafe_trigger(), &rig.targets())
            .await;
        assert_eq!(run.done(), vec![PolicyAction::RaiseResumeBlock]);
        assert!(rig.mount_fake.commands().is_empty());
        assert!(rig.guider_fake.commands().is_empty());
        assert_eq!(rig.camera.aborts(), 0);

        // A park already in progress is not aborted or repeated.
        let rig = Rig::new(Arc::new(ManualClock::new()));
        let parking = CommandId::new();
        rig.mount_fake.update(|snap| {
            snap.active_op = Some(crate::equipment::ops::ActiveOp {
                command_id: parking,
                kind: OpKind::Park,
                phase: crate::equipment::ops::OpPhase::InProgress,
                started_at: Utc::now(),
                started_mono_ns: 0,
                deadline_mono_ns: u64::MAX,
            });
        });
        let run = rig
            .policy(PolicySettings::default())
            .run(&unsafe_trigger(), &rig.targets())
            .await;
        assert!(rig.mount_fake.commands().is_empty());
        assert_eq!(rig.mount_fake.abort_path().calls(), 0);
        assert!(matches!(
            &run.step(PolicyAction::Park).unwrap().status,
            StepStatus::Skipped { reason } if reason.contains(&parking.to_string())
        ));
    }

    #[tokio::test]
    async fn a_busy_guider_is_cancelled_out_of_band_and_the_stop_retried() {
        let rig = Rig::new(Arc::new(ManualClock::new()));
        rig.camera.finish_exposure();
        rig.mount_fake.update(|snap| {
            snap.state.slewing = Observed::fresh_at(false, Source::DeviceState, rig.clock.as_ref());
        });
        // A dither holds the guider; no preemption from the stand-in gate.
        let dither = rig
            .guider
            .send(
                GuiderCmd::Dither {
                    amount: 5.0,
                    unit: PixelUnit::GuidePx,
                    ra_only: false,
                    settle: Settle::default(),
                },
                Permit::unmanaged(CommandId::new(), Priority::Exposure, CommandSource::System),
            )
            .await
            .unwrap();
        assert!(rig.guider_fake.has_pending());
        let run = rig
            .policy(PolicySettings::default())
            .run(&unsafe_trigger(), &rig.targets())
            .await;
        assert!(run
            .step(PolicyAction::StopGuiding)
            .unwrap()
            .status
            .is_done());
        assert_eq!(rig.guider_fake.cancels().len(), 1);
        assert!(matches!(dither.wait().await, OpOutcome::Aborted { .. }));
        assert_eq!(rig.guider_fake.commands().last(), Some(&GuiderCmd::Stop));
        let guider_audits: Vec<_> = rig
            .audits()
            .into_iter()
            .filter(|(op, _, _)| *op == OpKind::GuiderStop)
            .map(|(_, stage, accepted)| (stage, accepted))
            .collect();
        assert_eq!(
            guider_audits,
            vec![
                (AuditStage::Proposed, true),
                (AuditStage::Validated, true),
                (AuditStage::Executed, false),
                (AuditStage::Proposed, true),
                (AuditStage::Validated, true),
                (AuditStage::Executed, true),
                (AuditStage::Outcome, true),
            ]
        );
    }

    /// A gate that refuses everything, aborts included.
    struct RefusingGate(Arc<RecordingSink>);

    impl SafetyGatePort for RefusingGate {
        fn authorize<'a>(
            &'a self,
            _cmd: &'a GatedCommand,
        ) -> BoxFuture<'a, Result<Permit, GateError>> {
            Box::pin(async {
                Err(GateError::Refused {
                    reason: "test gate".into(),
                })
            })
        }
        fn audit(&self, record: AuditRecord) {
            self.0.record(GroundTruthRecord::Audit(Box::new(record)));
        }
    }

    #[tokio::test]
    async fn aborts_are_attempted_even_when_the_gate_refuses() {
        let rig = Rig::new(Arc::new(ManualClock::new()));
        let stopper = rig.stop_mount_on_cancel();
        let gate = Arc::new(RefusingGate(rig.recorder.clone()));
        let run = rig
            .policy_with_gate(PolicySettings::default(), gate)
            .run(&unsafe_trigger(), &rig.targets())
            .await;
        stopper.join().unwrap();
        assert!(rig.guider_fake.commands().is_empty());
        assert!(rig.mount_fake.commands().is_empty());
        assert_eq!(rig.camera.aborts(), 1);
        assert_eq!(rig.mount_fake.abort_path().calls(), 1);
        assert!(run
            .step(PolicyAction::AbortExposure)
            .unwrap()
            .status
            .is_done());
        assert!(run.step(PolicyAction::AbortSlew).unwrap().status.is_done());
        for action in [
            PolicyAction::StopGuiding,
            PolicyAction::Park,
            PolicyAction::StopTracking,
        ] {
            assert!(
                matches!(run.step(action).unwrap().status, StepStatus::Failed { .. }),
                "{action:?}"
            );
        }
        assert!(rig.block_kinds().contains(&ResumeBlockKind::MountLost));
    }

    #[tokio::test]
    async fn nothing_connected_still_raises_the_block() {
        let rig = Rig::new(Arc::new(ManualClock::new()));
        let run = rig
            .policy(PolicySettings::default())
            .run(&unsafe_trigger(), &PolicyTargets::default())
            .await;
        assert_eq!(run.done(), vec![PolicyAction::RaiseResumeBlock]);
        assert_eq!(run.steps.len(), 5);
        assert!(rig.audits().is_empty());
        assert_eq!(rig.block_kinds(), vec![ResumeBlockKind::SafetyUnsafe]);
    }

    // -- Supervisor, end to end through the sim safety monitor --------------

    /// Stands in for the WP1A-2 worker: one tick polls the sim, publishes
    /// the snapshot, and applies the §2.8 health rules (degraded on a failed
    /// poll, lost after `lost_after_failures` = 3).
    struct MonitorHarness {
        driver: SimSafetyMonitor,
        end: WorkerEnd<SafetyCmd, SafetySnapshot>,
        handle: SafetyMonitorHandle,
        clock: Arc<dyn Clock>,
        failures: u32,
    }

    impl MonitorHarness {
        fn new(clock: Arc<dyn Clock>, control: SimSafetyControl, name: &str) -> Self {
            let identity = DeviceIdentity::new(
                DeviceKind::SafetyMonitor,
                DriverId::Sim,
                Endpoint::InProcess { name: name.into() },
                format!("sim:safety_monitor/{name}"),
                name,
            );
            let mut driver = SimSafetyMonitor::with_identity(identity, Arc::clone(&clock), control);
            driver.connect(Duration::from_secs(1)).unwrap();
            let identity = driver.identity().clone();
            let initial = SafetySnapshot {
                connection_id: identity.connection_id,
                ..SafetySnapshot::default()
            };
            let health = HealthState::ready(clock.now_utc(), clock.now_mono_ns());
            let (handle, end) = DeviceHandle::new_pair(identity, None, initial, health);
            Self {
                driver,
                end,
                handle,
                clock,
                failures: 0,
            }
        }

        fn tick(&mut self) {
            let poll = self.driver.poll(&PollPlan::default());
            let id = self.handle.connection_id();
            let snapshot = safety_snapshot(&mut self.driver, &poll, id, self.clock.as_ref());
            self.end.publish(snapshot);
            let now_utc = self.clock.now_utc();
            let now_mono = self.clock.now_mono_ns();
            match poll {
                Ok(_) => {
                    self.failures = 0;
                    self.end.set_health(HealthState::ready(now_utc, now_mono));
                }
                Err(err) => {
                    self.failures += 1;
                    let failures = self.failures;
                    self.end.update_health(|health| {
                        health.state = if failures >= 3 {
                            LinkState::Lost
                        } else {
                            LinkState::Degraded
                        };
                        health.consecutive_failures = failures;
                        health.failure_class = err.class.failure_class();
                        health.reason = Some(err.to_string());
                        health.changed_mono_ns = now_mono;
                    });
                }
            }
        }
    }

    fn operator() -> CommandSource {
        CommandSource::Operator { overriding: false }
    }

    #[tokio::test]
    async fn end_to_end_one_degraded_poll_never_parks_and_a_confirmed_unsafe_edge_does() {
        let manual = Arc::new(ManualClock::new());
        let rig = Rig::new(Arc::clone(&manual) as Arc<dyn Clock>);
        let control = SimSafetyControl::new();
        let mut monitor = MonitorHarness::new(Arc::clone(&rig.clock), control.clone(), "roof");
        let supervisor = rig.supervisor(PolicySettings::default());
        supervisor.add_monitor(monitor.handle.clone());

        // Before the first reading: unknown during startup, no block.
        let step = supervisor.step().await;
        assert_eq!(step.evaluation.summary.state, SafetyGate::Unknown);
        assert!(step.evaluation.startup);
        assert!(rig.block_kinds().is_empty());

        monitor.tick();
        let step = supervisor.step().await;
        assert_eq!(step.evaluation.summary.state, SafetyGate::Safe);
        assert!(step.run.is_none());

        // One failed poll (a Wi-Fi blip): degraded, unknown, never parks.
        control.fail_next_reads(1);
        manual.advance(SAFETY_POLL_INTERVAL);
        monitor.tick();
        assert_eq!(monitor.handle.health().state, LinkState::Degraded);
        let step = supervisor.step().await;
        assert_eq!(step.evaluation.summary.state, SafetyGate::Unknown);
        assert!(step.run.is_none() && step.evaluation.triggers.is_empty());
        assert!(!supervisor.summary().state.allows_automatic());
        assert_eq!(rig.block_kinds(), vec![ResumeBlockKind::SafetyUnknown]);
        assert!(rig.mount_fake.commands().is_empty());
        assert!(rig.guider_fake.commands().is_empty());
        assert_eq!(rig.camera.aborts(), 0);
        assert_eq!(rig.mount_fake.abort_path().calls(), 0);

        // The next good poll clears the unknown block.
        manual.advance(SAFETY_POLL_INTERVAL);
        monitor.tick();
        assert_eq!(
            supervisor.step().await.evaluation.summary.state,
            SafetyGate::Safe
        );
        assert!(rig.block_kinds().is_empty());

        // A confirmed unsafe reading runs the policy, in order and audited.
        control.set_safe(false);
        control.set_detail(Some("rain"));
        manual.advance(SAFETY_POLL_INTERVAL);
        monitor.tick();
        let stopper = rig.stop_mount_on_cancel();
        let step = supervisor.step().await;
        stopper.join().unwrap();
        let run = step
            .run
            .expect("the policy ran on the confirmed unsafe edge");
        assert!(matches!(
            &run.trigger,
            PolicyTrigger::ConfirmedUnsafe { detail: Some(d), .. } if d == "rain"
        ));
        assert_eq!(
            rig.order(),
            vec!["guider_stop", "camera_abort", "mount_park(aborts=1)"]
        );
        assert_eq!(supervisor.summary().state, SafetyGate::Unsafe);
        assert_eq!(rig.block_kinds(), vec![ResumeBlockKind::SafetyUnsafe]);
        assert_eq!(supervisor.runs().len(), 1);

        // Still unsafe on the next poll: latched, no second run.
        manual.advance(SAFETY_POLL_INTERVAL);
        monitor.tick();
        assert!(supervisor.step().await.run.is_none());
        assert_eq!(rig.mount_fake.commands(), vec![MountCmd::Park]);

        // Resume is manual and refused while unsafe.
        assert!(supervisor.request_resume(&operator()).is_err());
        control.set_safe(true);
        manual.advance(SAFETY_POLL_INTERVAL);
        monitor.tick();
        assert_eq!(
            supervisor.step().await.evaluation.summary.state,
            SafetyGate::Safe
        );
        assert_eq!(
            rig.block_kinds(),
            vec![ResumeBlockKind::SafetyUnsafe],
            "a safe reading alone does not resume (Tier 1 is manual)"
        );
        let cleared = supervisor.request_resume(&operator()).unwrap();
        assert_eq!(cleared.len(), 1);
        assert!(rig.block_kinds().is_empty());

        // Re-armed: a new edge runs again, and leaves the parked, stopped
        // rig alone.
        control.set_safe(false);
        manual.advance(SAFETY_POLL_INTERVAL);
        monitor.tick();
        let run = supervisor.step().await.run.expect("re-armed after safe");
        assert_eq!(run.done(), vec![PolicyAction::RaiseResumeBlock]);
        assert_eq!(rig.mount_fake.commands(), vec![MountCmd::Park]);

        let names = rig.recorder.event_names();
        for expected in ["safety_changed", "resume_blocked", "resume_cleared"] {
            assert!(names.contains(&expected), "{expected} in {names:?}");
        }
    }

    #[tokio::test]
    async fn a_lost_monitor_runs_the_policy_only_after_the_loss_threshold() {
        let manual = Arc::new(ManualClock::new());
        let rig = Rig::new(Arc::clone(&manual) as Arc<dyn Clock>);
        let control = SimSafetyControl::new();
        let mut monitor = MonitorHarness::new(Arc::clone(&rig.clock), control.clone(), "allsky");
        let supervisor = rig.supervisor(PolicySettings::default());
        supervisor.add_monitor(monitor.handle.clone());
        monitor.tick();
        supervisor.step().await;

        control.set_failing(true);
        for _ in 0..2 {
            manual.advance(SAFETY_POLL_INTERVAL);
            monitor.tick();
            assert_eq!(monitor.handle.health().state, LinkState::Degraded);
            assert!(
                supervisor.step().await.run.is_none(),
                "degraded never parks"
            );
        }
        assert!(rig.mount_fake.commands().is_empty());
        manual.advance(SAFETY_POLL_INTERVAL);
        monitor.tick();
        assert_eq!(monitor.handle.health().state, LinkState::Lost);
        let stopper = rig.stop_mount_on_cancel();
        let run = supervisor.step().await.run.expect("lost runs the policy");
        stopper.join().unwrap();
        assert!(matches!(run.trigger, PolicyTrigger::MonitorLost { .. }));
        assert_eq!(rig.mount_fake.commands(), vec![MountCmd::Park]);
    }

    #[tokio::test]
    async fn a_new_edge_parks_again_when_the_rig_was_unparked_during_the_episode() {
        let manual = Arc::new(ManualClock::new());
        let rig = Rig::new(Arc::clone(&manual) as Arc<dyn Clock>);
        let control = SimSafetyControl::new();
        let mut monitor = MonitorHarness::new(Arc::clone(&rig.clock), control.clone(), "roof");
        let supervisor = rig.supervisor(PolicySettings::default());
        supervisor.add_monitor(monitor.handle.clone());
        monitor.tick();
        supervisor.step().await;

        // Unsafe: the policy parks.
        control.set_safe(false);
        manual.advance(SAFETY_POLL_INTERVAL);
        monitor.tick();
        let stopper = rig.stop_mount_on_cancel();
        supervisor.step().await.run.expect("the first edge runs");
        stopper.join().unwrap();
        assert_eq!(rig.mount_fake.commands(), vec![MountCmd::Park]);

        // The monitor is lost. The rig is secured, so the Lost edge is
        // latched rather than run again.
        control.set_failing(true);
        for _ in 0..3 {
            manual.advance(SAFETY_POLL_INTERVAL);
            monitor.tick();
        }
        assert_eq!(monitor.handle.health().state, LinkState::Lost);
        let step = supervisor.step().await;
        assert_eq!(step.evaluation.summary.state, SafetyGate::Unknown);
        assert!(step.run.is_none());
        assert!(matches!(
            step.suppressed.as_slice(),
            [PolicyTrigger::MonitorLost { .. }]
        ));

        // While safety is unknown the operator unparks and tracks again
        // (operator commands pass the safety check).
        rig.mount_fake.update(|snap| {
            snap.state.at_park = Observed::fresh_at(false, Source::DeviceState, rig.clock.as_ref());
            snap.state.tracking = Observed::fresh_at(true, Source::DeviceState, rig.clock.as_ref());
        });

        // The link recovers and the sky is still unsafe: the new confirmed
        // edge finds the rig unsecured and parks it again.
        control.set_failing(false);
        manual.advance(SAFETY_POLL_INTERVAL);
        monitor.tick();
        let step = supervisor.step().await;
        let run = step.run.expect("an unsecured rig is secured again");
        assert!(matches!(run.trigger, PolicyTrigger::ConfirmedUnsafe { .. }));
        assert!(run.step(PolicyAction::Park).unwrap().status.is_done());
        assert_eq!(
            rig.mount_fake.commands(),
            vec![MountCmd::Park, MountCmd::Park]
        );
        assert_eq!(supervisor.runs().len(), 2);

        // Secured again: a later edge in the same episode is latched.
        control.fail_next_reads(1);
        manual.advance(SAFETY_POLL_INTERVAL);
        monitor.tick();
        assert!(supervisor.step().await.run.is_none());
        manual.advance(SAFETY_POLL_INTERVAL);
        monitor.tick();
        let step = supervisor.step().await;
        assert!(step.run.is_none());
        assert_eq!(step.suppressed.len(), 1);
        assert_eq!(rig.mount_fake.commands().len(), 2);

        // on_unsafe = stop_tracking: tracking turned back on while unknown
        // is stopped again on the next edge.
        let rig = Rig::new(Arc::clone(&manual) as Arc<dyn Clock>);
        rig.mount_fake.update(|snap| {
            snap.state.slewing = Observed::fresh_at(false, Source::DeviceState, rig.clock.as_ref());
        });
        let control = SimSafetyControl::new();
        let mut monitor = MonitorHarness::new(Arc::clone(&rig.clock), control.clone(), "rain");
        let supervisor = rig.supervisor(PolicySettings {
            on_unsafe: OnUnsafe::StopTracking,
            ..PolicySettings::default()
        });
        supervisor.add_monitor(monitor.handle.clone());
        monitor.tick();
        supervisor.step().await;
        control.set_safe(false);
        manual.advance(SAFETY_POLL_INTERVAL);
        monitor.tick();
        supervisor.step().await.run.expect("the first edge runs");
        control.fail_next_reads(1);
        manual.advance(SAFETY_POLL_INTERVAL);
        monitor.tick();
        assert!(supervisor.step().await.run.is_none(), "degraded");
        rig.mount_fake.update(|snap| {
            snap.state.tracking = Observed::fresh_at(true, Source::DeviceState, rig.clock.as_ref());
        });
        manual.advance(SAFETY_POLL_INTERVAL);
        monitor.tick();
        supervisor
            .step()
            .await
            .run
            .expect("tracking was turned back on");
        assert_eq!(
            rig.mount_fake.commands(),
            vec![MountCmd::SetTracking(false), MountCmd::SetTracking(false)]
        );
    }

    #[tokio::test]
    async fn a_trigger_during_a_run_is_retried_once_the_run_ends() {
        let manual = Arc::new(ManualClock::new());
        let rig = Rig::new(Arc::clone(&manual) as Arc<dyn Clock>);
        let control = SimSafetyControl::new();
        let mut monitor = MonitorHarness::new(Arc::clone(&rig.clock), control.clone(), "roof");
        let supervisor = rig.supervisor(PolicySettings::default());
        supervisor.add_monitor(monitor.handle.clone());
        monitor.tick();
        supervisor.step().await;

        // Hold the running flag as a run in progress would.
        supervisor.running.store(true, Ordering::Release);
        control.set_safe(false);
        manual.advance(SAFETY_POLL_INTERVAL);
        monitor.tick();
        let step = supervisor.step().await;
        assert!(step.run.is_none());
        assert_eq!(step.suppressed.len(), 1);
        manual.advance(SAFETY_POLL_INTERVAL);
        monitor.tick();
        assert!(supervisor.step().await.run.is_none(), "still running");
        assert!(rig.mount_fake.commands().is_empty());

        // The run ends with the rig still unsecured: the kept trigger runs.
        supervisor.running.store(false, Ordering::Release);
        manual.advance(SAFETY_POLL_INTERVAL);
        monitor.tick();
        let stopper = rig.stop_mount_on_cancel();
        let run = supervisor
            .step()
            .await
            .run
            .expect("the kept trigger runs after the run");
        stopper.join().unwrap();
        assert!(matches!(run.trigger, PolicyTrigger::ConfirmedUnsafe { .. }));
        assert_eq!(rig.mount_fake.commands(), vec![MountCmd::Park]);

        // Once only.
        manual.advance(SAFETY_POLL_INTERVAL);
        monitor.tick();
        assert!(supervisor.step().await.run.is_none());
        assert_eq!(supervisor.runs().len(), 1);
    }

    #[tokio::test]
    async fn value_not_set_is_unknown_but_never_a_lost_link() {
        let manual = Arc::new(ManualClock::new());
        let rig = Rig::new(Arc::clone(&manual) as Arc<dyn Clock>);
        let control = SimSafetyControl::new();
        let mut monitor = MonitorHarness::new(Arc::clone(&rig.clock), control.clone(), "allsky");
        let supervisor = rig.supervisor(PolicySettings::default());
        supervisor.add_monitor(monitor.handle.clone());
        monitor.tick();
        supervisor.step().await;

        // A monitor that answers ValueNotSet: unknown, gated fail-safe, but
        // the link stays Ready, so it never goes Lost and never parks.
        control.set_not_set(true);
        for _ in 0..5 {
            manual.advance(SAFETY_POLL_INTERVAL);
            monitor.tick();
            assert_eq!(monitor.handle.health().state, LinkState::Ready);
            assert_eq!(monitor.handle.snapshot().level.state, ObsState::NotSet);
            assert_eq!(monitor.handle.snapshot().level.value, None);
            let step = supervisor.step().await;
            assert_eq!(step.evaluation.summary.state, SafetyGate::Unknown);
            assert!(step.run.is_none() && step.evaluation.triggers.is_empty());
        }
        assert!(!supervisor.summary().state.allows_automatic());
        assert_eq!(rig.block_kinds(), vec![ResumeBlockKind::SafetyUnknown]);
        assert!(rig.mount_fake.commands().is_empty());
        assert!(rig.guider_fake.commands().is_empty());

        control.set_not_set(false);
        manual.advance(SAFETY_POLL_INTERVAL);
        monitor.tick();
        assert_eq!(
            supervisor.step().await.evaluation.summary.state,
            SafetyGate::Safe
        );
        assert!(rig.block_kinds().is_empty());
    }

    #[tokio::test]
    async fn with_no_monitor_the_state_is_none_and_nothing_runs() {
        let rig = Rig::new(Arc::new(ManualClock::new()));
        let supervisor = rig.supervisor(PolicySettings::default());
        let step = supervisor.step().await;
        assert_eq!(step.evaluation.summary.state, SafetyGate::None);
        assert!(step.run.is_none());
        assert!(supervisor.summary().state.allows_automatic());
        assert!(rig.block_kinds().is_empty());
        assert!(rig.recorder.is_empty());
        // Resume with nothing to clear is fine.
        assert!(supervisor.request_resume(&operator()).unwrap().is_empty());
    }

    #[tokio::test]
    async fn two_monitors_combine_with_and_and_a_disconnect_removes_one() {
        let manual = Arc::new(ManualClock::new());
        let rig = Rig::new(Arc::clone(&manual) as Arc<dyn Clock>);
        let (a_control, b_control) = (SimSafetyControl::new(), SimSafetyControl::new());
        let mut a = MonitorHarness::new(Arc::clone(&rig.clock), a_control, "cloud");
        let mut b = MonitorHarness::new(Arc::clone(&rig.clock), b_control.clone(), "rain");
        let supervisor = rig.supervisor(PolicySettings::default());
        supervisor.add_monitor(a.handle.clone());
        supervisor.add_monitor(b.handle.clone());
        a.tick();
        b.tick();
        let step = supervisor.step().await;
        assert_eq!(step.evaluation.summary.state, SafetyGate::Safe);
        assert_eq!(step.evaluation.summary.sources.len(), 2);

        b_control.set_safe(false);
        manual.advance(SAFETY_POLL_INTERVAL);
        a.tick();
        b.tick();
        let stopper = rig.stop_mount_on_cancel();
        let run = supervisor.step().await.run.expect("either monitor unsafe");
        stopper.join().unwrap();
        assert_eq!(run.trigger.connection_id(), b.handle.connection_id());

        // The operator disconnects the rain sensor: only `cloud` remains.
        b.end.retire(manual.now_mono_ns());
        manual.advance(SAFETY_POLL_INTERVAL);
        a.tick();
        let step = supervisor.step().await;
        assert_eq!(supervisor.monitor_count(), 1);
        assert_eq!(step.evaluation.summary.state, SafetyGate::Safe);
        assert!(supervisor.remove_monitor(a.handle.connection_id()));
        assert_eq!(
            supervisor.step().await.evaluation.summary.state,
            SafetyGate::None
        );
    }

    #[tokio::test]
    async fn the_background_loop_runs_the_policy() {
        let rig = Rig::new(Arc::new(SystemClock));
        let control = SimSafetyControl::new();
        let mut monitor = MonitorHarness::new(Arc::clone(&rig.clock), control.clone(), "loop");
        let supervisor = rig.supervisor(PolicySettings::default());
        supervisor.add_monitor(monitor.handle.clone());
        let task = supervisor.spawn(Duration::from_millis(10));
        let mut summary = supervisor.watch_summary();

        monitor.tick();
        tokio::time::timeout(
            Duration::from_secs(10),
            summary.wait_for(|s| s.state == SafetyGate::Safe),
        )
        .await
        .expect("the loop evaluates")
        .unwrap();

        let stopper = rig.stop_mount_on_cancel();
        control.set_safe(false);
        monitor.tick();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while supervisor.runs().is_empty() || supervisor.policy_running() {
            assert!(std::time::Instant::now() < deadline, "the policy never ran");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        stopper.join().unwrap();
        assert_eq!(rig.mount_fake.commands(), vec![MountCmd::Park]);
        assert!(!supervisor.policy_running());
        task.abort();
    }

    #[test]
    fn settings_come_from_the_configuration() {
        let config = AchernarConfig::default();
        let settings = PolicySettings::from_config(&config.safety, &config.mount);
        assert_eq!(
            settings.on_unsafe,
            OnUnsafe::Park,
            "park_on_safety_violation"
        );
        assert!(settings.abort_exposure);
        assert_eq!(
            settings.park_timeout,
            Duration::from_secs(config.mount.slew_timeout_s + 30)
        );
        let mut mount = config.mount.clone();
        mount.safety.park_on_safety_violation = false;
        assert_eq!(
            PolicySettings::from_config(&config.safety, &mount).on_unsafe,
            OnUnsafe::StopTracking
        );

        let clock: Arc<dyn Clock> = Arc::new(ManualClock::new());
        let recorder = RecordingSink::new();
        let events: Arc<dyn EventSink> = Arc::new(RecorderEventSink::new(recorder.clone()));
        let blocks: Arc<dyn ResumeBlockStore> = Arc::new(MemoryResumeBlocks::new(
            Arc::clone(&events),
            Arc::clone(&clock),
        ));
        let mut configs = config.equipment_configs();
        configs.safety.resume = SafetyResume::Auto;
        configs.safety.warning_is_unsafe = false;
        let supervisor = SafetySupervisor::from_config(
            &configs,
            Arc::new(AuditingSafetyGate::new(
                recorder.clone(),
                Arc::clone(&clock),
            )),
            blocks,
            events,
            Arc::new(PolicyTargets::default()),
            clock,
        );
        assert_eq!(supervisor.policy().settings().on_unsafe, OnUnsafe::Park);
        assert!(!lock(&supervisor.state).hub.settings().warning_is_unsafe);
        assert_eq!(supervisor.monitor_count(), 0);
    }
}

//! The command gate (spec §2.9, D9, D12, D19, H21). Owner: WP1A-2.
//!
//! Every motion command passes [`CommandGate::authorize`] as a
//! [`GatedCommand`], in this order:
//!
//! 1. **Ownership** (`control::ownership::DomainLedger`): every domain the
//!    operation needs is claimed, compound operations claim several (§2.9
//!    table, `GatedOp::default_domains`). The returned [`Permit`] releases
//!    them on drop, so ownership is released on success, error, timeout,
//!    cancel and worker panic alike.
//! 2. **Arbiter** (`control::action_arbiter::arbitrate`): a request preempts
//!    a holder only with a strictly higher rank; an operator only with
//!    `overriding = true` and only below safety (audited as an override);
//!    an automatic command never preempts an operator-held domain. The
//!    preempted operation is cancelled by its worker when the preempting
//!    command arrives (`Aborted { by }`).
//! 3. **Safety checks**: the [`SafetyCheck`] plug-ins registered by device
//!    packages, plus two built-ins: the camera boundary (commands that may
//!    run only between exposures) and the §2.8 fail-safe (a safety monitor
//!    that is not Ready refuses new automatic motion).
//! 4. **Audit**: one `control_audit_logs` record per stage (proposed, then
//!    validated or rejected, then executed, then outcome; v6 §7.2).
//!
//! Before ownership the gate refuses, without claiming anything: relay
//! commands that are direct-only (D12), policy actuation that is switched
//! off (G-j), and automatic sources while a resume block is active (H22,
//! HTTP 423). Abort and e-stop skip steps 1-3 and are always audited (D19).
//!
//! [`CommandGate::submit`] is the executor: authorize, stop PHD2 before a
//! slew, park, flip or center (D20, `coord.rs`), send to the worker, audit
//! the acceptance and the outcome, and raise an `outcome_unknown` resume
//! block when the outcome is unknown.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
use std::time::Duration;

use tokio::sync::oneshot;
use uuid::Uuid;

use crate::control::action_arbiter::{arbitrate, Arbitration};
use crate::control::ownership::{DomainClaim, DomainLedger};

use super::commands::{
    Domain, GateError, GatedCommand, GatedOp, GuiderCmd, MountCmd, SafetyRejection,
};
use super::coord::{self, CoordError, PreMotion};
use super::events::{AuditRecord, AuditStage, EquipmentEvent, EventKind, GroundTruthRecord};
use super::handle::{AbortTicket, DeviceHandle, GuiderHandle, OpTicket, Permit};
use super::ops::OpKind;
use super::resume::{BlockRequest, ClearError};
use super::snapshot::{EquipmentSnapshot, ResumeBlock, ResumeBlockKind};
use super::traits::{Clock, SafetyCheck};
use super::types::{
    AutonomyMode, CommandId, CommandSource, DeviceError, DeviceKind, OpOutcome, Priority,
};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// What the gate needs from the hub. Implemented by `hub::EquipmentHub`'s
/// core; gate tests use a small fake.
pub trait GateContext: Send + Sync {
    fn clock(&self) -> Arc<dyn Clock>;
    /// The aggregate the safety checks read.
    fn snapshot(&self) -> Arc<EquipmentSnapshot>;
    fn active_blocks(&self) -> Vec<ResumeBlock>;
    /// Raise a resume block (the hub emits `resume_blocked`).
    fn raise_block(&self, request: BlockRequest) -> ResumeBlock;
    /// The operator clear path (the hub emits `resume_cleared`).
    fn clear_block(&self, id: Uuid, by: &CommandSource) -> Result<ResumeBlock, ClearError>;
    fn emit(&self, event: EquipmentEvent);
    fn record(&self, record: GroundTruthRecord);
    /// The connected guider, for the D20 pre-motion stop.
    fn guider(&self) -> Option<GuiderHandle> {
        None
    }
}

/// Gate behaviour taken from the configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct GateConfig {
    /// `[guider] allow_policy_actuation` (M-1 `set_algo_param`, Tier 2).
    pub allow_policy_actuation: bool,
    /// How long the D20 pre-motion stop waits for PHD2 to report `Stopped`.
    pub stop_guiding_timeout: Duration,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            allow_policy_actuation: false,
            stop_guiding_timeout: coord::STOP_GUIDING_TIMEOUT,
        }
    }
}

/// A submitted command: the forwarded ticket, plus what the pre-motion stop
/// did (`was_guiding`, D20).
#[derive(Debug)]
pub struct Submitted {
    pub ticket: OpTicket,
    pub pre_motion: Option<PreMotion>,
}

/// Why [`CommandGate::submit`] failed.
#[derive(Debug, Clone, PartialEq)]
pub enum SubmitError {
    Gate(GateError),
    /// The guider would not stop before the motion (`guider_stop_failed`).
    Coord(CoordError),
    /// The worker refused the command.
    Device(DeviceError),
}

impl SubmitError {
    pub fn http_status(&self) -> u16 {
        match self {
            SubmitError::Gate(err) => err.http_status(),
            SubmitError::Coord(err) => err.http_status(),
            SubmitError::Device(err) => err.class.http_status(),
        }
    }

    /// API error code (`{error: {code, message}}`, §2.12).
    pub fn code(&self, kind: DeviceKind) -> String {
        match self {
            SubmitError::Gate(err) => err.code().to_string(),
            SubmitError::Coord(err) => err.code().to_string(),
            SubmitError::Device(err) => err.api_code(kind),
        }
    }
}

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubmitError::Gate(err) => err.fmt(f),
            SubmitError::Coord(err) => err.fmt(f),
            SubmitError::Device(err) => err.fmt(f),
        }
    }
}

impl std::error::Error for SubmitError {}

impl From<GateError> for SubmitError {
    fn from(err: GateError) -> Self {
        SubmitError::Gate(err)
    }
}

/// The domains a command actually claims. A jog renewal, a destination
/// side-of-pier query and a guider refresh run alongside the operation that
/// holds the device, so they claim nothing.
pub fn ownership_domains(cmd: &GatedCommand) -> Vec<Domain> {
    if matches!(
        cmd.op,
        GatedOp::Mount(MountCmd::RenewJog | MountCmd::QueryDestinationSideOfPier(_))
            | GatedOp::Guider(GuiderCmd::Refresh)
    ) {
        return Vec::new();
    }
    let mut domains = cmd.domains.clone();
    domains.sort();
    domains.dedup();
    domains
}

fn domain_kind(domain: Domain) -> Option<DeviceKind> {
    match domain {
        Domain::Device(kind) => Some(kind),
        Domain::CameraBoundary => Some(DeviceKind::Camera),
    }
}

/// Write one audit record through `ctx`.
fn audit_with(
    ctx: &dyn GateContext,
    cmd: &GatedCommand,
    stage: AuditStage,
    accepted: bool,
    reason: Option<String>,
    annotations: Vec<String>,
    outcome: Option<OpOutcome>,
) {
    let mut record = AuditRecord::for_command(cmd, stage, accepted, reason, ctx.clock().now_utc());
    record.annotations = annotations;
    record.outcome = outcome;
    ctx.record(GroundTruthRecord::Audit(Box::new(record)));
}

fn outcome_reason(outcome: &OpOutcome) -> Option<String> {
    match outcome {
        OpOutcome::Succeeded => None,
        OpOutcome::Failed(err) | OpOutcome::Unknown(err) => Some(err.to_string()),
        OpOutcome::Aborted { by } => Some(match by {
            Some(by) => format!("aborted by {by}"),
            None => "aborted".into(),
        }),
        OpOutcome::TimedOut => Some("timed out".into()),
    }
}

/// The command gate.
pub struct CommandGate {
    ctx: Arc<dyn GateContext>,
    ledger: Arc<Mutex<DomainLedger>>,
    checks: RwLock<Vec<Arc<dyn SafetyCheck>>>,
    config: RwLock<GateConfig>,
    shutting_down: AtomicBool,
}

impl std::fmt::Debug for CommandGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandGate")
            .field("checks", &self.check_names())
            .field("shutting_down", &self.is_shutting_down())
            .finish()
    }
}

impl CommandGate {
    /// A gate with the built-in checks registered.
    pub fn new(ctx: Arc<dyn GateContext>, config: GateConfig) -> Self {
        let builtins: Vec<Arc<dyn SafetyCheck>> =
            vec![Arc::new(CameraBoundaryCheck), Arc::new(SafetyFailSafeCheck)];
        Self {
            ctx,
            ledger: Arc::new(Mutex::new(DomainLedger::new())),
            checks: RwLock::new(builtins),
            config: RwLock::new(config),
            shutting_down: AtomicBool::new(false),
        }
    }

    pub fn config(&self) -> GateConfig {
        self.config
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub fn set_config(&self, config: GateConfig) {
        *self.config.write().unwrap_or_else(|e| e.into_inner()) = config;
    }

    /// Register a device package's check (§2.9 step 3 table). Checks run in
    /// registration order and must not call back into the gate.
    pub fn register_check(&self, check: Arc<dyn SafetyCheck>) {
        self.checks
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .push(check);
    }

    pub fn check_names(&self) -> Vec<&'static str> {
        self.checks
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|check| check.name())
            .collect()
    }

    /// Stop accepting gated commands (§2.15 step 1). Abort and e-stop still
    /// pass.
    pub fn begin_shutdown(&self) {
        self.shutting_down.store(true, Ordering::Release);
    }

    pub fn is_shutting_down(&self) -> bool {
        self.shutting_down.load(Ordering::Acquire)
    }

    /// Who holds `domain` now.
    pub fn holder(&self, domain: Domain) -> Option<DomainClaim> {
        lock(&self.ledger).holder(domain).cloned()
    }

    /// The command that preempted `command_id`, if any. A compound
    /// operation checks this between steps.
    pub fn preempted_by(&self, command_id: CommandId) -> Option<CommandId> {
        lock(&self.ledger).preempted_by(command_id)
    }

    /// Every held domain and its holder (for status and diagnostics).
    pub fn holders(&self) -> Vec<(Domain, DomainClaim)> {
        lock(&self.ledger)
            .claims()
            .map(|(domain, claim)| (*domain, claim.clone()))
            .collect()
    }

    /// True when no domain is held.
    pub fn is_idle(&self) -> bool {
        lock(&self.ledger).is_empty()
    }

    /// True when `command_id` still holds every one of `domains`.
    pub fn holds_all(&self, command_id: CommandId, domains: &[Domain]) -> bool {
        lock(&self.ledger).holds_all(command_id, domains)
    }

    // -- audit ---------------------------------------------------------------

    fn audit(
        &self,
        cmd: &GatedCommand,
        stage: AuditStage,
        accepted: bool,
        reason: Option<String>,
        annotations: Vec<String>,
        outcome: Option<OpOutcome>,
    ) {
        audit_with(
            self.ctx.as_ref(),
            cmd,
            stage,
            accepted,
            reason,
            annotations,
            outcome,
        );
    }

    /// Audit the `executed` stage (the worker accepted the command).
    pub fn record_executed(&self, cmd: &GatedCommand, annotations: Vec<String>) {
        self.audit(cmd, AuditStage::Executed, true, None, annotations, None);
    }

    /// Audit the `outcome` stage.
    pub fn record_outcome(&self, cmd: &GatedCommand, outcome: &OpOutcome) {
        self.audit(
            cmd,
            AuditStage::Outcome,
            outcome.is_success(),
            outcome_reason(outcome),
            Vec::new(),
            Some(outcome.clone()),
        );
    }

    fn reject(&self, cmd: &GatedCommand, err: GateError) -> GateError {
        self.audit(
            cmd,
            AuditStage::Rejected,
            false,
            Some(err.to_string()),
            Vec::new(),
            None,
        );
        err
    }

    fn emit_gate(&self, cmd: &GatedCommand, kind: Option<DeviceKind>, event: EventKind) {
        let mut event =
            EquipmentEvent::new(event, self.ctx.clock().as_ref()).with_command(cmd.command_id);
        event.device_kind = kind;
        self.ctx.emit(event);
    }

    // -- refusals before ownership ---------------------------------------------

    fn policy_refusal(&self, cmd: &GatedCommand) -> Option<GateError> {
        if matches!(cmd.source, CommandSource::Relay { .. }) && !cmd.op.relay_allowed() {
            return Some(GateError::Refused {
                reason: "not available over the relay; use the direct daemon API".into(),
            });
        }
        if matches!(cmd.source, CommandSource::Policy { .. })
            && matches!(cmd.op, GatedOp::Guider(GuiderCmd::SetAlgoParam { .. }))
            && !self.config().allow_policy_actuation
        {
            return Some(GateError::Refused {
                reason: "policy actuation of guider parameters is disabled \
                         ([guider] allow_policy_actuation = false)"
                    .into(),
            });
        }
        None
    }

    fn resume_refusal(&self, cmd: &GatedCommand) -> Option<GateError> {
        let kind = cmd.kind();
        if !cmd.source.is_automatic() || !(kind == OpKind::Capture || kind.moves_hardware()) {
            return None;
        }
        let blocks = self.ctx.active_blocks();
        if blocks.is_empty() {
            return None;
        }
        Some(GateError::ResumeBlocked {
            blocks: blocks.into_iter().map(|block| block.kind).collect(),
        })
    }

    fn run_checks(
        &self,
        cmd: &GatedCommand,
        snapshot: &EquipmentSnapshot,
        annotations: &mut Vec<String>,
    ) -> Result<(), SafetyRejection> {
        let checks: Vec<Arc<dyn SafetyCheck>> = self
            .checks
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        for check in checks {
            if !check.applies_to(cmd) {
                continue;
            }
            check.check(cmd, snapshot)?;
            annotations.extend(check.annotations(cmd, snapshot));
        }
        Ok(())
    }

    fn arbitrate_all(
        ledger: &DomainLedger,
        cmd: &GatedCommand,
        domains: &[Domain],
    ) -> Result<Vec<(Domain, CommandId, bool)>, GateError> {
        let mut preempted = Vec::new();
        for &domain in domains {
            match arbitrate(
                ledger.holder(domain),
                cmd.command_id,
                cmd.rank(),
                &cmd.source,
            ) {
                Arbitration::Grant => {}
                Arbitration::Preempt {
                    holder,
                    override_applied,
                } => preempted.push((domain, holder, override_applied)),
                Arbitration::Conflict { holder } => {
                    return Err(GateError::OwnershipConflict {
                        domain,
                        holder: Some(holder),
                        holder_source: ledger.holder(domain).map(|claim| claim.source.clone()),
                    })
                }
            }
        }
        Ok(preempted)
    }

    fn conflict(&self, cmd: &GatedCommand, err: GateError) -> GateError {
        if let GateError::OwnershipConflict { domain, holder, .. } = &err {
            self.emit_gate(
                cmd,
                domain_kind(*domain),
                EventKind::OwnershipConflict {
                    domain: *domain,
                    holder: *holder,
                    requested_by: cmd.source.clone(),
                },
            );
        }
        self.reject(cmd, err)
    }

    fn safety_rejected(&self, cmd: &GatedCommand, rejection: SafetyRejection) -> GateError {
        self.emit_gate(
            cmd,
            cmd.domains.iter().copied().find_map(domain_kind),
            EventKind::SafetyRejected {
                check: rejection.check.clone(),
                reason: rejection.reason.clone(),
            },
        );
        self.reject(cmd, GateError::SafetyRejected(rejection))
    }

    // -- authorisation ---------------------------------------------------------

    /// Authorise a command (§2.9). On success the returned permit holds the
    /// command's domains until it drops.
    pub fn authorize(&self, cmd: &GatedCommand) -> Result<Permit, GateError> {
        self.audit(cmd, AuditStage::Proposed, true, None, Vec::new(), None);
        if cmd.op.bypasses_gate_checks() {
            self.audit(
                cmd,
                AuditStage::Validated,
                true,
                Some("abort path: ownership, arbiter and safety checks are skipped (D19)".into()),
                Vec::new(),
                None,
            );
            return Ok(Permit::new(cmd, None));
        }
        if self.is_shutting_down() {
            return Err(self.reject(cmd, GateError::ShuttingDown));
        }
        if let Some(err) = self.policy_refusal(cmd) {
            return Err(self.reject(cmd, err));
        }
        if let Some(err) = self.resume_refusal(cmd) {
            return Err(self.reject(cmd, err));
        }
        let domains = ownership_domains(cmd);

        // Steps 1-2, first pass: refuse a conflict before running any check,
        // so a 409 always wins over a 422.
        if let Err(err) = Self::arbitrate_all(&lock(&self.ledger), cmd, &domains) {
            return Err(self.conflict(cmd, err));
        }

        // Step 3, outside the ledger lock (plug-ins never run under it).
        let snapshot = self.ctx.snapshot();
        let mut annotations = Vec::new();
        if let Err(rejection) = self.run_checks(cmd, &snapshot, &mut annotations) {
            return Err(self.safety_rejected(cmd, rejection));
        }

        // Steps 1-2 again, then commit: another command may have claimed a
        // domain while the checks ran.
        let now = self.ctx.clock().now_utc();
        let mut ledger = lock(&self.ledger);
        let preempted = match Self::arbitrate_all(&ledger, cmd, &domains) {
            Ok(preempted) => preempted,
            Err(err) => {
                drop(ledger);
                return Err(self.conflict(cmd, err));
            }
        };
        let claim = DomainClaim {
            command_id: cmd.command_id,
            rank: cmd.rank(),
            priority: cmd.priority,
            source: cmd.source.clone(),
            claimed_at: now,
        };
        for &domain in &domains {
            if let Some(old) = ledger.claim(domain, claim.clone()) {
                if old.command_id != cmd.command_id {
                    ledger.note_preempted(old.command_id, cmd.command_id);
                }
            }
        }
        drop(ledger);

        for (domain, holder, override_applied) in &preempted {
            if *override_applied {
                self.emit_gate(
                    cmd,
                    domain_kind(*domain),
                    EventKind::OverrideApplied {
                        domain: *domain,
                        cancelled: Some(*holder),
                    },
                );
            }
            annotations.push(format!("preempted {holder} on {domain:?}"));
        }
        self.audit(cmd, AuditStage::Validated, true, None, annotations, None);
        let ledger = Arc::downgrade(&self.ledger);
        let command_id = cmd.command_id;
        let release: Box<dyn FnOnce() + Send> = Box::new(move || {
            if let Some(ledger) = ledger.upgrade() {
                lock(&ledger).release(command_id);
            }
        });
        Ok(Permit::new(cmd, Some(release)))
    }

    /// A permit for one step of a compound operation (a focuser move inside
    /// an autofocus run, the wheel move inside a filter handshake). The
    /// parent must still hold every domain the step claims, and must not
    /// have been preempted; the step's own safety checks run. The permit
    /// holds no ownership of its own: the parent's does.
    pub fn child_permit(&self, parent: &Permit, cmd: &GatedCommand) -> Result<Permit, GateError> {
        self.audit(
            cmd,
            AuditStage::Proposed,
            true,
            Some(format!("step of {}", parent.command_id())),
            Vec::new(),
            None,
        );
        if cmd.op.bypasses_gate_checks() {
            self.audit(cmd, AuditStage::Validated, true, None, Vec::new(), None);
            return Ok(Permit::new(cmd, None));
        }
        if self.is_shutting_down() {
            return Err(self.reject(cmd, GateError::ShuttingDown));
        }
        if let Some(err) = self.policy_refusal(cmd) {
            return Err(self.reject(cmd, err));
        }
        if let Some(err) = self.resume_refusal(cmd) {
            return Err(self.reject(cmd, err));
        }
        let domains = ownership_domains(cmd);
        let refusal = {
            let ledger = lock(&self.ledger);
            if let Some(by) = ledger.preempted_by(parent.command_id()) {
                let domain = domains
                    .first()
                    .copied()
                    .or_else(|| parent.domains().first().copied())
                    .unwrap_or(Domain::CameraBoundary);
                Some(GateError::OwnershipConflict {
                    domain,
                    holder: Some(by),
                    holder_source: ledger.holder(domain).map(|claim| claim.source.clone()),
                })
            } else {
                domains.iter().find_map(|&domain| {
                    let holder = ledger.holder(domain);
                    (holder.map(|claim| claim.command_id) != Some(parent.command_id())).then(|| {
                        GateError::OwnershipConflict {
                            domain,
                            holder: holder.map(|claim| claim.command_id),
                            holder_source: holder.map(|claim| claim.source.clone()),
                        }
                    })
                })
            }
        };
        if let Some(err) = refusal {
            return Err(self.conflict(cmd, err));
        }
        let snapshot = self.ctx.snapshot();
        let mut annotations = Vec::new();
        if let Err(rejection) = self.run_checks(cmd, &snapshot, &mut annotations) {
            return Err(self.safety_rejected(cmd, rejection));
        }
        self.audit(cmd, AuditStage::Validated, true, None, annotations, None);
        Ok(Permit::new(cmd, None))
    }

    // -- execution -------------------------------------------------------------

    /// Authorise, coordinate (D20), send, and audit a device command. The
    /// returned ticket resolves with the outcome, which is also audited; an
    /// `Unknown` outcome raises an `outcome_unknown` resume block. Must run
    /// inside a tokio runtime.
    pub async fn submit<C, S>(
        &self,
        handle: &DeviceHandle<C, S>,
        cmd: GatedCommand,
        device_cmd: C,
    ) -> Result<Submitted, SubmitError>
    where
        C: Send + 'static,
        S: Send + Sync + 'static,
    {
        let permit = self.authorize(&cmd)?;
        let pre_motion = if coord::needs_guider_stop(&cmd.op) {
            let guider = self.ctx.guider();
            let clock = self.ctx.clock();
            let timeout = self.config().stop_guiding_timeout;
            match coord::stop_guiding_before_motion(
                self,
                guider.as_ref(),
                &cmd,
                clock.as_ref(),
                timeout,
            )
            .await
            {
                Ok(report) => Some(report),
                Err(err) => {
                    drop(permit);
                    self.record_outcome(&cmd, &OpOutcome::Failed(err.to_device_error()));
                    return Err(SubmitError::Coord(err));
                }
            }
        } else {
            None
        };
        let ticket = match handle.send(device_cmd, permit).await {
            Ok(ticket) => ticket,
            Err(err) => {
                self.audit(
                    &cmd,
                    AuditStage::Rejected,
                    false,
                    Some(format!("the device refused the command: {err}")),
                    Vec::new(),
                    None,
                );
                return Err(SubmitError::Device(err));
            }
        };
        let mut annotations = Vec::new();
        if let Some(report) = &pre_motion {
            annotations.push(format!("was_guiding={}", report.was_guiding));
            if let Some(warning) = &report.warning {
                annotations.push(format!("pre_motion_warning={warning}"));
            }
        }
        self.record_executed(&cmd, annotations);

        let (tx, rx) = oneshot::channel();
        let ctx = Arc::clone(&self.ctx);
        let kind = handle.kind();
        let connection_id = handle.connection_id();
        let command_id = cmd.command_id;
        tokio::spawn(async move {
            let outcome = ticket.wait().await;
            audit_with(
                ctx.as_ref(),
                &cmd,
                AuditStage::Outcome,
                outcome.is_success(),
                outcome_reason(&outcome),
                Vec::new(),
                Some(outcome.clone()),
            );
            if let OpOutcome::Unknown(err) = &outcome {
                ctx.raise_block(
                    BlockRequest::new(
                        ResumeBlockKind::OutcomeUnknown {
                            device: kind,
                            op: cmd.kind(),
                        },
                        err.message.clone(),
                    )
                    .for_device(kind, Some(connection_id))
                    .with_command(Some(command_id)),
                );
            }
            let _ = tx.send(outcome);
        });
        Ok(Submitted {
            ticket: OpTicket {
                command_id,
                done: rx,
            },
            pre_motion,
        })
    }

    /// Abort a device (D19): audited, never refused by ownership or checks,
    /// and attempted on a degraded or lost handle until it is retired.
    pub fn abort<C, S>(&self, handle: &DeviceHandle<C, S>, source: CommandSource) -> AbortTicket {
        let cmd = GatedCommand::new(
            GatedOp::Abort {
                kind: handle.kind(),
            },
            source,
            AutonomyMode::Manual,
            Priority::Safety,
        );
        // Abort bypasses the checks, so this always succeeds and audits the
        // proposed and validated stages.
        let _ = self.authorize(&cmd);
        let ticket = handle.abort();
        self.audit(
            &cmd,
            AuditStage::Executed,
            ticket.abort_path_ran || ticket.cancel_sent,
            None,
            vec![
                format!("abort_ticket={}", ticket.command_id),
                format!("abort_path_ran={}", ticket.abort_path_ran),
                format!("cancel_sent={}", ticket.cancel_sent),
            ],
            None,
        );
        ticket
    }

    /// The operator clear (`POST equipment/resume-blocks/clear {id}`),
    /// audited. `safety_*` blocks are refused (409).
    pub fn clear_resume_block(
        &self,
        id: Uuid,
        source: CommandSource,
    ) -> Result<ResumeBlock, GateError> {
        let cmd = GatedCommand::new(
            GatedOp::ClearResumeBlock { id },
            source.clone(),
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        self.audit(&cmd, AuditStage::Proposed, true, None, Vec::new(), None);
        match self.ctx.clear_block(id, &source) {
            Ok(block) => {
                self.audit(&cmd, AuditStage::Validated, true, None, Vec::new(), None);
                self.record_outcome(&cmd, &OpOutcome::Succeeded);
                Ok(block)
            }
            Err(err) => Err(self.reject(
                &cmd,
                GateError::Refused {
                    reason: err.to_string(),
                },
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in checks
// ---------------------------------------------------------------------------

/// Commands in the camera-boundary domain (dither, `set_algo_param`, flip)
/// run only between exposures. An unknown camera state refuses automatic
/// sources and is annotated for operators.
#[derive(Debug, Clone, Copy, Default)]
pub struct CameraBoundaryCheck;

impl SafetyCheck for CameraBoundaryCheck {
    fn name(&self) -> &'static str {
        "camera_boundary"
    }

    fn applies_to(&self, cmd: &GatedCommand) -> bool {
        cmd.domains.contains(&Domain::CameraBoundary)
    }

    fn check(&self, cmd: &GatedCommand, snap: &EquipmentSnapshot) -> Result<(), SafetyRejection> {
        match snap.camera.exposing.known() {
            Some(true) => Err(SafetyRejection::new(
                self.name(),
                "an exposure is integrating; this command runs only between exposures",
            )),
            Some(false) => Ok(()),
            None if cmd.source.is_automatic() => Err(SafetyRejection::new(
                self.name(),
                "the camera state is unknown, so an exposure boundary cannot be confirmed",
            )),
            None => Ok(()),
        }
    }

    fn annotations(&self, _cmd: &GatedCommand, snap: &EquipmentSnapshot) -> Vec<String> {
        if snap.camera.exposing.known().is_none() {
            vec!["camera_state_unknown".into()]
        } else {
            Vec::new()
        }
    }
}

/// §2.8 fail-safe: a safety monitor in any state other than Ready, or a
/// reading that is not confirmed safe, refuses new automatic motion and
/// automatic captures. Operator and safety sources are not affected.
#[derive(Debug, Clone, Copy, Default)]
pub struct SafetyFailSafeCheck;

impl SafetyCheck for SafetyFailSafeCheck {
    fn name(&self) -> &'static str {
        "safety_fail_safe"
    }

    fn applies_to(&self, cmd: &GatedCommand) -> bool {
        let kind = cmd.kind();
        cmd.source.is_automatic() && (kind.moves_hardware() || kind == OpKind::Capture)
    }

    fn check(&self, _cmd: &GatedCommand, snap: &EquipmentSnapshot) -> Result<(), SafetyRejection> {
        if snap.safety.state.allows_automatic() {
            Ok(())
        } else {
            Err(SafetyRejection::new(
                self.name(),
                format!(
                    "the safety state is {:?}; new automatic actions need a confirmed safe \
                     monitor",
                    snap.safety.state
                ),
            ))
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::equipment::commands::{FocuserCmd, GuiderCmd};
    use crate::equipment::resume::ResumeBlocks;
    use crate::equipment::snapshot::{FocuserSnapshot, SafetyGate};
    use crate::equipment::testing::fake_handles::{fake_focuser, FakeResponse};
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::types::{GuideDirection, Observed, Source};

    /// A minimal gate context.
    pub(crate) struct TestCtx {
        pub clock: Arc<ManualClock>,
        pub snapshot: Mutex<EquipmentSnapshot>,
        pub blocks: ResumeBlocks,
        pub events: Mutex<Vec<EquipmentEvent>>,
        pub records: Mutex<Vec<GroundTruthRecord>>,
        pub guider: Mutex<Option<GuiderHandle>>,
    }

    impl TestCtx {
        pub(crate) fn new() -> Arc<Self> {
            let clock = Arc::new(ManualClock::new());
            let mut snapshot = EquipmentSnapshot::default();
            snapshot.camera.exposing = Observed::fresh_at(false, Source::Derived, clock.as_ref());
            Arc::new(Self {
                blocks: ResumeBlocks::new(clock.clone()),
                clock,
                snapshot: Mutex::new(snapshot),
                events: Mutex::new(Vec::new()),
                records: Mutex::new(Vec::new()),
                guider: Mutex::new(None),
            })
        }

        pub(crate) fn event_names(&self) -> Vec<&'static str> {
            lock(&self.events).iter().map(|e| e.name()).collect()
        }

        pub(crate) fn stages(&self, id: CommandId) -> Vec<AuditStage> {
            lock(&self.records)
                .iter()
                .filter_map(|record| match record {
                    GroundTruthRecord::Audit(audit) if audit.command_id == id => Some(audit.stage),
                    _ => None,
                })
                .collect()
        }
    }

    impl GateContext for TestCtx {
        fn clock(&self) -> Arc<dyn Clock> {
            self.clock.clone()
        }
        fn snapshot(&self) -> Arc<EquipmentSnapshot> {
            Arc::new(lock(&self.snapshot).clone())
        }
        fn active_blocks(&self) -> Vec<ResumeBlock> {
            self.blocks.active()
        }
        fn raise_block(&self, request: BlockRequest) -> ResumeBlock {
            self.blocks.raise(request).into_block()
        }
        fn clear_block(&self, id: Uuid, by: &CommandSource) -> Result<ResumeBlock, ClearError> {
            self.blocks.clear(id, by)
        }
        fn emit(&self, event: EquipmentEvent) {
            lock(&self.events).push(event);
        }
        fn record(&self, record: GroundTruthRecord) {
            lock(&self.records).push(record);
        }
        fn guider(&self) -> Option<GuiderHandle> {
            lock(&self.guider).clone()
        }
    }

    pub(crate) fn gate_with_ctx() -> (Arc<CommandGate>, Arc<TestCtx>) {
        let ctx = TestCtx::new();
        let gate = Arc::new(CommandGate::new(ctx.clone(), GateConfig::default()));
        (gate, ctx)
    }

    fn operator(overriding: bool) -> CommandSource {
        CommandSource::Operator { overriding }
    }

    fn policy() -> CommandSource {
        CommandSource::Policy {
            model: "fw2".into(),
            version: "1.0".into(),
        }
    }

    fn mount_cmd(source: CommandSource, priority: Priority) -> GatedCommand {
        GatedCommand::new(
            GatedOp::Mount(MountCmd::Park),
            source,
            AutonomyMode::Manual,
            priority,
        )
    }

    #[test]
    fn operator_versus_automatic_preemption_matrix() {
        let mount = Domain::Device(DeviceKind::Mount);
        let holders: Vec<(&str, CommandSource, Priority)> = vec![
            ("operator", operator(false), Priority::Exposure),
            ("recommendation", policy(), Priority::Recommendation),
            ("exposure", policy(), Priority::Exposure),
            (
                "autofocus",
                CommandSource::Trigger {
                    rule: "af_temp".into(),
                },
                Priority::Autofocus,
            ),
            ("recovery", CommandSource::System, Priority::MountRecovery),
            ("safety", CommandSource::Safety, Priority::Safety),
        ];
        let requests: Vec<(&str, CommandSource, Priority)> = vec![
            ("operator", operator(false), Priority::Exposure),
            ("override", operator(true), Priority::Exposure),
            (
                "relay_override",
                CommandSource::Relay { overriding: true },
                Priority::Exposure,
            ),
            ("recommendation", policy(), Priority::Recommendation),
            ("exposure", policy(), Priority::Exposure),
            (
                "autofocus",
                CommandSource::Trigger {
                    rule: "af_temp".into(),
                },
                Priority::Autofocus,
            ),
            ("recovery", CommandSource::System, Priority::MountRecovery),
            ("safety", CommandSource::Safety, Priority::Safety),
            ("estop_rank", CommandSource::Safety, Priority::EStop),
        ];
        // Expected: does `request` take the domain from `holder`?
        let expect = |holder: &str, request: &str| -> bool {
            let holder_rank = match holder {
                "operator" => 5,
                "recommendation" => 1,
                "exposure" => 2,
                "autofocus" => 3,
                "recovery" => 4,
                "safety" => 6,
                _ => unreachable!(),
            };
            match request {
                "operator" => false,
                "override" | "relay_override" => holder_rank < 6,
                "recommendation" => holder_rank < 1,
                "exposure" => holder_rank < 2,
                "autofocus" => holder_rank < 3,
                "recovery" => holder_rank < 4,
                "safety" => holder_rank < 6,
                "estop_rank" => holder_rank < 7,
                _ => unreachable!(),
            }
        };
        for (holder_name, holder_source, holder_priority) in &holders {
            for (request_name, request_source, request_priority) in &requests {
                let (gate, _ctx) = gate_with_ctx();
                let held = mount_cmd(holder_source.clone(), *holder_priority);
                let held_permit = gate.authorize(&held).expect("free domain is granted");
                let request = mount_cmd(request_source.clone(), *request_priority);
                let result = gate.authorize(&request);
                let label = format!("{request_name} vs {holder_name}");
                if expect(holder_name, request_name) {
                    let permit = result.unwrap_or_else(|e| panic!("{label}: {e}"));
                    assert_eq!(
                        gate.holder(mount).map(|c| c.command_id),
                        Some(request.command_id),
                        "{label}"
                    );
                    assert_eq!(gate.preempted_by(held.command_id), Some(request.command_id));
                    // The preempted holder's late release frees nothing.
                    drop(held_permit);
                    assert_eq!(
                        gate.holder(mount).map(|c| c.command_id),
                        Some(request.command_id),
                        "{label}"
                    );
                    drop(permit);
                    assert!(gate.holder(mount).is_none(), "{label}");
                } else {
                    match result {
                        Err(GateError::OwnershipConflict { holder, .. }) => {
                            assert_eq!(holder, Some(held.command_id), "{label}");
                        }
                        other => panic!("{label}: expected 409, got {other:?}"),
                    }
                    assert_eq!(
                        gate.holder(mount).map(|c| c.command_id),
                        Some(held.command_id),
                        "{label}"
                    );
                }
            }
        }
    }

    #[test]
    fn conflicts_and_overrides_are_evented_and_audited() {
        let (gate, ctx) = gate_with_ctx();
        let held = mount_cmd(policy(), Priority::Exposure);
        let _permit = gate.authorize(&held).unwrap();
        let plain = mount_cmd(operator(false), Priority::Exposure);
        let err = gate.authorize(&plain).unwrap_err();
        assert_eq!(err.http_status(), 409);
        assert_eq!(
            ctx.stages(plain.command_id),
            vec![AuditStage::Proposed, AuditStage::Rejected]
        );
        let over = mount_cmd(operator(true), Priority::Exposure);
        let _over_permit = gate.authorize(&over).unwrap();
        assert_eq!(
            ctx.stages(over.command_id),
            vec![AuditStage::Proposed, AuditStage::Validated]
        );
        assert_eq!(
            ctx.event_names(),
            vec!["ownership_conflict", "override_applied"]
        );
        let audits = lock(&ctx.records);
        let validated = audits
            .iter()
            .find_map(|r| match r {
                GroundTruthRecord::Audit(a)
                    if a.command_id == over.command_id && a.stage == AuditStage::Validated =>
                {
                    Some(a.clone())
                }
                _ => None,
            })
            .unwrap();
        assert!(validated.override_applied);
        assert!(validated.annotations[0].starts_with("preempted"));
    }

    #[test]
    fn permits_release_on_drop_and_on_panic() {
        let (gate, _ctx) = gate_with_ctx();
        let focuser = Domain::Device(DeviceKind::Focuser);
        let cmd = GatedCommand::new(
            GatedOp::Focuser(FocuserCmd::MoveTo(10)),
            operator(false),
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        let permit = gate.authorize(&cmd).unwrap();
        assert!(gate.holder(focuser).is_some());
        let joined = std::thread::spawn(move || {
            let _held = permit;
            panic!("worker panic while holding the permit");
        })
        .join();
        assert!(joined.is_err());
        assert!(gate.holder(focuser).is_none(), "released by unwinding");
        // Compound operations claim every domain and release them together.
        let handshake = GatedCommand::new(
            GatedOp::FilterHandshake {
                target: crate::equipment::commands::FilterTarget::Slot(2),
            },
            policy(),
            AutonomyMode::Bounded,
            Priority::Recommendation,
        );
        let permit = gate.authorize(&handshake).unwrap();
        for kind in [
            DeviceKind::FilterWheel,
            DeviceKind::Focuser,
            DeviceKind::Guider,
        ] {
            assert!(gate.holder(Domain::Device(kind)).is_some());
        }
        drop(permit);
        assert!(gate.holder(Domain::Device(DeviceKind::Guider)).is_none());
    }

    #[test]
    fn child_permits_need_a_parent_that_still_holds_the_domain() {
        let (gate, _ctx) = gate_with_ctx();
        let af = GatedCommand::new(
            GatedOp::Autofocus { run_id: None },
            CommandSource::Trigger { rule: "af".into() },
            AutonomyMode::Bounded,
            Priority::Autofocus,
        );
        let parent = gate.authorize(&af).unwrap();
        let step = GatedCommand::new(
            GatedOp::Focuser(FocuserCmd::MoveBy(50)),
            af.source.clone(),
            AutonomyMode::Bounded,
            Priority::Autofocus,
        );
        assert!(gate.child_permit(&parent, &step).is_ok());
        // A step on a domain the parent never claimed is refused.
        let mount_step = GatedCommand::new(
            GatedOp::Mount(MountCmd::SetTracking(false)),
            af.source.clone(),
            AutonomyMode::Bounded,
            Priority::Autofocus,
        );
        assert!(matches!(
            gate.child_permit(&parent, &mount_step),
            Err(GateError::OwnershipConflict { holder: None, .. })
        ));
        // An operator override on the focuser preempts the run; its next step
        // is refused.
        let manual = GatedCommand::new(
            GatedOp::Focuser(FocuserCmd::MoveTo(1000)),
            operator(true),
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        let _manual = gate.authorize(&manual).unwrap();
        assert_eq!(gate.preempted_by(af.command_id), Some(manual.command_id));
        assert!(matches!(
            gate.child_permit(&parent, &step),
            Err(GateError::OwnershipConflict { holder: Some(h), .. }) if h == manual.command_id
        ));
    }

    struct BelowTen;

    impl SafetyCheck for BelowTen {
        fn name(&self) -> &'static str {
            "mount_limits"
        }
        fn applies_to(&self, cmd: &GatedCommand) -> bool {
            matches!(cmd.op, GatedOp::Mount(MountCmd::Slew { .. }))
        }
        fn check(
            &self,
            cmd: &GatedCommand,
            _snap: &EquipmentSnapshot,
        ) -> Result<(), SafetyRejection> {
            match &cmd.op {
                GatedOp::Mount(MountCmd::Slew { target }) if target.dec_deg < -10.0 => Err(
                    SafetyRejection::new(self.name(), "below the altitude limit"),
                ),
                _ => Ok(()),
            }
        }
        fn annotations(&self, cmd: &GatedCommand, _snap: &EquipmentSnapshot) -> Vec<String> {
            if cmd.source.is_operator() {
                vec!["limits_unchecked".into()]
            } else {
                Vec::new()
            }
        }
    }

    #[test]
    fn safety_checks_reject_after_ownership_and_annotate_accepts() {
        let (gate, ctx) = gate_with_ctx();
        gate.register_check(Arc::new(BelowTen));
        assert_eq!(
            gate.check_names(),
            vec!["camera_boundary", "safety_fail_safe", "mount_limits"]
        );
        let slew = |dec: f64, source: CommandSource| {
            GatedCommand::new(
                GatedOp::Mount(MountCmd::Slew {
                    target: crate::equipment::types::EqCoord::new(
                        1.0,
                        dec,
                        crate::equipment::types::EqFrame::J2000,
                    ),
                }),
                source,
                AutonomyMode::Manual,
                Priority::Exposure,
            )
        };
        let low = slew(-30.0, operator(false));
        let err = gate.authorize(&low).unwrap_err();
        assert_eq!(err.http_status(), 422);
        assert!(gate.holder(Domain::Device(DeviceKind::Mount)).is_none());
        assert_eq!(ctx.event_names(), vec!["safety_rejected"]);
        // Ownership is checked first: a held domain gives 409, not 422.
        let held = gate
            .authorize(&mount_cmd(policy(), Priority::Exposure))
            .unwrap();
        assert_eq!(
            gate.authorize(&slew(-30.0, policy()))
                .unwrap_err()
                .http_status(),
            409
        );
        drop(held);
        let ok = slew(20.0, operator(false));
        let _permit = gate.authorize(&ok).unwrap();
        let note = lock(&ctx.records).iter().find_map(|r| match r {
            GroundTruthRecord::Audit(a)
                if a.command_id == ok.command_id && a.stage == AuditStage::Validated =>
            {
                Some(a.annotations.clone())
            }
            _ => None,
        });
        assert_eq!(note, Some(vec!["limits_unchecked".to_string()]));
    }

    #[test]
    fn camera_boundary_and_fail_safe_builtins() {
        let (gate, ctx) = gate_with_ctx();
        let dither = || {
            GatedCommand::new(
                GatedOp::Guider(GuiderCmd::Dither {
                    amount: 5.0,
                    unit: crate::equipment::commands::PixelUnit::GuidePx,
                    ra_only: false,
                    settle: crate::equipment::commands::Settle::default(),
                }),
                policy(),
                AutonomyMode::Bounded,
                Priority::Recommendation,
            )
        };
        assert!(gate.authorize(&dither()).is_ok());
        lock(&ctx.snapshot).camera.exposing =
            Observed::fresh_at(true, Source::Derived, ctx.clock.as_ref());
        let err = gate.authorize(&dither()).unwrap_err();
        assert!(matches!(&err, GateError::SafetyRejected(r) if r.check == "camera_boundary"));
        lock(&ctx.snapshot).camera.exposing = Observed::not_set(Source::Derived);
        assert!(
            gate.authorize(&dither()).is_err(),
            "unknown refuses automatic"
        );
        // A monitor that is not Ready refuses new automatic motion only.
        lock(&ctx.snapshot).camera.exposing =
            Observed::fresh_at(false, Source::Derived, ctx.clock.as_ref());
        lock(&ctx.snapshot).safety.state = SafetyGate::Unknown;
        let err = gate
            .authorize(&mount_cmd(CommandSource::System, Priority::MountRecovery))
            .unwrap_err();
        assert!(matches!(&err, GateError::SafetyRejected(r) if r.check == "safety_fail_safe"));
        assert!(gate
            .authorize(&mount_cmd(CommandSource::Safety, Priority::Safety))
            .is_ok());
    }

    #[test]
    fn resume_blocks_refuse_automatic_sources_with_423() {
        let (gate, ctx) = gate_with_ctx();
        let block = ctx.raise_block(BlockRequest::new(
            ResumeBlockKind::FilterHandshakeIncomplete,
            "wheel timeout",
        ));
        let capture = |source: CommandSource| {
            GatedCommand::new(
                GatedOp::Capture {
                    capture_type: Some("light".into()),
                },
                source,
                AutonomyMode::Bounded,
                Priority::Exposure,
            )
        };
        let auto = capture(CommandSource::Trigger {
            rule: "sequence".into(),
        });
        let err = gate.authorize(&auto).unwrap_err();
        assert_eq!(err.http_status(), 423);
        assert_eq!(
            err,
            GateError::ResumeBlocked {
                blocks: vec![ResumeBlockKind::FilterHandshakeIncomplete]
            }
        );
        // Operator captures still run.
        let manual = gate.authorize(&capture(operator(false))).unwrap();
        drop(manual);
        // Clearing is audited; safety blocks cannot be cleared by the operator.
        let cleared = gate.clear_resume_block(block.id, operator(false)).unwrap();
        assert_eq!(cleared.id, block.id);
        assert!(gate.authorize(&auto).is_ok());
        let unsafe_block =
            ctx.raise_block(BlockRequest::new(ResumeBlockKind::SafetyUnsafe, "rain"));
        assert!(matches!(
            gate.clear_resume_block(unsafe_block.id, operator(false)),
            Err(GateError::Refused { .. })
        ));
    }

    #[test]
    fn relay_policy_and_shutdown_refusals() {
        let (gate, _ctx) = gate_with_ctx();
        let pulse = GatedCommand::new(
            GatedOp::Mount(MountCmd::PulseGuide {
                dir: GuideDirection::North,
                ms: 300,
            }),
            CommandSource::Relay { overriding: false },
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        assert!(matches!(
            gate.authorize(&pulse),
            Err(GateError::Refused { reason }) if reason.contains("relay")
        ));
        let algo = GatedCommand::new(
            GatedOp::Guider(GuiderCmd::SetAlgoParam {
                axis: crate::equipment::commands::Axis::Ra,
                name: "aggression".into(),
                value: 0.6,
            }),
            policy(),
            AutonomyMode::Bounded,
            Priority::Recommendation,
        );
        assert!(matches!(
            gate.authorize(&algo),
            Err(GateError::Refused { .. })
        ));
        gate.set_config(GateConfig {
            allow_policy_actuation: true,
            ..GateConfig::default()
        });
        assert!(gate.authorize(&algo).is_ok());
        gate.begin_shutdown();
        assert!(matches!(
            gate.authorize(&mount_cmd(operator(false), Priority::Exposure)),
            Err(GateError::ShuttingDown)
        ));
        // Abort still passes, even on a held domain and during shutdown.
        let abort = GatedCommand::new(
            GatedOp::Mount(MountCmd::AbortSlew),
            operator(false),
            AutonomyMode::Manual,
            Priority::Safety,
        );
        assert!(gate.authorize(&abort).is_ok());
    }

    #[tokio::test]
    async fn submit_audits_every_stage_and_unknown_outcomes_block_resumption() {
        let (gate, ctx) = gate_with_ctx();
        let (handle, fake) = fake_focuser(FocuserSnapshot::default());
        let cmd = GatedCommand::new(
            GatedOp::Focuser(FocuserCmd::MoveBy(40)),
            operator(false),
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        let id = cmd.command_id;
        let submitted = gate
            .submit(&handle, cmd, FocuserCmd::MoveBy(40))
            .await
            .unwrap();
        assert!(submitted.pre_motion.is_none());
        assert_eq!(submitted.ticket.wait().await, OpOutcome::Succeeded);
        assert_eq!(
            ctx.stages(id),
            vec![
                AuditStage::Proposed,
                AuditStage::Validated,
                AuditStage::Executed,
                AuditStage::Outcome
            ]
        );
        // The worker refusing a command is audited as rejected.
        fake.set_responder(|_, _| FakeResponse::reject(DeviceError::invalid_value("bounds")));
        let refused = GatedCommand::new(
            GatedOp::Focuser(FocuserCmd::MoveTo(-1)),
            operator(false),
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        let err = gate
            .submit(&handle, refused.clone(), FocuserCmd::MoveTo(-1))
            .await
            .unwrap_err();
        assert_eq!(err.http_status(), 400);
        assert_eq!(err.code(DeviceKind::Focuser), "focuser_invalid_value");
        assert_eq!(
            ctx.stages(refused.command_id),
            vec![
                AuditStage::Proposed,
                AuditStage::Validated,
                AuditStage::Rejected
            ]
        );
        // An unknown outcome raises the outcome_unknown block.
        fake.set_responder(|_, _| {
            FakeResponse::finish(OpOutcome::Unknown(DeviceError::outcome_unknown(
                "lost mid-move",
            )))
        });
        let lost = GatedCommand::new(
            GatedOp::Focuser(FocuserCmd::MoveBy(5)),
            operator(false),
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        let submitted = gate
            .submit(&handle, lost, FocuserCmd::MoveBy(5))
            .await
            .unwrap();
        assert!(matches!(
            submitted.ticket.wait().await,
            OpOutcome::Unknown(_)
        ));
        assert_eq!(
            ctx.blocks.active()[0].kind,
            ResumeBlockKind::OutcomeUnknown {
                device: DeviceKind::Focuser,
                op: OpKind::FocuserMove
            }
        );
        // Abort is audited and reaches the device's abort path.
        let ticket = gate.abort(&handle, operator(false));
        assert!(ticket.abort_path_ran);
        assert!(ticket.outcome().await.is_ok());
        assert_eq!(fake.abort_path().calls(), 1);
    }
}

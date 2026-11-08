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
//! block when the outcome is unknown. The device command it sends is always
//! the one inside the [`GatedOp`] the checks validated
//! ([`GatedDeviceCommand`]). A compound operation (flip, center, autofocus,
//! filter handshake) is authorised once and its device steps go through
//! [`CommandGate::submit_step`], so every step's own target is checked too.
//!
//! [`AbortEpochs`] make an abort or e-stop also stop commands that were
//! authorised before it but still wait in a worker's queue (D19), and stop
//! a compound operation authorised before it from moving that device again
//! ([`CommandGate::child_permit`] refuses its motion steps). Abort is
//! audited at every stage, outcome included ([`GateAbort`]).

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
use std::time::{Duration, Instant};

use tokio::sync::oneshot;
use uuid::Uuid;

use crate::control::action_arbiter::{arbitrate, Arbitration};
use crate::control::ownership::{DomainClaim, DomainLedger};

use super::commands::{
    CoverCalibratorCmd, Domain, DomeCmd, FilterWheelCmd, FocuserCmd, GateError, GatedCommand,
    GatedOp, GuiderCmd, MountCmd, RotatorCmd, SafetyRejection, SwitchCmd,
};
use super::coord::{self, CoordError, PreMotion};
use super::events::{AuditRecord, AuditStage, EquipmentEvent, EventKind, GroundTruthRecord};
use super::handle::{DeviceHandle, GuiderHandle, OpTicket, Permit};
use super::ops::OpKind;
use super::resume::{BlockRequest, ClearError};
use super::snapshot::{EquipmentSnapshot, ResumeBlock, ResumeBlockKind};
use super::traits::{Clock, SafetyCheck};
use super::types::{
    kind_name, AutonomyMode, CommandId, CommandSource, DeviceError, DeviceKind, DeviceResult,
    ErrorClass, OpOutcome, Priority,
};
use super::worker::DeviceCommand;

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

// ---------------------------------------------------------------------------
// Device commands taken from the validated op
// ---------------------------------------------------------------------------

/// A device command type the gate can execute. The command is taken from
/// the [`GatedOp`] that the safety checks validated, so what reaches the
/// hardware is always what was checked.
pub trait GatedDeviceCommand: Sized + Send + 'static {
    /// The command `op` carries, or `None` when `op` is not a command of
    /// this kind (a compound operation, or another device kind).
    fn from_gated(op: &GatedOp) -> Option<Self>;
}

macro_rules! gated_device_command {
    ($($variant:ident => $ty:ty),* $(,)?) => {
        $(impl GatedDeviceCommand for $ty {
            fn from_gated(op: &GatedOp) -> Option<Self> {
                match op {
                    GatedOp::$variant(cmd) => Some(cmd.clone()),
                    _ => None,
                }
            }
        })*
    };
}

gated_device_command!(
    Mount => MountCmd,
    Focuser => FocuserCmd,
    FilterWheel => FilterWheelCmd,
    Guider => GuiderCmd,
    Rotator => RotatorCmd,
    CoverCalibrator => CoverCalibratorCmd,
    Switch => SwitchCmd,
    Dome => DomeCmd,
);

/// The parent of a compound operation's step: what
/// [`CommandGate::child_permit_of`] needs from the parent's permit. Unlike
/// a `&Permit` it can be held across an `.await` in a `Send` future.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepOwner {
    pub command_id: CommandId,
    pub domains: Vec<Domain>,
}

impl StepOwner {
    pub fn of(permit: &Permit) -> Self {
        Self {
            command_id: permit.command_id(),
            domains: permit.domains().to_vec(),
        }
    }

    pub fn holds(&self, domain: Domain) -> bool {
        self.domains.contains(&domain)
    }
}

// ---------------------------------------------------------------------------
// Abort epochs
// ---------------------------------------------------------------------------

/// Abort epochs (D19, §2.9): an abort or e-stop must also stop commands
/// that were authorised before it but still wait in a worker's queue (for
/// example behind a slow driver call).
///
/// Every abort and e-stop issued through the gate bumps the epoch of each
/// device kind it targets, before the worker's cancel is queued. Every
/// permit the gate issues is stamped with the epochs of its device domains
/// as they were when its authorisation began, and the stamp is forgotten
/// when the permit drops. A worker that handles a cancel reads its kind's
/// epoch; from then on it refuses a motion command whose stamp is older
/// (resolved `Aborted { by }`, never started). A command authorised after
/// the abort carries the new epoch and runs normally.
#[derive(Debug, Default)]
pub struct AbortEpochs {
    inner: Mutex<EpochState>,
}

#[derive(Debug, Default)]
struct EpochState {
    epochs: BTreeMap<DeviceKind, u64>,
    issued: HashMap<CommandId, Vec<(DeviceKind, u64)>>,
}

impl AbortEpochs {
    pub fn new() -> Self {
        Self::default()
    }

    /// The current epoch of `kind` (0 until its first abort).
    pub fn epoch(&self, kind: DeviceKind) -> u64 {
        lock(&self.inner).epochs.get(&kind).copied().unwrap_or(0)
    }

    /// An abort or e-stop targets `kind`: start a new epoch. Call before
    /// the device's cancel is queued.
    pub fn bump(&self, kind: DeviceKind) -> u64 {
        let mut inner = lock(&self.inner);
        let epoch = inner.epochs.entry(kind).or_insert(0);
        *epoch += 1;
        *epoch
    }

    /// The current epochs of `kinds`, to stamp a permit with.
    pub fn stamp(&self, kinds: &[DeviceKind]) -> Vec<(DeviceKind, u64)> {
        let inner = lock(&self.inner);
        kinds
            .iter()
            .map(|kind| (*kind, inner.epochs.get(kind).copied().unwrap_or(0)))
            .collect()
    }

    fn record(&self, command_id: CommandId, stamp: Vec<(DeviceKind, u64)>) {
        if !stamp.is_empty() {
            lock(&self.inner).issued.insert(command_id, stamp);
        }
    }

    fn forget(&self, command_id: CommandId) {
        lock(&self.inner).issued.remove(&command_id);
    }

    /// The epoch of `kind` when `command_id`'s authorisation began, while
    /// its permit is alive. `None` for a permit the gate did not stamp
    /// (an unmanaged permit, or a domain the command does not claim).
    pub fn issued_epoch(&self, command_id: CommandId, kind: DeviceKind) -> Option<u64> {
        lock(&self.inner)
            .issued
            .get(&command_id)
            .and_then(|stamp| stamp.iter().find(|(k, _)| *k == kind).map(|(_, e)| *e))
    }

    /// Live stamps (diagnostics and tests).
    pub fn outstanding(&self) -> usize {
        lock(&self.inner).issued.len()
    }
}

/// True for an op that starts motion (the abort epochs refuse such a
/// command authorised before an abort, D19).
pub fn starts_motion(op: &GatedOp) -> bool {
    match op {
        GatedOp::Mount(cmd) => cmd.starts_motion(),
        GatedOp::Focuser(cmd) => cmd.starts_motion(),
        GatedOp::FilterWheel(cmd) => cmd.starts_motion(),
        GatedOp::Guider(cmd) => cmd.starts_motion(),
        GatedOp::Rotator(cmd) => cmd.starts_motion(),
        GatedOp::CoverCalibrator(cmd) => cmd.starts_motion(),
        GatedOp::Switch(cmd) => cmd.starts_motion(),
        GatedOp::Dome(cmd) => cmd.starts_motion(),
        other => other.kind().moves_hardware(),
    }
}

/// The device kinds a command's stamp covers.
fn stamped_kinds(cmd: &GatedCommand) -> Vec<DeviceKind> {
    let mut kinds: Vec<DeviceKind> = cmd
        .domains
        .iter()
        .filter_map(|domain| match domain {
            Domain::Device(kind) => Some(*kind),
            Domain::CameraBoundary => None,
        })
        .collect();
    kinds.sort();
    kinds.dedup();
    kinds
}

// ---------------------------------------------------------------------------
// Audited abort
// ---------------------------------------------------------------------------

/// How long [`CommandGate::abort`] waits for the abort path before it
/// audits the outcome as unknown (the Alpaca abort agent times out at 5 s).
pub const ABORT_OUTCOME_TIMEOUT: Duration = Duration::from_secs(10);

/// What [`CommandGate::abort`] returns: the handle's
/// [`AbortTicket`](super::handle::AbortTicket) fields, with the abort
/// path's result also audited as the `outcome` stage (H21, D19).
#[derive(Debug)]
pub struct GateAbort {
    /// The audited abort command.
    pub command_id: CommandId,
    /// The id the worker resolves the cancelled operation with
    /// (`Aborted { by }`).
    pub ticket_id: CommandId,
    /// A `Cancel` was queued at the head of the worker's queue.
    pub cancel_sent: bool,
    /// The out-of-band abort path was started.
    pub abort_path_ran: bool,
    result: oneshot::Receiver<DeviceResult<()>>,
}

impl GateAbort {
    /// The abort path's result (audited once it arrives).
    pub async fn outcome(self) -> DeviceResult<()> {
        self.result.await.unwrap_or_else(|_| {
            Err(DeviceError::outcome_unknown(
                "the abort path stopped without reporting a result",
            ))
        })
    }

    /// Non-blocking check.
    pub fn try_outcome(&mut self) -> Option<DeviceResult<()>> {
        match self.result.try_recv() {
            Ok(result) => Some(result),
            Err(oneshot::error::TryRecvError::Empty) => None,
            Err(oneshot::error::TryRecvError::Closed) => Some(Err(DeviceError::outcome_unknown(
                "the abort path stopped without reporting a result",
            ))),
        }
    }
}

/// Run `f` on a blocking thread: tokio's blocking pool inside a runtime, a
/// dedicated thread otherwise.
pub(crate) fn run_detached(name: &str, f: impl FnOnce() + Send + 'static) {
    match tokio::runtime::Handle::try_current() {
        Ok(runtime) => drop(runtime.spawn_blocking(f)),
        Err(_) => {
            let _ = std::thread::Builder::new().name(name.into()).spawn(f);
        }
    }
}

fn abort_outcome(result: &DeviceResult<()>) -> OpOutcome {
    match result {
        Ok(()) => OpOutcome::Succeeded,
        Err(err) if err.class == ErrorClass::OutcomeUnknown => OpOutcome::Unknown(err.clone()),
        Err(err) => OpOutcome::Failed(err.clone()),
    }
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
    aborts: Arc<AbortEpochs>,
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
            aborts: Arc::new(AbortEpochs::new()),
        }
    }

    /// The abort epochs workers compare permits against (D19). Pass them to
    /// every worker this gate's permits reach (`WorkerDeps::aborts`).
    pub fn abort_epochs(&self) -> Arc<AbortEpochs> {
        Arc::clone(&self.aborts)
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

    /// A release callback that forgets `command_id`'s abort-epoch stamp,
    /// then runs `then`.
    fn stamped_release(
        &self,
        command_id: CommandId,
        stamp: Vec<(DeviceKind, u64)>,
        then: Option<Box<dyn FnOnce() + Send>>,
    ) -> Option<Box<dyn FnOnce() + Send>> {
        if stamp.is_empty() {
            return then;
        }
        self.aborts.record(command_id, stamp);
        let aborts = Arc::downgrade(&self.aborts);
        Some(Box::new(move || {
            if let Some(aborts) = aborts.upgrade() {
                aborts.forget(command_id);
            }
            if let Some(then) = then {
                then();
            }
        }))
    }

    /// Authorise a command (§2.9). On success the returned permit holds the
    /// command's domains until it drops.
    pub fn authorize(&self, cmd: &GatedCommand) -> Result<Permit, GateError> {
        // The stamp is taken before anything else, so an abort or e-stop that
        // lands while this command is being authorised still stops it (D19).
        let stamp = self.aborts.stamp(&stamped_kinds(cmd));
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
        Ok(Permit::new(
            cmd,
            self.stamped_release(command_id, stamp, Some(release)),
        ))
    }

    /// A permit for one step of a compound operation (a focuser move inside
    /// an autofocus run, the wheel move inside a filter handshake). The
    /// parent must still hold every domain the step claims, and must not
    /// have been preempted; the step's own safety checks run. The permit
    /// holds no ownership of its own: the parent's does.
    pub fn child_permit(&self, parent: &Permit, cmd: &GatedCommand) -> Result<Permit, GateError> {
        self.child_permit_of(&StepOwner::of(parent), cmd)
    }

    /// [`CommandGate::child_permit`] for a parent known by its id and
    /// domains.
    pub fn child_permit_of(
        &self,
        parent: &StepOwner,
        cmd: &GatedCommand,
    ) -> Result<Permit, GateError> {
        let stamp = self.aborts.stamp(&stamped_kinds(cmd));
        self.audit(
            cmd,
            AuditStage::Proposed,
            true,
            Some(format!("step of {}", parent.command_id)),
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
        if let Some(kind) = self.aborted_since(parent, cmd) {
            return Err(self.reject(
                cmd,
                GateError::Refused {
                    reason: format!(
                        "the {} was aborted after {} was authorised; its motion steps no longer run",
                        kind_name(kind),
                        parent.command_id
                    ),
                },
            ));
        }
        let domains = ownership_domains(cmd);
        let refusal = {
            let ledger = lock(&self.ledger);
            if let Some(by) = ledger.preempted_by(parent.command_id) {
                let domain = domains
                    .first()
                    .copied()
                    .or_else(|| parent.domains.first().copied())
                    .unwrap_or(Domain::CameraBoundary);
                Some(GateError::OwnershipConflict {
                    domain,
                    holder: Some(by),
                    holder_source: ledger.holder(domain).map(|claim| claim.source.clone()),
                })
            } else {
                domains.iter().find_map(|&domain| {
                    let holder = ledger.holder(domain);
                    (holder.map(|claim| claim.command_id) != Some(parent.command_id)).then(|| {
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
        Ok(Permit::new(
            cmd,
            self.stamped_release(cmd.command_id, stamp, None),
        ))
    }

    /// A device the step would move was aborted (or e-stopped) after the
    /// parent was authorised (D19): a compound operation does not carry on
    /// moving it. Stops, reads and sets stay allowed.
    fn aborted_since(&self, parent: &StepOwner, cmd: &GatedCommand) -> Option<DeviceKind> {
        if !starts_motion(&cmd.op) {
            return None;
        }
        stamped_kinds(cmd).into_iter().find(|&kind| {
            self.aborts
                .issued_epoch(parent.command_id, kind)
                .is_some_and(|issued| issued < self.aborts.epoch(kind))
        })
    }

    // -- execution -------------------------------------------------------------

    /// The device command inside `cmd`, or the audited refusal of a command
    /// that carries none for this kind (a compound operation).
    fn device_command<C: GatedDeviceCommand>(
        &self,
        cmd: &GatedCommand,
        kind: DeviceKind,
        audit_proposed: bool,
    ) -> Result<C, SubmitError> {
        match C::from_gated(&cmd.op) {
            Some(device_cmd) => Ok(device_cmd),
            None => {
                if audit_proposed {
                    self.audit(cmd, AuditStage::Proposed, true, None, Vec::new(), None);
                }
                Err(SubmitError::Gate(self.reject(
                    cmd,
                    GateError::Refused {
                        reason: format!(
                            "{:?} is not a {} command; a compound operation is authorised \
                             once and its device steps are sent with submit_step",
                            cmd.kind(),
                            kind_name(kind)
                        ),
                    },
                )))
            }
        }
    }

    /// Authorise, coordinate (D20), send, and audit a device command. The
    /// command sent is the one inside `cmd.op`, the op the safety checks
    /// validated. The returned ticket resolves with the outcome, which is
    /// also audited; an `Unknown` outcome raises an `outcome_unknown` resume
    /// block. Must run inside a tokio runtime.
    pub async fn submit<C, S>(
        &self,
        handle: &DeviceHandle<C, S>,
        cmd: GatedCommand,
    ) -> Result<Submitted, SubmitError>
    where
        C: GatedDeviceCommand,
        S: Send + Sync + 'static,
    {
        let device_cmd = self.device_command::<C>(&cmd, handle.kind(), true)?;
        let permit = self.authorize(&cmd)?;
        let owner = StepOwner::of(&permit);
        self.execute(handle, cmd, device_cmd, permit, owner).await
    }

    /// One device step of a compound operation (the slew of a flip or a
    /// center, the focuser move of an autofocus run): a child permit of
    /// `parent` (the step's own checks run), the D20 pre-motion stop under
    /// the parent's guider claim when it holds one, then the same send and
    /// audit as [`CommandGate::submit`].
    pub fn submit_step<'a, C, S>(
        &'a self,
        handle: &'a DeviceHandle<C, S>,
        parent: &Permit,
        step: GatedCommand,
    ) -> impl std::future::Future<Output = Result<Submitted, SubmitError>> + Send + 'a
    where
        C: GatedDeviceCommand,
        S: Send + Sync + 'static,
    {
        let owner = StepOwner::of(parent);
        let prepared = self
            .device_command::<C>(&step, handle.kind(), false)
            .and_then(|device_cmd| {
                self.child_permit_of(&owner, &step)
                    .map(|permit| (device_cmd, permit))
                    .map_err(SubmitError::Gate)
            });
        async move {
            let (device_cmd, permit) = prepared?;
            self.execute(handle, step, device_cmd, permit, owner).await
        }
    }

    async fn execute<C, S>(
        &self,
        handle: &DeviceHandle<C, S>,
        cmd: GatedCommand,
        device_cmd: C,
        permit: Permit,
        owner: StepOwner,
    ) -> Result<Submitted, SubmitError>
    where
        C: Send + 'static,
        S: Send + Sync + 'static,
    {
        let pre_motion = if coord::needs_guider_stop(&cmd.op) {
            let guider = self.ctx.guider();
            let clock = self.ctx.clock();
            let timeout = self.config().stop_guiding_timeout;
            match coord::stop_guiding_before_motion(
                self,
                guider.as_ref(),
                &cmd,
                Some(&owner),
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
            if report.was_guiding {
                annotations.push(format!("guider_stop_sent={}", report.stop_sent));
            }
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

    /// Abort a device (D19): audited at every stage (the outcome once the
    /// abort path reports, or as unknown after [`ABORT_OUTCOME_TIMEOUT`]),
    /// never refused by ownership or checks, and attempted on a degraded or
    /// lost handle until it is retired. Commands authorised before the abort
    /// that still wait in the worker's queue are refused by the worker
    /// ([`AbortEpochs`]).
    pub fn abort<C, S>(&self, handle: &DeviceHandle<C, S>, source: CommandSource) -> GateAbort {
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
        // Before the cancel is queued, so the worker sees the new epoch.
        self.aborts.bump(handle.kind());
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
        let (tx, rx) = oneshot::channel();
        let audited = GateAbort {
            command_id: cmd.command_id,
            ticket_id: ticket.command_id,
            cancel_sent: ticket.cancel_sent,
            abort_path_ran: ticket.abort_path_ran,
            result: rx,
        };
        let ctx = Arc::clone(&self.ctx);
        let mut ticket = ticket;
        run_detached("eq-abort-audit", move || {
            let deadline = Instant::now() + ABORT_OUTCOME_TIMEOUT;
            let result = loop {
                if let Some(result) = ticket.try_outcome() {
                    break result;
                }
                if Instant::now() >= deadline {
                    break Err(DeviceError::outcome_unknown(format!(
                        "the abort path had not reported after {} s",
                        ABORT_OUTCOME_TIMEOUT.as_secs()
                    )));
                }
                std::thread::sleep(Duration::from_millis(5));
            };
            let outcome = abort_outcome(&result);
            audit_with(
                ctx.as_ref(),
                &cmd,
                AuditStage::Outcome,
                outcome.is_success(),
                outcome_reason(&outcome),
                Vec::new(),
                Some(outcome),
            );
            let _ = tx.send(result);
        });
        audited
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

    /// The operator's manual safety resume (§3.10, Tier 1) asked for through
    /// the clear of a `safety_*` block `id`, audited with the same stages as
    /// [`CommandGate::clear_resume_block`]. `resume` is the safety layer's
    /// own clear (`SafetySupervisor::request_resume`): it clears every
    /// safety block, and only on a confirmed safe reading or with no monitor,
    /// so its refusal reason becomes a 409 here.
    pub fn resume_safety(
        &self,
        id: Uuid,
        source: CommandSource,
        resume: impl FnOnce(&CommandSource) -> Result<Vec<ResumeBlock>, String>,
    ) -> Result<Vec<ResumeBlock>, GateError> {
        let cmd = GatedCommand::new(
            GatedOp::ClearResumeBlock { id },
            source.clone(),
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        self.audit(&cmd, AuditStage::Proposed, true, None, Vec::new(), None);
        match resume(&source) {
            Ok(cleared) => {
                self.audit(&cmd, AuditStage::Validated, true, None, Vec::new(), None);
                self.record_outcome(&cmd, &OpOutcome::Succeeded);
                Ok(cleared)
            }
            Err(reason) => Err(self.reject(&cmd, GateError::Refused { reason })),
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
        let submitted = gate.submit(&handle, cmd).await.unwrap();
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
        let err = gate.submit(&handle, refused.clone()).await.unwrap_err();
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
        let submitted = gate.submit(&handle, lost).await.unwrap();
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
        // Abort is audited at every stage, outcome included, and reaches the
        // device's abort path.
        let ticket = gate.abort(&handle, operator(false));
        let abort_id = ticket.command_id;
        assert!(ticket.abort_path_ran);
        assert!(ticket.outcome().await.is_ok());
        assert_eq!(fake.abort_path().calls(), 1);
        assert_eq!(
            ctx.stages(abort_id),
            vec![
                AuditStage::Proposed,
                AuditStage::Validated,
                AuditStage::Executed,
                AuditStage::Outcome
            ]
        );
        // A failing abort path is audited as a failed outcome.
        fake.abort_path()
            .set_result(Err(DeviceError::transport("abort agent refused")));
        let failed = gate.abort(&handle, operator(false));
        let failed_id = failed.command_id;
        assert!(failed.outcome().await.is_err());
        let outcome = lock(&ctx.records).iter().find_map(|r| match r {
            GroundTruthRecord::Audit(a)
                if a.command_id == failed_id && a.stage == AuditStage::Outcome =>
            {
                Some(a.outcome.clone())
            }
            _ => None,
        });
        assert!(matches!(outcome, Some(Some(OpOutcome::Failed(_)))));
    }

    #[tokio::test]
    async fn submit_sends_exactly_the_command_the_checks_validated() {
        let (gate, ctx) = gate_with_ctx();
        gate.register_check(Arc::new(BelowTen));
        let (mount, mount_fake) = crate::equipment::testing::fake_handles::fake_mount(
            crate::equipment::snapshot::MountSnapshot::default(),
        );
        let slew = |dec: f64| {
            GatedCommand::new(
                GatedOp::Mount(MountCmd::Slew {
                    target: crate::equipment::types::EqCoord::new(
                        1.0,
                        dec,
                        crate::equipment::types::EqFrame::J2000,
                    ),
                }),
                CommandSource::Trigger { rule: "seq".into() },
                AutonomyMode::Bounded,
                Priority::Exposure,
            )
        };
        let low = gate.submit(&mount, slew(-80.0)).await.unwrap_err();
        assert_eq!(low.http_status(), 422);
        assert!(
            mount_fake.commands().is_empty(),
            "a refused target never moves"
        );
        let ok = gate.submit(&mount, slew(20.0)).await.unwrap();
        assert_eq!(ok.ticket.wait().await, OpOutcome::Succeeded);
        assert_eq!(
            mount_fake.commands(),
            vec![MountCmd::Slew {
                target: crate::equipment::types::EqCoord::new(
                    1.0,
                    20.0,
                    crate::equipment::types::EqFrame::J2000
                )
            }]
        );
        // A step of a compound operation is checked on its own target too.
        let center = GatedCommand::new(
            GatedOp::Center {
                target: crate::equipment::types::EqCoord::new(
                    1.0,
                    -80.0,
                    crate::equipment::types::EqFrame::J2000,
                ),
                tolerance_arcmin: None,
                max_attempts: None,
            },
            CommandSource::Trigger { rule: "seq".into() },
            AutonomyMode::Bounded,
            Priority::MountRecovery,
        );
        let parent = gate.authorize(&center).unwrap();
        let mut step = slew(-80.0);
        step.priority = Priority::MountRecovery;
        let err = gate.submit_step(&mount, &parent, step).await.unwrap_err();
        assert_eq!(err.http_status(), 422);
        assert_eq!(mount_fake.commands().len(), 1);
        drop(parent);
        let _ = ctx;
    }

    #[test]
    fn submit_futures_are_send() {
        fn assert_send<T: Send>(_: &T) {}
        let (gate, _ctx) = gate_with_ctx();
        let (handle, _fake) = fake_focuser(FocuserSnapshot::default());
        let cmd = GatedCommand::new(
            GatedOp::Focuser(FocuserCmd::MoveBy(1)),
            operator(false),
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        let submit = gate.submit(&handle, cmd.clone());
        assert_send(&submit);
        drop(submit);
        let parent = gate
            .authorize(&GatedCommand::new(
                GatedOp::Autofocus { run_id: None },
                operator(false),
                AutonomyMode::Manual,
                Priority::Autofocus,
            ))
            .unwrap();
        let step = gate.submit_step(&handle, &parent, cmd);
        assert_send(&step);
    }

    #[tokio::test]
    async fn a_compound_operation_stops_moving_a_device_aborted_after_it_began() {
        let (gate, _ctx) = gate_with_ctx();
        let (focuser, _fake) = fake_focuser(FocuserSnapshot::default());
        let source = CommandSource::Trigger { rule: "af".into() };
        let af = GatedCommand::new(
            GatedOp::Autofocus { run_id: None },
            source.clone(),
            AutonomyMode::Bounded,
            Priority::Autofocus,
        );
        let parent = gate.authorize(&af).unwrap();
        let step = |cmd: FocuserCmd| {
            GatedCommand::new(
                GatedOp::Focuser(cmd),
                source.clone(),
                AutonomyMode::Bounded,
                Priority::Autofocus,
            )
        };
        assert!(gate
            .child_permit(&parent, &step(FocuserCmd::MoveBy(5)))
            .is_ok());
        // The operator aborts the focuser mid-run.
        let abort = gate.abort(&focuser, operator(false));
        assert!(abort.outcome().await.is_ok());
        let err = gate
            .child_permit(&parent, &step(FocuserCmd::MoveBy(5)))
            .unwrap_err();
        assert!(
            matches!(&err, GateError::Refused { reason } if reason.contains("aborted")),
            "{err}"
        );
        // Stops and plain sets still run as steps.
        assert!(gate
            .child_permit(&parent, &step(FocuserCmd::SetTempComp(false)))
            .is_ok());
        assert!(gate.child_permit(&parent, &step(FocuserCmd::Halt)).is_ok());
        // A run authorised after the abort moves normally.
        drop(parent);
        let again = gate.authorize(&af).unwrap();
        assert!(gate
            .child_permit(&again, &step(FocuserCmd::MoveBy(5)))
            .is_ok());
        assert!(starts_motion(&GatedOp::Mount(MountCmd::SetTracking(true))));
        assert!(!starts_motion(&GatedOp::Mount(MountCmd::SetTracking(
            false
        ))));
        assert!(!starts_motion(&GatedOp::Mount(MountCmd::MoveAxis {
            axis: 0,
            rate_deg_s: 0.0
        })));
        assert!(starts_motion(&GatedOp::Flip));
        assert!(!starts_motion(&GatedOp::Guider(GuiderCmd::Stop)));
    }

    #[test]
    fn permits_carry_abort_epoch_stamps_until_they_drop() {
        let (gate, _ctx) = gate_with_ctx();
        let epochs = gate.abort_epochs();
        let focuser = || {
            GatedCommand::new(
                GatedOp::Focuser(FocuserCmd::MoveTo(10)),
                operator(false),
                AutonomyMode::Manual,
                Priority::Exposure,
            )
        };
        let before = focuser();
        let permit = gate.authorize(&before).unwrap();
        assert_eq!(
            epochs.issued_epoch(before.command_id, DeviceKind::Focuser),
            Some(0)
        );
        assert_eq!(
            epochs.issued_epoch(before.command_id, DeviceKind::Mount),
            None
        );
        assert_eq!(epochs.bump(DeviceKind::Focuser), 1);
        drop(permit);
        assert_eq!(epochs.outstanding(), 0, "the stamp is forgotten on drop");
        let after = focuser();
        let permit = gate.authorize(&after).unwrap();
        assert_eq!(
            epochs.issued_epoch(after.command_id, DeviceKind::Focuser),
            Some(1)
        );
        // Child permits are stamped too; abort and halts are not.
        let af = GatedCommand::new(
            GatedOp::Autofocus { run_id: None },
            CommandSource::Trigger { rule: "af".into() },
            AutonomyMode::Bounded,
            Priority::Autofocus,
        );
        drop(permit);
        let parent = gate.authorize(&af).unwrap();
        let step = GatedCommand::new(
            GatedOp::Focuser(FocuserCmd::MoveBy(5)),
            af.source.clone(),
            AutonomyMode::Bounded,
            Priority::Autofocus,
        );
        let child = gate.child_permit(&parent, &step).unwrap();
        assert_eq!(
            epochs.issued_epoch(step.command_id, DeviceKind::Focuser),
            Some(1)
        );
        drop(child);
        assert_eq!(
            epochs.issued_epoch(step.command_id, DeviceKind::Focuser),
            None
        );
        let halt = GatedCommand::new(
            GatedOp::Focuser(FocuserCmd::Halt),
            operator(false),
            AutonomyMode::Manual,
            Priority::Safety,
        );
        let _halt = gate.authorize(&halt).unwrap();
        assert_eq!(
            epochs.issued_epoch(halt.command_id, DeviceKind::Focuser),
            None
        );
    }
}

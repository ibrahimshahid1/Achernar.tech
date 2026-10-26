//! Device commands and the gated-command envelope (spec §2.4, §2.9). Frozen
//! by WP0.
//!
//! Every motion command passes the `CommandGate` (WP1A-2) as a
//! [`GatedCommand`]: ownership, then arbiter, then [`SafetyCheck`] plug-ins,
//! then audit (D9). Abort and e-stop skip the first three steps and are
//! always audited.
//!
//! [`SafetyCheck`]: super::traits::SafetyCheck

use std::fmt;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::ops::OpKind;
use super::snapshot::ResumeBlockKind;
use super::types::{
    AltAz, AutonomyMode, CommandId, CommandSource, DeviceError, DeviceKind, DriveRate, EqCoord,
    ErrorClass, GuideDirection, PierSide, Priority, Site,
};

// ---------------------------------------------------------------------------
// Per-kind commands
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "cmd", content = "args", rename_all = "snake_case")]
pub enum MountCmd {
    /// The target carries its frame; the worker converts to the native frame.
    Slew {
        target: EqCoord,
    },
    SlewAltAz(AltAz),
    AbortSlew,
    Sync(EqCoord),
    SetTracking(bool),
    SetTrackingRate(DriveRate),
    Park,
    Unpark,
    SetPark,
    FindHome,
    /// Direct transport only (D12); needs a `RenewJog` within
    /// `jog_deadman_ms`.
    MoveAxis {
        axis: u8,
        rate_deg_s: f64,
    },
    RenewJog,
    /// Direct transport only (D12); refused while PHD2 guides (D20).
    PulseGuide {
        dir: GuideDirection,
        ms: u32,
    },
    SetGuideRates {
        ra_arcsec_s: f64,
        dec_arcsec_s: f64,
    },
    SetSite(Site),
    SetUtc(DateTime<Utc>),
    SetSideOfPier(PierSide),
    /// The answer is published as `MountSnapshot::destination_side_of_pier`.
    QueryDestinationSideOfPier(EqCoord),
}

impl MountCmd {
    pub fn op_kind(&self) -> OpKind {
        match self {
            MountCmd::Slew { .. } => OpKind::Slew,
            MountCmd::SlewAltAz(_) => OpKind::SlewAltAz,
            MountCmd::AbortSlew => OpKind::AbortSlew,
            MountCmd::Sync(_) => OpKind::Sync,
            MountCmd::SetTracking(_) => OpKind::SetTracking,
            MountCmd::SetTrackingRate(_) => OpKind::SetTrackingRate,
            MountCmd::Park => OpKind::Park,
            MountCmd::Unpark => OpKind::Unpark,
            MountCmd::SetPark => OpKind::SetPark,
            MountCmd::FindHome => OpKind::Home,
            MountCmd::MoveAxis { .. } | MountCmd::RenewJog => OpKind::MoveAxis,
            MountCmd::PulseGuide { .. } => OpKind::PulseGuide,
            MountCmd::SetGuideRates { .. } => OpKind::SetGuideRates,
            MountCmd::SetSite(_) => OpKind::SetSite,
            MountCmd::SetUtc(_) => OpKind::SetUtc,
            MountCmd::SetSideOfPier(_) => OpKind::SetSideOfPier,
            MountCmd::QueryDestinationSideOfPier(_) => OpKind::DestinationSideOfPier,
        }
    }

    /// D12: unsafe at relay latency; published for the direct transport only.
    pub fn relay_allowed(&self) -> bool {
        !matches!(
            self,
            MountCmd::MoveAxis { .. } | MountCmd::RenewJog | MountCmd::PulseGuide { .. }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "cmd", content = "args", rename_all = "snake_case")]
pub enum FocuserCmd {
    /// Logical absolute target, in steps.
    MoveTo(i32),
    /// Relative move, in steps.
    MoveBy(i32),
    Halt,
    SetTempComp(bool),
    /// INDI `FOCUS_SYNC` only (O4 removed MoonLite); `Unsupported` elsewhere.
    Sync(i32),
}

impl FocuserCmd {
    pub fn op_kind(&self) -> OpKind {
        match self {
            FocuserCmd::MoveTo(_) | FocuserCmd::MoveBy(_) => OpKind::FocuserMove,
            FocuserCmd::Halt => OpKind::FocuserHalt,
            FocuserCmd::SetTempComp(_) => OpKind::FocuserTempComp,
            FocuserCmd::Sync(_) => OpKind::FocuserSync,
        }
    }
}

/// The handshake (§3.4 W-c) sits above this.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "cmd", content = "args", rename_all = "snake_case")]
pub enum FilterWheelCmd {
    /// 0-based slot.
    SetPosition(u32),
}

impl FilterWheelCmd {
    pub fn op_kind(&self) -> OpKind {
        match self {
            FilterWheelCmd::SetPosition(_) => OpKind::FilterMove,
        }
    }
}

/// PHD2 settle parameters (`{pixels, time, timeout}`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Settle {
    pub pixels: f64,
    pub time_s: f64,
    pub timeout_s: f64,
}

impl Default for Settle {
    /// The `[guider]` defaults (§2.11).
    fn default() -> Self {
        Self {
            pixels: 1.5,
            time_s: 10.0,
            timeout_s: 40.0,
        }
    }
}

/// Dither amounts always carry an explicit unit (§2.14).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PixelUnit {
    GuidePx,
    MainPx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Axis {
    Ra,
    Dec,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "cmd", content = "args", rename_all = "snake_case")]
pub enum GuiderCmd {
    Start {
        settle: Settle,
        recalibrate: bool,
    },
    Stop,
    Pause {
        full: bool,
    },
    Resume,
    Dither {
        amount: f64,
        unit: PixelUnit,
        ra_only: bool,
        settle: Settle,
    },
    /// The M-1 actuator: behind the gate, default off (G-j).
    SetAlgoParam {
        axis: Axis,
        name: String,
        value: f64,
    },
    Refresh,
}

impl GuiderCmd {
    pub fn op_kind(&self) -> OpKind {
        match self {
            GuiderCmd::Start { .. } => OpKind::GuiderStart,
            GuiderCmd::Stop => OpKind::GuiderStop,
            GuiderCmd::Pause { .. } => OpKind::GuiderPause,
            GuiderCmd::Resume => OpKind::GuiderResume,
            GuiderCmd::Dither { .. } => OpKind::Dither,
            GuiderCmd::SetAlgoParam { .. } => OpKind::SetAlgoParam,
            GuiderCmd::Refresh => OpKind::GuiderRefresh,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "cmd", content = "args", rename_all = "snake_case")]
pub enum RotatorCmd {
    MoveRelative(f64),
    /// Sky position angle.
    MoveAbsolute(f64),
    MoveMechanical(f64),
    Sync(f64),
    Halt,
    SetReverse(bool),
}

impl RotatorCmd {
    pub fn op_kind(&self) -> OpKind {
        match self {
            RotatorCmd::MoveRelative(_)
            | RotatorCmd::MoveAbsolute(_)
            | RotatorCmd::MoveMechanical(_) => OpKind::RotatorMove,
            RotatorCmd::Sync(_) => OpKind::RotatorSync,
            RotatorCmd::Halt => OpKind::RotatorHalt,
            RotatorCmd::SetReverse(_) => OpKind::RotatorReverse,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "cmd", content = "args", rename_all = "snake_case")]
pub enum CoverCalibratorCmd {
    Open,
    Close,
    HaltCover,
    CalibratorOn { brightness: u32 },
    CalibratorOff,
}

impl CoverCalibratorCmd {
    pub fn op_kind(&self) -> OpKind {
        match self {
            CoverCalibratorCmd::Open => OpKind::CoverOpen,
            CoverCalibratorCmd::Close => OpKind::CoverClose,
            CoverCalibratorCmd::HaltCover => OpKind::CoverHalt,
            CoverCalibratorCmd::CalibratorOn { .. } => OpKind::CalibratorOn,
            CoverCalibratorCmd::CalibratorOff => OpKind::CalibratorOff,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "cmd", content = "args", rename_all = "snake_case")]
pub enum SwitchCmd {
    Set { id: u32, value: f64 },
}

impl SwitchCmd {
    pub fn op_kind(&self) -> OpKind {
        match self {
            SwitchCmd::Set { .. } => OpKind::SwitchSet,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "cmd", content = "args", rename_all = "snake_case")]
pub enum DomeCmd {
    OpenShutter,
    CloseShutter,
    Park,
    SlewAzimuth(f64),
    Abort,
    SetSlaved(bool),
}

impl DomeCmd {
    pub fn op_kind(&self) -> OpKind {
        match self {
            DomeCmd::OpenShutter => OpKind::DomeOpenShutter,
            DomeCmd::CloseShutter => OpKind::DomeCloseShutter,
            DomeCmd::Park => OpKind::DomePark,
            DomeCmd::SlewAzimuth(_) => OpKind::DomeSlew,
            DomeCmd::Abort => OpKind::DomeAbort,
            DomeCmd::SetSlaved(_) => OpKind::DomeSlave,
        }
    }
}

/// Weather devices take no commands except an explicit refresh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum WeatherCmd {
    Refresh,
}

/// Safety monitors take no commands except an explicit re-read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum SafetyCmd {
    Refresh,
}

// ---------------------------------------------------------------------------
// Gated commands
// ---------------------------------------------------------------------------

/// An ownership domain (§2.9 step 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Domain {
    Device(DeviceKind),
    /// Commands that may run only between exposures.
    CameraBoundary,
}

/// A filter-handshake target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterTarget {
    Slot(u32),
    Name(String),
}

/// `POST mount/site-sync {direction}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SiteSyncDirection {
    ToMount,
    FromMount,
}

/// What a gated command does.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", content = "args", rename_all = "snake_case")]
pub enum GatedOp {
    Mount(MountCmd),
    Focuser(FocuserCmd),
    FilterWheel(FilterWheelCmd),
    Guider(GuiderCmd),
    Rotator(RotatorCmd),
    CoverCalibrator(CoverCalibratorCmd),
    Switch(SwitchCmd),
    Dome(DomeCmd),
    /// §3.4 W-c.
    FilterHandshake {
        target: FilterTarget,
    },
    /// Copy `[filter_wheel]` TOML seeds into the wheel's profile (D17).
    FilterImportSeed,
    /// Operator-declared origin of an emulated relative focuser; profile
    /// only, no hardware command.
    FocuserSetPosition {
        position: i32,
    },
    /// M-n.
    Center {
        target: EqCoord,
        tolerance_arcmin: Option<f64>,
        max_attempts: Option<u32>,
    },
    /// M-l (Tier 2).
    Flip,
    Autofocus {
        run_id: Option<String>,
    },
    /// A camera capture; refused for automatic sources while a resume block
    /// is active (H22, HTTP 423).
    Capture {
        capture_type: Option<String>,
    },
    SiteSync {
        direction: SiteSyncDirection,
    },
    ClearResumeBlock {
        id: Uuid,
    },
    /// Skips ownership, arbiter and checks; always audited (D19).
    Abort {
        kind: DeviceKind,
    },
    /// Skips ownership, arbiter and checks; always audited (D19).
    EStop {
        abort_exposure: bool,
    },
}

impl GatedOp {
    pub fn kind(&self) -> OpKind {
        match self {
            GatedOp::Mount(cmd) => cmd.op_kind(),
            GatedOp::Focuser(cmd) => cmd.op_kind(),
            GatedOp::FilterWheel(cmd) => cmd.op_kind(),
            GatedOp::Guider(cmd) => cmd.op_kind(),
            GatedOp::Rotator(cmd) => cmd.op_kind(),
            GatedOp::CoverCalibrator(cmd) => cmd.op_kind(),
            GatedOp::Switch(cmd) => cmd.op_kind(),
            GatedOp::Dome(cmd) => cmd.op_kind(),
            GatedOp::FilterHandshake { .. } => OpKind::FilterHandshake,
            GatedOp::FilterImportSeed => OpKind::FilterImportSeed,
            GatedOp::FocuserSetPosition { .. } => OpKind::FocuserSetPosition,
            GatedOp::Center { .. } => OpKind::Center,
            GatedOp::Flip => OpKind::Flip,
            GatedOp::Autofocus { .. } => OpKind::Autofocus,
            GatedOp::Capture { .. } => OpKind::Capture,
            GatedOp::SiteSync { .. } => OpKind::SiteSync,
            GatedOp::ClearResumeBlock { .. } => OpKind::ClearResumeBlock,
            GatedOp::Abort { .. } => OpKind::Abort,
            GatedOp::EStop { .. } => OpKind::EStop,
        }
    }

    /// Abort and e-stop skip ownership, arbiter and safety checks (§2.9).
    pub fn bypasses_gate_checks(&self) -> bool {
        matches!(self, GatedOp::Abort { .. } | GatedOp::EStop { .. })
            || matches!(
                self,
                GatedOp::Mount(MountCmd::AbortSlew)
                    | GatedOp::Focuser(FocuserCmd::Halt)
                    | GatedOp::Rotator(RotatorCmd::Halt)
                    | GatedOp::CoverCalibrator(CoverCalibratorCmd::HaltCover)
                    | GatedOp::Dome(DomeCmd::Abort)
            )
    }

    /// D12: refused by the relay dispatcher.
    pub fn relay_allowed(&self) -> bool {
        match self {
            GatedOp::Mount(cmd) => cmd.relay_allowed(),
            _ => true,
        }
    }

    /// The ownership domains the operation claims (§2.9 step 1 table).
    /// `af_filter_configured` adds the wheel to an autofocus run.
    pub fn default_domains(&self, af_filter_configured: bool) -> Vec<Domain> {
        use DeviceKind::*;
        let device = Domain::Device;
        match self {
            GatedOp::Mount(_) | GatedOp::SiteSync { .. } => vec![device(Mount)],
            GatedOp::Focuser(_) | GatedOp::FocuserSetPosition { .. } => vec![device(Focuser)],
            GatedOp::FilterWheel(_) | GatedOp::FilterImportSeed => vec![device(FilterWheel)],
            GatedOp::Guider(GuiderCmd::Dither { .. } | GuiderCmd::SetAlgoParam { .. }) => {
                vec![device(Guider), Domain::CameraBoundary]
            }
            GatedOp::Guider(_) => vec![device(Guider)],
            GatedOp::Rotator(_) => vec![device(Rotator)],
            GatedOp::CoverCalibrator(_) => vec![device(CoverCalibrator)],
            GatedOp::Switch(_) => vec![device(Switch)],
            GatedOp::Dome(_) => vec![device(Dome)],
            GatedOp::FilterHandshake { .. } => {
                vec![device(FilterWheel), device(Focuser), device(Guider)]
            }
            GatedOp::Flip => vec![device(Mount), device(Guider), Domain::CameraBoundary],
            GatedOp::Center { .. } => vec![device(Mount), device(Camera)],
            GatedOp::Autofocus { .. } => {
                let mut domains = vec![device(Focuser), device(Camera)];
                if af_filter_configured {
                    domains.push(device(FilterWheel));
                }
                domains
            }
            GatedOp::Capture { .. } => vec![device(Camera)],
            GatedOp::ClearResumeBlock { .. } | GatedOp::Abort { .. } | GatedOp::EStop { .. } => {
                Vec::new()
            }
        }
    }
}

/// A command presented to the gate (§2.9).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GatedCommand {
    pub command_id: CommandId,
    /// Owned device kinds, plus the camera-boundary pseudo-domain.
    pub domains: Vec<Domain>,
    pub op: GatedOp,
    pub source: CommandSource,
    pub mode: AutonomyMode,
    pub priority: Priority,
    /// Operation parameters as audited (`control_audit_logs`), including any
    /// policy context such as the policy version.
    pub params: serde_json::Value,
    pub overriding: bool,
}

impl GatedCommand {
    /// A command with a new id, the op's default domains, its parameters as
    /// JSON, and `overriding` taken from the source.
    pub fn new(op: GatedOp, source: CommandSource, mode: AutonomyMode, priority: Priority) -> Self {
        let params = serde_json::to_value(&op).unwrap_or(serde_json::Value::Null);
        Self {
            command_id: CommandId::new(),
            domains: op.default_domains(false),
            overriding: source.overriding(),
            op,
            source,
            mode,
            priority,
            params,
        }
    }

    pub fn with_domains(mut self, domains: Vec<Domain>) -> Self {
        self.domains = domains;
        self
    }

    pub fn with_params(mut self, params: serde_json::Value) -> Self {
        self.params = params;
        self
    }

    pub fn kind(&self) -> OpKind {
        self.op.kind()
    }

    /// Preemption rank of the permit this command would hold (§2.9 step 2).
    pub fn rank(&self) -> u8 {
        self.priority.preemption_rank(&self.source)
    }

    pub fn claims(&self, kind: DeviceKind) -> bool {
        self.domains.contains(&Domain::Device(kind))
    }
}

/// A [`SafetyCheck`](super::traits::SafetyCheck) refusal (HTTP 422).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafetyRejection {
    /// The check's name.
    pub check: String,
    pub reason: String,
    pub details: Option<serde_json::Value>,
}

impl SafetyRejection {
    pub fn new(check: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            check: check.into(),
            reason: reason.into(),
            details: None,
        }
    }
}

impl fmt::Display for SafetyRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.check, self.reason)
    }
}

/// Why the gate refused a command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "error", rename_all = "snake_case")]
pub enum GateError {
    /// HTTP 409 `ownership_conflict`.
    OwnershipConflict {
        domain: Domain,
        holder: Option<CommandId>,
        holder_source: Option<CommandSource>,
    },
    /// HTTP 422.
    SafetyRejected(SafetyRejection),
    /// HTTP 423: a resume block refuses an automatic source.
    ResumeBlocked { blocks: Vec<ResumeBlockKind> },
    /// HTTP 412.
    NotConnected { kind: DeviceKind },
    /// HTTP 409: refused by policy, e.g. `allow_policy_actuation = false`.
    Refused { reason: String },
    /// HTTP 503: the daemon is shutting down (§2.15).
    ShuttingDown,
}

impl GateError {
    pub fn http_status(&self) -> u16 {
        match self {
            GateError::OwnershipConflict { .. } | GateError::Refused { .. } => 409,
            GateError::SafetyRejected(_) => 422,
            GateError::ResumeBlocked { .. } => 423,
            GateError::NotConnected { .. } => 412,
            GateError::ShuttingDown => 503,
        }
    }

    /// API error code (`{error: {code, message}}`).
    pub fn code(&self) -> &'static str {
        match self {
            GateError::OwnershipConflict { .. } => "ownership_conflict",
            GateError::SafetyRejected(_) => "safety_rejected",
            GateError::ResumeBlocked { .. } => "resume_block_active",
            GateError::NotConnected { .. } => "not_connected",
            GateError::Refused { .. } => "refused",
            GateError::ShuttingDown => "shutting_down",
        }
    }

    pub fn to_device_error(&self) -> DeviceError {
        let class = match self {
            GateError::OwnershipConflict { .. } => ErrorClass::OwnershipConflict,
            GateError::SafetyRejected(_) => ErrorClass::SafetyRejected,
            GateError::ResumeBlocked { .. } | GateError::Refused { .. } => {
                ErrorClass::InvalidOperation
            }
            GateError::NotConnected { .. } => ErrorClass::NotConnected,
            GateError::ShuttingDown => ErrorClass::Cancelled,
        };
        DeviceError::new(class, self.to_string())
    }
}

impl fmt::Display for GateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GateError::OwnershipConflict { domain, holder, .. } => match holder {
                Some(holder) => write!(f, "{domain:?} is owned by command {holder}"),
                None => write!(f, "{domain:?} is owned by another command"),
            },
            GateError::SafetyRejected(rejection) => write!(f, "safety check failed: {rejection}"),
            GateError::ResumeBlocked { blocks } => {
                write!(
                    f,
                    "automatic resumption is blocked by {} resume block(s)",
                    blocks.len()
                )
            }
            GateError::NotConnected { kind } => write!(f, "{kind:?} is not connected"),
            GateError::Refused { reason } => write!(f, "refused: {reason}"),
            GateError::ShuttingDown => f.write_str("the daemon is shutting down"),
        }
    }
}

impl std::error::Error for GateError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::types::EqFrame;
    use serde_json::json;

    #[test]
    fn commands_serialise_with_explicit_tags_and_round_trip() {
        let cmds = vec![
            GatedOp::Mount(MountCmd::Slew {
                target: EqCoord::new(5.5, 22.0, EqFrame::J2000),
            }),
            GatedOp::Mount(MountCmd::AbortSlew),
            GatedOp::Mount(MountCmd::SetTracking(true)),
            GatedOp::Mount(MountCmd::SetSideOfPier(PierSide::East)),
            GatedOp::Focuser(FocuserCmd::MoveBy(-40)),
            GatedOp::FilterWheel(FilterWheelCmd::SetPosition(2)),
            GatedOp::Guider(GuiderCmd::Dither {
                amount: 5.0,
                unit: PixelUnit::GuidePx,
                ra_only: false,
                settle: Settle::default(),
            }),
            GatedOp::Rotator(RotatorCmd::Halt),
            GatedOp::CoverCalibrator(CoverCalibratorCmd::CalibratorOn { brightness: 128 }),
            GatedOp::Switch(SwitchCmd::Set { id: 1, value: 0.5 }),
            GatedOp::Dome(DomeCmd::SlewAzimuth(181.0)),
            GatedOp::FilterHandshake {
                target: FilterTarget::Name("Ha".into()),
            },
            GatedOp::EStop {
                abort_exposure: true,
            },
        ];
        for op in &cmds {
            let text = serde_json::to_string(op).unwrap();
            assert_eq!(
                &serde_json::from_str::<GatedOp>(&text).unwrap(),
                op,
                "{text}"
            );
        }
        assert_eq!(
            serde_json::to_value(&cmds[1]).unwrap(),
            json!({"op": "mount", "args": {"cmd": "abort_slew"}})
        );
        assert_eq!(
            serde_json::to_value(&cmds[4]).unwrap(),
            json!({"op": "focuser", "args": {"cmd": "move_by", "args": -40}})
        );
        assert_eq!(
            serde_json::to_value(WeatherCmd::Refresh).unwrap(),
            json!({"cmd": "refresh"})
        );
    }

    #[test]
    fn domains_follow_the_ownership_table() {
        use DeviceKind::*;
        let d = Domain::Device;
        let handshake = GatedOp::FilterHandshake {
            target: FilterTarget::Slot(1),
        };
        assert_eq!(
            handshake.default_domains(false),
            vec![d(FilterWheel), d(Focuser), d(Guider)]
        );
        let dither = GatedOp::Guider(GuiderCmd::Dither {
            amount: 3.0,
            unit: PixelUnit::MainPx,
            ra_only: true,
            settle: Settle::default(),
        });
        assert_eq!(
            dither.default_domains(false),
            vec![d(Guider), Domain::CameraBoundary]
        );
        assert_eq!(
            GatedOp::Flip.default_domains(false),
            vec![d(Mount), d(Guider), Domain::CameraBoundary]
        );
        let center = GatedOp::Center {
            target: EqCoord::new(1.0, 2.0, EqFrame::J2000),
            tolerance_arcmin: None,
            max_attempts: None,
        };
        assert_eq!(center.default_domains(false), vec![d(Mount), d(Camera)]);
        let af = GatedOp::Autofocus { run_id: None };
        assert_eq!(af.default_domains(false), vec![d(Focuser), d(Camera)]);
        assert_eq!(
            af.default_domains(true),
            vec![d(Focuser), d(Camera), d(FilterWheel)]
        );
        assert!(GatedOp::EStop {
            abort_exposure: false
        }
        .default_domains(false)
        .is_empty());
    }

    #[test]
    fn jog_and_pulse_are_direct_only_and_aborts_bypass_checks() {
        assert!(!GatedOp::Mount(MountCmd::MoveAxis {
            axis: 0,
            rate_deg_s: 1.0
        })
        .relay_allowed());
        assert!(!GatedOp::Mount(MountCmd::PulseGuide {
            dir: GuideDirection::North,
            ms: 300
        })
        .relay_allowed());
        assert!(GatedOp::Mount(MountCmd::Park).relay_allowed());
        assert!(GatedOp::Mount(MountCmd::AbortSlew).bypasses_gate_checks());
        assert!(GatedOp::Focuser(FocuserCmd::Halt).bypasses_gate_checks());
        assert!(!GatedOp::Focuser(FocuserCmd::MoveTo(10)).bypasses_gate_checks());
    }

    #[test]
    fn gated_commands_carry_rank_params_and_override() {
        let cmd = GatedCommand::new(
            GatedOp::Mount(MountCmd::Park),
            CommandSource::Operator { overriding: true },
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        assert!(cmd.overriding);
        assert_eq!(cmd.rank(), 5);
        assert!(cmd.claims(DeviceKind::Mount));
        assert_eq!(cmd.kind(), OpKind::Park);
        assert_eq!(cmd.params, json!({"op": "mount", "args": {"cmd": "park"}}));
        let text = serde_json::to_string(&cmd).unwrap();
        assert_eq!(serde_json::from_str::<GatedCommand>(&text).unwrap(), cmd);
    }

    #[test]
    fn gate_errors_map_to_http() {
        let blocked = GateError::ResumeBlocked {
            blocks: vec![ResumeBlockKind::FilterHandshakeIncomplete],
        };
        assert_eq!(blocked.http_status(), 423);
        assert_eq!(blocked.code(), "resume_block_active");
        let rejected =
            GateError::SafetyRejected(SafetyRejection::new("mount_limits", "below 10 deg"));
        assert_eq!(rejected.http_status(), 422);
        assert_eq!(rejected.to_device_error().class, ErrorClass::SafetyRejected);
        let conflict = GateError::OwnershipConflict {
            domain: Domain::Device(DeviceKind::Focuser),
            holder: None,
            holder_source: None,
        };
        assert_eq!(conflict.http_status(), 409);
        assert_eq!(conflict.to_device_error().class.http_status(), 409);
    }
}

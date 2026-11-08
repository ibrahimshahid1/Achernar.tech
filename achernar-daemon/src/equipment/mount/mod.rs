//! Mount operations and checks M-a to M-k (spec §3.2). The meridian flip
//! (M-l) is WP2-6's `flip.rs`; the center loop (M-n) is WP1B-4's
//! `D/mount/center.rs`.
//!
//! Owner: WP1A-4. Independent implementation from the specification; NINA
//! was used only as a behavioural reference (§6).
//!
//! # How the worker drives a mount
//!
//! The generic device worker (WP1A-2) owns a `Box<dyn MountDriver>` and one
//! [`MountRuntime`] per connection, both on the worker thread:
//!
//! 1. After `Device::connect` and the initial poll it calls
//!    [`MountRuntime::on_connected`] (frame policy M-a, site and UTC M-b,
//!    optics claims M-m).
//! 2. For every command it calls [`MountRuntime::plan`] **before** its own
//!    "one operation per device" conflict rule. [`MountPlan::Absorbed`]
//!    (a jog renewal or a second jog axis) and [`MountPlan::CancelActive`]
//!    (`AbortSlew` while an operation runs: cancel it, then start the
//!    returned abort operation) must bypass that rule. Every other command
//!    comes back as [`MountPlan::Start`] even while an operation runs, and
//!    the worker's rule decides: `InvalidOperation`, unless the permit
//!    outranks the active one (§2.5, §2.9; [`CommandMeta::outranks`]), in
//!    which case the active operation's `cancel()` runs first.
//! 3. After every poll it calls [`MountRuntime::after_poll`], which builds
//!    the published [`MountSnapshot`] (derived Alt/Az, LST, HA, J2000,
//!    `can_flip`, `pier_side_static`, clock delta), emits edge events and
//!    runs the limit watchdog (M-k), then it calls the active operation's
//!    `step`.
//! 4. It drains [`MountRuntime::take_events`] into the hub after each poll,
//!    step and command. Events are already stamped with the clock, the
//!    device kind and the connection id; the hub assigns `seq`.
//!
//! Every mount command, instant setters included, is an [`Operation`]: its
//! `start` sends the initiator and its `step` verifies the read-back (C-2,
//! C-5), so the worker needs no mount-specific code.

pub mod flip;
pub mod guide;
pub mod jog;
pub mod limits;
pub mod park;
pub mod site;
pub mod slew;
pub mod sync;
pub mod tracking;

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::config::{EquipmentConfig, EquipmentConfigs, MountConfig, SiteConfig, SiteSyncMode};

use super::commands::MountCmd;
use super::events::{EquipmentEvent, EventKind, ParkMethod};
use super::handle::{MountHandle, Permit};
use super::ops::{ActiveOp, OpKind, Operation};
use super::snapshot::{DestinationSideOfPier, MountCapabilities, MountSnapshot, MountState};
use super::traits::{Clock, FrameConverter, MountDriver, PollPlan};
use super::types::{
    AlignmentMode, CommandId, CommandSource, DeviceError, DeviceIdentity, DeviceKind, DeviceResult,
    EqCoord, EqFrame, HealthState, JNowFlavour, MountSyncMode, ObsState, Observed, OpOutcome,
    PierSide, Priority, Site, Source,
};

use self::jog::JogDemand;
use self::limits::{LimitWatchdog, MountLimits};
use self::sync::PointingOffsets;

/// Sidereal rate: arcseconds of hour angle per SI second.
pub const SIDEREAL_RATE_ARCSEC_S: f64 = 15.041_067_178_7;
/// Sidereal seconds per SI second.
pub const SIDEREAL_PER_SI: f64 = 1.002_737_909_35;
/// Active cadence while an Achernar operation moves the mount (§2.7, 4 Hz).
pub const ACTIVE_POLL: Duration = Duration::from_millis(250);
/// Cadence during an Achernar-issued pulse (§2.7, 10 Hz).
pub const PULSE_POLL: Duration = Duration::from_millis(100);
/// How long a setter waits for its read-back to agree (C-2).
pub const VERIFY_TIMEOUT: Duration = Duration::from_secs(5);
/// Deadline of an instant setter (write plus verification).
pub const SETTER_DEADLINE: Duration = Duration::from_secs(15);

/// The driver type mount operations run on. The alias fixes the trait-object
/// lifetime to `'static`, as in `Operation<dyn MountDriver>`.
pub type DynMount = dyn MountDriver;

/// A boxed mount operation, as the worker drives it.
pub type MountOp = Box<dyn Operation<DynMount>>;

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Duration from seconds, treating negative, NaN and infinite values as zero.
pub(crate) fn secs(value: f64) -> Duration {
    if value.is_finite() && value > 0.0 {
        Duration::from_secs_f64(value)
    } else {
        Duration::ZERO
    }
}

pub(crate) fn duration_ns(duration: Duration) -> u64 {
    super::clock::duration_ns(duration)
}

/// Milliseconds between two monotonic readings.
pub(crate) fn elapsed_ms(from_mono_ns: u64, to_mono_ns: u64) -> u64 {
    to_mono_ns.saturating_sub(from_mono_ns) / 1_000_000
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// Mount behaviour, from `[mount]`, `[site]` and `[equipment]` (§2.11).
/// Rig-specific numbers never live here (D17).
#[derive(Debug, Clone, PartialEq)]
pub struct MountSettings {
    /// After the driver's own SlewSettleTime, which is inside Slewing.
    pub settle: Duration,
    pub slew_timeout: Duration,
    pub arrival_tolerance_arcmin: f64,
    pub sync_mode: MountSyncMode,
    /// Write UTC at connect when the delta exceeds `time_tolerance_s`.
    pub time_sync: bool,
    pub time_tolerance_s: f64,
    pub frame_override: Option<EqFrame>,
    pub jnow_flavour: JNowFlavour,
    pub max_pulse_ms: u32,
    pub guide_dec_reverse_on_pier_west: bool,
    pub jog_deadman: Duration,
    /// `[mount] guide_rate` (a fraction of sidereal) in arcsec/s; used only
    /// when the mount reports no guide rate, with source `Profile`.
    pub guide_rate_fallback_arcsec_s: Option<f64>,
    pub site_sync: SiteSyncMode,
    pub site_tolerance_deg: f64,
    pub elevation_tolerance_m: f64,
    /// `[equipment] lost_after_s`: the limit watchdog announces blindness
    /// once mount data is older than this.
    pub lost_after: Duration,
    pub estop_stops_tracking: bool,
}

impl MountSettings {
    pub fn new(mount: &MountConfig, site: &SiteConfig, equipment: &EquipmentConfig) -> Self {
        let fallback = mount.guide_rate;
        Self {
            settle: secs(mount.settle_s),
            slew_timeout: Duration::from_secs(mount.slew_timeout_s.max(1)),
            arrival_tolerance_arcmin: mount.arrival_tolerance_arcmin.max(0.0),
            sync_mode: mount.sync_mode,
            time_sync: mount.time_sync,
            time_tolerance_s: mount.time_tolerance_s.max(0.0),
            frame_override: mount.equatorial_system_override.frame(),
            jnow_flavour: mount.jnow_flavour,
            max_pulse_ms: mount.max_pulse_ms.max(1),
            guide_dec_reverse_on_pier_west: mount.guide_dec_reverse_on_pier_west,
            jog_deadman: Duration::from_millis(mount.jog_deadman_ms.max(1)),
            guide_rate_fallback_arcsec_s: (fallback.is_finite() && fallback > 0.0)
                .then_some(fallback * SIDEREAL_RATE_ARCSEC_S),
            site_sync: site.sync,
            site_tolerance_deg: site.tolerance_deg.max(0.0),
            elevation_tolerance_m: site.elevation_tolerance_m.max(0.0),
            lost_after: Duration::from_secs(equipment.lost_after_s.max(1)),
            estop_stops_tracking: equipment.estop_stops_tracking,
        }
    }

    pub fn from_configs(configs: &EquipmentConfigs) -> Self {
        Self::new(&configs.mount, &configs.site, &configs.equipment)
    }

    pub fn driver_options(&self) -> MountDriverOptions {
        MountDriverOptions {
            jnow_flavour: self.jnow_flavour,
            frame_override: self.frame_override,
        }
    }
}

impl Default for MountSettings {
    fn default() -> Self {
        Self::new(
            &MountConfig::default(),
            &SiteConfig::default(),
            &EquipmentConfig::default(),
        )
    }
}

/// What every mount driver needs to report its native frame (M-a).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MountDriverOptions {
    pub jnow_flavour: JNowFlavour,
    /// `[mount] equatorial_system_override`.
    pub frame_override: Option<EqFrame>,
}

/// The native frame per M-a: the override wins, else EquatorialSystem
/// (read only from InterfaceVersion 2), else `Unknown` with a reason.
pub fn native_frame_policy(
    equatorial_system: Option<i32>,
    interface_version: Option<i32>,
    options: &MountDriverOptions,
) -> (EqFrame, Option<String>) {
    if let Some(frame) = options.frame_override {
        return (frame, None);
    }
    match equatorial_system {
        Some(code) => {
            let (frame, reason) = EqFrame::from_equatorial_system(code, options.jnow_flavour);
            (frame, reason.map(str::to_string))
        }
        None if interface_version.unwrap_or(1) < 2 => (
            EqFrame::Unknown,
            Some("equatorial_system_needs_interface_v2".into()),
        ),
        None => (
            EqFrame::Unknown,
            Some("equatorial_system_unreadable".into()),
        ),
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// Where an [`EventOutbox`] forwards its events once the spine wires it to
/// the hub.
pub type EventForward = Arc<dyn Fn(EquipmentEvent) + Send + Sync>;

/// Stamps and buffers events raised on the worker thread. The worker drains
/// them into the hub, which assigns `seq`; once a forwarder is installed
/// ([`EventOutbox::set_forward`], the spine's `MountKind`), events go to the
/// hub as they are raised, so an operation's `slew_settled` or `unparked`
/// is published when it happens rather than at the next poll.
#[derive(Clone)]
pub struct EventOutbox {
    events: Arc<Mutex<Vec<EquipmentEvent>>>,
    forward: Arc<Mutex<Option<EventForward>>>,
    clock: Arc<dyn Clock>,
    connection_id: Uuid,
}

impl EventOutbox {
    pub fn new(clock: Arc<dyn Clock>, connection_id: Uuid) -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            forward: Arc::new(Mutex::new(None)),
            clock,
            connection_id,
        }
    }

    /// Forward every event to `forward` from now on (the buffered ones
    /// first, in order); `None` buffers again.
    pub fn set_forward(&self, forward: Option<EventForward>) {
        *lock(&self.forward) = forward.clone();
        if let Some(forward) = forward {
            for event in self.drain() {
                forward(event);
            }
        }
    }

    pub fn emit(&self, kind: EventKind, command_id: Option<CommandId>) {
        let mut event = EquipmentEvent::new(kind, self.clock.as_ref())
            .for_device(DeviceKind::Mount, self.connection_id);
        event.command_id = command_id;
        let forward = lock(&self.forward).clone();
        match forward {
            Some(forward) => forward(event),
            None => lock(&self.events).push(event),
        }
    }

    /// Everything emitted since the last drain, in order.
    pub fn drain(&self) -> Vec<EquipmentEvent> {
        std::mem::take(&mut *lock(&self.events))
    }

    /// Catalogue names of the undrained events (tests and diagnostics).
    pub fn names(&self) -> Vec<&'static str> {
        lock(&self.events)
            .iter()
            .map(EquipmentEvent::name)
            .collect()
    }

    pub fn len(&self) -> usize {
        lock(&self.events).len()
    }

    pub fn is_empty(&self) -> bool {
        lock(&self.events).is_empty()
    }
}

impl fmt::Debug for EventOutbox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EventOutbox")
            .field("connection_id", &self.connection_id)
            .field("pending", &self.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Shared per-connection state
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct SharedState {
    capabilities: MountCapabilities,
    /// `[site]`, or the mount's site under `from_mount` (M-b).
    session_site: Option<Site>,
    /// The site the mount reports, refreshed by every snapshot.
    mount_site: Option<Site>,
    sync_generation: u64,
    offsets: PointingOffsets,
    destination: Option<DestinationSideOfPier>,
    destination_unsupported: bool,
    clock_delta_s: Option<f64>,
    jog: JogDemand,
    announced_tracking: Option<bool>,
    announced_park: Option<bool>,
    /// Tracking-off writes the limit watchdog made (M-k).
    limit_stops: u64,
}

/// Per-connection mount state shared by the runtime and its operations. All
/// users run on the worker thread; the mutex only makes it `Sync`.
pub struct MountShared {
    settings: MountSettings,
    converter: Arc<dyn FrameConverter>,
    clock: Arc<dyn Clock>,
    events: EventOutbox,
    connection_id: Uuid,
    state: Mutex<SharedState>,
}

impl fmt::Debug for MountShared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MountShared")
            .field("connection_id", &self.connection_id)
            .field("settings", &self.settings)
            .field("state", &*lock(&self.state))
            .finish()
    }
}

impl MountShared {
    pub fn new(
        settings: MountSettings,
        converter: Arc<dyn FrameConverter>,
        clock: Arc<dyn Clock>,
        connection_id: Uuid,
    ) -> Arc<Self> {
        Arc::new(Self {
            events: EventOutbox::new(Arc::clone(&clock), connection_id),
            settings,
            converter,
            clock,
            connection_id,
            state: Mutex::new(SharedState::default()),
        })
    }

    fn state(&self) -> MutexGuard<'_, SharedState> {
        lock(&self.state)
    }

    pub fn settings(&self) -> &MountSettings {
        &self.settings
    }

    pub fn clock(&self) -> &dyn Clock {
        self.clock.as_ref()
    }

    pub fn converter(&self) -> &dyn FrameConverter {
        self.converter.as_ref()
    }

    pub fn events(&self) -> &EventOutbox {
        &self.events
    }

    pub fn connection_id(&self) -> Uuid {
        self.connection_id
    }

    pub fn emit(&self, kind: EventKind, command_id: Option<CommandId>) {
        self.events.emit(kind, command_id);
    }

    pub fn capabilities(&self) -> MountCapabilities {
        self.state().capabilities.clone()
    }

    pub fn set_capabilities(&self, capabilities: MountCapabilities) {
        self.state().capabilities = capabilities;
    }

    pub fn session_site(&self) -> Option<Site> {
        self.state().session_site
    }

    pub fn set_session_site(&self, site: Option<Site>) {
        self.state().session_site = site;
    }

    /// The mount's own site, for geometry when no session site is known. An
    /// elevation the mount does not report (NaN) takes the session site's,
    /// else 0 m, so the frame converter never sees NaN.
    pub(crate) fn set_mount_site(&self, site: Option<Site>) {
        let mut state = self.state();
        let fallback = state.session_site;
        state.mount_site = site.map(|s| site::with_known_elevation(s, fallback.as_ref()));
    }

    /// The best site for geometry: the session site, else the mount's own.
    pub fn geometry_site(&self) -> Option<Site> {
        let state = self.state();
        state.session_site.or(state.mount_site)
    }

    pub fn sync_generation(&self) -> u64 {
        self.state().sync_generation
    }

    /// Increment `sync_generation` (M-g) and return the new value.
    pub fn bump_sync_generation(&self) -> u64 {
        let mut state = self.state();
        state.sync_generation += 1;
        state.sync_generation
    }

    pub fn with_offsets<R>(&self, f: impl FnOnce(&mut PointingOffsets) -> R) -> R {
        f(&mut self.state().offsets)
    }

    pub fn destination(&self) -> Option<DestinationSideOfPier> {
        self.state().destination
    }

    pub fn set_destination(&self, answer: DestinationSideOfPier) {
        self.state().destination = Some(answer);
    }

    pub fn destination_unsupported(&self) -> bool {
        let state = self.state();
        state.destination_unsupported
            || state
                .capabilities
                .unsupported_members
                .iter()
                .any(|m| m == "destinationsideofpier")
    }

    pub fn mark_destination_unsupported(&self) {
        self.state().destination_unsupported = true;
    }

    pub fn clock_delta_s(&self) -> Option<f64> {
        self.state().clock_delta_s
    }

    pub fn set_clock_delta_s(&self, delta: Option<f64>) {
        self.state().clock_delta_s = delta;
    }

    pub fn with_jog<R>(&self, f: impl FnOnce(&mut JogDemand) -> R) -> R {
        f(&mut self.state().jog)
    }

    pub fn jog_active(&self) -> bool {
        self.state().jog.active()
    }

    /// Record a tracking state announced by a `tracking_changed` event, so
    /// the tracker does not announce it again.
    pub fn set_announced_tracking(&self, on: bool) {
        self.state().announced_tracking = Some(on);
    }

    pub(crate) fn announced_tracking(&self) -> Option<bool> {
        self.state().announced_tracking
    }

    /// The limit watchdog wrote tracking off (M-k). A tracking-on operation
    /// that is still verifying must not read the poll from before this write
    /// as its success.
    pub(crate) fn note_limit_stop(&self) {
        self.state().limit_stops += 1;
    }

    /// How many tracking-off writes the limit watchdog has made.
    pub(crate) fn limit_stops(&self) -> u64 {
        self.state().limit_stops
    }

    pub fn set_announced_park(&self, parked: bool) {
        self.state().announced_park = Some(parked);
    }

    pub(crate) fn announced_park(&self) -> Option<bool> {
        self.state().announced_park
    }

    /// Convert a commanded coordinate into the native frame (M-a, D10).
    ///
    /// - A target in frame `Unknown` means "the mount's own frame" (the
    ///   legacy `/mount/slew` semantics) and passes unconverted.
    /// - A known frame equal to the native frame passes unconverted.
    /// - Otherwise it is converted through the `FrameConverter` at epoch now,
    ///   unless the native frame is unknown, which refuses the command
    ///   (frame-sensitive features are disabled with a reason).
    pub fn to_native(&self, target: EqCoord) -> DeviceResult<EqCoord> {
        validate_coord(&target)?;
        let caps = self.capabilities();
        let native = caps.native_frame;
        if target.frame == EqFrame::Unknown {
            return Ok(EqCoord {
                frame: native,
                ..target
            }
            .normalized());
        }
        if target.frame == native {
            return Ok(target.normalized());
        }
        if native == EqFrame::Unknown {
            return Err(DeviceError::invalid_operation(format!(
                "the mount's native frame is unknown ({}); a {} target cannot be converted. \
                 Set [mount] equatorial_system_override or send the target in the mount frame",
                caps.frame_reason.as_deref().unwrap_or("no reason reported"),
                target.frame.as_str()
            )));
        }
        let site = self.geometry_site();
        let converted = self
            .converter
            .convert(target, native, self.clock.now_utc(), site.as_ref());
        Ok(EqCoord {
            frame: native,
            ..converted
        }
        .normalized())
    }
}

/// RA must be finite; Dec within [-90, 90].
pub fn validate_coord(target: &EqCoord) -> DeviceResult<()> {
    if !target.ra_h.is_finite() || !target.dec_deg.is_finite() {
        return Err(DeviceError::invalid_value(
            "RA and Dec must be finite numbers",
        ));
    }
    if !(-90.0..=90.0).contains(&target.dec_deg) {
        return Err(DeviceError::invalid_value(format!(
            "declination {} is outside [-90, 90]",
            target.dec_deg
        )));
    }
    Ok(())
}

/// The mount position from a state, when both coordinates are known.
pub fn position(state: &MountState, frame: EqFrame) -> Option<EqCoord> {
    match (state.ra_h.known(), state.dec_deg.known()) {
        (Some(ra), Some(dec)) => Some(EqCoord::new(*ra, *dec, frame)),
        _ => None,
    }
}

/// The position only when both coordinates are fresh.
pub fn fresh_position(state: &MountState, frame: EqFrame) -> Option<EqCoord> {
    match (state.ra_h.fresh_value(), state.dec_deg.fresh_value()) {
        (Some(ra), Some(dec)) => Some(EqCoord::new(*ra, *dec, frame)),
        _ => None,
    }
}

/// True when the mount is known to be parked (fresh or stale).
pub fn is_parked(state: &MountState) -> bool {
    state.at_park.known() == Some(&true)
}

pub(crate) fn parked_error(what: &str) -> DeviceError {
    DeviceError::new(
        super::types::ErrorClass::InvalidWhileParked,
        format!("the mount is parked; unpark it before {what}"),
    )
}

/// Every value marked stale, for a tick whose state could not be read.
pub fn stale_state(state: &MountState) -> MountState {
    MountState {
        ra_h: state.ra_h.clone().into_stale(),
        dec_deg: state.dec_deg.clone().into_stale(),
        alt_deg: state.alt_deg.clone().into_stale(),
        az_deg: state.az_deg.clone().into_stale(),
        sidereal_time_h: state.sidereal_time_h.clone().into_stale(),
        utc: state.utc.clone().into_stale(),
        slewing: state.slewing.clone().into_stale(),
        tracking: state.tracking.clone().into_stale(),
        tracking_rate: state.tracking_rate.clone().into_stale(),
        at_park: state.at_park.clone().into_stale(),
        at_home: state.at_home.clone().into_stale(),
        side_of_pier: state.side_of_pier.clone().into_stale(),
        is_pulse_guiding: state.is_pulse_guiding.clone().into_stale(),
        guide_rate_ra_arcsec_s: state.guide_rate_ra_arcsec_s.clone().into_stale(),
        guide_rate_dec_arcsec_s: state.guide_rate_dec_arcsec_s.clone().into_stale(),
        site: state.site.clone().into_stale(),
        target: state.target.clone().into_stale(),
    }
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

/// Who issued the command being turned into an operation, and the
/// preemption rank of its permit (§2.9 step 2).
#[derive(Debug, Clone, PartialEq)]
pub struct CommandMeta {
    pub command_id: CommandId,
    pub source: CommandSource,
    pub priority: Priority,
    /// The permit's preemption rank (7 e-stop ... 1 recommendation; an
    /// operator command holds at least 5).
    pub rank: u8,
}

impl CommandMeta {
    /// A command with the lowest priority its source allows: `Safety` for
    /// the safety source, `Recommendation` otherwise (an operator still
    /// holds rank 5). Use [`Self::from_permit`] or [`Self::with_priority`]
    /// when the real permit is known.
    pub fn new(command_id: CommandId, source: CommandSource) -> Self {
        let priority = match source {
            CommandSource::Safety => Priority::Safety,
            _ => Priority::Recommendation,
        };
        let rank = priority.preemption_rank(&source);
        Self {
            command_id,
            source,
            priority,
            rank,
        }
    }

    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self.rank = priority.preemption_rank(&self.source);
        self
    }

    pub fn from_permit(permit: &Permit) -> Self {
        Self {
            command_id: permit.command_id(),
            source: permit.source().clone(),
            priority: permit.priority(),
            rank: permit.rank(),
        }
    }

    /// The §2.9 arbiter rule, for a worker that keeps the active
    /// operation's rank: an operator preempts only with `overriding` and
    /// only a holder below safety; anyone else needs a strictly higher rank.
    pub fn outranks(&self, holder_rank: u8) -> bool {
        if self.source.is_operator() {
            self.source.overriding() && holder_rank < Priority::Safety.rank()
        } else {
            self.rank > holder_rank
        }
    }
}

/// What the worker does with a mount command.
pub enum MountPlan {
    /// Run this operation. When another operation is active the worker's
    /// conflict rule decides (§2.5, §2.9): `InvalidOperation` unless this
    /// command's permit outranks the active one's (see
    /// [`CommandMeta::outranks`]), in which case the active operation's
    /// `cancel()` runs first and it resolves `Aborted { by }`.
    Start(MountOp),
    /// Absorbed by the active operation (a jog renewal or rate change);
    /// resolve the command with this outcome at once.
    Absorbed(OpOutcome),
    /// `AbortSlew` while an operation runs: cancel it as an abort would,
    /// bypassing the conflict rule, then start this [`slew::AbortSlewOp`],
    /// which sends AbortSlew plus MoveAxis 0 on jogging axes and confirms
    /// Slewing false. The active operation's own cancel may do no I/O (a
    /// sync, a setter), and a slew or jog by another client may be running.
    CancelActive(MountOp),
    /// Refused before any hardware command.
    Refused(DeviceError),
}

impl fmt::Debug for MountPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MountPlan::Start(op) => write!(f, "Start({:?})", op.kind()),
            MountPlan::Absorbed(outcome) => write!(f, "Absorbed({outcome:?})"),
            MountPlan::CancelActive(op) => write!(f, "CancelActive({:?})", op.kind()),
            MountPlan::Refused(err) => write!(f, "Refused({err})"),
        }
    }
}

/// Build the operation for a command (no operation active).
pub fn build_operation(
    shared: &Arc<MountShared>,
    cmd: MountCmd,
    meta: CommandMeta,
) -> DeviceResult<MountOp> {
    let shared = Arc::clone(shared);
    Ok(match cmd {
        MountCmd::Slew { target } => Box::new(slew::SlewOp::equatorial(shared, meta, target)),
        MountCmd::SlewAltAz(target) => Box::new(slew::SlewOp::altaz(shared, meta, target)),
        MountCmd::AbortSlew => Box::new(slew::AbortSlewOp::new(shared, meta)),
        MountCmd::Sync(target) => Box::new(sync::SyncOp::new(shared, meta, target)),
        MountCmd::SetTracking(on) => Box::new(tracking::SetTrackingOp::new(shared, meta, on)),
        MountCmd::SetTrackingRate(rate) => {
            Box::new(tracking::SetTrackingRateOp::new(shared, meta, rate))
        }
        MountCmd::Park => Box::new(park::ParkOp::park(shared, meta)),
        MountCmd::Unpark => Box::new(park::ParkOp::unpark(shared, meta)),
        MountCmd::SetPark => Box::new(park::SetParkOp::new(shared, meta)),
        MountCmd::FindHome => Box::new(park::HomeOp::new(shared, meta)),
        MountCmd::MoveAxis { axis, rate_deg_s } => {
            Box::new(jog::JogOp::new(shared, meta, axis, rate_deg_s)?)
        }
        MountCmd::RenewJog => {
            return Err(DeviceError::invalid_operation("no jog is active to renew"))
        }
        MountCmd::PulseGuide { dir, ms } => {
            Box::new(guide::PulseGuideOp::new(shared, meta, dir, ms))
        }
        MountCmd::SetGuideRates {
            ra_arcsec_s,
            dec_arcsec_s,
        } => Box::new(guide::SetGuideRatesOp::new(
            shared,
            meta,
            ra_arcsec_s,
            dec_arcsec_s,
        )?),
        MountCmd::SetSite(site) => Box::new(site::SetSiteOp::new(shared, meta, site)?),
        MountCmd::SetUtc(utc) => Box::new(site::SetUtcOp::new(shared, meta, utc)),
        MountCmd::SetSideOfPier(side) => Box::new(slew::SetSideOfPierOp::new(shared, meta, side)?),
        MountCmd::QueryDestinationSideOfPier(target) => {
            Box::new(slew::DestinationQueryOp::new(shared, meta, target))
        }
    })
}

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------

/// What connect-time processing found (M-a, M-b, M-m).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PostConnect {
    pub capabilities: MountCapabilities,
    pub site: site::SiteTimeReport,
}

/// Mount-specific logic for one connection, run on its worker thread.
pub struct MountRuntime {
    shared: Arc<MountShared>,
    tracker: MountTracker,
    watchdog: LimitWatchdog,
    last_snapshot: Option<MountSnapshot>,
    /// The watchdog's tracking-off for the current crossing was announced.
    limit_stop_announced: bool,
}

impl fmt::Debug for MountRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MountRuntime")
            .field("shared", &self.shared)
            .field("tracker", &self.tracker)
            .finish()
    }
}

impl MountRuntime {
    /// `configured_site` is `[site]` (`SiteConfig::site()`), or `None` when
    /// it is unknown.
    pub fn new(
        settings: MountSettings,
        limits: MountLimits,
        converter: Arc<dyn FrameConverter>,
        clock: Arc<dyn Clock>,
        connection_id: Uuid,
        configured_site: Option<Site>,
    ) -> Self {
        let shared = MountShared::new(settings, converter, clock, connection_id);
        shared.set_session_site(configured_site);
        Self {
            shared,
            tracker: MountTracker::default(),
            watchdog: LimitWatchdog::new(limits),
            last_snapshot: None,
            limit_stop_announced: false,
        }
    }

    pub fn shared(&self) -> Arc<MountShared> {
        Arc::clone(&self.shared)
    }

    pub fn settings(&self) -> &MountSettings {
        self.shared.settings()
    }

    /// Device-specific post-connect (§2.6 step 8): cache the capabilities
    /// (frame policy already applied by the driver), then verify or sync the
    /// site and the mount clock (M-b). Connect never moves hardware (D16).
    pub fn on_connected(&mut self, drv: &mut dyn MountDriver) -> DeviceResult<PostConnect> {
        let capabilities = drv.capabilities()?;
        self.shared.set_capabilities(capabilities.clone());
        let site = site::verify_site_and_time(drv, &self.shared, self.shared.session_site());
        Ok(PostConnect { capabilities, site })
    }

    /// Turn a command into what the worker should do. Call it before the
    /// worker's own conflict rule (see the module docs). Only jog renewals,
    /// further jog axes and `AbortSlew` are decided here; any other command
    /// is built as [`MountPlan::Start`] even while an operation is active,
    /// so the worker's preemption rule can let a higher-ranked command (a
    /// safety park, mount recovery, e-stop tracking off) cancel it.
    pub fn plan(&mut self, cmd: MountCmd, meta: CommandMeta, active: Option<OpKind>) -> MountPlan {
        match (cmd, active) {
            (MountCmd::RenewJog, Some(OpKind::MoveAxis)) => {
                let now = self.shared.clock().now_mono_ns();
                self.shared.with_jog(|jog| jog.renew(now));
                MountPlan::Absorbed(OpOutcome::Succeeded)
            }
            (MountCmd::MoveAxis { axis, rate_deg_s }, Some(OpKind::MoveAxis)) => {
                match jog::validated_rate(&self.shared.capabilities(), axis, rate_deg_s) {
                    Ok(rate) => {
                        let now = self.shared.clock().now_mono_ns();
                        self.shared.with_jog(|jog| jog.demand(axis, rate, now));
                        MountPlan::Absorbed(OpOutcome::Succeeded)
                    }
                    Err(err) => MountPlan::Refused(err),
                }
            }
            (MountCmd::AbortSlew, Some(_)) => MountPlan::CancelActive(Box::new(
                slew::AbortSlewOp::new(Arc::clone(&self.shared), meta),
            )),
            (cmd, _) => match build_operation(&self.shared, cmd, meta) {
                Ok(op) => MountPlan::Start(op),
                Err(err) => MountPlan::Refused(err),
            },
        }
    }

    /// Build the published snapshot after a poll, emit edge events and run
    /// the limit watchdog (M-k), which stops tracking on a crossing.
    pub fn after_poll(
        &mut self,
        drv: &mut dyn MountDriver,
        active: Option<ActiveOp>,
    ) -> MountSnapshot {
        let state = match drv.state() {
            Ok(state) => state,
            Err(_) => stale_state(&self.tracker.last_state),
        };
        let snapshot = self.tracker.update(&self.shared, state, active);
        let now_mono = self.shared.clock().now_mono_ns();
        let site = self.shared.geometry_site();
        let dec_of_date = self.tracker.of_date().map(|p| p.dec_deg);
        let verdict =
            self.watchdog
                .evaluate_with_dec(&snapshot, site.as_ref(), dec_of_date, now_mono);
        for event in verdict.events {
            self.shared.emit(event, None);
        }
        if !self.watchdog.tripped() {
            self.limit_stop_announced = false;
        }
        if verdict.stop_tracking {
            // Repeated on every poll while tracking still reads on (the
            // watchdog escalates with outcome_unknown); announced once.
            let stopped = drv.set_tracking(false);
            if stopped.is_ok() {
                self.shared.note_limit_stop();
            }
            match stopped {
                Ok(()) if !self.limit_stop_announced => {
                    self.limit_stop_announced = true;
                    self.shared.emit(
                        EventKind::TrackingChanged {
                            on: Some(false),
                            rate: snapshot.state.tracking_rate.value,
                            source: Some(CommandSource::Safety),
                            reason: Some("limit_reached".into()),
                        },
                        None,
                    );
                    self.shared.set_announced_tracking(false);
                }
                Ok(()) => {}
                Err(err) => tracing::warn!(
                    "mount limit watchdog could not stop tracking (request {}): {err}",
                    verdict.stop_attempt
                ),
            }
        }
        self.last_snapshot = Some(snapshot.clone());
        snapshot
    }

    /// Every event emitted since the last drain.
    pub fn take_events(&self) -> Vec<EquipmentEvent> {
        self.shared.events().drain()
    }

    pub fn last_snapshot(&self) -> Option<&MountSnapshot> {
        self.last_snapshot.as_ref()
    }

    /// §2.6: a slew in progress refuses a disconnect unless `force`, in which
    /// case the caller aborts first (see [`Self::prepare_disconnect`]).
    /// Disconnect never parks.
    pub fn check_disconnect(&self, active: Option<OpKind>, force: bool) -> DeviceResult<()> {
        let slewing = self
            .last_snapshot
            .as_ref()
            .and_then(|s| s.state.slewing.known().copied())
            == Some(true);
        let slew_op = matches!(
            active,
            Some(
                OpKind::Slew
                    | OpKind::SlewAltAz
                    | OpKind::Park
                    | OpKind::Unpark
                    | OpKind::Home
                    | OpKind::SetSideOfPier
            )
        );
        if (slewing || slew_op) && !force {
            return Err(DeviceError::invalid_operation(
                "a slew is in progress; disconnect with force=true aborts it first",
            ));
        }
        Ok(())
    }

    /// Before a forced disconnect: abort any slew and stop any jog.
    pub fn prepare_disconnect(&mut self, drv: &mut dyn MountDriver) -> DeviceResult<()> {
        let result = drv.abort_slew();
        self.stop_jog(drv);
        result
    }

    /// Daemon shutdown (§2.15): stop any jog and abort an Achernar-initiated
    /// slew. Tracking, park state and PHD2 guiding are left as they are; a
    /// park or unpark in progress is allowed to finish. Never parks.
    pub fn shutdown(
        &mut self,
        drv: &mut dyn MountDriver,
        active: Option<OpKind>,
    ) -> DeviceResult<()> {
        self.stop_jog(drv);
        match active {
            Some(OpKind::Slew | OpKind::SlewAltAz | OpKind::Home | OpKind::SetSideOfPier) => {
                drv.abort_slew()
            }
            _ => Ok(()),
        }
    }

    /// MoveAxis 0 on every axis the jog demand says may be moving (a
    /// running jog, or a stop that failed earlier) and, while a jog
    /// operation runs, on every CanMoveAxis axis. Failures are logged and
    /// stay recorded for the next stop.
    fn stop_jog(&mut self, drv: &mut dyn MountDriver) {
        let caps = self.shared.capabilities();
        let jogging = self.shared.jog_active();
        let demanded = self.shared.with_jog(|jog| jog.moving_axes());
        let axes: Vec<u8> = (0..jog::AXES)
            .filter(|axis| {
                demanded.contains(axis) || (jogging && caps.can_move_axis[usize::from(*axis)])
            })
            .collect();
        let _ = jog::stop_axes(&self.shared, drv, &axes, None);
        self.shared.with_jog(|jog| jog.end());
    }
}

/// The e-stop response budget (§2.9 step 4).
pub const ESTOP_BUDGET: Duration = Duration::from_secs(2);

/// What [`estop_mount`] did, step by step (§2.9 step 1).
#[derive(Debug, Clone, PartialEq)]
pub struct MountEstopReport {
    /// The out-of-band abort path: AbortSlew, plus MoveAxis 0 on the axes
    /// the driver knows are jogging.
    pub abort: DeviceResult<()>,
    /// The e-stop-rank commands queued with the abort, each with whether
    /// the worker accepted it (its start, which sends the hardware
    /// commands): tracking off, which first sends MoveAxis 0 on every
    /// CanMoveAxis axis, when `stops_tracking`; otherwise MoveAxis 0 on
    /// each CanMoveAxis axis.
    pub stops: Vec<(MountCmd, DeviceResult<()>)>,
}

impl MountEstopReport {
    /// `Ok` only when every step was accepted; otherwise the first failure
    /// (the abort path, then the queued stops in order).
    pub fn result(&self) -> DeviceResult<()> {
        self.abort.clone()?;
        for (_, result) in &self.stops {
            result.clone()?;
        }
        Ok(())
    }

    /// The tracking-off step's result; `None` when tracking was left on.
    pub fn tracking_off(&self) -> Option<&DeviceResult<()>> {
        self.stops
            .iter()
            .find(|(cmd, _)| *cmd == MountCmd::SetTracking(false))
            .map(|(_, result)| result)
    }
}

async fn within_estop_budget<T>(
    step: &str,
    future: impl std::future::Future<Output = DeviceResult<T>>,
) -> DeviceResult<()> {
    match tokio::time::timeout(ESTOP_BUDGET, future).await {
        Ok(result) => result.map(|_| ()),
        Err(_) => Err(DeviceError::outcome_unknown(format!(
            "{step} was not confirmed within the {} s e-stop budget",
            ESTOP_BUDGET.as_secs()
        ))),
    }
}

/// E-stop for a mount (§2.9 step 1), for `estop.rs` (WP1A-2): the handle's
/// abort path (AbortSlew, plus MoveAxis 0 on jogging axes), then MoveAxis
/// 0 on every CanMoveAxis axis (also axes another client moved) and, when
/// `stops_tracking`, tracking off.
///
/// The stops are queued from the safety source at e-stop rank together
/// with the abort, not after it, so a slow AbortSlew never delays them, and
/// the worker's preemption rule lets them cancel whatever operation is
/// still stopping (see [`MountRuntime::plan`]). Only one command ever
/// contends with the active operation: with `stops_tracking` it is the
/// tracking off, which at e-stop rank sends MoveAxis 0 on every axis before
/// Tracking false (see [`tracking::SetTrackingOp`]); without, the first
/// MoveAxis 0 starts a one-shot stop and absorbs the others, which still
/// reach the hardware. Each step is bounded by [`ESTOP_BUDGET`] and
/// reported in the [`MountEstopReport`], so a refused or unconfirmed step
/// is never reported as success; a step that does not answer in time
/// reports `outcome_unknown`, though its command still runs. Confirmation
/// that motion stopped arrives as events.
pub async fn estop_mount(handle: &MountHandle, stops_tracking: bool) -> MountEstopReport {
    estop_mount_reporting(handle, stops_tracking, None).await
}

/// [`estop_mount`], also sending the abort path's result on `abort_done` as
/// soon as it arrives, so the e-stop (`estop.rs`) can report it within its
/// own budget while the stops are still being confirmed.
pub async fn estop_mount_reporting(
    handle: &MountHandle,
    stops_tracking: bool,
    abort_done: Option<tokio::sync::oneshot::Sender<DeviceResult<()>>>,
) -> MountEstopReport {
    let ticket = handle.abort();
    let caps = handle.snapshot().capabilities.clone();
    let cmds: Vec<MountCmd> = if stops_tracking {
        vec![MountCmd::SetTracking(false)]
    } else {
        (0..jog::AXES)
            .filter(|axis| caps.can_move_axis[usize::from(*axis)])
            .map(|axis| MountCmd::MoveAxis {
                axis,
                rate_deg_s: 0.0,
            })
            .collect()
    };
    let abort = async move {
        let result = within_estop_budget("the abort path", ticket.outcome()).await;
        if let Some(tx) = abort_done {
            let _ = tx.send(result.clone());
        }
        result
    };
    // `join_all` polls in order, so the commands are queued in that order.
    let stops = futures::future::join_all(cmds.into_iter().map(|cmd| {
        let permit = Permit::unmanaged(CommandId::new(), Priority::EStop, CommandSource::Safety);
        let sent = handle.send(cmd.clone(), permit);
        async move {
            let result = within_estop_budget("the e-stop stop", sent).await;
            if let Err(err) = &result {
                tracing::warn!("e-stop step {cmd:?} on the mount failed: {err}");
            }
            (cmd, result)
        }
    }));
    let (abort, stops) = tokio::join!(abort, stops);
    MountEstopReport { abort, stops }
}

// ---------------------------------------------------------------------------
// Snapshot builder and edge events
// ---------------------------------------------------------------------------

/// A slew on the hand controller or another client, seen only through
/// Slewing (no Achernar operation active).
#[derive(Debug, Clone, Copy, PartialEq)]
enum ExternalSlew {
    Moving,
    Settling { until_mono_ns: u64 },
}

/// Minimum |HA| for a slew end to count as evidence about the pier side.
const STATIC_EVIDENCE_HA_H: f64 = 1.0;

/// Builds [`MountSnapshot`]s and emits the edge events (M-j, observed
/// tracking and park changes, slews made by other clients).
#[derive(Debug, Default)]
pub struct MountTracker {
    initialised: bool,
    last_side: Option<PierSide>,
    side_changed: bool,
    /// (HA >= 0, side) at slew ends with |HA| >= 1 h.
    rest_sides: Vec<(bool, PierSide)>,
    pier_static: bool,
    prev_slewing: Option<bool>,
    external: Option<ExternalSlew>,
    last_state: MountState,
    /// The last position in the frame of date (see [`of_date_position`]).
    of_date: Option<EqCoord>,
}

fn motion_op(kind: OpKind) -> bool {
    matches!(
        kind,
        OpKind::Slew
            | OpKind::SlewAltAz
            | OpKind::Park
            | OpKind::Unpark
            | OpKind::Home
            | OpKind::SetSideOfPier
            | OpKind::MoveAxis
            | OpKind::AbortSlew
    )
}

impl MountTracker {
    pub fn pier_side_static(&self) -> bool {
        self.pier_static
    }

    /// The last built snapshot's position in the frame of date.
    pub fn of_date(&self) -> Option<EqCoord> {
        self.of_date
    }

    /// Build the snapshot for `state` and emit edge events.
    pub fn update(
        &mut self,
        shared: &MountShared,
        state: MountState,
        active: Option<ActiveOp>,
    ) -> MountSnapshot {
        let clock = shared.clock();
        let now_utc = clock.now_utc();
        let now_mono = clock.now_mono_ns();
        let caps = shared.capabilities();
        let settings = shared.settings();
        let mut state = state;

        shared.set_mount_site(state.site.known().copied());
        let site = shared.geometry_site();
        let native = caps.native_frame;
        let pos = position(&state, native);
        let of_date = pos.map(|p| of_date_position(shared, p, site.as_ref(), now_utc));
        derive_geometry(&mut state, of_date, site.as_ref(), now_utc, now_mono);
        self.of_date = of_date;
        if let Some(rate) = settings.guide_rate_fallback_arcsec_s {
            for value in [
                &mut state.guide_rate_ra_arcsec_s,
                &mut state.guide_rate_dec_arcsec_s,
            ] {
                if value.value.is_none() {
                    *value = Observed::fresh(rate, Source::Profile, now_utc, now_mono);
                }
            }
        }

        let j2000 = pos.and_then(|p| match native {
            EqFrame::J2000 => Some(p),
            EqFrame::Unknown => None,
            _ => Some(
                shared
                    .converter()
                    .convert(p, EqFrame::J2000, now_utc, site.as_ref())
                    .normalized(),
            ),
        });
        let ha_h = state
            .sidereal_time_h
            .known()
            .copied()
            .zip(of_date)
            .map(|(lst, p)| limits::hour_angle_h(lst, p.ra_h));
        let clock_delta_s = clock_delta(&state).or_else(|| shared.clock_delta_s());

        let active_kind = active.as_ref().map(|op| op.kind);
        self.edges(shared, &state, active_kind, ha_h, pos, now_mono);

        let side_known = matches!(
            state.side_of_pier.known(),
            Some(PierSide::East | PierSide::West)
        );
        let can_flip = caps.alignment_mode == Some(AlignmentMode::GermanPolar)
            && side_known
            && !self.pier_static;

        let snapshot = MountSnapshot {
            connection_id: shared.connection_id(),
            capabilities: caps,
            j2000,
            ha_h,
            can_flip,
            pier_side_static: self.pier_static,
            clock_delta_s,
            sync_generation: shared.sync_generation(),
            jog_active: shared.jog_active(),
            destination_side_of_pier: shared.destination(),
            active_op: active,
            published_at: now_utc,
            published_mono_ns: now_mono,
            state: state.clone(),
        };
        self.last_state = state;
        self.initialised = true;
        snapshot
    }

    fn edges(
        &mut self,
        shared: &MountShared,
        state: &MountState,
        active: Option<OpKind>,
        ha_h: Option<f64>,
        pos: Option<EqCoord>,
        now_mono: u64,
    ) {
        // Pier side (M-j).
        if let Some(side @ (PierSide::East | PierSide::West)) =
            state.side_of_pier.fresh_value().copied()
        {
            if let Some(last) = self.last_side {
                if last != side {
                    shared.emit(
                        EventKind::PierSideChanged {
                            from: last,
                            to: side,
                        },
                        None,
                    );
                    self.side_changed = true;
                    self.pier_static = false;
                }
            }
            self.last_side = Some(side);
        }

        // Slewing edges: evidence for `pier_side_static`, and slews made by
        // other clients (an Achernar operation emits its own slew events).
        let slewing = state.slewing.fresh_value().copied();
        let external_allowed = !active.is_some_and(motion_op);
        if let Some(now) = slewing {
            if self.prev_slewing == Some(true) && !now {
                if let (Some(ha), Some(side @ (PierSide::East | PierSide::West))) =
                    (ha_h, state.side_of_pier.known().copied())
                {
                    if ha.abs() >= STATIC_EVIDENCE_HA_H {
                        self.rest_sides.push((ha >= 0.0, side));
                    }
                }
                if external_allowed && self.external == Some(ExternalSlew::Moving) {
                    shared.emit(EventKind::SlewArrived { position: pos }, None);
                    self.external = Some(ExternalSlew::Settling {
                        until_mono_ns: now_mono + duration_ns(shared.settings().settle),
                    });
                } else {
                    self.external = None;
                }
            } else if now && self.prev_slewing != Some(true) {
                if self.initialised && external_allowed {
                    shared.emit(EventKind::SlewStarted, None);
                    self.external = Some(ExternalSlew::Moving);
                } else {
                    self.external = None;
                }
            }
            self.prev_slewing = Some(now);
        }
        if !external_allowed {
            self.external = None;
        }
        if let Some(ExternalSlew::Settling { until_mono_ns }) = self.external {
            if now_mono >= until_mono_ns {
                if !is_parked(state) {
                    shared.emit(
                        EventKind::SlewSettled {
                            position: pos,
                            error_arcmin: None,
                        },
                        None,
                    );
                }
                self.external = None;
            }
        }

        // `pier_side_static`: slews to both sides of the meridian ended on
        // the same side, on a German equatorial mount.
        let caps_gem = shared.capabilities().alignment_mode == Some(AlignmentMode::GermanPolar);
        if caps_gem && !self.side_changed && !self.pier_static {
            let west_targets = self.rest_sides.iter().any(|(positive, _)| *positive);
            let east_targets = self.rest_sides.iter().any(|(positive, _)| !*positive);
            let first = self.rest_sides.first().map(|(_, side)| *side);
            let one_side = self.rest_sides.iter().all(|(_, side)| Some(*side) == first);
            if west_targets && east_targets && one_side {
                if let Some(side) = first {
                    self.pier_static = true;
                    shared.emit(EventKind::PierSideStatic { side }, None);
                }
            }
        }

        // Tracking observed changing outside an operation that announces it.
        if let Some(on) = state.tracking.fresh_value().copied() {
            let announces_itself = matches!(
                active,
                Some(OpKind::SetTracking | OpKind::Slew | OpKind::SlewAltAz)
            );
            if shared.announced_tracking() != Some(on) {
                if self.initialised && !announces_itself {
                    shared.emit(
                        EventKind::TrackingChanged {
                            on: Some(on),
                            rate: state.tracking_rate.value,
                            source: None,
                            reason: Some("observed".into()),
                        },
                        None,
                    );
                }
                shared.set_announced_tracking(on);
            }
        }

        // AtPark changing outside a park or unpark operation.
        if let Some(parked) = state.at_park.fresh_value().copied() {
            let announces_itself = matches!(active, Some(OpKind::Park | OpKind::Unpark));
            if shared.announced_park() != Some(parked) {
                if self.initialised && !announces_itself {
                    let kind = if parked {
                        EventKind::Parked {
                            method: ParkMethod::Driver,
                        }
                    } else {
                        EventKind::Unparked
                    };
                    shared.emit(kind, None);
                }
                shared.set_announced_park(parked);
            }
        }
    }
}

fn derived<T>(value: T, inputs_fresh: bool, now_utc: DateTime<Utc>, now_mono: u64) -> Observed<T> {
    let mut obs = Observed::fresh(value, Source::Derived, now_utc, now_mono);
    if !inputs_fresh {
        obs.state = ObsState::Stale;
    }
    obs
}

/// Fill LST, Alt and Az from the site when the mount reports none (R2 for
/// mounts and adapters without them). Reported values are never replaced.
/// `of_date` is the mount position in the frame of date
/// ([`of_date_position`]), so a J2000 mount's derived Alt/Az carry no
/// precession error.
fn derive_geometry(
    state: &mut MountState,
    of_date: Option<EqCoord>,
    site: Option<&Site>,
    now_utc: DateTime<Utc>,
    now_mono: u64,
) {
    let Some(site) = site else {
        return;
    };
    if state.sidereal_time_h.value.is_none() {
        state.sidereal_time_h = Observed::fresh(
            limits::lst_hours(now_utc, site.lon_deg),
            Source::Derived,
            now_utc,
            now_mono,
        );
    }
    if state.alt_deg.value.is_some() && state.az_deg.value.is_some() {
        return;
    }
    let (Some(pos), Some(lst)) = (of_date, state.sidereal_time_h.known().copied()) else {
        return;
    };
    let fresh = state.ra_h.is_fresh() && state.dec_deg.is_fresh();
    let altaz = limits::alt_az(
        limits::hour_angle_h(lst, pos.ra_h),
        pos.dec_deg,
        site.lat_deg,
    );
    if state.alt_deg.value.is_none() {
        state.alt_deg = derived(altaz.alt_deg, fresh, now_utc, now_mono);
    }
    if state.az_deg.value.is_none() {
        state.az_deg = derived(altaz.az_deg, fresh, now_utc, now_mono);
    }
}

/// The native position in the frame of date, for hour angle and Alt/Az: a
/// J2000 position is converted through the [`FrameConverter`]; JNow
/// positions, and `Unknown` (taken as the mount's own frame of date, as the
/// legacy API does), pass unchanged.
fn of_date_position(
    shared: &MountShared,
    pos: EqCoord,
    site: Option<&Site>,
    now_utc: DateTime<Utc>,
) -> EqCoord {
    match pos.frame {
        EqFrame::J2000 => shared
            .converter()
            .convert(pos, EqFrame::JNowApparent, now_utc, site)
            .normalized(),
        _ => pos,
    }
}

/// Mount UTC minus host UTC, from a fresh `utcdate` read (M-b).
pub fn clock_delta(state: &MountState) -> Option<f64> {
    let mount = state.utc.fresh_value()?;
    let host = state.utc.observed_at?;
    Some((*mount - host).num_milliseconds() as f64 / 1000.0)
}

// ---------------------------------------------------------------------------
// Status projection (§2.13)
// ---------------------------------------------------------------------------

/// The §2.13 `mount` block for `/status` and `TELEMETRY_UPDATE`: today's
/// keys (`name`, `connected`, `tracking`, `ra`, `dec` mount-native, `parked`,
/// `pier_side` as `"east"`/`"west"`/null) plus every new nullable field.
pub fn status_block(
    snapshot: &MountSnapshot,
    identity: &DeviceIdentity,
    health: &HealthState,
    session_site: Option<&Site>,
    now_mono_ns: u64,
) -> Value {
    let s = &snapshot.state;
    let caps = &snapshot.capabilities;
    let site = session_site.copied().or(s.site.value);
    let observed = [&s.ra_h, &s.dec_deg]
        .into_iter()
        .filter_map(|o| o.observed_at.zip(o.mono_ns))
        .max_by_key(|(_, mono)| *mono);
    json!({
        "name": identity.name,
        "connected": health.state.is_connected(),
        "tracking": s.tracking.value,
        "ra": s.ra_h.value,
        "dec": s.dec_deg.value,
        "parked": s.at_park.value,
        "pier_side": s.side_of_pier.value.and_then(PierSide::as_str),
        "slewing": s.slewing.value,
        "at_home": s.at_home.value,
        "alt": s.alt_deg.value,
        "az": s.az_deg.value,
        "lst_h": s.sidereal_time_h.value,
        "ha_h": snapshot.ha_h,
        "frame": caps.native_frame.as_str(),
        "ra_j2000": snapshot.j2000.map(|c| c.ra_h),
        "dec_j2000": snapshot.j2000.map(|c| c.dec_deg),
        "tracking_rate": s.tracking_rate.value,
        "is_pulse_guiding": s.is_pulse_guiding.value,
        "guide_rate_ra_arcsec_s": s.guide_rate_ra_arcsec_s.value,
        "guide_rate_dec_arcsec_s": s.guide_rate_dec_arcsec_s.value,
        "alignment_mode": caps.alignment_mode,
        "can_flip": snapshot.can_flip,
        "site": site,
        "clock_delta_s": snapshot.clock_delta_s,
        "observed_at": observed.map(|(at, _)| at),
        "age_ms": observed.map(|(_, mono)| now_mono_ns.saturating_sub(mono) / 1_000_000),
        "state": health.state,
    })
}

/// The §2.13 keys, in order. Tests assert the status block carries each.
pub const STATUS_KEYS: &[&str] = &[
    "name",
    "connected",
    "tracking",
    "ra",
    "dec",
    "parked",
    "pier_side",
    "slewing",
    "at_home",
    "alt",
    "az",
    "lst_h",
    "ha_h",
    "frame",
    "ra_j2000",
    "dec_j2000",
    "tracking_rate",
    "is_pulse_guiding",
    "guide_rate_ra_arcsec_s",
    "guide_rate_dec_arcsec_s",
    "alignment_mode",
    "can_flip",
    "site",
    "clock_delta_s",
    "observed_at",
    "age_ms",
    "state",
];

/// A poll plan for tests and for connect-time reads outside the worker.
pub fn plan(tick: u64, active: bool, extra_members: &[&str]) -> PollPlan {
    PollPlan {
        tick,
        interval: if active {
            ACTIVE_POLL
        } else {
            Duration::from_secs(1)
        },
        active,
        prefer_device_state: true,
        extra_members: extra_members.iter().map(|m| (*m).to_string()).collect(),
    }
}

// ---------------------------------------------------------------------------
// Test support: a miniature worker loop
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::equipment::ops::{OpPhase, OpProgress};
    use crate::equipment::sim::mount::{SimMount, SimMountProfile};
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::traits::{Device, IdentityConverter};

    pub struct Rig {
        pub clock: Arc<ManualClock>,
        pub sim: SimMount,
        pub runtime: MountRuntime,
    }

    pub fn site() -> Site {
        Site {
            lat_deg: 40.0,
            lon_deg: -75.0,
            elev_m: 100.0,
            source: crate::equipment::types::SiteSource::Config,
        }
    }

    pub fn operator() -> CommandMeta {
        CommandMeta::new(
            CommandId::new(),
            CommandSource::Operator { overriding: false },
        )
    }

    pub fn rig(profile: SimMountProfile) -> Rig {
        rig_with(profile, MountSettings::default(), Some(site()))
    }

    pub fn rig_with(
        profile: SimMountProfile,
        settings: MountSettings,
        configured: Option<Site>,
    ) -> Rig {
        let clock = Arc::new(ManualClock::new());
        let mut sim = SimMount::new(profile, clock.clone());
        sim.connect(Duration::from_secs(5)).expect("sim connect");
        let identity = sim.identity().clone();
        let mut runtime = MountRuntime::new(
            settings,
            MountLimits::default(),
            Arc::new(IdentityConverter),
            clock.clone(),
            identity.connection_id,
            configured,
        );
        sim.poll(&plan(0, false, &[])).expect("initial poll");
        runtime.on_connected(&mut sim).expect("post connect");
        runtime.after_poll(&mut sim, None);
        Rig {
            clock,
            sim,
            runtime,
        }
    }

    impl Rig {
        pub fn tick(&mut self, by: Duration, active: Option<ActiveOp>) -> MountSnapshot {
            self.clock.advance(by);
            let _ = self.sim.poll(&plan(1, active.is_some(), &[]));
            self.runtime.after_poll(&mut self.sim, active)
        }

        pub fn names(&self) -> Vec<&'static str> {
            self.runtime.shared().events().names()
        }

        /// Plan and drive one command to completion, like the worker.
        pub fn run(&mut self, cmd: MountCmd, meta: CommandMeta) -> DeviceResult<OpOutcome> {
            let plan = self.runtime.plan(cmd, meta.clone(), None);
            let op = match plan {
                MountPlan::Start(op) => op,
                MountPlan::Refused(err) => return Err(err),
                other => panic!("unexpected plan {other:?}"),
            };
            Ok(drive(self, op, &meta, 100_000))
        }

        /// Plan `cmd` while `active` (started with `active_meta`) runs, and
        /// apply the WP1A-2 worker's rules: an absorbed command resolves at
        /// once; `CancelActive` cancels and then drives the abort; `Start`
        /// is refused unless it outranks the active permit, in which case
        /// the active operation is cancelled first. Returns the command's
        /// outcome and whether the active operation was cancelled.
        pub fn run_during(
            &mut self,
            cmd: MountCmd,
            meta: CommandMeta,
            active: &mut MountOp,
            active_meta: &CommandMeta,
        ) -> (DeviceResult<OpOutcome>, bool) {
            match self.runtime.plan(cmd, meta.clone(), Some(active.kind())) {
                MountPlan::Absorbed(outcome) => (Ok(outcome), false),
                MountPlan::Refused(err) => (Err(err), false),
                MountPlan::CancelActive(op) => {
                    let _ = active.cancel(&mut self.sim);
                    (Ok(drive(self, op, &meta, 100_000)), true)
                }
                MountPlan::Start(op) => {
                    if !meta.outranks(active_meta.rank) {
                        return (
                            Err(DeviceError::invalid_operation(
                                "another operation is active on this device",
                            )),
                            false,
                        );
                    }
                    let _ = active.cancel(&mut self.sim);
                    (Ok(drive(self, op, &meta, 100_000)), true)
                }
            }
        }
    }

    /// J2000 to any other frame adds `ra_h` and `dec_deg`; the reverse
    /// subtracts them. A stand-in for precession in frame tests.
    pub struct ShiftConverter {
        pub ra_h: f64,
        pub dec_deg: f64,
    }

    impl FrameConverter for ShiftConverter {
        fn convert(
            &self,
            c: EqCoord,
            to: EqFrame,
            _at: DateTime<Utc>,
            _site: Option<&Site>,
        ) -> EqCoord {
            let sign = match (c.frame, to) {
                (from, to) if from == to => 0.0,
                (EqFrame::J2000, _) => 1.0,
                (_, EqFrame::J2000) => -1.0,
                _ => 0.0,
            };
            EqCoord::new(
                EqCoord::wrap_ra_h(c.ra_h + sign * self.ra_h),
                c.dec_deg + sign * self.dec_deg,
                to,
            )
        }
    }

    pub fn active_op(kind: OpKind, meta: &CommandMeta, clock: &ManualClock) -> ActiveOp {
        ActiveOp {
            command_id: meta.command_id,
            kind,
            phase: OpPhase::InProgress,
            started_at: clock.now_utc(),
            started_mono_ns: clock.now_mono_ns(),
            deadline_mono_ns: u64::MAX,
        }
    }

    /// Start `op`, then poll and step until it finishes or its deadline
    /// passes (the worker then cancels it and resolves `TimedOut`).
    pub fn drive(
        rig: &mut Rig,
        mut op: MountOp,
        meta: &CommandMeta,
        max_steps: usize,
    ) -> OpOutcome {
        let started = rig.clock.now_mono_ns();
        let deadline = started.saturating_add(duration_ns(op.deadline()));
        if let Err(err) = op.start(&mut rig.sim, rig.clock.as_ref()) {
            return OpOutcome::Failed(err);
        }
        let active = active_op(op.kind(), meta, &rig.clock);
        let mut wait = ACTIVE_POLL;
        for _ in 0..max_steps {
            rig.tick(wait, Some(active.clone()));
            if rig.clock.now_mono_ns() >= deadline {
                let _ = op.cancel(&mut rig.sim);
                return OpOutcome::TimedOut;
            }
            match op.step(&mut rig.sim, rig.clock.as_ref()) {
                OpProgress::Done(outcome) => return outcome,
                OpProgress::Continue { next_poll } => {
                    wait = next_poll.max(Duration::from_millis(1))
                }
                OpProgress::Settling { until_mono_ns } => {
                    let now = rig.clock.now_mono_ns();
                    wait = Duration::from_nanos(until_mono_ns.saturating_sub(now))
                        .clamp(Duration::from_millis(1), ACTIVE_POLL);
                }
            }
        }
        panic!(
            "operation {:?} did not finish in {max_steps} steps",
            op.kind()
        );
    }

    /// Start `op` and step it `n` times without finishing it.
    pub fn advance(
        rig: &mut Rig,
        op: &mut MountOp,
        meta: &CommandMeta,
        n: usize,
    ) -> Option<OpOutcome> {
        let active = active_op(op.kind(), meta, &rig.clock);
        for _ in 0..n {
            rig.tick(ACTIVE_POLL, Some(active.clone()));
            if let OpProgress::Done(outcome) = op.step(&mut rig.sim, rig.clock.as_ref()) {
                return Some(outcome);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use crate::equipment::sim::mount::SimMountProfile;
    use crate::equipment::traits::Device;
    use crate::equipment::types::{DriverId, LinkState};

    #[test]
    fn settings_follow_the_config_defaults() {
        let settings = MountSettings::default();
        assert_eq!(settings.settle, Duration::from_secs(5));
        assert_eq!(settings.slew_timeout, Duration::from_secs(600));
        assert_eq!(settings.max_pulse_ms, 5000);
        assert_eq!(settings.jog_deadman, Duration::from_millis(1000));
        assert!(
            !settings.time_sync,
            "connect writes nothing by default (D16)"
        );
        assert_eq!(settings.site_sync, SiteSyncMode::VerifyOnly);
        assert_eq!(settings.frame_override, None);
        let fallback = settings.guide_rate_fallback_arcsec_s.unwrap();
        assert!((fallback - 0.5 * SIDEREAL_RATE_ARCSEC_S).abs() < 1e-9);
        let mut mount = MountConfig {
            guide_rate: 0.0,
            settle_s: f64::NAN,
            ..MountConfig::default()
        };
        mount.equatorial_system_override = crate::config::EquatorialSystemOverride::J2000;
        let settings =
            MountSettings::new(&mount, &SiteConfig::default(), &EquipmentConfig::default());
        assert_eq!(settings.guide_rate_fallback_arcsec_s, None);
        assert_eq!(settings.settle, Duration::ZERO);
        assert_eq!(settings.frame_override, Some(EqFrame::J2000));
    }

    #[test]
    fn frame_policy_covers_m_a_cases_and_the_override() {
        let options = MountDriverOptions::default();
        assert_eq!(
            native_frame_policy(Some(1), Some(4), &options),
            (EqFrame::JNowApparent, None)
        );
        let mean = MountDriverOptions {
            jnow_flavour: JNowFlavour::Mean,
            frame_override: None,
        };
        assert_eq!(
            native_frame_policy(Some(1), Some(4), &mean).0,
            EqFrame::JNowMean
        );
        assert_eq!(
            native_frame_policy(Some(2), Some(3), &options).0,
            EqFrame::J2000
        );
        for code in [3, 4] {
            assert_eq!(
                native_frame_policy(Some(code), Some(4), &options),
                (EqFrame::Unknown, Some("unsupported_epoch".into()))
            );
        }
        assert_eq!(
            native_frame_policy(Some(0), Some(4), &options).0,
            EqFrame::Unknown
        );
        assert_eq!(
            native_frame_policy(None, Some(1), &options),
            (
                EqFrame::Unknown,
                Some("equatorial_system_needs_interface_v2".into())
            )
        );
        let forced = MountDriverOptions {
            frame_override: Some(EqFrame::J2000),
            ..options
        };
        assert_eq!(
            native_frame_policy(Some(0), Some(4), &forced),
            (EqFrame::J2000, None)
        );
    }

    #[test]
    fn esp32_profile_publishes_every_status_field() {
        let mut rig = rig(SimMountProfile::esp32_like());
        let snapshot = rig.tick(Duration::from_secs(1), None);
        let identity = rig.sim.identity().clone();
        let health = HealthState::ready(rig.clock.now_utc(), rig.clock.now_mono_ns());
        let block = status_block(
            &snapshot,
            &identity,
            &health,
            Some(&site()),
            rig.clock.now_mono_ns(),
        );
        for key in STATUS_KEYS {
            assert!(block.get(*key).is_some(), "status block lacks {key}");
        }
        assert_eq!(block["frame"], json!("jnow_apparent"));
        assert_eq!(block["alignment_mode"], json!("polar"));
        assert_eq!(block["can_flip"], json!(false));
        assert_eq!(block["pier_side"], json!("east"));
        assert_eq!(block["parked"], json!(false));
        assert_eq!(block["slewing"], json!(false));
        assert_eq!(block["state"], json!("ready"));
        assert_eq!(block["connected"], json!(true));
        assert!(block["alt"].is_number() && block["az"].is_number());
        assert!(block["lst_h"].is_number() && block["ha_h"].is_number());
        assert!(
            block["ra_j2000"].is_number(),
            "J2000 published beside native"
        );
        assert!(block["guide_rate_ra_arcsec_s"].is_number());
        assert!(block["clock_delta_s"].is_number());
        assert_eq!(block["age_ms"], json!(0));
        assert_eq!(identity.driver, DriverId::Sim);
        let lost = HealthState {
            state: LinkState::Lost,
            ..health
        };
        let block = status_block(&snapshot, &identity, &lost, None, rig.clock.now_mono_ns());
        assert_eq!(block["connected"], json!(false));
        assert_eq!(block["state"], json!("lost"));
    }

    #[test]
    fn unknown_values_stay_null_in_status() {
        let snapshot = MountSnapshot::default();
        let identity = crate::equipment::testing::fake_handles::fake_identity(DeviceKind::Mount);
        let block = status_block(&snapshot, &identity, &HealthState::default(), None, 0);
        for key in [
            "tracking",
            "ra",
            "dec",
            "parked",
            "pier_side",
            "alt",
            "lst_h",
            "site",
            "age_ms",
        ] {
            assert_eq!(
                block[key],
                Value::Null,
                "{key} must be null, never invented"
            );
        }
        assert_eq!(block["frame"], json!("unknown"));
    }

    #[test]
    fn site_derives_alt_az_and_lst_when_the_mount_reports_none() {
        let mut profile = SimMountProfile::german_equatorial();
        profile.reports_alt_az = false;
        profile.reports_sidereal_time = false;
        let mut rig = rig(profile);
        let snapshot = rig.tick(Duration::from_secs(1), None);
        assert_eq!(snapshot.state.alt_deg.source, Source::Derived);
        assert_eq!(snapshot.state.sidereal_time_h.source, Source::Derived);
        assert!(snapshot.state.alt_deg.is_fresh());
        // Without a site nothing is derived.
        let mut profile = SimMountProfile::german_equatorial();
        profile.reports_alt_az = false;
        profile.reports_sidereal_time = false;
        profile.site = None;
        let mut rig = rig_with(profile, MountSettings::default(), None);
        let snapshot = rig.tick(Duration::from_secs(1), None);
        assert_eq!(snapshot.state.alt_deg.value, None);
        assert_eq!(snapshot.ha_h, None);
    }

    #[test]
    fn a_j2000_mount_derives_alt_az_and_hour_angle_from_coordinates_of_date() {
        let mut profile = SimMountProfile::esp32_like();
        profile.equatorial_system = Some(2);
        profile.reports_alt_az = false;
        let mut rig = rig(profile);
        rig.runtime = MountRuntime::new(
            MountSettings::default(),
            MountLimits::default(),
            Arc::new(ShiftConverter {
                ra_h: 0.1,
                dec_deg: 0.3,
            }),
            rig.clock.clone(),
            rig.sim.identity().connection_id,
            Some(site()),
        );
        rig.runtime.on_connected(&mut rig.sim).unwrap();
        let snapshot = rig.tick(Duration::from_secs(1), None);
        assert_eq!(snapshot.capabilities.native_frame, EqFrame::J2000);
        let s = &snapshot.state;
        let (lst, ra, dec) = (
            s.sidereal_time_h.value.unwrap(),
            s.ra_h.value.unwrap(),
            s.dec_deg.value.unwrap(),
        );
        let ha_of_date = limits::hour_angle_h(lst, ra + 0.1);
        let expected = limits::alt_az(ha_of_date, dec + 0.3, site().lat_deg);
        assert_eq!(s.alt_deg.source, Source::Derived);
        assert!((s.alt_deg.value.unwrap() - expected.alt_deg).abs() < 1e-9);
        assert!((s.az_deg.value.unwrap() - expected.az_deg).abs() < 1e-9);
        assert!((snapshot.ha_h.unwrap() - ha_of_date).abs() < 1e-9);
        let raw = limits::alt_az(limits::hour_angle_h(lst, ra), dec, site().lat_deg);
        assert!(
            (raw.alt_deg - expected.alt_deg).abs() > 0.1,
            "the test shift must matter"
        );
        assert_eq!(
            rig.runtime.tracker.of_date().unwrap().frame,
            EqFrame::JNowApparent
        );
    }

    #[test]
    fn pier_side_changes_and_can_flip_on_a_german_mount() {
        let mut rig = rig(SimMountProfile::german_equatorial());
        let first = rig.tick(Duration::from_secs(1), None);
        assert!(first.can_flip, "a GEM with a reported pier side can flip");
        let lst = first.state.sidereal_time_h.value.unwrap();
        // West of the meridian, then east of it: the side changes.
        for ha in [2.0, -2.0] {
            let target = EqCoord::new(EqCoord::wrap_ra_h(lst - ha), 20.0, EqFrame::Unknown);
            let outcome = rig.run(MountCmd::Slew { target }, operator()).unwrap();
            assert_eq!(outcome, OpOutcome::Succeeded);
        }
        assert!(rig.names().contains(&"pier_side_changed"));
        let snapshot = rig.tick(Duration::from_secs(1), None);
        assert!(snapshot.can_flip && !snapshot.pier_side_static);
    }

    #[test]
    fn a_static_pier_side_on_a_german_mount_is_published_and_disables_flips() {
        let mut profile = SimMountProfile::german_equatorial();
        profile.pier_side = crate::equipment::sim::mount::SimPierSide::Static(PierSide::East);
        let mut rig = rig(profile);
        let lst = rig
            .tick(Duration::from_secs(1), None)
            .state
            .sidereal_time_h
            .value
            .unwrap();
        for ha in [2.0, -2.0] {
            let target = EqCoord::new(EqCoord::wrap_ra_h(lst - ha), 20.0, EqFrame::Unknown);
            assert_eq!(
                rig.run(MountCmd::Slew { target }, operator()).unwrap(),
                OpOutcome::Succeeded
            );
        }
        let snapshot = rig.tick(Duration::from_secs(1), None);
        assert!(snapshot.pier_side_static);
        assert!(!snapshot.can_flip, "D14: a static pier side never flips");
        let names = rig.names();
        assert_eq!(
            names.iter().filter(|n| **n == "pier_side_static").count(),
            1
        );
        assert!(!names.contains(&"pier_side_changed"));
    }

    #[test]
    fn the_esp32_profile_never_flips_even_with_slews_on_both_sides() {
        let mut rig = rig(SimMountProfile::esp32_like());
        let lst = rig
            .tick(Duration::from_secs(1), None)
            .state
            .sidereal_time_h
            .value
            .unwrap();
        for ha in [2.0, -2.0] {
            let target = EqCoord::new(EqCoord::wrap_ra_h(lst - ha), 20.0, EqFrame::Unknown);
            assert_eq!(
                rig.run(MountCmd::Slew { target }, operator()).unwrap(),
                OpOutcome::Succeeded
            );
        }
        let snapshot = rig.tick(Duration::from_secs(1), None);
        assert!(!snapshot.can_flip, "Polar alignment never flips (D14)");
        assert!(
            !snapshot.pier_side_static,
            "static detection is for GermanPolar only"
        );
    }

    #[test]
    fn slews_by_other_clients_still_produce_arrived_and_settled() {
        let mut rig = rig(SimMountProfile::esp32_like());
        rig.runtime.take_events();
        let lst = rig
            .tick(Duration::from_secs(1), None)
            .state
            .sidereal_time_h
            .value
            .unwrap();
        rig.sim.control().external_goto(EqCoord::new(
            EqCoord::wrap_ra_h(lst - 1.0),
            30.0,
            EqFrame::JNowApparent,
        ));
        for _ in 0..200 {
            rig.tick(Duration::from_millis(500), None);
        }
        let names = rig.names();
        let started = names.iter().position(|n| *n == "slew_started").unwrap();
        let arrived = names.iter().position(|n| *n == "slew_arrived").unwrap();
        let settled = names.iter().position(|n| *n == "slew_settled").unwrap();
        assert!(started < arrived && arrived < settled, "{names:?}");
    }

    #[test]
    fn observed_tracking_and_park_changes_are_announced_once() {
        let mut rig = rig(SimMountProfile::esp32_like());
        rig.runtime.take_events();
        rig.sim.control().set_tracking_external(false);
        rig.tick(Duration::from_secs(1), None);
        rig.tick(Duration::from_secs(1), None);
        let events = rig.runtime.take_events();
        let tracking: Vec<_> = events
            .iter()
            .filter(|e| e.name() == "tracking_changed")
            .collect();
        assert_eq!(tracking.len(), 1);
        assert!(matches!(
            &tracking[0].event,
            EventKind::TrackingChanged { on: Some(false), reason: Some(r), source: None, .. } if r == "observed"
        ));
        // Our own tracking command announces itself, so no duplicate.
        assert_eq!(
            rig.run(MountCmd::SetTracking(true), operator()).unwrap(),
            OpOutcome::Succeeded
        );
        rig.tick(Duration::from_secs(1), None);
        let events = rig.runtime.take_events();
        let tracking: Vec<_> = events
            .iter()
            .filter(|e| e.name() == "tracking_changed")
            .collect();
        assert_eq!(tracking.len(), 1);
        assert!(tracking[0].command_id.is_some());
    }

    #[test]
    fn jog_renewals_and_second_axes_are_absorbed_and_abort_cancels() {
        let mut rig = rig(SimMountProfile::esp32_like());
        let meta = operator();
        assert!(matches!(
            rig.runtime.plan(MountCmd::RenewJog, meta.clone(), None),
            MountPlan::Refused(_)
        ));
        assert!(matches!(
            rig.runtime
                .plan(MountCmd::RenewJog, meta.clone(), Some(OpKind::MoveAxis)),
            MountPlan::Absorbed(OpOutcome::Succeeded)
        ));
        assert!(matches!(
            rig.runtime.plan(
                MountCmd::MoveAxis {
                    axis: 1,
                    rate_deg_s: 0.5
                },
                meta.clone(),
                Some(OpKind::MoveAxis)
            ),
            MountPlan::Absorbed(OpOutcome::Succeeded)
        ));
        assert!(matches!(
            rig.runtime
                .plan(MountCmd::AbortSlew, meta.clone(), Some(OpKind::Slew)),
            MountPlan::CancelActive(op) if op.kind() == OpKind::AbortSlew
        ));
        // Anything else is left to the worker's preemption rule.
        assert!(matches!(
            rig.runtime.plan(MountCmd::Park, meta.clone(), Some(OpKind::Slew)),
            MountPlan::Start(op) if op.kind() == OpKind::Park
        ));
        assert!(matches!(
            rig.runtime
                .plan(MountCmd::RenewJog, meta, Some(OpKind::Slew)),
            MountPlan::Refused(_)
        ));
    }

    fn meta_with(source: CommandSource, priority: Priority) -> CommandMeta {
        CommandMeta::new(CommandId::new(), source).with_priority(priority)
    }

    #[test]
    fn command_meta_ranks_follow_the_arbiter_table() {
        let operator = operator();
        assert_eq!(operator.rank, 5, "an operator holds rank 5");
        let safety = CommandMeta::new(CommandId::new(), CommandSource::Safety);
        assert_eq!((safety.priority, safety.rank), (Priority::Safety, 6));
        let recovery = meta_with(CommandSource::System, Priority::MountRecovery);
        assert_eq!(recovery.rank, 4);
        assert!(safety.outranks(operator.rank));
        assert!(!recovery.outranks(operator.rank), "never over an operator");
        assert!(!operator.outranks(1), "an operator needs override");
        let overriding = CommandMeta::new(
            CommandId::new(),
            CommandSource::Operator { overriding: true },
        );
        assert!(overriding.outranks(5) && !overriding.outranks(6));
        let estop = meta_with(CommandSource::Safety, Priority::EStop);
        assert!(estop.outranks(6));
        let permit = Permit::unmanaged(CommandId::new(), Priority::EStop, CommandSource::Safety);
        let from = CommandMeta::from_permit(&permit);
        assert_eq!((from.priority, from.rank), (Priority::EStop, 7));
    }

    /// Start an operator slew to 3 h west of the meridian and step it until
    /// the mount is moving.
    fn slewing(rig: &mut Rig, meta: &CommandMeta) -> MountOp {
        let lst = rig
            .tick(Duration::from_secs(1), None)
            .state
            .sidereal_time_h
            .value
            .unwrap();
        let target = EqCoord::new(EqCoord::wrap_ra_h(lst - 3.0), 60.0, EqFrame::Unknown);
        let mut op = build_operation(
            &rig.runtime.shared(),
            MountCmd::Slew { target },
            meta.clone(),
        )
        .unwrap();
        op.start(&mut rig.sim, rig.clock.as_ref()).unwrap();
        assert_eq!(advance(rig, &mut op, meta, 2), None);
        assert!(rig.sim.control().slewing());
        op
    }

    #[test]
    fn a_safety_park_preempts_an_operator_slew() {
        let mut rig = rig(SimMountProfile::esp32_like());
        let slew_meta = operator();
        let mut slew = slewing(&mut rig, &slew_meta);
        // The operator's own park during its slew is refused by the rule.
        let (refused, cancelled) =
            rig.run_during(MountCmd::Park, operator(), &mut slew, &slew_meta);
        assert!(!cancelled);
        assert_eq!(
            refused.unwrap_err().class,
            crate::equipment::types::ErrorClass::InvalidOperation
        );
        assert!(rig.sim.control().slewing(), "the slew goes on");
        // The unsafe policy's park outranks it: the slew is cancelled first.
        let safety = CommandMeta::new(CommandId::new(), CommandSource::Safety);
        let (outcome, cancelled) = rig.run_during(MountCmd::Park, safety, &mut slew, &slew_meta);
        assert!(cancelled);
        assert_eq!(outcome.unwrap(), OpOutcome::Succeeded);
        let snapshot = rig.tick(Duration::from_secs(1), None);
        assert!(is_parked(&snapshot.state));
        let calls = rig.sim.control().calls();
        let abort = calls.iter().position(|c| c == "abort_slew").unwrap();
        let park = calls.iter().position(|c| c == "park").unwrap();
        assert!(abort < park, "{calls:?}");
    }

    #[test]
    fn safety_tracking_off_preempts_an_operator_slew() {
        let mut rig = rig(SimMountProfile::esp32_like());
        let slew_meta = operator();
        let mut slew = slewing(&mut rig, &slew_meta);
        let estop = meta_with(CommandSource::Safety, Priority::EStop);
        let (outcome, cancelled) =
            rig.run_during(MountCmd::SetTracking(false), estop, &mut slew, &slew_meta);
        assert!(cancelled);
        assert_eq!(outcome.unwrap(), OpOutcome::Succeeded);
        assert!(!rig.sim.control().tracking());
        assert!(!rig.sim.control().slewing());
    }

    #[test]
    fn abort_during_a_setter_still_stops_a_slew_by_another_client() {
        let mut rig = rig(SimMountProfile::esp32_like());
        let lst = rig
            .tick(Duration::from_secs(1), None)
            .state
            .sidereal_time_h
            .value
            .unwrap();
        rig.sim.control().external_goto(EqCoord::new(
            EqCoord::wrap_ra_h(lst - 3.0),
            60.0,
            EqFrame::JNowApparent,
        ));
        assert!(rig.sim.control().slewing());
        // A rate setter is running; its own cancel does no I/O.
        let meta = operator();
        let mut setter = build_operation(
            &rig.runtime.shared(),
            MountCmd::SetTrackingRate(crate::equipment::types::DriveRate::Sidereal),
            meta.clone(),
        )
        .unwrap();
        setter.start(&mut rig.sim, rig.clock.as_ref()).unwrap();
        let (outcome, cancelled) =
            rig.run_during(MountCmd::AbortSlew, operator(), &mut setter, &meta);
        assert!(cancelled);
        assert_eq!(outcome.unwrap(), OpOutcome::Succeeded);
        assert!(!rig.sim.control().slewing(), "AbortSlew reached the mount");
        assert!(rig
            .sim
            .control()
            .calls()
            .contains(&"abort_slew".to_string()));
    }

    #[test]
    fn disconnect_is_refused_mid_slew_unless_forced() {
        let mut rig = rig(SimMountProfile::esp32_like());
        assert!(rig.runtime.check_disconnect(None, false).is_ok());
        assert!(rig
            .runtime
            .check_disconnect(Some(OpKind::Slew), false)
            .is_err());
        assert!(rig
            .runtime
            .check_disconnect(Some(OpKind::Slew), true)
            .is_ok());
        assert!(rig.runtime.prepare_disconnect(&mut rig.sim).is_ok());
        assert!(rig
            .runtime
            .shutdown(&mut rig.sim, Some(OpKind::Slew))
            .is_ok());
    }

    #[test]
    fn events_carry_the_device_and_connection() {
        let rig = rig(SimMountProfile::esp32_like());
        let shared = rig.runtime.shared();
        shared.emit(EventKind::Homed, Some(CommandId::new()));
        let events = rig.runtime.take_events();
        let homed = events.iter().find(|e| e.name() == "homed").unwrap();
        assert_eq!(homed.device_kind, Some(DeviceKind::Mount));
        assert_eq!(homed.connection_id, Some(rig.sim.identity().connection_id));
        assert!(homed.command_id.is_some());
        assert!(rig.runtime.take_events().is_empty());
    }

    fn mount_with_axes() -> MountSnapshot {
        let mut snapshot = MountSnapshot::default();
        snapshot.capabilities.can_move_axis = [true, true, false];
        snapshot
    }

    #[tokio::test]
    async fn estop_aborts_then_queues_one_stop_at_estop_rank() {
        let (handle, fake) = crate::equipment::testing::fake_handles::fake_mount(mount_with_axes());
        let report = estop_mount(&handle, true).await;
        assert!(report.result().is_ok(), "{report:?}");
        assert_eq!(fake.abort_path().calls(), 1);
        assert!(fake.wait_for_commands(1, Duration::from_secs(5)));
        assert_eq!(
            fake.commands(),
            vec![MountCmd::SetTracking(false)],
            "one e-stop-rank command, which also stops the axes"
        );
        assert_eq!(report.tracking_off(), Some(&Ok(())));
        // Tracking left on: MoveAxis 0 on each CanMoveAxis axis instead.
        let report = estop_mount(&handle, false).await;
        assert_eq!(fake.abort_path().calls(), 2);
        assert_eq!(report.tracking_off(), None);
        assert!(fake.wait_for_commands(3, Duration::from_secs(5)));
        assert_eq!(
            fake.commands()[1..],
            [
                MountCmd::MoveAxis {
                    axis: 0,
                    rate_deg_s: 0.0
                },
                MountCmd::MoveAxis {
                    axis: 1,
                    rate_deg_s: 0.0
                },
            ]
        );
        assert!(report.result().is_ok(), "{report:?}");
    }

    #[test]
    fn estop_tracking_off_stops_every_axis_first_and_preempts_a_slew() {
        let mut rig = rig(SimMountProfile::esp32_like());
        let slew_meta = operator();
        let mut slew = slewing(&mut rig, &slew_meta);
        // Another client has the secondary axis moving.
        rig.sim.move_axis(1, 2.0).unwrap();
        let before = rig.sim.control().calls().len();
        let estop = meta_with(CommandSource::Safety, Priority::EStop);
        let (outcome, cancelled) =
            rig.run_during(MountCmd::SetTracking(false), estop, &mut slew, &slew_meta);
        assert!(cancelled, "e-stop rank preempts the operator slew");
        assert_eq!(outcome.unwrap(), OpOutcome::Succeeded);
        assert_eq!(rig.sim.control().axis_rates(), [0.0; 3]);
        assert!(!rig.sim.control().tracking());
        let calls = rig.sim.control().calls()[before..].to_vec();
        let at = |name: &str| calls.iter().position(|c| c == name);
        let (axis0, axis1, off) = (
            at("move_axis:0:0").expect("axis 0 stopped"),
            at("move_axis:1:0").expect("axis 1 stopped"),
            at("tracking_off").expect("tracking off"),
        );
        assert!(
            axis0 < off && axis1 < off,
            "the section 2.9 order: {calls:?}"
        );
        assert!(
            !calls.iter().any(|c| c.starts_with("move_axis:2")),
            "axis 2 cannot move"
        );
        // An ordinary tracking off does not touch the axes.
        rig.sim.move_axis(1, 2.0).unwrap();
        let outcome = rig.run(MountCmd::SetTracking(false), operator()).unwrap();
        assert_eq!(outcome, OpOutcome::Succeeded);
        assert_eq!(rig.sim.control().axis_rates()[1], 2.0);
    }

    #[test]
    fn estop_tracking_off_reports_a_failed_axis_stop_but_still_stops_tracking() {
        let mut rig = rig(SimMountProfile::esp32_like());
        rig.sim
            .control()
            .update_faults(|f| f.move_axis_fails = true);
        let estop = meta_with(CommandSource::Safety, Priority::EStop);
        let outcome = rig.run(MountCmd::SetTracking(false), estop).unwrap();
        assert!(
            matches!(&outcome, OpOutcome::Failed(e) if e.message.contains("MoveAxis")),
            "{outcome:?}"
        );
        assert!(!rig.sim.control().tracking(), "tracking off was still sent");
    }

    #[test]
    fn estop_move_axis_stops_without_tracking_stop_both_axes_through_the_worker_rules() {
        let mut rig = rig(SimMountProfile::esp32_like());
        let slew_meta = operator();
        let mut slew = slewing(&mut rig, &slew_meta);
        rig.sim.move_axis(1, 2.0).unwrap();
        let estop = meta_with(CommandSource::Safety, Priority::EStop);
        let axis = |axis| MountCmd::MoveAxis {
            axis,
            rate_deg_s: 0.0,
        };
        // The first MoveAxis 0 outranks the slew: the worker cancels it and
        // starts a one-shot stop.
        let MountPlan::Start(mut stop) =
            rig.runtime.plan(axis(0), estop.clone(), Some(OpKind::Slew))
        else {
            panic!("e-stop MoveAxis 0 must be planned to start");
        };
        assert!(estop.outranks(slew_meta.rank));
        slew.cancel(&mut rig.sim).unwrap();
        stop.start(&mut rig.sim, rig.clock.as_ref()).unwrap();
        // The second is absorbed by it, not refused as a conflict, and still
        // reaches the hardware although this stop never moved axis 1.
        assert!(matches!(
            rig.runtime
                .plan(axis(1), estop.clone(), Some(OpKind::MoveAxis)),
            MountPlan::Absorbed(OpOutcome::Succeeded)
        ));
        assert_eq!(
            advance(&mut rig, &mut stop, &estop, 4),
            Some(OpOutcome::Succeeded)
        );
        assert_eq!(rig.sim.control().axis_rates(), [0.0; 3]);
        assert!(!rig.runtime.shared().jog_active());
    }

    #[tokio::test]
    async fn estop_reports_a_tracking_off_that_is_refused() {
        use crate::equipment::testing::fake_handles::FakeResponse;
        let (handle, fake) =
            crate::equipment::testing::fake_handles::fake_mount(MountSnapshot::default());
        fake.set_responder(|cmd, _| {
            if matches!(cmd, MountCmd::SetTracking(false)) {
                FakeResponse::reject(DeviceError::invalid_operation("refused"))
            } else {
                FakeResponse::succeed()
            }
        });
        let report = estop_mount(&handle, true).await;
        assert!(report.abort.is_ok());
        assert!(matches!(report.tracking_off(), Some(Err(e)) if e.message == "refused"));
        assert!(report.result().is_err(), "e-stop must not report success");
        assert_eq!(fake.commands().len(), 1, "sent once, never retried");
    }

    #[tokio::test]
    async fn estop_queues_tracking_off_without_waiting_for_a_slow_abort() {
        use crate::equipment::testing::fake_handles::FakeResponse;
        let (handle, fake) =
            crate::equipment::testing::fake_handles::fake_mount(MountSnapshot::default());
        fake.abort_path()
            .set_delay(Some(Duration::from_millis(800)));
        let arrived = Arc::new(Mutex::new(None));
        let seen = Arc::clone(&arrived);
        fake.set_responder(move |cmd, _| {
            if matches!(cmd, MountCmd::SetTracking(false)) {
                *lock(&seen) = Some(std::time::Instant::now());
            }
            FakeResponse::succeed()
        });
        let started = std::time::Instant::now();
        let report = estop_mount(&handle, true).await;
        assert!(report.result().is_ok(), "{report:?}");
        let at = lock(&arrived).expect("tracking off was sent");
        assert!(
            at.duration_since(started) < Duration::from_millis(500),
            "tracking off waited for the abort path: {:?}",
            at.duration_since(started)
        );
        assert!(started.elapsed() >= Duration::from_millis(800));
    }
}

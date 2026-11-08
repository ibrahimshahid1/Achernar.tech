//! The integration spine's equipment-side half (spec §2.6, §2.10, §5.3
//! WP1B-1). Owner: WP1B-1.
//!
//! [`SpineContext`] is what one daemon instance knows about its equipment
//! besides the hub: the parsed `[equipment]`-family configuration, the
//! persisted Alpaca ClientID, the profile store, the per-connection runtimes
//! the API needs (the mount's shared state, the focuser session, the filter
//! changer, the PHD2 connection, the safety supervisor) and the swappable
//! gate checks.
//!
//! The connect half builds a driver for a public driver id (§2.12, D1, O3,
//! O4), runs the device's connect sequence on a blocking thread (§2.6 steps
//! 1-8) and starts its worker ([`spawn_device`]). The API layer
//! (`api/equipment.rs`) publishes the slot and the registry (steps 9-10).
//!
//! Nothing here opens a device, a socket or a file until a `/connect`
//! arrives (D13); the inert context of `AppState::new` holds defaults only.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
use std::time::Duration;

use futures::future::BoxFuture;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::config::{AchernarConfig, EquipmentConfigs};

use super::alpaca::client::{AlpacaClient, ClientConfig, ClientId};
use super::alpaca::discovery::{self, AlpacaAddress, AlpacaOption, DiscoveryOptions};
use super::alpaca::filterwheel::AlpacaFilterWheel;
use super::alpaca::focuser::AlpacaFocuser;
use super::alpaca::observingconditions::AlpacaObservingConditions;
use super::alpaca::safetymonitor::AlpacaSafetyMonitor;
use super::alpaca::telescope::AlpacaTelescope;
use super::astro;
use super::commands::{GateError, GatedCommand};
use super::events::{AuditRecord, EquipmentEvent, GroundTruthRecord};
use super::filter::change::{FilterChanger, HandshakeServices, TokioTimer};
use super::filter::reconcile::{reconcile_at_connect, ReconciledWheel};
use super::filter::{shared_wheel, FilterEventSink, FilterSettings, SharedWheel};
use super::focuser::{FocuserDeps, FocuserSafetyCheck, FocuserSession};
use super::guider::service::Phd2Connection;
use super::handle::{DeviceHandle, Permit};
use super::hub::EquipmentHub;
use super::kinds::{FilterWheelKind, FocuserKind, MountKind, SafetyKind, WeatherKind};
use super::legacy::ols_mount::open_ols_mount;
use super::mount::limits::{MountLimits, MountLimitsCheck};
use super::mount::{MountRuntime, MountSettings, MountShared};

use super::filter::reconcile::FilterWheelProfile;
use super::profile::ProfileStore;
use super::resume::BlockRequest as HubBlockRequest;
use super::safety::monitor_hub::SafetyGateCheck;
use super::safety::policy::{SafetySupervisor, SharedTargets};
use super::safety::{BlockRequest, EventSink as SafetyEventSink, ResumeBlockStore, SafetyGatePort};
use super::sim::filterwheel::{SimFilterWheel, SimFilterWheelConfig, SimFilterWheelControl};
use super::sim::focuser::{SimFocuser, SimFocuserConfig, SimFocuserControl};
use super::sim::mount::{SimMount, SimMountControl, SimMountProfile};
use super::sim::safety::{SimSafetyControl, SimSafetyMonitor};
use super::sim::weather::{SimWeather, SimWeatherControl};
use super::snapshot::{
    EquipmentSnapshot, FilterWheelSnapshot, FocuserSnapshot, MountSnapshot, ResumeBlock,
    SafetySnapshot, WeatherSnapshot,
};
use super::traits::{
    AlpacaTransport, Clock, ConnectReport, Device, FilterWheelDriver, FocuserDriver, MountDriver,
    SafetyCheck, SafetyDriver, WeatherDriver,
};
use super::types::{
    CommandSource, DeviceError, DeviceIdentity, DeviceKind, DeviceResult, DriverId, HealthState,
};
use super::worker::{
    spawn_worker, Cadence, DeviceCommand, WorkerConfig, WorkerDeps, WorkerHandle, WorkerKind,
};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// Driver ids (§2.12, D1, O3, O4, O8)
// ---------------------------------------------------------------------------

/// A public driver id resolved for one device kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriverChoice {
    /// Native Alpaca over the WP1A-1 client.
    Alpaca,
    /// The in-process simulator.
    Sim,
    /// The OLS `indi:remote` mount (O3 interim; where compiled).
    OlsIndi,
    /// PHD2's event server (guider).
    Phd2,
}

/// Why a driver id cannot be used for a kind: 400 for an unknown or a
/// retired id, 501 for one that is planned or not compiled into this build.
#[derive(Debug, Clone, PartialEq)]
pub struct DriverRefusal {
    pub status: u16,
    pub code: &'static str,
    pub message: String,
}

impl DriverRefusal {
    fn planned(message: impl Into<String>) -> Self {
        Self {
            status: 501,
            code: "driver_not_available",
            message: message.into(),
        }
    }

    /// A driver id that was removed: refused with its replacement named,
    /// never mapped to another driver behind the caller's back.
    fn retired(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            code: "driver_retired",
            message: message.into(),
        }
    }

    fn unknown(kind: DeviceKind, driver: &str) -> Self {
        Self {
            status: 400,
            code: "unsupported_driver",
            message: format!(
                "unknown {} driver '{driver}'; GET /api/v1/drivers lists what this build offers",
                super::types::kind_name(kind)
            ),
        }
    }
}

/// True when the OLS bridge was built with libindi (`WITH_INDI`).
pub fn ols_indi_compiled() -> bool {
    let flags = unsafe { crate::hardware::ols_ffi::ols_driver_capability_flags() as u32 };
    flags & crate::hardware::ols_ffi::OLS_CAP_CAMERA_INDI != 0
}

/// Resolve `driver` for `kind` (§2.12). INDI reaches the OLS `indi:remote`
/// mount where compiled (D1, O3) and is otherwise Tier 2; LX200, MoonLite
/// and USB serial are reached through indiserver or ASCOM Remote (O4);
/// rotator, cover, switch and dome are Tier 2 (WP2-0, WP2-5). The OLS
/// Alpaca adapter's hidden id `ols_alpaca` was removed with the OLS Alpaca
/// mount and focuser (O2) and is refused with `alpaca` named instead.
pub fn resolve_driver(kind: DeviceKind, driver: &str) -> Result<DriverChoice, DriverRefusal> {
    let driver = driver.trim().to_ascii_lowercase();
    let tier2 = |what: &str| {
        DriverRefusal::planned(format!(
            "{what} is planned for Tier 2 and is not available in this build"
        ))
    };
    let ols_alpaca_retired = || {
        DriverRefusal::retired(format!(
            "driver 'ols_alpaca' (the OLS Alpaca {} adapter) was removed at the end of Tier 1 \
             (owner decision O2); use driver 'alpaca', the native Alpaca client, with the same \
             option (http(s)://host:port#device=N)",
            super::types::kind_name(kind)
        ))
    };
    match kind {
        DeviceKind::Mount => match driver.as_str() {
            "alpaca" => Ok(DriverChoice::Alpaca),
            "sim" | "simulator" => Ok(DriverChoice::Sim),
            "ols_alpaca" => Err(ols_alpaca_retired()),
            "indi" | "ols_indi" if ols_indi_compiled() => Ok(DriverChoice::OlsIndi),
            "indi" | "ols_indi" => Err(DriverRefusal::planned(
                "INDI mounts need the OLS bridge built with libindi (interim, O3) or the \
                 native INDI client (Tier 2); neither is in this build",
            )),
            "lx200" | "usb" | "usb_serial" => Err(DriverRefusal::planned(
                "LX200 mounts are reached through indi_lx200generic (native INDI client, Tier 2) \
                 or an ASCOM driver behind ASCOM Remote (O4); there is no native serial driver",
            )),
            other => Err(DriverRefusal::unknown(kind, other)),
        },
        DeviceKind::Focuser => match driver.as_str() {
            "alpaca" => Ok(DriverChoice::Alpaca),
            "sim" | "simulator" => Ok(DriverChoice::Sim),
            "ols_alpaca" => Err(ols_alpaca_retired()),
            "indi" => Err(tier2("The native INDI focuser")),
            "moonlite" | "usb" | "usb_serial" => Err(DriverRefusal::planned(
                "MoonLite and USB serial focusers are reached through indi_moonlite (native \
                 INDI client, Tier 2) or ASCOM Remote (O4); there is no native serial driver",
            )),
            other => Err(DriverRefusal::unknown(kind, other)),
        },
        DeviceKind::FilterWheel | DeviceKind::Weather | DeviceKind::SafetyMonitor => {
            match driver.as_str() {
                "alpaca" => Ok(DriverChoice::Alpaca),
                "sim" | "simulator" => Ok(DriverChoice::Sim),
                "indi" => Err(tier2("The native INDI client")),
                other => Err(DriverRefusal::unknown(kind, other)),
            }
        }
        DeviceKind::Guider => match driver.as_str() {
            "phd2" => Ok(DriverChoice::Phd2),
            "direct" => Err(tier2("The direct guider (O8)")),
            other => Err(DriverRefusal::unknown(kind, other)),
        },
        DeviceKind::Rotator
        | DeviceKind::CoverCalibrator
        | DeviceKind::Switch
        | DeviceKind::Dome => Err(tier2(&format!(
            "The {} driver",
            super::types::kind_name(kind)
        ))),
        DeviceKind::Camera => Err(DriverRefusal::unknown(kind, &driver)),
    }
}

// ---------------------------------------------------------------------------
// Swappable gate checks
// ---------------------------------------------------------------------------

/// A gate plug-in whose implementation follows the connected device (the
/// gate has no unregister; a reconnect swaps the inner check).
pub struct SlotCheck {
    name: &'static str,
    inner: RwLock<Option<Arc<dyn SafetyCheck>>>,
}

impl SlotCheck {
    pub fn new(name: &'static str) -> Arc<Self> {
        Arc::new(Self {
            name,
            inner: RwLock::new(None),
        })
    }

    pub fn set(&self, check: Option<Arc<dyn SafetyCheck>>) {
        *self.inner.write().unwrap_or_else(|e| e.into_inner()) = check;
    }

    fn current(&self) -> Option<Arc<dyn SafetyCheck>> {
        self.inner.read().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

impl SafetyCheck for SlotCheck {
    fn name(&self) -> &'static str {
        self.current().map_or(self.name, |check| check.name())
    }

    fn applies_to(&self, cmd: &GatedCommand) -> bool {
        self.current().is_some_and(|check| check.applies_to(cmd))
    }

    fn check(
        &self,
        cmd: &GatedCommand,
        snap: &EquipmentSnapshot,
    ) -> Result<(), super::commands::SafetyRejection> {
        match self.current() {
            Some(check) => check.check(cmd, snap),
            None => Ok(()),
        }
    }

    fn annotations(&self, cmd: &GatedCommand, snap: &EquipmentSnapshot) -> Vec<String> {
        self.current()
            .map(|check| check.annotations(cmd, snap))
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Ports over the hub (WP1A-6, WP1A-8)
// ---------------------------------------------------------------------------

/// The hub as the filter layer's event sink and resume-block registry.
pub struct HubFilterSink {
    hub: Arc<EquipmentHub>,
}

impl FilterEventSink for HubFilterSink {
    fn publish(&self, event: EquipmentEvent) {
        self.hub.emit(event);
    }

    fn raise_resume_block(&self, request: HubBlockRequest) -> ResumeBlock {
        self.hub.raise_block(request)
    }
}

/// The hub and gate as the safety layer's ports.
pub struct HubSafetyPorts {
    hub: Arc<EquipmentHub>,
}

impl SafetyEventSink for HubSafetyPorts {
    fn emit(&self, event: EquipmentEvent) {
        self.hub.emit(event);
    }
}

impl ResumeBlockStore for HubSafetyPorts {
    fn raise(&self, request: BlockRequest) -> ResumeBlock {
        let mut hub_request =
            HubBlockRequest::new(request.kind, request.reason).with_command(request.command_id);
        if let Some(kind) = request.device_kind {
            hub_request = hub_request.for_device(kind, request.connection_id);
        }
        self.hub.raise_block(hub_request)
    }

    fn clear(&self, id: Uuid, by: &CommandSource) -> Option<ResumeBlock> {
        self.hub.clear_block_by_id(id, by)
    }

    fn active(&self) -> Vec<ResumeBlock> {
        self.hub.resume().active()
    }
}

impl SafetyGatePort for HubSafetyPorts {
    fn authorize<'a>(&'a self, cmd: &'a GatedCommand) -> BoxFuture<'a, Result<Permit, GateError>> {
        Box::pin(async move { self.hub.gate().authorize(cmd) })
    }

    fn audit(&self, record: AuditRecord) {
        self.hub
            .recorder()
            .record(GroundTruthRecord::Audit(Box::new(record)));
    }
}

// ---------------------------------------------------------------------------
// Worker tuning
// ---------------------------------------------------------------------------

/// Worker behaviour from `[equipment]`, with test overrides.
#[derive(Debug, Clone, Default)]
pub struct WorkerTuning {
    /// Replaces every kind's §2.7 cadence (tests).
    pub cadence: Option<Cadence>,
    /// Replaces the `[equipment]` health policy (tests).
    pub policy: Option<super::health_bridge::HealthPolicy>,
    /// The `sim` mount's profile (default: the ESP32-like profile).
    pub sim_mount_profile: Option<SimMountProfile>,
}

// ---------------------------------------------------------------------------
// The context
// ---------------------------------------------------------------------------

/// The simulators' control handles, for fault injection in tests and for
/// operators exercising a simulator.
#[derive(Clone, Default)]
pub struct SimControls {
    pub mount: Option<SimMountControl>,
    pub focuser: Option<SimFocuserControl>,
    pub filter_wheel: Option<SimFilterWheelControl>,
    pub weather: Option<SimWeatherControl>,
    pub safety: Option<SimSafetyControl>,
}

/// What the spine keeps per daemon instance (see the module docs).
pub struct SpineContext {
    pub configs: EquipmentConfigs,
    pub data_dir: Option<PathBuf>,
    client_id: ClientId,
    profiles: Option<ProfileStore>,
    clock: Arc<dyn Clock>,
    tuning: RwLock<WorkerTuning>,
    mount_check: Arc<SlotCheck>,
    focuser_check: Arc<SlotCheck>,
    filter_check: Arc<SlotCheck>,
    mount_shared: Mutex<Option<Arc<MountShared>>>,
    focuser_session: Mutex<Option<FocuserSession>>,
    filter: Mutex<Option<Arc<FilterChanger>>>,
    guider: Mutex<Option<Phd2Connection>>,
    safety: Mutex<Option<Arc<SafetySupervisor>>>,
    targets: Arc<SharedTargets>,
    sims: Mutex<SimControls>,
    tasks: Mutex<Vec<tokio::task::JoinHandle<()>>>,
    recorder: Mutex<Option<Arc<super::recorder::Recorder>>>,
}

impl std::fmt::Debug for SpineContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpineContext")
            .field("client_id", &self.client_id)
            .field("profiles", &self.profiles.is_some())
            .finish()
    }
}

impl SpineContext {
    /// The context for `hub`. Registers the device checks on its gate once
    /// (their implementations follow the connected devices) and the safety
    /// gate check (§2.8 fail-safe for new automatic actions).
    pub fn new(
        hub: &EquipmentHub,
        configs: EquipmentConfigs,
        data_dir: Option<PathBuf>,
        profiles: Option<ProfileStore>,
    ) -> Self {
        let client_id = match data_dir.as_deref() {
            Some(dir) => match ClientId::load_or_create(dir, configs.equipment.alpaca_client_id) {
                Ok((id, _)) => id,
                Err(err) => {
                    tracing::warn!(
                        "Alpaca ClientID could not be persisted ({err}); using one for this run"
                    );
                    ClientId::new(configs.equipment.alpaca_client_id)
                        .unwrap_or_else(ClientId::random)
                }
            },
            None => {
                ClientId::new(configs.equipment.alpaca_client_id).unwrap_or_else(ClientId::random)
            }
        };
        let mount_check = SlotCheck::new(super::mount::limits::CHECK_NAME);
        let focuser_check = SlotCheck::new("focuser_travel");
        let filter_check = SlotCheck::new("filter_wheel");
        let gate = hub.gate();
        gate.register_check(Arc::clone(&mount_check) as Arc<dyn SafetyCheck>);
        gate.register_check(Arc::clone(&focuser_check) as Arc<dyn SafetyCheck>);
        gate.register_check(Arc::clone(&filter_check) as Arc<dyn SafetyCheck>);
        gate.register_check(Arc::new(SafetyGateCheck));
        Self {
            configs,
            data_dir,
            client_id,
            profiles,
            clock: hub.clock(),
            tuning: RwLock::new(WorkerTuning::default()),
            mount_check,
            focuser_check,
            filter_check,
            mount_shared: Mutex::new(None),
            focuser_session: Mutex::new(None),
            filter: Mutex::new(None),
            guider: Mutex::new(None),
            safety: Mutex::new(None),
            targets: SharedTargets::new(),
            sims: Mutex::new(SimControls::default()),
            tasks: Mutex::new(Vec::new()),
            recorder: Mutex::new(None),
        }
    }

    /// The recorder `with_equipment` started, for the shutdown flush (§2.15).
    pub fn set_recorder(&self, recorder: Option<Arc<super::recorder::Recorder>>) {
        *lock(&self.recorder) = recorder;
    }

    pub fn recorder(&self) -> Option<Arc<super::recorder::Recorder>> {
        lock(&self.recorder).clone()
    }

    /// The context of `AppState::new`: default configuration, nothing
    /// persisted (no data directory, no profile store).
    pub fn inert(hub: &EquipmentHub) -> Self {
        Self::new(
            hub,
            AchernarConfig::default().equipment_configs(),
            None,
            None,
        )
    }

    pub fn clock(&self) -> Arc<dyn Clock> {
        Arc::clone(&self.clock)
    }

    pub fn client_id(&self) -> ClientId {
        self.client_id
    }

    pub fn profiles(&self) -> Option<&ProfileStore> {
        self.profiles.as_ref()
    }

    pub fn connect_timeout(&self) -> Duration {
        Duration::from_secs(self.configs.equipment.connect_timeout_s.max(1))
    }

    pub fn set_tuning(&self, tuning: WorkerTuning) {
        *self.tuning.write().unwrap_or_else(|e| e.into_inner()) = tuning;
    }

    pub fn tuning(&self) -> WorkerTuning {
        self.tuning
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub fn sims(&self) -> SimControls {
        lock(&self.sims).clone()
    }

    pub fn targets(&self) -> Arc<SharedTargets> {
        Arc::clone(&self.targets)
    }

    pub fn mount_shared(&self) -> Option<Arc<MountShared>> {
        lock(&self.mount_shared).clone()
    }

    pub fn focuser_session(&self) -> Option<FocuserSession> {
        lock(&self.focuser_session).clone()
    }

    pub fn filter_changer(&self) -> Option<Arc<FilterChanger>> {
        lock(&self.filter).clone()
    }

    pub fn safety_supervisor(&self) -> Option<Arc<SafetySupervisor>> {
        lock(&self.safety).clone()
    }

    /// Per-exposure guide statistics (H9), while PHD2 is connected.
    pub fn guider_windows(&self) -> Option<super::guider::service::GuiderWindows> {
        lock(&self.guider).as_ref().map(|c| c.windows.clone())
    }

    /// Keep a task until the context drops.
    pub fn adopt_task(&self, task: tokio::task::JoinHandle<()>) {
        let mut tasks = lock(&self.tasks);
        tasks.retain(|t| !t.is_finished());
        tasks.push(task);
    }

    /// Forget what a disconnected device left behind.
    pub fn forget(&self, kind: DeviceKind) {
        match kind {
            DeviceKind::Mount => {
                lock(&self.mount_shared).take();
                self.mount_check.set(None);
                self.targets.set_mount(None);
                lock(&self.sims).mount = None;
            }
            DeviceKind::Focuser => {
                lock(&self.focuser_session).take();
                self.focuser_check.set(None);
                lock(&self.sims).focuser = None;
            }
            DeviceKind::FilterWheel => {
                lock(&self.filter).take();
                self.filter_check.set(None);
                lock(&self.sims).filter_wheel = None;
            }
            DeviceKind::Guider => {
                if let Some(mut connection) = lock(&self.guider).take() {
                    // The service ends after its disconnect; aborting the
                    // task is safe either way (the socket closes).
                    connection.abort_task();
                }
                self.targets.set_guider(None);
            }
            DeviceKind::Weather => {
                lock(&self.sims).weather = None;
            }
            DeviceKind::SafetyMonitor => {
                lock(&self.sims).safety = None;
            }
            _ => {}
        }
    }

    pub fn set_guider(&self, connection: Phd2Connection) {
        self.targets.set_guider(Some(connection.handle.clone()));
        if let Some(mut old) = lock(&self.guider).replace(connection) {
            old.abort_task();
        }
    }

    /// The safety supervisor, created (and its 1 s evaluation started) when
    /// the first monitor connects. With no monitor the state stays `none`.
    pub fn ensure_safety(&self, hub: &Arc<EquipmentHub>) -> Arc<SafetySupervisor> {
        // The on-unsafe policy aborts the exposure through the hub's camera
        // hook (`AppState` installs it; §3.10 step 3).
        self.targets.set_camera(hub.camera());
        let mut slot = lock(&self.safety);
        if let Some(existing) = slot.as_ref() {
            return Arc::clone(existing);
        }
        let ports = Arc::new(HubSafetyPorts {
            hub: Arc::clone(hub),
        });
        let supervisor = SafetySupervisor::from_config(
            &self.configs,
            Arc::clone(&ports) as Arc<dyn SafetyGatePort>,
            Arc::clone(&ports) as Arc<dyn ResumeBlockStore>,
            Arc::clone(&ports) as Arc<dyn SafetyEventSink>,
            Arc::clone(&self.targets) as Arc<dyn super::safety::policy::PolicyTargetSource>,
            Arc::clone(&self.clock),
        );
        if tokio::runtime::Handle::try_current().is_ok() {
            self.adopt_task(supervisor.spawn(Duration::from_secs(1)));
            // The hub's aggregate (and the gate's fail-safe check) reads the
            // supervisor's combined safety.
            let mut summary = supervisor.watch_summary();
            let hub = Arc::downgrade(hub);
            self.adopt_task(tokio::spawn(async move {
                loop {
                    let current = summary.borrow_and_update().clone();
                    match hub.upgrade() {
                        Some(hub) => hub.set_safety_summary(Some(current)),
                        None => break,
                    }
                    if summary.changed().await.is_err() {
                        break;
                    }
                }
            }));
        }
        *slot = Some(Arc::clone(&supervisor));
        supervisor
    }

    fn worker_config(&self) -> WorkerConfig {
        let tuning = self.tuning();
        WorkerConfig {
            policy: tuning.policy.unwrap_or_else(|| {
                super::health_bridge::HealthPolicy::from_config(&self.configs.equipment)
            }),
            connect_timeout: self.connect_timeout(),
            connect_on_start: false,
            announce_connected: true,
            // O11 b: motion needs a permit the gate issued.
            require_gate_permits: true,
            ..WorkerConfig::default()
        }
    }

    fn alpaca_client(
        &self,
        address: &AlpacaAddress,
        device_type: &str,
    ) -> DeviceResult<AlpacaClient> {
        AlpacaClient::new(
            &address.base,
            device_type,
            address.number,
            ClientConfig::new(self.client_id),
        )
    }
}

impl Drop for SpineContext {
    fn drop(&mut self) {
        for task in lock(&self.tasks).drain(..) {
            task.abort();
        }
        if let Some(mut connection) = lock(&self.guider).take() {
            connection.abort_task();
        }
    }
}

// ---------------------------------------------------------------------------
// Alpaca addressing (§2.6 steps 1, 5; D13)
// ---------------------------------------------------------------------------

/// Resolve an Alpaca `/connect` option (blocking: broadcast discovery and
/// the management API are network calls). An empty or `"auto"` option is
/// refused unless `[equipment] allow_broadcast_discovery` (D13); when it is
/// enabled, the first server that lists a device of this type is used.
pub fn resolve_alpaca(
    ctx: &SpineContext,
    option: &str,
    device_type: &str,
) -> DeviceResult<AlpacaAddress> {
    let allow = ctx.configs.equipment.allow_broadcast_discovery;
    match discovery::parse_option(option, allow)? {
        AlpacaOption::Address(address) => Ok(address),
        AlpacaOption::Broadcast => {
            let options = DiscoveryOptions {
                port: ctx.configs.equipment.alpaca_discovery_port,
                ..DiscoveryOptions::default()
            };
            let report = discovery::discover(allow, &options)?;
            for server in report.servers {
                let base = server.base_url();
                let Ok(management) =
                    discovery::ManagementClient::new(&base, ctx.client_id, Duration::from_secs(5))
                else {
                    continue;
                };
                let Ok(devices) = management.configured_devices() else {
                    continue;
                };
                let number = devices
                    .of_type(device_type)
                    .next()
                    .map(|device| device.device_number);
                if let Some(number) = number {
                    let text = format!("{base}#device={number}");
                    if let Ok(AlpacaOption::Address(address)) =
                        discovery::parse_option(&text, false)
                    {
                        return Ok(address);
                    }
                }
            }
            Err(DeviceError::not_connected(format!(
                "Alpaca discovery found no {device_type} on the network"
            )))
        }
    }
}

/// The UniqueID the server's management API reports for this device (§2.6
/// step 5), best effort: an old server without the API keeps the
/// address-based identity key.
fn alpaca_unique_id(
    ctx: &SpineContext,
    address: &AlpacaAddress,
    device_type: &str,
) -> Option<String> {
    let management =
        discovery::ManagementClient::new(&address.base, ctx.client_id, Duration::from_secs(5))
            .ok()?;
    let devices = management.configured_devices().ok()?;
    devices
        .unique_id(device_type, address.number)
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_string)
}

fn alpaca_identity(
    kind: DeviceKind,
    address: &AlpacaAddress,
    device_type: &str,
    unique_id: Option<String>,
    name: &str,
) -> DeviceIdentity {
    let mut identity = DeviceIdentity::new(
        kind,
        DriverId::Alpaca,
        address.endpoint(device_type),
        address.identity_key(device_type, unique_id.as_deref()),
        name,
    );
    identity.unique_id = unique_id;
    identity.setup_url = Some(format!(
        "{}/setup/v1/{device_type}/{}/setup",
        address.base, address.number
    ));
    identity
}

fn transports(client: AlpacaClient) -> (Arc<dyn AlpacaTransport>, Arc<dyn AlpacaTransport>) {
    let abort = Arc::new(client.abort_channel()) as Arc<dyn AlpacaTransport>;
    (Arc::new(client) as Arc<dyn AlpacaTransport>, abort)
}

// ---------------------------------------------------------------------------
// Connect targets (§2.6 step 1)
// ---------------------------------------------------------------------------

/// What a validated `/connect` option names, known before anything is
/// opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectTarget {
    /// A native Alpaca device at a known address.
    Alpaca(AlpacaAddress),
    /// Alpaca broadcast discovery (opted in, D13): the server is found
    /// during the connect.
    Discovery,
    /// PHD2's event server.
    Phd2 { host: String, port: u16 },
    /// The in-process simulator, or the OLS INDI adapter (the OLS bridge
    /// opens its own connection).
    Local,
}

/// §2.6 step 1: parse and validate a `/connect` option without opening
/// anything (no socket, no OLS call): the Alpaca URL and `#device=N`, the
/// D13 refusal of an empty or `"auto"` option, and PHD2's `host:port`. The API
/// runs it before it touches a connected device, so a refused or malformed
/// option leaves that device connected.
pub fn validate_connect_option(
    ctx: &SpineContext,
    choice: DriverChoice,
    option: &str,
) -> DeviceResult<ConnectTarget> {
    let allow = ctx.configs.equipment.allow_broadcast_discovery;
    Ok(match choice {
        DriverChoice::Alpaca => match discovery::parse_option(option, allow)? {
            AlpacaOption::Address(address) => ConnectTarget::Alpaca(address),
            AlpacaOption::Broadcast => ConnectTarget::Discovery,
        },
        DriverChoice::Phd2 => {
            let (host, port) = super::guider::service::parse_option(option)?;
            ConnectTarget::Phd2 { host, port }
        }
        DriverChoice::Sim | DriverChoice::OlsIndi => ConnectTarget::Local,
    })
}

/// Before a connected device of `kind` is let go for a replacement, check
/// that the replacement's server answers (blocking; `timeout` bounds each
/// step): a TCP connection to the Alpaca server or to PHD2, closed at once,
/// and for Alpaca the management API's device list, which must include the
/// requested type and number when the server provides one. A mistyped
/// address or device number is then refused while the current device stays
/// connected. Nothing is sent to a device (D16). Discovery, the simulator
/// and the OLS INDI adapter are not checked.
pub fn preflight_replacement(
    ctx: &SpineContext,
    kind: DeviceKind,
    target: &ConnectTarget,
    timeout: Duration,
) -> DeviceResult<()> {
    match target {
        ConnectTarget::Alpaca(address) => {
            reachable(&address.host, address.port, timeout)?;
            let Some(device_type) = super::alpaca::device_type_for_kind(kind) else {
                return Ok(());
            };
            // Best effort: an old server without the management API is
            // checked by the connect itself.
            let listed = discovery::ManagementClient::new(&address.base, ctx.client_id, timeout)
                .and_then(|management| management.configured_devices());
            if let Ok(devices) = listed {
                if devices.skipped.is_empty() && devices.find(device_type, address.number).is_none()
                {
                    return Err(DeviceError::invalid_value(format!(
                        "{} lists no {device_type} with device number {} (POST /api/v1/discover \
                         lists what it serves)",
                        address.base, address.number
                    )));
                }
            }
            Ok(())
        }
        ConnectTarget::Phd2 { host, port } => reachable(host, *port, timeout),
        ConnectTarget::Discovery | ConnectTarget::Local => Ok(()),
    }
}

/// A TCP connection to `host:port` within `timeout`, closed at once.
fn reachable(host: &str, port: u16, timeout: Duration) -> DeviceResult<()> {
    use std::net::{TcpStream, ToSocketAddrs};
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    let addresses = (bare, port).to_socket_addrs().map_err(|err| {
        DeviceError::transport(format!("{host}:{port} cannot be resolved: {err}"))
    })?;
    let mut last = None;
    for address in addresses {
        match TcpStream::connect_timeout(&address, timeout) {
            Ok(stream) => {
                drop(stream);
                return Ok(());
            }
            Err(err) => last = Some(err),
        }
    }
    Err(match last {
        Some(err) if err.kind() == std::io::ErrorKind::TimedOut => DeviceError::timeout(format!(
            "{host}:{port} did not answer within {} ms",
            timeout.as_millis()
        )),
        Some(err) => DeviceError::transport(format!("{host}:{port} cannot be reached: {err}")),
        None => DeviceError::transport(format!("{host}:{port} resolves to no address")),
    })
}

// ---------------------------------------------------------------------------
// Prepared connections (§2.6 steps 2-8, run on a blocking thread)
// ---------------------------------------------------------------------------

/// A connected mount, not yet behind its worker.
pub struct PreparedMount {
    pub driver: Box<dyn MountDriver>,
    pub runtime: MountRuntime,
    pub report: ConnectReport,
    pub warnings: Vec<String>,
}

/// A connected focuser, not yet behind its worker.
pub struct PreparedFocuser {
    pub driver: Box<dyn FocuserDriver>,
    pub session: FocuserSession,
    pub report: ConnectReport,
    pub warnings: Vec<String>,
}

/// A connected filter wheel, not yet behind its worker.
pub struct PreparedWheel {
    pub driver: Box<dyn FilterWheelDriver>,
    pub model: SharedWheel,
    pub report: ConnectReport,
}

/// A connected weather or safety device.
pub struct PreparedSensor<D: ?Sized> {
    pub driver: Box<D>,
    pub report: ConnectReport,
}

/// The mount for `choice` (§2.12), connected, with its runtime's
/// post-connect steps done (M-a, M-b, M-m). Never moves the mount (D16).
pub fn prepare_mount(
    ctx: &SpineContext,
    choice: DriverChoice,
    option: &str,
) -> DeviceResult<PreparedMount> {
    let configs = &ctx.configs;
    let settings = MountSettings::from_configs(configs);
    let options = settings.driver_options();
    let clock = ctx.clock();
    let mut driver: Box<dyn MountDriver> = match choice {
        DriverChoice::Alpaca => {
            let address = resolve_alpaca(ctx, option, "telescope")?;
            let unique_id = alpaca_unique_id(ctx, &address, "telescope");
            let identity = alpaca_identity(
                DeviceKind::Mount,
                &address,
                "telescope",
                unique_id,
                "Alpaca telescope",
            );
            let (transport, abort) = transports(ctx.alpaca_client(&address, "telescope")?);
            Box::new(AlpacaTelescope::new(
                identity,
                transport,
                abort,
                Arc::clone(&clock),
                options,
            ))
        }
        DriverChoice::Sim => {
            let profile = ctx
                .tuning()
                .sim_mount_profile
                .unwrap_or_else(SimMountProfile::esp32_like);
            let sim = SimMount::new(profile, Arc::clone(&clock)).with_options(options);
            lock(&ctx.sims).mount = Some(sim.control());
            Box::new(sim)
        }
        DriverChoice::OlsIndi => Box::new(open_ols_mount(option, Arc::clone(&clock), options)?),
        DriverChoice::Phd2 => return Err(DeviceError::invalid_value("phd2 is not a mount driver")),
    };
    let report = driver.connect(ctx.connect_timeout())?;
    let connection_id = driver.identity().connection_id;
    let mut limits = MountLimits::new(&configs.mount.safety, &configs.mount, &configs.equipment);
    let mut warnings = report.warnings.clone();
    match astro::horizon::load_configured(configs.mount.safety.horizon_profile()) {
        Ok(Some(horizon)) => limits = limits.with_horizon(Arc::new(horizon)),
        Ok(None) => {}
        Err(err) => warnings.push(format!("horizon profile not loaded: {err}")),
    }
    let site = configs.site.site().ok().flatten();
    let mut runtime = MountRuntime::new(
        settings,
        limits,
        astro::frame_converter(),
        Arc::clone(&clock),
        connection_id,
        site,
    );
    if let Err(err) = runtime.on_connected(driver.as_mut()) {
        let _ = driver.disconnect();
        return Err(err);
    }
    Ok(PreparedMount {
        driver,
        runtime,
        report,
        warnings,
    })
}

/// The mount limits plug-in for the gate (§2.9 step 3, H13).
pub fn mount_limits_check(ctx: &SpineContext) -> Arc<dyn SafetyCheck> {
    let configs = &ctx.configs;
    let mut limits = MountLimits::new(&configs.mount.safety, &configs.mount, &configs.equipment);
    if let Ok(Some(horizon)) =
        astro::horizon::load_configured(configs.mount.safety.horizon_profile())
    {
        limits = limits.with_horizon(Arc::new(horizon));
    }
    Arc::new(MountLimitsCheck::new(limits, astro::frame_converter()))
}

/// The identity profile document for `identity_key`, read from the
/// `device_profiles` store. Blocking: call it on a blocking thread inside a
/// tokio runtime (the connect sequence runs there). `None` without a store,
/// outside a runtime, or for a key with no row.
pub fn load_profile_blocking(ctx: &SpineContext, identity_key: &str) -> Option<Value> {
    let store = ctx.profiles.clone()?;
    let runtime = tokio::runtime::Handle::try_current().ok()?;
    match runtime.block_on(store.load(identity_key)) {
        Ok(profile) => profile.map(|p| p.doc),
        Err(err) => {
            tracing::warn!("device profile {identity_key} could not be read: {err}");
            None
        }
    }
}

/// The focuser for `choice`, connected through [`FocuserSession::connect`]
/// (F-d, F-g, F-i; the identity profile restores a relative focuser,
/// §2.15).
pub fn prepare_focuser(
    ctx: &SpineContext,
    choice: DriverChoice,
    option: &str,
    deps: FocuserDeps,
) -> DeviceResult<PreparedFocuser> {
    let clock = ctx.clock();
    let driver: Box<dyn FocuserDriver> = match choice {
        DriverChoice::Alpaca => {
            let address = resolve_alpaca(ctx, option, "focuser")?;
            let unique_id = alpaca_unique_id(ctx, &address, "focuser");
            let (transport, abort) = transports(ctx.alpaca_client(&address, "focuser")?);
            Box::new(
                AlpacaFocuser::new(
                    address.base.clone(),
                    address.number,
                    transport,
                    abort,
                    Arc::clone(&clock),
                )
                .with_unique_id(unique_id.as_deref()),
            )
        }
        DriverChoice::Sim => {
            let mut config = SimFocuserConfig::absolute();
            if let Ok(position) = option.trim().parse::<i32>() {
                config.start_position = position.clamp(0, config.max_step);
            }
            let sim = SimFocuser::new(config, Arc::clone(&clock));
            lock(&ctx.sims).focuser = Some(sim.control());
            Box::new(sim)
        }
        DriverChoice::OlsIndi | DriverChoice::Phd2 => {
            return Err(DeviceError::invalid_value("not a focuser driver"))
        }
    };
    prepare_focuser_with(ctx, driver, deps)
}

/// A focuser connected from an existing driver (also used by tests with a
/// scripted simulator, and with the older `SimFocuser` behind its adapter).
pub fn prepare_focuser_with(
    ctx: &SpineContext,
    mut driver: Box<dyn FocuserDriver>,
    deps: FocuserDeps,
) -> DeviceResult<PreparedFocuser> {
    let profile = load_profile_blocking(ctx, &driver.identity().identity_key);
    let connection = FocuserSession::connect(
        driver.as_mut(),
        ctx.connect_timeout(),
        profile.as_ref(),
        &ctx.configs,
        deps,
    )?;
    Ok(PreparedFocuser {
        driver,
        session: connection.session,
        report: connection.report,
        warnings: connection.warnings,
    })
}

/// The filter wheel for `choice`, connected, with its slots reconciled
/// against the identity profile (W-b). TOML seeds are never applied (D17).
pub fn prepare_wheel(
    ctx: &SpineContext,
    hub: &Arc<EquipmentHub>,
    choice: DriverChoice,
    option: &str,
) -> DeviceResult<PreparedWheel> {
    let clock = ctx.clock();
    let mut driver: Box<dyn FilterWheelDriver> = match choice {
        DriverChoice::Alpaca => {
            let address = resolve_alpaca(ctx, option, "filterwheel")?;
            let unique_id = alpaca_unique_id(ctx, &address, "filterwheel");
            let identity = alpaca_identity(
                DeviceKind::FilterWheel,
                &address,
                "filterwheel",
                unique_id,
                "Alpaca filter wheel",
            );
            let client = ctx.alpaca_client(&address, "filterwheel")?;
            Box::new(AlpacaFilterWheel::new(
                identity,
                Arc::new(client) as Arc<dyn AlpacaTransport>,
                Arc::clone(&clock),
            ))
        }
        DriverChoice::Sim => {
            let sim = SimFilterWheel::new(SimFilterWheelConfig::default(), Arc::clone(&clock));
            lock(&ctx.sims).filter_wheel = Some(sim.control());
            Box::new(sim)
        }
        _ => return Err(DeviceError::invalid_value("not a filter-wheel driver")),
    };
    let profile = match load_profile_blocking(ctx, &driver.identity().identity_key) {
        Some(doc) => match FilterWheelProfile::from_profile_json(&doc) {
            Ok(profile) => Some(profile),
            Err(err) => {
                // A malformed profile is reported, never replaced by seeds.
                tracing::warn!("{err}; the wheel connects with device values only");
                None
            }
        },
        None => None,
    };
    let report = driver.connect(ctx.connect_timeout())?;
    let slots = match driver.slots() {
        Ok(slots) => slots,
        Err(err) => {
            let _ = driver.disconnect();
            return Err(err);
        }
    };
    let profile = profile.as_ref();
    let sink = HubFilterSink {
        hub: Arc::clone(hub),
    };
    let wheel: ReconciledWheel = reconcile_at_connect(
        driver.identity(),
        &slots,
        profile,
        clock.as_ref(),
        &sink,
        hub.recorder().as_ref(),
    );
    Ok(PreparedWheel {
        driver,
        model: shared_wheel(wheel),
        report,
    })
}

/// An ObservingConditions device for `choice`, connected.
pub fn prepare_weather(
    ctx: &SpineContext,
    choice: DriverChoice,
    option: &str,
) -> DeviceResult<PreparedSensor<dyn WeatherDriver>> {
    let clock = ctx.clock();
    let mut driver: Box<dyn WeatherDriver> = match choice {
        DriverChoice::Alpaca => {
            let address = resolve_alpaca(ctx, option, "observingconditions")?;
            let unique_id = alpaca_unique_id(ctx, &address, "observingconditions");
            let identity = alpaca_identity(
                DeviceKind::Weather,
                &address,
                "observingconditions",
                unique_id,
                "Alpaca observing conditions",
            );
            let client = ctx.alpaca_client(&address, "observingconditions")?;
            Box::new(AlpacaObservingConditions::new(
                identity,
                Arc::new(client) as Arc<dyn AlpacaTransport>,
                Arc::clone(&clock),
            ))
        }
        DriverChoice::Sim => {
            let control = SimWeatherControl::new();
            lock(&ctx.sims).weather = Some(control.clone());
            Box::new(SimWeather::new(Arc::clone(&clock), control))
        }
        _ => return Err(DeviceError::invalid_value("not a weather driver")),
    };
    let report = driver.connect(ctx.connect_timeout())?;
    Ok(PreparedSensor { driver, report })
}

/// A SafetyMonitor for `choice`, connected.
pub fn prepare_safety(
    ctx: &SpineContext,
    choice: DriverChoice,
    option: &str,
) -> DeviceResult<PreparedSensor<dyn SafetyDriver>> {
    let clock = ctx.clock();
    let mut driver: Box<dyn SafetyDriver> = match choice {
        DriverChoice::Alpaca => {
            let address = resolve_alpaca(ctx, option, "safetymonitor")?;
            let unique_id = alpaca_unique_id(ctx, &address, "safetymonitor");
            let identity = alpaca_identity(
                DeviceKind::SafetyMonitor,
                &address,
                "safetymonitor",
                unique_id,
                "Alpaca safety monitor",
            );
            let client = ctx.alpaca_client(&address, "safetymonitor")?;
            Box::new(AlpacaSafetyMonitor::new(
                identity,
                Arc::new(client) as Arc<dyn AlpacaTransport>,
                Arc::clone(&clock),
            ))
        }
        DriverChoice::Sim => {
            let control = SimSafetyControl::new();
            lock(&ctx.sims).safety = Some(control.clone());
            Box::new(SimSafetyMonitor::new(Arc::clone(&clock), control))
        }
        _ => return Err(DeviceError::invalid_value("not a safety-monitor driver")),
    };
    let report = driver.connect(ctx.connect_timeout())?;
    Ok(PreparedSensor { driver, report })
}

// ---------------------------------------------------------------------------
// Workers (§2.5, O11 i)
// ---------------------------------------------------------------------------

/// Put a connected driver behind its worker: the handle pair, the hub slot
/// and ring (attached before the worker starts, so its first events pass the
/// hub's connection filter), the worker thread and its adoption by the hub.
pub fn spawn_device<K>(
    hub: &EquipmentHub,
    ctx: &SpineContext,
    kind: K,
    driver: Box<K::Driver>,
    initial: K::Snap,
) -> DeviceResult<DeviceHandle<K::Cmd, K::Snap>>
where
    K: WorkerKind,
    K::Cmd: DeviceCommand,
    DeviceHandle<K::Cmd, K::Snap>: super::hub::HubDevice<Snap = K::Snap>,
{
    let identity = driver.identity().clone();
    let abort_path = driver.abort_path();
    let clock = hub.clock();
    let health = HealthState::ready(clock.now_utc(), clock.now_mono_ns());
    let (handle, end) = DeviceHandle::new_pair(identity.clone(), abort_path, initial, health);
    let ring = hub.attach(handle.clone());
    let worker: WorkerHandle = spawn_worker(
        kind,
        driver,
        end,
        ctx.worker_config(),
        WorkerDeps {
            clock,
            events: hub.event_sink(),
            recorder: hub.recorder(),
            ring: Some(ring),
            watchdog: Some(hub.watchdog()),
            aborts: Some(hub.gate().abort_epochs()),
        },
    )
    .map_err(|err| {
        hub.detach(identity.kind, identity.connection_id);
        DeviceError::driver(format!("the device worker could not start: {err}"))
    })?;
    hub.adopt_worker(identity.kind, identity.connection_id, worker);
    Ok(handle)
}

fn apply_cadence<K>(kind: K, ctx: &SpineContext, with: impl FnOnce(K, Cadence) -> K) -> K {
    match ctx.tuning().cadence {
        Some(cadence) => with(kind, cadence),
        None => kind,
    }
}

/// Start a prepared mount (O11 i): the runtime behind [`MountKind`], the
/// limits plug-in swapped in, and the safety policy's target updated.
pub fn start_mount(
    hub: &Arc<EquipmentHub>,
    ctx: &SpineContext,
    prepared: PreparedMount,
) -> DeviceResult<super::handle::MountHandle> {
    let shared = prepared.runtime.shared();
    let initial = MountSnapshot {
        connection_id: prepared.driver.identity().connection_id,
        capabilities: shared.capabilities(),
        ..MountSnapshot::default()
    };
    let kind = apply_cadence(
        MountKind::new(prepared.runtime, hub.event_sink(), ctx.clock()),
        ctx,
        MountKind::with_cadence,
    );
    let handle = spawn_device(hub, ctx, kind, prepared.driver, initial)?;
    *lock(&ctx.mount_shared) = Some(shared);
    ctx.mount_check.set(Some(mount_limits_check(ctx)));
    ctx.targets.set_mount(Some(handle.clone()));
    Ok(handle)
}

/// Start a prepared focuser behind [`FocuserKind`] and swap in its travel
/// and A06 checks.
pub fn start_focuser(
    hub: &Arc<EquipmentHub>,
    ctx: &SpineContext,
    prepared: PreparedFocuser,
) -> DeviceResult<super::handle::FocuserHandle> {
    let initial = FocuserSnapshot {
        connection_id: prepared.driver.identity().connection_id,
        capabilities: prepared.session.capabilities().clone(),
        logical_position: prepared.session.logical_position(),
        position_source: prepared.session.position_source(),
        ..FocuserSnapshot::default()
    };
    let session = prepared.session.clone();
    let kind = apply_cadence(
        FocuserKind::new(prepared.session, ctx.clock()),
        ctx,
        FocuserKind::with_cadence,
    );
    let handle = spawn_device(hub, ctx, kind, prepared.driver, initial)?;
    *lock(&ctx.focuser_session) = Some(session);
    ctx.focuser_check.set(Some(Arc::new(FocuserSafetyCheck::new(
        &ctx.configs.focus_safety,
    ))));
    Ok(handle)
}

/// Start a prepared wheel behind [`FilterWheelKind`], build its handshake
/// (W-c) with gate-issued step permits (O11 b) and hub-assigned resume
/// blocks (O11 a), and swap in the filter check.
pub fn start_wheel(
    hub: &Arc<EquipmentHub>,
    ctx: &SpineContext,
    prepared: PreparedWheel,
) -> DeviceResult<super::handle::FilterWheelHandle> {
    let settings = FilterSettings::from_configs(&ctx.configs);
    let initial = FilterWheelSnapshot {
        connection_id: prepared.driver.identity().connection_id,
        slots: super::filter::read_wheel(&prepared.model).filter_slots(),
        ..FilterWheelSnapshot::default()
    };
    let kind = apply_cadence(
        FilterWheelKind::new(settings.clone(), prepared.model.clone(), ctx.clock()),
        ctx,
        FilterWheelKind::with_cadence,
    );
    let handle = spawn_device(hub, ctx, kind, prepared.driver, initial)?;
    let services = HandshakeServices {
        clock: ctx.clock(),
        timer: Arc::new(TokioTimer),
        events: Arc::new(HubFilterSink {
            hub: Arc::clone(hub),
        }),
        recorder: hub.recorder(),
        child_checks: Arc::clone(hub.gate()) as Arc<dyn super::filter::change::ChildCommandCheck>,
    };
    let changer = Arc::new(FilterChanger::new(
        handle.clone(),
        prepared.model,
        settings,
        services,
    ));
    ctx.filter_check.set(Some(Arc::new(changer.safety_check())));
    *lock(&ctx.filter) = Some(changer);
    Ok(handle)
}

/// Start a prepared weather device and its `environment_log` writer.
pub fn start_weather(
    hub: &Arc<EquipmentHub>,
    ctx: &SpineContext,
    prepared: PreparedSensor<dyn WeatherDriver>,
) -> DeviceResult<super::handle::WeatherHandle> {
    let initial = WeatherSnapshot {
        connection_id: prepared.driver.identity().connection_id,
        ..WeatherSnapshot::default()
    };
    let kind = apply_cadence(
        WeatherKind::new(ctx.clock()),
        ctx,
        WeatherKind::with_cadence,
    );
    let handle = spawn_device(hub, ctx, kind, prepared.driver, initial)?;
    if tokio::runtime::Handle::try_current().is_ok() {
        let ports = Arc::new(HubSafetyPorts {
            hub: Arc::clone(hub),
        });
        let logger = super::safety::WeatherLogger::new(
            &ctx.configs.weather,
            hub.recorder(),
            ports as Arc<dyn SafetyEventSink>,
        );
        let follow = logger.follow(handle.clone(), ctx.clock());
        ctx.adopt_task(tokio::spawn(follow));
    }
    Ok(handle)
}

/// Start a prepared safety monitor and add it to the supervisor.
pub fn start_safety(
    hub: &Arc<EquipmentHub>,
    ctx: &SpineContext,
    prepared: PreparedSensor<dyn SafetyDriver>,
) -> DeviceResult<super::handle::SafetyMonitorHandle> {
    let initial = SafetySnapshot {
        connection_id: prepared.driver.identity().connection_id,
        ..SafetySnapshot::default()
    };
    let kind = apply_cadence(SafetyKind::new(ctx.clock()), ctx, SafetyKind::with_cadence);
    let handle = spawn_device(hub, ctx, kind, prepared.driver, initial)?;
    ctx.ensure_safety(hub).add_monitor(handle.clone());
    Ok(handle)
}

/// `device_connected` capabilities and identity for a `/connect` response.
pub fn identity_json(identity: &DeviceIdentity) -> Value {
    json!({
        "kind": identity.kind,
        "driver": identity.driver,
        "identity_key": identity.identity_key,
        "connection_id": identity.connection_id,
        "name": identity.name,
        "description": identity.description,
        "driver_info": identity.driver_info,
        "driver_version": identity.driver_version,
        "interface_version": identity.interface_version,
        "unique_id": identity.unique_id,
        "setup_url": identity.setup_url,
        "qualification": identity.qualification,
        "endpoint": identity.endpoint.describe(),
    })
}

/// Record the last-used device per kind (§2.6 "Last-used device").
pub fn record_last_used(hub: &EquipmentHub, identity: &DeviceIdentity, option: &str) {
    hub.recorder().record(GroundTruthRecord::Profile(
        super::profile::patch::last_used(identity, option, hub.clock().now_utc()),
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::commands::{FilterWheelCmd, GatedOp};
    use crate::equipment::filter::change::ChildCommandCheck;
    use crate::equipment::gate::StepOwner;
    use crate::equipment::hub::HubConfig;
    use crate::equipment::snapshot::ResumeBlockKind;
    use crate::equipment::testing::fake_handles::RecordingSink;
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::types::{AutonomyMode, ErrorClass, Priority};

    fn hub() -> Arc<EquipmentHub> {
        Arc::new(EquipmentHub::new(
            HubConfig::default(),
            Arc::new(ManualClock::new()),
            RecordingSink::new(),
        ))
    }

    #[test]
    fn public_driver_ids_route_per_d1_o3_and_o4() {
        use DeviceKind::*;
        assert_eq!(resolve_driver(Mount, "alpaca"), Ok(DriverChoice::Alpaca));
        assert_eq!(resolve_driver(Mount, "Simulator"), Ok(DriverChoice::Sim));
        assert_eq!(resolve_driver(Focuser, "sim"), Ok(DriverChoice::Sim));
        assert_eq!(resolve_driver(Guider, "phd2"), Ok(DriverChoice::Phd2));
        // INDI reaches the OLS indi:remote mount only where compiled (O3).
        let indi = resolve_driver(Mount, "indi");
        if ols_indi_compiled() {
            assert_eq!(indi, Ok(DriverChoice::OlsIndi));
        } else {
            assert_eq!(indi.unwrap_err().status, 501);
        }
        // No native LX200 or MoonLite (O4); Tier 2 kinds are 501; unknown
        // ids are 400.
        for (kind, id) in [
            (Mount, "lx200"),
            (Focuser, "moonlite"),
            (Focuser, "indi"),
            (Guider, "direct"),
            (Rotator, "alpaca"),
            (Dome, "sim"),
        ] {
            assert_eq!(resolve_driver(kind, id).unwrap_err().status, 501, "{id}");
        }
        assert_eq!(resolve_driver(Mount, "nonsense").unwrap_err().status, 400);
    }

    /// O2: with the OLS Alpaca mount and focuser deleted, every public
    /// driver id an existing setup can name still resolves, the Alpaca ids
    /// to the native client (D1), and the removed hidden id `ols_alpaca` is
    /// refused with its replacement named, never routed to another driver.
    #[test]
    fn every_public_driver_id_still_resolves_and_ols_alpaca_names_its_replacement() {
        use crate::equipment::types::kind_name;
        use DeviceKind::*;
        for kind in [Mount, Focuser, FilterWheel, Weather, SafetyMonitor] {
            for (id, choice) in [
                ("alpaca", DriverChoice::Alpaca),
                (" Alpaca ", DriverChoice::Alpaca),
                ("sim", DriverChoice::Sim),
                ("simulator", DriverChoice::Sim),
            ] {
                assert_eq!(resolve_driver(kind, id), Ok(choice), "{kind:?} {id:?}");
            }
        }
        assert_eq!(resolve_driver(Guider, "phd2"), Ok(DriverChoice::Phd2));
        for id in ["indi", "ols_indi"] {
            match resolve_driver(Mount, id) {
                Ok(choice) => {
                    assert!(ols_indi_compiled(), "{id}");
                    assert_eq!(choice, DriverChoice::OlsIndi);
                }
                Err(refusal) => {
                    assert!(!ols_indi_compiled(), "{id}");
                    assert_eq!(
                        (refusal.status, refusal.code),
                        (501, "driver_not_available")
                    );
                }
            }
        }
        for kind in [Mount, Focuser] {
            for id in ["ols_alpaca", "OLS_ALPACA"] {
                let refusal = resolve_driver(kind, id).unwrap_err();
                assert_eq!(
                    (refusal.status, refusal.code),
                    (400, "driver_retired"),
                    "{id}"
                );
                assert!(
                    refusal.message.contains("use driver 'alpaca'")
                        && refusal.message.contains(kind_name(kind)),
                    "{}",
                    refusal.message
                );
            }
        }
        // It was never an id of any other kind.
        assert_eq!(
            resolve_driver(FilterWheel, "ols_alpaca").unwrap_err().code,
            "unsupported_driver"
        );
    }

    #[test]
    fn the_filter_layer_gets_hub_assigned_resume_block_ids() {
        // O11 a: the hub assigns the id, and an already active block of the
        // kind comes back unchanged.
        let hub = hub();
        let sink = HubFilterSink {
            hub: Arc::clone(&hub),
        };
        let first = sink.raise_resume_block(HubBlockRequest::new(
            ResumeBlockKind::FilterHandshakeIncomplete,
            "timeout",
        ));
        let again = sink.raise_resume_block(HubBlockRequest::new(
            ResumeBlockKind::FilterHandshakeIncomplete,
            "wrong slot",
        ));
        assert_eq!(first.id, again.id);
        assert_eq!(hub.resume().active(), vec![first]);
    }

    #[test]
    fn handshake_steps_get_gate_permits_that_an_abort_supersedes() {
        // O11 b: the gate issues the step permit (stamped, so the worker can
        // validate it) and refuses motion steps once the device was aborted
        // after the parent was authorised.
        let hub = hub();
        let gate = Arc::clone(hub.gate());
        let parent_cmd = GatedCommand::new(
            GatedOp::FilterHandshake {
                target: crate::equipment::commands::FilterTarget::Slot(1),
            },
            CommandSource::Operator { overriding: false },
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        let parent = gate.authorize(&parent_cmd).unwrap();
        let owner = StepOwner::of(&parent);
        let step = || {
            GatedCommand::new(
                GatedOp::FilterWheel(FilterWheelCmd::SetPosition(1)),
                CommandSource::Operator { overriding: false },
                AutonomyMode::Manual,
                Priority::Exposure,
            )
        };
        let checks: &dyn ChildCommandCheck = gate.as_ref();
        let permit = checks.child_permit(&owner, &step()).unwrap();
        assert!(gate
            .abort_epochs()
            .issued_epoch(permit.command_id(), DeviceKind::FilterWheel)
            .is_some());
        drop(permit);
        gate.abort_epochs().bump(DeviceKind::FilterWheel);
        let err = checks.child_permit(&owner, &step()).unwrap_err();
        assert_eq!(err.to_device_error().class, ErrorClass::InvalidOperation);
    }

    #[test]
    fn slot_checks_follow_the_connected_device() {
        struct Refuse;
        impl SafetyCheck for Refuse {
            fn name(&self) -> &'static str {
                "refuse"
            }
            fn applies_to(&self, _cmd: &GatedCommand) -> bool {
                true
            }
            fn check(
                &self,
                _cmd: &GatedCommand,
                _snap: &EquipmentSnapshot,
            ) -> Result<(), super::super::commands::SafetyRejection> {
                Err(super::super::commands::SafetyRejection::new(
                    "refuse", "always",
                ))
            }
        }
        let slot = SlotCheck::new("slot");
        let cmd = GatedCommand::new(
            GatedOp::FilterWheel(FilterWheelCmd::SetPosition(0)),
            CommandSource::Operator { overriding: false },
            AutonomyMode::Manual,
            Priority::Exposure,
        );
        assert!(!slot.applies_to(&cmd), "an empty slot checks nothing");
        slot.set(Some(Arc::new(Refuse)));
        assert!(slot.applies_to(&cmd));
        assert_eq!(slot.name(), "refuse");
        assert!(slot.check(&cmd, &EquipmentSnapshot::default()).is_err());
        slot.set(None);
        assert!(slot.check(&cmd, &EquipmentSnapshot::default()).is_ok());
    }

    #[test]
    fn the_inert_context_persists_nothing() {
        let hub = EquipmentHub::empty();
        let ctx = SpineContext::inert(&hub);
        assert!(ctx.data_dir.is_none() && ctx.profiles().is_none());
        assert!((1..=65_535).contains(&ctx.client_id().get()));
        assert!(!ctx.configs.equipment.allow_broadcast_discovery);
        // The device checks are registered once on the gate.
        let names = hub.gate().check_names();
        for name in [
            crate::equipment::mount::limits::CHECK_NAME,
            "focuser_travel",
            "filter_wheel",
        ] {
            assert!(names.contains(&name), "{name}: {names:?}");
        }
    }
}

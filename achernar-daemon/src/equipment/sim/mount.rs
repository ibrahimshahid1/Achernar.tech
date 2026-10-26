//! Kinematic simulator mount (§5.3 WP1A-4), also selectable as driver
//! `sim`. Owner: WP1A-4.
//!
//! The simulator moves on the injected [`Clock`], so tests drive it with a
//! `ManualClock` and never sleep. It keeps a mechanical hour angle and
//! declination: tracking advances the hour angle at the sidereal rate (RA
//! constant); with tracking off the hour angle is fixed and RA drifts with
//! LST. GoTo, park and home slew both axes at `slew_rate_deg_s`.
//!
//! It follows the ASCOM rules a driver must respect: slews while parked fail
//! with InvalidWhileParked, a Platform 7 equatorial slew with tracking off
//! fails with InvalidOperation, an Alt/Az slew needs tracking off, and Can*
//! flags gate their members. Profiles:
//!
//! - [`SimMountProfile::esp32_like`]: the user's ESP32 Alpaca mount (spec
//!   §9.1 item 3): InterfaceVersion 4 with DeviceState, Polar, Topocentric,
//!   a constant SideOfPier of 0 (east), park, no home, settable guide rates,
//!   TrackingRates 0..3, and the ASCOM-compliant CanSlew = false with
//!   CanSlewAsync = true (O10).
//! - [`SimMountProfile::german_equatorial`]: a Platform 6 German equatorial
//!   mount (InterfaceVersion 3, per-member reads) whose pier side follows
//!   the hour angle, with home and settable pier side.
//!
//! [`SimMountControl`] injects faults and acts as "another client" (hand
//! controller slews, external tracking changes).

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use chrono::{DateTime, Utc};

use crate::equipment::mount::limits::{alt_az, ha_dec_from_alt_az, hour_angle_h, lst_hours};
use crate::equipment::mount::{native_frame_policy, MountDriverOptions, SIDEREAL_PER_SI};
use crate::equipment::snapshot::{AxisRates, MountCapabilities, MountState, RateRange};
use crate::equipment::traits::{
    AbortPath, Clock, ConnectPath, ConnectReport, Device, MountDriver, PollMethod, PollPlan,
    RawPoll,
};
use crate::equipment::types::{
    identity_key, AlignmentMode, AltAz, DeviceError, DeviceIdentity, DeviceKind, DeviceResult,
    DriveRate, DriverId, Endpoint, EqCoord, ErrorClass, GuideDirection, Observed, PierSide, Site,
    SiteSource, Source,
};

/// How the simulator reports SideOfPier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimPierSide {
    /// A constant value, as the ESP32 firmware reports.
    Static(PierSide),
    /// East when the hour angle at the end of a slew is >= 0, else west.
    GermanEquatorial,
    /// `sideofpier` is not implemented.
    Unsupported,
}

/// A simulated mount model.
#[derive(Debug, Clone, PartialEq)]
pub struct SimMountProfile {
    pub name: String,
    pub interface_version: i32,
    /// Platform 7 DeviceState.
    pub device_state: bool,
    pub alignment_mode: AlignmentMode,
    /// ASCOM `EquatorialSystem`; `None` = not implemented.
    pub equatorial_system: Option<i32>,
    pub does_refraction: Option<bool>,
    pub can_park: bool,
    pub can_unpark: bool,
    pub can_set_park: bool,
    pub can_find_home: bool,
    pub can_slew: bool,
    pub can_slew_async: bool,
    pub can_slew_altaz: bool,
    pub can_slew_altaz_async: bool,
    pub can_sync: bool,
    pub can_set_tracking: bool,
    pub can_set_pier_side: bool,
    pub can_pulse_guide: bool,
    pub can_set_guide_rates: bool,
    pub can_move_axis: [bool; 3],
    /// AxisRates of every movable axis, deg/s.
    pub axis_rates: Vec<RateRange>,
    pub tracking_rates: Vec<DriveRate>,
    pub pier_side: SimPierSide,
    pub reports_alt_az: bool,
    pub reports_sidereal_time: bool,
    pub reports_at_home: bool,
    pub reports_pulse_guiding: bool,
    pub destination_side_supported: bool,
    /// The mount's own site; `None` = site members not implemented.
    pub site: Option<Site>,
    /// Mount clock minus host clock, seconds.
    pub clock_offset_s: f64,
    /// RA and Dec guide rates, deg/s; `None` = not implemented.
    pub guide_rate_deg_s: Option<(f64, f64)>,
    pub slew_rate_deg_s: f64,
    /// Park position (hour angle h, Dec deg).
    pub park_position: (f64, f64),
    pub home_position: (f64, f64),
    pub start_parked: bool,
    pub start_tracking: bool,
    /// Start position (hour angle h, Dec deg).
    pub start_position: (f64, f64),
    pub focal_length_mm: Option<f64>,
    pub aperture_mm: Option<f64>,
}

fn test_site() -> Site {
    Site {
        lat_deg: 40.0,
        lon_deg: -75.0,
        elev_m: 100.0,
        source: SiteSource::Mount,
    }
}

impl SimMountProfile {
    /// The ESP32 Alpaca mount profile (spec §9.1 item 3, O10).
    pub fn esp32_like() -> Self {
        let half_sidereal = 0.5 * crate::equipment::mount::SIDEREAL_RATE_ARCSEC_S / 3600.0;
        Self {
            name: "ESP32 Alpaca Mount (simulated)".into(),
            interface_version: 4,
            device_state: true,
            alignment_mode: AlignmentMode::Polar,
            equatorial_system: Some(1),
            does_refraction: Some(false),
            can_park: true,
            can_unpark: true,
            can_set_park: false,
            can_find_home: false,
            can_slew: false,
            can_slew_async: true,
            can_slew_altaz: false,
            can_slew_altaz_async: false,
            can_sync: true,
            can_set_tracking: true,
            can_set_pier_side: false,
            can_pulse_guide: true,
            can_set_guide_rates: true,
            can_move_axis: [true, true, false],
            axis_rates: vec![RateRange {
                min_deg_s: 0.0,
                max_deg_s: 4.0,
            }],
            tracking_rates: vec![
                DriveRate::Sidereal,
                DriveRate::Lunar,
                DriveRate::Solar,
                DriveRate::King,
            ],
            pier_side: SimPierSide::Static(PierSide::East),
            reports_alt_az: true,
            reports_sidereal_time: true,
            reports_at_home: true,
            reports_pulse_guiding: true,
            destination_side_supported: false,
            site: Some(test_site()),
            clock_offset_s: 0.0,
            guide_rate_deg_s: Some((half_sidereal, half_sidereal)),
            slew_rate_deg_s: 3.0,
            park_position: (0.0, 89.0),
            home_position: (0.0, 89.0),
            start_parked: false,
            start_tracking: true,
            start_position: (0.5, 30.0),
            focal_length_mm: None,
            aperture_mm: None,
        }
    }

    /// A Platform 6 German equatorial mount with per-member reads.
    pub fn german_equatorial() -> Self {
        Self {
            name: "German Equatorial Mount (simulated)".into(),
            interface_version: 3,
            device_state: false,
            alignment_mode: AlignmentMode::GermanPolar,
            can_set_park: true,
            can_find_home: true,
            can_slew: true,
            can_slew_altaz: true,
            can_slew_altaz_async: true,
            can_set_pier_side: true,
            can_set_guide_rates: false,
            axis_rates: vec![RateRange {
                min_deg_s: 0.0,
                max_deg_s: 2.0,
            }],
            pier_side: SimPierSide::GermanEquatorial,
            destination_side_supported: true,
            focal_length_mm: Some(800.0),
            aperture_mm: Some(200.0),
            ..Self::esp32_like()
        }
    }

    /// The ESP32 profile at InterfaceVersion 3: no DeviceState, Platform 6
    /// connect.
    pub fn platform6() -> Self {
        Self {
            interface_version: 3,
            device_state: false,
            ..Self::esp32_like()
        }
    }
}

/// Faults a test can inject.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SimFaults {
    /// Park motion completes but AtPark never reads true.
    pub park_never_sets_atpark: bool,
    pub unpark_never_clears_atpark: bool,
    /// GoTo never arrives (Slewing stays true).
    pub slew_hangs: bool,
    /// IsPulseGuiding never reads true (a pulse completes instantly).
    pub instant_pulse: bool,
    /// SyncToCoordinates is accepted but does not change the position.
    pub sync_ignored: bool,
    /// DestinationSideOfPier returns pierUnknown.
    pub destination_unknown: bool,
    /// The next N polls fail with a transport error.
    pub fail_polls: u32,
    /// Tracking writes are accepted but ignored.
    pub tracking_writes_ignored: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum MotionKind {
    Goto,
    AltAz,
    Park,
    Home,
    Flip(PierSide),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum MotionTarget {
    /// Follows the sky: the target hour angle grows with LST.
    RaDec { ra_h: f64, dec_deg: f64 },
    /// Fixed in the mount frame.
    HaDec { ha_h: f64, dec_deg: f64 },
    /// A timed motion that ends where it started (a pier flip).
    Timed { remaining_s: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Motion {
    kind: MotionKind,
    target: MotionTarget,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Pulse {
    dir: GuideDirection,
    until_mono_ns: u64,
}

/// One cached poll.
#[derive(Debug, Clone, PartialEq)]
struct Sample {
    at: DateTime<Utc>,
    mono_ns: u64,
    ra_h: f64,
    dec_deg: f64,
    altaz: Option<AltAz>,
    lst_h: f64,
    utc: DateTime<Utc>,
    slewing: bool,
    tracking: bool,
    rate: DriveRate,
    at_park: bool,
    at_home: bool,
    side: Option<PierSide>,
    pulse_guiding: bool,
    guide_rates: Option<(f64, f64)>,
    site: Option<Site>,
    target: Option<(f64, f64)>,
}

#[derive(Debug)]
struct Core {
    profile: SimMountProfile,
    faults: SimFaults,
    connected: bool,
    ha_h: f64,
    dec_deg: f64,
    tracking: bool,
    rate: DriveRate,
    at_park: bool,
    at_home: bool,
    side: PierSide,
    motion: Option<Motion>,
    axis_rates: [f64; 3],
    pulse: Option<Pulse>,
    site: Option<Site>,
    clock_offset_s: f64,
    guide_rates: Option<(f64, f64)>,
    park_position: (f64, f64),
    target: Option<(f64, f64)>,
    last_mono_ns: u64,
    calls: Vec<String>,
}

fn wrap_ha(ha: f64) -> f64 {
    let h = ha.rem_euclid(24.0);
    if h > 12.0 {
        h - 24.0
    } else {
        h
    }
}

fn not_implemented(member: &str) -> DeviceError {
    DeviceError::with_code(
        ErrorClass::Unsupported,
        crate::equipment::types::ascom_error::NOT_IMPLEMENTED,
        format!("{member} is not implemented by this simulated mount"),
    )
}

fn parked() -> DeviceError {
    DeviceError::with_code(
        ErrorClass::InvalidWhileParked,
        crate::equipment::types::ascom_error::INVALID_WHILE_PARKED,
        "the simulated mount is parked",
    )
}

impl Core {
    fn new(profile: SimMountProfile, now_mono_ns: u64) -> Self {
        let side = match profile.pier_side {
            SimPierSide::Static(side) => side,
            SimPierSide::GermanEquatorial => {
                if profile.start_position.0 >= 0.0 {
                    PierSide::East
                } else {
                    PierSide::West
                }
            }
            SimPierSide::Unsupported => PierSide::Unknown,
        };
        let (ha_h, dec_deg) = if profile.start_parked {
            profile.park_position
        } else {
            profile.start_position
        };
        Self {
            faults: SimFaults::default(),
            connected: false,
            ha_h,
            dec_deg,
            tracking: profile.start_tracking && !profile.start_parked,
            rate: DriveRate::Sidereal,
            at_park: profile.start_parked,
            at_home: false,
            side,
            motion: None,
            axis_rates: [0.0; 3],
            pulse: None,
            site: profile.site,
            clock_offset_s: profile.clock_offset_s,
            guide_rates: profile.guide_rate_deg_s,
            park_position: profile.park_position,
            target: None,
            last_mono_ns: now_mono_ns,
            calls: Vec::new(),
            profile,
        }
    }

    fn lon(&self) -> f64 {
        self.site.map(|s| s.lon_deg).unwrap_or(0.0)
    }

    fn lst(&self, utc: DateTime<Utc>) -> f64 {
        lst_hours(utc, self.lon())
    }

    fn ra(&self, utc: DateTime<Utc>) -> f64 {
        EqCoord::wrap_ra_h(self.lst(utc) - self.ha_h)
    }

    fn slewing(&self) -> bool {
        self.motion.is_some() || self.axis_rates.iter().any(|r| *r != 0.0)
    }

    fn pulse_guiding(&self, now_mono_ns: u64) -> bool {
        self.pulse.is_some_and(|p| now_mono_ns < p.until_mono_ns)
    }

    fn advance(&mut self, clock: &dyn Clock) {
        let now_mono = clock.now_mono_ns();
        let now_utc = clock.now_utc();
        let dt = now_mono.saturating_sub(self.last_mono_ns) as f64 / 1e9;
        self.last_mono_ns = now_mono;
        if dt <= 0.0 {
            return;
        }
        if self.tracking && !self.at_park {
            self.ha_h = wrap_ha(self.ha_h + dt * SIDEREAL_PER_SI / 3600.0);
        }
        self.ha_h = wrap_ha(self.ha_h - self.axis_rates[0] / 15.0 * dt);
        self.dec_deg = (self.dec_deg + self.axis_rates[1] * dt).clamp(-90.0, 90.0);
        if let Some(pulse) = self.pulse {
            let pulse_end = pulse.until_mono_ns.min(now_mono);
            let start = now_mono.saturating_sub((dt * 1e9) as u64);
            let pulse_dt = pulse_end.saturating_sub(start) as f64 / 1e9;
            let (ra_rate, dec_rate) = self.guide_rates.unwrap_or((0.002, 0.002));
            match pulse.dir {
                GuideDirection::North => self.dec_deg += dec_rate * pulse_dt,
                GuideDirection::South => self.dec_deg -= dec_rate * pulse_dt,
                GuideDirection::East => self.ha_h -= ra_rate / 15.0 * pulse_dt,
                GuideDirection::West => self.ha_h += ra_rate / 15.0 * pulse_dt,
            }
            if now_mono >= pulse.until_mono_ns {
                self.pulse = None;
            }
        }
        let Some(mut motion) = self.motion else {
            return;
        };
        if self.faults.slew_hangs && motion.kind == MotionKind::Goto {
            return;
        }
        let step_deg = self.profile.slew_rate_deg_s.max(0.01) * dt;
        let arrived = match &mut motion.target {
            MotionTarget::Timed { remaining_s } => {
                *remaining_s -= dt;
                *remaining_s <= 0.0
            }
            target => {
                let (target_ha, target_dec) = match *target {
                    MotionTarget::RaDec { ra_h, dec_deg } => {
                        (hour_angle_h(self.lst(now_utc), ra_h), dec_deg)
                    }
                    MotionTarget::HaDec { ha_h, dec_deg } => (ha_h, dec_deg),
                    MotionTarget::Timed { .. } => unreachable!(),
                };
                let dha_deg = wrap_ha(target_ha - self.ha_h) * 15.0;
                let ddec = target_dec - self.dec_deg;
                let ha_done = dha_deg.abs() <= step_deg;
                let dec_done = ddec.abs() <= step_deg;
                self.ha_h = if ha_done {
                    target_ha
                } else {
                    wrap_ha(self.ha_h + step_deg.copysign(dha_deg) / 15.0)
                };
                self.dec_deg = if dec_done {
                    target_dec
                } else {
                    self.dec_deg + step_deg.copysign(ddec)
                };
                ha_done && dec_done
            }
        };
        if !arrived {
            self.motion = Some(motion);
            return;
        }
        self.motion = None;
        match motion.kind {
            MotionKind::Goto | MotionKind::AltAz => {
                if self.profile.pier_side == SimPierSide::GermanEquatorial {
                    self.side = if self.ha_h >= 0.0 {
                        PierSide::East
                    } else {
                        PierSide::West
                    };
                }
            }
            MotionKind::Park => {
                self.tracking = false;
                if !self.faults.park_never_sets_atpark {
                    self.at_park = true;
                }
            }
            MotionKind::Home => self.at_home = true,
            MotionKind::Flip(side) => self.side = side,
        }
    }

    fn sample(&self, clock: &dyn Clock) -> Sample {
        let at = clock.now_utc();
        let mono_ns = clock.now_mono_ns();
        let lst_h = self.lst(at);
        let altaz = self
            .site
            .filter(|_| self.profile.reports_alt_az)
            .map(|site| alt_az(self.ha_h, self.dec_deg, site.lat_deg));
        Sample {
            at,
            mono_ns,
            ra_h: self.ra(at),
            dec_deg: self.dec_deg,
            altaz,
            lst_h,
            utc: at + chrono::Duration::milliseconds((self.clock_offset_s * 1000.0) as i64),
            slewing: self.slewing(),
            tracking: self.tracking,
            rate: self.rate,
            at_park: self.at_park,
            at_home: self.at_home,
            side: match self.profile.pier_side {
                SimPierSide::Unsupported => None,
                _ => Some(self.side),
            },
            pulse_guiding: self.pulse_guiding(mono_ns),
            guide_rates: self.guide_rates,
            site: self.site,
            target: self.target,
        }
    }

    fn stop(&mut self) {
        self.motion = None;
        self.axis_rates = [0.0; 3];
    }
}

fn lock(core: &Mutex<Core>) -> MutexGuard<'_, Core> {
    core.lock().unwrap_or_else(|e| e.into_inner())
}

/// The simulator's out-of-band stop: AbortSlew plus MoveAxis 0 (D4).
pub struct SimMountAbort {
    core: Arc<Mutex<Core>>,
    clock: Arc<dyn Clock>,
}

impl AbortPath for SimMountAbort {
    fn abort(&self) -> DeviceResult<()> {
        let mut core = lock(&self.core);
        core.advance(self.clock.as_ref());
        core.calls.push("abort_path".into());
        core.stop();
        Ok(())
    }
}

/// Test and server handle: fault injection and "other client" actions.
#[derive(Clone)]
pub struct SimMountControl {
    core: Arc<Mutex<Core>>,
    clock: Arc<dyn Clock>,
}

impl SimMountControl {
    pub fn set_faults(&self, faults: SimFaults) {
        lock(&self.core).faults = faults;
    }

    pub fn update_faults(&self, f: impl FnOnce(&mut SimFaults)) {
        f(&mut lock(&self.core).faults);
    }

    /// A GoTo started by another client (the hand controller).
    pub fn external_goto(&self, target: EqCoord) {
        let mut core = lock(&self.core);
        core.advance(self.clock.as_ref());
        core.motion = Some(Motion {
            kind: MotionKind::Goto,
            target: MotionTarget::RaDec {
                ra_h: target.ra_h,
                dec_deg: target.dec_deg,
            },
        });
    }

    /// Another client changes tracking.
    pub fn set_tracking_external(&self, on: bool) {
        let mut core = lock(&self.core);
        core.advance(self.clock.as_ref());
        core.tracking = on;
    }

    /// Teleport the mount (no slew).
    pub fn place(&self, coord: EqCoord) {
        let mut core = lock(&self.core);
        core.advance(self.clock.as_ref());
        let lst = core.lst(self.clock.now_utc());
        core.ha_h = hour_angle_h(lst, coord.ra_h);
        core.dec_deg = coord.dec_deg;
    }

    pub fn slewing(&self) -> bool {
        let mut core = lock(&self.core);
        core.advance(self.clock.as_ref());
        core.slewing()
    }

    pub fn tracking(&self) -> bool {
        lock(&self.core).tracking
    }

    pub fn axis_rates(&self) -> [f64; 3] {
        lock(&self.core).axis_rates
    }

    /// Current (RA h, Dec deg).
    pub fn position(&self) -> (f64, f64) {
        let mut core = lock(&self.core);
        core.advance(self.clock.as_ref());
        (core.ra(self.clock.now_utc()), core.dec_deg)
    }

    /// Driver and abort-path calls, in order (`"park"`, `"abort_path"`, ...).
    pub fn calls(&self) -> Vec<String> {
        lock(&self.core).calls.clone()
    }

    pub fn site(&self) -> Option<Site> {
        lock(&self.core).site
    }

    pub fn clock_offset_s(&self) -> f64 {
        lock(&self.core).clock_offset_s
    }
}

/// The simulator driver.
pub struct SimMount {
    identity: DeviceIdentity,
    core: Arc<Mutex<Core>>,
    clock: Arc<dyn Clock>,
    options: MountDriverOptions,
    abort: Arc<SimMountAbort>,
    last: Option<Sample>,
    last_poll_failed: bool,
}

impl SimMount {
    pub fn new(profile: SimMountProfile, clock: Arc<dyn Clock>) -> Self {
        let mut identity = DeviceIdentity::new(
            DeviceKind::Mount,
            DriverId::Sim,
            Endpoint::InProcess {
                name: "sim-mount".into(),
            },
            identity_key::sim(DeviceKind::Mount),
            profile.name.clone(),
        );
        identity.interface_version = Some(profile.interface_version);
        identity.description = Some("Achernar kinematic mount simulator".into());
        identity.driver_info = Some("achernar sim/mount.rs".into());
        let core = Arc::new(Mutex::new(Core::new(profile, clock.now_mono_ns())));
        let abort = Arc::new(SimMountAbort {
            core: Arc::clone(&core),
            clock: Arc::clone(&clock),
        });
        Self {
            identity,
            core,
            clock,
            options: MountDriverOptions::default(),
            abort,
            last: None,
            last_poll_failed: false,
        }
    }

    pub fn with_options(mut self, options: MountDriverOptions) -> Self {
        self.options = options;
        self
    }

    pub fn control(&self) -> SimMountControl {
        SimMountControl {
            core: Arc::clone(&self.core),
            clock: Arc::clone(&self.clock),
        }
    }

    fn core(&self) -> MutexGuard<'_, Core> {
        let mut core = lock(&self.core);
        core.advance(self.clock.as_ref());
        core
    }

    /// Lock the core for a command: connected, advanced, logged.
    fn command(&self, name: &str) -> DeviceResult<MutexGuard<'_, Core>> {
        let mut core = self.core();
        if !core.connected {
            return Err(DeviceError::not_connected(
                "the simulated mount is not connected",
            ));
        }
        core.calls.push(name.to_string());
        Ok(core)
    }

    fn source(&self) -> Source {
        if lock(&self.core).profile.device_state {
            Source::DeviceState
        } else {
            Source::Property
        }
    }
}

fn obs<T>(value: T, sample: &Sample, source: Source) -> Observed<T> {
    Observed::fresh(value, source, sample.at, sample.mono_ns)
}

impl Device for SimMount {
    fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    fn connect(&mut self, _timeout: Duration) -> DeviceResult<ConnectReport> {
        let mut core = self.core();
        core.connected = true;
        core.calls.push("connect".into());
        let profile = &core.profile;
        let mut unsupported = Vec::new();
        if !profile.device_state {
            unsupported.push("devicestate".to_string());
        }
        if !profile.destination_side_supported {
            unsupported.push("destinationsideofpier".to_string());
        }
        Ok(ConnectReport {
            path: if profile.interface_version >= 4 {
                ConnectPath::Platform7
            } else {
                ConnectPath::Platform6
            },
            interface_version: Some(profile.interface_version),
            device_state: profile.device_state,
            unsupported_members: unsupported,
            warnings: Vec::new(),
            duration_ms: 0,
        })
    }

    fn disconnect(&mut self) -> DeviceResult<()> {
        let mut core = self.core();
        core.connected = false;
        core.calls.push("disconnect".into());
        Ok(())
    }

    fn is_connected(&mut self) -> DeviceResult<bool> {
        Ok(self.core().connected)
    }

    fn poll(&mut self, _plan: &PollPlan) -> DeviceResult<RawPoll> {
        let started = self.clock.now_mono_ns();
        let mut core = self.core();
        if !core.connected {
            return Err(DeviceError::not_connected(
                "the simulated mount is not connected",
            ));
        }
        if core.faults.fail_polls > 0 {
            core.faults.fail_polls -= 1;
            drop(core);
            self.last_poll_failed = true;
            return Err(DeviceError::transport("simulated poll failure"));
        }
        let sample = core.sample(self.clock.as_ref());
        let device_state = core.profile.device_state;
        drop(core);
        let device_time = device_state.then_some(sample.utc);
        self.last = Some(sample);
        self.last_poll_failed = false;
        Ok(RawPoll {
            method: if device_state {
                PollMethod::DeviceState
            } else {
                PollMethod::InProcess
            },
            started_mono_ns: started,
            finished_mono_ns: self.clock.now_mono_ns(),
            device_time,
            members: Vec::new(),
        })
    }

    fn supported_actions(&mut self) -> DeviceResult<Vec<String>> {
        Ok(Vec::new())
    }

    fn action(&mut self, name: &str, _params: &str) -> DeviceResult<String> {
        Err(DeviceError::with_code(
            ErrorClass::Unsupported,
            crate::equipment::types::ascom_error::ACTION_NOT_IMPLEMENTED,
            format!("action {name} is not implemented"),
        ))
    }

    fn abort_path(&self) -> Option<Arc<dyn AbortPath>> {
        Some(Arc::clone(&self.abort) as Arc<dyn AbortPath>)
    }
}

impl MountDriver for SimMount {
    fn capabilities(&mut self) -> DeviceResult<MountCapabilities> {
        let core = self.core();
        let p = &core.profile;
        let (native_frame, frame_reason) = native_frame_policy(
            p.equatorial_system,
            Some(p.interface_version),
            &self.options,
        );
        let mut unsupported = Vec::new();
        if !p.device_state {
            unsupported.push("devicestate".to_string());
        }
        if !p.destination_side_supported {
            unsupported.push("destinationsideofpier".to_string());
        }
        Ok(MountCapabilities {
            can_park: p.can_park,
            can_unpark: p.can_unpark,
            can_set_park: p.can_set_park,
            can_find_home: p.can_find_home,
            can_slew: p.can_slew,
            can_slew_async: p.can_slew_async,
            can_slew_altaz: p.can_slew_altaz,
            can_slew_altaz_async: p.can_slew_altaz_async,
            can_sync: p.can_sync,
            can_sync_altaz: false,
            can_set_tracking: p.can_set_tracking,
            can_set_pier_side: p.can_set_pier_side,
            can_pulse_guide: p.can_pulse_guide,
            can_set_guide_rates: p.can_set_guide_rates,
            can_move_axis: p.can_move_axis,
            can_set_declination_rate: false,
            can_set_right_ascension_rate: false,
            alignment_mode: Some(p.alignment_mode),
            equatorial_system: p.equatorial_system,
            native_frame,
            frame_reason,
            does_refraction: p.does_refraction,
            tracking_rates: p.tracking_rates.clone(),
            axis_rates: (0..3u8)
                .filter(|axis| p.can_move_axis[usize::from(*axis)])
                .map(|axis| AxisRates {
                    axis,
                    ranges: p.axis_rates.clone(),
                })
                .collect(),
            focal_length_mm: p.focal_length_mm,
            aperture_diameter_mm: p.aperture_mm,
            coordinate_precision_arcsec: None,
            interface_version: Some(p.interface_version),
            device_state: p.device_state,
            unsupported_members: unsupported,
        })
    }

    fn state(&mut self) -> DeviceResult<MountState> {
        let Some(sample) = self.last.clone() else {
            return Ok(MountState::default());
        };
        let source = self.source();
        let profile = lock(&self.core).profile.clone();
        let frame = native_frame_policy(
            profile.equatorial_system,
            Some(profile.interface_version),
            &self.options,
        )
        .0;
        let mut state = MountState {
            ra_h: obs(sample.ra_h, &sample, source),
            dec_deg: obs(sample.dec_deg, &sample, source),
            alt_deg: match sample.altaz {
                Some(a) => obs(a.alt_deg, &sample, source),
                None => Observed::unsupported(source),
            },
            az_deg: match sample.altaz {
                Some(a) => obs(a.az_deg, &sample, source),
                None => Observed::unsupported(source),
            },
            sidereal_time_h: if profile.reports_sidereal_time {
                obs(sample.lst_h, &sample, source)
            } else {
                Observed::unsupported(source)
            },
            utc: obs(sample.utc, &sample, source),
            slewing: obs(sample.slewing, &sample, source),
            tracking: obs(sample.tracking, &sample, source),
            tracking_rate: obs(sample.rate, &sample, Source::Property),
            at_park: obs(sample.at_park, &sample, source),
            at_home: if profile.reports_at_home {
                obs(sample.at_home, &sample, source)
            } else {
                Observed::unsupported(source)
            },
            side_of_pier: match sample.side {
                Some(side) => obs(side, &sample, source),
                None => Observed::unsupported(source),
            },
            is_pulse_guiding: if profile.reports_pulse_guiding {
                obs(sample.pulse_guiding, &sample, source)
            } else {
                Observed::unsupported(source)
            },
            guide_rate_ra_arcsec_s: match sample.guide_rates {
                Some((ra, _)) => obs(ra * 3600.0, &sample, Source::Property),
                None => Observed::unsupported(Source::Property),
            },
            guide_rate_dec_arcsec_s: match sample.guide_rates {
                Some((_, dec)) => obs(dec * 3600.0, &sample, Source::Property),
                None => Observed::unsupported(Source::Property),
            },
            site: match sample.site {
                Some(site) => obs(
                    Site {
                        source: SiteSource::Mount,
                        ..site
                    },
                    &sample,
                    Source::Property,
                ),
                None => Observed::unsupported(Source::Property),
            },
            target: match sample.target {
                Some((ra, dec)) => obs(EqCoord::new(ra, dec, frame), &sample, Source::Property),
                None => Observed::not_set(Source::Property),
            },
        };
        if self.last_poll_failed {
            state = crate::equipment::mount::stale_state(&state);
        }
        Ok(state)
    }

    fn slew_async(&mut self, target: EqCoord) -> DeviceResult<()> {
        let mut core = self.command("slew_async")?;
        if !core.profile.can_slew_async {
            return Err(not_implemented("SlewToCoordinatesAsync"));
        }
        if core.at_park {
            return Err(parked());
        }
        if !core.tracking && core.profile.interface_version >= 4 {
            return Err(DeviceError::with_code(
                ErrorClass::InvalidOperation,
                crate::equipment::types::ascom_error::INVALID_OPERATION,
                "an equatorial slew needs tracking on",
            ));
        }
        if !(-90.0..=90.0).contains(&target.dec_deg) {
            return Err(DeviceError::invalid_value("declination out of range"));
        }
        core.target = Some((target.ra_h, target.dec_deg));
        core.at_home = false;
        core.motion = Some(Motion {
            kind: MotionKind::Goto,
            target: MotionTarget::RaDec {
                ra_h: target.ra_h,
                dec_deg: target.dec_deg,
            },
        });
        Ok(())
    }

    fn slew_altaz_async(&mut self, target: AltAz) -> DeviceResult<()> {
        let mut core = self.command("slew_altaz_async")?;
        if !core.profile.can_slew_altaz_async {
            return Err(not_implemented("SlewToAltAzAsync"));
        }
        if core.at_park {
            return Err(parked());
        }
        if core.tracking {
            return Err(DeviceError::invalid_operation(
                "an Alt/Az slew needs tracking off",
            ));
        }
        let Some(site) = core.site else {
            return Err(DeviceError::invalid_operation(
                "the simulated mount has no site",
            ));
        };
        let (ha_h, dec_deg) = ha_dec_from_alt_az(target, site.lat_deg);
        core.at_home = false;
        core.motion = Some(Motion {
            kind: MotionKind::AltAz,
            target: MotionTarget::HaDec { ha_h, dec_deg },
        });
        Ok(())
    }

    fn abort_slew(&mut self) -> DeviceResult<()> {
        let mut core = self.command("abort_slew")?;
        if core.at_park {
            return Err(parked());
        }
        core.stop();
        Ok(())
    }

    fn sync(&mut self, target: EqCoord) -> DeviceResult<()> {
        let mut core = self.command("sync")?;
        if !core.profile.can_sync {
            return Err(not_implemented("SyncToCoordinates"));
        }
        if core.at_park {
            return Err(parked());
        }
        if !core.tracking {
            return Err(DeviceError::invalid_operation("sync needs tracking on"));
        }
        if !core.faults.sync_ignored {
            let lst = core.lst(self.clock.now_utc());
            core.ha_h = hour_angle_h(lst, target.ra_h);
            core.dec_deg = target.dec_deg;
        }
        Ok(())
    }

    fn set_tracking(&mut self, on: bool) -> DeviceResult<()> {
        let mut core = self.command(if on { "tracking_on" } else { "tracking_off" })?;
        if !core.profile.can_set_tracking {
            return Err(not_implemented("Tracking (write)"));
        }
        if on && core.at_park {
            return Err(parked());
        }
        if !core.faults.tracking_writes_ignored {
            core.tracking = on;
        }
        Ok(())
    }

    fn set_tracking_rate(&mut self, rate: DriveRate) -> DeviceResult<()> {
        let mut core = self.command("tracking_rate")?;
        if !core.profile.tracking_rates.contains(&rate) {
            return Err(DeviceError::invalid_value("rate not in TrackingRates"));
        }
        core.rate = rate;
        Ok(())
    }

    fn park(&mut self) -> DeviceResult<()> {
        let mut core = self.command("park")?;
        if !core.profile.can_park {
            return Err(not_implemented("Park"));
        }
        if core.at_park {
            return Ok(());
        }
        let (ha_h, dec_deg) = core.park_position;
        core.at_home = false;
        core.axis_rates = [0.0; 3];
        core.motion = Some(Motion {
            kind: MotionKind::Park,
            target: MotionTarget::HaDec { ha_h, dec_deg },
        });
        Ok(())
    }

    fn unpark(&mut self) -> DeviceResult<()> {
        let mut core = self.command("unpark")?;
        if !core.profile.can_unpark {
            return Err(not_implemented("Unpark"));
        }
        if !core.faults.unpark_never_clears_atpark {
            core.at_park = false;
        }
        Ok(())
    }

    fn set_park(&mut self) -> DeviceResult<()> {
        let mut core = self.command("set_park")?;
        if !core.profile.can_set_park {
            return Err(not_implemented("SetPark"));
        }
        if core.at_park {
            return Err(parked());
        }
        core.park_position = (core.ha_h, core.dec_deg);
        Ok(())
    }

    fn find_home(&mut self) -> DeviceResult<()> {
        let mut core = self.command("find_home")?;
        if !core.profile.can_find_home {
            return Err(not_implemented("FindHome"));
        }
        if core.at_park {
            return Err(parked());
        }
        let (ha_h, dec_deg) = core.profile.home_position;
        core.motion = Some(Motion {
            kind: MotionKind::Home,
            target: MotionTarget::HaDec { ha_h, dec_deg },
        });
        Ok(())
    }

    fn move_axis(&mut self, axis: u8, rate_deg_s: f64) -> DeviceResult<()> {
        let mut core = self.command(&format!("move_axis:{axis}:{rate_deg_s}"))?;
        let index = usize::from(axis);
        if index > 2 || !core.profile.can_move_axis[index] {
            return Err(not_implemented("MoveAxis on this axis"));
        }
        if core.at_park && rate_deg_s != 0.0 {
            return Err(parked());
        }
        let magnitude = rate_deg_s.abs();
        let valid = magnitude == 0.0
            || core
                .profile
                .axis_rates
                .iter()
                .any(|r| magnitude >= r.min_deg_s && magnitude <= r.max_deg_s);
        if !valid {
            return Err(DeviceError::invalid_value("rate outside AxisRates"));
        }
        core.axis_rates[index] = rate_deg_s;
        Ok(())
    }

    fn pulse_guide(&mut self, dir: GuideDirection, ms: u32) -> DeviceResult<()> {
        let mut core = self.command(&format!("pulse_guide:{dir:?}:{ms}"))?;
        if !core.profile.can_pulse_guide {
            return Err(not_implemented("PulseGuide"));
        }
        if core.at_park {
            return Err(parked());
        }
        let now = self.clock.now_mono_ns();
        core.pulse = Some(Pulse {
            dir,
            until_mono_ns: if core.faults.instant_pulse {
                now
            } else {
                now + u64::from(ms) * 1_000_000
            },
        });
        Ok(())
    }

    fn set_guide_rates(&mut self, ra_deg_s: f64, dec_deg_s: f64) -> DeviceResult<()> {
        let mut core = self.command("set_guide_rates")?;
        if !core.profile.can_set_guide_rates {
            return Err(not_implemented("GuideRateRightAscension (write)"));
        }
        core.guide_rates = Some((ra_deg_s, dec_deg_s));
        Ok(())
    }

    fn set_site(&mut self, site: &Site) -> DeviceResult<()> {
        let mut core = self.command("set_site")?;
        if core.site.is_none() {
            return Err(not_implemented("SiteLatitude (write)"));
        }
        core.site = Some(Site {
            source: SiteSource::Mount,
            ..*site
        });
        Ok(())
    }

    fn set_utc(&mut self, utc: DateTime<Utc>) -> DeviceResult<()> {
        let mut core = self.command("set_utc")?;
        core.clock_offset_s = (utc - self.clock.now_utc()).num_milliseconds() as f64 / 1000.0;
        Ok(())
    }

    fn destination_side_of_pier(&mut self, target: EqCoord) -> DeviceResult<PierSide> {
        let core = self.command("destination_side_of_pier")?;
        if !core.profile.destination_side_supported {
            return Err(not_implemented("DestinationSideOfPier"));
        }
        if core.faults.destination_unknown {
            return Ok(PierSide::Unknown);
        }
        let ha = hour_angle_h(core.lst(self.clock.now_utc()), target.ra_h);
        Ok(if ha >= 0.0 {
            PierSide::East
        } else {
            PierSide::West
        })
    }

    fn set_side_of_pier(&mut self, side: PierSide) -> DeviceResult<()> {
        let mut core = self.command("set_side_of_pier")?;
        if !core.profile.can_set_pier_side {
            return Err(not_implemented("SideOfPier (write)"));
        }
        if core.at_park {
            return Err(parked());
        }
        if side == PierSide::Unknown {
            return Err(DeviceError::invalid_value("pier side must be east or west"));
        }
        core.motion = Some(Motion {
            kind: MotionKind::Flip(side),
            target: MotionTarget::Timed { remaining_s: 10.0 },
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::types::EqFrame;

    fn connected(profile: SimMountProfile) -> (Arc<ManualClock>, SimMount) {
        let clock = Arc::new(ManualClock::new());
        let mut sim = SimMount::new(profile, clock.clone());
        sim.connect(Duration::from_secs(1)).unwrap();
        (clock, sim)
    }

    fn poll(sim: &mut SimMount) -> MountState {
        sim.poll(&PollPlan::default()).unwrap();
        sim.state().unwrap()
    }

    #[test]
    fn the_esp32_profile_matches_the_firmware_facts() {
        let (_clock, mut sim) = connected(SimMountProfile::esp32_like());
        let caps = sim.capabilities().unwrap();
        assert_eq!(caps.interface_version, Some(4));
        assert!(caps.device_state);
        assert_eq!(caps.alignment_mode, Some(AlignmentMode::Polar));
        assert_eq!(caps.native_frame, EqFrame::JNowApparent);
        assert!(caps.can_park && !caps.can_find_home && !caps.can_set_pier_side);
        assert!(caps.can_set_guide_rates);
        assert!(
            !caps.can_slew && caps.can_slew_async,
            "O10: compliant Alpaca"
        );
        assert_eq!(caps.tracking_rates.len(), 4);
        let state = poll(&mut sim);
        assert_eq!(state.side_of_pier.value, Some(PierSide::East));
        assert_eq!(state.slewing.source, Source::DeviceState);
        assert!(state.alt_deg.is_fresh() && state.sidereal_time_h.is_fresh());
    }

    #[test]
    fn tracking_holds_ra_and_tracking_off_lets_it_drift() {
        let (clock, mut sim) = connected(SimMountProfile::esp32_like());
        let ra0 = poll(&mut sim).ra_h.value.unwrap();
        clock.advance(Duration::from_secs(600));
        let ra1 = poll(&mut sim).ra_h.value.unwrap();
        assert!((ra1 - ra0).abs() < 1e-6, "tracking keeps RA fixed");
        sim.set_tracking(false).unwrap();
        clock.advance(Duration::from_secs(3600));
        let ra2 = poll(&mut sim).ra_h.value.unwrap();
        let drift = EqCoord::wrap_ra_h(ra2 - ra1);
        assert!((drift - SIDEREAL_PER_SI).abs() < 1e-3, "{drift}");
    }

    #[test]
    fn gotos_move_at_the_slew_rate_and_follow_ascom_rules() {
        let (clock, mut sim) = connected(SimMountProfile::esp32_like());
        let state = poll(&mut sim);
        let target = EqCoord::new(
            EqCoord::wrap_ra_h(state.ra_h.value.unwrap() + 1.0),
            45.0,
            EqFrame::JNowApparent,
        );
        sim.slew_async(target).unwrap();
        assert_eq!(poll(&mut sim).slewing.value, Some(true));
        clock.advance(Duration::from_secs(60));
        let state = poll(&mut sim);
        assert_eq!(state.slewing.value, Some(false));
        assert!((state.dec_deg.value.unwrap() - 45.0).abs() < 1e-9);
        assert!((state.ra_h.value.unwrap() - target.ra_h).abs() < 1e-6);
        sim.set_tracking(false).unwrap();
        assert_eq!(
            sim.slew_async(target).unwrap_err().class,
            ErrorClass::InvalidOperation
        );
        sim.park().unwrap();
        clock.advance(Duration::from_secs(120));
        let state = poll(&mut sim);
        assert_eq!(state.at_park.value, Some(true));
        assert_eq!(state.tracking.value, Some(false));
        assert_eq!(
            sim.slew_async(target).unwrap_err().class,
            ErrorClass::InvalidWhileParked
        );
        assert_eq!(sim.find_home().unwrap_err().class, ErrorClass::Unsupported);
    }

    #[test]
    fn the_abort_path_stops_a_goto_and_axes_out_of_band() {
        let (clock, mut sim) = connected(SimMountProfile::esp32_like());
        sim.slew_async(EqCoord::new(1.0, -20.0, EqFrame::JNowApparent))
            .unwrap();
        clock.advance(Duration::from_secs(1));
        assert_eq!(poll(&mut sim).slewing.value, Some(true));
        sim.abort_path().unwrap().abort().unwrap();
        assert_eq!(poll(&mut sim).slewing.value, Some(false));
        sim.move_axis(0, 2.0).unwrap();
        assert_eq!(poll(&mut sim).slewing.value, Some(true));
        assert_eq!(
            sim.move_axis(0, 9.0).unwrap_err().class,
            ErrorClass::InvalidValue
        );
        sim.abort_path().unwrap().abort().unwrap();
        assert_eq!(sim.control().axis_rates(), [0.0; 3]);
        assert!(sim.control().calls().contains(&"abort_path".to_string()));
    }

    #[test]
    fn german_mounts_choose_the_pier_side_from_the_hour_angle() {
        let (clock, mut sim) = connected(SimMountProfile::german_equatorial());
        let lst = poll(&mut sim).sidereal_time_h.value.unwrap();
        let east_target = EqCoord::new(EqCoord::wrap_ra_h(lst + 2.0), 10.0, EqFrame::JNowApparent);
        assert_eq!(
            sim.destination_side_of_pier(east_target).unwrap(),
            PierSide::West
        );
        sim.slew_async(east_target).unwrap();
        clock.advance(Duration::from_secs(120));
        assert_eq!(poll(&mut sim).side_of_pier.value, Some(PierSide::West));
        sim.control()
            .update_faults(|f| f.destination_unknown = true);
        assert_eq!(
            sim.destination_side_of_pier(east_target).unwrap(),
            PierSide::Unknown
        );
    }

    #[test]
    fn failed_polls_leave_stale_values() {
        let (_clock, mut sim) = connected(SimMountProfile::platform6());
        let fresh = poll(&mut sim);
        assert!(fresh.ra_h.is_fresh());
        assert_eq!(fresh.ra_h.source, Source::Property);
        sim.control().update_faults(|f| f.fail_polls = 1);
        assert!(sim.poll(&PollPlan::default()).is_err());
        let stale = sim.state().unwrap();
        assert_eq!(stale.ra_h.state, crate::equipment::types::ObsState::Stale);
        assert_eq!(stale.ra_h.value, fresh.ra_h.value);
        assert!(poll(&mut sim).ra_h.is_fresh());
    }

    #[test]
    fn pulses_report_is_pulse_guiding_for_their_duration() {
        let (clock, mut sim) = connected(SimMountProfile::esp32_like());
        sim.pulse_guide(GuideDirection::North, 300).unwrap();
        assert_eq!(poll(&mut sim).is_pulse_guiding.value, Some(true));
        clock.advance(Duration::from_millis(300));
        assert_eq!(poll(&mut sim).is_pulse_guiding.value, Some(false));
        sim.control().update_faults(|f| f.instant_pulse = true);
        sim.pulse_guide(GuideDirection::South, 300).unwrap();
        assert_eq!(poll(&mut sim).is_pulse_guiding.value, Some(false));
    }
}

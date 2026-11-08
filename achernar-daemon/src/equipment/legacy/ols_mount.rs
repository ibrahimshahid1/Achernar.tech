//! The OLS legacy mount adapter `ols_indi` (D1, O3), run through the same
//! worker as the native drivers. Owner: WP1A-4.
//!
//! It fronts the OLS `indi:remote` mount and is deleted once the native INDI
//! client passes acceptance in Tier 2 (O3). Until then, Tier 1 routes the
//! public `indi` mount id here where `WITH_INDI` is compiled. Its Alpaca
//! twin `ols_alpaca` was deleted at the end of Tier 1, with the OLS Alpaca
//! mount itself, once the native Alpaca mount passed acceptance (O2).
//!
//! What the OLS wrapper (`D/hardware/mount.rs`) can and cannot do, per the
//! §7.1 "O" column, is published honestly instead of hidden:
//!
//! - RA/Dec (R1) with frame `Unknown` unless `equatorial_system_override`
//!   is set; tracking on/off (R4); GoTo, sync and pulses (R8, completion by
//!   duration).
//! - Slewing, AtPark, SideOfPier, Alt/Az, LST, UTC, guide rates and the
//!   tracking rate are `Unsupported`. A GoTo therefore ends `Unknown`
//!   (C-5: completion comes only from a completion property), and Alt/Az
//!   and LST are derived from the site by the snapshot builder.
//! - The tracking-rate write is refused: the OLS wrapper sets Lunar when
//!   asked for Solar.
//! - AbortSlew is attempted with OLS `stop()`, which halts manual motion but
//!   does not abort a GoTo; the abort therefore reports `Unsupported` rather
//!   than claiming success.
//! - OLS itself forces tracking on for a GoTo and applies the limits in
//!   `~/.ols_mount_config.txt`; Achernar's gate checks still run first.
//!
//! The adapter reaches OLS through [`OlsMountBackend`], implemented for
//! `hardware::mount::Mount` here and by a fake in tests. Calls take a read
//! lock (OLS locks internally, so the abort path never waits for a slow
//! read); only `connect` takes the write lock.

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Duration;

use chrono::{DateTime, Utc};

use crate::equipment::mount::{native_frame_policy, MountDriverOptions};
use crate::equipment::snapshot::{MountCapabilities, MountState};
use crate::equipment::traits::{
    AbortPath, Clock, ConnectPath, ConnectReport, Device, MountDriver, PollMethod, PollPlan,
    RawPoll,
};
use crate::equipment::types::{
    identity_key, AltAz, DeviceError, DeviceIdentity, DeviceKind, DeviceResult, DriveRate,
    DriverId, Endpoint, EqCoord, EqFrame, ErrorClass, GuideDirection, Observed, PierSide, Site,
    Source,
};

/// The OLS mount operations the adapter uses.
pub trait OlsMountBackend: Send + Sync + 'static {
    fn connect(&mut self, option: &str) -> Result<(), String>;
    fn is_connected(&self) -> bool;
    /// (RA hours, Dec degrees), mount frame.
    fn position(&self) -> Result<(f64, f64), String>;
    fn goto(&self, ra_h: f64, dec_deg: f64) -> Result<(), String>;
    fn sync(&self, ra_h: f64, dec_deg: f64) -> Result<(), String>;
    /// OLS `stop()`: halts manual motion; does not abort a GoTo.
    fn stop(&self) -> Result<(), String>;
    fn tracking(&self) -> Result<bool, String>;
    fn set_tracking(&self, on: bool) -> Result<(), String>;
    fn set_location(&self, lat_deg: f64, lon_deg: f64) -> Result<(), String>;
    fn pulse_guide_supported(&self) -> Result<bool, String>;
    /// Positive `ns_ms` is North, positive `we_ms` is West (OLS convention).
    fn pulse_guide(&self, ns_ms: f32, we_ms: f32) -> Result<(), String>;
    fn shutdown(&self) -> Result<(), String>;
}

impl OlsMountBackend for crate::hardware::mount::Mount {
    fn connect(&mut self, option: &str) -> Result<(), String> {
        self.connect_with_option(option).map_err(|e| e.to_string())
    }

    fn is_connected(&self) -> bool {
        crate::hardware::mount::Mount::is_connected(self)
    }

    fn position(&self) -> Result<(f64, f64), String> {
        self.get_position()
            .map(|p| (p.ra, p.dec))
            .map_err(|e| e.to_string())
    }

    fn goto(&self, ra_h: f64, dec_deg: f64) -> Result<(), String> {
        crate::hardware::mount::Mount::goto(
            self,
            crate::hardware::mount::EqCoord {
                ra: ra_h,
                dec: dec_deg,
            },
        )
        .map_err(|e| e.to_string())
    }

    fn sync(&self, ra_h: f64, dec_deg: f64) -> Result<(), String> {
        crate::hardware::mount::Mount::sync(
            self,
            crate::hardware::mount::EqCoord {
                ra: ra_h,
                dec: dec_deg,
            },
        )
        .map_err(|e| e.to_string())
    }

    fn stop(&self) -> Result<(), String> {
        crate::hardware::mount::Mount::stop(self).map_err(|e| e.to_string())
    }

    fn tracking(&self) -> Result<bool, String> {
        self.is_tracking().map_err(|e| e.to_string())
    }

    fn set_tracking(&self, on: bool) -> Result<(), String> {
        crate::hardware::mount::Mount::set_tracking(self, on).map_err(|e| e.to_string())
    }

    fn set_location(&self, lat_deg: f64, lon_deg: f64) -> Result<(), String> {
        crate::hardware::mount::Mount::set_location(self, lat_deg, lon_deg)
            .map_err(|e| e.to_string())
    }

    fn pulse_guide_supported(&self) -> Result<bool, String> {
        crate::hardware::mount::Mount::pulse_guide_supported(self)
            .map(|(supported, _, _)| supported)
            .map_err(|e| e.to_string())
    }

    fn pulse_guide(&self, ns_ms: f32, we_ms: f32) -> Result<(), String> {
        crate::hardware::mount::Mount::pulse_guide(self, ns_ms, we_ms).map_err(|e| e.to_string())
    }

    fn shutdown(&self) -> Result<(), String> {
        crate::hardware::mount::Mount::shutdown(self).map_err(|e| e.to_string())
    }
}

/// The option handed to the OLS wrapper. The public INDI option is
/// `host:port#device=<INDI device name>` (§2.12); OLS `indi:remote` takes
/// only `host[:port]` and picks the telescope itself, so the interim adapter
/// drops the fragment (the native INDI client of Tier 2 uses it).
pub fn backend_option(option: &str) -> String {
    let option = option.split('#').next().unwrap_or_default().trim();
    crate::hardware::mount::MountDriver::Indi.ols_option(option)
}

/// The `#device=` fragment of an option, if one names a device.
fn device_fragment(option: &str) -> Option<&str> {
    option
        .split_once("#device=")
        .map(|(_, device)| device.trim())
        .filter(|device| !device.is_empty())
}

/// `host`, `port` from `http(s)://host:port...` or `host[:port]`.
fn host_port(option: &str, default_port: u16) -> (String, u16) {
    let rest = option
        .split_once("://")
        .map_or(option, |(_, rest)| rest)
        .split(['/', '#', '?'])
        .next()
        .unwrap_or_default();
    match rest.rsplit_once(':') {
        Some((host, port)) => (host.to_string(), port.parse().unwrap_or(default_port)),
        None => (rest.to_string(), default_port),
    }
}

/// The identity of the OLS INDI mount for `option` (§2.6 identity keys).
pub fn ols_identity(option: &str) -> DeviceIdentity {
    let (host, port) = host_port(option.trim(), 7624);
    let host = if host.is_empty() {
        "localhost".to_string()
    } else {
        host
    };
    let mut identity = DeviceIdentity::new(
        DeviceKind::Mount,
        DriverId::OlsIndi,
        Endpoint::Indi {
            host: host.clone(),
            port,
            device: "telescope".into(),
        },
        identity_key::indi(&host, port, "telescope"),
        format!("OLS {} mount", DriverId::OlsIndi),
    );
    identity.driver_info = Some("OpenLiveStacker mount wrapper (legacy adapter)".into());
    identity
}

/// Members the OLS wrapper cannot provide, published as unsupported.
const UNAVAILABLE: &[&str] = &[
    "slewing",
    "atpark",
    "athome",
    "sideofpier",
    "altitude",
    "azimuth",
    "siderealtime",
    "utcdate",
    "ispulseguiding",
    "trackingrate",
    "guideraterightascension",
    "guideratedeclination",
    "siteelevation",
    "devicestate",
    "destinationsideofpier",
];

fn unsupported(what: &str) -> DeviceError {
    DeviceError::unsupported(format!(
        "{what} is not available through the OLS legacy mount adapter; use the native driver"
    ))
}

fn read<B>(lock: &RwLock<B>) -> RwLockReadGuard<'_, B> {
    lock.read().unwrap_or_else(|e| e.into_inner())
}

fn write<B>(lock: &RwLock<B>) -> RwLockWriteGuard<'_, B> {
    lock.write().unwrap_or_else(|e| e.into_inner())
}

/// Why an abort through the adapter reports `Unsupported`.
const ABORT_UNSUPPORTED: &str =
    "OLS stop() halts manual motion but cannot abort a GoTo through the interim INDI adapter";

/// OLS `stop()`, then an honest `Unsupported`: OLS cannot abort a GoTo.
pub struct OlsMountAbort<B: OlsMountBackend> {
    backend: Arc<RwLock<B>>,
}

impl<B: OlsMountBackend> AbortPath for OlsMountAbort<B> {
    fn abort(&self) -> DeviceResult<()> {
        read(&self.backend).stop().map_err(DeviceError::transport)?;
        Err(DeviceError::unsupported(ABORT_UNSUPPORTED))
    }

    fn tracking_off(&self) -> DeviceResult<()> {
        read(&self.backend)
            .set_tracking(false)
            .map_err(DeviceError::transport)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Sample {
    at: DateTime<Utc>,
    mono_ns: u64,
    ra_h: f64,
    dec_deg: f64,
    tracking: Option<bool>,
}

/// The legacy adapter driver.
pub struct OlsMountAdapter<B: OlsMountBackend> {
    identity: DeviceIdentity,
    backend: Arc<RwLock<B>>,
    option: String,
    clock: Arc<dyn Clock>,
    options: MountDriverOptions,
    connected: bool,
    can_pulse_guide: bool,
    last: Option<Sample>,
    last_poll_failed: bool,
}

impl<B: OlsMountBackend> OlsMountAdapter<B> {
    pub fn new(
        identity: DeviceIdentity,
        backend: B,
        option: impl Into<String>,
        clock: Arc<dyn Clock>,
        options: MountDriverOptions,
    ) -> Self {
        Self {
            identity,
            backend: Arc::new(RwLock::new(backend)),
            option: option.into(),
            clock,
            options,
            connected: false,
            can_pulse_guide: false,
            last: None,
            last_poll_failed: false,
        }
    }

    fn backend(&self) -> DeviceResult<RwLockReadGuard<'_, B>> {
        if !self.connected {
            return Err(DeviceError::not_connected("the OLS mount is not connected"));
        }
        Ok(read(&self.backend))
    }
}

/// Create the OLS INDI mount wrapper and its adapter (does not connect).
/// Calls into the OLS bridge, so it is never used by tests.
pub fn open_ols_mount(
    option: &str,
    clock: Arc<dyn Clock>,
    options: MountDriverOptions,
) -> DeviceResult<OlsMountAdapter<crate::hardware::mount::Mount>> {
    let mount = crate::hardware::mount::Mount::new(crate::hardware::mount::MountDriver::Indi)
        .map_err(|e| DeviceError::driver(e.to_string()))?;
    Ok(OlsMountAdapter::new(
        ols_identity(option),
        mount,
        option,
        clock,
        options,
    ))
}

impl<B: OlsMountBackend> Device for OlsMountAdapter<B> {
    fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    fn connect(&mut self, _timeout: Duration) -> DeviceResult<ConnectReport> {
        let started = self.clock.now_mono_ns();
        if let Some(device) = device_fragment(&self.option) {
            tracing::warn!(
                "the OLS INDI mount adapter cannot select INDI device {device:?}; OLS uses the \
                 first telescope on the server (the native INDI client will honour it)"
            );
        }
        let option = backend_option(&self.option);
        write(&self.backend)
            .connect(&option)
            .map_err(DeviceError::transport)?;
        let backend = read(&self.backend);
        if !backend.is_connected() {
            return Err(DeviceError::not_connected(
                "OLS reports the mount disconnected after connect",
            ));
        }
        self.can_pulse_guide = backend.pulse_guide_supported().unwrap_or(false);
        drop(backend);
        self.connected = true;
        Ok(ConnectReport {
            path: ConnectPath::Indi,
            interface_version: None,
            device_state: false,
            unsupported_members: UNAVAILABLE.iter().map(|m| (*m).to_string()).collect(),
            warnings: vec![
                "legacy OLS INDI adapter: Slewing, AtPark, SideOfPier and guide rates are \
                 unavailable, so GoTo outcomes are unknown; the native INDI client replaces it \
                 in Tier 2 (O3)"
                    .into(),
            ],
            duration_ms: self.clock.now_mono_ns().saturating_sub(started) / 1_000_000,
        })
    }

    /// Never parks (D16).
    fn disconnect(&mut self) -> DeviceResult<()> {
        let result = read(&self.backend)
            .shutdown()
            .map_err(DeviceError::transport);
        self.connected = false;
        result
    }

    fn is_connected(&mut self) -> DeviceResult<bool> {
        Ok(self.connected && read(&self.backend).is_connected())
    }

    fn poll(&mut self, _plan: &PollPlan) -> DeviceResult<RawPoll> {
        let started = self.clock.now_mono_ns();
        let (position, tracking) = {
            let backend = self.backend()?;
            (backend.position(), backend.tracking().ok())
        };
        match position {
            Ok((ra_h, dec_deg)) => {
                self.last = Some(Sample {
                    at: self.clock.now_utc(),
                    mono_ns: self.clock.now_mono_ns(),
                    ra_h,
                    dec_deg,
                    tracking,
                });
                self.last_poll_failed = false;
                Ok(RawPoll {
                    method: PollMethod::InProcess,
                    started_mono_ns: started,
                    finished_mono_ns: self.clock.now_mono_ns(),
                    device_time: None,
                    members: Vec::new(),
                })
            }
            Err(err) => {
                self.last_poll_failed = true;
                Err(DeviceError::transport(err))
            }
        }
    }

    fn supported_actions(&mut self) -> DeviceResult<Vec<String>> {
        Ok(Vec::new())
    }

    fn action(&mut self, name: &str, _params: &str) -> DeviceResult<String> {
        Err(unsupported(&format!("action {name}")))
    }

    fn abort_path(&self) -> Option<Arc<dyn AbortPath>> {
        Some(Arc::new(OlsMountAbort {
            backend: Arc::clone(&self.backend),
        }) as Arc<dyn AbortPath>)
    }
}

impl<B: OlsMountBackend> MountDriver for OlsMountAdapter<B> {
    fn capabilities(&mut self) -> DeviceResult<MountCapabilities> {
        let (native_frame, frame_reason) = match native_frame_policy(None, Some(2), &self.options) {
            (EqFrame::Unknown, _) => (
                EqFrame::Unknown,
                Some("ols_legacy_frame_unknown".to_string()),
            ),
            known => known,
        };
        Ok(MountCapabilities {
            can_slew_async: true,
            can_sync: true,
            can_set_tracking: true,
            can_pulse_guide: self.can_pulse_guide,
            native_frame,
            frame_reason,
            unsupported_members: UNAVAILABLE.iter().map(|m| (*m).to_string()).collect(),
            ..MountCapabilities::default()
        })
    }

    fn state(&mut self) -> DeviceResult<MountState> {
        let unsupported = || Observed::unsupported(Source::Property);
        let Some(sample) = self.last else {
            return Ok(MountState::default());
        };
        let fresh = |value| Observed::fresh(value, Source::Property, sample.at, sample.mono_ns);
        let mut state = MountState {
            ra_h: fresh(sample.ra_h),
            dec_deg: fresh(sample.dec_deg),
            alt_deg: unsupported(),
            az_deg: unsupported(),
            sidereal_time_h: unsupported(),
            utc: Observed::unsupported(Source::Property),
            slewing: Observed::unsupported(Source::Property),
            tracking: match sample.tracking {
                Some(on) => Observed::fresh(on, Source::Property, sample.at, sample.mono_ns),
                None => Observed::error(Source::Property),
            },
            tracking_rate: Observed::unsupported(Source::Property),
            at_park: Observed::unsupported(Source::Property),
            at_home: Observed::unsupported(Source::Property),
            side_of_pier: Observed::unsupported(Source::Property),
            is_pulse_guiding: Observed::unsupported(Source::Property),
            guide_rate_ra_arcsec_s: unsupported(),
            guide_rate_dec_arcsec_s: unsupported(),
            site: Observed::unsupported(Source::Property),
            target: Observed::unsupported(Source::Property),
        };
        if self.last_poll_failed {
            state = crate::equipment::mount::stale_state(&state);
        }
        Ok(state)
    }

    fn slew_async(&mut self, target: EqCoord) -> DeviceResult<()> {
        let target = target.normalized();
        self.backend()?
            .goto(target.ra_h, target.dec_deg)
            .map_err(DeviceError::driver)
    }

    fn slew_altaz_async(&mut self, _target: AltAz) -> DeviceResult<()> {
        Err(unsupported("an Alt/Az slew"))
    }

    fn abort_slew(&mut self) -> DeviceResult<()> {
        self.backend()?.stop().map_err(DeviceError::transport)?;
        Err(DeviceError::unsupported(ABORT_UNSUPPORTED))
    }

    fn sync(&mut self, target: EqCoord) -> DeviceResult<()> {
        let target = target.normalized();
        self.backend()?
            .sync(target.ra_h, target.dec_deg)
            .map_err(DeviceError::driver)
    }

    fn set_tracking(&mut self, on: bool) -> DeviceResult<()> {
        self.backend()?
            .set_tracking(on)
            .map_err(DeviceError::driver)
    }

    fn set_tracking_rate(&mut self, _rate: DriveRate) -> DeviceResult<()> {
        Err(unsupported(
            "the tracking rate (the OLS wrapper sets Lunar for Solar)",
        ))
    }

    fn park(&mut self) -> DeviceResult<()> {
        Err(unsupported("park"))
    }

    fn unpark(&mut self) -> DeviceResult<()> {
        Err(unsupported("unpark"))
    }

    fn set_park(&mut self) -> DeviceResult<()> {
        Err(unsupported("set park"))
    }

    fn find_home(&mut self) -> DeviceResult<()> {
        Err(unsupported("find home"))
    }

    fn move_axis(&mut self, _axis: u8, _rate_deg_s: f64) -> DeviceResult<()> {
        Err(unsupported("MoveAxis in deg/s"))
    }

    fn pulse_guide(&mut self, dir: GuideDirection, ms: u32) -> DeviceResult<()> {
        if !self.can_pulse_guide {
            return Err(unsupported("pulse guiding"));
        }
        let ms = ms as f32;
        let (ns, we) = match dir {
            GuideDirection::North => (ms, 0.0),
            GuideDirection::South => (-ms, 0.0),
            GuideDirection::West => (0.0, ms),
            GuideDirection::East => (0.0, -ms),
        };
        self.backend()?
            .pulse_guide(ns, we)
            .map_err(DeviceError::driver)
    }

    fn set_guide_rates(&mut self, _ra_deg_s: f64, _dec_deg_s: f64) -> DeviceResult<()> {
        Err(unsupported("guide rates"))
    }

    /// Latitude and longitude only; OLS has no elevation.
    fn set_site(&mut self, site: &Site) -> DeviceResult<()> {
        self.backend()?
            .set_location(site.lat_deg, site.lon_deg)
            .map_err(DeviceError::driver)
    }

    fn set_utc(&mut self, _utc: DateTime<Utc>) -> DeviceResult<()> {
        Err(unsupported("UTCDate"))
    }

    fn destination_side_of_pier(&mut self, _target: EqCoord) -> DeviceResult<PierSide> {
        Err(DeviceError::new(
            ErrorClass::Unsupported,
            "DestinationSideOfPier is not available through the OLS legacy adapter",
        ))
    }

    fn set_side_of_pier(&mut self, _side: PierSide) -> DeviceResult<()> {
        Err(unsupported("SideOfPier"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::commands::MountCmd;
    use crate::equipment::mount::limits::MountLimits;
    use crate::equipment::mount::{plan, CommandMeta, MountPlan, MountRuntime, MountSettings};
    use crate::equipment::ops::OpProgress;
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::traits::IdentityConverter;
    use crate::equipment::types::{CommandId, CommandSource, ObsState, OpOutcome, SiteSource};
    use std::sync::Mutex;

    #[derive(Default)]
    struct FakeState {
        connected: bool,
        option: String,
        position: (f64, f64),
        tracking: bool,
        calls: Vec<String>,
        fail_position: bool,
    }

    #[derive(Clone, Default)]
    struct FakeOls(Arc<Mutex<FakeState>>);

    impl FakeOls {
        fn state(&self) -> std::sync::MutexGuard<'_, FakeState> {
            self.0.lock().unwrap()
        }
    }

    impl OlsMountBackend for FakeOls {
        fn connect(&mut self, option: &str) -> Result<(), String> {
            let mut s = self.state();
            s.connected = true;
            s.option = option.to_string();
            s.position = (5.0, 20.0);
            s.tracking = true;
            Ok(())
        }
        fn is_connected(&self) -> bool {
            self.state().connected
        }
        fn position(&self) -> Result<(f64, f64), String> {
            let s = self.state();
            if s.fail_position {
                Err("socket closed".into())
            } else {
                Ok(s.position)
            }
        }
        fn goto(&self, ra_h: f64, dec_deg: f64) -> Result<(), String> {
            let mut s = self.state();
            s.calls.push(format!("goto {ra_h} {dec_deg}"));
            s.position = (ra_h, dec_deg);
            Ok(())
        }
        fn sync(&self, ra_h: f64, dec_deg: f64) -> Result<(), String> {
            let mut s = self.state();
            s.calls.push("sync".into());
            s.position = (ra_h, dec_deg);
            Ok(())
        }
        fn stop(&self) -> Result<(), String> {
            self.state().calls.push("stop".into());
            Ok(())
        }
        fn tracking(&self) -> Result<bool, String> {
            Ok(self.state().tracking)
        }
        fn set_tracking(&self, on: bool) -> Result<(), String> {
            self.state().tracking = on;
            Ok(())
        }
        fn set_location(&self, lat_deg: f64, lon_deg: f64) -> Result<(), String> {
            self.state()
                .calls
                .push(format!("location {lat_deg} {lon_deg}"));
            Ok(())
        }
        fn pulse_guide_supported(&self) -> Result<bool, String> {
            Ok(true)
        }
        fn pulse_guide(&self, ns_ms: f32, we_ms: f32) -> Result<(), String> {
            self.state().calls.push(format!("pulse {ns_ms} {we_ms}"));
            Ok(())
        }
        fn shutdown(&self) -> Result<(), String> {
            self.state().connected = false;
            Ok(())
        }
    }

    fn adapter(option: &str) -> (FakeOls, Arc<ManualClock>, OlsMountAdapter<FakeOls>) {
        let fake = FakeOls::default();
        let clock = Arc::new(ManualClock::new());
        let adapter = OlsMountAdapter::new(
            ols_identity(option),
            fake.clone(),
            option,
            clock.clone(),
            MountDriverOptions::default(),
        );
        (fake, clock, adapter)
    }

    #[test]
    fn identities_follow_the_indi_key_rule() {
        let id = ols_identity("pi.local:7625");
        assert_eq!(id.driver, DriverId::OlsIndi);
        assert_eq!(id.identity_key, "indi:pi.local:7625/telescope");
        assert_eq!(
            ols_identity("").identity_key,
            "indi:localhost:7624/telescope"
        );
    }

    /// §2.12's INDI option `host:port#device=<name>`: OLS `indi:remote`
    /// takes only `host[:port]` (without a port it would take the whole
    /// string as the host name), so the interim adapter drops the fragment;
    /// the identity keeps its host and port.
    #[test]
    fn the_indi_device_fragment_never_reaches_ols() {
        for (option, ols) in [
            ("pi.local:7625#device=EQMod Mount", "pi.local:7625"),
            ("pi.local#device=EQMod Mount", "pi.local"),
            ("pi.local:7625", "pi.local:7625"),
            ("#device=EQMod Mount", "localhost:7624"),
            ("", "localhost:7624"),
        ] {
            assert_eq!(backend_option(option), ols, "{option:?}");
            let (fake, _clock, mut drv) = adapter(option);
            drv.connect(Duration::from_secs(1)).unwrap();
            assert_eq!(fake.state().option, ols, "{option:?}");
        }
        assert_eq!(
            ols_identity("pi.local:7625#device=EQMod Mount").identity_key,
            "indi:pi.local:7625/telescope"
        );
        assert_eq!(device_fragment("h:1#device= EQMod "), Some("EQMod"));
        assert_eq!(device_fragment("h:1#device="), None);
        assert_eq!(device_fragment("h:1"), None);
    }

    #[test]
    fn the_adapter_publishes_what_ols_cannot_do_as_unsupported() {
        let (fake, _clock, mut drv) = adapter("auto");
        let report = drv.connect(Duration::from_secs(1)).unwrap();
        assert_eq!(
            fake.state().option,
            "localhost:7624",
            "OLS indi option mapping"
        );
        assert_eq!(report.path, ConnectPath::Indi);
        assert!(report.unsupported_members.contains(&"slewing".to_string()));
        let caps = drv.capabilities().unwrap();
        assert_eq!(caps.native_frame, EqFrame::Unknown);
        assert_eq!(
            caps.frame_reason.as_deref(),
            Some("ols_legacy_frame_unknown")
        );
        assert!(caps.can_slew_async && !caps.can_park && caps.tracking_rates.is_empty());
        drv.poll(&PollPlan::default()).unwrap();
        let state = drv.state().unwrap();
        assert_eq!(state.ra_h.value, Some(5.0));
        assert_eq!(state.tracking.value, Some(true));
        assert_eq!(state.slewing.state, ObsState::Unsupported);
        assert_eq!(state.at_park.state, ObsState::Unsupported);
        assert_eq!(
            drv.set_tracking_rate(DriveRate::Solar).unwrap_err().class,
            ErrorClass::Unsupported
        );
        assert_eq!(drv.park().unwrap_err().class, ErrorClass::Unsupported);
    }

    #[test]
    fn pulses_map_to_the_ols_sign_convention() {
        let (fake, _clock, mut drv) = adapter("pi.local:7624");
        drv.connect(Duration::from_secs(1)).unwrap();
        for dir in [
            GuideDirection::North,
            GuideDirection::South,
            GuideDirection::West,
            GuideDirection::East,
        ] {
            drv.pulse_guide(dir, 200).unwrap();
        }
        assert_eq!(
            fake.state().calls,
            vec!["pulse 200 0", "pulse -200 0", "pulse 0 200", "pulse 0 -200"]
        );
    }

    #[test]
    fn abort_is_attempted_but_reported_as_unsupported() {
        let (fake, _clock, mut drv) = adapter("pi.local:7624");
        drv.connect(Duration::from_secs(1)).unwrap();
        let err = drv.abort_path().unwrap().abort().unwrap_err();
        assert_eq!(err.class, ErrorClass::Unsupported);
        assert_eq!(fake.state().calls, vec!["stop"]);
        assert_eq!(drv.abort_slew().unwrap_err().class, ErrorClass::Unsupported);
    }

    #[test]
    fn a_legacy_goto_ends_unknown_because_slewing_cannot_be_read() {
        let (fake, clock, mut drv) = adapter("pi.local:7624");
        drv.connect(Duration::from_secs(1)).unwrap();
        drv.poll(&PollPlan::default()).unwrap();
        let site = Site {
            lat_deg: 40.0,
            lon_deg: -75.0,
            elev_m: 100.0,
            source: SiteSource::Config,
        };
        let mut runtime = MountRuntime::new(
            MountSettings::default(),
            MountLimits::default(),
            Arc::new(IdentityConverter),
            clock.clone(),
            drv.identity().connection_id,
            Some(site),
        );
        runtime.on_connected(&mut drv).unwrap();
        let snapshot = runtime.after_poll(&mut drv, None);
        assert_eq!(
            snapshot.state.alt_deg.source,
            Source::Derived,
            "Alt/Az derived from the site"
        );
        let meta = CommandMeta::new(
            CommandId::new(),
            CommandSource::Operator { overriding: false },
        );
        let target = EqCoord::new(6.0, 30.0, EqFrame::Unknown);
        let MountPlan::Start(mut op) = runtime.plan(MountCmd::Slew { target }, meta, None) else {
            panic!("refused");
        };
        op.start(&mut drv, clock.as_ref()).unwrap();
        assert_eq!(fake.state().calls, vec!["goto 6 30"]);
        clock.advance(Duration::from_millis(250));
        drv.poll(&plan(1, true, &[])).unwrap();
        let outcome = match op.step(&mut drv, clock.as_ref()) {
            OpProgress::Done(outcome) => outcome,
            other => panic!("{other:?}"),
        };
        assert!(matches!(outcome, OpOutcome::Unknown(_)), "{outcome:?}");
        // A J2000 target is refused: the legacy frame is unknown.
        let meta = CommandMeta::new(
            CommandId::new(),
            CommandSource::Operator { overriding: false },
        );
        let j2000 = EqCoord::new(6.0, 30.0, EqFrame::J2000);
        let MountPlan::Start(mut op) = runtime.plan(MountCmd::Slew { target: j2000 }, meta, None)
        else {
            panic!("refused");
        };
        assert_eq!(
            op.start(&mut drv, clock.as_ref()).unwrap_err().class,
            ErrorClass::InvalidOperation
        );
    }

    #[test]
    fn a_failed_read_leaves_stale_values_and_disconnect_never_parks() {
        let (fake, _clock, mut drv) = adapter("pi.local:7624");
        drv.connect(Duration::from_secs(1)).unwrap();
        drv.poll(&PollPlan::default()).unwrap();
        fake.state().fail_position = true;
        assert!(drv.poll(&PollPlan::default()).is_err());
        assert_eq!(drv.state().unwrap().ra_h.state, ObsState::Stale);
        drv.disconnect().unwrap();
        assert!(!fake.state().calls.iter().any(|c| c.contains("park")));
        assert_eq!(
            drv.slew_async(EqCoord::default()).unwrap_err().class,
            ErrorClass::NotConnected
        );
    }
}

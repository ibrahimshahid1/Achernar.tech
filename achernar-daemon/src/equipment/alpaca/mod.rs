//! ASCOM Alpaca client and device drivers, hand-written over `ureq` (O7).
//!
//! Module tree created by WP0; owner: WP1A-1. The core (this file,
//! [`client`], [`errors`], [`form`], [`devicestate`], [`discovery`]) is an
//! independent implementation of the ASCOM Alpaca API reference, the Master
//! Interfaces and the Alpaca OpenAPI description (MIT), with the ASCOM
//! Library (MIT) as the behaviour reference (§6). No code is copied.
//!
//! What every Alpaca device driver gets from the core:
//!
//! - [`client::AlpacaClient`], the [`AlpacaTransport`] over HTTP, and its
//!   out-of-band [`client::AbortChannel`] (D4);
//! - the C-3 response checks and typed `Value` extraction ([`errors`]);
//! - form encoding with exact-case keys ([`form`]);
//! - the DeviceState parser ([`devicestate`]);
//! - the management API and opt-in UDP discovery ([`discovery`]);
//! - here: device-type names, the Platform 7 thresholds and the common
//!   connect, disconnect and identity sequence of §2.6, so each device
//!   driver implements `Device::connect` with one call.
//!
//! The device drivers (`telescope`, `focuser`, ...) belong to their own work
//! packages (§2.2).

pub mod client;
pub mod covercalibrator;
pub mod devicestate;
pub mod discovery;
pub mod dome;
pub mod errors;
pub mod filterwheel;
pub mod focuser;
pub mod form;
pub mod observingconditions;
pub mod rotator;
pub mod safetymonitor;
pub mod switch;
pub mod telescope;

use std::time::Duration;

use crate::equipment::traits::{AlpacaTransport, Clock, ConnectPath, ConnectReport, TimeoutClass};
use crate::equipment::types::{DeviceError, DeviceIdentity, DeviceKind, DeviceResult};

use self::client::UnsupportedMembers;
use self::errors::{expect_bool, expect_i32, expect_string, expect_string_list};

/// The Alpaca API version in every device path (`/api/v1/...`).
pub const ALPACA_API_VERSION: u32 = 1;

/// The Alpaca device type that serves an equipment kind. The guider has
/// none (it is PHD2); cameras stay on OLS (O1) but the name is listed so
/// discovery can report them.
pub fn device_type_for_kind(kind: DeviceKind) -> Option<&'static str> {
    match kind {
        DeviceKind::Camera => Some("camera"),
        DeviceKind::Mount => Some("telescope"),
        DeviceKind::Focuser => Some("focuser"),
        DeviceKind::FilterWheel => Some("filterwheel"),
        DeviceKind::Rotator => Some("rotator"),
        DeviceKind::CoverCalibrator => Some("covercalibrator"),
        DeviceKind::Switch => Some("switch"),
        DeviceKind::Weather => Some("observingconditions"),
        DeviceKind::SafetyMonitor => Some("safetymonitor"),
        DeviceKind::Dome => Some("dome"),
        DeviceKind::Guider => None,
    }
}

/// The equipment kind of an Alpaca device type (any case).
pub fn kind_for_device_type(device_type: &str) -> Option<DeviceKind> {
    DeviceKind::ALL.into_iter().find(|kind| {
        device_type_for_kind(*kind).is_some_and(|name| name.eq_ignore_ascii_case(device_type))
    })
}

/// The interface version from which a device type uses the Platform 7
/// `connect`/`connecting` members and offers `devicestate` (§2.6 step 4).
pub fn platform7_interface_version(device_type: &str) -> Option<i32> {
    match device_type.to_ascii_lowercase().as_str() {
        "telescope" | "focuser" | "rotator" | "camera" => Some(4),
        "filterwheel" | "switch" | "safetymonitor" | "dome" => Some(3),
        "covercalibrator" | "observingconditions" => Some(2),
        _ => None,
    }
}

/// How [`connect_device`] waits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConnectOptions {
    /// `[equipment] connect_timeout_s` (default 30 s, generous for slow
    /// ESP32 Wi-Fi).
    pub timeout: Duration,
    /// Platform 7 `connecting` poll interval (250 ms).
    pub poll_interval: Duration,
}

impl Default for ConnectOptions {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            poll_interval: Duration::from_millis(250),
        }
    }
}

impl ConnectOptions {
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            timeout,
            ..Self::default()
        }
    }
}

/// `interfaceversion`, or 1 when it cannot be read (the ASCOM Library rule,
/// §2.6 step 3).
pub fn read_interface_version(transport: &dyn AlpacaTransport) -> i32 {
    transport
        .get("interfaceversion", &[], TimeoutClass::Connect)
        .and_then(|value| expect_i32("interfaceversion", &value))
        .unwrap_or(1)
}

fn read_bool(
    transport: &dyn AlpacaTransport,
    member: &str,
    class: TimeoutClass,
) -> DeviceResult<bool> {
    transport
        .get(member, &[], class)
        .and_then(|value| expect_bool(member, &value))
}

/// The common Alpaca connect (§2.6 steps 3, 4 and the DeviceState probe).
///
/// - At or above the Platform 7 threshold: PUT `connect`, then poll
///   `connecting` every `poll_interval` until it is false, within `timeout`.
///   A device that declares Platform 7 but does not know `connect` falls back
///   to the Platform 6 path with a warning.
/// - Below it: PUT `Connected=True`.
/// - Either way, GET `connected` must then read true.
/// - Platform 7 only: one `devicestate` GET; a non-empty array enables the
///   one-GET poll. An unknown member (HTTP 400/404, NotImplemented) is
///   memoised as unsupported, never a failure.
///
/// Connecting never moves hardware (D16). Blocking: call it from a worker
/// thread.
pub fn connect_device(
    transport: &dyn AlpacaTransport,
    device_type: &str,
    clock: &dyn Clock,
    options: &ConnectOptions,
) -> DeviceResult<ConnectReport> {
    let started = clock.now_mono_ns();
    let mut report = ConnectReport {
        interface_version: Some(read_interface_version(transport)),
        ..ConnectReport::default()
    };
    let mut unsupported = UnsupportedMembers::default();
    let version = report.interface_version.unwrap_or(1);
    let platform7 = platform7_interface_version(device_type).is_some_and(|min| version >= min);

    report.path = ConnectPath::Platform6;
    if platform7 {
        match transport.put("connect", &[], TimeoutClass::Connect) {
            Ok(_) => {
                report.path = ConnectPath::Platform7;
                wait_while_connecting(transport, clock, options, &mut unsupported, &mut report)?;
            }
            Err(error) if error.is_unknown_member() => {
                unsupported.note("connect", &error);
                report.warnings.push(format!(
                    "device declares InterfaceVersion {version} but has no Connect method; \
                     using the Platform 6 Connected property"
                ));
            }
            Err(error) => return Err(error),
        }
    }
    if report.path == ConnectPath::Platform6 {
        transport.put(
            "connected",
            &[("Connected", form::flag(true).into())],
            TimeoutClass::Connect,
        )?;
    }
    if !read_bool(transport, "connected", TimeoutClass::Connect)? {
        return Err(DeviceError::not_connected(
            "the device reports Connected = false after connecting",
        ));
    }

    if platform7 {
        match transport.get("devicestate", &[], TimeoutClass::Ordinary) {
            Ok(value) => match devicestate::parse_device_state(&value) {
                Ok(state) => {
                    report.device_state = !state.is_empty();
                    if state.is_empty() {
                        report
                            .warnings
                            .push("devicestate is empty; polling members individually".into());
                    }
                }
                Err(error) => report.warnings.push(format!(
                    "devicestate is malformed ({error}); polling members individually"
                )),
            },
            Err(error) if error.is_unknown_member() => {
                unsupported.note("devicestate", &error);
            }
            Err(error) => report.warnings.push(format!(
                "devicestate could not be read ({error}); polling members individually"
            )),
        }
    } else {
        // Below Platform 7 the member does not exist (§2.6 step 6).
        unsupported.insert("devicestate");
    }
    report.unsupported_members = unsupported.to_vec();
    report.duration_ms = clock.now_mono_ns().saturating_sub(started) / 1_000_000;
    Ok(report)
}

fn wait_while_connecting(
    transport: &dyn AlpacaTransport,
    clock: &dyn Clock,
    options: &ConnectOptions,
    unsupported: &mut UnsupportedMembers,
    report: &mut ConnectReport,
) -> DeviceResult<()> {
    let deadline = clock
        .now_mono_ns()
        .saturating_add(crate::equipment::clock::duration_ns(options.timeout));
    // Bounded by attempts as well as by the clock, so an injected clock
    // that does not advance cannot loop forever.
    let interval_ns = crate::equipment::clock::duration_ns(options.poll_interval).max(1);
    let max_polls = crate::equipment::clock::duration_ns(options.timeout) / interval_ns + 1;
    for _ in 0..=max_polls {
        match read_bool(transport, "connecting", TimeoutClass::Connect) {
            Ok(false) => return Ok(()),
            Ok(true) => {}
            Err(error) if error.is_unknown_member() => {
                unsupported.note("connecting", &error);
                report.warnings.push(
                    "device has Connect but no Connecting; checking Connected instead".into(),
                );
                return Ok(());
            }
            Err(error) => return Err(error),
        }
        if clock.now_mono_ns() >= deadline {
            break;
        }
        std::thread::sleep(options.poll_interval);
    }
    Err(DeviceError::timeout(format!(
        "the device was still connecting after {} s",
        options.timeout.as_secs_f64()
    )))
}

/// The common Alpaca disconnect: Platform 7 `disconnect` then `connecting`
/// until false, else `Connected=False`. Never parks or stops anything else
/// (D16); a server that tracks clients by ClientID (the ESP32 mount) keeps
/// serving its other clients.
pub fn disconnect_device(
    transport: &dyn AlpacaTransport,
    device_type: &str,
    interface_version: Option<i32>,
    clock: &dyn Clock,
    options: &ConnectOptions,
) -> DeviceResult<()> {
    let version = interface_version.unwrap_or_else(|| read_interface_version(transport));
    let platform7 = platform7_interface_version(device_type).is_some_and(|min| version >= min);
    if platform7 {
        match transport.put("disconnect", &[], TimeoutClass::Connect) {
            Ok(_) => {
                let mut unsupported = UnsupportedMembers::default();
                let mut report = ConnectReport::default();
                return wait_while_connecting(
                    transport,
                    clock,
                    options,
                    &mut unsupported,
                    &mut report,
                );
            }
            Err(error) if error.is_unknown_member() => {}
            Err(error) => return Err(error),
        }
    }
    transport
        .put(
            "connected",
            &[("Connected", form::flag(false).into())],
            TimeoutClass::Connect,
        )
        .map(|_| ())
}

/// Identity members read after connect (§2.6 step 5).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AlpacaIdentityInfo {
    pub name: String,
    pub description: Option<String>,
    pub driver_info: Option<String>,
    pub driver_version: Option<String>,
    pub interface_version: Option<i32>,
    pub supported_actions: Vec<String>,
}

/// Read Name (required), Description, DriverInfo, DriverVersion,
/// InterfaceVersion and SupportedActions. A missing optional member is
/// unknown, never invented.
pub fn read_identity(transport: &dyn AlpacaTransport) -> DeviceResult<AlpacaIdentityInfo> {
    let text = |member: &str| {
        transport
            .get(member, &[], TimeoutClass::Ordinary)
            .and_then(|value| expect_string(member, &value))
            .ok()
            .map(|text| text.trim().to_string())
            .filter(|text| !text.is_empty())
    };
    let name = transport
        .get("name", &[], TimeoutClass::Ordinary)
        .and_then(|value| expect_string("name", &value))?;
    Ok(AlpacaIdentityInfo {
        name: name.trim().to_string(),
        description: text("description"),
        driver_info: text("driverinfo"),
        driver_version: text("driverversion"),
        interface_version: transport
            .get("interfaceversion", &[], TimeoutClass::Connect)
            .and_then(|value| expect_i32("interfaceversion", &value))
            .ok(),
        supported_actions: transport
            .get("supportedactions", &[], TimeoutClass::Ordinary)
            .and_then(|value| expect_string_list("supportedactions", &value))
            .unwrap_or_default(),
    })
}

impl AlpacaIdentityInfo {
    /// Fill a [`DeviceIdentity`] with what the device claims, the UniqueID
    /// from `configureddevices` and the setup page (§2.6 step 5).
    pub fn apply_to(
        &self,
        identity: &mut DeviceIdentity,
        unique_id: Option<&str>,
        setup_url: Option<String>,
    ) {
        if !self.name.is_empty() {
            identity.name = self.name.clone();
        }
        identity.description = self.description.clone();
        identity.driver_info = self.driver_info.clone();
        identity.driver_version = self.driver_version.clone();
        identity.interface_version = self.interface_version.or(identity.interface_version);
        identity.unique_id = unique_id
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(str::to_string);
        identity.setup_url = setup_url;
    }
}

#[cfg(test)]
mod tests {
    use super::client::{AlpacaClient, ClientConfig, ClientId};
    use super::*;
    use crate::equipment::testing::alpaca_server::{ScriptedDevice, ServerReply, TestAlpacaServer};
    use crate::equipment::testing::fake_transport::{FakeAlpacaTransport, FakeReply, Method};
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::types::{DriverId, ErrorClass, Qualification};
    use serde_json::json;

    fn quick() -> ConnectOptions {
        ConnectOptions {
            timeout: Duration::from_millis(200),
            poll_interval: Duration::from_millis(1),
        }
    }

    fn client(server: &TestAlpacaServer, device_type: &str) -> AlpacaClient {
        AlpacaClient::new(
            &server.base_url(),
            device_type,
            0,
            ClientConfig::new(ClientId::new(31).unwrap()),
        )
        .unwrap()
    }

    #[test]
    fn device_types_and_platform7_thresholds_follow_section_2_6() {
        assert_eq!(device_type_for_kind(DeviceKind::Mount), Some("telescope"));
        assert_eq!(
            device_type_for_kind(DeviceKind::Weather),
            Some("observingconditions")
        );
        assert_eq!(device_type_for_kind(DeviceKind::Guider), None);
        assert_eq!(
            kind_for_device_type("FilterWheel"),
            Some(DeviceKind::FilterWheel)
        );
        assert_eq!(kind_for_device_type("guider"), None);
        let thresholds = [
            ("telescope", 4),
            ("focuser", 4),
            ("filterwheel", 3),
            ("rotator", 4),
            ("camera", 4),
            ("covercalibrator", 2),
            ("switch", 3),
            ("observingconditions", 2),
            ("safetymonitor", 3),
            ("dome", 3),
        ];
        for (device_type, version) in thresholds {
            assert_eq!(
                platform7_interface_version(device_type),
                Some(version),
                "{device_type}"
            );
        }
        assert_eq!(platform7_interface_version("video"), None);
    }

    #[test]
    fn the_esp32_like_mount_connects_on_the_platform7_path_with_devicestate() {
        let server = TestAlpacaServer::start(vec![ScriptedDevice::esp32_like_mount(0)]);
        let mount = client(&server, "telescope");
        let report = connect_device(&mount, "telescope", &ManualClock::new(), &quick()).unwrap();
        assert_eq!(report.path, ConnectPath::Platform7);
        assert_eq!(report.interface_version, Some(4));
        assert!(report.device_state);
        assert!(report.unsupported_members.is_empty());
        let members: Vec<String> = server.requests().iter().map(|r| r.member.clone()).collect();
        assert_eq!(
            members,
            vec![
                "interfaceversion",
                "connect",
                "connecting",
                "connecting",
                "connected",
                "devicestate"
            ]
        );
        // Connecting moved nothing (D16).
        assert!(server
            .requests()
            .iter()
            .all(|r| r.verb == errors::Verb::Get || r.member == "connect"));

        let info = read_identity(&mount).unwrap();
        assert_eq!(info.name, "ESP32-like mount (test profile)");
        assert_eq!(info.driver_version.as_deref(), Some("1.0"));
        assert_eq!(info.interface_version, Some(4));
        let mut identity = DeviceIdentity::new(
            DeviceKind::Mount,
            DriverId::Alpaca,
            mount.endpoint(),
            mount.identity_key(Some("test-esp32-mount-0001")),
            "pending",
        );
        info.apply_to(
            &mut identity,
            Some("test-esp32-mount-0001"),
            Some(mount.setup_url()),
        );
        assert_eq!(identity.identity_key, "alpaca:test-esp32-mount-0001");
        assert_eq!(identity.name, info.name);
        assert_eq!(
            identity.qualification,
            Qualification::ImplementedUnqualified
        );
        assert!(identity
            .setup_url
            .unwrap()
            .ends_with("/setup/v1/telescope/0/setup"));

        disconnect_device(&mount, "telescope", Some(4), &ManualClock::new(), &quick()).unwrap();
        assert_eq!(
            mount.get("connected", &[], TimeoutClass::Connect).unwrap(),
            json!(false)
        );
    }

    #[test]
    fn the_platform6_focuser_connects_through_the_connected_property() {
        let server = TestAlpacaServer::start(vec![ScriptedDevice::platform6_focuser(0)]);
        let focuser = client(&server, "focuser");
        let report = connect_device(&focuser, "focuser", &ManualClock::new(), &quick()).unwrap();
        assert_eq!(report.path, ConnectPath::Platform6);
        assert_eq!(report.interface_version, Some(3));
        assert!(!report.device_state);
        assert_eq!(report.unsupported_members, vec!["devicestate".to_string()]);
        let members: Vec<String> = server.requests().iter().map(|r| r.member.clone()).collect();
        assert_eq!(members, vec!["interfaceversion", "connected", "connected"]);
        let put = &server.requests()[1];
        assert_eq!(form::find(&put.params, "Connected", true), Some("True"));
        disconnect_device(&focuser, "focuser", Some(3), &ManualClock::new(), &quick()).unwrap();
        assert_eq!(
            focuser
                .get("connected", &[], TimeoutClass::Connect)
                .unwrap(),
            json!(false)
        );
    }

    #[test]
    fn a_failed_interface_version_read_is_treated_as_version_1() {
        let fake = FakeAlpacaTransport::new();
        fake.on_get("interfaceversion", FakeReply::Timeout)
            .on_put("connected", FakeReply::ok())
            .on_get("connected", FakeReply::value(true));
        let report = connect_device(&fake, "telescope", &ManualClock::new(), &quick()).unwrap();
        assert_eq!(report.interface_version, Some(1));
        assert_eq!(report.path, ConnectPath::Platform6);
        assert_eq!(fake.calls_to(Method::Put, "connect"), 0);
        assert_eq!(fake.calls_to(Method::Get, "devicestate"), 0);
    }

    #[test]
    fn a_platform7_claim_without_connect_falls_back_and_an_unknown_devicestate_is_memoised() {
        let fake = FakeAlpacaTransport::new();
        fake.on_get("interfaceversion", FakeReply::value(4))
            .on_put("connect", FakeReply::http(400, "Unknown method connect"))
            .on_put("connected", FakeReply::ok())
            .on_get("connected", FakeReply::value(true))
            .on_get("devicestate", FakeReply::http(404, "Not Found"));
        let report = connect_device(&fake, "focuser", &ManualClock::new(), &quick()).unwrap();
        assert_eq!(report.path, ConnectPath::Platform6);
        assert!(!report.device_state);
        assert_eq!(
            report.unsupported_members,
            vec!["connect".to_string(), "devicestate".to_string()]
        );
        assert_eq!(report.warnings.len(), 1);
    }

    #[test]
    fn connecting_that_never_finishes_times_out_and_connected_false_is_refused() {
        let fake = FakeAlpacaTransport::new();
        fake.on_get("interfaceversion", FakeReply::value(3))
            .on_put("connect", FakeReply::ok())
            .on_get("connecting", FakeReply::value(true));
        let err = connect_device(&fake, "filterwheel", &ManualClock::new(), &quick()).unwrap_err();
        assert_eq!(err.class, ErrorClass::Timeout);
        assert!(
            fake.calls_to(Method::Get, "connecting") <= 202,
            "bounded by attempts"
        );

        let fake = FakeAlpacaTransport::new();
        fake.on_get("interfaceversion", FakeReply::value(1))
            .on_put("connected", FakeReply::ok())
            .on_get("connected", FakeReply::value(false));
        let err =
            connect_device(&fake, "safetymonitor", &ManualClock::new(), &quick()).unwrap_err();
        assert_eq!(err.class, ErrorClass::NotConnected);

        let fake = FakeAlpacaTransport::new();
        fake.on_get("interfaceversion", FakeReply::value(1))
            .on_put("connected", FakeReply::alpaca_error(0x500, "port busy"));
        let err =
            connect_device(&fake, "safetymonitor", &ManualClock::new(), &quick()).unwrap_err();
        assert_eq!(err.class, ErrorClass::Driver);
    }

    #[test]
    fn an_empty_or_malformed_devicestate_falls_back_to_member_polls() {
        let server = TestAlpacaServer::start(vec![ScriptedDevice::new("filterwheel", 0, "Wheel")
            .interface_version(3)
            .platform7(0)
            .on_get("devicestate", ServerReply::value(json!([])))]);
        let wheel = client(&server, "filterwheel");
        let report = connect_device(&wheel, "filterwheel", &ManualClock::new(), &quick()).unwrap();
        assert_eq!(report.path, ConnectPath::Platform7);
        assert!(!report.device_state);
        assert!(report.warnings[0].contains("empty"));

        let fake = FakeAlpacaTransport::new();
        fake.on_get("interfaceversion", FakeReply::value(3))
            .on_put("connect", FakeReply::ok())
            .on_get("connecting", FakeReply::value(false))
            .on_get("connected", FakeReply::value(true))
            .on_get("devicestate", FakeReply::value(json!({"not": "an array"})));
        let report = connect_device(&fake, "dome", &ManualClock::new(), &quick()).unwrap();
        assert!(!report.device_state);
        assert!(report.warnings[0].contains("malformed"));
    }

    #[test]
    fn identity_needs_a_name_but_never_invents_the_rest() {
        let fake = FakeAlpacaTransport::new();
        fake.on_get("name", FakeReply::value("  Wheel  "))
            .on_get("description", FakeReply::not_implemented())
            .on_get("driverversion", FakeReply::value(""));
        let info = read_identity(&fake).unwrap();
        assert_eq!(info.name, "Wheel");
        assert_eq!(info.description, None);
        assert_eq!(info.driver_info, None);
        assert_eq!(info.driver_version, None);
        assert_eq!(info.interface_version, None);
        assert!(info.supported_actions.is_empty());
        let fake = FakeAlpacaTransport::new();
        assert!(read_identity(&fake).is_err());
    }
}

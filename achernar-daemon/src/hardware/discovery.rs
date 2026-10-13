//! Device autodiscovery — user-triggered "scan for devices".
//!
//! `driver_capability_report()` answers "which drivers were compiled into this
//! build?". This module answers the different question "what is actually
//! plugged in *right now*?" by instantiating each locally-attached camera driver
//! and calling its device enumeration. For mounts/focusers, an operator may
//! instead query the configured-device list of one explicitly addressed Alpaca
//! server. Those entries are reported configuration, not confirmed hardware.
//!
//! # Deliberate scope: local buses and explicit endpoints
//!
//! Only USB/SDK drivers are probed (ASI, SVBony, ToupTek, UVC, gphoto2).
//! Network transports are **not** automatically scanned here:
//!
//! * **Alpaca** has a real discovery standard (ASCOM Alpaca Discovery: a UDP
//!   broadcast of `alpacadiscovery1` to port 32227, to which devices reply with
//!   their HTTP port). That is a network broadcast with its own failure modes
//!   and firewall implications, so it belongs behind its own explicit opt-in.
//! * **INDI / INDIGO** require an explicit `host:port` (7624 by default);
//!   scanning for them means port-scanning the LAN, which we will not do
//!   silently from an observatory control daemon.
//! * **Simulator** and **watch-folder** are not hardware — they are always
//!   selectable via the capability report and need a user-supplied option.
//!
//! # Why probing is user-triggered, never automatic
//!
//! Creating a vendor SDK driver can claim exclusive USB handles, power-cycle a
//! device, or block on a wedged connection. Doing that spontaneously while a
//! session is imaging risks interrupting a capture. So discovery runs only when
//! the operator presses "Scan", and refuses outright while a camera is
//! connected — consistent with the manual-first principle: the daemon reports
//! what it found and the operator chooses what to connect.

use serde::{Deserialize, Serialize};

use super::camera::OlsCameraDriver;
use super::ols_ffi;
use crate::state::AppState;

/// How long a single driver probe may take before we stop waiting.
///
/// Note this abandons the blocking task rather than cancelling it — an FFI call
/// already inside a vendor SDK cannot be interrupted. The task is left to
/// finish and drop its driver on its own thread; the point is that the HTTP
/// request still returns promptly instead of hanging on a wedged device.
const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// One camera actually present on a local bus.
#[derive(Debug, Clone, Serialize)]
pub struct DiscoveredCamera {
    /// Driver id to pass back to `POST /connect` (e.g. "svb").
    pub driver_id: &'static str,
    pub driver_label: &'static str,
    /// Index within that driver's enumeration — what `open_camera` takes.
    pub index: i32,
    /// Vendor-reported device name.
    pub name: String,
    /// Round-trips directly into `POST /connect`. This is stable only for this
    /// scan; driver enumeration order can change after USB reconnect.
    pub selection_key: String,
    /// Null until the native driver exposes a serial number or VID/PID tuple.
    pub hardware_id: Option<String>,
    pub transport: &'static str,
}

/// Result of probing one driver. Reported even on failure: "the ASI SDK is
/// present but found nothing" and "the ASI probe errored" are different
/// situations for a user staring at a dark observatory, and collapsing them
/// into an empty list would hide the difference.
#[derive(Debug, Clone, Serialize)]
pub struct ProbeOutcome {
    pub driver_id: &'static str,
    pub driver_label: &'static str,
    /// "found" | "empty" | "not_compiled" | "failed" | "timeout" | "skipped"
    pub status: &'static str,
    pub device_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct DiscoveryReport {
    pub cameras: Vec<DiscoveredCamera>,
    pub probes: Vec<ProbeOutcome>,
    pub mounts: Vec<DiscoveredDevice>,
    pub focusers: Vec<DiscoveredDevice>,
    pub filter_wheels: Vec<DiscoveredDevice>,
    pub device_probes: Vec<DeviceProbeOutcome>,
    pub scanned_at: chrono::DateTime<chrono::Utc>,
    /// Human-readable caveats (what was deliberately not scanned, and why).
    pub notes: Vec<String>,
}

/// A configured device reported by an explicitly addressed server. This does
/// not certify that the device is physically attached or can connect.
#[derive(Debug, Clone, Serialize)]
pub struct DiscoveredDevice {
    pub driver_id: &'static str,
    pub driver_label: &'static str,
    /// Alpaca DeviceNumber, not position in the management response array.
    pub index: i32,
    pub name: String,
    pub selection_key: String,
    pub hardware_id: Option<String>,
    pub transport: &'static str,
    pub availability: &'static str,
    /// Round-trip unchanged to /connect. Includes the exact device selector.
    pub option: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DeviceProbeOutcome {
    pub equipment_type: &'static str,
    pub driver_id: &'static str,
    pub status: &'static str,
    pub device_count: usize,
    pub detail: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryRequest {
    pub equipment_type: Option<String>,
    pub driver: Option<String>,
    /// Explicit Alpaca base URL. Never defaults to network discovery.
    pub option: Option<String>,
}

/// Scan local cameras or query one explicitly supplied Alpaca server. Native
/// mount/focuser driver lists contain driver definitions, not attached devices,
/// and must never be presented as discovered hardware.
pub async fn discover_equipment(
    state: &AppState,
    request: DiscoveryRequest,
) -> Result<DiscoveryReport, String> {
    let kind = request.equipment_type.as_deref().unwrap_or("all");
    if !matches!(
        kind,
        "all" | "camera" | "mount" | "focuser" | "filter_wheel"
    ) {
        return Err("equipment_type must be all, camera, mount, focuser or filter_wheel".into());
    }
    if request
        .driver
        .as_deref()
        .is_some_and(|driver| driver != "alpaca")
    {
        return Err("Only an explicit Alpaca endpoint supports auxiliary discovery; INDI and serial drivers require manual configuration".into());
    }
    let endpoint = request
        .option
        .as_deref()
        .map(validate_alpaca_endpoint)
        .transpose()?;
    let mut report = if matches!(kind, "all" | "camera") {
        discover_cameras(state).await
    } else {
        DiscoveryReport {
            scanned_at: chrono::Utc::now(),
            ..Default::default()
        }
    };
    let wants_mount = matches!(kind, "all" | "mount");
    let wants_focuser = matches!(kind, "all" | "focuser");
    if wants_mount || wants_focuser {
        let result = if let Some(base) = endpoint {
            Some(
                tokio::task::spawn_blocking(move || query_alpaca_devices(&base))
                    .await
                    .map_err(|_| "Alpaca enumeration worker failed".to_string())
                    .and_then(|result| result),
            )
        } else {
            None
        };
        for (wanted, equipment_type) in [(wants_mount, "mount"), (wants_focuser, "focuser")] {
            if !wanted {
                continue;
            }
            let (devices, status, detail) = match &result {
                Some(Ok((mounts, focusers))) => {
                    let devices = if equipment_type == "mount" { mounts.clone() } else { focusers.clone() };
                    let status = if devices.is_empty() { "empty" } else { "reported" };
                    (devices, status, "Configured devices reported by the addressed Alpaca server; physical availability is verified only on connect".to_string())
                }
                Some(Err(error)) => (Vec::new(), "failed", error.clone()),
                None => (Vec::new(), "manual_required", "Supply an explicit Alpaca server URL to list its configured devices. INDI/serial devices require manual configuration; no network broadcast or port scan was run".into()),
            };
            report.device_probes.push(DeviceProbeOutcome {
                equipment_type,
                driver_id: "alpaca",
                status,
                device_count: devices.len(),
                detail,
            });
            if equipment_type == "mount" {
                report.mounts = devices;
            } else {
                report.focusers = devices;
            }
        }
    }
    if matches!(kind, "all" | "filter_wheel") {
        report.device_probes.push(DeviceProbeOutcome {
            equipment_type: "filter_wheel",
            driver_id: "none",
            status: "unavailable",
            device_count: 0,
            detail:
                "Filter-wheel discovery and hardware connection are not implemented in this build"
                    .into(),
        });
    }
    Ok(report)
}

fn validate_alpaca_endpoint(value: &str) -> Result<String, String> {
    let uri: axum::http::Uri = value
        .trim()
        .parse()
        .map_err(|_| "Invalid Alpaca base URL")?;
    if !matches!(uri.scheme_str(), Some("http" | "https"))
        || uri.host().is_none()
        || uri
            .authority()
            .is_some_and(|authority| authority.as_str().contains('@'))
        || uri.path() != "/" && !uri.path().is_empty()
        || uri.query().is_some()
        || value.contains('#')
    {
        return Err("Supply an HTTP(S) Alpaca server base URL without credentials, path, query or device suffix".into());
    }
    Ok(format!(
        "{}://{}",
        uri.scheme_str().unwrap(),
        uri.authority().unwrap()
    ))
}

fn query_alpaca_devices(
    base: &str,
) -> Result<(Vec<DiscoveredDevice>, Vec<DiscoveredDevice>), String> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(PROBE_TIMEOUT))
        .max_redirects(0)
        .build()
        .new_agent();
    let mut response = agent
        .get(&format!("{base}/management/v1/configureddevices"))
        .call()
        .map_err(|error| format!("Alpaca server query failed: {error}"))?;
    let bytes = response
        .body_mut()
        .with_config()
        .limit(1024 * 1024)
        .read_to_vec()
        .map_err(|error| format!("Could not read Alpaca device list: {error}"))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|_| "Alpaca server returned an invalid JSON device list".to_string())?;
    parse_alpaca_devices(base, &value)
}

fn parse_alpaca_devices(
    base: &str,
    value: &serde_json::Value,
) -> Result<(Vec<DiscoveredDevice>, Vec<DiscoveredDevice>), String> {
    if let Some(code) = value.get("ErrorNumber") {
        let code = code
            .as_i64()
            .ok_or("Alpaca server returned an invalid ErrorNumber")?;
        if code != 0 {
            return Err("Alpaca server reported an error while listing configured devices".into());
        }
    }
    let items = value
        .get("Value")
        .and_then(|v| v.as_array())
        .ok_or("Alpaca device list is missing its Value array")?;
    let mut mounts = Vec::new();
    let mut focusers = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for item in items {
        let device_type = item
            .get("DeviceType")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let kind = match device_type.as_str() {
            "telescope" => "mount",
            "focuser" => "focuser",
            _ => continue,
        };
        let index = item
            .get("DeviceNumber")
            .and_then(|v| v.as_i64())
            .and_then(|number| i32::try_from(number).ok())
            .filter(|number| *number >= 0)
            .ok_or("Alpaca server reported an invalid DeviceNumber")?;
        if !seen.insert((kind, index)) {
            return Err(
                "Alpaca server reported duplicate device numbers for one device type".into(),
            );
        }
        let name = item
            .get("DeviceName")
            .and_then(|v| v.as_str())
            .filter(|name| !name.trim().is_empty())
            .ok_or("Alpaca server reported a device without a name")?
            .to_string();
        let device = DiscoveredDevice {
            driver_id: "alpaca",
            driver_label: "ASCOM Alpaca",
            index,
            name,
            selection_key: format!("{kind}:alpaca:{base}:{index}"),
            hardware_id: item
                .get("UniqueID")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            transport: "alpaca",
            availability: "reported",
            option: format!("{base}#device={index}"),
        };
        if kind == "mount" {
            mounts.push(device);
        } else {
            focusers.push(device);
        }
    }
    Ok((mounts, focusers))
}

/// Locally-attached camera drivers worth probing, with the build-capability
/// flag that says whether the SDK was compiled in.
const LOCAL_CAMERA_DRIVERS: &[(&str, &str, u32)] = &[
    ("asi", "ZWO ASI", ols_ffi::OLS_CAP_CAMERA_ASI),
    ("svb", "SVBony", ols_ffi::OLS_CAP_CAMERA_SVB),
    ("toup", "ToupTek", ols_ffi::OLS_CAP_CAMERA_TOUP),
    ("uvc", "UVC / Webcam", ols_ffi::OLS_CAP_CAMERA_UVC),
    ("gphoto2", "DSLR (gphoto2)", ols_ffi::OLS_CAP_CAMERA_GPHOTO2),
];

/// Returns `anyhow::Result` to match `OlsCameraDriver`'s constructors, which
/// use anyhow rather than the daemon's `AchernarError`. The probe only ever
/// stringifies the error for the report, so no conversion is warranted.
fn create_local_driver(driver_id: &str) -> anyhow::Result<OlsCameraDriver> {
    match driver_id {
        "asi" => OlsCameraDriver::new_asi(),
        "svb" => OlsCameraDriver::new_svb(),
        "toup" => OlsCameraDriver::new_toup(),
        "uvc" => OlsCameraDriver::new_uvc(),
        "gphoto2" => OlsCameraDriver::new_gphoto2(),
        other => Err(anyhow::anyhow!(
            "'{other}' is not a locally-probeable camera driver"
        )),
    }
}

/// Marker used to tell "this driver id is not probeable" apart from "the SDK
/// failed to initialize". Both are errors; only the first is a programming bug.
const NOT_PROBEABLE: &str = "is not a locally-probeable camera driver";

/// True when the build has this driver's SDK compiled in.
fn is_compiled(flag: u32) -> bool {
    let flags = unsafe { ols_ffi::ols_driver_capability_flags() as u32 };
    flags & flag != 0
}

/// Probe every local camera driver and report what is attached.
///
/// Refuses to probe while a camera is connected: re-opening a driver that owns
/// the live device can disturb an in-progress capture.
pub async fn discover_cameras(state: &AppState) -> DiscoveryReport {
    let mut notes = vec![
        "Only locally-attached USB/SDK drivers are scanned. Alpaca, INDI and \
         INDIGO are network transports and need an explicit address."
            .to_string(),
    ];

    // Scanning creates SDK driver instances and may claim USB handles. Keep it
    // in the same camera command domain as connect/capture, but do not queue a
    // browser scan behind a multi-minute exposure or autofocus run.
    let acquisition = match state.acquisition.clone().try_lock_owned() {
        Ok(guard) => guard,
        Err(_) => {
            notes.push(
                "Camera control is busy, so no probe was run. Try again after the current \
                 exposure or autofocus operation completes."
                    .to_string(),
            );
            return DiscoveryReport {
                cameras: Vec::new(),
                probes: LOCAL_CAMERA_DRIVERS
                    .iter()
                    .map(|&(id, label, _)| ProbeOutcome {
                        driver_id: id,
                        driver_label: label,
                        status: "skipped",
                        device_count: 0,
                        detail: Some("camera control is busy".to_string()),
                    })
                    .collect(),
                scanned_at: chrono::Utc::now(),
                notes,
                ..Default::default()
            };
        }
    };

    let camera_in_use = state
        .camera
        .lock()
        .map(|guard| guard.is_some())
        .unwrap_or(true);

    if camera_in_use {
        notes.push(
            "A camera is currently connected, so no probe was run. Disconnect it \
             before scanning to avoid interrupting an active capture."
                .to_string(),
        );
        return DiscoveryReport {
            cameras: Vec::new(),
            probes: LOCAL_CAMERA_DRIVERS
                .iter()
                .map(|&(id, label, _)| ProbeOutcome {
                    driver_id: id,
                    driver_label: label,
                    status: "skipped",
                    device_count: 0,
                    detail: Some("a camera is already connected".to_string()),
                })
                .collect(),
            scanned_at: chrono::Utc::now(),
            notes,
            ..Default::default()
        };
    }

    let mut cameras = Vec::new();
    let mut probes = Vec::new();

    // Destructured by value (`&(..)`): iterating a slice yields references, and
    // the report fields are `&'static str`, not `&&str`.
    for (probe_index, &(driver_id, driver_label, flag)) in LOCAL_CAMERA_DRIVERS.iter().enumerate() {
        if !is_compiled(flag) {
            probes.push(ProbeOutcome {
                driver_id,
                driver_label,
                status: "not_compiled",
                device_count: 0,
                detail: Some("SDK not compiled into this daemon build".to_string()),
            });
            continue;
        }

        // The driver holds a raw pointer and is not Send, so it is created,
        // enumerated and dropped entirely inside the blocking closure — only
        // the resulting names cross back.
        let mut probe = tokio::task::spawn_blocking(move || {
            let mut driver = create_local_driver(driver_id)?;
            driver.list_cameras()
        });

        match tokio::time::timeout(PROBE_TIMEOUT, &mut probe).await {
            Ok(Ok(Ok(names))) => {
                for (index, name) in names.iter().enumerate() {
                    cameras.push(DiscoveredCamera {
                        driver_id,
                        driver_label,
                        index: index as i32,
                        name: name.clone(),
                        selection_key: format!("{driver_id}:{index}"),
                        hardware_id: None,
                        transport: "local",
                    });
                }
                probes.push(ProbeOutcome {
                    driver_id,
                    driver_label,
                    status: if names.is_empty() { "empty" } else { "found" },
                    device_count: names.len(),
                    detail: None,
                });
            }
            Ok(Ok(Err(err))) => {
                // A driver that errors is normal when no device of that brand is
                // attached; log at debug and carry on to the next driver.
                tracing::debug!("discovery: {driver_id} probe failed: {err}");
                probes.push(ProbeOutcome {
                    driver_id,
                    driver_label,
                    status: "failed",
                    device_count: 0,
                    detail: Some(err.to_string()),
                });
            }
            Ok(Err(join_err)) => {
                tracing::warn!("discovery: {driver_id} probe panicked: {join_err}");
                probes.push(ProbeOutcome {
                    driver_id,
                    driver_label,
                    status: "failed",
                    device_count: 0,
                    detail: Some("probe task panicked".to_string()),
                });
            }
            Err(_) => {
                tracing::warn!("discovery: {driver_id} probe timed out");
                probes.push(ProbeOutcome {
                    driver_id,
                    driver_label,
                    status: "timeout",
                    device_count: 0,
                    detail: Some(format!(
                        "no response within {}s; the device may be wedged",
                        PROBE_TIMEOUT.as_secs()
                    )),
                });

                // A blocking FFI call cannot be cancelled. Keep camera
                // ownership until the abandoned SDK probe really exits and do
                // not start more SDKs after it. This may leave discovery busy
                // until daemon restart if a vendor SDK is permanently wedged,
                // which is safer than opening the same device concurrently.
                for &(remaining_id, remaining_label, _) in &LOCAL_CAMERA_DRIVERS[probe_index + 1..]
                {
                    probes.push(ProbeOutcome {
                        driver_id: remaining_id,
                        driver_label: remaining_label,
                        status: "skipped",
                        device_count: 0,
                        detail: Some(format!("scan stopped after the {driver_id} SDK timed out")),
                    });
                }
                notes.push(format!(
                    "The {driver_label} SDK did not return. Camera control remains busy until \
                     that probe exits; restart the daemon if it is permanently wedged."
                ));
                tokio::spawn(async move {
                    let _acquisition = acquisition;
                    let _ = probe.await;
                });
                return DiscoveryReport {
                    cameras,
                    probes,
                    scanned_at: chrono::Utc::now(),
                    notes,
                    ..Default::default()
                };
            }
        }
    }

    if cameras.is_empty() {
        notes.push(
            "No cameras found. Check the USB connection and that the camera is \
             powered, then scan again."
                .to_string(),
        );
    }

    DiscoveryReport {
        cameras,
        probes,
        scanned_at: chrono::Utc::now(),
        notes,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_endpoint_rejects_auto_paths_credentials_and_selectors() {
        assert_eq!(
            validate_alpaca_endpoint(" http://127.0.0.1:11111/ ").unwrap(),
            "http://127.0.0.1:11111"
        );
        assert_eq!(
            validate_alpaca_endpoint("https://[::1]:11111").unwrap(),
            "https://[::1]:11111"
        );
        for endpoint in [
            "auto",
            "localhost:11111",
            "ftp://localhost",
            "http://user:secret@localhost",
            "http://localhost/api/v1",
            "http://localhost?scan=all",
            "http://localhost#device=7",
        ] {
            assert!(validate_alpaca_endpoint(endpoint).is_err(), "{endpoint}");
        }
    }

    #[test]
    fn management_list_preserves_device_numbers_and_connection_options() {
        let value = serde_json::json!({"Value": [
            {"DeviceType":"Telescope", "DeviceName":"First mount", "DeviceNumber":3, "UniqueID":"mount-a"},
            {"DeviceType":"Camera", "DeviceName":"Not an auxiliary device", "DeviceNumber":9},
            {"DeviceType":"Focuser", "DeviceName":"Focus unit", "DeviceNumber":7},
            {"DeviceType":"TELESCOPE", "DeviceName":"Second mount", "DeviceNumber":0}
        ]});
        let (mounts, focusers) = parse_alpaca_devices("http://localhost:11111", &value).unwrap();
        assert_eq!(
            mounts.iter().map(|device| device.index).collect::<Vec<_>>(),
            vec![3, 0]
        );
        assert_eq!(focusers.len(), 1);
        assert_eq!(focusers[0].index, 7);
        assert_eq!(focusers[0].option, "http://localhost:11111#device=7");
        assert_eq!(mounts[0].hardware_id.as_deref(), Some("mount-a"));
        assert_eq!(mounts[0].availability, "reported");
        assert_ne!(mounts[0].selection_key, mounts[1].selection_key);
        let json = serde_json::to_value(&focusers[0]).unwrap();
        assert_eq!(json["driver_id"], "alpaca");
        assert_eq!(json["transport"], "alpaca");
        assert_eq!(json["option"], "http://localhost:11111#device=7");
    }

    #[test]
    fn management_list_rejects_ambiguous_or_invalid_device_numbers() {
        for number in [
            serde_json::json!(-1),
            serde_json::json!(2147483648_i64),
            serde_json::json!(1.5),
            serde_json::json!("1"),
            serde_json::Value::Null,
        ] {
            let value = serde_json::json!({"Value":[
                {"DeviceType":"Telescope", "DeviceName":"Mount", "DeviceNumber":number}
            ]});
            assert!(parse_alpaca_devices("http://localhost", &value).is_err());
        }
        let duplicate = serde_json::json!({"Value":[
            {"DeviceType":"Telescope", "DeviceName":"A", "DeviceNumber":2},
            {"DeviceType":"Telescope", "DeviceName":"B", "DeviceNumber":2}
        ]});
        assert!(parse_alpaca_devices("http://localhost", &duplicate).is_err());
        // Device numbers are scoped by type, so mount 2 and focuser 2 are valid.
        let distinct_types = serde_json::json!({"Value":[
            {"DeviceType":"Telescope", "DeviceName":"Mount", "DeviceNumber":2},
            {"DeviceType":"Focuser", "DeviceName":"Focus", "DeviceNumber":2}
        ]});
        assert!(parse_alpaca_devices("http://localhost", &distinct_types).is_ok());
    }

    #[test]
    fn management_list_never_fabricates_devices_from_bad_or_unrelated_responses() {
        for response in [
            serde_json::json!({}),
            serde_json::json!({"Value":{}}),
            serde_json::json!({"ErrorNumber":1024,"Value":[]}),
            serde_json::json!({"ErrorNumber":"invalid","Value":[]}),
            serde_json::json!({"Value":[{"DeviceType":"Focuser","DeviceNumber":0}]}),
        ] {
            assert!(parse_alpaca_devices("http://localhost", &response).is_err());
        }
        let (mounts, focusers) = parse_alpaca_devices(
            "http://localhost",
            &serde_json::json!({"Value":[
                {"DeviceType":"FilterWheel", "DeviceName":"Wheel", "DeviceNumber":0}
            ]}),
        )
        .unwrap();
        assert!(mounts.is_empty() && focusers.is_empty());
    }

    #[test]
    fn discovery_request_rejects_unknown_fields_and_wrong_types() {
        assert!(serde_json::from_value::<DiscoveryRequest>(serde_json::json!({})).is_ok());
        assert!(serde_json::from_value::<DiscoveryRequest>(
            serde_json::json!({"equipment_type":"mount", "option":"http://localhost"})
        )
        .is_ok());
        assert!(
            serde_json::from_value::<DiscoveryRequest>(serde_json::json!({"optino":"auto"}))
                .is_err()
        );
        assert!(
            serde_json::from_value::<DiscoveryRequest>(serde_json::json!({"option":123})).is_err()
        );
    }

    #[test]
    fn explicit_server_query_only_reads_its_management_endpoint() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];
            while !request.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
                let count = stream.read(&mut buffer).unwrap();
                assert!(count > 0 && request.len() < 16 * 1024);
                request.extend_from_slice(&buffer[..count]);
            }
            assert!(String::from_utf8(request)
                .unwrap()
                .starts_with("GET /management/v1/configureddevices HTTP/1.1\r\n"));
            let body = r#"{"Value":[{"DeviceType":"Telescope","DeviceName":"Reported mount","DeviceNumber":4}]}"#;
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
        });
        let (mounts, focusers) = query_alpaca_devices(&format!("http://{address}")).unwrap();
        server.join().unwrap();
        assert_eq!(mounts[0].index, 4);
        assert!(focusers.is_empty());
    }

    #[test]
    fn rejects_non_local_drivers() {
        // Network and pseudo-drivers must not be probeable through this path —
        // scanning them would mean broadcasting or port-scanning the LAN.
        for id in ["alpaca", "indi", "indigo", "sim", "wdir", "nonsense"] {
            let err = create_local_driver(id)
                .err()
                .unwrap_or_else(|| panic!("'{id}' must not be locally probeable"));
            assert!(
                err.to_string().contains(NOT_PROBEABLE),
                "'{id}' must be rejected as unprobeable, got: {err}"
            );
        }
    }

    #[test]
    fn local_driver_table_is_consistent() {
        // Every id in the probe table must be constructible, or a scan would
        // report a permanent phantom failure for it.
        for &(id, label, _) in LOCAL_CAMERA_DRIVERS {
            assert!(!id.is_empty() && !label.is_empty());
            // Creating the driver may legitimately fail here (no SDK, no
            // hardware, no CI USB bus). What must NOT happen is the id being
            // unrecognized — that would make the scan report a permanent
            // phantom failure for a driver it advertises.
            if let Err(err) = create_local_driver(id) {
                assert!(
                    !err.to_string().contains(NOT_PROBEABLE),
                    "probe table lists '{id}' but create_local_driver does not recognize it"
                );
            }
        }
    }

    #[test]
    fn report_serializes_for_the_frontend() {
        let report = DiscoveryReport {
            cameras: vec![DiscoveredCamera {
                driver_id: "svb",
                driver_label: "SVBony",
                index: 0,
                name: "SVBONY SV105".to_string(),
                selection_key: "svb:0".to_string(),
                hardware_id: None,
                transport: "local",
            }],
            probes: vec![ProbeOutcome {
                driver_id: "asi",
                driver_label: "ZWO ASI",
                status: "empty",
                device_count: 0,
                detail: None,
            }],
            scanned_at: chrono::Utc::now(),
            notes: vec!["note".to_string()],
            ..Default::default()
        };
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["cameras"][0]["driver_id"], "svb");
        assert_eq!(json["cameras"][0]["index"], 0);
        assert_eq!(json["probes"][0]["status"], "empty");
        // `detail: None` must be omitted, not serialized as null.
        assert!(json["probes"][0].get("detail").is_none());
    }
}

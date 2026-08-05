//! Device autodiscovery — user-triggered "scan for devices".
//!
//! `driver_capability_report()` answers "which drivers were compiled into this
//! build?". This module answers the different question "what is actually
//! plugged in *right now*?" by instantiating each locally-attached driver and
//! calling its device enumeration.
//!
//! # Deliberate scope: local buses only
//!
//! Only USB/SDK drivers are probed (ASI, SVBony, ToupTek, UVC, gphoto2).
//! Network transports are **not** scanned here:
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

use serde::Serialize;

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

#[derive(Debug, Clone, Serialize)]
pub struct DiscoveryReport {
    pub cameras: Vec<DiscoveredCamera>,
    pub probes: Vec<ProbeOutcome>,
    pub scanned_at: chrono::DateTime<chrono::Utc>,
    /// Human-readable caveats (what was deliberately not scanned, and why).
    pub notes: Vec<String>,
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
        };
    }

    let mut cameras = Vec::new();
    let mut probes = Vec::new();

    // Destructured by value (`&(..)`): iterating a slice yields references, and
    // the report fields are `&'static str`, not `&&str`.
    for &(driver_id, driver_label, flag) in LOCAL_CAMERA_DRIVERS {
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
        let probe = tokio::task::spawn_blocking(move || {
            let mut driver = create_local_driver(driver_id)?;
            driver.list_cameras()
        });

        match tokio::time::timeout(PROBE_TIMEOUT, probe).await {
            Ok(Ok(Ok(names))) => {
                for (index, name) in names.iter().enumerate() {
                    cameras.push(DiscoveredCamera {
                        driver_id,
                        driver_label,
                        index: index as i32,
                        name: name.clone(),
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        };
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["cameras"][0]["driver_id"], "svb");
        assert_eq!(json["cameras"][0]["index"], 0);
        assert_eq!(json["probes"][0]["status"], "empty");
        // `detail: None` must be omitted, not serialized as null.
        assert!(json["probes"][0].get("detail").is_none());
    }
}

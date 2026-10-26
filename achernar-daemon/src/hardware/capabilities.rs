//! Build-time hardware capability reporting (`GET /drivers`, spec §2.12,
//! D21).
//!
//! Cameras come from the OLS bridge's compile flags. Every other kind comes
//! from the equipment layer's real implementation: `enabled` and `reason`
//! say what this build can connect, and `qualification` says how far the
//! driver has been proven (`planned`, `implemented_unqualified`,
//! `qualified`). Only a rig acceptance record makes a device identity
//! `qualified` (§5.7, WP1B-6), so every implemented driver here is
//! `implemented_unqualified`. The hidden legacy ids `ols_alpaca` and
//! `ols_indi` stay connectable but are not listed.

use serde::Serialize;
use std::ffi::CStr;

use super::ols_ffi;

#[derive(Debug, Clone, Serialize)]
pub struct DriverCapabilityReport {
    pub bridge_version: String,
    pub camera: Vec<DriverCapability>,
    pub mount: Vec<DriverCapability>,
    pub focuser: Vec<DriverCapability>,
    pub filter_wheel: Vec<DriverCapability>,
    pub guider: Vec<DriverCapability>,
    pub rotator: Vec<DriverCapability>,
    pub cover_calibrator: Vec<DriverCapability>,
    pub switch: Vec<DriverCapability>,
    pub weather: Vec<DriverCapability>,
    pub safety_monitor: Vec<DriverCapability>,
    pub dome: Vec<DriverCapability>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DriverCapability {
    pub id: &'static str,
    pub label: &'static str,
    pub compiled: bool,
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub option_hint: Option<&'static str>,
    /// D21: `planned`, `implemented_unqualified` or `qualified`.
    pub qualification: &'static str,
}

const UNQUALIFIED: &str = "implemented_unqualified";
const PLANNED: &str = "planned";

/// A native equipment-layer driver that this build implements.
fn native_driver(
    id: &'static str,
    label: &'static str,
    reason: &'static str,
    option_hint: Option<&'static str>,
) -> DriverCapability {
    DriverCapability {
        id,
        label,
        compiled: true,
        enabled: true,
        reason: Some(reason),
        option_hint,
        qualification: UNQUALIFIED,
    }
}

/// A driver that is planned for a later tier, or reached through another
/// path (O4): listed so the UI can explain it, never enabled.
fn planned(id: &'static str, label: &'static str, reason: &'static str) -> DriverCapability {
    DriverCapability {
        id,
        label,
        compiled: false,
        enabled: false,
        reason: Some(reason),
        option_hint: None,
        qualification: PLANNED,
    }
}

const ALPACA_HINT: Option<&str> = Some("http(s)://host:port#device=N");
const ALPACA_NATIVE: &str = "Native Alpaca client; needs a reachable Alpaca server and device. \
                             An empty or \"auto\" option is refused unless \
                             [equipment] allow_broadcast_discovery = true.";
const INDI_TIER2: &str = "The native INDI client is planned for Tier 2.";
const TIER2_KIND: &str = "This device class is planned for Tier 2 (Alpaca and simulator).";

fn alpaca_and_sim(sim_reason: &'static str) -> Vec<DriverCapability> {
    vec![
        native_driver("alpaca", "Alpaca", ALPACA_NATIVE, ALPACA_HINT),
        native_driver("sim", "Simulator", sim_reason, None),
        planned("indi", "INDI", INDI_TIER2),
    ]
}

fn tier2_kind() -> Vec<DriverCapability> {
    vec![
        planned("alpaca", "Alpaca", TIER2_KIND),
        planned("sim", "Simulator", TIER2_KIND),
    ]
}

pub fn driver_capability_report() -> DriverCapabilityReport {
    let flags = native_flags();
    let indi = has(flags, ols_ffi::OLS_CAP_CAMERA_INDI);

    DriverCapabilityReport {
        bridge_version: bridge_version(),
        camera: vec![
            available(
                "sim",
                "Simulator",
                "Built into the daemon for smoke tests and development.",
                Some("sim data directory"),
            ),
            available(
                "alpaca",
                "Alpaca",
                "Built into the OLS bridge; requires a reachable Alpaca server/device.",
                Some("Alpaca base URL"),
            ),
            native(
                "asi",
                "ZWO ASI",
                has(flags, ols_ffi::OLS_CAP_CAMERA_ASI),
                "ASI SDK support was not compiled into this daemon build.",
            ),
            native(
                "svb",
                "SVBony",
                has(flags, ols_ffi::OLS_CAP_CAMERA_SVB),
                "SVBony SDK support was not compiled into this daemon build.",
            ),
            native(
                "toup",
                "ToupTek",
                has(flags, ols_ffi::OLS_CAP_CAMERA_TOUP),
                "ToupTek SDK support was not compiled into this daemon build.",
            ),
            native(
                "uvc",
                "UVC / Webcam",
                has(flags, ols_ffi::OLS_CAP_CAMERA_UVC),
                "libuvc support was not compiled into this daemon build.",
            ),
            native_with_hint(
                "gphoto2",
                "DSLR (gphoto2)",
                has(flags, ols_ffi::OLS_CAP_CAMERA_GPHOTO2),
                "libgphoto2 support was not compiled into this daemon build.",
                None,
            ),
            native_with_hint(
                "indi",
                "INDI",
                indi,
                "libindi support was not compiled into this daemon build.",
                Some("INDI server/device name"),
            ),
            native(
                "indigo",
                "INDIGO",
                has(flags, ols_ffi::OLS_CAP_CAMERA_INDIGO),
                "INDIGO support was not compiled into this daemon build.",
            ),
            native_with_hint(
                "wdir",
                "Watch Folder",
                has(flags, ols_ffi::OLS_CAP_CAMERA_WDIR),
                "Watch-folder camera support is only compiled on Linux.",
                Some("directory path"),
            ),
        ],
        mount: vec![
            native_driver("alpaca", "Alpaca", ALPACA_NATIVE, ALPACA_HINT),
            native_driver(
                "sim",
                "Simulator",
                "In-process kinematic mount simulator (ESP32-like profile).",
                None,
            ),
            DriverCapability {
                id: "indi",
                label: "INDI",
                compiled: indi,
                enabled: indi,
                reason: Some(if indi {
                    "Interim OLS indi:remote adapter (O3); the native INDI client is Tier 2."
                } else {
                    "libindi support was not compiled into this daemon build, and the native \
                     INDI client is planned for Tier 2."
                }),
                option_hint: Some("indiserver host:port (default localhost:7624)"),
                qualification: if indi { UNQUALIFIED } else { PLANNED },
            },
        ],
        focuser: vec![
            native_driver("alpaca", "Alpaca", ALPACA_NATIVE, ALPACA_HINT),
            native_driver(
                "sim",
                "Simulator",
                "In-process focuser simulator; the option is the initial position.",
                Some("initial position"),
            ),
            planned("indi", "INDI", INDI_TIER2),
            planned(
                "moonlite",
                "MoonLite",
                "Reached through indi_moonlite (native INDI client, Tier 2) or an ASCOM driver \
                 behind ASCOM Remote (O4); there is no native serial driver.",
            ),
            planned(
                "usb",
                "USB Serial",
                "Serial focusers are reached through their INDI driver (Tier 2) or ASCOM Remote \
                 (O4); there is no native serial driver.",
            ),
        ],
        filter_wheel: alpaca_and_sim("In-process filter-wheel simulator."),
        guider: vec![
            native_driver(
                "phd2",
                "PHD2",
                "PHD2's event server; PHD2 owns the guide camera and the guide loop (O6).",
                Some("host:port (instance n listens on 4400 + n - 1)"),
            ),
            planned(
                "direct",
                "Direct guider",
                "Mount pulse guiding without PHD2 is planned for Tier 2 (O8).",
            ),
        ],
        rotator: tier2_kind(),
        cover_calibrator: tier2_kind(),
        switch: tier2_kind(),
        weather: alpaca_and_sim("In-process weather simulator."),
        safety_monitor: alpaca_and_sim("In-process safety-monitor simulator."),
        dome: tier2_kind(),
    }
}

fn native_flags() -> u32 {
    unsafe { ols_ffi::ols_driver_capability_flags() as u32 }
}

fn bridge_version() -> String {
    unsafe {
        let ptr = ols_ffi::ols_bridge_version();
        if ptr.is_null() {
            "unknown".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

fn has(flags: u32, flag: u32) -> bool {
    flags & flag != 0
}

fn available(
    id: &'static str,
    label: &'static str,
    reason: &'static str,
    option_hint: Option<&'static str>,
) -> DriverCapability {
    DriverCapability {
        id,
        label,
        compiled: true,
        enabled: true,
        reason: Some(reason),
        option_hint,
        qualification: UNQUALIFIED,
    }
}

fn native(
    id: &'static str,
    label: &'static str,
    compiled: bool,
    missing_reason: &'static str,
) -> DriverCapability {
    native_with_hint(id, label, compiled, missing_reason, None)
}

fn native_with_hint(
    id: &'static str,
    label: &'static str,
    compiled: bool,
    missing_reason: &'static str,
    option_hint: Option<&'static str>,
) -> DriverCapability {
    DriverCapability {
        id,
        label,
        compiled,
        enabled: compiled,
        reason: Some(if compiled {
            "Compiled into this daemon build; real hardware may still be absent."
        } else {
            missing_reason
        }),
        option_hint,
        qualification: UNQUALIFIED,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_is_reported_with_its_qualification() {
        let report = serde_json::to_value(driver_capability_report()).unwrap();
        for key in [
            "camera",
            "mount",
            "focuser",
            "filter_wheel",
            "guider",
            "rotator",
            "cover_calibrator",
            "switch",
            "weather",
            "safety_monitor",
            "dome",
        ] {
            let entries = report[key].as_array().unwrap_or_else(|| panic!("{key}"));
            assert!(!entries.is_empty(), "{key} lists no driver");
            for entry in entries {
                let q = entry["qualification"].as_str().unwrap();
                // Nothing is qualified without a rig acceptance record (D21).
                assert!(matches!(q, "planned" | "implemented_unqualified"), "{q}");
                if q == "planned" {
                    assert_eq!(entry["enabled"], false, "{key}/{}", entry["id"]);
                }
            }
        }
    }

    #[test]
    fn the_indi_focuser_is_no_longer_advertised_as_enabled() {
        let report = driver_capability_report();
        for (kind, entries) in [
            ("focuser", &report.focuser),
            ("filter_wheel", &report.filter_wheel),
            ("weather", &report.weather),
        ] {
            let indi = entries.iter().find(|d| d.id == "indi").unwrap();
            assert!(!indi.enabled, "{kind} INDI is Tier 2");
            assert_eq!(indi.qualification, "planned");
        }
        let alpaca = report.mount.iter().find(|d| d.id == "alpaca").unwrap();
        assert!(alpaca.enabled && alpaca.reason.unwrap().contains("Native Alpaca"));
        assert!(report.guider.iter().any(|d| d.id == "phd2" && d.enabled));
        assert!(report.rotator.iter().all(|d| !d.enabled));
        // The legacy OLS adapters are hidden ids (§2.12).
        assert!(report.mount.iter().all(|d| !d.id.starts_with("ols_")));
    }
}

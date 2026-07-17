//! Build-time hardware capability reporting for the OLS bridge.

use serde::Serialize;
use std::ffi::CStr;

use super::ols_ffi;

#[derive(Debug, Clone, Serialize)]
pub struct DriverCapabilityReport {
    pub bridge_version: String,
    pub camera: Vec<DriverCapability>,
    pub mount: Vec<DriverCapability>,
    pub focuser: Vec<DriverCapability>,
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
            available(
                "alpaca",
                "Alpaca",
                "Built into the OLS bridge; requires a reachable Alpaca server/device.",
                Some("Alpaca base URL"),
            ),
            available(
                "sim",
                "Simulator",
                "Uses the Alpaca-compatible simulator path.",
                Some("Alpaca simulator URL"),
            ),
            native_with_hint(
                "indi",
                "INDI",
                indi,
                "libindi support was not compiled into this daemon build.",
                Some("INDI telescope device"),
            ),
        ],
        focuser: vec![
            available(
                "alpaca",
                "Alpaca",
                "Built into the OLS bridge; requires a reachable Alpaca server/device.",
                Some("Alpaca base URL"),
            ),
            available(
                "sim",
                "Simulator",
                "Daemon-local in-memory simulator for smoke tests and development.",
                Some("initial position"),
            ),
            native_with_hint(
                "indi",
                "INDI",
                indi,
                "libindi support was not compiled into this daemon build.",
                Some("INDI focuser device"),
            ),
            disabled(
                "moonlite",
                "MoonLite",
                "MoonLite focuser support is exposed in Rust but not compiled through the OLS bridge yet.",
            ),
            disabled(
                "usb",
                "USB Serial",
                "USB serial focuser support is exposed in Rust but not compiled through the OLS bridge yet.",
            ),
        ],
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
    }
}

fn disabled(id: &'static str, label: &'static str, reason: &'static str) -> DriverCapability {
    DriverCapability {
        id,
        label,
        compiled: false,
        enabled: false,
        reason: Some(reason),
        option_hint: None,
    }
}

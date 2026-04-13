//! Raw FFI bindings to the OLS bridge C API.
//!
//! These are unsafe C function declarations that map 1:1 to the
//! functions defined in `ols-bridge/include/ols_bridge.h`.
//! Safe wrappers live in sibling modules (camera.rs, mount.rs, focuser.rs).

#![allow(non_camel_case_types)]
#![allow(dead_code)]

use std::os::raw::{c_char, c_double, c_float, c_int, c_void};

// ═══════════════════════════════════════════════════════════════════
//  Error Constants
// ═══════════════════════════════════════════════════════════════════

pub const OLS_OK: c_int = 0;
pub const OLS_ERR_GENERIC: c_int = 1;
pub const OLS_ERR_NOT_FOUND: c_int = 2;
pub const OLS_ERR_INVALID: c_int = 3;
pub const OLS_ERR_IO: c_int = 4;
pub const OLS_ERR_NOT_CONNECTED: c_int = 5;

// ═══════════════════════════════════════════════════════════════════
//  Stream / Pixel Format Enums
// ═══════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OlsStreamType {
    Yuv2   = 0,
    Mjpeg  = 1,
    Rgb24  = 2,
    Rgb48  = 3,
    Raw8   = 4,
    Raw16  = 5,
    Mono8  = 6,
    Mono16 = 7,
    Error  = 8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OlsBayerType {
    Na = 0,
    Rg = 1,
    Bg = 2,
    Gr = 3,
    Gb = 4,
}

// ═══════════════════════════════════════════════════════════════════
//  Camera Structures
// ═══════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OlsStreamFormat {
    pub format: OlsStreamType,
    pub width: i32,
    pub height: i32,
    pub bin: i32,
    pub framerate: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OlsFrame {
    pub format: OlsStreamType,
    pub bayer: OlsBayerType,
    pub frame_counter: i32,
    pub unix_timestamp: f64,
    pub width: i32,
    pub height: i32,
    pub data: *const u8,
    pub data_size: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OlsCamOptionId {
    AutoExp      = 0,
    AutoWb       = 1,
    AutoFocus    = 2,
    Exp          = 3,
    Wb           = 4,
    WbR          = 5,
    WbB          = 6,
    Focus        = 7,
    Gain         = 8,
    Gamma        = 9,
    Brightness   = 10,
    Contrast     = 11,
    Temperature  = 12,
    CoolerTarget = 13,
    CoolerOn     = 14,
    FanOn        = 15,
    CoolerPowerPerc = 16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OlsCamOptionType {
    Bool      = 0,
    Number    = 1,
    Msec      = 2,
    Percent   = 3,
    Kelvin    = 4,
    Celsius   = 5,
    Selection = 6,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OlsCamParam {
    pub option: OlsCamOptionId,
    pub option_type: OlsCamOptionType,
    pub read_only: i32,
    pub step_size: f64,
    pub min_val: f64,
    pub max_val: f64,
    pub def_val: f64,
    pub cur_val: f64,
}

pub type OlsFrameCallback =
    unsafe extern "C" fn(frame: *const OlsFrame, user_data: *mut c_void);

// ═══════════════════════════════════════════════════════════════════
//  Mount Types
// ═══════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OlsMountProto {
    None   = 0,
    Serial = 1,
    Inet   = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OlsMountTracking {
    Sidereal = 0,
    Solar    = 1,
    Lunar    = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OlsMountSlew {
    Stop = 0,
    N    = 2,   // 1<<1
    S    = 6,   // 3<<1
    W    = 8,   // 1<<3
    E    = 24,  // 3<<3
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OlsMeridianFlip {
    Unsupported = 0,
    Flip        = 1,
    Stop        = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OlsEqCoord {
    pub ra: f64,   // hours
    pub dec: f64,  // degrees
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OlsAltAzCoord {
    pub alt: f64,  // degrees
    pub az: f64,   // degrees
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct OlsMountDriverInfo {
    pub name: [c_char; 128],
    pub option: [c_char; 128],
    pub has_option: i32,
}

// ═══════════════════════════════════════════════════════════════════
//  Focuser Types
// ═══════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Debug, Clone)]
pub struct OlsFocuserDriverInfo {
    pub name: [c_char; 128],
    pub option: [c_char; 128],
}

// ═══════════════════════════════════════════════════════════════════
//  Opaque Handle Types
// ═══════════════════════════════════════════════════════════════════

#[repr(C)]
pub struct OlsCameraDriver {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OlsCamera {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OlsMountInterface {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OlsFocuserInterface {
    _private: [u8; 0],
}

// ═══════════════════════════════════════════════════════════════════
//  Extern C Declarations
// ═══════════════════════════════════════════════════════════════════

extern "C" {
    // ─── Error ───
    pub fn ols_last_error() -> *const c_char;

    // ─── Version ───
    pub fn ols_bridge_version() -> *const c_char;

    // ═══ Camera Driver API ═══
    pub fn ols_sim_driver_create(
        sim_data_dir: *const c_char,
        out_driver: *mut *mut OlsCameraDriver,
    ) -> c_int;

    pub fn ols_alpaca_camera_driver_create(
        url: *const c_char,
        out_driver: *mut *mut OlsCameraDriver,
    ) -> c_int;

    pub fn ols_asi_camera_driver_create(
        out_driver: *mut *mut OlsCameraDriver,
    ) -> c_int;

    pub fn ols_svb_camera_driver_create(
        out_driver: *mut *mut OlsCameraDriver,
    ) -> c_int;

    pub fn ols_toup_camera_driver_create(
        out_driver: *mut *mut OlsCameraDriver,
    ) -> c_int;

    pub fn ols_driver_list_cameras(
        driver: *mut OlsCameraDriver,
        out_count: *mut i32,
    ) -> c_int;

    pub fn ols_driver_camera_name(
        driver: *mut OlsCameraDriver,
        index: i32,
        name_buf: *mut c_char,
        name_buf_len: i32,
    ) -> c_int;

    pub fn ols_driver_open_camera(
        driver: *mut OlsCameraDriver,
        index: i32,
        out_camera: *mut *mut OlsCamera,
    ) -> c_int;

    pub fn ols_driver_destroy(driver: *mut OlsCameraDriver);

    // ═══ Camera Instance API ═══
    pub fn ols_camera_name(
        cam: *mut OlsCamera,
        name_buf: *mut c_char,
        name_buf_len: i32,
    ) -> c_int;

    pub fn ols_camera_format_count(cam: *mut OlsCamera, out_count: *mut i32) -> c_int;

    pub fn ols_camera_format_get(
        cam: *mut OlsCamera,
        index: i32,
        out_fmt: *mut OlsStreamFormat,
    ) -> c_int;

    pub fn ols_camera_start_stream(
        cam: *mut OlsCamera,
        fmt: *const OlsStreamFormat,
        callback: OlsFrameCallback,
        user_data: *mut c_void,
    ) -> c_int;

    pub fn ols_camera_stop_stream(cam: *mut OlsCamera) -> c_int;

    pub fn ols_camera_option_count(cam: *mut OlsCamera, out_count: *mut i32) -> c_int;

    pub fn ols_camera_option_id(
        cam: *mut OlsCamera,
        index: i32,
        out_id: *mut OlsCamOptionId,
    ) -> c_int;

    pub fn ols_camera_get_param(
        cam: *mut OlsCamera,
        id: OlsCamOptionId,
        out_param: *mut OlsCamParam,
    ) -> c_int;

    pub fn ols_camera_set_param(
        cam: *mut OlsCamera,
        id: OlsCamOptionId,
        value: c_double,
    ) -> c_int;

    pub fn ols_camera_destroy(cam: *mut OlsCamera);

    // ═══ Mount API ═══
    pub fn ols_mount_interface_create(out_mi: *mut *mut OlsMountInterface) -> c_int;

    pub fn ols_mount_driver_list(
        mi: *mut OlsMountInterface,
        out_drivers: *mut OlsMountDriverInfo,
        max_drivers: i32,
        out_count: *mut i32,
    ) -> c_int;

    pub fn ols_mount_load_driver(
        mi: *mut OlsMountInterface,
        driver_name: *const c_char,
        option: *const c_char,
    ) -> c_int;

    pub fn ols_mount_load_mount(mi: *mut OlsMountInterface) -> c_int;

    pub fn ols_mount_driver_loaded(mi: *mut OlsMountInterface) -> i32;

    pub fn ols_mount_is_loaded(mi: *mut OlsMountInterface) -> i32;

    pub fn ols_mount_connect(mi: *mut OlsMountInterface, load_alignment: i32) -> c_int;

    pub fn ols_mount_connected(mi: *mut OlsMountInterface) -> i32;

    pub fn ols_mount_get_current(
        mi: *mut OlsMountInterface,
        out_coord: *mut OlsEqCoord,
    ) -> c_int;

    pub fn ols_mount_goto(
        mi: *mut OlsMountInterface,
        ra: c_double,
        dec: c_double,
    ) -> c_int;

    pub fn ols_mount_sync(
        mi: *mut OlsMountInterface,
        ra: c_double,
        dec: c_double,
    ) -> c_int;

    pub fn ols_mount_slew(
        mi: *mut OlsMountInterface,
        direction: OlsMountSlew,
        speed: i32,
    ) -> c_int;

    pub fn ols_mount_get_tracking(
        mi: *mut OlsMountInterface,
        out_mode: *mut OlsMountTracking,
    ) -> c_int;

    pub fn ols_mount_get_tracking_state(
        mi: *mut OlsMountInterface,
        out_state: *mut i32,
    ) -> c_int;

    pub fn ols_mount_set_tracking_state(
        mi: *mut OlsMountInterface,
        track: i32,
    ) -> c_int;

    pub fn ols_mount_set_tracking_mode(
        mi: *mut OlsMountInterface,
        mode: OlsMountTracking,
    ) -> c_int;

    pub fn ols_mount_set_lat_lon(
        mi: *mut OlsMountInterface,
        lat: c_double,
        lon: c_double,
    ) -> c_int;

    pub fn ols_mount_get_lat_lon(
        mi: *mut OlsMountInterface,
        out_lat: *mut c_double,
        out_lon: *mut c_double,
    ) -> c_int;

    pub fn ols_mount_get_slew_rate(
        mi: *mut OlsMountInterface,
        out_current: *mut i32,
        out_count: *mut i32,
    ) -> c_int;

    pub fn ols_mount_pulse_guide_supported(
        mi: *mut OlsMountInterface,
        out_supported: *mut i32,
        out_ns_rate: *mut c_float,
        out_we_rate: *mut c_float,
    ) -> c_int;

    pub fn ols_mount_pulse_guide(
        mi: *mut OlsMountInterface,
        ns_ms: c_float,
        we_ms: c_float,
    ) -> c_int;

    pub fn ols_mount_get_meridian(
        mi: *mut OlsMountInterface,
        out_flip: *mut OlsMeridianFlip,
    ) -> c_int;

    pub fn ols_mount_set_meridian(
        mi: *mut OlsMountInterface,
        flip: OlsMeridianFlip,
    ) -> c_int;

    pub fn ols_mount_get_alt_limits(
        mi: *mut OlsMountInterface,
        out_low: *mut i32,
        out_high: *mut i32,
    ) -> c_int;

    pub fn ols_mount_set_alt_limits(
        mi: *mut OlsMountInterface,
        low: i32,
        high: i32,
    ) -> c_int;

    pub fn ols_mount_shutdown(mi: *mut OlsMountInterface) -> c_int;

    pub fn ols_mount_interface_destroy(mi: *mut OlsMountInterface);

    // ═══ Focuser API ═══
    pub fn ols_focuser_interface_create(out_fi: *mut *mut OlsFocuserInterface) -> c_int;

    pub fn ols_focuser_driver_list(
        fi: *mut OlsFocuserInterface,
        out_drivers: *mut OlsFocuserDriverInfo,
        max_drivers: i32,
        out_count: *mut i32,
    ) -> c_int;

    pub fn ols_focuser_load_driver(
        fi: *mut OlsFocuserInterface,
        driver_name: *const c_char,
        option: *const c_char,
    ) -> c_int;

    pub fn ols_focuser_is_loaded(fi: *mut OlsFocuserInterface) -> i32;

    pub fn ols_focuser_position(
        fi: *mut OlsFocuserInterface,
        out_position: *mut i32,
    ) -> c_int;

    pub fn ols_focuser_move_to(fi: *mut OlsFocuserInterface, position: i32) -> c_int;

    pub fn ols_focuser_move_by(fi: *mut OlsFocuserInterface, delta: i32) -> c_int;

    pub fn ols_focuser_abort(fi: *mut OlsFocuserInterface) -> c_int;

    pub fn ols_focuser_is_moving(
        fi: *mut OlsFocuserInterface,
        out_moving: *mut i32,
    ) -> c_int;

    pub fn ols_focuser_is_absolute(
        fi: *mut OlsFocuserInterface,
        out_absolute: *mut i32,
    ) -> c_int;

    pub fn ols_focuser_max_range(
        fi: *mut OlsFocuserInterface,
        out_range: *mut i32,
    ) -> c_int;

    pub fn ols_focuser_unload(fi: *mut OlsFocuserInterface) -> c_int;

    pub fn ols_focuser_interface_destroy(fi: *mut OlsFocuserInterface);
}

// ═══════════════════════════════════════════════════════════════════
//  Helper: Convert OLS error to Result
// ═══════════════════════════════════════════════════════════════════

/// Convert an OLS error code into a Rust Result.
/// On failure, reads the thread-local error string from the C side.
pub fn check_ols_error(code: c_int) -> Result<(), String> {
    if code == OLS_OK {
        Ok(())
    } else {
        let msg = unsafe {
            let ptr = ols_last_error();
            if ptr.is_null() {
                "Unknown OLS error".to_string()
            } else {
                std::ffi::CStr::from_ptr(ptr)
                    .to_string_lossy()
                    .into_owned()
            }
        };
        Err(msg)
    }
}

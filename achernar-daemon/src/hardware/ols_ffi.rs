//! Raw FFI bindings to the OLS bridge C API.
//!
//! These are unsafe extern "C" declarations that match `ols_bridge.h`.
//! Use the safe wrappers in `camera.rs` / `mount.rs` instead.

#![allow(non_camel_case_types, dead_code)]

use std::os::raw::{c_char, c_double, c_int, c_void};

// ─── Error codes ───
pub const OLS_OK: c_int = 0;
pub const OLS_ERR_GENERIC: c_int = 1;
pub const OLS_ERR_NOT_FOUND: c_int = 2;
pub const OLS_ERR_INVALID: c_int = 3;
pub const OLS_ERR_IO: c_int = 4;

// ─── Stream types ───
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OlsStreamType {
    Yuv2 = 0,
    Mjpeg = 1,
    Rgb24 = 2,
    Rgb48 = 3,
    Raw8 = 4,
    Raw16 = 5,
    Mono8 = 6,
    Mono16 = 7,
    Error = 8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OlsBayerType {
    Na = 0,
    Rg = 1,
    Bg = 2,
    Gr = 3,
    Gb = 4,
}

// ─── Camera option IDs ───
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OlsCamOptionId {
    AutoExp = 0,
    AutoWb = 1,
    AutoFocus = 2,
    Exp = 3,
    Wb = 4,
    WbR = 5,
    WbB = 6,
    Focus = 7,
    Gain = 8,
    Gamma = 9,
    Brightness = 10,
    Contrast = 11,
    Temperature = 12,
    CoolerTarget = 13,
    CoolerOn = 14,
    FanOn = 15,
    CoolerPowerPerc = 16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OlsCamOptionType {
    Bool = 0,
    Number = 1,
    Msec = 2,
    Percent = 3,
    Kelvin = 4,
    Celsius = 5,
    Selection = 6,
}

// ─── Structs ───
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
#[derive(Debug)]
pub struct OlsFrame {
    pub format: OlsStreamType,
    pub bayer: OlsBayerType,
    pub frame_counter: i32,
    pub unix_timestamp: c_double,
    pub width: i32,
    pub height: i32,
    pub data: *const u8,
    pub data_size: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OlsCamParam {
    pub option: OlsCamOptionId,
    pub option_type: OlsCamOptionType,
    pub read_only: i32,
    pub step_size: c_double,
    pub min_val: c_double,
    pub max_val: c_double,
    pub def_val: c_double,
    pub cur_val: c_double,
}

// ─── Opaque handles ───
pub enum OlsCameraDriver_t {}
pub enum OlsCamera_t {}

// ─── Frame callback type ───
pub type OlsFrameCallback =
    Option<unsafe extern "C" fn(frame: *const OlsFrame, user_data: *mut c_void)>;

// ─── External functions ───
extern "C" {
    pub fn ols_last_error() -> *const c_char;
    pub fn ols_bridge_version() -> *const c_char;

    // Driver API
    pub fn ols_sim_driver_create(
        sim_data_dir: *const c_char,
        out_driver: *mut *mut OlsCameraDriver_t,
    ) -> c_int;

    pub fn ols_driver_list_cameras(
        driver: *mut OlsCameraDriver_t,
        out_count: *mut i32,
    ) -> c_int;

    pub fn ols_driver_camera_name(
        driver: *mut OlsCameraDriver_t,
        index: i32,
        name_buf: *mut c_char,
        name_buf_len: i32,
    ) -> c_int;

    pub fn ols_driver_open_camera(
        driver: *mut OlsCameraDriver_t,
        index: i32,
        out_camera: *mut *mut OlsCamera_t,
    ) -> c_int;

    pub fn ols_driver_destroy(driver: *mut OlsCameraDriver_t);

    // Camera API
    pub fn ols_camera_name(
        cam: *mut OlsCamera_t,
        name_buf: *mut c_char,
        name_buf_len: i32,
    ) -> c_int;

    pub fn ols_camera_format_count(cam: *mut OlsCamera_t, out_count: *mut i32) -> c_int;

    pub fn ols_camera_format_get(
        cam: *mut OlsCamera_t,
        index: i32,
        out_fmt: *mut OlsStreamFormat,
    ) -> c_int;

    pub fn ols_camera_start_stream(
        cam: *mut OlsCamera_t,
        fmt: *const OlsStreamFormat,
        callback: OlsFrameCallback,
        user_data: *mut c_void,
    ) -> c_int;

    pub fn ols_camera_stop_stream(cam: *mut OlsCamera_t) -> c_int;

    pub fn ols_camera_option_count(cam: *mut OlsCamera_t, out_count: *mut i32) -> c_int;

    pub fn ols_camera_option_id(
        cam: *mut OlsCamera_t,
        index: i32,
        out_id: *mut OlsCamOptionId,
    ) -> c_int;

    pub fn ols_camera_get_param(
        cam: *mut OlsCamera_t,
        id: OlsCamOptionId,
        out_param: *mut OlsCamParam,
    ) -> c_int;

    pub fn ols_camera_set_param(
        cam: *mut OlsCamera_t,
        id: OlsCamOptionId,
        value: c_double,
    ) -> c_int;

    pub fn ols_camera_destroy(cam: *mut OlsCamera_t);
}

//! Raw FFI bindings to the OLS bridge C API.
//!
//! These are unsafe C function declarations that map 1:1 to the
//! functions defined in `ols-bridge/include/ols_bridge.h`.
//! Safe wrappers live in sibling modules (camera.rs, mount.rs, focuser.rs).

#![allow(non_camel_case_types)]
#![allow(dead_code)]

use std::os::raw::{c_char, c_double, c_float, c_int, c_uint, c_void};

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
//  Build Capability Flags
// ═══════════════════════════════════════════════════════════════════

pub const OLS_CAP_CAMERA_ASI: u32 = 1 << 0;
pub const OLS_CAP_CAMERA_SVB: u32 = 1 << 1;
pub const OLS_CAP_CAMERA_TOUP: u32 = 1 << 2;
pub const OLS_CAP_CAMERA_GPHOTO2: u32 = 1 << 3;
pub const OLS_CAP_CAMERA_UVC: u32 = 1 << 4;
pub const OLS_CAP_CAMERA_INDI: u32 = 1 << 5;
pub const OLS_CAP_CAMERA_INDIGO: u32 = 1 << 6;
pub const OLS_CAP_CAMERA_WDIR: u32 = 1 << 7;

/// Bridge was built with LibRaw, so RAW containers can be decoded. Without it
/// a gphoto2 body still connects, streams and captures — it just archives a
/// container nothing downstream can read.
pub const OLS_CAP_DECODE_LIBRAW: u32 = 1 << 16;

// ═══════════════════════════════════════════════════════════════════
//  Stream / Pixel Format Enums
// ═══════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// Region-of-interest fraction (`roi_num`/`roi_den`) the driver applies.
    /// DSLR drivers enumerate several formats that share width/height/bin and
    /// differ only here, so these are needed to identify a format uniquely.
    pub roi_num: i32,
    pub roi_den: i32,
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
    /// Unmodified bytes of the file the camera produced (CR2/CR3/RW2/JPEG),
    /// or null. `data` above has already been black-level-corrected, scaled
    /// and cropped by the RAW decoder; these bytes have not.
    pub source_data: *const u8,
    pub source_data_size: usize,
    /// Lowercase extension without the dot ("cr3", "rw2", "jpg"), or null.
    pub source_ext: *const c_char,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    AverageBin = 17,
    BlackLevel = 18,
    ConvGainHcg = 19,
    ConvGainHdr = 20,
    LowNoise = 21,
    HighFullwell = 22,
    Iso = 23,
    Shutter = 24,
    Viewfinder = 25,
    CaptureTarget = 26,
    KeepImages = 27,
    CaptureDelay = 28,
    Zoom = 29,
    LiveStretch = 30,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OlsCamOptionType {
    Bool = 0,
    Number = 1,
    Msec = 2,
    Percent = 3,
    Kelvin = 4,
    Celsius = 5,
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

pub type OlsFrameCallback = unsafe extern "C" fn(frame: *const OlsFrame, user_data: *mut c_void);

// ═══════════════════════════════════════════════════════════════════
//  ABI Version
// ═══════════════════════════════════════════════════════════════════

/// Must match `OLS_BRIDGE_ABI_VERSION` in `ols-bridge/include/ols_bridge.h`.
///
/// These declarations are a hand-maintained copy of that header. A bridge and
/// a daemon built from different revisions link fine and then read each
/// other's structs at the wrong offsets, which shows up as corrupted frames
/// rather than an error — so the versions are checked at startup instead.
pub const OLS_BRIDGE_ABI_VERSION: u32 = 3;

// ═══════════════════════════════════════════════════════════════════
//  Single-Shot Still Capture
// ═══════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OlsStillRequest {
    pub exposure_seconds: c_double,
    /// ISO / gain. `<= 0` leaves the body's current setting alone.
    pub sensitivity: c_double,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OlsStillCaptureInfo {
    /// Unix timestamps taken by the driver around the trigger. `started_at` is
    /// the value for FITS `DATE-OBS`.
    pub started_at: c_double,
    pub completed_at: c_double,
    /// Read back from the camera after being applied — not the request.
    pub applied_exposure: c_double,
    pub applied_sensitivity: c_double,
    pub shutter_label: *const c_char,
    pub iso_label: *const c_char,
    pub asset_count: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OlsCapturedAsset {
    pub camera_path: *const c_char,
    pub extension: *const c_char,
    pub bytes: *const u8,
    pub byte_length: usize,
}

#[repr(C)]
pub struct OlsStillCapture {
    _private: [u8; 0],
}

// ═══════════════════════════════════════════════════════════════════
//  RAW Decode (LibRaw)
// ═══════════════════════════════════════════════════════════════════

#[repr(C)]
pub struct OlsRawImage {
    _private: [u8; 0],
}

/// Everything the decode deliberately did *not* apply, so it can be applied
/// downstream. See the header comment on `ols_decode_raw` for why the pixels
/// are untouched sensor ADU.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OlsRawImageInfo {
    pub width: i32,
    pub height: i32,
    pub bayer: OlsBayerType,
    /// Storage depth per sample: always 16. The sensor's real depth is implied
    /// by `white_level` (16383 → 14-bit), which is the value saturation tests
    /// must use — the samples do not reach the top of the container.
    pub bits_per_sample: i32,
    /// Global black pedestal in ADU. NOT subtracted from the pixels.
    pub black_level: i32,
    /// Per-CFA-channel black offsets, added to `black_level`.
    pub cblack: [i32; 4],
    /// Dimensions of LibRaw's 2-D black table, `[0, 0]` when there is none.
    /// Anything larger than 2x2 is not representable in `cblack`, so a
    /// non-trivial value means the recorded black level is incomplete.
    pub cblack_pattern: [i32; 2],
    /// Saturation point in ADU. NOT normalised to.
    pub white_level: i32,
    /// As-shot white balance multipliers. Recorded, never applied.
    pub cam_mul: [f32; 4],
    pub make: *const c_char,
    pub model: *const c_char,
}

// ═══════════════════════════════════════════════════════════════════
//  Mount Types
// ═══════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OlsMountProto {
    None = 0,
    Serial = 1,
    Inet = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OlsMountTracking {
    Sidereal = 0,
    Solar = 1,
    Lunar = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OlsMountSlew {
    Stop = 0,
    N = 2,  // 1<<1
    S = 6,  // 3<<1
    W = 8,  // 1<<3
    E = 24, // 3<<3
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OlsMeridianFlip {
    Unsupported = 0,
    Flip = 1,
    Stop = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OlsEqCoord {
    pub ra: f64,  // hours
    pub dec: f64, // degrees
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OlsAltAzCoord {
    pub alt: f64, // degrees
    pub az: f64,  // degrees
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
    pub fn ols_bridge_abi_version() -> c_uint;

    // ─── Build Capabilities ───
    pub fn ols_driver_capability_flags() -> c_uint;

    // ═══ Camera Driver API ═══
    pub fn ols_sim_driver_create(
        sim_data_dir: *const c_char,
        out_driver: *mut *mut OlsCameraDriver,
    ) -> c_int;

    pub fn ols_alpaca_camera_driver_create(
        url: *const c_char,
        out_driver: *mut *mut OlsCameraDriver,
    ) -> c_int;

    pub fn ols_asi_camera_driver_create(out_driver: *mut *mut OlsCameraDriver) -> c_int;

    pub fn ols_svb_camera_driver_create(out_driver: *mut *mut OlsCameraDriver) -> c_int;

    pub fn ols_toup_camera_driver_create(out_driver: *mut *mut OlsCameraDriver) -> c_int;

    pub fn ols_wdir_camera_driver_create(
        watch_dir: *const c_char,
        out_driver: *mut *mut OlsCameraDriver,
    ) -> c_int;

    pub fn ols_gphoto2_camera_driver_create(out_driver: *mut *mut OlsCameraDriver) -> c_int;

    pub fn ols_uvc_camera_driver_create(out_driver: *mut *mut OlsCameraDriver) -> c_int;

    pub fn ols_indi_camera_driver_create(out_driver: *mut *mut OlsCameraDriver) -> c_int;

    pub fn ols_indigo_camera_driver_create(out_driver: *mut *mut OlsCameraDriver) -> c_int;

    pub fn ols_driver_list_cameras(driver: *mut OlsCameraDriver, out_count: *mut i32) -> c_int;

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
    pub fn ols_camera_name(cam: *mut OlsCamera, name_buf: *mut c_char, name_buf_len: i32) -> c_int;

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

    pub fn ols_camera_set_param(cam: *mut OlsCamera, id: OlsCamOptionId, value: c_double) -> c_int;

    /// Fetch the choice labels of a selection-type option (`iso`,
    /// `shutterspeed`, `capturetarget`, …) as consecutive NUL-terminated
    /// strings in choice-index order.
    ///
    /// Pass a null `buf` to learn `out_count` / `out_required` without writing.
    pub fn ols_camera_param_choices(
        cam: *mut OlsCamera,
        id: OlsCamOptionId,
        buf: *mut c_char,
        buf_len: usize,
        out_count: *mut i32,
        out_required: *mut usize,
    ) -> c_int;

    // ─── Single-shot still capture ───
    pub fn ols_camera_supports_still(cam: *mut OlsCamera, out_supported: *mut i32) -> c_int;

    pub fn ols_camera_capture_still(
        cam: *mut OlsCamera,
        request: *const OlsStillRequest,
        out_capture: *mut *mut OlsStillCapture,
    ) -> c_int;

    pub fn ols_capture_info(
        capture: *const OlsStillCapture,
        out_info: *mut OlsStillCaptureInfo,
    ) -> c_int;

    pub fn ols_capture_asset_count(capture: *const OlsStillCapture) -> i32;

    pub fn ols_capture_asset_get(
        capture: *const OlsStillCapture,
        index: i32,
        out_asset: *mut OlsCapturedAsset,
    ) -> c_int;

    pub fn ols_camera_delete_captured(
        cam: *mut OlsCamera,
        capture: *const OlsStillCapture,
    ) -> c_int;

    pub fn ols_capture_destroy(capture: *mut OlsStillCapture);

    pub fn ols_camera_destroy(cam: *mut OlsCamera);

    // ─── RAW decode ───
    //
    // `data` is borrowed for the duration of the call only. Pass width/height
    // of -1 for the full active area; `bin` must be 1.
    pub fn ols_decode_raw(
        data: *const u8,
        len: usize,
        width: i32,
        height: i32,
        bin: i32,
        out_image: *mut *mut OlsRawImage,
    ) -> c_int;

    pub fn ols_rawimage_info(image: *const OlsRawImage, out_info: *mut OlsRawImageInfo) -> c_int;

    pub fn ols_rawimage_pixels(image: *const OlsRawImage, out_len: *mut usize) -> *const u8;

    pub fn ols_rawimage_destroy(image: *mut OlsRawImage);

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

    pub fn ols_mount_get_current(mi: *mut OlsMountInterface, out_coord: *mut OlsEqCoord) -> c_int;

    pub fn ols_mount_goto(mi: *mut OlsMountInterface, ra: c_double, dec: c_double) -> c_int;

    pub fn ols_mount_sync(mi: *mut OlsMountInterface, ra: c_double, dec: c_double) -> c_int;

    pub fn ols_mount_slew(mi: *mut OlsMountInterface, direction: OlsMountSlew, speed: i32)
        -> c_int;

    pub fn ols_mount_get_tracking(
        mi: *mut OlsMountInterface,
        out_mode: *mut OlsMountTracking,
    ) -> c_int;

    pub fn ols_mount_get_tracking_state(mi: *mut OlsMountInterface, out_state: *mut i32) -> c_int;

    pub fn ols_mount_set_tracking_state(mi: *mut OlsMountInterface, track: i32) -> c_int;

    pub fn ols_mount_set_tracking_mode(mi: *mut OlsMountInterface, mode: OlsMountTracking)
        -> c_int;

    pub fn ols_mount_set_lat_lon(mi: *mut OlsMountInterface, lat: c_double, lon: c_double)
        -> c_int;

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

    pub fn ols_mount_set_meridian(mi: *mut OlsMountInterface, flip: OlsMeridianFlip) -> c_int;

    pub fn ols_mount_get_alt_limits(
        mi: *mut OlsMountInterface,
        out_low: *mut i32,
        out_high: *mut i32,
    ) -> c_int;

    pub fn ols_mount_set_alt_limits(mi: *mut OlsMountInterface, low: i32, high: i32) -> c_int;

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

    pub fn ols_focuser_position(fi: *mut OlsFocuserInterface, out_position: *mut i32) -> c_int;

    pub fn ols_focuser_move_to(fi: *mut OlsFocuserInterface, position: i32) -> c_int;

    pub fn ols_focuser_move_by(fi: *mut OlsFocuserInterface, delta: i32) -> c_int;

    pub fn ols_focuser_abort(fi: *mut OlsFocuserInterface) -> c_int;

    pub fn ols_focuser_is_moving(fi: *mut OlsFocuserInterface, out_moving: *mut i32) -> c_int;

    pub fn ols_focuser_is_absolute(fi: *mut OlsFocuserInterface, out_absolute: *mut i32) -> c_int;

    pub fn ols_focuser_max_range(fi: *mut OlsFocuserInterface, out_range: *mut i32) -> c_int;

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
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Err(msg)
    }
}

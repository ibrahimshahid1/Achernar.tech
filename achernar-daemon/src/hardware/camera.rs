//! Safe Rust wrapper around OLS camera FFI.
//!
//! Provides `OlsCameraDriver` and `OlsCamera` types that manage
//! the lifecycle of OLS C++ objects and convert between C and Rust types.

use super::ols_ffi;
use crate::processing::debayer::{
    ByteOrder, CfaPattern, NativeThreshold, PixelGeometry, SampleDomain, SampleType, SensorColor,
    SensorDescriptor,
};
use anyhow::{anyhow, Result};
use std::ffi::{CStr, CString};
use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

/// Get the last error message from the OLS bridge.
pub(crate) fn last_error() -> String {
    unsafe {
        let ptr = ols_ffi::ols_last_error();
        if ptr.is_null() {
            "Unknown error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

/// Borrow a C string, or an empty String when the pointer is null.
/// The bridge promises non-null for its label fields; this is belt-and-braces
/// so a driver bug degrades to "no label" rather than a segfault.
fn cstr_or_default(ptr: *const std::os::raw::c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }
}

/// Check an OLS error code and convert to Result.
pub(crate) fn check(code: i32) -> Result<()> {
    if code == ols_ffi::OLS_OK {
        Ok(())
    } else {
        Err(anyhow!("OLS error ({}): {}", code, last_error()))
    }
}

/// Stream format info (Rust-safe version of OlsStreamFormat)
#[derive(Debug, Clone, Copy)]
pub struct StreamFormat {
    pub format: ols_ffi::OlsStreamType,
    pub width: i32,
    pub height: i32,
    pub bin: i32,
    pub roi_num: i32,
    pub roi_den: i32,
    pub framerate: f32,
}

impl StreamFormat {
    /// True when this looks like a driver-provided continuous live-view feed
    /// rather than a full-resolution still: any format advertising a frame
    /// rate or the OLS `-1` sentinel for an unknown frame rate. ASI, SVBony,
    /// simulator and watch-folder streams all use `-1`; labelling those as one
    /// mechanical shutter actuation per frame is actively misleading.
    /// UVC Mono8/YUV2 feeds are just as continuous as MJPEG; pixel encoding
    /// does not decide whether a mechanical shutter is actuated.
    ///
    /// The gphoto2 driver enumerates live view first precisely so that callers
    /// which do not choose a format never actuate the shutter; this predicate
    /// lets the daemon label the mode in API responses.
    pub fn is_liveview(&self) -> bool {
        self.framerate != 0.0
    }
}

impl From<ols_ffi::OlsStreamFormat> for StreamFormat {
    fn from(f: ols_ffi::OlsStreamFormat) -> Self {
        Self {
            format: f.format,
            width: f.width,
            height: f.height,
            bin: f.bin,
            roi_num: f.roi_num,
            roi_den: f.roi_den,
            framerate: f.framerate,
        }
    }
}

impl From<&StreamFormat> for ols_ffi::OlsStreamFormat {
    fn from(f: &StreamFormat) -> Self {
        Self {
            format: f.format,
            width: f.width,
            height: f.height,
            bin: f.bin,
            roi_num: f.roi_num,
            roi_den: f.roi_den,
            framerate: f.framerate,
        }
    }
}

/// Camera parameter info (Rust-safe version of OlsCamParam)
#[derive(Debug, Clone, Copy)]
pub struct CamParam {
    pub option: ols_ffi::OlsCamOptionId,
    pub option_type: ols_ffi::OlsCamOptionType,
    pub read_only: bool,
    pub step_size: f64,
    pub min_val: f64,
    pub max_val: f64,
    pub def_val: f64,
    pub cur_val: f64,
}

/// The camera-authored original a frame was decoded from.
///
/// For a DSLR or mirrorless body this is the CR2/CR3/RW2/NEF exactly as the
/// camera wrote it. It matters because `FrameData::data` is already a lossy
/// interpretation: LibRaw has subtracted the black level, applied a scale
/// factor, cropped to the active area and dropped the maker notes. A training
/// set built only from decoded pixels is permanently welded to whatever decode
/// path was current the night it was captured — you can never re-derive it
/// with a better demosaic, a different calibration, or corrected black points.
#[derive(Debug, Clone)]
pub struct SourceImage {
    /// Lowercase extension without the dot: "cr3", "cr2", "rw2", "jpg".
    pub ext: String,
    /// The file's bytes, verbatim.
    pub bytes: Vec<u8>,
}

/// A captured frame with owned data
#[derive(Debug, Clone)]
pub struct FrameData {
    /// Daemon-assigned stream generation. Unlike a driver counter this never
    /// resets on reconnect, so old-camera pixels cannot cross a lifecycle.
    pub stream_generation: u64,
    pub format: ols_ffi::OlsStreamType,
    pub bayer: ols_ffi::OlsBayerType,
    pub frame_counter: i32,
    pub unix_timestamp: f64,
    pub width: i32,
    pub height: i32,
    pub data: Vec<u8>,
    /// The original file the pixels came from, when the driver supplied one.
    /// `None` for sensor drivers that DMA pixels straight out of hardware, and
    /// for live-view preview frames (which are not science data).
    pub source: Option<SourceImage>,
    /// Linear saturation point of the sensor, in this frame's own ADU.
    ///
    /// Per-frame and not global, because a rig with a guider and an imaging
    /// camera has two different answers at the same time. `None` means unknown,
    /// and unknown must stay distinguishable from "does not saturate": a
    /// saturation gate computed from the container depth sits at 64224 on a
    /// 14-bit body whose data tops out at 16383, so it never fires and every
    /// clipped star passes.
    ///
    /// Sourced from `[camera] saturation_adu` in `achernar.toml` for streams,
    /// and from the RAW decoder's `white_level` on the capture path. No driver
    /// reports it today; when one can, this is the field it fills.
    pub saturation_adu: Option<f32>,
    /// Explicit pixel-domain contract. Absence means scientific analysis is
    /// unavailable; previews can still consume the original buffer.
    pub sensor_descriptor: Option<SensorDescriptor>,
}

impl FrameData {
    pub fn analysis_sensor_descriptor(&self) -> crate::error::Result<&SensorDescriptor> {
        let unavailable = |reason: &str| {
            crate::error::AchernarError::Hardware(crate::error::HardwareError::Communication(
                reason.to_string(),
            ))
        };
        if self.source.is_some() {
            return Err(unavailable(
                "Legacy SourceImage pixels have an unqualified decode domain (A04)",
            ));
        }
        let descriptor = self.sensor_descriptor.as_ref().ok_or_else(|| {
            unavailable(
            "Scientific analysis unavailable: no qualified sensor/CFA descriptor for this frame"
        )
        })?;
        if descriptor.width != self.width.max(0) as usize
            || descriptor.height != self.height.max(0) as usize
        {
            return Err(unavailable(
                "Sensor descriptor dimensions differ from frame identity",
            ));
        }
        let compatible = match (self.format, &descriptor.sample_type, &descriptor.color) {
            (ols_ffi::OlsStreamType::Mono8, SampleType::U8, SensorColor::Mono)
            | (ols_ffi::OlsStreamType::Mono16, SampleType::U16, SensorColor::Mono) => {
                self.bayer == ols_ffi::OlsBayerType::Na
            }
            (
                ols_ffi::OlsStreamType::Raw8,
                SampleType::U8,
                SensorColor::Bayer {
                    pattern,
                    origin_x,
                    origin_y,
                },
            )
            | (
                ols_ffi::OlsStreamType::Raw16,
                SampleType::U16,
                SensorColor::Bayer {
                    pattern,
                    origin_x,
                    origin_y,
                },
            ) => bayer_pattern(self.bayer)
                .map(|actual| {
                    (0..2).all(|y| {
                        (0..2).all(|x| {
                            pattern.channel(x + origin_x % 2, y + origin_y % 2)
                                == actual.channel(x, y)
                        })
                    })
                })
                .unwrap_or(false),
            _ => false,
        };
        if !compatible {
            return Err(unavailable("Preview, unknown CFA, or incompatible sensor descriptor cannot enter scientific analysis"));
        }
        descriptor
            .validate(self.data.len())
            .map_err(|error| unavailable(&error.to_string()))?;
        Ok(descriptor)
    }
}

fn bayer_pattern(bayer: ols_ffi::OlsBayerType) -> Option<CfaPattern> {
    match bayer {
        ols_ffi::OlsBayerType::Rg => Some(CfaPattern::Rggb),
        ols_ffi::OlsBayerType::Bg => Some(CfaPattern::Bggr),
        ols_ffi::OlsBayerType::Gr => Some(CfaPattern::Grbg),
        ols_ffi::OlsBayerType::Gb => Some(CfaPattern::Gbrg),
        ols_ffi::OlsBayerType::Na => None,
    }
}

/// The native stream ABI supplies packed host-endian rows and Bayer phase at
/// buffer origin. Restrict color to full-ROI/bin1 because it does not convey
/// crop offsets or Bayer-aware binning. Coordinates refer only to this buffer.
fn stream_sensor_descriptor(
    format: &StreamFormat,
    frame: &ols_ffi::OlsFrame,
    has_source: bool,
    saturation_adu: Option<f32>,
) -> Option<SensorDescriptor> {
    if has_source
        || frame.format != format.format
        || frame.width <= 0
        || frame.height <= 0
        || frame.width != format.width
        || frame.height != format.height
    {
        return None;
    }
    let sample_type = match frame.format {
        ols_ffi::OlsStreamType::Mono8 | ols_ffi::OlsStreamType::Raw8 => SampleType::U8,
        ols_ffi::OlsStreamType::Mono16 | ols_ffi::OlsStreamType::Raw16 => SampleType::U16,
        _ => return None,
    };
    let color = match frame.format {
        ols_ffi::OlsStreamType::Mono8 | ols_ffi::OlsStreamType::Mono16
            if frame.bayer == ols_ffi::OlsBayerType::Na =>
        {
            SensorColor::Mono
        }
        ols_ffi::OlsStreamType::Raw8 | ols_ffi::OlsStreamType::Raw16
            if format.bin == 1 && format.roi_num == format.roi_den && format.roi_den > 0 =>
        {
            SensorColor::Bayer {
                pattern: bayer_pattern(frame.bayer)?,
                origin_x: 0,
                origin_y: 0,
            }
        }
        _ => return None,
    };
    let width = frame.width as usize;
    let height = frame.height as usize;
    let bytes_per_sample = if matches!(sample_type, SampleType::U8) {
        1
    } else {
        2
    };
    let stride = width.checked_mul(bytes_per_sample)?;
    if frame.data_size != stride.checked_mul(height)? {
        return None;
    }
    Some(SensorDescriptor {
        schema_version: 1,
        width,
        height,
        row_stride_bytes: stride,
        sample_type,
        byte_order: if cfg!(target_endian = "little") {
            ByteOrder::Little
        } else {
            ByteOrder::Big
        },
        color,
        domain: SampleDomain::NativeAdu,
        geometry: PixelGeometry::identity("stream_buffer_pixels"),
        saturation: saturation_adu
            .filter(|value| value.is_finite() && *value > 0.0)
            .map(|adu| NativeThreshold {
                adu,
                source: "camera_configuration_declaration".to_string(),
            }),
        calibration_ids: Vec::new(),
    })
}

/// One file a single exposure produced, copied out of the bridge.
#[derive(Debug, Clone)]
pub struct CapturedAsset {
    /// Original camera path for provenance and later reconciliation. A local
    /// commit alone does not authorize deleting the camera copy.
    pub camera_path: String,
    /// Lowercase extension without the dot: "cr3", "cr2", "rw2", "jpg".
    pub extension: String,
    pub bytes: Vec<u8>,
}

impl CapturedAsset {
    /// Asset role, from the container the camera wrote.
    pub fn role(&self) -> &'static str {
        match self.extension.as_str() {
            "jpg" | "jpeg" => "camera_jpeg",
            _ => "native_raw",
        }
    }

    /// IANA media type, as specific as we can be. Recorded per asset so a
    /// consumer never has to re-sniff the bytes to know what it has.
    pub fn media_type(&self) -> &'static str {
        match self.extension.as_str() {
            "jpg" | "jpeg" => "image/jpeg",
            "cr2" => "image/x-canon-cr2",
            "cr3" => "image/x-canon-cr3",
            "rw2" => "image/x-panasonic-rw2",
            "nef" => "image/x-nikon-nef",
            "arw" => "image/x-sony-arw",
            "dng" => "image/x-adobe-dng",
            _ => "application/octet-stream",
        }
    }
}

/// Recovered outcome of one shutter actuation; inspect bundle completeness.
///
/// Everything here describes that single exposure and nothing else. It never
/// passes through the streaming frame callback, so there is no window in which
/// the pixels and the metadata can drift apart.
#[derive(Debug, Clone)]
pub struct StillCapture {
    /// Unix timestamp taken immediately before the trigger — the value for
    /// FITS `DATE-OBS`. A clock read after the call returns is late by the
    /// exposure plus however long the download took.
    pub started_at: f64,
    pub completed_at: f64,
    /// Settings read back from the camera after being applied.
    pub applied_exposure_s: f64,
    /// ISO / gain as read back; `<= 0` when the body reports nothing numeric.
    pub applied_sensitivity: f64,
    /// The camera's own labels ("1/125", "800"). Empty when not applicable.
    pub shutter_label: String,
    pub iso_label: String,
    /// Every file the exposure produced — two on a RAW+JPEG body.
    pub assets: Vec<CapturedAsset>,
    /// Event/download evidence. Unknown expected companion count is preserved.
    pub bundle: serde_json::Value,
    pub bundle_complete: bool,
}

impl StillCapture {
    /// The asset a science pipeline should treat as canonical: the native RAW
    /// when there is one, otherwise the camera JPEG.
    pub fn primary(&self) -> Option<&CapturedAsset> {
        self.assets
            .iter()
            .find(|a| a.role() == "native_raw")
            .or_else(|| self.assets.first())
    }

    pub fn exposure_duration_s(&self) -> f64 {
        (self.completed_at - self.started_at).max(0.0)
    }
}

/// Live handle to a completed capture inside the bridge.
///
/// Held long enough to copy the bytes out. `Drop` releases bridge buffers and
/// retains camera files.
pub struct StillCaptureHandle {
    handle: *mut ols_ffi::OlsStillCapture,
}

// Safety: the handle is an owned, immutable snapshot; the bridge does not
// touch it again until destroy.
unsafe impl Send for StillCaptureHandle {}

impl Drop for StillCaptureHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ols_ffi::ols_capture_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

/// Verify the compiled bridge matches the FFI declarations this binary was
/// built against.
///
/// Layout drift between the two does not fail to link — it silently reads
/// structs at the wrong offsets, which surfaces as corrupt frames or a crash
/// deep in a driver. Call once at startup.
pub fn check_abi() -> Result<u32> {
    let actual = unsafe { ols_ffi::ols_bridge_abi_version() };
    if actual != ols_ffi::OLS_BRIDGE_ABI_VERSION {
        return Err(anyhow!(
            "OLS bridge ABI mismatch: daemon expects v{}, linked bridge is v{}. \
             Rebuild the bridge (`cargo clean -p achernar-daemon && cargo build`).",
            ols_ffi::OLS_BRIDGE_ABI_VERSION,
            actual
        ));
    }
    Ok(actual)
}

/// OLS Camera Driver - manages driver lifecycle and camera enumeration
pub struct OlsCameraDriver {
    handle: *mut ols_ffi::OlsCameraDriver,
}

// Safety: The OLS driver uses internal locking
unsafe impl Send for OlsCameraDriver {}

impl OlsCameraDriver {
    /// Create a sim camera driver
    pub fn new_sim(data_dir: &str) -> Result<Self> {
        let c_dir = CString::new(data_dir)?;
        let mut handle = std::ptr::null_mut();
        unsafe {
            check(ols_ffi::ols_sim_driver_create(c_dir.as_ptr(), &mut handle))?;
        }
        Ok(Self { handle })
    }

    /// Create an Alpaca camera driver (connects to Alpaca server over network).
    pub fn new_alpaca(url: &str) -> Result<Self> {
        let c_url = CString::new(url)?;
        let mut handle = std::ptr::null_mut();
        unsafe {
            check(ols_ffi::ols_alpaca_camera_driver_create(
                c_url.as_ptr(),
                &mut handle,
            ))?;
        }
        Ok(Self { handle })
    }

    /// Create an ASI (ZWO) camera driver.
    pub fn new_asi() -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        unsafe {
            check(ols_ffi::ols_asi_camera_driver_create(&mut handle))?;
        }
        Ok(Self { handle })
    }

    /// Create an SVBony camera driver.
    pub fn new_svb() -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        unsafe {
            check(ols_ffi::ols_svb_camera_driver_create(&mut handle))?;
        }
        Ok(Self { handle })
    }

    /// Create a ToupTek camera driver.
    pub fn new_toup() -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        unsafe {
            check(ols_ffi::ols_toup_camera_driver_create(&mut handle))?;
        }
        Ok(Self { handle })
    }

    /// Create a watch-directory pseudo-camera that ingests frames from a folder.
    /// Requires no vendor SDK — useful for development and bench testing.
    pub fn new_wdir(watch_dir: &str) -> Result<Self> {
        let c_dir = CString::new(watch_dir)?;
        let mut handle = std::ptr::null_mut();
        unsafe {
            check(ols_ffi::ols_wdir_camera_driver_create(
                c_dir.as_ptr(),
                &mut handle,
            ))?;
        }
        Ok(Self { handle })
    }

    /// Create a gphoto2 (DSLR) camera driver. Requires libgphoto2 at build time.
    pub fn new_gphoto2() -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        unsafe {
            check(ols_ffi::ols_gphoto2_camera_driver_create(&mut handle))?;
        }
        Ok(Self { handle })
    }

    /// Create a UVC (USB video) camera driver. Requires libuvc at build time.
    pub fn new_uvc() -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        unsafe {
            check(ols_ffi::ols_uvc_camera_driver_create(&mut handle))?;
        }
        Ok(Self { handle })
    }

    /// Create an INDI camera driver. Requires libindi at build time.
    pub fn new_indi() -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        unsafe {
            check(ols_ffi::ols_indi_camera_driver_create(&mut handle))?;
        }
        Ok(Self { handle })
    }

    /// Create an INDIGO camera driver. Requires the INDIGO libraries at build time.
    pub fn new_indigo() -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        unsafe {
            check(ols_ffi::ols_indigo_camera_driver_create(&mut handle))?;
        }
        Ok(Self { handle })
    }

    /// List available cameras
    pub fn list_cameras(&mut self) -> Result<Vec<String>> {
        let mut count: i32 = 0;
        unsafe {
            check(ols_ffi::ols_driver_list_cameras(self.handle, &mut count))?;
        }

        let mut cameras = Vec::with_capacity(count as usize);
        for i in 0..count {
            let mut name_buf = vec![0u8; 256];
            unsafe {
                check(ols_ffi::ols_driver_camera_name(
                    self.handle,
                    i,
                    name_buf.as_mut_ptr() as *mut i8,
                    256,
                ))?;
            }
            let name = CStr::from_bytes_until_nul(&name_buf)
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            cameras.push(name);
        }

        Ok(cameras)
    }

    /// Open a camera by index
    pub fn open_camera(&mut self, index: i32) -> Result<OlsCamera> {
        let mut cam_handle = std::ptr::null_mut();
        unsafe {
            check(ols_ffi::ols_driver_open_camera(
                self.handle,
                index,
                &mut cam_handle,
            ))?;
        }
        Ok(OlsCamera {
            handle: cam_handle,
            _callback_data: None,
            fault_handler: None,
            stream_progress: Arc::new(Mutex::new(StreamProgress::default())),
        })
    }
}

impl Drop for OlsCameraDriver {
    fn drop(&mut self) {
        unsafe {
            ols_ffi::ols_driver_destroy(self.handle);
        }
    }
}

/// OLS Camera - wraps a single camera instance
pub struct OlsCamera {
    handle: *mut ols_ffi::OlsCamera,
    _callback_data: Option<Box<CallbackData>>,
    fault_handler: Option<Arc<dyn Fn(String) + Send + Sync>>,
    stream_progress: Arc<Mutex<StreamProgress>>,
}

// Safety: OLS camera uses internal locking
unsafe impl Send for OlsCamera {}

struct CallbackData {
    latest_frame: Arc<Mutex<Option<FrameData>>>,
    stream_generation: u64,
    active_generation: Arc<std::sync::atomic::AtomicU64>,
    /// Stamped onto every frame this stream produces. Held per-stream rather
    /// than read from a global so two cameras on one rig can carry different
    /// values. See `FrameData::saturation_adu`.
    saturation_adu: Option<f32>,
    format: StreamFormat,
    fault_handler: Option<Arc<dyn Fn(String) + Send + Sync>>,
    stream_progress: Arc<Mutex<StreamProgress>>,
}

#[derive(Default)]
struct StreamProgress {
    last_frame: Option<std::time::Instant>,
    /// Only positive, declared preview rates have a bounded frame cadence.
    /// The -1 sentinel may be a minutes-long astronomical exposure.
    max_silence: Option<std::time::Duration>,
}

impl OlsCamera {
    pub fn set_fault_handler(&mut self, handler: Arc<dyn Fn(String) + Send + Sync>) {
        self.fault_handler = Some(handler);
    }

    /// True means a passive hardware transaction succeeded; false means this
    /// driver's idle presence check is explicitly unsupported.
    pub fn check_connection(&self) -> Result<bool> {
        let rc = unsafe { ols_ffi::ols_camera_check_health(self.handle) };
        if rc == ols_ffi::OLS_ERR_NOT_SUPPORTED {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    pub fn stream_health(&self) -> Result<()> {
        let progress = self.stream_progress.lock().unwrap();
        if let (Some(last), Some(limit)) = (progress.last_frame, progress.max_silence) {
            if last.elapsed() > limit {
                return Err(anyhow!("Camera preview stopped delivering frames"));
            }
        }
        Ok(())
    }

    /// Get camera name
    pub fn name(&self) -> Result<String> {
        let mut name_buf = vec![0u8; 256];
        unsafe {
            check(ols_ffi::ols_camera_name(
                self.handle as *mut _,
                name_buf.as_mut_ptr() as *mut i8,
                256,
            ))?;
        }
        Ok(CStr::from_bytes_until_nul(&name_buf)
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned())
    }

    /// Get supported stream formats
    pub fn formats(&self) -> Result<Vec<StreamFormat>> {
        let mut count: i32 = 0;
        unsafe {
            check(ols_ffi::ols_camera_format_count(
                self.handle as *mut _,
                &mut count,
            ))?;
        }

        let mut formats = Vec::with_capacity(count as usize);
        for i in 0..count {
            let mut fmt = ols_ffi::OlsStreamFormat {
                format: ols_ffi::OlsStreamType::Mjpeg,
                width: 0,
                height: 0,
                bin: 0,
                roi_num: 1,
                roi_den: 1,
                framerate: 0.0,
            };
            unsafe {
                check(ols_ffi::ols_camera_format_get(
                    self.handle as *mut _,
                    i,
                    &mut fmt,
                ))?;
            }
            formats.push(StreamFormat::from(fmt));
        }

        Ok(formats)
    }

    /// Start streaming, storing frames into the provided Arc<Mutex<Option<FrameData>>>
    ///
    /// `saturation_adu` is the sensor's linear saturation point in this
    /// stream's ADU, stamped onto every frame. Pass `None` when it is not
    /// known — the quality gate then reports itself as not performed instead of
    /// silently passing every star. See `FrameData::saturation_adu`.
    pub fn start_stream(
        &mut self,
        format: &StreamFormat,
        frame_store: Arc<Mutex<Option<FrameData>>>,
        saturation_adu: Option<f32>,
        stream_generation: u64,
        active_generation: Arc<std::sync::atomic::AtomicU64>,
    ) -> Result<()> {
        if self.supports_still_capture() && !format.is_liveview() {
            return Err(anyhow!("DSLR still streams cannot archive complete exposure bundles; use dedicated still capture (A02)."));
        }
        let cb_data = Box::new(CallbackData {
            latest_frame: frame_store,
            saturation_adu,
            format: format.clone(),
            stream_generation,
            active_generation,
            fault_handler: self.fault_handler.clone(),
            stream_progress: self.stream_progress.clone(),
        });
        *self.stream_progress.lock().unwrap() = StreamProgress {
            last_frame: Some(std::time::Instant::now()),
            max_silence: (format.framerate.is_finite() && format.framerate > 0.0).then(|| {
                std::time::Duration::from_secs_f64((10.0 / format.framerate as f64).max(5.0))
            }),
        };
        let user_data = &*cb_data as *const CallbackData as *mut c_void;

        let ffi_fmt: ols_ffi::OlsStreamFormat = format.into();

        unsafe {
            check(ols_ffi::ols_camera_start_stream(
                self.handle,
                &ffi_fmt,
                frame_callback_trampoline,
                user_data,
            ))?;
        }

        // Keep callback data alive as long as stream is active
        self._callback_data = Some(cb_data);
        Ok(())
    }

    /// Stop streaming
    pub fn stop_stream(&mut self) -> Result<()> {
        unsafe {
            check(ols_ffi::ols_camera_stop_stream(self.handle))?;
        }
        self._callback_data = None;
        *self.stream_progress.lock().unwrap() = StreamProgress::default();
        Ok(())
    }

    /// Get camera parameter
    pub fn get_parameter(&self, id: ols_ffi::OlsCamOptionId) -> Result<CamParam> {
        let mut param = ols_ffi::OlsCamParam {
            option: ols_ffi::OlsCamOptionId::Exp,
            option_type: ols_ffi::OlsCamOptionType::Number,
            read_only: 0,
            step_size: 0.0,
            min_val: 0.0,
            max_val: 0.0,
            def_val: 0.0,
            cur_val: 0.0,
        };
        unsafe {
            check(ols_ffi::ols_camera_get_param(
                self.handle as *mut _,
                id,
                &mut param,
            ))?;
        }
        Ok(CamParam {
            option: param.option,
            option_type: param.option_type,
            read_only: param.read_only != 0,
            step_size: param.step_size,
            min_val: param.min_val,
            max_val: param.max_val,
            def_val: param.def_val,
            cur_val: param.cur_val,
        })
    }

    /// Set camera parameter
    pub fn set_parameter(&self, id: ols_ffi::OlsCamOptionId, value: f64) -> Result<()> {
        if id == ols_ffi::OlsCamOptionId::KeepImages {
            crate::storage::capture_retention::validate_legacy_keep_images(value)
                .map_err(|reason| anyhow!(reason))?;
        }
        unsafe {
            check(ols_ffi::ols_camera_set_param(
                self.handle as *mut _,
                id,
                value,
            ))
        }
    }

    /// Human-readable labels for a selection-type option, in choice-index
    /// order — so `labels[n]` is what the camera does when you
    /// `set_parameter(id, n as f64)`.
    ///
    /// DSLR bodies expose exposure and gain only as enumerations
    /// (`shutterspeed` = ["30", "25", …, "1/4000", "bulb"], `iso` = ["Auto",
    /// "100", "200", …]). The numeric index carries no physical meaning; the
    /// label is the value. Without these the daemon can change the exposure
    /// but cannot report what exposure it made.
    ///
    /// Returns an empty vector for continuous options — not an error.
    pub fn parameter_choices(&self, id: ols_ffi::OlsCamOptionId) -> Result<Vec<String>> {
        let mut count: i32 = 0;
        let mut required: usize = 0;

        // First call sizes the buffer, second fills it. The choice list is
        // fixed per body, but it is cheap enough to re-read and this avoids
        // caching stale values after a mode-dial change.
        unsafe {
            check(ols_ffi::ols_camera_param_choices(
                self.handle as *mut _,
                id,
                std::ptr::null_mut(),
                0,
                &mut count,
                &mut required,
            ))?;
        }
        if count <= 0 || required == 0 {
            return Ok(Vec::new());
        }

        let mut buf = vec![0u8; required];
        unsafe {
            check(ols_ffi::ols_camera_param_choices(
                self.handle as *mut _,
                id,
                buf.as_mut_ptr() as *mut std::os::raw::c_char,
                buf.len(),
                &mut count,
                &mut required,
            ))?;
        }

        // The buffer holds `count` consecutive NUL-terminated strings. Split on
        // NUL and drop the trailing empty produced by the final terminator.
        let labels: Vec<String> = buf
            .split(|b| *b == 0)
            .filter(|s| !s.is_empty())
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .collect();

        if labels.len() != count as usize {
            tracing::warn!(
                "Option {:?}: driver reported {} choices, parsed {}",
                id,
                count,
                labels.len()
            );
        }
        Ok(labels)
    }

    /// Whether this camera implements single-shot still capture.
    pub fn supports_still_capture(&self) -> bool {
        let mut supported: i32 = 0;
        unsafe {
            if ols_ffi::ols_camera_supports_still(self.handle as *mut _, &mut supported)
                != ols_ffi::OLS_OK
            {
                return false;
            }
        }
        supported != 0
    }

    /// Take exactly one exposure and return its files.
    ///
    /// Blocking — libgphoto2 is synchronous and a full-frame download over USB
    /// 2.0 takes seconds. Call this from `spawn_blocking`, never on an async
    /// runtime thread.
    ///
    /// Dropping the returned handle releases bridge buffers without deleting
    /// camera files. Camera-file deletion is unavailable pending archive
    /// durability/recovery qualification; a local commit alone is insufficient.
    pub fn capture_still(
        &self,
        exposure_seconds: f64,
        sensitivity: Option<f64>,
    ) -> Result<(StillCapture, StillCaptureHandle)> {
        let request = ols_ffi::OlsStillRequest {
            exposure_seconds,
            // <= 0 tells the driver to leave the body's current ISO alone,
            // so a capture without an explicit gain never overrides what the
            // operator dialled in.
            sensitivity: sensitivity.unwrap_or(-1.0),
        };

        let mut raw: *mut ols_ffi::OlsStillCapture = std::ptr::null_mut();
        unsafe {
            check(ols_ffi::ols_camera_capture_still_snapshot(
                self.handle,
                &request,
                &mut raw,
            ))?;
        }
        if raw.is_null() {
            return Err(anyhow!("Bridge returned a null capture handle"));
        }
        // Own it immediately: every early return below must still release it.
        let handle = StillCaptureHandle { handle: raw };

        let mut info = ols_ffi::OlsStillCaptureInfo {
            started_at: 0.0,
            completed_at: 0.0,
            applied_exposure: 0.0,
            applied_sensitivity: 0.0,
            shutter_label: std::ptr::null(),
            iso_label: std::ptr::null(),
            asset_count: 0,
        };
        unsafe { check(ols_ffi::ols_capture_info(handle.handle, &mut info))? };

        let mut evidence = ols_ffi::OlsCaptureBundleInfo {
            evidence_version: 0,
            struct_size: std::mem::size_of::<ols_ffi::OlsCaptureBundleInfo>() as u32,
            event_complete: 0,
            discovered_count: 0,
            downloaded_count: 0,
            missing_count: 0,
            error_count: 0,
            bundle_complete: 0,
            evidence_json: std::ptr::null(),
            error: std::ptr::null(),
        };
        unsafe {
            check(ols_ffi::ols_capture_bundle_info(
                handle.handle,
                &mut evidence,
            ))?
        };
        let evidence_text = cstr_or_default(evidence.evidence_json);
        let parsed_evidence = serde_json::from_str::<serde_json::Value>(&evidence_text);
        let evidence_valid = parsed_evidence.as_ref().is_ok_and(|json| {
            json["evidence_version"] == 1
                && json["triggered"] == true
                && json["event_complete"] == true
                && json["completion_grace_complete"] == true
                && json["bundle_complete"] == true
        });
        let mut bundle = parsed_evidence.unwrap_or_else(|error| {
            serde_json::json!({
                "bundle_complete": false,
                "evidence_parse_error": error.to_string(),
                "unparsed_evidence": evidence_text,
            })
        });
        if !bundle.is_object() {
            bundle = serde_json::json!({"bundle_complete": false, "invalid_evidence": bundle});
        }
        let mut copy_errors = Vec::new();

        let mut assets = Vec::with_capacity(info.asset_count.max(0) as usize);
        for i in 0..info.asset_count {
            let mut a = ols_ffi::OlsCapturedAsset {
                camera_path: std::ptr::null(),
                extension: std::ptr::null(),
                bytes: std::ptr::null(),
                byte_length: 0,
            };
            if let Err(error) =
                unsafe { check(ols_ffi::ols_capture_asset_get(handle.handle, i, &mut a)) }
            {
                copy_errors.push(format!("Asset {i} could not be copied: {error}"));
                continue;
            }
            if a.bytes.is_null() || a.byte_length == 0 {
                copy_errors.push(format!("Asset {i} was empty"));
                continue;
            }
            assets.push(CapturedAsset {
                camera_path: cstr_or_default(a.camera_path),
                extension: cstr_or_default(a.extension).to_lowercase(),
                bytes: unsafe { std::slice::from_raw_parts(a.bytes, a.byte_length).to_vec() },
            });
        }

        let bundle_complete = evidence.evidence_version == 1
            && evidence.bundle_complete == 1
            && evidence.event_complete == 1
            && evidence.discovered_count > 0
            && evidence.discovered_count == evidence.downloaded_count
            && evidence.downloaded_count as usize == assets.len()
            && evidence.missing_count == 0
            && evidence.error_count == 0
            && evidence_valid
            && copy_errors.is_empty();
        bundle["bundle_complete"] = serde_json::json!(bundle_complete);
        bundle["copy_errors"] = serde_json::json!(copy_errors);
        bundle["discovered_count"] = serde_json::json!(evidence.discovered_count);
        bundle["downloaded_count"] = serde_json::json!(assets.len());
        bundle["missing_count"] = serde_json::json!(evidence.missing_count);

        let capture = StillCapture {
            started_at: info.started_at,
            completed_at: info.completed_at,
            applied_exposure_s: info.applied_exposure,
            applied_sensitivity: info.applied_sensitivity,
            shutter_label: cstr_or_default(info.shutter_label),
            iso_label: cstr_or_default(info.iso_label),
            assets,
            bundle,
            bundle_complete,
        };
        Ok((capture, handle))
    }

    /// Get supported option IDs
    pub fn supported_options(&self) -> Result<Vec<ols_ffi::OlsCamOptionId>> {
        let mut count: i32 = 0;
        unsafe {
            check(ols_ffi::ols_camera_option_count(
                self.handle as *mut _,
                &mut count,
            ))?;
        }

        let mut options = Vec::with_capacity(count as usize);
        for i in 0..count {
            let mut id = ols_ffi::OlsCamOptionId::Exp;
            unsafe {
                check(ols_ffi::ols_camera_option_id(
                    self.handle as *mut _,
                    i,
                    &mut id,
                ))?;
            }
            options.push(id);
        }

        Ok(options)
    }
}

impl Drop for OlsCamera {
    fn drop(&mut self) {
        // Stop stream first if active
        if self._callback_data.is_some() {
            let _ = self.stop_stream();
        }
        unsafe {
            ols_ffi::ols_camera_destroy(self.handle);
        }
    }
}

/// C callback trampoline - copies frame data into the Rust FrameData store
unsafe extern "C" fn frame_callback_trampoline(
    frame: *const ols_ffi::OlsFrame,
    user_data: *mut c_void,
) {
    if frame.is_null() || user_data.is_null() {
        return;
    }

    let cb_data = &*(user_data as *const CallbackData);
    if cb_data
        .active_generation
        .load(std::sync::atomic::Ordering::Acquire)
        != cb_data.stream_generation
    {
        return;
    }
    let frame_ref = &*frame;

    // A terminal SDK error invalidates publication immediately. Cleanup runs
    // off this callback: stopping here can join the current native thread.
    if frame_ref.format == ols_ffi::OlsStreamType::Error {
        let reason = if !frame_ref.data.is_null() && frame_ref.data_size > 0 {
            String::from_utf8_lossy(std::slice::from_raw_parts(
                frame_ref.data,
                frame_ref.data_size,
            ))
            .trim_end_matches('\0')
            .to_string()
        } else {
            "Camera stream reported a hardware error".to_string()
        };
        if let Some(handler) = &cb_data.fault_handler {
            handler(reason);
        }
        return;
    }

    if frame_ref.data.is_null() || frame_ref.data_size == 0 {
        return;
    }
    cb_data.stream_progress.lock().unwrap().last_frame = Some(std::time::Instant::now());

    // Copy frame data into owned Vec
    let data_slice = std::slice::from_raw_parts(frame_ref.data, frame_ref.data_size);

    // Copy the camera-authored original too, while it is still alive. Both
    // pointers are only valid for the duration of this callback, so there is
    // no deferring this: either we copy here or the CR3 is gone forever.
    let source = if !frame_ref.source_data.is_null() && frame_ref.source_data_size > 0 {
        let ext = if frame_ref.source_ext.is_null() {
            "raw".to_string()
        } else {
            CStr::from_ptr(frame_ref.source_ext)
                .to_string_lossy()
                .into_owned()
        };
        let bytes =
            std::slice::from_raw_parts(frame_ref.source_data, frame_ref.source_data_size).to_vec();
        Some(SourceImage { ext, bytes })
    } else {
        None
    };

    let frame_data = FrameData {
        stream_generation: cb_data.stream_generation,
        format: frame_ref.format,
        bayer: frame_ref.bayer,
        frame_counter: frame_ref.frame_counter,
        unix_timestamp: frame_ref.unix_timestamp,
        width: frame_ref.width,
        height: frame_ref.height,
        data: data_slice.to_vec(),
        source,
        saturation_adu: cb_data.saturation_adu,
        sensor_descriptor: stream_sensor_descriptor(
            &cb_data.format,
            frame_ref,
            !frame_ref.source_data.is_null() && frame_ref.source_data_size > 0,
            cb_data.saturation_adu,
        ),
    };

    if let Ok(mut store) = cb_data.latest_frame.lock() {
        if cb_data
            .active_generation
            .load(std::sync::atomic::Ordering::Acquire)
            == cb_data.stream_generation
        {
            *store = Some(frame_data);
        }
    }
}

#[cfg(test)]
mod analysis_descriptor_tests {
    use super::*;
    use ols_ffi::{OlsBayerType as B, OlsStreamType as F};

    fn format(kind: F) -> StreamFormat {
        StreamFormat {
            format: kind,
            width: 8,
            height: 8,
            bin: 1,
            roi_num: 1,
            roi_den: 1,
            framerate: -1.0,
        }
    }

    fn native_frame(kind: F, bayer: B, bytes: &[u8]) -> ols_ffi::OlsFrame {
        ols_ffi::OlsFrame {
            format: kind,
            bayer,
            frame_counter: 1,
            unix_timestamp: 1.0,
            width: 8,
            height: 8,
            data: bytes.as_ptr(),
            data_size: bytes.len(),
            source_data: std::ptr::null(),
            source_data_size: 0,
            source_ext: std::ptr::null(),
        }
    }

    #[test]
    fn raw_stream_requires_known_phase_unbinned_full_roi_and_exact_packing() {
        let bytes = vec![100_u8; 128];
        let mut format = format(F::Raw16);
        let mut frame = native_frame(F::Raw16, B::Rg, &bytes);
        let descriptor = stream_sensor_descriptor(&format, &frame, false, Some(16383.0)).unwrap();
        assert_eq!(
            descriptor.geometry.coordinate_system,
            "stream_buffer_pixels"
        );
        assert!(matches!(
            descriptor.color,
            SensorColor::Bayer {
                pattern: CfaPattern::Rggb,
                origin_x: 0,
                origin_y: 0
            }
        ));
        frame.bayer = B::Na;
        assert!(stream_sensor_descriptor(&format, &frame, false, None).is_none());
        frame.bayer = B::Rg;
        format.bin = 2;
        assert!(stream_sensor_descriptor(&format, &frame, false, None).is_none());
        format.bin = 1;
        format.roi_num = 1;
        format.roi_den = 2;
        assert!(stream_sensor_descriptor(&format, &frame, false, None).is_none());
        format.roi_den = 1;
        frame.data_size -= 1;
        assert!(stream_sensor_descriptor(&format, &frame, false, None).is_none());
    }

    #[test]
    fn legacy_decodes_and_previews_never_acquire_a_scientific_descriptor() {
        let bytes = vec![100_u8; 64];
        let frame = native_frame(F::Mono8, B::Na, &bytes);
        assert!(stream_sensor_descriptor(&format(F::Mono8), &frame, true, None).is_none());
        assert!(stream_sensor_descriptor(&format(F::Mono8), &frame, false, None).is_some());
        for kind in [F::Mjpeg, F::Rgb24, F::Rgb48, F::Yuv2] {
            let frame = native_frame(kind, B::Na, &bytes);
            assert!(stream_sensor_descriptor(&format(kind), &frame, false, None).is_none());
        }
    }
}

//! Safe Rust wrapper around OLS camera FFI.
//!
//! Provides `OlsCameraDriver` and `OlsCamera` types that manage
//! the lifecycle of OLS C++ objects and convert between C and Rust types.

use super::ols_ffi;
use anyhow::{anyhow, Result};
use std::ffi::{CStr, CString};
use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

/// Get the last error message from the OLS bridge.
fn last_error() -> String {
    unsafe {
        let ptr = ols_ffi::ols_last_error();
        if ptr.is_null() {
            "Unknown error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

/// Check an OLS error code and convert to Result.
fn check(code: i32) -> Result<()> {
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
    pub framerate: f32,
}

impl From<ols_ffi::OlsStreamFormat> for StreamFormat {
    fn from(f: ols_ffi::OlsStreamFormat) -> Self {
        Self {
            format: f.format,
            width: f.width,
            height: f.height,
            bin: f.bin,
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

/// A captured frame with owned data
#[derive(Debug, Clone)]
pub struct FrameData {
    pub format: ols_ffi::OlsStreamType,
    pub bayer: ols_ffi::OlsBayerType,
    pub frame_counter: i32,
    pub unix_timestamp: f64,
    pub width: i32,
    pub height: i32,
    pub data: Vec<u8>,
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
}

// Safety: OLS camera uses internal locking
unsafe impl Send for OlsCamera {}

struct CallbackData {
    latest_frame: Arc<Mutex<Option<FrameData>>>,
}

impl OlsCamera {
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
    pub fn start_stream(
        &mut self,
        format: &StreamFormat,
        frame_store: Arc<Mutex<Option<FrameData>>>,
    ) -> Result<()> {
        let cb_data = Box::new(CallbackData {
            latest_frame: frame_store,
        });
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
        unsafe {
            check(ols_ffi::ols_camera_set_param(
                self.handle as *mut _,
                id,
                value,
            ))
        }
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
    let frame_ref = &*frame;

    // Skip error frames
    if frame_ref.format == ols_ffi::OlsStreamType::Error {
        return;
    }

    // Copy frame data into owned Vec
    let data_slice = std::slice::from_raw_parts(frame_ref.data, frame_ref.data_size);

    let frame_data = FrameData {
        format: frame_ref.format,
        bayer: frame_ref.bayer,
        frame_counter: frame_ref.frame_counter,
        unix_timestamp: frame_ref.unix_timestamp,
        width: frame_ref.width,
        height: frame_ref.height,
        data: data_slice.to_vec(),
    };

    if let Ok(mut store) = cb_data.latest_frame.lock() {
        *store = Some(frame_data);
    }
}

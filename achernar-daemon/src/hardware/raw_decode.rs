//! Safe wrapper around the bridge's LibRaw decode.
//!
//! Turns the bytes of a camera-authored RAW container into a 16-bit Bayer
//! plane of **untouched sensor ADU** — no black subtraction, no scaling, no
//! white balance. Everything needed to calibrate the plane later travels
//! alongside it in [`RawInfo`]. The long argument for that choice lives on
//! `ols_decode_raw` in `ols-bridge/include/ols_bridge.h`; the one-line version
//! is that a linear rescale adds no information, severs ADU from electrons, and
//! cannot be undone.
//!
//! # Memory
//!
//! [`RawImage`] borrows, it does not copy. `pixels()` hands out a slice into
//! the C++ allocation, and the borrow checker ties that slice's lifetime to the
//! handle. A 24 MP frame is ~48 MB; copying it into a `Vec` on the way to the
//! FITS encoder would double that for no reason, which is why
//! [`crate::fits::encode_bayer_u16`] takes a slice.

use super::camera::{check, last_error};
use super::ols_ffi;
use anyhow::{anyhow, Result};
use std::ffi::CStr;

/// Whether this build can decode RAW at all.
///
/// Worth checking before a capture rather than after: a body set to RAW-only
/// on a bridge without LibRaw will archive every exposure and derive no FITS,
/// and the operator would rather learn that at connect time.
pub fn libraw_available() -> bool {
    let flags = unsafe { ols_ffi::ols_driver_capability_flags() as u32 };
    flags & ols_ffi::OLS_CAP_DECODE_LIBRAW != 0
}

/// What the decode deliberately did not apply.
#[derive(Debug, Clone)]
pub struct RawInfo {
    pub width: u32,
    pub height: u32,
    pub bayer: ols_ffi::OlsBayerType,
    /// Storage depth: always 16. Use [`RawInfo::white_level`] for the sensor's
    /// real range — a 14-bit body tops out at 16383, and a saturation test
    /// against 65535 can never fire.
    pub bits_per_sample: u32,
    /// Global black pedestal in ADU. Not subtracted from the pixels.
    pub black_level: i32,
    /// Per-CFA-channel black offsets, added to `black_level`.
    pub cblack: [i32; 4],
    /// Dimensions of LibRaw's 2-D black table; `[0, 0]` when there is none.
    /// Anything past 2x2 is not captured by `cblack`, so a non-trivial value
    /// means the recorded black level is incomplete and calibration should come
    /// from bias frames.
    pub cblack_pattern: [i32; 2],
    /// Saturation point in ADU. Not normalised to.
    pub white_level: i32,
    /// As-shot white balance multipliers. Recorded, never applied.
    pub cam_mul: [f32; 4],
    pub make: String,
    pub model: String,
}

impl RawInfo {
    /// Whether `cblack` fully describes this body's black level.
    pub fn black_level_is_complete(&self) -> bool {
        let cells = self.cblack_pattern[0].saturating_mul(self.cblack_pattern[1]);
        cells <= 4
    }
}

/// An owned decoded frame. The pixel buffer lives in C++ and is released on
/// `Drop`.
pub struct RawImage {
    handle: *mut ols_ffi::OlsRawImage,
    info: RawInfo,
}

// The handle is a plain heap allocation with no thread affinity and no interior
// mutability: after `decode` returns, nothing writes to it until `Drop`. Send so
// a decode can be moved off a `spawn_blocking` worker; Sync because every
// accessor takes `&self` and only reads.
unsafe impl Send for RawImage {}
unsafe impl Sync for RawImage {}

impl RawImage {
    pub fn info(&self) -> &RawInfo {
        &self.info
    }

    /// The pixel plane: `width * height * 2` bytes, row-major, tightly packed,
    /// native-endian `u16`.
    ///
    /// Borrowed from the C++ allocation — the returned slice cannot outlive
    /// `self`, which is the whole point of the handle.
    pub fn pixels(&self) -> &[u8] {
        let mut len: usize = 0;
        let ptr = unsafe { ols_ffi::ols_rawimage_pixels(self.handle, &mut len) };
        if ptr.is_null() || len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

impl Drop for RawImage {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ols_ffi::ols_rawimage_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

impl std::fmt::Debug for RawImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawImage").field("info", &self.info).finish()
    }
}

/// Decode a RAW container into a full-frame Bayer plane.
///
/// `bytes` is borrowed for the duration of the call only — the bridge copies
/// what it keeps, so the caller may drop the container immediately after.
pub fn decode(bytes: &[u8]) -> Result<RawImage> {
    decode_region(bytes, None)
}

/// Decode, optionally selecting a centred `(width, height)` sub-region.
///
/// The sub-region offset is forced even by the bridge, so the Bayer phase
/// reported in [`RawInfo`] holds for the returned plane regardless of the
/// region requested.
pub fn decode_region(bytes: &[u8], region: Option<(u32, u32)>) -> Result<RawImage> {
    if bytes.is_empty() {
        return Err(anyhow!("Cannot decode an empty buffer"));
    }

    let (w, h) = match region {
        Some((w, h)) => (
            i32::try_from(w).map_err(|_| anyhow!("region width {w} out of range"))?,
            i32::try_from(h).map_err(|_| anyhow!("region height {h} out of range"))?,
        ),
        None => (-1, -1),
    };

    let mut handle: *mut ols_ffi::OlsRawImage = std::ptr::null_mut();
    // bin is 1: binning raw ADU would change saturation semantics the metadata
    // cannot express. Bin after calibration.
    let code = unsafe {
        ols_ffi::ols_decode_raw(bytes.as_ptr(), bytes.len(), w, h, 1, &mut handle)
    };
    check(code)?;
    if handle.is_null() {
        return Err(anyhow!(
            "Bridge reported success but returned no image: {}",
            last_error()
        ));
    }

    // Wrap immediately, before anything else can fail, so every early return
    // below still runs Drop.
    let mut image = RawImage {
        handle,
        info: RawInfo {
            width: 0,
            height: 0,
            bayer: ols_ffi::OlsBayerType::Na,
            bits_per_sample: 16,
            black_level: 0,
            cblack: [0; 4],
            cblack_pattern: [0; 2],
            white_level: 0,
            cam_mul: [0.0; 4],
            make: String::new(),
            model: String::new(),
        },
    };

    let mut raw_info = std::mem::MaybeUninit::<ols_ffi::OlsRawImageInfo>::uninit();
    check(unsafe { ols_ffi::ols_rawimage_info(image.handle, raw_info.as_mut_ptr()) })?;
    let raw_info = unsafe { raw_info.assume_init() };

    if raw_info.width <= 0 || raw_info.height <= 0 {
        return Err(anyhow!(
            "Decoder returned a {}x{} image",
            raw_info.width,
            raw_info.height
        ));
    }

    image.info = RawInfo {
        width: raw_info.width as u32,
        height: raw_info.height as u32,
        bayer: raw_info.bayer,
        bits_per_sample: raw_info.bits_per_sample.max(0) as u32,
        black_level: raw_info.black_level,
        cblack: raw_info.cblack,
        cblack_pattern: raw_info.cblack_pattern,
        white_level: raw_info.white_level,
        cam_mul: raw_info.cam_mul,
        make: cstr(raw_info.make),
        model: cstr(raw_info.model),
    };

    // The bridge promises a tightly-packed plane. Check it here rather than
    // trusting it: every consumer indexes this buffer as width*height*2, and a
    // short buffer is an out-of-bounds read rather than a wrong picture.
    let expected = image.info.width as usize * image.info.height as usize * 2;
    let actual = image.pixels().len();
    if actual != expected {
        return Err(anyhow!(
            "Decoder returned {actual} bytes for a {}x{} 16-bit plane (expected {expected})",
            image.info.width,
            image.info.height
        ));
    }

    Ok(image)
}

fn cstr(ptr: *const std::os::raw::c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_buffer_is_rejected_before_ffi() {
        assert!(decode(&[]).is_err());
    }

    #[test]
    fn black_level_completeness() {
        let mut info = RawInfo {
            width: 1,
            height: 1,
            bayer: ols_ffi::OlsBayerType::Rg,
            bits_per_sample: 16,
            black_level: 2048,
            cblack: [0; 4],
            cblack_pattern: [0, 0],
            white_level: 16383,
            cam_mul: [1.0; 4],
            make: "Canon".into(),
            model: "EOS R5".into(),
        };
        assert!(info.black_level_is_complete());

        // A 2x2 table folds into cblack[0..4].
        info.cblack_pattern = [2, 2];
        assert!(info.black_level_is_complete());

        // Anything larger does not, and must be recorded as incomplete.
        info.cblack_pattern = [6, 6];
        assert!(!info.black_level_is_complete());
    }
}

//! Safe wrapper around the bridge's LibRaw decode.
//!
//! RAW containers become a borrowed 16-bit plane, unchanged by the application
//! since LibRaw unpack: no additional black subtraction, scaling or white
//! balance. Camera/container/LibRaw transformations may already precede that
//! output. Versioned decoder evidence and the complete black table travel in
//! [`RawInfo`]. Calibration remains a separate signed/float operation.
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
use anyhow::{anyhow, ensure, Result};
use serde::{Deserialize, Serialize};
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

/// Versioned decoder evidence. All geometry is in LibRaw's unrotated raw-array
/// coordinates. Metadata limits are not independently measured detector limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawDecodeMetadata {
    pub schema_version: u32,
    pub decoder_version: String,
    pub unpack_function: String,
    pub processing_version: String,
    pub sample_domain: String,
    pub raw_width: u32,
    pub raw_height: u32,
    pub visible_width: u32,
    pub visible_height: u32,
    pub left_margin: u32,
    pub top_margin: u32,
    pub source_pitch_bytes: usize,
    pub crop_x: u32,
    pub crop_y: u32,
    pub width: u32,
    pub height: u32,
    pub pixel_aspect: f64,
    pub orientation: i32,
    pub filters: u32,
    pub color_description: String,
    pub bayer_at_crop_origin: String,
    pub black_table: Vec<u32>,
    pub black_table_complete: bool,
    pub linear_max: [u32; 4],
    pub white_level_source: String,
    pub nonlinearity_state: String,
    pub black_subtracted: bool,
    pub scaled: bool,
    pub white_balance_applied: bool,
    pub demosaic_applied: bool,
    pub rotation_applied: bool,
    pub bin: u32,
    pub unpack_options: u32,
    pub unpack_specials: Option<u32>,
    pub shot_select: u32,
}

impl RawDecodeMetadata {
    fn validate(&self, info: &RawInfo) -> Result<()> {
        ensure!(
            self.schema_version == 1
                && self.processing_version == "achernar-libraw-unpack-v1"
                && self.sample_domain == "libraw_unpack_output_adu"
                && !self.decoder_version.is_empty(),
            "Unsupported or incomplete RAW decoder metadata"
        );
        ensure!(
            !self.black_subtracted
                && !self.scaled
                && !self.white_balance_applied
                && !self.demosaic_applied
                && !self.rotation_applied
                && self.bin == 1,
            "RAW decoder applied an incompatible transformation"
        );
        let x1 = self.crop_x as u64 + self.width as u64;
        let y1 = self.crop_y as u64 + self.height as u64;
        ensure!(
            self.width == info.width
                && self.height == info.height
                && self.width > 0
                && self.height > 0
                && self.visible_width > 0
                && self.visible_height > 0
                && self.left_margin as u64 + self.visible_width as u64 <= self.raw_width as u64
                && self.top_margin as u64 + self.visible_height as u64 <= self.raw_height as u64
                && self.crop_x >= self.left_margin
                && self.crop_y >= self.top_margin
                && x1 <= self.left_margin as u64 + self.visible_width as u64
                && y1 <= self.top_margin as u64 + self.visible_height as u64
                && self.source_pitch_bytes as u64 >= self.raw_width as u64 * 2
                && self.pixel_aspect.is_finite()
                && self.pixel_aspect > 0.0,
            "RAW decoder geometry is inconsistent"
        );
        let expected_pattern = match info.bayer {
            ols_ffi::OlsBayerType::Rg => "RGGB",
            ols_ffi::OlsBayerType::Bg => "BGGR",
            ols_ffi::OlsBayerType::Gr => "GRBG",
            ols_ffi::OlsBayerType::Gb => "GBRG",
            ols_ffi::OlsBayerType::Na => "",
        };
        ensure!(
            self.bayer_at_crop_origin == expected_pattern,
            "RAW CFA metadata disagrees with the plane"
        );
        ensure!(
            info.bits_per_sample == 16 && (0..=65535).contains(&info.white_level),
            "RAW sample/white metadata exceeds the unpack domain"
        );
        ensure!(
            self.white_level_source == "libraw_unpack_declared_maximum"
                && self.nonlinearity_state == "not_measured",
            "Unsupported RAW threshold provenance"
        );
        ensure!(
            self.black_table_complete && info.black_table_dimensions_match(self.black_table.len()),
            "Incomplete or malformed RAW black table"
        );
        Ok(())
    }
}

/// What the decode deliberately did not apply.
#[derive(Debug, Clone)]
pub struct RawInfo {
    pub width: u32,
    pub height: u32,
    pub bayer: ols_ffi::OlsBayerType,
    /// Storage depth: always 16. Container capacity is separate from the
    /// declared unpack-domain white code and measured detector linearity.
    pub bits_per_sample: u32,
    /// Global black pedestal in ADU. Not subtracted from the pixels.
    pub black_level: i32,
    /// Per-CFA-channel black offsets, added to `black_level`.
    pub cblack: [i32; 4],
    /// Dimensions of LibRaw's 2-D black table; `[0, 0]` when there is none.
    /// Spatial entries are separate from per-channel offsets, even at 2x2.
    /// Complete table values are carried in decoder_metadata.
    pub cblack_pattern: [i32; 2],
    /// Declared unpack-domain maximum; not a measured nonlinearity threshold.
    pub white_level: i32,
    /// As-shot white balance multipliers. Recorded, never applied.
    pub cam_mul: [f32; 4],
    pub make: String,
    pub model: String,
    pub decoder_metadata: Option<RawDecodeMetadata>,
}

impl RawInfo {
    fn black_table_dimensions_match(&self, count: usize) -> bool {
        let [rows, cols] = self.cblack_pattern;
        rows >= 0
            && cols >= 0
            && ((rows == 0) == (cols == 0))
            && (rows as usize).checked_mul(cols as usize) == Some(count)
    }
    /// Complete metadata means both per-channel offsets and all spatial entries.
    pub fn black_level_is_complete(&self) -> bool {
        match &self.decoder_metadata {
            Some(metadata) => {
                metadata.black_table_complete
                    && self.black_table_dimensions_match(metadata.black_table.len())
            }
            None => self.cblack_pattern == [0, 0],
        }
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
        f.debug_struct("RawImage")
            .field("info", &self.info)
            .finish()
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
    let code =
        unsafe { ols_ffi::ols_decode_raw(bytes.as_ptr(), bytes.len(), w, h, 1, &mut handle) };
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
            decoder_metadata: None,
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
        decoder_metadata: None,
    };

    let metadata_ptr = unsafe { ols_ffi::ols_rawimage_metadata_json(image.handle) };
    ensure!(!metadata_ptr.is_null(), "RAW decoder metadata missing");
    let metadata: RawDecodeMetadata =
        serde_json::from_slice(unsafe { CStr::from_ptr(metadata_ptr) }.to_bytes())?;
    metadata.validate(&image.info)?;
    image.info.decoder_metadata = Some(metadata);

    // The bridge promises a tightly-packed plane. Check it here rather than
    // trusting it: every consumer indexes this buffer as width*height*2, and a
    // short buffer is an out-of-bounds read rather than a wrong picture.
    let expected = (image.info.width as usize)
        .checked_mul(image.info.height as usize)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| anyhow!("RAW sample dimensions overflow"))?;
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
            decoder_metadata: None,
        };
        assert!(info.black_level_is_complete());

        // A 2x2 spatial table is not represented by four channel offsets.
        info.cblack_pattern = [2, 2];
        assert!(!info.black_level_is_complete());

        // Anything larger does not, and must be recorded as incomplete.
        info.cblack_pattern = [6, 6];
        assert!(!info.black_level_is_complete());
    }

    #[test]
    #[ignore = "set ACHERNAR_RAW_FIXTURE to an existing RAW; no camera access"]
    fn existing_raw_decoder_metadata_rejects_contradictions() {
        let bytes = std::fs::read(std::env::var("ACHERNAR_RAW_FIXTURE").unwrap()).unwrap();
        let image = decode_region(&bytes, Some((128, 128))).unwrap();
        let info = image.info();
        let metadata = info.decoder_metadata.as_ref().unwrap();
        metadata.validate(info).unwrap();
        let mut modified = metadata.clone();
        modified.scaled = true;
        assert!(modified.validate(info).is_err());
        let mut modified = metadata.clone();
        modified.source_pitch_bytes = 1;
        assert!(modified.validate(info).is_err());
        let mut modified = metadata.clone();
        modified.crop_x = modified.raw_width;
        assert!(modified.validate(info).is_err());
        let mut modified = metadata.clone();
        modified.bayer_at_crop_origin = "unknown".into();
        assert!(modified.validate(info).is_err());
        let mut modified = metadata.clone();
        modified.white_level_source = "container_capacity".into();
        assert!(modified.validate(info).is_err());
        let mut modified = metadata.clone();
        let mut info = info.clone();
        info.cblack_pattern = [2, 2];
        modified.black_table = vec![10, 11, 12, 13];
        modified.validate(&info).unwrap();
        info.decoder_metadata = Some(modified.clone());
        assert!(info.black_level_is_complete());
        modified.black_table.pop();
        assert!(modified.validate(&info).is_err());
        info.decoder_metadata = Some(modified);
        assert!(!info.black_level_is_complete());
        info.white_level = 65536;
        assert!(metadata.validate(&info).is_err());
    }
}

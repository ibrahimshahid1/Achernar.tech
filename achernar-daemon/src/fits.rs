//! Minimal, dependency-free FITS writer for single-frame captures.
//!
//! Writes a standard primary HDU (header + image data) so captures are readable
//! by astropy, DS9, PixInsight, etc. Handles 8/16-bit mono and raw frames,
//! interleaved RGB (stored as a planar 3-axis cube), and MJPEG previews (decoded
//! to RGB first via the `image` crate already in the build).
//!
//! This intentionally does not depend on cfitsio / OLS `fitsmat.cpp`: a single
//! uncompressed primary HDU is simple to emit correctly in pure Rust, and avoids
//! widening the native bridge. OLS's writer can be bridged later if we need
//! tiled compression or richer WCS headers.

use crate::hardware::camera::FrameData;
use crate::hardware::ols_ffi::{OlsBayerType, OlsStreamType};
use anyhow::{anyhow, Result};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

/// FITS blocks are 2880 bytes; both the header and the data unit are padded to a
/// whole number of blocks.
const BLOCK: usize = 2880;

/// Sensor calibration constants for a frame whose pixels are untouched ADU.
///
/// These cards travel together or not at all. Their presence is the file's
/// assertion that nothing has been subtracted from or multiplied into the
/// pixels since LibRaw unpack. Camera/decoder transformations can precede
/// unpack; these constants are metadata, not independently measured calibration. Emitting `BLKLEVEL` on a frame that has already had
/// black subtracted would be worse than emitting nothing, because a downstream
/// pipeline would subtract it twice.
#[derive(Debug, Clone)]
pub struct SensorCalibration {
    /// Black pedestal in ADU, **not** subtracted.
    pub black_level: i32,
    /// Per-CFA-channel black offsets, added to `black_level`. `[0; 4]` on
    /// bodies with a single pedestal.
    pub cblack: [i32; 4],
    pub cblack_pattern: [i32; 2],
    /// Declared unpack-domain white code in ADU; zero means unavailable.
    /// It is separate from container capacity and measured nonlinearity.
    pub white_level: i32,
    /// Offset of the Bayer pattern origin from the image origin. Zero when the
    /// decoder's crop preserved the phase. Written explicitly rather than
    /// omitted so a reader can tell "the offset is zero" from "nobody
    /// considered the offset".
    pub bayer_x_offset: i32,
    pub bayer_y_offset: i32,
    pub decoder_metadata: Option<crate::hardware::raw_decode::RawDecodeMetadata>,
}

/// Acquisition metadata recorded in the FITS header.
///
/// `exposure_s` and `gain` must be the values the hardware *actually applied*,
/// read back after setting them — never the values that were requested. On a
/// DSLR the two routinely differ: the body quantises to its shutter ladder, so
/// a 3.5 s request becomes a 4 s exposure. A header that records the request
/// is a header that lies about the photograph.
#[derive(Debug, Clone)]
pub struct FitsMeta {
    pub exposure_s: f64,
    pub gain: u32,
    pub temperature_c: f64,
    pub binning: u32,
    pub instrument: String,
    /// UTC start-of-exposure, ISO-8601 (FITS `DATE-OBS`).
    pub date_obs: String,
    /// ISO speed for DSLR bodies, where `gain` is a selection index rather
    /// than a sensor gain. `None` for astronomy cameras.
    pub iso: Option<f64>,
    /// The camera's own shutter-speed label ("1/125", "30", "bulb"), recorded
    /// verbatim so the exposure can be traced back to a specific body setting.
    pub shutter_label: Option<String>,
    /// Filename of the preserved original (`<id>.cr3`), when one was archived.
    pub source_file: Option<String>,
    /// Present when the pixels are raw sensor ADU. See [`SensorCalibration`].
    pub calibration: Option<SensorCalibration>,
}

/// A header card supplied by the caller rather than derived from
/// [`FitsMeta`]: the equipment keywords of a capture (equipment spec H1:
/// `OBJCTRA`, `RA`, `PIERSIDE`, `FOCPOS`, `FILTER`, `SITELAT`, ...).
///
/// A card is written only when its value is known. The caller omits a card
/// whose value is missing; a non-finite float is dropped here as a last line
/// of defence, so a header never carries `NAN` or a placeholder 0.
#[derive(Debug, Clone, PartialEq)]
pub struct FitsCard {
    pub key: String,
    pub value: FitsValue,
    pub comment: String,
}

/// The value of a [`FitsCard`].
#[derive(Debug, Clone, PartialEq)]
pub enum FitsValue {
    Str(String),
    Float(f64),
    Int(i64),
}

impl FitsCard {
    pub fn string(key: &str, value: impl Into<String>, comment: &str) -> Self {
        Self::new(key, FitsValue::Str(value.into()), comment)
    }

    pub fn float(key: &str, value: f64, comment: &str) -> Self {
        Self::new(key, FitsValue::Float(value), comment)
    }

    pub fn int(key: &str, value: i64, comment: &str) -> Self {
        Self::new(key, FitsValue::Int(value), comment)
    }

    fn new(key: &str, value: FitsValue, comment: &str) -> Self {
        Self {
            key: key.to_string(),
            value,
            comment: comment.to_string(),
        }
    }

    /// A FITS keyword: 1-8 characters of `A-Z`, `0-9`, `-` and `_`.
    fn valid_key(&self) -> bool {
        !self.key.is_empty()
            && self.key.len() <= 8
            && self
                .key
                .bytes()
                .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
    }

    /// The 80-column card, or `None` when the card cannot be written
    /// truthfully (an invalid keyword, a non-finite number).
    fn render(&self) -> Option<String> {
        if !self.valid_key() {
            return None;
        }
        Some(match &self.value {
            FitsValue::Str(value) => card_string(&self.key, value, &self.comment),
            FitsValue::Float(value) if value.is_finite() => {
                card_float(&self.key, *value, &self.comment)
            }
            FitsValue::Float(_) => return None,
            FitsValue::Int(value) => card_int(&self.key, *value, &self.comment),
        })
    }
}

/// Right ascension in hours as `HH MM SS.ss` (the `OBJCTRA` convention).
/// Rounding carries, so 23h59m59.999s is written `00 00 00.00`, never
/// `23 59 60.00`.
pub fn sexagesimal_ra(ra_h: f64) -> Option<String> {
    if !ra_h.is_finite() {
        return None;
    }
    let centiseconds = (ra_h.rem_euclid(24.0) * 360_000.0).round() as i64 % 8_640_000;
    let (h, rest) = (centiseconds / 360_000, centiseconds % 360_000);
    let (m, cs) = (rest / 6_000, rest % 6_000);
    Some(format!("{h:02} {m:02} {:02}.{:02}", cs / 100, cs % 100))
}

/// Declination in degrees as `+DD MM SS.s` (the `OBJCTDEC` convention).
pub fn sexagesimal_dec(dec_deg: f64) -> Option<String> {
    if !dec_deg.is_finite() || dec_deg.abs() > 90.0 {
        return None;
    }
    let sign = if dec_deg < 0.0 { '-' } else { '+' };
    let deciseconds = (dec_deg.abs() * 36_000.0).round() as i64;
    let (d, rest) = (deciseconds / 36_000, deciseconds % 36_000);
    let (m, ds) = (rest / 600, rest % 600);
    Some(format!("{sign}{d:02} {m:02} {:02}.{}", ds / 10, ds % 10))
}

/// Directory captures are written to. `./frames`, overridable via
/// `ACHERNAR_FRAMES_DIR`.
pub fn frames_dir() -> String {
    std::env::var("ACHERNAR_FRAMES_DIR").unwrap_or_else(|_| "frames".to_string())
}

/// Encode `frame` as a complete FITS file in memory, with caller-supplied
/// header cards (equipment H1; none for a bare capture).
///
/// Returns bytes rather than writing them, because a capture is committed by
/// [`crate::storage::capture_store`], which stages every asset into one
/// directory, hashes it, and renames the bundle into place atomically. A
/// function that writes its own file behind the store's back produces exactly
/// the orphan the staging protocol exists to prevent.
pub fn encode_capture_with(
    frame: &FrameData,
    meta: &FitsMeta,
    extra: &[FitsCard],
) -> Result<Vec<u8>> {
    let (bitpix, naxes, data) = frame_to_pixels(frame)?;
    let bayer = bayer_pattern(frame.bayer);
    let header = build_header(bitpix, &naxes, meta, bayer, extra);

    let mut out = Vec::with_capacity(header.len() + data.len());
    out.extend_from_slice(&header);
    out.extend_from_slice(&data);
    Ok(out)
}

/// Encode a canonical linear analysis plane without integer conversion,
/// clipping, scaling or acquisition metadata inferred from a request.
/// NaNs are retained; the analysis bundle must carry their invalid mask.
pub fn encode_linear_analysis(
    pixels: &[f32],
    width: usize,
    height: usize,
    units: &str,
    processing_version: &str,
) -> Result<Vec<u8>> {
    validate_analysis_shape(pixels.len(), width, height)?;
    let mut out = analysis_header(-32, width, height, units, processing_version);
    for value in pixels {
        out.extend_from_slice(&value.to_be_bytes());
    }
    pad(&mut out, 0);
    Ok(out)
}

/// A separate unsigned byte FITS plane: zero is clear, one is flagged.
pub fn encode_analysis_mask(
    pixels: &[u8],
    width: usize,
    height: usize,
    processing_version: &str,
) -> Result<Vec<u8>> {
    validate_analysis_shape(pixels.len(), width, height)?;
    if pixels.iter().any(|value| *value > 1) {
        return Err(anyhow!("Analysis masks must contain only zero or one"));
    }
    let mut out = analysis_header(8, width, height, "1", processing_version);
    out.extend_from_slice(pixels);
    pad(&mut out, 0);
    Ok(out)
}

fn validate_analysis_shape(length: usize, width: usize, height: usize) -> Result<()> {
    if width == 0 || height == 0 || width.checked_mul(height) != Some(length) {
        return Err(anyhow!("Analysis FITS shape does not match its samples"));
    }
    Ok(())
}

fn analysis_header(
    bitpix: i32,
    width: usize,
    height: usize,
    units: &str,
    processing_version: &str,
) -> Vec<u8> {
    // The JSON descriptor is authoritative for full configuration, masks and
    // geometry; the FITS itself records enough to identify that descriptor.
    let cards = [
        card_logical("SIMPLE", true, "conforms to FITS standard"),
        card_int("BITPIX", bitpix as i64, ""),
        card_int("NAXIS", 2, ""),
        card_int("NAXIS1", width as i64, ""),
        card_int("NAXIS2", height as i64, ""),
        card_string("BUNIT", units, "analysis units; see descriptor"),
        card_string("ANALVER", processing_version, "processing version"),
        card_string("GEOMREF", "analysis.json", "geometry and mask descriptor"),
        "END".to_string(),
    ];
    let mut bytes = Vec::new();
    for card in cards {
        let mut card = card.into_bytes();
        card.truncate(80);
        card.resize(80, b' ');
        bytes.extend_from_slice(&card);
    }
    pad(&mut bytes, b' ');
    bytes
}

#[cfg(test)]
mod analysis_encoding_tests {
    use super::*;

    #[test]
    fn float_fits_preserves_fraction_negative_and_nan_bits() {
        let values = [-4.25f32, 0.125, 65535.0, f32::NAN];
        let bytes = encode_linear_analysis(&values, 2, 2, "ADU", "linear_v1").unwrap();
        let header = std::str::from_utf8(&bytes[..BLOCK]).unwrap();
        assert!(header.contains("BITPIX  =                  -32"));
        assert!(!header.contains("BZERO"));
        assert!(!header.contains("EXPTIME"));
        for (sample, encoded) in values.iter().zip(bytes[BLOCK..BLOCK + 16].chunks_exact(4)) {
            assert_eq!(
                sample.to_bits(),
                u32::from_be_bytes(encoded.try_into().unwrap())
            );
        }
        assert_eq!(bytes.len() % BLOCK, 0);
    }

    #[test]
    fn mask_fits_keeps_binary_samples_and_refuses_bad_shape() {
        let bytes = encode_analysis_mask(&[0, 1, 1, 0], 2, 2, "linear_v1").unwrap();
        assert_eq!(&bytes[BLOCK..BLOCK + 4], &[0, 1, 1, 0]);
        assert!(encode_analysis_mask(&[2], 1, 1, "linear_v1").is_err());
        assert!(encode_linear_analysis(&[1.0], 0, 1, "ADU", "linear_v1").is_err());
        assert!(encode_linear_analysis(&[1.0], 2, 1, "ADU", "linear_v1").is_err());
    }

    #[test]
    fn camera_jpeg_can_archive_a_fits_without_claiming_linear_analysis() {
        let image = image::DynamicImage::new_rgb8(2, 2);
        let mut jpeg = std::io::Cursor::new(Vec::new());
        image
            .write_to(&mut jpeg, image::ImageOutputFormat::Jpeg(90))
            .unwrap();
        let meta = FitsMeta {
            exposure_s: 1.0,
            gain: 0,
            temperature_c: f64::NAN,
            binning: 1,
            instrument: "fixture".into(),
            date_obs: "2026-10-12T00:00:00Z".into(),
            iso: None,
            shutter_label: None,
            source_file: None,
            calibration: None,
        };
        let derived = fits_from_source(jpeg.get_ref(), "jpg", &meta, &[])
            .unwrap()
            .unwrap();
        assert!(!derived.bytes.is_empty());
        assert!(derived.analysis_assets.is_empty());
        assert!(derived
            .analysis_pending_reason
            .unwrap()
            .contains("processed image domain"));
    }

    /// Explicit local file opt-in; never connects to or triggers a camera.
    #[test]
    #[ignore = "set ACHERNAR_RAW_FIXTURE to an existing RAW; optional ACHERNAR_RAW_FIXTURE_OUTPUT exports for Astropy"]
    fn existing_raw_crop_exports_production_sensor_and_analysis_assets() {
        let path = std::env::var("ACHERNAR_RAW_FIXTURE").expect("RAW fixture path required");
        let bytes = fs::read(path).unwrap();
        let image = crate::hardware::raw_decode::decode_region(&bytes, Some((128, 128))).unwrap();
        let info = image.info();
        assert!(info.black_level_is_complete());
        let metadata = info.decoder_metadata.as_ref().unwrap();
        let meta = FitsMeta {
            exposure_s: f64::NAN,
            gain: 0,
            temperature_c: f64::NAN,
            binning: 1,
            instrument: format!("{} {}", info.make, info.model),
            date_obs: String::new(),
            iso: None,
            shutter_label: None,
            source_file: None,
            calibration: Some(SensorCalibration {
                black_level: info.black_level,
                cblack: info.cblack,
                cblack_pattern: info.cblack_pattern,
                white_level: info.white_level,
                bayer_x_offset: 0,
                bayer_y_offset: 0,
                decoder_metadata: Some(metadata.clone()),
            }),
        };
        let sensor = encode_bayer_u16(
            image.pixels(),
            info.width,
            info.height,
            &meta,
            bayer_pattern(info.bayer),
        )
        .unwrap();
        let assets = raw_analysis_assets(info, image.pixels(), &sensor).unwrap();
        assert_eq!(assets.len(), 6);
        for asset in &assets {
            assert_eq!(
                asset.derived_from,
                Some(crate::storage::capture_store::AssetRole::SensorFits)
            );
            assert_eq!(asset.derived_from_filename.as_deref(), Some("image.fits"));
        }
        if let Ok(directory) = std::env::var("ACHERNAR_RAW_FIXTURE_OUTPUT") {
            let directory = PathBuf::from(directory);
            fs::create_dir_all(directory.join("derived")).unwrap();
            fs::create_dir_all(directory.join("analysis")).unwrap();
            fs::write(directory.join("derived/image.fits"), sensor).unwrap();
            fs::write(directory.join("sensor.u16"), image.pixels()).unwrap();
            fs::write(
                directory.join("raw_info.json"),
                serde_json::to_vec_pretty(&serde_json::json!({
                    "metadata": metadata, "black_level": info.black_level, "cblack": info.cblack,
                    "cblack_pattern": info.cblack_pattern, "white_level": info.white_level,
                    "cam_mul": info.cam_mul, "make": info.make, "model": info.model,
                }))
                .unwrap(),
            )
            .unwrap();
            for asset in assets {
                fs::write(directory.join("analysis").join(asset.filename), asset.bytes).unwrap();
            }
            fs::write(
                directory.join("negative.fits"),
                encode_linear_analysis(
                    &[-4.25, 0.125, 65535.0, f32::NAN],
                    2,
                    2,
                    "adu",
                    "encoding-fixture-v1",
                )
                .unwrap(),
            )
            .unwrap();
        }
    }
}

/// Encode a 16-bit single-plane image straight from a borrowed buffer.
///
/// `pixels` is `width * height * 2` bytes of row-major native-endian `u16` —
/// exactly the layout [`crate::hardware::raw_decode::RawImage::pixels`] hands
/// out, which is the point: a 24 MP decode is ~48 MB, and a signature that took
/// an owning `Vec` would force a copy of it for no reason other than the type.
///
/// The samples are written as FITS `BITPIX = 16` (signed big-endian) with
/// `BZERO = 32768`, which is the standard, exactly reversible encoding of
/// unsigned 16-bit data. No values are altered.
///
/// Writes no equipment keywords: the capture path derives its FITS through
/// [`fits_from_source`], which carries them. This plain form is kept for the
/// plate-solve fixtures and tests.
#[cfg_attr(not(test), allow(dead_code))]
pub fn encode_bayer_u16(
    pixels: &[u8],
    width: u32,
    height: u32,
    meta: &FitsMeta,
    bayer: Option<&str>,
) -> Result<Vec<u8>> {
    encode_bayer_u16_with(pixels, width, height, meta, bayer, &[])
}

/// [`encode_bayer_u16`] with caller-supplied header cards.
fn encode_bayer_u16_with(
    pixels: &[u8],
    width: u32,
    height: u32,
    meta: &FitsMeta,
    bayer: Option<&str>,
    extra: &[FitsCard],
) -> Result<Vec<u8>> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| anyhow!("16-bit plane dimensions overflow"))?;
    if width == 0 || height == 0 || pixels.len() != expected {
        return Err(anyhow!(
            "16-bit buffer must exactly match a nonempty plane: {} != {}",
            pixels.len(),
            expected
        ));
    }

    let naxes = vec![width as i64, height as i64];
    let header = build_header(16, &naxes, meta, bayer, extra);

    let mut out = Vec::with_capacity(header.len() + expected + BLOCK);
    out.extend_from_slice(&header);
    push_u16_native_as_fits(&pixels[..expected], &mut out);
    pad(&mut out, 0);
    Ok(out)
}

/// Append **native-endian** `u16` samples as FITS `BITPIX = 16` (signed,
/// big-endian, offset by `BZERO = 32768`).
///
/// Native, not little — the RAW plane was `memcpy`'d out of LibRaw's in-memory
/// `ushort` array, so its byte order is the host's by construction. Camera
/// stream buffers are a different case: those arrive over the wire as
/// little-endian and `frame_to_pixels` decodes them as such.
fn push_u16_native_as_fits(src: &[u8], out: &mut Vec<u8>) {
    for chunk in src.chunks_exact(2) {
        let v = u16::from_ne_bytes([chunk[0], chunk[1]]);
        let signed = (v as i32 - 32768) as i16;
        out.extend_from_slice(&signed.to_be_bytes());
    }
}

/// A FITS derived from a camera-authored file.
pub struct DerivedFits {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub bit_depth: u32,
    /// What produced it, recorded as provenance on the asset.
    pub decoder: &'static str,
    /// Present for RAW containers: what the decoder chose not to apply, for
    /// the asset's `acquisition_json`. Recorded, never applied.
    pub raw_info: Option<crate::hardware::raw_decode::RawInfo>,
    pub analysis_assets: Vec<crate::storage::capture_store::PendingAsset>,
    pub analysis_pending_reason: Option<String>,
}

/// Derive a FITS from the bytes the camera itself wrote.
///
/// `Ok(None)` means "this build has no decoder for that container" — not an
/// error. The native file has already been preserved by the caller, so the
/// FITS can be derived later by a build that does; refusing the whole capture
/// because a derivation is unavailable would throw away a perfectly good
/// exposure.
///
/// RAW containers decode through LibRaw in the C++ bridge, which yields
/// untouched sensor ADU plus the calibration constants needed to interpret
/// them. A build without LibRaw returns `Ok(None)` here rather than an error,
/// so the exposure is still archived and can be converted later.
///
/// `extra` holds the caller's header cards (equipment H1); they are in the
/// header before any analysis asset hashes the FITS.
pub fn fits_from_source(
    bytes: &[u8],
    extension: &str,
    meta: &FitsMeta,
    extra: &[FitsCard],
) -> Result<Option<DerivedFits>> {
    match extension.to_ascii_lowercase().as_str() {
        // Every container LibRaw handles that a body in this rig might write.
        // DNG is included because some bodies and most adapters emit it, and a
        // DNG that carries a Bayer plane decodes exactly like a CR3; the ones
        // that do not (linear/demosaiced DNG) are rejected inside the bridge
        // and land in the `Err` path below, which is non-fatal.
        "cr2" | "cr3" | "rw2" | "nef" | "arw" | "dng" | "orf" | "pef" | "raf" | "srw" => {
            if !crate::hardware::raw_decode::libraw_available() {
                return Ok(None);
            }
            let image = crate::hardware::raw_decode::decode(bytes)?;
            let info = image.info().clone();

            // The calibration constants belong in the header of *this* file, so
            // clone the caller's metadata rather than making every caller know
            // what a decoder is about to discover.
            let mut meta = meta.clone();
            meta.calibration = Some(SensorCalibration {
                black_level: info.black_level,
                cblack: info.cblack,
                cblack_pattern: info.cblack_pattern,
                white_level: info.white_level,
                // Zero, and not by luck: the bridge evaluates the CFA at the
                // raw coordinates of the crop origin, so `info.bayer` already
                // describes pixel (0, 0) of the plane it returned. On a body
                // with an odd left margin it reports GRBG where the sensor is
                // RGGB — which is the correct answer for this plane, not a bug.
                bayer_x_offset: 0,
                bayer_y_offset: 0,
                decoder_metadata: info.decoder_metadata.clone(),
            });

            let bayer = bayer_pattern(info.bayer);
            // `pixels()` borrows the C++ buffer; nothing is copied on the way
            // into the encoder.
            let out = encode_bayer_u16_with(
                image.pixels(),
                info.width,
                info.height,
                &meta,
                bayer,
                extra,
            )?;
            let analysis = raw_analysis_assets(&info, image.pixels(), &out);
            let (analysis_assets, analysis_pending_reason) = match analysis {
                Ok(assets) => (assets, None),
                Err(error) => (Vec::new(), Some(error.to_string())),
            };

            Ok(Some(DerivedFits {
                bytes: out,
                width: info.width,
                height: info.height,
                bit_depth: 16,
                decoder: "libraw",
                raw_info: Some(info),
                analysis_assets,
                analysis_pending_reason,
            }))
        }
        "jpg" | "jpeg" => {
            let img = image::load_from_memory(bytes)?.to_rgb8();
            let (w, h) = (img.width() as usize, img.height() as usize);
            let data = interleaved_rgb_to_planar(img.as_raw(), w, h);
            let naxes = vec![w as i64, h as i64, 3];
            let header = build_header(8, &naxes, meta, None, extra);
            let mut out = Vec::with_capacity(header.len() + data.len());
            out.extend_from_slice(&header);
            out.extend_from_slice(&data);
            Ok(Some(DerivedFits {
                bytes: out,
                width: w as u32,
                height: h as u32,
                bit_depth: 8,
                decoder: "image-rs/jpeg",
                raw_info: None,
                analysis_assets: Vec::new(),
                analysis_pending_reason: Some(
                    "Camera JPEG is a processed image domain, not untouched linear sensor samples"
                        .into(),
                ),
            }))
        }
        _ => Ok(None),
    }
}

fn raw_analysis_assets(
    info: &crate::hardware::raw_decode::RawInfo,
    pixels: &[u8],
    sensor_fits: &[u8],
) -> Result<Vec<crate::storage::capture_store::PendingAsset>> {
    use crate::processing::debayer::*;
    // LibRaw reports phase at the crop origin. The affine mapping names its
    // raw array, with unapplied orientation/aspect recorded as decoder evidence.
    let pattern = match info.bayer {
        OlsBayerType::Rg => CfaPattern::Rggb,
        OlsBayerType::Bg => CfaPattern::Bggr,
        OlsBayerType::Gr => CfaPattern::Grbg,
        OlsBayerType::Gb => CfaPattern::Gbrg,
        OlsBayerType::Na => {
            return Err(anyhow!(
                "RAW CFA is unknown; cannot infer monochrome from a missing pattern"
            ))
        }
    };
    let metadata = info
        .decoder_metadata
        .as_ref()
        .ok_or_else(|| anyhow!("Missing untouched decoder metadata"))?;
    if (metadata.pixel_aspect - 1.0).abs() > 1e-12 {
        return Err(anyhow!(
            "Non-square RAW pixels require qualified geometry/measurement handling"
        ));
    }
    let descriptor = SensorDescriptor {
        schema_version: 1,
        width: info.width as usize,
        height: info.height as usize,
        row_stride_bytes: info.width as usize * 2,
        sample_type: SampleType::U16,
        byte_order: if cfg!(target_endian = "little") {
            ByteOrder::Little
        } else {
            ByteOrder::Big
        },
        color: SensorColor::Bayer {
            pattern,
            origin_x: 0,
            origin_y: 0,
        },
        domain: SampleDomain::NativeAdu,
        geometry: PixelGeometry {
            analysis_to_sensor: [
                [1.0, 0.0, metadata.crop_x as f64],
                [0.0, 1.0, metadata.crop_y as f64],
            ],
            coordinate_system: "libraw_raw_array".into(),
        },
        saturation: (info.white_level > 0).then(|| NativeThreshold {
            adu: info.white_level as f32,
            source: format!("{}; nonlinearity_not_measured", metadata.white_level_source),
        }),
        calibration_ids: Vec::new(),
    };
    let analysis = analyze(&SensorFrame {
        descriptor: &descriptor,
        data: pixels,
        masks: NativeMasks::default(),
    })?;
    let mut assets = crate::storage::analysis_assets::analysis_assets(
        &analysis,
        "image.fits",
        sensor_fits,
        pixels,
    )?;
    for asset in &mut assets {
        asset.provenance["decoder_metadata"] = serde_json::to_value(metadata)?;
        if asset.role == crate::storage::capture_store::AssetRole::AnalysisDescriptor {
            let mut descriptor: serde_json::Value = serde_json::from_slice(&asset.bytes)?;
            descriptor["sensor_decode"] = serde_json::json!({
                "metadata": metadata, "black_level": info.black_level,
                "cblack": info.cblack, "cblack_pattern": info.cblack_pattern,
                "white_level": info.white_level, "cam_mul": info.cam_mul,
                "make": info.make, "model": info.model,
            });
            asset.bytes = serde_json::to_vec_pretty(&descriptor)?;
        }
    }
    Ok(assets)
}

/// Write `frame` to `<frames_dir>/<id>.fits` and return the path.
///
/// Retained for callers outside the capture path (autofocus preservation and
/// the sim/bench harnesses). New capture code should go through
/// `encode_capture_with` + the capture store so the file is catalogued.
pub fn write_capture(frame: &FrameData, meta: &FitsMeta, id: &str) -> Result<String> {
    let dir = frames_dir();
    fs::create_dir_all(&dir)?;
    let path = PathBuf::from(&dir).join(format!("{id}.fits"));
    let bytes = encode_capture_with(frame, meta, &[])?;
    let mut file = fs::File::create(&path)?;
    file.write_all(&bytes)?;
    Ok(path.to_string_lossy().into_owned())
}

/// Convert a frame into (BITPIX, axis lengths, big-endian padded data unit).
fn frame_to_pixels(frame: &FrameData) -> Result<(i32, Vec<i64>, Vec<u8>)> {
    let w = frame.width as usize;
    let h = frame.height as usize;

    match frame.format {
        OlsStreamType::Mono8 | OlsStreamType::Raw8 => {
            let expected = w * h;
            if frame.data.len() < expected {
                return Err(anyhow!(
                    "8-bit buffer too small: {} < {}",
                    frame.data.len(),
                    expected
                ));
            }
            let mut out = frame.data[..expected].to_vec();
            pad(&mut out, 0);
            Ok((8, vec![w as i64, h as i64], out))
        }
        OlsStreamType::Mono16 | OlsStreamType::Raw16 => {
            let expected = w * h * 2;
            if frame.data.len() < expected {
                return Err(anyhow!(
                    "16-bit buffer too small: {} < {}",
                    frame.data.len(),
                    expected
                ));
            }
            // Camera samples are little-endian unsigned 16-bit. FITS BITPIX=16 is
            // signed big-endian, so we store (value - 32768) and set BZERO=32768.
            let mut out = Vec::with_capacity(expected);
            for chunk in frame.data[..expected].chunks_exact(2) {
                let v = u16::from_le_bytes([chunk[0], chunk[1]]);
                let signed = (v as i32 - 32768) as i16;
                out.extend_from_slice(&signed.to_be_bytes());
            }
            pad(&mut out, 0);
            Ok((16, vec![w as i64, h as i64], out))
        }
        OlsStreamType::Rgb24 => {
            let expected = w * h * 3;
            if frame.data.len() < expected {
                return Err(anyhow!(
                    "rgb buffer too small: {} < {}",
                    frame.data.len(),
                    expected
                ));
            }
            Ok((
                8,
                vec![w as i64, h as i64, 3],
                interleaved_rgb_to_planar(&frame.data, w, h),
            ))
        }
        OlsStreamType::Mjpeg => {
            // Decode the JPEG preview to RGB8, then store planar.
            let img = image::load_from_memory(&frame.data)?.to_rgb8();
            let (iw, ih) = (img.width() as usize, img.height() as usize);
            Ok((
                8,
                vec![iw as i64, ih as i64, 3],
                interleaved_rgb_to_planar(img.as_raw(), iw, ih),
            ))
        }
        other => Err(anyhow!("unsupported capture format {:?}", other)),
    }
}

/// Reorder interleaved RGBRGB... into planar RRR...GGG...BBB and pad to a block.
fn interleaved_rgb_to_planar(data: &[u8], w: usize, h: usize) -> Vec<u8> {
    let plane = w * h;
    let mut out = vec![0u8; plane * 3];
    for i in 0..plane {
        out[i] = data[i * 3];
        out[plane + i] = data[i * 3 + 1];
        out[2 * plane + i] = data[i * 3 + 2];
    }
    pad(&mut out, 0);
    out
}

/// Pad `buf` with `fill` up to the next 2880-byte boundary.
fn pad(buf: &mut Vec<u8>, fill: u8) {
    let rem = buf.len() % BLOCK;
    if rem != 0 {
        buf.resize(buf.len() + (BLOCK - rem), fill);
    }
}

fn bayer_pattern(b: OlsBayerType) -> Option<&'static str> {
    match b {
        OlsBayerType::Na => None,
        OlsBayerType::Rg => Some("RGGB"),
        OlsBayerType::Bg => Some("BGGR"),
        OlsBayerType::Gr => Some("GRBG"),
        OlsBayerType::Gb => Some("GBRG"),
    }
}

fn build_header(
    bitpix: i32,
    naxes: &[i64],
    meta: &FitsMeta,
    bayer: Option<&str>,
    extra: &[FitsCard],
) -> Vec<u8> {
    let mut cards: Vec<String> = Vec::new();
    cards.push(card_logical("SIMPLE", true, "conforms to FITS standard"));
    cards.push(card_int("BITPIX", bitpix as i64, "bits per pixel"));
    cards.push(card_int("NAXIS", naxes.len() as i64, "number of axes"));
    for (i, n) in naxes.iter().enumerate() {
        cards.push(card_int(&format!("NAXIS{}", i + 1), *n, ""));
    }
    if bitpix == 16 {
        cards.push(card_float("BZERO", 32768.0, "unsigned 16-bit offset"));
        cards.push(card_float("BSCALE", 1.0, ""));
    }
    if meta.exposure_s.is_finite() && meta.exposure_s > 0.0 {
        cards.push(card_float("EXPTIME", meta.exposure_s, "[s] exposure time"));
    }
    cards.push(card_int("GAIN", meta.gain as i64, "sensor gain"));
    if let Some(iso) = meta.iso {
        // ISOSPEED is the conventional card for DSLR captures; GAIN alone is
        // meaningless on a body whose gain control is an ISO enumeration.
        cards.push(card_float("ISOSPEED", iso, "ISO speed rating"));
    }
    if let Some(shutter) = meta.shutter_label.as_deref() {
        cards.push(card_string("SHUTTER", shutter, "camera shutter setting"));
    }
    if let Some(src) = meta.source_file.as_deref() {
        cards.push(card_string("RAWFILE", src, "preserved camera original"));
    }
    if meta.temperature_c.is_finite() {
        cards.push(card_float(
            "CCD-TEMP",
            meta.temperature_c,
            "[degC] sensor temperature",
        ));
    }
    cards.push(card_int(
        "XBINNING",
        meta.binning as i64,
        "binning factor, X axis",
    ));
    cards.push(card_int(
        "YBINNING",
        meta.binning as i64,
        "binning factor, Y axis",
    ));
    cards.push(card_string("INSTRUME", &meta.instrument, "camera"));
    if !meta.date_obs.is_empty() {
        cards.push(card_string(
            "DATE-OBS",
            &meta.date_obs,
            "UTC start of exposure",
        ));
    }
    if let Some(b) = bayer {
        cards.push(card_string("BAYERPAT", b, "colour filter array pattern"));
    }
    if let Some(cal) = meta.calibration.as_ref() {
        // Raw ADU. Nothing below has been applied to the pixels — these cards
        // are what a calibration step needs in order to apply it correctly.
        cards.push(card_string(
            "BUNIT",
            "ADU",
            "raw sensor units, uncalibrated",
        ));
        cards.push(card_int(
            "BLKLEVEL",
            cal.black_level as i64,
            "[ADU] black level, NOT subtracted",
        ));
        if cal.cblack.iter().any(|v| *v != 0) {
            // Per-CFA-channel pedestals, in LibRaw's channel order. Non-standard
            // cards, but dropping them would leave BLKLEVEL wrong by a few ADU
            // per channel — the systematic that shows up as a colour cast in a
            // stacked background.
            for (i, v) in cal.cblack.iter().enumerate() {
                cards.push(card_int(
                    &format!("CBLACK{i}"),
                    *v as i64,
                    "[ADU] per-CFA-channel black offset",
                ));
            }
        }
        if cal.white_level > 0 {
            cards.push(card_int(
                "SATURATE",
                cal.white_level as i64,
                "[ADU] declared white code; not measured linearity",
            ));
        }
        if let Some(md) = &cal.decoder_metadata {
            cards.push(card_string("RAWVER", &md.decoder_version, "LibRaw version"));
            cards.push(card_string(
                "RAWPROC",
                &md.processing_version,
                "unpack contract; no application calibration",
            ));
            cards.push(card_int(
                "RAWOPT",
                md.unpack_options as i64,
                "LibRaw unpack option flags",
            ));
            if let Some(specials) = md.unpack_specials {
                cards.push(card_int(
                    "RAWSPEC",
                    specials as i64,
                    "LibRaw unpack specials",
                ));
            }
            cards.push(card_int(
                "RAWSHOT",
                md.shot_select as i64,
                "LibRaw image selection",
            ));
            for (key, value) in [
                ("RAWW", md.raw_width),
                ("RAWH", md.raw_height),
                ("VISW", md.visible_width),
                ("VISH", md.visible_height),
                ("RAWX0", md.crop_x),
                ("RAWY0", md.crop_y),
            ] {
                cards.push(card_int(
                    key,
                    value as i64,
                    "unrotated LibRaw raw-array geometry",
                ));
            }
            cards.push(card_int(
                "RAWPITCH",
                md.source_pitch_bytes as i64,
                "source row stride in bytes",
            ));
            cards.push(card_int(
                "RAWFLIP",
                md.orientation as i64,
                "LibRaw orientation; NOT applied",
            ));
            cards.push(card_float(
                "PIXASPEC",
                md.pixel_aspect,
                "LibRaw pixel width/height ratio",
            ));
            cards.push(card_int(
                "BLKROWS",
                cal.cblack_pattern[0] as i64,
                "spatial black table rows",
            ));
            cards.push(card_int(
                "BLKCOLS",
                cal.cblack_pattern[1] as i64,
                "spatial black table columns",
            ));
            cards.push(card_int(
                "BLKTABN",
                md.black_table.len() as i64,
                "spatial black entries; separate from CBLACK",
            ));
            for (i, value) in md.black_table.iter().enumerate() {
                cards.push(card_int(
                    &format!("BLKC{i:04}"),
                    *value as i64,
                    "[ADU] LibRaw spatial black table",
                ));
            }
            for (i, value) in md.linear_max.iter().enumerate() {
                if *value > 0 {
                    cards.push(card_int(
                        &format!("LINMAX{i}"),
                        *value as i64,
                        "[ADU] metadata linear maximum; unmeasured",
                    ));
                }
            }
            cards.push(card_string(
                "NONLIN",
                "UNMEAS",
                "nonlinearity has not been measured",
            ));
            cards.push(card_string(
                "WHITESRC",
                "LIBRAW",
                "declared unpack maximum, not container capacity",
            ));
        }
        if bayer.is_some() {
            cards.push(card_int(
                "XBAYROFF",
                cal.bayer_x_offset as i64,
                "Bayer pattern X offset",
            ));
            cards.push(card_int(
                "YBAYROFF",
                cal.bayer_y_offset as i64,
                "Bayer pattern Y offset",
            ));
        }
    }
    // Caller cards come after the acquisition cards and never replace one:
    // a keyword the header already carries (or a repeated caller keyword)
    // is skipped, so a header can never hold two values for one keyword.
    let mut written: std::collections::BTreeSet<String> = cards
        .iter()
        .map(|card| card.get(..8).unwrap_or(card).trim_end().to_string())
        .collect();
    written.insert("CREATOR".into());
    written.insert("END".into());
    for card in extra {
        if written.contains(&card.key) {
            continue;
        }
        if let Some(rendered) = card.render() {
            written.insert(card.key.clone());
            cards.push(rendered);
        }
    }
    cards.push(card_string("CREATOR", "Achernar daemon", ""));
    cards.push("END".to_string());

    let mut bytes = Vec::with_capacity(cards.len() * 80);
    for card in cards {
        let mut s = card.into_bytes();
        s.truncate(80);
        while s.len() < 80 {
            s.push(b' ');
        }
        bytes.extend_from_slice(&s);
    }
    // Pad the header with spaces to a whole 2880-byte block.
    let rem = bytes.len() % BLOCK;
    if rem != 0 {
        bytes.resize(bytes.len() + (BLOCK - rem), b' ');
    }
    bytes
}

// ── Reading a header back ──────────────────────────────────────────────────

/// Shape of a FITS primary image, read from its header alone.
///
/// The plate solver needs the true image size — not the size a capture request
/// asked for, and not a database column that older rows leave null — to turn a
/// pixel scale into the field height ASTAP wants and to find the centre pixel.
/// It also needs to know whether the plane is an undebayered mosaic: a solver
/// that mistakes the Bayer grid for structure detects thousands of false stars.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct FitsImageGeometry {
    pub width: u32,
    pub height: u32,
    /// `NAXIS3` for a colour cube, `1` for a single plane.
    pub planes: u32,
    pub bitpix: i32,
    /// `BAYERPAT`, present when the single plane is a colour filter mosaic.
    pub bayer_pattern: Option<String>,
    /// `XBINNING`, when the writer recorded it.
    pub binning: Option<u32>,
}

/// Read [`FitsImageGeometry`] from the primary header of `bytes`.
///
/// Only the header is parsed; the data unit is never touched. Anything that is
/// not a two- or three-axis primary image is an error rather than a guess.
pub fn image_geometry(bytes: &[u8]) -> Result<FitsImageGeometry> {
    let mut simple = false;
    let mut bitpix = None;
    let mut naxis = None;
    let mut axes: [Option<i64>; 3] = [None; 3];
    let mut bayer_pattern = None;
    let mut binning = None;
    let mut ended = false;

    for (index, card) in bytes.chunks_exact(80).enumerate() {
        let key = std::str::from_utf8(&card[..8])
            .map_err(|_| anyhow!("FITS card {index} has a non-ASCII keyword"))?
            .trim_end();
        if index == 0 && key != "SIMPLE" {
            return Err(anyhow!("Not a FITS file: the first card is not SIMPLE"));
        }
        if key == "END" {
            ended = true;
            break;
        }
        let Some(value) = card_value(card) else {
            continue;
        };
        match key {
            "SIMPLE" => simple = value == "T",
            "BITPIX" => bitpix = value.parse::<i32>().ok(),
            "NAXIS" => naxis = value.parse::<i64>().ok(),
            "NAXIS1" => axes[0] = value.parse().ok(),
            "NAXIS2" => axes[1] = value.parse().ok(),
            "NAXIS3" => axes[2] = value.parse().ok(),
            "BAYERPAT" => bayer_pattern = Some(value).filter(|v| !v.is_empty()),
            // Some writers record binning as a float ("2.0").
            "XBINNING" => {
                binning = value
                    .parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite() && *v >= 1.0 && v.fract() == 0.0)
                    .map(|v| v as u32)
            }
            _ => {}
        }
    }

    if !ended {
        return Err(anyhow!("FITS header has no END card"));
    }
    if !simple {
        return Err(anyhow!("SIMPLE is not T: not a standard FITS file"));
    }
    let bitpix = bitpix
        .filter(|b| [8, 16, 32, 64, -32, -64].contains(b))
        .ok_or_else(|| anyhow!("FITS BITPIX is missing or invalid"))?;
    let axis = |n: Option<i64>, name: &str| -> Result<u32> {
        n.filter(|v| *v > 0)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| anyhow!("FITS {name} is missing or not a positive length"))
    };
    let planes = match naxis {
        Some(2) => 1,
        Some(3) => axis(axes[2], "NAXIS3")?,
        other => {
            return Err(anyhow!(
                "Expected a 2- or 3-axis primary image, found NAXIS = {other:?}"
            ))
        }
    };
    Ok(FitsImageGeometry {
        width: axis(axes[0], "NAXIS1")?,
        height: axis(axes[1], "NAXIS2")?,
        planes,
        bitpix,
        bayer_pattern,
        binning,
    })
}

/// The value field of a `KEYWORD = value / comment` card, unquoted. `None` for
/// cards with no value indicator (COMMENT, HISTORY, blank).
fn card_value(card: &[u8]) -> Option<String> {
    if card.get(8..10) != Some(b"= ".as_slice()) {
        return None;
    }
    let field = std::str::from_utf8(&card[10..]).ok()?.trim_start();
    if let Some(rest) = field.strip_prefix('\'') {
        // A quote inside a FITS string is written as two quotes.
        let mut out = String::new();
        let mut chars = rest.chars().peekable();
        while let Some(c) = chars.next() {
            if c != '\'' {
                out.push(c);
            } else if chars.peek() == Some(&'\'') {
                out.push('\'');
                chars.next();
            } else {
                break;
            }
        }
        return Some(out.trim_end().to_string());
    }
    Some(field.split('/').next()?.trim().to_string())
}

#[cfg(test)]
mod geometry_tests {
    use super::*;

    fn meta() -> FitsMeta {
        FitsMeta {
            exposure_s: 4.0,
            gain: 100,
            temperature_c: f64::NAN,
            binning: 2,
            instrument: "fixture".into(),
            date_obs: String::new(),
            iso: None,
            shutter_label: None,
            source_file: None,
            calibration: None,
        }
    }

    #[test]
    fn reads_shape_mosaic_and_binning_from_a_capture_header() {
        let pixels = vec![0u8; 6 * 4 * 2];
        let bytes = encode_bayer_u16(&pixels, 6, 4, &meta(), Some("RGGB")).unwrap();
        let geometry = image_geometry(&bytes).unwrap();
        assert_eq!(
            geometry,
            FitsImageGeometry {
                width: 6,
                height: 4,
                planes: 1,
                bitpix: 16,
                bayer_pattern: Some("RGGB".into()),
                binning: Some(2),
            }
        );
    }

    #[test]
    fn a_plane_without_bayerpat_is_not_called_a_mosaic() {
        let bytes = encode_linear_analysis(&[0.0; 6], 3, 2, "ADU", "linear_v1").unwrap();
        let geometry = image_geometry(&bytes).unwrap();
        assert_eq!((geometry.width, geometry.height), (3, 2));
        assert_eq!(geometry.bitpix, -32);
        assert_eq!(geometry.bayer_pattern, None);
        assert_eq!(geometry.binning, None);
    }

    #[test]
    fn refuses_bytes_that_are_not_a_fits_image() {
        assert!(image_geometry(b"not a fits file at all").is_err());
        let mut truncated = encode_linear_analysis(&[0.0; 4], 2, 2, "ADU", "v").unwrap();
        truncated.truncate(80 * 3);
        assert!(image_geometry(&truncated).is_err(), "no END card");
    }

    #[test]
    fn string_values_unescape_doubled_quotes_and_ignore_comments() {
        let mut card = format!("{:<8}= {:<20} / note", "OBJECT", "'O''Brien''s'");
        card.truncate(80);
        let mut bytes = card.into_bytes();
        bytes.resize(80, b' ');
        assert_eq!(card_value(&bytes).as_deref(), Some("O'Brien's"));
    }
}

// ── FITS card formatting (80 columns; "= " value indicator in cols 9-10) ──

fn card_kv(key: &str, value: String, comment: &str) -> String {
    let mut card = format!("{:<8}= {:>20}", key, value);
    if !comment.is_empty() {
        card.push_str(" / ");
        card.push_str(comment);
    }
    card
}

fn card_logical(key: &str, v: bool, c: &str) -> String {
    card_kv(key, if v { "T" } else { "F" }.to_string(), c)
}

fn card_int(key: &str, v: i64, c: &str) -> String {
    card_kv(key, v.to_string(), c)
}

fn card_float(key: &str, v: f64, c: &str) -> String {
    card_kv(key, format!("{:.6}", v), c)
}

/// Longest string value that fits one card: 80 columns, minus the 10-column
/// keyword and value indicator, minus the two quotes.
const MAX_STRING_VALUE: usize = 68;

fn card_string(key: &str, v: &str, c: &str) -> String {
    // String values are single-quoted and left-justified (min 8 chars inside).
    // A quote inside the value is written as two quotes; a value too long for
    // one card is cut on a character boundary so the closing quote is never
    // lost to the 80-column truncation (which would corrupt the header).
    let mut escaped = String::new();
    for ch in v
        .chars()
        .filter(|ch| ch.is_ascii() && !ch.is_ascii_control())
    {
        let width = if ch == '\'' { 2 } else { 1 };
        if escaped.len() + width > MAX_STRING_VALUE {
            break;
        }
        if ch == '\'' {
            escaped.push('\'');
        }
        escaped.push(ch);
    }
    let quoted = format!("'{:<8}'", escaped);
    let mut card = format!("{:<8}= {:<20}", key, quoted);
    if !c.is_empty() && card.len() + 3 < 80 {
        card.push_str(" / ");
        card.push_str(c);
    }
    card
}

#[cfg(test)]
mod card_tests {
    use super::*;

    fn meta() -> FitsMeta {
        FitsMeta {
            exposure_s: 30.0,
            gain: 120,
            temperature_c: f64::NAN,
            binning: 1,
            instrument: "fixture".into(),
            date_obs: "2026-10-26T04:00:00Z".into(),
            iso: None,
            shutter_label: None,
            source_file: None,
            calibration: None,
        }
    }

    /// Every `KEY = value` card of a primary header, in order.
    fn header_cards(bytes: &[u8]) -> Vec<(String, Option<String>)> {
        let mut out = Vec::new();
        for card in bytes.as_chunks::<80>().0 {
            let key = std::str::from_utf8(&card[..8])
                .unwrap()
                .trim_end()
                .to_string();
            if key == "END" {
                break;
            }
            out.push((key, card_value(card)));
        }
        out
    }

    fn value(cards: &[(String, Option<String>)], key: &str) -> Option<String> {
        cards
            .iter()
            .find(|(k, _)| k == key)
            .and_then(|(_, v)| v.clone())
    }

    #[test]
    fn caller_cards_are_written_before_end_and_never_duplicate_a_keyword() {
        let extra = vec![
            FitsCard::string("OBJCTRA", "05 35 17.30", "[hms J2000] telescope RA"),
            FitsCard::float("RA", 83.822083, "[deg J2000] telescope RA"),
            FitsCard::int("FOCPOS", 12_345, "[step] focuser position"),
            // Already written from FitsMeta: the caller cannot replace it.
            FitsCard::float("EXPTIME", 999.0, "must not appear twice"),
            // A repeated caller keyword keeps the first value.
            FitsCard::int("FOCPOS", 1, "duplicate"),
            // Never written: a non-finite value and an invalid keyword.
            FitsCard::float("AIRMASS", f64::NAN, "unknown"),
            FitsCard::float("toolongkey", 1.0, "invalid"),
        ];
        let bytes = encode_bayer_u16_with(&[0u8; 4 * 2 * 2], 4, 2, &meta(), None, &extra).unwrap();
        let cards = header_cards(&bytes);
        let keys: Vec<&str> = cards.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys.iter().filter(|k| **k == "EXPTIME").count(), 1);
        assert_eq!(value(&cards, "EXPTIME").as_deref(), Some("30.000000"));
        assert_eq!(keys.iter().filter(|k| **k == "FOCPOS").count(), 1);
        assert_eq!(value(&cards, "FOCPOS").as_deref(), Some("12345"));
        assert_eq!(value(&cards, "OBJCTRA").as_deref(), Some("05 35 17.30"));
        assert_eq!(value(&cards, "RA").as_deref(), Some("83.822083"));
        assert!(!keys.contains(&"AIRMASS"), "a missing value is omitted");
        assert!(!keys.iter().any(|k| k.eq_ignore_ascii_case("toolongk")));
        // CREATOR stays the last card before END, and the header still parses.
        assert_eq!(keys.last(), Some(&"CREATOR"));
        let geometry = image_geometry(&bytes).unwrap();
        assert_eq!((geometry.width, geometry.height), (4, 2));
        assert_eq!(bytes.len() % BLOCK, 0);
    }

    #[test]
    fn string_cards_escape_quotes_and_keep_the_closing_quote() {
        let card = card_string("FILTER", "O'III", "filter");
        let mut bytes = card.into_bytes();
        bytes.resize(80, b' ');
        assert_eq!(card_value(&bytes).as_deref(), Some("O'III"));

        let long = "x".repeat(120);
        let card = card_string("INSTRUME", &long, "a comment that cannot fit");
        assert!(card.len() <= 80, "{}", card.len());
        let mut bytes = card.into_bytes();
        bytes.resize(80, b' ');
        assert_eq!(
            card_value(&bytes).map(|v| v.len()),
            Some(MAX_STRING_VALUE),
            "truncated inside the quotes, never past them"
        );
        // Non-ASCII and control characters never reach a FITS header.
        let card = card_string("FILTER", "H\u{3b1}\u{7}", "");
        let mut bytes = card.into_bytes();
        bytes.resize(80, b' ');
        assert_eq!(card_value(&bytes).as_deref(), Some("H"));
    }

    #[test]
    fn sexagesimal_formatting_rounds_with_carry() {
        assert_eq!(sexagesimal_ra(5.588_139).as_deref(), Some("05 35 17.30"));
        assert_eq!(sexagesimal_ra(23.999_999_9).as_deref(), Some("00 00 00.00"));
        assert_eq!(sexagesimal_ra(-1.0).as_deref(), Some("23 00 00.00"));
        assert_eq!(sexagesimal_ra(f64::NAN), None);
        assert_eq!(sexagesimal_dec(-5.390_278).as_deref(), Some("-05 23 25.0"));
        assert_eq!(
            sexagesimal_dec(41.999_999_9).as_deref(),
            Some("+42 00 00.0")
        );
        assert_eq!(sexagesimal_dec(91.0), None);
    }

    #[test]
    fn the_plain_encoders_write_no_equipment_keywords() {
        let bytes = encode_bayer_u16(&[0u8; 2 * 2 * 2], 2, 2, &meta(), None).unwrap();
        let cards = header_cards(&bytes);
        for key in ["RA", "DEC", "OBJCTRA", "FOCPOS", "FILTER", "SITELAT"] {
            assert!(value(&cards, key).is_none(), "{key}");
        }
    }
}

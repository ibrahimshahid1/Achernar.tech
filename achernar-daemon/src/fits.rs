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
/// pixels — the numbers are what the ADC produced, and everything needed to
/// calibrate them is here. Emitting `BLKLEVEL` on a frame that has already had
/// black subtracted would be worse than emitting nothing, because a downstream
/// pipeline would subtract it twice.
#[derive(Debug, Clone, Copy)]
pub struct SensorCalibration {
    /// Black pedestal in ADU, **not** subtracted.
    pub black_level: i32,
    /// Per-CFA-channel black offsets, added to `black_level`. `[0; 4]` on
    /// bodies with a single pedestal.
    pub cblack: [i32; 4],
    /// Saturation point in ADU. On a 14-bit body this is 16383, not 65535 —
    /// which is the entire reason the card exists. A saturation test against
    /// the container depth can never fire.
    pub white_level: i32,
    /// Offset of the Bayer pattern origin from the image origin. Zero when the
    /// decoder's crop preserved the phase. Written explicitly rather than
    /// omitted so a reader can tell "the offset is zero" from "nobody
    /// considered the offset".
    pub bayer_x_offset: i32,
    pub bayer_y_offset: i32,
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

/// Directory captures are written to. `./frames`, overridable via
/// `ACHERNAR_FRAMES_DIR`.
pub fn frames_dir() -> String {
    std::env::var("ACHERNAR_FRAMES_DIR").unwrap_or_else(|_| "frames".to_string())
}

/// Encode `frame` as a complete FITS file in memory.
///
/// Returns bytes rather than writing them, because a capture is committed by
/// [`crate::storage::capture_store`], which stages every asset into one
/// directory, hashes it, and renames the bundle into place atomically. A
/// function that writes its own file behind the store's back produces exactly
/// the orphan the staging protocol exists to prevent.
pub fn encode_capture(frame: &FrameData, meta: &FitsMeta) -> Result<Vec<u8>> {
    let (bitpix, naxes, data) = frame_to_pixels(frame)?;
    let bayer = bayer_pattern(frame.bayer);
    let header = build_header(bitpix, &naxes, meta, bayer);

    let mut out = Vec::with_capacity(header.len() + data.len());
    out.extend_from_slice(&header);
    out.extend_from_slice(&data);
    Ok(out)
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
pub fn encode_bayer_u16(
    pixels: &[u8],
    width: u32,
    height: u32,
    meta: &FitsMeta,
    bayer: Option<&str>,
) -> Result<Vec<u8>> {
    let expected = width as usize * height as usize * 2;
    if pixels.len() < expected {
        return Err(anyhow!(
            "16-bit buffer too small: {} < {}",
            pixels.len(),
            expected
        ));
    }

    let naxes = vec![width as i64, height as i64];
    let header = build_header(16, &naxes, meta, bayer);

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
pub fn fits_from_source(
    bytes: &[u8],
    extension: &str,
    meta: &FitsMeta,
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
                white_level: info.white_level,
                // Zero, and not by luck: the bridge evaluates the CFA at the
                // raw coordinates of the crop origin, so `info.bayer` already
                // describes pixel (0, 0) of the plane it returned. On a body
                // with an odd left margin it reports GRBG where the sensor is
                // RGGB — which is the correct answer for this plane, not a bug.
                bayer_x_offset: 0,
                bayer_y_offset: 0,
            });

            let bayer = bayer_pattern(info.bayer);
            // `pixels()` borrows the C++ buffer; nothing is copied on the way
            // into the encoder.
            let out = encode_bayer_u16(image.pixels(), info.width, info.height, &meta, bayer)?;

            Ok(Some(DerivedFits {
                bytes: out,
                width: info.width,
                height: info.height,
                bit_depth: 16,
                decoder: "libraw",
                raw_info: Some(info),
            }))
        }
        "jpg" | "jpeg" => {
            let img = image::load_from_memory(bytes)?.to_rgb8();
            let (w, h) = (img.width() as usize, img.height() as usize);
            let data = interleaved_rgb_to_planar(img.as_raw(), w, h);
            let naxes = vec![w as i64, h as i64, 3];
            let header = build_header(8, &naxes, meta, None);
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
            }))
        }
        _ => Ok(None),
    }
}

/// Write `frame` to `<frames_dir>/<id>.fits` and return the path.
///
/// Retained for callers outside the capture path (autofocus preservation and
/// the sim/bench harnesses). New capture code should go through
/// `encode_capture` + the capture store so the file is catalogued.
pub fn write_capture(frame: &FrameData, meta: &FitsMeta, id: &str) -> Result<String> {
    let dir = frames_dir();
    fs::create_dir_all(&dir)?;
    let path = PathBuf::from(&dir).join(format!("{id}.fits"));
    let bytes = encode_capture(frame, meta)?;
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

fn build_header(bitpix: i32, naxes: &[i64], meta: &FitsMeta, bayer: Option<&str>) -> Vec<u8> {
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
    cards.push(card_float("EXPTIME", meta.exposure_s, "[s] exposure time"));
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
    cards.push(card_string(
        "DATE-OBS",
        &meta.date_obs,
        "UTC start of exposure",
    ));
    if let Some(b) = bayer {
        cards.push(card_string("BAYERPAT", b, "colour filter array pattern"));
    }
    if let Some(cal) = meta.calibration.as_ref() {
        // Raw ADU. Nothing below has been applied to the pixels — these cards
        // are what a calibration step needs in order to apply it correctly.
        cards.push(card_string("BUNIT", "ADU", "raw sensor units, uncalibrated"));
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
        cards.push(card_int(
            "SATURATE",
            cal.white_level as i64,
            "[ADU] saturation level",
        ));
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

fn card_string(key: &str, v: &str, c: &str) -> String {
    // String values are single-quoted and left-justified (min 8 chars inside).
    let quoted = format!("'{:<8}'", v);
    let mut card = format!("{:<8}= {:<20}", key, quoted);
    if !c.is_empty() {
        card.push_str(" / ");
        card.push_str(c);
    }
    card
}

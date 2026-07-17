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

/// Acquisition metadata recorded in the FITS header.
pub struct FitsMeta {
    pub exposure_s: f64,
    pub gain: u32,
    pub temperature_c: f64,
    pub binning: u32,
    pub instrument: String,
    /// UTC start-of-exposure, ISO-8601 (FITS `DATE-OBS`).
    pub date_obs: String,
}

/// Write `frame` to `<frames_dir>/<id>.fits` and return the absolute-ish path.
/// `frames_dir` defaults to `./frames`, overridable via `ACHERNAR_FRAMES_DIR`.
pub fn write_capture(frame: &FrameData, meta: &FitsMeta, id: &str) -> Result<String> {
    let dir = std::env::var("ACHERNAR_FRAMES_DIR").unwrap_or_else(|_| "frames".to_string());
    fs::create_dir_all(&dir)?;
    let path = PathBuf::from(&dir).join(format!("{id}.fits"));

    let (bitpix, naxes, data) = frame_to_pixels(frame)?;
    let bayer = bayer_pattern(frame.bayer);
    let header = build_header(bitpix, &naxes, meta, bayer);

    let mut file = fs::File::create(&path)?;
    file.write_all(&header)?;
    file.write_all(&data)?;
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

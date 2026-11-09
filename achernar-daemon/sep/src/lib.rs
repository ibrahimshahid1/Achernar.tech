//! Safe Rust API over the vendored, pinned SEP 1.4.1 C library
//! (`vendor/sep-1.4.1`, see `vendor/sep-1.4.1/SOURCE.md`).
//!
//! Scope: spatial background estimation (`sep_background` and its spline
//! evaluators) and source extraction (`sep_extract`). Aperture photometry is
//! compiled (it is part of the library) but not exposed.
//!
//! Every `unsafe` block of the Achernar SEP integration lives in this crate.
//! Callers get owned Rust values, typed errors, and validated inputs:
//!
//! * Geometry is checked before any call: non-zero dimensions, each at most
//!   [`MAX_DIMENSION`], at most [`MAX_PIXELS`] pixels, buffer lengths equal to
//!   `width * height`. SEP itself validates none of this (`background.c`
//!   divides by `bw`, takes `% w`, and reads `bh` rows past a zero-height
//!   image).
//! * Unmasked pixels must be finite and below SEP's `BIG` sentinel; NaN is
//!   rejected rather than passed to SEP (`backhisto` casts every pixel to an
//!   `int64_t` bin before it tests the mask, which is undefined for NaN).
//! * Masks are `u8`, any non-zero value excludes the pixel; they are passed as
//!   `SEP_TBYTE` with `maskthresh = 0.0`, and SEP tests `mask > maskthresh`
//!   (`extract.c:apply_mask_line`) / `mask <= maskthresh` for a valid pixel
//!   (`background.c:backstat`, `backhisto`).
//!
//! # Thread safety
//!
//! SEP 1.4.1 keeps process-global configuration: `extract.c`
//! `static _Atomic size_t extract_pixstack` and `deblend.c`
//! `static _Atomic int nsonmax`. `nsonmax` is read when `allocdeblend` sizes
//! the deblending buffers and again inside `deblend` for indexing, so changing
//! it during another thread's extraction can overrun those buffers. The error
//! detail (`util.c` `_errdetail_buffer`) and the extraction scratch state
//! (`plistsize`, `plistoff_*`, `randseed`) are `_Thread_local`; the detail is
//! not cleared at the start of a call, only by `sep_get_errdetail`. On MSVC
//! `sepcore.h` defines `_Atomic` away and maps `rand_r` to `rand()`.
//!
//! This crate therefore runs every FFI call sequence — configuring the pixel
//! stack and sub-object limit, the call itself, and reading the error message
//! and detail on the same thread — under one process-wide [`Mutex`]. Calls
//! are serialized. No other code in the process may call SEP directly; the
//! `links = "sep"` key in `Cargo.toml` keeps a second copy of the library out
//! of the binary.
//!
//! # Determinism
//!
//! Background estimation and everything about non-deblended objects is
//! deterministic. Deblended children (`SEP_OBJ_MERGED`) are not fully
//! portable: `deblend.c:gatherup` hands each parent pixel outside the
//! children's cores to one child drawn with `rand_r(&randseed)`.
//! `sep_extract` reseeds `randseed = 1`, so with a POSIX `rand_r` a call is
//! repeatable on one C library, but the generator differs between C
//! libraries (macOS libSystem and glibc give different children for the
//! parity fixture). On MSVC `sepcore.h` maps `rand_r(SEED)` to `rand()`,
//! which ignores the seed and keeps per-thread state between calls, so even
//! the same frame processed twice on one thread can deblend differently.
//! What does not depend on the generator: each child's barycentre
//! (`analyse.c` robust branch keeps the pre-assignment centre), and the
//! union of a parent's children, their total `npix` and their total flux.
//!
//! # Allocation and work bounds
//!
//! The C code misbehaves on some allocation-failure paths (for example
//! `sep_extract` frees an uninitialised `arraybuffer` if an early allocation
//! fails), so the inputs that size SEP's allocations are bounded here:
//! [`MAX_PIXELS`], [`MAX_PIXSTACK`], [`MAX_DEBLEND_NTHRESH`],
//! [`MAX_FILTER_MESHES`], [`MAX_MESHES`] and [`MAX_KERNEL_DIM`]. The one
//! super-linear step of `sep_background`, the bad-mesh search, is bounded by
//! [`MAX_BAD_MESH_WORK`].

mod ffi;

use std::borrow::Cow;
use std::fmt;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr::{self, NonNull};
use std::sync::{Mutex, PoisonError};

/// The vendored SEP release.
pub const SEP_VERSION: &str = "1.4.1";

/// C compiler flags this crate passes when building the vendored sources
/// (see `build.rs` for the reasoning behind each).
pub const BUILD_FLAGS: &str = env!("ACHERNAR_SEP_BUILD_FLAGS");

/// Largest accepted width or height: `MAXPICSIZE` (2^20) from SEP's
/// `extract.h`, which `analyse.c:preanalyse` relies on for its bounding-box
/// initialisation. It is also far below `c_int::MAX`, the limit the Python
/// wrapper enforces (`sep.pyx:_check_array_get_dims`).
pub const MAX_DIMENSION: usize = 1 << 20;

/// Largest accepted frame, 2^28 = 268,435,456 pixels (e.g. 16384 x 16384).
///
/// Large enough for the biggest current astronomical CMOS sensors (about
/// 151 MP) with margin; small enough that linear pixel indices fit `u32`
/// (footprints are returned as `u32`) and that the maps this API allocates
/// (`f32` background + RMS, 8 bytes per pixel) stay within 2 GiB.
pub const MAX_PIXELS: usize = 1 << 28;

/// Largest accepted extraction pixel stack. SEP allocates
/// `pixstack * plistsize` bytes up front and touches every page
/// (`extract.c`; `plistsize` is 32-44 bytes on LP64), so 10^7 bounds that
/// allocation near 440 MB. The SEP default is 300,000 (about 13 MB).
pub const MAX_PIXSTACK: usize = 10_000_000;

/// SEP's default pixel stack (`extract.c`).
pub const DEFAULT_PIXSTACK: usize = 300_000;

/// SEP's default sub-object limit (`deblend.c` `nsonmax`).
pub const DEFAULT_SUB_OBJECT_LIMIT: usize = 1024;

/// `deblend.c` stores sub-object indices as `short`.
pub const MAX_SUB_OBJECT_LIMIT: usize = i16::MAX as usize;

/// `allocdeblend` allocates `deblend_nthresh * nsonmax * 16` shorts in `int`
/// arithmetic; 256 thresholds at the default limit is 8 MiB.
pub const MAX_DEBLEND_NTHRESH: usize = 256;

/// Largest accepted mesh-filter size (meshes). `filterback` allocates two
/// windows of `(2*(fw/2)+1) * (2*(fh/2)*nx+1)` floats.
pub const MAX_FILTER_MESHES: usize = 63;

/// Largest accepted mesh grid (`nx * ny`); SEP keeps four `float` grids.
pub const MAX_MESHES: usize = 1 << 20;

/// Largest accepted `bad_meshes * total_meshes` for one `sep_background`.
///
/// `filterback` replaces every mesh below support by scanning all meshes for
/// the nearest supported ones (`background.c`, nested loop), so its cost is
/// proportional to that product; everything else in `sep_background` is
/// linear in pixels or meshes. Measured on the development host (Apple M3
/// Pro, release build, `tests::bad_mesh_work_host_timing`, half of each frame
/// masked): 0.50-0.54 ns per unit, and 1.95e9 units took 1.00 s for the whole
/// call. 1.5e9 therefore keeps one call near 0.8 s on that host while the
/// process-wide SEP lock is held (an available-host figure, not a deployment
/// budget). Typical frames are far below it: 64 px meshes on 6000x4000 are
/// 5,952 meshes, at most 3.5e7 units even with every mesh bad. Frames above
/// the bound fail with [`SepError::WorkBound`] before SEP is called.
pub const MAX_BAD_MESH_WORK: u64 = 1_500_000_000;

/// Largest accepted convolution kernel width/height.
pub const MAX_KERNEL_DIM: usize = 63;

/// SEP's `BIG` (`sepcore.h`, 1e30). `backstat` treats `pix <= -BIG` as a bad
/// pixel and masked noise is set to `BIG`, so data must stay below it.
const SEP_BIG: f32 = 1e30;

/// Object flags from `sep.h`, with the numeric values SEP 1.4.1 uses.
pub mod flags {
    /// Object is the result of deblending.
    pub const OBJ_MERGED: u16 = 0x0001;
    /// Object footprint touches the image boundary (row 0, row h-1, column
    /// 0 or column w-1; `extract.c`).
    pub const OBJ_TRUNC: u16 = 0x0002;
    /// Defined but never set in an output catalog: deblending overflow is a
    /// fatal error (`DEBLEND_OVERFLOW`) in 1.4.1.
    pub const OBJ_DOVERFLOW: u16 = 0x0004;
    /// x and y fully correlated; SEP added 1/12 to x2 and y2 (`analyse.c`).
    pub const OBJ_SINGU: u16 = 0x0008;
    /// Aperture flags (aperture photometry is not exposed by this crate).
    pub const APER_TRUNC: u16 = 0x0010;
    pub const APER_HASMASKED: u16 = 0x0020;
    pub const APER_ALLMASKED: u16 = 0x0040;
    pub const APER_NONPOSITIVE: u16 = 0x0080;
}

/// SEP status codes (`sepcore.h`, not installed upstream; values copied).
pub mod status {
    pub const MEMORY_ALLOC_ERROR: i32 = 1;
    pub const PIXSTACK_FULL: i32 = 2;
    pub const ILLEGAL_DTYPE: i32 = 3;
    pub const ILLEGAL_SUBPIX: i32 = 4;
    pub const NON_ELLIPSE_PARAMS: i32 = 5;
    pub const ILLEGAL_APER_PARAMS: i32 = 6;
    pub const DEBLEND_OVERFLOW: i32 = 7;
    pub const LINE_NOT_IN_BUF: i32 = 8;
    pub const RELTHRESH_NO_NOISE: i32 = 9;
    pub const UNKNOWN_NOISE_TYPE: i32 = 10;
}

/// Failure of a SEP call or of the input validation in front of it.
#[derive(Debug, Clone, PartialEq)]
pub enum SepError {
    /// Zero dimension, a dimension above [`MAX_DIMENSION`], or a buffer
    /// whose length is not `width * height`.
    InvalidShape {
        width: usize,
        height: usize,
        len: usize,
        reason: &'static str,
    },
    /// More than [`MAX_PIXELS`] pixels.
    FrameTooLarge {
        width: usize,
        height: usize,
        limit: usize,
    },
    MaskLengthMismatch {
        expected: usize,
        actual: usize,
    },
    NoiseLengthMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidKernel {
        reason: String,
    },
    /// An unmasked data pixel is NaN, infinite, or at least SEP's `BIG`
    /// (1e30) in magnitude.
    NonFinitePixel {
        x: usize,
        y: usize,
        value: f32,
    },
    /// An unmasked noise value (or the scalar noise) is not finite and
    /// positive. Zero noise is rejected: with a relative threshold it makes
    /// the pixel's threshold zero.
    InvalidNoise {
        x: Option<usize>,
        y: Option<usize>,
        value: f64,
    },
    InvalidParameter {
        name: &'static str,
        reason: String,
    },
    /// SEP returned a structure that violates its own documented invariants
    /// (index out of range, null array). Not expected; checked so that a
    /// library defect cannot become an out-of-bounds access here.
    InconsistentOutput {
        reason: String,
    },
    /// The frame would make `filterback`'s bad-mesh search exceed
    /// [`MAX_BAD_MESH_WORK`]; refused before calling SEP.
    WorkBound {
        bad_meshes: usize,
        meshes: usize,
        limit: u64,
    },
    /// SEP returned a non-zero status: `code` (see [`status`]), the
    /// `sep_get_errmsg` text, and the `sep_get_errdetail` text (may be empty).
    Library {
        code: i32,
        message: String,
        detail: String,
    },
}

impl fmt::Display for SepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SepError::InvalidShape {
                width,
                height,
                len,
                reason,
            } => write!(f, "invalid shape {width}x{height} (len {len}): {reason}"),
            SepError::FrameTooLarge {
                width,
                height,
                limit,
            } => write!(f, "frame {width}x{height} exceeds {limit} pixels"),
            SepError::MaskLengthMismatch { expected, actual } => {
                write!(f, "mask length {actual}, expected {expected}")
            }
            SepError::NoiseLengthMismatch { expected, actual } => {
                write!(f, "noise length {actual}, expected {expected}")
            }
            SepError::InvalidKernel { reason } => write!(f, "invalid kernel: {reason}"),
            SepError::NonFinitePixel { x, y, value } => {
                write!(f, "unmasked pixel ({x}, {y}) is not usable by SEP: {value}")
            }
            SepError::InvalidNoise { x, y, value } => match (x, y) {
                (Some(x), Some(y)) => write!(f, "unmasked noise at ({x}, {y}) is {value}"),
                _ => write!(f, "scalar noise is {value}"),
            },
            SepError::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter {name}: {reason}")
            }
            SepError::WorkBound {
                bad_meshes,
                meshes,
                limit,
            } => write!(
                f,
                "work bound: {bad_meshes} of {meshes} meshes below support would cost \
                 {} > {limit} bad-mesh search units",
                *bad_meshes as u64 * *meshes as u64
            ),
            SepError::InconsistentOutput { reason } => {
                write!(f, "inconsistent SEP output: {reason}")
            }
            SepError::Library {
                code,
                message,
                detail,
            } => {
                if detail.is_empty() {
                    write!(f, "SEP error {code}: {message}")
                } else {
                    write!(f, "SEP error {code}: {message}: {detail}")
                }
            }
        }
    }
}

impl std::error::Error for SepError {}

/// The process-wide lock around every SEP call sequence (see module docs).
static SEP_LOCK: Mutex<()> = Mutex::new(());

fn with_sep<R>(f: impl FnOnce() -> R) -> R {
    // The protected state is SEP's C globals, which every call sequence sets
    // before use, so a panic in another holder leaves nothing to repair.
    let _guard = SEP_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    f()
}

fn c_buffer_to_string(buffer: &[c_char]) -> String {
    let bytes: Vec<u8> = buffer
        .iter()
        .map(|&c| c as u8)
        .take_while(|&b| b != 0)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Reads and clears SEP's thread-local error detail.
///
/// # Safety
/// Caller holds `SEP_LOCK`, on the thread that made the failing call.
unsafe fn take_errdetail() -> String {
    let mut detail = [0 as c_char; ffi::ERRDETAIL_LEN];
    // SAFETY: `sep_get_errdetail` strcpy's at most DETAILSIZE (512) bytes
    // including the NUL into the buffer.
    unsafe { ffi::sep_get_errdetail(detail.as_mut_ptr()) };
    detail[ffi::ERRDETAIL_LEN - 1] = 0;
    c_buffer_to_string(&detail)
}

/// # Safety
/// Caller holds `SEP_LOCK`, on the thread that made the failing call.
unsafe fn library_error(code: c_int) -> SepError {
    let mut message = [0 as c_char; ffi::ERRMSG_LEN];
    // SAFETY: `sep_get_errmsg` writes at most 60 characters plus NUL.
    unsafe { ffi::sep_get_errmsg(code, message.as_mut_ptr()) };
    message[ffi::ERRMSG_LEN - 1] = 0;
    SepError::Library {
        code,
        message: c_buffer_to_string(&message),
        // SAFETY: forwarded caller contract.
        detail: unsafe { take_errdetail() },
    }
}

/// The version string compiled into the C library (`sep_version_string`).
pub fn library_version() -> String {
    with_sep(|| {
        // SAFETY: `sep_version_string` is a `const char * const` initialised
        // from the SEP_VERSION_STRING literal (util.c); never null, NUL
        // terminated, immutable.
        unsafe { std::ffi::CStr::from_ptr(ffi::sep_version_string) }
            .to_string_lossy()
            .into_owned()
    })
}

fn check_geometry(len: usize, width: usize, height: usize) -> Result<usize, SepError> {
    let shape = |reason| SepError::InvalidShape {
        width,
        height,
        len,
        reason,
    };
    if width == 0 || height == 0 {
        return Err(shape("width and height must be at least 1"));
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(shape("dimension exceeds SEP MAXPICSIZE (2^20)"));
    }
    let pixels = width
        .checked_mul(height)
        .filter(|&n| n <= MAX_PIXELS)
        .ok_or(SepError::FrameTooLarge {
            width,
            height,
            limit: MAX_PIXELS,
        })?;
    if len != pixels {
        return Err(shape("buffer length differs from width * height"));
    }
    Ok(pixels)
}

fn check_mask(mask: Option<&[u8]>, pixels: usize) -> Result<(), SepError> {
    match mask {
        Some(mask) if mask.len() != pixels => Err(SepError::MaskLengthMismatch {
            expected: pixels,
            actual: mask.len(),
        }),
        _ => Ok(()),
    }
}

fn is_masked(mask: Option<&[u8]>, index: usize) -> bool {
    mask.is_some_and(|mask| mask[index] != 0)
}

/// Whether SEP can use `value` as data: finite and below its `BIG` sentinel
/// (1e30) in magnitude. Callers building a detector mask should exclude every
/// pixel for which this is false; unmasked ones are rejected.
pub fn usable_data_value(value: f32) -> bool {
    value.is_finite() && value.abs() < SEP_BIG
}

fn usable_value(value: f32) -> bool {
    usable_data_value(value)
}

/// SEP's mesh-support rule (`background.c:backstat`, `BACK_MINGOODFRAC =
/// 0.5`): mesh `(i, j)` spans its *actual* pixels (the last column and row of
/// meshes are partial) and is "bad" iff
/// `(float)npix < (float)(bw_i * h_j * 0.5)`, where `npix` counts pixels with
/// `mask <= 0` and `pix > -BIG`. For data this crate accepts (unmasked pixels
/// are usable) that is the unmasked count. One flag per mesh, row-major,
/// `true` = supported.
pub fn mesh_support(mask: Option<&[u8]>, width: usize, height: usize, grid: MeshGrid) -> Vec<bool> {
    let Some(mask) = mask else {
        return vec![true; grid.nx * grid.ny];
    };
    let mut usable = vec![0u64; grid.nx * grid.ny];
    for (y, row) in mask.chunks_exact(width).take(height).enumerate() {
        let mesh_row = (y / grid.bh) * grid.nx;
        for (x, &m) in row.iter().enumerate() {
            if m == 0 {
                usable[mesh_row + x / grid.bw] += 1;
            }
        }
    }
    usable
        .iter()
        .enumerate()
        .map(|(index, &npix)| {
            let (i, j) = (index % grid.nx, index / grid.nx);
            let mesh_w = (width - i * grid.bw).min(grid.bw);
            let mesh_h = (height - j * grid.bh).min(grid.bh);
            !((npix as f32) < ((mesh_w * mesh_h) as f64 * 0.5) as f32)
        })
        .collect()
}

fn first_unusable_unmasked(
    data: &[f32],
    mask: Option<&[u8]>,
    width: usize,
) -> Result<(), SepError> {
    match data
        .iter()
        .enumerate()
        .find(|&(i, &v)| !usable_value(v) && !is_masked(mask, i))
    {
        Some((i, &value)) => Err(SepError::NonFinitePixel {
            x: i % width,
            y: i / width,
            value,
        }),
        None => Ok(()),
    }
}

fn parameter(name: &'static str, reason: impl Into<String>) -> SepError {
    SepError::InvalidParameter {
        name,
        reason: reason.into(),
    }
}

/// `sep_background` parameters. SEP/SExtractor defaults: 64 x 64 px meshes,
/// 3 x 3 mesh median filter, `fthresh = 0` (always filter).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BackgroundParams {
    /// Mesh width and height in pixels. The last mesh column/row is partial
    /// when the frame is not a multiple (`backstat` uses its actual size).
    pub bw: usize,
    pub bh: usize,
    /// Median-filter size in meshes; odd, because `filterback` uses the
    /// half-width `fw/2` and an even size would silently act as `fw + 1`.
    pub fw: usize,
    pub fh: usize,
    /// A mesh is replaced by its window median iff
    /// `|median - value| >= fthresh` (input units).
    pub fthresh: f64,
}

impl Default for BackgroundParams {
    fn default() -> Self {
        Self {
            bw: 64,
            bh: 64,
            fw: 3,
            fh: 3,
            fthresh: 0.0,
        }
    }
}

impl BackgroundParams {
    fn validate(&self, width: usize, height: usize) -> Result<(), SepError> {
        for (name, value) in [("bw", self.bw), ("bh", self.bh)] {
            if value == 0 || value > MAX_DIMENSION {
                return Err(parameter(
                    name,
                    format!("{value} not in 1..={MAX_DIMENSION}"),
                ));
            }
        }
        for (name, value) in [("fw", self.fw), ("fh", self.fh)] {
            if value == 0 || value > MAX_FILTER_MESHES || value % 2 == 0 {
                return Err(parameter(
                    name,
                    format!("{value} is not an odd size in 1..={MAX_FILTER_MESHES}"),
                ));
            }
        }
        if !self.fthresh.is_finite() || self.fthresh < 0.0 {
            return Err(parameter(
                "fthresh",
                format!("{} is not finite and >= 0", self.fthresh),
            ));
        }
        let grid = MeshGrid::for_frame(width, height, self.bw, self.bh);
        if grid.nx * grid.ny > MAX_MESHES {
            return Err(parameter(
                "bw/bh",
                format!("{}x{} meshes exceed {MAX_MESHES}", grid.nx, grid.ny),
            ));
        }
        Ok(())
    }
}

/// SEP's mesh tiling: `nx = (w-1)/bw + 1`, `ny = (h-1)/bh + 1`
/// (`background.c:sep_background`). Mesh `(i, j)` covers columns
/// `i*bw .. min((i+1)*bw, w)` and rows `j*bh .. min((j+1)*bh, h)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshGrid {
    pub nx: usize,
    pub ny: usize,
    pub bw: usize,
    pub bh: usize,
}

impl MeshGrid {
    pub fn for_frame(width: usize, height: usize, bw: usize, bh: usize) -> Self {
        Self {
            nx: (width.max(1) - 1) / bw.max(1) + 1,
            ny: (height.max(1) - 1) / bh.max(1) + 1,
            bw,
            bh,
        }
    }
}

/// An owned SEP background model (`sep_bkg`), freed with `sep_bkg_free` on
/// drop.
pub struct Background {
    ptr: NonNull<ffi::sep_bkg>,
    width: usize,
    height: usize,
    grid: MeshGrid,
    global_level: f32,
    global_rms: f32,
    mesh_level: Vec<f32>,
    mesh_rms: Vec<f32>,
}

// SAFETY: `Background` exclusively owns the `sep_bkg` allocation; nothing
// else holds the pointer. SEP functions taking it are called only under
// `SEP_LOCK` and keep no thread-affine state about it.
unsafe impl Send for Background {}
// SAFETY: `&Background` methods only pass the pointer to SEP functions that
// take `const sep_bkg *` and allocate their own temporaries
// (`bkg_line_flt_internal`), and those calls are serialized by `SEP_LOCK`.
unsafe impl Sync for Background {}

impl fmt::Debug for Background {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Background")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("grid", &self.grid)
            .field("global_level", &self.global_level)
            .field("global_rms", &self.global_rms)
            .finish()
    }
}

type ArrayEvaluator = unsafe extern "C" fn(*const ffi::sep_bkg, *mut c_void, c_int) -> c_int;
type LineEvaluator = unsafe extern "C" fn(*const ffi::sep_bkg, i64, *mut c_void, c_int) -> c_int;

impl Background {
    /// Runs `sep_background` on a row-major `f32` frame.
    ///
    /// `mask`: one byte per pixel, non-zero excludes the pixel. A mesh with
    /// fewer than half of its pixels usable is "bad" and takes the average of
    /// its nearest good meshes (`backstat`, `BACK_MINGOODFRAC = 0.5`;
    /// `filterback`). If *no* mesh is good SEP still succeeds, with level 0
    /// and RMS 1 everywhere; callers that need support must check it
    /// themselves (see [`MeshGrid`]).
    pub fn estimate(
        data: &[f32],
        width: usize,
        height: usize,
        mask: Option<&[u8]>,
        params: &BackgroundParams,
    ) -> Result<Self, SepError> {
        Self::estimate_checked(data, width, height, mask, params, true)
    }

    /// [`Background::estimate`] without [`MAX_BAD_MESH_WORK`], to measure it.
    #[cfg(test)]
    fn estimate_unbounded(
        data: &[f32],
        width: usize,
        height: usize,
        mask: Option<&[u8]>,
        params: &BackgroundParams,
    ) -> Result<Self, SepError> {
        Self::estimate_checked(data, width, height, mask, params, false)
    }

    fn estimate_checked(
        data: &[f32],
        width: usize,
        height: usize,
        mask: Option<&[u8]>,
        params: &BackgroundParams,
        bound_work: bool,
    ) -> Result<Self, SepError> {
        let pixels = check_geometry(data.len(), width, height)?;
        check_mask(mask, pixels)?;
        params.validate(width, height)?;
        first_unusable_unmasked(data, mask, width)?;
        if bound_work {
            let grid = MeshGrid::for_frame(width, height, params.bw, params.bh);
            let meshes = grid.nx * grid.ny;
            let bad_meshes = mesh_support(mask, width, height, grid)
                .iter()
                .filter(|ok| !**ok)
                .count();
            if bad_meshes as u64 * meshes as u64 > MAX_BAD_MESH_WORK {
                return Err(SepError::WorkBound {
                    bad_meshes,
                    meshes,
                    limit: MAX_BAD_MESH_WORK,
                });
            }
        }

        // `backhisto` computes `(int64_t)(pix / qscale + cste)` for every
        // pixel before testing its mask, which is undefined behaviour in C for
        // NaN/inf. Masked pixels never enter the statistics, so replacing the
        // unusable ones by 0 changes no result and removes the UB.
        let data: Cow<[f32]> = match mask {
            Some(m)
                if data
                    .iter()
                    .zip(m)
                    .any(|(&v, &k)| k != 0 && !usable_value(v)) =>
            {
                Cow::Owned(
                    data.iter()
                        .zip(m)
                        .map(|(&v, &k)| if k != 0 && !usable_value(v) { 0.0 } else { v })
                        .collect(),
                )
            }
            _ => Cow::Borrowed(data),
        };

        let image = ffi::sep_image {
            data: data.as_ptr().cast(),
            noise: ptr::null(),
            mask: mask.map_or(ptr::null(), |m| m.as_ptr().cast()),
            segmap: ptr::null(),
            dtype: ffi::SEP_TFLOAT,
            ndtype: 0,
            mdtype: if mask.is_some() { ffi::SEP_TBYTE } else { 0 },
            sdtype: 0,
            segids: ptr::null_mut(),
            idcounts: ptr::null_mut(),
            numids: 0,
            w: width as i64,
            h: height as i64,
            noiseval: 0.0,
            noise_type: ffi::SEP_NOISE_NONE,
            gain: 0.0,
            maskthresh: 0.0,
        };

        let (ptr, global_level, global_rms) = with_sep(|| {
            let mut out: *mut ffi::sep_bkg = ptr::null_mut();
            // SAFETY: `image` points at `width*height` f32 (and u8 mask)
            // values that outlive the call; dimensions and parameters were
            // validated above; `out` is a valid out-pointer. Lock held.
            let code = unsafe {
                take_errdetail();
                ffi::sep_background(
                    &image,
                    params.bw as i64,
                    params.bh as i64,
                    params.fw as i64,
                    params.fh as i64,
                    params.fthresh,
                    &mut out,
                )
            };
            if code != ffi::RETURN_OK {
                // SAFETY: lock held, same thread. On error `background.c`
                // frees its output and sets `*bkg = NULL`; freeing a null
                // model is a no-op (`sep_bkg_free`).
                let error = unsafe { library_error(code) };
                unsafe { ffi::sep_bkg_free(out) };
                return Err(error);
            }
            let Some(out) = NonNull::new(out) else {
                return Err(SepError::InconsistentOutput {
                    reason: "sep_background returned OK and a null model".into(),
                });
            };
            // SAFETY: `out` is the valid model SEP just returned.
            let globals = unsafe {
                (
                    ffi::sep_bkg_global(out.as_ptr()),
                    ffi::sep_bkg_globalrms(out.as_ptr()),
                )
            };
            Ok((out, globals.0, globals.1))
        })?;

        // From here on the model is owned and freed by `Background::drop`,
        // including on the validation error below.
        let mut background = Self {
            ptr,
            width,
            height,
            grid: MeshGrid::for_frame(width, height, params.bw, params.bh),
            global_level,
            global_rms,
            mesh_level: Vec::new(),
            mesh_rms: Vec::new(),
        };
        // SAFETY: `ptr` is valid; the model is only read here.
        let model = unsafe { ptr.as_ref() };
        let expected = background.grid;
        if model.w != width as i64
            || model.h != height as i64
            || model.nx != expected.nx as i64
            || model.ny != expected.ny as i64
            || model.n != (expected.nx * expected.ny) as i64
            || model.back.is_null()
            || model.sigma.is_null()
        {
            return Err(SepError::InconsistentOutput {
                reason: format!(
                    "sep_bkg {}x{} grid {}x{} (n {}) for a {width}x{height} frame, \
                     expected grid {}x{}",
                    model.w, model.h, model.nx, model.ny, model.n, expected.nx, expected.ny
                ),
            });
        }
        let meshes = expected.nx * expected.ny;
        // SAFETY: `back` and `sigma` are `n = nx*ny` floats allocated by
        // `sep_background` and owned by the model.
        unsafe {
            background.mesh_level = std::slice::from_raw_parts(model.back, meshes).to_vec();
            background.mesh_rms = std::slice::from_raw_parts(model.sigma, meshes).to_vec();
        }
        Ok(background)
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// `sep_bkg_global`: the median of the filtered mesh levels
    /// (`filterback`, `fqmedian`; mean of the two middle values for an even
    /// mesh count). The `sep.h` struct comment "global mean" is wrong.
    pub fn global_level(&self) -> f32 {
        self.global_level
    }

    /// `sep_bkg_globalrms`: the median of the filtered mesh RMS values. If
    /// that median is <= 0, SEP substitutes the median of the strictly
    /// positive mesh RMS values, or **1.0** if there is none (`filterback`).
    /// Check [`Background::mesh_rms`] before treating it as measured.
    pub fn global_rms(&self) -> f32 {
        self.global_rms
    }

    pub fn mesh_grid(&self) -> MeshGrid {
        self.grid
    }

    /// Filtered mesh levels, `nx*ny`, row-major (x fastest): the spline nodes.
    pub fn mesh_level(&self) -> &[f32] {
        &self.mesh_level
    }

    /// Filtered mesh RMS values (clipped histogram sigma), `nx*ny`, row-major.
    pub fn mesh_rms(&self) -> &[f32] {
        &self.mesh_rms
    }

    /// Full-resolution background `B(x,y)` from `sep_bkg_array`: bicubic
    /// (separable natural cubic spline) interpolation of the mesh nodes
    /// (`background.c:bkg_line_flt_internal`). This is what Python
    /// `sep.Background.back()` returns. Not `sep_bkg_pix`, which is bilinear
    /// with nodes offset by half a pixel in x and is never used by Python.
    pub fn back_array(&self) -> Result<Vec<f32>, SepError> {
        self.evaluate_array(ffi::sep_bkg_array)
    }

    /// Full-resolution RMS `R(x,y)` from `sep_bkg_rmsarray` (same spline as
    /// [`Background::back_array`]); Python `sep.Background.rms()`.
    pub fn rms_array(&self) -> Result<Vec<f32>, SepError> {
        self.evaluate_array(ffi::sep_bkg_rmsarray)
    }

    /// Row `y` of [`Background::back_array`] (`sep_bkg_line`).
    pub fn back_line(&self, y: usize, out: &mut [f32]) -> Result<(), SepError> {
        self.evaluate_line(ffi::sep_bkg_line, y, out)
    }

    /// Row `y` of [`Background::rms_array`] (`sep_bkg_rmsline`).
    pub fn rms_line(&self, y: usize, out: &mut [f32]) -> Result<(), SepError> {
        self.evaluate_line(ffi::sep_bkg_rmsline, y, out)
    }

    fn evaluate_array(&self, evaluator: ArrayEvaluator) -> Result<Vec<f32>, SepError> {
        let mut out = vec![0.0f32; self.width * self.height];
        with_sep(|| {
            // SAFETY: `out` holds w*h floats, the SEP_TFLOAT fast path writes
            // exactly h rows of w floats; model valid; lock held.
            let code = unsafe {
                take_errdetail();
                evaluator(self.ptr.as_ptr(), out.as_mut_ptr().cast(), ffi::SEP_TFLOAT)
            };
            if code != ffi::RETURN_OK {
                // SAFETY: lock held, same thread.
                return Err(unsafe { library_error(code) });
            }
            Ok(())
        })?;
        Ok(out)
    }

    fn evaluate_line(
        &self,
        evaluator: LineEvaluator,
        y: usize,
        out: &mut [f32],
    ) -> Result<(), SepError> {
        if y >= self.height {
            return Err(parameter(
                "y",
                format!("row {y} outside 0..{}", self.height),
            ));
        }
        if out.len() != self.width {
            return Err(SepError::InvalidShape {
                width: self.width,
                height: 1,
                len: out.len(),
                reason: "line buffer length differs from width",
            });
        }
        with_sep(|| {
            // SAFETY: `out` holds w floats, the SEP_TFLOAT path writes w;
            // `y` is in range; model valid; lock held.
            let code = unsafe {
                take_errdetail();
                evaluator(
                    self.ptr.as_ptr(),
                    y as i64,
                    out.as_mut_ptr().cast(),
                    ffi::SEP_TFLOAT,
                )
            };
            if code != ffi::RETURN_OK {
                // SAFETY: lock held, same thread.
                return Err(unsafe { library_error(code) });
            }
            Ok(())
        })
    }
}

impl Drop for Background {
    fn drop(&mut self) {
        // No code path drops a `Background` while holding `SEP_LOCK`.
        with_sep(|| {
            // SAFETY: `ptr` came from `sep_background`, is owned uniquely and
            // freed exactly once here.
            unsafe { ffi::sep_bkg_free(self.ptr.as_ptr()) }
        });
    }
}

/// Noise passed to `sep_extract`, as a standard deviation in data units.
#[derive(Debug, Clone, Copy)]
pub enum Noise<'a> {
    /// No noise: the threshold is absolute (data units).
    None,
    /// One standard deviation for every pixel: the threshold is
    /// `thresh * sigma`, and SEP uses the CONV filter even if MATCHED was
    /// asked for (`extract.c`).
    Scalar(f64),
    /// Per-pixel standard deviation, row-major: the threshold is
    /// `thresh * sigma(x,y)`.
    Array(&'a [f32]),
}

/// SEP filter types (`sep.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    /// Correlate with the kernel normalised by `sum|k|`; detect where the
    /// filtered value exceeds `thresh * sigma(x,y)` (`convolve.c:convolve`).
    Conv,
    /// `sum(k f / v) / sqrt(sum(k^2 / v))` compared with `thresh` itself
    /// (`convolve.c:matched_filter`). Only effective with a kernel and a noise
    /// array; SEP silently uses CONV otherwise.
    Matched,
}

/// A convolution kernel, row-major. Odd sizes only: SEP centres the kernel at
/// `(convw/2, convh/2)`, so an even kernel shifts the detection image by half
/// a pixel.
#[derive(Debug, Clone, PartialEq)]
pub struct Kernel {
    values: Vec<f32>,
    width: usize,
    height: usize,
}

impl Kernel {
    pub fn new(values: Vec<f32>, width: usize, height: usize) -> Result<Self, SepError> {
        let invalid = |reason: String| SepError::InvalidKernel { reason };
        for (name, size) in [("width", width), ("height", height)] {
            if size == 0 || size > MAX_KERNEL_DIM || size % 2 == 0 {
                return Err(invalid(format!(
                    "{name} {size} is not odd in 1..={MAX_KERNEL_DIM}"
                )));
            }
        }
        if values.len() != width * height {
            return Err(invalid(format!(
                "{} values for a {width}x{height} kernel",
                values.len()
            )));
        }
        if let Some(value) = values.iter().find(|v| !v.is_finite()) {
            return Err(invalid(format!("non-finite value {value}")));
        }
        // `sep_extract` normalises by sum|k|; zero would divide by zero.
        if values.iter().map(|v| v.abs() as f64).sum::<f64>() <= 0.0 {
            return Err(invalid("sum of |k| is zero".into()));
        }
        Ok(Self {
            values,
            width,
            height,
        })
    }

    /// Kernel from rows (`rows[y][x]`); all rows must have the same length.
    pub fn from_rows(rows: &[Vec<f32>]) -> Result<Self, SepError> {
        let height = rows.len();
        let width = rows.first().map_or(0, Vec::len);
        if rows.iter().any(|row| row.len() != width) {
            return Err(SepError::InvalidKernel {
                reason: "rows have different lengths".into(),
            });
        }
        Self::new(rows.concat(), width, height)
    }

    /// The SEP/SExtractor default 3x3 kernel `[[1,2,1],[2,4,2],[1,2,1]]`
    /// (`sep.pyx` `default_kernel`).
    pub fn sep_default() -> Self {
        Self {
            values: vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0],
            width: 3,
            height: 3,
        }
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }
}

/// `sep_extract` parameters. [`ExtractParams::new`] uses the Python
/// `sep.extract` defaults apart from the threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractParams {
    /// Relative to the noise when [`Noise::Scalar`]/[`Noise::Array`] is
    /// given (`SEP_THRESH_REL`, as the Python wrapper does), absolute
    /// otherwise.
    pub thresh: f32,
    pub minarea: usize,
    /// `None` disables filtering.
    pub filter: Option<Kernel>,
    pub filter_type: FilterType,
    pub deblend_nthresh: usize,
    /// Minimum contrast for a deblended branch, as a fraction of the parent
    /// flux; 1.0 disables deblending.
    pub deblend_cont: f64,
    pub clean: bool,
    pub clean_param: f64,
    /// SEP's active-object pixel stack (`sep_set_extract_pixstack`).
    pub pixstack: usize,
    /// SEP's per-level sub-object limit (`sep_set_sub_object_limit`).
    pub sub_object_limit: usize,
    /// Build the segmentation map (object index + 1 per pixel, 0 for sky),
    /// as `sep.extract(..., segmentation_map=True)`.
    pub segmentation_map: bool,
}

impl ExtractParams {
    pub fn new(thresh: f32) -> Self {
        Self {
            thresh,
            minarea: 5,
            filter: Some(Kernel::sep_default()),
            filter_type: FilterType::Matched,
            deblend_nthresh: 32,
            deblend_cont: 0.005,
            clean: true,
            clean_param: 1.0,
            pixstack: DEFAULT_PIXSTACK,
            sub_object_limit: DEFAULT_SUB_OBJECT_LIMIT,
            segmentation_map: false,
        }
    }

    fn validate(&self, relative: bool) -> Result<(), SepError> {
        if !self.thresh.is_finite() || (relative && self.thresh <= 0.0) {
            return Err(parameter(
                "thresh",
                format!("{} is not finite (and positive when relative)", self.thresh),
            ));
        }
        // `minarea <= 0` with cleaning writes into a zero-size allocation
        // (`analysemthresh`); it is a C `int`.
        if self.minarea == 0 || self.minarea > c_int::MAX as usize {
            return Err(parameter(
                "minarea",
                format!("{} not in 1..=INT_MAX", self.minarea),
            ));
        }
        if self.deblend_nthresh == 0 || self.deblend_nthresh > MAX_DEBLEND_NTHRESH {
            return Err(parameter(
                "deblend_nthresh",
                format!("{} not in 1..={MAX_DEBLEND_NTHRESH}", self.deblend_nthresh),
            ));
        }
        if !(0.0..=1.0).contains(&self.deblend_cont) {
            return Err(parameter(
                "deblend_cont",
                format!("{} not in [0, 1]", self.deblend_cont),
            ));
        }
        // `clean` evaluates pow(amp/thresh, 1/clean_param).
        if !self.clean_param.is_finite() || self.clean_param <= 0.0 {
            return Err(parameter(
                "clean_param",
                format!("{} is not finite and positive", self.clean_param),
            ));
        }
        if self.pixstack == 0 || self.pixstack > MAX_PIXSTACK {
            return Err(parameter(
                "pixstack",
                format!("{} not in 1..={MAX_PIXSTACK}", self.pixstack),
            ));
        }
        if self.sub_object_limit == 0 || self.sub_object_limit > MAX_SUB_OBJECT_LIMIT {
            return Err(parameter(
                "sub_object_limit",
                format!(
                    "{} not in 1..={MAX_SUB_OBJECT_LIMIT}",
                    self.sub_object_limit
                ),
            ));
        }
        Ok(())
    }
}

/// One extracted object, copied out of the C catalog. Coordinates are 0-based
/// with (0, 0) at the centre of the first pixel, x the column.
#[derive(Debug, Clone, PartialEq)]
pub struct Object {
    /// Object threshold (data units); the mean of per-pixel thresholds when
    /// the threshold varies (`extract.c:get_mean_thresh`).
    pub thresh: f32,
    /// Footprint size.
    pub npix: usize,
    /// Footprint pixels whose unfiltered value exceeds `thresh`.
    pub tnpix: usize,
    pub xmin: usize,
    pub xmax: usize,
    pub ymin: usize,
    pub ymax: usize,
    /// First moments weighted by the *filtered* values when a kernel is used,
    /// by the data otherwise (`analyse.c:analyse`).
    pub x: f64,
    pub y: f64,
    pub x2: f64,
    pub y2: f64,
    pub xy: f64,
    pub errx2: f64,
    pub erry2: f64,
    pub errxy: f64,
    pub a: f32,
    pub b: f32,
    /// Radians, `atan2(2 xy, x2 - y2) / 2`, from +x toward +y.
    pub theta: f32,
    pub cxx: f32,
    pub cyy: f32,
    pub cxy: f32,
    /// Sum of filtered values over the footprint.
    pub cflux: f32,
    /// Sum of data values over the footprint (masked pixels count as 0).
    pub flux: f32,
    pub cpeak: f32,
    pub peak: f32,
    pub xcpeak: usize,
    pub ycpeak: usize,
    /// Pixel of the unfiltered peak.
    pub xpeak: usize,
    pub ypeak: usize,
    /// Raw SEP flags (see [`flags`]).
    pub flag: u16,
    /// Footprint as row-major linear indices `x + width*y`.
    pub pixels: Vec<u32>,
}

impl Object {
    pub fn has_flag(&self, flag: u16) -> bool {
        self.flag & flag != 0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Catalog {
    pub objects: Vec<Object>,
    /// Segmentation map when requested: `i + 1` on the pixels of
    /// `objects[i]`, 0 elsewhere (as `sep.pyx` builds it).
    pub segmentation: Option<Vec<i32>>,
    /// The filter SEP actually applied (`None`: no kernel). MATCHED becomes
    /// CONV unless a kernel and a noise array are both present.
    pub filter_applied: Option<FilterType>,
    /// Whether `thresh` was relative to the noise.
    pub thresh_relative: bool,
}

/// Groups deblended children (`SEP_OBJ_MERGED`) into families: objects whose
/// footprints are 8-connected through other children, i.e. the children of
/// one detected parent. A detection footprint is an 8-connected component
/// (`lutz.c`), and two separate detections cannot touch, so the families are
/// the parents.
///
/// `objects` yields each object's footprint (linear indices `x + width*y`)
/// and whether it is a deblended child, in catalog order; the result lists
/// catalog indices per family, ascending, families ordered by first member.
///
/// Within a family, which child receives a faint pixel depends on the C
/// library's `rand_r` (see the crate docs), while the family's pixel set,
/// total `npix` and total flux do not.
pub fn deblend_families<'a>(
    objects: impl IntoIterator<Item = (&'a [u32], bool)>,
    width: usize,
    height: usize,
) -> Vec<Vec<usize>> {
    const NONE: usize = usize::MAX;
    let mut owner = vec![NONE; width * height];
    let mut merged = Vec::new();
    for (index, (pixels, is_child)) in objects.into_iter().enumerate() {
        if is_child {
            merged.push(index);
            for &p in pixels {
                if let Some(slot) = owner.get_mut(p as usize) {
                    *slot = index;
                }
            }
        }
    }
    let count = merged.iter().max().map_or(0, |&m| m + 1);
    let mut parent: Vec<usize> = (0..count).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for y in 0..height {
        for x in 0..width {
            let a = owner[y * width + x];
            if a == NONE {
                continue;
            }
            // Right, below-left, below and below-right cover all 8 neighbours
            // once per pair.
            for (dx, dy) in [(1isize, 0isize), (-1, 1), (0, 1), (1, 1)] {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if nx < 0 || nx >= width as isize || ny >= height as isize {
                    continue;
                }
                let b = owner[ny as usize * width + nx as usize];
                if b != NONE && b != a {
                    let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                    parent[ra.max(rb)] = ra.min(rb);
                }
            }
        }
    }
    let mut families: Vec<Vec<usize>> = Vec::new();
    let mut family_of_root = vec![NONE; count];
    for &index in &merged {
        let r = root(&mut parent, index);
        if family_of_root[r] == NONE {
            family_of_root[r] = families.len();
            families.push(Vec::new());
        }
        families[family_of_root[r]].push(index);
    }
    families
}

/// Frees a `sep_catalog` on every exit path.
struct CatalogGuard(*mut ffi::sep_catalog);

impl Drop for CatalogGuard {
    fn drop(&mut self) {
        // SAFETY: the pointer is null or a catalog returned by `sep_extract`,
        // freed exactly once; `sep_catalog_free(NULL)` is a no-op. Dropped
        // inside the locked section that produced it.
        unsafe { ffi::sep_catalog_free(self.0) };
    }
}

/// Runs `sep_extract` on a background-subtracted frame.
///
/// `mask`: non-zero excludes a pixel. SEP sets masked data to 0 and masked
/// noise to 1e30 before filtering (`extract.c:apply_mask_line`), so masked
/// pixels contribute nothing to the filtered image (CONV: value 0; MATCHED:
/// variance overflows to infinity), but a masked pixel can still join a
/// footprint through its neighbours' filtered signal.
pub fn extract(
    residual: &[f32],
    width: usize,
    height: usize,
    mask: Option<&[u8]>,
    noise: Noise<'_>,
    params: &ExtractParams,
) -> Result<Catalog, SepError> {
    let pixels = check_geometry(residual.len(), width, height)?;
    check_mask(mask, pixels)?;
    let relative = !matches!(noise, Noise::None);
    params.validate(relative)?;
    match noise {
        Noise::None => {}
        Noise::Scalar(sigma) => {
            if !sigma.is_finite() || sigma <= 0.0 || sigma >= SEP_BIG as f64 {
                return Err(SepError::InvalidNoise {
                    x: None,
                    y: None,
                    value: sigma,
                });
            }
        }
        Noise::Array(sigma) => {
            if sigma.len() != pixels {
                return Err(SepError::NoiseLengthMismatch {
                    expected: pixels,
                    actual: sigma.len(),
                });
            }
            if let Some((i, &value)) = sigma
                .iter()
                .enumerate()
                .find(|&(i, &v)| !(usable_value(v) && v > 0.0) && !is_masked(mask, i))
            {
                return Err(SepError::InvalidNoise {
                    x: Some(i % width),
                    y: Some(i / width),
                    value: value as f64,
                });
            }
        }
    }
    first_unusable_unmasked(residual, mask, width)?;

    let filter_applied = params
        .filter
        .as_ref()
        .map(|_| match (params.filter_type, noise) {
            (FilterType::Matched, Noise::Array(_)) => FilterType::Matched,
            _ => FilterType::Conv,
        });

    let (noise_ptr, noiseval, noise_type, ndtype) = match noise {
        // `sep_extract` frees its noise buffer whenever `noise != NULL`, but
        // initialises it only for STDDEV/VAR: never pair a pointer with NONE.
        Noise::None => (ptr::null(), 0.0, ffi::SEP_NOISE_NONE, 0),
        Noise::Scalar(sigma) => (ptr::null(), sigma, ffi::SEP_NOISE_STDDEV, 0),
        Noise::Array(sigma) => (
            sigma.as_ptr().cast(),
            0.0,
            ffi::SEP_NOISE_STDDEV,
            ffi::SEP_TFLOAT,
        ),
    };
    let image = ffi::sep_image {
        data: residual.as_ptr().cast(),
        noise: noise_ptr,
        mask: mask.map_or(ptr::null(), |m| m.as_ptr().cast()),
        segmap: ptr::null(),
        dtype: ffi::SEP_TFLOAT,
        ndtype,
        mdtype: if mask.is_some() { ffi::SEP_TBYTE } else { 0 },
        sdtype: 0,
        segids: ptr::null_mut(),
        idcounts: ptr::null_mut(),
        numids: 0,
        w: width as i64,
        h: height as i64,
        noiseval,
        noise_type,
        gain: 0.0,
        maskthresh: 0.0,
    };
    let (conv, convw, convh) = match &params.filter {
        Some(kernel) => (
            kernel.values.as_ptr(),
            kernel.width as i64,
            kernel.height as i64,
        ),
        None => (ptr::null(), 0, 0),
    };
    let filter_code = match params.filter_type {
        FilterType::Conv => ffi::SEP_FILTER_CONV,
        FilterType::Matched => ffi::SEP_FILTER_MATCHED,
    };
    let thresh_type = if relative {
        ffi::SEP_THRESH_REL
    } else {
        ffi::SEP_THRESH_ABS
    };

    let objects = with_sep(|| {
        let mut raw: *mut ffi::sep_catalog = ptr::null_mut();
        // SAFETY: `image` points at validated buffers of w*h elements that
        // outlive the call; the kernel holds convw*convh floats; all integer
        // parameters were range-checked to fit `int`; `raw` is a valid
        // out-pointer initialised to NULL (early returns do not write it).
        // The globals are set and the call made under the lock.
        let code = unsafe {
            ffi::sep_set_extract_pixstack(params.pixstack);
            ffi::sep_set_sub_object_limit(params.sub_object_limit as c_int);
            take_errdetail();
            ffi::sep_extract(
                &image,
                params.thresh,
                thresh_type,
                params.minarea as c_int,
                conv,
                convw,
                convh,
                filter_code,
                params.deblend_nthresh as c_int,
                params.deblend_cont,
                c_int::from(params.clean),
                params.clean_param,
                &mut raw,
            )
        };
        let guard = CatalogGuard(raw);
        if code != ffi::RETURN_OK {
            // SAFETY: lock held, same thread as the failing call.
            return Err(unsafe { library_error(code) });
        }
        if guard.0.is_null() {
            return Err(SepError::InconsistentOutput {
                reason: "sep_extract returned OK and a null catalog".into(),
            });
        }
        // SAFETY: a non-null catalog returned by a successful sep_extract,
        // alive until `guard` drops at the end of this closure.
        unsafe { copy_catalog(&*guard.0, width, height) }
    })?;

    let segmentation = params.segmentation_map.then(|| {
        let mut map = vec![0i32; pixels];
        for (index, object) in objects.iter().enumerate() {
            // `index + 1 <= nobj <= c_int::MAX`.
            let id = index as i32 + 1;
            for &pixel in &object.pixels {
                map[pixel as usize] = id;
            }
        }
        map
    });

    Ok(Catalog {
        objects,
        segmentation,
        filter_applied,
        thresh_relative: relative,
    })
}

/// # Safety
/// `ptr` is null with `len == 0`, or points at `len` initialised `T`s that
/// stay alive for `'a`.
unsafe fn c_array<'a, T>(ptr: *const T, len: usize, name: &str) -> Result<&'a [T], SepError> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(SepError::InconsistentOutput {
            reason: format!("catalog field {name} is null for {len} objects"),
        });
    }
    // SAFETY: forwarded caller contract.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// Copies the C catalog into owned Rust values, checking every index.
///
/// # Safety
/// `catalog` was returned by a successful `sep_extract` on a `width x height`
/// image and has not been freed.
unsafe fn copy_catalog(
    catalog: &ffi::sep_catalog,
    width: usize,
    height: usize,
) -> Result<Vec<Object>, SepError> {
    let inconsistent = |reason: String| SepError::InconsistentOutput { reason };
    let nobj = usize::try_from(catalog.nobj)
        .map_err(|_| inconsistent(format!("nobj {}", catalog.nobj)))?;
    let pixels = width * height;
    let index = |value: i64, limit: usize, name: &str| -> Result<usize, SepError> {
        usize::try_from(value)
            .ok()
            .filter(|&v| v < limit)
            .ok_or_else(|| inconsistent(format!("{name} {value} outside 0..{limit}")))
    };
    // SAFETY (every `c_array` call below): each catalog array has `nobj`
    // elements (`extract.c:convert_to_catalog`) and lives as long as
    // `catalog`.
    let field = |ptr, name| unsafe { c_array(ptr, nobj, name) };
    let field_f64 = |ptr, name| unsafe { c_array::<f64>(ptr, nobj, name) };
    let field_i64 = |ptr, name| unsafe { c_array::<i64>(ptr, nobj, name) };
    let thresh = field(catalog.thresh, "thresh")?;
    let npix = field_i64(catalog.npix, "npix")?;
    let tnpix = field_i64(catalog.tnpix, "tnpix")?;
    let xmin = field_i64(catalog.xmin, "xmin")?;
    let xmax = field_i64(catalog.xmax, "xmax")?;
    let ymin = field_i64(catalog.ymin, "ymin")?;
    let ymax = field_i64(catalog.ymax, "ymax")?;
    let x = field_f64(catalog.x, "x")?;
    let y = field_f64(catalog.y, "y")?;
    let x2 = field_f64(catalog.x2, "x2")?;
    let y2 = field_f64(catalog.y2, "y2")?;
    let xy = field_f64(catalog.xy, "xy")?;
    let errx2 = field_f64(catalog.errx2, "errx2")?;
    let erry2 = field_f64(catalog.erry2, "erry2")?;
    let errxy = field_f64(catalog.errxy, "errxy")?;
    let a = field(catalog.a, "a")?;
    let b = field(catalog.b, "b")?;
    let theta = field(catalog.theta, "theta")?;
    let cxx = field(catalog.cxx, "cxx")?;
    let cyy = field(catalog.cyy, "cyy")?;
    let cxy = field(catalog.cxy, "cxy")?;
    let cflux = field(catalog.cflux, "cflux")?;
    let flux = field(catalog.flux, "flux")?;
    let cpeak = field(catalog.cpeak, "cpeak")?;
    let peak = field(catalog.peak, "peak")?;
    let xcpeak = field_i64(catalog.xcpeak, "xcpeak")?;
    let ycpeak = field_i64(catalog.ycpeak, "ycpeak")?;
    let xpeak = field_i64(catalog.xpeak, "xpeak")?;
    let ypeak = field_i64(catalog.ypeak, "ypeak")?;
    // SAFETY: as above.
    let flag = unsafe { c_array(catalog.flag, nobj, "flag")? };
    let pix = unsafe { c_array(catalog.pix as *const *mut i64, nobj, "pix")? };

    let mut objects = Vec::with_capacity(nobj);
    for i in 0..nobj {
        let count = usize::try_from(npix[i])
            .ok()
            .filter(|&n| (1..=pixels).contains(&n))
            .ok_or_else(|| inconsistent(format!("npix {} for object {i}", npix[i])))?;
        // SAFETY: `pix[i]` points into `objectspix` at `npix[i]` indices
        // (`convert_to_catalog`, include_pixels = 1).
        let footprint = unsafe { c_array(pix[i] as *const i64, count, "pix[i]")? };
        let footprint = footprint
            .iter()
            .map(|&p| index(p, pixels, "footprint pixel").map(|p| p as u32))
            .collect::<Result<Vec<u32>, _>>()?;
        objects.push(Object {
            thresh: thresh[i],
            npix: count,
            tnpix: usize::try_from(tnpix[i])
                .ok()
                .filter(|&n| n <= count)
                .ok_or_else(|| inconsistent(format!("tnpix {} for object {i}", tnpix[i])))?,
            xmin: index(xmin[i], width, "xmin")?,
            xmax: index(xmax[i], width, "xmax")?,
            ymin: index(ymin[i], height, "ymin")?,
            ymax: index(ymax[i], height, "ymax")?,
            x: x[i],
            y: y[i],
            x2: x2[i],
            y2: y2[i],
            xy: xy[i],
            errx2: errx2[i],
            erry2: erry2[i],
            errxy: errxy[i],
            a: a[i],
            b: b[i],
            theta: theta[i],
            cxx: cxx[i],
            cyy: cyy[i],
            cxy: cxy[i],
            cflux: cflux[i],
            flux: flux[i],
            cpeak: cpeak[i],
            peak: peak[i],
            xcpeak: index(xcpeak[i], width, "xcpeak")?,
            ycpeak: index(ycpeak[i], height, "ycpeak")?,
            xpeak: index(xpeak[i], width, "xpeak")?,
            ypeak: index(ypeak[i], height, "ypeak")?,
            // C `short` flags; reinterpret the bits (all defined flags < 0x100).
            flag: flag[i] as u16,
            pixels: footprint,
        });
    }
    Ok(objects)
}

#[cfg(test)]
mod tests;

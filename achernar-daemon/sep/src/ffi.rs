//! Raw declarations for the subset of `sep.h` (SEP 1.4.1) this crate calls.
//!
//! Every struct mirrors `vendor/sep-1.4.1/src/sep.h` field for field, in
//! declaration order, with the C types mapped as `int` -> `c_int`, `short` ->
//! `c_short`, `int64_t` -> `i64`, `size_t` -> `usize`, `float` -> `f32`,
//! `double` -> `f64`, pointers -> raw pointers. The `layout` test checks every
//! size, alignment and field offset against the C compiler's own
//! `sizeof`/`offsetof` (`csrc/layout_probe.c`), so a mismatch fails the build's
//! tests rather than corrupting memory.
//!
//! Nothing here is public. The safe API in `lib.rs` is the only caller.
#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_short, c_void};

pub const SEP_TBYTE: c_int = 11;
pub const SEP_TFLOAT: c_int = 42;

pub const SEP_NOISE_NONE: c_short = 0;
pub const SEP_NOISE_STDDEV: c_short = 1;

pub const SEP_THRESH_REL: c_int = 0;
pub const SEP_THRESH_ABS: c_int = 1;

pub const SEP_FILTER_CONV: c_int = 0;
pub const SEP_FILTER_MATCHED: c_int = 1;

/// `sepcore.h` (not installed by upstream): status codes returned by SEP.
pub const RETURN_OK: c_int = 0;

/// `sep_get_errmsg` writes at most 60 characters plus NUL (`util.c`).
pub const ERRMSG_LEN: usize = 61;
/// `sep_get_errdetail` copies the 512-byte thread-local `_errdetail_buffer`
/// (`util.c`, `DETAILSIZE`).
pub const ERRDETAIL_LEN: usize = 512;

/// `sep_image` (sep.h).
#[repr(C)]
pub struct sep_image {
    pub data: *const c_void,
    pub noise: *const c_void,
    pub mask: *const c_void,
    pub segmap: *const c_void,
    pub dtype: c_int,
    pub ndtype: c_int,
    pub mdtype: c_int,
    pub sdtype: c_int,
    pub segids: *mut i64,
    pub idcounts: *mut i64,
    pub numids: i64,
    pub w: i64,
    pub h: i64,
    pub noiseval: f64,
    pub noise_type: c_short,
    pub gain: f64,
    pub maskthresh: f64,
}

/// `sep_bkg` (sep.h). `back`/`sigma` are the filtered `nx*ny` mesh grids
/// (row-major, x fastest); `dback`/`dsigma` their y spline second derivatives.
#[repr(C)]
pub struct sep_bkg {
    pub w: i64,
    pub h: i64,
    pub bw: i64,
    pub bh: i64,
    pub nx: i64,
    pub ny: i64,
    pub n: i64,
    pub global: f32,
    pub globalrms: f32,
    pub back: *mut f32,
    pub dback: *mut f32,
    pub sigma: *mut f32,
    pub dsigma: *mut f32,
}

/// `sep_catalog` (sep.h). Every array has `nobj` elements; `pix[i]` points
/// into `objectspix` and holds `npix[i]` linear indices `x + w*y`.
#[repr(C)]
pub struct sep_catalog {
    pub nobj: c_int,
    pub thresh: *mut f32,
    pub npix: *mut i64,
    pub tnpix: *mut i64,
    pub xmin: *mut i64,
    pub xmax: *mut i64,
    pub ymin: *mut i64,
    pub ymax: *mut i64,
    pub x: *mut f64,
    pub y: *mut f64,
    pub x2: *mut f64,
    pub y2: *mut f64,
    pub xy: *mut f64,
    pub errx2: *mut f64,
    pub erry2: *mut f64,
    pub errxy: *mut f64,
    pub a: *mut f32,
    pub b: *mut f32,
    pub theta: *mut f32,
    pub cxx: *mut f32,
    pub cyy: *mut f32,
    pub cxy: *mut f32,
    pub cflux: *mut f32,
    pub flux: *mut f32,
    pub cpeak: *mut f32,
    pub peak: *mut f32,
    pub xcpeak: *mut i64,
    pub ycpeak: *mut i64,
    pub xpeak: *mut i64,
    pub ypeak: *mut i64,
    pub flag: *mut c_short,
    pub pix: *mut *mut i64,
    pub objectspix: *mut i64,
}

extern "C" {
    pub fn sep_background(
        image: *const sep_image,
        bw: i64,
        bh: i64,
        fw: i64,
        fh: i64,
        fthresh: f64,
        bkg: *mut *mut sep_bkg,
    ) -> c_int;
    pub fn sep_bkg_global(bkg: *const sep_bkg) -> f32;
    pub fn sep_bkg_globalrms(bkg: *const sep_bkg) -> f32;
    pub fn sep_bkg_line(bkg: *const sep_bkg, y: i64, line: *mut c_void, dtype: c_int) -> c_int;
    pub fn sep_bkg_rmsline(bkg: *const sep_bkg, y: i64, line: *mut c_void, dtype: c_int) -> c_int;
    pub fn sep_bkg_array(bkg: *const sep_bkg, arr: *mut c_void, dtype: c_int) -> c_int;
    pub fn sep_bkg_rmsarray(bkg: *const sep_bkg, arr: *mut c_void, dtype: c_int) -> c_int;
    pub fn sep_bkg_free(bkg: *mut sep_bkg);

    pub fn sep_extract(
        image: *const sep_image,
        thresh: f32,
        thresh_type: c_int,
        minarea: c_int,
        conv: *const f32,
        convw: i64,
        convh: i64,
        filter_type: c_int,
        deblend_nthresh: c_int,
        deblend_cont: f64,
        clean_flag: c_int,
        clean_param: f64,
        catalog: *mut *mut sep_catalog,
    ) -> c_int;
    pub fn sep_set_extract_pixstack(val: usize);
    pub fn sep_set_sub_object_limit(val: c_int);
    pub fn sep_catalog_free(catalog: *mut sep_catalog);

    pub static sep_version_string: *const c_char;
    pub fn sep_get_errmsg(status: c_int, errtext: *mut c_char);
    pub fn sep_get_errdetail(errtext: *mut c_char);
}

#[cfg(test)]
mod layout {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    extern "C" {
        static achernar_sep_layout_image: [usize; 18];
        static achernar_sep_layout_bkg: [usize; 14];
        static achernar_sep_layout_catalog: [usize; 34];
        static achernar_sep_layout_scalars: [usize; 8];
    }

    #[test]
    fn rust_mirrors_match_the_c_compilers_layout() {
        // SAFETY: plain `const size_t[]` arrays defined in layout_probe.c with
        // exactly these lengths; read-only.
        let (image, bkg, catalog, scalars) = unsafe {
            (
                achernar_sep_layout_image,
                achernar_sep_layout_bkg,
                achernar_sep_layout_catalog,
                achernar_sep_layout_scalars,
            )
        };
        let rust_image = [
            size_of::<sep_image>(),
            offset_of!(sep_image, data),
            offset_of!(sep_image, noise),
            offset_of!(sep_image, mask),
            offset_of!(sep_image, segmap),
            offset_of!(sep_image, dtype),
            offset_of!(sep_image, ndtype),
            offset_of!(sep_image, mdtype),
            offset_of!(sep_image, sdtype),
            offset_of!(sep_image, segids),
            offset_of!(sep_image, idcounts),
            offset_of!(sep_image, numids),
            offset_of!(sep_image, w),
            offset_of!(sep_image, h),
            offset_of!(sep_image, noiseval),
            offset_of!(sep_image, noise_type),
            offset_of!(sep_image, gain),
            offset_of!(sep_image, maskthresh),
        ];
        assert_eq!(rust_image, image, "sep_image size/offsets");
        let rust_bkg = [
            size_of::<sep_bkg>(),
            offset_of!(sep_bkg, w),
            offset_of!(sep_bkg, h),
            offset_of!(sep_bkg, bw),
            offset_of!(sep_bkg, bh),
            offset_of!(sep_bkg, nx),
            offset_of!(sep_bkg, ny),
            offset_of!(sep_bkg, n),
            offset_of!(sep_bkg, global),
            offset_of!(sep_bkg, globalrms),
            offset_of!(sep_bkg, back),
            offset_of!(sep_bkg, dback),
            offset_of!(sep_bkg, sigma),
            offset_of!(sep_bkg, dsigma),
        ];
        assert_eq!(rust_bkg, bkg, "sep_bkg size/offsets");
        let rust_catalog = [
            size_of::<sep_catalog>(),
            offset_of!(sep_catalog, nobj),
            offset_of!(sep_catalog, thresh),
            offset_of!(sep_catalog, npix),
            offset_of!(sep_catalog, tnpix),
            offset_of!(sep_catalog, xmin),
            offset_of!(sep_catalog, xmax),
            offset_of!(sep_catalog, ymin),
            offset_of!(sep_catalog, ymax),
            offset_of!(sep_catalog, x),
            offset_of!(sep_catalog, y),
            offset_of!(sep_catalog, x2),
            offset_of!(sep_catalog, y2),
            offset_of!(sep_catalog, xy),
            offset_of!(sep_catalog, errx2),
            offset_of!(sep_catalog, erry2),
            offset_of!(sep_catalog, errxy),
            offset_of!(sep_catalog, a),
            offset_of!(sep_catalog, b),
            offset_of!(sep_catalog, theta),
            offset_of!(sep_catalog, cxx),
            offset_of!(sep_catalog, cyy),
            offset_of!(sep_catalog, cxy),
            offset_of!(sep_catalog, cflux),
            offset_of!(sep_catalog, flux),
            offset_of!(sep_catalog, cpeak),
            offset_of!(sep_catalog, peak),
            offset_of!(sep_catalog, xcpeak),
            offset_of!(sep_catalog, ycpeak),
            offset_of!(sep_catalog, xpeak),
            offset_of!(sep_catalog, ypeak),
            offset_of!(sep_catalog, flag),
            offset_of!(sep_catalog, pix),
            offset_of!(sep_catalog, objectspix),
        ];
        assert_eq!(rust_catalog, catalog, "sep_catalog size/offsets");
        let rust_scalars = [
            align_of::<sep_image>(),
            align_of::<sep_bkg>(),
            align_of::<sep_catalog>(),
            size_of::<c_int>(),
            size_of::<c_short>(),
            size_of::<usize>(),
            size_of::<f32>(),
            size_of::<f64>(),
        ];
        assert_eq!(rust_scalars, scalars, "alignments and scalar sizes");
    }
}

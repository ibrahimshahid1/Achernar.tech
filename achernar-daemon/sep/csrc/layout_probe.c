/* Layout probe for the Rust #[repr(C)] mirrors in src/ffi.rs.
 *
 * The C compiler that builds SEP reports sizeof/offsetof of every field of
 * the public structs in sep.h (v1.4.1), in declaration order. The Rust layout
 * test compares them with core::mem::size_of / offset_of!. Not part of SEP;
 * nothing in the library references these symbols. */
#include <stddef.h>

#include "sep.h"

const size_t achernar_sep_layout_image[] = {
    sizeof(sep_image),
    offsetof(sep_image, data),
    offsetof(sep_image, noise),
    offsetof(sep_image, mask),
    offsetof(sep_image, segmap),
    offsetof(sep_image, dtype),
    offsetof(sep_image, ndtype),
    offsetof(sep_image, mdtype),
    offsetof(sep_image, sdtype),
    offsetof(sep_image, segids),
    offsetof(sep_image, idcounts),
    offsetof(sep_image, numids),
    offsetof(sep_image, w),
    offsetof(sep_image, h),
    offsetof(sep_image, noiseval),
    offsetof(sep_image, noise_type),
    offsetof(sep_image, gain),
    offsetof(sep_image, maskthresh),
};

const size_t achernar_sep_layout_bkg[] = {
    sizeof(sep_bkg),
    offsetof(sep_bkg, w),
    offsetof(sep_bkg, h),
    offsetof(sep_bkg, bw),
    offsetof(sep_bkg, bh),
    offsetof(sep_bkg, nx),
    offsetof(sep_bkg, ny),
    offsetof(sep_bkg, n),
    offsetof(sep_bkg, global),
    offsetof(sep_bkg, globalrms),
    offsetof(sep_bkg, back),
    offsetof(sep_bkg, dback),
    offsetof(sep_bkg, sigma),
    offsetof(sep_bkg, dsigma),
};

const size_t achernar_sep_layout_catalog[] = {
    sizeof(sep_catalog),
    offsetof(sep_catalog, nobj),
    offsetof(sep_catalog, thresh),
    offsetof(sep_catalog, npix),
    offsetof(sep_catalog, tnpix),
    offsetof(sep_catalog, xmin),
    offsetof(sep_catalog, xmax),
    offsetof(sep_catalog, ymin),
    offsetof(sep_catalog, ymax),
    offsetof(sep_catalog, x),
    offsetof(sep_catalog, y),
    offsetof(sep_catalog, x2),
    offsetof(sep_catalog, y2),
    offsetof(sep_catalog, xy),
    offsetof(sep_catalog, errx2),
    offsetof(sep_catalog, erry2),
    offsetof(sep_catalog, errxy),
    offsetof(sep_catalog, a),
    offsetof(sep_catalog, b),
    offsetof(sep_catalog, theta),
    offsetof(sep_catalog, cxx),
    offsetof(sep_catalog, cyy),
    offsetof(sep_catalog, cxy),
    offsetof(sep_catalog, cflux),
    offsetof(sep_catalog, flux),
    offsetof(sep_catalog, cpeak),
    offsetof(sep_catalog, peak),
    offsetof(sep_catalog, xcpeak),
    offsetof(sep_catalog, ycpeak),
    offsetof(sep_catalog, xpeak),
    offsetof(sep_catalog, ypeak),
    offsetof(sep_catalog, flag),
    offsetof(sep_catalog, pix),
    offsetof(sep_catalog, objectspix),
};

/* Struct alignments, then the scalar C types the extern declarations assume. */
const size_t achernar_sep_layout_scalars[] = {
    _Alignof(sep_image),
    _Alignof(sep_bkg),
    _Alignof(sep_catalog),
    sizeof(int),
    sizeof(short),
    sizeof(size_t),
    sizeof(float),
    sizeof(double),
};

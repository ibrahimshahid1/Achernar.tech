# achernar-sep

Safe Rust API over the vendored SEP 1.4.1 C library: spatial background
(`sep_background` plus its spline evaluators) and source extraction
(`sep_extract`). Used by `achernar-daemon/src/vision/sep_backend.rs`.

## Pin

- SEP 1.4.1, C core only, in `vendor/sep-1.4.1/` with upstream checksums,
  per-file sha256 and licenses (`vendor/sep-1.4.1/SOURCE.md`).
- One memory-only patch (`patches/0001-...`): `convert_to_catalog` allocated
  `cflux`/`flux` twice and leaked `8 * nobj` bytes per extraction. No
  numerical effect.
- Built by `build.rs` with the `cc` crate; no network, works with
  `cargo build --offline`. Flags and their reasons are documented in
  `build.rs` and recorded in `achernar_sep::BUILD_FLAGS`, including any
  `CFLAGS`-family environment variables `cc` appends and the compiler path.
  A value-unsafe flag in them (`-ffast-math`, `-Ofast`,
  `-ffp-contract=fast`, ...) fails the build.
- Changing the SEP version or the compile flags changes measurements. Re-run
  the parity test and regenerate the fixture with the reference wheel when
  doing so.

## Safety boundary

All `unsafe` code of the SEP integration lives in this crate (`src/ffi.rs`
declarations, `src/lib.rs` call sites). The public API takes slices and plain
parameters and returns owned values or a typed `SepError`:

- shapes, mask/noise lengths, kernel shape and every parameter that sizes a
  SEP allocation are validated before the call (SEP validates none of them);
- unmasked data must be finite (NaN is rejected, not passed through);
- every FFI call sequence, including the process-global pixel stack and
  sub-object limit and the thread-local error detail, runs under one
  process-wide mutex;
- `Background` frees its model on drop; catalogs are copied and freed with
  `sep_catalog_free` on every path (0 leaks under macOS `leaks --atExit`
  across the unit tests);
- `ffi::layout` checks every struct size, alignment and field offset against
  the C compiler (`csrc/layout_probe.c`).

## Parity with the Python reference

`tests/parity.rs` compares this crate with the PyPI sep 1.4.1 wheel on the
committed fixture `tests/fixtures/parity_v1` (regenerate or check with
`tests/fixtures/make_parity_fixture.py [--check]` in the pinned environment).
Tolerances are predeclared (run decisions D-006). On macOS arm64 with Apple
clang 17 the background map, RMS map, globals, catalog and segmentation map
are bit-identical to the wheel with `-ffp-contract=on`. With
`ACHERNAR_SEP_FP_CONTRACT=off` (experiment only), 27-30 % of map pixels differ
by up to 3.6e-7 relative and theta by up to 4e-6 rad, still within D-006.
Other platforms and compilers have not been measured.

## Licensing (owner review required)

The C core is LGPL-3.0-or-later (`src/overlap.h` BSD-3-Clause). This crate
links it statically into the daemon. Distributing a daemon binary therefore
carries LGPLv3 obligations, in particular section 4 (Combined Works): the
recipient must be able to relink the application with a modified version of
the library, for example by shipping the object files or by linking SEP as a
shared library instead. The repository itself declares MIT. This has not been
reviewed and must be decided by the owner before any binary is distributed.
This is not legal advice.

## Known upstream hazards (not patched)

- `sep_extract` frees uninitialised `arraybuffer`s if an early allocation
  fails (before `arraybuffer_init`); the allocation bounds here make that
  path unlikely, not impossible.
- `backhisto` casts `pixel / qscale + cste` to `int64_t` before testing the
  mask; out-of-range values are undefined in C (hardware-defined on arm64 and
  x86-64, and such bins are discarded). Masked non-finite values are replaced
  by 0 before the call; unmasked ones are rejected.
- Deblending assigns faint pixels with `rand_r` (`deblend.c`); POSIX does not
  fix its algorithm and MSVC builds map it to `rand()`, so deblended
  footprints are bit-reproducible only per C library.
- `globalrms` falls back to 1.0 when no mesh RMS is positive, and a frame with
  no supported mesh returns level 0 / RMS 1 with success. Callers check mesh
  support and RMS themselves (the daemon adapter does).
- The RMS map is a natural cubic spline of the mesh RMS values and can go
  negative between meshes whose RMS changes sharply; callers must check it
  before using it as a noise map.

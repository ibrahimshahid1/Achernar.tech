//! Compiles the vendored SEP 1.4.1 C core (eight translation units, as listed
//! in upstream `CMakeLists.txt`/`Makefile`) plus a layout probe into one static
//! library. No network access and no generated sources: everything compiled is
//! under `vendor/` or `csrc/`, so `cargo build --offline` works.
//!
//! Flags, and why (GCC/Clang; MSVC gets only the defines):
//! * `-O3`: the upstream Makefile level and the level CPython wheels build
//!   with. Forced so debug builds of the daemon do not run SEP at `-O0`.
//! * `-std=gnu11`: SEP needs C11 `_Thread_local`/`_Atomic` (upstream 1.2.0
//!   changelog), `M_PI`, POSIX `rand_r`, and `void*` arithmetic
//!   (`aperture.i`). Strict `-std=c11` would hide `M_PI`/`rand_r` on glibc.
//! * `-DSEP_VERSION_STRING="1.4.1"`: `util.c` otherwise embeds "1.3.6".
//! * `-D_USE_MATH_DEFINES=1`: as upstream `setup.py` (needed for `M_PI` on
//!   MSVC, harmless elsewhere).
//! * `-fmacro-prefix-map=<vendor dir>/=`: `QMALLOC`/`QCALLOC` (`sepcore.h`)
//!   `sprintf` a message containing `__FILE__` into a 160-byte stack buffer
//!   on allocation failure. With an absolute build path that message exceeds
//!   160 bytes (clang `-Wformat-overflow` reports 182-204 bytes for this
//!   tree), i.e. a stack overflow on the out-of-memory path. Mapping the
//!   prefix makes `__FILE__` `src/<file>.c`, exactly the strings found in the
//!   PyPI 1.4.1 wheel, and the warning disappears.
//! * `-ffp-contract=on`: contraction of `a*b+c` into one fused multiply-add
//!   *within a single expression* (ISO C `FP_CONTRACT` semantics). This is
//!   clang's default, and the PyPI sep 1.4.1 macOS arm64 wheel (the parity
//!   reference) was built with it: its disassembly has 316 FMA-family
//!   instructions (312 outside the Cython wrapper; 305 in a local build), and
//!   the per-function counts match a local `-O3` clang build of these
//!   files exactly for `backstat`, `backguess`, `filterback`,
//!   `bkg_line_flt_internal`, `convolve`, `deblend`, `clean` (differences only
//!   in `analyse` 23/17 and `makebackspline` 4/3, consistent with different
//!   unrolling by the wheel's older Apple clang). `off` would remove all of
//!   them and move results by rounding-level amounts away from the reference;
//!   GCC's GNU-mode default `fast` would contract *across* statements, which
//!   neither the reference nor ISO C does. Passing `on` explicitly pins the
//!   clang behaviour and makes GCC >= 14 match it (older GCC treats `on` as
//!   `off`). Evidence (`tests/parity.rs`, macOS arm64, Apple clang 17): with
//!   `on` the wheel's maps, globals, catalog and segmentation map are
//!   reproduced bit for bit; with `off` 27-30% of map pixels differ (by at
//!   most 3.6e-7 relative, inside decisions.md D-006).
//! * No `-ffast-math` (or any value-unsafe flag): SEP's histogram mode and
//!   clipping are order- and NaN-sensitive.
use std::env;
use std::path::PathBuf;

const SEP_SOURCES: [&str; 8] = [
    "analyse.c",
    "aperture.c",
    "background.c",
    "convolve.c",
    "deblend.c",
    "extract.c",
    "lutz.c",
    "util.c",
];

fn main() {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR"));
    let vendor = manifest_dir.join("vendor").join("sep-1.4.1");
    let src = vendor.join("src");

    let mut build = cc::Build::new();
    build
        .include(&src)
        .opt_level(3)
        .define("SEP_VERSION_STRING", "\"1.4.1\"")
        .define("_USE_MATH_DEFINES", "1");

    let msvc = build.get_compiler().is_like_msvc();
    // Recorded into the crate (`achernar_sep::BUILD_FLAGS`) for provenance;
    // the machine-specific prefix-map path is summarised.
    let mut recorded = vec![
        "-O3",
        "-DSEP_VERSION_STRING=\"1.4.1\"",
        "-D_USE_MATH_DEFINES=1",
    ];
    if !msvc {
        let contract = fp_contract_flag();
        build.flag("-std=gnu11");
        build.flag(contract);
        let mut prefix_map = std::ffi::OsString::from("-fmacro-prefix-map=");
        prefix_map.push(vendor.as_os_str());
        prefix_map.push("/=");
        build.flag(prefix_map);
        recorded.extend([
            "-std=gnu11",
            contract,
            "-fmacro-prefix-map=<vendor/sep-1.4.1>/=",
        ]);
    }
    println!(
        "cargo:rustc-env=ACHERNAR_SEP_BUILD_FLAGS={}",
        recorded.join(" ")
    );

    for file in SEP_SOURCES {
        build.file(src.join(file));
    }
    // sizeof/offsetof of the public structs, read only by the layout test.
    build.file(manifest_dir.join("csrc").join("layout_probe.c"));
    build.compile("achernar_sep_1_4_1");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc/layout_probe.c");
    println!("cargo:rerun-if-env-changed=ACHERNAR_SEP_FP_CONTRACT");
    for entry in std::fs::read_dir(&src).expect("vendored SEP sources are present") {
        let path = entry.expect("readable vendor dir").path();
        println!("cargo:rerun-if-changed={}", path.display());
    }
    if !msvc {
        // SEP calls libm (sqrt, pow, atan2, log); libSystem provides it on
        // macOS, glibc needs it named.
        println!("cargo:rustc-link-lib=m");
    }
}

/// `-ffp-contract=on` (see the module docs). `ACHERNAR_SEP_FP_CONTRACT=off`
/// exists only to reproduce the contraction experiment recorded in the
/// README; any other value is rejected so a typo cannot silently change
/// measurements.
fn fp_contract_flag() -> &'static str {
    match env::var("ACHERNAR_SEP_FP_CONTRACT").as_deref() {
        Err(_) | Ok("on") => "-ffp-contract=on",
        Ok("off") => "-ffp-contract=off",
        Ok(other) => panic!("ACHERNAR_SEP_FP_CONTRACT must be `on` or `off`, not {other:?}"),
    }
}

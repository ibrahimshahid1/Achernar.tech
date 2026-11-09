//! Cross-language parity: the vendored SEP 1.4.1 C core called through this
//! crate versus the PyPI sep 1.4.1 wheel, on identical float32 input, mask and
//! parameters (`tests/fixtures/make_parity_fixture.py`, fixture `parity_v1`).
//!
//! Tolerances are decisions.md D-006 of run 20261108T2233, declared before any
//! result was seen; they are not to be relaxed to make this pass:
//! * background and RMS maps: |d| <= 1e-5 * max(1, |ref|) per pixel;
//! * globalback / globalrms: |d| <= 1e-5 * max(1, |ref|);
//! * catalog: same object count, npix, xmin/xmax/ymin/ymax and flags;
//!   x, y |d| <= 1e-4 px; flux, a, b |d| <= 1e-5 * max(1, |ref|);
//!   theta |d| <= 1e-5 rad; segmentation map exactly equal.
//!
//! Observed residuals are printed (`cargo test -p achernar-sep --test parity
//! -- --nocapture`).
//!
//! # Deblended children and the C library (review B2)
//!
//! `deblend.c:gatherup` gives each parent pixel outside the children's cores
//! to one `SEP_OBJ_MERGED` child drawn with the C library's `rand_r`, so the
//! children's `npix`, flux and shape are exact only on the C library that made
//! the fixture (`meta.json`: `rust_target_os`, `c_library`). The rule for
//! other platforms was fixed from the source before any foreign-libc run of
//! this test, and is checked on *every* platform (including the fixture's,
//! where the exact checks run as well):
//! * same object count; every non-MERGED object, the maps and the globals
//!   exactly as D-006 above;
//! * MERGED children grouped into parent families (`deblend_families`); per
//!   family: the same member indices, the union of the footprints identical,
//!   total `npix` identical, total flux within 1e-5 * max(1, |ref|), the union
//!   of the members' flags identical;
//! * per child: the MERGED bit, and x, y within 1e-4 px (`analyse.c` robust
//!   branch keeps each child's pre-assignment barycentre);
//! * segmentation: exact on non-MERGED pixels; a family's pixels carry ids of
//!   that family.
//!
//! Off the fixture's platform the per-child npix/flux/shape checks are not
//! verifiable and the test says so on stdout; it does not skip.
//! `ACHERNAR_SEP_PARITY_ASSUME_FOREIGN_LIBC=1` forces that mode (used to
//! check the rule against a build with a different `rand_r`).

use std::collections::HashMap;
use std::path::PathBuf;

use achernar_sep::{
    extract, Background, BackgroundParams, ExtractParams, FilterType, Kernel, Noise,
};

const WIDTH: usize = 192;
const HEIGHT: usize = 160;

fn fixture(name: &str) -> Vec<u8> {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "fixtures",
        "parity_v1",
        name,
    ]
    .iter()
    .collect();
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn f32s(name: &str) -> Vec<f32> {
    let bytes = fixture(name);
    assert_eq!(bytes.len(), WIDTH * HEIGHT * 4, "{name} size");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn i32s(name: &str) -> Vec<i32> {
    let bytes = fixture(name);
    assert_eq!(bytes.len(), WIDTH * HEIGHT * 4, "{name} size");
    bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn tsv(name: &str) -> Vec<HashMap<String, f64>> {
    let text = String::from_utf8(fixture(name)).unwrap();
    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .unwrap()
        .split('\t')
        .map(str::to_owned)
        .collect();
    lines
        .map(|line| {
            header
                .iter()
                .cloned()
                .zip(line.split('\t').map(|v| v.parse::<f64>().unwrap()))
                .collect()
        })
        .collect()
}

fn globals() -> (f64, f64) {
    let text = String::from_utf8(fixture("globals.tsv")).unwrap();
    let value = |key: &str| -> f64 {
        text.lines()
            .find_map(|line| line.strip_prefix(&format!("{key}\t")))
            .unwrap()
            .parse()
            .unwrap()
    };
    (value("globalback"), value("globalrms"))
}

fn inputs() -> (Vec<f32>, Vec<u8>) {
    let mask = fixture("mask.u8");
    assert_eq!(mask.len(), WIDTH * HEIGHT);
    (f32s("input.f32"), mask)
}

fn background_params() -> BackgroundParams {
    BackgroundParams {
        bw: 32,
        bh: 32,
        fw: 3,
        fh: 3,
        fthresh: 0.0,
    }
}

fn extract_params() -> ExtractParams {
    ExtractParams {
        thresh: 5.0,
        minarea: 5,
        filter: Some(Kernel::sep_default()),
        filter_type: FilterType::Conv,
        deblend_nthresh: 32,
        deblend_cont: 0.005,
        clean: true,
        clean_param: 1.0,
        pixstack: achernar_sep::DEFAULT_PIXSTACK,
        sub_object_limit: achernar_sep::DEFAULT_SUB_OBJECT_LIMIT,
        segmentation_map: true,
    }
}

/// Residual exactly as the fixture script builds it: float32 `data - back`,
/// masked pixels 0.
fn residual(data: &[f32], back: &[f32], mask: &[u8]) -> Vec<f32> {
    data.iter()
        .zip(back)
        .zip(mask)
        .map(|((d, b), m)| if *m != 0 { 0.0 } else { d - b })
        .collect()
}

struct MapDiff {
    max_abs: f64,
    max_scaled: f64,
    identical: usize,
}

/// `max_scaled` is max over pixels of |d| / max(1, |ref|), the D-006 metric.
fn compare_map(ours: &[f32], reference: &[f32]) -> MapDiff {
    assert_eq!(ours.len(), reference.len());
    let mut diff = MapDiff {
        max_abs: 0.0,
        max_scaled: 0.0,
        identical: 0,
    };
    for (&o, &r) in ours.iter().zip(reference) {
        assert!(o.is_finite() && r.is_finite());
        let d = (o as f64 - r as f64).abs();
        diff.max_abs = diff.max_abs.max(d);
        diff.max_scaled = diff.max_scaled.max(d / (r as f64).abs().max(1.0));
        diff.identical += usize::from(o.to_bits() == r.to_bits());
    }
    diff
}

#[test]
fn background_maps_and_globals_match_the_python_wheel() {
    let (data, mask) = inputs();
    let background =
        Background::estimate(&data, WIDTH, HEIGHT, Some(&mask), &background_params()).unwrap();
    let back = compare_map(&background.back_array().unwrap(), &f32s("back.f32"));
    let rms = compare_map(&background.rms_array().unwrap(), &f32s("rms.f32"));
    let (ref_back, ref_rms) = globals();
    let d_global = (background.global_level() as f64 - ref_back).abs();
    let d_globalrms = (background.global_rms() as f64 - ref_rms).abs();
    println!("C build flags: {}", achernar_sep::BUILD_FLAGS);
    println!(
        "back: max|d| {:.3e}, max|d|/max(1,|ref|) {:.3e}, bit-identical {}/{}",
        back.max_abs,
        back.max_scaled,
        back.identical,
        WIDTH * HEIGHT
    );
    println!(
        "rms:  max|d| {:.3e}, max|d|/max(1,|ref|) {:.3e}, bit-identical {}/{}",
        rms.max_abs,
        rms.max_scaled,
        rms.identical,
        WIDTH * HEIGHT
    );
    println!("globalback |d| {d_global:.3e} (ref {ref_back}), globalrms |d| {d_globalrms:.3e} (ref {ref_rms})");
    assert!(back.max_scaled <= 1e-5, "background map exceeds D-006");
    assert!(rms.max_scaled <= 1e-5, "RMS map exceeds D-006");
    assert!(
        d_global <= 1e-5 * ref_back.abs().max(1.0),
        "globalback exceeds D-006"
    );
    assert!(
        d_globalrms <= 1e-5 * ref_rms.abs().max(1.0),
        "globalrms exceeds D-006"
    );
}

/// A top-level string field of meta.json (written one key per line).
fn meta_field(key: &str) -> String {
    let meta = String::from_utf8(fixture("meta.json")).unwrap();
    let prefix = format!("\"{key}\": \"");
    meta.lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix(&prefix)
                .and_then(|rest| rest.strip_suffix("\","))
                .map(str::to_owned)
        })
        .unwrap_or_else(|| panic!("meta.json records {key}"))
}

/// Whether this run is on the C library the fixture was generated with.
fn fixture_libc_matches() -> bool {
    meta_field("rust_target_os") == std::env::consts::OS
        && std::env::var_os("ACHERNAR_SEP_PARITY_ASSUME_FOREIGN_LIBC").is_none()
}

const MERGED: u16 = achernar_sep::flags::OBJ_MERGED;

/// The family rule of the module docs (all platforms).
fn check_merged_families(
    label: &str,
    catalog: &achernar_sep::Catalog,
    reference: &[HashMap<String, f64>],
    reference_segmap: &[i32],
) {
    let mut reference_pixels: Vec<Vec<u32>> = vec![Vec::new(); reference.len()];
    for (p, &id) in reference_segmap.iter().enumerate() {
        if id > 0 {
            reference_pixels[id as usize - 1].push(p as u32);
        }
    }
    let ref_merged: Vec<bool> = reference
        .iter()
        .map(|r| r["flag"] as u16 & MERGED != 0)
        .collect();
    let families = achernar_sep::deblend_families(
        reference_pixels
            .iter()
            .zip(&ref_merged)
            .map(|(p, &m)| (p.as_slice(), m)),
        WIDTH,
        HEIGHT,
    );
    let ours_families = achernar_sep::deblend_families(
        catalog
            .objects
            .iter()
            .map(|o| (o.pixels.as_slice(), o.has_flag(MERGED))),
        WIDTH,
        HEIGHT,
    );
    assert_eq!(ours_families, families, "{label}: deblend families");
    let mut family_of = vec![usize::MAX; reference.len()];
    for (f, members) in families.iter().enumerate() {
        let union = |pixels: &mut dyn Iterator<Item = u32>| {
            let mut all: Vec<u32> = pixels.collect();
            all.sort_unstable();
            all
        };
        let ours_union = union(
            &mut members
                .iter()
                .flat_map(|&i| catalog.objects[i].pixels.iter().copied()),
        );
        let ref_union = union(
            &mut members
                .iter()
                .flat_map(|&i| reference_pixels[i].iter().copied()),
        );
        assert_eq!(ours_union, ref_union, "{label}: family {f} pixel set");
        let npix: usize = members.iter().map(|&i| catalog.objects[i].npix).sum();
        let ref_npix: f64 = members.iter().map(|&i| reference[i]["npix"]).sum();
        assert_eq!(npix as f64, ref_npix, "{label}: family {f} total npix");
        let flux: f64 = members
            .iter()
            .map(|&i| catalog.objects[i].flux as f64)
            .sum();
        let ref_flux: f64 = members.iter().map(|&i| reference[i]["flux"]).sum();
        assert!(
            (flux - ref_flux).abs() <= 1e-5 * ref_flux.abs().max(1.0),
            "{label}: family {f} total flux {flux} vs {ref_flux}"
        );
        let flags = members
            .iter()
            .fold(0u16, |acc, &i| acc | catalog.objects[i].flag);
        let ref_flags = members
            .iter()
            .fold(0u16, |acc, &i| acc | reference[i]["flag"] as u16);
        assert_eq!(flags, ref_flags, "{label}: family {f} flag union");
        for &i in members {
            family_of[i] = f;
            let ours = &catalog.objects[i];
            assert!(ours.has_flag(MERGED), "{label}: object {i} MERGED bit");
            assert!(
                (ours.x - reference[i]["x"]).abs() <= 1e-4
                    && (ours.y - reference[i]["y"]).abs() <= 1e-4,
                "{label}: child {i} barycentre"
            );
        }
        println!(
            "{label}: family {f} members {members:?}: npix {npix}, total flux {flux:.6} (ref {ref_flux:.6})"
        );
    }
    let segmap = catalog.segmentation.as_ref().unwrap();
    for (p, (&ours, &r)) in segmap.iter().zip(reference_segmap).enumerate() {
        let ok = match r {
            0 => ours == 0,
            r if !ref_merged[r as usize - 1] => ours == r,
            r => ours > 0 && family_of[ours as usize - 1] == family_of[r as usize - 1],
        };
        assert!(
            ok,
            "{label}: segmentation at pixel {p}: ours {ours}, reference {r}"
        );
    }
}

fn check_catalog(label: &str, catalog: &achernar_sep::Catalog) {
    let exact = fixture_libc_matches();
    let reference = tsv("catalog.tsv");
    println!(
        "{label}: {} objects (reference {})",
        catalog.objects.len(),
        reference.len()
    );
    assert_eq!(
        catalog.objects.len(),
        reference.len(),
        "{label}: object count"
    );
    let (mut dx, mut dy, mut dflux, mut da, mut db, mut dtheta) =
        (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    let scaled = |ours: f64, r: f64| (ours - r).abs() / r.abs().max(1.0);
    let reference_segmap = i32s("segmap.i32");
    check_merged_families(label, catalog, &reference, &reference_segmap);
    if !exact {
        println!(
            "{label}: NOTE: fixture made with {} ({}); this run treats its C library as \
             different (target_os {}). Per-child npix/flux/shape parity of SEP_OBJ_MERGED \
             children is NOT verifiable here (deblend.c gatherup uses the C library's \
             rand_r). Checked instead: exact non-merged objects, per-family invariants, \
             child barycentres.",
            meta_field("c_library"),
            meta_field("rust_target_os"),
            std::env::consts::OS
        );
    }
    for (i, (ours, r)) in catalog.objects.iter().zip(&reference).enumerate() {
        if !exact && r["flag"] as u16 & MERGED != 0 {
            continue;
        }
        let int = |key: &str| r[key] as i64;
        assert_eq!(ours.npix as i64, int("npix"), "{label} object {i} npix");
        assert_eq!(ours.xmin as i64, int("xmin"), "{label} object {i} xmin");
        assert_eq!(ours.xmax as i64, int("xmax"), "{label} object {i} xmax");
        assert_eq!(ours.ymin as i64, int("ymin"), "{label} object {i} ymin");
        assert_eq!(ours.ymax as i64, int("ymax"), "{label} object {i} ymax");
        assert_eq!(ours.flag as i64, int("flag"), "{label} object {i} flag");
        dx = dx.max((ours.x - r["x"]).abs());
        dy = dy.max((ours.y - r["y"]).abs());
        dflux = dflux.max(scaled(ours.flux as f64, r["flux"]));
        da = da.max(scaled(ours.a as f64, r["a"]));
        db = db.max(scaled(ours.b as f64, r["b"]));
        dtheta = dtheta.max((ours.theta as f64 - r["theta"]).abs());
    }
    let segmap = catalog.segmentation.as_ref().unwrap();
    assert_eq!(segmap.len(), reference_segmap.len());
    // Off the fixture's C library only the family rule applies to merged
    // pixels (checked above); count exact mismatches elsewhere.
    let merged_id = |id: i32| id > 0 && reference[id as usize - 1]["flag"] as u16 & MERGED != 0;
    let segmap_mismatch = segmap
        .iter()
        .zip(&reference_segmap)
        .filter(|(a, b)| a != b && (exact || !merged_id(**b)))
        .count();
    println!(
        "{label}: max |dx| {dx:.3e} px, |dy| {dy:.3e} px, flux {dflux:.3e}, a {da:.3e}, b {db:.3e} (scaled), |dtheta| {dtheta:.3e} rad, segmap mismatches {segmap_mismatch}"
    );
    assert!(dx <= 1e-4 && dy <= 1e-4, "{label}: centroid exceeds D-006");
    assert!(dflux <= 1e-5, "{label}: flux exceeds D-006");
    assert!(da <= 1e-5 && db <= 1e-5, "{label}: a/b exceed D-006");
    assert!(dtheta <= 1e-5, "{label}: theta exceeds D-006");
    assert_eq!(segmap_mismatch, 0, "{label}: segmentation map differs");
    println!(
        "{label}: per-child exact checks on MERGED children: {}",
        if exact {
            "run (fixture C library)"
        } else {
            "NOT VERIFIABLE on this platform"
        }
    );
}

#[test]
fn extraction_on_the_reference_maps_matches_the_python_wheel() {
    // Isolates sep_extract: residual and noise from the wheel's own maps.
    let (data, mask) = inputs();
    let rms = f32s("rms.f32");
    let resid = residual(&data, &f32s("back.f32"), &mask);
    let catalog = extract(
        &resid,
        WIDTH,
        HEIGHT,
        Some(&mask),
        Noise::Array(&rms),
        &extract_params(),
    )
    .unwrap();
    check_catalog("extract(reference maps)", &catalog);
}

#[test]
fn end_to_end_background_and_extraction_match_the_python_wheel() {
    let (data, mask) = inputs();
    let background =
        Background::estimate(&data, WIDTH, HEIGHT, Some(&mask), &background_params()).unwrap();
    let rms = background.rms_array().unwrap();
    let resid = residual(&data, &background.back_array().unwrap(), &mask);
    let catalog = extract(
        &resid,
        WIDTH,
        HEIGHT,
        Some(&mask),
        Noise::Array(&rms),
        &extract_params(),
    )
    .unwrap();
    check_catalog("end-to-end", &catalog);
}

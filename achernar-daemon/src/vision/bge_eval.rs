//! Evaluation bridge (format `bge-eval-bridge-v1`, see `ml/eval/bge`).
//!
//! An ignored test that runs the production background dispatch, star
//! detection/gates/HFR and trailing code on frames written by the Python
//! benchmark, and writes one JSON result per case and backend. It never
//! touches hardware or the daemon's databases.
//!
//! ```text
//! ACHERNAR_BGE_EVAL_MANIFEST=<dir>/manifest.json ACHERNAR_BGE_EVAL_OUT=<out> \
//!   cargo test --release bge_eval_bridge -- --ignored --nocapture
//! ```
//!
//! BGE v2 (`docs/bge-v2-spec.md` §4): an optional `"star_gates":
//! {"blend_test", "regroup_deblended"}` beside `"settings"`, at the top level
//! and per case, merged the same way. A backend is `legacy_global_clip` or
//! `sep` (the merged settings as given) or a method id of §4, which forces
//! the backend, noise policy, blend test and regrouping:
//! `rust_legacy_global_clip` = `rust_legacy__g1`, `rust_legacy__g2`,
//! `rust_sep` = `rust_sep__n1g1`, `rust_sep__n2g1`, `rust_sep__n1g2`,
//! `rust_sep__n2g2`. Each record echoes the effective `star_gates` record
//! and `background.config` so the label can be verified.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::{json, Value};

use super::background_dispatch::analyze_background;
use super::background_model::{
    BackgroundBackend, BackgroundField, BackgroundSettings, NoisePolicy,
};
use super::quality::{
    estimate_quality_with_policy, BlendTest, FocusSourcePolicy, LumaFrame, StarGateSettings,
};
use super::trailing::estimate_trailing;
use crate::processing::debayer::AnalysisMasks;

fn read_f32(path: &Path, expected: usize) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() != expected * 4 {
        return Err(format!(
            "{}: {} bytes, expected {}",
            path.display(),
            bytes.len(),
            expected * 4
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_mask(path: &Path, expected: usize) -> Result<Vec<u8>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() != expected || bytes.iter().any(|v| *v > 1) {
        return Err(format!(
            "{}: not a {expected}-pixel 0/1 mask",
            path.display()
        ));
    }
    Ok(bytes)
}

/// `<stem><suffix>`; case ids may contain dots, so no `with_extension`.
fn with_suffix(stem: &Path, suffix: &str) -> PathBuf {
    let mut name = stem.as_os_str().to_os_string();
    name.push(suffix);
    PathBuf::from(name)
}

fn write_f32(path: &Path, values: &[f32]) -> std::io::Result<()> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes)
}

/// Merge JSON objects (`overlay` wins), so partial settings use defaults.
fn merge(base: &mut Value, overlay: &Value) {
    match (base, overlay) {
        // An absent override (`manifest["settings"]` missing) changes nothing.
        (_, Value::Null) => {}
        (Value::Object(base), Value::Object(overlay)) => {
            for (key, value) in overlay {
                merge(base.entry(key.clone()).or_insert(Value::Null), value);
            }
        }
        (base, overlay) => *base = overlay.clone(),
    }
}

fn frame_for(case: &Value, root: &Path) -> Result<LumaFrame, String> {
    let width = case["width"].as_u64().ok_or("width")? as usize;
    let height = case["height"].as_u64().ok_or("height")? as usize;
    let n = width * height;
    let pixels = read_f32(&root.join(case["pixels"].as_str().ok_or("pixels")?), n)?;
    let mut masks = AnalysisMasks::default();
    if let Some(declared) = case["masks"].as_object() {
        for (name, path) in declared {
            let mask = read_mask(&root.join(path.as_str().ok_or("mask path")?), n)?;
            match name.as_str() {
                "bad_pixel" => masks.bad_pixel = mask,
                "invalid" => masks.invalid = mask,
                "edge" => masks.edge = mask,
                "native_saturation" => masks.native_saturation = mask,
                other => return Err(format!("unknown mask {other}")),
            }
        }
    }
    Ok(LumaFrame {
        pixels,
        width,
        height,
        max_adu: 65535.0,
        saturation_adu: case["saturation_adu"].as_f64().map(|v| v as f32),
        masks,
        analysis_descriptor: None,
    })
}

/// A §4 method id: the backend it runs and the variant it forces.
struct MethodVariant {
    backend: BackgroundBackend,
    noise_policy: Option<NoisePolicy>,
    star_gates: StarGateSettings,
}

fn method_variant(id: &str) -> Option<MethodVariant> {
    let (g1, g2) = (
        StarGateSettings {
            blend_test: BlendTest::PeakCountV1,
            regroup_deblended: false,
        },
        StarGateSettings {
            blend_test: BlendTest::DipAnnulusV2,
            regroup_deblended: true,
        },
    );
    let (n1, n2) = (NoisePolicy::FrameWideV1, NoisePolicy::MaskedLocalV2);
    let (backend, noise_policy, star_gates) = match id {
        "rust_legacy_global_clip" | "rust_legacy__g1" => {
            (BackgroundBackend::LegacyGlobalClip, None, g1)
        }
        "rust_legacy__g2" => (BackgroundBackend::LegacyGlobalClip, None, g2),
        "rust_sep" | "rust_sep__n1g1" => (BackgroundBackend::Sep, Some(n1), g1),
        "rust_sep__n2g1" => (BackgroundBackend::Sep, Some(n2), g1),
        "rust_sep__n1g2" => (BackgroundBackend::Sep, Some(n1), g2),
        "rust_sep__n2g2" => (BackgroundBackend::Sep, Some(n2), g2),
        _ => return None,
    };
    Some(MethodVariant {
        backend,
        noise_policy,
        star_gates,
    })
}

/// Effective background settings and star gates of one case and backend.
fn case_settings(
    manifest: &Value,
    case: &Value,
    backend: &str,
) -> Result<(BackgroundSettings, StarGateSettings), String> {
    let mut settings = serde_json::to_value(BackgroundSettings::default()).unwrap();
    merge(&mut settings, &manifest["settings"]);
    merge(&mut settings, &case["settings"]);
    let mut settings = serde_json::from_value::<BackgroundSettings>(settings)
        .map_err(|error| format!("settings: {error}"))?;
    let mut gates = serde_json::to_value(StarGateSettings::default()).unwrap();
    merge(&mut gates, &manifest["star_gates"]);
    merge(&mut gates, &case["star_gates"]);
    let mut gates = serde_json::from_value::<StarGateSettings>(gates)
        .map_err(|error| format!("star_gates: {error}"))?;
    match backend {
        "legacy_global_clip" => settings.backend = BackgroundBackend::LegacyGlobalClip,
        "sep" => settings.backend = BackgroundBackend::Sep,
        other => {
            let variant =
                method_variant(other).ok_or_else(|| format!("unknown backend {other}"))?;
            settings.backend = variant.backend;
            if let Some(policy) = variant.noise_policy {
                settings.sep.noise_policy = policy;
            }
            gates = variant.star_gates;
        }
    }
    Ok((settings, gates))
}

fn write_u8(path: &Path, values: &[u8]) -> std::io::Result<()> {
    std::fs::write(path, values)
}

fn run_case(
    frame: &LumaFrame,
    settings: &BackgroundSettings,
    star_gates: &StarGateSettings,
    write_maps: Option<&Path>,
) -> Value {
    let started = Instant::now();
    let background = analyze_background(frame, settings);
    let background_ms = started.elapsed().as_secs_f64() * 1e3;

    let policy = FocusSourcePolicy::new(star_gates, settings.detection.accept_deblended_for_focus);
    let started = Instant::now();
    let quality = estimate_quality_with_policy(frame, &background, policy);
    let quality_ms = started.elapsed().as_secs_f64() * 1e3;

    let started = Instant::now();
    let trailing = estimate_trailing(frame, &background);
    let trailing_ms = started.elapsed().as_secs_f64() * 1e3;

    if let (
        Some(stem),
        Some(BackgroundField::Map {
            level,
            rms,
            invalid_noise,
            ..
        }),
    ) = (write_maps, background.field.as_ref())
    {
        let _ = write_f32(&with_suffix(stem, ".back.f32"), level);
        let _ = write_f32(&with_suffix(stem, ".rms.f32"), rms);
        // masked_local_v2's M_inv, when any pixel is locally invalid.
        if let Some(invalid) = invalid_noise {
            let _ = write_u8(&with_suffix(stem, ".invalid.u8"), invalid);
        }
    }

    let candidates: Vec<Value> = quality
        .candidates
        .iter()
        .map(|c| {
            json!({"x": c.x, "y": c.y, "xmin": c.min_x, "xmax": c.max_x, "ymin": c.min_y,
                   "ymax": c.max_y, "npix": c.area, "flux": c.flux, "peak": c.peak,
                   "deblended": c.deblended, "family_size": c.family_size})
        })
        .collect();
    let catalog = background.catalog.as_ref().map(|catalog| {
        catalog
            .objects
            .iter()
            .map(|o| {
                json!({"x": o.x, "y": o.y, "xmin": o.xmin, "xmax": o.xmax, "ymin": o.ymin,
                       "ymax": o.ymax, "npix": o.npix, "flux": o.flux,
                       "peak_residual": o.peak_residual, "peak_raw": o.peak_raw, "a": o.a,
                       "b": o.b, "theta": o.theta, "flags": o.flags, "deblended": o.deblended,
                       "truncated": o.truncated})
            })
            .collect::<Vec<_>>()
    });
    let accepted_stars: Vec<Value> = quality
        .used_stars
        .iter()
        .map(|s| json!({"x": s.x, "y": s.y, "hfr_px": s.hfr_px}))
        .collect();
    json!({
        "refinement": background.result.refinement,
        "star_gates": policy.config_json(),
        "star_gates_config_sha256": quality.star_gates_config_sha256,
        "gate_diagnostics": quality.gate_diagnostics,
        "accepted_stars": accepted_stars,
        "background": background.result,
        "catalog_error": background.catalog_error,
        "detection": quality.detection,
        "candidates": candidates,
        "catalog": catalog,
        "quality": {
            "status": quality.status,
            "rejection_reasons": quality.rejection_reasons,
            "hfr_px": quality.hfr_px,
            "hfd_px": quality.hfd_px,
            "fwhm_px": quality.fwhm_px,
            "star_count_detected": quality.star_count_detected,
            "star_count_used": quality.star_count_used,
            "star_rejections": quality.star_rejections,
            "gate_diagnostics": quality.gate_diagnostics,
            "star_gates_config_sha256": quality.star_gates_config_sha256,
            "saturation_gate": quality.saturation_gate,
            "used_stars": quality.used_stars.iter()
                .map(|s| json!({"x": s.x, "y": s.y, "hfr_px": s.hfr_px}))
                .collect::<Vec<_>>(),
        },
        "trailing": {
            "severity": trailing.severity,
            "sensor_pa_deg": trailing.sensor_pa_deg,
            "policy": trailing.policy,
            "unavailable_reason": trailing.unavailable_reason,
        },
        "timing_ms": {"background": background_ms, "quality": quality_ms, "trailing": trailing_ms},
        "timing_note": "available-host software timing of a debug or release test build, not a deployment target",
    })
}

#[test]
#[ignore = "evaluation bridge; run explicitly with ACHERNAR_BGE_EVAL_MANIFEST/OUT"]
fn bge_eval_bridge() {
    let manifest_path = PathBuf::from(
        std::env::var("ACHERNAR_BGE_EVAL_MANIFEST").expect("set ACHERNAR_BGE_EVAL_MANIFEST"),
    );
    let out =
        PathBuf::from(std::env::var("ACHERNAR_BGE_EVAL_OUT").expect("set ACHERNAR_BGE_EVAL_OUT"));
    std::fs::create_dir_all(&out).unwrap();
    let root = manifest_path.parent().unwrap().to_path_buf();
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).expect("manifest JSON");
    assert_eq!(
        manifest["schema"], "bge-eval-bridge-v1",
        "unsupported manifest schema"
    );
    let backends: Vec<String> = manifest["backends"]
        .as_array()
        .map(|b| {
            b.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_else(|| vec!["legacy_global_clip".into(), "sep".into()]);
    let write_maps = manifest["write_maps"].as_bool().unwrap_or(false);
    let mut failures = 0usize;

    for case in manifest["cases"].as_array().expect("cases") {
        let id = case["id"].as_str().expect("case id").to_string();
        let frame = frame_for(case, &root);
        for backend in &backends {
            let stem = out.join(format!("{id}__{backend}"));
            let result = match &frame {
                Err(error) => Err(error.clone()),
                Ok(frame) => match case_settings(&manifest, case, backend) {
                    Ok((settings, gates)) => {
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            run_case(
                                frame,
                                &settings,
                                &gates,
                                write_maps.then_some(stem.as_path()),
                            )
                        }))
                        .map_err(|_| "panic during analysis".to_string())
                    }
                    Err(error) => Err(error),
                },
            };
            let mut record = json!({"case": id, "backend": backend, "error": Value::Null});
            match result {
                Ok(body) => merge(&mut record, &body),
                Err(error) => {
                    failures += 1;
                    record["error"] = json!(error);
                }
            }
            std::fs::write(
                with_suffix(&stem, ".json"),
                serde_json::to_vec_pretty(&record).unwrap(),
            )
            .unwrap();
        }
    }
    eprintln!(
        "bge_eval_bridge: wrote results to {} ({failures} case errors)",
        out.display()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn method_ids_force_the_spec_variants_over_the_manifest() {
        let manifest = json!({
            "settings": {"sep": {"noise_policy": "frame_wide_v1", "bw": 32}},
            "star_gates": {"blend_test": "peak_count_v1"},
        });
        let case = json!({"star_gates": {"regroup_deblended": false}});
        let (settings, gates) = case_settings(&manifest, &case, "sep").unwrap();
        assert_eq!(settings.backend, BackgroundBackend::Sep);
        assert_eq!(settings.sep.noise_policy, NoisePolicy::FrameWideV1);
        assert_eq!(settings.sep.bw, 32);
        assert_eq!(
            gates,
            StarGateSettings {
                blend_test: BlendTest::PeakCountV1,
                regroup_deblended: false
            }
        );
        // Without overlays a plain backend runs the v2 defaults.
        let (settings, gates) = case_settings(&json!({}), &json!({}), "sep").unwrap();
        assert_eq!(settings.sep.noise_policy, NoisePolicy::MaskedLocalV2);
        assert_eq!(gates, StarGateSettings::default());

        for (id, backend, policy, blend, regroup) in [
            (
                "rust_sep",
                BackgroundBackend::Sep,
                NoisePolicy::FrameWideV1,
                BlendTest::PeakCountV1,
                false,
            ),
            (
                "rust_sep__n1g1",
                BackgroundBackend::Sep,
                NoisePolicy::FrameWideV1,
                BlendTest::PeakCountV1,
                false,
            ),
            (
                "rust_sep__n2g1",
                BackgroundBackend::Sep,
                NoisePolicy::MaskedLocalV2,
                BlendTest::PeakCountV1,
                false,
            ),
            (
                "rust_sep__n1g2",
                BackgroundBackend::Sep,
                NoisePolicy::FrameWideV1,
                BlendTest::DipAnnulusV2,
                true,
            ),
            (
                "rust_sep__n2g2",
                BackgroundBackend::Sep,
                NoisePolicy::MaskedLocalV2,
                BlendTest::DipAnnulusV2,
                true,
            ),
        ] {
            let (settings, gates) = case_settings(&manifest, &case, id).unwrap();
            assert_eq!(settings.backend, backend, "{id}");
            assert_eq!(settings.sep.noise_policy, policy, "{id}");
            assert_eq!(settings.sep.bw, 32, "{id}: other settings still merge");
            assert_eq!(
                (gates.blend_test, gates.regroup_deblended),
                (blend, regroup),
                "{id}"
            );
        }
        for (id, blend) in [
            ("rust_legacy_global_clip", BlendTest::PeakCountV1),
            ("rust_legacy__g1", BlendTest::PeakCountV1),
            ("rust_legacy__g2", BlendTest::DipAnnulusV2),
        ] {
            let (settings, gates) = case_settings(&json!({}), &json!({}), id).unwrap();
            assert_eq!(
                settings.backend,
                BackgroundBackend::LegacyGlobalClip,
                "{id}"
            );
            assert_eq!(gates.blend_test, blend, "{id}");
        }
        assert!(case_settings(&json!({}), &json!({}), "rust_sep__n3g1").is_err());
        assert!(case_settings(
            &json!({"star_gates": {"blend_test": "nope"}}),
            &json!({}),
            "sep"
        )
        .is_err());
    }
}

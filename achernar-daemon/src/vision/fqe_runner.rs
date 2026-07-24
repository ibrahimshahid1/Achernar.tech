//! FQE runner — Section 7.6.1 / 8.1.
//!
//! Real ONNX inference for the Focus Quality Estimator. Loads the exported
//! two-head model (`mu_log_hfr`, `log_var`) via ONNX Runtime and turns
//! normalized grayscale star crops into predicted HFRs (px) plus a log-HFR
//! uncertainty. The uncertainty is **calibrated only when** a sibling
//! `calibration.json` provides both a fitted per-star `sigma_scale` and a
//! post-pooling `frame_sigma_scale` (fit on validation groups by
//! `ml/train/calibrate_uncertainty.py`). `is_calibrated()` refers to the final
//! frame quantity published on the bus, so it is false when either scale is
//! missing or belongs to different model weights.
//!
//! The runner is SHADOW-MODE infrastructure: the classical curve-of-growth
//! measurement remains the control authority until the FQE passes the frozen
//! baselines in `ml/eval/baselines.py` (promotion gate, spec Section 7.10).

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;
use sha2::{Digest, Sha256};

use crate::error::InferenceError;

/// Per-crop FQE prediction. All widths are in pixels; uncertainty is reported in
/// the model's native log-HFR domain (and as a pixel-free log sigma).
#[derive(Debug, Clone, Copy)]
pub struct FqeOutput {
    pub hfr_px: f64,
    pub mu_log_hfr: f64,
    /// Aleatoric standard deviation of log(HFR) (the heteroscedastic head),
    /// multiplied by the fitted `sigma_scale` when calibration is present.
    pub sigma_log_hfr: f64,
}

/// Expected square crop side (px). The exported model is fixed to this.
pub const FQE_CROP_SIZE: usize = 64;

pub struct FqeRunner {
    // ONNX Runtime `Session::run` takes `&mut self`, so guard it with a Mutex to
    // keep the runner shareable behind an `Arc`.
    session: Option<Mutex<Session>>,
    /// Multiplier on each star's aleatoric sigma, fitted post-hoc.
    star_sigma_scale: f64,
    /// Multiplier on the final robustly pooled frame sigma.
    frame_sigma_scale: f64,
    calibrated: bool,
    /// Version directory name the model was loaded from (e.g. "1.1.0").
    model_version: Option<String>,
}

impl FqeRunner {
    /// Load the model at `model_path` if it exists. A missing path is not an
    /// error: the runner reports `is_loaded() == false` and `predict` returns
    /// `ModelNotFound`, which the caller maps to a null (shadow fields stay
    /// `None`) rather than fabricating a score.
    ///
    /// A `calibration.json` next to the model is read for `sigma_scale`.
    pub fn new(model_path: Option<String>) -> crate::error::Result<Self> {
        let mut star_sigma_scale = 1.0_f64;
        let mut frame_sigma_scale = 1.0_f64;
        let mut calibrated = false;
        let mut model_version = None;

        let session = match model_path.as_deref() {
            Some(path) if std::path::Path::new(path).exists() => {
                let session = Session::builder()
                    .map_err(onnx_err)?
                    .with_optimization_level(GraphOptimizationLevel::Level3)
                    .map_err(onnx_err)?
                    .with_intra_threads(1)
                    .map_err(onnx_err)?
                    .commit_from_file(path)
                    .map_err(onnx_err)?;

                let parent = Path::new(path).parent();
                model_version = parent
                    .and_then(|p| p.file_name())
                    .map(|n| n.to_string_lossy().to_string());
                if let Some((star_scale, frame_scale, ok)) = load_sigma_scales(Path::new(path)) {
                    star_sigma_scale = star_scale;
                    frame_sigma_scale = frame_scale;
                    calibrated = ok;
                }
                tracing::info!(
                    "FQE model loaded: {path} (version={:?}, star_sigma_scale={:.3}, \
                     frame_sigma_scale={:.3}, calibrated={})",
                    model_version,
                    star_sigma_scale,
                    frame_sigma_scale,
                    calibrated
                );
                Some(Mutex::new(session))
            }
            Some(path) => {
                tracing::warn!("FQE model path '{path}' does not exist; FQE inference disabled");
                None
            }
            None => None,
        };
        Ok(Self {
            session,
            star_sigma_scale,
            frame_sigma_scale,
            calibrated,
            model_version,
        })
    }

    /// Disabled runner (no model). `is_loaded() == false`, predictions error.
    pub fn disabled() -> Self {
        Self {
            session: None,
            star_sigma_scale: 1.0,
            frame_sigma_scale: 1.0,
            calibrated: false,
            model_version: None,
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.session.is_some()
    }

    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    pub fn model_version(&self) -> Option<&str> {
        self.model_version.as_deref()
    }

    /// Apply this after `aggregate_fqe`; calibration of individual stars does not
    /// imply calibrated coverage for their robustly pooled frame distribution.
    pub fn frame_sigma_scale(&self) -> f64 {
        self.frame_sigma_scale
    }

    /// Run inference on a single normalized grayscale crop (`FQE_CROP_SIZE` square,
    /// values in ~[0, 1] from `preprocess::robust_normalize`).
    pub fn predict(&self, crop: &ndarray::Array2<f32>) -> crate::error::Result<FqeOutput> {
        let mut outputs = self.predict_batch(std::slice::from_ref(crop))?;
        outputs
            .pop()
            .ok_or_else(|| InferenceError::OnnxError("empty batch output".to_string()).into())
    }

    /// Run inference on N crops in one ONNX Runtime call (the exported graph has
    /// a dynamic batch axis). One session run for a frame's stars instead of N —
    /// this is the batching the audit flagged as missing.
    pub fn predict_batch(
        &self,
        crops: &[ndarray::Array2<f32>],
    ) -> crate::error::Result<Vec<FqeOutput>> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| InferenceError::ModelNotFound("FQE model is not loaded".to_string()))?;
        if crops.is_empty() {
            return Ok(Vec::new());
        }
        for crop in crops {
            let (h, w) = (crop.shape()[0], crop.shape()[1]);
            if h != FQE_CROP_SIZE || w != FQE_CROP_SIZE {
                return Err(InferenceError::OnnxError(format!(
                    "FQE expects {FQE_CROP_SIZE}x{FQE_CROP_SIZE} crops, got {h}x{w}"
                ))
                .into());
            }
        }

        // (N, 1, H, W) NCHW input. We build the tensor from a (shape, &[f32])
        // tuple rather than an ndarray, because `ort` links its own ndarray
        // version (0.16) which differs from the daemon's (0.15); the tuple form
        // avoids that cross-version trait mismatch entirely.
        let n = crops.len();
        let mut data: Vec<f32> = Vec::with_capacity(n * FQE_CROP_SIZE * FQE_CROP_SIZE);
        for crop in crops {
            data.extend(crop.iter().copied());
        }
        let shape = [n, 1, FQE_CROP_SIZE, FQE_CROP_SIZE];
        let tensor = TensorRef::from_array_view((shape, data.as_slice())).map_err(onnx_err)?;

        let mut guard = session
            .lock()
            .map_err(|_| InferenceError::OnnxError("FQE session mutex poisoned".to_string()))?;
        let outputs = guard.run(ort::inputs![tensor]).map_err(onnx_err)?;

        let (_, mu) = outputs["mu_log_hfr"]
            .try_extract_tensor::<f32>()
            .map_err(onnx_err)?;
        let (_, log_var) = outputs["log_var"]
            .try_extract_tensor::<f32>()
            .map_err(onnx_err)?;
        if mu.len() < n || log_var.len() < n {
            return Err(InferenceError::OnnxError(format!(
                "FQE batch output shorter than input: mu={}, log_var={}, n={n}",
                mu.len(),
                log_var.len()
            ))
            .into());
        }

        let mut predictions = Vec::with_capacity(n);
        for i in 0..n {
            let mu_log_hfr = mu[i] as f64;
            let raw_log_var = log_var[i] as f64;
            if !mu_log_hfr.is_finite() || !raw_log_var.is_finite() {
                return Err(InferenceError::OnnxError(format!(
                    "FQE produced a non-finite output at batch index {i}"
                ))
                .into());
            }
            let lv = raw_log_var.clamp(-8.0, 8.0);
            let hfr_px = mu_log_hfr.exp();
            let sigma_log_hfr = (0.5 * lv).exp() * self.star_sigma_scale;
            if !hfr_px.is_finite() || hfr_px <= 0.0 || !sigma_log_hfr.is_finite() {
                return Err(InferenceError::OnnxError(format!(
                    "FQE output overflowed at batch index {i}"
                ))
                .into());
            }
            predictions.push(FqeOutput {
                hfr_px,
                mu_log_hfr,
                sigma_log_hfr,
            });
        }
        Ok(predictions)
    }
}

/// Pool per-star FQE predictions into a frame-level `(hfr_px, sigma_log_hfr)`.
/// Parity with `ml/models/fqe_losses.py::aggregate_stars`:
///   frame log-HFR = median(mu); uncertainty = sqrt(MAD(mu)^2 * 1.4826^2 +
///   mean(sigma_i^2) / n). Returns `None` for an empty input — never a
///   spurious 0.
pub fn aggregate_fqe(outputs: &[FqeOutput]) -> Option<(f64, f64)> {
    if outputs.is_empty() {
        return None;
    }
    let mut mus: Vec<f64> = outputs.iter().map(|o| o.mu_log_hfr).collect();
    mus.sort_by(|a, b| a.total_cmp(b));
    let median = interpolated_median_sorted(&mus);

    let mut deviations: Vec<f64> = mus.iter().map(|m| (m - median).abs()).collect();
    deviations.sort_by(|a, b| a.total_cmp(b));
    let between = interpolated_median_sorted(&deviations) * 1.4826;

    let mean_var = outputs
        .iter()
        .map(|o| o.sigma_log_hfr * o.sigma_log_hfr)
        .sum::<f64>()
        / outputs.len() as f64;
    let sigma_log = (between * between + mean_var / outputs.len() as f64).sqrt();

    Some((median.exp(), sigma_log))
}

/// `np.median` parity: mean of the two middle order statistics for even n.
fn interpolated_median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// Resolve the newest published FQE model under `<models_root>/focus_quality/`.
/// Picks the highest `major.minor.patch` directory containing an artifact whose
/// metadata explicitly marks it runtime-eligible for shadow inference. Reference
/// graphs and synthetic-only experiments stay loadable by an explicit path, but
/// are never auto-selected by the daemon.
pub fn resolve_latest_model(models_root: &str) -> Option<PathBuf> {
    let base = Path::new(models_root).join("focus_quality");
    let mut best: Option<((u64, u64, u64), PathBuf)> = None;
    for entry in std::fs::read_dir(&base).ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let mut parts = name.split('.').map(|p| p.parse::<u64>());
        let version = match (parts.next(), parts.next(), parts.next(), parts.next()) {
            (Some(Ok(a)), Some(Ok(b)), Some(Ok(c)), None) => (a, b, c),
            _ => continue,
        };
        let candidate = entry.path().join("model.onnx");
        if !candidate.exists() || !is_runtime_eligible(&entry.path()) {
            continue;
        }
        if best.as_ref().map(|(v, _)| version > *v).unwrap_or(true) {
            best = Some((version, candidate));
        }
    }
    best.map(|(_, path)| path)
}

fn is_runtime_eligible(model_dir: &Path) -> bool {
    let raw = match std::fs::read_to_string(model_dir.join("metadata.json")) {
        Ok(raw) => raw,
        Err(_) => return false,
    };
    let json: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(json) => json,
        Err(_) => return false,
    };
    let training = match json.get("training_info") {
        Some(training) => training,
        None => return false,
    };
    let expected_hash = json.get("sha256").and_then(serde_json::Value::as_str);
    let actual_hash = std::fs::read(model_dir.join("model.onnx"))
        .ok()
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)));
    expected_hash.is_some()
        && expected_hash == actual_hash.as_deref()
        && training.get("status").and_then(serde_json::Value::as_str) == Some("trained")
        && training
            .get("runtime_eligible")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
        && matches!(
            training
                .get("promotion_status")
                .and_then(serde_json::Value::as_str),
            Some("shadow" | "promoted")
        )
}

/// Read the per-star and post-pooling scales from `calibration.json`. Calibration
/// is accepted only when it names the same model hash as `metadata.json`; a
/// copied/stale file must not silently bless different weights. A valid legacy
/// star-only scale may still improve raw star sigmas, but the frame result is not
/// called calibrated until both scales are present.
fn load_sigma_scales(model_path: &Path) -> Option<(f64, f64, bool)> {
    let model_dir = model_path.parent()?;
    let raw = std::fs::read_to_string(model_dir.join("calibration.json")).ok()?;
    let json: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let metadata_raw = std::fs::read_to_string(model_dir.join("metadata.json")).ok()?;
    let metadata: serde_json::Value = serde_json::from_str(&metadata_raw).ok()?;
    let calibration_hash = json.get("model_sha256").and_then(serde_json::Value::as_str);
    let metadata_hash = metadata.get("sha256").and_then(serde_json::Value::as_str);
    let actual_hash = format!("{:x}", Sha256::digest(std::fs::read(model_path).ok()?));
    let hash_matches = calibration_hash == Some(actual_hash.as_str())
        && metadata_hash == Some(actual_hash.as_str());
    if !hash_matches {
        return Some((1.0, 1.0, false));
    }
    let star_scale = json
        .get("sigma_scale")
        .and_then(serde_json::Value::as_f64)
        .filter(|scale| scale.is_finite() && *scale > 0.0);
    let frame_scale = json
        .get("frame_sigma_scale")
        .and_then(serde_json::Value::as_f64)
        .filter(|scale| scale.is_finite() && *scale > 0.0);
    let calibrated = star_scale.is_some() && frame_scale.is_some();
    Some((
        star_scale.unwrap_or(1.0),
        frame_scale.unwrap_or(1.0),
        calibrated,
    ))
}

fn onnx_err(err: ort::Error) -> crate::error::AchernarError {
    InferenceError::OnnxError(err.to_string()).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_reports_not_loaded() {
        let runner = FqeRunner::new(Some("/nonexistent/model.onnx".to_string())).unwrap();
        assert!(!runner.is_loaded());
        assert!(!runner.is_calibrated());
        let crop = ndarray::Array2::<f32>::zeros((FQE_CROP_SIZE, FQE_CROP_SIZE));
        assert!(runner.predict(&crop).is_err());
    }

    #[test]
    fn aggregation_matches_python_aggregate_stars() {
        // mu = [ln 2, ln 3, ln 10] -> median ln 3; MAD = median(|mu - ln3|)
        //   |ln2-ln3| = 0.405465, |0|, |ln10-ln3| = 1.203973 -> MAD = 0.405465
        // between = 0.405465 * 1.4826 = 0.601143
        // sigmas = [0.1, 0.2, 0.3] -> mean var = (0.01+0.04+0.09)/3 = 0.0466667
        // sigma_log = sqrt(0.601143^2 + 0.0466667/3) = sqrt(0.376929) = 0.613945
        let outs = [
            FqeOutput {
                hfr_px: 2.0,
                mu_log_hfr: 2.0_f64.ln(),
                sigma_log_hfr: 0.1,
            },
            FqeOutput {
                hfr_px: 3.0,
                mu_log_hfr: 3.0_f64.ln(),
                sigma_log_hfr: 0.2,
            },
            FqeOutput {
                hfr_px: 10.0,
                mu_log_hfr: 10.0_f64.ln(),
                sigma_log_hfr: 0.3,
            },
        ];
        let (hfr, sigma) = aggregate_fqe(&outs).unwrap();
        assert!((hfr - 3.0).abs() < 1e-9, "median pooling: hfr={hfr}");
        assert!((sigma - 0.613945).abs() < 1e-4, "sigma={sigma}");
        assert!(aggregate_fqe(&[]).is_none());
    }

    #[test]
    fn resolves_highest_semver_model_dir() {
        let root = std::env::temp_dir().join(format!("fqe_resolve_{}", std::process::id()));
        let base = root.join("focus_quality");
        for version in ["1.0.0", "1.1.0", "0.9.9", "not-a-version"] {
            let dir = base.join(version);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.onnx"), b"stub").unwrap();
            std::fs::write(
                dir.join("metadata.json"),
                format!(
                    r#"{{"sha256":"{}","training_info":{{"status":"trained","runtime_eligible":true,"promotion_status":"shadow"}}}}"#,
                    format!("{:x}", Sha256::digest(b"stub"))
                ),
            )
            .unwrap();
        }
        // 2.0.0 exists but has no model.onnx -> must be skipped.
        std::fs::create_dir_all(base.join("2.0.0")).unwrap();

        let resolved = resolve_latest_model(root.to_str().unwrap()).unwrap();
        assert!(resolved.ends_with("focus_quality/1.1.0/model.onnx"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn resolver_skips_reference_and_synthetic_only_artifacts() {
        let root = std::env::temp_dir().join(format!("fqe_eligibility_{}", std::process::id()));
        let base = root.join("focus_quality");
        for (version, status, eligible) in [
            ("1.0.0", "reference", false),
            ("1.1.0", "trained", false),
            ("1.2.0", "trained", true),
        ] {
            let dir = base.join(version);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.onnx"), b"stub").unwrap();
            std::fs::write(
                dir.join("metadata.json"),
                format!(
                    r#"{{"sha256":"{}","training_info":{{"status":"{status}","runtime_eligible":{eligible},"promotion_status":"shadow"}}}}"#,
                    format!("{:x}", Sha256::digest(b"stub"))
                ),
            )
            .unwrap();
        }

        let resolved = resolve_latest_model(root.to_str().unwrap()).unwrap();
        assert!(resolved.ends_with("focus_quality/1.2.0/model.onnx"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn calibration_requires_the_matching_model_hash() {
        let dir = std::env::temp_dir().join(format!("fqe_calibration_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model_path = dir.join("model.onnx");
        std::fs::write(&model_path, b"model-a").unwrap();
        let model_hash = format!("{:x}", Sha256::digest(b"model-a"));
        std::fs::write(
            dir.join("metadata.json"),
            format!(r#"{{"sha256":"{model_hash}"}}"#),
        )
        .unwrap();
        std::fs::write(
            dir.join("calibration.json"),
            r#"{"sigma_scale":2.5,"model_sha256":"model-b"}"#,
        )
        .unwrap();
        assert_eq!(load_sigma_scales(&model_path), Some((1.0, 1.0, false)));

        std::fs::write(
            dir.join("calibration.json"),
            format!(
                r#"{{"sigma_scale":2.5,"frame_sigma_scale":1.25,"model_sha256":"{model_hash}"}}"#
            ),
        )
        .unwrap();
        assert_eq!(load_sigma_scales(&model_path), Some((2.5, 1.25, true)));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn checked_in_reference_model_has_the_runtime_contract() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/focus_quality/1.0.0/model.onnx");
        assert!(
            path.exists(),
            "checked-in FQE reference artifact is missing"
        );
        let runner = FqeRunner::new(Some(path.to_string_lossy().into_owned())).unwrap();
        assert!(runner.is_loaded());
        assert!(!runner.is_calibrated());
        let crops = vec![
            ndarray::Array2::<f32>::zeros((FQE_CROP_SIZE, FQE_CROP_SIZE)),
            ndarray::Array2::<f32>::ones((FQE_CROP_SIZE, FQE_CROP_SIZE)),
        ];
        let outputs = runner.predict_batch(&crops).unwrap();
        assert_eq!(outputs.len(), 2);
        assert!(outputs.iter().all(|output| output.hfr_px.is_finite()));
    }

    // Integration test against a real exported model. Set FQE_ONNX_MODEL to the
    // path of a `focus_quality_*.onnx` (see ml/export/export_quality_onnx.py).
    // Skips when unset so CI stays hermetic.
    #[test]
    fn runs_real_onnx_model_when_available() {
        let Ok(path) = std::env::var("FQE_ONNX_MODEL") else {
            eprintln!("FQE_ONNX_MODEL unset; skipping real-model inference test");
            return;
        };
        let runner = FqeRunner::new(Some(path)).unwrap();
        assert!(runner.is_loaded());
        // Deterministic row-major ramp so a NCHW/layout mismatch would change the
        // result — lets us cross-check numerically against Python onnxruntime.
        let mut crop = ndarray::Array2::<f32>::zeros((FQE_CROP_SIZE, FQE_CROP_SIZE));
        for y in 0..FQE_CROP_SIZE {
            for x in 0..FQE_CROP_SIZE {
                crop[[y, x]] =
                    (y * FQE_CROP_SIZE + x) as f32 / (FQE_CROP_SIZE * FQE_CROP_SIZE) as f32;
            }
        }
        let out = runner.predict(&crop).unwrap();
        eprintln!(
            "PARITY_RUST mu_log_hfr={:.6} hfr_px={:.6} sigma_log_hfr={:.6}",
            out.mu_log_hfr, out.hfr_px, out.sigma_log_hfr
        );
        assert!(out.hfr_px.is_finite() && out.hfr_px > 0.0);
        assert!(out.sigma_log_hfr.is_finite() && out.sigma_log_hfr > 0.0);
    }
}

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
//!
//! **Prototype mode** (`[vision.fqe] prototype = true`, owner authorization
//! F-006 of FQE run 20261111T0038): no bundle is qualified for the A03
//! analysis domain, so the vision loop never runs the FQE. The opt-in
//! prototype lets exactly one kind of bundle run anyway: one whose
//! `metadata.json` carries the provenance block `ml/pipelines/rebuild_fqe.py
//! --from-run ... --prototype-runtime-eligible` writes after the F-006
//! evidence gate passed ([`verify_prototype_bundle`]). Its predictions are
//! labelled prototype on the bus; classical HFR stays the authority. Without
//! the flag nothing here changes.

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

/// `training_info.prototype.gate.evidence_schema` written by
/// `ml/pipelines/rebuild_fqe.py` (`PROTOTYPE_GATE_SCHEMA`).
pub const PROTOTYPE_GATE_SCHEMA: &str = "fqe-prototype-gate-evidence-v1";

/// What a verified prototype bundle declares about itself (from its
/// `metadata.json`). Carried by the runner so every prediction can be
/// labelled with it.
#[derive(Debug, Clone, PartialEq)]
pub struct FqePrototype {
    /// sha256 of the graph bytes the session was built from; equals the
    /// metadata `sha256` (the gated `model.onnx`).
    pub model_sha256: String,
    /// sha256 of the F-006 gate evidence the export was refused without.
    pub evidence_sha256: String,
    /// The bundle's own domain statement (public archive data; not qualified
    /// for R50/Z114).
    pub domain: String,
    /// The frozen `fqe_dev_experiment` run the weights came from.
    pub run_name: Option<String>,
}

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
    /// `Some` only when prototype mode was requested AND the loaded bundle
    /// passed [`verify_prototype_bundle`].
    prototype: Option<FqePrototype>,
    /// Prototype mode was requested but no verified bundle is loaded: why.
    prototype_refusal: Option<String>,
}

impl FqeRunner {
    /// Load the model at `model_path` if it exists. A missing path is not an
    /// error: the runner reports `is_loaded() == false` and `predict` returns
    /// `ModelNotFound`, which the caller maps to a null (shadow fields stay
    /// `None`) rather than fabricating a score.
    ///
    /// A `calibration.json` next to the model is read for `sigma_scale`.
    pub fn new(model_path: Option<String>) -> crate::error::Result<Self> {
        Self::new_with_prototype(model_path, false)
    }

    /// [`FqeRunner::new`], plus `[vision.fqe] prototype`. With
    /// `prototype_mode` false this is exactly `new`. With it true the graph
    /// is read once, those bytes are what ONNX Runtime loads, and the same
    /// bytes must pass [`verify_prototype_graph`] against the bundle beside
    /// them (so a sibling such as `model.int8.onnx` cannot run under the
    /// prototype block of `model.onnx`); otherwise the runner records why
    /// ([`FqeRunner::prototype_refusal`]) and the vision loop keeps the FQE
    /// disabled.
    pub fn new_with_prototype(
        model_path: Option<String>,
        prototype_mode: bool,
    ) -> crate::error::Result<Self> {
        let mut star_sigma_scale = 1.0_f64;
        let mut frame_sigma_scale = 1.0_f64;
        let mut calibrated = false;
        let mut model_version = None;
        // Prototype mode only: the exact bytes the session was built from.
        let mut loaded_graph: Option<Vec<u8>> = None;

        let session = match model_path.as_deref() {
            Some(path) if std::path::Path::new(path).exists() => {
                let builder = Session::builder()
                    .map_err(onnx_err)?
                    .with_optimization_level(GraphOptimizationLevel::Level3)
                    .map_err(onnx_err)?
                    .with_intra_threads(1)
                    .map_err(onnx_err)?;
                let session = if prototype_mode {
                    let bytes = std::fs::read(path).map_err(|err| {
                        InferenceError::ModelNotFound(format!("{path} unreadable: {err}"))
                    })?;
                    let session = builder.commit_from_memory(&bytes).map_err(onnx_err)?;
                    loaded_graph = Some(bytes);
                    session
                } else {
                    builder.commit_from_file(path).map_err(onnx_err)?
                };

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
        let (prototype, prototype_refusal) = match (prototype_mode, model_path.as_deref()) {
            (false, _) => (None, None),
            (true, Some(path)) if session.is_some() => {
                match loaded_graph
                    .as_deref()
                    .map(|bytes| verify_prototype_graph(Path::new(path), bytes))
                {
                    Some(Ok(prototype)) => {
                        tracing::warn!(
                            "FQE PROTOTYPE enabled: {path} (sha256 {}, evidence {}); {}. \
                             Predictions are labelled prototype; classical HFR stays \
                             the authority.",
                            prototype.model_sha256,
                            prototype.evidence_sha256,
                            prototype.domain
                        );
                        (Some(prototype), None)
                    }
                    Some(Err(reason)) => {
                        tracing::warn!("FQE prototype refused for {path}: {reason}");
                        (None, Some(format!("{path}: {reason}")))
                    }
                    None => (None, Some(format!("{path}: graph bytes not retained"))),
                }
            }
            (true, Some(path)) => (None, Some(format!("{path}: model not loaded"))),
            (true, None) => (
                None,
                Some(
                    "no runtime-eligible shadow bundle with a verified prototype block was found"
                        .to_string(),
                ),
            ),
        };
        Ok(Self {
            session,
            star_sigma_scale,
            frame_sigma_scale,
            calibrated,
            model_version,
            prototype,
            prototype_refusal,
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
            prototype: None,
            prototype_refusal: None,
        }
    }

    /// Disabled runner in prototype mode: no model, and `reason` explains
    /// why on every observation.
    pub fn disabled_prototype(reason: String) -> Self {
        Self {
            prototype_refusal: Some(reason),
            ..Self::disabled()
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.session.is_some()
    }

    /// The verified prototype this runner serves, when prototype mode is on
    /// and the bundle passed [`verify_prototype_bundle`]. Only then does the
    /// vision loop run it.
    pub fn prototype(&self) -> Option<&FqePrototype> {
        self.prototype.as_ref().filter(|_| self.is_loaded())
    }

    /// Prototype mode was requested, but no verified prototype is loaded.
    pub fn prototype_refusal(&self) -> Option<&str> {
        self.prototype_refusal.as_deref()
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

/// Prototype mode's resolver: the highest `major.minor.patch` bundle under
/// `<models_root>/focus_quality/` that is runtime-eligible (as
/// [`resolve_latest_model`] requires) AND passes [`verify_prototype_bundle`].
/// Used only when `[vision.fqe] prototype = true`; a bundle without the
/// prototype block is never selected by it.
pub fn resolve_latest_prototype_model(models_root: &str) -> Option<PathBuf> {
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
        if !candidate.exists()
            || !is_runtime_eligible(&entry.path())
            || verify_prototype_bundle(&entry.path()).is_err()
        {
            continue;
        }
        if best.as_ref().map(|(v, _)| version > *v).unwrap_or(true) {
            best = Some((version, candidate));
        }
    }
    best.map(|(_, path)| path)
}

/// [`verify_prototype_graph`] for the bundle's own `model.onnx` (the graph
/// the prototype resolver selects).
pub fn verify_prototype_bundle(model_dir: &Path) -> Result<FqePrototype, String> {
    let model_path = model_dir.join("model.onnx");
    let model_bytes =
        std::fs::read(&model_path).map_err(|err| format!("model.onnx unreadable: {err}"))?;
    verify_prototype_graph(&model_path, &model_bytes)
}

/// Accept the graph `model_bytes` (read from `model_path`) as an F-006
/// prototype only when everything `rebuild_fqe.py --from-run ...
/// --prototype-runtime-eligible` writes after a passed evidence gate is
/// present in `model_path`'s directory and consistent:
///
/// * `sha256` equals the sha256 of `model_bytes`, the graph that runs. The
///   gate's evidence (ONNX parity included) is about the FP32 `model.onnx`
///   it hashed; any other file, e.g. the `model.int8.onnx` an export with
///   `--quantize` writes beside it, is refused;
/// * `training_info`: `status` "trained", `runtime_eligible` true,
///   `promotion_status` exactly "shadow", no `prototype_refused`;
/// * `training_info.prototype`: `runtime_eligible_override` true,
///   `real_multi_run_data` false, `ready_for_promotion_review` false,
///   `qualified_for` empty, a `domain` statement, and a `gate` with `status`
///   "passed", `evidence_schema` [`PROTOTYPE_GATE_SCHEMA`], a 64-hex
///   `evidence_sha256`, `test_partition_used` false and passed ONNX parity on
///   fixtures and development crops;
/// * the graph contract: `input_format.image` `[batch, 1, 64, 64]` and
///   `mu_log_hfr`/`log_var` outputs;
/// * `norm_params.json` declares the per-crop 1/99 percentile [0, 1]
///   normalization this daemon implements (`preprocess::robust_normalize`).
pub fn verify_prototype_graph(
    model_path: &Path,
    model_bytes: &[u8],
) -> Result<FqePrototype, String> {
    use serde_json::Value;

    let model_dir = model_path
        .parent()
        .ok_or("model path has no bundle directory")?;
    let read_json = |name: &str| -> Result<Value, String> {
        let raw = std::fs::read_to_string(model_dir.join(name))
            .map_err(|err| format!("{name} unreadable: {err}"))?;
        serde_json::from_str(&raw).map_err(|err| format!("{name} is not JSON: {err}"))
    };
    let metadata = read_json("metadata.json")?;
    let actual = format!("{:x}", Sha256::digest(model_bytes));
    if metadata.get("sha256").and_then(Value::as_str) != Some(actual.as_str()) {
        let name = model_path.file_name().map_or_else(
            || model_path.display().to_string(),
            |name| name.to_string_lossy().into_owned(),
        );
        return Err(format!(
            "metadata sha256 does not match {name}, the graph that would run"
        ));
    }
    let training = metadata.get("training_info").ok_or("no training_info")?;
    let text =
        |value: &Value, key: &str| value.get(key).and_then(Value::as_str).map(str::to_string);
    let flag = |value: &Value, key: &str| value.get(key).and_then(Value::as_bool);
    if text(training, "status").as_deref() != Some("trained")
        || flag(training, "runtime_eligible") != Some(true)
        || text(training, "promotion_status").as_deref() != Some("shadow")
    {
        return Err(
            "training_info is not trained / runtime_eligible / promotion_status \"shadow\"".into(),
        );
    }
    if training.get("prototype_refused").is_some() {
        return Err("training_info records a refused prototype export".into());
    }
    let prototype = training
        .get("prototype")
        .filter(|value| value.is_object())
        .ok_or("no training_info.prototype block (rebuild_fqe --prototype-runtime-eligible)")?;
    if flag(prototype, "runtime_eligible_override") != Some(true)
        || flag(prototype, "real_multi_run_data") != Some(false)
        || flag(prototype, "ready_for_promotion_review") != Some(false)
        || prototype
            .get("qualified_for")
            .and_then(Value::as_array)
            .map_or(true, |qualified| !qualified.is_empty())
    {
        return Err("prototype block does not declare an unqualified runtime override".into());
    }
    let domain = text(prototype, "domain")
        .filter(|domain| !domain.trim().is_empty())
        .ok_or("prototype block has no domain statement")?;
    let gate = prototype.get("gate").ok_or("prototype block has no gate")?;
    let evidence_sha256 = text(gate, "evidence_sha256")
        .filter(|sha| sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit()))
        .ok_or("prototype gate has no evidence_sha256")?;
    let parity_passed =
        |key: &str| gate.get(key).and_then(|parity| flag(parity, "passed")) == Some(true);
    if text(gate, "status").as_deref() != Some("passed")
        || text(gate, "evidence_schema").as_deref() != Some(PROTOTYPE_GATE_SCHEMA)
        || flag(gate, "test_partition_used") != Some(false)
        || !parity_passed("onnx_parity_fixtures")
        || !parity_passed("onnx_parity_dev_crops")
    {
        return Err(format!(
            "prototype gate is not a passed {PROTOTYPE_GATE_SCHEMA} gate with ONNX parity"
        ));
    }
    let input = metadata.pointer("/input_format/image").cloned();
    if input != Some(serde_json::json!(["batch", 1, 64, 64]))
        || metadata.pointer("/output_format/mu_log_hfr").is_none()
        || metadata.pointer("/output_format/log_var").is_none()
    {
        return Err("graph contract is not image [batch,1,64,64] -> mu_log_hfr, log_var".into());
    }
    let norm = read_json("norm_params.json")?;
    let number = |key: &str| norm.get(key).and_then(Value::as_f64);
    let scale_to: Option<Vec<f64>> = norm
        .get("scale_to")
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(Value::as_f64).collect());
    if text(&norm, "scheme").as_deref() != Some("per_crop_percentile")
        || number("p_lo") != Some(1.0)
        || number("p_hi") != Some(99.0)
        || scale_to.as_deref() != Some(&[0.0, 1.0][..])
    {
        return Err("norm_params.json is not the per-crop 1/99 percentile [0, 1] contract".into());
    }
    Ok(FqePrototype {
        model_sha256: actual,
        evidence_sha256,
        domain,
        run_name: text(prototype, "run_name"),
    })
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

/// Test fixture: `<root>/focus_quality/<version>/` holding a copy of the
/// checked-in random-weight `models/focus_quality/1.0.0/model.onnx` plus the
/// files `rebuild_fqe.py --from-run ... --prototype-runtime-eligible` writes
/// after a passed F-006 gate (metadata.json with the `training_info.prototype`
/// block, norm_params.json, and the uncalibrated public calibration.json).
/// `edit` changes the metadata before it is written. Returns the model path.
/// Random weights: the predictions are finite but meaningless.
#[cfg(test)]
pub(crate) fn write_test_prototype_bundle(
    root: &Path,
    version: &str,
    edit: impl FnOnce(&mut serde_json::Value),
) -> PathBuf {
    let reference =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/focus_quality/1.0.0/model.onnx");
    write_test_prototype_bundle_from(root, version, &reference, edit)
}

/// [`write_test_prototype_bundle`] with another ONNX graph as `model.onnx`.
#[cfg(test)]
pub(crate) fn write_test_prototype_bundle_from(
    root: &Path,
    version: &str,
    reference: &Path,
    edit: impl FnOnce(&mut serde_json::Value),
) -> PathBuf {
    let dir = root.join("focus_quality").join(version);
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("model.onnx");
    std::fs::copy(reference, &model).unwrap();
    let sha256 = format!("{:x}", Sha256::digest(std::fs::read(&model).unwrap()));
    let domain = "public archive development data (stellar_sim_v1 synthetic reference / \
                  hst_acs_corpus_v1 SEP measurement reference); not qualified for R50/Z114";
    let parity = serde_json::json!({"max_abs_delta_mu": 0.0, "tolerance": 1e-4, "passed": true});
    let gate = serde_json::json!({
        "status": "passed",
        "evidence_schema": PROTOTYPE_GATE_SCHEMA,
        "evidence_name": "gate_evidence.json",
        "evidence_sha256": "ab".repeat(32),
        "population": {"source_id": "stellar_sim_v1", "partition": "val"},
        "test_partition_used": false,
        "onnx_parity_fixtures": parity,
        "onnx_parity_dev_crops": parity,
    });
    let prototype = serde_json::json!({
        "runtime_eligible_override": true,
        "authorization": "daemon test fixture (shape of rebuild_fqe F-006 block)",
        "note": "random weights; integration only",
        "domain": domain,
        "data_roles": ["synthetic_reference"],
        "source_ids": ["stellar_sim_v1"],
        "contract_sha256": "0".repeat(64),
        "qualified_for": [],
        "not_qualified_for": ["R50", "Z114", "autofocus control authority", "promotion review"],
        "real_multi_run_data": false,
        "ready_for_promotion_review": false,
        "run_name": "fixture-s0",
        "gate": gate,
    });
    let mut metadata = serde_json::json!({
        "model_name": "focus_quality",
        "version": version,
        "task": "focus_quality",
        "framework": "onnx",
        "input_format": {"image": ["batch", 1, 64, 64]},
        "output_format": {"mu_log_hfr": ["batch"], "log_var": ["batch"]},
        "sha256": sha256,
        "training_info": {
            "status": "trained",
            "promotion_status": "shadow",
            "runtime_eligible": true,
            "preprocessing": "per-crop robust_normalize p1/p99 -> [0,1] (norm_params.json)",
            "domain": domain,
            "prototype": prototype,
        },
    });
    edit(&mut metadata);
    std::fs::write(dir.join("metadata.json"), metadata.to_string()).unwrap();
    std::fs::write(
        dir.join("norm_params.json"),
        serde_json::json!({
            "scheme": "per_crop_percentile",
            "p_lo": 1.0,
            "p_hi": 99.0,
            "scale_to": [0.0, 1.0],
            "spec_section": "7.6.1"
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.join("calibration.json"),
        serde_json::json!({
            "sigma_scale": null,
            "frame_sigma_scale": null,
            "model_sha256": sha256,
            "model_filename": "model.onnx"
        })
        .to_string(),
    )
    .unwrap();
    model
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

    fn scratch_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("fqe_{tag}_{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        root
    }

    fn drop_prototype_block(metadata: &mut serde_json::Value) {
        metadata["training_info"]
            .as_object_mut()
            .unwrap()
            .remove("prototype");
    }

    /// The fixture shaped like a `rebuild_fqe --prototype-runtime-eligible`
    /// export is accepted; every way of not being one is refused.
    #[test]
    fn prototype_verification_accepts_only_a_passed_f006_prototype_block() {
        use serde_json::{json, Value};
        let root = scratch_root("prototype_verify");
        let model = write_test_prototype_bundle(&root, "1.1.0", |_| {});
        let prototype = verify_prototype_bundle(model.parent().unwrap()).unwrap();
        assert_eq!(prototype.run_name.as_deref(), Some("fixture-s0"));
        assert_eq!(prototype.evidence_sha256, "ab".repeat(32));
        assert!(prototype.domain.contains("not qualified for R50/Z114"));
        assert_eq!(
            prototype.model_sha256,
            format!("{:x}", Sha256::digest(std::fs::read(&model).unwrap()))
        );

        let refusals: [(&str, fn(&mut Value)); 16] = [
            (
                "no prototype block: no training_info.prototype block",
                drop_prototype_block,
            ),
            (
                "prototype not an object: no training_info.prototype block",
                |m| m["training_info"]["prototype"] = json!(true),
            ),
            ("promoted: promotion_status", |m| {
                m["training_info"]["promotion_status"] = json!("promoted")
            }),
            ("not runtime eligible: runtime_eligible", |m| {
                m["training_info"]["runtime_eligible"] = json!(false)
            }),
            ("reference status: not trained", |m| {
                m["training_info"]["status"] = json!("reference")
            }),
            ("refused export: refused prototype export", |m| {
                m["training_info"]["prototype_refused"] = json!({"status": "refused"})
            }),
            ("no override: unqualified runtime override", |m| {
                m["training_info"]["prototype"]["runtime_eligible_override"] = json!(false)
            }),
            ("claims qualification: unqualified runtime override", |m| {
                m["training_info"]["prototype"]["qualified_for"] = json!(["R50"])
            }),
            ("promotion evidence: unqualified runtime override", |m| {
                m["training_info"]["prototype"]["ready_for_promotion_review"] = json!(true)
            }),
            ("no domain: no domain statement", |m| {
                m["training_info"]["prototype"]["domain"] = json!(" ")
            }),
            (
                "gate not passed: passed fqe-prototype-gate-evidence-v1 gate",
                |m| m["training_info"]["prototype"]["gate"]["status"] = json!("refused"),
            ),
            (
                "other evidence schema: passed fqe-prototype-gate-evidence-v1 gate",
                |m| m["training_info"]["prototype"]["gate"]["evidence_schema"] = json!("v0"),
            ),
            (
                "test partition used: passed fqe-prototype-gate-evidence-v1 gate",
                |m| m["training_info"]["prototype"]["gate"]["test_partition_used"] = json!(true),
            ),
            ("dev-crop parity failed: ONNX parity", |m| {
                m["training_info"]["prototype"]["gate"]["onnx_parity_dev_crops"]["passed"] =
                    json!(false)
            }),
            ("graph contract: graph contract", |m| {
                m["input_format"]["image"] = json!(["batch", 3, 64, 64])
            }),
            ("stale metadata hash: does not match model.onnx", |m| {
                m["sha256"] = json!("0".repeat(64))
            }),
        ];
        for (index, (case, edit)) in refusals.into_iter().enumerate() {
            let (case, expected) = case.split_once(": ").unwrap();
            let model = write_test_prototype_bundle(&root, &format!("2.0.{index}"), edit);
            let error = verify_prototype_bundle(model.parent().unwrap()).unwrap_err();
            assert!(error.contains(expected), "{case}: {error}");
        }

        // A normalization other than the per-crop 1/99 [0, 1] contract.
        let model = write_test_prototype_bundle(&root, "3.0.0", |_| {});
        std::fs::write(
            model.parent().unwrap().join("norm_params.json"),
            r#"{"scheme":"per_crop_percentile","p_lo":0.5,"p_hi":99.5,"scale_to":[0.0,1.0]}"#,
        )
        .unwrap();
        assert!(verify_prototype_bundle(model.parent().unwrap())
            .unwrap_err()
            .contains("norm_params"));
        std::fs::remove_dir_all(&root).ok();
    }

    /// The ordinary resolver and `FqeRunner::new` are unchanged; only
    /// prototype mode with a verified block yields a prototype runner.
    #[test]
    fn prototype_mode_needs_both_the_flag_and_a_verified_bundle() {
        let root = scratch_root("prototype_mode");
        let prototype_model = write_test_prototype_bundle(&root, "1.1.0", |_| {});
        let plain_model = write_test_prototype_bundle(&root, "1.2.0", drop_prototype_block);
        let root_str = root.to_str().unwrap();
        assert!(resolve_latest_model(root_str)
            .unwrap()
            .ends_with("focus_quality/1.2.0/model.onnx"));
        assert!(resolve_latest_prototype_model(root_str)
            .unwrap()
            .ends_with("focus_quality/1.1.0/model.onnx"));

        let path = |model: &PathBuf| Some(model.to_string_lossy().into_owned());
        let flag_off = FqeRunner::new(path(&prototype_model)).unwrap();
        assert!(flag_off.is_loaded());
        assert!(flag_off.prototype().is_none());
        assert!(flag_off.prototype_refusal().is_none());

        let prototype = FqeRunner::new_with_prototype(path(&prototype_model), true).unwrap();
        assert!(prototype.is_loaded());
        assert_eq!(prototype.model_version(), Some("1.1.0"));
        assert!(
            !prototype.is_calibrated(),
            "public bundles ship an uncalibrated sigma"
        );
        assert_eq!(
            prototype.prototype().unwrap().run_name.as_deref(),
            Some("fixture-s0")
        );
        assert!(prototype.prototype_refusal().is_none());

        let refused = FqeRunner::new_with_prototype(path(&plain_model), true).unwrap();
        assert!(refused.is_loaded());
        assert!(refused.prototype().is_none());
        assert!(refused
            .prototype_refusal()
            .unwrap()
            .contains("no training_info.prototype block"));

        // A different graph beside the verified one (as `--quantize` writes
        // `model.int8.onnx`), configured explicitly: it loads, but it is not
        // the graph the prototype block's hash and gate describe, so it is
        // refused. Stand-in: model.onnx with a ModelProto doc_string (field
        // 6) appended, a valid ONNX file with other bytes.
        let bundle = prototype_model.parent().unwrap();
        let mut other_graph = std::fs::read(&prototype_model).unwrap();
        let doc = b"int8 stand-in";
        other_graph.extend_from_slice(&[0x32, doc.len() as u8]);
        other_graph.extend_from_slice(doc);
        let int8 = bundle.join("model.int8.onnx");
        std::fs::write(&int8, &other_graph).unwrap();
        let int8_runner = FqeRunner::new_with_prototype(path(&int8), true).unwrap();
        assert!(
            int8_runner.is_loaded(),
            "the stand-in must be a valid graph"
        );
        assert!(int8_runner.prototype().is_none());
        let reason = int8_runner.prototype_refusal().unwrap();
        assert!(
            reason.contains("metadata sha256 does not match model.int8.onnx"),
            "{reason}"
        );
        // The binding is to the bytes, not the file name: an identical copy
        // is the gated graph.
        let copy = bundle.join("model.copy.onnx");
        std::fs::copy(&prototype_model, &copy).unwrap();
        let copied = FqeRunner::new_with_prototype(path(&copy), true).unwrap();
        assert_eq!(
            copied.prototype().map(|p| p.model_sha256.clone()),
            prototype.prototype().map(|p| p.model_sha256.clone())
        );
        // The resolver still selects the bundle's own model.onnx.
        assert!(resolve_latest_prototype_model(root_str)
            .unwrap()
            .ends_with("focus_quality/1.1.0/model.onnx"));

        let nothing = FqeRunner::new_with_prototype(None, true).unwrap();
        assert!(!nothing.is_loaded() && nothing.prototype().is_none());
        assert!(nothing.prototype_refusal().is_some());
        assert!(FqeRunner::disabled_prototype("why".into())
            .prototype()
            .is_none());

        std::fs::remove_dir_all(prototype_model.parent().unwrap()).unwrap();
        assert!(resolve_latest_prototype_model(root_str).is_none());
        std::fs::remove_dir_all(&root).ok();
    }
}

//! Model registry — Section 6.7.
//!
//! Discovers versioned model bundles under a base directory, selects the latest
//! semver per model, and verifies artifact integrity (SHA-256) before the daemon
//! loads anything. This generalizes the resolution/verification logic that
//! `vision::fqe_runner` already established (highest-semver dir, lowercase-hex
//! SHA-256 over `model.onnx`, `training_info.runtime_eligible`) so every ONNX
//! model shares one loader instead of copy-pasting the checks per subsystem.
//!
//! Bundles are laid out as `<base>/<model_name>/<version>/{metadata.json,
//! model.onnx, ...}`. Metadata schemas differ by model family — across the
//! current bundles only `model_name`, `version`, `task`, and `sha256` are
//! universal (focus_quality carries `framework`/`input_format`, while
//! focus_offset/cloud_classifier/trailing_detector carry `architecture`/
//! `training`/`export`). So the scan types only that common core and preserves
//! every other key verbatim in `extra`. A single malformed or partial bundle is
//! skipped and logged; it never aborts the scan of the others.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::error::InferenceError;

/// The subset of `metadata.json` shared by every bundle, plus a passthrough for
/// family-specific keys. Non-universal fields are `Option`/`extra` on purpose:
/// a missing optional key must never fail the scan of an otherwise-valid bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_name: String,
    pub version: String,
    pub task: String,
    pub sha256: String,
    #[serde(default)]
    pub created_at: Option<String>,
    /// Neural bundles (e.g. FQE) carry `training_info`; tabular/tree bundles
    /// carry `training`; classical bundles carry neither. Kept typed because the
    /// runtime-eligibility gate reads it.
    #[serde(default)]
    pub training_info: Option<serde_json::Value>,
    /// All remaining, family-specific keys (input_format, architecture, export,
    /// training, input_features, …) preserved without forcing a schema on them.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl ModelMetadata {
    /// True only when the bundle declares itself promotable for live inference:
    /// `training_info.status == "trained"`, `runtime_eligible == true`, and a
    /// `promotion_status` of `shadow` or `promoted`. Bundles without
    /// `training_info` (classical/tabular) are never eligible by this gate and
    /// must be loaded through their own path. Mirrors
    /// `vision::fqe_runner::is_runtime_eligible`.
    pub fn is_runtime_eligible(&self) -> bool {
        let Some(training) = self.training_info.as_ref() else {
            return false;
        };
        training.get("status").and_then(|v| v.as_str()) == Some("trained")
            && training.get("runtime_eligible").and_then(|v| v.as_bool()) == Some(true)
            && matches!(
                training.get("promotion_status").and_then(|v| v.as_str()),
                Some("shadow" | "promoted")
            )
    }
}

/// Semver parsed from a directory name. Non-numeric or malformed names
/// (e.g. "latest", "not-a-version", "1.2") return `None` and are ignored by
/// version selection.
fn parse_semver(name: &str) -> Option<(u64, u64, u64)> {
    let mut parts = name.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None; // more than three components — not a bare semver
    }
    Some((major, minor, patch))
}

/// Tracks available model bundles discovered on the local filesystem.
pub struct ModelRegistry {
    base_path: PathBuf,
    models: HashMap<String, ModelMetadata>,
}

impl ModelRegistry {
    pub fn new(base_path: &str) -> Self {
        Self {
            base_path: PathBuf::from(base_path),
            models: HashMap::new(),
        }
    }

    /// Scan `<base>/<model>/<version>/metadata.json` and register the highest
    /// semver per model. Resilient by design: a missing base dir yields an empty
    /// registry (the daemon may start before models are downloaded), and any
    /// unreadable/unparseable bundle is skipped with a warning rather than
    /// failing the whole scan. Idempotent — clears and rebuilds each call.
    pub fn scan(&mut self) -> crate::error::Result<()> {
        self.models.clear();
        if !self.base_path.is_dir() {
            tracing::warn!(
                "model registry base '{}' does not exist; no models registered",
                self.base_path.display()
            );
            return Ok(());
        }

        for model_entry in std::fs::read_dir(&self.base_path)? {
            let model_dir = match model_entry {
                Ok(entry) if entry.path().is_dir() => entry.path(),
                Ok(_) => continue, // stray file at the base level
                Err(err) => {
                    tracing::warn!("skipping unreadable registry entry: {err}");
                    continue;
                }
            };
            let Some(model_name) = model_dir.file_name().and_then(|n| n.to_str()).map(str::to_string)
            else {
                continue;
            };

            // Highest semver version dir that actually contains a metadata.json.
            let mut best: Option<((u64, u64, u64), PathBuf)> = None;
            let version_iter = match std::fs::read_dir(&model_dir) {
                Ok(iter) => iter,
                Err(err) => {
                    tracing::warn!("skipping model '{model_name}': {err}");
                    continue;
                }
            };
            for version_dir in version_iter.flatten().map(|e| e.path()) {
                if !version_dir.is_dir() || !version_dir.join("metadata.json").is_file() {
                    continue;
                }
                let Some(version) = version_dir
                    .file_name()
                    .and_then(|n| n.to_str())
                    .and_then(parse_semver)
                else {
                    continue; // non-semver dir name (e.g. "latest") — ignore
                };
                if best.as_ref().map(|(v, _)| version > *v).unwrap_or(true) {
                    best = Some((version, version_dir));
                }
            }

            let Some((_, version_dir)) = best else {
                continue; // no versioned metadata under this model
            };

            match Self::read_metadata(&version_dir) {
                Ok(mut meta) => {
                    // Trust the directory name as the authoritative version so
                    // model_path() always resolves; warn on any drift.
                    if let Some(dir_version) = version_dir.file_name().and_then(|n| n.to_str()) {
                        if meta.version != dir_version {
                            tracing::warn!(
                                "model '{model_name}' metadata version '{}' != dir '{}'; using dir",
                                meta.version,
                                dir_version
                            );
                            meta.version = dir_version.to_string();
                        }
                    }
                    tracing::info!(
                        "registered model '{model_name}' v{} (runtime_eligible={})",
                        meta.version,
                        meta.is_runtime_eligible()
                    );
                    self.models.insert(model_name, meta);
                }
                Err(err) => {
                    tracing::warn!("skipping '{model_name}': bad metadata.json: {err}");
                }
            }
        }

        tracing::info!("model registry scan complete: {} model(s)", self.models.len());
        Ok(())
    }

    fn read_metadata(version_dir: &Path) -> crate::error::Result<ModelMetadata> {
        let raw = std::fs::read_to_string(version_dir.join("metadata.json"))?;
        Ok(serde_json::from_str(&raw)?)
    }

    /// Metadata for a registered model (its selected latest version).
    pub fn get(&self, model_name: &str) -> Option<&ModelMetadata> {
        self.models.get(model_name)
    }

    /// All registered models, keyed by name.
    pub fn models(&self) -> &HashMap<String, ModelMetadata> {
        &self.models
    }

    /// Directory of the registered version: `<base>/<model>/<version>`.
    pub fn model_dir(&self, model_name: &str) -> Option<PathBuf> {
        self.models
            .get(model_name)
            .map(|m| self.base_path.join(model_name).join(&m.version))
    }

    /// Path to the ONNX artifact, but only when the registered bundle actually
    /// ships one (classical/tabular bundles won't). `None` signals "not
    /// loadable as ONNX" rather than fabricating a nonexistent path.
    pub fn model_path(&self, model_name: &str) -> Option<PathBuf> {
        let path = self.model_dir(model_name)?.join("model.onnx");
        path.is_file().then_some(path)
    }

    /// Path to any sibling artifact in the bundle (calibration.json,
    /// norm_params.json, kalman_state.json, …) when present.
    pub fn artifact_path(&self, model_name: &str, file: &str) -> Option<PathBuf> {
        let path = self.model_dir(model_name)?.join(file);
        path.is_file().then_some(path)
    }

    /// Verify the on-disk `model.onnx` matches the SHA-256 recorded in
    /// metadata.json. Returns `Ok(false)` on mismatch (corrupted or swapped
    /// weights), and errors only when the model is unregistered or the artifact
    /// cannot be read. This is the integrity gate the daemon runs before load.
    pub fn verify_hash(&self, model_name: &str) -> crate::error::Result<bool> {
        let meta = self
            .models
            .get(model_name)
            .ok_or_else(|| InferenceError::ModelNotFound(format!("'{model_name}' not registered")))?;
        let path = self
            .model_dir(model_name)
            .map(|d| d.join("model.onnx"))
            .ok_or_else(|| InferenceError::ModelNotFound(model_name.to_string()))?;
        let bytes = std::fs::read(&path)
            .map_err(|err| InferenceError::ModelNotFound(format!("{}: {err}", path.display())))?;
        let actual = format!("{:x}", Sha256::digest(&bytes));
        Ok(actual.eq_ignore_ascii_case(&meta.sha256))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a minimal but schema-valid bundle at `<base>/<model>/<version>/`.
    fn write_bundle(base: &Path, model: &str, version: &str, onnx: &[u8], eligible: bool) {
        let dir = base.join(model).join(version);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.onnx"), onnx).unwrap();
        let sha = format!("{:x}", Sha256::digest(onnx));
        let training = if eligible {
            r#","training_info":{"status":"trained","runtime_eligible":true,"promotion_status":"shadow"}"#
        } else {
            r#","training_info":{"status":"reference","runtime_eligible":false,"promotion_status":"never"}"#
        };
        std::fs::write(
            dir.join("metadata.json"),
            format!(
                r#"{{"model_name":"{model}","version":"{version}","task":"t","sha256":"{sha}","created_at":"now"{training}}}"#
            ),
        )
        .unwrap();
    }

    #[test]
    fn scan_registers_highest_semver_and_verifies_hash() {
        let root = std::env::temp_dir().join(format!("registry_scan_{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        write_bundle(&root, "focus_quality", "1.0.0", b"a", false);
        write_bundle(&root, "focus_quality", "1.2.0", b"onnx-bytes", true);
        write_bundle(&root, "focus_quality", "0.9.9", b"c", true);
        // A malformed bundle must be skipped, not abort the whole scan.
        let bad = root.join("cloud_classifier").join("1.0.0");
        std::fs::create_dir_all(&bad).unwrap();
        std::fs::write(bad.join("metadata.json"), b"{ not json").unwrap();
        // A non-semver dir name is ignored by version selection.
        std::fs::create_dir_all(root.join("focus_quality").join("latest")).unwrap();

        let mut reg = ModelRegistry::new(root.to_str().unwrap());
        reg.scan().unwrap();

        let meta = reg.get("focus_quality").expect("focus_quality registered");
        assert_eq!(meta.version, "1.2.0", "must pick highest semver");
        assert!(meta.is_runtime_eligible());
        assert!(reg.get("cloud_classifier").is_none(), "malformed bundle skipped");
        assert!(reg.verify_hash("focus_quality").unwrap(), "hash matches metadata");
        assert!(reg
            .model_path("focus_quality")
            .unwrap()
            .ends_with("focus_quality/1.2.0/model.onnx"));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn verify_hash_detects_tampered_artifact() {
        let root = std::env::temp_dir().join(format!("registry_verify_{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        write_bundle(&root, "focus_quality", "1.0.0", b"original", true);
        let mut reg = ModelRegistry::new(root.to_str().unwrap());
        reg.scan().unwrap();
        assert!(reg.verify_hash("focus_quality").unwrap());

        // Swap the artifact for different bytes — metadata sha256 no longer matches.
        std::fs::write(
            root.join("focus_quality").join("1.0.0").join("model.onnx"),
            b"tampered",
        )
        .unwrap();
        assert!(!reg.verify_hash("focus_quality").unwrap(), "tamper detected");
        assert!(reg.verify_hash("missing").is_err(), "unregistered model errors");

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn missing_base_dir_yields_empty_registry() {
        let mut reg = ModelRegistry::new("/nonexistent/models/root");
        reg.scan().unwrap();
        assert!(reg.models().is_empty());
    }
}

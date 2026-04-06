//! Unified error types for the Achernar daemon.
//! Implements Section 5.3 — Error Handling Strategy.

use thiserror::Error;

/// Top-level daemon error type.
#[derive(Debug, Error)]
pub enum AchernarError {
    #[error("Hardware error: {0}")]
    Hardware(#[from] HardwareError),

    #[error("Focus error: {0}")]
    Focus(#[from] FocusError),

    #[error("Control error: {0}")]
    Control(#[from] ControlError),

    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),

    #[error("Config error: {0}")]
    Config(#[from] ConfigError),

    #[error("ML inference error: {0}")]
    Inference(#[from] InferenceError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("{0}")]
    Other(String),
}

#[derive(Debug, Error)]
pub enum HardwareError {
    #[error("Device not found: {0}")]
    DeviceNotFound(String),

    #[error("Device busy: {0}")]
    DeviceBusy(String),

    #[error("Communication error: {0}")]
    Communication(String),

    #[error("Timeout waiting for device: {0}")]
    Timeout(String),

    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

#[derive(Debug, Error)]
pub enum FocusError {
    #[error("Focus run failed: {0}")]
    RunFailed(String),

    #[error("Safety violation: {0}")]
    SafetyViolation(String),

    #[error("No focuser connected")]
    NoFocuser,

    #[error("Convergence failed after {0} iterations")]
    ConvergenceFailed(u32),

    #[error("Backlash compensation error: {0}")]
    BacklashError(String),
}

#[derive(Debug, Error)]
pub enum ControlError {
    #[error("Action rejected: {0}")]
    ActionRejected(String),

    #[error("Device locked by: {0}")]
    DeviceLocked(String),

    #[error("Mode not allowed: {0}")]
    ModeNotAllowed(String),

    #[error("Safety guard blocked action: {0}")]
    SafetyBlocked(String),
}

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("Database error: {0}")]
    Database(String),

    #[error("Migration error: {0}")]
    Migration(String),

    #[error("Record not found: {0}")]
    NotFound(String),
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Config file not found: {0}")]
    FileNotFound(String),

    #[error("Config parse error: {0}")]
    ParseError(String),

    #[error("Missing required field: {0}")]
    MissingField(String),
}

#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("ONNX runtime error: {0}")]
    OnnxError(String),

    #[error("Invalid input shape: expected {expected}, got {got}")]
    InvalidShape { expected: String, got: String },

    #[error("Model version mismatch: {0}")]
    VersionMismatch(String),
}

pub type Result<T> = std::result::Result<T, AchernarError>;

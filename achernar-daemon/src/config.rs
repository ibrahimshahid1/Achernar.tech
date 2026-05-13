//! Configuration parsing for achernar.toml.
//! Implements all config sections including [focus], [focus.safety],
//! [focus.backlash], [focus.models], [mount], [filter_wheel].

use serde::Deserialize;

/// Top-level daemon configuration parsed from achernar.toml.
#[derive(Debug, Clone, Deserialize)]
pub struct AchernarConfig {
    #[serde(default)]
    pub daemon: DaemonConfig,
    #[serde(default)]
    pub focus: FocusConfig,
    #[serde(default)]
    pub mount: MountConfig,
    #[serde(default)]
    pub filter_wheel: FilterWheelConfig,
    #[serde(default)]
    pub camera: CameraConfig,
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DaemonConfig {
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default)]
    pub data_dir: Option<String>,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            port: default_port(),
            log_level: default_log_level(),
            data_dir: None,
        }
    }
}

/// Focus subsystem configuration — Section 7.3, 7.4.
#[derive(Debug, Clone, Deserialize)]
pub struct FocusConfig {
    #[serde(default)]
    pub safety: FocusSafetyConfig,
    #[serde(default)]
    pub backlash: BacklashConfig,
    #[serde(default)]
    pub models: FocusModelsConfig,
    /// Autofocus step size for initial sweep
    #[serde(default = "default_step_size")]
    pub initial_step_size: i32,
    /// Number of samples per focus position
    #[serde(default = "default_samples_per_position")]
    pub samples_per_position: u32,
    /// Settle time in ms after each focus move
    #[serde(default = "default_settle_ms")]
    pub settle_ms: u64,
}

impl Default for FocusConfig {
    fn default() -> Self {
        Self {
            safety: FocusSafetyConfig::default(),
            backlash: BacklashConfig::default(),
            models: FocusModelsConfig::default(),
            initial_step_size: default_step_size(),
            samples_per_position: default_samples_per_position(),
            settle_ms: default_settle_ms(),
        }
    }
}

/// Focus safety guard configuration — Section 7.3.3.
#[derive(Debug, Clone, Deserialize)]
pub struct FocusSafetyConfig {
    #[serde(default)]
    pub min_position: i32,
    #[serde(default = "default_max_position")]
    pub max_position: i32,
    #[serde(default = "default_max_step_single")]
    pub max_step_single: i32,
    #[serde(default = "default_max_total_travel")]
    pub max_total_travel_per_run: i32,
    #[serde(default = "default_min_confidence")]
    pub min_confidence_to_act: f64,
    #[serde(default = "default_max_failed")]
    pub max_failed_improvements: u32,
    #[serde(default = "default_settle_ms")]
    pub settle_ms: u64,
    #[serde(default)]
    pub backlash_steps: i32,
}

impl Default for FocusSafetyConfig {
    fn default() -> Self {
        Self {
            min_position: 0,
            max_position: default_max_position(),
            max_step_single: default_max_step_single(),
            max_total_travel_per_run: default_max_total_travel(),
            min_confidence_to_act: default_min_confidence(),
            max_failed_improvements: default_max_failed(),
            settle_ms: default_settle_ms(),
            backlash_steps: 0,
        }
    }
}

/// Backlash compensation configuration — Section 7.3.5.
#[derive(Debug, Clone, Deserialize)]
pub struct BacklashConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub compensation_steps: i32,
    #[serde(default)]
    pub always_approach_from: Option<String>,
}

impl Default for BacklashConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            compensation_steps: 0,
            always_approach_from: None,
        }
    }
}

/// Focus model paths configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct FocusModelsConfig {
    #[serde(default)]
    pub quality_model_path: Option<String>,
    #[serde(default)]
    pub offset_model_path: Option<String>,
    #[serde(default)]
    pub drift_model_path: Option<String>,
}

impl Default for FocusModelsConfig {
    fn default() -> Self {
        Self {
            quality_model_path: None,
            offset_model_path: None,
            drift_model_path: None,
        }
    }
}

/// Mount configuration — Section 8.3.
#[derive(Debug, Clone, Deserialize)]
pub struct MountConfig {
    #[serde(default)]
    pub safety: MountSafetyConfig,
    #[serde(default)]
    pub guide_rate: f64,
    #[serde(default)]
    pub max_slew_rate: f64,
}

impl Default for MountConfig {
    fn default() -> Self {
        Self {
            safety: MountSafetyConfig::default(),
            guide_rate: 0.5,
            max_slew_rate: 8.0,
        }
    }
}

/// Mount safety configuration — Section 8.3.
#[derive(Debug, Clone, Deserialize)]
pub struct MountSafetyConfig {
    #[serde(default)]
    pub min_altitude: f64,
    #[serde(default = "default_max_ha")]
    pub max_hour_angle: f64,
    #[serde(default)]
    pub horizon_model_path: Option<String>,
    #[serde(default)]
    pub park_on_safety_violation: bool,
}

impl Default for MountSafetyConfig {
    fn default() -> Self {
        Self {
            min_altitude: 0.0,
            max_hour_angle: default_max_ha(),
            horizon_model_path: None,
            park_on_safety_violation: true,
        }
    }
}

/// Filter wheel configuration — Section 8.2.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct FilterWheelConfig {
    #[serde(default)]
    pub filter_names: Vec<String>,
    #[serde(default)]
    pub focus_offsets: Vec<i32>,
}

/// Camera configuration.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct CameraConfig {
    #[serde(default)]
    pub default_gain: Option<u32>,
    #[serde(default)]
    pub default_offset: Option<u32>,
    #[serde(default)]
    pub cooling_target: Option<f64>,
}

/// Storage configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct StorageConfig {
    #[serde(default = "default_db_path")]
    pub database_path: String,
    #[serde(default)]
    pub frames_dir: Option<String>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            database_path: default_db_path(),
            frames_dir: None,
        }
    }
}

/// Telemetry/logging configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct TelemetryConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
    #[serde(default)]
    pub json_output: bool,
    #[serde(default)]
    pub log_dir: Option<String>,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            json_output: false,
            log_dir: None,
        }
    }
}

impl AchernarConfig {
    /// Load config from achernar.toml, falling back to defaults.
    pub fn load() -> crate::error::Result<Self> {
        let config_path =
            std::env::var("ACHERNAR_CONFIG").unwrap_or_else(|_| "achernar.toml".to_string());

        if std::path::Path::new(&config_path).exists() {
            let content = std::fs::read_to_string(&config_path).map_err(|e| {
                crate::error::AchernarError::Other(format!("Failed to read config: {}", e))
            })?;
            toml::from_str(&content).map_err(|e| {
                crate::error::AchernarError::Other(format!("Failed to parse config: {}", e))
            })
        } else {
            tracing::info!("No config file found at {}, using defaults", config_path);
            Ok(Self::default())
        }
    }
}

impl Default for AchernarConfig {
    fn default() -> Self {
        Self {
            daemon: DaemonConfig::default(),
            focus: FocusConfig::default(),
            mount: MountConfig::default(),
            filter_wheel: FilterWheelConfig::default(),
            camera: CameraConfig::default(),
            storage: StorageConfig::default(),
            telemetry: TelemetryConfig::default(),
        }
    }
}

// Default value functions
fn default_port() -> u16 {
    3030
}
fn default_log_level() -> String {
    "info".to_string()
}
fn default_step_size() -> i32 {
    50
}
fn default_samples_per_position() -> u32 {
    3
}
fn default_settle_ms() -> u64 {
    500
}
fn default_max_position() -> i32 {
    50000
}
fn default_max_step_single() -> i32 {
    500
}
fn default_max_total_travel() -> i32 {
    5000
}
fn default_min_confidence() -> f64 {
    0.6
}
fn default_max_failed() -> u32 {
    3
}
fn default_max_ha() -> f64 {
    5.5
}
fn default_db_path() -> String {
    "achernar.db".to_string()
}

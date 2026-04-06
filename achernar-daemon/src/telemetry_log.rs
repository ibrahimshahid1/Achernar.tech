//! Structured tracing-rs logging with JSON rotation.
//! Implements telemetry infrastructure for the daemon.

use tracing_subscriber::{fmt, EnvFilter, prelude::*};

/// Initialize the tracing/telemetry subsystem.
pub fn init_telemetry(config: &crate::config::TelemetryConfig) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&config.level));

    if config.json_output {
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt::layer().json())
            .init();
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt::layer().with_target(false))
            .init();
    }

    tracing::info!("Telemetry initialized at level={}", config.level);
}

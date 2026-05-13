//! Environmental sensors — Arduino/USB, DS18B20, BME280.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentReading {
    pub temperature_c: Option<f64>,
    pub humidity_pct: Option<f64>,
    pub pressure_hpa: Option<f64>,
    pub dew_point_c: Option<f64>,
    pub sky_temperature_c: Option<f64>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub struct EnvironmentSensors {
    port: Option<String>,
    connected: bool,
}

impl EnvironmentSensors {
    pub fn new(port: Option<String>) -> Self {
        Self {
            port,
            connected: false,
        }
    }

    pub async fn connect(&mut self) -> crate::error::Result<()> {
        todo!("Connect to environment sensors via serial/USB")
    }

    pub fn read(&self) -> crate::error::Result<EnvironmentReading> {
        todo!("Read current environment data")
    }
}

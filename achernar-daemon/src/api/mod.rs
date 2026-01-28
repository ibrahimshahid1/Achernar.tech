pub mod camera;
pub mod mount;
pub mod focuser;

use axum::{http::StatusCode, Json};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct EquipmentStatus {
    pub connected: bool,
    pub camera: Option<CameraStatus>,
    pub mount: Option<MountStatus>,
    pub focuser: Option<FocuserStatus>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CameraStatus {
    pub name: String,
    pub temperature: f64,
    pub cooling: bool,
    pub exposing: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MountStatus {
    pub name: String,
    pub ra: f64,
    pub dec: f64,
    pub tracking: bool,
    pub parked: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FocuserStatus {
    pub name: String,
    pub position: i32,
    pub moving: bool,
}

pub async fn get_status() -> Result<Json<EquipmentStatus>, StatusCode> {
    // TODO: Implement actual status retrieval
    let status = EquipmentStatus {
        connected: false,
        camera: None,
        mount: None,
        focuser: None,
    };
    
    Ok(Json(status))
}

pub async fn connect_equipment() -> Result<StatusCode, StatusCode> {
    // TODO: Implement equipment connection
    Ok(StatusCode::OK)
}

pub async fn disconnect_equipment() -> Result<StatusCode, StatusCode> {
    // TODO: Implement equipment disconnection
    Ok(StatusCode::OK)
}

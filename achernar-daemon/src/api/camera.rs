use axum::{http::StatusCode, Json};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct CaptureParams {
    pub exposure: f64,
    pub gain: Option<u32>,
    pub offset: Option<u32>,
    pub binning: Option<u32>,
    pub filter: Option<String>,
    pub temperature: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct ImageData {
    pub id: String,
    pub width: u32,
    pub height: u32,
    pub metadata: ImageMetadata,
}

#[derive(Debug, Serialize)]
pub struct ImageMetadata {
    pub exposure: f64,
    pub gain: u32,
    pub temperature: f64,
    pub timestamp: String,
}

/**
 * High-resolution capture
 * This is separate from the live view stream
 * Saves full-resolution RAW/FITS to disk
 */
pub async fn capture(
    Json(params): Json<CaptureParams>,
) -> Result<Json<ImageData>, StatusCode> {
    // TODO: Implement actual high-res capture logic
    tracing::info!(
        "High-res capture requested: {} seconds (saved to disk)",
        params.exposure
    );
    
    let image = ImageData {
        id: uuid::Uuid::new_v4().to_string(),
        width: 4096, // Full resolution
        height: 3072,
        metadata: ImageMetadata {
            exposure: params.exposure,
            gain: params.gain.unwrap_or(100),
            temperature: -10.0,
            timestamp: chrono::Utc::now().to_rfc3339(),
        },
    };
    
    Ok(Json(image))
}

pub async fn abort() -> Result<StatusCode, StatusCode> {
    // TODO: Implement capture abort
    tracing::info!("High-res capture abort requested");
    Ok(StatusCode::OK)
}

/**
 * MJPEG Stream (alternative to WebSocket)
 * Can be used directly in <img> src attribute
 */
pub async fn stream_mjpeg() -> Result<String, StatusCode> {
    // TODO: Implement MJPEG streaming
    // Return multipart/x-mixed-replace content type
    tracing::info!("MJPEG stream requested");
    
    // For now, return placeholder
    Ok("MJPEG stream not yet implemented".to_string())
}

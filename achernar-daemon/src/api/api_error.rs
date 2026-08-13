use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};

/// Stable JSON error returned by the hardware-control API.
///
/// A bare HTTP 409 tells a frontend that something failed, but not whether it
/// should ask the operator to stop a stream, reconnect a camera, or correct an
/// invalid value. Keep the machine-readable code stable; the message is for
/// humans and logs.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
}

impl ApiError {
    pub fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }

    pub fn no_camera() -> Self {
        Self::new(
            StatusCode::PRECONDITION_FAILED,
            "camera_not_connected",
            "No camera is connected",
        )
    }

    pub fn hardware(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_GATEWAY, "camera_hardware_error", message)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({
                "error": {
                    "code": self.code,
                    "message": self.message,
                }
            })),
        )
            .into_response()
    }
}

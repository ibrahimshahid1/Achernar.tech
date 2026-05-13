//! Dither policy — Section 8.3 (M-2).
//! DitherPolicyMLP inference for optimal dither parameters.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DitherParams {
    pub dx_pixels: f64,
    pub dy_pixels: f64,
    pub settle_timeout_s: f64,
    pub settle_threshold: f64,
}

pub struct DitherPolicy;

impl DitherPolicy {
    pub fn predict_dither(&self, _seeing: f64, _guide_rms: f64, _pixel_scale: f64) -> DitherParams {
        todo!("Implement DitherPolicyMLP inference — Section 8.3 M-2")
    }
}

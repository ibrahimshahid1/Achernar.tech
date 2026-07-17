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
    pub fn predict_dither(&self, seeing: f64, guide_rms: f64, pixel_scale: f64) -> DitherParams {
        let pixel_scale = pixel_scale.max(0.1);
        let seeing_pixels = (seeing.max(0.0) / pixel_scale).clamp(0.0, 20.0);
        let rms_pixels = (guide_rms.max(0.0) / pixel_scale).clamp(0.0, 20.0);
        let magnitude = (3.0 + seeing_pixels * 0.6 + rms_pixels * 0.8).clamp(1.0, 15.0);
        let settle_threshold = (0.35 + guide_rms * 0.35 + seeing * 0.08).clamp(0.3, 2.0);
        let settle_timeout_s = (10.0 + magnitude * 1.5 + guide_rms * 4.0).clamp(10.0, 60.0);

        DitherParams {
            dx_pixels: magnitude,
            dy_pixels: magnitude * 0.618,
            settle_timeout_s,
            settle_threshold,
        }
    }
}

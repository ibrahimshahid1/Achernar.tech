//! Bare-array background entry point (no masks, no camera provenance).
//!
//! Formerly a "classical SEP wrapper" that called the custom estimator,
//! reported its clipped median as both a mean and a median, and hard-coded
//! zero gradients. It now returns the versioned legacy result with honest
//! provenance: no masks were supplied, saturation is unknown, units are
//! unverified, and no gradient exists because the method has no spatial
//! model. It has no callers in the daemon; frames from cameras go through
//! `quality::background_from_luma` or the configured backend instead.

use super::background_model::BackgroundAnalysis;
use super::quality::{background_from_luma, LumaFrame};

pub struct BackgroundEstimator;

impl BackgroundEstimator {
    /// Legacy global estimate of a bare array (row-major, `shape = (h, w)`).
    pub fn estimate(image: &ndarray::Array2<f32>) -> BackgroundAnalysis {
        let (height, width) = image.dim();
        let frame = LumaFrame {
            pixels: image.iter().copied().collect(),
            width,
            height,
            // Container depth is not a saturation point; nothing is known.
            max_adu: f32::MAX,
            saturation_adu: None,
            masks: Default::default(),
            analysis_descriptor: None,
        };
        background_from_luma(&frame)
    }
}

#[cfg(test)]
mod tests {
    use super::BackgroundEstimator;
    use crate::vision::background_model::{BackgroundMethod, BackgroundReason};

    #[test]
    fn bare_array_result_declares_what_it_does_not_know() {
        let image = ndarray::Array2::from_shape_fn((16, 16), |(y, x)| 100.0 + (x + y) as f32);
        let analysis = BackgroundEstimator::estimate(&image);
        let result = &analysis.result;
        assert_eq!(result.method, BackgroundMethod::LegacyGlobalClip);
        assert_eq!(result.units, "unverified");
        assert_eq!(result.support.saturation_evidence, "unknown");
        assert_eq!(result.saturation_fraction, None);
        assert!(result
            .support
            .masks
            .values()
            .all(|state| state == "unavailable"));
        // A ramp has a gradient; this method cannot measure one, so none is
        // reported rather than a zero.
        assert!(result.plane.is_none());
        assert_eq!(
            result.plane_unavailable_reason,
            Some(BackgroundReason::NoSpatialModel)
        );
    }
}

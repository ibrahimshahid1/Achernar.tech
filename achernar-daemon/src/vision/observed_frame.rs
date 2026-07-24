//! ObservedFrame — Section 8.1.
//!
//! The exact camera frame that produced an observation, published together as an
//! immutable envelope. Autofocus persistence consumes this so the pixels it
//! saves provably belong to the HFR/star-count label attached to them, and so
//! "freshness" can be keyed on the camera `frame_counter` rather than the
//! wall-clock time at which the vision loop happened to process a frame.
//!
//! Without this envelope, persistence clones whatever frame is *currently*
//! latest (which may be a later exposure), and reprocessing one camera frame
//! could masquerade as several distinct samples.

use std::sync::Arc;

use crate::hardware::camera::FrameData;

use super::shared_observation::SharedObservation;

#[derive(Clone)]
pub struct ObservedFrame {
    /// Camera-assigned frame counter (monotonic per stream). Freshness key.
    pub frame_counter: i32,
    /// Camera capture timestamp (seconds since the Unix epoch).
    pub capture_unix_time: f64,
    /// The exact pixels the observation was computed from.
    pub frame: Arc<FrameData>,
    /// The observation derived from `frame`.
    pub observation: SharedObservation,
}

//! Vision / Shared Perception — Section 8.1.
//! Shared observation pipeline, vision loop, and inference runners.

pub mod background;
pub mod background_estimator;
pub mod coc_runner;
pub mod coc_shadow_runner;
pub mod coc_shadow;
pub mod fqe_runner;
pub mod observed_frame;
pub mod preprocess;
pub mod quality;
pub mod shared_observation;
pub mod ted_runner;
pub mod trailing;
pub mod vision_loop;

//! Storage layer — Sections 5.2, 7.4, 8.4.
//! SQLite repository modules for all persistent data.

pub mod device_profiles_repo;
pub mod filter_switch_repo;
pub mod focus_actions_repo;
pub mod focus_drift_repo;
pub mod focus_model_registry_repo;
pub mod focus_predictions_repo;
pub mod focus_runs_repo;
pub mod focus_samples_repo;
pub mod frames_repo;
pub mod guider_state_repo;
pub mod local_store;
pub mod plate_solve_repo;
pub mod pointing_model_repo;
pub mod sessions_repo;
pub mod sync_client;

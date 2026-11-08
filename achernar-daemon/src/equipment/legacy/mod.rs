//! OLS legacy adapters run through the same worker until they are deleted.
//!
//! The OLS Alpaca mount and the OLS focuser were deleted at the end of Tier 1
//! once the native Alpaca drivers passed acceptance (O2). What remains:
//! `ols_mount` is the OLS `indi:remote` mount (`ols_indi`), an interim adapter
//! until the native INDI client passes acceptance in Tier 2 (O3); `sim_focuser`
//! is the older in-process focuser simulator behind a worker, for tests.
//!
//! Module tree created by WP0; owner: WP0 (module tree); each file is owned by
//! its device package.

pub mod ols_mount;
pub mod sim_focuser;

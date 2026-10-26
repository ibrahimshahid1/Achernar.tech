//! PHD2 client and guider service (G-a to G-m); the direct guider and M-1
//! actuation are WP2-7.
//!
//! Module tree created by WP0; owner: WP1A-7.

pub mod algo;
pub mod direct;
pub mod dither_select;
pub mod phd2_client;
pub mod phd2_proto;
pub mod rms;
pub mod service;
pub mod steps;

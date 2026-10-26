//! Native equipment layer (spec `docs/equipment-layer-design-2026-10-25.md`).
//!
//! Cameras stay on OpenLiveStacker (O1). Every other device class (mount,
//! focuser, filter wheel, guider, rotator, cover calibrator, switch, weather,
//! safety monitor, dome) is driven from here, and every Achernar v6 model
//! reads device data through [`traits::EquipmentReader`] rather than driver
//! objects (D6).
//!
//! WP0 froze the contracts below and created the whole module tree; every
//! other file is a doc-only placeholder owned by the package named in it
//! (§2.2). Frozen contracts (§2.4):
//!
//! - [`types`]: errors, `Observed<T>`, identity, coordinates, sources,
//!   priorities, outcomes, health.
//! - [`traits`]: driver traits, `AlpacaTransport`, `FrameConverter`,
//!   `SafetyCheck`, `CameraActivity`, `Clock`, `EquipmentReader`,
//!   `RecorderSink`.
//! - [`events`]: the event catalogue and the ground-truth records.
//! - [`snapshot`]: device states, snapshots, windows and model contexts.
//! - [`commands`]: device commands and `GatedCommand`.
//! - [`ops`]: the `Operation` trait the worker drives.
//! - [`clock`]: the production clock.
//! - [`handle`]: `DeviceHandle`, permits and tickets.
//! - `testing` (tests only): `FakeAlpacaTransport`, fake handles,
//!   `ManualClock`, plus `IdentityConverter` and `NullRecorder` in `traits`.
//!
//! Owner decision O4 dropped the native LX200 and MoonLite drivers, so the
//! §2.2 `serial/` tree does not exist; those devices are reached through
//! indiserver or ASCOM Remote.
//!
//! Default configuration opens nothing and broadcasts nothing (D13). Nothing
//! in this module runs until the spine (WP1B-1) wires the hub into
//! `AppState`.

// The contracts are consumed by later work packages; until the spine wires
// the hub into `AppState`, nothing outside this module calls them.
#![allow(dead_code)]

pub mod clock;
pub mod commands;
pub mod events;
pub mod handle;
pub mod ops;
pub mod snapshot;
pub mod traits;
pub mod types;

pub mod busy;
pub mod coord;
pub mod estop;
pub mod gate;
pub mod health_bridge;
pub mod hub;
pub mod profile;
pub mod recorder;
pub mod resume;
pub mod ring;
pub mod worker;

pub mod alpaca;
pub mod astro;
pub mod filter;
pub mod focuser;
pub mod guider;
pub mod indi;
pub mod legacy;
pub mod mount;
pub mod safety;
pub mod sim;

#[cfg(test)]
pub mod testing;

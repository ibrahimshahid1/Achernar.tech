//! Test doubles for the frozen contracts (spec §2.4 "testing/", §5.1 rule
//! 8). Compiled only for tests.
//!
//! - [`fake_transport`]: `FakeAlpacaTransport`, scripted Alpaca replies
//!   including HTTP 400 with a plain-text body.
//! - [`fake_handles`]: `FakeDevice` handles for every kind, `FakeAbortPath`,
//!   `FakeCameraActivity`, `RecordingSink` and `FakeReader`.
//! - [`manual_clock`] provides `ManualClock`.
//! - `IdentityConverter` and `NullRecorder` live in `traits`, because an
//!   inert hub uses them outside tests too.
//!
//! The other modules belong to later packages (§2.2):
//!
//! - [`alpaca_server`]: an in-process Alpaca HTTP server with scripted
//!   devices, and the adapter that serves the simulator drivers (O11 l).
//! - [`conformance`]: the generic per-trait conformance suites.
//! - [`phd2_mock`]: a scripted PHD2 event server.
//! - [`interop`]: the Tier 1 interop and acceptance tests, CI-runnable and
//!   local-only (§5.7, O9).

pub mod fake_handles;
pub mod fake_transport;
pub mod manual_clock;

pub mod alpaca_server;
pub mod conformance;
pub mod interop;
pub mod phd2_mock;

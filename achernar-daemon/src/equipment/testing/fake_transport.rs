//! `FakeAlpacaTransport`: scripted Alpaca member replies (spec §2.4
//! "testing/"), including HTTP 400 with a plain-text body.
//!
//! Replies are mapped to errors with the same constructors the real client
//! (`alpaca/client.rs`, WP1A-1) uses, so a driver sees one error vocabulary
//! from both. A member nobody scripted answers like an older Alpaca server
//! that does not know it: HTTP 400 with a plain-text body.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::equipment::traits::{AlpacaTransport, TimeoutClass};
use crate::equipment::types::{DeviceError, DeviceResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Method {
    Get,
    Put,
}

/// A latch a test releases to unblock a [`FakeReply::Blocked`] call.
#[derive(Debug, Clone, Default)]
pub struct Latch(Arc<(Mutex<bool>, Condvar)>);

impl Latch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn release(&self) {
        let (released, cv) = &*self.0;
        *released.lock().unwrap_or_else(|e| e.into_inner()) = true;
        cv.notify_all();
    }

    /// Wait for release, at most `timeout` of real time. True if released.
    pub fn wait(&self, timeout: Duration) -> bool {
        let (released, cv) = &*self.0;
        let deadline = Instant::now() + timeout;
        let mut guard = released.lock().unwrap_or_else(|e| e.into_inner());
        while !*guard {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            guard = cv
                .wait_timeout(guard, deadline - now)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        true
    }
}

/// One scripted reply.
#[derive(Debug, Clone)]
pub enum FakeReply {
    /// HTTP 200, `ErrorNumber` 0, this `Value`.
    Value(serde_json::Value),
    /// HTTP 200 with a non-zero `ErrorNumber`.
    AlpacaError {
        number: i32,
        message: String,
    },
    /// A non-200 status with a plain-text body (C-3 step 1).
    Http {
        status: u16,
        body: String,
    },
    Timeout,
    Transport(String),
    /// Real-time delay, then the inner reply.
    Delay(Duration, Box<FakeReply>),
    /// Blocks until the latch is released (at most 30 s), then the inner
    /// reply. Other calls proceed meanwhile.
    Blocked(Latch, Box<FakeReply>),
}

impl FakeReply {
    pub fn value(value: impl Into<serde_json::Value>) -> Self {
        FakeReply::Value(value.into())
    }

    /// A void PUT's success.
    pub fn ok() -> Self {
        FakeReply::Value(serde_json::Value::Null)
    }

    pub fn not_implemented() -> Self {
        FakeReply::AlpacaError {
            number: crate::equipment::types::ascom_error::NOT_IMPLEMENTED,
            message: "Property not implemented".into(),
        }
    }

    pub fn value_not_set() -> Self {
        FakeReply::AlpacaError {
            number: crate::equipment::types::ascom_error::VALUE_NOT_SET,
            message: "Value not set".into(),
        }
    }

    pub fn alpaca_error(number: i32, message: impl Into<String>) -> Self {
        FakeReply::AlpacaError {
            number,
            message: message.into(),
        }
    }

    pub fn http(status: u16, body: impl Into<String>) -> Self {
        FakeReply::Http {
            status,
            body: body.into(),
        }
    }

    fn resolve(self) -> DeviceResult<serde_json::Value> {
        match self {
            FakeReply::Value(value) => Ok(value),
            FakeReply::AlpacaError { number, message } => {
                Err(DeviceError::from_alpaca_error(number, &message))
            }
            FakeReply::Http { status, body } => Err(DeviceError::from_http_status(status, &body)),
            FakeReply::Timeout => Err(DeviceError::timeout("fake transport timed out")),
            FakeReply::Transport(message) => Err(DeviceError::transport(message)),
            FakeReply::Delay(delay, inner) => {
                std::thread::sleep(delay);
                inner.resolve()
            }
            FakeReply::Blocked(latch, inner) => {
                if latch.wait(Duration::from_secs(30)) {
                    inner.resolve()
                } else {
                    Err(DeviceError::timeout(
                        "fake transport latch was never released",
                    ))
                }
            }
        }
    }
}

/// One recorded call.
#[derive(Debug, Clone, PartialEq)]
pub struct FakeCall {
    pub method: Method,
    pub member: String,
    pub params: Vec<(String, String)>,
    pub class: TimeoutClass,
}

#[derive(Default)]
struct Script {
    queued: HashMap<(Method, String), VecDeque<FakeReply>>,
    sticky: HashMap<(Method, String), FakeReply>,
    calls: Vec<FakeCall>,
}

/// Scripted [`AlpacaTransport`]. Per member: queued one-shot replies are
/// consumed first, in order; then the sticky reply repeats; an unscripted
/// member answers HTTP 400 `Unknown member`.
#[derive(Default)]
pub struct FakeAlpacaTransport {
    script: Mutex<Script>,
}

impl FakeAlpacaTransport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn shared(self) -> Arc<Self> {
        Arc::new(self)
    }

    fn script(&self) -> MutexGuard<'_, Script> {
        self.script.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn key(method: Method, member: &str) -> (Method, String) {
        (method, member.to_ascii_lowercase())
    }

    /// Sticky GET reply.
    pub fn on_get(&self, member: &str, reply: FakeReply) -> &Self {
        self.script()
            .sticky
            .insert(Self::key(Method::Get, member), reply);
        self
    }

    /// Sticky PUT reply.
    pub fn on_put(&self, member: &str, reply: FakeReply) -> &Self {
        self.script()
            .sticky
            .insert(Self::key(Method::Put, member), reply);
        self
    }

    /// One-shot GET replies, consumed in order before the sticky reply.
    pub fn queue_get(&self, member: &str, replies: impl IntoIterator<Item = FakeReply>) -> &Self {
        self.script()
            .queued
            .entry(Self::key(Method::Get, member))
            .or_default()
            .extend(replies);
        self
    }

    /// One-shot PUT replies, consumed in order before the sticky reply.
    pub fn queue_put(&self, member: &str, replies: impl IntoIterator<Item = FakeReply>) -> &Self {
        self.script()
            .queued
            .entry(Self::key(Method::Put, member))
            .or_default()
            .extend(replies);
        self
    }

    pub fn calls(&self) -> Vec<FakeCall> {
        self.script().calls.clone()
    }

    pub fn calls_to(&self, method: Method, member: &str) -> usize {
        let member = member.to_ascii_lowercase();
        self.script()
            .calls
            .iter()
            .filter(|call| call.method == method && call.member == member)
            .count()
    }

    pub fn clear_calls(&self) {
        self.script().calls.clear();
    }

    fn call(
        &self,
        method: Method,
        member: &str,
        params: &[(&str, String)],
        class: TimeoutClass,
    ) -> DeviceResult<serde_json::Value> {
        let key = Self::key(method, member);
        let reply = {
            let mut script = self.script();
            script.calls.push(FakeCall {
                method,
                member: key.1.clone(),
                params: params
                    .iter()
                    .map(|(name, value)| ((*name).to_string(), value.clone()))
                    .collect(),
                class,
            });
            let queued = script.queued.get_mut(&key).and_then(VecDeque::pop_front);
            queued
                .or_else(|| script.sticky.get(&key).cloned())
                .unwrap_or_else(|| FakeReply::http(400, format!("Unknown member '{}'", key.1)))
        };
        // Resolve outside the lock so a blocked call never stalls another
        // (an abort on a second agent, D4).
        reply.resolve()
    }
}

impl AlpacaTransport for FakeAlpacaTransport {
    fn get(
        &self,
        member: &str,
        query: &[(&str, String)],
        class: TimeoutClass,
    ) -> DeviceResult<serde_json::Value> {
        self.call(Method::Get, member, query, class)
    }

    fn put(
        &self,
        member: &str,
        form: &[(&str, String)],
        class: TimeoutClass,
    ) -> DeviceResult<serde_json::Value> {
        self.call(Method::Put, member, form, class)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::types::ErrorClass;
    use serde_json::json;

    #[test]
    fn fake_transport_replays_http_400_plain_text() {
        let fake = FakeAlpacaTransport::new();
        fake.on_get(
            "devicestate",
            FakeReply::http(400, "Unknown member: devicestate\r\n"),
        );
        let err = fake
            .get("DeviceState", &[], TimeoutClass::Ordinary)
            .unwrap_err();
        assert_eq!(err.http_status(), Some(400));
        assert_eq!(err.class, ErrorClass::InvalidValue);
        assert!(err.is_unknown_member(), "probes memoise it as unsupported");
        assert!(err.message.contains("Unknown member: devicestate"));

        // An unscripted member behaves like an older server that lacks it.
        let err = fake
            .get("covermoving", &[], TimeoutClass::Ordinary)
            .unwrap_err();
        assert_eq!(err.http_status(), Some(400));
        assert!(err.message.contains("covermoving"));

        fake.on_put("move", FakeReply::http(500, "motor driver fault"));
        let err = fake
            .put("move", &[("Position", "100".into())], TimeoutClass::Long)
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::Driver);
        assert_eq!(err.http_status(), Some(500));
    }

    #[test]
    fn queued_replies_come_first_then_the_sticky_reply_repeats() {
        let fake = FakeAlpacaTransport::new();
        fake.queue_get("slewing", [FakeReply::value(true), FakeReply::Timeout])
            .on_get("slewing", FakeReply::value(false));
        assert_eq!(
            fake.get("slewing", &[], TimeoutClass::Ordinary).unwrap(),
            json!(true)
        );
        assert_eq!(
            fake.get("slewing", &[], TimeoutClass::Ordinary)
                .unwrap_err()
                .class,
            ErrorClass::Timeout
        );
        for _ in 0..3 {
            assert_eq!(
                fake.get("slewing", &[], TimeoutClass::Ordinary).unwrap(),
                json!(false)
            );
        }
        fake.on_get("temperature", FakeReply::value_not_set());
        assert!(fake
            .get("temperature", &[], TimeoutClass::Ordinary)
            .unwrap_err()
            .is_not_set());
        fake.on_get("tempcomp", FakeReply::not_implemented());
        assert_eq!(
            fake.get("tempcomp", &[], TimeoutClass::Ordinary)
                .unwrap_err()
                .class,
            ErrorClass::Unsupported
        );
    }

    #[test]
    fn calls_are_recorded_with_exact_case_parameters() {
        let fake = FakeAlpacaTransport::new();
        fake.on_put("slewtocoordinatesasync", FakeReply::ok());
        fake.put(
            "slewtocoordinatesasync",
            &[
                ("RightAscension", "5.5".into()),
                ("Declination", "-5.25".into()),
            ],
            TimeoutClass::Long,
        )
        .unwrap();
        let calls = fake.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].method, Method::Put);
        assert_eq!(calls[0].class, TimeoutClass::Long);
        assert_eq!(calls[0].params[0], ("RightAscension".into(), "5.5".into()));
        assert_eq!(fake.calls_to(Method::Put, "SlewToCoordinatesAsync"), 1);
        fake.clear_calls();
        assert!(fake.calls().is_empty());
    }

    #[test]
    fn a_blocked_call_does_not_stall_other_calls() {
        let fake = Arc::new(FakeAlpacaTransport::new());
        let latch = Latch::new();
        fake.on_put(
            "move",
            FakeReply::Blocked(latch.clone(), Box::new(FakeReply::ok())),
        );
        fake.on_put("halt", FakeReply::ok());
        let mover = Arc::clone(&fake);
        let blocked = std::thread::spawn(move || {
            mover.put("move", &[("Position", "10".into())], TimeoutClass::Long)
        });
        while fake.calls_to(Method::Put, "move") == 0 {
            std::thread::yield_now();
        }
        let started = Instant::now();
        assert!(fake.put("halt", &[], TimeoutClass::Long).is_ok());
        assert!(started.elapsed() < Duration::from_secs(1));
        latch.release();
        assert!(blocked.join().unwrap().is_ok());
    }
}

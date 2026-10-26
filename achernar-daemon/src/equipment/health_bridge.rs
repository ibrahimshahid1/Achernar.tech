//! Device health from worker freshness (spec §2.5, §2.8, D8). Owner: WP1A-2.
//!
//! Three parts:
//!
//! 1. The health state machine `Connecting -> Ready -> Degraded -> Lost ->
//!    Disconnected`, as pure functions over [`HealthState`]:
//!    [`on_poll_ok`], [`on_poll_failed`] and [`on_stale`]. One slow read
//!    degrades a device; it never loses it (`lost_after_failures`
//!    consecutive failures, or no good poll for `lost_after_s`, do).
//! 2. The [`Watchdog`]: a worker blocked inside a driver call cannot report
//!    its own staleness, so the hub ticks every registered worker (every
//!    250 ms) and the watched worker applies [`on_stale`].
//! 3. The [`RegistryBridge`]: projects health and a JSON snapshot into the
//!    existing `DeviceRegistry` through `checked(kind, connection_id, ..)`,
//!    so `/status` and telemetry keep reading only the registry and never a
//!    driver (`health_tests.rs:181-205`). A projection from an old
//!    connection id can never revive a device: `checked` refuses it.
//!
//! The registry `status` keeps its two values (D8): `Ready` and `Degraded`
//! are `connected`; `Lost` and `Disconnected` are `disconnected`.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, RwLock, Weak};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde_json::Value;

use super::clock::duration_ns;
use super::types::{
    DeviceError, DeviceIdentity, DeviceKind, ErrorClass, FailureClass, HealthState, LinkState,
};
use crate::config::EquipmentConfig;
use crate::hardware::health::DeviceRegistry;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// Policy and transitions
// ---------------------------------------------------------------------------

/// `[equipment] lost_after_s` and `lost_after_failures`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HealthPolicy {
    pub lost_after: Duration,
    pub lost_after_failures: u32,
}

impl Default for HealthPolicy {
    fn default() -> Self {
        Self {
            lost_after: Duration::from_secs(10),
            lost_after_failures: 3,
        }
    }
}

impl HealthPolicy {
    pub fn from_config(config: &EquipmentConfig) -> Self {
        Self {
            lost_after: Duration::from_secs(config.lost_after_s.max(1)),
            lost_after_failures: config.lost_after_failures.max(1),
        }
    }

    /// The time-based loss threshold for a device polled every `interval`.
    ///
    /// `lost_after` is measured from the last good poll. For a kind whose
    /// cadence is close to `lost_after` (weather polls every 10 s) that would
    /// lose a healthy device between two ticks, so the threshold is never
    /// shorter than three intervals.
    pub fn lost_threshold(&self, interval: Duration) -> Duration {
        self.lost_after.max(interval.saturating_mul(3))
    }
}

/// What a health update changed.
#[derive(Debug, Clone, PartialEq)]
pub enum HealthChange {
    Unchanged,
    /// Back to (or first reached) `Ready`.
    Ready {
        from: LinkState,
    },
    Degraded {
        class: FailureClass,
        reason: Option<String>,
    },
    /// The loss threshold is reached; the worker tries its one reconnect
    /// (§2.6) before declaring `Lost`.
    LostCandidate {
        class: FailureClass,
        reason: Option<String>,
    },
    Lost {
        class: FailureClass,
        reason: Option<String>,
    },
}

/// The registry `failure_class` of a failed poll.
pub fn failure_class_of(err: &DeviceError) -> FailureClass {
    err.class.failure_class().unwrap_or(FailureClass::Driver)
}

/// A poll succeeded.
pub fn on_poll_ok(h: &mut HealthState, now_utc: DateTime<Utc>, now_mono: u64) -> HealthChange {
    h.consecutive_failures = 0;
    h.last_ok_at = Some(now_utc);
    h.last_ok_mono_ns = Some(now_mono);
    match h.state {
        // Loss is sticky for the connection: a fresh connect is needed.
        LinkState::Lost | LinkState::Disconnected => HealthChange::Unchanged,
        LinkState::Ready => HealthChange::Unchanged,
        from @ (LinkState::Connecting | LinkState::Degraded) => {
            h.state = LinkState::Ready;
            h.failure_class = None;
            h.reason = None;
            h.changed_mono_ns = now_mono;
            HealthChange::Ready { from }
        }
    }
}

/// A poll failed with `err`.
pub fn on_poll_failed(
    h: &mut HealthState,
    err: &DeviceError,
    policy: &HealthPolicy,
    interval: Duration,
    now_mono: u64,
) -> HealthChange {
    if matches!(h.state, LinkState::Lost | LinkState::Disconnected) {
        return HealthChange::Unchanged;
    }
    h.consecutive_failures = h.consecutive_failures.saturating_add(1);
    let class = failure_class_of(err);
    let reason = Some(err.to_string());
    h.failure_class = Some(class);
    h.reason = reason.clone();
    let since = h.last_ok_mono_ns.unwrap_or(h.changed_mono_ns);
    let silent = Duration::from_nanos(now_mono.saturating_sub(since));
    // The device itself says it is not connected: reconnect once now.
    let not_connected = err.class == ErrorClass::NotConnected;
    if not_connected
        || h.consecutive_failures >= policy.lost_after_failures
        || silent >= policy.lost_threshold(interval)
    {
        return HealthChange::LostCandidate { class, reason };
    }
    if h.state != LinkState::Degraded {
        h.state = LinkState::Degraded;
        h.changed_mono_ns = now_mono;
        return HealthChange::Degraded { class, reason };
    }
    HealthChange::Unchanged
}

/// Watchdog check: no completed good poll for a while.
///
/// - older than 2 x `interval`: `Ready -> Degraded` (timeout);
/// - older than [`HealthPolicy::lost_threshold`]: `Lost` (timeout).
pub fn on_stale(
    h: &mut HealthState,
    interval: Duration,
    policy: &HealthPolicy,
    now_mono: u64,
) -> HealthChange {
    if !matches!(h.state, LinkState::Ready | LinkState::Degraded) {
        return HealthChange::Unchanged;
    }
    let since = h.last_ok_mono_ns.unwrap_or(h.changed_mono_ns);
    let age = Duration::from_nanos(now_mono.saturating_sub(since));
    if age >= policy.lost_threshold(interval) {
        let reason = Some(format!("no successful poll for {:.1} s", age.as_secs_f64()));
        mark_lost(h, FailureClass::Timeout, reason.clone(), now_mono);
        return HealthChange::Lost {
            class: FailureClass::Timeout,
            reason,
        };
    }
    if h.state == LinkState::Ready && age >= interval.saturating_mul(2) {
        let reason = Some(format!(
            "newest data is {:.1} s old (cadence {:.1} s)",
            age.as_secs_f64(),
            interval.as_secs_f64()
        ));
        h.state = LinkState::Degraded;
        h.failure_class = Some(FailureClass::Timeout);
        h.reason = reason.clone();
        h.changed_mono_ns = now_mono;
        return HealthChange::Degraded {
            class: FailureClass::Timeout,
            reason,
        };
    }
    HealthChange::Unchanged
}

/// Set `Lost` with its class and reason.
pub fn mark_lost(h: &mut HealthState, class: FailureClass, reason: Option<String>, now_mono: u64) {
    h.state = LinkState::Lost;
    h.failure_class = Some(class);
    h.reason = reason;
    h.changed_mono_ns = now_mono;
}

// ---------------------------------------------------------------------------
// Watchdog
// ---------------------------------------------------------------------------

/// A worker the watchdog checks.
pub trait Watched: Send + Sync {
    /// Apply the staleness rules at `now_mono_ns`. Returns false once the
    /// worker has finished, so the watchdog can forget it.
    fn check(&self, now_mono_ns: u64) -> bool;
}

/// Ticks every registered worker. The hub's publisher calls [`tick`]
/// every 250 ms; tests call it directly with a manual clock.
///
/// [`tick`]: Watchdog::tick
#[derive(Default)]
pub struct Watchdog {
    entries: Mutex<Vec<Weak<dyn Watched>>>,
}

impl std::fmt::Debug for Watchdog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Watchdog")
            .field("entries", &lock(&self.entries).len())
            .finish()
    }
}

impl Watchdog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, watched: Weak<dyn Watched>) {
        lock(&self.entries).push(watched);
    }

    /// Check every live worker; returns how many are still watched.
    pub fn tick(&self, now_mono_ns: u64) -> usize {
        // Checks run outside the lock: a check may emit events.
        let live: Vec<Arc<dyn Watched>> = lock(&self.entries)
            .iter()
            .filter_map(Weak::upgrade)
            .collect();
        let finished: Vec<*const ()> = live
            .iter()
            .filter(|w| !w.check(now_mono_ns))
            .map(|w| Arc::as_ptr(w) as *const ())
            .collect();
        let mut entries = lock(&self.entries);
        entries.retain(|weak| match weak.upgrade() {
            Some(w) => !finished.contains(&(Arc::as_ptr(&w) as *const ())),
            None => false,
        });
        entries.len()
    }

    pub fn len(&self) -> usize {
        lock(&self.entries).len()
    }

    pub fn is_empty(&self) -> bool {
        lock(&self.entries).is_empty()
    }
}

// ---------------------------------------------------------------------------
// Registry projection
// ---------------------------------------------------------------------------

/// What a projector sees.
#[derive(Debug, Clone, Copy)]
pub struct ProjectionInput<'a> {
    pub kind: DeviceKind,
    pub identity: &'a DeviceIdentity,
    pub health: &'a HealthState,
    /// The typed snapshot as JSON.
    pub snapshot: &'a Value,
    pub age_ms: Option<u64>,
}

/// Builds the registry snapshot JSON for one kind. The spine (WP1B-1)
/// installs the frontend-shaped `/status` blocks (§2.13) here.
pub type Projector = Arc<dyn Fn(&ProjectionInput<'_>) -> Option<Value> + Send + Sync>;

/// The default projection: the typed snapshot plus the additive health
/// fields `state`, `failure_class`, `reason` and `age_ms` (§2.8).
pub fn default_projection(input: &ProjectionInput<'_>) -> Option<Value> {
    let mut value = input.snapshot.clone();
    if let Value::Object(map) = &mut value {
        map.insert(
            "state".into(),
            serde_json::to_value(input.health.state).unwrap_or(Value::Null),
        );
        map.insert(
            "failure_class".into(),
            serde_json::to_value(input.health.failure_class).unwrap_or(Value::Null),
        );
        map.insert(
            "reason".into(),
            input
                .health
                .reason
                .clone()
                .map(Value::String)
                .unwrap_or(Value::Null),
        );
        map.insert(
            "age_ms".into(),
            input.age_ms.map(Value::from).unwrap_or(Value::Null),
        );
    }
    Some(value)
}

/// Writes worker health into the existing `DeviceRegistry`.
pub struct RegistryBridge {
    registry: Arc<RwLock<DeviceRegistry>>,
    projectors: BTreeMap<DeviceKind, Projector>,
}

impl std::fmt::Debug for RegistryBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryBridge")
            .field("projectors", &self.projectors.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl RegistryBridge {
    pub fn new(registry: Arc<RwLock<DeviceRegistry>>) -> Self {
        Self {
            registry,
            projectors: BTreeMap::new(),
        }
    }

    pub fn with_projector(mut self, kind: DeviceKind, projector: Projector) -> Self {
        self.projectors.insert(kind, projector);
        self
    }

    /// Project one device. `Ready`/`Degraded` refresh the registry entry
    /// through `checked` (refused for a stale connection id); `Lost` and
    /// `Disconnected` mark it disconnected, guarded by the connection id.
    /// Returns true when the registry accepted the update.
    pub fn project(
        &self,
        identity: &DeviceIdentity,
        health: &HealthState,
        snapshot: &Value,
        age_ms: Option<u64>,
    ) -> bool {
        let kind = identity.kind;
        let id = identity.connection_id.to_string();
        let mut registry = self.registry.write().unwrap_or_else(|e| e.into_inner());
        match health.state {
            LinkState::Ready | LinkState::Degraded => {
                let input = ProjectionInput {
                    kind,
                    identity,
                    health,
                    snapshot,
                    age_ms,
                };
                let value = match self.projectors.get(&kind) {
                    Some(projector) => projector(&input),
                    None => default_projection(&input),
                };
                registry.checked(kind, &id, true, value)
            }
            LinkState::Lost | LinkState::Disconnected => {
                registry.disconnected(kind, Some(&id), health.reason.clone())
            }
            LinkState::Connecting => false,
        }
    }
}

/// Nanoseconds for a policy duration (re-exported for the worker).
pub fn interval_ns(interval: Duration) -> u64 {
    duration_ns(interval)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::testing::fake_handles::fake_identity;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const S: u64 = 1_000_000_000;

    #[test]
    fn one_slow_read_degrades_and_failures_lose() {
        let policy = HealthPolicy::default();
        let interval = Duration::from_secs(1);
        let mut h = HealthState::connecting(0);
        assert_eq!(
            on_poll_ok(&mut h, Utc::now(), S),
            HealthChange::Ready {
                from: LinkState::Connecting
            }
        );
        // A 4 s stall: the watchdog degrades the device, never loses it.
        assert!(matches!(
            on_stale(&mut h, interval, &policy, 5 * S),
            HealthChange::Degraded {
                class: FailureClass::Timeout,
                ..
            }
        ));
        assert_eq!(
            on_stale(&mut h, interval, &policy, 6 * S),
            HealthChange::Unchanged
        );
        // The slow read then completes.
        assert_eq!(
            on_poll_ok(&mut h, Utc::now(), 6 * S),
            HealthChange::Ready {
                from: LinkState::Degraded
            }
        );
        assert_eq!(h.state, LinkState::Ready);
        // N consecutive failures make it a loss candidate.
        let err = DeviceError::transport("connection refused");
        assert!(matches!(
            on_poll_failed(&mut h, &err, &policy, interval, 7 * S),
            HealthChange::Degraded { .. }
        ));
        assert_eq!(
            on_poll_failed(&mut h, &err, &policy, interval, 8 * S),
            HealthChange::Unchanged
        );
        assert!(matches!(
            on_poll_failed(&mut h, &err, &policy, interval, 9 * S),
            HealthChange::LostCandidate {
                class: FailureClass::Disconnected,
                ..
            }
        ));
        mark_lost(&mut h, FailureClass::Disconnected, None, 9 * S);
        // Loss is sticky: a late good poll does not revive it.
        assert_eq!(
            on_poll_ok(&mut h, Utc::now(), 10 * S),
            HealthChange::Unchanged
        );
        assert_eq!(h.state, LinkState::Lost);
    }

    #[test]
    fn staleness_loses_after_the_threshold_and_not_connected_reconnects_now() {
        let policy = HealthPolicy::default();
        let mut h = HealthState::ready(Utc::now(), 0);
        assert!(matches!(
            on_stale(&mut h, Duration::from_secs(1), &policy, 10 * S),
            HealthChange::Lost { .. }
        ));
        assert_eq!(h.state, LinkState::Lost);
        // A slow kind gets at least three intervals.
        let mut weather = HealthState::ready(Utc::now(), 0);
        let ten = Duration::from_secs(10);
        assert_eq!(policy.lost_threshold(ten), Duration::from_secs(30));
        assert!(matches!(
            on_stale(&mut weather, ten, &policy, 25 * S),
            HealthChange::Degraded { .. }
        ));
        assert_eq!(weather.state, LinkState::Degraded);
        let mut fresh = HealthState::ready(Utc::now(), 0);
        let err = DeviceError::not_connected("device reports Connected=false");
        assert!(matches!(
            on_poll_failed(&mut fresh, &err, &policy, Duration::from_secs(1), S),
            HealthChange::LostCandidate { .. }
        ));
    }

    struct Probe {
        checks: AtomicUsize,
        finish_after: usize,
    }

    impl Watched for Probe {
        fn check(&self, _now: u64) -> bool {
            self.checks.fetch_add(1, Ordering::SeqCst) + 1 < self.finish_after
        }
    }

    #[test]
    fn watchdog_ticks_live_workers_and_forgets_finished_ones() {
        let watchdog = Watchdog::new();
        let a = Arc::new(Probe {
            checks: AtomicUsize::new(0),
            finish_after: 2,
        });
        let b = Arc::new(Probe {
            checks: AtomicUsize::new(0),
            finish_after: 100,
        });
        watchdog.register(Arc::downgrade(&a) as Weak<dyn Watched>);
        watchdog.register(Arc::downgrade(&b) as Weak<dyn Watched>);
        assert_eq!(watchdog.tick(0), 2);
        assert_eq!(watchdog.tick(1), 1, "a finished on its second check");
        drop(b);
        assert_eq!(watchdog.tick(2), 0);
        assert!(watchdog.is_empty());
    }

    #[test]
    fn registry_projection_never_revives_an_old_connection() {
        let registry = Arc::new(RwLock::new(DeviceRegistry::default()));
        let bridge = RegistryBridge::new(Arc::clone(&registry));
        let identity = fake_identity(DeviceKind::Mount);
        let snapshot = serde_json::json!({"tracking": true});
        let ready = HealthState::ready(Utc::now(), 1);
        // Not registered as connected yet: the projection is refused.
        assert!(!bridge.project(&identity, &ready, &snapshot, Some(5)));
        registry.write().unwrap().connected(
            DeviceKind::Mount,
            identity.connection_id.to_string(),
            None,
        );
        assert!(bridge.project(&identity, &ready, &snapshot, Some(5)));
        let value = registry
            .read()
            .unwrap()
            .snapshot(DeviceKind::Mount)
            .unwrap();
        assert_eq!(value["tracking"], serde_json::json!(true));
        assert_eq!(value["state"], serde_json::json!("ready"));
        assert_eq!(value["age_ms"], serde_json::json!(5));
        // A different (old) connection id is ignored.
        let old = fake_identity(DeviceKind::Mount);
        assert!(!bridge.project(&old, &ready, &snapshot, None));
        // Lost maps to the registry's "disconnected" (D8).
        let mut lost = ready.clone();
        mark_lost(&mut lost, FailureClass::Timeout, Some("silent".into()), 2);
        assert!(bridge.project(&identity, &lost, &snapshot, None));
        let health = registry.read().unwrap().0[&DeviceKind::Mount].clone();
        assert!(!health.connected);
        assert_eq!(health.status, "disconnected");
        assert_eq!(health.last_error.as_deref(), Some("silent"));
        // And a late projection from the lost connection cannot revive it.
        assert!(!bridge.project(&identity, &ready, &snapshot, None));
        // A custom projector shapes the status block.
        let custom = RegistryBridge::new(Arc::clone(&registry)).with_projector(
            DeviceKind::Mount,
            Arc::new(|input: &ProjectionInput<'_>| {
                Some(serde_json::json!({"name": input.identity.name}))
            }),
        );
        registry.write().unwrap().connected(
            DeviceKind::Mount,
            identity.connection_id.to_string(),
            None,
        );
        assert!(custom.project(&identity, &ready, &snapshot, None));
        assert_eq!(
            registry
                .read()
                .unwrap()
                .snapshot(DeviceKind::Mount)
                .unwrap(),
            serde_json::json!({"name": identity.name})
        );
    }
}

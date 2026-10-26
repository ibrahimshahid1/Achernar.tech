//! Resume blocks (spec §2.9 "Resume blocks", H22, v6 §8.2.1). Owner:
//! WP1A-2.
//!
//! A [`ResumeBlock`] makes "unknown position or an incomplete handshake
//! blocks automatic exposure resumption" enforceable. While any block is
//! active the gate refuses automatic sources (HTTP 423); operator commands
//! run, and their frames are flagged `resume_block_active`.
//!
//! Lifecycle:
//! - **raise**: one active block per [`ResumeBlockKind`] (an outcome-unknown
//!   block is per device and operation). Raising an already active kind
//!   returns the existing block, so two producers reporting the same fault
//!   (the worker's event and the executor's outcome) raise it once.
//! - **clear**: `safety_*` blocks clear only on a confirmed safe reading under
//!   the resume policy ([`ResumeBlocks::clear_safety`]); every other kind is
//!   cleared by the operator ([`ResumeBlocks::clear`]), which the gate
//!   audits.
//! - **history**: cleared blocks are kept (bounded) so an exposure window can
//!   tell whether a block was active at any time during the exposure.
//!
//! This type only holds state; the hub emits `resume_blocked` and
//! `resume_cleared` for every change.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::watch;
use uuid::Uuid;

use super::snapshot::{ResumeBlock, ResumeBlockKind};
use super::traits::Clock;
use super::types::{CommandId, CommandSource, DeviceKind};

/// Cleared blocks kept for window queries.
const HISTORY_LIMIT: usize = 512;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// What to raise.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockRequest {
    pub kind: ResumeBlockKind,
    pub device_kind: Option<DeviceKind>,
    pub connection_id: Option<Uuid>,
    pub command_id: Option<CommandId>,
    pub reason: String,
}

impl BlockRequest {
    pub fn new(kind: ResumeBlockKind, reason: impl Into<String>) -> Self {
        Self {
            kind,
            device_kind: None,
            connection_id: None,
            command_id: None,
            reason: reason.into(),
        }
    }

    pub fn for_device(mut self, kind: DeviceKind, connection_id: Option<Uuid>) -> Self {
        self.device_kind = Some(kind);
        self.connection_id = connection_id;
        self
    }

    pub fn with_command(mut self, command_id: Option<CommandId>) -> Self {
        self.command_id = command_id;
        self
    }
}

/// Result of [`ResumeBlocks::raise`].
#[derive(Debug, Clone, PartialEq)]
pub enum Raised {
    New(ResumeBlock),
    /// The kind was already blocked; nothing changed.
    Existing(ResumeBlock),
}

impl Raised {
    pub fn block(&self) -> &ResumeBlock {
        match self {
            Raised::New(block) | Raised::Existing(block) => block,
        }
    }

    pub fn into_block(self) -> ResumeBlock {
        match self {
            Raised::New(block) | Raised::Existing(block) => block,
        }
    }

    pub fn is_new(&self) -> bool {
        matches!(self, Raised::New(_))
    }
}

/// Why a clear was refused.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "error", rename_all = "snake_case")]
pub enum ClearError {
    NotFound {
        id: Uuid,
    },
    /// `safety_*` blocks clear only on a confirmed safe reading (§3.10).
    NotOperatorClearable {
        kind: ResumeBlockKind,
    },
}

impl std::fmt::Display for ClearError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClearError::NotFound { id } => write!(f, "no active resume block {id}"),
            ClearError::NotOperatorClearable { kind } => write!(
                f,
                "{kind:?} clears only on a confirmed safe reading under the resume policy"
            ),
        }
    }
}

impl std::error::Error for ClearError {}

/// A block that was cleared, kept for window queries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClearedBlock {
    pub block: ResumeBlock,
    pub cleared_at: DateTime<Utc>,
    pub cleared_mono_ns: u64,
    pub by: CommandSource,
}

#[derive(Debug, Default)]
struct Inner {
    active: Vec<ResumeBlock>,
    history: VecDeque<ClearedBlock>,
}

/// The resume-block registry and its watch.
pub struct ResumeBlocks {
    clock: Arc<dyn Clock>,
    inner: Mutex<Inner>,
    tx: watch::Sender<Vec<ResumeBlock>>,
}

impl std::fmt::Debug for ResumeBlocks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResumeBlocks")
            .field("active", &lock(&self.inner).active)
            .finish()
    }
}

impl ResumeBlocks {
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        let (tx, _) = watch::channel(Vec::new());
        Self {
            clock,
            inner: Mutex::new(Inner::default()),
            tx,
        }
    }

    fn publish(&self, inner: &Inner) {
        self.tx.send_replace(inner.active.clone());
    }

    /// Raise a block, or return the active one of the same kind.
    pub fn raise(&self, request: BlockRequest) -> Raised {
        let mut inner = lock(&self.inner);
        if let Some(existing) = inner.active.iter().find(|b| b.kind == request.kind) {
            return Raised::Existing(existing.clone());
        }
        let block = ResumeBlock {
            id: Uuid::new_v4(),
            kind: request.kind,
            raised_at: self.clock.now_utc(),
            raised_mono_ns: self.clock.now_mono_ns(),
            device_kind: request.device_kind,
            connection_id: request.connection_id,
            command_id: request.command_id,
            reason: request.reason,
        };
        inner.active.push(block.clone());
        self.publish(&inner);
        Raised::New(block)
    }

    fn take(inner: &mut Inner, index: usize, by: CommandSource, clock: &dyn Clock) -> ResumeBlock {
        let block = inner.active.remove(index);
        inner.history.push_back(ClearedBlock {
            block: block.clone(),
            cleared_at: clock.now_utc(),
            cleared_mono_ns: clock.now_mono_ns(),
            by,
        });
        while inner.history.len() > HISTORY_LIMIT {
            inner.history.pop_front();
        }
        block
    }

    /// The operator path (`POST equipment/resume-blocks/clear {id}`); the
    /// gate audits it. `safety_*` blocks are refused.
    pub fn clear(&self, id: Uuid, by: &CommandSource) -> Result<ResumeBlock, ClearError> {
        let mut inner = lock(&self.inner);
        let Some(index) = inner.active.iter().position(|b| b.id == id) else {
            return Err(ClearError::NotFound { id });
        };
        if !inner.active[index].kind.operator_clears() {
            return Err(ClearError::NotOperatorClearable {
                kind: inner.active[index].kind.clone(),
            });
        }
        let block = Self::take(&mut inner, index, by.clone(), self.clock.as_ref());
        self.publish(&inner);
        Ok(block)
    }

    /// The safety supervisor's path for one block it raised itself (its
    /// `safety_unknown` block once safety reads safe or none again). Any
    /// kind; never exposed to operators. `None` when it was not active.
    pub fn clear_by_owner(&self, id: Uuid, by: &CommandSource) -> Option<ResumeBlock> {
        let mut inner = lock(&self.inner);
        let index = inner.active.iter().position(|b| b.id == id)?;
        let block = Self::take(&mut inner, index, by.clone(), self.clock.as_ref());
        self.publish(&inner);
        Some(block)
    }

    /// The safety-policy path: a confirmed safe reading under the resume
    /// policy clears every `safety_*` block. Returns what was cleared.
    pub fn clear_safety(&self, by: &CommandSource) -> Vec<ResumeBlock> {
        let mut inner = lock(&self.inner);
        let mut cleared = Vec::new();
        while let Some(index) = inner.active.iter().position(|b| !b.kind.operator_clears()) {
            cleared.push(Self::take(
                &mut inner,
                index,
                by.clone(),
                self.clock.as_ref(),
            ));
        }
        if !cleared.is_empty() {
            self.publish(&inner);
        }
        cleared
    }

    pub fn active(&self) -> Vec<ResumeBlock> {
        lock(&self.inner).active.clone()
    }

    pub fn is_blocked(&self) -> bool {
        !lock(&self.inner).active.is_empty()
    }

    pub fn subscribe(&self) -> watch::Receiver<Vec<ResumeBlock>> {
        self.tx.subscribe()
    }

    pub fn history(&self) -> Vec<ClearedBlock> {
        lock(&self.inner).history.iter().cloned().collect()
    }

    /// True when any block was active at some instant of `[start, end]`
    /// (the window's `resume_block_active`).
    pub fn active_during(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> bool {
        let inner = lock(&self.inner);
        inner.active.iter().any(|b| b.raised_at <= end)
            || inner
                .history
                .iter()
                .any(|c| c.block.raised_at <= end && c.cleared_at >= start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::ops::OpKind;
    use crate::equipment::testing::manual_clock::ManualClock;

    fn operator() -> CommandSource {
        CommandSource::Operator { overriding: false }
    }

    #[test]
    fn resume_block_lifecycle_raise_dedupe_clear_and_history() {
        let clock = Arc::new(ManualClock::new());
        let blocks = ResumeBlocks::new(clock.clone());
        let mut rx = blocks.subscribe();
        assert!(!blocks.is_blocked());

        let wheel = BlockRequest::new(
            ResumeBlockKind::OutcomeUnknown {
                device: DeviceKind::FilterWheel,
                op: OpKind::FilterMove,
            },
            "wheel timeout",
        )
        .for_device(DeviceKind::FilterWheel, None);
        let first = blocks.raise(wheel.clone());
        assert!(first.is_new());
        assert!(rx.has_changed().unwrap());
        assert_eq!(rx.borrow_and_update().len(), 1);
        // The same fault reported twice raises one block.
        let again = blocks.raise(wheel);
        assert!(!again.is_new());
        assert_eq!(again.block().id, first.block().id);
        // A different device or op is a different block.
        blocks.raise(BlockRequest::new(
            ResumeBlockKind::OutcomeUnknown {
                device: DeviceKind::Mount,
                op: OpKind::Slew,
            },
            "lost mid-slew",
        ));
        let unsafe_block = blocks
            .raise(BlockRequest::new(ResumeBlockKind::SafetyUnsafe, "rain"))
            .into_block();
        assert_eq!(blocks.active().len(), 3);

        clock.advance_ms(1000);
        let cleared = blocks.clear(first.block().id, &operator()).unwrap();
        assert_eq!(cleared.id, first.block().id);
        assert_eq!(
            blocks.clear(first.block().id, &operator()),
            Err(ClearError::NotFound {
                id: first.block().id
            })
        );
        // The operator cannot clear a safety block; only a confirmed safe
        // reading can.
        assert!(matches!(
            blocks.clear(unsafe_block.id, &operator()),
            Err(ClearError::NotOperatorClearable { .. })
        ));
        let safety = blocks.clear_safety(&CommandSource::Safety);
        assert_eq!(safety.len(), 1);
        assert_eq!(safety[0].kind, ResumeBlockKind::SafetyUnsafe);
        assert_eq!(blocks.active().len(), 1);
        assert_eq!(blocks.history().len(), 2);
        assert_eq!(blocks.history()[0].by, operator());
    }

    #[test]
    fn active_during_covers_cleared_and_open_blocks() {
        let clock = Arc::new(ManualClock::new());
        let t0 = clock.now_utc();
        let blocks = ResumeBlocks::new(clock.clone());
        clock.advance_ms(10_000);
        let block = blocks
            .raise(BlockRequest::new(ResumeBlockKind::Estop, "e-stop"))
            .into_block();
        clock.advance_ms(10_000);
        blocks.clear(block.id, &operator()).unwrap();
        let s = |n: i64| t0 + chrono::Duration::seconds(n);
        assert!(!blocks.active_during(s(0), s(5)), "before the block");
        assert!(blocks.active_during(s(5), s(12)), "overlaps the raise");
        assert!(blocks.active_during(s(15), s(25)), "overlaps the clear");
        assert!(!blocks.active_during(s(21), s(30)), "after the clear");
        clock.advance_ms(10_000);
        blocks.raise(BlockRequest::new(ResumeBlockKind::MountLost, "lost"));
        assert!(blocks.active_during(s(29), s(31)));
        assert!(!blocks.active_during(s(22), s(29)));
    }
}

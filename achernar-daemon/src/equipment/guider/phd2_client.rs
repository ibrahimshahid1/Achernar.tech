//! PHD2 event-server client (spec §3.5 G-a; WP1A-7).
//!
//! One TCP connection carries both directions: Achernar's requests, PHD2's
//! replies, and PHD2's unsolicited events. A reader task frames lines
//! ([`LineFramer`]) and routes each one: replies go to the waiting request
//! (by integer id), events go to the service's channel stamped with the
//! daemon receive time. A writer task sends requests, so callers never hold
//! a lock across an `.await`.
//!
//! Every wait is measured on the injected [`Clock`] (D22): the client checks
//! its deadlines on a short real-time tick, so a `ManualClock` test drives
//! timeouts deterministically by advancing time.
//!
//! The connection is lost on EOF or a socket error (G-a). Every pending
//! request then fails with `Disconnected`, and the event channel receives
//! [`ClientEvent::Closed`]. Tier 1 has no automatic reconnect.
//!
//! [`Phd2AbortPath`] is the out-of-band stop (D4, D19): a separate,
//! short-lived connection that sends `stop_capture`, so an e-stop never
//! queues behind the service. It connects first to the address the main
//! connection reached ([`Phd2Client::peer_addr`]), then to every resolved
//! address, IPv4 first: PHD2's event server listens on IPv4 only, while
//! `localhost` usually resolves to `::1` first.

use std::collections::HashMap;
use std::future::Future;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream as StdTcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;

use super::phd2_proto::{
    method, parse_line, LineFramer, Phd2Event, Phd2Message, RpcError, RpcRequest,
};
use crate::equipment::clock::duration_ns;
use crate::equipment::traits::{AbortPath, Clock};
use crate::equipment::types::{DeviceError, DeviceResult};

/// `host:port` for messages, with an IPv6 literal in brackets.
fn display_peer(host: &str, port: u16) -> String {
    if host.contains(':') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

/// Real-time granularity of every clock-measured wait.
pub const TICK: Duration = Duration::from_millis(20);

/// Wait for `fut`, but give up once the clock passes `deadline_mono_ns`.
/// A future that is already ready wins even after the deadline.
pub async fn until_deadline<F: Future>(
    fut: F,
    clock: &dyn Clock,
    deadline_mono_ns: u64,
) -> Option<F::Output> {
    tokio::pin!(fut);
    loop {
        tokio::select! {
            biased;
            out = &mut fut => return Some(out),
            _ = tokio::time::sleep(TICK) => {}
        }
        if clock.now_mono_ns() >= deadline_mono_ns {
            return None;
        }
    }
}

/// Per-method timeouts. Most methods answer at once (guide, dither and
/// stop_capture return before the work completes; completion arrives as
/// events). Connecting equipment or switching a profile can take PHD2 much
/// longer.
pub fn method_timeout(name: &str, base: Duration) -> Duration {
    match name {
        method::SET_CONNECTED | method::SET_PROFILE => base.max(Duration::from_secs(60)),
        method::FIND_STAR | method::FLIP_CALIBRATION => base.max(Duration::from_secs(30)),
        _ => base,
    }
}

/// What the reader task delivers to the service.
#[derive(Debug, Clone, PartialEq)]
pub enum ClientEvent {
    Event {
        event: Box<Phd2Event>,
        received_at: DateTime<Utc>,
        received_mono_ns: u64,
    },
    /// A line that was not JSON, or JSON that was neither event nor reply.
    Malformed { line: String, error: String },
    /// EOF, a socket error, or a local close.
    Closed { reason: String, local: bool },
}

type Pending = HashMap<u64, oneshot::Sender<Result<Value, RpcError>>>;

struct Inner {
    clock: Arc<dyn Clock>,
    base_timeout: Duration,
    next_id: AtomicU64,
    pending: Mutex<Pending>,
    writer: mpsc::UnboundedSender<String>,
    closed: AtomicBool,
    local_close: AtomicBool,
    close_reason: Mutex<Option<String>>,
    /// Set once the connection is closed; both tasks watch it.
    shutdown: watch::Sender<bool>,
    peer: String,
    /// The address the TCP connect actually reached.
    peer_addr: Option<SocketAddr>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

impl Inner {
    /// Mark the connection closed and fail every pending request.
    fn mark_closed(&self, reason: &str) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        *lock(&self.close_reason) = Some(reason.to_string());
        let pending: Vec<_> = lock(&self.pending).drain().collect();
        drop(pending); // dropping the senders fails the waiters
        self.shutdown.send_replace(true);
    }

    fn disconnected_error(&self) -> DeviceError {
        let reason = lock(&self.close_reason)
            .clone()
            .unwrap_or_else(|| "connection closed".into());
        DeviceError::disconnected(format!("PHD2 at {} is not connected: {reason}", self.peer))
    }
}

/// A connected PHD2 client. Cheap to clone.
#[derive(Clone)]
pub struct Phd2Client {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Phd2Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Phd2Client")
            .field("peer", &self.inner.peer)
            .field("closed", &self.is_closed())
            .finish()
    }
}

/// The reader and writer tasks of one connection.
#[derive(Debug)]
pub struct ClientTasks {
    pub reader: JoinHandle<()>,
    pub writer: JoinHandle<()>,
}

impl ClientTasks {
    pub fn abort(&self) {
        self.reader.abort();
        self.writer.abort();
    }
}

impl Drop for ClientTasks {
    /// The tasks hold the socket; they never outlive their owner, even when
    /// the owner is cancelled mid-connect or its task is aborted.
    fn drop(&mut self) {
        self.abort();
    }
}

impl Phd2Client {
    /// Open the TCP connection and start the reader and writer tasks. Must
    /// run inside a tokio runtime. `connect_timeout` bounds the TCP connect
    /// only (real time: it is network I/O, not a protocol rule).
    pub async fn connect(
        host: &str,
        port: u16,
        connect_timeout: Duration,
        rpc_timeout: Duration,
        clock: Arc<dyn Clock>,
    ) -> DeviceResult<(Self, mpsc::UnboundedReceiver<ClientEvent>, ClientTasks)> {
        let peer = display_peer(host, port);
        let stream =
            match tokio::time::timeout(connect_timeout, TcpStream::connect((host, port))).await {
                Ok(Ok(stream)) => stream,
                Ok(Err(err)) => {
                    return Err(DeviceError::disconnected(format!(
                        "cannot reach PHD2 at {peer}: {err}"
                    )))
                }
                Err(_) => {
                    return Err(DeviceError::timeout(format!(
                        "PHD2 at {peer} did not accept a connection within {connect_timeout:?}"
                    )))
                }
            };
        let _ = stream.set_nodelay(true);
        // `connect` tried every resolved address; keep the one that worked
        // so the abort path does not have to guess (`localhost` → `::1`).
        let peer_addr = stream.peer_addr().ok();
        let (read_half, write_half) = stream.into_split();
        let (line_tx, line_rx) = mpsc::unbounded_channel();
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let (shutdown, _) = watch::channel(false);
        let inner = Arc::new(Inner {
            clock,
            base_timeout: rpc_timeout,
            next_id: AtomicU64::new(1),
            pending: Mutex::new(HashMap::new()),
            writer: line_tx,
            closed: AtomicBool::new(false),
            local_close: AtomicBool::new(false),
            close_reason: Mutex::new(None),
            shutdown,
            peer,
            peer_addr,
        });
        let reader = tokio::spawn(read_loop(read_half, Arc::clone(&inner), event_tx));
        let writer = tokio::spawn(write_loop(write_half, Arc::clone(&inner), line_rx));
        Ok((Self { inner }, event_rx, ClientTasks { reader, writer }))
    }

    pub fn peer(&self) -> &str {
        &self.inner.peer
    }

    /// The socket address the connection reached (one of the addresses
    /// `host` resolved to).
    pub fn peer_addr(&self) -> Option<SocketAddr> {
        self.inner.peer_addr
    }

    pub fn is_closed(&self) -> bool {
        self.inner.closed.load(Ordering::Acquire)
    }

    /// Close the connection from our side. Never sends `set_connected
    /// [false]` (D16): PHD2 and its equipment are left as they are.
    pub fn close(&self) {
        self.inner.local_close.store(true, Ordering::Release);
        self.inner.mark_closed("closed by Achernar");
    }

    /// Call with the per-method timeout.
    pub async fn call(&self, name: &str, params: Option<Value>) -> DeviceResult<Value> {
        let timeout = method_timeout(name, self.inner.base_timeout);
        self.call_with_timeout(name, params, timeout).await
    }

    /// Call, failing with `Timeout` once the clock passes `timeout`.
    pub async fn call_with_timeout(
        &self,
        name: &str,
        params: Option<Value>,
        timeout: Duration,
    ) -> DeviceResult<Value> {
        let inner = &self.inner;
        if self.is_closed() {
            return Err(inner.disconnected_error());
        }
        let id = inner.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        lock(&inner.pending).insert(id, tx);
        // A close between the check above and the insert would leave this
        // waiter unresolved; re-check after inserting.
        if self.is_closed() {
            lock(&inner.pending).remove(&id);
            return Err(inner.disconnected_error());
        }
        let line = RpcRequest::new(name, params, id).to_line();
        if inner.writer.send(line).is_err() {
            lock(&inner.pending).remove(&id);
            return Err(inner.disconnected_error());
        }
        let deadline = inner
            .clock
            .now_mono_ns()
            .saturating_add(duration_ns(timeout));
        match until_deadline(rx, inner.clock.as_ref(), deadline).await {
            Some(Ok(Ok(value))) => Ok(value),
            Some(Ok(Err(rpc))) => Err(rpc.to_device_error(name)),
            Some(Err(_)) => Err(inner.disconnected_error()),
            None => {
                lock(&inner.pending).remove(&id);
                Err(DeviceError::timeout(format!(
                    "PHD2 did not answer {name} within {timeout:?}"
                )))
            }
        }
    }

    /// Requests still waiting for a reply.
    pub fn pending_requests(&self) -> usize {
        lock(&self.inner.pending).len()
    }
}

fn dispatch(inner: &Inner, events: &mpsc::UnboundedSender<ClientEvent>, line: &str) {
    let received_at = inner.clock.now_utc();
    let received_mono_ns = inner.clock.now_mono_ns();
    match parse_line(line) {
        Ok(Phd2Message::Event(event)) => {
            let _ = events.send(ClientEvent::Event {
                event: Box::new(event),
                received_at,
                received_mono_ns,
            });
        }
        Ok(Phd2Message::Reply(reply)) => {
            let waiter = reply.id.and_then(|id| lock(&inner.pending).remove(&id));
            match waiter {
                Some(waiter) => {
                    let _ = waiter.send(reply.outcome);
                }
                None => tracing::debug!(
                    peer = %inner.peer,
                    id = ?reply.id,
                    "PHD2 reply for an unknown or expired request"
                ),
            }
        }
        Ok(Phd2Message::Other(value)) => {
            let _ = events.send(ClientEvent::Malformed {
                line: line.to_string(),
                error: format!("neither an event nor a reply: {value}"),
            });
        }
        Err(err) => {
            let _ = events.send(ClientEvent::Malformed {
                line: line.to_string(),
                error: err.message,
            });
        }
    }
}

async fn read_loop(
    mut read_half: OwnedReadHalf,
    inner: Arc<Inner>,
    events: mpsc::UnboundedSender<ClientEvent>,
) {
    let mut framer = LineFramer::new();
    let mut buf = vec![0u8; 8192];
    let mut shutdown = inner.shutdown.subscribe();
    let reason = loop {
        let read = tokio::select! {
            read = read_half.read(&mut buf) => read,
            _ = shutdown.wait_for(|closed| *closed) => break "closed by Achernar".to_string(),
        };
        match read {
            Ok(0) => break "PHD2 closed the connection".to_string(),
            Ok(n) => {
                for line in framer.push(&buf[..n]) {
                    dispatch(&inner, &events, &line);
                }
            }
            Err(err) => break format!("socket error: {err}"),
        }
    };
    if let Some(line) = framer.finish() {
        dispatch(&inner, &events, &line);
    }
    inner.mark_closed(&reason);
    let local = inner.local_close.load(Ordering::Acquire);
    let reason = lock(&inner.close_reason).clone().unwrap_or(reason);
    let _ = events.send(ClientEvent::Closed { reason, local });
}

async fn write_loop(
    mut write_half: OwnedWriteHalf,
    inner: Arc<Inner>,
    mut lines: mpsc::UnboundedReceiver<String>,
) {
    let mut shutdown = inner.shutdown.subscribe();
    loop {
        let line = tokio::select! {
            line = lines.recv() => line,
            _ = shutdown.wait_for(|closed| *closed) => None,
        };
        let Some(line) = line else { break };
        if let Err(err) = write_half.write_all(line.as_bytes()).await {
            inner.mark_closed(&format!("socket error: {err}"));
            break;
        }
    }
    let _ = write_half.shutdown().await;
}

// ---------------------------------------------------------------------------
// Out-of-band stop
// ---------------------------------------------------------------------------

/// Sends `stop_capture` over its own short-lived connection (D4, D19; e-stop
/// "guider: stop_capture", §2.9). PHD2 serves several clients at once, so
/// this never waits for the service. Runs on a blocking thread.
#[derive(Debug, Clone)]
pub struct Phd2AbortPath {
    host: String,
    port: u16,
    timeout: Duration,
    /// The address the main connection reached; tried first.
    peer_addr: Option<SocketAddr>,
}

/// Connect order for the abort path: `known` (the address the main
/// connection reached), then the resolved addresses with IPv4 first and the
/// resolver's order kept within each family, without repeats.
fn abort_candidates(
    known: Option<SocketAddr>,
    resolved: impl IntoIterator<Item = SocketAddr>,
) -> Vec<SocketAddr> {
    let mut resolved: Vec<SocketAddr> = resolved.into_iter().collect();
    resolved.sort_by_key(SocketAddr::is_ipv6);
    let mut out: Vec<SocketAddr> = Vec::new();
    for addr in known.into_iter().chain(resolved) {
        if !out.contains(&addr) {
            out.push(addr);
        }
    }
    out
}

impl Phd2AbortPath {
    pub fn new(host: impl Into<String>, port: u16, timeout: Duration) -> Self {
        Self {
            host: host.into(),
            port,
            timeout,
            peer_addr: None,
        }
    }

    /// Try `addr` (normally [`Phd2Client::peer_addr`]) before resolving the
    /// host again.
    pub fn with_peer_addr(mut self, addr: Option<SocketAddr>) -> Self {
        self.peer_addr = addr;
        self
    }

    /// Connect within `deadline`: the known address first, then every
    /// resolved address, IPv4 first. Each attempt gets an equal share of
    /// the time left, so one address that drops packets cannot use it all.
    fn open(&self, peer: &str, deadline: Instant) -> DeviceResult<StdTcpStream> {
        fn attempt(addr: SocketAddr, budget: Duration) -> Result<StdTcpStream, String> {
            if budget.is_zero() {
                return Err(format!("{addr}: no time left"));
            }
            StdTcpStream::connect_timeout(&addr, budget).map_err(|e| format!("{addr}: {e}"))
        }
        let left = || deadline.saturating_duration_since(Instant::now());
        let mut failures: Vec<String> = Vec::new();
        if let Some(addr) = self.peer_addr {
            // Leave half the budget for the other addresses.
            match attempt(addr, left() / 2) {
                Ok(stream) => return Ok(stream),
                Err(e) => failures.push(e),
            }
        }
        match (self.host.as_str(), self.port).to_socket_addrs() {
            Ok(resolved) => {
                // The known address was tried above.
                let candidates: Vec<SocketAddr> = abort_candidates(self.peer_addr, resolved)
                    .into_iter()
                    .filter(|a| Some(*a) != self.peer_addr)
                    .collect();
                for (i, addr) in candidates.iter().enumerate() {
                    let n = u32::try_from(candidates.len() - i).unwrap_or(u32::MAX);
                    match attempt(*addr, left() / n.max(1)) {
                        Ok(stream) => return Ok(stream),
                        Err(e) => failures.push(e),
                    }
                }
            }
            Err(e) => failures.push(format!("cannot resolve: {e}")),
        }
        if failures.is_empty() {
            failures.push("no address".into());
        }
        Err(DeviceError::disconnected(format!(
            "cannot reach PHD2 at {peer}: {}",
            failures.join("; ")
        )))
    }

    fn stop_capture(&self) -> DeviceResult<()> {
        let deadline = Instant::now() + self.timeout;
        let peer = display_peer(&self.host, self.port);
        let mut stream = self.open(&peer, deadline)?;
        let _ = stream.set_nodelay(true);
        let io_err = |e: std::io::Error| DeviceError::transport(format!("PHD2 at {peer}: {e}"));
        stream
            .set_write_timeout(Some(self.timeout))
            .map_err(io_err)?;
        stream
            .write_all(
                RpcRequest::new(method::STOP_CAPTURE, None, 1)
                    .to_line()
                    .as_bytes(),
            )
            .map_err(io_err)?;
        let mut framer = LineFramer::new();
        let mut buf = [0u8; 4096];
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(DeviceError::timeout(format!(
                    "PHD2 at {peer} did not confirm stop_capture within {:?}",
                    self.timeout
                )));
            }
            stream.set_read_timeout(Some(left)).map_err(io_err)?;
            let n = match stream.read(&mut buf) {
                Ok(0) => {
                    return Err(DeviceError::disconnected(format!(
                        "PHD2 at {peer} closed the connection before confirming stop_capture"
                    )))
                }
                Ok(n) => n,
                Err(e)
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    ) =>
                {
                    continue
                }
                Err(e) => return Err(io_err(e)),
            };
            for line in framer.push(&buf[..n]) {
                if let Ok(Phd2Message::Reply(reply)) = parse_line(&line) {
                    if reply.id == Some(1) {
                        return reply
                            .outcome
                            .map(|_| ())
                            .map_err(|e| e.to_device_error(method::STOP_CAPTURE));
                    }
                }
            }
        }
    }
}

impl AbortPath for Phd2AbortPath {
    fn abort(&self) -> DeviceResult<()> {
        self.stop_capture()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::testing::phd2_mock::Phd2Mock;
    use crate::equipment::types::ErrorClass;
    use serde_json::json;

    async fn client_for(
        mock: &Phd2Mock,
        clock: Arc<dyn Clock>,
    ) -> (
        Phd2Client,
        mpsc::UnboundedReceiver<ClientEvent>,
        ClientTasks,
    ) {
        Phd2Client::connect(
            "127.0.0.1",
            mock.port(),
            Duration::from_secs(5),
            Duration::from_secs(10),
            clock,
        )
        .await
        .unwrap()
    }

    async fn next_event(rx: &mut mpsc::UnboundedReceiver<ClientEvent>) -> ClientEvent {
        tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("event in time")
            .expect("channel open")
    }

    #[tokio::test]
    async fn replies_reach_their_callers_and_events_reach_the_channel() {
        let mock = Phd2Mock::start().await;
        let clock: Arc<dyn Clock> = Arc::new(ManualClock::new());
        let (client, mut rx, tasks) = client_for(&mock, clock).await;
        // The connect burst arrives as events.
        let ClientEvent::Event { event, .. } = next_event(&mut rx).await else {
            panic!("expected Version");
        };
        assert_eq!(event.name, "Version");
        let ClientEvent::Event { event, .. } = next_event(&mut rx).await else {
            panic!("expected AppState");
        };
        assert_eq!(event.name, "AppState");
        assert_eq!(
            client.call(method::GET_APP_STATE, None).await.unwrap(),
            json!("Stopped")
        );
        assert_eq!(
            client.call(method::GET_PIXEL_SCALE, None).await.unwrap(),
            json!(1.5)
        );
        // Concurrent requests are matched by id, not by order.
        let (a, b) = tokio::join!(
            client.call(method::GET_EXPOSURE, None),
            client.call(method::GET_DEC_GUIDE_MODE, None)
        );
        assert_eq!(a.unwrap(), json!(2000));
        assert_eq!(b.unwrap(), json!("Auto"));
        let ids: Vec<u64> = mock.requests().iter().map(|r| r.id).collect();
        assert!(
            ids.windows(2).all(|w| w[0] < w[1]),
            "integer ids increase: {ids:?}"
        );
        client.close();
        tasks.abort();
    }

    #[tokio::test]
    async fn rpc_errors_and_unsupported_methods_are_classified() {
        let mock = Phd2Mock::start().await;
        mock.with_state(|s| {
            s.unsupported_methods.insert("get_algo_param_names".into());
            s.dither_error = Some((1, "cannot dither if not guiding".into()));
        });
        let clock: Arc<dyn Clock> = Arc::new(ManualClock::new());
        let (client, _rx, tasks) = client_for(&mock, clock).await;
        let err = client
            .call(method::GET_ALGO_PARAM_NAMES, Some(json!(["ra"])))
            .await
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::Unsupported);
        let err = client
            .call(method::DITHER, Some(json!({"amount": 1})))
            .await
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::InvalidOperation);
        assert!(err.message.contains("cannot dither if not guiding"));
        client.close();
        tasks.abort();
    }

    #[tokio::test]
    async fn an_unanswered_request_times_out_on_the_injected_clock() {
        let mock = Phd2Mock::start().await;
        mock.with_state(|s| {
            s.silent_methods.insert("get_exposure".into());
        });
        let manual = Arc::new(ManualClock::new());
        let clock: Arc<dyn Clock> = manual.clone();
        let (client, _rx, tasks) = client_for(&mock, clock).await;
        let caller = client.clone();
        let call = tokio::spawn(async move {
            caller
                .call_with_timeout(method::GET_EXPOSURE, None, Duration::from_secs(10))
                .await
        });
        assert!(
            mock.wait_for_request("get_exposure", Duration::from_secs(5))
                .await
        );
        // Real time passes but the clock does not: still waiting.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!call.is_finished());
        manual.advance(Duration::from_secs(11));
        let err = tokio::time::timeout(Duration::from_secs(5), call)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::Timeout);
        assert_eq!(client.pending_requests(), 0);
        assert_eq!(
            method_timeout(method::SET_CONNECTED, Duration::from_secs(10)),
            Duration::from_secs(60)
        );
        assert_eq!(
            method_timeout(method::GET_EXPOSURE, Duration::from_secs(10)),
            Duration::from_secs(10)
        );
        client.close();
        tasks.abort();
    }

    #[tokio::test]
    async fn a_socket_drop_fails_pending_calls_and_reports_the_loss() {
        let mock = Phd2Mock::start().await;
        mock.with_state(|s| {
            s.silent_methods.insert("get_exposure".into());
        });
        let clock: Arc<dyn Clock> = Arc::new(ManualClock::new());
        let (client, mut rx, tasks) = client_for(&mock, clock).await;
        let caller = client.clone();
        let call = tokio::spawn(async move { caller.call(method::GET_EXPOSURE, None).await });
        assert!(
            mock.wait_for_request("get_exposure", Duration::from_secs(5))
                .await
        );
        mock.drop_clients();
        let err = tokio::time::timeout(Duration::from_secs(5), call)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::Disconnected);
        loop {
            if let ClientEvent::Closed { local, reason } = next_event(&mut rx).await {
                assert!(!local, "{reason}");
                break;
            }
        }
        assert!(client.is_closed());
        let err = client.call(method::GET_APP_STATE, None).await.unwrap_err();
        assert_eq!(err.class, ErrorClass::Disconnected);
        tasks.abort();
    }

    #[tokio::test]
    async fn crlf_and_lf_framing_both_work_end_to_end() {
        for crlf in [false, true] {
            let mock = Phd2Mock::start().await;
            mock.with_state(|s| s.crlf = crlf);
            let clock: Arc<dyn Clock> = Arc::new(ManualClock::new());
            let (client, mut rx, tasks) = client_for(&mock, clock).await;
            let mut names = Vec::new();
            while names.len() < 2 {
                if let ClientEvent::Event { event, .. } = next_event(&mut rx).await {
                    names.push(event.name);
                }
            }
            assert_eq!(names, vec!["Version", "AppState"], "crlf={crlf}");
            assert_eq!(
                client.call(method::GET_CONNECTED, None).await.unwrap(),
                json!(true),
                "crlf={crlf}"
            );
            client.close();
            tasks.abort();
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_abort_path_sends_stop_capture_on_its_own_connection() {
        let mock = Phd2Mock::start().await;
        mock.with_state(|s| s.app_state = "Guiding".into());
        let path = Phd2AbortPath::new("127.0.0.1", mock.port(), Duration::from_secs(5));
        let result = tokio::task::spawn_blocking(move || path.abort())
            .await
            .unwrap();
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(mock.method_count("stop_capture"), 1);
        assert_eq!(mock.state().app_state, "Stopped");
        // Nothing listening: a prompt, classified failure.
        let closed = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        };
        let path = Phd2AbortPath::new("127.0.0.1", closed, Duration::from_secs(2));
        let err = tokio::task::spawn_blocking(move || path.abort())
            .await
            .unwrap()
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::Disconnected);
    }

    #[test]
    fn abort_candidates_put_the_known_address_first_then_ipv4() {
        let v6: SocketAddr = "[::1]:4400".parse().unwrap();
        let v4: SocketAddr = "127.0.0.1:4400".parse().unwrap();
        let lan: SocketAddr = "192.168.1.5:4400".parse().unwrap();
        // getaddrinfo("localhost") on macOS, Linux and Windows: ::1 first.
        assert_eq!(abort_candidates(None, [v6, v4]), vec![v4, v6]);
        assert_eq!(abort_candidates(Some(v4), [v6, v4]), vec![v4, v6]);
        assert_eq!(abort_candidates(Some(v6), [v6, v4]), vec![v6, v4]);
        assert_eq!(
            abort_candidates(Some(lan), [v6, lan, v4]),
            vec![lan, v4, v6],
            "no repeats; resolver order kept within a family"
        );
        assert!(abort_candidates(None, []).is_empty());
        assert_eq!(display_peer("::1", 4400), "[::1]:4400");
        assert_eq!(display_peer("localhost", 4401), "localhost:4401");
    }

    /// PHD2's event server (like the mock) listens on IPv4 only, and
    /// `localhost` (the default `/connect` host) usually resolves to `::1`
    /// first. Both the main connection and the abort path must still reach
    /// it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn localhost_reaches_an_ipv4_only_phd2_on_both_connections() {
        let mock = Phd2Mock::start().await;
        mock.with_state(|s| s.app_state = "Guiding".into());
        let clock: Arc<dyn Clock> = Arc::new(ManualClock::new());
        let (client, _rx, tasks) = Phd2Client::connect(
            "localhost",
            mock.port(),
            Duration::from_secs(5),
            Duration::from_secs(10),
            clock,
        )
        .await
        .unwrap();
        let reached = client.peer_addr().expect("the connected address");
        assert!(reached.is_ipv4(), "{reached}");
        assert_eq!(reached.port(), mock.port());
        assert_eq!(client.peer(), format!("localhost:{}", mock.port()));

        // Resolving again, without the address the main connection used.
        let path = Phd2AbortPath::new("localhost", mock.port(), Duration::from_secs(5));
        let result = tokio::task::spawn_blocking(move || path.abort())
            .await
            .unwrap();
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(mock.method_count("stop_capture"), 1);

        // With the address the main connection reached.
        let path = Phd2AbortPath::new("localhost", mock.port(), Duration::from_secs(5))
            .with_peer_addr(client.peer_addr());
        let result = tokio::task::spawn_blocking(move || path.abort())
            .await
            .unwrap();
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(mock.method_count("stop_capture"), 2);
        client.close();
        tasks.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_abort_path_falls_back_to_resolution_when_the_known_address_fails() {
        let mock = Phd2Mock::start().await;
        let dead: SocketAddr = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap()
        };
        let path = Phd2AbortPath::new("127.0.0.1", mock.port(), Duration::from_secs(5))
            .with_peer_addr(Some(dead));
        let result = tokio::task::spawn_blocking(move || path.abort())
            .await
            .unwrap();
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(mock.method_count("stop_capture"), 1);
        // Every address refused: the message names each attempt.
        let closed = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        };
        let path = Phd2AbortPath::new("127.0.0.1", closed, Duration::from_secs(2))
            .with_peer_addr(Some(dead));
        let err = tokio::task::spawn_blocking(move || path.abort())
            .await
            .unwrap()
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::Disconnected);
        assert!(err.message.contains(&dead.to_string()), "{}", err.message);
        assert!(
            err.message.contains(&format!("127.0.0.1:{closed}")),
            "{}",
            err.message
        );
    }
}

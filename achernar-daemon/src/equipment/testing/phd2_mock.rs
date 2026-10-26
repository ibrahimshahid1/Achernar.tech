//! Loopback line-JSON PHD2 mock (spec §5.6; WP1A-7). Tests only.
//!
//! A tokio TCP server on `127.0.0.1:0` that speaks the PHD2 event-server
//! protocol well enough to exercise the client and the guider service on
//! every OS (no external process, no fixed port):
//!
//! - On connect it sends the connect burst: `Version`, any extra events in
//!   [`MockState::connect_extra`], then `AppState`.
//! - It answers the methods the service uses from [`MockState`], and emits
//!   the events PHD2 would (`guide` and `dither` settle scripts,
//!   `stop_capture`, `set_paused`).
//! - Tests inject events ([`Phd2Mock::send_event`]), drop every client
//!   ([`Phd2Mock::drop_clients`]), make methods silent or unsupported, and
//!   override replies with a responder.
//! - LF or CRLF framing ([`MockState::crlf`]).

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Notify};
use tokio::task::JoinHandle;

use crate::equipment::guider::phd2_proto::LineFramer;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// A settle script for `guide` or `dither`.
#[derive(Debug, Clone, PartialEq)]
pub struct SettleScript {
    /// Emit `SettleBegin` and the `Settling` events right after the reply.
    pub auto: bool,
    /// `Settling.Distance` values, one event each.
    pub distances: Vec<f64>,
    /// `SettleDone {Status, Error}`; `None` holds the settle open.
    pub done: Option<(i64, Option<String>)>,
    pub total_frames: u64,
    pub dropped_frames: u64,
}

impl Default for SettleScript {
    fn default() -> Self {
        Self {
            auto: true,
            distances: vec![2.5, 1.0, 0.4],
            done: Some((0, None)),
            total_frames: 3,
            dropped_frames: 0,
        }
    }
}

/// The mock's PHD2 state. Public fields; mutate with
/// [`Phd2Mock::with_state`].
#[derive(Debug, Clone, PartialEq)]
pub struct MockState {
    pub app_state: String,
    pub connected: bool,
    pub version: String,
    pub subver: String,
    pub msg_version: i64,
    pub profile: String,
    pub camera: String,
    pub mount: String,
    pub pixel_scale: Option<f64>,
    pub calibrated: bool,
    pub dec_guide_mode: String,
    pub exposure_ms: u64,
    pub algo_names_ra: Vec<String>,
    pub algo_names_dec: Vec<String>,
    /// `(axis, name)` → value.
    pub algo_values: HashMap<(String, String), f64>,
    pub crlf: bool,
    pub send_connect_burst: bool,
    pub connect_extra: Vec<Value>,
    pub guide_script: SettleScript,
    pub dither_script: SettleScript,
    pub dither_dx_dy: (f64, f64),
    /// A JSON-RPC error for `dither` (e.g. a re-entrant settle).
    pub dither_error: Option<(i64, String)>,
    /// `stop_capture` replies but PHD2 keeps guiding.
    pub stop_ignored: bool,
    /// Methods that never get a reply.
    pub silent_methods: BTreeSet<String>,
    /// Methods answered with JSON-RPC "method not found".
    pub unsupported_methods: BTreeSet<String>,
    /// Next `Timestamp` for events that carry none; advances 1 s per event.
    pub next_timestamp: f64,
}

impl Default for MockState {
    fn default() -> Self {
        let mut algo_values = HashMap::new();
        for (axis, name, value) in [
            ("ra", "minMove", 0.15),
            ("ra", "hysteresis", 0.1),
            ("ra", "aggression", 0.7),
            ("dec", "minMove", 0.2),
            ("dec", "aggressiveness", 0.5),
        ] {
            algo_values.insert((axis.to_string(), name.to_string()), value);
        }
        Self {
            app_state: "Stopped".into(),
            connected: true,
            version: "2.6.13".into(),
            subver: "dev1".into(),
            msg_version: 1,
            profile: "Mock Profile".into(),
            camera: "Mock Guide Camera".into(),
            mount: "On-camera".into(),
            pixel_scale: Some(1.5),
            calibrated: true,
            dec_guide_mode: "Auto".into(),
            exposure_ms: 2000,
            algo_names_ra: vec!["minMove".into(), "hysteresis".into(), "aggression".into()],
            algo_names_dec: vec!["minMove".into(), "aggressiveness".into()],
            algo_values,
            crlf: true,
            send_connect_burst: true,
            connect_extra: Vec::new(),
            guide_script: SettleScript::default(),
            dither_script: SettleScript::default(),
            dither_dx_dy: (2.4, -1.8),
            dither_error: None,
            stop_ignored: false,
            silent_methods: BTreeSet::new(),
            unsupported_methods: BTreeSet::new(),
            next_timestamp: 1_800_000_000.0,
        }
    }
}

/// One request the mock received.
#[derive(Debug, Clone, PartialEq)]
pub struct MockRequest {
    pub method: String,
    pub params: Value,
    pub id: u64,
}

/// A scripted reply: the result or a JSON-RPC error, then events.
#[derive(Debug, Clone, PartialEq)]
pub struct MockReply {
    pub result: Result<Value, (i64, String)>,
    pub events: Vec<Value>,
}

type Responder = Box<dyn FnMut(&MockRequest, &mut MockState) -> Option<MockReply> + Send>;

enum Out {
    Line(Value),
    Close,
}

struct Shared {
    state: Mutex<MockState>,
    requests: Mutex<Vec<MockRequest>>,
    clients: Mutex<Vec<mpsc::UnboundedSender<Out>>>,
    tasks: Mutex<Vec<JoinHandle<()>>>,
    responder: Mutex<Option<Responder>>,
    request_seen: Notify,
}

/// The running mock. Dropping it stops the server and every client task.
pub struct Phd2Mock {
    port: u16,
    shared: Arc<Shared>,
    accept: JoinHandle<()>,
}

impl Phd2Mock {
    pub async fn start() -> Self {
        Self::start_with(MockState::default()).await
    }

    pub async fn start_with(state: MockState) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind the PHD2 mock on loopback");
        let port = listener.local_addr().expect("mock address").port();
        let shared = Arc::new(Shared {
            state: Mutex::new(state),
            requests: Mutex::new(Vec::new()),
            clients: Mutex::new(Vec::new()),
            tasks: Mutex::new(Vec::new()),
            responder: Mutex::new(None),
            request_seen: Notify::new(),
        });
        let accept = tokio::spawn(accept_loop(listener, Arc::clone(&shared)));
        Self {
            port,
            shared,
            accept,
        }
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn with_state(&self, f: impl FnOnce(&mut MockState)) {
        f(&mut lock(&self.shared.state));
    }

    pub fn state(&self) -> MockState {
        lock(&self.shared.state).clone()
    }

    /// Override replies. Returning `None` falls back to the built-in reply.
    pub fn set_responder(
        &self,
        responder: impl FnMut(&MockRequest, &mut MockState) -> Option<MockReply> + Send + 'static,
    ) {
        *lock(&self.shared.responder) = Some(Box::new(responder));
    }

    /// Send an event to every connected client. `Timestamp`, `Host` and
    /// `Inst` are added when missing.
    pub fn send_event(&self, event: Value) {
        let event = stamp_event(&self.shared, event);
        broadcast(&self.shared, event);
    }

    /// A complete GuideStep with PHD2's field names.
    pub fn guide_step(frame: u64, ra_raw: f64, dec_raw: f64) -> Value {
        json!({
            "Event": "GuideStep",
            "Frame": frame,
            "Time": frame as f64 * 2.0,
            "Mount": "On-camera",
            "dx": ra_raw * 0.9 - dec_raw * 0.1,
            "dy": ra_raw * 0.1 + dec_raw * 0.9,
            "RADistanceRaw": ra_raw,
            "DECDistanceRaw": dec_raw,
            "RADistanceGuide": ra_raw * 0.7,
            "DECDistanceGuide": dec_raw * 0.5,
            "RADuration": 120,
            "RADirection": if ra_raw >= 0.0 { "West" } else { "East" },
            "DECDuration": 80,
            "DECDirection": if dec_raw >= 0.0 { "North" } else { "South" },
            "StarMass": 12345.0,
            "SNR": 25.0,
            "HFD": 2.2,
            "AvgDist": 0.3,
            "RALimited": false,
            "DecLimited": false,
            "ErrorCode": 0,
        })
    }

    /// Close every client connection (a socket drop).
    pub fn drop_clients(&self) {
        let clients: Vec<_> = lock(&self.shared.clients).drain(..).collect();
        for client in clients {
            let _ = client.send(Out::Close);
        }
    }

    pub fn client_count(&self) -> usize {
        let mut clients = lock(&self.shared.clients);
        clients.retain(|c| !c.is_closed());
        clients.len()
    }

    pub fn requests(&self) -> Vec<MockRequest> {
        lock(&self.shared.requests).clone()
    }

    pub fn method_count(&self, method: &str) -> usize {
        lock(&self.shared.requests)
            .iter()
            .filter(|r| r.method == method)
            .count()
    }

    pub fn last_request(&self, method: &str) -> Option<MockRequest> {
        lock(&self.shared.requests)
            .iter()
            .rev()
            .find(|r| r.method == method)
            .cloned()
    }

    /// Wait (real time) until at least one `method` request arrived.
    pub async fn wait_for_request(&self, method: &str, timeout: Duration) -> bool {
        self.wait_for_count(method, 1, timeout).await
    }

    /// Wait (real time) until at least `n` `method` requests arrived.
    pub async fn wait_for_count(&self, method: &str, n: usize, timeout: Duration) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let notified = self.shared.request_seen.notified();
            if self.method_count(method) >= n {
                return true;
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return self.method_count(method) >= n;
            }
        }
    }
}

impl Drop for Phd2Mock {
    fn drop(&mut self) {
        self.accept.abort();
        // Close every socket: the writer tasks end, which releases the
        // shared state they hold.
        self.drop_clients();
        for task in lock(&self.shared.tasks).drain(..) {
            task.abort();
        }
    }
}

fn stamp_event(shared: &Shared, mut event: Value) -> Value {
    if let Some(obj) = event.as_object_mut() {
        if !obj.contains_key("Timestamp") {
            let mut state = lock(&shared.state);
            obj.insert("Timestamp".into(), json!(state.next_timestamp));
            state.next_timestamp += 1.0;
        }
        obj.entry("Host").or_insert_with(|| json!("mock"));
        obj.entry("Inst").or_insert_with(|| json!(1));
    }
    event
}

fn broadcast(shared: &Shared, event: Value) {
    let mut clients = lock(&shared.clients);
    clients.retain(|c| c.send(Out::Line(event.clone())).is_ok());
}

async fn accept_loop(listener: TcpListener, shared: Arc<Shared>) {
    while let Ok((stream, _)) = listener.accept().await {
        let task = tokio::spawn(serve_client(stream, Arc::clone(&shared)));
        lock(&shared.tasks).push(task);
    }
}

async fn serve_client(stream: TcpStream, shared: Arc<Shared>) {
    let _ = stream.set_nodelay(true);
    let (mut read_half, mut write_half) = stream.into_split();
    let (tx, mut rx) = mpsc::unbounded_channel::<Out>();
    // The connect burst goes out first, through the same queue.
    let burst = {
        let state = lock(&shared.state);
        if state.send_connect_burst {
            let mut events = vec![json!({
                "Event": "Version",
                "PHDVersion": state.version,
                "PHDSubver": state.subver,
                "MsgVersion": state.msg_version,
                "OverlapSupport": true,
            })];
            events.extend(state.connect_extra.iter().cloned());
            events.push(json!({"Event": "AppState", "State": state.app_state}));
            events
        } else {
            Vec::new()
        }
    };
    for event in burst {
        let _ = tx.send(Out::Line(stamp_event(&shared, event)));
    }
    lock(&shared.clients).push(tx.clone());
    let writer_shared = Arc::clone(&shared);
    let writer = tokio::spawn(async move {
        while let Some(out) = rx.recv().await {
            match out {
                Out::Line(value) => {
                    let ending = if lock(&writer_shared.state).crlf {
                        "\r\n"
                    } else {
                        "\n"
                    };
                    let line = format!("{value}{ending}");
                    if write_half.write_all(line.as_bytes()).await.is_err() {
                        break;
                    }
                }
                Out::Close => {
                    let _ = write_half.shutdown().await;
                    break;
                }
            }
        }
    });
    let mut framer = LineFramer::new();
    let mut buf = vec![0u8; 4096];
    loop {
        let n = tokio::select! {
            read = read_half.read(&mut buf) => match read {
                Ok(0) | Err(_) => break,
                Ok(n) => n,
            },
            _ = tx.closed() => break,
        };
        for line in framer.push(&buf[..n]) {
            handle_request(&shared, &tx, &line);
        }
        if writer.is_finished() {
            break;
        }
    }
    writer.abort();
}

fn handle_request(shared: &Shared, tx: &mpsc::UnboundedSender<Out>, line: &str) {
    let Ok(value) = serde_json::from_str::<Value>(line) else {
        return;
    };
    let Some(method) = value.get("method").and_then(Value::as_str) else {
        return;
    };
    let request = MockRequest {
        method: method.to_string(),
        params: value.get("params").cloned().unwrap_or(Value::Null),
        id: value.get("id").and_then(Value::as_u64).unwrap_or(0),
    };
    lock(&shared.requests).push(request.clone());
    shared.request_seen.notify_waiters();
    let reply = {
        let mut state = lock(&shared.state);
        if state.silent_methods.contains(&request.method) {
            None
        } else if state.unsupported_methods.contains(&request.method) {
            Some(MockReply {
                result: Err((-32601, "method not found".into())),
                events: Vec::new(),
            })
        } else {
            let custom = lock(&shared.responder)
                .as_mut()
                .and_then(|responder| responder(&request, &mut state));
            Some(custom.unwrap_or_else(|| builtin_reply(&request, &mut state)))
        }
    };
    let Some(reply) = reply else { return };
    let line = match reply.result {
        Ok(result) => json!({"jsonrpc": "2.0", "result": result, "id": request.id}),
        Err((code, message)) => json!({
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": request.id,
        }),
    };
    let _ = tx.send(Out::Line(line));
    for event in reply.events {
        let event = stamp_event(shared, event);
        let _ = tx.send(Out::Line(event.clone()));
        // Other clients see PHD2's events too.
        let clients = lock(&shared.clients);
        for client in clients.iter().filter(|c| !c.same_channel(tx)) {
            let _ = client.send(Out::Line(event.clone()));
        }
    }
}

fn settle_events(script: &SettleScript, extra_first: Vec<Value>) -> Vec<Value> {
    let mut events = extra_first;
    if !script.auto {
        return events;
    }
    events.push(json!({"Event": "SettleBegin"}));
    for (i, distance) in script.distances.iter().enumerate() {
        events.push(json!({
            "Event": "Settling",
            "Distance": distance,
            "Time": i as f64 * 2.0,
            "SettleTime": 10.0,
            "StarLocked": true,
        }));
    }
    if let Some((status, error)) = &script.done {
        let mut done = json!({
            "Event": "SettleDone",
            "Status": status,
            "TotalFrames": script.total_frames,
            "DroppedFrames": script.dropped_frames,
        });
        if let Some(error) = error {
            done["Error"] = json!(error);
        }
        events.push(done);
    }
    events
}

fn ok(result: Value) -> MockReply {
    MockReply {
        result: Ok(result),
        events: Vec::new(),
    }
}

fn builtin_reply(request: &MockRequest, state: &mut MockState) -> MockReply {
    let param = |i: usize, key: &str| {
        request
            .params
            .get(i)
            .or_else(|| request.params.get(key))
            .cloned()
    };
    match request.method.as_str() {
        "get_app_state" => ok(json!(state.app_state)),
        "get_connected" => ok(json!(state.connected)),
        "set_connected" => {
            state.connected = param(0, "connected")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            ok(json!(0))
        }
        "get_profile" => ok(json!({"id": 1, "name": state.profile})),
        "get_current_equipment" => ok(json!({
            "camera": {"name": state.camera, "connected": true},
            "mount": {"name": state.mount, "connected": true},
        })),
        "get_pixel_scale" => ok(json!(state.pixel_scale)),
        "get_calibrated" => ok(json!(state.calibrated)),
        "get_calibration_data" => {
            if state.calibrated {
                ok(
                    json!({"xAngle": 12.5, "yAngle": 102.4, "xRate": 4.1, "yRate": 3.9, "xParity": "+", "yParity": "-", "declination": 20.0}),
                )
            } else {
                MockReply {
                    result: Err((1, "not calibrated".into())),
                    events: Vec::new(),
                }
            }
        }
        "get_dec_guide_mode" => ok(json!(state.dec_guide_mode)),
        "get_exposure" => ok(json!(state.exposure_ms)),
        "get_algo_param_names" => {
            let axis = param(0, "axis").and_then(|v| v.as_str().map(str::to_string));
            match axis.as_deref() {
                Some("ra") => ok(json!(state.algo_names_ra)),
                Some("dec") => ok(json!(state.algo_names_dec)),
                _ => MockReply {
                    result: Err((-32602, "bad axis".into())),
                    events: Vec::new(),
                },
            }
        }
        "get_algo_param" => {
            let axis = param(0, "axis").and_then(|v| v.as_str().map(str::to_string));
            let name = param(1, "name").and_then(|v| v.as_str().map(str::to_string));
            match axis
                .zip(name)
                .and_then(|key| state.algo_values.get(&key).copied())
            {
                Some(value) => ok(json!(value)),
                None => MockReply {
                    result: Err((1, "could not get param".into())),
                    events: Vec::new(),
                },
            }
        }
        "set_algo_param" => ok(json!(0)),
        "guide" => {
            let starting = state.app_state != "Guiding";
            state.app_state = "Guiding".into();
            let first = if starting {
                vec![json!({"Event": "StartGuiding"})]
            } else {
                Vec::new()
            };
            MockReply {
                result: Ok(json!(0)),
                events: settle_events(&state.guide_script, first),
            }
        }
        "dither" => {
            if let Some((code, message)) = &state.dither_error {
                return MockReply {
                    result: Err((*code, message.clone())),
                    events: Vec::new(),
                };
            }
            if state.app_state != "Guiding" {
                return MockReply {
                    result: Err((1, "cannot dither if not guiding".into())),
                    events: Vec::new(),
                };
            }
            let (dx, dy) = state.dither_dx_dy;
            MockReply {
                result: Ok(json!(0)),
                events: settle_events(
                    &state.dither_script,
                    vec![json!({"Event": "GuidingDithered", "dx": dx, "dy": dy})],
                ),
            }
        }
        "stop_capture" => {
            if state.stop_ignored {
                return ok(json!(0));
            }
            let was_guiding = state.app_state == "Guiding" || state.app_state == "Paused";
            state.app_state = "Stopped".into();
            let mut events = Vec::new();
            if was_guiding {
                events.push(json!({"Event": "GuidingStopped"}));
            }
            events.push(json!({"Event": "LoopingExposuresStopped"}));
            MockReply {
                result: Ok(json!(0)),
                events,
            }
        }
        "set_paused" => {
            let paused = param(0, "paused")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let events = if paused {
                state.app_state = "Paused".into();
                vec![json!({"Event": "Paused"})]
            } else if state.app_state == "Paused" {
                state.app_state = "Guiding".into();
                vec![json!({"Event": "Resumed"})]
            } else {
                Vec::new()
            };
            MockReply {
                result: Ok(json!(0)),
                events,
            }
        }
        _ => MockReply {
            result: Err((-32601, "method not found".into())),
            events: Vec::new(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncBufReadExt, BufReader};

    #[tokio::test]
    async fn the_mock_sends_the_connect_burst_and_answers_requests() {
        let mock = Phd2Mock::start().await;
        let stream = TcpStream::connect(("127.0.0.1", mock.port()))
            .await
            .unwrap();
        let (read_half, mut write_half) = stream.into_split();
        let mut lines = BufReader::new(read_half).lines();
        let version: Value =
            serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(version["Event"], json!("Version"));
        assert!(version["Timestamp"].is_number());
        let state: Value =
            serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(state["State"], json!("Stopped"));
        write_half
            .write_all(b"{\"method\":\"get_pixel_scale\",\"id\":5}\r\n")
            .await
            .unwrap();
        let reply: Value =
            serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(reply, json!({"jsonrpc": "2.0", "result": 1.5, "id": 5}));
        assert_eq!(mock.method_count("get_pixel_scale"), 1);
        mock.send_event(Phd2Mock::guide_step(1, 0.1, -0.1));
        let step: Value = serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(step["Event"], json!("GuideStep"));
        assert_eq!(step["Host"], json!("mock"));
        mock.drop_clients();
        assert!(lines.next_line().await.unwrap().is_none());
    }
}

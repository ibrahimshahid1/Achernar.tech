//! `AlpacaClient`: the Alpaca transport over `ureq` (spec §3.1 C-2, C-3;
//! D4, D16). Owner: WP1A-1.
//!
//! **Blocking by design.** Every call here does synchronous network I/O and
//! is made only from a device worker thread or `spawn_blocking` (D3), never
//! from a tokio worker.
//!
//! What the client does for every request:
//!
//! - Adds `ClientID` (one stable, random id in 1..=65535 per install, see
//!   [`ClientId::load_or_create`]; D16) and `ClientTransactionID` (a
//!   process-global counter starting at 1, see [`next_transaction_id`]).
//!   The echoed id is checked; a mismatch is logged and counted, never fatal.
//! - Encodes parameters with [`super::form`] (exact-case keys, invariant
//!   numbers). PUT bodies are `application/x-www-form-urlencoded`.
//! - Applies the C-3 timeout classes. The effective timeout is the longer of
//!   the caller's class and the class the C-3 table gives the member, so a
//!   `park` is never cut off at 10 s. The TCP connect phase is capped at 5 s
//!   so a stalled connect is reported as one.
//! - Runs the C-3 response checks ([`super::errors::check_response`]).
//! - **Retries once, after 100 ms, only on a timeout or a socket error**
//!   (the ASCOM Library rule; C-2). HTTP errors, Alpaca errors and malformed
//!   replies are never retried. A GET is retried after any timeout or socket
//!   error. A PUT is retried only when the request provably never reached the
//!   device (connection refused, no usable address, DNS failure, or a
//!   connect-phase timeout): a PUT that may have been delivered is not re-sent, because a
//!   duplicated relative focuser move or guide pulse would move hardware
//!   twice. Its error says the outcome is unknown.
//!
//! Connections: GETs use a pooled agent (keep-alive suits the 1-10 Hz polls).
//! PUTs use an agent that never reuses a connection, which is what makes the
//! "provably not delivered" test above exact. The abort agent ([`AbortChannel`])
//! is a third, unpooled agent with a 5 s timeout, so an abort never queues
//! behind a request on the worker's connection (D4); a device that serves one
//! HTTP request at a time can still delay it, and that shows up as abort
//! latency, not as a hang.

use std::collections::BTreeSet;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::errors::{check_response, AlpacaReply, Verb};
use super::form;
use crate::equipment::traits::{AbortPath, AlpacaTransport, Clock, TimeoutClass};
use crate::equipment::types::{
    identity_key, DeviceError, DeviceResult, Endpoint, ErrorClass, ObsState, Observed, Source,
};

// ---------------------------------------------------------------------------
// ClientID and ClientTransactionID
// ---------------------------------------------------------------------------

/// File in the data directory that holds the persisted ClientID.
pub const CLIENT_ID_FILE: &str = "alpaca_client_id";

/// The Alpaca `ClientID`: 1..=65535, stable per daemon install (D16). The
/// ESP32 mount tracks connections per ClientID and stops motion only when
/// the last client leaves, so the id must survive restarts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ClientId(u32);

/// Where a [`ClientId`] came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClientIdSource {
    /// `[equipment] alpaca_client_id`.
    Configured,
    /// Read back from [`CLIENT_ID_FILE`].
    Persisted,
    /// Generated now and written to [`CLIENT_ID_FILE`].
    Generated,
}

impl ClientId {
    pub const MIN: u32 = 1;
    pub const MAX: u32 = 65_535;

    pub fn new(value: u32) -> Option<Self> {
        (Self::MIN..=Self::MAX)
            .contains(&value)
            .then_some(Self(value))
    }

    pub fn get(self) -> u32 {
        self.0
    }

    /// A uniformly random id in 1..=65535 (`rand`, owner decision O7).
    pub fn random() -> Self {
        Self(rand::thread_rng().gen_range(Self::MIN..=Self::MAX))
    }

    /// The configured id when it is in range; otherwise the id persisted in
    /// `data_dir`, generating and persisting one on first use. An
    /// out-of-range configured value is ignored here (the config loader
    /// already warns about it). A persisted file that does not hold a valid
    /// id is replaced.
    pub fn load_or_create(data_dir: &Path, configured: u32) -> io::Result<(Self, ClientIdSource)> {
        if let Some(id) = Self::new(configured) {
            return Ok((id, ClientIdSource::Configured));
        }
        let path = data_dir.join(CLIENT_ID_FILE);
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                if let Some(id) = text.trim().parse::<u32>().ok().and_then(Self::new) {
                    return Ok((id, ClientIdSource::Persisted));
                }
                tracing::warn!(
                    path = %path.display(),
                    "persisted Alpaca ClientID is invalid; generating a new one"
                );
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        let id = Self::random();
        std::fs::create_dir_all(data_dir)?;
        let temp = data_dir.join(format!("{CLIENT_ID_FILE}.tmp-{}", std::process::id()));
        std::fs::write(&temp, format!("{}\n", id.0))?;
        std::fs::rename(&temp, &path)?;
        Ok((id, ClientIdSource::Generated))
    }
}

impl std::fmt::Display for ClientId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

static NEXT_TRANSACTION_ID: AtomicU32 = AtomicU32::new(1);

/// The next process-global `ClientTransactionID`: starts at 1, increases by
/// one per request across every client in the process, and skips 0 when the
/// 32-bit counter wraps.
pub fn next_transaction_id() -> u32 {
    loop {
        let id = NEXT_TRANSACTION_ID.fetch_add(1, Ordering::Relaxed);
        if id != 0 {
            return id;
        }
    }
}

// ---------------------------------------------------------------------------
// Timeouts
// ---------------------------------------------------------------------------

/// The duration of each C-3 timeout class. The default is the C-3 table;
/// tests shorten it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeoutTable {
    pub connect: Duration,
    pub ordinary: Duration,
    pub long: Duration,
}

impl Default for TimeoutTable {
    fn default() -> Self {
        Self {
            connect: TimeoutClass::Connect.duration(),
            ordinary: TimeoutClass::Ordinary.duration(),
            long: TimeoutClass::Long.duration(),
        }
    }
}

impl TimeoutTable {
    pub fn get(&self, class: TimeoutClass) -> Duration {
        match class {
            TimeoutClass::Connect => self.connect,
            TimeoutClass::Ordinary => self.ordinary,
            TimeoutClass::Long => self.long,
        }
    }
}

/// The C-3 timeout class of a member (lower-case Alpaca member name).
pub fn member_timeout_class(member: &str) -> TimeoutClass {
    match member.to_ascii_lowercase().as_str() {
        "connect" | "connecting" | "connected" | "interfaceversion" => TimeoutClass::Connect,
        // Mount motion, slews and syncs.
        "abortslew" | "findhome" | "moveaxis" | "park" | "unpark" | "pulseguide"
        | "slewtocoordinates" | "slewtocoordinatesasync" | "slewtoaltaz" | "slewtoaltazasync"
        | "slewtotarget" | "slewtotargetasync" | "synctocoordinates" | "synctoaltaz"
        | "synctotarget"
        // Focuser `move` and `halt`, and the rotator moves.
        | "move" | "halt" | "moveabsolute" | "movemechanical" => TimeoutClass::Long,
        _ => TimeoutClass::Ordinary,
    }
}

// ---------------------------------------------------------------------------
// Configuration and statistics
// ---------------------------------------------------------------------------

/// Client behaviour. [`ClientConfig::new`] gives the C-2/C-3 values.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub client_id: ClientId,
    pub timeouts: TimeoutTable,
    /// Upper bound on the TCP connect phase of any request.
    pub connect_phase_cap: Duration,
    /// Timeout of every abort-channel request (§2.5: 5 s).
    pub abort_timeout: Duration,
    /// Pause before the single retry (C-2: 100 ms).
    pub retry_delay: Duration,
    /// Largest reply body accepted.
    pub max_body_bytes: u64,
}

impl ClientConfig {
    pub fn new(client_id: ClientId) -> Self {
        Self {
            client_id,
            timeouts: TimeoutTable::default(),
            connect_phase_cap: Duration::from_secs(5),
            abort_timeout: Duration::from_secs(5),
            retry_delay: Duration::from_millis(100),
            max_body_bytes: 4 * 1024 * 1024,
        }
    }
}

/// Counters for diagnostics and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ClientStats {
    /// Logical requests (a retry is not counted twice).
    pub requests: u64,
    pub retries: u64,
    /// Replies whose echoed `ClientTransactionID` did not match.
    pub transaction_mismatches: u64,
    /// Replies accepted without a `Content-Type` header.
    pub missing_content_type: u64,
}

#[derive(Debug, Default)]
struct Counters {
    requests: AtomicU64,
    retries: AtomicU64,
    transaction_mismatches: AtomicU64,
    missing_content_type: AtomicU64,
}

// ---------------------------------------------------------------------------
// The client
// ---------------------------------------------------------------------------

struct Shared {
    /// `scheme://host:port`, no trailing slash.
    base: String,
    host: String,
    port: u16,
    /// Lower-case Alpaca device type (`telescope`, `focuser`, ...).
    device_type: String,
    number: u32,
    /// `{base}/api/v1/{device_type}/{number}/`.
    prefix: String,
    config: ClientConfig,
    read_agent: ureq::Agent,
    command_agent: ureq::Agent,
    abort_agent: ureq::Agent,
    counters: Counters,
}

/// One Alpaca device's transport. Cheap to clone; clones share agents and
/// counters.
#[derive(Clone)]
pub struct AlpacaClient {
    shared: Arc<Shared>,
}

/// The out-of-band transport for abort and halt (D4): its own unpooled agent
/// and a 5 s timeout. Obtained with [`AlpacaClient::abort_channel`].
#[derive(Clone)]
pub struct AbortChannel {
    shared: Arc<Shared>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lane {
    Read,
    Command,
    Abort,
}

/// Why one HTTP attempt failed, for the retry decision.
enum Failure {
    /// The request provably never reached the server.
    NotSent(DeviceError),
    /// The request may have reached the server.
    MaybeSent(DeviceError),
    /// A definitive answer (HTTP or Alpaca error, malformed reply) or a
    /// local error. Never retried.
    Final(DeviceError),
}

/// A validated `http(s)://host:port` base URL.
pub(crate) struct BaseUrl {
    pub base: String,
    pub host: String,
    pub port: u16,
}

/// Validate and normalise an Alpaca server base URL: HTTP(S), a host, no
/// credentials, no path beyond `/`, no query or fragment.
pub(crate) fn parse_base_url(text: &str) -> DeviceResult<BaseUrl> {
    let text = text.trim();
    let uri: ureq::http::Uri = text
        .parse()
        .map_err(|_| DeviceError::invalid_value(format!("invalid Alpaca server URL '{text}'")))?;
    let scheme = uri.scheme_str().map(str::to_ascii_lowercase);
    let authority = uri.authority();
    let valid = matches!(scheme.as_deref(), Some("http" | "https"))
        && uri.host().is_some_and(|host| !host.is_empty())
        && authority.is_some_and(|a| !a.as_str().contains('@'))
        && matches!(uri.path(), "" | "/")
        && uri.query().is_none()
        && !text.contains('#');
    if !valid {
        return Err(DeviceError::invalid_value(format!(
            "'{text}' is not an HTTP(S) Alpaca server base URL without credentials, path, query or fragment"
        )));
    }
    let scheme = scheme.unwrap_or_default();
    // Host names are case-insensitive; one spelling keeps identity keys and
    // endpoints stable.
    let host = uri.host().unwrap_or_default().to_ascii_lowercase();
    let port = uri
        .port_u16()
        .unwrap_or(if scheme == "https" { 443 } else { 80 });
    let host_for_url = if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]")
    } else {
        host.clone()
    };
    Ok(BaseUrl {
        base: format!("{scheme}://{host_for_url}:{port}"),
        host: host
            .trim_start_matches('[')
            .trim_end_matches(']')
            .to_string(),
        port,
    })
}

fn build_agent(pooled: bool) -> ureq::Agent {
    let builder = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .max_redirects(0)
        .accept(super::errors::ALPACA_JSON_MIME)
        .user_agent(concat!("Achernar/", env!("CARGO_PKG_VERSION")))
        .no_delay(true);
    let builder = if pooled {
        builder
            .max_idle_connections(8)
            .max_idle_connections_per_host(2)
            .max_idle_age(Duration::from_secs(15))
    } else {
        builder
            .max_idle_connections(0)
            .max_idle_connections_per_host(0)
    };
    builder.build().new_agent()
}

fn validate_member(member: &str) -> DeviceResult<()> {
    if !member.is_empty()
        && member
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
    {
        Ok(())
    } else {
        Err(DeviceError::invalid_value(format!(
            "'{member}' is not a lower-case Alpaca member name"
        )))
    }
}

fn validate_device_type(device_type: &str) -> DeviceResult<String> {
    let lower = device_type.trim().to_ascii_lowercase();
    if !lower.is_empty() && lower.bytes().all(|b| b.is_ascii_lowercase()) {
        Ok(lower)
    } else {
        Err(DeviceError::invalid_value(format!(
            "'{device_type}' is not an Alpaca device type"
        )))
    }
}

impl AlpacaClient {
    /// A client for device `number` of `device_type` (`telescope`,
    /// `focuser`, ...) on the server at `base` (`http://host:port`).
    pub fn new(
        base: &str,
        device_type: &str,
        number: u32,
        config: ClientConfig,
    ) -> DeviceResult<Self> {
        let url = parse_base_url(base)?;
        let device_type = validate_device_type(device_type)?;
        let prefix = format!("{}/api/v1/{device_type}/{number}/", url.base);
        Ok(Self {
            shared: Arc::new(Shared {
                base: url.base,
                host: url.host,
                port: url.port,
                device_type,
                number,
                prefix,
                config,
                read_agent: build_agent(true),
                command_agent: build_agent(false),
                abort_agent: build_agent(false),
                counters: Counters::default(),
            }),
        })
    }

    /// A client for an [`Endpoint::AlpacaHttp`].
    pub fn for_endpoint(endpoint: &Endpoint, config: ClientConfig) -> DeviceResult<Self> {
        match endpoint {
            Endpoint::AlpacaHttp {
                base,
                device_type,
                number,
            } => Self::new(base, device_type, *number, config),
            other => Err(DeviceError::invalid_value(format!(
                "not an Alpaca endpoint: {}",
                other.describe()
            ))),
        }
    }

    pub fn base(&self) -> &str {
        &self.shared.base
    }

    pub fn host(&self) -> &str {
        &self.shared.host
    }

    pub fn port(&self) -> u16 {
        self.shared.port
    }

    pub fn device_type(&self) -> &str {
        &self.shared.device_type
    }

    pub fn number(&self) -> u32 {
        self.shared.number
    }

    pub fn client_id(&self) -> ClientId {
        self.shared.config.client_id
    }

    pub fn endpoint(&self) -> Endpoint {
        Endpoint::AlpacaHttp {
            base: self.shared.base.clone(),
            device_type: self.shared.device_type.clone(),
            number: self.shared.number,
        }
    }

    /// The Alpaca setup page, `/setup/v1/{type}/{n}/setup` (§2.6 step 5).
    pub fn setup_url(&self) -> String {
        format!(
            "{}/setup/v1/{}/{}/setup",
            self.shared.base, self.shared.device_type, self.shared.number
        )
    }

    /// The §2.6 identity key: `alpaca:{UniqueID}`, or
    /// `alpaca:{host}:{port}/{type}/{n}` without one.
    pub fn identity_key(&self, unique_id: Option<&str>) -> String {
        identity_key::alpaca(
            unique_id,
            &self.shared.host,
            self.shared.port,
            &self.shared.device_type,
            self.shared.number,
        )
    }

    /// The out-of-band abort transport (D4).
    pub fn abort_channel(&self) -> AbortChannel {
        AbortChannel {
            shared: Arc::clone(&self.shared),
        }
    }

    pub fn stats(&self) -> ClientStats {
        self.shared.stats()
    }
}

impl AbortChannel {
    pub fn stats(&self) -> ClientStats {
        self.shared.stats()
    }
}

impl Shared {
    fn stats(&self) -> ClientStats {
        ClientStats {
            requests: self.counters.requests.load(Ordering::Relaxed),
            retries: self.counters.retries.load(Ordering::Relaxed),
            transaction_mismatches: self.counters.transaction_mismatches.load(Ordering::Relaxed),
            missing_content_type: self.counters.missing_content_type.load(Ordering::Relaxed),
        }
    }

    fn agent(&self, lane: Lane) -> &ureq::Agent {
        match lane {
            Lane::Read => &self.read_agent,
            Lane::Command => &self.command_agent,
            Lane::Abort => &self.abort_agent,
        }
    }

    fn timeout_for(&self, member: &str, class: TimeoutClass, lane: Lane) -> Duration {
        let table = &self.config.timeouts;
        let timeout = table
            .get(class)
            .max(table.get(member_timeout_class(member)));
        if lane == Lane::Abort {
            timeout.min(self.config.abort_timeout)
        } else {
            timeout
        }
    }

    fn request(
        &self,
        lane: Lane,
        verb: Verb,
        member: &str,
        params: &[(&str, String)],
        class: TimeoutClass,
    ) -> DeviceResult<Value> {
        validate_member(member)?;
        let timeout = self.timeout_for(member, class, lane);
        // One transaction id per logical request; the retry reuses it, so a
        // server log shows the retry as the same transaction.
        let transaction_id = next_transaction_id();
        self.counters.requests.fetch_add(1, Ordering::Relaxed);
        let first = self.attempt(lane, verb, member, params, transaction_id, timeout);
        let retry_reason = match first {
            Ok(reply) => return Ok(self.accept(reply, transaction_id, member)),
            Err(Failure::Final(error)) => return Err(error),
            Err(Failure::NotSent(error)) => error,
            Err(Failure::MaybeSent(error)) if verb == Verb::Get => error,
            Err(Failure::MaybeSent(mut error)) => {
                error.message.push_str(
                    "; not retried: the PUT may have reached the device, so its outcome is unknown",
                );
                return Err(error);
            }
        };
        tracing::debug!(
            device = %self.prefix, member, error = %retry_reason,
            "Alpaca request failed at the transport; retrying once"
        );
        self.counters.retries.fetch_add(1, Ordering::Relaxed);
        std::thread::sleep(self.config.retry_delay);
        match self.attempt(lane, verb, member, params, transaction_id, timeout) {
            Ok(reply) => Ok(self.accept(reply, transaction_id, member)),
            Err(Failure::Final(error)) => Err(error),
            Err(Failure::NotSent(mut error) | Failure::MaybeSent(mut error)) => {
                error.message.push_str(" (after one retry)");
                Err(error)
            }
        }
    }

    fn accept(&self, reply: AlpacaReply, transaction_id: u32, member: &str) -> Value {
        if reply.content_type_missing {
            self.counters
                .missing_content_type
                .fetch_add(1, Ordering::Relaxed);
        }
        if let Some(echoed) = reply.client_transaction_id {
            if echoed != transaction_id {
                self.counters
                    .transaction_mismatches
                    .fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    device = %self.prefix, member, sent = transaction_id, echoed,
                    "Alpaca server echoed a different ClientTransactionID"
                );
            }
        }
        reply.value
    }

    fn attempt(
        &self,
        lane: Lane,
        verb: Verb,
        member: &str,
        params: &[(&str, String)],
        transaction_id: u32,
        timeout: Duration,
    ) -> Result<AlpacaReply, Failure> {
        let mut pairs: Vec<(&str, String)> = params.to_vec();
        pairs.push((form::CLIENT_ID_KEY, self.config.client_id.get().to_string()));
        pairs.push((form::CLIENT_TRANSACTION_ID_KEY, transaction_id.to_string()));
        let encoded = form::encode_pairs(&pairs);
        let url = format!("{}{member}", self.prefix);
        let connect_timeout = timeout.min(self.config.connect_phase_cap);
        let agent = self.agent(lane);
        let result = match verb {
            Verb::Get => agent
                .get(format!("{url}?{encoded}"))
                .config()
                .timeout_global(Some(timeout))
                .timeout_connect(Some(connect_timeout))
                .build()
                .call(),
            Verb::Put => agent
                .put(&url)
                .config()
                .timeout_global(Some(timeout))
                .timeout_connect(Some(connect_timeout))
                .build()
                .content_type(form::FORM_CONTENT_TYPE)
                .send(encoded.as_bytes()),
        };
        let mut response = result.map_err(classify_error)?;
        let status = response.status().as_u16();
        let content_type = response
            .headers()
            .get(ureq::http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let body = response
            .body_mut()
            .with_config()
            .limit(self.config.max_body_bytes)
            .read_to_vec()
            .map_err(|error| match classify_error(error) {
                // The server already has the request once a body is arriving.
                Failure::NotSent(e) => Failure::MaybeSent(e),
                other => other,
            })?;
        check_response(verb, status, content_type.as_deref(), &body).map_err(Failure::Final)
    }
}

/// Map a `ureq` failure to a device error and say whether the request can
/// have reached the server.
fn classify_error(error: ureq::Error) -> Failure {
    use ureq::Timeout;
    match error {
        ureq::Error::Timeout(which) => {
            let message = format!("Alpaca request timed out ({which:?} phase)");
            match which {
                Timeout::Resolve | Timeout::Connect => {
                    Failure::NotSent(DeviceError::timeout(message))
                }
                _ => Failure::MaybeSent(DeviceError::timeout(message)),
            }
        }
        ureq::Error::Io(io_error) => {
            let message = format!("Alpaca socket error: {io_error}");
            match io_error.kind() {
                // Only possible while connecting. Unreachable-host and
                // unreachable-network errors are not listed: they can also
                // arrive on an established connection after the request was
                // sent, so a PUT must not be re-sent on them.
                io::ErrorKind::ConnectionRefused | io::ErrorKind::AddrNotAvailable => {
                    Failure::NotSent(DeviceError::transport(message))
                }
                io::ErrorKind::TimedOut => Failure::MaybeSent(DeviceError::timeout(message)),
                _ => Failure::MaybeSent(DeviceError::transport(message)),
            }
        }
        ureq::Error::ConnectionFailed => Failure::NotSent(DeviceError::transport(
            "could not connect to the Alpaca server",
        )),
        ureq::Error::HostNotFound => {
            Failure::NotSent(DeviceError::transport("Alpaca server host not found"))
        }
        ureq::Error::Protocol(protocol) => Failure::MaybeSent(DeviceError::transport(format!(
            "malformed HTTP from the Alpaca server: {protocol}"
        ))),
        ureq::Error::BadUri(uri) => Failure::Final(DeviceError::invalid_value(format!(
            "invalid Alpaca URL: {uri}"
        ))),
        ureq::Error::BodyExceedsLimit(limit) => Failure::Final(DeviceError::driver(format!(
            "Alpaca reply is larger than {limit} bytes"
        ))),
        ureq::Error::TooManyRedirects | ureq::Error::RedirectFailed => Failure::Final(
            DeviceError::transport("the Alpaca server answered with a redirect"),
        ),
        other => Failure::Final(DeviceError::transport(format!(
            "Alpaca request failed: {other}"
        ))),
    }
}

impl AlpacaTransport for AlpacaClient {
    fn get(
        &self,
        member: &str,
        query: &[(&str, String)],
        class: TimeoutClass,
    ) -> DeviceResult<Value> {
        self.shared
            .request(Lane::Read, Verb::Get, member, query, class)
    }

    fn put(
        &self,
        member: &str,
        form: &[(&str, String)],
        class: TimeoutClass,
    ) -> DeviceResult<Value> {
        self.shared
            .request(Lane::Command, Verb::Put, member, form, class)
    }
}

impl AlpacaTransport for AbortChannel {
    fn get(
        &self,
        member: &str,
        query: &[(&str, String)],
        class: TimeoutClass,
    ) -> DeviceResult<Value> {
        self.shared
            .request(Lane::Abort, Verb::Get, member, query, class)
    }

    fn put(
        &self,
        member: &str,
        form: &[(&str, String)],
        class: TimeoutClass,
    ) -> DeviceResult<Value> {
        self.shared
            .request(Lane::Abort, Verb::Put, member, form, class)
    }
}

// ---------------------------------------------------------------------------
// Abort paths
// ---------------------------------------------------------------------------

/// One PUT of an abort sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct AbortStep {
    pub member: String,
    pub form: Vec<(&'static str, String)>,
}

/// An [`AbortPath`] that PUTs one or more members over the abort channel,
/// e.g. `abortslew` (mount), `halt` (focuser, rotator), `haltcover`. Every
/// step is attempted even when an earlier one fails (best effort, D19); the
/// first error is returned.
pub struct PutAbortPath {
    transport: Arc<dyn AlpacaTransport>,
    steps: Vec<AbortStep>,
}

impl PutAbortPath {
    pub fn new(transport: Arc<dyn AlpacaTransport>, member: &str) -> Self {
        Self {
            transport,
            steps: vec![AbortStep {
                member: member.to_ascii_lowercase(),
                form: Vec::new(),
            }],
        }
    }

    /// Add a step, e.g. `moveaxis` with `Axis=0, Rate=0` after `abortslew`.
    pub fn then(mut self, member: &str, form: Vec<(&'static str, String)>) -> Self {
        self.steps.push(AbortStep {
            member: member.to_ascii_lowercase(),
            form,
        });
        self
    }

    pub fn steps(&self) -> &[AbortStep] {
        &self.steps
    }
}

impl AbortPath for PutAbortPath {
    fn abort(&self) -> DeviceResult<()> {
        let mut first_error = None;
        for step in &self.steps {
            if let Err(error) =
                self.transport
                    .put(&step.member, &step.form, member_timeout_class(&step.member))
            {
                tracing::warn!(member = %step.member, %error, "Alpaca abort step failed");
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

// ---------------------------------------------------------------------------
// Timing and the C-2 read policy
// ---------------------------------------------------------------------------

/// When one request was observed (D5): the midpoint of the request on the
/// host clock, and its latency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestTiming {
    pub observed_at: DateTime<Utc>,
    pub mono_ns: u64,
    pub latency_ms: u32,
    pub started_mono_ns: u64,
    pub finished_mono_ns: u64,
}

/// Run one call and time it with `clock`.
pub fn timed<T>(clock: &dyn Clock, call: impl FnOnce() -> T) -> (T, RequestTiming) {
    let started_utc = clock.now_utc();
    let started_mono_ns = clock.now_mono_ns();
    let result = call();
    let finished_mono_ns = clock.now_mono_ns().max(started_mono_ns);
    let half_ns = (finished_mono_ns - started_mono_ns) / 2;
    let observed_at =
        started_utc + chrono::Duration::nanoseconds(i64::try_from(half_ns).unwrap_or(i64::MAX));
    let latency_ms =
        u32::try_from((finished_mono_ns - started_mono_ns) / 1_000_000).unwrap_or(u32::MAX);
    (
        result,
        RequestTiming {
            observed_at,
            mono_ns: started_mono_ns + half_ns,
            latency_ms,
            started_mono_ns,
            finished_mono_ns,
        },
    )
}

/// One member read classified by the C-2 property policy.
#[derive(Debug, Clone, PartialEq)]
pub enum MemberRead<T> {
    Value(T),
    /// ASCOM ValueNotSet: unknown, not a failure.
    NotSet,
    /// NotImplemented, or HTTP 400/404 on the member: memoise it as
    /// unsupported for the session.
    Unsupported(DeviceError),
    /// Any other failure: keep the last value, marked `Stale`.
    Failed(DeviceError),
}

impl<T> MemberRead<T> {
    pub fn classify(result: DeviceResult<T>) -> Self {
        match result {
            Ok(value) => MemberRead::Value(value),
            Err(error) if error.is_not_set() => MemberRead::NotSet,
            Err(error) if error.is_unknown_member() => MemberRead::Unsupported(error),
            Err(error) => MemberRead::Failed(error),
        }
    }

    /// The `Observed` value this read produces. `previous` is the member's
    /// last observation, kept as `Stale` when this read failed (C-2).
    pub fn into_observed(
        self,
        source: Source,
        timing: &RequestTiming,
        previous: Option<Observed<T>>,
    ) -> Observed<T> {
        match self {
            MemberRead::Value(value) => {
                Observed::fresh(value, source, timing.observed_at, timing.mono_ns)
                    .with_latency_ms(timing.latency_ms)
            }
            MemberRead::NotSet => Observed {
                observed_at: Some(timing.observed_at),
                mono_ns: Some(timing.mono_ns),
                latency_ms: Some(timing.latency_ms),
                ..Observed::not_set(source)
            },
            MemberRead::Unsupported(_) => Observed::unsupported(source),
            MemberRead::Failed(_) => match previous {
                Some(previous) => previous.into_stale(),
                None => Observed::error(source),
            },
        }
    }

    pub fn error(&self) -> Option<&DeviceError> {
        match self {
            MemberRead::Unsupported(error) | MemberRead::Failed(error) => Some(error),
            MemberRead::Value(_) | MemberRead::NotSet => None,
        }
    }

    /// The observation state this read implies.
    pub fn state(&self) -> ObsState {
        match self {
            MemberRead::Value(_) => ObsState::Fresh,
            MemberRead::NotSet => ObsState::NotSet,
            MemberRead::Unsupported(_) => ObsState::Unsupported,
            MemberRead::Failed(_) => ObsState::Error,
        }
    }
}

/// Members memoised as unsupported for the session (§2.6 step 6, C-2).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UnsupportedMembers {
    members: BTreeSet<String>,
}

impl UnsupportedMembers {
    pub fn contains(&self, member: &str) -> bool {
        self.members.contains(&member.to_ascii_lowercase())
    }

    pub fn insert(&mut self, member: &str) {
        self.members.insert(member.to_ascii_lowercase());
    }

    /// Memoise `member` when `error` says the device does not know it.
    /// Returns true when it was memoised.
    pub fn note(&mut self, member: &str, error: &DeviceError) -> bool {
        let unknown = error.is_unknown_member() || error.class == ErrorClass::Unsupported;
        if unknown {
            self.insert(member);
        }
        unknown
    }

    pub fn to_vec(&self) -> Vec<String> {
        self.members.iter().cloned().collect()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::testing::alpaca_server::{
        HttpScript, RawAction, ScriptedDevice, ServerReply, TestAlpacaServer,
    };
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::types::ascom_error;
    use serde_json::json;
    use std::sync::Mutex;
    use std::time::Instant;

    fn quick_config() -> ClientConfig {
        let mut config = ClientConfig::new(ClientId::new(4242).unwrap());
        config.timeouts = TimeoutTable {
            connect: Duration::from_millis(400),
            ordinary: Duration::from_millis(400),
            long: Duration::from_millis(400),
        };
        config.abort_timeout = Duration::from_millis(400);
        config.retry_delay = Duration::from_millis(20);
        config
    }

    fn client_for(
        server: &TestAlpacaServer,
        device_type: &str,
        config: ClientConfig,
    ) -> AlpacaClient {
        AlpacaClient::new(&server.base_url(), device_type, 0, config).unwrap()
    }

    fn focuser_server() -> TestAlpacaServer {
        TestAlpacaServer::start(vec![ScriptedDevice::new("focuser", 0, "Test focuser")
            .property("position", json!(1200))
            .property("ismoving", json!(false))])
    }

    #[test]
    fn the_c3_error_table_maps_every_alpaca_error_number() {
        let cases = [
            (ascom_error::NOT_IMPLEMENTED, ErrorClass::Unsupported),
            (ascom_error::INVALID_VALUE, ErrorClass::InvalidValue),
            (ascom_error::NOT_CONNECTED, ErrorClass::NotConnected),
            (
                ascom_error::INVALID_WHILE_PARKED,
                ErrorClass::InvalidWhileParked,
            ),
            (
                ascom_error::INVALID_WHILE_SLAVED,
                ErrorClass::InvalidOperation,
            ),
            (ascom_error::INVALID_OPERATION, ErrorClass::InvalidOperation),
            (ascom_error::ACTION_NOT_IMPLEMENTED, ErrorClass::Unsupported),
            (ascom_error::OPERATION_CANCELLED, ErrorClass::Cancelled),
            (0x500, ErrorClass::Driver),
            (0x6A1, ErrorClass::Driver),
            (0xFFF, ErrorClass::Driver),
        ];
        let mut device = ScriptedDevice::new("telescope", 0, "Errors");
        for (index, (number, _)) in cases.iter().enumerate() {
            device = device.on_get(
                &format!("m{index}"),
                ServerReply::AlpacaError {
                    number: *number,
                    message: format!("error {number:#x}"),
                },
            );
        }
        let device = device.on_get(
            "notset",
            ServerReply::AlpacaError {
                number: ascom_error::VALUE_NOT_SET,
                message: "not set".into(),
            },
        );
        let server = TestAlpacaServer::start(vec![device]);
        let client = client_for(&server, "telescope", quick_config());
        for (index, (number, class)) in cases.iter().enumerate() {
            let err = client
                .get(&format!("m{index}"), &[], TimeoutClass::Ordinary)
                .unwrap_err();
            assert_eq!(err.class, *class, "{number:#x}");
            assert_eq!(err.alpaca_error_number(), Some(*number));
            assert_eq!(err.message, format!("error {number:#x}"));
        }
        let err = client
            .get("notset", &[], TimeoutClass::Ordinary)
            .unwrap_err();
        assert!(err.is_not_set(), "ValueNotSet is unknown, not a failure");
        assert!(matches!(
            MemberRead::classify(Err::<(), _>(err)),
            MemberRead::NotSet
        ));
        assert_eq!(client.stats().retries, 0, "Alpaca errors are never retried");
    }

    #[test]
    fn plain_text_http_400_and_500_bodies_are_kept() {
        let device = ScriptedDevice::new("focuser", 0, "Plain")
            .on_get(
                "devicestate",
                ServerReply::text(400, "Unknown member: devicestate\r\n"),
            )
            .on_put("move", ServerReply::text(500, "Stepper driver fault"))
            .on_get("gone", ServerReply::text(404, "Not Found"));
        let server = TestAlpacaServer::start(vec![device]);
        let client = client_for(&server, "focuser", quick_config());

        let err = client
            .get("devicestate", &[], TimeoutClass::Ordinary)
            .unwrap_err();
        assert_eq!(err.http_status(), Some(400));
        assert_eq!(err.class, ErrorClass::InvalidValue);
        assert!(err.is_unknown_member());
        assert!(
            err.message.contains("Unknown member: devicestate"),
            "{}",
            err.message
        );
        let mut memo = UnsupportedMembers::default();
        assert!(memo.note("devicestate", &err));
        assert!(memo.contains("DeviceState"));

        let err = client
            .put("move", &[("Position", form::int(100))], TimeoutClass::Long)
            .unwrap_err();
        assert_eq!(err.http_status(), Some(500));
        assert_eq!(err.class, ErrorClass::Driver);
        assert!(err.message.contains("Stepper driver fault"));

        let err = client.get("gone", &[], TimeoutClass::Ordinary).unwrap_err();
        assert_eq!(err.class, ErrorClass::Unsupported);
        // An unscripted member answers like an older server: HTTP 400.
        let err = client
            .get("covermoving", &[], TimeoutClass::Ordinary)
            .unwrap_err();
        assert!(err.is_unknown_member());
        assert_eq!(client.stats().retries, 0, "HTTP errors are never retried");
        assert_eq!(server.requests().len(), 4);
    }

    #[test]
    fn a_null_value_on_error_is_not_parsed_and_wrong_content_types_are_refused() {
        let device = ScriptedDevice::new("telescope", 0, "Null")
            .on_get("rightascension", ServerReply::raw(
                200,
                Some("application/json"),
                r#"{"Value": null, "ErrorNumber": 1031, "ErrorMessage": "Not connected", "ServerTransactionID": 1}"#,
            ))
            .on_get("html", ServerReply::raw(200, Some("text/html"), "<html>captive portal</html>"))
            .on_get("bare", ServerReply::raw(200, None, r#"{"Value": 12, "ErrorNumber": 0, "ErrorMessage": ""}"#));
        let server = TestAlpacaServer::start(vec![device]);
        let client = client_for(&server, "telescope", quick_config());
        let err = client
            .get("rightascension", &[], TimeoutClass::Ordinary)
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::NotConnected);
        assert_eq!(err.message, "Not connected");
        let err = client.get("html", &[], TimeoutClass::Ordinary).unwrap_err();
        assert_eq!(err.class, ErrorClass::Transport);
        assert!(err.message.contains("text/html"));
        assert_eq!(
            client.get("bare", &[], TimeoutClass::Ordinary).unwrap(),
            json!(12)
        );
        assert_eq!(client.stats().missing_content_type, 1);
    }

    #[test]
    fn requests_carry_client_ids_and_exact_case_form_parameters() {
        let device = ScriptedDevice::new("telescope", 0, "Params")
            .on_put("slewtocoordinatesasync", ServerReply::Ok)
            .property("rightascension", json!(5.5));
        let server = TestAlpacaServer::start(vec![device]);
        let client = client_for(&server, "telescope", quick_config());
        client
            .put(
                "slewtocoordinatesasync",
                &[
                    ("RightAscension", form::num(5.5).unwrap()),
                    ("Declination", form::num(-5.25).unwrap()),
                ],
                TimeoutClass::Long,
            )
            .unwrap();
        assert_eq!(
            client
                .get("rightascension", &[], TimeoutClass::Ordinary)
                .unwrap(),
            json!(5.5)
        );
        let requests = server.requests();
        assert_eq!(requests.len(), 2);
        let put = &requests[0];
        assert_eq!(put.verb, Verb::Put);
        assert_eq!(put.content_type.as_deref(), Some(form::FORM_CONTENT_TYPE));
        assert_eq!(form::find(&put.params, "RightAscension", true), Some("5.5"));
        assert_eq!(form::find(&put.params, "Declination", true), Some("-5.25"));
        assert_eq!(form::find(&put.params, "ClientID", true), Some("4242"));
        assert!(form::find(&put.params, "ClientTransactionID", true).is_some());
        let get = &requests[1];
        assert_eq!(get.verb, Verb::Get);
        assert_eq!(form::find(&get.params, "clientid", false), Some("4242"));
        assert!(get.client_transaction_id.unwrap() > put.client_transaction_id.unwrap());
    }

    #[test]
    fn transaction_ids_are_global_monotonic_and_never_zero() {
        let server = focuser_server();
        let first = client_for(&server, "focuser", quick_config());
        let second = client_for(&server, "focuser", quick_config());
        for _ in 0..5 {
            first.get("position", &[], TimeoutClass::Ordinary).unwrap();
            second.get("ismoving", &[], TimeoutClass::Ordinary).unwrap();
        }
        let ids: Vec<u32> = server
            .requests()
            .iter()
            .map(|r| r.client_transaction_id.unwrap())
            .collect();
        assert_eq!(ids.len(), 10);
        assert!(ids.iter().all(|id| *id >= 1));
        assert!(
            ids.windows(2).all(|w| w[0] < w[1]),
            "one counter for every client: {ids:?}"
        );

        // From several std threads at once: every id is unique.
        server.clear_requests();
        let client = Arc::new(client_for(&server, "focuser", quick_config()));
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let client = Arc::clone(&client);
                std::thread::spawn(move || {
                    for _ in 0..10 {
                        client.get("position", &[], TimeoutClass::Ordinary).unwrap();
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        let mut ids: Vec<u32> = server
            .requests()
            .iter()
            .map(|r| r.client_transaction_id.unwrap())
            .collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 40);
        assert_eq!(client.stats().transaction_mismatches, 0);
    }

    #[test]
    fn a_mismatched_transaction_echo_is_counted_but_not_fatal() {
        let device = ScriptedDevice::new("focuser", 0, "Echo").on_get(
            "position",
            ServerReply::raw(
                200,
                Some("application/json"),
                r#"{"Value": 7, "ClientTransactionID": 4000000000, "ServerTransactionID": 1, "ErrorNumber": 0, "ErrorMessage": ""}"#,
            ),
        );
        let server = TestAlpacaServer::start(vec![device]);
        let client = client_for(&server, "focuser", quick_config());
        assert_eq!(
            client.get("position", &[], TimeoutClass::Ordinary).unwrap(),
            json!(7)
        );
        assert_eq!(client.stats().transaction_mismatches, 1);
    }

    #[test]
    fn transaction_counter_skips_zero_on_wrap() {
        // Exercise the wrap rule on a scratch counter with the same logic.
        let counter = AtomicU32::new(u32::MAX);
        let next = || loop {
            let id = counter.fetch_add(1, Ordering::Relaxed);
            if id != 0 {
                return id;
            }
        };
        assert_eq!(next(), u32::MAX);
        assert_eq!(next(), 1, "0 is skipped after the wrap");
        assert!(next_transaction_id() >= 1);
    }

    #[test]
    fn a_get_is_retried_once_after_a_socket_error() {
        let ok = HttpScript::alpaca_ok(json!(1200));
        let script = HttpScript::start(vec![RawAction::CloseWithoutReply, ok]);
        let client = AlpacaClient::new(&script.base_url(), "focuser", 0, quick_config()).unwrap();
        let started = Instant::now();
        assert_eq!(
            client.get("position", &[], TimeoutClass::Ordinary).unwrap(),
            json!(1200)
        );
        assert!(
            started.elapsed() >= Duration::from_millis(20),
            "waits the retry delay"
        );
        assert_eq!(client.stats().retries, 1);
        let seen = script.requests();
        assert_eq!(seen.len(), 2);
        assert_eq!(
            seen[0].transaction_id(),
            seen[1].transaction_id(),
            "the retry is the same transaction"
        );
    }

    #[test]
    fn a_get_is_retried_once_after_a_timeout_and_then_fails() {
        let script = HttpScript::start(vec![
            RawAction::Stall(Duration::from_millis(900)),
            RawAction::Stall(Duration::from_millis(900)),
        ]);
        let client = AlpacaClient::new(&script.base_url(), "focuser", 0, quick_config()).unwrap();
        let err = client
            .get("position", &[], TimeoutClass::Ordinary)
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::Timeout, "{err}");
        assert!(err.message.contains("after one retry"), "{}", err.message);
        assert_eq!(client.stats().retries, 1);
        assert_eq!(script.requests().len(), 2, "exactly one retry");
    }

    #[test]
    fn a_put_that_may_have_been_delivered_is_never_resent() {
        // Timed out after the request was sent.
        let script = HttpScript::start(vec![
            RawAction::Stall(Duration::from_millis(900)),
            HttpScript::alpaca_ok(Value::Null),
        ]);
        let client = AlpacaClient::new(&script.base_url(), "focuser", 0, quick_config()).unwrap();
        let err = client
            .put("move", &[("Position", form::int(50))], TimeoutClass::Long)
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::Timeout);
        assert!(
            err.message.contains("outcome is unknown"),
            "{}",
            err.message
        );
        assert_eq!(client.stats().retries, 0);
        assert_eq!(
            script.requests().len(),
            1,
            "a relative move must not be sent twice"
        );

        // Connection dropped after the request was read.
        let script = HttpScript::start(vec![
            RawAction::CloseWithoutReply,
            HttpScript::alpaca_ok(Value::Null),
        ]);
        let client = AlpacaClient::new(&script.base_url(), "telescope", 0, quick_config()).unwrap();
        let err = client
            .put(
                "pulseguide",
                &[("Direction", form::int(0)), ("Duration", form::int(300))],
                TimeoutClass::Long,
            )
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::Transport);
        assert_eq!(script.requests().len(), 1);
    }

    #[test]
    fn a_put_that_never_reached_the_device_is_retried_once() {
        // Nothing listens on this port, so the request provably was not
        // delivered. Unix refuses the connection at once; Windows retries the
        // SYN and the connect phase times out instead. Both are connect-phase
        // failures and both are retried.
        let port = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        };
        let client = AlpacaClient::new(
            &format!("http://127.0.0.1:{port}"),
            "telescope",
            0,
            quick_config(),
        )
        .unwrap();
        let err = client
            .put("abortslew", &[], TimeoutClass::Long)
            .unwrap_err();
        assert!(
            matches!(err.class, ErrorClass::Transport | ErrorClass::Timeout),
            "{err}"
        );
        assert!(err.message.contains("after one retry"), "{}", err.message);
        assert_eq!(client.stats().retries, 1);
        assert_eq!(client.stats().requests, 1);
    }

    #[test]
    fn malformed_and_definitive_replies_are_not_retried() {
        let script = HttpScript::start(vec![
            RawAction::Respond {
                status: 200,
                content_type: Some("application/json".into()),
                body: "{oops".into(),
            },
            HttpScript::alpaca_ok(json!(1)),
        ]);
        let client = AlpacaClient::new(&script.base_url(), "focuser", 0, quick_config()).unwrap();
        let err = client
            .get("position", &[], TimeoutClass::Ordinary)
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::Driver);
        assert_eq!(client.stats().retries, 0);
        assert_eq!(script.requests().len(), 1);
    }

    #[test]
    fn timeout_classes_follow_c3_and_never_shorten_a_long_member() {
        assert_eq!(member_timeout_class("connecting"), TimeoutClass::Connect);
        assert_eq!(
            member_timeout_class("InterfaceVersion"),
            TimeoutClass::Connect
        );
        for member in [
            "abortslew",
            "findhome",
            "moveaxis",
            "park",
            "unpark",
            "pulseguide",
            "slewtocoordinatesasync",
            "slewtoaltazasync",
            "synctocoordinates",
            "move",
            "halt",
            "moveabsolute",
            "movemechanical",
        ] {
            assert_eq!(member_timeout_class(member), TimeoutClass::Long, "{member}");
        }
        assert_eq!(
            member_timeout_class("rightascension"),
            TimeoutClass::Ordinary
        );
        assert_eq!(member_timeout_class("devicestate"), TimeoutClass::Ordinary);
        assert_eq!(
            TimeoutTable::default().get(TimeoutClass::Connect),
            Duration::from_secs(5)
        );
        assert_eq!(
            TimeoutTable::default().get(TimeoutClass::Ordinary),
            Duration::from_secs(10)
        );
        assert_eq!(
            TimeoutTable::default().get(TimeoutClass::Long),
            Duration::from_secs(100)
        );

        let client = AlpacaClient::new(
            "http://127.0.0.1:1",
            "telescope",
            0,
            ClientConfig::new(ClientId::new(1).unwrap()),
        )
        .unwrap();
        let shared = &client.shared;
        assert_eq!(
            shared.timeout_for("park", TimeoutClass::Ordinary, Lane::Command),
            Duration::from_secs(100)
        );
        assert_eq!(
            shared.timeout_for("connected", TimeoutClass::Connect, Lane::Read),
            Duration::from_secs(5)
        );
        assert_eq!(
            shared.timeout_for("tracking", TimeoutClass::Long, Lane::Command),
            Duration::from_secs(100)
        );
        assert_eq!(
            shared.timeout_for("abortslew", TimeoutClass::Long, Lane::Abort),
            Duration::from_secs(5)
        );
    }

    #[test]
    fn the_abort_channel_does_not_queue_behind_a_blocked_request() {
        let device = ScriptedDevice::new("focuser", 0, "Slow")
            .on_put(
                "move",
                ServerReply::Delay(Duration::from_millis(1500), Box::new(ServerReply::Ok)),
            )
            .on_put("halt", ServerReply::Ok);
        let server = TestAlpacaServer::start(vec![device]);
        let mut config = quick_config();
        config.timeouts.long = Duration::from_secs(5);
        let client = client_for(&server, "focuser", config);
        let abort: Arc<dyn AlpacaTransport> = Arc::new(client.abort_channel());
        let mover = client.clone();
        let blocked = std::thread::spawn(move || {
            mover.put("move", &[("Position", form::int(5000))], TimeoutClass::Long)
        });
        while server.requests().is_empty() {
            std::thread::sleep(Duration::from_millis(5));
        }
        let started = Instant::now();
        PutAbortPath::new(abort, "halt").abort().unwrap();
        assert!(
            started.elapsed() < Duration::from_millis(1000),
            "abort waited {:?}",
            started.elapsed()
        );
        assert!(blocked.join().unwrap().is_ok());
        let members: Vec<String> = server.requests().iter().map(|r| r.member.clone()).collect();
        assert_eq!(members, vec!["move", "halt"]);
    }

    #[test]
    fn put_abort_path_attempts_every_step_and_reports_the_first_error() {
        let fake = Arc::new(crate::equipment::testing::fake_transport::FakeAlpacaTransport::new());
        fake.on_put(
            "abortslew",
            crate::equipment::testing::fake_transport::FakeReply::Timeout,
        );
        fake.on_put(
            "moveaxis",
            crate::equipment::testing::fake_transport::FakeReply::ok(),
        );
        let path = PutAbortPath::new(fake.clone(), "AbortSlew").then(
            "moveaxis",
            vec![("Axis", form::int(0)), ("Rate", form::int(0))],
        );
        let err = path.abort().unwrap_err();
        assert_eq!(err.class, ErrorClass::Timeout);
        let calls = fake.calls();
        assert_eq!(calls.len(), 2, "the second step still ran");
        assert_eq!(calls[0].class, TimeoutClass::Long);
        assert_eq!(calls[1].params[0], ("Axis".to_string(), "0".to_string()));
    }

    #[test]
    fn base_urls_are_validated_and_identity_follows_section_2_6() {
        let config = ClientConfig::new(ClientId::new(9).unwrap());
        let client = AlpacaClient::new("HTTP://ESP32.local", "Focuser", 1, config.clone()).unwrap();
        assert_eq!(client.base(), "http://esp32.local:80");
        assert_eq!(client.device_type(), "focuser");
        assert_eq!(
            client.setup_url(),
            "http://esp32.local:80/setup/v1/focuser/1/setup"
        );
        assert_eq!(client.identity_key(None), "alpaca:esp32.local:80/focuser/1");
        assert_eq!(client.identity_key(Some("u-1")), "alpaca:u-1");
        assert_eq!(
            client.endpoint(),
            Endpoint::AlpacaHttp {
                base: "http://esp32.local:80".into(),
                device_type: "focuser".into(),
                number: 1
            }
        );
        let ipv6 =
            AlpacaClient::new("http://[::1]:11111/", "telescope", 0, config.clone()).unwrap();
        assert_eq!(ipv6.base(), "http://[::1]:11111");
        assert_eq!(ipv6.host(), "::1");
        for bad in [
            "",
            "auto",
            "ftp://host:1",
            "http://user:pw@host:1",
            "http://host:1/api",
            "http://host:1?x=1",
            "http://host:1#device=0",
        ] {
            assert!(
                AlpacaClient::new(bad, "telescope", 0, config.clone()).is_err(),
                "{bad}"
            );
        }
        assert!(AlpacaClient::new("http://h:1", "tele scope", 0, config.clone()).is_err());
        let err = client
            .get("Position", &[], TimeoutClass::Ordinary)
            .unwrap_err();
        assert_eq!(
            err.class,
            ErrorClass::InvalidValue,
            "member names are lower-case"
        );
        assert!(
            AlpacaClient::for_endpoint(&Endpoint::InProcess { name: "x".into() }, config).is_err()
        );
    }

    #[test]
    fn client_ids_are_configured_or_persisted_per_install() {
        let dir = std::env::temp_dir().join(format!("achernar-clientid-{}", uuid::Uuid::new_v4()));
        let (configured, source) = ClientId::load_or_create(&dir, 777).unwrap();
        assert_eq!(
            (configured.get(), source),
            (777, ClientIdSource::Configured)
        );
        assert!(!dir.exists(), "a configured id writes nothing");

        let (generated, source) = ClientId::load_or_create(&dir, 0).unwrap();
        assert_eq!(source, ClientIdSource::Generated);
        assert!((1..=65_535).contains(&generated.get()));
        let (again, source) = ClientId::load_or_create(&dir, 0).unwrap();
        assert_eq!((again, source), (generated, ClientIdSource::Persisted));
        // Out of range in the config: ignored in favour of the persisted id.
        let (kept, source) = ClientId::load_or_create(&dir, 70_000).unwrap();
        assert_eq!((kept, source), (generated, ClientIdSource::Persisted));

        std::fs::write(dir.join(CLIENT_ID_FILE), "0\n").unwrap();
        let (replaced, source) = ClientId::load_or_create(&dir, 0).unwrap();
        assert_eq!(source, ClientIdSource::Generated);
        assert_ne!(replaced.get(), 0);
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp-"))
            .collect();
        assert!(leftovers.is_empty());
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(ClientId::new(0), None);
        assert_eq!(ClientId::new(65_536), None);
        assert!(ClientId::new(65_535).is_some());
    }

    #[test]
    fn timed_reads_use_the_request_midpoint_and_follow_the_c2_policy() {
        let clock = ManualClock::new();
        let start = clock.now_utc();
        let (value, timing) = timed(&clock, || {
            clock.advance(Duration::from_millis(80));
            Ok::<_, DeviceError>(json!(12.5))
        });
        assert_eq!(timing.latency_ms, 80);
        assert_eq!(timing.mono_ns, 40_000_000);
        assert_eq!(
            timing.observed_at,
            start + chrono::Duration::milliseconds(40)
        );
        let parsed = value.and_then(|v| super::super::errors::expect_f64("temperature", &v));
        let observed = MemberRead::classify(parsed).into_observed(Source::Property, &timing, None);
        assert!(observed.is_fresh());
        assert_eq!(observed.latency_ms, Some(80));

        let previous = observed.clone();
        let failed: MemberRead<f64> = MemberRead::classify(Err(DeviceError::timeout("slow")));
        assert_eq!(failed.state(), ObsState::Error);
        let stale = failed.into_observed(Source::Property, &timing, Some(previous));
        assert_eq!(stale.state, ObsState::Stale);
        assert_eq!(stale.known(), Some(&12.5));
        let never: MemberRead<f64> = MemberRead::classify(Err(DeviceError::transport("down")));
        assert_eq!(
            never.into_observed(Source::Property, &timing, None).state,
            ObsState::Error
        );

        let not_set: MemberRead<f64> = MemberRead::classify(Err(DeviceError::from_alpaca_error(
            ascom_error::VALUE_NOT_SET,
            "",
        )));
        let observed = not_set.into_observed(Source::Property, &timing, None);
        assert_eq!((observed.state, observed.value), (ObsState::NotSet, None));
        assert!(observed.observed_at.is_some(), "the device was asked");

        let unsupported: MemberRead<f64> =
            MemberRead::classify(Err(DeviceError::from_http_status(404, "")));
        assert!(unsupported.error().is_some());
        let observed = unsupported.into_observed(Source::Property, &timing, None);
        assert_eq!(observed.state, ObsState::Unsupported);
    }

    #[test]
    fn the_client_is_safe_to_share_between_std_threads() {
        // The worker thread polls while an abort runs on another thread.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AlpacaClient>();
        assert_send_sync::<AbortChannel>();
        assert_send_sync::<PutAbortPath>();
        let server = focuser_server();
        let client = client_for(&server, "focuser", quick_config());
        let results = Arc::new(Mutex::new(Vec::new()));
        let workers: Vec<_> = (0..3)
            .map(|_| {
                let client = client.clone();
                let results = Arc::clone(&results);
                std::thread::spawn(move || {
                    let value = client.get("position", &[], TimeoutClass::Ordinary);
                    results.lock().unwrap().push(value.is_ok());
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(*results.lock().unwrap(), vec![true, true, true]);
    }
}

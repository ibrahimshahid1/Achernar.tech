//! Alpaca addressing, the management API and opt-in UDP discovery (D13,
//! §2.6, §2.12). Owner: WP1A-1.
//!
//! **Nothing here runs by itself, and nothing is broadcast unless
//! `[equipment] allow_broadcast_discovery = true`** (D13). Every function is
//! blocking and is called from `spawn_blocking` or a worker thread.
//!
//! - [`parse_option`]: the `/connect` option for an Alpaca device,
//!   `http(s)://host:port#device=N`. An empty or `"auto"` option means
//!   "find it by broadcast" and is refused unless broadcast is enabled.
//! - [`ManagementClient`]: `/management/apiversions`, `/v1/description` and
//!   `/v1/configureddevices`, with the same ClientID, ClientTransactionID
//!   and C-3 checks as device requests. [`ConfiguredDevices`] finds a
//!   device's UniqueID (the identity key, §2.6) and re-finds a device whose
//!   number changed by its stored UniqueID.
//! - [`discover`]: the Alpaca discovery protocol over IPv4. It sends the
//!   ASCII datagram `alpacadiscovery1` to the discovery port (default 32227,
//!   configurable) of the limited broadcast address and of every IPv4
//!   interface's subnet broadcast address (Unix; on Windows the limited
//!   broadcast only), or to explicit addresses, then collects
//!   `{"AlpacaPort": n}` replies until the timeout. It refuses to run at all
//!   unless the caller passes `allow_broadcast_discovery = true`.

use std::collections::BTreeSet;
use std::io;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, SocketAddrV4, UdpSocket};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::client::{agent_settings, next_transaction_id, parse_base_url, AgentBuilder, ClientId};
use super::errors::{check_response, expect_string, expect_u32, field, Verb};
use super::form;
use crate::equipment::types::{identity_key, DeviceError, DeviceResult, Endpoint};

/// The discovery request datagram.
pub const DISCOVERY_MESSAGE: &[u8] = b"alpacadiscovery1";
/// The default Alpaca discovery port (`[equipment] alpaca_discovery_port`).
pub const DEFAULT_DISCOVERY_PORT: u16 = 32227;

fn broadcast_refused() -> DeviceError {
    DeviceError::invalid_operation(
        "Alpaca broadcast discovery is disabled; give an explicit http(s)://host:port#device=N \
         or set [equipment] allow_broadcast_discovery = true",
    )
}

// ---------------------------------------------------------------------------
// The /connect option
// ---------------------------------------------------------------------------

/// An explicitly addressed Alpaca device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlpacaAddress {
    /// `scheme://host:port`.
    pub base: String,
    pub host: String,
    pub port: u16,
    /// The Alpaca DeviceNumber (`#device=N`; 0 when omitted).
    pub number: u32,
}

impl AlpacaAddress {
    pub fn endpoint(&self, device_type: &str) -> Endpoint {
        Endpoint::AlpacaHttp {
            base: self.base.clone(),
            device_type: device_type.to_ascii_lowercase(),
            number: self.number,
        }
    }

    /// The §2.6 identity key for this address.
    pub fn identity_key(&self, device_type: &str, unique_id: Option<&str>) -> String {
        identity_key::alpaca(unique_id, &self.host, self.port, device_type, self.number)
    }

    /// Round-trips through [`parse_option`].
    pub fn option(&self) -> String {
        format!("{}#device={}", self.base, self.number)
    }
}

/// What an Alpaca `/connect` option asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlpacaOption {
    Address(AlpacaAddress),
    /// Empty or `"auto"`: find the device by broadcast. Only returned when
    /// broadcast discovery is enabled.
    Broadcast,
}

/// Parse an Alpaca `/connect` option (§2.12): `http(s)://host:port`, with an
/// optional `#device=N`. An empty or `"auto"` option is refused unless
/// `allow_broadcast_discovery` (D13).
pub fn parse_option(option: &str, allow_broadcast_discovery: bool) -> DeviceResult<AlpacaOption> {
    let option = option.trim();
    if option.is_empty() || option.eq_ignore_ascii_case("auto") {
        return if allow_broadcast_discovery {
            Ok(AlpacaOption::Broadcast)
        } else {
            Err(broadcast_refused())
        };
    }
    let (base, fragment) = match option.split_once('#') {
        Some((base, fragment)) => (base, Some(fragment)),
        None => (option, None),
    };
    let number = match fragment {
        None => 0,
        Some(fragment) => fragment
            .strip_prefix("device=")
            .and_then(|n| n.trim().parse::<u32>().ok())
            .ok_or_else(|| {
                DeviceError::invalid_value(format!(
                    "Alpaca option '{option}': the fragment must be #device=N"
                ))
            })?,
    };
    let url = parse_base_url(base)?;
    Ok(AlpacaOption::Address(AlpacaAddress {
        base: url.base,
        host: url.host,
        port: url.port,
        number,
    }))
}

// ---------------------------------------------------------------------------
// Management API
// ---------------------------------------------------------------------------

/// `/management/v1/description`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ServerDescription {
    pub server_name: Option<String>,
    pub manufacturer: Option<String>,
    pub manufacturer_version: Option<String>,
    pub location: Option<String>,
}

/// One `/management/v1/configureddevices` entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfiguredDevice {
    pub device_name: String,
    /// Lower-case Alpaca type (`telescope`, `filterwheel`, ...).
    pub device_type: String,
    pub device_number: u32,
    /// `None` when the server sends none or an empty string.
    pub unique_id: Option<String>,
}

/// The configured devices of one server, with the entries that were skipped
/// and why. A configured entry is what the server reports, not proof that
/// hardware is attached; that is verified only on connect.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ConfiguredDevices {
    pub devices: Vec<ConfiguredDevice>,
    pub skipped: Vec<String>,
}

impl ConfiguredDevices {
    pub fn of_type<'a>(
        &'a self,
        device_type: &'a str,
    ) -> impl Iterator<Item = &'a ConfiguredDevice> {
        self.devices
            .iter()
            .filter(move |d| d.device_type.eq_ignore_ascii_case(device_type))
    }

    pub fn find(&self, device_type: &str, number: u32) -> Option<&ConfiguredDevice> {
        self.devices
            .iter()
            .find(|d| d.device_type.eq_ignore_ascii_case(device_type) && d.device_number == number)
    }

    /// The UniqueID for the identity key (§2.6 step 5).
    pub fn unique_id(&self, device_type: &str, number: u32) -> Option<&str> {
        self.find(device_type, number)
            .and_then(|d| d.unique_id.as_deref())
    }

    /// Re-find a device by its stored UniqueID (its number may have
    /// changed; §2.6 "Re-finding an Alpaca device").
    pub fn by_unique_id(&self, unique_id: &str) -> Option<&ConfiguredDevice> {
        let unique_id = unique_id.trim();
        if unique_id.is_empty() {
            return None;
        }
        self.devices
            .iter()
            .find(|d| d.unique_id.as_deref() == Some(unique_id))
    }
}

/// Parse a `configureddevices` Value. Invalid or duplicate entries are
/// skipped with a reason, never repaired.
pub fn parse_configured_devices(value: &Value) -> DeviceResult<ConfiguredDevices> {
    let items = value
        .as_array()
        .ok_or_else(|| DeviceError::driver("configureddevices Value is not an array"))?;
    let mut out = ConfiguredDevices::default();
    let mut seen = BTreeSet::new();
    for (index, item) in items.iter().enumerate() {
        let Some(object) = item.as_object() else {
            out.skipped.push(format!("entry {index} is not an object"));
            continue;
        };
        let text = |name: &str| {
            field(object, name)
                .and_then(|v| expect_string(name, v).ok())
                .map(|s| s.trim().to_string())
        };
        let device_type = text("DeviceType")
            .filter(|t| !t.is_empty() && t.bytes().all(|b| b.is_ascii_alphabetic()))
            .map(|t| t.to_ascii_lowercase());
        let number = field(object, "DeviceNumber").and_then(|v| expect_u32("DeviceNumber", v).ok());
        let name = text("DeviceName").filter(|n| !n.is_empty());
        let (Some(device_type), Some(device_number), Some(device_name)) =
            (device_type, number, name)
        else {
            out.skipped.push(format!(
                "entry {index} lacks a valid DeviceType, DeviceNumber or DeviceName"
            ));
            continue;
        };
        if !seen.insert((device_type.clone(), device_number)) {
            out.skipped.push(format!(
                "entry {index} repeats {device_type} number {device_number}"
            ));
            continue;
        }
        out.devices.push(ConfiguredDevice {
            device_name,
            device_type,
            device_number,
            unique_id: text("UniqueID").filter(|id| !id.is_empty()),
        });
    }
    Ok(out)
}

/// The management agent's settings: the shared Alpaca ones (no proxy, no
/// redirects, the Alpaca `Accept`), one overall timeout, no pooling.
fn management_settings(builder: AgentBuilder, timeout: Duration) -> AgentBuilder {
    agent_settings(builder)
        .timeout_global(Some(timeout))
        .max_idle_connections(0)
}

/// The Alpaca management API of one server. Blocking.
pub struct ManagementClient {
    base: String,
    client_id: ClientId,
    timeout: Duration,
    agent: ureq::Agent,
}

impl ManagementClient {
    /// `base` is `http(s)://host:port`; `timeout` bounds each request.
    pub fn new(base: &str, client_id: ClientId, timeout: Duration) -> DeviceResult<Self> {
        let url = parse_base_url(base)?;
        let agent = management_settings(ureq::Agent::config_builder(), timeout)
            .build()
            .new_agent();
        Ok(Self {
            base: url.base,
            client_id,
            timeout,
            agent,
        })
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    fn get(&self, path: &str) -> DeviceResult<Value> {
        let transaction_id = next_transaction_id();
        let query = form::encode_pairs(&[
            (form::CLIENT_ID_KEY, self.client_id.get().to_string()),
            (form::CLIENT_TRANSACTION_ID_KEY, transaction_id.to_string()),
        ]);
        let url = format!("{}/management/{path}?{query}", self.base);
        let mut response = self
            .agent
            .get(&url)
            .config()
            .timeout_global(Some(self.timeout))
            .build()
            .call()
            .map_err(|error| match error {
                ureq::Error::Timeout(_) => {
                    DeviceError::timeout(format!("Alpaca management request timed out: {path}"))
                }
                other => {
                    DeviceError::transport(format!("Alpaca management request failed: {other}"))
                }
            })?;
        let status = response.status().as_u16();
        let content_type = response
            .headers()
            .get(ureq::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let body = response
            .body_mut()
            .with_config()
            .limit(1024 * 1024)
            .read_to_vec()
            .map_err(|error| {
                DeviceError::transport(format!("could not read the management reply: {error}"))
            })?;
        check_response(Verb::Get, status, content_type.as_deref(), &body).map(|reply| reply.value)
    }

    /// `/management/apiversions`.
    pub fn api_versions(&self) -> DeviceResult<Vec<u32>> {
        let value = self.get("apiversions")?;
        value
            .as_array()
            .ok_or_else(|| DeviceError::driver("apiversions Value is not an array"))?
            .iter()
            .map(|v| expect_u32("apiversions", v))
            .collect()
    }

    /// `/management/v1/description`.
    pub fn description(&self) -> DeviceResult<ServerDescription> {
        let value = self.get("v1/description")?;
        let object = value
            .as_object()
            .ok_or_else(|| DeviceError::driver("description Value is not an object"))?;
        let text = |name: &str| {
            field(object, name)
                .and_then(Value::as_str)
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        };
        Ok(ServerDescription {
            server_name: text("ServerName"),
            manufacturer: text("Manufacturer"),
            manufacturer_version: text("ManufacturerVersion"),
            location: text("Location"),
        })
    }

    /// `/management/v1/configureddevices`.
    pub fn configured_devices(&self) -> DeviceResult<ConfiguredDevices> {
        parse_configured_devices(&self.get("v1/configureddevices")?)
    }
}

// ---------------------------------------------------------------------------
// UDP discovery
// ---------------------------------------------------------------------------

/// Where discovery datagrams go.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoveryTargets {
    /// 255.255.255.255 plus each IPv4 interface's subnet broadcast address.
    Broadcast,
    /// These addresses only (a known host, or a chosen subnet broadcast).
    Explicit(Vec<Ipv4Addr>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveryOptions {
    /// `[equipment] alpaca_discovery_port`.
    pub port: u16,
    /// How long replies are collected.
    pub timeout: Duration,
    pub targets: DiscoveryTargets,
    /// Stop early after this many distinct servers.
    pub max_servers: usize,
}

impl Default for DiscoveryOptions {
    fn default() -> Self {
        Self {
            port: DEFAULT_DISCOVERY_PORT,
            timeout: Duration::from_secs(2),
            targets: DiscoveryTargets::Broadcast,
            max_servers: 64,
        }
    }
}

/// One server that answered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveredServer {
    pub address: IpAddr,
    pub alpaca_port: u16,
    /// Milliseconds after the request was sent.
    pub replied_after_ms: u64,
}

impl DiscoveredServer {
    /// `http://address:port`, ready for [`ManagementClient::new`].
    pub fn base_url(&self) -> String {
        match self.address {
            IpAddr::V4(v4) => format!("http://{v4}:{}", self.alpaca_port),
            IpAddr::V6(v6) => format!("http://[{v6}]:{}", self.alpaca_port),
        }
    }
}

/// The outcome of one discovery round.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct DiscoveryReport {
    pub servers: Vec<DiscoveredServer>,
    /// The addresses the request was sent to.
    pub sent_to: Vec<String>,
    /// Datagrams that were not a valid `{"AlpacaPort": n}` reply.
    pub ignored_datagrams: u64,
    /// Targets the request could not be sent to, with the error.
    pub send_errors: Vec<String>,
}

/// The subnet broadcast address of an interface address and netmask, or
/// `None` for a host route (/32) or a /31 point-to-point link.
pub fn subnet_broadcast(address: Ipv4Addr, netmask: Ipv4Addr) -> Option<Ipv4Addr> {
    let mask = u32::from(netmask);
    if mask.count_ones() >= 31 {
        return None;
    }
    Some(Ipv4Addr::from(u32::from(address) | !mask))
}

/// The broadcast targets: the limited broadcast and, on Unix, every up,
/// non-loopback IPv4 interface's subnet broadcast.
pub fn broadcast_targets() -> Vec<Ipv4Addr> {
    let mut targets = vec![Ipv4Addr::BROADCAST];
    for target in interface_broadcasts() {
        if !targets.contains(&target) {
            targets.push(target);
        }
    }
    targets
}

#[cfg(unix)]
fn interface_broadcasts() -> Vec<Ipv4Addr> {
    fn ipv4(address: *const libc::sockaddr) -> Option<Ipv4Addr> {
        if address.is_null() {
            return None;
        }
        // SAFETY: `address` is non-null and points to a sockaddr owned by
        // the getifaddrs list, which stays alive until freeifaddrs below.
        let family = i32::from(unsafe { (*address).sa_family });
        if family != libc::AF_INET {
            return None;
        }
        // SAFETY: the family is AF_INET, so the storage is a sockaddr_in.
        let v4 = unsafe { &*address.cast::<libc::sockaddr_in>() };
        Some(Ipv4Addr::from(u32::from_be(v4.sin_addr.s_addr)))
    }

    let mut out = Vec::new();
    let mut head: *mut libc::ifaddrs = std::ptr::null_mut();
    // SAFETY: getifaddrs writes a list head we free with freeifaddrs.
    if unsafe { libc::getifaddrs(&mut head) } != 0 {
        return out;
    }
    let mut cursor = head;
    while !cursor.is_null() {
        // SAFETY: `cursor` is a non-null node of the getifaddrs list.
        let entry = unsafe { &*cursor };
        let flags = entry.ifa_flags;
        let wanted = flags & (libc::IFF_UP as libc::c_uint) != 0
            && flags & (libc::IFF_BROADCAST as libc::c_uint) != 0
            && flags & (libc::IFF_LOOPBACK as libc::c_uint) == 0;
        if wanted {
            if let (Some(address), Some(mask)) = (ipv4(entry.ifa_addr), ipv4(entry.ifa_netmask)) {
                if let Some(broadcast) = subnet_broadcast(address, mask) {
                    if !out.contains(&broadcast) {
                        out.push(broadcast);
                    }
                }
            }
        }
        cursor = entry.ifa_next;
    }
    // SAFETY: `head` came from a successful getifaddrs and is freed once.
    unsafe { libc::freeifaddrs(head) };
    out
}

#[cfg(not(unix))]
fn interface_broadcasts() -> Vec<Ipv4Addr> {
    // Windows: no interface enumeration without another crate; the limited
    // broadcast and explicit targets are used.
    Vec::new()
}

/// Parse one discovery reply: `{"AlpacaPort": n}` with n in 1..=65535.
pub fn parse_discovery_reply(datagram: &[u8]) -> Option<u16> {
    let value: Value = serde_json::from_slice(datagram).ok()?;
    let port = field(value.as_object()?, "AlpacaPort")?.as_u64()?;
    u16::try_from(port).ok().filter(|port| *port != 0)
}

/// Run one discovery round. Refused with `InvalidOperation`, before any
/// socket is opened, unless `allow_broadcast_discovery` (D13).
pub fn discover(
    allow_broadcast_discovery: bool,
    options: &DiscoveryOptions,
) -> DeviceResult<DiscoveryReport> {
    if !allow_broadcast_discovery {
        return Err(broadcast_refused());
    }
    if options.port == 0 {
        return Err(DeviceError::invalid_value(
            "the Alpaca discovery port must not be 0",
        ));
    }
    let targets = match &options.targets {
        DiscoveryTargets::Broadcast => broadcast_targets(),
        DiscoveryTargets::Explicit(addresses) => addresses.clone(),
    };
    let io_error =
        |error: io::Error| DeviceError::transport(format!("Alpaca discovery socket: {error}"));
    let socket = UdpSocket::bind(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0)).map_err(io_error)?;
    socket.set_broadcast(true).map_err(io_error)?;

    let mut report = DiscoveryReport::default();
    let sent_at = Instant::now();
    let mut delivered = 0;
    for target in &targets {
        let destination = SocketAddr::V4(SocketAddrV4::new(*target, options.port));
        report.sent_to.push(destination.to_string());
        match socket.send_to(DISCOVERY_MESSAGE, destination) {
            Ok(_) => delivered += 1,
            Err(error) => report.send_errors.push(format!("{destination}: {error}")),
        }
    }
    if delivered == 0 {
        return Err(DeviceError::transport(format!(
            "Alpaca discovery could not send to any target: {}",
            report.send_errors.join("; ")
        )));
    }

    let deadline = sent_at + options.timeout;
    let mut seen = BTreeSet::new();
    let mut buffer = [0_u8; 1024];
    loop {
        let now = Instant::now();
        if now >= deadline || report.servers.len() >= options.max_servers {
            break;
        }
        socket
            .set_read_timeout(Some((deadline - now).max(Duration::from_millis(1))))
            .map_err(io_error)?;
        match socket.recv_from(&mut buffer) {
            Ok((length, from)) => {
                let datagram = &buffer[..length];
                match parse_discovery_reply(datagram) {
                    Some(port) if seen.insert((from.ip(), port)) => {
                        report.servers.push(DiscoveredServer {
                            address: from.ip(),
                            alpaca_port: port,
                            replied_after_ms: u64::try_from(sent_at.elapsed().as_millis())
                                .unwrap_or(u64::MAX),
                        });
                    }
                    Some(_) => {}
                    None => report.ignored_datagrams += 1,
                }
            }
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock
                        | io::ErrorKind::TimedOut
                        | io::ErrorKind::Interrupted
                ) => {}
            // Windows reports an ICMP port-unreachable from an earlier send
            // as a reset on the next receive; it is not fatal.
            Err(error) if error.kind() == io::ErrorKind::ConnectionReset => {}
            Err(error) => return Err(io_error(error)),
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::testing::alpaca_server::{ScriptedDevice, ServerReply, TestAlpacaServer};
    use crate::equipment::types::ErrorClass;
    use serde_json::json;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    /// A loopback discovery responder on an ephemeral port. It answers each
    /// `alpacadiscovery1` with every reply in `replies`.
    struct Responder {
        port: u16,
        received: Arc<Mutex<Vec<Vec<u8>>>>,
        stop: Arc<AtomicBool>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl Responder {
        fn start(replies: Vec<Vec<u8>>) -> Self {
            let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
            socket
                .set_read_timeout(Some(Duration::from_millis(20)))
                .unwrap();
            let port = socket.local_addr().unwrap().port();
            let received = Arc::new(Mutex::new(Vec::new()));
            let stop = Arc::new(AtomicBool::new(false));
            let thread = {
                let received = Arc::clone(&received);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let mut buffer = [0_u8; 256];
                    while !stop.load(Ordering::Relaxed) {
                        if let Ok((length, from)) = socket.recv_from(&mut buffer) {
                            received.lock().unwrap().push(buffer[..length].to_vec());
                            if &buffer[..length] == DISCOVERY_MESSAGE {
                                for reply in &replies {
                                    let _ = socket.send_to(reply, from);
                                }
                            }
                        }
                    }
                })
            };
            Self {
                port,
                received,
                stop,
                thread: Some(thread),
            }
        }

        fn received(&self) -> Vec<Vec<u8>> {
            self.received.lock().unwrap().clone()
        }
    }

    impl Drop for Responder {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }

    fn loopback(port: u16, timeout_ms: u64) -> DiscoveryOptions {
        DiscoveryOptions {
            port,
            timeout: Duration::from_millis(timeout_ms),
            targets: DiscoveryTargets::Explicit(vec![Ipv4Addr::LOCALHOST]),
            max_servers: 64,
        }
    }

    #[test]
    fn discovery_finds_a_loopback_responder_on_an_ephemeral_port() {
        let responder = Responder::start(vec![
            br#"{"AlpacaPort": 11111}"#.to_vec(),
            br#"{"AlpacaPort": 11111}"#.to_vec(),
            b"not json".to_vec(),
            br#"{"AlpacaPort": 0}"#.to_vec(),
            br#"{"alpacaport": 32323}"#.to_vec(),
        ]);
        let report = discover(true, &loopback(responder.port, 400)).unwrap();
        let ports: Vec<u16> = report.servers.iter().map(|s| s.alpaca_port).collect();
        assert_eq!(
            ports,
            vec![11111, 32323],
            "duplicates are merged, bad replies ignored"
        );
        assert_eq!(report.ignored_datagrams, 2);
        assert_eq!(report.servers[0].address, IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_eq!(report.servers[0].base_url(), "http://127.0.0.1:11111");
        assert_eq!(
            report.sent_to,
            vec![format!("127.0.0.1:{}", responder.port)]
        );
        assert_eq!(responder.received(), vec![DISCOVERY_MESSAGE.to_vec()]);
    }

    #[test]
    fn discovery_is_refused_and_sends_nothing_unless_enabled() {
        let responder = Responder::start(vec![br#"{"AlpacaPort": 11111}"#.to_vec()]);
        let err = discover(false, &loopback(responder.port, 100)).unwrap_err();
        assert_eq!(err.class, ErrorClass::InvalidOperation);
        assert!(err.message.contains("allow_broadcast_discovery"));
        std::thread::sleep(Duration::from_millis(60));
        assert!(responder.received().is_empty(), "no datagram was sent");
        let err = discover(
            true,
            &DiscoveryOptions {
                port: 0,
                ..loopback(1, 10)
            },
        )
        .unwrap_err();
        assert_eq!(err.class, ErrorClass::InvalidValue);
    }

    #[test]
    fn discovery_with_no_responder_returns_an_empty_report_in_time() {
        let port = {
            let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
            socket.local_addr().unwrap().port()
        };
        let started = Instant::now();
        let report = discover(true, &loopback(port, 150)).unwrap();
        assert!(report.servers.is_empty());
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn replies_and_broadcast_addresses_are_parsed_strictly() {
        assert_eq!(parse_discovery_reply(br#"{"AlpacaPort":4567}"#), Some(4567));
        assert_eq!(parse_discovery_reply(br#"{"AlpacaPort":70000}"#), None);
        assert_eq!(parse_discovery_reply(br#"{"AlpacaPort":"80"}"#), None);
        assert_eq!(parse_discovery_reply(b"alpacadiscovery1"), None);
        assert_eq!(
            subnet_broadcast(
                Ipv4Addr::new(192, 168, 1, 37),
                Ipv4Addr::new(255, 255, 255, 0)
            ),
            Some(Ipv4Addr::new(192, 168, 1, 255))
        );
        assert_eq!(
            subnet_broadcast(Ipv4Addr::new(10, 1, 2, 3), Ipv4Addr::new(255, 255, 0, 0)),
            Some(Ipv4Addr::new(10, 1, 255, 255))
        );
        assert_eq!(
            subnet_broadcast(Ipv4Addr::new(10, 0, 0, 1), Ipv4Addr::BROADCAST),
            None
        );
        let targets = broadcast_targets();
        assert_eq!(targets[0], Ipv4Addr::BROADCAST);
        assert!(
            !targets.contains(&Ipv4Addr::new(127, 255, 255, 255)),
            "loopback is skipped"
        );
    }

    #[test]
    fn options_are_explicit_unless_broadcast_is_enabled() {
        for refused in ["", "  ", "auto", "AUTO"] {
            let err = parse_option(refused, false).unwrap_err();
            assert_eq!(err.class, ErrorClass::InvalidOperation, "{refused:?}");
            assert_eq!(
                parse_option(refused, true).unwrap(),
                AlpacaOption::Broadcast
            );
        }
        let AlpacaOption::Address(address) =
            parse_option("http://ESP32.local:80#device=1", false).unwrap()
        else {
            panic!("expected an address");
        };
        assert_eq!(address.base, "http://esp32.local:80");
        assert_eq!(
            (address.host.as_str(), address.port, address.number),
            ("esp32.local", 80, 1)
        );
        assert_eq!(address.option(), "http://esp32.local:80#device=1");
        assert_eq!(
            parse_option(&address.option(), false).unwrap(),
            AlpacaOption::Address(address.clone())
        );
        assert_eq!(
            address.identity_key("Focuser", None),
            "alpaca:esp32.local:80/focuser/1"
        );
        assert_eq!(address.identity_key("focuser", Some("abc")), "alpaca:abc");
        assert_eq!(
            address.endpoint("Focuser"),
            Endpoint::AlpacaHttp {
                base: "http://esp32.local:80".into(),
                device_type: "focuser".into(),
                number: 1
            }
        );
        let AlpacaOption::Address(default) = parse_option("http://10.0.0.5:11111", false).unwrap()
        else {
            panic!("expected an address");
        };
        assert_eq!(default.number, 0);
        for bad in [
            "http://h:1#dev=1",
            "http://h:1#device=-1",
            "http://h:1/api#device=0",
            "telnet://h:1",
            "h:1",
        ] {
            assert!(parse_option(bad, false).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_management_api_lists_devices_and_finds_unique_ids() {
        let server = TestAlpacaServer::start(vec![
            ScriptedDevice::esp32_like_mount(0),
            ScriptedDevice::platform6_focuser(0),
            ScriptedDevice::new("filterwheel", 2, "Wheel").unique_id("wheel-uid"),
        ]);
        let management = ManagementClient::new(
            &server.base_url(),
            ClientId::new(55).unwrap(),
            Duration::from_secs(2),
        )
        .unwrap();
        assert_eq!(management.api_versions().unwrap(), vec![1]);
        let description = management.description().unwrap();
        assert_eq!(
            description.server_name.as_deref(),
            Some("Achernar test Alpaca server")
        );
        let devices = management.configured_devices().unwrap();
        assert_eq!(devices.devices.len(), 3);
        assert!(devices.skipped.is_empty());
        assert_eq!(
            devices.unique_id("telescope", 0),
            Some("test-esp32-mount-0001")
        );
        assert_eq!(
            devices.unique_id("Focuser", 0),
            None,
            "an empty UniqueID is none"
        );
        assert_eq!(devices.by_unique_id("wheel-uid").unwrap().device_number, 2);
        assert_eq!(devices.find("filterwheel", 2).unwrap().device_name, "Wheel");
        assert!(devices.by_unique_id("").is_none());
        let requests = server.requests();
        assert!(requests
            .iter()
            .all(|r| r.client_id == Some(55) && r.client_transaction_id.is_some()));
    }

    #[test]
    fn configured_device_lists_skip_invalid_and_duplicate_entries() {
        let parsed = parse_configured_devices(&json!([
            {"DeviceName": "Mount", "DeviceType": "Telescope", "DeviceNumber": 0, "UniqueID": "m"},
            {"DeviceName": "Mount 2", "DeviceType": "telescope", "DeviceNumber": 0, "UniqueID": "m2"},
            {"DeviceName": "", "DeviceType": "Focuser", "DeviceNumber": 0},
            {"DeviceName": "Neg", "DeviceType": "Focuser", "DeviceNumber": -1},
            {"DeviceName": "Bad type", "DeviceType": "Tele scope", "DeviceNumber": 1},
            "junk",
            {"devicename": "Weather", "devicetype": "ObservingConditions", "devicenumber": 0},
        ]))
        .unwrap();
        assert_eq!(parsed.devices.len(), 2);
        assert_eq!(parsed.skipped.len(), 5);
        assert_eq!(parsed.devices[1].device_type, "observingconditions");
        assert!(parse_configured_devices(&json!({"a": 1})).is_err());
    }

    #[test]
    fn management_errors_follow_the_c3_checks() {
        let server = TestAlpacaServer::start(vec![
            ScriptedDevice::new("focuser", 0, "F").on_get("position", ServerReply::value(1))
        ]);
        let management = ManagementClient::new(
            &server.base_url(),
            ClientId::new(1).unwrap(),
            Duration::from_millis(500),
        )
        .unwrap();
        assert!(management.base().starts_with("http://127.0.0.1:"));
        drop(server);
        let err = management.api_versions().unwrap_err();
        assert!(
            matches!(err.class, ErrorClass::Transport | ErrorClass::Timeout),
            "{err}"
        );
        assert!(
            ManagementClient::new("auto", ClientId::new(1).unwrap(), Duration::from_secs(1))
                .is_err()
        );
    }

    #[test]
    fn management_requests_never_go_through_a_proxy() {
        // As ureq's default builder is when HTTP_PROXY or HTTPS_PROXY is set;
        // the process environment is not touched.
        let proxied = ureq::Agent::config_builder()
            .proxy(Some(ureq::Proxy::new("http://127.0.0.1:9").unwrap()));
        let agent = management_settings(proxied, Duration::from_secs(1))
            .build()
            .new_agent();
        assert!(agent.config().proxy().is_none());
        let management = ManagementClient::new(
            "http://127.0.0.1:1",
            ClientId::new(1).unwrap(),
            Duration::from_secs(1),
        )
        .unwrap();
        assert!(management.agent.config().proxy().is_none());
    }
}

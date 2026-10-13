use futures::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::time::{interval, sleep, Duration};
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::{relay_commands::RelayCommand, state::AppState};

const DEFAULT_RELAY_DEV_TOKEN: &str = "achernar-dev-token";

#[derive(Clone, Debug)]
pub struct CloudClientConfig {
    pub cloud_http: String,
    pub cloud_ws: Option<String>,
    pub relay_token: String,
    pub daemon_id: Option<String>,
    pub hardware_id: String,
}

#[derive(Debug, Deserialize)]
struct RegisterResponse {
    daemon_id: String,
    gateway_url: String,
}

impl CloudClientConfig {
    pub fn from_env() -> Option<Self> {
        let cloud_http = std::env::var("ACHERNAR_CLOUD_HTTP").ok()?;
        let relay_token = std::env::var("ACHERNAR_RELAY_DEV_TOKEN")
            .unwrap_or_else(|_| DEFAULT_RELAY_DEV_TOKEN.to_string());
        let hardware_id =
            std::env::var("ACHERNAR_HARDWARE_ID").unwrap_or_else(|_| local_hardware_id());

        Some(Self {
            cloud_http,
            cloud_ws: std::env::var("ACHERNAR_CLOUD_WS").ok(),
            relay_token,
            daemon_id: std::env::var("ACHERNAR_DAEMON_ID").ok(),
            hardware_id,
        })
    }

    fn register_url(&self) -> String {
        let base = self.cloud_http.trim_end_matches('/');
        if base.ends_with("/api") {
            format!("{base}/sync/register/")
        } else {
            format!("{base}/api/sync/register/")
        }
    }

    fn gateway_url(&self, registration: &RegisterResponse) -> String {
        let base = self
            .cloud_ws
            .as_deref()
            .map(|url| url.trim_end_matches('/'));
        let url = if let Some(base) = base {
            if base.ends_with("/ws") {
                format!("{base}/gateway/{}/", registration.daemon_id)
            } else {
                format!("{base}/ws/gateway/{}/", registration.daemon_id)
            }
        } else {
            registration.gateway_url.clone()
        };

        let separator = if url.contains('?') { '&' } else { '?' };
        format!("{url}{separator}token={}", self.relay_token)
    }
}

pub fn spawn_if_configured(state: AppState) {
    if let Some(config) = CloudClientConfig::from_env() {
        tokio::spawn(async move {
            run_forever(state, config).await;
        });
    } else {
        tracing::info!("ACHERNAR_CLOUD_HTTP not set; Django reverse tunnel disabled");
    }
}

async fn run_forever(state: AppState, config: CloudClientConfig) {
    let mut backoff = Duration::from_secs(1);
    // Accepted hardware operations survive a gateway disconnect. Keep their
    // capacity bound across reconnects, rather than creating another allowance
    // while old native operations are still running.
    let command_slots = std::sync::Arc::new(tokio::sync::Semaphore::new(16));
    loop {
        match register(&config).await {
            Ok(registration) => {
                let gateway_url = config.gateway_url(&registration);
                tracing::info!(daemon_id = %registration.daemon_id, "Registered daemon with Django relay");
                match run_gateway(
                    state.clone(),
                    config.clone(),
                    registration,
                    gateway_url,
                    command_slots.clone(),
                )
                .await
                {
                    Ok(()) => backoff = Duration::from_secs(1),
                    Err(err) => {
                        tracing::warn!("Django gateway disconnected: {err}");
                    }
                }
            }
            Err(err) => {
                tracing::warn!("Django relay registration failed: {err}");
            }
        }

        sleep(backoff).await;
        backoff = (backoff * 2).min(Duration::from_secs(30));
    }
}

async fn register(config: &CloudClientConfig) -> Result<RegisterResponse, String> {
    let config = config.clone();
    tokio::task::spawn_blocking(move || {
        let body = json!({
            "daemon_id": config.daemon_id,
            "hardware_id": config.hardware_id,
            "daemon_version": env!("CARGO_PKG_VERSION"),
        });
        let mut response = ureq::post(&config.register_url())
            .header("Content-Type", "application/json")
            .header("X-Achernar-Relay-Token", &config.relay_token)
            .send(body.to_string())
            .map_err(|e| e.to_string())?;
        let text = response
            .body_mut()
            .read_to_string()
            .map_err(|e| e.to_string())?;
        serde_json::from_str::<RegisterResponse>(&text).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

async fn run_gateway(
    state: AppState,
    config: CloudClientConfig,
    registration: RegisterResponse,
    gateway_url: String,
    command_slots: std::sync::Arc<tokio::sync::Semaphore>,
) -> Result<(), String> {
    let (ws, _) = connect_async(&gateway_url)
        .await
        .map_err(|e| e.to_string())?;
    let (mut write, mut read) = ws.split();

    let hello = json!({
        "type": "HELLO",
        "daemon_id": registration.daemon_id,
        "hardware_id": config.hardware_id,
        "daemon_version": env!("CARGO_PKG_VERSION"),
    });
    write
        .send(Message::Text(hello.to_string()))
        .await
        .map_err(|e| e.to_string())?;

    let mut heartbeat = interval(Duration::from_secs(10));
    let mut telemetry = interval(Duration::from_secs(1));
    let mut device_rx = state.device_updates.subscribe();
    // Hardware commands may take seconds. Their completion must not block
    // disconnect events, telemetry or heartbeats on the gateway writer.
    let (result_tx, mut result_rx) = tokio::sync::mpsc::channel::<Value>(16);

    loop {
        tokio::select! {
            _ = heartbeat.tick() => {
                let msg = json!({
                    "type": "HEARTBEAT",
                    "timestamp": chrono::Utc::now().timestamp(),
                });
                write.send(Message::Text(msg.to_string())).await.map_err(|e| e.to_string())?;
            }
            _ = telemetry.tick() => {
                let msg = crate::telemetry::telemetry_event(&state);
                write.send(Message::Text(msg.to_string())).await.map_err(|e| e.to_string())?;
            }
            device_result = device_rx.recv() => {
                if matches!(device_result, Err(tokio::sync::broadcast::error::RecvError::Closed)) { return Ok(()); }
                let msg=crate::telemetry::telemetry_event(&state);
                write.send(Message::Text(msg.to_string())).await.map_err(|e|e.to_string())?;
            }
            Some(result) = result_rx.recv() => {
                write.send(Message::Text(result.to_string())).await.map_err(|e|e.to_string())?;
            }
            incoming = read.next() => {
                let incoming = incoming.ok_or_else(|| "gateway stream ended".to_string())?;
                match incoming.map_err(|e| e.to_string())? {
                    Message::Text(text) => {
                        let command_state=state.clone();
                        let command_text=text.clone();
                        let accepted=spawn_bounded_command(&command_slots,result_tx.clone(),async move {
                            handle_gateway_text(command_state,&command_text).await
                        });
                        if accepted.is_err() {
                                let parsed=serde_json::from_str::<Value>(&text).unwrap_or(Value::Null);
                                let busy=json!({"type":"COMMAND_RESULT","command_id":parsed.get("command_id"),
                                    "ok":false,"error":"Daemon command capacity is busy; retry after pending operations complete"});
                                write.send(Message::Text(busy.to_string())).await.map_err(|e|e.to_string())?;
                        }
                    }
                    Message::Close(_) => return Ok(()),
                    _ => {}
                }
            }
        }
    }
}

/// Relay arms include synchronous vendor calls and std mutex acquisition.
/// Polling those futures on Tokio workers could starve health deadlines and
/// telemetry. A blocking worker drives each accepted future while the gateway
/// remains free to service device events. No receiver lifetime cancels hardware.
fn spawn_bounded_command<F>(
    slots: &std::sync::Arc<tokio::sync::Semaphore>,
    completed: tokio::sync::mpsc::Sender<Value>,
    command: F,
) -> Result<(), ()>
where
    F: std::future::Future<Output = Option<Value>> + Send + 'static,
{
    let permit = slots.clone().try_acquire_owned().map_err(|_| ())?;
    let runtime = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        if let Some(result) = runtime.block_on(command) {
            let _ = completed.blocking_send(result);
        }
    });
    Ok(())
}

async fn handle_gateway_text(state: AppState, text: &str) -> Option<Value> {
    let data: Value = match serde_json::from_str(text) {
        Ok(data) => data,
        Err(err) => {
            tracing::warn!("Ignoring malformed gateway message: {err}");
            return None;
        }
    };

    if data.get("type").and_then(Value::as_str) != Some("COMMAND") {
        return None;
    }

    let command = match serde_json::from_value::<RelayCommand>(data) {
        Ok(command) => command,
        Err(err) => {
            return Some(json!({
                "type": "COMMAND_RESULT",
                "command_id": null,
                "ok": false,
                "error": format!("Invalid command payload: {err}"),
            }));
        }
    };

    let command_id = command.command_id.clone();
    match crate::relay_commands::execute(state, &command).await {
        Ok(payload) => Some(json!({
            "type": "COMMAND_RESULT",
            "command_id": command_id,
            "ok": true,
            "payload": payload,
        })),
        Err(error) => Some(json!({
            "type": "COMMAND_RESULT",
            "command_id": command_id,
            "ok": false,
            "error": error,
        })),
    }
}

fn local_hardware_id() -> String {
    let hostname = std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "local".to_string());
    format!("achernar-{hostname}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn blocking_command_does_not_starve_runtime() {
        let slots = std::sync::Arc::new(tokio::sync::Semaphore::new(1));
        let (results, mut received) = tokio::sync::mpsc::channel(1);
        let (release, wait) = std::sync::mpsc::channel();
        let (started, ready) = tokio::sync::oneshot::channel();
        spawn_bounded_command(&slots, results, async move {
            let _ = started.send(());
            let _ = wait.recv_timeout(std::time::Duration::from_secs(1));
            Some(json!({"complete":true}))
        }).unwrap();
        ready.await.unwrap();
        // A synchronous SDK call must leave even a single Tokio worker free.
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(matches!(received.try_recv(), Err(tokio::sync::mpsc::error::TryRecvError::Empty)));
        release.send(()).unwrap();
        assert_eq!(tokio::time::timeout(Duration::from_secs(2), received.recv()).await.unwrap(), Some(json!({"complete":true})));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn disconnected_gateway_keeps_global_capacity_and_accepted_work() {
        let slots = std::sync::Arc::new(tokio::sync::Semaphore::new(1));
        let reconnect_slots = slots.clone();
        let (results, received) = tokio::sync::mpsc::channel(1);
        let (release, wait) = std::sync::mpsc::channel();
        let (started, ready) = tokio::sync::oneshot::channel();
        let finished = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let completed = finished.clone();
        spawn_bounded_command(&slots, results, async move {
            let _ = started.send(());
            let _ = wait.recv_timeout(std::time::Duration::from_secs(1));
            completed.store(true, std::sync::atomic::Ordering::SeqCst);
            Some(json!({"complete":true}))
        }).unwrap();
        ready.await.unwrap();
        drop(received);
        let (new_results, _new_received) = tokio::sync::mpsc::channel(1);
        assert!(spawn_bounded_command(&reconnect_slots, new_results, async { None }).is_err());
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while slots.available_permits() == 0 { tokio::task::yield_now().await; }
        }).await.unwrap();
        assert!(finished.load(std::sync::atomic::Ordering::SeqCst));
    }
}

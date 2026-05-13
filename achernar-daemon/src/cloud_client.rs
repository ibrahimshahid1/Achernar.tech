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
        let base = self.cloud_ws.as_deref().map(str::trim_end_matches);
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
    loop {
        match register(&config).await {
            Ok(registration) => {
                let gateway_url = config.gateway_url(&registration);
                tracing::info!(daemon_id = %registration.daemon_id, "Registered daemon with Django relay");
                match run_gateway(state.clone(), config.clone(), registration, gateway_url).await {
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
            incoming = read.next() => {
                let incoming = incoming.ok_or_else(|| "gateway stream ended".to_string())?;
                match incoming.map_err(|e| e.to_string())? {
                    Message::Text(text) => {
                        if let Some(result) = handle_gateway_text(state.clone(), &text).await {
                            write.send(Message::Text(result.to_string())).await.map_err(|e| e.to_string())?;
                        }
                    }
                    Message::Close(_) => return Ok(()),
                    _ => {}
                }
            }
        }
    }
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

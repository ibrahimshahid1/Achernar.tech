use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::State,
    response::Response,
    routing::{get, post},
    Router,
};
use std::net::{IpAddr, SocketAddr};
use tokio::time::{interval, Duration};
use tower_http::cors::CorsLayer;
use tracing::{info, Level};

mod ai;
mod api;
mod cloud_client;
mod config;
mod control;
mod error;
mod filter_wheel;
mod fits;
mod focus;
mod hardware;
mod models;
mod mount;
mod processing;
mod relay_commands;
mod state;
mod storage;
mod telemetry;
mod telemetry_log;
mod vision;

use state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();

    info!("Starting Achernar Daemon v{}", env!("CARGO_PKG_VERSION"));

    // Report OLS bridge version
    unsafe {
        let ver = std::ffi::CStr::from_ptr(hardware::ols_ffi::ols_bridge_version());
        info!("OLS Bridge version: {}", ver.to_string_lossy());
    }

    // Load achernar.toml configuration (falls back to defaults if absent).
    // Section 5.4 — the daemon now honors achernar.toml, not just env vars.
    // Set ACHERNAR_CONFIG to point at a specific file, otherwise ./achernar.toml.
    let config = config::AchernarConfig::load().unwrap_or_default();
    info!(
        "Config: daemon.port={}, log_level={}, focus.safety.max_position={}, mount.max_slew_rate={}",
        config.daemon.port,
        config.daemon.log_level,
        config.focus.safety.max_position,
        config.mount.max_slew_rate
    );

    // Create shared application state
    let app_state = AppState::new();
    cloud_client::spawn_if_configured(app_state.clone());

    // Build API routes
    let api_v1 = Router::new()
        .route("/status", get(api::get_status))
        .route("/drivers", get(api::get_driver_capabilities))
        .route("/connect", post(api::connect_equipment))
        .route("/disconnect", post(api::disconnect_equipment))
        .nest("/camera", camera_routes())
        .nest("/mount", mount_routes())
        .nest("/focuser", focuser_routes());

    let app = Router::new()
        .route("/health", get(health_check))
        .nest("/api/v1", api_v1)
        .route("/ws/telemetry", get(telemetry_ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(app_state.clone());

    // Bind to port
    let port = std::env::var("ACHERNAR_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(config.daemon.port);

    let host = std::env::var("ACHERNAR_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let ip: IpAddr = host.parse()?;
    let addr = SocketAddr::from((ip, port));
    info!("Daemon listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_check() -> &'static str {
    "OK"
}

fn camera_routes() -> Router<AppState> {
    Router::new()
        .route("/capture", post(api::camera::capture))
        .route("/abort", post(api::camera::abort))
        .route(
            "/parameters",
            get(api::camera::get_parameters).post(api::camera::set_parameter),
        )
        .route("/formats", get(api::camera::get_formats))
        .route("/stream/start", post(api::camera::start_stream))
        .route("/stream/mjpeg", get(api::camera::stream_mjpeg))
        .route("/stream", get(camera_stream_ws))
}

fn mount_routes() -> Router<AppState> {
    Router::new()
        .route("/slew", post(api::mount::slew))
        .route("/slew-target", post(api::mount::slew_target))
        .route("/sync", post(api::mount::sync))
        .route("/park", post(api::mount::park))
        .route("/unpark", post(api::mount::unpark))
        .route("/abort", post(api::mount::abort))
        .route("/tracking", post(api::mount::set_tracking))
}

fn focuser_routes() -> Router<AppState> {
    Router::new()
        .route("/move", post(api::focuser::move_relative))
        .route("/moveto", post(api::focuser::move_absolute))
        .route("/position", get(api::focuser::get_position))
        .route("/abort", post(api::focuser::abort))
}

/// WebSocket handler for telemetry — now sends real hardware state
async fn telemetry_ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| handle_telemetry_socket(socket, state))
}

async fn handle_telemetry_socket(mut socket: WebSocket, state: AppState) {
    info!("Telemetry stream started");
    let mut ticker = interval(Duration::from_millis(1000));

    loop {
        ticker.tick().await;

        let telemetry_str = crate::telemetry::telemetry_event(&state).to_string();

        if let Err(e) = socket.send(Message::Text(telemetry_str)).await {
            info!("Telemetry socket closed: {}", e);
            break;
        }
    }
}

/// WebSocket handler for live view streaming — sends real frame data
async fn camera_stream_ws(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| handle_stream_socket(socket, state))
}

async fn handle_stream_socket(mut socket: WebSocket, state: AppState) {
    info!("Live view stream started");
    let mut ticker = interval(Duration::from_millis(100)); // ~10 FPS

    loop {
        ticker.tick().await;

        // Get latest frame
        let frame_data = {
            let frame = state.latest_frame.lock().unwrap();
            frame.clone()
        };

        let frame_bytes = match frame_data {
            // JPEG-encode the frame so the browser can render it directly as an
            // image/jpeg blob, regardless of the camera's native pixel format.
            Some(f) => match api::camera::encode_frame_jpeg(&f) {
                Some(jpeg) => {
                    // Packet: [width(4)][height(4)][timestamp(8)][format(4)=JPEG][jpeg...]
                    let mut packet = Vec::with_capacity(20 + jpeg.len());
                    packet.extend_from_slice(&(f.width as u32).to_le_bytes());
                    packet.extend_from_slice(&(f.height as u32).to_le_bytes());
                    let ts = (f.unix_timestamp * 1000.0) as u64;
                    packet.extend_from_slice(&ts.to_le_bytes());
                    // Payload is always JPEG now (OlsStreamType::Mjpeg == 1).
                    packet.extend_from_slice(&1u32.to_le_bytes());
                    packet.extend_from_slice(&jpeg);
                    packet
                }
                // Format we don't preview-encode yet: signal "waiting".
                None => vec![0u8; 20],
            },
            None => {
                // No frame available yet, send empty packet
                vec![0u8; 20]
            }
        };

        match socket.send(Message::Binary(frame_bytes)).await {
            Ok(_) => {}
            Err(e) => {
                info!("Stream connection closed: {}", e);
                break;
            }
        }
    }

    info!("Live view stream ended");
}

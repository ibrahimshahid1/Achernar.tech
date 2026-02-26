use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::State,
    response::Response,
    routing::{get, post},
    Router,
};
use std::net::SocketAddr;
use tokio::time::{interval, Duration};
use tower_http::cors::CorsLayer;
use tracing::{info, Level};

mod hardware;
mod processing;
mod ai;
mod api;
mod state;

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

    // Create shared application state
    let app_state = AppState::new();

    // Build API routes
    let api_v1 = Router::new()
        .route("/status", get(api::get_status))
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
        .unwrap_or(3030);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
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
async fn telemetry_ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> Response {
    ws.on_upgrade(move |socket| handle_telemetry_socket(socket, state))
}

async fn handle_telemetry_socket(mut socket: WebSocket, state: AppState) {
    info!("Telemetry stream started");
    let mut ticker = interval(Duration::from_millis(1000));

    loop {
        ticker.tick().await;

        // Build telemetry from real state
        let camera_status = {
            let name = state.camera_name.lock().unwrap();
            let streaming = state.streaming.lock().unwrap();
            let frame = state.latest_frame.lock().unwrap();
            let frame_info = frame.as_ref().map(|f| {
                serde_json::json!({
                    "frame_counter": f.frame_counter,
                    "width": f.width,
                    "height": f.height,
                    "format": format!("{:?}", f.format),
                })
            });
            serde_json::json!({
                "connected": name.is_some(),
                "name": name.as_deref().unwrap_or("none"),
                "streaming": *streaming,
                "latest_frame": frame_info,
            })
        };

        let telemetry = serde_json::json!({
            "type": "TELEMETRY_UPDATE",
            "timestamp": chrono::Utc::now().timestamp(),
            "payload": {
                "camera": camera_status,
                "mount": {
                    "ra": 0.0,
                    "dec": 0.0,
                    "pier_side": "unknown",
                    "connected": false,
                },
                "guiding": {
                    "rms_total": 0.0,
                    "dx": 0.0,
                    "dy": 0.0,
                },
                "ai": {
                    "focus_hfr": 0.0,
                    "cloud_prob": 0.0,
                    "recommended_exposure": 0.0,
                },
            }
        });

        let telemetry_str = serde_json::to_string(&telemetry).unwrap_or_default();

        if let Err(e) = socket.send(Message::Text(telemetry_str)).await {
            info!("Telemetry socket closed: {}", e);
            break;
        }
    }
}

/// WebSocket handler for live view streaming — sends real frame data
async fn camera_stream_ws(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> Response {
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
            Some(f) => {
                // Build frame packet: [width(4)][height(4)][timestamp(8)][format(4)][data...]
                let mut packet = Vec::with_capacity(20 + f.data.len());
                packet.extend_from_slice(&(f.width as u32).to_le_bytes());
                packet.extend_from_slice(&(f.height as u32).to_le_bytes());
                let ts = (f.unix_timestamp * 1000.0) as u64;
                packet.extend_from_slice(&ts.to_le_bytes());
                packet.extend_from_slice(&(f.format as u32).to_le_bytes());
                packet.extend_from_slice(&f.data);
                packet
            }
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

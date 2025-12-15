use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::Response,
    routing::{get, post},
    Router,
};
use std::net::SocketAddr;
use tokio::time::{interval, Duration};
use tower_http::cors::CorsLayer;
use tracing::{info, Level};
use tracing_subscriber;

mod hardware;
mod processing;
mod ai;
mod api;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();

    info!("Starting Achernar Daemon v{}", env!("CARGO_PKG_VERSION"));

    // Build our application with routes
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/status", get(api::get_status))
        .route("/connect", post(api::connect_equipment))
        .route("/disconnect", post(api::disconnect_equipment))
        .nest("/camera", camera_routes())
        .nest("/mount", mount_routes())
        .nest("/focuser", focuser_routes())
        .layer(CorsLayer::permissive()); // Allow CORS for local access

    // Bind to localhost with dynamic port or default 3030
    let port = std::env::var("ACHERNAR_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3030);
    
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    
    info!("Daemon listening on http://{}", addr);
    
    // Start the server
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_check() -> &'static str {
    "OK"
}

fn camera_routes() -> Router {
    Router::new()
        .route("/capture", post(api::camera::capture))
        .route("/abort", post(api::camera::abort))
        // WebSocket live view stream
        .route("/stream", get(camera_stream_ws))
        // MJPEG stream (alternative)
        .route("/stream/mjpeg", get(api::camera::stream_mjpeg))
}

fn mount_routes() -> Router {
    Router::new()
        .route("/slew", post(api::mount::slew))
        .route("/slew-target", post(api::mount::slew_target))
        .route("/sync", post(api::mount::sync))
        .route("/park", post(api::mount::park))
        .route("/unpark", post(api::mount::unpark))
        .route("/abort", post(api::mount::abort))
        .route("/tracking", post(api::mount::set_tracking))
}

fn focuser_routes() -> Router {
    Router::new()
        .route("/move", post(api::focuser::move_relative))
        .route("/moveto", post(api::focuser::move_absolute))
        .route("/position", get(api::focuser::get_position))
        .route("/abort", post(api::focuser::abort))
}

/**
 * WebSocket handler for live view streaming
 * Provides low-latency binary frame delivery for focusing and AI inference
 */
async fn camera_stream_ws(ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(handle_stream_socket)
}

async fn handle_stream_socket(mut socket: WebSocket) {
    info!("Live view stream started");
    
    // Stream at ~10 FPS (adjust based on actual camera capabilities)
    let mut ticker = interval(Duration::from_millis(100));
    
    loop {
        ticker.tick().await;
        
        // TODO: Get actual frame from camera
        // For now, send dummy frame data
        let frame = generate_dummy_frame();
        
        match socket.send(Message::Binary(frame)).await {
            Ok(_) => {},
            Err(e) => {
                info!("Stream connection closed: {}", e);
                break;
            }
        }
    }
    
    info!("Live view stream ended");
}

/**
 * Generate dummy frame for testing
 * Frame format: [width(4)][height(4)][timestamp(8)][focusScore(4)][imageData(...)]
 */
fn generate_dummy_frame() -> Vec<u8> {
    let width: u32 = 640;
    let height: u32 = 480;
    let timestamp: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let focus_score: f32 = 0.0; // Will be computed by AI module
    
    let mut frame = Vec::new();
    
    // Header
    frame.extend_from_slice(&width.to_le_bytes());
    frame.extend_from_slice(&height.to_le_bytes());
    frame.extend_from_slice(&timestamp.to_le_bytes());
    frame.extend_from_slice(&focus_score.to_le_bytes());
    
    // Dummy JPEG image data (would be actual camera frame)
    // For now, just add some placeholder bytes
    let dummy_image = vec![0u8; 1024]; // Small JPEG would be here
    frame.extend_from_slice(&dummy_image);
    
    frame
}

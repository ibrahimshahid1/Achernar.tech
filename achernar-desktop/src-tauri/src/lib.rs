use serde::Serialize;
use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::Mutex,
    time::Duration,
};
use tauri::{Manager, State};

const DEFAULT_DAEMON_PORT: u16 = 3030;

#[derive(Debug, Serialize)]
pub struct DaemonStatus {
    running: bool,
    managed: bool,
    pid: Option<u32>,
    url: String,
    detail: String,
}

pub struct DaemonManager {
    child: Mutex<Option<Child>>,
    host: String,
    port: u16,
}

impl DaemonManager {
    fn new() -> Self {
        let host = std::env::var("ACHERNAR_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
        let port = std::env::var("ACHERNAR_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(DEFAULT_DAEMON_PORT);
        Self {
            child: Mutex::new(None),
            host,
            port,
        }
    }

    fn health_url(&self) -> String {
        format!("http://{}:{}/health", self.host, self.port)
    }

    fn socket_addr(&self) -> Result<SocketAddr, String> {
        format!("{}:{}", self.host, self.port)
            .parse()
            .map_err(|e| format!("Invalid daemon address: {e}"))
    }

    fn is_healthy(&self) -> bool {
        let Ok(addr) = self.socket_addr() else {
            return false;
        };
        let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(350)) else {
            return false;
        };
        let _ = stream.set_read_timeout(Some(Duration::from_millis(350)));
        let request = format!(
            "GET /health HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\n\r\n",
            self.host, self.port
        );
        if stream.write_all(request.as_bytes()).is_err() {
            return false;
        }
        let mut response = String::new();
        stream.read_to_string(&mut response).is_ok() && response.contains("200 OK")
    }

    fn daemon_bin() -> PathBuf {
        if let Ok(path) = std::env::var("ACHERNAR_DAEMON_BIN") {
            return PathBuf::from(path);
        }

        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .map(PathBuf::from)
            .unwrap_or_else(|| manifest_dir.clone());
        let bin_name = if cfg!(windows) {
            "achernar-daemon.exe"
        } else {
            "achernar-daemon"
        };
        let debug_bin = repo_root
            .join("achernar-daemon")
            .join("target")
            .join("debug")
            .join(bin_name);
        if debug_bin.exists() {
            return debug_bin;
        }
        repo_root
            .join("achernar-daemon")
            .join("target")
            .join("release")
            .join(bin_name)
    }

    fn start(&self) -> Result<DaemonStatus, String> {
        if self.is_healthy() {
            return Ok(DaemonStatus {
                running: true,
                managed: false,
                pid: None,
                url: self.health_url(),
                detail: "Using existing local daemon".to_string(),
            });
        }

        let mut child_guard = self
            .child
            .lock()
            .map_err(|_| "Daemon lock poisoned".to_string())?;
        if child_guard.is_some() {
            return Ok(DaemonStatus {
                running: self.is_healthy(),
                managed: true,
                pid: child_guard.as_ref().map(Child::id),
                url: self.health_url(),
                detail: "Daemon process is already managed by Tauri".to_string(),
            });
        }

        let bin = Self::daemon_bin();
        if !bin.exists() {
            return Err(format!(
                "Daemon binary not found at {}. Build achernar-daemon or set ACHERNAR_DAEMON_BIN.",
                bin.display()
            ));
        }

        let mut command = Command::new(bin);
        command
            .env("ACHERNAR_HOST", &self.host)
            .env("ACHERNAR_PORT", self.port.to_string())
            .env(
                "ACHERNAR_RELAY_DEV_TOKEN",
                std::env::var("ACHERNAR_RELAY_DEV_TOKEN")
                    .unwrap_or_else(|_| "achernar-dev-token".to_string()),
            )
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        for key in [
            "ACHERNAR_CLOUD_HTTP",
            "ACHERNAR_CLOUD_WS",
            "ACHERNAR_DAEMON_ID",
            "ACHERNAR_HARDWARE_ID",
            "ACHERNAR_CONFIG",
        ] {
            if let Ok(value) = std::env::var(key) {
                command.env(key, value);
            }
        }

        let child = command
            .spawn()
            .map_err(|e| format!("Failed to launch daemon: {e}"))?;
        let pid = child.id();
        *child_guard = Some(child);
        drop(child_guard);

        for _ in 0..20 {
            if self.is_healthy() {
                return Ok(DaemonStatus {
                    running: true,
                    managed: true,
                    pid: Some(pid),
                    url: self.health_url(),
                    detail: "Daemon launched by Tauri".to_string(),
                });
            }
            std::thread::sleep(Duration::from_millis(150));
        }

        Ok(DaemonStatus {
            running: false,
            managed: true,
            pid: Some(pid),
            url: self.health_url(),
            detail: "Daemon process launched, but health check has not passed yet".to_string(),
        })
    }

    fn stop(&self) -> Result<DaemonStatus, String> {
        let mut child_guard = self
            .child
            .lock()
            .map_err(|_| "Daemon lock poisoned".to_string())?;
        if let Some(mut child) = child_guard.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        Ok(DaemonStatus {
            running: self.is_healthy(),
            managed: false,
            pid: None,
            url: self.health_url(),
            detail: "Managed daemon stopped".to_string(),
        })
    }

    fn status(&self) -> DaemonStatus {
        let pid = self
            .child
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(Child::id));
        DaemonStatus {
            running: self.is_healthy(),
            managed: pid.is_some(),
            pid,
            url: self.health_url(),
            detail: if pid.is_some() {
                "Managed daemon process".to_string()
            } else {
                "No managed daemon process".to_string()
            },
        }
    }
}

impl Drop for DaemonManager {
    fn drop(&mut self) {
        if let Ok(mut child_guard) = self.child.lock() {
            if let Some(mut child) = child_guard.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
fn daemon_status(manager: State<'_, DaemonManager>) -> DaemonStatus {
    manager.status()
}

#[tauri::command]
fn restart_daemon(manager: State<'_, DaemonManager>) -> Result<DaemonStatus, String> {
    let _ = manager.stop()?;
    manager.start()
}

#[tauri::command]
fn stop_daemon(manager: State<'_, DaemonManager>) -> Result<DaemonStatus, String> {
    manager.stop()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(DaemonManager::new())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            greet,
            daemon_status,
            restart_daemon,
            stop_daemon
        ])
        .setup(|app| {
            let manager = app.state::<DaemonManager>();
            if let Err(err) = manager.start() {
                eprintln!("Achernar daemon sidecar was not started: {err}");
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

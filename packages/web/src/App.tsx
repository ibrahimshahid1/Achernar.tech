import { useState } from "react";
import { useTelemetrySocket } from "./hooks/useTelemetrySocket";
import { useDaemonStatus } from "./hooks/useDaemonStatus";
import { useTelemetryStore } from "./stores/telemetryStore";
import { daemonApi } from "./api/daemon";
import type { CaptureParams } from "./types/achernar";
import "./App.css";

/**
 * Achernar Web Application
 * PDF-Compliant Architecture (Sections 5.1-5.3, 16.2-16.3)
 * 
 * Architecture:
 * - Daemon API: Local edge control (PDF Section 5.1.1)
 * - Cloud API: Django backend (PDF Section 16.3)
 * - Telemetry: WebSocket real-time updates (PDF Section 5.1.2)
 * - State: Zustand for telemetry + React Query for server state (PDF Section 5.3)
 */
function App() {
  // Real-time telemetry from WebSocket (PDF Section 5.1.2)
  useTelemetrySocket();
  const telemetryConnected = useTelemetryStore((s) => s.connected);
  const equipment = useTelemetryStore((s) => s.equipment);
  const guiding = useTelemetryStore((s) => s.guiding);

  // Daemon status polling via React Query
  const { data: daemonStatus, isLoading, error } = useDaemonStatus();

  // Capture form state
  const [captureParams, setCaptureParams] = useState<CaptureParams>({
    exposure: 30,
    gain: 100,
    binning: 1,
    count: 1,
  });
  const [capturing, setCapturing] = useState(false);

  const handleCapture = async () => {
    setCapturing(true);
    try {
      const result = await daemonApi.captureImage(captureParams);
      console.log("Capture started:", result);
    } catch (err) {
      console.error("Capture failed:", err);
    } finally {
      setCapturing(false);
    }
  };

  const handleSlew = async () => {
    try {
      await daemonApi.slewMount(14.5, -32.0); // Example: M13
      console.log("Mount slewing to M13");
    } catch (err) {
      console.error("Slew failed:", err);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🔭 Achernar</h1>
        <p className="tagline">Unified Astronomy Imaging & Control System</p>
        <div className="connection-status">
          <span className={telemetryConnected ? "status-connected" : "status-disconnected"}>
            {telemetryConnected ? "● Telemetry Connected" : "○ Telemetry Disconnected"}
          </span>
          <span className="separator">|</span>
          <span className={!error ? "status-connected" : "status-disconnected"}>
            {!error ? "● Daemon OK" : "○ Daemon Offline"}
          </span>
        </div>
      </header>

      <main className="app-main">
        {error && (
          <div className="error-banner">
            <strong>⚠ Daemon Connection Failed</strong>
            <p>Make sure the Rust daemon is running:</p>
            <code>cd achernar-daemon && cargo run</code>
          </div>
        )}

        {isLoading && (
          <div className="loading">
            <div className="spinner"></div>
            <p>Connecting to daemon...</p>
          </div>
        )}

        {daemonStatus && (
          <div className="dashboard">
            {/* Real-Time Telemetry Section (PDF Section 5.1.2) */}
            <section className="telemetry-section">
              <h2>📡 Real-Time Telemetry</h2>
              <div className="telemetry-grid">
                <div className="telemetry-card">
                  <h3>🔭 Mount</h3>
                  {equipment?.mount && (
                    <ul>
                      <li>RA: {equipment.mount.ra.toFixed(4)}°</li>
                      <li>Dec: {equipment.mount.dec.toFixed(4)}°</li>
                      <li>Tracking: {equipment.mount.tracking ? "On" : "Off"}</li>
                    </ul>
                  )}
                </div>

                <div className="telemetry-card">
                  <h3>📷 Camera</h3>
                  {equipment?.camera && (
                    <ul>
                      <li>Temp: {equipment.camera.temperature.toFixed(1)}°C</li>
                      <li>Status: {equipment.camera.status}</li>
                      {equipment.camera.progress !== undefined && (
                        <li>Progress: {(equipment.camera.progress * 100).toFixed(0)}%</li>
                      )}
                    </ul>
                  )}
                </div>

                {guiding && guiding.rms_total !== undefined && (
                  <div className="telemetry-card">
                    <h3>🎯 Guiding</h3>
                    <ul>
                      <li>RMS: {guiding.rms_total.toFixed(2)}"</li>
                      <li>dX: {guiding.dx?.toFixed(2)}"</li>
                      <li>dY: {guiding.dy?.toFixed(2)}"</li>
                    </ul>
                  </div>
                )}

                {equipment?.focuser && (
                  <div className="telemetry-card">
                    <h3>🎯 Focuser</h3>
                    <ul>
                      <li>Position: {equipment.focuser.position}</li>
                      <li>Status: {equipment.focuser.moving ? "Moving" : "Idle"}</li>
                    </ul>
                  </div>
                )}
              </div>
            </section>

            {/* Camera Control Section */}
            <section className="control-section">
              <h2>📷 Camera Control</h2>
              <div className="camera-control">
                <div className="control-grid">
                  <label>
                    Exposure (s)
                    <input
                      type="number"
                      value={captureParams.exposure}
                      onChange={(e) =>
                        setCaptureParams({ ...captureParams, exposure: Number(e.target.value) })
                      }
                    />
                  </label>

                  <label>
                    Gain
                    <input
                      type="number"
                      value={captureParams.gain}
                      onChange={(e) =>
                        setCaptureParams({ ...captureParams, gain: Number(e.target.value) })
                      }
                    />
                  </label>

                  <label>
                    Binning
                    <select
                      value={captureParams.binning}
                      onChange={(e) =>
                        setCaptureParams({ ...captureParams, binning: Number(e.target.value) })
                      }
                    >
                      <option value={1}>1x1</option>
                      <option value={2}>2x2</option>
                      <option value={4}>4x4</option>
                    </select>
                  </label>

                  <label>
                    Count
                    <input
                      type="number"
                      min={1}
                      value={captureParams.count}
                      onChange={(e) =>
                        setCaptureParams({ ...captureParams, count: Number(e.target.value) })
                      }
                    />
                  </label>
                </div>

                <div className="button-group">
                  <button onClick={handleCapture} disabled={capturing}>
                    {capturing ? "Capturing..." : "Start Capture"}
                  </button>
                  <button onClick={handleSlew}>Slew to M13</button>
                </div>
              </div>
            </section>

            {/* Architecture Info */}
            <section className="info-section">
              <h2>ℹ️ System Architecture (PDF-Compliant)</h2>
              <div className="info-card">
                <h3>✅ Active Features</h3>
                <ul>
                  <li><strong>Daemon API:</strong> Local edge control (PDF Section 5.1.1)</li>
                  <li><strong>WebSocket Telemetry:</strong> Real-time @ 100ms (PDF Section 5.1.2)</li>
                  <li><strong>Zustand Store:</strong> High-frequency UI state (PDF Section 5.3.2)</li>
                  <li><strong>React Query:</strong> Server state management</li>
                  <li><strong>Type Safety:</strong> Full TypeScript (PDF Section 5.2.2)</li>
                </ul>

                <h3>📋 Next Steps</h3>
                <ul>
                  <li>Start Rust daemon: <code>cd achernar-daemon && cargo run</code></li>
                  <li>Start Django backend: <code>docker-compose up -d</code></li>
                  <li>Watch telemetry update in real-time</li>
                </ul>

                <h3>🔌 Endpoints</h3>
                <ul>
                  <li><strong>Daemon:</strong> <code>http://localhost:3030/api/v1</code></li>
                  <li><strong>WebSocket:</strong> <code>ws://localhost:3030/ws/telemetry</code></li>
                  <li><strong>Cloud:</strong> <code>http://localhost:8000/api/v1</code></li>
                </ul>
              </div>
            </section>
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>Achernar v0.1.0 | PDF-Compliant Architecture | Mode: Cloud Web App</p>
      </footer>
    </div>
  );
}

export default App;

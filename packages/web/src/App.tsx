import { useState, useEffect } from 'react';
import { TelescopeProvider, CloudClient, CameraControl } from '@achernar/ui';
import type { EquipmentStatus } from '@achernar/ui';
import './App.css';

/**
 * Achernar Web Application
 * Cloud-based astronomy imaging platform
 */
function App() {
  const [client] = useState(() => new CloudClient('/api/v1'));
  const [status, setStatus] = useState<EquipmentStatus | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Connect to equipment on mount
  useEffect(() => {
    const connectEquipment = async () => {
      try {
        await client.connect();
        const currentStatus = await client.getStatus();
        setStatus(currentStatus);
        setConnected(true);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to connect');
        console.error('Connection error:', err);
      }
    };

    connectEquipment();

    return () => {
      client.disconnect().catch(console.error);
    };
  }, [client]);

  return (
    <TelescopeProvider client={client}>
      <div className="app">
        <header className="app-header">
          <h1>🔭 Achernar</h1>
          <p className="tagline">Astronomy Imaging & Control Platform</p>
          <div className="connection-status">
            {connected ? (
              <span className="status-connected">● Connected</span>
            ) : (
              <span className="status-disconnected">○ Disconnected</span>
            )}
          </div>
        </header>

        <main className="app-main">
          {error && (
            <div className="error-banner">
              <strong>Error:</strong> {error}
              <p className="error-hint">
                Make sure the Django backend is running at http://localhost:8000
              </p>
            </div>
          )}

          {connected && status && (
            <div className="dashboard">
              <section className="equipment-section">
                <h2>Equipment Status</h2>
                <div className="status-grid">
                  {status.camera && (
                    <div className="status-card">
                      <h3>📷 Camera</h3>
                      <ul>
                        <li>Name: {status.camera.name}</li>
                        <li>Temperature: {status.camera.temperature}°C</li>
                        <li>Cooling: {status.camera.cooling ? 'On' : 'Off'}</li>
                        <li>Streaming: {status.camera.streaming ? 'Active' : 'Inactive'}</li>
                      </ul>
                    </div>
                  )}

                  {status.mount && (
                    <div className="status-card">
                      <h3>🔭 Mount</h3>
                      <ul>
                        <li>Name: {status.mount.name}</li>
                        <li>RA: {status.mount.ra.toFixed(4)}°</li>
                        <li>Dec: {status.mount.dec.toFixed(4)}°</li>
                        <li>Tracking: {status.mount.tracking ? 'On' : 'Off'}</li>
                      </ul>
                    </div>
                  )}

                  {status.focuser && (
                    <div className="status-card">
                      <h3>🎯 Focuser</h3>
                      <ul>
                        <li>Name: {status.focuser.name}</li>
                        <li>Position: {status.focuser.position}</li>
                        <li>Status: {status.focuser.moving ? 'Moving' : 'Idle'}</li>
                      </ul>
                    </div>
                  )}
                </div>
              </section>

              <section className="control-section">
                <h2>Camera Control</h2>
                <CameraControl />
              </section>

              <section className="info-section">
                <h2>ℹ️ Getting Started</h2>
                <div className="info-card">
                  <h3>Test the Live Stream</h3>
                  <p>The Camera Control panel above connects to the shared UI package!</p>
                  <ul>
                    <li>Adjust exposure, gain, and binning settings</li>
                    <li>Click "Capture" to take a high-resolution image</li>
                    <li>Images are saved to the Django backend</li>
                  </ul>

                  <h3>Architecture</h3>
                  <p>This web app uses:</p>
                  <ul>
                    <li><code>@achernar/ui</code> - Shared React components</li>
                    <li><code>CloudClient</code> - Django REST API communication</li>
                    <li><code>TelescopeProvider</code> - Dependency injection</li>
                  </ul>
                </div>
              </section>
            </div>
          )}

          {!connected && !error && (
            <div className="loading">
              <div className="spinner"></div>
              <p>Connecting to equipment...</p>
            </div>
          )}
        </main>

        <footer className="app-footer">
          <p>Achernar v0.1.0 | Mode: Cloud Web App</p>
        </footer>
      </div>
    </TelescopeProvider>
  );
}

export default App;

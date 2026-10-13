// Environment configuration aligned with PDF Section 16.2
export const env = {
    // Daemon API (local edge, Section 5.1.1)
    daemonApiBase: import.meta.env.VITE_DAEMON_URL || 'http://localhost:3030/api/v1',
    daemonWsUrl: import.meta.env.VITE_DAEMON_WS || 'ws://localhost:3030/ws/telemetry',

    // Cloud API (Django apps, Section 16.3) - NO /v1 for Django
    cloudApiBase: import.meta.env.VITE_CLOUD_API || 'http://localhost:8000/api',
    cloudWsUrl: import.meta.env.VITE_CLOUD_WS || (
        typeof window !== 'undefined'
            ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`
            : 'ws://localhost:8000/ws'
    ),

    // Browser escape hatch for rigs where the web client is on the same LAN as
    // the daemon. Default browser mode stays on the Django relay path.
    enableBrowserDirectDaemon: ["1", "true", "yes"].includes(
        String(import.meta.env.VITE_ENABLE_BROWSER_DIRECT_DAEMON || "").toLowerCase()
    ),

    // Optional image-only socket for a remote rig. This never changes the
    // command transport; same-machine browsers select the local socket automatically.
    cameraPreviewWs: String(import.meta.env.VITE_CAMERA_PREVIEW_WS || ""),
};

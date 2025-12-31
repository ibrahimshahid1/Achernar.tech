// Environment configuration aligned with PDF Section 16.2
export const env = {
    // Daemon API (local edge, Section 5.1.1)
    daemonApiBase: import.meta.env.VITE_DAEMON_URL || 'http://localhost:3030/api/v1',
    daemonWsUrl: import.meta.env.VITE_DAEMON_WS || 'ws://localhost:3030/ws/telemetry',

    // Cloud API (Django apps, Section 16.3)
    cloudApiBase: import.meta.env.VITE_CLOUD_API || '/api/v1',
    cloudWsUrl: import.meta.env.VITE_CLOUD_WS || (
        typeof window !== 'undefined'
            ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`
            : 'ws://localhost:8000/ws'
    ),
};

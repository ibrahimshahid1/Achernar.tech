import { useEffect, useMemo } from "react";
import { env } from "../env";
import { getHardwareTransport } from "../lib/runtime";
import { useTelemetryStore } from "../stores/telemetryStore";
import type { TelemetryUpdateEvent } from "../types/achernar";

/**
 * WebSocket Telemetry Hook
 * Connects to the daemon WebSocket for real-time updates.
 * Per PDF Section 5.1.2: WebSocket Events (Real-Time Telemetry)
 *
 * Transport-aware: a direct daemon WebSocket is only reachable when running
 * inside the desktop (Tauri) runtime. In the browser ("django" transport) the
 * daemon is not directly reachable, so we skip the socket and rely on the
 * Django relay instead of throwing connection errors.
 */
export function useTelemetrySocket() {
    const setConnected = useTelemetryStore((s) => s.setConnected);
    const applyTelemetry = useTelemetryStore((s) => s.applyTelemetry);
    const hardwareTransport = getHardwareTransport();

    const wsUrl = useMemo(() => env.daemonWsUrl, []);

    useEffect(() => {
        if (hardwareTransport === "django") {
            setConnected(false);
            return;
        }

        const ws = new WebSocket(wsUrl);

        ws.onopen = () => setConnected(true);
        ws.onclose = () => setConnected(false);
        ws.onerror = () => setConnected(false);

        ws.onmessage = (msg) => {
            try {
                const data = JSON.parse(msg.data) as TelemetryUpdateEvent;
                if (data?.type === "TELEMETRY_UPDATE") {
                    applyTelemetry(data);
                }
            } catch {
                // Ignore malformed telemetry
            }
        };

        return () => ws.close();
    }, [wsUrl, setConnected, applyTelemetry, hardwareTransport]);
}

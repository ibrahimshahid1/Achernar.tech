import { useEffect, useMemo } from "react";
import { env } from "../env";
import { useTelemetryStore } from "../stores/telemetryStore";
import type { TelemetryUpdateEvent } from "../types/achernar";

/**
 * WebSocket Telemetry Hook
 * Connects to daemon WebSocket for real-time updates
 * Per PDF Section 5.1.2: WebSocket Events (Real-Time Telemetry)
 */
export function useTelemetrySocket() {
    const setConnected = useTelemetryStore((s) => s.setConnected);
    const applyTelemetry = useTelemetryStore((s) => s.applyTelemetry);

    const wsUrl = useMemo(() => env.daemonWsUrl, []);

    useEffect(() => {
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
    }, [wsUrl, setConnected, applyTelemetry]);
}

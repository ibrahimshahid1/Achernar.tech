import { useEffect, useRef, useCallback } from "react";
import { env } from "../env";
import { useTelemetryStore } from "../stores/telemetryStore";

/**
 * All WebSocket event types from the daemon — Section 7.5.5.
 */
export type DaemonWsEventType =
    | "TELEMETRY_UPDATE"
    | "FOCUS_RUN_STARTED"
    | "FOCUS_RUN_STATE_CHANGED"
    | "FOCUS_SAMPLE_CAPTURED"
    | "FOCUS_PREDICTION_MADE"
    | "FOCUS_ACTION_PROPOSED"
    | "FOCUS_ACTION_VALIDATED"
    | "FOCUS_ACTION_EXECUTED"
    | "FOCUS_ACTION_REJECTED"
    | "FOCUS_FALLBACK_TRIGGERED"
    | "FOCUS_RUN_CONVERGED"
    | "FOCUS_RUN_FAILED"
    | "FOCUS_DRIFT_ALERT"
    | "VISION_UPDATE";

export interface DaemonWsMessage {
    type: DaemonWsEventType;
    [key: string]: unknown;
}

type EventHandler = (data: DaemonWsMessage) => void;

/**
 * Full WebSocket connection to daemon at ws://localhost:3030/ws/telemetry.
 * Dispatches all FOCUS_* event types defined in Section 7.5.5.
 */
export function useDaemonWS(handlers?: Partial<Record<DaemonWsEventType, EventHandler>>) {
    const wsRef = useRef<WebSocket | null>(null);
    const setConnected = useTelemetryStore((s) => s.setConnected);
    const applyTelemetry = useTelemetryStore((s) => s.applyTelemetry);

    const handleMessage = useCallback(
        (event: MessageEvent) => {
            try {
                const data = JSON.parse(event.data) as DaemonWsMessage;

                // Always apply telemetry updates to the store
                if (data.type === "TELEMETRY_UPDATE") {
                    applyTelemetry(data as any);
                }

                // Dispatch to specific handler if registered
                const handler = handlers?.[data.type];
                if (handler) {
                    handler(data);
                }
            } catch {
                // Ignore malformed messages
            }
        },
        [handlers, applyTelemetry]
    );

    useEffect(() => {
        const ws = new WebSocket(env.daemonWsUrl);
        wsRef.current = ws;

        ws.onopen = () => setConnected(true);
        ws.onclose = () => setConnected(false);
        ws.onerror = () => setConnected(false);
        ws.onmessage = handleMessage;

        return () => {
            ws.close();
            wsRef.current = null;
        };
    }, [env.daemonWsUrl, setConnected, handleMessage]);

    return wsRef;
}

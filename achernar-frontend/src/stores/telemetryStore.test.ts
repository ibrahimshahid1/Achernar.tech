import { afterEach, describe, expect, it } from "vitest";
import { useTelemetryStore } from "./telemetryStore";

afterEach(() => useTelemetryStore.getState().setConnected(false));

describe("telemetry disconnects", () => {
    it("clears last equipment, timestamp and analysis when the socket disconnects", () => {
        const state = useTelemetryStore.getState();
        state.setConnected(true);
        state.applyTelemetry({ type: "TELEMETRY_UPDATE", timestamp: 100, payload: { camera: { connected: true, name: "Canon", streaming: true }, ai: { star_count: 10 } } });
        expect(useTelemetryStore.getState().equipment?.camera?.name).toBe("Canon");
        state.setConnected(false);
        expect(useTelemetryStore.getState()).toMatchObject({ connected: false, equipment: {}, guiding: null, ai: null, lastTimestamp: undefined });
    });

    it("replaces connected camera state with absence instead of merging the last name", () => {
        const state = useTelemetryStore.getState();
        state.applyTelemetry({ type: "TELEMETRY_UPDATE", timestamp: 100, payload: { camera: { connected: true, name: "Canon" } } });
        state.applyTelemetry({ type: "TELEMETRY_UPDATE", timestamp: 101, payload: { camera: { connected: false, name: "Old Canon" } } });
        expect(useTelemetryStore.getState().equipment?.camera).toBeUndefined();
        expect(useTelemetryStore.getState().equipment?.connected).toBe(false);
    });
});

import { create } from "zustand";
import type { EquipmentState, TelemetryUpdateEvent } from "../types/achernar";

type AiMetrics = NonNullable<TelemetryUpdateEvent["payload"]["ai"]>;
type GuidingState = TelemetryUpdateEvent["payload"]["guiding"];

type TelemetryState = {
    connected: boolean;
    lastTimestamp?: number;
    equipment?: Partial<EquipmentState>;
    /** Null until a guider exists and reports. */
    guiding: GuidingState;
    /** Null when the latest frame was never analysed. */
    ai: AiMetrics | null;

    setConnected: (v: boolean) => void;
    applyTelemetry: (evt: TelemetryUpdateEvent) => void;
};

/**
 * Telemetry is a full snapshot, not a patch.
 *
 * The daemon rebuilds every field each second, so this replaces rather than
 * merges. The previous version fell back through
 * `evt.payload.mount?.ra ?? prev.equipment?.mount?.ra ?? 0`, which had two
 * failure modes now that the daemon reports null for unmeasured values: a null
 * reading would render the *last known* value as though it were current, and a
 * device that disconnected would keep displaying its final reading forever.
 * A stale number that looks live is worse than a visible gap.
 */
export const useTelemetryStore = create<TelemetryState>((set) => ({
    connected: false,
    equipment: {},
    guiding: null,
    ai: null,

    setConnected: (v) => set(v ? { connected: true } : {
        connected: false,
        lastTimestamp: undefined,
        equipment: {},
        guiding: null,
        ai: null,
    }),

    applyTelemetry: (evt) => {
        const { mount, camera, focuser } = evt.payload;

        set({
            lastTimestamp: evt.timestamp,
            equipment: {
                mount: mount?.connected === true
                    ? {
                          name: mount.name,
                          // Never assume connected — the daemon reports it.
                          connected: mount.connected ?? false,
                          ra: mount.ra ?? null,
                          dec: mount.dec ?? null,
                          tracking: mount.tracking ?? null,
                          parked: mount.parked ?? null,
                          pier_side: mount.pier_side ?? null,
                      }
                    : undefined,
                camera: camera?.connected === true
                    ? {
                          name: camera.name,
                          connected: camera.connected ?? false,
                          cooling: camera.cooling ?? false,
                          temperature: camera.temperature ?? null,
                          status: camera.status as NonNullable<EquipmentState["camera"]>["status"],
                          progress: camera.progress,
                          streaming: camera.streaming,
                      }
                    : undefined,
                focuser: focuser?.connected === true
                    ? {
                          name: focuser.name,
                          connected: focuser.connected ?? false,
                          position: focuser.position ?? null,
                          moving: focuser.moving ?? false,
                      }
                    : undefined,
                // The daemon is connected; whether any hardware is, is the
                // per-device `connected` flag above.
                connected: Boolean(mount?.connected || camera?.connected || focuser?.connected),
            },
            guiding: evt.payload.guiding ?? null,
            ai: evt.payload.ai ?? null,
        });
    },
}));

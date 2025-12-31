import { create } from "zustand";
import type { EquipmentState, TelemetryUpdateEvent } from "../types/achernar";

type TelemetryState = {
    connected: boolean;
    lastTimestamp?: number;
    equipment?: Partial<EquipmentState>;
    guiding?: { rms_total?: number; dx?: number; dy?: number };

    setConnected: (v: boolean) => void;
    applyTelemetry: (evt: TelemetryUpdateEvent) => void;
};

export const useTelemetryStore = create<TelemetryState>((set) => ({
    connected: false,
    equipment: {},
    guiding: {},

    setConnected: (v) => set({ connected: v }),

    applyTelemetry: (evt) =>
        set((prev) => ({
            lastTimestamp: evt.timestamp,
            equipment: {
                mount: {
                    ...prev.equipment?.mount,
                    ra: evt.payload.mount?.ra ?? prev.equipment?.mount?.ra ?? 0,
                    dec: evt.payload.mount?.dec ?? prev.equipment?.mount?.dec ?? 0,
                    connected: prev.equipment?.mount?.connected ?? true,
                    tracking: evt.payload.mount?.tracking ?? prev.equipment?.mount?.tracking ?? false,
                },
                camera: {
                    ...prev.equipment?.camera,
                    temperature: evt.payload.camera?.temperature ?? prev.equipment?.camera?.temperature ?? 0,
                    status: (evt.payload.camera?.status as any) ?? prev.equipment?.camera?.status ?? "idle",
                    progress: evt.payload.camera?.progress ?? prev.equipment?.camera?.progress,
                    connected: prev.equipment?.camera?.connected ?? true,
                    cooling: prev.equipment?.camera?.cooling ?? false,
                },
                focuser: {
                    ...prev.equipment?.focuser,
                    position: evt.payload.focuser?.position ?? prev.equipment?.focuser?.position ?? 0,
                    moving: evt.payload.focuser?.moving ?? prev.equipment?.focuser?.moving ?? false,
                    connected: prev.equipment?.focuser?.connected ?? true,
                },
                connected: true,
            },
            guiding: evt.payload.guiding ?? prev.guiding,
        })),
}));

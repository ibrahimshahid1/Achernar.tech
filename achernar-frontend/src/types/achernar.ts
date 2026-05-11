// Keep these aligned with PDF design doc section 5.2.2
// Types match Rust daemon API responses and Django backend models

export type EquipmentType = "mount" | "focuser" | "camera" | "filter-wheel" | "dome";

export interface ImagingSession {
    id: string;
    target: string;
    startTime: string; // ISO string
    frameCount: number;
    qualityScore: number; // aggregated AI score
}

// UI-friendly subset of EquipmentState from daemon
export interface EquipmentState {
    connected: boolean;
    mount?: {
        connected: boolean;
        tracking: boolean;
        ra: number;
        dec: number;
        pier_side?: string;
    };
    camera?: {
        connected: boolean;
        cooling: boolean;
        temperature: number;
        status: "idle" | "exposing" | "downloading";
        progress?: number; // 0..1
    };
    focuser?: {
        connected: boolean;
        position: number;
        moving: boolean;
    };
}

// Telemetry event per PDF websocket spec
export interface TelemetryUpdateEvent {
    type: "TELEMETRY_UPDATE";
    timestamp: number;
    payload: {
        mount?: { ra?: number; dec?: number; pier_side?: string; tracking?: boolean };
        camera?: { temperature?: number; status?: string; progress?: number };
        guiding?: { rms_total?: number; dx?: number; dy?: number };
        focuser?: { position?: number; moving?: boolean };
    };
}

export interface EquipmentCardModel {
    id: string;
    type: EquipmentType;
    name: string;
    site: string;
    status: "nominal" | "warning" | "offline";
    tags: string[];
    meta?: Record<string, string | number | boolean>;
}

export interface CaptureParams {
    exposure: number;
    gain: number;
    count?: number;
    filter?: string;
    binning?: number;
    offset?: number;
    temperature?: number;
}

export interface ImageData {
    id: string;
    width: number;
    height: number;
    metadata: {
        exposure: number;
        gain: number;
        temperature: number;
        timestamp: string;
    };
}

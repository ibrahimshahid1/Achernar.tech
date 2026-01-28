// Keep these aligned with PDF Section 5.2.2
// Frontend Interfaces (TypeScript)

export type EquipmentType = "mount" | "focuser" | "camera" | "filter-wheel" | "dome";

export interface ImagingSession {
    id: string;
    target: string;
    start_time: string; // ISO string in UI (avoid Date serialization issues)
    frame_count: number;
    quality_score: number; // aggregated AI score
}

// Minimal, UI-friendly subset of your EquipmentState (PDF Section 5.2.2)
export interface EquipmentState {
    mount: {
        connected: boolean;
        tracking: boolean;
        ra: number;
        dec: number;
    };
    camera: {
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

// Telemetry event per PDF Section 5.1.2 (WebSocket Events)
export interface TelemetryUpdateEvent {
    type: "TELEMETRY_UPDATE";
    timestamp: number;
    payload: {
        mount?: { ra?: number; dec?: number; pier_side?: string };
        camera?: { temperature?: number; status?: string; progress?: number };
        guiding?: { rms_total?: number; dx?: number; dy?: number };
        focuser?: { position?: number; temperature?: number };
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

// Capture params per PDF Section 4.3
export interface CaptureParams {
    exposure: number;
    gain?: number;
    offset?: number;
    binning?: number;
    filter?: string;
    count?: number; // sequence count
}

// Image data response
export interface ImageData {
    id: string;
    filename: string;
    filter: string;
    exposure: number;
    temperature: number;
    focus_hfr?: number;
    cloud_prob?: number;
    background_adu?: number;
    created_at: string;
}

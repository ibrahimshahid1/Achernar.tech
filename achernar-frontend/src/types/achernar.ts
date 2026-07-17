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
    daemon_id?: string;
    hardware_id?: string;
    last_heartbeat?: string;
    mount?: {
        name?: string;
        connected?: boolean;
        tracking: boolean;
        ra: number;
        dec: number;
        parked?: boolean;
        pier_side?: string;
    };
    camera?: {
        name?: string;
        connected?: boolean;
        cooling: boolean;
        temperature: number;
        status?: "idle" | "exposing" | "downloading";
        exposing?: boolean;
        streaming?: boolean;
        progress?: number; // 0..1
        gain?: number;
        exposure_s?: number;
        latest_frame?: {
            frame_counter?: number;
            width?: number;
            height?: number;
            format?: string;
        };
    };
    focuser?: {
        name?: string;
        connected?: boolean;
        position: number;
        moving: boolean;
        absolute?: boolean;
        max_range?: number;
        hfr?: number;
        temperature?: number;
    };
    filter_wheel?: {
        name?: string;
        connected?: boolean;
        filter?: string;
        slot?: number;
        moving?: boolean;
        offset?: number;
    };
}

export interface DaemonSummary {
    id: string;
    hardware_id: string;
    daemon_version?: string;
    site_name?: string;
    is_connected: boolean;
    last_heartbeat?: string;
}

export interface RelayCommandResult {
    type?: "COMMAND_RESULT";
    command_id?: string;
    daemon_id?: string;
    ok: boolean | null;
    pending?: boolean;
    payload?: Record<string, unknown>;
    error?: string;
}

// Telemetry event per PDF websocket spec
export interface TelemetryUpdateEvent {
    type: "TELEMETRY_UPDATE";
    timestamp: number;
    payload: {
        mount?: { name?: string; connected?: boolean; ra?: number; dec?: number; pier_side?: string; tracking?: boolean };
        camera?: { name?: string; connected?: boolean; temperature?: number; status?: string; progress?: number; streaming?: boolean };
        guiding?: { rms_total?: number; dx?: number; dy?: number };
        focuser?: { name?: string; connected?: boolean; position?: number; moving?: boolean };
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
    summary?: string;
    source?: string;
    metrics?: Array<{
        label: string;
        value: string | number | boolean;
        tone?: "good" | "warn" | "bad" | "muted";
    }>;
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

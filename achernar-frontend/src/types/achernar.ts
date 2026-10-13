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

// UI-friendly subset of EquipmentState from daemon.
//
// `null` means "the daemon did not measure this", and is distinct from any
// value the field could legitimately hold. RA 0h / Dec +0deg is a real
// coordinate, focuser step 0 is a real position, and 0 degC is a real
// temperature, so none of them can double as "unknown". Render null as an
// em dash, never as a number.
export interface EquipmentState {
    connected: boolean;
    daemon_connected?: boolean;
    telemetry_stale?: boolean;
    updated_at?: string;
    device_health?: Record<string, { connected: boolean; status?: string; reason?: string | null; last_checked_at?: string | null }>;
    daemon_id?: string;
    hardware_id?: string;
    last_heartbeat?: string;
    mount?: {
        name?: string;
        connected?: boolean;
        tracking: boolean | null;
        ra: number | null;
        dec: number | null;
        parked?: boolean | null;
        pier_side?: string | null;
    } | null;
    camera?: {
        name?: string;
        connected?: boolean;
        connection_id?: string | null;
        cooling: boolean;
        temperature: number | null;
        status?: "disconnected" | "connecting" | "idle" | "streaming" | "exposing" | "downloading" | "processing" | "disconnecting" | "error";
        last_error?: string | null;
        exposing?: boolean;
        streaming?: boolean;
        progress?: number; // 0..1
        gain?: number;
        exposure_s?: number;
        latest_frame?: {
            stream_generation?: number;
            frame_counter?: number;
            width?: number;
            height?: number;
            format?: string;
            timestamp?: number;
        };
    } | null;
    focuser?: {
        name?: string;
        connected?: boolean;
        position: number | null;
        moving: boolean;
        absolute?: boolean;
        max_range?: number | null;
        hfr?: number | null;
        temperature?: number | null;
    } | null;
    filter_wheel?: {
        name?: string;
        connected?: boolean;
        filter?: string;
        slot?: number;
        moving?: boolean;
        offset?: number;
    } | null;
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
        mount?: { name?: string; connected?: boolean; ra?: number | null; dec?: number | null; pier_side?: string | null; tracking?: boolean | null; parked?: boolean | null };
        camera?: { name?: string; connected?: boolean; temperature?: number | null; status?: string; progress?: number; streaming?: boolean; cooling?: boolean };
        // Null until a guider exists. There is no guider in the system today,
        // so a zeroed RMS block would be a measurement of nothing.
        guiding?: { rms_total?: number | null; dx?: number | null; dy?: number | null } | null;
        focuser?: { name?: string; connected?: boolean; position?: number | null; moving?: boolean };
        // Null means the frame was never analysed. `observation_valid: false`
        // means it was analysed and the result was unusable — different facts.
        ai?: {
            focus_hfr?: number | null;
            cloud_prob?: number | null;
            star_count?: number | null;
            star_count_used?: number | null;
            observation_valid?: boolean | null;
            recommended_exposure?: number | null;
        };
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

export type RetentionPolicy = "retain_on_camera" | "delete_after_verified_local_commit";

export interface RetentionAcknowledgement {
    policy: RetentionPolicy;
    policy_version: number;
}

/** The daemon's recorded decision; a local archive alone does not prove a card copy. */
export interface CaptureRetentionDecision {
    requested_policy: RetentionPolicy;
    effective_policy: RetentionPolicy;
    policy_version: number;
    camera_copy_state: "retained" | "not_applicable" | "unknown";
    camera_files_deleted: boolean;
    deletion_eligible: boolean;
    reason: string;
}

export interface CaptureParams {
    exposure: number;
    /**
     * Optional on purpose. The daemon treats `<= 0` as "leave the operator's
     * setting alone", so sending 0 for a body that has no numeric gain control
     * is not neutral — it overrides the ISO they set on the camera. Omit the
     * field instead.
     */
    gain?: number;
    count?: number;
    filter?: string;
    binning?: number;
    offset?: number;
    temperature?: number;
    /** Omission retains originals. Deletion remains unavailable until storage qualification. */
    retention_policy?: RetentionPolicy;
    retention_acknowledgement?: RetentionAcknowledgement;
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

// Keep these aligned with PDF design doc section 5.2.2
// Types match Rust daemon API responses and Django backend models

export type EquipmentType =
    | "mount"
    | "focuser"
    | "camera"
    | "filter-wheel"
    | "guider"
    | "weather"
    | "safety-monitor"
    | "dome";

// ── Equipment layer contract (daemon spec §2.8, §2.13, D8, D21) ───────────
//
// Every block below mirrors what the daemon publishes in `/status` and
// `TELEMETRY_UPDATE`. New keys are additive and nullable on the daemon side,
// so every one is optional here: an older daemon that does not send a key
// must read as "not reported", never as a plausible value.

/** How far a driver or device identity has been proven (D21). */
export type QualificationState = "planned" | "implemented_unqualified" | "qualified";

/** A device identity's qualification as the daemon serialises it. */
export interface Qualification {
    state: QualificationState;
    /** The rig acceptance record, only for `qualified`. */
    record_id?: string;
}

/** Worker link state (§2.8). `lost` reports `connected: false`. */
export type LinkState = "connecting" | "ready" | "degraded" | "lost" | "disconnected";

/** v6 §4.1 failure classes, set while a device is degraded or lost. */
export type FailureClass =
    | "disconnected"
    | "unsupported"
    | "timeout"
    | "outcome_unknown"
    | "driver"
    | "driver_panic"
    | "invalid";

/** One `device_health` entry. `status` keeps its two values (D8). */
export interface DeviceHealthEntry {
    connected: boolean;
    status?: string;
    connection_id?: string | null;
    probe_supported?: boolean | null;
    last_checked_at?: string | null;
    last_error?: string | null;
    state?: LinkState | null;
    failure_class?: FailureClass | null;
    /** Mirrors `last_error`. */
    reason?: string | null;
}

/** Keys every equipment-layer device block carries. */
export interface DeviceBlockCommon {
    name?: string;
    connected?: boolean;
    driver?: string | null;
    connection_id?: string | null;
    /** Stable profile key (§2.6). Published by newer daemons only. */
    identity_key?: string | null;
    qualification?: Qualification | null;
    link_state?: LinkState | null;
    failure_class?: FailureClass | null;
    reason?: string | null;
    observed_at?: string | null;
    age_ms?: number | null;
}

export type EqFrame = "j2000" | "jnow_mean" | "jnow_apparent" | "unknown";
export type DriveRate = "sidereal" | "lunar" | "solar" | "king";
export type AlignmentMode = "alt_az" | "polar" | "german_polar";
export type PierSide = "east" | "west";
export type GuiderAppState =
    | "stopped"
    | "selected"
    | "calibrating"
    | "guiding"
    | "lost_lock"
    | "paused"
    | "looping"
    | "unknown";
export type ObsState = "fresh" | "stale" | "unsupported" | "not_set" | "error";
export type SafetyLevel = "safe" | "warning" | "unsafe";
/** The combined safety state of every monitor (§3.10, §2.8 fail-safe rule). */
export type SafetyGate = "none" | "safe" | "unsafe" | "unknown";

export interface Site {
    lat_deg: number;
    lon_deg: number;
    elev_m: number;
    source?: "config" | "mount" | "operator";
}

/** §2.13 `mount` block: today's keys plus every new one. */
export interface MountBlock extends DeviceBlockCommon {
    tracking: boolean | null;
    /** Mount-native frame (`frame`), hours. */
    ra: number | null;
    dec: number | null;
    parked?: boolean | null;
    pier_side?: PierSide | string | null;
    slewing?: boolean | null;
    at_home?: boolean | null;
    alt?: number | null;
    az?: number | null;
    lst_h?: number | null;
    ha_h?: number | null;
    frame?: EqFrame | null;
    ra_j2000?: number | null;
    dec_j2000?: number | null;
    tracking_rate?: DriveRate | null;
    is_pulse_guiding?: boolean | null;
    guide_rate_ra_arcsec_s?: number | null;
    guide_rate_dec_arcsec_s?: number | null;
    alignment_mode?: AlignmentMode | null;
    can_flip?: boolean | null;
    site?: Site | null;
    clock_delta_s?: number | null;
    /** The worker link state (the mount block's own `state`). */
    state?: LinkState | null;
}

export type PositionSource =
    | "device"
    | "emulated"
    | "emulated_restored"
    | "emulated_origin_unknown"
    | "unknown_after_reconnect";

/** §2.13 `focuser` block. `position` is logical; null while the origin is unknown. */
export interface FocuserBlock extends DeviceBlockCommon {
    /** Null until IsMoving has been read: unknown is not "idle". */
    moving: boolean | null;
    position: number | null;
    raw_position?: number | null;
    absolute?: boolean;
    max_range?: number | null;
    max_increment?: number | null;
    hfr?: number | null;
    temperature?: number | null;
    temperature_source?: string | null;
    temperature_age_ms?: number | null;
    step_size_um?: number | null;
    temp_comp?: boolean | null;
    temp_comp_available?: boolean | null;
    position_source?: PositionSource | null;
    backlash_model?: { model: string; in_steps?: number; out_steps?: number } | null;
    backlash_owner?: "achernar" | "firmware" | null;
    settling?: boolean | null;
    active_op?: string | null;
    state?: LinkState | null;
}

export interface FilterSlotView {
    /** 0-based slot index. */
    index: number;
    name: string;
    /** Focus offset from the identity profile, in focuser steps. */
    offset: number | null;
}

/** §2.13 `filter_wheel` block, the frontend's shape plus `slots` and `verified`. */
export interface FilterWheelBlock extends DeviceBlockCommon {
    filter?: string | null;
    /** 0-based; null while moving or unknown. */
    slot?: number | null;
    position?: number | null;
    moving?: boolean;
    offset?: number | null;
    slots?: FilterSlotView[];
    /** The reported slot was verified by a handshake read. */
    verified?: boolean;
    verified_slot?: number | null;
    active_op?: string | null;
    state?: LinkState | null;
}

/** §2.13 `guiding` block. `state` is PHD2's AppState; `link_state` is the worker's. */
export interface GuidingBlock extends DeviceBlockCommon {
    state?: GuiderAppState | null;
    rms_ra_arcsec?: number | null;
    rms_dec_arcsec?: number | null;
    /** Arcsec; null without a pixel scale. */
    rms_total?: number | null;
    rms_ra_px?: number | null;
    rms_dec_px?: number | null;
    rms_total_px?: number | null;
    steps_n?: number | null;
    dx?: number | null;
    dy?: number | null;
    snr?: number | null;
    star_mass?: number | null;
    hfd?: number | null;
    settling?: boolean | null;
    settle?: { distance_px: number; time_s: number; settle_time_s: number; star_locked: boolean } | null;
    pixel_scale?: number | null;
    calibrated?: boolean | null;
    last_dither?: { at: string; dx: number | null; dy: number | null; settle_status: number | null } | null;
    phd2_version?: string | null;
    profile?: string | null;
    camera_name?: string | null;
}

/** One observed field: a value or an explicit unknown, with its state and age. */
export interface ObservedField {
    value: number | null;
    state?: ObsState | null;
    observed_at?: string | null;
    age_ms?: number | null;
    source?: string | null;
}

export type WeatherFieldKey =
    | "cloud_cover_pct"
    | "dew_point_c"
    | "humidity_pct"
    | "pressure_hpa"
    | "rain_rate_mm_h"
    | "sky_brightness_lux"
    | "sky_quality_mag_arcsec2"
    | "sky_temperature_c"
    | "star_fwhm_arcsec"
    | "temperature_c"
    | "wind_direction_deg"
    | "wind_gust_ms"
    | "wind_speed_ms";

/** §2.13 `weather` block. */
export interface WeatherBlock extends DeviceBlockCommon {
    fields?: Partial<Record<WeatherFieldKey, ObservedField>>;
    temperature?: number | null;
    humidity?: number | null;
    dew_point?: number | null;
    sensor_ages_s?: Record<string, number>;
    state?: LinkState | null;
}

/** §2.13 `safety_monitor` block: this monitor's level with its age. */
export interface SafetyMonitorBlock extends DeviceBlockCommon {
    level?: SafetyLevel | null;
    /** Null when the level is unknown or stale: unknown is never safe. */
    is_safe?: boolean | null;
    reading?: { value: SafetyLevel | null; state?: ObsState | null; age_ms?: number | null } | null;
    detail?: string | null;
    state?: LinkState | null;
}

/** Critical-transaction flags (H20). */
export interface BusySet {
    exposing: boolean;
    autofocus: boolean;
    slewing: boolean;
    settling: boolean;
    parking: boolean;
    flipping: boolean;
    centering: boolean;
    dithering: boolean;
    guider_settling: boolean;
    filter_changing: boolean;
    cover_moving: boolean;
    calibrator_changing: boolean;
}

export type ResumeBlockKind =
    | { kind: "filter_handshake_incomplete" }
    | { kind: "outcome_unknown"; device: string; op: string }
    | { kind: "safety_unsafe" }
    | { kind: "safety_unknown" }
    | { kind: "guider_lost" }
    | { kind: "mount_lost" }
    | { kind: "limit_reached" }
    | { kind: "estop" };

/** An active block on automatic exposure resumption (H22). */
export interface ResumeBlock {
    id: string;
    kind: ResumeBlockKind;
    raised_at: string;
    raised_mono_ns?: number;
    device_kind?: string | null;
    connection_id?: string | null;
    command_id?: string | null;
    reason: string;
}

/** One journal or `EQUIPMENT_EVENT` entry (§4.2): the envelope plus `kind` and `payload`. */
export interface EquipmentEvent {
    seq: number;
    event_id: string;
    at_utc: string;
    mono_ns?: number;
    device_kind?: string | null;
    connection_id?: string | null;
    command_id?: string | null;
    device_time?: string | null;
    kind: string;
    payload?: unknown;
}

export interface EquipmentEventMessage {
    type: "EQUIPMENT_EVENT";
    event: EquipmentEvent;
}

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
    device_health?: Record<string, DeviceHealthEntry>;
    daemon_id?: string;
    hardware_id?: string;
    last_heartbeat?: string;
    mount?: MountBlock | null;
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
    focuser?: FocuserBlock | null;
    filter_wheel?: FilterWheelBlock | null;
    /** The guider block; `/status` also sends it as `guider`. */
    guiding?: GuidingBlock | null;
    guider?: GuidingBlock | null;
    weather?: WeatherBlock | null;
    safety_monitor?: SafetyMonitorBlock | null;
    /** Tier 2 kinds: null until their drivers exist. */
    rotator?: (DeviceBlockCommon & Record<string, unknown>) | null;
    cover_calibrator?: (DeviceBlockCommon & Record<string, unknown>) | null;
    switch?: (DeviceBlockCommon & Record<string, unknown>) | null;
    dome?: (DeviceBlockCommon & Record<string, unknown>) | null;
    busy?: BusySet | null;
    resume_blocks?: ResumeBlock[] | null;
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
        mount?: Partial<MountBlock> | null;
        camera?: { name?: string; connected?: boolean; temperature?: number | null; status?: string; progress?: number; streaming?: boolean; cooling?: boolean };
        // Null until a guider is connected and reports: a zeroed RMS block
        // would be a measurement of nothing.
        guiding?: GuidingBlock | null;
        focuser?: Partial<FocuserBlock> | null;
        filter_wheel?: FilterWheelBlock | null;
        weather?: WeatherBlock | null;
        safety_monitor?: SafetyMonitorBlock | null;
        device_health?: Record<string, DeviceHealthEntry>;
        busy?: BusySet | null;
        resume_blocks?: ResumeBlock[] | null;
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

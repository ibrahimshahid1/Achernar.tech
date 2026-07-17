import { env } from "../env";
import { httpJson } from "./http";
import type { EquipmentState, CaptureParams, RelayCommandResult } from "../types/achernar";

/** A single manually-controllable camera option (from GET /camera/parameters). */
export interface CamParam {
    name: string;
    kind: string; // bool | number | msec | percent | kelvin | celsius | selection
    read_only: boolean;
    min: number;
    max: number;
    step: number;
    default: number;
    current: number;
}

/** A selectable stream format / binning mode (from GET /camera/formats). */
export interface CamFormat {
    index: number;
    format: string;
    width: number;
    height: number;
    bin: number;
    framerate: number;
}

export interface DriverCapability {
    id: string;
    label: string;
    compiled: boolean;
    enabled: boolean;
    reason?: string;
    option_hint?: string;
}

export interface DriverCapabilityReport {
    bridge_version: string;
    camera: DriverCapability[];
    mount: DriverCapability[];
    focuser: DriverCapability[];
}

/** Result of a single-shot capture (from POST /camera/capture). */
export interface CaptureResult {
    id: string;
    width: number;
    height: number;
    path?: string;
    metadata: {
        exposure: number;
        gain: number;
        temperature: number;
        timestamp: string;
    };
}

/**
 * Daemon API (Edge Control)
 * PDF Section 5.1.1 - Daemon Endpoints
 */
export const daemonApi = {
    // GET /api/v1/status
    async getStatus(): Promise<EquipmentState> {
        return httpJson<EquipmentState>(`${env.daemonApiBase}/status`);
    },

    // GET /api/v1/drivers — build-time native hardware support report
    async getDriverCapabilities(): Promise<DriverCapabilityReport> {
        return httpJson<DriverCapabilityReport>(`${env.daemonApiBase}/drivers`);
    },

    // POST /api/v1/mount/slew
    async slewMount(payload: { ra: number; dec: number }): Promise<{ ok: boolean }> {
        return httpJson(`${env.daemonApiBase}/mount/slew`, {
            method: "POST",
            body: payload
        });
    },

    // POST /api/v1/camera/capture — single-shot, saves a FITS frame
    async capture(payload: CaptureParams): Promise<CaptureResult> {
        return httpJson<CaptureResult>(
            `${env.daemonApiBase}/camera/capture`,
            { method: "POST", body: payload }
        );
    },

    // POST /api/v1/camera/stream/start (optionally choose format/binning by index)
    async startCameraStream(formatIndex?: number): Promise<{ status: string; format: string; width: number; height: number; bin?: number }> {
        const query = typeof formatIndex === "number" ? `?format_index=${formatIndex}` : "";
        return httpJson(`${env.daemonApiBase}/camera/stream/start${query}`, {
            method: "POST",
        });
    },

    // POST /api/v1/camera/abort
    async stopCameraStream(): Promise<void> {
        await httpJson(`${env.daemonApiBase}/camera/abort`, { method: "POST" });
    },

    // GET /api/v1/camera/parameters — full manual control surface
    async getCameraParameters(): Promise<CamParam[]> {
        return httpJson<CamParam[]>(`${env.daemonApiBase}/camera/parameters`);
    },

    // POST /api/v1/camera/parameters — set one option (exposure, gain, cooler_target, ...)
    async setCameraParameter(name: string, value: number): Promise<{ status: string; name: string; value: number }> {
        return httpJson(`${env.daemonApiBase}/camera/parameters`, {
            method: "POST",
            body: { name, value },
        });
    },

    // GET /api/v1/camera/formats — selectable resolutions / binning modes
    async getCameraFormats(): Promise<CamFormat[]> {
        return httpJson<CamFormat[]>(`${env.daemonApiBase}/camera/formats`);
    },

    // POST /api/v1/connect — connect a camera driver (e.g. "svb" for SVBony)
    async connectCamera(driver: string, option?: string): Promise<{ status: string; camera?: string; cameras_found?: string[] }> {
        return httpJson(`${env.daemonApiBase}/connect`, {
            method: "POST",
            body: { equipment_type: "camera", driver, option },
        });
    },

    // WS /api/v1/camera/stream
    cameraStreamWsUrl() {
        const url = new URL(env.daemonApiBase);
        url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
        url.pathname = `${url.pathname.replace(/\/$/, "")}/camera/stream`;
        url.search = "";
        return url.toString();
    },

    // POST /api/v1/focuser/move
    async moveFocuser(payload: { steps: number }): Promise<void> {
        await httpJson(`${env.daemonApiBase}/focuser/move`, {
            method: "POST",
            body: payload,
        });
    },

    async relayCommand(action: string, payload: Record<string, unknown> = {}): Promise<RelayCommandResult> {
        switch (action) {
            case "drivers.capabilities":
                return { ok: true, payload: await this.getDriverCapabilities() as unknown as Record<string, unknown> };
            case "equipment.status":
                return { ok: true, payload: await this.getStatus() as unknown as Record<string, unknown> };
            case "equipment.connect":
                return {
                    ok: true,
                    payload: await httpJson(`${env.daemonApiBase}/connect`, {
                        method: "POST",
                        body: payload,
                    }) as Record<string, unknown>,
                };
            case "camera.parameters":
                return { ok: true, payload: await this.getCameraParameters() as unknown as Record<string, unknown> };
            case "camera.formats":
                return { ok: true, payload: await this.getCameraFormats() as unknown as Record<string, unknown> };
            case "camera.set_parameter":
                return {
                    ok: true,
                    payload: await this.setCameraParameter(
                        String(payload.name ?? ""),
                        Number(payload.value ?? 0),
                    ) as unknown as Record<string, unknown>,
                };
            case "camera.start_stream":
                return {
                    ok: true,
                    payload: await this.startCameraStream(
                        typeof payload.format_index === "number" ? payload.format_index : undefined,
                    ) as unknown as Record<string, unknown>,
                };
            case "camera.abort":
                await this.stopCameraStream();
                return { ok: true, payload: { status: "stopped" } };
            case "camera.capture":
                return { ok: true, payload: await this.capture(payload as unknown as CaptureParams) as unknown as Record<string, unknown> };
            case "mount.slew":
                return {
                    ok: true,
                    payload: await this.slewMount({
                        ra: Number(payload.ra ?? 0),
                        dec: Number(payload.dec ?? 0),
                    }) as unknown as Record<string, unknown>,
                };
            case "mount.set_tracking":
                return {
                    ok: true,
                    payload: await httpJson(`${env.daemonApiBase}/mount/tracking`, {
                        method: "POST",
                        body: { enabled: Boolean(payload.enabled) },
                    }) as Record<string, unknown>,
                };
            case "mount.abort":
                return {
                    ok: true,
                    payload: await httpJson(`${env.daemonApiBase}/mount/abort`, {
                        method: "POST",
                    }) as Record<string, unknown>,
                };
            case "mount.sync":
                return {
                    ok: true,
                    payload: await httpJson(`${env.daemonApiBase}/mount/sync`, {
                        method: "POST",
                        body: {
                            ra: Number(payload.ra ?? 0),
                            dec: Number(payload.dec ?? 0),
                        },
                    }) as Record<string, unknown>,
                };
            case "focuser.move_relative":
                await this.moveFocuser({ steps: Number(payload.steps ?? 0) });
                return { ok: true, payload: { status: "moving", steps: Number(payload.steps ?? 0) } };
            case "focuser.move_absolute":
                return {
                    ok: true,
                    payload: await httpJson(`${env.daemonApiBase}/focuser/moveto`, {
                        method: "POST",
                        body: { position: Number(payload.position ?? 0) },
                    }) as Record<string, unknown>,
                };
            case "focuser.position":
                return {
                    ok: true,
                    payload: await httpJson(`${env.daemonApiBase}/focuser/position`) as Record<string, unknown>,
                };
            case "focuser.abort":
                return {
                    ok: true,
                    payload: await httpJson(`${env.daemonApiBase}/focuser/abort`, {
                        method: "POST",
                    }) as Record<string, unknown>,
                };
            default:
                throw new Error(`Unsupported local daemon action: ${action}`);
        }
    },
};

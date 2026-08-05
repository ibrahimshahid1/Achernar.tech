import { env } from "../env";
import { httpJson } from "./http";
import type { EquipmentState, ImagingSession, RelayCommandResult, DaemonSummary, CaptureParams } from "../types/achernar";
import type { CamFormat, CamParam, CaptureResult, DriverCapabilityReport } from "./daemon";

/**
 * Cloud API (Django Apps)
 * Per PDF Section 16.3: Django Backend Services
 * Endpoints are organized by Django app modules
 */
export const cloudApi = {
    async listDaemons(): Promise<DaemonSummary[]> {
        return httpJson<DaemonSummary[]>(`${env.cloudApiBase}/equipment/daemons/`);
    },

    /**
     * Imaging App - Session management
     */
    async listSessions(): Promise<ImagingSession[]> {
        return httpJson<ImagingSession[]>(`${env.cloudApiBase}/imaging/sessions/`);
    },

    /**
     * Equipment App - Connected equipment from daemon relay
     */
    async getConnectedEquipment(): Promise<Array<{
        id: string;
        type: string;
        name: string;
        site: string;
        status: string;
        tags: string[];
        meta: Record<string, any>;
    }>> {
        return httpJson(`${env.cloudApiBase}/equipment/connected/`);
    },

    /**
     * Equipment App - Raw daemon status relayed by Django
     */
    async getDaemonStatus(): Promise<EquipmentState> {
        return httpJson<EquipmentState>(`${env.cloudApiBase}/equipment/daemon-status/`);
    },

    async startCameraStream(formatIndex?: number): Promise<{ status: string; format: string; width: number; height: number; bin?: number }> {
        return httpJson(`${env.cloudApiBase}/equipment/camera/start-stream/`, {
            method: "POST",
            body: typeof formatIndex === "number" ? { payload: { format_index: formatIndex } } : {},
        });
    },

    async cameraSnapshot(): Promise<{
        status: string;
        format: string;
        mime_type?: string;
        width: number;
        height: number;
        frame_counter?: number;
        timestamp?: number;
        data_base64: string;
    }> {
        return httpJson(`${env.cloudApiBase}/equipment/camera/snapshot/`, {
            method: "POST",
        });
    },

    async relayCommand(
        action: string,
        payload: Record<string, unknown> = {},
        daemonId?: string,
    ): Promise<RelayCommandResult> {
        return httpJson(`${env.cloudApiBase}/equipment/command/`, {
            method: "POST",
            body: {
                action,
                payload,
                daemon_id: daemonId,
            },
        });
    },

    async relayPayload<T>(
        action: string,
        payload: Record<string, unknown> = {},
        daemonId?: string,
    ): Promise<T> {
        const result = await this.relayCommand(action, payload, daemonId);
        if (result.pending) {
            throw new Error(`${action} is still pending on the daemon.`);
        }
        if (result.ok === false) {
            throw new Error(result.error ?? `${action} failed.`);
        }
        return (result.payload ?? {}) as T;
    },

    async getDriverCapabilities(): Promise<DriverCapabilityReport> {
        return this.relayPayload<DriverCapabilityReport>("drivers.capabilities");
    },

    /** `index` selects which camera within the driver's enumeration; see the
     *  daemon-transport version of this method. */
    async connectCamera(
        driver: string,
        option?: string,
        index?: number,
    ): Promise<{ status: string; camera?: string; cameras_found?: string[]; index?: number }> {
        return this.relayPayload("equipment.connect", {
            equipment_type: "camera",
            driver,
            option,
            index,
        });
    },

    async stopCameraStream(): Promise<void> {
        await this.relayPayload("camera.abort");
    },

    async getCameraParameters(): Promise<CamParam[]> {
        return this.relayPayload<CamParam[]>("camera.parameters");
    },

    async setCameraParameter(name: string, value: number): Promise<{ status: string; name: string; value: number }> {
        return this.relayPayload("camera.set_parameter", { name, value });
    },

    async getCameraFormats(): Promise<CamFormat[]> {
        return this.relayPayload<CamFormat[]>("camera.formats");
    },

    async capture(payload: CaptureParams): Promise<CaptureResult> {
        return this.relayPayload<CaptureResult>("camera.capture", payload as unknown as Record<string, unknown>);
    },

    /**
     * Equipment App - Equipment matrix (full registry)
     */
    async listEquipmentMatrix(): Promise<
        Array<{ id: string; type: string; name: string; site: string; status: string }>
    > {
        return httpJson(`${env.cloudApiBase}/equipment/matrix/`);
    },

    /**
     * Atlas  App - Target database
     */
    async listAtlasTargets(): Promise<Array<{ id: string; name: string; ra: number; dec: number }>> {
        return httpJson(`${env.cloudApiBase}/atlas/targets/`);
    },

    /**
     * Docs App - Documentation
     */
    async listDocs(): Promise<Array<{ id: string; title: string; updated_at: string }>> {
        return httpJson(`${env.cloudApiBase}/docs/`);
    },

    /**
     * AI Engine App - Cloud-based recommendations
     */
    async getExposureRecommendation(context: {
        target: string;
        filter: string;
        moon_alt: number;
    }): Promise<{ exposure: number; gain: number; confidence: number }> {
        return httpJson(`${env.cloudApiBase}/ai-engine/recommend/exposure`, {
            method: "POST",
            body: context,
        });
    },
};

import { env } from "../env";
import { httpJson } from "./http";
import type { EquipmentState, CaptureParams, ImageData } from "../types/achernar";

/**
 * Daemon API (Local Edge)
 * Communicates with Rust daemon on localhost
 * Per PDF Section 5.1.1: Daemon Endpoints
 */
export const daemonApi = {
    /**
     * Get global equipment state
     * GET /status
     */
    async getStatus(): Promise<EquipmentState> {
        return httpJson<EquipmentState>(`${env.daemonApiBase}/status`);
    },

    /**
     * Trigger camera capture
     * POST /camera/capture
     */
    async captureImage(params: CaptureParams): Promise<{ job_id: string; status: string }> {
        return httpJson(`${env.daemonApiBase}/camera/capture`, {
            method: "POST",
            body: params,
        });
    },

    /**
     * Slew mount to coordinates
     * POST /mount/slew
     */
    async slewMount(ra: number, dec: number): Promise<void> {
        await httpJson(`${env.daemonApiBase}/mount/slew`, {
            method: "POST",
            body: { ra, dec },
        });
    },

    /**
     * Move focuser
     * POST /focuser/move
     */
    async moveFocuser(steps: number): Promise<void> {
        await httpJson(`${env.daemonApiBase}/focuser/move`, {
            method: "POST",
            body: { steps },
        });
    },

    /**
     * Run AI analysis on frame
     * POST /ai/analyze
     */
    async analyzeFrame(frameId: string): Promise<{
        focus_hfr: number;
        cloud_prob: number;
        background_adu: number;
    }> {
        return httpJson(`${env.daemonApiBase}/ai/analyze`, {
            method: "POST",
            body: { frame_id: frameId },
        });
    },
};

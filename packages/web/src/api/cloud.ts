import { env } from "../env";
import { httpJson } from "./http";
import type { ImagingSession } from "../types/achernar";

/**
 * Cloud API (Django Apps)
 * Per PDF Section 16.3: Django Backend Services
 * Endpoints are organized by Django app modules
 */
export const cloudApi = {
    /**
     * Imaging App - Session management
     */
    async listSessions(): Promise<ImagingSession[]> {
        return httpJson<ImagingSession[]>(`${env.cloudApiBase}/imaging/sessions/`);
    },

    /**
     * Equipment App - Equipment matrix
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

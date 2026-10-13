import { achernarApi } from "./achernar";

export interface CocContext {
    generation: number;
    connection_id: string;
    width: number;
    height: number;
}
export interface CocRoi { left: number; top: number; right: number; bottom: number }
export interface CocSample {
    sample_id: string;
    context: CocContext;
    source_timestamp: number;
    received_at: string;
    source_format: string;
    frame_counter: number;
    inference_ms: number;
    model: string;
    model_sha256: string;
    image_size: number;
    input_png_base64: string;
    overlay_png_base64: string;
    roi: CocRoi | null;
    summary: {
        status: string; reason: string | null;
        cloud_area_fraction_of_roi: number | null;
        contamination_area_fraction_of_roi: number | null;
        cloud_fraction_of_classified_sky: number | null;
        probability_calibrated: false;
        runtime_eligible: false;
    };
}
export interface CocStatus {
    contract_version: "coc-segmentation-v1";
    shadow_only: true;
    runtime_eligible: false;
    enabled: boolean;
    model: string;
    status: string;
    reason: string | null;
    models: { id: string; loaded: boolean; model_sha256: string | null; error: string | null }[];
    latest: CocSample | null;
    roi: CocRoi | null;
    roi_valid_for_current_camera: boolean;
}

export function validRoi(roi: CocRoi): boolean {
    return Object.values(roi).every(Number.isFinite) && roi.left >= 0 && roi.top >= 0
        && roi.right <= 1 && roi.bottom <= 1 && roi.right > roi.left && roi.bottom > roi.top;
}

export function freshCocSample(status: CocStatus | undefined, failed: boolean, updatedAt: number,
    now: number, cameraId: string | null, streaming: boolean, frame?: {stream_generation?: number; width?: number; height?: number} | null): CocSample | null {
    const sample = status?.latest;
    if (!status?.enabled || status.status !== "ready" || failed || !updatedAt || now - updatedAt > 4000
        || !streaming || !cameraId || sample?.context.connection_id !== cameraId
        || !frame || sample.context.generation !== frame.stream_generation || sample.context.width !== frame.width || sample.context.height !== frame.height
        || !Number.isFinite(sample.source_timestamp) || now / 1000 - sample.source_timestamp > 5
        || now / 1000 - sample.source_timestamp < -1) return null;
    return sample;
}

async function request<T>(action: string, payload: Record<string, unknown> = {}): Promise<T> {
    const response = await achernarApi.command(action, payload);
    if (response.pending) throw new Error("COC request is still pending; refresh status before retrying.");
    if (response.ok === false) throw new Error(response.error ?? "COC request failed.");
    return response.payload as T;
}

export const cocApi = {
    async status(): Promise<CocStatus> {
        const value = await request<CocStatus>("coc.status");
        if (value?.contract_version !== "coc-segmentation-v1" || value.shadow_only !== true
            || value.runtime_eligible !== false || !Array.isArray(value.models)) {
            throw new Error("The daemon does not expose the expected COC shadow contract.");
        }
        return value;
    },
    configure(enabled: boolean, model: string, roi: CocRoi | null, context: CocContext | null): Promise<CocStatus> {
        if (roi && (!validRoi(roi) || !context)) return Promise.reject(new Error("Select a valid sky region on a current frame."));
        return request("coc.configure", { enabled, model, roi, context });
    },
    freeze(sampleId: string): Promise<CocSample> {
        return request("coc.freeze_sample", { sample_id: sampleId });
    },
    save(sampleId: string, notes: string): Promise<{ status: string; sample_id: string; path: string; training_approved: false }> {
        return request("coc.save_sample", { sample_id: sampleId, notes });
    },
};

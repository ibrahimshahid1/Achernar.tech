import type { CameraDescriptor, CamFormat, CaptureResult } from "../api/achernar";
import type { CaptureParams } from "../types/achernar";

interface CaptureApi {
    capture: (params: CaptureParams) => Promise<CaptureResult>;
    startCameraStream: (formatIndex?: number) => Promise<unknown>;
}

export function canStartCameraPreview(camera: CameraDescriptor | null, format?: CamFormat): boolean {
    return camera?.connected === true
        && camera.capabilities.supports_stream
        && format !== undefined
        && (!camera.capabilities.supports_dedicated_still || format.mode === "liveview");
}

/** Route from daemon capabilities, never from an encoding or driver-name guess. */
export async function captureCameraBundle(
    api: CaptureApi,
    camera: CameraDescriptor | null,
    format: CamFormat | undefined,
    params: Pick<CaptureParams, "exposure" | "gain">,
    onStreamStarted: () => void,
): Promise<CaptureResult> {
    if (!camera?.connected) {
        throw new Error("No camera connected.");
    }
    const dedicated = camera.capabilities.supports_dedicated_still;
    if (!dedicated) {
        if (!camera.capabilities.supports_stream || !format) {
            throw new Error("Select an available sensor stream format before capturing.");
        }
        if (!camera.streaming) {
            await api.startCameraStream(format.index);
            onStreamStarted();
        }
    }
    return api.capture({
        ...params,
        binning: dedicated ? 1 : (format?.bin ?? 1),
        retention_policy: "retain_on_camera",
    });
}

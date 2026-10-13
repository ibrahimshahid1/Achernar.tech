import type { CaptureRetentionDecision } from "../types/achernar";

/** Keep the policy decision distinct from evidence of a persistent camera copy. */
export function captureRetentionSummary(retention?: CaptureRetentionDecision): string {
    if (!retention) {
        return "Camera retention was not reported. A retained second copy has not been verified.";
    }

    const policy = retention.effective_policy === "retain_on_camera"
        ? "Keep camera originals"
        : "Delete after verified local commit";
    let cameraCopy: string;
    switch (retention.camera_copy_state) {
        case "retained":
            cameraCopy = "Camera copy retained.";
            break;
        case "not_applicable":
            cameraCopy = "This acquisition has no separate camera copy.";
            break;
        default:
            cameraCopy = "A persistent camera copy has not been verified.";
    }
    const deletion = retention.camera_files_deleted
        ? "Camera files were deleted."
        : "No camera-file deletion was performed.";
    return `Effective policy: ${policy} (v${retention.policy_version}). ${cameraCopy} ${deletion} ${retention.reason}`;
}

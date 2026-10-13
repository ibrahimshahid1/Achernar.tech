import { describe, expect, it } from "vitest";
import type { CaptureRetentionDecision } from "../types/achernar";
import { captureRetentionSummary } from "./captureRetention";

const retainedPolicy: CaptureRetentionDecision = {
    requested_policy: "retain_on_camera",
    effective_policy: "retain_on_camera",
    policy_version: 1,
    camera_copy_state: "unknown",
    camera_files_deleted: false,
    deletion_eligible: false,
    reason: "Persistent camera storage has not been verified.",
};

describe("capture retention presentation", () => {
    it("does not turn a retain policy or local commit into a verified second copy", () => {
        const message = captureRetentionSummary(retainedPolicy);
        expect(message).toContain("Keep camera originals (v1)");
        expect(message).toContain("persistent camera copy has not been verified");
        expect(message).toContain("No camera-file deletion was performed");
        expect(message).not.toContain("Camera copy retained");
    });

    it("distinguishes a sensor-only acquisition from an unknown card target", () => {
        const message = captureRetentionSummary({ ...retainedPolicy, camera_copy_state: "not_applicable" });
        expect(message).toContain("no separate camera copy");
        expect(message).not.toContain("Camera copy retained");
    });

    it("reports a retained copy only when the daemon explicitly reports it", () => {
        expect(captureRetentionSummary({ ...retainedPolicy, camera_copy_state: "retained" }))
            .toContain("Camera copy retained");
    });

    it("does not infer retention when an older daemon omits the decision", () => {
        const message = captureRetentionSummary();
        expect(message).toContain("not reported");
        expect(message).toContain("retained second copy has not been verified");
    });

    it("shows the effective decision and deletion outcome rather than assuming the request applied", () => {
        const message = captureRetentionSummary({
            ...retainedPolicy,
            requested_policy: "delete_after_verified_local_commit",
            reason: "Deletion disabled until storage qualification.",
        });
        expect(message).toContain("Effective policy: Keep camera originals");
        expect(message).toContain("Deletion disabled until storage qualification");
        expect(message).not.toContain("Effective policy: Delete");
    });
});

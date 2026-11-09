import { useState } from "react";
import { useDaemonWS, type DaemonWsMessage } from "./useDaemonWS";
import { mapSharedObservation, type SharedObservation } from "../lib/sharedObservation";

export type {
    BackgroundProvenance,
    BackgroundSummary,
    SharedObservation,
    StarRejectionTallies,
} from "../lib/sharedObservation";

/**
 * Subscribe to VISION_UPDATE (SharedObservation) events.
 *
 * Missing measurements stay `null` and must render as "—" with their reason
 * (see `displayBackgroundLevel` / `displayBackgroundNoise`), never as 0.
 */
export function useSharedObs() {
    const [observation, setObservation] = useState<SharedObservation | null>(null);

    useDaemonWS({
        VISION_UPDATE: (data: DaemonWsMessage) => {
            const mapped = mapSharedObservation(data.observation);
            if (mapped) {
                setObservation(mapped);
            }
        },
    });

    return observation;
}

import { useState } from "react";
import { useDaemonWS, type DaemonWsMessage } from "./useDaemonWS";

export interface SharedObservation {
    focusQuality: number;
    hfr: number;
    fwhm: number;
    starCount: number;
    cloudCondition: string;
    cloudProbability: number;
    trailingDetected: boolean;
    trailingAxis: string;
    backgroundAdu: number;
    temperature: number | null;
}

/**
 * Subscribe to VisionUpdate (SharedObservation) events.
 */
export function useSharedObs() {
    const [observation, setObservation] = useState<SharedObservation | null>(null);

    useDaemonWS({
        VISION_UPDATE: (data: DaemonWsMessage) => {
            const obs = data.observation as any;
            if (obs) {
                setObservation({
                    focusQuality: obs.focus_quality ?? 0,
                    hfr: obs.hfr ?? 0,
                    fwhm: obs.fwhm ?? 0,
                    starCount: obs.star_count ?? 0,
                    cloudCondition: obs.cloud_condition ?? "unknown",
                    cloudProbability: obs.cloud_probability ?? 0,
                    trailingDetected: obs.trailing_detected ?? false,
                    trailingAxis: obs.trailing_axis ?? "none",
                    backgroundAdu: obs.background_adu ?? 0,
                    temperature: obs.temperature ?? null,
                });
            }
        },
    });

    return observation;
}

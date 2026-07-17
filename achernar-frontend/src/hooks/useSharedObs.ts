import { useState } from "react";
import { useDaemonWS, type DaemonWsMessage } from "./useDaemonWS";

export interface SharedObservation {
    timestamp: string;
    focusQuality: number;
    predictedHfr: number;
    sharpnessScore: number;
    qualityUncertainty: number;
    hfr: number;
    fwhm: number;
    seeingEstimateArcsec: number;
    starCount: number;
    cloudProbability: number;
    cloudEvent: string;
    trailingSeverity: number;
    trailingAxis: string | null;
    trailingDirectionPa: number | null;
    backgroundAdu: number;
    skyBackgroundAdu: number;
    backgroundSigma: number;
    snrProxyMedian: number;
    saturationFraction: number;
    frameQualityScore: number;
    observationValid: boolean;
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
                    timestamp: obs.timestamp ?? "",
                    focusQuality: obs.focus_quality ?? 0,
                    predictedHfr: obs.predicted_hfr ?? obs.hfr ?? 0,
                    sharpnessScore: obs.sharpness_score ?? 0,
                    qualityUncertainty: obs.quality_uncertainty ?? 1,
                    hfr: obs.hfr ?? 0,
                    fwhm: obs.fwhm ?? 0,
                    seeingEstimateArcsec: obs.seeing_estimate_arcsec ?? obs.fwhm ?? 0,
                    starCount: obs.star_count ?? 0,
                    cloudProbability: obs.cloud_probability ?? 0,
                    cloudEvent: obs.cloud_event ?? "None",
                    trailingSeverity: obs.trailing_severity ?? 0,
                    trailingAxis: obs.trailing_axis ?? null,
                    trailingDirectionPa: obs.trailing_direction_pa ?? null,
                    backgroundAdu: obs.background_adu ?? 0,
                    skyBackgroundAdu: obs.sky_background_adu ?? obs.background_adu ?? 0,
                    backgroundSigma: obs.background_sigma ?? 0,
                    snrProxyMedian: obs.snr_proxy_median ?? 0,
                    saturationFraction: obs.saturation_fraction ?? 0,
                    frameQualityScore: obs.frame_quality_score ?? 0,
                    observationValid: obs.observation_valid ?? false,
                });
            }
        },
    });

    return observation;
}

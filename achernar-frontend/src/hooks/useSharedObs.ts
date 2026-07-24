import { useState } from "react";
import { useDaemonWS, type DaemonWsMessage } from "./useDaemonWS";

/**
 * Mirror of the daemon's `SharedObservation` (Section 8.1).
 *
 * Width/quality fields are `number | null`: a missing measurement is `null`, and
 * must render as "--" rather than 0 (a 0 HFR would read as perfect focus on a
 * blank frame). The old `focusQuality` / `sharpnessScore` / `frameQualityScore`
 * reciprocal scores were removed — they were three views of one HFR value with
 * no calibrated meaning. Render physical widths/scatter plus `status` instead.
 */
/** Per-reason counts of stars rejected by the daemon's source-validity gates. */
export interface StarRejectionTallies {
    saturated: number;
    edge_truncated: number;
    low_snr: number;
    too_few_pixels: number;
    elongated: number;
    blended: number;
    invalid_hfr: number;
}

const EMPTY_TALLIES: StarRejectionTallies = {
    saturated: 0,
    edge_truncated: 0,
    low_snr: 0,
    too_few_pixels: 0,
    elongated: 0,
    blended: 0,
    invalid_hfr: 0,
};

export interface SharedObservation {
    timestamp: string;
    status: string; // "Valid" | "EmptyFrame" | "NoSources" | "NoUsableStars"
    rejectionReasons: string[];
    hfrPx: number | null;
    hfdPx: number | null;
    fwhmPx: number | null;
    imageFwhmArcsec: number | null;
    qualityUncertainty: number | null;
    hfrLogSigma: number | null;
    starCount: number;
    starCountUsed: number;
    /** Why detected stars were not used ("14 detected / 5 used" explained). */
    starRejections: StarRejectionTallies;
    /** FQE shadow prediction (px) — informational; classical hfrPx is authority. */
    fqeHfrPx: number | null;
    /** FQE frame log-HFR sigma, comparable to hfrLogSigma. */
    fqeSigmaLogHfr: number | null;
    /** Number of centered gated crops pooled into the FQE result. */
    fqeStarCount: number | null;
    /** Model version that produced the shadow prediction (e.g. "1.1.0"). */
    fqeModelVersion: string | null;
    /** Whether per-star and post-pooling frame calibration were both applied. */
    fqeCalibrated: boolean | null;
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
    pointingErrorArcmin: number | null;
    wcsResidualArcsec: number | null;
    observationValid: boolean;
}

const num = (v: unknown): number | null => (typeof v === "number" && isFinite(v) ? v : null);

/**
 * Subscribe to VISION_UPDATE (SharedObservation) events.
 */
export function useSharedObs() {
    const [observation, setObservation] = useState<SharedObservation | null>(null);

    useDaemonWS({
        VISION_UPDATE: (data: DaemonWsMessage) => {
            const obs = data.observation as any;
            if (obs) {
                setObservation({
                    timestamp: obs.timestamp ?? "",
                    status: obs.status ?? "NoSources",
                    rejectionReasons: obs.rejection_reasons ?? [],
                    hfrPx: num(obs.hfr_px),
                    hfdPx: num(obs.hfd_px),
                    fwhmPx: num(obs.fwhm_px),
                    imageFwhmArcsec: num(obs.image_fwhm_arcsec),
                    qualityUncertainty: num(obs.quality_uncertainty),
                    hfrLogSigma: num(obs.hfr_log_sigma),
                    starCount: obs.star_count ?? 0,
                    starCountUsed: obs.star_count_used ?? 0,
                    starRejections: { ...EMPTY_TALLIES, ...(obs.star_rejections ?? {}) },
                    fqeHfrPx: num(obs.fqe_hfr_px),
                    fqeSigmaLogHfr: num(obs.fqe_sigma_log_hfr),
                    fqeStarCount: num(obs.fqe_star_count),
                    fqeModelVersion: typeof obs.fqe_model_version === "string" ? obs.fqe_model_version : null,
                    fqeCalibrated: typeof obs.fqe_calibrated === "boolean" ? obs.fqe_calibrated : null,
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
                    pointingErrorArcmin: num(obs.pointing_error_arcmin),
                    wcsResidualArcsec: num(obs.wcs_residual_arcsec),
                    observationValid: obs.observation_valid ?? false,
                });
            }
        },
    });

    return observation;
}

/**
 * Pure mapping of the daemon's `SharedObservation` (VISION_UPDATE payload).
 *
 * Absence stays absent: a measurement the daemon did not make is `null` and
 * renders as "—" with its reason, never as 0. A 0 sky or 0 noise would read
 * as a clean, noiseless frame; a 0 HFR as perfect focus.
 *
 * Background provenance follows the daemon's contract
 * (`vision::background_model::BackgroundResult`, schema 1):
 * - `current`: a versioned `background` result is attached;
 * - `legacy_unverified`: only the old top-level numbers exist (observations
 *   from before the contract). A stored 0 there cannot be told apart from an
 *   unmeasured frame, so those numbers are shown as unverified;
 * - `absent`: no background information.
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
    masked_support: number;
    deblended: number;
}

const EMPTY_TALLIES: StarRejectionTallies = {
    saturated: 0,
    edge_truncated: 0,
    low_snr: 0,
    too_few_pixels: 0,
    elongated: 0,
    blended: 0,
    invalid_hfr: 0,
    masked_support: 0,
    deblended: 0,
};

export type BackgroundProvenance = "current" | "legacy_unverified" | "absent";

/** The parts of the daemon's `BackgroundResult` the UI needs. */
export interface BackgroundSummary {
    schemaVersion: number | null;
    /** `legacy_global_clip` or `sep_spatial_mesh`. */
    method: string | null;
    /** What the configuration asked for; differs from `method` on fallback. */
    requestedMethod: string | null;
    libraryVersion: string | null;
    configSha256: string | null;
    /** `measured` | `unavailable`. */
    levelStatus: string | null;
    /** `measured` | `unresolved` | `unavailable`. */
    noiseStatus: string | null;
    reasons: string[];
    backendError: string | null;
    /** Input units: `adu`, `adu_proxy`, or `unverified`. */
    units: string | null;
    noiseSource: string | null;
    /** Frame-level summary of B(x,y); `null` unless measured. */
    level: number | null;
    levelDefinition: string | null;
    /** Frame-level summary of the noise scale; `null` unless measured. */
    rms: number | null;
    rmsDefinition: string | null;
    /** Plane-fit slopes in input units per analysis pixel; `null` when the
     *  method has no spatial model or too few supported meshes. */
    gradientXPerPx: number | null;
    gradientYPerPx: number | null;
    planeResidualRms: number | null;
    saturationEvidence: string | null;
}

export interface SharedObservation {
    timestamp: string;
    /** "Valid" | "EmptyFrame" | "AnalysisUnavailable" | "NoSources" |
     *  "NoUsableStars" | "BackgroundUnavailable" | "DetectionUnavailable". */
    status: string;
    rejectionReasons: string[];
    hfrPx: number | null;
    hfdPx: number | null;
    fwhmPx: number | null;
    imageFwhmArcsec: number | null;
    qualityUncertainty: number | null;
    hfrLogSigma: number | null;
    /** `null` when the daemon reports that detection did not run (a current
     *  payload whose `detection` is null); legacy payloads keep their number. */
    starCount: number | null;
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
    cloudProbability: number | null;
    cloudEvent: string;
    /** Not a measurement when `trailingUnavailableReason` is set. */
    trailingSeverity: number | null;
    trailingUnavailableReason: string | null;
    trailingPolicy: string | null;
    trailingAxis: string | null;
    trailingDirectionPa: number | null;
    /** Authoritative background result, when the daemon attached one. */
    background: BackgroundSummary | null;
    backgroundProvenance: BackgroundProvenance;
    /** Deprecated top-level mirrors (see `background`). */
    backgroundAdu: number | null;
    skyBackgroundAdu: number | null;
    backgroundSigma: number | null;
    /** Legacy pixel-significance heuristic, not aperture/PSF SNR. */
    brightPixelSignificance: number | null;
    saturationFraction: number | null;
    detectionMethod: string | null;
    pointingErrorArcmin: number | null;
    wcsResidualArcsec: number | null;
    observationValid: boolean;
}

const num = (v: unknown): number | null => (typeof v === "number" && isFinite(v) ? v : null);
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
const strings = (v: unknown): string[] =>
    Array.isArray(v) ? v.filter((item): item is string => typeof item === "string") : [];
const record = (v: unknown): Record<string, unknown> | null =>
    v !== null && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;

export function mapBackground(raw: unknown): BackgroundSummary | null {
    const bg = record(raw);
    if (!bg) return null;
    const plane = record(bg.plane);
    const support = record(bg.support);
    const levelStatus = str(bg.level_status);
    const noiseStatus = str(bg.noise_status);
    return {
        schemaVersion: num(bg.schema_version),
        method: str(bg.method),
        requestedMethod: str(bg.requested_method),
        libraryVersion: str(bg.library_version),
        configSha256: str(bg.config_sha256),
        levelStatus,
        noiseStatus,
        reasons: strings(bg.reasons),
        backendError: str(bg.backend_error),
        units: str(bg.units),
        noiseSource: str(bg.noise_source),
        // Values are only trusted with their status, mirroring the daemon's
        // `level()` / `rms()` accessors.
        level: levelStatus === "measured" ? num(bg.level) : null,
        levelDefinition: str(bg.level_definition),
        rms: noiseStatus === "measured" ? num(bg.rms) : null,
        rmsDefinition: str(bg.rms_definition),
        gradientXPerPx: plane ? num(plane.gx_per_px) : null,
        gradientYPerPx: plane ? num(plane.gy_per_px) : null,
        planeResidualRms: plane ? num(plane.residual_rms) : null,
        saturationEvidence: support ? str(support.saturation_evidence) : null,
    };
}

/** Map a raw VISION_UPDATE `observation` object. */
export function mapSharedObservation(raw: unknown): SharedObservation | null {
    const obs = record(raw);
    if (!obs) return null;
    const background = mapBackground(obs.background);
    const backgroundAdu = num(obs.background_adu);
    const backgroundProvenance: BackgroundProvenance = background
        ? "current"
        : backgroundAdu !== null
          ? "legacy_unverified"
          : "absent";
    const detection = record(obs.detection);
    return {
        timestamp: str(obs.timestamp) ?? "",
        status: str(obs.status) ?? "NoSources",
        rejectionReasons: strings(obs.rejection_reasons),
        hfrPx: num(obs.hfr_px),
        hfdPx: num(obs.hfd_px),
        fwhmPx: num(obs.fwhm_px),
        imageFwhmArcsec: num(obs.image_fwhm_arcsec),
        qualityUncertainty: num(obs.quality_uncertainty),
        hfrLogSigma: num(obs.hfr_log_sigma),
        starCount: background && !detection ? null : (num(obs.star_count) ?? 0),
        starCountUsed: num(obs.star_count_used) ?? 0,
        starRejections: { ...EMPTY_TALLIES, ...(record(obs.star_rejections) ?? {}) },
        fqeHfrPx: num(obs.fqe_hfr_px),
        fqeSigmaLogHfr: num(obs.fqe_sigma_log_hfr),
        fqeStarCount: num(obs.fqe_star_count),
        fqeModelVersion: str(obs.fqe_model_version),
        fqeCalibrated: typeof obs.fqe_calibrated === "boolean" ? obs.fqe_calibrated : null,
        cloudProbability: num(obs.cloud_probability),
        cloudEvent: str(obs.cloud_event) ?? "None",
        trailingSeverity: str(obs.trailing_unavailable_reason) ? null : num(obs.trailing_severity),
        trailingUnavailableReason: str(obs.trailing_unavailable_reason),
        trailingPolicy: str(obs.trailing_policy),
        trailingAxis: str(obs.trailing_axis),
        trailingDirectionPa: num(obs.trailing_direction_pa),
        background,
        backgroundProvenance,
        backgroundAdu,
        skyBackgroundAdu: num(obs.sky_background_adu) ?? backgroundAdu,
        backgroundSigma: num(obs.background_sigma),
        brightPixelSignificance: num(obs.snr_proxy_median),
        saturationFraction: num(obs.saturation_fraction),
        detectionMethod: detection ? str(detection.method) : null,
        pointingErrorArcmin: num(obs.pointing_error_arcmin),
        wcsResidualArcsec: num(obs.wcs_residual_arcsec),
        observationValid: obs.observation_valid === true,
    };
}

export interface MeasurementDisplay {
    /** The value with units, or "—". */
    text: string;
    /** Why there is no value, or a provenance caveat; `null` when measured. */
    note: string | null;
}

const fmt = (value: number, units: string | null) =>
    `${value.toFixed(value !== 0 && Math.abs(value) < 10 ? 2 : 1)}${units ? ` ${units}` : ""}`;

/** Display text for the frame's background level. */
export function displayBackgroundLevel(obs: SharedObservation): MeasurementDisplay {
    if (obs.backgroundProvenance === "legacy_unverified" && obs.backgroundAdu !== null) {
        return { text: fmt(obs.backgroundAdu, "ADU"), note: "legacy record, unverified" };
    }
    const bg = obs.background;
    if (!bg) return { text: "—", note: "no background estimate" };
    if (bg.level === null) {
        return { text: "—", note: bg.reasons.join(", ") || bg.backendError || "unavailable" };
    }
    const fallback = bg.reasons.includes("fallback_used") ? `fallback from ${bg.requestedMethod}` : null;
    return { text: fmt(bg.level, bg.units), note: fallback };
}

/** Display text for the frame's background noise scale. */
export function displayBackgroundNoise(obs: SharedObservation): MeasurementDisplay {
    if (obs.backgroundProvenance === "legacy_unverified" && obs.backgroundSigma !== null) {
        return { text: fmt(obs.backgroundSigma, "ADU"), note: "legacy record, unverified" };
    }
    const bg = obs.background;
    if (!bg) return { text: "—", note: "no background estimate" };
    if (bg.rms === null) {
        const why = bg.noiseStatus === "unresolved" ? "noise unresolved" : "unavailable";
        return { text: "—", note: [why, ...bg.reasons].join(", ") };
    }
    return { text: fmt(bg.rms, bg.units), note: null };
}

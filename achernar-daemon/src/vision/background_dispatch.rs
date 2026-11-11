//! Background backend selection and the fallback policy.
//!
//! `legacy_global_clip` is the default. `sep` is an explicit opt-in
//! diagnostic backend: when it cannot produce a result, the configured
//! fallback decides between reporting the background unavailable (default)
//! and running the legacy estimator *labelled as a fallback*. Custom output is
//! never labelled SEP.

use super::background_model::{
    config_sha256, sep_background_config_json, BackgroundAnalysis, BackgroundBackend,
    BackgroundFallback, BackgroundMethod, BackgroundReason, BackgroundResult, BackgroundSettings,
    BackgroundSupport, LevelStatus, NoiseSource, NoiseStatus, BACKGROUND_SCHEMA_VERSION,
    SEP_LIBRARY_VERSION,
};
use super::quality::{background_from_luma, LumaFrame};

/// Why the SEP backend produced no usable analysis.
pub type BackendFailure = (BackgroundReason, String);

/// Run the configured backend on `frame`.
pub fn analyze_background(frame: &LumaFrame, settings: &BackgroundSettings) -> BackgroundAnalysis {
    match settings.backend {
        BackgroundBackend::LegacyGlobalClip => background_from_luma(frame),
        BackgroundBackend::Sep => analyze_with_sep(frame, settings, sep_backend),
    }
}

/// The vendored SEP 1.4.1 backend (`sep_backend::analyze_sep`). Blocking
/// native work serialized by the binding's process-wide mutex; the vision
/// loop calls it from the blocking pool.
fn sep_backend(
    frame: &LumaFrame,
    settings: &BackgroundSettings,
) -> Result<BackgroundAnalysis, BackendFailure> {
    super::sep_backend::analyze_sep(frame, settings)
        .map_err(|failure| (failure.reason, failure.detail))
}

/// Apply the fallback policy around a SEP backend call.
///
/// A SEP result whose background was measured is kept even when detection
/// could not run (unresolved noise, or an extraction error in
/// `catalog_error`): the frame then reports detection unavailable, not a
/// missing background. Only the configured legacy fallback replaces such a
/// result, and then the whole analysis (background and detection) is the
/// labelled legacy method. An `Err` (no SEP background at all) is
/// unavailable by default.
pub fn analyze_with_sep(
    frame: &LumaFrame,
    settings: &BackgroundSettings,
    sep: impl FnOnce(&LumaFrame, &BackgroundSettings) -> Result<BackgroundAnalysis, BackendFailure>,
) -> BackgroundAnalysis {
    let failure = match sep(frame, settings) {
        Ok(analysis) => match &analysis.catalog_error {
            Some(error)
                if analysis.catalog.is_none()
                    && analysis.result.noise_status == NoiseStatus::Measured
                    && settings.fallback == BackgroundFallback::LegacyGlobalClip =>
            {
                (BackgroundReason::BackendFailure, error.clone())
            }
            _ => return analysis,
        },
        Err(failure) => failure,
    };

    match settings.fallback {
        BackgroundFallback::Unavailable => sep_unavailable(frame, settings, failure),
        BackgroundFallback::LegacyGlobalClip => {
            let mut legacy = background_from_luma(frame);
            legacy.result.requested_method = BackgroundMethod::SepSpatialMesh;
            legacy.result.reasons.insert(0, failure.0);
            legacy
                .result
                .reasons
                .insert(0, BackgroundReason::FallbackUsed);
            legacy.result.backend_error = Some(failure.1);
            legacy
        }
    }
}

/// An unavailable SEP result that still records what was asked for.
fn sep_unavailable(
    frame: &LumaFrame,
    settings: &BackgroundSettings,
    (reason, detail): BackendFailure,
) -> BackgroundAnalysis {
    let config = sep_background_config_json(&settings.sep);
    BackgroundAnalysis {
        result: BackgroundResult {
            schema_version: BACKGROUND_SCHEMA_VERSION,
            method: BackgroundMethod::SepSpatialMesh,
            method_version: BackgroundMethod::SepSpatialMesh.method_version(),
            library: Some("sep".to_string()),
            library_version: Some(SEP_LIBRARY_VERSION.to_string()),
            requested_method: BackgroundMethod::SepSpatialMesh,
            config_sha256: config_sha256(&config),
            config,
            level_status: LevelStatus::Unavailable,
            noise_status: NoiseStatus::Unavailable,
            reasons: vec![reason],
            backend_error: Some(detail),
            units: frame
                .analysis_descriptor
                .as_ref()
                .map(|descriptor| descriptor.units.clone())
                .unwrap_or_else(|| "unverified".to_string()),
            coordinate_frame: "analysis_pixel".to_string(),
            width: frame.width as u32,
            height: frame.height as u32,
            noise_source: NoiseSource::Unavailable,
            support: BackgroundSupport {
                total_pixels: frame.pixels.len() as u64,
                usable_pixels: 0,
                clipped_sample_pixels: None,
                masks: frame.mask_provenance(),
                saturation_evidence: frame.saturation_evidence().to_string(),
            },
            level: None,
            level_definition: "SEP global background (unavailable)".to_string(),
            rms: None,
            rms_definition: "SEP global RMS (unavailable)".to_string(),
            mesh: None,
            plane: None,
            plane_unavailable_reason: Some(reason),
            bright_pixel_significance: None,
            saturation_fraction: None,
            refinement: None,
        },
        field: None,
        catalog: None,
        catalog_error: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::background_model::BackgroundField;

    fn frame() -> LumaFrame {
        LumaFrame {
            pixels: (0..32 * 32).map(|i| 100.0 + (i % 3) as f32).collect(),
            width: 32,
            height: 32,
            max_adu: 65535.0,
            saturation_adu: None,
            masks: Default::default(),
            analysis_descriptor: None,
        }
    }

    fn sep_settings(fallback: BackgroundFallback) -> BackgroundSettings {
        BackgroundSettings {
            backend: BackgroundBackend::Sep,
            fallback,
            ..Default::default()
        }
    }

    #[test]
    fn default_backend_is_the_legacy_estimator() {
        let analysis = analyze_background(&frame(), &BackgroundSettings::default());
        assert_eq!(analysis.result.method, BackgroundMethod::LegacyGlobalClip);
        assert_eq!(
            analysis.result.requested_method,
            BackgroundMethod::LegacyGlobalClip
        );
        assert!(!analysis
            .result
            .reasons
            .contains(&BackgroundReason::FallbackUsed));
    }

    #[test]
    fn failed_sep_is_unavailable_by_default_and_never_labelled_legacy() {
        let analysis = analyze_with_sep(
            &frame(),
            &sep_settings(BackgroundFallback::Unavailable),
            |_, _| {
                Err((
                    BackgroundReason::FrameSmallerThanMeshGrid,
                    "32x32 < 2x64".into(),
                ))
            },
        );
        assert_eq!(analysis.result.method, BackgroundMethod::SepSpatialMesh);
        assert_eq!(analysis.result.level_status, LevelStatus::Unavailable);
        assert_eq!(analysis.result.level, None);
        assert_eq!(
            analysis.result.reasons,
            vec![BackgroundReason::FrameSmallerThanMeshGrid]
        );
        assert!(analysis.field.is_none());
        assert_eq!(analysis.result.library_version.as_deref(), Some("1.4.1"));
    }

    #[test]
    fn configured_fallback_runs_legacy_and_says_so() {
        let analysis = analyze_with_sep(
            &frame(),
            &sep_settings(BackgroundFallback::LegacyGlobalClip),
            |_, _| Err((BackgroundReason::BackendFailure, "code 1".into())),
        );
        // The numbers are the legacy method's, labelled as such.
        assert_eq!(analysis.result.method, BackgroundMethod::LegacyGlobalClip);
        assert_eq!(
            analysis.result.requested_method,
            BackgroundMethod::SepSpatialMesh
        );
        assert_eq!(
            &analysis.result.reasons[..2],
            &[
                BackgroundReason::FallbackUsed,
                BackgroundReason::BackendFailure
            ]
        );
        assert_eq!(analysis.result.backend_error.as_deref(), Some("code 1"));
        assert!(analysis.result.level().is_some());
        assert!(matches!(
            analysis.field,
            Some(BackgroundField::Uniform { .. })
        ));
    }

    #[test]
    fn extraction_failure_keeps_the_measured_background_unless_falling_back() {
        // The adapter contract: an extraction error keeps B(x,y) and sets
        // catalog_error. Detection is then unavailable, not the background.
        let measured_without_catalog = |noise_status| {
            let mut ok = background_from_luma(&frame());
            ok.result.method = BackgroundMethod::SepSpatialMesh;
            ok.result.noise_status = noise_status;
            ok.catalog_error = Some("pixstack full".into());
            ok
        };
        let kept = analyze_with_sep(
            &frame(),
            &sep_settings(BackgroundFallback::Unavailable),
            |_, _| Ok(measured_without_catalog(NoiseStatus::Measured)),
        );
        assert_eq!(kept.result.method, BackgroundMethod::SepSpatialMesh);
        assert_eq!(kept.result.level_status, LevelStatus::Measured);
        assert!(kept.catalog.is_none());
        assert_eq!(kept.catalog_error.as_deref(), Some("pixstack full"));

        let fallback = analyze_with_sep(
            &frame(),
            &sep_settings(BackgroundFallback::LegacyGlobalClip),
            |_, _| Ok(measured_without_catalog(NoiseStatus::Measured)),
        );
        assert_eq!(fallback.result.method, BackgroundMethod::LegacyGlobalClip);
        assert_eq!(
            fallback.result.backend_error.as_deref(),
            Some("pixstack full")
        );
        assert!(fallback
            .result
            .reasons
            .contains(&BackgroundReason::FallbackUsed));

        // Unresolved noise is not a backend failure: no fallback.
        let unresolved = analyze_with_sep(
            &frame(),
            &sep_settings(BackgroundFallback::LegacyGlobalClip),
            |_, _| Ok(measured_without_catalog(NoiseStatus::Unresolved)),
        );
        assert_eq!(unresolved.result.method, BackgroundMethod::SepSpatialMesh);
        assert!(!unresolved
            .result
            .reasons
            .contains(&BackgroundReason::FallbackUsed));
    }
}

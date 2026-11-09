//! Opt-in SEP backend: spatial background `B(x,y)`, RMS `R(x,y)` and the
//! source catalog from the vendored SEP 1.4.1 library (`achernar-sep`),
//! mapped onto the versioned contract in `background_model.rs`
//! (BGE-01, BGE-02, BGE-11, BGE-16, BGE-23).
//!
//! What the adapter adds on top of SEP, and why:
//! * A detector mask (bad/invalid/edge, native saturation, non-finite), so
//!   SEP never sees an unusable pixel as data.
//! * Its own mesh-support count. SEP records nothing about which meshes it
//!   replaced, and with no supported mesh at all it returns level 0 / RMS 1
//!   with success (`background.c:filterback`).
//! * Its own noise validity. SEP's `globalrms` falls back to 1.0, and its
//!   bicubic RMS map can be zero or negative between meshes; neither is a
//!   measured noise, so detection does not run on them.
//!
//! Everything numeric comes from SEP's own evaluators (`sep_bkg_array`,
//! `sep_bkg_rmsarray`, `sep_bkg_global(rms)`, `sep_extract`), the same calls
//! the Python reference makes (`sep.Background.back()/rms()`, `sep.extract`).

use achernar_sep::{
    extract, flags, Background, BackgroundParams, ExtractParams, FilterType, Kernel, MeshGrid,
    Noise, SepError,
};

use super::background_model::{
    config_sha256, sep_background_config_json, sep_detection_config_json, BackgroundAnalysis,
    BackgroundField, BackgroundMethod, BackgroundReason, BackgroundResult, BackgroundSettings,
    BackgroundSupport, CatalogSource, CropSkySource, DetectionMethod, DetectionProvenance,
    LevelStatus, MeshSummary, NoiseSource, NoiseStatus, PlaneFit, SepDetectionParams,
    SourceCatalog, BACKGROUND_SCHEMA_VERSION, SEP_LIBRARY_VERSION,
};
use super::quality::{frame_geometry_valid, LumaFrame};

/// Why the SEP backend produced no analysis at all. `detail` carries the SEP
/// error code and message when `reason` is `BackendFailure`.
#[derive(Debug, Clone, PartialEq)]
pub struct SepBackendFailure {
    pub reason: BackgroundReason,
    pub detail: String,
}

impl SepBackendFailure {
    fn new(reason: BackgroundReason, detail: impl Into<String>) -> Self {
        Self {
            reason,
            detail: detail.into(),
        }
    }
}

/// `catalog_error` when detection is skipped because the noise is not
/// measured (zero, negative or non-finite RMS).
pub const NOISE_UNRESOLVED: &str = "noise_unresolved";

/// SEP's `BACK_MINGOODFRAC` (`background.c`): a mesh with fewer usable
/// pixels than this fraction of its actual size is "bad" and replaced.
const BACK_MINGOODFRAC: f64 = 0.5;

/// Fraction of the declared saturation point treated as clipped on legacy
/// frames without an analysis descriptor (as the legacy estimator does).
const LEGACY_SATURATION_FRACTION: f32 = 0.98;

const LEVEL_DEFINITION: &str = "SEP 1.4.1 sep_bkg_global: median (mean of the two middle values \
     for an even count) of the nx*ny mesh levels after bad-mesh replacement and the fw x fh \
     median filter, which replaces a mesh's level and sigma by the window medians where \
     |median - level| >= fthresh, with windows shrunk symmetrically at the grid edges \
     (background.c filterback). A mesh level is backguess's estimate from the mesh's usable \
     pixels after one mean +- 2 sigma clip (backstat) and iterative median +- 3 sigma \
     histogram clipping: 2.5*median - 1.5*mean when |mean - median| < 0.3*sigma, otherwise \
     the clipped median (the mean when sigma is 0). Meshes with < 50% usable pixels take the \
     mean of their nearest supported meshes.";

const RMS_DEFINITION: &str = "SEP 1.4.1 sep_bkg_globalrms: median of the nx*ny mesh sigmas \
     after bad-mesh replacement and the median filter; a mesh sigma is the standard deviation \
     of its clipped histogram (no truncation correction). Reported only when every mesh sigma \
     and the interpolated RMS map on every usable pixel are positive and finite; SEP's \
     substitutes for a non-positive median (median of positive sigmas, or 1.0) are never \
     reported.";

/// Runs the SEP background and, when the noise is measured, SEP extraction
/// on `frame` with `settings.sep` / `settings.detection`.
///
/// `Err` only when no background exists at all (malformed geometry, no
/// usable pixel, frame below the mesh grid, no supported mesh, SEP failure).
/// An extraction failure keeps the background and sets `catalog_error`.
pub fn analyze_sep(
    frame: &LumaFrame,
    settings: &BackgroundSettings,
) -> Result<BackgroundAnalysis, SepBackendFailure> {
    if !frame_geometry_valid(frame) {
        return Err(SepBackendFailure::new(
            BackgroundReason::MalformedGeometry,
            format!(
                "{} pixels for {}x{}, or a mask that is not a 0/1 plane of that size",
                frame.pixels.len(),
                frame.width,
                frame.height
            ),
        ));
    }
    let (width, height) = (frame.width, frame.height);

    // 1 = excluded. Non-finite pixels are excluded here so that SEP never
    // sees them as data; the safe API would otherwise reject the frame.
    let mask: Vec<u8> = frame
        .pixels
        .iter()
        .enumerate()
        .map(|(i, value)| {
            u8::from(
                !frame.measurement_pixel_valid(i)
                    || frame.native_saturation_at(i)
                    || !value.is_finite(),
            )
        })
        .collect();
    let usable_pixels = mask.iter().filter(|&&m| m == 0).count();
    if usable_pixels == 0 {
        return Err(SepBackendFailure::new(
            BackgroundReason::NoUsablePixels,
            "no finite pixel outside the bad/invalid/edge and native-saturation masks",
        ));
    }

    let sep_params = &settings.sep;
    let min_meshes = u64::from(sep_params.min_meshes_per_axis);
    if (width as u64) < min_meshes * u64::from(sep_params.bw)
        || (height as u64) < min_meshes * u64::from(sep_params.bh)
    {
        return Err(SepBackendFailure::new(
            BackgroundReason::FrameSmallerThanMeshGrid,
            format!(
                "{width}x{height} frame holds fewer than {min_meshes} meshes of {}x{} px per axis",
                sep_params.bw, sep_params.bh
            ),
        ));
    }
    let params = BackgroundParams {
        bw: sep_params.bw as usize,
        bh: sep_params.bh as usize,
        fw: sep_params.fw as usize,
        fh: sep_params.fh as usize,
        fthresh: f64::from(sep_params.fthresh),
    };
    if params.bw == 0 || params.bh == 0 {
        return Err(backend_failure(&SepError::InvalidParameter {
            name: "bw/bh",
            reason: "mesh size must be at least 1 px".into(),
        }));
    }
    let grid = MeshGrid::for_frame(width, height, params.bw, params.bh);
    let supported = mesh_support(&mask, width, height, grid);
    let meshes_below_support = supported.iter().filter(|ok| !**ok).count();
    if meshes_below_support == supported.len() {
        return Err(SepBackendFailure::new(
            BackgroundReason::InsufficientSupport,
            format!(
                "all {}x{} meshes have fewer than {:.0}% usable pixels",
                grid.nx,
                grid.ny,
                BACK_MINGOODFRAC * 100.0
            ),
        ));
    }

    let background = Background::estimate(&frame.pixels, width, height, Some(&mask), &params)
        .map_err(|e| backend_failure(&e))?;
    let level_map = background.back_array().map_err(|e| backend_failure(&e))?;
    let rms_map = background.rms_array().map_err(|e| backend_failure(&e))?;

    let mut reasons = Vec::new();
    if meshes_below_support > 0 {
        reasons.push(BackgroundReason::MeshesInterpolated);
    }
    let noise = noise_validity(
        background.global_rms(),
        background.mesh_rms(),
        &rms_map,
        &mask,
    );
    reasons.extend(noise.iter().copied());
    let noise_measured = noise.is_empty();

    let (plane, plane_unavailable_reason) =
        match fit_plane(background.mesh_level(), &supported, grid, width, height) {
            Some(plane) => (Some(plane), None),
            None => (None, Some(BackgroundReason::InsufficientMeshesForPlane)),
        };

    let config = sep_background_config_json(sep_params);
    let result = BackgroundResult {
        schema_version: BACKGROUND_SCHEMA_VERSION,
        method: BackgroundMethod::SepSpatialMesh,
        method_version: BackgroundMethod::SepSpatialMesh.method_version(),
        library: Some("sep".to_string()),
        library_version: Some(SEP_LIBRARY_VERSION.to_string()),
        requested_method: BackgroundMethod::SepSpatialMesh,
        config_sha256: config_sha256(&config),
        config,
        level_status: LevelStatus::Measured,
        noise_status: if noise_measured {
            NoiseStatus::Measured
        } else {
            NoiseStatus::Unresolved
        },
        reasons,
        backend_error: None,
        units: frame
            .analysis_descriptor
            .as_ref()
            .map(|descriptor| descriptor.units.clone())
            .unwrap_or_else(|| "unverified".to_string()),
        coordinate_frame: "analysis_pixel".to_string(),
        width: width as u32,
        height: height as u32,
        noise_source: NoiseSource::EmpiricalMeshRms,
        support: BackgroundSupport {
            total_pixels: frame.pixels.len() as u64,
            usable_pixels: usable_pixels as u64,
            clipped_sample_pixels: None,
            masks: frame.mask_provenance(),
            saturation_evidence: frame.saturation_evidence().to_string(),
        },
        level: Some(background.global_level()),
        level_definition: LEVEL_DEFINITION.to_string(),
        rms: noise_measured.then(|| background.global_rms()),
        rms_definition: RMS_DEFINITION.to_string(),
        mesh: Some(mesh_summary(&background, grid, meshes_below_support)),
        plane,
        plane_unavailable_reason,
        // Legacy-only heuristic; the SEP path has per-source significance.
        bright_pixel_significance: None,
        saturation_fraction: saturation_fraction(frame, &mask),
    };

    let (catalog, catalog_error) = if noise_measured {
        match sep_catalog(frame, settings, &mask, &level_map, &rms_map) {
            Ok(catalog) => (Some(catalog), None),
            Err(error) => (None, Some(error)),
        }
    } else {
        (None, Some(NOISE_UNRESOLVED.to_string()))
    };

    Ok(BackgroundAnalysis {
        result,
        field: Some(BackgroundField::Map {
            level: level_map,
            rms: rms_map,
            width,
            height,
        }),
        catalog,
        catalog_error,
    })
}

fn backend_failure(error: &SepError) -> SepBackendFailure {
    SepBackendFailure::new(BackgroundReason::BackendFailure, error.to_string())
}

/// SEP's bad-mesh rule from `background.c:backstat`, evaluated on the
/// detector mask: mesh `(i, j)` spans its *actual* pixels (the last column
/// and row of meshes are partial), and it is bad iff
/// `(float)npix < (float)(bw_i * h_j * BACK_MINGOODFRAC)`, where `npix`
/// counts unmasked pixels with `pix > -BIG` (the mask already excludes every
/// non-finite pixel). Returns one flag per mesh, row-major, `true` = supported.
fn mesh_support(mask: &[u8], width: usize, height: usize, grid: MeshGrid) -> Vec<bool> {
    let mut usable = vec![0u64; grid.nx * grid.ny];
    for y in 0..height {
        let row = &mask[y * width..(y + 1) * width];
        let mesh_row = (y / grid.bh) * grid.nx;
        for (x, &m) in row.iter().enumerate() {
            if m == 0 {
                usable[mesh_row + x / grid.bw] += 1;
            }
        }
    }
    usable
        .iter()
        .enumerate()
        .map(|(index, &npix)| {
            let (i, j) = (index % grid.nx, index / grid.nx);
            let mesh_w = (width - i * grid.bw).min(grid.bw);
            let mesh_h = (height - j * grid.bh).min(grid.bh);
            !((npix as f32) < ((mesh_w * mesh_h) as f64 * BACK_MINGOODFRAC) as f32)
        })
        .collect()
}

/// Reasons the empirical noise is not a measurement, empty when it is.
///
/// A zero mesh sigma or zero map value is zero *scatter* (a quantized
/// plateau), not zero detector noise. A negative or non-finite value is the
/// natural cubic spline of `bkg_line_flt_internal` overshooting between
/// meshes whose sigmas differ sharply; it cannot serve as a noise map. Never
/// replaced by an epsilon.
fn noise_validity(
    global_rms: f32,
    mesh_rms: &[f32],
    rms_map: &[f32],
    mask: &[u8],
) -> Vec<BackgroundReason> {
    let (mut zero, mut invalid) = (false, false);
    let mut classify = |value: f32| {
        if !value.is_finite() || value < 0.0 {
            invalid = true;
        } else if value == 0.0 {
            zero = true;
        }
    };
    classify(global_rms);
    mesh_rms.iter().copied().for_each(&mut classify);
    rms_map
        .iter()
        .zip(mask)
        .filter(|(_, &m)| m == 0)
        .for_each(|(&r, _)| classify(r));
    let mut reasons = Vec::new();
    if zero {
        reasons.push(BackgroundReason::ZeroEmpiricalScatter);
    }
    if invalid {
        reasons.push(BackgroundReason::InvalidRmsMap);
    }
    reasons
}

/// Linear interpolation between order statistics (numpy's default
/// `percentile` method): rank `q * (n - 1)` of the sorted values.
fn percentile(sorted: &[f32], q: f64) -> f32 {
    let rank = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (rank.floor() as usize, rank.ceil() as usize);
    let t = rank - lo as f64;
    (sorted[lo] as f64 + t * (sorted[hi] as f64 - sorted[lo] as f64)) as f32
}

fn mesh_summary(background: &Background, grid: MeshGrid, below_support: usize) -> MeshSummary {
    let sorted = |values: &[f32]| {
        let mut values = values.to_vec();
        values.sort_by(f32::total_cmp);
        values
    };
    let level = sorted(background.mesh_level());
    let rms = sorted(background.mesh_rms());
    MeshSummary {
        nx: grid.nx as u32,
        ny: grid.ny as u32,
        bw: grid.bw as u32,
        bh: grid.bh as u32,
        level_min: level[0],
        level_p05: percentile(&level, 0.05),
        level_p50: percentile(&level, 0.50),
        level_p95: percentile(&level, 0.95),
        level_max: level[level.len() - 1],
        rms_p05: percentile(&rms, 0.05),
        rms_p50: percentile(&rms, 0.50),
        rms_p95: percentile(&rms, 0.95),
        meshes_below_support: below_support as u32,
    }
}

/// Centre of mesh `index` along one axis in analysis-pixel coordinates
/// (0-based pixel centres): the midpoint of its actual pixel span, so the
/// partial last mesh is centred on the pixels it holds.
fn mesh_centre(index: usize, size: usize, extent: usize) -> f64 {
    let start = index * size;
    let end = ((index + 1) * size).min(extent);
    (start + end - 1) as f64 / 2.0
}

/// Least-squares plane through the filtered levels of the supported meshes
/// (see [`PlaneFit`]); `None` unless the grid is at least 3 x 3 and the
/// supported meshes determine a plane. The median filter preserves a linear
/// gradient (a symmetric window's median of a plane is its centre value, and
/// `filterback` shrinks edge windows symmetrically), so curvature, vignetting
/// or glow show up in the residuals rather than in the slopes alone.
fn fit_plane(
    levels: &[f32],
    supported: &[bool],
    grid: MeshGrid,
    width: usize,
    height: usize,
) -> Option<PlaneFit> {
    if grid.nx < 3 || grid.ny < 3 {
        return None;
    }
    let x0 = (width - 1) as f64 / 2.0;
    let y0 = (height - 1) as f64 / 2.0;
    let points: Vec<(f64, f64, f64)> = (0..grid.nx * grid.ny)
        .filter(|&k| supported[k])
        .map(|k| {
            (
                mesh_centre(k % grid.nx, grid.bw, width) - x0,
                mesh_centre(k / grid.nx, grid.bh, height) - y0,
                levels[k] as f64,
            )
        })
        .collect();
    if points.len() < 3 {
        return None;
    }
    // Normal equations of v = b0 + gx*dx + gy*dy, solved by Cramer's rule.
    let (mut n, mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut sv, mut sxv, mut syv) = (0.0, 0.0, 0.0);
    for &(dx, dy, v) in &points {
        n += 1.0;
        sx += dx;
        sy += dy;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
        sv += v;
        sxv += dx * v;
        syv += dy * v;
    }
    let det3 = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let a = [[n, sx, sy], [sx, sxx, sxy], [sy, sxy, syy]];
    let det = det3(a);
    // Collinear mesh centres (e.g. only one supported row) leave a slope
    // undetermined; compare with the scale of the matrix.
    if !(det.abs() > 1e-9 * n * sxx.max(1.0) * syy.max(1.0)) {
        return None;
    }
    let solve = |column: usize| {
        let mut m = a;
        let rhs = [sv, sxv, syv];
        for (row, value) in rhs.iter().enumerate() {
            m[row][column] = *value;
        }
        det3(m) / det
    };
    let (b0, gx, gy) = (solve(0), solve(1), solve(2));
    let residuals: Vec<f64> = points
        .iter()
        .map(|&(dx, dy, v)| v - (b0 + gx * dx + gy * dy))
        .collect();
    let residual_rms = (residuals.iter().map(|r| r * r).sum::<f64>() / n).sqrt();
    let residual_max_abs = residuals.iter().fold(0.0f64, |m, r| m.max(r.abs()));
    Some(PlaneFit {
        x0: x0 as f32,
        y0: y0 as f32,
        b0: b0 as f32,
        gx_per_px: gx as f32,
        gy_per_px: gy as f32,
        meshes_used: points.len() as u32,
        residual_rms: residual_rms as f32,
        residual_max_abs: residual_max_abs as f32,
    })
}

/// Fraction behind [`BackgroundResult::saturation_fraction`]: `None` without
/// saturation evidence or without any finite pixel. Versioned analysis
/// frames count analysis pixels touched by clipped native support; legacy
/// frames with only a declared saturation point count usable pixels at/above
/// 98% of it, over the usable pixels, as the legacy estimator does.
fn saturation_fraction(frame: &LumaFrame, mask: &[u8]) -> Option<f32> {
    if frame.saturation_evidence() == "unknown" || !frame.pixels.iter().any(|v| v.is_finite()) {
        return None;
    }
    match (frame.analysis_descriptor.is_some(), frame.saturation_adu) {
        (false, Some(saturation)) if frame.saturation_evidence() == "legacy_peak_threshold" => {
            let threshold = saturation * LEGACY_SATURATION_FRACTION;
            let usable = mask.iter().filter(|&&m| m == 0).count();
            let clipped = frame
                .pixels
                .iter()
                .zip(mask)
                .filter(|(v, &m)| m == 0 && **v >= threshold)
                .count();
            Some(clipped as f32 / usable.max(1) as f32)
        }
        _ => Some(
            frame
                .masks
                .native_saturation
                .iter()
                .filter(|&&v| v != 0)
                .count() as f32
                / frame.pixels.len() as f32,
        ),
    }
}

fn extract_params(
    detection: &SepDetectionParams,
    retain_segmentation: bool,
) -> Result<ExtractParams, String> {
    let filter_type = match detection.filter_type.as_str() {
        "conv" => FilterType::Conv,
        "matched" => FilterType::Matched,
        other => return Err(format!("unknown filter_type {other:?} (conv or matched)")),
    };
    // An empty kernel disables filtering, as `filter_kernel=None` in Python.
    let filter = if detection.filter_kernel.is_empty() {
        None
    } else {
        Some(Kernel::from_rows(&detection.filter_kernel).map_err(|e| e.to_string())?)
    };
    Ok(ExtractParams {
        thresh: detection.thresh_sigma,
        minarea: detection.minarea as usize,
        filter,
        filter_type,
        deblend_nthresh: detection.deblend_nthresh as usize,
        deblend_cont: detection.deblend_cont,
        clean: detection.clean,
        clean_param: detection.clean_param,
        pixstack: usize::try_from(detection.pixstack).unwrap_or(usize::MAX),
        sub_object_limit: achernar_sep::DEFAULT_SUB_OBJECT_LIMIT,
        segmentation_map: retain_segmentation,
    })
}

/// `sep_extract` on `I - B(x,y)` with `R(x,y)` as the per-pixel noise.
///
/// Masked pixels of the residual are set to 0. `sep_extract` does the same
/// to its own line buffers (`extract.c:apply_mask_line`: data 0, noise
/// `BIG`) before filtering, so this changes no result; it keeps NaN and
/// clipped values out of the array handed to the library, matching the
/// Python reference (`resid[mask] = 0`).
fn sep_catalog(
    frame: &LumaFrame,
    settings: &BackgroundSettings,
    mask: &[u8],
    level: &[f32],
    rms: &[f32],
) -> Result<SourceCatalog, String> {
    let params = extract_params(&settings.detection, settings.retain_segmentation)?;
    let residual: Vec<f32> = frame
        .pixels
        .iter()
        .zip(level)
        .zip(mask)
        .map(|((&value, &background), &m)| if m != 0 { 0.0 } else { value - background })
        .collect();
    let catalog = extract(
        &residual,
        frame.width,
        frame.height,
        Some(mask),
        Noise::Array(rms),
        &params,
    )
    .map_err(|e| e.to_string())?;

    let objects: Vec<CatalogSource> = catalog
        .objects
        .iter()
        .map(|object| CatalogSource {
            x: object.x as f32,
            y: object.y as f32,
            xmin: object.xmin,
            xmax: object.xmax,
            ymin: object.ymin,
            ymax: object.ymax,
            npix: object.npix as u32,
            flux: object.flux,
            peak_residual: object.peak,
            peak_raw: frame.pixels[object.ypeak * frame.width + object.xpeak],
            a: object.a,
            b: object.b,
            theta: object.theta,
            flags: object.flag,
            deblended: object.has_flag(flags::OBJ_MERGED),
            truncated: object.has_flag(flags::OBJ_TRUNC),
            pixels: object.pixels.clone(),
        })
        .collect();
    let provenance = DetectionProvenance {
        method: DetectionMethod::SepExtract,
        config_sha256: config_sha256(&sep_detection_config_json(&settings.detection)),
        crop_sky_source: CropSkySource::ParentBackgroundMap,
        noise_floor_applied: false,
        objects_detected: objects.len() as u32,
        objects_deblended: objects.iter().filter(|o| o.deblended).count() as u32,
        objects_truncated: objects.iter().filter(|o| o.truncated).count() as u32,
    };
    Ok(SourceCatalog {
        provenance,
        objects,
        segmentation: catalog.segmentation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing::debayer::AnalysisMasks;
    use crate::vision::background_model::BackgroundSettings;
    use crate::vision::quality::background_from_luma;

    /// Deterministic N(0,1) (xorshift64* + Box-Muller).
    struct Rng(u64);

    impl Rng {
        fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let (u1, u2) = (self.uniform(), self.uniform());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    fn sky(
        width: usize,
        height: usize,
        sigma: f64,
        seed: u64,
        level: impl Fn(f64, f64) -> f64,
    ) -> Vec<f32> {
        let mut rng = Rng(seed);
        (0..width * height)
            .map(|i| {
                let (x, y) = ((i % width) as f64, (i / width) as f64);
                (level(x, y) + sigma * rng.normal()) as f32
            })
            .collect()
    }

    fn add_star(pixels: &mut [f32], width: usize, x0: f64, y0: f64, sigma: f64, peak: f64) {
        for (i, value) in pixels.iter_mut().enumerate() {
            let (x, y) = ((i % width) as f64, (i / width) as f64);
            let r2 = (x - x0).powi(2) + (y - y0).powi(2);
            *value += (peak * (-r2 / (2.0 * sigma * sigma)).exp()) as f32;
        }
    }

    fn frame(pixels: Vec<f32>, width: usize, height: usize) -> LumaFrame {
        LumaFrame {
            pixels,
            width,
            height,
            max_adu: 65535.0,
            saturation_adu: None,
            masks: Default::default(),
            analysis_descriptor: None,
        }
    }

    fn settings() -> BackgroundSettings {
        BackgroundSettings::default()
    }

    fn failure(result: Result<BackgroundAnalysis, SepBackendFailure>) -> BackgroundReason {
        result.expect_err("expected a backend failure").reason
    }

    fn map(analysis: &BackgroundAnalysis) -> (&[f32], &[f32]) {
        match analysis.field.as_ref().unwrap() {
            BackgroundField::Map { level, rms, .. } => (level, rms),
            other => panic!("expected a map, got {other:?}"),
        }
    }

    fn column_mean(values: &[f32], width: usize, x: usize) -> f64 {
        let height = values.len() / width;
        (0..height)
            .map(|y| values[y * width + x] as f64)
            .sum::<f64>()
            / height as f64
    }

    #[test]
    fn default_config_hash_matches_the_python_reference() {
        let analysis = analyze_sep(
            &frame(sky(256, 192, 5.0, 1, |_, _| 300.0), 256, 192),
            &settings(),
        )
        .unwrap();
        assert_eq!(
            analysis.result.config_sha256,
            "80eab0fbae4abf9f60f348adb7de3004067a3873cca29f996289e9dc14040b84"
        );
        assert_eq!(analysis.result.method, BackgroundMethod::SepSpatialMesh);
        assert_eq!(analysis.result.library_version.as_deref(), Some("1.4.1"));
        assert_eq!(analysis.result.units, "unverified");
        assert_eq!(analysis.result.noise_source, NoiseSource::EmpiricalMeshRms);
        assert_eq!(
            analysis.catalog.as_ref().unwrap().provenance.config_sha256,
            "4f4499cda7067107c0fe6d64319684c13d1ec62f590b4bee4b346c1459dc99df"
        );
    }

    #[test]
    fn gradient_frame_gives_a_spatial_level_where_legacy_gives_one_number() {
        // 0.5 ADU/px along x: columns 32 and 224 differ by 96 ADU.
        let (width, height) = (256, 256);
        let pixels = sky(width, height, 5.0, 2, |x, _| 1000.0 + 0.5 * x);
        let sep = analyze_sep(&frame(pixels.clone(), width, height), &settings()).unwrap();
        let (level, rms) = map(&sep);
        let difference = column_mean(level, width, 224) - column_mean(level, width, 32);
        assert!(
            (difference - 96.0).abs() < 2.0,
            "map difference {difference}"
        );
        assert!(rms.iter().all(|&r| r > 0.0));
        assert_eq!(sep.result.noise_status, NoiseStatus::Measured);

        // The legacy estimator summarises the same frame with one number.
        let legacy = background_from_luma(&frame(pixels, width, height));
        let field = legacy.field.expect("legacy level");
        assert!(!field.is_spatial());
        let row = 100 * width;
        assert_eq!(field.level_at(row + 32), field.level_at(row + 224));
    }

    #[test]
    fn frame_below_the_mesh_grid_is_reported_not_estimated() {
        // Default 64 px meshes, at least 2 per axis: 127 px is too small.
        let pixels = sky(127, 200, 3.0, 3, |_, _| 50.0);
        assert_eq!(
            failure(analyze_sep(&frame(pixels, 127, 200), &settings())),
            BackgroundReason::FrameSmallerThanMeshGrid
        );
    }

    #[test]
    fn all_masked_frame_has_no_usable_pixels() {
        let (width, height) = (128, 128);
        let mut masked = frame(vec![100.0; width * height], width, height);
        masked.masks = AnalysisMasks {
            bad_pixel: vec![1; width * height],
            ..AnalysisMasks::default()
        };
        assert_eq!(
            failure(analyze_sep(&masked, &settings())),
            BackgroundReason::NoUsablePixels
        );
        let nan = frame(vec![f32::NAN; width * height], width, height);
        assert_eq!(
            failure(analyze_sep(&nan, &settings())),
            BackgroundReason::NoUsablePixels
        );
    }

    #[test]
    fn malformed_geometry_is_rejected() {
        let mut bad = frame(vec![1.0; 100], 11, 10);
        assert_eq!(
            failure(analyze_sep(&bad, &settings())),
            BackgroundReason::MalformedGeometry
        );
        bad = frame(vec![1.0; 110], 11, 10);
        bad.masks.edge = vec![0; 3];
        assert_eq!(
            failure(analyze_sep(&bad, &settings())),
            BackgroundReason::MalformedGeometry
        );
    }

    #[test]
    fn no_supported_mesh_is_insufficient_support_not_seps_placeholder() {
        // Every 64x64 mesh is 75% masked: SEP itself would return level 0 and
        // RMS 1 with success (filterback).
        let (width, height) = (128, 128);
        let mut f = frame(vec![100.0; width * height], width, height);
        f.masks.invalid = (0..width * height)
            .map(|i| u8::from((i % width) % 64 >= 16))
            .collect();
        assert_eq!(
            failure(analyze_sep(&f, &settings())),
            BackgroundReason::InsufficientSupport
        );
    }

    #[test]
    fn constant_frame_has_a_level_and_unresolved_noise() {
        let (width, height) = (192, 160);
        let analysis = analyze_sep(
            &frame(vec![321.0; width * height], width, height),
            &settings(),
        )
        .unwrap();
        let result = &analysis.result;
        assert_eq!(result.level(), Some(321.0));
        assert_eq!(result.level_status, LevelStatus::Measured);
        assert_eq!(result.noise_status, NoiseStatus::Unresolved);
        assert_eq!(result.rms, None, "SEP's fabricated globalrms 1.0 leaked");
        assert!(result
            .reasons
            .contains(&BackgroundReason::ZeroEmpiricalScatter));
        assert!(analysis.catalog.is_none());
        assert_eq!(analysis.catalog_error.as_deref(), Some(NOISE_UNRESOLVED));
        assert!(analysis.field.is_some());
    }

    #[test]
    fn sharp_sigma_ridge_makes_the_rms_spline_negative_and_noise_invalid() {
        // W1 repro: a steep ridge inflates some meshes' clipped sigma; the
        // natural cubic spline of the mesh sigmas overshoots below zero.
        let (width, height) = (512, 512);
        let pixels = sky(width, height, 5.0, 4, |x, _| {
            1000.0 + 800.0 * (-((x - 256.0) / 60.0).powi(2)).exp()
        });
        let analysis = analyze_sep(&frame(pixels, width, height), &settings()).unwrap();
        let (_, rms) = map(&analysis);
        let min = rms.iter().copied().fold(f32::INFINITY, f32::min);
        let non_positive = rms.iter().filter(|&&r| r <= 0.0).count();
        println!("ridge RMS map: min {min}, {non_positive} pixels <= 0");
        assert!(min < 0.0, "expected a negative RMS map, min {min}");
        let result = &analysis.result;
        assert!(result.reasons.contains(&BackgroundReason::InvalidRmsMap));
        assert_eq!(result.noise_status, NoiseStatus::Unresolved);
        assert_eq!(result.rms, None);
        assert!(result.level().is_some());
        assert!(analysis.catalog.is_none());
        assert_eq!(analysis.catalog_error.as_deref(), Some(NOISE_UNRESOLVED));
    }

    #[test]
    fn non_finite_pixels_are_excluded_masked_or_not() {
        let (width, height) = (192, 160);
        let mut pixels = sky(width, height, 4.0, 5, |_, _| 200.0);
        // Unmasked NaN/inf: the adapter's detector mask excludes them.
        pixels[10 * width + 10] = f32::NAN;
        pixels[20 * width + 30] = f32::INFINITY;
        // NaN inside a declared bad-pixel mask.
        pixels[40 * width + 50] = f32::NAN;
        let mut f = frame(pixels, width, height);
        f.masks.bad_pixel = vec![0; width * height];
        f.masks.bad_pixel[40 * width + 50] = 1;
        f.masks.bad_pixel[41 * width + 50] = 1;
        let analysis = analyze_sep(&f, &settings()).unwrap();
        assert_eq!(
            analysis.result.support.usable_pixels as usize,
            width * height - 4
        );
        assert!((analysis.result.level().unwrap() - 200.0).abs() < 1.0);
        assert_eq!(analysis.result.noise_status, NoiseStatus::Measured);
        assert!(analysis.catalog.is_some());
        let (level, rms) = map(&analysis);
        assert!(level.iter().chain(rms).all(|v| v.is_finite()));
    }

    #[test]
    fn star_and_close_double_are_detected_with_the_deblended_flag() {
        let (width, height) = (256, 192);
        let mut pixels = sky(width, height, 5.0, 6, |x, y| 800.0 + 0.1 * x - 0.05 * y);
        add_star(&mut pixels, width, 60.3, 70.7, 1.8, 900.0);
        add_star(&mut pixels, width, 150.0, 100.0, 1.6, 1500.0);
        add_star(&mut pixels, width, 156.0, 101.0, 1.6, 1000.0);
        let mut settings = settings();
        settings.retain_segmentation = true;
        let analysis = analyze_sep(&frame(pixels.clone(), width, height), &settings).unwrap();
        let catalog = analysis.catalog.as_ref().unwrap();
        assert_eq!(catalog.objects.len(), 3, "{:?}", catalog.objects);
        assert_eq!(catalog.provenance.objects_detected, 3);
        assert_eq!(catalog.provenance.objects_deblended, 2);
        assert_eq!(catalog.provenance.method, DetectionMethod::SepExtract);
        assert_eq!(
            catalog.provenance.crop_sky_source,
            CropSkySource::ParentBackgroundMap
        );
        assert!(!catalog.provenance.noise_floor_applied);
        let single = catalog
            .objects
            .iter()
            .find(|o| (o.x - 60.3).abs() < 0.5)
            .expect("isolated star");
        assert!(!single.deblended && !single.truncated);
        assert!((single.y - 70.7).abs() < 0.1);
        let pair: Vec<_> = catalog.objects.iter().filter(|o| o.deblended).collect();
        assert_eq!(pair.len(), 2);
        assert!(pair.iter().all(|o| o.flags & flags::OBJ_MERGED != 0));
        let segmentation = catalog.segmentation.as_ref().unwrap();
        for (i, object) in catalog.objects.iter().enumerate() {
            assert_eq!(object.pixels.len(), object.npix as usize);
            assert!(object
                .pixels
                .iter()
                .all(|&p| segmentation[p as usize] == i as i32 + 1));
        }
        // Without retain_segmentation the map is not kept.
        let lean = analyze_sep(
            &frame(pixels, width, height),
            &BackgroundSettings::default(),
        )
        .unwrap();
        assert!(lean.catalog.unwrap().segmentation.is_none());
    }

    #[test]
    fn peak_raw_is_the_input_value_at_seps_unfiltered_peak() {
        let (width, height) = (192, 160);
        let mut pixels = sky(width, height, 3.0, 7, |_, _| 400.0);
        add_star(&mut pixels, width, 96.0, 80.0, 1.5, 2000.0);
        let analysis = analyze_sep(&frame(pixels.clone(), width, height), &settings()).unwrap();
        let star = &analysis.catalog.as_ref().unwrap().objects[0];
        assert_eq!(star.peak_raw, pixels[80 * width + 96]);
        let (level, _) = map(&analysis);
        assert!(
            (star.peak_residual - (pixels[80 * width + 96] - level[80 * width + 96])).abs() < 1e-3
        );
    }

    #[test]
    fn mesh_support_follows_seps_rule_including_partial_meshes() {
        // 200 px wide with 64 px meshes: the last column of meshes is 8 px.
        let (width, height) = (200, 128);
        let mut pixels = vec![0.0f32; width * height];
        let mut rng = Rng(8);
        for (i, value) in pixels.iter_mut().enumerate() {
            let mesh = (i % width) / 64 + 4 * ((i / width) / 64);
            *value = 100.0 * (mesh + 1) as f32 + rng.normal() as f32;
        }
        let mut f = frame(pixels, width, height);
        f.masks.invalid = vec![0; width * height];
        for y in 0..64 {
            // Partial mesh (3, 0): 8x64 px, mask 5 of its 8 columns -> bad.
            for x in 195..200 {
                f.masks.invalid[y * width + x] = 1;
            }
            // Mesh (1, 0): mask exactly half -> still supported ("<" rule).
            for x in 64..96 {
                f.masks.invalid[y * width + x] = 1;
            }
        }
        let mut settings = settings();
        settings.sep.fw = 1;
        settings.sep.fh = 1;
        let analysis = analyze_sep(&f, &settings).unwrap();
        let mesh = analysis.result.mesh.as_ref().unwrap();
        assert_eq!((mesh.nx, mesh.ny), (4, 2));
        assert_eq!(mesh.meshes_below_support, 1);
        assert!(analysis
            .result
            .reasons
            .contains(&BackgroundReason::MeshesInterpolated));
        // Unfiltered: SEP gave the bad mesh the mean of its nearest supported
        // meshes, (2,0) and (3,1), whose levels are ~300 and ~800.
        let grid = MeshGrid::for_frame(width, height, 64, 64);
        let mask: Vec<u8> = f.masks.invalid.clone();
        let supported = mesh_support(&mask, width, height, grid);
        assert_eq!(
            supported,
            vec![true, true, true, false, true, true, true, true]
        );
        let background = Background::estimate(
            &f.pixels,
            width,
            height,
            Some(&mask),
            &BackgroundParams {
                bw: 64,
                bh: 64,
                fw: 1,
                fh: 1,
                fthresh: 0.0,
            },
        )
        .unwrap();
        let levels = background.mesh_level();
        assert!((levels[3] - (levels[2] + levels[7]) / 2.0).abs() < 1e-3);
    }

    #[test]
    fn plane_fit_recovers_a_linear_gradient_and_flags_curvature() {
        // 512x384 with 64 px meshes: 8x6 grid. Mesh levels are mode
        // estimates over 4096 px with sigma 5, i.e. ~0.1 ADU each; over the
        // grid's spread of mesh centres (sum of dx^2 ~ 1.4e6 px^2) that is a
        // slope uncertainty ~1e-4 ADU/px, ~3e-4 allowing for the 3x3
        // filter's correlation. 2e-3 ADU/px is ~7 of those.
        let (width, height) = (512, 384);
        let (x0, y0) = ((width - 1) as f64 / 2.0, (height - 1) as f64 / 2.0);
        let (gx, gy) = (0.3, -0.2);
        let linear = sky(width, height, 5.0, 9, |x, y| {
            500.0 + gx * (x - x0) + gy * (y - y0)
        });
        let analysis = analyze_sep(&frame(linear, width, height), &settings()).unwrap();
        let plane = analysis.result.plane.as_ref().expect("plane fit");
        println!("linear field plane: {plane:?}");
        assert_eq!(plane.meshes_used, 48);
        assert_eq!((plane.x0, plane.y0), (x0 as f32, y0 as f32));
        assert!(
            (plane.gx_per_px as f64 - gx).abs() < 2e-3,
            "gx {}",
            plane.gx_per_px
        );
        assert!(
            (plane.gy_per_px as f64 - gy).abs() < 2e-3,
            "gy {}",
            plane.gy_per_px
        );
        assert!((plane.b0 - 500.0).abs() < 0.5, "b0 {}", plane.b0);
        assert!(
            plane.residual_rms < 0.5,
            "linear residual {}",
            plane.residual_rms
        );

        // Same slopes plus curvature 0.002 ADU/px^2 in x: ~65 ADU at the edges.
        let curved = sky(width, height, 5.0, 10, |x, y| {
            500.0 + gx * (x - x0) + gy * (y - y0) + 0.002 * (x - x0).powi(2)
        });
        let curved = analyze_sep(&frame(curved, width, height), &settings()).unwrap();
        let curved_plane = curved.result.plane.as_ref().unwrap();
        println!("curved field plane: {curved_plane:?}");
        assert!(
            curved_plane.residual_rms > 20.0 * plane.residual_rms.max(0.05),
            "curvature residual {} vs linear {}",
            curved_plane.residual_rms,
            plane.residual_rms
        );
        assert!(curved_plane.residual_max_abs > 10.0);
    }

    #[test]
    fn plane_needs_a_three_by_three_grid() {
        let (width, height) = (256, 128);
        let analysis = analyze_sep(
            &frame(sky(width, height, 5.0, 11, |_, _| 100.0), width, height),
            &settings(),
        )
        .unwrap();
        assert!(analysis.result.plane.is_none());
        assert_eq!(
            analysis.result.plane_unavailable_reason,
            Some(BackgroundReason::InsufficientMeshesForPlane)
        );
    }

    #[test]
    fn saturation_fraction_needs_evidence() {
        let (width, height) = (192, 160);
        let mut pixels = sky(width, height, 3.0, 12, |_, _| 100.0);
        for value in pixels.iter_mut().take(width) {
            *value = 1000.0;
        }
        let unknown = analyze_sep(&frame(pixels.clone(), width, height), &settings()).unwrap();
        assert_eq!(unknown.result.support.saturation_evidence, "unknown");
        assert_eq!(unknown.result.saturation_fraction, None);

        let mut declared = frame(pixels, width, height);
        declared.saturation_adu = Some(1000.0);
        let declared = analyze_sep(&declared, &settings()).unwrap();
        assert_eq!(
            declared.result.support.saturation_evidence,
            "legacy_peak_threshold"
        );
        let fraction = declared.result.saturation_fraction.unwrap();
        assert!((fraction - width as f32 / (width * height) as f32).abs() < 1e-6);
    }

    #[test]
    fn extraction_failure_keeps_the_background() {
        let (width, height) = (192, 160);
        let mut pixels = sky(width, height, 3.0, 13, |_, _| 100.0);
        add_star(&mut pixels, width, 96.0, 80.0, 4.0, 3000.0);
        let mut settings = settings();
        settings.detection.pixstack = 10;
        let analysis = analyze_sep(&frame(pixels, width, height), &settings).unwrap();
        assert_eq!(analysis.result.noise_status, NoiseStatus::Measured);
        assert!(analysis.result.level().is_some() && analysis.result.rms().is_some());
        assert!(analysis.catalog.is_none());
        let error = analysis.catalog_error.unwrap();
        assert!(error.contains("internal pixel buffer full"), "{error}");

        settings.detection.pixstack = 300_000;
        settings.detection.filter_type = "gaussian".into();
        let analysis = analyze_sep(
            &frame(sky(width, height, 3.0, 13, |_, _| 100.0), width, height),
            &settings,
        )
        .unwrap();
        assert!(analysis.catalog_error.unwrap().contains("filter_type"));
    }

    /// The adapter on the crate's Python-reference fixture (sep 1.4.1 wheel,
    /// `achernar-sep/tests/fixtures/parity_v1`): the detector mask (a bad-pixel
    /// mask with NaNs inside it), the residual construction and the catalog
    /// mapping reproduce the wheel within decisions.md D-006.
    #[test]
    fn adapter_reproduces_the_python_reference_fixture() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/sep/tests/fixtures/parity_v1/");
        let read = |name: &str| std::fs::read(format!("{dir}{name}")).unwrap();
        let f32s = |name: &str| -> Vec<f32> {
            read(name)
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };
        let (width, height) = (192, 160);
        let mut f = frame(f32s("input.f32"), width, height);
        f.masks.bad_pixel = read("mask.u8");
        let mut settings = settings();
        settings.sep.bw = 32;
        settings.sep.bh = 32;
        settings.retain_segmentation = true;
        let analysis = analyze_sep(&f, &settings).unwrap();

        let (level, rms) = map(&analysis);
        let within = |ours: &[f32], reference: &[f32]| {
            ours.iter()
                .zip(reference)
                .all(|(o, r)| (o - r).abs() as f64 <= 1e-5 * (r.abs() as f64).max(1.0))
        };
        assert!(within(level, &f32s("back.f32")), "background map");
        assert!(within(rms, &f32s("rms.f32")), "RMS map");
        let globals = String::from_utf8(read("globals.tsv")).unwrap();
        let global = |key: &str| -> f64 {
            globals
                .lines()
                .find_map(|line| line.strip_prefix(&format!("{key}\t")))
                .unwrap()
                .parse()
                .unwrap()
        };
        let result = &analysis.result;
        assert!((result.level().unwrap() as f64 - global("globalback")).abs() <= 1e-5 * 1052.2);
        assert!((result.rms().unwrap() as f64 - global("globalrms")).abs() <= 1e-5 * 12.5);
        // Mask rectangle x 130..170, y 60..90 leaves mesh (4, 2) 76% masked.
        assert_eq!(result.mesh.as_ref().unwrap().meshes_below_support, 1);

        let catalog = analysis.catalog.as_ref().unwrap();
        let text = String::from_utf8(read("catalog.tsv")).unwrap();
        let mut lines = text.lines();
        let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
        let rows: Vec<std::collections::HashMap<&str, f64>> = lines
            .map(|line| {
                header
                    .iter()
                    .copied()
                    .zip(line.split('\t').map(|v| v.parse::<f64>().unwrap()))
                    .collect()
            })
            .collect();
        assert_eq!(catalog.objects.len(), rows.len());
        for (ours, r) in catalog.objects.iter().zip(&rows) {
            assert_eq!(ours.npix as f64, r["npix"]);
            assert_eq!(
                [ours.xmin, ours.xmax, ours.ymin, ours.ymax].map(|v| v as f64),
                [r["xmin"], r["xmax"], r["ymin"], r["ymax"]]
            );
            assert_eq!(ours.flags as f64, r["flag"]);
            assert_eq!(ours.deblended, r["flag"] as u16 & flags::OBJ_MERGED != 0);
            assert_eq!(ours.truncated, r["flag"] as u16 & flags::OBJ_TRUNC != 0);
            assert!(
                (ours.x as f64 - r["x"]).abs() <= 1e-4 && (ours.y as f64 - r["y"]).abs() <= 1e-4
            );
            for (value, key) in [(ours.flux, "flux"), (ours.a, "a"), (ours.b, "b")] {
                assert!(
                    (value as f64 - r[key]).abs() <= 1e-5 * r[key].abs().max(1.0),
                    "{key}"
                );
            }
            assert!((ours.theta as f64 - r["theta"]).abs() <= 1e-5);
            assert_eq!(ours.peak_residual as f64, r["peak"]);
            let peak = r["ypeak"] as usize * width + r["xpeak"] as usize;
            assert_eq!(ours.peak_raw, f.pixels[peak]);
        }
        let segmap: Vec<i32> = read("segmap.i32")
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert_eq!(catalog.segmentation.as_ref().unwrap(), &segmap);
    }

    /// Available-host software timing, not a deployment target. Run with
    /// `cargo test --release -- --ignored sep_backend_host_timing --nocapture`
    /// (peak RSS from the OS, e.g. `/usr/bin/time -l` on macOS).
    #[test]
    #[ignore]
    fn sep_backend_host_timing() {
        let (width, height) = (6000, 4000);
        let mut pixels = sky(width, height, 8.0, 14, |x, y| {
            1000.0 + 0.01 * x - 0.02 * y + 1e-6 * (x - 3000.0).powi(2)
        });
        let mut rng = Rng(15);
        for _ in 0..2000 {
            let (x, y) = (rng.uniform() * width as f64, rng.uniform() * height as f64);
            let peak = 200.0 + 3000.0 * rng.uniform();
            let (x0, y0) = (x as usize, y as usize);
            for yy in y0.saturating_sub(8)..(y0 + 9).min(height) {
                for xx in x0.saturating_sub(8)..(x0 + 9).min(width) {
                    let r2 = (xx as f64 - x).powi(2) + (yy as f64 - y).powi(2);
                    pixels[yy * width + xx] += (peak * (-r2 / (2.0 * 2.0 * 2.0)).exp()) as f32;
                }
            }
        }
        let frame = frame(pixels, width, height);
        let settings = settings();
        let _ = analyze_sep(&frame, &settings).unwrap();
        let runs = 3;
        let start = std::time::Instant::now();
        let mut objects = 0;
        for _ in 0..runs {
            let analysis = analyze_sep(&frame, &settings).unwrap();
            objects = analysis.catalog.map_or(0, |c| c.objects.len());
        }
        let per_run = start.elapsed().as_secs_f64() / runs as f64;
        println!(
            "available host software timing (not a deployment target): analyze_sep on \
             {width}x{height} f32, default settings, {objects} objects: {:.1} ms per frame \
             (mean of {runs})",
            per_run * 1e3
        );
    }
}

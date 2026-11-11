//! Opt-in SEP backend: spatial background `B(x,y)`, RMS `R(x,y)` and the
//! source catalog from the vendored SEP 1.4.1 library (`achernar-sep`),
//! mapped onto the versioned contract in `background_model.rs`
//! (BGE-01, BGE-02, BGE-11, BGE-16, BGE-23).
//!
//! What the adapter adds on top of SEP, and why:
//! * A detector mask (bad/invalid/edge, native saturation, non-finite or
//!   beyond SEP's 1e30 sentinel), so SEP never sees an unusable pixel as
//!   data.
//! * Its own mesh-support count. SEP records nothing about which meshes it
//!   replaced, and with no supported mesh at all it returns level 0 / RMS 1
//!   with success (`background.c:filterback`).
//! * Its own noise validity. SEP's `globalrms` falls back to 1.0, and its
//!   bicubic RMS map can undershoot far below every mesh sigma it
//!   interpolates (even below zero); neither is a measured noise, so
//!   detection does not run on them (see [`RMS_MAP_LOWER_BOUND_FRACTION`]).
//!
//! Two noise policies (`[vision.background.sep] noise_policy`):
//! * `frame_wide_v1` (decision D-018): noise validity is decided for the
//!   whole frame; one bad pixel of the RMS map withholds the catalog
//!   everywhere.
//! * `masked_local_v2` (default, `docs/bge-v2-spec.md` §3): a source mask
//!   from a low-threshold extraction on pass 1 feeds a second SEP background;
//!   the RMS map is then judged per pixel against the smallest positive mesh
//!   sigma of the pixel's own spline cell. Up to `rms_max_invalid_fraction`
//!   of locally invalid pixels are masked out of detection and star support
//!   instead of withholding the catalog.
//!
//! Everything numeric comes from SEP's own evaluators (`sep_bkg_array`,
//! `sep_bkg_rmsarray`, `sep_bkg_global(rms)`, `sep_extract`), the same calls
//! the Python reference makes (`sep.Background.back()/rms()`, `sep.extract`).

use achernar_sep::{
    extract, flags, mask_ellipses, mesh_support, usable_data_value, Background, BackgroundParams,
    Ellipse, ExtractParams, FilterType, Kernel, MeshGrid, Noise, SepError,
};

use super::background_model::{
    config_sha256, sep_background_config_json, sep_detection_config_json, BackgroundAnalysis,
    BackgroundField, BackgroundMethod, BackgroundReason, BackgroundResult, BackgroundSettings,
    BackgroundSupport, CatalogSource, CropSkySource, DetectionMethod, DetectionProvenance,
    LevelStatus, MeshSummary, NoisePolicy, NoiseRefinement, NoiseSource, NoiseStatus,
    Pass2SkipReason, PlaneFit, SepDetectionParams, SourceCatalog, BACKGROUND_SCHEMA_VERSION,
    MASK_DEBLEND_CONT, SEP_LIBRARY_VERSION,
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
/// measured (zero scatter or an invalid RMS map).
pub const NOISE_UNRESOLVED: &str = "noise_unresolved";

/// Decision D-018 (an engineering bound, not from the literature): on a
/// usable pixel, `R(x,y)` below this fraction of the smallest strictly
/// positive filtered mesh sigma makes the RMS map invalid. The map is a
/// natural bicubic spline of the mesh sigmas (`bkg_line_flt_internal`); where
/// neighbouring sigmas change sharply it undershoots, and an undershoot that
/// stays positive (review B1: a 600 ADU ridge gives R = 0.55 against mesh
/// sigmas >= 4.87) turns noise into detections as surely as a negative one.
/// No measured mesh supports a value below half the smallest one.
pub const RMS_MAP_LOWER_BOUND_FRACTION: f32 = 0.5;

/// What the adapter found in SEP's RMS map on usable pixels; logged when the
/// map is rejected so the frame can be diagnosed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RmsMapCheck {
    /// Smallest finite `R(x,y)` over usable pixels.
    pub usable_min: Option<f32>,
    /// Smallest strictly positive filtered mesh sigma.
    pub mesh_sigma_min_positive: Option<f32>,
    /// `RMS_MAP_LOWER_BOUND_FRACTION * mesh_sigma_min_positive`.
    pub lower_bound: Option<f32>,
    /// Usable pixels with `0 < R < lower_bound`.
    pub below_bound: u64,
    /// Usable pixels with `R == 0`.
    pub zero: u64,
    /// Usable pixels with `R < 0` or non-finite `R`.
    pub negative_or_non_finite: u64,
}

/// Fraction of the declared saturation point treated as clipped on legacy
/// frames without an analysis descriptor (as the legacy estimator does).
const LEGACY_SATURATION_FRACTION: f32 = 0.98;

const LEVEL_DEFINITION: &str = "SEP 1.4.1 sep_bkg_global: median (mean of the two middle values \
     for an even count) of the nx*ny mesh levels after bad-mesh replacement and the fw x fh \
     median filter, which replaces a mesh's level and sigma by the window medians where \
     |median - level| >= fthresh, with windows shrunk symmetrically at the grid edges \
     (background.c filterback). A mesh level is backguess's estimate from a histogram of the \
     mesh's usable pixels: backstat takes their mean and sigma, recomputes both over the \
     pixels within that mean +- 2 sigma, and uses the recomputed values only to place the \
     histogram (qzero = mean - 5 sigma, width 10 sigma, at most 4096 bins); backhisto bins \
     every usable pixel inside that span; backguess clips the histogram iteratively at median \
     +- 3 sigma (up to 100 passes) and returns 2.5*median - 1.5*mean when |mean - median| < \
     0.3*sigma, otherwise the clipped median (the mean when sigma is 0). Meshes with < 50% \
     usable pixels take the mean of their nearest supported meshes.";

const RMS_DEFINITION: &str = "SEP 1.4.1 sep_bkg_globalrms: median of the nx*ny mesh sigmas \
     after bad-mesh replacement and the median filter; a mesh sigma is the standard deviation \
     of its clipped histogram (no truncation correction). Reported only when every mesh sigma \
     is positive and finite and, on every usable pixel, the interpolated RMS map R(x,y) is \
     finite and at least 0.5 x the smallest positive filtered mesh sigma (decision D-018; \
     zero R is zero scatter, negative, non-finite or lower R an invalid RMS map); SEP's \
     substitutes for a non-positive median (median of positive sigmas, or 1.0) are never \
     reported.";

const LEVEL_DEFINITION_V2_SUFFIX: &str = " masked_local_v2: when refinement.pass2_used, the \
     meshes are those of a second sep_background whose mask adds a source mask (a pass-1 \
     extraction at mask_thresh_sigma against max(R1, cell floor), its footprints and its \
     objects' SEP ellipses scaled by mask_ellipse_scale).";

const RMS_DEFINITION_V2: &str = "SEP 1.4.1 sep_bkg_globalrms: median of the nx*ny mesh sigmas \
     after bad-mesh replacement and the median filter (of the second, source-masked pass when \
     refinement.pass2_used); a mesh sigma is the standard deviation of its clipped histogram (no \
     truncation correction). masked_local_v2 (docs/bge-v2-spec.md 3.5): reported unless a mesh \
     sigma or the global value is zero or not finite and positive, a usable pixel has R = 0 \
     (zero scatter), or more than rms_max_invalid_fraction of the usable pixels are locally \
     invalid: R not finite, R < 0, or 0 < R < rms_local_fraction x the smallest finite positive \
     mesh sigma among the (up to four) spline nodes of the pixel's cell (the frame's smallest \
     positive mesh sigma when the cell has none). Locally invalid pixels are excluded from \
     detection and from star measurement support.";

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

    // 1 = excluded. Pixels SEP cannot use as data (non-finite, or at/beyond
    // its 1e30 BIG sentinel, e.g. a -1e30 blank value) are excluded here
    // with the safe API's own predicate, so one such pixel masks itself
    // instead of failing the frame.
    let mask: Vec<u8> = frame
        .pixels
        .iter()
        .enumerate()
        .map(|(i, value)| {
            u8::from(
                !frame.measurement_pixel_valid(i)
                    || frame.native_saturation_at(i)
                    || !usable_data_value(*value),
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
        fthresh: sep_params.fthresh,
    };
    if params.bw == 0 || params.bh == 0 {
        return Err(backend_failure(&SepError::InvalidParameter {
            name: "bw/bh",
            reason: "mesh size must be at least 1 px".into(),
        }));
    }
    let grid = MeshGrid::for_frame(width, height, params.bw, params.bh);
    // SEP's bad-mesh rule (background.c:backstat, BACK_MINGOODFRAC 0.5 of
    // each mesh's actual, possibly partial, size), from the safe API.
    let supported = mesh_support(Some(&mask), width, height, grid);
    let meshes_below_support = supported.iter().filter(|ok| !**ok).count();
    if meshes_below_support == supported.len() {
        return Err(SepBackendFailure::new(
            BackgroundReason::InsufficientSupport,
            format!(
                "all {}x{} meshes have fewer than 50% usable pixels",
                grid.nx, grid.ny
            ),
        ));
    }

    // Frames whose bad-mesh search would exceed achernar_sep's
    // MAX_BAD_MESH_WORK fail here with "work bound: ..." before SEP runs.
    let pass1 = Background::estimate(&frame.pixels, width, height, Some(&mask), &params)
        .map_err(|e| backend_failure(&e))?;
    let level1 = pass1.back_array().map_err(|e| backend_failure(&e))?;
    let rms1 = pass1.rms_array().map_err(|e| backend_failure(&e))?;

    let maps = match sep_params.noise_policy {
        NoisePolicy::FrameWideV1 => {
            frame_wide_v1(pass1, level1, rms1, &mask, supported, meshes_below_support)
        }
        NoisePolicy::MaskedLocalV2 => masked_local_v2(
            frame,
            settings,
            &mask,
            usable_pixels,
            &params,
            grid,
            Pass {
                background: pass1,
                level: level1,
                rms: rms1,
                supported,
                meshes_below_support,
            },
        )?,
    };
    let NoiseMaps {
        pass:
            Pass {
                background,
                level: level_map,
                rms: rms_map,
                supported,
                meshes_below_support,
            },
        skip_reason,
        noise,
        invalid,
        floor,
        refinement,
    } = maps;

    let mut reasons = Vec::new();
    if meshes_below_support > 0 {
        reasons.push(BackgroundReason::MeshesInterpolated);
    }
    reasons.extend(skip_reason);
    reasons.extend(noise.iter().copied());
    let noise_measured = !noise.iter().any(|reason| {
        matches!(
            reason,
            BackgroundReason::ZeroEmpiricalScatter | BackgroundReason::InvalidRmsMap
        )
    });

    let (plane, plane_unavailable_reason) =
        match fit_plane(background.mesh_level(), &supported, grid, width, height) {
            Some(plane) => (Some(plane), None),
            None => (None, Some(BackgroundReason::InsufficientMeshesForPlane)),
        };
    let v2 = sep_params.noise_policy == NoisePolicy::MaskedLocalV2;

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
        level_definition: if v2 {
            format!("{LEVEL_DEFINITION}{LEVEL_DEFINITION_V2_SUFFIX}")
        } else {
            LEVEL_DEFINITION.to_string()
        },
        rms: noise_measured.then(|| background.global_rms()),
        rms_definition: if v2 {
            RMS_DEFINITION_V2
        } else {
            RMS_DEFINITION
        }
        .to_string(),
        mesh: Some(mesh_summary(&background, grid, meshes_below_support)),
        plane,
        plane_unavailable_reason,
        // Legacy-only heuristic; the SEP path has per-source significance.
        bright_pixel_significance: None,
        saturation_fraction: saturation_fraction(frame, &mask),
        refinement,
    };

    let (catalog, catalog_error) = if noise_measured {
        // M_det, plus the locally invalid pixels under masked_local_v2
        // (never the source mask: sources must be detected).
        let detection_mask: std::borrow::Cow<[u8]> = match &invalid {
            Some(invalid) => mask
                .iter()
                .zip(invalid)
                .map(|(&m, &i)| m | i)
                .collect::<Vec<u8>>()
                .into(),
            None => (&mask[..]).into(),
        };
        match sep_catalog(
            frame,
            settings,
            &detection_mask,
            &level_map,
            &rms_map,
            floor.as_ref(),
        ) {
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
            invalid_noise: invalid,
        }),
        catalog,
        catalog_error,
    })
}

/// The noise array of the final masked_local_v2 extraction (amendment A-1,
/// `docs/bge-v2-spec.md` §3.6): `R`, with every masked pixel (`M_det ∪
/// M_inv`) set to its cell floor. The reported `R` map is not changed.
///
/// SEP's `apply_mask_line` gives every masked pixel the noise `BIG` except
/// those of the first `convh/2` rows: `arraybuffer_init` reads them before
/// the main loop, so the mask is never applied there (`extract.c`). Under
/// frame_wide_v1 every masked `R` was a positive spline value; under
/// masked_local_v2 a masked pixel can carry a negative `R`, its threshold
/// `thresh * R` turns negative, and row 0 yields zero-flux "objects" made
/// only of masked pixels (x = y = NaN, `OBJ_TRUNC`; seen on the 800 ADU ridge
/// fixture). The floor is positive, so that cannot happen.
fn noise_for_masked_extraction(
    rms: &[f32],
    mask: &[u8],
    floor: &CellFloor,
    width: usize,
) -> Vec<f32> {
    rms.iter()
        .zip(mask)
        .enumerate()
        .map(|(i, (&r, &m))| {
            if m != 0 {
                floor.at(i % width, i / width)
            } else {
                r
            }
        })
        .collect()
}

/// One SEP background pass and the maps evaluated from it.
struct Pass {
    background: Background,
    level: Vec<f32>,
    rms: Vec<f32>,
    /// SEP's mesh-support flags under the mask this pass ran with.
    supported: Vec<bool>,
    meshes_below_support: usize,
}

/// The maps a frame reports and detects with, and what the noise policy
/// concluded about them.
struct NoiseMaps {
    pass: Pass,
    /// `masked_local_v2`: why pass 2 was skipped, as a result reason.
    skip_reason: Option<BackgroundReason>,
    /// Noise reasons; the noise is measured unless one of them is
    /// `zero_empirical_scatter` or `invalid_rms_map`.
    noise: Vec<BackgroundReason>,
    /// `M_inv` (1 = locally invalid usable pixel), when any exists.
    invalid: Option<Vec<u8>>,
    /// masked_local_v2: the cell floor of the maps used (A-1 extraction
    /// noise on masked pixels).
    floor: Option<CellFloor>,
    refinement: Option<NoiseRefinement>,
}

fn log_rejected_map(noise: &[BackgroundReason], check: &RmsMapCheck) {
    if noise.contains(&BackgroundReason::InvalidRmsMap) {
        tracing::warn!(
            usable_min = ?check.usable_min,
            mesh_sigma_min_positive = ?check.mesh_sigma_min_positive,
            lower_bound = ?check.lower_bound,
            below_bound = check.below_bound,
            zero = check.zero,
            negative_or_non_finite = check.negative_or_non_finite,
            "SEP RMS map rejected (invalid_rms_map); detection not run"
        );
    }
}

/// Decision D-018 unchanged: the pass-1 maps and the frame-wide validity.
fn frame_wide_v1(
    background: Background,
    level: Vec<f32>,
    rms: Vec<f32>,
    mask: &[u8],
    supported: Vec<bool>,
    meshes_below_support: usize,
) -> NoiseMaps {
    let (noise, check) = noise_validity(background.global_rms(), background.mesh_rms(), &rms, mask);
    log_rejected_map(&noise, &check);
    NoiseMaps {
        pass: Pass {
            background,
            level,
            rms,
            supported,
            meshes_below_support,
        },
        skip_reason: None,
        noise,
        invalid: None,
        floor: None,
        refinement: None,
    }
}

/// The local reference of `masked_local_v2` (`docs/bge-v2-spec.md` §3.2):
/// for each pixel, the smallest finite strictly positive filtered mesh sigma
/// among the spline nodes of its cell, or the frame's smallest one when the
/// cell has none.
///
/// SEP's spline nodes sit at `(j + 0.5) * bw` (`bkg_line_flt_internal`), so
/// pixel `x` lies between nodes `xl = (2x - bw) div 2bw` and `xl + 1` (0 for
/// `2x < bw`), clamped to `[0, max(nx - 2, 0)]`; a single-node axis has
/// one node.
#[derive(Debug, Clone)]
pub struct CellFloor {
    col: Vec<u32>,
    row: Vec<u32>,
    cells_x: usize,
    cells: Vec<f32>,
}

/// The lower node of the spline cell holding `coord` along one axis.
pub fn cell_index(coord: usize, size: usize, nodes: usize) -> usize {
    let lower = if 2 * coord < size {
        0
    } else {
        (2 * coord - size) / (2 * size)
    };
    lower.min(nodes.saturating_sub(2))
}

impl CellFloor {
    /// `None` when no mesh sigma is finite and strictly positive: the floor
    /// is undefined and the noise unresolved.
    pub fn new(mesh_rms: &[f32], grid: MeshGrid, width: usize, height: usize) -> Option<Self> {
        let positive = |s: &f32| s.is_finite() && *s > 0.0;
        let global = mesh_rms.iter().copied().filter(positive).reduce(f32::min)?;
        let (cells_x, cells_y) = (
            grid.nx.saturating_sub(1).max(1),
            grid.ny.saturating_sub(1).max(1),
        );
        let axis = |lower: usize, nodes: usize| -> Vec<usize> {
            if nodes > 1 {
                vec![lower, lower + 1]
            } else {
                vec![0]
            }
        };
        let mut cells = Vec::with_capacity(cells_x * cells_y);
        for cy in 0..cells_y {
            for cx in 0..cells_x {
                let floor = axis(cy, grid.ny)
                    .into_iter()
                    .flat_map(|j| axis(cx, grid.nx).into_iter().map(move |i| (i, j)))
                    .map(|(i, j)| mesh_rms[j * grid.nx + i])
                    .filter(positive)
                    .reduce(f32::min)
                    .unwrap_or(global);
                cells.push(floor);
            }
        }
        Some(Self {
            col: (0..width)
                .map(|x| cell_index(x, grid.bw, grid.nx) as u32)
                .collect(),
            row: (0..height)
                .map(|y| cell_index(y, grid.bh, grid.ny) as u32)
                .collect(),
            cells_x,
            cells,
        })
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.cells[self.row[y] as usize * self.cells_x + self.col[x] as usize]
    }
}

/// §3.5 on the maps actually used.
struct LocalValidity {
    noise: Vec<BackgroundReason>,
    invalid: Option<Vec<u8>>,
    invalid_pixels: Option<u64>,
    floor: Option<CellFloor>,
    check: RmsMapCheck,
}

/// `masked_local_v2` noise validity (§3.5). The zero rule and the mesh-level
/// checks are v1's; the per-pixel rule compares `R` with its cell floor, and
/// only more than `max_invalid_fraction` invalid usable pixels make the
/// noise unresolved for the frame.
fn local_noise_validity(
    background: &Background,
    rms_map: &[f32],
    mask: &[u8],
    usable_pixels: usize,
    fraction: f64,
    max_invalid_fraction: f64,
) -> LocalValidity {
    let (width, height) = (background.width(), background.height());
    let Some(floor) = CellFloor::new(background.mesh_rms(), background.mesh_grid(), width, height)
    else {
        // No finite positive mesh sigma: unresolved exactly as v1 says.
        let (noise, check) = noise_validity(
            background.global_rms(),
            background.mesh_rms(),
            rms_map,
            mask,
        );
        return LocalValidity {
            noise,
            invalid: None,
            invalid_pixels: None,
            floor: None,
            check,
        };
    };
    let (mut zero, mut mesh_invalid) = (false, false);
    for value in
        std::iter::once(background.global_rms()).chain(background.mesh_rms().iter().copied())
    {
        if !value.is_finite() || value < 0.0 {
            mesh_invalid = true;
        } else if value == 0.0 {
            zero = true;
        }
    }
    let mut check = RmsMapCheck {
        usable_min: None,
        mesh_sigma_min_positive: background
            .mesh_rms()
            .iter()
            .copied()
            .filter(|s| s.is_finite() && *s > 0.0)
            .reduce(f32::min),
        lower_bound: None,
        below_bound: 0,
        zero: 0,
        negative_or_non_finite: 0,
    };
    let mut invalid = vec![0u8; width * height];
    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            let i = row + x;
            if mask[i] != 0 {
                continue;
            }
            let r = rms_map[i];
            if !r.is_finite() || r < 0.0 {
                check.negative_or_non_finite += 1;
                invalid[i] = 1;
            } else if r == 0.0 {
                check.zero += 1;
            } else if (r as f64) < fraction * floor.at(x, y) as f64 {
                check.below_bound += 1;
                invalid[i] = 1;
            }
            if r.is_finite() {
                check.usable_min = Some(check.usable_min.map_or(r, |m: f32| m.min(r)));
            }
        }
    }
    let invalid_pixels = check.negative_or_non_finite + check.below_bound;
    zero |= check.zero > 0;
    let frame_wide = invalid_pixels as f64 / usable_pixels.max(1) as f64 > max_invalid_fraction;
    let mut noise = Vec::new();
    if zero {
        noise.push(BackgroundReason::ZeroEmpiricalScatter);
    }
    if mesh_invalid || frame_wide {
        noise.push(BackgroundReason::InvalidRmsMap);
    }
    if invalid_pixels > 0 && noise.is_empty() {
        noise.push(BackgroundReason::RmsMapLocallyInvalid);
    }
    LocalValidity {
        noise,
        invalid: (invalid_pixels > 0).then_some(invalid),
        invalid_pixels: Some(invalid_pixels),
        floor: Some(floor),
        check,
    }
}

/// The source mask of §3.3: footprints of a low-threshold, non-deblended
/// extraction on `I - B1` against `max(R1, floor)`, plus each object's SEP
/// ellipse scaled by `mask_ellipse_scale` (`sep.mask_ellipse`). Returns the
/// mask (1 = source) and the object count.
fn source_mask(
    frame: &LumaFrame,
    settings: &BackgroundSettings,
    mask: &[u8],
    pass1: &Pass,
    floor: &CellFloor,
) -> Result<(Vec<u8>, u32), String> {
    let (width, height) = (frame.width, frame.height);
    let sep = &settings.sep;
    // The mask is built from the objects' footprints, which is exactly
    // SEP's segmentation map (seg > 0) without allocating it.
    let mut params = extract_params(&settings.detection, false)?;
    params.thresh = sep.mask_thresh_sigma as f32;
    params.minarea = sep.mask_minarea as usize;
    params.deblend_cont = MASK_DEBLEND_CONT;
    params.clean = false;
    let residual: Vec<f32> = frame
        .pixels
        .iter()
        .zip(&pass1.level)
        .zip(mask)
        .map(|((&value, &background), &m)| if m != 0 { 0.0 } else { value - background })
        .collect();
    let mut masking_noise = pass1.rms.clone();
    for y in 0..height {
        for (x, r) in masking_noise[y * width..(y + 1) * width]
            .iter_mut()
            .enumerate()
        {
            // NaN-ignoring max, as np.fmax.
            *r = r.max(floor.at(x, y));
        }
    }
    let catalog = extract(
        &residual,
        width,
        height,
        Some(mask),
        Noise::Array(&masking_noise),
        &params,
    )
    .map_err(|e| e.to_string())?;
    drop((residual, masking_noise));
    let mut source = vec![0u8; width * height];
    for object in &catalog.objects {
        for &p in &object.pixels {
            source[p as usize] = 1;
        }
    }
    let ellipses: Vec<Ellipse> = catalog
        .objects
        .iter()
        .map(|o| Ellipse {
            x: o.x,
            y: o.y,
            a: o.a as f64,
            b: o.b as f64,
            theta: o.theta as f64,
        })
        .collect();
    mask_ellipses(
        &mut source,
        width,
        height,
        &ellipses,
        sep.mask_ellipse_scale,
    )
    .map_err(|e| e.to_string())?;
    Ok((source, catalog.objects.len() as u32))
}

/// `masked_local_v2` (`docs/bge-v2-spec.md` §3): source mask, pass 2 and the
/// local validity rule. `Err` only when no background can be evaluated at
/// all (a SEP evaluator failure on pass 1's model).
fn masked_local_v2(
    frame: &LumaFrame,
    settings: &BackgroundSettings,
    mask: &[u8],
    usable_pixels: usize,
    params: &BackgroundParams,
    grid: MeshGrid,
    pass1: Pass,
) -> Result<NoiseMaps, SepBackendFailure> {
    let (width, height) = (frame.width, frame.height);
    let sep = &settings.sep;
    let mut refinement = NoiseRefinement {
        noise_policy: NoisePolicy::MaskedLocalV2,
        pass2_used: false,
        pass2_skipped_reason: None,
        source_mask_objects: None,
        source_mask_pixels: None,
        source_mask_fraction: None,
        invalid_rms_pixels: None,
        invalid_rms_fraction: None,
        detection_area_fraction: None,
    };
    let mut skip_reason = None;

    let pass = match CellFloor::new(pass1.background.mesh_rms(), grid, width, height) {
        None => {
            refinement.pass2_skipped_reason = Some(Pass2SkipReason::NoiseUnresolved);
            pass1
        }
        Some(floor) => match source_mask(frame, settings, mask, &pass1, &floor) {
            Err(error) => {
                tracing::warn!("masked_local_v2 source mask failed ({error}); pass 1 kept");
                refinement.pass2_skipped_reason = Some(Pass2SkipReason::SourceMaskFailed);
                skip_reason = Some(BackgroundReason::SourceMaskFailed);
                pass1
            }
            Ok((source, objects)) => {
                let source_pixels = source
                    .iter()
                    .zip(mask)
                    .filter(|(&s, &m)| s != 0 && m == 0)
                    .count();
                let fraction = source_pixels as f64 / usable_pixels.max(1) as f64;
                refinement.source_mask_objects = Some(objects);
                refinement.source_mask_pixels = Some(source_pixels as u64);
                refinement.source_mask_fraction = Some(fraction);
                let mut combined = source;
                for (c, &m) in combined.iter_mut().zip(mask) {
                    *c |= m;
                }
                let supported = mesh_support(Some(&combined), width, height, grid);
                let below = supported.iter().filter(|ok| !**ok).count();
                if fraction > sep.mask_max_fraction || below == supported.len() {
                    refinement.pass2_skipped_reason = Some(Pass2SkipReason::SourceMaskTooLarge);
                    skip_reason = Some(BackgroundReason::SourceMaskTooLarge);
                    pass1
                } else {
                    match second_pass(frame, &combined, params) {
                        Ok((background, level, rms)) => {
                            refinement.pass2_used = true;
                            Pass {
                                background,
                                level,
                                rms,
                                supported,
                                meshes_below_support: below,
                            }
                        }
                        Err(error) => {
                            tracing::warn!("masked_local_v2 pass 2 failed ({error}); pass 1 kept");
                            refinement.pass2_skipped_reason =
                                Some(Pass2SkipReason::Pass2BackendFailure);
                            pass1
                        }
                    }
                }
            }
        },
    };

    let validity = local_noise_validity(
        &pass.background,
        &pass.rms,
        mask,
        usable_pixels,
        sep.rms_local_fraction,
        sep.rms_max_invalid_fraction,
    );
    log_rejected_map(&validity.noise, &validity.check);
    if let Some(n_inv) = validity.invalid_pixels {
        let usable = usable_pixels.max(1) as f64;
        refinement.invalid_rms_pixels = Some(n_inv);
        refinement.invalid_rms_fraction = Some(n_inv as f64 / usable);
        refinement.detection_area_fraction =
            Some((usable_pixels as u64 - n_inv.min(usable_pixels as u64)) as f64 / usable);
        if validity.noise == [BackgroundReason::RmsMapLocallyInvalid] {
            tracing::debug!(
                invalid_pixels = n_inv,
                "masked_local_v2: locally invalid RMS pixels masked from detection"
            );
        }
    }
    Ok(NoiseMaps {
        pass,
        skip_reason,
        noise: validity.noise,
        invalid: validity.invalid,
        floor: validity.floor,
        refinement: Some(refinement),
    })
}

/// SEP background with the combined detector + source mask, and its maps.
fn second_pass(
    frame: &LumaFrame,
    combined: &[u8],
    params: &BackgroundParams,
) -> Result<(Background, Vec<f32>, Vec<f32>), SepError> {
    let background = Background::estimate(
        &frame.pixels,
        frame.width,
        frame.height,
        Some(combined),
        params,
    )?;
    let level = background.back_array()?;
    let rms = background.rms_array()?;
    Ok((background, level, rms))
}

fn backend_failure(error: &SepError) -> SepBackendFailure {
    SepBackendFailure::new(BackgroundReason::BackendFailure, error.to_string())
}

/// Reasons the empirical noise is not a measurement (empty when it is), and
/// what the RMS map looked like on usable pixels.
///
/// A zero mesh sigma or zero map value is zero *scatter* (a quantized
/// plateau), not zero detector noise. A negative, non-finite, or positive but
/// implausibly small map value (below [`RMS_MAP_LOWER_BOUND_FRACTION`] of the
/// smallest positive mesh sigma, decision D-018) is the natural cubic spline
/// of `bkg_line_flt_internal` undershooting between meshes whose sigmas
/// differ sharply; it cannot serve as a noise map. Never replaced by an
/// epsilon.
fn noise_validity(
    global_rms: f32,
    mesh_rms: &[f32],
    rms_map: &[f32],
    mask: &[u8],
) -> (Vec<BackgroundReason>, RmsMapCheck) {
    let (mut zero, mut invalid) = (false, false);
    let mut classify_sigma = |value: f32| {
        if !value.is_finite() || value < 0.0 {
            invalid = true;
        } else if value == 0.0 {
            zero = true;
        }
    };
    classify_sigma(global_rms);
    mesh_rms.iter().copied().for_each(&mut classify_sigma);

    let mesh_sigma_min_positive = mesh_rms
        .iter()
        .copied()
        .filter(|s| s.is_finite() && *s > 0.0)
        .reduce(f32::min);
    let lower_bound = mesh_sigma_min_positive.map(|s| RMS_MAP_LOWER_BOUND_FRACTION * s);
    let mut check = RmsMapCheck {
        usable_min: None,
        mesh_sigma_min_positive,
        lower_bound,
        below_bound: 0,
        zero: 0,
        negative_or_non_finite: 0,
    };
    for (&r, _) in rms_map.iter().zip(mask).filter(|(_, &m)| m == 0) {
        if !r.is_finite() || r < 0.0 {
            check.negative_or_non_finite += 1;
        } else if r == 0.0 {
            check.zero += 1;
        } else if lower_bound.is_some_and(|bound| r < bound) {
            check.below_bound += 1;
        }
        if r.is_finite() {
            check.usable_min = Some(check.usable_min.map_or(r, |m: f32| m.min(r)));
        }
    }
    zero |= check.zero > 0;
    invalid |= check.negative_or_non_finite > 0 || check.below_bound > 0;
    let mut reasons = Vec::new();
    if zero {
        reasons.push(BackgroundReason::ZeroEmpiricalScatter);
    }
    if invalid {
        reasons.push(BackgroundReason::InvalidRmsMap);
    }
    (reasons, check)
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
        // SEP takes a float32 kernel; narrowing here matches numpy's
        // astype(float32) in the Python wrapper.
        let rows: Vec<Vec<f32>> = detection
            .filter_kernel
            .iter()
            .map(|row| row.iter().map(|value| *value as f32).collect())
            .collect();
        Some(Kernel::from_rows(&rows).map_err(|e| e.to_string())?)
    };
    Ok(ExtractParams {
        // SEP's threshold is a C float; the Python wrapper narrows the same way.
        thresh: detection.thresh_sigma as f32,
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
///
/// `masked_floor` (masked_local_v2, amendment A-1): the noise handed to SEP
/// on masked pixels is the cell floor. See [`noise_for_masked_extraction`].
fn sep_catalog(
    frame: &LumaFrame,
    settings: &BackgroundSettings,
    mask: &[u8],
    level: &[f32],
    rms: &[f32],
    masked_floor: Option<&CellFloor>,
) -> Result<SourceCatalog, String> {
    let params = extract_params(&settings.detection, settings.retain_segmentation)?;
    let residual: Vec<f32> = frame
        .pixels
        .iter()
        .zip(level)
        .zip(mask)
        .map(|((&value, &background), &m)| if m != 0 { 0.0 } else { value - background })
        .collect();
    let floored;
    let noise = match masked_floor {
        Some(floor) => {
            floored = noise_for_masked_extraction(rms, mask, floor, frame.width);
            &floored[..]
        }
        None => rms,
    };
    let catalog = extract(
        &residual,
        frame.width,
        frame.height,
        Some(mask),
        Noise::Array(noise),
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

    /// The pre-v2 noise policy (decision D-018), for the tests that pin it.
    fn v1_settings() -> BackgroundSettings {
        let mut settings = BackgroundSettings::default();
        settings.sep.noise_policy = NoisePolicy::FrameWideV1;
        settings
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
        // frame_wide_v1: the record and hash from before the noise policy.
        let analysis = analyze_sep(
            &frame(sky(256, 192, 5.0, 1, |_, _| 300.0), 256, 192),
            &v1_settings(),
        )
        .unwrap();
        assert!(analysis.result.refinement.is_none());
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

        // masked_local_v2 (the default): its own pinned record; detection
        // configuration unchanged.
        let v2 = analyze_sep(
            &frame(sky(256, 192, 5.0, 1, |_, _| 300.0), 256, 192),
            &settings(),
        )
        .unwrap();
        assert_eq!(
            v2.result.config_sha256,
            "97bd28741f79a184aa55d8e64e8f368ffd1eb71e07b2338cc0b99ed60832a6f4"
        );
        assert_eq!(
            v2.catalog.as_ref().unwrap().provenance.config_sha256,
            "4f4499cda7067107c0fe6d64319684c13d1ec62f590b4bee4b346c1459dc99df"
        );
        let refinement = v2.result.refinement.as_ref().unwrap();
        assert_eq!(refinement.noise_policy, NoisePolicy::MaskedLocalV2);
        assert!(refinement.pass2_used);
        assert_eq!(refinement.invalid_rms_pixels, Some(0));
        assert_eq!(refinement.detection_area_fraction, Some(1.0));
        assert_eq!(v2.result.noise_status, NoiseStatus::Measured);
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
        let analysis = analyze_sep(&frame(pixels.clone(), width, height), &v1_settings()).unwrap();
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

        // masked_local_v2: the undershoot is judged per cell; the ~18% of
        // pixels it makes invalid are masked and the rest is searched,
        // without a false detection on this starless frame.
        let v2 = analyze_sep(&frame(pixels, width, height), &settings()).unwrap();
        assert_locally_invalid_without_detections(&v2, 0.10..0.25);
        // A-1 changes only the noise handed to SEP: the reported R map is
        // still the spline, negative where it undershoots.
        let (_, reported) = map(&v2);
        assert!(reported.iter().any(|&r| r < 0.0));
    }

    #[test]
    fn amendment_a1_gives_masked_pixels_their_cell_floor_as_extraction_noise() {
        let grid = MeshGrid::for_frame(256, 128, 64, 64);
        let sigma = vec![5.0, 9.0, 0.0, 7.0, 6.0, 3.0, 8.0, 4.0];
        let floor = CellFloor::new(&sigma, grid, 256, 128).unwrap();
        let rms: Vec<f32> = (0..256 * 128).map(|i| -1.0 + (i % 7) as f32).collect();
        let mut mask = vec![0u8; 256 * 128];
        for i in [0usize, 5, 255, 300, 256 * 127 + 200] {
            mask[i] = 1;
        }
        let noise = noise_for_masked_extraction(&rms, &mask, &floor, 256);
        for (i, (&n, &r)) in noise.iter().zip(&rms).enumerate() {
            if mask[i] == 1 {
                assert_eq!(n, floor.at(i % 256, i / 256), "pixel {i}");
                assert!(n > 0.0);
            } else {
                assert_eq!(n, r);
            }
        }
    }

    /// The masked_local_v2 outcome on the starless ridge frames.
    fn assert_locally_invalid_without_detections(
        v2: &BackgroundAnalysis,
        expected_fraction: std::ops::Range<f64>,
    ) {
        let result = &v2.result;
        println!("v2: {:?} {:?}", result.reasons, result.refinement);
        assert_eq!(result.reasons, vec![BackgroundReason::RmsMapLocallyInvalid]);
        assert_eq!(result.noise_status, NoiseStatus::Measured);
        assert!(result.rms().is_some());
        let refinement = result.refinement.as_ref().unwrap();
        assert!(refinement.pass2_used);
        let fraction = refinement.invalid_rms_fraction.unwrap();
        assert!(expected_fraction.contains(&fraction), "{fraction}");
        let detection_area = refinement.detection_area_fraction.unwrap();
        assert!((detection_area - (1.0 - fraction)).abs() < 1e-12);
        assert!(detection_area < 0.95);
        let catalog = v2.catalog.as_ref().expect("catalog on the valid area");
        assert!(catalog.objects.is_empty(), "{:?}", catalog.objects);
        let (_, rms) = map(v2);
        let Some(BackgroundField::Map {
            invalid_noise: Some(invalid),
            ..
        }) = v2.field.as_ref()
        else {
            panic!("the field carries M_inv");
        };
        assert_eq!(
            invalid.iter().filter(|&&m| m == 1).count() as u64,
            refinement.invalid_rms_pixels.unwrap()
        );
        assert!(rms.iter().zip(invalid).all(|(&r, &m)| r > 0.0 || m == 1));
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
        let supported = mesh_support(Some(&mask), width, height, grid);
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

    #[test]
    fn positive_rms_undershoot_below_the_mesh_sigmas_is_an_invalid_map() {
        // Review B1: a 600 ADU ridge. The spline of the mesh sigmas stays
        // positive but falls far below every mesh sigma; before D-018 it was
        // accepted and turned noise into ~100 detections.
        let (width, height) = (512, 512);
        let pixels = sky(width, height, 5.0, 4, |x, _| {
            1000.0 + 600.0 * (-((x - 256.0) / 60.0).powi(2)).exp()
        });
        let f = frame(pixels, width, height);
        let analysis = analyze_sep(&f, &v1_settings()).unwrap();
        let (_, rms) = map(&analysis);
        let min = rms.iter().copied().fold(f32::INFINITY, f32::min);
        let background = Background::estimate(
            &f.pixels,
            width,
            height,
            Some(&vec![0; width * height]),
            &BackgroundParams::default(),
        )
        .unwrap();
        let (reasons, check) = noise_validity(
            background.global_rms(),
            background.mesh_rms(),
            rms,
            &vec![0; width * height],
        );
        println!("amp-600 ridge RMS map: min {min}, check {check:?}");
        assert!(min > 0.0, "this case is the positive undershoot, min {min}");
        assert_eq!(check.negative_or_non_finite + check.zero, 0);
        assert!(check.below_bound > 0);
        assert!(check.usable_min.unwrap() < check.lower_bound.unwrap());
        assert_eq!(reasons, vec![BackgroundReason::InvalidRmsMap]);
        let result = &analysis.result;
        assert!(result.reasons.contains(&BackgroundReason::InvalidRmsMap));
        assert_eq!(result.noise_status, NoiseStatus::Unresolved);
        assert_eq!(result.rms, None);
        assert!(result.level().is_some());
        assert!(analysis.catalog.is_none());
        assert_eq!(analysis.catalog_error.as_deref(), Some(NOISE_UNRESOLVED));

        let v2 = analyze_sep(&f, &settings()).unwrap();
        assert_locally_invalid_without_detections(&v2, 0.10..0.25);

        // Above rms_max_invalid_fraction the v1 frame-wide outcome returns.
        let mut strict = settings();
        strict.sep.rms_max_invalid_fraction = 0.1;
        let fallback = analyze_sep(&f, &strict).unwrap();
        assert_eq!(
            fallback.result.reasons,
            vec![BackgroundReason::InvalidRmsMap]
        );
        assert_eq!(fallback.result.noise_status, NoiseStatus::Unresolved);
        assert!(fallback.catalog.is_none());
        assert_eq!(fallback.catalog_error.as_deref(), Some(NOISE_UNRESOLVED));
        let refinement = fallback.result.refinement.as_ref().unwrap();
        assert!(refinement.invalid_rms_fraction.unwrap() > 0.1);
    }

    #[test]
    fn sep_sentinel_values_are_masked_not_fatal() {
        // Review B5: a -1e30 blank value or a 1e31 spike is beyond SEP's BIG
        // sentinel; it is excluded like a NaN instead of failing the frame.
        let (width, height) = (192, 160);
        let mut pixels = sky(width, height, 4.0, 16, |_, _| 300.0);
        pixels[10 * width + 20] = -1e30;
        pixels[50 * width + 60] = 1e31;
        pixels[70 * width + 80] = 1e30;
        let analysis = analyze_sep(&frame(pixels, width, height), &settings()).unwrap();
        assert_eq!(
            analysis.result.support.usable_pixels as usize,
            width * height - 3
        );
        assert!((analysis.result.level().unwrap() - 300.0).abs() < 1.0);
        assert_eq!(analysis.result.noise_status, NoiseStatus::Measured);
        assert!(analysis.catalog.is_some());
    }

    #[test]
    fn bad_mesh_search_above_the_work_bound_is_a_backend_failure() {
        // Review B7: 8 px meshes on 2000x2000 with the left half invalid make
        // filterback's bad-mesh search 31,250 x 62,500 = 1.95e9 units, above
        // achernar_sep::MAX_BAD_MESH_WORK; refused before SEP runs.
        let (width, height) = (2000, 2000);
        let mut f = frame(vec![100.0; width * height], width, height);
        f.masks.invalid = (0..width * height)
            .map(|i| u8::from(i % width < width / 2))
            .collect();
        let mut settings = settings();
        settings.sep.bw = 8;
        settings.sep.bh = 8;
        let failure = analyze_sep(&f, &settings).unwrap_err();
        assert_eq!(failure.reason, BackgroundReason::BackendFailure);
        assert!(
            failure.detail.starts_with("work bound:"),
            "{}",
            failure.detail
        );
    }

    /// The adapter on the crate's Python-reference fixture (sep 1.4.1 wheel,
    /// `achernar-sep/tests/fixtures/parity_v1`): the detector mask (a bad-pixel
    /// mask with NaNs inside it), the residual construction and the catalog
    /// mapping reproduce the wheel within decisions.md D-006.
    ///
    /// Deblended children depend on the C library's `rand_r` (review B2): the
    /// per-child npix/flux/shape checks run only on the fixture's C library
    /// (`meta.json` `rust_target_os`); everywhere, the family rule of
    /// `achernar-sep/tests/parity.rs` (fixed from the source before any
    /// foreign-libc run) is checked, and off that platform the test says the
    /// per-child values are not verifiable.
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
        // The fixture was produced with the D-018 noise policy.
        let mut settings = v1_settings();
        settings.sep.bw = 32;
        settings.sep.bh = 32;
        settings.retain_segmentation = true;
        let analysis = analyze_sep(&f, &settings).unwrap();

        let (level, rms) = map(&analysis);
        let within = |name: &str, ours: &[f32], reference: &[f32]| {
            assert_eq!(ours.len(), width * height, "{name}: map length");
            assert_eq!(reference.len(), width * height, "{name}: reference length");
            assert!(
                ours.iter()
                    .zip(reference)
                    .all(|(o, r)| (o - r).abs() as f64 <= 1e-5 * (r.abs() as f64).max(1.0)),
                "{name}: map exceeds D-006"
            );
        };
        within("background", level, &f32s("back.f32"));
        within("RMS", rms, &f32s("rms.f32"));
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
        let meta = String::from_utf8(read("meta.json")).unwrap();
        let fixture_os = meta
            .lines()
            .find_map(|line| {
                line.trim()
                    .strip_prefix("\"rust_target_os\": \"")
                    .and_then(|rest| rest.strip_suffix("\","))
            })
            .expect("meta.json records rust_target_os");
        let exact = fixture_os == std::env::consts::OS
            && std::env::var_os("ACHERNAR_SEP_PARITY_ASSUME_FOREIGN_LIBC").is_none();
        let merged: Vec<bool> = rows
            .iter()
            .map(|r| r["flag"] as u16 & flags::OBJ_MERGED != 0)
            .collect();
        assert_eq!(catalog.objects.len(), rows.len());
        for (i, (ours, r)) in catalog.objects.iter().zip(&rows).enumerate() {
            // Child barycentres do not depend on rand_r (analyse.c robust
            // branch); everything else of a child does.
            assert_eq!(ours.deblended, merged[i]);
            assert!(
                (ours.x as f64 - r["x"]).abs() <= 1e-4 && (ours.y as f64 - r["y"]).abs() <= 1e-4
            );
            if merged[i] && !exact {
                continue;
            }
            assert_eq!(ours.npix as f64, r["npix"]);
            assert_eq!(
                [ours.xmin, ours.xmax, ours.ymin, ours.ymax].map(|v| v as f64),
                [r["xmin"], r["xmax"], r["ymin"], r["ymax"]]
            );
            assert_eq!(ours.flags as f64, r["flag"]);
            assert_eq!(ours.deblended, r["flag"] as u16 & flags::OBJ_MERGED != 0);
            assert_eq!(ours.truncated, r["flag"] as u16 & flags::OBJ_TRUNC != 0);
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
        assert_eq!(segmap.len(), width * height);

        // Family rule (all platforms): same families, identical union of
        // footprints and total npix, total flux within D-006, flag union.
        let mut reference_pixels = vec![Vec::new(); rows.len()];
        for (p, &id) in segmap.iter().enumerate() {
            if id > 0 {
                reference_pixels[id as usize - 1].push(p as u32);
            }
        }
        let families = achernar_sep::deblend_families(
            reference_pixels
                .iter()
                .zip(&merged)
                .map(|(p, &m)| (p.as_slice(), m)),
            width,
            height,
        );
        let ours_families = achernar_sep::deblend_families(
            catalog
                .objects
                .iter()
                .map(|o| (o.pixels.as_slice(), o.deblended)),
            width,
            height,
        );
        assert_eq!(ours_families, families);
        assert!(!families.is_empty(), "the fixture holds a deblended pair");
        let mut family_of = vec![usize::MAX; rows.len()];
        for (f, members) in families.iter().enumerate() {
            let sorted = |mut v: Vec<u32>| {
                v.sort_unstable();
                v
            };
            let ours_union = sorted(
                members
                    .iter()
                    .flat_map(|&i| catalog.objects[i].pixels.clone())
                    .collect(),
            );
            let ref_union = sorted(
                members
                    .iter()
                    .flat_map(|&i| reference_pixels[i].clone())
                    .collect(),
            );
            assert_eq!(ours_union, ref_union, "family {f} pixels");
            let npix: u32 = members.iter().map(|&i| catalog.objects[i].npix).sum();
            let ref_npix: f64 = members.iter().map(|&i| rows[i]["npix"]).sum();
            assert_eq!(npix as f64, ref_npix, "family {f} npix");
            let flux: f64 = members
                .iter()
                .map(|&i| catalog.objects[i].flux as f64)
                .sum();
            let ref_flux: f64 = members.iter().map(|&i| rows[i]["flux"]).sum();
            assert!((flux - ref_flux).abs() <= 1e-5 * ref_flux.abs().max(1.0));
            let flag_union = |f: &dyn Fn(usize) -> u16| members.iter().fold(0, |a, &i| a | f(i));
            assert_eq!(
                flag_union(&|i| catalog.objects[i].flags),
                flag_union(&|i| rows[i]["flag"] as u16)
            );
            for &i in members {
                family_of[i] = f;
            }
        }
        let ours_segmap = catalog.segmentation.as_ref().unwrap();
        assert_eq!(ours_segmap.len(), width * height);
        if exact {
            assert_eq!(ours_segmap, &segmap);
        } else {
            for (&ours, &r) in ours_segmap.iter().zip(&segmap) {
                assert!(match r {
                    0 => ours == 0,
                    r if !merged[r as usize - 1] => ours == r,
                    r => ours > 0 && family_of[ours as usize - 1] == family_of[r as usize - 1],
                });
            }
            println!(
                "NOTE: fixture C library differs from this run's (target_os {}): per-child \
                 npix/flux/shape of SEP_OBJ_MERGED children NOT VERIFIABLE here; checked the \
                 deblend-family invariants instead",
                std::env::consts::OS
            );
        }
    }

    #[test]
    fn cell_floor_uses_the_pixels_spline_cell_with_edge_and_single_node_rules() {
        // Nodes at (j + 0.5) * bw: with bw = 64, x = 0..31 lies before node 0
        // (cell 0), x = 32..95 between nodes 0 and 1, and so on; the last
        // cell extends past the last node.
        assert_eq!(cell_index(0, 64, 4), 0);
        assert_eq!(cell_index(31, 64, 4), 0);
        assert_eq!(cell_index(32, 64, 4), 0);
        assert_eq!(cell_index(95, 64, 4), 0);
        assert_eq!(cell_index(96, 64, 4), 1);
        assert_eq!(cell_index(160, 64, 4), 2);
        assert_eq!(cell_index(255, 64, 4), 2, "clamped to nx - 2");
        assert_eq!(cell_index(10_000, 64, 4), 2);
        assert_eq!(cell_index(500, 64, 1), 0, "a single-node axis");
        assert_eq!(cell_index(500, 64, 2), 0);

        // 4 x 2 mesh grid of a 256 x 128 frame.
        let grid = MeshGrid::for_frame(256, 128, 64, 64);
        let sigma = vec![
            5.0,
            9.0,
            0.0,
            7.0, //
            6.0,
            3.0,
            8.0,
            f32::NAN,
        ];
        let floor = CellFloor::new(&sigma, grid, 256, 128).unwrap();
        // Cell (0, 0): nodes (0,0) (1,0) (0,1) (1,1) -> min(5, 9, 6, 3).
        assert_eq!(floor.at(0, 0), 3.0);
        assert_eq!(floor.at(95, 127), 3.0);
        // Cell (1, 0): 9, 0 (not positive), 3, 8 -> 3.
        assert_eq!(floor.at(96, 0), 3.0);
        // Cell (2, 0) at the right edge: 0, 7, 8, NaN -> 7.
        assert_eq!(floor.at(255, 64), 7.0);
        assert_eq!(floor.at(160, 0), 7.0);

        // A single node per axis (one mesh column): the column's nodes only.
        let grid = MeshGrid::for_frame(50, 200, 64, 64);
        assert_eq!((grid.nx, grid.ny), (1, 4));
        let floor = CellFloor::new(&[4.0, 2.0, 0.0, 6.0], grid, 50, 200).unwrap();
        assert_eq!(floor.at(49, 0), 2.0);
        assert_eq!(floor.at(0, 96), 2.0, "cell 1: nodes 1, 2 -> 2 (0 skipped)");
        assert_eq!(floor.at(0, 160), 6.0, "cell 2: nodes 2, 3 -> 6");
        // A cell with no positive node takes the frame's smallest one.
        let floor = CellFloor::new(&[4.0, 0.0, 0.0, 2.5], grid, 50, 200).unwrap();
        assert_eq!(floor.at(0, 96), 2.5);
        // No positive finite sigma anywhere: undefined.
        assert!(CellFloor::new(&[0.0, f32::NAN, -1.0, 0.0], grid, 50, 200).is_none());
    }

    fn starfield(width: usize, height: usize, seed: u64, stars: usize) -> Vec<f32> {
        let mut pixels = sky(width, height, 5.0, seed, |x, y| 500.0 + 0.2 * x - 0.1 * y);
        let mut rng = Rng(seed + 1000);
        for _ in 0..stars {
            let (x, y) = (
                10.0 + rng.uniform() * (width - 20) as f64,
                10.0 + rng.uniform() * (height - 20) as f64,
            );
            add_star(
                &mut pixels,
                width,
                x,
                y,
                1.5 + rng.uniform(),
                300.0 + 3000.0 * rng.uniform(),
            );
        }
        pixels
    }

    #[test]
    fn masked_local_v2_masks_sources_for_pass_2_and_keeps_v1_detection_semantics() {
        let (width, height) = (384, 256);
        let f = frame(starfield(width, height, 21, 25), width, height);
        let v1 = analyze_sep(&f, &v1_settings()).unwrap();
        let v2 = analyze_sep(&f, &settings()).unwrap();
        let refinement = v2.result.refinement.as_ref().unwrap();
        println!("starfield v2 refinement: {refinement:?}");
        assert!(refinement.pass2_used);
        assert_eq!(refinement.pass2_skipped_reason, None);
        assert!(refinement.source_mask_objects.unwrap() >= 20);
        let fraction = refinement.source_mask_fraction.unwrap();
        assert!(fraction > 0.0 && fraction < 0.5, "{fraction}");
        assert_eq!(
            refinement.source_mask_pixels.unwrap() as f64 / (width * height) as f64,
            fraction
        );
        assert_eq!(v2.result.noise_status, NoiseStatus::Measured);
        assert_eq!(v2.result.reasons, Vec::<BackgroundReason>::new());
        // Masking the stars lowers the level estimate under them: pass 2 is
        // not pass 1 (whose maps v1 reports).
        let ((level1, _), (level2, _)) = (map(&v1), map(&v2));
        assert_ne!(level1, level2);
        // The same stars are found.
        let (n1, n2) = (
            v1.catalog.as_ref().unwrap().objects.len(),
            v2.catalog.as_ref().unwrap().objects.len(),
        );
        assert!((n1 as i64 - n2 as i64).abs() <= 2, "{n1} vs {n2}");
        assert!(v1.result.refinement.is_none());
    }

    #[test]
    fn a_giant_source_mask_skips_pass_2_and_keeps_pass_1() {
        let (width, height) = (384, 256);
        let f = frame(starfield(width, height, 22, 25), width, height);
        let v1 = analyze_sep(&f, &v1_settings()).unwrap();

        // More than mask_max_fraction of the usable pixels.
        let mut tight = settings();
        tight.sep.mask_max_fraction = 0.001;
        let skipped = analyze_sep(&f, &tight).unwrap();
        let refinement = skipped.result.refinement.as_ref().unwrap();
        assert!(!refinement.pass2_used);
        assert_eq!(
            refinement.pass2_skipped_reason,
            Some(Pass2SkipReason::SourceMaskTooLarge)
        );
        assert!(refinement.source_mask_fraction.unwrap() > 0.001);
        assert!(skipped
            .result
            .reasons
            .contains(&BackgroundReason::SourceMaskTooLarge));
        assert_eq!(map(&skipped), map(&v1), "pass 1 is v1's background");
        assert_eq!(skipped.result.level, v1.result.level);
        assert_eq!(skipped.result.mesh, v1.result.mesh);
        assert_eq!(skipped.result.noise_status, NoiseStatus::Measured);

        // Ellipses that cover every mesh: no mesh keeps SEP's 50% support.
        let mut huge = settings();
        huge.sep.mask_ellipse_scale = 1000.0;
        huge.sep.mask_max_fraction = 1.0;
        let skipped = analyze_sep(&f, &huge).unwrap();
        let refinement = skipped.result.refinement.as_ref().unwrap();
        assert!(!refinement.pass2_used);
        assert_eq!(
            refinement.pass2_skipped_reason,
            Some(Pass2SkipReason::SourceMaskTooLarge)
        );
        assert_eq!(refinement.source_mask_fraction, Some(1.0));
        assert_eq!(map(&skipped), map(&v1));
    }

    #[test]
    fn a_failed_source_mask_keeps_pass_1_and_says_so() {
        let (width, height) = (192, 160);
        let mut pixels = sky(width, height, 3.0, 13, |_, _| 100.0);
        add_star(&mut pixels, width, 96.0, 80.0, 4.0, 3000.0);
        let mut small = settings();
        small.detection.pixstack = 10;
        let analysis = analyze_sep(&frame(pixels.clone(), width, height), &small).unwrap();
        let refinement = analysis.result.refinement.as_ref().unwrap();
        assert_eq!(
            refinement.pass2_skipped_reason,
            Some(Pass2SkipReason::SourceMaskFailed)
        );
        assert!(!refinement.pass2_used);
        assert_eq!(refinement.source_mask_objects, None);
        assert_eq!(refinement.source_mask_pixels, None);
        assert_eq!(
            analysis.result.reasons,
            vec![BackgroundReason::SourceMaskFailed]
        );
        // The pass-1 noise is still judged locally and measured.
        assert_eq!(analysis.result.noise_status, NoiseStatus::Measured);
        assert_eq!(refinement.invalid_rms_pixels, Some(0));
        let mut v1 = v1_settings();
        v1.detection.pixstack = 10;
        let v1 = analyze_sep(&frame(pixels, width, height), &v1).unwrap();
        assert_eq!(map(&analysis), map(&v1));
    }

    #[test]
    fn masked_local_v2_keeps_the_zero_rule_and_v1_reasons_without_a_floor() {
        let (width, height) = (192, 160);
        let analysis = analyze_sep(
            &frame(vec![321.0; width * height], width, height),
            &settings(),
        )
        .unwrap();
        assert_eq!(analysis.result.noise_status, NoiseStatus::Unresolved);
        assert_eq!(
            analysis.result.reasons,
            vec![BackgroundReason::ZeroEmpiricalScatter]
        );
        let refinement = analysis.result.refinement.as_ref().unwrap();
        assert_eq!(
            refinement.pass2_skipped_reason,
            Some(Pass2SkipReason::NoiseUnresolved)
        );
        assert_eq!(refinement.invalid_rms_pixels, None);
        assert_eq!(refinement.detection_area_fraction, None);
        assert_eq!(analysis.catalog_error.as_deref(), Some(NOISE_UNRESOLVED));
    }

    #[test]
    fn locally_invalid_pixels_are_excluded_from_detection_and_star_support() {
        // The 600 ADU ridge (invalid columns either side of the ridge) with
        // stars: one inside the invalid band, one well clear of it.
        let (width, height) = (512, 512);
        let base = sky(width, height, 5.0, 4, |x, _| {
            1000.0 + 600.0 * (-((x - 256.0) / 60.0).powi(2)).exp()
        });
        let starless = analyze_sep(&frame(base.clone(), width, height), &settings()).unwrap();
        let Some(BackgroundField::Map {
            invalid_noise: Some(invalid),
            ..
        }) = starless.field.as_ref()
        else {
            panic!("expected M_inv");
        };
        let column_invalid =
            |x: usize| (0..height).filter(|&y| invalid[y * width + x] == 1).count();
        let inside = (0..width)
            .find(|&x| column_invalid(x) == height && column_invalid(x + 20) == height)
            .expect("a fully invalid band");
        let clear = (64..width - 64)
            .find(|&x| (x - 40..x + 40).all(|c| column_invalid(c) == 0))
            .expect("a clear band");
        println!("invalid band from x = {inside}, clear star at x = {clear}");
        let mut pixels = base;
        add_star(&mut pixels, width, (inside + 10) as f64, 256.0, 1.8, 2000.0);
        add_star(&mut pixels, width, clear as f64, 256.0, 1.8, 2000.0);
        let analysis = analyze_sep(&frame(pixels.clone(), width, height), &settings()).unwrap();
        assert!(analysis
            .result
            .reasons
            .contains(&BackgroundReason::RmsMapLocallyInvalid));
        let Some(BackgroundField::Map {
            invalid_noise: Some(invalid),
            ..
        }) = analysis.field.as_ref()
        else {
            panic!("expected M_inv");
        };
        let catalog = analysis.catalog.as_ref().unwrap();
        // No footprint pixel lies in M_inv.
        assert!(catalog
            .objects
            .iter()
            .all(|o| o.pixels.iter().all(|&p| invalid[p as usize] == 0)));
        assert!(catalog
            .objects
            .iter()
            .any(|o| (o.x - clear as f32).abs() < 1.0 && (o.y - 256.0).abs() < 1.0));
        let quality =
            crate::vision::quality::estimate_quality(&frame(pixels, width, height), &analysis);
        println!(
            "ridge+stars quality: {:?} {:?}",
            quality.star_rejections, quality.used_stars
        );
        assert!(quality
            .used_stars
            .iter()
            .any(|s| (s.x - clear as f32).abs() < 1.0));
        assert!(quality
            .used_stars
            .iter()
            .all(|s| (s.x - (inside + 10) as f32).abs() > 30.0));
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
        // Spec A10: median of 5, frame_wide_v1 against masked_local_v2.
        let runs = 5;
        let mut medians = Vec::new();
        for (name, settings) in [
            ("frame_wide_v1", v1_settings()),
            ("masked_local_v2", settings()),
        ] {
            let _ = analyze_sep(&frame, &settings).unwrap();
            let mut times = Vec::new();
            let mut objects = 0;
            let mut refinement = None;
            for _ in 0..runs {
                let start = std::time::Instant::now();
                let analysis = analyze_sep(&frame, &settings).unwrap();
                times.push(start.elapsed().as_secs_f64() * 1e3);
                objects = analysis.catalog.map_or(0, |c| c.objects.len());
                refinement = analysis.result.refinement;
            }
            times.sort_by(f64::total_cmp);
            let median = times[runs / 2];
            medians.push(median);
            println!(
                "available host software timing (not a deployment target): analyze_sep {name} \
                 on {width}x{height} f32, {objects} objects: median {median:.1} ms of {runs} \
                 (runs {times:.1?}); refinement {refinement:?}"
            );
        }
        println!(
            "masked_local_v2 / frame_wide_v1 median ratio: {:.2}",
            medians[1] / medians[0]
        );
    }
}

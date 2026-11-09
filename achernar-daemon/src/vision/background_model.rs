//! Versioned background-estimation contract for the vision bus.
//!
//! One frame has one authoritative [`BackgroundResult`]: which estimator ran,
//! with which effective configuration, over which pixels, in which units, and
//! whether its level and its noise scale were actually measured. Detection,
//! source gates, HFR and trailing consume the paired [`BackgroundField`], so no
//! stage silently re-derives its own sky from a different model (audit
//! BGE-01, BGE-02, BGE-06, BGE-13, BGE-21).
//!
//! Absence is explicit. A level or noise scale that could not be measured is
//! `None` with a reason; it is never a numeric zero. A measured level of zero
//! is a valid measurement and is reported as `Some(0.0)`.
//!
//! The level and its noise scale have separate validity because a constant or
//! coarsely quantized image can have a well-defined level and no resolved
//! noise (BGE-07): zero empirical scatter is not zero detector noise.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Schema of [`BackgroundResult`]. Bump on any change of field meaning.
pub const BACKGROUND_SCHEMA_VERSION: u32 = 1;

/// SEP release whose C sources are vendored and whose Python wheel is the
/// reference implementation. Changing it changes measurements.
pub const SEP_LIBRARY_VERSION: &str = "1.4.1";

/// The background estimator that produced a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundMethod {
    /// Custom global estimator in `background.rs`: frame median, a robust
    /// scale seed, one ±3-seed-sigma clipping pass, clipped median and
    /// population standard deviation. No spatial model. This is not SEP and
    /// not Source Extractor; it is retained for diagnostic comparison.
    #[default]
    LegacyGlobalClip,
    /// SEP spatial mesh background (Bertin & Arnouts 1996 §2 algorithm family
    /// as implemented by the pinned SEP C library), evaluated per pixel with
    /// SEP's own array evaluator.
    SepSpatialMesh,
}

impl BackgroundMethod {
    /// Implementation version of Achernar's use of the method. Bump when the
    /// adapter's semantics change even if the library does not.
    pub fn method_version(self) -> u32 {
        match self {
            BackgroundMethod::LegacyGlobalClip => 1,
            BackgroundMethod::SepSpatialMesh => 1,
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            BackgroundMethod::LegacyGlobalClip => "legacy_global_clip",
            BackgroundMethod::SepSpatialMesh => "sep_spatial_mesh",
        }
    }
}

/// Validity of the background level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LevelStatus {
    Measured,
    Unavailable,
}

/// Validity of the noise scale, separate from the level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoiseStatus {
    /// A positive noise scale was measured from the declared source.
    Measured,
    /// The level is known but the empirical scatter is zero (for example a
    /// quantized plateau). Detector noise is not known to be zero.
    Unresolved,
    /// No noise estimate exists (for example because no level exists).
    Unavailable,
}

/// Where the noise scale comes from. Sources are not interchangeable and are
/// never summed with each other without an explicit noise budget (BGE-08).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoiseSource {
    /// Population standard deviation of the legacy clipped global sample.
    EmpiricalGlobalClip,
    /// SEP's interpolated per-mesh clipped RMS map.
    EmpiricalMeshRms,
    /// A per-pixel error array supplied by an upstream pipeline (e.g. HST ERR).
    PipelineError,
    /// A calibrated detector model (gain/read noise). Not implemented.
    CalibratedDetectorModel,
    Unavailable,
}

/// Machine-readable reasons attached to a result. Several may apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundReason {
    /// Zero-sized frame, or pixel/mask lengths disagree with the geometry.
    MalformedGeometry,
    /// No finite pixel survives the detector-validity and saturation masks.
    NoUsablePixels,
    /// Too few usable pixels for the configured estimator.
    InsufficientSupport,
    /// The frame does not hold the minimum mesh grid for a spatial model.
    FrameSmallerThanMeshGrid,
    /// Usable pixels exist but their empirical scatter is zero.
    ZeroEmpiricalScatter,
    /// The interpolated RMS map is negative or non-finite on usable pixels.
    /// SEP's bicubic spline overshoots between meshes whose clipped sigmas
    /// differ sharply (for example a steep background ridge inflating some
    /// meshes' sigma), so the map cannot serve as a detection noise.
    InvalidRmsMap,
    /// The SEP library returned an error (see `backend_error`).
    BackendFailure,
    /// The selected backend failed and the configured fallback ran instead.
    FallbackUsed,
    /// The method has no spatial model, so no gradient can be reported.
    NoSpatialModel,
    /// Too few adequately supported meshes to fit a plane.
    InsufficientMeshesForPlane,
    /// Some meshes had less usable support than SEP requires and were
    /// replaced by SEP's interpolation from neighbours.
    MeshesInterpolated,
}

/// Effective configuration of the legacy global estimator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegacyGlobalClipConfig {
    pub clip_sigma: f32,
    pub clip_passes: u32,
    /// Robust seed scale: `max(1.4826*MAD, 0.5*(p84-p16))`.
    pub seed: String,
}

impl Default for LegacyGlobalClipConfig {
    fn default() -> Self {
        Self {
            clip_sigma: 3.0,
            clip_passes: 1,
            seed: "max(1.4826*mad,0.5*(p84-p16))".to_string(),
        }
    }
}

/// SEP background mesh parameters (`sep_background`). Starting values, not
/// qualified camera defaults: 64 px meshes and a 3x3 median filter of the
/// mesh grid, as in the SEP Python defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SepBackgroundParams {
    #[serde(default = "default_mesh")]
    pub bw: u32,
    #[serde(default = "default_mesh")]
    pub bh: u32,
    #[serde(default = "default_filter")]
    pub fw: u32,
    #[serde(default = "default_filter")]
    pub fh: u32,
    /// Filter threshold of SEP's mesh median filter (input units); 0 filters
    /// every mesh, as in the SEP Python default.
    /// Stored as f64 so the recorded configuration (and its hash) carries
    /// exactly the value the Python reference records; SEP takes a double.
    #[serde(default)]
    pub fthresh: f64,
    /// The spatial model requires at least this many meshes per axis. Below
    /// that a "mesh" model is a global estimate wearing a spatial label.
    #[serde(default = "default_min_meshes_per_axis")]
    pub min_meshes_per_axis: u32,
}

fn default_mesh() -> u32 {
    64
}
fn default_filter() -> u32 {
    3
}
fn default_min_meshes_per_axis() -> u32 {
    2
}

impl Default for SepBackgroundParams {
    fn default() -> Self {
        Self {
            bw: default_mesh(),
            bh: default_mesh(),
            fw: default_filter(),
            fh: default_filter(),
            fthresh: 0.0,
            min_meshes_per_axis: default_min_meshes_per_axis(),
        }
    }
}

/// SEP extraction parameters (`sep_extract`). The threshold is relative to
/// the per-pixel background RMS map. Starting values, not qualified defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SepDetectionParams {
    /// Detection threshold in units of the local RMS, R(x,y).
    /// f64 for exact provenance; narrowed to SEP's `float` at the call, as
    /// the Python wrapper does.
    #[serde(default = "default_thresh_sigma")]
    pub thresh_sigma: f64,
    #[serde(default = "default_minarea")]
    pub minarea: u32,
    /// `conv` or `matched`. SEP 1.4.1 compares a CONV-filtered residual with
    /// `thresh_sigma * R(x,y)`, but a MATCHED-filtered unit-variance S/N with
    /// `thresh_sigma` itself; for the 3x3 default kernel MATCHED is ~2.67x more
    /// sensitive at the same number (sqrt(sum k^2)/sum|k| = 6/16). SEP also
    /// silently downgrades MATCHED to CONV when the noise is scalar, which is
    /// what the Python labeler effectively ran with `err=globalrms`. `conv`
    /// keeps that detection semantics while the noise becomes the local map.
    #[serde(default = "default_filter_type")]
    pub filter_type: String,
    /// Row-major kernel. SEP Python default 3x3 `[[1,2,1],[2,4,2],[1,2,1]]`.
    /// f64 for exact provenance; narrowed to float32 for SEP, as numpy does.
    #[serde(default = "default_kernel")]
    pub filter_kernel: Vec<Vec<f64>>,
    #[serde(default = "default_deblend_nthresh")]
    pub deblend_nthresh: u32,
    #[serde(default = "default_deblend_cont")]
    pub deblend_cont: f64,
    #[serde(default = "default_true")]
    pub clean: bool,
    #[serde(default = "default_clean_param")]
    pub clean_param: f64,
    /// SEP's process-global pixel stack bound (`sep_set_extract_pixstack`).
    #[serde(default = "default_pixstack")]
    pub pixstack: u64,
    /// Accept SEP-deblended children as focus stars. Deblending (the
    /// library splitting a blend) and accepting a deblended star (a focus
    /// policy) are separate decisions; the default rejects them.
    #[serde(default)]
    pub accept_deblended_for_focus: bool,
}

fn default_thresh_sigma() -> f64 {
    5.0
}
fn default_minarea() -> u32 {
    5
}
fn default_filter_type() -> String {
    "conv".to_string()
}
fn default_kernel() -> Vec<Vec<f64>> {
    vec![
        vec![1.0, 2.0, 1.0],
        vec![2.0, 4.0, 2.0],
        vec![1.0, 2.0, 1.0],
    ]
}
fn default_deblend_nthresh() -> u32 {
    32
}
fn default_deblend_cont() -> f64 {
    0.005
}
fn default_true() -> bool {
    true
}
fn default_clean_param() -> f64 {
    1.0
}
fn default_pixstack() -> u64 {
    300_000
}

impl Default for SepDetectionParams {
    fn default() -> Self {
        Self {
            thresh_sigma: default_thresh_sigma(),
            minarea: default_minarea(),
            filter_type: default_filter_type(),
            filter_kernel: default_kernel(),
            deblend_nthresh: default_deblend_nthresh(),
            deblend_cont: default_deblend_cont(),
            clean: true,
            clean_param: default_clean_param(),
            pixstack: default_pixstack(),
            accept_deblended_for_focus: false,
        }
    }
}

/// Which backend the vision loop runs. `legacy_global_clip` is the default;
/// `sep` is an explicit opt-in diagnostic path, not a qualified default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundBackend {
    #[default]
    LegacyGlobalClip,
    Sep,
}

/// What happens when the selected SEP backend cannot produce a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundFallback {
    /// Report the background unavailable with the SEP failure reason.
    #[default]
    Unavailable,
    /// Run the legacy estimator and label the result as a fallback.
    LegacyGlobalClip,
}

/// `[vision.background]` in `achernar.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct BackgroundSettings {
    #[serde(default)]
    pub backend: BackgroundBackend,
    #[serde(default)]
    pub fallback: BackgroundFallback,
    #[serde(default)]
    pub sep: SepBackgroundParams,
    #[serde(default)]
    pub detection: SepDetectionParams,
    /// Keep SEP's segmentation map on the catalog (diagnostics only; one i32
    /// per analysis pixel).
    #[serde(default)]
    pub retain_segmentation: bool,
}

/// Scalar summaries of a spatial model over its mesh grid. Percentiles are
/// over the filtered mesh values (one per mesh), not over pixels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshSummary {
    pub nx: u32,
    pub ny: u32,
    pub bw: u32,
    pub bh: u32,
    pub level_min: f32,
    pub level_p05: f32,
    pub level_p50: f32,
    pub level_p95: f32,
    pub level_max: f32,
    pub rms_p05: f32,
    pub rms_p50: f32,
    pub rms_p95: f32,
    /// Meshes whose usable-pixel fraction was below SEP's acceptance limit
    /// (their values come from SEP's neighbour interpolation).
    pub meshes_below_support: u32,
}

/// Least-squares plane `level = b0 + gx*(x-x0) + gy*(y-y0)` fitted to the
/// adequately supported mesh values, with mesh centres in analysis-pixel
/// coordinates. Units: input intensity units per analysis pixel. The residual
/// fields are the evidence that two slopes do not describe curvature,
/// vignetting or glow; a large residual means the plane is a poor summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlaneFit {
    pub x0: f32,
    pub y0: f32,
    pub b0: f32,
    pub gx_per_px: f32,
    pub gy_per_px: f32,
    pub meshes_used: u32,
    /// RMS of (mesh value - plane) over the meshes used.
    pub residual_rms: f32,
    /// Largest |mesh value - plane| over the meshes used.
    pub residual_max_abs: f32,
}

/// Pixel support of the estimate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackgroundSupport {
    pub total_pixels: u64,
    /// Finite pixels outside the bad/invalid/edge and native-saturation masks.
    pub usable_pixels: u64,
    /// Pixels retained after the legacy clip (legacy method only).
    pub clipped_sample_pixels: Option<u64>,
    /// Which masks constrained the estimate: mask name (`bad_pixel`,
    /// `invalid`, `edge`, `native_saturation`) to `provided` or
    /// `unavailable`. A bare array without masks says `unavailable`, so an
    /// absent mask is never mistaken for a clean detector.
    pub masks: BTreeMap<String, String>,
    /// Evidence behind `saturation_fraction`: `native_threshold`,
    /// `native_mask`, `legacy_peak_threshold`, or `unknown`.
    pub saturation_evidence: String,
}

/// The authoritative, versioned background result for one frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackgroundResult {
    pub schema_version: u32,
    pub method: BackgroundMethod,
    pub method_version: u32,
    /// `sep` for the SEP method; `None` for Achernar-only methods.
    pub library: Option<String>,
    pub library_version: Option<String>,
    /// Method the configuration asked for; differs from `method` only when a
    /// configured fallback ran.
    pub requested_method: BackgroundMethod,
    /// Effective configuration, canonical JSON object.
    pub config: serde_json::Value,
    /// sha256 (hex) of the canonical JSON of `config`.
    pub config_sha256: String,
    pub level_status: LevelStatus,
    pub noise_status: NoiseStatus,
    pub reasons: Vec<BackgroundReason>,
    /// SEP error code and message when `BackendFailure` applies.
    pub backend_error: Option<String>,
    /// Input intensity units: the analysis descriptor's units (`adu`,
    /// `adu_proxy`), or `unverified` for frames without a descriptor.
    pub units: String,
    /// `analysis_pixel`: x is the column index, y the row index, both
    /// 0-based at pixel centres, on the frame's analysis grid.
    pub coordinate_frame: String,
    pub width: u32,
    pub height: u32,
    pub noise_source: NoiseSource,
    pub support: BackgroundSupport,
    /// Frame-level summary of the background level; `level_definition`
    /// states exactly what population it summarises. It is a summary, never
    /// a substitute for the spatial field during detection or measurement.
    pub level: Option<f32>,
    pub level_definition: String,
    /// Frame-level summary of the noise scale; `rms_definition` says how.
    pub rms: Option<f32>,
    pub rms_definition: String,
    pub mesh: Option<MeshSummary>,
    pub plane: Option<PlaneFit>,
    pub plane_unavailable_reason: Option<BackgroundReason>,
    /// Legacy heuristic, formerly `snr_proxy_median`: median of
    /// `(I-B)/sqrt(sigma^2+9)` over pixels with `I-B > 3*max(sigma,1)`, using
    /// the global level and scale. The fixed 9 ADU^2 term is not a camera
    /// calibration. This is a pixel-significance proxy, not aperture or PSF
    /// SNR. `None` when no pixel qualifies or the method does not compute it.
    pub bright_pixel_significance: Option<f32>,
    /// Fraction of analysis pixels touched by clipped native support (or, for
    /// legacy frames without a descriptor, usable pixels at/above 98% of the
    /// declared saturation point). `None` when the frame has no saturation
    /// evidence (`support.saturation_evidence == "unknown"`) or no finite
    /// pixel: 0.0 would otherwise read as "nothing clipped".
    pub saturation_fraction: Option<f32>,
}

impl BackgroundResult {
    /// The level, if it was measured.
    pub fn level(&self) -> Option<f32> {
        matches!(self.level_status, LevelStatus::Measured)
            .then_some(self.level)
            .flatten()
    }

    /// The noise scale, if it was measured and positive.
    pub fn rms(&self) -> Option<f32> {
        matches!(self.noise_status, NoiseStatus::Measured)
            .then_some(self.rms)
            .flatten()
    }

    /// Identity of the measurement semantics. A change of method, effective
    /// configuration, units or noise source makes old and new values
    /// incomparable (cloud-trend baselines restart on it).
    pub fn identity(&self) -> String {
        format!(
            "{}:{}:{}:{}:{:?}",
            self.method.id(),
            self.method_version,
            self.config_sha256,
            self.units,
            self.noise_source
        )
    }
}

/// Canonical JSON, byte-identical to Python's
/// `json.dumps(obj, sort_keys=True, separators=(",", ":"))` (default
/// `ensure_ascii=True`), which the Python reference uses for the same
/// configuration hashes: keys sorted by code point, no whitespace, floats in
/// Python `repr` form (shortest round-trip digits; exponent form `1e-05` /
/// `1e+16` when the decimal point position is <= -4 or > 16), and every
/// character outside `' '..='~'` escaped as `\uXXXX`.
pub fn canonical_json(value: &serde_json::Value) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &serde_json::Value, out: &mut String) {
    use serde_json::Value;
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) => match (number.as_u64(), number.as_i64()) {
            (Some(unsigned), _) => out.push_str(&unsigned.to_string()),
            (None, Some(signed)) => out.push_str(&signed.to_string()),
            (None, None) => out.push_str(&python_float_repr(
                number
                    .as_f64()
                    .expect("a JSON number is an integer or a float"),
            )),
        },
        Value::String(text) => write_python_string(text, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            // UTF-8 byte order equals code point order, which is Python's
            // `sorted` order for str keys.
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_python_string(key, out);
                out.push(':');
                write_canonical(&map[key], out);
            }
            out.push('}');
        }
    }
}

/// `json.dumps` string encoding with `ensure_ascii=True`.
fn write_python_string(text: &str, out: &mut String) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(ch),
            _ => {
                let mut units = [0u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

/// CPython `float.__repr__` (`repr`, "short" style) of a finite or
/// non-finite f64, as `json.dumps` writes it.
fn python_float_repr(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .to_string();
    }
    // `{:e}` gives the shortest round-trip digit count ("1.2e0", "-5e-3",
    // "1e16"). CPython breaks exact decimal ties to even where the shortest
    // formatter may not, so use the correctly rounded string of that many
    // digits (`{:.*e}`: exact expansion, ties to even) when it round-trips.
    // Cross-checked against CPython on 183,147 values (random bit patterns,
    // all magnitudes, subnormals, exponent-switch boundaries): 0 mismatches.
    let shortest = format!("{value:e}");
    let significant = shortest
        .split_once('e')
        .map(|(mantissa, _)| mantissa.chars().filter(char::is_ascii_digit).count())
        .unwrap_or(1);
    let rounded = format!("{:.*e}", significant.saturating_sub(1), value);
    let scientific = if rounded.parse::<f64>().ok() == Some(value) {
        rounded
    } else {
        shortest
    };
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("LowerExp always writes an exponent");
    let exponent: i32 = exponent.parse().expect("integer exponent");
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    // Position of the decimal point relative to the first significant digit.
    let decpt = exponent + 1;
    let mut text = String::new();
    if negative {
        text.push('-');
    }
    if decpt <= -4 || decpt > 16 {
        text.push_str(&digits[..1]);
        if digits.len() > 1 {
            text.push('.');
            text.push_str(&digits[1..]);
        }
        let shown = decpt - 1;
        text.push_str(&format!(
            "e{}{:02}",
            if shown < 0 { '-' } else { '+' },
            shown.abs()
        ));
    } else if decpt <= 0 {
        text.push_str("0.");
        text.push_str(&"0".repeat((-decpt) as usize));
        text.push_str(&digits);
    } else if decpt as usize >= digits.len() {
        text.push_str(&digits);
        text.push_str(&"0".repeat(decpt as usize - digits.len()));
        text.push_str(".0");
    } else {
        text.push_str(&digits[..decpt as usize]);
        text.push('.');
        text.push_str(&digits[decpt as usize..]);
    }
    text
}

pub fn config_sha256(value: &serde_json::Value) -> String {
    format!("{:x}", Sha256::digest(canonical_json(value).as_bytes()))
}

/// Canonical configuration object of the legacy method.
pub fn legacy_config_json(config: &LegacyGlobalClipConfig) -> serde_json::Value {
    serde_json::json!({
        "method": BackgroundMethod::LegacyGlobalClip.id(),
        "method_version": BackgroundMethod::LegacyGlobalClip.method_version(),
        "clip_sigma": config.clip_sigma,
        "clip_passes": config.clip_passes,
        "seed": config.seed,
    })
}

/// Canonical configuration object of the SEP background. The Python
/// reference builds the same object for the same parameters.
pub fn sep_background_config_json(params: &SepBackgroundParams) -> serde_json::Value {
    serde_json::json!({
        "method": BackgroundMethod::SepSpatialMesh.id(),
        "method_version": BackgroundMethod::SepSpatialMesh.method_version(),
        "library": "sep",
        "library_version": SEP_LIBRARY_VERSION,
        "bw": params.bw,
        "bh": params.bh,
        "fw": params.fw,
        "fh": params.fh,
        "fthresh": params.fthresh,
        "maskthresh": 0.0,
        "min_meshes_per_axis": params.min_meshes_per_axis,
    })
}

/// Canonical configuration object of SEP extraction.
pub fn sep_detection_config_json(params: &SepDetectionParams) -> serde_json::Value {
    serde_json::json!({
        "method": DetectionMethod::SepExtract.id(),
        "library": "sep",
        "library_version": SEP_LIBRARY_VERSION,
        "thresh_sigma": params.thresh_sigma,
        "noise": "background_rms_map",
        "minarea": params.minarea,
        "filter_type": params.filter_type,
        "filter_kernel": params.filter_kernel,
        "deblend_nthresh": params.deblend_nthresh,
        "deblend_cont": params.deblend_cont,
        "clean": params.clean,
        "clean_param": params.clean_param,
        "pixstack": params.pixstack,
        "accept_deblended_for_focus": params.accept_deblended_for_focus,
    })
}

/// Detection backbone that produced the candidate sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectionMethod {
    /// Custom 8-connected components above `max(B+5*max(sigma,1), B+8)` with
    /// the global level `B` and scale `sigma`; flux-weighted centroids of
    /// `max(I-B,0)`. No filtering, no deblending.
    LegacyConnectedComponents,
    /// SEP `sep_extract` on `I-B(x,y)` with the RMS map as noise.
    SepExtract,
}

impl DetectionMethod {
    pub fn id(self) -> &'static str {
        match self {
            DetectionMethod::LegacyConnectedComponents => "legacy_connected_components",
            DetectionMethod::SepExtract => "sep_extract",
        }
    }
}

/// How the per-star gates and HFR obtained their sky.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CropSkySource {
    /// Median and 1.4826*MAD of the whole measurement crop (legacy).
    LegacyCropMedian,
    /// The frame's spatial background and RMS maps evaluated in the crop.
    ParentBackgroundMap,
}

/// Provenance of the source catalog behind the quality measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectionProvenance {
    pub method: DetectionMethod,
    pub config_sha256: String,
    pub crop_sky_source: CropSkySource,
    /// `true` when the legacy detector replaced an unresolved or sub-1 ADU
    /// noise scale with its 1 ADU floor (a labelled heuristic, not a
    /// calibration).
    pub noise_floor_applied: bool,
    pub objects_detected: u32,
    /// SEP children produced by deblending (`SEP_OBJ_MERGED`).
    pub objects_deblended: u32,
    /// Objects SEP flagged as truncated at the image border.
    pub objects_truncated: u32,
}

/// The background evaluated where detection and measurement need it.
#[derive(Debug, Clone)]
pub enum BackgroundField {
    /// One level and scale for every pixel (legacy global estimate). `sigma`
    /// is the measured scale, possibly 0 when unresolved.
    Uniform { level: f32, sigma: f32 },
    /// Full-resolution maps from SEP's spline array evaluators, row-major on
    /// the analysis grid. Values at detector-masked pixels are interpolated
    /// from the mesh model, not measurements there; consumers that need
    /// evidence at a pixel must also check its mask.
    Map {
        level: Vec<f32>,
        rms: Vec<f32>,
        width: usize,
        height: usize,
    },
}

impl BackgroundField {
    pub fn level_at(&self, index: usize) -> f32 {
        match self {
            BackgroundField::Uniform { level, .. } => *level,
            BackgroundField::Map { level, .. } => level[index],
        }
    }

    pub fn rms_at(&self, index: usize) -> f32 {
        match self {
            BackgroundField::Uniform { sigma, .. } => *sigma,
            BackgroundField::Map { rms, .. } => rms[index],
        }
    }

    pub fn is_spatial(&self) -> bool {
        matches!(self, BackgroundField::Map { .. })
    }
}

/// One SEP-extracted source in analysis-pixel coordinates.
///
/// SEP 1.4.1 computes `x`, `y`, `a`, `b`, `theta` from moments weighted by
/// the *filtered* (convolved) residual, so they describe the smoothed image.
/// Consumers that need unfiltered shape (trailing) use `pixels`.
#[derive(Debug, Clone, PartialEq)]
pub struct CatalogSource {
    /// Barycentre of the filtered residual over the footprint (0-based).
    pub x: f32,
    pub y: f32,
    pub xmin: usize,
    pub xmax: usize,
    pub ymin: usize,
    pub ymax: usize,
    pub npix: u32,
    /// Unfiltered residual flux `sum(I-B)` over the footprint (SEP `flux`).
    pub flux: f32,
    /// Unfiltered peak residual `max(I-B)` (SEP `peak`).
    pub peak_residual: f32,
    /// Raw input value at the peak pixel (for legacy peak saturation gates).
    pub peak_raw: f32,
    pub a: f32,
    pub b: f32,
    /// Position angle of the major axis, radians, from +x toward +y.
    pub theta: f32,
    /// Raw SEP object flags.
    pub flags: u16,
    /// `SEP_OBJ_MERGED`: the object came out of deblending.
    pub deblended: bool,
    /// `SEP_OBJ_TRUNC`: the footprint touches the image border.
    pub truncated: bool,
    /// Row-major analysis-pixel indices of the object's detection footprint
    /// (SEP's per-object pixel list, after deblending).
    pub pixels: Vec<u32>,
}

/// Sources from one SEP extraction, shared by focus and trailing.
#[derive(Debug, Clone)]
pub struct SourceCatalog {
    pub provenance: DetectionProvenance,
    pub objects: Vec<CatalogSource>,
    /// SEP segmentation map (object index + 1 per pixel, 0 for sky) when
    /// `retain_segmentation` is on.
    pub segmentation: Option<Vec<i32>>,
}

/// Everything the background stage hands to detection and measurement.
#[derive(Debug, Clone)]
pub struct BackgroundAnalysis {
    pub result: BackgroundResult,
    /// `None` when the level is unavailable; detection does not run then.
    pub field: Option<BackgroundField>,
    /// Present for the SEP path, which detects together with the background.
    pub catalog: Option<SourceCatalog>,
    /// Why SEP extraction did not produce a catalog on the SEP path.
    pub catalog_error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_json_sorts_keys_recursively_and_hash_is_stable() {
        let value = serde_json::json!({"b": 1, "a": {"d": [1.0, 0.005], "c": true}});
        assert_eq!(
            canonical_json(&value),
            r#"{"a":{"c":true,"d":[1.0,0.005]},"b":1}"#
        );
        // Python: hashlib.sha256(json.dumps(v, sort_keys=True,
        // separators=(",", ":")).encode()).hexdigest()
        assert_eq!(
            config_sha256(&value),
            "7d20bb44973ccb470157278dd7cd020702ef30df26740d30d08329211e602ac4"
        );
    }

    #[test]
    fn default_sep_background_config_hash_matches_the_python_reference() {
        let config = sep_background_config_json(&SepBackgroundParams::default());
        assert_eq!(
            canonical_json(&config),
            r#"{"bh":64,"bw":64,"fh":3,"fthresh":0.0,"fw":3,"library":"sep","library_version":"1.4.1","maskthresh":0.0,"method":"sep_spatial_mesh","method_version":1,"min_meshes_per_axis":2}"#
        );
        assert_eq!(
            config_sha256(&config),
            "80eab0fbae4abf9f60f348adb7de3004067a3873cca29f996289e9dc14040b84"
        );
    }

    #[test]
    fn default_sep_detection_config_hash_matches_the_python_reference() {
        let config = sep_detection_config_json(&SepDetectionParams::default());
        assert_eq!(
            canonical_json(&config),
            r#"{"accept_deblended_for_focus":false,"clean":true,"clean_param":1.0,"deblend_cont":0.005,"deblend_nthresh":32,"filter_kernel":[[1.0,2.0,1.0],[2.0,4.0,2.0],[1.0,2.0,1.0]],"filter_type":"conv","library":"sep","library_version":"1.4.1","method":"sep_extract","minarea":5,"noise":"background_rms_map","pixstack":300000,"thresh_sigma":5.0}"#
        );
        assert_eq!(
            config_sha256(&config),
            "4f4499cda7067107c0fe6d64319684c13d1ec62f590b4bee4b346c1459dc99df"
        );
    }

    #[test]
    fn floats_and_strings_are_written_exactly_as_python_json_dumps() {
        // Reference strings produced by CPython 3.12 json.dumps(...,
        // sort_keys=True, separators=(",", ":")).
        let floats = serde_json::json!([
            1.2,
            1e-5,
            0.0001,
            1e16,
            1234567890123456.0,
            1.5e-7,
            -0.0,
            0.1,
            2.5e-310,
            1e300,
            123.456,
            5.0,
            0.005,
            64.0,
            -3.25e-12,
            1e22,
            9007199254740993.0,
            659904053142191.25,
            -20102787369635.5625
        ]);
        assert_eq!(
            canonical_json(&floats),
            "[1.2,1e-05,0.0001,1e+16,1234567890123456.0,1.5e-07,-0.0,0.1,2.5e-310,1e+300,\
             123.456,5.0,0.005,64.0,-3.25e-12,1e+22,9007199254740992.0,659904053142191.2,\
             -20102787369635.562]"
        );
        let strings =
            serde_json::json!({"z": "plain", "a": "tab\tq\"b\\e/\u{e9}\u{1F600}\u{1}\u{7f}"});
        assert_eq!(
            canonical_json(&strings),
            r#"{"a":"tab\tq\"b\\e/\u00e9\ud83d\ude00\u0001\u007f","z":"plain"}"#
        );
    }

    #[test]
    fn non_default_configs_hash_like_the_python_reference() {
        // Python: SepDetectionParams(thresh_sigma=1.2, deblend_cont=1e-5) and
        // SepBackgroundParams(fthresh=0.1); hashes from hashlib.sha256 over
        // json.dumps(..., sort_keys=True, separators=(",", ":")).
        let detection = SepDetectionParams {
            thresh_sigma: 1.2,
            deblend_cont: 1e-5,
            ..Default::default()
        };
        assert_eq!(
            config_sha256(&sep_detection_config_json(&detection)),
            "fd6c2088782a5017c1a69a18cf4f4aee73665b029927083266e664f5f1718554"
        );
        let background = SepBackgroundParams {
            fthresh: 0.1,
            ..Default::default()
        };
        assert_eq!(
            config_sha256(&sep_background_config_json(&background)),
            "1d87974ba87e717a55e01ee098d2892de8bc2f77807e56823f36265e6cbf48a1"
        );
    }

    #[test]
    fn unmeasured_level_and_noise_are_not_reported_as_values() {
        let result = BackgroundResult {
            schema_version: BACKGROUND_SCHEMA_VERSION,
            method: BackgroundMethod::LegacyGlobalClip,
            method_version: 1,
            library: None,
            library_version: None,
            requested_method: BackgroundMethod::LegacyGlobalClip,
            config: serde_json::json!({}),
            config_sha256: String::new(),
            level_status: LevelStatus::Measured,
            noise_status: NoiseStatus::Unresolved,
            reasons: vec![BackgroundReason::ZeroEmpiricalScatter],
            backend_error: None,
            units: "adu".into(),
            coordinate_frame: "analysis_pixel".into(),
            width: 1,
            height: 1,
            noise_source: NoiseSource::EmpiricalGlobalClip,
            support: BackgroundSupport {
                total_pixels: 1,
                usable_pixels: 1,
                clipped_sample_pixels: Some(1),
                masks: BTreeMap::new(),
                saturation_evidence: "unknown".into(),
            },
            level: Some(0.0),
            level_definition: String::new(),
            rms: None,
            rms_definition: String::new(),
            mesh: None,
            plane: None,
            plane_unavailable_reason: Some(BackgroundReason::NoSpatialModel),
            bright_pixel_significance: None,
            saturation_fraction: Some(0.0),
        };
        // A measured zero level is a value; unresolved noise is absent.
        assert_eq!(result.level(), Some(0.0));
        assert_eq!(result.rms(), None);
        let json = serde_json::to_value(&result).unwrap();
        assert!(json["rms"].is_null());
        assert_eq!(json["noise_status"], "unresolved");
    }
}

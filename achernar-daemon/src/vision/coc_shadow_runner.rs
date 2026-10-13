//! Native CPU inference for the hash-bound, three-class COC research bundle.
//!
//! This runner has no weather/control authority. Its outputs are source-domain
//! segmentation diagnostics; softmax scores are uncalibrated. RGB8 inputs must
//! already have their orientation applied and must be rendered images, not ADU.

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Mutex;

use ort::execution_providers::CPUExecutionProvider;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::tensor::TensorElementType;
use ort::value::{TensorRef, ValueType};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub const COC_IMAGE_SIZE: usize = 128;
const PIXELS: usize = COC_IMAGE_SIZE * COC_IMAGE_SIZE;
const CONTRACT: &str = "coc-segmentation-v1";
const REQUIRED_MEMBERS: [&str; 9] = [
    "model.onnx",
    "golden.npz",
    "parity.json",
    "preprocess.json",
    "postprocess.json",
    "quality.json",
    "training_run.json",
    "training_source_snapshot.zip",
    "supplemental_provenance.json",
];

#[derive(Clone, Debug, Serialize)]
pub struct CocDiagnosticSummary {
    pub contract_version: String,
    pub class_order: [&'static str; 3],
    pub status: String,
    pub reason: Option<String>,
    pub valid_region_pixels: usize,
    pub cloud_area_fraction_of_roi: Option<f64>,
    pub cloud_fraction_of_classified_sky: Option<f64>,
    pub contamination_area_fraction_of_roi: Option<f64>,
    pub mean_soft_cloud_share_of_sky: Option<f64>,
    pub probability_calibrated: bool,
    pub transparency: Option<f64>,
    pub five_class_condition: Option<String>,
    pub runtime_eligible: bool,
    pub promotion_status: &'static str,
}

pub struct CocShadowOutput {
    pub summary: CocDiagnosticSummary,
    /// Class-major [3,128,128] arrays; scores are uncalibrated softmax.
    pub logits: Vec<f32>,
    pub scores: Vec<f32>,
    pub classes: Vec<u8>,
    /// Interleaved RGB [128,128,3].
    pub overlay_rgb: Vec<u8>,
    pub input_rgb: Vec<u8>,
}

pub struct CocShadowRunner {
    session: Mutex<Session>,
    model_sha256: String,
    manifest_sha256: String,
}

impl CocShadowRunner {
    pub fn load_bundle(root: &Path) -> Result<Self, String> {
        let root = root.canonicalize().map_err(err)?;
        let manifest = verify_bundle(&root)?;
        // Bind inference to the same bytes whose digest is accepted; do not
        // reopen the graph after its final digest check.
        let graph = std::fs::read(root.join("model.onnx")).map_err(err)?;
        if format!("{:x}", Sha256::digest(&graph))
            != manifest["files_sha256"]["model.onnx"].as_str().unwrap()
        {
            return Err("COC graph changed while loading".into());
        }
        let session = Session::builder()
            .map_err(err)?
            .with_no_environment_execution_providers()
            .map_err(err)?
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(err)?
            .with_intra_threads(1)
            .map_err(err)?
            .with_inter_threads(1)
            .map_err(err)?
            .commit_from_memory(&graph)
            .map_err(err)?;
        if session.inputs.len() != 1
            || session.outputs.len() != 1
            || session.inputs[0].name != "image"
            || session.outputs[0].name != "logits"
            || !tensor_contract(&session.inputs[0].input_type)
            || !tensor_contract(&session.outputs[0].output_type)
        {
            return Err("COC graph must have FP32 image/logits tensors [batch,3,128,128]".into());
        }
        Ok(Self {
            session: Mutex::new(session),
            model_sha256: manifest["files_sha256"]["model.onnx"]
                .as_str()
                .unwrap()
                .into(),
            manifest_sha256: sha256_file(&root.join("manifest.json"))?,
        })
    }

    pub fn is_loaded(&self) -> bool {
        true
    }
    pub fn model_sha256(&self) -> &str {
        &self.model_sha256
    }
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    /// Accept exactly one normalized BCHW RGB image. Invalid inputs fail closed.
    pub fn run_preprocessed(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        if input.len() != 3 * PIXELS
            || input
                .iter()
                .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
        {
            return Err("COC input must be finite RGB FP32 [1,3,128,128] in [0,1]".into());
        }
        let tensor = TensorRef::from_array_view(([1, 3, COC_IMAGE_SIZE, COC_IMAGE_SIZE], input))
            .map_err(err)?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| "COC session mutex poisoned".to_string())?;
        let output = session.run(ort::inputs!["image" => tensor]).map_err(err)?;
        let (shape, logits) = output["logits"].try_extract_tensor::<f32>().map_err(err)?;
        if &**shape != [1, 3, COC_IMAGE_SIZE as i64, COC_IMAGE_SIZE as i64]
            || logits.len() != 3 * PIXELS
            || logits.iter().any(|x| !x.is_finite())
        {
            return Err("COC output must be finite FP32 [1,3,128,128]".into());
        }
        Ok(logits.to_vec())
    }

    pub fn predict_rgb8(
        &self,
        rgb: &[u8],
        width: u32,
        height: u32,
        roi_original: Option<&[u8]>,
    ) -> Result<CocShadowOutput, String> {
        let input_rgb = resize_rgb8(rgb, width, height)?;
        let input = rgb8_to_bchw(&input_rgb);
        let roi = roi_original
            .map(|r| resize_roi_nearest(r, width, height))
            .transpose()?;
        let logits = self.run_preprocessed(&input)?;
        let (mut summary, classes, scores) = summarize_logits(&logits, roi.as_deref(), 0.3)?;
        if let Some(region) = roi.as_deref() {
            if region.iter().any(|&v| v)
                && (0..3).all(|c| {
                    let mut min = f32::INFINITY;
                    let mut max = f32::NEG_INFINITY;
                    for p in 0..PIXELS {
                        if region[p] {
                            min = min.min(input[c * PIXELS + p]);
                            max = max.max(input[c * PIXELS + p]);
                        }
                    }
                    max - min <= 1e-6
                })
            {
                summary.status = "abstained".into();
                summary.reason = Some("uninformative_rgb_input".into());
                summary.cloud_area_fraction_of_roi = None;
                summary.cloud_fraction_of_classified_sky = None;
                summary.contamination_area_fraction_of_roi = None;
                summary.mean_soft_cloud_share_of_sky = None;
            }
        }
        let colors = [[50u8, 120, 220], [240, 240, 240], [230, 110, 40]];
        let mut overlay_rgb = vec![0; 3 * PIXELS];
        for p in 0..PIXELS {
            for c in 0..3 {
                overlay_rgb[3 * p + c] = (0.55 * input_rgb[3 * p + c] as f64
                    + 0.45 * colors[classes[p] as usize][c] as f64)
                    as u8;
            }
        }
        Ok(CocShadowOutput {
            summary,
            logits,
            scores,
            classes,
            overlay_rgb,
            input_rgb,
        })
    }
}

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn tensor_contract(value: &ValueType) -> bool {
    matches!(value, ValueType::Tensor { ty: TensorElementType::Float32, shape, .. }
        if &**shape == [-1, 3, COC_IMAGE_SIZE as i64, COC_IMAGE_SIZE as i64])
}

pub fn sha256_file(path: &Path) -> Result<String, String> {
    let mut f = File::open(path).map_err(err)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = f.read(&mut buffer).map_err(err)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Research manifests remain distinct from the daemon's promoted-model schema.
pub fn verify_bundle(root: &Path) -> Result<Value, String> {
    let root = root.canonicalize().map_err(err)?;
    let manifest: Value =
        serde_json::from_reader(File::open(root.join("manifest.json")).map_err(err)?)
            .map_err(err)?;
    if manifest["schema_version"] != 1
        || manifest["contract_version"] != CONTRACT
        || manifest["class_order"] != json!(["sky", "cloud", "contamination"])
        || manifest["runtime_eligible"] != false
        || manifest["promotion_status"] != "research"
        || manifest["family"] != "cloud_segmentation"
        || manifest["task"] != "cloud_segmentation"
        || manifest["framework"] != "onnx"
        || manifest["image_size"] != COC_IMAGE_SIZE
        || manifest["opset"] != 17
        || manifest["precision"]
            != json!({"weights":"FP32", "input":"FP32", "output":"FP32", "normalization_arithmetic":"FP64"})
        || manifest["input"] != json!({"image":["batch",3,128,128]})
        || manifest["output"] != json!({"logits":["batch",3,128,128]})
    {
        return Err("Unsupported COC research bundle contract".into());
    }
    let members = manifest["files_sha256"]
        .as_object()
        .ok_or("Missing COC integrity map")?;
    if REQUIRED_MEMBERS
        .iter()
        .any(|name| !members.contains_key(*name))
    {
        return Err("Incomplete COC integrity map".into());
    }
    for (member, digest) in members {
        let digest = digest.as_str().ok_or("Malformed COC member digest")?;
        if digest.len() != 64 || !digest.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err("Malformed COC member digest".into());
        }
        let path = root.join(member).canonicalize().map_err(err)?;
        if !path.starts_with(&root) || !path.is_file() || sha256_file(&path)? != digest {
            return Err(format!("COC member integrity failed: {member}"));
        }
    }
    let read_json = |name: &str| -> Result<Value, String> {
        serde_json::from_reader(File::open(root.join(name)).map_err(err)?).map_err(err)
    };
    let preprocess = read_json("preprocess.json")?;
    if preprocess
        != json!({"channels":"RGB", "contract_version":CONTRACT, "dtype":"float32",
        "exif_orientation":"must already be applied", "input":"rendered RGB uint8", "layout":"BCHW",
        "linear_ADU_input":false, "normalization":"divide by 255", "resize":"Pillow bilinear, square; categorical ROI nearest", "size":[128,128]})
    {
        return Err("Unsupported COC preprocessing contract".into());
    }
    let postprocess = read_json("postprocess.json")?;
    if postprocess["class_order"] != json!(["sky", "cloud", "contamination"])
        || postprocess["contamination_abstention_threshold"] != 0.3
        || postprocess["control_authority"] != false
        || postprocess["threshold_qualified"] != false
        || postprocess["pixel_scores"] != "uncalibrated softmax"
    {
        return Err("Unsupported COC postprocessing contract".into());
    }
    let parity = read_json("parity.json")?;
    if parity["final_test_used"] != false
        || parity["provider"] != "CPUExecutionProvider"
        || parity["normalization"]["normalization_arithmetic"] != "FP64 centered moments"
        || ["cases", "validation_cases"].iter().any(|key| {
            parity[*key].as_array().map_or(true, |cases| {
                cases.is_empty() || cases.iter().any(|c| c["passed"] != true)
            })
        })
    {
        return Err("COC bundle lacks passing research export parity".into());
    }
    Ok(manifest)
}

fn dimensions(width: u32, height: u32, channels: usize) -> Result<(usize, usize, usize), String> {
    if width == 0 || height == 0 || width > 20000 || height > 20000 {
        return Err("Invalid rendered COC image dimensions".into());
    }
    let (w, h) = (width as usize, height as usize);
    let len = w
        .checked_mul(h)
        .and_then(|v| v.checked_mul(channels))
        .ok_or("COC image size overflow")?;
    Ok((w, h, len))
}

/// Separable, antialiased bilinear resize with the Pillow RGB8 22-bit
/// coefficient/rounding contract. Horizontal intermediate pixels are uint8.
/// Formula reference: Pillow 12.3 src/libImaging/Resample.c (bilinear,
/// precompute_coeffs, normalize_coeffs_8bpc). This is independently implemented.
fn coefficients(input: usize) -> Vec<(usize, Vec<i64>)> {
    let scale = input as f64 / COC_IMAGE_SIZE as f64;
    let filter_scale = scale.max(1.0);
    (0..COC_IMAGE_SIZE)
        .map(|out| {
            let center = (out as f64 + 0.5) * scale;
            let begin = ((center - filter_scale + 0.5) as isize).max(0) as usize;
            let end = ((center + filter_scale + 0.5) as usize).min(input);
            let mut weights: Vec<f64> = (begin..end)
                .map(|i| (1.0 - ((i as f64 - center + 0.5) / filter_scale).abs()).max(0.0))
                .collect();
            let sum: f64 = weights.iter().sum();
            for w in &mut weights {
                *w /= sum;
            }
            (
                begin,
                weights
                    .into_iter()
                    .map(|v| (0.5 + v * ((1i64 << 22) as f64)) as i64)
                    .collect(),
            )
        })
        .collect()
}

fn resize_rgb8(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let (w, h, len) = dimensions(width, height, 3)?;
    if rgb.len() != len {
        return Err("COC RGB8 buffer length does not match geometry".into());
    }
    let horizontal = if w == COC_IMAGE_SIZE {
        rgb.to_vec()
    } else {
        let xweights = coefficients(w);
        let mut out = vec![0u8; h * COC_IMAGE_SIZE * 3];
        for y in 0..h {
            for (x, (start, weights)) in xweights.iter().enumerate() {
                for c in 0..3 {
                    let mut sum = 1i64 << 21;
                    for (i, weight) in weights.iter().enumerate() {
                        sum += rgb[(y * w + start + i) * 3 + c] as i64 * weight;
                    }
                    out[(y * COC_IMAGE_SIZE + x) * 3 + c] = (sum >> 22).clamp(0, 255) as u8;
                }
            }
        }
        out
    };
    if h == COC_IMAGE_SIZE {
        return Ok(horizontal);
    }
    let yweights = coefficients(h);
    let mut out = vec![0u8; PIXELS * 3];
    for (y, (start, weights)) in yweights.iter().enumerate() {
        for x in 0..COC_IMAGE_SIZE {
            for c in 0..3 {
                let mut sum = 1i64 << 21;
                for (i, weight) in weights.iter().enumerate() {
                    sum += horizontal[((start + i) * COC_IMAGE_SIZE + x) * 3 + c] as i64 * weight;
                }
                out[(y * COC_IMAGE_SIZE + x) * 3 + c] = (sum >> 22).clamp(0, 255) as u8;
            }
        }
    }
    Ok(out)
}

fn rgb8_to_bchw(rgb: &[u8]) -> Vec<f32> {
    let mut output = vec![0f32; 3 * PIXELS];
    for p in 0..PIXELS {
        for c in 0..3 {
            output[c * PIXELS + p] = rgb[3 * p + c] as f32 / 255.0;
        }
    }
    output
}

pub fn preprocess_rgb8(rgb: &[u8], width: u32, height: u32) -> Result<Vec<f32>, String> {
    Ok(rgb8_to_bchw(&resize_rgb8(rgb, width, height)?))
}

pub fn resize_roi_nearest(roi: &[u8], width: u32, height: u32) -> Result<Vec<bool>, String> {
    let (w, h, len) = dimensions(width, height, 1)?;
    if roi.len() != len || roi.iter().any(|v| !matches!(v, 0 | 1 | 255)) {
        return Err("COC ROI must be binary and match original image geometry".into());
    }
    let mut output = vec![false; PIXELS];
    for y in 0..COC_IMAGE_SIZE {
        for x in 0..COC_IMAGE_SIZE {
            let ix = (((x as f64 + 0.5) * w as f64 / COC_IMAGE_SIZE as f64) as usize).min(w - 1);
            let iy = (((y as f64 + 0.5) * h as f64 / COC_IMAGE_SIZE as f64) as usize).min(h - 1);
            output[y * COC_IMAGE_SIZE + x] = roi[iy * w + ix] != 0;
        }
    }
    Ok(output)
}

pub fn summarize_logits(
    logits: &[f32],
    region: Option<&[bool]>,
    max_contamination: f64,
) -> Result<(CocDiagnosticSummary, Vec<u8>, Vec<f32>), String> {
    if logits.len() != 3 * PIXELS
        || logits.iter().any(|v| !v.is_finite())
        || !max_contamination.is_finite()
        || !(0.0..=1.0).contains(&max_contamination)
    {
        return Err("Invalid COC logits/diagnostic threshold".into());
    }
    if region.map_or(false, |r| r.len() != PIXELS) {
        return Err("COC region must match model image geometry".into());
    }
    let mut summary = CocDiagnosticSummary {
        contract_version: CONTRACT.into(),
        class_order: ["sky", "cloud", "contamination"],
        status: "abstained".into(),
        reason: Some("valid_region_required".into()),
        valid_region_pixels: 0,
        cloud_area_fraction_of_roi: None,
        cloud_fraction_of_classified_sky: None,
        contamination_area_fraction_of_roi: None,
        mean_soft_cloud_share_of_sky: None,
        probability_calibrated: false,
        transparency: None,
        five_class_condition: None,
        runtime_eligible: false,
        promotion_status: "research",
    };
    let mut classes = vec![0u8; PIXELS];
    let mut scores = vec![0f32; 3 * PIXELS];
    let mut counts = [0usize; 3];
    let (mut cloud_soft, mut sky_soft) = (0.0, 0.0);
    for p in 0..PIXELS {
        let max = (logits[p] as f64)
            .max(logits[PIXELS + p] as f64)
            .max(logits[2 * PIXELS + p] as f64);
        let s = [
            (logits[p] as f64 - max).exp(),
            (logits[PIXELS + p] as f64 - max).exp(),
            (logits[2 * PIXELS + p] as f64 - max).exp(),
        ];
        let total: f64 = s.iter().sum();
        let mut class = 0usize;
        for c in 0..3 {
            scores[c * PIXELS + p] = (s[c] / total) as f32;
            if s[c] > s[class] {
                class = c;
            }
        }
        classes[p] = class as u8;
        if region.map_or(false, |r| r[p]) {
            counts[class] += 1;
            cloud_soft += s[1] / total;
            sky_soft += (s[0] + s[1]) / total;
        }
    }
    if region.is_none() {
        return Ok((summary, classes, scores));
    }
    let count: usize = counts.iter().sum();
    summary.valid_region_pixels = count;
    if count == 0 {
        summary.reason = Some("empty_valid_region".into());
        return Ok((summary, classes, scores));
    }
    let contamination = counts[2] as f64 / count as f64;
    summary.cloud_area_fraction_of_roi = Some(counts[1] as f64 / count as f64);
    summary.contamination_area_fraction_of_roi = Some(contamination);
    if counts[0] + counts[1] == 0 {
        summary.reason = Some("no_classified_sky".into());
    } else if contamination > max_contamination {
        summary.reason = Some("contamination_exceeds_diagnostic_threshold".into());
    } else {
        summary.status = "research_diagnostic".into();
        summary.reason = None;
        summary.cloud_fraction_of_classified_sky =
            Some(counts[1] as f64 / (counts[0] + counts[1]) as f64);
        summary.mean_soft_cloud_share_of_sky = Some(cloud_soft / sky_soft);
    }
    Ok((summary, classes, scores))
}

#[cfg(test)]
#[path = "coc_shadow_integration_tests.rs"]
mod worker_integration_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires locally downloaded hash-bound teacher bundle"]
    fn integrity_check_rejects_tamper_unsafe_paths_and_promotion() {
        let real = std::env::var_os("COC_TEST_BUNDLE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../ml/artifacts/coc/human-teacher-tiny-v1")
            });
        // Integration fixture is local-only: this check does not certify a
        // model's scientific quality and does not execute any camera command.
        assert!(
            real.join("manifest.json").exists(),
            "local teacher bundle required"
        );
        let temp =
            std::env::temp_dir().join(format!("achernar-coc-integrity-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&temp).unwrap();
        for member in REQUIRED_MEMBERS {
            std::fs::copy(real.join(member), temp.join(member)).unwrap();
        }
        let original = std::fs::read(real.join("manifest.json")).unwrap();
        std::fs::write(temp.join("manifest.json"), &original).unwrap();
        assert!(verify_bundle(&temp).is_ok());
        std::fs::write(temp.join("model.onnx"), b"corrupted").unwrap();
        assert!(verify_bundle(&temp).is_err());
        std::fs::copy(real.join("model.onnx"), temp.join("model.onnx")).unwrap();
        let mut manifest: Value = serde_json::from_slice(&original).unwrap();
        manifest["runtime_eligible"] = json!(true);
        std::fs::write(
            temp.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        assert!(verify_bundle(&temp).is_err());
        manifest["runtime_eligible"] = json!(false);
        manifest["files_sha256"]["../outside.bin"] = json!("0".repeat(64));
        std::fs::write(
            temp.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        assert!(verify_bundle(&temp).is_err());
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn rgb_and_roi_resize_match_pillow_12_3_fixtures() {
        // Generated independently with Pillow 12.3.0 RGB bilinear / L nearest.
        // The dimensions exercise upsampling, odd rectangular downsampling,
        // identity, full-HD and a DSLR-sized image. Hash the complete output.
        let cases = [
            (
                1,
                1,
                "4ff2bedbf4fd87e1730e4b61227629d159a5ad4cf38fb77cdc9d4b28ae0f0024",
                "111ce3c2a38d83a2e4706bde4abddd509d7f8248116c6832b06745bdc349e09f",
            ),
            (
                7,
                11,
                "1633577c022e2b7d93453585f3f67d39fb4b6fbca8e374c16c119ef1897c10e0",
                "9af88266b417c567673b9bbec1c67a22b958687f2989aa6fd013fae27f275704",
            ),
            (
                128,
                128,
                "640bdcc15f1f8476c459752574383139e9fb1a22dc0cdfdaa16df08c565f9d85",
                "32f46a4f557071afc981472bb29314b24a444187a5b7a7486c204e03c57bb221",
            ),
            (
                257,
                65,
                "0789e61022628bdf7bf1b8e026bac0d5576b00c96db44bdfaa469a125d4a75ec",
                "3608b1367598a2670783d0a9779adadaf80ee844871381840284c84cb05e5418",
            ),
            (
                512,
                512,
                "12b734f558a6170ef56c5db2aeca32d8d979c610659e27e0948dd50837af096a",
                "4088ac24467fb5765abeef26d9472e1adca9028400755fce77fdb53b4b2efdb1",
            ),
            (
                1920,
                1080,
                "3ee47323a35318476bf903a3ece27f3c08a22dac0f0cdfb6ea026c3472bc3e91",
                "41b9bdc9cfe0f730e4e166e94309979a803e00d590d19140d40e53c94ff7cbcd",
            ),
            (
                6000,
                4000,
                "c9c4dedc5fc7710a4e20acb060dd0313481fa082414ce6ca77d5126ad72cc2ed",
                "52b9e36e1d58397ecebcb0fefe6759bec417bd89125a1ded56e937884b905798",
            ),
        ];
        for (w, h, rgb_hash, roi_hash) in cases {
            let mut rgb = Vec::with_capacity(w * h * 3);
            let mut roi = Vec::with_capacity(w * h);
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        rgb.push(((x * 17 + y * 29 + c * 47 + (x * y) % 251) % 256) as u8);
                    }
                    roi.push(u8::from((x + 2 * y) % 7 < 3));
                }
            }
            let output = resize_rgb8(&rgb, w as u32, h as u32).unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(&output)),
                rgb_hash,
                "RGB {w}x{h}"
            );
            let output: Vec<u8> = resize_roi_nearest(&roi, w as u32, h as u32)
                .unwrap()
                .into_iter()
                .map(u8::from)
                .collect();
            assert_eq!(
                format!("{:x}", Sha256::digest(&output)),
                roi_hash,
                "ROI {w}x{h}"
            );
        }
    }

    #[test]
    fn preprocessing_rejects_invalid_geometry_and_is_rgb_channel_first() {
        assert!(preprocess_rgb8(&[], 0, 1).is_err());
        assert!(preprocess_rgb8(&[0], 1, 1).is_err());
        let x = preprocess_rgb8(&[10, 20, 255], 1, 1).unwrap();
        assert!(x[..PIXELS].iter().all(|v| *v == 10.0f32 / 255.0));
        assert!(x[PIXELS..2 * PIXELS].iter().all(|v| *v == 20.0f32 / 255.0));
        assert!(x[2 * PIXELS..].iter().all(|v| *v == 1.0));
        assert!(resize_roi_nearest(&[2], 1, 1).is_err());
    }

    #[test]
    fn summary_abstains_without_independent_region_and_on_contamination() {
        let mut logits = vec![0.0f32; PIXELS * 3];
        logits[PIXELS..2 * PIXELS].fill(2.0);
        let (s, c, _) = summarize_logits(&logits, None, 0.3).unwrap();
        assert_eq!(s.reason.as_deref(), Some("valid_region_required"));
        assert_eq!(s.cloud_area_fraction_of_roi, None);
        assert!(c.iter().all(|x| *x == 1));
        let region = vec![true; PIXELS];
        let (s, _, _) = summarize_logits(&logits, Some(&region), 0.3).unwrap();
        assert_eq!(s.status, "research_diagnostic");
        assert_eq!(s.cloud_fraction_of_classified_sky, Some(1.0));
        logits[2 * PIXELS..].fill(3.0);
        assert_eq!(
            summarize_logits(&logits, Some(&region), 0.3)
                .unwrap()
                .0
                .reason
                .as_deref(),
            Some("no_classified_sky")
        );
        assert_eq!(
            summarize_logits(&logits, Some(&vec![false; PIXELS]), 0.3)
                .unwrap()
                .0
                .reason
                .as_deref(),
            Some("empty_valid_region")
        );
        logits[0] = f32::NAN;
        assert!(summarize_logits(&logits, Some(&region), 0.3).is_err());
    }

    #[test]
    #[ignore = "requires hash-bound local trained bundle and Python golden FP32 fixture paths"]
    fn real_onnx_bundle_matches_python_golden() {
        let bundle = std::env::var("COC_TEST_BUNDLE").expect("COC_TEST_BUNDLE");
        let input = read_fp32(&std::env::var("COC_TEST_INPUT").expect("COC_TEST_INPUT"));
        let expected = read_fp32(&std::env::var("COC_TEST_LOGITS").expect("COC_TEST_LOGITS"));
        let runner = CocShadowRunner::load_bundle(Path::new(&bundle)).unwrap();
        let got = runner.run_preprocessed(&input).unwrap();
        assert_eq!(got.len(), expected.len());
        let max_error = got
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            got.iter()
                .zip(&expected)
                .all(|(a, b)| (a - b).abs() <= 1e-4 + 1e-4 * b.abs()),
            "max absolute error {max_error}"
        );
        println!(
            "Real COC ONNX golden parity: {} logits, max absolute error {max_error}; model {}",
            got.len(),
            runner.model_sha256()
        );
        assert!(runner
            .run_preprocessed(&vec![f32::NAN; 3 * PIXELS])
            .is_err());
    }

    #[test]
    #[ignore = "requires local trained bundle and independent Python RGB/ROI/summary fixture paths"]
    fn native_diagnostics_match_python_fixture_and_uniform_abstention() {
        let bundle = std::env::var("COC_TEST_BUNDLE").expect("COC_TEST_BUNDLE");
        let rgb = std::fs::read(std::env::var("COC_TEST_RGB").expect("COC_TEST_RGB")).unwrap();
        let roi = std::fs::read(std::env::var("COC_TEST_ROI").expect("COC_TEST_ROI")).unwrap();
        let expected: Value = serde_json::from_slice(
            &std::fs::read(std::env::var("COC_TEST_SUMMARY").expect("COC_TEST_SUMMARY")).unwrap(),
        )
        .unwrap();
        let runner = CocShadowRunner::load_bundle(Path::new(&bundle)).unwrap();
        let output = runner.predict_rgb8(&rgb, 128, 128, Some(&roi)).unwrap();
        let actual = serde_json::to_value(&output.summary).unwrap();
        for (key, value) in expected["summary"].as_object().unwrap() {
            if let Some(value) = value.as_f64() {
                assert!(
                    (actual[key].as_f64().unwrap() - value).abs() < 1e-6,
                    "{key}"
                );
            } else {
                assert_eq!(&actual[key], value, "{key}");
            }
        }
        assert_eq!(
            format!("{:x}", Sha256::digest(&output.classes)),
            expected["classes_sha256"].as_str().unwrap()
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(&output.overlay_rgb)),
            expected["overlay_sha256"].as_str().unwrap()
        );
        assert_eq!(output.input_rgb, rgb);
        let uniform = runner
            .predict_rgb8(&vec![0; 3 * PIXELS], 128, 128, Some(&roi))
            .unwrap();
        assert_eq!(
            uniform.summary.reason.as_deref(),
            Some("uninformative_rgb_input")
        );
        assert_eq!(uniform.summary.cloud_area_fraction_of_roi, None);
        let no_roi = runner.predict_rgb8(&rgb, 128, 128, None).unwrap();
        assert_eq!(
            no_roi.summary.reason.as_deref(),
            Some("valid_region_required")
        );
    }

    fn read_fp32(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(bytes.len() % 4, 0);
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
}

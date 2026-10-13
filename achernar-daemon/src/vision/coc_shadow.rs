//! Rendered-preview COC research worker. Never publishes control observations.
use std::collections::VecDeque;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use base64::{engine::general_purpose::STANDARD, Engine};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::hardware::camera::FrameData;
use crate::hardware::ols_ffi::{OlsBayerType, OlsStreamType};
use crate::state::AppState;
use super::coc_shadow_runner::{CocShadowOutput, CocShadowRunner, COC_IMAGE_SIZE};

const FRESH_SECONDS: f64 = 5.0;
const HISTORY: usize = 8;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoiRect {
    pub left: f64,
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
}

impl RoiRect {
    pub fn validate(&self) -> Result<(), String> {
        if ![self.left, self.top, self.right, self.bottom].iter().all(|x| x.is_finite())
            || self.left < 0.0 || self.top < 0.0 || self.right > 1.0 || self.bottom > 1.0
            || self.right <= self.left || self.bottom <= self.top {
            return Err("ROI must be a nonempty independently selected rectangle inside the image".into());
        }
        Ok(())
    }
    fn mask(&self, width: u32, height: u32) -> Vec<u8> {
        (0..height).flat_map(|y| (0..width).map(move |x| {
            let (x, y) = ((x as f64 + 0.5) / width as f64, (y as f64 + 0.5) / height as f64);
            u8::from(x >= self.left && x < self.right && y >= self.top && y < self.bottom)
        })).collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FrameContext {
    pub generation: u64,
    pub connection_id: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigureRequest {
    pub enabled: bool,
    pub model: String,
    pub roi: Option<RoiRect>,
    pub context: Option<FrameContext>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SaveRequest {
    pub sample_id: String,
    #[serde(default)]
    pub notes: String,
}

struct Model {
    id: String,
    runner: Option<Arc<CocShadowRunner>>,
    error: Option<String>,
}

#[derive(Clone)]
struct Sample {
    id: String,
    context: FrameContext,
    source_timestamp: f64,
    payload: Value,
    files: Vec<(String, Vec<u8>)>,
}

struct Inner {
    enabled: bool,
    model: String,
    roi: Option<RoiRect>,
    roi_context: Option<FrameContext>,
    revision: u64,
    history: VecDeque<Arc<Sample>>,
    reason: Option<String>,
    frozen: Option<(Arc<Sample>, Instant)>,
}

pub struct CocShadowService {
    models: Vec<Model>,
    inner: Mutex<Inner>,
    root: PathBuf,
    session_id: String,
}

impl CocShadowService {
    pub fn from_env(data_dir: &Path) -> Self {
        let models = [("teacher", "ACHERNAR_COC_BUNDLE"), ("human_only", "ACHERNAR_COC_BASELINE_BUNDLE")]
            .into_iter().map(|(id, key)| {
                let bundle = std::env::var_os(key).map(PathBuf::from);
                let loaded = bundle.as_deref().map(CocShadowRunner::load_bundle);
                let (runner, error) = match loaded {
                    Some(Ok(runner)) => (Some(Arc::new(runner)), None),
                    Some(Err(error)) => (None, Some(error)),
                    None => (None, Some(format!("{key} is not configured"))),
                };
                if let Some(error) = &error { tracing::warn!("COC shadow {id}: {error}"); }
                Model { id: id.into(), runner, error }
            }).collect();
        let absolute_data = if data_dir.is_absolute() {data_dir.to_path_buf()} else {std::env::current_dir().unwrap_or_else(|_|PathBuf::from(".")).join(data_dir)};
        Self { models, root: absolute_data.join("coc-shadow"), session_id: uuid::Uuid::new_v4().to_string(),
            inner: Mutex::new(Inner { enabled: false, model: "teacher".into(), roi: None, roi_context: None,
                revision: 0, history: VecDeque::new(), reason: None, frozen: None }) }
    }

    pub fn configure(&self, request: ConfigureRequest, current: Option<&FrameContext>) -> Result<(), String> {
        let model = self.models.iter().find(|m| m.id == request.model).ok_or("Unknown COC model")?;
        if request.enabled && model.runner.is_none() {
            return Err(model.error.clone().unwrap_or_else(|| "Model unavailable".into()));
        }
        if let Some(roi) = request.roi {
            roi.validate()?;
            if request.context.as_ref() != current || current.is_none() {
                return Err("ROI belongs to an old camera or image geometry; select it again".into());
            }
        }
        let mut inner = self.inner.lock().unwrap();
        inner.enabled = request.enabled;
        inner.model = request.model;
        inner.roi = request.roi;
        inner.roi_context = request.context.filter(|_| inner.roi.is_some());
        inner.revision += 1;
        inner.history.clear();
        inner.frozen = None;
        inner.reason = None;
        Ok(())
    }

    pub fn snapshot(&self, current: Option<&FrameContext>, now: f64) -> Value {
        let inner = self.inner.lock().unwrap();
        let latest = inner.history.back().filter(|sample| {
            inner.enabled && Some(&sample.context) == current && source_fresh(sample.source_timestamp, now)
        });
        let (status, reason) = if !inner.enabled { ("disabled", Some("Enable shadow inference to inspect preview frames".to_string())) }
            else if current.is_none() { ("waiting_for_frame", Some("Connect a camera and start rendered RGB live view".into())) }
            else if latest.is_some() { ("ready", None) }
            else if let Some(reason) = &inner.reason { ("unavailable", Some(reason.clone())) }
            else { ("waiting_for_frame", Some("Waiting for a fresh supported preview frame".into())) };
        json!({"contract_version":"coc-segmentation-v1", "shadow_only":true,"runtime_eligible":false,
            "enabled":inner.enabled,"model":inner.model,"status":status,"reason":reason,
            "models":self.models.iter().map(|m| json!({"id":m.id,"loaded":m.runner.is_some(),
                "model_sha256":m.runner.as_ref().map(|r|r.model_sha256()),"error":m.error})).collect::<Vec<_>>(),
            "latest":latest.map(|sample| &sample.payload),"roi":inner.roi,
            "roi_valid_for_current_camera":inner.roi.is_some() && inner.roi_context.as_ref()==current})
    }

    pub fn freeze(&self, sample_id: &str, current: Option<&FrameContext>) -> Result<Value, String> {
        let mut inner = self.inner.lock().unwrap();
        if !inner.enabled { return Err("Shadow inference is disabled".into()); }
        let now=chrono::Utc::now().timestamp_millis() as f64/1000.0;
        let sample=inner.history.iter().find(|s|s.id==sample_id && Some(&s.context)==current
            && source_fresh(s.source_timestamp,now)).cloned().ok_or("Frame expired or camera changed; freeze a current frame")?;
        inner.frozen=Some((sample.clone(),Instant::now()));
        Ok(sample.payload.clone())
    }

    pub fn save(&self, request: SaveRequest, state: &AppState) -> Result<Value, String> {
        // Linearize acceptance against disconnect/restart, then release camera
        // locks before durable I/O. A later disconnect cannot change these bytes.
        let sample = {
            let _barrier=state.camera_observation_barrier.lock().unwrap();
            let current=current_context(state);
            let inner=self.inner.lock().unwrap();
            if !inner.enabled { return Err("Shadow inference is disabled".into()); }
            inner.frozen.as_ref().filter(|(s,when)|s.id==request.sample_id && Some(&s.context)==current.as_ref()
                && when.elapsed() <= Duration::from_secs(120)).map(|(s,_)|s.clone())
                .ok_or("Frozen frame expired or camera changed; freeze a new review frame")?
        };
        self.save_sample(request,&sample,&state.rig_id)
    }

    fn save_sample(&self, request: SaveRequest, sample: &Sample, rig_id: &str) -> Result<Value, String> {
        if request.notes.len() > 4000 { return Err("Review notes exceed 4000 bytes".into()); }
        let destination = self.root.join(&self.session_id).join(&sample.id);
        let staging = destination.with_extension("partial");
        std::fs::create_dir_all(staging.parent().ok_or("Missing sample parent")?).map_err(|e|e.to_string())?;
        std::fs::create_dir(&staging).map_err(|e| format!("Sample already saved or staging exists: {e}"))?;
        let mut hashes = serde_json::Map::new();
        for (name, bytes) in &sample.files {
            durable_write(&staging.join(name), bytes)?;
            hashes.insert(name.clone(), json!({"sha256":format!("{:x}",Sha256::digest(bytes)),"bytes":bytes.len()}));
        }
        let manifest = json!({"schema_version":1,"task":"coc_local_preview_review","sample_id":sample.id,
            "session_id":self.session_id,"rig_id":rig_id,"saved_at":chrono::Utc::now().to_rfc3339(),
            "sample":sample.payload,"files":hashes,"operator_notes":request.notes,
            "label_type":"unadjudicated_review","training_approved":false,"runtime_eligible":false,
            "independent_night_id":Value::Null,"local_camera_qualified":false,
            "limitations":["Rendered preview diagnostic; not a linear RAW/science exposure or pixel-level human ground truth.",
                "Session IDs do not prove independent nights. Preserve unknowns and obtain blinded/adjudicated labels before evaluation."]});
        durable_write(&staging.join("manifest.json"), &serde_json::to_vec_pretty(&manifest).map_err(|e|e.to_string())?)?;
        std::fs::File::open(&staging).and_then(|f|f.sync_all()).map_err(|e|e.to_string())?;
        std::fs::rename(&staging,&destination).map_err(|e|e.to_string())?;
        std::fs::File::open(destination.parent().unwrap()).and_then(|f|f.sync_all()).map_err(|e|e.to_string())?;
        Ok(json!({"status":"saved","sample_id":sample.id,"path":destination,"training_approved":false}))
    }
}

fn durable_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(path).map_err(|e|e.to_string())?;
    file.write_all(bytes).and_then(|_|file.sync_all()).map_err(|e|e.to_string())
}

pub fn source_fresh(timestamp: f64, now: f64) -> bool {
    timestamp.is_finite() && now.is_finite() && now-timestamp >= -1.0 && now-timestamp <= FRESH_SECONDS
}

pub fn current_context(state: &AppState) -> Option<FrameContext> {
    let runtime = state.camera_runtime.read().unwrap().clone();
    if !runtime.streaming { return None; }
    let id = runtime.connection_id?;
    let latest = state.latest_frame.lock().unwrap();
    let frame = latest.as_ref()?;
    if frame.stream_generation != state.current_camera_generation() || frame.width <= 0 || frame.height <= 0 {
        return None;
    }
    Some(FrameContext { generation:frame.stream_generation, connection_id:id,
        width:frame.width as u32,height:frame.height as u32 })
}

fn png(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let image = image::RgbImage::from_raw(width,height,rgb.to_vec()).ok_or("Invalid RGB geometry")?;
    let mut bytes = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image).write_to(&mut bytes,image::ImageOutputFormat::Png).map_err(|e|e.to_string())?;
    Ok(bytes.into_inner())
}

fn class_png(classes: &[u8]) -> Result<Vec<u8>, String> {
    let image = image::GrayImage::from_raw(COC_IMAGE_SIZE as u32,COC_IMAGE_SIZE as u32,classes.to_vec()).ok_or("Invalid class geometry")?;
    let mut bytes = Cursor::new(Vec::new());
    image::DynamicImage::ImageLuma8(image).write_to(&mut bytes,image::ImageOutputFormat::Png).map_err(|e|e.to_string())?;
    Ok(bytes.into_inner())
}

// The preview and independent ROI use the stored pixel geometry. Non-identity
// EXIF display orientation must not silently rotate only the operator's view.
fn validate_jpeg_orientation(bytes: &[u8]) -> Result<(), String> {
    let invalid = || "JPEG EXIF orientation is malformed or non-identity; use an unrotated rendered preview".to_string();
    if bytes.get(..2) != Some(&[0xff, 0xd8]) { return Err(invalid()); }
    let mut offset = 2usize;
    while offset < bytes.len() {
        if bytes[offset] != 0xff { return Err(invalid()); }
        while bytes.get(offset) == Some(&0xff) { offset += 1; }
        let marker = *bytes.get(offset).ok_or_else(invalid)?;
        offset += 1;
        if marker == 0xda || marker == 0xd9 { return Ok(()); }
        if marker == 0x01 || (0xd0..=0xd7).contains(&marker) { continue; }
        let len_bytes = bytes.get(offset..offset+2).ok_or_else(invalid)?;
        let length = u16::from_be_bytes([len_bytes[0],len_bytes[1]]) as usize;
        if length < 2 { return Err(invalid()); }
        let segment = bytes.get(offset+2..offset+length).ok_or_else(invalid)?;
        if marker == 0xe1 && segment.starts_with(b"Exif\0\0") {
            let tiff = &segment[6..];
            let little = match tiff.get(..2) { Some(b"II")=>true, Some(b"MM")=>false, _=>return Err(invalid()) };
            let short = |at: usize| -> Result<u16,String> {
                let b=tiff.get(at..at+2).ok_or_else(invalid)?;
                Ok(if little {u16::from_le_bytes([b[0],b[1]])} else {u16::from_be_bytes([b[0],b[1]])})
            };
            let long = |at: usize| -> Result<u32,String> {
                let b: [u8;4]=tiff.get(at..at+4).ok_or_else(invalid)?.try_into().map_err(|_|invalid())?;
                Ok(if little {u32::from_le_bytes(b)} else {u32::from_be_bytes(b)})
            };
            if short(2)? != 42 { return Err(invalid()); }
            let ifd=long(4)? as usize;
            if ifd < 8 { return Err(invalid()); }
            let count=short(ifd)? as usize;
            for entry in 0..count {
                let at=ifd+2+12*entry;
                tiff.get(at..at+12).ok_or_else(invalid)?;
                if short(at)?==0x112 && (short(at+2)?!=3 || long(at+4)?!=1 || short(at+8)?!=1) {
                    return Err(invalid());
                }
            }
        }
        offset += length;
    }
    Err(invalid())
}

fn rendered_rgb(frame: &FrameData) -> Result<(Vec<u8>,u32,u32),String> {
    if frame.width<=0 || frame.height<=0 { return Err("Invalid preview geometry".into()); }
    if frame.sensor_descriptor.is_some() || frame.bayer != OlsBayerType::Na {
        return Err("COC accepts rendered RGB preview, not sensor ADU/Bayer data".into());
    }
    match frame.format {
        OlsStreamType::Mjpeg => {
            validate_jpeg_orientation(&frame.data)?;
            let image = image::load_from_memory_with_format(&frame.data,image::ImageFormat::Jpeg).map_err(|e|e.to_string())?.to_rgb8();
            if image.width()!=frame.width as u32 || image.height()!=frame.height as u32 { return Err("JPEG dimensions differ from frame metadata".into()); }
            Ok((image.into_raw(),frame.width as u32,frame.height as u32))
        }
        OlsStreamType::Rgb24 if frame.data.len()==frame.width as usize*frame.height as usize*3 =>
            Ok((frame.data.clone(),frame.width as u32,frame.height as u32)),
        _ => Err("Unsupported COC input: use MJPEG or rendered RGB24 live view".into()),
    }
}

fn build_sample(frame: &FrameData, context: FrameContext, output: CocShadowOutput, roi: Option<RoiRect>,
    source_rgb: &[u8], model: &str, model_sha: &str, manifest_sha: &str, elapsed: f64) -> Result<Sample,String> {
    let size = COC_IMAGE_SIZE as u32;
    let input = png(&output.input_rgb,size,size)?;
    let overlay = png(&output.overlay_rgb,size,size)?;
    let classes = class_png(&output.classes)?;
    let id = uuid::Uuid::new_v4().to_string();
    let payload = json!({"sample_id":id,"context":context,"frame_counter":frame.frame_counter,
        "source_timestamp":frame.unix_timestamp,"received_at":chrono::Utc::now().to_rfc3339(),
        "source_format":format!("{:?}",frame.format),"inference_ms":elapsed,"model":model,"model_sha256":model_sha,
        "bundle_manifest_sha256":manifest_sha,
        "summary":output.summary,"roi":roi,"image_size":size,
        "input_png_base64":STANDARD.encode(&input),"overlay_png_base64":STANDARD.encode(&overlay)});
    let mut logits = Vec::with_capacity(output.logits.len()*4);
    for x in output.logits { logits.extend(x.to_le_bytes()); }
    let mut files = vec![("source_rgb.png".into(),png(source_rgb,context.width,context.height)?),
        ("model_input.png".into(),input),("overlay.png".into(),overlay),("classes.png".into(),classes),
        ("logits.f32le".into(),logits),
        ("logits.json".into(),serde_json::to_vec(&json!({"shape":[1,3,size,size],"dtype":"float32","byte_order":"little","class_order":["sky","cloud","contamination"]})).unwrap())];
    if frame.format==OlsStreamType::Mjpeg { files.push(("source_preview.jpg".into(),frame.data.clone())); }
    if let Some(roi) = roi {
        let mask=image::GrayImage::from_raw(context.width,context.height,roi.mask(context.width,context.height)).unwrap();
        let mut bytes=Cursor::new(Vec::new());
        image::DynamicImage::ImageLuma8(mask).write_to(&mut bytes,image::ImageOutputFormat::Png).map_err(|e|e.to_string())?;
        files.push(("independent_roi.png".into(),bytes.into_inner()));
    }
    Ok(Sample {id,context,source_timestamp:frame.unix_timestamp,payload,files})
}

pub fn spawn(state: AppState) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut tick=tokio::time::interval(Duration::from_secs(1));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut last=None;
        loop {
            tick.tick().await;
            let service=state.coc_shadow.clone();
            let Some(context)=current_context(&state) else { last=None; continue; };
            let Some(frame)=state.latest_frame.lock().unwrap().clone() else { continue; };
            if frame.stream_generation!=context.generation || frame.width as u32!=context.width || frame.height as u32!=context.height { continue; }
            let (revision,model_id,roi)={
                let mut inner=service.inner.lock().unwrap();
                if !inner.enabled { continue; }
                if inner.roi_context.as_ref()!=Some(&context) && inner.roi.is_some() {
                    inner.roi=None; inner.roi_context=None; inner.revision+=1; inner.history.clear(); inner.frozen=None;
                }
                (inner.revision,inner.model.clone(),inner.roi)
            };
            let identity=(context.generation,frame.frame_counter,revision);
            if last==Some(identity) { continue; }
            last=Some(identity);
            let Some(model)=service.models.iter().find(|m|m.id==model_id) else { continue; };
            let Some(runner)=model.runner.clone() else { continue; };
            let model_sha=runner.model_sha256().to_string();
            let manifest_sha=runner.manifest_sha256().to_string();
            let worker_context=context.clone();
            let result=tokio::task::spawn_blocking(move || {
                let now=chrono::Utc::now().timestamp_millis() as f64/1000.0;
                if !source_fresh(frame.unix_timestamp,now) { return Err("Preview frame is stale or its clock is invalid".into()); }
                let (rgb,width,height)=rendered_rgb(&frame)?;
                let mask=roi.map(|r|r.mask(width,height));
                let start=Instant::now();
                let output=runner.predict_rgb8(&rgb,width,height,mask.as_deref())?;
                build_sample(&frame,worker_context,output,roi,&rgb,&model_id,&model_sha,&manifest_sha,start.elapsed().as_secs_f64()*1000.0)
            }).await.unwrap_or_else(|e|Err(format!("COC worker failed: {e}")));
            let _barrier=state.camera_observation_barrier.lock().unwrap();
            if current_context(&state).as_ref()!=Some(&context) { continue; }
            let mut inner=service.inner.lock().unwrap();
            if !inner.enabled || inner.revision!=revision { continue; }
            match result {
                Ok(sample)=>{ inner.reason=None; inner.history.push_back(Arc::new(sample));
                    while inner.history.len()>HISTORY {inner.history.pop_front();} }
                Err(error)=>{inner.history.clear();inner.reason=Some(error);}
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn jpeg_orientation_must_preserve_independent_roi_geometry() {
        // SOI, APP1 containing one primary TIFF orientation, EOI.
        for little in [true,false] {
            for orientation in [1u16,3,6,8] {
                let mut tiff=if little {b"II".to_vec()} else {b"MM".to_vec()};
                let u16b=|v:u16|if little {v.to_le_bytes()} else {v.to_be_bytes()};
                let u32b=|v:u32|if little {v.to_le_bytes()} else {v.to_be_bytes()};
                tiff.extend(u16b(42));tiff.extend(u32b(8));tiff.extend(u16b(1));
                tiff.extend(u16b(0x112));tiff.extend(u16b(3));tiff.extend(u32b(1));
                tiff.extend(u16b(orientation));tiff.extend([0,0]);tiff.extend(u32b(0));
                let mut segment=b"Exif\0\0".to_vec();segment.extend(tiff);
                let mut jpeg=vec![0xff,0xd8,0xff,0xe1];
                jpeg.extend(((segment.len()+2) as u16).to_be_bytes());jpeg.extend(segment);jpeg.extend([0xff,0xd9]);
                assert_eq!(validate_jpeg_orientation(&jpeg).is_ok(),orientation==1);
                jpeg.truncate(jpeg.len()-10);
                assert!(validate_jpeg_orientation(&jpeg).is_err());
            }
        }
        assert!(validate_jpeg_orientation(&[0xff,0xd8,0xff,0xd9]).is_ok());
        assert!(validate_jpeg_orientation(&[]).is_err());
    }
    #[test]
    fn roi_does_not_infer_ignored_or_foreground_pixels_as_valid_sky() {
        let roi=RoiRect {left:0.25,top:0.0,right:0.75,bottom:0.5};
        assert!(roi.validate().is_ok());
        assert_eq!(roi.mask(4,2),vec![0,1,1,0,0,0,0,0]);
        assert!(RoiRect{left:0.8,right:0.2,..roi}.validate().is_err());
        assert!(RoiRect{left:f64::NAN,..roi}.validate().is_err());
    }
    #[test]
    fn stale_and_future_source_frames_are_not_current_readings() {
        assert!(source_fresh(100.0,101.0));
        assert!(!source_fresh(100.0,106.0));
        assert!(!source_fresh(102.0,100.0));
        assert!(!source_fresh(f64::NAN,100.0));
    }
}

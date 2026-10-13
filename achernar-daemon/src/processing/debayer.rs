//! Versioned linear analysis of borrowed, untouched sensor samples.
//! Stetson (1987) PDF pp5/7 and Astropy (2013) pp5/6 motivate detector masks,
//! units and coordinate transforms; they do not qualify this demosaic.
//! See docs/a03-validation-protocol-2026-10-12.md for the frozen validation.

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const PROCESSING_VERSION: &str = "achernar-linear-bilinear-rgb-mean-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SampleType {
    U8,
    U16,
    F32,
}
impl SampleType {
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::F32 => 4,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ByteOrder {
    Little,
    Big,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CfaPattern {
    Rggb,
    Bggr,
    Grbg,
    Gbrg,
}
impl CfaPattern {
    /// RGB channel index at a CFA coordinate, before applying crop origin.
    pub fn channel(self, x: usize, y: usize) -> usize {
        let i = (y % 2) * 2 + x % 2;
        match self {
            Self::Rggb => [0, 1, 1, 2][i],
            Self::Bggr => [2, 1, 1, 0][i],
            Self::Grbg => [1, 0, 2, 1][i],
            Self::Gbrg => [1, 2, 0, 1][i],
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorColor {
    Mono,
    Bayer {
        pattern: CfaPattern,
        origin_x: usize,
        origin_y: usize,
    },
    Unknown,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SampleDomain {
    NativeAdu,
    LinearResidualAdu,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeThreshold {
    pub adu: f32,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PixelGeometry {
    /// Pixel-centre mapping to the named source coordinate system.
    pub analysis_to_sensor: [[f64; 3]; 2],
    /// decoded_plane and stream_buffer do not claim an absolute sensor origin.
    pub coordinate_system: String,
}
impl PixelGeometry {
    pub fn identity(system: &str) -> Self {
        Self {
            analysis_to_sensor: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            coordinate_system: system.into(),
        }
    }
    pub fn map_to_sensor(&self, x: f64, y: f64) -> [f64; 2] {
        let m = self.analysis_to_sensor;
        [
            m[0][0] * x + m[0][1] * y + m[0][2],
            m[1][0] * x + m[1][1] * y + m[1][2],
        ]
    }
    /// Anisotropic geometry requires a source-coordinate refit, not a scalar
    /// multiplier on HFR. Unknown absolute origins do not become native units.
    pub fn isotropic_native_scale(&self) -> Option<f64> {
        if self.coordinate_system != "native_sensor" {
            return None;
        }
        let m = self.analysis_to_sensor;
        let xx = m[0][0].powi(2) + m[1][0].powi(2);
        let yy = m[0][1].powi(2) + m[1][1].powi(2);
        let xy = m[0][0] * m[0][1] + m[1][0] * m[1][1];
        let tol = 1e-12 * xx.max(yy).max(1.0);
        (xx.is_finite() && yy.is_finite() && xx > 0.0 && (xx - yy).abs() <= tol && xy.abs() <= tol)
            .then(|| xx.sqrt())
    }
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.coordinate_system.trim().is_empty(),
            "Missing coordinate system"
        );
        ensure!(
            self.analysis_to_sensor
                .iter()
                .flatten()
                .all(|v| v.is_finite()),
            "Nonfinite coordinate transform"
        );
        let m = self.analysis_to_sensor;
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        ensure!(
            det.is_finite() && det != 0.0,
            "Singular coordinate transform"
        );
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorDescriptor {
    pub schema_version: u32,
    pub width: usize,
    pub height: usize,
    pub row_stride_bytes: usize,
    pub sample_type: SampleType,
    pub byte_order: ByteOrder,
    pub color: SensorColor,
    pub domain: SampleDomain,
    pub geometry: PixelGeometry,
    pub saturation: Option<NativeThreshold>,
    pub calibration_ids: Vec<String>,
}
impl SensorDescriptor {
    pub fn validate(&self, byte_length: usize) -> Result<()> {
        ensure!(
            self.schema_version == 1,
            "Unsupported sensor descriptor version"
        );
        ensure!(
            self.width > 0 && self.height > 0,
            "Invalid sensor dimensions"
        );
        self.width
            .checked_mul(self.height)
            .ok_or_else(|| anyhow::anyhow!("Sensor dimensions overflow"))?;
        let packed = self
            .width
            .checked_mul(self.sample_type.bytes_per_sample())
            .ok_or_else(|| anyhow::anyhow!("Sensor row overflow"))?;
        ensure!(
            self.row_stride_bytes >= packed,
            "Sensor row stride too short"
        );
        let expected = self
            .row_stride_bytes
            .checked_mul(self.height)
            .ok_or_else(|| anyhow::anyhow!("Sensor payload overflow"))?;
        ensure!(
            byte_length == expected,
            "Sensor payload length does not match descriptor: {byte_length} != {expected}"
        );
        ensure!(
            self.color != SensorColor::Unknown,
            "Unknown CFA/color makes scientific analysis unavailable"
        );
        if matches!(self.color, SensorColor::Bayer { .. }) {
            ensure!(
                self.width >= 3 && self.height >= 3,
                "Bayer analysis requires an interior footprint"
            );
        }
        self.geometry.validate()?;
        if self.domain == SampleDomain::LinearResidualAdu {
            ensure!(
                self.sample_type == SampleType::F32
                    && !self.calibration_ids.is_empty()
                    && self.calibration_ids.iter().all(|id| !id.trim().is_empty()),
                "Residual samples require float32 and calibration lineage"
            );
            ensure!(self.saturation.is_none(),"Native threshold cannot apply to calibrated residuals; supply native saturation mask");
        } else {
            ensure!(
                self.calibration_ids.is_empty(),
                "Untouched samples cannot claim applied calibration"
            );
        }
        if let Some(t) = &self.saturation {
            ensure!(
                t.adu.is_finite() && t.adu > 0.0 && !t.source.trim().is_empty(),
                "Invalid native saturation evidence"
            );
            let max = match self.sample_type {
                SampleType::U8 => 255.0,
                SampleType::U16 => 65535.0,
                SampleType::F32 => f32::MAX,
            };
            ensure!(t.adu <= max, "Native threshold exceeds sample container");
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, Default)]
pub struct NativeMasks<'a> {
    pub bad_pixel: Option<&'a [u8]>,
    pub invalid: Option<&'a [u8]>,
    pub native_saturation: Option<&'a [u8]>,
}
pub struct SensorFrame<'a> {
    pub descriptor: &'a SensorDescriptor,
    pub data: &'a [u8],
    pub masks: NativeMasks<'a>,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisMasks {
    pub bad_pixel: Vec<u8>,
    pub invalid: Vec<u8>,
    pub edge: Vec<u8>,
    pub native_saturation: Vec<u8>,
}
impl AnalysisMasks {
    pub fn empty(n: usize) -> Self {
        Self {
            bad_pixel: vec![0; n],
            invalid: vec![0; n],
            edge: vec![0; n],
            native_saturation: vec![0; n],
        }
    }
    pub fn excluded(&self, i: usize) -> bool {
        self.bad_pixel[i] != 0
            || self.invalid[i] != 0
            || self.edge[i] != 0
            || self.native_saturation[i] != 0
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisDescriptor {
    pub schema_version: u32,
    pub processing_version: String,
    pub config_sha256: String,
    /// Exact input bytes including row padding. Descriptor/config hash separate.
    pub source_sha256: String,
    pub sensor: SensorDescriptor,
    pub units: String,
    pub intensity_definition: String,
    pub training_eligible: bool,
    pub geometry: PixelGeometry,
    pub native_mask_sha256: BTreeMap<String, String>,
    pub mask_states: BTreeMap<String, String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisFrame {
    pub pixels: Vec<f32>,
    pub masks: AnalysisMasks,
    pub descriptor: AnalysisDescriptor,
}

pub fn analyze(sensor: &SensorFrame<'_>) -> Result<AnalysisFrame> {
    let (pixels, masks, _) = reconstruct(sensor, false)?;
    let d = sensor.descriptor;
    let mut native_mask_sha256 = BTreeMap::new();
    let mut mask_states = BTreeMap::new();
    for (name, mask) in [
        ("bad_pixel", sensor.masks.bad_pixel),
        ("invalid", sensor.masks.invalid),
        ("native_saturation", sensor.masks.native_saturation),
    ] {
        if let Some(mask) = mask {
            native_mask_sha256.insert(name.into(), digest(mask));
        }
        mask_states.insert(
            name.into(),
            if mask.is_some() {
                "provided"
            } else {
                "unavailable"
            }
            .into(),
        );
    }
    mask_states.insert(
        "invalid".into(),
        if sensor.masks.invalid.is_some() {
            "provided_and_finite_checks"
        } else {
            "finite_checks_only"
        }
        .into(),
    );
    if d.saturation.is_some() {
        mask_states.insert(
            "native_saturation".into(),
            if sensor.masks.native_saturation.is_some() {
                "provided_or_native_threshold"
            } else {
                "native_threshold"
            }
            .into(),
        );
    }
    mask_states.insert(
        "edge".into(),
        if matches!(d.color, SensorColor::Bayer { .. }) {
            "interpolation_footprint_1px"
        } else {
            "identity_no_interpolation"
        }
        .into(),
    );
    let config = serde_json::json!({"processing_version":PROCESSING_VERSION,"sensor":d,"native_mask_sha256":native_mask_sha256,"rgb_weights":[1.0/3.0,1.0/3.0,1.0/3.0],"border":"normalize_available_support_exclude_1px","invalid":"NaN_and_mask","qualification":"algorithm_only"});
    Ok(AnalysisFrame {
        pixels,
        masks,
        descriptor: AnalysisDescriptor {
            schema_version: 1,
            processing_version: PROCESSING_VERSION.into(),
            config_sha256: digest(&serde_json::to_vec(&config)?),
            source_sha256: digest(sensor.data),
            sensor: d.clone(),
            units: if d.color == SensorColor::Mono {
                "adu"
            } else {
                "adu_proxy"
            }
            .into(),
            intensity_definition: if d.color == SensorColor::Mono {
                "identity_linear_samples"
            } else {
                "arithmetic_mean_of_linear_bilinear_rgb_uncalibrated_intensity_proxy"
            }
            .into(),
            training_eligible: false,
            geometry: d.geometry.clone(),
            native_mask_sha256,
            mask_states,
        },
    })
}
/// Explicit RGB order for reference checks. Production does not retain RGB.
pub fn linear_rgb(sensor: &SensorFrame<'_>) -> Result<Vec<[f32; 3]>> {
    Ok(reconstruct(sensor, true)?.2.unwrap())
}
fn digest(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}
type Reconstruction = (Vec<f32>, AnalysisMasks, Option<Vec<[f32; 3]>>);
fn reconstruct(sensor: &SensorFrame<'_>, keep_rgb: bool) -> Result<Reconstruction> {
    let d = sensor.descriptor;
    d.validate(sensor.data.len())?;
    let n = d.width * d.height;
    for mask in [
        sensor.masks.bad_pixel,
        sensor.masks.invalid,
        sensor.masks.native_saturation,
    ]
    .into_iter()
    .flatten()
    {
        ensure!(
            mask.len() == n && mask.iter().all(|v| *v <= 1),
            "Native mask must be a dimension-matched binary plane"
        );
    }
    let values = read_samples(sensor);
    let mut pixels = Vec::with_capacity(n);
    let mut masks = AnalysisMasks::empty(n);
    let mut rgb = keep_rgb.then(|| Vec::with_capacity(n));
    for y in 0..d.height {
        for x in 0..d.width {
            let i = y * d.width + x;
            let mut sums = [0.0_f64; 3];
            let mut weights = [0.0_f64; 3];
            let mut bad = false;
            let mut invalid = false;
            let mut saturated = false;
            let mut include = |nx: usize, ny: usize, c: usize, w: f64| {
                let j = ny * d.width + nx;
                let v = values[j];
                bad |= sensor.masks.bad_pixel.is_some_and(|m| m[j] != 0);
                invalid |= !v.is_finite() || sensor.masks.invalid.is_some_and(|m| m[j] != 0);
                saturated |= sensor.masks.native_saturation.is_some_and(|m| m[j] != 0)
                    || d.saturation.as_ref().is_some_and(|t| v >= t.adu);
                sums[c] += v as f64 * w;
                weights[c] += w;
            };
            match d.color {
                SensorColor::Mono => {
                    for c in 0..3 {
                        include(x, y, c, 1.0);
                    }
                }
                SensorColor::Bayer {
                    pattern,
                    origin_x,
                    origin_y,
                } => {
                    for ny in y.saturating_sub(1)..=(y + 1).min(d.height - 1) {
                        for nx in x.saturating_sub(1)..=(x + 1).min(d.width - 1) {
                            let c = pattern.channel(nx % 2 + origin_x % 2, ny % 2 + origin_y % 2);
                            let dx = nx.abs_diff(x);
                            let dy = ny.abs_diff(y);
                            let w = if c == 1 {
                                match (dx, dy) {
                                    (0, 0) => 1.0,
                                    (1, 0) | (0, 1) => 0.25,
                                    _ => 0.0,
                                }
                            } else {
                                match (dx, dy) {
                                    (0, 0) => 1.0,
                                    (1, 0) | (0, 1) => 0.5,
                                    _ => 0.25,
                                }
                            };
                            if w > 0.0 {
                                include(nx, ny, c, w);
                            }
                        }
                    }
                    masks.edge[i] =
                        u8::from(x == 0 || y == 0 || x + 1 == d.width || y + 1 == d.height);
                }
                SensorColor::Unknown => unreachable!("validated above"),
            }
            let mut color = [0.0_f32; 3];
            let mut intensity = 0.0_f64;
            for c in 0..3 {
                invalid |= weights[c] == 0.0;
                let v = sums[c] / weights[c];
                color[c] = v as f32;
                intensity += v / 3.0;
            }
            let v = if d.color == SensorColor::Mono {
                values[i]
            } else {
                intensity as f32
            };
            invalid |= bad || !v.is_finite() || color.iter().any(|v| !v.is_finite());
            masks.bad_pixel[i] = u8::from(bad);
            masks.invalid[i] = u8::from(invalid);
            masks.native_saturation[i] = u8::from(saturated);
            pixels.push(if invalid { f32::NAN } else { v });
            if let Some(rgb) = rgb.as_mut() {
                rgb.push(if invalid { [f32::NAN; 3] } else { color });
            }
        }
    }
    Ok((pixels, masks, rgb))
}
fn read_samples(sensor: &SensorFrame<'_>) -> Vec<f32> {
    let d = sensor.descriptor;
    let size = d.sample_type.bytes_per_sample();
    let mut result = Vec::with_capacity(d.width * d.height);
    for row in sensor.data.chunks_exact(d.row_stride_bytes) {
        for px in row[..d.width * size].chunks_exact(size) {
            result.push(match d.sample_type {
                SampleType::U8 => px[0] as f32,
                SampleType::U16 => {
                    (match d.byte_order {
                        ByteOrder::Little => u16::from_le_bytes([px[0], px[1]]),
                        ByteOrder::Big => u16::from_be_bytes([px[0], px[1]]),
                    }) as f32
                }
                SampleType::F32 => match d.byte_order {
                    ByteOrder::Little => f32::from_le_bytes(px.try_into().unwrap()),
                    ByteOrder::Big => f32::from_be_bytes(px.try_into().unwrap()),
                },
            });
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(color: SensorColor) -> SensorDescriptor {
        SensorDescriptor {
            schema_version: 1,
            width: 5,
            height: 5,
            row_stride_bytes: 10,
            sample_type: SampleType::U16,
            byte_order: ByteOrder::Little,
            color,
            domain: SampleDomain::NativeAdu,
            geometry: PixelGeometry::identity("native_sensor"),
            saturation: Some(NativeThreshold {
                adu: 16383.0,
                source: "synthetic_test_white_code".into(),
            }),
            calibration_ids: vec![],
        }
    }
    fn plane(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn all_patterns_and_crop_parities_preserve_rgb_identity_and_native_bytes() {
        for pattern in [
            CfaPattern::Rggb,
            CfaPattern::Bggr,
            CfaPattern::Grbg,
            CfaPattern::Gbrg,
        ] {
            for origin_y in 0..2 {
                for origin_x in 0..2 {
                    let d = descriptor(SensorColor::Bayer {
                        pattern,
                        origin_x,
                        origin_y,
                    });
                    let values: Vec<u16> = (0..25)
                        .map(|i| {
                            [3000, 600, 90][pattern.channel(i % 5 + origin_x, i / 5 + origin_y)]
                        })
                        .collect();
                    let data = plane(&values);
                    let sensor = SensorFrame {
                        descriptor: &d,
                        data: &data,
                        masks: Default::default(),
                    };
                    assert_eq!(linear_rgb(&sensor).unwrap()[12], [3000.0, 600.0, 90.0]);
                    let output = analyze(&sensor).unwrap();
                    assert_eq!(output.pixels[12], 1230.0);
                    assert_eq!(output.descriptor.units, "adu_proxy");
                    assert!(!output.descriptor.training_eligible);
                    assert_eq!(data, plane(&values));
                }
            }
        }
    }

    #[test]
    fn native_clipping_and_bad_pixels_propagate_before_interpolation_dilutes_them() {
        let d = descriptor(SensorColor::Bayer {
            pattern: CfaPattern::Rggb,
            origin_x: 0,
            origin_y: 0,
        });
        let mut values = vec![100; 25];
        values[12] = 16383;
        let data = plane(&values);
        let sensor = SensorFrame {
            descriptor: &d,
            data: &data,
            masks: Default::default(),
        };
        let output = analyze(&sensor).unwrap();
        assert!(output.pixels[12] < 16383.0);
        assert_eq!(output.masks.native_saturation[6], 1);
        assert_eq!(output.masks.native_saturation[0], 0);
        assert_eq!(output.masks.edge[0], 1);
        let mut bad = vec![0; 25];
        bad[12] = 1;
        let output = analyze(&SensorFrame {
            masks: NativeMasks {
                bad_pixel: Some(&bad),
                ..Default::default()
            },
            ..sensor
        })
        .unwrap();
        assert!(output.pixels[6].is_nan());
        assert_eq!(output.masks.bad_pixel[6], 1);
        assert_eq!(output.masks.invalid[6], 1);
    }

    #[test]
    fn big_endian_padded_rows_are_exact_and_partial_or_extra_payloads_are_refused() {
        let mut d = descriptor(SensorColor::Mono);
        d.byte_order = ByteOrder::Big;
        d.row_stride_bytes = 12;
        let mut data = Vec::new();
        for _ in 0..5 {
            for value in [0u16, 1, 256, 16383, 65535] {
                data.extend(value.to_be_bytes());
            }
            data.extend([99, 99]);
        }
        let sensor = SensorFrame {
            descriptor: &d,
            data: &data,
            masks: Default::default(),
        };
        let output = analyze(&sensor).unwrap();
        assert_eq!(&output.pixels[..5], &[0.0, 1.0, 256.0, 16383.0, 65535.0]);
        assert_eq!(output.descriptor.source_sha256, digest(&data));
        assert!(d.validate(data.len() - 1).is_err());
        assert!(d.validate(data.len() + 1).is_err());
        d.row_stride_bytes = 9;
        assert!(d.validate(data.len()).is_err());
    }

    #[test]
    fn signed_residuals_remain_signed_and_native_threshold_cannot_be_reused() {
        let mut d = descriptor(SensorColor::Mono);
        d.sample_type = SampleType::F32;
        d.row_stride_bytes = 20;
        d.domain = SampleDomain::LinearResidualAdu;
        d.saturation = None;
        d.calibration_ids = vec!["synthetic_bias_v1".into()];
        let mut values = vec![-1.5f32; 25];
        values[12] = f32::NAN;
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let output = analyze(&SensorFrame {
            descriptor: &d,
            data: &data,
            masks: Default::default(),
        })
        .unwrap();
        assert_eq!(output.pixels[0], -1.5);
        assert_eq!(output.masks.invalid[12], 1);
        assert_eq!(
            output.descriptor.mask_states["native_saturation"],
            "unavailable"
        );
        d.saturation = Some(NativeThreshold {
            adu: 16383.0,
            source: "raw_threshold".into(),
        });
        assert!(d.validate(data.len()).is_err());
        d.saturation = None;
        d.calibration_ids.clear();
        assert!(d.validate(data.len()).is_err());
    }

    #[test]
    fn coordinate_mapping_preserves_rotation_and_rejects_anisotropic_hfr_scaling() {
        let mut g = PixelGeometry {
            analysis_to_sensor: [[0.0, -2.0, 11.0], [2.0, 0.0, 7.0]],
            coordinate_system: "native_sensor".into(),
        };
        assert_eq!(g.map_to_sensor(3.0, 4.0), [3.0, 13.0]);
        assert_eq!(g.isotropic_native_scale(), Some(2.0));
        g.analysis_to_sensor[0][1] = -3.0;
        assert_eq!(g.isotropic_native_scale(), None);
        assert_eq!(
            PixelGeometry::identity("decoded_plane").isotropic_native_scale(),
            None
        );
    }

    #[test]
    fn config_digest_binds_masks_threshold_and_geometry_without_changing_sample_digest() {
        let mut d = descriptor(SensorColor::Mono);
        let data = plane(&vec![100; 25]);
        let first = analyze(&SensorFrame {
            descriptor: &d,
            data: &data,
            masks: Default::default(),
        })
        .unwrap();
        d.saturation.as_mut().unwrap().adu = 14000.0;
        let second = analyze(&SensorFrame {
            descriptor: &d,
            data: &data,
            masks: Default::default(),
        })
        .unwrap();
        assert_ne!(
            first.descriptor.config_sha256,
            second.descriptor.config_sha256
        );
        assert_eq!(
            first.descriptor.source_sha256,
            second.descriptor.source_sha256
        );
        let bad = vec![0; 25];
        let third = analyze(&SensorFrame {
            descriptor: &d,
            data: &data,
            masks: NativeMasks {
                bad_pixel: Some(&bad),
                ..Default::default()
            },
        })
        .unwrap();
        assert_ne!(
            second.descriptor.config_sha256,
            third.descriptor.config_sha256
        );
        d.color = SensorColor::Unknown;
        assert!(d.validate(data.len()).is_err());
    }
}

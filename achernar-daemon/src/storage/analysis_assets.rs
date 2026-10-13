//! Reproducible A03 analysis derivatives. Sensor/native assets remain canonical
//! acquisition evidence. A failed derivative must not discard that evidence.

use anyhow::{anyhow, Result};
use sha2::{Digest, Sha256};

use super::capture_store::{AssetRole, PendingAsset};
use crate::processing::debayer::{AnalysisFrame, PROCESSING_VERSION};

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Append these assets after `sensor`. The exact source FITS parent and sample
/// bytes are independently identified by content hashes. No database lookup is
/// needed to rerun the documented transformation from an archived bundle.
pub fn analysis_assets(
    analysis: &AnalysisFrame,
    sensor_filename: &str,
    sensor_fits: &[u8],
    source_data: &[u8],
) -> Result<Vec<PendingAsset>> {
    if sensor_filename.is_empty() || sensor_fits.is_empty() {
        return Err(anyhow!(
            "Analysis requires a nonempty exact sensor FITS parent"
        ));
    }
    let descriptor = &analysis.descriptor;
    descriptor.sensor.validate(source_data.len())?;
    if descriptor.schema_version != 1
        || descriptor.processing_version != PROCESSING_VERSION
        || descriptor.geometry != descriptor.sensor.geometry
    {
        return Err(anyhow!(
            "Analysis descriptor version or geometry is incompatible with this archive writer"
        ));
    }
    let packed = descriptor
        .sensor
        .width
        .checked_mul(descriptor.sensor.sample_type.bytes_per_sample())
        .ok_or_else(|| anyhow!("Sensor row size overflow"))?;
    if descriptor.sensor.row_stride_bytes != packed {
        return Err(anyhow!("Padded source rows require an immutable exact-source payload asset before analysis can be archived reproducibly"));
    }
    if !descriptor.native_mask_sha256.is_empty() {
        return Err(anyhow!("Supplied native input masks require immutable input-mask assets before analysis can be archived reproducibly"));
    }
    let width = descriptor.sensor.width;
    let height = descriptor.sensor.height;
    let count = width
        .checked_mul(height)
        .ok_or_else(|| anyhow!("Analysis dimensions overflow"))?;
    if count == 0
        || analysis.pixels.len() != count
        || descriptor.source_sha256 != sha256(source_data)
        || descriptor.training_eligible
    {
        return Err(anyhow!(
            "Analysis shape, source hash or eligibility is incompatible with A03 archive"
        ));
    }
    let masks = [
        ("bad_pixel", &analysis.masks.bad_pixel),
        ("invalid", &analysis.masks.invalid),
        ("edge", &analysis.masks.edge),
        ("native_saturation", &analysis.masks.native_saturation),
    ];
    if masks
        .iter()
        .any(|(_, mask)| mask.len() != count || mask.iter().any(|flag| *flag > 1))
    {
        return Err(anyhow!("Analysis mask dimensions or values are invalid"));
    }
    if analysis
        .pixels
        .iter()
        .zip(&analysis.masks.invalid)
        .any(|(value, invalid)| !value.is_finite() && *invalid == 0)
    {
        return Err(anyhow!(
            "Nonfinite analysis samples require the invalid mask"
        ));
    }
    let parent_sha256 = sha256(sensor_fits);
    let provenance = serde_json::json!({
        "schema_version": 1,
        "processing_version": descriptor.processing_version,
        "config_sha256": descriptor.config_sha256,
        "parent": {"role": "sensor_fits", "filename": sensor_filename, "sha256": parent_sha256},
        "training_label_eligible": false,
    });
    let intensity = crate::fits::encode_linear_analysis(
        &analysis.pixels,
        width,
        height,
        &descriptor.units,
        &descriptor.processing_version,
    )?;
    let intensity_hash = sha256(&intensity);
    let mut assets = vec![PendingAsset::new(
        AssetRole::AnalysisFits,
        "linear.fits",
        "application/fits",
        intensity,
    )
    .with_dimensions(width as i64, height as i64, 32)
    .derived_from_asset(AssetRole::SensorFits, sensor_filename)
    .with_provenance(provenance.clone())];
    let mut mask_records = Vec::new();
    for (kind, mask) in masks {
        let filename = format!("{kind}.fits");
        let bytes =
            crate::fits::encode_analysis_mask(mask, width, height, &descriptor.processing_version)?;
        mask_records.push(serde_json::json!({
            "kind": kind, "role": "analysis_mask", "filename": filename,
            "sha256": sha256(&bytes), "flagged_value": 1,
        }));
        let mut mask_provenance = provenance.clone();
        mask_provenance["mask_kind"] = serde_json::json!(kind);
        assets.push(
            PendingAsset::new(AssetRole::AnalysisMask, filename, "application/fits", bytes)
                .with_dimensions(width as i64, height as i64, 8)
                .derived_from_asset(AssetRole::SensorFits, sensor_filename)
                .with_provenance(mask_provenance),
        );
    }
    let archive_descriptor = serde_json::json!({
        "schema_version": 1,
        "analysis": descriptor,
        "parent": provenance["parent"],
        "source_samples_sha256": descriptor.source_sha256,
        "intensity": {"role": "analysis_fits", "filename": "linear.fits", "sha256": intensity_hash, "dtype": "float32"},
        "masks": mask_records,
        "calibration_state": if descriptor.sensor.calibration_ids.is_empty() { "unapplied" } else { "recorded_in_sensor_descriptor" },
        "training_label_eligible": false,
        "validity": "computational_baseline_unqualified_for_optical_labels",
    });
    assets.push(
        PendingAsset::new(
            AssetRole::AnalysisDescriptor,
            "analysis.json",
            "application/json",
            serde_json::to_vec_pretty(&archive_descriptor)?,
        )
        .derived_from_asset(AssetRole::SensorFits, sensor_filename)
        .with_provenance(provenance),
    );
    Ok(assets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing::debayer::*;

    fn fixture() -> (AnalysisFrame, Vec<u8>) {
        let descriptor = SensorDescriptor {
            schema_version: 1,
            width: 2,
            height: 2,
            row_stride_bytes: 8,
            sample_type: SampleType::F32,
            byte_order: ByteOrder::Little,
            color: SensorColor::Mono,
            domain: SampleDomain::LinearResidualAdu,
            geometry: PixelGeometry::identity("synthetic_sensor_plane"),
            saturation: None,
            calibration_ids: vec!["synthetic-residual-calibration-v1".into()],
        };
        let data: Vec<u8> = [-1.25f32, 0.125, 20.0, 500.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let analysis = analyze(&SensorFrame {
            descriptor: &descriptor,
            data: &data,
            masks: Default::default(),
        })
        .unwrap();
        (analysis, data)
    }

    #[test]
    fn descriptor_identifies_exact_parent_intensity_and_all_mask_content() {
        let (analysis, data) = fixture();
        let assets = analysis_assets(
            &analysis,
            "sensor.fits",
            b"exact parent sensor bytes",
            &data,
        )
        .unwrap();
        assert_eq!(assets.len(), 6);
        let descriptor: serde_json::Value = serde_json::from_slice(&assets[5].bytes).unwrap();
        assert_eq!(
            descriptor["parent"]["sha256"],
            sha256(b"exact parent sensor bytes")
        );
        assert_eq!(descriptor["source_samples_sha256"], sha256(&data));
        assert_eq!(descriptor["intensity"]["sha256"], sha256(&assets[0].bytes));
        for (record, asset) in descriptor["masks"]
            .as_array()
            .unwrap()
            .iter()
            .zip(&assets[1..5])
        {
            assert_eq!(record["filename"], asset.filename);
            assert_eq!(record["sha256"], sha256(&asset.bytes));
        }
        for asset in assets {
            assert_eq!(asset.derived_from, Some(AssetRole::SensorFits));
            assert_eq!(asset.derived_from_filename.as_deref(), Some("sensor.fits"));
        }
        assert_eq!(descriptor["training_label_eligible"], false);
        assert_eq!(
            descriptor["calibration_state"],
            "recorded_in_sensor_descriptor"
        );
    }

    #[test]
    fn wrong_source_mask_or_unflagged_nonfinite_samples_are_refused() {
        let (mut analysis, data) = fixture();
        assert!(analysis_assets(&analysis, "sensor.fits", b"parent", b"wrong source").is_err());
        analysis.pixels[0] = f32::NAN;
        assert!(analysis_assets(&analysis, "sensor.fits", b"parent", &data).is_err());
        analysis.masks.invalid[0] = 1;
        let assets = analysis_assets(&analysis, "sensor.fits", b"parent", &data).unwrap();
        let encoded = u32::from_be_bytes(assets[0].bytes[2880..2884].try_into().unwrap());
        assert_eq!(encoded, f32::NAN.to_bits());
        assert_eq!(assets[2].bytes[2880], 1);
        analysis.masks.edge.pop();
        assert!(analysis_assets(&analysis, "sensor.fits", b"parent", &data).is_err());
    }

    #[test]
    fn supplied_input_masks_cannot_be_replaced_by_propagated_output_masks() {
        let (analysis, data) = fixture();
        let bad = [1, 0, 0, 0];
        let masked = analyze(&SensorFrame {
            descriptor: &analysis.descriptor.sensor,
            data: &data,
            masks: NativeMasks {
                bad_pixel: Some(&bad),
                ..Default::default()
            },
        })
        .unwrap();
        let error = analysis_assets(&masked, "sensor.fits", b"parent", &data).unwrap_err();
        assert!(error.to_string().contains("input-mask assets"));
    }

    #[test]
    fn padded_sources_and_incompatible_mutated_descriptors_are_refused() {
        let (mut analysis, data) = fixture();
        let mut padded_descriptor = analysis.descriptor.sensor.clone();
        padded_descriptor.row_stride_bytes = 12;
        let mut padded = data[..8].to_vec();
        padded.extend_from_slice(&[7; 4]);
        padded.extend_from_slice(&data[8..]);
        padded.extend_from_slice(&[9; 4]);
        let padded_analysis = analyze(&SensorFrame {
            descriptor: &padded_descriptor,
            data: &padded,
            masks: Default::default(),
        })
        .unwrap();
        assert!(
            analysis_assets(&padded_analysis, "sensor.fits", b"parent", &padded)
                .unwrap_err()
                .to_string()
                .contains("Padded source rows")
        );
        analysis.descriptor.schema_version = 99;
        assert!(analysis_assets(&analysis, "sensor.fits", b"parent", &data).is_err());
        analysis.descriptor.schema_version = 1;
        analysis.descriptor.processing_version = "unsupported-version".into();
        assert!(analysis_assets(&analysis, "sensor.fits", b"parent", &data).is_err());
        analysis.descriptor.processing_version = PROCESSING_VERSION.into();
        analysis.descriptor.sensor.width += 1;
        assert!(analysis_assets(&analysis, "sensor.fits", b"parent", &data).is_err());
    }
}

//! Filter-wheel reconcile at connect (spec §3.4 W-b, D17). Owner: WP1A-6.
//!
//! An independent rule set:
//!
//! - The slot count comes from the device.
//! - Names: the identity profile, else the device names (Alpaca `names`;
//!   INDI `FILTER_NAME` in definition order), else `Slot n` (1-based label).
//! - Offsets: the identity profile when set, else the device's
//!   `FocusOffsets`. INDI has none.
//! - **The `[filter_wheel]` TOML arrays are never applied automatically.**
//!   [`import_seed`] copies them into the profile, and only when the operator
//!   asks (`POST filter-wheel/import-seed`, audited by the gate).
//! - Both versions are kept: the profile keeps its own values and records
//!   what the device reported (`device_reported`), and every disagreement is
//!   listed for a `filter_config_mismatch` event.
//!
//! Offsets carry a quality. Only a `measured` profile offset is certain;
//! device claims and imported seeds are `estimated`, so a change that applies
//! them asks for focus verification (W-c step 10).
//!
//! Profile JSON (`device_profiles.profile_json` of the wheel's identity; other
//! keys belong to other packages and are left alone):
//!
//! ```json
//! {"slots": [{"index": 0, "name": "L", "offset": 0, "offset_quality": "measured",
//!             "throughput": 0.95, "bandwidth_nm": 300.0}],
//!  "settle_ms": 750,
//!  "device_reported": {"slot_count": 7, "names": ["L", ...], "offsets": [0, ...],
//!                      "seen_at": "..."},
//!  "seed_import": {"imported_at": "...", "names": [...], "offsets": [...],
//!                  "applied_slots": 7}}
//! ```

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::config::FilterWheelConfig;
use crate::equipment::commands::FilterTarget;
use crate::equipment::events::{EquipmentEvent, EventKind, GroundTruthRecord, ProfileUpdate};
use crate::equipment::snapshot::{FilterSlot, SlotInfo, SlotValueSource};
use crate::equipment::traits::{Clock, RecorderSink};
use crate::equipment::types::{DeviceError, DeviceIdentity, DeviceKind};

use super::FilterEventSink;

/// How trustworthy a focus offset is (W-c step 10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OffsetQuality {
    /// Measured on this optical train (autofocus per filter).
    Measured,
    /// A driver claim, an imported seed, or a profile value without a
    /// recorded measurement.
    #[default]
    Estimated,
    /// No offset is known.
    Unmeasured,
}

impl OffsetQuality {
    /// Only a measured offset needs no focus verification after a change.
    pub fn is_certain(self) -> bool {
        self == OffsetQuality::Measured
    }
}

/// One slot of the wheel's identity profile.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ProfileSlot {
    /// 0-based slot.
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<i32>,
    /// Missing means `estimated` when an offset is present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset_quality: Option<OffsetQuality>,
    /// Relative throughput, 0..1 (FW-1 input).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub throughput: Option<f64>,
    /// Passband width, nm (FW-1 input).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bandwidth_nm: Option<f64>,
}

/// What the device reported at the last connect ("both versions are
/// stored").
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DeviceReported {
    pub slot_count: u32,
    pub names: Vec<Option<String>>,
    pub offsets: Vec<Option<i32>>,
    pub seen_at: Option<DateTime<Utc>>,
}

/// Provenance of an explicit seed import.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SeedImport {
    pub imported_at: Option<DateTime<Utc>>,
    pub names: Vec<String>,
    pub offsets: Vec<i32>,
    pub applied_slots: u32,
}

/// The filter-wheel part of a `device_profiles` row. Unknown keys are
/// ignored, so other packages' profile entries never break it.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FilterWheelProfile {
    pub slots: Vec<ProfileSlot>,
    /// Per-wheel settle time; wins over `[filter_wheel] settle_ms`.
    pub settle_ms: Option<u64>,
    pub device_reported: Option<DeviceReported>,
    pub seed_import: Option<SeedImport>,
}

impl FilterWheelProfile {
    /// Parse a `profile_json` value. A malformed filter part is an error the
    /// caller reports; it is never silently replaced by seeds.
    pub fn from_profile_json(value: &Value) -> Result<Self, String> {
        if value.is_null() {
            return Ok(Self::default());
        }
        serde_json::from_value(value.clone())
            .map_err(|err| format!("filter-wheel profile is malformed: {err}"))
    }

    /// The profile entry for a slot (the first one, if duplicated).
    pub fn slot(&self, index: u32) -> Option<&ProfileSlot> {
        self.slots.iter().find(|slot| slot.index == index)
    }
}

/// One reconciled slot: the frozen [`FilterSlot`] plus the offset quality
/// and the FW-1 metadata.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ReconciledSlot {
    pub slot: FilterSlot,
    pub offset_quality: OffsetQuality,
    pub throughput: Option<f64>,
    pub bandwidth_nm: Option<f64>,
}

/// The wheel after W-b.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ReconciledWheel {
    /// What the device reported, in slot order.
    pub device_slots: Vec<SlotInfo>,
    pub profile: Option<FilterWheelProfile>,
    pub slots: Vec<ReconciledSlot>,
    /// Profile settle time, if any.
    pub settle_ms: Option<u64>,
    /// Profile-versus-device disagreements (`filter_config_mismatch`).
    pub differences: Vec<String>,
}

fn clean(name: Option<&str>) -> Option<String> {
    name.map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
}

/// The fallback label for an unnamed slot (1-based for people; the index
/// stays 0-based).
pub fn default_slot_name(index: u32) -> String {
    format!("Slot {}", index + 1)
}

/// Reconcile what the device reports with the identity profile (W-b). The
/// slot count is the device's; TOML seeds are not an input.
pub fn reconcile(
    device_slots: &[SlotInfo],
    profile: Option<&FilterWheelProfile>,
) -> ReconciledWheel {
    let count = u32::try_from(device_slots.len()).unwrap_or(u32::MAX);
    let mut differences = Vec::new();
    let mut slots = Vec::with_capacity(device_slots.len());

    for (position, device) in device_slots.iter().enumerate() {
        let index = u32::try_from(position).unwrap_or(u32::MAX);
        let entry = profile.and_then(|p| p.slot(index));
        let device_name = clean(device.device_name.as_deref());
        let profile_name = clean(entry.and_then(|e| e.name.as_deref()));

        let (name, name_source) = match (&profile_name, &device_name) {
            (Some(name), _) => (name.clone(), SlotValueSource::Profile),
            (None, Some(name)) => (name.clone(), SlotValueSource::Device),
            (None, None) => (default_slot_name(index), SlotValueSource::None),
        };
        if let (Some(p), Some(d)) = (&profile_name, &device_name) {
            if p != d {
                differences.push(format!(
                    "slot {} name: profile '{p}', device '{d}'",
                    index + 1
                ));
            }
        }

        let profile_offset = entry.and_then(|e| e.offset);
        let (offset, offset_source, offset_quality) =
            match (profile_offset, device.device_focus_offset) {
                (Some(offset), _) => (
                    Some(offset),
                    SlotValueSource::Profile,
                    entry
                        .and_then(|e| e.offset_quality)
                        .filter(|q| *q != OffsetQuality::Unmeasured)
                        .unwrap_or(OffsetQuality::Estimated),
                ),
                (None, Some(offset)) => (
                    Some(offset),
                    SlotValueSource::Device,
                    OffsetQuality::Estimated,
                ),
                (None, None) => (None, SlotValueSource::None, OffsetQuality::Unmeasured),
            };
        if let (Some(p), Some(d)) = (profile_offset, device.device_focus_offset) {
            if p != d {
                differences.push(format!(
                    "slot {} offset: profile {p}, device {d}",
                    index + 1
                ));
            }
        }

        slots.push(ReconciledSlot {
            slot: FilterSlot {
                index,
                name,
                name_source,
                offset,
                offset_source,
            },
            offset_quality,
            throughput: entry.and_then(|e| e.throughput),
            bandwidth_nm: entry.and_then(|e| e.bandwidth_nm),
        });
    }

    if let Some(profile) = profile {
        for extra in profile.slots.iter().filter(|slot| slot.index >= count) {
            differences.push(format!(
                "profile slot {} ({}) is beyond the device's {count} slots and is ignored",
                extra.index + 1,
                clean(extra.name.as_deref()).unwrap_or_else(|| "unnamed".into())
            ));
        }
        if let Some(previous) = &profile.device_reported {
            if previous.slot_count != 0 && previous.slot_count != count {
                differences.push(format!(
                    "the device reported {} slots at its last connect and {count} now",
                    previous.slot_count
                ));
            }
        }
    }

    ReconciledWheel {
        device_slots: device_slots.to_vec(),
        profile: profile.cloned(),
        slots,
        settle_ms: profile.and_then(|p| p.settle_ms),
        differences,
    }
}

impl ReconciledWheel {
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn slot(&self, index: u32) -> Option<&ReconciledSlot> {
        self.slots.get(usize::try_from(index).ok()?)
    }

    pub fn name_of(&self, index: u32) -> Option<&str> {
        self.slot(index).map(|slot| slot.slot.name.as_str())
    }

    /// The frozen snapshot form (§2.13 `slots[{index, name, offset}]`).
    pub fn filter_slots(&self) -> Vec<FilterSlot> {
        self.slots.iter().map(|slot| slot.slot.clone()).collect()
    }

    /// Resolve a handshake target to a 0-based slot. Names match after
    /// trimming, ignoring case; a name used by two slots is ambiguous.
    pub fn resolve(&self, target: &FilterTarget) -> Result<u32, DeviceError> {
        if self.is_empty() {
            return Err(DeviceError::invalid_operation(
                "the filter wheel reports no slots",
            ));
        }
        match target {
            FilterTarget::Slot(index) => {
                if self.slot(*index).is_some() {
                    Ok(*index)
                } else {
                    Err(DeviceError::invalid_value(format!(
                        "slot {index} does not exist; the wheel has slots 0..{}",
                        self.len() - 1
                    )))
                }
            }
            FilterTarget::Name(name) => {
                let wanted = name.trim();
                let matches: Vec<u32> = self
                    .slots
                    .iter()
                    .filter(|slot| slot.slot.name.trim().eq_ignore_ascii_case(wanted))
                    .map(|slot| slot.slot.index)
                    .collect();
                match matches.as_slice() {
                    [index] => Ok(*index),
                    [] => Err(DeviceError::invalid_value(format!(
                        "no slot is named '{wanted}'"
                    ))),
                    many => Err(DeviceError::invalid_value(format!(
                        "the name '{wanted}' is ambiguous: slots {many:?} share it"
                    ))),
                }
            }
        }
    }

    /// The autofocus filter's slot, when `[filter_wheel] af_filter` names
    /// one that resolves.
    pub fn af_slot(&self, af_filter: Option<&str>) -> Option<u32> {
        let name = af_filter?.trim();
        if name.is_empty() {
            return None;
        }
        self.resolve(&FilterTarget::Name(name.to_string())).ok()
    }

    /// `filter_config_mismatch`, when the profile and the device disagree.
    pub fn mismatch_event(&self) -> Option<EventKind> {
        if self.differences.is_empty() {
            None
        } else {
            Some(EventKind::FilterConfigMismatch {
                differences: self.differences.clone(),
            })
        }
    }

    /// What the device reported, for the profile.
    pub fn device_reported(&self, seen_at: DateTime<Utc>) -> DeviceReported {
        DeviceReported {
            slot_count: u32::try_from(self.device_slots.len()).unwrap_or(u32::MAX),
            names: self
                .device_slots
                .iter()
                .map(|slot| clean(slot.device_name.as_deref()))
                .collect(),
            offsets: self
                .device_slots
                .iter()
                .map(|slot| slot.device_focus_offset)
                .collect(),
            seen_at: Some(seen_at),
        }
    }

    /// A profile patch recording the device's version next to the profile's
    /// own values. It never changes a profile name or offset.
    pub fn device_patch(&self, identity_key: &str, now: DateTime<Utc>) -> ProfileUpdate {
        ProfileUpdate {
            identity_key: identity_key.to_string(),
            kind: DeviceKind::FilterWheel,
            patch: json!({ "device_reported": self.device_reported(now) }),
            updated_at: now,
        }
    }
}

/// W-b at connect: reconcile, publish `filter_config_mismatch` when the two
/// versions differ, and store the device version in the profile.
pub fn reconcile_at_connect(
    identity: &DeviceIdentity,
    device_slots: &[SlotInfo],
    profile: Option<&FilterWheelProfile>,
    clock: &dyn Clock,
    events: &dyn FilterEventSink,
    recorder: &dyn RecorderSink,
) -> ReconciledWheel {
    let wheel = reconcile(device_slots, profile);
    if let Some(kind) = wheel.mismatch_event() {
        events.publish(
            EquipmentEvent::new(kind, clock)
                .for_device(DeviceKind::FilterWheel, identity.connection_id),
        );
    }
    recorder.record(GroundTruthRecord::Profile(
        wheel.device_patch(&identity.identity_key, clock.now_utc()),
    ));
    wheel
}

/// Result of an explicit seed import.
#[derive(Debug, Clone, PartialEq)]
pub struct SeedImportResult {
    /// The `device_profiles` merge patch to record.
    pub update: ProfileUpdate,
    /// The profile after the import, for an immediate re-reconcile.
    pub profile: FilterWheelProfile,
    pub applied_slots: u32,
    pub warnings: Vec<String>,
}

/// Copy the `[filter_wheel]` seeds into the wheel's identity profile (D17).
/// Only called for `POST filter-wheel/import-seed`. Seeded offsets are
/// `estimated`. Entries beyond the device's slot count are ignored with a
/// warning; profile slots the seeds do not cover are kept.
pub fn import_seed(
    seeds: &FilterWheelConfig,
    wheel: &ReconciledWheel,
    identity_key: &str,
    now: DateTime<Utc>,
) -> Result<SeedImportResult, DeviceError> {
    if seeds.filter_names.is_empty() && seeds.focus_offsets.is_empty() {
        return Err(DeviceError::invalid_value(
            "[filter_wheel] has no filter_names or focus_offsets to import",
        ));
    }
    if wheel.is_empty() {
        return Err(DeviceError::invalid_operation(
            "the filter wheel reports no slots; nothing can be imported",
        ));
    }
    let count = wheel.len();
    let mut profile = wheel.profile.clone().unwrap_or_default();
    let mut by_index: BTreeMap<u32, ProfileSlot> = BTreeMap::new();
    for slot in profile.slots.drain(..) {
        by_index.entry(slot.index).or_insert(slot);
    }

    let mut warnings = Vec::new();
    let mut applied = 0_u32;
    let entries = seeds.filter_names.len().max(seeds.focus_offsets.len());
    for position in 0..entries {
        let index = u32::try_from(position).unwrap_or(u32::MAX);
        if position >= count {
            warnings.push(format!(
                "seed entry {} is ignored: the wheel has {count} slots",
                index + 1
            ));
            continue;
        }
        let entry = by_index.entry(index).or_insert_with(|| ProfileSlot {
            index,
            ..ProfileSlot::default()
        });
        let mut touched = false;
        if let Some(name) = clean(seeds.filter_names.get(position).map(String::as_str)) {
            entry.name = Some(name);
            touched = true;
        }
        if let Some(offset) = seeds.focus_offsets.get(position) {
            entry.offset = Some(*offset);
            entry.offset_quality = Some(OffsetQuality::Estimated);
            touched = true;
        }
        if touched {
            applied += 1;
        }
    }
    if seeds.filter_names.len() != seeds.focus_offsets.len()
        && !seeds.filter_names.is_empty()
        && !seeds.focus_offsets.is_empty()
    {
        warnings.push(format!(
            "[filter_wheel] has {} filter_names and {} focus_offsets",
            seeds.filter_names.len(),
            seeds.focus_offsets.len()
        ));
    }

    profile.slots = by_index.into_values().collect();
    profile.seed_import = Some(SeedImport {
        imported_at: Some(now),
        names: seeds.filter_names.clone(),
        offsets: seeds.focus_offsets.clone(),
        applied_slots: applied,
    });
    let update = ProfileUpdate {
        identity_key: identity_key.to_string(),
        kind: DeviceKind::FilterWheel,
        patch: json!({
            "slots": profile.slots,
            "seed_import": profile.seed_import,
        }),
        updated_at: now,
    };
    Ok(SeedImportResult {
        update,
        profile,
        applied_slots: applied,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::filter::test_support::CollectingSink;
    use crate::equipment::testing::fake_handles::{fake_identity, RecordingSink};
    use crate::equipment::testing::manual_clock::ManualClock;

    fn device(names: &[Option<&str>], offsets: &[Option<i32>]) -> Vec<SlotInfo> {
        names
            .iter()
            .zip(offsets.iter())
            .enumerate()
            .map(|(i, (name, offset))| SlotInfo {
                index0: i as u32,
                device_name: name.map(str::to_string),
                device_focus_offset: *offset,
            })
            .collect()
    }

    fn seeds() -> FilterWheelConfig {
        FilterWheelConfig {
            filter_names: vec!["L".into(), "R".into(), "G".into()],
            focus_offsets: vec![0, 12, 8],
            ..FilterWheelConfig::default()
        }
    }

    #[test]
    fn names_and_offsets_follow_profile_then_device_then_default() {
        let slots = device(&[Some("Lum"), Some("  "), None], &[Some(5), None, Some(-3)]);
        let profile = FilterWheelProfile {
            slots: vec![ProfileSlot {
                index: 1,
                name: Some("Ha".into()),
                offset: Some(110),
                offset_quality: Some(OffsetQuality::Measured),
                ..ProfileSlot::default()
            }],
            settle_ms: Some(900),
            ..FilterWheelProfile::default()
        };
        let wheel = reconcile(&slots, Some(&profile));
        assert_eq!(wheel.len(), 3, "the slot count comes from the device");
        let s = &wheel.slots;
        assert_eq!(
            (s[0].slot.name.as_str(), s[0].slot.name_source),
            ("Lum", SlotValueSource::Device)
        );
        assert_eq!(
            (s[1].slot.name.as_str(), s[1].slot.name_source),
            ("Ha", SlotValueSource::Profile)
        );
        assert_eq!(
            (s[2].slot.name.as_str(), s[2].slot.name_source),
            ("Slot 3", SlotValueSource::None)
        );
        assert_eq!(
            (
                s[0].slot.offset,
                s[0].slot.offset_source,
                s[0].offset_quality
            ),
            (Some(5), SlotValueSource::Device, OffsetQuality::Estimated)
        );
        assert_eq!(
            (
                s[1].slot.offset,
                s[1].slot.offset_source,
                s[1].offset_quality
            ),
            (Some(110), SlotValueSource::Profile, OffsetQuality::Measured)
        );
        assert_eq!(s[2].slot.offset, Some(-3));
        assert_eq!(wheel.settle_ms, Some(900));
        assert!(wheel.differences.is_empty());
        assert_eq!(wheel.mismatch_event(), None);
    }

    #[test]
    fn toml_seeds_are_never_an_input_to_reconcile() {
        // A wheel without device offsets and without a profile has no
        // offsets at all, whatever `[filter_wheel] focus_offsets` says.
        let slots = device(&[Some("L"), Some("R"), Some("G")], &[None, None, None]);
        let config = seeds();
        assert!(!config.focus_offsets.is_empty());
        let wheel = reconcile(&slots, None);
        assert!(wheel.slots.iter().all(|slot| slot.slot.offset.is_none()));
        assert!(wheel
            .slots
            .iter()
            .all(|slot| slot.offset_quality == OffsetQuality::Unmeasured));
    }

    #[test]
    fn disagreements_are_listed_and_both_versions_are_kept() {
        let slots = device(&[Some("Red"), Some("G")], &[Some(10), Some(4)]);
        let profile = FilterWheelProfile {
            slots: vec![
                ProfileSlot {
                    index: 0,
                    name: Some("Ha".into()),
                    offset: Some(100),
                    ..ProfileSlot::default()
                },
                ProfileSlot {
                    index: 5,
                    name: Some("SII".into()),
                    ..ProfileSlot::default()
                },
            ],
            device_reported: Some(DeviceReported {
                slot_count: 7,
                ..DeviceReported::default()
            }),
            ..FilterWheelProfile::default()
        };
        let wheel = reconcile(&slots, Some(&profile));
        assert_eq!(wheel.slots[0].slot.name, "Ha");
        assert_eq!(wheel.slots[0].slot.offset, Some(100));
        assert_eq!(wheel.differences.len(), 4, "{:?}", wheel.differences);
        assert!(wheel.differences[0].contains("profile 'Ha', device 'Red'"));
        assert!(wheel.differences[1].contains("profile 100, device 10"));
        assert!(wheel.differences[2].contains("SII"));
        assert!(wheel.differences[3].contains("7 slots"));
        assert!(matches!(
            wheel.mismatch_event(),
            Some(EventKind::FilterConfigMismatch { differences }) if differences.len() == 4
        ));
        let now = Utc::now();
        let patch = wheel.device_patch("alpaca:wheel-1", now);
        assert_eq!(patch.kind, DeviceKind::FilterWheel);
        assert_eq!(patch.patch["device_reported"]["slot_count"], json!(2));
        assert_eq!(patch.patch["device_reported"]["names"], json!(["Red", "G"]));
        assert_eq!(patch.patch["device_reported"]["offsets"], json!([10, 4]));
        assert!(
            patch.patch.get("slots").is_none(),
            "recording the device version never touches profile values"
        );
    }

    #[test]
    fn connect_reconcile_publishes_the_mismatch_and_stores_the_device_version() {
        let clock = ManualClock::new();
        let events = CollectingSink::new();
        let recorder = RecordingSink::new();
        let identity = fake_identity(DeviceKind::FilterWheel);
        let slots = device(&[Some("Red")], &[Some(1)]);
        let profile = FilterWheelProfile {
            slots: vec![ProfileSlot {
                index: 0,
                name: Some("Ha".into()),
                ..ProfileSlot::default()
            }],
            ..FilterWheelProfile::default()
        };
        let wheel = reconcile_at_connect(
            &identity,
            &slots,
            Some(&profile),
            &clock,
            events.as_ref(),
            recorder.as_ref(),
        );
        assert_eq!(wheel.slots[0].slot.name, "Ha");
        assert_eq!(events.names(), vec!["filter_config_mismatch"]);
        assert_eq!(
            events.events()[0].connection_id,
            Some(identity.connection_id)
        );
        let records = recorder.records();
        assert!(matches!(
            &records[0],
            GroundTruthRecord::Profile(update) if update.identity_key == identity.identity_key
        ));
    }

    #[test]
    fn targets_resolve_by_slot_or_name_and_refuse_ambiguity() {
        let slots = device(
            &[Some("L"), Some("Ha"), Some("ha "), Some("OIII")],
            &[None, None, None, None],
        );
        let wheel = reconcile(&slots, None);
        assert_eq!(wheel.resolve(&FilterTarget::Slot(3)).unwrap(), 3);
        assert_eq!(
            wheel.resolve(&FilterTarget::Name(" oiii".into())).unwrap(),
            3
        );
        let err = wheel.resolve(&FilterTarget::Slot(4)).unwrap_err();
        assert_eq!(err.class, crate::equipment::types::ErrorClass::InvalidValue);
        assert!(wheel
            .resolve(&FilterTarget::Name("Ha".into()))
            .unwrap_err()
            .message
            .contains("ambiguous"));
        assert!(wheel.resolve(&FilterTarget::Name("SII".into())).is_err());
        assert_eq!(wheel.af_slot(Some("L")), Some(0));
        assert_eq!(wheel.af_slot(Some("")), None);
        assert_eq!(wheel.af_slot(None), None);
        assert!(ReconciledWheel::default()
            .resolve(&FilterTarget::Slot(0))
            .is_err());
    }

    #[test]
    fn seed_import_is_explicit_estimated_and_keeps_other_profile_slots() {
        let slots = device(&[Some("A"), Some("B")], &[Some(1), Some(2)]);
        let profile = FilterWheelProfile {
            slots: vec![ProfileSlot {
                index: 1,
                name: Some("Kept".into()),
                offset: Some(50),
                offset_quality: Some(OffsetQuality::Measured),
                throughput: Some(0.9),
                ..ProfileSlot::default()
            }],
            ..FilterWheelProfile::default()
        };
        let wheel = reconcile(&slots, Some(&profile));
        let config = FilterWheelConfig {
            filter_names: vec!["L".into()],
            focus_offsets: vec![7, 9, 11],
            ..FilterWheelConfig::default()
        };
        let now = Utc::now();
        let result = import_seed(&config, &wheel, "alpaca:w", now).unwrap();
        assert_eq!(result.applied_slots, 2);
        assert_eq!(result.warnings.len(), 2, "{:?}", result.warnings);
        let after = reconcile(&slots, Some(&result.profile));
        assert_eq!(after.slots[0].slot.name, "L");
        assert_eq!(after.slots[0].slot.offset, Some(7));
        assert_eq!(after.slots[0].offset_quality, OffsetQuality::Estimated);
        assert_eq!(after.slots[1].slot.name, "Kept");
        assert_eq!(after.slots[1].slot.offset, Some(9));
        assert_eq!(after.slots[1].offset_quality, OffsetQuality::Estimated);
        assert_eq!(after.slots[1].throughput, Some(0.9));
        assert_eq!(result.update.patch["slots"][0]["offset"], json!(7));
        assert_eq!(
            result.update.patch["seed_import"]["applied_slots"],
            json!(2)
        );

        let empty = FilterWheelConfig::default();
        assert!(import_seed(&empty, &wheel, "alpaca:w", now).is_err());
        assert!(import_seed(&seeds(), &ReconciledWheel::default(), "alpaca:w", now).is_err());
    }

    #[test]
    fn profiles_parse_tolerantly_and_round_trip() {
        let value = json!({
            "slots": [{"index": 2, "name": "SII", "offset": 30, "offset_quality": "measured"}],
            "settle_ms": 250,
            "last_used": {"driver": "alpaca"},
            "qualification": {"state": "planned"}
        });
        let profile = FilterWheelProfile::from_profile_json(&value).unwrap();
        assert_eq!(profile.slot(2).unwrap().name.as_deref(), Some("SII"));
        assert_eq!(profile.settle_ms, Some(250));
        let back: FilterWheelProfile =
            serde_json::from_value(serde_json::to_value(&profile).unwrap()).unwrap();
        assert_eq!(back, profile);
        assert_eq!(
            FilterWheelProfile::from_profile_json(&Value::Null).unwrap(),
            FilterWheelProfile::default()
        );
        assert!(FilterWheelProfile::from_profile_json(&json!({"slots": "nope"})).is_err());
    }
}

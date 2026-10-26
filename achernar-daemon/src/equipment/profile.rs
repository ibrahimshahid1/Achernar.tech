//! Identity-keyed device profiles (spec §2.6, §2.15, §4.3, D17; table
//! `device_profiles`). Owner: WP1A-9.
//!
//! Rig-specific numbers come only from here, keyed by the stable identity key
//! of §2.6, never from global TOML (D17). A profile is one JSON document per
//! identity; [`ProfileUpdate`] patches it with RFC 7396 JSON merge-patch
//! semantics (objects merge, `null` deletes a key, anything else replaces),
//! applied inside SQLite with `json_patch` so concurrent writers never lose
//! each other's keys.
//!
//! Two write paths:
//! - [`ProfileStore::merge`]: awaited, for operator edits (import-seed,
//!   set-position, backlash, roles). The caller learns whether it landed.
//! - `GroundTruthRecord::Profile` through the recorder: fire-and-forget, for
//!   worker-originated updates such as the relative-focuser position after
//!   every completed move (F-b). If the recorder ever drops a focuser update,
//!   it marks that focuser's position unknown (see `recorder.rs`), so a lost
//!   write can never resurrect a stale origin.
//!
//! Document layout (every section optional; unknown keys are kept):
//!
//! ```text
//! last_used     { driver, option, name, endpoint, connection_id, at }   (§2.6)
//! focuser       { backlash, backlash_owner, min_position, max_position,
//!                 settle_ms, temperature_role,
//!                 relative_position { position, state, at, connection_id,
//!                                     move_id, cause } }               (F-b, §2.15)
//! filter_wheel  { settle_ms, seed_imported_at,
//!                 slots { "<index>": { name, offset, offset_status,
//!                                      throughput, bandwidth_nm } } } (W-b, D17)
//! mount         { site_sync, focal_length_mm, aperture_mm }            (M-b, M-m)
//! rotator       { sync_offset_deg }                                    (§3.6)
//! switch        { roles { "<channel id>": "<role>" } }                (§3.8)
//! qualification { "<record id>": { .. } }                              (D21)
//! ```
//!
//! Collections are maps, not arrays, because a merge patch replaces arrays
//! whole: patching one filter slot must not erase the others.
//!
//! Restart rule (§2.15): a relative focuser's logical position is restored as
//! `emulated_restored` only when the stored record was written after a
//! completed move (or an operator declaration) with no unknown close since.
//! Anything else is `emulated_origin_unknown`.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sqlx::{Executor, Row, Sqlite, SqlitePool};
use uuid::Uuid;

use super::events::ProfileUpdate;
use super::snapshot::{BacklashModel, BacklashOwner, PositionSource};
use super::types::{kind_name, DeviceIdentity, DeviceKind, DriverId, TemperatureRole};
use crate::storage::columns;

/// Top-level section names of a profile document.
pub mod keys {
    pub const LAST_USED: &str = "last_used";
    pub const FOCUSER: &str = "focuser";
    pub const RELATIVE_POSITION: &str = "relative_position";
    pub const FILTER_WHEEL: &str = "filter_wheel";
    pub const MOUNT: &str = "mount";
    pub const ROTATOR: &str = "rotator";
    pub const SWITCH: &str = "switch";
    pub const QUALIFICATION: &str = "qualification";
}

/// Profile times are stored as `columns::time` text (milliseconds, `Z`), so
/// they order correctly inside SQLite JSON.
mod ms_time {
    use chrono::{DateTime, Utc};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(at: &DateTime<Utc>, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&crate::storage::columns::time(at))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<DateTime<Utc>, D::Error> {
        let text = String::deserialize(deserializer)?;
        crate::storage::columns::parse_time(&text)
            .ok_or_else(|| serde::de::Error::custom(format!("invalid time {text:?}")))
    }

    pub mod opt {
        use chrono::{DateTime, Utc};
        use serde::{Deserialize, Deserializer, Serializer};

        pub fn serialize<S: Serializer>(
            at: &Option<DateTime<Utc>>,
            serializer: S,
        ) -> Result<S::Ok, S::Error> {
            match at {
                Some(at) => super::serialize(at, serializer),
                None => serializer.serialize_none(),
            }
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(
            deserializer: D,
        ) -> Result<Option<DateTime<Utc>>, D::Error> {
            Ok(Option::<String>::deserialize(deserializer)?
                .as_deref()
                .and_then(crate::storage::columns::parse_time))
        }
    }
}

// ---------------------------------------------------------------------------
// Typed sections
// ---------------------------------------------------------------------------

/// The last device used per kind, so the UI can offer a reconnect without a
/// scan (NINA's `OfflineDevice`, §2.6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LastUsed {
    pub driver: DriverId,
    /// The `/connect` option, e.g. `http://esp32.local:80#device=0`.
    pub option: String,
    pub name: Option<String>,
    pub endpoint: Option<String>,
    pub connection_id: Option<Uuid>,
    #[serde(with = "ms_time")]
    pub at: DateTime<Utc>,
}

/// How the last relative-focuser position record was closed (§2.15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionCloseState {
    /// Written after a completed move; the position is the logical position.
    Completed,
    /// Declared by the operator (`focuser/set-position`) or synced (INDI
    /// `FOCUS_SYNC`); the position is known without a move.
    Declared,
    /// A move was started and its completion was never recorded.
    InMotion,
    /// The position is unknown: a failed, aborted or timed-out move, a lost
    /// connection, a reconnect, or a dropped profile write.
    Unknown,
}

/// The relative-focuser logical position and its close state (F-b, §2.15).
/// Serialised with explicit nulls, so each record fully replaces the last.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelativePosition {
    pub position: Option<i32>,
    pub state: PositionCloseState,
    #[serde(with = "ms_time")]
    pub at: DateTime<Utc>,
    pub connection_id: Option<Uuid>,
    pub move_id: Option<Uuid>,
    pub cause: Option<String>,
}

impl RelativePosition {
    /// After a completed move: the new logical position.
    pub fn completed(
        position: i32,
        connection_id: Uuid,
        move_id: Option<Uuid>,
        at: DateTime<Utc>,
    ) -> Self {
        Self {
            position: Some(position),
            state: PositionCloseState::Completed,
            at,
            connection_id: Some(connection_id),
            move_id,
            cause: None,
        }
    }

    /// Before a move's initiator is sent. Until the completed record lands,
    /// a restart restores nothing.
    pub fn in_motion(connection_id: Uuid, move_id: Option<Uuid>, at: DateTime<Utc>) -> Self {
        Self {
            position: None,
            state: PositionCloseState::InMotion,
            at,
            connection_id: Some(connection_id),
            move_id,
            cause: None,
        }
    }

    /// The origin is lost, with the reason (`reconnect`, `move_timed_out`,
    /// `connection_lost`, ...).
    pub fn unknown(
        connection_id: Option<Uuid>,
        cause: impl Into<String>,
        at: DateTime<Utc>,
    ) -> Self {
        Self {
            position: None,
            state: PositionCloseState::Unknown,
            at,
            connection_id,
            move_id: None,
            cause: Some(cause.into()),
        }
    }

    /// An operator declaration or a device sync of the logical position.
    pub fn declared(position: i32, cause: impl Into<String>, at: DateTime<Utc>) -> Self {
        Self {
            position: Some(position),
            state: PositionCloseState::Declared,
            at,
            connection_id: None,
            move_id: None,
            cause: Some(cause.into()),
        }
    }
}

/// What a relative focuser starts from after a restart (§2.15, F-b).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RestoredPosition {
    /// `Some` only when `source` is `EmulatedRestored`.
    pub position: Option<i32>,
    pub source: PositionSource,
    /// Why the origin is or is not restored.
    pub reason: String,
    pub record_at: Option<DateTime<Utc>>,
}

/// Apply the restart rule to a stored record.
pub fn restore_rule(record: Option<&RelativePosition>) -> RestoredPosition {
    let unknown = |reason: String, at: Option<DateTime<Utc>>| RestoredPosition {
        position: None,
        source: PositionSource::EmulatedOriginUnknown,
        reason,
        record_at: at,
    };
    let Some(record) = record else {
        return unknown("no_record".into(), None);
    };
    match (record.state, record.position) {
        (PositionCloseState::Completed | PositionCloseState::Declared, Some(position)) => {
            RestoredPosition {
                position: Some(position),
                source: PositionSource::EmulatedRestored,
                reason: match record.state {
                    PositionCloseState::Completed => "completed_move".into(),
                    _ => "declared".into(),
                },
                record_at: Some(record.at),
            }
        }
        (PositionCloseState::Completed | PositionCloseState::Declared, None) => {
            unknown("no_position".into(), Some(record.at))
        }
        (PositionCloseState::InMotion, _) => unknown("closed_in_motion".into(), Some(record.at)),
        (PositionCloseState::Unknown, _) => unknown(
            record
                .cause
                .clone()
                .unwrap_or_else(|| "unknown_close".into()),
            Some(record.at),
        ),
    }
}

/// Focuser profile values (F-a, F-e, F-f).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct FocuserProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backlash: Option<BacklashModel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backlash_owner: Option<BacklashOwner>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_position: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_position: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub settle_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature_role: Option<TemperatureRole>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relative_position: Option<RelativePosition>,
}

/// One filter slot's profile values (W-b, FW-1).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct FilterSlotProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Focuser steps relative to the reference filter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<i32>,
    /// `measured`, `estimated` or `seed` (imported from TOML, D17).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub throughput: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bandwidth_nm: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct FilterWheelProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub settle_ms: Option<u64>,
    #[serde(
        default,
        with = "ms_time::opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub seed_imported_at: Option<DateTime<Utc>>,
    /// Keyed by the 0-based slot index as a string.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub slots: BTreeMap<String, FilterSlotProfile>,
}

impl FilterWheelProfile {
    pub fn slot(&self, index0: u32) -> Option<&FilterSlotProfile> {
        self.slots.get(&index0.to_string())
    }
}

/// Mount profile values (M-b site-sync choice, M-m optics; the rig profile
/// stays authoritative over the driver's optics claims).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct MountProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub site_sync: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focal_length_mm: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aperture_mm: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RotatorProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sync_offset_deg: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SwitchProfile {
    /// Channel id (as a string) to role: `dew_heater`, `power`, `fan`, ...
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub roles: BTreeMap<String, String>,
}

// ---------------------------------------------------------------------------
// Patch builders
// ---------------------------------------------------------------------------

/// Builders for [`ProfileUpdate`]s, so every package writes the same layout.
pub mod patch {
    use super::*;

    /// A merge patch of one top-level section.
    pub fn section<T: Serialize>(
        identity_key: &str,
        kind: DeviceKind,
        section: &str,
        value: &T,
        updated_at: DateTime<Utc>,
    ) -> ProfileUpdate {
        let mut patch = serde_json::Map::new();
        patch.insert(
            section.to_string(),
            serde_json::to_value(value).unwrap_or(serde_json::Value::Null),
        );
        ProfileUpdate {
            identity_key: identity_key.to_string(),
            kind,
            patch: serde_json::Value::Object(patch),
            updated_at,
        }
    }

    /// Record the device just connected as the last used of its kind.
    pub fn last_used(identity: &DeviceIdentity, option: &str, at: DateTime<Utc>) -> ProfileUpdate {
        let value = LastUsed {
            driver: identity.driver,
            option: option.to_string(),
            name: Some(identity.name.clone()),
            endpoint: Some(identity.endpoint.describe()),
            connection_id: Some(identity.connection_id),
            at,
        };
        section(
            &identity.identity_key,
            identity.kind,
            keys::LAST_USED,
            &value,
            at,
        )
    }

    /// Replace the relative-focuser position record.
    pub fn relative_position(identity_key: &str, record: &RelativePosition) -> ProfileUpdate {
        let focuser = serde_json::json!({ keys::RELATIVE_POSITION: record });
        section(
            identity_key,
            DeviceKind::Focuser,
            keys::FOCUSER,
            &focuser,
            record.at,
        )
    }

    /// Import the `[filter_wheel]` TOML seeds into a wheel's profile, the
    /// only way they ever reach hardware (D17, `POST filter-wheel/import-seed`).
    /// Offsets are marked `seed` so they are never mistaken for measured ones.
    pub fn filter_seed_import(
        identity_key: &str,
        names: &[String],
        offsets: &[i32],
        at: DateTime<Utc>,
    ) -> ProfileUpdate {
        let count = names.len().max(offsets.len());
        let slots = (0..count)
            .map(|index| {
                let offset = offsets.get(index).copied();
                let slot = FilterSlotProfile {
                    name: names.get(index).cloned(),
                    offset,
                    offset_status: offset.map(|_| "seed".to_string()),
                    ..FilterSlotProfile::default()
                };
                (index.to_string(), slot)
            })
            .collect();
        let wheel = FilterWheelProfile {
            seed_imported_at: Some(at),
            slots,
            ..FilterWheelProfile::default()
        };
        section(
            identity_key,
            DeviceKind::FilterWheel,
            keys::FILTER_WHEEL,
            &wheel,
            at,
        )
    }

    /// Add a qualification record (D21) under its id.
    pub fn qualification_record(
        identity_key: &str,
        kind: DeviceKind,
        record_id: &str,
        details: serde_json::Value,
        at: DateTime<Utc>,
    ) -> ProfileUpdate {
        let records = serde_json::json!({ record_id: details });
        section(identity_key, kind, keys::QUALIFICATION, &records, at)
    }
}

// ---------------------------------------------------------------------------
// Stored profiles
// ---------------------------------------------------------------------------

/// One `device_profiles` row.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DeviceProfile {
    pub identity_key: String,
    pub kind: Option<DeviceKind>,
    pub doc: serde_json::Value,
    pub updated_at: Option<DateTime<Utc>>,
}

impl DeviceProfile {
    /// One value by path, e.g. `["focuser", "relative_position"]`. A value
    /// whose shape does not match `T` is treated as absent.
    pub fn get<T: DeserializeOwned>(&self, path: &[&str]) -> Option<T> {
        let mut value = &self.doc;
        for key in path {
            value = value.get(key)?;
        }
        if value.is_null() {
            return None;
        }
        serde_json::from_value(value.clone()).ok()
    }

    pub fn last_used(&self) -> Option<LastUsed> {
        self.get(&[keys::LAST_USED])
    }

    pub fn focuser(&self) -> Option<FocuserProfile> {
        self.get(&[keys::FOCUSER])
    }

    /// Read on its own, so a malformed sibling field cannot hide it.
    pub fn relative_position(&self) -> Option<RelativePosition> {
        self.get(&[keys::FOCUSER, keys::RELATIVE_POSITION])
    }

    pub fn filter_wheel(&self) -> Option<FilterWheelProfile> {
        self.get(&[keys::FILTER_WHEEL])
    }

    pub fn mount(&self) -> Option<MountProfile> {
        self.get(&[keys::MOUNT])
    }

    pub fn rotator(&self) -> Option<RotatorProfile> {
        self.get(&[keys::ROTATOR])
    }

    pub fn switch(&self) -> Option<SwitchProfile> {
        self.get(&[keys::SWITCH])
    }
}

/// A patch that is not a JSON object would replace the whole document.
pub fn validate_patch(update: &ProfileUpdate) -> Result<(), String> {
    if update.identity_key.trim().is_empty() {
        return Err("profile update without an identity key".into());
    }
    if !update.patch.is_object() {
        return Err(format!(
            "profile patch for {} is not a JSON object",
            update.identity_key
        ));
    }
    Ok(())
}

/// Apply one update in a single statement (SQLite `json_patch`, RFC 7396).
/// Used by the recorder inside its batch transaction and by
/// [`ProfileStore::merge`]. The caller validates with [`validate_patch`].
pub async fn apply_update<'c, E>(executor: E, update: &ProfileUpdate) -> Result<(), sqlx::Error>
where
    E: Executor<'c, Database = Sqlite>,
{
    sqlx::query(
        r#"
        INSERT INTO device_profiles (identity_key, kind, profile_json, updated_at)
        VALUES (?1, ?2, json_patch('{}', ?3), ?4)
        ON CONFLICT(identity_key) DO UPDATE SET
            kind         = excluded.kind,
            profile_json = json_patch(
                CASE WHEN json_valid(device_profiles.profile_json)
                     THEN device_profiles.profile_json ELSE '{}' END,
                ?3),
            updated_at   = excluded.updated_at
        "#,
    )
    .bind(&update.identity_key)
    .bind(kind_name(update.kind))
    .bind(update.patch.to_string())
    .bind(columns::time(&update.updated_at))
    .execute(executor)
    .await?;
    Ok(())
}

/// A `focuser_profiles` row (migrations 002, 005, 009): the V-curve scale
/// and drift coefficient measured for a focuser.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FocuserScaleRow {
    pub id: String,
    pub rig_id: String,
    pub focuser_name: String,
    pub identity_key: Option<String>,
    pub steps_per_micron: Option<f64>,
    pub scale_measured_at: Option<String>,
    pub temp_coefficient: Option<f64>,
}

/// Display names that never identify a device; rows keyed by them are not
/// re-keyed automatically.
fn is_ambiguous_name(name: &str) -> bool {
    let name = name.trim();
    name.is_empty() || name.eq_ignore_ascii_case("unknown")
}

/// The `device_profiles` store.
#[derive(Debug, Clone)]
pub struct ProfileStore {
    pool: SqlitePool,
}

impl ProfileStore {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    pub async fn load(&self, identity_key: &str) -> Result<Option<DeviceProfile>, sqlx::Error> {
        let row = sqlx::query(
            "SELECT identity_key, kind, profile_json, updated_at FROM device_profiles \
             WHERE identity_key = ?",
        )
        .bind(identity_key)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|row| {
            let kind: String = row.get("kind");
            let doc: String = row.get("profile_json");
            let updated_at: String = row.get("updated_at");
            DeviceProfile {
                identity_key: row.get("identity_key"),
                kind: columns::from_name(Some(&kind)),
                doc: serde_json::from_str(&doc).unwrap_or_else(|_| serde_json::json!({})),
                updated_at: columns::parse_time(&updated_at),
            }
        }))
    }

    /// Apply an update now and wait for it (the operator path).
    pub async fn merge(&self, update: &ProfileUpdate) -> Result<(), ProfileError> {
        validate_patch(update).map_err(ProfileError::Invalid)?;
        apply_update(&self.pool, update).await?;
        Ok(())
    }

    /// The restart rule for a relative focuser (§2.15).
    pub async fn restore_relative_position(
        &self,
        identity_key: &str,
    ) -> Result<RestoredPosition, sqlx::Error> {
        let record = self
            .load(identity_key)
            .await?
            .and_then(|profile| profile.relative_position());
        Ok(restore_rule(record.as_ref()))
    }

    /// The most recently used device of `kind`, with its identity key.
    pub async fn last_used(
        &self,
        kind: DeviceKind,
    ) -> Result<Option<(String, LastUsed)>, sqlx::Error> {
        let rows = sqlx::query(
            r#"
            SELECT identity_key, json_extract(profile_json, '$.last_used') AS last_used
            FROM device_profiles
            WHERE kind = ? AND json_valid(profile_json)
              AND json_extract(profile_json, '$.last_used.at') IS NOT NULL
            ORDER BY json_extract(profile_json, '$.last_used.at') DESC
            "#,
        )
        .bind(kind_name(kind))
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.into_iter().find_map(|row| {
            let key: String = row.get("identity_key");
            let text: Option<String> = row.get("last_used");
            columns::parse_json::<LastUsed>(text.as_deref()).map(|last| (key, last))
        }))
    }

    /// Re-key a `focuser_profiles` row written under a display name (the
    /// pre-equipment-layer key) to `identity_key`, on the first connect of a
    /// matching identity (§4.3). At most one row is re-keyed per rig, the most
    /// recently updated match, and only while the rig has no row for this
    /// identity yet. Ambiguous names (`""`, `unknown`) are never matched: a
    /// scale measured on an unidentified focuser must not be attributed to a
    /// specific device. Returns the number of rows re-keyed (0 or 1).
    pub async fn rekey_focuser_profiles(
        &self,
        identity_key: &str,
        rig_id: &str,
        legacy_names: &[&str],
    ) -> Result<u64, sqlx::Error> {
        let names: Vec<&str> = legacy_names
            .iter()
            .copied()
            .filter(|name| !is_ambiguous_name(name))
            .collect();
        if names.is_empty() || identity_key.trim().is_empty() {
            return Ok(0);
        }
        let result = sqlx::query(
            r#"
            UPDATE focuser_profiles
            SET identity_key = ?1, updated_at = datetime('now')
            WHERE id = (
                SELECT id FROM focuser_profiles
                WHERE rig_id = ?2 AND identity_key IS NULL
                  AND focuser_name IN (SELECT value FROM json_each(?3))
                ORDER BY updated_at DESC, rowid DESC
                LIMIT 1
            )
            AND NOT EXISTS (
                SELECT 1 FROM focuser_profiles WHERE rig_id = ?2 AND identity_key = ?1
            )
            "#,
        )
        .bind(identity_key)
        .bind(rig_id)
        .bind(columns::json(&names))
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    /// The `focuser_profiles` row of one identity on one rig.
    pub async fn focuser_scale(
        &self,
        identity_key: &str,
        rig_id: &str,
    ) -> Result<Option<FocuserScaleRow>, sqlx::Error> {
        let row = sqlx::query(
            "SELECT id, rig_id, focuser_name, identity_key, steps_per_micron, scale_measured_at, \
             temp_coefficient FROM focuser_profiles WHERE identity_key = ? AND rig_id = ? \
             ORDER BY updated_at DESC LIMIT 1",
        )
        .bind(identity_key)
        .bind(rig_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|row| FocuserScaleRow {
            id: row.get("id"),
            rig_id: row.get("rig_id"),
            focuser_name: row.get("focuser_name"),
            identity_key: row.get("identity_key"),
            steps_per_micron: row.get("steps_per_micron"),
            scale_measured_at: row.get("scale_measured_at"),
            temp_coefficient: row.get("temp_coefficient"),
        }))
    }
}

/// Why an awaited profile write failed.
#[derive(Debug)]
pub enum ProfileError {
    /// The update was refused before reaching the database.
    Invalid(String),
    Database(sqlx::Error),
}

impl std::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProfileError::Invalid(reason) => write!(f, "invalid profile update: {reason}"),
            ProfileError::Database(err) => write!(f, "profile store: {err}"),
        }
    }
}

impl std::error::Error for ProfileError {}

impl From<sqlx::Error> for ProfileError {
    fn from(err: sqlx::Error) -> Self {
        ProfileError::Database(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::testing::manual_clock::ManualClock;
    use crate::equipment::traits::Clock;
    use crate::equipment::types::Endpoint;
    use serde_json::json;

    fn update(key: &str, patch: serde_json::Value, at: DateTime<Utc>) -> ProfileUpdate {
        ProfileUpdate {
            identity_key: key.into(),
            kind: DeviceKind::Focuser,
            patch,
            updated_at: at,
        }
    }

    #[tokio::test]
    async fn updates_merge_by_rfc7396() {
        let store = ProfileStore::new(crate::storage::test_pool().await);
        let clock = ManualClock::new();
        let key = "alpaca:esp32-focuser";
        store
            .merge(&update(
                key,
                json!({"focuser": {"settle_ms": 500, "min_position": 0, "max_position": 30000},
                       "notes": ["a", "b"]}),
                clock.now_utc(),
            ))
            .await
            .unwrap();
        clock.advance_ms(1000);
        store
            .merge(&update(
                key,
                json!({"focuser": {"settle_ms": 750, "min_position": null}, "notes": ["c"]}),
                clock.now_utc(),
            ))
            .await
            .unwrap();
        let profile = store.load(key).await.unwrap().unwrap();
        assert_eq!(
            profile.doc,
            json!({"focuser": {"settle_ms": 750, "max_position": 30000}, "notes": ["c"]}),
            "objects merge, null deletes, arrays replace"
        );
        assert_eq!(profile.kind, Some(DeviceKind::Focuser));
        assert_eq!(profile.updated_at, Some(clock.now_utc()));
        let focuser = profile.focuser().unwrap();
        assert_eq!(focuser.settle_ms, Some(750));
        assert_eq!(focuser.min_position, None);

        // A non-object patch would replace the document: refused.
        let err = store
            .merge(&update(key, json!("oops"), clock.now_utc()))
            .await
            .unwrap_err();
        assert!(matches!(err, ProfileError::Invalid(_)), "{err}");
        assert!(store.load("nobody").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn typed_section_patches_never_clobber_siblings() {
        let store = ProfileStore::new(crate::storage::test_pool().await);
        let at = ManualClock::new().now_utc();
        let key = "alpaca:wheel-1";
        let names: Vec<String> = ["L", "R", "G"].iter().map(|s| s.to_string()).collect();
        store
            .merge(&patch::filter_seed_import(key, &names, &[0, 12], at))
            .await
            .unwrap();
        // A later measurement of slot 1 only.
        let mut slots = BTreeMap::new();
        slots.insert(
            "1".to_string(),
            FilterSlotProfile {
                offset: Some(15),
                offset_status: Some("measured".into()),
                ..FilterSlotProfile::default()
            },
        );
        let wheel = FilterWheelProfile {
            slots,
            ..FilterWheelProfile::default()
        };
        store
            .merge(&patch::section(
                key,
                DeviceKind::FilterWheel,
                keys::FILTER_WHEEL,
                &wheel,
                at,
            ))
            .await
            .unwrap();
        let wheel = store
            .load(key)
            .await
            .unwrap()
            .unwrap()
            .filter_wheel()
            .unwrap();
        assert_eq!(wheel.seed_imported_at, Some(at));
        assert_eq!(
            wheel.slot(0).unwrap().offset_status.as_deref(),
            Some("seed")
        );
        let slot1 = wheel.slot(1).unwrap();
        assert_eq!(slot1.name.as_deref(), Some("R"), "name kept");
        assert_eq!(slot1.offset, Some(15));
        assert_eq!(slot1.offset_status.as_deref(), Some("measured"));
        let slot2 = wheel.slot(2).unwrap();
        assert_eq!((slot2.offset, slot2.offset_status.as_deref()), (None, None));
    }

    #[tokio::test]
    async fn relative_position_close_states_follow_the_restart_rule() {
        let store = ProfileStore::new(crate::storage::test_pool().await);
        let clock = ManualClock::new();
        let key = "alpaca:relative-focuser";
        let connection = Uuid::new_v4();
        let write = |record: RelativePosition| {
            let store = store.clone();
            async move {
                store
                    .merge(&patch::relative_position(key, &record))
                    .await
                    .unwrap();
                store.restore_relative_position(key).await.unwrap()
            }
        };

        let none = store.restore_relative_position(key).await.unwrap();
        assert_eq!(none.source, PositionSource::EmulatedOriginUnknown);
        assert_eq!(none.reason, "no_record");

        let move_id = Uuid::new_v4();
        let restored = write(RelativePosition::completed(
            1200,
            connection,
            Some(move_id),
            clock.now_utc(),
        ))
        .await;
        assert_eq!(restored.source, PositionSource::EmulatedRestored);
        assert_eq!(restored.position, Some(1200));
        assert_eq!(restored.reason, "completed_move");

        // A move started and never completed: nothing is restored, and the
        // old position does not leak through the merge.
        clock.advance_ms(1000);
        let restored = write(RelativePosition::in_motion(
            connection,
            None,
            clock.now_utc(),
        ))
        .await;
        assert_eq!(restored.source, PositionSource::EmulatedOriginUnknown);
        assert_eq!(restored.position, None);
        assert_eq!(restored.reason, "closed_in_motion");
        let stored = store.load(key).await.unwrap().unwrap();
        assert_eq!(
            stored.doc["focuser"]["relative_position"]["position"],
            json!(null)
        );
        assert_eq!(
            stored.doc["focuser"]["relative_position"]["move_id"],
            json!(null)
        );

        // A completed move after it restores again.
        clock.advance_ms(1000);
        let restored = write(RelativePosition::completed(
            1450,
            connection,
            None,
            clock.now_utc(),
        ))
        .await;
        assert_eq!(restored.position, Some(1450));

        // An unknown close (reconnect) in between blocks the restore.
        clock.advance_ms(1000);
        let restored = write(RelativePosition::unknown(
            Some(connection),
            "reconnect",
            clock.now_utc(),
        ))
        .await;
        assert_eq!(restored.source, PositionSource::EmulatedOriginUnknown);
        assert_eq!(restored.reason, "reconnect");

        // An operator declaration makes the origin known again.
        clock.advance_ms(1000);
        let restored = write(RelativePosition::declared(
            1000,
            "operator_set_position",
            clock.now_utc(),
        ))
        .await;
        assert_eq!(restored.source, PositionSource::EmulatedRestored);
        assert_eq!(restored.position, Some(1000));
        assert_eq!(restored.reason, "declared");

        // A record that does not parse is treated as unknown, never as zero.
        store
            .merge(&update(
                key,
                json!({"focuser": {"relative_position": {"position": "x", "state": "completed"}}}),
                clock.now_utc(),
            ))
            .await
            .unwrap();
        let restored = store.restore_relative_position(key).await.unwrap();
        assert_eq!(restored.source, PositionSource::EmulatedOriginUnknown);
        assert_eq!(restored.position, None);
    }

    #[test]
    fn the_restart_rule_needs_a_position() {
        let at = ManualClock::new().now_utc();
        let mut record = RelativePosition::completed(10, Uuid::new_v4(), None, at);
        record.position = None;
        assert_eq!(restore_rule(Some(&record)).reason, "no_position");
        assert_eq!(
            restore_rule(Some(&record)).source,
            PositionSource::EmulatedOriginUnknown
        );
    }

    #[tokio::test]
    async fn last_used_is_the_newest_connect_per_kind() {
        let store = ProfileStore::new(crate::storage::test_pool().await);
        let clock = ManualClock::new();
        let mount = |key: &str, name: &str| {
            DeviceIdentity::new(
                DeviceKind::Mount,
                DriverId::Alpaca,
                Endpoint::AlpacaHttp {
                    base: "http://10.0.0.5:11111".into(),
                    device_type: "telescope".into(),
                    number: 0,
                },
                key,
                name,
            )
        };
        let old = mount("alpaca:old-mount", "Old");
        let new = mount("alpaca:new-mount", "New");
        store
            .merge(&patch::last_used(
                &old,
                "http://10.0.0.5:11111#device=0",
                clock.now_utc(),
            ))
            .await
            .unwrap();
        clock.advance_ms(60_000);
        store
            .merge(&patch::last_used(
                &new,
                "http://10.0.0.6:11111#device=0",
                clock.now_utc(),
            ))
            .await
            .unwrap();
        let (key, last) = store.last_used(DeviceKind::Mount).await.unwrap().unwrap();
        assert_eq!(key, "alpaca:new-mount");
        assert_eq!(last.option, "http://10.0.0.6:11111#device=0");
        assert_eq!(last.driver, DriverId::Alpaca);
        assert_eq!(last.connection_id, Some(new.connection_id));
        assert_eq!(last.at, clock.now_utc());
        assert!(store
            .last_used(DeviceKind::Focuser)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn focuser_profiles_are_rekeyed_once_on_first_connect() {
        let pool = crate::storage::test_pool().await;
        let store = ProfileStore::new(pool.clone());
        for (rig, id) in [("rig-1", "a"), ("rig-2", "b")] {
            sqlx::query("INSERT INTO rigs (id, name) VALUES (?, ?)")
                .bind(rig)
                .bind(id)
                .execute(&pool)
                .await
                .unwrap();
        }
        for (id, rig, name, scale) in [
            ("fp-1", "rig-1", "Alpaca focuser", 3.2),
            ("fp-2", "rig-2", "Alpaca focuser", 4.0),
            ("fp-3", "rig-1", "unknown", 9.9),
        ] {
            sqlx::query(
                "INSERT INTO focuser_profiles (id, rig_id, focuser_name, steps_per_micron) \
                 VALUES (?, ?, ?, ?)",
            )
            .bind(id)
            .bind(rig)
            .bind(name)
            .bind(scale)
            .execute(&pool)
            .await
            .unwrap();
        }
        let key = "alpaca:esp32-focuser";
        assert_eq!(
            store
                .rekey_focuser_profiles(key, "rig-1", &["Alpaca focuser", "unknown"])
                .await
                .unwrap(),
            1
        );
        let row = store.focuser_scale(key, "rig-1").await.unwrap().unwrap();
        assert_eq!(row.id, "fp-1");
        assert_eq!(row.steps_per_micron, Some(3.2));
        // The other rig's row and the ambiguous row are untouched.
        let keys: Vec<(String, Option<String>)> =
            sqlx::query_as("SELECT id, identity_key FROM focuser_profiles ORDER BY id")
                .fetch_all(&pool)
                .await
                .unwrap();
        assert_eq!(
            keys,
            vec![
                ("fp-1".to_string(), Some(key.to_string())),
                ("fp-2".to_string(), None),
                ("fp-3".to_string(), None),
            ]
        );
        // Second connect: nothing more to re-key.
        assert_eq!(
            store
                .rekey_focuser_profiles(key, "rig-1", &["Alpaca focuser"])
                .await
                .unwrap(),
            0
        );
        assert_eq!(
            store
                .rekey_focuser_profiles(key, "rig-1", &["unknown", " "])
                .await
                .unwrap(),
            0
        );
        assert!(store.focuser_scale(key, "rig-2").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn concurrent_writers_on_a_file_database_keep_every_key() {
        let dir = std::env::temp_dir().join(format!("achernar-profile-{}", Uuid::new_v4()));
        let db = dir.join("achernar.db");
        let local = crate::storage::local_store::LocalStore::open(db.to_str().unwrap())
            .await
            .unwrap();
        let store = ProfileStore::new(local.pool().clone());
        let at = ManualClock::new().now_utc();
        let key = "alpaca:shared";
        let mut tasks = Vec::new();
        for i in 0..16 {
            let store = store.clone();
            tasks.push(tokio::spawn(async move {
                let patch =
                    json!({ "switch": { "roles": { i.to_string(): format!("role-{i}") } } });
                store.merge(&update(key, patch, at)).await
            }));
        }
        for task in tasks {
            task.await.unwrap().unwrap();
        }
        let roles = store
            .load(key)
            .await
            .unwrap()
            .unwrap()
            .switch()
            .unwrap()
            .roles;
        assert_eq!(roles.len(), 16);
        drop(store);
        drop(local);
        let _ = std::fs::remove_dir_all(dir);
    }
}

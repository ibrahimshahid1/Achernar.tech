//! Camera-file retention is independent of a local archive commit.
//!
//! A12 must qualify persistence/recovery before a destructive policy can be
//! enabled. Until then no request or legacy Keep Images setting authorizes
//! camera-file deletion, and an unverified camera target is not a second copy.

use serde::{Deserialize, Serialize};

pub const RETENTION_POLICY_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RetentionPolicy {
    #[default]
    RetainOnCamera,
    DeleteAfterVerifiedLocalCommit,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct RetentionAcknowledgement {
    pub policy: RetentionPolicy,
    pub policy_version: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraCopyState {
    /// Reserved for a separately verified persistent camera copy.
    Retained,
    /// The sensor produced pixels directly, without a camera-authored file.
    NotApplicable,
    /// Not deleting a file does not prove the camera target is persistent.
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct RetentionDecision {
    pub requested_policy: RetentionPolicy,
    pub effective_policy: RetentionPolicy,
    pub policy_version: u32,
    pub camera_copy_state: CameraCopyState,
    pub camera_files_deleted: bool,
    pub deletion_eligible: bool,
    pub reason: String,
}

impl RetentionDecision {
    pub fn retaining(camera_authored_files: bool) -> Self {
        Self::for_request(RetentionPolicy::RetainOnCamera, camera_authored_files)
    }

    /// Preserve the request even when policy validation denies deletion. The
    /// effective policy remains retain until A12 qualifies reclamation.
    pub fn for_request(requested_policy: RetentionPolicy, camera_authored_files: bool) -> Self {
        Self {
            requested_policy,
            effective_policy: RetentionPolicy::RetainOnCamera,
            policy_version: RETENTION_POLICY_VERSION,
            camera_copy_state: if camera_authored_files {
                CameraCopyState::Unknown
            } else {
                CameraCopyState::NotApplicable
            },
            camera_files_deleted: false,
            deletion_eligible: false,
            reason: if camera_authored_files {
                "Camera originals were not deleted; persistent camera storage has not been verified."
            } else {
                "This sensor exposure has no camera-authored file to retain."
            }
            .to_string(),
        }
    }
}

/// Deliberately no acknowledgement or local-commit boolean can enable deletion.
/// A future qualified receipt requires an explicit implementation and review.
pub fn validate_retention_policy(policy: RetentionPolicy) -> Result<(), &'static str> {
    match policy {
        RetentionPolicy::RetainOnCamera => Ok(()),
        RetentionPolicy::DeleteAfterVerifiedLocalCommit => Err(
            "Camera-file deletion is disabled until verified persistence and recovery are qualified (A12). Use retain_on_camera.",
        ),
    }
}

pub fn validate_legacy_keep_images(value: f64) -> Result<(), &'static str> {
    if value.is_finite() && value == 1.0 {
        Ok(())
    } else {
        Err("Keep Images must remain enabled until verified persistence and recovery are qualified (A12).")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_retains_and_does_not_claim_a_verified_second_copy() {
        assert_eq!(RetentionPolicy::default(), RetentionPolicy::RetainOnCamera);
        let decision = RetentionDecision::retaining(true);
        assert_eq!(decision.camera_copy_state, CameraCopyState::Unknown);
        assert!(!decision.deletion_eligible);
        assert!(!decision.camera_files_deleted);
        let json = serde_json::to_value(&decision).unwrap();
        assert_eq!(json["effective_policy"], "retain_on_camera");
        assert_eq!(json["camera_copy_state"], "unknown");
        assert_eq!(json["policy_version"], 1);
        assert_eq!(
            serde_json::from_value::<RetentionDecision>(json).unwrap(),
            decision
        );
    }

    #[test]
    fn direct_sensor_exposures_have_no_camera_file_claim() {
        let decision = RetentionDecision::retaining(false);
        assert_eq!(decision.camera_copy_state, CameraCopyState::NotApplicable);
        assert!(!decision.camera_files_deleted);
        assert!(!decision.deletion_eligible);
    }

    #[test]
    fn destructive_policy_remains_unavailable() {
        assert!(validate_retention_policy(RetentionPolicy::RetainOnCamera).is_ok());
        assert!(
            validate_retention_policy(RetentionPolicy::DeleteAfterVerifiedLocalCommit)
                .unwrap_err()
                .contains("A12")
        );
        let decision =
            RetentionDecision::for_request(RetentionPolicy::DeleteAfterVerifiedLocalCommit, true);
        assert_eq!(
            decision.requested_policy,
            RetentionPolicy::DeleteAfterVerifiedLocalCommit
        );
        assert_eq!(decision.effective_policy, RetentionPolicy::RetainOnCamera);
        assert!(!decision.camera_files_deleted);
        assert!(!decision.deletion_eligible);
    }

    #[test]
    fn legacy_toggle_cannot_enable_deletion_or_accept_nonfinite_values() {
        assert!(validate_legacy_keep_images(1.0).is_ok());
        for value in [
            0.0,
            -1.0,
            0.5,
            2.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert!(validate_legacy_keep_images(value).is_err());
        }
    }
}

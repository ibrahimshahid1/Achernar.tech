//! `GET weather` (spec §2.12, §3.9). Owner: WP1B-1.
//!
//! The ObservingConditions reading with every field's state and age, from
//! the latest snapshot; never a driver call. NaN, ValueNotSet and
//! NotImplemented are null, never a synthesised value (D5).

use axum::{extract::State, Json};
use serde_json::Value;

use super::api_error::ApiError;
use super::equipment as eq;
use crate::state::AppState;

pub fn status_value(state: &AppState) -> Result<Value, ApiError> {
    let handle = eq::weather_handle(state)?;
    let identity = handle.identity();
    let mut block = crate::equipment::status::weather_block(
        &handle.snapshot(),
        &identity,
        &handle.health(),
        state.equipment.clock().now_mono_ns(),
    );
    block["identity"] = crate::equipment::spine::identity_json(&identity);
    Ok(block)
}

pub async fn status(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    status_value(&state).map(Json)
}

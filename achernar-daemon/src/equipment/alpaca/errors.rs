//! Alpaca response checks (spec §3.1 C-3, Alpaca API reference §6). Owner:
//! WP1A-1.
//!
//! A reply is checked strictly in this order, and a later step never runs
//! when an earlier one fails:
//!
//! 1. **HTTP status.** A non-200 reply carries a plain-text body: 400 means a
//!    bad request or an unknown member, 500 a device failure. It is mapped
//!    with [`DeviceError::from_http_status`] (probes then test
//!    [`DeviceError::is_unknown_member`]).
//! 2. **Content type.** It must be `application/json` (parameters such as
//!    `charset` are allowed). A different type (an HTML captive portal, a web
//!    server that is not Alpaca) is a transport failure. A *missing* header is
//!    tolerated when the body is a valid Alpaca JSON object, because some
//!    minimal embedded servers omit it; the client counts these replies.
//! 3. **`ErrorNumber` and `ErrorMessage`**, parsed before anything else in
//!    the body. A non-zero number is mapped with
//!    [`DeviceError::from_alpaca_error`]; `Value` is then ignored (it may be
//!    null or of any type on error).
//! 4. **`Value`**, parsed only after that. A GET must carry it; a void PUT may
//!    omit it.
//!
//! JSON member names are matched case-insensitively. Bare `NaN`/`Infinity`
//! tokens (emitted by some Python and .NET servers for unavailable
//! ObservingConditions values) are accepted and become the strings `"NaN"`,
//! `"Infinity"` and `"-Infinity"`, which [`expect_f64`] reads as non-finite
//! numbers for the weather drivers to publish as unknown.
//!
//! The typed extractors (`expect_*`) turn a `Value` into a Rust value; a wrong
//! type is a `Driver` failure naming the member, never a silent default.

use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::equipment::types::{DeviceError, DeviceResult, ErrorClass};

/// The JSON media type every Alpaca reply must carry.
pub const ALPACA_JSON_MIME: &str = "application/json";

/// Longest plain-text body quoted in an error message.
const MAX_QUOTED_BODY: usize = 512;

/// HTTP method of an Alpaca request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Verb {
    Get,
    Put,
}

impl Verb {
    pub fn as_str(self) -> &'static str {
        match self {
            Verb::Get => "GET",
            Verb::Put => "PUT",
        }
    }
}

/// A reply that passed every check.
#[derive(Debug, Clone, PartialEq)]
pub struct AlpacaReply {
    /// The `Value` member; `Null` for a void PUT.
    pub value: Value,
    /// The echoed `ClientTransactionID`, when the server sent one.
    pub client_transaction_id: Option<u32>,
    pub server_transaction_id: Option<u32>,
    /// True when the reply had no `Content-Type` header (step 2 leniency).
    pub content_type_missing: bool,
}

/// True for `application/json`, with or without parameters.
pub fn is_json_content_type(content_type: &str) -> bool {
    let media_type = content_type.split(';').next().unwrap_or_default().trim();
    media_type.eq_ignore_ascii_case(ALPACA_JSON_MIME)
}

/// Run the C-3 checks on one HTTP reply.
pub fn check_response(
    verb: Verb,
    status: u16,
    content_type: Option<&str>,
    body: &[u8],
) -> DeviceResult<AlpacaReply> {
    // Step 1: HTTP status, with the plain-text body.
    if status != 200 {
        let text = String::from_utf8_lossy(body);
        return Err(DeviceError::from_http_status(status, &truncate(&text)));
    }

    // Step 2: content type.
    let content_type = content_type.map(str::trim).filter(|ct| !ct.is_empty());
    if let Some(ct) = content_type {
        if !is_json_content_type(ct) {
            let text = String::from_utf8_lossy(body);
            return Err(DeviceError::transport(format!(
                "not an Alpaca reply: content type '{ct}' instead of '{ALPACA_JSON_MIME}' ({})",
                truncate(&text)
            )));
        }
    }
    let object = match parse_json_lenient(body) {
        Some(Value::Object(object)) => object,
        Some(other) => {
            return Err(DeviceError::driver(format!(
                "Alpaca reply is not a JSON object: {}",
                truncate(&other.to_string())
            )))
        }
        None => {
            let text = String::from_utf8_lossy(body);
            let class = if content_type.is_some() {
                ErrorClass::Driver
            } else {
                // No content type and no JSON: this is not an Alpaca server.
                ErrorClass::Transport
            };
            return Err(DeviceError::new(
                class,
                format!("Alpaca reply is not valid JSON: {}", truncate(&text)),
            ));
        }
    };

    // Step 3: ErrorNumber and ErrorMessage, before Value.
    let error_number = match field(&object, "ErrorNumber") {
        None | Some(Value::Null) => None,
        Some(raw) => Some(integral(raw).ok_or_else(|| {
            DeviceError::driver(format!("Alpaca reply has an invalid ErrorNumber: {raw}"))
        })?),
    };
    let has_value = field(&object, "Value").is_some();
    if error_number.is_none() && !has_value && verb == Verb::Get {
        return Err(DeviceError::driver(
            "Alpaca reply has neither ErrorNumber nor Value",
        ));
    }
    if let Some(number) = error_number.filter(|n| *n != 0) {
        let number = i32::try_from(number).map_err(|_| {
            DeviceError::driver(format!("Alpaca ErrorNumber {number} is out of range"))
        })?;
        let message = field(&object, "ErrorMessage")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();
        return Err(DeviceError::from_alpaca_error(number, &message));
    }

    // Step 4: Value.
    let value = match (verb, field(&object, "Value")) {
        (_, Some(value)) => value.clone(),
        (Verb::Put, None) => Value::Null,
        (Verb::Get, None) => {
            return Err(DeviceError::driver(
                "successful Alpaca GET reply has no Value",
            ))
        }
    };
    Ok(AlpacaReply {
        value,
        client_transaction_id: transaction_id(&object, "ClientTransactionID"),
        server_transaction_id: transaction_id(&object, "ServerTransactionID"),
        content_type_missing: content_type.is_none(),
    })
}

/// A JSON object member, matched exactly first and then case-insensitively.
pub fn field<'a>(object: &'a serde_json::Map<String, Value>, name: &str) -> Option<&'a Value> {
    object.get(name).or_else(|| {
        object
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value)
    })
}

fn transaction_id(object: &serde_json::Map<String, Value>, name: &str) -> Option<u32> {
    field(object, name)
        .and_then(integral)
        .and_then(|id| u32::try_from(id).ok())
}

/// An integer, also accepting an integral float such as `0.0`.
fn integral(value: &Value) -> Option<i64> {
    if let Some(n) = value.as_i64() {
        return Some(n);
    }
    let float = value.as_f64()?;
    if float.fract() == 0.0 && float.abs() < 9.0e15 {
        // Integral and well inside the exactly representable range.
        return Some(float as i64);
    }
    None
}

fn truncate(text: &str) -> String {
    let text = text.trim();
    if text.chars().count() <= MAX_QUOTED_BODY {
        return text.to_string();
    }
    let mut out: String = text.chars().take(MAX_QUOTED_BODY).collect();
    out.push_str("...");
    out
}

/// Parse JSON, tolerating bare `NaN`, `Infinity` and `-Infinity` tokens
/// outside strings. `None` when the body is not JSON even then.
pub fn parse_json_lenient(body: &[u8]) -> Option<Value> {
    if let Ok(value) = serde_json::from_slice::<Value>(body) {
        return Some(value);
    }
    let text = std::str::from_utf8(body).ok()?;
    let sanitized = quote_non_finite_literals(text)?;
    serde_json::from_str(&sanitized).ok()
}

/// Replace bare non-finite literals outside strings with quoted strings.
/// Returns `None` when nothing was replaced (the body is simply malformed).
fn quote_non_finite_literals(text: &str) -> Option<String> {
    const LITERALS: [&str; 3] = ["-Infinity", "Infinity", "NaN"];
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len() + 16);
    let mut in_string = false;
    let mut escaped = false;
    let mut replaced = false;
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
        } else if byte == b'"' {
            in_string = true;
        } else if let Some(literal) = LITERALS
            .iter()
            .find(|literal| text[index..].starts_with(**literal))
        {
            out.push('"');
            out.push_str(literal);
            out.push('"');
            index += literal.len();
            replaced = true;
            continue;
        }
        // Copy one whole UTF-8 character.
        let width = utf8_width(byte);
        out.push_str(&text[index..index + width]);
        index += width;
    }
    replaced.then_some(out)
}

fn utf8_width(first: u8) -> usize {
    match first {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

// ---------------------------------------------------------------------------
// Typed Value extraction
// ---------------------------------------------------------------------------

fn wrong_type(member: &str, expected: &str, got: &Value) -> DeviceError {
    let shown = got.to_string();
    DeviceError::driver(format!(
        "member '{member}': expected {expected}, got {}",
        truncate(&shown)
    ))
}

/// A number. The strings `"NaN"`, `"Infinity"` and `"-Infinity"` become
/// the non-finite values; drivers publish those as unknown, never as data.
pub fn expect_f64(member: &str, value: &Value) -> DeviceResult<f64> {
    if let Some(number) = value.as_f64() {
        return Ok(number);
    }
    match value.as_str().map(str::trim) {
        Some("NaN") => Ok(f64::NAN),
        Some("Infinity") => Ok(f64::INFINITY),
        Some("-Infinity") => Ok(f64::NEG_INFINITY),
        _ => Err(wrong_type(member, "a number", value)),
    }
}

/// A boolean. The strings `"true"`/`"false"` (any case) are accepted.
pub fn expect_bool(member: &str, value: &Value) -> DeviceResult<bool> {
    if let Some(flag) = value.as_bool() {
        return Ok(flag);
    }
    value
        .as_str()
        .and_then(super::form::parse_flag)
        .ok_or_else(|| wrong_type(member, "a boolean", value))
}

/// An integer, also accepting an integral float such as `3.0`.
pub fn expect_i64(member: &str, value: &Value) -> DeviceResult<i64> {
    integral(value).ok_or_else(|| wrong_type(member, "an integer", value))
}

pub fn expect_i32(member: &str, value: &Value) -> DeviceResult<i32> {
    let wide = expect_i64(member, value)?;
    i32::try_from(wide).map_err(|_| wrong_type(member, "a 32-bit integer", value))
}

pub fn expect_u32(member: &str, value: &Value) -> DeviceResult<u32> {
    let wide = expect_i64(member, value)?;
    u32::try_from(wide).map_err(|_| wrong_type(member, "a non-negative 32-bit integer", value))
}

pub fn expect_string(member: &str, value: &Value) -> DeviceResult<String> {
    value
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| wrong_type(member, "a string", value))
}

fn expect_array<'a>(member: &str, value: &'a Value) -> DeviceResult<&'a Vec<Value>> {
    value
        .as_array()
        .ok_or_else(|| wrong_type(member, "an array", value))
}

pub fn expect_string_list(member: &str, value: &Value) -> DeviceResult<Vec<String>> {
    expect_array(member, value)?
        .iter()
        .map(|item| expect_string(member, item))
        .collect()
}

pub fn expect_f64_list(member: &str, value: &Value) -> DeviceResult<Vec<f64>> {
    expect_array(member, value)?
        .iter()
        .map(|item| expect_f64(member, item))
        .collect()
}

pub fn expect_i64_list(member: &str, value: &Value) -> DeviceResult<Vec<i64>> {
    expect_array(member, value)?
        .iter()
        .map(|item| expect_i64(member, item))
        .collect()
}

/// An ISO 8601 date-time such as `utcdate`, parsed permissively (see
/// [`super::devicestate::parse_timestamp`]). A string without an offset is
/// read as UTC here, because `UTCDate` is defined as UTC.
pub fn expect_utc_datetime(member: &str, value: &Value) -> DeviceResult<DateTime<Utc>> {
    let text = value
        .as_str()
        .ok_or_else(|| wrong_type(member, "an ISO 8601 date-time string", value))?;
    super::devicestate::parse_timestamp_with(text, chrono::FixedOffset::east_opt(0).unwrap())
        .map(|stamp| stamp.utc)
        .ok_or_else(|| wrong_type(member, "an ISO 8601 date-time string", value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equipment::types::ascom_error;
    use serde_json::json;

    fn ok_body(value: Value) -> Vec<u8> {
        json!({"Value": value, "ClientTransactionID": 7, "ServerTransactionID": 99,
               "ErrorNumber": 0, "ErrorMessage": ""})
        .to_string()
        .into_bytes()
    }

    #[test]
    fn a_good_reply_yields_its_value_and_transaction_ids() {
        let reply = check_response(
            Verb::Get,
            200,
            Some("application/json"),
            &ok_body(json!(5.5)),
        )
        .unwrap();
        assert_eq!(reply.value, json!(5.5));
        assert_eq!(reply.client_transaction_id, Some(7));
        assert_eq!(reply.server_transaction_id, Some(99));
        assert!(!reply.content_type_missing);
        let reply = check_response(
            Verb::Get,
            200,
            Some("Application/JSON; charset=utf-8"),
            &ok_body(json!(true)),
        )
        .unwrap();
        assert_eq!(reply.value, json!(true));
    }

    #[test]
    fn http_status_is_checked_first_with_its_plain_text_body() {
        // Even a JSON-looking body is ignored on a non-200 status.
        let err = check_response(
            Verb::Get,
            400,
            Some("text/plain"),
            b"Unknown member 'devicestate'\r\n",
        )
        .unwrap_err();
        assert_eq!(err.http_status(), Some(400));
        assert_eq!(err.class, ErrorClass::InvalidValue);
        assert!(err.is_unknown_member());
        assert_eq!(err.message, "HTTP 400: Unknown member 'devicestate'");
        let err = check_response(Verb::Put, 500, Some("application/json"), &ok_body(json!(1)))
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::Driver);
        assert_eq!(err.http_status(), Some(500));
        let err = check_response(Verb::Get, 404, None, b"").unwrap_err();
        assert_eq!(err.class, ErrorClass::Unsupported);
        assert!(err.is_unknown_member());
        let long = "x".repeat(4000);
        let err = check_response(Verb::Get, 500, None, long.as_bytes()).unwrap_err();
        assert!(err.message.len() < 600, "long bodies are truncated");
    }

    #[test]
    fn a_non_json_content_type_is_refused_before_the_body_is_read() {
        let err =
            check_response(Verb::Get, 200, Some("text/html"), &ok_body(json!(1))).unwrap_err();
        assert_eq!(err.class, ErrorClass::Transport);
        assert!(err.message.contains("text/html"), "{}", err.message);
        // A missing content type is tolerated for a valid Alpaca object.
        let reply = check_response(Verb::Get, 200, None, &ok_body(json!(3))).unwrap();
        assert!(reply.content_type_missing);
        // ...but not for something that is not JSON at all.
        let err = check_response(Verb::Get, 200, None, b"<html>login</html>").unwrap_err();
        assert_eq!(err.class, ErrorClass::Transport);
        let err =
            check_response(Verb::Get, 200, Some("application/json"), b"{not json").unwrap_err();
        assert_eq!(err.class, ErrorClass::Driver);
    }

    #[test]
    fn error_number_is_parsed_before_value_and_a_null_value_is_fine_on_error() {
        let body = json!({"Value": null, "ErrorNumber": ascom_error::NOT_CONNECTED,
                          "ErrorMessage": "Not connected", "ClientTransactionID": 1});
        let err = check_response(
            Verb::Get,
            200,
            Some("application/json"),
            body.to_string().as_bytes(),
        )
        .unwrap_err();
        assert_eq!(err.class, ErrorClass::NotConnected);
        assert_eq!(err.alpaca_error_number(), Some(0x407));
        assert_eq!(err.message, "Not connected");
        // A Value of the wrong type is never looked at on error.
        let body = json!({"Value": {"weird": []}, "ErrorNumber": 0x40B, "ErrorMessage": "busy"});
        let err = check_response(
            Verb::Put,
            200,
            Some("application/json"),
            body.to_string().as_bytes(),
        )
        .unwrap_err();
        assert_eq!(err.class, ErrorClass::InvalidOperation);
        // No Value member at all on error.
        let body = json!({"ErrorNumber": 0x402, "ErrorMessage": "not set"});
        let err = check_response(
            Verb::Get,
            200,
            Some("application/json"),
            body.to_string().as_bytes(),
        )
        .unwrap_err();
        assert!(err.is_not_set());
        let body = json!({"ErrorNumber": "zero", "Value": 1});
        let err = check_response(
            Verb::Get,
            200,
            Some("application/json"),
            body.to_string().as_bytes(),
        )
        .unwrap_err();
        assert_eq!(err.class, ErrorClass::Driver);
        let body = json!({"ErrorNumber": 0.0, "Value": 1, "errormessage": ""});
        assert_eq!(
            check_response(
                Verb::Get,
                200,
                Some("application/json"),
                body.to_string().as_bytes()
            )
            .unwrap()
            .value,
            json!(1)
        );
    }

    #[test]
    fn get_needs_a_value_but_a_void_put_does_not() {
        let body = json!({"ErrorNumber": 0, "ErrorMessage": "", "ClientTransactionID": 3});
        let err = check_response(
            Verb::Get,
            200,
            Some("application/json"),
            body.to_string().as_bytes(),
        )
        .unwrap_err();
        assert_eq!(err.class, ErrorClass::Driver);
        let reply = check_response(
            Verb::Put,
            200,
            Some("application/json"),
            body.to_string().as_bytes(),
        )
        .unwrap();
        assert_eq!(reply.value, Value::Null);
        assert_eq!(reply.client_transaction_id, Some(3));
        // Member names are matched case-insensitively.
        let body = br#"{"value": 4, "errornumber": 0, "clienttransactionid": 9}"#;
        let reply = check_response(Verb::Get, 200, Some("application/json"), body).unwrap();
        assert_eq!(
            (reply.value, reply.client_transaction_id),
            (json!(4), Some(9))
        );
        let err = check_response(Verb::Get, 200, Some("application/json"), b"[1,2]").unwrap_err();
        assert_eq!(err.class, ErrorClass::Driver);
        let err = check_response(Verb::Get, 200, Some("application/json"), b"{}").unwrap_err();
        assert_eq!(err.class, ErrorClass::Driver);
    }

    #[test]
    fn bare_non_finite_literals_are_tolerated_outside_strings_only() {
        let body =
            br#"{"Value": NaN, "ErrorNumber": 0, "ErrorMessage": "NaN inside a string stays"}"#;
        let reply = check_response(Verb::Get, 200, Some("application/json"), body).unwrap();
        assert!(expect_f64("skyquality", &reply.value).unwrap().is_nan());
        let parsed = parse_json_lenient(br#"{"a": [Infinity, -Infinity], "b": "x\"NaN"}"#).unwrap();
        assert_eq!(parsed["a"][0], json!("Infinity"));
        assert_eq!(parsed["a"][1], json!("-Infinity"));
        assert_eq!(parsed["b"], json!("x\"NaN"));
        assert!(parse_json_lenient(b"{broken").is_none());
        assert!(parse_json_lenient("{\"s\": \"\u{e9}\", \"v\": NaN}".as_bytes()).is_some());
    }

    #[test]
    fn typed_extractors_refuse_wrong_types_with_the_member_name() {
        assert_eq!(expect_f64("ra", &json!(5)).unwrap(), 5.0);
        assert!(expect_f64("ra", &json!("Infinity")).unwrap().is_infinite());
        let err = expect_f64("rightascension", &json!("5.5h")).unwrap_err();
        assert_eq!(err.class, ErrorClass::Driver);
        assert!(err.message.contains("rightascension"));
        assert!(expect_bool("slewing", &json!(true)).unwrap());
        assert!(!expect_bool("slewing", &json!("False")).unwrap());
        assert!(expect_bool("slewing", &json!(1)).is_err());
        assert_eq!(expect_i64("position", &json!(1200.0)).unwrap(), 1200);
        assert!(expect_i64("position", &json!(1200.5)).is_err());
        assert_eq!(expect_i32("sideofpier", &json!(-1)).unwrap(), -1);
        assert!(expect_u32("position", &json!(-1)).is_err());
        assert!(expect_i32("x", &json!(5_000_000_000_i64)).is_err());
        assert_eq!(
            expect_string("name", &json!("ESP32 mount")).unwrap(),
            "ESP32 mount"
        );
        assert!(expect_string("name", &Value::Null).is_err());
        assert_eq!(
            expect_string_list("names", &json!(["L", "R"])).unwrap(),
            vec!["L".to_string(), "R".to_string()]
        );
        assert_eq!(
            expect_i64_list("focusoffsets", &json!([0, 12, -8])).unwrap(),
            vec![0, 12, -8]
        );
        assert_eq!(
            expect_f64_list("rates", &json!([0.5, 1])).unwrap(),
            vec![0.5, 1.0]
        );
        assert!(expect_string_list("names", &json!(["L", 3])).is_err());
        let utc = expect_utc_datetime("utcdate", &json!("2026-10-26T21:30:00.1234567Z")).unwrap();
        assert_eq!(utc.to_rfc3339(), "2026-10-26T21:30:00.123456700+00:00");
        let naive = expect_utc_datetime("utcdate", &json!("2026-10-26T21:30:00")).unwrap();
        assert_eq!(naive.to_rfc3339(), "2026-10-26T21:30:00+00:00");
        assert!(expect_utc_datetime("utcdate", &json!("yesterday")).is_err());
    }
}

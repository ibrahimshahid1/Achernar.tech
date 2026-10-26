//! Alpaca request encoding (spec §3.1 C-3, O7). Owner: WP1A-1.
//!
//! PUT bodies are `application/x-www-form-urlencoded`; GET parameters use
//! the same encoding in the query string. Parameter **keys keep their exact
//! case** (`RightAscension`, `Position`, `Brightness`, ...): the Alpaca API
//! makes PUT keys case-sensitive. Values are formatted in the invariant
//! culture: `.` as the decimal separator, no thousands separators, no
//! exponent, and booleans as `True`/`False` (the form the ASCOM Library
//! sends; servers compare them case-insensitively).
//!
//! Encoding follows the WHATWG `application/x-www-form-urlencoded`
//! serializer: ASCII alphanumerics and `*-._` pass through, a space becomes
//! `+`, and every other byte of the UTF-8 encoding is percent-encoded. The
//! `percent-encoding` crate does the byte work (owner decision O7).

use percent_encoding::{percent_decode_str, utf8_percent_encode, AsciiSet, NON_ALPHANUMERIC};

use crate::equipment::types::{DeviceError, DeviceResult};

/// Bytes the form serializer percent-encodes: everything except ASCII
/// alphanumerics and `*`, `-`, `.`, `_`. A space is handled separately.
const FORM_ENCODE_SET: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'*')
    .remove(b'-')
    .remove(b'.')
    .remove(b'_');

/// The PUT body content type.
pub const FORM_CONTENT_TYPE: &str = "application/x-www-form-urlencoded";

/// Parameter names the client adds to every request (C-3).
pub const CLIENT_ID_KEY: &str = "ClientID";
pub const CLIENT_TRANSACTION_ID_KEY: &str = "ClientTransactionID";

/// Encode one key or value.
pub fn encode_component(text: &str) -> String {
    // A literal '%' is itself encoded as "%25", so "%20" in the output can
    // only have come from a space.
    utf8_percent_encode(text, FORM_ENCODE_SET)
        .to_string()
        .replace("%20", "+")
}

/// Encode `key=value` pairs joined by `&`, keys in their exact case.
pub fn encode_pairs<K: AsRef<str>, V: AsRef<str>>(pairs: &[(K, V)]) -> String {
    let mut out = String::new();
    for (index, (key, value)) in pairs.iter().enumerate() {
        if index > 0 {
            out.push('&');
        }
        out.push_str(&encode_component(key.as_ref()));
        out.push('=');
        out.push_str(&encode_component(value.as_ref()));
    }
    out
}

/// Decode a form body or query string. Used by the in-process test server
/// and by tests; malformed UTF-8 is replaced, never fatal.
pub fn decode_pairs(text: &str) -> Vec<(String, String)> {
    text.split('&')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let (key, value) = part.split_once('=').unwrap_or((part, ""));
            (decode_component(key), decode_component(value))
        })
        .collect()
}

fn decode_component(text: &str) -> String {
    let spaced = text.replace('+', " ");
    percent_decode_str(&spaced).decode_utf8_lossy().into_owned()
}

/// An invariant-culture number. Non-finite values are refused: no Alpaca
/// member accepts NaN or infinity, and sending one would move hardware to
/// an undefined target.
pub fn num(value: f64) -> DeviceResult<String> {
    if !value.is_finite() {
        return Err(DeviceError::invalid_value(format!(
            "refusing to send the non-finite number {value} to a device"
        )));
    }
    // Rust's `Display` for f64 is locale-free, never uses an exponent and
    // prints the shortest text that round-trips.
    if value == 0.0 {
        // Avoid "-0", which some parsers reject.
        return Ok("0".into());
    }
    Ok(value.to_string())
}

/// An integer parameter.
pub fn int(value: impl Into<i64>) -> String {
    value.into().to_string()
}

/// A boolean parameter, `True` or `False`.
pub fn flag(value: bool) -> &'static str {
    if value {
        "True"
    } else {
        "False"
    }
}

/// Look up a decoded parameter. `case_sensitive` is true for PUT bodies and
/// false for GET query strings (the Alpaca API rule).
pub fn find<'a>(pairs: &'a [(String, String)], key: &str, case_sensitive: bool) -> Option<&'a str> {
    pairs
        .iter()
        .find(|(name, _)| {
            if case_sensitive {
                name == key
            } else {
                name.eq_ignore_ascii_case(key)
            }
        })
        .map(|(_, value)| value.as_str())
}

/// Parse an Alpaca boolean parameter value, case-insensitively.
pub fn parse_flag(text: &str) -> Option<bool> {
    let text = text.trim();
    if text.eq_ignore_ascii_case("true") {
        Some(true)
    } else if text.eq_ignore_ascii_case("false") {
        Some(false)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_characters_are_percent_encoded_and_spaces_become_plus() {
        assert_eq!(encode_component("abcXYZ019*-._"), "abcXYZ019*-._");
        assert_eq!(encode_component("a b"), "a+b");
        assert_eq!(encode_component("a+b"), "a%2Bb");
        assert_eq!(encode_component("100%"), "100%25");
        assert_eq!(encode_component("x=1&y=2"), "x%3D1%26y%3D2");
        assert_eq!(
            encode_component("/?#[]@!$'(),;:~"),
            "%2F%3F%23%5B%5D%40%21%24%27%28%29%2C%3B%3A%7E"
        );
        assert_eq!(encode_component("Lum\u{e9}"), "Lum%C3%A9");
        assert_eq!(encode_component(""), "");
    }

    #[test]
    fn pairs_keep_their_exact_case_and_round_trip() {
        let pairs = [
            ("RightAscension", "5.5".to_string()),
            ("Declination", "-5.25".to_string()),
            ("Action", "set dew heater=on & go".to_string()),
            ("Parameters", "{\"a\": 100%}".to_string()),
        ];
        let body = encode_pairs(&pairs);
        assert!(
            body.starts_with("RightAscension=5.5&Declination=-5.25&"),
            "{body}"
        );
        assert!(!body.contains(' '));
        let decoded = decode_pairs(&body);
        assert_eq!(decoded.len(), 4);
        for ((key, value), (dkey, dvalue)) in pairs.iter().zip(&decoded) {
            assert_eq!(key, dkey);
            assert_eq!(value, dvalue);
        }
        assert_eq!(find(&decoded, "rightascension", false), Some("5.5"));
        assert_eq!(find(&decoded, "rightascension", true), None);
        assert_eq!(find(&decoded, "RightAscension", true), Some("5.5"));
        assert_eq!(encode_pairs::<&str, &str>(&[]), "");
        assert_eq!(decode_pairs("a&b=&=c")[0], ("a".to_string(), String::new()));
    }

    #[test]
    fn numbers_use_the_invariant_culture_without_exponents() {
        assert_eq!(num(5.5).unwrap(), "5.5");
        assert_eq!(num(-5.25).unwrap(), "-5.25");
        assert_eq!(num(12.0).unwrap(), "12");
        assert_eq!(num(-0.0).unwrap(), "0");
        assert_eq!(num(1e-7).unwrap(), "0.0000001");
        assert_eq!(num(2.5e15).unwrap(), "2500000000000000");
        assert_eq!(num(0.1 + 0.2).unwrap(), "0.30000000000000004");
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = num(bad).unwrap_err();
            assert_eq!(err.class, crate::equipment::types::ErrorClass::InvalidValue);
        }
        assert_eq!(int(-42_i32), "-42");
        assert_eq!(int(4_000_000_000_u32), "4000000000");
        assert_eq!(flag(true), "True");
        assert_eq!(flag(false), "False");
        assert_eq!(parse_flag(" TRUE "), Some(true));
        assert_eq!(parse_flag("false"), Some(false));
        assert_eq!(parse_flag("1"), None);
    }
}

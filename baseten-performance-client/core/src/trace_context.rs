//! W3C Trace Context (`traceparent`) parsing and formatting.
//!
//! See https://www.w3.org/TR/trace-context/#traceparent-header.

use crate::errors::ClientError;
use rand::Rng;
use std::fmt;

pub const TRACEPARENT_HEADER_NAME: &str = "traceparent";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceParent {
    pub trace_id: [u8; 16],
    pub span_id: [u8; 8],
    pub flags: u8,
}

impl TraceParent {
    /// Parses a `traceparent` header value, rejecting anything a W3C-compliant receiver would.
    pub fn parse(value: &str) -> Result<Self, ClientError> {
        let invalid = |reason: &str| {
            ClientError::InvalidParameter(format!(
                "traceparent {:?} is not a valid W3C traceparent ({}); expected \
                 00-<32 hex trace id>-<16 hex span id>-<2 hex flags>",
                value, reason
            ))
        };

        let value = value.trim();
        let mut parts = value.split('-');
        let (Some(version), Some(trace_id), Some(span_id), Some(flags)) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return Err(invalid("wrong number of fields"));
        };
        let rest: Vec<&str> = parts.collect();

        let version = decode_hex::<1>(version).ok_or_else(|| invalid("bad version"))?[0];
        if version == 0xff {
            return Err(invalid("version ff is forbidden"));
        }
        // Version 00 has exactly four fields; later versions may append more.
        if version == 0 && !rest.is_empty() {
            return Err(invalid("version 00 has exactly four fields"));
        }

        let trace_id = decode_hex::<16>(trace_id).ok_or_else(|| invalid("bad trace id"))?;
        if trace_id == [0; 16] {
            return Err(invalid("all-zero trace id"));
        }
        let span_id = decode_hex::<8>(span_id).ok_or_else(|| invalid("bad parent span id"))?;
        if span_id == [0; 8] {
            return Err(invalid("all-zero parent span id"));
        }
        let flags = decode_hex::<1>(flags).ok_or_else(|| invalid("bad flags"))?[0];

        Ok(Self {
            trace_id,
            span_id,
            flags,
        })
    }
}

impl fmt::Display for TraceParent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "00-{}-{}-{:02x}",
            encode_hex(&self.trace_id),
            encode_hex(&self.span_id),
            self.flags
        )
    }
}

pub(crate) fn new_trace_id() -> [u8; 16] {
    loop {
        let id: [u8; 16] = rand::rng().random();
        if id != [0; 16] {
            return id;
        }
    }
}

pub(crate) fn new_span_id() -> [u8; 8] {
    loop {
        let id: [u8; 8] = rand::rng().random();
        if id != [0; 8] {
            return id;
        }
    }
}

pub(crate) fn encode_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}

/// Lowercase-only, as the spec requires.
fn decode_hex<const N: usize>(text: &str) -> Option<[u8; N]> {
    if text.len() != N * 2 {
        return None;
    }
    let nibble = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    };
    let bytes = text.as_bytes();
    let mut out = [0u8; N];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = (nibble(bytes[2 * i])? << 4) | nibble(bytes[2 * i + 1])?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

    #[test]
    fn parse_round_trips() {
        let parsed = TraceParent::parse(VALID).unwrap();
        assert_eq!(
            encode_hex(&parsed.trace_id),
            "4bf92f3577b34da6a3ce929d0e0e4736"
        );
        assert_eq!(encode_hex(&parsed.span_id), "00f067aa0ba902b7");
        assert_eq!(parsed.flags, 0x01);
        assert_eq!(parsed.to_string(), VALID);
    }

    #[test]
    fn parse_accepts_unsampled_and_surrounding_whitespace() {
        let parsed =
            TraceParent::parse(" 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00 ")
                .unwrap();
        assert_eq!(parsed.flags, 0x00);
    }

    #[test]
    fn parse_accepts_future_version_with_extra_fields() {
        let parsed =
            TraceParent::parse("01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra")
                .unwrap();
        assert_eq!(parsed.to_string(), VALID);
    }

    #[test]
    fn parse_rejects_malformed_values() {
        for bad in [
            "",
            "garbage",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",
            "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
            "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-1",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902zz-01",
        ] {
            let err = TraceParent::parse(bad).expect_err(bad);
            assert!(
                matches!(err, ClientError::InvalidParameter(ref msg) if msg.contains("traceparent")),
                "{bad}: {err:?}"
            );
        }
    }

    #[test]
    fn new_ids_are_nonzero_distinct_and_round_trip() {
        let a = TraceParent {
            trace_id: new_trace_id(),
            span_id: new_span_id(),
            flags: 0x01,
        };
        assert_ne!(a.trace_id, [0; 16]);
        assert_ne!(a.span_id, [0; 8]);
        assert_ne!(a.trace_id, new_trace_id());
        assert_eq!(TraceParent::parse(&a.to_string()).unwrap(), a);
    }
}

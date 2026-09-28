//! Opt-in client spans, exported over OTLP/HTTP (JSON, gzip) from a dedicated background thread.
//!
//! Enabled only by `BASETEN_PERFORMANCE_CLIENT_OTLP_ENDPOINT`, never by the standard `OTEL_*`
//! variables: an application's own SDK commonly points those at its own backend, and its
//! `OTEL_EXPORTER_OTLP_HEADERS` credentials must not be sent to this endpoint. That is also why
//! this is not built on `opentelemetry-otlp`, whose exporter always merges those env headers in.
//!
//! Recording never blocks a request: spans go through a bounded channel and are dropped (and
//! counted) when it is full. Export is best effort; spans still queued at process exit are lost.

use crate::errors::ClientError;
use crate::trace_context::{
    encode_hex, new_span_id, new_trace_id, TraceContext, TraceParent, SAMPLED_FLAG,
};
use flate2::write::GzEncoder;
use flate2::Compression;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_ENCODING, CONTENT_TYPE};
use serde_json::{json, Value};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

pub const OTLP_ENDPOINT_ENV_VAR: &str = "BASETEN_PERFORMANCE_CLIENT_OTLP_ENDPOINT";
pub const OTLP_HEADERS_ENV_VAR: &str = "BASETEN_PERFORMANCE_CLIENT_OTLP_HEADERS";

const TRACES_PATH: &str = "/v1/traces";
const DEFAULT_SERVICE_NAME: &str = "baseten-performance-client";
const SCOPE_NAME: &str = "baseten_performance_client";
const QUEUE_CAPACITY: usize = 65_536;
const MAX_EXPORT_BATCH: usize = 2048;
const FLUSH_INTERVAL: Duration = Duration::from_secs(1);
const EXPORT_TIMEOUT: Duration = Duration::from_secs(5);
const WARN_INTERVAL: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AttrValue {
    Str(String),
    Int(i64),
    Bool(bool),
}

impl From<&str> for AttrValue {
    fn from(value: &str) -> Self {
        AttrValue::Str(value.to_string())
    }
}

impl From<String> for AttrValue {
    fn from(value: String) -> Self {
        AttrValue::Str(value)
    }
}

impl From<i64> for AttrValue {
    fn from(value: i64) -> Self {
        AttrValue::Int(value)
    }
}

impl From<bool> for AttrValue {
    fn from(value: bool) -> Self {
        AttrValue::Bool(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpanKind {
    Internal = 1,
    Client = 3,
}

#[derive(Debug)]
struct SpanData {
    trace_id: [u8; 16],
    span_id: [u8; 8],
    parent_span_id: Option<[u8; 8]>,
    /// W3C trace flags, inherited from the parent so downstream sampling state is preserved.
    flags: u8,
    name: String,
    kind: SpanKind,
    start_unix_nano: u64,
    end_unix_nano: u64,
    attributes: Vec<(&'static str, AttrValue)>,
    events: Vec<(u64, &'static str)>,
    error: Option<String>,
}

/// What every HTTP attempt of one client call propagates.
#[derive(Debug, Clone)]
pub(crate) enum CallTraceContext {
    /// Forward the caller's context unchanged: export is off, or the caller didn't sample.
    Propagate(TraceContext),
    /// Each attempt is a CLIENT span under the call span.
    Record {
        trace_id: [u8; 16],
        call_span_id: [u8; 8],
        flags: u8,
        tracestate: Option<Arc<str>>,
    },
}

/// A span being recorded. It is exported once finished; a span dropped unfinished (its request
/// was aborted) is exported marked cancelled rather than leaked.
#[derive(Debug)]
pub(crate) struct RecordingSpan {
    data: Option<SpanData>,
    /// Shared by the two attempts of a hedge race; set once the hedge is actually sent, so an
    /// attempt dropped mid-race is recorded as the race's loser.
    hedge_race: Option<Arc<AtomicBool>>,
}

impl RecordingSpan {
    fn start(
        trace_id: [u8; 16],
        parent_span_id: Option<[u8; 8]>,
        flags: u8,
        name: String,
        kind: SpanKind,
    ) -> Self {
        Self {
            data: Some(SpanData {
                trace_id,
                span_id: new_span_id(),
                parent_span_id,
                flags,
                name,
                kind,
                start_unix_nano: unix_nanos_now(),
                end_unix_nano: 0,
                attributes: Vec::new(),
                events: Vec::new(),
                error: None,
            }),
            hedge_race: None,
        }
    }

    fn data_mut(&mut self) -> &mut SpanData {
        self.data
            .as_mut()
            .expect("span data is present until the span is finished")
    }

    pub(crate) fn traceparent(&self) -> TraceParent {
        let data = self
            .data
            .as_ref()
            .expect("span data is present until the span is finished");
        TraceParent {
            trace_id: data.trace_id,
            span_id: data.span_id,
            flags: data.flags,
        }
    }

    pub(crate) fn set_attribute(&mut self, key: &'static str, value: impl Into<AttrValue>) {
        self.data_mut().attributes.push((key, value.into()));
    }

    pub(crate) fn add_event(&mut self, name: &'static str) {
        self.data_mut().events.push((unix_nanos_now(), name));
    }

    /// Marks the span failed: semconv `error.type` plus an error status.
    pub(crate) fn set_error(&mut self, error_type: String) {
        let data = self.data_mut();
        data.attributes
            .push(("error.type", AttrValue::Str(error_type.clone())));
        data.error = Some(error_type);
    }

    pub(crate) fn set_hedge_race(&mut self, hedge_race: Arc<AtomicBool>) {
        self.hedge_race = Some(hedge_race);
    }

    pub(crate) fn leave_hedge_race(&mut self) {
        self.hedge_race = None;
    }

    pub(crate) fn finish(mut self) {
        if let Some(data) = self.take_ended() {
            export(data);
        }
    }

    fn take_ended(&mut self) -> Option<SpanData> {
        let mut data = self.data.take()?;
        data.end_unix_nano = unix_nanos_now();
        Some(data)
    }
}

impl Drop for RecordingSpan {
    fn drop(&mut self) {
        let Some(mut data) = self.take_ended() else {
            return;
        };
        let lost_hedge_race = self
            .hedge_race
            .as_ref()
            .is_some_and(|race| race.load(Ordering::SeqCst));
        let key = if lost_hedge_race {
            "b10.perfclient.hedge_cancelled"
        } else {
            "b10.perfclient.cancelled"
        };
        data.attributes.push((key, AttrValue::Bool(true)));
        export(data);
    }
}

/// Starts the span for one client call (embed, rerank, ...) when export is on, and returns what
/// the call's HTTP attempts should propagate. A caller that didn't sample its span keeps that
/// decision: its context is forwarded and nothing is recorded, as a parent-based sampler would.
pub(crate) fn start_call_span(
    parent: Option<&TraceContext>,
    operation: &str,
) -> (Option<CallTraceContext>, Option<RecordingSpan>) {
    if exporter().is_none() || parent.is_some_and(|parent| !parent.parent.sampled()) {
        return (parent.cloned().map(CallTraceContext::Propagate), None);
    }
    let (trace_id, parent_span_id, flags, tracestate) = match parent {
        Some(parent) => (
            parent.parent.trace_id,
            Some(parent.parent.span_id),
            parent.parent.flags,
            parent.tracestate.clone(),
        ),
        None => (new_trace_id(), None, SAMPLED_FLAG, None),
    };
    let span = RecordingSpan::start(
        trace_id,
        parent_span_id,
        flags,
        format!("perfclient.{}", operation),
        SpanKind::Internal,
    );
    let context = CallTraceContext::Record {
        trace_id,
        call_span_id: span.traceparent().span_id,
        flags,
        tracestate,
    };
    (Some(context), Some(span))
}

pub(crate) fn finish_span<T>(span: Option<RecordingSpan>, result: &Result<T, ClientError>) {
    match result {
        Ok(_) => {
            if let Some(span) = span {
                span.finish();
            }
        }
        Err(err) => fail_span(span, error_type(err)),
    }
}

pub(crate) fn fail_span(span: Option<RecordingSpan>, error_type: String) {
    if let Some(mut span) = span {
        span.set_error(error_type);
        span.finish();
    }
}

pub(crate) fn record_attempt_attributes(
    span: &mut RecordingSpan,
    method: &str,
    url: &str,
    customer_request_id: &str,
    resend_count: u32,
    hedge: bool,
) {
    span.set_attribute("http.request.method", method);
    if let Ok(mut parsed) = reqwest::Url::parse(url) {
        if let Some(host) = parsed.host_str() {
            span.set_attribute("server.address", host);
        }
        if let Some(port) = parsed.port_or_known_default() {
            span.set_attribute("server.port", i64::from(port));
        }
        // Userinfo and query strings can carry credentials.
        let _ = parsed.set_username("");
        let _ = parsed.set_password(None);
        parsed.set_query(None);
        parsed.set_fragment(None);
        span.set_attribute("url.full", parsed.to_string());
    }
    span.set_attribute("b10.customer_request_id", customer_request_id);
    if resend_count > 0 {
        span.set_attribute("http.request.resend_count", i64::from(resend_count));
    }
    if hedge {
        span.set_attribute("b10.perfclient.hedge", true);
    }
}

/// Starts a CLIENT span for one HTTP attempt under the call span, when recording.
pub(crate) fn start_attempt_span(
    context: Option<&CallTraceContext>,
    method: &str,
) -> Option<RecordingSpan> {
    match context {
        Some(CallTraceContext::Record {
            trace_id,
            call_span_id,
            flags,
            ..
        }) => Some(RecordingSpan::start(
            *trace_id,
            Some(*call_span_id),
            *flags,
            method.to_string(),
            SpanKind::Client,
        )),
        Some(CallTraceContext::Propagate(_)) | None => None,
    }
}

/// The `traceparent` and `tracestate` an attempt sends: its own span when recording, else the
/// caller's context unchanged.
pub(crate) fn attempt_trace_headers<'a>(
    context: Option<&'a CallTraceContext>,
    attempt_span: Option<&RecordingSpan>,
) -> Option<(TraceParent, Option<&'a str>)> {
    match context? {
        CallTraceContext::Propagate(parent) => Some((parent.parent, parent.tracestate.as_deref())),
        CallTraceContext::Record { tracestate, .. } => {
            attempt_span.map(|span| (span.traceparent(), tracestate.as_deref()))
        }
    }
}

pub(crate) fn error_type(err: &ClientError) -> String {
    match err {
        ClientError::LocalTimeout(_, _) => "local_timeout".to_string(),
        ClientError::RemoteTimeout(_, _) => "remote_timeout".to_string(),
        ClientError::Network(_) => "network".to_string(),
        ClientError::Connect(_) => "connect".to_string(),
        ClientError::Http { status, .. } => status.to_string(),
        ClientError::InvalidParameter(_) => "invalid_parameter".to_string(),
        ClientError::Serialization(_) => "serialization".to_string(),
        ClientError::Cancellation(_) => "cancelled".to_string(),
    }
}

fn unix_nanos_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

struct SpanExporter {
    sender: mpsc::Sender<SpanData>,
    dropped: AtomicU64,
    last_drop_warning: Mutex<Option<Instant>>,
}

static EXPORTER: OnceLock<Option<SpanExporter>> = OnceLock::new();

fn exporter() -> Option<&'static SpanExporter> {
    EXPORTER.get_or_init(SpanExporter::from_env).as_ref()
}

fn export(data: SpanData) {
    let Some(exporter) = exporter() else {
        return;
    };
    if exporter.sender.try_send(data).is_err() {
        let dropped = exporter.dropped.fetch_add(1, Ordering::Relaxed) + 1;
        if should_warn(&exporter.last_drop_warning) {
            tracing::warn!(
                "client span queue full or exporter stopped; {} spans dropped so far",
                dropped
            );
        }
    }
}

fn should_warn(last_warning: &Mutex<Option<Instant>>) -> bool {
    let Ok(mut last) = last_warning.lock() else {
        return false;
    };
    if last.is_some_and(|at| at.elapsed() < WARN_INTERVAL) {
        return false;
    }
    *last = Some(Instant::now());
    true
}

impl SpanExporter {
    fn from_env() -> Option<Self> {
        let endpoint = std::env::var(OTLP_ENDPOINT_ENV_VAR).ok()?;
        let endpoint = endpoint.trim();
        if endpoint.is_empty() {
            return None;
        }
        let traces_url = traces_url(endpoint);
        let headers = parse_headers(&std::env::var(OTLP_HEADERS_ENV_VAR).unwrap_or_default());
        let resource = resource_attributes();

        let (sender, receiver) = mpsc::channel(QUEUE_CAPACITY);
        // Own thread and runtime: the exporter must outlive whichever runtime made the first
        // request (the Python and Node bindings, and tests, each bring their own).
        let spawned = std::thread::Builder::new()
            .name("perfclient-otlp".to_string())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(err) => {
                        tracing::warn!("client span export disabled: runtime failed: {}", err);
                        return;
                    }
                };
                runtime.block_on(run_exporter(receiver, traces_url, headers, resource));
            });
        if let Err(err) = spawned {
            tracing::warn!("client span export disabled: thread failed: {}", err);
            return None;
        }

        Some(Self {
            sender,
            dropped: AtomicU64::new(0),
            last_drop_warning: Mutex::new(None),
        })
    }
}

fn traces_url(endpoint: &str) -> String {
    let base = endpoint.trim_end_matches('/');
    if base.ends_with(TRACES_PATH) {
        base.to_string()
    } else {
        format!("{}{}", base, TRACES_PATH)
    }
}

/// `OTEL_EXPORTER_OTLP_HEADERS` format: comma-separated `key=value`, values percent-encoded.
fn parse_headers(raw: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for pair in raw
        .split(',')
        .map(str::trim)
        .filter(|pair| !pair.is_empty())
    {
        let parsed = pair.split_once('=').and_then(|(key, value)| {
            let name = HeaderName::from_bytes(key.trim().as_bytes()).ok()?;
            let value = HeaderValue::from_str(&percent_decode(value.trim())?).ok()?;
            Some((name, value))
        });
        match parsed {
            Some((name, value)) => {
                headers.insert(name, value);
            }
            // Only the key is logged: values are usually credentials.
            None => tracing::warn!(
                "ignoring malformed {} entry with key {:?}",
                OTLP_HEADERS_ENV_VAR,
                pair.split('=').next().unwrap_or_default()
            ),
        }
    }
    headers
}

fn percent_decode(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = std::str::from_utf8(bytes.get(i + 1..i + 3)?).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

fn resource_attributes() -> Value {
    let service_name = std::env::var("OTEL_SERVICE_NAME")
        .ok()
        .filter(|name| !name.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_SERVICE_NAME.to_string());
    Value::Array(vec![
        attribute_json("service.name", &AttrValue::Str(service_name)),
        attribute_json(
            "service.version",
            &AttrValue::Str(env!("CARGO_PKG_VERSION").to_string()),
        ),
        attribute_json(
            "telemetry.sdk.name",
            &AttrValue::Str(SCOPE_NAME.to_string()),
        ),
        attribute_json(
            "telemetry.sdk.language",
            &AttrValue::Str("rust".to_string()),
        ),
        attribute_json(
            "telemetry.sdk.version",
            &AttrValue::Str(env!("CARGO_PKG_VERSION").to_string()),
        ),
    ])
}

async fn run_exporter(
    mut receiver: mpsc::Receiver<SpanData>,
    traces_url: String,
    headers: HeaderMap,
    resource: Value,
) {
    let client = match reqwest::Client::builder().timeout(EXPORT_TIMEOUT).build() {
        Ok(client) => client,
        Err(err) => {
            tracing::warn!("client span export disabled: HTTP client failed: {}", err);
            return;
        }
    };
    let last_failure_warning = Mutex::new(None);
    let mut batch = Vec::with_capacity(MAX_EXPORT_BATCH);
    let mut ticker = tokio::time::interval(FLUSH_INTERVAL);
    loop {
        tokio::select! {
            received = receiver.recv() => match received {
                Some(span) => {
                    batch.push(span);
                    if batch.len() >= MAX_EXPORT_BATCH {
                        send_batch(&client, &traces_url, &headers, &resource, &mut batch, &last_failure_warning).await;
                    }
                }
                None => {
                    send_batch(&client, &traces_url, &headers, &resource, &mut batch, &last_failure_warning).await;
                    return;
                }
            },
            _ = ticker.tick() => {
                if !batch.is_empty() {
                    send_batch(&client, &traces_url, &headers, &resource, &mut batch, &last_failure_warning).await;
                }
            }
        }
    }
}

async fn send_batch(
    client: &reqwest::Client,
    traces_url: &str,
    headers: &HeaderMap,
    resource: &Value,
    batch: &mut Vec<SpanData>,
    last_failure_warning: &Mutex<Option<Instant>>,
) {
    if batch.is_empty() {
        return;
    }
    let span_count = batch.len();
    let body = encode_request(resource, batch);
    batch.clear();

    let outcome = match gzip(&body) {
        Ok(compressed) => client
            .post(traces_url)
            .headers(headers.clone())
            .header(CONTENT_TYPE, "application/json")
            .header(CONTENT_ENCODING, "gzip")
            .body(compressed)
            .send()
            .await
            .map_err(|err| err.to_string())
            .and_then(|response| {
                if response.status().is_success() {
                    Ok(())
                } else {
                    Err(format!("HTTP {}", response.status()))
                }
            }),
        Err(err) => Err(format!("gzip failed: {}", err)),
    };
    if let Err(reason) = outcome {
        if should_warn(last_failure_warning) {
            tracing::warn!(
                "dropping {} client spans: export to {} failed: {}",
                span_count,
                traces_url,
                reason
            );
        }
    }
}

fn gzip(body: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut encoder = GzEncoder::new(Vec::with_capacity(body.len() / 4), Compression::fast());
    encoder.write_all(body)?;
    encoder.finish()
}

/// An OTLP `ExportTraceServiceRequest` in the protobuf-JSON mapping (hex ids, 64-bit integers as
/// strings).
fn encode_request(resource: &Value, spans: &[SpanData]) -> Vec<u8> {
    let spans: Vec<Value> = spans.iter().map(span_json).collect();
    let request = json!({
        "resourceSpans": [{
            "resource": {"attributes": resource},
            "scopeSpans": [{
                "scope": {"name": SCOPE_NAME, "version": env!("CARGO_PKG_VERSION")},
                "spans": spans,
            }],
        }],
    });
    serde_json::to_vec(&request).expect("serde_json::Value always serializes")
}

fn span_json(span: &SpanData) -> Value {
    let mut value = json!({
        "traceId": encode_hex(&span.trace_id),
        "spanId": encode_hex(&span.span_id),
        "name": span.name,
        "kind": span.kind as i32,
        "startTimeUnixNano": span.start_unix_nano.to_string(),
        "endTimeUnixNano": span.end_unix_nano.to_string(),
        "attributes": span
            .attributes
            .iter()
            .map(|(key, value)| attribute_json(key, value))
            .collect::<Vec<_>>(),
        "events": span
            .events
            .iter()
            .map(|(time, name)| json!({"timeUnixNano": time.to_string(), "name": name}))
            .collect::<Vec<_>>(),
    });
    if let Some(parent) = span.parent_span_id {
        value["parentSpanId"] = Value::String(encode_hex(&parent));
    }
    if let Some(ref message) = span.error {
        // STATUS_CODE_ERROR
        value["status"] = json!({"code": 2, "message": message});
    }
    value
}

fn attribute_json(key: &str, value: &AttrValue) -> Value {
    let value = match value {
        AttrValue::Str(text) => json!({"stringValue": text}),
        AttrValue::Int(number) => json!({"intValue": number.to_string()}),
        AttrValue::Bool(flag) => json!({"boolValue": flag}),
    };
    json!({"key": key, "value": value})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traces_url_appends_signal_path_once() {
        assert_eq!(
            traces_url("https://otlp.example.com"),
            "https://otlp.example.com/v1/traces"
        );
        assert_eq!(
            traces_url("https://otlp.example.com/"),
            "https://otlp.example.com/v1/traces"
        );
        assert_eq!(
            traces_url("https://otlp.example.com/v1/traces"),
            "https://otlp.example.com/v1/traces"
        );
    }

    #[test]
    fn parse_headers_percent_decodes_and_skips_malformed_entries() {
        let headers = parse_headers(
            "authorization=Basic%20eHlsYTpzM2NyZXQ=, x-team = a%2Cb ,novalue,bad name=x",
        );
        assert_eq!(headers.len(), 2);
        assert_eq!(headers["authorization"], "Basic eHlsYTpzM2NyZXQ=");
        assert_eq!(headers["x-team"], "a,b");
    }

    #[test]
    fn percent_decode_rejects_truncated_escapes() {
        assert_eq!(percent_decode("a%20b").as_deref(), Some("a b"));
        assert_eq!(percent_decode("a%2"), None);
        assert_eq!(percent_decode("a%zz"), None);
    }

    #[test]
    fn span_json_uses_otlp_json_mapping() {
        let span = SpanData {
            trace_id: [0x4b; 16],
            span_id: [0x01; 8],
            parent_span_id: Some([0x02; 8]),
            flags: SAMPLED_FLAG,
            name: "POST".to_string(),
            kind: SpanKind::Client,
            start_unix_nano: 10,
            end_unix_nano: 20,
            attributes: vec![
                ("http.response.status_code", AttrValue::Int(503)),
                ("b10.perfclient.hedge", AttrValue::Bool(true)),
            ],
            events: vec![(15, "http.response.headers")],
            error: Some("503".to_string()),
        };
        let value = span_json(&span);
        assert_eq!(value["traceId"], "4b".repeat(16));
        assert_eq!(value["spanId"], "01".repeat(8));
        assert_eq!(value["parentSpanId"], "02".repeat(8));
        assert_eq!(value["kind"], 3);
        assert_eq!(value["startTimeUnixNano"], "10");
        assert_eq!(value["attributes"][0]["value"]["intValue"], "503");
        assert_eq!(value["attributes"][1]["value"]["boolValue"], true);
        assert_eq!(value["events"][0]["name"], "http.response.headers");
        assert_eq!(value["status"]["code"], 2);
    }

    #[test]
    fn attempt_trace_headers_forward_the_parent_only_when_not_recording() {
        let parent = TraceContext::parse(
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00",
            Some("vendor=opaque"),
        )
        .unwrap();
        let propagate = CallTraceContext::Propagate(parent.clone());
        assert_eq!(
            attempt_trace_headers(Some(&propagate), None),
            Some((parent.parent, Some("vendor=opaque")))
        );
        assert_eq!(attempt_trace_headers(None, None), None);

        let record = CallTraceContext::Record {
            trace_id: parent.parent.trace_id,
            call_span_id: [7; 8],
            flags: 0x03,
            tracestate: parent.tracestate.clone(),
        };
        let span = start_attempt_span(Some(&record), "POST").expect("recording context");
        let (sent, tracestate) = attempt_trace_headers(Some(&record), Some(&span)).unwrap();
        assert_eq!(sent.trace_id, parent.parent.trace_id);
        assert_eq!(sent.flags, 0x03, "recorded spans keep the parent's flags");
        assert_ne!(sent.span_id, [7; 8]);
        assert_eq!(tracestate, Some("vendor=opaque"));
        assert!(start_attempt_span(Some(&propagate), "POST").is_none());
        // Dropping an unfinished span with export disabled must not panic.
        drop(span);
    }
}

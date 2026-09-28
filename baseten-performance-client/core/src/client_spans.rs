//! Best-effort OTLP/HTTP JSON export, isolated from the application's OTEL_* settings.
use crate::{split_policy::RequestProcessingConfig, ClientError};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{json, Value};
use std::sync::{mpsc, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const ENDPOINT: &str = "BASETEN_PERFORMANCE_CLIENT_OTLP_ENDPOINT";
const HEADERS: &str = "BASETEN_PERFORMANCE_CLIENT_OTLP_HEADERS";
type Exporter = mpsc::SyncSender<Value>;
static EXPORTER: OnceLock<Result<Option<Exporter>, String>> = OnceLock::new();

fn init_exporter() -> Result<Option<Exporter>, String> {
    let endpoint = std::env::var(ENDPOINT).unwrap_or_default();
    if endpoint.trim().is_empty() {
        return Ok(None);
    }
    let mut url =
        reqwest::Url::parse(endpoint.trim()).map_err(|_| format!("invalid {ENDPOINT}"))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(format!("{ENDPOINT} must be an HTTP(S) URL"));
    }
    let path = url.path().trim_end_matches('/').to_owned();
    url.set_path(&path);
    if !url.path().trim_end_matches('/').ends_with("/v1/traces") {
        url.set_path(&format!("{}/v1/traces", url.path().trim_end_matches('/')));
    }
    let mut headers = HeaderMap::new();
    for entry in std::env::var(HEADERS)
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.trim().is_empty())
    {
        let (name, value) = entry
            .split_once('=')
            .ok_or_else(|| format!("invalid {HEADERS}"))?;
        headers.insert(
            HeaderName::from_bytes(name.trim().as_bytes())
                .map_err(|_| format!("invalid {HEADERS} name"))?,
            HeaderValue::from_str(value.trim()).map_err(|_| format!("invalid {HEADERS} value"))?,
        );
    }
    let (sender, receiver) = mpsc::sync_channel(1024);
    std::thread::Builder::new().name("perfclient-otlp".into()).spawn(move || {
        // Blocking reqwest owns a runtime; construct and drop it outside the caller's runtime.
        let client = match reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(5)).redirect(reqwest::redirect::Policy::none()).build() {
            Ok(client) => client,
            Err(_) => { tracing::warn!("Could not initialize performance-client OTLP transport"); return; }
        };
        let mut last_warning = None;
        while let Ok(first) = receiver.recv() {
            let mut spans = vec![first];
            spans.extend(receiver.try_iter().take(255));
            let body = json!({"resourceSpans": [{
                "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "baseten-performance-client"}}]},
                "scopeSpans": [{"scope": {"name": "baseten_performance_client"}, "spans": spans}]
            }]});
            match client.post(url.clone()).headers(headers.clone()).json(&body).send() {
                Ok(response) if response.status().is_success() => {},
                _ => {
                    if last_warning.is_none_or(|last: Instant| last.elapsed() >= Duration::from_secs(60)) {
                        tracing::warn!("Performance-client OTLP export failed; batch dropped");
                        last_warning = Some(Instant::now());
                    }
                },
            }
        }
    }).map_err(|_| "Could not start performance-client OTLP worker".to_string())?;
    Ok(Some(sender))
}

pub(crate) struct CallSpan {
    sender: &'static Exporter,
    data: Value,
    started: Instant,
    start_ns: u128,
    completed: bool,
}

impl CallSpan {
    pub(crate) fn start(
        config: &mut RequestProcessingConfig,
        name: &'static str,
    ) -> Result<Option<Self>, ClientError> {
        let Some(sender) = EXPORTER
            .get_or_init(init_exporter)
            .as_ref()
            .map_err(|error| ClientError::InvalidParameter(error.clone()))?
        else {
            return Ok(None);
        };
        let headers = config.extra_headers.get_or_insert_default();
        let value = |name: &str| -> Result<Option<String>, ClientError> {
            let mut values = headers
                .iter()
                .filter(|(key, _)| key.eq_ignore_ascii_case(name));
            let result = values.next().map(|(_, value)| value.clone());
            if values.next().is_some() {
                return Err(ClientError::InvalidParameter(format!(
                    "duplicate {name} headers"
                )));
            }
            Ok(result)
        };
        let parent = value("traceparent")?;
        let state = value("tracestate")?;
        let (trace_id, parent_id, flags) = match parent.as_deref() {
            Some(parent) => parse_parent(parent)?,
            None => (uuid::Uuid::new_v4().simple().to_string(), String::new(), 1),
        };
        if flags & 1 == 0 {
            return Ok(None);
        }
        let span_id = uuid::Uuid::new_v4().simple().to_string()[..16].to_string();
        headers.retain(|key, _| !key.eq_ignore_ascii_case("traceparent"));
        headers.insert(
            "traceparent".into(),
            format!("00-{trace_id}-{span_id}-{flags:02x}"),
        );
        let start_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        Ok(Some(Self {
            sender,
            started: Instant::now(),
            start_ns,
            completed: false,
            data: json!({"traceId": trace_id, "spanId": span_id, "parentSpanId": parent_id,
                "traceState": state.unwrap_or_default(), "flags": flags, "name": name,
                "kind": 1, "startTimeUnixNano": start_ns.to_string()}),
        }))
    }

    pub(crate) fn complete(span: &mut Option<Self>) {
        if let Some(span) = span {
            span.completed = true;
        }
    }
}

fn parse_parent(parent: &str) -> Result<(String, String, u8), ClientError> {
    let parts: Vec<_> = parent.split('-').collect();
    let hex = |s: &str, len| {
        s.len() == len
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    };
    if parts.len() != 4
        || !hex(parts[0], 2)
        || parts[0] == "ff"
        || !hex(parts[1], 32)
        || !hex(parts[2], 16)
        || !hex(parts[3], 2)
        || parts[1].bytes().all(|b| b == b'0')
        || parts[2].bytes().all(|b| b == b'0')
    {
        return Err(ClientError::InvalidParameter(
            "export requires a valid 55-character W3C traceparent".into(),
        ));
    }
    Ok((
        parts[1].into(),
        parts[2].into(),
        u8::from_str_radix(parts[3], 16).unwrap(),
    ))
}

impl Drop for CallSpan {
    fn drop(&mut self) {
        self.data["endTimeUnixNano"] = (self.start_ns + self.started.elapsed().as_nanos())
            .to_string()
            .into();
        if !self.completed {
            self.data["status"] = json!({"code": 2, "message": "call failed or was cancelled"});
        }
        // Never block inference on telemetry; pending spans may be lost on process exit.
        if self
            .sender
            .try_send(std::mem::take(&mut self.data))
            .is_err()
        {
            tracing::debug!("Performance-client OTLP queue unavailable; span dropped");
        }
    }
}

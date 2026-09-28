//! Private OpenTelemetry provider; never changes the application's global tracing setup.
use crate::{split_policy::RequestProcessingConfig, ClientError};
use opentelemetry::{
    propagation::TextMapPropagator,
    trace::{Span as _, SpanContext, SpanKind, Status, TraceContextExt, Tracer, TracerProvider},
    Context, TraceFlags,
};
use opentelemetry_http::{HttpClient, HttpError};
use opentelemetry_otlp::{Compression, Protocol, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::{
    propagation::TraceContextPropagator,
    trace::{BatchConfigBuilder, BatchSpanProcessor, Sampler, SdkTracer, SdkTracerProvider, Span},
    Resource,
};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_ENCODING, CONTENT_TYPE};
use std::{collections::HashMap, sync::OnceLock, time::Duration};

const ENDPOINT: &str = "BASETEN_PERFORMANCE_CLIENT_OTLP_ENDPOINT";
const HEADERS: &str = "BASETEN_PERFORMANCE_CLIENT_OTLP_HEADERS";
static TRACER: OnceLock<Result<Option<SdkTracer>, String>> = OnceLock::new();

#[derive(Debug)]
struct CollectorTransport {
    client: reqwest::blocking::Client,
    headers: HeaderMap,
}

#[async_trait::async_trait]
impl HttpClient for CollectorTransport {
    async fn send_bytes(
        &self,
        mut request: http::Request<bytes::Bytes>,
    ) -> Result<http::Response<bytes::Bytes>, HttpError> {
        // The OTLP builder merges OTEL_* headers even with explicit configuration.
        *request.headers_mut() = self.headers.clone();
        request
            .headers_mut()
            .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        request
            .headers_mut()
            .insert(CONTENT_ENCODING, HeaderValue::from_static("gzip"));
        self.client.send_bytes(request).await
    }
}

fn init_tracer() -> Result<Option<SdkTracer>, String> {
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
    if !path.ends_with("/v1/traces") {
        url.set_path(&format!("{path}/v1/traces"));
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
    // Blocking reqwest must be constructed outside the caller's Tokio runtime.
    std::thread::Builder::new()
        .name("perfclient-otel-init".into())
        .spawn(move || {
            let client = reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(5))
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .map_err(|e| e.to_string())?;
            let exporter = opentelemetry_otlp::SpanExporter::builder()
                .with_http()
                .with_http_client(CollectorTransport { client, headers })
                .with_endpoint(url.as_str())
                .with_protocol(Protocol::HttpJson)
                .with_timeout(Duration::from_secs(5))
                .with_compression(Compression::Gzip)
                .build()
                .map_err(|e| e.to_string())?;
            let processor = BatchSpanProcessor::builder(exporter)
                .with_batch_config(
                    BatchConfigBuilder::default()
                        .with_max_queue_size(1024)
                        .with_max_export_batch_size(256)
                        .with_scheduled_delay(Duration::from_secs(1))
                        .build(),
                )
                .build();
            let provider = SdkTracerProvider::builder()
                .with_span_processor(processor)
                .with_sampler(Sampler::ParentBased(Box::new(Sampler::AlwaysOn)))
                .with_resource(
                    Resource::builder_empty()
                        .with_service_name("baseten-performance-client")
                        .build(),
                )
                .build();
            Ok(Some(provider.tracer("baseten_performance_client")))
        })
        .map_err(|e| e.to_string())?
        .join()
        .map_err(|_| "Could not initialize performance-client OpenTelemetry".to_string())?
}

pub(crate) fn start_call_span(
    config: &mut RequestProcessingConfig,
    name: &'static str,
) -> Result<Option<Span>, ClientError> {
    let Some(tracer) = TRACER
        .get_or_init(init_tracer)
        .as_ref()
        .map_err(|error| ClientError::InvalidParameter(error.clone()))?
    else {
        return Ok(None);
    };
    let headers = config.extra_headers.get_or_insert_default();
    let mut carrier = HashMap::new();
    for (key, value) in headers.iter().filter(|(key, _)| {
        key.eq_ignore_ascii_case("traceparent") || key.eq_ignore_ascii_case("tracestate")
    }) {
        if carrier
            .insert(key.to_ascii_lowercase(), value.clone())
            .is_some()
        {
            return Err(ClientError::InvalidParameter(format!(
                "duplicate {key} headers"
            )));
        }
    }
    // SDK 0.31 rejects version-00 flags above 02; normalize only its extraction carrier.
    let flags = carrier
        .get_mut("traceparent")
        .and_then(|parent| {
            *parent = parent.trim().to_owned();
            let flags = u8::from_str_radix(parent.get(53..55)?, 16).ok()?;
            parent.replace_range(53..55, if flags & 1 == 1 { "01" } else { "00" });
            Some(flags)
        })
        .unwrap_or_default();
    let propagator = TraceContextPropagator::new();
    let mut parent = propagator.extract_with_context(&Context::new(), &carrier);
    if carrier.contains_key("traceparent") {
        let context = parent.span().span_context().clone();
        if !context.is_valid() {
            return Err(ClientError::InvalidParameter(
                "invalid W3C traceparent".into(),
            ));
        }
        if !context.is_sampled() {
            return Ok(None);
        }
        parent = Context::new().with_remote_span_context(SpanContext::new(
            context.trace_id(),
            context.span_id(),
            TraceFlags::new(flags & 3),
            true,
            context.trace_state().clone(),
        ));
    }
    let mut span = tracer
        .span_builder(name)
        .with_kind(SpanKind::Client)
        .start_with_context(tracer, &parent);
    headers.retain(|key, _| {
        !key.eq_ignore_ascii_case("traceparent") && !key.eq_ignore_ascii_case("tracestate")
    });
    let context = span.span_context();
    headers.insert(
        "traceparent".into(),
        format!(
            "00-{}-{}-{:02x}",
            context.trace_id(),
            context.span_id(),
            context.trace_flags(),
        ),
    );
    if let Some(state) = carrier
        .remove("tracestate")
        .filter(|state| !state.is_empty())
    {
        headers.insert("tracestate".into(), state);
    }
    // SDK Drop exports on early return or cancellation; successful calls overwrite this status.
    span.set_status(Status::error("call failed or was cancelled"));
    Ok(Some(span))
}

//! Client span export end to end: a mock model server records the `traceparent` each attempt
//! sends, and a mock OTLP receiver records the exported spans.
//!
//! The exporter reads its env vars once per process, so every test here calls `otlp()` first
//! (which sets them) and filters exported spans by a trace id no other test uses.

use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap as AxumHeaderMap, StatusCode},
    response::IntoResponse,
    routing::post,
    Json, Router,
};
use baseten_performance_client_core::*;
use flate2::read::GzDecoder;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::Read;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

const PARENT: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
const OTLP_AUTH: &str = "Basic dGVzdA==";

/// (authorization, content-encoding) of each export request.
type ExportHeaders = Arc<Mutex<Vec<(Option<String>, Option<String>)>>>;

struct MockOtlp {
    spans: Arc<Mutex<Vec<Value>>>,
    resources: Arc<Mutex<Vec<Value>>>,
    request_headers: ExportHeaders,
}

fn otlp() -> &'static MockOtlp {
    static MOCK: OnceLock<MockOtlp> = OnceLock::new();
    MOCK.get_or_init(|| {
        let spans = Arc::new(Mutex::new(Vec::new()));
        let resources = Arc::new(Mutex::new(Vec::new()));
        let request_headers = Arc::new(Mutex::new(Vec::new()));
        let state = (
            Arc::clone(&spans),
            Arc::clone(&resources),
            Arc::clone(&request_headers),
        );
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("otlp listener binds");
        listener.set_nonblocking(true).expect("nonblocking");
        let addr = listener.local_addr().expect("otlp addr");
        // Own runtime: #[tokio::test] runtimes end with their test, this must outlive all of them.
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Runtime::new().expect("otlp runtime");
            runtime.block_on(async move {
                let app = Router::new()
                    .route("/v1/traces", post(otlp_handler))
                    .with_state(state);
                let listener = tokio::net::TcpListener::from_std(listener).expect("tokio listener");
                axum::serve(listener, app).await.expect("otlp server");
            });
        });
        std::env::set_var(OTLP_ENDPOINT_ENV_VAR, format!("http://{}", addr));
        std::env::set_var(OTLP_HEADERS_ENV_VAR, "authorization=Basic%20dGVzdA==");
        MockOtlp {
            spans,
            resources,
            request_headers,
        }
    })
}

type OtlpState = (
    Arc<Mutex<Vec<Value>>>,
    Arc<Mutex<Vec<Value>>>,
    ExportHeaders,
);

async fn otlp_handler(
    State((spans, resources, request_headers)): State<OtlpState>,
    headers: AxumHeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let header = |name: &str| {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string)
    };
    request_headers
        .lock()
        .unwrap()
        .push((header("authorization"), header("content-encoding")));

    let mut json_bytes = Vec::new();
    GzDecoder::new(&body[..])
        .read_to_end(&mut json_bytes)
        .expect("body is gzip");
    let request: Value = serde_json::from_slice(&json_bytes).expect("body is JSON");
    for resource_spans in request["resourceSpans"].as_array().unwrap() {
        resources
            .lock()
            .unwrap()
            .push(resource_spans["resource"].clone());
        for scope_spans in resource_spans["scopeSpans"].as_array().unwrap() {
            for span in scope_spans["spans"].as_array().unwrap() {
                spans.lock().unwrap().push(span.clone());
            }
        }
    }
    StatusCode::OK
}

async fn wait_for_spans(trace_id: &str, expected: usize) -> Vec<Value> {
    for _ in 0..100 {
        let matching: Vec<Value> = otlp()
            .spans
            .lock()
            .unwrap()
            .iter()
            .filter(|span| span["traceId"] == trace_id)
            .cloned()
            .collect();
        if matching.len() >= expected {
            return matching;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!(
        "timed out waiting for {} spans of trace {}",
        expected, trace_id
    );
}

fn attr<'a>(span: &'a Value, key: &str) -> Option<&'a Value> {
    span["attributes"]
        .as_array()?
        .iter()
        .find(|attribute| attribute["key"] == key)
        .map(|attribute| &attribute["value"])
}

/// Model server: records each request's traceparent and tracestate header(s), fails the first
/// `failures` requests with 503, and delays the first request by `first_delay`.
struct ModelServer {
    base_url: String,
    traceparents: Arc<Mutex<Vec<Vec<String>>>>,
    tracestates: Arc<Mutex<Vec<Vec<String>>>>,
    handle: tokio::task::JoinHandle<()>,
}

impl Drop for ModelServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

#[derive(Clone)]
struct ModelState {
    traceparents: Arc<Mutex<Vec<Vec<String>>>>,
    tracestates: Arc<Mutex<Vec<Vec<String>>>>,
    requests: Arc<AtomicUsize>,
    failures: usize,
    first_delay: Duration,
}

async fn start_model_server(failures: usize, first_delay: Duration) -> ModelServer {
    let traceparents = Arc::new(Mutex::new(Vec::new()));
    let tracestates = Arc::new(Mutex::new(Vec::new()));
    let state = ModelState {
        traceparents: Arc::clone(&traceparents),
        tracestates: Arc::clone(&tracestates),
        requests: Arc::new(AtomicUsize::new(0)),
        failures,
        first_delay,
    };
    let app = Router::new()
        .route("/v1/embeddings", post(model_handler))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("model listener binds");
    let addr = listener.local_addr().expect("model addr");
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("model server");
    });
    ModelServer {
        base_url: format!("http://{}", addr),
        traceparents,
        tracestates,
        handle,
    }
}

async fn model_handler(
    State(state): State<ModelState>,
    headers: AxumHeaderMap,
    Json(request): Json<CoreOpenAIEmbeddingsRequest>,
) -> impl IntoResponse {
    let index = state.requests.fetch_add(1, Ordering::SeqCst);
    let values = |name: &str| -> Vec<String> {
        headers
            .get_all(name)
            .iter()
            .map(|value| value.to_str().unwrap().to_string())
            .collect()
    };
    state
        .traceparents
        .lock()
        .unwrap()
        .push(values("traceparent"));
    state.tracestates.lock().unwrap().push(values("tracestate"));
    if index == 0 && !state.first_delay.is_zero() {
        tokio::time::sleep(state.first_delay).await;
    }
    if index < state.failures {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "busy"})),
        )
            .into_response();
    }
    let data: Vec<Value> = request
        .input
        .iter()
        .enumerate()
        .map(|(index, _)| json!({"object": "embedding", "embedding": [0.1, 0.2], "index": index}))
        .collect();
    Json(json!({
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }))
    .into_response()
}

fn client_for(server: &ModelServer) -> PerformanceClientCore {
    PerformanceClientCore::new(
        server.base_url.clone(),
        Some("test-key".to_string()),
        1,
        None,
        None,
        None,
    )
    .expect("client builds")
}

fn single_request_preference() -> RequestProcessingPreference {
    RequestProcessingPreference::new()
        .with_max_concurrent_requests(1)
        .with_batch_size(1)
        .with_timeout_s(2.0)
}

async fn embed(
    client: &PerformanceClientCore,
    preference: &RequestProcessingPreference,
) -> Result<(), ClientError> {
    client
        .process_embeddings_requests(
            vec!["hello".to_string()],
            "test-model".to_string(),
            None,
            None,
            None,
            preference,
        )
        .await
        .map(|_| ())
}

fn sent(server: &ModelServer) -> Vec<TraceParent> {
    server
        .traceparents
        .lock()
        .unwrap()
        .iter()
        .map(|values| {
            assert_eq!(
                values.len(),
                1,
                "exactly one traceparent per attempt: {values:?}"
            );
            TraceParent::parse(&values[0]).expect("sent traceparent is valid")
        })
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{:02x}", byte)).collect()
}

#[tokio::test]
async fn retried_call_exports_call_span_and_one_client_span_per_attempt() {
    otlp();
    let parent = TraceParent::parse(PARENT).unwrap();
    let server = start_model_server(1, Duration::ZERO).await;
    let preference = single_request_preference()
        .with_max_retries(1)
        .with_retry_budget_pct(1.0)
        .with_traceparent(PARENT.to_string());

    embed(&client_for(&server), &preference)
        .await
        .expect("retry succeeds");

    let attempts = sent(&server);
    assert_eq!(attempts.len(), 2);
    for attempt in &attempts {
        assert_eq!(attempt.trace_id, parent.trace_id);
        assert_eq!(attempt.flags, 0x01);
        assert_ne!(attempt.span_id, parent.span_id);
    }
    assert_ne!(attempts[0].span_id, attempts[1].span_id);

    let spans = wait_for_spans(&hex(&parent.trace_id), 3).await;
    let call = spans
        .iter()
        .find(|span| span["name"] == "perfclient.embed")
        .expect("call span exported");
    assert_eq!(call["kind"], 1);
    assert_eq!(call["parentSpanId"], hex(&parent.span_id));
    assert_eq!(
        attr(call, "b10.perfclient.request_count").unwrap()["intValue"],
        "1"
    );
    assert!(
        call.get("status").is_none(),
        "successful call has no error status"
    );

    let attempt_span = |span_id: &[u8; 8]| {
        spans
            .iter()
            .find(|span| span["spanId"] == hex(span_id))
            .unwrap_or_else(|| panic!("attempt span {} exported", hex(span_id)))
    };
    let first = attempt_span(&attempts[0].span_id);
    let second = attempt_span(&attempts[1].span_id);
    for span in [first, second] {
        assert_eq!(span["name"], "POST");
        assert_eq!(span["kind"], 3);
        assert_eq!(span["parentSpanId"], call["spanId"]);
        assert_eq!(
            attr(span, "http.request.method").unwrap()["stringValue"],
            "POST"
        );
        assert_eq!(
            attr(span, "server.address").unwrap()["stringValue"],
            "127.0.0.1"
        );
        assert!(attr(span, "url.full").unwrap()["stringValue"]
            .as_str()
            .unwrap()
            .ends_with("/v1/embeddings"));
        assert!(attr(span, "b10.customer_request_id").is_some());
        assert_eq!(span["events"][0]["name"], "http.response.headers");
        let start: u64 = span["startTimeUnixNano"].as_str().unwrap().parse().unwrap();
        let end: u64 = span["endTimeUnixNano"].as_str().unwrap().parse().unwrap();
        assert!(start > 0 && end >= start);
    }
    assert_eq!(
        attr(first, "http.response.status_code").unwrap()["intValue"],
        "503"
    );
    assert_eq!(attr(first, "error.type").unwrap()["stringValue"], "503");
    assert_eq!(first["status"]["code"], 2);
    assert!(attr(first, "http.request.resend_count").is_none());
    assert_eq!(
        attr(second, "http.response.status_code").unwrap()["intValue"],
        "200"
    );
    assert_eq!(
        attr(second, "http.request.resend_count").unwrap()["intValue"],
        "1"
    );
    assert!(second.get("status").is_none());

    let exports = otlp().request_headers.lock().unwrap().clone();
    assert!(exports
        .iter()
        .all(|(auth, encoding)| auth.as_deref() == Some(OTLP_AUTH)
            && encoding.as_deref() == Some("gzip")));
    let resources = otlp().resources.lock().unwrap().clone();
    let service_name = resources[0]["attributes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|attribute| attribute["key"] == "service.name")
        .expect("service.name resource attribute");
    assert!(service_name["value"]["stringValue"].is_string());
}

#[tokio::test]
async fn hedge_gets_its_own_span_and_the_loser_is_marked() {
    otlp();
    let parent =
        TraceParent::parse("00-11111111111111111111111111111111-2222222222222222-01").unwrap();
    let server = start_model_server(0, Duration::from_millis(800)).await;
    let preference = single_request_preference()
        .with_max_retries(0)
        .with_hedge_delay(0.2)
        .with_hedge_budget_pct(1.0)
        .with_traceparent(parent.to_string());

    embed(&client_for(&server), &preference)
        .await
        .expect("hedged request succeeds");

    let attempts = sent(&server);
    assert_eq!(attempts.len(), 2);
    assert_ne!(attempts[0].span_id, attempts[1].span_id);

    let spans = wait_for_spans(&hex(&parent.trace_id), 3).await;
    let primary = spans
        .iter()
        .find(|span| span["spanId"] == hex(&attempts[0].span_id))
        .expect("primary attempt span");
    let hedge = spans
        .iter()
        .find(|span| span["spanId"] == hex(&attempts[1].span_id))
        .expect("hedge attempt span");
    assert_eq!(
        attr(hedge, "b10.perfclient.hedge").unwrap()["boolValue"],
        true
    );
    assert_eq!(
        attr(hedge, "http.response.status_code").unwrap()["intValue"],
        "200"
    );
    assert!(attr(hedge, "b10.perfclient.hedge_cancelled").is_none());
    assert!(attr(primary, "b10.perfclient.hedge").is_none());
    assert_eq!(
        attr(primary, "b10.perfclient.hedge_cancelled").unwrap()["boolValue"],
        true
    );
}

#[tokio::test]
async fn call_without_parent_starts_a_new_trace() {
    otlp();
    let server = start_model_server(0, Duration::ZERO).await;

    embed(&client_for(&server), &single_request_preference())
        .await
        .expect("request succeeds");

    let attempts = sent(&server);
    assert_eq!(attempts.len(), 1);
    let spans = wait_for_spans(&hex(&attempts[0].trace_id), 2).await;
    let call = spans
        .iter()
        .find(|span| span["name"] == "perfclient.embed")
        .expect("call span");
    assert!(call.get("parentSpanId").is_none(), "call span is a root");
    let attempt = spans
        .iter()
        .find(|span| span["spanId"] == hex(&attempts[0].span_id))
        .expect("attempt span");
    assert_eq!(attempt["parentSpanId"], call["spanId"]);
}

#[tokio::test]
async fn failed_call_marks_call_span_error() {
    otlp();
    let parent =
        TraceParent::parse("00-33333333333333333333333333333333-4444444444444444-01").unwrap();
    let server = start_model_server(usize::MAX, Duration::ZERO).await;
    let preference = single_request_preference()
        .with_max_retries(0)
        .with_traceparent(parent.to_string());

    let err = embed(&client_for(&server), &preference)
        .await
        .expect_err("503 without retries fails");
    assert!(
        matches!(err, ClientError::Http { status: 503, .. }),
        "{err:?}"
    );

    let spans = wait_for_spans(&hex(&parent.trace_id), 2).await;
    let call = spans
        .iter()
        .find(|span| span["name"] == "perfclient.embed")
        .expect("call span");
    assert_eq!(call["status"]["code"], 2);
    assert_eq!(attr(call, "error.type").unwrap()["stringValue"], "503");
}

#[tokio::test]
async fn traceparent_in_extra_headers_wins() {
    otlp();
    let server = start_model_server(0, Duration::ZERO).await;
    let caller_value = "00-55555555555555555555555555555555-6666666666666666-01";
    let preference = single_request_preference()
        .with_traceparent("00-99999999999999999999999999999999-aaaaaaaaaaaaaaaa-01".to_string())
        .with_extra_headers(HashMap::from([(
            "Traceparent".to_string(),
            caller_value.to_string(),
        )]));

    embed(&client_for(&server), &preference)
        .await
        .expect("request succeeds");

    let seen = server.traceparents.lock().unwrap().clone();
    assert_eq!(seen, vec![vec![caller_value.to_string()]]);
}

#[tokio::test]
async fn unsampled_parent_is_forwarded_and_not_recorded() {
    otlp();
    let parent = "00-0123456789abcdef0123456789abcdef-1111111111111111-00";
    let server = start_model_server(0, Duration::ZERO).await;
    let preference = single_request_preference().with_traceparent(parent.to_string());

    embed(&client_for(&server), &preference)
        .await
        .expect("request succeeds");

    assert_eq!(
        *server.traceparents.lock().unwrap(),
        vec![vec![parent.to_string()]]
    );
    // Longer than the exporter's 1s flush interval, so a recorded span would have arrived.
    tokio::time::sleep(Duration::from_millis(2500)).await;
    let recorded = otlp()
        .spans
        .lock()
        .unwrap()
        .iter()
        .filter(|span| span["traceId"] == "0123456789abcdef0123456789abcdef")
        .count();
    assert_eq!(
        recorded, 0,
        "an unsampled parent's call must not be recorded"
    );
}

#[tokio::test]
async fn recorded_attempt_keeps_parent_flags_and_tracestate() {
    otlp();
    let parent =
        TraceParent::parse("00-fedcba9876543210fedcba9876543210-2222222222222222-03").unwrap();
    let server = start_model_server(0, Duration::ZERO).await;
    let preference = single_request_preference()
        .with_traceparent(parent.to_string())
        .with_tracestate("vendor=opaque".to_string());

    embed(&client_for(&server), &preference)
        .await
        .expect("request succeeds");

    let attempts = sent(&server);
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].trace_id, parent.trace_id);
    assert_eq!(attempts[0].flags, 0x03, "the parent's trace flags are kept");
    assert_ne!(attempts[0].span_id, parent.span_id);
    assert_eq!(
        *server.tracestates.lock().unwrap(),
        vec![vec!["vendor=opaque".to_string()]]
    );

    let spans = wait_for_spans(&hex(&parent.trace_id), 2).await;
    assert!(
        spans
            .iter()
            .any(|span| span["spanId"] == hex(&attempts[0].span_id)),
        "the span on the wire is the exported attempt span: {spans:?}"
    );
}

//! Trace-context propagation with span export off (no OTLP endpoint configured in this process).

use axum::{
    extract::State,
    http::{HeaderMap as AxumHeaderMap, StatusCode},
    response::IntoResponse,
    routing::post,
    Json, Router,
};
use baseten_performance_client_core::*;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const PARENT: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00";

/// The trace headers of each request the model server received, in order.
#[derive(Clone, Default)]
struct Seen {
    traceparents: Arc<Mutex<Vec<Vec<String>>>>,
    tracestates: Arc<Mutex<Vec<Vec<String>>>>,
}

impl Seen {
    fn traceparents(&self) -> Vec<Vec<String>> {
        self.traceparents.lock().unwrap().clone()
    }

    fn tracestates(&self) -> Vec<Vec<String>> {
        self.tracestates.lock().unwrap().clone()
    }
}

#[derive(Clone)]
struct ModelState {
    seen: Seen,
    requests: Arc<AtomicUsize>,
    failures: usize,
}

fn header_values(headers: &AxumHeaderMap, name: &str) -> Vec<String> {
    headers
        .get_all(name)
        .iter()
        .map(|value| value.to_str().unwrap().to_string())
        .collect()
}

async fn model_handler(
    State(state): State<ModelState>,
    headers: AxumHeaderMap,
    Json(request): Json<CoreOpenAIEmbeddingsRequest>,
) -> impl IntoResponse {
    let index = state.requests.fetch_add(1, Ordering::SeqCst);
    state
        .seen
        .traceparents
        .lock()
        .unwrap()
        .push(header_values(&headers, "traceparent"));
    state
        .seen
        .tracestates
        .lock()
        .unwrap()
        .push(header_values(&headers, "tracestate"));
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
        .map(|(index, _)| json!({"object": "embedding", "embedding": [0.1], "index": index}))
        .collect();
    Json(json!({
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }))
    .into_response()
}

async fn embed_against_server(
    failures: usize,
    preference: RequestProcessingPreference,
) -> (Result<(), ClientError>, Seen) {
    assert!(
        std::env::var(OTLP_ENDPOINT_ENV_VAR).is_err(),
        "these tests cover export-off behavior"
    );
    let seen = Seen::default();
    let state = ModelState {
        seen: seen.clone(),
        requests: Arc::new(AtomicUsize::new(0)),
        failures,
    };
    let app = Router::new()
        .route("/v1/embeddings", post(model_handler))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = PerformanceClientCore::new(
        format!("http://{}", addr),
        Some("test-key".to_string()),
        1,
        None,
        None,
        None,
    )
    .unwrap();
    let preference = preference
        .with_max_concurrent_requests(1)
        .with_batch_size(1)
        .with_timeout_s(2.0)
        .with_retry_budget_pct(1.0);
    let result = client
        .process_embeddings_requests(
            vec!["hello".to_string()],
            "test-model".to_string(),
            None,
            None,
            None,
            &preference,
        )
        .await
        .map(|_| ());
    handle.abort();
    (result, seen)
}

#[tokio::test]
async fn parent_is_forwarded_unchanged_on_every_attempt() {
    let (result, seen) = embed_against_server(
        1,
        RequestProcessingPreference::new()
            .with_max_retries(1)
            .with_traceparent(PARENT.to_string()),
    )
    .await;
    result.expect("retry succeeds");
    assert_eq!(
        seen.traceparents(),
        vec![vec![PARENT.to_string()], vec![PARENT.to_string()]]
    );
    assert_eq!(seen.tracestates(), vec![Vec::<String>::new(); 2]);
}

#[tokio::test]
async fn tracestate_is_forwarded_unchanged_on_every_attempt() {
    let (result, seen) = embed_against_server(
        1,
        RequestProcessingPreference::new()
            .with_max_retries(1)
            .with_traceparent(PARENT.to_string())
            .with_tracestate("congo=t61rcWkgMzE,rojo=00f067aa0ba902b7".to_string()),
    )
    .await;
    result.expect("retry succeeds");
    assert_eq!(
        seen.tracestates(),
        vec![vec!["congo=t61rcWkgMzE,rojo=00f067aa0ba902b7".to_string()]; 2]
    );
}

#[tokio::test]
async fn tracestate_without_traceparent_is_rejected_before_any_request() {
    let (result, seen) = embed_against_server(
        0,
        RequestProcessingPreference::new().with_tracestate("vendor=opaque".to_string()),
    )
    .await;
    let err = result.expect_err("tracestate alone rejected");
    assert!(
        matches!(err, ClientError::InvalidParameter(ref msg) if msg.contains("tracestate")),
        "{err:?}"
    );
    assert!(seen.traceparents().is_empty());
}

#[tokio::test]
async fn no_parent_sends_no_traceparent() {
    let (result, seen) = embed_against_server(0, RequestProcessingPreference::new()).await;
    result.expect("request succeeds");
    assert_eq!(seen.traceparents(), vec![Vec::<String>::new()]);
}

#[tokio::test]
async fn invalid_traceparent_is_rejected_before_any_request() {
    let (result, seen) = embed_against_server(
        0,
        RequestProcessingPreference::new().with_traceparent("not-a-traceparent".to_string()),
    )
    .await;
    let err = result.expect_err("invalid traceparent rejected");
    assert!(
        matches!(err, ClientError::InvalidParameter(ref msg) if msg.contains("traceparent")),
        "{err:?}"
    );
    assert!(seen.traceparents().is_empty());
}

#[tokio::test]
async fn trace_headers_in_extra_headers_are_sent_exactly_once() {
    let (result, seen) = embed_against_server(
        0,
        RequestProcessingPreference::new()
            .with_traceparent(PARENT.to_string())
            .with_extra_headers(HashMap::from([
                ("TraceParent".to_string(), PARENT.to_string()),
                ("TRACESTATE".to_string(), "vendor=opaque".to_string()),
            ])),
    )
    .await;
    result.expect("the same parent in both places is not a conflict");
    assert_eq!(seen.traceparents(), vec![vec![PARENT.to_string()]]);
    assert_eq!(seen.tracestates(), vec![vec!["vendor=opaque".to_string()]]);
}

#[tokio::test]
async fn trace_header_set_twice_in_extra_headers_is_rejected() {
    let (result, seen) = embed_against_server(
        0,
        RequestProcessingPreference::new().with_extra_headers(HashMap::from([
            ("traceparent".to_string(), PARENT.to_string()),
            ("Traceparent".to_string(), PARENT.to_string()),
        ])),
    )
    .await;
    let err = result.expect_err("duplicate header rejected");
    assert!(
        matches!(err, ClientError::InvalidParameter(ref msg) if msg.contains("more than once")),
        "{err:?}"
    );
    assert!(seen.traceparents().is_empty());
}

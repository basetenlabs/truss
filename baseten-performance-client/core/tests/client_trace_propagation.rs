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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const PARENT: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00";

#[derive(Clone)]
struct ModelState {
    traceparents: Arc<Mutex<Vec<Vec<String>>>>,
    requests: Arc<AtomicUsize>,
    failures: usize,
}

async fn model_handler(
    State(state): State<ModelState>,
    headers: AxumHeaderMap,
    Json(request): Json<CoreOpenAIEmbeddingsRequest>,
) -> impl IntoResponse {
    let index = state.requests.fetch_add(1, Ordering::SeqCst);
    state.traceparents.lock().unwrap().push(
        headers
            .get_all("traceparent")
            .iter()
            .map(|value| value.to_str().unwrap().to_string())
            .collect(),
    );
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
) -> (Result<(), ClientError>, Vec<Vec<String>>) {
    assert!(
        std::env::var(OTLP_ENDPOINT_ENV_VAR).is_err(),
        "these tests cover export-off behavior"
    );
    let traceparents = Arc::new(Mutex::new(Vec::new()));
    let state = ModelState {
        traceparents: Arc::clone(&traceparents),
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
    let seen = traceparents.lock().unwrap().clone();
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
        seen,
        vec![vec![PARENT.to_string()], vec![PARENT.to_string()]]
    );
}

#[tokio::test]
async fn no_parent_sends_no_traceparent() {
    let (result, seen) = embed_against_server(0, RequestProcessingPreference::new()).await;
    result.expect("request succeeds");
    assert_eq!(seen, vec![Vec::<String>::new()]);
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
    assert!(seen.is_empty());
}

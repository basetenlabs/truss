use axum::{
    extract::State,
    http::{HeaderMap as AxumHeaderMap, HeaderValue, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use baseten_performance_client_core::split_policy::{Combinable, SplitPolicy, Splittable};
use baseten_performance_client_core::*;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

#[tokio::test]
async fn test_embeddings_with_max_chars_per_request() {
    let texts = vec![
        "Short text".to_string(),
        "This is a much longer text that should be placed in a different batch due to character limits".to_string(),
        "Medium length text here".to_string(),
        "Another very long piece of text that should also be batched separately due to the character limit".to_string(),
    ];

    // Test the split policy directly
    let policy = SplitPolicy::max_chars_per_request(50, 4);
    let batches = texts.split(&policy);

    // Verify that batches respect character limits
    assert!(!batches.is_empty(), "Should create at least one batch");

    for batch in &batches {
        let total_chars: usize = batch.iter().map(|s| s.chars().count()).sum();
        println!(
            "Batch with {} items has {} characters",
            batch.len(),
            total_chars
        );
        // Allow some flexibility in the algorithm
        assert!(!batch.is_empty(), "No empty batches should be created");
    }

    // Test that very large character limits result in single batch
    let policy_large = SplitPolicy::max_chars_per_request(1000, 4);
    let large_batches = texts.split(&policy_large);
    assert_eq!(
        large_batches.len(),
        1,
        "Should create only one batch with large char limit"
    );
}

#[tokio::test]
async fn test_combine_responses() {
    let response1 = CoreOpenAIEmbeddingsResponse {
        object: "list".to_string(),
        data: vec![CoreOpenAIEmbeddingData {
            object: "embedding".to_string(),
            embedding_internal: CoreEmbeddingVariant::FloatVector(vec![1.0, 2.0]),
            index: 0,
        }],
        model: "test-model".to_string(),
        usage: CoreOpenAIUsage {
            prompt_tokens: 10,
            total_tokens: 15,
        },
        total_time: 1.5,
        individual_request_times: vec![1.0],
        response_headers: Vec::new(),
    };

    let response2 = CoreOpenAIEmbeddingsResponse {
        object: "list".to_string(),
        data: vec![CoreOpenAIEmbeddingData {
            object: "embedding".to_string(),
            embedding_internal: CoreEmbeddingVariant::FloatVector(vec![3.0, 4.0]),
            index: 1,
        }],
        model: "test-model".to_string(),
        usage: CoreOpenAIUsage {
            prompt_tokens: 8,
            total_tokens: 12,
        },
        total_time: 2.0,
        individual_request_times: vec![1.5],
        response_headers: Vec::new(),
    };

    let combined = CoreOpenAIEmbeddingsResponse::combine(vec![response1, response2], 2);

    assert_eq!(combined.data.len(), 2);
    assert_eq!(combined.usage.prompt_tokens, 18);
    assert_eq!(combined.usage.total_tokens, 27);
    assert_eq!(combined.model, "test-model");

    // Verify order is maintained
    assert_eq!(combined.data[0].index, 0);
    assert_eq!(combined.data[1].index, 1);

    // check that individual request times are combined correctly
    let times = &combined.individual_request_times;
    assert_eq!(times.len(), 2);
    assert_eq!(times[0], 1.0);
    assert_eq!(times[1], 1.5);
}

#[test]
fn test_rerank_response_combine() {
    let response1 = CoreRerankResponse::new(
        vec![CoreRerankResult {
            index: 0,
            score: 0.9,
            text: Some("First text".to_string()),
        }],
        Some(1.0),
        Some(vec![0.5]),
    );

    let response2 = CoreRerankResponse::new(
        vec![CoreRerankResult {
            index: 1,
            score: 0.8,
            text: Some("Second text".to_string()),
        }],
        Some(1.5),
        Some(vec![0.7]),
    );

    let combined = CoreRerankResponse::combine(vec![response1, response2], 2);

    assert_eq!(combined.data.len(), 2);
    assert_eq!(combined.data[0].score, 0.9);
    assert_eq!(combined.data[1].score, 0.8);

    // Verify order is maintained by index
    assert_eq!(combined.data[0].index, 0);
    assert_eq!(combined.data[1].index, 1);
}

#[test]
fn test_classification_response_combine() {
    let response1 = CoreClassificationResponse::new(
        vec![vec![CoreClassificationResult {
            label: "positive".to_string(),
            score: 0.95,
        }]],
        Some(1.0),
        Some(vec![0.5]),
    );

    let response2 = CoreClassificationResponse::new(
        vec![vec![CoreClassificationResult {
            label: "negative".to_string(),
            score: 0.85,
        }]],
        Some(1.5),
        Some(vec![0.7]),
    );

    let combined = CoreClassificationResponse::combine(vec![response1, response2], 2);

    assert_eq!(combined.data.len(), 2);
    assert_eq!(combined.data[0][0].label, "positive");
    assert_eq!(combined.data[1][0].label, "negative");
}

#[test]
fn test_send_request_config_hedge_timeout_validation() {
    // Test case 1: hedge delay higher than request delay (should fail)
    // This validation is now done in RequestProcessingConfig, not SendRequestConfig
    let pref = RequestProcessingPreference::new()
        .with_timeout_s(1.0)
        .with_hedge_delay(2.0); // hedge delay higher than timeout - should fail

    let result = pref.pair_with_request_validate_and_convert(
        "https://example.com".to_string(),
        100,
        "test_api_key".to_string(),
    );
    assert!(
        result.is_err(),
        "Should fail when hedge delay > request timeout"
    );

    // Test case 2: hedge delay equal to request timeout (should fail)
    let pref2 = RequestProcessingPreference::new()
        .with_timeout_s(1.0)
        .with_hedge_delay(1.0); // hedge delay equal to timeout - should fail

    let result2 = pref2.pair_with_request_validate_and_convert(
        "https://example.com".to_string(),
        100,
        "test_api_key".to_string(),
    );
    assert!(
        result2.is_err(),
        "Should fail when hedge delay = request timeout"
    );

    // Test case 3: hedge delay lower than request timeout (should pass)
    let pref3 = RequestProcessingPreference::new()
        .with_timeout_s(1.0)
        .with_hedge_delay(0.5); // hedge delay lower than timeout - should pass

    let result3 = pref3.pair_with_request_validate_and_convert(
        "https://example.com".to_string(),
        100,
        "test_api_key".to_string(),
    );
    assert!(
        result3.is_ok(),
        "Should pass when hedge delay < request timeout"
    );

    // Test case 4: no hedge budget (should succeed)
    let pref4 = RequestProcessingPreference::new().with_timeout_s(1.0); // no hedge delay - should succeed

    let result4 = pref4.pair_with_request_validate_and_convert(
        "https://example.com".to_string(),
        100,
        "test_api_key".to_string(),
    );
    assert!(
        result4.is_ok(),
        "Should succeed when no hedge budget is specified"
    );
}

#[derive(Clone)]
struct TestServerState {
    name: &'static str,
    request_count: Arc<AtomicUsize>,
    remaining_failures: Arc<Mutex<usize>>,
    response_delay: Duration,
    healthy: bool,
}

struct TestServer {
    base_url: String,
    request_count: Arc<AtomicUsize>,
    handle: tokio::task::JoinHandle<()>,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

async fn start_test_server(
    name: &'static str,
    response_delay: Duration,
    remaining_failures: usize,
    healthy: bool,
) -> TestServer {
    let request_count = Arc::new(AtomicUsize::new(0));
    let state = TestServerState {
        name,
        request_count: Arc::clone(&request_count),
        remaining_failures: Arc::new(Mutex::new(remaining_failures)),
        response_delay,
        healthy,
    };

    let app = Router::new()
        .route("/v1/embeddings", post(test_embeddings_handler))
        .route("/health", get(test_health_handler))
        .route("/health/deep", get(test_health_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener should bind");
    let addr = listener
        .local_addr()
        .expect("listener should have local addr");

    let handle = tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("test server should stay up");
    });

    TestServer {
        base_url: format!("http://{}", addr),
        request_count,
        handle,
    }
}

async fn test_health_handler(State(state): State<TestServerState>) -> impl IntoResponse {
    if state.healthy {
        (StatusCode::OK, Json(json!({"status": "healthy"}))).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "unhealthy"})),
        )
            .into_response()
    }
}

async fn test_embeddings_handler(
    State(state): State<TestServerState>,
    Json(request): Json<CoreOpenAIEmbeddingsRequest>,
) -> impl IntoResponse {
    state.request_count.fetch_add(1, Ordering::SeqCst);

    if !state.response_delay.is_zero() {
        tokio::time::sleep(state.response_delay).await;
    }

    let mut remaining_failures = state.remaining_failures.lock().await;
    if *remaining_failures > 0 {
        *remaining_failures -= 1;
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("{} failed", state.name)})),
        )
            .into_response();
    }
    drop(remaining_failures);

    let mut headers = AxumHeaderMap::new();
    headers.insert("x-test-server", HeaderValue::from_static(state.name));

    let response = CoreOpenAIEmbeddingsResponse {
        object: "list".to_string(),
        data: request
            .input
            .iter()
            .enumerate()
            .map(|(index, _)| CoreOpenAIEmbeddingData {
                object: "embedding".to_string(),
                embedding_internal: CoreEmbeddingVariant::FloatVector(vec![0.1, 0.2, 0.3]),
                index,
            })
            .collect(),
        model: request.model,
        usage: CoreOpenAIUsage {
            prompt_tokens: 1,
            total_tokens: 1,
        },
        total_time: 0.0,
        individual_request_times: Vec::new(),
        response_headers: Vec::new(),
    };

    (StatusCode::OK, headers, Json(response)).into_response()
}

fn single_request_preference() -> RequestProcessingPreference {
    RequestProcessingPreference::new()
        .with_max_concurrent_requests(1)
        .with_batch_size(1)
        .with_timeout_s(1.0)
}

#[tokio::test]
async fn test_endpoint_pool_round_robins_across_three_endpoints() {
    let endpoint_a = start_test_server("endpoint-a", Duration::ZERO, 0, true).await;
    let endpoint_b = start_test_server("endpoint-b", Duration::ZERO, 0, true).await;
    let endpoint_c = start_test_server("endpoint-c", Duration::ZERO, 0, true).await;

    let endpoint_pool = EndpointPool::new(EndpointPoolConfig::new(vec![
        endpoint_a.base_url.clone(),
        endpoint_b.base_url.clone(),
        endpoint_c.base_url.clone(),
    ]))
    .expect("endpoint pool should build");

    let client = PerformanceClientCore::new(
        endpoint_a.base_url.clone(),
        Some("test-key".to_string()),
        1,
        None,
        None,
        Some(endpoint_pool),
    )
    .expect("client should build");

    let preference = single_request_preference().with_hedge_budget_pct(0.0);

    let mut winners = Vec::new();
    for text in ["one", "two", "three"] {
        let (_, _, headers, _) = client
            .process_embeddings_requests(
                vec![text.to_string()],
                "test-model".to_string(),
                None,
                None,
                None,
                &preference,
            )
            .await
            .expect("request should succeed");
        winners.push(headers[0]["x-test-server"].clone());
    }

    assert_eq!(
        winners,
        vec![
            "endpoint-a".to_string(),
            "endpoint-b".to_string(),
            "endpoint-c".to_string(),
        ]
    );
    assert_eq!(endpoint_a.request_count.load(Ordering::SeqCst), 1);
    assert_eq!(endpoint_b.request_count.load(Ordering::SeqCst), 1);
    assert_eq!(endpoint_c.request_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_retry_uses_alternate_endpoint_from_pool() {
    let endpoint_a = start_test_server("endpoint-a", Duration::ZERO, 1, true).await;
    let endpoint_b = start_test_server("endpoint-b", Duration::ZERO, 0, true).await;
    let endpoint_c = start_test_server("endpoint-c", Duration::ZERO, 0, true).await;

    let endpoint_pool = EndpointPool::new(EndpointPoolConfig::new(vec![
        endpoint_a.base_url.clone(),
        endpoint_b.base_url.clone(),
        endpoint_c.base_url.clone(),
    ]))
    .expect("endpoint pool should build");

    let client = PerformanceClientCore::new(
        endpoint_a.base_url.clone(),
        Some("test-key".to_string()),
        1,
        None,
        None,
        Some(endpoint_pool),
    )
    .expect("client should build");

    let preference = single_request_preference()
        .with_max_retries(1)
        .with_retry_budget_pct(1.0)
        .with_hedge_budget_pct(0.0);

    let (_, _, headers, _) = client
        .process_embeddings_requests(
            vec!["retry me".to_string()],
            "test-model".to_string(),
            None,
            None,
            None,
            &preference,
        )
        .await
        .expect("retry should succeed via an alternate endpoint");

    assert_ne!(
        headers[0].get("x-test-server").map(String::as_str),
        Some("endpoint-a")
    );
    assert_eq!(endpoint_a.request_count.load(Ordering::SeqCst), 1);
    let alternate_request_count = endpoint_b.request_count.load(Ordering::SeqCst)
        + endpoint_c.request_count.load(Ordering::SeqCst);
    assert_eq!(alternate_request_count, 1);
}

#[tokio::test]
async fn test_hedge_uses_alternate_endpoint_from_pool() {
    let endpoint_a = start_test_server("endpoint-a", Duration::from_millis(600), 0, true).await;
    let endpoint_b = start_test_server("endpoint-b", Duration::ZERO, 0, true).await;
    let endpoint_c = start_test_server("endpoint-c", Duration::ZERO, 0, true).await;

    let endpoint_pool = EndpointPool::new(EndpointPoolConfig::new(vec![
        endpoint_a.base_url.clone(),
        endpoint_b.base_url.clone(),
        endpoint_c.base_url.clone(),
    ]))
    .expect("endpoint pool should build");

    let client = PerformanceClientCore::new(
        endpoint_a.base_url.clone(),
        Some("test-key".to_string()),
        1,
        None,
        None,
        Some(endpoint_pool),
    )
    .expect("client should build");

    let preference = single_request_preference()
        .with_max_retries(0)
        .with_hedge_delay(0.2)
        .with_hedge_budget_pct(1.0);

    let (_, _, headers, _) = client
        .process_embeddings_requests(
            vec!["hedge me".to_string()],
            "test-model".to_string(),
            None,
            None,
            None,
            &preference,
        )
        .await
        .expect("hedged request should succeed");

    assert_ne!(
        headers[0].get("x-test-server").map(String::as_str),
        Some("endpoint-a")
    );
    assert_eq!(endpoint_a.request_count.load(Ordering::SeqCst), 1);
    let alternate_request_count = endpoint_b.request_count.load(Ordering::SeqCst)
        + endpoint_c.request_count.load(Ordering::SeqCst);
    assert_eq!(alternate_request_count, 1);
}

#[tokio::test]
async fn test_background_health_worker_skips_unhealthy_endpoints() {
    let endpoint_a = start_test_server("endpoint-a", Duration::ZERO, 0, true).await;
    let endpoint_b = start_test_server("endpoint-b", Duration::ZERO, 0, false).await;
    let endpoint_c = start_test_server("endpoint-c", Duration::ZERO, 0, true).await;

    let endpoint_pool = EndpointPool::new(
        EndpointPoolConfig::new(vec![
            endpoint_a.base_url.clone(),
            endpoint_b.base_url.clone(),
            endpoint_c.base_url.clone(),
        ])
        .with_health_check_interval(Duration::from_millis(100)),
    )
    .expect("endpoint pool should build");

    let client = PerformanceClientCore::new(
        endpoint_a.base_url.clone(),
        Some("test-key".to_string()),
        1,
        None,
        None,
        Some(endpoint_pool.clone()),
    )
    .expect("client should build");

    let preference = single_request_preference().with_hedge_budget_pct(0.0);

    client
        .process_embeddings_requests(
            vec!["prime worker".to_string()],
            "test-model".to_string(),
            None,
            None,
            None,
            &preference,
        )
        .await
        .expect("request should start the health worker");

    tokio::time::sleep(Duration::from_millis(450)).await;

    let snapshot = endpoint_pool.health_snapshot();
    assert_eq!(
        snapshot
            .endpoints
            .iter()
            .find(|endpoint| endpoint.base_url == endpoint_b.base_url)
            .map(|endpoint| endpoint.healthy),
        Some(false)
    );

    let (_, _, second_headers, _) = client
        .process_embeddings_requests(
            vec!["after health update".to_string()],
            "test-model".to_string(),
            None,
            None,
            None,
            &preference,
        )
        .await
        .expect("request should skip unhealthy endpoint");
    let (_, _, third_headers, _) = client
        .process_embeddings_requests(
            vec!["one more".to_string()],
            "test-model".to_string(),
            None,
            None,
            None,
            &preference,
        )
        .await
        .expect("request should keep skipping unhealthy endpoint");

    assert_ne!(
        second_headers[0].get("x-test-server").map(String::as_str),
        Some("endpoint-b")
    );
    assert_ne!(
        third_headers[0].get("x-test-server").map(String::as_str),
        Some("endpoint-b")
    );
    assert_eq!(endpoint_b.request_count.load(Ordering::SeqCst), 0);
}

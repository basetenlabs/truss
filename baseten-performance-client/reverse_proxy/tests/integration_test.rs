use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use baseten_performance_client_core::{
    CoreOpenAIEmbeddingsRequest, CoreOpenAIEmbeddingData,
    CoreOpenAIUsage, CoreRerankRequest, CoreRerankResult, CoreRerankResponse,
    CoreClassifyRequest, CoreClassificationResult, CoreClassificationResponse,
    CoreEmbeddingVariant, HttpMethod, PerformanceClientCore, RequestProcessingPreference,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::sleep;
use tracing::{error, info};

// Copy MockServer and related structs here for the integration test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockServerConfig {
    pub stall_until_request: Option<usize>,
    pub error_until_request: Option<usize>,
    pub stall_duration_ms: Option<u64>,
    pub response_delay_ms: Option<u64>,
    pub return_429_until: Option<f64>,
    pub internal_server_error_no_stall: bool,
}

impl Default for MockServerConfig {
    fn default() -> Self {
        Self {
            stall_until_request: None,
            error_until_request: None,
            stall_duration_ms: None,
            response_delay_ms: None,
            return_429_until: None,
            internal_server_error_no_stall: false,
        }
    }
}

#[derive(Clone)]
pub struct MockServer {
    port: u16,
    request_count: Arc<AtomicUsize>,
    stall_count: Arc<AtomicUsize>,
    error_count: Arc<AtomicUsize>,
    config: Arc<RwLock<MockServerConfig>>,
}

impl MockServer {
    pub fn new(port: u16) -> Self {
        Self {
            port,
            request_count: Arc::new(AtomicUsize::new(0)),
            stall_count: Arc::new(AtomicUsize::new(0)),
            error_count: Arc::new(AtomicUsize::new(0)),
            config: Arc::new(RwLock::new(MockServerConfig::default())),
        }
    }

    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let app = self.create_app();

        let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", self.port))
            .await
            .map_err(|e| format!("Failed to bind to port {}: {}", self.port, e))?;

        info!("Mock server starting on port {}", self.port);

        axum::serve(listener, app).await?;

        Ok(())
    }

    fn create_app(&self) -> Router {
        let app = Router::new()
            .route("/v1/embeddings", post(embeddings_handler))
            .route("/rerank", post(rerank_handler))
            .route("/predict", post(classify_handler))
            .route("/classify", post(classify_handler))
            .route("/reset", get(reset_handler))
            .route("/config", post(config_handler))
            .route("/stats", get(stats_handler))
            .route("/health_internal", get(health_handler))
            .with_state(self.clone());

        app
    }

    pub async fn update_config(&self, config: MockServerConfig) {
        let mut cfg = self.config.write().await;
        *cfg = config;
    }

    pub async fn get_stats(&self) -> HashMap<String, usize> {
        HashMap::from([
            (
                "request_count".to_string(),
                self.request_count.load(Ordering::Relaxed),
            ),
            (
                "stall_count".to_string(),
                self.stall_count.load(Ordering::Relaxed),
            ),
            (
                "error_count".to_string(),
                self.error_count.load(Ordering::Relaxed),
            ),
        ])
    }

    pub async fn reset_stats(&self) {
        self.request_count.store(0, Ordering::Relaxed);
        self.stall_count.store(0, Ordering::Relaxed);
        self.error_count.store(0, Ordering::Relaxed);
    }
}

async fn health_handler(State(server): State<MockServer>) -> impl IntoResponse {
    Json(json!({
        "service": "mock-server",
        "status": "healthy",
        "port": server.port,
        "request_count": server.request_count.load(Ordering::Relaxed)
    }))
}

async fn reset_handler(State(server): State<MockServer>) -> impl IntoResponse {
    let stats = server.get_stats().await;
    server.reset_stats().await;
    (StatusCode::OK, Json(stats)).into_response()
}

async fn config_handler(
    State(server): State<MockServer>,
    Json(config): Json<MockServerConfig>,
) -> impl IntoResponse {
    server.update_config(config).await;
    (StatusCode::OK, Json(json!({"status": "updated"}))).into_response()
}

async fn stats_handler(State(server): State<MockServer>) -> impl IntoResponse {
    let stats = server.get_stats().await;
    (StatusCode::OK, Json(stats)).into_response()
}

// Simplified handlers for integration test
async fn embeddings_handler(
    State(server): State<MockServer>,
_headers: HeaderMap,
    body: String,
) -> impl IntoResponse {
    let request: CoreOpenAIEmbeddingsRequest = match serde_json::from_str(&body) {
        Ok(req) => req,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Invalid JSON"})),
            )
                .into_response();
        }
    };

    let request_num = server.request_count.fetch_add(1, Ordering::Relaxed) + 1;
    info!("Received embeddings request #{}", request_num);

    // Check config for special behaviors
    let config = server.config.read().await;

    // Check for errors
    if let Some(error_until) = config.error_until_request {
        if request_num <= error_until {
            server.error_count.fetch_add(1, Ordering::Relaxed);
            if config.internal_server_error_no_stall {
                error!("Returning 400 for request #{}", request_num);
                return (StatusCode::BAD_REQUEST, Json(json!({"error": "Bad request"}))).into_response();
            }
        }
    }

    // Create mock response
    let data: Vec<CoreOpenAIEmbeddingData> = request
        .input
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let embedding: Vec<f32> = text
                .chars()
                .take(10)
                .enumerate()
                .map(|(j, c)| ((c as u32 + i as u32 * 1000 + j as u32 * 100) as f32) / 1000.0)
                .collect();

            CoreOpenAIEmbeddingData {
                object: "embedding".to_string(),
                embedding_internal: CoreEmbeddingVariant::FloatVector(embedding),
                index: i,
            }
        })
        .collect();

    let usage = CoreOpenAIUsage {
        prompt_tokens: request.input.len() as u32 * 10,
        total_tokens: request.input.len() as u32 * 10,
    };

    // Return the response in the format expected by the core library
    let response = json!({
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": usage,
        "total_time": 0.0,
        "individual_request_times": [0.0],
        "response_headers": []
    });

    (StatusCode::OK, Json(response)).into_response()
}

async fn rerank_handler(
    State(_server): State<MockServer>,
    _headers: HeaderMap,
    body: String,
) -> impl IntoResponse {
    // Parse the request body
    let request: CoreRerankRequest = match serde_json::from_str(&body) {
        Ok(req) => req,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Invalid JSON"})),
            )
                .into_response();
        }
    };

    // Debug: print the request
    eprintln!("DEBUG: Rerank request received: query={}, texts={:?}", request.query, request.texts);

    // Create mock rerank response in the format expected by the real API
    let data: Vec<CoreRerankResult> = request
        .texts
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let score = if text.to_lowercase().contains(&request.query.to_lowercase()) {
                0.9 + (i as f32 * 0.01)
            } else {
                0.1 + (i as f32 * 0.01)
            };

            CoreRerankResult {
                index: i,
                score: score as f64,
                text: Some(text.clone()),
            }
        })
        .collect();

    // Return the response using the proper struct
    let response = CoreRerankResponse::new(data, Some(0.0), Some(vec![0.0]));
    // Return as plain text to see if the issue is with JSON serialization
    eprintln!("DEBUG: Rerank response created: {:?}", response);
    (StatusCode::OK, Json(response)).into_response()
}

async fn classify_handler(
    State(_server): State<MockServer>,
    _headers: HeaderMap,
    body: String,
) -> impl IntoResponse {
    let request: CoreClassifyRequest = match serde_json::from_str(&body) {
        Ok(req) => req,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Invalid JSON"})),
            )
                .into_response();
        }
    };

    // Create mock classification response
    let data: Vec<Vec<CoreClassificationResult>> = request
        .inputs
        .iter()
        .map(|_text| {
            vec![CoreClassificationResult {
                label: "positive".to_string(),
                score: 0.7,
            }]
        })
        .collect();

// Return the response in the format expected by the core library
    let response = CoreClassificationResponse::new(data, Some(0.0), Some(vec![0.0]));

    (StatusCode::OK, Json(response)).into_response()
}

/// Integration test for the reverse proxy
/// This test verifies that the reverse proxy correctly forwards requests
/// to the mock server and handles various scenarios.
pub struct IntegrationTest {
    proxy_port: u16,
    mock_server_port: u16,
    target_url: String,
    mock_server: Arc<MockServer>,
}

impl IntegrationTest {
    pub fn new(proxy_port: u16, mock_server_port: u16) -> Self {
        let target_url = format!("http://0.0.0.0:{}", mock_server_port);
        let mock_server = Arc::new(MockServer::new(mock_server_port));

        Self {
            proxy_port,
            mock_server_port,
            target_url,
            mock_server,
        }
    }

    /// Run all integration test scenarios
    pub async fn run_all_scenarios(&self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Starting integration tests for reverse proxy");

        // Start mock server
        let mock_server_clone = self.mock_server.clone();
        let mock_server_handle = tokio::spawn(async move {
            if let Err(e) = mock_server_clone.start().await {
                error!("Mock server error: {}", e);
            }
        });

        // Give mock server time to start
        sleep(Duration::from_millis(500)).await;

        // Verify mock server is running
        self.wait_for_service(self.mock_server_port, "mock server")
            .await?;

        // Start reverse proxy server in-process
        let proxy_port = self.proxy_port;
        let mock_server_port = self.mock_server_port;

        info!(
            "Starting reverse proxy on port {} targeting mock server on port {}",
            proxy_port, mock_server_port
        );

        // Create proxy config
        let proxy_config = baseten_reverse_proxy_lib::config::ProxyConfig {
            port: proxy_port,
            default_target_url: Some(format!("http://0.0.0.0:{}", mock_server_port)),
            upstream_api_key: Some("test_api_key".to_string()),
            http_version: 2,
            default_preferences: RequestProcessingPreference::new(),
        };

        let proxy_config = Arc::new(proxy_config);
        let proxy_config_clone = proxy_config.clone();

        // Start reverse proxy server
        let proxy_handle = tokio::spawn(async move {
            if let Err(e) = baseten_reverse_proxy_lib::create_server(proxy_config_clone).await {
                error!("Reverse proxy error: {}", e);
            }
        });

        // Give reverse proxy time to start
        sleep(Duration::from_millis(1000)).await;

        // Verify reverse proxy is running
        self.wait_for_service(self.proxy_port, "reverse proxy")
            .await?;

        // Run test scenarios
        self.scenario_basic_embeddings().await?;
        self.scenario_large_batch().await?;
        self.scenario_with_preferences().await?;
        self.scenario_error_handling().await?;
        self.scenario_concurrent_requests().await?;
        self.scenario_rerank_endpoint().await?;
        self.scenario_classify_endpoint().await?;
        self.scenario_generic_batch().await?;

        // Shutdown services
        info!("Shutting down services");
        drop(proxy_handle);
        drop(mock_server_handle);

        info!("All integration test scenarios passed!");
        Ok(())
    }

    /// Wait for a service to be ready on the given port
    async fn wait_for_service(
        &self,
        port: u16,
        service_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        let url = format!("http://0.0.0.0:{}/health_internal", port);

        for i in 0..10 {
            match client.get(&url).send().await {
                Ok(response) if response.status().is_success() => {
                    info!("{} is ready on port {}", service_name, port);
                    return Ok(());
                }
                Ok(_) => {
                    info!(
                        "Waiting for {} to be ready (attempt {}/10)",
                        service_name,
                        i + 1
                    );
                }
                Err(_) => {
                    info!(
                        "Waiting for {} to be ready (attempt {}/10)",
                        service_name,
                        i + 1
                    );
                }
            }
            sleep(Duration::from_millis(500)).await;
        }

        Err(format!("{} failed to start on port {}", service_name, port).into())
    }

    /// Test basic embeddings request through proxy
    async fn scenario_basic_embeddings(&self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Testing basic embeddings request through proxy");

        let proxy_url = format!("http://0.0.0.0:{}", self.proxy_port);

        let client =
            PerformanceClientCore::new(proxy_url, Some("test_api_key".to_string()), 2, None)?;

        let response = client
            .process_embeddings_requests(
                vec!["Hello world".to_string(), "How are you?".to_string()],
                "text-embedding-ada-002".to_string(),
                None,
                None,
                None,
                &RequestProcessingPreference::new(),
            )
            .await?;

        // Verify response
        let (response, durations, _headers, total_time) = response;
        assert_eq!(response.data.len(), 2);
        assert_eq!(response.model, "text-embedding-ada-002");
        assert_eq!(response.object, "list");

        // Verify proxy metadata
        assert!(total_time.as_secs_f64() > 0.0);
        assert_eq!(durations.len(), 1); // Single batch

        info!("Basic embeddings test passed");
        Ok(())
    }

    /// Test large batch request through proxy
    async fn scenario_large_batch(&self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Testing large batch request through proxy");

        let proxy_url = format!("http://0.0.0.0:{}", self.proxy_port);
        let client =
            PerformanceClientCore::new(proxy_url, Some("test_api_key".to_string()), 2, None)?;

        // Create a large batch
        let inputs: Vec<String> = (0..100).map(|i| format!("Test input {}", i)).collect();

        let response = client
            .process_embeddings_requests(
                inputs,
                "text-embedding-ada-002".to_string(),
                None,
                None,
                None,
                &RequestProcessingPreference::new()
                    .with_batch_size(32)
                    .with_max_concurrent_requests(64),
            )
            .await?;

        // Verify response
        let (response, durations, _headers, _total_time) = response;
        assert_eq!(response.data.len(), 100);
        assert_eq!(durations.len(), 4); // 100/32 = 4 batches

        info!("Large batch test passed");
        Ok(())
    }

    /// Test request with custom preferences
    async fn scenario_with_preferences(&self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Testing request with custom preferences");

        let proxy_url = format!("http://0.0.0.0:{}", self.proxy_port);
        let client =
            PerformanceClientCore::new(proxy_url, Some("test_api_key".to_string()), 2, None)?;

        let response = client
            .process_embeddings_requests(
                vec!["Test with preferences".to_string()],
                "text-embedding-ada-002".to_string(),
                None,
                None,
                None,
                &RequestProcessingPreference::new()
                    .with_max_concurrent_requests(128)
                    .with_batch_size(64)
                    .with_timeout_s(60.0)
                    .with_hedge_budget_pct(0.15)
                    .with_retry_budget_pct(0.08),
            )
            .await?;

        // Verify response
        let (response, _durations, _headers, _total_time) = response;
        assert_eq!(response.data.len(), 1);

        info!("Custom preferences test passed");
        Ok(())
    }

    /// Test error handling through proxy
    async fn scenario_error_handling(&self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Testing error handling through proxy");

        // Reset mock server stats to ensure predictable request numbering
        self.mock_server.reset_stats().await;
        info!("Mock server stats reset");

        // Configure mock server to return errors for the next 2 requests
        let error_config = MockServerConfig {
            error_until_request: Some(2),
            internal_server_error_no_stall: true,
            ..Default::default()
        };
        self.mock_server.update_config(error_config).await;

        let proxy_url = format!("http://0.0.0.0:{}", self.proxy_port);
        let client =
            PerformanceClientCore::new(proxy_url, Some("test_api_key".to_string()), 2, None)?;

        // First request should fail - use preferences that disable hedging and retries
        let no_hedge_prefs = RequestProcessingPreference::new()
            .with_max_concurrent_requests(1)
            .with_hedge_budget_pct(0.0)
            .with_retry_budget_pct(0.0)
            .with_max_retries(0);

        let result1 = client
            .process_embeddings_requests(
                vec!["Error test 1".to_string()],
                "text-embedding-ada-002".to_string(),
                None,
                None,
                None,
                &no_hedge_prefs,
            )
            .await;

        info!("First request result: {:?}", result1);
        assert!(result1.is_err(), "First request should fail");

        // Second request should also fail
        let result2 = client
            .process_embeddings_requests(
                vec!["Error test 2".to_string()],
                "text-embedding-ada-002".to_string(),
                None,
                None,
                None,
                &no_hedge_prefs,
            )
            .await;

        assert!(result2.is_err(), "Second request should fail");

        // Reset config
        self.mock_server
            .update_config(MockServerConfig::default())
            .await;

        // Small delay to ensure config reset takes effect
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Third request should succeed
        let result3 = client
            .process_embeddings_requests(
                vec!["Error test 3".to_string()],
                "text-embedding-ada-002".to_string(),
                None,
                None,
                None,
                &no_hedge_prefs,
            )
            .await;

        assert!(result3.is_ok(), "Third request should succeed");

        info!("Error handling test passed");
        Ok(())
    }

    /// Test concurrent requests through proxy
    async fn scenario_concurrent_requests(&self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Testing concurrent requests through proxy");

        let proxy_url = format!("http://0.0.0.0:{}", self.proxy_port);
        let client = Arc::new(PerformanceClientCore::new(
            proxy_url,
            Some("test_api_key".to_string()),
            2,
            None,
        )?);

        // Create multiple concurrent requests
        let mut handles = Vec::new();

        for i in 0..10 {
            let client_clone = client.clone();
            let handle = tokio::spawn(async move {
                client_clone
                    .process_embeddings_requests(
                        vec![format!("Concurrent test {}", i)],
                        "text-embedding-ada-002".to_string(),
                        None,
                        None,
                        None,
                        &RequestProcessingPreference::new()
                            .with_max_concurrent_requests(32)
                            .with_batch_size(8),
                    )
                    .await
            });
            handles.push(handle);
        }

        // Wait for all requests to complete
        let results = futures::future::join_all(handles).await;

        // Verify all requests succeeded
        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok(response_result) => match response_result {
                    Ok((response, _, _, _)) => {
                        assert_eq!(response.data.len(), 1);
                        info!("Concurrent request {} succeeded", i);
                    }
                    Err(e) => {
                        return Err(format!("Concurrent request {} failed: {}", i, e).into());
                    }
                },
                Err(e) => {
                    return Err(format!("Concurrent request {} task failed: {}", i, e).into());
                }
            }
        }

        info!("Concurrent requests test passed");
        Ok(())
    }

    /// Test rerank endpoint through proxy
    async fn scenario_rerank_endpoint(&self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Testing rerank endpoint through proxy");

        let proxy_url = format!("http://0.0.0.0:{}", self.proxy_port);
        let client =
            PerformanceClientCore::new(proxy_url, Some("test_api_key".to_string()), 2, None)?;

        info!("Sending rerank request to proxy");
        let response = client
            .process_rerank_requests(
                "What is the capital of France?".to_string(),
                vec![
                    "Paris is the capital of France".to_string(),
                    "London is the capital of England".to_string(),
                    "Berlin is the capital of Germany".to_string(),
                ],
                false,
                Some("rerank-model".to_string()),
                true,
                true,
                "Right".to_string(),
                &RequestProcessingPreference::new(),
            )
            .await;

        match &response {
            Ok(_) => info!("Rerank request completed successfully"),
            Err(e) => info!("Rerank request failed: {:?}", e),
        }

        let response = response?;

        // Verify response
        let (response, _durations, _headers, _total_time) = response;
        assert_eq!(response.data.len(), 3);

        // Paris should have the highest score since it contains "capital"
        let paris_score = response.data[0].score;
        let london_score = response.data[1].score;
        let berlin_score = response.data[2].score;

        assert!(paris_score > london_score);
        assert!(paris_score > berlin_score);

        info!("Rerank endpoint test passed");
        Ok(())
    }

    /// Test classify endpoint through proxy
    async fn scenario_classify_endpoint(&self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Testing classify endpoint through proxy");

        let proxy_url = format!("http://0.0.0.0:{}", self.proxy_port);
        let client =
            PerformanceClientCore::new(proxy_url, Some("test_api_key".to_string()), 2, None)?;

        let response = client
            .process_classify_requests(
                vec![
                    "I love this product!".to_string(),
                    "This is terrible.".to_string(),
                    "It's okay, nothing special.".to_string(),
                ],
                Some("classify-model".to_string()),
                true,
                true,
                "Right".to_string(),
                &RequestProcessingPreference::new(),
            )
            .await?;

        // Verify response
        let (response, _durations, _headers, _total_time) = response;
        assert_eq!(response.data.len(), 3);

        // Each response should have classification results with scores
        for classification in &response.data {
            assert!(!classification.is_empty());
            assert!(classification[0].score >= 0.0);
        }

        info!("Classify endpoint test passed");
        Ok(())
    }

    /// Test generic batch endpoint through proxy
    async fn scenario_generic_batch(&self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Testing generic batch endpoint through proxy");

        let proxy_url = format!("http://0.0.0.0:{}", self.proxy_port);
        let client =
            PerformanceClientCore::new(proxy_url, Some("test_api_key".to_string()), 2, None)?;

        let payloads = vec![
            json!({"test": "payload1"}),
            json!({"test": "payload2"}),
            json!({"test": "payload3"}),
        ];

        let custom_headers =
            HashMap::from([("X-Custom-Header".to_string(), "CustomValue".to_string())]);

        let (responses, total_time) = client
            .process_batch_post_requests(
                "/custom-endpoint".to_string(),
                payloads,
                &RequestProcessingPreference::new(),
                Some(custom_headers),
                HttpMethod::POST,
            )
            .await?;

        // Verify response
        assert_eq!(responses.len(), 3);
        assert!(total_time.as_secs_f64() > 0.0);

        info!("Generic batch endpoint test passed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_integration_scenarios() {
        // Set log level if not already set (core library initializes tracing automatically)
        if std::env::var("RUST_LOG").is_err() {
            std::env::set_var("RUST_LOG", "info");
        }

        let test = IntegrationTest::new(8081, 8082);

        if let Err(e) = test.run_all_scenarios().await {
            panic!("Integration test failed: {}", e);
        }
    }
}

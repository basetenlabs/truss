use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use baseten_performance_client_core::{
    PerformanceClientCore, RequestProcessingPreference, HttpMethod,
};
use serde_json::json;
use tracing::{info, error};

mod mock_server;
use mock_server::MockServer;

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
        self.wait_for_service(self.mock_server_port, "mock server").await?;

        // Start reverse proxy server in-process
        let proxy_port = self.proxy_port;
        let mock_server_port = self.mock_server_port;

        info!("Starting reverse proxy on port {} targeting mock server on port {}", proxy_port, mock_server_port);

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
        self.wait_for_service(self.proxy_port, "reverse proxy").await?;

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
    async fn wait_for_service(&self, port: u16, service_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        let url = format!("http://0.0.0.0:{}/health_internal", port);

        for i in 0..10 {
            match client.get(&url).send().await {
                Ok(response) if response.status().is_success() => {
                    info!("{} is ready on port {}", service_name, port);
                    return Ok(());
                }
                Ok(_) => {
                    info!("Waiting for {} to be ready (attempt {}/10)", service_name, i + 1);
                }
                Err(_) => {
                    info!("Waiting for {} to be ready (attempt {}/10)", service_name, i + 1);
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

        let client = PerformanceClientCore::new(
            proxy_url,
            Some("test_api_key".to_string()),
            2,
            None,
        )?;

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

        let client = PerformanceClientCore::new(
            self.target_url.clone(),
            Some("test_api_key".to_string()),
            2,
            None,
        )?;

        // Create a large batch
        let inputs: Vec<String> = (0..100)
            .map(|i| format!("Test input {}", i))
            .collect();

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

        let client = PerformanceClientCore::new(
            self.target_url.clone(),
            Some("test_api_key".to_string()),
            2,
            None,
        )?;

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

        // Configure mock server to return errors
        let error_config = mock_server::MockServerConfig {
            error_until_request: Some(2),
            internal_server_error_no_stall: true,
            ..Default::default()
        };
        self.mock_server.update_config(error_config).await;

        let client = PerformanceClientCore::new(
            self.target_url.clone(),
            Some("test_api_key".to_string()),
            2,
            None,
        )?;

        // First request should fail
        let result1 = client
            .process_embeddings_requests(
                vec!["Error test 1".to_string()],
                "text-embedding-ada-002".to_string(),
                None,
                None,
                None,
                &RequestProcessingPreference::new(),
            )
            .await;

        assert!(result1.is_err(), "First request should fail");

        // Second request should also fail
        let result2 = client
            .process_embeddings_requests(
                vec!["Error test 2".to_string()],
                "text-embedding-ada-002".to_string(),
                None,
                None,
                None,
                &RequestProcessingPreference::new(),
            )
            .await;

        assert!(result2.is_err(), "Second request should fail");

        // Reset config
        self.mock_server.update_config(mock_server::MockServerConfig::default()).await;

        // Third request should succeed
        let result3 = client
            .process_embeddings_requests(
                vec!["Error test 3".to_string()],
                "text-embedding-ada-002".to_string(),
                None,
                None,
                None,
                &RequestProcessingPreference::new(),
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
                Ok(response_result) => {
                    match response_result {
                        Ok((response, _, _, _)) => {
                            assert_eq!(response.data.len(), 1);
                            info!("Concurrent request {} succeeded", i);
                        }
                        Err(e) => {
                            return Err(format!("Concurrent request {} failed: {}", i, e).into());
                        }
                    }
                }
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

        let client = PerformanceClientCore::new(
            self.target_url.clone(),
            Some("test_api_key".to_string()),
            2,
            None,
        )?;

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
            .await?;

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

        let client = PerformanceClientCore::new(
            self.target_url.clone(),
            Some("test_api_key".to_string()),
            2,
            None,
        )?;

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

        let client = PerformanceClientCore::new(
            self.target_url.clone(),
            Some("test_api_key".to_string()),
            2,
            None,
        )?;

        let payloads = vec![
            json!({"test": "payload1"}),
            json!({"test": "payload2"}),
            json!({"test": "payload3"}),
        ];

        let custom_headers = HashMap::from([
            ("X-Custom-Header".to_string(), "CustomValue".to_string()),
        ]);

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

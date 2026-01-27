// Unified request handler proxies requests to Baseten Performance Client Core.
// This enables client-controllable fanout behavior.

use axum::http::{HeaderMap, StatusCode};
use baseten_performance_client_core::{
    errors::ClientError, CancellationToken, CoreClassifyRequest, CoreOpenAIEmbeddingsRequest,
    CoreOpenAIEmbeddingsResponse, CoreRerankRequest, HttpMethod, PerformanceClientCore,
    RequestProcessingPreference,
};

use serde_json::{json, Value};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

use crate::config::ProxyConfig;
use crate::headers::{
    extract_api_key_from_header, extract_customer_request_id, extract_target_url_from_header,
    parse_preferences_from_header,
};

#[derive(Clone)]
pub struct UnifiedHandler {
    config: Arc<ProxyConfig>,
    client: Arc<PerformanceClientCore>,
}

impl UnifiedHandler {
    pub fn new(config: Arc<ProxyConfig>, client: Arc<PerformanceClientCore>) -> Self {
        Self { config, client }
    }

    /// Convert ClientError to appropriate StatusCode and formatted error message
    fn client_error_to_response(error: &ClientError) -> (StatusCode, String) {
        match error {
            ClientError::Http {
                status, message, ..
            } => {
                let status_code =
                    StatusCode::from_u16(*status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                let formatted_message = format!("Message from upstream: {}", message);
                (status_code, formatted_message)
            }
            ClientError::InvalidParameter(message) => (
                StatusCode::BAD_REQUEST,
                format!("Message from upstream: {}", message),
            ),
            ClientError::Serialization(message) => (
                StatusCode::BAD_REQUEST,
                format!("Message from upstream: {}", message),
            ),
            ClientError::LocalTimeout(message, _) => (
                StatusCode::REQUEST_TIMEOUT,
                format!("Message from upstream: {}", message),
            ),
            ClientError::RemoteTimeout(message, _) => (
                StatusCode::GATEWAY_TIMEOUT,
                format!("Message from upstream: {}", message),
            ),
            ClientError::Network(message) => (
                StatusCode::BAD_GATEWAY,
                format!("Message from upstream: {}", message),
            ),
            ClientError::Connect(message) => (
                StatusCode::BAD_GATEWAY,
                format!("Message from upstream: {}", message),
            ),
            ClientError::Cancellation(message) => (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Message from upstream: {}", message),
            ),
        }
    }

    pub async fn handle_request(
        &self,
        path: &str,
        method: HttpMethod,
        headers: HeaderMap,
        body: Value,
    ) -> Result<Value, (StatusCode, String)> {
        // Extract common request elements
        let api_key = match extract_api_key_from_header(&headers) {
            Ok(key) => Some(key),
            Err(_) => {
                // Fall back to upstream API key if no header provided
                self.config.upstream_api_key.clone()
            }
        };

        // Reject request if no API key is available (neither in header nor upstream)
        let api_key = api_key.ok_or_else(|| {
            error!("No API key provided in Authorization header or upstream configuration");
            (
                StatusCode::UNAUTHORIZED,
                "No API key provided in Authorization header or upstream configuration".to_string(),
            )
        })?;

        let mut preferences =
            parse_preferences_from_header(&headers, &self.config.default_preferences);
        let customer_request_id = extract_customer_request_id(&headers);

        debug!("Processing request for path: {}", path);
        if let Some(ref id) = customer_request_id {
            debug!("Customer request ID: {}", id);
        }

        // Get target URL from header, falling back to default
        let target_url = {
            let per_request_target = extract_target_url_from_header(&headers);
            self.config
                .get_target_url(per_request_target)
                .map_err(|e| {
                    error!("Failed to get target URL: {}", e);
                    (
                        StatusCode::BAD_REQUEST,
                        format!("Failed to get target URL: {}", e),
                    )
                })?
        };

        debug!("Using target URL: {}", target_url);

        // Create cancellation token that auto-cancels when dropped (RAII)
        // This ensures proxying stops when the axum handler is revoked
        let cancel_token = CancellationToken::new(true);
        preferences = preferences
            .with_cancel_token(cancel_token)
            .with_primary_api_key_override(api_key);
        let client = self.client.clone();
        // Route request based on path
        match path {
            "/v1/embeddings" => self.handle_embeddings(client, body, preferences).await,
            "/rerank" => self.handle_rerank(client, body, preferences).await,
            "/predict" | "/classify" => self.handle_classify(client, body, preferences).await,
            _ => {
                // Handle generic batch requests
                self.handle_generic_batch(client, path, method, body, preferences)
                    .await
            }
        }
    }

    async fn handle_embeddings(
        &self,
        client: Arc<PerformanceClientCore>,
        body: Value,
        preferences: RequestProcessingPreference,
    ) -> Result<Value, (StatusCode, String)> {
        // Validate request body against CoreOpenAIEmbeddingsRequest schema
        let embeddings_request: CoreOpenAIEmbeddingsRequest = serde_json::from_value(body.clone())
            .map_err(|e| {
                warn!("Invalid embeddings request body: {}", e);
                (
                    StatusCode::BAD_REQUEST,
                    format!("Invalid embeddings request body: {}", e),
                )
            })?;

        // Create request with validated embeddings data
        let request = CoreOpenAIEmbeddingsRequest {
            ..embeddings_request
        };

        info!(
            "Processing embeddings request: {} texts, model: {}",
            request.input.len(),
            request.model
        );

        let (response, durations, _headers, total_time) = client
            .process_embeddings_requests(
                request.input,
                request.model,
                request.encoding_format,
                request.dimensions,
                request.user,
                &preferences,
            )
            .await
            .map_err(|e| {
                error!("Embeddings request failed: {:?}", e);
                Self::client_error_to_response(&e)
            })?;

        debug!(
            "Embeddings request completed in {:.3}s, {} batches",
            total_time.as_secs_f64(),
            durations.len()
        );

        // Create response with proxy metadata
        let response_with_metadata = CoreOpenAIEmbeddingsResponse {
            object: response.object,
            data: response.data,
            model: response.model,
            usage: response.usage,
            total_time: total_time.as_secs_f64(),
            individual_request_times: response.individual_request_times,
            response_headers: vec![],
        };

        Ok(serde_json::to_value(response_with_metadata).unwrap())
    }

    async fn handle_rerank(
        &self,
        client: Arc<PerformanceClientCore>,
        body: Value,
        preferences: RequestProcessingPreference,
    ) -> Result<Value, (StatusCode, String)> {
        debug!("Handling rerank request");

        // Validate request body against CoreRerankRequest schema
        let rerank_request: CoreRerankRequest =
            serde_json::from_value(body.clone()).map_err(|e| {
                error!("Failed to parse rerank request: {}", e);
                (
                    StatusCode::BAD_REQUEST,
                    format!("Invalid rerank request format: {}", e),
                )
            })?;

        info!(
            "Processing rerank request: {} texts, query length: {}",
            rerank_request.texts.len(),
            rerank_request.query.len()
        );

        let (response, durations, _headers, total_time) = client
            .process_rerank_requests(
                rerank_request.query,
                rerank_request.texts,
                rerank_request.raw_scores,
                rerank_request.model,
                rerank_request.return_text,
                rerank_request.truncate,
                rerank_request.truncation_direction,
                &preferences,
            )
            .await
            .map_err(|e| {
                error!("Rerank request failed: {:?}", e);
                Self::client_error_to_response(&e)
            })?;

        debug!(
            "Rerank request completed in {:.3}s, {} batches",
            total_time.as_secs_f64(),
            durations.len()
        );

        // Return only the data array as expected by the core library
        // The core library expects Vec<CoreRerankResult>, not the full response object
        debug!("Rerank response created successfully");
        Ok(serde_json::to_value(response.data).unwrap())
    }

    async fn handle_classify(
        &self,
        client: Arc<PerformanceClientCore>,
        body: Value,
        preferences: RequestProcessingPreference,
    ) -> Result<Value, (StatusCode, String)> {
        // Validate request body against CoreClassifyRequest schema
        let classify_request: CoreClassifyRequest =
            serde_json::from_value(body.clone()).map_err(|e| {
                warn!("Invalid classify request body: {}", e);
                (
                    StatusCode::BAD_REQUEST,
                    format!("Invalid classify request body: {}", e),
                )
            })?;

        info!(
            "Processing classify request: {} inputs",
            classify_request.inputs.len()
        );

        // Flatten inputs from Vec<Vec<String>> to Vec<String> for processing
        let flattened_inputs: Vec<String> = classify_request.inputs.into_iter().flatten().collect();

        let (response, durations, _headers, total_time) = client
            .process_classify_requests(
                flattened_inputs,
                classify_request.model,
                classify_request.raw_scores,
                classify_request.truncate,
                classify_request.truncation_direction,
                &preferences,
            )
            .await
            .map_err(|e| {
                error!("Classify request failed: {:?}", e);
                Self::client_error_to_response(&e)
            })?;

        debug!(
            "Classify request completed in {:.3}s, {} batches",
            total_time.as_secs_f64(),
            durations.len()
        );

        // Return only the data array as expected by the core library
        // The core library expects Vec<Vec<CoreClassificationResult>>, not the full response object
        Ok(serde_json::to_value(response.data).unwrap())
    }

    async fn handle_generic_batch(
        &self,
        client: Arc<PerformanceClientCore>,
        path: &str,
        method: HttpMethod,
        body: Value,
        preferences: RequestProcessingPreference,
    ) -> Result<Value, (StatusCode, String)> {
        // Extract url_path and payloads from request body
        let url_path = body
            .get("url_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                warn!("Missing url_path in generic batch request");
                (
                    StatusCode::BAD_REQUEST,
                    "Missing url_path in generic batch request".to_string(),
                )
            })?
            .to_string();

        let payloads = body
            .get("payloads")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                warn!("Missing payloads in generic batch request");
                (
                    StatusCode::BAD_REQUEST,
                    "Missing payloads in generic batch request".to_string(),
                )
            })?
            .to_vec();

        let custom_headers = body
            .get("custom_headers")
            .and_then(|v| v.as_object())
            .map(|obj| {
                let mut headers = std::collections::HashMap::new();
                for (key, value) in obj {
                    if let Some(value_str) = value.as_str() {
                        headers.insert(key.clone(), value_str.to_string());
                    }
                }
                headers
            });

        info!(
            "Processing generic batch request: path={}, method={}, {} payloads",
            url_path,
            format!("{:?}", method),
            payloads.len()
        );

        let (responses, total_time) = client
            .process_batch_post_requests(url_path, payloads, &preferences, custom_headers, method)
            .await
            .map_err(|e| {
                error!("Generic batch request failed: {:?}", e);
                Self::client_error_to_response(&e)
            })?;

        debug!(
            "Generic batch request completed in {:.3}s, {} responses",
            total_time.as_secs_f64(),
            responses.len()
        );

        // Convert responses to JSON format with metadata
        let json_responses: Vec<Value> = responses
            .into_iter()
            .enumerate()
            .map(|(i, (response, _headers, duration))| {
                json!({
                    "index": i,
                    "response": response,
                    "duration": duration.as_secs_f64()
                })
            })
            .collect();

        let final_response = json!({
            "responses": json_responses,
            "proxy_metadata": {
                "total_time": total_time.as_secs_f64(),
                "response_count": json_responses.len(),
                "path": path,
                "method": format!("{:?}", method)
            }
        });

        Ok(final_response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants;
    use axum::http::{HeaderMap, HeaderValue};
    use baseten_performance_client_core::RequestProcessingPreference;

    // Test CLI struct for tests
    #[derive(Debug, Clone)]
    struct TestCli {
        port: u16,
        target_url: Option<String>,
        upstream_api_key: Option<String>,
        http_version: u8,
        max_concurrent_requests: usize,
        batch_size: usize,
        timeout_s: f64,
    }

    impl ProxyConfig {
        /// Create ProxyConfig from TestCli (for tests)
        fn from_test_cli(cli: TestCli) -> Result<Self, Box<dyn std::error::Error>> {
            let default_preferences = RequestProcessingPreference::new()
                .with_max_concurrent_requests(cli.max_concurrent_requests)
                .with_batch_size(cli.batch_size)
                .with_timeout_s(cli.timeout_s);

            // Resolve upstream API key (from file if starts with /) - ASAP resolution
            let upstream_api_key = if let Some(key) = cli.upstream_api_key {
                if key.starts_with('/') {
                    // Read API key from file immediately and replace with content
                    Some(
                        std::fs::read_to_string(&key)
                            .map_err(|e| format!("Failed to read API key file '{}': {}", key, e))?
                            .trim()
                            .to_string(),
                    )
                } else {
                    Some(key)
                }
            } else {
                None
            };

            Ok(Self {
                port: cli.port,
                default_target_url: cli.target_url,
                upstream_api_key,
                http_version: cli.http_version,
                default_preferences,
            })
        }
    }

    #[test]
    fn test_extract_api_key_from_header_success() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_static("Bearer my-api-key"),
        );

        let result = extract_api_key_from_header(&headers);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "my-api-key");
    }

    #[test]
    fn test_extract_api_key_from_header_missing() {
        let headers = HeaderMap::new();

        let result = extract_api_key_from_header(&headers);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn test_extract_customer_request_id() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-baseten-customer-request-id",
            HeaderValue::from_static("req-12345"),
        );

        let result = extract_customer_request_id(&headers);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "req-12345");
    }

    #[test]
    fn test_extract_customer_request_id_missing() {
        let headers = HeaderMap::new();

        let result = extract_customer_request_id(&headers);
        assert!(result.is_none());
    }

    #[test]
    fn test_proxy_config_from_cli_with_upstream_key() {
        let cli = TestCli {
            port: 9090,
            target_url: Some("https://api.example.com".to_string()),
            upstream_api_key: Some("test-api-key".to_string()),
            http_version: 1,
            max_concurrent_requests: 128,
            batch_size: 64,
            timeout_s: 60.0,
        };

        let config = ProxyConfig::from_test_cli(cli).unwrap();

        assert_eq!(config.port, 9090);
        assert_eq!(
            config.default_target_url,
            Some("https://api.example.com".to_string())
        );
        assert_eq!(config.upstream_api_key, Some("test-api-key".to_string()));
        assert_eq!(config.http_version, 1);
    }

    #[test]
    fn test_proxy_config_from_cli_no_upstream_key() {
        let cli = TestCli {
            port: 9090,
            target_url: Some("https://api.example.com".to_string()),
            upstream_api_key: None,
            http_version: 1,
            max_concurrent_requests: 128,
            batch_size: 64,
            timeout_s: 60.0,
        };

        let config = ProxyConfig::from_test_cli(cli).unwrap();

        assert_eq!(config.port, 9090);
        assert_eq!(
            config.default_target_url,
            Some("https://api.example.com".to_string())
        );
        assert_eq!(config.upstream_api_key, None);
        assert_eq!(config.http_version, 1);
        assert_eq!(
            config.default_preferences.max_concurrent_requests,
            Some(128)
        );
        assert_eq!(config.default_preferences.batch_size, Some(64));
        assert_eq!(config.default_preferences.timeout_s, Some(60.0));
    }

    #[test]
    fn test_get_target_url_with_default() {
        let config = ProxyConfig {
            port: 8080,
            default_target_url: Some("https://default.api.com".to_string()),
            upstream_api_key: Some("test-key".to_string()),
            http_version: 2,
            default_preferences: RequestProcessingPreference::new(),
        };

        // Test with no per-request target (should use default)
        let result = config.get_target_url(None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "https://default.api.com");

        // Test with per-request target (should override default)
        let result = config.get_target_url(Some("https://override.api.com".to_string()));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "https://override.api.com");
    }

    #[test]
    fn test_get_target_url_no_default() {
        let config = ProxyConfig {
            port: 8080,
            default_target_url: None,
            upstream_api_key: Some("test-key".to_string()),
            http_version: 2,
            default_preferences: RequestProcessingPreference::new(),
        };

        // Test with no per-request target and no default (should fail)
        let result = config.get_target_url(None);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "No target URL configured");

        // Test with per-request target (should work)
        let result = config.get_target_url(Some("https://per-request.api.com".to_string()));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "https://per-request.api.com");
    }

    #[test]
    fn test_extract_target_url_from_header() {
        use crate::headers::extract_target_url_from_header;
        use axum::http::{HeaderMap, HeaderValue};

        // Test with X-Target-Host header present
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-target-host",
            HeaderValue::from_static("https://api.example.com"),
        );

        let result = extract_target_url_from_header(&headers);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "https://api.example.com");

        // Test with X-Target-Host header missing
        let headers = HeaderMap::new();

        let result = extract_target_url_from_header(&headers);
        assert!(result.is_none());

        // Test with case-insensitive header
        let mut headers = HeaderMap::new();
        headers.insert(
            "X-Target-Host",
            HeaderValue::from_static("https://api.test.com"),
        );

        let result = extract_target_url_from_header(&headers);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "https://api.test.com");
    }

    #[test]
    fn test_parse_preferences_from_header_with_multiple_fields() {
        use crate::headers::parse_preferences_from_header;
        use axum::http::{HeaderMap, HeaderValue};
        use baseten_performance_client_core::RequestProcessingPreference;

        let mut headers = HeaderMap::new();
        headers.insert(
            "x-baseten-request-preferences",
            HeaderValue::from_static(r#"{"max_concurrent_requests": 128, "timeout_s": 60.0}"#),
        );

        let default_preferences = RequestProcessingPreference::new();
        let preferences = parse_preferences_from_header(&headers, &default_preferences);

        assert_eq!(preferences.max_concurrent_requests, Some(128));
        assert_eq!(preferences.timeout_s, Some(60.0));
    }

    #[test]
    fn test_extract_target_url_from_file() {
        use crate::headers::extract_target_url_from_header;
        use axum::http::{HeaderMap, HeaderValue};
        use std::fs;

        // Create a temporary file with target host
        let temp_dir = std::env::temp_dir();
        let target_host_file = temp_dir.join("test_target_host.txt");
        fs::write(&target_host_file, "https://api.from-file.com\n").unwrap();

        let mut headers = HeaderMap::new();
        headers.insert(
            constants::TARGET_HOST_HEADER,
            HeaderValue::from_str(&target_host_file.to_string_lossy()).unwrap(),
        );

        let result = extract_target_url_from_header(&headers);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "https://api.from-file.com");

        // Clean up
        fs::remove_file(&target_host_file).unwrap();
    }

    #[test]
    fn test_proxy_config_from_cli_with_file_api_key() {
        use std::env;
        use std::fs;

        // Create a temporary file with API key
        let temp_dir = env::temp_dir();
        let api_key_file = temp_dir.join("test_api_key.txt");
        fs::write(&api_key_file, "file-api-key-12345\n").unwrap();

        let cli = TestCli {
            port: 9090,
            target_url: Some("https://api.example.com".to_string()),
            upstream_api_key: Some(api_key_file.to_string_lossy().to_string()),
            http_version: 1,
            max_concurrent_requests: 128,
            batch_size: 64,
            timeout_s: 60.0,
        };

        let config = ProxyConfig::from_test_cli(cli).unwrap();

        assert_eq!(
            config.upstream_api_key,
            Some("file-api-key-12345".to_string())
        );

        // Clean up
        fs::remove_file(&api_key_file).unwrap();
    }

    #[test]
    fn test_proxy_config_from_cli_missing_api_key() {
        let cli = TestCli {
            port: 9090,
            target_url: Some("https://api.example.com".to_string()),
            upstream_api_key: None,
            http_version: 1,
            max_concurrent_requests: 128,
            batch_size: 64,
            timeout_s: 60.0,
        };

        let config = ProxyConfig::from_test_cli(cli).unwrap();
        assert_eq!(config.upstream_api_key, None);
    }
}

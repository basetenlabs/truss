use axum::http::{HeaderMap, StatusCode};
use baseten_performance_client_core::{
    CancellationToken, HttpMethod, PerformanceClientCore, RequestProcessingPreference,
};
use serde_json::{json, Value};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

use crate::config::ProxyConfig;
use crate::headers::{
    extract_api_key_from_header, extract_customer_request_id, extract_model_from_header,
    parse_preferences_from_header,
};

#[derive(Clone)]
pub struct UnifiedHandler {
    config: Arc<ProxyConfig>,
}

impl UnifiedHandler {
    pub fn new(config: Arc<ProxyConfig>) -> Self {
        Self { config }
    }

    pub async fn handle_request(
        &self,
        path: &str,
        method: HttpMethod,
        headers: HeaderMap,
        body: Value,
    ) -> Result<Value, StatusCode> {
        // Extract common elements
        let api_key = extract_api_key_from_header(&headers)?;
        let mut preferences = parse_preferences_from_header(&headers, &self.config.default_preferences);
        let customer_request_id = extract_customer_request_id(&headers);

        debug!("Processing request for path: {}", path);
        if let Some(ref id) = customer_request_id {
            debug!("Customer request ID: {}", id);
        }

        // Create cancellation token that will auto-cancel when dropped (RAII)
        // This ensures that when the axum handler is revoked, the proxy stops proxying
        let cancel_token = CancellationToken::new();
        preferences = preferences.with_cancel_token(cancel_token);

        // Create client with extracted API key
        let client = PerformanceClientCore::new(
            self.config.target_url.clone(),
            Some(api_key),
            self.config.http_version,
            None,
        )
        .map_err(|e| {
            error!("Failed to create PerformanceClientCore: {:?}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        // Route based on path
        match path {
            "/v1/embeddings" => {
                let model = extract_model_from_header(&headers)?;
                self.handle_embeddings(client, body, model, preferences, customer_request_id)
                    .await
            }
            "/rerank" => self.handle_rerank(client, body, preferences, customer_request_id).await,
            "/predict" | "/classify" => {
                self.handle_classify(client, body, preferences, customer_request_id)
                    .await
            }
            _ => {
                // Handle generic batch requests
                self.handle_generic_batch(client, path, method, body, preferences, customer_request_id)
                    .await
            }
        }
    }

    async fn handle_embeddings(
        &self,
        client: PerformanceClientCore,
        body: Value,
        model: String,
        preferences: RequestProcessingPreference,
        customer_request_id: Option<String>,
    ) -> Result<Value, StatusCode> {
        // Extract input array from request body
        let input = body.get("input")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                warn!("Missing or invalid 'input' field in embeddings request");
                StatusCode::BAD_REQUEST
            })?
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let encoding_format = body.get("encoding_format")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let dimensions = body.get("dimensions")
            .and_then(|v| v.as_u64())
            .map(|d| d as u32);

        let user = body.get("user")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        info!(
            "Processing embeddings request: {} texts, model: {}",
            input.len(),
            model
        );

        let (response, durations, headers, total_time) = client
            .process_embeddings_requests(
                input,
                model,
                encoding_format,
                dimensions,
                user,
                &preferences,
            )
            .await
            .map_err(|e| {
                error!("Embeddings request failed: {:?}", e);
                StatusCode::BAD_GATEWAY
            })?;

        debug!(
            "Embeddings request completed in {:.3}s, {} batches",
            total_time.as_secs_f64(),
            durations.len()
        );

        // Create response with proxy metadata
        let response_value = json!({
            "object": response.object,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "data_count": response.data.len(),
            "proxy_metadata": {
                "total_time": total_time.as_secs_f64(),
                "batch_count": durations.len(),
                "customer_request_id": customer_request_id,
                "individual_request_times": response.individual_request_times,
                "response_headers_count": headers.len()
            }
        });

        Ok(response_value)
    }

    async fn handle_rerank(
        &self,
        client: PerformanceClientCore,
        body: Value,
        preferences: RequestProcessingPreference,
        customer_request_id: Option<String>,
    ) -> Result<Value, StatusCode> {
        // Extract required fields
        let query = body.get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                warn!("Missing 'query' field in rerank request");
                StatusCode::BAD_REQUEST
            })?
            .to_string();

        let texts = body.get("texts")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                warn!("Missing 'texts' field in rerank request");
                StatusCode::BAD_REQUEST
            })?
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let raw_scores = body.get("raw_scores")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let model = body.get("model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let return_text = body.get("return_text")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let truncate = body.get("truncate")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let truncation_direction = body.get("truncation_direction")
            .and_then(|v| v.as_str())
            .unwrap_or("Right")
            .to_string();

        info!(
            "Processing rerank request: {} texts, query length: {}",
            texts.len(),
            query.len()
        );

        let (response, durations, headers, total_time) = client
            .process_rerank_requests(
                query,
                texts,
                raw_scores,
                model,
                return_text,
                truncate,
                truncation_direction,
                &preferences,
            )
            .await
            .map_err(|e| {
                error!("Rerank request failed: {:?}", e);
                StatusCode::BAD_GATEWAY
            })?;

        debug!(
            "Rerank request completed in {:.3}s, {} batches",
            total_time.as_secs_f64(),
            durations.len()
        );

        // Create response with proxy metadata
        let response_value = json!({
            "data_count": response.data.len(),
            "proxy_metadata": {
                "total_time": total_time.as_secs_f64(),
                "batch_count": durations.len(),
                "customer_request_id": customer_request_id,
                "individual_request_times": response.individual_request_times,
                "response_headers_count": headers.len()
            }
        });

        Ok(response_value)
    }

    async fn handle_classify(
        &self,
        client: PerformanceClientCore,
        body: Value,
        preferences: RequestProcessingPreference,
        customer_request_id: Option<String>,
    ) -> Result<Value, StatusCode> {
        // Extract inputs field - handle both string and array formats
        let inputs = if let Some(input_str) = body.get("inputs").and_then(|v| v.as_str()) {
            vec![input_str.to_string()]
        } else if let Some(input_array) = body.get("inputs").and_then(|v| v.as_array()) {
            input_array
                .iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        } else {
            return Err(StatusCode::BAD_REQUEST);
        };

        let model = body.get("model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let raw_scores = body.get("raw_scores")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let truncate = body.get("truncate")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let truncation_direction = body.get("truncation_direction")
            .and_then(|v| v.as_str())
            .unwrap_or("Right")
            .to_string();

        info!(
            "Processing classify request: {} inputs",
            inputs.len()
        );

        let (response, durations, headers, total_time) = client
            .process_classify_requests(
                inputs,
                model,
                raw_scores,
                truncate,
                truncation_direction,
                &preferences,
            )
            .await
            .map_err(|e| {
                error!("Classify request failed: {:?}", e);
                StatusCode::BAD_GATEWAY
            })?;

        debug!(
            "Classify request completed in {:.3}s, {} batches",
            total_time.as_secs_f64(),
            durations.len()
        );

        // Create response with proxy metadata
        let response_value = json!({
            "data_count": response.data.len(),
            "proxy_metadata": {
                "total_time": total_time.as_secs_f64(),
                "batch_count": durations.len(),
                "customer_request_id": customer_request_id,
                "individual_request_times": response.individual_request_times,
                "response_headers_count": headers.len()
            }
        });

        Ok(response_value)
    }

    async fn handle_generic_batch(
        &self,
        client: PerformanceClientCore,
        path: &str,
        method: HttpMethod,
        body: Value,
        preferences: RequestProcessingPreference,
        customer_request_id: Option<String>,
    ) -> Result<Value, StatusCode> {
        // Extract url_path and payloads from body
        let url_path = body
            .get("url_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                warn!("Missing url_path in generic batch request");
                StatusCode::BAD_REQUEST
            })?
            .to_string();

        let payloads = body
            .get("payloads")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                warn!("Missing payloads in generic batch request");
                StatusCode::BAD_REQUEST
            })?
            .to_vec();

        let custom_headers = body
            .get("custom_headers")
            .and_then(|v| v.as_object())
            .and_then(|obj| {
                let mut headers = std::collections::HashMap::new();
                for (key, value) in obj {
                    if let Some(value_str) = value.as_str() {
                        headers.insert(key.clone(), value_str.to_string());
                    }
                }
                Some(headers)
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
                StatusCode::BAD_GATEWAY
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
            .map(|(i, (response, headers, duration))| {
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
                "customer_request_id": customer_request_id,
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
    use axum::http::{HeaderMap, HeaderValue, Method};
    use baseten_performance_client_core::{RequestProcessingPreference, HttpMethod};
    use serde_json::json;
    use std::sync::Arc;

    fn create_test_handler() -> UnifiedHandler {
        let config = ProxyConfig {
            port: 8080,
            target_url: "https://api.test.com".to_string(),
            http_version: 2,
            default_preferences: RequestProcessingPreference::new()
                .with_max_concurrent_requests(32)
                .with_batch_size(16)
                .with_timeout_s(30.0),
        };
        UnifiedHandler::new(Arc::new(config))
    }

    fn create_test_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("Bearer test-api-key"));
        headers.insert("x-baseten-model", HeaderValue::from_static("test-model"));
        headers.insert("x-baseten-customer-request-id", HeaderValue::from_static("req-123"));
        headers
    }

    #[test]
    fn test_extract_api_key_from_header_success() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("Bearer my-api-key"));

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
    fn test_extract_model_from_header_success() {
        let mut headers = HeaderMap::new();
        headers.insert("x-baseten-model", HeaderValue::from_static("text-embedding-ada-002"));

        let result = extract_model_from_header(&headers);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "text-embedding-ada-002");
    }

    #[test]
    fn test_extract_model_from_header_missing() {
        let headers = HeaderMap::new();

        let result = extract_model_from_header(&headers);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_extract_customer_request_id() {
        let mut headers = HeaderMap::new();
        headers.insert("x-baseten-customer-request-id", HeaderValue::from_static("req-12345"));

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
    fn test_proxy_config_from_cli() {
        let cli = crate::Cli {
            port: 9090,
            target_url: "https://api.example.com".to_string(),
            http_version: 1,
            max_concurrent_requests: 128,
            batch_size: 64,
            timeout_s: 60.0,
            log_level: "debug".to_string(),
        };

        let config = ProxyConfig::from_cli(cli);

        assert_eq!(config.port, 9090);
        assert_eq!(config.target_url, "https://api.example.com");
        assert_eq!(config.http_version, 1);
        assert_eq!(config.default_preferences.max_concurrent_requests, Some(128));
        assert_eq!(config.default_preferences.batch_size, Some(64));
        assert_eq!(config.default_preferences.timeout_s, Some(60.0));
    }
}

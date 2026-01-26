use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use axum::{
    extract::State,
    http::{StatusCode, HeaderMap},
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{info, warn, error};

/// Mock server for testing the reverse proxy
/// This simulates a real API server with various behaviors
#[derive(Clone)]
pub struct MockServer {
    port: u16,
    request_count: Arc<AtomicUsize>,
    stall_count: Arc<AtomicUsize>,
    error_count: Arc<AtomicUsize>,
    config: Arc<RwLock<MockServerConfig>>,
}

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

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub input: Vec<String>,
    pub model: String,
    pub encoding_format: Option<String>,
    pub dimensions: Option<u32>,
    pub user: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RerankRequest {
    pub query: String,
    pub texts: Vec<String>,
    pub model: Option<String>,
    pub raw_scores: Option<bool>,
    pub return_text: Option<bool>,
    pub truncate: Option<bool>,
    pub truncation_direction: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RerankResult {
    pub index: usize,
    pub score: f32,
    pub text: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RerankResponse {
    pub data: Vec<RerankResult>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClassifyRequest {
    pub inputs: Vec<String>,
    pub model: Option<String>,
    pub raw_scores: Option<bool>,
    pub truncate: Option<bool>,
    pub truncation_direction: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub scores: HashMap<String, f32>,
    pub text: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClassificationResponse {
    pub data: Vec<Vec<ClassificationResult>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HijackPayload {
    pub number_of_requests: usize,
    pub random_payload: String,
    pub uuid_int: u32,
    pub max_batch_size: usize,
    pub max_chars_per_request: Option<usize>,
    pub send_429_until_time: Option<f64>,
    pub stall_x_many_requests: Option<usize>,
    pub stall_for_seconds: Option<f64>,
    pub internal_server_error_no_stall: bool,
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
            .with_state(self.clone());

        app
    }

    pub async fn update_config(&self, config: MockServerConfig) {
        let mut cfg = self.config.write().await;
        *cfg = config;
    }

    pub async fn get_stats(&self) -> HashMap<String, usize> {
        HashMap::from([
            ("request_count".to_string(), self.request_count.load(Ordering::Relaxed)),
            ("stall_count".to_string(), self.stall_count.load(Ordering::Relaxed)),
            ("error_count".to_string(), self.error_count.load(Ordering::Relaxed)),
        ])
    }

    pub async fn reset_stats(&self) {
        self.request_count.store(0, Ordering::Relaxed);
        self.stall_count.store(0, Ordering::Relaxed);
        self.error_count.store(0, Ordering::Relaxed);
    }
}



async fn embeddings_handler(
    State(server): State<MockServer>,
    headers: HeaderMap,
    body: String,
) -> impl IntoResponse {
    // Parse the request body
    let request: EmbeddingRequest = match serde_json::from_str(&body) {
        Ok(req) => req,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Invalid JSON"}))
            ).into_response();
        }
    };
    let request_num = server.request_count.fetch_add(1, Ordering::Relaxed) + 1;

    info!("Received embeddings request #{}", request_num);

    // Check for customer request ID header
    if let Some(customer_id) = headers.get("x-baseten-customer-request-id") {
        if let Ok(id_str) = customer_id.to_str() {
            if !id_str.starts_with("perfclient") {
                error!("Invalid customer request ID: {}", id_str);
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": "Invalid customer request ID"}))
                ).into_response();
            }
        }
    } else {
        error!("Missing customer request ID header");
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Missing customer request ID header"}))
        ).into_response();
    }

    // Check config for special behaviors
    let config = server.config.read().await;

    // Check for 429 responses
    if let Some(until_time) = config.return_429_until {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        if current_time < until_time {
            warn!("Returning 429 for request #{}", request_num);
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(json!({"error": "Too many requests, please try again later"}))
            ).into_response();
        }
    }

    // Check for stalling
    if let Some(stall_until) = config.stall_until_request {
        if request_num <= stall_until {
            server.stall_count.fetch_add(1, Ordering::Relaxed);
            if let Some(stall_duration) = config.stall_duration_ms {
                info!("Stalling request #{} for {}ms", request_num, stall_duration);
                tokio::time::sleep(Duration::from_millis(stall_duration)).await;
            }
        }
    }

    // Check for errors
    if let Some(error_until) = config.error_until_request {
        if request_num <= error_until {
            server.error_count.fetch_add(1, Ordering::Relaxed);
            if config.internal_server_error_no_stall {
                error!("Returning 500 for request #{}", request_num);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": "Internal server error, no stall"}))
                ).into_response();
            }
        }
    }

    // Add response delay if configured
    if let Some(delay_ms) = config.response_delay_ms {
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
    }

    // Create mock response
    let data: Vec<EmbeddingData> = request
        .input
        .iter()
        .enumerate()
        .map(|(i, text)| {
            // Create a simple embedding based on text hash
            let embedding: Vec<f32> = text
                .chars()
                .take(10) // Limit to first 10 chars
                .enumerate()
                .map(|(j, c)| ((c as u32 + i as u32 * 1000 + j as u32 * 100) as f32) / 1000.0)
                .collect();

            EmbeddingData {
                object: "embedding".to_string(),
                embedding,
                index: i,
            }
        })
        .collect();

    let usage = EmbeddingUsage {
        prompt_tokens: request.input.len() as u32 * 10, // Mock token count
        total_tokens: request.input.len() as u32 * 10,
    };

    let response = EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: request.model,
        usage,
    };

    (StatusCode::OK, Json(response)).into_response()
}

async fn rerank_handler(
    State(server): State<MockServer>,
    headers: HeaderMap,
    body: String,
) -> impl IntoResponse {
    // Parse the request body
    let request: RerankRequest = match serde_json::from_str(&body) {
        Ok(req) => req,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Invalid JSON"}))
            ).into_response();
        }
    };
    let request_num = server.request_count.fetch_add(1, Ordering::Relaxed) + 1;

    info!("Received rerank request #{}", request_num);

    // Similar header validation as embeddings
    if let Some(customer_id) = headers.get("x-baseten-customer-request-id") {
        if let Ok(id_str) = customer_id.to_str() {
            if !id_str.starts_with("perfclient") {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": "Invalid customer request ID"}))
                ).into_response();
            }
        }
    } else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Missing customer request ID header"}))
        ).into_response();
    }

    // Create mock rerank response
    let data: Vec<RerankResult> = request
        .texts
        .iter()
        .enumerate()
        .map(|(i, text)| {
            // Simple scoring based on text similarity to query
            let score = if text.to_lowercase().contains(&request.query.to_lowercase()) {
                0.9 + (i as f32 * 0.01)
            } else {
                0.1 + (i as f32 * 0.01)
            };

            RerankResult {
                index: i,
                score,
                text: Some(text.clone()),
            }
        })
        .collect();

    let response = RerankResponse { data };

    (StatusCode::OK, Json(response)).into_response()
}

async fn classify_handler(
    State(server): State<MockServer>,
    headers: HeaderMap,
    body: String,
) -> impl IntoResponse {
    // Parse the request body
    let request: ClassifyRequest = match serde_json::from_str(&body) {
        Ok(req) => req,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Invalid JSON"}))
            ).into_response();
        }
    };
    let request_num = server.request_count.fetch_add(1, Ordering::Relaxed) + 1;

    info!("Received classify request #{}", request_num);

    // Similar header validation
    if let Some(customer_id) = headers.get("x-baseten-customer-request-id") {
        if let Ok(id_str) = customer_id.to_str() {
            if !id_str.starts_with("perfclient") {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": "Invalid customer request ID"}))
                ).into_response();
            }
        }
    } else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Missing customer request ID header"}))
        ).into_response();
    }

    // Create mock classification response
    let data: Vec<Vec<ClassificationResult>> = request
        .inputs
        .iter()
        .map(|text| {
            let mut scores = HashMap::new();
            scores.insert("positive".to_string(), 0.7);
            scores.insert("negative".to_string(), 0.2);
            scores.insert("neutral".to_string(), 0.1);

            vec![ClassificationResult {
                scores,
                text: Some(text.clone()),
            }]
        })
        .collect();

    let response = ClassificationResponse { data };

    (StatusCode::OK, Json(response)).into_response()
}

async fn reset_handler(
    State(server): State<MockServer>,
) -> impl IntoResponse {
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

async fn stats_handler(
    State(server): State<MockServer>,
) -> impl IntoResponse {
    let stats = server.get_stats().await;
    (StatusCode::OK, Json(stats)).into_response()
}

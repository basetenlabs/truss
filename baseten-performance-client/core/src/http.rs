use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type alias for HTTP response headers
pub type HeaderMap = HashMap<String, String>;

/// HTTP methods supported by the batch client
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HttpMethod {
    #[default]
    POST,
    PUT,
    PATCH,
    DELETE,
    GET,
    HEAD,
    OPTIONS,
}

impl From<HttpMethod> for reqwest::Method {
    fn from(method: HttpMethod) -> Self {
        match method {
            HttpMethod::POST => reqwest::Method::POST,
            HttpMethod::PUT => reqwest::Method::PUT,
            HttpMethod::PATCH => reqwest::Method::PATCH,
            HttpMethod::DELETE => reqwest::Method::DELETE,
            HttpMethod::GET => reqwest::Method::GET,
            HttpMethod::HEAD => reqwest::Method::HEAD,
            HttpMethod::OPTIONS => reqwest::Method::OPTIONS,
        }
    }
}

impl HttpMethod {
    /// Parse a string into an HttpMethod, defaulting to POST for None or empty strings
    pub fn from_str(method: Option<&str>) -> Result<Self, String> {
        match method {
            Some("GET") => Ok(HttpMethod::GET),
            Some("PUT") => Ok(HttpMethod::PUT),
            Some("PATCH") => Ok(HttpMethod::PATCH),
            Some("DELETE") => Ok(HttpMethod::DELETE),
            Some("HEAD") => Ok(HttpMethod::HEAD),
            Some("OPTIONS") => Ok(HttpMethod::OPTIONS),
            Some("POST") | None => Ok(HttpMethod::POST),
            Some(invalid) => Err(format!(
                "Invalid HTTP method '{}'. Supported methods: GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS",
                invalid
            )),
        }
    }

    /// Returns true if this HTTP method typically has a request body
    pub fn has_body(&self) -> bool {
        match self {
            // pattern helps when extending in future
            HttpMethod::POST | HttpMethod::PUT | HttpMethod::PATCH => true,
            HttpMethod::GET | HttpMethod::DELETE | HttpMethod::HEAD | HttpMethod::OPTIONS => false,
        }
    }
}

// Default functions for serde
fn default_total_time() -> f64 {
    -1.0
}

fn default_individual_request_times() -> Vec<f64> {
    Vec::new()
}

fn default_response_headers() -> Vec<HeaderMap> {
    Vec::new()
}

// --- Core OpenAI Compatible Structures ---
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CoreOpenAIEmbeddingsRequest {
    pub input: Vec<String>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum CoreEmbeddingVariant {
    Base64(String),
    FloatVector(Vec<f32>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CoreOpenAIEmbeddingData {
    pub object: String,
    #[serde(rename = "embedding")]
    pub embedding_internal: CoreEmbeddingVariant,
    pub index: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CoreOpenAIUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CoreOpenAIEmbeddingsResponse {
    pub object: String,
    pub data: Vec<CoreOpenAIEmbeddingData>,
    pub model: String,
    pub usage: CoreOpenAIUsage,
    #[serde(default = "default_total_time")]
    pub total_time: f64,
    #[serde(default = "default_individual_request_times")]
    pub individual_request_times: Vec<f64>,
    #[serde(default = "default_response_headers")]
    pub response_headers: Vec<HeaderMap>,
}

// --- Core Rerank Structures ---
#[derive(Serialize, Deserialize, Debug)]
pub struct CoreRerankRequest {
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub raw_scores: bool,
    pub return_text: bool,
    pub texts: Vec<String>,
    pub truncate: bool,
    pub truncation_direction: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CoreRerankResult {
    pub index: usize,
    pub score: f64,
    pub text: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CoreRerankResponse {
    pub object: String,
    pub data: Vec<CoreRerankResult>,
    #[serde(default = "default_total_time")]
    pub total_time: f64,
    #[serde(default = "default_individual_request_times")]
    pub individual_request_times: Vec<f64>,
    #[serde(default = "default_response_headers")]
    pub response_headers: Vec<HeaderMap>,
}

impl CoreRerankResponse {
    pub fn new(
        data: Vec<CoreRerankResult>,
        total_time: Option<f64>,
        individual_request_times: Option<Vec<f64>>,
    ) -> Self {
        CoreRerankResponse {
            object: "list".to_string(),
            data,
            total_time: total_time.unwrap_or(-1.0),
            individual_request_times: individual_request_times.unwrap_or_default(),
            response_headers: Vec::new(),
        }
    }
}

// --- Core Classification Structures ---
#[derive(Serialize, Deserialize, Debug)]
pub struct CoreClassifyRequest {
    pub inputs: Vec<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub raw_scores: bool,
    pub truncate: bool,
    pub truncation_direction: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CoreClassificationResult {
    pub label: String,
    pub score: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CoreClassificationResponse {
    pub object: String,
    pub data: Vec<Vec<CoreClassificationResult>>,
    #[serde(default = "default_total_time")]
    pub total_time: f64,
    #[serde(default = "default_individual_request_times")]
    pub individual_request_times: Vec<f64>,
    #[serde(default = "default_response_headers")]
    pub response_headers: Vec<HeaderMap>,
}

impl CoreClassificationResponse {
    pub fn new(
        data: Vec<Vec<CoreClassificationResult>>,
        total_time: Option<f64>,
        individual_request_times: Option<Vec<f64>>,
    ) -> Self {
        CoreClassificationResponse {
            object: "list".to_string(),
            data,
            total_time: total_time.unwrap_or(-1.0),
            individual_request_times: individual_request_times.unwrap_or_default(),
            response_headers: Vec::new(),
        }
    }
}

// --- Core Batch Post Response ---
pub struct CoreBatchPostResponse {
    pub data: Vec<serde_json::Value>,
    pub total_time: f64,
    pub individual_request_times: Vec<f64>,
    pub response_headers: Vec<HeaderMap>,
}

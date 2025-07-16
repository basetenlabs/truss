use serde::{Deserialize, Serialize};

// --- Core OpenAI Compatible Structures ---
#[derive(Serialize, Debug, Clone)]
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

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum CoreEmbeddingVariant {
    Base64(String),
    FloatVector(Vec<f32>),
}

#[derive(Deserialize, Debug, Clone)]
pub struct CoreOpenAIEmbeddingData {
    pub object: String,
    #[serde(rename = "embedding")]
    pub embedding_internal: CoreEmbeddingVariant,
    pub index: usize,
}

#[derive(Deserialize, Debug, Clone)]
pub struct CoreOpenAIUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct CoreOpenAIEmbeddingsResponse {
    pub object: String,
    pub data: Vec<CoreOpenAIEmbeddingData>,
    pub model: String,
    pub usage: CoreOpenAIUsage,
    pub total_time: Option<f64>,
    pub individual_request_times: Option<Vec<f64>>,
}

// --- Core Rerank Structures ---
#[derive(Serialize, Debug)]
pub struct CoreRerankRequest {
    pub query: String,
    pub raw_scores: bool,
    pub return_text: bool,
    pub texts: Vec<String>,
    pub truncate: bool,
    pub truncation_direction: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct CoreRerankResult {
    pub index: usize,
    pub score: f64,
    pub text: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CoreRerankResponse {
    pub object: String,
    pub data: Vec<CoreRerankResult>,
    pub total_time: Option<f64>,
    pub individual_request_times: Option<Vec<f64>>,
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
            total_time,
            individual_request_times,
        }
    }
}

// --- Core Classification Structures ---
#[derive(Serialize, Debug)]
pub struct CoreClassifyRequest {
    pub inputs: Vec<Vec<String>>,
    pub raw_scores: bool,
    pub truncate: bool,
    pub truncation_direction: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct CoreClassificationResult {
    pub label: String,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct CoreClassificationResponse {
    pub object: String,
    pub data: Vec<Vec<CoreClassificationResult>>,
    pub total_time: Option<f64>,
    pub individual_request_times: Option<Vec<f64>>,
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
            total_time,
            individual_request_times,
        }
    }
}

// --- Core Batch Post Response ---
pub struct CoreBatchPostResponse {
    pub data: Vec<serde_json::Value>,
    pub total_time: f64,
    pub individual_request_times: Vec<f64>,
    pub response_headers: Vec<std::collections::HashMap<String, String>>,
}

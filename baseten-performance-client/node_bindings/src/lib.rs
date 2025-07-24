use baseten_performance_client_core::{
  ClientError, CoreClassificationResponse, CoreEmbeddingVariant, CoreOpenAIEmbeddingsResponse,
  CoreRerankResponse, PerformanceClientCore, DEFAULT_BATCH_SIZE, DEFAULT_CONCURRENCY,
  DEFAULT_REQUEST_TIMEOUT_S,
};
use napi_derive::napi;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;

// Use constants from core crate - no more hardcoded values!

// Helper function to create NAPI errors
fn create_napi_error(msg: &str) -> napi::Error {
  napi::Error::new(napi::Status::InvalidArg, msg)
}

// Convert core client errors to NAPI errors
fn convert_core_error_to_napi_error(error: ClientError) -> napi::Error {
  match error {
    ClientError::InvalidParameter(msg) => create_napi_error(&msg),
    ClientError::Network(msg) => napi::Error::new(napi::Status::GenericFailure, msg),
    ClientError::Http { status: _, message } => {
      napi::Error::new(napi::Status::GenericFailure, message)
    }
    ClientError::Serialization(msg) => napi::Error::new(napi::Status::GenericFailure, msg),
    ClientError::Timeout(msg) => napi::Error::new(napi::Status::GenericFailure, msg),
    ClientError::Cancellation(msg) => napi::Error::new(napi::Status::GenericFailure, msg),
  }
}

// Response types for Node.js (simplified for napi compatibility)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIEmbeddingData {
  pub object: String,
  pub embedding: Vec<f32>,
  pub index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIUsage {
  pub prompt_tokens: u32,
  pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIEmbeddingsResponse {
  pub object: String,
  pub data: Vec<OpenAIEmbeddingData>,
  pub model: String,
  pub usage: OpenAIUsage,
  pub total_time: f64,
  pub individual_request_times: Vec<f64>,
  pub response_headers: Vec<HashMap<String, String>>,
}

impl From<CoreOpenAIEmbeddingsResponse> for OpenAIEmbeddingsResponse {
  fn from(core_response: CoreOpenAIEmbeddingsResponse) -> Self {
    Self {
      object: core_response.object,
      data: core_response
        .data
        .into_iter()
        .map(|data| {
          let embedding = match data.embedding_internal {
            CoreEmbeddingVariant::FloatVector(vec) => vec,
            CoreEmbeddingVariant::Base64(_) => {
              // For now, return empty vec for base64 - could be enhanced later
              Vec::new()
            }
          };
          OpenAIEmbeddingData {
            object: data.object,
            embedding,
            index: data.index as u32,
          }
        })
        .collect(),
      model: core_response.model,
      usage: OpenAIUsage {
        prompt_tokens: core_response.usage.prompt_tokens,
        total_tokens: core_response.usage.total_tokens,
      },
      total_time: core_response.total_time,
      individual_request_times: core_response.individual_request_times,
      response_headers: core_response.response_headers,
    }
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
  pub index: u32,
  pub score: f64,
  pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResponse {
  pub object: String,
  pub data: Vec<RerankResult>,
  pub total_time: f64,
  pub individual_request_times: Vec<f64>,
  pub response_headers: Vec<HashMap<String, String>>,
}

impl From<CoreRerankResponse> for RerankResponse {
  fn from(core_response: CoreRerankResponse) -> Self {
    Self {
      object: core_response.object,
      data: core_response
        .data
        .into_iter()
        .map(|result| RerankResult {
          index: result.index as u32,
          score: result.score,
          text: result.text,
        })
        .collect(),
      total_time: core_response.total_time,
      individual_request_times: core_response.individual_request_times,
      response_headers: core_response.response_headers,
    }
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
  pub label: String,
  pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResponse {
  pub object: String,
  pub data: Vec<Vec<ClassificationResult>>,
  pub total_time: f64,
  pub individual_request_times: Vec<f64>,
  pub response_headers: Vec<HashMap<String, String>>,
}

impl From<CoreClassificationResponse> for ClassificationResponse {
  fn from(core_response: CoreClassificationResponse) -> Self {
    Self {
      object: core_response.object,
      data: core_response
        .data
        .into_iter()
        .map(|group| {
          group
            .into_iter()
            .map(|result| ClassificationResult {
              label: result.label,
              score: result.score,
            })
            .collect()
        })
        .collect(),
      total_time: core_response.total_time,
      individual_request_times: core_response.individual_request_times,
      response_headers: core_response.response_headers,
    }
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchPostResponse {
  pub data: Vec<JsonValue>,
  pub response_headers: Vec<HashMap<String, String>>,
  pub individual_request_times: Vec<f64>,
  pub total_time: f64,
}

#[napi]
pub struct PerformanceClient {
  core_client: PerformanceClientCore,
}

#[napi]
impl PerformanceClient {
  #[napi(constructor)]
  pub fn new(base_url: String, api_key: Option<String>) -> napi::Result<Self> {
    let core_client =
      PerformanceClientCore::new(base_url, api_key, 2).map_err(convert_core_error_to_napi_error)?;

    Ok(Self { core_client })
  }

  #[napi]
  pub async fn embed(
    &self,
    input: Vec<String>,
    model: String,
    encoding_format: Option<String>,
    dimensions: Option<u32>,
    user: Option<String>,
    max_concurrent_requests: Option<u32>,
    batch_size: Option<u32>,
    timeout_s: Option<f64>,
    max_chars_per_request: Option<u32>,
    hedge_delay: Option<f64>,
  ) -> napi::Result<serde_json::Value> {
    if input.is_empty() {
      return Err(create_napi_error("Input list cannot be empty"));
    }

    let max_concurrent_requests =
      max_concurrent_requests.unwrap_or(DEFAULT_CONCURRENCY as u32) as usize;
    let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE as u32) as usize;
    let timeout_s = timeout_s.unwrap_or(DEFAULT_REQUEST_TIMEOUT_S);
    let max_chars_per_request = max_chars_per_request.map(|x| x as usize);
    let hedge_delay = hedge_delay;

    let result = self
      .core_client
      .process_embeddings_requests(
        input,
        model,
        encoding_format,
        dimensions,
        user,
        max_concurrent_requests,
        batch_size,
        max_chars_per_request,
        timeout_s,
        hedge_delay,
      )
      .await
      .map_err(convert_core_error_to_napi_error)?;

    let (core_response, batch_durations, total_time) = result;
    let mut response = OpenAIEmbeddingsResponse::from(core_response);
    response.total_time = total_time.as_secs_f64();
    response.individual_request_times = batch_durations
      .into_iter()
      .map(|d| d.as_secs_f64())
      .collect();

    serde_json::to_value(response)
      .map_err(|e| create_napi_error(&format!("Serialization error: {}", e)))
  }

  #[napi]
  pub async fn rerank(
    &self,
    query: String,
    texts: Vec<String>,
    raw_scores: Option<bool>,
    return_text: Option<bool>,
    truncate: Option<bool>,
    truncation_direction: Option<String>,
    max_concurrent_requests: Option<u32>,
    batch_size: Option<u32>,
    timeout_s: Option<f64>,
    max_chars_per_request: Option<u32>,
    hedge_delay: Option<f64>,
  ) -> napi::Result<serde_json::Value> {
    if texts.is_empty() {
      return Err(create_napi_error("Texts list cannot be empty"));
    }

    let max_concurrent_requests =
      max_concurrent_requests.unwrap_or(DEFAULT_CONCURRENCY as u32) as usize;
    let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE as u32) as usize;
    let timeout_s = timeout_s.unwrap_or(DEFAULT_REQUEST_TIMEOUT_S);
    let max_chars_per_request = max_chars_per_request.map(|x| x as usize);
    let hedge_delay = hedge_delay;

    let result = self
      .core_client
      .process_rerank_requests(
        query,
        texts,
        raw_scores.unwrap_or(false),
        return_text.unwrap_or(false),
        truncate.unwrap_or(false),
        truncation_direction.unwrap_or_else(|| "Right".to_string()),
        max_concurrent_requests,
        batch_size,
        max_chars_per_request,
        timeout_s,
        hedge_delay,
      )
      .await
      .map_err(convert_core_error_to_napi_error)?;

    let (core_response, batch_durations, total_time) = result;
    let mut response = RerankResponse::from(core_response);
    response.total_time = total_time.as_secs_f64();
    response.individual_request_times = batch_durations
      .into_iter()
      .map(|d| d.as_secs_f64())
      .collect();

    serde_json::to_value(response)
      .map_err(|e| create_napi_error(&format!("Serialization error: {}", e)))
  }

  #[napi]
  pub async fn classify(
    &self,
    inputs: Vec<String>,
    raw_scores: Option<bool>,
    truncate: Option<bool>,
    truncation_direction: Option<String>,
    max_concurrent_requests: Option<u32>,
    batch_size: Option<u32>,
    timeout_s: Option<f64>,
    max_chars_per_request: Option<u32>,
    hedge_delay: Option<f64>,
  ) -> napi::Result<serde_json::Value> {
    if inputs.is_empty() {
      return Err(create_napi_error("Inputs list cannot be empty"));
    }

    let max_concurrent_requests =
      max_concurrent_requests.unwrap_or(DEFAULT_CONCURRENCY as u32) as usize;
    let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE as u32) as usize;
    let timeout_s = timeout_s.unwrap_or(DEFAULT_REQUEST_TIMEOUT_S);
    let max_chars_per_request = max_chars_per_request.map(|x| x as usize);
    let hedge_delay = hedge_delay;

    let result = self
      .core_client
      .process_classify_requests(
        inputs,
        raw_scores.unwrap_or(false),
        truncate.unwrap_or(false),
        truncation_direction.unwrap_or_else(|| "Right".to_string()),
        max_concurrent_requests,
        batch_size,
        max_chars_per_request,
        timeout_s,
        hedge_delay,
      )
      .await
      .map_err(convert_core_error_to_napi_error)?;

    let (core_response, batch_durations, total_time) = result;
    let mut response = ClassificationResponse::from(core_response);
    response.total_time = total_time.as_secs_f64();
    response.individual_request_times = batch_durations
      .into_iter()
      .map(|d| d.as_secs_f64())
      .collect();

    serde_json::to_value(response)
      .map_err(|e| create_napi_error(&format!("Serialization error: {}", e)))
  }

  #[napi]
  pub async fn batch_post(
    &self,
    url_path: String,
    payloads: Vec<JsonValue>,
    max_concurrent_requests: Option<u32>,
    timeout_s: Option<f64>,
  ) -> napi::Result<serde_json::Value> {
    if payloads.is_empty() {
      return Err(create_napi_error("Payloads list cannot be empty"));
    }

    let max_concurrent_requests =
      max_concurrent_requests.unwrap_or(DEFAULT_CONCURRENCY as u32) as usize;
    let timeout_s = timeout_s.unwrap_or(DEFAULT_REQUEST_TIMEOUT_S);

    let result = self
      .core_client
      .process_batch_post_requests(url_path, payloads, max_concurrent_requests, timeout_s, None)
      .await
      .map_err(convert_core_error_to_napi_error)?;

    let (results, total_time) = result;
    let mut data = Vec::new();
    let mut response_headers = Vec::new();
    let mut individual_request_times = Vec::new();

    for (json_value, headers, duration) in results {
      data.push(json_value);
      response_headers.push(headers);
      individual_request_times.push(duration.as_secs_f64());
    }

    let response = BatchPostResponse {
      data,
      response_headers,
      individual_request_times,
      total_time: total_time.as_secs_f64(),
    };

    serde_json::to_value(response)
      .map_err(|e| create_napi_error(&format!("Serialization error: {}", e)))
  }
}

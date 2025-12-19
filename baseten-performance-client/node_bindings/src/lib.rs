#![allow(clippy::too_many_arguments)]

use baseten_performance_client_core::{
  ClientError, CoreClassificationResponse, CoreEmbeddingVariant, CoreOpenAIEmbeddingsResponse,
  CoreRerankResponse, HttpClientWrapper as HttpClientWrapperRs, PerformanceClientCore,
  RequestProcessingPreference as RustRequestProcessingPreference, DEFAULT_BATCH_SIZE,
  DEFAULT_CONCURRENCY, DEFAULT_HEDGE_BUDGET_PERCENTAGE, DEFAULT_REQUEST_TIMEOUT_S,
  DEFAULT_RETRY_BUDGET_PERCENTAGE,
};
use napi_derive::napi;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;

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
    ClientError::Connect(msg) => napi::Error::new(napi::Status::GenericFailure, msg),
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

/// User-facing configuration for request processing.
/// Provides sensible defaults and getters for all properties.
#[napi]
pub struct RequestProcessingPreference {
  inner: RustRequestProcessingPreference,
}

#[napi]
impl RequestProcessingPreference {
  #[napi(constructor)]
  pub fn new(
    max_concurrent_requests: Option<u32>,
    batch_size: Option<u32>,
    timeout_s: Option<f64>,
    max_chars_per_request: Option<u32>,
    hedge_delay: Option<f64>,
    total_timeout_s: Option<f64>,
    hedge_budget_pct: Option<f64>,
    retry_budget_pct: Option<f64>,
  ) -> Self {
    RequestProcessingPreference {
      inner: RustRequestProcessingPreference {
        max_concurrent_requests: max_concurrent_requests.unwrap_or(DEFAULT_CONCURRENCY as u32)
          as usize,
        batch_size: batch_size.unwrap_or(DEFAULT_BATCH_SIZE as u32) as usize,
        max_chars_per_request: max_chars_per_request.map(|x| x as usize),
        timeout_s: timeout_s.unwrap_or(DEFAULT_REQUEST_TIMEOUT_S),
        hedge_delay,
        total_timeout_s,
        hedge_budget_pct: hedge_budget_pct.unwrap_or(DEFAULT_HEDGE_BUDGET_PERCENTAGE),
        retry_budget_pct: retry_budget_pct.unwrap_or(DEFAULT_RETRY_BUDGET_PERCENTAGE),
        max_retries: None,
        initial_backoff: None,
      },
    }
  }

  #[napi(getter)]
  pub fn max_concurrent_requests(&self) -> u32 {
    self.inner.max_concurrent_requests as u32
  }

  #[napi(getter)]
  pub fn batch_size(&self) -> u32 {
    self.inner.batch_size as u32
  }

  #[napi(getter)]
  pub fn timeout_s(&self) -> f64 {
    self.inner.timeout_s
  }

  #[napi(getter)]
  pub fn max_chars_per_request(&self) -> Option<u32> {
    self.inner.max_chars_per_request.map(|x| x as u32)
  }

  #[napi(getter)]
  pub fn hedge_delay(&self) -> Option<f64> {
    self.inner.hedge_delay
  }

  #[napi(getter)]
  pub fn total_timeout_s(&self) -> Option<f64> {
    self.inner.total_timeout_s
  }

  #[napi(getter)]
  pub fn hedge_budget_pct(&self) -> f64 {
    self.inner.hedge_budget_pct
  }

  #[napi(getter)]
  pub fn retry_budget_pct(&self) -> f64 {
    self.inner.retry_budget_pct
  }
}

#[napi]
pub struct HttpClientWrapper {
  inner: Arc<HttpClientWrapperRs>,
}

#[napi]
impl HttpClientWrapper {
  #[napi(constructor)]
  pub fn new(http_version: Option<u8>) -> napi::Result<Self> {
    let http_version = http_version.unwrap_or(1);
    let inner = HttpClientWrapperRs::new(http_version).map_err(convert_core_error_to_napi_error)?;
    Ok(Self { inner })
  }
}

#[napi]
pub struct PerformanceClient {
  core_client: PerformanceClientCore,
}

#[napi]
impl PerformanceClient {
  #[napi(constructor)]
  pub fn new(
    base_url: String,
    api_key: Option<String>,
    http_version: Option<u8>,
    client_wrapper: Option<&HttpClientWrapper>,
  ) -> napi::Result<Self> {
    let http_version = http_version.unwrap_or(1);
    let wrapper = client_wrapper.map(|c| Arc::clone(&c.inner));
    let core_client = PerformanceClientCore::new(base_url, api_key, http_version, wrapper)
      .map_err(convert_core_error_to_napi_error)?;

    Ok(Self { core_client })
  }

  #[napi]
  pub fn get_client_wrapper(&self) -> HttpClientWrapper {
    HttpClientWrapper {
      inner: self.core_client.get_client_wrapper(),
    }
  }

  #[napi]
  pub async fn embed(
    &self,
    input: Vec<String>,
    model: String,
    encoding_format: Option<String>,
    dimensions: Option<u32>,
    user: Option<String>,
    preference: Option<&RequestProcessingPreference>,
  ) -> napi::Result<serde_json::Value> {
    if input.is_empty() {
      return Err(create_napi_error("Input list cannot be empty"));
    }

    let pref = preference
      .map(|p| p.inner.clone())
      .unwrap_or_default();

    let result = self
      .core_client
      .process_embeddings_requests(input, model, encoding_format, dimensions, user, &pref)
      .await
      .map_err(convert_core_error_to_napi_error)?;

    let (core_response, batch_durations, headers, total_time) = result;
    let mut response = OpenAIEmbeddingsResponse::from(core_response);
    response.total_time = total_time.as_secs_f64();
    response.individual_request_times = batch_durations
      .into_iter()
      .map(|d: std::time::Duration| d.as_secs_f64())
      .collect();
    response.response_headers = headers;

    serde_json::to_value(response)
      .map_err(|e| create_napi_error(&format!("Serialization error: {}", e)))
  }

  #[napi]
  pub async fn rerank(
    &self,
    query: String,
    texts: Vec<String>,
    raw_scores: Option<bool>,
    model: Option<String>,
    return_text: Option<bool>,
    truncate: Option<bool>,
    truncation_direction: Option<String>,
    preference: Option<&RequestProcessingPreference>,
  ) -> napi::Result<serde_json::Value> {
    if texts.is_empty() {
      return Err(create_napi_error("Texts list cannot be empty"));
    }

    let pref = preference.map(|p| p.inner.clone()).unwrap_or_default();

    let result = self
      .core_client
      .process_rerank_requests(
        query,
        texts,
        raw_scores.unwrap_or(false),
        model,
        return_text.unwrap_or(false),
        truncate.unwrap_or(false),
        truncation_direction.unwrap_or_else(|| "Right".to_string()),
        &pref,
      )
      .await
      .map_err(convert_core_error_to_napi_error)?;

    let (core_response, batch_durations, headers, total_time) = result;
    let mut response = RerankResponse::from(core_response);
    response.total_time = total_time.as_secs_f64();
    response.individual_request_times = batch_durations
      .into_iter()
      .map(|d: std::time::Duration| d.as_secs_f64())
      .collect();
    response.response_headers = headers;

    serde_json::to_value(response)
      .map_err(|e| create_napi_error(&format!("Serialization error: {}", e)))
  }

  #[napi]
  pub async fn classify(
    &self,
    inputs: Vec<String>,
    model: Option<String>,
    raw_scores: Option<bool>,
    truncate: Option<bool>,
    truncation_direction: Option<String>,
    preference: Option<&RequestProcessingPreference>,
  ) -> napi::Result<serde_json::Value> {
    if inputs.is_empty() {
      return Err(create_napi_error("Inputs list cannot be empty"));
    }

    let pref = preference.map(|p| p.inner.clone()).unwrap_or_default();

    let result = self
      .core_client
      .process_classify_requests(
        inputs,
        model,
        raw_scores.unwrap_or(false),
        truncate.unwrap_or(false),
        truncation_direction.unwrap_or_else(|| "Right".to_string()),
        &pref,
      )
      .await
      .map_err(convert_core_error_to_napi_error)?;

    let (core_response, batch_durations, headers, total_time) = result;
    let mut response = ClassificationResponse::from(core_response);
    response.total_time = total_time.as_secs_f64();
    response.individual_request_times = batch_durations
      .into_iter()
      .map(|d: std::time::Duration| d.as_secs_f64())
      .collect();
    response.response_headers = headers;

    serde_json::to_value(response)
      .map_err(|e| create_napi_error(&format!("Serialization error: {}", e)))
  }

  #[napi]
  pub async fn batch_post(
    &self,
    url_path: String,
    payloads: Vec<JsonValue>,
    preference: Option<&RequestProcessingPreference>,
    custom_headers: Option<std::collections::HashMap<String, String>>,
  ) -> napi::Result<serde_json::Value> {
    if payloads.is_empty() {
      return Err(create_napi_error("Payloads list cannot be empty"));
    }

    let pref = preference.map(|p| p.inner.clone()).unwrap_or_default();

    let result = self
      .core_client
      .process_batch_post_requests(url_path, payloads, &pref, custom_headers)
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

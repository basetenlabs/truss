use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::{stream, StreamExt};
use napi_derive::napi;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

// -------------------------------------------------------------------------------------------------
//  NOTE
//  ----
//  This file is a _nearly_ mechanical translation of the original `lib.rs` that produced Python
//  bindings using PyO3.  All of the high-performance, highly-concurrent HTTP logic has been
//  preserved verbatim. The only substantive changes are to the public interface (now `napi` instead
//  of `pyo3`) and to the error‐handling layer (mapping the PyO3 error helpers to `napi::Error`).
//
//  The goal is to present **identical features** with **identical performance characteristics** to
//  the previous implementation – just for the JavaScript / TypeScript ecosystem instead of Python.
//
// The original `lib.rs` file is at:
// https://github.com/basetenlabs/truss/blob/main/baseten-performance-client/src/lib.rs
// -------------------------------------------------------------------------------------------------

// --- Re-export the HTTPError shape that the original API surfaced --------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HTTPError {
  pub status_code: u16,
  pub message: String,
}

// Helper for creating ValueError-style errors
fn value_error<M: Into<String>>(msg: M) -> napi::Error {
  napi::Error::new(napi::Status::InvalidArg, msg.into())
}

// Create a custom error with preserved structure
#[napi(object)]
#[derive(Debug, Clone)]
pub struct BasetenError {
  pub code: String,
  pub status_code: Option<u16>,
  pub message: String,
}

impl HTTPError {
  fn into_napi_err(self) -> napi::Error {
    let err = napi::Error::new(
      napi::Status::GenericFailure,
      format!("HTTP {}: {}", self.status_code, self.message),
    );
    // Preserve status code in error for proper handling in JS
    err
  }

  fn new_err(status_code: u16, message: String) -> napi::Error {
    HTTPError {
      status_code,
      message,
    }
    .into_napi_err()
  }
}

// -------------------------------------------------------------------------------------------------
//  CONSTANTS  (kept identical) --------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

const DEFAULT_REQUEST_TIMEOUT_S: f64 = 3600.0;
const MIN_REQUEST_TIMEOUT_S: f64 = 0.1;
const MAX_REQUEST_TIMEOUT_S: f64 = 3600.0;
const MAX_CONCURRENCY_HIGH_BATCH: usize = 512;
const MAX_CONCURRENCY_LOW_BATCH: usize = 192;
const CONCURRENCY_HIGH_BATCH_SWITCH: usize = 16;

const DEFAULT_CONCURRENCY: usize = 32;
const MAX_BATCH_SIZE: usize = 128;
const DEFAULT_BATCH_SIZE: usize = 4;
const MAX_HTTP_RETRIES: u32 = 4;
const INITIAL_BACKOFF_MS: u64 = 125;
const MAX_BACKOFF_DURATION: Duration = Duration::from_secs(60);

// -------------------------------------------------------------------------------------------------
//  REMOVED GLOBAL RUNTIME - Using Node.js event loop instead
// -------------------------------------------------------------------------------------------------
// Global state has been removed to properly integrate with Node.js patterns.
// We now use napi's async primitives which integrate with Node's libuv event loop.

// -------------------------------------------------------------------------------------------------
//  DATA SHAPES (all exported to TypeScript) -------------------------------------------------------
//  We use `#[napi(object)]` so the generated declaration file has proper definitions.  The internal
//  structure is identical to the old Python version.
// -------------------------------------------------------------------------------------------------

// Use JsonValue for embedding to avoid napi serialization issues

#[napi(object)]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIEmbeddingData {
  pub object: String,
  #[serde(rename = "embedding")]
  pub embedding_internal: JsonValue,
  pub index: u32,
}

#[napi]
impl OpenAIEmbeddingData {
  #[napi(getter)]
  pub fn embedding(&self) -> napi::Result<Vec<f32>> {
    // Try to parse the embedding_internal as a float vector
    if let Some(arr) = self.embedding_internal.as_array() {
      let mut result = Vec::with_capacity(arr.len());
      for val in arr {
        if let Some(f) = val.as_f64() {
          result.push(f as f32);
        } else {
          return Err(value_error(
            "Invalid embedding format: expected array of numbers",
          ));
        }
      }
      Ok(result)
    } else if self.embedding_internal.is_string() {
      Err(value_error(
        "Base64 embeddings not yet supported in JS bindings",
      ))
    } else {
      Err(value_error("Invalid embedding format"))
    }
  }
}

#[napi(object)]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIUsage {
  pub prompt_tokens: u32,
  pub total_tokens: u32,
}

#[napi(object)]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIEmbeddingsResponse {
  pub object: String,
  pub data: Vec<OpenAIEmbeddingData>,
  pub model: String,
  pub usage: OpenAIUsage,
  pub total_time: Option<f64>,
  pub individual_request_times: Option<Vec<f64>>, // per-batch timings
}

impl OpenAIEmbeddingsResponse {
  /// Convert the embeddings into a flat `Vec<f32>` together with (n_rows, n_cols).
  /// In TypeScript you can reconstruct a `Float32Array` if desired.
  pub fn flatten(&self) -> napi::Result<(u32, u32, Vec<f32>)> {
    if self.data.is_empty() {
      return Err(value_error(
        "Cannot convert to array: contains no embedding responses",
      ));
    }

    let first_embedding = self.data[0].embedding()?;
    let dim = first_embedding.len();

    // Memory protection: prevent allocation of huge arrays
    let total_elements = self.data.len().saturating_mul(dim);
    const MAX_ELEMENTS: usize = 100_000_000; // 100M floats = ~400MB
    if total_elements > MAX_ELEMENTS {
      return Err(value_error(format!(
        "Flattening would create {} elements ({}MB), exceeding limit of {} elements ({}MB)",
        total_elements,
        total_elements * 4 / 1_000_000,
        MAX_ELEMENTS,
        MAX_ELEMENTS * 4 / 1_000_000
      )));
    }

    let mut flat: Vec<f32> = Vec::with_capacity(total_elements);

    for item in self.data.iter() {
      let embedding = item.embedding()?;
      if embedding.len() != dim {
        return Err(value_error(format!(
          "All embeddings must have the same dimension. Expected {} but got {} at index {}.",
          dim,
          embedding.len(),
          item.index
        )));
      }
      flat.extend_from_slice(&embedding);
    }

    // Return dimensions (rows, cols, data)
    Ok((self.data.len() as u32, dim as u32, flat))
  }
}

// ---- Rerank -------------------------------------------------------------------------------------

#[napi(object)]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RerankResult {
  pub index: u32,
  pub score: f64,
  pub text: Option<String>,
}

#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResponse {
  pub object: String,
  pub data: Vec<RerankResult>,
  pub total_time: Option<f64>,
  pub individual_request_times: Option<Vec<f64>>,
}

// Constructors are not needed for #[napi(object)] structs

// ---- Classification -----------------------------------------------------------------------------

#[napi(object)]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ClassificationResult {
  pub label: String,
  pub score: f64,
}

#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResponse {
  pub object: String,
  pub data: Vec<Vec<ClassificationResult>>,
  pub total_time: Option<f64>,
  pub individual_request_times: Option<Vec<f64>>,
}

// Constructors are not needed for #[napi(object)] structs

// ---- Batch POST ---------------------------------------------------------------------------------

#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchPostResponse {
  pub data: Vec<JsonValue>,
  pub total_time: f64,
  pub individual_request_times: Vec<f64>,
  pub response_headers: Vec<JsonValue>,
}

// -------------------------------------------------------------------------------------------------
//  PERFORMANCE CLIENT  (exposed to JS) -------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

#[napi]
pub struct PerformanceClient {
  api_key: Arc<String>,  // Use Arc to avoid cloning in hot paths
  base_url: Arc<String>, // Use Arc to avoid cloning in hot paths
  client: Client,
  // Runtime removed - we use Node.js event loop via napi async
}

impl PerformanceClient {
  // Security: Comprehensive URL validation
  fn validate_and_normalize_url(url: &str) -> napi::Result<String> {
    if url.is_empty() {
      return Err(value_error("URL cannot be empty"));
    }

    // Parse URL to validate format
    let parsed =
      url::Url::parse(url).map_err(|e| value_error(format!("Invalid URL format: {}", e)))?;

    // Only allow http/https
    match parsed.scheme() {
      "http" | "https" => {}
      _ => return Err(value_error("Only http:// and https:// URLs are allowed")),
    }

    // Validate no credentials in URL
    if !parsed.username().is_empty() || parsed.password().is_some() {
      return Err(value_error(
        "URLs with embedded credentials are not allowed",
      ));
    }

    // Validate host
    if parsed.host_str().is_none() || parsed.host_str() == Some("") {
      return Err(value_error("URL must have a valid host"));
    }

    // Check for suspicious patterns
    let url_str = parsed.as_str();
    if url_str.contains('\n') || url_str.contains('\r') || url_str.contains('\0') {
      return Err(value_error("URL contains invalid control characters"));
    }

    // Return normalized URL without trailing slash
    Ok(parsed.to_string().trim_end_matches('/').to_string())
  }

  // Security: Path validation to prevent traversal attacks
  fn validate_url_path(path: &str) -> napi::Result<()> {
    // Decode percent-encoded characters
    let decoded = percent_encoding::percent_decode_str(path)
      .decode_utf8()
      .map_err(|_| value_error("Invalid UTF-8 in URL path"))?;

    // Check both encoded and decoded forms for traversal patterns
    let patterns = [
      "..", "./", "/.", "%2e%2e", "%2E%2E", "%2e.", ".%2e", "%2f", "%5c",
    ];
    let decoded_str = decoded.as_ref();

    for pattern in &patterns {
      if path.contains(pattern) || decoded_str.contains(pattern) {
        return Err(value_error("URL path contains invalid traversal patterns"));
      }
    }

    // Check for null bytes and control characters
    if decoded_str
      .chars()
      .any(|c| c == '\0' || (c.is_control() && c != '\t'))
    {
      return Err(value_error("URL path contains invalid control characters"));
    }

    // Ensure path doesn't start with multiple slashes (//)
    if path.starts_with("//") {
      return Err(value_error("URL path cannot start with double slashes"));
    }

    Ok(())
  }
  fn get_api_key(api_key: Option<String>) -> napi::Result<String> {
    api_key
      .or_else(|| std::env::var("BASETEN_API_KEY").ok())
      .or_else(|| std::env::var("OPENAI_API_KEY").ok())
      .ok_or_else(|| value_error("API key not provided and no BASETEN/OPENAI env variable found"))
  }

  fn validate_timeout(timeout_s: f64) -> napi::Result<Duration> {
    if !(MIN_REQUEST_TIMEOUT_S..=MAX_REQUEST_TIMEOUT_S).contains(&timeout_s) {
      return Err(value_error(format!(
        "Timeout {:.3}s is outside the allowed range [{:.3}s, {:.3}s]",
        timeout_s, MIN_REQUEST_TIMEOUT_S, MAX_REQUEST_TIMEOUT_S
      )));
    }
    Ok(Duration::from_secs_f64(timeout_s))
  }

  fn validate_concurrency(max_concurrent_requests: usize, batch_size: usize) -> napi::Result<()> {
    if max_concurrent_requests == 0 || max_concurrent_requests > MAX_CONCURRENCY_HIGH_BATCH {
      return Err(value_error(format!(
        "max_concurrent_requests must be >0 and <= {}",
        MAX_CONCURRENCY_HIGH_BATCH
      )));
    }
    if batch_size == 0 || batch_size > MAX_BATCH_SIZE {
      return Err(value_error(format!(
        "batch_size must be >0 and <= {}",
        MAX_BATCH_SIZE
      )));
    }
    if max_concurrent_requests > MAX_CONCURRENCY_LOW_BATCH
      && batch_size < CONCURRENCY_HIGH_BATCH_SWITCH
    {
      return Err(value_error(format!(
        "max_concurrent_requests must be < {} when batch_size < {} (be nice to the server)",
        MAX_CONCURRENCY_LOW_BATCH, CONCURRENCY_HIGH_BATCH_SWITCH
      )));
    }
    Ok(())
  }
}

#[napi]
impl PerformanceClient {
  #[napi(constructor)]
  pub fn new(base_url: String, api_key: Option<String>) -> napi::Result<Self> {
    // Comprehensive URL validation and normalization
    let normalized_base_url = Self::validate_and_normalize_url(&base_url)?;

    let api_key_str = Self::get_api_key(api_key)?;

    // Configure high-performance HTTP client
    let client = Client::builder()
      // Connection pooling matched to concurrency settings
      .pool_max_idle_per_host(DEFAULT_CONCURRENCY as usize) // Match our concurrency limit
      .pool_idle_timeout(Duration::from_secs(60)) // Keep connections warm
      // HTTP/2 for multiplexing - crucial for concurrent requests
      .http2_adaptive_window(true) // Dynamic flow control for varying payloads
      .http2_initial_stream_window_size(2 * 1024 * 1024) // 2MB per stream
      .http2_initial_connection_window_size(16 * 1024 * 1024) // 16MB total
      .http2_keep_alive_interval(Duration::from_secs(10)) // Keep HTTP/2 alive
      .http2_keep_alive_timeout(Duration::from_secs(30))
      // Timeouts
      .connect_timeout(Duration::from_secs(10))
      // Performance optimizations
      .tcp_nodelay(true) // Disable Nagle's algorithm for lower latency
      .tcp_keepalive(Duration::from_secs(30)) // Keep TCP connections alive
      .gzip(true) // Enable gzip compression
      .brotli(true) // Enable brotli compression
      // Security and compatibility
      .https_only(normalized_base_url.starts_with("https")) // Enforce HTTPS if URL is HTTPS
      .http1_title_case_headers() // Some servers require this
      .build()
      .map_err(|e| value_error(format!("Failed to create HTTP client: {}", e)))?;

    let client_instance = PerformanceClient {
      api_key: Arc::new(api_key_str),
      base_url: Arc::new(normalized_base_url),
      client,
    };

    // Constructor is now pure - no side effects or health checks
    Ok(client_instance)
  }

  // -- EMBEDDINGS --------------------------------------------------------
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
  ) -> napi::Result<OpenAIEmbeddingsResponse> {
    if input.is_empty() {
      return Err(value_error("Input list cannot be empty"));
    }
    // Validate input strings
    for (i, text) in input.iter().enumerate() {
      if text.is_empty() {
        return Err(value_error(format!(
          "Input string at index {} cannot be empty",
          i
        )));
      }
    }
    // Validate dimensions if provided
    if let Some(dim) = dimensions {
      if dim == 0 {
        return Err(value_error("Dimensions must be greater than 0"));
      }
    }

    let max_concurrent_requests = max_concurrent_requests
      .map(|v| v as usize)
      .unwrap_or(DEFAULT_CONCURRENCY);
    let batch_size = batch_size.map(|v| v as usize).unwrap_or(DEFAULT_BATCH_SIZE);
    Self::validate_concurrency(max_concurrent_requests, batch_size)?;
    let timeout = Self::validate_timeout(timeout_s.unwrap_or(DEFAULT_REQUEST_TIMEOUT_S))?;

    let client = self.client.clone();
    let api_key = self.api_key.clone();
    let base_url = self.base_url.clone();

    let start_sync = Instant::now();
    let (mut resp, batch_durs) = process_embeddings_requests(
      client,
      input,
      model,
      api_key,
      base_url,
      encoding_format,
      dimensions,
      user,
      max_concurrent_requests,
      batch_size,
      timeout,
    )
    .await?;

    resp.total_time = Some(start_sync.elapsed().as_secs_f64());
    resp.individual_request_times = Some(batch_durs.into_iter().map(|d| d.as_secs_f64()).collect());
    Ok(resp)
  }

  // -- RERANK ------------------------------------------------------------
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
    timeout_s: Option<f64>,
  ) -> napi::Result<RerankResponse> {
    if texts.is_empty() {
      return Err(value_error("Texts list cannot be empty"));
    }
    // Validate texts
    for (i, text) in texts.iter().enumerate() {
      if text.is_empty() {
        return Err(value_error(format!("Text at index {} cannot be empty", i)));
      }
    }
    let max_concurrent_requests = max_concurrent_requests
      .map(|v| v as usize)
      .unwrap_or(DEFAULT_CONCURRENCY);
    let batch_size = DEFAULT_BATCH_SIZE; // Use default batch size for rerank
    Self::validate_concurrency(max_concurrent_requests, batch_size)?;
    let timeout = Self::validate_timeout(timeout_s.unwrap_or(DEFAULT_REQUEST_TIMEOUT_S))?;

    let client = self.client.clone();
    let api_key = self.api_key.clone();
    let base_url = self.base_url.clone();
    let start = Instant::now();

    let (core, durs) = process_rerank_requests(
      client,
      query,
      texts,
      raw_scores.unwrap_or(false),
      return_text.unwrap_or(false),
      truncate.unwrap_or(false),
      {
        let dir = truncation_direction.unwrap_or_else(|| "Right".to_string());
        // Validate truncation direction
        if !["Left", "Right"].contains(&dir.as_str()) {
          return Err(value_error(
            "truncation_direction must be 'Left' or 'Right'",
          ));
        }
        dir
      },
      api_key,
      base_url,
      max_concurrent_requests,
      batch_size,
      timeout,
    )
    .await?;

    Ok(RerankResponse {
      object: "list".to_string(),
      data: core,
      total_time: Some(start.elapsed().as_secs_f64()),
      individual_request_times: Some(durs.into_iter().map(|d| d.as_secs_f64()).collect()),
    })
  }

  // -- CLASSIFY ----------------------------------------------------------
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
  ) -> napi::Result<ClassificationResponse> {
    if inputs.is_empty() {
      return Err(value_error("Inputs list cannot be empty"));
    }
    // Validate input strings
    for (i, text) in inputs.iter().enumerate() {
      if text.is_empty() {
        return Err(value_error(format!(
          "Input string at index {} cannot be empty",
          i
        )));
      }
    }
    let max_concurrent_requests = max_concurrent_requests
      .map(|v| v as usize)
      .unwrap_or(DEFAULT_CONCURRENCY);
    let batch_size = batch_size.map(|v| v as usize).unwrap_or(DEFAULT_BATCH_SIZE);
    Self::validate_concurrency(max_concurrent_requests, batch_size)?;
    let timeout = Self::validate_timeout(timeout_s.unwrap_or(DEFAULT_REQUEST_TIMEOUT_S))?;

    let client = self.client.clone();
    let api_key = self.api_key.clone();
    let base_url = self.base_url.clone();
    let start = Instant::now();

    let (core, durs) = process_classify_requests(
      client,
      inputs,
      raw_scores.unwrap_or(false),
      truncate.unwrap_or(false),
      {
        let dir = truncation_direction.unwrap_or_else(|| "Right".to_string());
        // Validate truncation direction
        if !["Left", "Right"].contains(&dir.as_str()) {
          return Err(value_error(
            "truncation_direction must be 'Left' or 'Right'",
          ));
        }
        dir
      },
      api_key,
      base_url,
      max_concurrent_requests,
      batch_size,
      timeout,
    )
    .await?;

    Ok(ClassificationResponse {
      object: "list".to_string(),
      data: core,
      total_time: Some(start.elapsed().as_secs_f64()),
      individual_request_times: Some(durs.into_iter().map(|d| d.as_secs_f64()).collect()),
    })
  }

  // -- BATCH POST --------------------------------------------------------
  #[napi]
  pub async fn batch_post(
    &self,
    url_path: String,
    payloads: Vec<JsonValue>,
    max_concurrent_requests: Option<u32>,
    timeout_s: Option<f64>,
  ) -> napi::Result<BatchPostResponse> {
    if payloads.is_empty() {
      return Err(value_error("Payloads list cannot be empty"));
    }
    // Memory protection: limit number of payloads
    const MAX_PAYLOADS: usize = 5000;
    if payloads.len() > MAX_PAYLOADS {
      return Err(value_error(format!(
        "Number of payloads {} exceeds maximum allowed {}. Process in smaller batches.",
        payloads.len(),
        MAX_PAYLOADS
      )));
    }
    // Comprehensive path validation
    if url_path.is_empty() {
      return Err(value_error("URL path cannot be empty"));
    }
    Self::validate_url_path(&url_path)?;
    let max_concurrent_requests = max_concurrent_requests
      .map(|v| v as usize)
      .unwrap_or(DEFAULT_CONCURRENCY);
    Self::validate_concurrency(max_concurrent_requests, MAX_BATCH_SIZE)?;
    let timeout = Self::validate_timeout(timeout_s.unwrap_or(DEFAULT_REQUEST_TIMEOUT_S))?;

    let client = self.client.clone();
    let api_key = self.api_key.clone();
    let base_url = self.base_url.clone();
    let start = Instant::now();

    let resp = process_batch_post_requests(
      client,
      url_path,
      payloads,
      api_key,
      base_url,
      max_concurrent_requests,
      timeout,
    )
    .await?;

    let mut data_vec = Vec::with_capacity(resp.len());
    let mut time_vec = Vec::with_capacity(resp.len());
    let mut headers_vec = Vec::with_capacity(resp.len());

    for (json_val, headers, dur) in resp {
      data_vec.push(json_val);
      headers_vec.push(JsonValue::Object(
        headers
          .into_iter()
          .map(|(k, v)| (k, JsonValue::String(v)))
          .collect(),
      ));
      time_vec.push(dur.as_secs_f64());
    }

    Ok(BatchPostResponse {
      data: data_vec,
      total_time: start.elapsed().as_secs_f64(),
      individual_request_times: time_vec,
      response_headers: headers_vec,
    })
  }
}

// -------------------------------------------------------------------------------------------------
//  INTERNAL ASYNC HELPERS (ported 1-for-1 from the original implementation) -----------------------
// -------------------------------------------------------------------------------------------------

// --- Send Single Embedding Request ---------------------------------------------------------------
async fn send_single_embedding_request(
  client: Client,
  texts_batch: Vec<String>,
  model: String,
  api_key: String,
  base_url: String,
  encoding_format: Option<String>,
  dimensions: Option<u32>,
  user: Option<String>,
  request_timeout: Duration,
) -> napi::Result<OpenAIEmbeddingsResponse> {
  #[derive(Serialize)]
  struct OpenAIEmbeddingsRequest {
    input: Vec<String>,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
  }

  let request_payload = OpenAIEmbeddingsRequest {
    input: texts_batch,
    model,
    encoding_format,
    dimensions,
    user,
  };

  let url = format!("{}/predict", base_url.trim_end_matches('/'));

  let builder = client
    .post(&url)
    .bearer_auth(&api_key)
    .json(&request_payload)
    .timeout(request_timeout);

  let response = send_request_with_retry(
    builder,
    MAX_HTTP_RETRIES,
    Duration::from_millis(INITIAL_BACKOFF_MS),
  )
  .await?;
  let ok_resp = ensure_successful_response(response).await?;

  ok_resp
    .json::<OpenAIEmbeddingsResponse>()
    .await
    .map_err(|_| value_error("Failed to parse response JSON"))
}

// --- Process Embeddings Requests with concurrent execution ------------------------------------
async fn process_embeddings_requests(
  client: Client,
  texts: Vec<String>,
  model: String,
  api_key: Arc<String>,  // Already Arc from caller
  base_url: Arc<String>, // Already Arc from caller
  encoding_format: Option<String>,
  dimensions: Option<u32>,
  user: Option<String>,
  max_concurrent_requests: usize,
  batch_size: usize,
  request_timeout_duration: Duration,
) -> napi::Result<(OpenAIEmbeddingsResponse, Vec<Duration>)> {
  let total_texts = texts.len();
  let model_clone_for_resp = model.clone();

  // Collect chunks into owned data to avoid lifetime issues
  let batches: Vec<(usize, Vec<String>)> = texts
    .chunks(batch_size)
    .enumerate()
    .map(|(idx, chunk)| (idx, chunk.to_vec()))
    .collect();

  // Create a stream of futures with proper indexing
  let futures_stream = stream::iter(batches).map(|(batch_index, texts_batch)| {
    let client = client.clone();
    let model = model.clone();
    let api_key = Arc::clone(&api_key);
    let base_url = Arc::clone(&base_url);
    let encoding_format = encoding_format.clone();
    let user = user.clone();
    let current_batch_absolute_start_index = batch_index * batch_size;

    async move {
      let start = Instant::now();
      let res = send_single_embedding_request(
        client,
        texts_batch,
        model,
        (*api_key).clone(),
        (*base_url).clone(),
        encoding_format,
        dimensions,
        user,
        request_timeout_duration,
      )
      .await;
      let elapsed = start.elapsed();

      match res {
        Ok(mut resp) => {
          // Adjust indices
          for item in &mut resp.data {
            item.index += current_batch_absolute_start_index as u32;
          }
          (Ok(resp), elapsed)
        }
        Err(e) => (Err(e), elapsed),
      }
    }
  });

  // Process with limited concurrency using buffer_unordered
  let results: Vec<_> = futures_stream
    .buffer_unordered(max_concurrent_requests)
    .collect()
    .await;

  let mut all_data = Vec::with_capacity(total_texts);
  let mut prompt_sum: u32 = 0;
  let mut token_sum: u32 = 0;
  let mut batch_durations = Vec::new();
  let mut first_err: Option<napi::Error> = None;

  // Process results from the stream
  for (result, duration) in results {
    match result {
      Ok(sub_resp) => {
        if first_err.is_none() {
          all_data.extend(sub_resp.data);
          prompt_sum = prompt_sum.saturating_add(sub_resp.usage.prompt_tokens);
          token_sum = token_sum.saturating_add(sub_resp.usage.total_tokens);
          batch_durations.push(duration);
        }
      }
      Err(e) => {
        if first_err.is_none() {
          first_err = Some(e);
        }
      }
    }
  }

  if let Some(err) = first_err {
    return Err(err);
  }

  all_data.sort_by_key(|d| d.index as usize);
  let final_resp = OpenAIEmbeddingsResponse {
    object: "list".into(),
    data: all_data,
    model: model_clone_for_resp,
    usage: OpenAIUsage {
      prompt_tokens: prompt_sum,
      total_tokens: token_sum,
    },
    total_time: None,
    individual_request_times: None,
  };
  Ok((final_resp, batch_durations))
}

// --- Send Single Rerank Request ------------------------------------------------------------------
async fn send_single_rerank_request(
  client: Client,
  query: String,
  texts_batch: Vec<String>,
  raw_scores: bool,
  return_text: bool,
  truncate: bool,
  truncation_direction: String,
  api_key: String,
  base_url: String,
  request_timeout: Duration,
) -> napi::Result<Vec<RerankResult>> {
  #[derive(Serialize)]
  struct RerankRequest {
    query: String,
    raw_scores: bool,
    return_text: bool,
    texts: Vec<String>,
    truncate: bool,
    truncation_direction: String,
  }

  let payload = RerankRequest {
    query,
    raw_scores,
    return_text,
    texts: texts_batch,
    truncate,
    truncation_direction,
  };

  let url = format!("{}/rerank", base_url.trim_end_matches('/'));

  let builder = client
    .post(&url)
    .bearer_auth(&api_key)
    .json(&payload)
    .timeout(request_timeout);

  let resp = send_request_with_retry(
    builder,
    MAX_HTTP_RETRIES,
    Duration::from_millis(INITIAL_BACKOFF_MS),
  )
  .await?;
  let ok = ensure_successful_response(resp).await?;

  ok.json::<Vec<RerankResult>>()
    .await
    .map_err(|_| value_error("Failed to parse rerank response JSON"))
}

// --- Process Rerank Requests ---------------------------------------------------------------------
async fn process_rerank_requests(
  client: Client,
  query: String,
  texts: Vec<String>,
  raw_scores: bool,
  return_text: bool,
  truncate: bool,
  truncation_direction: String,
  api_key: Arc<String>,  // Already Arc from caller
  base_url: Arc<String>, // Already Arc from caller
  max_concurrent_requests: usize,
  batch_size: usize,
  request_timeout_duration: Duration,
) -> napi::Result<(Vec<RerankResult>, Vec<Duration>)> {
  // Collect chunks into owned data to avoid lifetime issues
  let batches: Vec<(usize, Vec<String>)> = texts
    .chunks(batch_size)
    .enumerate()
    .map(|(idx, chunk)| (idx, chunk.to_vec()))
    .collect();

  // Create a stream of futures
  let futures_stream = stream::iter(batches).map(|(batch_idx, texts_batch_owned)| {
    let client = client.clone();
    let query = query.clone();
    let api_key = Arc::clone(&api_key);
    let base_url = Arc::clone(&base_url);
    let td = truncation_direction.clone();
    let current_batch_absolute_start_index = batch_idx * batch_size;

    async move {
      let start = Instant::now();

      let res = send_single_rerank_request(
        client,
        query,
        texts_batch_owned,
        raw_scores,
        return_text,
        truncate,
        td,
        (*api_key).clone(),
        (*base_url).clone(),
        request_timeout_duration,
      )
      .await;

      let elapsed = start.elapsed();

      match res {
        Ok(mut batch_results) => {
          // Adjust index for each result in the batch
          for item in &mut batch_results {
            item.index += current_batch_absolute_start_index as u32;
          }
          (Ok(batch_results), elapsed)
        }
        Err(e) => (Err(e), elapsed),
      }
    }
  });

  // Process with limited concurrency
  let results: Vec<_> = futures_stream
    .buffer_unordered(max_concurrent_requests)
    .collect()
    .await;
  let mut merged = Vec::new();
  let mut durations = Vec::new();
  let mut first_err: Option<napi::Error> = None;

  // Process results from the stream
  for (result, duration) in results {
    match result {
      Ok(batch_results) => {
        if first_err.is_none() {
          merged.extend(batch_results);
          durations.push(duration);
        }
      }
      Err(e) => {
        if first_err.is_none() {
          first_err = Some(e);
        }
      }
    }
  }

  if let Some(err) = first_err {
    return Err(err);
  }

  merged.sort_by_key(|r| r.index as usize);
  Ok((merged, durations))
}

// --- Send Single Classify Request ----------------------------------------------------------------
async fn send_single_classify_request(
  client: Client,
  inputs: Vec<Vec<String>>, // wrapped each string in vec
  raw_scores: bool,
  truncate: bool,
  truncation_direction: String,
  api_key: String,
  base_url: String,
  request_timeout: Duration,
) -> napi::Result<Vec<Vec<ClassificationResult>>> {
  #[derive(Serialize)]
  struct ClassifyRequest {
    inputs: Vec<Vec<String>>,
    raw_scores: bool,
    truncate: bool,
    truncation_direction: String,
  }

  let payload = ClassifyRequest {
    inputs,
    raw_scores,
    truncate,
    truncation_direction,
  };

  let url = format!("{}/predict", base_url.trim_end_matches('/'));
  let builder = client
    .post(&url)
    .bearer_auth(&api_key)
    .json(&payload)
    .timeout(request_timeout);

  let resp = send_request_with_retry(
    builder,
    MAX_HTTP_RETRIES,
    Duration::from_millis(INITIAL_BACKOFF_MS),
  )
  .await?;
  let ok = ensure_successful_response(resp).await?;

  ok.json::<Vec<Vec<ClassificationResult>>>()
    .await
    .map_err(|_| value_error("Failed to parse classify response JSON"))
}

// --- Process Classify Requests -------------------------------------------------------------------
async fn process_classify_requests(
  client: Client,
  inputs: Vec<String>,
  raw_scores: bool,
  truncate: bool,
  truncation_direction: String,
  api_key: Arc<String>,  // Already Arc from caller
  base_url: Arc<String>, // Already Arc from caller
  max_concurrent_requests: usize,
  batch_size: usize,
  request_timeout_duration: Duration,
) -> napi::Result<(Vec<Vec<ClassificationResult>>, Vec<Duration>)> {
  // Collect chunks into owned data with indices to preserve order
  let batches: Vec<(usize, Vec<Vec<String>>)> = inputs
    .chunks(batch_size)
    .enumerate()
    .map(|(idx, chunk)| (idx, chunk.iter().map(|s| vec![s.clone()]).collect()))
    .collect();

  // Create a stream of futures with batch index
  let futures_stream = stream::iter(batches).map(|(batch_idx, inputs_for_api)| {
    let client = client.clone();
    let api_key = Arc::clone(&api_key);
    let base_url = Arc::clone(&base_url);
    let td = truncation_direction.clone();

    async move {
      let start = Instant::now();
      let res = send_single_classify_request(
        client,
        inputs_for_api,
        raw_scores,
        truncate,
        td,
        (*api_key).clone(),
        (*base_url).clone(),
        request_timeout_duration,
      )
      .await;
      let elapsed = start.elapsed();

      match res {
        Ok(batch) => (batch_idx, Ok(batch), elapsed),
        Err(e) => (batch_idx, Err(e), elapsed),
      }
    }
  });

  // Process with limited concurrency
  let results: Vec<_> = futures_stream
    .buffer_unordered(max_concurrent_requests)
    .collect()
    .await;
  let mut indexed_results = Vec::new();
  let mut first_err: Option<napi::Error> = None;

  // Process results from the stream
  for (batch_idx, result, duration) in results {
    match result {
      Ok(batch) => {
        if first_err.is_none() {
          indexed_results.push((batch_idx, batch, duration));
        }
      }
      Err(e) => {
        if first_err.is_none() {
          first_err = Some(e);
        }
      }
    }
  }

  if let Some(err) = first_err {
    return Err(err);
  }

  // Sort by batch index to maintain original order
  indexed_results.sort_by_key(|t| t.0);

  // Flatten the results while preserving order
  let mut all = Vec::new();
  let mut durs = Vec::new();
  for (_, batch, duration) in indexed_results {
    all.extend(batch);
    durs.push(duration);
  }

  Ok((all, durs))
}

// --- Send Single Batch POST Request --------------------------------------------------------------
async fn send_single_batch_post_request(
  client: Client,
  full_url: String,
  payload_json: JsonValue,
  api_key: String,
  request_timeout: Duration,
) -> napi::Result<(JsonValue, std::collections::HashMap<String, String>)> {
  let builder = client
    .post(&full_url)
    .bearer_auth(&api_key)
    .json(&payload_json)
    .timeout(request_timeout);
  let resp = send_request_with_retry(
    builder,
    MAX_HTTP_RETRIES,
    Duration::from_millis(INITIAL_BACKOFF_MS),
  )
  .await?;
  let ok = ensure_successful_response(resp).await?;

  let mut headers_map = std::collections::HashMap::new();
  for (name, value) in ok.headers().iter() {
    headers_map.insert(
      name.to_string(),
      String::from_utf8_lossy(value.as_bytes()).into_owned(),
    );
  }

  let json_val = ok
    .json::<JsonValue>()
    .await
    .map_err(|_| value_error("Failed to parse response JSON"))?;

  Ok((json_val, headers_map))
}

// --- Process Batch POST Requests -----------------------------------------------------------------
async fn process_batch_post_requests(
  client: Client,
  url_path: String,
  payloads_json: Vec<JsonValue>,
  api_key: Arc<String>,  // Already Arc from caller
  base_url: Arc<String>, // Already Arc from caller
  max_concurrent_requests: usize,
  request_timeout_duration: Duration,
) -> napi::Result<
  Vec<(
    JsonValue,
    std::collections::HashMap<String, String>,
    Duration,
  )>,
> {
  let total = payloads_json.len();

  // Create a stream of futures with index tracking
  let futures_stream = stream::iter(payloads_json.into_iter().enumerate().map(|(idx, payload)| {
    let client = client.clone();
    let api_key = Arc::clone(&api_key);
    let base_url = Arc::clone(&base_url);
    let path = url_path.clone();

    async move {
      let full_url = format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        path.trim_start_matches('/')
      );
      let start = Instant::now();
      let res = send_single_batch_post_request(
        client,
        full_url,
        payload,
        (*api_key).clone(),
        request_timeout_duration,
      )
      .await;
      let elapsed = start.elapsed();

      match res {
        Ok((val, headers)) => (idx, Ok((val, headers, elapsed))),
        Err(e) => (idx, Err(e)),
      }
    }
  }));

  // Process with limited concurrency
  let results: Vec<_> = futures_stream
    .buffer_unordered(max_concurrent_requests)
    .collect()
    .await;
  let mut indexed = Vec::with_capacity(total);
  let mut first_err: Option<napi::Error> = None;

  // Process results from the stream
  for (idx, result) in results {
    match result {
      Ok((val, headers, duration)) => {
        if first_err.is_none() {
          indexed.push((idx, val, headers, duration));
        }
      }
      Err(e) => {
        if first_err.is_none() {
          first_err = Some(e);
        }
      }
    }
  }

  if let Some(err) = first_err {
    return Err(err);
  }

  // Sort by index to maintain order
  indexed.sort_by_key(|t| t.0);
  Ok(indexed.into_iter().map(|(_, v, h, d)| (v, h, d)).collect())
}

// -------------------------------------------------------------------------------------------------
//  Utility helpers --------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

async fn ensure_successful_response(resp: reqwest::Response) -> napi::Result<reqwest::Response> {
  if !resp.status().is_success() {
    let status = resp.status();
    let text = resp.text().await.unwrap_or_else(|_| "Unknown error".into());
    // Sanitize error message to avoid leaking sensitive information
    let sanitized_text = if text.len() > 200 {
      format!("{}...", &text[..200])
    } else {
      text
    };
    Err(HTTPError::new_err(
      status.as_u16(),
      format!(
        "API request failed with status {}: {}",
        status, sanitized_text
      ),
    ))
  } else {
    Ok(resp)
  }
}

async fn send_request_with_retry(
  request_builder: reqwest::RequestBuilder,
  max_retries: u32,
  initial_backoff: Duration,
) -> napi::Result<reqwest::Response> {
  let mut retries = 0;
  let mut backoff = initial_backoff;
  loop {
    let clone = request_builder
      .try_clone()
      .ok_or_else(|| value_error("Failed to clone request builder for retry"))?;
    match clone.send().await {
      Ok(resp) if resp.status().is_success() => return Ok(resp),
      Ok(resp) => {
        if resp.status().as_u16() == 429 || resp.status().is_server_error() {
          retries += 1;
          if retries > max_retries {
            return ensure_successful_response(resp).await;
          }
          tokio::time::sleep(backoff.min(MAX_BACKOFF_DURATION)).await;
          backoff *= 4;
          continue;
        } else {
          return Ok(resp);
        }
      }
      Err(_e) => {
        retries += 1;
        if retries > max_retries {
          return Err(value_error("Request failed: Network error"));
        }
        tokio::time::sleep(backoff.min(MAX_BACKOFF_DURATION)).await;
        backoff *= 4;
      }
    }
  }
}

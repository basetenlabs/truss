// implements client for openai embeddings
// as python api that fans out to multiple requests
use futures::future::join_all;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use once_cell::sync::Lazy; // Import Lazy
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_async_runtimes;
use pythonize::{depythonize, pythonize};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
// For handling untyped JSON
use std::collections::HashMap; // Add this
use std::sync::atomic::{AtomicBool, Ordering}; // Add this
use std::sync::Arc;
use std::time::{Duration, Instant}; // Ensure Instant is imported
use std::vec;
use tokio::runtime::Runtime;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinError;

// --- Constants ---
const DEFAULT_REQUEST_TIMEOUT_S: f64 = 3600.0;
const MIN_REQUEST_TIMEOUT_S: f64 = 0.1;
const MAX_REQUEST_TIMEOUT_S: f64 = 3600.0;
const MAX_CONCURRENCY_HIGH_BATCH: usize = 512;
const MAX_CONCURRENCY_LOW_BATCH: usize = 192;
const CONCURRENCY_HIGH_BATCH_SWITCH: usize = 16;
const DEFAULT_CONCURRENCY: usize = 32;
const MAX_BATCH_SIZE: usize = 128;
const DEFAULT_BATCH_SIZE: usize = 16;
const MAX_HTTP_RETRIES: u32 = 4; // Max number of retries for HTTP 429 or network errors
const INITIAL_BACKOFF_MS: u64 = 125; // Initial backoff in milliseconds
const MAX_BACKOFF_DURATION: Duration = Duration::from_secs(60); // Max backoff duration
const WARNING_SLOW_PROVIDERS: [&str; 3] = ["fireworks.ai", "together.ai", "modal.com"]; // Providers that are known to be slow with this client

// --- Global Tokio Runtime ---
static CTRL_C_RECEIVED: AtomicBool = AtomicBool::new(false); // New global flag
                                                             // Add this constant
const CANCELLATION_ERROR_MESSAGE_DETAIL: &str = "Operation cancelled due to a previous error";
const CTRL_C_ERROR_MESSAGE_DETAIL: &str = "Operation cancelled by Ctrl+C"; // New constant for Ctrl+C

static GLOBAL_RUNTIME: Lazy<Arc<Runtime>> = Lazy::new(|| {
    let runtime = Arc::new(Runtime::new().expect("Failed to create global Tokio runtime"));
    let runtime_clone_for_signal = Arc::clone(&runtime);
    // Spawn a task to listen for Ctrl+C
    runtime_clone_for_signal.spawn(async {
        if tokio::signal::ctrl_c().await.is_ok() {
            CTRL_C_RECEIVED.store(true, Ordering::SeqCst);
        }
    });
    runtime
});

pyo3::import_exception!(requests, HTTPError);

// --- OpenAI Compatible Structures ---
#[derive(Serialize, Debug, Clone)]
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

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
enum EmbeddingVariant {
    Base64(String),
    FloatVector(Vec<f32>),
}

#[derive(Deserialize, Debug, Clone)]
#[pyclass]
struct OpenAIEmbeddingData {
    #[pyo3(get)]
    object: String,
    #[serde(rename = "embedding")]
    embedding_internal: EmbeddingVariant,
    #[pyo3(get)]
    index: usize,
}

#[pymethods]
impl OpenAIEmbeddingData {
    #[getter]
    fn embedding(&self, py: Python) -> PyObject {
        match &self.embedding_internal {
            EmbeddingVariant::Base64(s) => s.to_object(py),
            EmbeddingVariant::FloatVector(v) => v.to_object(py),
        }
    }
}

#[derive(Deserialize, Debug, Clone)]
#[pyclass]
struct OpenAIUsage {
    #[pyo3(get)]
    prompt_tokens: u32,
    #[pyo3(get)]
    total_tokens: u32,
}

#[derive(Deserialize, Debug, Clone)]
#[pyclass(get_all)]
struct OpenAIEmbeddingsResponse {
    // Ensure struct is public for field access
    object: String,
    data: Vec<OpenAIEmbeddingData>,
    model: String,
    usage: OpenAIUsage,
    total_time: Option<f64>,
    individual_request_times: Option<Vec<f64>>, // Renamed from individual_request_times
}

#[pymethods]
impl OpenAIEmbeddingsResponse {
    /// Converts the embeddings data into a 2D NumPy array.
    ///
    /// Each row in the array corresponds to an embedding. The data type of the array
    /// will be float32. The shape will be (number_of_embeddings, embedding_dimension).
    ///
    /// Returns:
    ///     numpy.ndarray: A 2D NumPy array of f32.
    ///
    /// Raises:
    ///     PyValueError: If any embedding is not a float vector (e.g., base64 encoded),
    ///                   if embeddings have inconsistent dimensions, or if data is empty
    ///                   and an array cannot be formed (though empty data returns a (0,0) array).
    ///     ImportError: (At runtime from Python) If NumPy is not installed in the Python environment.
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if self.data.is_empty() {
            // error if empty data
            return Err(PyValueError::new_err(
                "Cannot convert to array: contains no embedding responses",
            ));
        }

        let num_embeddings = self.data.len();
        let mut embedding_dim_opt: Option<usize> = None;
        let mut flat_data: Vec<f32> = Vec::new(); // Reserve capacity later if dim is known

        for (idx, item_data) in self.data.iter().enumerate() {
            match &item_data.embedding_internal {
                EmbeddingVariant::FloatVector(v) => {
                    if idx == 0 {
                        // Determine dimension from the first embedding
                        let dim = v.len();
                        embedding_dim_opt = Some(dim);
                        // Pre-allocate assuming all vectors have this dimension.
                        // Only reserve if dim > 0 to avoid issues with 0 * num_embeddings.
                        if dim > 0 {
                            flat_data.reserve_exact(num_embeddings * dim);
                        }
                    }

                    // embedding_dim_opt is guaranteed to be Some if self.data is not empty
                    // and the first element is a FloatVector.
                    let expected_dim = embedding_dim_opt.unwrap(); // Safe if first element was FloatVector

                    if v.len() != expected_dim {
                        return Err(PyValueError::new_err(format!(
                            "All embeddings must have the same dimension. Expected {} but got {} at index {}.",
                            expected_dim, v.len(), item_data.index
                        )));
                    }
                    flat_data.extend_from_slice(v);
                }
                EmbeddingVariant::Base64(_) => {
                    return Err(PyValueError::new_err(format!(
                        "Cannot convert to array: found Base64 encoded embedding at index {}. Only float vectors are supported.",
                        item_data.index
                    )));
                }
            }
        }

        // If the loop completed, all items were FloatVectors of consistent dimension.
        // embedding_dim_opt will be Some(actual_dimension) or Some(0) if all embeddings were empty.
        let final_embedding_dim = embedding_dim_opt.unwrap_or(0); // Should be Some if data was not empty and no error.

        let array = Array2::from_shape_vec((num_embeddings, final_embedding_dim), flat_data)
            .map_err(|e| {
                PyValueError::new_err(format!("Failed to create ndarray from embeddings: {}", e))
            })?;

        Ok(array.into_pyarray_bound(py))
    }
}

#[derive(Serialize, Debug)]
struct RerankRequest {
    query: String,
    raw_scores: bool,
    return_text: bool,
    texts: Vec<String>,
    truncate: bool,
    truncation_direction: String,
}

#[derive(Deserialize, Debug, Clone)]
#[pyclass]
struct RerankResult {
    #[pyo3(get)]
    index: usize,
    #[pyo3(get)]
    score: f64,
    #[pyo3(get)]
    text: Option<String>,
}

#[pyclass(get_all, frozen)]
#[derive(Debug, Clone)]
struct RerankResponse {
    // Made struct public for field visibility
    object: String,
    data: Vec<RerankResult>,
    total_time: Option<f64>,
    individual_request_times: Option<Vec<f64>>,
}

#[pymethods]
impl RerankResponse {
    #[new]
    #[pyo3(signature = (data, total_time = None, individual_request_times = None))]
    fn new(
        data: Vec<RerankResult>,
        total_time: Option<f64>,
        individual_request_times: Option<Vec<f64>>,
    ) -> Self {
        RerankResponse {
            object: "list".to_string(),
            data,
            total_time,
            individual_request_times,
        }
    }
}

#[derive(Serialize, Debug)]
struct ClassifyRequest {
    inputs: Vec<Vec<String>>, // changed from String
    raw_scores: bool,
    truncate: bool,
    truncation_direction: String,
}

#[derive(Deserialize, Debug, Clone)]
#[pyclass]
struct ClassificationResult {
    #[pyo3(get)]
    label: String,
    #[pyo3(get)]
    score: f64,
}

#[pyclass(get_all, frozen)]
#[derive(Debug, Clone)]
struct ClassificationResponse {
    // Made struct public
    object: String,
    data: Vec<Vec<ClassificationResult>>,
    total_time: Option<f64>,
    individual_request_times: Option<Vec<f64>>,
}

#[pymethods]
impl ClassificationResponse {
    #[new]
    #[pyo3(signature = (data, total_time = None, individual_request_times = None))]
    fn new(
        data: Vec<Vec<ClassificationResult>>,
        total_time: Option<f64>,
        individual_request_times: Option<Vec<f64>>,
    ) -> Self {
        ClassificationResponse {
            object: "list".to_string(),
            data,
            total_time,
            individual_request_times,
        }
    }
}

#[pyclass(get_all)]
struct BatchPostResponse {
    data: Vec<PyObject>,
    total_time: f64,
    individual_request_times: Vec<f64>,
    response_headers: Vec<PyObject>, // New field for headers
}

// --- PerformanceClient Definition ---
#[pyclass]
struct PerformanceClient {
    api_key: String,
    base_url: String,
    client: Client,
    runtime: Arc<Runtime>,
}

impl PerformanceClient {
    fn get_api_key(api_key: Option<String>) -> Result<String, PyErr> {
        if let Some(key) = api_key {
            return Ok(key);
        }
        if let Ok(key) = std::env::var("BASETEN_API_KEY") {
            return Ok(key);
        }
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            return Ok(key);
        }
        Err(PyValueError::new_err(
            "API key not provided and no environment variable `BASETEN_API_KEY` found",
        ))
    }

    fn validate_and_get_timeout_duration(timeout_s: f64) -> Result<Duration, PyErr> {
        let resolved_timeout_s = timeout_s;
        if !(MIN_REQUEST_TIMEOUT_S..=MAX_REQUEST_TIMEOUT_S).contains(&resolved_timeout_s) {
            return Err(PyValueError::new_err(format!(
                "Timeout {:.3}s is outside the allowed range [{:.3}s, {:.3}s].",
                resolved_timeout_s, MIN_REQUEST_TIMEOUT_S, MAX_REQUEST_TIMEOUT_S
            )));
        }
        Ok(Duration::from_secs_f64(resolved_timeout_s))
    }

    fn validate_concurrency_parameters(
        max_concurrent_requests: usize,
        batch_size: usize,
    ) -> PyResult<()> {
        if max_concurrent_requests == 0 || max_concurrent_requests > MAX_CONCURRENCY_HIGH_BATCH {
            return Err(PyValueError::new_err(format!(
                "max_concurrent_requests must be greater than 0 and less than or equal to {}",
                MAX_CONCURRENCY_HIGH_BATCH
            )));
        } else if batch_size == 0 || batch_size > MAX_BATCH_SIZE {
            return Err(PyValueError::new_err(format!(
                "batch_size must be greater than 0 and less than or equal to {}",
                MAX_BATCH_SIZE
            )));
        } else if max_concurrent_requests > MAX_CONCURRENCY_LOW_BATCH
            && batch_size < CONCURRENCY_HIGH_BATCH_SWITCH
        {
            return Err(PyValueError::new_err(format!(
                "max_concurrent_requests must be less than {} when batch_size is less than {}. Please be nice to the server side.",
                MAX_CONCURRENCY_LOW_BATCH, CONCURRENCY_HIGH_BATCH_SWITCH
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl PerformanceClient {
    #[new]
    #[pyo3(signature = (base_url, api_key = None))]
    fn new(base_url: String, api_key: Option<String>) -> PyResult<Self> {
        let api_key = PerformanceClient::get_api_key(api_key)?;
        if WARNING_SLOW_PROVIDERS
            .iter()
            .any(|&provider| base_url.contains(provider))
        {
            eprintln!(
                "Warning: Using {} as the base URL might be slow. You should consider using baseten.com instead.",
                base_url.clone()
            );
        }
        Ok(PerformanceClient {
            api_key,
            base_url,
            client: Client::new(),
            runtime: Arc::clone(&GLOBAL_RUNTIME),
        })
    }

    #[getter]
    fn api_key(&self) -> PyResult<String> {
        Ok(self.api_key.clone())
    }

    #[pyo3(signature = (input, model, encoding_format = None, dimensions = None, user = None, max_concurrent_requests = DEFAULT_CONCURRENCY, batch_size = DEFAULT_BATCH_SIZE, timeout_s = DEFAULT_REQUEST_TIMEOUT_S))]
    fn embed(
        &self,
        py: Python,
        input: Vec<String>,
        model: String,
        encoding_format: Option<String>,
        dimensions: Option<u32>,
        user: Option<String>,
        max_concurrent_requests: usize,
        batch_size: usize,
        timeout_s: f64,
    ) -> PyResult<OpenAIEmbeddingsResponse> {
        // Return OpenAIEmbeddingsResponse directly
        if input.is_empty() {
            return Err(PyValueError::new_err("Input list cannot be empty"));
        }
        PerformanceClient::validate_concurrency_parameters(max_concurrent_requests, batch_size)?;
        let timeout_duration = PerformanceClient::validate_and_get_timeout_duration(timeout_s)?;

        let client_clone = self.client.clone();
        let api_key_clone = self.api_key.clone();
        let base_url_clone = self.base_url.clone();
        let rt = Arc::clone(&self.runtime);
        let time_start_sync_op = Instant::now();

        let result_from_async_task: Result<(OpenAIEmbeddingsResponse, Vec<Duration>), PyErr> = py
            .allow_threads(move || {
                let (tx, rx) = std::sync::mpsc::channel::<
                    Result<(OpenAIEmbeddingsResponse, Vec<Duration>), PyErr>,
                >();

                rt.spawn(async move {
                    let res = process_embeddings_requests(
                        client_clone,
                        input,
                        model,
                        api_key_clone,
                        base_url_clone,
                        encoding_format,
                        dimensions,
                        user,
                        max_concurrent_requests,
                        batch_size,
                        timeout_duration,
                    )
                    .await;
                    let _ = tx.send(res);
                });

                match rx.recv() {
                    Ok(inner_result) => inner_result,
                    Err(e) => Err(PyValueError::new_err(format!(
                        "Failed to receive result from async task (channel error): {}",
                        e
                    ))),
                }
            });

        let (mut api_response, batch_durations) = result_from_async_task?;
        let total_time_val = time_start_sync_op.elapsed().as_secs_f64();
        let individual_times_val: Vec<f64> = batch_durations
            .into_iter()
            .map(|d| d.as_secs_f64())
            .collect();

        api_response.total_time = Some(total_time_val);
        api_response.individual_request_times = Some(individual_times_val);

        Ok(api_response)
    }

    #[pyo3(name = "async_embed", signature = (input, model, encoding_format = None, dimensions = None, user = None, max_concurrent_requests = DEFAULT_CONCURRENCY, batch_size = DEFAULT_BATCH_SIZE, timeout_s = DEFAULT_REQUEST_TIMEOUT_S))]
    fn async_embed<'py>(
        &self,
        py: Python<'py>,
        input: Vec<String>,
        model: String,
        encoding_format: Option<String>,
        dimensions: Option<u32>,
        user: Option<String>,
        max_concurrent_requests: usize,
        batch_size: usize,
        timeout_s: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        if input.is_empty() {
            return Err(PyValueError::new_err("Input list cannot be empty"));
        }
        PerformanceClient::validate_concurrency_parameters(max_concurrent_requests, batch_size)?;
        let timeout_duration = PerformanceClient::validate_and_get_timeout_duration(timeout_s)?;

        let client_clone = self.client.clone();
        let api_key_clone = self.api_key.clone();
        let base_url_clone = self.base_url.clone();

        let future = async move {
            let time_start_async_op = Instant::now();
            let (mut api_response, batch_durations) = process_embeddings_requests(
                client_clone,
                input,
                model,
                api_key_clone,
                base_url_clone,
                encoding_format,
                dimensions,
                user,
                max_concurrent_requests,
                batch_size,
                timeout_duration,
            )
            .await?;

            let total_time_val = time_start_async_op.elapsed().as_secs_f64();
            let individual_times_val: Vec<f64> = batch_durations
                .into_iter()
                .map(|d| d.as_secs_f64())
                .collect();

            api_response.total_time = Some(total_time_val);
            api_response.individual_request_times = Some(individual_times_val);

            Ok(api_response)
        };

        pyo3_async_runtimes::tokio::future_into_py(py, future)
    }

    #[pyo3(signature = (query, texts, raw_scores = false, return_text = false, truncate = false, truncation_direction = "Right", max_concurrent_requests = DEFAULT_CONCURRENCY, batch_size = DEFAULT_BATCH_SIZE, timeout_s = DEFAULT_REQUEST_TIMEOUT_S))]
    fn rerank(
        &self,
        py: Python,
        query: String,
        texts: Vec<String>,
        raw_scores: bool,
        return_text: bool,
        truncate: bool,
        truncation_direction: &str,
        max_concurrent_requests: usize,
        batch_size: usize,
        timeout_s: f64,
    ) -> PyResult<RerankResponse> {
        // Return RerankResponse directly
        if texts.is_empty() {
            return Err(PyValueError::new_err("Texts list cannot be empty"));
        }
        PerformanceClient::validate_concurrency_parameters(max_concurrent_requests, batch_size)?;
        let timeout_duration = PerformanceClient::validate_and_get_timeout_duration(timeout_s)?;
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let base_url = self.base_url.clone();
        let rt = Arc::clone(&self.runtime);
        let truncation_direction_owned = truncation_direction.to_string();
        let time_start = Instant::now();

        // process_rerank_requests now returns Result<(Vec<RerankResult>, Vec<Duration>), PyErr>
        let result_from_async_task: Result<(Vec<RerankResult>, Vec<Duration>), PyErr> = py
            .allow_threads(move || {
                let (tx, rx) =
                    std::sync::mpsc::channel::<Result<(Vec<RerankResult>, Vec<Duration>), PyErr>>();
                rt.spawn(async move {
                    let res = process_rerank_requests(
                        client,
                        query,
                        texts,
                        raw_scores,
                        return_text,
                        truncate,
                        truncation_direction_owned,
                        api_key,
                        base_url,
                        max_concurrent_requests,
                        batch_size,
                        timeout_duration,
                    )
                    .await;
                    let _ = tx.send(res);
                });
                rx.recv()
                    .map_err(|e| {
                        PyValueError::new_err(format!(
                            "Failed to receive rerank result (channel error): {}",
                            e
                        ))
                    })
                    .and_then(|inner_result| inner_result)
            });
        let (core_data, batch_durations) = result_from_async_task?;
        let total_time_val = time_start.elapsed().as_secs_f64();
        let individual_times_val: Vec<f64> = batch_durations
            .into_iter()
            .map(|d| d.as_secs_f64())
            .collect();

        Ok(RerankResponse::new(
            core_data,
            Some(total_time_val),
            Some(individual_times_val),
        ))
    }

    #[pyo3(name = "async_rerank", signature = (query, texts, raw_scores = false, return_text = false, truncate = false, truncation_direction = "Right", max_concurrent_requests = DEFAULT_CONCURRENCY, batch_size = DEFAULT_BATCH_SIZE, timeout_s = DEFAULT_REQUEST_TIMEOUT_S))]
    fn async_rerank<'py>(
        &self,
        py: Python<'py>,
        query: String,
        texts: Vec<String>,
        raw_scores: bool,
        return_text: bool,
        truncate: bool,
        truncation_direction: &str,
        max_concurrent_requests: usize,
        batch_size: usize,
        timeout_s: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        if texts.is_empty() {
            return Err(PyValueError::new_err("Texts list cannot be empty"));
        }
        PerformanceClient::validate_concurrency_parameters(max_concurrent_requests, batch_size)?;
        let timeout_duration = PerformanceClient::validate_and_get_timeout_duration(timeout_s)?;

        let client_clone = self.client.clone();
        let api_key_clone = self.api_key.clone();
        let base_url_clone = self.base_url.clone();
        let truncation_direction_owned = truncation_direction.to_string();

        let future = async move {
            let time_start_async_op = Instant::now();
            // process_rerank_requests returns (Vec<RerankResult>, Vec<Duration>)
            let (core_data, batch_durations) = process_rerank_requests(
                client_clone,
                query,
                texts,
                raw_scores,
                return_text,
                truncate,
                truncation_direction_owned,
                api_key_clone,
                base_url_clone,
                max_concurrent_requests,
                batch_size,
                timeout_duration,
            )
            .await?;

            let total_time_val = time_start_async_op.elapsed().as_secs_f64();
            let individual_times_val: Vec<f64> = batch_durations
                .into_iter()
                .map(|d| d.as_secs_f64())
                .collect();

            Ok(RerankResponse::new(
                core_data,
                Some(total_time_val),
                Some(individual_times_val),
            ))
        };

        pyo3_async_runtimes::tokio::future_into_py(py, future)
    }

    #[pyo3(signature = (inputs, raw_scores = false, truncate = false, truncation_direction = "Right", max_concurrent_requests = DEFAULT_CONCURRENCY, batch_size = DEFAULT_BATCH_SIZE, timeout_s = DEFAULT_REQUEST_TIMEOUT_S))]
    fn classify(
        &self,
        py: Python,
        inputs: Vec<String>,
        raw_scores: bool,
        truncate: bool,
        truncation_direction: &str,
        max_concurrent_requests: usize,
        batch_size: usize,
        timeout_s: f64,
    ) -> PyResult<ClassificationResponse> {
        // Return ClassificationResponse directly
        if inputs.is_empty() {
            return Err(PyValueError::new_err("Inputs list cannot be empty"));
        }
        PerformanceClient::validate_concurrency_parameters(max_concurrent_requests, batch_size)?;
        let timeout_duration = PerformanceClient::validate_and_get_timeout_duration(timeout_s)?;
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let base_url = self.base_url.clone();
        let rt = Arc::clone(&self.runtime);
        let truncation_direction_owned = truncation_direction.to_string();
        let time_start = Instant::now();

        // process_classify_requests returns Result<(Vec<Vec<ClassificationResult>>, Vec<Duration>), PyErr>
        let result_from_async_task: Result<(Vec<Vec<ClassificationResult>>, Vec<Duration>), PyErr> =
            py.allow_threads(move || {
                let (tx, rx) = std::sync::mpsc::channel::<
                    Result<(Vec<Vec<ClassificationResult>>, Vec<Duration>), PyErr>,
                >();
                rt.spawn(async move {
                    let res = process_classify_requests(
                        client,
                        inputs,
                        raw_scores,
                        truncate,
                        truncation_direction_owned,
                        api_key,
                        base_url,
                        max_concurrent_requests,
                        batch_size,
                        timeout_duration,
                    )
                    .await;
                    let _ = tx.send(res);
                });
                rx.recv()
                    .map_err(|e| {
                        PyValueError::new_err(format!(
                            "Failed to receive classify result (channel error): {}",
                            e
                        ))
                    })
                    .and_then(|inner_result| inner_result)
            });

        let (core_data, batch_durations) = result_from_async_task?;
        let total_time_val = time_start.elapsed().as_secs_f64();
        let individual_times_val: Vec<f64> = batch_durations
            .into_iter()
            .map(|d| d.as_secs_f64())
            .collect();

        Ok(ClassificationResponse::new(
            core_data,
            Some(total_time_val),
            Some(individual_times_val),
        ))
    }

    #[pyo3(name = "async_classify", signature = (inputs, raw_scores = false, truncate = false, truncation_direction = "Right", max_concurrent_requests = DEFAULT_CONCURRENCY, batch_size = DEFAULT_BATCH_SIZE, timeout_s = DEFAULT_REQUEST_TIMEOUT_S))]
    fn async_classify<'py>(
        &self,
        py: Python<'py>,
        inputs: Vec<String>,
        raw_scores: bool,
        truncate: bool,
        truncation_direction: &str,
        max_concurrent_requests: usize,
        batch_size: usize,
        timeout_s: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        if inputs.is_empty() {
            return Err(PyValueError::new_err("Inputs list cannot be empty"));
        }
        PerformanceClient::validate_concurrency_parameters(max_concurrent_requests, batch_size)?;
        let timeout_duration = PerformanceClient::validate_and_get_timeout_duration(timeout_s)?;

        let client_clone = self.client.clone();
        let api_key_clone = self.api_key.clone();
        let base_url_clone = self.base_url.clone();
        let truncation_direction_owned = truncation_direction.to_string();

        let future = async move {
            let time_start_async_op = Instant::now();
            // process_classify_requests returns (Vec<Vec<ClassificationResult>>, Vec<Duration>)
            let (core_data, batch_durations) = process_classify_requests(
                client_clone,
                inputs,
                raw_scores,
                truncate,
                truncation_direction_owned,
                api_key_clone,
                base_url_clone,
                max_concurrent_requests,
                batch_size,
                timeout_duration,
            )
            .await?;

            let total_time_val = time_start_async_op.elapsed().as_secs_f64();
            let individual_times_val: Vec<f64> = batch_durations
                .into_iter()
                .map(|d| d.as_secs_f64())
                .collect();

            Ok(ClassificationResponse::new(
                core_data,
                Some(total_time_val),
                Some(individual_times_val),
            ))
        };

        pyo3_async_runtimes::tokio::future_into_py(py, future)
    }

    #[pyo3(signature = (url_path, payloads, max_concurrent_requests = DEFAULT_CONCURRENCY, timeout_s = DEFAULT_REQUEST_TIMEOUT_S))]
    fn batch_post(
        &self,
        py: Python,
        url_path: String,
        payloads: Vec<PyObject>,
        max_concurrent_requests: usize,
        timeout_s: f64,
    ) -> PyResult<BatchPostResponse> {
        if payloads.is_empty() {
            return Err(PyValueError::new_err("Payloads list cannot be empty"));
        }
        PerformanceClient::validate_concurrency_parameters(max_concurrent_requests, 128)?;
        let timeout_duration = PerformanceClient::validate_and_get_timeout_duration(timeout_s)?;

        let mut payloads_json: Vec<JsonValue> = Vec::with_capacity(payloads.len());
        for (idx, py_obj) in payloads.into_iter().enumerate() {
            let bound_obj = py_obj.bind(py);
            let json_val = depythonize(bound_obj).map_err(|e| {
                PyValueError::new_err(format!(
                    "Failed to depythonize payload at index {}: {}",
                    idx, e
                ))
            })?;
            payloads_json.push(json_val);
        }

        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let base_url = self.base_url.clone();
        let rt = Arc::clone(&self.runtime);
        let time_start = std::time::Instant::now();

        // The async task now returns Result<Vec<(JsonValue, HashMap<String, String>, Duration)>, PyErr>
        let result_from_async_task: Result<
            Vec<(JsonValue, HashMap<String, String>, Duration)>,
            PyErr,
        > = py.allow_threads(move || {
            let (tx, rx) = std::sync::mpsc::channel::<
                Result<Vec<(JsonValue, HashMap<String, String>, Duration)>, PyErr>,
            >();
            rt.spawn(async move {
                let res = process_batch_post_requests(
                    client,
                    url_path,
                    payloads_json,
                    api_key,
                    base_url,
                    max_concurrent_requests,
                    timeout_duration,
                )
                .await;
                let _ = tx.send(res);
            });
            rx.recv()
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "Failed to receive result from async task (channel error): {}",
                        e
                    ))
                })
                .and_then(|inner_result| inner_result)
        });

        let response_data_with_times_and_headers = result_from_async_task?;

        let mut results_py: Vec<PyObject> =
            Vec::with_capacity(response_data_with_times_and_headers.len());
        let mut individual_request_times_collected: Vec<f64> =
            Vec::with_capacity(response_data_with_times_and_headers.len());
        let mut collected_headers_py: Vec<PyObject> =
            Vec::with_capacity(response_data_with_times_and_headers.len());

        for (idx, (json_val, headers_map, duration)) in
            response_data_with_times_and_headers.into_iter().enumerate()
        {
            let py_obj_bound = pythonize(py, &json_val).map_err(|e| {
                PyValueError::new_err(format!(
                    "Failed to pythonize response data at index {}: {}",
                    idx, e
                ))
            })?;
            results_py.push(py_obj_bound.to_object(py));

            let headers_py_obj = pythonize(py, &headers_map).map_err(|e| {
                PyValueError::new_err(format!(
                    "Failed to pythonize headers at index {}: {}",
                    idx, e
                ))
            })?;
            collected_headers_py.push(headers_py_obj.to_object(py));

            individual_request_times_collected.push(duration.as_secs_f64());
        }

        let total_time = time_start.elapsed().as_secs_f64();

        Ok(BatchPostResponse {
            data: results_py,
            total_time,
            individual_request_times: individual_request_times_collected,
            response_headers: collected_headers_py,
        })
    }

    #[pyo3(name = "async_batch_post", signature = (url_path, payloads, max_concurrent_requests = DEFAULT_CONCURRENCY, timeout_s = DEFAULT_REQUEST_TIMEOUT_S))]
    fn async_batch_post<'py>(
        &self,
        py: Python<'py>,
        url_path: String,
        payloads: Vec<PyObject>,
        max_concurrent_requests: usize,
        timeout_s: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        if payloads.is_empty() {
            return Err(PyValueError::new_err("Payloads list cannot be empty"));
        }
        PerformanceClient::validate_concurrency_parameters(max_concurrent_requests, 128)?;
        let timeout_duration = PerformanceClient::validate_and_get_timeout_duration(timeout_s)?;

        let mut payloads_json: Vec<JsonValue> = Vec::with_capacity(payloads.len());
        for (idx, py_obj) in payloads.into_iter().enumerate() {
            let bound_obj = py_obj.bind(py);
            let json_val = depythonize(bound_obj).map_err(|e| {
                PyValueError::new_err(format!(
                    "Failed to depythonize payload at index {}: {}",
                    idx, e
                ))
            })?;
            payloads_json.push(json_val);
        }

        let client_clone = self.client.clone();
        let api_key_clone = self.api_key.clone();
        let base_url_clone = self.base_url.clone();

        let future = async move {
            let time_start_async_op = std::time::Instant::now();

            let response_data_with_times_and_headers = process_batch_post_requests(
                client_clone,
                url_path,
                payloads_json,
                api_key_clone,
                base_url_clone,
                max_concurrent_requests,
                timeout_duration,
            )
            .await?;

            let total_time_async_op = time_start_async_op.elapsed().as_secs_f64();

            Python::with_gil(|py_gil| {
                let mut results_py: Vec<PyObject> =
                    Vec::with_capacity(response_data_with_times_and_headers.len());
                let mut individual_request_times_collected: Vec<f64> =
                    Vec::with_capacity(response_data_with_times_and_headers.len());
                let mut collected_headers_py: Vec<PyObject> =
                    Vec::with_capacity(response_data_with_times_and_headers.len());

                for (idx, (json_val, headers_map, duration)) in
                    response_data_with_times_and_headers.into_iter().enumerate()
                {
                    let py_obj_bound = pythonize(py_gil, &json_val).map_err(|e| {
                        PyValueError::new_err(format!(
                            "Failed to pythonize response data at index {}: {}",
                            idx, e
                        ))
                    })?;
                    results_py.push(py_obj_bound.to_object(py_gil));

                    let headers_py_obj = pythonize(py_gil, &headers_map).map_err(|e| {
                        PyValueError::new_err(format!(
                            "Failed to pythonize headers at index {}: {}",
                            idx, e
                        ))
                    })?;
                    collected_headers_py.push(headers_py_obj.to_object(py_gil));

                    individual_request_times_collected.push(duration.as_secs_f64());
                }

                Ok(BatchPostResponse {
                    data: results_py,
                    total_time: total_time_async_op,
                    individual_request_times: individual_request_times_collected,
                    response_headers: collected_headers_py,
                })
            })
        };
        pyo3_async_runtimes::tokio::future_into_py(py, future)
    }
}

// --- Send Single Embedding Request ---
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
) -> Result<OpenAIEmbeddingsResponse, PyErr> {
    let request_payload = OpenAIEmbeddingsRequest {
        input: texts_batch,
        model,
        encoding_format,
        dimensions,
        user,
    };

    let url = format!("{}/v1/embeddings", base_url.trim_end_matches('/'));

    let request_builder = client
        .post(&url)
        .bearer_auth(api_key.clone()) // Clone api_key if it's used after this
        .json(&request_payload)
        .timeout(request_timeout);

    let response = send_request_with_retry(
        request_builder,
        MAX_HTTP_RETRIES,
        Duration::from_millis(INITIAL_BACKOFF_MS),
    )
    .await?;

    let successful_response = ensure_successful_response(response).await?;

    successful_response
        .json::<OpenAIEmbeddingsResponse>()
        .await
        .map_err(|e| PyValueError::new_err(format!("Failed to parse response JSON: {}", e)))
}

// --- Process Embeddings Requests ---
async fn process_embeddings_requests(
    client: Client,
    texts: Vec<String>,
    model: String,
    api_key: String,
    base_url: String,
    encoding_format: Option<String>,
    dimensions: Option<u32>,
    user: Option<String>,
    max_concurrent_requests: usize,
    batch_size: usize,
    request_timeout_duration: Duration,
) -> Result<(OpenAIEmbeddingsResponse, Vec<Duration>), PyErr> {
    // Updated return type
    let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));
    let mut tasks = Vec::new();
    let total_texts = texts.len();
    let cancel_token = Arc::new(AtomicBool::new(false));
    let model_for_response = model.clone();

    for (batch_index, user_text_batch) in texts.chunks(batch_size).enumerate() {
        let client_clone = client.clone();
        let model_for_task = model.clone();
        let api_key_clone = api_key.clone();
        let base_url_clone = base_url.clone();
        let encoding_format_clone = encoding_format.clone();
        let dimensions_clone = dimensions;
        let user_clone = user.clone();
        let user_text_batch_owned = user_text_batch.to_vec();
        let semaphore_clone = Arc::clone(&semaphore);
        let cancel_token_clone = Arc::clone(&cancel_token);
        let individual_request_timeout = request_timeout_duration;
        let current_batch_absolute_start_index = batch_index * batch_size;

        tasks.push(tokio::spawn(async move {
            let _permit =
                acquire_permit_or_cancel(semaphore_clone, cancel_token_clone.clone()).await?;

            let request_time_start = Instant::now(); // Measure time for single request
            let result = send_single_embedding_request(
                client_clone,
                user_text_batch_owned,
                model_for_task,
                api_key_clone,
                base_url_clone,
                encoding_format_clone,
                dimensions_clone,
                user_clone,
                individual_request_timeout,
            )
            .await;
            let request_time_elapsed = request_time_start.elapsed(); // Get duration

            match result {
                Ok(mut response) => {
                    for item in &mut response.data {
                        item.index += current_batch_absolute_start_index;
                    }
                    // Return the full response from send_single_embedding_request and its duration
                    Ok((response, request_time_elapsed))
                }
                Err(e) => {
                    cancel_token_clone.store(true, Ordering::SeqCst);
                    Err(e)
                }
            }
        }));
    }

    let task_join_results = join_all(tasks).await;

    let mut all_embedding_data: Vec<OpenAIEmbeddingData> = Vec::with_capacity(total_texts);
    let mut aggregated_prompt_tokens: u32 = 0;
    let mut aggregated_total_tokens: u32 = 0;
    let mut individual_batch_durations: Vec<Duration> = Vec::new(); // To store durations
    let mut first_error: Option<PyErr> = None;

    for result in task_join_results {
        // D for process_task_outcome is now (OpenAIEmbeddingsResponse, Duration)
        if let Some((response_part, duration_part)) =
            process_task_outcome(result, &mut first_error, &cancel_token)
        {
            all_embedding_data.extend(response_part.data);
            aggregated_prompt_tokens =
                aggregated_prompt_tokens.saturating_add(response_part.usage.prompt_tokens);
            aggregated_total_tokens =
                aggregated_total_tokens.saturating_add(response_part.usage.total_tokens);
            individual_batch_durations.push(duration_part); // Collect duration
        }
    }

    if let Some(err) = first_error {
        return Err(err);
    }

    all_embedding_data.sort_by_key(|d| d.index);
    let final_response = OpenAIEmbeddingsResponse {
        object: "list".to_string(),
        data: all_embedding_data,
        model: model_for_response,
        usage: OpenAIUsage {
            prompt_tokens: aggregated_prompt_tokens,
            total_tokens: aggregated_total_tokens,
        },
        total_time: None,               // These will be set by the client method
        individual_request_times: None, // These will be set by the client method
    };
    Ok((final_response, individual_batch_durations)) // Return response and durations
}

// --- Send Single Rerank Request ---
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
) -> Result<Vec<RerankResult>, PyErr> {
    let request_payload = RerankRequest {
        query,
        raw_scores,
        return_text,
        texts: texts_batch,
        truncate,
        truncation_direction,
    };

    let url = format!("{}/rerank", base_url.trim_end_matches('/'));

    let request_builder = client
        .post(&url)
        .bearer_auth(api_key.clone())
        .json(&request_payload)
        .timeout(request_timeout);

    let response = send_request_with_retry(
        request_builder,
        MAX_HTTP_RETRIES,
        Duration::from_millis(INITIAL_BACKOFF_MS),
    )
    .await?;

    let successful_response = ensure_successful_response(response).await?;

    successful_response
        .json::<Vec<RerankResult>>()
        .await
        .map_err(|e| PyValueError::new_err(format!("Failed to parse rerank response JSON: {}", e)))
}

// --- Process Rerank Requests ---
async fn process_rerank_requests(
    client: Client,
    query: String,
    texts: Vec<String>,
    raw_scores: bool,
    return_text: bool,
    truncate: bool,
    truncation_direction: String,
    api_key: String,
    base_url: String,
    max_concurrent_requests: usize,
    batch_size: usize,
    request_timeout_duration: Duration,
) -> Result<(Vec<RerankResult>, Vec<Duration>), PyErr> {
    // Updated return type
    let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));
    let mut tasks = Vec::new();
    let cancel_token = Arc::new(AtomicBool::new(false));

    for (batch_idx, texts_batch) in texts.chunks(batch_size).enumerate() {
        let client_clone = client.clone();
        let query_clone = query.clone();
        let api_key_clone = api_key.clone();
        let base_url_clone = base_url.clone();
        let truncation_direction_clone = truncation_direction.clone();
        let texts_batch_owned = texts_batch.to_vec();
        let semaphore_clone = Arc::clone(&semaphore);
        let cancel_token_clone = Arc::clone(&cancel_token);
        let individual_request_timeout = request_timeout_duration;
        // Calculate the starting index for this batch relative to the original `texts` Vec
        let current_batch_absolute_start_index = batch_idx * batch_size;

        tasks.push(tokio::spawn(async move {
            let _permit =
                acquire_permit_or_cancel(semaphore_clone, cancel_token_clone.clone()).await?;

            let request_time_start = Instant::now();
            let result = send_single_rerank_request(
                client_clone,
                query_clone,
                texts_batch_owned,
                raw_scores,
                return_text,
                truncate,
                truncation_direction_clone,
                api_key_clone,
                base_url_clone,
                individual_request_timeout,
            )
            .await;
            let request_time_elapsed = request_time_start.elapsed();

            match result {
                Ok(mut batch_results) => {
                    // Adjust index for each result in the batch
                    for item in &mut batch_results {
                        item.index += current_batch_absolute_start_index;
                    }
                    Ok((batch_results, request_time_elapsed))
                }
                Err(e) => {
                    cancel_token_clone.store(true, Ordering::SeqCst);
                    Err(e)
                }
            }
        }));
    }

    let task_join_results = join_all(tasks).await;

    let mut all_results_data: Vec<RerankResult> = Vec::new();
    let mut individual_batch_durations: Vec<Duration> = Vec::new();
    let mut first_error: Option<PyErr> = None;

    for result in task_join_results {
        // D for process_task_outcome is (Vec<RerankResult>, Duration)
        if let Some((batch_data_part, duration)) =
            process_task_outcome(result, &mut first_error, &cancel_token)
        {
            all_results_data.extend(batch_data_part);
            individual_batch_durations.push(duration);
        }
    }

    if let Some(err) = first_error {
        return Err(err);
    }

    all_results_data.sort_by_key(|d| d.index);
    Ok((all_results_data, individual_batch_durations)) // Return core data and durations
}

// --- Send Single Classify Request ---
async fn send_single_classify_request(
    client: Client,
    inputs: Vec<Vec<String>>,
    raw_scores: bool,
    truncate: bool,
    truncation_direction: String,
    api_key: String,
    base_url: String,
    request_timeout: Duration,
) -> Result<Vec<Vec<ClassificationResult>>, PyErr> {
    let request_payload = ClassifyRequest {
        inputs,
        raw_scores,
        truncate,
        truncation_direction,
    };

    let url = format!("{}/predict", base_url.trim_end_matches('/'));

    let request_builder = client
        .post(&url)
        .bearer_auth(api_key.clone())
        .json(&request_payload)
        .timeout(request_timeout);

    let response = send_request_with_retry(
        request_builder,
        MAX_HTTP_RETRIES,
        Duration::from_millis(INITIAL_BACKOFF_MS),
    )
    .await?;

    let successful_response = ensure_successful_response(response).await?;

    successful_response
        .json::<Vec<Vec<ClassificationResult>>>()
        .await
        .map_err(|e| {
            PyValueError::new_err(format!("Failed to parse classify response JSON: {}", e))
        })
}

// --- Process Classify Requests ---
async fn process_classify_requests(
    client: Client,
    inputs: Vec<String>,
    raw_scores: bool,
    truncate: bool,
    truncation_direction: String,
    api_key: String,
    base_url: String,
    max_concurrent_requests: usize,
    batch_size: usize,
    request_timeout_duration: Duration,
) -> Result<(Vec<Vec<ClassificationResult>>, Vec<Duration>), PyErr> {
    // Updated return type
    let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));
    let mut tasks = Vec::new();
    let cancel_token = Arc::new(AtomicBool::new(false));

    for input_chunk_slice in inputs.chunks(batch_size) {
        let client_clone = client.clone();
        let api_key_clone = api_key.clone();
        let base_url_clone = base_url.clone();
        let truncation_direction_clone = truncation_direction.clone();
        let inputs_for_api_owned: Vec<Vec<String>> =
            input_chunk_slice.iter().map(|s| vec![s.clone()]).collect();
        let semaphore_clone = Arc::clone(&semaphore);
        let cancel_token_clone = Arc::clone(&cancel_token);
        let individual_request_timeout = request_timeout_duration;

        tasks.push(tokio::spawn(async move {
            let _permit =
                acquire_permit_or_cancel(semaphore_clone, cancel_token_clone.clone()).await?;

            let request_time_start = Instant::now();
            let result = send_single_classify_request(
                client_clone,
                inputs_for_api_owned,
                raw_scores,
                truncate,
                truncation_direction_clone,
                api_key_clone,
                base_url_clone,
                individual_request_timeout,
            )
            .await;
            let request_time_elapsed = request_time_start.elapsed();

            match result {
                Ok(batch_results) => Ok((batch_results, request_time_elapsed)),
                Err(e) => {
                    cancel_token_clone.store(true, Ordering::SeqCst);
                    Err(e)
                }
            }
        }));
    }

    let task_join_results = join_all(tasks).await;

    let mut all_results_data: Vec<Vec<ClassificationResult>> = Vec::new();
    let mut individual_batch_durations: Vec<Duration> = Vec::new();
    let mut first_error: Option<PyErr> = None;

    for result in task_join_results {
        // D for process_task_outcome is (Vec<Vec<ClassificationResult>>, Duration)
        if let Some((batch_data_part, duration)) =
            process_task_outcome(result, &mut first_error, &cancel_token)
        {
            all_results_data.extend(batch_data_part); // extend, not append, if batch_data_part is Vec<Vec<...>>
            individual_batch_durations.push(duration);
        }
    }

    if let Some(err) = first_error {
        return Err(err);
    }

    // Note: Classification results are a Vec of Vecs, order is preserved by batch.
    // If individual items within batches had original indices, sorting would be needed here.
    // Assuming the API returns results for a batch in the order inputs were sent for that batch.
    Ok((all_results_data, individual_batch_durations)) // Return core data and durations
}

// --- Send Single Batch Post Request ---
// Now returns (JsonValue, HashMap<String, String>)
async fn send_single_batch_post_request(
    client: Client,
    full_url: String,
    payload_json: JsonValue,
    api_key: String,
    request_timeout: Duration,
) -> Result<(JsonValue, HashMap<String, String>), PyErr> {
    // Updated return type
    let request_builder = client
        .post(&full_url)
        .bearer_auth(api_key.clone())
        .json(&payload_json)
        .timeout(request_timeout);

    let response = send_request_with_retry(
        request_builder,
        MAX_HTTP_RETRIES,
        Duration::from_millis(INITIAL_BACKOFF_MS),
    )
    .await?;

    let successful_response = ensure_successful_response(response).await?;

    // Extract headers
    let mut headers_map = HashMap::new();
    for (name, value) in successful_response.headers().iter() {
        headers_map.insert(
            name.as_str().to_string(),
            String::from_utf8_lossy(value.as_bytes()).into_owned(),
        );
    }

    let response_json_value: JsonValue = successful_response
        .json::<JsonValue>()
        .await
        .map_err(|e| PyValueError::new_err(format!("Failed to parse response JSON: {}", e)))?;

    Ok((response_json_value, headers_map)) // Return JSON and headers
}

// --- Process Batch Post Requests ---
// Now returns Result<Vec<(JsonValue, HashMap<String, String>, Duration)>, PyErr>
async fn process_batch_post_requests(
    client: Client,
    url_path: String,
    payloads_json: Vec<JsonValue>,
    api_key: String,
    base_url: String,
    max_concurrent_requests: usize,
    request_timeout_duration: Duration,
) -> Result<Vec<(JsonValue, HashMap<String, String>, Duration)>, PyErr> {
    // Updated return type
    let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));
    let mut tasks = Vec::new();
    let cancel_token = Arc::new(AtomicBool::new(false));
    let total_payloads = payloads_json.len();

    for (index, payload_item_json) in payloads_json.into_iter().enumerate() {
        let client_clone = client.clone();
        let api_key_clone = api_key.clone();
        let base_url_clone = base_url.clone();
        let url_path_clone = url_path.clone();
        let semaphore_clone = Arc::clone(&semaphore);
        let cancel_token_clone = Arc::clone(&cancel_token);
        let individual_request_timeout = request_timeout_duration;

        tasks.push(tokio::spawn(async move {
            let permit_guard =
                acquire_permit_or_cancel(semaphore_clone, cancel_token_clone.clone()).await?;

            let full_url = format!(
                "{}/{}",
                base_url_clone.trim_end_matches('/'),
                url_path_clone.trim_start_matches('/')
            );
            let request_time_start = std::time::Instant::now();
            // send_single_batch_post_request now returns (JsonValue, HashMap<String, String>)
            let result_tuple = send_single_batch_post_request(
                client_clone,
                full_url,
                payload_item_json,
                api_key_clone,
                individual_request_timeout,
            )
            .await;

            drop(permit_guard);
            let request_time_elapsed = request_time_start.elapsed();

            match result_tuple {
                Ok((response_json_value, headers_map)) => {
                    // Return with original index, JsonValue, Headers, and Duration
                    Ok((
                        index,
                        response_json_value,
                        headers_map,
                        request_time_elapsed,
                    ))
                }
                Err(e) => {
                    cancel_token_clone.store(true, Ordering::SeqCst);
                    Err(e)
                }
            }
        }));
    }

    let task_join_results = join_all(tasks).await;
    // D for process_task_outcome will be (usize, JsonValue, HashMap<String, String>, Duration)
    let mut indexed_results: Vec<(usize, JsonValue, HashMap<String, String>, Duration)> =
        Vec::with_capacity(total_payloads);
    let mut first_error: Option<PyErr> = None;

    for result in task_join_results {
        if let Some(indexed_data_part) =
            process_task_outcome(result, &mut first_error, &cancel_token)
        {
            indexed_results.push(indexed_data_part);
        }
    }

    if let Some(err) = first_error {
        return Err(err);
    }

    indexed_results.sort_by_key(|&(original_index, _, _, _)| original_index);

    // Map to the final Vec<(JsonValue, HashMap<String, String>, Duration)>
    let final_results: Vec<(JsonValue, HashMap<String, String>, Duration)> = indexed_results
        .into_iter()
        .map(|(_, val, headers, dur)| (val, headers, dur))
        .collect();

    Ok(final_results)
}

// Helper function to process task results and manage errors
fn process_task_outcome<D>(
    task_join_result: Result<Result<D, PyErr>, JoinError>, // Removed OwnedSemaphorePermit from here
    first_error: &mut Option<PyErr>,
    cancel_token: &Arc<AtomicBool>,
) -> Option<D> {
    match task_join_result {
        Ok(Ok(data)) => {
            // Changed from Ok(Ok((data, _permit)))
            // Task succeeded
            if first_error.is_none() {
                Some(data)
            } else {
                // A dominant error already occurred, discard this successful result
                None
            }
        }
        Ok(Err(current_err)) => {
            // Task returned a PyErr (API error or deliberate cancellation)
            let is_current_err_cancellation = current_err
                .to_string()
                .ends_with(CANCELLATION_ERROR_MESSAGE_DETAIL);

            if let Some(ref existing_err) = first_error {
                let is_existing_err_cancellation = existing_err
                    .to_string()
                    .ends_with(CANCELLATION_ERROR_MESSAGE_DETAIL);
                if is_existing_err_cancellation && !is_current_err_cancellation {
                    // If existing error is the generic cancellation message,
                    // and current error is a more specific one, replace it.
                    *first_error = Some(current_err);
                }
                // Otherwise, keep the `first_error` as is.
            } else {
                // No error recorded yet, take this one.
                *first_error = Some(current_err);
            }
            None
        }
        Err(join_err) => {
            // Task panicked
            let panic_py_err = PyValueError::new_err(format!("Tokio task panicked: {}", join_err));
            if let Some(ref existing_err) = first_error {
                let is_existing_err_cancellation = existing_err
                    .to_string()
                    .ends_with(CANCELLATION_ERROR_MESSAGE_DETAIL);
                if is_existing_err_cancellation {
                    // If existing error is the generic cancellation, prefer the panic error.
                    *first_error = Some(panic_py_err);
                }
                // Otherwise (existing error is specific), keep the existing specific error.
            } else {
                // No error recorded yet, take the panic error.
                *first_error = Some(panic_py_err);
            }
            // Ensure other tasks are signalled to cancel if one panics,
            // as the panicked task itself wouldn't have set the cancel_token.
            cancel_token.store(true, Ordering::SeqCst);
            None
        }
    }
}

// --- Helper function to acquire permit and check for cancellation ---
async fn acquire_permit_or_cancel(
    semaphore: Arc<Semaphore>,
    local_cancel_token: Arc<AtomicBool>,
) -> Result<OwnedSemaphorePermit, PyErr> {
    // Select between acquiring a permit and a cancellation signal.
    // This avoids holding a permit if a cancellation signal is already present or arrives quickly.
    tokio::select! {
        biased; // Prioritize checking cancellation signals first.

        // Check for global Ctrl+C
        _ = tokio::time::sleep(Duration::from_millis(1)), if CTRL_C_RECEIVED.load(Ordering::SeqCst) => {
            local_cancel_token.store(true, Ordering::SeqCst); // Signal local tasks too
            return Err(PyValueError::new_err(CTRL_C_ERROR_MESSAGE_DETAIL));
        }

        // Check for local cancellation token
        _ = tokio::time::sleep(Duration::from_millis(1)), if local_cancel_token.load(Ordering::SeqCst) => {
            return Err(PyValueError::new_err(CANCELLATION_ERROR_MESSAGE_DETAIL));
        }

        // Try to acquire the permit
        permit_result = semaphore.acquire_owned() => {
            let permit = permit_result.map_err(|e| {
                PyValueError::new_err(format!("Semaphore acquire_owned failed: {}", e))
            })?;

            // Re-check cancellation signals after acquiring the permit, in case they occurred
            // while waiting for the permit.
            if CTRL_C_RECEIVED.load(Ordering::SeqCst) {
                local_cancel_token.store(true, Ordering::SeqCst);
                // Permit is dropped here as it goes out of scope if we return Err.
                return Err(PyValueError::new_err(CTRL_C_ERROR_MESSAGE_DETAIL));
            }
            if local_cancel_token.load(Ordering::SeqCst) {
                // Permit is dropped here.
                return Err(PyValueError::new_err(CANCELLATION_ERROR_MESSAGE_DETAIL));
            }
            Ok(permit)
        }
    }
}

// Helper function to check for a successful response.
async fn ensure_successful_response(
    response: reqwest::Response,
) -> Result<reqwest::Response, PyErr> {
    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        Err(PyErr::new::<HTTPError, _>((
            status.as_u16(),
            format!("API request failed with status {}: {}", status, error_text),
        )))
    } else {
        Ok(response)
    }
}

async fn send_request_with_retry(
    request_builder: reqwest::RequestBuilder,
    max_retries: u32,
    initial_backoff: Duration,
) -> Result<reqwest::Response, PyErr> {
    let mut retries_done = 0;
    let mut current_backoff = initial_backoff;

    loop {
        let request_builder_clone = request_builder
            .try_clone()
            .ok_or_else(|| PyValueError::new_err("Failed to clone request builder for retry"))?;

        match request_builder_clone.send().await {
            Ok(response) => {
                if response.status().is_success() {
                    return Ok(response);
                }

                // Not a success, check if it's a retryable HTTP error (429 or 5xx)
                if response.status().as_u16() == 429 || response.status().is_server_error() {
                    if retries_done >= max_retries {
                        // Max retries performed for these HTTP errors
                        // Ensure this goes through your HTTPError mapping
                        return ensure_successful_response(response).await;
                    }
                    // If not max retries, fall through to common retry logic below
                } else {
                    // Non-retryable HTTP error (e.g., 400, 401, 403, 404)
                    // Ensure this goes through your HTTPError mapping and then return
                    return ensure_successful_response(response).await;
                }
            }
            Err(network_err) => { // This handles "error sending request" (reqwest::Error)
                println!("Network/send error: {}", network_err);
                if retries_done >= 2 {
                    // Max retries performed for network/send errors
                    return Err(PyValueError::new_err(format!(
                        "Request failed after {} retries with network/send error: {}",
                        retries_done, network_err
                    )));
                }
                // If not max retries, fall through to common retry logic below
                // Optionally, log the intermediate network_err here for debugging:
                // eprintln!("Network/send error (retry {} of {}): {}. Retrying...", retries_done + 1, max_retries, network_err);
            }
        }

        // If we reach here, it means a retry is needed (either for a retryable HTTP error
        // or a network/send error) and we haven't exhausted retries.
        retries_done += 1;

        // Exponential backoff
        // Using MAX_BACKOFF_DURATION as a const Duration as per the function's context in your file.
        // If MAX_BACKOFF_DURATION was changed to a Lazy<Duration> elsewhere, it would need dereferencing: *MAX_BACKOFF_DURATION
        let backoff_duration = current_backoff.min(MAX_BACKOFF_DURATION);
        tokio::time::sleep(backoff_duration).await;
        current_backoff = current_backoff.saturating_mul(4); // Using saturating_mul and original factor
    }
}

// --- PyO3 Module Definition ---
#[pymodule]
fn baseten_performance_client(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PerformanceClient>()?;
    m.add_class::<OpenAIEmbeddingsResponse>()?;
    m.add_class::<OpenAIEmbeddingData>()?;
    m.add_class::<OpenAIUsage>()?;
    m.add_class::<RerankResult>()?;
    m.add_class::<RerankResponse>()?; // This is the modified one
    m.add_class::<ClassificationResult>()?;
    m.add_class::<ClassificationResponse>()?; // This is the modified one
    m.add_class::<BatchPostResponse>()?;
    // Remove EmbeddingsClientResponse, RerankClientResponse, ClassifyClientResponse if they were added
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

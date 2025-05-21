// implements client for openai embeddings
// as python api that fans out to multiple requests
use futures::future::join_all;
use once_cell::sync::Lazy; // Import Lazy
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::mpsc; // Add this import
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;
use tokio::sync::Semaphore; // Add this for timeout support

// --- Global Tokio Runtime ---
static GLOBAL_RUNTIME: Lazy<Arc<Runtime>> =
    Lazy::new(|| Arc::new(Runtime::new().expect("Failed to create global Tokio runtime")));

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
#[pyclass]
struct OpenAIEmbeddingsResponse {
    #[pyo3(get)]
    object: String,
    #[pyo3(get)]
    data: Vec<OpenAIEmbeddingData>,
    #[pyo3(get)]
    model: String,
    #[pyo3(get)]
    usage: OpenAIUsage,
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
    text: String,
}

#[derive(Serialize, Debug)]
struct ClassifyRequest {
    inputs: String,
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

// --- SyncClient Definition ---
#[pyclass]
struct SyncClient {
    api_key: String,
    api_base: String,
    client: Client,
    runtime: Arc<Runtime>,
}

impl SyncClient {
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

    fn validate_and_get_timeout_duration(timeout_s: Option<f64>) -> Result<Duration, PyErr> {
        let resolved_timeout_s = timeout_s.unwrap_or(DEFAULT_REQUEST_TIMEOUT_S);
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
        if max_concurrent_requests == 0 || max_concurrent_requests > MAX_CONCURRENCY {
            return Err(PyValueError::new_err(format!(
                "max_concurrent_requests must be greater than 0 and less than {}",
                MAX_CONCURRENCY
            )));
        }
        if batch_size == 0 || batch_size > MAX_BATCH_SIZE {
            return Err(PyValueError::new_err(format!(
                "batch_size must be greater than 0 and less than {}",
                MAX_BATCH_SIZE
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl SyncClient {
    #[new]
    #[pyo3(signature = (api_base, api_key = None))]
    fn new(api_base: String, api_key: Option<String>) -> PyResult<Self> {
        let api_key = SyncClient::get_api_key(api_key)?;
        Ok(SyncClient {
            api_key,
            api_base,
            client: Client::new(),
            runtime: Arc::clone(&GLOBAL_RUNTIME),
        })
    }

    #[getter]
    fn api_key(&self) -> PyResult<String> {
        Ok(self.api_key.clone())
    }

    #[pyo3(signature = (input, model, encoding_format = None, dimensions = None, user = None, max_concurrent_requests = 64, batch_size = 4, timeout_s = None))]
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
        timeout_s: Option<f64>, // New timeout parameter
    ) -> PyResult<PyObject> {
        if input.is_empty() {
            return Err(PyValueError::new_err("Input list cannot be empty"));
        }
        SyncClient::validate_concurrency_parameters(max_concurrent_requests, batch_size)?;

        // Validate and set the request timeout.
        let timeout_duration = SyncClient::validate_and_get_timeout_duration(timeout_s)?;

        let model_string: String = model.to_string();

        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let api_base = self.api_base.clone();
        let rt = Arc::clone(&self.runtime);

        let result_from_async_task: Result<OpenAIEmbeddingsResponse, PyErr> =
            py.allow_threads(move || {
                let (tx, rx) = mpsc::channel::<Result<OpenAIEmbeddingsResponse, PyErr>>();

                rt.spawn(async move {
                    let res = process_embeddings_requests(
                        client,
                        input,
                        model_string,
                        api_key,
                        api_base,
                        encoding_format,
                        dimensions,
                        user,
                        max_concurrent_requests,
                        batch_size,
                        timeout_duration, // Pass timeout to overall processing
                    )
                    .await;
                    if tx.send(res).is_err() {
                        // Receiver dropped, nothing to do.
                    }
                });

                match rx.recv() {
                    Ok(res) => res,
                    Err(e) => Err(PyValueError::new_err(format!(
                        "Failed to receive result from async task: {}",
                        e
                    ))),
                }
            });

        let successful_response = result_from_async_task?;
        Python::with_gil(|py| Ok(successful_response.into_py(py)))
    }

    #[pyo3(signature = (query, texts, raw_scores = false, return_text = false, truncate = false, truncation_direction = "Right", max_concurrent_requests = 64, batch_size = 4, timeout_s = None))]
    fn rerank(
        &self,
        py: Python,
        query: String,
        texts: Vec<String>,
        raw_scores: bool,
        return_text: bool,
        truncate: bool,
        truncation_direction: &str, // Changed name and type
        max_concurrent_requests: usize,
        batch_size: usize,
        timeout_s: Option<f64>,
    ) -> PyResult<PyObject> {
        if texts.is_empty() {
            return Err(PyValueError::new_err("Texts list cannot be empty"));
        }
        SyncClient::validate_concurrency_parameters(max_concurrent_requests, batch_size)?;
        let timeout_duration = SyncClient::validate_and_get_timeout_duration(timeout_s)?;
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let api_base = self.api_base.clone();
        let rt = Arc::clone(&self.runtime);

        let truncation_direction = truncation_direction.to_string(); // Convert to String

        let result_from_async_task: Result<Vec<RerankResult>, PyErr> =
            py.allow_threads(move || {
                let (tx, rx) = mpsc::channel::<Result<Vec<RerankResult>, PyErr>>();
                rt.spawn(async move {
                    let res = process_rerank_requests(
                        client,
                        query,
                        texts,
                        raw_scores,
                        return_text,
                        truncate,
                        truncation_direction, // Now this is a String
                        api_key,
                        api_base,
                        max_concurrent_requests,
                        batch_size,
                        timeout_duration,
                    )
                    .await;
                    let _ = tx.send(res);
                });
                rx.recv().map_err(|e| {
                    PyValueError::new_err(format!("Failed to receive rerank result: {}", e))
                })
            })?;

        Python::with_gil(|py| Ok(result_from_async_task?.into_py(py)))
    }

    #[pyo3(signature = (inputs, raw_scores = false, truncate = false, truncation_direction = "Right", max_concurrent_requests = 64, batch_size = 4, timeout_s = None))]
    fn classify(
        &self,
        py: Python,
        inputs: Vec<String>,
        raw_scores: bool,
        truncate: bool,
        truncation_direction: &str, // Changed name and type
        max_concurrent_requests: usize,
        batch_size: usize,
        timeout_s: Option<f64>,
    ) -> PyResult<PyObject> {
        SyncClient::validate_concurrency_parameters(max_concurrent_requests, batch_size)?;
        let timeout_duration = SyncClient::validate_and_get_timeout_duration(timeout_s)?;
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let api_base = self.api_base.clone();
        let rt = Arc::clone(&self.runtime);

        let truncation_direction = truncation_direction.to_string(); // Convert to String

        let result_from_async_task: Result<Vec<ClassificationResult>, PyErr> =
            py.allow_threads(move || {
                let (tx, rx) = mpsc::channel::<Result<Vec<ClassificationResult>, PyErr>>();
                rt.spawn(async move {
                    let res = process_classify_requests(
                        client,
                        inputs,
                        raw_scores,
                        truncate,
                        truncation_direction, // Now this is a String
                        api_key,
                        api_base,
                        max_concurrent_requests,
                        batch_size,
                        timeout_duration,
                    )
                    .await;
                    let _ = tx.send(res);
                });
                rx.recv().map_err(|e| {
                    PyValueError::new_err(format!("Failed to receive classify result: {}", e))
                })
            })?;

        Python::with_gil(|py| Ok(result_from_async_task?.into_py(py)))
    }
}

// --- Modification in send_single_embedding_request ---
async fn send_single_embedding_request(
    client: Client,
    texts_batch: Vec<String>,
    model: String,
    api_key: String,
    api_base: String,
    encoding_format: Option<String>,
    dimensions: Option<u32>,
    user: Option<String>,
    request_timeout: Duration, // New parameter for individual request timeout
) -> Result<OpenAIEmbeddingsResponse, PyErr> {
    let request_payload = OpenAIEmbeddingsRequest {
        input: texts_batch,
        model,
        encoding_format,
        dimensions,
        user,
    };

    let url = format!("{}/sync/v1/embeddings", api_base.trim_end_matches('/'));

    let response = client
        .post(&url)
        .bearer_auth(api_key)
        .json(&request_payload)
        .timeout(request_timeout) // Apply the timeout here.
        .send()
        .await
        .map_err(|e| PyValueError::new_err(format!("Request failed: {}", e)))?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(PyValueError::new_err(format!(
            "API request failed with status {}: {}",
            status, error_text
        )));
    }

    response
        .json::<OpenAIEmbeddingsResponse>()
        .await
        .map_err(|e| PyValueError::new_err(format!("Failed to parse response JSON: {}", e)))
}

// --- Modification in process_embeddings_requests ---
async fn process_embeddings_requests(
    client: Client,
    texts: Vec<String>,
    model: String,
    api_key: String,
    api_base: String,
    encoding_format: Option<String>,
    dimensions: Option<u32>,
    user: Option<String>,
    max_concurrent_requests: usize,
    batch_size: usize,
    overall_timeout: Duration, // New overall timeout parameter
) -> Result<OpenAIEmbeddingsResponse, PyErr> {
    let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));
    let mut tasks = Vec::new();
    let total_texts = texts.len();

    for (batch_index, user_text_batch) in texts.chunks(batch_size).enumerate() {
        let client_clone = client.clone();
        let model_clone = model.clone();
        let api_key_clone = api_key.clone();
        let api_base_clone = api_base.clone();
        let encoding_format_clone = encoding_format.clone();
        let dimensions_clone = dimensions;
        let user_clone = user.clone();
        let user_text_batch_owned = user_text_batch.to_vec();
        let semaphore_clone = Arc::clone(&semaphore);

        let current_batch_absolute_start_index = batch_index * batch_size;

        tasks.push(tokio::spawn(async move {
            let _permit = semaphore_clone
                .acquire()
                .await
                .expect("Semaphore acquire failed");
            let mut response = send_single_embedding_request(
                client_clone,
                user_text_batch_owned,
                model_clone,
                api_key_clone,
                api_base_clone,
                encoding_format_clone,
                dimensions_clone,
                user_clone,
                overall_timeout, // Use the overall timeout for the individual request
            )
            .await?;

            for item in &mut response.data {
                item.index += current_batch_absolute_start_index;
            }
            Result::<_, PyErr>::Ok((response.data, response.usage))
        }));
    }

    // Wrap joining tasks with overall_timeout, so the entire operation fails if not finished in time.
    match tokio::time::timeout(overall_timeout, join_all(tasks)).await {
        Ok(task_results) => {
            let mut all_embedding_data: Vec<OpenAIEmbeddingData> = Vec::with_capacity(total_texts);
            let mut aggregated_prompt_tokens: u32 = 0;
            let mut aggregated_total_tokens: u32 = 0;

            for result in task_results {
                match result {
                    Ok(Ok((data_part, usage_part))) => {
                        all_embedding_data.extend(data_part);
                        aggregated_prompt_tokens =
                            aggregated_prompt_tokens.saturating_add(usage_part.prompt_tokens);
                        aggregated_total_tokens =
                            aggregated_total_tokens.saturating_add(usage_part.total_tokens);
                    }
                    Ok(Err(py_err)) => return Err(py_err),
                    Err(join_err) => {
                        return Err(PyValueError::new_err(format!(
                            "Tokio task join error: {}",
                            join_err
                        )))
                    }
                }
            }
            all_embedding_data.sort_by_key(|d| d.index);
            Ok(OpenAIEmbeddingsResponse {
                object: "list".to_string(),
                data: all_embedding_data,
                model,
                usage: OpenAIUsage {
                    prompt_tokens: aggregated_prompt_tokens,
                    total_tokens: aggregated_total_tokens,
                },
            })
        }
        Err(_) => Err(PyValueError::new_err(format!(
            "Overall embedding operation timed out after {:?}",
            overall_timeout
        ))),
    }
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
    api_base: String,
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

    let url = format!("{}/sync/rerank", api_base.trim_end_matches('/'));

    let response = client
        .post(&url)
        .bearer_auth(api_key)
        .json(&request_payload)
        .timeout(request_timeout)
        .send()
        .await
        .map_err(|e| PyValueError::new_err(format!("Request failed: {}", e)))?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(PyValueError::new_err(format!(
            "API request failed with status {}: {}",
            status, error_text
        )));
    }

    response
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
    api_base: String,
    max_concurrent_requests: usize,
    batch_size: usize,
    overall_timeout: Duration,
) -> Result<Vec<RerankResult>, PyErr> {
    let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));
    let mut tasks = Vec::new();

    for (batch_index, texts_batch) in texts.chunks(batch_size).enumerate() {
        let client_clone = client.clone();
        let query_clone = query.clone();
        let api_key_clone = api_key.clone();
        let api_base_clone = api_base.clone();
        let truncation_direction_clone = truncation_direction.clone();
        let texts_batch_owned = texts_batch.to_vec();
        let semaphore_clone = Arc::clone(&semaphore);

        tasks.push(tokio::spawn(async move {
            let _permit = semaphore_clone
                .acquire()
                .await
                .expect("Semaphore acquire failed");
            send_single_rerank_request(
                client_clone,
                query_clone,
                texts_batch_owned,
                raw_scores,
                return_text,
                truncate,
                truncation_direction_clone,
                api_key_clone,
                api_base_clone,
                overall_timeout,
            )
            .await
        }));
    }

    match tokio::time::timeout(overall_timeout, join_all(tasks)).await {
        Ok(task_results) => {
            let mut all_results: Vec<RerankResult> = Vec::new();
            for result in task_results {
                match result {
                    Ok(Ok(mut batch_results)) => all_results.append(&mut batch_results),
                    Ok(Err(py_err)) => return Err(py_err),
                    Err(join_err) => {
                        return Err(PyValueError::new_err(format!(
                            "Tokio task join error: {}",
                            join_err
                        )))
                    }
                }
            }
            all_results.sort_by_key(|d| d.index);
            Ok(all_results)
        }
        Err(_) => Err(PyValueError::new_err(format!(
            "Overall rerank operation timed out after {:?}",
            overall_timeout
        ))),
    }
}

// --- Send Single Classify Request ---
async fn send_single_classify_request(
    client: Client,
    input: String,
    raw_scores: bool,
    truncate: bool,
    truncation_direction: String,
    api_key: String,
    api_base: String,
    request_timeout: Duration,
) -> Result<Vec<ClassificationResult>, PyErr> {
    let request_payload = ClassifyRequest {
        inputs: input,
        raw_scores,
        truncate,
        truncation_direction,
    };

    let url = format!("{}/sync/predict", api_base.trim_end_matches('/')); // route for classify is /predict

    let response = client
        .post(&url)
        .bearer_auth(api_key)
        .json(&request_payload)
        .timeout(request_timeout)
        .send()
        .await
        .map_err(|e| PyValueError::new_err(format!("Request failed: {}", e)))?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(PyValueError::new_err(format!(
            "API request failed with status {}: {}",
            status, error_text
        )));
    }

    response
        .json::<Vec<ClassificationResult>>()
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
    api_base: String,
    max_concurrent_requests: usize,
    batch_size: usize,
    overall_timeout: Duration,
) -> Result<Vec<ClassificationResult>, PyErr> {
    let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));
    let mut tasks = Vec::new();

    for input in inputs.chunks(batch_size) {
        let client_clone = client.clone();
        let api_key_clone = api_key.clone();
        let api_base_clone = api_base.clone();
        let truncation_direction_clone = truncation_direction.clone();
        let input_owned = input.join("\n"); // Assuming each input is a string, join for batch
        let semaphore_clone = Arc::clone(&semaphore);

        tasks.push(tokio::spawn(async move {
            let _permit = semaphore_clone
                .acquire()
                .await
                .expect("Semaphore acquire failed");
            send_single_classify_request(
                client_clone,
                input_owned,
                raw_scores,
                truncate,
                truncation_direction_clone,
                api_key_clone,
                api_base_clone,
                overall_timeout,
            )
            .await
        }));
    }

    match tokio::time::timeout(overall_timeout, join_all(tasks)).await {
        Ok(task_results) => {
            let mut all_results: Vec<ClassificationResult> = Vec::new();
            for result in task_results {
                match result {
                    Ok(Ok(mut batch_results)) => all_results.append(&mut batch_results),
                    Ok(Err(py_err)) => return Err(py_err),
                    Err(join_err) => {
                        return Err(PyValueError::new_err(format!(
                            "Tokio task join error: {}",
                            join_err
                        )))
                    }
                }
            }
            Ok(all_results)
        }
        Err(_) => Err(PyValueError::new_err(format!(
            "Overall classify operation timed out after {:?}",
            overall_timeout
        ))),
    }
}

// --- PyO3 Module Definition ---
#[pymodule]
fn truss_client_bei(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SyncClient>()?;
    m.add_class::<OpenAIEmbeddingsResponse>()?;
    m.add_class::<OpenAIEmbeddingData>()?;
    m.add_class::<OpenAIUsage>()?;
    m.add_class::<RerankResult>()?;
    m.add_class::<ClassificationResult>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

// --- Timeout Constants ---
const DEFAULT_REQUEST_TIMEOUT_S: f64 = 3600.0;
const MIN_REQUEST_TIMEOUT_S: f64 = 0.1;
const MAX_REQUEST_TIMEOUT_S: f64 = 3600.0;
const MAX_CONCURRENCY: usize = 512;
const MAX_BATCH_SIZE: usize = 128;

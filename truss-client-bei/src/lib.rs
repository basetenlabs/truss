// implements client for openai embeddings
// as python api that fans out to multiple requests
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::runtime::Runtime;
use tokio::sync::Semaphore;
use std::sync::Arc;
use futures::future::join_all;
use once_cell::sync::Lazy; // Import Lazy
use std::sync::mpsc; // Add this import
use std::time::Duration; // Add this for timeout support

// --- Global Tokio Runtime ---
static GLOBAL_RUNTIME: Lazy<Arc<Runtime>> = Lazy::new(|| {
    Arc::new(Runtime::new().expect("Failed to create global Tokio runtime"))
});

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
#[serde(untagged)] // Allows Serde to attempt deserializing as String, then Vec<f32>
enum EmbeddingVariant {
    Base64(String),
    FloatVector(Vec<f32>),
}

#[derive(Deserialize, Debug, Clone)]
#[pyclass]
struct OpenAIEmbeddingData {
    #[pyo3(get)]
    object: String,
    // The 'embedding' field in JSON will be deserialized into this internal field.
    // The original `#[pyo3(get)] embedding: Vec<f32>` is replaced by this mechanism.
    #[serde(rename = "embedding")]
    embedding_internal: EmbeddingVariant,
    #[pyo3(get)]
    index: usize,
}

// ADD this impl block for OpenAIEmbeddingData to provide a Python getter for the embedding
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

#[pyclass]
struct SyncClient {
    api_key: String,
    api_base: String,
    client: Client,
    runtime: Arc<Runtime>, // This will now hold a clone of the global runtime's Arc
}

impl SyncClient {
    fn get_api_key(api_key: Option<String>) -> Result<String, PyErr> {
        // Check if api_key is provided
        if let Some(key) = api_key {
            return Ok(key);
        }
        // Check if BASETEN_API_KEY is set in the environment
        if let Ok(key) = std::env::var("BASETEN_API_KEY") {
            return Ok(key);
        }
        // Check if OPENAI_API_KEY is set in the environment
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            return Ok(key);
        }
        // If none of the above, return an error
        Err(PyValueError::new_err("API key not provided and no environment variable `BASETEN_API_KEY` found"))
    }
}

#[pymethods]
impl SyncClient {
    #[new]
    #[pyo3(signature = (api_base, api_key = None ))]
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
        timeout_s: Option<f64>  // New timeout parameter
    ) -> PyResult<PyObject> {
        if input.is_empty() {
            return Err(PyValueError::new_err("Input list cannot be empty"));
        }
        if max_concurrent_requests == 0 || max_concurrent_requests > 256 {
            return Err(PyValueError::new_err("max_concurrent_requests must be greater than 0 and less than 256"));
        }
        if batch_size == 0 || batch_size > 256 {
            return Err(PyValueError::new_err("batch_size must be greater than 0 and less than 256"));
        }

        // Validate and set the request timeout.
        let resolved_timeout_s = timeout_s.unwrap_or(DEFAULT_REQUEST_TIMEOUT_S);
        if !(MIN_REQUEST_TIMEOUT_S..=MAX_REQUEST_TIMEOUT_S).contains(&resolved_timeout_s) {
            return Err(PyValueError::new_err(format!(
                "Timeout {:.3}s is outside the allowed range [{:.3}s, {:.3}s].",
                resolved_timeout_s, MIN_REQUEST_TIMEOUT_S, MAX_REQUEST_TIMEOUT_S
            )));
        }
        let timeout_duration = Duration::from_secs_f64(resolved_timeout_s);

        let model_string: String = model.to_string();

        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let api_base = self.api_base.clone();
        let rt = Arc::clone(&self.runtime);

        let result_from_async_task: Result<OpenAIEmbeddingsResponse, PyErr> = py.allow_threads(move || {
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
    request_timeout: Duration,  // New parameter for individual request timeout
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
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
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
            let _permit = semaphore_clone.acquire().await.expect("Semaphore acquire failed");
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
                        aggregated_prompt_tokens = aggregated_prompt_tokens.saturating_add(usage_part.prompt_tokens);
                        aggregated_total_tokens = aggregated_total_tokens.saturating_add(usage_part.total_tokens);
                    }
                    Ok(Err(py_err)) => return Err(py_err),
                    Err(join_err) => return Err(PyValueError::new_err(format!("Tokio task join error: {}", join_err))),
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

// --- PyO3 Module Definition ---
#[pymodule]
fn truss_client_bei(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SyncClient>()?;
    m.add_class::<OpenAIEmbeddingsResponse>()?;
    m.add_class::<OpenAIEmbeddingData>()?;
    m.add_class::<OpenAIUsage>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

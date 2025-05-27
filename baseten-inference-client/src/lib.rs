// implements client for openai embeddings
// as python api that fans out to multiple requests
use futures::future::join_all;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use once_cell::sync::Lazy; // Import Lazy
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3_async_runtimes;
use pythonize::{depythonize, pythonize};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue; // For handling untyped JSON
use std::sync::atomic::{AtomicBool, Ordering}; // Add this
use std::sync::Arc;
use std::time::Duration;
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
    object: String,
    data: Vec<RerankResult>,
}

#[pymethods]
impl RerankResponse {
    #[new]
    fn new(data: Vec<RerankResult>) -> Self {
        RerankResponse {
            object: "list".to_string(),
            data,
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
    object: String,
    data: Vec<Vec<ClassificationResult>>, // Changed to Vec<Vec<ClassificationResult>>
}

#[pymethods]
impl ClassificationResponse {
    #[new]
    fn new(data: Vec<Vec<ClassificationResult>>) -> Self {
        // Changed parameter type
        ClassificationResponse {
            object: "list".to_string(),
            data,
        }
    }
}

// --- InferenceClient Definition ---
#[pyclass]
struct InferenceClient {
    api_key: String,
    api_base: String,
    client: Client,
    runtime: Arc<Runtime>,
}

impl InferenceClient {
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
impl InferenceClient {
    #[new]
    #[pyo3(signature = (api_base, api_key = None))]
    fn new(api_base: String, api_key: Option<String>) -> PyResult<Self> {
        let api_key = InferenceClient::get_api_key(api_key)?;
        Ok(InferenceClient {
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

    #[pyo3(signature = (input, model, encoding_format = None, dimensions = None, user = None, max_concurrent_requests = DEFAULT_CONCURRENCY, batch_size = DEFAULT_BATCH_SIZE, timeout_s = DEFAULT_REQUEST_TIMEOUT_S))]
    fn embed(
        &self,
        py: Python,
        input: Vec<String>,
        model: String, // model is already String
        encoding_format: Option<String>,
        dimensions: Option<u32>,
        user: Option<String>,
        max_concurrent_requests: usize,
        batch_size: usize,
        timeout_s: f64,
    ) -> PyResult<PyObject> {
        if input.is_empty() {
            return Err(PyValueError::new_err("Input list cannot be empty"));
        }
        InferenceClient::validate_concurrency_parameters(max_concurrent_requests, batch_size)?;
        let timeout_duration = InferenceClient::validate_and_get_timeout_duration(timeout_s)?;

        let client_clone = self.client.clone();
        let api_key_clone = self.api_key.clone();
        let api_base_clone = self.api_base.clone();
        let rt = Arc::clone(&self.runtime);

        // input, model, encoding_format, dimensions, user will be moved into the closures
        // by the `move` keywords.

        let result_from_async_task: Result<OpenAIEmbeddingsResponse, PyErr> =
            py.allow_threads(move || {
                let (tx, rx) =
                    std::sync::mpsc::channel::<Result<OpenAIEmbeddingsResponse, PyErr>>();

                rt.spawn(async move {
                    let res = process_embeddings_requests(
                        client_clone,
                        input, // Use directly
                        model, // Use directly
                        api_key_clone,
                        api_base_clone,
                        encoding_format, // Use directly
                        dimensions,      // Use directly
                        user,            // Use directly
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

        let successful_response = result_from_async_task?;
        Python::with_gil(|py_gil| Ok(successful_response.into_py(py_gil)))
    }

    #[pyo3(name = "aembed", signature = (input, model, encoding_format = None, dimensions = None, user = None, max_concurrent_requests = DEFAULT_CONCURRENCY, batch_size = DEFAULT_BATCH_SIZE, timeout_s = DEFAULT_REQUEST_TIMEOUT_S))]
    fn aembed<'py>(
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
        InferenceClient::validate_concurrency_parameters(max_concurrent_requests, batch_size)?;
        let timeout_duration = InferenceClient::validate_and_get_timeout_duration(timeout_s)?;

        let client_clone = self.client.clone();
        let api_key_clone = self.api_key.clone();
        let api_base_clone = self.api_base.clone();
        // input, model, encoding_format, dimensions, user will be moved into the async block.

        let future = async move {
            process_embeddings_requests(
                client_clone,
                input,
                model,
                api_key_clone,
                api_base_clone,
                encoding_format,
                dimensions,
                user,
                max_concurrent_requests,
                batch_size,
                timeout_duration,
            )
            .await
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
    ) -> PyResult<PyObject> {
        if texts.is_empty() {
            return Err(PyValueError::new_err("Texts list cannot be empty"));
        }
        InferenceClient::validate_concurrency_parameters(max_concurrent_requests, batch_size)?;
        let timeout_duration = InferenceClient::validate_and_get_timeout_duration(timeout_s)?;
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let api_base = self.api_base.clone();
        let rt = Arc::clone(&self.runtime);
        let truncation_direction = truncation_direction.to_string();

        let result_from_async_task: Result<RerankResponse, PyErr> = py.allow_threads(move || {
            let (tx, rx) = std::sync::mpsc::channel::<Result<RerankResponse, PyErr>>();
            rt.spawn(async move {
                let res = process_rerank_requests(
                    client,
                    query,
                    texts,
                    raw_scores,
                    return_text,
                    truncate,
                    truncation_direction,
                    api_key,
                    api_base,
                    max_concurrent_requests,
                    batch_size,
                    timeout_duration,
                )
                .await;
                let _ = tx.send(res);
            });
            rx.recv() // Returns Result<Result<RerankResponse, PyErr>, RecvError>
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "Failed to receive rerank result (channel error): {}",
                        e
                    ))
                })
                .and_then(|inner_result| inner_result) // Flattens to Result<RerankResponse, PyErr>
        });
        let successful_response = result_from_async_task?;
        Python::with_gil(|py| Ok(successful_response.into_py(py)))
    }

    #[pyo3(name = "arerank", signature = (query, texts, raw_scores = false, return_text = false, truncate = false, truncation_direction = "Right", max_concurrent_requests = DEFAULT_CONCURRENCY, batch_size = DEFAULT_BATCH_SIZE, timeout_s = DEFAULT_REQUEST_TIMEOUT_S))]
    fn arerank<'py>(
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
        InferenceClient::validate_concurrency_parameters(max_concurrent_requests, batch_size)?;
        let timeout_duration = InferenceClient::validate_and_get_timeout_duration(timeout_s)?;

        let client_clone = self.client.clone();
        let api_key_clone = self.api_key.clone();
        let api_base_clone = self.api_base.clone();
        let truncation_direction = truncation_direction.to_string(); // Convert to String

        let future = async move {
            process_rerank_requests(
                client_clone,
                query,
                texts,
                raw_scores,
                return_text,
                truncate,
                truncation_direction,
                api_key_clone,
                api_base_clone,
                max_concurrent_requests,
                batch_size,
                timeout_duration,
            )
            .await
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
    ) -> PyResult<PyObject> {
        if inputs.is_empty() {
            return Err(PyValueError::new_err("Inputs list cannot be empty"));
        }
        InferenceClient::validate_concurrency_parameters(max_concurrent_requests, batch_size)?;
        let timeout_duration = InferenceClient::validate_and_get_timeout_duration(timeout_s)?;
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let api_base = self.api_base.clone();
        let rt = Arc::clone(&self.runtime);
        let truncation_direction = truncation_direction.to_string();

        let result_from_async_task: Result<ClassificationResponse, PyErr> =
            py.allow_threads(move || {
                let (tx, rx) = std::sync::mpsc::channel::<Result<ClassificationResponse, PyErr>>();
                rt.spawn(async move {
                    let res = process_classify_requests(
                        client,
                        inputs,
                        raw_scores,
                        truncate,
                        truncation_direction,
                        api_key,
                        api_base,
                        max_concurrent_requests,
                        batch_size,
                        timeout_duration,
                    )
                    .await;
                    let _ = tx.send(res);
                });
                rx.recv() // Returns Result<Result<ClassificationResponse, PyErr>, RecvError>
                    .map_err(|e| {
                        PyValueError::new_err(format!(
                            "Failed to receive classify result (channel error): {}",
                            e
                        ))
                    })
                    .and_then(|inner_result| inner_result) // Flattens to Result<ClassificationResponse, PyErr>
            });

        Python::with_gil(|py| Ok(result_from_async_task?.into_py(py)))
    }

    #[pyo3(name = "aclassify", signature = (inputs, raw_scores = false, truncate = false, truncation_direction = "Right", max_concurrent_requests = DEFAULT_CONCURRENCY, batch_size = DEFAULT_BATCH_SIZE, timeout_s = DEFAULT_REQUEST_TIMEOUT_S))]
    fn aclassify<'py>(
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
        InferenceClient::validate_concurrency_parameters(max_concurrent_requests, batch_size)?;
        let timeout_duration = InferenceClient::validate_and_get_timeout_duration(timeout_s)?;

        let client_clone = self.client.clone();
        let api_key_clone = self.api_key.clone();
        let api_base_clone = self.api_base.clone();
        let truncation_direction = truncation_direction.to_string(); // Convert to String

        let future = async move {
            process_classify_requests(
                client_clone,
                inputs,
                raw_scores,
                truncate,
                truncation_direction,
                api_key_clone,
                api_base_clone,
                max_concurrent_requests,
                batch_size,
                timeout_duration,
            )
            .await
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
    ) -> PyResult<PyObject> {
        if payloads.is_empty() {
            return Err(PyValueError::new_err("Payloads list cannot be empty"));
        }
        InferenceClient::validate_concurrency_parameters(max_concurrent_requests, 1)?; // Batch size is effectively 1
        let timeout_duration = InferenceClient::validate_and_get_timeout_duration(timeout_s)?;

        // Depythonize all payloads in the current thread (GIL is held)
        let mut payloads_json: Vec<JsonValue> = Vec::with_capacity(payloads.len());
        for (idx, py_obj) in payloads.into_iter().enumerate() {
            // Bind PyObject to current GIL lifetime to get a Bound object for depythonize
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
        let api_base = self.api_base.clone();
        let rt = Arc::clone(&self.runtime);

        // The async task now receives Vec<JsonValue> and returns Result<Vec<JsonValue>, PyErr>
        let result_from_async_task: Result<Vec<JsonValue>, PyErr> = py.allow_threads(move || {
            let (tx, rx) = std::sync::mpsc::channel::<Result<Vec<JsonValue>, PyErr>>();
            rt.spawn(async move {
                let res = process_batch_post_requests(
                    client,
                    url_path,
                    payloads_json, // Pass depythonized JSON values
                    api_key,
                    api_base,
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

        let response_json_values = result_from_async_task?;

        // Pythonize all results in the current thread (GIL is held)
        let mut results_py: Vec<PyObject> = Vec::with_capacity(response_json_values.len());
        for (idx, json_val) in response_json_values.into_iter().enumerate() {
            let py_obj_bound = pythonize(py, &json_val).map_err(|e| {
                PyValueError::new_err(format!(
                    "Failed to pythonize response at index {}: {}",
                    idx, e
                ))
            })?;
            // Convert Bound<'_, PyAny> to PyObject
            results_py.push(py_obj_bound.to_object(py));
        }

        // Use the updated PyList::new_bound or PyList::new as per PyO3 v0.21+
        // PyList::new_bound is suitable here for an iterable of PyObjects.
        let py_object_list = PyList::new_bound(py, &results_py);
        Ok(py_object_list.into())
    }

    #[pyo3(name = "abatch_post", signature = (url_path, payloads, max_concurrent_requests = DEFAULT_CONCURRENCY, timeout_s = DEFAULT_REQUEST_TIMEOUT_S))]
    fn abatch_post<'py>(
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
        InferenceClient::validate_concurrency_parameters(max_concurrent_requests, 1)?; // Batch size is effectively 1
        let timeout_duration = InferenceClient::validate_and_get_timeout_duration(timeout_s)?;

        // Depythonize all payloads in the current thread (GIL is held by `py` argument)
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
        let api_base_clone = self.api_base.clone();

        let future = async move {
            let response_json_values = process_batch_post_requests(
                client_clone,
                url_path,
                payloads_json,
                api_key_clone,
                api_base_clone,
                max_concurrent_requests,
                timeout_duration,
            )
            .await?; // Propagates PyErr from process_batch_post_requests

            // Pythonize results - this part needs the GIL.
            // The `future_into_py` function ensures the future is polled in a context
            // where acquiring the GIL is possible if the future's output type requires it
            // for conversion (like our PyResult<PyObject> here).
            Python::with_gil(|py_gil| {
                let mut results_py: Vec<PyObject> = Vec::with_capacity(response_json_values.len());
                for (idx, json_val) in response_json_values.into_iter().enumerate() {
                    let py_obj_bound = pythonize(py_gil, &json_val).map_err(|e| {
                        PyValueError::new_err(format!(
                            "Failed to pythonize response at index {}: {}",
                            idx, e
                        ))
                    })?;
                    results_py.push(py_obj_bound.to_object(py_gil));
                }
                Ok(PyList::new_bound(py_gil, &results_py).to_object(py_gil))
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

    let url = format!("{}/v1/embeddings", api_base.trim_end_matches('/'));

    let response = client
        .post(&url)
        .bearer_auth(api_key)
        .json(&request_payload)
        .timeout(request_timeout) // Apply the timeout here.
        .send()
        .await
        .map_err(|e| PyValueError::new_err(format!("Request failed: {}", e)))?;

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
    api_base: String,
    encoding_format: Option<String>,
    dimensions: Option<u32>,
    user: Option<String>,
    max_concurrent_requests: usize,
    batch_size: usize,
    request_timeout_duration: Duration,
) -> Result<OpenAIEmbeddingsResponse, PyErr> {
    let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));
    let mut tasks = Vec::new();
    let total_texts = texts.len();
    let cancel_token = Arc::new(AtomicBool::new(false)); // This is the per-operation token
    let model_for_response = model.clone();

    for (batch_index, user_text_batch) in texts.chunks(batch_size).enumerate() {
        let client_clone = client.clone();
        let model_for_task = model.clone();
        let api_key_clone = api_key.clone();
        let api_base_clone = api_base.clone();
        let encoding_format_clone = encoding_format.clone();
        let dimensions_clone = dimensions;
        let user_clone = user.clone();
        let user_text_batch_owned = user_text_batch.to_vec();
        let semaphore_clone = Arc::clone(&semaphore);
        let cancel_token_clone = Arc::clone(&cancel_token); // Local cancel token
        let individual_request_timeout = request_timeout_duration;
        let current_batch_absolute_start_index = batch_index * batch_size;

        tasks.push(tokio::spawn(async move {
            let _permit =
                acquire_permit_or_cancel(semaphore_clone, cancel_token_clone.clone()).await?;

            let result = send_single_embedding_request(
                client_clone,
                user_text_batch_owned,
                model_for_task,
                api_key_clone,
                api_base_clone,
                encoding_format_clone,
                dimensions_clone,
                user_clone,
                individual_request_timeout,
            )
            .await;

            match result {
                Ok(mut response) => {
                    for item in &mut response.data {
                        item.index += current_batch_absolute_start_index;
                    }
                    Ok((response.data, response.usage))
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
    let mut first_error: Option<PyErr> = None;

    for result in task_join_results {
        if let Some((data_part, usage_part)) =
            process_task_outcome(result, &mut first_error, &cancel_token)
        {
            all_embedding_data.extend(data_part);
            aggregated_prompt_tokens =
                aggregated_prompt_tokens.saturating_add(usage_part.prompt_tokens);
            aggregated_total_tokens =
                aggregated_total_tokens.saturating_add(usage_part.total_tokens);
        }
    }

    if let Some(err) = first_error {
        return Err(err);
    }

    all_embedding_data.sort_by_key(|d| d.index);
    Ok(OpenAIEmbeddingsResponse {
        object: "list".to_string(),
        data: all_embedding_data,
        model: model_for_response,
        usage: OpenAIUsage {
            prompt_tokens: aggregated_prompt_tokens,
            total_tokens: aggregated_total_tokens,
        },
    })
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

    let url = format!("{}/rerank", api_base.trim_end_matches('/'));

    let response = client
        .post(&url)
        .bearer_auth(api_key)
        .json(&request_payload)
        .timeout(request_timeout)
        .send()
        .await
        .map_err(|e| PyValueError::new_err(format!("Request failed: {}", e)))?;

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
    api_base: String,
    max_concurrent_requests: usize,
    batch_size: usize,
    request_timeout_duration: Duration,
) -> Result<RerankResponse, PyErr> {
    let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));
    let mut tasks = Vec::new();
    let cancel_token = Arc::new(AtomicBool::new(false));

    for (_batch_index, texts_batch) in texts.chunks(batch_size).enumerate() {
        let client_clone = client.clone();
        let query_clone = query.clone();
        let api_key_clone = api_key.clone();
        let api_base_clone = api_base.clone();
        let truncation_direction_clone = truncation_direction.clone();
        let texts_batch_owned = texts_batch.to_vec();
        let semaphore_clone = Arc::clone(&semaphore);
        let cancel_token_clone = Arc::clone(&cancel_token);
        let individual_request_timeout = request_timeout_duration;

        tasks.push(tokio::spawn(async move {
            let _permit =
                acquire_permit_or_cancel(semaphore_clone, cancel_token_clone.clone()).await?;

            let result = send_single_rerank_request(
                client_clone,
                query_clone,
                texts_batch_owned,
                raw_scores,
                return_text,
                truncate,
                truncation_direction_clone,
                api_key_clone,
                api_base_clone,
                individual_request_timeout,
            )
            .await;

            match result {
                Ok(batch_results) => Ok(batch_results),
                Err(e) => {
                    cancel_token_clone.store(true, Ordering::SeqCst);
                    Err(e)
                }
            }
        }));
    }

    let task_join_results = join_all(tasks).await;

    let mut all_results: Vec<RerankResult> = Vec::new();
    let mut first_error: Option<PyErr> = None;

    for result in task_join_results {
        if let Some(mut batch_results_part) =
            process_task_outcome(result, &mut first_error, &cancel_token)
        {
            all_results.append(&mut batch_results_part);
        }
    }

    if let Some(err) = first_error {
        return Err(err);
    }

    all_results.sort_by_key(|d| d.index);
    Ok(RerankResponse {
        object: "list".to_string(),
        data: all_results,
    })
}

// --- Send Single Classify Request ---
async fn send_single_classify_request(
    client: Client,
    inputs: Vec<Vec<String>>,
    raw_scores: bool,
    truncate: bool,
    truncation_direction: String,
    api_key: String,
    api_base: String,
    request_timeout: Duration,
) -> Result<Vec<Vec<ClassificationResult>>, PyErr> {
    let request_payload = ClassifyRequest {
        inputs,
        raw_scores,
        truncate,
        truncation_direction,
    };

    let url = format!("{}/predict", api_base.trim_end_matches('/'));

    let response = client
        .post(&url)
        .bearer_auth(api_key)
        .json(&request_payload)
        .timeout(request_timeout)
        .send()
        .await
        .map_err(|e| PyValueError::new_err(format!("Request failed: {}", e)))?;

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
    inputs: Vec<String>, // Python provides Vec<String>
    raw_scores: bool,
    truncate: bool,
    truncation_direction: String,
    api_key: String,
    api_base: String,
    max_concurrent_requests: usize,
    batch_size: usize,
    request_timeout_duration: Duration,
) -> Result<ClassificationResponse, PyErr> {
    let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));
    let mut tasks = Vec::new();
    let cancel_token = Arc::new(AtomicBool::new(false));

    for input_chunk_slice in inputs.chunks(batch_size) {
        let client_clone = client.clone();
        let api_key_clone = api_key.clone();
        let api_base_clone = api_base.clone();
        let truncation_direction_clone = truncation_direction.clone();
        let inputs_for_api_owned: Vec<Vec<String>> =
            input_chunk_slice.iter().map(|s| vec![s.clone()]).collect();
        let semaphore_clone = Arc::clone(&semaphore);
        let cancel_token_clone = Arc::clone(&cancel_token);
        let individual_request_timeout = request_timeout_duration;

        tasks.push(tokio::spawn(async move {
            let _permit =
                acquire_permit_or_cancel(semaphore_clone, cancel_token_clone.clone()).await?;

            let result = send_single_classify_request(
                client_clone,
                inputs_for_api_owned,
                raw_scores,
                truncate,
                truncation_direction_clone,
                api_key_clone,
                api_base_clone,
                individual_request_timeout,
            )
            .await;

            match result {
                Ok(batch_results) => Ok(batch_results),
                Err(e) => {
                    cancel_token_clone.store(true, Ordering::SeqCst);
                    Err(e)
                }
            }
        }));
    }

    let task_join_results = join_all(tasks).await;

    let mut all_results: Vec<Vec<ClassificationResult>> = Vec::new();
    let mut first_error: Option<PyErr> = None;

    for result in task_join_results {
        if let Some(mut batch_results_part) =
            process_task_outcome(result, &mut first_error, &cancel_token)
        {
            all_results.append(&mut batch_results_part);
        }
    }

    if let Some(err) = first_error {
        return Err(err);
    }

    Ok(ClassificationResponse {
        object: "list".to_string(),
        data: all_results,
    })
}

// --- Send Single Batch Post Request ---
// Now takes JsonValue and returns JsonValue
async fn send_single_batch_post_request(
    client: Client,
    full_url: String,
    payload_json: JsonValue,
    api_key: String,
    request_timeout: Duration,
) -> Result<JsonValue, PyErr> {
    // No depythonize here

    let response = client
        .post(&full_url)
        .bearer_auth(api_key)
        .json(&payload_json)
        .timeout(request_timeout)
        .send()
        .await
        .map_err(|e| PyValueError::new_err(format!("Request failed: {}", e)))?;

    let successful_response = ensure_successful_response(response).await?;

    // Get response as serde_json::Value
    let response_json_value: JsonValue = successful_response
        .json::<JsonValue>()
        .await
        .map_err(|e| PyValueError::new_err(format!("Failed to parse response JSON: {}", e)))?;

    // No pythonize here, return JsonValue
    Ok(response_json_value)
}

// --- Process Batch Post Requests ---
// Now takes Vec<JsonValue> and returns Result<Vec<JsonValue>, PyErr>
async fn process_batch_post_requests(
    client: Client,
    url_path: String,
    payloads_json: Vec<JsonValue>, // Takes Vec<JsonValue>
    api_key: String,
    api_base: String,
    max_concurrent_requests: usize,
    request_timeout_duration: Duration,
) -> Result<Vec<JsonValue>, PyErr> {
    // Returns Vec<JsonValue>
    let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));
    let mut tasks = Vec::new();
    let cancel_token = Arc::new(AtomicBool::new(false));
    let total_payloads = payloads_json.len();

    for (index, payload_item_json) in payloads_json.into_iter().enumerate() {
        // Iterate over JsonValue
        let client_clone = client.clone();
        let api_key_clone = api_key.clone();
        let api_base_clone = api_base.clone();
        let url_path_clone = url_path.clone();
        let semaphore_clone = Arc::clone(&semaphore);
        let cancel_token_clone = Arc::clone(&cancel_token);
        let individual_request_timeout = request_timeout_duration;

        // payload_item_json is moved into its own task
        tasks.push(tokio::spawn(async move {
            let permit_guard =
                acquire_permit_or_cancel(semaphore_clone, cancel_token_clone.clone()).await?;

            let full_url = format!(
                "{}/{}",
                api_base_clone.trim_end_matches('/'),
                url_path_clone.trim_start_matches('/')
            );

            let result = send_single_batch_post_request(
                client_clone,
                full_url,
                payload_item_json, // Pass JsonValue
                api_key_clone,
                individual_request_timeout,
            )
            .await;

            drop(permit_guard);

            match result {
                Ok(response_json_value) => Ok((index, response_json_value)), // Return with original index and JsonValue
                Err(e) => {
                    cancel_token_clone.store(true, Ordering::SeqCst);
                    Err(e)
                }
            }
        }));
    }

    let task_join_results = join_all(tasks).await;
    let mut indexed_results: Vec<(usize, JsonValue)> = Vec::with_capacity(total_payloads); // Stores JsonValue
    let mut first_error: Option<PyErr> = None;

    for result in task_join_results {
        // D is (usize, JsonValue)
        if let Some(indexed_data_part) =
            process_task_outcome(result, &mut first_error, &cancel_token)
        {
            indexed_results.push(indexed_data_part);
        }
    }

    if let Some(err) = first_error {
        return Err(err);
    }

    indexed_results.sort_by_key(|&(original_index, _)| original_index);

    let final_results: Vec<JsonValue> = indexed_results.into_iter().map(|(_, val)| val).collect(); // Collect JsonValue

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

// --- PyO3 Module Definition ---
#[pymodule]
fn baseten_inference_client(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<InferenceClient>()?;
    m.add_class::<OpenAIEmbeddingsResponse>()?;
    m.add_class::<OpenAIEmbeddingData>()?;
    m.add_class::<OpenAIUsage>()?;
    m.add_class::<RerankResult>()?;
    m.add_class::<RerankResponse>()?; // Add RerankResponse
    m.add_class::<ClassificationResult>()?;
    m.add_class::<ClassificationResponse>()?; // Add ClassificationResponse
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

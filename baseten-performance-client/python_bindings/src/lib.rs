use baseten_performance_client_core::*;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use once_cell::sync::Lazy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_async_runtimes;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::runtime::Runtime;

// --- Global Tokio Runtime (Python-specific) ---
static CTRL_C_RECEIVED: AtomicBool = AtomicBool::new(false);

static GLOBAL_RUNTIME: Lazy<Arc<Runtime>> = Lazy::new(|| {
    let runtime = Arc::new(Runtime::new().expect("Failed to create global Tokio runtime"));
    let runtime_clone_for_signal: Arc<Runtime> = Arc::clone(&runtime);
    // Spawn a task to listen for Ctrl+C
    runtime_clone_for_signal.spawn(async {
        if tokio::signal::ctrl_c().await.is_ok() {
            CTRL_C_RECEIVED.store(true, Ordering::SeqCst);
        }
    });
    runtime
});

// Import Python exceptions
pyo3::import_exception!(requests, HTTPError);
pyo3::import_exception!(requests, Timeout);

// --- Python API Types (keeping original names) ---
#[derive(Debug, Clone)]
#[pyclass]
struct OpenAIEmbeddingData {
    #[pyo3(get)]
    object: String,
    embedding_internal: CoreEmbeddingVariant,
    #[pyo3(get)]
    index: usize,
}

#[pymethods]
impl OpenAIEmbeddingData {
    #[getter]
    #[allow(deprecated)]
    fn embedding(&self, py: Python) -> PyObject {
        match &self.embedding_internal {
            CoreEmbeddingVariant::Base64(s) => s.to_object(py),
            CoreEmbeddingVariant::FloatVector(v) => v.to_object(py),
        }
    }
}

impl From<CoreOpenAIEmbeddingData> for OpenAIEmbeddingData {
    fn from(core: CoreOpenAIEmbeddingData) -> Self {
        OpenAIEmbeddingData {
            object: core.object,
            embedding_internal: core.embedding_internal,
            index: core.index,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
struct OpenAIUsage {
    #[pyo3(get)]
    prompt_tokens: u32,
    #[pyo3(get)]
    total_tokens: u32,
}

impl From<CoreOpenAIUsage> for OpenAIUsage {
    fn from(core: CoreOpenAIUsage) -> Self {
        OpenAIUsage {
            prompt_tokens: core.prompt_tokens,
            total_tokens: core.total_tokens,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(get_all)]
struct OpenAIEmbeddingsResponse {
    object: String,
    data: Vec<OpenAIEmbeddingData>,
    model: String,
    usage: OpenAIUsage,
    total_time: f64,
    individual_request_times: Vec<f64>,
}

impl From<CoreOpenAIEmbeddingsResponse> for OpenAIEmbeddingsResponse {
    fn from(core: CoreOpenAIEmbeddingsResponse) -> Self {
        OpenAIEmbeddingsResponse {
            object: core.object,
            data: core
                .data
                .into_iter()
                .map(OpenAIEmbeddingData::from)
                .collect(),
            model: core.model,
            usage: OpenAIUsage::from(core.usage),
            total_time: core.total_time,
            individual_request_times: core.individual_request_times,
        }
    }
}

#[pymethods]
impl OpenAIEmbeddingsResponse {
    /// Converts the embeddings data into a 2D NumPy array.
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if self.data.is_empty() {
            return Err(PyValueError::new_err(
                "Cannot convert to array: contains no embedding responses",
            ));
        }

        let num_embeddings = self.data.len();
        let mut embedding_dim_opt: Option<usize> = None;
        let mut flat_data: Vec<f32> = Vec::new();

        for (idx, item_data) in self.data.iter().enumerate() {
            match &item_data.embedding_internal {
                CoreEmbeddingVariant::FloatVector(v) => {
                    if idx == 0 {
                        let dim = v.len();
                        embedding_dim_opt = Some(dim);
                        if dim > 0 {
                            flat_data.reserve_exact(num_embeddings * dim);
                        }
                    }

                    let expected_dim = embedding_dim_opt.unwrap();
                    if v.len() != expected_dim {
                        return Err(PyValueError::new_err(format!(
                            "All embeddings must have the same dimension. Expected {} but got {} at index {}.",
                            expected_dim, v.len(), item_data.index
                        )));
                    }
                    flat_data.extend_from_slice(v);
                }
                CoreEmbeddingVariant::Base64(_) => {
                    return Err(PyValueError::new_err(format!(
                        "Cannot convert to array: found Base64 encoded embedding at index {}. Only float vectors are supported.",
                        item_data.index
                    )));
                }
            }
        }

        let final_embedding_dim = embedding_dim_opt.unwrap_or(0);
        let array = Array2::from_shape_vec((num_embeddings, final_embedding_dim), flat_data)
            .map_err(|e| {
                PyValueError::new_err(format!("Failed to create ndarray from embeddings: {}", e))
            })?;
        #[allow(deprecated)]
        Ok(array.into_pyarray_bound(py))
    }
}

// --- Rerank Response Types ---
#[derive(Debug, Clone)]
#[pyclass]
struct RerankResult {
    #[pyo3(get)]
    index: usize,
    #[pyo3(get)]
    score: f64,
    #[pyo3(get)]
    text: Option<String>,
}

impl From<CoreRerankResult> for RerankResult {
    fn from(core: CoreRerankResult) -> Self {
        RerankResult {
            index: core.index,
            score: core.score,
            text: core.text,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(get_all, frozen)]
struct RerankResponse {
    object: String,
    data: Vec<RerankResult>,
    total_time: f64,
    individual_request_times: Vec<f64>,
}

impl From<CoreRerankResponse> for RerankResponse {
    fn from(core: CoreRerankResponse) -> Self {
        RerankResponse {
            object: core.object,
            data: core.data.into_iter().map(RerankResult::from).collect(),
            total_time: core.total_time,
            individual_request_times: core.individual_request_times,
        }
    }
}

#[pymethods]
impl RerankResponse {
    #[new]
    #[pyo3(signature = (data, total_time, individual_request_times))]
    fn new(data: Vec<RerankResult>, total_time: f64, individual_request_times: Vec<f64>) -> Self {
        RerankResponse {
            object: "list".to_string(),
            data,
            total_time,
            individual_request_times,
        }
    }
}

// --- Classification Response Types ---
#[derive(Debug, Clone)]
#[pyclass]
struct ClassificationResult {
    #[pyo3(get)]
    label: String,
    #[pyo3(get)]
    score: f64,
}

impl From<CoreClassificationResult> for ClassificationResult {
    fn from(core: CoreClassificationResult) -> Self {
        ClassificationResult {
            label: core.label,
            score: core.score,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(get_all, frozen)]
struct ClassificationResponse {
    object: String,
    data: Vec<Vec<ClassificationResult>>,
    total_time: f64,
    individual_request_times: Vec<f64>,
}

impl From<CoreClassificationResponse> for ClassificationResponse {
    fn from(core: CoreClassificationResponse) -> Self {
        ClassificationResponse {
            object: core.object,
            data: core
                .data
                .into_iter()
                .map(|batch| batch.into_iter().map(ClassificationResult::from).collect())
                .collect(),
            total_time: core.total_time,
            individual_request_times: core.individual_request_times,
        }
    }
}

#[pymethods]
impl ClassificationResponse {
    #[new]
    #[pyo3(signature = (data, total_time, individual_request_times))]
    fn new(
        data: Vec<Vec<ClassificationResult>>,
        total_time: f64,
        individual_request_times: Vec<f64>,
    ) -> Self {
        ClassificationResponse {
            object: "list".to_string(),
            data,
            total_time,
            individual_request_times,
        }
    }
}

// --- Batch Post Response Types ---
#[pyclass(get_all)]
struct BatchPostResponse {
    data: Vec<PyObject>,
    total_time: f64,
    individual_request_times: Vec<f64>,
    response_headers: Vec<PyObject>,
}

#[pyclass]
struct PerformanceClient {
    core_client: PerformanceClientCore,
    runtime: Arc<Runtime>,
}

impl PerformanceClient {
    fn convert_core_error_to_py_err(err: ClientError) -> PyErr {
        match err {
            ClientError::Timeout(msg) => PyErr::new::<Timeout, _>((408, msg)),
            ClientError::Http { status, message } => PyErr::new::<HTTPError, _>((status, message)),
            ClientError::InvalidParameter(msg) => PyValueError::new_err(msg),
            ClientError::Serialization(msg) => PyValueError::new_err(msg),
            ClientError::Network(msg) => PyValueError::new_err(msg),
            ClientError::Cancellation(msg) => PyValueError::new_err(msg),
            ClientError::Connect(msg) => PyValueError::new_err(msg),
        }
    }
}

#[pymethods]
impl PerformanceClient {
    #[new]
    #[pyo3(signature = (base_url, api_key = None, http_version = 1))]
    fn new(base_url: String, api_key: Option<String>, http_version: u8) -> PyResult<Self> {
        let core_client = PerformanceClientCore::new(base_url, api_key, http_version)
            .map_err(Self::convert_core_error_to_py_err)?;

        Ok(PerformanceClient {
            core_client,
            runtime: Arc::clone(&GLOBAL_RUNTIME),
        })
    }

    #[getter]
    fn api_key(&self) -> PyResult<String> {
        Ok(self.core_client.api_key.clone())
    }

    #[pyo3(signature = (input, model, encoding_format = None, dimensions = None, user = None, max_concurrent_requests = DEFAULT_CONCURRENCY, batch_size = DEFAULT_BATCH_SIZE, timeout_s = DEFAULT_REQUEST_TIMEOUT_S, max_chars_per_request = None, hedge_delay = None))]
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
        max_chars_per_request: Option<usize>,
        hedge_delay: Option<f64>,
    ) -> PyResult<OpenAIEmbeddingsResponse> {
        if input.is_empty() {
            return Err(PyValueError::new_err("Input list cannot be empty"));
        }

        let core_client = self.core_client.clone();
        let rt: Arc<Runtime> = Arc::clone(&self.runtime);

        let result_from_async_task = py.allow_threads(move || {
            let (tx, rx) = std::sync::mpsc::channel();

            rt.spawn(async move {
                let res = core_client
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
                    .await;
                let _ = tx.send(res);
            });

            match rx.recv() {
                Ok(inner_result) => inner_result.map_err(Self::convert_core_error_to_py_err),
                Err(e) => Err(PyValueError::new_err(format!(
                    "Failed to receive result from async task (channel error): {}",
                    e
                ))),
            }
        })?;

        let (core_response, batch_durations, total_time_val) = result_from_async_task;
        let individual_times_val: Vec<f64> = batch_durations
            .into_iter()
            .map(|d| d.as_secs_f64())
            .collect();

        let mut api_response = OpenAIEmbeddingsResponse::from(core_response);
        api_response.total_time = total_time_val.as_secs_f64();
        api_response.individual_request_times = individual_times_val;

        Ok(api_response)
    }

    #[pyo3(name = "async_embed", signature = (input, model, encoding_format = None, dimensions = None, user = None, max_concurrent_requests = DEFAULT_CONCURRENCY, batch_size = DEFAULT_BATCH_SIZE, timeout_s = DEFAULT_REQUEST_TIMEOUT_S, max_chars_per_request = None, hedge_delay = None))]
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
        max_chars_per_request: Option<usize>,
        hedge_delay: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if input.is_empty() {
            return Err(PyValueError::new_err("Input list cannot be empty"));
        }

        let core_client = self.core_client.clone();

        let future = async move {
            let (core_response, batch_durations, core_total_time) = core_client
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
                .map_err(Self::convert_core_error_to_py_err)?;

            let individual_times_val: Vec<f64> = batch_durations
                .into_iter()
                .map(|d| d.as_secs_f64())
                .collect();

            let mut api_response = OpenAIEmbeddingsResponse::from(core_response);
            api_response.total_time = core_total_time.as_secs_f64();
            api_response.individual_request_times = individual_times_val;

            Ok(api_response)
        };

        pyo3_async_runtimes::tokio::future_into_py(py, future)
    }

    #[pyo3(signature = (query, texts, raw_scores = false, return_text = false, truncate = false, truncation_direction = "Right", max_concurrent_requests = DEFAULT_CONCURRENCY, batch_size = DEFAULT_BATCH_SIZE, timeout_s = DEFAULT_REQUEST_TIMEOUT_S, max_chars_per_request = None, hedge_delay = None))]
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
        max_chars_per_request: Option<usize>,
        hedge_delay: Option<f64>,
    ) -> PyResult<RerankResponse> {
        if texts.is_empty() {
            return Err(PyValueError::new_err("Texts list cannot be empty"));
        }

        let core_client = self.core_client.clone();
        let rt = Arc::clone(&self.runtime);
        let truncation_direction_owned = truncation_direction.to_string();

        let result_from_async_task = py.allow_threads(move || {
            let (tx, rx) = std::sync::mpsc::channel();

            rt.spawn(async move {
                let res = core_client
                    .process_rerank_requests(
                        query,
                        texts,
                        raw_scores,
                        return_text,
                        truncate,
                        truncation_direction_owned,
                        max_concurrent_requests,
                        batch_size,
                        max_chars_per_request,
                        timeout_s,
                        hedge_delay,
                    )
                    .await;
                let _ = tx.send(res);
            });

            match rx.recv() {
                Ok(inner_result) => inner_result.map_err(Self::convert_core_error_to_py_err),
                Err(e) => Err(PyValueError::new_err(format!(
                    "Failed to receive rerank result (channel error): {}",
                    e
                ))),
            }
        })?;

        let (core_response, batch_durations, total_time_val) = result_from_async_task;
        let individual_times_val: Vec<f64> = batch_durations
            .into_iter()
            .map(|d| d.as_secs_f64())
            .collect();

        let mut api_response = RerankResponse::from(core_response);
        api_response.total_time = total_time_val.as_secs_f64();
        api_response.individual_request_times = individual_times_val;

        Ok(api_response)
    }

    #[pyo3(name = "async_rerank", signature = (query, texts, raw_scores = false, return_text = false, truncate = false, truncation_direction = "Right", max_concurrent_requests = DEFAULT_CONCURRENCY, batch_size = DEFAULT_BATCH_SIZE, timeout_s = DEFAULT_REQUEST_TIMEOUT_S, max_chars_per_request = None, hedge_delay = None))]
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
        max_chars_per_request: Option<usize>,
        hedge_delay: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if texts.is_empty() {
            return Err(PyValueError::new_err("Texts list cannot be empty"));
        }

        let core_client = self.core_client.clone();
        let truncation_direction_owned = truncation_direction.to_string();

        let future = async move {
            let (core_response, batch_durations, core_total_time) = core_client
                .process_rerank_requests(
                    query,
                    texts,
                    raw_scores,
                    return_text,
                    truncate,
                    truncation_direction_owned,
                    max_concurrent_requests,
                    batch_size,
                    max_chars_per_request,
                    timeout_s,
                    hedge_delay,
                )
                .await
                .map_err(Self::convert_core_error_to_py_err)?;

            let individual_times_val: Vec<f64> = batch_durations
                .into_iter()
                .map(|d| d.as_secs_f64())
                .collect();

            let mut api_response = RerankResponse::from(core_response);
            api_response.total_time = core_total_time.as_secs_f64();
            api_response.individual_request_times = individual_times_val;

            Ok(api_response)
        };

        pyo3_async_runtimes::tokio::future_into_py(py, future)
    }

    #[pyo3(signature = (inputs, raw_scores = false, truncate = false, truncation_direction = "Right", max_concurrent_requests = DEFAULT_CONCURRENCY, batch_size = DEFAULT_BATCH_SIZE, timeout_s = DEFAULT_REQUEST_TIMEOUT_S, max_chars_per_request = None, hedge_delay = None))]
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
        max_chars_per_request: Option<usize>,
        hedge_delay: Option<f64>,
    ) -> PyResult<ClassificationResponse> {
        if inputs.is_empty() {
            return Err(PyValueError::new_err("Inputs list cannot be empty"));
        }

        let core_client = self.core_client.clone();
        let rt = Arc::clone(&self.runtime);
        let truncation_direction_owned = truncation_direction.to_string();

        let result_from_async_task = py.allow_threads(move || {
            let (tx, rx) = std::sync::mpsc::channel();

            rt.spawn(async move {
                let res = core_client
                    .process_classify_requests(
                        inputs,
                        raw_scores,
                        truncate,
                        truncation_direction_owned,
                        max_concurrent_requests,
                        batch_size,
                        max_chars_per_request,
                        timeout_s,
                        hedge_delay,
                    )
                    .await;
                let _ = tx.send(res);
            });

            match rx.recv() {
                Ok(inner_result) => inner_result.map_err(Self::convert_core_error_to_py_err),
                Err(e) => Err(PyValueError::new_err(format!(
                    "Failed to receive classify result (channel error): {}",
                    e
                ))),
            }
        })?;

        let (core_response, batch_durations, core_total_time) = result_from_async_task;
        let individual_times_val: Vec<f64> = batch_durations
            .into_iter()
            .map(|d| d.as_secs_f64())
            .collect();

        let mut api_response = ClassificationResponse::from(core_response);
        api_response.total_time = core_total_time.as_secs_f64();
        api_response.individual_request_times = individual_times_val;

        Ok(api_response)
    }

    #[pyo3(name = "async_classify", signature = (inputs, raw_scores = false, truncate = false, truncation_direction = "Right", max_concurrent_requests = DEFAULT_CONCURRENCY, batch_size = DEFAULT_BATCH_SIZE, timeout_s = DEFAULT_REQUEST_TIMEOUT_S, max_chars_per_request = None, hedge_delay = None))]
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
        max_chars_per_request: Option<usize>,
        hedge_delay: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if inputs.is_empty() {
            return Err(PyValueError::new_err("Inputs list cannot be empty"));
        }

        let core_client = self.core_client.clone();
        let truncation_direction_owned = truncation_direction.to_string();

        let future = async move {
            let (core_response, batch_durations, core_total_time) = core_client
                .process_classify_requests(
                    inputs,
                    raw_scores,
                    truncate,
                    truncation_direction_owned,
                    max_concurrent_requests,
                    batch_size,
                    max_chars_per_request,
                    timeout_s,
                    hedge_delay,
                )
                .await
                .map_err(Self::convert_core_error_to_py_err)?;

            let individual_times_val: Vec<f64> = batch_durations
                .into_iter()
                .map(|d| d.as_secs_f64())
                .collect();

            let mut api_response = ClassificationResponse::from(core_response);
            api_response.total_time = core_total_time.as_secs_f64();
            api_response.individual_request_times = individual_times_val;

            Ok(api_response)
        };

        pyo3_async_runtimes::tokio::future_into_py(py, future)
    }

    #[pyo3(signature = (url_path, payloads, max_concurrent_requests = DEFAULT_CONCURRENCY, timeout_s = DEFAULT_REQUEST_TIMEOUT_S, hedge_delay = None))]
    fn batch_post(
        &self,
        py: Python,
        url_path: String,
        payloads: Vec<PyObject>,
        max_concurrent_requests: usize,
        timeout_s: f64,
        hedge_delay: Option<f64>,
    ) -> PyResult<BatchPostResponse> {
        if payloads.is_empty() {
            return Err(PyValueError::new_err("Payloads list cannot be empty"));
        }

        let mut payloads_json: Vec<serde_json::Value> = Vec::with_capacity(payloads.len());
        for (idx, py_obj) in payloads.into_iter().enumerate() {
            let bound_obj = py_obj.bind(py);
            let json_val = pythonize::depythonize(bound_obj).map_err(|e| {
                PyValueError::new_err(format!(
                    "Failed to depythonize payload at index {}: {}",
                    idx, e
                ))
            })?;
            payloads_json.push(json_val);
        }

        let core_client = self.core_client.clone();
        let rt = Arc::clone(&self.runtime);

        let result_from_async_task = py.allow_threads(move || {
            let (tx, rx) = std::sync::mpsc::channel();

            rt.spawn(async move {
                let res = core_client
                    .process_batch_post_requests(
                        url_path,
                        payloads_json,
                        max_concurrent_requests,
                        timeout_s,
                        hedge_delay,
                    )
                    .await;
                let _ = tx.send(res);
            });

            match rx.recv() {
                Ok(inner_result) => inner_result.map_err(Self::convert_core_error_to_py_err),
                Err(e) => Err(PyValueError::new_err(format!(
                    "Failed to receive batch_post result (channel error): {}",
                    e
                ))),
            }
        })?;

        let (response_data_with_times_and_headers, total_time) = result_from_async_task;

        let mut results_py: Vec<PyObject> =
            Vec::with_capacity(response_data_with_times_and_headers.len());
        let mut individual_request_times_collected: Vec<f64> =
            Vec::with_capacity(response_data_with_times_and_headers.len());
        let mut collected_headers_py: Vec<PyObject> =
            Vec::with_capacity(response_data_with_times_and_headers.len());

        for (idx, (json_val, headers_map, duration)) in
            response_data_with_times_and_headers.into_iter().enumerate()
        {
            let py_obj_bound = pythonize::pythonize(py, &json_val).map_err(|e| {
                PyValueError::new_err(format!(
                    "Failed to pythonize response data at index {}: {}",
                    idx, e
                ))
            })?;
            #[allow(deprecated)]
            results_py.push(py_obj_bound.to_object(py));

            let headers_py_obj = pythonize::pythonize(py, &headers_map).map_err(|e| {
                PyValueError::new_err(format!(
                    "Failed to pythonize headers at index {}: {}",
                    idx, e
                ))
            })?;
            #[allow(deprecated)]
            collected_headers_py.push(headers_py_obj.to_object(py));

            individual_request_times_collected.push(duration.as_secs_f64());
        }
        let total_time = total_time.as_secs_f64();

        Ok(BatchPostResponse {
            data: results_py,
            total_time,
            individual_request_times: individual_request_times_collected,
            response_headers: collected_headers_py,
        })
    }

    #[pyo3(name = "async_batch_post", signature = (url_path, payloads, max_concurrent_requests = DEFAULT_CONCURRENCY, timeout_s = DEFAULT_REQUEST_TIMEOUT_S, hedge_delay = None))]
    fn async_batch_post<'py>(
        &self,
        py: Python<'py>,
        url_path: String,
        payloads: Vec<PyObject>,
        max_concurrent_requests: usize,
        timeout_s: f64,
        hedge_delay: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if payloads.is_empty() {
            return Err(PyValueError::new_err("Payloads list cannot be empty"));
        }

        let mut payloads_json: Vec<serde_json::Value> = Vec::with_capacity(payloads.len());
        for (idx, py_obj) in payloads.into_iter().enumerate() {
            let bound_obj = py_obj.bind(py);
            let json_val = pythonize::depythonize(bound_obj).map_err(|e| {
                PyValueError::new_err(format!(
                    "Failed to depythonize payload at index {}: {}",
                    idx, e
                ))
            })?;
            payloads_json.push(json_val);
        }

        let core_client = self.core_client.clone();

        let future = async move {
            let (response_data_with_times_and_headers, total_time) = core_client
                .process_batch_post_requests(
                    url_path,
                    payloads_json,
                    max_concurrent_requests,
                    timeout_s,
                    hedge_delay,
                )
                .await
                .map_err(Self::convert_core_error_to_py_err)?;

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
                    let py_obj_bound = pythonize::pythonize(py_gil, &json_val).map_err(|e| {
                        PyValueError::new_err(format!(
                            "Failed to pythonize response data at index {}: {}",
                            idx, e
                        ))
                    })?;
                    #[allow(deprecated)]
                    results_py.push(py_obj_bound.into_py(py_gil));

                    let headers_py_obj =
                        pythonize::pythonize(py_gil, &headers_map).map_err(|e| {
                            PyValueError::new_err(format!(
                                "Failed to pythonize headers at index {}: {}",
                                idx, e
                            ))
                        })?;
                    #[allow(deprecated)]
                    collected_headers_py.push(headers_py_obj.to_object(py_gil));

                    individual_request_times_collected.push(duration.as_secs_f64());
                }

                Ok(BatchPostResponse {
                    data: results_py,
                    total_time: total_time.as_secs_f64(),
                    individual_request_times: individual_request_times_collected,
                    response_headers: collected_headers_py,
                })
            })
        };

        pyo3_async_runtimes::tokio::future_into_py(py, future)
    }
}

#[pymodule]
fn baseten_performance_client(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PerformanceClient>()?;
    m.add_class::<OpenAIEmbeddingsResponse>()?;
    m.add_class::<OpenAIEmbeddingData>()?;
    m.add_class::<OpenAIUsage>()?;
    m.add_class::<RerankResult>()?;
    m.add_class::<RerankResponse>()?;
    m.add_class::<ClassificationResult>()?;
    m.add_class::<ClassificationResponse>()?;
    m.add_class::<BatchPostResponse>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

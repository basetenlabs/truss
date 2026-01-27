#![allow(clippy::too_many_arguments)]

use baseten_performance_client_core::{
    CancellationToken as CoreCancellationToken, ClientError, CoreClassificationResponse,
    CoreClassificationResult, CoreEmbeddingVariant, CoreOpenAIEmbeddingData,
    CoreOpenAIEmbeddingsResponse, CoreOpenAIUsage, CoreRerankResponse, CoreRerankResult,
    HttpClientWrapper as HttpClientWrapperRs, PerformanceClientCore,
    RequestProcessingPreference as RustRequestProcessingPreference, DEFAULT_BATCH_SIZE,
    DEFAULT_CONCURRENCY, DEFAULT_REQUEST_TIMEOUT_S, HEDGE_BUDGET_PERCENTAGE, INITIAL_BACKOFF_MS,
    MAX_HTTP_RETRIES, RETRY_BUDGET_PERCENTAGE,
};

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use once_cell::sync::Lazy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::runtime::Runtime;

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

// --- Global Tokio Runtime (Python-specific) ---
static CTRL_C_RECEIVED: AtomicBool = AtomicBool::new(false);

static GLOBAL_RUNTIME: Lazy<Arc<Runtime>> = Lazy::new(|| {
    let runtime = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create global multi-threaded Tokio runtime"),
    );
    let runtime_clone_for_signal: Arc<Runtime> = Arc::clone(&runtime);
    // Spawn a task to listen for Ctrl+C
    runtime_clone_for_signal.spawn(async {
        if tokio::signal::ctrl_c().await.is_ok() {
            CTRL_C_RECEIVED.store(true, Ordering::SeqCst);
        }
    });
    runtime
});

/// Run a future with Ctrl+C cancellation support.
/// When Ctrl+C is received, the future is dropped, triggering JoinSetGuard cleanup.
async fn run_with_ctrl_c<F, T>(future: F) -> Result<T, ClientError>
where
    F: std::future::Future<Output = Result<T, ClientError>>,
{
    tokio::select! {
        biased;

        // Poll for Ctrl+C signal periodically
        _ = async {
            loop {
                if CTRL_C_RECEIVED.load(Ordering::SeqCst) {
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        } => {
            Err(ClientError::Cancellation("Interrupted by Ctrl+C".to_string()))
        }

        // Run the actual future
        result = future => result,
    }
}

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
    response_headers: Vec<std::collections::HashMap<String, String>>,
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
            response_headers: core.response_headers,
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
        Ok(array.into_pyarray(py))
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
    response_headers: Vec<std::collections::HashMap<String, String>>,
}

impl From<CoreRerankResponse> for RerankResponse {
    fn from(core: CoreRerankResponse) -> Self {
        RerankResponse {
            object: core.object,
            data: core.data.into_iter().map(RerankResult::from).collect(),
            total_time: core.total_time,
            individual_request_times: core.individual_request_times,
            response_headers: core.response_headers,
        }
    }
}

#[pymethods]
impl RerankResponse {
    #[new]
    #[pyo3(signature = (data, total_time, individual_request_times, response_headers = None))]
    fn new(
        data: Vec<RerankResult>,
        total_time: f64,
        individual_request_times: Vec<f64>,
        response_headers: Option<Vec<std::collections::HashMap<String, String>>>,
    ) -> Self {
        RerankResponse {
            object: "list".to_string(),
            data,
            total_time,
            individual_request_times,
            response_headers: response_headers.unwrap_or_default(),
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
    response_headers: Vec<std::collections::HashMap<String, String>>,
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
            response_headers: core.response_headers,
        }
    }
}

#[pymethods]
impl ClassificationResponse {
    #[new]
    #[pyo3(signature = (data, total_time, individual_request_times, response_headers = None))]
    fn new(
        data: Vec<Vec<ClassificationResult>>,
        total_time: f64,
        individual_request_times: Vec<f64>,
        response_headers: Option<Vec<std::collections::HashMap<String, String>>>,
    ) -> Self {
        ClassificationResponse {
            object: "list".to_string(),
            data,
            total_time,
            individual_request_times,
            response_headers: response_headers.unwrap_or_default(),
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

#[pyclass(name = "HttpClientWrapper")]
#[derive(Clone)]
struct HttpClientWrapper {
    inner: Arc<HttpClientWrapperRs>,
}

#[pymethods]
impl HttpClientWrapper {
    #[new]
    #[pyo3(signature = (http_version = 1))]
    fn new(http_version: u8) -> PyResult<Self> {
        let inner = HttpClientWrapperRs::new(http_version)
            .map_err(PerformanceClient::convert_core_error_to_py_err)?;
        Ok(HttpClientWrapper { inner })
    }

    fn __repr__(&self) -> String {
        "HttpClientWrapper(...)".to_string()
    }
}

/// User-facing configuration for request processing with budget percentages.
/// CancellationToken for cancelling async operations
#[derive(Debug, Clone)]
#[pyclass]
pub struct CancellationToken {
    inner: CoreCancellationToken,
}

#[pymethods]
impl CancellationToken {
    /// Create a new cancellation token
    #[new]
    pub fn new() -> Self {
        Self {
            inner: CoreCancellationToken::new(false),
        }
    }

    /// Cancel all operations using this token
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    /// Check if cancellation has been requested
    pub fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }
}

/// Provides sensible defaults and getters for all properties.
#[derive(Debug, Clone)]
#[pyclass]
pub struct RequestProcessingPreference {
    #[pyo3(get, set)]
    pub max_concurrent_requests: usize,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub max_chars_per_request: Option<usize>,
    #[pyo3(get, set)]
    pub timeout_s: f64,
    #[pyo3(get, set)]
    pub hedge_delay: Option<f64>,
    #[pyo3(get, set)]
    pub total_timeout_s: Option<f64>,
    #[pyo3(get, set)]
    pub hedge_budget_pct: f64,
    #[pyo3(get, set)]
    pub retry_budget_pct: f64,
    #[pyo3(get, set)]
    pub max_retries: u32,
    #[pyo3(get, set)]
    pub initial_backoff_ms: u64,
    #[pyo3(get, set)]
    pub cancel_token: Option<CancellationToken>,
    #[pyo3(get, set)]
    pub primary_api_key_override: Option<String>,
    inner: RustRequestProcessingPreference,
}

#[pymethods]
impl RequestProcessingPreference {
    #[new]
    #[pyo3(signature = (
        max_concurrent_requests = None,
        batch_size = None,
        timeout_s = None,
        max_chars_per_request = None,
        hedge_delay = None,
        total_timeout_s = None,
        hedge_budget_pct = None,
        retry_budget_pct = None,
        max_retries = None,
        initial_backoff_ms = None,
        cancel_token = None,
        primary_api_key_override = None
    ))]
    fn new(
        max_concurrent_requests: Option<usize>,
        batch_size: Option<usize>,
        timeout_s: Option<f64>,
        max_chars_per_request: Option<usize>,
        hedge_delay: Option<f64>,
        total_timeout_s: Option<f64>,
        hedge_budget_pct: Option<f64>,
        retry_budget_pct: Option<f64>,
        max_retries: Option<u32>,
        initial_backoff_ms: Option<u64>,
        cancel_token: Option<CancellationToken>,
        primary_api_key_override: Option<String>,
    ) -> Self {
        let rust_pref = RustRequestProcessingPreference {
            max_concurrent_requests,
            batch_size,
            max_chars_per_request,
            timeout_s,
            hedge_delay,
            total_timeout_s,
            hedge_budget_pct,
            retry_budget_pct,
            max_retries,
            initial_backoff_ms,
            cancel_token: cancel_token.as_ref().map(|token| token.inner.clone()),
            primary_api_key_override: primary_api_key_override,
        };

        // Apply defaults using the same method as Rust core
        let complete = rust_pref.with_defaults();

        RequestProcessingPreference {
            max_concurrent_requests: complete
                .max_concurrent_requests
                .unwrap_or(DEFAULT_CONCURRENCY),
            batch_size: complete.batch_size.unwrap_or(DEFAULT_BATCH_SIZE),
            max_chars_per_request: complete.max_chars_per_request,
            timeout_s: complete.timeout_s.unwrap_or(DEFAULT_REQUEST_TIMEOUT_S),
            hedge_delay: complete.hedge_delay,
            total_timeout_s: complete.total_timeout_s,
            hedge_budget_pct: complete.hedge_budget_pct.unwrap_or(HEDGE_BUDGET_PERCENTAGE),
            retry_budget_pct: complete.retry_budget_pct.unwrap_or(RETRY_BUDGET_PERCENTAGE),
            max_retries: complete.max_retries.unwrap_or(MAX_HTTP_RETRIES) as u32,
            initial_backoff_ms: complete.initial_backoff_ms.unwrap_or(INITIAL_BACKOFF_MS),
            cancel_token,
            primary_api_key_override: complete.primary_api_key_override,
            inner: rust_pref,
        }
    }

    /// Create a new preference with default values (class method)
    #[classmethod]
    fn default(_cls: &Bound<'_, PyType>) -> PyResult<Self> {
        Ok(Self::new(
            None, None, None, None, None, None, None, None, None, None, None, None,
        ))
    }

    /// Return a string representation
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "RequestProcessingPreference(max_concurrent_requests={}, batch_size={}, timeout_s={:.3}, hedge_delay={:?}, total_timeout_s={:?}, hedge_budget_pct={:.3}, retry_budget_pct={:.3}, max_retries={}, initial_backoff_ms={})",
            self.max_concurrent_requests,
            self.batch_size,
            self.timeout_s,
            self.hedge_delay,
            self.total_timeout_s,
            self.hedge_budget_pct,
            self.retry_budget_pct,
            self.max_retries,
            self.initial_backoff_ms
        ))
    }

    /// Return a string representation
    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }
}

#[pyclass]
struct PerformanceClient {
    core_client: PerformanceClientCore,
    runtime: Arc<Runtime>,
}

impl PerformanceClient {
    fn convert_core_error_to_py_err(err: ClientError) -> PyErr {
        match err {
            ClientError::LocalTimeout(msg, _) => {
                PyErr::new::<Timeout, _>((408, format!("Local timeout: {}", msg), "local"))
            }
            ClientError::RemoteTimeout(msg, _) => {
                PyErr::new::<Timeout, _>((408, format!("Remote timeout: {}", msg), "remote"))
            }
            ClientError::Http {
                status,
                message,
                customer_request_id: _,
            } => PyErr::new::<HTTPError, _>((status, message)),
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
    #[pyo3(signature = (base_url, api_key = None, http_version = 1, client_wrapper = None))]
    fn new(
        base_url: String,
        api_key: Option<String>,
        http_version: u8,
        client_wrapper: Option<HttpClientWrapper>,
    ) -> PyResult<Self> {
        let wrapper = client_wrapper.map(|w| w.inner);
        let core_client = PerformanceClientCore::new(base_url, api_key, http_version, wrapper)
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

    fn get_client_wrapper(&self) -> HttpClientWrapper {
        HttpClientWrapper {
            inner: self.core_client.get_client_wrapper(),
        }
    }

    #[pyo3(signature = (input, model, encoding_format = None, dimensions = None, user = None, preference = None))]
    fn embed(
        &self,
        py: Python,
        input: Vec<String>,
        model: String,
        encoding_format: Option<String>,
        dimensions: Option<u32>,
        user: Option<String>,
        preference: Option<&RequestProcessingPreference>,
    ) -> PyResult<OpenAIEmbeddingsResponse> {
        if input.is_empty() {
            return Err(PyValueError::new_err("Input list cannot be empty"));
        }

        // Validate model string
        if model.trim().is_empty() {
            return Err(PyValueError::new_err("Model name cannot be empty"));
        }

        // Validate dimensions
        if let Some(dim) = dimensions {
            if dim == 0 || dim > 1000000 {
                return Err(PyValueError::new_err(
                    "Dimensions must be between 1 and 1000000",
                ));
            }
        }

        // Validate user string length
        if let Some(ref user_str) = user {
            if user_str.len() > 1000 {
                return Err(PyValueError::new_err(
                    "User string cannot exceed 1000 characters",
                ));
            }
        }

        // Validate encoding format
        if let Some(ref format) = encoding_format {
            if !matches!(format.as_str(), "float" | "base64") {
                return Err(PyValueError::new_err(
                    "Encoding format must be 'float' or 'base64'",
                ));
            }
        }

        let core_client = self.core_client.clone();
        let rt: Arc<Runtime> = Arc::clone(&self.runtime);

        // Use provided preference or create default
        let rust_preference = preference.map(|p| p.inner.clone()).unwrap_or_default();

        let result_from_async_task = py.allow_threads(move || {
            rt.block_on(run_with_ctrl_c(async move {
                core_client
                    .process_embeddings_requests(
                        input,
                        model,
                        encoding_format,
                        dimensions,
                        user,
                        &rust_preference,
                    )
                    .await
            }))
            .map_err(Self::convert_core_error_to_py_err)
        })?;

        let (core_response, batch_durations, headers, total_time_val) = result_from_async_task;
        let individual_times_val: Vec<f64> = batch_durations
            .into_iter()
            .map(|d| d.as_secs_f64())
            .collect();

        let mut api_response = OpenAIEmbeddingsResponse::from(core_response);
        api_response.total_time = total_time_val.as_secs_f64();
        api_response.individual_request_times = individual_times_val;
        api_response.response_headers = headers;

        Ok(api_response)
    }

    #[pyo3(name = "async_embed", signature = (input, model, encoding_format = None, dimensions = None, user = None, preference = None))]
    fn async_embed<'py>(
        &self,
        py: Python<'py>,
        input: Vec<String>,
        model: String,
        encoding_format: Option<String>,
        dimensions: Option<u32>,
        user: Option<String>,
        preference: Option<&RequestProcessingPreference>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if input.is_empty() {
            return Err(PyValueError::new_err("Input list cannot be empty"));
        }

        // Validate model string
        if model.trim().is_empty() {
            return Err(PyValueError::new_err("Model name cannot be empty"));
        }

        // Validate dimensions
        if let Some(dim) = dimensions {
            if dim == 0 || dim > 1000000 {
                return Err(PyValueError::new_err(
                    "Dimensions must be between 1 and 1000000",
                ));
            }
        }

        // Validate user string length
        if let Some(ref user_str) = user {
            if user_str.len() > 1000 {
                return Err(PyValueError::new_err(
                    "User string cannot exceed 1000 characters",
                ));
            }
        }

        // Validate encoding format
        if let Some(ref format) = encoding_format {
            if !matches!(format.as_str(), "float" | "base64") {
                return Err(PyValueError::new_err(
                    "Encoding format must be 'float' or 'base64'",
                ));
            }
        }

        let core_client = self.core_client.clone();

        // Use provided preference or create default
        let rust_preference = preference.map(|p| p.inner.clone()).unwrap_or_default();

        // Extract cancellation token if present
        let cancel_token = preference.and_then(|p| p.cancel_token.clone());

        let future = async move {
            // Check for cancellation before starting the request
            if let Some(ref token) = cancel_token {
                if token.is_cancelled() {
                    return Err(PyValueError::new_err("Operation cancelled by token"));
                }
            }

            let (core_response, batch_durations, headers, core_total_time) = core_client
                .process_embeddings_requests(
                    input,
                    model,
                    encoding_format,
                    dimensions,
                    user,
                    &rust_preference,
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
            api_response.response_headers = headers;

            Ok(api_response)
        };

        pyo3_async_runtimes::tokio::future_into_py(py, future)
    }

    #[pyo3(signature = (query, texts, raw_scores = false, model = None, return_text = false, truncate = false, truncation_direction = "Right", preference = None))]
    fn rerank(
        &self,
        py: Python,
        query: String,
        texts: Vec<String>,
        raw_scores: bool,
        model: Option<String>,
        return_text: bool,
        truncate: bool,
        truncation_direction: &str,
        preference: Option<&RequestProcessingPreference>,
    ) -> PyResult<RerankResponse> {
        if texts.is_empty() {
            return Err(PyValueError::new_err("Texts list cannot be empty"));
        }

        let core_client = self.core_client.clone();
        let rt = Arc::clone(&self.runtime);
        let truncation_direction_owned = truncation_direction.to_string();

        // Use provided preference or create default
        let rust_preference = preference.map(|p| p.inner.clone()).unwrap_or_default();

        let result_from_async_task = py.allow_threads(move || {
            rt.block_on(run_with_ctrl_c(async move {
                core_client
                    .process_rerank_requests(
                        query,
                        texts,
                        raw_scores,
                        model,
                        return_text,
                        truncate,
                        truncation_direction_owned,
                        &rust_preference,
                    )
                    .await
            }))
            .map_err(Self::convert_core_error_to_py_err)
        })?;

        let (core_response, batch_durations, headers, total_time_val) = result_from_async_task;
        let individual_times_val: Vec<f64> = batch_durations
            .into_iter()
            .map(|d| d.as_secs_f64())
            .collect();

        let mut api_response = RerankResponse::from(core_response);
        api_response.total_time = total_time_val.as_secs_f64();
        api_response.individual_request_times = individual_times_val;
        api_response.response_headers = headers;

        Ok(api_response)
    }

    #[pyo3(name = "async_rerank", signature = (query, texts, raw_scores = false, model = None, return_text = false, truncate = false, truncation_direction = "Right", preference = None))]
    fn async_rerank<'py>(
        &self,
        py: Python<'py>,
        query: String,
        texts: Vec<String>,
        raw_scores: bool,
        model: Option<String>,
        return_text: bool,
        truncate: bool,
        truncation_direction: &str,
        preference: Option<&RequestProcessingPreference>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if texts.is_empty() {
            return Err(PyValueError::new_err("Texts list cannot be empty"));
        }

        let core_client = self.core_client.clone();
        let truncation_direction_owned = truncation_direction.to_string();

        // Use provided preference or create default
        let rust_preference = preference.map(|p| p.inner.clone()).unwrap_or_default();

        let future = async move {
            let (core_response, batch_durations, headers, core_total_time) = core_client
                .process_rerank_requests(
                    query,
                    texts,
                    raw_scores,
                    model,
                    return_text,
                    truncate,
                    truncation_direction_owned,
                    &rust_preference,
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
            api_response.response_headers = headers;

            Ok(api_response)
        };

        pyo3_async_runtimes::tokio::future_into_py(py, future)
    }

    #[pyo3(signature = (inputs, model = None, raw_scores = false, truncate = false, truncation_direction = "Right", preference = None))]
    fn classify(
        &self,
        py: Python,
        inputs: Vec<String>,
        model: Option<String>,
        raw_scores: bool,
        truncate: bool,
        truncation_direction: &str,
        preference: Option<&RequestProcessingPreference>,
    ) -> PyResult<ClassificationResponse> {
        if inputs.is_empty() {
            return Err(PyValueError::new_err("Inputs list cannot be empty"));
        }

        let core_client = self.core_client.clone();
        let rt = Arc::clone(&self.runtime);
        let truncation_direction_owned = truncation_direction.to_string();

        // Use provided preference or create default
        let rust_preference = preference.map(|p| p.inner.clone()).unwrap_or_default();

        let result_from_async_task = py.allow_threads(move || {
            rt.block_on(run_with_ctrl_c(async move {
                core_client
                    .process_classify_requests(
                        inputs,
                        model,
                        raw_scores,
                        truncate,
                        truncation_direction_owned,
                        &rust_preference,
                    )
                    .await
            }))
            .map_err(Self::convert_core_error_to_py_err)
        })?;

        let (core_response, batch_durations, headers, core_total_time) = result_from_async_task;
        let individual_times_val: Vec<f64> = batch_durations
            .into_iter()
            .map(|d| d.as_secs_f64())
            .collect();

        let mut api_response = ClassificationResponse::from(core_response);
        api_response.total_time = core_total_time.as_secs_f64();
        api_response.individual_request_times = individual_times_val;
        api_response.response_headers = headers;

        Ok(api_response)
    }

    #[pyo3(name = "async_classify", signature = (inputs, model = None, raw_scores = false, truncate = false, truncation_direction = "Right", preference = None))]
    fn async_classify<'py>(
        &self,
        py: Python<'py>,
        inputs: Vec<String>,
        model: Option<String>,
        raw_scores: bool,
        truncate: bool,
        truncation_direction: &str,
        preference: Option<&RequestProcessingPreference>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if inputs.is_empty() {
            return Err(PyValueError::new_err("Inputs list cannot be empty"));
        }

        let core_client = self.core_client.clone();
        let truncation_direction_owned = truncation_direction.to_string();

        // Use provided preference or create default
        let rust_preference = preference.map(|p| p.inner.clone()).unwrap_or_default();

        let future = async move {
            let (core_response, batch_durations, headers, core_total_time) = core_client
                .process_classify_requests(
                    inputs,
                    model,
                    raw_scores,
                    truncate,
                    truncation_direction_owned,
                    &rust_preference,
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
            api_response.response_headers = headers;

            Ok(api_response)
        };

        pyo3_async_runtimes::tokio::future_into_py(py, future)
    }

    #[pyo3(signature = (url_path, payloads, preference = None, custom_headers = None, method = None))]
    fn batch_post(
        &self,
        py: Python,
        url_path: String,
        payloads: Vec<PyObject>,
        preference: Option<&RequestProcessingPreference>,
        custom_headers: Option<std::collections::HashMap<String, String>>,
        method: Option<String>,
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

        // Use provided preference or create default
        let rust_preference = preference.map(|p| p.inner.clone()).unwrap_or_default();

        // Parse method parameter using core function
        let http_method =
            baseten_performance_client_core::http::HttpMethod::from_str(method.as_deref())
                .map_err(|e| PyValueError::new_err(e))?;

        let result_from_async_task = py.allow_threads(move || {
            rt.block_on(run_with_ctrl_c(async move {
                core_client
                    .process_batch_post_requests(
                        url_path,
                        payloads_json,
                        &rust_preference,
                        custom_headers,
                        http_method,
                    )
                    .await
            }))
            .map_err(Self::convert_core_error_to_py_err)
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

    #[pyo3(name = "async_batch_post", signature = (url_path, payloads, preference = None, custom_headers = None, method = None))]
    fn async_batch_post<'py>(
        &self,
        py: Python<'py>,
        url_path: String,
        payloads: Vec<PyObject>,
        preference: Option<&RequestProcessingPreference>,
        custom_headers: Option<std::collections::HashMap<String, String>>,
        method: Option<String>,
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

        // Use provided preference or create default
        let rust_preference = preference.map(|p| p.inner.clone()).unwrap_or_default();

        // Parse method parameter using core function
        let http_method =
            baseten_performance_client_core::http::HttpMethod::from_str(method.as_deref())
                .map_err(|e| PyValueError::new_err(e))?;

        let future = async move {
            let (response_data_with_times_and_headers, total_time) = core_client
                .process_batch_post_requests(
                    url_path,
                    payloads_json,
                    &rust_preference,
                    custom_headers,
                    http_method,
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
    m.add_class::<HttpClientWrapper>()?;
    m.add_class::<RequestProcessingPreference>()?;
    m.add_class::<CancellationToken>()?;
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

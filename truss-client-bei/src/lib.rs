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
            runtime: Arc::clone(&GLOBAL_RUNTIME), // Clone the Arc to the global runtime
        })
    }

    #[getter]
    fn api_key(&self) -> PyResult<String> {
        Ok(self.api_key.clone())
    }

    #[pyo3(signature = (input, model, encoding_format = None, dimensions = None, user = None, max_concurrent_requests = 64, batch_size = 4))]
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

        let model_string: String = model.to_string(); // Convert &str to String

        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let api_base = self.api_base.clone();
        let rt = Arc::clone(&self.runtime);

        // py.allow_threads releases the GIL, allowing the current thread to block
        // on rx.recv() without freezing other Python threads.
        let result_from_async_task: Result<OpenAIEmbeddingsResponse, PyErr> = py.allow_threads(move || {
            // Create a channel to receive the result from the spawned Tokio task.
            let (tx, rx) = mpsc::channel::<Result<OpenAIEmbeddingsResponse, PyErr>>();

            // Spawn the asynchronous processing onto the Tokio runtime.
            rt.spawn(async move {
                let res = process_embeddings_requests(
                    client,
                    input, // `input` is moved here
                    model_string, // Use the owned String
                    api_key,
                    api_base,
                    encoding_format,
                    dimensions,
                    user,
                    max_concurrent_requests,
                    batch_size
                )
                .await;
                // Send the result back to the waiting synchronous thread.
                // If sending fails, it means the receiver has been dropped,
                // which would happen if rx.recv() itself errors or the thread panics.
                if tx.send(res).is_err() {
                    // Optional: Log an error if the receiver is gone.
                    // eprintln!("Failed to send async result: receiver dropped.");
                }
            });

            // Block the current thread (which has released the GIL)
            // waiting for the result from the spawned task.
            match rx.recv() {
                Ok(res) => res, // This is the Result<OpenAIEmbeddingsResponse, PyErr>
                Err(e) => Err(PyValueError::new_err(format!(
                    "Failed to receive result from async task: {}",
                    e
                ))),
            }
        });

        // Handle the Result before accessing its fields.
        // If result_from_async_task is an Err, the `?` will propagate it.
        // Otherwise, successful_response will be OpenAIEmbeddingsResponse.
        let successful_response = result_from_async_task?;

        // Convert the successful result to Python objects.
        Python::with_gil(|py| {
            // Directly convert successful_response to a PyObject
            Ok(successful_response.into_py(py))
        })
    }
}

async fn send_single_embedding_request(
    client: Client,
    texts_batch: Vec<String>,
    model: String,
    api_key: String,
    api_base: String,
    encoding_format: Option<String>,
    dimensions: Option<u32>,
    user: Option<String>,
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
) -> Result<OpenAIEmbeddingsResponse, PyErr> {
    let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));
    let mut tasks = Vec::new();
    let total_texts = texts.len();

    // This loop iterates through chunks based on user-defined batch_size
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

        // Calculate the absolute start index for items in this specific user_textBatch
        let current_batch_absolute_start_index = batch_index * batch_size;

        tasks.push(tokio::spawn(async move {
            let _permit = semaphore_clone.acquire().await.expect("Semaphore acquire failed");
            let mut response = send_single_embedding_request(
                client_clone,
                user_text_batch_owned, // Send the user-defined batch
                model_clone,
                api_key_clone,
                api_base_clone,
                encoding_format_clone,
                dimensions_clone,
                user_clone,
            )
            .await?;

            // Adjust indices: the response from send_single_embedding_request will have indices starting from 0
            // relative to user_text_batch_owned. We need to shift them to be absolute.
            for item in &mut response.data {
                item.index += current_batch_absolute_start_index;
            }
            // Return the processed data and usage for this batch
            Result::<_, PyErr>::Ok((response.data, response.usage))
        }));
    }

    let results = join_all(tasks).await;

    let mut all_embedding_data: Vec<OpenAIEmbeddingData> = Vec::with_capacity(total_texts);
    let mut aggregated_prompt_tokens: u32 = 0;
    let mut aggregated_total_tokens: u32 = 0;

    for result in results {
        match result {
            Ok(Ok((data_part, usage_part))) => { // Adjusted to expect a tuple
                all_embedding_data.extend(data_part);
                aggregated_prompt_tokens = aggregated_prompt_tokens.saturating_add(usage_part.prompt_tokens);
                aggregated_total_tokens = aggregated_total_tokens.saturating_add(usage_part.total_tokens);
            }
            Ok(Err(py_err)) => return Err(py_err),
            Err(join_err) => return Err(PyValueError::new_err(format!("Tokio join error: {}", join_err))),
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

// --- PyO3 Module Definition ---
#[pymodule]
fn truss_client_bei(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SyncClient>()?;
    m.add_class::<OpenAIEmbeddingsResponse>()?; // Add class to module
    m.add_class::<OpenAIEmbeddingData>()?;    // Add class to module
    m.add_class::<OpenAIUsage>()?;             // Add class to module
    Ok(())
}

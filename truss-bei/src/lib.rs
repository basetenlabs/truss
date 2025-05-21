// implements client for openai embeddings
// as python api that fans out to multiple requests
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyList, PyFloat};
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
struct OpenAIEmbeddingData {
    object: String,
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Deserialize, Debug, Clone)]
struct OpenAIUsage {
    prompt_tokens: u32,
    total_tokens: u32,
}

#[derive(Deserialize, Debug, Clone)]
struct OpenAIEmbeddingsResponse {
    object: String,
    data: Vec<OpenAIEmbeddingData>,
    model: String,
    usage: OpenAIUsage,
}

// --- SyncBeiClient Class ---
#[pyclass]
struct SyncBeiClient {
    api_key: String,
    api_base: String,
    client: Client,
    runtime: Arc<Runtime>, // This will now hold a clone of the global runtime's Arc
}

#[pymethods]
impl SyncBeiClient {
    #[new]
    #[pyo3(signature = (api_key, api_base = "https://api.openai.com".to_string()))]
    fn new(api_key: String, api_base: String) -> PyResult<Self> {
        Ok(SyncBeiClient {
            api_key,
            api_base,
            client: Client::new(),
            runtime: Arc::clone(&GLOBAL_RUNTIME), // Clone the Arc to the global runtime
        })
    }

    #[pyo3(signature = (input, model, encoding_format = None, dimensions = None, user = None, max_concurrent_requests = 64))]
    fn embeddings(
        &self,
        py: Python,
        input: Vec<String>, // Will be moved into the async task
        model: String,     // Will be moved
        encoding_format: Option<String>, // Will be moved
        dimensions: Option<u32>,         // Is Copy
        user: Option<String>,            // Will be moved
        max_concurrent_requests: usize,  // Is Copy
    ) -> PyResult<PyObject> {
        if input.is_empty() {
            return Err(PyValueError::new_err("Input list cannot be empty"));
        }
        if max_concurrent_requests == 0 {
            return Err(PyValueError::new_err("max_concurrent_requests must be greater than 0"));
        }

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
                    model,
                    api_key,
                    api_base,
                    encoding_format,
                    dimensions,
                    user,
                    max_concurrent_requests,
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
        }); // Removed ? here, assuming py.allow_threads flattens PyResult<PyResult<T>> to PyResult<T>

        // Handle the Result before accessing its fields.
        // If result_from_async_task is an Err, the `?` will propagate it.
        // Otherwise, successful_response will be OpenAIEmbeddingsResponse.
        let successful_response = result_from_async_task?;

        // Convert the successful result to Python objects.
        Python::with_gil(|py| {
            let py_embeddings_list = PyList::empty_bound(py);
            for embedding_data in successful_response.data { // Now using successful_response
                let py_embedding = PyList::new_bound(py, embedding_data.embedding.iter().map(|&f| PyFloat::new_bound(py, f.into())));
                py_embeddings_list.append(py_embedding)?;
            }
            Ok(py_embeddings_list.to_object(py))
        })
    }
}

// Max number of input items per OpenAI API request
const OPENAI_MAX_INPUT_ITEMS_PER_REQUEST: usize = 2048;

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

    let url = format!("{}/v1/embeddings", api_base.trim_end_matches('/'));

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
) -> Result<OpenAIEmbeddingsResponse, PyErr> {
    let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));
    let mut tasks = Vec::new();
    let total_texts = texts.len();

    for (chunk_index, text_chunk) in texts.chunks(OPENAI_MAX_INPUT_ITEMS_PER_REQUEST).enumerate() {
        let client_clone = client.clone();
        let model_clone = model.clone();
        let api_key_clone = api_key.clone();
        let api_base_clone = api_base.clone();
        let encoding_format_clone = encoding_format.clone();
        let dimensions_clone = dimensions;
        let user_clone = user.clone();
        let text_chunk_owned = text_chunk.to_vec();
        let semaphore_clone = Arc::clone(&semaphore);
        let current_chunk_start_index = chunk_index * OPENAI_MAX_INPUT_ITEMS_PER_REQUEST;

        // Tasks are spawned on the global runtime by default if not explicitly on another.
        // If send_single_embedding_request needs to be spawned on the specific runtime
        // held by SyncBeiClient (which is the GLOBAL_RUNTIME), it's implicitly handled.
        tasks.push(tokio::spawn(async move { // This will use the ambient runtime context
            let _permit = semaphore_clone.acquire().await.expect("Semaphore acquire failed");
            let mut response = send_single_embedding_request(
                client_clone,
                text_chunk_owned,
                model_clone,
                api_key_clone,
                api_base_clone,
                encoding_format_clone,
                dimensions_clone,
                user_clone,
            )
            .await?;

            for item in &mut response.data {
                item.index += current_chunk_start_index;
            }
            Result::<OpenAIEmbeddingsResponse, PyErr>::Ok(response)
        }));
    }

    let results = join_all(tasks).await;

    let mut all_embedding_data: Vec<OpenAIEmbeddingData> = Vec::with_capacity(total_texts);
    let mut aggregated_prompt_tokens: u32 = 0;
    let mut aggregated_total_tokens: u32 = 0;

    for result in results {
        match result {
            Ok(Ok(response_part)) => {
                all_embedding_data.extend(response_part.data);
                aggregated_prompt_tokens = aggregated_prompt_tokens.saturating_add(response_part.usage.prompt_tokens);
                aggregated_total_tokens = aggregated_total_tokens.saturating_add(response_part.usage.total_tokens);
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
fn truss_bei_client(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SyncBeiClient>()?;
    Ok(())
}

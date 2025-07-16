use crate::constants::*;
use crate::errors::ClientError;
use crate::http::*;
use crate::utils::*;
use futures::future::join_all;
use rand::Rng;
use reqwest::Client;
use std::cmp::min;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

pub struct SendRequestConfig {
    pub max_retries: u32,
    pub initial_backoff: Duration,
    pub retry_budget: Option<Arc<AtomicUsize>>,
    pub cancel_token: Arc<AtomicBool>,
}

#[derive(Clone)]
pub enum HttpClientWrapper {
    Http1(Arc<Client>),
    Http2(Arc<Vec<(Arc<AtomicUsize>, Arc<Client>)>>),
}

pub struct ClientGuard {
    client: Arc<Client>,
    counter: Option<Arc<AtomicUsize>>,
}

impl ClientGuard {
    fn new(client: Arc<Client>, counter: Option<Arc<AtomicUsize>>) -> Self {
        if let Some(ref counter) = counter {
            counter.fetch_add(1, Ordering::Relaxed);
        }
        Self { client, counter }
    }
}

impl Drop for ClientGuard {
    fn drop(&mut self) {
        if let Some(ref counter) = self.counter {
            counter.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl Deref for ClientGuard {
    type Target = Client;

    fn deref(&self) -> &Self::Target {
        &self.client
    }
}

impl HttpClientWrapper {
    pub fn get_client(&self) -> ClientGuard {
        match self {
            HttpClientWrapper::Http1(client) => ClientGuard::new(client.clone(), None),
            HttpClientWrapper::Http2(pool) => {
                if let Some((count, client)) = pool
                    .iter()
                    .find(|(count, _)| count.load(Ordering::Relaxed) < HTTP2_CLIENT_MAX_QUEUED)
                {
                    return ClientGuard::new(client.clone(), Some(count.clone()));
                }

                let (count, client) = pool
                    .iter()
                    .min_by_key(|(count, _)| count.load(Ordering::Relaxed))
                    .unwrap();

                ClientGuard::new(client.clone(), Some(count.clone()))
            }
        }
    }
}

#[derive(Clone)]
pub struct PerformanceClientCore {
    pub api_key: String,
    pub base_url: String,
    pub client_wrapper: HttpClientWrapper,
}

impl PerformanceClientCore {
    pub fn get_api_key(api_key: Option<String>) -> Result<String, ClientError> {
        if let Some(key) = api_key {
            return Ok(key);
        }
        if let Ok(key) = std::env::var("BASETEN_API_KEY") {
            return Ok(key);
        }
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            return Ok(key);
        }
        Err(ClientError::InvalidParameter(
            "API key not provided and no environment variable `BASETEN_API_KEY` found".to_string(),
        ))
    }

    pub fn cap_concurrency_baseten_staging(base_url: &str, concurrency_desired: usize) -> usize {
        if STAGING_ADDRESS
            .iter()
            .any(|provider| !provider.is_empty() && base_url.contains(provider))
        {
            let cap = rand::thread_rng().gen_range(16..=16384);
            min(cap, concurrency_desired)
        } else {
            concurrency_desired
        }
    }

    pub fn validate_and_get_timeout_duration(timeout_s: f64) -> Result<Duration, ClientError> {
        if !(MIN_REQUEST_TIMEOUT_S..=MAX_REQUEST_TIMEOUT_S).contains(&timeout_s) {
            return Err(ClientError::InvalidParameter(format!(
                "Timeout {:.3}s is outside the allowed range [{:.3}s, {:.3}s].",
                timeout_s, MIN_REQUEST_TIMEOUT_S, MAX_REQUEST_TIMEOUT_S
            )));
        }
        Ok(Duration::from_secs_f64(timeout_s))
    }

    pub fn validate_concurrency_parameters(
        max_concurrent_requests: usize,
        batch_size: usize,
        base_url: &str,
    ) -> Result<usize, ClientError> {
        let actual_concurrency =
            Self::cap_concurrency_baseten_staging(base_url, max_concurrent_requests);

        if max_concurrent_requests == 0 || max_concurrent_requests > MAX_CONCURRENCY_HIGH_BATCH {
            return Err(ClientError::InvalidParameter(format!(
                "max_concurrent_requests must be greater than 0 and less than or equal to {}",
                MAX_CONCURRENCY_HIGH_BATCH
            )));
        } else if batch_size == 0 || batch_size > MAX_BATCH_SIZE {
            return Err(ClientError::InvalidParameter(format!(
                "batch_size must be greater than 0 and less than or equal to {}",
                MAX_BATCH_SIZE
            )));
        } else if max_concurrent_requests > MAX_CONCURRENCY_LOW_BATCH
            && batch_size < CONCURRENCY_HIGH_BATCH_SWITCH
        {
            return Err(ClientError::InvalidParameter(format!(
                "max_concurrent_requests must be less than {} when batch_size is less than {}. Please be nice to the server side.",
                MAX_CONCURRENCY_LOW_BATCH, CONCURRENCY_HIGH_BATCH_SWITCH
            )));
        }
        Ok(actual_concurrency)
    }

    /// Validates common request parameters and returns validated values
    /// This consolidates validation logic used across all API methods
    pub fn validate_request_parameters(
        &self,
        max_concurrent_requests: usize,
        batch_size: usize,
        timeout_s: f64,
    ) -> Result<(usize, Duration), ClientError> {
        let validated_concurrency = Self::validate_concurrency_parameters(
            max_concurrent_requests,
            batch_size,
            &self.base_url,
        )?;

        let validated_timeout = Self::validate_and_get_timeout_duration(timeout_s)?;

        Ok((validated_concurrency, validated_timeout))
    }

    pub fn get_http_client(http_version: u8) -> Result<Client, ClientError> {
        let mut client_builder = Client::builder();

        if http_version == 2 {
            client_builder = client_builder
                .http2_initial_connection_window_size(HTTP2_WINDOW_SIZE)
                .http2_initial_stream_window_size(HTTP2_WINDOW_SIZE)
                .http2_max_frame_size(65_536)
                .http2_prior_knowledge();
        } else {
            client_builder = client_builder.http1_only();
        }

        client_builder
            .pool_max_idle_per_host(32_768)
            .pool_idle_timeout(Duration::from_secs(240))
            .tcp_nodelay(true)
            .user_agent(concat!(
                "baseten-performance-client/",
                env!("CARGO_PKG_VERSION")
            ))
            .build()
            .map_err(|e| ClientError::Network(format!("Failed to create HTTP client: {}", e)))
    }

    pub fn new(
        base_url: String,
        api_key: Option<String>,
        http_version: u8,
    ) -> Result<Self, ClientError> {
        let api_key = Self::get_api_key(api_key)?;

        if WARNING_SLOW_PROVIDERS
            .iter()
            .any(|&provider| base_url.contains(provider))
        {
            eprintln!(
                "Warning: Using {} as the base URL might be slow. Consider using baseten.com instead.",
                base_url
            );
        }

        let client_wrapper = if http_version == 2 {
            let mut pool = Vec::with_capacity(HTTP2_CLIENT_POOL_SIZE);
            for _ in 0..HTTP2_CLIENT_POOL_SIZE {
                let client = Self::get_http_client(2)?;
                pool.push((Arc::new(AtomicUsize::new(0)), Arc::new(client)));
            }
            HttpClientWrapper::Http2(Arc::new(pool))
        } else {
            let client = Self::get_http_client(1)?;
            HttpClientWrapper::Http1(Arc::new(client))
        };

        Ok(PerformanceClientCore {
            api_key,
            base_url,
            client_wrapper,
        })
    }

    // Core embeddings processing logic
    pub async fn process_embeddings_requests(
        &self,
        texts: Vec<String>,
        model: String,
        encoding_format: Option<String>,
        dimensions: Option<u32>,
        user: Option<String>,
        max_concurrent_requests: usize,
        batch_size: usize,
        request_timeout_duration: Duration,
    ) -> Result<(CoreOpenAIEmbeddingsResponse, Vec<Duration>), ClientError> {
        let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));
        let mut tasks = Vec::new();
        let total_texts = texts.len();
        let total_requests = (total_texts + batch_size - 1) / batch_size;
        let retry_budget = Arc::new(AtomicUsize::new(calculate_retry_timeout_budget(
            total_requests,
        )));
        let cancel_token = Arc::new(AtomicBool::new(false));
        let model_for_response = model.clone();

        for (batch_index, user_text_batch) in texts.chunks(batch_size).enumerate() {
            let client_wrapper_clone = self.client_wrapper.clone();
            let model_for_task = model.clone();
            let api_key_clone = self.api_key.clone();
            let base_url_clone = self.base_url.clone();
            let encoding_format_clone = encoding_format.clone();
            let dimensions_clone = dimensions;
            let user_clone = user.clone();
            let user_text_batch_owned = user_text_batch.to_vec();
            let semaphore_clone = Arc::clone(&semaphore);
            let cancel_token_clone = Arc::clone(&cancel_token);
            let retry_budget_clone = Arc::clone(&retry_budget);
            let individual_request_timeout = request_timeout_duration;
            let current_batch_absolute_start_index = batch_index * batch_size;

            tasks.push(tokio::spawn(async move {
                let _permit =
                    acquire_permit_or_cancel(semaphore_clone, cancel_token_clone.clone(), None)
                        .await?;
                let client = client_wrapper_clone.get_client();

                let request_time_start = Instant::now();
                let config = SendRequestConfig {
                    max_retries: MAX_HTTP_RETRIES,
                    initial_backoff: Duration::from_millis(INITIAL_BACKOFF_MS),
                    retry_budget: Some(retry_budget_clone),
                    cancel_token: cancel_token_clone.clone(),
                };

                let result = send_single_embedding_request(
                    &client,
                    user_text_batch_owned,
                    model_for_task,
                    api_key_clone,
                    base_url_clone,
                    encoding_format_clone,
                    dimensions_clone,
                    user_clone,
                    individual_request_timeout,
                    &config,
                )
                .await;
                let request_time_elapsed = request_time_start.elapsed();

                match result {
                    Ok(mut response) => {
                        for item in &mut response.data {
                            item.index += current_batch_absolute_start_index;
                        }
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

        let mut all_embedding_data: Vec<CoreOpenAIEmbeddingData> = Vec::with_capacity(total_texts);
        let mut aggregated_prompt_tokens: u32 = 0;
        let mut aggregated_total_tokens: u32 = 0;
        let mut individual_batch_durations: Vec<Duration> = Vec::new();
        let mut first_error: Option<ClientError> = None;

        for result in task_join_results {
            if let Some((response_part, duration_part)) =
                process_task_outcome(result, &mut first_error, &cancel_token)
            {
                all_embedding_data.extend(response_part.data);
                aggregated_prompt_tokens =
                    aggregated_prompt_tokens.saturating_add(response_part.usage.prompt_tokens);
                aggregated_total_tokens =
                    aggregated_total_tokens.saturating_add(response_part.usage.total_tokens);
                individual_batch_durations.push(duration_part);
            }
        }

        if let Some(err) = first_error {
            return Err(err);
        }

        all_embedding_data.sort_by_key(|d| d.index);
        let final_response = CoreOpenAIEmbeddingsResponse {
            object: "list".to_string(),
            data: all_embedding_data,
            model: model_for_response,
            usage: CoreOpenAIUsage {
                prompt_tokens: aggregated_prompt_tokens,
                total_tokens: aggregated_total_tokens,
            },
            total_time: None,
            individual_request_times: None,
        };
        Ok((final_response, individual_batch_durations))
    }

    pub async fn process_rerank_requests(
        &self,
        query: String,
        texts: Vec<String>,
        raw_scores: bool,
        return_text: bool,
        truncate: bool,
        truncation_direction: String,
        max_concurrent_requests: usize,
        batch_size: usize,
        request_timeout_duration: Duration,
    ) -> Result<(CoreRerankResponse, Vec<Duration>), ClientError> {
        let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));
        let mut tasks = Vec::new();
        let total_requests = (texts.len() + batch_size - 1) / batch_size;
        let retry_budget = Arc::new(AtomicUsize::new(calculate_retry_timeout_budget(
            total_requests,
        )));
        let cancel_token = Arc::new(AtomicBool::new(false));

        for (batch_idx, texts_batch) in texts.chunks(batch_size).enumerate() {
            let client_wrapper_clone = self.client_wrapper.clone();
            let query_clone = query.clone();
            let api_key_clone = self.api_key.clone();
            let base_url_clone = self.base_url.clone();
            let truncation_direction_clone = truncation_direction.clone();
            let texts_batch_owned = texts_batch.to_vec();
            let semaphore_clone = Arc::clone(&semaphore);
            let cancel_token_clone = Arc::clone(&cancel_token);
            let retry_budget_clone = Arc::clone(&retry_budget);
            let individual_request_timeout = request_timeout_duration;
            let current_batch_absolute_start_index = batch_idx * batch_size;

            tasks.push(tokio::spawn(async move {
                let _permit =
                    acquire_permit_or_cancel(semaphore_clone, cancel_token_clone.clone(), None)
                        .await?;
                let client = client_wrapper_clone.get_client();

                let request_time_start = Instant::now();
                let config = SendRequestConfig {
                    max_retries: MAX_HTTP_RETRIES,
                    initial_backoff: Duration::from_millis(INITIAL_BACKOFF_MS),
                    retry_budget: Some(retry_budget_clone),
                    cancel_token: cancel_token_clone.clone(),
                };
                let result = send_single_rerank_request(
                    &client,
                    query_clone,
                    texts_batch_owned,
                    raw_scores,
                    return_text,
                    truncate,
                    truncation_direction_clone,
                    api_key_clone,
                    base_url_clone,
                    individual_request_timeout,
                    &config,
                )
                .await;
                let request_time_elapsed = request_time_start.elapsed();

                match result {
                    Ok(mut batch_results) => {
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

        let mut all_results_data: Vec<CoreRerankResult> = Vec::new();
        let mut individual_batch_durations: Vec<Duration> = Vec::new();
        let mut first_error: Option<ClientError> = None;

        for result in task_join_results {
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
        let core_response = CoreRerankResponse::new(all_results_data, None, None);
        Ok((core_response, individual_batch_durations))
    }

    // Core classify processing logic
    pub async fn process_classify_requests(
        &self,
        inputs: Vec<String>,
        raw_scores: bool,
        truncate: bool,
        truncation_direction: String,
        max_concurrent_requests: usize,
        batch_size: usize,
        request_timeout_duration: Duration,
    ) -> Result<(CoreClassificationResponse, Vec<Duration>), ClientError> {
        let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));
        let mut tasks = Vec::new();
        let total_requests = (inputs.len() + batch_size - 1) / batch_size;
        let retry_budget = Arc::new(AtomicUsize::new(calculate_retry_timeout_budget(
            total_requests,
        )));
        let cancel_token = Arc::new(AtomicBool::new(false));

        for input_chunk_slice in inputs.chunks(batch_size) {
            let client_wrapper_clone = self.client_wrapper.clone();
            let api_key_clone = self.api_key.clone();
            let base_url_clone = self.base_url.clone();
            let truncation_direction_clone = truncation_direction.clone();
            let inputs_for_api_owned: Vec<Vec<String>> =
                input_chunk_slice.iter().map(|s| vec![s.clone()]).collect();
            let semaphore_clone = Arc::clone(&semaphore);
            let cancel_token_clone = Arc::clone(&cancel_token);
            let retry_budget_clone = Arc::clone(&retry_budget);
            let individual_request_timeout = request_timeout_duration;

            tasks.push(tokio::spawn(async move {
                let _permit =
                    acquire_permit_or_cancel(semaphore_clone, cancel_token_clone.clone(), None)
                        .await?;
                let client = client_wrapper_clone.get_client();

                let request_time_start = Instant::now();
                let config = SendRequestConfig {
                    max_retries: MAX_HTTP_RETRIES,
                    initial_backoff: Duration::from_millis(INITIAL_BACKOFF_MS),
                    retry_budget: Some(retry_budget_clone),
                    cancel_token: cancel_token_clone.clone(),
                };
                let result = send_single_classify_request(
                    &client,
                    inputs_for_api_owned,
                    raw_scores,
                    truncate,
                    truncation_direction_clone,
                    api_key_clone,
                    base_url_clone,
                    individual_request_timeout,
                    &config,
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

        let mut all_results_data: Vec<Vec<CoreClassificationResult>> = Vec::new();
        let mut individual_batch_durations: Vec<Duration> = Vec::new();
        let mut first_error: Option<ClientError> = None;

        for result in task_join_results {
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

        let core_response = CoreClassificationResponse::new(all_results_data, None, None);
        Ok((core_response, individual_batch_durations))
    }

    // Core batch post processing logic
    pub async fn process_batch_post_requests(
        &self,
        url_path: String,
        payloads_json: Vec<serde_json::Value>,
        max_concurrent_requests: usize,
        request_timeout_duration: Duration,
    ) -> Result<Vec<(serde_json::Value, std::collections::HashMap<String, String>, Duration)>, ClientError> {
        let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));
        let mut tasks = Vec::new();
        let cancel_token = Arc::new(AtomicBool::new(false));
        let total_payloads = payloads_json.len();
        let retry_budget = Arc::new(AtomicUsize::new(calculate_retry_timeout_budget(
            total_payloads,
        )));

        for (index, payload_item_json) in payloads_json.into_iter().enumerate() {
            let client_wrapper_clone = self.client_wrapper.clone();
            let api_key_clone = self.api_key.clone();
            let base_url_clone = self.base_url.clone();
            let url_path_clone = url_path.clone();
            let semaphore_clone = Arc::clone(&semaphore);
            let cancel_token_clone = Arc::clone(&cancel_token);
            let retry_budget_clone = Arc::clone(&retry_budget);
            let individual_request_timeout = request_timeout_duration;

            tasks.push(tokio::spawn(async move {
                let _permit =
                    acquire_permit_or_cancel(semaphore_clone, cancel_token_clone.clone(), None)
                        .await?;
                let client = client_wrapper_clone.get_client();

                let full_url = format!(
                    "{}/{}",
                    base_url_clone.trim_end_matches('/'),
                    url_path_clone.trim_start_matches('/')
                );
                let request_time_start = std::time::Instant::now();
                let config = SendRequestConfig {
                    max_retries: MAX_HTTP_RETRIES,
                    initial_backoff: Duration::from_millis(INITIAL_BACKOFF_MS),
                    retry_budget: Some(retry_budget_clone),
                    cancel_token: cancel_token_clone.clone(),
                };

                let result_tuple = send_single_batch_post_request(
                    &client,
                    full_url,
                    payload_item_json,
                    api_key_clone,
                    individual_request_timeout,
                    &config,
                )
                .await;

                let request_time_elapsed = request_time_start.elapsed();

                match result_tuple {
                    Ok((response_json_value, headers_map)) => {
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
        let mut indexed_results: Vec<(usize, serde_json::Value, std::collections::HashMap<String, String>, Duration)> =
            Vec::with_capacity(total_payloads);
        let mut first_error: Option<ClientError> = None;

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

        let final_results: Vec<(serde_json::Value, std::collections::HashMap<String, String>, Duration)> = indexed_results
            .into_iter()
            .map(|(_, val, headers, dur)| (val, headers, dur))
            .collect();

        Ok(final_results)
    }
}

// Helper functions for sending individual requests
async fn send_single_embedding_request(
    client: &Client,
    texts_batch: Vec<String>,
    model: String,
    api_key: String,
    base_url: String,
    encoding_format: Option<String>,
    dimensions: Option<u32>,
    user: Option<String>,
    request_timeout: Duration,
    config: &SendRequestConfig,
) -> Result<CoreOpenAIEmbeddingsResponse, ClientError> {
    let request_payload = CoreOpenAIEmbeddingsRequest {
        input: texts_batch,
        model,
        encoding_format,
        dimensions,
        user,
    };

    let url = format!("{}/v1/embeddings", base_url.trim_end_matches('/'));

    let request_builder = client
        .post(&url)
        .bearer_auth(api_key)
        .json(&request_payload)
        .timeout(request_timeout);

    let response = send_request_with_retry(request_builder, config).await?;
    let successful_response = ensure_successful_response(response).await?;

    successful_response
        .json::<CoreOpenAIEmbeddingsResponse>()
        .await
        .map_err(|e| ClientError::Serialization(format!("Failed to parse response JSON: {}", e)))
}

async fn send_single_rerank_request(
    client: &Client,
    query: String,
    texts_batch: Vec<String>,
    raw_scores: bool,
    return_text: bool,
    truncate: bool,
    truncation_direction: String,
    api_key: String,
    base_url: String,
    request_timeout: Duration,
    config: &SendRequestConfig,
) -> Result<Vec<CoreRerankResult>, ClientError> {
    let request_payload = CoreRerankRequest {
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
        .bearer_auth(api_key)
        .json(&request_payload)
        .timeout(request_timeout);

    let response = send_request_with_retry(request_builder, config).await?;
    let successful_response = ensure_successful_response(response).await?;

    successful_response
        .json::<Vec<CoreRerankResult>>()
        .await
        .map_err(|e| ClientError::Serialization(format!("Failed to parse rerank response JSON: {}", e)))
}

async fn send_single_classify_request(
    client: &Client,
    inputs: Vec<Vec<String>>,
    raw_scores: bool,
    truncate: bool,
    truncation_direction: String,
    api_key: String,
    base_url: String,
    request_timeout: Duration,
    config: &SendRequestConfig,
) -> Result<Vec<Vec<CoreClassificationResult>>, ClientError> {
    let request_payload = CoreClassifyRequest {
        inputs,
        raw_scores,
        truncate,
        truncation_direction,
    };

    let url = format!("{}/predict", base_url.trim_end_matches('/'));

    let request_builder = client
        .post(&url)
        .bearer_auth(api_key)
        .json(&request_payload)
        .timeout(request_timeout);

    let response = send_request_with_retry(request_builder, config).await?;
    let successful_response = ensure_successful_response(response).await?;

    successful_response
        .json::<Vec<Vec<CoreClassificationResult>>>()
        .await
        .map_err(|e| ClientError::Serialization(format!("Failed to parse classify response JSON: {}", e)))
}

async fn send_single_batch_post_request(
    client: &Client,
    full_url: String,
    payload_json: serde_json::Value,
    api_key: String,
    request_timeout: Duration,
    config: &SendRequestConfig,
) -> Result<(serde_json::Value, std::collections::HashMap<String, String>), ClientError> {
    let request_builder = client
        .post(&full_url)
        .bearer_auth(api_key)
        .json(&payload_json)
        .timeout(request_timeout);

    let response = send_request_with_retry(request_builder, config).await?;
    let successful_response = ensure_successful_response(response).await?;

    // Extract headers
    let mut headers_map = std::collections::HashMap::new();
    for (name, value) in successful_response.headers().iter() {
        headers_map.insert(
            name.as_str().to_string(),
            String::from_utf8_lossy(value.as_bytes()).into_owned(),
        );
    }

    let response_json_value: serde_json::Value = successful_response
        .json::<serde_json::Value>()
        .await
        .map_err(|e| ClientError::Serialization(format!("Failed to parse response JSON: {}", e)))?;

    Ok((response_json_value, headers_map))
}

async fn ensure_successful_response(
    response: reqwest::Response,
) -> Result<reqwest::Response, ClientError> {
    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        Err(ClientError::Http {
            status: status.as_u16(),
            message: format!("API request failed with status {}: {}", status, error_text),
        })
    } else {
        Ok(response)
    }
}

async fn send_request_with_retry(
    request_builder: reqwest::RequestBuilder,
    config: &SendRequestConfig,
) -> Result<reqwest::Response, ClientError> {
    let mut retries_done = 0;
    let mut current_backoff = config.initial_backoff;
    let max_retries = config.max_retries;

    loop {
        let request_builder_clone = request_builder.try_clone().ok_or_else(|| {
            ClientError::Network("Failed to clone request builder for retry".to_string())
        })?;

        match request_builder_clone.send().await {
            Ok(response) => {
                if response.status().is_success() {
                    return Ok(response);
                }

                let status = response.status();
                if status.as_u16() == 429 || status.is_server_error() {
                    if retries_done >= max_retries {
                        return ensure_successful_response(response).await;
                    }
                } else if status.is_client_error() {
                    config.cancel_token.store(true, Ordering::SeqCst);
                    return ensure_successful_response(response).await;
                } else {
                    return ensure_successful_response(response).await;
                }
            }
            Err(network_err) => {
                if network_err.is_timeout() {
                    if let Some(retry_budget) = config.retry_budget.as_ref() {
                        if retry_budget.fetch_sub(1, Ordering::SeqCst) < 1 {
                            config.cancel_token.store(true, Ordering::SeqCst);
                            return Err(ClientError::Timeout(format!(
                                "Request timed out and retry budget is exhausted: {}",
                                network_err
                            )));
                        }
                    }
                }

                if retries_done >= 2 {
                    config.cancel_token.store(true, Ordering::SeqCst);
                    return Err(ClientError::from(network_err));
                }
            }
        }

        retries_done += 1;
        let backoff_duration = current_backoff.min(MAX_BACKOFF_DURATION);
        tokio::time::sleep(backoff_duration).await;
        current_backoff = current_backoff.saturating_mul(4);
    }
}

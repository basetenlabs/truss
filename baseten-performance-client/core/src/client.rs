use crate::constants::*;
use crate::errors::ClientError;
use crate::http::*;
use crate::split_policy::*;
use crate::utils::*;
use futures::future::join_all;
use reqwest::Client;
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
    pub hedge_budget: Option<(Arc<AtomicUsize>, Duration)>,
    pub timeout: Duration,
}

impl SendRequestConfig {
    /// Create a new SendRequestConfig with validation
    pub fn new(
        max_retries: u32,
        initial_backoff: Duration,
        retry_budget: Option<Arc<AtomicUsize>>,
        cancel_token: Arc<AtomicBool>,
        hedge_budget: Option<(Arc<AtomicUsize>, Duration)>,
        timeout: Duration,
    ) -> Result<Self, ClientError> {
        // Validate that hedging timeout is higher than request timeout
        if let Some((_, hedge_timeout)) = &hedge_budget {
            if hedge_timeout <= &timeout {
                return Err(ClientError::InvalidParameter(format!(
                    "Hedge timeout ({:.3}s) must be higher than request timeout ({:.3}s)",
                    hedge_timeout.as_secs_f64(),
                    timeout.as_secs_f64()
                )));
            }
        }

        Ok(SendRequestConfig {
            max_retries,
            initial_backoff,
            retry_budget,
            cancel_token,
            hedge_budget,
            timeout,
        })
    }
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
    ) -> Result<usize, ClientError> {
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
        Ok(max_concurrent_requests)
    }

    /// Validates common request parameters and returns validated values
    /// This consolidates validation logic used across all API methods
    pub fn validate_request_parameters(
        &self,
        max_concurrent_requests: usize,
        batch_size: usize,
        timeout_s: f64,
    ) -> Result<(usize, Duration), ClientError> {
        let validated_concurrency =
            Self::validate_concurrency_parameters(max_concurrent_requests, batch_size)?;

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
}

impl PerformanceClientCore {
    // Generic batch processing method - handles pre-batched requests for ALL API types
    async fn process_batched_requests<T, R>(
        &self,
        batches: Vec<Vec<String>>,
        config: &RequestProcessingConfig,
        create_payload: impl Fn(Vec<String>) -> T + Send + Sync + 'static,
        endpoint_url: impl Fn(&str) -> String + Send + Sync + 'static,
        adjust_indices: impl Fn(&mut R, usize) + Send + Sync + 'static,
    ) -> Result<(R, Vec<Duration>, Duration), ClientError>
    where
        T: serde::Serialize + Send + 'static,
        R: serde::de::DeserializeOwned + Combinable + Send + 'static,
    {
        let start_time = std::time::Instant::now();
        let request_timeout_duration = config.timeout_duration();

        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_requests));
        let mut tasks = Vec::new();
        let total_requests = batches.len();
        let retry_budget = Arc::new(AtomicUsize::new(calculate_retry_timeout_budget(
            total_requests,
        )));
        let cancel_token = Arc::new(AtomicBool::new(false));
        let hedge_config: Option<(Arc<AtomicUsize>, Duration)> = {
            match config.hedge_delay {
                Some(delay) => Some((
                    Arc::new(AtomicUsize::new(calculate_hedge_budget(
                        total_requests,
                    ))),
                    Duration::from_secs_f64(delay),
                )),
                None => None,
            }
        };

        let mut current_absolute_index = 0;
        for batch in batches {
            let current_batch_absolute_start_index = current_absolute_index;
            current_absolute_index += batch.len();

            let client_wrapper_clone = self.client_wrapper.clone();
            let api_key_clone = self.api_key.clone();
            let base_url_clone = self.base_url.clone();
            let semaphore_clone = Arc::clone(&semaphore);
            let cancel_token_clone = Arc::clone(&cancel_token);
            let retry_budget_clone = Arc::clone(&retry_budget);
            let hedge_config_clone = hedge_config.clone();

            // Clone the closures to move them into the async block
            let payload = create_payload(batch.clone());
            let url = endpoint_url(&base_url_clone);

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
                    hedge_budget: hedge_config_clone,
                    timeout: request_timeout_duration,
                };

                // Send request with pre-created payload and URL
                let response: R = send_http_request_with_retry(
                    &client,
                    url,
                    payload,
                    api_key_clone,
                    request_timeout_duration,
                    &config,
                )
                .await?;

                let request_time_elapsed = request_time_start.elapsed();
                Ok((
                    response,
                    request_time_elapsed,
                    current_batch_absolute_start_index,
                ))
            }));
        }

        let task_join_results = join_all(tasks).await;
        let mut responses = Vec::new();
        let mut individual_batch_durations: Vec<Duration> = Vec::new();
        let mut first_error: Option<ClientError> = None;

        for result in task_join_results {
            match result {
                Ok(Ok((mut response, duration, start_index))) => {
                    adjust_indices(&mut response, start_index);
                    responses.push(response);
                    individual_batch_durations.push(duration);
                }
                Ok(Err(e)) => {
                    cancel_token.store(true, Ordering::SeqCst);
                    first_error = Some(e);
                    break;
                }
                Err(e) => {
                    first_error = Some(ClientError::Network(format!("Task join error: {}", e)));
                    break;
                }
            }
        }

        if let Some(err) = first_error {
            return Err(err);
        }

        let combined_response = R::combine(responses);
        let total_time = start_time.elapsed();

        Ok((combined_response, individual_batch_durations, total_time))
    }

    // Helper to create batches with policy and config
    fn create_batches_with_config(
        &self,
        inputs: Vec<String>,
        config: &RequestProcessingConfig,
    ) -> Vec<Vec<String>> {
        if let Some(max_chars) = config.max_chars_per_request {
            let policy =
                SplitPolicy::max_chars_per_request(max_chars, config.max_concurrent_requests);
            inputs.split(&policy)
        } else {
            inputs
                .chunks(config.batch_size)
                .map(|chunk| chunk.to_vec())
                .collect()
        }
    }
    // Core embeddings processing logic with unified interface
    pub async fn process_embeddings_requests(
        &self,
        texts: Vec<String>,
        model: String,
        encoding_format: Option<String>,
        dimensions: Option<u32>,
        user: Option<String>,
        max_concurrent_requests: usize,
        batch_size: usize,
        max_chars_per_request: Option<usize>,
        timeout_s: f64,
        hedge_delay: Option<f64>,
    ) -> Result<(CoreOpenAIEmbeddingsResponse, Vec<Duration>, Duration), ClientError> {
        // Create and validate config
        let config = RequestProcessingConfig::new(
            max_concurrent_requests,
            batch_size,
            timeout_s,
            self.base_url.clone(),
            hedge_delay,
            max_chars_per_request,
        )?;

        // Create batches
        let batches = self.create_batches_with_config(texts, &config);

        let (mut response, durations, total_time) = self
            .process_batched_requests(
                batches,
                &config,
                move |batch| CoreOpenAIEmbeddingsRequest {
                    input: batch,
                    model: model.clone(),
                    encoding_format: encoding_format.clone(),
                    dimensions,
                    user: user.clone(),
                },
                |base_url| format!("{}/v1/embeddings", base_url.trim_end_matches('/')),
                |response: &mut CoreOpenAIEmbeddingsResponse, start_index| {
                    for item in &mut response.data {
                        item.index += start_index;
                    }
                },
            )
            .await?;

        // Set timing information
        response.total_time = total_time.as_secs_f64();
        response.individual_request_times = durations.iter().map(|d| d.as_secs_f64()).collect();

        Ok((response, durations, total_time))
    }

    // Core rerank processing logic with unified interface
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
        max_chars_per_request: Option<usize>,
        timeout_s: f64,
        hedge_delay: Option<f64>,
    ) -> Result<(CoreRerankResponse, Vec<Duration>, Duration), ClientError> {
        // Create and validate config
        let config = RequestProcessingConfig::new(
            max_concurrent_requests,
            batch_size,
            timeout_s,
            self.base_url.clone(),
            hedge_delay,
            max_chars_per_request,
        )?;

        // Create batches
        let batches = self.create_batches_with_config(texts, &config);

        let (results, durations, total_time) = self
            .process_batched_requests(
                batches,
                &config,
                move |batch| CoreRerankRequest {
                    query: query.clone(),
                    texts: batch,
                    raw_scores,
                    return_text,
                    truncate,
                    truncation_direction: truncation_direction.clone(),
                },
                |base_url| format!("{}/rerank", base_url.trim_end_matches('/')),
                |results: &mut Vec<CoreRerankResult>, start_index| {
                    for item in results {
                        item.index += start_index;
                    }
                },
            )
            .await?;

        // Convert Vec<CoreRerankResult> to CoreRerankResponse
        let mut response = CoreRerankResponse::new(results, None, None);

        // Set timing information
        response.total_time = total_time.as_secs_f64();
        response.individual_request_times = durations.iter().map(|d| d.as_secs_f64()).collect();

        Ok((response, durations, total_time))
    }

    // Core classify processing logic with unified interface
    pub async fn process_classify_requests(
        &self,
        inputs: Vec<String>,
        raw_scores: bool,
        truncate: bool,
        truncation_direction: String,
        max_concurrent_requests: usize,
        batch_size: usize,
        max_chars_per_request: Option<usize>,
        timeout_s: f64,
        hedge_delay: Option<f64>,
    ) -> Result<(CoreClassificationResponse, Vec<Duration>, Duration), ClientError> {
        // Create and validate config
        let config = RequestProcessingConfig::new(
            max_concurrent_requests,
            batch_size,
            timeout_s,
            self.base_url.clone(),
            hedge_delay,
            max_chars_per_request,
        )?;

        // Create batches
        let batches = self.create_batches_with_config(inputs, &config);

        let (results, durations, total_time) = self
            .process_batched_requests(
                batches,
                &config,
                move |batch| {
                    // Convert each string to Vec<String> as expected by the API
                    let inputs: Vec<Vec<String>> = batch.into_iter().map(|s| vec![s]).collect();
                    CoreClassifyRequest {
                        inputs,
                        raw_scores,
                        truncate,
                        truncation_direction: truncation_direction.clone(),
                    }
                },
                |base_url| format!("{}/predict", base_url.trim_end_matches('/')),
                |_results: &mut Vec<Vec<CoreClassificationResult>>, _start_index| {
                    // Classification responses don't have index fields to adjust
                },
            )
            .await?;

        // Convert Vec<Vec<CoreClassificationResult>> to CoreClassificationResponse
        let mut response = CoreClassificationResponse::new(results, None, None);

        // Set timing information
        response.total_time = total_time.as_secs_f64();
        response.individual_request_times = durations.iter().map(|d| d.as_secs_f64()).collect();

        Ok((response, durations, total_time))
    }

    // Core batch post processing
    // TODO: Use unified processing system for batch post requests
    pub async fn process_batch_post_requests(
        &self,
        url_path: String,
        payloads_json: Vec<serde_json::Value>,
        max_concurrent_requests: usize,
        timeout_s: f64,
        hedge_delay: Option<f64>,
    ) -> Result<
        (
            Vec<(
                serde_json::Value,
                std::collections::HashMap<String, String>,
                Duration,
            )>,
            Duration,
        ),
        ClientError,
    > {
        // todo: use unified processing system, but not allowing char based policy.
        let start_time = std::time::Instant::now();

        // Validate parameters internally (using batch_size of 128 for validation)
        let (validated_concurrency, request_timeout_duration) =
            self.validate_request_parameters(max_concurrent_requests, 128, timeout_s)?;
        let semaphore = Arc::new(Semaphore::new(validated_concurrency));
        let mut tasks = Vec::new();
        let cancel_token = Arc::new(AtomicBool::new(false));
        let total_payloads = payloads_json.len();
        let retry_budget = Arc::new(AtomicUsize::new(calculate_retry_timeout_budget(
            total_payloads,
        )));
        let hedge_budget_delay: Option<(Arc<AtomicUsize>, Duration)> = hedge_delay.map(|delay| {
            (
                Arc::new(AtomicUsize::new(calculate_retry_timeout_budget(
                    total_payloads,
                ))),
                Duration::from_secs_f64(delay),
            )
        });
        for (index, payload_item_json) in payloads_json.into_iter().enumerate() {
            let client_wrapper_clone = self.client_wrapper.clone();
            let api_key_clone = self.api_key.clone();
            let base_url_clone = self.base_url.clone();
            let url_path_clone = url_path.clone();
            let semaphore_clone = Arc::clone(&semaphore);
            let cancel_token_clone = Arc::clone(&cancel_token);
            let retry_budget_clone = Arc::clone(&retry_budget);
            let individual_request_timeout = request_timeout_duration;
            let hedge_budget_clone = hedge_budget_delay.clone();

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
                    hedge_budget: hedge_budget_clone,
                    timeout: individual_request_timeout,
                };

                let result_tuple = send_http_request_with_headers(
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
                    Ok((response_json_value, headers_map)) => Ok((
                        index,
                        response_json_value,
                        headers_map,
                        request_time_elapsed,
                    )),
                    Err(e) => {
                        cancel_token_clone.store(true, Ordering::SeqCst);
                        Err(e)
                    }
                }
            }));
        }

        let task_join_results = join_all(tasks).await;
        let mut indexed_results: Vec<(
            usize,
            serde_json::Value,
            std::collections::HashMap<String, String>,
            Duration,
        )> = Vec::with_capacity(total_payloads);
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

        let final_results: Vec<(
            serde_json::Value,
            std::collections::HashMap<String, String>,
            Duration,
        )> = indexed_results
            .into_iter()
            .map(|(_, val, headers, dur)| (val, headers, dur))
            .collect();

        let total_time = start_time.elapsed();
        Ok((final_results, total_time))
    }
}

// Unified HTTP request helper
async fn send_http_request_with_retry<T, R>(
    client: &Client,
    url: String,
    payload: T,
    api_key: String,
    request_timeout: Duration,
    config: &SendRequestConfig,
) -> Result<R, ClientError>
where
    T: serde::Serialize,
    R: serde::de::DeserializeOwned,
{
    let request_builder = client
        .post(&url)
        .bearer_auth(api_key)
        .json(&payload)
        .timeout(request_timeout);

    // let response = send_request_with_retry(request_builder, config).await?;
    // TODO race tokio tasks: sleep(hedge_delay) + request_with_retry + request, get the first one.
    let response = send_request_with_retry(request_builder, config).await?;

    let successful_response = ensure_successful_response(response).await?;

    successful_response
        .json::<R>()
        .await
        .map_err(|e| ClientError::Serialization(format!("Failed to parse response JSON: {}", e)))
}

// Unified HTTP request helper with headers extraction
async fn send_http_request_with_headers<T>(
    client: &Client,
    url: String,
    payload: T,
    api_key: String,
    request_timeout: Duration,
    config: &SendRequestConfig,
) -> Result<(serde_json::Value, std::collections::HashMap<String, String>), ClientError>
where
    T: serde::Serialize,
{
    let request_builder = client
        .post(&url)
        .bearer_auth(api_key)
        .json(&payload)
        .timeout(request_timeout);

    let response = send_request_with_retry(request_builder, config).await?;
    // hedge here with a second workstream, awaiting the first one or time of hedge
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

        // Only hedge on the first request (retries_done == 0)
        let should_hedge = retries_done == 0 && config.hedge_budget.is_some();

        let response = if should_hedge {
            send_request_with_hedging(request_builder_clone, config).await?
        } else {
            request_builder_clone
                .send()
                .await
                .map_err(ClientError::from)?
        };

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

        retries_done += 1;
        let backoff_duration = current_backoff.min(MAX_BACKOFF_DURATION);
        tokio::time::sleep(backoff_duration).await;
        current_backoff = current_backoff.saturating_mul(4);
    }
}

async fn send_request_with_hedging(
    request_builder: reqwest::RequestBuilder,
    config: &SendRequestConfig,
) -> Result<reqwest::Response, ClientError> {
    let (hedge_budget, hedge_delay) = config.hedge_budget.as_ref().unwrap();

    // Check if we have hedge budget available
    if hedge_budget.load(Ordering::SeqCst) <= 0 {
        // No hedge budget, use normal request
        return request_builder.send().await.map_err(ClientError::from);
    }

    let request_builder_hedge = request_builder.try_clone().ok_or_else(|| {
        ClientError::Network("Failed to clone request builder for hedging".to_string())
    })?;

    // Start the original request
    let mut original_request = tokio::spawn(async move { request_builder.send().await });

    // Wait for hedge delay
    let hedge_timer = tokio::time::sleep(*hedge_delay);

    tokio::select! {
        // Original request completed before hedge delay
        result = &mut original_request => {
            match result {
                Ok(response_result) => response_result.map_err(ClientError::from),
                Err(join_err) => Err(ClientError::Network(format!("Original request task failed: {}", join_err))),
            }
        }
        // Hedge delay expired, start hedged request
        _ = hedge_timer => {
            // Decrement hedge budget
            if hedge_budget.fetch_sub(1, Ordering::SeqCst) > 0 {
                let mut hedge_request = tokio::spawn(async move {
                    request_builder_hedge.send().await
                });

                // Race between original and hedged request
                tokio::select! {
                    result = &mut original_request => {
                        // Cancel hedge request if original completes first
                        hedge_request.abort();
                        match result {
                            Ok(response_result) => response_result.map_err(ClientError::from),
                            Err(join_err) => Err(ClientError::Network(format!("Original request task failed: {}", join_err))),
                        }
                    }
                    result = &mut hedge_request => {
                        // Hedge request completed first
                        original_request.abort();
                        match result {
                            Ok(response_result) => response_result.map_err(ClientError::from),
                            Err(join_err) => Err(ClientError::Network(format!("Hedge request task failed: {}", join_err))),
                        }
                    }
                }
            } else {
                // No hedge budget left, wait for original request
                match original_request.await {
                    Ok(response_result) => response_result.map_err(ClientError::from),
                    Err(join_err) => Err(ClientError::Network(format!("Original request task failed: {}", join_err))),
                }
            }
        }
    }
}

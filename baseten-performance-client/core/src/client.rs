use crate::constants::*;
use crate::errors::ClientError;
use crate::http::*;
use crate::http_client::*;
use crate::split_policy::*;
use crate::utils::{
    acquire_permit_or_cancel, calculate_hedge_budget, calculate_retry_timeout_budget,
    process_joinset_outcome,
};

use reqwest::Client;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

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
    pub base_url: Arc<str>,
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
            .pool_max_idle_per_host(8192)
            .pool_idle_timeout(Duration::from_secs(30))
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

        if WARNING_SLOW_PROVIDERS
            .iter()
            .any(|&provider| base_url.contains(provider))
        {
            eprintln!(
                "Warning: Using {} as the base URL might be slow. Consider using baseten.com instead.",
                base_url
            );
        }

        Ok(PerformanceClientCore {
            api_key,
            base_url: base_url.into(),
            client_wrapper,
        })
    }
}

impl PerformanceClientCore {
    // Generic batch processing method - handles pre-batched requests for ALL API types
    // Uses JoinSet for better cancellation and lazy execution
    async fn process_batched_requests<T, R>(
        &self,
        batches: Vec<Vec<String>>,
        config: &RequestProcessingConfig,
        create_payload: impl Fn(Vec<String>) -> T + Send + Sync + 'static,
        endpoint_url: Arc<str>,
        adjust_indices: impl Fn(&mut R, usize) + Send + Sync + 'static,
    ) -> Result<(R, Vec<Duration>, Duration), ClientError>
    where
        T: serde::Serialize + Send + 'static,
        R: serde::de::DeserializeOwned + Combinable + Send + 'static,
    {
        let start_time = std::time::Instant::now();
        let request_timeout_duration = config.timeout_duration();

        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_requests));
        let total_requests = batches.len();
        let retry_budget = Arc::new(AtomicUsize::new(calculate_retry_timeout_budget(
            total_requests,
        )));
        let cancel_token = Arc::new(AtomicBool::new(false));
        let hedge_config: Option<(Arc<AtomicUsize>, Duration)> = config.hedge_delay.map(|delay| {
            (
                Arc::new(AtomicUsize::new(calculate_hedge_budget(total_requests))),
                Duration::from_secs_f64(delay),
            )
        });

        // Calculate expected capacity from original batches for pre-allocation
        let expected_capacity: usize = batches.iter().map(|batch| batch.len()).sum();

        // Use JoinSet for better performance and cancellation
        let mut join_set: JoinSet<Result<(R, Duration, usize, usize), ClientError>> =
            JoinSet::new();
        let mut indexed_results: Vec<(usize, R, Duration, usize)> =
            Vec::with_capacity(total_requests);

        let mut current_absolute_index = 0;
        for (batch_index, batch) in batches.into_iter().enumerate() {
            let current_batch_absolute_start_index = current_absolute_index;
            current_absolute_index += batch.len();

            // Only clone what's needed inside the async block
            let client_wrapper = self.client_wrapper.clone();
            let api_key = self.api_key.clone();
            let semaphore = Arc::clone(&semaphore);
            let cancel_token = Arc::clone(&cancel_token);
            let retry_budget = Arc::clone(&retry_budget);
            let hedge_config = hedge_config.clone();

            // Create payload and URL outside async block
            let payload = create_payload(batch);
            let url = endpoint_url.clone();

            join_set.spawn(async move {
                let _permit =
                    acquire_permit_or_cancel(semaphore, cancel_token.clone(), None).await?;
                let client = client_wrapper.get_client();

                let request_time_start = Instant::now();
                let config = SendRequestConfig {
                    max_retries: MAX_HTTP_RETRIES,
                    initial_backoff: Duration::from_millis(INITIAL_BACKOFF_MS),
                    retry_budget: retry_budget,
                    cancel_token: cancel_token.clone(),
                    hedge_budget: hedge_config,
                    timeout: request_timeout_duration,
                };

                // Send request with pre-created payload and URL
                let response: R = send_http_request_with_retry(
                    &client,
                    url.to_string(),
                    payload,
                    api_key,
                    request_timeout_duration,
                    &config,
                )
                .await?;

                let request_time_elapsed = request_time_start.elapsed();
                Ok((
                    response,
                    request_time_elapsed,
                    current_batch_absolute_start_index,
                    batch_index, // Include batch index for ordering
                ))
            });
        }

        // Process results as they complete with fast-fail cancellation
        while let Some(task_result) = join_set.join_next().await {
            match process_joinset_outcome(task_result, &cancel_token) {
                Ok((response, duration, start_index, batch_index)) => {
                    indexed_results.push((batch_index, response, duration, start_index));
                }
                Err(e) => {
                    // Cancel all remaining tasks immediately
                    cancel_token.store(true, Ordering::SeqCst);
                    join_set.abort_all();
                    return Err(e);
                }
            }
        }

        // Sort results by original batch order to preserve ordering
        indexed_results.sort_by_key(|&(batch_index, _, _, _)| batch_index);

        // Extract responses and durations in correct order
        let mut responses = Vec::with_capacity(total_requests);
        let mut individual_batch_durations = Vec::with_capacity(total_requests);

        for (_, mut response, duration, start_index) in indexed_results {
            adjust_indices(&mut response, start_index);
            responses.push(response);
            individual_batch_durations.push(duration);
        }

        let combined_response = R::combine(responses, expected_capacity);
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
            let policy = SplitPolicy::max_chars_per_request(max_chars, config.batch_size);
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
            self.base_url.to_string(),
            hedge_delay,
            max_chars_per_request,
        )?;

        // Create batches
        let batches = self.create_batches_with_config(texts, &config);

        // Pre-compute endpoint URL
        let endpoint_url: Arc<str> =
            format!("{}/v1/embeddings", config.base_url.trim_end_matches('/')).into();

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
                endpoint_url,
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
            self.base_url.to_string(),
            hedge_delay,
            max_chars_per_request,
        )?;

        // Create batches
        let batches = self.create_batches_with_config(texts, &config);

        // Pre-compute endpoint URL
        let endpoint_url: Arc<str> =
            format!("{}/rerank", config.base_url.trim_end_matches('/')).into();

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
                endpoint_url,
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
            self.base_url.to_string(),
            hedge_delay,
            max_chars_per_request,
        )?;

        // Create batches
        let batches = self.create_batches_with_config(inputs, &config);

        // Pre-compute endpoint URL
        let endpoint_url: Arc<str> =
            format!("{}/predict", config.base_url.trim_end_matches('/')).into();

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
                endpoint_url,
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

    // Core batch post processing - optimized with JoinSet
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
        let start_time = std::time::Instant::now();

        // Validate parameters internally (using batch_size of 128 for validation)
        let (validated_concurrency, request_timeout_duration) =
            self.validate_request_parameters(max_concurrent_requests, 128, timeout_s)?;
        let semaphore = Arc::new(Semaphore::new(validated_concurrency));
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

        // Use JoinSet for better performance and cancellation
        let mut join_set: JoinSet<
            Result<
                (
                    usize,
                    serde_json::Value,
                    std::collections::HashMap<String, String>,
                    Duration,
                ),
                ClientError,
            >,
        > = JoinSet::new();
        let mut indexed_results: Vec<(
            usize,
            serde_json::Value,
            std::collections::HashMap<String, String>,
            Duration,
        )> = Vec::with_capacity(total_payloads);

        for (index, payload_item_json) in payloads_json.into_iter().enumerate() {
            let client_wrapper = self.client_wrapper.clone();
            let api_key = self.api_key.clone();
            let base_url = self.base_url.clone();
            let url_path = url_path.clone();
            let semaphore = Arc::clone(&semaphore);
            let cancel_token = Arc::clone(&cancel_token);
            let retry_budget = Arc::clone(&retry_budget);
            let individual_request_timeout = request_timeout_duration;
            let hedge_budget = hedge_budget_delay.clone();

            join_set.spawn(async move {
                let _permit =
                    acquire_permit_or_cancel(semaphore, cancel_token.clone(), None).await?;
                let client = client_wrapper.get_client();

                let full_url = format!(
                    "{}/{}",
                    base_url.trim_end_matches('/'),
                    url_path.trim_start_matches('/')
                );
                let request_time_start = std::time::Instant::now();
                let config = SendRequestConfig {
                    max_retries: MAX_HTTP_RETRIES,
                    initial_backoff: Duration::from_millis(INITIAL_BACKOFF_MS),
                    retry_budget: retry_budget,
                    cancel_token: cancel_token.clone(),
                    hedge_budget,
                    timeout: individual_request_timeout,
                };

                let result_tuple = send_http_request_with_headers(
                    &client,
                    full_url,
                    payload_item_json,
                    api_key,
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
                        cancel_token.store(true, Ordering::SeqCst);
                        Err(e)
                    }
                }
            });
        }

        // Process results as they complete with fast-fail cancellation
        while let Some(task_result) = join_set.join_next().await {
            match process_joinset_outcome(task_result, &cancel_token) {
                Ok(indexed_data) => {
                    indexed_results.push(indexed_data);
                }
                Err(e) => {
                    // Cancel all remaining tasks immediately
                    cancel_token.store(true, Ordering::SeqCst);
                    join_set.abort_all();
                    return Err(e);
                }
            }
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

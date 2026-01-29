use crate::cancellation::JoinSetGuard;
use crate::constants::*;
use crate::errors::ClientError;
use crate::http::*;
use crate::http_client::*;
use crate::split_policy::*;
use crate::utils::process_joinset_outcome;

use reqwest::Client;
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tracing;

/// HTTP client wrapper that supports both HTTP/1.1 and HTTP/2
///
/// This enum provides a unified interface for different HTTP client implementations:
/// - `Http1`: Single HTTP/1.1 client for simple use cases
/// - `Http2`: Pool of HTTP/2 clients for high-performance concurrent requests
///
/// Sharing among multiple baseten-performance-client instances is supported and encouraged via `Arc`.
#[derive(Clone)]
pub enum HttpClientWrapper {
    Http1(Arc<Client>),
    Http2(Arc<Vec<(Arc<AtomicUsize>, Arc<Client>)>>),
}

/// RAII guard for HTTP client with reference counting
/// Ensures the client is kept alive as long as any requests are using it
///
/// This is an internal implementation detail for managing HTTP client lifetime.
/// Users typically don't need to interact with this directly.
pub(crate) struct ClientGuard {
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
    pub fn new(http_version: u8) -> Result<Arc<Self>, ClientError> {
        let wrapper = if http_version == 2 {
            let mut pool = Vec::with_capacity(HTTP2_CLIENT_POOL_SIZE);
            for _ in 0..HTTP2_CLIENT_POOL_SIZE {
                let client = PerformanceClientCore::get_http_client(2)?;
                pool.push((Arc::new(AtomicUsize::new(0)), Arc::new(client)));
            }
            HttpClientWrapper::Http2(Arc::new(pool))
        } else {
            let client = PerformanceClientCore::get_http_client(1)?;
            HttpClientWrapper::Http1(Arc::new(client))
        };
        Ok(Arc::new(wrapper))
    }

    /// Get an HTTP client with proper reference counting
    ///
    /// This returns a guard that automatically manages client lifetime and
    /// reference counting. The client is kept alive until the guard is dropped.
    ///
    /// This is an internal method used by the core request processing logic.
    /// External users typically don't need to call this directly.
    pub(crate) fn get_client(&self) -> ClientGuard {
        match self {
            HttpClientWrapper::Http1(client) => ClientGuard::new(client.clone(), None),
            HttpClientWrapper::Http2(pool) => {
                if let Some((count, client)) = pool
                    .iter()
                    .find(|(count, _)| count.load(Ordering::Relaxed) < HTTP2_CLIENT_OPTIMUM_QUEUED)
                {
                    // return the first client with available high-speed capacity
                    // assures that open connections are all on first n clients, and others are closed if low
                    // load. This helps with connection management in networking stack.
                    return ClientGuard::new(client.clone(), Some(count.clone()));
                }
                // If all clients are busy, pick the least loaded one

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
/// High-performance client for making concurrent API requests with advanced features
///
/// This is the main entry point for using the performance client library. It provides:
/// - Automatic request batching and concurrency management
/// - Built-in retry logic with exponential backoff
/// - Request hedging for improved latency
/// - Support for multiple API endpoints (embeddings, rerank, classification)
/// - HTTP/1.1 and HTTP/2 support
/// - Comprehensive error handling and metrics
///
/// # Example
///
/// ```rust,no_run
/// use baseten_performance_client_core::{PerformanceClientCore, RequestProcessingPreference};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create client with default settings
/// let client = PerformanceClientCore::new(
///     "https://api.example.com".to_string(),
///     Some("your-api-key".to_string()),
///     2, // HTTP/2
///     None,
/// )?;
///
/// // Configure processing preferences
/// let preference = RequestProcessingPreference::new()
///     .with_max_concurrent_requests(10)
///     .with_batch_size(5);
///
/// // Make requests
/// let (response, durations, headers, total_time) = client
///     .process_embeddings_requests(
///         vec!["text1".to_string(), "text2".to_string()],
///         "text-embedding-ada-002".to_string(),
///         None, None, None,
///         &preference,
///     )
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct PerformanceClientCore {
    /// API key for authentication
    ///
    /// This can be set during client creation or read from the
    /// `PERFORMANCE_CLIENT_API_KEY` environment variable.
    pub api_key: String,

    /// Base URL for the API endpoint
    ///
    /// Should include the protocol (https://) and domain, but not the
    /// specific endpoint path.
    pub base_url: Arc<str>,

    /// Internal HTTP client wrapper
    ///
    /// This manages the underlying HTTP client(s) and connection pooling.
    /// It's exposed as public for advanced use cases but typically shouldn't
    /// be modified directly.
    pub client_wrapper: Arc<HttpClientWrapper>,
}

impl PerformanceClientCore {
    pub fn new(
        base_url: String,
        api_key: Option<String>,
        http_version: u8,
        client_wrapper: Option<Arc<HttpClientWrapper>>,
    ) -> Result<Self, ClientError> {
        let api_key = Self::get_api_key(api_key)?;

        let client_wrapper = if let Some(wrapper) = client_wrapper {
            wrapper
        } else {
            HttpClientWrapper::new(http_version)?
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
        total_timeout_s: Option<f64>,
    ) -> Result<(usize, Duration), ClientError> {
        let validated_concurrency =
            Self::validate_concurrency_parameters(max_concurrent_requests, batch_size)?;

        let validated_timeout = Self::validate_and_get_timeout_duration(timeout_s)?;

        if let Some(total_timeout) = total_timeout_s {
            let validated_total_timeout = Self::validate_and_get_timeout_duration(total_timeout)?;
            if validated_total_timeout < validated_timeout {
                return Err(ClientError::InvalidParameter(format!(
                    "total_timeout_s ({:.3}s) must be greater than or equal to timeout_s ({:.3}s).",
                    total_timeout, timeout_s
                )));
            }
        }

        Ok((validated_concurrency, validated_timeout))
    }

    pub fn get_client_wrapper(&self) -> Arc<HttpClientWrapper> {
        Arc::clone(&self.client_wrapper)
    }

    pub fn get_http_client(http_version: u8) -> Result<Client, ClientError> {
        let mut client_builder = Client::builder();

        if http_version == 2 {
            client_builder = client_builder
                .http2_initial_connection_window_size(HTTP2_WINDOW_SIZE)
                .http2_initial_stream_window_size(HTTP2_WINDOW_SIZE)
                .http2_max_frame_size(65_536)
                .http2_prior_knowledge()
                .pool_max_idle_per_host(16);
        } else {
            client_builder = client_builder.http1_only().pool_max_idle_per_host(512);
        }

        client_builder
            .pool_idle_timeout(Duration::from_secs(30))
            .tcp_nodelay(true)
            .user_agent(concat!(
                "baseten-performance-client/",
                env!("CARGO_PKG_VERSION")
            ))
            .build()
            .map_err(|e| ClientError::Network(format!("Failed to create HTTP client: {}", e)))
    }
}

impl PerformanceClientCore {
    // Generic batch processing method - handles pre-batched requests for ALL API types
    // Uses JoinSetGuard for automatic cancellation on drop (RAII pattern)
    #[allow(clippy::too_many_arguments)]
    async fn process_batched_requests<T, R>(
        &self,
        batches: Vec<Vec<String>>,
        config: &RequestProcessingConfig,
        create_payload: impl Fn(Vec<String>) -> T + Send + Sync + 'static,
        endpoint_url: Arc<str>,
        adjust_indices: impl Fn(&mut R, usize) + Send + Sync + 'static,
        total_timeout: Option<Duration>,
    ) -> Result<(R, Vec<Duration>, Vec<HeaderMap>, Duration), ClientError>
    where
        T: serde::Serialize + Send + 'static,
        R: serde::de::DeserializeOwned + Combinable + Send + 'static,
    {
        let start_time = std::time::Instant::now();
        let request_timeout_duration = config.timeout_duration();

        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_requests));
        let total_requests = batches.len();

        // Update budgets with actual total_requests count
        config.update_budgets(total_requests);

        if config.hedge_delay.is_some() {
            tracing::debug!("Hedging enabled with delay: {:?}", config.hedge_delay);
        }

        let expected_capacity: usize = batches.iter().map(|batch| batch.len()).sum();

        tracing::debug!(
            "initial budgets before requests: retry_budget={} hedge_budget={} customer_id={}",
            config
                .retry_budget
                .load(std::sync::atomic::Ordering::SeqCst),
            config
                .hedge_budget
                .load(std::sync::atomic::Ordering::SeqCst),
            config.customer_request_id.to_string()
        );

        #[allow(clippy::type_complexity)]
        let mut join_set: JoinSetGuard<
            Result<(R, Duration, usize, usize, HeaderMap), ClientError>,
        > = JoinSetGuard::with_cancel_token(config.cancel_token.clone());
        let mut indexed_results: Vec<(usize, R, Duration, usize, HeaderMap)> =
            Vec::with_capacity(total_requests);

        let mut current_absolute_index = 0;
        for (batch_index, batch) in batches.into_iter().enumerate() {
            // Clone config for this iteration
            let config_clone = config.clone();
            let current_batch_absolute_start_index = current_absolute_index;
            current_absolute_index += batch.len();

            // Only clone what's needed inside the async block
            let client_wrapper = self.client_wrapper.clone();
            let api_key = self.api_key.clone();
            let semaphore = Arc::clone(&semaphore);

            // Create payload and URL outside async block
            let payload = create_payload(batch);
            let url = endpoint_url.clone();

            // Generate individual request ID for this batch
            let request_customer_id = config_clone.create_request_customer_id(batch_index);

            join_set.spawn(async move {
                let _permit = semaphore
                    .acquire_owned()
                    .await
                    .map_err(|e| ClientError::Network(format!("Semaphore closed: {}", e)))?;
                let client = client_wrapper.get_client();

                let request_time_start = Instant::now();

                // Send request with pre-created payload and URL
                let (response, headers): (R, HeaderMap) = send_http_request_with_retry(
                    &client,
                    url.to_string(),
                    payload,
                    api_key,
                    request_timeout_duration,
                    &config_clone,
                    request_customer_id,
                )
                .await?;

                let request_time_elapsed = request_time_start.elapsed();
                Ok((
                    response,
                    request_time_elapsed,
                    current_batch_absolute_start_index,
                    batch_index,
                    headers,
                ))
            });
        }

        // Process results as they complete, racing against total timeout if specified
        loop {
            let task_result = if let Some(timeout_duration) = total_timeout {
                let remaining = timeout_duration.saturating_sub(start_time.elapsed());
                if remaining.is_zero() {
                    return Err(ClientError::LocalTimeout(
                        format!(
                            "Total operation timed out after {:.2}s",
                            timeout_duration.as_secs_f64()
                        ),
                        None,
                    ));
                }
                match tokio::time::timeout(remaining, join_set.join_next()).await {
                    Ok(Some(result)) => result,
                    Ok(None) => break, // All tasks completed
                    Err(_) => {
                        return Err(ClientError::LocalTimeout(
                            format!(
                                "Total operation timed out after {:.2}s",
                                timeout_duration.as_secs_f64()
                            ),
                            Some(config.customer_request_id.to_string()),
                        ));
                    }
                }
            } else {
                match join_set.join_next().await {
                    Some(result) => result,
                    None => break, // All tasks completed
                }
            };

            match process_joinset_outcome(task_result) {
                Ok((response, duration, start_index, batch_index, headers)) => {
                    indexed_results.push((batch_index, response, duration, start_index, headers));
                }
                Err(e) => {
                    // JoinSetGuard::drop will abort remaining tasks automatically
                    return Err(e);
                }
            }
        }

        tracing::debug!(
            "Remaining budgets after requests: retry_budget={} hedge_budget={} customer_id={}",
            config
                .retry_budget
                .load(std::sync::atomic::Ordering::SeqCst),
            config
                .hedge_budget
                .load(std::sync::atomic::Ordering::SeqCst),
            config.customer_request_id.to_string()
        );

        // Sort results by original batch order to preserve ordering
        indexed_results.sort_by_key(|&(batch_index, _, _, _, _)| batch_index);
        // Extract responses and durations in correct order
        let mut responses = Vec::with_capacity(total_requests);
        let mut individual_batch_durations = Vec::with_capacity(total_requests);
        let mut collected_headers = Vec::with_capacity(total_requests);

        for (_, mut response, duration, start_index, headers) in indexed_results {
            adjust_indices(&mut response, start_index);
            responses.push(response);
            individual_batch_durations.push(duration);
            collected_headers.push(headers);
        }

        let combined_response = R::combine(responses, expected_capacity);
        let total_time = start_time.elapsed();

        Ok((
            combined_response,
            individual_batch_durations,
            collected_headers,
            total_time,
        ))
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
    // Cancellation: dropping this future will automatically abort all in-flight requests
    pub async fn process_embeddings_requests(
        &self,
        texts: Vec<String>,
        model: String,
        encoding_format: Option<String>,
        dimensions: Option<u32>,
        user: Option<String>,
        preference: &RequestProcessingPreference,
    ) -> Result<
        (
            CoreOpenAIEmbeddingsResponse,
            Vec<Duration>,
            Vec<HeaderMap>,
            Duration,
        ),
        ClientError,
    > {
        // Create and validate config from preference
        let config = preference.pair_with_request_validate_and_convert(
            self.base_url.to_string(),
            texts.len(),
            self.api_key.clone(),
        )?;
        // Create batches
        let batches = self.create_batches_with_config(texts, &config);

        // Pre-compute endpoint URL
        let endpoint_url: Arc<str> =
            format!("{}/v1/embeddings", config.base_url.trim_end_matches('/')).into();

        let total_timeout = config.total_timeout_duration();

        let (mut response, durations, headers, total_time) = self
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
                total_timeout,
            )
            .await?;

        // Set timing information
        response.total_time = total_time.as_secs_f64();
        response.individual_request_times = durations.iter().map(|d| d.as_secs_f64()).collect();
        response.response_headers = headers.clone();

        Ok((response, durations, headers, total_time))
    }

    // Core rerank processing logic with unified interface
    // Cancellation: dropping this future will automatically abort all in-flight requests
    #[allow(clippy::too_many_arguments)]
    pub async fn process_rerank_requests(
        &self,
        query: String,
        texts: Vec<String>,
        raw_scores: bool,
        model: Option<String>,
        return_text: bool,
        truncate: bool,
        truncation_direction: String,
        preference: &RequestProcessingPreference,
    ) -> Result<(CoreRerankResponse, Vec<Duration>, Vec<HeaderMap>, Duration), ClientError> {
        // Create and validate config from preference
        let config = preference.pair_with_request_validate_and_convert(
            self.base_url.to_string(),
            texts.len(),
            self.api_key.clone(),
        )?;

        // Create batches
        let batches = self.create_batches_with_config(texts, &config);

        // Pre-compute endpoint URL
        let endpoint_url: Arc<str> =
            format!("{}/rerank", config.base_url.trim_end_matches('/')).into();

        let total_timeout = config.total_timeout_duration();

        let (results, durations, headers, total_time) = self
            .process_batched_requests(
                batches,
                &config,
                move |batch| CoreRerankRequest {
                    query: query.clone(),
                    texts: batch,
                    model: model.clone(),
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
                total_timeout,
            )
            .await?;

        // Convert Vec<CoreRerankResult> to CoreRerankResponse
        let mut response = CoreRerankResponse::new(results, None, None);

        // Set timing information
        response.total_time = total_time.as_secs_f64();
        response.individual_request_times = durations.iter().map(|d| d.as_secs_f64()).collect();
        response.response_headers = headers.clone();

        Ok((response, durations, headers, total_time))
    }

    // Core classify processing logic with unified interface
    // Cancellation: dropping this future will automatically abort all in-flight requests
    pub async fn process_classify_requests(
        &self,
        inputs: Vec<String>,
        model: Option<String>,
        raw_scores: bool,
        truncate: bool,
        truncation_direction: String,
        preference: &RequestProcessingPreference,
    ) -> Result<
        (
            CoreClassificationResponse,
            Vec<Duration>,
            Vec<HeaderMap>,
            Duration,
        ),
        ClientError,
    > {
        // Create and validate config from preference
        let config = preference.pair_with_request_validate_and_convert(
            self.base_url.to_string(),
            inputs.len(),
            self.api_key.clone(),
        )?;

        // Create batches
        let batches = self.create_batches_with_config(inputs, &config);

        // Pre-compute endpoint URL
        let endpoint_url: Arc<str> =
            format!("{}/predict", config.base_url.trim_end_matches('/')).into();

        let total_timeout = config.total_timeout_duration();

        let (results, durations, headers, total_time) = self
            .process_batched_requests(
                batches,
                &config,
                move |batch| {
                    // Convert each string to Vec<String> as expected by the API
                    let inputs: Vec<Vec<String>> = batch.into_iter().map(|s| vec![s]).collect();
                    CoreClassifyRequest {
                        inputs,
                        model: model.clone(),
                        raw_scores,
                        truncate,
                        truncation_direction: truncation_direction.clone(),
                    }
                },
                endpoint_url,
                |_results: &mut Vec<Vec<CoreClassificationResult>>, _start_index| {
                    // Classification responses don't have index fields to adjust
                },
                total_timeout,
            )
            .await?;

        // Convert Vec<Vec<CoreClassificationResult>> to CoreClassificationResponse
        let mut response = CoreClassificationResponse::new(results, None, None);

        // Set timing information
        response.total_time = total_time.as_secs_f64();
        response.individual_request_times = durations.iter().map(|d| d.as_secs_f64()).collect();
        response.response_headers = headers.clone();

        Ok((response, durations, headers, total_time))
    }

    // Core batch post processing - optimized with JoinSetGuard
    // Cancellation: dropping this future will automatically abort all in-flight requests
    pub async fn process_batch_post_requests(
        &self,
        url_path: String,
        payloads_json: Vec<serde_json::Value>,
        preference: &RequestProcessingPreference,
        method: crate::http::HttpMethod,
    ) -> Result<(Vec<(serde_json::Value, HeaderMap, Duration)>, Duration), ClientError> {
        let start_time = std::time::Instant::now();
        let total_payloads = payloads_json.len();

        // Create and validate config from preference
        let config = preference.pair_with_request_validate_and_convert(
            self.base_url.to_string(),
            total_payloads,
            self.api_key.clone(),
        )?;

        let total_timeout = config.total_timeout_duration();
        let request_timeout_duration = config.timeout_duration();
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_requests));

        // JoinSetGuard automatically aborts all tasks and sets cancel_token on drop
        let mut join_set: JoinSetGuard<
            Result<(usize, serde_json::Value, HeaderMap, Duration), ClientError>,
        > = JoinSetGuard::with_cancel_token(config.cancel_token.clone());
        let mut indexed_results: Vec<(usize, serde_json::Value, HeaderMap, Duration)> =
            Vec::with_capacity(total_payloads);

        for (index, payload_item_json) in payloads_json.into_iter().enumerate() {
            // Clone config for this iteration
            let config_clone = config.clone();
            let client_wrapper = self.client_wrapper.clone();
            let api_key = self.api_key.clone();
            let base_url = self.base_url.clone();
            let url_path = url_path.clone();
            let semaphore: Arc<Semaphore> = Arc::clone(&semaphore);
            let individual_request_timeout = request_timeout_duration;
            let method = method;

            // Generate individual request ID for this batch
            let request_customer_id = config.create_request_customer_id(index);

            join_set.spawn(async move {
                let _permit = semaphore
                    .acquire_owned()
                    .await
                    .map_err(|e| ClientError::Network(format!("Semaphore closed: {}", e)))?;
                let client = client_wrapper.get_client();

                let full_url = format!(
                    "{}/{}",
                    base_url.trim_end_matches('/'),
                    url_path.trim_start_matches('/')
                );
                let request_time_start = std::time::Instant::now();

                let result_tuple = send_http_request_with_headers(
                    &client,
                    full_url,
                    payload_item_json,
                    api_key,
                    individual_request_timeout,
                    &config_clone,
                    request_customer_id,
                    method,
                )
                .await;

                let request_time_elapsed = request_time_start.elapsed();

                result_tuple.map(|(response_json_value, headers_map)| {
                    (
                        index,
                        response_json_value,
                        headers_map,
                        request_time_elapsed,
                    )
                })
            });
        }

        // Process results as they complete, racing against total timeout if specified
        loop {
            let task_result = if let Some(timeout_duration) = total_timeout {
                let remaining = timeout_duration.saturating_sub(start_time.elapsed());
                if remaining.is_zero() {
                    return Err(ClientError::LocalTimeout(
                        format!(
                            "Total operation timed out after {:.2}s",
                            timeout_duration.as_secs_f64()
                        ),
                        None,
                    ));
                }
                match tokio::time::timeout(remaining, join_set.join_next()).await {
                    Ok(Some(result)) => result,
                    Ok(None) => break,
                    Err(_) => {
                        return Err(ClientError::LocalTimeout(
                            format!(
                                "Total operation timed out after {:.2}s",
                                timeout_duration.as_secs_f64()
                            ),
                            Some(config.customer_request_id.to_string()),
                        ));
                    }
                }
            } else {
                match join_set.join_next().await {
                    Some(result) => result,
                    None => break,
                }
            };

            match process_joinset_outcome(task_result) {
                Ok(indexed_data) => {
                    indexed_results.push(indexed_data);
                }
                Err(e) => {
                    // JoinSetGuard::drop will abort remaining tasks automatically
                    return Err(e);
                }
            }
        }

        indexed_results.sort_by_key(|&(original_index, _, _, _)| original_index);

        let final_results: Vec<(serde_json::Value, HeaderMap, Duration)> = indexed_results
            .into_iter()
            .map(|(_, val, headers, dur)| (val, headers, dur))
            .collect();

        let total_time = start_time.elapsed();
        Ok((final_results, total_time))
    }
}

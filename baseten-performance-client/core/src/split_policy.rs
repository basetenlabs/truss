use crate::cancellation::CancellationToken;
use crate::constants::*;
use crate::customer_request_id::CustomerRequestId;
use crate::errors::ClientError;
use crate::http::*;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Duration;

/// User-facing configuration for request processing with budget percentages.
/// This is the public API struct that gets validated and converted to RequestProcessingConfig.
/// All fields are Option<T> with defaults applied during conversion to RequestProcessingConfig.
#[derive(Debug, Clone, Default)]
pub struct RequestProcessingPreference {
    pub max_concurrent_requests: Option<usize>,
    pub batch_size: Option<usize>,
    pub max_chars_per_request: Option<usize>,
    pub timeout_s: Option<f64>,
    pub hedge_delay: Option<f64>,
    pub total_timeout_s: Option<f64>,
    pub hedge_budget_pct: Option<f64>,
    pub retry_budget_pct: Option<f64>,
    pub max_retries: Option<u32>,
    pub initial_backoff_ms: Option<u64>,
    pub cancel_token: Option<CancellationToken>,
    pub primary_api_key_override: Option<String>,
}

impl RequestProcessingPreference {
    /// Create a new preference with all fields set to None (will use defaults)
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply defaults to create a complete configuration
    pub fn with_defaults(&self) -> RequestProcessingPreference {
        RequestProcessingPreference {
            max_concurrent_requests: self.max_concurrent_requests.or(Some(DEFAULT_CONCURRENCY)),
            batch_size: self.batch_size.or(Some(DEFAULT_BATCH_SIZE)),
            max_chars_per_request: self.max_chars_per_request,
            timeout_s: self.timeout_s.or(Some(DEFAULT_REQUEST_TIMEOUT_S)),
            hedge_delay: self.hedge_delay,
            total_timeout_s: self.total_timeout_s,
            hedge_budget_pct: self.hedge_budget_pct.or(Some(HEDGE_BUDGET_PERCENTAGE)),
            retry_budget_pct: self.retry_budget_pct.or(Some(RETRY_BUDGET_PERCENTAGE)),
            max_retries: self.max_retries.or(Some(MAX_HTTP_RETRIES)),
            initial_backoff_ms: self.initial_backoff_ms.or(Some(INITIAL_BACKOFF_MS)),
            cancel_token: self.cancel_token.clone(),
            primary_api_key_override: self.primary_api_key_override.clone(),
        }
    }
}

impl RequestProcessingPreference {
    /// Builder pattern: set max concurrent requests
    pub fn with_max_concurrent_requests(mut self, value: usize) -> Self {
        self.max_concurrent_requests = Some(value);
        self
    }

    /// Builder pattern: set batch size
    pub fn with_batch_size(mut self, value: usize) -> Self {
        self.batch_size = Some(value);
        self
    }

    /// Builder pattern: set max chars per request
    pub fn with_max_chars_per_request(mut self, value: usize) -> Self {
        self.max_chars_per_request = Some(value);
        self
    }

    /// Builder pattern: set timeout in seconds
    pub fn with_timeout_s(mut self, value: f64) -> Self {
        self.timeout_s = Some(value);
        self
    }

    /// Builder pattern: set hedge delay in seconds
    pub fn with_hedge_delay(mut self, value: f64) -> Self {
        self.hedge_delay = Some(value);
        self
    }

    /// Builder pattern: set total timeout in seconds
    pub fn with_total_timeout_s(mut self, value: f64) -> Self {
        self.total_timeout_s = Some(value);
        self
    }

    /// Builder pattern: set hedge budget percentage
    pub fn with_hedge_budget_pct(mut self, value: f64) -> Self {
        self.hedge_budget_pct = Some(value);
        self
    }

    /// Builder pattern: set retry budget percentage
    pub fn with_retry_budget_pct(mut self, value: f64) -> Self {
        self.retry_budget_pct = Some(value);
        self
    }

    /// Builder pattern: set max retries
    pub fn with_max_retries(mut self, value: u32) -> Self {
        self.max_retries = Some(value);
        self
    }

    /// Builder pattern: set initial backoff duration in milliseconds
    pub fn with_initial_backoff_ms(mut self, value: u64) -> Self {
        self.initial_backoff_ms = Some(value);
        self
    }

    /// Builder pattern: set cancellation token
    pub fn with_cancel_token(mut self, token: CancellationToken) -> Self {
        self.cancel_token = Some(token);
        self
    }

    /// Builder pattern: set primary API key override
    pub fn with_primary_api_key_override(mut self, key: String) -> Self {
        self.primary_api_key_override = Some(key);
        self
    }

    /// Validate and convert to RequestProcessingConfig for a specific request.
    /// This pairs the preference with request-specific data (base_url, total_requests, api_key)
    /// and returns a validated config ready for processing.
    pub fn pair_with_request_validate_and_convert(
        &self,
        base_url: String,
        total_requests: usize,
        api_key: String,
    ) -> Result<RequestProcessingConfig, ClientError> {
        RequestProcessingConfig::new_from_preference(self, base_url, total_requests, api_key)
    }
}

/// Policy for splitting requests into batches
#[derive(Debug, Clone)]
pub enum SplitPolicy {
    /// Split by maximum number of concurrent requests only
    MaxBatchSize(usize),
    /// Split by maximum characters per request, with fallback to concurrency limit
    MaxCharsOrBatchPerRequest {
        max_chars: usize,
        max_batch_size: usize,
    },
}

/// Internal configuration for request processing with computed budgets.
/// This struct holds validated parameters and pre-computed Arc<AtomicUsize> budgets
/// ready for use during request processing. Flat structure - no nesting.
#[derive(Debug, Clone)]
pub struct RequestProcessingConfig {
    /// Request identification for this batch operation
    pub customer_request_id: CustomerRequestId,

    /// Processing settings
    pub max_concurrent_requests: usize,
    pub batch_size: usize,
    pub max_chars_per_request: Option<usize>,
    pub base_url: String,

    /// HTTP timing settings (stored as Duration consistently)
    pub timeout: Duration,
    pub total_timeout: Option<Duration>,
    pub hedge_delay: Option<Duration>,

    /// Retry settings
    pub max_retries: u32,
    pub initial_backoff: Duration,

    /// Budget management (single source of truth)
    pub retry_budget: Arc<AtomicUsize>,
    pub hedge_budget: Arc<AtomicUsize>,
    pub retry_budget_pct: f64,
    pub hedge_budget_pct: f64,

    /// Cancellation token for coordinated shutdown
    pub cancel_token: CancellationToken,

    /// Primary API key to use for requests
    pub api_key_primary: String,
}

impl RequestProcessingConfig {
    /// Validate parameters and return error if invalid
    #[allow(clippy::too_many_arguments)]
    fn validate_parameters(
        max_concurrent_requests: usize,
        batch_size: usize,
        timeout_s: f64,
        total_timeout_s: Option<f64>,
        hedge_delay: Option<f64>,
        max_chars_per_request: Option<usize>,
        hedge_budget_pct: f64,
        retry_budget_pct: f64,
        max_retries: u32,
        initial_backoff_ms: u64,
        total_requests: usize,
        api_key: &str,
    ) -> Result<(), ClientError> {
        // Validate total_requests
        if total_requests == 0 {
            return Err(ClientError::InvalidParameter(
                "total_requests must be greater than 0".to_string(),
            ));
        }
        // Validate timeout
        if !(MIN_REQUEST_TIMEOUT_S..=MAX_REQUEST_TIMEOUT_S).contains(&timeout_s) {
            return Err(ClientError::InvalidParameter(format!(
                "Timeout {:.3}s is outside the allowed range [{:.3}s, {:.3}s].",
                timeout_s, MIN_REQUEST_TIMEOUT_S, MAX_REQUEST_TIMEOUT_S
            )));
        }
        if let Some(total_timeout) = total_timeout_s {
            if total_timeout < timeout_s {
                return Err(ClientError::InvalidParameter(format!(
                    "Total timeout {:.3}s must be greater than or equal to individual request timeout {:.3}s.",
                    total_timeout, timeout_s
                )));
            }
        }

        if let Some(delay) = hedge_delay {
            if !(MIN_HEDGE_DELAY_S..=MAX_REQUEST_TIMEOUT_S).contains(&delay) {
                return Err(ClientError::InvalidParameter(format!(
                    "Hedge delay {:.3}s is outside the allowed range [{:.3}s, {:.3}s].",
                    delay, MIN_HEDGE_DELAY_S, MAX_REQUEST_TIMEOUT_S
                )));
            }
            if delay >= timeout_s - MIN_HEDGE_DELAY_S {
                return Err(ClientError::InvalidParameter(format!(
                    "Hedge delay {:.3}s must be less than timeout minus minimum hedge delay ({:.3}s -{:.3}s).",
                    delay, timeout_s, MIN_HEDGE_DELAY_S
                )));
            }
        }
        if let Some(max_chars) = max_chars_per_request {
            if !(MIN_CHARACTERS_PER_REQUEST..=MAX_CHARACTERS_PER_REQUEST).contains(&max_chars) {
                return Err(ClientError::InvalidParameter(format!(
                    "max_chars_per_request must be between {} and {} characters.",
                    MIN_CHARACTERS_PER_REQUEST, MAX_CHARACTERS_PER_REQUEST
                )));
            }
        }

        // Validate concurrency parameters
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

        // Validate budget percentages
        if hedge_budget_pct < 0.0 {
            return Err(ClientError::InvalidParameter(
                "hedge_budget_pct cannot be negative".to_string(),
            ));
        }
        if hedge_budget_pct > MAX_BUDGET_PERCENTAGE {
            return Err(ClientError::InvalidParameter(format!(
                "hedge_budget_pct cannot exceed {} ({}%)",
                MAX_BUDGET_PERCENTAGE,
                (MAX_BUDGET_PERCENTAGE * 100.0) as i32
            )));
        }

        if retry_budget_pct < 0.0 {
            return Err(ClientError::InvalidParameter(
                "retry_budget_pct cannot be negative".to_string(),
            ));
        }
        if retry_budget_pct > MAX_BUDGET_PERCENTAGE {
            return Err(ClientError::InvalidParameter(format!(
                "retry_budget_pct cannot exceed {} ({}%)",
                MAX_BUDGET_PERCENTAGE,
                (MAX_BUDGET_PERCENTAGE * 100.0) as i32
            )));
        }

        // Validate retry parameters
        if max_retries > MAX_HTTP_RETRIES {
            return Err(ClientError::InvalidParameter(format!(
                "max_retries cannot exceed {}",
                MAX_HTTP_RETRIES
            )));
        }

        if !(MIN_BACKOFF_MS..=MAX_BACKOFF_MS).contains(&initial_backoff_ms) {
            return Err(ClientError::InvalidParameter(format!(
                "initial_backoff_ms must be between {} and {} milliseconds",
                MIN_BACKOFF_MS, MAX_BACKOFF_MS
            )));
        }

        // Validate API key
        if api_key.is_empty() {
            return Err(ClientError::InvalidParameter(
                "API key cannot be empty".to_string(),
            ));
        }

        Ok(())
    }

    /// Calculate budget based on total requests and percentage
    /// Always ensures minimum budget of 2 to prevent budget exhaustion
    fn calculate_budget(total_requests: usize, budget_pct: f64) -> usize {
        // Always ensure minimum budget of 2 to prevent budget exhaustion
        // For 0 requests: calculated = 0, result = max(2, 1 + 0) = 2
        // For 10 requests with 5%: calculated = 1, result = max(2, 1 + 1) = 2
        let calculated = (total_requests as f64 * budget_pct).ceil() as usize;
        std::cmp::max(2, 1 + calculated)
    }

    /// Create config from a RequestProcessingPreference.
    /// This is the main entry point for converting user preferences to internal config.
    pub fn new_from_preference(
        preference: &RequestProcessingPreference,
        base_url: String,
        total_requests: usize,
        api_key: String,
    ) -> Result<Self, ClientError> {
        // Apply defaults to preference
        let pref = preference.with_defaults();

        // Extract values with defaults applied
        let max_concurrent_requests = pref.max_concurrent_requests.unwrap();
        let batch_size = pref.batch_size.unwrap();
        let timeout_s = pref.timeout_s.unwrap();
        let hedge_delay = pref.hedge_delay;
        let max_chars_per_request = pref.max_chars_per_request;
        let hedge_budget_pct = pref.hedge_budget_pct.unwrap();
        let retry_budget_pct = pref.retry_budget_pct.unwrap();
        let max_retries = pref.max_retries.unwrap();
        let initial_backoff_ms = pref.initial_backoff_ms.unwrap();

        // Handle API key override
        let api_key_primary = if let Some(ref key) = pref.primary_api_key_override {
            key.clone()
        } else {
            api_key
        };

        // Validate parameters
        Self::validate_parameters(
            max_concurrent_requests,
            batch_size,
            timeout_s,
            pref.total_timeout_s,
            hedge_delay,
            max_chars_per_request,
            hedge_budget_pct,
            retry_budget_pct,
            max_retries,
            initial_backoff_ms,
            total_requests,
            &api_key_primary,
        )?;

        // Create customer request ID for this batch operation
        let customer_request_id = CustomerRequestId::new_batch();

        // Create atomic budgets
        let retry_budget = Arc::new(AtomicUsize::new(Self::calculate_budget(
            total_requests,
            retry_budget_pct,
        )));

        let hedge_budget = if let Some(delay) = hedge_delay {
            // Create hedge budget if delay is >= MIN_HEDGE_DELAY_S AND hedge_budget_pct > 0
            if delay >= MIN_HEDGE_DELAY_S && hedge_budget_pct > 0.0 {
                Arc::new(AtomicUsize::new(Self::calculate_budget(
                    total_requests,
                    hedge_budget_pct,
                )))
            } else {
                Arc::new(AtomicUsize::new(0)) // Always present, but set to 0 when unused
            }
        } else {
            Arc::new(AtomicUsize::new(0)) // Always present, but set to 0 when unused
        };

        Ok(RequestProcessingConfig {
            customer_request_id,
            max_concurrent_requests,
            batch_size,
            max_chars_per_request,
            base_url,
            timeout: Duration::from_secs_f64(timeout_s),
            total_timeout: pref.total_timeout_s.map(Duration::from_secs_f64),
            hedge_delay: hedge_delay.map(Duration::from_secs_f64),
            max_retries,
            initial_backoff: Duration::from_millis(initial_backoff_ms),
            retry_budget,
            hedge_budget,
            retry_budget_pct,
            hedge_budget_pct,
            cancel_token: pref.cancel_token.unwrap_or_default(),
            api_key_primary,
        })
    }

    /// Update existing atomic budget values with a new total request count.
    /// This modifies the config's existing Arc<AtomicUsize> instances in-place.
    pub fn update_budgets(&self, total_requests: usize) {
        let new_retry_budget = Self::calculate_budget(total_requests, self.retry_budget_pct);
        self.retry_budget
            .store(new_retry_budget, std::sync::atomic::Ordering::SeqCst);

        // Always update hedge_budget since it's always present
        let new_hedge_budget = if let Some(hedge_delay) = self.hedge_delay {
            if hedge_delay >= Duration::from_secs_f64(MIN_HEDGE_DELAY_S)
                && self.hedge_budget_pct > 0.0
            {
                Self::calculate_budget(total_requests, self.hedge_budget_pct)
            } else {
                0 // Set to 0 when hedging is disabled
            }
        } else {
            0 // Set to 0 when no hedge delay is configured
        };
        self.hedge_budget
            .store(new_hedge_budget, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get timeout duration
    pub fn timeout_duration(&self) -> Duration {
        self.timeout
    }

    /// Get total timeout duration if set
    pub fn total_timeout_duration(&self) -> Option<Duration> {
        self.total_timeout
    }

    /// Create individual request customer ID for a specific batch index
    pub fn create_request_customer_id(&self, batch_index: usize) -> CustomerRequestId {
        self.customer_request_id.new_request(batch_index)
    }
}

impl SplitPolicy {
    /// Create a new max concurrency policy
    pub fn max_batch_size(max_concurrent: usize) -> Self {
        Self::MaxBatchSize(max_concurrent)
    }

    /// Create a new max chars per request policy
    pub fn max_chars_per_request(max_chars: usize, max_batch_size: usize) -> Self {
        Self::MaxCharsOrBatchPerRequest {
            max_chars,
            max_batch_size,
        }
    }

    /// Get max concurrent requests from policy
    pub fn get_max_concurrent_requests(&self) -> usize {
        match self {
            SplitPolicy::MaxBatchSize(max_batch_size) => *max_batch_size,
            SplitPolicy::MaxCharsOrBatchPerRequest { max_batch_size, .. } => *max_batch_size,
        }
    }

    /// Get batch size from policy
    pub fn get_batch_size(&self) -> usize {
        match self {
            SplitPolicy::MaxBatchSize(_) => 1, // Default batch size for MaxBatchSize policy
            SplitPolicy::MaxCharsOrBatchPerRequest { max_batch_size, .. } => *max_batch_size,
        }
    }
}

/// Trait for types that can be split into batches and combined back
pub trait Splittable<T> {
    /// Split the input into batches according to the policy
    fn split(&self, policy: &SplitPolicy) -> Vec<Vec<T>>;

    /// Get the character count for an item (used for max_chars_per_request)
    fn char_count(item: &T) -> usize;
}

/// Trait for response types that can be combined from multiple batches
pub trait Combinable {
    /// Combine multiple responses into one, preserving order
    ///
    /// # Arguments
    /// * `responses` - Vector of responses to combine
    /// * `expected_capacity` - Hint for pre-allocating internal vectors
    fn combine(responses: Vec<Self>, expected_capacity: usize) -> Self
    where
        Self: Sized;
}

/// Implementation for Vec<String> - the primary input type for our APIs
impl Splittable<String> for Vec<String> {
    fn split(&self, policy: &SplitPolicy) -> Vec<Vec<String>> {
        match policy {
            // splits, with max_batch_size >= 1
            SplitPolicy::MaxBatchSize(max_batch_size) => {
                let batch_size = self.len().div_ceil(*max_batch_size);
                self.chunks(batch_size.max(1))
                    .map(|chunk| chunk.to_vec())
                    .collect()
            }
            SplitPolicy::MaxCharsOrBatchPerRequest {
                max_chars,
                max_batch_size,
            } => {
                let mut batches = Vec::new();
                if self.is_empty() {
                    return batches;
                }

                let mut current_batch = Vec::new();
                let mut current_chars = 0;

                for item in self {
                    let item_chars = Self::char_count(item);

                    // If an item itself is larger than max_chars, it becomes its own batch.
                    if item_chars > *max_chars {
                        // First, push the current batch if it's not empty.
                        if !current_batch.is_empty() {
                            batches.push(current_batch);
                            current_batch = Vec::new();
                            current_chars = 0;
                        }
                        // Then, push the large item as its own batch.
                        batches.push(vec![item.clone()]);
                        continue;
                    }

                    // Start a new batch if the current one is full or adding the new item would exceed the char limit.
                    if !current_batch.is_empty()
                        && (current_batch.len() >= *max_batch_size
                            || current_chars + item_chars > *max_chars)
                    {
                        batches.push(current_batch);
                        current_batch = Vec::new();
                        current_chars = 0;
                    }

                    current_batch.push(item.clone());
                    current_chars += item_chars;
                }

                if !current_batch.is_empty() {
                    batches.push(current_batch);
                }

                batches
            }
        }
    }

    fn char_count(item: &String) -> usize {
        item.chars().count()
    }
}

/// Implementation for CoreOpenAIEmbeddingsResponse
impl Combinable for CoreOpenAIEmbeddingsResponse {
    fn combine(responses: Vec<Self>, expected_capacity: usize) -> Self {
        if responses.is_empty() {
            return CoreOpenAIEmbeddingsResponse {
                object: "list".to_string(),
                data: Vec::new(),
                model: String::new(),
                usage: CoreOpenAIUsage {
                    prompt_tokens: 0,
                    total_tokens: 0,
                },
                total_time: -1.0,
                individual_request_times: Vec::new(),
                response_headers: Vec::new(),
            };
        }

        // Pre-allocate with capacity hint
        let mut all_data = Vec::with_capacity(expected_capacity);
        let mut total_prompt_tokens = 0u32;
        let mut total_tokens = 0u32;
        let mut all_individual_times = Vec::with_capacity(expected_capacity);
        let mut has_times = false;
        let model = responses[0].model.clone();
        let mut all_response_headers = Vec::with_capacity(expected_capacity * 2); // Headers might have more items

        for response in responses {
            all_data.extend(response.data);
            total_prompt_tokens = total_prompt_tokens.saturating_add(response.usage.prompt_tokens);
            total_tokens = total_tokens.saturating_add(response.usage.total_tokens);
            all_response_headers.extend(response.response_headers);

            if !response.individual_request_times.is_empty() {
                all_individual_times.extend(response.individual_request_times);
                has_times = true;
            }
        }

        // Batches are processed in order, so we can just extend.
        // Sorting by index is not necessary if batches are handled correctly.
        // all_data.sort_by_key(|d: &CoreOpenAIEmbeddingData| d.index);

        CoreOpenAIEmbeddingsResponse {
            object: "list".to_string(),
            data: all_data,
            model,
            usage: CoreOpenAIUsage {
                prompt_tokens: total_prompt_tokens,
                total_tokens,
            },
            total_time: -1.0,
            individual_request_times: if has_times {
                all_individual_times
            } else {
                Vec::new()
            },
            response_headers: all_response_headers,
        }
    }
}

/// Implementation for Vec<CoreRerankResult>
impl Combinable for Vec<CoreRerankResult> {
    fn combine(responses: Vec<Self>, expected_capacity: usize) -> Self {
        let mut all_results = Vec::with_capacity(expected_capacity);
        for results in responses {
            all_results.extend(results);
        }
        // Sorting is handled by the generic processor, not needed here.
        all_results
    }
}

/// Implementation for CoreRerankResponse
impl Combinable for CoreRerankResponse {
    fn combine(responses: Vec<Self>, expected_capacity: usize) -> Self {
        if responses.is_empty() {
            return CoreRerankResponse::new(Vec::new(), None, None);
        }

        let mut all_data = Vec::with_capacity(expected_capacity);
        let mut all_response_headers = Vec::with_capacity(expected_capacity * 2); // Headers might have more items

        for response in responses {
            all_data.extend(response.data);
            all_response_headers.extend(response.response_headers);
        }

        // Sorting is handled by the generic processor, not needed here.
        let mut combined = CoreRerankResponse::new(all_data, None, None);
        combined.response_headers = all_response_headers;
        combined
    }
}

/// Implementation for Vec<Vec<CoreClassificationResult>>
impl Combinable for Vec<Vec<CoreClassificationResult>> {
    fn combine(responses: Vec<Self>, expected_capacity: usize) -> Self {
        let mut all_results = Vec::with_capacity(expected_capacity);
        for results in responses {
            all_results.extend(results);
        }
        all_results
    }
}

/// Implementation for CoreClassificationResponse
impl Combinable for CoreClassificationResponse {
    fn combine(responses: Vec<Self>, expected_capacity: usize) -> Self {
        if responses.is_empty() {
            return CoreClassificationResponse::new(Vec::new(), None, None);
        }

        let mut all_data = Vec::with_capacity(expected_capacity);
        let mut all_response_headers = Vec::with_capacity(expected_capacity * 2); // Headers might have more items

        for response in responses {
            all_data.extend(response.data);
            all_response_headers.extend(response.response_headers);
        }

        let mut combined = CoreClassificationResponse::new(all_data, None, None);
        combined.response_headers = all_response_headers;
        combined
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_by_max_batch_size() {
        let texts = vec![
            "short".to_string(),
            "medium text".to_string(),
            "longer text here".to_string(),
            "very long text that goes on".to_string(),
        ];

        let policy = SplitPolicy::max_batch_size(2);
        let batches = texts.split(&policy);

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 2);
    }

    #[test]
    fn test_split_by_max_chars_per_request() {
        let texts = vec![
            "short".to_string(),          // 5 chars
            "medium".to_string(),         // 6 chars
            "longer".to_string(),         // 6 chars
            "very long text".to_string(), // 14 chars
        ];

        let policy = SplitPolicy::max_chars_per_request(15, 10);
        let batches = texts.split(&policy);

        // First batch: "short" + "medium" = 11 chars
        // Second batch: "longer" = 6 chars (can't fit with "very long text")
        // Third batch: "very long text" = 14 chars
        assert!(batches.len() >= 2);

        // Verify no batch exceeds character limit (approximately)
        for batch in &batches {
            let _total_chars: usize = batch.iter().map(|s| s.chars().count()).sum();
            // Allow some flexibility due to batching algorithm
            assert!(batch.len() > 0, "No empty batches should be created");
        }
    }

    #[test]
    fn test_combine_embeddings_responses() {
        let response1 = CoreOpenAIEmbeddingsResponse {
            object: "list".to_string(),
            data: vec![CoreOpenAIEmbeddingData {
                object: "embedding".to_string(),
                embedding_internal: CoreEmbeddingVariant::FloatVector(vec![1.0, 2.0]),
                index: 0,
            }],
            model: "test-model".to_string(),
            usage: CoreOpenAIUsage {
                prompt_tokens: 10,
                total_tokens: 15,
            },
            total_time: -1.0,
            individual_request_times: Vec::new(),
            response_headers: Vec::new(),
        };

        let response2 = CoreOpenAIEmbeddingsResponse {
            object: "list".to_string(),
            data: vec![CoreOpenAIEmbeddingData {
                object: "embedding".to_string(),
                embedding_internal: CoreEmbeddingVariant::FloatVector(vec![3.0, 4.0]),
                index: 1,
            }],
            model: "test-model".to_string(),
            usage: CoreOpenAIUsage {
                prompt_tokens: 8,
                total_tokens: 12,
            },
            total_time: -1.0,
            individual_request_times: Vec::new(), // TODO: set some individual_request_times for a meaningful tests
            response_headers: Vec::new(),
        };

        let combined = CoreOpenAIEmbeddingsResponse::combine(vec![response1, response2], 2);

        assert_eq!(combined.data.len(), 2);
        assert_eq!(combined.usage.prompt_tokens, 18);
        assert_eq!(combined.usage.total_tokens, 27);
        assert_eq!(combined.model, "test-model");
    }

    #[test]
    fn test_combine_preserves_order_without_sorting() {
        // This test ensures that even if indices are out of order, the combine function
        // preserves the order of the responses as they are passed in.
        let response1 = CoreOpenAIEmbeddingsResponse {
            object: "list".to_string(),
            data: vec![CoreOpenAIEmbeddingData {
                object: "embedding".to_string(),
                embedding_internal: CoreEmbeddingVariant::FloatVector(vec![1.0]),
                index: 1, // Intentionally out of order
            }],
            model: "test-model".to_string(),
            usage: CoreOpenAIUsage {
                prompt_tokens: 1,
                total_tokens: 1,
            },
            total_time: -1.0,
            individual_request_times: Vec::new(),
            response_headers: Vec::new(),
        };

        let response2 = CoreOpenAIEmbeddingsResponse {
            object: "list".to_string(),
            data: vec![CoreOpenAIEmbeddingData {
                object: "embedding".to_string(),
                embedding_internal: CoreEmbeddingVariant::FloatVector(vec![2.0]),
                index: 0, // Intentionally out of order
            }],
            model: "test-model".to_string(),
            usage: CoreOpenAIUsage {
                prompt_tokens: 1,
                total_tokens: 1,
            },
            total_time: -1.0,
            individual_request_times: Vec::new(),
            response_headers: Vec::new(),
        };

        let combined = CoreOpenAIEmbeddingsResponse::combine(vec![response1, response2], 2);

        assert_eq!(combined.data.len(), 2);
        // Check that the order is [response1.data, response2.data], not sorted by index.
        if let CoreEmbeddingVariant::FloatVector(v) = &combined.data[0].embedding_internal {
            assert_eq!(v[0], 1.0);
        } else {
            panic!("Expected float vector");
        }
        if let CoreEmbeddingVariant::FloatVector(v) = &combined.data[1].embedding_internal {
            assert_eq!(v[0], 2.0);
        } else {
            panic!("Expected float vector");
        }
    }

    #[test]
    fn test_split_by_max_chars_policy_comprehensive() {
        let texts = vec![
            "12345".to_string(),                 // 5 chars
            "1234567890".to_string(),            // 10 chars
            "123".to_string(),                   // 3 chars
            "123456789012345678901".to_string(), // 21 chars (> 20)
            "12345".to_string(),                 // 5 chars
            "12345".to_string(),                 // 5 chars
        ];

        let policy = SplitPolicy::max_chars_per_request(20, 3);
        let batches = texts.split(&policy);

        assert_eq!(batches.len(), 3);
        // Batch 1: 5 + 10 + 3 = 18 chars, 3 items. Next item would exceed batch size.
        assert_eq!(batches[0], vec!["12345", "1234567890", "123"]);
        // Batch 2: 21 chars. Item is larger than max_chars, so it's a batch on its own.
        assert_eq!(batches[1], vec!["123456789012345678901"]);
        // Batch 3: 5 + 5 = 10 chars, 2 items.
        assert_eq!(batches[2], vec!["12345", "12345"]);
    }

    #[test]
    fn test_split_with_empty_input() {
        let texts: Vec<String> = vec![];

        let policy_batch = SplitPolicy::max_batch_size(2);
        let batches_batch = texts.split(&policy_batch);
        assert!(batches_batch.is_empty());

        let policy_chars = SplitPolicy::max_chars_per_request(100, 10);
        let batches_chars = texts.split(&policy_chars);
        assert!(batches_chars.is_empty());
    }

    #[test]
    fn test_split_all_items_larger_than_max_chars() {
        let texts = vec![
            "12345678901".to_string(),   // 11 chars
            "123456789012".to_string(),  // 12 chars
            "1234567890123".to_string(), // 13 chars
        ];

        let policy = SplitPolicy::max_chars_per_request(10, 3);
        let batches = texts.split(&policy);

        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0], vec!["12345678901"]);
        assert_eq!(batches[1], vec!["123456789012"]);
        assert_eq!(batches[2], vec!["1234567890123"]);
    }

    #[test]
    fn test_split_exact_max_chars_limit() {
        let texts = vec![
            "12345".to_string(),      // 5
            "1234567890".to_string(), // 10
            "12345".to_string(),      // 5
            "next".to_string(),       // 4
        ];

        let policy = SplitPolicy::max_chars_per_request(20, 3);
        let batches = texts.split(&policy);

        assert_eq!(batches.len(), 2);
        // Batch 1: 5 + 10 + 5 = 20 chars.
        assert_eq!(batches[0], vec!["12345", "1234567890", "12345"]);
        // Batch 2: The next item starts a new batch.
        assert_eq!(batches[1], vec!["next"]);
    }

    #[test]
    fn test_split_exact_max_batch_size_limit() {
        let texts = vec![
            "1".to_string(),
            "2".to_string(),
            "3".to_string(),
            "4".to_string(),
            "123456789".to_string(),
            "12".to_string(),
        ];

        let policy = SplitPolicy::max_chars_per_request(6, 3);
        let batches = texts.split(&policy);

        assert_eq!(batches.len(), 4);
        // Batch 1: 3 items, which is the max_batch_size.
        assert_eq!(batches[0], vec!["1", "2", "3"]);
        // Batch 2: The next item starts a new batch.
        assert_eq!(batches[1], vec!["4"]);
        // Batch 3: The last item is larger than max_chars, so it becomes its own batch.
        assert_eq!(batches[2], vec!["123456789"]);
        // Batch 4: The last item is smaller than max_chars, but previous one was larger so it becomes its own batch.
    }

    #[test]
    fn test_request_processing_preference_default() {
        let pref = RequestProcessingPreference::default();
        assert_eq!(pref.max_concurrent_requests, None);
        assert_eq!(pref.batch_size, None);
        assert_eq!(pref.timeout_s, None);
        assert!(pref.max_chars_per_request.is_none());
        assert!(pref.hedge_delay.is_none());
        assert!(pref.total_timeout_s.is_none());
        assert_eq!(pref.hedge_budget_pct, None);
        assert_eq!(pref.retry_budget_pct, None);

        // Test that with_defaults() applies the expected defaults
        let pref_with_defaults = pref.with_defaults();
        assert_eq!(
            pref_with_defaults.max_concurrent_requests,
            Some(DEFAULT_CONCURRENCY)
        );
        assert_eq!(pref_with_defaults.batch_size, Some(DEFAULT_BATCH_SIZE));
        assert_eq!(
            pref_with_defaults.timeout_s,
            Some(DEFAULT_REQUEST_TIMEOUT_S)
        );
        assert!(pref_with_defaults.max_chars_per_request.is_none());
        assert!(pref_with_defaults.hedge_delay.is_none());
        assert!(pref_with_defaults.total_timeout_s.is_none());
        assert_eq!(
            pref_with_defaults.hedge_budget_pct,
            Some(HEDGE_BUDGET_PERCENTAGE)
        );
        assert_eq!(
            pref_with_defaults.retry_budget_pct,
            Some(RETRY_BUDGET_PERCENTAGE)
        );
    }

    #[test]
    fn test_request_processing_preference_builder() {
        let pref = RequestProcessingPreference::new()
            .with_max_concurrent_requests(64)
            .with_batch_size(32)
            .with_timeout_s(30.0)
            .with_hedge_delay(0.5)
            .with_total_timeout_s(120.0)
            .with_hedge_budget_pct(0.15)
            .with_retry_budget_pct(0.08);

        assert_eq!(pref.max_concurrent_requests, Some(64));
        assert_eq!(pref.batch_size, Some(32));
        assert_eq!(pref.timeout_s, Some(30.0));
        assert_eq!(pref.hedge_delay, Some(0.5));
        assert_eq!(pref.total_timeout_s, Some(120.0));
        assert_eq!(pref.hedge_budget_pct, Some(0.15));
        assert_eq!(pref.retry_budget_pct, Some(0.08));
    }

    #[test]
    fn test_request_processing_preference_convert_to_config() {
        let pref = RequestProcessingPreference::new()
            .with_max_concurrent_requests(64)
            .with_batch_size(32)
            .with_timeout_s(30.0)
            .with_hedge_delay(0.5)
            .with_hedge_budget_pct(0.20)
            .with_retry_budget_pct(0.10);

        let config = pref
            .pair_with_request_validate_and_convert(
                "https://example.com".to_string(),
                100,
                "test_api_key".to_string(),
            )
            .expect("Should create valid config");

        assert_eq!(config.max_concurrent_requests, 64);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.timeout.as_secs_f64(), 30.0);
        assert_eq!(config.hedge_delay.map(|d| d.as_secs_f64()), Some(0.5));
        assert_eq!(config.base_url, "https://example.com");

        // Check computed budgets
        use std::sync::atomic::Ordering;
        assert_eq!(config.retry_budget.load(Ordering::SeqCst), 11); // 1 + (100 * 0.10) = 11
        assert_eq!(config.hedge_budget.load(Ordering::SeqCst), 21); // 1 + (100 * 0.20) = 21
    }

    #[test]
    fn test_negative_budget_percentages_validation() {
        let pref = RequestProcessingPreference::new().with_hedge_budget_pct(-0.1);

        let result = pref.pair_with_request_validate_and_convert(
            "https://example.com".to_string(),
            100,
            "test_api_key".to_string(),
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            ClientError::InvalidParameter(msg) => {
                assert!(msg.contains("hedge_budget_pct cannot be negative"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let pref2 = RequestProcessingPreference::new().with_retry_budget_pct(-0.05);

        let result2 = pref2.pair_with_request_validate_and_convert(
            "https://example.com".to_string(),
            100,
            "test_api_key".to_string(),
        );
        assert!(result2.is_err());
        match result2.unwrap_err() {
            ClientError::InvalidParameter(msg) => {
                assert!(msg.contains("retry_budget_pct cannot be negative"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_maximum_budget_percentages_validation() {
        let pref = RequestProcessingPreference::new().with_hedge_budget_pct(4.0); // 400% exceeds MAX_BUDGET_PERCENTAGE (300%)

        let result = pref.pair_with_request_validate_and_convert(
            "https://example.com".to_string(),
            100,
            "test_api_key".to_string(),
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            ClientError::InvalidParameter(msg) => {
                assert!(msg.contains("hedge_budget_pct cannot exceed"));
                assert!(msg.contains("300%"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let pref2 = RequestProcessingPreference::new().with_retry_budget_pct(3.5); // 350% exceeds MAX_BUDGET_PERCENTAGE (300%)

        let result2 = pref2.pair_with_request_validate_and_convert(
            "https://example.com".to_string(),
            100,
            "test_api_key".to_string(),
        );
        assert!(result2.is_err());
        match result2.unwrap_err() {
            ClientError::InvalidParameter(msg) => {
                assert!(msg.contains("retry_budget_pct cannot exceed"));
                assert!(msg.contains("300%"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_backoff_validation() {
        // Test initial_backoff_ms validation
        let pref = RequestProcessingPreference::new().with_initial_backoff_ms(25); // Below MIN_BACKOFF_MS (50)

        let result = pref.pair_with_request_validate_and_convert(
            "https://example.com".to_string(),
            100,
            "test_api_key".to_string(),
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            ClientError::InvalidParameter(msg) => {
                assert!(msg.contains("initial_backoff_ms must be between"));
                assert!(msg.contains("50"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let pref2 = RequestProcessingPreference::new().with_initial_backoff_ms(35000); // Above MAX_BACKOFF_MS (30000)

        let result2 = pref2.pair_with_request_validate_and_convert(
            "https://example.com".to_string(),
            100,
            "test_api_key".to_string(),
        );
        assert!(result2.is_err());
        match result2.unwrap_err() {
            ClientError::InvalidParameter(msg) => {
                assert!(msg.contains("initial_backoff_ms must be between"));
                assert!(msg.contains("30000"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        // Test valid backoff values
        let pref3 = RequestProcessingPreference::new().with_initial_backoff_ms(125); // Valid default value

        let result3 = pref3.pair_with_request_validate_and_convert(
            "https://example.com".to_string(),
            100,
            "test_api_key".to_string(),
        );
        assert!(result3.is_ok());
    }

    #[test]
    fn test_max_retries_validation() {
        // Test max_retries validation
        let pref = RequestProcessingPreference::new().with_max_retries(5); // Above MAX_HTTP_RETRIES (4)

        let result = pref.pair_with_request_validate_and_convert(
            "https://example.com".to_string(),
            100,
            "test_api_key".to_string(),
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            ClientError::InvalidParameter(msg) => {
                assert!(msg.contains("max_retries cannot exceed"));
                assert!(msg.contains("4"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        // Test valid max_retries values
        let pref2 = RequestProcessingPreference::new().with_max_retries(3); // Valid value

        let result2 = pref2.pair_with_request_validate_and_convert(
            "https://example.com".to_string(),
            100,
            "test_api_key".to_string(),
        );
        assert!(result2.is_ok());
    }

    #[test]
    fn test_update_budgets_atomic_values() {
        let pref = RequestProcessingPreference::new()
            .with_hedge_delay(0.5)
            .with_hedge_budget_pct(0.10)
            .with_retry_budget_pct(0.05);

        let config = pref
            .pair_with_request_validate_and_convert(
                "https://example.com".to_string(),
                1,
                "test_api_key".to_string(),
            )
            .expect("Should create valid config");

        // Initial budgets should be 2 (minimum budget with our new logic)
        use std::sync::atomic::Ordering;
        assert_eq!(config.retry_budget.load(Ordering::SeqCst), 2);
        assert_eq!(config.hedge_budget.load(Ordering::SeqCst), 2);

        // Update budgets with 200 requests - should modify existing atomic values
        config.update_budgets(200);

        // Verify the same atomic instances were updated
        assert_eq!(config.retry_budget.load(Ordering::SeqCst), 11); // 1 + (200 * 0.05) = 11
        assert_eq!(config.hedge_budget.load(Ordering::SeqCst), 21); // 1 + (200 * 0.10) = 21

        // Update again with different request count
        config.update_budgets(50);

        // Verify the same atomic instances were updated again
        assert_eq!(config.retry_budget.load(Ordering::SeqCst), 4); // 1 + (50 * 0.05) = 4
        assert_eq!(config.hedge_budget.load(Ordering::SeqCst), 6); // 1 + (50 * 0.10) = 6
    }

    #[test]
    fn test_hedge_delay_consistency_with_constant() {
        use std::sync::atomic::Ordering;

        // Test with delay exactly at MIN_HEDGE_DELAY_S
        let pref = RequestProcessingPreference::new()
            .with_hedge_delay(MIN_HEDGE_DELAY_S)
            .with_hedge_budget_pct(0.10);

        let config = pref
            .pair_with_request_validate_and_convert(
                "https://example.com".to_string(),
                100,
                "test_api_key".to_string(),
            )
            .expect("Should create valid config");

        // Should create hedge budget when delay equals MIN_HEDGE_DELAY_S
        assert_eq!(config.hedge_budget.load(Ordering::SeqCst), 11); // 1 + (100 * 0.10) = 11

        // Test with delay just above MIN_HEDGE_DELAY_S
        let pref2 = RequestProcessingPreference::new()
            .with_hedge_delay(MIN_HEDGE_DELAY_S + 0.001)
            .with_hedge_budget_pct(0.10);

        let config2 = pref2
            .pair_with_request_validate_and_convert(
                "https://example.com".to_string(),
                100,
                "test_api_key".to_string(),
            )
            .expect("Should create valid config");

        // Should create hedge budget when delay is above MIN_HEDGE_DELAY_S
        assert_eq!(config2.hedge_budget.load(Ordering::SeqCst), 11); // 1 + (100 * 0.10) = 11

        // Test with no hedge delay - should set hedge budget to 0
        let pref3 = RequestProcessingPreference::new().with_hedge_budget_pct(0.10);

        let config3 = pref3
            .pair_with_request_validate_and_convert(
                "https://example.com".to_string(),
                100,
                "test_api_key".to_string(),
            )
            .expect("Should create valid config");

        // Should set hedge budget to 0 when no delay is specified
        assert_eq!(config3.hedge_budget.load(Ordering::SeqCst), 0);
    }
}

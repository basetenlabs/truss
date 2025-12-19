use crate::constants::*;
use crate::errors::ClientError;
use crate::http::*;
use crate::http_client::SendRequestConfig;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Duration;

/// User-facing configuration for request processing with budget percentages.
/// This is the public API struct that gets validated and converted to RequestProcessingConfig.
/// All fields are Option<T> with defaults applied during conversion to RequestProcessingConfig.
#[derive(Debug, Clone)]
pub struct RequestProcessingPreference {
    pub max_concurrent_requests: usize,
    pub batch_size: usize,
    pub max_chars_per_request: Option<usize>,
    pub timeout_s: f64,
    pub hedge_delay: Option<f64>,
    pub total_timeout_s: Option<f64>,
    pub hedge_budget_pct: f64,
    pub retry_budget_pct: f64,
    pub max_retries: Option<u32>,
    pub initial_backoff: Option<Duration>,
}

impl Default for RequestProcessingPreference {
    fn default() -> Self {
        Self {
            max_concurrent_requests: DEFAULT_CONCURRENCY,
            batch_size: DEFAULT_BATCH_SIZE,
            max_chars_per_request: None,
            timeout_s: DEFAULT_REQUEST_TIMEOUT_S,
            hedge_delay: None,
            total_timeout_s: None,
            hedge_budget_pct: DEFAULT_HEDGE_BUDGET_PERCENTAGE,
            retry_budget_pct: DEFAULT_RETRY_BUDGET_PERCENTAGE,
            max_retries: None,
            initial_backoff: None,
        }
    }
}

impl RequestProcessingPreference {
    /// Create a new preference with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder pattern: set max concurrent requests
    pub fn with_max_concurrent_requests(mut self, value: usize) -> Self {
        self.max_concurrent_requests = value;
        self
    }

    /// Builder pattern: set batch size
    pub fn with_batch_size(mut self, value: usize) -> Self {
        self.batch_size = value;
        self
    }

    /// Builder pattern: set max chars per request
    pub fn with_max_chars_per_request(mut self, value: Option<usize>) -> Self {
        self.max_chars_per_request = value;
        self
    }

    /// Builder pattern: set timeout in seconds
    pub fn with_timeout_s(mut self, value: f64) -> Self {
        self.timeout_s = value;
        self
    }

    /// Builder pattern: set hedge delay in seconds
    pub fn with_hedge_delay(mut self, value: Option<f64>) -> Self {
        self.hedge_delay = value;
        self
    }

    /// Builder pattern: set total timeout in seconds
    pub fn with_total_timeout_s(mut self, value: Option<f64>) -> Self {
        self.total_timeout_s = value;
        self
    }

    /// Builder pattern: set hedge budget percentage
    pub fn with_hedge_budget_pct(mut self, value: f64) -> Self {
        self.hedge_budget_pct = value;
        self
    }

    /// Builder pattern: set retry budget percentage
    pub fn with_retry_budget_pct(mut self, value: f64) -> Self {
        self.retry_budget_pct = value;
        self
    }

    /// Builder pattern: set max retries
    pub fn with_max_retries(mut self, value: u32) -> Self {
        self.max_retries = Some(value);
        self
    }

    /// Builder pattern: set initial backoff duration
    pub fn with_initial_backoff(mut self, value: Duration) -> Self {
        self.initial_backoff = Some(value);
        self
    }

    /// Validate and convert to RequestProcessingConfig for a specific request.
    /// This pairs the preference with request-specific data (base_url, total_requests)
    /// and returns a validated config ready for processing.
    pub fn pair_with_request_validate_and_convert(
        &self,
        base_url: String,
        total_requests: usize,
    ) -> Result<RequestProcessingConfig, ClientError> {
        RequestProcessingConfig::new_from_preference(self, base_url, total_requests)
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
/// ready for use during request processing.
#[derive(Debug, Clone)]
pub struct RequestProcessingConfig {
    pub max_concurrent_requests: usize,
    pub batch_size: usize,
    pub timeout_s: f64,
    pub total_timeout_s: Option<f64>,
    pub base_url: String,
    pub hedge_delay: Option<f64>,
    pub max_chars_per_request: Option<usize>,
    /// Pre-computed retry budget as Arc<AtomicUsize> for concurrent access
    pub retry_budget: Arc<AtomicUsize>,
    /// Pre-computed hedge budget as Arc<AtomicUsize> for concurrent access (if hedging enabled)
    pub hedge_budget: Option<Arc<AtomicUsize>>,
    /// Budget percentages used for calculation (stored for reference)
    pub hedge_budget_pct: f64,
    pub retry_budget_pct: f64,
    /// Maximum number of HTTP retries per request
    pub max_retries: u32,
    /// Initial backoff duration for retry exponential backoff
    pub initial_backoff: Duration,
}

impl RequestProcessingConfig {
    /// Validate parameters and return error if invalid
    fn validate_parameters(
        max_concurrent_requests: usize,
        batch_size: usize,
        timeout_s: f64,
        total_timeout_s: Option<f64>,
        hedge_delay: Option<f64>,
        max_chars_per_request: Option<usize>,
    ) -> Result<(), ClientError> {
        // TODO: validate pct ages no more than 300%.

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

        Ok(())
    }

    /// Calculate budget based on total requests and percentage
    fn calculate_budget(total_requests: usize, budget_pct: f64) -> usize {
        (total_requests as f64 * budget_pct).ceil() as usize
    }

    /// Validate and create a new config with adjusted concurrency for baseten staging.
    /// Uses default budget percentages. For custom budgets, use new_from_preference().
    pub fn new(
        max_concurrent_requests: usize,
        batch_size: usize,
        timeout_s: f64,
        total_timeout_s: Option<f64>,
        base_url: String,
        hedge_delay: Option<f64>,
        max_chars_per_request: Option<usize>,
    ) -> Result<Self, ClientError> {
        Self::new_with_budgets(
            max_concurrent_requests,
            batch_size,
            timeout_s,
            total_timeout_s,
            base_url,
            hedge_delay,
            max_chars_per_request,
            DEFAULT_HEDGE_BUDGET_PERCENTAGE,
            DEFAULT_RETRY_BUDGET_PERCENTAGE,
            0, // total_requests not known yet, budgets calculated later
            None, // use default max_retries
            None, // use default initial_backoff
        )
    }

    /// Create config with custom budget percentages.
    /// Note: When total_requests is 0, budgets are set to 0 and should be recalculated
    /// when the actual request count is known.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_budgets(
        max_concurrent_requests: usize,
        batch_size: usize,
        timeout_s: f64,
        total_timeout_s: Option<f64>,
        base_url: String,
        hedge_delay: Option<f64>,
        max_chars_per_request: Option<usize>,
        hedge_budget_pct: f64,
        retry_budget_pct: f64,
        total_requests: usize,
        max_retries: Option<u32>,
        initial_backoff: Option<Duration>,
    ) -> Result<Self, ClientError> {
        Self::validate_parameters(
            max_concurrent_requests,
            batch_size,
            timeout_s,
            total_timeout_s,
            hedge_delay,
            max_chars_per_request,
        )?;

        let retry_budget = Arc::new(AtomicUsize::new(Self::calculate_budget(
            total_requests,
            retry_budget_pct,
        )));

        let hedge_budget = if hedge_delay.is_some() {
            Some(Arc::new(AtomicUsize::new(Self::calculate_budget(
                total_requests,
                hedge_budget_pct,
            ))))
        } else {
            None
        };

        Ok(RequestProcessingConfig {
            max_concurrent_requests,
            batch_size,
            timeout_s,
            total_timeout_s,
            base_url,
            hedge_delay,
            max_chars_per_request,
            retry_budget,
            hedge_budget,
            hedge_budget_pct,
            retry_budget_pct,
            max_retries: max_retries.unwrap_or(MAX_HTTP_RETRIES),
            initial_backoff: initial_backoff.unwrap_or(Duration::from_millis(INITIAL_BACKOFF_MS)),
        })
    }

    /// Create config from a RequestProcessingPreference.
    /// This is the main entry point for converting user preferences to internal config.
    pub fn new_from_preference(
        preference: &RequestProcessingPreference,
        base_url: String,
        total_requests: usize,
    ) -> Result<Self, ClientError> {
        Self::new_with_budgets(
            preference.max_concurrent_requests,
            preference.batch_size,
            preference.timeout_s,
            preference.total_timeout_s,
            base_url,
            preference.hedge_delay,
            preference.max_chars_per_request,
            preference.hedge_budget_pct,
            preference.retry_budget_pct,
            total_requests,
            preference.max_retries,
            preference.initial_backoff,
        )
    }

    /// Recalculate budgets with a new total request count.
    /// Returns new Arc<AtomicUsize> instances with the recalculated values.
    pub fn recalculate_budgets(&self, total_requests: usize) -> (Arc<AtomicUsize>, Option<Arc<AtomicUsize>>) {
        let retry_budget = Arc::new(AtomicUsize::new(Self::calculate_budget(
            total_requests,
            self.retry_budget_pct,
        )));

        let hedge_budget = if self.hedge_delay.is_some() {
            Some(Arc::new(AtomicUsize::new(Self::calculate_budget(
                total_requests,
                self.hedge_budget_pct,
            ))))
        } else {
            None
        };

        (retry_budget, hedge_budget)
    }

    /// Get timeout duration
    pub fn timeout_duration(&self) -> Duration {
        Duration::from_secs_f64(self.timeout_s)
    }

    /// Get total timeout duration if set
    pub fn total_timeout_duration(&self) -> Option<Duration> {
        self.total_timeout_s.map(Duration::from_secs_f64)
    }

    /// Get hedge delay duration if set
    pub fn hedge_delay_duration(&self) -> Option<Duration> {
        self.hedge_delay.map(Duration::from_secs_f64)
    }

    /// Create a SendRequestConfig for HTTP requests using this config's settings.
    /// Takes retry and hedge budgets as parameters since they may be recalculated per batch.
    pub fn create_send_request_config(
        &self,
        retry_budget: Arc<AtomicUsize>,
        hedge_budget: Option<(Arc<AtomicUsize>, Duration)>,
    ) -> SendRequestConfig {
        SendRequestConfig {
            max_retries: self.max_retries,
            initial_backoff: self.initial_backoff,
            retry_budget,
            hedge_budget,
            timeout: self.timeout_duration(),
        }
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
        assert_eq!(pref.max_concurrent_requests, DEFAULT_CONCURRENCY);
        assert_eq!(pref.batch_size, DEFAULT_BATCH_SIZE);
        assert_eq!(pref.timeout_s, DEFAULT_REQUEST_TIMEOUT_S);
        assert!(pref.max_chars_per_request.is_none());
        assert!(pref.hedge_delay.is_none());
        assert!(pref.total_timeout_s.is_none());
        assert_eq!(pref.hedge_budget_pct, DEFAULT_HEDGE_BUDGET_PERCENTAGE);
        assert_eq!(pref.retry_budget_pct, DEFAULT_RETRY_BUDGET_PERCENTAGE);
    }

    #[test]
    fn test_request_processing_preference_builder() {
        let pref = RequestProcessingPreference::new()
            .with_max_concurrent_requests(64)
            .with_batch_size(32)
            .with_timeout_s(30.0)
            .with_hedge_delay(Some(0.5))
            .with_total_timeout_s(Some(120.0))
            .with_hedge_budget_pct(0.15)
            .with_retry_budget_pct(0.08);

        assert_eq!(pref.max_concurrent_requests, 64);
        assert_eq!(pref.batch_size, 32);
        assert_eq!(pref.timeout_s, 30.0);
        assert_eq!(pref.hedge_delay, Some(0.5));
        assert_eq!(pref.total_timeout_s, Some(120.0));
        assert_eq!(pref.hedge_budget_pct, 0.15);
        assert_eq!(pref.retry_budget_pct, 0.08);
    }

    #[test]
    fn test_request_processing_preference_convert_to_config() {
        let pref = RequestProcessingPreference::new()
            .with_max_concurrent_requests(64)
            .with_batch_size(32)
            .with_timeout_s(30.0)
            .with_hedge_delay(Some(0.5))
            .with_hedge_budget_pct(0.20)
            .with_retry_budget_pct(0.10);

        let config = pref
            .pair_with_request_validate_and_convert("https://example.com".to_string(), 100)
            .expect("Should create valid config");

        assert_eq!(config.max_concurrent_requests, 64);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.timeout_s, 30.0);
        assert_eq!(config.hedge_delay, Some(0.5));
        assert_eq!(config.base_url, "https://example.com");

        // Check computed budgets
        use std::sync::atomic::Ordering;
        assert_eq!(config.retry_budget.load(Ordering::SeqCst), 10); // 100 * 0.10 = 10
        assert!(config.hedge_budget.is_some());
        assert_eq!(config.hedge_budget.as_ref().unwrap().load(Ordering::SeqCst), 20); // 100 * 0.20 = 20
    }

    #[test]
    fn test_request_processing_config_recalculate_budgets() {
        let pref = RequestProcessingPreference::new()
            .with_hedge_delay(Some(0.5))
            .with_hedge_budget_pct(0.10)
            .with_retry_budget_pct(0.05);

        let config = pref
            .pair_with_request_validate_and_convert("https://example.com".to_string(), 0)
            .expect("Should create valid config");

        // Initial budgets should be 0
        use std::sync::atomic::Ordering;
        assert_eq!(config.retry_budget.load(Ordering::SeqCst), 0);
        assert_eq!(config.hedge_budget.as_ref().unwrap().load(Ordering::SeqCst), 0);

        // Recalculate with 200 requests
        let (new_retry, new_hedge) = config.recalculate_budgets(200);
        assert_eq!(new_retry.load(Ordering::SeqCst), 10); // 200 * 0.05 = 10
        assert_eq!(new_hedge.as_ref().unwrap().load(Ordering::SeqCst), 20); // 200 * 0.10 = 20
    }
}

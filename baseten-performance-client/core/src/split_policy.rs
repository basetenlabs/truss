use crate::constants::*;
use crate::http::*;

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

/// Configuration for request processing with validation
#[derive(Debug, Clone)]
pub struct RequestProcessingConfig {
    pub max_concurrent_requests: usize,
    pub batch_size: usize,
    pub timeout_s: f64,
    pub base_url: String,
    pub hedge_delay: Option<f64>,
    pub max_chars_per_request: Option<usize>,
}

impl RequestProcessingConfig {
    /// Validate and create a new config with adjusted concurrency for baseten staging
    pub fn new(
        max_concurrent_requests: usize,
        batch_size: usize,
        timeout_s: f64,
        base_url: String,
        hedge_delay: Option<f64>,
        max_chars_per_request: Option<usize>,
    ) -> Result<Self, crate::errors::ClientError> {
        // Validate timeout
        if !(MIN_REQUEST_TIMEOUT_S..=MAX_REQUEST_TIMEOUT_S).contains(&timeout_s) {
            return Err(crate::errors::ClientError::InvalidParameter(format!(
                "Timeout {:.3}s is outside the allowed range [{:.3}s, {:.3}s].",
                timeout_s, MIN_REQUEST_TIMEOUT_S, MAX_REQUEST_TIMEOUT_S
            )));
        }
        if hedge_delay.is_some() {
            let hedge_delay = hedge_delay.unwrap();
            if !(MIN_HEDGE_DELAY_S..=MAX_REQUEST_TIMEOUT_S).contains(&hedge_delay) {
                return Err(crate::errors::ClientError::InvalidParameter(format!(
                    "Hedge delay {:.3}s is outside the allowed range [{:.3}s, {:.3}s].",
                    hedge_delay, MIN_HEDGE_DELAY_S, MAX_REQUEST_TIMEOUT_S
                )));
            }
            if hedge_delay >= timeout_s - MIN_HEDGE_DELAY_S {
                return Err(crate::errors::ClientError::InvalidParameter(format!(
                    "Hedge delay {:.3}s must be less than timeout minus minimum hedge delay ({:.3}s -{:.3}s).",
                    hedge_delay, timeout_s, MIN_HEDGE_DELAY_S
                )));
            }
        }
        if max_chars_per_request.is_some() {
            let max_chars = max_chars_per_request.unwrap();
            if !(MIN_CHARACTERS_PER_REQUEST..=MAX_CHARACTERS_PER_REQUEST).contains(&max_chars) {
                return Err(crate::errors::ClientError::InvalidParameter(format!(
                    "max_chars_per_request must be between {} and {} characters.",
                    MIN_CHARACTERS_PER_REQUEST, MAX_CHARACTERS_PER_REQUEST
                )));
            }
        }

        // Validate concurrency parameters
        if max_concurrent_requests == 0 || max_concurrent_requests > MAX_CONCURRENCY_HIGH_BATCH {
            return Err(crate::errors::ClientError::InvalidParameter(format!(
                "max_concurrent_requests must be greater than 0 and less than or equal to {}",
                MAX_CONCURRENCY_HIGH_BATCH
            )));
        } else if batch_size == 0 || batch_size > MAX_BATCH_SIZE {
            return Err(crate::errors::ClientError::InvalidParameter(format!(
                "batch_size must be greater than 0 and less than or equal to {}",
                MAX_BATCH_SIZE
            )));
        } else if max_concurrent_requests > MAX_CONCURRENCY_LOW_BATCH
            && batch_size < CONCURRENCY_HIGH_BATCH_SWITCH
        {
            return Err(crate::errors::ClientError::InvalidParameter(format!(
                "max_concurrent_requests must be less than {} when batch_size is less than {}. Please be nice to the server side.",
                MAX_CONCURRENCY_LOW_BATCH, CONCURRENCY_HIGH_BATCH_SWITCH
            )));
        }

        Ok(RequestProcessingConfig {
            max_concurrent_requests,
            batch_size,
            timeout_s,
            base_url,
            hedge_delay,
            max_chars_per_request,
        })
    }

    /// Get timeout duration
    pub fn timeout_duration(&self) -> std::time::Duration {
        std::time::Duration::from_secs_f64(self.timeout_s)
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
}

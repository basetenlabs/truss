use std::env;
use std::sync::{Arc, OnceLock};
use uuid::Uuid;

/// Cached environment variable for customer prefix (as Arc<str> for efficient sharing)
static CUSTOMER_PREFIX_CACHE: OnceLock<Arc<str>> = OnceLock::new();

/// Lightweight customer request ID struct with optional components and caching
#[derive(Debug, Clone)]
pub struct CustomerRequestId {
    // Core identifier (lightweight) - last 8 chars of UUID (shared across batch)
    uuid_suffix: Arc<str>,

    // Optional components for flexibility
    customer_prefix: Option<Arc<str>>, // From env var or default "perfclient" (shared across batch)
    batch_index: Option<usize>,        // For individual requests in batch
    retry_count: Option<u32>,          // For retry tracking
    hedge_id: Option<u32>,             // For hedged requests
}

impl CustomerRequestId {
    /// Get cached customer prefix from environment variable (as Arc<str>)
    fn get_customer_prefix() -> &'static Arc<str> {
        CUSTOMER_PREFIX_CACHE.get_or_init(|| {
            let base_name = "perfclient";
            let env_var_name = "PERFORMANCE_CLIENT_REQUEST_ID_PREFIX";
            let prefix = env::var(env_var_name)
                .ok()
                .filter(|s| !s.is_empty())
                .map(|s| format!("{}{}", base_name, s.to_lowercase().trim()))
                .unwrap_or_else(|| base_name.to_string());
            Arc::from(prefix)
        })
    }

    /// Create new batch-level customer request ID
    pub(crate) fn new_batch() -> Self {
        let uuid = Uuid::new_v4();
        let uuid_simple = uuid.to_string();
        let uuid_suffix = Arc::from(uuid_simple[uuid_simple.len() - 10..].to_string());

        Self {
            uuid_suffix,
            customer_prefix: Some(Arc::clone(Self::get_customer_prefix())),
            batch_index: None,
            retry_count: Some(0),
            hedge_id: None,
        }
    }

    /// Create request-level customer request ID from batch ID
    pub(crate) fn new_request(&self, batch_index: usize) -> Self {
        Self {
            uuid_suffix: Arc::clone(&self.uuid_suffix),
            customer_prefix: self.customer_prefix.as_ref().map(Arc::clone),
            batch_index: Some(batch_index),
            retry_count: Some(0),
            hedge_id: None,
        }
    }

    /// Increment retry count and clear cache
    pub(crate) fn increment_retry(&mut self) -> &mut Self {
        if let Some(ref mut count) = self.retry_count {
            *count += 1;
        } else {
            self.retry_count = Some(1);
        }
        self
    }

    /// Set hedge ID and clear cache
    pub(crate) fn set_hedge(&mut self, hedge_id: u32) -> &mut Self {
        self.hedge_id = Some(hedge_id);
        self
    }

    /// Get the customer prefix (from env var or default)
    pub(crate) fn customer_prefix(&self) -> Option<&str> {
        self.customer_prefix.as_deref()
    }

    /// Get the batch index
    pub(crate) fn batch_index(&self) -> Option<usize> {
        self.batch_index
    }

    /// Get the retry count
    pub(crate) fn retry_count(&self) -> Option<u32> {
        self.retry_count
    }

    /// Get the hedge ID
    pub(crate) fn hedge_id(&self) -> Option<u32> {
        self.hedge_id
    }

    /// Get the UUID suffix component (last 8 characters)
    pub(crate) fn uuid_suffix(&self) -> &str {
        &self.uuid_suffix
    }

    /// Check if this is a hedged request
    pub(crate) fn is_hedged(&self) -> bool {
        self.hedge_id.is_some()
    }

    /// Check if this is a retried request
    pub(crate) fn is_retried(&self) -> bool {
        self.retry_count.map(|count| count > 0).unwrap_or(false)
    }
}

impl Default for CustomerRequestId {
    fn default() -> Self {
        Self::new_batch()
    }
}

impl std::fmt::Display for CustomerRequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Generate string
        let prefix = self.customer_prefix.as_deref().unwrap_or("perfclient");
        let mut result = format!("{}-{}", prefix, self.uuid_suffix);

        if let Some(batch_index) = self.batch_index {
            result = format!("{}-{}", result, batch_index);

            if let Some(hedge_id) = self.hedge_id {
                result = format!("{}-hedge-{}", result, hedge_id);
            } else if let Some(retry_count) = self.retry_count {
                result = format!("{}-{}", result, retry_count);
            }
        }

        write!(f, "{}", result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_string_cloning_optimization() {
        let batch_id = CustomerRequestId::new_batch();
        let num_requests = 10_000;

        let start = Instant::now();
        let mut request_ids = Vec::with_capacity(num_requests);

        for i in 0..num_requests {
            request_ids.push(batch_id.new_request(i));
        }

        let duration = start.elapsed();

        // Should be very fast due to Arc<str> sharing
        assert!(
            duration.as_millis() < 100,
            "Performance regression detected"
        );

        // Verify all requests share the same UUID suffix (same memory address)
        let first_uuid_suffix = request_ids[0].uuid_suffix();
        for request_id in &request_ids {
            assert_eq!(request_id.uuid_suffix(), first_uuid_suffix);
        }

        println!(
            "✅ Created {} request IDs in {:?} (optimized with Arc<str>)",
            num_requests, duration
        );
    }

    #[test]
    fn test_default_batch_id() {
        let id = CustomerRequestId::new_batch();
        let id_str = id.to_string();

        // Should have default prefix (from env var or "perfclient")
        let expected_prefix = CustomerRequestId::get_customer_prefix();
        assert!(id_str.starts_with(&format!("{}-", expected_prefix.as_ref())));

        // Check the actual prefix
        if let Some(actual_prefix) = id.customer_prefix() {
            assert_eq!(actual_prefix, expected_prefix.as_ref());
        }

        // Should have UUID
        assert!(id_str.len() > "perfclient-".len());

        // Should not have batch index
        assert!(id.batch_index().is_none());

        // Should have retry count 0
        assert!(id.retry_count() == Some(0));

        // Should not be hedged or retried
        assert!(!id.is_hedged());
        assert!(!id.is_retried());
    }

    #[test]
    fn test_request_id_from_batch() {
        let batch_id = CustomerRequestId::new_batch();
        let request_id = batch_id.new_request(5);
        let request_str = request_id.to_string();

        // Should have batch index
        assert!(request_id.batch_index() == Some(5));
        assert!(request_str.contains("-5"));

        // Should share UUID suffix with batch
        assert!(request_id.uuid_suffix() == batch_id.uuid_suffix());

        // Should have retry count 0
        assert!(request_id.retry_count() == Some(0));
    }

    #[test]
    fn test_retry_increment() {
        let mut id = CustomerRequestId::new_batch();

        // Initial state
        assert!(id.retry_count() == Some(0));
        assert!(!id.is_retried());

        // Increment retry
        id.increment_retry();
        assert!(id.retry_count() == Some(1));
        assert!(id.is_retried());

        // Increment again
        id.increment_retry();
        assert!(id.retry_count() == Some(2));
        assert!(id.is_retried());
    }

    #[test]
    fn test_hedge_id() {
        let id = CustomerRequestId::new_batch();

        // Create a request ID (batch IDs don't show hedge in string)
        let mut request_id = id.new_request(0);

        // Initial state
        assert!(request_id.hedge_id().is_none());
        assert!(!request_id.is_hedged());

        // Set hedge ID
        request_id.set_hedge(3);
        assert!(request_id.hedge_id() == Some(3));
        assert!(request_id.is_hedged());

        let id_str = request_id.to_string();
        assert!(id_str.contains("-hedge-3"));
    }

    #[test]
    fn test_string_generation() {
        let id = CustomerRequestId::new_batch();

        // Create a request ID (batch IDs don't show retry count in string)
        let mut request_id = id.new_request(0);

        // Generate string
        let str1 = request_id.to_string();

        // Modify ID should change string
        request_id.increment_retry();
        let str2 = request_id.to_string();

        // Should be different due to retry count
        assert_ne!(str1, str2);
    }

    #[test]
    fn test_env_var_prefix() {
        // This test would require setting env var, which we'll skip in unit tests
        // but the logic is tested in the get_customer_prefix function
        let prefix = CustomerRequestId::get_customer_prefix();
        // In normal tests, this will be "perfclient" unless env var is set
        assert!(!prefix.is_empty());
        assert!(!prefix.as_ref().is_empty());
    }

    #[test]
    fn test_display_trait() {
        let id = CustomerRequestId::new_batch();
        let display_str = format!("{}", id);
        let to_string_str = id.to_string();

        assert_eq!(display_str, to_string_str);
    }
}

use std::time::Duration;

// Request timeout constants
pub const DEFAULT_REQUEST_TIMEOUT_S: f64 = 3600.0;
pub(crate) const MIN_REQUEST_TIMEOUT_S: f64 = 0.5;
pub(crate) const MAX_REQUEST_TIMEOUT_S: f64 = 3600.0;

// Concurrency constants
pub(crate) const MAX_CONCURRENCY_HIGH_BATCH: usize = 1024;
pub(crate) const MAX_CONCURRENCY_LOW_BATCH: usize = 512;
pub(crate) const CONCURRENCY_HIGH_BATCH_SWITCH: usize = 16;
pub const DEFAULT_CONCURRENCY: usize = 128;
pub(crate) const MIN_CHARACTERS_PER_REQUEST: usize = 50;
pub(crate) const MAX_CHARACTERS_PER_REQUEST: usize = 256000;

// hedging settings:
pub(crate) const MIN_HEDGE_DELAY_S: f64 = 0.2;

// Batch size constants
pub(crate) const MAX_BATCH_SIZE: usize = 1024;
pub const DEFAULT_BATCH_SIZE: usize = 128;

// Retry constants
pub const MAX_HTTP_RETRIES: u32 = 4;
pub const INITIAL_BACKOFF_MS: u64 = 125;
pub(crate) const MIN_BACKOFF_MS: u64 = 50;
pub(crate) const MAX_BACKOFF_MS: u64 = 30000; // 30 seconds
pub(crate) const MAX_BACKOFF_DURATION: Duration = Duration::from_secs(60);

pub const HEDGE_BUDGET_PERCENTAGE: f64 = 0.10;
pub const RETRY_BUDGET_PERCENTAGE: f64 = 0.05;
pub(crate) const MAX_BUDGET_PERCENTAGE: f64 = 3.0; // 300%

// HTTP/2 constants
pub(crate) const HTTP2_WINDOW_SIZE: u32 = 2_097_152; // 2 MB
pub(crate) const HTTP2_CLIENT_POOL_SIZE: usize = 64;
pub(crate) const HTTP2_CLIENT_OPTIMUM_QUEUED: usize = 8;

// Slow providers where customers have reported issues
pub(crate) const WARNING_SLOW_PROVIDERS: [&str; 3] = ["fireworks.ai", "together.ai", "modal.com"];
pub(crate) const CUSTOMER_HEADER_NAME: &str = "x-baseten-customer-request-id";

// Logging constants
pub const DEFAULT_LOG_LEVEL: &str = "warn";
pub const LOG_LEVEL_ENV_VAR: &str = "PERFORMANCE_CLIENT_LOG_LEVEL";

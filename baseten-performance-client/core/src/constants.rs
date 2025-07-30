use std::time::Duration;

// Request timeout constants
pub const DEFAULT_REQUEST_TIMEOUT_S: f64 = 3600.0;
pub const MIN_REQUEST_TIMEOUT_S: f64 = 1.0;
pub const MAX_REQUEST_TIMEOUT_S: f64 = 3600.0;

// Concurrency constants
pub const MAX_CONCURRENCY_HIGH_BATCH: usize = 1024;
pub const MAX_CONCURRENCY_LOW_BATCH: usize = 512;
pub const CONCURRENCY_HIGH_BATCH_SWITCH: usize = 16;
pub const DEFAULT_CONCURRENCY: usize = 32;
pub const MIN_CHARACTERS_PER_REQUEST: usize = 50;
pub const MAX_CHARACTERS_PER_REQUEST: usize = 256000;

// hedging settings:
pub const MIN_HEDGE_DELAY_S: f64 = 0.2;
pub const HEDGE_BUDGET_PERCENTAGE: f64 = 0.10;

// Batch size constants
pub const MAX_BATCH_SIZE: usize = 1024;
pub const DEFAULT_BATCH_SIZE: usize = 128;

// Retry constants
pub const MAX_HTTP_RETRIES: u32 = 4;
pub const INITIAL_BACKOFF_MS: u64 = 125;
pub const MAX_BACKOFF_DURATION: Duration = Duration::from_secs(60);
pub const RETRY_TIMEOUT_BUDGET_PERCENTAGE: f64 = 0.05;

// HTTP/2 constants
pub const HTTP2_WINDOW_SIZE: u32 = 2_097_152; // 2 MB
pub const HTTP2_CLIENT_POOL_SIZE: usize = 64;
pub const HTTP2_CLIENT_MAX_QUEUED: usize = 8;

// Error messages
pub const CANCELLATION_ERROR_MESSAGE_DETAIL: &str = "Operation cancelled due to a previous error";
pub const CTRL_C_ERROR_MESSAGE_DETAIL: &str = "Operation cancelled by Ctrl+C";

// Slow providers where customers have reported issues
pub const WARNING_SLOW_PROVIDERS: [&str; 3] = ["fireworks.ai", "together.ai", "modal.com"];

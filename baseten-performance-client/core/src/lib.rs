pub mod cancellation;
pub mod client;
pub mod constants;
pub mod customer_request_id;
pub mod errors;
pub mod http;
pub mod http_client;
pub mod split_policy;
pub mod utils;

// JoinSetGuard is internal only - not reexported
pub use cancellation::CancellationToken;
pub use client::{HttpClientWrapper, PerformanceClientCore};
pub use constants::*;
pub use errors::ClientError;
pub use http::*;
// http_client is internal only - not reexported
pub use split_policy::RequestProcessingPreference;
pub use utils::*;

/// Initialize tracing with default WARN level
/// This is called automatically when the library is loaded
#[ctor::ctor]
fn init_tracing() {
    // Check for PERFORMANCE_CLIENT_LOG_LEVEL first (highest priority)
    if let Ok(custom_level) = std::env::var(crate::constants::LOG_LEVEL_ENV_VAR) {
        std::env::set_var("RUST_LOG", custom_level);
    }
    // Only set default if RUST_LOG is not already set
    else if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", crate::constants::DEFAULT_LOG_LEVEL);
    }

    // Initialize subscriber only once, and only if not already initialized
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        // Try to initialize tracing, but don't panic if it's already initialized
        if let Err(_) = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init()
        {
            // Tracing is already initialized, which is fine
        }
    });
}

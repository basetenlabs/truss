pub mod cancellation;
pub mod client;
pub mod constants;
pub mod customer_request_id;
pub mod errors;
pub mod http;
pub mod http_client;
pub mod split_policy;
pub mod utils;

pub use cancellation::JoinSetGuard;
pub use client::{HttpClientWrapper, PerformanceClientCore};
pub use constants::*;
pub use errors::{convert_reqwest_error_with_customer_id, ClientError};
pub use http::*;
pub use http_client::*;
pub use split_policy::*;
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

    // Initialize subscriber only once
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .init();
    });
}

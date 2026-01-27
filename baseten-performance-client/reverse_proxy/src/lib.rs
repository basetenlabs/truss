pub mod config;
pub mod handlers;
pub mod headers;
pub mod server;
pub mod constants;

// Re-export the main server function for testing
pub use server::create_server;

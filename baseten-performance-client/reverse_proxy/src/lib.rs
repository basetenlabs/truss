pub mod config;
pub mod constants;
pub mod handlers;
pub mod headers;
pub mod server;

// Re-export the main server function for testing
pub use server::create_server;

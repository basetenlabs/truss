use baseten_performance_client_core::RequestProcessingPreference;
use clap::Parser;
use std::sync::Arc;

mod config;
mod constants;
mod handlers;
mod headers;
mod server;

use config::ProxyConfig;

impl ProxyConfig {
    pub fn from(cli: Cli) -> Result<Self, Box<dyn std::error::Error>> {
        let default_preferences = RequestProcessingPreference::new()
            .with_max_concurrent_requests(cli.max_concurrent_requests)
            .with_batch_size(cli.batch_size)
            .with_timeout_s(cli.timeout_s);

        // Resolve upstream API key (from file if starts with /) - ASAP resolution
        let upstream_api_key = if let Some(key) = cli.upstream_api_key {
            if key.starts_with('/') {
                // Read API key from file immediately and replace with content
                Some(
                    std::fs::read_to_string(&key)
                        .map_err(|e| format!("Failed to read API key file '{}': {}", key, e))?
                        .trim()
                        .to_string(),
                )
            } else {
                Some(key)
            }
        } else {
            None
        };

        Ok(Self {
            port: cli.port,
            default_target_url: cli.target_url,
            upstream_api_key,
            http_version: cli.http_version,
            default_preferences,
        })
    }
}

#[derive(Parser)]
#[command(name = "baseten-reverse-proxy")]
#[command(about = "A high-performance reverse proxy for Baseten APIs")]
#[command(version)]
struct Cli {
    #[arg(short, long, default_value = "8080")]
    port: u16,

    #[arg(short = 'u', long)]
    target_url: Option<String>,

    #[arg(short = 'k', long)]
    upstream_api_key: Option<String>,

    #[arg(long, default_value = "2")]
    http_version: u8,

    #[arg(long, default_value = "64")]
    max_concurrent_requests: usize,

    #[arg(long, default_value = "32")]
    batch_size: usize,

    #[arg(long, default_value = "30.0")]
    timeout_s: f64,

    #[arg(long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Set log level via environment variable (core library initializes tracing automatically)
    std::env::set_var("RUST_LOG", &cli.log_level);

    let config = ProxyConfig::from(cli)?;

    tracing::info!("Starting Baseten Reverse Proxy");
    match &config.default_target_url {
        Some(url) => tracing::info!("Default Target URL: {}", url),
        None => tracing::info!("No default target URL configured (must be provided per request)"),
    }
    tracing::info!("Port: {}", config.port);
    tracing::info!("HTTP Version: {}", config.http_version);
    if let Some(ref key) = config.upstream_api_key {
        tracing::info!(
            "Upstream API Key: {}***",
            &key[..std::cmp::min(8, key.len())]
        );
    }

    server::create_server(Arc::new(config)).await?;

    Ok(())
}

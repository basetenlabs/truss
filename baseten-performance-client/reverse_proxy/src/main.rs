use clap::Parser;
use std::sync::Arc;
use std::str::FromStr;
use tracing::Level;

mod config;
mod handlers;
mod headers;
mod server;
mod constants;

use config::ProxyConfig;

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

    // Initialize tracing
    let log_level = Level::from_str(&cli.log_level)
        .map_err(|_| format!("Invalid log level: {}", cli.log_level))?;

    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .init();

    let config = ProxyConfig::from_cli(cli)?;

    tracing::info!("Starting Baseten Reverse Proxy");
    match &config.default_target_url {
        Some(url) => tracing::info!("Default Target URL: {}", url),
        None => tracing::info!("No default target URL configured (must be provided per request)"),
    }
    tracing::info!("Port: {}", config.port);
    tracing::info!("HTTP Version: {}", config.http_version);
    if let Some(ref key) = config.upstream_api_key {
        tracing::info!("Upstream API Key: {}***", &key[..std::cmp::min(8, key.len())]);
    }

    server::create_server(Arc::new(config)).await?;

    Ok(())
}

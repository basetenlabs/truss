use baseten_performance_client_core::RequestProcessingPreference;

#[derive(Debug, Clone)]
pub struct ProxyConfig {
    pub port: u16,
    pub target_url: String,
    pub http_version: u8,
    pub default_preferences: RequestProcessingPreference,
}

impl ProxyConfig {
    pub fn from_cli(cli: crate::Cli) -> Self {
        let default_preferences = RequestProcessingPreference::new()
            .with_max_concurrent_requests(cli.max_concurrent_requests)
            .with_batch_size(cli.batch_size)
            .with_timeout_s(cli.timeout_s);

        Self {
            port: cli.port,
            target_url: cli.target_url,
            http_version: cli.http_version,
            default_preferences,
        }
    }
}

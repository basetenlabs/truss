use baseten_performance_client_core::RequestProcessingPreference;

#[derive(Debug, Clone)]
pub struct ProxyConfig {
    pub port: u16,
    pub default_target_url: Option<String>,
    pub upstream_api_key: Option<String>,
    pub http_version: u8,
    pub default_preferences: RequestProcessingPreference,
}

impl ProxyConfig {
    pub fn from_cli(cli: crate::Cli) -> Result<Self, Box<dyn std::error::Error>> {
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
            None // No upstream API key provided - must be provided per request
        };

        Ok(Self {
            port: cli.port,
            default_target_url: cli.target_url,
            upstream_api_key,
            http_version: cli.http_version,
            default_preferences,
        })
    }

    /// Get the target URL for a request, using per-request override if available,
    /// otherwise falling back to the default target URL
    pub fn get_target_url(&self, per_request_target: Option<String>) -> Result<String, String> {
        per_request_target
            .or(self.default_target_url.clone())
            .ok_or_else(|| "No target URL configured".to_string())
    }
}

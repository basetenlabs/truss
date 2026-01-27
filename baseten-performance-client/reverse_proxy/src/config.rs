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
    /// Get the target URL for a request, using per-request override if available,
    /// otherwise falling back to the default target URL
    pub fn get_target_url(&self, per_request_target: Option<String>) -> Result<String, String> {
        per_request_target
            .or(self.default_target_url.clone())
            .ok_or_else(|| "No target URL configured".to_string())
    }
}

use baseten_performance_client_core::RequestProcessingPreference;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct ProxyConfig {
    pub port: u16,
    pub default_target_url: Option<String>,
    pub upstream_api_key: Option<String>,
    pub http_version: u8,
    pub default_preferences: RequestProcessingPreference,
    pub tokenizers: HashMap<String, TokenizerConfig>,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            default_target_url: None,
            upstream_api_key: None,
            http_version: 2,
            default_preferences: RequestProcessingPreference::new(),
            tokenizers: HashMap::new(),
        }
    }
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

#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    pub model_id: String,
    pub tokenizer_path: PathBuf,
}

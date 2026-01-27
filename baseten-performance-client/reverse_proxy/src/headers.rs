use axum::http::{HeaderMap, StatusCode};
use baseten_performance_client_core::RequestProcessingPreference;
use serde_json;
use crate::constants;
/// Extract API key from Authorization header
pub fn extract_api_key_from_header(headers: &HeaderMap) -> Result<String, StatusCode> {
    headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(|s| s.to_string())
        .ok_or(StatusCode::UNAUTHORIZED)
}


/// Parse RequestProcessingPreference from X-Baseten-Request-Preferences header
pub fn parse_preferences_from_header(
    headers: &HeaderMap,
    default_preferences: &RequestProcessingPreference,
) -> RequestProcessingPreference {
    // Try to extract preferences from header
    if let Some(preference_header) = headers.get("x-baseten-request-preferences") {
        if let Ok(preference_str) = preference_header.to_str() {
            if let Ok(preference_json) = serde_json::from_str::<serde_json::Value>(preference_str) {
                let mut preferences = default_preferences.clone();

                // Extract preference fields as needed
                if let Some(max_concurrent) = preference_json
                    .get("max_concurrent_requests")
                    .and_then(|v| v.as_u64())
                {
                    preferences.max_concurrent_requests = Some(max_concurrent as usize);
                }

                if let Some(batch_size) = preference_json.get("batch_size").and_then(|v| v.as_u64())
                {
                    preferences.batch_size = Some(batch_size as usize);
                }

                if let Some(timeout) = preference_json.get("timeout_s").and_then(|v| v.as_f64()) {
                    preferences.timeout_s = Some(timeout);
                }

                return preferences;
            }
        }
    }

    // Return defaults if header parsing fails
    default_preferences.clone()
}

/// Extract target URL from X-Target-Host header (optional)
/// If the value starts with '/', read from file
pub fn extract_target_url_from_header(headers: &HeaderMap) -> Option<String> {
    headers
        .get(constants::TARGET_HOST_HEADER)
        .and_then(|h| h.to_str().ok())
        .map(|s| {
            if s.starts_with('/') {
                // Read from file if starts with /
                match std::fs::read_to_string(s) {
                    Ok(content) => content.trim().to_string(),
                    Err(e) => {
                        tracing::warn!("Failed to read target host file '{}': {}", s, e);
                        s.to_string() // Fall back to the string value if file read fails
                    }
                }
            } else {
                s.to_string()
            }
        })
}

/// Extract customer request ID from header
pub fn extract_customer_request_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-baseten-customer-request-id")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string())
}

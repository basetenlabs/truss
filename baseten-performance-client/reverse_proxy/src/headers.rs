use axum::http::{HeaderMap, StatusCode};
use baseten_performance_client_core::RequestProcessingPreference;
use std::collections::HashMap;

/// Extract API key from Authorization header
pub fn extract_api_key_from_header(headers: &HeaderMap) -> Result<String, StatusCode> {
    headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(|s| s.to_string())
        .ok_or(StatusCode::UNAUTHORIZED)
}

/// Extract model from X-Baseten-Model header
pub fn extract_model_from_header(headers: &HeaderMap) -> Result<String, StatusCode> {
    headers
        .get("x-baseten-model")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string())
        .ok_or(StatusCode::BAD_REQUEST)
}

/// Parse RequestProcessingPreference from X-Baseten-Request-Preferences header
pub fn parse_preferences_from_header(
    headers: &HeaderMap,
    default_preferences: &RequestProcessingPreference,
) -> RequestProcessingPreference {
    // For now, just return defaults since we can't easily deserialize the complex struct
    // In a real implementation, you might want to parse individual fields from the header
    default_preferences.clone()
}

/// Extract customer request ID from header
pub fn extract_customer_request_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-baseten-customer-request-id")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string())
}

/// Convert HeaderMap to HashMap for JSON serialization
pub fn header_map_to_hashmap(headers: &HeaderMap) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for (name, value) in headers {
        if let Ok(value_str) = value.to_str() {
            map.insert(name.to_string(), value_str.to_string());
        }
    }
    map
}

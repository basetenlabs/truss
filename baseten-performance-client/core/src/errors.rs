use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum ClientError {
    LocalTimeout(String, Option<String>),
    RemoteTimeout(String, Option<String>),
    Network(String),
    Connect(String),
    Http {
        status: u16,
        message: String,
        customer_request_id: Option<String>,
    },
    InvalidParameter(String),
    Serialization(String),
    Cancellation(String),
}

impl fmt::Display for ClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClientError::LocalTimeout(msg, customer_request_id) => {
                if let Some(ref cid) = customer_request_id {
                    write!(f, "Local timeout error (Request ID: {}): {}", cid, msg)
                } else {
                    write!(f, "Local timeout error: {}", msg)
                }
            }
            ClientError::RemoteTimeout(msg, customer_request_id) => {
                if let Some(ref cid) = customer_request_id {
                    write!(f, "Remote timeout error (Request ID: {}): {}", cid, msg)
                } else {
                    write!(f, "Remote timeout error: {}", msg)
                }
            }
            ClientError::Network(msg) => write!(f, "Network error: {}", msg),
            ClientError::Http {
                status,
                message,
                customer_request_id,
            } => {
                if let Some(ref cid) = customer_request_id {
                    write!(
                        f,
                        "HTTP {} error (Request ID: {}): {}",
                        status, cid, message
                    )
                } else {
                    write!(f, "HTTP {} error: {}", status, message)
                }
            }
            ClientError::Connect(msg) => write!(f, "Connection error: {}", msg),
            ClientError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            ClientError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            ClientError::Cancellation(msg) => write!(f, "Operation cancelled: {}", msg),
        }
    }
}

impl Error for ClientError {}

/// Classifies timeout errors as local or remote based on error analysis
pub fn classify_timeout_error(
    err: &reqwest::Error,
    customer_request_id: Option<String>,
) -> ClientError {
    // Extract URL information
    let url_info = err
        .url()
        .map(|u| u.to_string())
        .unwrap_or_else(|| "unknown URL".to_string());

    // Build enhanced error message with context
    let base_message = err.to_string();
    let mut enhanced_message = String::new();

    // Add URL context
    enhanced_message.push_str(&format!("URL: {}", url_info));

    // Add error kind context
    let error_kind = if err.is_connect() {
        "connect"
    } else if err.is_request() {
        "request"
    } else if err.is_body() {
        "body"
    } else if err.is_decode() {
        "decode"
    } else {
        "unknown"
    };
    enhanced_message.push_str(&format!(" | Error kind: {}", error_kind));

    // Add the original error message
    enhanced_message.push_str(&format!(" | Details: {}", base_message));

    // Add source chain information
    let mut source = err.source();
    if let Some(first_source) = source {
        enhanced_message.push_str(&format!(" | Source: {}", first_source));

        // Add additional sources if they exist
        source = first_source.source();
        while let Some(e) = source {
            enhanced_message.push_str(&format!(" | Caused by: {}", e));
            source = e.source();
        }
    }

    // Enhanced classification using multiple criteria
    let msg_lower = base_message.to_lowercase();
    let is_local_timeout =
        // Message-based indicators
        msg_lower.contains("sending") ||
        msg_lower.contains("connect") ||
        msg_lower.contains("dns") ||
        msg_lower.contains("connection") ||
        msg_lower.contains("timeout while") ||
        msg_lower.contains("unable to send") ||
        (msg_lower.contains("timed out") && (msg_lower.contains("waiting") || msg_lower.contains("connecting"))) ||
        // Error kind-based indicators
        err.is_connect() ||
        // Source-based indicators (look for specific timeout types in source chain)
        has_local_timeout_source(err);

    if is_local_timeout {
        ClientError::LocalTimeout(
            format!(
                "Unable to send request within timeout: {}",
                enhanced_message
            ),
            customer_request_id,
        )
    } else {
        ClientError::RemoteTimeout(
            format!(
                "Server did not respond within timeout: {}",
                enhanced_message
            ),
            customer_request_id,
        )
    }
}

/// Checks if the error source chain indicates a local timeout
fn has_local_timeout_source(err: &reqwest::Error) -> bool {
    let mut source = err.source();
    while let Some(e) = source {
        let source_str = e.to_string().to_lowercase();
        if source_str.contains("connect")
            || source_str.contains("dns")
            || source_str.contains("resolve")
            || source_str.contains("binding")
            || source_str.contains("network unreachable")
            || source_str.contains("connection refused")
        {
            return true;
        }
        source = e.source();
    }
    false
}

impl From<reqwest::Error> for ClientError {
    fn from(err: reqwest::Error) -> Self {
        let mut message = err.to_string();
        let mut source = err.source();
        while let Some(e) = source {
            message.push_str(&format!("\n  caused by: {}", e));
            source = e.source();
        }

        if err.is_timeout() {
            classify_timeout_error(&err, None) // Will be set by caller
        } else if err.is_connect() {
            ClientError::Connect(message)
        } else {
            ClientError::Network(message)
        }
    }
}

impl From<serde_json::Error> for ClientError {
    fn from(err: serde_json::Error) -> Self {
        ClientError::Serialization(err.to_string())
    }
}

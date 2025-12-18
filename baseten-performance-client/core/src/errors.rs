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
                Self::format_with_request_id(f, "Local timeout error", msg, customer_request_id)
            }
            ClientError::RemoteTimeout(msg, customer_request_id) => {
                Self::format_with_request_id(f, "Remote timeout error", msg, customer_request_id)
            }
            ClientError::Network(msg) => write!(f, "Network error: {}", msg),
            ClientError::Connect(msg) => write!(f, "Connection error: {}", msg),
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
            ClientError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            ClientError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            ClientError::Cancellation(msg) => write!(f, "Cancellation error: {}", msg),
        }
    }
}

impl ClientError {
    fn format_with_request_id(
        f: &mut fmt::Formatter<'_>,
        error_type: &str,
        msg: &str,
        customer_request_id: &Option<String>,
    ) -> fmt::Result {
        if let Some(ref cid) = customer_request_id {
            write!(f, "{} (Request ID: {}): {}", error_type, cid, msg)
        } else {
            write!(f, "{}: {}", error_type, msg)
        }
    }
}

impl Error for ClientError {}

/// Build enhanced error message with comprehensive context
fn build_enhanced_error_message(err: &reqwest::Error, customer_request_id: Option<&str>) -> String {
    let mut parts = Vec::new();

    // Add customer request ID if available
    if let Some(cid) = customer_request_id {
        parts.push(format!("Request ID: {}", cid));
    }

    // Add the main error message (Display format)
    parts.push(format!("Error: {}", err));

    // Add Debug format information (contains "(Connect)" and other details)
    parts.push(format!("Debug: {:?}", err));

    // Add source chain information
    let mut source = err.source();
    if let Some(first_source) = source {
        parts.push(format!("Source: {}", first_source));
        source = first_source.source();
        while let Some(e) = source {
            parts.push(format!("Caused by: {}", e));
            source = e.source();
        }
    }

    // Add error type classification
    let error_type = if err.is_connect() {
        "Connection error"
    } else if err.is_timeout() {
        "Timeout error"
    } else if err.is_request() {
        "Request error"
    } else {
        "Network error"
    };
    parts.push(format!("Type: {}", error_type));

    parts.join("\n  ")
}

/// Determine if timeout is local (connection issues) or remote (server response)
fn is_local_timeout(err: &reqwest::Error) -> bool {
    err.is_connect()
        || {
            let mut source = err.source();
            while let Some(e) = source {
                let source_str = e.to_string().to_lowercase();
                if source_str.contains("connect")
                    || source_str.contains("dns")
                    || source_str.contains("resolve")
                    || source_str.contains("network unreachable")
                    || source_str.contains("connection refused")
                {
                    return true;
                }
                source = e.source();
            }
            false
        }
        || {
            let base_message = err.to_string().to_lowercase();
            base_message.contains("dns")
                || base_message.contains("resolve")
                || base_message.contains("network unreachable")
                || base_message.contains("connection refused")
        }
}

/// Convert reqwest error to ClientError with customer request ID context
pub fn convert_reqwest_error_with_customer_id(
    err: reqwest::Error,
    customer_request_id: crate::customer_request_id::CustomerRequestId,
) -> ClientError {
    let message = build_enhanced_error_message(&err, Some(&customer_request_id.to_string()));

    if err.is_timeout() {
        if is_local_timeout(&err) {
            ClientError::LocalTimeout(
                format!("Unable to send request within timeout: {}", message),
                Some(customer_request_id.to_string()),
            )
        } else {
            ClientError::RemoteTimeout(
                format!("Server did not respond within timeout: {}", message),
                Some(customer_request_id.to_string()),
            )
        }
    } else if err.is_connect() {
        ClientError::Connect(message)
    } else {
        ClientError::Network(message)
    }
}

/// Classify timeout errors as local or remote (legacy function for backward compatibility)
fn classify_timeout_error(
    err: &reqwest::Error,
    customer_request_id: Option<String>,
) -> ClientError {
    let message = build_enhanced_error_message(err, customer_request_id.as_deref());

    if err.is_timeout() {
        if is_local_timeout(err) {
            ClientError::LocalTimeout(
                format!("Unable to send request within timeout: {}", message),
                customer_request_id,
            )
        } else {
            ClientError::RemoteTimeout(
                format!("Server did not respond within timeout: {}", message),
                customer_request_id,
            )
        }
    } else {
        ClientError::Network(message)
    }
}

impl From<reqwest::Error> for ClientError {
    fn from(err: reqwest::Error) -> Self {
        let message = build_enhanced_error_message(&err, None);

        if err.is_timeout() {
            classify_timeout_error(&err, None)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_enhanced_connection_error_formatting() {
        let client = reqwest::Client::new();
        let result = client.get("https://bla.notexist").send();

        match result.await {
            Ok(_) => panic!("Expected connection error but got success"),
            Err(reqwest_err) => {
                assert!(reqwest_err.is_connect(), "Expected connection error");
                let client_error: ClientError = reqwest_err.into();
                let error_str = format!("{}", client_error);

                assert!(
                    error_str.contains("(Connect)"),
                    "Enhanced error should contain '(Connect)' from Debug format. Got: {}",
                    error_str
                );
                assert!(
                    error_str.contains("Debug:"),
                    "Enhanced error should contain 'Debug:' information. Got: {}",
                    error_str
                );
                assert!(
                    error_str.contains("Type: Connection error"),
                    "Enhanced error should contain type information. Got: {}",
                    error_str
                );

                println!("✅ Enhanced connection error formatting test passed");
                println!("Error message: {}", error_str);
            }
        }
    }

    #[tokio::test]
    async fn test_error_builder_with_customer_request_id() {
        let client = reqwest::Client::new();
        let reqwest_err = match client.get("http://localhost:99999").send().await {
            Ok(_) => panic!("Expected error but got success"),
            Err(e) => e,
        };

        let customer_id = crate::customer_request_id::CustomerRequestId::new_batch();
        let client_error = convert_reqwest_error_with_customer_id(reqwest_err, customer_id);

        let error_str = format!("{}", client_error);
        assert!(error_str.contains("Request ID:"));
        assert!(error_str.contains("perfclient-"));
    }
}

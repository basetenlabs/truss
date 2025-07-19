use thiserror::Error;

/// Common errors that can occur during storage operations
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Authentication failed: {message}")]
    Authentication { message: String },

    #[error("Network error: {message}")]
    Network { message: String },

    #[error("Invalid URI format: {uri} - {reason}")]
    InvalidUri { uri: String, reason: String },

    #[error("File not found: {path}")]
    FileNotFound { path: String },

    #[error("Permission denied: {operation}")]
    PermissionDenied { operation: String },

    #[error("Storage service unavailable: {service}")]
    ServiceUnavailable { service: String },

    #[error("Configuration error: {message}")]
    Configuration { message: String },

    #[error("Download failed: {message}")]
    Download { message: String },

    #[error("Cache operation failed: {message}")]
    Cache { message: String },

    #[error("Provider error ({provider}): {message}")]
    Provider { provider: String, message: String },

    #[error("Unknown error: {0}")]
    Unknown(#[from] anyhow::Error),
}

impl StorageError {
    /// Check if this error is recoverable (e.g., network issues that might resolve with retry)
    pub fn is_recoverable(&self) -> bool {
        matches!(self,
            StorageError::Network { .. } |
            StorageError::ServiceUnavailable { .. } |
            StorageError::Download { .. }
        )
    }

    /// Check if this error is due to missing credentials/authentication
    pub fn is_authentication_error(&self) -> bool {
        matches!(self, StorageError::Authentication { .. })
    }

    /// Check if this error is due to configuration issues
    pub fn is_configuration_error(&self) -> bool {
        matches!(self,
            StorageError::Configuration { .. } |
            StorageError::InvalidUri { .. }
        )
    }

    /// Get a user-friendly error message for common issues
    pub fn user_friendly_message(&self) -> String {
        match self {
            StorageError::Authentication { .. } => {
                "Authentication failed. Please check your credentials and ensure the secret file contains valid authentication information.".to_string()
            }
            StorageError::InvalidUri { uri, .. } => {
                format!("Invalid URI format: '{}'. Please check the URI follows the correct format for your storage provider.", uri)
            }
            StorageError::Network { .. } => {
                "Network error occurred. Please check your internet connection and try again.".to_string()
            }
            StorageError::Configuration { message } => {
                format!("Configuration error: {}. Please check your storage provider settings.", message)
            }
            _ => self.to_string()
        }
    }

    /// Create an authentication error
    pub fn auth_error(message: impl Into<String>) -> Self {
        StorageError::Authentication { message: message.into() }
    }

    /// Create a network error
    pub fn network_error(message: impl Into<String>) -> Self {
        StorageError::Network { message: message.into() }
    }

    /// Create an invalid URI error
    pub fn invalid_uri(uri: impl Into<String>, reason: impl Into<String>) -> Self {
        StorageError::InvalidUri {
            uri: uri.into(),
            reason: reason.into()
        }
    }

    /// Create a provider-specific error
    pub fn provider_error(provider: impl Into<String>, message: impl Into<String>) -> Self {
        StorageError::Provider {
            provider: provider.into(),
            message: message.into()
        }
    }
}

/// Convert common error types to StorageError
impl From<reqwest::Error> for StorageError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() || err.is_connect() {
            StorageError::Network { message: err.to_string() }
        } else if err.is_status() {
            match err.status() {
                Some(status) if status.is_client_error() => {
                    if status.as_u16() == 401 || status.as_u16() == 403 {
                        StorageError::Authentication { message: err.to_string() }
                    } else if status.as_u16() == 404 {
                        StorageError::FileNotFound { path: "remote resource".to_string() }
                    } else {
                        StorageError::Network { message: err.to_string() }
                    }
                }
                Some(status) if status.is_server_error() => {
                    StorageError::ServiceUnavailable {
                        service: "remote storage".to_string()
                    }
                }
                _ => StorageError::Network { message: err.to_string() }
            }
        } else {
            StorageError::Network { message: err.to_string() }
        }
    }
}

impl From<object_store::Error> for StorageError {
    fn from(err: object_store::Error) -> Self {
        let error_msg = err.to_string().to_lowercase();

        if error_msg.contains("auth") || error_msg.contains("credential") || error_msg.contains("unauthorized") {
            StorageError::Authentication { message: err.to_string() }
        } else if error_msg.contains("not found") || error_msg.contains("404") {
            StorageError::FileNotFound { path: "storage object".to_string() }
        } else if error_msg.contains("network") || error_msg.contains("timeout") || error_msg.contains("connection") {
            StorageError::Network { message: err.to_string() }
        } else if error_msg.contains("service") || error_msg.contains("unavailable") || error_msg.contains("503") {
            StorageError::ServiceUnavailable { service: "storage service".to_string() }
        } else {
            StorageError::Provider {
                provider: "object_store".to_string(),
                message: err.to_string()
            }
        }
    }
}

impl From<std::io::Error> for StorageError {
    fn from(err: std::io::Error) -> Self {
        match err.kind() {
            std::io::ErrorKind::NotFound => {
                StorageError::FileNotFound { path: "local file".to_string() }
            }
            std::io::ErrorKind::PermissionDenied => {
                StorageError::PermissionDenied { operation: "file operation".to_string() }
            }
            std::io::ErrorKind::TimedOut => {
                StorageError::Network { message: "I/O timeout".to_string() }
            }
            _ => StorageError::Unknown(anyhow::anyhow!("I/O error: {}", err))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_classification() {
        let auth_err = StorageError::auth_error("Invalid token");
        assert!(auth_err.is_authentication_error());
        assert!(!auth_err.is_recoverable());

        let network_err = StorageError::network_error("Connection timeout");
        assert!(network_err.is_recoverable());
        assert!(!network_err.is_authentication_error());

        let config_err = StorageError::invalid_uri("invalid://uri", "unknown scheme");
        assert!(config_err.is_configuration_error());
        assert!(!config_err.is_recoverable());
    }

    #[test]
    fn test_user_friendly_messages() {
        let auth_err = StorageError::auth_error("Invalid credentials");
        assert!(auth_err.user_friendly_message().contains("Authentication failed"));

        let uri_err = StorageError::invalid_uri("bad://uri", "unsupported");
        assert!(uri_err.user_friendly_message().contains("Invalid URI format"));
    }
}

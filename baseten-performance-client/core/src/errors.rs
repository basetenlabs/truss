use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum ClientError {
    Timeout(String),
    Network(String),
    Http { status: u16, message: String },
    InvalidParameter(String),
    Serialization(String),
    Cancellation(String),
}

impl fmt::Display for ClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClientError::Timeout(msg) => write!(f, "Timeout error: {}", msg),
            ClientError::Network(msg) => write!(f, "Network error: {}", msg),
            ClientError::Http { status, message } => write!(f, "HTTP {} error: {}", status, message),
            ClientError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            ClientError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            ClientError::Cancellation(msg) => write!(f, "Operation cancelled: {}", msg),
        }
    }
}

impl Error for ClientError {}

impl From<reqwest::Error> for ClientError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            ClientError::Timeout(err.to_string())
        } else if err.is_connect() {
            ClientError::Network(format!("Connection error: {}", err))
        } else {
            ClientError::Network(err.to_string())
        }
    }
}

impl From<serde_json::Error> for ClientError {
    fn from(err: serde_json::Error) -> Self {
        ClientError::Serialization(err.to_string())
    }
}

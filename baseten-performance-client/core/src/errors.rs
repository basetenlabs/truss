use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum ClientError {
    Timeout(String),
    Network(String),
    Connect(String),
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
            ClientError::Http { status, message } => {
                write!(f, "HTTP {} error: {}", status, message)
            }
            ClientError::Connect(msg) => write!(f, "Connection error: {}", msg),
            ClientError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            ClientError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            ClientError::Cancellation(msg) => write!(f, "Operation cancelled: {}", msg),
        }
    }
}

impl Error for ClientError {}

impl From<reqwest::Error> for ClientError {
    fn from(err: reqwest::Error) -> Self {
        let mut message = err.to_string();
        let mut source = err.source();
        while let Some(e) = source {
            message.push_str(&format!("\n  caused by: {}", e));
            source = e.source();
        }

        if err.is_timeout() {
            ClientError::Timeout(message)
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

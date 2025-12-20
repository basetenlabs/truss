use std::collections::HashMap;

/// Enhanced stream events for channel-based communication
#[derive(Debug)]
pub enum StreamEvent {
    /// JSON data parsed from the event
    Json(serde_json::Value),
    /// Plain text data from the event
    Text(String),
    /// Stream ended successfully
    End,
    /// Error occurred during streaming
    Error(crate::errors::ClientError),
}

impl Clone for StreamEvent {
    fn clone(&self) -> Self {
        match self {
            StreamEvent::Json(value) => StreamEvent::Json(value.clone()),
            StreamEvent::Text(text) => StreamEvent::Text(text.clone()),
            StreamEvent::End => StreamEvent::End,
            StreamEvent::Error(_) => {
                // For errors, we can't clone the ClientError, so we create a new error
                // This is a limitation but acceptable for streaming use case
                StreamEvent::Error(crate::errors::ClientError::Network("Error event cannot be cloned".to_string()))
            }
        }
    }
}

impl PartialEq for StreamEvent {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (StreamEvent::Json(a), StreamEvent::Json(b)) => a == b,
            (StreamEvent::Text(a), StreamEvent::Text(b)) => a == b,
            (StreamEvent::End, StreamEvent::End) => true,
            (StreamEvent::Error(_), StreamEvent::Error(_)) => true, // Compare errors loosely
            _ => false,
        }
    }
}

impl Eq for StreamEvent {}

/// Represents a Server-Sent Events (SSE) event
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SSEEvent {
    /// JSON data parsed from the event
    JsonData(serde_json::Value),
    /// Plain text data from the event
    TextData(String),
    /// A comment line starting with :
    Comment(String),
    /// Connection established event with response details
    Connected(ConnectionDetails),
}

impl From<SSEEvent> for StreamEvent {
    fn from(sse_event: SSEEvent) -> Self {
        match sse_event {
            SSEEvent::JsonData(json) => StreamEvent::Json(json),
            SSEEvent::TextData(text) => StreamEvent::Text(text),
            SSEEvent::Comment(text) => StreamEvent::Text(format!(":{}", text)),
            SSEEvent::Connected(details) => {
                // Convert connection details to JSON
                let mut details_map = serde_json::Map::new();
                details_map.insert("status".to_string(), serde_json::Value::Number(serde_json::Number::from(details.status)));
                details_map.insert("headers".to_string(), serde_json::to_value(&details.headers).unwrap_or_default());
                StreamEvent::Json(serde_json::Value::Object(details_map))
            }
        }
    }
}

/// Details about the SSE connection
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConnectionDetails {
    /// HTTP status code of the connection
    pub status: u16,
    /// Response headers
    pub headers: HashMap<String, String>,
}

/// A parsed SSE event
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Event {
    /// Event type (defaults to "message" if not specified)
    pub event_type: String,
    /// Event data (may contain newlines if multiple data fields were present)
    pub data: String,
    /// Optional event ID for reconnection
    pub id: Option<String>,
    /// Optional retry time in milliseconds
    pub retry: Option<u64>,
}

impl SSEEvent {
    /// Attempt to parse event data as JSON
    pub fn from_event(event: Event) -> Self {
        // Try JSON parsing first
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&event.data) {
            SSEEvent::JsonData(json)
        } else {
            SSEEvent::TextData(event.data)
        }
    }
}

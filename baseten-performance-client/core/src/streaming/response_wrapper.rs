use crate::streaming::*;
use crate::ClientError;
use pin_project::pin_project;
use std::pin::Pin;
use std::time::Duration;
use futures::Stream;

/// Streaming response wrapper
#[pin_project]
pub struct StreamingResponse {
    #[pin]
    pub events: Pin<Box<dyn Stream<Item = Result<SSEEvent, ClientError>> + Send>>,
    pub total_time: Duration,
    pub response_headers: std::collections::HashMap<String, String>,
}

impl StreamingResponse {
    /// Create a new streaming response
    pub fn new(
        events: Pin<Box<dyn Stream<Item = Result<SSEEvent, ClientError>> + Send>>,
        total_time: Duration,
        response_headers: std::collections::HashMap<String, String>,
    ) -> Self {
        Self {
            events,
            total_time,
            response_headers,
        }
    }
}

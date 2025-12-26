//! Client extension for streaming functionality
//!
//! This module provides an extension trait to add streaming methods to the core client
//! without cluttering the main client implementation.

use crate::streaming::request_handler;
use crate::streaming::{StreamEvent};
use crate::{PerformanceClientCore, RequestProcessingPreference};
use tokio::task::JoinHandle;

/// Extension trait to add streaming functionality to PerformanceClientCore
pub trait StreamingClientExt {
    /// Stream Server-Sent Events from an endpoint
    ///
    /// This method provides a channel-based streaming interface for SSE events.
    ///
    /// # Arguments
    /// * `endpoint` - The API endpoint to stream from (e.g., "/v1/chat/completions")
    /// * `payload` - JSON payload for the request
    /// * `method` - HTTP method (defaults to POST if None)
    /// * `preference` - Request processing preferences
    ///
    /// # Returns
    /// A tuple containing the receiver channel and a task handle for cleanup
    async fn stream(
        &self,
        endpoint: String,
        payload: serde_json::Value,
        method: Option<String>,
        preference: &RequestProcessingPreference,
    ) -> Result<(tokio::sync::mpsc::Receiver<StreamEvent>, JoinHandle<()>), crate::errors::ClientError>;
}

impl StreamingClientExt for PerformanceClientCore {
    async fn stream(
        &self,
        endpoint: String,
        payload: serde_json::Value,
        method: Option<String>,
        preference: &RequestProcessingPreference,
    ) -> Result<(tokio::sync::mpsc::Receiver<StreamEvent>, JoinHandle<()>), crate::errors::ClientError> {
        request_handler::stream(
            &self.api_key,
            &self.base_url,
            &self.get_client_wrapper(),
            endpoint,
            payload,
            method,
            preference,
        ).await
    }
}

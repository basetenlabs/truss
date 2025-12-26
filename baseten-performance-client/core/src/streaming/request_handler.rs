//! Streaming request handler
//!
//! This module contains the streaming request logic extracted from the main client
//! to reduce clutter and improve organization.

use crate::streaming::{StreamEvent, SSEEvent, EventParser, ConnectionDetails, StreamingResponse};
use crate::customer_request_id::CustomerRequestId;
use crate::{RequestProcessingPreference, HttpClientWrapper, ClientError};

use async_stream::stream;
use std::collections::HashMap;
use tokio::task::JoinHandle;
use tokio_stream::StreamExt;

/// Core streaming functionality extracted from the main client

    /// Stream Server-Sent Events from an endpoint
    ///
    /// This method uses the core HTTP client for the request
    /// and the dedicated SSE parser for processing the response.
    ///
    /// # Arguments
    /// * `api_key` - API key for authentication
    /// * `base_url` - Base URL for the API endpoint
    /// * `client_wrapper` - HTTP client wrapper
    /// * `endpoint` - The API endpoint to stream from (e.g., "/v1/chat/completions")
    /// * `payload` - JSON payload for the request
    /// * `method` - HTTP method (defaults to POST if None)
    /// * `preference` - Request processing preferences
    ///
    /// # Returns
    /// A `StreamingResponse` containing a stream of `SSEEvent`s
    pub async fn stream_request(
        api_key: &str,
        base_url: &str,
        client_wrapper: &HttpClientWrapper,
        endpoint: String,
        payload: serde_json::Value,
        method: Option<String>,
        preference: &RequestProcessingPreference,
    ) -> Result<StreamingResponse, ClientError> {
        let start_time = std::time::Instant::now();
        let url = format!("{}{}", base_url, endpoint);

        // Get HTTP client from existing wrapper
        let client = client_wrapper.get_client();

        // Build request using existing HTTP client infrastructure
        let mut request_builder = match method.as_deref() {
            Some("GET") => client.get(&url),
            Some("POST") | Some(_) | None => client.post(&url),
        };

        request_builder = request_builder
            .bearer_auth(api_key)
            .json(&payload)
            .timeout(std::time::Duration::from_secs_f64(preference.timeout_s.unwrap_or(3600.0)));

        // Add customer request ID header (following existing pattern)
        let customer_request_id = CustomerRequestId::new_batch();
        request_builder = request_builder.header(
            crate::constants::CUSTOMER_HEADER_NAME,
            customer_request_id.to_string(),
        );

        // Send request and get streaming response
        let response = request_builder.send().await
            .map_err(|e| crate::errors::convert_reqwest_error_with_customer_id(e, customer_request_id.clone()))?;

        // Ensure successful response
        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ClientError::Http {
                status: status.as_u16(),
                message: format!("API request failed with status {}: {}", status, error_text),
                customer_request_id: Some(customer_request_id.to_string()),
            });
        }

        // Extract response headers
        let mut headers_map = HashMap::new();
        for (name, value) in response.headers().iter() {
            headers_map.insert(
                name.as_str().to_string(),
                String::from_utf8_lossy(value.as_bytes()).into_owned(),
            );
        }

        // Create streaming response
        let events_stream = Box::pin(stream! {
            let mut bytes_stream = response.bytes_stream();
            let mut parser = EventParser::new();

            while let Some(chunk_result) = bytes_stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        // Process the chunk with the SSE parser
                        if let Err(parse_err) = parser.process_bytes(&chunk) {
                            yield Err(ClientError::Network(format!("SSE parse error: {}", parse_err)));
                            break;
                        }

                        // Extract any parsed events
                        while let Some(sse) = parser.get_event() {
                            let event = match sse {
                                crate::streaming::event_parser::SSE::Event(event) => {
                                    // Try JSON parsing first, fall back to text
                                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&event.data) {
                                        SSEEvent::JsonData(json)
                                    } else {
                                        SSEEvent::TextData(event.data)
                                    }
                                }
                                crate::streaming::event_parser::SSE::Comment(text) => SSEEvent::Comment(text),
                                crate::streaming::event_parser::SSE::Connected(details) => SSEEvent::Connected(ConnectionDetails {
                                    status: details.response().status(),
                                    headers: details.response().headers(),
                                }),
                            };
                            yield Ok(event);
                        }
                    }
                    Err(e) => {
                        let client_error = crate::errors::convert_reqwest_error_with_customer_id(e, customer_request_id.clone());
                        yield Err(client_error);
                        break;
                    }
                }
            }
        });

        let total_time = start_time.elapsed();

        Ok(StreamingResponse::new(events_stream, total_time, headers_map))
    }

    /// Channel-based streaming request with background task processing
    ///
    /// This method provides a more efficient streaming approach by using a channel
    /// to communicate between a background processing task and the main thread.
    ///
    /// # Arguments
    /// * `api_key` - API key for authentication
    /// * `base_url` - Base URL for the API endpoint
    /// * `client_wrapper` - HTTP client wrapper
    /// * `endpoint` - The API endpoint to stream from (e.g., "/v1/chat/completions")
    /// * `payload` - JSON payload for the request
    /// * `method` - HTTP method (defaults to POST if None)
    /// * `preference` - Request processing preferences
    ///
    /// # Returns
    /// A tuple containing the receiver channel and a task handle for cleanup
    pub async fn stream(
        api_key: &str,
        base_url: &str,
        client_wrapper: &HttpClientWrapper,
        endpoint: String,
        payload: serde_json::Value,
        method: Option<String>,
        preference: &RequestProcessingPreference,
    ) -> Result<(tokio::sync::mpsc::Receiver<StreamEvent>, JoinHandle<()>), ClientError> {
        use tokio::sync::mpsc;

        let (tx, rx) = mpsc::channel(100); // Buffer up to 100 events
        let api_key_owned = api_key.to_string();
        let base_url_owned = base_url.to_string();
        let client_wrapper_owned = client_wrapper.clone();
        let endpoint_clone = endpoint.clone();
        let payload_clone = payload.clone();
        let method_clone = method.clone();
        let preference_clone = preference.clone();

        // Spawn background task to process the stream
        let task_handle = tokio::spawn(async move {
            let result = stream_request(
                &api_key_owned,
                &base_url_owned,
                &client_wrapper_owned,
                endpoint_clone,
                payload_clone,
                method_clone,
                &preference_clone,
            ).await;

            match result {
                Ok(streaming_response) => {
                    let mut event_stream = streaming_response.events;

                    while let Some(event_result) = event_stream.next().await {
                        match event_result {
                            Ok(sse_event) => {
                                let stream_event = StreamEvent::from(sse_event);
                                if tx.send(stream_event).await.is_err() {
                                    // Receiver was dropped, stop processing
                                    break;
                                }
                            }
                            Err(e) => {
                                let _ = tx.send(StreamEvent::Error(e)).await;
                                break;
                            }
                        }
                    }

                    // Send end event
                    let _ = tx.send(StreamEvent::End).await;
                }
                Err(e) => {
                    let _ = tx.send(StreamEvent::Error(e)).await;
                }
            }
        });

        Ok((rx, task_handle))
    }

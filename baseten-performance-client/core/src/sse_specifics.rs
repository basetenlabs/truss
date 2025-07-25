use crate::errors::ClientError;
use eventsource_client::{ClientBuilder, Client};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use futures::StreamExt;

/// StreamEvent enum for messages sent from the background streaming task
#[derive(Debug)]
pub enum StreamEvent {
    Json(Value),
    Text(String),
    Error(ClientError),
    End,
}

/// Configuration for SSE streaming requests
pub struct StreamConfig {
    pub url: String,
    pub headers: std::collections::HashMap<String, String>,
    pub method: String,
    pub body: Option<String>,
}

/// SSE streaming client that handles Server-Sent Events
pub struct SSEClient {
    api_key: String,
    base_url: String,
}

impl SSEClient {
    pub fn new(api_key: String, base_url: String) -> Self {
        Self { api_key, base_url }
    }

    /// Start streaming SSE events from the given endpoint with a POST request
    pub fn stream_events(
        &self,
        endpoint: &str,
        payload: Value,
        method: String,
    ) -> Result<(mpsc::Receiver<StreamEvent>, JoinHandle<()>), ClientError> {
        let (tx, rx) = mpsc::channel::<StreamEvent>(100); // Buffer size of 100

        let base = self.base_url.trim_end_matches('/');
        let endpoint = endpoint.trim_start_matches('/');
        let full_url = format!("{}/{}", base, endpoint);

        if !full_url.starts_with("http://") && !full_url.starts_with("https://") {
            return Err(ClientError::InvalidParameter(format!("Invalid URL format: {} (base: {}, endpoint: {})", full_url, base, endpoint)));
        }

        // Clone data for the async task
        let api_key = self.api_key.clone();
        let payload_json = serde_json::to_string(&payload)
            .map_err(|e| ClientError::InvalidParameter(format!("Failed to serialize payload: {}", e)))?;

        // Spawn async task to handle streaming
        let handle = tokio::spawn(async move {
            let result = async {
                // Build the SSE client with POST method and body
                let client = ClientBuilder::for_url(&full_url)
                    .map_err(|e| ClientError::Network(format!("Failed to create SSE client: {}", e)))?
                    .method(method)
                    .body(payload_json)
                    .header("Authorization", &format!("Bearer {}", api_key))
                    .map_err(|e| ClientError::Network(format!("Failed to set authorization header: {}", e)))?
                    .header("Content-Type", "application/json")
                    .map_err(|e| ClientError::Network(format!("Failed to set content-type header: {}", e)))?
                    .build();

                // Start streaming
                let mut stream = client.stream();

                while let Some(event_result) = stream.next().await {
                    match event_result {
                        Ok(event) => {
                            match event {
                                eventsource_client::SSE::Event(evt) => {
                                    // Check for OpenAI's end-of-stream signal
                                    if evt.data == "[DONE]" {
                                        break;
                                    }

                                    // Try to parse as JSON
                                    match serde_json::from_str::<Value>(&evt.data) {
                                        Ok(json_val) => {
                                            if tx.send(StreamEvent::Json(json_val)).await.is_err() {
                                                break; // receiver gone
                                            }
                                        }
                                        Err(_) => {
                                            // Send as text if JSON parsing fails
                                            if tx.send(StreamEvent::Text(evt.data)).await.is_err() {
                                                break;
                                            }
                                        }
                                    }
                                }
                                eventsource_client::SSE::Comment(comment) => {
                                    // Send comments as text
                                    if tx.send(StreamEvent::Text(comment)).await.is_err() {
                                        break;
                                    }
                                }
                                eventsource_client::SSE::Connected(_) => {
                                    // Connection established, continue streaming
                                    continue;
                                }
                            }
                        }
                        Err(e) => {
                            // Convert eventsource error to ClientError
                            let client_error = match e {
                                eventsource_client::Error::TimedOut => ClientError::Timeout("SSE stream timed out".to_string()),
                                eventsource_client::Error::Eof => ClientError::Network("SSE stream ended unexpectedly".to_string()),
                                eventsource_client::Error::StreamClosed => ClientError::Network("SSE stream was closed".to_string()),
                                _ => ClientError::Network(format!("SSE stream error: {}", e)),
                            };

                            if tx.send(StreamEvent::Error(client_error)).await.is_err() {
                                break;
                            }
                            break;
                        }
                    }
                }

                Ok::<(), ClientError>(())
            }.await;

            // Handle the final result
            if let Err(err) = result {
                let _ = tx.send(StreamEvent::Error(err)).await;
            }
            let _ = tx.send(StreamEvent::End).await;
        });

        Ok((rx, handle))
    }
}

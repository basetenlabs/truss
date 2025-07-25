use crate::errors::ClientError;
use eventsource_client::{ClientBuilder, Client};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use futures::StreamExt;
use std::sync::Arc;

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

type HttpsConnector = hyper_rustls::HttpsConnector<hyper::client::HttpConnector>;

pub struct SSEClient {
    api_key: String,
    base_url: String,
    http_client: Arc<hyper::Client<HttpsConnector>>,
}

impl SSEClient {
    pub fn new(api_key: String, base_url: String) -> Self {
        // Create optimized hyper client with connection pooling
        let connector = hyper_rustls::HttpsConnectorBuilder::new()
            .with_native_roots()
            .https_or_http()
            .enable_http1()
            .enable_http2()
            .build();


        let http_client = Arc::new(
            hyper::Client::builder()
                .pool_max_idle_per_host(512) // Allow connection reuse
                .pool_idle_timeout(std::time::Duration::from_secs(240))
                .build(connector),
        );

        Self {
            api_key,
            base_url,
            http_client,
        }
    }

    /// Start streaming SSE events from the given endpoint with a POST request
    pub fn stream(
        &self,
        endpoint: &str,
        payload: Value,
        method: String,
    ) -> Result<(mpsc::Receiver<StreamEvent>, JoinHandle<()>), ClientError> {
        let (tx, rx) = mpsc::channel::<StreamEvent>(32); // Reduced buffer size for better performance

        let base = self.base_url.trim_end_matches('/');
        let endpoint = endpoint.trim_start_matches('/');
        let full_url = format!("{}/{}", base, endpoint);

        if !full_url.starts_with("http://") && !full_url.starts_with("https://") {
            return Err(ClientError::InvalidParameter(format!("Invalid URL format: {} (base: {}, endpoint: {})", full_url, base, endpoint)));
        }

        // Clone data for the async task
        let api_key = self.api_key.clone();
        let http_client = Arc::clone(&self.http_client);
        let payload_json = serde_json::to_string(&payload)
            .map_err(|e| ClientError::InvalidParameter(format!("Failed to serialize payload: {}", e)))?;

        // Pre-create auth header to avoid string formatting in async task
        let auth_header = format!("Bearer {}", api_key);

        // Spawn optimized async task using shared HTTP client for connection reuse
        let handle = tokio::spawn(async move {
            let result = async {
                // Build client with shared HTTP client for connection pooling
                let client = ClientBuilder::for_url(&full_url)
                    .map_err(|e| ClientError::Network(format!("Failed to create SSE client: {}", e)))?
                    .method(method)
                    .body(payload_json)
                    .header("Authorization", &auth_header)
                    .map_err(|e| ClientError::Network(format!("Failed to set authorization header: {}", e)))?
                    .header("Content-Type", "application/json")
                    .map_err(|e| ClientError::Network(format!("Failed to set content-type header: {}", e)))?
                    .build_with_http_client((*http_client).clone());

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

use crate::cancellation::JoinSetGuard;
use crate::constants::*;
use crate::customer_request_id::CustomerRequestId;
use crate::errors::{convert_reqwest_error_with_customer_id, ClientError};
use rand::Rng;
use reqwest::Client;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub struct SendRequestConfig {
    pub max_retries: u32,
    pub initial_backoff: Duration,
    pub retry_budget: Arc<AtomicUsize>,
    pub hedge_budget: Option<(Arc<AtomicUsize>, Duration)>,
    pub timeout: Duration,
    pub customer_request_id: CustomerRequestId,
}

impl SendRequestConfig {
    /// Create a new SendRequestConfig with customer request ID
    pub fn new(
        max_retries: u32,
        initial_backoff: Duration,
        retry_budget: Arc<AtomicUsize>,
        hedge_budget: Option<(Arc<AtomicUsize>, Duration)>,
        timeout: Duration,
        customer_request_id: CustomerRequestId,
    ) -> Result<Self, ClientError> {
        // Validate that hedging timeout is higher than request timeout
        if let Some((_, hedge_timeout)) = &hedge_budget {
            if hedge_timeout >= &timeout {
                return Err(ClientError::InvalidParameter(format!(
                    "Hedge timeout ({:.3}s) must be higher than request timeout ({:.3}s)",
                    hedge_timeout.as_secs_f64(),
                    timeout.as_secs_f64()
                )));
            }
        }

        Ok(SendRequestConfig {
            max_retries,
            initial_backoff,
            retry_budget,
            hedge_budget,
            timeout,
            customer_request_id,
        })
    }
}

// Unified HTTP request helper
pub async fn send_http_request_with_retry<T, R>(
    client: &Client,
    url: String,
    payload: T,
    api_key: String,
    request_timeout: Duration,
    config: &SendRequestConfig,
) -> Result<(R, std::collections::HashMap<String, String>), ClientError>
where
    T: serde::Serialize,
    R: serde::de::DeserializeOwned,
{
    let mut request_builder = client
        .post(&url)
        .bearer_auth(api_key)
        .json(&payload)
        .timeout(request_timeout);

    // Add customer request ID header
    request_builder =
        request_builder.header(CUSTOMER_HEADER_NAME, config.customer_request_id.to_string());

    let response = send_request_with_retry(request_builder, config).await?;
    let successful_response =
        ensure_successful_response(response, Some(config.customer_request_id.to_string())).await?;

    // Extract headers
    let mut headers_map = std::collections::HashMap::new();
    for (name, value) in successful_response.headers().iter() {
        headers_map.insert(
            name.as_str().to_string(),
            String::from_utf8_lossy(value.as_bytes()).into_owned(),
        );
    }

    let response_data: R = successful_response
        .json::<R>()
        .await
        .map_err(|e| ClientError::Serialization(format!("Failed to parse response JSON: {}", e)))?;

    Ok((response_data, headers_map))
}

// Unified HTTP request helper with headers extraction
pub async fn send_http_request_with_headers<T>(
    client: &Client,
    url: String,
    payload: T,
    api_key: String,
    request_timeout: Duration,
    config: &SendRequestConfig,
    custom_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<(serde_json::Value, std::collections::HashMap<String, String>), ClientError>
where
    T: serde::Serialize,
{
    let mut request_builder = client
        .post(&url)
        .bearer_auth(api_key)
        .json(&payload)
        .timeout(request_timeout);

    // Add customer request ID header
    request_builder =
        request_builder.header(CUSTOMER_HEADER_NAME, config.customer_request_id.to_string());

    if let Some(headers) = custom_headers {
        for (key, value) in headers {
            request_builder = request_builder.header(key, value);
        }
    }

    let response = send_request_with_retry(request_builder, config).await?;
    // hedge here with a second workstream, awaiting the first one or time of hedge
    let successful_response =
        ensure_successful_response(response, Some(config.customer_request_id.to_string())).await?;

    // Extract headers
    let mut headers_map = std::collections::HashMap::new();
    for (name, value) in successful_response.headers().iter() {
        headers_map.insert(
            name.as_str().to_string(),
            String::from_utf8_lossy(value.as_bytes()).into_owned(),
        );
    }

    let response_json_value: serde_json::Value = successful_response
        .json::<serde_json::Value>()
        .await
        .map_err(|e| ClientError::Serialization(format!("Failed to parse response JSON: {}", e)))?;

    Ok((response_json_value, headers_map))
}

async fn ensure_successful_response(
    response: reqwest::Response,
    customer_request_id: Option<String>,
) -> Result<reqwest::Response, ClientError> {
    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        Err(ClientError::Http {
            status: status.as_u16(),
            message: format!("API request failed with status {}: {}", status, error_text),
            customer_request_id,
        })
    } else {
        Ok(response)
    }
}

async fn send_request_with_retry(
    request_builder: reqwest::RequestBuilder,
    config: &SendRequestConfig,
) -> Result<reqwest::Response, ClientError> {
    let mut retries_done = 0;
    let mut current_backoff = config.initial_backoff;
    let max_retries = config.max_retries;

    loop {
        let request_builder_clone = request_builder.try_clone().ok_or_else(|| {
            ClientError::Network("Failed to clone request builder for retry".to_string())
        })?;

        // Only hedge on the first request (retries_done <= 1)
        let should_hedge = retries_done <= 1 && config.hedge_budget.is_some();

        let response_result = if should_hedge {
            send_request_with_hedging(request_builder_clone, config).await
        } else {
            request_builder_clone.send().await.map_err(|e| {
                convert_reqwest_error_with_customer_id(e, config.customer_request_id.clone())
            })
        };

        let should_retry = match &response_result {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    return Ok(response_result.unwrap());
                }

                let is_retryable_status = status.as_u16() == 429 || status.is_server_error();
                if is_retryable_status {
                    true
                } else {
                    // For non-retryable client errors (e.g., 400), cancel other requests and propagate the error.
                    false
                }
            }
            Err(client_error) => {
                // For network errors, check if we have a retry budget.
                match client_error {
                    ClientError::LocalTimeout(_, _) => {
                        config.retry_budget.fetch_sub(1, Ordering::SeqCst) > 0
                    }
                    ClientError::RemoteTimeout(_, _) => {
                        config.retry_budget.fetch_sub(1, Ordering::SeqCst) > 0
                    }
                    // connect can happen if e.g. number of tcp streams in linux is exhausted.
                    ClientError::Connect(_) => retries_done <= 1,
                    ClientError::Network(_) => {
                        if retries_done == 0 {
                            true
                        } else {
                            config.retry_budget.fetch_sub(1, Ordering::SeqCst) > 0
                        }
                    }
                    _ => {
                        // For other errors, we do not retry.
                        eprintln!(
                            "unexpected client error, no retry: this should not happen: {}",
                            client_error
                        );
                        false
                    }
                }
            }
        };
        let should_retry = should_retry && retries_done < max_retries;

        if !should_retry {
            return match response_result {
                Ok(resp) => {
                    ensure_successful_response(resp, Some(config.customer_request_id.to_string()))
                        .await
                }
                Err(err) => Err(err),
            };
        }

        retries_done += 1;
        let jitter = rand::rng().random_range(0..100);
        let backoff_duration =
            current_backoff.min(MAX_BACKOFF_DURATION) + Duration::from_millis(jitter);
        tokio::time::sleep(backoff_duration).await;
        current_backoff = current_backoff.saturating_mul(4);
    }
}

pub async fn send_request_with_hedging(
    request_builder: reqwest::RequestBuilder,
    config: &SendRequestConfig,
) -> Result<reqwest::Response, ClientError> {
    let (hedge_budget, hedge_delay) = config.hedge_budget.as_ref().unwrap();

    // Check if we have hedge budget available
    if hedge_budget.load(Ordering::SeqCst) == 0 {
        // No hedge budget, use normal request
        return request_builder.send().await.map_err(ClientError::from);
    }

    let request_builder_hedge = request_builder.try_clone().ok_or_else(|| {
        ClientError::Network("Failed to clone request builder for hedging".to_string())
    })?;

    // Use JoinSetGuard to ensure all spawned tasks are aborted on drop
    let mut join_set: JoinSetGuard<Result<reqwest::Response, ClientError>> = JoinSetGuard::new();

    // Start the original request
    join_set.spawn(async move { request_builder.send().await.map_err(ClientError::from) });

    // Wait for hedge delay
    let hedge_timer = tokio::time::sleep(*hedge_delay);

    tokio::select! {
        biased;

        // Original request completed before hedge delay
        result = join_set.join_next() => {
            match result {
                Some(Ok(response_result)) => response_result,
                Some(Err(join_err)) => Err(ClientError::Network(format!("Original request task failed: {}", join_err))),
                None => Err(ClientError::Network("No task result available".to_string())),
            }
        }
        // Hedge delay expired, start hedged request
        _ = hedge_timer => {
            // Decrement hedge budget
            if hedge_budget.fetch_sub(1, Ordering::SeqCst) > 0 {
                join_set.spawn(async move {
                    request_builder_hedge.send().await.map_err(ClientError::from)
                });

                // Race between original and hedged request - first to complete wins
                match join_set.join_next().await {
                    Some(Ok(response_result)) => response_result,
                    Some(Err(join_err)) => Err(ClientError::Network(format!("Request task failed: {}", join_err))),
                    None => Err(ClientError::Network("No task result available".to_string())),
                }
                // JoinSetGuard will abort the remaining task on drop
            } else {
                // No hedge budget left, wait for original request
                match join_set.join_next().await {
                    Some(Ok(response_result)) => response_result,
                    Some(Err(join_err)) => Err(ClientError::Network(format!("Original request task failed: {}", join_err))),
                    None => Err(ClientError::Network("No task result available".to_string())),
                }
            }
        }
    }
    // JoinSetGuard drops here, aborting any remaining tasks
}

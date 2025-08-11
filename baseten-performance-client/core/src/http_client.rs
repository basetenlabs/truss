use crate::constants::*;
use crate::errors::ClientError;
use rand::Rng;
use reqwest::Client;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub struct SendRequestConfig {
    pub max_retries: u32,
    pub initial_backoff: Duration,
    pub retry_budget: Arc<AtomicUsize>,
    pub cancel_token: Arc<AtomicBool>,
    pub hedge_budget: Option<(Arc<AtomicUsize>, Duration)>,
    pub timeout: Duration,
}

impl SendRequestConfig {
    /// Create a new SendRequestConfig with validation
    pub fn new(
        max_retries: u32,
        initial_backoff: Duration,
        retry_budget: Arc<AtomicUsize>,
        cancel_token: Arc<AtomicBool>,
        hedge_budget: Option<(Arc<AtomicUsize>, Duration)>,
        timeout: Duration,
    ) -> Result<Self, ClientError> {
        // Validate that hedging timeout is higher than request timeout
        if let Some((_, hedge_timeout)) = &hedge_budget {
            if hedge_timeout <= &timeout {
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
            cancel_token,
            hedge_budget,
            timeout,
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
) -> Result<R, ClientError>
where
    T: serde::Serialize,
    R: serde::de::DeserializeOwned,
{
    let request_builder = client
        .post(&url)
        .bearer_auth(api_key)
        .json(&payload)
        .timeout(request_timeout);

    // let response = send_request_with_retry(request_builder, config).await?;
    // TODO race tokio tasks: sleep(hedge_delay) + request_with_retry + request, get the first one.
    let response = send_request_with_retry(request_builder, config).await?;

    let successful_response = ensure_successful_response(response).await?;

    successful_response
        .json::<R>()
        .await
        .map_err(|e| ClientError::Serialization(format!("Failed to parse response JSON: {}", e)))
}

// Unified HTTP request helper with headers extraction
pub async fn send_http_request_with_headers<T>(
    client: &Client,
    url: String,
    payload: T,
    api_key: String,
    request_timeout: Duration,
    config: &SendRequestConfig,
) -> Result<(serde_json::Value, std::collections::HashMap<String, String>), ClientError>
where
    T: serde::Serialize,
{
    let request_builder = client
        .post(&url)
        .bearer_auth(api_key)
        .json(&payload)
        .timeout(request_timeout);

    let response = send_request_with_retry(request_builder, config).await?;
    // hedge here with a second workstream, awaiting the first one or time of hedge
    let successful_response = ensure_successful_response(response).await?;

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

pub async fn ensure_successful_response(
    response: reqwest::Response,
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
        })
    } else {
        Ok(response)
    }
}

pub async fn send_request_with_retry(
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
            request_builder_clone
                .send()
                .await
                .map_err(ClientError::from)
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
                    config.cancel_token.store(true, Ordering::SeqCst);
                    false
                }
            }
            Err(client_error) => {
                // For network errors, check if we have a retry budget.
                match client_error {
                    ClientError::Timeout(_) => {
                        println!("client timeout error: {}", client_error);
                        config.retry_budget.fetch_sub(1, Ordering::SeqCst) > 0
                    }
                    // connect can happen if e.g. number of tcp streams in linux is exhausted.
                    ClientError::Connect(_) => retries_done <= 1,
                    ClientError::Network(_) => {
                        if retries_done == 0 {
                            true
                        } else {
                            println!("client network error (likely re-use expired http connection): {}", client_error);
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
        let should_retry = should_retry
            && !config.cancel_token.load(Ordering::SeqCst)
            && retries_done < max_retries;

        if !should_retry {
            return match response_result {
                Ok(resp) => ensure_successful_response(resp).await,
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

    // Start the original request
    let mut original_request =
        tokio::spawn(async move { request_builder.send().await.map_err(ClientError::from) });

    // Wait for hedge delay
    let hedge_timer = tokio::time::sleep(*hedge_delay);

    tokio::select! {
        // Original request completed before hedge delay
        result = &mut original_request => {
            match result {
                Ok(response_result) => response_result.map_err(ClientError::from),
                Err(join_err) => Err(ClientError::Network(format!("Original request task failed: {}", join_err))),
            }
        }
        // Hedge delay expired, start hedged request
        _ = hedge_timer => {
            // Decrement hedge budget
            if hedge_budget.fetch_sub(1, Ordering::SeqCst) > 0 {
                let mut hedge_request = tokio::spawn(async move {
                    request_builder_hedge.send().await
                });

                // Race between original and hedged request
                tokio::select! {
                    result = &mut original_request => {
                        // Cancel hedge request if original completes first
                        hedge_request.abort();
                        match result {
                            Ok(response_result) => response_result.map_err(ClientError::from),
                            Err(join_err) => Err(ClientError::Network(format!("Original request task failed: {}", join_err))),
                        }
                    }
                    result = &mut hedge_request => {
                        // Hedge request completed first
                        original_request.abort();
                        println!("hedged request completed first, original request cancelled");
                        match result {
                            Ok(response_result) => response_result.map_err(ClientError::from),
                            Err(join_err) => Err(ClientError::Network(format!("Hedge request task failed: {}", join_err))),
                        }
                    }
                }
            } else {
                // No hedge budget left, wait for original request
                match original_request.await {
                    Ok(response_result) => response_result.map_err(ClientError::from),
                    Err(join_err) => Err(ClientError::Network(format!("Original request task failed: {}", join_err))),
                }
            }
        }
    }
}

use crate::cancellation::JoinSetGuard;
use crate::constants::*;
use crate::customer_request_id::CustomerRequestId;
use crate::errors::{convert_reqwest_error_with_customer_id, ClientError};
use crate::utils::{calculate_hedge_budget, calculate_retry_timeout_budget};
use rand::Rng;
use reqwest::Client;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tracing;

/// Shared budgets for retry and hedging operations
#[derive(Debug, Clone)]
pub struct SharedBudgets {
    pub retry_budget: Arc<AtomicUsize>,
    pub hedge_budget: Option<Arc<AtomicUsize>>,
}

impl SharedBudgets {
    pub fn new(total_requests: usize, hedge_delay: Option<f64>) -> Self {
        let retry_budget = Arc::new(AtomicUsize::new(calculate_retry_timeout_budget(
            total_requests,
        )));

        let hedge_budget = hedge_delay.filter(|&delay| delay >= 0.2).map(|_delay| {
            let budget = calculate_hedge_budget(total_requests);
            tracing::debug!(
                "Creating hedge budget with {} requests, budget: {}",
                total_requests,
                budget
            );
            Arc::new(AtomicUsize::new(budget))
        });

        Self {
            retry_budget,
            hedge_budget,
        }
    }
}

pub struct SendRequestConfig {
    pub max_retries: u32,
    pub initial_backoff: Duration,
    pub budgets: SharedBudgets,
    pub hedge_delay: Option<Duration>,
    pub timeout: Duration,
    pub customer_request_id: CustomerRequestId,
}

impl SendRequestConfig {
    pub fn new(
        max_retries: u32,
        initial_backoff: Duration,
        budgets: SharedBudgets,
        hedge_delay: Option<Duration>,
        timeout: Duration,
        customer_request_id: CustomerRequestId,
    ) -> Result<Self, ClientError> {
        if let Some(hedge_timeout) = &hedge_delay {
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
            budgets,
            hedge_delay,
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
        let should_hedge = retries_done <= 1 && config.budgets.hedge_budget.is_some();

        if should_hedge {
            tracing::info!(
                "Hedging request - retries_done: {}, hedge_budget_available: {}",
                retries_done,
                config.budgets.hedge_budget.is_some()
            );
        }

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
                        let remaining_budget =
                            config.budgets.retry_budget.fetch_sub(1, Ordering::SeqCst);
                        tracing::debug!(
                            "Local timeout encountered, retrying... Remaining retry budget: {} {}",
                            remaining_budget,
                            config.customer_request_id.to_string()
                        );
                        remaining_budget > 0
                    }
                    ClientError::RemoteTimeout(_, _) => {
                        let remaining_budget =
                            config.budgets.retry_budget.fetch_sub(1, Ordering::SeqCst);
                        tracing::debug!(
                            "Remote timeout encountered, retrying... Remaining retry budget: {} {}",
                            remaining_budget,
                            config.customer_request_id.to_string()
                        );
                        remaining_budget > 0
                    }
                    // connect can happen if e.g. number of tcp streams in linux is exhausted.
                    ClientError::Connect(_) => retries_done <= 1,
                    ClientError::Network(_) => {
                        if retries_done == 0 {
                            true
                        } else {
                            let remaining_budget =
                                config.budgets.retry_budget.fetch_sub(1, Ordering::SeqCst);
                            tracing::debug!("Network error encountered, retrying... Remaining retry budget: {} {}", remaining_budget, config.customer_request_id.to_string());
                            remaining_budget > 0
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
    // Validate that we have both hedge budget and hedge delay
    let hedge_budget = config.budgets.hedge_budget.as_ref().ok_or_else(|| {
        tracing::warn!("Unreachable: Hedge budget not available for hedging");
        ClientError::InvalidParameter("Hedge budget not available for hedging".to_string())
    })?;
    let hedge_delay = config.hedge_delay.ok_or_else(|| {
        tracing::warn!("Unreachable: Hedge delay not configured for hedging");
        ClientError::InvalidParameter("Hedge delay not configured for hedging".to_string())
    })?;

    // Check if we have hedge budget available
    if hedge_budget.load(Ordering::SeqCst) == 0 {
        tracing::debug!("No hedge budget available, using normal request");
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
    let hedge_timer = tokio::time::sleep(hedge_delay);

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
            // Decrement hedge budget and check if we had budget available
            let budget_before_decrement = hedge_budget.fetch_sub(1, Ordering::SeqCst);
            tracing::debug!("Hedge budget decremented from {} to {}", budget_before_decrement, budget_before_decrement.saturating_sub(1));

            // Allow hedging if we had budget before decrement (budget was > 0)
            if budget_before_decrement > 0 {
                join_set.spawn(async move {
                    let result = request_builder_hedge.send().await.map_err(ClientError::from);
                    tracing::debug!("hedged request faster than original");
                    result
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

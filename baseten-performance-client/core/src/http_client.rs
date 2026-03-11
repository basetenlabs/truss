use crate::cancellation::JoinSetGuard;
use crate::constants::*;
use crate::customer_request_id::CustomerRequestId;
use crate::errors::{convert_reqwest_error_with_customer_id, ClientError};
use crate::split_policy::RequestProcessingConfig;

use rand::Rng;
use reqwest::Client;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tracing;

// Unified HTTP request helper
pub(crate) async fn send_http_request_with_retry<T, R>(
    client: &Client,
    url: String,
    payload: T,
    api_key: String,
    request_timeout: Duration,
    config: &RequestProcessingConfig,
    customer_request_id: CustomerRequestId,
) -> Result<(R, std::collections::HashMap<String, String>), ClientError>
where
    T: serde::Serialize,
    R: serde::de::DeserializeOwned,
{
    let customer_request_id_header = customer_request_id.to_string();
    let response = send_request_with_retry(&url, config, |attempt_url| {
        let mut request_builder = client
            .post(attempt_url)
            .bearer_auth(&api_key)
            .json(&payload)
            .timeout(request_timeout)
            .header(CUSTOMER_HEADER_NAME, &customer_request_id_header);

        if let Some(ref headers) = config.extra_headers {
            for (key, value) in headers {
                request_builder = request_builder.header(key, value);
            }
        }

        request_builder
    })
    .await?;

    let successful_response =
        ensure_successful_response(response, Some(customer_request_id.to_string())).await?;

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
#[allow(clippy::too_many_arguments)]
pub(crate) async fn send_http_request_with_headers<T>(
    client: &Client,
    url: String,
    payload: T,
    api_key: String,
    request_timeout: Duration,
    config: &RequestProcessingConfig,
    customer_request_id: CustomerRequestId,
    method: crate::http::HttpMethod,
) -> Result<(serde_json::Value, std::collections::HashMap<String, String>), ClientError>
where
    T: serde::Serialize,
{
    let customer_request_id_header = customer_request_id.to_string();
    let response = send_request_with_retry(&url, config, |attempt_url| {
        let mut request_builder = client
            .request(method.into(), attempt_url)
            .bearer_auth(&api_key)
            .timeout(request_timeout)
            .header(CUSTOMER_HEADER_NAME, &customer_request_id_header);

        if method.has_body() {
            request_builder = request_builder.json(&payload);
        }

        if let Some(ref headers) = config.extra_headers {
            for (key, value) in headers {
                request_builder = request_builder.header(key, value);
            }
        }

        request_builder
    })
    .await?;

    let successful_response =
        ensure_successful_response(response, Some(customer_request_id.to_string())).await?;

    // Extract headers
    let mut headers_map = std::collections::HashMap::new();
    for (name, value) in successful_response.headers().iter() {
        headers_map.insert(
            name.as_str().to_string(),
            String::from_utf8_lossy(value.as_bytes()).into_owned(),
        );
    }

    let response_json_value: serde_json::Value =
        if method.has_body() || matches!(method, crate::http::HttpMethod::GET) {
            successful_response
                .json::<serde_json::Value>()
                .await
                .map_err(|e| {
                    ClientError::Serialization(format!("Failed to parse response JSON: {}", e))
                })?
        } else {
            serde_json::Value::Object(serde_json::Map::new())
        };

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
    original_url: &str,
    config: &RequestProcessingConfig,
    build_request: impl Fn(&str) -> reqwest::RequestBuilder,
) -> Result<reqwest::Response, ClientError> {
    let mut retries_done = 0;
    let mut current_backoff = config.initial_backoff;
    let max_retries = config.max_retries;
    let mut attempted_endpoint_indices: Vec<usize> = Vec::new();

    loop {
        let (attempt_url, selected_endpoint_index) =
            config.select_attempt_url(original_url, &attempted_endpoint_indices);
        if !attempted_endpoint_indices.contains(&selected_endpoint_index) {
            attempted_endpoint_indices.push(selected_endpoint_index);
        }

        // Only hedge on the first request (retries_done <= 1)
        let should_hedge = retries_done <= 1
            && config.hedge_delay.is_some()
            && config.hedge_budget.load(Ordering::SeqCst) > 0;

        if should_hedge {
            tracing::info!(
                "Hedging request - retries_done: {}, hedge_budget_available: {}",
                retries_done,
                config.hedge_budget.load(Ordering::SeqCst)
            );
        }

        let response_result: Result<reqwest::Response, ClientError> = if should_hedge {
            let request_builder = build_request(&attempt_url);
            let hedge_url = config.select_hedge_url(original_url, selected_endpoint_index);
            let hedge_builder = build_request(&hedge_url);
            send_request_with_hedging(request_builder, hedge_builder, config).await
        } else {
            build_request(&attempt_url).send().await.map_err(|e| {
                convert_reqwest_error_with_customer_id(e, config.customer_request_id.clone())
            })
        };

        // Decide retry exactly once per iteration.
        let should_retry_iteration = match response_result {
            Ok(resp) => {
                let status = resp.status();

                if status.is_success() {
                    return Ok(resp);
                }

                let retryable = is_retryable_status(status.as_u16());
                let should_retry = retryable && retries_done < max_retries;

                if !should_retry {
                    return ensure_successful_response(
                        resp,
                        Some(config.customer_request_id.to_string()),
                    )
                    .await;
                }

                // Retryable status: drain the body so the connection can be reused.
                let _ = resp.bytes().await;
                true
            }

            Err(client_error) => {
                let should_retry = match &client_error {
                    ClientError::LocalTimeout(_, _) => {
                        let remaining_budget = config.retry_budget.fetch_sub(1, Ordering::SeqCst);
                        tracing::debug!(
                            "Local timeout encountered, retrying... Remaining retry budget: {} {}",
                            remaining_budget,
                            config.customer_request_id.to_string()
                        );
                        remaining_budget > 0
                    }
                    ClientError::RemoteTimeout(_, _) => {
                        let remaining_budget = config.retry_budget.fetch_sub(1, Ordering::SeqCst);
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
                                config.retry_budget.fetch_sub(1, Ordering::SeqCst);
                            tracing::debug!(
                                "Network error encountered, retrying... Remaining retry budget: {} {}",
                                remaining_budget,
                                config.customer_request_id.to_string()
                            );
                            remaining_budget > 0
                        }
                    }
                    _ => {
                        tracing::warn!(
                            "unexpected client error, no retry: this should not happen: {}",
                            client_error
                        );
                        false
                    }
                } && retries_done < max_retries;

                if !should_retry {
                    return Err(client_error);
                }

                true
            }
        };

        if !should_retry_iteration {
            unreachable!("retry loop must return before reaching this branch");
        }

        // If we got here, we are retrying this iteration.
        retries_done += 1;
        let jitter = rand::rng().random_range(0..100);
        let backoff_duration =
            current_backoff.min(MAX_BACKOFF_DURATION) + Duration::from_millis(jitter);
        tokio::time::sleep(backoff_duration).await;
        current_backoff = current_backoff.saturating_mul(4);
    }
}

pub(crate) async fn send_request_with_hedging(
    request_builder: reqwest::RequestBuilder,
    request_builder_hedge: reqwest::RequestBuilder,
    config: &RequestProcessingConfig,
) -> Result<reqwest::Response, ClientError> {
    // Validate that we have hedge budget and hedge delay
    let hedge_budget = &config.hedge_budget;
    let hedge_delay = config.hedge_delay.ok_or_else(|| {
        tracing::warn!("Unreachable: Hedge delay not configured for hedging");
        ClientError::InvalidParameter("Hedge delay not configured for hedging".to_string())
    })?;

    // Check if we have hedge budget available
    if hedge_budget.load(Ordering::SeqCst) == 0 {
        tracing::debug!("No hedge budget available, using normal request");
        return request_builder.send().await.map_err(ClientError::from);
    }

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

/// Determine if an HTTP status code is retryable
fn is_retryable_status(status: u16) -> bool {
    match status {
        // Rate limiting
        429 => true,
        // Server errors (5xx)
        500..=599 => true,
        // Request timeout
        408 => true,
        // Some client errors that might be transient
        409 => true,  // Conflict
        422 => false, // Unprocessable Entity - not retryable
        423 => false, // Locked - not retryable
        425 => false, // Too Early - not retryable
        // All other client errors (4xx except above) - not retryable
        400..=499 => false,
        // Unexpected status codes
        _ => false,
    }
}

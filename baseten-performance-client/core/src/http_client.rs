use crate::cancellation::JoinSetGuard;
use crate::client_spans::{self, RecordingSpan};
use crate::constants::*;
use crate::customer_request_id::CustomerRequestId;
use crate::errors::{convert_reqwest_error_with_customer_id, ClientError};
use crate::split_policy::RequestProcessingConfig;
use crate::trace_context::{TRACEPARENT_HEADER_NAME, TRACESTATE_HEADER_NAME};

use rand::Rng;
use reqwest::{
    header::{ACCEPT, ACCEPT_ENCODING, CONTENT_TYPE},
    Client,
};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing;

fn add_timeout_headers(
    request_builder: reqwest::RequestBuilder,
    request_timeout: Duration,
) -> reqwest::RequestBuilder {
    let timeout_ms = ((request_timeout.as_secs_f64() * 1000.0).ceil() as u64).to_string();
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let deadline_ms = (now_ms + request_timeout.as_millis() as u64).to_string();
    request_builder
        .header(REQUEST_TIMEOUT_HEADER_NAME, timeout_ms)
        .header(REQUEST_DEADLINE_HEADER_NAME, deadline_ms)
}

// Unified HTTP request helper
pub(crate) async fn send_http_request_with_retry<T, R>(
    client: &Client,
    request_suffix: String,
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
    let labels = AttemptLabels {
        method: reqwest::Method::POST.as_str(),
        customer_request_id: &customer_request_id_header,
    };
    let (response, span) =
        send_request_with_retry(&request_suffix, config, &labels, |attempt_url| {
            let mut request_builder = client
                .post(attempt_url)
                .bearer_auth(&api_key)
                .json(&payload)
                .timeout(request_timeout)
                .header(CUSTOMER_HEADER_NAME, &customer_request_id_header);

            request_builder = add_timeout_headers(request_builder, request_timeout);
            request_builder = add_response_negotiation_headers(request_builder, config);

            if let Some(ref headers) = config.extra_headers {
                for (key, value) in headers {
                    request_builder = request_builder.header(key, value);
                }
            }

            request_builder
        })
        .await?;

    let successful_response =
        match ensure_successful_response(response, Some(customer_request_id.to_string())).await {
            Ok(response) => response,
            Err(err) => {
                client_spans::fail_span(span, client_spans::error_type(&err));
                return Err(err);
            }
        };

    // Extract headers
    let mut headers_map = std::collections::HashMap::new();
    for (name, value) in successful_response.headers().iter() {
        headers_map.insert(
            name.as_str().to_string(),
            String::from_utf8_lossy(value.as_bytes()).into_owned(),
        );
    }

    let response_data: Result<R, ClientError> = parse_response_body(successful_response).await;
    client_spans::finish_span(span, &response_data);

    Ok((response_data?, headers_map))
}

// Unified HTTP request helper with headers extraction
#[allow(clippy::too_many_arguments)]
pub(crate) async fn send_http_request_with_headers<T>(
    client: &Client,
    request_suffix: String,
    payload: T,
    api_key: String,
    request_timeout: Duration,
    config: &RequestProcessingConfig,
    customer_request_id: CustomerRequestId,
    method: crate::http::HttpMethod,
) -> Result<(rmpv::Value, std::collections::HashMap<String, String>), ClientError>
where
    T: serde::Serialize,
{
    let customer_request_id_header = customer_request_id.to_string();
    let reqwest_method = reqwest::Method::from(method);
    let labels = AttemptLabels {
        method: reqwest_method.as_str(),
        customer_request_id: &customer_request_id_header,
    };
    let (response, span) =
        send_request_with_retry(&request_suffix, config, &labels, |attempt_url| {
            let mut request_builder = client
                .request(reqwest_method.clone(), attempt_url)
                .bearer_auth(&api_key)
                .timeout(request_timeout)
                .header(CUSTOMER_HEADER_NAME, &customer_request_id_header);

            request_builder = add_timeout_headers(request_builder, request_timeout);

            request_builder = add_response_negotiation_headers(request_builder, config);

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
        match ensure_successful_response(response, Some(customer_request_id.to_string())).await {
            Ok(response) => response,
            Err(err) => {
                client_spans::fail_span(span, client_spans::error_type(&err));
                return Err(err);
            }
        };

    // Extract headers
    let mut headers_map = std::collections::HashMap::new();
    for (name, value) in successful_response.headers().iter() {
        headers_map.insert(
            name.as_str().to_string(),
            String::from_utf8_lossy(value.as_bytes()).into_owned(),
        );
    }

    let response_value: Result<rmpv::Value, ClientError> =
        if method.has_body() || matches!(method, crate::http::HttpMethod::GET) {
            parse_response_body(successful_response).await
        } else {
            Ok(rmpv::Value::Map(Vec::new()))
        };
    client_spans::finish_span(span, &response_value);

    Ok((response_value?, headers_map))
}

fn add_response_negotiation_headers(
    mut request_builder: reqwest::RequestBuilder,
    config: &RequestProcessingConfig,
) -> reqwest::RequestBuilder {
    if !extra_headers_contains(config, ACCEPT.as_str()) {
        request_builder = request_builder.header(ACCEPT, "application/json, application/msgpack");
    }

    if !extra_headers_contains(config, ACCEPT_ENCODING.as_str()) {
        request_builder = request_builder.header(ACCEPT_ENCODING, "zstd");
    }

    request_builder
}

fn extra_headers_contains(config: &RequestProcessingConfig, header_name: &str) -> bool {
    config.extra_headers.as_ref().is_some_and(|headers| {
        headers
            .keys()
            .any(|key| key.eq_ignore_ascii_case(header_name))
    })
}

async fn parse_response_body<R>(response: reqwest::Response) -> Result<R, ClientError>
where
    R: serde::de::DeserializeOwned,
{
    if is_msgpack_response(&response) {
        let bytes = response.bytes().await.map_err(|e| {
            ClientError::Serialization(format!("Failed to read MessagePack response body: {}", e))
        })?;
        return rmp_serde::from_slice::<R>(&bytes).map_err(|e| {
            ClientError::Serialization(format!("Failed to parse response MessagePack: {}", e))
        });
    }

    response
        .json::<R>()
        .await
        .map_err(|e| ClientError::Serialization(format!("Failed to parse response JSON: {}", e)))
}

fn is_msgpack_response(response: &reqwest::Response) -> bool {
    response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(is_msgpack_content_type)
}

fn is_msgpack_content_type(content_type: &str) -> bool {
    matches!(
        content_type
            .split(';')
            .next()
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "application/msgpack"
            | "application/x-msgpack"
            | "application/messagepack"
            | "application/vnd.msgpack"
    )
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

/// Identifies an attempt on its span.
struct AttemptLabels<'a> {
    method: &'a str,
    customer_request_id: &'a str,
}

/// Builds one attempt's request along with its span (when recording) and `traceparent` header.
fn build_attempt(
    build_request: &impl Fn(&str) -> reqwest::RequestBuilder,
    url: &str,
    config: &RequestProcessingConfig,
    labels: &AttemptLabels<'_>,
    resend_count: u32,
    hedge: bool,
) -> (reqwest::RequestBuilder, Option<RecordingSpan>) {
    let mut span = client_spans::start_attempt_span(config.call_trace.as_ref(), labels.method);
    if let Some(span) = span.as_mut() {
        client_spans::record_attempt_attributes(
            span,
            labels.method,
            url,
            labels.customer_request_id,
            resend_count,
            hedge,
        );
    }

    let mut request_builder = build_request(url);
    // The config holds no trace headers in extra_headers, so these are the only ones sent.
    if let Some((traceparent, tracestate)) =
        client_spans::attempt_trace_headers(config.call_trace.as_ref(), span.as_ref())
    {
        request_builder = request_builder.header(TRACEPARENT_HEADER_NAME, traceparent.to_string());
        if let Some(tracestate) = tracestate {
            request_builder = request_builder.header(TRACESTATE_HEADER_NAME, tracestate);
        }
    }
    (request_builder, span)
}

/// Sends one attempt, marking time-to-headers on its span. The span comes back with the response,
/// still owed a finish once the body is consumed; a failed send finishes it here.
async fn send_attempt(
    request_builder: reqwest::RequestBuilder,
    mut span: Option<RecordingSpan>,
    map_err: impl FnOnce(reqwest::Error) -> ClientError,
) -> Result<(reqwest::Response, Option<RecordingSpan>), ClientError> {
    match request_builder.send().await {
        Ok(response) => {
            if let Some(span) = span.as_mut() {
                span.add_event("http.response.headers");
                span.set_attribute(
                    "http.response.status_code",
                    i64::from(response.status().as_u16()),
                );
            }
            Ok((response, span))
        }
        Err(err) => {
            let err = map_err(err);
            client_spans::fail_span(span, client_spans::error_type(&err));
            Err(err)
        }
    }
}

async fn send_request_with_retry(
    request_suffix: &str,
    config: &RequestProcessingConfig,
    labels: &AttemptLabels<'_>,
    build_request: impl Fn(&str) -> reqwest::RequestBuilder,
) -> Result<(reqwest::Response, Option<RecordingSpan>), ClientError> {
    let mut retries_done = 0;
    let mut current_backoff = config.initial_backoff;
    let max_retries = config.max_retries;
    let mut attempted_endpoint_indices: HashSet<usize> = HashSet::new();

    loop {
        let indices_vec: Vec<usize> = attempted_endpoint_indices.iter().copied().collect();
        let (attempt_url, selected_endpoint) =
            config.select_attempt_url(request_suffix, &indices_vec);
        let selected_endpoint_index = selected_endpoint.endpoint_index;
        let retry_semaphore = selected_endpoint.retry_attempt_semaphore.clone();
        attempted_endpoint_indices.insert(selected_endpoint_index);

        // Acquire a permit for the primary retry attempt before issuing the request.
        // Retries may still hedge when hedge budget is available; that behavior is
        // intentional, and this semaphore only limits the selected retry attempt.
        let maybe_retry_permit = if retries_done > 0 {
            retry_semaphore.acquire_owned().await.ok()
        } else {
            None
        };

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

        let response_result = if should_hedge {
            let primary = build_attempt(
                &build_request,
                &attempt_url,
                config,
                labels,
                retries_done,
                false,
            );
            let (hedge_url, hedge_selection) =
                config.select_hedge_url(request_suffix, selected_endpoint_index);
            let _ = hedge_selection;
            send_request_with_hedging(
                primary,
                || {
                    build_attempt(
                        &build_request,
                        &hedge_url,
                        config,
                        labels,
                        retries_done,
                        true,
                    )
                },
                config,
            )
            .await
        } else {
            let (request_builder, span) = build_attempt(
                &build_request,
                &attempt_url,
                config,
                labels,
                retries_done,
                false,
            );
            send_attempt(request_builder, span, |e| {
                convert_reqwest_error_with_customer_id(e, config.customer_request_id.clone())
            })
            .await
        };

        // Decide retry exactly once per iteration.
        let should_retry_iteration = match response_result {
            Ok((resp, span)) => {
                let status = resp.status();

                if status.is_success() {
                    return Ok((resp, span));
                }

                let retryable = is_retryable_status(status.as_u16(), config);
                let should_retry = retryable && retries_done < max_retries;

                if !should_retry {
                    let result = ensure_successful_response(
                        resp,
                        Some(config.customer_request_id.to_string()),
                    )
                    .await;
                    client_spans::finish_span(span, &result);
                    return result.map(|resp| (resp, None));
                }

                // Retryable status: drain the body so the connection can be reused.
                let _ = resp.bytes().await;
                client_spans::fail_span(span, status.as_u16().to_string());
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

        // Drop permit before backoff sleep
        drop(maybe_retry_permit);

        let jitter = rand::rng().random_range(0..100);
        let backoff_duration = current_backoff.min(Duration::from_millis(MAX_BACKOFF_MS))
            + Duration::from_millis(jitter);
        tokio::time::sleep(backoff_duration).await;
        current_backoff = current_backoff.saturating_mul(4);
    }
}

type AttemptResult = Result<(reqwest::Response, Option<RecordingSpan>), ClientError>;

fn spawn_hedged_request_cleanup(mut join_set: JoinSetGuard<AttemptResult>) {
    join_set.abort_all();
    tokio::spawn(async move {
        while let Some(result) = join_set.join_next().await {
            // A loser aborted mid-flight drops its span inside the task, which marks it
            // hedge_cancelled; one that finished before the abort is marked here.
            if let Ok(Ok((response, span))) = result {
                let _ = response.bytes().await;
                if let Some(mut span) = span {
                    span.set_attribute("b10.perfclient.hedge_cancelled", true);
                    span.finish();
                }
            }
        }
    });
}

/// Races the primary attempt against a hedge sent after `hedge_delay`. The hedge's request and
/// span are built only if it is actually sent, so an unsent hedge leaves no span behind.
pub(crate) async fn send_request_with_hedging(
    primary: (reqwest::RequestBuilder, Option<RecordingSpan>),
    build_hedge: impl FnOnce() -> (reqwest::RequestBuilder, Option<RecordingSpan>),
    config: &RequestProcessingConfig,
) -> AttemptResult {
    let (request_builder, mut primary_span) = primary;
    // Only recorded spans read the race state, so the disabled path allocates nothing for it.
    let hedge_race = primary_span.as_mut().map(|span| {
        let race = Arc::new(AtomicBool::new(false));
        span.set_hedge_race(Arc::clone(&race));
        race
    });

    // Validate that we have hedge budget and hedge delay
    let hedge_budget = &config.hedge_budget;
    let hedge_delay = config.hedge_delay.ok_or_else(|| {
        tracing::warn!("Unreachable: Hedge delay not configured for hedging");
        ClientError::InvalidParameter("Hedge delay not configured for hedging".to_string())
    })?;

    // Check if we have hedge budget available
    if hedge_budget.load(Ordering::SeqCst) == 0 {
        tracing::debug!("No hedge budget available, using normal request");
        return send_attempt(request_builder, primary_span, ClientError::from).await;
    }

    // Use JoinSetGuard to ensure all spawned tasks are aborted on drop
    let mut join_set: JoinSetGuard<AttemptResult> = JoinSetGuard::new();

    // Start the original request
    join_set.spawn(send_attempt(
        request_builder,
        primary_span,
        ClientError::from,
    ));

    // Wait for hedge delay
    let hedge_timer = tokio::time::sleep(hedge_delay);

    let response_result = tokio::select! {
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
                let (request_builder_hedge, mut hedge_span) = build_hedge();
                if let Some(race) = hedge_race.as_ref() {
                    if let Some(span) = hedge_span.as_mut() {
                        span.set_hedge_race(Arc::clone(race));
                    }
                    race.store(true, Ordering::SeqCst);
                }
                join_set.spawn(async move {
                    let result = send_attempt(request_builder_hedge, hedge_span, ClientError::from).await;
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
    };

    let mut response_result = response_result;
    if let Ok((_, Some(span))) = response_result.as_mut() {
        // The winner is finished by the caller; a later cancellation is not a lost race.
        span.leave_hedge_race();
    }

    spawn_hedged_request_cleanup(join_set);
    response_result
}

/// Determine if an HTTP status code is retryable
fn is_retryable_status(status: u16, config: &RequestProcessingConfig) -> bool {
    if config.is_explicitly_non_retryable_status(status) {
        return false;
    }

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

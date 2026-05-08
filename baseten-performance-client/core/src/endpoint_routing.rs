use crate::client::HttpClientWrapper;
use crate::errors::ClientError;

use futures::future::join_all;
use reqwest::Client;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const DEFAULT_HEALTH_CHECK_PATH: &str = "/health";
const DEFAULT_HEALTH_CHECK_INTERVAL: Duration = Duration::from_secs(10);
const DEFAULT_HEALTH_CHECK_TIMEOUT: Duration = Duration::from_secs(6);
const DEFAULT_HEALTH_CHECK_RETRIES: u32 = 2;
const HEALTH_CHECK_RETRY_DELAY: Duration = Duration::from_millis(200);
const MIN_HEALTH_CHECK_TIMEOUT: Duration = Duration::from_millis(100);
const MIN_HEALTH_CHECK_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EndpointHealthStatus {
    pub base_url: String,
    pub healthy: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EndpointPoolHealthSnapshot {
    pub endpoints: Vec<EndpointHealthStatus>,
}

#[derive(Debug, Clone)]
pub struct EndpointHealthCheckConfig {
    pub path_or_url: String,
    pub extend_base_url: bool,
    pub timeout_is_no_vote: bool,
}

impl EndpointHealthCheckConfig {
    pub fn relative(path: String) -> Self {
        Self {
            path_or_url: path,
            extend_base_url: true,
            timeout_is_no_vote: false,
        }
    }

    pub fn absolute(url: String) -> Self {
        Self {
            path_or_url: url,
            extend_base_url: false,
            timeout_is_no_vote: false,
        }
    }

    pub fn with_timeout_is_no_vote(mut self, timeout_is_no_vote: bool) -> Self {
        self.timeout_is_no_vote = timeout_is_no_vote;
        self
    }
}

#[derive(Debug, Clone)]
pub struct EndpointHealthConfig {
    pub checks: Vec<EndpointHealthCheckConfig>,
    pub fail_on_first: bool,
}

#[derive(Debug, Clone)]
pub struct EndpointPoolConfig {
    pub urls: Vec<String>,
    pub weights: Option<Vec<f64>>,
    pub endpoint_health: Option<Vec<EndpointHealthConfig>>,
    pub health_check_interval: Duration,
    pub health_check_timeout: Duration,
    pub health_check_retries: u32,
    pub health_check_client_wrapper: Arc<HttpClientWrapper>,
}

impl EndpointPoolConfig {
    pub fn new(urls: Vec<String>, health_check_client_wrapper: Arc<HttpClientWrapper>) -> Self {
        Self {
            urls,
            weights: None,
            endpoint_health: None,
            health_check_interval: DEFAULT_HEALTH_CHECK_INTERVAL,
            health_check_timeout: DEFAULT_HEALTH_CHECK_TIMEOUT,
            health_check_retries: DEFAULT_HEALTH_CHECK_RETRIES,
            health_check_client_wrapper,
        }
    }

    pub fn with_health_check_interval(mut self, interval: Duration) -> Self {
        self.health_check_interval = interval;
        self
    }

    pub fn with_health_check_timeout(mut self, timeout: Duration) -> Self {
        self.health_check_timeout = timeout;
        self
    }

    pub fn with_health_check_retries(mut self, retries: u32) -> Self {
        self.health_check_retries = retries;
        self
    }

    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    pub fn with_endpoint_health(mut self, endpoint_health: Vec<EndpointHealthConfig>) -> Self {
        self.endpoint_health = Some(endpoint_health);
        self
    }

    pub fn with_standard_health_checks(
        mut self,
        deep_health_urls: Option<Vec<String>>,
        fail_on_first: bool,
        deployment_health_path: Option<String>,
        deployment_timeout_is_no_vote: bool,
        deep_timeout_is_no_vote: bool,
    ) -> Self {
        let deployment_health_path =
            deployment_health_path.unwrap_or_else(|| DEFAULT_HEALTH_CHECK_PATH.to_string());
        let endpoint_count = self.urls.len();
        self.endpoint_health = Some(match deep_health_urls {
            Some(deep_health_urls) => deep_health_urls
                .into_iter()
                .map(|deep_url| EndpointHealthConfig {
                    checks: vec![
                        EndpointHealthCheckConfig::relative(deployment_health_path.clone())
                            .with_timeout_is_no_vote(deployment_timeout_is_no_vote),
                        EndpointHealthCheckConfig::absolute(deep_url)
                            .with_timeout_is_no_vote(deep_timeout_is_no_vote),
                    ],
                    fail_on_first,
                })
                .collect(),
            None => (0..endpoint_count)
                .map(|_| EndpointHealthConfig {
                    checks: vec![EndpointHealthCheckConfig::relative(
                        deployment_health_path.clone(),
                    )
                    .with_timeout_is_no_vote(deployment_timeout_is_no_vote)],
                    fail_on_first,
                })
                .collect(),
        });
        self
    }

    pub fn primary_url(&self) -> Option<&str> {
        self.urls.first().map(String::as_str)
    }

    fn validate(&self) -> Result<(), ClientError> {
        if self.urls.is_empty() {
            return Err(ClientError::InvalidParameter(
                "endpoint pool must contain at least one URL".to_string(),
            ));
        }

        let mut normalized_urls = Vec::with_capacity(self.urls.len());
        for url in &self.urls {
            if url.trim().is_empty() {
                return Err(ClientError::InvalidParameter(
                    "endpoint pool URLs cannot be empty".to_string(),
                ));
            }

            let normalized = normalize_url(url);
            if normalized_urls
                .iter()
                .any(|existing| existing == &normalized)
            {
                return Err(ClientError::InvalidParameter(
                    "endpoint pool URLs must be unique".to_string(),
                ));
            }
            normalized_urls.push(normalized);
        }

        if self.health_check_timeout < MIN_HEALTH_CHECK_TIMEOUT {
            return Err(ClientError::InvalidParameter(format!(
                "health_check_timeout must be at least {}ms",
                MIN_HEALTH_CHECK_TIMEOUT.as_millis()
            )));
        }

        if self.health_check_interval < MIN_HEALTH_CHECK_INTERVAL {
            return Err(ClientError::InvalidParameter(format!(
                "health_check_interval must be at least {}ms",
                MIN_HEALTH_CHECK_INTERVAL.as_millis()
            )));
        }

        if let Some(weights) = &self.weights {
            if weights.len() != self.urls.len() {
                return Err(ClientError::InvalidParameter(
                    "endpoint pool weights length must match urls length".to_string(),
                ));
            }
            if weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight < 0.0)
            {
                return Err(ClientError::InvalidParameter(
                    "endpoint pool weights must be finite and non-negative".to_string(),
                ));
            }
            if weights.iter().all(|weight| *weight == 0.0) {
                return Err(ClientError::InvalidParameter(
                    "endpoint pool weights cannot all be zero".to_string(),
                ));
            }
        }

        if let Some(endpoint_health) = &self.endpoint_health {
            if endpoint_health.len() != self.urls.len() {
                return Err(ClientError::InvalidParameter(
                    "endpoint_health length must match urls length".to_string(),
                ));
            }

            for health_cfg in endpoint_health {
                if health_cfg.checks.is_empty() {
                    return Err(ClientError::InvalidParameter(
                        "endpoint_health checks cannot be empty".to_string(),
                    ));
                }

                for check in &health_cfg.checks {
                    if check.path_or_url.trim().is_empty() {
                        return Err(ClientError::InvalidParameter(
                            "endpoint_health check path_or_url cannot be empty".to_string(),
                        ));
                    }
                    if !check.extend_base_url && reqwest::Url::parse(&check.path_or_url).is_err() {
                        return Err(ClientError::InvalidParameter(
                            "absolute endpoint_health check URLs must be valid".to_string(),
                        ));
                    }
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
struct ManagedEndpoint {
    base_url: Arc<str>,
    health_state: Arc<EndpointHealthState>,
}

#[derive(Debug)]
struct EndpointHealthState {
    healthy: AtomicBool,
}

impl EndpointHealthState {
    fn new() -> Self {
        Self {
            healthy: AtomicBool::new(true),
        }
    }

    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::SeqCst)
    }

    #[cfg(test)]
    fn set_for_test(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::SeqCst);
    }

    fn apply_vote(&self, vote: HealthVote) -> (bool, bool) {
        match vote {
            HealthVote::Healthy => {
                let was_healthy = self.healthy.swap(true, Ordering::SeqCst);
                (!was_healthy, false)
            }
            HealthVote::Unhealthy => {
                let was_healthy = self.healthy.swap(false, Ordering::SeqCst);
                (false, was_healthy)
            }
            HealthVote::NoVote => (false, false),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct EndpointSelection {
    pub endpoint_index: usize,
    pub base_url: Arc<str>,
}

#[derive(Debug, Clone)]
pub(crate) struct SingleEndpoint {
    base_url: Arc<str>,
}

impl SingleEndpoint {
    pub(crate) fn new(base_url: impl Into<Arc<str>>) -> Self {
        Self {
            base_url: base_url.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum EndpointRouter {
    Single(SingleEndpoint),
    Pool(Arc<EndpointPool>),
}

impl EndpointRouter {
    pub(crate) fn single(base_url: impl Into<Arc<str>>) -> Arc<Self> {
        Arc::new(Self::Single(SingleEndpoint::new(base_url)))
    }

    pub(crate) fn pooled(endpoint_pool: Arc<EndpointPool>) -> Arc<Self> {
        Arc::new(Self::Pool(endpoint_pool))
    }

    pub(crate) fn primary_url(&self) -> &str {
        match self {
            EndpointRouter::Single(endpoint) => endpoint.base_url.as_ref(),
            EndpointRouter::Pool(pool) => pool.primary_url(),
        }
    }

    pub(crate) fn ensure_health_worker_started(&self, api_key: &str) {
        if let EndpointRouter::Pool(pool) = self {
            pool.ensure_health_worker_started(api_key);
        }
    }

    pub(crate) fn select_endpoint(
        &self,
        excluded_indices: &[usize],
    ) -> Result<EndpointSelection, ClientError> {
        match self {
            EndpointRouter::Single(endpoint) => Ok(EndpointSelection {
                endpoint_index: 0,
                base_url: Arc::clone(&endpoint.base_url),
            }),
            EndpointRouter::Pool(pool) => pool.select_endpoint(excluded_indices),
        }
    }

    pub(crate) fn select_attempt_url(
        &self,
        request_suffix: &str,
        excluded_indices: &[usize],
    ) -> Result<(String, EndpointSelection), ClientError> {
        match self {
            EndpointRouter::Single(endpoint) => Ok((
                build_url_for_selected_endpoint(endpoint.base_url.as_ref(), request_suffix),
                EndpointSelection {
                    endpoint_index: 0,
                    base_url: Arc::clone(&endpoint.base_url),
                },
            )),
            EndpointRouter::Pool(pool) => pool.select_attempt_url(request_suffix, excluded_indices),
        }
    }

    pub(crate) fn select_hedge_url(
        &self,
        request_suffix: &str,
        original_endpoint_index: usize,
    ) -> Result<(String, EndpointSelection), ClientError> {
        match self {
            EndpointRouter::Single(endpoint) => Ok((
                build_url_for_selected_endpoint(endpoint.base_url.as_ref(), request_suffix),
                EndpointSelection {
                    endpoint_index: 0,
                    base_url: Arc::clone(&endpoint.base_url),
                },
            )),
            EndpointRouter::Pool(pool) => {
                pool.select_hedge_url(request_suffix, original_endpoint_index)
            }
        }
    }
}

#[derive(Debug)]
pub struct EndpointPool {
    endpoints: Vec<ManagedEndpoint>,
    weights: Vec<f64>,
    endpoint_health: Vec<EndpointHealthConfig>,
    health_check_interval: Duration,
    health_check_timeout: Duration,
    health_check_retries: u32,
    health_check_client_wrapper: Arc<HttpClientWrapper>,
    round_robin_counter: AtomicUsize,
    health_worker_started: AtomicBool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HealthVote {
    Healthy,
    Unhealthy,
    NoVote,
}

impl EndpointPool {
    pub fn new(config: EndpointPoolConfig) -> Result<Arc<Self>, ClientError> {
        config.validate()?;
        let endpoint_count = config.urls.len();
        let weights = config
            .weights
            .clone()
            .unwrap_or_else(|| vec![1.0; endpoint_count]);
        let endpoint_health = config.endpoint_health.clone().unwrap_or_else(|| {
            (0..endpoint_count)
                .map(|_| EndpointHealthConfig {
                    checks: build_default_health_checks(),
                    fail_on_first: false,
                })
                .collect()
        });
        let health_check_interval = config.health_check_interval;
        let health_check_timeout = config.health_check_timeout;
        let health_check_retries = config.health_check_retries;
        let health_check_client_wrapper = config.health_check_client_wrapper;

        Ok(Arc::new(Self {
            endpoints: config
                .urls
                .into_iter()
                .map(|url| ManagedEndpoint {
                    base_url: Arc::<str>::from(url),
                    health_state: Arc::new(EndpointHealthState::new()),
                })
                .collect(),
            weights,
            endpoint_health,
            health_check_interval,
            health_check_timeout,
            health_check_retries,
            health_check_client_wrapper,
            round_robin_counter: AtomicUsize::new(0),
            health_worker_started: AtomicBool::new(false),
        }))
    }

    pub(crate) fn endpoint_count(&self) -> usize {
        self.endpoints.len()
    }

    pub(crate) fn primary_url(&self) -> &str {
        self.endpoints[0].base_url.as_ref()
    }

    fn select_from_candidates(&self, candidates: &[usize]) -> Option<EndpointSelection> {
        let ticket = self.round_robin_counter.fetch_add(1, Ordering::SeqCst);
        let endpoint_index = select_candidate_index(candidates, &self.weights, ticket as u64)?;

        Some(EndpointSelection {
            endpoint_index,
            base_url: Arc::clone(&self.endpoints[endpoint_index].base_url),
        })
    }

    fn candidate_indices(&self, excluded_indices: &[usize], healthy_only: bool) -> Vec<usize> {
        self.endpoints
            .iter()
            .enumerate()
            .filter(|(index, endpoint)| {
                (!healthy_only || endpoint.health_state.is_healthy())
                    && !excluded_indices.contains(index)
            })
            .map(|(index, _)| index)
            .collect()
    }

    pub(crate) fn select_endpoint(
        &self,
        excluded_indices: &[usize],
    ) -> Result<EndpointSelection, ClientError> {
        let candidate_tiers = [
            self.candidate_indices(excluded_indices, true),
            self.candidate_indices(&[], true),
            self.candidate_indices(excluded_indices, false),
            self.candidate_indices(&[], false),
        ];

        for candidates in &candidate_tiers {
            if let Some(selection) = self.select_from_candidates(candidates) {
                return Ok(selection);
            }
        }

        Err(ClientError::Network(
            "No endpoints available for selection".to_string(),
        ))
    }

    pub(crate) fn select_hedge_endpoint(
        &self,
        original_endpoint_index: usize,
    ) -> Result<EndpointSelection, ClientError> {
        self.select_endpoint(&[original_endpoint_index])
    }

    pub(crate) fn select_attempt_url(
        &self,
        request_suffix: &str,
        excluded_indices: &[usize],
    ) -> Result<(String, EndpointSelection), ClientError> {
        let selected_endpoint = self.select_endpoint(excluded_indices)?;
        Ok((
            build_url_for_selected_endpoint(selected_endpoint.base_url.as_ref(), request_suffix),
            selected_endpoint,
        ))
    }

    pub(crate) fn select_hedge_url(
        &self,
        request_suffix: &str,
        original_endpoint_index: usize,
    ) -> Result<(String, EndpointSelection), ClientError> {
        let selected_endpoint = self.select_hedge_endpoint(original_endpoint_index)?;
        Ok((
            build_url_for_selected_endpoint(selected_endpoint.base_url.as_ref(), request_suffix),
            selected_endpoint,
        ))
    }

    pub fn health_snapshot(&self) -> EndpointPoolHealthSnapshot {
        EndpointPoolHealthSnapshot {
            endpoints: self
                .endpoints
                .iter()
                .map(|endpoint| EndpointHealthStatus {
                    base_url: endpoint.base_url.to_string(),
                    healthy: endpoint.health_state.is_healthy(),
                })
                .collect(),
        }
    }

    #[cfg(test)]
    pub(crate) fn set_endpoint_health(&self, endpoint_index: usize, healthy: bool) -> bool {
        if let Some(endpoint) = self.endpoints.get(endpoint_index) {
            endpoint.health_state.set_for_test(healthy);
            true
        } else {
            false
        }
    }

    fn apply_health_vote(&self, endpoint_index: usize, vote: HealthVote) -> (bool, bool) {
        let Some(endpoint) = self.endpoints.get(endpoint_index) else {
            return (false, false);
        };
        endpoint.health_state.apply_vote(vote)
    }

    pub(crate) async fn refresh_health(&self, api_key: &str) -> EndpointPoolHealthSnapshot {
        let client_wrapper = Arc::clone(&self.health_check_client_wrapper);
        let endpoint_health = self.endpoint_health.clone();
        let health_check_timeout = self.health_check_timeout;
        let health_check_retries = self.health_check_retries;

        let results = join_all(self.endpoints.iter().enumerate().map(|(index, endpoint)| {
            let endpoint_health = endpoint_health[index].clone();
            let client_wrapper = Arc::clone(&client_wrapper);
            async move {
                let client = client_wrapper.get_client();
                let vote = check_endpoint_health(
                    &client,
                    endpoint.base_url.as_ref(),
                    &endpoint_health,
                    api_key,
                    health_check_timeout,
                    health_check_retries,
                )
                .await;
                tracing::debug!(
                    endpoint_base_url = endpoint.base_url.as_ref(),
                    vote = ?vote,
                    "endpoint pool health refresh vote"
                );
                (index, vote)
            }
        }))
        .await;

        let mut became_unhealthy = Vec::new();
        let mut became_healthy = Vec::new();

        for (index, vote) in results {
            let (restored_to_rotation, removed_from_rotation) = self.apply_health_vote(index, vote);
            if removed_from_rotation {
                became_unhealthy.push(self.endpoints[index].base_url.to_string());
            }
            if restored_to_rotation {
                became_healthy.push(self.endpoints[index].base_url.to_string());
            }
        }

        if !became_unhealthy.is_empty() {
            tracing::warn!(
                removed_endpoints = ?became_unhealthy,
                "endpoint pool removed endpoints from healthy rotation"
            );
        }
        if !became_healthy.is_empty() {
            tracing::info!(
                restored_endpoints = ?became_healthy,
                "endpoint pool restored endpoints to healthy rotation"
            );
        }

        self.health_snapshot()
    }

    pub(crate) fn ensure_health_worker_started(self: &Arc<Self>, api_key: &str) {
        if self.endpoint_count() <= 1 || self.health_worker_started.load(Ordering::SeqCst) {
            return;
        }

        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::warn!(
                "endpoint pool configured, but no Tokio runtime is active; health worker start deferred"
            );
            return;
        };

        if self
            .health_worker_started
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return;
        }

        let weak_pool = Arc::downgrade(self);
        let api_key = api_key.to_string();

        let base_interval = self.health_check_interval;
        tracing::debug!(
            endpoint_count = self.endpoint_count(),
            health_check_interval_s = base_interval.as_secs_f64(),
            health_check_timeout_s = self.health_check_timeout.as_secs_f64(),
            health_check_retries = self.health_check_retries,
            "starting endpoint pool health worker"
        );
        let task = async move {
            let mut current_interval = base_interval;
            loop {
                let Some(endpoint_pool) = weak_pool.upgrade() else {
                    break;
                };

                let start_time = std::time::Instant::now();
                endpoint_pool.refresh_health(&api_key).await;

                let response_time = start_time.elapsed();
                if response_time > current_interval / 2 {
                    current_interval = std::cmp::min(
                        current_interval.saturating_mul(3) / 2,
                        base_interval.saturating_mul(10),
                    );
                } else {
                    current_interval = base_interval;
                }

                tokio::time::sleep(current_interval).await;
            }
        };

        handle.spawn(task);
    }
}

fn normalize_url(url: &str) -> String {
    url.trim().trim_end_matches('/').to_string()
}

fn weighted_target(ticket: u64, total_weight: f64) -> f64 {
    // Deterministic per-attempt pseudo-random target in [0, total_weight)
    // to support arbitrary floating-point weight totals.
    let mixed = splitmix64(ticket);
    let unit = (mixed as f64) / (u64::MAX as f64 + 1.0);
    unit * total_weight
}

fn select_candidate_index(candidates: &[usize], weights: &[f64], ticket: u64) -> Option<usize> {
    if candidates.is_empty() {
        return None;
    }

    let configured_total_weight: f64 = candidates
        .iter()
        .map(|&candidate| weights[candidate])
        .filter(|&weight| weight > 0.0)
        .sum();
    if configured_total_weight <= 0.0 {
        let fallback_index = candidates[(splitmix64(ticket) as usize) % candidates.len()];
        return Some(fallback_index);
    }

    let mut target = weighted_target(ticket, configured_total_weight);
    let mut chosen = None;
    for &candidate in candidates {
        let weight = weights[candidate];
        if weight <= 0.0 {
            continue;
        }
        chosen = Some(candidate);
        if target < weight {
            break;
        }
        target -= weight;
    }

    chosen
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

pub(crate) fn normalize_request_suffix(path_or_suffix: &str) -> String {
    let trimmed = path_or_suffix.trim();
    if trimmed.is_empty() {
        "/".to_string()
    } else if trimmed.starts_with('/') {
        trimmed.to_string()
    } else {
        format!("/{}", trimmed)
    }
}

pub(crate) fn build_url_for_selected_endpoint(
    selected_base_url: &str,
    request_suffix: &str,
) -> String {
    let selected_base_url = selected_base_url.trim_end_matches('/');
    let normalized_suffix = normalize_request_suffix(request_suffix);
    format!("{}{}", selected_base_url, normalized_suffix)
}

async fn check_endpoint_health(
    client: &Client,
    base_url: &str,
    endpoint_health: &EndpointHealthConfig,
    api_key: &str,
    timeout: Duration,
    retries: u32,
) -> HealthVote {
    let mut saw_negative_vote = false;

    let mut saw_positive_vote = false;
    for check in &endpoint_health.checks {
        match run_health_check_attempts(client, base_url, check, api_key, timeout, retries).await {
            HealthVote::Healthy => {
                saw_positive_vote = true;
            }
            HealthVote::Unhealthy => {
                saw_negative_vote = true;
                if endpoint_health.fail_on_first {
                    return HealthVote::Unhealthy;
                }
            }
            HealthVote::NoVote => {}
        }
    }

    if saw_negative_vote {
        HealthVote::Unhealthy
    } else if saw_positive_vote {
        HealthVote::Healthy
    } else {
        HealthVote::NoVote
    }
}

async fn run_health_check_attempts(
    client: &Client,
    base_url: &str,
    check: &EndpointHealthCheckConfig,
    api_key: &str,
    timeout: Duration,
    retries: u32,
) -> HealthVote {
    let health_url = resolve_health_check_url(base_url, check);
    let mut saw_negative_vote = false;

    for attempt in 0..=retries {
        match client
            .get(&health_url)
            .bearer_auth(api_key)
            .timeout(timeout)
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                let _ = response.bytes().await;
                tracing::trace!(
                    endpoint_base_url = base_url,
                    health_check_url = health_url,
                    attempt = attempt + 1,
                    "endpoint pool health check succeeded"
                );
                // escape hatch: a healthy vote will override any negative votes.
                return HealthVote::Healthy;
            }
            Ok(response) => {
                let _ = response.bytes().await;
                saw_negative_vote = true;
                tracing::debug!(
                    endpoint_base_url = base_url,
                    health_check_url = health_url,
                    attempt = attempt + 1,
                    "endpoint pool health check returned non-success status"
                );
            }
            Err(error) => {
                if error.is_timeout() && check.timeout_is_no_vote {
                    tracing::debug!(
                        endpoint_base_url = base_url,
                        health_check_url = health_url,
                        attempt = attempt + 1,
                        "endpoint pool health check timed out and was treated as no-vote"
                    );
                } else {
                    saw_negative_vote = true;
                    tracing::debug!(
                        endpoint_base_url = base_url,
                        health_check_url = health_url,
                        attempt = attempt + 1,
                        timeout = error.is_timeout(),
                        error = %error,
                        "endpoint pool health check request failed"
                    );
                }
            }
        }

        if attempt < retries {
            tokio::time::sleep(HEALTH_CHECK_RETRY_DELAY).await;
        }
    }

    if saw_negative_vote {
        HealthVote::Unhealthy
    } else {
        HealthVote::NoVote
    }
}

fn build_default_health_checks() -> Vec<EndpointHealthCheckConfig> {
    vec![EndpointHealthCheckConfig::relative(
        DEFAULT_HEALTH_CHECK_PATH.to_string(),
    )]
}

fn resolve_health_check_url(base_url: &str, check: &EndpointHealthCheckConfig) -> String {
    if check.extend_base_url {
        format!(
            "{}/{}",
            base_url.trim_end_matches('/'),
            check.path_or_url.trim_start_matches('/')
        )
    } else {
        check.path_or_url.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{http::StatusCode, response::IntoResponse, routing::get, Router};
    use std::sync::atomic::AtomicUsize;

    fn health_check_wrapper() -> Arc<HttpClientWrapper> {
        HttpClientWrapper::new(1, None).expect("test health-check client wrapper should build")
    }

    async fn start_health_test_server() -> String {
        async fn healthy() -> impl IntoResponse {
            (StatusCode::OK, "ok")
        }

        async fn unhealthy() -> impl IntoResponse {
            (StatusCode::SERVICE_UNAVAILABLE, "unhealthy")
        }

        let app = Router::new()
            .route("/health", get(healthy))
            .route("/health/deep", get(unhealthy));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let addr = listener
            .local_addr()
            .expect("listener should have local addr");

        tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("health test server should stay up");
        });

        format!("http://{}", addr)
    }

    async fn start_flaky_health_test_server(failures_before_success: usize) -> String {
        let remaining_failures = Arc::new(AtomicUsize::new(failures_before_success));

        let app = Router::new().route(
            "/health/flaky",
            get({
                let remaining_failures = Arc::clone(&remaining_failures);
                move || {
                    let remaining_failures = Arc::clone(&remaining_failures);
                    async move {
                        let previous = remaining_failures.fetch_update(
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                            |current| current.checked_sub(1),
                        );
                        match previous {
                            Ok(_) => (StatusCode::SERVICE_UNAVAILABLE, "unhealthy").into_response(),
                            Err(_) => (StatusCode::OK, "ok").into_response(),
                        }
                    }
                }
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let addr = listener
            .local_addr()
            .expect("listener should have local addr");

        tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("flaky health test server should stay up");
        });

        format!("http://{}", addr)
    }

    #[test]
    fn test_endpoint_pool_validation_rejects_duplicates() {
        let result = EndpointPool::new(EndpointPoolConfig::new(
            vec![
                "https://example.com".to_string(),
                "https://example.com/".to_string(),
            ],
            health_check_wrapper(),
        ));

        assert!(result.is_err());
    }

    #[test]
    fn test_equal_weights_use_deterministic_mixed_routing() {
        let pool = EndpointPool::new(EndpointPoolConfig::new(
            vec![
                "https://a.example.com".to_string(),
                "https://b.example.com".to_string(),
                "https://c.example.com".to_string(),
            ],
            health_check_wrapper(),
        ))
        .expect("pool should be valid");

        let mut seen = [0usize; 3];
        for _ in 0..48 {
            let selected = pool.select_endpoint(&[]).unwrap();
            seen[selected.endpoint_index] += 1;
        }

        assert!(seen.iter().all(|count| *count > 0));
    }

    #[test]
    fn test_unhealthy_endpoint_is_skipped_when_alternatives_exist() {
        let pool = EndpointPool::new(EndpointPoolConfig::new(
            vec![
                "https://a.example.com".to_string(),
                "https://b.example.com".to_string(),
                "https://c.example.com".to_string(),
            ],
            health_check_wrapper(),
        ))
        .expect("pool should be valid");

        pool.set_endpoint_health(1, false);

        let first = pool.select_endpoint(&[]).unwrap();
        let second = pool.select_endpoint(&[]).unwrap();

        assert_ne!(first.endpoint_index, 1);
        assert_ne!(second.endpoint_index, 1);
    }

    #[test]
    fn test_endpoint_health_validation_rejects_empty_checks() {
        let result = EndpointPool::new(
            EndpointPoolConfig::new(
                vec!["https://a.example.com".to_string()],
                health_check_wrapper(),
            )
            .with_endpoint_health(vec![EndpointHealthConfig {
                checks: Vec::new(),
                fail_on_first: false,
            }]),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_selection_defaults_to_primary_when_others_are_zero() {
        let pool = EndpointPool::new(
            EndpointPoolConfig::new(
                vec![
                    "https://a.example.com".to_string(),
                    "https://b.example.com".to_string(),
                    "https://c.example.com".to_string(),
                ],
                health_check_wrapper(),
            )
            .with_weights(vec![1.0, 0.0, 0.0]),
        )
        .expect("pool should be valid");

        for _ in 0..5 {
            let selected = pool.select_endpoint(&[]).unwrap();
            assert_eq!(selected.endpoint_index, 0);
        }
    }

    #[test]
    fn test_weight_zero_endpoint_used_when_primary_unhealthy() {
        let pool = EndpointPool::new(
            EndpointPoolConfig::new(
                vec![
                    "https://a.example.com".to_string(),
                    "https://b.example.com".to_string(),
                    "https://c.example.com".to_string(),
                ],
                health_check_wrapper(),
            )
            .with_weights(vec![1.0, 0.0, 0.0]),
        )
        .expect("pool should be valid");
        pool.set_endpoint_health(0, false);

        let selected = pool.select_endpoint(&[]).unwrap();
        assert_ne!(selected.endpoint_index, 0);
    }

    #[test]
    fn test_endpoint_health_flips_immediately_on_unhealthy_vote() {
        let pool = EndpointPool::new(EndpointPoolConfig::new(
            vec!["https://a.example.com".to_string()],
            health_check_wrapper(),
        ))
        .expect("pool should be valid");

        assert_eq!(pool.health_snapshot().endpoints[0].healthy, true);

        pool.apply_health_vote(0, HealthVote::Unhealthy);
        assert_eq!(pool.health_snapshot().endpoints[0].healthy, false);

        pool.apply_health_vote(0, HealthVote::Healthy);
        assert_eq!(pool.health_snapshot().endpoints[0].healthy, true);
    }

    #[tokio::test]
    async fn test_deep_health_failure_overrides_shallow_success() {
        let base_url = start_health_test_server().await;
        let client = reqwest::Client::new();
        let endpoint_health = EndpointHealthConfig {
            checks: vec![
                EndpointHealthCheckConfig::relative("/health".to_string()),
                EndpointHealthCheckConfig::relative("/health/deep".to_string()),
            ],
            fail_on_first: false,
        };

        let vote = check_endpoint_health(
            &client,
            &base_url,
            &endpoint_health,
            "test-api-key",
            Duration::from_secs(1),
            0,
        )
        .await;

        assert_eq!(vote, HealthVote::Unhealthy);
    }

    #[tokio::test]
    async fn test_health_check_retry_succeeds_on_later_attempt() {
        let base_url = start_flaky_health_test_server(1).await;
        let client = reqwest::Client::new();
        let endpoint_health = EndpointHealthConfig {
            checks: vec![EndpointHealthCheckConfig::relative(
                "/health/flaky".to_string(),
            )],
            fail_on_first: false,
        };

        let vote = check_endpoint_health(
            &client,
            &base_url,
            &endpoint_health,
            "test-api-key",
            Duration::from_secs(1),
            1,
        )
        .await;

        assert_eq!(vote, HealthVote::Healthy);
    }

    #[test]
    fn test_weight_validation_rejects_all_zero_weights() {
        let result = EndpointPool::new(
            EndpointPoolConfig::new(
                vec![
                    "https://a.example.com".to_string(),
                    "https://b.example.com".to_string(),
                ],
                health_check_wrapper(),
            )
            .with_weights(vec![0.0, 0.0]),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_fractional_weights_do_not_stick_to_first_endpoint() {
        let pool = EndpointPool::new(
            EndpointPoolConfig::new(
                vec![
                    "https://a.example.com".to_string(),
                    "https://b.example.com".to_string(),
                ],
                health_check_wrapper(),
            )
            .with_weights(vec![0.1, 0.9]),
        )
        .expect("pool should be valid");

        let mut saw_first = false;
        let mut saw_second = false;
        for _ in 0..64 {
            let selected = pool.select_endpoint(&[]).unwrap();
            if selected.endpoint_index == 0 {
                saw_first = true;
            }
            if selected.endpoint_index == 1 {
                saw_second = true;
            }
        }

        assert!(saw_first);
        assert!(saw_second);
    }

    #[test]
    fn test_pure_selector_is_deterministic_for_same_ticket() {
        let candidates = vec![0usize, 1, 2];
        let weights = vec![1.0, 3.0, 2.0];

        let first = select_candidate_index(&candidates, &weights, 42);
        let second = select_candidate_index(&candidates, &weights, 42);

        assert_eq!(first, second);
    }

    #[test]
    fn test_pure_selector_uses_weights_across_many_tickets() {
        let candidates = vec![0usize, 1];
        let weights = vec![1.0, 3.0];

        let mut counts = [0usize; 2];
        for ticket in 0..4000u64 {
            let index = select_candidate_index(&candidates, &weights, ticket).unwrap();
            counts[index] += 1;
        }

        assert!((900..=1100).contains(&counts[0]));
        assert!((2900..=3100).contains(&counts[1]));
    }

    #[test]
    fn test_pure_selector_returns_none_for_empty_candidates() {
        let candidates: Vec<usize> = vec![];
        let weights = vec![1.0, 1.0];

        assert_eq!(select_candidate_index(&candidates, &weights, 0), None);
    }

    #[test]
    fn test_pure_selector_skips_zero_weight_candidates() {
        let candidates = vec![0usize, 1, 2];
        let weights = vec![0.0, 5.0, 0.0];

        for ticket in 0..64u64 {
            assert_eq!(
                select_candidate_index(&candidates, &weights, ticket),
                Some(1)
            );
        }
    }

    #[test]
    fn test_pure_selector_falls_back_within_candidate_set_when_weights_are_zero() {
        let candidates = vec![1usize, 2];
        let weights = vec![1.0, 0.0, 0.0];

        for ticket in 0..64u64 {
            let selected = select_candidate_index(&candidates, &weights, ticket).unwrap();
            assert!(candidates.contains(&selected));
        }
    }
}

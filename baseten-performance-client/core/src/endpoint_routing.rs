use crate::client::HttpClientWrapper;
use crate::errors::ClientError;

use nonempty::NonEmpty;
use reqwest::Client;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const DEFAULT_HEALTH_CHECK_PATH: &str = "/health";
const DEFAULT_HEALTH_CHECK_INTERVAL: Duration = Duration::from_secs(10);
const DEFAULT_HEALTH_CHECK_TIMEOUT: Duration = Duration::from_secs(6);
const DEFAULT_HEALTH_CHECK_RETRIES: u32 = 2;
pub const DEFAULT_TIMEOUT_IS_NO_VOTE: bool = true;
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
            timeout_is_no_vote: DEFAULT_TIMEOUT_IS_NO_VOTE,
        }
    }

    pub fn absolute(url: String) -> Self {
        Self {
            path_or_url: url,
            extend_base_url: false,
            timeout_is_no_vote: DEFAULT_TIMEOUT_IS_NO_VOTE,
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
pub struct EndpointConfig {
    pub base_url: String,
    pub health_api_key: String,
    pub endpoint_health: Option<EndpointHealthConfig>,
    pub health_check_interval: Duration,
    pub health_check_timeout: Duration,
    pub health_check_retries: u32,
    pub health_check_client_wrapper: Arc<HttpClientWrapper>,
}

impl EndpointConfig {
    pub fn new(
        base_url: String,
        health_api_key: String,
        health_check_client_wrapper: Arc<HttpClientWrapper>,
    ) -> Self {
        Self {
            base_url,
            health_api_key,
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

    pub fn with_endpoint_health(mut self, endpoint_health: EndpointHealthConfig) -> Self {
        self.endpoint_health = Some(endpoint_health);
        self
    }

    pub fn with_standard_health_checks(
        mut self,
        deep_health_url: Option<String>,
        fail_on_first: bool,
        deployment_health_path: Option<String>,
        deployment_timeout_is_no_vote: bool,
        deep_timeout_is_no_vote: bool,
    ) -> Self {
        let deployment_health_path =
            deployment_health_path.unwrap_or_else(|| DEFAULT_HEALTH_CHECK_PATH.to_string());
        self.endpoint_health = Some(match deep_health_url {
            Some(deep_url) => EndpointHealthConfig {
                checks: vec![
                    EndpointHealthCheckConfig::relative(deployment_health_path)
                        .with_timeout_is_no_vote(deployment_timeout_is_no_vote),
                    EndpointHealthCheckConfig::absolute(deep_url)
                        .with_timeout_is_no_vote(deep_timeout_is_no_vote),
                ],
                fail_on_first,
            },
            None => EndpointHealthConfig {
                checks: vec![EndpointHealthCheckConfig::relative(deployment_health_path)
                    .with_timeout_is_no_vote(deployment_timeout_is_no_vote)],
                fail_on_first,
            },
        });
        self
    }

    fn validate(&self) -> Result<(), ClientError> {
        if self.base_url.trim().is_empty() {
            return Err(ClientError::InvalidParameter(
                "endpoint base_url cannot be empty".to_string(),
            ));
        }
        if self.health_api_key.trim().is_empty() {
            return Err(ClientError::InvalidParameter(
                "endpoint health_api_key cannot be empty".to_string(),
            ));
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

        if let Some(endpoint_health) = &self.endpoint_health {
            if endpoint_health.checks.is_empty() {
                return Err(ClientError::InvalidParameter(
                    "endpoint_health checks cannot be empty".to_string(),
                ));
            }

            for check in &endpoint_health.checks {
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

        Ok(())
    }
}

#[derive(Debug)]
struct EndpointInner {
    base_url: Arc<str>,
    health_api_key: String,
    health_state: EndpointHealthState,
    endpoint_health: EndpointHealthConfig,
    health_check_interval: Duration,
    health_check_timeout: Duration,
    health_check_retries: u32,
    health_check_client_wrapper: Arc<HttpClientWrapper>,
}

#[derive(Debug, Clone)]
pub struct Endpoint {
    inner: Arc<EndpointInner>,
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

impl Endpoint {
    pub fn new(config: EndpointConfig) -> Result<Self, ClientError> {
        config.validate()?;
        let endpoint_health = config
            .endpoint_health
            .unwrap_or_else(|| EndpointHealthConfig {
                checks: build_default_health_checks(),
                fail_on_first: false,
            });

        let inner = Arc::new(EndpointInner {
            base_url: Arc::<str>::from(normalize_url(&config.base_url)),
            health_api_key: config.health_api_key,
            health_state: EndpointHealthState::new(),
            endpoint_health,
            health_check_interval: config.health_check_interval,
            health_check_timeout: config.health_check_timeout,
            health_check_retries: config.health_check_retries,
            health_check_client_wrapper: config.health_check_client_wrapper,
        });
        start_endpoint_health_worker(&inner)?;
        Ok(Self { inner })
    }

    pub fn base_url(&self) -> &str {
        self.inner.base_url.as_ref()
    }

    fn is_healthy(&self) -> bool {
        self.inner.health_state.is_healthy()
    }

    #[cfg(test)]
    fn set_healthy_for_test(&self, healthy: bool) {
        self.inner.health_state.set_for_test(healthy);
    }

    fn apply_health_vote(&self, vote: HealthVote) -> (bool, bool) {
        self.inner.health_state.apply_vote(vote)
    }

    async fn refresh_health(&self) {
        let client = self.inner.health_check_client_wrapper.get_client();
        let vote = check_endpoint_health(
            &client,
            self.base_url(),
            &self.inner.endpoint_health,
            &self.inner.health_api_key,
            self.inner.health_check_timeout,
            self.inner.health_check_retries,
        )
        .await;

        tracing::debug!(
            endpoint_base_url = self.base_url(),
            vote = ?vote,
            "endpoint health refresh vote"
        );

        let (restored_to_rotation, removed_from_rotation) = self.apply_health_vote(vote);
        if removed_from_rotation {
            tracing::warn!(
                endpoint_base_url = self.base_url(),
                "endpoint marked as unhealthy"
            );
        }
        if restored_to_rotation {
            tracing::info!(
                endpoint_base_url = self.base_url(),
                "endpoint restored as healthy"
            );
        }
    }
}

#[derive(Debug, Clone)]
pub struct EndpointPoolConfig {
    pub endpoints: Vec<Endpoint>,
    pub weights: Option<Vec<f64>>,
}

impl EndpointPoolConfig {
    pub fn new(endpoints: Vec<Endpoint>) -> Self {
        Self {
            endpoints,
            weights: None,
        }
    }

    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    fn resolve(self) -> Result<ResolvedEndpointPoolConfig, ClientError> {
        if self.endpoints.is_empty() {
            return Err(ClientError::InvalidParameter(
                "endpoint pool must contain at least one endpoint".to_string(),
            ));
        }

        let mut normalized_urls = Vec::with_capacity(self.endpoints.len());
        for endpoint in &self.endpoints {
            let normalized = normalize_url(endpoint.base_url());
            if normalized_urls
                .iter()
                .any(|existing| existing == &normalized)
            {
                return Err(ClientError::InvalidParameter(
                    "endpoint pool endpoints must be unique".to_string(),
                ));
            }
            normalized_urls.push(normalized);
        }

        if let Some(weights) = &self.weights {
            if weights.len() != self.endpoints.len() {
                return Err(ClientError::InvalidParameter(
                    "endpoint pool weights length must match endpoints length".to_string(),
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

        let endpoints = NonEmpty::from_vec(self.endpoints).ok_or_else(|| {
            ClientError::InvalidParameter(
                "endpoint pool must contain at least one endpoint".to_string(),
            )
        })?;
        let weights = self.weights.unwrap_or_else(|| vec![1.0; endpoints.len()]);

        Ok(ResolvedEndpointPoolConfig { endpoints, weights })
    }
}

#[derive(Debug)]
struct ResolvedEndpointPoolConfig {
    endpoints: NonEmpty<Endpoint>,
    weights: Vec<f64>,
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

    pub(crate) fn select_endpoint(&self, excluded_indices: &[usize]) -> EndpointSelection {
        match self {
            EndpointRouter::Single(endpoint) => EndpointSelection {
                endpoint_index: 0,
                base_url: Arc::clone(&endpoint.base_url),
            },
            EndpointRouter::Pool(pool) => pool.select_endpoint(excluded_indices),
        }
    }

    pub(crate) fn select_attempt_url(
        &self,
        request_suffix: &str,
        excluded_indices: &[usize],
    ) -> (String, EndpointSelection) {
        match self {
            EndpointRouter::Single(endpoint) => (
                build_url_for_selected_endpoint(endpoint.base_url.as_ref(), request_suffix),
                EndpointSelection {
                    endpoint_index: 0,
                    base_url: Arc::clone(&endpoint.base_url),
                },
            ),
            EndpointRouter::Pool(pool) => pool.select_attempt_url(request_suffix, excluded_indices),
        }
    }

    pub(crate) fn select_hedge_url(
        &self,
        request_suffix: &str,
        original_endpoint_index: usize,
    ) -> (String, EndpointSelection) {
        match self {
            EndpointRouter::Single(endpoint) => (
                build_url_for_selected_endpoint(endpoint.base_url.as_ref(), request_suffix),
                EndpointSelection {
                    endpoint_index: 0,
                    base_url: Arc::clone(&endpoint.base_url),
                },
            ),
            EndpointRouter::Pool(pool) => {
                pool.select_hedge_url(request_suffix, original_endpoint_index)
            }
        }
    }
}

#[derive(Debug)]
pub struct EndpointPool {
    endpoints: NonEmpty<Endpoint>,
    weights: Vec<f64>,
    round_robin_counter: AtomicUsize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HealthVote {
    Healthy,
    Unhealthy,
    NoVote,
}

impl EndpointPool {
    pub fn new(config: EndpointPoolConfig) -> Result<Arc<Self>, ClientError> {
        let resolved = config.resolve()?;

        Ok(Arc::new(Self {
            endpoints: resolved.endpoints,
            weights: resolved.weights,
            round_robin_counter: AtomicUsize::new(0),
        }))
    }

    pub(crate) fn primary_url(&self) -> &str {
        self.endpoints[0].base_url()
    }

    fn select_from_candidates(&self, candidates: &[usize]) -> Option<EndpointSelection> {
        let ticket = self.round_robin_counter.fetch_add(1, Ordering::SeqCst);
        let endpoint_index = select_candidate_index(candidates, &self.weights, ticket as u64)?;

        Some(EndpointSelection {
            endpoint_index,
            base_url: Arc::clone(&self.endpoints[endpoint_index].inner.base_url),
        })
    }

    fn candidate_indices(&self, excluded_indices: &[usize], healthy_only: bool) -> Vec<usize> {
        self.endpoints
            .iter()
            .enumerate()
            .filter(|(index, endpoint)| {
                (!healthy_only || endpoint.is_healthy()) && !excluded_indices.contains(index)
            })
            .map(|(index, _)| index)
            .collect()
    }

    pub(crate) fn select_endpoint(&self, excluded_indices: &[usize]) -> EndpointSelection {
        let candidate_tiers = [
            self.candidate_indices(excluded_indices, true),
            self.candidate_indices(&[], true),
            self.candidate_indices(excluded_indices, false),
            self.candidate_indices(&[], false),
        ];

        for candidates in &candidate_tiers[..3] {
            if let Some(selection) = self.select_from_candidates(candidates) {
                return selection;
            }
        }

        self.select_from_candidates(&candidate_tiers[3])
            .expect("endpoint pool must always be able to select a fallback endpoint")
    }

    pub(crate) fn select_hedge_endpoint(
        &self,
        original_endpoint_index: usize,
    ) -> EndpointSelection {
        self.select_endpoint(&[original_endpoint_index])
    }

    pub(crate) fn select_attempt_url(
        &self,
        request_suffix: &str,
        excluded_indices: &[usize],
    ) -> (String, EndpointSelection) {
        let selected_endpoint = self.select_endpoint(excluded_indices);
        (
            build_url_for_selected_endpoint(selected_endpoint.base_url.as_ref(), request_suffix),
            selected_endpoint,
        )
    }

    pub(crate) fn select_hedge_url(
        &self,
        request_suffix: &str,
        original_endpoint_index: usize,
    ) -> (String, EndpointSelection) {
        let selected_endpoint = self.select_hedge_endpoint(original_endpoint_index);
        (
            build_url_for_selected_endpoint(selected_endpoint.base_url.as_ref(), request_suffix),
            selected_endpoint,
        )
    }

    pub fn health_snapshot(&self) -> EndpointPoolHealthSnapshot {
        EndpointPoolHealthSnapshot {
            endpoints: self
                .endpoints
                .iter()
                .map(|endpoint| EndpointHealthStatus {
                    base_url: endpoint.base_url().to_string(),
                    healthy: endpoint.is_healthy(),
                })
                .collect(),
        }
    }

    #[cfg(test)]
    pub(crate) fn set_endpoint_health(&self, endpoint_index: usize, healthy: bool) -> bool {
        if let Some(endpoint) = self.endpoints.get(endpoint_index) {
            endpoint.set_healthy_for_test(healthy);
            true
        } else {
            false
        }
    }
}

fn start_endpoint_health_worker(inner: &Arc<EndpointInner>) -> Result<(), ClientError> {
    let weak_endpoint = Arc::downgrade(inner);
    let endpoint_base_url = inner.base_url.to_string();
    let base_interval = inner.health_check_interval;
    let health_check_timeout_s = inner.health_check_timeout.as_secs_f64();
    let health_check_retries = inner.health_check_retries;
    let (started_tx, started_rx) = std::sync::mpsc::channel();

    std::thread::Builder::new()
        .name(format!(
            "bpclient-health-{}",
            splitmix64(endpoint_base_url.len() as u64)
        ))
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();
            match runtime {
                Ok(runtime) => {
                    let _ = started_tx.send(Ok(()));
                    runtime.block_on(async move {
                        tracing::debug!(
                            endpoint_base_url,
                            health_check_interval_s = base_interval.as_secs_f64(),
                            health_check_timeout_s,
                            health_check_retries,
                            "starting endpoint health worker"
                        );

                        let mut current_interval = base_interval;
                        loop {
                            let Some(inner) = weak_endpoint.upgrade() else {
                                break;
                            };
                            let endpoint = Endpoint { inner };
                            let start_time = std::time::Instant::now();
                            endpoint.refresh_health().await;

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
                    });
                }
                Err(error) => {
                    let _ = started_tx.send(Err(error.to_string()));
                }
            }
        })
        .map_err(|error| {
            ClientError::Connect(format!(
                "Failed to spawn endpoint health worker thread: {}",
                error
            ))
        })?;

    match started_rx.recv() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(ClientError::Connect(format!(
            "Failed to start endpoint health worker runtime: {}",
            error
        ))),
        Err(error) => Err(ClientError::Connect(format!(
            "Failed to receive endpoint health worker startup signal: {}",
            error
        ))),
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

    fn endpoint(base_url: &str) -> Endpoint {
        Endpoint::new(EndpointConfig::new(
            base_url.to_string(),
            "test-api-key".to_string(),
            health_check_wrapper(),
        ))
        .expect("endpoint should be valid")
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
        let result = EndpointPool::new(EndpointPoolConfig::new(vec![
            endpoint("https://example.com"),
            endpoint("https://example.com/"),
        ]));

        assert!(result.is_err());
    }

    #[test]
    fn test_equal_weights_use_deterministic_mixed_routing() {
        let pool = EndpointPool::new(EndpointPoolConfig::new(vec![
            endpoint("https://a.example.com"),
            endpoint("https://b.example.com"),
            endpoint("https://c.example.com"),
        ]))
        .expect("pool should be valid");

        let mut seen = [0usize; 3];
        for _ in 0..48 {
            let selected = pool.select_endpoint(&[]);
            seen[selected.endpoint_index] += 1;
        }

        assert!(seen.iter().all(|count| *count > 0));
    }

    #[test]
    fn test_unhealthy_endpoint_is_skipped_when_alternatives_exist() {
        let pool = EndpointPool::new(EndpointPoolConfig::new(vec![
            endpoint("https://a.example.com"),
            endpoint("https://b.example.com"),
            endpoint("https://c.example.com"),
        ]))
        .expect("pool should be valid");

        pool.set_endpoint_health(1, false);

        let first = pool.select_endpoint(&[]);
        let second = pool.select_endpoint(&[]);

        assert_ne!(first.endpoint_index, 1);
        assert_ne!(second.endpoint_index, 1);
    }

    #[test]
    fn test_endpoint_health_validation_rejects_empty_checks() {
        let result = Endpoint::new(
            EndpointConfig::new(
                "https://a.example.com".to_string(),
                "test-api-key".to_string(),
                health_check_wrapper(),
            )
            .with_endpoint_health(EndpointHealthConfig {
                checks: Vec::new(),
                fail_on_first: false,
            }),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_selection_defaults_to_primary_when_others_are_zero() {
        let pool = EndpointPool::new(
            EndpointPoolConfig::new(vec![
                endpoint("https://a.example.com"),
                endpoint("https://b.example.com"),
                endpoint("https://c.example.com"),
            ])
            .with_weights(vec![1.0, 0.0, 0.0]),
        )
        .expect("pool should be valid");

        for _ in 0..5 {
            let selected = pool.select_endpoint(&[]);
            assert_eq!(selected.endpoint_index, 0);
        }
    }

    #[test]
    fn test_weight_zero_endpoint_used_when_primary_unhealthy() {
        let pool = EndpointPool::new(
            EndpointPoolConfig::new(vec![
                endpoint("https://a.example.com"),
                endpoint("https://b.example.com"),
                endpoint("https://c.example.com"),
            ])
            .with_weights(vec![1.0, 0.0, 0.0]),
        )
        .expect("pool should be valid");
        pool.set_endpoint_health(0, false);

        let selected = pool.select_endpoint(&[]);
        assert_ne!(selected.endpoint_index, 0);
    }

    #[test]
    fn test_endpoint_health_flips_immediately_on_unhealthy_vote() {
        let endpoint = endpoint("https://a.example.com");

        assert!(endpoint.is_healthy());

        endpoint.apply_health_vote(HealthVote::Unhealthy);
        assert!(!endpoint.is_healthy());

        endpoint.apply_health_vote(HealthVote::Healthy);
        assert!(endpoint.is_healthy());
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
            EndpointPoolConfig::new(vec![
                endpoint("https://a.example.com"),
                endpoint("https://b.example.com"),
            ])
            .with_weights(vec![0.0, 0.0]),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_fractional_weights_do_not_stick_to_first_endpoint() {
        let pool = EndpointPool::new(
            EndpointPoolConfig::new(vec![
                endpoint("https://a.example.com"),
                endpoint("https://b.example.com"),
            ])
            .with_weights(vec![0.1, 0.9]),
        )
        .expect("pool should be valid");

        let mut saw_first = false;
        let mut saw_second = false;
        for _ in 0..64 {
            let selected = pool.select_endpoint(&[]);
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

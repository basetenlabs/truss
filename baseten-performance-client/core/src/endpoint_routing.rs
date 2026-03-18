use crate::errors::ClientError;

use futures::future::join_all;
use reqwest::Client;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const DEFAULT_HEALTH_CHECK_PATH: &str = "/health";
const DEFAULT_HEALTH_CHECK_INTERVAL: Duration = Duration::from_secs(10);
const DEFAULT_HEALTH_CHECK_TIMEOUT: Duration = Duration::from_secs(2);
const DEFAULT_HEALTH_CHECK_RETRIES: u32 = 3;
const HEALTH_CHECK_RETRY_DELAY: Duration = Duration::from_millis(100);
const MIN_HEALTH_CHECK_TIMEOUT: Duration = Duration::from_millis(50);
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
            timeout_is_no_vote: true,
        }
    }

    pub fn absolute(url: String) -> Self {
        Self {
            path_or_url: url,
            extend_base_url: false,
            timeout_is_no_vote: true,
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
}

impl EndpointPoolConfig {
    pub fn new(urls: Vec<String>) -> Self {
        Self {
            urls,
            weights: None,
            endpoint_health: None,
            health_check_interval: DEFAULT_HEALTH_CHECK_INTERVAL,
            health_check_timeout: DEFAULT_HEALTH_CHECK_TIMEOUT,
            health_check_retries: DEFAULT_HEALTH_CHECK_RETRIES,
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
    healthy: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
pub(crate) struct EndpointSelection {
    pub endpoint_index: usize,
    pub base_url: Arc<str>,
}

#[derive(Debug)]
pub struct EndpointPool {
    endpoints: Vec<ManagedEndpoint>,
    weights: Vec<f64>,
    endpoint_health: Vec<EndpointHealthConfig>,
    health_check_interval: Duration,
    health_check_timeout: Duration,
    health_check_retries: u32,
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

        Ok(Arc::new(Self {
            endpoints: config
                .urls
                .into_iter()
                .map(|url| ManagedEndpoint {
                    base_url: Arc::<str>::from(url),
                    healthy: Arc::new(AtomicBool::new(true)),
                })
                .collect(),
            weights,
            endpoint_health,
            health_check_interval,
            health_check_timeout,
            health_check_retries,
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

    pub(crate) fn health_check_interval(&self) -> Duration {
        self.health_check_interval
    }

    pub(crate) fn mark_health_worker_started(&self) -> bool {
        !self.health_worker_started.swap(true, Ordering::SeqCst)
    }

    fn select_from_candidates(&self, candidates: &[usize]) -> Option<EndpointSelection> {
        if candidates.is_empty() {
            return None;
        }

        let ticket = self.round_robin_counter.fetch_add(1, Ordering::SeqCst);
        let configured_total_weight: f64 = candidates
            .iter()
            .map(|&candidate| self.weights[candidate])
            .filter(|&weight| weight > 0.0)
            .sum();
        let use_equal_weights = configured_total_weight <= 0.0;
        let all_equal_positive_weights = !use_equal_weights
            && candidates.iter().all(|&candidate| {
                let weight = self.weights[candidate];
                weight > 0.0 && (weight - self.weights[candidates[0]]).abs() <= f64::EPSILON
            });

        if use_equal_weights || all_equal_positive_weights {
            let endpoint_index = candidates[ticket % candidates.len()];
            return Some(EndpointSelection {
                endpoint_index,
                base_url: Arc::clone(&self.endpoints[endpoint_index].base_url),
            });
        }

        let mut target = weighted_target(ticket as u64, configured_total_weight);
        let mut chosen = None;
        for &candidate in candidates {
            let weight = self.weights[candidate];
            if weight <= 0.0 {
                continue;
            }
            chosen = Some(candidate);
            if target < weight {
                break;
            }
            target -= weight;
        }
        let endpoint_index = chosen?;

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
                (!healthy_only || endpoint.healthy.load(Ordering::SeqCst))
                    && !excluded_indices.contains(index)
            })
            .map(|(index, _)| index)
            .collect()
    }

    pub(crate) fn select_endpoint(&self, excluded_indices: &[usize]) -> Result<EndpointSelection, ClientError> {
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
        configured_base_url: &str,
        original_url: &str,
        excluded_indices: &[usize],
    ) -> Result<(String, usize), ClientError> {
        let selected_endpoint = self.select_endpoint(excluded_indices)?;
        let attempt_url = rewrite_url_for_selected_endpoint(
            original_url,
            configured_base_url,
            selected_endpoint.base_url.as_ref(),
        );
        Ok((attempt_url, selected_endpoint.endpoint_index))
    }

    pub(crate) fn select_hedge_url(
        &self,
        configured_base_url: &str,
        original_url: &str,
        original_endpoint_index: usize,
    ) -> Result<String, ClientError> {
        let selected_endpoint = self.select_hedge_endpoint(original_endpoint_index)?;
        Ok(rewrite_url_for_selected_endpoint(
            original_url,
            configured_base_url,
            selected_endpoint.base_url.as_ref(),
        ))
    }

    pub fn health_snapshot(&self) -> EndpointPoolHealthSnapshot {
        EndpointPoolHealthSnapshot {
            endpoints: self
                .endpoints
                .iter()
                .map(|endpoint| EndpointHealthStatus {
                    base_url: endpoint.base_url.to_string(),
                    healthy: endpoint.healthy.load(Ordering::SeqCst),
                })
                .collect(),
        }
    }

    pub(crate) fn set_endpoint_health(&self, endpoint_index: usize, healthy: bool) -> bool {
        if let Some(endpoint) = self.endpoints.get(endpoint_index) {
            endpoint.healthy.store(healthy, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    pub(crate) async fn refresh_health(
        &self,
        client: &Client,
        api_key: &str,
    ) -> EndpointPoolHealthSnapshot {
        let endpoint_health = self.endpoint_health.clone();
        let health_check_timeout = self.health_check_timeout;
        let health_check_retries = self.health_check_retries;

        let results = join_all(self.endpoints.iter().enumerate().map(|(index, endpoint)| {
            let endpoint_health = endpoint_health[index].clone();
            async move {
                let vote = check_endpoint_health(
                    client,
                    endpoint.base_url.as_ref(),
                    &endpoint_health,
                    api_key,
                    health_check_timeout,
                    health_check_retries,
                )
                .await;
                (index, vote)
            }
        }))
        .await;

        for (index, vote) in results {
            match vote {
                HealthVote::Healthy => {
                    self.set_endpoint_health(index, true);
                }
                HealthVote::Unhealthy => {
                    self.set_endpoint_health(index, false);
                }
                HealthVote::NoVote => {}
            }
        }

        self.health_snapshot()
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

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn rewrite_url_for_selected_endpoint(
    original_url: &str,
    configured_base_url: &str,
    selected_base_url: &str,
) -> String {
    let configured_base_url = configured_base_url.trim_end_matches('/');
    let selected_base_url = selected_base_url.trim_end_matches('/');

    if configured_base_url == selected_base_url {
        return original_url.to_string();
    }

    if let Ok(parsed_original) = reqwest::Url::parse(original_url) {
        if let Ok(parsed_selected) = reqwest::Url::parse(selected_base_url) {
            let mut rewritten = parsed_selected.clone();
            rewritten.set_path(parsed_original.path());
            rewritten.set_query(parsed_original.query());
            return rewritten.to_string();
        }
    }

    if let Some(suffix) = original_url.strip_prefix(configured_base_url) {
        return format!("{}{}", selected_base_url, suffix);
    }

    original_url.to_string()
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

    for attempt in 0..=retries {
        for check in &endpoint_health.checks {
            let health_url = resolve_health_check_url(base_url, check);
            match client
                .get(&health_url)
                .bearer_auth(api_key)
                .timeout(timeout)
                .send()
                .await
            {
                Ok(response) if response.status().is_success() => return HealthVote::Healthy,
                Ok(_) => {
                    saw_negative_vote = true;
                    if endpoint_health.fail_on_first {
                        break;
                    }
                }
                Err(error) => {
                    if error.is_timeout() && check.timeout_is_no_vote {
                        continue;
                    }
                    saw_negative_vote = true;
                    if endpoint_health.fail_on_first {
                        break;
                    }
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

    #[test]
    fn test_endpoint_pool_validation_rejects_duplicates() {
        let result = EndpointPool::new(EndpointPoolConfig::new(vec![
            "https://example.com".to_string(),
            "https://example.com/".to_string(),
        ]));

        assert!(result.is_err());
    }

    #[test]
    fn test_round_robin_spreads_across_multiple_healthy_endpoints() {
        let pool = EndpointPool::new(EndpointPoolConfig::new(vec![
            "https://a.example.com".to_string(),
            "https://b.example.com".to_string(),
            "https://c.example.com".to_string(),
        ]))
        .expect("pool should be valid");

        let first = pool.select_endpoint(&[]).unwrap();
        let second = pool.select_endpoint(&[]).unwrap();
        let third = pool.select_endpoint(&[]).unwrap();

        assert_eq!(first.endpoint_index, 0);
        assert_eq!(second.endpoint_index, 1);
        assert_eq!(third.endpoint_index, 2);
    }

    #[test]
    fn test_unhealthy_endpoint_is_skipped_when_alternatives_exist() {
        let pool = EndpointPool::new(EndpointPoolConfig::new(vec![
            "https://a.example.com".to_string(),
            "https://b.example.com".to_string(),
            "https://c.example.com".to_string(),
        ]))
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
            EndpointPoolConfig::new(vec!["https://a.example.com".to_string()])
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
            EndpointPoolConfig::new(vec![
                "https://a.example.com".to_string(),
                "https://b.example.com".to_string(),
                "https://c.example.com".to_string(),
            ])
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
            EndpointPoolConfig::new(vec![
                "https://a.example.com".to_string(),
                "https://b.example.com".to_string(),
                "https://c.example.com".to_string(),
            ])
            .with_weights(vec![1.0, 0.0, 0.0]),
        )
        .expect("pool should be valid");
        pool.set_endpoint_health(0, false);

        let selected = pool.select_endpoint(&[]).unwrap();
        assert_ne!(selected.endpoint_index, 0);
    }

    #[test]
    fn test_weight_validation_rejects_all_zero_weights() {
        let result = EndpointPool::new(
            EndpointPoolConfig::new(vec![
                "https://a.example.com".to_string(),
                "https://b.example.com".to_string(),
            ])
            .with_weights(vec![0.0, 0.0]),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_fractional_weights_do_not_stick_to_first_endpoint() {
        let pool = EndpointPool::new(
            EndpointPoolConfig::new(vec![
                "https://a.example.com".to_string(),
                "https://b.example.com".to_string(),
            ])
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
}

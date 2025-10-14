use anyhow::Result;
use log::{error, info};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tokio::fs;
use tokio::sync::mpsc;

const METRICS_OUTPUT_PATH: &str = "/tmp/truss_transfer_stats.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDownloadMetric {
    pub file_name: String,
    pub file_size_bytes: u64,
    pub download_time_secs: f64,
    pub download_speed_mb_s: f64,
}

#[derive(Debug, Clone)]
pub enum MetricEvent {
    FileDownload(FileDownloadMetric),
    B10fsReadSpeed(f64),
    B10fsDecision(bool),
    B10fsUsage { hot_start: bool, size: u64 },
    ManifestSize(u64),
    Success(bool),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    pub total_manifest_size_bytes: u64,
    pub total_download_time_secs: f64,
    pub total_aggregated_mb_s: Option<f64>,
    pub file_downloads: Vec<FileDownloadMetric>,
    pub b10fs_read_speed_mbps: Option<f64>,
    pub b10fs_decision_to_use: bool,
    pub b10fs_enabled: bool,
    pub b10fs_hot_starts_files: usize,
    pub b10fs_hot_starts_bytes: u64,
    pub b10fs_cold_starts_files: usize,
    pub b10fs_cold_starts_bytes: u64,
    pub success: bool,
    pub timestamp: u64,
}

impl Default for AggregatedMetrics {
    fn default() -> Self {
        Self {
            total_manifest_size_bytes: 0,
            total_download_time_secs: 0.0,
            total_aggregated_mb_s: None,
            file_downloads: Vec::new(),
            b10fs_read_speed_mbps: None,
            b10fs_enabled: *crate::constants::BASETEN_FS_ENABLED,
            b10fs_decision_to_use: false,
            b10fs_hot_starts_files: 0,
            b10fs_hot_starts_bytes: 0,
            b10fs_cold_starts_files: 0,
            b10fs_cold_starts_bytes: 0,
            success: false,
            timestamp: chrono::Utc::now().timestamp() as u64,
        }
    }
}

pub struct MetricsCollector {
    tx: Option<mpsc::UnboundedSender<MetricEvent>>,
    start_time: Instant,
    aggregator_handle: Option<tokio::task::JoinHandle<AggregatedMetrics>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let start_time = Instant::now();

        let aggregator_handle = tokio::spawn(async move { Self::aggregator_task(rx).await });

        Self {
            tx: Some(tx),
            start_time,
            aggregator_handle: Some(aggregator_handle),
        }
    }

    pub fn sender(&self) -> Option<mpsc::UnboundedSender<MetricEvent>> {
        self.tx.clone()
    }

    async fn aggregator_task(mut rx: mpsc::UnboundedReceiver<MetricEvent>) -> AggregatedMetrics {
        let mut metrics = AggregatedMetrics::default();

        while let Some(event) = rx.recv().await {
            match event {
                MetricEvent::FileDownload(file_metric) => {
                    metrics.file_downloads.push(file_metric);
                }
                MetricEvent::B10fsReadSpeed(speed) => {
                    metrics.b10fs_read_speed_mbps = Some(speed);
                }
                MetricEvent::B10fsDecision(decision) => {
                    metrics.b10fs_decision_to_use = decision;
                }
                MetricEvent::B10fsUsage { hot_start, size } => {
                    if hot_start {
                        metrics.b10fs_hot_starts_files += 1;
                        metrics.b10fs_hot_starts_bytes += size;
                    } else {
                        metrics.b10fs_cold_starts_files += 1;
                        metrics.b10fs_cold_starts_bytes += size;
                    }
                }
                MetricEvent::ManifestSize(size) => {
                    metrics.total_manifest_size_bytes = size;
                }
                MetricEvent::Success(success) => {
                    metrics.success = success;
                }
            }
        }

        metrics
    }

    async fn finalize(mut self) -> Result<()> {
        let elapsed = self.start_time.elapsed();

        // Close the sender to signal the aggregator to finish
        self.tx.take();

        // Wait for aggregator to finish processing
        let mut metrics = if let Some(handle) = self.aggregator_handle.take() {
            handle.await?
        } else {
            AggregatedMetrics::default()
        };

        // Set total time
        metrics.total_download_time_secs = elapsed.as_secs_f64();
        metrics.timestamp = chrono::Utc::now().timestamp() as u64;
        // Calculate total aggregated Mbps if there are file downloads
        if !metrics.file_downloads.is_empty() && metrics.total_download_time_secs > 0.0 {
            let total_bytes: u64 = metrics
                .file_downloads
                .iter()
                .map(|f| f.file_size_bytes)
                .sum();
            metrics.total_aggregated_mb_s =
                Some((total_bytes as f64) / (metrics.total_download_time_secs * 1_000_000.0));
        } else {
            metrics.total_aggregated_mb_s = None;
        }

        // Write to file
        let json = serde_json::to_string_pretty(&metrics)?;
        fs::write(METRICS_OUTPUT_PATH, json).await?;

        info!(
            "Metrics written to {}: {} files downloaded in {:.2}s, total size: {} bytes",
            METRICS_OUTPUT_PATH,
            metrics.file_downloads.len(),
            metrics.total_download_time_secs,
            metrics.total_manifest_size_bytes
        );

        Ok(())
    }
}

impl Drop for MetricsCollector {
    fn drop(&mut self) {
        // We can't run async code in Drop, so we spawn a blocking task
        // Note: This is a best-effort attempt. For proper cleanup, call finalize() explicitly.
        if self.aggregator_handle.is_some() {
            error!(
                "MetricsCollector dropped without calling finalize(). Metrics may be incomplete."
            );
        }
    }
}

/// RAII guard that ensures metrics are flushed on drop
pub struct MetricsGuard {
    collector: Option<MetricsCollector>,
}

impl MetricsGuard {
    pub fn new() -> Self {
        Self {
            collector: Some(MetricsCollector::new()),
        }
    }

    pub fn sender(&self) -> Option<mpsc::UnboundedSender<MetricEvent>> {
        self.collector.as_ref().and_then(|c| c.sender())
    }

    /// Explicitly finalize and flush metrics (recommended over relying on Drop)
    pub async fn finalize(mut self) -> Result<()> {
        if let Some(collector) = self.collector.take() {
            collector.finalize().await?;
        }
        Ok(())
    }
}

impl Drop for MetricsGuard {
    fn drop(&mut self) {
        if let Some(collector) = self.collector.take() {
            // Spawn a task to finalize asynchronously
            // This is a best-effort cleanup
            tokio::spawn(async move {
                if let Err(e) = collector.finalize().await {
                    error!("Failed to finalize metrics in Drop: {}", e);
                }
            });
        }
    }
}

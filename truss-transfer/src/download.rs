use std::path::Path;

use anyhow::Result;
use log::{debug, info, warn};

use crate::constants::*;
use crate::download_core::{
    check_metadata_size, download_azure_to_path, download_gcs_to_path, download_http_to_path_fast,
    download_s3_to_path,
};
use crate::metrics::{FileDownloadMetric, MetricEvent};
use std::sync::Arc;
use std::time::Instant;
use tokio::fs;
use tokio::sync::{mpsc, Semaphore};

/// Attempts to use b10cache (if enabled) to symlink the file; falls back to downloading.
/// Now handles both HTTP and GCS downloads with unified caching logic.
pub async fn download_file_with_cache(
    pointer: &crate::types::BasetenPointer,
    download_dir: &Path,
    file_name: &str,
    read_from_b10cache: bool,
    write_to_b10cache: bool,
    num_workers: usize,
    semaphore_range_dw: Arc<Semaphore>,
    metrics_sender: mpsc::UnboundedSender<MetricEvent>,
) -> Result<String> {
    let destination = download_dir.join(file_name); // if file_name is absolute, discards download_dir
    let cache_filepath = Path::new(&*CACHE_DIR).join(&pointer.hash);

    // Skip download if file exists with the expected size.
    if check_metadata_size(&destination, pointer.size).await {
        info!(
            "File {} already exists with correct size. Skipping download.",
            file_name
        );
        return Ok(file_name.to_string());
    } else if destination.exists() {
        warn!(
            "File {} exists but size mismatch. Redownloading.",
            file_name
        );
    }

    // If b10cache is enabled, try symlinking from the cache
    if read_from_b10cache {
        // Check metadata and size first
        if check_metadata_size(&cache_filepath, pointer.size).await {
            debug!(
                "Found {} in b10cache. Attempting to create symlink...",
                pointer.hash
            );
            if let Err(e) =
                crate::cache::create_symlink_or_skip(&cache_filepath, &destination, pointer.size)
                    .await
            {
                warn!(
                    "Symlink creation failed: {}.  Proceeding with direct download.",
                    e
                );
            } else {
                info!(
                    "Symlink created successfully. Skipping download for {}.",
                    file_name
                );
                // Record b10fs hot start (cache hit)
                let _ = metrics_sender.send(MetricEvent::B10fsUsage {
                    hot_start: true,
                    size: pointer.size,
                });
                return Ok(file_name.to_string());
            }
        } else if !cache_filepath.exists() {
            debug!(
                "{} not found in b10cache. Proceeding to download.",
                pointer.hash
            );
        } else {
            warn!(
                "Found {} in b10cache but size mismatch. b10cache is inconsistent. Proceeding to download.",
                pointer.hash
            );
        }
    }

    // Download the file to the local path based on resolution type
    let actual_download_start = Instant::now();
    match &pointer.resolution {
        crate::types::Resolution::Http(http_resolution) => {
            download_http_to_path_fast(
                &http_resolution.url,
                &destination,
                pointer.size,
                &pointer.runtime_secret_name,
                num_workers,
                semaphore_range_dw,
            )
            .await?;
        }
        crate::types::Resolution::Gcs(gcs_resolution) => {
            download_gcs_to_path(
                &gcs_resolution.bucket_name,
                &gcs_resolution.path,
                &destination,
                pointer.size,
                &pointer.runtime_secret_name,
            )
            .await?;
        }
        crate::types::Resolution::S3(s3_resolution) => {
            download_s3_to_path(
                &s3_resolution.bucket_name,
                &s3_resolution.key,
                &destination,
                pointer.size,
                &pointer.runtime_secret_name,
            )
            .await?;
        }
        crate::types::Resolution::Azure(azure_resolution) => {
            download_azure_to_path(
                &azure_resolution.account_name,
                &azure_resolution.container_name,
                &azure_resolution.blob_name,
                &destination,
                pointer.size,
                &pointer.runtime_secret_name,
            )
            .await?;
        }
    }

    let download_elapsed = actual_download_start.elapsed();
    let is_correct_size = check_metadata_size(&destination, pointer.size).await;

    // Record download metrics (cold start - had to download)
    let download_time_secs = download_elapsed.as_secs_f64();
    let download_speed_mb_s = if download_time_secs > 0.0 {
        (pointer.size as f64 / (1024.0 * 1024.0)) / download_time_secs
    } else {
        0.0
    };

    let _ = metrics_sender.send(MetricEvent::FileDownload(FileDownloadMetric {
        file_name: file_name.to_string(),
        file_size_bytes: pointer.size,
        download_time_secs,
        download_speed_mb_s,
    }));

    // Record b10fs cold start (cache miss - had to download)
    if read_from_b10cache {
        let _ = metrics_sender.send(MetricEvent::B10fsUsage {
            hot_start: false,
            size: pointer.size,
        });
    }

    // After the file is locally downloaded, optionally move it to b10cache.
    if write_to_b10cache && is_correct_size {
        match crate::cache::handle_write_b10cache(&destination, &cache_filepath).await {
            Ok(_) => debug!("b10cache handled successfully."),
            Err(e) => {
                // even if the handle_write_b10cache fails, we still continue.
                warn!("Failed to handle b10cache: {}", e);
            }
        }
    } else if !is_correct_size {
        warn!(
            "Downloaded file {} has incorrect size. Expected {}, got {}.",
            destination.display(),
            pointer.size,
            fs::metadata(&destination).await?.len()
        );
    }

    Ok(file_name.to_string())
}

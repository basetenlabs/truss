use std::io::Write;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

use crate::constants::{
    TRUSS_TRANSFER_RANGE_DOWNLOAD_WORKERS_PER_FILE, TRUSS_TRANSFER_USE_RANGE_DOWNLOAD,
};
use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use futures_util::StreamExt;
use log::{debug, info, warn};
use object_store::ObjectStore;
use reqwest::{Client, Url};
use std::sync::Arc;
use tokio::fs;
use tokio::sync::mpsc;
use tokio::sync::Semaphore;
use tokio::task::{self, JoinHandle};
use tokio::time::Instant;
use tokio::time::{interval, Duration};

use crate::secrets::get_hf_secret_from_file;

/// Check if file exists with expected size
pub async fn check_metadata_size(path: &Path, size: u64) -> bool {
    match fs::metadata(path).await {
        Ok(metadata) => size == metadata.len(),
        Err(_) => false, // If metadata cannot be accessed, consider it a size mismatch
    }
}

/// Sanitize a URL by removing query parameters if they exist for logging purposes.
fn sanitize_url(url: &str) -> String {
    if let Some(index) = url.find('?') {
        url[..index].to_string()
    } else {
        url.to_string()
    }
}

fn spawn_download_monitor(path: PathBuf, total_size: u64) -> JoinHandle<()> {
    tokio::spawn(async move {
        let start_time = Instant::now();
        let mut ticker = interval(Duration::from_secs(30));
        let mut last_size = 0;

        //
        ticker.tick().await;
        let mut last_tick_time = Instant::now();
        loop {
            ticker.tick().await;
            let current_size = match fs::metadata(&path).await {
                Ok(metadata) => metadata.len(),
                Err(_) => 0, // If metadata cannot be accessed, assume 0
            };
            let elapsed_secs = start_time.elapsed().as_secs_f64();
            let elapsed_since_last_tick = last_tick_time.elapsed().as_secs_f64();
            last_tick_time = Instant::now();

            if elapsed_secs > 0.0 {
                let avg_speed_mbps = (current_size as f64 / (1024.0 * 1024.0)) / elapsed_secs;
                let current_speed_mbps = if elapsed_since_last_tick > 0.0 {
                    ((current_size - last_size) as f64 / (1024.0 * 1024.0))
                        / elapsed_since_last_tick
                } else {
                    0.0
                };
                last_size = current_size;
                let progress_percent = if total_size > 0 {
                    (current_size as f64 / total_size as f64) * 100.0
                } else {
                    0.0
                };
                info!(
                    "Download for {:?} {:.2}% ({} / {} bytes, avg: {:.2} MB/s, curr: {:.2} MB/s)",
                    path,
                    progress_percent,
                    current_size,
                    total_size,
                    avg_speed_mbps,
                    current_speed_mbps
                );
            }
        }
    })
}

// RAII guard to ensure the download monitor is aborted when the guard is dropped.
struct DownloadMonitorGuard(JoinHandle<()>);

impl Drop for DownloadMonitorGuard {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Fast path for HTTP downloads using async tokio operations
/// Automatically detects HuggingFace URLs and uses hf_transfer for parallel downloads
pub async fn download_http_to_path_fast(
    url: &str,
    path: &Path,
    size: u64,
    runtime_secret_name: &str,
    num_workers: usize,
    semaphore_range_dw: Arc<Semaphore>,
) -> Result<()> {
    // Use hf_transfer for HuggingFace downloads
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await.context(format!(
            "Failed to create directory for download path: {parent:?}"
        ))?;
    }

    let sanitized_url = sanitize_url(url);
    debug!(
        "Starting HTTP download to {:?} from {}",
        path, sanitized_url
    );

    // The monitor will be automatically aborted when `_monitor_guard` goes out of scope.
    let _monitor_guard = DownloadMonitorGuard(spawn_download_monitor(path.to_path_buf(), size));
    let is_hf_url = if let Ok(parsed_url) = Url::parse(url) {
        if let Some(host) = parsed_url.host_str() {
            host == "huggingface.co"
                || host.ends_with(".huggingface.co")
                || host == "hf.co"
                || host.ends_with(".hf.co")
        } else {
            false
        }
    } else {
        false
    };

    let auth_token = if is_hf_url {
        get_hf_secret_from_file(runtime_secret_name)
    } else {
        info!("no hf token, since using {url}");
        None
    };

    if *TRUSS_TRANSFER_USE_RANGE_DOWNLOAD {
        // global concurrency
        let concurrency = *TRUSS_TRANSFER_RANGE_DOWNLOAD_WORKERS_PER_FILE;
        let result = crate::hf_transfer::download_async(
            url.to_string(),
            path.to_string_lossy().to_string(),
            concurrency, // max_files
            3,           // parallel_failures
            5,           // max_retries
            auth_token,
            None, // callback
            size,
            semaphore_range_dw,
        )
        .await;

        if let Err(e) = result {
            return Err(anyhow!("Range HTTP download failed: {}", e));
        }

        // assure that the file got flushed, without asking each file to flush it
        for i in (0..1000).rev() {
            if check_metadata_size(path, size).await {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
            if i == 0 {
                warn!(
                    "Download completed, but flush() not complete + metadata of {} not synced to disk.",
                    path.display()
                );
                // force sync
                // fs::File::open(path)
                //     .await
                //     .context(format!("Failed to open file: {:?}", path))?
                //     .sync_all()
                //     .await
                //     .context(format!("Failed to sync file: {:?}", path))?;
                // if !check_metadata_size(&path, size).await {
                //     error!(
                //         "File {} size mismatch after sync. Expected {}, got {}",
                //         path.display(),
                //         size,
                //         fs::metadata(&path).await?.len()
                //     );
                // }
            }
        }
        info!("Completed range HTTP download to {:?}", path);
    } else {
        let mut client_builder = Client::builder();
        client_builder = client_builder.http1_only();

        if num_workers >= 32 {
            debug!("Disabling proxy for reqwest client as num_workers >= 32");
            client_builder = client_builder.no_proxy();
        }
        let client = client_builder.build()?;

        let mut request_builder = client.get(url);
        if let Some(token) = auth_token {
            request_builder = request_builder.header("Authorization", format!("Bearer {token}"));
        }

        let resp = request_builder
            .send()
            .await
            .context("Failed to send HTTP request")?;

        if !resp.status().is_success() {
            return Err(anyhow!(
                "HTTP request failed with status: {} for URL: {}",
                resp.status(),
                sanitized_url
            ));
        }

        let mut file = fs::File::create(path)
            .await
            .context(format!("Failed to create file: {path:?}"))?;
        let mut stream = resp.bytes_stream();

        // Direct async write without mpsc channel
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.context("Failed to read chunk from HTTP stream")?;
            file.write_all(&chunk)
                .await
                .context("Failed to write chunk to file")?;
        }

        file.flush().await.context("Failed to flush file")?;
        let final_size = fs::metadata(path).await?.len();
        if final_size != size {
            return Err(anyhow!(
                "Downloaded file size mismatch for {:?} (expected {}, got {})",
                path,
                size,
                final_size
            ));
        }
        info!("Completed HTTP download to {:?}", path);
    }

    Ok(())
}

async fn download_from_object_store(
    storage: &dyn ObjectStore,
    object_path: &object_store::path::Path,
    local_path: &Path,
    size: u64,
    source_description: &str,
) -> Result<()> {
    if let Some(parent) = local_path.parent() {
        fs::create_dir_all(parent).await.context(format!(
            "Failed to create directory for download path: {parent:?}"
        ))?;
    }

    debug!(
        "Starting download to {:?} from {}",
        local_path, source_description
    );

    let get_result = storage
        .get(object_path)
        .await
        .map_err(|e| anyhow!("Failed to download from {}: {}", source_description, e))?;

    let mut stream = get_result.into_stream();

    let local_path_clone = local_path.to_path_buf();
    let mut file = task::spawn_blocking(move || std::fs::File::create(local_path_clone))
        .await?
        .context(format!("Failed to create file: {local_path:?}"))?;

    let _monitor_guard =
        DownloadMonitorGuard(spawn_download_monitor(local_path.to_path_buf(), size));

    // Create a channel to act as a buffer between network and disk.
    let (tx, mut rx) = mpsc::channel::<Result<Bytes, object_store::Error>>(512);

    // Writer task: receives chunks from the channel and writes them to disk.
    let source_description_clone = source_description.to_string();
    let writer_handle = task::spawn_blocking(move || {
        while let Some(chunk_result) = rx.blocking_recv() {
            let chunk = chunk_result.map_err(|e| {
                anyhow!(
                    "Error reading chunk from {}: {}",
                    source_description_clone,
                    e
                )
            })?;
            file.write_all(&chunk)
                .context("Failed to write chunk to file")?;
        }
        // Ensure all data is written to disk before the task finishes.
        file.flush()?;
        Ok::<_, anyhow::Error>(())
    });

    // Downloader task: fetches chunks from the store and sends them to the writer.
    let downloader_handle = tokio::spawn(async move {
        while let Some(chunk_result) = stream.next().await {
            if tx.send(chunk_result).await.is_err() {
                break; // Writer task terminated.
            }
        }
        Ok::<_, anyhow::Error>(())
    });

    // Wait for both tasks to complete.
    let (downloader_result, writer_result) = tokio::try_join!(downloader_handle, writer_handle)?;
    downloader_result?;
    writer_result?;

    let metadata = fs::metadata(local_path).await?;
    let written = metadata.len();
    if written != size {
        return Err(anyhow!(
            "Downloaded file size mismatch for {:?} (expected {}, got {})",
            local_path,
            size,
            written
        ));
    }

    info!("Completed download to {:?}", local_path);
    Ok(())
}

/// Pure download function for S3 objects
pub async fn download_s3_to_path(
    bucket_name: &str,
    object_key: &str,
    path: &Path,
    size: u64,
    runtime_secret_name: &str,
) -> Result<()> {
    use crate::create::aws_metadata::s3_storage;

    let s3 = s3_storage(bucket_name, runtime_secret_name)
        .map_err(|e| anyhow!("Failed to create S3 client: {}", e))?;
    let object_path = object_store::path::Path::from(object_key);
    let source_description = format!("s3://{bucket_name}/{object_key}");

    download_from_object_store(&s3, &object_path, path, size, &source_description).await
}

/// Pure download function for Azure Blob Storage
pub async fn download_azure_to_path(
    account_name: &str,
    container_name: &str,
    blob_name: &str,
    path: &Path,
    size: u64,
    runtime_secret_name: &str,
) -> Result<()> {
    use crate::create::azure_metadata::azure_storage;

    let azure = azure_storage(account_name, runtime_secret_name)
        .map_err(|e| anyhow!("Failed to create Azure client: {}", e))?;
    let object_path = object_store::path::Path::from(blob_name);
    let source_description = format!("azure://{account_name}/{container_name}/{blob_name}");

    download_from_object_store(&azure, &object_path, path, size, &source_description).await
}

/// Pure download function for GCS objects
pub async fn download_gcs_to_path(
    bucket_name: &str,
    object_key: &str,
    path: &Path,
    size: u64,
    runtime_secret_name: &str,
) -> Result<()> {
    use crate::create::gcs_metadata::gcs_storage;

    let gcs = gcs_storage(bucket_name, runtime_secret_name)
        .map_err(|e| anyhow!("Failed to create GCS client: {}", e))?;
    let object_path = object_store::path::Path::from(object_key);
    let source_description = format!("gs://{bucket_name}/{object_key}");

    download_from_object_store(&gcs, &object_path, path, size, &source_description).await
}

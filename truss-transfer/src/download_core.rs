use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use futures_util::StreamExt;
use log::{debug, info};
use object_store::ObjectStore;
use reqwest::Client;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio::time::{interval, Duration};

use crate::secrets::get_hf_secret_from_file;

/// Check if file exists with expected size
pub async fn check_metadata_size(path: &Path, size: u64) -> bool {
    match fs::metadata(path).await {
        Ok(metadata) => size == metadata.len() as u64,
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

/// Pure download function for HTTP URLs with authentication support
pub async fn download_http_to_path(
    url: &str,
    path: &Path,
    size: u64,
    runtime_secret_name: &str,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await.context(format!(
            "Failed to create directory for download path: {:?}",
            parent
        ))?;
    }

    let sanitized_url = sanitize_url(url);
    debug!(
        "Starting HTTP download to {:?} from {}",
        path, sanitized_url
    );

    let client = Client::builder().build()?;

    let mut request_builder = client.get(url);

    // Add authentication if it's a HuggingFace URL
    if url.starts_with("https://huggingface.co") {
        if let Some(token) = get_hf_secret_from_file(runtime_secret_name) {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", token));
        }
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
        .context(format!("Failed to create file: {:?}", path))?;

    let mut stream = resp.bytes_stream();

    // Start of monitoring logic
    let monitor_handle = spawn_download_monitor(path.to_path_buf(), size);
    // End of monitoring logic

    let download_result: Result<()> = async {
        while let Some(chunk_result) = stream.next().await {
            let chunk: Bytes = chunk_result.context("Failed to read chunk from HTTP stream")?;
            file.write_all(&chunk)
                .await
                .context("Failed to write chunk to file")?;
        }
        Ok(())
    }
    .await;

    // Stop the monitor task now that the download is complete or has failed.
    monitor_handle.abort();

    // Now handle the result of the download
    download_result?;

    // Ensure all data is written to disk
    file.flush().await?;
    file.sync_all().await?;

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
            "Failed to create directory for download path: {:?}",
            parent
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

    let mut file = fs::File::create(local_path)
        .await
        .context(format!("Failed to create file: {:?}", local_path))?;

    let monitor_handle = spawn_download_monitor(local_path.to_path_buf(), size);

    let download_result: Result<()> = async {
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result
                .map_err(|e| anyhow!("Error reading chunk from {}: {}", source_description, e))?;
            file.write_all(&chunk)
                .await
                .context("Failed to write chunk to file")?;
        }
        Ok(())
    }
    .await;

    monitor_handle.abort();
    download_result?;

    file.flush().await?;
    file.sync_all().await?;

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
    let source_description = format!("s3://{}/{}", bucket_name, object_key);

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
    let source_description = format!("azure://{}/{}/{}", account_name, container_name, blob_name);

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
    let source_description = format!("gs://{}/{}", bucket_name, object_key);

    download_from_object_store(&gcs, &object_path, path, size, &source_description).await
}

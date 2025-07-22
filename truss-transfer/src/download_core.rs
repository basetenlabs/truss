use std::path::Path;

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use futures_util::StreamExt;
use log::{debug, info};
use reqwest::Client;
use tokio::fs;
use tokio::io::AsyncWriteExt;

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

/// Pure download function for HTTP URLs with authentication support
pub async fn download_http_to_path(
    client: &Client,
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
    let mut downloaded: u64 = 0;

    while let Some(chunk_result) = stream.next().await {
        let chunk: Bytes = chunk_result.context("Failed to read chunk from HTTP stream")?;
        downloaded += chunk.len() as u64;
        file.write_all(&chunk)
            .await
            .context("Failed to write chunk to file")?;
    }

    // Ensure all data is written to disk
    file.flush().await?;
    file.sync_all().await?;

    if downloaded != size {
        return Err(anyhow!(
            "Downloaded file size mismatch for {:?} (expected {}, got {})",
            path,
            size,
            downloaded
        ));
    }

    info!("Completed HTTP download to {:?}", path);
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
    use object_store::ObjectStore;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await.context(format!(
            "Failed to create directory for download path: {:?}",
            parent
        ))?;
    }

    debug!(
        "Starting S3 download to {:?} from s3://{}/{}",
        path, bucket_name, object_key
    );

    let s3 = s3_storage(bucket_name, runtime_secret_name)
        .map_err(|e| anyhow!("Failed to create S3 client: {}", e))?;

    let object_path = object_store::path::Path::from(object_key);

    // Download the object
    let get_result = s3
        .get(&object_path)
        .await
        .map_err(|e| anyhow!("Failed to download from S3: {}", e))?;

    let stream = get_result.into_stream();

    let mut file = fs::File::create(path).await?;
    let mut stream = stream;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| anyhow!("Error reading chunk from S3: {}", e))?;
        file.write_all(&chunk).await?;
    }

    // Ensure all data is written to disk
    file.flush().await?;
    file.sync_all().await?;

    let metadata = fs::metadata(path).await?;
    let written = metadata.len();
    if written as u64 != size {
        return Err(anyhow!(
            "Downloaded file size mismatch for {:?} (expected {}, got {})",
            path,
            size,
            written
        ));
    }

    info!("Completed S3 download to {:?}", path);
    Ok(())
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
    use object_store::ObjectStore;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await.context(format!(
            "Failed to create directory for download path: {:?}",
            parent
        ))?;
    }

    debug!(
        "Starting Azure download to {:?} from azure://{}/{}/{}",
        path, account_name, container_name, blob_name
    );

    let azure = azure_storage(account_name, runtime_secret_name)
        .map_err(|e| anyhow!("Failed to create Azure client: {}", e))?;

    let object_path = object_store::path::Path::from(blob_name);

    // Download the blob
    let get_result = azure
        .get(&object_path)
        .await
        .map_err(|e| anyhow!("Failed to download from Azure: {}", e))?;

    let stream = get_result.into_stream();

    let mut file = fs::File::create(path).await?;
    let mut stream = stream;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| anyhow!("Error reading chunk from Azure: {}", e))?;
        file.write_all(&chunk).await?;
    }

    // Ensure all data is written to disk
    file.flush().await?;
    file.sync_all().await?;

    let metadata = fs::metadata(path).await?;
    let written = metadata.len();
    if written as u64 != size {
        return Err(anyhow!(
            "Downloaded file size mismatch for {:?} (expected {}, got {})",
            path,
            size,
            written
        ));
    }

    info!("Completed Azure download to {:?}", path);
    Ok(())
}

/// Pure download function for GCS objects
pub async fn download_gcs_to_path(
    bucket_name: &str,
    object_path: &str,
    path: &Path,
    size: u64,
    runtime_secret_name: &str,
) -> Result<()> {
    use crate::create::gcs_metadata::gcs_storage;
    use object_store::ObjectStore;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await.context(format!(
            "Failed to create directory for download path: {:?}",
            parent
        ))?;
    }

    debug!(
        "Starting GCS download to {:?} from gs://{}/{}",
        path, bucket_name, object_path
    );

    let gcs = gcs_storage(bucket_name, runtime_secret_name)
        .map_err(|e| anyhow!("Failed to create GCS client: {}", e))?;

    let object_path = object_store::path::Path::from(object_path);

    // Download the object
    let get_result = gcs
        .get(&object_path)
        .await
        .map_err(|e| anyhow!("Failed to download from GCS: {}", e))?;

    let stream = get_result.into_stream();

    let mut file = fs::File::create(path).await?;
    let mut stream = stream;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| anyhow!("Error reading chunk from GCS: {}", e))?;
        file.write_all(&chunk).await?;
    }

    // Ensure all data is written to disk
    file.flush().await?;
    file.sync_all().await?;

    let metadata = fs::metadata(path).await?;
    let written = metadata.len();
    if written as u64 != size {
        return Err(anyhow!(
            "Downloaded file size mismatch for {:?} (expected {}, got {})",
            path,
            size,
            written
        ));
    }

    info!("Completed GCS download to {:?}", path);
    Ok(())
}

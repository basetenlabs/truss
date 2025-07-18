use std::path::Path;

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use futures_util::StreamExt;
use log::{debug, info, warn};
use reqwest::Client;
use tokio::fs;
use tokio::io::AsyncWriteExt;

use crate::constants::*;

/// Check if file exists with expected size
pub async fn check_metadata_size(path: &Path, size: u64) -> bool {
    match fs::metadata(path).await {
        Ok(metadata) => size == metadata.len() as u64,
        Err(_) => false, // If metadata cannot be accessed, consider it a size mismatch
    }
}

/// Attempts to use b10cache (if enabled) to symlink the file; falls back to downloading.
pub async fn download_file_with_cache(
    client: &Client,
    url: &str,
    download_dir: &Path,
    file_name: &str,
    hash: &str,
    size: u64,
    runtime_secret_name: &str,
    read_from_b10cache: bool,
    write_to_b10cache: bool,
) -> Result<()> {
    let destination = download_dir.join(file_name); // if file_name is absolute, discards download_dir
    let cache_path = Path::new(CACHE_DIR).join(hash);

    // Skip download if file exists with the expected size.
    if check_metadata_size(&destination, size).await {
        info!(
            "File {} already exists with correct size. Skipping download.",
            file_name
        );
        return Ok(());
    } else if destination.exists() {
        warn!(
            "File {} exists but size mismatch. Redownloading.",
            file_name
        );
    }

    // If b10cache is enabled, try symlinking from the cache
    if read_from_b10cache {
        // Check metadata and size first
        if check_metadata_size(&cache_path, size).await {
            debug!(
                "Found {} in b10cache. Attempting to create symlink...",
                hash
            );
            if let Err(e) =
                crate::cache::create_symlink_or_skip(&cache_path, &destination, size).await
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
                return Ok(());
            }
        } else {
            warn!(
                "Found {} in b10cache but size mismatch. b10cache is inconsistent. Proceeding to download.",
                hash
            );
        }
    }
    // Download the file to the local path
    download_to_path(client, url, &destination, size, runtime_secret_name).await?;

    // After the file is locally downloaded, optionally move it to b10cache.
    if write_to_b10cache {
        match crate::cache::handle_write_b10cache(&destination, &cache_path).await {
            Ok(_) => debug!("b10cache handled successfully."),
            Err(e) => {
                // even if the handle_write_b10cache fails, we still continue.
                warn!("Failed to handle b10cache: {}", e);
            }
        }
    }

    Ok(())
}

/// Sanitize a URL by removing query parameters if they exist for logging purposes.
fn sanitize_url(url: &str) -> String {
    if let Some(index) = url.find('?') {
        url[..index].to_string()
    } else {
        url.to_string()
    }
}

/// Stream a download from `url` into the specified `path`.
/// Returns an error if the download fails or if the file size does not match the expected size.
pub async fn download_to_path(
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
    debug!("Starting download to {:?} from {}", path, sanitized_url);
    let mut request_builder = client.get(url);
    if url.starts_with("https://huggingface.co") {
        if let Some(token) = get_secret_from_file(runtime_secret_name) {
            request_builder = request_builder.header("Authorization", format!("Bearer {}", token));
        }
    }
    let resp = request_builder
        .send()
        .await?
        .error_for_status()
        .map_err(|e| {
            let status = e.status().map_or("unknown".into(), |s| s.to_string());
            anyhow!("HTTP status {} for url ({})", status, sanitized_url,)
        })?;
    let mut stream = resp.bytes_stream();

    let mut file = fs::File::create(path).await?;
    while let Some(chunk_result) = stream.next().await {
        let chunk: Bytes = chunk_result?;
        file.write_all(&chunk).await?;
    }
    // make sure all data is written to disk
    file.flush().await?;
    file.sync_all().await?;

    let metadata = fs::metadata(path).await?;
    let written = metadata.len();
    if written as u64 != size {
        Err(anyhow!(
            "Downloaded file size mismatch for {:?} (expected {}, got {})",
            path,
            size,
            written
        ))?;
    }

    info!("Completed download to {:?}", path);
    Ok(())
}

/// Read secret from file (e.g. for `hf_access_token` reads /secrets/hf_access_token and returns its contents)
pub fn get_secret_from_file(name: &str) -> Option<String> {
    let path = Path::new(SECRETS_BASE_PATH).join(name);
    if path.exists() {
        if let Ok(contents) = std::fs::read_to_string(&path) {
            let trimmed = contents.trim().to_string();
            if !trimmed.is_empty() {
                debug!("Found secret in token in {}", path.display());
                return Some(trimmed);
            }
        }
    }
    warn!(
        "No secret found in {path}. Using unauthenticated access. Make sure to set `{name}` in your Baseten.co secrets and add `secrets:- {name}: null` to your config.yaml.",
        path = path.display(),
        name = name
    );
    None
}

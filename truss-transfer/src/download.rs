use std::path::Path;

use anyhow::Result;
use log::{debug, info, warn};

use crate::constants::*;
use crate::download_core::{
    check_metadata_size, download_azure_to_path, download_gcs_to_path, download_http_to_path,
    download_s3_to_path,
};

/// Attempts to use b10cache (if enabled) to symlink the file; falls back to downloading.
/// Now handles both HTTP and GCS downloads with unified caching logic.
pub async fn download_file_with_cache(
    pointer: &crate::types::BasetenPointer,
    download_dir: &Path,
    file_name: &str,
    read_from_b10cache: bool,
    write_to_b10cache: bool,
) -> Result<()> {
    let destination = download_dir.join(file_name); // if file_name is absolute, discards download_dir
    let cache_path = Path::new(CACHE_DIR).join(&pointer.hash);

    // Skip download if file exists with the expected size.
    if check_metadata_size(&destination, pointer.size).await {
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
        if check_metadata_size(&cache_path, pointer.size).await {
            debug!(
                "Found {} in b10cache. Attempting to create symlink...",
                pointer.hash
            );
            if let Err(e) =
                crate::cache::create_symlink_or_skip(&cache_path, &destination, pointer.size).await
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
                pointer.hash
            );
        }
    }

    // Download the file to the local path based on resolution type
    match &pointer.resolution {
        crate::types::Resolution::Http(http_resolution) => {
            download_http_to_path(
                &http_resolution.url,
                &destination,
                pointer.size,
                &pointer.runtime_secret_name,
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

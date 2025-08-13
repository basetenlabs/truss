use std::collections::{HashMap, HashSet};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::Path;
#[cfg(windows)]
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result};
use chrono;
use fs2;
use log::{debug, info, warn};
use tokio::fs;
use tokio::io::AsyncReadExt;
use tokio::time::{interval, Duration};

use crate::constants::*;
use crate::download_core::check_metadata_size;

/// Helper function to get the file's last access time as a Unix timestamp.
/// On Unix, it uses `metadata.atime()`. On Windows, it uses `metadata.accessed()`.
fn get_atime(metadata: &std::fs::Metadata) -> std::io::Result<i64> {
    #[cfg(unix)]
    {
        Ok(metadata.atime())
    }
    #[cfg(windows)]
    {
        let accessed = metadata.accessed()?;
        let duration = accessed.duration_since(UNIX_EPOCH).unwrap_or_default();
        Ok(duration.as_secs() as i64)
    }
}

/// Cleans up cache files and calculates total cache utilization.
/// Returns a tuple: (available_volume_bytes, manifest_files_in_cache_bytes).
/// Files are cleaned up if they:
/// - Have not been accessed within the threshold AND are not in `current_hashes`
pub async fn cleanup_b10cache_and_get_space_stats(
    current_hashes: &HashSet<String>,
    manifest_hash_to_size_map: &HashMap<String, u64>,
) -> Result<(u64, u64)> {
    // Returns (available_volume_bytes, manifest_files_in_cache_bytes)
    let cleanup_threshold_hours = *TRUSS_TRANSFER_B10FS_CLEANUP_HOURS;
    let cache_dir_path = Path::new(CACHE_DIR);
    let now = chrono::Utc::now().timestamp();
    let threshold_seconds = cleanup_threshold_hours * 3600;

    let mut dir = fs::read_dir(cache_dir_path)
        .await
        .with_context(|| format!("Failed to read cache directory: {:?}", cache_dir_path))?;

    let mut total_bytes_managed_in_cache = 0u64; // All files kept in cache (manifest or not old enough)
    let mut total_files_managed_in_cache = 0usize;
    let mut manifest_files_in_cache_bytes = 0u64; // Size of files from current manifest, correctly cached

    info!(
        "Analyzing b10cache at {} with a cleanup threshold of {} hours ({} days)",
        CACHE_DIR,
        cleanup_threshold_hours,
        cleanup_threshold_hours as f64 / 24.0
    );

    while let Some(entry) = dir.next_entry().await? {
        let path = entry.path();
        // Only process files
        if path.is_file() {
            let metadata = fs::metadata(&path).await?;
            let actual_file_size = metadata.len();
            let atime = get_atime(&metadata)?;

            if let Some(file_name_hash) = path.file_name().and_then(|n| n.to_str()) {
                if let Some(expected_size) = manifest_hash_to_size_map.get(file_name_hash) {
                    // File is part of the current manifest
                    total_bytes_managed_in_cache += actual_file_size;
                    total_files_managed_in_cache += 1;
                    if actual_file_size == *expected_size {
                        manifest_files_in_cache_bytes += actual_file_size;
                    } else {
                        warn!(
                            "Cached file {} (hash: {}) has incorrect size. Expected {}, got {}. Not counting towards current manifest cache size.",
                            path.display(), file_name_hash, expected_size, actual_file_size
                        );
                        // This file, though in manifest, is corrupted in cache.
                        // It won't be deleted by age here, but download logic might replace it if it tries to use it.
                    }
                } else if now - atime > threshold_seconds as i64
                    && !current_hashes.contains(file_name_hash)
                {
                    // File is NOT in current manifest AND is older than threshold
                    info!(
                        "Deleting old cached file {} ({} bytes): not in current manifest and not accessed for over {} hours",
                        file_name_hash, actual_file_size, cleanup_threshold_hours
                    );
                    if let Err(e) = fs::remove_file(&path).await {
                        warn!("Failed to delete cached file {:?}: {}", path, e);
                    }
                } else {
                    // File is NOT in current manifest but is NOT old enough to delete OR
                    // it IS in current_hashes but somehow not in manifest_hash_to_size_map (should not happen if maps are consistent)
                    // or it was in manifest_hash_to_size_map but had wrong size (already handled above for manifest_files_in_cache_bytes)
                    if !manifest_hash_to_size_map.contains_key(file_name_hash) {
                        // only log if truly not in manifest
                        debug!(
                            "Keeping non-manifest file {} ({} bytes): last accessed {} minutes ago.",
                            file_name_hash,
                            actual_file_size,
                            (now - atime) / 60
                        );
                        total_bytes_managed_in_cache += actual_file_size;
                        total_files_managed_in_cache += 1;
                    }
                }
            }
        }
    }

    info!(
        "Cache analysis complete: {} files managed in cache totaling {:.2} GB. Correctly cached files from current manifest: {:.2} GB.",
        total_files_managed_in_cache,
        total_bytes_managed_in_cache as f64 / (1024.0 * 1024.0 * 1024.0),
        manifest_files_in_cache_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // Get available disk space for CACHE_DIR's volume
    let stats = fs2::statvfs(cache_dir_path)
        .with_context(|| format!("Failed to get volume stats for {:?}", cache_dir_path))?;
    let available_bytes = stats.available_space(); // f_bavail * f_frsize (available to non-root)

    info!(
        "Total available space on volume for {}: {:.2} GB ({} bytes)",
        CACHE_DIR,
        available_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        available_bytes
    );

    Ok((available_bytes, manifest_files_in_cache_bytes))
}

/// Handling new b10cache:
/// 1. Copy the local file (download_path) to a temporary cache file with a ".incomplete" suffix.
/// 2. Verify that the copied file's size matches the expected size.
/// - (1.), (2.), (3.) with error handling for concurrency.
/// 3. If the sizes match:
///    - Atomically rename the temporary file to the final cache path.
///    - Delete the local file (deduplicate).
///    - Create a symlink from the cache file to the original local path.
/// 4. If the sizes do not match:
///    - Delete the .incomplete file and keep the local file.
pub async fn handle_write_b10cache(download_path: &Path, cache_path: &Path) -> Result<()> {
    info!(
        "b10cache enabled: copying file from {:?} to cache and creating symlink back to {:?}",
        download_path, cache_path
    );
    let size = fs::metadata(download_path).await?.len();
    // check if cache_path exists and has the same size, concurrency with other replica.
    if cache_path.exists() {
        let cache_metadata = fs::metadata(cache_path).await?;
        if cache_metadata.len() as u64 == size {
            info!(
                "Cache file {:?} already exists with the same size. Skipping copy to b10fs.",
                cache_path
            );
            update_atime_by_reading(cache_path)
                .await
                .context("Failed to update atime for cache file")?;
            return Ok(());
        }
    }

    // Build the temporary incomplete file path.
    let mut should_copy = true;
    let incomplete_cache_path = cache_path.with_extension("incomplete");
    if incomplete_cache_path.exists() {
        // Check if the incomplete file has the expected size.
        if check_metadata_size(&incomplete_cache_path, size).await {
            should_copy = false;
        } else {
            warn!(
                "Incomplete cache file {:?} already exists. Deleting it.",
                incomplete_cache_path
            );
            match fs::remove_file(&incomplete_cache_path).await {
                Ok(_) => info!("Deleted incomplete cache file."),
                Err(e) => warn!("Failed to delete incomplete cache file: {}", e),
            }
        }
    }

    if should_copy {
        // Copy the local file to the incomplete cache file with progress monitoring
        info!(
            "Copying local file {:?} to temporary incomplete cache file {:?}",
            download_path, incomplete_cache_path
        );

        // Setup progress monitoring with atomics
        let monitor_path = incomplete_cache_path.to_path_buf();
        let monitor_handle = tokio::spawn({
            async move {
                let mut ticker = interval(Duration::from_secs(10));
                let mut last_size = 0;
                loop {
                    ticker.tick().await;
                    let current_size = match fs::metadata(&monitor_path).await {
                        Ok(metadata) => metadata.len() as u64,
                        Err(e) => {
                            warn!("Failed to read metadata for {:?}: {}", monitor_path, e);
                            continue;
                        }
                    };

                    if current_size == last_size {
                        warn!(
                            "No progress for {:?}: {} bytes written so far.",
                            monitor_path, current_size
                        );
                    } else {
                        let progress_mb = current_size as f64 / (1024.0 * 1024.0);
                        info!(
                            "Copy progress for {:?}: {:.2} MB written so far.",
                            monitor_path, progress_mb
                        );
                    }
                    last_size = current_size;
                }
            }
        });

        let copy_result = fs::copy(download_path, &incomplete_cache_path).await;

        // Stop monitoring regardless of copy result
        monitor_handle.abort();

        match copy_result {
            Ok(_) => info!("Successfully copied to incomplete cache file."),
            Err(e) => {
                warn!(
                    "Failed to copy local file to incomplete cache file: {}. Maybe b10cache has no storage or permission issues.",
                    e
                );
                return Ok(());
            }
        }
    }

    let incomplete_metadata = match fs::metadata(&incomplete_cache_path).await {
        Ok(metadata) => metadata,
        Err(e) => {
            warn!(
                "Failed to get metadata for incomplete cache file: {}. Maybe b10cache has no storage or concurrency issue.",
                e
            );
            return Ok(());
        }
    };

    if incomplete_metadata.len() as u64 != size {
        warn!(
            "Size mismatch in incomplete cache file: expected {} bytes, got {} bytes. Keeping local file and cleaning up b10cache - perhaps related to high concurrency.",
            size,
            incomplete_metadata.len()
        );
        match fs::remove_file(&incomplete_cache_path).await {
            Ok(_) => info!("Deleted incomplete cache file."),
            Err(e) => warn!("Failed to delete incomplete cache file: {}", e),
        };
        return Ok(());
    }

    // Atomically rename the incomplete file to the final cache file.
    info!(
        "Atomic rename: renaming incomplete cache file {:?} to final cache file {:?}",
        incomplete_cache_path, cache_path
    );
    fs::rename(&incomplete_cache_path, cache_path)
        .await
        .with_context(|| {
            format!(
                "Failed to atomically rename incomplete cache file {:?} to final cache file {:?}",
                incomplete_cache_path, cache_path
            )
        })?;

    // Delete the local file as its copy is now in the cache.
    info!("Deleting local file at {:?}", download_path);
    fs::remove_file(download_path)
        .await
        .with_context(|| format!("Failed to delete local file {:?}", download_path))?;

    // Create a symlink from the cache file to the original download location.
    info!(
        "Creating symlink from cache file {:?} to local file path {:?}",
        cache_path, download_path
    );
    create_symlink_or_skip(cache_path, download_path, size)
        .await
        .with_context(|| {
            format!(
                "Failed to create symlink from cache file {:?} to local file path {:?}",
                cache_path, download_path
            )
        })?;

    info!(
        "Successfully handled b10cache for file: {:?}",
        download_path
    );
    Ok(())
}

/// Verifies that the file exists and updates its atime by reading it
pub async fn update_atime_by_reading(path: &Path) -> Result<()> {
    // Open the file in read-only mode.
    let mut file = fs::File::open(path)
        .await
        .with_context(|| format!("Failed to open file {:?} for updating atime", path))?;
    let mut buffer = [0u8; 1];
    let _ = file.read(&mut buffer).await?;
    Ok(())
}

/// Create a symlink from `src` to `dst` if `dst` does not exist.
/// Returns Ok(()) if `dst` already exists.
pub async fn create_symlink_or_skip(src: &Path, dst: &Path, size: u64) -> Result<()> {
    let src_metadata = fs::metadata(src).await?;
    if src_metadata.len() as u64 != size {
        warn!(
            "File size mismatch before symlink to {:?}. Expected {}, got {}",
            dst,
            size,
            src_metadata.len()
        );
    }
    if dst.exists() {
        return Ok(());
    }
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)
            .context("Failed to create parent directory for symlink destination")?;
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(src, dst).context("Failed to create Unix symlink")?;
    }
    #[cfg(windows)]
    {
        std::os::windows::fs::symlink_file(src, dst).context("Failed to create Windows symlink")?;
    }
    update_atime_by_reading(src)
        .await
        .context("Failed to update atime after symlink")?;
    Ok(())
}

use std::collections::{HashMap, HashSet};
use std::env;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, Context, Result};
use chrono;
use futures_util::stream::{FuturesUnordered, StreamExt};
use log::{error, info, warn};
use rand;
use serde_json;
use serde_yaml;
use tokio::fs;
use tokio::sync::Semaphore;

use crate::bindings::{init_logger_once, resolve_truss_transfer_download_dir};
use crate::cache::cleanup_b10cache_and_get_space_stats;
use crate::constants::*;
use crate::download::download_file_with_cache;
use crate::speed_checks::is_b10cache_fast_heuristic;
use crate::types::{BasetenPointer, BasetenPointerManifest, Resolution};

// Global lock to serialize downloads (NOTE: this is process-local only)
// For multi-process synchronization (e.g. in a "double start" scenario),
// consider using a file-based lock instead.
static GLOBAL_DOWNLOAD_LOCK: OnceLock<Arc<Mutex<()>>> = OnceLock::new();

/// Initialize the global lock if it hasn't been initialized yet.
fn get_global_lock() -> &'static Arc<Mutex<()>> {
    GLOBAL_DOWNLOAD_LOCK.get_or_init(|| Arc::new(Mutex::new(())))
}

/// Shared entrypoint for both Python and CLI
pub fn lazy_data_resolve_entrypoint(download_dir: Option<String>) -> Result<String> {
    init_logger_once();
    let num_workers = *TRUSS_TRANSFER_NUM_WORKERS as usize;
    let download_dir = resolve_truss_transfer_download_dir(download_dir);

    // Ensure the global lock is initialized
    let lock = get_global_lock();

    info!("truss_transfer_cli, version: {}", env!("CARGO_PKG_VERSION"));
    let _guard = lock
        .lock()
        .map_err(|_| anyhow!("Global lock was poisoned"))?;
    info!("Starting downloads to: {}", download_dir);

    // Build the Tokio runtime after acquiring the lock.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("Failed to build Tokio runtime")?;

    // Run the asynchronous logic.
    rt.block_on(async { lazy_data_resolve_async(download_dir.clone().into(), num_workers).await })?;
    Ok(download_dir)
}

/// Asynchronous implementation of the lazy data resolver logic.
async fn lazy_data_resolve_async(download_dir: PathBuf, num_workers: usize) -> Result<()> {
    info!("Checking for manifest files in multiple locations...");

    let mut num_workers = num_workers;

    // 1. Check multiple manifest locations and collect all available manifests
    let mut all_manifests = Vec::new();
    let mut found_paths = Vec::new();

    for manifest_path_str in LAZY_DATA_RESOLVER_PATHS {
        let manifest_path = Path::new(manifest_path_str);
        if manifest_path.is_file() {
            info!("Found manifest file at: {}", manifest_path_str);
            found_paths.push(manifest_path_str);

            // 2. Parse YAML/JSON asynchronously
            let file_data = fs::read_to_string(manifest_path)
                .await
                .with_context(|| format!("Unable to read manifest from {}", manifest_path_str))?;

            // todo: try both JSON and YAML parsing
            // If it fails, we will try the other format
            let bptr_manifest: BasetenPointerManifest = if manifest_path_str.ends_with(".json") {
                serde_json::from_str(&file_data).with_context(|| {
                    format!("Failed to parse JSON manifest from {}", manifest_path_str)
                })?
            } else {
                serde_yaml::from_str(&file_data).with_context(|| {
                    format!("Failed to parse YAML manifest from {}", manifest_path_str)
                })?
            };

            all_manifests.push(bptr_manifest);
        }
    }

    if all_manifests.is_empty() {
        return Err(anyhow!(
            "No manifest files found at any of the following locations: {}. Please ensure at least one manifest file exists before running.",
            LAZY_DATA_RESOLVER_PATHS.join(", ")
        ));
    }

    // 3. Merge all manifests
    let merged_manifest = merge_manifests(all_manifests)?;
    info!(
        "Successfully merged {} manifests from {}. Total pointers: {}",
        found_paths.len(),
        found_paths
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        merged_manifest.pointers.len()
    );

    // 4. Validate expiration and build the resolution map
    let mut resolution_map = build_resolution_map(&merged_manifest)?;
    info!("All pointers validated successfully.");

    // 5. Check if b10cache is enabled
    let allowed_b10_cache = match env::var(BASETEN_FS_ENABLED_ENV_VAR)
        .unwrap_or_else(|_| "false".into())
        .to_lowercase()
        .as_str()
    {
        "1" | "true" => true,
        _ => false,
    };
    let mut read_from_b10cache = allowed_b10_cache;
    let mut write_to_b10cache = allowed_b10_cache;

    if allowed_b10_cache {
        info!("b10cache is enabled.");
        // create cache directory if it doesn't exist
        fs::create_dir_all(CACHE_DIR)
            .await
            .context("Failed to create b10cache directory")?;
        // shuffle the resolution map to randomize the order of downloads
        // in a multi-worker initial start scenario (cold-boost + x)
        // This is to avoid the same file being downloaded by multiple workers
        use rand::seq::SliceRandom;
        resolution_map.shuffle(&mut rand::rng());

        let current_hashes = current_hashes_from_manifest(&merged_manifest);
        let manifest_hash_to_size_map: HashMap<String, u64> = merged_manifest
            .pointers
            .iter()
            .map(|p| (p.hash.clone(), p.size))
            .collect();

        let sum_manifest_size_bytes: u64 = merged_manifest.pointers.iter().map(|p| p.size).sum();

        // Clean up cache and get space statistics
        let (available_volume_bytes, manifest_files_in_cache_bytes) =
            cleanup_b10cache_and_get_space_stats(&current_hashes, &manifest_hash_to_size_map)
                .await?;

        let additional_bytes_to_cache =
            sum_manifest_size_bytes.saturating_sub(manifest_files_in_cache_bytes);
        let min_required_headroom_bytes =
            TRUSS_TRANSFER_B10FS_MIN_REQUIRED_AVAILABLE_SPACE_GB * 1024 * 1024 * 1024;

        if available_volume_bytes > additional_bytes_to_cache + min_required_headroom_bytes {
            info!(
                "Sufficient space for b10cache write: Available on volume: {:.2}GB, Additional for current manifest: {:.2}GB, Required headroom: {}GB. Enabling write to b10cache.",
                available_volume_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                additional_bytes_to_cache as f64 / (1024.0 * 1024.0 * 1024.0),
                TRUSS_TRANSFER_B10FS_MIN_REQUIRED_AVAILABLE_SPACE_GB
            );
            // write_to_b10cache remains true (its default if allowed_b10_cache is true)
        } else {
            warn!(
                "Insufficient space for b10cache write: Available on volume: {:.2}GB, Additional for current manifest: {:.2}GB, Required headroom: {}GB. Disabling write to b10cache.",
                available_volume_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                additional_bytes_to_cache as f64 / (1024.0 * 1024.0 * 1024.0),
                TRUSS_TRANSFER_B10FS_MIN_REQUIRED_AVAILABLE_SPACE_GB
            );
            write_to_b10cache = false;
        }

        // todo: check speed of b10cache (e.g. is faster than 100MB/s)
        // and stop using b10cache if download speed is faster
        match is_b10cache_fast_heuristic(&merged_manifest).await {
            Ok(speed) => {
                if speed {
                    info!("b10cache is faster than downloading.");
                } else {
                    info!("b10cache is slower than downloading. Not reading from b10cache.");
                    read_from_b10cache = false;
                }
            }
            Err(e) => {
                warn!("Failed to check b10cache speed: {}", e);
            }
        }

        // only use at max 3 workers for b10cache, to avoid conflicts on parallel writes
        if write_to_b10cache {
            num_workers = num_workers.min(3);
        }
        info!(
            "b10cache use: Read: {}, Write: {}",
            read_from_b10cache, write_to_b10cache
        );
    } else {
        info!("b10cache is not enabled for read or write.");
    }

    // 6. Build concurrency limit
    info!("Using {} concurrent workers.", num_workers);

    let semaphore = Arc::new(Semaphore::new(num_workers));
    // resolve the gcs / s3 and pre-sign the urls
    // 6.1 TODO: create features for this to pre-sign url at runtime.

    // 7. Spawn download tasks
    info!("Spawning download tasks...");
    let mut tasks: FuturesUnordered<
        tokio::task::JoinHandle<std::result::Result<(), anyhow::Error>>,
    > = FuturesUnordered::new();
    for (file_name, pointer) in resolution_map {
        let download_dir = download_dir.clone();
        let sem_clone = semaphore.clone();
        tasks.push(tokio::spawn(async move {
            let _permit = sem_clone.acquire_owned().await;
            log::debug!("Handling file: {}", file_name);
            download_file_with_cache(
                &pointer,
                &download_dir,
                &file_name,
                read_from_b10cache,
                write_to_b10cache,
            )
            .await
        }));
    }

    // 8. Await all tasks
    info!("Awaiting completion of all download tasks...");
    while let Some(join_result) = tasks.next().await {
        match join_result {
            Ok(Ok(())) => {} // task succeeded
            Ok(Err(e)) => {
                error!("A download task failed: {}", e);
                return Err(anyhow!("Download failure: {}", e));
            }
            Err(e) => {
                error!("A Tokio task panicked: {}", e);
                return Err(anyhow!("Tokio task panicked: {}", e));
            }
        }
    }

    info!("All downloads completed successfully!");
    Ok(())
}

/// Validate expiration and build a vector of (file_name, (URL, hash, size)).
pub fn build_resolution_map(
    bptr_manifest: &BasetenPointerManifest,
) -> Result<Vec<(String, BasetenPointer)>> {
    let now = chrono::Utc::now().timestamp();
    let mut out = Vec::new();

    for bptr in &bptr_manifest.pointers {
        match &bptr.resolution {
            Resolution::Http(http_resolution) => {
                if http_resolution.expiration_timestamp < now {
                    error!(
                        "Pointer {} has expired at {}. Current time is {}. This will lead to a download failure.",
                        bptr.file_name, http_resolution.expiration_timestamp, now
                    );
                }
            }
            _ => {
                // GCS or other types do not have expiration, so we skip this check
            }
        }

        if bptr.hash.contains('/') {
            return Err(anyhow!(
                "Hash {} contains '/', which is not allowed",
                bptr.hash
            ));
        }
        out.push((bptr.file_name.clone(), bptr.clone()));
    }

    Ok(out)
}

pub fn current_hashes_from_manifest(manifest: &BasetenPointerManifest) -> HashSet<String> {
    manifest.pointers.iter().map(|p| p.hash.clone()).collect()
}

/// Merge multiple manifests into a single manifest, handling duplicate files
pub fn merge_manifests(manifests: Vec<BasetenPointerManifest>) -> Result<BasetenPointerManifest> {
    let mut merged_pointers = Vec::new();
    let mut seen_files = HashSet::new();
    let manifests_count = manifests.len();

    for manifest in manifests {
        for pointer in manifest.pointers {
            let file_key = format!("{}:{}", pointer.file_name, pointer.hash);

            if seen_files.contains(&file_key) {
                // Skip duplicate files (same file name and hash)
                continue;
            }

            // Check for conflicting files (same file name but different hash)
            let conflicting_file = merged_pointers.iter().find(|p: &&BasetenPointer| {
                p.file_name == pointer.file_name && p.hash != pointer.hash
            });

            if let Some(conflicting) = conflicting_file {
                warn!(
                    "Conflicting file found: {} has hash {} in one manifest and {} in another. Using the first one.",
                    pointer.file_name, conflicting.hash, pointer.hash
                );
                continue;
            }

            seen_files.insert(file_key);
            merged_pointers.push(pointer);
        }
    }

    info!(
        "Merged {} pointers from {} manifests",
        merged_pointers.len(),
        manifests_count
    );

    Ok(BasetenPointerManifest {
        pointers: merged_pointers,
    })
}

use std::collections::HashSet;
use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use futures_util::stream::{FuturesUnordered, StreamExt};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use reqwest::Client;
use serde::Deserialize;
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
#[cfg(windows)]
use std::time::UNIX_EPOCH;
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Semaphore;
// For logging
use env_logger::Builder;
use log::{debug, error, info, warn, LevelFilter};

// Constants
static LAZY_DATA_RESOLVER_PATH: &str = "/bptr/bptr-manifest";
static CACHE_DIR: &str = "/cache/org/artifacts/truss_transfer_managed_v1";
static BLOB_DOWNLOAD_TIMEOUT_SECS: u64 = 21600; // 6 hours
static BASETEN_FS_ENABLED_ENV_VAR: &str = "BASETEN_FS_ENABLED";
static TRUSS_TRANSFER_NUM_WORKERS_DEFAULT: usize = 64;
static TRUSS_TRANSFER_DOWNLOAD_DIR_ENV_VAR: &str = "TRUSS_TRANSFER_DOWNLOAD_DIR";
static TRUSS_TRANSFER_B10FS_CLEANUP_HOURS_ENV_VAR: &str = "TRUSS_TRANSFER_B10FS_CLEANUP_HOURS";
static TRUSS_TRANSFER_B10FS_DEFAULT_CLEANUP_HOURS: u64 = 14 * 24; // 14 days
static TRUSS_TRANSFER_DOWNLOAD_DIR_FALLBACK: &str = "/tmp/bptr-resolved";
static HF_TOKEN_PATH: &str = "/secrets/hf_access_token";

// Global lock to serialize downloads (NOTE: this is process-local only)
// For multi-process synchronization (e.g. in a “double start” scenario),
// consider using a file-based lock instead.
static GLOBAL_DOWNLOAD_LOCK: OnceLock<Arc<Mutex<()>>> = OnceLock::new();

/// Initialize the global lock if it hasn't been initialized yet.
fn get_global_lock() -> &'static Arc<Mutex<()>> {
    GLOBAL_DOWNLOAD_LOCK.get_or_init(|| Arc::new(Mutex::new(())))
}

static INIT_LOGGER: Once = Once::new();

fn init_logger_once() {
    // Initialize the logger with a default level of `info`
    INIT_LOGGER.call_once(|| {
        // Check if the environment variable "RUST_LOG" is set.
        // If not, default to "info".
        let rust_log = env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
        let level = rust_log.parse::<LevelFilter>().unwrap_or(LevelFilter::Info);

        let _ = Builder::new()
            .filter_level(level)
            .format(|buf, record| {
                // Prettier log format: [timestamp] [LEVEL] [module] message
                writeln!(
                    buf,
                    "[{}] [{:<5}] {}",
                    chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"),
                    record.level(),
                    record.args()
                )
            })
            .try_init();
    });
}

fn resolve_truss_transfer_download_dir(optional_download_dir: Option<String>) -> String {
    // Order:
    // 1. optional_download_dir, if provided
    // 2. TRUSS_TRANSFER_DOWNLOAD_DIR_ENV_VAR environment variable
    // 3. TRUSS_TRANSFER_DOWNLOAD_DIR_FALLBACK (with a warning)
    optional_download_dir
        .or_else(|| env::var(TRUSS_TRANSFER_DOWNLOAD_DIR_ENV_VAR).ok())
        .unwrap_or_else(|| {
            warn!(
                "No download directory provided. Please set `export {}=/path/to/dir` or pass it as an argument. Using fallback: {}",
                TRUSS_TRANSFER_DOWNLOAD_DIR_ENV_VAR, TRUSS_TRANSFER_DOWNLOAD_DIR_FALLBACK
            );
            TRUSS_TRANSFER_DOWNLOAD_DIR_FALLBACK.into()
        })
}

/// Corresponds to `Resolution` in the Python code
#[derive(Debug, Deserialize)]
struct Resolution {
    url: String,
    expiration_timestamp: i64,
}

/// Corresponds to `BasetenPointer` in the Python code
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct BasetenPointer {
    resolution: Resolution,
    uid: String,
    file_name: String,
    hashtype: String,
    hash: String,
    size: u64,
}

/// Corresponds to `BasetenPointerManifest` in the Python code
#[derive(Debug, Deserialize)]
struct BasetenPointerManifest {
    pointers: Vec<BasetenPointer>,
}

/// Python-callable function to read the manifest and download data.
/// By default, it will use the `TRUSS_TRANSFER_DOWNLOAD_DIR` environment variable.
#[pyfunction]
#[pyo3(signature = (download_dir=None))]
fn lazy_data_resolve(download_dir: Option<String>) -> PyResult<String> {
    lazy_data_resolve_entrypoint(download_dir).map_err(|err| PyException::new_err(err.to_string()))
}

/// Shared entrypoint for both Python and CLI
fn lazy_data_resolve_entrypoint(download_dir: Option<String>) -> Result<String> {
    init_logger_once();
    let num_workers = TRUSS_TRANSFER_NUM_WORKERS_DEFAULT;
    let download_dir = resolve_truss_transfer_download_dir(download_dir);

    // Ensure the global lock is initialized
    let lock = get_global_lock();

    info!("Acquiring global download lock...");
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
    info!(
        "Checking if manifest file exists at `{}`...",
        LAZY_DATA_RESOLVER_PATH
    );

    let mut num_workers = num_workers;

    // 1. Check if bptr-manifest file exists
    let manifest_path = Path::new(LAZY_DATA_RESOLVER_PATH);
    if !manifest_path.is_file() {
        return Err(anyhow!(
            "Manifest file not found at `{}`. Please ensure the file exists before running.",
            LAZY_DATA_RESOLVER_PATH
        ));
    }

    // 2. Parse YAML asynchronously
    info!("Manifest file found. Reading YAML...");
    let yaml_data = fs::read_to_string(manifest_path)
        .await
        .context("Unable to read YAML from bptr-manifest")?;
    let bptr_manifest: BasetenPointerManifest =
        serde_yaml::from_str(&yaml_data).context("Failed to parse Baseten pointer manifest")?;
    info!(
        "Successfully read manifest. Number of pointers: {}",
        bptr_manifest.pointers.len()
    );

    // 3. Validate expiration and build the resolution map
    let mut resolution_map = build_resolution_map(&bptr_manifest)?;
    info!("All pointers validated successfully.");

    // 4. Check if b10cache is enabled
    let allowed_b10_cache = match env::var(BASETEN_FS_ENABLED_ENV_VAR)
        .unwrap_or_else(|_| "false".into())
        .to_lowercase()
        .as_str()
    {
        "1" | "true" => true,
        _ => false,
    };
    let read_from_b10cache = allowed_b10_cache;
    let mut write_to_b10cache = allowed_b10_cache;

    if allowed_b10_cache {
        info!("b10cache is enabled.");
        // create cache directory if it doesn't exist
        fs::create_dir_all(CACHE_DIR)
            .await
            .context("Failed to create b10cache directory")?;
        // shuffle the resolution map to randomize the order of downloads
        // in a multi-worker inital start scenario (cold-boost + x)
        // This is to avoid the same file being downloaded by multiple workers
        use rand::seq::SliceRandom;
        resolution_map.shuffle(&mut rand::rng());

        // Clean up cache files that have not been accessed within the threshold
        let current_hashes = current_hashes_from_manifest(&bptr_manifest);
        let (cache_size, _) = cleanup_b10cache_and_calculate_size(&current_hashes).await?;
        // warn if cache size is over 425GB, disabling write to b10cache
        if cache_size > 425 * 1024 * 1024 * 1024 {
            warn!("b10cache size is over 425GB. Consider cleaning up the cache. Disabling write to cache.");
            write_to_b10cache = false;
        }
        // todo: check speed of b10cache (e.g. is faster than 100MB/s)
        // and stop using b10cache if download speed is faster
        match is_b10cache_fast_heuristic(&bptr_manifest).await {
            Ok(speed) => {
                if speed {
                    info!("b10cache is faster than downloading. Using b10cache: Read.");
                } else {
                    info!("b10cache is slower than downloading. Not reading from b10cache.");
                    // TODO: switch to downloading
                    // read_from_b10cache = true;
                }
            }
            Err(e) => {
                warn!("Failed to check b10cache speed: {}", e);
            }
        }

        // only use at max 2 workers for b10cache, to avoid conflicts on parallel writes
        if write_to_b10cache {
            num_workers = num_workers.min(2);
        }
        info!(
            "b10cache use: Read: {}, Write: {}",
            read_from_b10cache, write_to_b10cache
        );
    } else {
        info!("b10cache is not enabled for read or write.");
    }

    // 5. Build concurrency limit
    info!("Using {} concurrent workers.", num_workers);

    let semaphore = Arc::new(Semaphore::new(num_workers));
    let client = Client::builder()
        // https://github.com/hyperium/hyper/issues/2136#issuecomment-589488526
        .tcp_keepalive(std::time::Duration::from_secs(15))
        .http2_keep_alive_interval(Some(std::time::Duration::from_secs(15)))
        .timeout(std::time::Duration::from_secs(BLOB_DOWNLOAD_TIMEOUT_SECS))
        .build()?;

    // 6. Spawn download tasks
    info!("Spawning download tasks...");
    let mut tasks: FuturesUnordered<
        tokio::task::JoinHandle<std::result::Result<(), anyhow::Error>>,
    > = FuturesUnordered::new();
    for (file_name, (resolved_url, hash, size)) in resolution_map {
        let download_dir = download_dir.clone();
        let client = client.clone();
        let sem_clone = semaphore.clone();
        tasks.push(tokio::spawn(async move {
            let _permit = sem_clone.acquire_owned().await;
            info!("Handling file: {}", file_name);
            download_file_with_cache(
                &client,
                &resolved_url,
                &download_dir,
                &file_name,
                &hash,
                size,
                read_from_b10cache,
                write_to_b10cache,
            )
            .await
        }));
    }

    // 7. Await all tasks
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
fn build_resolution_map(
    bptr_manifest: &BasetenPointerManifest,
) -> Result<Vec<(String, (String, String, u64))>> {
    let now = chrono::Utc::now().timestamp();
    let mut out = Vec::new();

    for bptr in &bptr_manifest.pointers {
        if bptr.resolution.expiration_timestamp < now {
            return Err(anyhow!("Baseten pointer lazy data resolution has expired"));
        }
        if bptr.hash.contains('/') {
            return Err(anyhow!(
                "Hash {} contains '/', which is not allowed",
                bptr.hash
            ));
        }
        out.push((
            bptr.file_name.clone(),
            (bptr.resolution.url.clone(), bptr.hash.clone(), bptr.size),
        ));
    }

    Ok(out)
}
async fn check_metadata_size(path: &Path, size: u64) -> bool {
    match fs::metadata(path).await {
        Ok(metadata) => size == metadata.len() as u64,
        Err(_) => false, // If metadata cannot be accessed, consider it a size mismatch
    }
}

/// Attempts to use b10cache (if enabled) to symlink the file; falls back to downloading.
async fn download_file_with_cache(
    client: &Client,
    url: &str,
    download_dir: &Path,
    file_name: &str,
    hash: &str,
    size: u64,
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
            info!(
                "Found {} in b10cache. Attempting to create symlink...",
                hash
            );
            if let Err(e) = create_symlink_or_skip(&cache_path, &destination, size).await {
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
    download_to_path(client, url, &destination, size).await?;

    // After the file is locally downloaded, optionally move it to b10cache.
    if write_to_b10cache {
        match handle_write_b10cache(&destination, &cache_path).await {
            Ok(_) => info!("b10cache handled successfully."),
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
async fn download_to_path(client: &Client, url: &str, path: &Path, size: u64) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await.context(format!(
            "Failed to create directory for download path: {:?}",
            parent
        ))?;
    }

    let sanitized_url = sanitize_url(url);
    info!("Starting download to {:?} from {}", path, sanitized_url);
    let mut request_builder = client.get(url);
    if url.starts_with("https://huggingface.co") {
        if let Some(token) = get_hf_token() {
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

fn get_hf_token() -> Option<String> {
    if let Ok(env_token) = std::env::var("HF_TOKEN") {
        if !env_token.is_empty() {
            debug!("Found HF token in environment variable");
            return Some(env_token);
        }
    }
    if std::path::Path::new(HF_TOKEN_PATH).exists() {
        if let Ok(contents) = std::fs::read_to_string(HF_TOKEN_PATH) {
            let trimmed = contents.trim().to_string();
            if !trimmed.is_empty() {
                debug!("Found HF token in {}", HF_TOKEN_PATH);
                return Some(trimmed);
            }
        }
    }
    warn!(
        "No HF token found in environment variable or {}. Using unauthenticated access to download from huggingface.co. Make sure you set `hf_access_token` in your Baseten.co secrets and add `secrets:- hf_access_token: null` to your config.yaml.",
        HF_TOKEN_PATH
    );
    None
}

fn get_b10fs_cleanup_threshold_hours() -> u64 {
    env::var(TRUSS_TRANSFER_B10FS_CLEANUP_HOURS_ENV_VAR)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(TRUSS_TRANSFER_B10FS_DEFAULT_CLEANUP_HOURS)
}

fn current_hashes_from_manifest(manifest: &BasetenPointerManifest) -> HashSet<String> {
    manifest.pointers.iter().map(|p| p.hash.clone()).collect()
}

/// Helper function to get the file’s last access time as a Unix timestamp.
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

/// cleans up cache files and calculates total cache utilization.
/// Returns a tuple of (bytes_used, files_count) after cleanup.
/// Files are cleaned up if they:
/// - Have not been accessed within the threshold
/// - Their filename (assumed to be the file's hash) is not in `current_hashes`
pub async fn cleanup_b10cache_and_calculate_size(
    current_hashes: &HashSet<String>,
) -> Result<(u64, usize)> {
    let cleanup_threshold_hours = get_b10fs_cleanup_threshold_hours();
    let cache_dir = Path::new(CACHE_DIR);
    let now = chrono::Utc::now().timestamp();
    let threshold_seconds = cleanup_threshold_hours * 3600;

    let mut dir = fs::read_dir(cache_dir).await?;

    let mut total_bytes = 0u64;
    let mut total_files = 0usize;

    info!(
        "Analyzing b10cache with a threshold of {} hours ({} days)",
        cleanup_threshold_hours,
        cleanup_threshold_hours as f64 / 24.0
    );

    while let Some(entry) = dir.next_entry().await? {
        let path = entry.path();
        // Only process files
        if path.is_file() {
            let metadata = fs::metadata(&path).await?;
            let file_size = metadata.len();
            let atime = get_atime(&metadata)?;

            if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                if now - atime > threshold_seconds as i64 && !current_hashes.contains(file_name) {
                    info!(
                        "Deleting cached file {} ({} bytes): not accessed for over {} hours",
                        file_name, file_size, cleanup_threshold_hours
                    );
                    fs::remove_file(&path).await?;
                } else {
                    info!(
                        "Keeping file {} ({} bytes): last accessed {} minutes ago",
                        file_name,
                        file_size,
                        (now - atime) / 60
                    );
                    total_bytes += file_size;
                    total_files += 1;
                }
            }
        }
    }

    info!(
        "Cache utilization after cleanup: {} files using {} bytes ({:.2} GB)",
        total_files,
        total_bytes,
        total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    Ok((total_bytes, total_files))
}
/// handling new b10cache:
/// 1. Copy the local file (download_path) to a temporary cache file with a “.incomplete” suffix.
/// 2. Verify that the copied file’s size matches the expected size.
/// - (1.), (2.), (3.) with error handling for concurrency.
/// 3. If the sizes match:
///    - Atomically rename the temporary file to the final cache path.
///    - Delete the local file (deduplicate).
///    - Create a symlink from the cache file to the original local path.
/// 4. If the sizes do not match:
///    - Delete the .incomplete file and keep the local file.
async fn handle_write_b10cache(download_path: &Path, cache_path: &Path) -> Result<()> {
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
        // Copy the local file to the incomplete cache file.
        info!(
            "Copying local file {:?} to temporary incomplete cache file {:?}",
            download_path, incomplete_cache_path
        );
        match fs::copy(download_path, &incomplete_cache_path).await {
            Ok(_) => info!("Successfully copied to incomplete cache file."),
            Err(e) => {
                warn!("Failed to copy local file to incomplete cache file: {}. Maybe b10cache has no storage.", e);
                return Ok(());
            }
        }
    }

    let incomplete_metadata = match fs::metadata(&incomplete_cache_path).await {
        Ok(metadata) => metadata,
        Err(e) => {
            warn!("Failed to get metadata for incomplete cache file: {}. Maybe b10cache has no storage or concurrency issue.", e);
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

/// Heuristic: Check if b10cache is faster than downloading by reading the first 128MB of a file in the cache.
/// If the read speed is greater than e.g. 114MB/s, it returns true.
/// If no file in the cache is larger than 128MB, it returns true.
/// Otherwise, it returns false.
async fn is_b10cache_fast_heuristic(manifest: &BasetenPointerManifest) -> Result<bool> {
    let benchmark_size: usize = 128 * 1024 * 1024; // 128MB
    // random number, uniform between 25 and 250 MB/s as a threshold
    let desired_speed: f64 = 25.0 + rand::random::<f64>() * (225.0);

    for bptr in &manifest.pointers {
        let cache_path = Path::new(CACHE_DIR).join(&bptr.hash);

        if bptr.size > benchmark_size as u64 && cache_path.exists() {
            let metadata = fs::metadata(&cache_path).await?;
            let file_size = metadata.len();
            if file_size == bptr.size as u64 {
                let mut file = fs::File::open(&cache_path)
                    .await
                    .with_context(|| format!("Failed to open file {:?}", cache_path))?;
                // benchmark, read 100MB
                let mut buffer = vec![0u8; benchmark_size]; // 100MB buffer
                let start_time = std::time::Instant::now();
                let bytes_read = file.read_exact(&mut buffer).await;
                let elapsed_time = start_time.elapsed();
                if bytes_read.is_ok() {
                    let elapsed_secs = elapsed_time.as_secs_f64();
                    let speed = (buffer.len() as f64 / 1024.0 / 1024.0) / elapsed_secs; // MB/s
                    info!("Read speed of b10cache approx. {:.2} MB/s", speed);
                    if speed > desired_speed {
                        return Ok(true); // Use b10cache
                    } else {
                        return Ok(false); // Don't use b10cache
                    }
                } else {
                    // If reading fails, log the error and continue
                    warn!(
                        "Failed to read file {:?}: {}",
                        cache_path,
                        bytes_read.unwrap_err()
                    );
                }
            }
        }
    }
    // no file > 512MB found in cache
    return Ok(true);
}

// verifies that the file exists and updates its atime by reading it
async fn update_atime_by_reading(path: &Path) -> Result<()> {
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
async fn create_symlink_or_skip(src: &Path, dst: &Path, size: u64) -> Result<()> {
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

/// Running the CLI directly.
#[cfg(feature = "cli")]
fn main() -> anyhow::Result<()> {
    init_logger_once();
    info!("truss_transfer_cli, version: {}", env!("CARGO_PKG_VERSION"));

    // Pass the first CLI argument as the download directory, if provided.
    let download_dir = std::env::args().nth(1);
    if let Err(e) = lazy_data_resolve_entrypoint(download_dir) {
        error!("Error during execution: {}", e);
        std::process::exit(1);
    }
    Ok(())
}

/// Python module definition
#[pymodule]
fn truss_transfer(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lazy_data_resolve, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_resolve_truss_transfer_download_dir_with_arg() {
        // If an argument is provided, it should take precedence.
        let dir = "my/download/dir".to_string();
        let result = resolve_truss_transfer_download_dir(Some(dir.clone()));
        assert_eq!(result, dir);
    }

    #[test]
    fn test_resolve_truss_transfer_download_dir_from_env() {
        // Set the environment variable and ensure it is used.
        let test_dir = "env_download_dir".to_string();
        env::set_var(TRUSS_TRANSFER_DOWNLOAD_DIR_ENV_VAR, test_dir.clone());
        let result = resolve_truss_transfer_download_dir(None);
        assert_eq!(result, test_dir);
        env::remove_var(TRUSS_TRANSFER_DOWNLOAD_DIR_ENV_VAR);
    }

    #[test]
    fn test_resolve_truss_transfer_download_dir_fallback() {
        // Ensure that when no arg and no env var are provided, the fallback is returned.
        env::remove_var(TRUSS_TRANSFER_DOWNLOAD_DIR_ENV_VAR);
        let result = resolve_truss_transfer_download_dir(None);
        assert_eq!(result, TRUSS_TRANSFER_DOWNLOAD_DIR_FALLBACK.to_string());
    }

    #[test]
    fn test_build_resolution_map_valid() {
        // Create a pointer with an expiration timestamp in the future.
        let future_timestamp = chrono::Utc::now().timestamp() + 3600; // one hour in the future
        let pointer = BasetenPointer {
            resolution: Resolution {
                url: "http://example.com/file".into(),
                expiration_timestamp: future_timestamp,
            },
            uid: "123".into(),
            file_name: "file.txt".into(),
            hashtype: "sha256".into(),
            hash: "abcdef".into(),
            size: 1024,
        };
        let manifest = BasetenPointerManifest {
            pointers: vec![pointer],
        };
        let result = build_resolution_map(&manifest);
        assert!(result.is_ok());
        let map = result.unwrap();
        assert_eq!(map.len(), 1);
        assert_eq!(map[0].0, "file.txt");
        assert_eq!(map[0].1 .0, "http://example.com/file");
        assert_eq!(map[0].1 .1, "abcdef");
        assert_eq!(map[0].1 .2, 1024);
    }

    #[test]
    fn test_build_resolution_map_expired() {
        // Create a pointer that has already expired.
        let past_timestamp = chrono::Utc::now().timestamp() - 3600; // one hour in the past
        let pointer = BasetenPointer {
            resolution: Resolution {
                url: "http://example.com/file".into(),
                expiration_timestamp: past_timestamp,
            },
            uid: "123".into(),
            file_name: "file.txt".into(),
            hashtype: "sha256".into(),
            hash: "abcdef".into(),
            size: 1024,
        };
        let manifest = BasetenPointerManifest {
            pointers: vec![pointer],
        };
        let result = build_resolution_map(&manifest);
        assert!(result.is_err());
    }

    #[test]
    fn test_init_logger_once() {
        // Calling init_logger_once multiple times should not panic.
        init_logger_once();
        init_logger_once();
    }
}

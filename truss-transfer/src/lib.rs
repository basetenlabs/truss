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
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;

// For logging
use env_logger::Builder;
use log::{debug, error, info, warn, LevelFilter};

// Constants
static LAZY_DATA_RESOLVER_PATH: &str = "/bptr/bptr-manifest";
static CACHE_DIR: &str = "/cache/org/artifacts";
static BLOB_DOWNLOAD_TIMEOUT_SECS: u64 = 21600; // 6 hours
static BASETEN_FS_ENABLED_ENV_VAR: &str = "BASETEN_FS_ENABLED";
static TRUSS_TRANSFER_NUM_WORKERS_DEFAULT: usize = 64;
static TRUSS_TRANSFER_DOWNLOAD_DIR_ENV_VAR: &str = "TRUSS_TRANSFER_DOWNLOAD_DIR";
static TRUSS_TRANSFER_DOWNLOAD_DIR_FALLBACK: &str = "/tmp/bptr-resolved";

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
    size: i64,
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
    let resolution_map = build_resolution_map(&bptr_manifest)?;
    info!("All pointers validated successfully.");

    // 4. Check if b10cache is enabled
    let uses_b10_cache = match env::var(BASETEN_FS_ENABLED_ENV_VAR)
        .unwrap_or_else(|_| "false".into())
        .to_lowercase()
        .as_str()
    {
        "1" | "true" => true,
        _ => false,
    };
    info!(
        "b10cache enabled: {}",
        if uses_b10_cache { "True" } else { "False" }
    );

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
    let mut tasks = FuturesUnordered::new();
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
                uses_b10_cache,
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
) -> Result<Vec<(String, (String, String, i64))>> {
    let now = chrono::Utc::now().timestamp();
    let mut out = Vec::new();

    for bptr in &bptr_manifest.pointers {
        if bptr.resolution.expiration_timestamp < now {
            return Err(anyhow!("Baseten pointer lazy data resolution has expired"));
        }
        out.push((
            bptr.file_name.clone(),
            (bptr.resolution.url.clone(), bptr.hash.clone(), bptr.size),
        ));
    }

    Ok(out)
}

/// Attempts to use b10cache (if enabled) to symlink the file; falls back to downloading.
async fn download_file_with_cache(
    client: &Client,
    url: &str,
    download_dir: &Path,
    file_name: &str,
    hash: &str,
    size: i64,
    uses_b10_cache: bool,
) -> Result<()> {
    let destination = download_dir.join(file_name);

    // Skip download if file exists with the expected size.
    if destination.exists() {
        if let Ok(metadata) = fs::metadata(&destination).await {
            if metadata.len() as i64 == size {
                info!(
                    "File {} already exists with correct size. Skipping download.",
                    file_name
                );
                return Ok(());
            } else {
                info!(
                    "File {} exists but size mismatch. Redownloading.",
                    file_name
                );
            }
        }
    }

    // If b10cache is enabled, try symlinking from the cache

    if uses_b10_cache {
        let cache_path = Path::new(CACHE_DIR).join(hash);
        if fs::metadata(&cache_path).await.is_ok() {
            info!(
                "Found {} in b10cache. Attempting to create symlink...",
                hash
            );
            if let Err(e) = create_symlink_or_skip(&cache_path, &destination) {
                debug!(
                    "Symlink creation failed: {}. Proceeding with direct download.",
                    e
                );
            } else {
                info!(
                    "Symlink created successfully. Skipping download for {}.",
                    file_name
                );
                return Ok(());
            }
        }
    }

    if uses_b10_cache {
        let cache_path = Path::new(CACHE_DIR).join(hash);
        info!("Downloading file to cache path: {:?}", cache_path);
        if let Err(e) = download_to_path(client, url, &cache_path, size).await {
            info!(
                "Download to b10cache failed ({}). Falling back to direct download.",
                e
            );
            download_to_path(client, url, &destination, size).await?;
        } else {
            info!("Download to b10cache succeeded. Creating symlink to final destination.");
            if let Err(e) = create_symlink_or_skip(&cache_path, &destination) {
                warn!(
                    "Symlink creation failed: {}. Falling back to direct download.",
                    e
                );
                download_to_path(client, url, &destination, size).await?;
            }
        }
    } else {
        info!(
            "b10cache not enabled. Downloading file directly to {:?}",
            destination
        );
        download_to_path(client, url, &destination, size).await?;
    }

    Ok(())
}

/// Stream a download from `url` into the specified `path`.
async fn download_to_path(client: &Client, url: &str, path: &Path, size: i64) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .await
            .context("Failed to create parent directory for download path")?;
    }

    info!("Starting download to {:?}", path);
    let resp = client.get(url).send().await?.error_for_status()?;
    let mut stream = resp.bytes_stream();

    let mut file = fs::File::create(path).await?;
    while let Some(chunk_result) = stream.next().await {
        let chunk: Bytes = chunk_result?;
        file.write_all(&chunk).await?;
    }

    if size > 0 {
        let metadata = fs::metadata(path).await?;
        let written = metadata.len();
        if written as i64 != size {
            warn!(
                "Downloaded file size mismatch for {:?} (expected {}, got {})",
                path, size, written
            );
        } else {
            info!("Download size verified: {} bytes", size);
        }
    }

    info!("Completed download to {:?}", path);
    Ok(())
}

/// Create a symlink from `src` to `dst` if `dst` does not exist.
/// Returns Ok(()) if `dst` already exists.
fn create_symlink_or_skip(src: &Path, dst: &Path) -> Result<()> {
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

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
    if uses_b10_cache {
        info!("b10cache is enabled.");
        // create cache directory if it doesn't exist
        fs::create_dir_all(CACHE_DIR)
            .await
            .context("Failed to create b10cache directory")?;
    } else {
        info!("b10cache is not enabled.");
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
) -> Result<Vec<(String, (String, String, u64))>> {
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
    uses_b10_cache: bool,
) -> Result<()> {
    let destination = download_dir.join(file_name);
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
    if uses_b10_cache {
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
    if uses_b10_cache {
        handle_b10cache(&destination, &cache_path, size).await?;
    }

    Ok(())
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

    info!("Starting download to {:?}", path);
    let resp = client.get(url).send().await?.error_for_status()?;
    let mut stream = resp.bytes_stream();

    let mut file = fs::File::create(path).await?;
    while let Some(chunk_result) = stream.next().await {
        let chunk: Bytes = chunk_result?;
        file.write_all(&chunk).await?;
    }

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

// Logic for leaving the behind in b10cache for subsequent calls by other processes getting a cache hit.
// 1. Move the file to b10cache, symlinking back to the original location.
// 2. If the move fails due to cross-device link error (EEXDEV), copy the file instead.
// Should be a no-op/optional call from perspective of the caller, a file will be
// at `download_path` and optionally in `cache_path` if b10cache is enabled.
async fn handle_b10cache(download_path: &Path, cache_path: &Path, size: u64) -> Result<()> {
    info!(
        "b10cache enabled: moving file to {:?} and creating symlink back to {:?}",
        cache_path, download_path
    );

    // can be e.g. out of memory, permission denied, etc.
    if let Err(e) = fs::rename(download_path, cache_path).await {
        // handle cross-device link error (EEXDEV) by falling back to copy
        if let Some(18) = e.raw_os_error() {
            warn!("Cross-device link error (EEXDEV). Attempting copy fallback.");
            if let Err(copy_err) = fs::copy(download_path, cache_path).await {
                warn!(
                    "Failed to copy file to b10cache: {}. Keeping local copy.",
                    copy_err
                );
            } else {
                info!("File copied to b10cache successfully.");
            }
        } else {
            warn!("Failed to move file to b10cache: {}. Keeping local copy and not populating b10cache.", e);
        }
    } else {
        match create_symlink_or_skip(cache_path, download_path, size).await {
            Ok(()) => info!("Symlink from b10cache created successfully."),
            Err(e) => {
                return Err(e);
            }
        }
    }

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

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use chrono::Utc;
use futures_util::stream::{FuturesUnordered, StreamExt}; // <-- from futures-util
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use reqwest::Client;
use serde::Deserialize;
use tokio::fs as async_fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;

// Constants
static LAZY_DATA_RESOLVER_PATH: &str = "/bptr/bptr-manifest";
static CACHE_DIR: &str = "/cache/org/artifacts";
static BLOB_DOWNLOAD_TIMEOUT_SECS: u64 = 21600; // 6 hours
static BASETEN_FS_ENABLED_ENV_VAR: &str = "BASETEN_FS_ENABLED";
static TRUSS_TRANSFER_NUM_WORKERS_DEFAULT: usize = 64;
static TRUSS_TRANSFER_DOWNLOAD_DIR_ENV_VAR: &str = "TRUSS_TRANSFER_DOWNLOAD_DIR";
static TRUSS_TRANSFER_DOWNLOAD_DIR_FALLBACK: &str = "/tmp/bptr-resolved";

// Global lock to serialize downloads
static GLOBAL_DOWNLOAD_LOCK: OnceLock<Arc<Mutex<()>>> = OnceLock::new();

/// Initialize the global lock if it hasn't been initialized yet.
fn get_global_lock() -> &'static Arc<Mutex<()>> {
    GLOBAL_DOWNLOAD_LOCK.get_or_init(|| Arc::new(Mutex::new(())))
}

fn resolve_truss_transfer_download_dir(optional_download_dir: Option<String>) -> String {
    // Order:
    // 1. optional_download_dir, if provided
    // 2. TRUSS_TRANSFER_DOWNLOAD_DIR_ENV_VAR
    // 3. TRUSS_TRANSFER_DOWNLOAD_DIR_FALLBACK and print a warning
    optional_download_dir
        .or_else(|| env::var(TRUSS_TRANSFER_DOWNLOAD_DIR_ENV_VAR).ok())
        .unwrap_or_else(|| {
            println!(
                "[WARN] No download directory provided. Please set `export {}=/path/to/dir` or pass it as an argument. Using fallback: {}",
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
    lazy_data_resolve_entrypoint(download_dir)
        .map(|resolved_dir| resolved_dir)
        .map_err(|err| PyException::new_err(err.to_string()))
}

/// Shared entrypoint for both Python and CLI
fn lazy_data_resolve_entrypoint(download_dir: Option<String>) -> Result<String> {
    let num_workers = TRUSS_TRANSFER_NUM_WORKERS_DEFAULT;

    let download_dir = resolve_truss_transfer_download_dir(download_dir);

    // Ensure the global lock is initialized
    let lock = get_global_lock();

    // Acquire the global lock, will be dropped when guard goes out of scope (also in case of errors)
    println!("[INFO] Acquiring global download lock...");
    let _guard = lock.lock().unwrap();

    // Build the runtime after acquiring the lock
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("Failed to build Tokio runtime")?;

    // Run the async logic within the runtime
    rt.block_on(async { lazy_data_resolve_async(download_dir.clone().into(), num_workers).await })?;
    Ok(download_dir)
}

/// Asynchronous implementation of the lazy data resolver logic.
async fn lazy_data_resolve_async(download_dir: PathBuf, num_workers: usize) -> Result<()> {
    println!(
        "[INFO] Checking if manifest file at `{}` exists...",
        LAZY_DATA_RESOLVER_PATH
    );

    // 1. Check if bptr-manifest file exists
    let manifest_path = Path::new(LAZY_DATA_RESOLVER_PATH);
    if !manifest_path.is_file() {
        println!("[INFO] Manifest file not found, nothing to resolve.");
        return Ok(());
    }

    // 2. Parse YAML
    println!("[INFO] Found manifest file. Reading YAML...");
    let yaml_data =
        fs::read_to_string(manifest_path).context("Unable to read YAML from bptr-manifest")?;
    let bptr_manifest: BasetenPointerManifest =
        serde_yaml::from_str(&yaml_data).context("Failed to parse Baseten pointer manifest")?;
    println!(
        "[INFO] Successfully read manifest. Number of pointers: {}",
        bptr_manifest.pointers.len()
    );

    // 3. Validate expiration and build the resolution map
    println!("[INFO] Validating pointers...");
    let resolution_map = build_resolution_map(&bptr_manifest)?;
    println!("[INFO] All pointers validated OK.");

    // 4. Check if Baseten FS cache is enabled
    let uses_b10_cache =
        env::var(BASETEN_FS_ENABLED_ENV_VAR).unwrap_or_else(|_| "False".into()) == "True";
    println!(
        "[INFO] Baseten FS cache enabled: {}",
        if uses_b10_cache { "Yes" } else { "No" }
    );

    // 5. Build concurrency limit
    println!("[INFO] Using {num_workers} concurrent workers.");
    let semaphore = Arc::new(Semaphore::new(num_workers));
    let client = Client::builder()
        // https://github.com/hyperium/hyper/issues/2136#issuecomment-589488526
        .tcp_keepalive(std::time::Duration::from_secs(15))
        .http2_keep_alive_interval(Some(std::time::Duration::from_secs(15)))
        .timeout(std::time::Duration::from_secs(BLOB_DOWNLOAD_TIMEOUT_SECS))
        .build()?;

    // 6. Spawn tasks
    println!("[INFO] Spawning download tasks...");
    let mut tasks = FuturesUnordered::new();
    for (file_name, (resolved_url, hash, size)) in resolution_map {
        let download_dir = download_dir.clone();
        let client = client.clone();
        let sem_clone = semaphore.clone();
        let uses_b10_cache = uses_b10_cache;
        tasks.push(tokio::spawn(async move {
            let _permit = sem_clone.acquire_owned().await;
            // Log which file is being downloaded
            println!("[INFO] Now handling file: {file_name}");
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
    println!("[INFO] Waiting for all download tasks to complete...");
    while let Some(join_result) = tasks.next().await {
        match join_result {
            Ok(Ok(())) => {
                // success
            }
            Ok(Err(e)) => {
                println!("[ERROR] A download failed: {e}");
                return Err(anyhow!("Download failure: {e}"));
            }
            Err(e) => {
                println!("[ERROR] A Tokio task panicked: {e}");
                return Err(anyhow!("Tokio task panicked: {e}"));
            }
        }
    }

    println!("[INFO] All downloads completed successfully!");
    Ok(())
}

/// Validate expiration and build vector of (file_name -> (URL, hash, size))
fn build_resolution_map(
    bptr_manifest: &BasetenPointerManifest,
) -> Result<Vec<(String, (String, String, i64))>> {
    let now = Utc::now().timestamp();
    let mut out = vec![];

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

/// Attempts to find file in cache (if enabled), symlink it, or else downloads it.
/// Fallback to direct download if caching fails.
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

    // If the file already exists, check if it's the correct size, and skip download if so
    if destination.exists() {
        let metadata = fs::metadata(&destination)?;
        if metadata.len() as i64 == size {
            println!(
                "[INFO] File {file_name} already exists with correct size. Skipping download."
            );
            return Ok(());
        } else {
            println!("[INFO] File {file_name} exists but size mismatch. Redownloading.");
        }
    }

    // If Baseten FS cache is enabled, try symlinking from the cache
    if uses_b10_cache {
        let cache_path = Path::new(CACHE_DIR).join(hash);
        if cache_path.exists() {
            println!("[INFO] Found {hash} in cache. Attempting to symlink...");
            if let Err(e) = create_symlink_or_skip(&cache_path, &destination) {
                println!("[DEBUG] Symlink from cache failed: {e}");
            } else {
                println!("[INFO] Symlink successful, skipping download.");
                // If we succeeded in symlinking, we can stop here
                return Ok(());
            }
        }
    }

    // If we reach here, we must actually download
    if uses_b10_cache {
        // Attempt to download to the cache, then symlink
        let cache_path = Path::new(CACHE_DIR).join(hash);
        println!("[INFO] Downloading file to cache path: {:?}", cache_path);
        if let Err(e) = download_to_path(client, url, &cache_path, size).await {
            println!("[DEBUG] Download to cache failed ({e}). Falling back to direct path.");
            // fallback
            download_to_path(client, url, &destination, size).await?;
        } else {
            // success in caching => symlink to final dest
            println!("[INFO] Download to cache successful, creating symlink to final destination.");
            if let Err(e) = create_symlink_or_skip(&cache_path, &destination) {
                println!("[DEBUG] Symlink to data dir failed: {e}");
            }
        }
    } else {
        // No caching => direct download
        println!(
            "[INFO] No caching enabled. Downloading file directly: {:?}",
            destination
        );
        download_to_path(client, url, &destination, size).await?;
    }

    Ok(())
}

/// Streaming download from `url` → `path`
async fn download_to_path(client: &Client, url: &str, path: &Path, size: i64) -> Result<()> {
    // Create parent dirs
    if let Some(parent) = path.parent() {
        async_fs::create_dir_all(parent).await?;
    }

    println!("[INFO] Starting download to {:?}", path);

    // Start the request
    let resp = client.get(url).send().await?.error_for_status()?;
    // resp.bytes_stream() => Stream of Result<Bytes, reqwest::Error>
    let mut stream = resp.bytes_stream();

    let mut file = async_fs::File::create(path).await?;

    while let Some(chunk_result) = stream.next().await {
        // chunk_result is Result<Bytes, reqwest::Error>
        let chunk: Bytes = chunk_result?;
        // Write to disk
        file.write_all(&chunk).await?;
    }

    // Optional size check
    if size > 0 {
        let written = file.metadata().await?.len();
        if written as i64 != size {
            eprintln!(
                "Warning: downloaded file size mismatch (expected {}, got {}) at {:?}",
                size, written, path
            );
        } else {
            println!("[INFO] Download size matches expected size: {size} bytes.");
        }
    }

    println!("[INFO] Completed download to {:?}", path);
    Ok(())
}

fn create_symlink_or_skip(src: &Path, dst: &Path) -> Result<()> {
    if dst.exists() {
        return Ok(());
    }
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)
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

/// running the CLI directly.
#[cfg(feature = "cli")]
fn main() -> anyhow::Result<()> {
    println!(
        "[INFO] truss_transfer_cli, version: {}",
        env!("CARGO_PKG_VERSION")
    );

    let download_dir = std::env::args().nth(1);

    let _ = lazy_data_resolve_entrypoint(download_dir.into());
    Ok(())
}

/// Python module definition
#[pymodule]
fn truss_transfer(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lazy_data_resolve, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

// Constants used throughout the truss_transfer crate
use once_cell::sync::Lazy;
use std::env;

fn is_truthy(value: &str) -> bool {
    let lower = value.to_lowercase();
    lower == "true" || lower == "1" || lower == "yes" || lower == "y"
}

/// Alternative manifest paths to check
pub static LAZY_DATA_RESOLVER_PATHS: &[&str] = &[
    "/bptr/bptr-manifest",
    "/bptr/bptr-manifest.json",
    "/static-bptr/static-bptr-manifest.json",
];

/// Cache directory for b10fs

pub static CACHE_DIR: Lazy<String> = Lazy::new(|| {
    env::var("TRUSS_TRANSFER_CACHE_DIR")
        .unwrap_or_else(|_| "/cache/org/artifacts/truss_transfer_managed_v1".to_string())
});
pub static TRUSS_TRANSFER_LOG: Lazy<String> = Lazy::new(|| {
    env::var("TRUSS_TRANSFER_LOG")
        .or_else(|_| env::var("RUST_LOG"))
        .unwrap_or_else(|_| "info".to_string())
});

/// Environment variable to enable Baseten FS
pub static BASETEN_FS_ENABLED: Lazy<bool> = Lazy::new(|| {
    env::var("BASETEN_FS_ENABLED")
        .ok()
        .map(|s| is_truthy(&s))
        .unwrap_or(false)
});

pub static HF_TOKEN: Lazy<Option<String>> = Lazy::new(|| {
    env::var("HF_TOKEN")
        .or_else(|_| env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
});

/// Number of download workers, initialized from the `TRUSS_TRANSFER_NUM_WORKERS`
/// environment variable, with a default of 64.
pub static TRUSS_TRANSFER_NUM_WORKERS: Lazy<u8> = Lazy::new(|| {
    env::var("TRUSS_TRANSFER_NUM_WORKERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6)
});

/// Environment variable for download directory

/// Cleanup hours for b10fs, initialized from the `TRUSS_TRANSFER_B10FS_CLEANUP_HOURS`
/// environment variable, with a default of 96 hours (4 days).
pub static TRUSS_TRANSFER_B10FS_CLEANUP_HOURS: Lazy<u64> = Lazy::new(|| {
    env::var("TRUSS_TRANSFER_B10FS_CLEANUP_HOURS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4 * 24)
});

pub static TRUSS_TRANSFER_PAGE_AFTER_DOWNLOAD: Lazy<bool> = Lazy::new(|| {
    env::var("TRUSS_TRANSFER_PAGE_AFTER_DOWNLOAD")
        .ok()
        .map(|s| is_truthy(&s))
        .unwrap_or(false)
});

pub static TRUSS_TRANSFER_USE_RANGE_DOWNLOAD: Lazy<bool> = Lazy::new(|| {
    env::var("TRUSS_TRANSFER_USE_RANGE_DOWNLOAD")
        .ok()
        .map(|s| is_truthy(&s))
        .unwrap_or(false)
});

pub static TRUSS_TRANSFER_RANGE_DOWNLOAD_WORKERS: Lazy<usize> = Lazy::new(|| {
    env::var("TRUSS_TRANSFER_RANGE_DOWNLOAD_WORKERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(192)
});

pub static TRUSS_TRANSFER_RANGE_DOWNLOAD_WORKERS_PER_FILE: Lazy<usize> = Lazy::new(|| {
    env::var("TRUSS_TRANSFER_RANGE_DOWNLOAD_WORKERS_PER_FILE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(84)
});

pub static TRUSS_TRANSFER_DOWNLOAD_DIR: Lazy<String> = Lazy::new(|| {
    env::var("TRUSS_TRANSFER_DOWNLOAD_DIR").unwrap_or_else(|_| "/tmp/truss_transfer".to_string())
});

/// Base path for secrets
pub static SECRETS_BASE_PATH: &str = "/secrets";

pub static RUNTIME_MODEL_CACHE_PATH: &str = "/app/model_cache";

/// Desired download speed for b10fs (MB/s), determined by environment variable or heuristic.
pub static TRUSS_TRANSFER_B10FS_DESIRED_SPEED_MBPS: Lazy<f64> = Lazy::new(|| {
    if let Ok(speed) = env::var("TRUSS_TRANSFER_B10FS_DOWNLOAD_SPEED_MBPS") {
        if let Ok(speed) = speed.parse::<f64>() {
            return speed;
        }
    }

    // if we have 16 or fewer cpu cores, use a lower speed
    let speed_threshold = if num_cpus::get() <= 16 {
        TRUSS_TRANSFER_B10FS_DOWNLOAD_SPEED_MBPS_FEW_CORES
    } else {
        TRUSS_TRANSFER_B10FS_DOWNLOAD_SPEED_MBPS
    };

    // fallback to a random number between 10 MB/s and speed_threshold
    10.0 + rand::random::<f64>() * (speed_threshold - 10.0)
});

/// Default download speed for b10fs (MB/s)
/// Typical disk write: 1.3GB/s, followed by read of 2GB/s
pub static TRUSS_TRANSFER_B10FS_DOWNLOAD_SPEED_MBPS: f64 = 400.0;

/// Download speed for instances with few cores (MB/s)
/// Typical disk write: 250MB/s, followed by read of 350MB/s
pub static TRUSS_TRANSFER_B10FS_DOWNLOAD_SPEED_MBPS_FEW_CORES: f64 = 90.0;

/// Minimum required available space in GB for b10fs
pub static TRUSS_TRANSFER_B10FS_MIN_REQUIRED_AVAILABLE_SPACE_GB: u64 = 100;

// 128MB
// random number, uniform between 25 and 400 MB/s as a threshold
// using random instead of fixed number to e.g. avoid catastrophic
// events e.g. huggingface is down, where b10cache will have more load.
pub static B10FS_BENCHMARK_SIZE: usize = 128 * 1024 * 1024; // 128MB

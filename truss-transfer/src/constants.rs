// Constants used throughout the truss_transfer crate

/// Alternative manifest paths to check
pub static LAZY_DATA_RESOLVER_PATHS: &[&str] = &[
    "/bptr/bptr-manifest",
    "/bptr/bptr-manifest.json",
    "/bptr/static-bptr-manifest.json",
];

/// Cache directory for b10fs
pub static CACHE_DIR: &str = "/cache/org/artifacts/truss_transfer_managed_v1";

/// Timeout for blob downloads in seconds (6 hours)
pub static BLOB_DOWNLOAD_TIMEOUT_SECS: u64 = 21600;

/// Environment variable to enable Baseten FS
pub static BASETEN_FS_ENABLED_ENV_VAR: &str = "BASETEN_FS_ENABLED";

/// Default number of download workers
pub static TRUSS_TRANSFER_NUM_WORKERS_DEFAULT: usize = 64;

/// Environment variable for download directory
pub static TRUSS_TRANSFER_DOWNLOAD_DIR_ENV_VAR: &str = "TRUSS_TRANSFER_DOWNLOAD_DIR";

/// Environment variable for b10fs cleanup hours
pub static TRUSS_TRANSFER_B10FS_CLEANUP_HOURS_ENV_VAR: &str = "TRUSS_TRANSFER_B10FS_CLEANUP_HOURS";

/// Default cleanup hours for b10fs (4 days)
pub static TRUSS_TRANSFER_B10FS_DEFAULT_CLEANUP_HOURS: u64 = 4 * 24;

/// Fallback download directory
pub static TRUSS_TRANSFER_DOWNLOAD_DIR_FALLBACK: &str = "/tmp/bptr-resolved";

/// Base path for secrets
pub static SECRETS_BASE_PATH: &str = "/secrets";

pub static RUNTIME_MODEL_CACHE_PATH: &str = "/app/model_cache";

/// Environment variable for b10fs download speed
pub static TRUSS_TRANSFER_B10FS_DOWNLOAD_SPEED_ENV_VAR: &str =
    "TRUSS_TRANSFER_B10FS_DOWNLOAD_SPEED_MBPS";

/// Default download speed for b10fs (MB/s)
pub static TRUSS_TRANSFER_B10FS_DOWNLOAD_SPEED_MBPS: f64 = 350.0;

/// Download speed for instances with few cores (MB/s)
pub static TRUSS_TRANSFER_B10FS_DOWNLOAD_SPEED_MBPS_FEW_CORES: f64 = 90.0;

/// Minimum required available space in GB for b10fs
pub static TRUSS_TRANSFER_B10FS_MIN_REQUIRED_AVAILABLE_SPACE_GB: u64 = 100;

// 128MB
// random number, uniform between 25 and 400 MB/s as a threshold
// using random instead of fixed number to e.g. avoid catastrophic
// events e.g. huggingface is down, where b10cache will have more load.
pub static B10FS_BENCHMARK_SIZE: usize = 128 * 1024 * 1024; // 128MB

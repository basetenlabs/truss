use std::env;
use std::io::Write;
use std::sync::Once;

use chrono;
use env_logger::Builder;
use log::{error, info, warn, LevelFilter};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::constants::*;
use crate::core::lazy_data_resolve_entrypoint;

#[cfg(feature = "cli")]
use anyhow::Result;

static INIT_LOGGER: Once = Once::new();

/// Initialize the logger with a default level of `info`
pub fn init_logger_once() {
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

/// Resolve the download directory using the provided directory or environment variables
pub fn resolve_truss_transfer_download_dir(optional_download_dir: Option<String>) -> String {
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

/// Python-callable function to read the manifest and download data.
/// By default, it will use the `TRUSS_TRANSFER_DOWNLOAD_DIR` environment variable.
#[pyfunction]
#[pyo3(signature = (download_dir=None))]
pub fn lazy_data_resolve(download_dir: Option<String>) -> PyResult<String> {
    Python::with_gil(|py| py.allow_threads(|| lazy_data_resolve_entrypoint(download_dir)))
        .map_err(|err| PyException::new_err(err.to_string()))
}

/// Running the CLI directly.
#[cfg(feature = "cli")]
pub fn main() -> anyhow::Result<()> {
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
pub fn truss_transfer(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lazy_data_resolve, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

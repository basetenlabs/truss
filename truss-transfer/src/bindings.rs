use std::env;
use std::io::Write;
use std::sync::Once;

use chrono;
use env_logger::Builder;
use log::{warn, LevelFilter};

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, wrap_pyfunction};

use crate::constants::RUNTIME_MODEL_CACHE_PATH;
use crate::constants::*;
use crate::core::lazy_data_resolve_entrypoint;
use crate::create::create_basetenpointer;
use crate::types::{ModelRepo, ResolutionType};
static INIT_LOGGER: Once = Once::new();
use once_cell::sync::Lazy;
use std::sync::Arc;
use tokio::runtime::Runtime;

// --- Global Tokio Runtime ---
static GLOBAL_RUNTIME: Lazy<Arc<Runtime>> = Lazy::new(|| {
    Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create global multi-threaded Tokio runtime"),
    )
});

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

// create PyModelRepo
#[pyclass]
#[derive(Clone)]
pub struct PyModelRepo {
    #[pyo3(get, set)]
    pub repo_id: String,
    #[pyo3(get, set)]
    pub revision: String,
    #[pyo3(get, set)]
    pub allow_patterns: Option<Vec<String>>,
    #[pyo3(get, set)]
    pub ignore_patterns: Option<Vec<String>>,
    #[pyo3(get, set)]
    pub volume_folder: String,
    #[pyo3(get, set)]
    pub runtime_secret_name: String,
    #[pyo3(get, set)]
    pub kind: String,
}

#[pymethods]
impl PyModelRepo {
    #[new]
    #[pyo3(signature = (
        repo_id,
        revision,
        volume_folder,
        kind = "hf".to_string(),
        allow_patterns = None,
        ignore_patterns = None,
        runtime_secret_name = "hf_access_token".to_string(),
    ))]
    pub fn new(
        repo_id: String,
        revision: String,
        volume_folder: String,
        kind: String,
        allow_patterns: Option<Vec<String>>,
        ignore_patterns: Option<Vec<String>>,
        runtime_secret_name: String,
    ) -> Self {
        PyModelRepo {
            repo_id,
            revision,
            allow_patterns,
            ignore_patterns,
            volume_folder,
            runtime_secret_name,
            kind,
        }
    }
}

// impl into ModelRepo for Bound<'_, PyModelRepo>
impl TryFrom<Bound<'_, PyModelRepo>> for ModelRepo {
    type Error = PyErr;

    fn try_from(py_model_repo: Bound<'_, PyModelRepo>) -> Result<Self, Self::Error> {
        let py_model_repo = py_model_repo.borrow();
        Ok(ModelRepo {
            repo_id: py_model_repo.repo_id.clone(),
            revision: py_model_repo.revision.clone(),
            allow_patterns: py_model_repo.allow_patterns.clone(),
            ignore_patterns: py_model_repo.ignore_patterns.clone(),
            volume_folder: py_model_repo.volume_folder.clone(),
            runtime_secret_name: py_model_repo.runtime_secret_name.clone(),
            kind: match py_model_repo.kind.as_str() {
                "http" | "hf" => ResolutionType::Http,
                "gcs" => ResolutionType::Gcs,
                "s3" => ResolutionType::S3,
                "azure" => ResolutionType::Azure,
                _ => {
                    return Err(PyException::new_err(format!(
                        "Unknown kind: {}",
                        py_model_repo.kind
                    )));
                }
            },
        })
    }
}

/// Python function for creating a BasetenPointer JSON from a list of ModelRepo
/// This creates BasetenPointer objects from HuggingFace model repositories
/// signature is create_basetenpointer_from_models(models: Vec<ModelRepo>) -> PyResult<String> {
#[pyfunction]
#[pyo3(signature = (models, model_path=RUNTIME_MODEL_CACHE_PATH.to_string()))]
pub fn create_basetenpointer_from_models(
    models: Vec<Bound<'_, PyModelRepo>>,
    model_path: String,
) -> PyResult<String> {
    // convert PyModelRepo to ModelRepo
    let models: PyResult<Vec<ModelRepo>> = models.into_iter().map(TryInto::try_into).collect();
    GLOBAL_RUNTIME
        .block_on(async move { create_basetenpointer(models?, model_path).await })
        .map_err(|e| PyException::new_err(e.to_string()))
}

/// Python module definition
#[pymodule]
pub fn truss_transfer(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lazy_data_resolve, m)?)?;
    m.add_function(wrap_pyfunction!(create_basetenpointer_from_models, m)?)?;
    m.add_class::<PyModelRepo>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

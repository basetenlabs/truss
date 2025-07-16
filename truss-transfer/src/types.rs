use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum ResolutionType {
    #[serde(rename = "http", alias = "hf")]
    Http,
    #[serde(rename = "gcs")]
    Gcs,
}

impl Default for ResolutionType {
    fn default() -> Self {
        ResolutionType::Http
    }
}

/// Corresponds to `Resolution` in the Python code
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Resolution {
    pub url: String,
    #[serde(default)]
    pub resolution_type: ResolutionType,
    pub expiration_timestamp: i64,
}

fn default_runtime_secret_name() -> String {
    // TODO: remove this default once its adopted.
    "hf_access_token".to_string()
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct BasetenPointer {
    pub resolution: Option<Resolution>,
    pub uid: String,
    pub file_name: String,
    pub hashtype: String,
    pub hash: String,
    pub size: u64,
    // defaults to `hf_access_token` if not provided
    #[serde(default = "default_runtime_secret_name")]
    pub runtime_secret_name: String,
}

#[pyclass]
#[derive(Debug, Deserialize, Serialize)]
pub struct BasetenPointerManifest {
    pub pointers: Vec<BasetenPointer>,
}

/// Model cache entry configuration
#[pyclass]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelRepo {
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
}

#[pymethods]
impl ModelRepo {
    #[new]
    #[pyo3(signature = (
        repo_id,
        revision,
        volume_folder,
        allow_patterns = None,
        ignore_patterns = None,
        runtime_secret_name = "hf_access_token".to_string(),
    ))]
    pub fn new(
        repo_id: String,
        revision: String,
        volume_folder: String,
        allow_patterns: Option<Vec<String>>,
        ignore_patterns: Option<Vec<String>>,
        runtime_secret_name: String,
    ) -> Self {
        ModelRepo {
            repo_id,
            revision,
            allow_patterns,
            ignore_patterns,
            volume_folder,
            runtime_secret_name: runtime_secret_name
        }
    }
}

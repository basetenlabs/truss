use object_store::Error;
use pyo3::pyclass;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
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
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelRepo {
    pub repo_id: String,
    pub revision: String,
    pub allow_patterns: Option<Vec<String>>,
    pub ignore_patterns: Option<Vec<String>>,
    pub volume_folder: String,
    pub runtime_secret_name: String,
    pub kind: ResolutionType,
}

/// Error types for GCS operations
#[derive(Debug, thiserror::Error)]
pub enum GcsError {
    #[error("Invalid metadata")]
    InvalidMetadata,
    #[error("Object store error: {0}")]
    ObjectStore(#[from] object_store::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid GCS URI: {0}")]
    InvalidUri(String),
}

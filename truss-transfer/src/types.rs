use pyo3::pyclass;
use serde::{Deserialize, Serialize};
use serde_json;

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub enum ResolutionType {
    #[serde(rename = "http", alias = "hf")]
    Http,
    #[serde(rename = "gcs")]
    Gcs,
}

impl ToString for ResolutionType {
    fn to_string(&self) -> String {
        match self {
            ResolutionType::Http => "http".to_string(),
            ResolutionType::Gcs => "gcs".to_string(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct HttpResolution {
    pub url: String,
    pub expiration_timestamp: i64,
    resolution_type: ResolutionType,
}

impl HttpResolution {
    pub fn new(url: String, expiration_timestamp: i64) -> Self {
        Self {
            url,
            expiration_timestamp,
            resolution_type: ResolutionType::Http,
        }
    }
}

/// GCS resolution with bucket name
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct GcsResolution {
    pub url: String,
    pub bucket_name: String,
    resolution_type: ResolutionType,
}

impl GcsResolution {
    pub fn new(url: String, bucket_name: String) -> Self {
        Self {
            url,
            bucket_name,
            resolution_type: ResolutionType::Gcs,
        }
    }
}

/// Union type representing different resolution types
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "resolution_type")]
pub enum Resolution {
    #[serde(rename = "http")]
    Http(HttpResolution),
    #[serde(rename = "gcs")]
    Gcs(GcsResolution),
}
fn default_runtime_secret_name() -> String {
    // TODO: remove this default once its adopted.
    "hf_access_token".to_string()
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct BasetenPointer {
    pub resolution: Resolution,
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
    #[error("Object store error: {0}")]
    ObjectStore(#[from] object_store::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid GCS URI: {0}")]
    InvalidUri(String),
}

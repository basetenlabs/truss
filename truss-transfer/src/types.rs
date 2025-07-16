use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
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
#[derive(Debug, Deserialize, Clone)]
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

/// Corresponds to `BasetenPointer` in the Python code
#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
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

/// Corresponds to `BasetenPointerManifest` in the Python code
#[derive(Debug, Deserialize)]
pub struct BasetenPointerManifest {
    pub pointers: Vec<BasetenPointer>,
}

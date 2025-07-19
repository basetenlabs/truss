use pyo3::pyclass;
use serde::{Deserialize, Serialize};
use serde_json;

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub enum ResolutionType {
    #[serde(rename = "http", alias = "hf")]
    Http,
    #[serde(rename = "gcs")]
    Gcs,
    #[serde(rename = "s3")]
    S3,
    #[serde(rename = "azure")]
    Azure,
}

impl ToString for ResolutionType {
    fn to_string(&self) -> String {
        match self {
            ResolutionType::Http => "http".to_string(),
            ResolutionType::Gcs => "gcs".to_string(),
            ResolutionType::S3 => "s3".to_string(),
            ResolutionType::Azure => "azure".to_string(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct HttpResolution {
    pub url: String,
    pub expiration_timestamp: i64,
}

impl HttpResolution {
    pub fn new(url: String, expiration_timestamp: i64) -> Self {
        Self {
            url,
            expiration_timestamp,
        }
    }
}

/// GCS resolution with bucket name
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct GcsResolution {
    pub path: String,
    pub bucket_name: String,
}

impl GcsResolution {
    pub fn new(path: String, bucket_name: String) -> Self {
        Self { path, bucket_name }
    }
}

/// S3 resolution with bucket name and region
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct S3Resolution {
    pub bucket_name: String,
    pub key: String,
    pub region: Option<String>,
}

impl S3Resolution {
    pub fn new(bucket_name: String, key: String, region: Option<String>) -> Self {
        Self {
            bucket_name,
            key,
            region,
        }
    }
}

/// Azure Blob Storage resolution with container and blob path
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AzureResolution {
    pub account_name: String,
    pub container_name: String,
    pub blob_name: String,
}

impl AzureResolution {
    pub fn new(account_name: String, container_name: String, blob_name: String) -> Self {
        Self {
            account_name,
            container_name,
            blob_name,
        }
    }
}

/// Union type representing different resolution types
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(from = "MaybeTaggedResolution", into = "TaggedResolution")]
pub enum Resolution {
    Http(HttpResolution),
    Gcs(GcsResolution),
    S3(S3Resolution),
    Azure(AzureResolution),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum MaybeTaggedResolution {
    Tagged(TaggedResolution),
    UntaggedHttp {
        url: String,
        expiration_timestamp: i64,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "resolution_type")]
enum TaggedResolution {
    #[serde(rename = "http")]
    Http(HttpResolution),
    #[serde(rename = "gcs")]
    Gcs(GcsResolution),
    #[serde(rename = "s3")]
    S3(S3Resolution),
    #[serde(rename = "azure")]
    Azure(AzureResolution),
}

impl From<MaybeTaggedResolution> for Resolution {
    fn from(resolution: MaybeTaggedResolution) -> Resolution {
        match resolution {
            MaybeTaggedResolution::Tagged(TaggedResolution::Http(http)) => Resolution::Http(http),
            MaybeTaggedResolution::Tagged(TaggedResolution::Gcs(gcs)) => Resolution::Gcs(gcs),
            MaybeTaggedResolution::Tagged(TaggedResolution::S3(s3)) => Resolution::S3(s3),
            MaybeTaggedResolution::Tagged(TaggedResolution::Azure(azure)) => {
                Resolution::Azure(azure)
            }
            MaybeTaggedResolution::UntaggedHttp {
                url,
                expiration_timestamp,
            } => Resolution::Http(HttpResolution {
                url,
                expiration_timestamp,
            }),
        }
    }
}

impl Into<TaggedResolution> for Resolution {
    fn into(self) -> TaggedResolution {
        match self {
            Resolution::Http(http) => TaggedResolution::Http(http),
            Resolution::Gcs(gcs) => TaggedResolution::Gcs(gcs),
            Resolution::S3(s3) => TaggedResolution::S3(s3),
            Resolution::Azure(azure) => TaggedResolution::Azure(azure),
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_http_resolution_from_json() {
        let json = r#"{
            "resolution_type": "http",
            "url": "https://cdn.baseten.co/docs/production/Gettysburg.mp3",
            "expiration_timestamp": 2683764059
        }"#;

        let resolution: HttpResolution = serde_json::from_str(json).unwrap();
        assert_eq!(
            resolution.url,
            "https://cdn.baseten.co/docs/production/Gettysburg.mp3"
        );
        assert_eq!(resolution.expiration_timestamp, 2683764059);
    }

    #[test]
    fn test_gcs_resolution_from_json() {
        let json = r#"{
            "resolution_type": "gcs",
            "path": "models/llama-3-2-1b-instruct/model.bin",
            "bucket_name": "llama-3-2-1b-instruct"
        }"#;

        let resolution: GcsResolution = serde_json::from_str(json).unwrap();
        assert_eq!(resolution.path, "models/llama-3-2-1b-instruct/model.bin");
        assert_eq!(resolution.bucket_name, "llama-3-2-1b-instruct");
    }

    #[test]
    fn test_resolution_enum_http_from_json() {
        let json = r#"{
            "resolution_type": "http",
            "url": "https://cdn.baseten.co/docs/production/Gettysburg.mp3",
            "expiration_timestamp": 2683764059
        }"#;

        let resolution: Resolution = serde_json::from_str(json).unwrap();
        match resolution {
            Resolution::Http(http_res) => {
                assert_eq!(
                    http_res.url,
                    "https://cdn.baseten.co/docs/production/Gettysburg.mp3"
                );
            }
            _ => panic!("Expected HTTP resolution"),
        }
    }

    #[test]
    fn test_resolution_enum_gcs_from_json() {
        let json = r#"{
            "resolution_type": "gcs",
            "path": "models/llama-3-2-1b-instruct/model.bin",
            "bucket_name": "llama-3-2-1b-instruct"
        }"#;

        let resolution: Resolution = serde_json::from_str(json).unwrap();
        match resolution {
            Resolution::Gcs(gcs_res) => {
                assert_eq!(gcs_res.path, "models/llama-3-2-1b-instruct/model.bin");
                assert_eq!(gcs_res.bucket_name, "llama-3-2-1b-instruct");
            }
            _ => panic!("Expected GCS resolution"),
        }
    }

    #[test]
    fn test_baseten_pointer_manifest_current_format() {
        let json = r#"{
            "pointers": [
                {
                    "resolution": {
                        "resolution_type": "http",
                        "url": "https://cdn.baseten.co/docs/production/Gettysburg.mp3",
                        "expiration_timestamp": 2683764059
                    },
                    "uid": "8c6b2f215f0333437cdc3fe7c79be0c802847d2f2a0ccdc0bb251814e63cf375",
                    "file_name": "random_github_file.yml",
                    "hashtype": "blake3",
                    "hash": "8c6b2f215f0333437cdc3fe7c79be0c802847d2f2a0ccdc0bb251814e63cf375",
                    "size": 75649
                },
                {
                    "resolution": {
                        "resolution_type": "gcs",
                        "path": "models/llama-3-2-1b-instruct/model.bin",
                        "bucket_name": "llama-3-2-1b-instruct"
                    },
                    "uid": "8c6b2f215f0333437cdc3fe7c79be0c802847d2f2a0ccdc0bb251814e63cf375",
                    "file_name": "model.bin",
                    "hashtype": "blake3",
                    "hash": "8c6b2f215f0333437cdc3fe7c79be0c802847d2f2a0ccdc0bb251814e63c2f375",
                    "size": 75649
                }
            ]
        }"#;

        let manifest: BasetenPointerManifest = serde_json::from_str(json).unwrap();
        assert_eq!(manifest.pointers.len(), 2);

        match &manifest.pointers[0].resolution {
            Resolution::Http(_) => {}
            _ => panic!("Expected HTTP resolution for first pointer"),
        }

        match &manifest.pointers[1].resolution {
            Resolution::Gcs(_) => {}
            _ => panic!("Expected GCS resolution for second pointer"),
        }
    }

    // Historic format test - old manifests without resolution_type should default to HTTP
    #[test]
    fn test_baseten_pointer_manifest_historic_format() {
        let json = r#"{
            "pointers": [
                {
                    "resolution": {
                        "url": "https://cdn.baseten.co/docs/production/Gettysburg.mp3",
                        "expiration_timestamp": 2683764059
                    },
                    "uid": "8c6b2f215f0333437cdc3fe7c79be0c802847d2f2a0ccdc0bb251814e63cf375",
                    "file_name": "random_github_file.yml",
                    "hashtype": "blake3",
                    "hash": "8c6b2f215f0333437cdc3fe7c79be0c802847d2f2a0ccdc0bb251814e63cf375",
                    "size": 75649
                },
                {
                    "resolution": {
                        "url": "https://cdn.baseten.co/docs/production/Gettysburg.mp3",
                        "expiration_timestamp": 2683764059
                    },
                    "uid": "8c6b2f215f0333437cdc3fe7c79be0c802847d2f2a0ccdc0bb251814e63cf375",
                    "file_name": "random_github_file2.yml",
                    "hashtype": "blake3",
                    "hash": "8c6b2f215f0333437cdc3fe7c79be0c802847d2f2a0ccdc0bb251814e63c2f375",
                    "size": 75649
                }
            ]
        }"#;

        let manifest: BasetenPointerManifest = serde_json::from_str(json).unwrap();
        assert_eq!(manifest.pointers.len(), 2);

        for pointer in &manifest.pointers {
            match &pointer.resolution {
                Resolution::Http(_) => {}
                _ => panic!("Expected HTTP resolution for historic format"),
            }
        }
    }

    // Real-world GCS manifest test with duplicate resolution_type fields
    #[test]
    fn test_baseten_pointer_manifest_gcs_real_world() {
        let json = r#"{
            "pointers": [
                {
                    "resolution": {
                        "resolution_type": "gcs",
                        "path": "tokenizer.json",
                        "bucket_name": "llama-3-2-1b-instruct"
                    },
                    "uid": "gcs-3ff6b653d22a2676f3a03bd7b5d7ff88",
                    "file_name": "/app/model_cache/tokenizer.json",
                    "hashtype": "md5",
                    "hash": "3ff6b653d22a2676f3a03bd7b5d7ff88",
                    "size": 9085657,
                    "runtime_secret_name": "gcs-account"
                },
                {
                    "resolution": {
                        "path": "USE_POLICY.md",
                        "bucket_name": "llama-3-2-1b-instruct",
                        "resolution_type": "gcs"
                    },
                    "uid": "gcs-1729addd533ca7a6e956e3f077f4a4e9",
                    "file_name": "/app/model_cache/USE_POLICY.md",
                    "hashtype": "md5",
                    "hash": "1729addd533ca7a6e956e3f077f4a4e9",
                    "size": 6021,
                    "runtime_secret_name": "gcs-account"
                }
            ]
        }"#;

        let manifest: BasetenPointerManifest = serde_json::from_str(json).unwrap();
        assert_eq!(manifest.pointers.len(), 2);

        // Check first pointer
        match &manifest.pointers[0].resolution {
            Resolution::Gcs(gcs_res) => {
                assert_eq!(gcs_res.path, "tokenizer.json");
                assert_eq!(gcs_res.bucket_name, "llama-3-2-1b-instruct");
            }
            _ => panic!("Expected GCS resolution for first pointer"),
        }
        assert_eq!(
            manifest.pointers[0].uid,
            "gcs-3ff6b653d22a2676f3a03bd7b5d7ff88"
        );
        assert_eq!(
            manifest.pointers[0].file_name,
            "/app/model_cache/tokenizer.json"
        );
        assert_eq!(manifest.pointers[0].hashtype, "md5");
        assert_eq!(
            manifest.pointers[0].hash,
            "3ff6b653d22a2676f3a03bd7b5d7ff88"
        );
        assert_eq!(manifest.pointers[0].size, 9085657);
        assert_eq!(manifest.pointers[0].runtime_secret_name, "gcs-account");

        // Check second pointer
        match &manifest.pointers[1].resolution {
            Resolution::Gcs(gcs_res) => {
                assert_eq!(gcs_res.path, "USE_POLICY.md");
                assert_eq!(gcs_res.bucket_name, "llama-3-2-1b-instruct");
            }
            _ => panic!("Expected GCS resolution for second pointer"),
        }
        assert_eq!(
            manifest.pointers[1].uid,
            "gcs-1729addd533ca7a6e956e3f077f4a4e9"
        );
        assert_eq!(
            manifest.pointers[1].file_name,
            "/app/model_cache/USE_POLICY.md"
        );
        assert_eq!(manifest.pointers[1].hashtype, "md5");
        assert_eq!(
            manifest.pointers[1].hash,
            "1729addd533ca7a6e956e3f077f4a4e9"
        );
        assert_eq!(manifest.pointers[1].size, 6021);
        assert_eq!(manifest.pointers[1].runtime_secret_name, "gcs-account");
    }
}

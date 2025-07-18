use crate::types::{BasetenPointer, ModelRepo, Resolution, ResolutionType};
use chrono;
use log::{debug, info};
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::ObjectStore;
use serde_json;
use std::collections::HashMap;
use std::fs;
use rand;
use futures_util::stream::StreamExt;
// super::filter::filter_repo_files;

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

/// GCS file metadata
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GcsFileMetadata {
    /// ETag or equivalent hash (deprecate, using md5)
    pub etag: String,
    /// File size in bytes
    pub size: u64,
    /// GCS object path
    pub path: String,
    /// Last modified timestamp
    pub last_modified: chrono::DateTime<chrono::Utc>,
    // TODO: get md5 hash instead of etag, to de-duplicate across GCS and multiple repos
}

/// Parse GCS URI (gs://bucket/path) into bucket and prefix
fn parse_gcs_uri(uri: &str) -> Result<(String, String), GcsError> {
    if !uri.starts_with("gs://") {
        return Err(GcsError::InvalidUri(format!("URI must start with gs://: {}", uri)));
    }

    let path = &uri[5..]; // Remove "gs://"
    let parts: Vec<&str> = path.splitn(2, '/').collect();

    let bucket = parts[0].to_string();
    let prefix = if parts.len() > 1 { parts[1].to_string() } else { String::new() };

    Ok((bucket, prefix))
}

/// Check if file should be ignored based on patterns
/// REPLACE WITH super::filter_repo_files!
fn should_ignore_file(
    file_path: &str,
    allow_patterns: Option<&[String]>,
    ignore_patterns: Option<&[String]>,
) -> bool {
    // If there are ignore patterns and this file matches any, ignore it
    if let Some(ignore) = ignore_patterns {
        for pattern in ignore {
            if glob_match(pattern, file_path) {
                return true;
            }
        }
    }

    // If there are allow patterns, file must match at least one
    if let Some(allow) = allow_patterns {
        for pattern in allow {
            if glob_match(pattern, file_path) {
                return false; // Found a match, don't ignore
            }
        }
        return true; // No match found, ignore
    }

    false // No patterns or no ignore match
}

/// Simple glob pattern matching
/// // ot needed as should_ignore_file is replaced with super::filter_repo_files
fn glob_match(pattern: &str, text: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    if pattern.contains('*') {
        let parts: Vec<&str> = pattern.split('*').collect();
        let mut text_pos = 0;

        for (i, part) in parts.iter().enumerate() {
            if part.is_empty() {
                continue;
            }

            if i == 0 {
                // First part must match from beginning
                if !text[text_pos..].starts_with(part) {
                    return false;
                }
                text_pos += part.len();
            } else if i == parts.len() - 1 {
                // Last part must match at end
                return text[text_pos..].ends_with(part);
            } else {
                // Middle parts must exist somewhere
                if let Some(pos) = text[text_pos..].find(part) {
                    text_pos += pos + part.len();
                } else {
                    return false;
                }
            }
        }
        true
    } else {
        text == pattern
    }
}

/// Get metadata for all files in GCS bucket
async fn metadata_gcs_bucket(
    repo_id: &str,
    runtime_secret_name: &str,
    allow_patterns: Option<&[String]>,
    ignore_patterns: Option<&[String]>,
) -> Result<HashMap<String, GcsFileMetadata>, GcsError> {
    let (bucket, prefix) = parse_gcs_uri(repo_id)?;

    // Read GCS service account credentials
    let secret_path = format!("/secrets/{}", runtime_secret_name);
    let credentials_json = fs::read_to_string(&secret_path)?;

    // Build GCS client
    let gcs = GoogleCloudStorageBuilder::new()
        .with_service_account_key(&credentials_json)
        .with_bucket_name(&bucket)
        .build()?;

    let mut file_metadata = HashMap::new();

    // List objects with prefix
    let prefix_path = if prefix.is_empty() {
        object_store::path::Path::from("")
    } else {
        object_store::path::Path::from(prefix.clone())
    };

    debug!("Listing GCS objects in bucket: {}, prefix: {}", bucket, prefix);

    let mut object_stream = gcs.list(Some(&prefix_path));

    while let Some(object_result) = object_stream.next().await {
        let object = object_result?;
        let file_path = object.location.to_string();
        let file_name = if prefix.is_empty() {
            file_path.clone()
        } else {
            file_path.strip_prefix(&format!("{}/", prefix))
                .unwrap_or(&file_path)
                .to_string()
        };

        // Skip if this file should be ignored
        if should_ignore_file(&file_name, allow_patterns, ignore_patterns) {
            debug!("Ignoring file: {}", file_name);
            continue;
        }
        // use filter_repo_files instead
        // filter_repo_files(files, allow_patterns, ignore_patterns)?;

        let metadata = GcsFileMetadata {
            etag: object.e_tag.unwrap_or_else(|| format!("gcs-{}", rand::random::<u64>())),
            size: object.size,
            path: file_path.clone(),
            last_modified: object.last_modified,
        };

        file_metadata.insert(file_name, metadata);
    }

    info!("Found {} files in GCS bucket {}", file_metadata.len(), bucket);
    Ok(file_metadata)
}

/// Convert GCS ModelRepo to BasetenPointer format
pub async fn model_cache_gcs_to_b10ptr(
    models: Vec<&ModelRepo>,
) -> Result<Vec<BasetenPointer>, GcsError> {
    let mut basetenpointers = Vec::new();

    for model in models {
        if model.kind != ResolutionType::Gcs {
            continue;
        }

        info!("Processing GCS model: {}", model.repo_id);

        let metadata = metadata_gcs_bucket(
            &model.repo_id,
            &model.runtime_secret_name,
            model.allow_patterns.as_deref(),
            model.ignore_patterns.as_deref(),
        ).await?;

        for (file_name, file_metadata) in metadata {
            let full_file_path = if model.volume_folder.is_empty() {
                file_name.clone()
            } else {
                format!("{}/{}", model.volume_folder, file_name)
            };

            // Create a temporary HTTP URL for the GCS object
            // This will be replaced with pre-signed URLs in resolution phase
            let (bucket, _) = parse_gcs_uri(&model.repo_id)?;
            let temp_url = format!("https://storage.googleapis.com/{}/{}", bucket, file_metadata.path);

            let pointer = BasetenPointer {
                resolution: Some(Resolution {
                    url: temp_url,
                    resolution_type: ResolutionType::Gcs,
                    expiration_timestamp: (chrono::Utc::now() + chrono::Duration::hours(24)).timestamp(),
                }),
                uid: format!("gcs-{}", file_metadata.etag),
                file_name: full_file_path,
                hashtype: "etag".to_string(),
                hash: file_metadata.etag,
                size: file_metadata.size,
                runtime_secret_name: model.runtime_secret_name.clone(),
            };

            basetenpointers.push(pointer);
        }
    }

    info!("Created {} GCS basetenpointers", basetenpointers.len());
    Ok(basetenpointers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gcs_uri() {
        // Test valid URIs
        let (bucket, prefix) = parse_gcs_uri("gs://my-bucket/path/to/file").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(prefix, "path/to/file");

        let (bucket, prefix) = parse_gcs_uri("gs://my-bucket").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(prefix, "");

        // Test invalid URIs
        assert!(parse_gcs_uri("s3://my-bucket").is_err());
        assert!(parse_gcs_uri("invalid").is_err());
    }

    #[test]
    fn test_glob_match() {
        assert!(glob_match("*.txt", "file.txt"));
        assert!(glob_match("*.json", "config.json"));
        assert!(!glob_match("*.txt", "file.json"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("prefix*", "prefix_file.txt"));
        assert!(!glob_match("prefix*", "other_file.txt"));
    }

    #[test]
    fn test_should_ignore_file() {
        let allow_patterns = vec!["*.safetensors".to_string(), "*.json".to_string()];
        let ignore_patterns = vec!["*.md".to_string()];

        // Should allow safetensors files
        assert!(!should_ignore_file("model.safetensors", Some(&allow_patterns), Some(&ignore_patterns)));

        // Should ignore md files
        assert!(should_ignore_file("README.md", Some(&allow_patterns), Some(&ignore_patterns)));

        // Should ignore files not in allow patterns
        assert!(should_ignore_file("model.txt", Some(&allow_patterns), Some(&ignore_patterns)));

        // Should allow when no patterns specified
        assert!(!should_ignore_file("any_file.txt", None, None));
    }
}

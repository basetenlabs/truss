use super::filter::should_ignore_file;
use crate::constants::RUNTIME_MODEL_CACHE_PATH;
use crate::types::{
    BasetenPointer, GcsError, GcsResolution, ModelRepo, Resolution, ResolutionType,
};
use chrono;
use futures_util::stream::StreamExt;
use log::{debug, info};
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::ObjectStore;
use rand;
use std::collections::HashMap;
use std::fs;

/// GCS file metadata
#[derive(Debug, Clone)]
pub struct GcsFileMetadata {
    /// MD5 hash of the file content
    pub md5_hash: String,
    /// File size in bytes
    pub size: u64,
    /// GCS object path
    pub path: String,
    /// Last modified timestamp
    pub last_modified: chrono::DateTime<chrono::Utc>,
}

/// Parse GCS URI (gs://bucket/path) into bucket and prefix
fn parse_gcs_uri(uri: &str) -> Result<(String, String), GcsError> {
    if !uri.starts_with("gs://") {
        return Err(GcsError::InvalidUri(format!(
            "URI must start with gs://: {}",
            uri
        )));
    }

    let path = &uri[5..]; // Remove "gs://"
    let parts: Vec<&str> = path.splitn(2, '/').collect();

    let bucket = parts[0].to_string();
    let prefix = if parts.len() > 1 {
        parts[1].to_string()
    } else {
        String::new()
    };

    Ok((bucket, prefix))
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

    debug!(
        "Listing GCS objects in bucket: {}, prefix: {}",
        bucket, prefix
    );

    let mut object_stream = gcs.list(Some(&prefix_path));

    while let Some(object_result) = object_stream.next().await {
        let object = object_result?;
        let file_path = object.location.to_string();
        let file_name = if prefix.is_empty() {
            file_path.clone()
        } else {
            file_path
                .strip_prefix(&format!("{}/", prefix))
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

        // Extract MD5 hash from GCS metadata or use ETag as fallback
        let md5_hash = object
            .e_tag
            .as_ref()
            .and_then(|etag| {
                // GCS ETags often contain MD5 hash in quotes
                if etag.starts_with('"') && etag.ends_with('"') && etag.len() == 34 {
                    Some(etag[1..33].to_string()) // Extract MD5 from quoted ETag
                } else {
                    None
                }
            })
            .unwrap_or_else(|| format!("gcs-{}", rand::random::<u64>()));

        if object.size == 0 {
            debug!("Skipping empty lock file: {}", file_name);
            continue; // Skip empty files
        }

        let metadata = GcsFileMetadata {
            md5_hash,
            size: object.size,
            path: file_name.clone(),
            last_modified: object.last_modified,
        };

        file_metadata.insert(file_name, metadata);
    }

    info!(
        "Found {} files in GCS bucket {}",
        file_metadata.len(),
        bucket
    );
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
        )
        .await?;

        for (file_name, file_metadata) in metadata {
            let full_file_path = format!(
                "{}/{}",
                RUNTIME_MODEL_CACHE_PATH, file_name
            );
            // Create a temporary HTTP URL for the GCS object
            // This will be replaced with pre-signed URLs in resolution phase
            let (bucket, _) = parse_gcs_uri(&model.repo_id)?;

            let pointer = BasetenPointer {
                resolution: Resolution::Gcs(GcsResolution::new(file_metadata.path, bucket)),
                uid: format!("gcs-{}", file_metadata.md5_hash),
                file_name: full_file_path,
                hashtype: "md5".to_string(),
                hash: file_metadata.md5_hash,
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
}

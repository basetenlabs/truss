use crate::types::{BasetenPointer, ModelRepo};
use chrono;
// use log::{debug, info};
// NOTE: object_store dependency temporarily removed due to API compatibility issues
// use object_store::gcp::GoogleCloudStorageBuilder;
// use object_store::ObjectStore;
// use serde_json;
// use std::collections::HashMap;
// use std::fs;
// use std::path::Path;
// use std::sync::Arc;

/// Error types for GCS operations
#[derive(Debug, thiserror::Error)]
pub enum GcsError {
    #[error("Invalid metadata")]
    #[allow(dead_code)]
    InvalidMetadata,
}

/// GCS file metadata
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GcsFileMetadata {
    /// ETag or equivalent hash
    pub etag: String,
    /// File size in bytes
    pub size: u64,
    /// GCS object path
    pub path: String,
    /// Last modified timestamp
    pub last_modified: chrono::DateTime<chrono::Utc>,
}

/// Convert GCS ModelRepo to BasetenPointer format
pub async fn model_cache_gcs_to_b10ptr(
    models: Vec<&ModelRepo>,
) -> Result<Vec<BasetenPointer>, GcsError> {
    let basetenpointers = Vec::new();
    // return dummy data for now
    Ok(basetenpointers)

    // for model in models {
    //     if model.kind != ResolutionType::Gcs {
    //         continue;
    //     }

    //     info!("Processing GCS model: {}", model.repo_id);

    //     let metadata = metadata_gcs_bucket(
    //         &model.repo_id,
    //         &model.runtime_secret_name,
    //         model.allow_patterns.as_deref(),
    //         model.ignore_patterns.as_deref(),
    //     ).await?;
    //
    //     for (file_name, file_metadata) in metadata {
    //         let full_file_path = if model.volume_folder.is_empty() {
    //             file_name.clone()
    //         } else {
    //             format!("{}/{}", model.volume_folder, file_name)
    //         };

    //         // Create a temporary HTTP URL for the GCS object
    //         // This will be replaced with pre-signed URLs in resolution phase
    //         let temp_url = format!("https://storage.googleapis.com/{}", file_metadata.path);

    //         let pointer = BasetenPointer {
    //             resolution: Some(Resolution {
    //                 url: temp_url,
    //                 resolution_type: ResolutionType::Gcs,
    //                 expiration_timestamp: (chrono::Utc::now() + chrono::Duration::hours(24)).timestamp(),
    //             }),
    //             uid: format!("gcs-{}", file_metadata.etag),
    //             file_name: full_file_path,
    //             hashtype: "etag".to_string(),
    //             hash: file_metadata.etag,
    //             size: file_metadata.size,
    //             runtime_secret_name: model.runtime_secret_name.clone(),
    //         };

    //         basetenpointers.push(pointer);
    //     }
    // }

    // info!("Created {} GCS basetenpointers", basetenpointers.len());
    // Ok(basetenpointers)
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     // use crate::types::ResolutionType;

//     #[test]
//     fn test_parse_gcs_uri() {
//         // Test valid URIs
//         let (bucket, prefix) = parse_gcs_uri("gs://my-bucket/path/to/file").unwrap();
//         assert_eq!(bucket, "my-bucket");
//         assert_eq!(prefix, "path/to/file");

//         let (bucket, prefix) = parse_gcs_uri("gs://my-bucket").unwrap();
//         assert_eq!(bucket, "my-bucket");
//         assert_eq!(prefix, "");

//         // Test invalid URIs
//         assert!(parse_gcs_uri("s3://my-bucket").is_err());
//         assert!(parse_gcs_uri("invalid").is_err());
//     }

//     #[test]
//     fn test_glob_match() {
//         assert!(glob_match("*.txt", "file.txt"));
//         assert!(glob_match("*.json", "config.json"));
//         assert!(!glob_match("*.txt", "file.json"));
//         assert!(glob_match("*", "anything"));
//         assert!(glob_match("prefix*", "prefix_file.txt"));
//         assert!(!glob_match("prefix*", "other_file.txt"));
//     }

//     #[test]
//     fn test_should_ignore_file() {
//         let allow_patterns = vec!["*.safetensors".to_string(), "*.json".to_string()];
//         let ignore_patterns = vec!["*.md".to_string()];

//         // Should allow safetensors files
//         assert!(!should_ignore_file("model.safetensors", Some(&allow_patterns), Some(&ignore_patterns)));

//         // Should ignore md files
//         assert!(should_ignore_file("README.md", Some(&allow_patterns), Some(&ignore_patterns)));

//         // Should ignore files not in allow patterns
//         assert!(should_ignore_file("model.txt", Some(&allow_patterns), Some(&ignore_patterns)));

//         // Should allow when no patterns specified
//         assert!(!should_ignore_file("any_file.txt", None, None));
//     }
// }

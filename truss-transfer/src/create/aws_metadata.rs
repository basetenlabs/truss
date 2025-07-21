use anyhow::{anyhow, Result};
use log::info;

use crate::constants::RUNTIME_MODEL_CACHE_PATH;
use crate::secrets::get_secret_from_file;
use crate::types::{BasetenPointer, ModelRepo, Resolution, S3Resolution};
/// Parse S3 URI into bucket and key components
/// Expected format: s3://bucket-name/path/to/object
pub fn parse_s3_uri(uri: &str) -> Result<(String, String)> {
    if !uri.starts_with("s3://") {
        return Err(anyhow!("Invalid S3 URI format: {}", uri));
    }

    let without_prefix = &uri[5..]; // Remove "s3://"
    let parts: Vec<&str> = without_prefix.splitn(2, '/').collect();

    if parts.len() != 2 {
        return Err(anyhow!(
            "Invalid S3 URI format: missing key part in {}",
            uri
        ));
    }

    let bucket = parts[0].to_string();
    let key = parts[1].to_string();

    if bucket.is_empty() || key.is_empty() {
        return Err(anyhow!(
            "Invalid S3 URI format: empty bucket or key in {}",
            uri
        ));
    }

    Ok((bucket, key))
}

/// AWS S3 file metadata structure
#[derive(Debug, Clone)]
pub struct S3FileMetadata {
    pub bucket: String,
    pub key: String,
    pub size: u64,
    pub etag: String,
    pub region: Option<String>,
}

/// AWS credentials structure for parsing from single file
#[derive(Debug, serde::Deserialize)]
struct AwsCredentials {
    access_key_id: String,
    secret_access_key: String,
    region: Option<String>,
}

/// Create AWS S3 storage client using object_store
/// Reads all AWS configuration from a single file
pub fn s3_storage(
    bucket_name: &str,
    runtime_secret_name: &str,
) -> Result<Box<dyn object_store::ObjectStore>, anyhow::Error> {
    use object_store::aws::{AmazonS3, AmazonS3Builder};

    let mut builder = AmazonS3Builder::new().with_bucket_name(bucket_name);

    // Read AWS credentials from single file
    if let Some(credentials_content) = get_secret_from_file(runtime_secret_name) {
        // Try to parse as JSON first
        if let Ok(credentials) = serde_json::from_str::<AwsCredentials>(&credentials_content) {
            builder = builder
                .with_access_key_id(credentials.access_key_id)
                .with_secret_access_key(credentials.secret_access_key);

            if let Some(region) = credentials.region {
                builder = builder.with_region(region);
            }
        } else {
            // Fallback: try to parse as simple key=value format
            for line in credentials_content.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }

                if let Some((key, value)) = line.split_once('=') {
                    match key.trim().to_lowercase().as_str() {
                        "access_key_id" | "aws_access_key_id" => {
                            builder = builder.with_access_key_id(value.trim());
                        }
                        "secret_access_key" | "aws_secret_access_key" => {
                            builder = builder.with_secret_access_key(value.trim());
                        }
                        "region" | "aws_default_region" => {
                            builder = builder.with_region(value.trim());
                        }
                        _ => {} // Ignore unknown keys
                    }
                }
            }
        }
    }

    let s3: AmazonS3 = builder
        .build()
        .map_err(|e| anyhow!("Failed to create S3 client: {}", e))?;

    Ok(Box::new(s3))
}

/// Single repo wrapper for the main S3 function
pub async fn create_aws_basetenpointers(repo: &ModelRepo) -> Result<Vec<BasetenPointer>> {
    model_cache_s3_to_b10ptr(vec![repo]).await
}

/// Convert S3 ModelRepo to BasetenPointer format
pub async fn model_cache_s3_to_b10ptr(models: Vec<&ModelRepo>) -> Result<Vec<BasetenPointer>> {
    let mut basetenpointers = Vec::new();

    for model in models {
        info!("Processing S3 model: {}", model.repo_id);

        let (bucket, prefix) = parse_s3_uri(&model.repo_id)?;

        // For now, we'll create a simple implementation that assumes the S3 URI points to a specific object
        // In the future, this could be extended to list objects with the given prefix
        let key = prefix;
        let size = 0; // This would need to be fetched from S3 metadata
        let etag = "unknown"; // This would need to be fetched from S3 metadata

        let s3_resolution = S3Resolution::new(bucket.clone(), key.clone(), None);

        let uid = format!("s3:{}:{}", bucket, key);
        let file_name = format!(
            "{}/{}/{}",
            RUNTIME_MODEL_CACHE_PATH,
            model.volume_folder,
            key.split('/').last().unwrap_or(&key)
        );

        let pointer = BasetenPointer {
            resolution: Resolution::S3(s3_resolution),
            uid,
            file_name,
            hashtype: "etag".to_string(),
            hash: etag.to_string(),
            size,
            runtime_secret_name: model.runtime_secret_name.clone(),
        };

        basetenpointers.push(pointer);
    }

    Ok(basetenpointers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_s3_uri() {
        let (bucket, key) = parse_s3_uri("s3://my-bucket/path/to/file.txt").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "path/to/file.txt");

        let (bucket, key) = parse_s3_uri("s3://bucket/single-file").unwrap();
        assert_eq!(bucket, "bucket");
        assert_eq!(key, "single-file");

        // Test error cases
        assert!(parse_s3_uri("invalid-uri").is_err());
        assert!(parse_s3_uri("s3://bucket-only").is_err());
        assert!(parse_s3_uri("s3:///empty-bucket").is_err());
    }
}

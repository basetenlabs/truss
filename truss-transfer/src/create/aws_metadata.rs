use crate::create::object_storage_client::get_client_options;
use crate::secrets::get_secret_from_file;
use anyhow::{anyhow, Result};
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

/// AWS credentials structure for parsing from single file
#[derive(Debug, serde::Deserialize)]
struct AwsCredentials {
    access_key_id: String,
    secret_access_key: String,
    region: String,
    session_token: Option<String>,
}

/// Create AWS S3 storage client using object_store
/// Reads all AWS configuration from a single file
/// Returns concrete AmazonS3 type to support the Signer trait for pre-signed URLs
pub fn s3_storage(
    bucket_name: &str,
    runtime_secret_name: &str,
) -> Result<object_store::aws::AmazonS3, anyhow::Error> {
    use object_store::aws::AmazonS3Builder;

    let mut builder = AmazonS3Builder::new()
        .with_bucket_name(bucket_name)
        .with_client_options(get_client_options());

    // Read AWS credentials from single file
    if let Some(credentials_content) = get_secret_from_file(runtime_secret_name) {
        // Try to parse as JSON first
        if let Ok(credentials) = serde_json::from_str::<AwsCredentials>(&credentials_content) {
            builder = builder
                .with_access_key_id(credentials.access_key_id)
                .with_secret_access_key(credentials.secret_access_key);
            // get Session token if exists, else use access key, else error.
            if let Some(session_token) = credentials.session_token {
                builder = builder.with_token(session_token);
            }

            builder = builder.with_region(credentials.region);
        } else {
            return Err(anyhow!("Failed to parse AWS credentials from JSON. The json needs to be in the format: {{\"access_key_id\": \"...\", \"secret_access_key\": \"...\", \"region\": \"...\"}}"));
        }
    } else {
        return Err(anyhow!(
            "Failed to read AWS credentials from not existing file: {}",
            runtime_secret_name
        ));
    }

    builder
        .build()
        .map_err(|e| anyhow!("Failed to create S3 client: {}", e))
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

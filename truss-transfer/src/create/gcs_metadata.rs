use crate::types::{BasetenPointer, GcsError, ModelRepo};
use object_store::gcp::GoogleCloudStorageBuilder;
use std::fs;

/// Parse GCS URI (gs://bucket/path) into bucket and prefix
pub fn parse_gcs_uri(uri: &str) -> Result<(String, String), GcsError> {
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

pub fn gcs_storage(
    bucket: &str,
    runtime_secret_name: &str,
) -> Result<object_store::gcp::GoogleCloudStorage, GcsError> {
    let secret_path = format!("/secrets/{}", runtime_secret_name);
    let credentials_json = fs::read_to_string(&secret_path).map_err(|e| GcsError::Io(e))?;

    GoogleCloudStorageBuilder::new()
        .with_service_account_key(&credentials_json)
        .with_bucket_name(bucket)
        .build()
        .map_err(GcsError::ObjectStore)
}

/// Convert GCS ModelRepo to BasetenPointer format
/// Single repo wrapper for the main GCS function
pub async fn create_gcs_basetenpointers(
    repo: &ModelRepo,
) -> Result<Vec<BasetenPointer>, anyhow::Error> {
    // Use new common implementation
    use crate::create::{
        common_metadata::create_single_cloud_basetenpointers, gcs_provider::GcsProvider,
    };
    create_single_cloud_basetenpointers(&GcsProvider, repo).await
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

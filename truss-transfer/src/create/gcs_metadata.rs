use crate::create::object_storage_client::get_client_options;
use crate::secrets::get_secret;
use crate::types::GcsError;
use object_store::gcp::GoogleCloudStorageBuilder;

/// Parse GCS URI (gs://bucket/path) into bucket and prefix
pub fn parse_gcs_uri(uri: &str) -> Result<(String, String), GcsError> {
    if !uri.starts_with("gs://") {
        return Err(GcsError::InvalidUri(format!(
            "URI must start with gs://: {uri}"
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
    let key = get_secret(runtime_secret_name).ok_or_else(|| {
        GcsError::InvalidUri(format!(
            "GCS credential '{}' not found in environment variable or file",
            runtime_secret_name
        ))
    })?;

    GoogleCloudStorageBuilder::new()
        .with_service_account_key(&key)
        .with_bucket_name(bucket)
        .with_client_options(get_client_options())
        .build()
        .map_err(GcsError::ObjectStore)
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

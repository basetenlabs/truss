use anyhow::Result;
use object_store::ObjectStore;
use rand;

use super::azure_metadata::{azure_storage, parse_azure_uri};
use crate::create::common_metadata::CloudMetadataProvider;
use crate::types::{AzureResolution, Resolution};

/// Azure implementation of CloudMetadataProvider
pub struct AzureProvider;

#[async_trait::async_trait]
impl CloudMetadataProvider for AzureProvider {
    fn parse_uri(&self, uri: &str) -> Result<(String, String)> {
        let (account, container, blob_prefix) = parse_azure_uri(uri)?;
        // Combine account and container as "bucket" for consistency
        let bucket = format!("{}/{}", account, container);
        Ok((bucket, blob_prefix))
    }

    fn create_object_store(
        &self,
        bucket: &str,
        runtime_secret_name: &str,
    ) -> Result<Box<dyn ObjectStore>> {
        // Extract account name from bucket (format: "account/container")
        let account = bucket.split('/').next().unwrap_or(bucket);
        let azure = azure_storage(account, runtime_secret_name)?;
        Ok(azure)
    }

    fn create_resolution(&self, bucket: &str, object_path: &str) -> Resolution {
        // Extract account and container from bucket
        let parts: Vec<&str> = bucket.split('/').collect();
        let account = parts.get(0).unwrap_or(&"").to_string();
        let container = parts.get(1).unwrap_or(&"").to_string();

        Resolution::Azure(AzureResolution::new(
            account,
            container,
            object_path.to_string(),
        ))
    }

    fn hash_type(&self) -> &'static str {
        "etag"
    }

    fn extract_hash(&self, meta: &object_store::ObjectMeta) -> String {
        meta.e_tag
            .clone()
            .unwrap_or_else(|| format!("azure-{}", rand::random::<u64>()))
    }

    fn generate_uid(&self, bucket: &str, object_path: &str, _hash: &str) -> String {
        // Extract account and container for UID
        let parts: Vec<&str> = bucket.split('/').collect();
        let account = parts.get(0).unwrap_or(&"");
        let container = parts.get(1).unwrap_or(&"");

        format!("azure:{}:{}:{}", account, container, object_path)
    }
}

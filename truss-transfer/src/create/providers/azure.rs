use anyhow::Result;
use async_trait::async_trait;
use log::{debug, warn};
use object_store::ObjectStore;
use rand;

use crate::create::azure_metadata::{azure_storage, parse_azure_uri};
use crate::create::common_metadata::{create_single_cloud_basetenpointers, CloudMetadataProvider};
use crate::create::provider::StorageProvider;
use crate::types::{AzureResolution, BasetenPointer, ModelRepo, Resolution, ResolutionType};

/// Azure implementation of CloudMetadataProvider
pub struct AzureProvider;

impl AzureProvider {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
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

#[async_trait]
impl StorageProvider for AzureProvider {
    fn name(&self) -> &'static str {
        "Azure Blob Storage"
    }

    fn can_handle(&self, repo: &ModelRepo) -> bool {
        repo.repo_id.starts_with("azure://")
            || repo.repo_id.contains(".blob.core.windows.net")
            || matches!(repo.kind, ResolutionType::Azure)
    }

    async fn create_pointers(
        &self,
        repo: &ModelRepo,
        model_path: &String,
    ) -> Result<Vec<BasetenPointer>> {
        debug!(
            "Creating Azure Blob Storage pointers for repo: {}",
            repo.repo_id
        );
        if !self.can_handle(repo) {
            warn!(
                "Azure Blob Storage provider cannot handle repo: {}",
                repo.repo_id
            );
        }
        create_single_cloud_basetenpointers(self, repo, model_path).await
    }
}

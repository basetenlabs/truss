use anyhow::Result;
use async_trait::async_trait;
use log::debug;

use crate::create::azure_metadata::create_azure_basetenpointers;
use crate::create::provider::StorageProvider;
use crate::types::{BasetenPointer, ModelRepo, ResolutionType};

pub struct AzureProvider;

impl AzureProvider {
    pub fn new() -> Self {
        Self
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

    async fn create_pointers(&self, repo: &ModelRepo) -> Result<Vec<BasetenPointer>> {
        debug!(
            "Creating Azure Blob Storage pointers for repo: {}",
            repo.repo_id
        );
        create_azure_basetenpointers(repo).await
    }
}

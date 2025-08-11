use anyhow::Result;
use async_trait::async_trait;
use log::{debug, warn};

use crate::create::hf_metadata::create_hf_basetenpointers;
use crate::create::provider::StorageProvider;
use crate::types::{BasetenPointer, ModelRepo, ResolutionType};

pub struct HuggingFaceProvider;

impl HuggingFaceProvider {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl StorageProvider for HuggingFaceProvider {
    fn name(&self) -> &'static str {
        "HuggingFace"
    }

    fn can_handle(&self, repo: &ModelRepo) -> bool {
        // HuggingFace repos don't start with storage protocol URIs
        !repo.repo_id.starts_with("gs://")
            && !repo.repo_id.starts_with("s3://")
            && !repo.repo_id.starts_with("azure://")
            && matches!(repo.kind, ResolutionType::Http)
    }

    async fn create_pointers(
        &self,
        repo: &ModelRepo,
        model_path: &String,
    ) -> Result<Vec<BasetenPointer>> {
        debug!("Creating HuggingFace pointers for repo: {}", repo.repo_id);
        if !self.can_handle(repo) {
            warn!("HuggingFace provider cannot handle repo: {}", repo.repo_id);
        }
        create_hf_basetenpointers(repo, model_path)
            .await
            .map_err(|e| anyhow::Error::from(e))
    }
}

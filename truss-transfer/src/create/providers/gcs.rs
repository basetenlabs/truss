use anyhow::Result;
use async_trait::async_trait;
use log::debug;

use crate::types::{BasetenPointer, ModelRepo, ResolutionType};
use crate::create::provider::StorageProvider;
use crate::create::gcs_metadata::create_gcs_basetenpointers;

pub struct GcsProvider;

impl GcsProvider {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl StorageProvider for GcsProvider {
    fn name(&self) -> &'static str {
        "Google Cloud Storage"
    }

    fn can_handle(&self, repo: &ModelRepo) -> bool {
        repo.repo_id.starts_with("gs://") || matches!(repo.kind, ResolutionType::Gcs)
    }

    async fn create_pointers(&self, repo: &ModelRepo) -> Result<Vec<BasetenPointer>> {
        debug!("Creating GCS pointers for repo: {}", repo.repo_id);
        create_gcs_basetenpointers(repo).await
    }
}

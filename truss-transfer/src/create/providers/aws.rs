use anyhow::Result;
use async_trait::async_trait;
use log::debug;

use crate::types::{BasetenPointer, ModelRepo, ResolutionType};
use crate::create::provider::StorageProvider;
use crate::create::aws_metadata::create_aws_basetenpointers;

pub struct AwsProvider;

impl AwsProvider {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl StorageProvider for AwsProvider {
    fn name(&self) -> &'static str {
        "Amazon S3"
    }

    fn can_handle(&self, repo: &ModelRepo) -> bool {
        repo.repo_id.starts_with("s3://") || matches!(repo.kind, ResolutionType::S3)
    }

    async fn create_pointers(&self, repo: &ModelRepo) -> Result<Vec<BasetenPointer>> {
        debug!("Creating AWS S3 pointers for repo: {}", repo.repo_id);
        create_aws_basetenpointers(repo).await
    }
}

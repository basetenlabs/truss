use anyhow::Result;
use async_trait::async_trait;
use log::{debug, warn};
use object_store::ObjectStore;
use rand;

use crate::create::aws_metadata::{parse_s3_uri, s3_storage};
use crate::create::common_metadata::{create_single_cloud_basetenpointers, CloudMetadataProvider};
use crate::create::provider::StorageProvider;
use crate::types::{BasetenPointer, ModelRepo, Resolution, ResolutionType, S3Resolution};

/// AWS S3 implementation of CloudMetadataProvider
pub struct AwsProvider;

impl AwsProvider {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl CloudMetadataProvider for AwsProvider {
    fn parse_uri(&self, uri: &str) -> Result<(String, String)> {
        parse_s3_uri(uri)
    }

    fn create_object_store(
        &self,
        bucket: &str,
        runtime_secret_name: &str,
    ) -> Result<Box<dyn ObjectStore>> {
        let s3 = s3_storage(bucket, runtime_secret_name)?;
        Ok(s3)
    }

    fn create_resolution(&self, bucket: &str, object_path: &str) -> Resolution {
        Resolution::S3(S3Resolution::new(
            bucket.to_string(),
            object_path.to_string(),
            None,
        ))
    }

    fn hash_type(&self) -> &'static str {
        "etag"
    }

    fn extract_hash(&self, meta: &object_store::ObjectMeta) -> String {
        meta.e_tag
            .clone()
            .unwrap_or_else(|| format!("s3-{}", rand::random::<u64>()))
    }

    fn generate_uid(&self, bucket: &str, object_path: &str, _hash: &str) -> String {
        format!("s3:{}:{}", bucket, object_path)
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

    async fn create_pointers(
        &self,
        repo: &ModelRepo,
        model_path: &String,
    ) -> Result<Vec<BasetenPointer>> {
        debug!("Creating AWS S3 pointers for repo: {}", repo.repo_id);
        if !self.can_handle(repo) {
            warn!("AWS S3 provider cannot handle repo: {}", repo.repo_id);
        }
        create_single_cloud_basetenpointers(self, repo, model_path).await
    }
}

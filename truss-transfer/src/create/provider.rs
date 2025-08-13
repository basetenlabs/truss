/// Provider trait for different storage backends
/// This allows for easy extension to new providers like AWS S3, Azure Blob Storage, etc.
use anyhow::Result;
use async_trait::async_trait;

use crate::types::{BasetenPointer, ModelRepo};

#[async_trait]
pub trait StorageProvider {
    /// Get the provider name for logging and identification
    fn name(&self) -> &'static str;

    /// Check if this provider can handle the given model repository
    /// TODO: enable usage of this trait.
    fn can_handle(&self, repo: &ModelRepo) -> bool;

    /// Create basetenpointers for files in the repository
    async fn create_pointers(
        &self,
        repo: &ModelRepo,
        model_path: &String,
    ) -> Result<Vec<BasetenPointer>>;
}

/// Factory function to get the appropriate provider for a repository
pub fn get_provider_for_repo(repo: &ModelRepo) -> Result<Box<dyn StorageProvider + Send + Sync>> {
    use super::providers::{AwsProvider, AzureProvider, GcsProvider, HuggingFaceProvider};
    use crate::types::ResolutionType;

    match repo.kind {
        ResolutionType::Http => {
            if repo.repo_id.starts_with("gs://") {
                // Handle the case where it's a GCS URI but marked as HTTP
                Ok(Box::new(GcsProvider::new()))
            } else if repo.repo_id.starts_with("s3://") {
                // Handle the case where it's an S3 URI but marked as HTTP
                Ok(Box::new(AwsProvider::new()))
            } else if repo.repo_id.starts_with("azure://")
                || repo.repo_id.contains(".blob.core.windows.net")
            {
                // Handle the case where it's an Azure URI but marked as HTTP
                Ok(Box::new(AzureProvider::new()))
            } else {
                // Assume it's a HuggingFace repo
                Ok(Box::new(HuggingFaceProvider::new()))
            }
        }
        ResolutionType::Gcs => Ok(Box::new(GcsProvider::new())),
        ResolutionType::S3 => Ok(Box::new(AwsProvider::new())),
        ResolutionType::Azure => Ok(Box::new(AzureProvider::new())),
    }
}

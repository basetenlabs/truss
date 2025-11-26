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
                // Baseten Training should use BASETEN_TRAINING resolution type
                Ok(Box::new(AwsProvider::new(false)))
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
        ResolutionType::S3 => Ok(Box::new(AwsProvider::new(false))),
        ResolutionType::Azure => Ok(Box::new(AzureProvider::new())),
        ResolutionType::BasetenTraining => Ok(Box::new(AwsProvider::new(true))),
    }
}

#[cfg(test)]
mod tests {
    use super::get_provider_for_repo;
    use crate::create::providers::AwsProvider;
    use crate::types::{ModelRepo, ResolutionType};

    #[test]
    fn test_baseten_training_produces_aws_provider_with_true() {
        let repo = ModelRepo {
            repo_id: "s3://test-bucket/test-path".to_string(),
            revision: "main".to_string(),
            allow_patterns: None,
            ignore_patterns: None,
            volume_folder: "test_folder".to_string(),
            runtime_secret_name: "test_secret".to_string(),
            kind: ResolutionType::BasetenTraining,
        };

        let provider = get_provider_for_repo(&repo).expect("Should create provider");

        // Verify it's an AWS provider by checking the name
        assert_eq!(provider.name(), "Amazon S3");

        // To verify use_training_secrets is true, we need to downcast the provider
        // Since we can't easily downcast Box<dyn StorageProvider>, we use unsafe
        // pointer casting which is safe here because we know BasetenTraining
        // always produces an AwsProvider
        let aws_provider = unsafe {
            let raw_ptr = provider.as_ref() as *const dyn super::StorageProvider;
            let aws_ptr = raw_ptr as *const AwsProvider;
            aws_ptr.as_ref().expect("Provider should be AwsProvider")
        };

        assert_eq!(aws_provider.use_training_secrets(), true);

        // Verify the provider can handle the repo
        assert!(provider.can_handle(&repo));
    }
}

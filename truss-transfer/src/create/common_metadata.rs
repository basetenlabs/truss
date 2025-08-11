use anyhow::{anyhow, Result};
use futures_util::stream::StreamExt;
use log::{debug, info};
use object_store::ObjectStore;

use super::filter::{normalize_hash, should_ignore_file};
use crate::types::{BasetenPointer, ModelRepo, Resolution};

/// Common metadata extraction interface for cloud storage providers
#[async_trait::async_trait]
pub trait CloudMetadataProvider {
    /// Parse provider-specific URI into bucket/container and prefix components
    fn parse_uri(&self, uri: &str) -> Result<(String, String)>;

    /// Create object store client for the provider
    fn create_object_store(
        &self,
        bucket: &str,
        runtime_secret_name: &str,
    ) -> Result<Box<dyn ObjectStore>>;

    /// Create provider-specific resolution from bucket and object path
    fn create_resolution(&self, bucket: &str, object_path: &str) -> Resolution;

    /// Get hash type used by this provider (e.g., "md5", "etag")
    fn hash_type(&self) -> &'static str;

    /// Extract hash from object metadata, with fallback generation
    fn extract_hash(&self, meta: &object_store::ObjectMeta) -> String;

    /// Generate unique identifier for the object using hash
    fn generate_uid(&self, bucket: &str, object_path: &str, hash: &str) -> String;
}

/// Generic metadata extraction function that works with any cloud provider
pub async fn extract_cloud_metadata<T: CloudMetadataProvider>(
    provider: &T,
    models: Vec<&ModelRepo>,
    model_path: String,
) -> Result<Vec<BasetenPointer>> {
    let mut basetenpointers = Vec::new();

    for model in models {
        info!(
            "Processing {} model: {}",
            std::any::type_name::<T>(),
            model.repo_id
        );

        let (bucket, prefix) = provider.parse_uri(&model.repo_id)?;

        // Create storage client
        let object_store = provider.create_object_store(&bucket, &model.runtime_secret_name)?;

        // List all objects with the given prefix
        let prefix_path = if prefix.is_empty() {
            object_store::path::Path::from("")
        } else {
            object_store::path::Path::from(prefix.clone())
        };

        let mut list_stream = object_store.list(Some(&prefix_path));

        while let Some(meta) = list_stream
            .next()
            .await
            .transpose()
            .map_err(|e| anyhow!("Failed to list objects: {}", e))?
        {
            let object_path = meta.location.to_string();

            // Extract relative path from the prefix (for file naming)
            let relative_path = if prefix.is_empty() {
                object_path.clone()
            } else {
                object_path
                    .strip_prefix(&format!("{}/", prefix))
                    .unwrap_or(&object_path)
                    .to_string()
            };

            // Apply filtering based on allow/ignore patterns
            if should_ignore_file(
                &relative_path,
                model.allow_patterns.as_deref(),
                model.ignore_patterns.as_deref(),
            ) {
                debug!("Ignoring file: {}", relative_path);
                continue;
            }

            let hash = provider.extract_hash(&meta);
            let resolution = provider.create_resolution(&bucket, &object_path);
            let uid = provider.generate_uid(&bucket, &object_path, &hash);

            let file_name = format!(
                "{}/{}/{}",
                model_path,
                model.volume_folder,
                relative_path.split('/').last().unwrap_or(&relative_path)
            );

            let pointer = BasetenPointer {
                resolution,
                uid,
                file_name,
                hashtype: provider.hash_type().to_string(),
                hash: normalize_hash(&hash),
                size: meta.size,
                runtime_secret_name: model.runtime_secret_name.clone(),
            };

            basetenpointers.push(pointer);
        }
    }

    info!("Created {} basetenpointers", basetenpointers.len());
    Ok(basetenpointers)
}

/// Single repo wrapper for cloud metadata extraction
pub async fn create_single_cloud_basetenpointers<T: CloudMetadataProvider>(
    provider: &T,
    repo: &ModelRepo,
    model_path: &String,
) -> Result<Vec<BasetenPointer>> {
    extract_cloud_metadata(provider, vec![repo], model_path.clone()).await
}

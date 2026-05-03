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

    fn last_modified_time(&self, meta: &object_store::ObjectMeta) -> chrono::DateTime<chrono::Utc> {
        meta.last_modified
    }

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

        // Track per-model object counts so we can fail fast on misconfigured
        // weights sources (bad URI/prefix or filters that match nothing) instead
        // of silently producing an empty manifest and starting the container.
        let mut listed_count: usize = 0;
        let mut kept_for_model: usize = 0;

        while let Some(meta) = list_stream
            .next()
            .await
            .transpose()
            .map_err(|e| anyhow!("Failed to list objects under '{}': {}", model.repo_id, e))?
        {
            listed_count += 1;
            let object_path = meta.location.to_string();

            // Extract relative path from the prefix (for file naming)
            let relative_path = if prefix.is_empty() {
                object_path.clone()
            } else {
                // Try multiple formats to handle trailing slashes correctly
                let prefix_with_slash = if prefix.ends_with('/') {
                    prefix.clone()
                } else {
                    format!("{prefix}/")
                };
                // stip away the bucket name.
                object_path
                    .strip_prefix(&prefix_with_slash)
                    .or_else(|| object_path.strip_prefix(&prefix))
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
            let last_modified_time = provider.last_modified_time(&meta);
            let resolution = provider.create_resolution(&bucket, &object_path);
            let uid = provider.generate_uid(&bucket, &object_path, &hash);

            let file_name = format!("{}/{}/{}", model_path, model.volume_folder, relative_path);

            let pointer = BasetenPointer {
                resolution,
                uid,
                file_name,
                hashtype: provider.hash_type().to_string(),
                hash: normalize_hash(&hash),
                size: meta.size,
                last_modified_time: Some(last_modified_time),
                runtime_secret_name: model.runtime_secret_name.clone(),
            };

            basetenpointers.push(pointer);
            kept_for_model += 1;
        }

        if listed_count == 0 {
            return Err(anyhow!(
                "No objects found at '{}'. Verify the URI/prefix exists and that the provided \
                 credentials ('{}') have list access to it.",
                model.repo_id,
                model.runtime_secret_name
            ));
        }
        if kept_for_model == 0 {
            return Err(anyhow!(
                "No files at '{}' matched the configured allow_patterns={:?} / \
                 ignore_patterns={:?}. Adjust the patterns or remove them to include files.",
                model.repo_id,
                model.allow_patterns,
                model.ignore_patterns
            ));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GcsResolution, ResolutionType};
    use bytes::Bytes;
    use object_store::memory::InMemory;
    use object_store::path::Path as ObjPath;
    use std::sync::Arc;

    /// Test provider that wraps an in-memory ObjectStore so we can drive
    /// extract_cloud_metadata without hitting the network. Holds an
    /// Arc<dyn ObjectStore> (which has a blanket ObjectStore impl in the
    /// crate) so we can hand out shared, pre-populated stores via
    /// create_object_store while keeping the trait's Box<dyn ObjectStore>
    /// return type.
    struct TestProvider {
        store: Arc<dyn ObjectStore>,
        prefix: String,
    }

    #[async_trait::async_trait]
    impl CloudMetadataProvider for TestProvider {
        fn parse_uri(&self, _uri: &str) -> Result<(String, String)> {
            Ok(("test-bucket".to_string(), self.prefix.clone()))
        }

        fn create_object_store(
            &self,
            _bucket: &str,
            _runtime_secret_name: &str,
        ) -> Result<Box<dyn ObjectStore>> {
            Ok(Box::new(Arc::clone(&self.store)))
        }

        fn create_resolution(&self, bucket: &str, object_path: &str) -> Resolution {
            Resolution::Gcs(GcsResolution::new(
                object_path.to_string(),
                bucket.to_string(),
            ))
        }

        fn hash_type(&self) -> &'static str {
            "etag"
        }

        fn extract_hash(&self, meta: &object_store::ObjectMeta) -> String {
            meta.e_tag
                .clone()
                .unwrap_or_else(|| "test-etag".to_string())
        }

        fn generate_uid(&self, bucket: &str, object_path: &str, _hash: &str) -> String {
            format!("test:{bucket}:{object_path}")
        }
    }

    fn make_repo(allow: Option<Vec<String>>, ignore: Option<Vec<String>>) -> ModelRepo {
        ModelRepo {
            repo_id: "s3://test-bucket/missing/prefix".to_string(),
            revision: "".to_string(),
            allow_patterns: allow,
            ignore_patterns: ignore,
            volume_folder: "test_vol".to_string(),
            runtime_secret_name: "aws-secret-json".to_string(),
            kind: ResolutionType::S3,
        }
    }

    #[tokio::test]
    async fn errors_when_listing_returns_zero_objects() {
        // Empty in-memory store -> `list` yields nothing.
        let provider = TestProvider {
            store: Arc::new(InMemory::new()),
            prefix: "missing/prefix".to_string(),
        };
        let repo = make_repo(None, None);

        let err = extract_cloud_metadata(&provider, vec![&repo], "/app/model_cache".to_string())
            .await
            .expect_err("expected an error when no objects are listed");

        let msg = err.to_string();
        assert!(
            msg.contains("No objects found"),
            "unexpected error message: {msg}"
        );
        assert!(
            msg.contains(&repo.repo_id),
            "error should reference repo_id, got: {msg}"
        );
        assert!(
            msg.contains(&repo.runtime_secret_name),
            "error should reference runtime secret name, got: {msg}"
        );
    }

    #[tokio::test]
    async fn errors_when_all_files_filtered_out() {
        // Populate the store with two real objects under the prefix.
        let store = InMemory::new();
        for name in &["model.bin", "config.json"] {
            store
                .put(
                    &ObjPath::from(format!("data/{name}")),
                    Bytes::from_static(b"hello").into(),
                )
                .await
                .unwrap();
        }
        let provider = TestProvider {
            store: Arc::new(store),
            prefix: "data".to_string(),
        };

        // allow_patterns matches none of the listed objects -> all filtered out.
        let repo = make_repo(Some(vec!["*.zzz".to_string()]), None);

        let err = extract_cloud_metadata(&provider, vec![&repo], "/app/model_cache".to_string())
            .await
            .expect_err("expected an error when filters drop every object");

        let msg = err.to_string();
        assert!(
            msg.contains("No files at"),
            "unexpected error message: {msg}"
        );
        assert!(
            msg.contains("*.zzz"),
            "error message should mention the failing allow_pattern, got: {msg}"
        );
    }

    #[tokio::test]
    async fn succeeds_when_at_least_one_file_matches() {
        let store = InMemory::new();
        store
            .put(
                &ObjPath::from("data/model.safetensors"),
                Bytes::from_static(b"weights").into(),
            )
            .await
            .unwrap();
        store
            .put(
                &ObjPath::from("data/README.md"),
                Bytes::from_static(b"docs").into(),
            )
            .await
            .unwrap();
        let provider = TestProvider {
            store: Arc::new(store),
            prefix: "data".to_string(),
        };

        let repo = make_repo(Some(vec!["*.safetensors".to_string()]), None);

        let pointers =
            extract_cloud_metadata(&provider, vec![&repo], "/app/model_cache".to_string())
                .await
                .expect("expected success when at least one file matches");
        assert_eq!(pointers.len(), 1);
        assert!(pointers[0].file_name.ends_with("model.safetensors"));
    }
}

use anyhow::Result;
use object_store::ObjectStore;
use rand;

use super::gcs_metadata::{gcs_storage, parse_gcs_uri};
use crate::create::common_metadata::CloudMetadataProvider;
use crate::types::{GcsResolution, Resolution};

/// GCS implementation of CloudMetadataProvider
pub struct GcsProvider;

#[async_trait::async_trait]
impl CloudMetadataProvider for GcsProvider {
    fn parse_uri(&self, uri: &str) -> Result<(String, String)> {
        parse_gcs_uri(uri).map_err(Into::into)
    }

    fn create_object_store(
        &self,
        bucket: &str,
        runtime_secret_name: &str,
    ) -> Result<Box<dyn ObjectStore>> {
        let gcs = gcs_storage(bucket, runtime_secret_name)?;
        Ok(Box::new(gcs))
    }

    fn create_resolution(&self, bucket: &str, object_path: &str) -> Resolution {
        Resolution::Gcs(GcsResolution::new(
            object_path.to_string(),
            bucket.to_string(),
        ))
    }

    fn hash_type(&self) -> &'static str {
        "md5"
    }

    fn extract_hash(&self, meta: &object_store::ObjectMeta) -> String {
        // Extract MD5 hash from GCS metadata or use ETag as fallback
        meta.e_tag
            .as_ref()
            .and_then(|etag| {
                // GCS ETags often contain MD5 hash in quotes
                if etag.starts_with('"') && etag.ends_with('"') && etag.len() == 34 {
                    Some(etag[1..33].to_string()) // Extract MD5 from quoted ETag
                } else {
                    None
                }
            })
            .unwrap_or_else(|| format!("gcs-{}", rand::random::<u64>()))
    }

    fn generate_uid(&self, _bucket: &str, _object_path: &str, hash: &str) -> String {
        format!("gcs-{}", hash)
    }
}

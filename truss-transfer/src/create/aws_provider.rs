use anyhow::Result;
use object_store::ObjectStore;
use rand;

use super::aws_metadata::{parse_s3_uri, s3_storage};
use crate::create::common_metadata::CloudMetadataProvider;
use crate::types::{Resolution, S3Resolution};

/// AWS S3 implementation of CloudMetadataProvider
pub struct AwsProvider;

#[async_trait::async_trait]
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

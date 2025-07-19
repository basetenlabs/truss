use serde::{Deserialize, Serialize};

use super::ResolutionType;

fn default_s3_resolution_type() -> ResolutionType {
    ResolutionType::S3
}

/// S3 resolution with bucket name and region
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct S3Resolution {
    pub bucket_name: String,
    pub key: String,
    pub region: Option<String>,
    #[serde(default = "default_s3_resolution_type")]
    resolution_type: ResolutionType,
}

impl S3Resolution {
    pub fn new(bucket_name: String, key: String, region: Option<String>) -> Self {
        Self {
            bucket_name,
            key,
            region,
            resolution_type: ResolutionType::S3,
        }
    }
}

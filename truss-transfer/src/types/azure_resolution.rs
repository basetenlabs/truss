use serde::{Deserialize, Serialize};

use super::ResolutionType;

fn default_azure_resolution_type() -> ResolutionType {
    ResolutionType::Azure
}

/// Azure Blob Storage resolution with container and blob path
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AzureResolution {
    pub account_name: String,
    pub container_name: String,
    pub blob_name: String,
    #[serde(default = "default_azure_resolution_type", skip_serializing)]
    resolution_type: ResolutionType,
}

impl AzureResolution {
    pub fn new(account_name: String, container_name: String, blob_name: String) -> Self {
        Self {
            account_name,
            container_name,
            blob_name,
            resolution_type: ResolutionType::Azure,
        }
    }
}

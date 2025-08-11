use anyhow::{anyhow, Result};

use crate::secrets::get_secret_from_file;

/// Parse Azure Blob Storage URI into account, container, and blob components
/// Expected format: azure://accountname/containername/path/to/blob
/// or https://accountname.blob.core.windows.net/containername/path/to/blob
pub fn parse_azure_uri(uri: &str) -> Result<(String, String, String)> {
    if uri.starts_with("azure://") {
        let without_prefix = &uri[8..]; // Remove "azure://"
        let parts: Vec<&str> = without_prefix.splitn(3, '/').collect();

        if parts.len() != 3 {
            return Err(anyhow!(
                "Invalid Azure URI format: expected azure://account/container/blob, got {}",
                uri
            ));
        }

        let account = parts[0].to_string();
        let container = parts[1].to_string();
        let blob = parts[2].to_string();

        if account.is_empty() || container.is_empty() || blob.is_empty() {
            return Err(anyhow!(
                "Invalid Azure URI format: empty account, container, or blob in {}",
                uri
            ));
        }

        Ok((account, container, blob))
    } else if uri.contains(".blob.core.windows.net") {
        // Parse https://accountname.blob.core.windows.net/containername/path/to/blob
        let url =
            url::Url::parse(uri).map_err(|_| anyhow!("Invalid Azure Blob Storage URL: {}", uri))?;

        let host = url
            .host_str()
            .ok_or_else(|| anyhow!("No host found in Azure URL: {}", uri))?;

        let account = host
            .split('.')
            .next()
            .ok_or_else(|| anyhow!("Could not extract account name from host: {}", host))?
            .to_string();

        let path = url.path();
        if path.is_empty() || path == "/" {
            return Err(anyhow!("No path found in Azure URL: {}", uri));
        }

        let path_parts: Vec<&str> = path.trim_start_matches('/').splitn(2, '/').collect();
        if path_parts.len() != 2 {
            return Err(anyhow!(
                "Invalid Azure URL path format: expected /container/blob, got {}",
                path
            ));
        }

        let container = path_parts[0].to_string();
        let blob = path_parts[1].to_string();

        Ok((account, container, blob))
    } else {
        Err(anyhow!("Invalid Azure URI format: must start with azure:// or be a valid Azure Blob Storage URL"))
    }
}

/// Azure credentials structure for parsing from single file
#[derive(Debug, serde::Deserialize)]
struct AzureCredentials {
    account_key: String,
    #[serde(default)]
    use_emulator: bool,
}

/// Create Azure Blob Storage client using object_store
/// Reads all Azure configuration from a single file
pub fn azure_storage(
    account_name: &str,
    runtime_secret_name: &str,
) -> Result<Box<dyn object_store::ObjectStore>, anyhow::Error> {
    use object_store::azure::{MicrosoftAzure, MicrosoftAzureBuilder};

    let mut builder = MicrosoftAzureBuilder::new().with_account(account_name);

    // Read Azure credentials from single file
    if let Some(credentials_content) = get_secret_from_file(runtime_secret_name) {
        // Try to parse as JSON first
        if let Ok(credentials) = serde_json::from_str::<AzureCredentials>(&credentials_content) {
            builder = builder.with_access_key(credentials.account_key);

            // Note: SAS token support would require proper URL parsing and query parameter handling
            // For now, we focus on access key authentication
            // else if let Some(sas_token) = credentials.sas_token {
            //     builder = builder.with_sas_token(sas_token);
            // }

            if credentials.use_emulator {
                builder = builder.with_use_emulator(true);
            }
        } else {
            return Err(anyhow!("Failed to parse Azure credentials from JSON. The json needs to be in the format: {{\"account_key\": \"...\"}}"));
        }
    } else {
        return Err(anyhow!(
            "Failed to read Azure credentials from not existing file: {}",
            runtime_secret_name
        ));
    }

    let azure: MicrosoftAzure = builder
        .build()
        .map_err(|e| anyhow!("Failed to create Azure client: {}", e))?;

    Ok(Box::new(azure))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_azure_uri() {
        // Test azure:// format
        let (account, container, blob) =
            parse_azure_uri("azure://myaccount/mycontainer/path/to/file.txt").unwrap();
        assert_eq!(account, "myaccount");
        assert_eq!(container, "mycontainer");
        assert_eq!(blob, "path/to/file.txt");

        // Test HTTPS URL format
        let (account, container, blob) =
            parse_azure_uri("https://myaccount.blob.core.windows.net/mycontainer/path/to/file.txt")
                .unwrap();
        assert_eq!(account, "myaccount");
        assert_eq!(container, "mycontainer");
        assert_eq!(blob, "path/to/file.txt");

        let (account, container, blob) =
            parse_azure_uri("azure://account/container/single-file").unwrap();
        assert_eq!(account, "account");
        assert_eq!(container, "container");
        assert_eq!(blob, "single-file");

        // Test error cases
        assert!(parse_azure_uri("invalid-uri").is_err());
        assert!(parse_azure_uri("azure://account-only").is_err());
        assert!(parse_azure_uri("azure://account/container-only").is_err());
        assert!(parse_azure_uri("azure:///empty-account/container/blob").is_err());
    }
}

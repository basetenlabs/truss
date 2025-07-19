use log::{debug, warn};
use std::fs;
use std::path::Path;

use crate::constants::SECRETS_BASE_PATH;

/// Get secret from file system based on runtime secret name
/// Returns None if the secret file doesn't exist or can't be read
pub fn get_secret_from_file(runtime_secret_name: &str) -> Option<String> {
    let secret_path = Path::new(SECRETS_BASE_PATH).join(runtime_secret_name);
    debug!("Attempting to read secret from {:?}", secret_path);
    
    match fs::read_to_string(&secret_path) {
        Ok(content) => {
            let trimmed = content.trim();
            if trimmed.is_empty() {
                debug!("Secret file {:?} is empty", secret_path);
                None
            } else {
                debug!("Secret found in {:?}", secret_path);
                Some(trimmed.to_string())
            }
        }
        Err(_) => {
            warn!(
                "No secret found in {path}. Using unauthenticated access. Make sure to set `{name}` in your Baseten.co secrets and add `secrets:- {name}: null` to your config.yaml.",
                path = secret_path.display(),
                name = runtime_secret_name
            );
            None
        }
    }
}

/// Check if runtime secret name corresponds to a HuggingFace token
pub fn is_hf_token(runtime_secret_name: &str) -> bool {
    runtime_secret_name.contains("hf_access_token") || runtime_secret_name == "hf_token"
}

/// Check if runtime secret name corresponds to a GCS service account
pub fn is_gcs_service_account(runtime_secret_name: &str) -> bool {
    runtime_secret_name.contains("gcs") || runtime_secret_name.contains("service_account")
}

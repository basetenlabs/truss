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

pub fn get_secret_path(runtime_secret_name: &str) -> String {
    Path::new(SECRETS_BASE_PATH)
        .join(runtime_secret_name)
        .display()
        .to_string()
}

/// Get HuggingFace token from multiple sources
/// 1. Check file system at /secrets/{runtime_secret_name}
/// 2. Check environment variables: HF_TOKEN or HUGGING_FACE_HUB_TOKEN
/// 3. Return None if not found
pub fn get_hf_secret_from_file(hf_token_name: &str) -> Option<String> {
    if let Some(token) = get_secret_from_file(hf_token_name) {
        Some(token)
    } else if let Ok(token) = std::env::var("HF_TOKEN") {
        warn!(
            "No secret found in {}, using HF_TOKEN environment variable",
            hf_token_name
        );
        Some(token.trim().to_string())
    } else if let Ok(token) = std::env::var("HUGGING_FACE_HUB_TOKEN") {
        warn!(
            "No secret found in {}, using HUGGING_FACE_HUB_TOKEN environment variable",
            hf_token_name
        );
        Some(token.trim().to_string())
    } else {
        None
    }
}

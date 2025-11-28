use log::{debug, warn};
use once_cell::sync::Lazy;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::constants::{HF_TOKEN, SECRETS_BASE_PATH, SECRET_PATH_WHITELIST};

static WARNED_SECRETS: Lazy<Mutex<HashSet<String>>> = Lazy::new(|| Mutex::new(HashSet::new()));

/// Check if the full secret path is allowed based on whitelist prefixes
fn is_secret_path_allowed(path_read: PathBuf) -> bool {
    let path_str = path_read.to_string_lossy();
    SECRET_PATH_WHITELIST
        .iter()
        .any(|prefix| path_str.starts_with(prefix))
}

/// Get secret from file system based on runtime secret name
/// Returns None if the secret file doesn't exist or can't be read
pub fn get_secret_from_file(runtime_secret_name: &str) -> Option<String> {
    let secret_path = Path::new(SECRETS_BASE_PATH).join(runtime_secret_name);

    if !is_secret_path_allowed(secret_path.clone()) {
        warn!(
            "Secret path '{}' does not match any allowed prefix in whitelist",
            secret_path.display()
        );
        return None;
    }

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
            let mut warned_secrets = WARNED_SECRETS.lock().unwrap();
            if warned_secrets.insert(runtime_secret_name.to_string()) {
                warn!(
                    "No secret found in {path}. Using unauthenticated access. Make sure to set `{name}` in your Baseten.co secrets and add `secrets:- {name}: null` to your config.yaml.",
                    path = secret_path.display(),
                    name = runtime_secret_name
                );
            }
            None
        }
    }
}

pub fn get_secret_path(runtime_secret_name: &str) -> String {
    let secret_path = Path::new(SECRETS_BASE_PATH).join(runtime_secret_name);

    if !is_secret_path_allowed(secret_path.clone()) {
        warn!(
            "Secret path '{}' does not match any allowed prefix in whitelist",
            secret_path.display()
        );
        return String::new();
    }

    secret_path.display().to_string()
}

/// Get HuggingFace token from multiple sources
/// 1. Check file system at /secrets/{runtime_secret_name}
/// 2. Check environment variables: HF_TOKEN or HUGGING_FACE_HUB_TOKEN
/// 3. Return None if not found
pub fn get_hf_secret_from_file(hf_token_name: &str) -> Option<String> {
    if let Some(token) = get_secret_from_file(hf_token_name) {
        Some(token)
    } else {
        (*HF_TOKEN).clone()
    }
}

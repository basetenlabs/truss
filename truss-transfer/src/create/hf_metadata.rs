use super::{filter_repo_files, normalize_hash};
use crate::secrets::get_hf_secret_from_file;
use crate::types::{BasetenPointer, HttpResolution, ModelRepo, Resolution, ResolutionType};
use hf_hub::api::tokio::{Api, ApiBuilder};
use hf_hub::{Repo, RepoType};
use log::{debug, warn};
use std::collections::HashMap;
use std::env;
use std::path::Path;
use tokio::time::{sleep, Duration};

/// Get HuggingFace token from multiple sources
/// 1. Check file system at /secrets/{runtime_secret_name}
/// 2. Check environment variables: HF_TOKEN or HUGGING_FACE_HUB_TOKEN
/// 3. Return None if not found
fn get_hf_token(runtime_secret_name: &str) -> Option<String> {
    // 1. Try to read from secrets file
    let secret = get_hf_secret_from_file(runtime_secret_name);
    if secret.is_some() {
        return secret;
    }

    // 2. Try environment variables
    // Check for HF_TOKEN first (common in many setups)
    if let Ok(token) = env::var("HF_TOKEN") {
        let trimmed = token.trim().to_string();
        if !trimmed.is_empty() {
            debug!("Found HF token in HF_TOKEN environment variable");
            return Some(trimmed);
        }
    }

    // Check for HUGGING_FACE_HUB_TOKEN (official HF environment variable)
    if let Ok(token) = env::var("HUGGING_FACE_HUB_TOKEN") {
        let trimmed = token.trim().to_string();
        if !trimmed.is_empty() {
            debug!("Found HF token in HUGGING_FACE_HUB_TOKEN environment variable");
            return Some(trimmed);
        }
    }

    // 3. No token found
    warn!(
        "No HuggingFace token found secrets or environment variables (HF_TOKEN, HUGGING_FACE_HUB_TOKEN). Using unauthenticated access.",
    );
    None
}

/// Error types for HuggingFace operations
#[derive(Debug, thiserror::Error)]
pub enum HfError {
    #[error("HuggingFace API error: {0}")]
    Api(#[from] hf_hub::api::tokio::ApiError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Timeout error after retries")]
    Timeout,
    #[error("Pattern matching error: {0}")]
    Pattern(String),
    #[error("Invalid metadata")]
    InvalidMetadata,
}

/// HuggingFace file metadata
/// todo: use the one from the hf-hub crate
/// and dont resolve yourself
#[derive(Debug, Clone)]
pub struct HfFileMetadata {
    pub etag: String,
    pub url: String,
    pub size: u64,
}

/// Get HuggingFace file metadata using the hf-hub crate
pub async fn get_hf_metadata(
    api: &Api,
    repo_id: &str,
    revision: &str,
    filename: &str,
) -> Result<HfFileMetadata, HfError> {
    let repo = Repo::with_revision(repo_id.to_string(), RepoType::Model, revision.to_string());
    let api_repo = api.repo(repo);

    // Create the URL for the file
    let url = api_repo.url(filename);

    // Use reqwest to get metadata, should not be needed.
    // better to resolve it using the hf-hub crate
    let client = reqwest::Client::new();
    let response = client
        .head(&url)
        .send()
        .await
        .map_err(|_e| HfError::InvalidMetadata)?;

    let etag = response
        .headers()
        .get("etag")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .replace('"', "");

    let size = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    Ok(HfFileMetadata { etag, url, size })
}

/// Get metadata for all files in a HuggingFace repository
pub async fn metadata_hf_repo(
    repo_id: &str,
    revision: &str,
    allow_patterns: Option<&[String]>,
    ignore_patterns: Option<&[String]>,
    token: Option<String>,
) -> Result<HashMap<String, HfFileMetadata>, HfError> {
    let api = ApiBuilder::new()
        .with_token(token)
        .build()
        .map_err(|_e| HfError::InvalidMetadata)?;
    let repo = Repo::with_revision(repo_id.to_string(), RepoType::Model, revision.to_string());
    let api_repo = api.repo(repo);

    // Get repository info to get the actual revision and file list
    let repo_info = api_repo
        .info()
        .await
        .map_err(|_e| HfError::InvalidMetadata)?;
    let real_revision = &repo_info.sha;

    if revision != real_revision {
        eprintln!(
            "Warning: revision {} is moving, using {} instead. \
            Please update your code to use `revision={}` instead otherwise you will keep moving.",
            revision, real_revision, real_revision
        );
    }

    // Extract file names from siblings
    let files: Vec<String> = repo_info
        .siblings
        .into_iter()
        .map(|s| s.rfilename)
        .collect();
    let filtered_files = filter_repo_files(files, allow_patterns, ignore_patterns)?;

    let mut metadata_map: HashMap<String, HfFileMetadata> = HashMap::new();

    for file in filtered_files {
        let metadata = get_hf_metadata(&api, repo_id, real_revision, &file).await?;
        metadata_map.insert(file, metadata);
    }

    Ok(metadata_map)
}

pub async fn create_hf_basetenpointers(
    model: &ModelRepo,
    model_path: &String,
) -> Result<Vec<BasetenPointer>, HfError> {
    let mut basetenpointers = Vec::new();

    // skip if its not a hf repo
    if model.kind != ResolutionType::Http {
        return Ok(basetenpointers);
    }
    let mut exception: Option<HfError> = None;
    let mut metadata_result = None;

    // Get token for this model
    let token = get_hf_token(&model.runtime_secret_name);

    // Retry mechanism (up to 3 times)
    for attempt in 0..3 {
        match metadata_hf_repo(
            &model.repo_id,
            &model.revision,
            model.allow_patterns.as_deref(),
            model.ignore_patterns.as_deref(),
            token.clone(),
        )
        .await
        {
            Ok(metadata) => {
                metadata_result = Some(metadata);
                break;
            }
            Err(e) => {
                eprintln!("Attempt {} failed: {:?}", attempt + 1, e);
                exception = Some(e);
                if attempt < 2 {
                    sleep(Duration::from_secs(5)).await;
                }
            }
        }
    }

    let metadata_hf_repo_list = match metadata_result {
        Some(metadata) => metadata,
        None => return Err(exception.unwrap_or(HfError::Timeout)),
    };

    // Convert metadata to BasetenPointer format
    for (filename, metadata) in metadata_hf_repo_list {
        let uid = format!("{}:{}:{}", model.repo_id, model.revision, filename);
        let runtime_path = format!("{}/{}", model_path, model.volume_folder);
        let file_path = Path::new(&runtime_path).join(&filename);

        let pointer = BasetenPointer {
            resolution: Resolution::Http(HttpResolution::new(
                metadata.url,
                4044816725, // 90 years in the future
            )),
            uid,
            file_name: file_path.to_string_lossy().to_string(),
            hashtype: "etag".to_string(),
            hash: normalize_hash(&metadata.etag),
            size: metadata.size,
            runtime_secret_name: model.runtime_secret_name.clone(),
        };

        basetenpointers.push(pointer);
    }

    Ok(basetenpointers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_repo_files_allow_patterns() {
        let files = vec![
            "model.safetensors".to_string(),
            "config.json".to_string(),
            "tokenizer.json".to_string(),
            "README.md".to_string(),
        ];

        let allow_patterns = vec!["*.json".to_string()];
        let result = filter_repo_files(files, Some(&allow_patterns), None).unwrap();

        assert_eq!(result.len(), 2);
        assert!(result.contains(&"config.json".to_string()));
        assert!(result.contains(&"tokenizer.json".to_string()));
    }

    #[test]
    fn test_filter_repo_files_ignore_patterns() {
        let files = vec![
            "model.safetensors".to_string(),
            "config.json".to_string(),
            "tokenizer.json".to_string(),
            "README.md".to_string(),
        ];

        let ignore_patterns = vec!["*.md".to_string(), "*.json".to_string()];
        let result = filter_repo_files(files, None, Some(&ignore_patterns)).unwrap();

        assert_eq!(result.len(), 1);
        assert!(result.contains(&"model.safetensors".to_string()));
    }

    #[tokio::test]
    async fn test_model_repo_creation() {
        let model_repo = ModelRepo {
            repo_id: "test/model".to_string(),
            revision: "main".to_string(),
            allow_patterns: None,
            kind: ResolutionType::Http,
            ignore_patterns: Some(vec!["*.md".to_string()]),
            volume_folder: "test".to_string(),
            runtime_secret_name: "hf_access_token".to_string(),
        };

        assert_eq!(model_repo.repo_id, "test/model");
        assert_eq!(model_repo.revision, "main");
        assert_eq!(model_repo.volume_folder, "test");
        assert_eq!(model_repo.runtime_secret_name, "hf_access_token");
    }

    #[test]
    fn test_get_hf_token_priority() {
        // Test that environment variables have the right priority
        use std::env;

        // Set both environment variables
        env::set_var("HF_TOKEN", "hf_token_value");
        env::set_var("HUGGING_FACE_HUB_TOKEN", "hf_hub_token_value");

        // HF_TOKEN should take precedence over HUGGING_FACE_HUB_TOKEN
        let token = get_hf_token("hf_access_token");
        assert_eq!(token, Some("hf_token_value".to_string()));

        // Clean up
        env::remove_var("HF_TOKEN");
        env::remove_var("HUGGING_FACE_HUB_TOKEN");
    }
}

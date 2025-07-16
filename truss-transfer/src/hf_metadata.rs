use crate::types::{BasetenPointer, ModelRepo, Resolution, ResolutionType};
use hf_hub::api::tokio::{Api, ApiBuilder};
use hf_hub::{Repo, RepoType};
use std::collections::HashMap;
use std::path::Path;
use std::vec;
use tokio::time::{sleep, Duration};

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

/// Filter repository files based on patterns
pub fn filter_repo_files(
    files: Vec<String>,
    allow_patterns: Option<&[String]>,
    ignore_patterns: Option<&[String]>,
) -> Result<Vec<String>, HfError> {
    let mut filtered_files = files;

    // Apply allow patterns (if specified, only keep files that match)
    if let Some(patterns) = allow_patterns {
        filtered_files = filtered_files
            .into_iter()
            .filter(|file| {
                patterns.iter().any(|pattern| {
                    glob::Pattern::new(pattern)
                        .map_err(|e| HfError::Pattern(e.to_string()))
                        .map(|p| p.matches(file))
                        .unwrap_or(false)
                })
            })
            .collect();
    }

    // Apply ignore patterns (remove files that match)
    if let Some(patterns) = ignore_patterns {
        filtered_files = filtered_files
            .into_iter()
            .filter(|file| {
                !patterns.iter().any(|pattern| {
                    glob::Pattern::new(pattern)
                        .map_err(|e| HfError::Pattern(e.to_string()))
                        .map(|p| p.matches(file))
                        .unwrap_or(false)
                })
            })
            .collect();
    }

    Ok(filtered_files)
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
) -> Result<HashMap<String, HfFileMetadata>, HfError> {
    let api = ApiBuilder::new()
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

    let mut metadata_map = HashMap::new();

    for file in filtered_files {
        let metadata = get_hf_metadata(&api, repo_id, real_revision, &file).await?;
        metadata_map.insert(file, metadata);
    }

    Ok(metadata_map)
}

/// Convert ModelCache to BasetenPointer format
pub async fn model_cache_hf_to_b10ptr(
    cache: Vec<ModelRepo>,
) -> Result<Vec<BasetenPointer>, HfError> {
    let mut basetenpointers = Vec::new();

    for model in &cache {
        let mut exception: Option<HfError> = None;
        let mut metadata_result = None;

        // Retry mechanism (up to 3 times)
        for attempt in 0..3 {
            match metadata_hf_repo(
                &model.repo_id,
                &model.revision,
                model.allow_patterns.as_deref(),
                model.ignore_patterns.as_deref(),
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
            let runtime_path = format!("/app/model_cache/{}", model.volume_folder);
            let file_path = Path::new(&runtime_path).join(&filename);

            let pointer = BasetenPointer {
                resolution: Some(Resolution {
                    url: metadata.url,
                    resolution_type: ResolutionType::Http,
                    expiration_timestamp: 4044816725, // 90 years in the future
                }),
                uid,
                file_name: file_path.to_string_lossy().to_string(),
                hashtype: "etag".to_string(),
                hash: metadata.etag,
                size: metadata.size,
                runtime_secret_name: model.runtime_secret_name.clone(),
            };

            basetenpointers.push(pointer);
        }
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
            ignore_patterns: Some(vec!["*.md".to_string()]),
            volume_folder: "test".to_string(),
            runtime_secret_name: "hf_access_token".to_string(),
        };

        assert_eq!(model_repo.repo_id, "test/model");
        assert_eq!(model_repo.revision, "main");
        assert_eq!(model_repo.volume_folder, "test");
        assert_eq!(model_repo.runtime_secret_name, "hf_access_token");
    }
}

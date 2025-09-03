use super::{filter_repo_files, normalize_hash};
use crate::secrets::get_hf_secret_from_file;
use crate::types::{BasetenPointer, HttpResolution, ModelRepo, Resolution, ResolutionType};
use hf_hub::api::tokio::{Api, ApiBuilder};
use hf_hub::{Repo, RepoType};
use std::collections::HashMap;
use std::path::Path;
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
    #[error("Invalid metadata: {0}")]
    InvalidMetadata(String),
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
    token: Option<String>,
) -> Result<HfFileMetadata, HfError> {
    // Define HuggingFace-specific header constants
    const HUGGINGFACE_HEADER_X_LINKED_ETAG: &str = "X-Linked-Etag";
    const HUGGINGFACE_HEADER_X_LINKED_SIZE: &str = "X-Linked-Size";

    let repo = Repo::with_revision(repo_id.to_string(), RepoType::Model, revision.to_string());
    let api_repo = api.repo(repo);

    // Create the URL for the file
    let url = api_repo.url(filename);

    // TODO: client with different redicted policy, see python huggingface_hub method.
    let client = reqwest::Client::builder()
        .build()
        .map_err(|e| HfError::InvalidMetadata(format!("Failed to build request client: {e}")))?;

    // Build request with Accept-Encoding header
    let mut req = client.head(&url).header("Accept-Encoding", "identity");

    // Add token if provided
    if let Some(t) = token.as_ref() {
        req = req.header("Authorization", format!("Bearer {}", t));
    }

    // Send initial request
    let response = req
        .send()
        .await
        .map_err(|e| HfError::InvalidMetadata(format!("Failed to send request to {url}: {e}")))?;

    let headers = response.headers().clone();
    response
        .error_for_status()
        .map_err(|e| HfError::Pattern(format!("HTTP Error: {e}")))?;

    // Extract etag from headers
    let etag = headers
        .get(HUGGINGFACE_HEADER_X_LINKED_ETAG)
        .or_else(|| headers.get("etag"))
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .replace('"', ""); // Remove quotes from etag

    // Extract size from headers
    let size = headers
        .get(HUGGINGFACE_HEADER_X_LINKED_SIZE)
        .or_else(|| headers.get("content-length"))
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
        .with_token(token.clone())
        .build()
        .map_err(|e| HfError::InvalidMetadata(format!("Failed to build HuggingFace API: {e}")))?;
    let repo = Repo::with_revision(repo_id.to_string(), RepoType::Model, revision.to_string());
    let api_repo = api.repo(repo);

    // Get repository info to get the actual revision and file list
    let repo_info = api_repo.info().await.map_err(|e| {
        HfError::InvalidMetadata(format!("Failed to get repo info for {repo_id}: {e}"))
    })?;
    let real_revision = &repo_info.sha;

    if revision != real_revision {
        eprintln!(
            "Warning: Huggingface revision {revision} is not fixed, using {real_revision} instead. \
            Please update your code to use `revision={real_revision}` instead otherwise its unclear which files you want to download + \
            and will update in future deployments."
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
        let metadata = get_hf_metadata(&api, repo_id, real_revision, &file, token.clone()).await?;
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
    let token = get_hf_secret_from_file(&model.runtime_secret_name);

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
}

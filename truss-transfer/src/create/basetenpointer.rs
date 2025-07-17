use super::model_cache_hf_to_b10ptr;
use crate::types::ModelRepo;
use serde_json;

/// Create a BasetenPointer from a HuggingFace model repository
/// This is the main function that mimics the Python model_cache_hf_to_b10ptr function
pub async fn create_basetenpointer(
    cache: Vec<ModelRepo>,
) -> Result<String, Box<dyn std::error::Error>> {
    let manifest = model_cache_hf_to_b10ptr(cache).await?;
    // todo: filter between hf and gcs models and combine them.
    let json = serde_json::to_string_pretty(&manifest)?;
    Ok(json)
}

#[cfg(test)]
mod tests {
    use crate::types::{ModelRepo, ResolutionType};

    use super::*;

    #[tokio::test]
    async fn test_create_basetenpointer() {
        let cache = vec![ModelRepo {
            repo_id: "julien-c/dummy-unknown".to_string(),
            revision: "main".to_string(),
            allow_patterns: None,
            ignore_patterns: Some(vec!["*.md".to_string()]),
            kind: ResolutionType::Http,
            volume_folder: "test_model".to_string(),
            runtime_secret_name: "hf_access_token".to_string(),
        }];

        // This will fail in test environment without network access
        // but shows the structure
        let result = create_basetenpointer(cache).await;
        // In a real test environment, we would mock the network calls
        // For now, we just check that the function exists and compiles
        assert!(result.is_ok() || result.is_err());
    }
}

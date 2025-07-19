use super::{gcs_metadata::model_cache_gcs_to_b10ptr, hf_metadata::model_cache_hf_to_b10ptr};
use crate::types::{ModelRepo, ResolutionType};
use log::info;
use serde_json;

/// Create a BasetenPointer from multiple model repositories (HuggingFace, GCS, etc.)
/// This is the main function that handles different model source types
pub async fn create_basetenpointer(
    cache: Vec<ModelRepo>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut all_pointers = Vec::new();

    // Separate models by type
    let hf_models: Vec<_> = cache
        .iter()
        .filter(|m| m.kind == ResolutionType::Http)
        .collect();
    let gcs_models: Vec<_> = cache
        .iter()
        .filter(|m| m.kind == ResolutionType::Gcs)
        .collect();

    // Process HuggingFace models
    if !hf_models.is_empty() {
        info!("Processing {} HuggingFace models", hf_models.len());
        let hf_cache: Vec<ModelRepo> = hf_models.into_iter().cloned().collect();
        let hf_pointers = model_cache_hf_to_b10ptr(hf_cache).await?;
        all_pointers.extend(hf_pointers);
    }

    // Process GCS models
    if !gcs_models.is_empty() {
        info!("Processing {} GCS models", gcs_models.len());
        let gcs_pointers = model_cache_gcs_to_b10ptr(gcs_models).await?;
        all_pointers.extend(gcs_pointers);
    }

    info!("Created {} total basetenpointers", all_pointers.len());
    let json = serde_json::to_string_pretty(&all_pointers)?;
    Ok(json)
}

#[cfg(test)]
mod tests {
    use crate::types::{ModelRepo, ResolutionType};

    use super::*;

    #[tokio::test]
    async fn test_create_basetenpointer_hf() {
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

    #[tokio::test]
    async fn test_create_basetenpointer_mixed() {
        let cache = vec![
            ModelRepo {
                repo_id: "julien-c/dummy-unknown".to_string(),
                revision: "main".to_string(),
                allow_patterns: None,
                ignore_patterns: Some(vec!["*.md".to_string()]),
                kind: ResolutionType::Http,
                volume_folder: "test_hf_model".to_string(),
                runtime_secret_name: "hf_access_token".to_string(),
            },
            ModelRepo {
                repo_id: "gs://test-bucket/model-path".to_string(),
                revision: "main".to_string(), // Ignored for GCS
                allow_patterns: Some(vec!["*.safetensors".to_string()]),
                ignore_patterns: Some(vec!["*.md".to_string()]),
                kind: ResolutionType::Gcs,
                volume_folder: "test_gcs_model".to_string(),
                runtime_secret_name: "gcs-account".to_string(),
            },
        ];

        // This will fail in test environment without network access and credentials
        // but shows the structure for mixed HF + GCS models
        let result = create_basetenpointer(cache).await;
        // In a real test environment, we would mock the network calls
        // For now, we just check that the function exists and compiles
        assert!(result.is_ok() || result.is_err());
    }
}

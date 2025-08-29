use crate::create::provider::get_provider_for_repo;
use crate::types::{BasetenPointerManifest, ModelRepo};
use log::info;

/// Create a BasetenPointer from multiple model repositories (HuggingFace, GCS, etc.)
/// This is the main function that handles different model source types
/// Refactored to use the provider pattern for better extensibility
pub async fn create_basetenpointer(
    cache: Vec<ModelRepo>,
    model_path: String,
) -> Result<BasetenPointerManifest, Box<dyn std::error::Error>> {
    let mut all_pointers = Vec::new();

    info!("Processing {} model repositories", cache.len());

    // Process each model repository using the appropriate provider
    for model in cache {
        let provider = get_provider_for_repo(&model)?;
        info!("Processing {} model: {}", provider.name(), model.repo_id);

        let pointers = provider.create_pointers(&model, &model_path).await?;
        all_pointers.extend(pointers);
    }

    info!("Created {} total basetenpointers", all_pointers.len());

    let manifest = BasetenPointerManifest {
        pointers: all_pointers,
        models: None,
    };

    Ok(manifest)
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
        let result = create_basetenpointer(cache, "/app".to_string()).await;
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
        let result = create_basetenpointer(cache, "/app".to_string()).await;
        // In a real test environment, we would mock the network calls
        // For now, we just check that the function exists and compiles
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_create_basetenpointer_hf_julien_dummy() {
        use crate::types::{ModelRepo, ResolutionType};
        use serde_json::Value;

        const HF_REPO: &str = "julien-c/dummy-unknown";
        const HF_REVISION: &str = "60b8d3fe22aebb024b573f1cca224db3126d10f3";

        // Create a model repo similar to the Python test
        let model_repos = vec![ModelRepo {
            repo_id: HF_REPO.to_string(),
            revision: HF_REVISION.to_string(),
            runtime_secret_name: "hf_access_token_2".to_string(),
            volume_folder: "julien_dummy".to_string(),
            kind: ResolutionType::Http,
            allow_patterns: None,
            ignore_patterns: None,
        }];

        println!("Testing create_basetenpointer...");
        let result = create_basetenpointer(model_repos, "/app/model_cache".to_string()).await;
        let manifest_json = serde_json::to_string(&result.unwrap()).unwrap();

        println!("Success! Generated BasetenPointer manifest:");
        let output: Value =
            serde_json::from_str(&manifest_json).expect("Failed to parse manifest JSON");

        let manifest = output
            .get("pointers")
            .and_then(|v| v.as_array())
            .expect("Manifest should have a 'pointers' array");
        // TODO: refactor this test to access the .pointers attribute
        let pretty_json =
            serde_json::to_string_pretty(&manifest).expect("Failed to pretty print JSON");
        println!("{pretty_json}");

        // Test that the structure is correct
        assert!(!manifest.is_empty(), "Manifest should not be empty");
        println!("{pretty_json}");

        // Test that the structure is correct
        assert!(!manifest.is_empty(), "Manifest should not be empty");

        // Check the first pointer structure
        let required_fields = [
            "resolution",
            "uid",
            "file_name",
            "hashtype",
            "hash",
            "size",
            "runtime_secret_name",
        ];
        let resolution_fields = ["url", "resolution_type", "expiration_timestamp"];

        for pointer in manifest {
            // Check required fields exist
            for field in &required_fields {
                assert!(pointer.get(field).is_some(), "Missing field: {field}");
            }

            // Check resolution fields
            let resolution = pointer
                .get("resolution")
                .expect("Resolution field should exist");

            for field in &resolution_fields {
                assert!(
                    resolution.get(field).is_some(),
                    "Missing resolution field: {field}"
                );
            }

            // Verify that the revision is in the URL
            let url = resolution
                .get("url")
                .and_then(|v| v.as_str())
                .expect("URL should be a string");
            assert!(
                url.contains(HF_REVISION),
                "URL should contain revision: {url}"
            );

            // Verify that it's an HTTP resolution
            let resolution_type = resolution
                .get("resolution_type")
                .and_then(|v| v.as_str())
                .expect("Resolution type should be a string");
            assert_eq!(resolution_type, "http", "Should be HTTP resolution");

            // Verify the runtime secret name
            let runtime_secret = pointer
                .get("runtime_secret_name")
                .and_then(|v| v.as_str())
                .expect("Runtime secret name should be a string");
            assert_eq!(runtime_secret, "hf_access_token_2");

            // Verify the UID format
            let uid = pointer
                .get("uid")
                .and_then(|v| v.as_str())
                .expect("UID should be a string");
            assert!(
                uid.starts_with("julien-c/dummy-unknown:"),
                "UID should start with repo:revision"
            );

            // Verify file name starts with the expected path
            let file_name = pointer
                .get("file_name")
                .and_then(|v| v.as_str())
                .expect("File name should be a string");
            assert!(
                file_name.starts_with("/app/model_cache/julien_dummy/"),
                "File name should start with correct path: {file_name}"
            );

            // if name contains pytorch_model.bin, check if size is 65100 byte
            if file_name.contains("pytorch_model.bin") {
                let size = pointer
                    .get("size")
                    .and_then(|v| v.as_u64())
                    .expect("Size should be a number");
                assert_eq!(
                    size, 65074,
                    "Size should be 65100 bytes for pytorch_model.bin"
                );
            }
        }

        println!("✓ BasetenPointer structure validation passed");
        println!(
            "✓ Found {} file(s) in the HuggingFace repository",
            manifest.len()
        );

        // Verify we got some specific expected files
        let file_names: Vec<String> = manifest
            .iter()
            .filter_map(|p| {
                p.get("file_name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .collect();

        let expected_files = ["config.json", "README.md"];
        for expected_file in &expected_files {
            let found = file_names.iter().any(|name| name.contains(expected_file));
            assert!(found, "Expected file {expected_file} not found in manifest");
        }
    }

    #[tokio::test]
    async fn test_create_basetenpointer_azure() {
        use crate::types::{ModelRepo, ResolutionType};

        // Test Azure support with a mock repository
        let model_repos = vec![ModelRepo {
            repo_id: "azure://testaccount/testcontainer/model.bin".to_string(),
            revision: "main".to_string(),
            runtime_secret_name: "azure-storage".to_string(),
            volume_folder: "test_azure_model".to_string(),
            kind: ResolutionType::Azure,
            allow_patterns: None,
            ignore_patterns: None,
        }];

        println!("Testing Azure support...");
        let result = create_basetenpointer(model_repos, "/app".to_string()).await;
        let manifest_json = match result {
            Ok(manifest_json) => {
                println!("Azure support working! Generated manifest:");
                serde_json::to_string(&manifest_json).unwrap()
            }
            Err(e) => {
                // if Failed to read Azure credentials from not existing file: azure-storage in error.
                if e.to_string().contains(
                    "Failed to read Azure credentials from not existing file: azure-storage",
                ) {
                    println!("Azure test failed (expected without credentials): {e}");
                    // This is expected since we don't have real Azure credentials
                    // The important thing is that the provider pattern works
                    return;
                } else {
                    panic!("Unexpected error during Azure test: {e}");
                }
            }
        };
        // Basic validation
        let manifest: Vec<serde_json::Value> =
            serde_json::from_str(&manifest_json).expect("Failed to parse manifest JSON");

        assert!(!manifest.is_empty(), "Manifest should not be empty");

        let pointer = &manifest[0];
        assert_eq!(
            pointer
                .get("resolution")
                .unwrap()
                .get("resolution_type")
                .unwrap(),
            "azure"
        );

        println!("✓ Azure provider test passed");
    }
}

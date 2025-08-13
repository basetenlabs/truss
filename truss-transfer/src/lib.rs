// Modular truss_transfer library
// This library provides functionality for downloading and caching files
// with support for Python and CLI bindings

// Module declarations
mod bindings;
mod cache;
mod constants;
mod core;
mod create;
mod download;
mod download_core;
mod secrets;
mod speed_checks;
mod types;

// CLI module (only when cli feature is enabled)
#[cfg(feature = "cli")]
mod cli;

// Re-export the main public API to maintain compatibility
pub use bindings::{lazy_data_resolve, truss_transfer};
pub use cache::cleanup_b10cache_and_get_space_stats;
pub use core::lazy_data_resolve_entrypoint;

// Re-export the main function for CLI usage
#[cfg(feature = "cli")]
pub use cli::main;

// Re-export types for external use
pub use types::{
    BasetenPointer, BasetenPointerManifest, GcsResolution, HttpResolution, ModelRepo, Resolution,
    ResolutionType,
};

// Re-export HuggingFace functionality
pub use create::{metadata_hf_repo, HfError};

// Re-export BasetenPointer API
pub use create::create_basetenpointer;

// Tests module
#[cfg(test)]
mod tests {
    use crate::bindings::resolve_truss_transfer_download_dir;
    use crate::constants::*;
    use crate::types::*;
    use std::env;

    #[test]
    fn test_resolve_truss_transfer_download_dir_with_arg() {
        // If an argument is provided, it should take precedence.
        let dir = "my/download/dir".to_string();
        let result = resolve_truss_transfer_download_dir(Some(dir.clone()));
        assert_eq!(result, dir);
    }

    #[test]
    fn test_resolve_truss_transfer_download_dir_from_env() {
        // Set the environment variable and ensure it is used.
        let test_dir = "env_download_dir".to_string();
        env::set_var(TRUSS_TRANSFER_DOWNLOAD_DIR_ENV_VAR, test_dir.clone());
        let result = resolve_truss_transfer_download_dir(None);
        assert_eq!(result, test_dir);
        env::remove_var(TRUSS_TRANSFER_DOWNLOAD_DIR_ENV_VAR);
    }

    #[test]
    fn test_resolve_truss_transfer_download_dir_fallback() {
        // Ensure that when no arg and no env var are provided, the fallback is returned.
        env::remove_var(TRUSS_TRANSFER_DOWNLOAD_DIR_ENV_VAR);
        let result = resolve_truss_transfer_download_dir(None);
        assert_eq!(result, TRUSS_TRANSFER_DOWNLOAD_DIR_FALLBACK.to_string());
    }

    #[test]
    fn test_build_resolution_map_valid() {
        // Create a pointer with an expiration timestamp in the future.
        let future_timestamp = chrono::Utc::now().timestamp() + 3600; // one hour in the future
        let pointer = BasetenPointer {
            resolution: Resolution::Http(HttpResolution::new(
                "http://example.com/file".into(),
                future_timestamp,
            )),
            uid: "123".into(),
            file_name: "file.txt".into(),
            hashtype: "sha256".into(),
            hash: "abcdef".into(),
            size: 1024,
            runtime_secret_name: "hf_access_token".into(),
        };
        let manifest = BasetenPointerManifest {
            pointers: vec![pointer],
        };
        let result = crate::core::build_resolution_map(&manifest);
        assert!(result.is_ok());
        let map = result.unwrap();
        assert_eq!(map.len(), 1);
        assert_eq!(map[0].0, "file.txt");
    }

    #[test]
    fn test_build_resolution_map_expired() {
        // Create a pointer that has already expired.
        let past_timestamp = chrono::Utc::now().timestamp() - 3600; // one hour in the past
        let pointer = BasetenPointer {
            resolution: Resolution::Http(HttpResolution::new(
                "http://example.com/file".into(),
                past_timestamp,
            )),
            uid: "123".into(),
            file_name: "file.txt".into(),
            hashtype: "sha256".into(),
            hash: "abcdef".into(),
            size: 1024,
            runtime_secret_name: "hf_access_token".into(),
        };
        let manifest = BasetenPointerManifest {
            pointers: vec![pointer],
        };
        let _result = crate::core::build_resolution_map(&manifest);
        // used to be that we raise an error here, but now we just log a warning
    }

    #[test]
    fn test_init_logger_once() {
        // Calling init_logger_once multiple times should not panic.
        crate::bindings::init_logger_once();
        crate::bindings::init_logger_once();
    }

    #[test]
    fn test_build_resolution_map_invalid_hash_with_slash() {
        // Create a pointer with a hash containing a slash (should fail)
        let future_timestamp = chrono::Utc::now().timestamp() + 3600;
        let pointer = BasetenPointer {
            resolution: Resolution::Http(HttpResolution::new(
                "http://example.com/file".into(),
                future_timestamp,
            )),
            uid: "123".into(),
            file_name: "file.txt".into(),
            hashtype: "sha256".into(),
            hash: "abc/def".into(), // Invalid hash with slash
            size: 1024,
            runtime_secret_name: "hf_access_token".into(),
        };
        let manifest = BasetenPointerManifest {
            pointers: vec![pointer],
        };
        let result = crate::core::build_resolution_map(&manifest);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Hash"));
    }

    #[test]
    fn test_build_resolution_map_with_resolution() {
        // Create a pointer with valid resolution (should succeed)
        let future_timestamp = chrono::Utc::now().timestamp() + 3600;
        let pointer = BasetenPointer {
            resolution: Resolution::Http(HttpResolution::new(
                "http://example.com/file".into(),
                future_timestamp,
            )),
            uid: "123".into(),
            file_name: "file.txt".into(),
            hashtype: "sha256".into(),
            hash: "abcdef".into(),
            size: 1024,
            runtime_secret_name: "hf_access_token".into(),
        };
        let manifest = BasetenPointerManifest {
            pointers: vec![pointer],
        };
        let result = crate::core::build_resolution_map(&manifest);
        assert!(result.is_ok()); // Should still be OK as resolution is checked only when available
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_merge_manifests_basic() {
        let future_timestamp = chrono::Utc::now().timestamp() + 3600;

        // Create two manifests with different files
        let manifest1 = BasetenPointerManifest {
            pointers: vec![BasetenPointer {
                resolution: Resolution::Http(HttpResolution::new(
                    "http://example.com/file1".into(),
                    future_timestamp,
                )),
                uid: "123".into(),
                file_name: "file1.txt".into(),
                hashtype: "sha256".into(),
                hash: "hash1".into(),
                size: 1024,
                runtime_secret_name: "hf_access_token".into(),
            }],
        };

        let manifest2 = BasetenPointerManifest {
            pointers: vec![BasetenPointer {
                resolution: Resolution::Http(HttpResolution::new(
                    "http://example.com/file2".into(),
                    future_timestamp,
                )),
                uid: "456".into(),
                file_name: "file2.txt".into(),
                hashtype: "sha256".into(),
                hash: "hash2".into(),
                size: 2048,
                runtime_secret_name: "hf_access_token".into(),
            }],
        };

        let result = crate::core::merge_manifests(vec![manifest1, manifest2]);
        assert!(result.is_ok());
        let merged = result.unwrap();
        assert_eq!(merged.pointers.len(), 2);

        // Verify both files are present
        let file_names: Vec<&str> = merged
            .pointers
            .iter()
            .map(|p| p.file_name.as_str())
            .collect();
        assert!(file_names.contains(&"file1.txt"));
        assert!(file_names.contains(&"file2.txt"));
    }

    #[test]
    fn test_merge_manifests_duplicates() {
        let future_timestamp = chrono::Utc::now().timestamp() + 3600;

        // Create two manifests with the same file (same name and hash)
        let pointer = BasetenPointer {
            resolution: Resolution::Http(HttpResolution::new(
                "http://example.com/file".into(),
                future_timestamp,
            )),
            uid: "123".into(),
            file_name: "file.txt".into(),
            hashtype: "sha256".into(),
            hash: "samehash".into(),
            size: 1024,
            runtime_secret_name: "hf_access_token".into(),
        };

        let manifest1 = BasetenPointerManifest {
            pointers: vec![pointer.clone()],
        };

        let manifest2 = BasetenPointerManifest {
            pointers: vec![pointer],
        };

        let result = crate::core::merge_manifests(vec![manifest1, manifest2]);
        assert!(result.is_ok());
        let merged = result.unwrap();
        assert_eq!(merged.pointers.len(), 1); // Should deduplicate
    }

    #[test]
    fn test_merge_manifests_conflicts() {
        let future_timestamp = chrono::Utc::now().timestamp() + 3600;

        // Create two manifests with files that have the same name but different hashes
        let pointer1 = BasetenPointer {
            resolution: Resolution::Http(HttpResolution::new(
                "http://example.com/file".into(),
                future_timestamp,
            )),
            uid: "123".into(),
            file_name: "file.txt".into(),
            hashtype: "sha256".into(),
            hash: "hash1".into(),
            size: 1024,
            runtime_secret_name: "hf_access_token".into(),
        };

        let pointer2 = BasetenPointer {
            resolution: Resolution::Http(HttpResolution::new(
                "http://example.com/file".into(),
                future_timestamp,
            )),
            uid: "456".into(),
            file_name: "file.txt".into(), // Same name
            hashtype: "sha256".into(),
            hash: "hash2".into(), // Different hash
            size: 2048,
            runtime_secret_name: "hf_access_token".into(),
        };

        let manifest1 = BasetenPointerManifest {
            pointers: vec![pointer1],
        };

        let manifest2 = BasetenPointerManifest {
            pointers: vec![pointer2],
        };

        let result = crate::core::merge_manifests(vec![manifest1, manifest2]);
        assert!(result.is_ok());
        let merged = result.unwrap();
        assert_eq!(merged.pointers.len(), 1); // Should keep only the first one
        assert_eq!(merged.pointers[0].hash, "hash1"); // Should keep the first hash
    }

    #[test]
    fn test_merge_manifests_empty() {
        let result = crate::core::merge_manifests(vec![]);
        assert!(result.is_ok());
        let merged = result.unwrap();
        assert_eq!(merged.pointers.len(), 0);
    }

    #[test]
    fn test_current_hashes_from_manifest() {
        let future_timestamp = chrono::Utc::now().timestamp() + 3600;

        let manifest = BasetenPointerManifest {
            pointers: vec![
                BasetenPointer {
                    resolution: Resolution::Http(HttpResolution::new(
                        "http://example.com/file1".into(),
                        future_timestamp,
                    )),
                    uid: "123".into(),
                    file_name: "file1.txt".into(),
                    hashtype: "sha256".into(),
                    hash: "hash1".into(),
                    size: 1024,
                    runtime_secret_name: "hf_access_token".into(),
                },
                BasetenPointer {
                    resolution: Resolution::Http(HttpResolution::new(
                        "http://example.com/file2".into(),
                        future_timestamp,
                    )),
                    uid: "456".into(),
                    file_name: "file2.txt".into(),
                    hashtype: "sha256".into(),
                    hash: "hash2".into(),
                    size: 2048,
                    runtime_secret_name: "hf_access_token".into(),
                },
            ],
        };

        let hashes = crate::core::current_hashes_from_manifest(&manifest);
        assert_eq!(hashes.len(), 2);
        assert!(hashes.contains("hash1"));
        assert!(hashes.contains("hash2"));
    }

    #[test]
    fn test_constants_values() {
        // Test that constants have expected values
        assert_eq!(TRUSS_TRANSFER_DOWNLOAD_DIR_FALLBACK, "/tmp/bptr-resolved");
        assert_eq!(CACHE_DIR, "/cache/org/artifacts/truss_transfer_managed_v1");
        assert_eq!(BLOB_DOWNLOAD_TIMEOUT_SECS, 21600);
        assert_eq!(*TRUSS_TRANSFER_NUM_WORKERS, 32);
        assert_eq!(*TRUSS_TRANSFER_B10FS_CLEANUP_HOURS, 4 * 24);
        assert_eq!(TRUSS_TRANSFER_B10FS_DOWNLOAD_SPEED_MBPS, 350.0);
        assert_eq!(TRUSS_TRANSFER_B10FS_MIN_REQUIRED_AVAILABLE_SPACE_GB, 100);
        assert_eq!(B10FS_BENCHMARK_SIZE, 128 * 1024 * 1024);

        // Test environment variable names
        assert_eq!(BASETEN_FS_ENABLED_ENV_VAR, "BASETEN_FS_ENABLED");
        assert_eq!(
            TRUSS_TRANSFER_DOWNLOAD_DIR_ENV_VAR,
            "TRUSS_TRANSFER_DOWNLOAD_DIR"
        );
        assert_eq!(
            TRUSS_TRANSFER_B10FS_DOWNLOAD_SPEED_ENV_VAR,
            "TRUSS_TRANSFER_B10FS_DOWNLOAD_SPEED_MBPS"
        );

        assert_eq!(SECRETS_BASE_PATH, "/secrets");

        // Test manifest paths
        assert_eq!(LAZY_DATA_RESOLVER_PATHS.len(), 4);
        assert!(LAZY_DATA_RESOLVER_PATHS.contains(&"/bptr/bptr-manifest"));
        assert!(LAZY_DATA_RESOLVER_PATHS.contains(&"/bptr/bptr-manifest.json"));
        assert!(LAZY_DATA_RESOLVER_PATHS.contains(&"/bptr/static-bptr-manifest.json"));
        assert!(LAZY_DATA_RESOLVER_PATHS.contains(&"/static-bptr/static-bptr-manifest.json"));
    }

    #[test]
    fn test_resolution_type_conversion() {
        // Test that ResolutionType enum has expected values
        let http_resolution = ResolutionType::Http;
        let gcs_resolution = ResolutionType::Gcs;

        // These should be different
        assert_ne!(http_resolution, gcs_resolution);

        // Test that they can be used in Resolution enum
        let resolution = Resolution::Http(HttpResolution::new(
            "http://example.com/file".into(),
            chrono::Utc::now().timestamp() + 3600,
        ));
        match resolution {
            Resolution::Http(_) => assert_eq!(http_resolution, ResolutionType::Http),
            _ => panic!("Expected HTTP resolution"),
        }
    }

    #[test]
    fn test_model_repo_creation() {
        // Test ModelRepo struct creation
        let repo = ModelRepo {
            repo_id: "test/repo".into(),
            revision: "main".into(),
            allow_patterns: Some(vec!["*.txt".into()]),
            ignore_patterns: Some(vec!["*.log".into()]),
            volume_folder: "test_folder".into(),
            runtime_secret_name: "hf_token".into(),
            kind: ResolutionType::Http,
        };

        assert_eq!(repo.repo_id, "test/repo");
        assert_eq!(repo.revision, "main");
        assert_eq!(repo.volume_folder, "test_folder");
        assert_eq!(repo.runtime_secret_name, "hf_token");
        assert_eq!(repo.kind, ResolutionType::Http);
        assert_eq!(repo.allow_patterns.as_ref().unwrap().len(), 1);
        assert_eq!(repo.ignore_patterns.as_ref().unwrap().len(), 1);
    }
}

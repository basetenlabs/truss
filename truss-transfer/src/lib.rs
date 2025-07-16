// Modular truss_transfer library
// This library provides functionality for downloading and caching files
// with support for Python and CLI bindings

// Module declarations
mod bindings;
mod cache;
mod constants;
mod core;
mod download;
mod speed_checks;
mod types;

// Re-export the main public API to maintain compatibility
pub use bindings::{lazy_data_resolve, truss_transfer};
pub use cache::cleanup_b10cache_and_get_space_stats;
pub use core::lazy_data_resolve_entrypoint;

// Re-export the main function for CLI usage
#[cfg(feature = "cli")]
pub use bindings::main;

// Re-export types for external use
pub use types::{BasetenPointer, BasetenPointerManifest, Resolution, ResolutionType};

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
            resolution: Resolution {
                url: "http://example.com/file".into(),
                resolution_type: ResolutionType::Http,
                expiration_timestamp: future_timestamp,
            },
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
            resolution: Resolution {
                url: "http://example.com/file".into(),
                resolution_type: ResolutionType::Http,
                expiration_timestamp: past_timestamp,
            },
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
        assert!(result.is_err());
    }

    #[test]
    fn test_init_logger_once() {
        // Calling init_logger_once multiple times should not panic.
        crate::bindings::init_logger_once();
        crate::bindings::init_logger_once();
    }
}

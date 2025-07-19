use std::env;
use std::path::{Path, PathBuf};

use crate::constants::*;

/// Configuration for truss_transfer operations
#[derive(Debug, Clone)]
pub struct TrussConfig {
    /// Directory where downloads are placed (can be different from model cache)
    pub download_dir: PathBuf,
    /// Directory where models are expected to be in the container runtime
    pub model_cache_path: PathBuf,
    /// Base path for secrets
    pub secrets_path: PathBuf,
    /// Cache directory for b10fs
    pub cache_dir: PathBuf,
    /// Whether to keep local copies after moving to b10fs
    pub keep_local_copies: bool,
    /// Number of download workers
    pub num_workers: usize,
    /// B10fs cleanup hours
    pub b10fs_cleanup_hours: u64,
}

impl Default for TrussConfig {
    fn default() -> Self {
        Self {
            download_dir: PathBuf::from(TRUSS_TRANSFER_DOWNLOAD_DIR_FALLBACK),
            model_cache_path: PathBuf::from(RUNTIME_MODEL_CACHE_PATH_DEFAULT),
            secrets_path: PathBuf::from(SECRETS_BASE_PATH),
            cache_dir: PathBuf::from(CACHE_DIR),
            keep_local_copies: true, // Changed default to keep copies
            num_workers: TRUSS_TRANSFER_NUM_WORKERS_DEFAULT,
            b10fs_cleanup_hours: TRUSS_TRANSFER_B10FS_DEFAULT_CLEANUP_HOURS,
        }
    }
}

impl TrussConfig {
    /// Create a new configuration from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // Download directory
        if let Ok(download_dir) = env::var(TRUSS_TRANSFER_DOWNLOAD_DIR_ENV_VAR) {
            config.download_dir = PathBuf::from(download_dir);
        }

        // Model cache path
        if let Ok(model_cache_path) = env::var(TRUSS_TRANSFER_MODEL_CACHE_PATH_ENV_VAR) {
            config.model_cache_path = PathBuf::from(model_cache_path);
        }

        // Keep local copies setting
        if let Ok(keep_local) = env::var("TRUSS_TRANSFER_KEEP_LOCAL_COPIES") {
            config.keep_local_copies = keep_local.to_lowercase() == "true";
        }

        // Number of workers
        if let Ok(workers_str) = env::var("TRUSS_TRANSFER_NUM_WORKERS") {
            if let Ok(workers) = workers_str.parse::<usize>() {
                config.num_workers = workers;
            }
        }

        // B10fs cleanup hours
        if let Ok(cleanup_hours_str) = env::var(TRUSS_TRANSFER_B10FS_CLEANUP_HOURS_ENV_VAR) {
            if let Ok(cleanup_hours) = cleanup_hours_str.parse::<u64>() {
                config.b10fs_cleanup_hours = cleanup_hours;
            }
        }

        config
    }

    /// Get the download directory, ensuring it exists
    pub fn ensure_download_dir(&self) -> std::io::Result<&Path> {
        std::fs::create_dir_all(&self.download_dir)?;
        Ok(&self.download_dir)
    }

    /// Get the model cache path for a specific volume folder
    pub fn get_model_cache_path(&self, volume_folder: &str) -> PathBuf {
        self.model_cache_path.join(volume_folder)
    }

    /// Get the secrets path for a specific secret name
    pub fn get_secrets_path(&self, secret_name: &str) -> PathBuf {
        self.secrets_path.join(secret_name)
    }

    /// Resolve the actual file path where a file should be downloaded
    /// This allows flexibility between download dir and final model cache location
    pub fn resolve_download_path(&self, file_name: &str, volume_folder: &str) -> PathBuf {
        if file_name.starts_with('/') {
            // Absolute path - use as is but potentially redirect to download dir
            if file_name.starts_with(&self.model_cache_path.to_string_lossy().as_ref()) {
                // If it's pointing to model cache, use download dir instead during download
                let relative_path = file_name.strip_prefix(&self.model_cache_path.to_string_lossy().as_ref())
                    .unwrap_or(file_name);
                self.download_dir.join(relative_path.trim_start_matches('/'))
            } else {
                PathBuf::from(file_name)
            }
        } else {
            // Relative path - put in volume folder within download dir
            self.download_dir.join(volume_folder).join(file_name)
        }
    }

    /// Get the final destination path where the file should end up
    pub fn get_final_destination(&self, file_name: &str) -> PathBuf {
        if file_name.starts_with('/') {
            PathBuf::from(file_name)
        } else {
            // This shouldn't happen with well-formed BasetenPointer, but handle gracefully
            self.model_cache_path.join(file_name)
        }
    }
}

/// Resolve the download directory from environment or use fallback
pub fn resolve_download_dir() -> PathBuf {
    env::var(TRUSS_TRANSFER_DOWNLOAD_DIR_ENV_VAR)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(TRUSS_TRANSFER_DOWNLOAD_DIR_FALLBACK))
}

/// Resolve the model cache path from environment or use default
pub fn resolve_model_cache_path() -> PathBuf {
    env::var(TRUSS_TRANSFER_MODEL_CACHE_PATH_ENV_VAR)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(RUNTIME_MODEL_CACHE_PATH_DEFAULT))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_config_default() {
        let config = TrussConfig::default();
        assert_eq!(config.download_dir, PathBuf::from(TRUSS_TRANSFER_DOWNLOAD_DIR_FALLBACK));
        assert_eq!(config.model_cache_path, PathBuf::from(RUNTIME_MODEL_CACHE_PATH_DEFAULT));
        assert!(config.keep_local_copies);
    }

    #[test]
    fn test_config_from_env() {
        env::set_var(TRUSS_TRANSFER_DOWNLOAD_DIR_ENV_VAR, "/custom/download");
        env::set_var(TRUSS_TRANSFER_MODEL_CACHE_PATH_ENV_VAR, "/custom/model_cache");
        env::set_var("TRUSS_TRANSFER_KEEP_LOCAL_COPIES", "false");

        let config = TrussConfig::from_env();

        assert_eq!(config.download_dir, PathBuf::from("/custom/download"));
        assert_eq!(config.model_cache_path, PathBuf::from("/custom/model_cache"));
        assert!(!config.keep_local_copies);

        // Cleanup
        env::remove_var(TRUSS_TRANSFER_DOWNLOAD_DIR_ENV_VAR);
        env::remove_var(TRUSS_TRANSFER_MODEL_CACHE_PATH_ENV_VAR);
        env::remove_var("TRUSS_TRANSFER_KEEP_LOCAL_COPIES");
    }

    #[test]
    fn test_resolve_download_path() {
        let config = TrussConfig::default();

        // Test absolute path pointing to model cache
        let download_path = config.resolve_download_path("/app/model_cache/volume1/model.bin", "volume1");
        assert!(download_path.starts_with(&config.download_dir));

        // Test relative path
        let download_path = config.resolve_download_path("model.bin", "volume1");
        assert_eq!(download_path, config.download_dir.join("volume1").join("model.bin"));

        // Test absolute path not in model cache
        let download_path = config.resolve_download_path("/custom/path/model.bin", "volume1");
        assert_eq!(download_path, PathBuf::from("/custom/path/model.bin"));
    }
}

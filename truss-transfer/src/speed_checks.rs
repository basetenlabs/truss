use std::env;
use std::path::Path;

use anyhow::{Context, Result};
use log::{info, warn};
use num_cpus;
use rand;
use tokio::fs;
use tokio::io::AsyncReadExt;

use crate::constants::*;
use crate::types::BasetenPointerManifest;

/// Get the desired download speed from the environment variable or use a heuristic.
/// Heuristically, if the number of CPU cores is 64 or fewer, use a lower speed.
/// If the environment variable is not set, use a random number between 25 and 300 MB/s.
fn get_desired_speed() -> f64 {
    if let Ok(speed) = env::var(TRUSS_TRANSFER_B10FS_DOWNLOAD_SPEED_ENV_VAR) {
        if let Ok(speed) = speed.parse::<f64>() {
            return speed;
        }
    }

    // if we have 16 or fewer cpu cores, use a lower speed
    let speed_threshold = if num_cpus::get() <= 16 {
        TRUSS_TRANSFER_B10FS_DOWNLOAD_SPEED_MBPS_FEW_CORES
    } else {
        TRUSS_TRANSFER_B10FS_DOWNLOAD_SPEED_MBPS
    };

    // fallback to a random number between 10 MB/s and speed_threshold
    10.0 + rand::random::<f64>() * (speed_threshold - 10.0)
}

/// Heuristic: Check if b10cache is faster than downloading by reading the first 128MB of a file in the cache.
/// If the read speed is greater than e.g. 114MB/s, it returns true.
/// If no file in the cache is larger than 128MB, it returns true.
/// Otherwise, it returns false.
pub async fn is_b10cache_fast_heuristic(manifest: &BasetenPointerManifest) -> Result<bool> {
    let desired_speed: f64 = get_desired_speed();

    for bptr in &manifest.pointers {
        let cache_path = Path::new(CACHE_DIR).join(&bptr.hash);

        if bptr.size > (2 * B10FS_BENCHMARK_SIZE as u64) && cache_path.exists() {
            let metadata = fs::metadata(&cache_path).await?;
            let file_size = metadata.len();
            if file_size == bptr.size as u64 {
                let mut file = fs::File::open(&cache_path)
                    .await
                    .with_context(|| format!("Failed to open file {:?}", cache_path))?;
                // benchmark, read 100MB
                let mut buffer = vec![0u8; B10FS_BENCHMARK_SIZE]; // 100MB buffer
                let start_time = std::time::Instant::now();
                let bytes_read = file.read_exact(&mut buffer).await;
                let elapsed_time = start_time.elapsed();
                if bytes_read.is_ok() {
                    let elapsed_secs = elapsed_time.as_secs_f64();
                    let speed = (buffer.len() as f64 / 1024.0 / 1024.0) / elapsed_secs; // MB/s
                    warn!(
                        "b10cache: Read speed of {:.2} MB/s, desired: {:.2} MB/s",
                        speed, desired_speed
                    );
                    if speed > desired_speed {
                        return Ok(true); // Use b10cache
                    } else {
                        return Ok(false); // Don't use b10cache
                    }
                } else {
                    // If reading fails, log the error and continue
                    warn!(
                        "Failed to read file {:?}: {}",
                        cache_path,
                        bytes_read.unwrap_err()
                    );
                }
            }
        }
    }
    info!("Skipping b10cache speed check.");
    // no file > 512MB found in cache
    return Ok(true);
}

// Cloud storage range download utilities
use anyhow::{anyhow, Context, Result};
use futures_util::stream::FuturesUnordered;
use futures_util::StreamExt;
use log::{debug};
use object_store::{path::Path as ObjectPath, ObjectStore};
use rand::{rng, Rng};
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;
use tokio::fs::OpenOptions;
use tokio::io::{AsyncSeekExt, AsyncWriteExt, SeekFrom};
use tokio::sync::Semaphore;
use tokio::time::{sleep, Duration};

const BASE_WAIT_TIME: usize = 300;
const MAX_WAIT_TIME: usize = 10_000;
const MAX_RETRIES: usize = 5;

fn jitter() -> usize {
    rng().random_range(0..=500)
}

pub fn exponential_backoff(base_wait_time: usize, n: usize, max: usize) -> usize {
    (base_wait_time + n.pow(2) + jitter()).min(max)
}

/// Chunk size for range downloads (8 MB)
const RANGE_DOWNLOAD_CHUNK_MB: u64 = 8 * 1024 * 1024;

/// Check if range downloads should be used based on file size and configuration
pub fn should_use_cloud_range_download(file_size: u64) -> bool {
    use crate::constants::TRUSS_TRANSFER_USE_RANGE_DOWNLOAD;

    // Only use range downloads for files larger than threshold and when enabled
    *TRUSS_TRANSFER_USE_RANGE_DOWNLOAD && file_size > RANGE_DOWNLOAD_CHUNK_MB
}

/// High-concurrency range download using positioned writes for large files
/// Each chunk writes directly to its file position as it completes - no memory buffering
pub async fn download_cloud_range_streaming(
    storage: Arc<dyn ObjectStore>,
    object_path: &ObjectPath,
    local_path: &Path,
    file_size: u64,
    max_concurrency: usize,
) -> Result<()> {
    let chunk_size: u64 = RANGE_DOWNLOAD_CHUNK_MB;
    let total_chunks = file_size.div_ceil(chunk_size);

    debug!(
        "Starting range download: {} chunks of {}MB each, {} max concurrent workers",
        total_chunks,
        chunk_size / (1024 * 1024),
        max_concurrency
    );

    let semaphore = Arc::new(Semaphore::new(max_concurrency));
    let mut tasks = FuturesUnordered::new();

    // Spawn tasks for each chunk
    for chunk_index in 0..total_chunks {
        let start = chunk_index * chunk_size;
        let end = std::cmp::min(start + chunk_size, file_size);
        let range = start..end;

        let storage_clone = storage.clone();
        let object_path_clone = object_path.clone();
        let local_path_clone = local_path.to_path_buf();
        let semaphore_clone = semaphore.clone();

        let task = tokio::spawn(async move {
            let _permit = semaphore_clone
                .acquire()
                .await
                .map_err(|e| anyhow!("Failed to acquire semaphore: {}", e))?;

            debug!(
                "Downloading chunk {} ({}..{})",
                chunk_index, range.start, range.end
            );

            // Retry loop following hf_transfer pattern
            let mut attempts = 0;
            loop {
                match download_chunk_with_positioned_write(
                    &storage_clone,
                    &object_path_clone,
                    &local_path_clone,
                    range.clone(),
                    chunk_index,
                )
                .await
                {
                    Ok(bytes_written) => {
                        debug!("Completed chunk {} ({} bytes)", chunk_index, bytes_written);
                        break;
                    }
                    Err(e) => {
                        if attempts >= MAX_RETRIES {
                            return Err(anyhow!(
                                "Failed chunk {} after {} retries: {}",
                                chunk_index,
                                MAX_RETRIES,
                                e
                            ));
                        }

                        let wait_time =
                            exponential_backoff(BASE_WAIT_TIME, attempts, MAX_WAIT_TIME);
                        debug!(
                            "Chunk {} failed (attempt {}), retrying in {}ms: {}",
                            chunk_index,
                            attempts + 1,
                            wait_time,
                            e
                        );
                        sleep(Duration::from_millis(wait_time as u64)).await;
                        attempts += 1;
                    }
                }
            }

            Ok::<(), anyhow::Error>(())
        });

        tasks.push(task);
    }

    // Wait for all chunks to complete
    let mut completed = 0;
    while let Some(result) = tasks.next().await {
        result.context("Chunk download task panicked")??;
        completed += 1;

        if completed % 50 == 0 {
            debug!("Completed {}/{} chunks for {}", completed, total_chunks, local_path.display());
        }
    }

    debug!("All {} chunks downloaded successfully", total_chunks);

    // Verify final file size
    let final_size = tokio::fs::metadata(local_path).await?.len();
    if final_size != file_size {
        return Err(anyhow!(
            "File size mismatch after range download: expected {}, got {}",
            file_size,
            final_size
        ));
    }

    Ok(())
}

/// Download a single chunk and write it to the positioned file location
/// Closely follows the hf_transfer::download_chunk pattern
async fn download_chunk_with_positioned_write(
    storage: &dyn ObjectStore,
    object_path: &ObjectPath,
    local_path: &Path,
    range: Range<u64>,
    _chunk_index: u64,
) -> Result<usize> {
    // Get the chunk data from cloud storage
    let content = storage
        .get_range(object_path, range.clone())
        .await
        .context(format!(
            "Failed to download range {}..{}",
            range.start, range.end
        ))?;

    // Write chunk directly to its position in the file (exactly like hf_transfer)
    let mut file = OpenOptions::new()
        .write(true)
        .truncate(false) // Don't truncate - preserve existing content
        .create(true) // Create if doesn't exist
        .open(local_path)
        .await
        .context("Failed to open file for writing chunk")?;

    file.seek(SeekFrom::Start(range.start)).await?;

    file.write_all(&content).await?;

    Ok(content.len())
}

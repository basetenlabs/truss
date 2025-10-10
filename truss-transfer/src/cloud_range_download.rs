// Cloud storage range download utilities
use anyhow::{anyhow, Context, Result};
use futures_util::stream::FuturesUnordered;
use futures_util::StreamExt;
use log::{debug, info};
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
const PARALLEL_FAILURES: usize = 6;

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

    // Only use range downloads for files larger than 0 bytes
    *TRUSS_TRANSFER_USE_RANGE_DOWNLOAD
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
    if file_size == 0 {
        // Create an empty file, as some object stores do not support 0-byte range requests
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(local_path)
            .await
            .context("Failed to create empty file")?;
        return Ok(());
    }

    let semaphore = Arc::new(Semaphore::new(max_concurrency));
    // todo: parallel failures could be improved
    let semaphore_failures = Arc::new(Semaphore::new(PARALLEL_FAILURES));
    let mut tasks = FuturesUnordered::new();

    // Determine which chunks should be flushed (last 16 or max_concurrency chunks)
    let flush_threshold = total_chunks.saturating_sub(max_concurrency.min(16) as u64);

    // Spawn tasks for each chunk
    for chunk_index in 0..total_chunks {
        let start = chunk_index * chunk_size;
        let end = std::cmp::min(start + chunk_size, file_size);
        let range = start..end;
        // only flush last chunk.
        let should_flush = chunk_index >= flush_threshold;

        let storage_clone = storage.clone();
        let object_path_clone = object_path.clone();
        let local_path_clone = local_path.to_path_buf();
        let semaphore_clone = semaphore.clone();
        let semaphore_failures_clone = semaphore_failures.clone();

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
                    should_flush,
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
                        // Limit parallel retries to avoid overwhelming the system
                        let permit = semaphore_failures_clone
                            .acquire()
                            .await
                            .map_err(|e| anyhow!("Failed to acquire failure semaphore: {}", e))?;

                        let wait_time =
                            exponential_backoff(BASE_WAIT_TIME, attempts, MAX_WAIT_TIME);
                        info!(
                            "Chunk {} failed (attempt {}), retrying in {}ms: {}",
                            chunk_index,
                            attempts + 1,
                            wait_time,
                            e
                        );
                        sleep(Duration::from_millis(wait_time as u64)).await;
                        attempts += 1;
                        drop(permit); // Release failure semaphore
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
        result.context("Chunk download task paniced")??;
        completed += 1;

        if completed % 50 == 0 {
            debug!(
                "Completed {}/{} chunks for {}",
                completed,
                total_chunks,
                local_path.display()
            );
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
    should_flush: bool,
) -> Result<usize> {
    // TODO: add retries here to the requests, with parallel backoff failure strategy
    let content = storage
        .get_range(object_path, range.clone())
        .await
        .context(format!(
            "Failed to download range {}..{} for object {:?}",
            range.start, range.end, object_path
        ))?;

    let mut file = OpenOptions::new()
        .write(true)
        .truncate(false)
        .create(true)
        .open(local_path)
        .await
        .context(format!(
            "Failed to open file {:?} for writing chunk at position {}",
            local_path, range.start
        ))?;

    file.seek(SeekFrom::Start(range.start))
        .await
        .context(format!(
            "Failed to seek to position {} in {:?}",
            range.start, local_path
        ))?;

    file.write_all(&content).await.context(format!(
        "Failed to write {} bytes at position {} to {:?}",
        content.len(),
        range.start,
        local_path
    ))?;

    // Flush only for the last max_concurrency chunks to ensure data is written to disk
    if should_flush {
        file.flush().await.context(format!(
            "Failed to flush chunk data at position {} to {:?}",
            range.start, local_path
        ))?;
    }

    Ok(content.len())
}

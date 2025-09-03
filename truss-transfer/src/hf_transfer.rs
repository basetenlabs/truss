// copied from https://github.com/huggingface/hf_transfer/blob/main/src/lib.rs Apache License
// Do not modify.

use futures_util::stream::FuturesUnordered;
use futures_util::StreamExt;
use rand::{rng, Rng};
use reqwest::header::{HeaderMap, HeaderValue, ToStrError, AUTHORIZATION, CONTENT_RANGE, RANGE};
use reqwest::Url;
use std::fmt::Display;
use std::io::SeekFrom;
use std::sync::Arc;
use std::time::Duration;
use tokio::fs::OpenOptions;
use tokio::io::AsyncSeekExt;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;
use tokio::time::sleep;

use anyhow::{anyhow, Result};

const BASE_WAIT_TIME: usize = 300;
const MAX_WAIT_TIME: usize = 10_000;
const CHUNK_SIZE: usize = 10 * 1024 * 1024;

fn jitter() -> usize {
    rng().random_range(0..=500)
}

pub fn exponential_backoff(base_wait_time: usize, n: usize, max: usize) -> usize {
    (base_wait_time + n.pow(2) + jitter()).min(max)
}

#[allow(clippy::too_many_arguments)]
pub async fn download_async(
    url: String,
    filename: String,
    max_files: usize,
    parallel_failures: usize,
    max_retries: usize,
    auth_token: Option<String>,
    callback: Option<Box<dyn Fn(usize) + Send + Sync>>,
    check_file_size: u64,
    semaphore_global: Arc<Semaphore>,
) -> Result<()> {
    let client = reqwest::Client::builder()
        // https://github.com/hyperium/hyper/issues/2136#issuecomment-589488526
        .http2_keep_alive_timeout(Duration::from_secs(15))
        .no_proxy()
        .http2_initial_stream_window_size(CHUNK_SIZE as u32)
        .http2_initial_connection_window_size(2 * CHUNK_SIZE as u32)
        .build()
        .unwrap();

    let chunk_size = CHUNK_SIZE;
    let mut headers = HeaderMap::new();

    if let Some(token) = auth_token.as_ref() {
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}"))?,
        );
    }

    let response = client
        .get(&url)
        .headers(headers.clone())
        .header(RANGE, "bytes=0-0")
        .send()
        .await
        .map_err(|err| anyhow!("Error while downloading: {err}"))?
        .error_for_status()
        .map_err(|err| anyhow!(err.to_string()))?;

    // Only call the final redirect URL to avoid overloading the Hub with requests and also
    // altering the download count
    let redirected_url = response.url();
    if Url::parse(&url)
        .map_err(|err| anyhow!("failed to parse url: {err}"))?
        .host()
        != redirected_url.host()
    {
        headers.remove(AUTHORIZATION);
    }

    let content_range = response
        .headers()
        .get(CONTENT_RANGE)
        .ok_or(anyhow!("No content length"))?
        .to_str()
        .map_err(|err| anyhow!("Error while downloading: {err}"))?;

    let size: Vec<&str> = content_range.split('/').collect();
    if check_file_size != size[1].parse::<u64>()? {
        return Err(anyhow!(
            "File size mismatch according to blib range: expected {}, got {}",
            check_file_size,
            size[1]
        ));
    }

    // Content-Range: bytes 0-0/702517648
    // https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Range
    let length: usize = size
        .last()
        .ok_or(anyhow!("Error while downloading: No size was detected"))?
        .parse()
        .map_err(|err| anyhow!("Error while downloading: {err}"))?;

    let mut handles = FuturesUnordered::new();
    let semaphore = Arc::new(Semaphore::new(max_files));
    let parallel_failures_semaphore = Arc::new(Semaphore::new(parallel_failures));

    for start in (0..length).step_by(chunk_size) {
        let url = redirected_url.to_string();
        let filename = filename.clone();
        let client = client.clone();
        let headers = headers.clone();

        let stop = std::cmp::min(start + chunk_size - 1, length - 1);
        let semaphore = semaphore.clone();
        let parallel_failures_semaphore = parallel_failures_semaphore.clone();
        let semaphore_global = semaphore_global.clone();
        handles.push(tokio::spawn(async move {
            let permit = semaphore
                .acquire_owned()
                .await
                .map_err(|err| anyhow!("Error while downloading: {err}"))?;
            let permit_global = semaphore_global.acquire_owned().await.map_err(|err| {
                anyhow!("Failed to acquire global semaphore: {err}")
            })?;
            let mut chunk = download_chunk(&client, &url, &filename, start, stop, headers.clone()).await;
            let mut i = 0;
            if parallel_failures > 0 {
                while let Err(dlerr) = chunk {
                    if i >= max_retries {
                        return Err(anyhow!(
                            "Failed after too many retries ({max_retries}): {dlerr}"
                        ));
                    }
                    let parallel_failure_permit = parallel_failures_semaphore.clone().try_acquire_owned().map_err(|err| {
                        anyhow!(
                            "Failed too many failures in parallel ({parallel_failures}): {dlerr} ({err})"
                        )
                    })?;

                    let wait_time = exponential_backoff(BASE_WAIT_TIME, i, MAX_WAIT_TIME);
                    sleep(Duration::from_millis(wait_time as u64)).await;

                    chunk = download_chunk(&client, &url, &filename, start, stop, headers.clone()).await;
                    i += 1;
                    drop(parallel_failure_permit);
                }
            }
            drop(permit);
            drop(permit_global);
            chunk.map_err(|e| anyhow!("Downloading error {e}")).and(Ok(stop - start))
        }));
    }

    // Output the chained result
    while let Some(result) = handles.next().await {
        match result {
            Ok(Ok(size)) => {
                if let Some(ref callback) = callback {
                    callback(size);
                }
            }
            Ok(Err(err)) => {
                return Err(err);
            }
            Err(err) => {
                return Err(anyhow!("Error while downloading: {err}"));
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
enum Error {
    Io(std::io::Error),
    Request(reqwest::Error),
    ToStrError(ToStrError),
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<reqwest::Error> for Error {
    fn from(value: reqwest::Error) -> Self {
        Self::Request(value)
    }
}

impl From<ToStrError> for Error {
    fn from(value: ToStrError) -> Self {
        Self::ToStrError(value)
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(io) => write!(f, "Io: {io}"),
            Self::Request(req) => write!(f, "Request: {req}"),
            Self::ToStrError(req) => write!(f, "Response non ascii: {req}"),
        }
    }
}

impl std::error::Error for Error {}

async fn download_chunk(
    client: &reqwest::Client,
    url: &str,
    filename: &str,
    start: usize,
    stop: usize,
    headers: HeaderMap,
) -> Result<(), Error> {
    // Process each socket concurrently.
    let range = format!("bytes={start}-{stop}");
    let mut file = OpenOptions::new()
        .write(true)
        .truncate(false)
        .create(true)
        .open(filename)
        .await?;
    file.seek(SeekFrom::Start(start as u64)).await?;
    let response = client
        .get(url)
        .headers(headers)
        .header(RANGE, range)
        .send()
        .await?
        .error_for_status()?;
    let content = response.bytes().await?;
    file.write_all(&content).await?;
    Ok(())
}

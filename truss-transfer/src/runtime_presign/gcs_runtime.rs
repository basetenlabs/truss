use crate::types::GcsError;
use crate::types::BasetenPointer;
use crate::create::gcs_metadata::gcs_storage;

/// Resolve GCS pointer by generating a pre-signed URL
/// This swaps the pointer kind from GCS to HTTP and updates the URL
pub async fn modify_gcs_pointer(
    pointers: BasetenPointer,
) -> Result<(BasetenPointer), GcsError> {
    // mutate
    let gcs = gcs_storage(
        &pointers.resolution.bucket_name,
        &pointers.runtime_secret_name,
    )?;

    let url = gcs
        .get(&pointers.resolution.path) // no real presign
        .await
        .map_err(|e| GcsError::Io(e.to_string()))?
        .url()
        .to_string();

    Ok(BasetenPointer {
        resolution: Resolution::Http(HttpResolution::new(url, 9999999999)),
        uid: pointers.uid,
        file_name: pointers.file_name,
        hashtype: pointers.hashtype,
        hash: pointers.hash,
        size: pointers.size,
        runtime_secret_name: pointers.runtime_secret_name,
    })
}

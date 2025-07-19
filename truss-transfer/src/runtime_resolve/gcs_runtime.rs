use crate::types::{GcsError, BasetenPointer, Resolution, HttpResolution};
use crate::create::gcs_metadata::gcs_storage;
use chrono::{Utc, Duration};

/// Resolve GCS pointer by generating a pre-signed URL
/// This swaps the pointer kind from GCS to HTTP and updates the URL
pub async fn download_gcs_pointer(
    pointer: BasetenPointer,
) -> Result<BasetenPointer, GcsError> {

}

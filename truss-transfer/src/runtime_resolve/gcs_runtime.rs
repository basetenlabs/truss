use crate::types::{GcsError, BasetenPointer, Resolution, HttpResolution};
use crate::create::gcs_metadata::gcs_storage;
use chrono::{Utc, Duration};

// delete
pub async fn download_gcs_pointer(
    pointer: BasetenPointer,
) -> Result<BasetenPointer, GcsError> {

}

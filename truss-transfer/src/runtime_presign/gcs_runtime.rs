use crate::types::GcsError;
use crate::types::BasetenPointer;


/// Resolve GCS pointer by generating a pre-signed URL
/// This swaps the pointer kind from GCS to HTTP and updates the URL
pub async fn modify_gcs_pointer(
    pointer: &mut BasetenPointer,
    expiration_minutes: u64,
) -> Result<(BasetenPointer), GcsError> {
    // mutate
    Ok(pointer.clone())
}

use crate::constants::RETRY_TIMEOUT_BUDGET_PERCENTAGE;
use crate::constants::{CANCELLATION_ERROR_MESSAGE_DETAIL, CTRL_C_ERROR_MESSAGE_DETAIL};
use crate::errors::ClientError;
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

// Global state for staging addresses
pub static STAGING_ADDRESS: Lazy<Vec<String>> = Lazy::new(|| {
    option_env!("PERF_CLIENT_STAGING_ADDRESS")
        .unwrap_or(std::str::from_utf8(crate::constants::DEFAULT_STAGING_ADDRESS).unwrap())
        .split(',')
        .map(String::from)
        .collect()
});

/// Calculate retry timeout budget based on total requests
pub fn calculate_retry_timeout_budget(total_requests: usize) -> usize {
    (total_requests as f64 * RETRY_TIMEOUT_BUDGET_PERCENTAGE).ceil() as usize
}

/// Helper function to acquire permit and check for cancellation
/// Note: CTRL_C handling should be implemented by the binding layer
pub async fn acquire_permit_or_cancel(
    semaphore: Arc<Semaphore>,
    local_cancel_token: Arc<AtomicBool>,
    global_cancel_token: Option<Arc<AtomicBool>>,
) -> Result<OwnedSemaphorePermit, ClientError> {
    tokio::select! {
        biased;

        // Check for global cancellation (e.g., Ctrl+C) if provided
        _ = tokio::time::sleep(Duration::from_millis(1)), if global_cancel_token.as_ref().map_or(false, |token| token.load(Ordering::SeqCst)) => {
            local_cancel_token.store(true, Ordering::SeqCst);
            return Err(ClientError::Cancellation(CTRL_C_ERROR_MESSAGE_DETAIL.to_string()));
        }

        // Check for local cancellation token
        _ = tokio::time::sleep(Duration::from_millis(1)), if local_cancel_token.load(Ordering::SeqCst) => {
            return Err(ClientError::Cancellation(CANCELLATION_ERROR_MESSAGE_DETAIL.to_string()));
        }

        // Try to acquire the permit
        permit_result = semaphore.acquire_owned() => {
            let permit = permit_result.map_err(|e| {
                ClientError::Network(format!("Semaphore acquire_owned failed: {}", e))
            })?;

            // Re-check cancellation signals after acquiring the permit
            if global_cancel_token.as_ref().map_or(false, |token| token.load(Ordering::SeqCst)) {
                local_cancel_token.store(true, Ordering::SeqCst);
                return Err(ClientError::Cancellation(CTRL_C_ERROR_MESSAGE_DETAIL.to_string()));
            }
            if local_cancel_token.load(Ordering::SeqCst) {
                return Err(ClientError::Cancellation(CANCELLATION_ERROR_MESSAGE_DETAIL.to_string()));
            }
            Ok(permit)
        }
    }
}

/// Process task outcome and manage errors
pub fn process_task_outcome<D>(
    task_join_result: Result<Result<D, ClientError>, tokio::task::JoinError>,
    first_error: &mut Option<ClientError>,
    cancel_token: &Arc<AtomicBool>,
) -> Option<D> {
    match task_join_result {
        Ok(Ok(data)) => {
            if first_error.is_none() {
                Some(data)
            } else {
                None
            }
        }
        Ok(Err(current_err)) => {
            let is_current_err_cancellation = current_err
                .to_string()
                .ends_with(CANCELLATION_ERROR_MESSAGE_DETAIL);

            if let Some(ref existing_err) = first_error {
                let is_existing_err_cancellation = existing_err
                    .to_string()
                    .ends_with(CANCELLATION_ERROR_MESSAGE_DETAIL);
                if is_existing_err_cancellation && !is_current_err_cancellation {
                    *first_error = Some(current_err);
                }
            } else {
                *first_error = Some(current_err);
            }
            None
        }
        Err(join_err) => {
            let panic_err = ClientError::Network(format!("Tokio task panicked: {}", join_err));
            if let Some(ref existing_err) = first_error {
                let is_existing_err_cancellation = existing_err
                    .to_string()
                    .ends_with(CANCELLATION_ERROR_MESSAGE_DETAIL);
                if is_existing_err_cancellation {
                    *first_error = Some(panic_err);
                }
            } else {
                *first_error = Some(panic_err);
            }
            cancel_token.store(true, Ordering::SeqCst);
            None
        }
    }
}

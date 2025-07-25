use crate::constants::{CANCELLATION_ERROR_MESSAGE_DETAIL, CTRL_C_ERROR_MESSAGE_DETAIL};
use crate::constants::{HEDGE_BUDGET_PERCENTAGE, RETRY_TIMEOUT_BUDGET_PERCENTAGE};
use crate::errors::ClientError;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Calculate retry timeout budget based on total requests
pub fn calculate_retry_timeout_budget(total_requests: usize) -> usize {
    (total_requests as f64 * RETRY_TIMEOUT_BUDGET_PERCENTAGE).ceil() as usize
}

pub fn calculate_hedge_budget(total_requests: usize) -> usize {
    (total_requests as f64 * HEDGE_BUDGET_PERCENTAGE).ceil() as usize
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
        _ = tokio::time::sleep(Duration::from_millis(5)), if global_cancel_token.as_ref().is_some_and(|token| token.load(Ordering::SeqCst)) => {
            local_cancel_token.store(true, Ordering::SeqCst);
            Err(ClientError::Cancellation(CTRL_C_ERROR_MESSAGE_DETAIL.to_string()))?
        }

        // Check for local cancellation token
        _ = tokio::time::sleep(Duration::from_millis(5)), if local_cancel_token.load(Ordering::SeqCst) => {
            Err(ClientError::Cancellation(CANCELLATION_ERROR_MESSAGE_DETAIL.to_string()))?
        }

        // Try to acquire the permit
        permit_result = semaphore.acquire_owned() => {
            let permit = permit_result.map_err(|e| {
                ClientError::Network(format!("Semaphore acquire_owned failed: {}", e))
            })?;

            // Re-check cancellation signals after acquiring the permit
            if global_cancel_token.as_ref().is_some_and(|token| token.load(Ordering::SeqCst)) {
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

/// Process JoinSet task outcome with improved error handling
pub fn process_joinset_outcome<T>(
    task_result: Result<Result<T, ClientError>, tokio::task::JoinError>,
    cancel_token: &Arc<AtomicBool>,
) -> Result<T, ClientError> {
    match task_result {
        Ok(Ok(data)) => Ok(data),
        Ok(Err(client_error)) => {
            cancel_token.store(true, Ordering::SeqCst);
            Err(client_error)
        }
        Err(join_error) => {
            cancel_token.store(true, Ordering::SeqCst);
            if join_error.is_cancelled() {
                Err(ClientError::Cancellation("Task was cancelled".to_string()))
            } else if join_error.is_panic() {
                Err(ClientError::Network(format!(
                    "Task panicked: {}",
                    join_error
                )))
            } else {
                Err(ClientError::Network(format!(
                    "Task join error: {}",
                    join_error
                )))
            }
        }
    }
}

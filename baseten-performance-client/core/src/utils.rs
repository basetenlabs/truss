use crate::constants::{HEDGE_BUDGET_PERCENTAGE, RETRY_TIMEOUT_BUDGET_PERCENTAGE};
use crate::errors::ClientError;

/// Calculate retry timeout budget based on total requests
pub fn calculate_retry_timeout_budget(total_requests: usize) -> usize {
    // if the budget goes from 1->0 the budget is exhaused. So always set it to intially 2.
    1 + ((total_requests as f64 * RETRY_TIMEOUT_BUDGET_PERCENTAGE).ceil() as usize)
}

pub fn calculate_hedge_budget(total_requests: usize) -> usize {
    1 + ((total_requests as f64 * HEDGE_BUDGET_PERCENTAGE).ceil() as usize)
}

/// Process JoinSet task outcome with improved error handling
pub fn process_joinset_outcome<T>(
    task_result: Result<Result<T, ClientError>, tokio::task::JoinError>,
) -> Result<T, ClientError> {
    match task_result {
        Ok(Ok(data)) => Ok(data),
        Ok(Err(client_error)) => Err(client_error),
        Err(join_error) => {
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

use crate::constants::{HEDGE_BUDGET_PERCENTAGE, RETRY_BUDGET_PERCENTAGE};
use crate::errors::ClientError;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Atomically take one unit from a shared budget, returning whether a unit was available.
///
/// Budgets are shared by every concurrent request in an operation. A plain `fetch_sub`
/// wraps an exhausted budget around to `usize::MAX`, which silently removes the cap for
/// all later callers; this never decrements below zero.
pub(crate) fn try_consume_budget(budget: &AtomicUsize) -> bool {
    budget
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
            remaining.checked_sub(1)
        })
        .is_ok()
}

/// Calculate retry timeout budget based on total requests
pub fn calculate_retry_timeout_budget(total_requests: usize) -> usize {
    // if the budget goes from 1->0 the budget is exhaused. So always set it to intially 2.
    1 + ((total_requests as f64 * RETRY_BUDGET_PERCENTAGE).ceil() as usize)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_try_consume_budget_stops_at_zero() {
        let budget = AtomicUsize::new(2);

        assert!(try_consume_budget(&budget));
        assert!(try_consume_budget(&budget));
        assert!(!try_consume_budget(&budget));
        assert!(!try_consume_budget(&budget));
        assert_eq!(budget.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_try_consume_budget_under_contention_never_overspends() {
        let initial_budget = 11;
        let threads = 32;
        let attempts_per_thread = 100;
        let budget = Arc::new(AtomicUsize::new(initial_budget));

        let handles = (0..threads)
            .map(|_| {
                let budget = Arc::clone(&budget);
                std::thread::spawn(move || {
                    (0..attempts_per_thread)
                        .filter(|_| try_consume_budget(&budget))
                        .count()
                })
            })
            .collect::<Vec<_>>();

        let granted: usize = handles
            .into_iter()
            .map(|handle| handle.join().expect("worker thread should not panic"))
            .sum();

        assert_eq!(granted, initial_budget);
        assert_eq!(budget.load(Ordering::SeqCst), 0);
    }
}

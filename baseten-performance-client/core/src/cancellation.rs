use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::task::JoinSet;

/// A cancellation token that can be used to cancel async operations.
///
/// Clone this token and pass it to functions that should be cancellable.
/// When `cancel()` is called, all operations checking this token will be cancelled.
#[derive(Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Cancel all operations using this token
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Check if cancellation has been requested
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

/// RAII guard that wraps a JoinSet and aborts all tasks when dropped.
///
/// This ensures that when a future is cancelled (e.g., via tokio::select! or Drop),
/// all spawned tasks are automatically aborted, preventing resource leaks.
pub struct JoinSetGuard<T: 'static> {
    join_set: JoinSet<T>,
    cancel_token: Option<CancellationToken>,
}

impl<T: 'static> JoinSetGuard<T> {
    pub fn new() -> Self {
        Self {
            join_set: JoinSet::new(),
            cancel_token: None,
        }
    }

    pub fn with_cancel_token(cancel_token: CancellationToken) -> Self {
        Self {
            join_set: JoinSet::new(),
            cancel_token: Some(cancel_token),
        }
    }

    pub fn spawn<F>(&mut self, task: F)
    where
        F: std::future::Future<Output = T> + Send + 'static,
        T: Send,
    {
        self.join_set.spawn(task);
    }

    pub async fn join_next(&mut self) -> Option<Result<T, tokio::task::JoinError>> {
        self.join_set.join_next().await
    }

    pub fn abort_all(&mut self) {
        if let Some(ref token) = self.cancel_token {
            token.cancel();
        }
        self.join_set.abort_all();
    }

    pub fn len(&self) -> usize {
        self.join_set.len()
    }

    pub fn is_empty(&self) -> bool {
        self.join_set.is_empty()
    }
}

impl<T: 'static> Default for JoinSetGuard<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: 'static> Drop for JoinSetGuard<T> {
    fn drop(&mut self) {
        if let Some(ref token) = self.cancel_token {
            token.cancel();
        }
        self.join_set.abort_all();
    }
}

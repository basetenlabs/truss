use crate::errors::ClientError;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken as TokioCancellationToken;

/// A cancellation token that can be used to cancel async operations.
///
/// Clones share the cancellation signal. Only the original handle honors
/// `cancel_on_drop`, so dropping a worker's clone does not cancel its siblings.
#[derive(Debug)]
pub struct CancellationToken {
    inner: Option<TokioCancellationToken>,
    cancel_on_drop: bool,
}

impl CancellationToken {
    pub fn new(cancel_on_drop: bool) -> Self {
        Self {
            inner: Some(TokioCancellationToken::new()),
            cancel_on_drop,
        }
    }

    pub(crate) fn inactive() -> Self {
        Self {
            inner: None,
            cancel_on_drop: false,
        }
    }

    /// Cancel all operations using this token
    pub fn cancel(&self) {
        if let Some(token) = &self.inner {
            token.cancel();
        }
    }

    /// Check if cancellation has been requested
    pub fn is_cancelled(&self) -> bool {
        self.inner
            .as_ref()
            .is_some_and(TokioCancellationToken::is_cancelled)
    }

    pub(crate) async fn cancelled(&self) {
        if let Some(token) = &self.inner {
            token.cancelled().await;
        } else {
            std::future::pending().await
        }
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new(false)
    }
}

impl Clone for CancellationToken {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            cancel_on_drop: false,
        }
    }
}

impl Drop for CancellationToken {
    fn drop(&mut self) {
        if self.cancel_on_drop {
            // Automatically cancel when dropped for RAII behavior
            // This ensures that when the axum handle is revoked, the proxy stops proxying
            self.cancel();
        }
    }
}

/// RAII guard that wraps a JoinSet and aborts all tasks when dropped.
///
/// This ensures that when a future is cancelled (e.g., via tokio::select! or Drop),
/// all spawned tasks are automatically aborted, preventing resource leaks.
pub(crate) struct JoinSetGuard<T: 'static> {
    join_set: JoinSet<T>,
    cancel_token: Option<TokioCancellationToken>,
}

impl<T: 'static> JoinSetGuard<T> {
    pub(crate) fn new() -> Self {
        Self {
            join_set: JoinSet::new(),
            cancel_token: None,
        }
    }

    pub(crate) fn with_cancel_token(token: CancellationToken) -> Self {
        Self {
            join_set: JoinSet::new(),
            cancel_token: token.inner.clone(),
        }
    }

    pub async fn join_next(&mut self) -> Option<Result<T, tokio::task::JoinError>> {
        self.join_set.join_next().await
    }

    pub(crate) fn abort_all(&mut self) {
        self.join_set.abort_all();
    }
}

impl<T: Send + 'static> JoinSetGuard<Result<T, ClientError>> {
    pub(crate) fn spawn<F>(&mut self, task: F)
    where
        F: std::future::Future<Output = Result<T, ClientError>> + Send + 'static,
    {
        if let Some(token) = self.cancel_token.clone() {
            self.join_set.spawn(async move {
                tokio::select! {
                    biased;
                    _ = token.cancelled() => {
                        Err(ClientError::Cancellation("Operation cancelled by token".to_string()))
                    }
                    result = task => result,
                }
            });
        } else {
            self.join_set.spawn(task);
        }
    }
}

impl<T: 'static> Default for JoinSetGuard<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: 'static> Drop for JoinSetGuard<T> {
    fn drop(&mut self) {
        self.abort_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_owning_token_cancels_on_drop() {
        let owner = CancellationToken::new(true);
        let observer = owner.clone();
        drop(owner.clone());
        assert!(!observer.is_cancelled());
        drop(owner);
        assert!(observer.is_cancelled());
    }
}

//! Python bindings for streaming functionality
//!
//! This module provides the Python-facing streaming API with channel-based architecture,
//! async iterator support, and proper resource management.

use crate::*;
use baseten_performance_client_core::streaming::{StreamEvent};
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;


/// Python iterator for Server-Sent Events (SSE) with channel-based architecture
#[pyclass]
pub struct EventStreamIter {
    receiver: Arc<tokio::sync::Mutex<mpsc::Receiver<StreamEvent>>>,
    task_handle: JoinHandle<()>,
    runtime: Arc<Runtime>,
}

impl Drop for EventStreamIter {
    fn drop(&mut self) {
        // When the iterator is garbage collected, ensure the background task is aborted.
        self.task_handle.abort();
    }
}

#[pymethods]
impl EventStreamIter {
    /// Create a new iterator from a stream
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Abort the stream and cancel the underlying request
    fn abort(&self) {
        // Abort the background task if it is still running
        self.task_handle.abort();
    }

    /// Get the next event (synchronous iteration)
    fn __next__(&self, py: Python) -> PyResult<Option<PyObject>> {
        // Do all async work first, staying within the same GIL context
        let receiver = self.receiver.clone();
        let rt = Arc::clone(&self.runtime);

        let msg = py.allow_threads(|| {
            rt.block_on(async {
                let mut receiver_guard = receiver.lock().await;
                receiver_guard.recv().await
            })
        });

        // Now handle the result within the original GIL context
        match msg {
            Some(StreamEvent::Json(value)) => {
                let py_obj = pythonize::pythonize(py, &value)
                    .map_err(|e| PyValueError::new_err(format!("Pythonize error: {}", e)))?;
                Ok(Some(py_obj.unbind()))
            }
            Some(StreamEvent::Text(text)) => {
                #[allow(deprecated)]
                Ok(Some(text.into_py(py)))
            }
            Some(StreamEvent::End) => {
                // Signal end of iteration by returning None
                Ok(None) // returning None -> StopIteration
            }
            Some(StreamEvent::Error(err)) => {
                // Convert our ClientError into a Python exception
                Err(PerformanceClient::convert_core_error_to_py_err(err))
            }
            None => {
                // Channel closed unexpectedly.
                Ok(None)
            }
        }
    }

    /// Async iterator protocol - implement __aiter__
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Async iterator protocol - implement __anext__
    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let future = py.allow_threads(|| {
            // Use a future to handle the async nature of the stream
            let receiver = self.receiver.clone();

            async move {
                let msg = {
                    let mut receiver_guard = receiver.lock().await;
                    receiver_guard.recv().await
                };

                match msg {
                    Some(StreamEvent::Json(value)) => {
                        // Convert JSON (serde_json::Value) to a Python object (dict/list/etc)
                        Python::with_gil(|py| {
                            let py_obj = pythonize::pythonize(py, &value)
                                .map_err(|e| PyValueError::new_err(format!("Pythonize error: {}", e)))?;
                            Ok(py_obj.unbind())
                        })
                    }
                    Some(StreamEvent::Text(text)) => {
                        // Return plain text as Python str
                        Python::with_gil(|py| {
                            #[allow(deprecated)]
                            Ok(text.into_py(py))
                        })
                    }
                    Some(StreamEvent::End) => {
                        // Signal end of iteration by raising StopAsyncIteration
                        Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                            "Stream ended",
                        ))
                    }
                    Some(StreamEvent::Error(err)) => {
                        // Convert our ClientError into a Python exception
                        Err(PerformanceClient::convert_core_error_to_py_err(err))
                    }
                    None => {
                        // Channel closed unexpectedly. End iteration.
                        Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                            "Stream closed",
                        ))
                    }
                }
            }
        });

        pyo3_async_runtimes::tokio::future_into_py(py, future)
    }

    /// String representation
    fn __repr__(&self) -> String {
        "EventStreamIter(...)".to_string()
    }
}

impl EventStreamIter {
    /// Create a new EventStreamIter with channel-based architecture
    pub fn new(
        receiver: mpsc::Receiver<StreamEvent>,
        task_handle: JoinHandle<()>,
        runtime: Arc<Runtime>,
    ) -> Self {
        Self {
            receiver: Arc::new(tokio::sync::Mutex::new(receiver)),
            task_handle,
            runtime,
        }
    }
}

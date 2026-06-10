import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from streaming_utils.utils.constants import (
    ASSIGNMENT_WORKER_TYPE,
    DIARIZATION_WORKER_TYPE,
    TRANSCRIPTION_WORKER_TYPE,
)
from streaming_utils.utils.error_utils import (
    AssignmentError,
    DiarizationError,
    TranscriptionError,
    log_stream_event,
)
from streaming_utils.utils.websocket_utils import WebSocketManager

logger = logging.getLogger(__name__)


async def handle_worker_error(
    error: Exception,
    stream_id: str,
    operation: str,
    ws_manager: WebSocketManager,
    worker_type: str = "UNKNOWN",
) -> None:
    """Log and handle worker errors gracefully with detailed context."""
    error_context = {
        "stream_id": stream_id,
        "operation": operation,
        "worker_type": worker_type,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }

    # logger.error(f"Worker error in {worker_type} for stream {stream_id} during {operation}: {error_context}")

    if worker_type == TRANSCRIPTION_WORKER_TYPE:
        await ws_manager.send_error_to_websocket(
            TranscriptionError(stream_id, f"Transcription failed: {str(error)}")
        )
        # raise TranscriptionError(stream_id, f"Transcription failed: {str(error)}")
    elif worker_type == DIARIZATION_WORKER_TYPE:
        await ws_manager.send_error_to_websocket(
            DiarizationError(stream_id, f"Diarization failed: {str(error)}")
        )
        # raise DiarizationError(stream_id, f"Diarization failed: {str(error)}")
    elif worker_type == ASSIGNMENT_WORKER_TYPE:
        await ws_manager.send_error_to_websocket(
            AssignmentError(stream_id, f"Assignment failed: {str(error)}")
        )
        # raise AssignmentError(stream_id, f"Assignment failed: {str(error)}")


class BaseWorker(ABC):
    """Abstract base class for all stream workers with common functionality."""

    def __init__(self, stream_id: str, worker_type: str, ws_manager: WebSocketManager):
        self.stream_id = stream_id
        self.worker_type = worker_type
        self.is_running = False
        self.worker_task: Optional[asyncio.Task] = None
        self.start_time: Optional[float] = None
        self.total_tasks_processed = 0
        self.total_processing_time = 0.0
        self.last_task_time: Optional[float] = None
        self.ws_manager = ws_manager
        log_stream_event(
            stream_id,
            f"{worker_type} worker created",
            {"worker_type": worker_type, "stream_id": stream_id},
            "DEBUG",
        )

    async def start(self) -> None:
        """Start the worker and begin processing tasks."""
        if self.is_running:
            logger.warning(
                f"⚠️ {self.worker_type} worker for stream {self.stream_id} is already running"
            )
            return

        try:
            self.is_running = True
            self.start_time = time.time()
            self.worker_task = asyncio.create_task(self._worker_loop())

            log_stream_event(
                self.stream_id,
                f"✅ {self.worker_type} worker started",
                {"start_time": self.start_time, "worker_type": self.worker_type},
            )

        except Exception as e:
            self.is_running = False
            logger.error(
                f"❌ Failed to start {self.worker_type} worker for stream {self.stream_id}: {e}"
            )
            raise

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        if not self.is_running:
            logger.info(f"ℹ️ {self.worker_type} worker for stream {self.stream_id} is stopped")
            return

        try:
            self.is_running = False

            if self.worker_task and not self.worker_task.done():
                self.worker_task.cancel()
                try:
                    await self.worker_task
                except asyncio.CancelledError:
                    pass

            duration = time.time() - self.start_time if self.start_time else 0.0

            log_stream_event(
                self.stream_id,
                f"🛑 {self.worker_type} worker stopped",
                {
                    "total_runtime": duration,
                    "total_tasks_processed": self.total_tasks_processed,
                    "total_processing_time": self.total_processing_time,
                    "worker_type": self.worker_type,
                },
            )

        except Exception as e:
            logger.error(
                f"❌ Error stopping {self.worker_type} worker for stream {self.stream_id}: {e}"
            )

    async def _worker_loop(self) -> None:
        """Main worker loop with error handling and logging."""
        log_stream_event(self.stream_id, f"✅ {self.worker_type} worker loop started")

        try:
            while self.is_running:
                try:
                    # Get task with timeout
                    task = await self._get_task_with_timeout()
                    if task is None:
                        continue  # Timeout, continue loop

                    # Process task
                    start_time = time.time()
                    await self._process_task(task)

                    # Update statistics
                    processing_time = time.time() - start_time
                    self.total_tasks_processed += 1
                    self.total_processing_time += processing_time
                    self.last_task_time = time.time()

                    log_stream_event(
                        self.stream_id,
                        f"{self.worker_type} task processed",
                        {
                            "task_processing_time": processing_time,
                            "total_tasks_processed": self.total_tasks_processed,
                            "total_processing_time": self.total_processing_time,
                        },
                        "DEBUG",
                    )

                except asyncio.CancelledError:
                    log_stream_event(self.stream_id, f"{self.worker_type} worker cancelled")
                    raise
                except Exception as e:
                    await handle_worker_error(
                        e, self.stream_id, "task processing", self.ws_manager, self.worker_type
                    )
                    # Continue processing other tasks

        except asyncio.CancelledError:
            log_stream_event(self.stream_id, f"🧹 {self.worker_type} worker loop cancelled")
            raise
        except Exception as e:
            logger.error(
                f"❌ {self.worker_type} worker loop failed for stream {self.stream_id}: {e}"
            )
            raise
        finally:
            log_stream_event(self.stream_id, f"🧹 {self.worker_type} worker loop ended")

    @abstractmethod
    async def _get_task_with_timeout(self) -> Optional[Any]:
        """Get task from queue with timeout. Return None if timeout."""
        pass

    @abstractmethod
    async def _process_task(self, task: Any) -> None:
        """Process a single task."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics for monitoring."""
        runtime = time.time() - self.start_time if self.start_time else 0.0
        avg_processing_time = (
            self.total_processing_time / self.total_tasks_processed
            if self.total_tasks_processed > 0
            else 0.0
        )

        return {
            "stream_id": self.stream_id,
            "worker_type": self.worker_type,
            "is_running": self.is_running,
            "runtime_seconds": runtime,
            "total_tasks_processed": self.total_tasks_processed,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "last_task_time": self.last_task_time,
            "worker_task_done": self.worker_task.done() if self.worker_task else True,
        }

    def log_stats(self) -> None:
        """Log current worker statistics."""
        stats = self.get_stats()
        log_stream_event(self.stream_id, f"{self.worker_type} worker statistics", stats, "DEBUG")

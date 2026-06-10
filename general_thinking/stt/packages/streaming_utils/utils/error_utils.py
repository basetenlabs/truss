import inspect
import logging
import time

logger = logging.getLogger(__name__)


class StreamError(Exception):
    """Base exception for stream processing errors."""

    def __init__(
        self,
        stream_id: str,
        message: str,
        error_type: str = "STREAM_ERROR",
        recoverable: bool = True,
    ):
        self.stream_id = stream_id
        self.message = message
        self.error_type = error_type
        self.recoverable = recoverable
        self.timestamp = time.time()
        super().__init__(f"Stream {stream_id}: {message}")


class TranscriptionError(StreamError):
    def __init__(self, stream_id: str, message: str, recoverable: bool = True):
        super().__init__(stream_id, message, "TRANSCRIPTION_ERROR", recoverable)


class DiarizationError(StreamError):
    def __init__(self, stream_id: str, message: str, recoverable: bool = True):
        super().__init__(stream_id, message, "DIARIZATION_ERROR", recoverable)


class AssignmentError(StreamError):
    def __init__(self, stream_id: str, message: str, recoverable: bool = True):
        super().__init__(stream_id, message, "ASSIGNMENT_ERROR", recoverable)


class WebSocketError(StreamError):
    def __init__(self, stream_id: str, message: str, recoverable: bool = False):
        super().__init__(stream_id, message, "WEBSOCKET_ERROR", recoverable)


class ConfigurationError(StreamError):
    def __init__(self, stream_id: str, message: str, recoverable: bool = False):
        super().__init__(stream_id, message, "CONFIGURATION_ERROR", recoverable)


class AudioProcessingError(StreamError):
    def __init__(self, stream_id: str, message: str, recoverable: bool = False):
        super().__init__(stream_id, message, "AUDIO_PROCESSING_ERROR", recoverable)


def get_calling_class_name() -> str:
    """Get the name of the class that called this function."""
    try:
        # Get the current frame (this function)
        current_frame = inspect.currentframe()
        # Get the caller's frame
        caller_frame = current_frame.f_back
        # Get the caller's caller's frame (the actual calling function)
        actual_caller_frame = caller_frame.f_back

        if actual_caller_frame:
            # Get the class name from the frame's locals
            if "self" in actual_caller_frame.f_locals:
                class_instance = actual_caller_frame.f_locals["self"]
                return class_instance.__class__.__name__
            # Fallback: try to get from frame info
            elif "cls" in actual_caller_frame.f_locals:
                class_instance = actual_caller_frame.f_locals["cls"]
                return class_instance.__name__
    except Exception:
        pass

    return "Unknown"


def log_stream_event(stream_id: str, event: str, details: dict = None, level: str = "INFO") -> None:
    """Centralized logging for stream events with consistent formatting."""
    # Get the calling class name
    class_name = get_calling_class_name()

    log_message = f"[{class_name}] Stream {stream_id}: {event}"
    if details:
        log_message += f" - {details}"

    if level.upper() == "DEBUG":
        logger.debug(log_message)
    elif level.upper() == "WARNING":
        logger.warning(log_message)
    elif level.upper() == "ERROR":
        logger.error(log_message)
    else:
        logger.info(log_message)


def log_performance_metric(
    stream_id: str, operation: str, duration: float, success: bool = True
) -> None:
    """Log performance metrics for operations."""
    status = "✅" if success else "❌"
    logger.debug(f"{status} Stream {stream_id} {operation} completed in {duration:.3f}s")


def create_error_context(stream_id: str, operation: str, additional_context: dict = None) -> dict:
    """Create standardized error context for consistent error reporting."""
    context = {"stream_id": stream_id, "operation": operation, "timestamp": time.time()}
    if additional_context:
        context.update(additional_context)
    return context

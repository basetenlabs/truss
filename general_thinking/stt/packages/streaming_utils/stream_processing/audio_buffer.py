import logging
import time
from typing import Any, Dict, Optional

from streaming_utils.utils.constants import MAX_AUDIO_BUFFER_DURATION, WHISPER_SAMPLE_RATE
from streaming_utils.utils.error_utils import log_stream_event

logger = logging.getLogger(__name__)


class AudioBuffer:
    """Manages audio buffering with automatic overflow handling and comprehensive logging."""

    def __init__(
        self,
        stream_id: str,
        sample_rate: int = WHISPER_SAMPLE_RATE,
        max_duration: float = MAX_AUDIO_BUFFER_DURATION,
        bytes_per_sample: int = 2,
    ):
        self.stream_id = stream_id
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.bytes_per_sample = bytes_per_sample
        self.buffer = bytearray()
        self.creation_time = time.time()
        self.total_chunks_added = 0
        self.total_bytes_added = 0
        self.overflow_count = 0

        log_stream_event(
            stream_id,
            "AudioBuffer initialized",
            {
                "sample_rate": sample_rate,
                "max_duration": max_duration,
                "bytes_per_sample": bytes_per_sample,
                "creation_time": self.creation_time,
            },
            "DEBUG",
        )

    def add_chunk(self, chunk: bytes) -> bool:
        """
        Add audio chunk to buffer, return True if buffer should be processed.

        Args:
            chunk: Audio bytes to add

        Returns:
            bool: True if buffer would exceed max duration and should be processed
        """
        start_time = time.time()

        try:
            # Check if adding this chunk would exceed max duration (like old implementation)
            # Calculate what the duration would be after adding this chunk
            current_samples = len(self.buffer) / self.bytes_per_sample
            new_samples = (len(self.buffer) + len(chunk)) / self.bytes_per_sample
            new_duration = new_samples / self.sample_rate

            should_process = new_duration > self.max_duration

            if should_process:
                self.overflow_count += 1
                log_stream_event(
                    self.stream_id,
                    "Audio buffer overflow detected",
                    {
                        "current_duration": self.duration_seconds(),
                        "new_duration_after_chunk": new_duration,
                        "max_duration": self.max_duration,
                        "buffer_size_bytes": len(self.buffer),
                        "chunk_size_bytes": len(chunk),
                        "overflow_count": self.overflow_count,
                    },
                    "WARNING",
                )

            # Add chunk to buffer
            self.buffer.extend(chunk)
            self.total_chunks_added += 1
            self.total_bytes_added += len(chunk)

            duration = time.time() - start_time
            log_stream_event(
                self.stream_id,
                "Audio chunk added to buffer",
                {
                    "chunk_size_bytes": len(chunk),
                    "total_buffer_size_bytes": len(self.buffer),
                    "current_duration": self.duration_seconds(),
                    "should_process": should_process,
                    "add_time": duration,
                },
                "DEBUG",
            )

            return should_process

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"❌ Stream {self.stream_id}: Failed to add audio chunk after {duration:.3f}s: {e}"
            )
            raise

    def duration_seconds(self) -> float:
        """Calculate current buffer duration in seconds."""
        total_samples = len(self.buffer) / self.bytes_per_sample
        return total_samples / self.sample_rate

    def get_buffer(self) -> bytes:
        """Get current buffer as bytes."""
        return bytes(self.buffer)

    def clear(self) -> None:
        """Clear the buffer and log statistics."""
        buffer_stats = self.get_stats()

        self.buffer = bytearray()

        log_stream_event(self.stream_id, "Audio buffer cleared", buffer_stats, "DEBUG")

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics for monitoring."""
        return {
            "stream_id": self.stream_id,
            "buffer_size_bytes": len(self.buffer),
            "duration_seconds": self.duration_seconds(),
            "total_chunks_added": self.total_chunks_added,
            "total_bytes_added": self.total_bytes_added,
            "overflow_count": self.overflow_count,
            "max_duration": self.max_duration,
            "sample_rate": self.sample_rate,
            "age_seconds": time.time() - self.creation_time,
        }

    def log_stats(self) -> None:
        """Log current buffer statistics."""
        stats = self.get_stats()
        log_stream_event(self.stream_id, "Audio buffer statistics", stats, "DEBUG")

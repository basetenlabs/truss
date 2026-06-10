import logging
import sys
import time
from typing import Any, Dict, List, Optional

from pyannote.core import Annotation
from streaming_utils.utils.error_utils import log_stream_event
from whisper_utils.data_types import Segment, StreamingWhisperInput, WhisperResult
from whisper_utils.utilities import merge_segments

logger = logging.getLogger(__name__)


class StreamState:
    """Encapsulates all state and data for a single stream with comprehensive logging."""

    def __init__(self, stream_id: str, metadata: StreamingWhisperInput, metrics: Dict[str, Any]):
        self.stream_id = stream_id
        self.metadata = metadata

        # Audio processing state
        self.time_offset = 0.0
        self.total_duration_sec = 0.0
        self.last_transcription_end_sec = 0.0

        # Transcription state
        self.transcription_num = 0
        self.final_segments_buffer: List[Segment] = []
        self.partial_segments_buffer: Optional[Segment] = None
        self.prev_whisper_result: Optional[WhisperResult] = None
        self.prefix = ""

        # Diarization state
        self.accumulated_diarization = Annotation()
        self.diarization_coverage_start = float("inf")
        self.diarization_coverage_end = 0.0

        # Synchronization state
        self.pending_assignments = []

        # Statistics
        self.creation_time = time.time()
        self.total_segments_processed = 0
        self.total_diarization_windows = 0
        self.prev_partial_time = time.time()
        self.prev_final_time = time.time()
        self.most_recent_partial_latency = 0.0

        self.metrics = metrics

        log_stream_event(
            stream_id,
            "StreamState initialized",
            {"metadata": str(metadata), "creation_time": self.creation_time},
            "DEBUG",
        )

    def observe_partial_latency(self) -> None:
        """Update partial latency."""
        time_now = time.time()
        latency = time_now - self.prev_partial_time
        self.most_recent_partial_latency = latency
        self.metrics["partial_latency_histogram"].observe(latency)
        self.prev_partial_time = time_now

    def observe_final_latency(self) -> None:
        """Update final latency."""
        time_now = time.time()
        latency = time_now - self.prev_final_time

        diarized = "true" if self.metadata.whisper_params.enable_diarization else "false"
        self.metrics["final_latency_histogram"].labels(diarized=diarized).observe(latency)

        self.prev_final_time = time_now

    def update_time_offset(self, new_offset: float) -> None:
        """Update time offset and log the change."""
        old_offset = self.time_offset
        self.time_offset = new_offset

        log_stream_event(
            self.stream_id,
            "Time offset updated",
            {
                "old_offset": old_offset,
                "new_offset": new_offset,
                "offset_change": new_offset - old_offset,
            },
            "DEBUG",
        )

    def update_total_duration(self, new_duration: float) -> None:
        """Update total duration and log the change."""
        old_duration = self.total_duration_sec
        self.total_duration_sec = new_duration

        log_stream_event(
            self.stream_id,
            "Total duration updated",
            {
                "old_duration": old_duration,
                "new_duration": new_duration,
                "duration_change": new_duration - old_duration,
            },
            "DEBUG",
        )

    def get_partial_result(self) -> WhisperResult:
        """
        Returns the current partial by merging several states
        """
        if not self.metadata.whisper_params.enable_diarization:
            return self.prev_whisper_result or WhisperResult()

        # Handle base case for partials without prioritize latency
        if not self.metadata.streaming_params.prioritize_latency:
            partial_result = WhisperResult(
                segments=[
                    merge_segments(
                        self.partial_segments_buffer,
                        *self.prev_whisper_result.segments if self.prev_whisper_result else [],
                    )
                ]
            )
            return partial_result

        partial_result = WhisperResult(
            segments=[
                merge_segments(
                    *self.final_segments_buffer,
                    self.partial_segments_buffer,
                    *self.prev_whisper_result.segments if self.prev_whisper_result else [],
                )
            ]
        )
        return partial_result

    # def add_final_segment(self, segment: Segment) -> None:
    #     """Add segment to final segments buffer."""
    #     self.final_segments_buffer.append(segment)
    #     self.total_segments_processed += 1

    #     log_stream_event(
    #         self.stream_id,
    #         "Final segment added to buffer",
    #         {
    #             "segment_text": segment.text,
    #             "segment_start": segment.start_time,
    #             "segment_end": segment.end_time,
    #             "buffer_size": len(self.final_segments_buffer),
    #             "total_segments_processed": self.total_segments_processed,
    #         },
    #         "DEBUG",
    #     )

    #     # Log updated buffer content
    #     self.log_final_segments_buffer()

    def pop_final_segment(self) -> Optional[Segment]:
        """Pop segment from final segments buffer."""
        if not self.final_segments_buffer:
            return None

        segment = self.final_segments_buffer.pop(0)

        log_stream_event(
            self.stream_id,
            "Final segment popped from buffer",
            {
                "segment_text": segment.text,
                "segment_start": segment.start_time,
                "segment_end": segment.end_time,
                "remaining_buffer_size": len(self.final_segments_buffer),
            },
            "DEBUG",
        )

        # Update memory metrics after buffer modification
        self.update_memory_metrics()

        return segment

    def update_partial_segment(self, segment: Optional[Segment]) -> None:
        """Update partial segments buffer."""
        old_segment = self.partial_segments_buffer
        self.partial_segments_buffer = segment

        log_stream_event(
            self.stream_id,
            "Partial segment updated",
            {
                "old_segment_text": old_segment.text if old_segment else None,
                "new_segment_text": segment.text if segment else None,
                "old_segment_start": old_segment.start_time if old_segment else None,
                "new_segment_start": segment.start_time if segment else None,
            },
            "DEBUG",
        )

    def update_diarization_coverage(self, start_time: float, end_time: float) -> None:
        """Update diarization coverage range."""
        old_start = self.diarization_coverage_start
        old_end = self.diarization_coverage_end

        self.diarization_coverage_start = min(self.diarization_coverage_start, start_time)
        self.diarization_coverage_end = max(self.diarization_coverage_end, end_time)

        if old_start != self.diarization_coverage_start or old_end != self.diarization_coverage_end:
            log_stream_event(
                self.stream_id,
                "Diarization coverage updated",
                {
                    "old_coverage": f"[{old_start:.2f}s, {old_end:.2f}s]",
                    "new_coverage": f"[{self.diarization_coverage_start:.2f}s, {self.diarization_coverage_end:.2f}s]",
                    "coverage_expansion": {
                        "start_expanded": old_start > self.diarization_coverage_start,
                        "end_expanded": old_end < self.diarization_coverage_end,
                    },
                },
                "DEBUG",
            )

    def has_sufficient_diarization_coverage(self, start_time: float, end_time: float) -> bool:
        """Check if we have diarization results that cover the given time range."""
        if len(self.accumulated_diarization) == 0:
            return False
        return self.diarization_coverage_end >= end_time

    def increment_transcription_num(self) -> int:
        """Increment transcription number and return new value."""
        self.transcription_num += 1

        log_stream_event(
            self.stream_id,
            "Transcription number incremented",
            {"new_transcription_num": self.transcription_num},
            "DEBUG",
        )

        return self.transcription_num

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive state statistics for monitoring."""
        return {
            "stream_id": self.stream_id,
            "time_offset": self.time_offset,
            "total_duration_sec": self.total_duration_sec,
            "transcription_num": self.transcription_num,
            "final_segments_buffer_size": len(self.final_segments_buffer),
            "partial_segments_buffer_exists": self.partial_segments_buffer is not None,
            "prefix_length": len(self.prefix),
            "diarization_coverage": f"[{self.diarization_coverage_start:.2f}s, {self.diarization_coverage_end:.2f}s]",
            "accumulated_diarization_segments": len(self.accumulated_diarization),
            "total_segments_processed": self.total_segments_processed,
            "total_diarization_windows": self.total_diarization_windows,
            "age_seconds": time.time() - self.creation_time,
        }

    def log_stats(self) -> None:
        """Log current state statistics."""
        stats = self.get_stats()
        log_stream_event(self.stream_id, "StreamState statistics", stats, "DEBUG")

    def update_memory_metrics(self) -> None:
        """Update memory leak monitoring metrics."""
        pass  # comment out for load testing
        # try:
        #     # Update final segments buffer size
        #     self.metrics["final_segments_buffer_size_gauge"].labels(stream_id=self.stream_id).set(
        #         len(self.final_segments_buffer)
        #     )

        #     # Update accumulated diarization size
        #     self.metrics["accumulated_diarization_size_gauge"].labels(stream_id=self.stream_id).set(
        #         len(self.accumulated_diarization)
        #     )

        #     # Update pending assignments size
        #     self.metrics["pending_assignments_size_gauge"].labels(stream_id=self.stream_id).set(
        #         len(self.pending_assignments)
        #     )

        #     # Estimate total memory usage
        #     # estimated_memory = self._estimate_memory_usage()
        #     # self.metrics["stream_state_memory_usage_gauge"].labels(
        #     #     stream_id=self.stream_id
        #     # ).set(estimated_memory)

        # except Exception as e:
        #     logger.error(f"❌ Stream {self.stream_id}: Failed to update memory metrics: {e}")

    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of StreamState in bytes."""
        try:
            # Base object size
            total_size = sys.getsizeof(self)

            # Add size of major data structures
            total_size += sys.getsizeof(self.final_segments_buffer)
            total_size += sys.getsizeof(self.accumulated_diarization)
            total_size += sys.getsizeof(self.pending_assignments)

            # Add size of individual segments in final_segments_buffer
            for segment in self.final_segments_buffer:
                total_size += sys.getsizeof(segment)
                if hasattr(segment, "text"):
                    total_size += sys.getsizeof(segment.text)
                if hasattr(segment, "word_timestamps"):
                    total_size += sys.getsizeof(segment.word_timestamps)

            # Add size of diarization segments - iterate through the annotation properly
            try:
                for segment, track, label in self.accumulated_diarization.itertracks(
                    yield_label=True
                ):
                    total_size += sys.getsizeof(segment)
                    total_size += sys.getsizeof(track)
                    total_size += sys.getsizeof(label)
            except Exception as diarization_error:
                # If we can't iterate through diarization, just add the annotation size
                logger.debug(f"Could not iterate through diarization segments: {diarization_error}")
                pass

            return total_size

        except Exception as e:
            logger.error(f"❌ Stream {self.stream_id}: Failed to estimate memory usage: {e}")
            return 0

    def log_final_segments_buffer(self) -> None:
        """Log detailed information about the final_segments_buffer (similar to model_old.py)."""
        try:
            if not self.final_segments_buffer:
                log_stream_event(self.stream_id, "Final segments buffer is empty", {}, "DEBUG")
                return

            # Log summary
            total_text = " ".join([seg.text for seg in self.final_segments_buffer])
            total_duration = sum(
                [seg.end_time - seg.start_time for seg in self.final_segments_buffer]
            )

            log_stream_event(
                self.stream_id,
                "Final segments buffer summary",
                {
                    "segment_count": len(self.final_segments_buffer),
                    "total_duration": f"{total_duration:.3f}s",
                    "total_text_length": len(total_text),
                    "total_text_preview": total_text[:100] + "..."
                    if len(total_text) > 100
                    else total_text,
                },
                "INFO",
            )

            # Log detailed segment information
            segment_details = []
            for i, segment in enumerate(self.final_segments_buffer):
                segment_details.append(
                    {
                        "index": i,
                        "text": segment.text[:50] + "..."
                        if len(segment.text) > 50
                        else segment.text,
                        "start_time": f"{segment.start_time:.3f}s",
                        "end_time": f"{segment.end_time:.3f}s",
                        "duration": f"{segment.end_time - segment.start_time:.3f}s",
                        "word_count": len(segment.word_timestamps)
                        if hasattr(segment, "word_timestamps")
                        else 0,
                        "speaker": getattr(segment, "speaker", "UNASSIGNED"),
                    }
                )

            log_stream_event(
                self.stream_id,
                "Final segments buffer details",
                {"segments": segment_details},
                "DEBUG",
            )

        except Exception as e:
            logger.error(f"❌ Stream {self.stream_id}: Failed to log final segments buffer: {e}")

    def log_queue_sizes(self) -> None:
        """Log queue sizes for all workers (if available)."""
        try:
            queue_info = {"stream_id": self.stream_id, "timestamp": time.time()}

            # Note: Queue sizes are logged in individual workers
            # This method can be used to log a summary if needed

            log_stream_event(self.stream_id, "Queue sizes summary", queue_info, "DEBUG")

        except Exception as e:
            logger.error(f"❌ Stream {self.stream_id}: Failed to log queue sizes: {e}")

    def reset_for_new_segment(self) -> None:
        """Reset state for processing a new audio segment."""
        old_transcription_num = self.transcription_num
        old_last_transcription_end = self.last_transcription_end_sec

        self.last_transcription_end_sec = 0.0

        log_stream_event(
            self.stream_id,
            "State reset for new segment",
            {
                "old_transcription_num": old_transcription_num,
                "old_last_transcription_end": old_last_transcription_end,
                "new_last_transcription_end": self.last_transcription_end_sec,
            },
            "DEBUG",
        )

    def cleanup(self) -> None:
        """Clean up StreamState to prevent memory leaks."""
        try:
            log_stream_event(
                self.stream_id,
                "Starting StreamState cleanup",
                {
                    "final_segments_buffer_size": len(self.final_segments_buffer),
                    "accumulated_diarization_size": len(self.accumulated_diarization),
                    "pending_assignments_size": len(self.pending_assignments),
                },
                "DEBUG",
            )

            # Clear all accumulating data structures
            self.final_segments_buffer.clear()
            self.accumulated_diarization = Annotation()  # Reset to empty annotation
            self.pending_assignments.clear()

            # Clear large objects
            self.prev_whisper_result = None
            self.partial_segments_buffer = None

            # Reset coverage tracking
            self.diarization_coverage_start = float("inf")
            self.diarization_coverage_end = 0.0

            # Reset other state
            self.prefix = ""
            self.transcription_num = 0
            self.last_transcription_end_sec = 0.0
            self.total_segments_processed = 0
            self.total_diarization_windows = 0

            log_stream_event(
                self.stream_id,
                "StreamState cleanup completed",
                {},
                "DEBUG",
            )

        except Exception as e:
            logger.error(f"❌ Stream {self.stream_id}: Failed to cleanup StreamState: {e}")

    # def cleanup_old_diarization_segments(self, current_time: float, buffer_seconds: float = 10.0) -> int:
    #     """Clean up old diarization segments to prevent memory leak.

    #     Args:
    #         current_time: Current processing time
    #         buffer_seconds: Keep segments within this buffer of current time

    #     Returns:
    #         Number of segments removed
    #     """
    #     try:
    #         from pyannote.core import Segment, Timeline

    #         if len(self.accumulated_diarization) == 0:
    #             return 0

    #         cleanup_threshold = current_time - buffer_seconds

    #         if cleanup_threshold <= 0:
    #             return 0

    #         segments_before = len(self.accumulated_diarization)

    #         # Create a timeline with segments that should be kept (after cleanup_threshold)
    #         keep_segment = Segment(cleanup_threshold, float('inf'))
    #         keep_timeline = Timeline(segments=[keep_segment], uri=self.accumulated_diarization.uri or "cleanup")

    #         # Use crop with 'loose' mode to efficiently remove old segments
    #         cleaned_annotation = self.accumulated_diarization.crop(keep_timeline, mode="loose")

    #         # Update the accumulated diarization
    #         self.accumulated_diarization = cleaned_annotation

    #         # Update diarization coverage
    #         if len(cleaned_annotation) > 0:
    #             timeline = cleaned_annotation.get_timeline(copy=False)
    #             extent = timeline.extent()
    #             new_start = extent[0]
    #             new_end = extent[1]
    #             self.update_diarization_coverage(new_start, new_end)
    #         else:
    #             self.diarization_coverage_start = float("inf")
    #             self.diarization_coverage_end = 0.0

    #         segments_removed = segments_before - len(self.accumulated_diarization)

    #         if segments_removed > 0:
    #             log_stream_event(
    #                 self.stream_id,
    #                 "Cleaned up old diarization segments (periodic cleanup)",
    #                 {
    #                     "segments_before": segments_before,
    #                     "segments_after": len(self.accumulated_diarization),
    #                     "segments_removed": segments_removed,
    #                     "cleanup_threshold": cleanup_threshold,
    #                     "current_time": current_time,
    #                     "buffer_seconds": buffer_seconds,
    #                 },
    #                 "DEBUG",
    #             )

    #             # Update memory metrics
    #             self.update_memory_metrics()

    #         return segments_removed

    #     except Exception as e:
    #         logger.error(f"❌ Stream {self.stream_id}: Failed to cleanup old diarization segments: {e}")
    #         return 0

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from streaming_utils.utils.constants import ASSIGNMENT_WORKER_TYPE
from streaming_utils.utils.error_utils import (
    AssignmentError,
    log_performance_metric,
    log_stream_event,
)
from streaming_utils.utils.websocket_utils import WebSocketManager
from whisper_utils.constants import DEFAULT_STREAMING_POST_PROCESSING_FLAGS
from whisper_utils.utilities import post_process_whisper_result

from ..stream_state import StreamState
from .base_worker import BaseWorker

logger = logging.getLogger(__name__)


class StreamAssignmentWorker(BaseWorker):
    """Manages speaker assignment for transcript segments in this stream."""

    def __init__(
        self,
        stream_id: str,
        state: StreamState,
        metrics: Dict[str, Any],
        ws_manager: WebSocketManager,
    ):
        super().__init__(stream_id, ASSIGNMENT_WORKER_TYPE, ws_manager)
        self.state = state
        self.ws_manager = ws_manager
        self.metrics = metrics

        # Queue for assignment tasks
        self.assignment_queue = asyncio.Queue()

        log_stream_event(self.stream_id, "StreamAssignmentWorker initialized")

    async def queue_assignment_task(
        self,
        transcription_num: int,
        final: bool,
        task_start_time: float = 0.0,
        audio_length_sec: Optional[float] = None,
    ) -> None:
        """Queue an assignment task for processing."""
        try:
            task = {
                "transcription_num": transcription_num,
                "final": final,
                "task_start_time": task_start_time,
                "audio_length_sec": audio_length_sec,
            }

            await self.assignment_queue.put(task)

            # Observe queue size
            self.metrics["assignment_queue_gauge"].set(self.assignment_queue.qsize())

            log_stream_event(
                self.stream_id,
                "Assignment task queued",
                {
                    "transcription_num": transcription_num,
                    "final": final,
                    "queue_size": self.assignment_queue.qsize(),
                },
                "DEBUG",
            )

        except asyncio.QueueFull:
            logger.warning(f"⚠️ Stream {self.stream_id}: Assignment queue is full, dropping task")
        except Exception as e:
            logger.error(f"❌ Stream {self.stream_id}: Failed to queue assignment task: {e}")

    async def process_assignment_task(
        self,
        transcription_num: int,
        final: bool,
        task_start_time: float = 0.0,
        audio_length_sec: Optional[float] = None,
    ) -> None:
        """Process an assignment task inline."""
        try:
            task = {
                "transcription_num": transcription_num,
                "final": final,
                "task_start_time": task_start_time,
                "audio_length_sec": audio_length_sec,
            }

            await self._process_task(task)

        except Exception as e:
            logger.error(f"❌ Stream {self.stream_id}: Failed to process assignment task: {e}")
            raise

    async def _get_task_with_timeout(self) -> Optional[Dict[str, Any]]:
        """Get task from queue with timeout."""
        try:
            return await asyncio.wait_for(self.assignment_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    async def _process_task(self, task: Dict[str, Any]) -> None:
        """Process a single assignment task."""
        start_time = time.time()

        try:
            transcription_num = task["transcription_num"]
            final = task["final"]
            task_start_time = task["task_start_time"]
            audio_length_sec = task.get("audio_length_sec")

            log_stream_event(
                self.stream_id,
                "Processing assignment task",
                {
                    "transcription_num": transcription_num,
                    "final": final,
                    "task_start_time": task_start_time,
                },
                "DEBUG",
            )

            # Read final segment
            final_segment = self.state.final_segments_buffer[0]
            if not final_segment:
                logger.warning(
                    f"⚠️ Stream {self.stream_id}: No final segment available for assignment"
                )
                return

            pre_assignment_latency = time.time() - task_start_time
            self.metrics["pre_assignment_latency_histogram"].observe(pre_assignment_latency)

            # Wait for diarization coverage
            await self._wait_for_diarization_coverage(
                final_segment.start_time, final_segment.end_time
            )

            # Assign speakers to segments
            final_result = self._assign_speakers_to_transcript(final_segment)

            # Pop to remove final segment
            self.state.pop_final_segment()

            # Log speaker assignment results (similar to model_old.py)
            log_stream_event(
                self.stream_id,
                "Speaker assignment completed",
                {
                    "transcription_num": transcription_num,
                    "final": final,
                    "segments_count": len(final_result.segments),
                    "diarization_data_count": (
                        len(final_result.diarization) if final_result.diarization else 0
                    ),
                },
                "DEBUG",
            )

            # Log detailed segment information with speakers
            if final_result.segments:
                segment_info = []
                for i, segment in enumerate(final_result.segments):
                    segment_info.append(
                        {
                            "index": i,
                            "speaker": getattr(segment, "speaker", "UNKNOWN"),
                            "text": (
                                segment.text[:50] + "..."
                                if len(segment.text) > 50
                                else segment.text
                            ),
                            "start_time": f"{segment.start_time:.3f}s",
                            "end_time": f"{segment.end_time:.3f}s",
                            "duration": f"{segment.end_time - segment.start_time:.3f}s",
                        }
                    )

                log_stream_event(
                    self.stream_id,
                    "Final result with speaker assignment",
                    {"segments": segment_info},
                    "DEBUG",
                )

            # Observe final latency
            self.state.observe_final_latency()

            # Send final result
            await self._send_diarized_final_transcription(
                final_result, transcription_num, final, audio_length_sec, task_start_time
            )

            processing_time = time.time() - start_time
            log_performance_metric(self.stream_id, "assignment_task", processing_time, success=True)

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"❌ Stream {self.stream_id}: Assignment task failed after {processing_time:.3f}s: {e}"
            )
            log_performance_metric(
                self.stream_id, "assignment_task", processing_time, success=False
            )
            if self.ws_manager:
                await self.ws_manager.send_error_to_websocket(
                    AssignmentError(
                        self.stream_id, f"Assignment task failed: {str(e)}", recoverable=True
                    ),
                )
        finally:
            # Mark task as done
            self.assignment_queue.task_done()

    async def _wait_for_diarization_coverage(self, start_time: float, end_time: float) -> None:
        """Wait until diarization results cover the audio time range."""
        from streaming_utils.utils.constants import (
            DIARIZATION_WAIT_INTERVAL,
            MAX_DIARIZATION_WAIT_TIME,
        )

        log_stream_event(
            self.stream_id,
            "Waiting for diarization coverage",
            {"start_time": start_time, "end_time": end_time},
            "DEBUG",
        )

        start_wait = self.state.creation_time + end_time

        while True:
            if self.state.has_sufficient_diarization_coverage(start_time, end_time):
                wait_time = time.time() - start_wait
                log_stream_event(
                    self.stream_id, "Diarization coverage ready", {"wait_time": wait_time}, "DEBUG"
                )
                return
            if time.time() - start_wait > MAX_DIARIZATION_WAIT_TIME:
                break
            await asyncio.sleep(DIARIZATION_WAIT_INTERVAL)

        # Timeout
        wait_time = time.time() - start_wait
        logger.warning(
            f"Stream {self.stream_id}: Timeout waiting for diarization coverage for segment [{start_time:.3f}s - {end_time:.3f}s] after {wait_time:.3f}s"
        )
        self.metrics["diarization_timeout_counter"].inc()

    def _assign_speakers_to_transcript(self, segment) -> Any:
        """Assign speakers to transcript segments using accumulated diarization annotation."""
        try:
            from streaming_utils.utils.constants import (
                DIARIZATION_PATCH_COLLAR,
                MIN_SEGMENT_DURATION,
            )
            from whisper_utils.data_types import WhisperResult
            from whisper_utils.utilities import assign_speakers_to_segments

            # Create WhisperResult with the segment
            whisper_result = WhisperResult(
                segments=[segment],
                language_code=None,
                language_prob=None,
                diarization=[],
            )

            # Check if diarization is enabled
            if not self.state.metadata.whisper_params.enable_diarization:
                log_stream_event(
                    self.stream_id,
                    "Diarization disabled, returning segment without speaker assignment",
                    {
                        "segment_text": segment.text,
                        "segment_start": segment.start_time,
                        "segment_end": segment.end_time,
                    },
                    "DEBUG",
                )
                return whisper_result

            # Check if we have diarization data
            if len(self.state.accumulated_diarization) == 0:
                log_stream_event(
                    self.stream_id,
                    "No diarization data available, returning segment without speaker assignment",
                    {
                        "segment_text": segment.text,
                        "segment_start": segment.start_time,
                        "segment_end": segment.end_time,
                    },
                    "DEBUG",
                )
                return whisper_result

            # Apply final patching like diart's PredictionAccumulator.get_prediction()
            final_annotation = self.state.accumulated_diarization.support(DIARIZATION_PATCH_COLLAR)

            # Apply post-processing: Remove segments smaller than minimum duration
            segments_before_filter = len(final_annotation)
            filtered_final_annotation = self._filter_short_segments(
                final_annotation, min_duration=MIN_SEGMENT_DURATION
            )
            segments_after_filter = len(filtered_final_annotation)

            if segments_before_filter != segments_after_filter:
                log_stream_event(
                    self.stream_id,
                    "Filtered short diarization segments",
                    {
                        "segments_before": segments_before_filter,
                        "segments_after": segments_after_filter,
                        "removed_count": segments_before_filter - segments_after_filter,
                        "min_duration": MIN_SEGMENT_DURATION,
                    },
                    "DEBUG",
                )

            # Create a mock diarization result compatible with assign_speakers_to_segments
            class AnnotationDiarizationResult:
                """Wrapper to make pyannote Annotation compatible with assign_speakers_to_segments."""

                def __init__(self, annotation):
                    self.annotation = annotation

                def itertracks(self, yield_label=True):
                    """Iterate over tracks with speaker labels."""
                    return self.annotation.itertracks(yield_label=yield_label)

            diarization_result = AnnotationDiarizationResult(filtered_final_annotation)

            # Extract diarization data in the format expected by assign_speakers_to_segments
            diarization_data = []
            diarize_speakers = []
            diarize_starts = []
            diarize_ends = []

            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                diarization_data.append({"speaker": speaker, "start": turn.start, "end": turn.end})
                diarize_speakers.append(speaker)
                diarize_starts.append(turn.start)
                diarize_ends.append(turn.end)

            diarization_stack = (diarization_data, diarize_speakers, diarize_starts, diarize_ends)

            # Assign speakers to segments
            segments_with_speakers, diarization_data = assign_speakers_to_segments(
                whisper_result.segments,
                diarization_stack,
                fill_nearest=True,
                diarization_granularity=self.state.metadata.whisper_params.diarization_granularity,
            )

            # Create new WhisperResult with speaker-assigned segments
            result_with_speakers = WhisperResult(
                segments=segments_with_speakers,
                language_code=whisper_result.language_code,
                language_prob=whisper_result.language_prob,
                diarization=diarization_data,
            )

            log_stream_event(
                self.stream_id,
                "Speaker assignment completed",
                {
                    "segment_text": segment.text,
                    "segment_start": segment.start_time,
                    "segment_end": segment.end_time,
                    "assigned_speaker": getattr(segments_with_speakers[0], "speaker", "UNKNOWN"),
                    "diarization_segments_count": len(diarization_data),
                },
                "DEBUG",
            )

            # Clean up old diarization segments to prevent memory leak
            self._cleanup_old_diarization_segments(segment.end_time)

            return result_with_speakers

        except Exception as e:
            logger.error(f"❌ Stream {self.stream_id}: Failed to assign speakers: {e}")
            import traceback

            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise

    def _cleanup_old_diarization_segments(self, current_end_time: float) -> None:
        """Clean up old diarization segments to prevent memory leak.

        Removes all diarization segments that ended before current_end_time,
        keeping only segments that might still be relevant for future assignments.
        """
        try:
            from streaming_utils.utils.constants import MIN_SEGMENT_DURATION

            if len(self.state.accumulated_diarization) == 0:
                return

            annotations = list(self.state.accumulated_diarization.itertracks(yield_label=True))

            keep_annotation_index = 0

            # find the latest annotation that would support current_end_time and has a duration greater than MIN_SEGMENT_DURATION
            for i in range(len(annotations)):
                if (
                    annotations[i][0].start < current_end_time
                    and annotations[i][0].duration > MIN_SEGMENT_DURATION
                ):
                    keep_annotation_index = i

            # Then, delete all annotations before keep_annotation_index
            # logger.info(f"Deleting {keep_annotation_index} annotations")
            for i in range(keep_annotation_index):
                del self.state.accumulated_diarization[annotations[i][0]]

        except Exception as e:
            logger.error(
                f"❌ Stream {self.stream_id}: Failed to cleanup old diarization segments: {e}"
            )
            # Don't raise - this is cleanup, not critical functionality

    def _filter_short_segments(self, annotation, min_duration: float = 1.0):
        """
        Filter out segments shorter than the specified minimum duration.

        Args:
            annotation: pyannote.core.Annotation object
            min_duration: Minimum duration in seconds (default: 1.0)

        Returns:
            Filtered pyannote.core.Annotation object
        """
        from pyannote.core import Annotation

        filtered_annotation = Annotation()

        for segment, track, speaker in annotation.itertracks(yield_label=True):
            duration = segment.end - segment.start
            if duration >= min_duration:
                filtered_annotation[segment, track] = speaker
            else:
                log_stream_event(
                    self.stream_id,
                    "Removing short diarization segment",
                    {
                        "speaker": speaker,
                        "start_time": segment.start,
                        "end_time": segment.end,
                        "duration": duration,
                        "min_duration": min_duration,
                    },
                    "DEBUG",
                )

        return filtered_annotation

    async def _send_diarized_final_transcription(
        self,
        result: Any,
        transcription_num: int,
        final: bool,
        audio_length_sec: Optional[float] = None,
        task_start_time: float = 0.0,
    ) -> None:
        """Send final result with speaker assignment."""
        try:
            from streaming_utils.utils.audio_utils import ensure_json_serializable
            from whisper_utils.data_types import StreamingWhisperResult

            # Observe pipeline latency (from _handle_audio_chunk to send_json_safe)
            pipeline_latency = None
            if task_start_time > 0:
                pipeline_latency = time.time() - task_start_time
                self.metrics["pipeline_final_latency_seconds_histogram"].labels(
                    diarized="true"
                ).observe(pipeline_latency)

            include_pipeline_latency = (
                self.state.metadata.streaming_params.include_pipeline_latency_metric
            )

            log_stream_event(
                self.stream_id,
                "Sending final result with speaker assignment",
                details={
                    "transcription_num": transcription_num,
                    "segments_count": len(result.segments),
                    "final_segments_buffer": self.state.final_segments_buffer,
                    "partial_segments_buffer": self.state.partial_segments_buffer,
                    "next_partial": self.state.get_partial_result().segments[0],
                },
                level="DEBUG",
            )

            streaming_result = StreamingWhisperResult(
                segments=result.segments,
                diarization=result.diarization,
                is_final=final,
                transcription_num=transcription_num,
                next_partial=self.state.get_partial_result().segments[0],
                audio_length_sec=audio_length_sec,
                pipeline_latency=pipeline_latency if include_pipeline_latency else None,
            )

            streaming_result = post_process_whisper_result(
                streaming_result, DEFAULT_STREAMING_POST_PROCESSING_FLAGS
            )

            # Convert to dict and ensure JSON serializable
            result_dict = streaming_result.model_dump()
            result_dict["type"] = "transcription"
            result_dict = ensure_json_serializable(result_dict, self.stream_id)

            await self.ws_manager.send_json_safe(result_dict)

        except Exception as e:
            logger.error(f"❌ Stream {self.stream_id}: Failed to send final result: {e}")
            raise

    async def wait_for_completion(self) -> None:
        """Wait for all queued tasks to complete."""
        try:
            await self.assignment_queue.join()
            log_stream_event(self.stream_id, "All assignment tasks completed")
        except Exception as e:
            logger.error(f"❌ Stream {self.stream_id}: Error waiting for assignment completion: {e}")
            raise

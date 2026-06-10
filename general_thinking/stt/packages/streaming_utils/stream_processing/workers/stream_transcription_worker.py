import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from streaming_utils.utils.audio_utils import ensure_json_serializable
from streaming_utils.utils.constants import TRANSCRIPTION_WORKER_TYPE
from streaming_utils.utils.error_utils import TranscriptionError, log_stream_event
from streaming_utils.utils.websocket_utils import WebSocketManager
from streaming_utils.whisper_service_base import WhisperServiceBase
from whisper_utils.constants import DEFAULT_STREAMING_POST_PROCESSING_FLAGS
from whisper_utils.data_types import StreamingWhisperInput, WhisperResult
from whisper_utils.utilities import (
    get_terminal_punctuation,
    merge_segments,
    post_process_whisper_result,
)

from ..stream_state import StreamState
from .base_worker import BaseWorker

logger = logging.getLogger(__name__)


class StreamTranscriptionWorker(BaseWorker):
    """Processes transcription tasks for a specific stream in chronological order."""

    def __init__(
        self,
        stream_id: str,
        state: StreamState,
        metrics: Dict[str, Any],
        ws_manager: WebSocketManager,
        metadata: StreamingWhisperInput,
        assignment_callback=None,
    ):
        super().__init__(stream_id, TRANSCRIPTION_WORKER_TYPE, ws_manager)
        self.state = state
        self.metadata = metadata
        self.assignment_callback = assignment_callback
        self.whisper_service = WhisperServiceBase.get_instance()
        self.terminal_punctuation = get_terminal_punctuation()

        # Queue for transcription tasks
        self.transcription_queue = asyncio.Queue()

        self.metrics = metrics

        log_stream_event(self.stream_id, "✅ StreamTranscriptionWorker initialized")

    async def queue_transcription_task(
        self,
        audio_buffer: bytes,
        metadata: StreamingWhisperInput,
        time_offset: float,
        curr_end_sec: float,
        transcription_num: int,
        final: bool,
        prefix: str = "",
        end_audio: bool = False,
        task_start_time: float = 0.0,
        audio_length_sec: Optional[float] = None,
    ) -> None:
        """Queue a transcription task for processing."""
        try:
            task = {
                "audio_buffer": audio_buffer,
                "metadata": metadata,
                "time_offset": time_offset,
                "curr_end_sec": curr_end_sec,
                "transcription_num": transcription_num,
                "final": final,
                "prefix": prefix,
                "end_audio": end_audio,
                "task_start_time": task_start_time,
                "audio_length_sec": audio_length_sec,
            }

            await self.transcription_queue.put(task)

            # Observe queue size
            self.metrics["transcription_queue_gauge"].set(self.transcription_queue.qsize())

            log_stream_event(
                self.stream_id,
                "Transcription task queued",
                {
                    "transcription_num": transcription_num,
                    "final": final,
                    "audio_buffer_size": len(audio_buffer),
                    "queue_size": self.transcription_queue.qsize(),
                },
                "DEBUG",
            )

        except asyncio.QueueFull:
            logger.warning(
                f"⚠️ Stream {self.stream_id}: Transcription queue is full, dropping task"
            )

    async def _get_task_with_timeout(self) -> Optional[Dict[str, Any]]:
        """Get task from queue with timeout."""
        try:
            return await asyncio.wait_for(self.transcription_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    async def _process_task(self, task: Dict[str, Any]) -> None:
        """Process a single transcription task."""
        start_time = time.time()

        try:
            # Extract task parameters
            audio_buffer = task["audio_buffer"]
            metadata = task["metadata"]
            time_offset = task["time_offset"]
            transcription_num = task["transcription_num"]
            final = task["final"]
            prefix = task["prefix"]
            end_audio = task.get("end_audio", False)
            task_start_time = task.get("task_start_time", 0.0)
            audio_length_sec = task.get("audio_length_sec")

            # Call Whisper service
            start_time = time.time()
            whisper_result = await self.whisper_service.transcribe(
                audio_data=audio_buffer,
                params=metadata.whisper_params,
                time_offset=time_offset,
                prefix=prefix,
                final=final,
                stream_id=self.stream_id,
            )
            transcription_time = time.time() - start_time
            self.metrics["transcription_latency_histogram"].observe(transcription_time)

            # Update prefix
            self.state.prefix = " ".join([segment.text for segment in whisper_result.segments])

            # Process result based on final vs partial
            if not final:
                await self._process_partial_transcript(
                    whisper_result, transcription_num, audio_length_sec, task_start_time
                )
            else:
                await self._process_final_transcript(
                    whisper_result, transcription_num, end_audio, task_start_time, audio_length_sec
                )

        except Exception as e:
            raise TranscriptionError(self.stream_id, f"Transcription task failed: {str(e)}")
        finally:
            # Mark task as done
            self.transcription_queue.task_done()

    async def _process_partial_transcript(
        self,
        whisper_result: WhisperResult,
        transcription_num: int,
        audio_length_sec: Optional[float] = None,
        task_start_time: float = 0.0,
    ) -> None:
        """Process partial transcription result."""
        try:
            # Check for hallucinations and skip if flag is enabled
            has_hallucination = any(
                segment.possible_hallucination for segment in whisper_result.segments
            )
            if has_hallucination and self.metadata.streaming_params.skip_partial_if_hallucination:
                log_stream_event(
                    self.stream_id,
                    "Possible hallucination detected, using previous partial",
                    {
                        "text": "".join([segment.text for segment in whisper_result.segments]),
                        "skip_partial_if_hallucination": self.metadata.streaming_params.skip_partial_if_hallucination,
                    },
                    "DEBUG",
                )
                # Use the previous partial if it exists
                if self.state.prev_whisper_result:
                    partial_result = self.state.get_partial_result()
                else:
                    # Hallucination for very first partial, just skip
                    return
            else:
                # Update prev whisper result and process normally (no hallucination, or hallucination with flag disabled)
                self.state.prev_whisper_result = whisper_result
                partial_result = self.state.get_partial_result()

            # Post process partial
            partial_result = post_process_whisper_result(
                partial_result, DEFAULT_STREAMING_POST_PROCESSING_FLAGS
            )

            # Observe partial latency
            self.state.observe_partial_latency()

            # Send partial result
            await self._send_result(
                partial_result,
                transcription_num,
                final=False,
                audio_length_sec=audio_length_sec,
                task_start_time=task_start_time,
            )

        except Exception as e:
            logger.error(f"❌ Stream {self.stream_id}: Failed to process partial result: {e}")
            await self.ws_manager.send_error_to_websocket(
                TranscriptionError(
                    self.stream_id, f"Partial transcription failed: {str(e)}", recoverable=True
                ),
            )

    async def _process_final_transcript(
        self,
        whisper_result: WhisperResult,
        transcription_num: int,
        end_audio: bool = False,
        task_start_time: float = 0.0,
        audio_length_sec: Optional[float] = None,
    ) -> None:
        """Process final transcription result."""

        # Check if diarization is enabled
        if (
            not self.state.metadata.whisper_params.enable_diarization
            or self.assignment_callback is None
        ):
            # No diarization, send result immediately
            whisper_result = post_process_whisper_result(
                whisper_result, DEFAULT_STREAMING_POST_PROCESSING_FLAGS
            )

            # Observe final latency
            self.state.observe_final_latency()

            await self._send_result(
                whisper_result,
                transcription_num,
                final=True,
                audio_length_sec=audio_length_sec,
                task_start_time=task_start_time,
                end_audio=end_audio,
            )

            return

        # Diarization enabled, process segments
        num_assignment_tasks = 0

        # Handle end_audio case (like old implementation)
        from whisper_utils.data_types import DiarizationGranularity

        if end_audio:
            log_stream_event(self.stream_id, "end_audio case", level="DEBUG")
            # if end_audio, flush current result
            # we know that all prior transcripts have been processed
            self.state.final_segments_buffer = whisper_result.segments
            num_assignment_tasks = 1
        # Handle different diarization modes
        elif (
            self.state.metadata.whisper_params.diarization_granularity
            == DiarizationGranularity.SENTENCE
        ):
            log_stream_event(self.stream_id, "sentence-level diarization case", level="DEBUG")
            num_assignment_tasks = self._process_sentence_level_segments(whisper_result)
        else:
            # Default segment-level assignment
            log_stream_event(self.stream_id, "segment-level diarization case", level="DEBUG")
            self.state.final_segments_buffer.extend(whisper_result.segments)
            num_assignment_tasks = 1

        # Log buffered transcription results (similar to model_old.py)
        log_stream_event(
            self.stream_id,
            "Buffered final transcription",
            {
                # "whisper_result_segments": whisper_result.segments,
                "final_segments_buffer": self.state.final_segments_buffer,
                # "partial_segment_buffer": self.state.partial_segments_buffer,
                # "num_assignment_tasks": num_assignment_tasks,
                # "transcription_num": transcription_num,
            },
            "DEBUG",
        )

        # Log detailed segment information
        if self.state.final_segments_buffer:
            segment_info = []
            for i, segment in enumerate(self.state.final_segments_buffer):
                segment_info.append(
                    {
                        "index": i,
                        "text": (
                            segment.text[:50] + "..." if len(segment.text) > 50 else segment.text
                        ),
                        "start_time": f"{segment.start_time:.3f}s",
                        "end_time": f"{segment.end_time:.3f}s",
                        "duration": f"{segment.end_time - segment.start_time:.3f}s",
                    }
                )

            log_stream_event(
                self.stream_id,
                "Final segments buffer details",
                {"segments": segment_info},
                "DEBUG",
            )

        # Queue assignment tasks
        if self.state.metadata.streaming_params.prioritize_latency:
            # Add to assignment queue
            for _ in range(num_assignment_tasks):
                if self.assignment_callback:
                    await self.assignment_callback(
                        transcription_num,
                        final=True,
                        task_start_time=task_start_time,
                        audio_length_sec=audio_length_sec,
                    )
        else:
            # Process assignment inline
            for _ in range(num_assignment_tasks):
                if self.assignment_callback:
                    await self.assignment_callback(
                        transcription_num,
                        final=True,
                        task_start_time=task_start_time,
                        audio_length_sec=audio_length_sec,
                    )

    def _process_sentence_level_segments(self, whisper_result: WhisperResult) -> int:
        """Process segments for sentence-level diarization."""
        log_stream_event(self.stream_id, "Processing sentence-level segments", level="DEBUG")

        # Merge with partial buffer
        self.state.partial_segments_buffer = merge_segments(
            self.state.partial_segments_buffer, *whisper_result.segments
        )

        # Clear previous partial since it has been added to partial segments
        self.state.prev_whisper_result = None

        num_assignment_tasks = 0

        # Check for punctuation
        if any(c in self.terminal_punctuation for c in self.state.partial_segments_buffer.text):
            from whisper_utils.data_types import Segment

            curr_word_buff = []

            def join_words_by_language_code(language_code: Optional[str], words: List[str]) -> str:
                if language_code and language_code in ["zh", "ja"]:
                    return "".join(words)
                else:
                    return " ".join(words)

            # Process each word
            for word in self.state.partial_segments_buffer.word_timestamps:
                # logger.info(f"{word=}")
                curr_word_buff.append(word)
                if word.word[-1].strip() in self.terminal_punctuation:
                    # logger.info(
                    #     f"Found terminal punctuation: {word.word[-1].strip()} from {word.word}"
                    # )
                    # Create sentence segment

                    sentence_segment = Segment(
                        start_time=curr_word_buff[0].start_time,
                        end_time=curr_word_buff[-1].start_time,
                        text=join_words_by_language_code(
                            self.state.partial_segments_buffer.language_code or "",
                            [w.word for w in curr_word_buff],
                        ),
                        word_timestamps=curr_word_buff,
                        language_code=self.state.partial_segments_buffer.language_code,
                    )

                    self.state.final_segments_buffer.append(sentence_segment)
                    num_assignment_tasks += 1
                    curr_word_buff = []

            # Handle remaining words
            if curr_word_buff:
                self.state.partial_segments_buffer = Segment(
                    start_time=curr_word_buff[0].start_time,
                    end_time=curr_word_buff[-1].start_time,
                    text=join_words_by_language_code(
                        self.state.partial_segments_buffer.language_code or "",
                        [w.word for w in curr_word_buff],
                    ),
                    word_timestamps=curr_word_buff,
                    language_code=self.state.partial_segments_buffer.language_code,
                )
            else:
                self.state.partial_segments_buffer = None

        return num_assignment_tasks

    async def _send_result(
        self,
        result: WhisperResult,
        transcription_num: int,
        final: bool,
        audio_length_sec: Optional[float] = None,
        task_start_time: float = 0.0,
        end_audio: bool = False,
    ) -> None:
        """Send result to WebSocket."""
        from whisper_utils.data_types import StreamingWhisperResult

        # Observe pipeline latency (from _handle_audio_chunk to send_json_safe)
        pipeline_latency = None
        if task_start_time > 0:
            pipeline_latency = time.time() - task_start_time
            if final:
                self.metrics["pipeline_final_latency_seconds_histogram"].labels(
                    diarized="false"
                ).observe(pipeline_latency)
            else:
                self.metrics["pipeline_partial_latency_seconds_histogram"].observe(pipeline_latency)

        include_pipeline_latency = self.metadata.streaming_params.include_pipeline_latency_metric

        streaming_result = StreamingWhisperResult(
            segments=result.segments,
            diarization=result.diarization,
            is_final=final,
            transcription_num=transcription_num,
            language_code=result.language_code,
            language_prob=result.language_prob,
            audio_length_sec=audio_length_sec,
            pipeline_latency=pipeline_latency if include_pipeline_latency else None,
            is_end_of_audio_flush=True if end_audio else None,
        )

        # Convert to dict and ensure JSON serializable
        result_dict = streaming_result.model_dump()
        # Only include is_end_of_audio_flush when True to avoid breaking existing clients
        if not result_dict.get("is_end_of_audio_flush"):
            result_dict.pop("is_end_of_audio_flush", None)
        result_dict["type"] = "transcription"
        result_dict = ensure_json_serializable(result_dict, self.stream_id)

        await self.ws_manager.send_json_safe(result_dict)

        log_stream_event(
            self.stream_id,
            f"{'Final' if final else 'Partial'} result sent",
            {"transcription_num": transcription_num, "segments_count": len(result.segments)},
            "DEBUG",
        )

    async def wait_for_completion(self) -> None:
        """Wait for all queued tasks to complete."""
        await self.transcription_queue.join()
        log_stream_event(self.stream_id, "All transcription tasks completed")

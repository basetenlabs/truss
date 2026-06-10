import asyncio
import logging
import time
from typing import Any, Dict, Optional

import pydantic
from fastapi import WebSocketDisconnect
from starlette.websockets import WebSocketDisconnect as StarletteWebSocketDisconnect
from streaming_utils.services.vad_service import VADService
from streaming_utils.utils.audio_utils import bytes_to_float_tensor
from streaming_utils.utils.constants import DEFAULT_CHUNK_SIZE
from streaming_utils.utils.error_utils import AudioProcessingError, StreamError, log_stream_event
from streaming_utils.utils.message_types import Message, MessageType
from streaming_utils.utils.websocket_utils import WebSocketManager
from websockets.exceptions import ConnectionClosedError
from whisper_utils.data_types import StreamingWhisperInput

from .audio_buffer import AudioBuffer
from .stream_state import StreamState
from .workers.stream_assignment_worker import StreamAssignmentWorker
from .workers.stream_diarization_worker import StreamDiarizationWorker
from .workers.stream_transcription_worker import StreamTranscriptionWorker

logger = logging.getLogger(__name__)


class StreamProcessor:
    """Main orchestrator managing the complete lifecycle of a single WebSocket stream."""

    def __init__(
        self,
        websocket: Any,  # WebSocketProtocol (chain) or FastAPIWebSocketAdapter (truss)
        stream_id: str,
        metadata: StreamingWhisperInput,
        metrics: Dict[str, Any],
        connections: Dict[str, int],
        diarization_chainlet=None,
    ):
        self.stream_id = stream_id
        self.websocket = websocket

        self.metadata = metadata
        self.metrics = metrics

        self.diarization_chainlet = diarization_chainlet

        self._apply_config_overrides(metadata)

        # Initialize WebSocket manager
        self.ws_manager = WebSocketManager(websocket, stream_id)

        # Initialize state
        self.state = StreamState(stream_id, metadata, metrics)

        self.connections = connections
        self.connections["active"] += 1
        self.connections["total"] += 1
        self.metrics["num_active_connections_gauge"].set(self.connections["active"])

        # Initialize audio buffer
        self.audio_buffer = AudioBuffer(
            stream_id=stream_id,
            sample_rate=metadata.streaming_params.sample_rate,
            max_duration=metadata.streaming_params.final_transcript_max_duration_s,
            bytes_per_sample=self.bytes_per_sample,
        )

        self.chunk_duration_s = DEFAULT_CHUNK_SIZE / metadata.streaming_params.sample_rate

        # Initialize VAD iterator
        self.vad_service = VADService.get_instance()
        self.vad_iterator = self.vad_service.create_vad_iterator(
            metadata.streaming_vad_config, stream_id
        )

        # Initialize workers with specific dependencies (avoiding circular imports)
        self.transcription_worker = StreamTranscriptionWorker(
            stream_id=self.stream_id,
            state=self.state,
            metrics=self.metrics,
            ws_manager=self.ws_manager,
            metadata=self.metadata,
            assignment_callback=self._handle_assignment_task if self.diarization_chainlet else None,
        )

        if self.diarization_chainlet:
            self.diarization_worker = StreamDiarizationWorker(
                stream_id=self.stream_id,
                state=self.state,
                metrics=self.metrics,
                diarization_chainlet=self.diarization_chainlet,
                ws_manager=self.ws_manager,
            )
            self.assignment_worker = StreamAssignmentWorker(
                stream_id=self.stream_id,
                state=self.state,
                metrics=self.metrics,
                ws_manager=self.ws_manager,
            )

        # Processing state
        self.is_processing = False
        self.processing_start_time: Optional[float] = None

        log_stream_event(
            stream_id,
            "StreamProcessor initialized",
        )

    def _apply_config_overrides(self, metadata: StreamingWhisperInput) -> None:
        """Apply config overrides from model_old.py to ensure compatibility."""
        # Validate diarization parameter always disable if there is no diarization
        if not self.diarization_chainlet:
            metadata.whisper_params.enable_diarization = False

        # Whisper params validation and overrides
        from whisper_utils.data_types import DiarizationGranularity

        # keep for backward compatibility
        if (
            metadata.whisper_params.enable_diarization
            and metadata.whisper_params.enable_word_level_diarization
        ):
            logger.warning(
                f"Detected deprecated diarization config (enable_word_level_diarization), setting diarization granularity to WORD"
            )
            metadata.whisper_params.diarization_granularity = DiarizationGranularity.WORD

        # keep for backward compatibility
        if (
            metadata.whisper_params.enable_diarization
            and metadata.whisper_params.enable_sentence_level_diarization
        ):
            logger.warning(
                f"Detected deprecated diarization config (enable_sentence_level_diarization), setting diarization granularity to SENTENCE"
            )
            metadata.whisper_params.diarization_granularity = DiarizationGranularity.SENTENCE

        if (
            metadata.whisper_params.enable_diarization
            and metadata.whisper_params.diarization_granularity
            in (DiarizationGranularity.WORD, DiarizationGranularity.SENTENCE)
            and not metadata.whisper_params.show_word_timestamps
        ):
            logger.warning(
                f"Setting `show_word_timestamps` to `true` to use `{metadata.whisper_params.diarization_granularity.value}` level diarization"
            )
            metadata.whisper_params.show_word_timestamps = True

        # Streaming params validation and overrides
        if (
            metadata.streaming_params.final_transcript_max_duration_s > 30
        ):  # TODO: decouple max duration from Whisper/VAD max
            logger.warning(
                f"Stream {self.stream_id}: Reducing final_transcript_max_duration_s from {metadata.streaming_params.final_transcript_max_duration_s} to 30"
            )
            metadata.streaming_params.final_transcript_max_duration_s = 30

        # Calculate bytes per sample for audio processing
        encoding = metadata.streaming_params.encoding
        self.bytes_per_sample = (
            2 if encoding == "pcm_s16le" else 1 if encoding == "pcm_mulaw" else 2
        )

        log_stream_event(
            self.stream_id,
            "🟢 Final metadata after overrides",
            details={"metadata": metadata},
            level="DEBUG",
        )

    async def _handle_assignment_task(
        self,
        transcription_num: int,
        final: bool,
        task_start_time: float = 0.0,
        audio_length_sec: Optional[float] = None,
    ) -> None:
        """Callback method for transcription worker to queue assignment tasks."""
        if (
            self.metadata.streaming_params.prioritize_latency
        ):  # Todo check if this aligns with the original logic of prioritize_latency
            await self.assignment_worker.queue_assignment_task(
                transcription_num, final, task_start_time, audio_length_sec
            )
        else:
            await self.assignment_worker.process_assignment_task(
                transcription_num, final, task_start_time, audio_length_sec
            )

    def log_queue_status(self) -> None:
        """Log current queue sizes for all workers."""
        queue_status = {
            "transcription_queue_size": self.transcription_worker.transcription_queue.qsize(),
            "diarization_queue_size": self.diarization_worker.diarization_queue.qsize()
            if self.diarization_chainlet
            else None,
            "assignment_queue_size": self.assignment_worker.assignment_queue.qsize()
            if self.diarization_chainlet
            else None,
            "audio_buffer_size_bytes": len(self.audio_buffer.get_buffer()),
            "audio_buffer_duration": f"{len(self.audio_buffer.get_buffer()) / (self.metadata.streaming_params.sample_rate * self.bytes_per_sample):.3f}s",
        }

        log_stream_event(self.stream_id, "Queue status", queue_status, "DEBUG")

    async def start(self) -> None:
        """Start all workers and begin processing."""
        if self.is_processing:
            logger.warning(f"⚠️ StreamProcessor for stream {self.stream_id} is already processing")
            return

        try:

            self.is_processing = True
            self.processing_start_time = time.time()

            log_stream_event(
                self.stream_id,
                "Starting StreamProcessor",
                {"processing_start_time": self.processing_start_time},
            )

            # Start all workers
            if self.diarization_chainlet:
                await asyncio.gather(
                    self.transcription_worker.start(),
                    self.diarization_worker.start(),
                    self.assignment_worker.start(),
                    self.process_audio_loop(),
                )
            else:
                await asyncio.gather(
                    self.transcription_worker.start(),
                    self.process_audio_loop(),
                )

        except Exception as e:
            self.is_processing = False
            logger.error(f"❌ Failed to start StreamProcessor for stream {self.stream_id}: {e}")
            await self.ws_manager.send_error_to_websocket(
                StreamError(
                    self.stream_id,
                    f"Failed to start StreamProcessor: {str(e)}",
                    recoverable=False,
                ),
            )
            raise
        finally:
            await self.cleanup()

    async def process_audio_loop(self) -> None:
        """Main audio consumption loop with comprehensive error handling."""
        log_stream_event(self.stream_id, "Audio processing loop started")

        try:
            while self.is_processing:
                message = await self.websocket.receive()
                if isinstance(message, bytes):
                    audio_chunk = message
                    await self._handle_audio_chunk(audio_chunk)
                elif isinstance(message, str):
                    await self._handle_text_message(message)
                else:
                    logger.warning(f"Unhandled message type received: {message}")
        except (WebSocketDisconnect, StarletteWebSocketDisconnect, ConnectionClosedError) as e:
            self.ws_manager.mark_closed()  # Mark connection as closed
            logger.info(
                f"Client disconnected during audio processing loop for stream {self.stream_id}: {e}"
            )
        except Exception as e:
            await self.ws_manager.send_error_to_websocket(
                AudioProcessingError(self.stream_id, f"Audio processing loop failed: {e}")
            )

    async def _handle_text_message(self, message: str) -> None:
        """Handle incoming text message."""
        log_stream_event(self.stream_id, "Received text message", {"message": message}, "INFO")
        try:
            special_message = Message.model_validate_json(message)
        except pydantic.ValidationError as e:
            # not fatal
            await self.ws_manager.send_error_to_websocket(
                StreamError(
                    self.stream_id,
                    f"Failed to validate text message: {str(e)}",
                    recoverable=True,
                ),
            )
            logger.error(f"❌ Stream {self.stream_id}: Failed to validate text message: {e}")
            return

        if special_message.type == MessageType.END_AUDIO:
            asyncio.create_task(self._handle_end_of_audio())
            response = Message(
                type=MessageType.END_AUDIO,
                trace_id=special_message.trace_id,
                body={"status": "acknowledged"},
            )  # maybe send stats in body
        elif special_message.type == MessageType.HEALTH_CHECK:
            healthy = self._health_check()
            body: Dict[str, Any] = {}
            body["healthy"] = healthy
            if special_message.body and special_message.body.get("get_partial_interval"):
                body["partial_interval"] = self.state.most_recent_partial_latency
            time_now = time.time()
            response = Message(
                type=MessageType.HEALTH_CHECK,
                trace_id=special_message.trace_id,
                body=body,
                timestamp=time_now,
            )
            if special_message.timestamp:
                self.metrics["health_check_latency_seconds_histogram"].observe(
                    time_now - special_message.timestamp
                )
        else:
            logger.warning(
                f"❌ Stream {self.stream_id}: Unhandled text message type: {special_message.type}"
            )
            response = Message(
                type=MessageType.ERROR,
                trace_id=special_message.trace_id,
                body={"error": f"Unhandled message type: {special_message.type}"},
            )
        await self.ws_manager.send_json_safe(response.model_dump())

    async def _handle_audio_chunk(self, audio_chunk: bytes) -> None:
        """Handle incoming audio chunk with comprehensive processing."""
        start_time = time.time()

        # Increment audio processed seconds counter
        self.metrics["audio_processed_seconds_counter"].inc(self.chunk_duration_s)

        # Convert to tensor
        audio_tensor = bytes_to_float_tensor(
            audio_chunk, self.metadata.streaming_params.encoding, self.stream_id
        )

        # Log audio chunk details (similar to model_old.py)
        log_stream_event(
            self.stream_id,
            "Processing audio chunk",
            {
                "chunk_size_bytes": len(audio_chunk),
                "tensor_shape": audio_tensor.shape,
                "tensor_duration": f"{audio_tensor.shape[0] / self.metadata.streaming_params.sample_rate:.3f}s",
                "time_offset": f"{self.state.time_offset:.3f}s",
            },
            "DEBUG",
        )

        # Push to diarization pipeline (non-blocking)
        if self.diarization_chainlet:
            await self.diarization_worker.process_new_audio_chunk(
                audio_tensor, self.state.time_offset
            )

        # Check if adding this chunk would exceed max duration (like old implementation)
        # Calculate what the duration would be after adding this chunk
        current_buffer = self.audio_buffer.get_buffer()
        new_size = len(current_buffer) + len(audio_chunk)
        duration_sec = new_size / (
            self.metadata.streaming_params.sample_rate * self.bytes_per_sample
        )
        would_overflow = (
            duration_sec > self.metadata.streaming_params.final_transcript_max_duration_s
        )

        should_process = False

        if would_overflow:
            log_stream_event(
                self.stream_id,
                "Buffer would overflow, processing current buffer first",
                {
                    "current_buffer_duration": self.audio_buffer.duration_seconds(),
                    "new_chunk_size": len(audio_chunk),
                    "would_be_duration": duration_sec,
                    "max_duration": self.metadata.streaming_params.final_transcript_max_duration_s,
                    "vad_triggered": self.vad_iterator.triggered,
                },
                "DEBUG",
            )

            if self.vad_iterator.triggered:
                # Speech detected in buffer, process normally
                await self._trigger_final_transcription(task_start_time=start_time)
            else:
                # No speech detected — skip transcription to avoid Whisper hallucinating on silence
                log_stream_event(
                    self.stream_id,
                    "Buffer overflow with no speech detected, skipping transcription",
                    {
                        "buffer_duration": self.audio_buffer.duration_seconds(),
                    },
                    "INFO",
                )
                self._skip_buffer_without_transcription()

            # Now add the new chunk to start a fresh buffer
            self.audio_buffer.add_chunk(audio_chunk)

            # Update state for the new buffer
            new_duration_sec = len(audio_chunk) / (
                self.metadata.streaming_params.sample_rate * self.bytes_per_sample
            )
            self.state.update_total_duration(self.state.time_offset + new_duration_sec)

        else:
            # Normal case: add chunk to buffer
            should_process = self.audio_buffer.add_chunk(audio_chunk)

            # Update state
            new_size = len(self.audio_buffer.get_buffer())
            duration_sec = new_size / (
                self.metadata.streaming_params.sample_rate * self.bytes_per_sample
            )
            self.state.update_total_duration(self.state.time_offset + duration_sec)

            # Check for processing triggers
            await self._check_processing_triggers(audio_tensor, should_process, start_time)

        processing_time = time.time() - start_time
        # Update memory metrics periodically
        self.state.update_memory_metrics()

        # Periodic cleanup of old diarization segments to prevent memory leak
        # Clean up segments older than 30 seconds
        # segments_removed = self.state.cleanup_old_diarization_segments(
        #     current_time=self.state.total_duration_sec,
        #     buffer_seconds=30.0
        # )

        # if segments_removed > 0:
        #     log_stream_event(
        #         self.stream_id,
        #         "Periodic diarization cleanup completed",
        #         {
        #             "segments_removed": segments_removed,
        #             "current_duration": self.state.total_duration_sec,
        #         },
        #         "DEBUG",
        #     )

        log_stream_event(
            self.stream_id,
            "Audio chunk processed",
            {
                "chunk_size_bytes": len(audio_chunk),
                "duration_sec": duration_sec,
                "would_overflow": would_overflow,
                "should_process": should_process,
                "processing_time": processing_time,
                "total_buffer_duration": f"{self.audio_buffer.duration_seconds():.3f}s",
                "audio_buffer_size_bytes": len(self.audio_buffer.get_buffer()),
                "transcription_queue_size": self.transcription_worker.transcription_queue.qsize(),
                "diarization_queue_size": self.diarization_worker.diarization_queue.qsize()
                if self.diarization_chainlet
                else None,
                "assignment_queue_size": self.assignment_worker.assignment_queue.qsize()
                if self.diarization_chainlet
                else None,
                "final_segments_buffer_size": len(self.state.final_segments_buffer),
                "diarization_coverage_start": self.state.diarization_coverage_start,
                "diarization_coverage_end": self.state.diarization_coverage_end,
            },
            "DEBUG",
        )

    async def _check_processing_triggers(
        self, audio_tensor, should_process: bool = False, task_start_time: float = 0.0
    ) -> None:
        """Check various triggers for transcription processing."""
        vad_audio_tensor = audio_tensor.to(self.vad_service.vad_model_device)

        vad_result = self.vad_iterator(vad_audio_tensor, return_seconds=True)

        # Check for VAD speech end
        if vad_result is not None:
            log_stream_event(
                self.stream_id,
                "VAD detected speech end",
                {"vad_result": vad_result},
                level="DEBUG",
            )
            await self._trigger_final_transcription(task_start_time=task_start_time)

        # Check for buffer overflow (from audio buffer)
        elif should_process:
            if self.vad_iterator.triggered:
                log_stream_event(
                    self.stream_id, "Buffer overflow detected, triggering final transcription"
                )
                await self._trigger_final_transcription(task_start_time=task_start_time)
            else:
                log_stream_event(
                    self.stream_id,
                    "Buffer overflow with no speech detected, skipping transcription",
                    {"buffer_duration": self.audio_buffer.duration_seconds()},
                    "INFO",
                )
                self._skip_buffer_without_transcription()

        # Check for partial transcript interval
        elif self.metadata.streaming_params.enable_partial_transcripts:
            audio_buffer_delta = (
                self.audio_buffer.duration_seconds() - self.state.last_transcription_end_sec
            )

            if (
                audio_buffer_delta >= self.metadata.streaming_params.partial_transcript_interval_s
                and self.vad_iterator.triggered
            ):
                log_stream_event(
                    self.stream_id,
                    "Partial transcript interval reached",
                    details={
                        "audio_buffer.duration_seconds()": self.audio_buffer.duration_seconds(),
                        "last_transcription_end_sec": self.state.last_transcription_end_sec,
                        "audio_buffer_delta": audio_buffer_delta,
                        "time_offset": self.state.time_offset,
                    },
                    level="DEBUG",
                )
                await self._trigger_partial_transcription(task_start_time=task_start_time)

    def _skip_buffer_without_transcription(self) -> None:
        """Clear the audio buffer and advance state without queuing transcription.

        Used when buffer overflows but VAD has not detected any speech,
        to avoid sending silent audio to Whisper (which causes hallucinations).
        """
        audio_length_sec = self.audio_buffer.duration_seconds()
        self.state.update_time_offset(self.state.time_offset + audio_length_sec)
        self.audio_buffer.clear()
        self.state.reset_for_new_segment()

    async def _trigger_final_transcription(
        self, end_audio: bool = False, task_start_time: float = 0.0
    ) -> None:
        """Trigger final transcription processing."""
        audio_buffer = self.audio_buffer.get_buffer()
        audio_length_sec = self.audio_buffer.duration_seconds()

        # Queue transcription task
        await self.transcription_worker.queue_transcription_task(
            audio_buffer=audio_buffer,
            metadata=self.metadata,
            time_offset=self.state.time_offset,
            curr_end_sec=self.state.total_duration_sec,
            transcription_num=self.state.increment_transcription_num(),
            final=True,
            prefix=self.state.prefix,
            end_audio=end_audio,
            task_start_time=task_start_time,
            audio_length_sec=audio_length_sec,
        )

        # Update state
        self.state.update_time_offset(
            self.state.time_offset
            + len(audio_buffer)
            / (self.metadata.streaming_params.sample_rate * self.bytes_per_sample)
        )
        self.audio_buffer.clear()
        self.state.reset_for_new_segment()

    async def _trigger_partial_transcription(self, task_start_time: float = 0.0) -> None:
        """Trigger partial transcription processing."""

        if self.transcription_worker.transcription_queue.qsize() > 10:
            logger.warning(
                f"⚠️ Stream {self.stream_id}: Transcription queue is full, dropping partial transcription"
            )
            self.metrics["num_dropped_partial_transcripts_counter"].inc()
            return

        audio_buffer = self.audio_buffer.get_buffer()
        audio_length_sec = self.audio_buffer.duration_seconds()

        # Queue transcription task
        await self.transcription_worker.queue_transcription_task(
            audio_buffer=audio_buffer,
            metadata=self.metadata,
            time_offset=self.state.time_offset,
            curr_end_sec=self.state.total_duration_sec,
            transcription_num=self.state.transcription_num,
            final=False,
            prefix=self.state.prefix,
            task_start_time=task_start_time,
            audio_length_sec=audio_length_sec,
        )

        # Update state
        self.state.last_transcription_end_sec = self.audio_buffer.duration_seconds()

        log_stream_event(
            self.stream_id,
            "Partial transcription triggered",
            details={
                "last_transcription_end_sec": self.state.last_transcription_end_sec,
                "audio_buffer_duration": self.audio_buffer.duration_seconds(),
                "time_offset": self.state.time_offset,
            },
            level="DEBUG",
        )

    async def _handle_end_of_audio(self) -> None:
        """Handle special 1-byte messages (e.g., end-of-stream)."""
        try:
            log_stream_event(self.stream_id, "Received end_of_audio message")

            # Wait for all transcription tasks to finish
            await self.transcription_worker.wait_for_completion()

            # Process final diarization if enabled
            if self.diarization_chainlet:
                await self.diarization_worker.process_final_diarization()
                await self.diarization_worker.wait_for_completion()

            # Trigger final transcription with all remaining audio
            audio_buffer = self.audio_buffer.get_buffer()
            if audio_buffer:
                if self.vad_iterator.triggered:
                    # Speech was detected in buffer, flush it
                    end_audio_task_start = time.time()
                    await self._trigger_final_transcription(
                        end_audio=True, task_start_time=end_audio_task_start
                    )
                    await self.transcription_worker.wait_for_completion()
                else:
                    # No speech in remaining buffer, skip to avoid hallucination
                    log_stream_event(
                        self.stream_id,
                        "end_audio: skipping final flush, no speech detected in remaining buffer",
                        {"buffer_duration": self.audio_buffer.duration_seconds()},
                        "INFO",
                    )
                    self._skip_buffer_without_transcription()

            if self.diarization_chainlet:
                await self.assignment_worker.wait_for_completion()

            log_stream_event(self.stream_id, "End_of_audio processing completed")
            await self.ws_manager.send_json_safe(
                Message(
                    type=MessageType.END_AUDIO, trace_id=None, body={"status": "finished"}
                ).model_dump()
            )
            self.is_processing = False

        except Exception as e:
            await self.ws_manager.send_error_to_websocket(
                StreamError(
                    self.stream_id,
                    f"Failed to handle end_of_audio: {str(e)}",
                    recoverable=False,
                ),
            )
            logger.error(f"❌ Stream {self.stream_id}: Failed to handle end_of_audio: {e}")
            raise

    def _health_check(self) -> bool:
        """Check if the stream is healthy."""
        transcription_alive = self.transcription_worker.is_running
        diarization_alive = (
            self.diarization_worker.is_running if self.diarization_chainlet else True
        )
        assignment_alive = self.assignment_worker.is_running if self.diarization_chainlet else True
        return transcription_alive and diarization_alive and assignment_alive

    async def cleanup(self) -> None:
        """Clean up all resources for this stream."""

        # Immediately stop processing to prevent new tasks from being queued
        self.is_processing = False

        # Update connections metrics
        self.connections["active"] -= 1
        self.metrics["num_active_connections_gauge"].set(self.connections["active"])

        # Stop all workers immediately to prevent them from processing more tasks
        stop_tasks = [self.transcription_worker.stop()]
        if self.diarization_chainlet:
            stop_tasks.append(self.diarization_worker.stop())
            stop_tasks.append(self.assignment_worker.stop())
        await asyncio.gather(*stop_tasks, return_exceptions=True)

        # Clean up StreamState to prevent memory leaks
        self.state.cleanup()

        log_stream_event(
            self.stream_id,
            "🧹 StreamProcessor cleanup completed",
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive stream statistics for monitoring."""
        runtime = time.time() - self.processing_start_time if self.processing_start_time else 0.0

        return {
            "stream_id": self.stream_id,
            "is_processing": self.is_processing,
            "total_runtime": runtime,
            "state_stats": self.state.get_stats(),
            "audio_buffer_stats": self.audio_buffer.get_stats(),
            "transcription_worker_stats": self.transcription_worker.get_stats(),
            "diarization_worker_stats": self.diarization_worker.get_stats()
            if self.diarization_chainlet
            else None,
            "assignment_worker_stats": self.assignment_worker.get_stats()
            if self.diarization_chainlet
            else None,
            "websocket_stats": self.ws_manager.get_connection_stats(),
        }

    def log_stats(self) -> None:
        """Log comprehensive stream statistics."""
        stats = self.get_stats()
        log_stream_event(self.stream_id, "StreamProcessor statistics", stats, "DEBUG")

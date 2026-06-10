import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import torch
from diart.diarization import SpeakerDiarization, SpeakerDiarizationConfig
from streaming_utils.utils.constants import (
    DIARIZATION_INTERVAL,
    DIARIZATION_PATCH_COLLAR,
    DIARIZATION_WORKER_TYPE,
)
from streaming_utils.utils.error_utils import (
    DiarizationError,
    log_performance_metric,
    log_stream_event,
)
from streaming_utils.utils.websocket_utils import WebSocketManager
from whisper_chain_utils.whisper_chain_data_types import DiarizationInput

from ..stream_state import StreamState
from .base_worker import BaseWorker

logger = logging.getLogger(__name__)


class StreamDiarizationWorker(BaseWorker):
    """Handles diarization tasks for a specific stream."""

    def __init__(
        self,
        stream_id: str,
        state: StreamState,
        metrics: Dict[str, Any],
        ws_manager: WebSocketManager,
        diarization_chainlet=None,
        sample_rate: int = 16000,
        duration_s: float = 5.0,
        step_s: float = 0.5,
        diarization_interval_s: float = DIARIZATION_INTERVAL,
    ):
        super().__init__(stream_id, DIARIZATION_WORKER_TYPE, ws_manager)
        self.state = state
        self.sample_rate = sample_rate
        self.duration_s = duration_s
        self.step_s = step_s
        self.diarization_interval_s = diarization_interval_s
        self.diarization_chainlet = diarization_chainlet

        # Queue for diarization tasks
        self.diarization_queue = asyncio.Queue()

        self.metrics = metrics

        # Thread pool executor for parallel diarization processing (like old implementation)
        self.diarizer_executor = ThreadPoolExecutor(max_workers=100)

        # Audio buffer for accumulating chunks until we have enough for diarization
        self.buffer_start_time = 0.0
        self.step_samples = int(self.step_s * self.sample_rate)
        self.required_samples = int(self.duration_s * self.sample_rate)
        self.num_samples_threshold = int(self.sample_rate * self.diarization_interval_s)

        self.max_buffer_size = int(self.diarization_interval_s * self.sample_rate * 2)

        self.audio_buffer = torch.zeros(self.max_buffer_size, dtype=torch.float32)
        self.buffer_fill_size = 0

        # create pipeline
        pipeline_config = SpeakerDiarizationConfig(
            latency="max",
            step=self.step_s,
            duration=self.duration_s,
        )

        self.pipeline = SpeakerDiarization(pipeline_config)

        log_stream_event(self.stream_id, "StreamDiarizationWorker initialized")

    async def process_new_audio_chunk(self, audio_tensor: torch.Tensor, time_offset: float) -> None:
        """Queue audio chunk for diarization processing."""
        if not self.diarization_chainlet:
            return

        # Initialize buffer start time if this is the first chunk
        if self.buffer_fill_size == 0:
            self.buffer_start_time = time_offset

        # Add audio to buffer
        self.audio_buffer[
            self.buffer_fill_size : self.buffer_fill_size + audio_tensor.shape[0]
        ] = audio_tensor
        self.buffer_fill_size += audio_tensor.shape[0]

        if self.buffer_fill_size >= self.num_samples_threshold:
            logger.debug(
                f"[Diarization] Enough audio for diarization. {self.audio_buffer.shape[0]} >= {self.required_samples}"
            )
            await self._queue_diarization_tasks()
        else:
            logger.debug(
                f"[Diarization] Not enough audio for diarization. {self.audio_buffer.shape[0]} < {self.required_samples}"
            )

    async def _queue_diarization_tasks(self, end_audio: bool = False) -> None:
        """Extract fixed-size sliding windows from the audio buffer and queue them for diarization."""
        if not self.diarization_chainlet:
            return
        # wait for enough windows to be available
        if self.buffer_fill_size < self.num_samples_threshold and not end_audio:
            return

        # if end_audio, pad to multiple of `step`
        if end_audio:
            padding_size = (
                self.step_samples
                - (self.buffer_fill_size % self.step_samples)
                + self.required_samples
            )
            padding = torch.zeros(padding_size, dtype=torch.float32)
            to_diarize = torch.cat([self.audio_buffer[: self.buffer_fill_size], padding])
        else:
            # take off as much as we can as long as it's a multiple of `step`
            leftover_size = self.buffer_fill_size % self.step_samples
            if leftover_size == 0:
                to_diarize = self.audio_buffer[: self.buffer_fill_size]
            else:
                to_diarize = self.audio_buffer[: self.buffer_fill_size - leftover_size]

        assert to_diarize.shape[0] % self.step_samples == 0

        # queue audio
        task = {
            "audio_tensor": to_diarize,
            "time_offset": self.buffer_start_time,
            "end_audio": end_audio,
        }
        await self.diarization_queue.put(task)

        # increment buffer start time
        self.buffer_start_time += (to_diarize.shape[0] - self.required_samples) / self.sample_rate
        self.buffer_fill_size -= to_diarize.shape[0] - self.required_samples
        self.audio_buffer = torch.cat(
            [
                self.audio_buffer[to_diarize.shape[0] - self.required_samples :],
                torch.zeros(to_diarize.shape[0] - self.required_samples),
            ]
        )

        self.metrics["diarization_queue_gauge"].set(self.diarization_queue.qsize())

    async def _get_task_with_timeout(self) -> Optional[Dict[str, Any]]:
        """Get task from queue with timeout."""
        try:
            # logger.info(f"[Diarization] Getting task from the queue. Queue size: {self.diarization_queue.qsize()}")
            return await asyncio.wait_for(self.diarization_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    async def _process_task(self, task: Dict[str, Any]) -> None:
        """Process a single diarization task."""
        from pyannote.core import SlidingWindow, SlidingWindowFeature

        if not self.diarization_chainlet:
            return

        start_time = time.time()

        audio_tensor = task["audio_tensor"]
        time_offset = task["time_offset"]
        end_audio = task["end_audio"]

        try:
            diarization_result = await self.diarization_chainlet.run_remote(
                DiarizationInput(audio_wav=audio_tensor.numpy(), time_offset=time_offset)
            )

            # create sliding windows
            waveforms = []
            for i in range(0, audio_tensor.shape[0] - self.required_samples, self.step_samples):
                sliding_window = SlidingWindow(
                    start=time_offset + i / self.sample_rate,
                    duration=1 / self.sample_rate,
                    step=1 / self.sample_rate,
                )
                waveform_data = audio_tensor[i : i + self.required_samples].reshape(-1, 1)
                waveform_data = SlidingWindowFeature(waveform_data, sliding_window)
                waveforms.append(waveform_data)

            results = self.pipeline(waveforms, diarization_result)

            # Process and accumulate results
            segments_found = 0
            for annotation, window in results:
                log_stream_event(
                    self.stream_id,
                    "Processing diarization annotation",
                    {
                        "annotation_type": type(annotation).__name__,
                        "annotation_length": len(annotation),
                        "window_extent": str(window.extent) if hasattr(window, "extent") else "N/A",
                    },
                    "DEBUG",
                )

                if len(annotation) > 0:
                    # Accumulate results and apply patch collar
                    self.state.accumulated_diarization.update(annotation)
                    # Apply support with patch collar to merge adjacent segments from same speaker
                    self.state.accumulated_diarization = self.state.accumulated_diarization.support(
                        DIARIZATION_PATCH_COLLAR
                    )
                    segments_found += len(annotation)

                    # Update memory metrics after diarization accumulation
                    self.state.update_memory_metrics()

                    # Log segments for debugging
                    for segment, _, speaker in annotation.itertracks(
                        yield_label=True
                    ):  # pyright: ignore[reportAssignmentType]
                        log_stream_event(
                            self.stream_id,
                            "Found diarization segment",
                            {
                                "speaker": speaker,
                                "start_time": segment.start,
                                "end_time": segment.end,
                                "duration": segment.end - segment.start,
                            },
                            "DEBUG",
                        )

            # Update state with results
            self.state.total_diarization_windows += 1

            # Update coverage tracking efficiently using the accumulated annotation (like old implementation)
            if len(self.state.accumulated_diarization) > 0:
                all_segments = list(self.state.accumulated_diarization.itertracks(yield_label=True))
                if all_segments:
                    segment_starts = [
                        segment.start for segment, _, _ in all_segments
                    ]  # pyright: ignore[reportAssignmentType]
                    segment_ends = [
                        segment.end for segment, _, _ in all_segments
                    ]  # pyright: ignore[reportAssignmentType]

                    # Update coverage range
                    self.state.diarization_coverage_start = min(segment_starts)
                    self.state.diarization_coverage_end = (
                        max(segment_ends) if not end_audio else float("inf")
                    )

            # Log accumulated diarization results (similar to model_old.py)
            if len(self.state.accumulated_diarization) > 0:
                all_segments = list(self.state.accumulated_diarization.itertracks(yield_label=True))
                segment_info = []
                for segment, _, speaker in all_segments:  # pyright: ignore[reportAssignmentType]
                    segment_info.append(
                        {
                            "speaker": speaker,
                            "start": f"{segment.start:.3f}s",
                            "end": f"{segment.end:.3f}s",
                            "duration": f"{segment.end - segment.start:.3f}s",
                        }
                    )

                log_stream_event(
                    self.stream_id,
                    "Accumulated diarization results",
                    {
                        "total_segments": len(all_segments),
                        "coverage_start": f"{self.state.diarization_coverage_start:.3f}s",
                        "coverage_end": f"{self.state.diarization_coverage_end:.3f}s",
                        "segments": segment_info,
                    },
                    "DEBUG",
                )

            processing_time = time.time() - start_time
            # logger.info(f"🟢 Diarization task time: {processing_time:.3f}s")
            log_performance_metric(
                self.stream_id, "diarization_task", processing_time, success=True
            )

        except Exception as e:
            processing_time = time.time() - start_time
            import traceback

            log_performance_metric(
                self.stream_id, "diarization_task", processing_time, success=False
            )
            raise DiarizationError(
                self.stream_id, f"Diarization task failed: {str(e)} {traceback.format_exc()}"
            )
        finally:
            # Mark task as done
            self.diarization_queue.task_done()

    async def process_final_diarization(self) -> None:
        """Process final diarization for the complete audio."""
        if not self.diarization_chainlet:
            return

        # Process any remaining audio in buffer with end_audio=True
        await self._queue_diarization_tasks(end_audio=True)

    async def wait_for_completion(self) -> None:
        """Wait for all queued tasks to complete."""
        try:
            await self.diarization_queue.join()
            log_stream_event(self.stream_id, "All diarization tasks completed")
        except Exception as e:
            logger.error(
                f"❌ Stream {self.stream_id}: Error waiting for diarization completion: {e}"
            )
            raise
        finally:
            # Clean up thread pool executor
            if hasattr(self, "diarizer_executor"):
                self.diarizer_executor.shutdown(wait=True)
                log_stream_event(self.stream_id, "🧹 Diarization thread pool executor was shut down")

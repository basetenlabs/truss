"""
WhisperService - Truss Implementation (Local Model Calls)

This is the truss-specific implementation that calls a locally loaded
WhisperRT model directly, eliminating network overhead.

Inherits all shared logic from WhisperServiceBase:
- Audio preprocessing
- Parameter handling
- Timestamp adjustment
- Logging and error handling
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import torch
from streaming_utils.utils.error_utils import TranscriptionError
from streaming_utils.whisper_service_base import WhisperServiceBase
from whisper_utils.data_types import MicroChunkInfo, WhisperParams, WhisperResult
from whisper_utils.utilities import call_whisper_model

logger = logging.getLogger(__name__)


class WhisperServiceTruss(WhisperServiceBase):
    """
    Truss implementation of WhisperService.

    Uses a locally loaded WhisperRT model for inference.
    All processing happens on the same GPU pod, eliminating network latency.
    """

    def __init__(self):
        super().__init__()
        self.whisper_model = None

    async def initialize(
        self,
        whisper_model,
        executor: ThreadPoolExecutor,
    ) -> None:
        """
        Initialize with local Whisper model and executor.

        Args:
            whisper_model: The loaded WhisperRT model instance
            executor: ThreadPoolExecutor for running inference
        """
        async with self._lock:
            if self.is_initialized:
                logger.info("🔄 WhisperService already initialized, skipping")
                return

            start_time = time.time()

            try:
                self.whisper_model = whisper_model
                self.executor = executor
                self.is_initialized = True

                duration = time.time() - start_time
                logger.info(f"✅ WhisperService (Truss) initialized successfully in {duration:.3f}s")

            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"❌ WhisperService initialization failed after {duration:.3f}s: {e}")
                raise TranscriptionError(
                    "global", f"WhisperService initialization failed: {str(e)}", recoverable=False
                )

    async def _do_inference(
        self,
        seg_info: MicroChunkInfo,
        waveform: torch.Tensor,
        whisper_params: WhisperParams,
        stream_id: str,
    ) -> WhisperResult:
        """
        Perform inference via local model call.

        This calls call_whisper_model() directly with the locally loaded
        WhisperRT model, eliminating network overhead.
        """
        whisper_result = await call_whisper_model(
            whisper_model=self.whisper_model,
            thread_pool_executor=self.executor,
            seg_info=seg_info,
            audio_chunk=waveform.squeeze(0).numpy(),
            fallback_chunks=[],
            whisper_params=whisper_params,
            asr_options={},
        )
        return whisper_result

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics for monitoring."""
        stats = super().get_stats()
        stats["mode"] = "truss"
        stats["model_loaded"] = self.whisper_model is not None
        return stats

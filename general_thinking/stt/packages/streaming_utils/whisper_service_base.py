"""
WhisperService Base Class - Shared Logic for Chain and Truss

This module provides the base class for WhisperService with all shared logic.
Chain and Truss implementations inherit from this and only override:
1. initialize() - Different dependencies (chainlet vs local model)
2. _do_inference() - Different inference mechanism (run_remote vs local call)

This eliminates ~180 lines of code duplication between implementations.

Location: streaming-utils/streaming_utils/ (shared between chain and truss)
- Chain implementation: asr-chains/streaming-asr-chain/packages/services/whisper_service_chain.py
- Truss implementation: asr-truss/streaming-asr-truss/model/truss_overrides/services/whisper_service_truss.py
"""

import asyncio
import copy
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import numpy as np
import torch
from streaming_utils.utils.error_utils import (
    TranscriptionError,
    log_performance_metric,
    log_stream_event,
)
from whisper_utils.constants import WHISPER_SAMPLE_RATE
from whisper_utils.data_types import MicroChunkInfo, WhisperParams, WhisperResult

logger = logging.getLogger(__name__)


class WhisperServiceBase(ABC):
    """
    Abstract base class for WhisperService.

    Provides all shared logic for audio preprocessing, parameter handling,
    timestamp adjustment, logging, and error handling.

    Singleton Pattern:
    - Subclasses call get_instance() to create/retrieve their singleton
    - The instance is also registered on the base class for generic access
    - Workers can call WhisperServiceBase.get_instance() to get the
      registered implementation without knowing the concrete class

    Subclasses must implement:
    - initialize(): Set up the inference mechanism
    - _do_inference(): Perform the actual Whisper inference
    """

    _instance: Optional["WhisperServiceBase"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.is_initialized: bool = False
        self.executor: Optional[ThreadPoolExecutor] = None

    @classmethod
    def get_instance(cls) -> "WhisperServiceBase":
        """
        Get singleton instance of WhisperService.

        When called on a concrete subclass (e.g., WhisperServiceChain.get_instance()),
        creates an instance of that subclass if none exists.

        When called on WhisperServiceBase directly, returns whatever instance
        was previously registered by a subclass. This allows workers to get
        the WhisperService without knowing whether it's chain or truss.

        Raises:
            RuntimeError: If called on base class before any subclass is initialized.
        """
        if cls is WhisperServiceBase:
            # Called on base class - return the registered instance
            if WhisperServiceBase._instance is None:
                raise RuntimeError(
                    "No WhisperService initialized. "
                    "Call get_instance() on a concrete subclass first "
                    "(e.g., WhisperServiceChain or WhisperServiceTruss)."
                )
            return WhisperServiceBase._instance

        # Called on a concrete subclass
        if cls._instance is None:
            instance = cls()
            cls._instance = instance
            # Also register on the base class for generic access
            WhisperServiceBase._instance = instance
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        cls._instance = None
        WhisperServiceBase._instance = None

    @abstractmethod
    async def initialize(self, *args, **kwargs) -> None:
        """
        Initialize the service with required dependencies.

        Chain implementation: Takes whisper_chainlet
        Truss implementation: Takes whisper_model and executor
        """
        pass

    @abstractmethod
    async def _do_inference(
        self,
        seg_info: MicroChunkInfo,
        waveform: torch.Tensor,
        whisper_params: WhisperParams,
        stream_id: str,
    ) -> WhisperResult:
        """
        Perform the actual Whisper inference.

        Chain implementation: Calls whisper_chainlet.run_remote()
        Truss implementation: Calls call_whisper_model() locally

        Args:
            seg_info: Chunk timing information
            waveform: Audio tensor
            whisper_params: Whisper parameters
            stream_id: Stream identifier for logging

        Returns:
            WhisperResult from inference
        """
        pass

    async def transcribe(
        self,
        audio_data: bytes,
        params: WhisperParams,
        time_offset: float = 0.0,
        prefix: str = "",
        final: bool = False,
        stream_id: str = "unknown",
    ) -> WhisperResult:
        """
        Process transcription request with comprehensive logging and error handling.

        This method contains all shared logic:
        1. Audio bytes → waveform conversion
        2. MicroChunkInfo creation
        3. Parameter adjustment for partial vs final
        4. Calls abstract _do_inference()
        5. Timestamp adjustment
        6. Logging and error handling

        Args:
            audio_data: Audio bytes to transcribe
            params: Whisper parameters
            time_offset: Time offset for timestamp adjustment
            prefix: Prefix for transcription
            final: Whether this is a final transcription
            stream_id: Stream identifier for logging

        Returns:
            WhisperResult: Transcription result

        Raises:
            TranscriptionError: If transcription fails
        """
        if not self.is_initialized:
            raise TranscriptionError(stream_id, "WhisperService not initialized", recoverable=False)

        start_time = time.time()

        try:
            # ============================================================
            # SHARED: Convert bytes to waveform
            # ============================================================
            int16_array = np.frombuffer(audio_data, dtype=np.int16)
            float_array = int16_array.astype(np.float32) / 32768.0
            waveform = torch.from_numpy(float_array)

            # ============================================================
            # SHARED: Create MicroChunkInfo with time offset
            # ============================================================
            seg_info = MicroChunkInfo(
                chunk_index=-1,
                start_time_sec=time_offset,
                end_time_sec=time_offset + len(float_array) / WHISPER_SAMPLE_RATE,
            )

            # ============================================================
            # SHARED: Handle missing chunk retry warning
            # ============================================================
            if params.use_missing_chunk_retry:
                params.use_missing_chunk_retry = False
                logger.warning(
                    f"⚠️ Stream {stream_id}: use_missing_chunk_retry is not supported for streaming mode"
                )

            # ============================================================
            # SHARED: Adjust parameters for partial vs final
            # ============================================================
            if not final:
                whisper_params = copy.deepcopy(params)
                whisper_params.whisper_sampling_params.sampling_temperatures = [0.0]
                whisper_params.prefix = prefix
            else:
                params.prefix = ""
                whisper_params = params

            # ============================================================
            # ABSTRACT: Call implementation-specific inference
            # ============================================================
            whisper_result = await self._do_inference(
                seg_info=seg_info,
                waveform=waveform,
                whisper_params=whisper_params,
                stream_id=stream_id,
            )

            # ============================================================
            # SHARED: Adjust timestamps in the result
            # ============================================================
            for segment in whisper_result.segments:
                segment.start_time += time_offset
                segment.end_time += time_offset

                # Also adjust word timestamps to maintain absolute timing
                for word in segment.word_timestamps:
                    word.start_time += time_offset
                    word.end_time += time_offset

            # ============================================================
            # SHARED: Logging
            # ============================================================
            processing_time = time.time() - start_time
            log_stream_event(
                stream_id,
                "Transcription completed successfully",
                {
                    "segments_count": len(whisper_result.segments),
                    "processing_time": processing_time,
                    "total_segments_text": " ".join([seg.text for seg in whisper_result.segments]),
                },
                "DEBUG",
            )

            return whisper_result

        except Exception as e:
            processing_time = time.time() - start_time
            log_performance_metric(stream_id, "transcription", processing_time, success=False)
            raise TranscriptionError(stream_id, f"Transcription failed: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics for monitoring."""
        return {
            "is_initialized": self.is_initialized,
            "executor_workers": self.executor._max_workers if self.executor else 0,
        }

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
            logger.info("🧹 WhisperService thread executor was shut down")

        logger.info(f"📊 WhisperService final stats: {self.get_stats()}")

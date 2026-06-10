"""
WhisperService - Chain Implementation (Remote Chainlet Calls)

This is the chain-specific implementation that calls a remote Whisper chainlet
via run_remote() for inference.

Inherits all shared logic from WhisperServiceBase:
- Audio preprocessing
- Parameter handling
- Timestamp adjustment
- Logging and error handling
"""

import asyncio
import logging
import time
from typing import Any, Dict

import chainlets.whisper_chainlet
import torch
from streaming_utils.utils.error_utils import TranscriptionError
from streaming_utils.whisper_service_base import WhisperServiceBase
from whisper_chain_utils.whisper_chain_data_types import WhisperChainletInput
from whisper_utils.data_types import MicroChunkInfo, WhisperParams, WhisperResult

logger = logging.getLogger(__name__)


class WhisperServiceChain(WhisperServiceBase):
    """
    Chain implementation of WhisperService.

    Uses a remote Whisper chainlet for inference via run_remote().
    The chainlet runs on a separate GPU pod.
    """

    def __init__(self):
        super().__init__()
        self.whisper_chainlet = None

    async def initialize(
        self,
        whisper_chainlet: chainlets.whisper_chainlet.Transcriber,
    ) -> None:
        """
        Initialize with remote Whisper chainlet.

        Args:
            whisper_chainlet: The remote chainlet for Whisper inference
        """
        async with self._lock:
            if self.is_initialized:
                logger.info("🔄 WhisperService already initialized, skipping")
                return

            start_time = time.time()

            try:
                self.whisper_chainlet = whisper_chainlet
                self.is_initialized = True

                duration = time.time() - start_time
                logger.info(f"✅ WhisperService (Chain) initialized successfully in {duration:.3f}s")

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
        Perform inference via remote chainlet call.

        This calls whisper_chainlet.run_remote() which sends the audio
        to a separate GPU pod for transcription.
        """
        whisper_chainlet_input = WhisperChainletInput(
            seg_info=seg_info,
            audio_wav=waveform.squeeze(0).numpy(),
            asr_options={},
            fallback_chunks=[],
            whisper_params=whisper_params,
        )

        whisper_result = await self.whisper_chainlet.run_remote(whisper_chainlet_input)
        return whisper_result

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics for monitoring."""
        stats = super().get_stats()
        stats["mode"] = "chain"
        stats["chainlet_connected"] = self.whisper_chainlet is not None
        return stats

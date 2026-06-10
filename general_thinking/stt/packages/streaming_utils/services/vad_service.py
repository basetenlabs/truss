import logging
import time
from typing import Any, Dict, Optional

import torch
from silero_vad import load_cpu_model, load_gpu_model
from silero_vad.utils_vad import BatchSettings, StreamingVADIterator
from streaming_utils.utils.constants import VAD_LOOKBACK_WINDOW_SECS as VAD_LOOKBACK
from streaming_utils.utils.error_utils import log_stream_event
from whisper_utils.constants import VAD_LOOKBACK_WINDOW_SECS

logger = logging.getLogger(__name__)


class VADService:
    """Singleton service managing VAD model and iterator creation."""

    _instance = None

    def __init__(self):
        self.vad_model_device = "cpu"  # cpu or cuda
        self.vad_model = None
        self.batch_settings = None
        self.is_initialized = False
        self.total_iterators_created = 0

        # logger.info("✅ VADService instance created")

    @classmethod
    def get_instance(cls) -> "VADService":
        """Get singleton instance of VADService."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(self) -> None:
        """Initialize VAD model and resources."""
        if self.is_initialized:
            logger.info("⏭️ VADService already initialized, skipping")
            return

        start_time = time.time()

        try:
            # Load VAD model
            if self.vad_model_device == "cuda":
                self.vad_model = load_gpu_model(model_type="onnx")
            elif self.vad_model_device == "cpu":
                self.vad_model = load_cpu_model(model_type="onnx")
            else:
                raise ValueError(f"Invalid VAD model device: {self.vad_model_device}")
            # logger.info(f"✅ Loaded VAD model on {self.vad_model_device}")

            # Create batch settings
            self.batch_settings = BatchSettings(lookback_window_secs=VAD_LOOKBACK)

            self.is_initialized = True
            duration = time.time() - start_time
            logger.info(f"✅ VADService initialized successfully in {duration:.3f}s")

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ VADService initialization failed after {duration:.3f}s: {e}")
            raise RuntimeError(f"VADService initialization failed: {str(e)}")

    def create_vad_iterator(
        self, config: Dict[str, Any], stream_id: str = "unknown"
    ) -> StreamingVADIterator:
        """
        Create VAD iterator with configuration and logging.

        Args:
            config: VAD configuration dictionary
            stream_id: Stream identifier for logging

        Returns:
            StreamingVADIterator: Configured VAD iterator

        Raises:
            RuntimeError: If VADService is not initialized
        """
        if not self.is_initialized:
            raise RuntimeError("VADService not initialized")

        start_time = time.time()

        try:
            log_stream_event(
                stream_id,
                "Creating VAD iterator",
                {"config": config, "batch_settings": str(self.batch_settings)},
                "DEBUG",
            )

            # Create iterator
            vad_iterator = StreamingVADIterator(id=stream_id, model=self.vad_model, **config)

            self.total_iterators_created += 1
            duration = time.time() - start_time

            log_stream_event(
                stream_id,
                "VAD iterator created successfully",
                {
                    "creation_time": duration,
                    "total_iterators_created": self.total_iterators_created,
                    "config": str(config),
                },
                "INFO",
            )

            return vad_iterator

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"❌ Stream {stream_id}: VAD iterator creation failed after {duration:.3f}s: {e}"
            )
            raise RuntimeError(f"VAD iterator creation failed for stream {stream_id}: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics for monitoring."""
        return {
            "is_initialized": self.is_initialized,
            "total_iterators_created": self.total_iterators_created,
            "batch_settings": str(self.batch_settings) if self.batch_settings else None,
            "vad_model_loaded": self.vad_model is not None,
            "vad_model_device": self.vad_model_device,
        }

    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info(f"📊 VADService final stats: {self.get_stats()}")

"""
Streaming ASR Truss - Whisper Large V3

This is the truss implementation of the streaming ASR chain.
It combines the WebSocket entry point with the local Whisper model,
eliminating the need for remote chainlet calls.

Key differences from the chain implementation:
1. WhisperRT model is loaded locally in this truss
2. WhisperServiceTruss uses local call_whisper_model() instead of run_remote()
3. Single GPU pod handles both orchestration and inference
4. FastAPIWebSocketAdapter bridges FastAPI WebSocket to WebSocketProtocol interface

Code Sharing Strategy:
- Most code is shared via streaming-utils/streaming_utils/ (external_package_dirs)
- StreamProcessor, workers, services, utils are all in streaming_utils
- WhisperServiceBase in streaming_utils provides the shared base class
- WhisperServiceTruss in truss_overrides/ provides the local model implementation
- FastAPIWebSocketAdapter in streaming_utils bridges the WebSocket interface gap
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Add model directory to path so 'truss_overrides' can be imported
_MODEL_DIR = Path(__file__).resolve().parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))


class Model:
    """
    Streaming ASR Truss Model.

    Handles WebSocket connections for real-time speech-to-text transcription
    with optional speaker diarization.
    """

    def __init__(self, **kwargs):
        self._secrets = kwargs.get("secrets", {})
        self.whisper_model = None
        self.vad_model = None
        self.executor = None
        self.connections = {
            "total": 0,
            "active": 0,
        }

        # Create metrics in __init__ to avoid duplicate registration on load() retry
        from streaming_utils.utilities import create_streaming_metrics

        self.metrics = create_streaming_metrics()

    def load(self):
        """Load models and initialize services."""
        from whisper_runtime import WhisperRT
        from whisper_utils.constants import DEFAULT_WHISPER_MODEL
        from whisper_utils.utilities import get_max_batch_size

        start_time = time.time()
        logger.info("Loading Streaming ASR Truss models...")

        # Get HuggingFace token — prefer baseten_hf_access_token (for internal
        # baseten-admin engine repos), fall back to hf_access_token for standard
        # customer deployments.
        hf_token = None
        for secret_name in ("baseten_hf_access_token", "hf_access_token"):
            try:
                hf_token = self._secrets.get(secret_name)
                if hf_token:
                    break
            except Exception:
                pass

        # Get max batch size based on GPU
        max_batch_size = get_max_batch_size()
        logger.info(f"Loading model with max batch size: {max_batch_size}")

        model_repo_id_override = os.environ.get("WHISPER_ENGINE_REPO_ID")
        if model_repo_id_override:
            logger.info(f"Using custom whisper engine repo: {model_repo_id_override}")

        # Load Whisper model (WhisperRT - TensorRT-LLM based)
        self.whisper_model = WhisperRT(
            DEFAULT_WHISPER_MODEL.value,
            max_batch_size=max_batch_size,
            hf_token=hf_token,
            model_repo_id_override=model_repo_id_override,
        )
        logger.info("✅ WhisperRT model loaded")

        # Warmup the model
        self.whisper_model.warmup()
        logger.info("✅ WhisperRT model warmed up")

        # Create thread pool executor for inference
        self.executor = ThreadPoolExecutor(max_workers=max_batch_size)

        # Load VAD model
        try:
            from silero_vad import load_cpu_model

            logger.info("Loading VAD model...")
            self.vad_model = load_cpu_model(model_type="onnx")
            logger.info("✅ VAD model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading VAD model: {str(e)}")
            raise

        # Initialize the WhisperServiceTruss singleton with local model
        from truss_overrides.services.whisper_service_truss import WhisperServiceTruss

        whisper_service = WhisperServiceTruss.get_instance()

        # Run async initialization in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                whisper_service.initialize(
                    whisper_model=self.whisper_model,
                    executor=self.executor,
                )
            )
        finally:
            loop.close()

        # Initialize VAD service (shared from streaming_utils)
        from streaming_utils.services.vad_service import VADService

        vad_service = VADService.get_instance()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(vad_service.initialize())
        finally:
            loop.close()

        duration = time.time() - start_time
        logger.info(f"✅ Model.load() completed successfully in {duration:.3f}s")
        logger.info(f"✅ WhisperService stats: {whisper_service.get_stats()}")

    async def _receive_metadata(self, ws, stream_id: str):
        """Receive and validate metadata from client."""
        # Shared error_utils from streaming_utils
        from streaming_utils.utils.error_utils import WebSocketError, log_stream_event
        from whisper_utils.data_types import StreamingWhisperInput

        start_time = time.time()

        try:
            log_stream_event(stream_id, "Receiving metadata")

            # Receive metadata JSON
            metadata_json = await ws.receive_text()
            metadata_receive_time = time.time()

            log_stream_event(
                stream_id,
                "Metadata received",
                {"receive_time": metadata_receive_time - start_time},
                "DEBUG",
            )

            # Parse and validate metadata
            metadata = StreamingWhisperInput.model_validate(json.loads(metadata_json))
            validation_time = time.time()

            log_stream_event(
                stream_id,
                "Metadata validation completed",
                {
                    "validation_time": validation_time - metadata_receive_time,
                    "total_time": validation_time - start_time,
                },
                "DEBUG",
            )

            log_stream_event(
                stream_id,
                "Metadata processing completed",
                {"final_metadata": str(metadata)},
                "DEBUG",
            )

            return metadata

        except json.JSONDecodeError as e:
            raise WebSocketError(stream_id, f"Invalid JSON in metadata: {str(e)}")
        except Exception as e:
            raise WebSocketError(stream_id, f"Metadata processing failed: {str(e)}")

    async def websocket(self, ws):
        """
        Handle WebSocket connection for streaming ASR.

        This is the main entry point for WebSocket connections.
        It receives audio chunks, performs VAD, transcription, and
        optionally diarization, sending results back to the client.

        Uses FastAPIWebSocketAdapter to bridge FastAPI WebSocket to the
        WebSocketProtocol interface expected by StreamProcessor.
        """
        import fastapi

        # Shared modules from streaming_utils
        from streaming_utils.stream_processing.stream_processor import StreamProcessor
        from streaming_utils.utils.error_utils import log_stream_event

        # Adapter to bridge FastAPI WebSocket → WebSocketProtocol interface
        from streaming_utils.websocket_adapter import FastAPIWebSocketAdapter

        logger.info("New WebSocket connection received")

        stream_id = str(uuid.uuid4())
        stream_processor = None

        try:
            log_stream_event(
                stream_id,
                "New WebSocket connection",
                {},
            )

            # Wrap FastAPI WebSocket with adapter before passing to StreamProcessor
            ws_adapter = FastAPIWebSocketAdapter(ws)

            metadata = await self._receive_metadata(ws_adapter, stream_id)

            stream_processor = StreamProcessor(
                ws_adapter,
                stream_id,
                metadata,
                self.metrics,
                self.connections,
                diarization_chainlet=None,  # No diarization chainlet in truss (for now)
            )

            await stream_processor.start()
        except fastapi.WebSocketDisconnect:
            log_stream_event(stream_id, "WebSocket disconnected")
        except Exception as e:
            logger.error(f"❌ Error in websocket handler: {e}")
            raise

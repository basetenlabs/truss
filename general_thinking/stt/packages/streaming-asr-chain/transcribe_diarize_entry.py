import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path

# Path setup - MUST be before chainlet imports
_BASETEN_WHISPER_ROOT = Path(__file__).resolve().parent.parent.parent
_WHISPER_UTILS_LIB = _BASETEN_WHISPER_ROOT / "whisper-utils"
_STREAMING_UTILS_LIB = _BASETEN_WHISPER_ROOT / "streaming-utils"
_LOCAL_SILERO_VAD_LIB = _BASETEN_WHISPER_ROOT / "silero-vad" / "src"
_ASR_CHAINS_LIB = _BASETEN_WHISPER_ROOT / "asr-chains"
sys.path.append(str(_WHISPER_UTILS_LIB))
sys.path.append(str(_STREAMING_UTILS_LIB))
sys.path.append(str(_ASR_CHAINS_LIB))

import chainlets.diarizer_chainlet
import chainlets.whisper_chainlet
import truss_chains as chains
from truss import truss_config
from whisper_utils.constants import CONCURRENCY_LIMIT_FROM_CHUNKER_CHAINLET

# Configure the logging
logger = logging.getLogger(__name__)


@chains.mark_entrypoint("Streaming ASR and Diarization")
class WebSocketEntry(chains.ChainletBase):

    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            apt_requirements=["ffmpeg", "libsox-dev", "sox", "wget"],
            pip_requirements=[
                "fastapi==0.116.1",
                "pandas==2.2.3",
                "torch==2.7.1",
                "torchaudio==2.7.1",
                "ffmpeg-python==0.2.0",
                "numpy==1.26.4",
                "httpx==0.27.2",
                "regex==2025.7.34",
                "onnxruntime==1.22.0",
                "einops>=0.3.0",
                "pyannote.core>=4.5",
            ],
            external_package_dirs=[
                chains.make_abs_path_here(str(_LOCAL_SILERO_VAD_LIB)),
                chains.make_abs_path_here(str(_WHISPER_UTILS_LIB)),
                chains.make_abs_path_here(str(_STREAMING_UTILS_LIB)),
                chains.make_abs_path_here(str(_ASR_CHAINS_LIB)),
            ],
        ),
        compute=chains.Compute(
            memory="1Gi",
            cpu_count=1,
            predict_concurrency=20,
        ),
        options=chains.ChainletOptions(
            enable_debug_logs=False,
            enable_b10_tracing=False,
            transport=truss_config.WebsocketOptions(
                kind="websocket", ping_timeout_seconds=24 * 60 * 60
            ),
        ),
        assets=chains.Assets(secret_keys=["hf_access_token"]),
    )
    _context: chains.DeploymentContext

    _whisper: chainlets.whisper_chainlet.Transcriber
    _diarizer: chainlets.diarizer_chainlet.Diarizer

    def __init__(
        self,
        whisper: chainlets.whisper_chainlet.Transcriber = chains.depends(
            chainlets.whisper_chainlet.Transcriber,
            retries=3,
            use_binary=True,
            concurrency_limit=CONCURRENCY_LIMIT_FROM_CHUNKER_CHAINLET,
        ),
        diarizer: chainlets.diarizer_chainlet.Diarizer = chains.depends(
            chainlets.diarizer_chainlet.Diarizer,
            retries=3,
            use_binary=True,
            concurrency_limit=1000,
        ),
        context: chains.DeploymentContext = chains.depends_context(),
    ) -> None:
        """Initialize all global services."""
        # Store parameters without importing dependencies
        self._context = context
        self._whisper = whisper
        self._diarizer = diarizer

        from packages.services.whisper_service_chain import WhisperServiceChain as WhisperService
        from streaming_utils.services.vad_service import VADService
        from streaming_utils.stream_processing.stream_processor import StreamProcessor
        from streaming_utils.utilities import create_streaming_metrics

        start_time = time.time()

        self.whisper_service = WhisperService.get_instance()
        self.vad_service = VADService.get_instance()

        self.connections = {
            "total": 0,
            "active": 0,
        }

        self.metrics = create_streaming_metrics()

        # Initialize services synchronously
        # Note: We need to run async initialization in a new event loop
        # since this is called during sync initialization
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(
                asyncio.gather(
                    self.whisper_service.initialize(self._whisper),
                    self.vad_service.initialize(),
                    return_exceptions=True,
                )
            )
        finally:
            loop.close()

        duration = time.time() - start_time
        logger.info(f"✅ Model.load() completed successfully in {duration:.3f}s")

        # Log service statistics
        logger.info(f"✅ WhisperService stats: {self.whisper_service.get_stats()}")
        logger.info(f"✅ VADService stats: {self.vad_service.get_stats()}")

    async def call_whisper_detect_language(self, audio_wav, whisper_params, asr_options: dict):
        from whisper_chain_utils.whisper_chain_data_types import WhisperChainletInput
        from whisper_utils.data_types import WhisperResult

        whisper_params = whisper_params.model_copy(deep=True)
        whisper_params.language_detection_only = True
        whisper_chainlet_input = WhisperChainletInput(
            audio_wav=audio_wav.numpy(),
            whisper_params=whisper_params,
            asr_options=asr_options,
        )
        whisper_transcribe_result = await self._whisper.run_remote(whisper_chainlet_input)
        return whisper_transcribe_result

    async def _receive_metadata(self, ws: chains.WebSocketProtocol, stream_id: str):
        """Receive and validate metadata from client."""
        import fastapi
        from starlette.websockets import WebSocketDisconnect as StarletteWebSocketDisconnect
        from streaming_utils.utils.error_utils import WebSocketError, log_stream_event
        from websockets.exceptions import ConnectionClosedError
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

            if metadata.stream_id:

                def is_valid_uuid(uuid_str: str) -> bool:
                    try:
                        uuid.UUID(uuid_str)
                        return True
                    except ValueError:
                        return False

                if is_valid_uuid(metadata.stream_id):
                    log_stream_event(
                        stream_id,
                        f"Switching to client provided stream_id: {metadata.stream_id} from original stream_id: {stream_id}",
                        {"stream_id": metadata.stream_id},
                        "INFO",
                    )
                    stream_id = metadata.stream_id

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

            return metadata, stream_id

        except (fastapi.WebSocketDisconnect, StarletteWebSocketDisconnect, ConnectionClosedError):
            # Let disconnection exceptions propagate - they're handled by run_remote
            raise
        except json.JSONDecodeError as e:
            raise WebSocketError(stream_id, f"Invalid JSON in metadata: {str(e)}")
        except Exception as e:
            raise WebSocketError(stream_id, f"Metadata processing failed: {str(e)}")

    async def call_whisper_transcribe(
        self,
        audio_wav,
        whisper_input,
        fallback_chunks,
        seg_info,
        audio_language: str,
    ):
        from whisper_chain_utils.whisper_chain_data_types import WhisperChainletInput

        whisper_chainlet_input = WhisperChainletInput(
            seg_info=seg_info,
            audio_wav=audio_wav.numpy(),
            asr_options=whisper_input.asr_options,
            fallback_chunks=fallback_chunks,
            whisper_params=whisper_input.whisper_params,
        )
        whisper_chainlet_input.whisper_params.audio_language = audio_language
        whisper_transcribe_result = await self._whisper.run_remote(whisper_chainlet_input)

        return whisper_transcribe_result

    async def run_remote(
        self,
        ws: chains.WebSocketProtocol,
    ) -> None:
        import fastapi
        from starlette.websockets import WebSocketDisconnect as StarletteWebSocketDisconnect
        from streaming_utils.stream_processing.stream_processor import StreamProcessor
        from streaming_utils.utils.error_utils import log_stream_event
        from websockets.exceptions import ConnectionClosedError

        logger.info("Running remote")

        stream_id = str(uuid.uuid4())

        stream_processor = None

        try:
            log_stream_event(
                stream_id,
                "New WebSocket connection",
                {},
            )
            metadata, stream_id = await self._receive_metadata(ws, stream_id)

            stream_processor = StreamProcessor(
                ws,
                stream_id,
                metadata,
                self.metrics or {},
                self.connections or {},
                self._diarizer,
            )

            await stream_processor.start()
        except (
            fastapi.WebSocketDisconnect,
            StarletteWebSocketDisconnect,
            ConnectionClosedError,
        ) as e:
            # Client disconnected - log disconnection info
            log_stream_event(stream_id, f"WebSocket disconnected: {e}")
        except Exception as e:
            logger.error(f"❌ Error in run_remote: {e}")
            raise

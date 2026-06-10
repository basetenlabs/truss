"""Baseten speech-to-text service implementation with verbose logging."""

import asyncio
import json
import time
from typing import AsyncGenerator, Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Baseten STT, you need to `pip install websockets`.")
    raise Exception(f"Missing module: {e}")


class BasetenSTTService(WebsocketSTTService):
    """Baseten real-time speech-to-text service.

    Connects to the Baseten streaming STT WebSocket, sends raw int16 PCM
    audio, and receives partial/final JSON transcription messages.
    """

    DEFAULT_METADATA: dict = {
        "whisper_params": {
            "audio_language": "auto",
        },
        "streaming_params": {
            "enable_partial_transcripts": True,
            "partial_transcript_interval_s": 0.3,
            "final_transcript_max_duration_s": 30,
        },
        "streaming_vad_config": {
            "threshold": 0.4,
            "speech_pad_ms": 30,
            "min_silence_duration_ms": 100,
        },
        "include_timing_info": True,
    }

    def _set_setting(self, key: str, value):
        settings = self._settings
        if isinstance(settings, dict):
            settings[key] = value
        else:
            setattr(settings, key, value)

    def __init__(
        self,
        *,
        api_key: str,
        url: str,
        model: str = "baseten-stt",
        language: Optional[str] = None,
        metadata: Optional[dict] = None,
        sample_rate: int = 16000,
        ttfs_p99_latency: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._metadata = metadata or self.DEFAULT_METADATA
        resolved_language = language
        if resolved_language is None:
            resolved_language = (
                self._metadata.get("whisper_params", {}).get("audio_language")
                if isinstance(self._metadata, dict)
                else None
            )

        # Silero VAD requires exactly 512 samples per chunk at 16kHz (1024 bytes).
        # Pipecat sends smaller frames (e.g. 320 samples / 640 bytes at 20ms),
        # so we buffer and re-chunk before forwarding.
        self._vad_chunk_bytes = 512 * 2  # 512 int16 samples = 1024 bytes
        self._audio_buffer = bytearray()
        self._audio_buffer_offset = 0

        self._connected = False
        self._receive_task = None
        self._connect_task = None
        self._audio_send_count = 0
        self._connect_lock = asyncio.Lock()
        self._ws_ready = asyncio.Event()
        self._warn_count = 0

        # VAD-gated hallucination filter: only emit transcriptions when
        # the user was speaking recently (prevents Whisper silence hallucinations).
        self._user_speaking = False
        self._last_speech_end = 0.0
        self._speech_grace_s = 5.0

        if hasattr(self, "set_model_name"):
            self.set_model_name(model)
        else:
            self._model_name = model
        self._set_setting("model", model)
        self._set_setting("language", resolved_language)

        logger.info(f"[BaseTen STT] Initialized — url={url}")

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        logger.info("[BaseTen STT] start() called")
        await super().start(frame)
        # Connect in the background so the StartFrame keeps propagating through
        # the rest of the pipeline (TTS warmup, output transport) instead of
        # stalling ~2s on the STT WebSocket handshake. run_stt() awaits
        # readiness via _ensure_connected() before sending audio.
        self._connect_task = asyncio.ensure_future(self._connect())

    async def stop(self, frame: EndFrame):
        logger.info("[BaseTen STT] stop() called")
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        logger.info("[BaseTen STT] cancel() called")
        await super().cancel(frame)
        await self._disconnect()

    async def _ensure_connected(self):
        """Wait for the initial connection or attempt a reconnect."""
        if self._websocket and self._websocket.state is State.OPEN:
            return True

        if not self._ws_ready.is_set():
            try:
                await asyncio.wait_for(self._ws_ready.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("[BaseTen STT] Timed out waiting for initial connection")

        if self._websocket and self._websocket.state is State.OPEN:
            return True

        async with self._connect_lock:
            if self._websocket and self._websocket.state is State.OPEN:
                return True
            logger.info("[BaseTen STT] Attempting reconnect from run_stt")
            try:
                await self._connect_websocket()
                if self._websocket and not self._receive_task:
                    self._receive_task = self.create_task(
                        self._receive_task_handler(self._report_error)
                    )
                return self._websocket and self._websocket.state is State.OPEN
            except Exception as e:
                logger.error(f"[BaseTen STT] Reconnect failed: {e}")
                return False

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Buffer incoming audio and send in 512-sample chunks for Silero VAD."""
        self._audio_buffer.extend(audio)

        if not (self._websocket and self._websocket.state is State.OPEN):
            await self._ensure_connected()

        if self._websocket and self._websocket.state is State.OPEN:
            available = len(self._audio_buffer) - self._audio_buffer_offset
            while available >= self._vad_chunk_bytes:
                chunk = bytes(
                    self._audio_buffer[
                        self._audio_buffer_offset : self._audio_buffer_offset + self._vad_chunk_bytes
                    ]
                )
                self._audio_buffer_offset += self._vad_chunk_bytes
                available -= self._vad_chunk_bytes

                self._audio_send_count += 1
                if self._audio_send_count == 1:
                    logger.info(
                        f"[BaseTen STT] Sending first audio chunk ({len(chunk)} bytes)"
                    )
                elif self._audio_send_count % 200 == 0:
                    logger.info(
                        f"[BaseTen STT] Audio chunks sent: {self._audio_send_count}"
                    )
                await self._websocket.send(chunk)

            if self._audio_buffer_offset > 16384:
                del self._audio_buffer[: self._audio_buffer_offset]
                self._audio_buffer_offset = 0
        else:
            self._warn_count += 1
            if self._warn_count == 1 or self._warn_count % 500 == 0:
                ws_state = self._websocket.state if self._websocket else "no websocket"
                logger.warning(
                    f"[BaseTen STT] Cannot send audio — ws state: {ws_state} "
                    f"(dropped {self._warn_count} frames)"
                )
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._user_speaking = True
            logger.info("[BaseTen STT] VAD: user started speaking")
            await self.start_processing_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_speaking = False
            self._last_speech_end = time.monotonic()
            logger.info("[BaseTen STT] VAD: user stopped speaking")

    # -- Connection lifecycle ------------------------------------------------

    def _build_metadata(self) -> dict:
        return self._metadata

    async def _connect(self):
        logger.info("[BaseTen STT] _connect() called")
        await super()._connect()
        async with self._connect_lock:
            await self._connect_websocket()
            if self._websocket and not self._receive_task:
                self._receive_task = self.create_task(
                    self._receive_task_handler(self._report_error)
                )
                logger.info("[BaseTen STT] Receive task started")

    async def _disconnect(self):
        logger.info("[BaseTen STT] _disconnect() called")
        await super()._disconnect()

        if self._connect_task and not self._connect_task.done():
            self._connect_task.cancel()
            try:
                await self._connect_task
            except (asyncio.CancelledError, Exception):
                pass
        self._connect_task = None

        try:
            if self._receive_task:
                await self.cancel_task(self._receive_task)
                self._receive_task = None
        finally:
            if self._websocket:
                await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.info("[BaseTen STT] WebSocket already open, skipping connect")
                return

            logger.info(f"[BaseTen STT] Connecting to {self._url}")

            headers = {"Authorization": f"Api-Key {self._api_key}"}
            self._websocket = await websocket_connect(
                self._url,
                additional_headers=headers,
            )
            logger.info("[BaseTen STT] WebSocket connected successfully")

            # Send the metadata handshake
            metadata = self._build_metadata()
            await self._websocket.send(json.dumps(metadata))
            logger.info(f"[BaseTen STT] Sent metadata: {json.dumps(metadata)}")

            self._connected = True
            self._audio_send_count = 0
            self._warn_count = 0
            self._audio_buffer.clear()
            self._audio_buffer_offset = 0
            self._ws_ready.set()
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._connected = False
            logger.error(f"[BaseTen STT] Connection failed: {e}")
            await self.push_error(error_msg=f"Unable to connect to Baseten STT: {e}", exception=e)
            raise

    async def _disconnect_websocket(self):
        try:
            if self._websocket:
                logger.info("[BaseTen STT] Closing WebSocket")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"[BaseTen STT] Error closing websocket: {e}")
        finally:
            self._websocket = None
            self._connected = False
            self._ws_ready.clear()
            await self._call_event_handler("on_disconnected")
            logger.info("[BaseTen STT] Disconnected")

    # -- Message handling ----------------------------------------------------

    async def _receive_messages(self):
        logger.info("[BaseTen STT] _receive_messages() loop started")
        msg_count = 0
        async for message in self._websocket:
            msg_count += 1
            try:
                data = json.loads(message)
                logger.debug(
                    f"[BaseTen STT] Received message #{msg_count}: {json.dumps(data)[:300]}"
                )
                await self._handle_message(data)
            except json.JSONDecodeError:
                logger.warning(
                    f"[BaseTen STT] Non-JSON message #{msg_count}: {str(message)[:200]}"
                )
        logger.warning("[BaseTen STT] _receive_messages() loop ended (server closed?)")

    def _speech_recent(self) -> bool:
        """True if the user is speaking or stopped within the grace period."""
        if self._user_speaking:
            return True
        if self._last_speech_end == 0.0:
            return False
        return (time.monotonic() - self._last_speech_end) < self._speech_grace_s

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        if is_final and not self._speech_recent():
            logger.warning(
                f"[BaseTen STT] Dropping likely hallucination (no recent VAD): '{transcript}'"
            )
            return

        if is_final:
            await self.push_frame(
                TranscriptionFrame(
                    transcript,
                    self._user_id,
                    time_now_iso8601(),
                    None,
                )
            )
            await self.stop_processing_metrics()
        else:
            await self.push_frame(
                InterimTranscriptionFrame(
                    transcript,
                    self._user_id,
                    time_now_iso8601(),
                    None,
                )
            )

    async def _handle_message(self, message: dict):
        try:
            is_final = message.get("is_final", False)
            segments = message.get("segments", [])

            if not segments:
                logger.warning(f"[BaseTen STT] Message has no segments: {message}")
                return

            transcript = " ".join(
                seg.get("text", "").strip() for seg in segments
            ).strip()

            if not transcript:
                logger.debug("[BaseTen STT] Empty transcript, skipping")
                return

            if is_final:
                logger.info(f"[BaseTen STT] FINAL transcript: '{transcript}'")
            else:
                logger.debug(f"[BaseTen STT] PARTIAL transcript: '{transcript}'")

            await self._handle_transcription(transcript, is_final)
        except Exception as e:
            logger.error(
                f"[BaseTen STT] Error handling message: {e} | raw: {message}"
            )

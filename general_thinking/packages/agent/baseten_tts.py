"""Baseten Orpheus TTS service with pre-warmed WebSocket connections."""

import asyncio
import json
import time
from typing import AsyncGenerator, Optional

from loguru import logger
from websockets.asyncio.client import connect
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService


class BasetenTTSService(TTSService):
    """Baseten Orpheus TTS with pre-warmed WebSocket connections.

    The Orpheus protocol is per-request: config → text → __END__ → audio → close.
    The server closes the connection after streaming audio, so true persistence
    isn't possible. Instead, we pre-open the next connection immediately after
    each request completes, so it's ready when the next run_tts() call arrives.
    This saves ~50-200ms of TCP+TLS+WS handshake latency per turn.
    """

    ORPHEUS_SAMPLE_RATE = 24000

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
        model: str = "baseten-orpheus",
        voice: str = "tara",
        language: Optional[str] = None,
        max_tokens: int = 6144,
        buffer_size: int = 30,
        send_full_text: bool = True,
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate or self.ORPHEUS_SAMPLE_RATE, **kwargs)
        self._api_key = api_key
        self._url = url
        self._max_tokens = max_tokens
        self._buffer_size = buffer_size
        self._send_full_text = send_full_text
        self._warm_ws: Optional[object] = None
        self._warm_ws_time: float = 0.0
        self._warm_task: Optional[asyncio.Task] = None
        self._shutting_down = False
        self._warm_ttl_secs = 25.0
        if hasattr(self, "set_model_name"):
            self.set_model_name(model)
        else:
            self._model_name = model
        if hasattr(self, "set_voice"):
            self.set_voice(voice)
        self._set_setting("voice", voice)
        self._set_setting("model", model)
        self._set_setting("language", language)
        self._voice_id = voice

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._shutting_down = False
        self._start_warmup()

    async def stop(self, frame: EndFrame):
        self._shutting_down = True
        await self._cleanup_warm()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        self._shutting_down = True
        await self._cleanup_warm()
        await super().cancel(frame)

    def _start_warmup(self):
        """Kick off a background task to pre-open a WebSocket."""
        if self._warm_task and not self._warm_task.done():
            return
        self._warm_task = asyncio.ensure_future(self._warmup_connection())

    async def _warmup_connection(self):
        """Open a WebSocket so it's ready for the next run_tts() call."""
        if self._shutting_down:
            return
        try:
            headers = {"Authorization": f"Api-Key {self._api_key}"}
            ws = await connect(self._url, additional_headers=headers, open_timeout=30)
            self._warm_ws = ws
            self._warm_ws_time = time.monotonic()
        except Exception as e:
            logger.warning(f"[TTS] Pre-warm connection failed (will retry on demand): {e}")
            self._warm_ws = None

    async def _cleanup_warm(self):
        """Close the pre-warmed connection and cancel the warmup task."""
        if self._warm_task and not self._warm_task.done():
            self._warm_task.cancel()
            try:
                await self._warm_task
            except (asyncio.CancelledError, Exception):
                pass
            self._warm_task = None
        if self._warm_ws:
            try:
                await self._warm_ws.close()
            except Exception:
                pass
            self._warm_ws = None

    async def _get_connection(self):
        """Return the pre-warmed connection, or open a fresh one."""
        if self._warm_task and not self._warm_task.done():
            try:
                await self._warm_task
            except Exception:
                pass

        ws = self._warm_ws
        ws_age = time.monotonic() - self._warm_ws_time if self._warm_ws_time else float("inf")
        self._warm_ws = None
        self._warm_ws_time = 0.0

        if ws and ws.state is State.OPEN and ws_age < self._warm_ttl_secs:
            return ws

        # Stale or closed — close it and open fresh
        if ws:
            try:
                await ws.close()
            except Exception:
                pass

        headers = {"Authorization": f"Api-Key {self._api_key}"}
        return await connect(self._url, additional_headers=headers, open_timeout=30)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)

            ws = await self._get_connection()

            await ws.send(json.dumps({
                "voice": self._voice_id,
                "max_tokens": self._max_tokens,
                "buffer_size": self._buffer_size,
            }))

            if self._send_full_text:
                await ws.send(text.strip())
            else:
                for word in text.strip().split():
                    await ws.send(word)

            await ws.send("__END__")

            yield TTSStartedFrame(context_id=context_id)

            async for message in ws:
                if isinstance(message, bytes):
                    await self.stop_ttfb_metrics()
                    yield TTSAudioRawFrame(
                        message,
                        self.sample_rate,
                        1,
                        context_id=context_id,
                    )

            yield TTSStoppedFrame(context_id=context_id)

        except Exception as e:
            logger.error(f"{self}: TTS error: {e}")
            yield ErrorFrame(error=f"Baseten TTS error: {e}")
        finally:
            # Pre-warm the next connection immediately
            self._start_warmup()

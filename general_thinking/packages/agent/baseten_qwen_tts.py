"""Baseten Qwen3-TTS streaming service for the WebSocket protocol.

Streams LLM text tokens directly to the TTS server as they arrive, bypassing
pipecat's built-in sentence aggregation. The server handles sentence splitting
and returns progressive PCM audio, enabling overlapped text-send / audio-receive.

Reference audio handling:
    * If ``ref_audio`` is an http(s) URL, it is forwarded to the server as-is
      inside ``session.config``.
    * If ``ref_audio`` is a local file path (or one is auto-discovered under
      ``voice_audio_dir/{voice}.{wav,mp3,flac,ogg}``), it is uploaded once via
      the WebSocket ``voice.add`` control message and afterwards referenced
      purely by ``voice`` name in subsequent ``session.config`` messages.
"""

import asyncio
import base64
import json
import os
import time
from typing import AsyncGenerator, Optional

from loguru import logger
from websockets.asyncio.client import connect
from websockets.protocol import State

from pipecat.frames.frames import (
    BotInterruptionFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    StartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.tts_service import TTSService


# Match the example client's max frame size so a base64-encoded reference
# audio (~1MB raw → ~1.3MB encoded) doesn't trip the default 1MB websockets
# message limit.
WS_MAX_SIZE = 16 * 1024 * 1024

_AUDIO_EXTS = ("wav", "mp3", "flac", "ogg")


class BasetenQwenTTSService(TTSService):
    """Baseten Qwen3-TTS streaming service over the per-deployment WebSocket.

    Protocol flow per LLM turn:
        1. Open WS, send ``session.config``
        2. Forward each ``TextFrame`` as ``{"type": "input.text", ...}``
        3. On ``LLMFullResponseEndFrame``, send ``{"type": "input.done"}``
        4. Background receiver reads ``audio.start`` / binary / ``audio.done``
           / ``session.done`` and pushes audio frames downstream
        5. Close session, pre-warm the next connection
    """

    DEFAULT_SAMPLE_RATE = 24000

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
        voice: str = "Vivian",
        task_type: str = "Base",
        response_format: str = "pcm",
        speed: float = 1.0,
        language: Optional[str] = None,
        stream_audio: bool = True,
        split_granularity: str = "sentence",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        voice_audio_dir: Optional[str] = None,
        consent: str = "user_consent",
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate or self.DEFAULT_SAMPLE_RATE, **kwargs)
        self._api_key = api_key
        self._url = url
        self._voice = voice
        self._task_type = task_type
        self._response_format = response_format
        self._speed = speed
        self._language = language
        self._stream_audio = stream_audio
        self._split_granularity = split_granularity
        self._consent = consent

        # Resolve reference audio. URLs pass through to session.config; local
        # files (whether from `ref_audio` or auto-discovered under
        # `voice_audio_dir`) are collected into `_voice_uploads` and pushed
        # once via voice.add on warmup. Subsequent sessions reference each
        # voice purely by name.
        self._ref_audio_url: Optional[str] = None
        self._ref_text: Optional[str] = ref_text
        # name → (audio_path, ref_text_or_None)
        self._voice_uploads: dict[str, tuple[str, Optional[str]]] = {}
        self._resolve_reference(ref_audio, voice_audio_dir)

        # Per-turn streaming state
        self._ws: Optional[object] = None
        self._receiver_task: Optional[asyncio.Task] = None
        # The receive loop signals turn completion through this event so that
        # `_handle_response_end` can wait for `session.done` before emitting
        # `TTSStoppedFrame`. Audio frames themselves are pushed downstream
        # directly from the receive loop — they do not go through a queue.
        self._session_done_event: asyncio.Event = asyncio.Event()
        self._session_error: Optional[dict] = None
        # Frame direction captured at session open so the background receive
        # task can push frames into the right pipeline direction without
        # depending on the inbound text frame's direction at the time of arrival.
        self._session_direction: FrameDirection = FrameDirection.DOWNSTREAM
        self._first_audio_seen = False
        self._session_active = False
        self._shutting_down = False
        self._accumulated_text = ""
        self._current_sample_rate = self.sample_rate
        self._context_id: Optional[str] = None

        # Pre-warmed connection
        self._warm_ws: Optional[object] = None
        self._warm_ws_time: float = 0.0
        self._warm_task: Optional[asyncio.Task] = None
        self._warm_ttl_secs = 25.0

        # Voice upload state (one-shot per process)
        self._voices_registered = not self._voice_uploads
        self._voice_lock = asyncio.Lock()
        self._voice_reg_task: Optional[asyncio.Task] = None
        # Set once the *active* voice is confirmed registered server-side.
        # The first streaming session waits only on this (not every voice
        # upload) before sending session.config.
        self._active_voice_ready: asyncio.Event = asyncio.Event()
        if self._voices_registered:
            self._active_voice_ready.set()

        if hasattr(self, "set_model_name"):
            self.set_model_name(task_type)
        else:
            self._model_name = task_type
        self._set_setting("voice", voice)
        self._set_setting("model", task_type)
        self._set_setting("language", language)

    # ── Reference audio resolution ───────────────────────────────

    def _resolve_reference(
        self, ref_audio: Optional[str], voice_audio_dir: Optional[str]
    ) -> None:
        """Build the bulk-upload registry from `ref_audio` + `voice_audio_dir`.

        Behavior:
            * `ref_audio` http(s) URL → forwarded as-is in `session.config`
              for the active voice (no upload).
            * `ref_audio` local file → registered under the active voice name.
            * `voice_audio_dir` → every audio file inside is registered under
              its basename (e.g. ``dan.wav`` → voice ``dan``). Sibling
              ``{name}.txt`` files are loaded as `ref_text`.
            * The constructor's `ref_text` arg, if provided, applies to the
              active voice only (overrides any sibling .txt for that voice).
        """
        if ref_audio:
            if ref_audio.startswith(("http://", "https://")):
                self._ref_audio_url = ref_audio
            elif os.path.isfile(ref_audio):
                text = self._ref_text or self._read_sibling_text(ref_audio)
                self._voice_uploads[self._voice] = (ref_audio, text)
                logger.info(
                    f"[TTS] Reference audio for active voice "
                    f"'{self._voice}': {ref_audio}"
                )
            else:
                logger.warning(
                    f"[TTS] ref_audio={ref_audio!r} is neither a URL nor an "
                    "existing file; ignoring."
                )

        if voice_audio_dir:
            self._discover_voices(voice_audio_dir)

    def _discover_voices(self, voice_audio_dir: str) -> None:
        """Scan `voice_audio_dir` and populate `_voice_uploads` for every
        audio file found, keyed by basename. Existing entries (from an
        explicit `ref_audio`) take priority and are not overwritten."""
        if not os.path.isdir(voice_audio_dir):
            logger.warning(
                f"[TTS] voice_audio_dir={voice_audio_dir!r} is not a directory; "
                "skipping bulk voice discovery."
            )
            return

        # Group entries by basename so multiple extensions for the same name
        # collapse to a single registration (first-found wins).
        for filename in sorted(os.listdir(voice_audio_dir)):
            name, ext = os.path.splitext(filename)
            ext = ext.lstrip(".").lower()
            if ext not in _AUDIO_EXTS:
                continue
            path = os.path.join(voice_audio_dir, filename)
            if not os.path.isfile(path):
                continue
            if name in self._voice_uploads:
                continue

            text = self._read_sibling_text(path)
            if name == self._voice and self._ref_text and not text:
                text = self._ref_text
            self._voice_uploads[name] = (path, text)

        if self._voice_uploads:
            names = ", ".join(sorted(self._voice_uploads))
            logger.info(
                f"[TTS] Discovered {len(self._voice_uploads)} voice(s) under "
                f"{voice_audio_dir}: {names}"
            )

    @staticmethod
    def _read_sibling_text(audio_path: str) -> Optional[str]:
        sibling = os.path.splitext(audio_path)[0] + ".txt"
        if not os.path.isfile(sibling):
            return None
        try:
            with open(sibling, "r", encoding="utf-8") as f:
                return f.read().strip()
        except OSError as e:
            logger.warning(f"[TTS] Failed to read transcript {sibling}: {e}")
            return None

    # ── Lifecycle ────────────────────────────────────────────────

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._shutting_down = False
        self._start_warmup()

    async def stop(self, frame: EndFrame):
        self._shutting_down = True
        await self._close_session()
        await self._cleanup_warm()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        self._shutting_down = True
        await self._close_session()
        await self._cleanup_warm()
        await super().cancel(frame)

    # ── Pre-warm + voice registration ────────────────────────────

    def _start_warmup(self):
        if self._warm_task and not self._warm_task.done():
            return
        self._warm_task = asyncio.ensure_future(self._warmup_connection())

    async def _warmup_connection(self):
        if self._shutting_down:
            return
        try:
            # Register voices in the background so the warm socket (and the
            # first session) don't block on the one-time upload of every voice.
            # _open_session waits only on the active voice via
            # _active_voice_ready.
            if not self._voices_registered and (
                self._voice_reg_task is None or self._voice_reg_task.done()
            ):
                self._voice_reg_task = asyncio.ensure_future(
                    self._ensure_voices_registered()
                )

            ws = await connect(
                self._url,
                additional_headers=self._auth_headers(),
                open_timeout=30,
                ping_interval=None,
                max_size=WS_MAX_SIZE,
                compression=None,
            )
            self._warm_ws = ws
            self._warm_ws_time = time.monotonic()
        except Exception as e:
            logger.warning(f"[TTS] Pre-warm connection failed: {e}")
            self._warm_ws = None

    async def _cleanup_warm(self):
        if self._warm_task and not self._warm_task.done():
            self._warm_task.cancel()
            try:
                await self._warm_task
            except (asyncio.CancelledError, Exception):
                pass
            self._warm_task = None
        if self._voice_reg_task and not self._voice_reg_task.done():
            self._voice_reg_task.cancel()
            try:
                await self._voice_reg_task
            except (asyncio.CancelledError, Exception):
                pass
            self._voice_reg_task = None
        if self._warm_ws:
            try:
                await self._warm_ws.close()
            except Exception:
                pass
            self._warm_ws = None

    async def _get_connection(self):
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

        if ws:
            try:
                await ws.close()
            except Exception:
                pass

        return await connect(
            self._url,
            additional_headers=self._auth_headers(),
            open_timeout=30,
            ping_interval=None,
            max_size=WS_MAX_SIZE,
            compression=None,
        )

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Api-Key {self._api_key}"}

    async def _ensure_voices_registered(self) -> None:
        """Upload every entry in ``_voice_uploads`` via ``voice.add`` (idempotent).

        Uses an out-of-band WebSocket connection so the streaming pre-warm
        socket stays clean. A single ``voice.list`` round-trip filters out
        names that are already registered server-side so re-launches of the
        process don't redundantly re-upload.
        """
        try:
          async with self._voice_lock:
            if self._voices_registered:
                return
            if not self._voice_uploads:
                self._voices_registered = True
                return

            try:
                ws = await connect(
                    self._url,
                    additional_headers=self._auth_headers(),
                    open_timeout=30,
                    ping_interval=None,
                    max_size=WS_MAX_SIZE,
                    compression=None,
                )
            except Exception as e:
                logger.warning(f"[TTS] Voice register: connection failed: {e}")
                return

            try:
                await ws.send(json.dumps({"type": "voice.list"}))
                resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
                existing = {
                    v.get("name")
                    for v in (resp.get("uploaded_voices") or [])
                    if v.get("name")
                }

                # If the active voice is already present (or has no upload),
                # unblock the first session immediately.
                if self._voice in existing or self._voice not in self._voice_uploads:
                    self._active_voice_ready.set()

                pending = [
                    (name, path, text)
                    for name, (path, text) in self._voice_uploads.items()
                    if name not in existing
                ]
                # Upload the active voice first so the first session can start
                # as soon as it lands, without waiting on the other voices.
                pending.sort(key=lambda item: item[0] != self._voice)
                skipped = sorted(set(self._voice_uploads) & existing)
                if skipped:
                    logger.info(
                        f"[TTS] Voices already registered server-side: "
                        f"{', '.join(skipped)}"
                    )

                uploaded = 0
                for name, path, text in pending:
                    if not os.path.isfile(path):
                        logger.warning(
                            f"[TTS] Skipping voice '{name}': file vanished "
                            f"({path})"
                        )
                        continue
                    try:
                        with open(path, "rb") as f:
                            audio_bytes = f.read()
                    except OSError as e:
                        logger.error(f"[TTS] Cannot read {path} for voice '{name}': {e}")
                        continue

                    ext = os.path.splitext(path)[1].lstrip(".").lower() or "wav"
                    msg = {
                        "type": "voice.add",
                        "name": name,
                        "consent": self._consent,
                        "audio_data": base64.b64encode(audio_bytes).decode(),
                        "audio_format": ext,
                    }
                    if text:
                        msg["ref_text"] = text

                    logger.info(
                        f"[TTS] Uploading voice '{name}' from {path} "
                        f"({len(audio_bytes):,} bytes)"
                    )
                    await ws.send(json.dumps(msg))
                    resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))

                    if resp.get("type") == "error":
                        logger.error(
                            f"[TTS] voice.add error for '{name}': "
                            f"{resp.get('message')}"
                        )
                    elif resp.get("success"):
                        logger.info(f"[TTS] Registered voice '{name}'")
                        uploaded += 1
                    else:
                        logger.warning(
                            f"[TTS] voice.add unexpected response for "
                            f"'{name}': {resp}"
                        )

                    # Release the first-session gate as soon as the active
                    # voice is uploaded, without waiting on the rest.
                    if name == self._voice:
                        self._active_voice_ready.set()

                self._voices_registered = True
                if uploaded:
                    logger.info(f"[TTS] Voice registration complete: {uploaded} new")
            except Exception as e:
                logger.error(f"[TTS] Voice registration failed: {e}")
            finally:
                try:
                    await ws.close()
                except Exception:
                    pass
        finally:
            # Never leave the first session blocked, even if registration
            # failed; _open_session also enforces a hard timeout.
            self._active_voice_ready.set()

    # ── Streaming session lifecycle ──────────────────────────────

    async def _open_session(self, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        # Block only on the active voice being registered server-side (voice
        # registration of the remaining voices continues in the background).
        if not self._active_voice_ready.is_set():
            try:
                await asyncio.wait_for(self._active_voice_ready.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "[TTS] Active voice not confirmed registered; proceeding anyway"
                )

        self._ws = await self._get_connection()
        self._session_done_event = asyncio.Event()
        self._session_error = None
        self._session_direction = direction
        self._first_audio_seen = False
        self._session_active = True
        self._accumulated_text = ""
        self._current_sample_rate = self.sample_rate

        config: dict = {
            "type": "session.config",
            "task_type": self._task_type,
            "response_format": self._response_format,
            "speed": self._speed,
            "split_granularity": self._split_granularity,
            "voice": self._voice,
        }
        if self._stream_audio:
            config["stream_audio"] = True
            # The example forces PCM whenever stream_audio is set, otherwise
            # the server emits a chunked container that we don't parse.
            config["response_format"] = "pcm"
        if self._language:
            config["language"] = self._language
        # Only forward ref_audio when it's an actual URL the server can fetch.
        # Local-file references are already represented by self._voice via
        # voice.add and don't need to be re-sent each turn.
        if self._ref_audio_url:
            config["ref_audio"] = self._ref_audio_url
            if self._ref_text:
                config["ref_text"] = self._ref_text

        await self._ws.send(json.dumps(config))
        logger.debug(
            f"[TTS] Session opened: voice={self._voice}, "
            f"format={config.get('response_format')}, "
            f"stream_audio={config.get('stream_audio', False)}"
        )

        self._receiver_task = asyncio.ensure_future(self._receive_loop())

    async def _receive_loop(self):
        """Background task: read WS messages and push audio downstream eagerly.

        Audio chunks are pushed onto the pipeline as soon as they arrive on the
        socket rather than being parked in a queue and drained on the next LLM
        token. This keeps TTS-to-speaker latency independent of LLM token
        cadence and avoids tail-latency padding between the last LLM token and
        the final ``input.done`` flush. ``session.done`` / ``error`` are
        surfaced through ``self._session_done_event`` so ``_handle_response_end``
        knows when it's safe to emit ``TTSStoppedFrame`` and close the socket.
        """
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    if not self._first_audio_seen:
                        self._first_audio_seen = True
                        await self.stop_ttfb_metrics()
                    await self.push_frame(
                        TTSAudioRawFrame(
                            message,
                            self._current_sample_rate,
                            1,
                            context_id=self._context_id,
                        ),
                        self._session_direction,
                    )
                else:
                    msg = json.loads(message)
                    msg_type = msg.get("type")

                    if msg_type == "audio.start":
                        self._current_sample_rate = int(
                            msg.get("sample_rate", self.sample_rate)
                        )
                        logger.debug(
                            f"[TTS] Sentence {msg.get('sentence_index')}: "
                            f"{msg.get('sentence_text', '')!r}"
                        )
                    elif msg_type == "audio.done":
                        if msg.get("error"):
                            logger.error(
                                f"[TTS] audio.done error sentence "
                                f"{msg.get('sentence_index')}"
                            )
                    elif msg_type == "session.done":
                        logger.debug(
                            f"[TTS] Session complete: "
                            f"{msg.get('total_sentences')} sentence(s)"
                        )
                        self._session_done_event.set()
                        return
                    elif msg_type == "error":
                        logger.error(f"[TTS] Server error: {msg.get('message')}")
                        self._session_error = msg
                        self._session_done_event.set()
                        return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self._shutting_down:
                logger.error(f"[TTS] Receiver error: {e}")
                self._session_error = {"message": str(e)}
                self._session_done_event.set()

    async def _close_session(self):
        if self._receiver_task and not self._receiver_task.done():
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except (asyncio.CancelledError, Exception):
                pass
            self._receiver_task = None

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        self._session_active = False
        self._session_done_event.set()

    # ── Frame processing (bypasses TTSService sentence aggregation) ──

    async def process_frame(self, frame, direction):
        await FrameProcessor.process_frame(self, frame, direction)

        if isinstance(frame, StartFrame):
            await self.start(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            await self._close_session()
            await self.stop(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, CancelFrame):
            await self._close_session()
            await self.cancel(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, (InterruptionFrame, BotInterruptionFrame)):
            # User barge-in / explicit truncate: tear down the in-flight TTS
            # session so the next turn opens a fresh server-side context and
            # we don't interleave the new turn's text with stale audio from
            # the cancelled one. Pre-warming the next connection here keeps
            # the latency budget for the next turn unchanged.
            await self._handle_interruption(frame, direction)
        elif isinstance(frame, TextFrame) and frame.text:
            await self._handle_text(frame, direction)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._handle_response_end(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _handle_text(self, frame: TextFrame, direction):
        if not self._session_active:
            self._context_id = getattr(frame, "context_id", None)
            # Emit TTSStartedFrame BEFORE the session.config send, so
            # downstream observers (UI, metrics) see "TTS started" without
            # waiting for the WS handshake / config round-trip.
            await self.push_frame(
                TTSStartedFrame(context_id=self._context_id), direction
            )
            await self.start_ttfb_metrics()
            await self._open_session(direction)

        self._accumulated_text += frame.text

        # Forward the text downstream so the assistant context aggregator
        # (placed right after TTS) records the spoken response in the LLM
        # context. Without this the bot has no memory of its own replies and
        # re-answers earlier turns.
        await self.push_frame(frame, direction)

        try:
            await self._ws.send(
                json.dumps({"type": "input.text", "text": frame.text})
            )
        except Exception as e:
            logger.error(f"[TTS] Send error: {e}")
            await self.push_frame(
                ErrorFrame(error=f"TTS send error: {e}"), direction
            )
            await self._close_session()
            return

    async def _handle_response_end(self, frame, direction):
        if self._session_active and self._ws:
            try:
                await self._ws.send(json.dumps({"type": "input.done"}))
            except Exception as e:
                logger.error(f"[TTS] Failed to send input.done: {e}")

            await self.start_tts_usage_metrics(self._accumulated_text)

            # Wait for the receive loop to surface session.done / error.
            # Audio frames were already pushed downstream eagerly by the
            # receive loop, so this is purely a synchronization barrier.
            try:
                await asyncio.wait_for(
                    self._session_done_event.wait(), timeout=60.0
                )
            except asyncio.TimeoutError:
                logger.warning("[TTS] Timed out waiting for session.done")

            if self._session_error is not None:
                await self.push_frame(
                    ErrorFrame(
                        error=f"TTS error: {self._session_error.get('message', 'unknown')}"
                    ),
                    direction,
                )

            await self.push_frame(
                TTSStoppedFrame(context_id=self._context_id), direction
            )

            await self._close_session()
            self._accumulated_text = ""
            self._context_id = None
            self._start_warmup()

        await self.push_frame(frame, direction)

    async def _handle_interruption(self, frame, direction):
        """Tear down any in-flight TTS session on barge-in / truncate.

        Without this, a mid-turn interruption leaves ``_session_active=True``
        with an open WebSocket and a buffered server-side context. The next
        turn's first TextFrame then skips ``_open_session`` and appends to
        the cancelled session, interleaving audio from two turns on the
        same server context. Closing the socket here forces the next turn
        to start a clean session against a freshly-warmed connection.
        """
        if self._session_active:
            logger.info("[TTS] Interruption — closing in-flight session")
            await self._close_session()
            self._accumulated_text = ""
            self._context_id = None
            self._start_warmup()
        await self.push_frame(frame, direction)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.warning("[TTS] run_tts called unexpectedly in streaming mode")
        yield ErrorFrame(error="Streaming mode bypasses run_tts")

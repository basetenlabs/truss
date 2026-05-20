"""STTHandler: decompress_bytes -> call_stt -> post_process_text.

Inner WS handler. Receives gzipped MOCKAUDIO bytes per WS frame from the
entrypoint, decompresses, fans out to STTMock over HTTP for the actual
"transcription," then post-processes (trims whitespace, normalizes punct)
and sends a JSON frame back. The envelope codec is duplicated inline
(rather than shared) because TrussChainlet trusses are self-contained.
"""

import base64
import gzip
import json
import logging

import httpx

from truss_chains import ServiceHandle

logger = logging.getLogger("voice_agent_mocked.stt_handler")

_HEAD = b"MOCKAUDIO|"
_TAIL = b"|END"


def _decompress(buf: bytes) -> bytes:
    return gzip.decompress(buf)


def _decode_envelope(buf: bytes) -> str:
    if not (buf.startswith(_HEAD) and buf.endswith(_TAIL)):
        raise ValueError(f"not a MOCKAUDIO envelope: {buf[:32]!r}...")
    return buf[len(_HEAD) : -len(_TAIL)].decode("utf-8")


def _post_process(text: str) -> str:
    return " ".join(text.split())


class Model:
    def __init__(self, **_) -> None:
        self._stt: ServiceHandle | None = None

    def load(self) -> None:
        self._stt = ServiceHandle("STTMock")
        logger.info(
            "[LOAD] STTHandler ready; STTMock predict_url=%s",
            self._stt.urls.predict_url,
        )

    async def _call_stt(self, audio: bytes) -> dict:
        assert self._stt is not None
        url, headers = self._stt.http_call_args(prefer_internal=True)
        logger.info("[STEP] call_stt: POST %s (%d bytes payload)", url, len(audio))
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                url,
                json={"audio_b64": base64.b64encode(audio).decode()},
                headers=headers,
            )
            r.raise_for_status()
            body = r.json()
        logger.info("[STEP] call_stt: STTMock returned text=%r", body.get("text"))
        return body

    async def websocket(self, ws) -> None:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                logger.info("[WS] client disconnect")
                return
            data = msg.get("bytes")
            if data is None and msg.get("text") is not None:
                if msg["text"] == "DONE":
                    logger.info("[WS] DONE received; closing")
                    await ws.close(code=1000, reason="client requested close")
                    return
                continue
            if data is None:
                continue
            logger.info("[WS] received %d gzipped bytes from entrypoint", len(data))
            try:
                decompressed = _decompress(data)
                logger.info(
                    "[STEP] decompress_bytes: %d -> %d bytes",
                    len(data),
                    len(decompressed),
                )
                stt = await self._call_stt(decompressed)
                processed = _post_process(stt["text"])
                logger.info("[STEP] post_process_text: %r", processed)
                reply = {
                    "chainlet": "STTHandler",
                    "text": processed,
                    "envelope_preview": _decode_envelope(decompressed),
                    "stt": stt,
                }
            except Exception as e:
                logger.exception("[ERROR] websocket handler")
                reply = {"chainlet": "STTHandler", "error": f"{type(e).__name__}: {e}"}
            await ws.send_text(json.dumps(reply))

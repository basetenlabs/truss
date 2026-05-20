"""TTSHandler: call_tts -> hash_result.

HTTP handler. Calls TTSMock for synthesis, then appends an integrity
hash (``|H:<6-hex>``) so the entrypoint can detect truncation.
"""

import base64
import hashlib
import logging

import httpx

from truss_chains import ServiceHandle

logger = logging.getLogger("voice_agent_mocked.tts_handler")


def _append_hash(buf: bytes) -> bytes:
    return buf + b"|H:" + hashlib.md5(buf).hexdigest()[:6].encode()


class Model:
    def __init__(self, **_) -> None:
        self._tts: ServiceHandle | None = None

    def load(self) -> None:
        self._tts = ServiceHandle("TTSMock")
        logger.info(
            "[LOAD] TTSHandler ready; TTSMock predict_url=%s",
            self._tts.urls.predict_url,
        )

    async def _call_tts(self, text: str) -> dict:
        assert self._tts is not None
        url, headers = self._tts.http_call_args(prefer_internal=True)
        logger.info("[STEP] call_tts: POST %s text=%r", url, text)
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, json={"text": text}, headers=headers)
            r.raise_for_status()
            body = r.json()
        logger.info(
            "[STEP] call_tts: TTSMock returned %d audio bytes", body.get("audio_len", 0)
        )
        return body

    async def predict(self, request: dict) -> dict:
        text = request["text"]
        logger.info("[BEGIN] tts_handler predict: text=%r", text)
        tts = await self._call_tts(text)
        audio = base64.b64decode(tts["audio_b64"])
        hashed = _append_hash(audio)
        logger.info(
            "[STEP] hash_result: %d -> %d bytes (appended 10-byte hash tag)",
            len(audio),
            len(hashed),
        )
        return {
            "chainlet": "TTSHandler",
            "audio_b64": base64.b64encode(hashed).decode(),
            "audio_len_hashed": len(hashed),
            "tts": tts,
        }

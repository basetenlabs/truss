"""TTSMock: text_to_bytes.

Encodes text into a MOCKAUDIO|<text>|END envelope and returns
base64-encoded bytes plus fake duration metadata. The envelope is
ASCII-readable so a reader inspecting the WS frame back to the client
can see the synthesized "speech" verbatim.
"""

import asyncio
import base64
import logging

logger = logging.getLogger("voice_agent_mocked.tts_mock")

_HEAD = b"MOCKAUDIO|"
_TAIL = b"|END"


def _encode_envelope(text: str) -> bytes:
    return _HEAD + text.encode("utf-8") + _TAIL


class Model:
    def __init__(self, **_) -> None:
        pass

    def load(self) -> None:
        logger.info("[LOAD] TTSMock ready")

    async def predict(self, request: dict) -> dict:
        text = request["text"]
        logger.info("[STEP] text_to_bytes: synthesizing %d chars: %r", len(text), text)
        await asyncio.sleep(0.06 + min(len(text), 200) / 4000)
        audio = _encode_envelope(text)
        logger.info("[STEP] text_to_bytes: produced %d bytes of MOCKAUDIO", len(audio))
        return {
            "chainlet": "TTSMock",
            "audio_b64": base64.b64encode(audio).decode(),
            "audio_len": len(audio),
            "duration_ms": 80 + len(text) * 4,
        }

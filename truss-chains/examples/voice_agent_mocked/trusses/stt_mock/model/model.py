"""STTMock: bytes_to_text.

Decodes the MOCKAUDIO|<text>|END envelope and returns the embedded text
plus deterministic confidence + latency metadata. Light asyncio.sleep
to make the LLM-stage gather behavior visible in client timings.
"""

import asyncio
import base64
import logging

logger = logging.getLogger("voice_agent_mocked.stt_mock")

_HEAD = b"MOCKAUDIO|"
_TAIL = b"|END"


def _decode_envelope(buf: bytes) -> str:
    if not (buf.startswith(_HEAD) and buf.endswith(_TAIL)):
        raise ValueError(f"not a MOCKAUDIO envelope: {buf[:32]!r}...")
    return buf[len(_HEAD) : -len(_TAIL)].decode("utf-8")


class Model:
    def __init__(self, **_) -> None:
        pass

    def load(self) -> None:
        logger.info("[LOAD] STTMock ready")

    async def predict(self, request: dict) -> dict:
        audio_b64 = request["audio_b64"]
        audio = base64.b64decode(audio_b64)
        duration_ms = max(len(audio) * 4, 80)
        logger.info(
            "[STEP] bytes_to_text: %d bytes inbound, simulating %dms STT",
            len(audio),
            duration_ms,
        )
        await asyncio.sleep(0.05 + min(duration_ms, 400) / 4000)
        text = _decode_envelope(audio)
        logger.info("[STEP] bytes_to_text: decoded text=%r", text)
        return {
            "chainlet": "STTMock",
            "text": text,
            "confidence": 0.92,
            "duration_ms": duration_ms,
            "latency_ms": 50 + min(duration_ms, 400) // 4,
        }

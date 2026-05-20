"""DiarizerMock: full_buffer_to_rttm.

Reads all FRAME lines from the buffer and emits one RTTM-shaped segment
per frame, preserving the embedded speaker. Adds ~150ms latency so the
``asyncio.gather`` payoff is visible in client timings.
"""

import asyncio
import base64
import logging

logger = logging.getLogger("perf_streaming_frontend.diarizer_mock")

_PREFIX = "FRAME|"


def _split_frames(buf: bytes) -> list[tuple[str, str]]:
    out = []
    for line in buf.split(b"\n"):
        s = line.decode("utf-8", errors="replace").strip()
        if not s.startswith(_PREFIX):
            continue
        parts = s[len(_PREFIX) :].split("|", 1)
        if len(parts) == 2:
            out.append((parts[0], parts[1]))
    return out


class Model:
    def __init__(self, **_) -> None:
        pass

    def load(self) -> None:
        logger.info("[LOAD] DiarizerMock ready")

    async def predict(self, request: dict) -> dict:
        buf = base64.b64decode(request["audio_b64"])
        logger.info("[STEP] full_buffer_to_rttm: %d bytes inbound", len(buf))
        await asyncio.sleep(0.15)
        frames = _split_frames(buf)
        segs = []
        t = 0.0
        for spk, sent in frames:
            dur = 0.4 + 0.05 * len(sent)
            segs.append(
                {
                    "speaker": spk,
                    "start": round(t, 2),
                    "end": round(t + dur, 2),
                    "text_hint": sent[:20],
                }
            )
            t += dur
        logger.info("[STEP] full_buffer_to_rttm: emitted %d RTTM segments", len(segs))
        return {
            "chainlet": "DiarizerMock",
            "format": "rttm",
            "segments": segs,
            "latency_ms": 150,
        }

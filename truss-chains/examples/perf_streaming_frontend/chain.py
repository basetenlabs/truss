"""Perf streaming front-end showcase — TC CS WS entrypoint in Go.

Topology::

    client ──WS bytes──> EntryStreamingGo (TC CS WS, Go)
                           │
                (A)  per FRAME|<spk>|<text>\\n line, HTTP POST
                           ▼
                       TranscriberMock (CB Py HTTP)
                           │  returns partial transcript text
                           │
                (B)  on end-byte (FRAME|END / DONE / close), HTTP POST
                           ▼  with full accumulated buffer
                       DiarizerMock (TC Py HTTP)
                           │  returns RTTM-shaped segments
                           │
              (A+B → C)  after both settle, HTTP POST
                           ▼
                       AssignerMock (TC CS HTTP, Rust)
                           │  returns [{speaker, text}, ...]
                           ▼
    client ◄─WS text── EntryStreamingGo

Polyglot: Go (entrypoint), Python (transcriber, diarizer), Rust (assigner).
"""

import asyncio
import hashlib
import logging

import pydantic

import truss_chains as chains

logger = logging.getLogger("perf_streaming_frontend.transcriber")

_TRUSS_OVERRIDE = (
    "truss @ git+https://github.com/basetenlabs/truss.git@matte/chains-trusschainlets"
)


class _TranscribeReply(pydantic.BaseModel):
    chainlet: str
    text: str
    order: int
    latency_ms: int


class TranscriberMock(chains.ChainletBase):
    """Partial-buffer transcriber. Strips the FRAME|<spk>|<text> envelope
    and returns ``text`` — the deterministic 'transcribed' sentence.

    Scalar args (not a wrapping Pydantic model) so the polyglot Go caller
    can post ``{"frame": ..., "order": ...}`` directly.
    """

    remote_config = chains.RemoteConfig(
        compute=chains.Compute(cpu_count=1, memory="256Mi"),
        docker_image=chains.DockerImage(
            base_image=chains.BasetenImage.PY311, pip_requirements=[_TRUSS_OVERRIDE]
        ),
    )

    async def run_remote(self, frame: str, order: int) -> _TranscribeReply:
        logger.info("[STEP] partial_buffer_to_text: order=%s frame=%r", order, frame)
        await asyncio.sleep(0.08)
        stripped = frame.strip()
        if stripped.startswith("FRAME|"):
            parts = stripped[len("FRAME|") :].split("|", 1)
            text = parts[1] if len(parts) == 2 else ""
        else:
            text = ""
        marker = hashlib.md5(stripped.encode()).hexdigest()[:4]
        out = f"{text} [#{marker}]" if text else ""
        logger.info("[STEP] partial_buffer_to_text: order=%s -> %r", order, out)
        return _TranscribeReply(
            chainlet="TranscriberMock", text=out, order=order, latency_ms=80
        )


class DiarizerMock(chains.TrussChainlet):
    truss_dir = "./trusses/diarizer_mock"


class AssignerMock(chains.TrussChainlet):
    truss_dir = "./trusses/assigner_mock"


@chains.mark_entrypoint("Perf Streaming Frontend")
class EntryStreamingGo(chains.TrussChainlet):
    truss_dir = "./trusses/entry_streaming_go"
    deps = [TranscriberMock, DiarizerMock, AssignerMock]

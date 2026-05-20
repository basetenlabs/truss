"""Voice agent showcase — CB WS entrypoint, sequential STT -> LLM -> TTS.

Topology::

    client ──WS bytes──> EntryVoiceAgent (CB WS)
                            │
            (A) STT branch  │  compress + gzip(MOCKAUDIO bytes)
                            ▼
                         STTHandler (TC Py WS)
                            │  HTTP JSON (decompressed bytes)
                            ▼
                         STTMock (TC Py HTTP)  ──► text
                            │
            (B) LLM branch  │  OpenAI client /v1/chat/completions
                            ▼
                         LLMMock (TC CS HTTP, real vLLM, --load-format dummy)
                            │  (random tokens — entrypoint uses
                            │   deterministic lookup for legibility)
                            ▼
                         "assistant_text"
                            │
            (C) TTS branch  │  HTTP JSON
                            ▼
                         TTSHandler (TC Py HTTP) ── HTTP JSON ─►
                         TTSMock (TC Py HTTP) ──► MOCKAUDIO bytes
                            │
                            ▼
    client ◄─WS bytes── EntryVoiceAgent

Per inbound WS audio frame: A -> B -> C runs sequentially. The
entrypoint returns one frame back per request containing the synthesized
audio bytes (MOCKAUDIO envelope + integrity hash).
"""

import gzip
import json
import logging

import httpx
import websockets

import truss_chains as chains
from truss_chains import ServiceHandle

_TRUSS_OVERRIDE = (
    "truss @ git+https://github.com/basetenlabs/truss.git@matte/chains-trusschainlets"
)

logger = logging.getLogger("voice_agent_mocked.entry")

_MOCKAUDIO_HEAD = b"MOCKAUDIO|"
_MOCKAUDIO_TAIL = b"|END"


def _mockaudio_encode(text: str) -> bytes:
    return _MOCKAUDIO_HEAD + text.encode("utf-8") + _MOCKAUDIO_TAIL


_DETERMINISTIC_LLM_REPLIES = {
    "What's the weather?": "Looks sunny, around 72 degrees.",
    "Tell me a joke.": "Two chains walk into a chainlet. The third one types.",
    "Who are you?": "I'm a mocked voice agent running on Baseten Chains.",
}
_DEFAULT_LLM_REPLY = "I heard you, but my training is purely vibes."


class STTMock(chains.TrussChainlet):
    truss_dir = "./trusses/stt_mock"


class STTHandler(chains.TrussChainlet):
    truss_dir = "./trusses/stt_handler"
    deps = [STTMock]


class LLMMock(chains.TrussChainlet):
    truss_dir = "./trusses/llm_mock"


class TTSMock(chains.TrussChainlet):
    truss_dir = "./trusses/tts_mock"


class TTSHandler(chains.TrussChainlet):
    truss_dir = "./trusses/tts_handler"
    deps = [TTSMock]


@chains.mark_entrypoint("Voice Agent Mocked")
class EntryVoiceAgent(chains.ChainletBase):
    remote_config = chains.RemoteConfig(
        compute=chains.Compute(cpu_count=1, memory="1Gi"),
        docker_image=chains.DockerImage(
            base_image=chains.BasetenImage.PY311,
            pip_requirements=[
                _TRUSS_OVERRIDE,
                "httpx>=0.27",
                "websockets>=12",
                "openai>=1.30",
            ],
        ),
    )

    def __init__(
        self,
        stt_handler: ServiceHandle = chains.depends(STTHandler),
        llm: ServiceHandle = chains.depends(LLMMock),
        tts_handler: ServiceHandle = chains.depends(TTSHandler),
    ) -> None:
        self._stt = stt_handler
        self._llm = llm
        self._tts = tts_handler

    @staticmethod
    def _compress_bytes(audio: bytes) -> bytes:
        out = gzip.compress(audio)
        logger.info(
            "[STEP 1] compress_bytes: %d -> %d bytes (gzip)", len(audio), len(out)
        )
        return out

    async def _call_stt_handler(self, audio: bytes) -> dict:
        compressed = self._compress_bytes(audio)
        url, headers = self._stt.ws_call_args()
        logger.info("[STEP 2] call_stt_handler: WS connect %s", url)
        async with websockets.connect(
            url, additional_headers=headers, open_timeout=15
        ) as ws:
            await ws.send(compressed)
            logger.info(
                "[STEP 2] call_stt_handler: sent %d bytes (gzipped) over WS",
                len(compressed),
            )
            reply = await ws.recv()
            await ws.send("DONE")
            try:
                await ws.recv()
            except websockets.exceptions.ConnectionClosedOK:
                pass
        parsed = json.loads(reply) if isinstance(reply, str) else {"_raw": reply}
        logger.info("[STEP 2] call_stt_handler: STT reply text=%r", parsed.get("text"))
        return parsed

    async def _call_llm(self, prompt: str) -> dict:
        import openai

        # vLLM serves OpenAI-compatible /v1/chat/completions; reach it through
        # the platform's /sync/<path> passthrough (only the external URL does
        # path passthrough; cluster-internal doesn't). sync_path="v1" gives
        # the OpenAI base_url; the client appends /chat/completions itself.
        base_url, headers = self._llm.http_call_args(sync_path="v1")
        logger.info("[STEP 3] call_llm: prompt=%r, vLLM base_url=%s", prompt, base_url)
        # Short per-attempt timeout (10s) + 2 retries: tail-latency spikes on
        # the shared CPU node usually resolve within seconds, but stale work
        # piles up on vLLM (each retry holds a --max-num-seqs slot). 10s is
        # tight enough to bail fast; 2 retries absorb transient flakes
        # without amplifying server load like the SDK default (timeout=60,
        # max_retries=2 → 3 minutes of stale work).
        client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key="vllm-doesnt-check-this",
            default_headers=headers,
            timeout=10,
            max_retries=2,
        )
        # Use the non-chat /v1/completions endpoint — pythia-14m's tokenizer
        # has no chat template and modern vLLM refuses /v1/chat/completions
        # without one. Completions takes a raw prompt and works on any base
        # model with a tokenizer.
        completion = await client.completions.create(
            model="EleutherAI/pythia-14m", prompt=prompt, max_tokens=24
        )
        raw = completion.choices[0].text or ""
        legible = _DETERMINISTIC_LLM_REPLIES.get(prompt, _DEFAULT_LLM_REPLY)
        tokens = completion.usage.completion_tokens if completion.usage else None
        logger.info(
            "[STEP 3] call_llm: vLLM returned %s tokens (random weights); deterministic_lookup=%r",
            tokens,
            legible,
        )
        return {
            "chainlet": "LLMMock",
            "vllm_raw": raw,
            "vllm_token_count": tokens,
            "vllm_model": completion.model,
            "legible_response": legible,
        }

    async def _call_tts_handler(self, text: str) -> dict:
        url, headers = self._tts.http_call_args(prefer_internal=True)
        logger.info("[STEP 4] call_tts_handler: POST %s text=%r", url, text)
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, json={"text": text}, headers=headers)
            r.raise_for_status()
            reply = r.json()
        logger.info(
            "[STEP 4] call_tts_handler: got %d bytes of audio_b64",
            len(reply.get("audio_b64") or ""),
        )
        return reply

    async def _process_one(self, audio_bytes: bytes) -> dict:
        logger.info(
            "[BEGIN] process_one: %d bytes of MOCKAUDIO inbound", len(audio_bytes)
        )
        stt = await self._call_stt_handler(audio_bytes)
        user_text = (stt.get("text") or "").strip() if not stt.get("error") else ""
        llm = await self._call_llm(user_text)
        tts = await self._call_tts_handler(llm["legible_response"])
        logger.info("[END] process_one: assistant_text=%r", llm["legible_response"])
        return {
            "stages": {"stt": stt, "llm": llm, "tts": tts},
            "user_text": user_text,
            "assistant_text": llm["legible_response"],
            "audio_b64": tts.get("audio_b64"),
        }

    async def run_remote(self, websocket: chains.WebSocketProtocol) -> None:
        try:
            while True:
                msg = await websocket.receive()
                if isinstance(msg, str):
                    if msg == "DONE":
                        await websocket.close(
                            code=1000, reason="client requested close"
                        )
                        return
                    data = _mockaudio_encode(msg)
                else:
                    data = msg
                result = await self._process_one(data)
                await websocket.send_text(json.dumps(result))
        except Exception as e:
            logger.exception("[WS] run_remote error")
            try:
                await websocket.send_text(
                    json.dumps({"error": f"{type(e).__name__}: {e}"})
                )
            except Exception:
                pass

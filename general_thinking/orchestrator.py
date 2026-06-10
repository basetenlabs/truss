"""FDE Voice Agent — Composable Chains migration.

Wraps the existing STT / LLM / TTS Truss directories as ``chains.TrussChainlet``
members and turns the orchestrator into a ``chains.ChainletBase`` WebSocket
entrypoint. The agent runtime (``run_bot``) is reused as-is — only the URL
plumbing changes: env-var reads → descriptor injection.

Push:

    truss chains push --remote matte chain.py --watch

The UI connects to the chain's entrypoint WebSocket URL (printed by the push).
"""

import logging

import truss_chains as chains
from truss_chains.remote_chainlet.truss_chainlet import (
    TrussHandle,
    get_baseten_chain_api_key,
)

log = logging.getLogger("orchestrator")

# ----- Existing Trusses as chain members ------------------------------------


class STT(chains.TrussChainlet):
    """Whisper streaming STT — the existing ``./stt/`` Truss, untouched."""

    truss_dir = "./stt"


class LLM(chains.TrussChainlet):
    """GPT-OSS-20B (TRT-LLM, OpenAI-compatible) — the existing ``./llm/``."""
    truss_dir = "./llm"


class TTS(chains.TrussChainlet):
    """Orpheus 3B streaming TTS — the existing ``./tts/`` Truss, untouched."""

    truss_dir = "./tts"


# ----- Orchestrator: WebSocket entrypoint -----------------------------------


@chains.mark_entrypoint("B10 Voice Agent")
class Orchestrator(chains.ChainletBase):
    """WebSocket entrypoint that runs the Pipecat voice pipeline.

    Replaces the legacy env-var-based wiring (``STT_URL`` / ``LLM_URL`` /
    ``TTS_URL``) with descriptor injection. The chain push deploys all four
    artifacts atomically; the platform's ``dynamic_chainlet_config`` ConfigMap
    delivers each sibling's URL into this pod.
    """

    remote_config = chains.RemoteConfig(
        compute=chains.Compute(cpu_count=8, memory="32Gi"),
        docker_image=chains.DockerImage(
            base_image=chains.BasetenImage.PY310,
            pip_requirements=[
                "pipecat-ai[websocket,openai,silero,runner,mcp]",
                "loguru",
                "python-dotenv",
                "aiohttp",
                "httpx[http2]",
            ],
            apt_requirements=["ffmpeg"],
        ),
    )

    def __init__(
        self,
        stt: TrussHandle = chains.depends(STT),
        llm: TrussHandle = chains.depends(LLM),
        tts: TrussHandle = chains.depends(TTS),
        context: chains.DeploymentContext = chains.depends_context(),
    ) -> None:
        # Strip Baseten's internal HTTP proxy env vars — they intercept the
        # outbound WebSocket connections to STT/TTS sibling models. (Same
        # mitigation the standalone orchestrator's model.py applied.)
        import os

        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ.pop(var, None)
        os.environ["NO_PROXY"] = "*"

        # Sibling URLs from TrussHandle — chain-internal endpoints authorized
        # by the framework-injected ``baseten_chain_api_key`` secret. STT/TTS
        # use WS; LLM uses the ``/sync/v1`` passthrough so the OpenAILLMService
        # can talk to vLLM's OpenAI-compatible endpoint via the api-gateway.
        self._stt_url = stt.ws_call_args().url
        self._tts_url = tts.ws_call_args().url
        self._llm_base_url = llm.http_call_args(sync_path="v1").url
        # The agent's pipecat clients attach Authorization: Api-Key <key>
        # using ``secrets["BASETEN_API_KEY"]``.
        self._secrets = {"BASETEN_API_KEY": get_baseten_chain_api_key()}

    async def run_remote(self, websocket: chains.WebSocketProtocol) -> None:
        # Imports live inside run_remote so codegen's parse step doesn't try to
        # resolve them in the local push environment.
        from packages.agent.agent import run_bot
        from pipecat.services.openai.llm import OpenAILLMService

        # Pipecat's FastAPIWebsocketTransport reads `client_state` /
        # `application_state` and uses `receive_bytes()` semantics that the
        # chains-supplied `WebsocketWrapperFastAPI` doesn't expose. Unwrap to the
        # underlying fastapi.WebSocket so pipecat sees the shape it expects.
        raw_ws = getattr(websocket, "_websocket", websocket)

        stt_kwargs = {'metadata': {'include_timing_info': True,
                      'streaming_params': {'enable_partial_transcripts': True,
                                           'final_transcript_max_duration_s': 30,
                                           'partial_transcript_interval_s': 0.3},
                      'streaming_vad_config': {'min_silence_duration_ms': 100,
                                               'speech_pad_ms': 30,
                                               'threshold': 0.4},
                      'whisper_params': {'audio_language': 'en'}},
         'sample_rate': 16000,
         'url': self._stt_url}

        llm_kwargs = {
            'model': 'openai/gpt-oss-20b',
            'system_prompt': ('You are a helpful, conversational AI speaking to a user in a live voice call. Speak naturally '
         'and keep your responses short and easy to understand. Do not include formatting, special '
         'characters, or instructional tags--everything you say will be spoken aloud exactly as written. '
         'Avoid technical words, markdown, or referencing AI processes. Never use abbreviations--say '
         "'versus' not 'vs', 'for example' not 'e.g.'. Do not use bullet points, numbered lists, or any "
         'structured formatting. Never mention that you are an AI or describe your thoughts; simply '
         'respond as if you are conversing with the user in real-time. Use your available tools to help '
         'the user.'),
            'base_url': self._llm_base_url,
            'params': OpenAILLMService.InputParams(**{'extra': {'reasoning_effort': 'low'}}),
        }

        mcp_servers = []

        tts_kwargs = {'provider': 'baseten_qwen',
         'response_format': 'pcm',
         'speed': 1.0,
         'split_granularity': 'sentence',
         'stream_audio': True,
         'task_type': 'Base',
         'url': self._tts_url,
         'voice': 'jim',
         'voice_audio_dir': '${PROJECT_ROOT}/data/voices'}

        await run_bot(
            raw_ws,
            secrets=self._secrets,
            stt_kwargs=stt_kwargs,
            llm_kwargs=llm_kwargs,
            tts_kwargs=tts_kwargs,
            mcp_servers=mcp_servers,
        )

"""Voice Agent (Mocked) — Composable Chains centerpiece example.

A CPU-only mock of the FDE voice-agent chain, designed to validate routing
behavior end-to-end without the cost of GPU model deploys. Same shape as
``/Users/mattelim/Documents/fde/e2e-voice``:

- ``STTMock``     (``TrussChainlet``, WS-fronted): bytes ``→`` ``{"text": "text-{N}"}``
- ``LLMMock``     (``TrussChainlet``, HTTP):       ``{"prompt"}`` ``→`` ``{"completion": "echo: ..."}``
- ``TTSMock``     (``TrussChainlet``, WS-fronted): text  ``→`` ``text.encode()``
- ``Orchestrator`` (``ChainletBase``, WS entrypoint): a single round-trip wires
  STT ``→`` LLM ``→`` TTS using the framework-injected ``DeployedServiceDescriptor``
  URLs and helpers — ``internal_ws_url`` / ``target_url`` for the URL,
  ``with_ws_auth_headers`` / ``with_auth_headers`` for headers. No GraphQL, no
  hardcoded hostnames.

Iteration loop:

    truss chains push --remote matte chain.py --watch

The orchestrator's ``run_remote`` reads bytes from the entrypoint WS, drives
the pipeline, and sends back the synthesized bytes. Useful as a regression
fixture for any baseten-local change that touches chain-internal routing.
"""

import json
import logging

import truss_chains as chains

log = logging.getLogger("voice-agent-mocked")

_TRUSS_OVERRIDE = (
    "truss @ git+https://github.com/basetenlabs/truss.git@matte/composable-chains"
)


# ----- Mock TrussChainlet members --------------------------------------------


class STTMock(chains.TrussChainlet):
    truss_dir = "./stt_mock"


class LLMMock(chains.TrussChainlet):
    truss_dir = "./llm_mock"


class TTSMock(chains.TrussChainlet):
    truss_dir = "./tts_mock"


# ----- Orchestrator: WebSocket entrypoint ------------------------------------


@chains.mark_entrypoint("VoiceAgentMocked")
class Orchestrator(chains.ChainletBase):
    """Single-round-trip WS entrypoint: bytes in -> bytes out, exercising the
    same STT/LLM/TTS sibling-call patterns as the FDE chain.

    Sibling URLs come straight from the framework-injected ``DeployedServiceDescriptor``
    objects — ``desc.ws_url`` / ``desc.target_url`` are populated by the runtime
    layer from the platform's dynamic chainlet config. No GraphQL oracle-id
    resolution, no hardcoded ``api.*.baseten.co`` host: the descriptor knows
    where to go.
    """

    remote_config = chains.RemoteConfig(
        compute=chains.Compute(cpu_count=1, memory="512Mi"),
        docker_image=chains.DockerImage(
            base_image=chains.BasetenImage.PY311,
            pip_requirements=[_TRUSS_OVERRIDE, "httpx>=0.27", "websockets>=12"],
        ),
        assets=chains.Assets(secret_keys=["baseten_api_key"]),
    )

    def __init__(
        self,
        stt: chains.DeployedServiceDescriptor = chains.depends(STTMock),
        llm: chains.DeployedServiceDescriptor = chains.depends(LLMMock),
        tts: chains.DeployedServiceDescriptor = chains.depends(TTSMock),
        context: chains.DeploymentContext = chains.depends_context(),
    ) -> None:
        # Strip Baseten's internal HTTP_PROXY env vars — they intercept outbound
        # WS connections from inside chainlet pods (same mitigation FDE applies).
        import os

        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ.pop(var, None)
        os.environ["NO_PROXY"] = "*"

        try:
            user_api_key = context.secrets["baseten_chain_api_key"]
        except KeyError as e:
            raise RuntimeError(
                "baseten_api_key secret is missing. Set it in the workspace and"
                " re-deploy. Used in the Authorization header for sibling calls."
            ) from e
        if not user_api_key or user_api_key == "***":
            raise RuntimeError("baseten_api_key secret is empty/placeholder.")

        # WS sibling URLs: prefer the cluster-local internal_ws_url (chain
        # hostname + cluster routing); fall back to ws_url. WS auth headers
        # are Authorization-only — see with_ws_auth_headers docstring for why
        # an explicit Host override breaks WS handshakes.
        stt_url = stt.internal_ws_url or stt.ws_url
        tts_url = tts.internal_ws_url or tts.ws_url
        if stt_url is None or tts_url is None:
            raise RuntimeError(
                "STT/TTS sibling chainlets must expose a WS URL "
                f"(stt_url={stt_url!r}, tts_url={tts_url!r})."
            )
        self._stt_url = stt_url
        self._tts_url = tts_url
        self._headers_stt = stt.with_ws_auth_headers(user_api_key)
        self._headers_tts = tts.with_ws_auth_headers(user_api_key)

        # HTTP sibling URL: target_url uses the workload-plane gateway hostname
        # for cluster-local routing; with_auth_headers adds the Host override
        # api-gateway needs to map gateway-host requests onto the chain.
        self._llm_url = llm.target_url
        self._headers_llm = llm.with_auth_headers(user_api_key)

        log.warning(
            "Resolved sibling URLs: stt=%s llm=%s tts=%s",
            self._stt_url,
            self._llm_url,
            self._tts_url,
        )

    async def run_remote(self, websocket: chains.WebSocketProtocol) -> None:
        import httpx
        import websockets as ws_client

        try:
            audio_in = await websocket.receive_bytes()

            # 1. STT: bytes -> text
            async with ws_client.connect(
                self._stt_url, additional_headers=self._headers_stt
            ) as sws:
                await sws.send(audio_in)
                resp = await sws.recv()
                text = json.loads(resp)["text"]
                log.warning("STTMock returned: %s", text)

            # 2. LLM: text -> completion
            async with httpx.AsyncClient(timeout=30, headers=self._headers_llm) as http:
                r = await http.post(self._llm_url, json={"prompt": text})
                r.raise_for_status()
                completion = r.json()["completion"]
                log.warning("LLMMock returned: %s", completion)

            # 3. TTS: text -> bytes
            async with ws_client.connect(
                self._tts_url, additional_headers=self._headers_tts
            ) as tws:
                await tws.send(json.dumps({}))  # mock config frame, ignored
                await tws.send(completion)
                audio_out = await tws.recv()
                assert isinstance(audio_out, bytes), (
                    f"TTSMock should return bytes, got {type(audio_out).__name__}"
                )
                log.warning("TTSMock returned %d bytes", len(audio_out))

            await websocket.send_bytes(audio_out)
        except Exception as e:
            log.exception("Orchestrator pipeline failed: %s", e)
            try:
                await websocket.send_text(json.dumps({"error": str(e)}))
            except Exception:
                pass

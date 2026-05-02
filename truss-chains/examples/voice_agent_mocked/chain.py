"""Voice Agent (Mocked) — Composable Chains centerpiece example.

A CPU-only mock of the FDE voice-agent chain, designed to validate routing
behavior end-to-end without the cost of GPU model deploys. Same shape as
``/Users/mattelim/Documents/fde/e2e-voice``:

- ``STTMock``     (``TrussChainlet``, WS-fronted): bytes ``→`` ``{"text": "text-{N}"}``
- ``LLMMock``     (``TrussChainlet``, HTTP):       ``{"prompt"}`` ``→`` ``{"completion": "echo: ..."}``
- ``TTSMock``     (``TrussChainlet``, WS-fronted): text  ``→`` ``text.encode()``
- ``Orchestrator`` (``ChainletBase``, WS entrypoint): a single round-trip wires
  STT ``→`` LLM ``→`` TTS using the same descriptor / runtime patterns the FDE
  chain uses (GraphQL oracle-id resolution + proxy-env strip).

Iteration loop:

    truss chains push --remote matte chain.py --watch

The orchestrator's ``run_remote`` reads bytes from the entrypoint WS, drives
the pipeline, and sends back the synthesized bytes. Useful as a regression
fixture for any baseten-local change that touches chain-internal routing.
"""

import json
import logging
from urllib.parse import urlparse

import requests

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


# ----- URL resolution (workaround for the platform routing gap) --------------
#
# Same approach as the FDE chain: query the dashboard GraphQL with the
# user-supplied baseten_api_key to map chainlet name -> oracle.id, then build
# `model-<oracle.id>.api.baseten.co/...` URLs. Replace with the descriptor's
# native fields once the platform fix lands.

_GRAPHQL_QUERY = (
    "query Chain($id: String!) {\n"
    "  chain(id: $id) {\n"
    "    deployments {\n"
    "      id\n"
    "      chainlets {\n"
    "        name oracle { id } oracle_version { id transport_kind }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}"
)


def _chain_id_from_descriptor(desc: chains.DeployedServiceDescriptor) -> str:
    """Extract chain id from ``internal_url.hostname`` (``chain-<id>...``)."""
    hostname = desc.internal_url.hostname
    first = hostname.split(".", 1)[0]
    if not first.startswith("chain-"):
        raise RuntimeError(f"Unexpected internal_url hostname shape: {hostname!r}")
    return first[len("chain-") :]


def _chain_deployment_id_from_descriptor(desc: chains.DeployedServiceDescriptor) -> str:
    """Extract chain deployment id from the descriptor's gateway URL path."""
    parts = urlparse(desc.internal_url.gateway_run_remote_url).path.split("/")
    # Path: /deployment/<chain_dep_id>/chainlet/<chainlet_id>/run_remote
    return parts[parts.index("deployment") + 1]


def _resolve_oracles(
    api_key: str, chain_id: str, chain_deployment_id: str
) -> dict[str, dict]:
    r = requests.post(
        "https://app.baseten.co/graphql/",
        headers={
            "Authorization": f"Api-Key {api_key}",
            "Content-Type": "application/json",
        },
        json={"query": _GRAPHQL_QUERY, "variables": {"id": chain_id}},
        timeout=10,
    )
    r.raise_for_status()
    payload = r.json()
    chain_data = (payload.get("data") or {}).get("chain") or {}
    deployments = chain_data.get("deployments") or []
    target = next((d for d in deployments if d["id"] == chain_deployment_id), None)
    if target is None:
        raise RuntimeError(
            f"GraphQL has no deployment {chain_deployment_id} on chain {chain_id}"
        )
    return {
        c["name"]: {
            "oracle_id": c["oracle"]["id"],
            "oracle_version_id": c["oracle_version"]["id"],
            "transport": c["oracle_version"]["transport_kind"],
        }
        for c in target["chainlets"]
    }


def _model_url(oracle_id: str, oracle_version_id: str, scheme: str, path: str) -> str:
    """Build a per-chainlet sibling URL using the published-deployment shape.

    ``<scheme>://model-<oracle_id>.api.baseten.co/deployment/<oracle_version_id>/<path>``

    Cluster-internally this shape works for both draft and published chainlets
    (verified empirically on chain ``2328913l``); the equivalent
    ``/development/<path>`` alias is unnecessary. Using the deployment shape
    uniformly keeps URLs stable across draft → published lifecycles.
    """
    return (
        f"{scheme}://model-{oracle_id}.api.baseten.co"
        f"/deployment/{oracle_version_id}/{path}"
    )


# ----- Orchestrator: WebSocket entrypoint ------------------------------------


@chains.mark_entrypoint("VoiceAgentMocked")
class Orchestrator(chains.ChainletBase):
    """Single-round-trip WS entrypoint: bytes in -> bytes out, exercising the
    same STT/LLM/TTS sibling-call patterns as the FDE chain."""

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
            user_api_key = context.secrets["baseten_api_key"]
        except KeyError as e:
            raise RuntimeError(
                "baseten_api_key secret is missing. Set it in the workspace and"
                " re-deploy. Used to resolve sibling oracle IDs via dashboard"
                " GraphQL until the platform exposes them via dynamic_chainlet_config."
            ) from e
        if not user_api_key or user_api_key == "***":
            raise RuntimeError("baseten_api_key secret is empty/placeholder.")

        chain_id = _chain_id_from_descriptor(stt)
        chain_deployment_id = _chain_deployment_id_from_descriptor(stt)
        oracles = _resolve_oracles(user_api_key, chain_id, chain_deployment_id)
        log.warning(
            "Resolved oracle map for chain=%s deployment=%s: %s",
            chain_id,
            chain_deployment_id,
            oracles,
        )

        self._stt_url = _model_url(
            oracles["STTMock"]["oracle_id"],
            oracles["STTMock"]["oracle_version_id"],
            "wss",
            "websocket",
        )
        self._llm_url = _model_url(
            oracles["LLMMock"]["oracle_id"],
            oracles["LLMMock"]["oracle_version_id"],
            "https",
            "predict",
        )
        self._tts_url = _model_url(
            oracles["TTSMock"]["oracle_id"],
            oracles["TTSMock"]["oracle_version_id"],
            "wss",
            "websocket",
        )
        self._headers = {"Authorization": f"Api-Key {user_api_key}"}

    async def run_remote(self, websocket: chains.WebSocketProtocol) -> None:
        import httpx
        import websockets as ws_client

        try:
            audio_in = await websocket.receive_bytes()

            # 1. STT: bytes -> text
            async with ws_client.connect(
                self._stt_url, additional_headers=self._headers
            ) as sws:
                await sws.send(audio_in)
                resp = await sws.recv()
                text = json.loads(resp)["text"]
                log.warning("STTMock returned: %s", text)

            # 2. LLM: text -> completion
            async with httpx.AsyncClient(timeout=30, headers=self._headers) as http:
                r = await http.post(self._llm_url, json={"prompt": text})
                r.raise_for_status()
                completion = r.json()["completion"]
                log.warning("LLMMock returned: %s", completion)

            # 3. TTS: text -> bytes
            async with ws_client.connect(
                self._tts_url, additional_headers=self._headers
            ) as tws:
                await tws.send(json.dumps({}))  # mock config frame, ignored
                await tws.send(completion)
                audio_out = await tws.recv()
                log.warning("TTSMock returned %d bytes", len(audio_out))

            await websocket.send_bytes(audio_out)
        except Exception as e:
            log.exception("Orchestrator pipeline failed: %s", e)
            try:
                await websocket.send_text(json.dumps({"error": str(e)}))
            except Exception:
                pass

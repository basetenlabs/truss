"""Composable Chains — Project 1 deployable demo.

Exercises **every** new public surface introduced by Project 1:

  Helpers on ``DeployedServiceDescriptor`` (`truss_chains.public_types`):
    - ``target_url``               (property)
    - ``ws_url``                   (property)
    - ``internal_ws_url``          (property)
    - ``with_auth_headers(api_key)`` (method)

  Public submodule ``truss_chains.runtime``:
    - ``get_service(name)``
    - ``list_services()``

The chain has two chainlets:

* ``Echo`` — trivial dependency, returns input uppercased.
* ``Caller`` — entrypoint. For each request it:
    1. calls ``Echo`` via the framework's auto-generated ``StubBase``
       (existing path),
    2. resolves ``Echo``'s descriptor *both* via
       ``DeploymentContext.get_service_descriptor`` (typed-chain path) and
       via ``truss_chains.runtime.get_service`` (public-runtime path),
    3. inspects every helper on each descriptor,
    4. enumerates all siblings via ``truss_chains.runtime.list_services``,
    5. issues a raw ``httpx`` POST to the URL the helpers resolved to.

Every step emits ``logging.info`` describing what's being exercised and why.
The pod logs (visible via the Baseten UI / CLI) read like a guided tour of
Project 1's API surface. The HTTP response also surfaces all the helper
outputs so an external invoker can see the same information.

Both chainlets are CPU-only (1 vCPU / 512Mi each), so this pushes quickly.
"""

import logging
from typing import Optional

import httpx
import pydantic

import truss_chains as chains
from truss_chains import runtime  # NEW public submodule from Project 1

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Echo — trivial dependency
# ---------------------------------------------------------------------------


_TRUSS_OVERRIDE = (
    # Override the published `truss_chains` with the Project 1 branch so
    # `truss_chains.runtime` (and the new descriptor helpers) are available
    # in every chainlet pod. Drop this once Project 1 is released to PyPI.
    "truss @ git+https://github.com/basetenlabs/truss.git@matte/composable-chains-1-runtime"
)


class Echo(chains.ChainletBase):
    """Returns its input uppercased. Exists solely so ``Caller`` has a sibling
    to discover and exercise the helpers against."""

    remote_config = chains.RemoteConfig(
        compute=chains.Compute(cpu_count=1, memory="512Mi"),
        docker_image=chains.DockerImage(pip_requirements=[_TRUSS_OVERRIDE]),
    )

    async def run_remote(self, text: str) -> str:
        return text.upper()


# ---------------------------------------------------------------------------
# Output models — surface all the helper values to the external invoker
# ---------------------------------------------------------------------------


class DescriptorSnapshot(pydantic.BaseModel):
    """Snapshot of every value the Project 1 helpers expose for a single
    ``DeployedServiceDescriptor``. We deliberately redact the Authorization
    header *value* so the response transcript doesn't leak the chain's API
    key — the key shape is interesting, the bytes are not."""

    name: str
    display_name: str
    predict_url: Optional[str]
    internal_url_gateway: Optional[str]
    internal_url_hostname: Optional[str]
    target_url: str  # NEW Project 1 property
    ws_url: Optional[str]  # NEW Project 1 property
    internal_ws_url: Optional[str]  # NEW Project 1 property
    auth_header_keys: list[str]  # keys present in with_auth_headers(...) output
    auth_header_host: Optional[str]  # Host header (chain hostname; not sensitive)
    auth_header_authorization_format: str  # e.g. "Api-Key <redacted>"


class CallerOutput(pydantic.BaseModel):
    """Side-by-side view of both invocation paths plus a guided tour of the
    helpers' return values."""

    via_stub: str
    via_byo_httpx: str
    match: bool

    # `context.get_service_descriptor("Echo")` path (typed-chain)
    descriptor_via_context: DescriptorSnapshot
    # `truss_chains.runtime.get_service("Echo")` path (public-runtime)
    descriptor_via_runtime: DescriptorSnapshot
    # `truss_chains.runtime.list_services()` — all siblings in the chain pod
    runtime_siblings: list[str]


def _snapshot(
    desc: chains.DeployedServiceDescriptor, api_key: str, source: str
) -> DescriptorSnapshot:
    """Build a snapshot, logging each helper invocation as it runs.

    ``source`` is just a label for the log line so the operator can tell
    apart the typed-context and public-runtime descriptor sources.
    """
    logger.info(
        "[%s] descriptor.name=%r display_name=%r", source, desc.name, desc.display_name
    )
    logger.info("[%s] descriptor.predict_url=%s", source, desc.predict_url)
    logger.info(
        "[%s] descriptor.internal_url=%s",
        source,
        f"{desc.internal_url}" if desc.internal_url else None,
    )

    # NEW Project 1 helper: target_url. Mirrors BasetenSession's selection
    # logic — prefer internal_url's gateway URL when present, fall back to
    # predict_url. Saves callers from re-implementing the precedence rule.
    target = desc.target_url
    logger.info(
        "[%s] HELPER target_url -> %s (preferring internal_url over predict_url)",
        source,
        target,
    )

    # NEW Project 1 helper: ws_url. predict_url with the http(s)://
    # scheme rewritten to ws(s)://. Returns None when predict_url is unset.
    ws = desc.ws_url
    logger.info("[%s] HELPER ws_url -> %s (predict_url with https→wss)", source, ws)

    # NEW Project 1 helper: internal_ws_url. Same scheme rewrite, but for
    # the cluster-local internal_url path.
    iws = desc.internal_ws_url
    logger.info(
        "[%s] HELPER internal_ws_url -> %s (internal_url with https→wss)", source, iws
    )

    # NEW Project 1 helper: with_auth_headers(api_key). Builds the dict
    # BasetenSession would build internally — Authorization always, Host
    # when internal_url is set. Saves callers from hand-rolling either.
    headers = desc.with_auth_headers(api_key)
    logger.info(
        "[%s] HELPER with_auth_headers -> keys=%s "
        "(Authorization always; Host iff internal_url is set)",
        source,
        sorted(headers.keys()),
    )

    # Reduce the headers dict to a non-sensitive shape before returning.
    auth_value = headers.get("Authorization", "")
    auth_scheme = auth_value.split(" ", 1)[0] if auth_value else ""
    return DescriptorSnapshot(
        name=desc.name,
        display_name=desc.display_name,
        predict_url=desc.predict_url,
        internal_url_gateway=(
            desc.internal_url.gateway_run_remote_url if desc.internal_url else None
        ),
        internal_url_hostname=(
            desc.internal_url.hostname if desc.internal_url else None
        ),
        target_url=target,
        ws_url=ws,
        internal_ws_url=iws,
        auth_header_keys=sorted(headers.keys()),
        auth_header_host=headers.get("Host"),
        auth_header_authorization_format=f"{auth_scheme} <redacted>",
    )


# ---------------------------------------------------------------------------
# Caller — entrypoint that exercises every Project 1 surface
# ---------------------------------------------------------------------------


@chains.mark_entrypoint("Composable Chains BYO-Client Demo")
class Caller(chains.ChainletBase):
    """Exercises every Project 1 helper while comparing the framework stub
    path against a raw ``httpx`` BYO-client path."""

    remote_config = chains.RemoteConfig(
        compute=chains.Compute(cpu_count=1, memory="512Mi"),
        docker_image=chains.DockerImage(
            pip_requirements=[_TRUSS_OVERRIDE, "httpx>=0.27"]
        ),
    )

    def __init__(
        self,
        echo: Echo = chains.depends(Echo),
        context: chains.DeploymentContext = chains.depends_context(),
    ) -> None:
        self._echo = echo
        self._context = context

    async def run_remote(self, text: str) -> CallerOutput:
        logger.info("=" * 72)
        logger.info("Caller.run_remote(text=%r) — Project 1 helper tour", text)
        logger.info("=" * 72)

        # ----- path 1: framework stub (existing behavior) ------------------
        logger.info("[stub] await self._echo.run_remote(text)")
        via_stub = await self._echo.run_remote(text)
        logger.info("[stub] result=%r", via_stub)

        # ----- gather descriptors via BOTH access paths --------------------
        api_key = self._context.get_baseten_api_key()

        # Path A: typed-chain access via the DeploymentContext.
        # `get_service_descriptor` is what auto-generated stubs use under the
        # hood. Project 1 does NOT introduce this method — it pre-existed.
        # Including it for contrast against the new public runtime API.
        logger.info("[ctx] context.get_service_descriptor('Echo') — typed-chain path")
        desc_ctx = self._context.get_service_descriptor("Echo")
        snap_ctx = _snapshot(desc_ctx, api_key, source="ctx")

        # Path B: public-runtime access via the new submodule.
        # NEW Project 1 entry point: truss_chains.runtime.get_service. Reads
        # the same /etc/b10_dynamic_config/dynamic_chainlet_config file as
        # the framework, but exposed publicly so any Truss (typed or raw,
        # post Project 2) can call it without depending on framework code.
        logger.info("[runtime] truss_chains.runtime.get_service('Echo') — public path")
        desc_runtime = runtime.get_service("Echo")
        snap_runtime = _snapshot(desc_runtime, api_key, source="runtime")

        # Note: the typed path historically clears `predict_url` when an
        # `internal_url` is present (mutually exclusive). The new public
        # runtime API carries both — so when both URLs are in the dynamic
        # config you'll see snap_ctx.predict_url=None but
        # snap_runtime.predict_url=<the predict url>. The descriptor_via_*
        # comparison surfaces this explicitly.

        # ----- list all siblings via the new public API --------------------
        # NEW Project 1 entry point: truss_chains.runtime.list_services.
        # Returns all siblings registered in the chain (including this
        # chainlet itself). Returns {} outside a chain context — meaning
        # raw Trusses can use this to detect whether they're running
        # inside a chain at all.
        logger.info("[runtime] truss_chains.runtime.list_services()")
        siblings_map = runtime.list_services()
        siblings = sorted(siblings_map.keys())
        logger.info("[runtime] siblings discovered: %s", siblings)

        # ----- BYO httpx call using the helpers ---------------------------
        # Use the typed-context descriptor for the HTTP call (same wire as
        # the stub would use). The helpers' job is to produce a target URL
        # and headers equivalent to what BasetenSession.__init__ builds
        # internally — so the BYO call should land at the same chainlet
        # and return the same answer as the stub.
        #
        # Re-derive the actual headers here from the descriptor (rather than
        # plucking them off `snap_ctx`, which deliberately redacts the
        # Authorization value to keep secrets out of the response).
        headers = desc_ctx.with_auth_headers(api_key)
        logger.info(
            "[httpx] POST %s headers=%s body=%s",
            snap_ctx.target_url,
            sorted(headers.keys()),
            {"text": text},
        )
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                snap_ctx.target_url, headers=headers, json={"text": text}
            )
            response.raise_for_status()
            via_byo_httpx = response.json()
        logger.info("[httpx] response=%r", via_byo_httpx)

        match = via_stub == via_byo_httpx
        logger.info(
            "MATCH=%s (stub=%r vs byo_httpx=%r)", match, via_stub, via_byo_httpx
        )

        return CallerOutput(
            via_stub=via_stub,
            via_byo_httpx=via_byo_httpx,
            match=match,
            descriptor_via_context=snap_ctx,
            descriptor_via_runtime=snap_runtime,
            runtime_siblings=siblings,
        )


# ---------------------------------------------------------------------------
# Local smoke (no deployment needed)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Inspect the helper shapes against a fake descriptor under run_local.
    # Useful for development; for real invocation post-deploy, see invoke.py.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    fake_echo = chains.DeployedServiceDescriptor(
        name="Echo",
        display_name="Echo",
        options=chains.RPCOptions(),
        predict_url="https://chain-demo.api.baseten.co/.../echo/run_remote",
        internal_url=chains.DeployedServiceDescriptor.InternalURL(
            gateway_run_remote_url="https://wp.api.baseten.co/.../echo/run_remote",
            hostname="chain-demo.api.baseten.co",
        ),
    )
    with chains.run_local(
        secrets={chains.public_types.CHAIN_API_KEY_SECRET_NAME: "local-dev-key"},
        chainlet_to_service={"Echo": fake_echo},
    ):
        caller = Caller()
        desc = caller._context.get_service_descriptor("Echo")
        _snapshot(desc, api_key="local-dev-key", source="local")

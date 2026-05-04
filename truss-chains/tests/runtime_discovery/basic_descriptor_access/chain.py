"""Simple typed chain showing access to the new descriptor helpers from
``context.get_service_descriptor``."""

from typing import Optional

import pydantic

import truss_chains as chains


class CallerOutput(pydantic.BaseModel):
    echoed: str
    target_url: str
    ws_url: Optional[str]
    internal_ws_url: Optional[str]
    auth_headers: dict[str, str]


class Echo(chains.ChainletBase):
    remote_config = chains.RemoteConfig(
        compute=chains.Compute(cpu_count=1, memory="512Mi")
    )

    async def run_remote(self, text: str) -> str:
        return text


class Caller(chains.ChainletBase):
    remote_config = chains.RemoteConfig(
        compute=chains.Compute(cpu_count=1, memory="512Mi")
    )

    def __init__(
        self,
        echo: Echo = chains.depends(Echo),
        context: chains.DeploymentContext = chains.depends_context(),
    ) -> None:
        self._echo = echo
        self._context = context

    async def run_remote(self, text: str) -> CallerOutput:
        # Pull the descriptor for the typed dependency and exercise the new
        # helpers. In a real chain you would use `target_url` / `ws_url` to
        # build your own client; here we just surface them so the test
        # harness can assert against them.
        desc = self._context.get_service_descriptor("Echo")
        api_key = self._context.get_baseten_api_key()
        return CallerOutput(
            echoed=await self._echo.run_remote(text),
            target_url=desc.target_url,
            ws_url=desc.ws_url,
            internal_ws_url=desc.internal_ws_url,
            auth_headers=desc.with_auth_headers(api_key),
        )

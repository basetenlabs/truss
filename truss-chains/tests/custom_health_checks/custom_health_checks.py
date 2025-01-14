import time

import truss_chains as chains


class SyncChainlet(chains.ChainletBase):
    def is_ready(self) -> bool:
        return True

    def run_remote(self, text: str) -> str:
        return text


class AsyncChainlet(chains.ChainletBase):
    async def is_ready(self) -> bool:
        return True

    async def run_remote(self, text: str) -> str:
        return text


class CustomHealthChecks(chains.ChainletBase):
    """Calls various chainlets in JSON mode."""

    def __init__(
        self,
        sync_chainlet=chains.depends(SyncChainlet),
        async_chainlet=chains.depends(AsyncChainlet),
    ):
        self._sync_chainlet = sync_chainlet
        self._async_chainlet = async_chainlet
        time.sleep(10)

    def is_ready(self) -> bool:
        return False

    async def run_remote(self, text: str) -> tuple[str, str]:
        sync_result = self._sync_chainlet.run_remote(text)
        async_result = await self._async_chainlet.run_remote(text)
        return sync_result, async_result

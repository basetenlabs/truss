import truss_chains as chains


class SyncChainlet(chains.ChainletBase):
    def is_ready(self) -> bool:
        return True

    def run_remote(self, text: str) -> str:
        return text


class AsyncChainlet(chains.ChainletBase):
    async def run_remote(self, text: str) -> str:
        return text


class CustomHealthChecks(chains.ChainletBase):
    """Calls various chainlets using custom health checks."""

    def __init__(
        self,
        sync_chainlet=chains.depends(SyncChainlet),
        async_chainlet=chains.depends(AsyncChainlet),
    ):
        self._sync_chainlet = sync_chainlet
        self._async_chainlet = async_chainlet
        self._should_succeed_health_checks = True

    def is_ready(self) -> bool:
        return self._should_succeed_health_checks

    async def run_remote(self, fail: bool) -> str:
        if fail:
            self._should_succeed_health_checks = False
        else:
            self._should_succeed_health_checks = True
        sync_result = self._sync_chainlet.run_remote("hello")
        async_result = await self._async_chainlet.run_remote("world")
        return sync_result + async_result

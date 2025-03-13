import asyncio
import time

import truss_chains as chains


class Dependency(chains.ChainletBase):
    async def run_remote(self) -> bool:
        await asyncio.sleep(1)
        return True


class DependencySync(chains.ChainletBase):
    def run_remote(self) -> bool:
        time.sleep(1)
        return True


@chains.mark_entrypoint  # ("My Chain Name")
class TimeoutChain(chains.ChainletBase):
    def __init__(
        self,
        dep=chains.depends(Dependency, timeout_sec=0.5),
        dep_sync=chains.depends(DependencySync, timeout_sec=0.5),
    ):
        self._dep = dep
        self._dep_sync = dep_sync

    async def run_remote(self, use_sync: bool) -> None:
        if use_sync:
            result = self._dep_sync.run_remote()
            print(result)
        else:
            result = await self._dep.run_remote()
            print(result)

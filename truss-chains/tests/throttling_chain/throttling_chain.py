import asyncio

import truss_chains as chains


class Dependency(chains.ChainletBase):
    # Don't throttle in the chainlet.
    remote_config = chains.RemoteConfig(compute=chains.Compute(predict_concurrency=100))

    async def run_remote(self) -> bool:
        await asyncio.sleep(0.5)
        return True


@chains.mark_entrypoint
class ThrottlingChain(chains.ChainletBase):
    remote_config = chains.RemoteConfig(
        options=chains.ChainletOptions(enable_debug_logs=True)
    )

    def __init__(self, dep=chains.depends(Dependency, concurrency_limit=4)):
        self._dep = dep
        dep._async_semaphore_wrapper._log_interval_sec = 0.0

    async def run_remote(self, num_requests: int = 4) -> float:
        t0 = asyncio.get_event_loop().time()
        tasks = [
            asyncio.create_task(self._dep.run_remote()) for _ in range(num_requests)
        ]
        await asyncio.gather(*tasks)
        t1 = asyncio.get_event_loop().time()
        print(f"Batch of {num_requests} requests took {t1 - t0:.2f}s.")
        return t1 - t0

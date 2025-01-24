import truss_chains as chains


class CustomHealthChecks(chains.ChainletBase):
    """Implements custom health checks."""

    def __init__(self):
        self._should_succeed_health_checks = True

    def is_healthy(self) -> bool:
        return self._should_succeed_health_checks

    async def run_remote(self, fail: bool) -> str:
        if fail:
            self._should_succeed_health_checks = False
        else:
            self._should_succeed_health_checks = True
        return f"health checks will {'succeed' if self._should_succeed_health_checks else 'fail'}"

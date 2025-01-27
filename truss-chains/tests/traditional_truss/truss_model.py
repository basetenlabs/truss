import truss_chains as chains


class PassthroughModel(chains.ModelBase):
    remote_config: chains.RemoteConfig = chains.RemoteConfig(  # type: ignore
        compute=chains.Compute(4, "1Gi"), name="OverridePassthroughModelName"
    )

    def __init__(self):
        self._call_count = 0

    async def predict(self, call_count_increment: int) -> int:
        self._call_count += call_count_increment
        return self._call_count

import truss_chains as chains


class PassthroughModel(chains.ModelBase):
    remote_config: chains.RemoteConfig = chains.RemoteConfig(  # type: ignore
        compute=chains.Compute(4, "1Gi"),
        name="OverridePassthroughModelName",
        docker_image=chains.DockerImage(
            pip_requirements=[
                "truss==0.9.59rc2",
            ]
        ),
    )

    def __init__(self):
        self._call_count = 0

    async def run_remote(self, call_count_increment: int) -> int:
        self._call_count += call_count_increment
        return self._call_count

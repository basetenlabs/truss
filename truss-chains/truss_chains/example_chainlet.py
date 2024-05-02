import truss_chains as chains


class DummyGenerateData(chains.ChainletBase):
    remote_config = chains.RemoteConfig(docker_image=chains.DockerImage())

    def run(self) -> str:
        return "abc"


# Nesting the classes is a hack to make it *appear* like SplitText is from a different
# module.
class shared_chainlet:
    class DummySplitText(chains.ChainletBase):
        remote_config = chains.RemoteConfig(docker_image=chains.DockerImage())

        def run(self, data: str) -> list[str]:
            return [data[:2], data[2:]]


class DummyExample(chains.ChainletBase):
    remote_config = chains.RemoteConfig(docker_image=chains.DockerImage())

    def __init__(
        self,
        context: chains.DeploymentContext = chains.provide_context(),
        data_generator: DummyGenerateData = chains.provide(DummyGenerateData),
        splitter: shared_chainlet.DummySplitText = chains.provide(
            shared_chainlet.DummySplitText
        ),
    ) -> None:
        super().__init__(context)
        self._data_generator = data_generator
        self._data_splitter = splitter

    def run(self) -> list[str]:
        return self._data_splitter.run(self._data_generator.run())

import truss_chains as chains


class DummyGenerateData(chains.ChainletBase):
    def run_remote(self) -> str:
        return "abc"


# Nesting the classes is a hack to make it *appear* like SplitText is from a different
# module.
class shared_chainlet:
    class DummySplitText(chains.ChainletBase):
        def run_remote(self, data: str) -> list[str]:
            return [data[:2], data[2:]]


class DummyExample(chains.ChainletBase):
    def __init__(
        self,
        data_generator: DummyGenerateData = chains.depends(DummyGenerateData),
        splitter: shared_chainlet.DummySplitText = chains.depends(
            shared_chainlet.DummySplitText
        ),
        context: chains.DeploymentContext = chains.depends_context(),
    ) -> None:
        self._data_generator = data_generator
        self._data_splitter = splitter
        self._context = context

    def run_remote(self) -> list[str]:
        return self._data_splitter.run_remote(self._data_generator.run_remote())

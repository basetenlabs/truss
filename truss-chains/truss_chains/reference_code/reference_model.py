import truss_chains as chains


class HelloWorld(chains.ModelBase):
    def __init__(self, context: chains.DeploymentContext = chains.depends_context()):
        self._call_count = 0

    def predict(self, call_count_increment: int) -> int:
        self._call_count += call_count_increment
        return self._call_count

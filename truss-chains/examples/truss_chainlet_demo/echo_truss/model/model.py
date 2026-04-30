"""Plain Truss — uppercase-echoes the input. Used by `truss_chainlet_demo`
to demonstrate that an existing Truss directory can be a chain member via
``chains.TrussChainlet`` without any rewrite as ``ChainletBase``."""


class Model:
    def __init__(self, **kwargs) -> None:
        pass

    def predict(self, request: dict) -> dict:
        text = str(request.get("text", ""))
        return {"out": text.upper()}

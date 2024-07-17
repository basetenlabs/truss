from .engine import Engine


class TrussExtension:
    def __init__(self, *args, **kwargs):
        self._engine = Engine(*args, **kwargs)

    def model_override(self):
        """Return a model object."""
        return self._engine

    def model_args(self) -> dict:
        """Return args to supply as input to Model class' __init__"""
        return {"engine": self._engine}

    def load(self):
        self._engine.load()

from .engine import Engine


# TODO(pankaj) Define an ABC base class for this. That baseclass should live in
# a new, smaller truss sub-library, perhaps called `truss-runtime`` for inclusion
# in Truss runtime. Once we have that sublibrary, we should define the Extension
# baseclass there and derive TrussExtension below from it.
class Extension:
    def __init__(self, *args, **kwargs):
        self._engine = Engine(*args, **kwargs)

    def model_override(self):
        """Return a model object.

        This is used if model.py is omitted, which is allowed when using trt_llm.
        """
        return self._engine

    def model_args(self) -> dict:
        """Return args to supply as input to Model class' __init__

        Model class can use this to invoke the trt_llm engine.

        Returned engine is a typical Truss model class that provides a predict
        function. The predict function is async and returns an async generator.
        """
        return {"engine": self._engine}

    def load(self):
        self._engine.load()

from .engine import Engine

# TODO(pankaj) Define an ABC base class for this. That baseclass should live in
# a new, smaller truss sub-library, perhaps called `truss-runtime`` for inclusion
# in Truss runtime. Once we have that sub-library, we should define the Extension
# base class there and derive Extension class below from it.
#
# That base class would look like:
# class TrussExtension(ABC):
#     @abstracemethod
#     def model_override(self):
#         pass

#     @abstractmethod
#     def model_args(self) -> dict:
#         pass


#     @abstractmethod
#     def load(self) -> dict:
#         pass
class Extension:
    """
    trt_llm truss extension.

    Provides model_args to supply to model class, which contain the trtllm
    engine that corresponds to provided config.

    This extension also provides a full replacement of the model class, which is
    to be used if user doesn't supply it. This may be desired behavior in many
    cases where users want to just go by config and don't want to do any pre or
    post-processing.
    """

    def __init__(self, *args, **kwargs):
        self._engine = Engine(*args, **kwargs)

    def model_override(self):
        """Return a model object.

        This is used if model.py is omitted, which is allowed when using trt_llm.
        """
        return self._engine

    def model_args(self) -> dict:
        """Return args to supply as input to Model class' __init__ method.

        Model class can use this to invoke the trt_llm engine.

        Returned engine is a typical Truss model class that provides a predict
        function. The predict function is async and returns an async generator.
        """
        return {"engine": self._engine}

    def load(self):
        self._engine.load()

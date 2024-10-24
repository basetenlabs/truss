"""
The `Model` class is an interface between the ML model that you're packaging and the model
server that you're running it on.

The main methods to implement here are:
* `load`: runs exactly once when the model server is spun up or patched and loads the
   model onto the model server. Include any logic for initializing your model, such
   as downloading model weights and loading the model into memory.
* `predict`: runs every time the model server is called. Include any logic for model
  inference and return the model output.

See https://truss.baseten.co/quickstart for more.
"""

import pydantic


class DummyData(pydantic.BaseModel):
    model_config = pydantic.ConfigDict()

    foo: str
    bar: int


class Model:
    def __init__(self, **kwargs):
        self._model = DummyData(foo="bla", bar=123)

    def predict(self, model_input):
        return self._model.model_dump_json(indent=4)

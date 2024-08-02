"""
The `Model` class is allows you to customize the behavior of your TensorRT-LLM engine.

The main methods to implement here are:
* `load`: runs exactly once when the model server is spun up or patched and loads the
   model onto the model server. Include any logic for initializing your model server.
* `predict`: runs every time the model server is called. Include any logic for model
  inference and return the model output.

See https://docs.baseten.co/performance/engine-builder-customization for more.
"""


class Model:
    def __init__(self, trt_llm, **kwargs):
        # Uncomment the following to get access
        # to various parts of the Truss config.

        # self._data_dir = kwargs["data_dir"]
        # self._config = kwargs["config"]
        # self._secrets = kwargs["secrets"]
        self._engine = trt_llm["engine"]

    def load(self):
        # Load
        pass

    async def predict(self, model_input):
        # Run model inference here
        return await self._engine.predict(model_input)

"""
The `Model` class is an interface between the ML model that you're packaging and the model
server that you're running it on.

The main method to implement here is:
* `predict`: runs every time the model server is called. Include any logic for model
  inference and return the model output.

See https://truss.baseten.co/quickstart for more.
"""

import truss_chains as baseten


class Model(baseten.ModelBase):
    # Configure resources for your model here.
    remote_config: baseten.RemoteConfig = baseten.RemoteConfig(name="{{ MODEL_NAME }}")

    def __init__(
        self, context: baseten.DeploymentContext = baseten.depends_context()
    ) -> None:
        # Access secrets via optional `context` variable.
        # Load model here and assign to self._model.
        self._model = None

    def predict(self, input_field: int) -> int:
        # Run model inference here
        return input_field

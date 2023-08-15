"""
This file is where the key logic for serving your model is defined.

The main methods to implement here are:
* `load`: In the `load` method, include any logic for initializing your model.
   This might include downloading model weights and loading the model into memory
* `predict`: This is where model inference happens

See https://truss.baseten.co/reference/structure for the full set of methods available
in this file.
"""


class Model:
    def __init__(self, **kwargs):
        # Uncomment the following to get access
        # to various parts of the Truss config.

        # self._data_dir = kwargs["data_dir"]
        # self._config = kwargs["config"]
        # self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        # Load your model here
        pass

    def predict(self, model_input):
        # Run model inference here
        return model_input

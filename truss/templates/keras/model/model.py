from typing import Any

import numpy as np
from tensorflow import keras


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        config = kwargs["config"]
        model_metadata = config["model_metadata"]
        self._model_binary_dir = model_metadata["model_binary_dir"]
        self._model = None

    def load(self):
        self._model = keras.models.load_model(
            str(self._data_dir / self._model_binary_dir)
        )

    def preprocess(self, model_input: Any) -> Any:
        """
        Incorporate pre-processing required by the model if desired here.

        These might be feature transformations that are tightly coupled to the model.
        """
        return model_input

    def postprocess(self, model_output: Any) -> Any:
        """
        Incorporate post-processing required by the model if desired here.
        """
        return model_output

    def predict(self, model_input: Any) -> Any:
        model_output = {}
        inputs = np.array(model_input)
        result = self._model.predict(inputs).tolist()
        model_output["predictions"] = result
        return model_output

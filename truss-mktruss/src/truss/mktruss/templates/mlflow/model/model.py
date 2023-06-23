from typing import Any

import mlflow
import numpy as np


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        config = kwargs["config"]
        model_metadata = config["model_metadata"]
        self._model_binary_dir = model_metadata["model_binary_dir"]
        self._model = None

    def load(self):
        model_binary_dir_path = self._data_dir / self._model_binary_dir
        self._model = mlflow.pyfunc.load_model(model_binary_dir_path / "model")

    def predict(self, model_input: Any) -> Any:
        model_output = {}
        inputs = np.array(model_input)
        result = self._model.predict(inputs).tolist()
        model_output["predictions"] = result
        return model_output

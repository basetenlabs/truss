from typing import Any

from constants import COUNT


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        print(COUNT)

    def predict(self, model_input: Any) -> Any:
        model_output = {}
        # Invoke model on model_input and calculate predictions here.
        model_output["predictions"] = COUNT
        return model_output

import pickle
from typing import Any


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._pipeline = None

    def load(self):
        with open(self._data_dir / "pipeline.cpick", "rb") as f:
            self._pipeline = pickle.load(f)

    def predict(self, model_input: Any) -> Any:
        return self._pipeline(model_input)

import time
from typing import Any, Dict, List


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        print("Starting loading over 20 seconds.")
        time.sleep(20)

    def predict(self, model_input: Any) -> Dict[str, List]:
        print("Taking 20 seconds to predict")
        time.sleep(20)

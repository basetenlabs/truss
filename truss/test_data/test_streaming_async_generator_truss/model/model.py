from typing import Any, Dict, List


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        # Load model here and assign to self._model.
        pass

    async def predict(self, model_input: Any) -> Dict[str, List]:
        # Invoke model on model_input and calculate predictions here.
        for i in range(5):
            yield str(i)

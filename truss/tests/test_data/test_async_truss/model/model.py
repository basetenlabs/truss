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

    async def preprocess(self, model_input: Dict):
        return {"preprocess_value": "value", **model_input}

    async def predict(self, model_input: Any) -> Dict[str, List]:
        return model_input

    async def postprocess(self, response: Dict):
        return {"postprocess_value": "value", **response}

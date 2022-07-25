from typing import Dict, List

from model.dalle_mini_model import DallEModel


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._model = None

    def load(self):
        # Load model here and assign to self._model.
        self._model = DallEModel()

    def preprocess(self, request: Dict) -> Dict:
        """
        Incorporate pre-processing required by the model if desired here.

        These might be feature transformations that are tightly coupled to the model.
        """
        return request

    def postprocess(self, request: Dict) -> Dict:
        """
        Incorporate post-processing required by the model if desired here.
        """
        return request

    def predict(self, request: Dict) -> Dict[str, List]:
        response = {}
        inputs = request["inputs"]  # noqa
        # Invoke model and calculate predictions here.
        response["predictions"] = self._model.predict(inputs)
        return response

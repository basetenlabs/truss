from typing import Dict, List


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._model = None
        self.leak = []

    def load(self):
        print("loading")

        pass

    def preprocess(self, request: Dict) -> Dict:
        print("pre-process")
        self.leak.append(list(range(10000)))
        return request

    def postprocess(self, request: Dict) -> Dict:
        print("post-process")
        return request

    def predict(self, request: Dict) -> Dict[str, List]:
        print("predict")
        inputs = request["inputs"]  # noqa
        return {"predictions": [1, 3, 4, 5]}

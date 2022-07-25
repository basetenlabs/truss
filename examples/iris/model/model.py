from typing import Dict, List

MODEL_BASENAME = "model"
MODEL_EXTENSIONS = [".joblib", ".pkl", ".pickle"]


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        config = kwargs["config"]
        model_metadata = config["model_metadata"]
        self._supports_predict_proba = model_metadata["supports_predict_proba"]
        self._model_binary_dir = model_metadata["model_binary_dir"]
        self._model = None

    def load(self):
        import joblib

        model_binary_dir_path = self._data_dir / self._model_binary_dir
        paths = [
            (model_binary_dir_path / MODEL_BASENAME).with_suffix(model_extension)
            for model_extension in MODEL_EXTENSIONS
        ]
        model_file_path = next(path for path in paths if path.exists())
        self._model = joblib.load(model_file_path)

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
        inputs = request["inputs"]
        result = self._model.predict(inputs)
        response["predictions"] = result
        if self._supports_predict_proba:
            response["probabilities"] = self._model.predict_proba(inputs).tolist()
        return response

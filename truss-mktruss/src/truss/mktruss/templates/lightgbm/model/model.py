from typing import Any

MODEL_BASENAME = "model"
MODEL_EXTENSIONS = [".joblib", ".pkl", ".pickle"]


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        config = kwargs["config"]
        model_metadata = config["model_metadata"]
        # LightGBM does not implement a `predict_proba` function
        self._supports_predict_proba = False
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

    def predict(self, model_input: Any) -> Any:
        model_output = {}
        result = self._model.predict(model_input)
        model_output["predictions"] = result
        if self._supports_predict_proba:
            model_output["probabilities"] = self._model.predict_proba(
                model_input
            ).tolist()
        return model_output

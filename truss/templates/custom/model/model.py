from typing import Any


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        # Load model here and assign to self._model.
        pass

    def preprocess(self, model_input: Any) -> Any:
        """
        Incorporate pre-processing required by the model if desired here.

        These might be feature transformations that are tightly coupled to the model.
        """
        return model_input

    def postprocess(self, model_output: Any) -> Any:
        """
        Incorporate post-processing required by the model if desired here.
        """
        return model_output

    def predict(self, model_input: Any) -> Any:
        model_output = {}
        # Invoke model on model_input and calculate predictions here.
        model_output["predictions"] = []
        return model_output

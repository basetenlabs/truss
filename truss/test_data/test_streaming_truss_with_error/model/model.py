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

    def predict(self, model_input: Any) -> Dict[str, List]:
        def inner():
            for i in range(5):
                # Raise error partway through if throw_error is set
                if i == 3 and model_input.get("throw_error"):
                    raise Exception("error")
                yield str(i)

        return inner()

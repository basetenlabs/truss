import logging
import traceback
from typing import Any

import torch
from transformers import pipeline


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        config = kwargs["config"]
        self._config = config
        model_metadata = config["model_metadata"]
        self._transformer_config = model_metadata["transformer_config"]
        self._has_named_args = model_metadata["has_named_args"]
        self._has_hybrid_args = model_metadata["has_hybrid_args"]
        self._model = None

    def load(self):
        transformer_config = self._transformer_config.copy()
        if torch.cuda.is_available():
            transformer_config["device"] = 0

        self._model = pipeline(
            task=self._config["model_type"],
            **transformer_config,
        )

    def predict(self, model_input: Any) -> Any:
        model_output = {}
        instances = model_input

        with torch.no_grad():
            if self._has_named_args:
                result = [self._model(**instance) for instance in instances]
            elif self._has_hybrid_args:
                try:
                    result = []
                    for instance in instances:
                        prompt = instance.pop("prompt")
                        result.append(self._model(prompt, **instance))
                except (KeyError, AttributeError):
                    logging.error(traceback.format_exc())
                    model_output["error"] = {
                        "traceback": f'Expected request as an object with text in "prompt"\n{traceback.format_exc()}'
                    }
                    return model_output
            else:
                result = self._model(instances)
        model_output["predictions"] = result
        return model_output

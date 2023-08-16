from pathlib import Path

import yaml
from triton_model_wrapper import ModelWrapper


class TritonPythonModel:
    def __init__(self):
        self._model_repository_path = None
        self._config_path = None
        self._config = None
        self._model_wrapper: ModelWrapper = None

    def _instantiate_model_wrapper(self, triton_config: dict):
        self._model_repository_path = (
            Path(triton_config["model_repository"]) / triton_config["model_version"]
        )
        self._config_path = self._model_repository_path / "truss" / "config.yaml"
        with open(self._config_path, encoding="utf-8") as config_file:
            self._config = yaml.safe_load(config_file)
        self._model_wrapper: ModelWrapper = ModelWrapper(self._config)
        self._model_wrapper.instantiate()

    def initialize(self, args):
        self._instantiate_model_wrapper(args)
        self._model_wrapper.load()

    def execute(self, requests):
        return self._model_wrapper.predict(requests)

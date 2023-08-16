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
        """
        Instantiates the model wrapper class as well as the user-defined model class.

        Args:
            triton_config (dict): Triton configuration dictionary. This contains information about the model repository
                path and the model version.

        Returns:
            None
        """
        self._model_repository_path = (
            Path(triton_config["model_repository"]) / triton_config["model_version"]
        )
        self._config_path = self._model_repository_path / "truss" / "config.yaml"
        with open(self._config_path, encoding="utf-8") as config_file:
            self._config = yaml.safe_load(config_file)
        self._model_wrapper: ModelWrapper = ModelWrapper(self._config)
        self._model_wrapper.instantiate()

    def initialize(self, args: dict):
        """
        Instantiates the model wrapper class and loads the user's model. This function is called by Triton upon startup.

        Args:
            args (dict): Triton configuration dictionary. This contains information about the model repository path and
                the model version as well as all the information defined in the config.pbtxt.

        Returns:
            None
        """
        self._instantiate_model_wrapper(args)
        self._model_wrapper.load()

    def execute(self, requests):
        """
        Executes the user's model on Triton InferenceRequest objects. This function is called by Triton upon inference.

        Args:
            requests (InferenceRequest): Triton InferenceRequest object. This contains the input data for the model.

        Returns:
            InferenceResponse: Triton InferenceResponse object. This contains the output data from the model.
        """
        return self._model_wrapper.predict(requests)

import importlib
import inspect
import sys
from pathlib import Path
from typing import Dict

import triton_python_backend_utils as pb_utils
from shared.secrets_resolver import SecretsResolver
from transform import transform_pydantic_to_triton, transform_triton_to_pydantic
from utils.pydantic import inspect_pydantic_model
from utils.signature import signature_accepts_keyword_arg


class ModelWrapper:
    def __init__(self, config: Dict):
        self._config = config
        self._model = None
        self._input_type = None
        self._output_type = None
        self._input_type_fields = None
        self._output_type_fields = None

    def instantiate(self):
        """
        Instantiates the model class defined in the config.
        """
        model_directory_prefix = Path("/app/model/1/truss")
        data_dir = model_directory_prefix / "data"

        if "bundled_packages_dir" in self._config:
            bundled_packages_path = model_directory_prefix / "bundled_packages"
            if bundled_packages_path.exists():
                sys.path.append(str(bundled_packages_path))
        model_module_name = str(
            Path(self._config["model_class_filename"]).with_suffix("")
        )
        module = importlib.import_module(
            f"truss.{self._config['model_module_dir']}.{model_module_name}"
        )
        model_class = getattr(module, self._config["model_class_name"])

        input_type_name = self._config.get("input_type_name", "Input")
        output_type_name = self._config.get("output_type_name", "Output")
        self._input_type = getattr(module, input_type_name)
        self._input_type_fields = inspect_pydantic_model(self._input_type)
        self._output_type = getattr(module, output_type_name)
        self._output_type_fields = inspect_pydantic_model(self._output_type)

        model_class_signature = inspect.signature(model_class)
        model_init_params = {}
        if signature_accepts_keyword_arg(model_class_signature, "config"):
            model_init_params["config"] = self._config
        if signature_accepts_keyword_arg(model_class_signature, "data_dir"):
            model_init_params["data_dir"] = data_dir
        if signature_accepts_keyword_arg(model_class_signature, "secrets"):
            model_init_params["secrets"] = SecretsResolver.get_secrets(self._config)

        self._model = model_class(**model_init_params)

    def load(self):
        """
        Invokes the underlying model's load method.
        """
        if self._model is None:
            raise ValueError("Model not instantiated")

        self._model.load()

    def predict(self, requests: list):
        """
        Converts Triton requests to Pydantic objects, invokes the underlying model's predict method, and converts the
        outputs to Triton objects.
        """
        requests = transform_triton_to_pydantic(requests, self._input_type)
        try:
            outputs = self._model.predict(requests)  # type: ignore
        except Exception as e:
            raise pb_utils.TritonModelException(e)

        return transform_pydantic_to_triton(outputs)

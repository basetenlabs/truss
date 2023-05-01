from typing import Any

import torch
from torch import package


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        config = kwargs["config"]
        self._model_metadata = config["model_metadata"]
        self._model = None
        self._model_dtype = None
        self._device = torch.device(
            "cuda:0"
            if torch.cuda.is_available() and config["resources"]["use_gpu"]
            else "cpu"
        )

    def load(self):
        imp = package.PackageImporter(
            self._data_dir / "model" / self._model_metadata["torch_package_file"]
        )
        package_name = self._model_metadata["torch_model_package_name"]
        model_pickle_filename = self._model_metadata["torch_model_pickle_filename"]
        self._model = imp.load_pickle(package_name, model_pickle_filename)
        self._model_dtype = list(self._model.parameters())[0].dtype

    def predict(self, model_input: Any) -> Any:
        model_output = {}
        with torch.no_grad():
            inputs = torch.tensor(
                model_input, dtype=self._model_dtype, device=self._device
            )
            model_output["predictions"] = self._model(inputs).tolist()
            return model_output

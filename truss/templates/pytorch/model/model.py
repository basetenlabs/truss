from typing import Dict, List

import torch
from torch import package


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs['data_dir']
        config = kwargs['config']
        self._model_metadata = config['model_metadata']
        self._model = None
        self._model_dtype = None
        self._device = torch.device(
            "cuda:0" if torch.cuda.is_available() and config['resources']['use_gpu'] else "cpu"
        )

    def load(self):
        imp = package.PackageImporter(self._data_dir / 'model' / self._model_metadata['torch_package_file'])
        package_name = self._model_metadata['torch_model_package_name']
        model_pickle_filename = self._model_metadata['torch_model_pickle_filename']
        self._model = imp.load_pickle(package_name, model_pickle_filename)
        self._model_dtype = list(self._model.parameters())[0].dtype

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
        with torch.no_grad():
            inputs = torch.tensor(
                request["inputs"], dtype=self._model_dtype, device=self._device
            )
            response = {}
            response["predictions"] = self._model(inputs).tolist()
            return response

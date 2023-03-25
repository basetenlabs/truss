import threading
from typing import Any, Dict, List, Tuple

import torch

from .img_img_model import Model as ImgImgModel
from .inpaint_model import Model as InpaintModel
from .text_img_model import Model as TextImgModel


class Model:
    def __init__(self, **kwargs) -> None:
        self._models = {
            "img_img": ImgImgModel(**kwargs),
            "text_img": TextImgModel(**kwargs),
            "inpaint": InpaintModel(**kwargs),
        }
        self._current_prediction_model_sub_name = None
        self._prediction_lock = threading.Lock()

    def load(self):
        for model in self._models.values():
            model.load()

    def unload(self):
        torch.cuda.empty_cache()

    def preprocess(self, request: Dict) -> Dict:
        _, model = self._sub_model_for_request(request)
        if not hasattr(model, "preprocess"):
            return request

        return model.preprocess(request)

    def predict(self, request: Dict) -> Dict[str, List]:
        with self._prediction_lock:
            model_name, model = self._sub_model_for_request(request)
            if self._current_prediction_model_sub_name != model_name:
                self.unload()
                model.load()
                self._current_prediction_model_sub_name = model_name

            return model.predict(request)

    def postprocess(self, request: Dict) -> Dict:
        _, model = self._sub_model_for_request(request)
        if not hasattr(model, "postprocess"):
            return request

        return model.postprocess(request)

    def _sub_model_for_request(self, request: Dict) -> Tuple[str, Any]:
        model_sub_name = request["model_sub_name"]
        if model_sub_name not in self._models:
            raise ValueError(f"Unsupported sub model `{model_sub_name}`")

        return model_sub_name, self._models[model_sub_name]

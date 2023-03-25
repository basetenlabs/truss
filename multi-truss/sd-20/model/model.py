import threading
from typing import Any, Dict, List

import torch

from .img_img_model import Model as ImgImgModel
from .inpaint_model import Model as InpaintModel
from .text_img_model import Model as TextImgModel

MODEL_SUB_NAME_ARG = "model_sub_name"


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
        # for model in self._models.values():
        #     model.load()
        pass

    def unload(self):
        torch.cuda.empty_cache()

    def preprocess(self, request: Dict) -> Dict:
        model_name = request.pop(MODEL_SUB_NAME_ARG)
        model = self._sub_model_for_request(model_name)

        if not hasattr(model, "preprocess"):
            return {
                MODEL_SUB_NAME_ARG: model_name,
                "response": request,
            }

        preprocess_resp = model.preprocess(request)
        print(preprocess_resp)
        return {
            MODEL_SUB_NAME_ARG: model_name,
            "response": preprocess_resp,
        }

    def predict(self, request: Dict) -> Dict[str, List]:
        with self._prediction_lock:
            model_name = request.pop(MODEL_SUB_NAME_ARG)
            model = self._sub_model_for_request(model_name)
            if self._current_prediction_model_sub_name != model_name:
                self.unload()
                model.load()
                self._current_prediction_model_sub_name = model_name

            resp = model.predict(request["response"])
            wrapped_resp = {
                MODEL_SUB_NAME_ARG: model_name,
                "response": resp,
            }
            print(wrapped_resp)
            return wrapped_resp

    def postprocess(self, request: Dict) -> Dict:
        model_name = request.pop(MODEL_SUB_NAME_ARG)
        model = self._sub_model_for_request(model_name)
        if not hasattr(model, "postprocess"):
            return request

        return model.postprocess(request["response"])

    def _sub_model_for_request(self, model_sub_name: str) -> Any:
        if model_sub_name not in self._models:
            raise ValueError(f"Unsupported sub model `{model_sub_name}`")

        return self._models[model_sub_name]

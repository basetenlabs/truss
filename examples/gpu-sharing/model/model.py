import threading
import time
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
        for model_name, model in self._models.items():
            print(f"Loading model {model_name}")
            print(
                f"GPU memory before loading model {model_name}: {torch.cuda.memory_allocated()}"
            )
            model.load()
            print(
                f"GPU memory after loading model {model_name}: {torch.cuda.memory_allocated()}"
            )
            # Keep model in cpu memory, but evict from GPU memory
            model.standby()
            print(
                f"GPU memory after standing by {model_name}: {torch.cuda.memory_allocated()}"
            )
            print(f"Model {model_name} loaded successfully")

    def preprocess(self, request: Dict) -> Dict:
        model_name = request.pop(MODEL_SUB_NAME_ARG)
        model = self._sub_model_for_request(model_name)

        if not hasattr(model, "preprocess"):
            return {
                MODEL_SUB_NAME_ARG: model_name,
                "response": request,
            }

        preprocess_resp = model.preprocess(request)
        return {
            MODEL_SUB_NAME_ARG: model_name,
            "response": preprocess_resp,
        }

    def predict(self, request: Dict) -> Dict[str, List]:
        with self._prediction_lock:
            model_name = request.pop(MODEL_SUB_NAME_ARG)
            model = self._sub_model_for_request(model_name)
            if self._current_prediction_model_sub_name != model_name:
                start_time = time.perf_counter()
                if self._current_prediction_model_sub_name is not None:
                    current_model = self._models[
                        self._current_prediction_model_sub_name
                    ]
                    print(f"GPU memory before: {torch.cuda.memory_allocated()}")
                    current_model.standby()
                    print(f"GPU memory after unload: {torch.cuda.memory_allocated()}")
                model.load()
                print(f"GPU memory after load: {torch.cuda.memory_allocated()}")
                print(
                    f"Time taken for model load {int((time.perf_counter() - start_time) * 1000)} ms"
                )
                self._current_prediction_model_sub_name = model_name

            start_time = time.perf_counter()
            resp = model.predict(request["response"])
            print(
                f"Time taken for predict {int((time.perf_counter() - start_time) * 1000)} ms"
            )
            wrapped_resp = {
                MODEL_SUB_NAME_ARG: model_name,
                "response": resp,
            }
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

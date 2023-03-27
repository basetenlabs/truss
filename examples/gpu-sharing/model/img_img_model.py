from dataclasses import asdict
from typing import Dict

import torch
from diffusers import StableDiffusionImg2ImgPipeline

from .base64_utils import b64_to_pil, pil_to_b64

STABLE_DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"


class Model:
    def __init__(self, **kwargs) -> None:
        self._model = None
        self._memory_model = None

    def load(self):
        if self._memory_model is None:
            self._memory_model = StableDiffusionImg2ImgPipeline.from_pretrained(
                STABLE_DIFFUSION_MODEL_ID,
                torch_dtype=torch.float16,
            )
        self._model = self._memory_model.to("cuda")

    def standby(self):
        self._model = self._model.to("cpu")

    def preprocess(self, request: Dict) -> Dict:
        # Convert from base64
        if "image" in request:
            request["image"] = b64_to_pil(request["image"]).convert("RGB")
        return request

    def postprocess(self, request: Dict) -> Dict:
        # Convert to base64
        request.images = [pil_to_b64(img) for img in request.images]
        return asdict(request)

    def predict(self, request: Dict):
        response = self._model(**request)
        return response

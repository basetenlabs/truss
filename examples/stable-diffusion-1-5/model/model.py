from dataclasses import asdict
from typing import Dict

import torch
from base64_utils import pil_to_b64
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

STABLE_DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"


class Model:
    def __init__(self, **kwargs) -> None:
        self._model = None

    def load(self):
        scheduler = EulerDiscreteScheduler.from_pretrained(
            STABLE_DIFFUSION_MODEL_ID,
            subfolder="scheduler",
        )
        self._model = StableDiffusionPipeline.from_pretrained(
            STABLE_DIFFUSION_MODEL_ID,
            scheduler=scheduler,
            torch_dtype=torch.float16,
        )
        self._model.unet.set_use_memory_efficient_attention_xformers(True)
        self._model = self._model.to("cuda")

    def postprocess(self, request: Dict) -> Dict:
        # Convert to base64
        request.images = [pil_to_b64(img) for img in request.images]
        return asdict(request)

    def predict(self, request: Dict):
        response = self._model(**request)
        return response

import base64
from dataclasses import asdict
from io import BytesIO
from typing import Dict

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

STABLE_DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"
BASE64_PREAMBLE = "data:image/png;base64,"


def pil_to_b64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return BASE64_PREAMBLE + str(img_str)[2:-1]


def b64_to_pil(b64_str):
    return Image.open(BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, ""))))


class Model:
    def __init__(self, **kwargs) -> None:
        self._model = None

    def load(self):
        self._model = StableDiffusionImg2ImgPipeline.from_pretrained(
            STABLE_DIFFUSION_MODEL_ID,
            torch_dtype=torch.float16,
        )
        self._model = self._model.to("cuda")

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

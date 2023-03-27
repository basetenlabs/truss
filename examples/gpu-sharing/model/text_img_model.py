import base64
from io import BytesIO
from typing import Dict, List

import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

MODEL_ID = "stabilityai/stable-diffusion-2-1-base"


class Model:
    def __init__(self, **kwargs) -> None:
        self._model = None
        self._memory_model = None

    def load(self):
        if self._memory_model is None:
            scheduler = EulerDiscreteScheduler.from_pretrained(
                MODEL_ID, subfolder="scheduler"
            )
            self._memory_model = StableDiffusionPipeline.from_pretrained(
                MODEL_ID, scheduler=scheduler, torch_dtype=torch.float16
            )
        self._model = self._memory_model.to("cuda")
        self._model.enable_xformers_memory_efficient_attention()

    def standby(self):
        self._model = self._model.to("cpu")

    def convert_to_b64(self, image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_b64

    @torch.inference_mode()
    def predict(self, request: Dict) -> Dict[str, List]:
        prompt = request.pop("prompt")
        results = []
        try:
            output = self._model(prompt=prompt, return_dict=False, **request)

            for image in output[0]:
                b64_results = self.convert_to_b64(image)
                results.append(b64_results)

        except Exception as exc:
            return {"status": "error", "data": None, "message": str(exc)}

        return {"status": "success", "data": results, "message": None}

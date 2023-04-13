from dataclasses import asdict
from typing import Dict, List

import torch
from base64_utils import b64_to_pil, pil_to_b64
from controlnet_aux import HEDdetector
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        # Load model here and assign to self._model.
        controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-scribble",
            torch_dtype=torch.float16,
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch.float16,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        self._model = pipe

    def postprocess(self, request: Dict) -> Dict:
        # Convert PIL to Base64
        request.images = [pil_to_b64(image) for image in request.images]
        return asdict(request)

    def predict(self, request: Dict) -> Dict[str, List]:
        # Invoke model and calculate predictions here.
        input_image = b64_to_pil(request["image"])
        hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")
        image = hed(input_image, scribble=True)
        response_image = self._model(request["prompt"], image, num_inference_steps=20)
        return response_image

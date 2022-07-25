from io import BytesIO
from typing import Dict, List

import clip
import numpy as np
import requests
import torch
from PIL import Image

DEFAULT_CLIP_MODEL = "ViT-B/32"


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._model = None
        self._device = None
        self._model_preprocesser = None

    def load(self):
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model, self._model_preprocesser = clip.load(
            DEFAULT_CLIP_MODEL, device=self._device
        )

    def preprocess(self, request: Dict) -> Dict:
        for instance in request["instances"]:
            self._map_image_url_to_array_in_instance(instance)
        return request

    def postprocess(self, request: Dict) -> Dict:
        return request

    def predict(self, request: Dict) -> Dict[str, List]:
        try:
            return {
                "predictions": [
                    self._predict_single(instance) for instance in request["instances"]
                ]
            }
        except Exception as e:
            raise Exception("Failed to predict") from e

    def _predict_single(self, instance: Dict):
        image = (
            self._model_preprocesser(
                Image.fromarray(
                    np.array(instance["image"], ndmin=3, dtype=np.dtype("uint8"))
                )
            )
            .unsqueeze(0)
            .to(self._device)
        )
        text = clip.tokenize(instance["labels"]).to(self._device)
        with torch.no_grad():
            self._model.encode_image(image)
            self._model.encode_text(text)
            logits_per_image, logits_per_text = self._model(image, text)
            [probs] = logits_per_image.softmax(dim=-1).cpu().numpy()
            return dict(zip(instance["labels"], probs))

    def _map_image_url_to_array_in_instance(self, instance: Dict) -> Dict:
        if "image" not in instance and "image_url" in instance:
            image_url = instance["image_url"]
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            instance["image"] = np.asarray(image)

from io import BytesIO
from typing import Dict, List

import numpy as np
import requests
import tensorflow as tf
from PIL import Image

TARGET_SIZE = [224, 224]


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._model = None

    def load(self):
        self._model = tf.keras.applications.resnet50.ResNet50(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
        )

    def preprocess(self, request: Dict) -> Dict:
        if "image_url" in request:
            image_url = request["image_url"]
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            scaled_image = image.resize(TARGET_SIZE, resample=Image.BICUBIC)
            rgb_image = scaled_image.convert("RGB")
            image = np.asarray(rgb_image) / 255
            return {"inputs": [image]}
        return request

    def predict(self, request: Dict) -> Dict[str, List]:
        response = {}
        inputs = request["inputs"]
        response["predictions"] = self._predict_single(inputs[0])
        return response

    def _predict_single(self, input):
        img = np.array(input).reshape(1, 224, 224, 3)
        return self._model.predict(img)

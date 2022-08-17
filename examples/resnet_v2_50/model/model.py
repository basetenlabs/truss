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
        self._labels_map = None

    def load(self):
        self._model = tf.saved_model.load(self._data_dir / "model")

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
        images = request["inputs"]
        # Convert input to batched tensor
        images = np.asarray(images)
        input_tensor = tf.convert_to_tensor(images)
        input_tensor = tf.cast(input_tensor, dtype=tf.float32)
        # Execute model function on batch
        model_fn = self._model.signatures["serving_default"]
        response["predictions"] = model_fn(input_tensor)["keras_layer"].numpy().tolist()
        return response

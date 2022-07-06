import base64
import urllib
from io import BytesIO
from typing import Dict, List

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

HUB_HANDLE = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
IMG_SIZE_CONSTRAINT = (256, 256)  # Recommended to keep it at 256.


class Model:
    def __init__(self) -> None:
        self._model = None

    def load(self):
        self._model = hub.load(HUB_HANDLE)

    def predict(self, request: Dict) -> Dict[str, List]:
        return {
            'predictions': [self._predict_single(instance) for instance in request['inputs']]
        }

    def _predict_single(self, instance: Dict) -> str:
        """Given a content image and a style image, returns a style transferred image
        as a base64 encoded string."""
        content_image_url = instance['content_image_url']
        style_image_url = instance['style_image_url']

        content_image = Image.open(urllib.request.urlopen(content_image_url)).convert('RGB')
        style_image = Image.open(urllib.request.urlopen(style_image_url)).convert('RGB')

        content_image = np.asarray(content_image).astype(np.float32)[np.newaxis, ...] / 255.
        style_image = np.asarray(style_image).astype(np.float32)[np.newaxis, ...] / 255.

        content_image = tf.image.resize(content_image, IMG_SIZE_CONSTRAINT, preserve_aspect_ratio=True)
        style_image = tf.image.resize(style_image, IMG_SIZE_CONSTRAINT)
        output = self._model(tf.constant(content_image), tf.constant(style_image))
        output = np.squeeze(output)

        output_image = Image.fromarray((output * 255).astype(np.uint8))
        fp = BytesIO()
        output_image.save(fp, format='png')
        fp.seek(0)
        return base64.b64encode(fp.read()).decode('utf-8')

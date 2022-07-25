import base64
import logging
import urllib
from io import BytesIO
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import models

logger = logging.getLogger(__name__)

PYTORCH_IMAGE_MEAN = [0.485, 0.456, 0.406]
PYTORCH_IMAGE_STD = [0.229, 0.224, 0.225]


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._model = None
        self._device = None
        self._model_preprocesser = None

    def load(self):
        self._model = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def preprocess(self, request: Dict) -> Dict:
        return request

    def postprocess(self, request: Dict) -> Dict:
        return request

    def predict(self, request: Dict) -> Dict[str, List]:
        return [self._predict_single(instance) for instance in request["instances"]]

    def _predict_single(self, instance: Dict):
        image_url = instance.get("image_url")
        img = Image.open(urllib.request.urlopen(image_url))
        # adjust resize for performance/speed trade-off
        image_transform = T.Compose(
            [
                T.Resize(512),
                T.ToTensor(),
                T.Normalize(mean=PYTORCH_IMAGE_MEAN, std=PYTORCH_IMAGE_STD),
            ]
        )
        network_input = image_transform(img).unsqueeze(0).to(self._device)
        network_output = self._model.to(self._device)(network_input)["out"]
        class_predictions = (
            torch.argmax(network_output.squeeze(), dim=0).detach().cpu().numpy()
        )
        rgb_ndarray = decode_segmap(class_predictions)
        output_image = Image.fromarray(rgb_ndarray)
        fp = BytesIO()
        output_image.save(fp, format="png")
        fp.seek(0)
        return base64.b64encode(fp.read()).decode("utf-8")


def decode_segmap(image, nc=21):
    label_colors = np.array(
        [
            (0, 0, 0),  # 0=background
            # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
            (128, 0, 0),
            (0, 128, 0),
            (128, 128, 0),
            (0, 0, 128),
            (128, 0, 128),
            # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
            (0, 128, 128),
            (128, 128, 128),
            (64, 0, 0),
            (192, 0, 0),
            (64, 128, 0),
            # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
            (192, 128, 0),
            (64, 0, 128),
            (192, 0, 128),
            (64, 128, 128),
            (192, 128, 128),
            # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
            (0, 64, 0),
            (128, 64, 0),
            (0, 192, 0),
            (128, 192, 0),
            (0, 64, 128),
        ]
    )
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for class_idx in range(0, nc):
        idx = image == class_idx
        r[idx] = label_colors[class_idx, 0]
        g[idx] = label_colors[class_idx, 1]
        b[idx] = label_colors[class_idx, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

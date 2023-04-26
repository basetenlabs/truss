import base64
import urllib
from io import BytesIO
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch  # noqa
import torchvision  # noqa
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

BASE64_PREAMBLE = "data:image/png;base64,"


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        # Load model here and assign to self._model.
        sam = sam_model_registry["vit_h"](
            checkpoint=str(self._data_dir / "sam_vit_h_4b8939.pth")
        )
        sam.to("cuda")
        self._model = SamAutomaticMaskGenerator(sam)

    def predict(self, model_input: Any) -> Any:
        def show_anns(anns):
            if len(anns) == 0:
                return
            sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []  # noqa
            color = []  # noqa
            for ann in sorted_anns:
                m = ann["segmentation"]
                img = np.ones((m.shape[0], m.shape[1], 3))
                color_mask = np.random.random((1, 3)).tolist()[0]
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack((img, m * 0.35)))

        input_image_url = model_input["image_url"]
        req = urllib.request.urlopen(input_image_url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        input_image_cv2 = cv2.imdecode(arr, -1)
        masks = self._model.generate(input_image_cv2)
        plt.figure(figsize=(20, 20))
        plt.imshow(input_image_cv2)
        show_anns(masks)
        plt.axis("off")
        buffered = BytesIO()
        plt.savefig(buffered, format="png")
        img_str = base64.b64encode(buffered.getvalue())
        return {"output": BASE64_PREAMBLE + str(img_str)[2:-1]}

import logging
import tempfile

import cv2
import numpy as np
import requests
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from model.utils import upload_file_to_s3
from PIL import Image
from realesrgan import RealESRGANer

logger = logging.getLogger(__name__)

bg_tile = 400  # Tile size for background sampler, 0 for no tile during testing
upscale = 2  # The final upsampling scale of the image
arch = "clean"  # The GFPGAN architecture. Option: clean | original
channel = 2  # Channel multiplier for large networks of StyleGAN2

aligned = False  # Input are aligned faces
only_center_face = False  # Only restore the center face
paste_back = True  # Paste the restored faces back to images

# The model path
GFPGAN_PATH = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
)
# The background upsampler
ESRGAN_PATH = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"

RESIZE_DEFAULT_MAX = 1400


class RestorationModel:
    def __init__(self, **kwargs) -> None:
        self._config = kwargs.get("config")
        self.s3_config = (
            {
                "aws_access_key_id": self._config["secrets"][
                    "gfpgan_aws_access_key_id"
                ],
                "aws_secret_access_key": self._config["secrets"][
                    "gfpgan_aws_secret_access_key"
                ],
                "aws_region": self._config["secrets"]["gfpgan_aws_region"],
            }
            if self._config
            else {}
        )
        self.s3_bucket = (
            self._config["secrets"]["gfpgan_aws_bucket"] if self._config else None
        )

    def load(self):
        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        self.bg_upsampler = RealESRGANer(
            scale=2,
            model_path=ESRGAN_PATH,
            model=self.model,
            tile=bg_tile,
            tile_pad=10,
            pre_pad=0,
            half=True,
        )
        self.restorer = GFPGANer(
            model_path=GFPGAN_PATH,
            upscale=upscale,
            arch=arch,
            channel_multiplier=channel,
            bg_upsampler=self.bg_upsampler,
        )

    def predict(self, inputs):
        restored_images = []
        for instance in inputs:
            try:
                image = load_img_from_url(instance["image_url"])
                (
                    input_img,
                    cropped_faces,
                    restored_faces,
                    restored_img,
                ) = self.restore_image(image)
                if restored_img is not None:
                    rgb_image = rotate_axis(restored_img)
                    url_for_image = upload_image(
                        rgb_image, self.s3_bucket, self.s3_config
                    )
                    restored_images.append(url_for_image)
                else:
                    logger.warning(f'Failed to restore image, {instance["image_url"]}')
            except Exception as e:
                logger.error(e)
        return restored_images

    def restore_image(self, input_img):
        cropped_faces, restored_faces, restored_img = self.restorer.enhance(
            input_img,
            has_aligned=aligned,
            only_center_face=only_center_face,
            paste_back=paste_back,
        )
        return input_img, cropped_faces, restored_faces, restored_img


def upload_image(image, bucket=None, aws_credentials=None):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png")
    image = Image.fromarray(image)
    image.save(temp_file.name, format="png")
    temp_file.seek(0)
    return upload_file_to_s3(
        temp_file.name, bucket=bucket, aws_credentials=aws_credentials
    )


def load_img_from_url(img_url: str):
    img_request = requests.get(img_url)
    img_request.raise_for_status()
    image_content = np.asarray(bytearray(img_request.content), dtype="uint8")
    image = cv2.imdecode(image_content, cv2.IMREAD_COLOR)
    scale = min(
        RESIZE_DEFAULT_MAX / image.shape[1], RESIZE_DEFAULT_MAX / image.shape[0]
    )
    if scale < 1:
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    return image


def rotate_axis(bgr_image):
    """The images from GFPGAN are BGR, need to convert to RGB"""
    return bgr_image[:, :, ::-1]

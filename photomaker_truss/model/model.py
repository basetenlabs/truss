import os

import torch
from photomaker import PhotoMakerStableDiffusionXLPipeline


class PhotoMakerModel:
    """
    The PhotoMakerModel handles downloading the PhotoMakerStableDiffusionXLPipeline,
    loading it into memory, and running predictions. It is optimized to determine
    GPU capabilities and set parameters accordingly for efficient operation.
    """

    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = (
            torch.float16
            if self.device == "cuda"
            and torch.cuda.get_device_properties(0).total_memory > 11e9
            else torch.float32
        )
        self.use_safetensors = True

    def load(self):
        """
        Initializes and loads the PhotoMakerStableDiffusionXLPipeline model with the appropriate
        settings for torch_dtype and use_safetensors based on the system's capabilities.
        Utilizes cached model weights for efficiency.
        """
        model_cache_path = "./.cache/photomaker_model"
        os.makedirs(model_cache_path, exist_ok=True)

        self.model = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            "TencentARC/PhotoMaker",
            cache_dir=model_cache_path,
            torch_dtype=self.torch_dtype,
            use_safetensors=self.use_safetensors,
        ).to(self.device)

        print("Model loaded successfully with cache support.")

    def predict(
        self,
        prompt,
        negative_prompt=None,
        num_images_per_prompt=1,
        num_inference_steps=50,
        start_merge_step=10,
    ):
        """
        Generates images based on the given prompts and configuration parameters.

        Args:
            prompt (str): The primary prompt for image generation.
            negative_prompt (str, optional): A prompt describing undesired features. Defaults to None.
            num_images_per_prompt (int, optional): The number of images to generate per prompt. Defaults to 1.
            num_inference_steps (int, optional): The number of inference steps for image generation. Defaults to 50.
            start_merge_step (int, optional): Specifies the step to start merging the images. Defaults to 10.

        Returns:
            List[Image]: A list of Pillow Image objects generated from the prompts.
        """
        generator = torch.Generator(device=self.device).manual_seed(42)

        images = self.model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            start_merge_step=start_merge_step,
            generator=generator,
        ).images

        return images

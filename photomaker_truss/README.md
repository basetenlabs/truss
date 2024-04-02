# PhotoMaker Truss README

## Introduction
Welcome to the PhotoMaker Truss documentation! The PhotoMaker truss is a cutting-edge AI-based framework designed to revolutionize the way we customize realistic human photos. With its powerful image-to-image transformation capabilities, it allows for high ID fidelity and text controllability, making it an indispensable tool for developers and researchers working on image-based AI applications.

## System Requirements
To use the PhotoMaker truss effectively, please ensure your system meets the following requirements:
- Python version 3.10
- PyTorch version 2.0.0 or higher
- CUDA and cuDNN installed for GPU support
- Minimum GPU memory of 11GB. For GPUs not supporting bfloat16, `torch.float16` is recommended to enhance performance.

## Installation
The following commands guide you through setting up your environment for the PhotoMaker truss. Note the inclusion of CUDA and cuDNN to leverage GPU capabilities:
```
conda create --name photomaker python=3.10
conda activate photomaker
pip install -U pip
pip install torch==2.0.1 torchvision==0.15.2 diffusers transformers numpy huggingface-hub safetensors omegaconf peft gradio>=4.0.0
pip install git+https://github.com/TencentARC/PhotoMaker.git
```

## Usage Example - Image-to-image Transformation
Below is a detailed example demonstrating how to simulate the model to perform image-to-image transformation effectively:
```python
from photomaker import PhotoMakerStableDiffusionXLPipeline
from huggingface_hub import hf_hub_download

# Download model weights
photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")

# Initialize the pipeline with GPU support
pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(photomaker_path, torch_dtype=torch.bfloat16, use_safetensors=True).to("cuda")

# Prepare your input image and prompt
input_id_images = # your code to load images
prompt = "A portrait of a person wearing sunglasses in the style of Van Gogh."

# Generate image
generated_images = pipe(prompt=prompt, input_id_images=input_id_images).images[0]
generated_images.save('transformed_image.png')
```
This example highlights the unique capability of PhotoMaker to transform input images based on descriptive text prompts, embodying the essence of image-based machine learning.

## Acknowledgments
We express our gratitude to Tencent ARC Lab, Nankai University, and all the contributors who have played a pivotal role in the development and enhancement of PhotoMaker. Your expertise and dedication have been fundamental to our progress.

## Disclaimer
PhotoMaker is intended to foster innovation and exploration within AI-driven image generation. Users are urged to utilize this tool ethically and in compliance with applicable regulations. The developers bear no responsibility for misuse or legal infractions.

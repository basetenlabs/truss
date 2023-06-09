# ControlNet

This is an example truss for a [ControlNet](https://github.com/lllyasviel/ControlNet) model trained on Stable Diffusion 1.5 using the `StableDiffusionControlNetPipeline` diffusers pipeline.

We've used the [`sd-controlnet-scribble`](https://huggingface.co/lllyasviel/sd-controlnet-scribble) model for this truss.

We encourage you to make trusses for any ControlNet model using this one as a baseline example.


# Deploy this ControlNet truss
To deploy this ControlNet truss example, simply run the following from the `examples/` directory:
```
import baseten
import truss
import os

controlnet_truss = truss.load("controlnet-scribble")

# Make sure you create a Baseten API key (follow the instructions here: https://docs.baseten.co/settings/api-keys)
baseten.login(os.environ["BASETEN_API_KEY"])

baseten.deploy(controlnet_truss, model_name="ControlNet Scribble", publish=True)
```
That's all you need to do to get this ControlNet model deployed!

_If you're looking to deploy a different ControlNet model, continue following along ⬇️_

## How to create a similar ControlNet truss
---
If you want to create a truss for other ControlNet model checkpoints, look no further.

Run the following to initialize an empty truss:
```
import truss
truss.init("my-controlnet-truss")
```
Inside the `my-controlnet-truss/model/model.py` file, replace the `load()` and `predict()` methods with the appropriate model code taken from one of the model checkpoint examples. For example, we used [the example code here](https://huggingface.co/lllyasviel/sd-controlnet-scribble#example) for the `controlnet-scribble` truss.

You can iteratively test your truss to make sure its configured properly. Modify the following to test your truss:
```
import truss
import urllib
from shared.base64_utils import b64_to_pil, pil_to_b64
from PIL import Image

# Load your "my-controlnet-truss" here
controlnet_truss = truss.load("controlnet-scribble")

# Use the example input provided in your ControlNet model checkpoint's example
image = Image.open(urllib.request.urlopen("https://huggingface.co/lllyasviel/sd-controlnet-scribble/resolve/main/images/bag.png"))
request = {
    "prompt": "bag",
    "image": pil_to_b64(image)
}
response = controlnet_truss.docker_predict(request)

# The outputted image will be saved as "output.png"
b64_to_pil(response["images"][0]).save("output.png")
```
Once you're happy with the prediction behavior here, follow the steps above to deploy your truss!

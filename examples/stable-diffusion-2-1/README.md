# Stable Diffusion 2.1

This is an example deploying stable diffusion with truss weights preloaded

## Prepare
The weights are ignored in this repository. You need to install them before reunning the model.

To do, the run the following code from this directory

```
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

_model_id = "stabilityai/stable-diffusion-2-1-base"
_scheduler = EulerDiscreteScheduler.from_pretrained(_model_id, subfolder="scheduler")
_model = StableDiffusionPipeline.from_pretrained(
    _model_id, scheduler=_scheduler, torch_dtype=torch.float16
)
_model.save_pretrained("./data/")
```

## Deploy to Baseten
To deploy the model, run the following from the root of the directory

```
import baseten, truss. os

sd = truss.load("./examples/stable-diffusion-2-1")

baseten.login(os.environ["BASETEN_API_KEY"])

baseten.deploy(sd, model_name"My SD 2.1", publish=True)
```

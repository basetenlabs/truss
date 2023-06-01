# Manually

You can package any model as a Truss. `truss.create()` is a convenient shortcut for packaging in-memory models built in supported frameworks, but the manual approach gives control and flexibility throughout the entire model packaging and deployment process.

This doc walks through the process of manually creating a Truss, using Stable Diffusion v1.5 as an example.

To get started, initialize the Truss with the following command in the CLI:

```
truss init sd_truss
```

This will create the following file structure:

```
sd_truss/       # Truss root
  data/         # Stores serialized models/weights/binaries
  model/        #
    __init__.py #
    model.py    # Implements Model class
  packages/     # Stores utility code for model.py
  config.yaml   # Config for model serving environment
  examples.yaml # Invocation examples
```

Most of our development work will happen in `models/model.py` and `config.yaml`.

### Loading your model

In `models/model.py`, the first function you'll need to implement is `load()`.

When the model is spun up to receive requests, `load()` is called exactly once and is guaranteed to finish before any predictions are attempted.

The purpose of `load()` is to set a value for `self._model`. This requires deserializing your model or otherwise loading in your model weights.

**Example: Stable Diffusion 1.5**

The exact code you'll need will depend on your model and framework. In this example, model weights for Stable Diffusion 1.5 are coming from the HuggingFace `diffusors` package.

This requires a couple of imports (don't worry, we'll cover adding Python requirements in a bit).

```python
from dataclasses import asdict
from typing import Dict

import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
```

The load function looks like:

```python
def load(self):
    scheduler = EulerDiscreteScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler",
    )
    self._model = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        scheduler=scheduler,
        torch_dtype=torch.float16,
    )
    self._model.unet.set_use_memory_efficient_attention_xformers(True)
    self._model = self._model.to("cuda")
```

`self._model` could be set using weights from anywhere. If you have custom weights, you can load them from your Truss' `data/` directory by [following this guide](https://github.com/basetenlabs/truss/blob/main/examples/stable-diffusion-1-5/data/README.md
).


### Implement model invocation

The other key function in your Truss is `predict()`, which handles model invocation.

As our loaded model is a `StableDiffusionPipeline` object, model invocation is pretty simple:

```python
def predict(self, model_input: Dict):
    response = self._model(**model_input)
    return response
```

All we have to do is pass the model input to the model.

But how do we make sure the model input is a valid format, and that the model output is usable?

### Implement processing functions

By default, pre- and post-processing functions are passthroughs. But if needed, you can implement these functions to make your model input and output match the specification of whatever app or API you're building.

There are [more in-depth docs on processing functions here](../develop/processing.md), but here's sample code for the Stable Diffusion example, which needs a postprocessing function but not a pre-processing function:

```python
def postprocess(self, model_output: Dict) -> Dict:
    # Convert to base64
    model_output["images"] = [pil_to_b64(img) for img in model_output["images"]]
    return asdict(model_output)
```

Eagle-eyed readers will note that `pil_to_b64()` is not a function that has been defined anywhere. How can we use it?

### Call upon shared packages

Here's that `pil_to_b64()` function from the last step:

```python
import base64
from io import BytesIO

from PIL import Image

def pil_to_b64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return "data:image/png;base64," + str(img_str)[2:-1]
```

You could just paste this into `models/model.py` and call it a day. But its better to factor out helper functions and utilities so that they can be re-used between multiple Trusses.

Let's create a folder `shared` at the same level as our root `sd_truss` directory (don't create it inside the Truss directory). Then create a file `shared/base64_utils.py`. It should look like this:

```
shared/
  base64_utils.py
sd_truss/
  ...
```

Paste the code from above into `shared/base64_utils.py`.

Let your Truss know where to look for external packages with the following lines in `config.yaml`:

```yaml
external_package_dirs:
- ../shared/
```

Note that this is an array in yaml; your Truss can depend on multiple external directories for packages.

Finally, at the top of `sd_truss/models/model.py`, add:

```python
from base64_utils import pil_to_b64
```

This will import your function from your external directory.

For more details on bundled and shared packages, see [this demo repository](https://github.com/bolasim/truss-packages-example) and the [bundled packages docs](../develop/bundled-packages.md).

### Set Python and system requirements

Now, we switch our attention to `config.yaml`. You can use this file to customize a great deal about your packaged model — [here's a complete reference](../develop/configuration.md) — but right now we just care about setting our Python requirements up so the model can run.

For that, find `requirements:` in the config file. In the Stable Diffusion 1.5 example, we set it to:

```yaml
requirements:
- diffusers
- transformers
- accelerate
- scipy
- safetensors
- xformers
- triton
```

These requirements work just like `requirements.txt` in a Python project, and you can pin versions with `package==1.2.3`.

### Set hardware requirements

Large models like Stable Diffusion require powerful hardware to run invocations. Set your packaged model's hardware requirements in `config.yaml`:

```yaml
resources:
  accelerator: A10G # Type of GPU required
  cpu: "8" # Number of vCPU cores required
  memory: 30Gi # Mibibytes (Mi) or Gibibytes (Gi) of RAM required
  use_gpu: true # If false, set accelerator: null
```

You've successfully packaged a model! If you have the required hardware, you can [test it locally](../develop/localhost.md), or [deploy it to Baseten](https://docs.baseten.co/deploying-models/deploy#stage-1-deploying-a-draft) to get a draft model for rapid iteration in a production environment.

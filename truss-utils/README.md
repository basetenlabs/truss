# truss-utils

This package is a repository of common functions that help with developing
production AI/ML models with Truss.

## Use

In the `model.py` of your truss, you can do something like the following:

```
from truss_utils.image import pil_to_64

class Model:
...

    def predict(self, model_input):
        # call Stable diffusion
        ...

        return pil_to_b64(image)
```
# Truss directory structure

A Truss is a directory containing the packaged model. This reference details the files and folders in said directory and their contents.

We have a number of [example models](https://github.com/basetenlabs/truss/tree/main/examples) to reference to see the Truss structure in action. The [iris model example](https://github.com/basetenlabs/truss/tree/main/examples/iris) demonstrates most of Truss' structure.

Truss' directory structure relies on a few conventions, but beyond those you may include any additional files or folders you like without interfering with the packaged model. Specifically, a Truss must have the following files:

```
config.yaml
examples.yaml
model/
    __init__.py
    model.py
```

The model.py file must implement the `load` and `predict` functions [detailed below](#modelpy).

And the following folders are optional but are part of the directory structure convention:

```
data/
    <data>
    <serialized model>
packages/
    <modules>
```

Here's a file-by-file breakdown.

### config.yaml

This file specifies the [configuration options](../develop/configuration.md) to be applied to the Truss.

### examples.yaml

This file provides [sample inputs](../develop/examples.md) for running your model.

## model/

This folder contains the packaged model code.

### __init__.py

This file exists for Python packaging purposes and may remain empty.

### model.py

This file contains the code that deserializes and runs the model, as well as the [pre- and post-processing functions](../develop/processing.md).

Here's an example `model/model.py` file:

```python
from tempfile import NamedTemporaryFile
from typing import Dict

import requests
import torch
import whisper


class Model:
    def __init__(self, **kwargs) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None

    def load(self):
        self._model = whisper.load_model("small", self.device)
        return

    def preprocess(self, request: Dict) -> Dict:
        resp = requests.get(request["url"])
        return {"response": resp.content}

    def postprocess(self, request: Dict) -> Dict:
        return request

    def predict(self, request: Dict) -> Dict:
        with NamedTemporaryFile() as fp:
            fp.write(request["response"])
            result = whisper.transcribe(
                self._model,
                fp.name,
                temperature=0,
                best_of=5,
                beam_size=5,
            )
            segments = [
                {"start": r["start"], "end": r["end"], "text": r["text"]}
                for r in result["segments"]
            ]
        return {
            "language": whisper.tokenizer.LANGUAGES[result["language"]],
            "segments": segments,
            "text": result["text"],
        }
```

Here is a breakdown of the functions in `model/model.py`:

#### __init__()

A model class is instantiated by a Truss model serving environment with the following parameters:

1. `config`: Provides access to the same config that's bundled with a truss, as a dictionary.
2. `data_dir`: Provides a pathlib directory where all the data bundled with the truss is provided.
3. `secrets`: This dictionary like object provides access to the secrets declared in the truss, but bound at runtime. The values returned by the secrets dictionary are dynamic, the secret value returned for the same key may be different over time, e.g. if it's updated. This means that when you update the secret values, and for many secrets it's a good practice to update them periodically, you don't have to redeploy the model.

This constructor can declare any subset of above parameters that it needs, they're bound by name as needed and the rest are omitted. One can omit all parameters or even omit the constructor.

#### load()

The model class can declare a load method. This method is guaranteed to be
invoked before any prediction calls are made. This is a good place for
downloading any data needed by the model. One can do this in the constructor as
well, but it's not ideal to block the constructor for a long time as it might
affect initialization of other components. So load is where you'd want to do any
expensive i/o operations.

If omitted this method is considered to be no-op.

#### predict()

Perhaps the most critical method, this is the method called for making
predictions. This method of the model call is passed input and the returned
output is the model's prediction for that input.
#### preprocess()

This method allows preprocessing input to the model. Model input is passed to
this method and the output becomes input to the predict method below.

If omitted, this method is assumed to be identity.

#### postprocess()

This method provides a way to modify the model output before returning. Output of the predict method is input to this method and the output of this method is what's returned to the caller.

If omitted, this method is assumed to be identity.

## data/

This optional folder has the most varied contents, and enumerating everything that could go in here is beyond the scope of these docs. The most likely thing to find in here is a serialized model, but this folder can contain any dependencies for serving the model like data sets, weights, parameters, or any associated exports with a serialized model.

## packages/

This optional folder is used to hold your own Python modules referenced in the model code.

{% hint style="info" %}
When you import your Truss, the import mechanism adds everything in the Truss' root directory and packages directory to the path.
{% endhint %}

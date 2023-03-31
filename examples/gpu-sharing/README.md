# GPU Sharing Example

This is an example of a truss that shares GPU across multiple models. This truss serves
three models: a text to image model, an image to image model and an in-painting mode.
Specific model can be targeted by providing `model_sub_name` in the request dictionary. Please
refers to examples.yaml for examples.
A model is loaded into GPU memory when needed. When a different model is invoked, then the previous
one is offloaded and replaced with the new one. All models remain in the regular memory though, so that
moving them to GPU memory later is faster.

## Usage
Like any other truss, you can load this truss and test in-memory then deploy it to a cloud provider as follows:
```
models = truss.load("./") # This loads all of the models
models.predict({
    "model_sub_name": "text_img", # Name of the model to sub into GPU
    "prompt": "red dog", # The rest of the kwargs are passed as input to that model
})

import baseten
baseten.deploy(models, model_name="Combo GPU Model", publish=True)
```

To use the included examples, please run `git lfs pull` before loading them via `models.examples()`.

## Adding another model
To add another model, do the following:
1. Create a new file with your model class under the `model/` directory. We recommend that you copy [an existing model](./model/text_img_model.py) and modify it to suit your needs.
2. Add the model to the registry in [`model/model.py`](./model/model.py#L16). The key you use here is what needs to be passed in as `model_sub_name` to use the models
3. Update `config.yaml` to include any additional system or python requirements
4. Follow the usage section for testing and deploying

## Custom Weights
All the examples in this section load the weights during the load at runtime. You may desire to load weights into the model for a variety of reasons. To do so, do the following:
1. Create a `data/` directory parallel to `model/`
2. Add the weights there for any of the models. You can make different sub-directories or organize as you wish.
3. In the model file for a single model, use the following line to get a reference to this folder
   ```
   self._data_dir = kwargs["data_dir"]
    ```
4. In the `load` function, you this variable to load the weights. Here is an example from the diffusers library.
   ```
   self._model = StableDiffusionPipeline.from_pretrained(str(self._data_dir / "path" / "to" / "weights")).to("cuda")
   ```

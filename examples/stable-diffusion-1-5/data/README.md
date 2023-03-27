# Adding *custom weights* to Stable Diffusion
If you have custom weights for Stable Diffusion and want to build a truss for it, its as simple as doing the following!

## Add weights to the data directory
In your truss, add your weights to the `data` directory, and make sure its specified in your `config.yaml` file here:
```
...
data_dir: data
...
```
Make sure you have the appropriate subfolders, your `data` directory structure should look something like this:
```
your_truss/
    ...
    data/
        feature_extractor/
        safety_checker/
        scheduler/
        text_encoder/
        tokenizer/
        unet/
        vae/
        model_index.json
    ...
```

## Reference the data directory in the `models.py` file
We can [utilize the functionality](https://huggingface.co/docs/diffusers/main/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.from_pretrained) to pass in a path to the `from_pretrained` method off of the `StableDiffusionPipeline`.

You'll need to reference this data directory in your `model.py` file, in this case `_data_dir` is the name of your `data` directory:

```
class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._model = None

    def load(self):
        self._model = StableDiffusionPipeline.from_pretrained(
            self._data_dir,
            torch_dtype=torch.float16,
        )
        self._model = self._model.to("cuda")

    ...
```

And this is all you need to load your custom weights and create a truss for your custom Stable Diffusion model!

## Truss in the process...
[Follow the documentation here](https://truss.baseten.co/develop/localhost) to build and run your truss!

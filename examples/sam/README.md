# Segment Anything Model

This is an example deploying Segment Anything Model (SAM) with truss weights preloaded

## Prepare
The weights are ignored in this repository. You need to install them before reunning the model.

To do this, follow the [Model Checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints) section to download the `vit_h` model to the `data` directory.

## Deploy to Baseten
To deploy the model, run the following from the root of the directory

```
import baseten, truss

sam = truss.load("./examples/sam")

baseten.login(os.environ["BASETEN_API_KEY"])

baseten.deploy(sam, model_name"My SAM", publish=True)
```

## Predict
Example prediction:
```
import baseten

model = baseten.deployed_model_id("<MODEL_ID>")

resp = model.predict({"image_url": "<URL TO IMAGE>"})

# Base64 string representation of the output image in resp["output"]
```

# Temporal Fusion Transformer

This is an example truss for the Temporal Fusion Transformer model using [this  `pytorch-forecasting` example](https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html)

## Prepare
The checkpoint used to load this model is ignored in this repository. You need to add your own checkpoint file to the `/data` directory and modify the name  in the `load()` function in `/model/model.py` below to be your own checkpoint's before rerunning the model.
```
def load(self):
    ...
    checkpoint_path = str(self._data_dir / "checkpoint.ckpt")
    ...
```

## Predict

To run a prediction on the truss, run the following.

`new_prediction_data` is taken from the [pytorch-forecasting example here](https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html#Predict-on-new-data)
```
import truss

tft = truss.load('./examples/temporal-fusion-transformer')
predictions = tft.docker_predict({"data": new_prediction_data.to_json(orient="records")})
```

## Deploy to Baseten
To deploy the model, run the following from the root of the directory

```
import baseten, truss, os

tft = truss.load("./examples/temporal-fusion-transformer")

baseten.login(os.environ["BASETEN_API_KEY"])

baseten.deploy(tft, model_name="Temporal Fusion Transformer - Stallion", publish=True)
```

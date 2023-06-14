from json import loads
from typing import Any

import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        # Load model here and assign to self._model.
        checkpoint_path = str(self._data_dir / "checkpoint.ckpt")
        self._model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)

    def predict(self, model_input: Any) -> Any:
        # Invoke model on model_input and calculate predictions here.
        prediction_data = pd.DataFrame.from_dict(loads(model_input.pop("data")))
        predictions = self._model.predict(prediction_data, mode="prediction")
        return predictions.numpy()

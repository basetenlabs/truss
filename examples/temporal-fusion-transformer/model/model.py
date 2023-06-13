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
        best_model_path = str(self._data_dir / "checkpoint.ckpt")
        self._model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    def predict(self, model_input: Any) -> Any:
        # Invoke model on model_input and calculate predictions here.
        prediction_data = pd.DataFrame.from_dict(loads(model_input.pop("data")))
        raw_predictions = self._model.predict(
            prediction_data, mode="raw", return_x=True
        )
        return raw_predictions.output.prediction.numpy()

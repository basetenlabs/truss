from typing import Dict, List

import xgboost as xgb

MODEL_BASENAME = 'model'
MODEL_EXTENSIONS = ['.ubj']


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs['data_dir']
        config = kwargs['config']
        model_metadata = config['model_metadata']
        # XGBoost models, saved and loaded via the native save/load
        # in XGBoost do not support predicting probabilities unless
        # they are a multi-class classification problem.
        # TODO: Integrate model_metadata field to determine model
        # objective function to determine if an XGBoost model
        # supports predicting probabilities.
        self._supports_predict_proba = False
        self._model_binary_dir = model_metadata['model_binary_dir']
        self._model = None

    def load(self):
        model_binary_dir_path = self._data_dir / self._model_binary_dir
        paths = [(model_binary_dir_path / MODEL_BASENAME).with_suffix(model_extension)
                 for model_extension in MODEL_EXTENSIONS]
        model_file_path = next(path for path in paths if path.exists())
        self._model = xgb.Booster()
        self._model.load_model(model_file_path)

    def preprocess(self, request: Dict) -> Dict:
        """
        Incorporate pre-processing required by the model if desired here.

        These might be feature transformations that are tightly coupled to the model.
        """
        return request

    def postprocess(self, request: Dict) -> Dict:
        """
        Incorporate post-processing required by the model if desired here.
        """
        return request

    def predict(self, request: Dict) -> Dict[str, List]:
        response = {}
        inputs = request['inputs']
        dmatrix_inputs = xgb.DMatrix(inputs)
        result = self._model.predict(dmatrix_inputs)
        response['predictions'] = result
        return response

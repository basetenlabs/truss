from pathlib import Path
from typing import Dict

from truss.model_framework import ModelFramework
from truss.model_inference import infer_xgboost_packages
from truss.types import ModelFrameworkType

MODEL_FILENAME = 'model.ubj'


class XGBoost(ModelFramework):

    def typ(self) -> ModelFrameworkType:
        return ModelFrameworkType.XGBOOST

    def infer_requirements(self) -> Dict[str, str]:
        return infer_xgboost_packages()

    def serialize_model_to_directory(self, model, target_directory: Path):
        model_filename = MODEL_FILENAME
        model_filepath = target_directory / model_filename
        model.save_model(model_filepath)

    def model_metadata(self, model) -> Dict[str, str]:
        return {
            'model_binary_dir': 'model',
            'supports_predict_proba': False,
        }

    def supports_model_class(self, model_class) -> bool:
        model_framework, _, _ = model_class.__module__.partition('.')
        return model_framework == ModelFrameworkType.XGBOOST.value

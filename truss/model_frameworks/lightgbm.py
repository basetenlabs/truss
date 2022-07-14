from pathlib import Path
from typing import Dict

from truss.model_framework import ModelFramework
from truss.model_inference import infer_lightgbm_packages
from truss.templates.server.common.util import model_supports_predict_proba
from truss.types import ModelFrameworkType

MODEL_FILENAME = 'model.joblib'


class LightGBM(ModelFramework):

    def typ(self) -> ModelFrameworkType:
        return ModelFrameworkType.LIGHTGBM

    def infer_requirements(self) -> Dict[str, str]:
        return infer_lightgbm_packages()

    def serialize_model_to_directory(self, model, target_directory: Path):
        import joblib
        model_filename = MODEL_FILENAME
        model_filepath = target_directory / model_filename
        joblib.dump(model, model_filepath, compress=True)

    def model_metadata(self, model) -> Dict[str, str]:
        supports_predict_proba = model_supports_predict_proba(model)
        return {
            'model_binary_dir': 'model',
            'supports_predict_proba': supports_predict_proba,
        }

    def supports_model_class(self, model_class) -> bool:
        model_framework, _, _ = model_class.__module__.partition('.')
        return model_framework == ModelFrameworkType.LIGHTGBM.value

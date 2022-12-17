from pathlib import Path
from typing import Dict, Set

from truss.constants import SKLEARN_REQ_MODULE_NAMES
from truss.model_framework import ModelFramework
from truss.templates.server.common.util import model_supports_predict_proba
from truss.types import ModelFrameworkType

MODEL_FILENAME = "model.joblib"


class Custom(ModelFramework):
    def typ(self) -> ModelFrameworkType:
        return ModelFrameworkType.CUSTOM

    def required_python_depedencies(self) -> Set[str]:
        return {}

    def serialize_model_to_directory(self, model, target_directory: Path):
        raise NotImplementedError()

    def model_metadata(self, model) -> Dict[str, str]:
        return {}

    def supports_model_class(self, model_class) -> bool:
        return False

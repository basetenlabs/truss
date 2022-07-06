from pathlib import Path
from typing import Dict

from truss.model_framework import ModelFramework
from truss.model_inference import infer_keras_packages
from truss.types import ModelFrameworkType


class Keras(ModelFramework):

    def typ(self) -> ModelFrameworkType:
        return ModelFrameworkType.KERAS

    def infer_requirements(self) -> Dict[str, str]:
        return infer_keras_packages()

    def serialize_model_to_directory(self, model, target_directory: Path):
        model.save(target_directory)

    def model_metadata(self, model) -> Dict[str, str]:
        return {
            'model_binary_dir': 'model',
        }

    def supports_model_class(self, model_class) -> bool:
        model_framework, _, _ = model_class.__module__.partition('.')
        return model_framework in [ModelFrameworkType.KERAS.value, ModelFrameworkType.TENSORFLOW.value]

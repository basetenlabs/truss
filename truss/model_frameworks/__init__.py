from typing import Any

from truss.errors import FrameworkNotSupportedError
from truss.model_framework import ModelFramework
from truss.model_frameworks.huggingface_transformer import HuggingfaceTransformer
from truss.model_frameworks.keras import Keras
from truss.model_frameworks.lightgbm import LightGBM
from truss.model_frameworks.mlflow import Mlflow
from truss.model_frameworks.pytorch import PyTorch
from truss.model_frameworks.sklearn import SKLearn
from truss.model_frameworks.xgboost import XGBoost
from truss.types import ModelFrameworkType

MODEL_FRAMEWORKS_BY_TYPE = {
    ModelFrameworkType.SKLEARN: SKLearn(),
    ModelFrameworkType.KERAS: Keras(),
    ModelFrameworkType.HUGGINGFACE_TRANSFORMER: HuggingfaceTransformer(),
    ModelFrameworkType.PYTORCH: PyTorch(),
    ModelFrameworkType.XGBOOST: XGBoost(),
    ModelFrameworkType.LIGHTGBM: LightGBM(),
    ModelFrameworkType.MLFLOW: Mlflow(),
}


SUPPORTED_MODEL_FRAMEWORKS = [
    ModelFrameworkType.SKLEARN,
    ModelFrameworkType.KERAS,
    ModelFrameworkType.TENSORFLOW,
    ModelFrameworkType.HUGGINGFACE_TRANSFORMER,
    ModelFrameworkType.XGBOOST,
    ModelFrameworkType.PYTORCH,
    ModelFrameworkType.LIGHTGBM,
]


def model_framework_from_model(model: Any) -> ModelFramework:
    return model_framework_from_model_class(model.__class__)


def model_framework_from_model_class(model_class) -> ModelFramework:
    for model_framework in MODEL_FRAMEWORKS_BY_TYPE.values():
        if model_framework.supports_model_class(model_class):
            return model_framework

    raise FrameworkNotSupportedError(
        "Model must be one of "
        + "/".join([t.value for t in SUPPORTED_MODEL_FRAMEWORKS])
    )

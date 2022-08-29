import logging
from pathlib import Path
from typing import Dict, Set

from truss.constants import HUGGINGFACE_TRANSFORMER_MODULE_NAME
from truss.model_framework import ModelFramework
from truss.types import ModelFrameworkType

logger = logging.getLogger(__name__)


class HuggingfaceTransformer(ModelFramework):
    def typ(self) -> ModelFrameworkType:
        return ModelFrameworkType.HUGGINGFACE_TRANSFORMER

    def required_python_depedencies(self) -> Set[str]:
        return HUGGINGFACE_TRANSFORMER_MODULE_NAME

    def serialize_model_to_directory(self, model, target_directory: Path):
        # For Huggingface models, all the important details are in metadata.
        pass

    def model_metadata(self, model) -> Dict[str, str]:
        hf_task = self.model_type(model)
        return {
            "transformer_config": {
                "model": _hf_model_name(model),
            },
            "has_hybrid_args": hf_task in {"text-generation"},
            "has_named_args": hf_task in {"zero-shot-classification"},
        }

    def model_type(self, model) -> str:
        return _infer_hf_task(model)

    def supports_model_class(self, model_class) -> bool:
        model_framework, _, _ = model_class.__module__.partition(".")
        return model_framework == "transformers"


def _hf_model_name(model) -> str:
    try:
        return model.model.config._name_or_path
    except AttributeError:
        logger.info(
            "Unable to infer a HuggingFace model on this task pipeline; transformer library will default"
        )


def _infer_hf_task(model) -> str:
    try:
        return model.task
    except AttributeError as error:
        logger.exception(
            "Unable to find a HuggingFace task on this object, did you call with a `Pipeline` object"
        )
        raise error

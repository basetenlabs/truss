import requests

from truss.base.constants import (
    HF_ACCESS_TOKEN_KEY,
    HF_MODELS_API_URL,
    TRTLLM_MIN_MEMORY_REQUEST_GI,
)
from truss.base.trt_llm_config import CheckpointSource
from truss.truss_handle.truss_handle import TrussHandle


def is_missing_secrets_for_trt_llm_builder(tr: TrussHandle) -> bool:
    for trt_llm_build_config in tr.spec.config.parsed_trt_llm_build_configs:
        source = trt_llm_build_config.checkpoint_repository.source
        hf_model_id = trt_llm_build_config.checkpoint_repository.repo
        if (
            source == CheckpointSource.HF
            and HF_ACCESS_TOKEN_KEY not in tr.spec.secrets
            and not _is_model_public(hf_model_id)
        ):
            return True
    return False


def memory_updated_for_trt_llm_builder(tr: TrussHandle) -> bool:
    if uses_trt_llm_builder(tr):
        if tr.spec.memory_in_bytes < TRTLLM_MIN_MEMORY_REQUEST_GI * 1024**3:
            tr.spec.config.resources.memory = f"{TRTLLM_MIN_MEMORY_REQUEST_GI}Gi"
            tr.spec.config.write_to_yaml_file(tr.spec.config_path, verbose=False)
            return True
    return False


def _is_model_public(model_id: str) -> bool:
    """Validate that a huggingface hub model is public.

    The hf hub API will return 401 when trying to access a private or gated model without auth.
    """
    response = requests.get(f"{HF_MODELS_API_URL}/{model_id}")
    return response.status_code == 200


def uses_trt_llm_builder(tr: TrussHandle) -> bool:
    return tr.spec.config.trt_llm is not None

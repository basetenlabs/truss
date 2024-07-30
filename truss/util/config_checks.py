import requests
from truss.config.trt_llm import CheckpointSource
from truss.constants import (
    HF_ACCESS_TOKEN_KEY,
    HF_MODELS_API_URL,
    TRTLLM_MIN_MEMORY_REQUEST_GI,
)
from truss.truss_handle import TrussHandle


def uses_trt_llm_builder(tr: TrussHandle) -> bool:
    return (
        tr.spec.config.trt_llm is not None and tr.spec.config.trt_llm.build is not None
    )


def check_secrets_for_trt_llm_builder(tr: TrussHandle) -> bool:
    if tr.spec.config.trt_llm and tr.spec.config.trt_llm.build:
        source = tr.spec.config.trt_llm.build.checkpoint_repository.source
        hf_model_id = tr.spec.config.trt_llm.build.checkpoint_repository.repo
        if (
            source == CheckpointSource.HF
            and HF_ACCESS_TOKEN_KEY not in tr.spec.secrets
            and not _is_model_public(hf_model_id)
        ):
            return False
    return True


def check_and_update_memory_for_trt_llm_builder(tr: TrussHandle) -> bool:
    if tr.spec.config.trt_llm and tr.spec.config.trt_llm.build:
        if tr.spec.memory_in_bytes < TRTLLM_MIN_MEMORY_REQUEST_GI * 1024**3:
            tr.spec.config.resources.memory = f"{TRTLLM_MIN_MEMORY_REQUEST_GI}Gi"
            tr.spec.config.write_to_yaml_file(tr.spec.config_path, verbose=False)
            return False
    return True


def _is_model_public(model_id: str) -> bool:
    """Validate that a huggingface hub model is public.

    The hf hub API will return 401 when trying to access a private or gated model without auth.
    """
    response = requests.get(f"{HF_MODELS_API_URL}/{model_id}")
    return response.status_code == 200

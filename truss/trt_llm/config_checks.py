import requests

from truss.base.constants import (
    HF_ACCESS_TOKEN_KEY,
    HF_MODELS_API_URL,
    OPENAI_COMPATIBLE_TAG,
    OPENAI_NON_COMPATIBLE_TAG,
    TRTLLM_MIN_MEMORY_REQUEST_GI,
)
from truss.base.trt_llm_config import CheckpointSource, TrussTRTLLMModel
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


def has_no_tags_trt_llm_builder(tr: TrussHandle) -> str:
    """
    support transitioning to more openai-compatible schema.
    # transitioning in three phases:
    # 1. set OPENAI_NON_COMPATIBLE_TAG as default
    # 2. set OPENAI_COMPATIBLE_TAG as default for new pushes. (June 2025)
    # 3. Reject models without tag (July 2025)
    # 4. disable new legacy-non-openai pushes server-side. (2026 eta)
    """
    if uses_trt_llm_builder(tr):
        assert tr.spec.config.trt_llm is not None
        current_tags = tr.spec.config.model_metadata.get("tags", [])
        if (
            OPENAI_COMPATIBLE_TAG in current_tags
            and OPENAI_NON_COMPATIBLE_TAG in current_tags
        ):
            raise ValueError(
                f"TRT-LLM models should have either model_metadata['tags'] = ['{OPENAI_COMPATIBLE_TAG}'] or ['{OPENAI_NON_COMPATIBLE_TAG}']. "
                f"Your current tags are both {current_tags}."
            )
        elif (
            OPENAI_NON_COMPATIBLE_TAG in current_tags
            or OPENAI_COMPATIBLE_TAG not in current_tags
        ) and tr.spec.config.trt_llm.build.speculator is not None:
            # spec-dec has no classic backend. OpenAI-mode is forced, regardless of tags.
            message = f"""TRT-LLM models with speculator should have model_metadata/tags section with ['openai-compatible'] tag.
Adding:
```yaml
model_metadata:
tags:
- {OPENAI_COMPATIBLE_TAG}
```
            """
            tr.spec.config.model_metadata["tags"] = [
                OPENAI_COMPATIBLE_TAG
            ] + tr.spec.config.model_metadata.get("tags", [])
            return message
        elif (
            tr.spec.config.trt_llm.build.base_model != TrussTRTLLMModel.ENCODER
            and not current_tags
            or not any(
                tag in current_tags
                for tag in (OPENAI_COMPATIBLE_TAG, OPENAI_NON_COMPATIBLE_TAG)
            )
        ):
            # inserting new tag server-side (Briton) and client side on truss push
            # transitioning in three phases:
            message = f"""
TRT-LLM models should have model_meta_data/tags section with either ['{OPENAI_COMPATIBLE_TAG}'] or ['{OPENAI_NON_COMPATIBLE_TAG}'].
Your current tags are `{current_tags}`.
As temporary measure, we are injecting the `tags: - {OPENAI_NON_COMPATIBLE_TAG}` to your config.yaml file.
Please migrate to the openai compatible schema as soon as possible, this behavior will be deprecated in the future.
```yaml
model_metadata:
tags:
- {OPENAI_COMPATIBLE_TAG}
```
"""
            tr.spec.config.model_metadata["tags"] = [
                OPENAI_NON_COMPATIBLE_TAG
            ] + tr.spec.config.model_metadata.get("tags", [])
            tr.spec.config.write_to_yaml_file(tr.spec.config_path, verbose=False)
            return message
    return ""


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
